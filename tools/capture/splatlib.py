"""Shared helpers: read/write 3DGS PLY files, the scene's up-aligned frame, and rendering with gsplat."""

import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization

NP = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 12}


def read_ply(path):
    with open(path, "rb") as f:
        names, n = [], 0
        while (line := f.readline().decode().strip()) != "end_header":
            t = line.split()
            if t[:2] == ["element", "vertex"]:
                n = int(t[2])
            elif t[0] == "property":
                names.append(t[2])
        data = np.frombuffer(f.read(), np.float32, count=n * len(names)).reshape(n, len(names))
    return names, data.copy()


def write_ply(path, names, data):
    with open(path, "wb") as f:
        f.write(f"ply\nformat binary_little_endian 1.0\nelement vertex {len(data)}\n".encode())
        f.write("".join(f"property float {n}\n" for n in names).encode())
        f.write(b"end_header\n")
        f.write(np.ascontiguousarray(data, np.float32).tobytes())


def qmat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])  # fmt: skip


def read_poses(sparse):
    """name -> (world-to-camera 4x4, (fx, fy, cx, cy), (w, h)) from COLMAP binaries."""
    cams = {}
    with open(Path(sparse) / "cameras.bin", "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            cid, model, w, h = struct.unpack("<iiQQ", f.read(24))
            p = struct.unpack(f"<{NP[model]}d", f.read(8 * NP[model]))
            cams[cid] = ((p[0], p[0], p[1], p[2]) if model in (0, 2, 3) else p[:4], (w, h))
    out = {}
    with open(Path(sparse) / "images.bin", "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            _, qw, qx, qy, qz, tx, ty, tz, cid = struct.unpack("<i7di", f.read(64))
            name = b""
            while (c := f.read(1)) != b"\0":
                name += c
            (k,) = struct.unpack("<Q", f.read(8))
            f.seek(24 * k, 1)
            w2c = np.eye(4)
            w2c[:3, :3], w2c[:3, 3] = qmat((qw, qx, qy, qz)), (tx, ty, tz)
            out[name.decode()] = (w2c, *cams[cid])
    return out


def up_frame(poses):
    """A world->aligned rotation with +z up (from the cameras' mean down vector) and the cameras' mean height."""
    # cameras are held level (little roll), so their right axes lie in the horizontal plane: up is the
    # direction most perpendicular to all of them (pitching the camera does not bias it)
    rights = np.stack([np.linalg.inv(p[0])[:3, :3] @ [1, 0, 0] for p in poses.values()])
    up = np.linalg.eigh(rights.T @ rights)[1][:, 0]
    downs = np.stack([np.linalg.inv(p[0])[:3, :3] @ [0, 1, 0] for p in poses.values()])  # camera +y: image down
    if up @ downs.mean(0) > 0:
        up = -up
    x = np.cross([0, 0, 1] if abs(up[2]) < 0.9 else [1, 0, 0], up)
    x /= np.linalg.norm(x)
    y = np.cross(up, x)
    R = np.stack([x, y, up])  # rows: aligned axes in world coordinates
    centers = np.stack([np.linalg.inv(p[0])[:3, 3] for p in poses.values()])
    return R, centers @ R.T


class Splat:
    def __init__(self, names, data, device="cuda"):
        ix = {n: i for i, n in enumerate(names)}
        g = lambda keys: torch.from_numpy(data[:, [ix[k] for k in keys]]).to(device)  # noqa: E731
        rest = sorted([n for n in names if n.startswith("f_rest_")], key=lambda s: int(s[7:]))
        self.means = g(["x", "y", "z"])
        self.quats = F.normalize(g([f"rot_{i}" for i in range(4)]), dim=-1)
        self.scales = torch.exp(g([f"scale_{i}" for i in range(3)]))
        self.opac = torch.sigmoid(g(["opacity"])[:, 0])
        dc = g([f"f_dc_{i}" for i in range(3)])[:, None]
        nrest = len(rest) // 3
        shn = g(rest).reshape(-1, 3, nrest).transpose(1, 2) if nrest else dc[:, :0]
        self.colors = torch.cat([dc, shn], 1)
        self.sh_degree = int(round(np.sqrt(nrest + 1))) - 1

    def render(self, w2c, K, W, H, mask=None, ortho=False, depth=False):
        sel = slice(None) if mask is None else mask
        out, alpha, _ = rasterization(
            self.means[sel], self.quats[sel], self.scales[sel], self.opac[sel], self.colors[sel],
            torch.as_tensor(w2c, dtype=torch.float32, device=self.means.device)[None],
            torch.as_tensor(K, dtype=torch.float32, device=self.means.device)[None], W, H,
            sh_degree=self.sh_degree, camera_model="ortho" if ortho else "pinhole",
            render_mode="RGB+ED" if depth else "RGB",
        )  # fmt: skip
        out = out[0]
        if depth:
            return out[..., :3].clamp(0, 1), out[..., 3], alpha[0, ..., 0]
        return out.clamp(0, 1)
