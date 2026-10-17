"""A coarse mesh of the scene for depth in the browser: TSDF fusion of the splat's depth renders
from the training cameras, in the web scene's metric frame (scene.json), marching cubes.

    python depth_mesh.py <crop.ply> <scene dir> <web dir> [voxel m]

Writes <web dir>/depth.ply (binary, float32 vertices, uint32 faces) and adds it to scene.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from skimage.measure import marching_cubes

from splatlib import Splat, read_ply, read_poses

src, scene, web = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
VOX = float(sys.argv[4]) if len(sys.argv) > 4 else 0.03
meta = json.loads((web / "scene.json").read_text())
M = np.array(meta["transform"]).reshape(4, 4).T  # COLMAP -> metric three.js frame (scale s)
s_scale = meta["metersPerUnit"]
names, data = read_ply(src)
splat = Splat(names, data)
poses = read_poses(scene / "sparse/0")

# volume over the scene's solid gaussians (metric frame)
op = 1 / (1 + np.exp(-data[:, names.index("opacity")]))
p = data[op > 0.3][:, :3] @ M[:3, :3].T + M[:3, 3]
lo, hi = np.percentile(p, 0.5, axis=0) - 0.1, np.percentile(p, 99.5, axis=0) + 0.1
dims = np.ceil((hi - lo) / VOX).astype(int)
print("volume", dims, "voxels of", VOX, "m")
dev = "cuda"
tsdf = torch.ones(*dims, device=dev)
wsum = torch.zeros(*dims, device=dev)
g = torch.stack(torch.meshgrid(*[torch.arange(d, device=dev) for d in dims], indexing="ij"), -1).reshape(-1, 3)
pts = torch.as_tensor(lo, device=dev, dtype=torch.float32) + (g.float() + 0.5) * VOX  # voxel centres, metric
Minv = torch.as_tensor(np.linalg.inv(M), device=dev, dtype=torch.float32)
pts_w = pts @ Minv[:3, :3].T + Minv[:3, 3]  # COLMAP frame
TRUNC = 4 * VOX
W = 464
inside = 0
for name, (w2c, (fx, fy, cx, cy), (w0, h0)) in poses.items():
    c = np.linalg.inv(w2c)[:3, 3]
    cm = M[:3, :3] @ c + M[:3, 3]
    if not ((cm > lo) & (cm < hi)).all():
        continue  # only cameras inside the volume
    inside += 1
    k = W / w0
    H = round(h0 * k)
    K = np.array([[fx * k, 0, cx * k], [0, fy * k, cy * k], [0, 0, 1]])
    _, depth, alpha = splat.render(w2c, K, W, H, depth=True)  # depth along the view axis, COLMAP units
    depth = torch.where(alpha > 0.5, depth, torch.zeros_like(depth)) * s_scale  # "ED": already normalized
    Rt = torch.as_tensor(w2c, device=dev, dtype=torch.float32)
    pc = pts_w @ Rt[:3, :3].T + Rt[:3, 3]
    z = pc[:, 2] * s_scale
    u = (pc[:, 0] / pc[:, 2] * K[0, 0] + K[0, 2]).round().long()
    v = (pc[:, 1] / pc[:, 2] * K[1, 1] + K[1, 2]).round().long()
    ok = (z > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    d = torch.zeros_like(z)
    d[ok] = depth[v[ok], u[ok]]
    sdf = d - z
    ok &= (d > 0) & (sdf > -TRUNC)
    t = (sdf / TRUNC).clamp(max=1)
    tf, wf = tsdf.view(-1), wsum.view(-1)
    tf[ok] = (tf[ok] * wf[ok] + t[ok]) / (wf[ok] + 1)
    wf[ok] += 1
print(inside, "cameras fused")
vol = tsdf.cpu().numpy()
vol[wsum.cpu().numpy() == 0] = 1
verts, faces, _, _ = marching_cubes(vol, 0.0, spacing=(VOX,) * 3)
verts = verts + lo + VOX / 2
# keep the floor and below-ceiling parts that matter; drop tiny floating pieces is skipped (coarse depth only)
print(f"{len(verts)} vertices, {len(faces)} faces")
with open(web / "depth.ply", "wb") as f:
    f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(verts)}\nproperty float x\nproperty float y\n"
             f"property float z\nelement face {len(faces)}\nproperty list uchar uint vertex_indices\nend_header\n").encode())  # fmt: skip
    f.write(verts.astype(np.float32).tobytes())
    rec = np.zeros(len(faces), dtype=[("n", "u1"), ("i", "<u4", 3)])
    rec["n"], rec["i"] = 3, faces[:, ::-1]  # outward winding for three.js
    f.write(rec.tobytes())
meta["depthMesh"] = "depth.ply"
(web / "scene.json").write_text(json.dumps(meta, indent=1))
print("wrote", web / "depth.ply", f"{(web / 'depth.ply').stat().st_size / 1e6:.1f} MB")
