"""Train a 3D Gaussian splat on part of a COLMAP scene with gsplat (seeded): MCMC densification,
anti-aliased rasterization, per-photo exposure correction, only the photos taken in a box.

    python train2.py <scene dir> <images subdir> <out dir> <steps> <xmin xmax ymin ymax> [cap]

The box is in the up-aligned frame of splatlib.up_frame; photos whose camera sits within it (plus a
0.3 margin) are used. All COLMAP points initialise the splat, so content seen through doorways still
has somewhere to go.

Writes <out>/splat.ply (standard 3DGS layout, loadable by web viewers) and a few held-out
renders next to their photos.
"""

import math
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from gsplat import rasterization
from gsplat.strategy import MCMCStrategy

from splatlib import up_frame

SCENE, IMDIR, OUT = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
STEPS = int(sys.argv[4])
BOX = [float(v) for v in sys.argv[5:9]]
CAP = int(sys.argv[9]) if len(sys.argv) > 9 else 1_500_000
OUT.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)
np.random.seed(0)
dev = "cuda"
NP = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 12}


def read_cameras(p):
    cams = {}
    with open(p, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            cid, model, w, h = struct.unpack("<iiQQ", f.read(24))
            cams[cid] = (model, w, h, struct.unpack(f"<{NP[model]}d", f.read(8 * NP[model])))
    return cams


def read_images(p):
    ims = []
    with open(p, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            iid, qw, qx, qy, qz, tx, ty, tz, cid = struct.unpack("<i7di", f.read(64))
            name = b""
            while (c := f.read(1)) != b"\0":
                name += c
            (k,) = struct.unpack("<Q", f.read(8))
            f.seek(24 * k, 1)
            ims.append((name.decode(), (qw, qx, qy, qz), (tx, ty, tz), cid))
    return sorted(ims)


def read_points(p):
    xyz, rgb = [], []
    with open(p, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            data = struct.unpack("<Q3d3Bd", f.read(43))
            xyz.append(data[1:4])
            rgb.append(data[4:7])
            (k,) = struct.unpack("<Q", f.read(8))
            f.seek(8 * k, 1)
    return np.array(xyz, np.float32), np.array(rgb, np.float32) / 255


def qmat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])  # fmt: skip


# ---- data
sp = SCENE / "sparse/0"
cams, ims = read_cameras(sp / "cameras.bin"), read_images(sp / "images.bin")
pts, cols = read_points(sp / "points3D.bin")
poses_for_frame = {}
for name, q, t, cid in ims:
    w2c = np.eye(4)
    w2c[:3, :3], w2c[:3, 3] = qmat(q), t
    poses_for_frame[name] = (w2c,)
R_up, cam_al = up_frame(poses_for_frame)
m = 0.3
inbox = (cam_al[:, 0] > BOX[0] - m) & (cam_al[:, 0] < BOX[1] + m) & (cam_al[:, 1] > BOX[2] - m) & (cam_al[:, 1] < BOX[3] + m)
ims = [im for im, k in zip(ims, inbox) if k]
first = Image.open(SCENE / IMDIR / ims[0][0])
W, H = first.size
views = []
for name, q, t, cid in ims:
    model, w0, h0, prm = cams[cid]
    fx, fy, cx, cy = (prm[0], prm[0], prm[1], prm[2]) if model in (0, 2, 3) else prm[:4]
    s = W / w0
    K = np.array([[fx * s, 0, cx * s], [0, fy * s, cy * s], [0, 0, 1]], np.float32)
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3], w2c[:3, 3] = qmat(q), t
    img = np.asarray(Image.open(SCENE / IMDIR / name).convert("RGB").resize((W, H)))  # uint8 in memory
    views.append((name, torch.from_numpy(K), torch.from_numpy(w2c), torch.from_numpy(img)))
test = set(range(0, len(views), 8))  # every 8th view held out
train_ids = [i for i in range(len(views)) if i not in test]
centers = np.stack([np.linalg.inv(v[2].numpy())[:3, 3] for v in views])
extent = float(np.linalg.norm(centers - centers.mean(0), axis=1).max()) * 1.1
print(f"{len(views)} views at {W}x{H}, {len(pts)} points, scene extent {extent:.2f}")

# ---- gaussians, initialised from the COLMAP points
N = len(pts)
p = torch.from_numpy(pts).to(dev)
d2 = torch.cdist(p[None], p[None]).squeeze(0) if N < 20000 else None
if d2 is None:  # mean distance to the 3 nearest neighbours, in chunks
    nn = []
    for i in range(0, N, 4096):
        nn.append(torch.cdist(p[i : i + 4096], p).topk(4, largest=False).values[:, 1:].mean(1))
    dist = torch.cat(nn)
else:
    dist = d2.topk(4, largest=False).values[:, 1:].mean(1)
SH = 3
C0 = 0.28209479177387814
sh0 = ((torch.from_numpy(cols).to(dev) - 0.5) / C0)[:, None]
params = torch.nn.ParameterDict({
    "means": torch.nn.Parameter(p),
    "scales": torch.nn.Parameter(torch.log(0.1 * dist.clamp(min=1e-7))[:, None].repeat(1, 3)),
    "quats": torch.nn.Parameter(torch.rand(N, 4, device=dev)),
    "opacities": torch.nn.Parameter(torch.logit(torch.full((N,), 0.5, device=dev))),
    "sh0": torch.nn.Parameter(sh0),
    "shN": torch.nn.Parameter(torch.zeros(N, (SH + 1) ** 2 - 1, 3, device=dev)),
})  # fmt: skip
lrs = {"means": 1.6e-4 * extent, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
opts = {k: torch.optim.Adam([params[k]], lr=lr, eps=1e-15) for k, lr in lrs.items()}
mean_decay = 0.01 ** (1 / STEPS)
strategy = MCMCStrategy(cap_max=CAP, refine_stop_iter=int(STEPS * 0.83), verbose=False)
strategy.check_sanity(params, opts)
state = strategy.initialize_state()
# per-photo exposure: a 3x3 colour matrix and offset, identity at the start
expo = torch.nn.Parameter(torch.zeros(len(views), 3, 4, device=dev))
expo_opt = torch.optim.Adam([expo], lr=1e-3)


def gauss_window(size=11, sigma=1.5):
    g = torch.exp(-((torch.arange(size) - size // 2) ** 2) / (2 * sigma**2))
    g = (g / g.sum())[:, None] @ (g / g.sum())[None]
    return g.expand(3, 1, size, size).contiguous().to(dev)


WIN = gauss_window()


def ssim(a, b):  # a, b: (1, 3, H, W)
    mu_a, mu_b = F.conv2d(a, WIN, padding=5, groups=3), F.conv2d(b, WIN, padding=5, groups=3)
    va = F.conv2d(a * a, WIN, padding=5, groups=3) - mu_a**2
    vb = F.conv2d(b * b, WIN, padding=5, groups=3) - mu_b**2
    cov = F.conv2d(a * b, WIN, padding=5, groups=3) - mu_a * mu_b
    c1, c2 = 0.01**2, 0.03**2
    return (((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (va + vb + c2))).mean()


def render(i, sh_degree):
    _, K, w2c, _ = views[i]
    colors = torch.cat([params["sh0"], params["shN"]], 1)
    img, alpha, info = rasterization(
        params["means"], F.normalize(params["quats"], dim=-1), torch.exp(params["scales"]),
        torch.sigmoid(params["opacities"]), colors, w2c[None].to(dev), K[None].to(dev), W, H,
        sh_degree=sh_degree, packed=False, absgrad=False, rasterize_mode="antialiased",
    )  # fmt: skip
    return img[0], info


t0 = time.time()
rng = np.random.default_rng(0)
for step in range(STEPS):
    i = int(rng.choice(train_ids))
    sh_degree = min(step // 1000, SH)
    img, info = render(i, sh_degree)
    gt = views[i][3].to(dev).float() / 255
    A = torch.eye(3, 4, device=dev) + expo[i]
    img = img @ A[:, :3].T + A[:, 3]
    l1 = (img - gt).abs().mean()
    loss = 0.8 * l1 + 0.2 * (1 - ssim(img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]))
    loss = loss + 0.01 * torch.sigmoid(params["opacities"]).mean() + 0.01 * torch.exp(params["scales"]).mean()
    loss.backward()
    for o in [*opts.values(), expo_opt]:
        o.step()
        o.zero_grad(set_to_none=True)
    for g in opts["means"].param_groups:
        g["lr"] *= mean_decay
    strategy.step_post_backward(params, opts, state, step, info, lr=opts["means"].param_groups[0]["lr"])
    if step % 1000 == 0 or step == STEPS - 1:
        print(f"step {step} loss {loss.item():.4f} gaussians {len(params['means'])} {time.time() - t0:.0f}s", flush=True)

# ---- held-out quality and side-by-side renders
psnrs, rows = [], []
with torch.no_grad():
    for k, i in enumerate(sorted(test)):
        img, _ = render(i, SH)
        img = img.clamp(0, 1)
        gt = views[i][3].to(dev).float() / 255
        psnrs.append(float(-10 * torch.log10(((img - gt) ** 2).mean())))
        if k % max(1, len(test) // 6) == 0 and len(rows) < 6:
            rows.append(np.concatenate([gt.cpu().numpy(), img.cpu().numpy()], 1))
print(f"held-out PSNR {np.mean(psnrs):.2f} dB over {len(psnrs)} views")
(OUT / "test_views.txt").write_text("\n".join(views[i][0] for i in sorted(test)))
Image.fromarray((np.concatenate(rows, 0) * 255).astype(np.uint8)).save(OUT / "heldout.jpg", quality=90)

# ---- export (standard 3DGS PLY)
with torch.no_grad():
    m = params["means"].cpu().numpy()
    f_dc = params["sh0"].squeeze(1).cpu().numpy()
    f_rest = params["shN"].transpose(1, 2).reshape(len(m), -1).cpu().numpy()
    op = params["opacities"][:, None].cpu().numpy()
    sc = params["scales"].cpu().numpy()
    rot = F.normalize(params["quats"], dim=-1).cpu().numpy()
names = (["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)]
         + [f"f_rest_{i}" for i in range(f_rest.shape[1])] + ["opacity"]
         + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)])  # fmt: skip
data = np.concatenate([m, np.zeros_like(m), f_dc, f_rest, op, sc, rot], 1).astype(np.float32)
with open(OUT / "splat.ply", "wb") as f:
    f.write(("ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % len(m)).encode())
    f.write("".join(f"property float {n}\n" for n in names).encode())
    f.write(b"end_header\n")
    f.write(data.tobytes())
print("wrote", OUT / "splat.ply", f"{len(m)} gaussians, {time.time() - t0:.0f}s")
