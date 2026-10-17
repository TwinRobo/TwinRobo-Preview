"""Cut a box (in the up-aligned frame from plan.py) out of a splat and render it next to the photos
taken inside the box.

    python crop_splat.py <splat.ply> <scene dir> <out dir> xmin xmax ymin ymax
"""

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from splatlib import Splat, read_ply, read_poses, up_frame, write_ply

src, scene, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
xmin, xmax, ymin, ymax = map(float, sys.argv[4:8])
out.mkdir(parents=True, exist_ok=True)
names, data = read_ply(src)
poses = read_poses(scene / "sparse/0")
R, cams = up_frame(poses)
ix = {n: i for i, n in enumerate(names)}
p = data[:, [ix["x"], ix["y"], ix["z"]]] @ R.T
zc = np.median(cams[:, 2])
inbox = (p[:, 0] > xmin) & (p[:, 0] < xmax) & (p[:, 1] > ymin) & (p[:, 1] < ymax)
zl, zh = np.percentile(p[inbox][:, 2], [0.5, 99.5])
keep = inbox & (p[:, 2] > zl - 0.05) & (p[:, 2] < zh + 0.05)
op = 1 / (1 + np.exp(-data[:, ix["opacity"]]))
keep &= op > 0.02  # nearly transparent gaussians add size, not look
crop = data[keep]
write_ply(out / "crop.ply", names, crop)
print(f"kept {keep.sum()} of {len(data)} gaussians ({keep.sum() * len(names) * 4 / 1e6:.0f} MB as PLY)")

# photos taken inside the box: render the crop from the same cameras
inside = [n for n, c in zip(poses, cams) if xmin < c[0] < xmax and ymin < c[1] < ymax]
print(len(inside), "photos taken inside the box")
s = Splat(names, crop)
rows = []
for name in inside[:: max(1, len(inside) // 6)][:6]:
    w2c, (fx, fy, cx, cy), (w0, h0) = poses[name]
    photo = Image.open(scene / "images_2" / name).convert("RGB")
    W, H = photo.size
    k = W / w0
    K = np.array([[fx * k, 0, cx * k], [0, fy * k, cy * k], [0, 0, 1]])
    img = s.render(w2c, K, W, H).cpu().numpy()
    rows.append(np.concatenate([np.asarray(photo) / 255, img], 1))
Image.fromarray((np.concatenate(rows, 0) * 255).astype(np.uint8)).save(out / "views.jpg", quality=88)
