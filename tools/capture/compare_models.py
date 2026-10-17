"""Old vs new splat on the new run's held-out photos (PSNR, side by side), plus an elevated view
from a spot no photo was taken from.

    python compare_models.py <old.ply> <new.ply> <scene dir> <test_views.txt> <out prefix>
"""

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from splatlib import Splat, read_ply, read_poses, up_frame

old_p, new_p, scene, tests, out = sys.argv[1], sys.argv[2], Path(sys.argv[3]), sys.argv[4], sys.argv[5]
poses = read_poses(scene / "sparse/0")
names = [n for n in Path(tests).read_text().split("\n") if n]
models = [Splat(*read_ply(p)) for p in (old_p, new_p)]
W = 1393
rows, psnr = [], [[], []]
for k, name in enumerate(names):
    w2c, (fx, fy, cx, cy), (w0, h0) = poses[name]
    photo = Image.open(scene / "images" / name).convert("RGB")
    H = round(photo.height * W / photo.width)
    gt = np.asarray(photo.resize((W, H), Image.LANCZOS)) / 255
    s = W / w0
    K = np.array([[fx * s, 0, cx * s], [0, fy * s, cy * s], [0, 0, 1]])
    ims = [m.render(w2c, K, W, H).cpu().numpy() for m in models]
    for j, im in enumerate(ims):
        psnr[j].append(-10 * np.log10(((im - gt) ** 2).mean()))
    if k % max(1, len(names) // 5) == 0 and len(rows) < 5:
        rows.append(np.concatenate([gt, *ims], 1))
print(f"{len(names)} held-out kitchen photos: PSNR old {np.mean(psnr[0]):.2f} dB, new {np.mean(psnr[1]):.2f} dB")
Image.fromarray((np.concatenate(rows, 0) * 255).astype(np.uint8)).resize((W * 3 // 2, H * len(rows) // 2)).save(
    f"{out}_heldout.jpg", quality=88)

# elevated view: 1 unit above the photos' mean height, over the kitchen's middle, looking down 45 degrees
R, cams = up_frame(poses)
inside = np.array([n in set(names) for n in poses])
c = cams[inside].mean(0) if inside.any() else cams.mean(0)
eye = c + [0.6, 0.6, 1.0]
fwd = np.array([-0.6, -0.6, -1.0]) / np.linalg.norm([0.6, 0.6, 1.0])
right = np.cross(fwd, [0, 0, 1]); right /= np.linalg.norm(right)
down = np.cross(fwd, right)
Rc = np.stack([right, down, fwd])  # camera axes (x right, y down, z forward) in the aligned frame
w2c = np.eye(4)
w2c[:3, :3] = Rc @ R
w2c[:3, 3] = -Rc @ eye
W2, H2 = 1200, 800
K = np.array([[700, 0, W2 / 2], [0, 700, H2 / 2], [0, 0, 1]])
ims = [m.render(w2c, K, W2, H2).cpu().numpy() for m in models]
Image.fromarray((np.concatenate(ims, 1) * 255).astype(np.uint8)).save(f"{out}_elevated.jpg", quality=88)
