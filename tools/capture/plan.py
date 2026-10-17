"""Top-down floor plan of a splat: gaussians between the floor and just under the ceiling, rendered
orthographically from above, with a metric grid (aligned frame: z up, units of the scene).

    python plan.py <splat.ply> <sparse dir> <out.jpg>
"""

import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

from splatlib import Splat, read_ply, read_poses, up_frame

names, data = read_ply(sys.argv[1])
poses = read_poses(sys.argv[2])
R, cams = up_frame(poses)
s = Splat(names, data)
p = s.means.cpu().numpy() @ R.T
z = p[:, 2]
zc = np.median(cams[:, 2])
zl, zh = np.percentile(z, [0.5, 99.5])
below, above = z[(z > zl) & (z < zc)], z[(z > zc) & (z < zh)]
hb, eb = np.histogram(below, bins=150)
ha, ea = np.histogram(above, bins=150)
floor, ceil = eb[np.argmax(hb)], ea[np.argmax(ha)]  # densest layers below and above the cameras
print(f"floor {floor:.2f}, ceiling {ceil:.2f} (scene units; ceiling - floor = {ceil - floor:.2f})")
keep = (z > floor - 0.05) & (z < floor + 0.8 * (ceil - floor))
lo, hi = np.percentile(p[keep][:, :2], [1, 99], axis=0)
if len(sys.argv) > 4:  # explicit bounds: xmin xmax ymin ymax
    b = [float(v) for v in sys.argv[4:8]]
    lo, hi = np.array([b[0], b[2]]), np.array([b[1], b[3]])
PX = 1400
scale = PX / (hi - lo).max()
W, H = [int(v) for v in (hi - lo) * scale]
# ortho camera above, looking down -z; image x = aligned x, image y = -aligned y
c2w_al = np.eye(4)
c2w_al[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).T
c2w_al[:3, 3] = [lo[0], hi[1], ceil + 5]
w2c = np.linalg.inv(c2w_al) @ np.block([[R, np.zeros((3, 1))], [np.zeros((1, 3)), 1]])
K = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])
img = s.render(w2c, K, W, H, mask=torch.from_numpy(keep).cuda(), ortho=True)
im = Image.fromarray((img.cpu().numpy() * 255).astype(np.uint8))
d = ImageDraw.Draw(im)
STEP = 0.5 if (hi - lo).max() < 8 else 1.0
for gx in np.arange(np.ceil(lo[0]), hi[0], STEP):
    X = (gx - lo[0]) * scale
    d.line([(X, 0), (X, H)], fill=(255, 0, 0), width=1)
    d.text((X + 2, 2), f"x={gx:g}", fill=(255, 0, 0))
for gy in np.arange(np.ceil(lo[1]), hi[1], STEP):
    Y = (hi[1] - gy) * scale
    d.line([(0, Y), (W, Y)], fill=(255, 0, 0), width=1)
    d.text((2, Y + 2), f"y={gy:g}", fill=(255, 0, 0))
im.save(sys.argv[3], quality=88)
np.save(sys.argv[3] + ".frame.npy", np.concatenate([R.ravel(), [floor, ceil]]))
print("plan", W, "x", H, "px,", f"{1 / scale * 100:.1f} cm/px (scene units)")
