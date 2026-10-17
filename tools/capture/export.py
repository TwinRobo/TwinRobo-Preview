"""Package a cropped splat for the web demo: metric scale, a three.js transform (y up, floor at 0)
and a 2D obstacle map for driving.

    python export.py <crop.ply> <scene dir> <out dir> <counter x0 x1 y0 y1> [counter height m]

The counter box (up-aligned frame from plan.py) is a horizontal surface of known height above the
floor (US kitchen counters: 36 in = 0.914 m); it sets the scale.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from splatlib import read_ply, read_poses, up_frame, write_ply

src, scene, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
cx0, cx1, cy0, cy1 = map(float, sys.argv[4:8])
COUNTER = float(sys.argv[8]) if len(sys.argv) > 8 else 0.914
out.mkdir(parents=True, exist_ok=True)
names, data = read_ply(src)
ix = {n: i for i, n in enumerate(names)}
R, cams = up_frame(read_poses(scene / "sparse/0"))
p = data[:, [ix["x"], ix["y"], ix["z"]]] @ R.T
op = 1 / (1 + np.exp(-data[:, ix["opacity"]]))
solid = op > 0.3


def layer(z, lo, hi, bins=200):
    h, e = np.histogram(z[(z > lo) & (z < hi)], bins=bins, range=(lo, hi))
    return e[np.argmax(h)] + (e[1] - e[0]) / 2


zc = np.median(cams[:, 2])
z0 = np.percentile(p[solid, 2], 1.5)  # nothing solid lies below the floor
floor = layer(p[solid, 2], z0 - 0.05, z0 + 0.05, bins=50)
on_counter = solid & (p[:, 0] > cx0) & (p[:, 0] < cx1) & (p[:, 1] > cy0) & (p[:, 1] < cy1)
top = layer(p[on_counter, 2], floor + 0.2 * (zc - floor), zc)
s = COUNTER / (top - floor)
print(f"floor {floor:.3f}, counter top {top:.3f} (units) -> {s:.3f} m per unit; camera height {(zc - floor) * s:.2f} m")

# metric, floor at 0, centred on the crop; three.js axes: x, y up, z = -aligned y
m = (p - [0, 0, floor]) * s
c = np.percentile(m[solid, :2], [1, 99], axis=0).mean(0)
m[:, :2] -= c
A = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)  # aligned (x, y, z) -> three (x, z, -y)
# world (COLMAP) -> three: x_three = A @ (s * R @ x - [c, s * floor]) = s * (A R) x + t
Rt = A @ R
t = -(A @ np.array([c[0], c[1], s * floor]))  # c is already in meters
M = np.eye(4)
M[:3, :3], M[:3, 3] = s * Rt, t
assert np.allclose(data[:, [ix["x"], ix["y"], ix["z"]]] @ (s * Rt).T + t, m @ A.T, atol=1e-4)

# obstacle map: solid gaussians between 5 cm and 1.6 m high, on a 5 cm grid
CELL = 0.05
xy = m[:, :2]
lo, hi = np.percentile(xy[solid], [0.5, 99.5], axis=0) + [[-0.2], [0.2]]
nx, ny = np.ceil((hi - lo) / CELL).astype(int)
occ = np.zeros((ny, nx), np.uint16)
floor_seen = np.zeros((ny, nx), np.uint16)
ij = np.floor((xy - lo) / CELL).astype(int)
ok = (ij[:, 0] >= 0) & (ij[:, 0] < nx) & (ij[:, 1] >= 0) & (ij[:, 1] < ny)
obst = ok & solid & (m[:, 2] > 0.05) & (m[:, 2] < 1.6)
np.add.at(occ, (ij[obst, 1], ij[obst, 0]), 1)
fl = ok & solid & (np.abs(m[:, 2]) < 0.04)
np.add.at(floor_seen, (ij[fl, 1], ij[fl, 0]), 1)
# drivable: not an obstacle, and connected to the most open cell (walls bound it; the floor itself is
# a few large gaussians, too sparse to map directly)
obstacle = occ >= 3
dist = np.zeros(obstacle.shape, int)
front = obstacle.copy()
front[[0, -1], :] = front[:, [0, -1]] = True
for d in range(1, 40):  # distance (cells) to the nearest obstacle or map edge
    grown = front.copy()
    grown[1:] |= front[:-1]; grown[:-1] |= front[1:]; grown[:, 1:] |= front[:, :-1]; grown[:, :-1] |= front[:, 1:]
    dist[grown & ~front] = d
    front = grown
dist[~front] = 40
near_floor = floor_seen > 0
for _ in range(6):  # floor evidence within 30 cm
    g = near_floor.copy()
    g[1:] |= near_floor[:-1]; g[:-1] |= near_floor[1:]; g[:, 1:] |= near_floor[:, :-1]; g[:, :-1] |= near_floor[:, 1:]
    near_floor = g
obstacle |= ~near_floor
dist[~near_floor] = 0
seed = np.unravel_index(np.argmax(dist), dist.shape)
reach = np.zeros(obstacle.shape, bool)
reach[seed] = True
while True:
    grown = reach.copy()
    grown[1:] |= reach[:-1]; grown[:-1] |= reach[1:]; grown[:, 1:] |= reach[:, :-1]; grown[:, :-1] |= reach[:, 1:]
    grown &= ~obstacle
    if (grown == reach).all():
        break
    reach = grown
# holes inside the drivable area where the floor went unseen (no obstacle there either) are floor too
outside = np.zeros(reach.shape, bool)
outside[[0, -1], :] = outside[:, [0, -1]] = True
outside &= ~reach
while True:
    g = outside.copy()
    g[1:] |= outside[:-1]; g[:-1] |= outside[1:]; g[:, 1:] |= outside[:, :-1]; g[:, :-1] |= outside[:, 1:]
    g &= ~reach
    if (g == outside).all():
        break
    outside = g
reach |= ~outside & (occ < 3)
blocked = ~reach
dbg = np.stack([np.clip(occ * 40, 0, 255), np.clip(floor_seen * 40, 0, 255), np.zeros_like(occ)], -1).astype(np.uint8)
Image.fromarray(dbg[::-1]).resize((nx * 4, ny * 4), Image.NEAREST).save(out / "occupancy_debug.png")
img = np.where(blocked, 0, 255).astype(np.uint8)
Image.fromarray(img[::-1]).save(out / "occupancy.png")  # row 0 = largest aligned y = most negative three z
spawn_ij = np.array(seed)
spawn = lo + (spawn_ij[::-1] + 0.5) * CELL

keep = op > 0.05
write_ply(out / "kitchen.ply", names, data[keep])
meta = {
    "splat": "kitchen.ply",
    "transform": M.T.ravel().tolist(),  # column-major, for THREE.Matrix4.fromArray
    "metersPerUnit": s,
    "occupancy": {"image": "occupancy.png", "cell": CELL, "origin": lo.tolist(), "size": [int(nx), int(ny)],
                  "note": "aligned x right, aligned y up the image (three.js z = -aligned y); dark = blocked"},
    "spawn": {"x": float(spawn[0]), "z": float(-spawn[1]),
              "lookAt": (lambda q: {"x": float(q[0]), "z": float(-q[1])})(
                  (np.array([(cx0 + cx1) / 2, (cy0 + cy1) / 2]) * s - c))},  # the counter
    "source": "Zip-NeRF 'nyc' scene (Barron et al., CC BY 4.0), cropped and trained with gsplat",
}  # fmt: skip
if os.environ.get("START"):  # the opening view: x, z (m, three.js), heading (deg), camera tilt (rad)
    x, z, hd, tl = map(float, os.environ["START"].split(","))
    meta["start"] = {"x": x, "z": z, "heading": float(np.radians(hd)), "tilt": tl}
(out / "scene.json").write_text(json.dumps(meta, indent=1))
print(f"{keep.sum()} gaussians -> {out / 'kitchen.ply'}; map {nx}x{ny} cells, {(~blocked).mean() * 100:.0f}% free; spawn {spawn.round(2)}")
