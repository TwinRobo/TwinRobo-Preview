"""3DGS PLY -> SPZ (Niantic's compressed splat format, version 2), as read by Spark.

    python ply2spz.py <in.ply> <out.spz> [sh degree 0..3, default 1]

Layout (gzipped): a 16-byte header (magic "NGSP", version, count, SH degree, fractional bits,
flags, reserved), then per field for all points: positions (3 x 24-bit fixed point), alphas
(u8), colours (3 x u8, DC scaled by 0.15), scales (3 x u8, (log + 10) * 16), rotations (3 x u8,
x y z of a w >= 0 unit quaternion), spherical harmonics (u8, 5 bits for degree 1, 4 above).
"""

import gzip
import struct
import sys

import numpy as np

from splatlib import read_ply

src, dst = sys.argv[1], sys.argv[2]
DEG = int(sys.argv[3]) if len(sys.argv) > 3 else 1
FRAC = 12
names, d = read_ply(src)
ix = {n: i for i, n in enumerate(names)}
col = lambda keys: d[:, [ix[k] for k in keys]]  # noqa: E731
n = len(d)

pos = np.round(col(["x", "y", "z"]) * (1 << FRAC)).astype(np.int64)
assert np.abs(pos).max() < (1 << 23), "positions out of range for 24-bit fixed point"
pos &= 0xFFFFFF
pos_b = np.stack([pos & 0xFF, (pos >> 8) & 0xFF, (pos >> 16) & 0xFF], -1).astype(np.uint8).reshape(n, 9)

u8 = lambda x: np.clip(np.round(x), 0, 255).astype(np.uint8)  # noqa: E731
alpha = u8(255 / (1 + np.exp(-col(["opacity"])[:, 0])))
colors = u8(col(["f_dc_0", "f_dc_1", "f_dc_2"]) * (0.15 * 255) + 0.5 * 255)
scales = u8((col(["scale_0", "scale_1", "scale_2"]) + 10) * 16)
q = col(["rot_0", "rot_1", "rot_2", "rot_3"])  # w x y z
q /= np.linalg.norm(q, axis=1, keepdims=True)
q[q[:, 0] < 0] *= -1
rots = u8(q[:, 1:] * 127.5 + 127.5)

parts = [pos_b.tobytes(), alpha.tobytes(), colors.tobytes(), scales.tobytes(), rots.tobytes()]
if DEG > 0:
    rest = sorted([k for k in names if k.startswith("f_rest_")], key=lambda s: int(s[7:]))
    per = len(rest) // 3  # coefficients per channel in the PLY (channel-major)
    sh = col(rest).reshape(n, 3, per).transpose(0, 2, 1)  # [n, coef, rgb]
    keep = (DEG + 1) ** 2 - 1
    sh = sh[:, :keep]
    qv = np.round(sh * 128 + 128)
    bucket = np.where(np.arange(keep) < 3, 8, 16)[None, :, None]  # degree 1: 5 bits, higher: 4 bits
    qv = np.clip(np.floor((qv + bucket / 2) / bucket) * bucket, 0, 255).astype(np.uint8)
    parts.append(qv.tobytes())
header = struct.pack("<IIIBBBB", 0x5053474E, 2, n, DEG, FRAC, 1, 0)  # flags: antialiased
with gzip.open(dst, "wb", compresslevel=9) as f:
    f.write(header + b"".join(parts))
import os  # noqa: E402

print(f"{n} splats, SH degree {DEG}: {os.path.getsize(src) / 1e6:.0f} MB -> {os.path.getsize(dst) / 1e6:.1f} MB")
