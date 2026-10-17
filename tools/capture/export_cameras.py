"""Export TwinRobo catalog cameras for the browser demo: web/cameras/index.json + one JSON each.

    python export_cameras.py <out dir> [widths 320,640]

Per camera (one eye, or a stereo module's eyes):
- the render: pinhole field of view that covers every sensor pixel, and the sensor's aspect ratio;
- the distortion: a map from sensor position to pinhole-render position (identity for stereo
  modules, whose SDK delivers rectified images);
- the blur: TwinRobo's PSF bank (field grid x depth layers, DeepLens, seeded) resampled to the web
  image width with tent weights (energy kept), cropped to the radius holding 99.5 % of the energy
  (at most 15 px), and compressed with an SVD to 4 basis kernels shared by all positions, depths and
  colours, plus their weights per node. The browser blends the weights by position and depth.
"""

import base64
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from twinrobo.camera import build_reference_optics, psf_bank_params, spec_intrinsics
from twinrobo.optics.cache import PSFCache
from twinrobo.optics.distortion import RadialDistortion
from twinrobo.registry import CatalogRegistry
from twinrobo.spec import CameraSpec
from twinrobo.stereo import ModuleRegistry

OUT = Path(sys.argv[1])
WIDTHS = [int(w) for w in (sys.argv[2] if len(sys.argv) > 2 else "320,640").split(",")]
RANK, KMAX = 4, 15
MAP = (33, 21)  # distortion map nodes (w, h)
OUT.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)


def fin(v):
    """JSON has no infinity: None for it."""
    return None if v is None or not math.isfinite(float(v)) else float(v)


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a, np.float32).tobytes()).decode()


def resample(psfs, f, n):
    """[..., ks, ks] kernels at sensor pixels -> [..., n, n] at pixels f times larger (tent weights)."""
    ks = psfs.shape[-1]
    c, cn = ks // 2, n // 2
    p = (torch.arange(ks, dtype=torch.float64) - c) / f
    i = torch.arange(n, dtype=torch.float64) - cn
    M = (1 - (p[None, :] - i[:, None]).abs()).clamp(min=0).to(psfs)  # [n, ks], columns sum to 1
    return M @ psfs @ M.T


def blur_package(bank, width):
    W, H = bank.sensor_resolution
    f = W / width
    psfs = bank.psfs.double()  # [L, Gh, Gw, C, ks, ks]
    ks = psfs.shape[-1]
    # radius holding 99.5 % of the energy, over all kernels (at web pixels)
    yy, xx = torch.meshgrid(torch.arange(ks) - ks // 2, torch.arange(ks) - ks // 2, indexing="ij")
    rr = torch.sqrt((xx**2 + yy**2).double()).to(psfs.device)
    flat = psfs.reshape(-1, ks * ks)
    order = torch.argsort(rr.flatten())
    cum = torch.cumsum(flat[:, order], 1) / flat.sum(1, keepdim=True)
    r995 = float(rr.flatten()[order][(cum < 0.995).sum(1).clamp(max=ks * ks - 1)].max())
    n = int(min(KMAX, 2 * math.ceil(r995 / f + 0.5) + 1))
    k = resample(psfs, f, n)
    k = k / k.sum((-1, -2), keepdim=True)
    X = k.reshape(-1, n * n)
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    r = min(RANK, n * n)
    B = Vh[:r]  # [r, n*n]
    Wt = X @ B.T  # [N, r]
    err = float(((Wt @ B - X).norm() / X.norm()))
    L, Gh, Gw, C = k.shape[:4]
    return {
        "size": n, "rank": r, "relative_error": round(err, 4), "r995_sensor_px": round(r995, 2),
        "basis": b64(B.reshape(r, n, n).cpu().numpy()),  # [r, n, n]
        "weights": b64(Wt.reshape(L, Gh, Gw, C, r).cpu().numpy()),  # [L, Gh, Gw, C, r]
    }  # fmt: skip


def camera_package(spec: CameraSpec, rectified: bool):
    ref = build_reference_optics(spec, "cuda")
    intr = spec_intrinsics(spec, ref)
    W, H = spec.sensor.resolution.width, spec.sensor.resolution.height
    dist = None if rectified else RadialDistortion.from_dict(spec.calibration.get("distortion"))
    # distortion map: sensor position (u, v in 0..1, v down) -> undistorted normalized ray (x, y)
    u = np.linspace(0, 1, MAP[0])
    v = np.linspace(0, 1, MAP[1])
    U, V = np.meshgrid(u, v)
    xd, yd = (U * W - 0.5 - intr.cx) / intr.fx, (V * H - 0.5 - intr.cy) / intr.fy
    x, y = dist.undistort(xd, yd) if dist is not None and not dist.is_identity else (xd, yd)
    tx, ty = float(np.abs(x).max() * 1.01), float(np.abs(y).max() * 1.01)
    blur = None
    if ref is not None:
        params = psf_bank_params(spec)
        bank = PSFCache().get_or_build(ref.bank_key(**params), lambda: ref.build_psf_bank(**params), device="cuda",
                                       source=spec.source_path)  # fmt: skip
        blur = {
            "grid": list(bank.grid), "depths_m": [fin(d) for d in bank.depths_m.cpu()],  # null = infinity
            "channels": bank.channels, "levels": {str(w): blur_package(bank, w) for w in WIDTHS},
        }  # fmt: skip
    return {
        "sensor": [W, H], "mono": spec.sensor.color == "mono",
        "render": {"tan_x": tx, "tan_y": ty, "vfov_deg": math.degrees(2 * math.atan(ty))},
        "map": {"size": list(MAP), "data": b64(np.stack([x / tx, y / ty], -1))},  # in -1..1 of the render
        "intrinsics": {"fx": intr.fx, "fy": intr.fy, "cx": intr.cx, "cy": intr.cy},  # sensor pixels
        "distortion": None if dist is None else dist.to_dict(),
        "rectified": rectified, "focus_m": fin(spec.lens.focus_distance_m), "f_number": spec.lens.f_number,
        "focal_mm": spec.lens.focal_length_mm, "blur": blur,
    }  # fmt: skip


reg = CatalogRegistry()
index, done = [], {}
for cid in reg.list():
    spec = reg.load(cid)
    pkg = camera_package(spec, rectified=False)
    name = cid.replace("/", "--")
    (OUT / f"{name}.json").write_text(json.dumps(pkg, allow_nan=False))
    done[(cid, False)] = name
    label = f"{spec.manufacturer} {spec.product}"
    lv = pkg["blur"]["levels"] if pkg["blur"] else {}
    print(f"{cid}: {pkg['sensor']}, vfov {pkg['render']['vfov_deg']:.1f} deg, "
          + ", ".join(f"{w}px kernel {l['size']} err {l['relative_error']}" for w, l in lv.items()))  # fmt: skip
    index.append({"id": cid, "label": label, "type": "mono" if pkg["mono"] else "camera", "eyes": {"left": name}})
for m in ModuleRegistry().list():
    eyes = {}
    for eye in ("left", "right"):
        spec = CameraSpec.from_yaml(m.eye(eye).spec)
        key = (spec.id, m.output == "rectified")
        if key not in done:
            name = spec.id.replace("/", "--") + ("--rectified" if key[1] else "")
            (OUT / f"{name}.json").write_text(json.dumps(camera_package(spec, rectified=key[1]), allow_nan=False))
            done[key] = name
        eyes[eye] = done[key]
    index.append({"id": m.id, "label": m.label, "type": "stereo", "eyes": eyes,
                  "right_translation_m": list(m.right.translation_m), "baseline_mm": round(1000 * m.baseline_m, 1),
                  "output": m.output})  # fmt: skip
    print(f"{m.id}: stereo, baseline {1000 * m.baseline_m:.0f} mm, {m.output}")
(OUT / "index.json").write_text(json.dumps({"cameras": index}, indent=1))
print(len(index), "entries ->", OUT / "index.json")
