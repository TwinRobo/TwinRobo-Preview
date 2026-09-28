"""Precompute the preview: one demo replayed through every robot camera and camera config.

Drives a running TwinRobo Studio (its HTTP API) and saves what the Studio shows, so the
static site needs no server:

- ``site/data/<scene>/scene.json``, ``scene.bin.gz``, ``tex/<id>.webp``, ``poses.json`` +
  ``poses.bin``: the 3D scene (geometry, textures and per-frame poses) for the three.js panel;
- ``site/data/<scene>/<camera>/<config>/{image,diff,depth}.mp4``: every frame of the
  demo through that camera config (H.264, 20 fps); ``diff`` is ``|camera - pinhole| x 10``
  (camera twins only), ``depth`` only for cameras that output depth (RealSense);
- ``site/data/<scene>/meta.json`` and ``site/data/manifest.json``: what the page lists.

    python -m twinrobo_viewer --port 8791 --demos ... --robocasa ...   # a Studio to bake from
    python tools/bake.py --studio http://127.0.0.1:8791 --scene libero
    python tools/bake.py --studio http://127.0.0.1:8791 --scene robocasa
    python tools/bake.py --manifest                                    # after both

Streams already on disk are skipped, so an interrupted bake resumes.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import io
import json
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "site" / "data"
FPS = 20  # the demos' control rate; the Studio plays them at this rate too
WIDTH = 640  # video width (height keeps the camera's aspect)
TEXTURE_MAX = 1024  # 3D-panel textures: longest side (WebP)
CRF = {"image": 26, "diff": 34, "depth": 34}  # H.264 quality per view (diff/depth: on demand)
POSE_ARRAYS = ("body_pos", "body_quat", "cam_pos", "cam_mat")  # float32, in poses.bin

SCENES = {
    "libero": {
        "id": "libero-spatial-cookie-box",
        "simulator": "libero",
        "run": "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_demo/demo_0",
    },
    "robocasa": {
        "id": "robocasa-counter-to-cabinet",
        "simulator": "robocasa",
        "run": "robocasa/PickPlaceCounterToCabinet/20250819/episode_000000",
    },
}
# The catalog cameras the preview shows (on the wrist), in this order.
CAMERAS = [
    "intel/realsense-d455/color",
    "logitech/c920",
    "luxonis/oak-d/mono",
    "stereolabs/zed-x/2.2mm",
]
METHODS = ("psf", "pupil", "raycast")  # TwinRobo's rendering methods (catalog cameras only)
# the camera the catalog cameras are applied to; the others stay pinhole
WRIST = "robot0_eye_in_hand"
# Lens-ray methods trace every sensor pixel: larger sensors (12 MP) are read out at this width
# (same lens and field of view), which a 24 GB GPU holds; the videos are 640 px anyway.
LENS_RAY_MAX_WIDTH = 1920


class Studio:
    def __init__(self, url: str):
        self.url = url.rstrip("/")

    def raw(self, path: str) -> bytes:
        with urllib.request.urlopen(self.url + path, timeout=1800) as r:
            return r.read()

    def get(self, path: str):
        return json.loads(self.raw(path))

    def post(self, path: str, body: dict):
        req = urllib.request.Request(
            self.url + path, json.dumps(body).encode(), {"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=1800) as r:
            j = json.loads(r.read())
        if isinstance(j, dict) and j.get("error"):
            raise RuntimeError(j["error"])
        return j


def decode(data_url: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1]))).convert("RGB")


class Video:
    """Frames piped to ffmpeg as raw RGB; H.264 with a short GOP so seeking is fast."""

    def __init__(self, path: Path, size: tuple[int, int], crf: int):
        self.path, self.tmp = path, path.with_suffix(".part.mp4")
        w, h = size
        self.proc = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow",
             "-crf", str(crf), "-g", "10", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
             str(self.tmp)],
            stdin=subprocess.PIPE,
        )  # fmt: skip
        self.size = size

    def add(self, img: Image.Image) -> None:
        self.proc.stdin.write(img.resize(self.size, Image.LANCZOS).tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed for {self.path}")
        self.tmp.replace(self.path)


def out_size(resolution: list[int]) -> tuple[int, int]:
    w, h = resolution
    W = min(WIDTH, w)
    return W, round(W * h / w / 2) * 2


def spec_facts(s: dict) -> dict:
    keys = ("id", "manufacturer", "product", "sensor", "width", "height", "pixel_pitch_um",
            "focal_length_mm", "f_number", "focus_distance_m", "fov_deg", "status", "shutter")  # fmt: skip
    return {k: s.get(k) for k in keys}


def bake_scene(studio: Studio, key: str, only: list[str] | None) -> None:
    sc = SCENES[key]
    out = DATA / sc["id"]
    out.mkdir(parents=True, exist_ok=True)
    studio.post("/api/source", {"simulator": sc["simulator"]})
    state = studio.post("/api/open", {"run": sc["run"]})
    run, robot = state["run"], state["robot"]
    info = next(r for r in state["robot_catalog"] if r["name"] == robot)
    n = run["num_frames"]
    print(f"{sc['id']}: {run['instruction']} · {robot} · {n} frames", flush=True)

    # 3D scene: the Studio's own geometry, textures and poses
    scene = studio.get("/api/scene")
    (out / "scene.json").write_text(json.dumps(scene))
    req = urllib.request.Request(studio.url + "/api/scene.bin", headers={"Accept-Encoding": "gzip"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        blob = r.read()
        if r.headers.get("Content-Encoding") != "gzip":
            blob = gzip.compress(blob)
    (out / "scene.bin.gz").write_bytes(blob)
    tex = out / "tex"
    tex.mkdir(exist_ok=True)
    for t in sorted({g["texture"] for g in scene["geoms"] if g.get("texture", -1) >= 0}):
        save_texture(Image.open(io.BytesIO(studio.raw(f"/api/texture/{t}.png"))), tex / f"{t}.webp")
    save_poses(studio.get("/api/poses"), out)

    specs = [s for s in studio.get("/api/specs")["specs"] if not s.get("error")]
    specs = sorted((s for s in specs if s["id"] in CAMERAS), key=lambda s: CAMERAS.index(s["id"]))
    for s in specs:
        s["outputs_depth"] = outputs_depth(s["id"])
    # the demo's recorded cameras, then the robot's own (e.g. its wrist); not the free robotview
    cams = list(dict.fromkeys(run["cameras"] + info["cameras"]))
    cams = [c for c in cams if c in state["cameras"] and c != "robot0_robotview"]
    streams: dict[str, dict] = {}
    meta_path = out / "meta.json"
    if meta_path.exists():
        streams = json.loads(meta_path.read_text()).get("streams", {})
    # PSF first for every camera (the page is usable early), then the lens-ray methods
    jobs = [(cam, None, "psf") for cam in cams]
    jobs += [(WRIST, spec, m) for m in METHODS for spec in specs if WRIST in cams]
    for cam, spec, method in jobs:
        base = state["cameras"][cam]["config"]
        cid = "pinhole" if spec is None else spec["id"]
        if only and cid not in only:
            continue
        key = cid if method == "psf" else f"{cid}|{method}"  # stream key in meta.json
        d = out / cam / (cid.replace("/", "--") + ("" if method == "psf" else f"--{method}"))
        if (d / "image.mp4").exists() and key in streams.get(cam, {}):
            continue
        cfg = {**base, "mode": "ideal"} if spec is None else {
            **base, "mode": "cameratwin", "spec": spec["path"], "render": method}  # fmt: skip
        if method != "psf" and spec["width"] > LENS_RAY_MAX_WIDTH:
            cfg["width"] = LENS_RAY_MAX_WIDTH
            cfg["height"] = round(spec["height"] * LENS_RAY_MAX_WIDTH / spec["width"] / 2) * 2
        t0 = time.time()
        studio.post("/api/camera", {"camera": cam, "config": cfg, "wait": True})
        d.mkdir(parents=True, exist_ok=True)
        vids: dict[str, Video] = {}
        psnr, twin = [], None
        for f in range(n):
            j = studio.get(f"/api/frame?camera={cam}&frame={f}&gain=10")
            size = out_size(j["resolution"])
            is_twin = j["mode"] == "cameratwin"
            if not vids:
                vids["image"] = Video(d / "image.mp4", size, CRF["image"])
                if is_twin:
                    vids["diff"] = Video(d / "diff.mp4", size, CRF["diff"])
                if spec is not None and spec.get("outputs_depth"):
                    vids["depth"] = Video(d / "depth.mp4", size, CRF["depth"])
                twin = j.get("twin")
            im = j["images"]
            vids["image"].add(decode(im["cameratwin"] if is_twin else im["ideal"]))
            if "diff" in vids:
                vids["diff"].add(decode(im["diff"]))
            if "depth" in vids:
                vids["depth"].add(decode(im["depth"]))
            psnr.append(round(j["metrics"]["psnr_full_db"], 2) if is_twin else None)
        for v in vids.values():
            v.close()
        streams.setdefault(cam, {})[key] = {
            "dir": str(d.relative_to(out)),
            "method": method if spec is not None else None,
            "views": sorted(vids),
            "size": list(size),
            "resolution": j["resolution"],
            "twin": twin,
            "psnr_db": psnr if spec is not None else None,
        }
        meta = {
            "id": sc["id"],
            "simulator": sc["simulator"],
            "run": run,
            "robot": robot,
            "robot_info": info,
            "cameras": cams,
            "wrist": WRIST,
            "methods": list(METHODS),
            "num_frames": n,
            "fps": FPS,
            "configs": [{"id": "pinhole", "label": "Pinhole"}]
            + [
                {
                    **spec_facts(s),
                    "label": label(s),
                    "outputs_depth": bool(s.get("outputs_depth")),
                }
                for s in specs
            ],
            "streams": streams,
        }
        meta_path.write_text(json.dumps(meta, indent=1))
        print(f"  {cam:24s} {key:46s} {time.time() - t0:6.1f} s", flush=True)


def outputs_depth(camera_id: str) -> bool:
    """Whether the real camera delivers depth (its catalog spec's ``outputs.depth``)."""
    from twinrobo import CatalogRegistry

    return CatalogRegistry().load(camera_id).outputs.depth


def label(s: dict) -> str:
    return " ".join(x for x in (s.get("manufacturer"), s.get("product")) if x) or s["id"]


def save_texture(img: Image.Image, path: Path) -> None:
    """A 3D-panel texture as WebP, longest side at most `TEXTURE_MAX`."""
    img = img.convert("RGBA" if img.mode in ("RGBA", "LA") else "RGB")
    if max(img.size) > TEXTURE_MAX:
        s = TEXTURE_MAX / max(img.size)
        img = img.resize(
            (max(1, round(img.width * s)), max(1, round(img.height * s))), Image.LANCZOS
        )
    img.save(path, "WEBP", quality=85, method=6)


def save_poses(poses: dict, out: Path) -> None:
    """Per-frame poses: counts in ``poses.json``, the arrays as float32 in ``poses.bin``."""
    import numpy as np

    arrays = [np.asarray(poses[k], dtype="<f4") for k in POSE_ARRAYS]
    head = {k: v for k, v in poses.items() if k not in POSE_ARRAYS}
    head["arrays"] = {k: [int(sum(a.size for a in arrays[:i])), int(arrays[i].size)]
                      for i, k in enumerate(POSE_ARRAYS)}  # fmt: skip
    (out / "poses.bin").write_bytes(b"".join(a.tobytes() for a in arrays))
    (out / "poses.json").write_text(json.dumps(head))


def reencode(f: Path, crf: int) -> None:
    tmp = f.with_suffix(".tmp.mp4")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(f), "-c:v", "libx264", "-preset",
                    "slow", "-crf", str(crf), "-g", "10", "-pix_fmt", "yuv420p", "-movflags",
                    "+faststart", str(tmp)], check=True)  # fmt: skip
    tmp.replace(f)


def encoded_crf(f: Path) -> int | None:
    """The CRF x264 recorded in a video's header (its settings string), if any."""
    import re

    m = re.search(rb"crf=(\d+)\.", f.read_bytes()[:200_000])
    return int(m.group(1)) if m else None


def compact(scene_dir: Path) -> None:
    """Bring a scene baked by an older version to the current formats (in place)."""
    for png in sorted((scene_dir / "tex").glob("*.png")):
        save_texture(Image.open(png), png.with_suffix(".webp"))
        png.unlink()
    poses = json.loads((scene_dir / "poses.json").read_text())
    if "arrays" not in poses:
        save_poses(poses, scene_dir)
    meta = json.loads((scene_dir / "meta.json").read_text())
    for cam in meta["streams"].values():
        for st in cam.values():
            for view in ("diff", "depth"):
                f = scene_dir / st["dir"] / f"{view}.mp4"
                if f.exists() and encoded_crf(f) != CRF[view]:
                    reencode(f, CRF[view])
    print(f"compacted {scene_dir.name}")


def write_manifest() -> None:
    scenes = []
    for key in SCENES:
        p = DATA / SCENES[key]["id"] / "meta.json"
        if p.exists():
            m = json.loads(p.read_text())
            scenes.append(
                {k: m[k] for k in ("id", "simulator", "run", "robot", "num_frames", "fps")}
            )
    (DATA / "manifest.json").write_text(json.dumps({"scenes": scenes}, indent=1))
    size = sum(f.stat().st_size for f in DATA.rglob("*") if f.is_file())
    print(f"manifest: {len(scenes)} scenes, {size / 1e6:.1f} MB in {DATA}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--studio", default="http://127.0.0.1:8791", help="a running TwinRobo Studio")
    ap.add_argument("--scene", choices=sorted(SCENES), action="append")
    ap.add_argument("--config", action="append", help="only these camera configs (debugging)")
    ap.add_argument("--manifest", action="store_true", help="write site/data/manifest.json")
    ap.add_argument("--compact", action="store_true", help="convert older bakes to the formats")
    args = ap.parse_args()
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg (with libx264) is needed")
    for key in args.scene or []:
        bake_scene(Studio(args.studio), key, args.config)
    if args.compact:
        for key in SCENES:
            if (DATA / SCENES[key]["id"] / "meta.json").exists():
                compact(DATA / SCENES[key]["id"])
    if args.manifest or args.scene:
        write_manifest()


if __name__ == "__main__":
    main()
