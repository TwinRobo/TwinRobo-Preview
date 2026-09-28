# TwinRobo Preview

A static, browser-only preview of [TwinRobo Studio](https://github.com/TwinRobo/TwinRobo-Studio):
robot demos replayed through real cameras' optics. Pick a camera for the robot's
wrist (the simulator's pinhole or a catalog camera such as the RealSense D455,
ZED X or Logitech C920) and a rendering method, press **Play**, and the replay
plays next to the 3D scene with every camera's frustum.

Everything is precomputed, so the site is plain files: host `site/` anywhere
(GitHub Pages is set up in `.github/workflows/pages.yml`).

- **Scenes:** one LIBERO-Spatial demo (Franka Panda, *pick up the black bowl next
  to the cookie box and place it on the plate*) and one RoboCasa demo (Panda on
  an Omron base, *pick the cereal from the counter and place it in the cabinet*).
- **Cameras:** the robot's wrist camera through four catalog cameras (Intel
  RealSense D455 RGB, Logitech C920, Luxonis OAK-D mono, Stereolabs ZED X / X Mini
  2.2 mm) or the simulator's pinhole, each with TwinRobo's three rendering methods
  (PSF, pupil raster, ray cast); the other views (agent views) are pinholes.
- **Views:** the image, the difference against the pinhole (catalog cameras)
  and depth (cameras that output depth: the RealSense D455).

Compared to the Studio, the preview only replays: there are no rigs, custom
cameras, camera-config editing, lens-ray settings, or capture.

## Run locally

```bash
python -m http.server 8793 --directory site   # open http://127.0.0.1:8793
```

The page needs a browser with WebGL and H.264 video (any current Chrome, Edge,
Safari or Firefox), and internet for three.js (loaded from jsDelivr).

## How it works

Pressing Play buffers first: the page downloads, in full, the video of every
camera on screen, then plays them together. The main view is the clock the
timeline, the 3D scene and the thumbnails follow. Each stream is one H.264 video
of the whole demo at 20 fps (the demos' control rate).

```
site/
  index.html                  the page (three.js scene, videos, timeline)
  data/manifest.json          the scenes
  data/<scene>/meta.json      cameras, camera configs and their streams
  data/<scene>/scene.json, scene.bin.gz, tex/*.webp, poses.json + poses.bin   the 3D scene
  data/<scene>/<camera>/<config>/{image,diff,depth}.mp4     the replays
```

## Regenerating the data

`tools/bake.py` records what a running TwinRobo Studio renders, through its
HTTP API, so the preview matches the Studio exactly. It needs the Studio (and
its datasets) and `ffmpeg` with libx264.

```bash
# a Studio to bake from (a separate port leaves your own session alone)
python -m twinrobo_viewer --port 8791 --demos <libero demos> --robocasa <robocasa demos>
python tools/bake.py --studio http://127.0.0.1:8791 --scene libero
python tools/bake.py --studio http://127.0.0.1:8791 --scene robocasa
```

Bake one scene at a time on a 24 GB GPU: the lens-ray methods fill it (sensors
wider than 1920 px are rendered at 1920 px for them). Streams already on disk are
skipped, so an interrupted bake resumes; delete a scene's folder to bake it
again. `python tools/bake.py --compact` converts data baked by an older version
to the current formats (WebP textures, binary poses, video quality). The scenes
are listed at the top of `tools/bake.py`.
