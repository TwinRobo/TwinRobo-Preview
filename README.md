# TwinRobo Preview

Static, browser-only previews of TwinRobo, in two apps:

| Path | App |
|---|---|
| `/` | a landing page linking the two |
| `/vla` | **Demo replay**: robot demos replayed through real cameras' optics (below) |
| `/capture` | **Capture**: drive a robot through a real kitchen and record what a catalog camera sees (see [Capture](#capture)) |

## Demo replay (`/vla`)

A preview of [TwinRobo Studio](https://github.com/TwinRobo/TwinRobo-Studio):
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
  2.2 mm) or the simulator's pinhole, each with TwinRobo's two rendering methods
  (PSF, ray cast); the other views (agent views) are pinholes.
- **Views:** the image and the difference against the pinhole (catalog cameras).
  There is no depth view: TwinRobo simulates what cameras image, not yet what
  depth cameras measure (see the SDK's roadmap).

Compared to the Studio, the preview only replays: there are no rigs, custom
cameras, camera-config editing, lens-ray settings, or capture.

## Run locally

```bash
python -m http.server 8793 --directory site   # open http://127.0.0.1:8793 (/vla, /capture)
```

The page needs a browser with WebGL and H.264 video (any current Chrome, Edge,
Safari or Firefox), and internet for three.js (loaded from jsDelivr).

## How it works

Pressing Play buffers first: the page downloads, in full, the video of every
camera on screen, then plays them together. The main view is the clock the
timeline, the 3D scene and the thumbnails follow. Each stream is one H.264 video
of the whole demo at 20 fps (the demos' control rate).

```
site/vla/
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

## Capture (`/capture`)

A real New York kitchen (the Zip-NeRF "nyc" capture by Barron et al., CC BY 4.0),
reconstructed as a Gaussian splat and shown with [Spark](https://sparkjs.dev/). Drive a
robot with W A S D, turn it and tilt its camera with the mouse, and record frames:
images, depth and camera poses, downloaded as a zip. Pick any camera of the TwinRobo
catalog, including the stereo modules (left, right or both eyes).

The camera's optics run in the browser. The image is rendered at full resolution and
warped by the lens distortion; the PSF blur, which varies with field position and depth,
is computed at a low resolution (320 or 640 px wide) and combined with the full-resolution
detail, scaled by how sharp the lens is at each pixel (`site/capture/optics.js`). Depth
comes from a coarse mesh of the scene. Captures stay in the browser tab until downloaded;
the page lists them for review and deletion.

```
site/capture/
  index.html, optics.js       the page and the camera optics (WebGL 2 shaders)
  kitchen.spz                 the kitchen splat (SPZ, metric after scene.json's transform)
  depth.ply                   coarse scene mesh for depth (TSDF of the splat's depth)
  occupancy.png, scene.json   where the robot can drive; transform, scale and start pose
  cameras/index.json, *.json  the catalog cameras: field of view, distortion map, PSF basis
```

### Rebuilding the scene and cameras

The scripts in `tools/capture/` make every file above. Training needs a CUDA GPU and
[gsplat](https://github.com/nerfstudio-project/gsplat) 1.5; the camera export needs
TwinRobo-Core (and DeepLens) for the PSF banks. All of them are seeded.

```bash
cd tools/capture
# 1. the Zip-NeRF nyc scene (undistorted): https://smerf-3d.github.io/ (CC BY 4.0)
#    -> <data>/nyc/{images,sparse/0}
# 2. a floor plan (up-aligned frame, scene units) to choose the box
python plan.py <splat.ply> <data>/nyc/sparse/0 plan.jpg
# 3. train on the photos taken in the kitchen box (MCMC, anti-aliased, exposure per photo)
python train2.py <data>/nyc images kitchen 30000 -6.6 -1.8 -4.0 1.2 1500000
# 4. crop, then export in metres with the obstacle map (counter box sets the scale: 0.914 m)
python crop_splat.py kitchen/splat.ply <data>/nyc kitchen_crop -6.6 -1.8 -4.0 1.2
START="-0.07,-0.96,129,-0.25" python export.py kitchen_crop/crop.ply <data>/nyc out -5.0 -4.2 -1.6 -1.15
python depth_mesh.py kitchen_crop/crop.ply <data>/nyc out 0.05
python ply2spz.py out/kitchen.ply out/kitchen.spz 3     # then point scene.json's "splat" at kitchen.spz
# 5. the catalog cameras (TwinRobo-Core installed)
python export_cameras.py out/cameras
```
