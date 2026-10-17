// The camera's optics in the browser: the image at full resolution, the PSF blur at a low one.
//
// For one eye, each frame:
//   1. the splat scene through a pinhole wide enough to cover the whole sensor (Spark, to a texture,
//      at the output resolution);
//   2. depth: the coarse scene mesh through the same pinhole (metres, low resolution);
//   3. pass A, low resolution (the "PSF resolution"), per sensor pixel: where the lens sends it
//      (distortion map), its colour and depth there, and the colour weighted by the PSF basis
//      weights for that field position and depth (bilinear over the bank's field grid, tent in
//      disparity between depth layers); also the PSF's centre weight;
//   4. pass B, low resolution: gather the weighted colours of the neighbours through the basis
//      kernels (scatter-as-gather: each source pixel spreads light with its own PSF);
//   5. pass C, full resolution: the low-resolution blurred image, plus the full-resolution detail
//      the low resolution cannot hold (warped image minus its low-resolution version) scaled by the
//      PSF's centre weight: kept where the lens is sharp, removed where its blur is wide.
// The PSF bank comes from TwinRobo (DeepLens), resampled to the low resolution and compressed to a
// few basis kernels (export_cameras.py).
import * as THREE from "three";

const f32 = (b64) => new Float32Array(Uint8Array.from(atob(b64), (c) => c.charCodeAt(0)).buffer);

function dataTex(data, w, h) {
  const t = new THREE.DataTexture(data, w, h, THREE.RGBAFormat, THREE.FloatType);
  t.minFilter = t.magFilter = THREE.NearestFilter;
  t.needsUpdate = true;
  return t;
}

const QUAD_VS = /* glsl */ `in vec3 position; void main() { gl_Position = vec4(position.xy, 0.0, 1.0); }`;

const DEPTH_VS = /* glsl */ `varying float vz;
void main() { vec4 p = modelViewMatrix * vec4(position, 1.0); vz = -p.z; gl_Position = projectionMatrix * p; }`;
const DEPTH_FS = /* glsl */ `varying float vz; void main() { gl_FragColor = vec4(vz, 0.0, 0.0, 1.0); }`;

const COMMON = /* glsl */ `precision highp float;
uniform sampler2D tMap;
uniform vec2 mapSize;
vec2 mapAt(vec2 s) {  // s: sensor position 0..1 (y down) -> render uv
  vec2 g = s * (mapSize - 1.0);
  ivec2 i = ivec2(clamp(floor(g), vec2(0.0), mapSize - 2.0));
  vec2 f = g - vec2(i);
  vec2 a = texelFetch(tMap, i, 0).xy, b = texelFetch(tMap, i + ivec2(1, 0), 0).xy;
  vec2 c = texelFetch(tMap, i + ivec2(0, 1), 0).xy, d = texelFetch(tMap, i + ivec2(1, 1), 0).xy;
  vec2 xy = mix(mix(a, b, f.x), mix(c, d, f.x), f.y);  // -1..1, y down
  return vec2(0.5 + 0.5 * xy.x, 0.5 - 0.5 * xy.y);
}
bool outside(vec2 uv) { return uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0; }
uniform float decodeIn;  // 1: the scene texture holds sRGB values; 0: the GPU decodes it to linear already
vec3 toLinear(vec3 c) { return decodeIn > 0.5 ? pow(c, vec3(2.2)) : c; }
vec3 toSRGB(vec3 c) { return pow(max(c, 0.0), vec3(1.0 / 2.2)); }
`;

// pass A (low resolution): a_c = PSF weights x colour for c = R, G, B; plus colour and centre weight
const PASS_A = COMMON + /* glsl */ `
layout(location = 0) out vec4 outR;
layout(location = 1) out vec4 outG;
layout(location = 2) out vec4 outB;
layout(location = 3) out vec4 outS;  // the unblurred colour (linear) and the PSF's centre weight
uniform sampler2D tColor, tDepth, tW;
uniform vec2 lowSize, grid;  // grid = (Gw, Gh)
uniform vec4 bCenter;  // the basis kernels' centre taps
uniform float layers, channels;
uniform float disp[16];  // 1 / depth of each layer, far -> near (0 = infinity)
vec4 wAt(int layer, ivec2 node, int c) {  // texture: x = node.x + Gw * layer, y = node.y + Gh * c
  return texelFetch(tW, ivec2(node.x + int(grid.x) * layer, node.y + int(grid.y) * c), 0);
}
vec4 weights(vec2 s, float z, int c) {
  float dz = z > 0.0 ? 1.0 / z : 0.0;
  int L = int(layers);
  int l0 = 0; float t = 0.0;
  if (L == 1 || dz <= disp[0]) { l0 = 0; t = 0.0; }
  else if (dz >= disp[L - 1]) { l0 = L - 2; t = 1.0; }
  else { for (int l = 0; l < 15; l++) { if (l < L - 1 && dz >= disp[l] && dz < disp[l + 1]) { l0 = l; t = (dz - disp[l]) / (disp[l + 1] - disp[l]); } } }
  vec2 g = s * (grid - 1.0);
  ivec2 i = ivec2(clamp(floor(g), vec2(0.0), max(grid - 2.0, vec2(0.0))));
  vec2 f = clamp(g - vec2(i), 0.0, 1.0);
  ivec2 i1 = ivec2(min(vec2(i) + 1.0, grid - 1.0));
  vec4 acc = vec4(0.0);
  for (int k = 0; k < 2; k++) {
    int l = min(l0 + k, L - 1); float wl = k == 0 ? 1.0 - t : t;
    vec4 w = mix(mix(wAt(l, i, c), wAt(l, ivec2(i1.x, i.y), c), f.x), mix(wAt(l, ivec2(i.x, i1.y), c), wAt(l, i1, c), f.x), f.y);
    acc += wl * w;
  }
  return acc;
}
void main() {
  vec2 s = vec2(gl_FragCoord.x / lowSize.x, 1.0 - gl_FragCoord.y / lowSize.y);
  vec2 uv = mapAt(s);
  vec3 col = outside(uv) ? vec3(0.0) : toLinear(texture(tColor, uv).rgb);
  float z = texture(tDepth, uv).r;
  int cg = channels > 1.5 ? 1 : 0, cb = channels > 1.5 ? 2 : 0;
  vec4 wr = weights(s, z, 0), wg = weights(s, z, cg), wb = weights(s, z, cb);
  outR = wr * col.r; outG = wg * col.g; outB = wb * col.b;
  float centre = (dot(wr, bCenter) + dot(wg, bCenter) + dot(wb, bCenter)) / 3.0;
  outS = vec4(col, clamp(centre, 0.0, 1.0));
}`;

// pass B (low resolution): gather through the basis kernels; N = kernel size (recompiled per camera)
const PASS_B = /* glsl */ `precision highp float;
out vec4 fragOut;
uniform sampler2D tR, tG, tB, tBasis;
uniform vec2 lowSize;
void main() {
  ivec2 x = ivec2(gl_FragCoord.xy);
  ivec2 lim = ivec2(lowSize) - 1;
  vec3 col = vec3(0.0);
  for (int j = 0; j < N; j++) for (int i = 0; i < N; i++) {
    // kernel tap (i, j) carries light from a source to (i - c, j - c), rows down; this pixel's
    // source is x minus that offset (GL rows go up)
    ivec2 o = ivec2(i - N / 2, -(j - N / 2));
    ivec2 sx = clamp(x - o, ivec2(0), lim);
    vec4 b = texelFetch(tBasis, ivec2(i, j), 0);
    col += vec3(dot(b, texelFetch(tR, sx, 0)), dot(b, texelFetch(tG, sx, 0)), dot(b, texelFetch(tB, sx, 0)));
  }
  fragOut = vec4(col, 1.0);  // linear
}`;

// pass C (full resolution): low-resolution blur + full-resolution detail scaled by the PSF's sharpness
const PASS_C = COMMON + /* glsl */ `
layout(location = 0) out vec4 outColor;  // sRGB
layout(location = 1) out vec4 outDepth;  // metres along the camera axis
uniform sampler2D tColor, tDepth, tBlur, tLow;
uniform vec2 outSize;
uniform float hasBlur, mono;
void main() {
  vec2 p = gl_FragCoord.xy / outSize;
  vec2 s = vec2(p.x, 1.0 - p.y);
  vec2 uv = mapAt(s);
  vec3 sharp = outside(uv) ? vec3(0.0) : toLinear(texture(tColor, uv).rgb);
  vec3 col = sharp;
  if (hasBlur > 0.5) {
    vec4 low = texture(tLow, p);  // unblurred low-resolution colour, PSF centre weight
    col = texture(tBlur, p).rgb + low.a * (sharp - low.rgb);
  }
  if (mono > 0.5) col = vec3(dot(col, vec3(0.2126, 0.7152, 0.0722)));
  outColor = vec4(toSRGB(col), 1.0);
  outDepth = vec4(outside(uv) ? 0.0 : texture(tDepth, uv).r, 0.0, 0.0, 1.0);
}`;

const SPARKS = new Map();
let SLOTS = 0;

export class EyeOptics {
  constructor(renderer, SparkRenderer, depthScene) {
    this.slot = SLOTS++;  // each eye keeps its own splat renderers (their sorts follow different views)
    this.renderer = renderer;
    this.SparkRenderer = SparkRenderer;
    this.depthScene = depthScene;
    this.camera = new THREE.PerspectiveCamera(70, 4 / 3, 0.02, 50);
    this.quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2));
    this.quadScene = new THREE.Scene();
    this.quadScene.add(this.quad);
    this.quadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.depthMat = new THREE.ShaderMaterial({ vertexShader: DEPTH_VS, fragmentShader: DEPTH_FS, side: THREE.DoubleSide });
  }

  // cam: an exported camera; psfWidth: the PSF resolution (a level of its blur: 320 or 640 px wide);
  // outWidth: the image width (full resolution)
  setCamera(cam, psfWidth, outWidth) {
    this.dispose();
    this.cam = cam;
    const [W, H] = cam.sensor;
    const size = (w) => [Math.max(2, Math.round(w)), Math.max(2, Math.round((w * H) / W))];
    this.out = size(outWidth);
    const lv = cam.blur?.levels?.[String(psfWidth)];
    this.hasBlur = !!lv;
    this.low = size(lv ? psfWidth : 2);
    const r = cam.render;
    this.camera.fov = r.vfov_deg;
    this.camera.aspect = r.tan_x / r.tan_y;
    this.camera.updateProjectionMatrix();
    // the pinhole render: about one render pixel per output pixel where the lens magnifies most
    const scale = Math.max(1, (r.tan_y / (H / 2)) * cam.intrinsics.fy);
    this.renderSize = [Math.round(this.out[1] * 1.05 * scale * this.camera.aspect), Math.round(this.out[1] * 1.05 * scale)];
    // Spark renderers are kept and shared per size (disposing one kills its sort worker mid-flight)
    const key = `${this.renderSize[0]}x${this.renderSize[1]}:${this.slot}`;
    if (!SPARKS.has(key)) SPARKS.set(key, new this.SparkRenderer({ renderer: this.renderer, target: { width: this.renderSize[0], height: this.renderSize[1] } }));
    this.spark = SPARKS.get(key);
    const tgt = (w, h, n = 1, filter = THREE.LinearFilter) => {
      const t = new THREE.WebGLRenderTarget(w, h, { type: THREE.HalfFloatType, count: n, minFilter: filter, magFilter: filter });
      t.textures.forEach((x) => (x.minFilter = x.magFilter = filter));
      return t;
    };
    const ds = this.hasBlur ? this.low[0] / this.out[0] : 0.5;
    this.depthTarget = tgt(Math.max(2, Math.round(this.renderSize[0] * ds)), Math.max(2, Math.round(this.renderSize[1] * ds)));
    this.aTarget = tgt(...this.low, 4, THREE.NearestFilter);
    this.aTarget.textures[3].minFilter = this.aTarget.textures[3].magFilter = THREE.LinearFilter;
    this.blurTarget = tgt(...this.low);
    this.outTarget = tgt(...this.out, 2);

    const m = cam.map;
    const md = f32(m.data);  // [h, w, 2]
    const mrgba = new Float32Array(m.size[0] * m.size[1] * 4);
    for (let k = 0; k < m.size[0] * m.size[1]; k++) { mrgba[4 * k] = md[2 * k]; mrgba[4 * k + 1] = md[2 * k + 1]; }
    this.mapTex = dataTex(mrgba, m.size[0], m.size[1]);

    let N = 1, basisTex = dataTex(new Float32Array([1, 0, 0, 0]), 1, 1), wTex = dataTex(new Float32Array(4), 1, 1);
    const disp = new Float32Array(16), bCenter = new THREE.Vector4(1, 0, 0, 0);
    let grid = [1, 1], layers = 1, channels = 1;
    if (lv) {
      N = lv.size;
      const rk = lv.rank, B = f32(lv.basis);  // [r, N, N]
      const bt = new Float32Array(N * N * 4);
      for (let k = 0; k < rk; k++) for (let p = 0; p < N * N; p++) bt[4 * p + k] = B[k * N * N + p];
      basisTex = dataTex(bt, N, N);
      const c = (N >> 1) * N + (N >> 1);
      bCenter.set(bt[4 * c], bt[4 * c + 1], bt[4 * c + 2], bt[4 * c + 3]);
      [grid, layers, channels] = [cam.blur.grid, cam.blur.depths_m.length, cam.blur.channels];
      const [Gw, Gh] = grid;
      const Wt = f32(lv.weights);  // [L, Gh, Gw, C, r]
      const wt = new Float32Array(Gw * layers * Gh * channels * 4);
      for (let l = 0; l < layers; l++) for (let y = 0; y < Gh; y++) for (let x = 0; x < Gw; x++) for (let ch = 0; ch < channels; ch++) {
        const src = (((l * Gh + y) * Gw + x) * channels + ch) * rk;
        const dst = ((y + Gh * ch) * (Gw * layers) + (x + Gw * l)) * 4;
        for (let k = 0; k < rk; k++) wt[dst + k] = Wt[src + k];
      }
      wTex = dataTex(wt, Gw * layers, Gh * channels);
      cam.blur.depths_m.forEach((d, i) => (disp[i] = d != null && d > 0 ? 1 / d : 0));  // null: infinity
    }
    this.textures = [basisTex, wTex];
    const common = { tMap: { value: this.mapTex }, mapSize: { value: new THREE.Vector2(...m.size) }, decodeIn: { value: 1 } };
    const raw = (fs, uniforms) => new THREE.RawShaderMaterial({ glslVersion: THREE.GLSL3, vertexShader: QUAD_VS, fragmentShader: fs, uniforms: { ...common, ...uniforms } });
    this.matA = raw(PASS_A, {
      tColor: { value: null }, tDepth: { value: this.depthTarget.texture }, tW: { value: wTex },
      lowSize: { value: new THREE.Vector2(...this.low) }, grid: { value: new THREE.Vector2(...grid) },
      layers: { value: layers }, channels: { value: channels }, disp: { value: Array.from(disp) }, bCenter: { value: bCenter },
    });
    this.matB = raw(PASS_B.replace("precision highp float;", `precision highp float;\n#define N ${N}`), {
      tR: { value: this.aTarget.textures[0] }, tG: { value: this.aTarget.textures[1] }, tB: { value: this.aTarget.textures[2] },
      tBasis: { value: basisTex }, lowSize: { value: new THREE.Vector2(...this.low) },
    });
    this.matC = raw(PASS_C, {
      tColor: { value: null }, tDepth: { value: this.depthTarget.texture }, tBlur: { value: this.blurTarget.texture },
      tLow: { value: this.aTarget.textures[3] }, outSize: { value: new THREE.Vector2(...this.out) },
      hasBlur: { value: this.hasBlur ? 1 : 0 }, mono: { value: cam.mono ? 1 : 0 },
    });
  }

  // One frame for the camera pose `poseMatrix` (an eye's world matrix; three.js camera frame).
  render(scene, poseMatrix, hide = []) {
    const r = this.renderer;
    poseMatrix.decompose(this.camera.position, this.camera.quaternion, this.camera.scale);
    this.camera.updateMatrixWorld(true);
    const vis = hide.map((o) => o.visible);
    hide.forEach((o) => (o.visible = false));
    const color = this.spark.renderTarget({ scene, camera: this.camera });
    const decode = color.texture.colorSpace === THREE.SRGBColorSpace ? 0 : 1;  // sRGB textures are decoded on sampling
    this.matA.uniforms.decodeIn.value = this.matC.uniforms.decodeIn.value = decode;
    hide.forEach((o, i) => (o.visible = vis[i]));
    const prevTarget = r.getRenderTarget();
    const pass = (mat, target) => { this.quad.material = mat; r.setRenderTarget(target); r.render(this.quadScene, this.quadCam); };
    // depth (low resolution)
    this.depthScene.overrideMaterial = this.depthMat;
    const cc = r.getClearColor(new THREE.Color()), ca = r.getClearAlpha();
    r.setRenderTarget(this.depthTarget);
    r.setClearColor(0x000000, 0); r.clear();
    r.render(this.depthScene, this.camera);
    r.setClearColor(cc, ca);
    this.depthScene.overrideMaterial = null;
    if (this.hasBlur) {
      this.matA.uniforms.tColor.value = color.texture;
      pass(this.matA, this.aTarget);
      pass(this.matB, this.blurTarget);
    }
    this.matC.uniforms.tColor.value = color.texture;
    pass(this.matC, this.outTarget);
    r.setRenderTarget(prevTarget);
    return this.outTarget.textures[0];
  }

  async read() {  // { rgba: RGBA8 rows, depth: Float32 metres; both bottom row first }
    const [w, h] = this.out;
    const half = new Uint16Array(w * h * 4);
    await this.renderer.readRenderTargetPixelsAsync(this.outTarget, 0, 0, w, h, half, 0, 0);
    const rgba = new Uint8Array(w * h * 4);
    for (let k = 0; k < half.length; k++) rgba[k] = Math.max(0, Math.min(255, Math.round(THREE.DataUtils.fromHalfFloat(half[k]) * 255)));
    await this.renderer.readRenderTargetPixelsAsync(this.outTarget, 0, 0, w, h, half, 0, 1);
    const depth = new Float32Array(w * h);
    for (let k = 0; k < w * h; k++) depth[k] = THREE.DataUtils.fromHalfFloat(half[4 * k]);
    return { rgba, depth };
  }

  dispose() {
    for (const k of ["depthTarget", "aTarget", "blurTarget", "outTarget", "mapTex", "matA", "matB", "matC"]) this[k]?.dispose?.();
    this.textures?.forEach((t) => t.dispose());
  }
}
