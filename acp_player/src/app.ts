// Client-side ACP reference player: one binary clip payload, local 50 Hz playback.
import * as THREE from './vendor/three.module.js';
import { Select } from './ui.js';

THREE.ColorManagement.enabled = false;

type Vec3 = [number, number, number];
type TypedArray = Float32Array | Int32Array | Uint32Array | Uint16Array | Uint8Array | Int8Array;

interface ArrayMeta { offset: number; dtype: string; shape: number[] }
interface ContactEditInfo { edited?: boolean; saved_at?: string; changed?: number }
interface TrackHeader {
  names: string[];
  source: string;
  provenance: string;
  frames: number;
  edit?: ContactEditInfo;
}
interface TerrainHeader { dataset: string; id: number; name: string; url: string }
interface ClipHeader {
  dataset: string;
  clip: string;
  group: string;
  fps: number;
  frames: number;
  bodies: string[];
  contact_body_index: Record<string, number>;
  primary: 'reference' | 'physics' | null;
  tracks: Partial<Record<'reference' | 'physics', TrackHeader>>;
  terrain: TerrainHeader;
  bake_ms: number;
  arrays: Record<string, ArrayMeta>;
}
interface ClipArrays {
  poses: Float32Array;
  [key: string]: TypedArray;
}
interface ClipPayload { header: ClipHeader; arrays: ClipArrays }
interface ClipEntry { name: string; group: string; status: string; note: string }
interface DatasetConfig { name: string; clips: ClipEntry[] }
interface CurationRecord { status?: string; note?: string; time?: string }
interface Config {
  default: string;
  datasets: DatasetConfig[];
  marks: Record<string, string>;
  body_labels: Record<string, string>;
  contact_colors: Record<string, Vec3>;
  contact_links: Record<string, string>;
  robot_url: string;
  grid: number[];
}

interface ElementIds {
  panel: HTMLElement;
  dataset: HTMLElement;
  group: HTMLElement;
  clip: HTMLElement;
  'clip-name': HTMLElement;
  'clip-group': HTMLElement;
  'copy-clip': HTMLButtonElement;
  'prev-clip': HTMLButtonElement;
  'next-clip': HTMLButtonElement;
  filter: HTMLInputElement;
  'load-error': HTMLElement;
  review: HTMLElement;
  note: HTMLTextAreaElement;
  keep: HTMLButtonElement;
  reject: HTMLButtonElement;
  clear: HTMLButtonElement;
  'next-unreviewed': HTMLButtonElement;
  'review-feedback': HTMLElement;
  status: HTMLElement;
  play: HTMLInputElement;
  loop: HTMLInputElement;
  speed: HTMLInputElement;
  'speed-val': HTMLElement;
  prev: HTMLButtonElement;
  next: HTMLButtonElement;
  follow: HTMLInputElement;
  'reset-view': HTMLButtonElement;
  snap: HTMLButtonElement;
  'ov-contacts': HTMLInputElement;
  'ov-physics': HTMLInputElement;
  'ov-targets': HTMLInputElement;
  'ov-trail': HTMLInputElement;
  'legend-colors': HTMLElement;
  'edit-toggle': HTMLButtonElement;
  'edit-undo': HTMLButtonElement;
  'edit-redo': HTMLButtonElement;
  'edit-save': HTMLButtonElement;
  'edit-revert': HTMLButtonElement;
  'edit-clear': HTMLButtonElement;
  'edit-parts': HTMLElement;
  'edit-mark': HTMLButtonElement;
  'edit-erase': HTMLButtonElement;
  'edit-status': HTMLElement;
  'edit-hint': HTMLElement;
  'tl-zoom-out': HTMLButtonElement;
  'tl-zoom-in': HTMLButtonElement;
  'tl-zoom-all': HTMLButtonElement;
  view: HTMLCanvasElement;
  viewport: HTMLElement;
  stage: HTMLElement;
  editbar: HTMLElement;
  'dock-resize': HTMLElement;
  timeline: HTMLElement;
  strip: HTMLCanvasElement;
  cursor: HTMLCanvasElement;
}

const $ = <K extends keyof ElementIds>(id: K): ElementIds[K] => document.getElementById(id) as ElementIds[K];
const clamp = (value: number, low: number, high: number): number => Math.min(high, Math.max(low, value));
const lerp = (a: number, b: number, t: number): number => a + (b - a) * t;

// 界面用中文；服务器返回的 BODY_LABELS 是英文，这里做一层映射。
const BODY_ZH = {
  pelvis: '骨盆',
  left_knee: '左膝', right_knee: '右膝',
  left_foot: '左足', right_foot: '右足',
  left_elbow: '左肘', right_elbow: '右肘',
  left_hand: '左手', right_hand: '右手',
};
const STATUS_ZH = { keep: '已保留', reject: '已剔除', edited: '已编辑', unreviewed: '未审阅' };

function bodyLabel(name: string): string {
  return (BODY_ZH as Record<string, string>)[name]
    || (state.config ? state.config.body_labels[name] : '') || name;
}

const state = {
  config: null as Config | null,
  dataset: null as string | null,
  group: '',                                   // selected group filter, '' = all groups
  entries: new Map<string, ClipEntry>(),       // clip name -> {name, group, status, note}
  clipOptions: [] as string[],
  clip: null as ClipPayload | null,            // {header, arrays}
  clipName: null as string | null,
  loadToken: 0,
  badClips: new Set<string>(),
  noteDrafts: new Map<string, string>(),       // `${dataset}/${clip}` -> draft note
  curation: {} as Record<string, CurationRecord>,
  shortToLink: {} as Record<string, string>,
  playing: true,
  loop: true,
  speed: 1,
  frame: 0,
  lastFrame: -1,
  lastSeconds: -1,
  lastText: 0,
  lastCursor: 0,
  lastPelvis: new THREE.Vector3(),
  follow: false,
  followArmed: false,
  dirty: true,
  cameraDirty: false,
  terrainUrl: null,
  surface: null,
  ground: 0,                                   // terrain height under the clip (world +Z)
  surfaceMesh: null,
  overlays: { contacts: true, physics: true, targets: true, trail: true },
  robotPayload: null,
  robot: null,
  tracks: {} as Record<string, {
    marker: THREE.InstancedMesh;
    targets: THREE.InstancedMesh | null;
    lines: THREE.LineSegments | null;
    shadow: THREE.InstancedMesh | null;        // contact footprint projected on the ground
    drops: THREE.LineSegments | null;          // thin line from contact to its footprint
    names: string[];
    count: number;
  }>,
  contactBodyIndex: {},
  trail: null as THREE.LineSegments | null,
  selects: {} as { dataset: Select; group: Select; clip: Select },
  scrubbing: false,
  timelineHeight: 210,
  dockDrag: null as { y: number; height: number } | null,
  // Visible frame range of the timeline (frame index space, half-cell margins
  // so frame 0 / frames-1 sit on the plot edges when fully zoomed out).
  viewStart: -0.5,
  viewEnd: 0.5,
  pan: null as { x: number; start: number; span: number } | null,
  edit: {
    enabled: false,
    saved: null as Uint8Array | null,      // flags as last loaded / saved on the server
    undo: [] as Uint8Array[],
    redo: [] as Uint8Array[],
    dirty: false,
    savedInfo: null as ContactEditInfo | null,
    body: null as string | null,           // selected contact row (short name)
    stroke: null as { value: number; frame: number; row: number; snapshot: Uint8Array | null } | null,
    anchor: null as { row: number; value: number; from: number; to: number; base: Uint8Array; pushed: boolean } | null,
    segment: null as {
      row: number; mode: 'start' | 'end' | 'move';
      start: number; end: number;      // the segment as it was when the drag began
      origin: number;                  // frame under the pointer at the press
      minStart: number; maxEnd: number; // room left by the neighbouring segments
      base: Uint8Array; pushed: boolean;
    } | null,
    notice: '',
    busy: false,
    drawPending: false,
  },
};

// ---------------------------------------------------------------------------
// Payload decoding
// ---------------------------------------------------------------------------

function parsePayload(buffer: ArrayBuffer): ClipPayload {
  const view = new DataView(buffer);
  const magic = new TextDecoder().decode(new Uint8Array(buffer, 0, 4));
  if (magic !== 'ACPC') throw new Error('bad payload magic');
  const headerLength = view.getUint32(4, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 8, headerLength)));
  const base = 8 + headerLength;
  const arrays: Record<string, TypedArray> = {};
  for (const [name, meta] of Object.entries(header.arrays) as Array<[string, ArrayMeta]>) {
    const count = meta.shape.reduce((a, b) => a * b, 1);
    const dtype = meta.dtype;
    const Ctor = dtype === '<f4' ? Float32Array
      : dtype === '<i4' ? Int32Array
        : dtype === '<u2' ? Uint16Array
          : dtype === '<u4' ? Uint32Array
            : dtype === '|u1' ? Uint8Array
              : dtype === '|i1' ? Int8Array
                : null;
    if (!Ctor) throw new Error(`unsupported dtype ${dtype}`);
    arrays[name] = new (Ctor as new (buffer: ArrayBuffer, byteOffset: number, length: number) => TypedArray)(
      buffer, base + meta.offset, count);
  }
  return { header, arrays: arrays as ClipArrays };
}

async function getBuffer(url: string): Promise<ClipPayload> {
  const response = await fetch(url, { cache: 'no-store' });
  if (!response.ok) {
    let detail = await response.text();
    try { detail = JSON.parse(detail).error || detail; } catch (_) { /* plain text */ }
    throw new Error(detail || `HTTP ${response.status}`);
  }
  return parsePayload(await response.arrayBuffer());
}

async function getJSON(url: string, options?: RequestInit): Promise<any> {
  const response = await fetch(url, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
  return body;
}

// ---------------------------------------------------------------------------
// Terrain surface: exact downward triangle rays, 1 m XY bins (mirrors Terrain.py)
// ---------------------------------------------------------------------------

class Surface {
  a: Float64Array;
  e1: Float64Array;
  e2: Float64Array;
  det: Float64Array;
  bins: Map<number, number[]>;
  count = 0;

  constructor(positions: Float32Array, indices: Uint32Array) {
    const triangles = indices.length / 3;
    this.a = new Float64Array(triangles * 3);
    this.e1 = new Float64Array(triangles * 3);
    this.e2 = new Float64Array(triangles * 3);
    this.det = new Float64Array(triangles);
    this.bins = new Map();
    let kept = 0;
    for (let t = 0; t < triangles; t += 1) {
      const ia = indices[t * 3] * 3;
      const ib = indices[t * 3 + 1] * 3;
      const ic = indices[t * 3 + 2] * 3;
      const ax = positions[ia], ay = positions[ia + 1], az = positions[ia + 2];
      const bx = positions[ib], by = positions[ib + 1], bz = positions[ib + 2];
      const cx = positions[ic], cy = positions[ic + 1], cz = positions[ic + 2];
      const e1x = bx - ax, e1y = by - ay, e1z = bz - az;
      const e2x = cx - ax, e2y = cy - ay, e2z = cz - az;
      const det = e1x * e2y - e1y * e2x;
      if (Math.abs(det) <= 1e-12) continue;
      const k = kept;
      this.a[k * 3] = ax; this.a[k * 3 + 1] = ay; this.a[k * 3 + 2] = az;
      this.e1[k * 3] = e1x; this.e1[k * 3 + 1] = e1y; this.e1[k * 3 + 2] = e1z;
      this.e2[k * 3] = e2x; this.e2[k * 3 + 1] = e2y; this.e2[k * 3 + 2] = e2z;
      this.det[k] = det;
      kept += 1;
      const minX = Math.floor(Math.min(ax, bx, cx));
      const maxX = Math.floor(Math.max(ax, bx, cx));
      const minY = Math.floor(Math.min(ay, by, cy));
      const maxY = Math.floor(Math.max(ay, by, cy));
      for (let x = minX; x <= maxX; x += 1) {
        for (let y = minY; y <= maxY; y += 1) {
          const key = (x + 16384) * 32768 + (y + 16384);
          let list = this.bins.get(key);
          if (!list) { list = []; this.bins.set(key, list); }
          list.push(k);
        }
      }
    }
    this.count = kept;
  }

  heights(points: Float32Array, originZ: number): Float32Array {
    const out = new Float32Array(points.length / 2).fill(NaN);
    for (let p = 0; p < out.length; p += 1) {
      const x = points[p * 2], y = points[p * 2 + 1];
      const key = (Math.floor(x) + 16384) * 32768 + (Math.floor(y) + 16384);
      const candidates = this.bins.get(key);
      if (!candidates) continue;
      let best = -Infinity;
      for (const t of candidates) {
        const o = t * 3;
        const ax = this.a[o], ay = this.a[o + 1], az = this.a[o + 2];
        const e1x = this.e1[o], e1y = this.e1[o + 1], e1z = this.e1[o + 2];
        const e2x = this.e2[o], e2y = this.e2[o + 1], e2z = this.e2[o + 2];
        const det = this.det[t];
        const dx = x - ax, dy = y - ay;
        const u = (dx * e2y - dy * e2x) / det;
        const v = (dy * e1x - dx * e1y) / det;
        if (u < -1e-8 || v < -1e-8 || u + v > 1 + 1e-8) continue;
        const z = az + u * e1z + v * e2z;
        if (z <= originZ && z > best) best = z;
      }
      if (best > -Infinity) out[p] = best;
    }
    return out;
  }
}

// ---------------------------------------------------------------------------
// Scene, lights, camera
// ---------------------------------------------------------------------------

const canvas = $('view');
const viewport = $('viewport');
const timelineEl = $('timeline');

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xffffff);

const camera = new THREE.PerspectiveCamera(45, 16 / 9, 0.05, 400);
camera.up.set(0, 0, 1);

scene.add(new THREE.HemisphereLight(0xffffff, 0xd9dce0, 2.0));
const key = new THREE.DirectionalLight(0xffffff, 1.25);
key.position.set(3, -4, 6);
scene.add(key);
const fill = new THREE.DirectionalLight(0xffffff, 0.45);
fill.position.set(-4, 3, 2);
scene.add(fill);

const robotGroup = new THREE.Group();
scene.add(robotGroup);
const overlayGroup = new THREE.Group();
scene.add(overlayGroup);

const controls = {
  target: new THREE.Vector3(),
  radius: 6,
  azimuth: 0,
  elevation: 0.35,
};

function applyCamera() {
  const ce = Math.cos(controls.elevation);
  camera.position.set(
    controls.target.x + controls.radius * ce * Math.cos(controls.azimuth),
    controls.target.y + controls.radius * ce * Math.sin(controls.azimuth),
    controls.target.z + controls.radius * Math.sin(controls.elevation),
  );
  camera.lookAt(controls.target);
  state.cameraDirty = true;
}

function setView(position, lookAt) {
  controls.target.set(lookAt[0], lookAt[1], lookAt[2]);
  const dx = position[0] - lookAt[0];
  const dy = position[1] - lookAt[1];
  const dz = position[2] - lookAt[2];
  controls.radius = Math.max(0.5, Math.hypot(dx, dy, dz));
  controls.azimuth = Math.atan2(dy, dx);
  controls.elevation = Math.asin(clamp(dz / controls.radius, -0.999, 0.999));
  applyCamera();
}

function resize() {
  const width = viewport.clientWidth;
  const height = viewport.clientHeight;
  if (!width || !height) return;
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  drawTimeline();
  state.cameraDirty = true;
}

new ResizeObserver(resize).observe(viewport);

// --- bottom dock: drag the splitter to trade viewport height for taller rows
const TL_MIN_HEIGHT = 96;
const TL_DEFAULT_HEIGHT = 210;
const TL_HEIGHT_KEY = 'acp.timelineHeight';

function timelineHeightLimit(): number {
  return Math.max(TL_MIN_HEIGHT + 24, $('stage').clientHeight - $('editbar').offsetHeight - 120);
}

function applyTimelineHeight(value: number, persist = true): void {
  const height = Math.round(clamp(value, TL_MIN_HEIGHT, timelineHeightLimit()));
  state.timelineHeight = height;
  timelineEl.style.height = `${height}px`;
  timelineEl.style.flex = `0 0 ${height}px`;
  if (persist) {
    try { localStorage.setItem(TL_HEIGHT_KEY, String(height)); } catch (_) { /* private mode */ }
  }
  resize();
}

function restoreTimelineHeight(): void {
  let stored = NaN;
  try { stored = Number(localStorage.getItem(TL_HEIGHT_KEY)); } catch (_) { /* ignore */ }
  applyTimelineHeight(Number.isFinite(stored) && stored > 0 ? stored : TL_DEFAULT_HEIGHT, false);
}

// Mouse orbit / pan / zoom (world +Z up).
let drag = null;
canvas.addEventListener('contextmenu', (event) => event.preventDefault());
canvas.addEventListener('pointerdown', (event) => {
  canvas.setPointerCapture(event.pointerId);
  drag = { x: event.clientX, y: event.clientY, pan: event.button !== 0 || event.shiftKey };
});
canvas.addEventListener('pointerup', (event) => {
  drag = null;
  if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
});
canvas.addEventListener('pointermove', (event) => {
  if (!drag) return;
  const dx = event.clientX - drag.x;
  const dy = event.clientY - drag.y;
  drag.x = event.clientX;
  drag.y = event.clientY;
  if (drag.pan) {
    const scale = 2 * controls.radius * Math.tan((camera.fov * Math.PI) / 360) / Math.max(canvas.clientHeight, 1);
    const right = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 0);
    const up = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 1);
    controls.target.addScaledVector(right, -dx * scale).addScaledVector(up, dy * scale);
  } else {
    controls.azimuth -= dx * 0.0055;
    controls.elevation = clamp(controls.elevation + dy * 0.0055, -1.45, 1.45);
  }
  applyCamera();
});
canvas.addEventListener('wheel', (event) => {
  event.preventDefault();
  controls.radius = clamp(controls.radius * Math.exp(event.deltaY * 0.0011), 0.25, 200);
  applyCamera();
}, { passive: false });

function headingXY(w, x, y, z) {
  const yaw = Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
  return [Math.cos(yaw), -Math.sin(yaw)];
}

// Mirrors webplay_acp.frame_clip_camera.
function frameClipCamera(basePos, heading, ground, fov, aspect) {
  const n = basePos.length / 3;
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (let i = 0; i < n; i += 1) {
    const x = basePos[i * 3], y = basePos[i * 3 + 1];
    minX = Math.min(minX, x); maxX = Math.max(maxX, x);
    minY = Math.min(minY, y); maxY = Math.max(maxY, y);
  }
  const cx = 0.5 * (minX + maxX);
  const cy = 0.5 * (minY + maxY);
  let sxx = 0, sxy = 0, syy = 0, radius = 0;
  for (let i = 0; i < n; i += 1) {
    const ox = basePos[i * 3] - cx;
    const oy = basePos[i * 3 + 1] - cy;
    sxx += ox * ox; sxy += ox * oy; syy += oy * oy;
    radius = Math.max(radius, Math.hypot(ox, oy));
  }
  const travelX = basePos[(n - 1) * 3] - basePos[0];
  const travelY = basePos[(n - 1) * 3 + 1] - basePos[1];
  let forward;
  if (radius >= 0.5 && Math.hypot(travelX, travelY) >= 0.5) {
    const theta = 0.5 * Math.atan2(2 * sxy, sxx - syy);
    forward = [Math.cos(theta), Math.sin(theta)];
    if (forward[0] * travelX + forward[1] * travelY < 0) forward = [-forward[0], -forward[1]];
  } else {
    forward = heading;
  }
  const norm = Math.hypot(forward[0], forward[1]) || 1;
  forward = [forward[0] / norm, forward[1] / norm];
  const sideAngle = (60 * Math.PI) / 180;
  const ca = Math.cos(sideAngle), sa = Math.sin(sideAngle);
  const side = [ca * forward[0] - sa * forward[1], sa * forward[0] + ca * forward[1]];
  const elevation = (20 * Math.PI) / 180;
  const eye = [Math.cos(elevation) * side[0], Math.cos(elevation) * side[1], Math.sin(elevation)];
  const lookAt = [cx, cy, ground + 0.6];
  const extent = radius + 1.0;
  const samples = [];
  for (let i = 0; i < n; i += 1) samples.push([basePos[i * 3], basePos[i * 3 + 1]]);
  for (let a = 0; a < 64; a += 1) {
    const angle = (a / 64) * Math.PI * 2;
    samples.push([cx + extent * Math.cos(angle), cy + extent * Math.sin(angle)]);
  }
  const right = [-eye[1], eye[0], 0];
  const rightNorm = Math.hypot(right[0], right[1]) || 1;
  right[0] /= rightNorm; right[1] /= rightNorm;
  const up = [
    eye[1] * right[2] - eye[2] * right[1],
    eye[2] * right[0] - eye[0] * right[2],
    eye[0] * right[1] - eye[1] * right[0],
  ];
  const tanV = Math.tan(0.5 * fov);
  const tanH = tanV * aspect;
  let distance = 0;
  for (const [sx, sy] of samples) {
    for (const sz of [ground, ground + 1.8]) {
      const qx = sx - lookAt[0], qy = sy - lookAt[1], qz = sz - lookAt[2];
      const along = qx * eye[0] + qy * eye[1] + qz * eye[2];
      const lateralR = qx * right[0] + qy * right[1] + qz * right[2];
      const lateralU = qx * up[0] + qy * up[1] + qz * up[2];
      distance = Math.max(distance,
        along + Math.abs(lateralR) / (0.6 * tanH),
        along + Math.abs(lateralU) / (0.6 * tanV));
    }
  }
  distance = Math.max(3.5, distance);
  return {
    position: [lookAt[0] + distance * eye[0], lookAt[1] + distance * eye[1], lookAt[2] + distance * eye[2]],
    lookAt,
  };
}

// ---------------------------------------------------------------------------
// Robot: one draw call per body, shared vertex buffers
// ---------------------------------------------------------------------------

function buildRobot(payload) {
  const { header, arrays } = payload;
  const material = new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide });
  const parts = [];
  for (const body of header.bodies) {
    const start = body.vertex_start * 3;
    const end = (body.vertex_start + body.vertex_count) * 3;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(arrays.positions.subarray(start, end), 3));
    geometry.setAttribute('normal', new THREE.BufferAttribute(arrays.normals.subarray(start, end), 3, true));
    geometry.setAttribute('color', new THREE.BufferAttribute(arrays.colors.subarray(start, end), 3, true));
    geometry.setIndex(new THREE.BufferAttribute(
      arrays.indices.subarray(body.index_start, body.index_start + body.index_count), 1));
    const mesh = new THREE.Mesh(geometry, material);
    mesh.frustumCulled = false;
    robotGroup.add(mesh);
    parts.push({ mesh, poseIndex: header.body_names.indexOf(body.name) });
  }
  state.robot = parts;
}

// ---------------------------------------------------------------------------
// Overlays
// ---------------------------------------------------------------------------

const sphereGeometry = new THREE.SphereGeometry(1, 14, 10);
const dummy = new THREE.Object3D();
const tmpVector = new THREE.Vector3();

// 亮色主题 + 部位配色：足/膝/手/肘/骨盆各一色（颜色来自服务端 config）。
function shortName(name) {
  return name.split('_').pop();
}

function colorFor(name) {
  const rgb = state.config.contact_colors[shortName(name)] || [120, 126, 134];
  return new THREE.Color(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255);
}

function clearObject(object, disposeGeometry = true) {
  overlayGroup.remove(object);
  if (disposeGeometry && object.geometry) object.geometry.dispose();
  if (object.dispose) object.dispose();
}

function clearOverlays() {
  for (const objects of Object.values(state.tracks)) {
    clearObject(objects.marker, false);
    if (objects.targets) clearObject(objects.targets, false);
    if (objects.lines) clearObject(objects.lines);
    if (objects.shadow) clearObject(objects.shadow);
    if (objects.drops) clearObject(objects.drops);
  }
  state.tracks = {};
  if (state.trail) { clearObject(state.trail); state.trail = null; }
}

function buildOverlays(payload) {
  clearOverlays();
  const { header, arrays } = payload;
  state.contactBodyIndex = header.contact_body_index || {};
  for (const key of ['reference', 'physics']) {
    const track = header.tracks[key];
    if (!track) continue;
    const count = track.names.length;
    const material = key === 'physics'
      ? new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.45, depthWrite: false })
      : new THREE.MeshBasicMaterial();
    const marker = new THREE.InstancedMesh(sphereGeometry, material, count);
    marker.frustumCulled = false;
    for (let i = 0; i < count; i += 1) marker.setColorAt(i, colorFor(track.names[i]));
    marker.instanceColor.needsUpdate = true;
    overlayGroup.add(marker);
    let targets = null;
    let lines = null;
    if (key === header.primary) {
      targets = new THREE.InstancedMesh(sphereGeometry,
        new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.35, depthWrite: false }), count);
      targets.frustumCulled = false;
      for (let i = 0; i < count; i += 1) targets.setColorAt(i, colorFor(track.names[i]));
      targets.instanceColor.needsUpdate = true;
      overlayGroup.add(targets);
      const lineGeometry = new THREE.BufferGeometry();
      lineGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(count * 6), 3));
      lineGeometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(count * 6), 3));
      lineGeometry.setDrawRange(0, 0);
      lines = new THREE.LineSegments(lineGeometry,
        new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.9 }));
      lines.frustumCulled = false;
      overlayGroup.add(lines);
    }
    // Ground projection: a flat disc under each active contact plus a faint
    // vertical drop line, so the foot/hand placement reads against the floor.
    let shadow = null;
    let drops = null;
    if (key === header.primary) {
      shadow = new THREE.InstancedMesh(new THREE.RingGeometry(0.84, 1, 28),
        new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.6, depthWrite: false, side: THREE.DoubleSide }), count);
      shadow.frustumCulled = false;
      for (let i = 0; i < count; i += 1) shadow.setColorAt(i, colorFor(track.names[i]));
      shadow.instanceColor.needsUpdate = true;
      overlayGroup.add(shadow);
      const dropGeometry = new THREE.BufferGeometry();
      dropGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(count * 6), 3));
      dropGeometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(count * 6), 3));
      dropGeometry.setDrawRange(0, 0);
      const dropColors = dropGeometry.attributes.color.array as Float32Array;
      for (let i = 0; i < count; i += 1) {
        const color = colorFor(track.names[i]);
        for (let k = 0; k < 2; k += 1) {
          const offset = (i * 2 + k) * 3;
          dropColors[offset] = color.r;
          dropColors[offset + 1] = color.g;
          dropColors[offset + 2] = color.b;
        }
      }
      drops = new THREE.LineSegments(dropGeometry,
        new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.5 }));
      drops.frustumCulled = false;
      overlayGroup.add(drops);
    }
    state.tracks[key] = { marker, targets, lines, shadow, drops, names: track.names, count };
  }
  // Pelvis trail (static for the clip).
  const poses = arrays.poses;
  const bodyCount = header.bodies.length;
  const pelvis = header.bodies.indexOf('pelvis');
  const frames = header.frames;
  const trailPositions = new Float32Array(Math.max(0, frames - 1) * 6);
  for (let f = 0; f + 1 < frames; f += 1) {
    const a = (f * bodyCount + pelvis) * 7;
    const b = ((f + 1) * bodyCount + pelvis) * 7;
    for (let k = 0; k < 3; k += 1) {
      trailPositions[f * 6 + k] = poses[a + k];
      trailPositions[f * 6 + 3 + k] = poses[b + k];
    }
  }
  const trailGeometry = new THREE.BufferGeometry();
  trailGeometry.setAttribute('position', new THREE.BufferAttribute(trailPositions, 3));
  state.trail = new THREE.LineSegments(trailGeometry,
    new THREE.LineBasicMaterial({ color: new THREE.Color(190 / 255, 110 / 255, 235 / 255), transparent: true, opacity: 0.75 }));
  state.trail.frustumCulled = false;
  overlayGroup.add(state.trail);
}

function trackFrame(times, seconds) {
  let low = 0, high = times.length;
  // times arrive as float32: at an exact 50 Hz boundary the stored value can be
  // a hair *above* ``seconds`` and searchsorted would pick the previous frame.
  const target = seconds + Math.max(1e-6, Math.abs(seconds) * 1e-6);
  while (low < high) {
    const mid = (low + high) >> 1;
    if (times[mid] <= target) low = mid + 1; else high = mid;
  }
  return clamp(low - 1, 0, times.length - 1);
}

function yawOf(w, x, y, z) {
  return Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
}

function updateScene(force) {
  if (!state.clip || !state.robot) return;
  const { header, arrays } = state.clip;
  const frames = header.frames;
  const bodyCount = header.bodies.length;
  const frame = clamp(Math.floor(state.frame), 0, frames - 1);
  const following = Math.min(frame + 1, frames - 1);
  const alpha = clamp(state.frame - frame, 0, 1);
  const poses = arrays.poses;
  const seconds = frame / header.fps;
  const now = performance.now();

  // Robot body transforms (linear position + quaternion slerp).
  for (const part of state.robot) {
    const index = part.poseIndex;
    if (index < 0) continue;
    const a = (frame * bodyCount + index) * 7;
    const b = (following * bodyCount + index) * 7;
    part.mesh.position.set(
      lerp(poses[a], poses[b], alpha),
      lerp(poses[a + 1], poses[b + 1], alpha),
      lerp(poses[a + 2], poses[b + 2], alpha),
    );
    // MuJoCo stores (w, x, y, z); three.js expects (x, y, z, w).
    part.mesh.quaternion.set(poses[a + 4], poses[a + 5], poses[a + 6], poses[a + 3])
      .slerp(new THREE.Quaternion(poses[b + 4], poses[b + 5], poses[b + 6], poses[b + 3]), alpha);
  }
  const pelvisIndex = header.bodies.indexOf('pelvis');
  const offset = (frame * bodyCount + pelvisIndex) * 7;
  const nextOffset = (following * bodyCount + pelvisIndex) * 7;
  const base = new THREE.Vector3(
    lerp(poses[offset], poses[nextOffset], alpha),
    lerp(poses[offset + 1], poses[nextOffset + 1], alpha),
    lerp(poses[offset + 2], poses[nextOffset + 2], alpha),
  );
  const pelvisQuat = new THREE.Quaternion(
    lerp(poses[offset + 4], poses[nextOffset + 4], alpha),
    lerp(poses[offset + 5], poses[nextOffset + 5], alpha),
    lerp(poses[offset + 6], poses[nextOffset + 6], alpha),
    lerp(poses[offset + 3], poses[nextOffset + 3], alpha),
  ).normalize();
  const yaw = yawOf(pelvisQuat.w, pelvisQuat.x, pelvisQuat.y, pelvisQuat.z);
  const cosYaw = Math.cos(yaw), sinYaw = Math.sin(yaw);

  // Contact markers.
  for (const key of ['physics', 'reference']) {
    const track = header.tracks[key];
    const objects = state.tracks[key];
    if (!track || !objects) continue;
    const enabled = key === 'reference' ? state.overlays.contacts : state.overlays.physics;
    const flag = arrays[`flag_${key}`];
    const pos = arrays[`pos_${key}`];
    const index = trackFrame(arrays[`times_${key}`], seconds);
    objects.marker.visible = enabled;
    for (let i = 0; i < objects.count; i += 1) {
      const at = index * objects.count + i;
      if (enabled && flag[at]) {
        dummy.position.set(pos[at * 3], pos[at * 3 + 1], pos[at * 3 + 2]);
        dummy.scale.setScalar(0.052);
        dummy.quaternion.identity();
        dummy.updateMatrix();
        objects.marker.setMatrixAt(i, dummy.matrix);
      } else {
        dummy.position.set(0, 0, 0);
        dummy.scale.setScalar(0);
        dummy.quaternion.identity();
        dummy.updateMatrix();
        objects.marker.setMatrixAt(i, dummy.matrix);
      }
    }
    objects.marker.instanceMatrix.needsUpdate = true;
    if (key === 'reference' && objects.shadow && objects.drops) {
      const dropPositions = objects.drops.geometry.attributes.position.array as Float32Array;
      let dropCount = 0;
      for (let i = 0; i < objects.count; i += 1) {
        const at = index * objects.count + i;
        if (enabled && flag[at]) {
          const px = pos[at * 3], py = pos[at * 3 + 1], pz = pos[at * 3 + 2];
          const floor = contactFloor(px, py, pz, state.ground);
          dummy.position.set(px, py, floor + 0.006);
          dummy.scale.setScalar(0.12);
          dummy.quaternion.identity();
          dummy.updateMatrix();
          objects.shadow.setMatrixAt(i, dummy.matrix);
          const offset = dropCount * 6;
          dropPositions[offset] = px;
          dropPositions[offset + 1] = py;
          dropPositions[offset + 2] = pz;
          dropPositions[offset + 3] = px;
          dropPositions[offset + 4] = py;
          dropPositions[offset + 5] = floor + 0.006;
          dropCount += 1;
        } else {
          dummy.position.set(0, 0, 0);
          dummy.scale.setScalar(0);
          dummy.quaternion.identity();
          dummy.updateMatrix();
          objects.shadow.setMatrixAt(i, dummy.matrix);
        }
      }
      objects.shadow.instanceMatrix.needsUpdate = true;
      objects.shadow.visible = enabled;
      objects.drops.geometry.setDrawRange(0, dropCount * 2);
      objects.drops.geometry.attributes.position.needsUpdate = true;
      objects.drops.visible = enabled;
    }
  }
  // Next-contact targets: strictly future onset, <=1.5 s, inside the yaw footprint.
  const primary = header.primary;
  const primaryTrack = primary ? header.tracks[primary] : null;
  const primaryObjects = primary ? state.tracks[primary] : null;
  if (primaryTrack && primaryObjects) {
    const count = primaryObjects.count;
    const next = arrays[`next_${primary}`];
    const pos = arrays[`pos_${primary}`];
    const times = arrays[`times_${primary}`];
    const trackIndex = trackFrame(times, seconds);
    const linePositions = primaryObjects.lines.geometry.attributes.position.array;
    const lineColors = primaryObjects.lines.geometry.attributes.color.array;
    let lineCount = 0;
    const valid = state.overlays.targets;
    primaryObjects.targets.visible = valid;
    primaryObjects.lines.visible = valid;
    for (let i = 0; i < count; i += 1) {
      const future = next[trackIndex * count + i];
      let ok = valid && future < primaryTrack.frames
        && times[Math.min(future, times.length - 1)] - times[trackIndex] <= 1.5 + 1e-8;
      let px = 0, py = 0, pz = 0;
      if (ok) {
        const at = (Math.min(future, times.length - 1) * count + i) * 3;
        px = pos[at]; py = pos[at + 1]; pz = pos[at + 2];
        ok = Number.isFinite(px) && Number.isFinite(py) && Number.isFinite(pz);
      }
      if (ok) {
        const dx = px - base.x, dy = py - base.y;
        const localX = dx * cosYaw + dy * sinYaw;
        const localY = -dx * sinYaw + dy * cosYaw;
        ok = Math.abs(localX) <= 0.8 && Math.abs(localY) <= 0.5;
      }
      if (ok) {
        dummy.position.set(px, py, pz);
        dummy.scale.setScalar(0.032);
        dummy.quaternion.identity();
        dummy.updateMatrix();
        primaryObjects.targets.setMatrixAt(i, dummy.matrix);
        const link = state.shortToLink[primaryTrack.names[i]];
        const bodyIndex = link === undefined ? undefined : state.contactBodyIndex[link];
        if (bodyIndex !== undefined) {
          const a = (frame * bodyCount + bodyIndex) * 7;
          const b = (following * bodyCount + bodyIndex) * 7;
          linePositions[lineCount * 6] = lerp(poses[a], poses[b], alpha);
          linePositions[lineCount * 6 + 1] = lerp(poses[a + 1], poses[b + 1], alpha);
          linePositions[lineCount * 6 + 2] = lerp(poses[a + 2], poses[b + 2], alpha);
          linePositions[lineCount * 6 + 3] = px;
          linePositions[lineCount * 6 + 4] = py;
          linePositions[lineCount * 6 + 5] = pz;
          const color = colorFor(primaryTrack.names[i]);
          for (let k = 0; k < 2; k += 1) {
            lineColors[lineCount * 6 + k * 3] = color.r;
            lineColors[lineCount * 6 + k * 3 + 1] = color.g;
            lineColors[lineCount * 6 + k * 3 + 2] = color.b;
          }
          lineCount += 1;
        }
      } else {
        dummy.position.set(0, 0, 0);
        dummy.scale.setScalar(0);
        dummy.quaternion.identity();
        dummy.updateMatrix();
        primaryObjects.targets.setMatrixAt(i, dummy.matrix);
      }
    }
    primaryObjects.targets.instanceMatrix.needsUpdate = true;
    primaryObjects.lines.geometry.attributes.position.needsUpdate = true;
    primaryObjects.lines.geometry.attributes.color.needsUpdate = true;
    primaryObjects.lines.geometry.setDrawRange(0, lineCount * 2);
  }

  if (state.trail) state.trail.visible = state.overlays.trail;

  // Camera follow keeps the user's orbit but tracks the pelvis.
  if (state.follow) {
    if (!state.followArmed) { state.followArmed = true; state.lastPelvis.copy(base); }
    const delta = tmpVector.copy(base).sub(state.lastPelvis);
    if (delta.lengthSq() > 0) {
      controls.target.add(delta);
      applyCamera();
    }
  }
  state.lastPelvis.copy(base);

  if (force || now - state.lastText >= 100) {
    $('status').innerHTML = `<span class="mono">${seconds.toFixed(2)} / ${((frames - 1) / header.fps).toFixed(2)} s</span>`
      + ` · 第 <span class="mono">${frame}</span> / ${frames - 1} 帧`;
    state.lastText = now;
  }
  if (force || now - state.lastCursor >= 50) { drawCursor(); state.lastCursor = now; }
  state.lastFrame = frame;
  state.lastSeconds = seconds;
  state.dirty = false;
}

// ---------------------------------------------------------------------------
// Contact timeline (canvas, drawn once per clip + a moving cursor)
// ---------------------------------------------------------------------------

const strip = $('strip');
const cursorCanvas = $('cursor');
const stripContext = strip.getContext('2d');
const cursorContext = cursorCanvas.getContext('2d');

function timelineGeometry() {
  const width = timelineEl.clientWidth || 1;
  const height = timelineEl.clientHeight || 1;
  // `top` leaves a ruler band above the rows: that strip is the playhead's
  // grab area, so scrubbing stays possible while edit mode owns the rows.
  const left = 98, right = 14, top = 26, bottom = 32;
  return { width, height, left, right, top, bottom, plotW: Math.max(width - left - right, 1), plotH: Math.max(height - top - bottom, 1) };
}

// --- timeline viewport (pan / zoom, like an editing app) -------------------

function viewSpan(): number {
  return Math.max(state.viewEnd - state.viewStart, 1e-6);
}

function timelineFrames(): number {
  return state.clip ? state.clip.header.frames : 1;
}

function frameToX(frame: number, g = timelineGeometry()): number {
  return g.left + ((frame - state.viewStart) / viewSpan()) * g.plotW;
}

function xToFrame(x: number, g = timelineGeometry()): number {
  return state.viewStart + ((x - g.left) / g.plotW) * viewSpan();
}

function clampTimelineView(): void {
  const frames = Math.max(timelineFrames(), 1);
  const minSpan = Math.min(8, frames);
  const span = clamp(state.viewEnd - state.viewStart, minSpan, frames);
  const start = clamp(state.viewStart, -0.5, frames - 0.5 - span);
  state.viewStart = start;
  state.viewEnd = start + span;
}

function resetTimelineView(): void {
  const frames = Math.max(timelineFrames(), 1);
  state.viewStart = -0.5;
  state.viewEnd = frames - 0.5;
}

function zoomTimeline(factor: number, anchorFrame?: number): void {
  if (!state.clip) return;
  const frames = Math.max(state.clip.header.frames, 1);
  const anchor = anchorFrame === undefined ? state.frame : anchorFrame;
  const span = viewSpan();
  const next = clamp(span / factor, Math.min(8, frames), frames);
  if (Math.abs(next - span) < 1e-9) return;
  const ratio = (anchor - state.viewStart) / span;
  state.viewStart = anchor - ratio * next;
  state.viewEnd = state.viewStart + next;
  clampTimelineView();
  drawTimeline();
}

function panTimeline(deltaFrames: number): void {
  if (!state.clip) return;
  state.viewStart += deltaFrames;
  state.viewEnd += deltaFrames;
  clampTimelineView();
  drawTimeline();
}

function zoomAtClientX(clientX: number, factor: number): void {
  if (!state.clip) return;
  const g = timelineGeometry();
  const rect = timelineEl.getBoundingClientRect();
  zoomTimeline(factor, xToFrame(clientX - rect.left, g));
}

// Keep the playhead on screen while playing a zoomed-in clip.
function followPlayhead(): void {
  if (!state.clip) return;
  const frames = Math.max(state.clip.header.frames, 1);
  const span = viewSpan();
  if (span >= frames - 1e-6) return;
  if (state.frame >= state.viewStart && state.frame <= state.viewEnd) return;
  const start = clamp(state.frame - span * 0.2, -0.5, frames - 0.5 - span);
  state.viewStart = start;
  state.viewEnd = start + span;
  drawTimeline();
}

function niceSecondStep(raw: number): number {
  const power = Math.pow(10, Math.floor(Math.log10(Math.max(raw, 1e-6))));
  const norm = raw / power;
  const step = norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10;
  return step * power;
}

function drawTimeline() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const geometry = timelineGeometry();
  strip.width = Math.round(geometry.width * dpr);
  strip.height = Math.round(geometry.height * dpr);
  cursorCanvas.width = strip.width;
  cursorCanvas.height = strip.height;
  stripContext.setTransform(dpr, 0, 0, dpr, 0, 0);
  cursorContext.setTransform(dpr, 0, 0, dpr, 0, 0);
  stripContext.clearRect(0, 0, geometry.width, geometry.height);
  const header = state.clip ? state.clip.header : null;
  const track = header && header.primary ? header.tracks[header.primary] : null;
  if (!track) {
    stripContext.fillStyle = '#6b7079';
    stripContext.font = '12px sans-serif';
    stripContext.fillText(state.clip ? '这段片段没有接触时间线' : '加载中…', 14, geometry.height / 2);
    cursorContext.clearRect(0, 0, geometry.width, geometry.height);
    return;
  }
  const key = header.primary;
  const arrays = state.clip.arrays;
  const flag = arrays[`flag_${key}`];
  const frames = track.frames;
  const rows = track.names.length;
  const { left, top, plotW, plotH } = geometry;
  // Ruler band: the strip above the rows is where the playhead is grabbed.
  stripContext.fillStyle = '#eceef1';
  stripContext.fillRect(left, 0, plotW, Math.max(top - 1, 1));
  stripContext.strokeStyle = 'rgba(20, 22, 26, .16)';
  stripContext.lineWidth = 1;
  stripContext.beginPath();
  stripContext.moveTo(left, top - 0.5);
  stripContext.lineTo(left + plotW, top - 0.5);
  stripContext.stroke();
  const rowH = plotH / rows;
  const span = viewSpan();
  const cellW = plotW / span;
  // Only paint the frames inside the viewport.
  const first = Math.max(0, Math.ceil(state.viewStart - 0.5));
  const last = Math.min(frames - 1, Math.floor(state.viewEnd + 0.5));
  const cellX = (frame: number) => frameToX(frame - 0.5, geometry);
  stripContext.fillStyle = '#f5f6f7';
  stripContext.fillRect(left, top, plotW, plotH);
  for (let i = 0; i < rows; i += 1) {
    const color = colorFor(track.names[i]);
    stripContext.fillStyle = `rgba(${Math.round(color.r * 255)}, ${Math.round(color.g * 255)}, ${Math.round(color.b * 255)}, .9)`;
    for (let f = first; f <= last; f += 1) {
      if (!flag[f * rows + i]) continue;
      stripContext.fillRect(cellX(f), top + i * rowH, Math.max(cellW, 0.75), Math.max(rowH - 0.4, 1));
    }
  }
  // Cells erased since the last save: hollow outline in the row's colour.
  const savedFlags = state.edit.saved;
  if (savedFlags && savedFlags.length === flag.length) {
    stripContext.lineWidth = 1;
    for (let i = 0; i < rows; i += 1) {
      const color = colorFor(track.names[i]);
      stripContext.strokeStyle = `rgba(${Math.round(color.r * 255)}, ${Math.round(color.g * 255)}, ${Math.round(color.b * 255)}, .6)`;
      for (let f = first; f <= last; f += 1) {
        const at = f * rows + i;
        if (flag[at] || !savedFlags[at]) continue;
        stripContext.strokeRect(cellX(f) + 0.5, top + i * rowH + 1.5,
          Math.max(cellW - 1, 1), Math.max(rowH - 3.5, 1));
      }
    }
  }
  // Edit mode: outline every contact block so its draggable edges are visible
  // (the per-frame fill and a run of the same colour read as one clip).
  if (state.edit.enabled) {
    stripContext.lineWidth = 1;
    stripContext.strokeStyle = 'rgba(20, 22, 26, .6)';
    for (let i = 0; i < rows; i += 1) {
      let runStart = -1;
      for (let f = first; f <= last + 1; f += 1) {
        const on = f <= last && !!flag[f * rows + i];
        if (on && runStart < 0) runStart = f;
        if (!on && runStart >= 0) {
          const x0 = cellX(runStart);
          const x1 = cellX(f);
          stripContext.strokeRect(x0 + 0.5, top + i * rowH + 1.5,
            Math.max(x1 - x0 - 1, 1), Math.max(rowH - 3, 1));
          runStart = -1;
        }
      }
    }
  }
  stripContext.strokeStyle = state.edit.enabled ? 'rgba(20, 22, 26, .85)' : 'rgba(20, 22, 26, .22)';
  stripContext.lineWidth = state.edit.enabled ? 1.5 : 1;
  stripContext.strokeRect(left + 0.5, top + 0.5, plotW - 1, plotH - 1);
  stripContext.fillStyle = '#5c626b';
  stripContext.font = `${clamp(rowH * 0.66, 7, 11).toFixed(1)}px sans-serif`;
  stripContext.textAlign = 'right';
  stripContext.textBaseline = 'middle';
  for (let i = 0; i < rows; i += 1) {
    stripContext.fillText(bodyLabel(track.names[i]), left - 8, top + (i + 0.5) * rowH);
  }
  // Time ticks adapt to the zoom level.
  const fps = header.fps;
  const startSec = state.viewStart / fps;
  const endSec = state.viewEnd / fps;
  const step = niceSecondStep((endSec - startSec) / 7);
  const decimals = step >= 1 ? 0 : step >= 0.1 ? 1 : 2;
  stripContext.textAlign = 'center';
  stripContext.textBaseline = 'top';
  for (let t = Math.ceil(startSec / step - 1e-9) * step; t <= endSec + 1e-9; t += step) {
    const x = frameToX(t * fps, geometry);
    stripContext.strokeStyle = 'rgba(20, 22, 26, .10)';
    stripContext.beginPath();
    stripContext.moveTo(x, top);
    stripContext.lineTo(x, top + plotH);
    stripContext.stroke();
    stripContext.fillText(`${t.toFixed(decimals)}s`, x, top + plotH + 7);
  }
  stripContext.textAlign = 'left';
  stripContext.fillText('时间 (s)', 12, top + plotH + 7);
  drawCursor();
}

function drawCursor() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const g = timelineGeometry();
  cursorContext.setTransform(dpr, 0, 0, dpr, 0, 0);
  cursorContext.clearRect(0, 0, g.width, g.height);
  if (!state.clip || !state.clip.header.primary) return;
  const header = state.clip.header;
  const x = frameToX(state.frame, g);
  if (x < g.left - 1 || x > g.left + g.plotW + 1) return;
  // Grab handle in the ruler band (drag it, or step with ←/→).
  cursorContext.fillStyle = '#14161a';
  cursorContext.beginPath();
  cursorContext.moveTo(x - 5, 1);
  cursorContext.lineTo(x + 5, 1);
  cursorContext.lineTo(x, 10);
  cursorContext.closePath();
  cursorContext.fill();
  cursorContext.strokeStyle = '#14161a';
  cursorContext.lineWidth = 1.5;
  cursorContext.beginPath();
  cursorContext.moveTo(x, 1);
  cursorContext.lineTo(x, g.top + g.plotH);
  cursorContext.stroke();
  cursorContext.fillStyle = '#14161a';
  cursorContext.font = '11px sans-serif';
  cursorContext.textAlign = x > g.left + g.plotW - 60 ? 'right' : 'left';
  cursorContext.textBaseline = 'top';
  cursorContext.fillText(`${(state.frame / header.fps).toFixed(2)} s`, x + 4, Math.max(g.top - 15, 1));
}

function scrubTo(clientX: number): void {
  if (!state.clip) return;
  const g = timelineGeometry();
  const rect = timelineEl.getBoundingClientRect();
  // Scrubbing behaves like an editor: playback stops and the playhead stays
  // on screen when the view is zoomed in.
  state.playing = false;
  $('play').checked = false;
  state.frame = clamp(xToFrame(clientX - rect.left, g), 0, state.clip.header.frames - 1);
  state.dirty = true;
  if (state.edit.anchor) extendAnchor(Math.floor(state.frame));
  followPlayhead();
}

// ---------------------------------------------------------------------------
// Hand-fixing contacts on the timeline
//
// Reference flags are edited in place; a snapshot of the flags as last loaded
// / saved is kept so we can show what changed and support undo. Saved fixes go
// to <ACP_CONTACT_EDIT_ROOT>/<dataset>/<clip> and the server applies them when
// baking, leaving the dataset's own .contacts.npz untouched.
//
// Two gestures share the timeline:
//   * free painting: click / drag cells to toggle them, Shift to erase;
//   * range mode: pick a body, press ＋接触 (or －擦除) at the playhead, then
//     drag the playhead; the range from the press to the playhead is filled.
// ---------------------------------------------------------------------------

function editTrack(): TrackHeader | null {
  return state.clip ? state.clip.header.tracks.reference || null : null;
}

function editFlags(): Uint8Array | null {
  return editTrack() && state.clip ? state.clip.arrays.flag_reference as Uint8Array : null;
}

function editPositions(): Float32Array | null {
  return editTrack() && state.clip ? (state.clip.arrays.pos_reference as Float32Array) || null : null;
}

function editRows(): number {
  const track = editTrack();
  return track ? track.names.length : 0;
}

function editRow(): number | null {
  const track = editTrack();
  if (!track) return null;
  const index = state.edit.body ? track.names.indexOf(state.edit.body) : 0;
  return index >= 0 ? index : (track.names.length ? 0 : null);
}

function contactBodyIndex(short: string): number | null {
  const link = state.shortToLink[short];
  const index = link === undefined ? undefined : state.contactBodyIndex[link];
  return index === undefined ? null : index;
}

function bodyPosition(short: string, frame: number): [number, number, number] | null {
  const clip = state.clip;
  const index = contactBodyIndex(short);
  if (!clip || index === null) return null;
  const at = (frame * clip.header.bodies.length + index) * 7;
  const poses = clip.arrays.poses;
  return [poses[at], poses[at + 1], poses[at + 2]];
}

// Newly enabled cells need a world position for the 3D marker. Keep the
// sidecar's own position when it has one; otherwise use the body's FK point.
function fillPosition(row: number, frame: number): void {
  const pos = editPositions();
  const track = editTrack();
  if (!pos || !track) return;
  const at = (frame * editRows() + row) * 3;
  const x = pos[at], y = pos[at + 1], z = pos[at + 2];
  if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)
      && (x !== 0 || y !== 0 || z !== 0)) return;
  const fallback = bodyPosition(track.names[row], frame);
  if (!fallback) return;
  pos[at] = fallback[0];
  pos[at + 1] = fallback[1];
  pos[at + 2] = fallback[2];
}

// A contact point for a frame that is not on yet: the segment's own point
// (so a dragged edge stays on the footprint), falling back to FK.
function positionAt(row: number, frame: number): [number, number, number] | null {
  const pos = editPositions();
  const track = editTrack();
  if (!pos || !track) return null;
  const at = (frame * editRows() + row) * 3;
  const x = pos[at], y = pos[at + 1], z = pos[at + 2];
  if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)
      && (x !== 0 || y !== 0 || z !== 0)) return [x, y, z];
  return bodyPosition(track.names[row], frame);
}

// Contiguous run of contact frames on one row around `frame` (null off-block).
function segmentAt(row: number, frame: number): { start: number; end: number } | null {
  const flags = editFlags();
  const clip = state.clip;
  if (!flags || !clip) return null;
  const rows = editRows();
  const frames = clip.header.frames;
  if (frame < 0 || frame >= frames || !flags[frame * rows + row]) return null;
  let start = frame;
  let end = frame;
  while (start > 0 && flags[(start - 1) * rows + row]) start -= 1;
  while (end < frames - 1 && flags[(end + 1) * rows + row]) end += 1;
  return { start, end };
}

// Keep at least one free frame between this segment and its neighbours.
function segmentLimits(row: number, start: number, end: number): { minStart: number; maxEnd: number } {
  const flags = editFlags();
  const clip = state.clip;
  if (!flags || !clip) return { minStart: start, maxEnd: end };
  const rows = editRows();
  const frames = clip.header.frames;
  let previous = -1;
  for (let frame = start - 1; frame >= 0; frame -= 1) {
    if (flags[frame * rows + row]) { previous = frame; break; }
  }
  let next = frames;
  for (let frame = end + 1; frame < frames; frame += 1) {
    if (flags[frame * rows + row]) { next = frame; break; }
  }
  return { minStart: previous < 0 ? 0 : previous + 2, maxEnd: next >= frames ? frames - 1 : next - 2 };
}

function diffFrames(a: Uint8Array, b: Uint8Array): number {
  const rows = editRows();
  if (!rows) return 0;
  let frames = 0;
  for (let base = 0; base < a.length; base += rows) {
    for (let row = 0; row < rows; row += 1) {
      if (a[base + row] !== b[base + row]) { frames += 1; break; }
    }
  }
  return frames;
}

// One button per contact body, in a fixed row: faster than a dropdown while
// scrubbing, and the colour dot matches the 3D markers / timeline rows.
function syncEditBody(): void {
  const root = $('edit-parts');
  const track = editTrack();
  const names = track ? track.names : [];
  if (!names.includes(state.edit.body)) state.edit.body = names[0] || null;
  root.textContent = '';
  for (const name of names) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'part-btn';
    button.classList.toggle('active', name === state.edit.body);
    button.title = `选择${bodyLabel(name)}`;
    const color = colorFor(name);
    const dot = document.createElement('i');
    dot.style.background = `rgb(${Math.round(color.r * 255)}, ${Math.round(color.g * 255)}, ${Math.round(color.b * 255)})`;
    button.append(dot, document.createTextNode(bodyLabel(name)));
    button.addEventListener('click', () => {
      finishAnchor(true);
      state.edit.body = name;
      state.edit.notice = '';
      syncEditBody();
      updateEditUI();
    });
    root.appendChild(button);
  }
}

function syncEditState(payload: ClipPayload): void {
  const track = payload.header.tracks.reference;
  const flags = track ? payload.arrays.flag_reference as Uint8Array : null;
  state.edit.saved = flags ? new Uint8Array(flags) : null;
  state.edit.undo = [];
  state.edit.redo = [];
  state.edit.dirty = false;
  state.edit.stroke = null;
  state.edit.anchor = null;
  state.edit.notice = '';
  state.edit.busy = false;
  state.edit.savedInfo = track && track.edit && track.edit.edited ? track.edit : null;
  if (!flags && state.edit.enabled) setEditEnabled(false);
  syncEditBody();
  updateEditUI();
}

function updateEditUI(): void {
  const flags = editFlags();
  const saved = state.edit.saved;
  const supported = !!flags && !!saved;
  const editable = supported && state.edit.enabled;
  const toggle = $('edit-toggle');
  toggle.disabled = !supported;
  toggle.textContent = state.edit.enabled ? '结束编辑' : '✎ 编辑接触';
  toggle.classList.toggle('active', state.edit.enabled);
  $('edit-undo').disabled = !state.edit.undo.length || state.edit.busy;
  $('edit-redo').disabled = !state.edit.redo.length || state.edit.busy;
  $('edit-save').disabled = !supported || !state.edit.dirty || state.edit.busy;
  $('edit-revert').disabled = !supported || !state.edit.dirty || state.edit.busy;
  $('edit-clear').disabled = !state.edit.savedInfo || state.edit.busy;
  $('tl-zoom-in').disabled = !state.clip;
  $('tl-zoom-out').disabled = !state.clip;
  $('tl-zoom-all').disabled = !state.clip;
  // Marking / erasing implicitly enters edit mode, so these stay clickable
  // whenever the clip has an editable contact track.
  $('edit-mark').disabled = !supported;
  $('edit-erase').disabled = !supported;
  $('edit-mark').classList.toggle('active', !!state.edit.anchor && state.edit.anchor.value === 1);
  $('edit-erase').classList.toggle('active', !!state.edit.anchor && state.edit.anchor.value === 0);
  for (const button of $('edit-parts').querySelectorAll<HTMLButtonElement>('.part-btn')) {
    button.disabled = !supported;
  }
  const hint = $('edit-hint');
  const anchor = state.edit.anchor;
  if (anchor) {
    hint.textContent = `第 ${anchor.from}–${anchor.to} 帧 · 拖时间轴刻度或 ←/→ 调终点 · `
      + `再点「${anchor.value ? '＋ 接触' : '－ 擦除'}」或 Enter 完成 · Esc 取消`;
  } else if (editable) {
    hint.textContent = '←/→ 逐帧（Shift 10 帧）· 拖顶部刻度条移动播放头 · 拖色块边缘改起止 · 空白处涂抹/Shift 擦除';
  } else if (supported) {
    hint.textContent = '←/→ 逐帧 · Shift+←/→ 10 帧 · ↑/↓ 跳接触点 · 滚轮平移 · Ctrl+滚轮缩放';
  } else {
    hint.textContent = state.clip ? '滚轮平移 · Ctrl+滚轮缩放' : '';
  }
  if (!supported) {
    $('edit-status').textContent = state.clip ? '这段没有参考接触，不能编辑' : '';
    return;
  }
  let text: string;
  if (state.edit.dirty) {
    text = `未保存 · ${diffFrames(flags, saved)} 帧改动`;
  } else if (state.edit.savedInfo) {
    const changed = state.edit.savedInfo.changed;
    text = changed && changed > 0 ? `已保存 · ${changed} 帧改动` : '已保存';
  } else {
    text = '无改动';
  }
  $('edit-status').textContent = state.edit.notice ? `${text} · ${state.edit.notice}` : text;
}

function setEditEnabled(value: boolean): void {
  if (value && !editFlags()) return;
  if (!value) finishAnchor(false);
  state.edit.enabled = value;
  state.edit.stroke = null;
  state.edit.notice = '';
  if (value) {
    state.playing = false;
    $('play').checked = false;
  }
  timelineEl.classList.toggle('editing', value);
  syncEditBody();
  updateEditUI();
  drawTimeline();
}

function pushSnapshot(snapshot: Uint8Array): void {
  state.edit.undo.push(snapshot);
  if (state.edit.undo.length > 80) state.edit.undo.shift();
  state.edit.redo = [];
}

function recomputeNext(): void {
  if (!state.clip) return;
  const track = editTrack();
  if (!track) return;
  const key = 'reference';
  const flag = state.clip.arrays[`flag_${key}`] as Uint8Array;
  const next = state.clip.arrays[`next_${key}`] as Int32Array;
  const frames = track.frames;
  const rows = track.names.length;
  const future = new Int32Array(rows).fill(frames);
  for (let frame = frames - 1; frame >= 0; frame -= 1) {
    for (let row = 0; row < rows; row += 1) {
      next[frame * rows + row] = future[row];
      const on = flag[frame * rows + row];
      const previous = frame === 0 ? 1 : flag[(frame - 1) * rows + row];
      if (on && !previous) future[row] = frame;
    }
  }
}

function commitEdit(): void {
  const flags = editFlags();
  const saved = state.edit.saved;
  if (!flags || !saved) return;
  recomputeNext();
  state.edit.dirty = diffFrames(flags, saved) > 0;
  state.edit.notice = '';
  state.dirty = true;
  updateEditUI();
  scheduleTimelineDraw();
}

// Paint [from, to] on one row; returns whether any cell actually changed.
function paintRange(row: number, from: number, to: number, value: number): boolean {
  const flags = editFlags();
  if (!flags) return false;
  const rows = editRows();
  let changed = false;
  for (let frame = Math.min(from, to); frame <= Math.max(from, to); frame += 1) {
    const at = frame * rows + row;
    if (flags[at] === value) continue;
    flags[at] = value;
    if (value) fillPosition(row, frame);
    changed = true;
  }
  if (changed) commitEdit();
  return changed;
}

// --- range mode ------------------------------------------------------------

// Fill [anchor.from, to] and restore everything the anchor touched outside it,
// so dragging the playhead back shrinks the interval instead of leaving a tail.
function applyAnchorRange(anchor: NonNullable<typeof state.edit.anchor>, to: number): void {
  const flags = editFlags();
  if (!flags) return;
  const rows = editRows();
  const low = Math.min(anchor.from, anchor.to, to);
  const high = Math.max(anchor.from, anchor.to, to);
  for (let frame = low; frame <= high; frame += 1) {
    const at = frame * rows + anchor.row;
    flags[at] = anchor.base[at];
  }
  for (let frame = Math.min(anchor.from, to); frame <= Math.max(anchor.from, to); frame += 1) {
    const at = frame * rows + anchor.row;
    flags[at] = anchor.value;
    if (anchor.value) fillPosition(anchor.row, frame);
  }
  anchor.to = to;
  if (!anchor.pushed && diffFrames(flags, anchor.base) > 0) {
    pushSnapshot(anchor.base);
    anchor.pushed = true;
  }
  commitEdit();
}

function startAnchor(value: number): void {
  const track = editTrack();
  const flags = editFlags();
  const row = editRow();
  if (!track || !flags || row === null) return;
  if (!state.edit.enabled) setEditEnabled(true);
  const frame = clamp(Math.round(state.frame), 0, track.frames - 1);
  const anchor = state.edit.anchor;
  if (anchor && anchor.row === row && anchor.value === value) {
    finishAnchor(true);
    return;
  }
  if (anchor) finishAnchor(true);
  state.playing = false;
  $('play').checked = false;
  const base = new Uint8Array(flags);
  state.edit.anchor = { row, value, from: frame, to: frame, base, pushed: false };
  state.edit.stroke = null;
  state.frame = frame;
  // The whole gesture becomes one undo step, the first time it changes a cell.
  applyAnchorRange(state.edit.anchor, frame);
  updateEditUI();
  drawTimeline();
  updateScene(true);
}

function extendAnchor(frame: number): void {
  const anchor = state.edit.anchor;
  if (!anchor) return;
  const track = editTrack();
  applyAnchorRange(anchor, clamp(frame, 0, (track ? track.frames : 1) - 1));
}

function finishAnchor(keep: boolean): void {
  const anchor = state.edit.anchor;
  if (!anchor) return;
  if (!keep) {
    const flags = editFlags();
    if (flags) {
      flags.set(anchor.base);
      // Cancelling restores the pre-anchor state, so the snapshot taken for
      // this gesture no longer represents a step worth undoing.
      if (anchor.pushed) state.edit.undo.pop();
      commitEdit();
    }
  }
  state.edit.anchor = null;
  updateEditUI();
  drawTimeline();
}

function undoEdit(): void {
  finishAnchor(true);
  const flags = editFlags();
  const snapshot = state.edit.undo.pop();
  if (!flags || !snapshot) return;
  state.edit.redo.push(new Uint8Array(flags));
  flags.set(snapshot);
  commitEdit();
}

function redoEdit(): void {
  finishAnchor(true);
  const flags = editFlags();
  const snapshot = state.edit.redo.pop();
  if (!flags || !snapshot) return;
  state.edit.undo.push(new Uint8Array(flags));
  flags.set(snapshot);
  commitEdit();
}

function revertEdits(): void {
  finishAnchor(false);
  const flags = editFlags();
  const saved = state.edit.saved;
  if (!flags || !saved || !state.edit.dirty) return;
  pushSnapshot(new Uint8Array(flags));
  flags.set(saved);
  commitEdit();
}

function bytesToBase64(bytes: Uint8Array): string {
  let binary = '';
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

async function saveEdits(): Promise<void> {
  const flags = editFlags();
  if (!state.clip || !state.clipName || !state.dataset || !flags || state.edit.busy) return;
  finishAnchor(true);
  const dataset = state.dataset;
  const clip = state.clipName;
  // Snapshot what we send: the user may keep painting while the POST is in
  // flight, and those newer edits must stay dirty afterwards.
  const sent = new Uint8Array(flags);
  const pos = editPositions();
  const sentPositions = pos
    ? new Uint8Array(pos.buffer, pos.byteOffset, pos.byteLength).slice()
    : null;
  state.edit.busy = true;
  state.edit.notice = '';
  updateEditUI();
  try {
    const body = await getJSON('/api/contact_edit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        dataset,
        clip,
        flags: bytesToBase64(sent),
        positions: sentPositions ? bytesToBase64(sentPositions) : undefined,
      }),
    });
    if (dataset !== state.dataset || clip !== state.clipName) return;
    state.edit.savedInfo = body.edited
      ? { edited: true, changed: body.changed, saved_at: body.saved_at }
      : null;
    state.edit.saved = sent;
    state.edit.dirty = diffFrames(flags, sent) > 0;
    state.edit.notice = state.edit.dirty ? '保存期间又有新改动，记得再保存' : '';
    drawTimeline();
  } catch (error) {
    state.edit.notice = `⚠ 保存失败: ${error.message}`;
  } finally {
    state.edit.busy = false;
    updateEditUI();
  }
}

async function clearSavedEdits(): Promise<void> {
  if (!state.clipName || !state.dataset || !state.edit.savedInfo || state.edit.busy) return;
  if (state.edit.dirty && !window.confirm('还有未保存的修改，清除已保存的编辑会一并丢弃。继续吗？')) return;
  finishAnchor(false);
  const dataset = state.dataset;
  const clip = state.clipName;
  state.edit.busy = true;
  updateEditUI();
  try {
    const body = await getJSON('/api/contact_edit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset, clip, revert: true }),
    });
    if (dataset !== state.dataset || clip !== state.clipName) return;
    if (body && body.unchanged) {
      state.edit.savedInfo = null;
      updateEditUI();
      return;
    }
    state.edit.dirty = false;
    state.edit.busy = false;
    await loadClip(clip);
  } catch (error) {
    state.edit.notice = `⚠ 清除失败: ${error.message}`;
  } finally {
    state.edit.busy = false;
    updateEditUI();
  }
}

function scheduleTimelineDraw(): void {
  if (state.edit.drawPending) return;
  state.edit.drawPending = true;
  requestAnimationFrame(() => {
    state.edit.drawPending = false;
    drawTimeline();
  });
}

// --- drag a contact block (like a clip in an editor) ------------------------

function startSegmentDrag(row: number, seg: { start: number; end: number }, frame: number, mode: 'start' | 'end' | 'move'): void {
  const flags = editFlags();
  const track = editTrack();
  if (!flags || !track) return;
  if (state.playing) {
    state.playing = false;
    $('play').checked = false;
  }
  finishAnchor(true);
  state.edit.stroke = null;
  const limits = segmentLimits(row, seg.start, seg.end);
  state.edit.segment = {
    row, mode, start: seg.start, end: seg.end, origin: frame,
    minStart: Math.min(limits.minStart, seg.start),
    maxEnd: Math.max(limits.maxEnd, seg.end),
    base: new Uint8Array(flags), pushed: false,
  };
  state.edit.body = track.names[row];
  state.frame = frame;
  state.dirty = true;
  syncEditBody();
  updateEditUI();
  drawTimeline();
}

function applySegmentDrag(frame: number): void {
  const drag = state.edit.segment;
  const flags = editFlags();
  if (!drag || !flags) return;
  const rows = editRows();
  let start = drag.start;
  let end = drag.end;
  const delta = frame - drag.origin;
  if (drag.mode === 'start') {
    start = clamp(drag.start + delta, drag.minStart, drag.end);
  } else if (drag.mode === 'end') {
    end = clamp(drag.end + delta, drag.start, drag.maxEnd);
  } else {
    const width = drag.end - drag.start;
    start = clamp(drag.start + delta, drag.minStart, drag.maxEnd - width);
    end = start + width;
  }
  // Restore the row over everything the gesture can touch, then write the new
  // block. Frames that were already in contact keep their recorded point
  // (heel-strike / toe-off rolls); frames added by the drag copy the block's
  // nearest contact point instead of following the swinging foot.
  const low = Math.min(drag.start, start);
  const high = Math.max(drag.end, end);
  for (let f = low; f <= high; f += 1) flags[f * rows + drag.row] = 0;
  for (let f = start; f <= end; f += 1) {
    const at = f * rows + drag.row;
    if (!drag.base[at]) {
      const source = positionAt(drag.row, clamp(f, drag.start, drag.end));
      const pos = editPositions();
      if (pos && source) {
        const to = at * 3;
        pos[to] = source[0];
        pos[to + 1] = source[1];
        pos[to + 2] = source[2];
      }
    }
    flags[at] = 1;
  }
  if (!drag.pushed && (start !== drag.start || end !== drag.end)) {
    pushSnapshot(drag.base);
    drag.pushed = true;
  }
  state.frame = drag.mode === 'end' ? end : start;
  state.dirty = true;
  commitEdit();
}

function finishSegmentDrag(): void {
  state.edit.segment = null;
  timelineEl.style.cursor = '';
}

function cancelSegmentDrag(): void {
  const drag = state.edit.segment;
  if (!drag) return;
  const flags = editFlags();
  if (flags) {
    flags.set(drag.base);
    if (drag.pushed) state.edit.undo.pop();
    commitEdit();
  }
  finishSegmentDrag();
}

// Pointer shape: ruler = scrub, edge = resize, inside = move, elsewhere = paint.
function updateTimelineCursor(event: PointerEvent): void {
  const g = timelineGeometry();
  const rect = timelineEl.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  const inPlotX = x >= g.left && x <= g.left + g.plotW;
  // Ruler band above the rows and the time axis below them both scrub.
  if (inPlotX && (y < g.top || y >= g.top + g.plotH)) {
    timelineEl.style.cursor = 'ew-resize';
    return;
  }
  if (!state.edit.enabled || state.edit.anchor || state.edit.segment || !state.clip) {
    timelineEl.style.cursor = '';
    return;
  }
  const hit = timelineHit(event);
  if (hit.row === null || hit.frame < 0) {
    timelineEl.style.cursor = '';
    return;
  }
  const seg = segmentAt(hit.row, hit.frame);
  if (!seg) {
    timelineEl.style.cursor = 'crosshair';
    return;
  }
  const edge = clamp((g.plotW / viewSpan()) * 0.35, 2, 6);
  const left = frameToX(seg.start - 0.5, g);
  const right = frameToX(seg.end + 0.5, g);
  const nearLeft = Math.abs(x - left) <= edge && Math.abs(x - left) <= Math.abs(x - right);
  const nearRight = Math.abs(x - right) <= edge;
  timelineEl.style.cursor = nearLeft || nearRight ? 'ew-resize' : 'grab';
}

interface TimelineHit { frame: number; row: number | null; inPlot: boolean }

function timelineHit(event: PointerEvent): TimelineHit {
  const track = editTrack();
  const g = timelineGeometry();
  const rect = timelineEl.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  const frames = track ? track.frames : (state.clip ? state.clip.header.frames : 1);
  if (x < g.left || x > g.left + g.plotW) return { frame: -1, row: null, inPlot: false };
  const frame = clamp(Math.round(xToFrame(x, g)), 0, Math.max(frames - 1, 0));
  const inPlot = !!track && y >= g.top && y < g.top + g.plotH;
  const rows = editRows();
  const row = inPlot ? clamp(Math.floor(((y - g.top) / g.plotH) * rows), 0, rows - 1) : null;
  return { frame, row, inPlot };
}

timelineEl.addEventListener('pointerdown', (event) => {
  if (!state.clip) return;
  // The timeline owns the keyboard while it is being used, so ←/→ and friends
  // never get eaten by a text box or a slider that was touched earlier.
  timelineEl.focus({ preventScroll: true });
  if (event.button === 1 || (event.button === 0 && event.altKey)) {
    event.preventDefault();
    timelineEl.setPointerCapture(event.pointerId);
    state.pan = { x: event.clientX, start: state.viewStart, span: viewSpan() };
    return;
  }
  if (event.button !== 0) return;
  const hit = timelineHit(event);
  if (hit.frame < 0) return;
  timelineEl.setPointerCapture(event.pointerId);
  if (state.edit.enabled && state.edit.anchor) {
    event.preventDefault();
    state.scrubbing = true;
    scrubTo(event.clientX);
    return;
  }
  if (state.edit.enabled && hit.row !== null) {
    const seg = event.shiftKey ? null : segmentAt(hit.row, hit.frame);
    if (seg) {
      event.preventDefault();
      const g = timelineGeometry();
      const rect = timelineEl.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const edge = clamp((g.plotW / viewSpan()) * 0.35, 2, 6);
      const left = frameToX(seg.start - 0.5, g);
      const right = frameToX(seg.end + 0.5, g);
      const mode = Math.abs(x - left) <= edge && Math.abs(x - left) <= Math.abs(x - right)
        ? 'start' : Math.abs(x - right) <= edge ? 'end' : 'move';
      startSegmentDrag(hit.row, seg, hit.frame, mode);
      timelineEl.style.cursor = mode === 'move' ? 'grabbing' : 'ew-resize';
      return;
    }
    event.preventDefault();
    if (state.playing) {
      state.playing = false;
      $('play').checked = false;
    }
    finishAnchor(true);
    state.frame = hit.frame;
    state.dirty = true;
    const flags = editFlags();
    if (!flags) return;
    const rows = editRows();
    const value = event.shiftKey ? 0 : (flags[hit.frame * rows + hit.row] ? 0 : 1);
    const snapshot = new Uint8Array(flags);
    const changed = paintRange(hit.row, hit.frame, hit.frame, value);
    state.edit.stroke = { value, frame: hit.frame, row: hit.row, snapshot: changed ? null : snapshot };
    if (changed) pushSnapshot(snapshot);
    updateEditUI();
  } else {
    event.preventDefault();
    state.scrubbing = true;
    scrubTo(event.clientX);
  }
});
timelineEl.addEventListener('pointermove', (event) => {
  if (state.pan) {
    const g = timelineGeometry();
    state.viewStart = state.pan.start - ((event.clientX - state.pan.x) / g.plotW) * state.pan.span;
    state.viewEnd = state.viewStart + state.pan.span;
    clampTimelineView();
    drawTimeline();
    return;
  }
  if (state.edit.segment) {
    const hit = timelineHit(event);
    if (hit.frame >= 0) applySegmentDrag(hit.frame);
    return;
  }
  const stroke = state.edit.stroke;
  if (stroke) {
    const hit = timelineHit(event);
    if (hit.row === null || hit.frame < 0) return;
    const changed = paintRange(hit.row, stroke.frame, hit.frame, stroke.value);
    if (changed && stroke.snapshot) {
      pushSnapshot(stroke.snapshot);
      stroke.snapshot = null;
      updateEditUI();
    }
    stroke.frame = hit.frame;
    stroke.row = hit.row;
    return;
  }
  if (state.scrubbing) {
    scrubTo(event.clientX);
    return;
  }
  updateTimelineCursor(event);
});
const endTimelineDrag = (event: PointerEvent): void => {
  state.edit.stroke = null;
  state.edit.segment = null;
  state.scrubbing = false;
  state.pan = null;
  timelineEl.style.cursor = '';
  if (timelineEl.hasPointerCapture(event.pointerId)) timelineEl.releasePointerCapture(event.pointerId);
};
timelineEl.addEventListener('pointerup', endTimelineDrag);
timelineEl.addEventListener('pointercancel', endTimelineDrag);
timelineEl.addEventListener('lostpointercapture', () => {
  state.edit.stroke = null;
  state.edit.segment = null;
  state.scrubbing = false;
  state.pan = null;
  timelineEl.style.cursor = '';
});
timelineEl.addEventListener('pointerleave', () => {
  if (!state.edit.segment && !state.edit.stroke && !state.pan && !state.scrubbing) timelineEl.style.cursor = '';
});

// Wheel: pan by default, Ctrl/Cmd (or a trackpad pinch) zooms around the cursor.
timelineEl.addEventListener('wheel', (event) => {
  if (!state.clip) return;
  event.preventDefault();
  const g = timelineGeometry();
  const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? g.plotH : 1;
  if (event.ctrlKey || event.metaKey) {
    const factor = clamp(Math.exp(-event.deltaY * unit * 0.0015), 0.7, 1.4);
    zoomAtClientX(event.clientX, factor);
  } else {
    panTimeline(((event.deltaX + event.deltaY) * unit / g.plotW) * viewSpan());
  }
}, { passive: false });

// ---------------------------------------------------------------------------
// Clip / dataset loading
// ---------------------------------------------------------------------------

function label(entry) {
  return `${state.config.marks[entry.status] || '○'} ${entry.name}`;
}

function curationFor(clips) {
  const out = {};
  for (const clip of clips) {
    if (clip.status !== 'unreviewed' || clip.note) out[clip.name] = { status: clip.status, note: clip.note };
  }
  return out;
}

function refreshGroups() {
  const counts = new Map<string, number>();
  for (const entry of state.entries.values()) {
    counts.set(entry.group, (counts.get(entry.group) || 0) + 1);
  }
  if (!counts.has(state.group)) state.group = '';
  const groups = [...counts.entries()].sort((a, b) => a[0].localeCompare(b[0]));
  state.selects.group.setOptions([
    { value: '', label: `全部分组（${state.entries.size}）` },
    ...groups.map(([group, count]) => ({ value: group, label: `${group}（${count}）` })),
  ]);
  state.selects.group.value = state.group;
}

function applyFilter(load) {
  const query = $('filter').value.trim().toLowerCase().split(/\s+/).filter(Boolean);
  const options = [...state.entries.values()].filter((entry) => {
    if (state.group && entry.group !== state.group) return false;
    // match the group as well: clips keep their source names (climb_58, ...), so the
    // skill/source tag (box_cross_php__php) is otherwise unfindable in the search box.
    const haystack = `${entry.name} ${entry.group}`.toLowerCase();
    return query.every((word) => haystack.includes(word));
  });
  state.clipOptions = options.map((entry) => entry.name);
  const clipSelect = state.selects.clip;
  clipSelect.disabled = !options.length;
  clipSelect.setOptions(options.map((entry) => ({ value: entry.name, label: label(entry), keywords: entry.group })));
  if (state.clipName && state.clipOptions.includes(state.clipName)) clipSelect.value = state.clipName;
  else if (options.length) clipSelect.value = options[0].name;
  if (options.length && load) {
    const pick = clipSelect.value;
    if (pick && (pick !== state.clipName || !state.clip)) loadClip(pick);
  }
}

async function refreshCuration() {
  const body = await getJSON(`/api/curation?dataset=${encodeURIComponent(state.dataset)}`);
  state.curation = body.curation || {};
  applyCuration();
}

function applyCuration() {
  for (const entry of state.entries.values()) {
    const record = state.curation[entry.name] || {};
    entry.status = record.status || 'unreviewed';
    entry.note = record.note || '';
  }
  const clipSelect = state.selects.clip;
  const current = clipSelect.value;
  clipSelect.setOptions(state.clipOptions.map((name) => {
    const entry = state.entries.get(name);
    return { value: name, label: entry ? label(entry) : name, keywords: entry ? entry.group : '' };
  }));
  clipSelect.value = current;
  showReview();
}

function showReview() {
  const name = state.clipName;
  const entry = name ? state.entries.get(name) : null;
  if (!entry) { $('review').innerHTML = ''; return; }
  const record = state.curation[name] || {};
  const status = record.status || 'unreviewed';
  const mark = (state.config.marks && state.config.marks[status]) || '';
  $('review').innerHTML = `<b>${mark} ${STATUS_ZH[status] || status}</b>`;
  if (document.activeElement !== $('note')) {
    const draft = state.noteDrafts.get(`${state.dataset}/${name}`);
    $('note').value = record.note !== undefined ? record.note : (draft || '');
  }
}

async function switchDataset(name: string) {
  if (state.edit.dirty && name !== state.dataset
      && !window.confirm('有未保存的接触修改，切换数据集会丢失。继续吗？')) {
    if (state.dataset) state.selects.dataset.value = state.dataset;
    return;
  }
  state.edit.dirty = false;
  state.edit.anchor = null;
  state.loadToken += 1;
  const previous = state.clipName;  // keep the clip across datasets: a policy rollout `<clip>__policy.npz` and its reference `<clip>.npz` are paired
  state.dataset = name;
  state.clipName = null;
  state.clip = null;
  $('clip-name').textContent = '—';
  $('clip-group').textContent = '—';
  clearOverlays();
  drawTimeline();
  const dataset = state.config.datasets.find((item) => item.name === name);
  const clips = dataset ? dataset.clips : [];
  state.entries = new Map(clips.map((clip) => [clip.name, { ...clip }]));
  if (previous) {
    const paired = previous.endsWith('__policy.npz') ? previous.replace(/__policy\.npz$/, '.npz') : previous.replace(/\.npz$/, '__policy.npz');
    state.clipName = [previous, paired].find((clip) => state.entries.has(clip)) || null;
  }
  state.curation = curationFor(clips);
  state.badClips = new Set();
  state.group = '';
  refreshGroups();
  $('filter').value = '';
  applyFilter(true);
}

function buildTerrain(payload) {
  if (state.surfaceMesh) {
    scene.remove(state.surfaceMesh);
    state.surfaceMesh.geometry.dispose();
    state.surfaceMesh.material.dispose();
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(payload.arrays.positions, 3));
  geometry.setIndex(new THREE.BufferAttribute(payload.arrays.indices, 1));
  const material = new THREE.MeshLambertMaterial({
    color: new THREE.Color(0xd7dade), side: THREE.DoubleSide, flatShading: true,
  });
  state.surfaceMesh = new THREE.Mesh(geometry, material);
  state.surfaceMesh.frustumCulled = false;
  scene.add(state.surfaceMesh);
  state.surface = new Surface(payload.arrays.positions, payload.arrays.indices);
}

// Ground height under an (x, y) contact, falling back to the clip's flat floor.
const floorProbe = new Float32Array(2);

function contactFloor(x: number, y: number, z: number, fallback: number): number {
  if (!state.surface || !state.surface.count) return fallback;
  floorProbe[0] = x;
  floorProbe[1] = y;
  const value = state.surface.heights(floorProbe, z + 0.02)[0];
  return Number.isFinite(value) ? value : fallback;
}

function resetView() {
  if (!state.clip) return;
  const { header, arrays } = state.clip;
  const poses = arrays.poses;
  const bodyCount = header.bodies.length;
  const pelvis = header.bodies.indexOf('pelvis');
  const frames = header.frames;
  const basePos = new Float32Array(frames * 3);
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity, minZ = Infinity;
  for (let f = 0; f < frames; f += 1) {
    const at = (f * bodyCount + pelvis) * 7;
    basePos[f * 3] = poses[at]; basePos[f * 3 + 1] = poses[at + 1]; basePos[f * 3 + 2] = poses[at + 2];
    minX = Math.min(minX, poses[at]); maxX = Math.max(maxX, poses[at]);
    minY = Math.min(minY, poses[at + 1]); maxY = Math.max(maxY, poses[at + 1]);
    maxZ = Math.max(maxZ, poses[at + 2]); minZ = Math.min(minZ, poses[at + 2]);
  }
  const first = pelvis * 7;
  const heading = headingXY(poses[first + 3], poses[first + 4], poses[first + 5], poses[first + 6]);
  let ground = minZ - 0.78;
  if (state.surface) {
    const centre = new Float32Array([0.5 * (minX + maxX), 0.5 * (minY + maxY)]);
    const value = state.surface.heights(centre, maxZ + 5)[0];
    if (Number.isFinite(value)) ground = value;
  }
  state.ground = ground;
  const view = frameClipCamera(basePos, heading, ground, (camera.fov * Math.PI) / 180, camera.aspect);
  setView(view.position, view.lookAt);
}

async function loadClip(name: string) {
  if (state.edit.dirty && state.clipName && name !== state.clipName
      && !window.confirm('有未保存的接触修改，切换片段会丢失。继续吗？')) {
    if (state.clipName) state.selects.clip.value = state.clipName;
    return;
  }
  const token = ++state.loadToken;
  $('load-error').textContent = '';
  if (!state.robot) {
    try {
      const robotPayload = state.robotPayload || await getBuffer(state.config.robot_url);
      state.robotPayload = robotPayload;
      buildRobot(robotPayload);
    } catch (error) {
      $('load-error').textContent = `⚠ 机器人网格载入失败: ${error.message}`;
      return;
    }
  }
  let payload;
  try {
    payload = await getBuffer(`/api/clip?dataset=${encodeURIComponent(state.dataset)}&clip=${encodeURIComponent(name)}`);
  } catch (error) {
    if (token !== state.loadToken) return;
    state.badClips.add(name);
    $('load-error').textContent = `⚠ 跳过 ${name}（读取失败）: ${error.message}`;
    const next = state.clipOptions.find((clip) => !state.badClips.has(clip));
    if (next && next !== name) loadClip(next);
    return;
  }
  if (token !== state.loadToken) return;
  const terrainUrl = payload.header.terrain.url;
  if (terrainUrl !== state.terrainUrl) {
    try {
      const terrain = await getBuffer(terrainUrl);
      if (token !== state.loadToken) return;
      buildTerrain(terrain);
      state.terrainUrl = terrainUrl;
    } catch (error) {
      if (token !== state.loadToken) return;
      state.badClips.add(name);
      $('load-error').textContent = `⚠ 无法载入地形 ${payload.header.terrain.name}: ${error.message}`;
      const next = state.clipOptions.find((clip) => !state.badClips.has(clip));
      if (next && next !== name) loadClip(next);
      return;
    }
  }
  if (token !== state.loadToken) return;
  state.clip = payload;
  state.clipName = name;
  resetTimelineView();
  $('clip-name').textContent = name;
  $('clip-group').textContent = state.entries.get(name)?.group ?? '';
  state.frame = 0;
  state.lastFrame = -1;
  state.lastSeconds = -1;
  state.dirty = true;
  state.followArmed = false;
  buildOverlays(payload);
  syncEditState(payload);
  if (state.clipOptions.includes(name)) state.selects.clip.value = name;
  showReview();
  resetView();
  drawTimeline();
  updateScene(true);
  renderer.render(scene, camera);
}

// ---------------------------------------------------------------------------
// Curation
// ---------------------------------------------------------------------------

async function postReview(status, noteOnly) {
  if (!state.clipName) return;
  const dataset = state.dataset;
  const clip = state.clipName;
  const note = $('note').value;
  try {
    const body = await getJSON('/api/review', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset, clip, status, note, note_only: noteOnly }),
    });
    if (dataset !== state.dataset) return;
    state.curation = body.curation || {};
    if (noteOnly && !(clip in state.curation)) {
      state.noteDrafts.set(`${dataset}/${clip}`, note);
      $('review-feedback').textContent = '备注已暂存，选“保留 / 剔除”后才会写入。';
    } else {
      state.noteDrafts.delete(`${dataset}/${clip}`);
      $('review-feedback').textContent = status ? `已保存：${clip}` : `已清除：${clip}`;
    }
    applyCuration();
    applyFilter(true);
  } catch (error) {
    $('review-feedback').textContent = `⚠ 保存失败: ${error.message}`;
  }
}

function nextUnreviewed() {
  const names = [...state.entries.keys()];
  if (!names.length) return;
  const start = state.clipName ? names.indexOf(state.clipName) : -1;
  const query = $('filter').value.trim().toLowerCase().split(/\s+/).filter(Boolean);
  for (let offset = 1; offset <= names.length; offset += 1) {
    const name = names[(start + offset + names.length) % names.length];
    const entry = state.entries.get(name);
    if (entry.status && entry.status !== 'unreviewed') continue;
    const haystack = `${entry.name} ${entry.group}`.toLowerCase();
    if (!query.every((word) => haystack.includes(word))) continue;
    applyFilter(false);
    state.selects.clip.value = name;
    loadClip(name);
    $('review-feedback').textContent = '已跳到下一段未审阅的片段。';
    return;
  }
  $('review-feedback').textContent = '当前筛选下没有未审阅的片段。';
}

// ---------------------------------------------------------------------------
// Playback
// ---------------------------------------------------------------------------

// Only real text entry keeps the arrow keys for itself; buttons, checkboxes
// and the timeline all let the global shortcuts drive the playhead.
function isTextEntry(target: EventTarget | null): boolean {
  const node = target as HTMLElement | null;
  if (!node || !(node instanceof HTMLElement)) return false;
  if (node.isContentEditable) return true;
  if (node instanceof HTMLTextAreaElement) return true;
  if (node instanceof HTMLInputElement) {
    const type = (node.type || 'text').toLowerCase();
    return !['range', 'checkbox', 'radio', 'button', 'submit', 'reset', 'color', 'file'].includes(type);
  }
  return false;
}

// Form controls that keep their own arrow-key behaviour while focused.
function keepsArrowKeys(target: EventTarget | null): boolean {
  const node = target as HTMLElement | null;
  if (!(node instanceof HTMLElement)) return false;
  if (node instanceof HTMLInputElement && node.type === 'range') return true;
  return node instanceof HTMLSelectElement;
}

function stepTo(frame: number): void {
  if (!state.clip) return;
  state.playing = false;
  $('play').checked = false;
  state.frame = clamp(Math.round(frame), 0, state.clip.header.frames - 1);
  state.dirty = true;
  if (state.edit.anchor) extendAnchor(Math.round(state.frame));
  followPlayhead();
}

function step(delta) {
  if (!state.clip) return;
  stepTo(state.frame + delta);
}

// ↑/↓: jump the playhead between contact edit points (frames where the
// contact pattern changes), the way an editor jumps between cuts. Uses the
// selected body; when that body has no contacts it follows the busiest one,
// so the keys never go dead on a leg-only or arm-only clip.
function jumpContact(direction: number): void {
  if (!state.clip) return;
  const flags = editFlags();
  if (!flags) return;
  const frames = state.clip.header.frames;
  const rows = editRows();
  if (!rows) return;
  const row = editRow();
  const counts = new Array(rows).fill(0);
  for (let frame = 0; frame < frames; frame += 1) {
    for (let index = 0; index < rows; index += 1) if (flags[frame * rows + index]) counts[index] += 1;
  }
  const busiest = counts.indexOf(Math.max(...counts));
  const track = row !== null && counts[row] > 0 ? row : busiest;
  if (counts[track] === 0) return;
  const active = (frame: number) => !!flags[frame * rows + track];
  const cut = (frame: number) => frame >= 0 && frame < frames && active(frame) !== active(frame - 1);
  const frame = Math.round(state.frame);
  if (direction > 0) {
    for (let at = frame + 1; at < frames; at += 1) {
      if (!cut(at)) continue;
      stepTo(at);
      return;
    }
    return;
  }
  for (let at = frame - 1; at >= 0; at -= 1) {
    if (!cut(at)) continue;
    stepTo(at);
    return;
  }
}

// 当前筛选结果里直接切上/下一段（跳过读取失败的片段）。
function stepClip(delta) {
  const names = state.clipOptions.filter((name) => !state.badClips.has(name));
  if (!names.length) return;
  const index = names.indexOf(state.clipName);
  const target = index < 0
    ? names[delta > 0 ? 0 : names.length - 1]
    : names[(index + delta + names.length) % names.length];
  if (!target || target === state.clipName) return;
  state.selects.clip.value = target;
  loadClip(target);
}

let lastTime = performance.now();

function animate(now) {
  const dt = Math.min((now - lastTime) / 1000, 0.1);
  lastTime = now;
  if (state.clip && state.playing) {
    state.frame += dt * state.clip.header.fps * state.speed;
    const frames = state.clip.header.frames;
    if (state.frame >= frames) {
      if (state.loop) state.frame %= frames;
      else {
        state.frame = frames - 1;
        state.playing = false;
        $('play').checked = false;
      }
    }
  }
  if (state.clip) updateScene(false);
  if (state.clip && state.playing) followPlayhead();
  state.cameraDirty = false;
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}

// ---------------------------------------------------------------------------
// UI wiring + boot
// ---------------------------------------------------------------------------

function wire() {
  const dockHandle = $('dock-resize');
  dockHandle.addEventListener('pointerdown', (event) => {
    if (event.button !== 0) return;
    event.preventDefault();
    dockHandle.setPointerCapture(event.pointerId);
    state.dockDrag = { y: event.clientY, height: state.timelineHeight };
    dockHandle.classList.add('dragging');
  });
  dockHandle.addEventListener('pointermove', (event) => {
    if (!state.dockDrag) return;
    applyTimelineHeight(state.dockDrag.height + (state.dockDrag.y - event.clientY));
  });
  const endDockDrag = (event: PointerEvent): void => {
    state.dockDrag = null;
    dockHandle.classList.remove('dragging');
    if (dockHandle.hasPointerCapture(event.pointerId)) dockHandle.releasePointerCapture(event.pointerId);
  };
  dockHandle.addEventListener('pointerup', endDockDrag);
  dockHandle.addEventListener('pointercancel', endDockDrag);
  dockHandle.addEventListener('lostpointercapture', () => {
    state.dockDrag = null;
    dockHandle.classList.remove('dragging');
  });
  dockHandle.addEventListener('dblclick', () => applyTimelineHeight(TL_DEFAULT_HEIGHT));
  window.addEventListener('resize', () => applyTimelineHeight(state.timelineHeight, false));
  state.selects = {
    dataset: new Select($('dataset'), { search: true, placeholder: '选择数据集' }),
    group: new Select($('group'), { search: true, placeholder: '全部分组', emptyText: '没有分组' }),
    clip: new Select($('clip'), { search: true, placeholder: '选择片段', emptyText: '没有匹配的片段' }),
  };

  state.selects.dataset.onChange = (value) => switchDataset(value);
  state.selects.group.onChange = (value) => {
    state.group = value;
    applyFilter(true);
  };
  $('filter').addEventListener('input', () => applyFilter(true));
  // Esc leaves the search box: the timeline gets the keyboard back.
  $('filter').addEventListener('keydown', (event) => {
    if (event.key !== 'Escape') return;
    event.preventDefault();
    (event.target as HTMLInputElement).blur();
    timelineEl.focus({ preventScroll: true });
  });
  state.selects.clip.onChange = (value) => { if (value) loadClip(value); };
  $('copy-clip').addEventListener('click', () => {
    if (!state.clipName) return;
    const area = document.createElement('textarea');
    area.value = state.clipName;
    area.style.position = 'fixed';
    area.style.opacity = '0';
    document.body.appendChild(area);
    area.select();
    let ok = false;
    try { ok = document.execCommand('copy'); } catch (_) { ok = false; }
    area.remove();
    const button = $('copy-clip');
    button.textContent = ok ? '已复制' : '复制失败';
    setTimeout(() => { button.textContent = '复制'; }, 1200);
  });
  $('edit-toggle').addEventListener('click', () => setEditEnabled(!state.edit.enabled));
  $('edit-mark').addEventListener('click', () => startAnchor(1));
  $('edit-erase').addEventListener('click', () => startAnchor(0));
  $('edit-undo').addEventListener('click', undoEdit);
  $('edit-redo').addEventListener('click', redoEdit);
  $('edit-save').addEventListener('click', () => { saveEdits(); });
  $('edit-revert').addEventListener('click', revertEdits);
  $('edit-clear').addEventListener('click', () => { clearSavedEdits(); });
  $('tl-zoom-in').addEventListener('click', () => zoomTimeline(1.5, state.frame));
  $('tl-zoom-out').addEventListener('click', () => zoomTimeline(1 / 1.5, state.frame));
  $('tl-zoom-all').addEventListener('click', () => { resetTimelineView(); drawTimeline(); });
  $('prev-clip').addEventListener('click', () => stepClip(-1));
  $('next-clip').addEventListener('click', () => stepClip(1));
  $('play').addEventListener('change', (event) => { state.playing = (event.target as HTMLInputElement).checked; });
  $('loop').addEventListener('change', (event) => { state.loop = (event.target as HTMLInputElement).checked; });
  $('speed').addEventListener('input', (event) => {
    state.speed = Number((event.target as HTMLInputElement).value);
    $('speed-val').textContent = `${state.speed}×`;
  });
  // After a mouse drag on the slider the arrows belong to the timeline again.
  $('speed').addEventListener('pointerup', () => timelineEl.focus({ preventScroll: true }));
  $('prev').addEventListener('click', () => step(-1));
  $('next').addEventListener('click', () => step(1));
  $('keep').addEventListener('click', () => postReview('keep', false));
  $('reject').addEventListener('click', () => postReview('reject', false));
  $('clear').addEventListener('click', () => postReview(null, false));
  $('next-unreviewed').addEventListener('click', nextUnreviewed);
  $('note').addEventListener('change', () => {
    if (state.clipName) postReview(null, true);
  });
  $('follow').addEventListener('change', (event) => {
    state.follow = (event.target as HTMLInputElement).checked;
    state.followArmed = false;
    state.dirty = true;
  });
  $('reset-view').addEventListener('click', resetView);
  $('snap').addEventListener('click', () => {
    if (!state.clip) return;
    controls.target.copy(state.lastPelvis);
    state.followArmed = false;
    applyCamera();
  });
  for (const [id, key] of [['ov-contacts', 'contacts'], ['ov-physics', 'physics'], ['ov-targets', 'targets'],
    ['ov-trail', 'trail']] as const) {
    $(id).addEventListener('change', (event) => {
      state.overlays[key] = (event.target as HTMLInputElement).checked;
      state.dirty = true;
    });
  }
  window.addEventListener('keydown', (event) => {
    const arrows = event.code === 'ArrowLeft' || event.code === 'ArrowRight'
      || event.code === 'ArrowUp' || event.code === 'ArrowDown';
    if (isTextEntry(event.target)) return;
    if (keepsArrowKeys(event.target) && (arrows || event.code === 'Home' || event.code === 'End')) return;
    const command = event.ctrlKey || event.metaKey;
    if (command && event.code === 'KeyZ') {
      event.preventDefault();
      if (event.shiftKey) redoEdit(); else undoEdit();
    } else if (command && event.code === 'KeyY') {
      event.preventDefault();
      redoEdit();
    } else if (command && event.code === 'KeyS') {
      event.preventDefault();
      saveEdits();
    } else if (!command && event.code === 'KeyE') {
      event.preventDefault();
      setEditEnabled(!state.edit.enabled);
    } else if (event.code === 'Enter' && state.edit.anchor) {
      event.preventDefault();
      finishAnchor(true);
      updateScene(true);
    } else if (event.code === 'Escape' && state.edit.segment) {
      event.preventDefault();
      cancelSegmentDrag();
      updateScene(true);
    } else if (event.code === 'Escape' && state.edit.anchor) {
      event.preventDefault();
      finishAnchor(false);
      updateScene(true);
    } else if (event.code === 'Escape' && state.edit.enabled) {
      event.preventDefault();
      setEditEnabled(false);
    } else if (event.code === 'Space') {
      event.preventDefault();
      state.playing = !state.playing;
      $('play').checked = state.playing;
    } else if (event.code === 'ArrowLeft') {
      event.preventDefault(); step(event.shiftKey ? -10 : -1);
    } else if (event.code === 'ArrowRight') {
      event.preventDefault(); step(event.shiftKey ? 10 : 1);
    } else if (event.code === 'ArrowUp') {
      event.preventDefault(); jumpContact(-1);
    } else if (event.code === 'ArrowDown') {
      event.preventDefault(); jumpContact(1);
    } else if (event.code === 'Home') {
      event.preventDefault(); stepTo(0);
    } else if (event.code === 'End') {
      event.preventDefault(); stepTo(state.clip ? state.clip.header.frames - 1 : 0);
    } else if (event.code === 'Comma') {
      event.preventDefault(); step(event.shiftKey ? -10 : -1);
    } else if (event.code === 'Period') {
      event.preventDefault(); step(event.shiftKey ? 10 : 1);
    } else if (event.code === 'BracketLeft') {
      event.preventDefault(); stepClip(-1);
    } else if (event.code === 'BracketRight') {
      event.preventDefault(); stepClip(1);
    } else if (event.code === 'Minus' || event.code === 'NumpadSubtract') {
      event.preventDefault(); zoomTimeline(1 / 1.5, state.frame);
    } else if (event.code === 'Equal' || event.code === 'NumpadAdd') {
      event.preventDefault(); zoomTimeline(1.5, state.frame);
    } else if (event.code === 'Digit0' || event.code === 'Numpad0') {
      event.preventDefault(); resetTimelineView(); drawTimeline();
    }
  });
}

async function boot() {
  wire();
  restoreTimelineHeight();
  // Playback starts on its own, so the checkbox has to say so.
  $('play').checked = state.playing;
  state.config = await getJSON('/api/config');
  const config = state.config;
  state.shortToLink = Object.fromEntries(Object.entries(config.contact_links).map(([link, short]) => [short, link]));
  state.selects.dataset.setOptions(config.datasets.map((dataset) => ({ value: dataset.name, label: dataset.name })));
  const legendNames = { foot: '足', knee: '膝', hand: '手', elbow: '肘', pelvis: '骨盆' };
  $('legend-colors').innerHTML = Object.entries(config.contact_colors)
    .map(([key, rgb]) => `<span><i style="background:rgb(${rgb[0]},${rgb[1]},${rgb[2]})"></i>${legendNames[key] || key}</span>`)
    .join('');
  const defaultName = config.datasets.some((dataset) => dataset.name === config.default)
    ? config.default : (config.datasets[0] || {}).name;
  state.selects.dataset.value = defaultName;
  await switchDataset(defaultName);
  resize();
  requestAnimationFrame(animate);
  setInterval(() => {
    if (document.hidden || !state.dataset) return;
    refreshCuration().catch(() => {});
  }, 5000);
}

boot().catch((error) => {
  $('load-error').textContent = `⚠ 初始化失败: ${error.message}`;
  console.error(error);
});

// Debug handle (read-only inspection from devtools / CDP tests).
(window as unknown as { __acp: typeof state }).__acp = state;
