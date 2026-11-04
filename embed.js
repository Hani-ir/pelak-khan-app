// Image embedding with DINOv2-small (Apache-2.0, int8 ONNX made by tools/export_dinov2.py) and
// nearest-photo model guessing. The embedding is a 384-number fingerprint of the picture;
// photos of the same model have similar fingerprints.
/* global ort */

const SIZE = 224;
const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];
let session = null;

export async function loadEmbedder() {
  session = await ort.InferenceSession.create('models/dinov2_small_int8.onnx', { executionProviders: ['wasm'] });
}

export const embedderReady = () => session !== null;

// The whole picture is padded to a square (not centre-cropped), so a truck shot from the side keeps its cab.
export async function embed(canvas) {
  const side = Math.max(canvas.width, canvas.height);
  const sq = document.createElement('canvas');
  sq.width = SIZE; sq.height = SIZE;
  const ctx = sq.getContext('2d');
  ctx.fillStyle = 'rgb(124,116,104)'; // the normalisation mean, i.e. "nothing" to the model
  ctx.fillRect(0, 0, SIZE, SIZE);
  ctx.imageSmoothingQuality = 'high';
  const s = SIZE / side;
  const w = canvas.width * s, h = canvas.height * s;
  ctx.drawImage(canvas, (SIZE - w) / 2, (SIZE - h) / 2, w, h);
  const px = ctx.getImageData(0, 0, SIZE, SIZE).data;
  const plane = SIZE * SIZE, input = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    for (let c = 0; c < 3; c++) input[c * plane + i] = (px[i * 4 + c] / 255 - MEAN[c]) / STD[c];
  }
  const res = await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', input, [1, 3, SIZE, SIZE]) });
  return Float32Array.from(res[session.outputNames[0]].data); // already length 1
}

// 384 floats in [-1, 1] -> 384 signed bytes -> base64 (512 characters)
export function packEmbedding(e) {
  const b = new Int8Array(e.length);
  for (let i = 0; i < e.length; i++) b[i] = Math.max(-127, Math.min(127, Math.round(e[i] * 127)));
  let s = '';
  for (const v of new Uint8Array(b.buffer)) s += String.fromCharCode(v);
  return btoa(s);
}

export function unpackEmbedding(text) {
  const raw = atob(text);
  const b = new Int8Array(raw.length);
  for (let i = 0; i < raw.length; i++) b[i] = (raw.charCodeAt(i) << 24) >> 24;
  const e = new Float32Array(b.length);
  let n = 0;
  for (let i = 0; i < b.length; i++) { e[i] = b[i] / 127; n += e[i] * e[i]; }
  n = Math.sqrt(n) || 1;
  for (let i = 0; i < e.length; i++) e[i] /= n;
  return e;
}

const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const keyOf = (m) => `${m.brand}|${m.series}|${m.model}`;

/**
 * items: [{plate, brand, series, model, vec}] with confirmed labels.
 * Returns null, or {model, reason: 'plate'|'look', score 0..1, matches, alternatives: [{model, score}]}.
 */
export function guessModel(items, plate, vec, k = 7) {
  const labelled = items.filter((it) => it.brand);
  if (plate) {
    const same = labelled.filter((it) => it.plate === plate);
    if (same.length) {
      const counts = new Map();
      for (const it of same) counts.set(keyOf(it), [(counts.get(keyOf(it))?.[0] ?? 0) + 1, it]);
      const [n, it] = [...counts.values()].sort((a, b) => b[0] - a[0])[0];
      return { model: pick(it), reason: 'plate', score: n / same.length, matches: same.length, alternatives: [] };
    }
  }
  if (!vec) return null;
  const scored = labelled.filter((it) => it.vec).map((it) => ({ it, sim: dot(vec, it.vec) }))
    .sort((a, b) => b.sim - a.sim).slice(0, k);
  if (!scored.length) return null;
  const votes = new Map();
  for (const { it, sim } of scored) {
    const v = votes.get(keyOf(it)) ?? { model: pick(it), weight: 0, best: 0, n: 0 };
    v.weight += Math.max(sim, 0); v.best = Math.max(v.best, sim); v.n++;
    votes.set(keyOf(it), v);
  }
  const ranked = [...votes.values()].sort((a, b) => b.weight - a.weight);
  const top = ranked[0];
  return {
    model: top.model,
    reason: 'look',
    score: Math.max(0, Math.min(1, top.best)), // how alike the closest labelled photo is
    matches: top.n,
    alternatives: ranked.slice(1, 3).map((r) => ({ model: r.model, score: Math.max(0, r.best) })),
  };
}

const pick = (it) => ({ brand: it.brand, series: it.series, model: it.model, type: it.type ?? null });
