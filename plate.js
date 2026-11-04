// Iranian plate detection (YOLOv8 ONNX) and reading (CRNN ONNX) in the browser.
// Same steps and numbers as detector/lpr_detector/plates.py, which is adapted from
// Platrix (MIT, Copyright (c) 2026 AliAkrami1375). Preprocessing uses OpenCV.js so the
// CRNN sees exactly what it was trained on.
/* global ort, cv */

const CRNN_W = 128, CRNN_H = 32;
const PLATE_LETTERS = 'ابپتثجحدزژسشصطعفقکگلمنوهی';
const PLATE_RE = new RegExp(`^\\d{2}[${PLATE_LETTERS}]\\d{5}$`);

export const isValidPlate = (text) => PLATE_RE.test(text || '');

let sessions = null;
let labels = [];
let fallbackPromise = null;

async function loadCv() {
  if (typeof cv === 'undefined') throw new Error('OpenCV.js لود نشد');
  if (cv instanceof Promise) globalThis.cv = await cv;
  if (!cv.Mat) await new Promise((resolve) => { cv.onRuntimeInitialized = resolve; });
}

export async function loadModels(onStep = () => {}) {
  ort.env.wasm.numThreads = 1; // GitHub Pages cannot send the headers multi-threading needs
  ort.env.wasm.wasmPaths = new URL('lib/', document.baseURI).href;
  onStep('OpenCV');
  await loadCv();
  onStep('مدل پیدا کردن پلاک');
  const det = await ort.InferenceSession.create('models/plate_yolo.onnx', { executionProviders: ['wasm'] });
  onStep('مدل خواندن پلاک');
  const ocr = await ort.InferenceSession.create('models/ocr_crnn.onnx', { executionProviders: ['wasm'] });
  labels = await (await fetch('models/ocr_crnn.labels.json')).json();
  sessions = { det, ocr, fallback: null };
}

function fallbackSession() {
  // second detector trained on hard frames (trucks, dim, small plates); loaded on first need
  fallbackPromise ??= ort.InferenceSession.create('models/plate_yolo_fallback.onnx', { executionProviders: ['wasm'] })
    .then((s) => (sessions.fallback = s));
  return fallbackPromise;
}

// rgba: cv.Mat CV_8UC4 of the photo. Returns [{x,y,w,h,detectConf,text,readConf,valid,score}], best first.
export async function readPlates(rgba) {
  let boxes = await detect(sessions.det, rgba, 416);
  if (!boxes.length) boxes = await detect(await fallbackSession(), rgba, 640);
  const out = [];
  for (const [x1, y1, x2, y2, conf] of boxes) {
    const roi = rgba.roi(new cv.Rect(x1, y1, x2 - x1, y2 - y1));
    const [text, readConf] = await readCrop(roi);
    roi.delete();
    out.push({ x: x1, y: y1, w: x2 - x1, h: y2 - y1, detectConf: conf, text, readConf, valid: isValidPlate(text), score: conf * readConf });
  }
  return out.sort((a, b) => (b.valid - a.valid) || (b.score - a.score));
}

async function detect(session, rgba, size) {
  const h0 = rgba.rows, w0 = rgba.cols;
  const ratio = Math.min(size / h0, size / w0);
  const nh = Math.round(h0 * ratio), nw = Math.round(w0 * ratio);
  const top = Math.round((size - nh) / 2 - 0.1), left = Math.round((size - nw) / 2 - 0.1);
  const rgb = new cv.Mat(); cv.cvtColor(rgba, rgb, cv.COLOR_RGBA2RGB);
  const resized = new cv.Mat(); cv.resize(rgb, resized, new cv.Size(nw, nh), 0, 0, cv.INTER_LINEAR);
  const canvas = new cv.Mat(size, size, cv.CV_8UC3, new cv.Scalar(114, 114, 114));
  resized.copyTo(canvas.roi(new cv.Rect(left, top, nw, nh)));
  const px = canvas.data, plane = size * size, input = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    input[i] = px[i * 3] / 255; input[plane + i] = px[i * 3 + 1] / 255; input[2 * plane + i] = px[i * 3 + 2] / 255;
  }
  rgb.delete(); resized.delete(); canvas.delete();

  const res = await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', input, [1, 3, size, size]) });
  const t = res[session.outputNames[0]];
  const [, rows, n] = t.dims; // (1, 4 + classes, N)
  const d = t.data, cand = [];
  for (let j = 0; j < n; j++) {
    let score = 0;
    for (let c = 4; c < rows; c++) score = Math.max(score, d[c * n + j]);
    if (score <= 0.5) continue;
    const cx = d[j], cy = d[n + j], w = d[2 * n + j], h = d[3 * n + j];
    cand.push([(cx - w / 2 - left) / ratio, (cy - h / 2 - top) / ratio, (cx + w / 2 - left) / ratio, (cy + h / 2 - top) / ratio, score]);
  }
  return nms(cand, 0.45)
    .map(([x1, y1, x2, y2, s]) => [Math.max(Math.trunc(x1), 0), Math.max(Math.trunc(y1), 0), Math.min(Math.trunc(x2), w0), Math.min(Math.trunc(y2), h0), s])
    .filter(([x1, y1, x2, y2]) => x2 > x1 && y2 > y1);
}

function nms(boxes, iouThreshold) {
  boxes.sort((a, b) => b[4] - a[4]);
  const keep = [];
  for (const b of boxes) {
    if (keep.every((k) => iou(k, b) <= iouThreshold)) keep.push(b);
  }
  return keep;
}

function iou(a, b) {
  const w = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const h = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = w * h;
  return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9);
}

async function readCrop(roiRgba) {
  let gray = new cv.Mat(); cv.cvtColor(roiRgba, gray, cv.COLOR_RGBA2GRAY);
  if (gray.rows < 40) {
    const up = new cv.Mat();
    cv.resize(gray, up, new cv.Size(Math.max(Math.trunc(gray.cols * 40 / gray.rows), 1), 40), 0, 0, cv.INTER_CUBIC);
    gray.delete(); gray = up;
  }
  const bil = new cv.Mat(); cv.bilateralFilter(gray, bil, 5, 45, 45);
  const clahe = new cv.CLAHE(2.0, new cv.Size(8, 8));
  const con = new cv.Mat(); clahe.apply(bil, con);
  const blur = new cv.Mat(); cv.GaussianBlur(con, blur, new cv.Size(0, 0), 1.0);
  const sharp = new cv.Mat(); cv.addWeighted(con, 1.4, blur, -0.4, 0, sharp);
  const small = new cv.Mat(); cv.resize(sharp, small, new cv.Size(CRNN_W, CRNN_H), 0, 0, cv.INTER_AREA);
  const input = Float32Array.from(small.data, (v) => v / 255);
  [gray, bil, con, blur, sharp, small].forEach((m) => m.delete()); clahe.delete();

  const ocr = sessions.ocr;
  const res = await ocr.run({ [ocr.inputNames[0]]: new ort.Tensor('float32', input, [1, 1, CRNN_H, CRNN_W]) });
  const t = res[ocr.outputNames[0]];
  const [, steps, classes] = t.dims;
  return ctcGreedy(t.data, steps, classes);
}

function ctcGreedy(logits, steps, classes) {
  const blank = labels.length;
  let prev = -1, text = '';
  const confs = [];
  for (let s = 0; s < steps; s++) {
    const row = logits.subarray(s * classes, (s + 1) * classes);
    let best = 0;
    for (let c = 1; c < classes; c++) if (row[c] > row[best]) best = c;
    if (best !== blank && best !== prev && best < labels.length) {
      let max = -Infinity; for (const v of row) max = Math.max(max, v);
      let sum = 0; for (const v of row) sum += Math.exp(v - max);
      text += labels[best];
      confs.push(Math.exp(row[best] - max) / sum);
    }
    prev = best;
  }
  return [text, confs.length ? confs.reduce((a, b) => a + b, 0) / confs.length : 0];
}
