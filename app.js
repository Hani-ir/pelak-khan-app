import { loadModels, readPlates, isValidPlate } from './plate.js';
import { loadEmbedder, embedderReady, embed, packEmbedding, unpackEmbedding, guessModel } from './embed.js';
import { loadRegions, describePlate, regionText } from './region.js';
import * as store from './store.js';
import { fetchIndex, isScriptUrl, testConnection, uploadBatch } from './drive.js';

const APP_VERSION = '0.6.0';
const MAX_WORK_SIDE = 2048; // phone photos are downscaled to this before plate reading
const OVERVIEW_SIDE = 1024;
const BATCH = 10; // photos per upload request
// Below this look-alike score the closest model is only offered, not pre-selected: a careless
// "save" would otherwise store a wrong label, and wrong labels make later guesses worse.
const PRESELECT_MIN = 0.5;

const $ = (id) => document.getElementById(id);
const FA_DIGITS = { '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4', '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9',
  '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4', '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9',
  // keyboards and OCR mix Arabic and Persian code points for the same letter (same table as the server)
  'ي': 'ی', 'ى': 'ی', 'ئ': 'ی', 'ﻯ': 'ی', 'ﻱ': 'ی', 'ك': 'ک', 'ﻙ': 'ک',
  'ة': 'ه', 'ۀ': 'ه', 'ھ': 'ه', 'ﻩ': 'ه', 'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ٱ': 'ا', 'ﺍ': 'ا',
  '‌': '', '‍': '', '‎': '', '‏': '', 'ـ': '' };
const LATIN = Object.fromEntries([...'ابپتثجحدزژسشصطعفقکگلمنوهی'].map((c, i) =>
  [c, 'A B P T Th J H D Z Zh S Sh Sad Ta Ein F Gh K G L M N V He Y'.split(' ')[i]]));

const settings = {
  driveUrl: localStorage.getItem('driveUrl') || '',
  driveKey: localStorage.getItem('driveKey') || '',
  device: localStorage.getItem('device') || '',
};
const connected = () => isScriptUrl(settings.driveUrl) && Boolean(settings.driveKey);
let catalog = [];
let chosen = JSON.parse(localStorage.getItem('lastModel') || 'null'); // null = unknown
let current = null; // analysis of the photo being labelled
// Several photos of one model, labelled together: { id, items: [{ photo, plate }], busy }. null = one photo.
let batch = null;
const MAX_BATCH = 30; // decoded photos stay in memory until saved
let ready = false;
let sending = false;
let labelIndex = { version: 1, items: [] }; // confirmed labels from every phone (index/labels.json)
const vecCache = new Map(); // packed embedding -> unpacked vector

function vecOf(packed) {
  if (!packed) return null;
  if (!vecCache.has(packed)) vecCache.set(packed, unpackEmbedding(packed));
  return vecCache.get(packed);
}

// Everything the guess can learn from: the shared index plus photos saved here but not sent yet.
async function knownLabels() {
  const shared = Array.isArray(labelIndex?.items) ? labelIndex.items : []; // a broken saved copy must not stop the app
  const seen = new Set(shared.map((it) => it.id));
  const pending = (await store.queued()).filter((q) => !seen.has(q.id) && q.info.brand)
    .map((q) => ({ id: q.id, plate: q.info.plate, brand: q.info.brand, series: q.info.series, model: q.info.model, type: q.info.vehicle_type, emb: q.emb }));
  return [...shared, ...pending].map((it) => ({ ...it, vec: vecOf(it.emb) }));
}

async function refreshIndex() {
  if (!connected() || !navigator.onLine) return;
  try {
    labelIndex = await fetchIndex(settings);
    await store.setMeta('labelIndex', labelIndex);
  } catch (e) {
    console.warn('label index', e);
  }
}

function status(text, isError = false) {
  $('status').textContent = text;
  $('status').classList.toggle('error', isError);
}

function toast(text, isError = false) {
  const t = $('toast');
  t.textContent = text;
  t.classList.toggle('error', isError);
  t.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { t.hidden = true; }, 2500);
}

function show(section) {
  for (const id of ['home', 'result', 'stats']) $(id).hidden = id !== section;
  window.scrollTo(0, 0);
}

const normalizePlate = (t) => [...(t || '')].map((c) => FA_DIGITS[c] ?? c).join('').replace(/[\s\-_|/]/g, '');
const plateForFile = (plate) => [...plate].map((c) => LATIN[c] ?? c).join('').replace(/[^A-Za-z0-9]/g, '');
const folderFor = (m) => (m ? `${m.brand}_${m.model}`.replace(/[^A-Za-z0-9._-]+/g, '-') : 'unknown');

async function updateQueueInfo() {
  const n = await store.queueCount();
  $('queueInfo').hidden = n === 0;
  $('queueInfo').textContent = !connected()
    ? `${n} عکس روی گوشی ذخیره است. برای ارسال، اتصال گوگل درایو را در تنظیمات (⚙) وارد کنید.`
    : `${n} عکس منتظر ارسال${navigator.onLine ? '' : ' (اینترنت وصل نیست)'}`;
}

// ---------- photo -> plate ----------

function scaledCanvas(src, longSide) {
  const w = src.width, h = src.height;
  const s = Math.min(1, longSide / Math.max(w, h));
  const c = document.createElement('canvas');
  c.width = Math.round(w * s); c.height = Math.round(h * s);
  c.getContext('2d').drawImage(src, 0, 0, c.width, c.height);
  return c;
}

async function encode(canvas, quality) {
  const webp = await new Promise((r) => canvas.toBlob(r, 'image/webp', quality));
  if (webp && webp.type === 'image/webp') return webp;
  return new Promise((r) => canvas.toBlob(r, 'image/jpeg', Math.min(quality + 0.1, 0.95))); // Safari has no WebP encoder
}

function plateCrop(work, p) {
  const pad = Math.round(p.h * 0.25);
  const x0 = Math.max(p.x - pad, 0), y0 = Math.max(p.y - pad, 0);
  const x1 = Math.min(p.x + p.w + pad, work.width), y1 = Math.min(p.y + p.h + pad, work.height);
  const s = Math.min(1, 600 / (x1 - x0));
  const c = document.createElement('canvas');
  c.width = Math.round((x1 - x0) * s); c.height = Math.round((y1 - y0) * s);
  c.getContext('2d').drawImage(work, x0, y0, x1 - x0, y1 - y0, 0, 0, c.width, c.height);
  return c;
}

// The cab sits above the front plate: comparing this region instead of the whole photo keeps the
// background (the same dealership yard in every photo) from making different models look alike.
// A 0.52 m plate is the ruler: ~3.1 m wide and ~3.4 m above it covers a truck cab, also at 45 degrees.
function vehicleRegion(work, p) {
  const x0 = Math.max(Math.round(p.x + p.w / 2 - 3 * p.w), 0);
  const x1 = Math.min(Math.round(p.x + p.w / 2 + 3 * p.w), work.width);
  const y0 = Math.max(Math.round(p.y - 6.5 * p.w), 0);
  const y1 = Math.min(Math.round(p.y + 2 * p.h), work.height);
  const c = document.createElement('canvas');
  c.width = x1 - x0; c.height = y1 - y0;
  c.getContext('2d').drawImage(work, x0, y0, c.width, c.height, 0, 0, c.width, c.height);
  return c;
}

// Plate, small copies to keep and the region used for the model guess, for one photo file.
async function analyze(file) {
  // decoding through a canvas also drops EXIF data such as GPS location
  const bmp = await createImageBitmap(file, { imageOrientation: 'from-image' });
  const work = scaledCanvas(bmp, MAX_WORK_SIDE);
  bmp.close();
  const ctx = work.getContext('2d');
  const mat = cv.matFromImageData(ctx.getImageData(0, 0, work.width, work.height));
  const t0 = performance.now();
  const plates = await readPlates(mat);
  mat.delete();
  const best = plates[0] || null;
  const overviewCanvas = scaledCanvas(work, OVERVIEW_SIDE);
  const overview = await encode(overviewCanvas, 0.5);
  const plateBlob = best ? await encode(plateCrop(work, best), 0.9) : null;
  const photo = { best, overview, plateBlob, ms: Math.round(performance.now() - t0), vec: null, guess: null, guessDone: false };
  return { photo, region: best ? vehicleRegion(work, best) : overviewCanvas };
}

function onPhotos(files) {
  const list = [...(files || [])];
  if (list.length <= 1) return onPhoto(list[0]);
  return onBatch(list);
}

async function onPhoto(file) {
  if (!file) return;
  if (!ready) { toast('مدل‌ها هنوز آماده نیستند، چند لحظه صبر کنید', true); return; }
  batch = null;
  status('در حال خواندن پلاک…');
  try {
    const { photo, region } = await analyze(file);
    current = photo;
    renderResult();
    status(photo.best ? `پلاک پیدا شد (${photo.ms} میلی‌ثانیه)` : 'پلاکی پیدا نشد؛ اگر دیده می‌شود، دستی بنویسید.');
    await guessFor(photo, region);
  } catch (e) {
    console.error(e);
    status(`خطا در پردازش عکس: ${e.message}`, true);
  }
}

// ---------- several photos of one model ----------

async function onBatch(files) {
  if (!ready) { toast('مدل‌ها هنوز آماده نیستند، چند لحظه صبر کنید', true); return; }
  if (files.length > MAX_BATCH) {
    toast(`حداکثر ${MAX_BATCH} عکس با هم؛ ${MAX_BATCH} عکس اول برداشته شد`, true);
    files = files.slice(0, MAX_BATCH);
  }
  const mine = { id: crypto.randomUUID(), items: [], busy: true, total: files.length };
  batch = mine;
  current = { guess: null, guessDone: false }; // the model card shows the vote of all photos
  renderResult();
  const known = await knownLabels();
  for (const [i, file] of files.entries()) {
    if (batch !== mine) return; // cancelled
    status(`در حال خواندن عکس ${i + 1} از ${files.length}…`);
    try {
      const { photo, region } = await analyze(file);
      try {
        if (embedderReady()) photo.vec = await embed(region);
      } catch (e) {
        console.warn('embedding', e);
      }
      const plate = photo.best?.text || '';
      photo.guess = guessModel(known, isValidPlate(plate) ? plate : null, photo.vec);
      photo.guessDone = true;
      mine.items.push({ photo, plate });
    } catch (e) {
      console.error(e);
      toast(`عکس ${i + 1} خوانده نشد: ${e.message}`, true);
    }
    if (batch === mine) renderBatch();
  }
  if (batch !== mine) return;
  mine.busy = false;
  current = { guess: voteGuess(mine.items.map((it) => it.photo.guess)), guessDone: true };
  applyGuess(current.guess);
  renderBatch();
  renderModels();
  const found = mine.items.filter((it) => it.plate).length;
  status(`${mine.items.length} عکس آماده؛ پلاک ${found} عکس خوانده شد. مدل را انتخاب و همه را ذخیره کنید.`);
}

// The model most photos point to (photos of one model may have different plates and colours).
function voteGuess(guesses) {
  const votes = new Map();
  for (const g of guesses) {
    if (!confident(g)) continue;
    const k = `${g.model.brand}|${g.model.series}|${g.model.model}`;
    const v = votes.get(k) ?? { g, n: 0, score: 0 };
    v.n++;
    v.score = Math.max(v.score, g.score);
    votes.set(k, v);
  }
  const top = [...votes.values()].sort((a, b) => b.n - a.n || b.score - a.score)[0];
  if (top) return { ...top.g, score: top.score, votes: top.n, of: guesses.length };
  const first = guesses.find(Boolean);
  return first ? { ...first, votes: 0, of: guesses.length } : null; // nothing confident: only offer it
}

function renderBatch() {
  const b = batch;
  $('batchTitle').textContent = b.busy
    ? `در حال خواندن ${b.items.length} از ${b.total} عکس…`
    : `${b.items.length} عکس از یک مدل؛ همه در پوشه همان مدل ذخیره می‌شوند`;
  const list = $('batchList');
  list.replaceChildren();
  b.items.forEach((it, i) => {
    const card = document.createElement('div');
    card.className = 'batch-item';
    const img = document.createElement('img');
    img.src = it.url ?? (it.url = URL.createObjectURL(it.photo.overview));
    img.alt = `عکس ${i + 1}`;
    const input = document.createElement('input');
    input.className = 'plate-input small';
    input.value = it.plate;
    input.placeholder = 'پلاک (اختیاری)';
    input.setAttribute('aria-label', `پلاک عکس ${i + 1}`);
    const check = document.createElement('p');
    const showCheck = () => {
      const p = normalizePlate(input.value);
      check.textContent = !p ? 'بدون پلاک' : isValidPlate(p) ? regionText(describePlate(p)) || 'قالب درست' : 'قالب پلاک درست نیست';
      check.className = !p ? 'muted' : isValidPlate(p) ? 'muted ok' : 'muted bad';
    };
    input.oninput = () => { it.plate = input.value; showCheck(); };
    showCheck();
    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'small-btn';
    remove.textContent = 'حذف از این دسته';
    remove.disabled = b.busy;
    remove.onclick = () => {
      b.items.splice(b.items.indexOf(it), 1);
      URL.revokeObjectURL(it.url);
      if (!b.items.length) { cancel(); return; }
      renderBatch();
    };
    card.append(img, input, check, remove);
    list.append(card);
  });
  $('saveBtn').disabled = b.busy || !b.items.length;
  $('saveBtn').textContent = b.busy ? 'لطفاً صبر کنید…' : `تأیید و ذخیره ${b.items.length} عکس`;
}

function cancel() {
  for (const it of batch?.items ?? []) if (it.url) URL.revokeObjectURL(it.url);
  batch = null;
  current = null;
  show('home');
  status('آماده');
}

// Pre-select the most likely model; the user confirms it by saving or picks another one.
async function guessFor(photo, region) {
  try {
    if (embedderReady()) photo.vec = await embed(region);
  } catch (e) {
    console.warn('embedding', e);
  }
  if (current !== photo) return; // the user already moved on
  const plate = normalizePlate($('plateInput').value);
  photo.guess = guessModel(await knownLabels(), isValidPlate(plate) ? plate : null, photo.vec);
  photo.guessDone = true;
  applyGuess(photo.guess);
  renderModels();
}

// A corrected plate may be one we already know: guess again (cheap, the photo fingerprint is reused).
async function reguessFromPlate() {
  const photo = current;
  if (!photo?.guessDone) return;
  const plate = normalizePlate($('plateInput').value);
  const g = guessModel(await knownLabels(), isValidPlate(plate) ? plate : null, photo.vec);
  if (current !== photo) return;
  const changed = JSON.stringify(g?.model) !== JSON.stringify(photo.guess?.model);
  photo.guess = g;
  if (changed) applyGuess(g);
  renderModels();
}

const confident = (g) => g && (g.reason === 'plate' || g.score >= PRESELECT_MIN);

function applyGuess(g) {
  if (confident(g)) chosen = g.model;
  else if (g) chosen = null; // weak look-alike: leave "unknown" until the user picks
}

function guessText(g) {
  if (g?.of) {
    const name = `${g.model.brand} ${g.model.model}`;
    return g.votes
      ? `حدس از روی ${g.votes} عکس از ${g.of}: ${name}. اگر درست نیست، مدل دیگری انتخاب کنید.`
      : `شبیه‌ترین مدل: ${name}، ولی مطمئن نیست؛ مدل را خودتان انتخاب کنید.`;
  }
  if (!g) return 'هنوز عکس تأییدشده‌ای برای حدس زدن نیست؛ مدل را انتخاب کنید. هر عکسی که تأیید کنید، حدس‌های بعدی را بهتر می‌کند.';
  const name = `${g.model.brand} ${g.model.model}`;
  if (g.reason === 'plate') return `این پلاک قبلاً با مدل ${name} ثبت شده است.`;
  const pct = Math.round(g.score * 100);
  if (!confident(g)) return `شبیه‌ترین مدل: ${name}، ولی شباهت فقط ${pct}٪ است؛ مدل را خودتان انتخاب کنید.`;
  const strength = pct >= 85 ? 'حدس قوی' : pct >= 70 ? 'حدس متوسط' : 'حدس ضعیف';
  return `${strength}: ${name} (شباهت ${pct}٪ با ${g.matches} عکس تأییدشده). اگر درست نیست، مدل دیگری انتخاب کنید.`;
}

function renderResult() {
  const many = Boolean(batch);
  $('preview').hidden = many;
  $('plateCard').hidden = many;
  $('batchCard').hidden = !many;
  $('saveBtn').disabled = false;
  $('saveBtn').textContent = 'تأیید و ذخیره';
  if (many) {
    $('noteInput').value = '';
    $('modelSearch').value = '';
    renderBatch();
    renderModels();
    show('result');
    return;
  }
  const { best, overview, plateBlob } = current;
  $('preview').src = URL.createObjectURL(overview);
  $('plateImg').hidden = !plateBlob;
  if (plateBlob) $('plateImg').src = URL.createObjectURL(plateBlob);
  $('plateInput').value = best?.text || '';
  $('plateHint').textContent = best
    ? `اطمینان خواندن: ${Math.round(best.readConf * 100)}٪. اگر اشتباه است، اصلاح کنید.`
    : 'پلاک خودکار پیدا نشد.';
  $('noteInput').value = '';
  $('modelSearch').value = '';
  checkPlate();
  renderModels();
  show('result');
}

function checkPlate() {
  const p = normalizePlate($('plateInput').value);
  const el = $('plateCheck');
  if (!p) { el.textContent = 'بدون پلاک ذخیره می‌شود.'; el.className = 'muted'; return; }
  const ok = isValidPlate(p);
  el.textContent = ok ? `قالب پلاک درست است. ${regionText(describePlate(p))}` : 'قالب پلاک ایران: ۲ رقم، یک حرف، ۵ رقم (مثلاً ۱۲ب۳۶۵۱۱)';
  el.className = ok ? 'muted ok' : 'muted bad';
}

// ---------- model picker ----------

function renderModels() {
  $('modelChosen').textContent = chosen ? `${chosen.brand} ${chosen.model}` : 'نامشخص';
  const g = current?.guess;
  if (current) $('modelGuess').textContent = current.guessDone ? guessText(g) : 'در حال حدس مدل…';
  const alts = $('altModels');
  alts.replaceChildren();
  const offers = [...(g && !confident(g) ? [{ model: g.model, score: g.score }] : []), ...(g?.alternatives ?? [])];
  for (const a of offers) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'small-btn';
    b.textContent = `${a.model.brand} ${a.model.model} (${Math.round(a.score * 100)}٪)`;
    b.onclick = () => pick(a.model);
    alts.append(b);
  }
  const q = $('modelSearch').value.trim().toLowerCase().split(/\s+/).filter(Boolean);
  const list = $('modelList');
  list.replaceChildren();
  let lastBrand = null, shown = 0;
  for (const m of catalog) {
    const hay = `${m.brand} ${m.series} ${m.model}`.toLowerCase();
    if (q.length && !q.every((w) => hay.includes(w))) continue;
    if (++shown > 60) break;
    if (m.brand !== lastBrand) {
      const g = document.createElement('div');
      g.className = 'group'; g.textContent = m.brand; list.append(g); lastBrand = m.brand;
    }
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = `${m.brand} ${m.model}`;
    if (chosen && chosen.brand === m.brand && chosen.series === m.series && chosen.model === m.model) b.className = 'selected';
    b.onclick = () => pick(m);
    list.append(b);
  }
}

function pick(m) {
  chosen = m;
  localStorage.setItem('lastModel', JSON.stringify(m)); // the next truck is often the same model
  renderModels();
}

// ---------- save & send ----------

// One queued photo set. Photos labelled together share batch_id, so they can be found again.
async function queuePhoto(photo, rawPlate, note, batchInfo) {
  const plate = normalizePlate(rawPlate);
  const now = new Date();
  const stamp = now.toISOString().replace(/[-:]/g, '').replace('T', '-').slice(0, 15);
  const id = crypto.randomUUID();
  const base = `${stamp}_${plate ? plateForFile(plate) : 'noplate'}_${id.slice(0, 4)}`;
  const info = {
    plate: plate || null,
    plate_valid: isValidPlate(plate),
    plate_read: photo.best?.text || null,
    plate_read_confidence: photo.best ? Number(photo.best.readConf.toFixed(3)) : null,
    plate_corrected: Boolean(photo.best && plate !== photo.best.text),
    plate_region: isValidPlate(plate) ? describePlate(plate) : null, // where the plate was issued, not where the customer lives
    model_guess: photo.guess
      ? { brand: photo.guess.model.brand, model: photo.guess.model.model, reason: photo.guess.reason, score: Number(photo.guess.score.toFixed(3)) }
      : null,
    model_guess_accepted: Boolean(photo.guess && chosen && photo.guess.model.brand === chosen.brand
      && photo.guess.model.series === chosen.series && photo.guess.model.model === chosen.model),
    brand: chosen?.brand ?? null,
    series: chosen?.series ?? null,
    model: chosen?.model ?? null,
    vehicle_type: chosen?.type ?? null,
    note: note || null,
    taken_at: now.toISOString(),
    device: settings.device || null,
    app_version: APP_VERSION,
    ...batchInfo,
  };
  // one folder per model, whatever the plate or colour: the photos teach the guess every angle
  const folder = folderFor(chosen);
  const emb = photo.vec ? packEmbedding(photo.vec) : null;
  await store.addToQueue({ id, folder, base, info, emb, full: photo.overview, plate: photo.plateBlob });
  await store.addHistory({ id, createdAt: info.taken_at, folder, plate, sent: false });
  return plate;
}

async function save() {
  const note = $('noteInput').value.trim();
  let message;
  if (batch) {
    if (batch.busy) return;
    const { id, items } = batch;
    for (const [i, it] of items.entries()) {
      await queuePhoto(it.photo, it.plate, note, { batch_id: id, batch_size: items.length, batch_index: i + 1 });
    }
    message = `${items.length} عکس در پوشه ${chosen ? `${chosen.brand} ${chosen.model}` : '«نامشخص»'} ذخیره شد`;
    for (const it of items) if (it.url) URL.revokeObjectURL(it.url);
    batch = null;
  } else {
    const plate = await queuePhoto(current, $('plateInput').value, note, {});
    message = `ذخیره شد${plate ? `: ${plate}` : ''}`;
  }
  current = null;
  show('home');
  toast(message);
  status('آماده');
  await updateQueueInfo();
  sendQueued();
}

const ext = (blob) => (blob.type === 'image/webp' ? 'webp' : 'jpg');

async function sendQueued() {
  if (sending || !connected() || !navigator.onLine) { await updateQueueInfo(); return; }
  sending = true;
  try {
    for (;;) {
      const items = (await store.queued()).sort((a, b) => a.info.taken_at.localeCompare(b.info.taken_at)).slice(0, BATCH);
      if (!items.length) break;
      const files = [];
      for (const it of items) {
        const dir = `photos/${it.folder}/${it.base}`;
        files.push({ path: `${dir}_full.${ext(it.full)}`, blob: it.full });
        if (it.plate) files.push({ path: `${dir}_plate.${ext(it.plate)}`, blob: it.plate });
        files.push({ path: `${dir}.json`, blob: new Blob([JSON.stringify(it.info, null, 2)], { type: 'application/json' }) });
      }
      const labels = items.filter((it) => it.info.brand) // only confirmed models teach the guess
        .map((it) => ({ id: it.id, plate: it.info.plate, brand: it.info.brand, series: it.info.series,
          model: it.info.model, type: it.info.vehicle_type, emb: it.emb ?? null, taken_at: it.info.taken_at }));
      const index = await uploadBatch(settings, files, labels);
      if (index) {
        labelIndex = index;
        await store.setMeta('labelIndex', labelIndex);
      }
      for (const it of items) {
        await store.removeFromQueue(it.id);
        await store.addHistory({ id: it.id, createdAt: it.info.taken_at, folder: it.folder, plate: it.info.plate, sent: true });
      }
      await updateQueueInfo();
    }
    if (ready) status('آماده؛ همه عکس‌ها ارسال شده‌اند');
  } catch (e) {
    console.error(e);
    status(`ارسال انجام نشد: ${e.message}`, true);
  } finally {
    sending = false;
    await updateQueueInfo();
  }
}

// ---------- stats & settings ----------

async function showStats() {
  const all = await store.history();
  const counts = new Map();
  for (const h of all) counts.set(h.folder, (counts.get(h.folder) || 0) + 1);
  const pending = all.filter((h) => !h.sent).length;
  $('statsSummary').textContent = `${all.length} عکس روی این گوشی ذخیره شده؛ ${pending} هنوز ارسال نشده.`;
  const body = $('statsTable').tBodies[0];
  body.replaceChildren();
  for (const [folder, n] of [...counts].sort((a, b) => b[1] - a[1])) {
    const tr = document.createElement('tr');
    const a = document.createElement('td'); a.textContent = folder === 'unknown' ? 'نامشخص' : folder.replace('_', ' ');
    const b = document.createElement('td'); b.textContent = String(n);
    tr.append(a, b); body.append(tr);
  }
  show('stats');
}

function openSettings() {
  $('driveUrlInput').value = settings.driveUrl;
  $('driveKeyInput').value = settings.driveKey;
  $('deviceInput').value = settings.device;
  $('settingsMsg').textContent = '';
  $('settingsDlg').showModal();
}

function readSettingsForm() {
  return { ...settings, driveUrl: $('driveUrlInput').value.trim(), driveKey: $('driveKeyInput').value.trim(), device: $('deviceInput').value.trim() };
}

async function testSettings() {
  $('settingsMsg').textContent = 'در حال بررسی…';
  try {
    const folder = await testConnection(readSettingsForm());
    $('settingsMsg').textContent = `اتصال درست است. عکس‌ها در پوشه «${folder}» گوگل درایو ذخیره می‌شوند.`;
  } catch (e) {
    $('settingsMsg').textContent = e.message;
  }
}

function saveSettings() {
  Object.assign(settings, readSettingsForm());
  localStorage.setItem('driveUrl', settings.driveUrl);
  localStorage.setItem('driveKey', settings.driveKey);
  localStorage.removeItem('token'); // the GitHub token of version 0.3 is not used any more
  localStorage.removeItem('repo');
  localStorage.setItem('device', settings.device);
  refreshIndex().then(sendQueued);
}

// The link from the script's email (sendLinks in drive/Code.gs): #connect=<base64url of {"u": url, "k": code}>.
// The part after # never leaves the phone, and it is removed from the address right away.
async function connectFromLink() {
  const m = location.hash.match(/^#connect=([\w-]+)$/);
  if (!m) return;
  history.replaceState(null, '', location.pathname + location.search);
  let link;
  try {
    const b64 = m[1].replace(/-/g, '+').replace(/_/g, '/');
    link = JSON.parse(new TextDecoder().decode(Uint8Array.from(atob(b64), (c) => c.charCodeAt(0))));
  } catch {
    toast('لینک اتصال خراب است؛ دوباره از ایمیل باز کنید', true);
    return;
  }
  if (!isScriptUrl(link.u) || typeof link.k !== 'string' || !link.k) {
    toast('لینک اتصال خراب است؛ دوباره از ایمیل باز کنید', true);
    return;
  }
  Object.assign(settings, { driveUrl: link.u, driveKey: link.k });
  localStorage.setItem('driveUrl', settings.driveUrl);
  localStorage.setItem('driveKey', settings.driveKey);
  try {
    await testConnection(settings);
    toast('اپ به گوگل درایو وصل شد');
  } catch (e) {
    toast(`لینک ذخیره شد ولی اتصال آزمایش نشد: ${e.message}`, true);
  }
  refreshIndex().then(sendQueued);
}

// ---------- new version ----------

// The service worker installs a new version in the background (sw.js: skipWaiting + claim) and then
// takes over this page; the code already running is still the old one until the page is reloaded.
function watchForUpdates() {
  if (!('serviceWorker' in navigator)) return;
  const hadController = Boolean(navigator.serviceWorker.controller); // false on the very first visit
  navigator.serviceWorker.addEventListener('controllerchange', () => {
    if (hadController) $('updateBar').hidden = false;
  });
  navigator.serviceWorker.register('sw.js').then((reg) => {
    // look for a new version whenever the app comes back to the screen, at most every 10 minutes
    let last = Date.now();
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'visible' && Date.now() - last > 10 * 60_000) {
        last = Date.now();
        reg.update().catch(() => {});
      }
    });
  }).catch((e) => console.warn('service worker', e));
  $('updateBar').onclick = () => {
    if (batch || current) { toast('اول عکس فعلی را ذخیره یا لغو کنید', true); return; }
    location.reload();
  };
}

// ---------- start ----------

async function start() {
  watchForUpdates();
  if (navigator.storage?.persist) navigator.storage.persist(); // ask the browser not to evict queued photos

  $('cameraInput').onchange = (e) => { onPhoto(e.target.files[0]); e.target.value = ''; };
  $('galleryInput').onchange = (e) => { onPhotos(e.target.files); e.target.value = ''; };
  $('plateInput').oninput = () => {
    checkPlate();
    reguessFromPlate();
  };
  $('modelSearch').oninput = renderModels;
  $('unknownBtn').onclick = () => { chosen = null; localStorage.removeItem('lastModel'); renderModels(); };
  $('saveBtn').onclick = () => save().catch((e) => toast(`ذخیره نشد: ${e.message}`, true));
  $('cancelBtn').onclick = cancel;
  $('statsBtn').onclick = showStats;
  $('backBtn').onclick = () => show('home');
  $('sendNowBtn').onclick = () => sendQueued().then(showStats);
  $('settingsBtn').onclick = openSettings;
  $('testBtn').onclick = testSettings;
  $('saveSettingsBtn').onclick = saveSettings;
  window.addEventListener('online', sendQueued);
  window.addEventListener('offline', updateQueueInfo);
  setInterval(sendQueued, 60_000);

  await connectFromLink();
  await updateQueueInfo();
  try {
    catalog = await (await fetch('catalog.json')).json();
    await loadRegions();
    await loadModels((step) => status(`در حال آماده‌سازی: ${step}…`));
    labelIndex = (await store.getMeta('labelIndex')) ?? labelIndex;
    ready = true;
    status('در حال آماده‌سازی: مدل حدس زدن مدل خودرو…');
    try {
      await loadEmbedder();
    } catch (e) {
      console.warn('embedder', e); // plate reading still works without it; guesses fall back to the same plate only
    }
    status(connected() ? 'آماده' : 'آماده. برای ارسال عکس‌ها، اتصال گوگل درایو را در تنظیمات (⚙) وارد کنید.');
    await refreshIndex();
    sendQueued();
  } catch (e) {
    console.error(e);
    status(`آماده‌سازی انجام نشد: ${e.message}`, true);
  }
}

// wait for the classic scripts (onnxruntime, OpenCV) that load with defer
if (document.readyState === 'complete') start(); else window.addEventListener('load', start);
