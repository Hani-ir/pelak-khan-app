// Offline support: everything the app needs (code, models, libraries) is cached on first visit.
const CACHE = 'pelak-khan-20261104230058';
const FILES = [
  './', 'index.html', 'app.js', 'plate.js', 'region.js', 'plate_regions.json', 'embed.js', 'store.js', 'drive.js', 'drive-setup.html', 'drive-setup.js', 'drive-script.txt', 'style.css', 'manifest.webmanifest', 'icon.svg',
  'icon-192.png', 'icon-512.png', 'Vazirmatn-Variable.woff2', 'catalog.json', 'lib/ort.wasm.min.js', 'lib/ort-wasm-simd-threaded.mjs', 'lib/ort-wasm-simd-threaded.wasm',
  'lib/opencv.js', 'models/plate_yolo.onnx', 'models/ocr_crnn.onnx', 'models/ocr_crnn.labels.json',
];

self.addEventListener('install', (e) => {
  // cache: 'reload' skips the browser's HTTP cache (GitHub Pages allows 10 minutes), otherwise a
  // new version could be installed with files left over from the previous one
  const fresh = FILES.map((f) => new Request(f, { cache: 'reload' }));
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(fresh)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (e) => {
  e.waitUntil(caches.keys()
    .then((keys) => Promise.all(keys.filter((k) => k.startsWith('pelak-khan-') && k !== CACHE).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});

self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== 'GET' || url.origin !== location.origin) return; // uploads to Google go to the network
  e.respondWith(caches.match(e.request, { ignoreSearch: true }).then((hit) => hit || fetch(e.request).then((res) => {
    // the fallback plate model is fetched only when first needed; keep it for offline use too
    if (res.ok && url.pathname.endsWith('.onnx')) {
      const copy = res.clone();
      caches.open(CACHE).then((c) => c.put(e.request, copy));
    }
    return res;
  })));
});
