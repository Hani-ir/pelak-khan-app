// Photos waiting to be sent (with their image files) and a small history (text only) of everything saved.
const DB = 'pelak-khan';

function open() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB, 2);
    req.onupgradeneeded = () => {
      const db = req.result;
      for (const name of ['queue', 'history']) {
        if (!db.objectStoreNames.contains(name)) db.createObjectStore(name, { keyPath: 'id' });
      }
      // v2: the shared label index, kept for offline guessing
      if (!db.objectStoreNames.contains('meta')) db.createObjectStore('meta');
    };
    req.onsuccess = () => {
      // let a newer version of the app (e.g. in another tab) upgrade the database instead of waiting forever
      req.result.onversionchange = () => req.result.close();
      resolve(req.result);
    };
    req.onerror = () => reject(req.error);
    req.onblocked = () => reject(new Error('پلاک‌خوان در زبانه دیگری باز است؛ آن را ببندید و دوباره باز کنید'));
  });
}

async function tx(store, mode, fn) {
  const db = await open();
  return new Promise((resolve, reject) => {
    const t = db.transaction(store, mode);
    const result = fn(t.objectStore(store));
    t.oncomplete = () => resolve(result?.result ?? result);
    t.onerror = () => reject(t.error);
  });
}

export const addToQueue = (item) => tx('queue', 'readwrite', (s) => s.put(item));
export const queued = () => tx('queue', 'readonly', (s) => s.getAll());
export const queueCount = () => tx('queue', 'readonly', (s) => s.count());
export const removeFromQueue = (id) => tx('queue', 'readwrite', (s) => s.delete(id));
export const addHistory = (entry) => tx('history', 'readwrite', (s) => s.put(entry));
export const history = () => tx('history', 'readonly', (s) => s.getAll());
export const getMeta = (key) => tx('meta', 'readonly', (s) => s.get(key));
export const setMeta = (key, value) => tx('meta', 'readwrite', (s) => s.put(value, key));
