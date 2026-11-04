// Sends saved photos to the user's Google Drive («Dst lubestation / pelak-khan») through the
// small Google Apps Script in drive/Code.gs, which runs in the user's own Google account.
// The phone only knows the script's Web app URL and a connection code: no password or token.

export class DriveError extends Error {
  constructor(code, message) { super(message); this.code = code; }
}

const MESSAGES = {
  bad_key: 'کد اتصال درست نیست. کد را دوباره از اسکریپت گوگل کپی کنید.',
  bad_request: 'درخواست ناقص بود (نسخه اپ و اسکریپت را بررسی کنید).',
  server: 'اسکریپت گوگل خطا داد',
};

export const isScriptUrl = (url) => /^https:\/\/script\.google\.com\/macros\/s\/[\w-]+\/exec$/.test(url || '');

async function call(settings, body) {
  if (!isScriptUrl(settings.driveUrl)) throw new DriveError('bad_url', 'لینک اسکریپت باید با https://script.google.com/macros/s/ شروع و با /exec تمام شود.');
  let res;
  try {
    // text/plain keeps this a "simple" request (no CORS preflight, which Apps Script does not answer)
    res = await fetch(settings.driveUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'text/plain;charset=utf-8' },
      body: JSON.stringify({ ...body, key: settings.driveKey }),
      redirect: 'follow',
      cache: 'no-store',
    });
  } catch {
    throw new DriveError('network', 'اتصال به گوگل درایو برقرار نشد');
  }
  let data;
  try {
    data = await res.json();
  } catch {
    // a Google sign-in page instead of JSON: the deployment is not open to "Anyone"
    throw new DriveError('not_json', 'پاسخ گوگل قابل خواندن نیست؛ در Deploy گزینه «Who has access: Anyone» را انتخاب کنید.');
  }
  if (!data.ok) throw new DriveError(data.error, MESSAGES[data.error] ? `${MESSAGES[data.error]}${data.message ? `: ${data.message}` : ''}` : `خطای درایو (${data.error})`);
  return data;
}

export async function testConnection(settings) {
  return (await call(settings, { action: 'ping' })).folder;
}

// The shared label index (confirmed models from every phone).
export async function fetchIndex(settings) {
  return (await call(settings, { action: 'index' })).index;
}

function toBase64(blob) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result).split(',')[1]);
    r.onerror = () => reject(r.error);
    r.readAsDataURL(blob);
  });
}

// files: [{path, blob}]; labels: index entries to add. Returns the updated index.
// Safe to repeat: the script keeps the first copy of a file name and adds labels by id.
export async function uploadBatch(settings, files, labels) {
  const payload = [];
  for (const f of files) payload.push({ path: f.path, type: f.blob.type || 'application/octet-stream', data: await toBase64(f.blob) });
  return (await call(settings, { action: 'upload', files: payload, labels })).index;
}
