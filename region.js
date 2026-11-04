// Province, city and plate type of an Iranian plate (e.g. 27ع34546 -> گیلان / رشت / عمومی).
// Same rules as the server (backend/app/plate_region.py): the last two digits are the region
// code, the letter tells shared codes apart and gives the plate type.
// Data: plate_regions.json from IranPlate-Vision (MIT, see THIRD_PARTY_NOTICES.md).

const LETTER_ALIASES = { 'ھ': 'ه', 'ا': 'الف' }; // data spells both forms; the OCR writes «الف» as «ا»
const canon = (l) => LETTER_ALIASES[l] ?? l;

let tables = null;

export async function loadRegions() {
  const data = await (await fetch('plate_regions.json')).json();
  const byCode = new Map(); // code -> [{ province, city, letters: Set | null }]
  for (const [province, cities] of Object.entries(data.carplates)) {
    for (const [city, codes] of Object.entries(cities)) {
      for (const [code, letters] of Object.entries(codes)) {
        if (!byCode.has(code)) byCode.set(code, []);
        byCode.get(code).push({ province, city, letters: letters?.length ? new Set(letters.map(canon)) : null });
      }
    }
  }
  const types = new Map();
  for (const t of data.carplate_types) for (const l of t.letters) types.set(canon(l), { type: t.type, id: t.id });
  tables = { byCode, types };
}

/** plate: normalized (digits, letter, digits). Returns null until loadRegions() is done. */
export function describePlate(plate) {
  if (!tables || !plate) return null;
  const letter = canon([...plate].filter((c) => !/\d/.test(c)).join(''));
  const t = tables.types.get(letter);
  const out = { province: null, city: null, plate_type: t?.type ?? null, plate_type_id: t?.id ?? null };
  const code = /\d\d$/.test(plate) ? plate.slice(-2) : '';
  const rows = tables.byCode.get(code) ?? [];
  // a city listed with this very letter is more specific than one that takes any letter
  const specific = rows.filter((r) => r.letters?.has(letter));
  const matches = specific.length ? specific : rows.filter((r) => !r.letters);
  const provinces = new Set((matches.length ? matches : rows).map((r) => r.province));
  if (provinces.size !== 1) return out;
  out.province = [...provinces][0];
  const cities = [...new Set(matches.map((r) => r.city))];
  if (cities.length === 1) out.city = cities[0];
  return out;
}

export function regionText(r) {
  if (!r) return '';
  const place = r.province ? [r.province, r.city].filter(Boolean).join(' / ') : 'محل پلاک نامشخص';
  return r.plate_type ? `${place} · ${r.plate_type}` : place;
}
