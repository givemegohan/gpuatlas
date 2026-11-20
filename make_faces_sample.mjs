// Generate the "faces" sample dataset: a fictional online
// community (54 members in 4 groups) whose nodes carry real
// avatar images, demonstrating gpuatlas' real-image atlas.
//
// Avatars are DiceBear "Miniavs" style PNGs (by Webpixels,
// CC BY 4.0 — attribution lives in the data's meta.attribution
// and in RESULTS.md), downloaded once from the DiceBear API and
// self-hosted under site/faces/avatars/. Handles are fictional
// romaji combinations generated here; the network is a seeded
// random directed follow graph. Re-running reproduces the same
// dataset (avatar bytes depend only on the DiceBear seed).
//
// Usage: node make_faces_sample.mjs
import { mkdir, writeFile, stat } from "node:fs/promises";
import { dirname } from "node:path";
import { fileURLToPath } from "node:url";

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const rnd = mulberry32(20261120);

// ---------- fictional handles ----------
const BASES = [
  "sora", "yoru", "hana", "mizu", "kaze", "hoshi", "ame", "tsuki",
  "umi", "mori", "yuki", "haru", "natsu", "aki", "fuyu", "hikari",
  "neko", "sakura", "momiji", "suzu", "aoi", "akane", "sumire",
  "tsubaki", "kaede", "hotaru", "komugi", "mochi", "purin",
  "matcha", "soda", "melon", "cotton", "piano", "harp",
];
const FORMS = [(b) => b, (b) => b + "ch", (b) => b + "_ch", (b) => b + "san",
  (b) => b + "_kun", (b) => b + "maru", (b) => b + "_tv", (b) => b + "works",
  (b) => b + "_lab", (b) => b + "chan", (b) => b + "_nook", (b) => b + "_2"];
const combos = [];
for (const b of BASES) for (const f of FORMS) combos.push(f(b));
// Seeded Fisher-Yates, then take 54 unique handles.
for (let i = combos.length - 1; i > 0; i--) {
  const j = Math.floor(rnd() * (i + 1));
  [combos[i], combos[j]] = [combos[j], combos[i]];
}
const handles = [...new Set(combos)].slice(0, 54);
if (handles.length !== 54) throw new Error("not enough unique handles");

// ---------- groups ----------
const GROUPS = [
  { name: "ゲーム実況", size: 14 },
  { name: "歌ってみた", size: 14 },
  { name: "イラスト", size: 13 },
  { name: "雑談", size: 13 },
];
const members = []; // {handle, group}
{
  let g = 0;
  for (const grp of GROUPS) {
    for (let k = 0; k < grp.size; k++) members.push({ handle: handles[members.length], group: g });
    g++;
  }
}
const byGroup = GROUPS.map((_, g) => members.map((m, i) => (m.group === g ? i : -1)).filter((i) => i >= 0));

// ---------- directed follow edges ----------
const edgeSet = new Set();
const edges = [];
function addEdge(a, b) {
  if (a === b) return false;
  const key = a + ">" + b;
  if (edgeSet.has(key)) return false;
  edgeSet.add(key);
  const same = members[a].group === members[b].group;
  edges.push({ a, b, strength: 1, width: 1, type: same ? "グループ内" : "グループ間", directed: true });
  return true;
}
function pickOther(i, pool) {
  for (let tries = 0; tries < 12; tries++) {
    const t = pool[Math.floor(rnd() * pool.length)];
    if (t !== i) return t;
  }
  return -1;
}
for (let i = 0; i < members.length; i++) {
  const inPool = byGroup[members[i].group];
  const kIn = 2 + Math.floor(rnd() * 3); // 2-4 in-group follows
  for (let k = 0; k < kIn; k++) {
    const t = pickOther(i, inPool);
    if (t >= 0) addEdge(i, t);
  }
  const kOut = Math.floor(rnd() * 1.6); // 0-1 cross-group follows
  for (let k = 0; k < kOut; k++) {
    const t = Math.floor(rnd() * members.length);
    if (members[t].group !== members[i].group) addEdge(i, t);
  }
}
// Guarantee at least one bridge in each ordered group direction.
for (let g = 0; g < GROUPS.length; g++) {
  for (let h = 0; h < GROUPS.length; h++) {
    if (g === h) continue;
    const has = edges.some((e) => members[e.a].group === g && members[e.b].group === h);
    if (!has) {
      addEdge(byGroup[g][Math.floor(rnd() * byGroup[g].length)],
              byGroup[h][Math.floor(rnd() * byGroup[h].length)]);
    }
  }
}

// ---------- radii from in-degree (followers) ----------
const indeg = new Array(members.length).fill(0);
for (const e of edges) indeg[e.b]++;
const nodes = members.map((m, i) => ({
  id: i,
  label: m.handle,
  community: m.group,
  radius: Math.min(2.4, Math.round((0.55 + 0.17 * indeg[i]) * 100) / 100),
  image: `avatars/${m.handle}.png`,
}));

const data = {
  meta: {
    title: "架空コミュニティのフォロー関係 — アバター画像サンプル",
    description:
      "実在しないオンラインコミュニティのフォロー関係を描いたサンプルです。54人のメンバーが4つのグループ（ゲーム実況／歌ってみた／イラスト／雑談）に分かれ、矢印はフォローの向き、ノードの大きさはフォロワー数に対応します。ズームすると各ノードにアバター画像が表示されます（既定では円形に切り抜いて表示され、ページの設定で四角いまま表示することもできます）。",
    attribution:
      "アバター画像: Miniavs by Webpixels（CC BY 4.0、https://creativecommons.org/licenses/by/4.0/）— DiceBear API（https://www.dicebear.com/styles/miniavs）で生成",
  },
  directed: true,
  nodes,
  edges,
};

// ---------- download avatars + write ----------
const here = dirname(fileURLToPath(import.meta.url));
const facesDir = new URL("../site/faces/", `file://${here}/`);
const avatarsDir = new URL("avatars/", facesDir);
await mkdir(avatarsDir, { recursive: true });

let totalBytes = 0;
const queue = [...members];
async function worker() {
  while (queue.length) {
    const m = queue.shift();
    const url = `https://api.dicebear.com/9.x/miniavs/png?seed=${encodeURIComponent(m.handle)}&size=256`;
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`avatar download failed for ${m.handle}: HTTP ${resp.status}`);
    const buf = Buffer.from(await resp.arrayBuffer());
    if (buf.length < 100 || buf[0] !== 0x89 || buf[1] !== 0x50) {
      throw new Error(`avatar for ${m.handle} is not a PNG (${buf.length} bytes)`);
    }
    await writeFile(new URL(`${m.handle}.png`, avatarsDir), buf);
    totalBytes += buf.length;
  }
}
await Promise.all(Array.from({ length: 6 }, worker));

const outPath = new URL("faces-data.json", facesDir);
await writeFile(outPath, JSON.stringify(data, null, 1) + "\n");
const jsonBytes = (await stat(outPath)).size;
console.log(`nodes=${nodes.length} edges=${edges.length} avatars=${members.length}`);
console.log(`avatar bytes total=${totalBytes}, faces-data.json=${jsonBytes}`);
console.log(`groups: ${GROUPS.map((g) => g.name).join(" / ")}`);
