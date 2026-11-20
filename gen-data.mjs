// Benchmark graph generator: 30,000 nodes, community structure.
// Deterministic (mulberry32, seed 42). ~90k edges target (avg degree ~6).
import { writeFileSync } from 'node:fs';

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rand = mulberry32(42);

const N = 30000;
const COMMUNITIES = 120;
// Community sizes: log-normal-ish spread
const sizes = [];
let total = 0;
for (let i = 0; i < COMMUNITIES; i++) {
  const s = Math.max(30, Math.round(250 * Math.exp((rand() - 0.5) * 1.6)));
  sizes.push(s); total += s;
}
// Normalize to exactly N
let acc = 0;
const nodes = []; // {id, community}
for (let c = 0; c < COMMUNITIES; c++) {
  const s = c === COMMUNITIES - 1 ? N - acc : Math.round((sizes[c] / total) * N);
  for (let i = 0; i < s && acc < N; i++, acc++) nodes.push({ id: acc, community: c });
}
while (nodes.length < N) nodes.push({ id: nodes.length, community: COMMUNITIES - 1 });

const byComm = new Map();
for (const n of nodes) {
  if (!byComm.has(n.community)) byComm.set(n.community, []);
  byComm.get(n.community).push(n.id);
}

const edgeSet = new Set();
const edges = [];
function addEdge(a, b) {
  if (a === b) return;
  const key = a < b ? a * N + b : b * N + a;
  if (edgeSet.has(key)) return;
  edgeSet.add(key);
  edges.push([a, b]);
}
// Intra-community: each node links to ~5 others in its community (preferential-ish: pick from a small hub pool + random)
for (const [c, members] of byComm) {
  const hubs = members.slice(0, Math.max(3, Math.floor(members.length * 0.05)));
  for (const id of members) {
    const k = 2 + Math.floor(rand() * 4); // 2-5 intra edges per node
    for (let i = 0; i < k; i++) {
      const target = rand() < 0.45
        ? hubs[Math.floor(rand() * hubs.length)]
        : members[Math.floor(rand() * members.length)];
      addEdge(id, target);
    }
  }
}
// Inter-community bridges: ~4% of edges
const bridgeTarget = Math.floor(edges.length * 0.04);
for (let i = 0; i < bridgeTarget; i++) {
  addEdge(Math.floor(rand() * N), Math.floor(rand() * N));
}

writeFileSync(new URL('./bench-data.json', import.meta.url), JSON.stringify({
  meta: { nodes: nodes.length, edges: edges.length, communities: COMMUNITIES, seed: 42 },
  nodes, edges,
}));
console.log('nodes', nodes.length, 'edges', edges.length, 'communities', COMMUNITIES);
