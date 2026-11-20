// Data loading for the gpuatlas wgpu viewer (usability chunk).
//
// Four input formats normalize into one internal model; everything
// here runs once at load time, so format support costs nothing at
// runtime. The viewer (index.html) turns the model into the typed
// arrays the Rust side consumes.
//
// Model:
//   {
//     nodes: [{
//       id: string,            // source id, kept for reference
//       label: string,
//       community: number,     // color group when no explicit color
//       color: [r, g, b]|null, // explicit color, components 0..1
//       radius: number|undefined,
//       x: number|undefined, y: number|undefined, // file positions
//       image: number|undefined, // atlas cell index (generated icon)
//       imageUrl: string|null,   // real image URL to load, if any
//     }],
//     edges: [{
//       a: number, b: number,  // node indices, a -> b when directed
//       strength: number,      // FA2 weight; negative = repulsive
//       width: number,         // device px when unlinked
//       type: string|null,
//       directed: boolean,
//     }],
//     types: [string],         // edge types, first-appearance order
//     directed: boolean,       // true when any edge is directed
//     meta: {title, description, attribution}|null, // native JSON
//   }
//
// An `image` value means one of three things:
//   - a number: an explicit cell index into the generated-icon
//     atlas;
//   - a non-empty string: a real image URL. The model carries it
//     as `imageUrl`; the viewer loads it into the atlas (resolved
//     against the data file's URL when the data itself came from
//     a URL) and falls back to a generated icon when the URL
//     cannot be resolved or the image fails to load;
//   - true: "has an image" — a deterministic generated icon
//     (hash of the node id).
// However a real image gets into the atlas, it is baked as a
// full square and clipped to a circle at render time: circular
// is the default node shape, and the page's crop toggle can
// switch images to plain squares instead. Either way the mode
// is display-only — polygonal node shapes are not supported,
// and overlap handling works on the radius alone, as for every
// other node.

/// Categorical palette for edge types, as hex strings for the
/// on-page legend. MUST mirror TYPE_PALETTE in
/// src/shaders/render.wgsl (same order, same colors).
export const TYPE_PALETTE_HEX = [
  "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
  "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#4e79a7", "#f28e2b",
];

export function typeColorHex(typeIndex) {
  return TYPE_PALETTE_HEX[typeIndex % TYPE_PALETTE_HEX.length];
}

/// Deterministic atlas cell for an image marker value (FNV-1a).
export function hashCell(text) {
  let h = 0x811c9dc5;
  const s = String(text);
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0) % 1024;
}

function hexColor(text) {
  const m = /^#?([0-9a-f]{6})$/i.exec(String(text).trim());
  if (!m) return null;
  const v = parseInt(m[1], 16);
  return [((v >> 16) & 255) / 255, ((v >> 8) & 255) / 255, (v & 255) / 255];
}

function anyColor(value) {
  if (Array.isArray(value) && value.length >= 3) {
    return [Number(value[0]), Number(value[1]), Number(value[2])];
  }
  if (typeof value === "string") {
    const hex = hexColor(value);
    if (hex) return hex;
    const parts = value.split(",").map((p) => Number(p.trim()));
    if (parts.length >= 3 && parts.every((p) => Number.isFinite(p))) {
      // "r,g,b" with components in 0..255 (a 0..1 triple also
      // passes through unchanged when every component is <= 1).
      return parts.slice(0, 3).map((p) => (p > 1 ? p / 255 : p));
    }
  }
  return null;
}

function truthy(value) {
  if (value === true) return true;
  if (typeof value === "number") return value !== 0;
  if (typeof value === "string") {
    return ["true", "1", "yes", "directed"].includes(value.trim().toLowerCase());
  }
  return false;
}

function num(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

/// Shared tail of every parser: collect the type list in
/// first-appearance order and the graph-level directed flag.
function finalize(nodes, edges) {
  const types = [];
  const seen = new Set();
  let directed = false;
  for (const e of edges) {
    if (e.type != null && !seen.has(e.type)) {
      seen.add(e.type);
      types.push(e.type);
    }
    if (e.directed) directed = true;
  }
  return { nodes, edges, types, directed, meta: null };
}

function imageCellOf(marker, fallbackKey) {
  if (marker === undefined || marker === null || marker === false || marker === "") {
    return undefined;
  }
  if (typeof marker === "number") return marker;
  return hashCell(marker === true ? fallbackKey : marker);
}

/// Split a raw `image` value into the model's two fields: a string
/// is an image URL; anything else resolves to a generated-icon
/// cell (or undefined) exactly as imageCellOf always did.
function imageFields(marker, fallbackKey) {
  if (typeof marker === "string" && marker !== "") {
    return { image: undefined, imageUrl: marker };
  }
  return { image: imageCellOf(marker, fallbackKey), imageUrl: null };
}

/// The optional top-level `meta` object of a native JSON file:
/// {title, description, attribution} strings for the page caption.
/// Anything else (including bench-data.json's statistics object,
/// which has none of these keys) yields null.
function sanitizeMeta(raw) {
  if (!raw || typeof raw !== "object") return null;
  const s = (v) => (typeof v === "string" && v.trim() !== "" ? v : null);
  const meta = { title: s(raw.title), description: s(raw.description), attribution: s(raw.attribution) };
  return meta.title || meta.description || meta.attribution ? meta : null;
}

// ---------------- native JSON ----------------
//
// The schema bench-data.json already uses, plus optional fields:
//   { "directed": false,                       // optional default
//     "nodes": [{ "id": 0, "label": "name",    // id/label optional
//                 "community": 0,              // color group
//                 "color": [r,g,b] | "#rrggbb", // explicit color
//                 "radius": 1.2, "x": 10, "y": -4,
//                 "image": 12 | "avatars/a.png" | true }],
//     "meta": { "title": "...", "description": "...",
//               "attribution": "..." },        // optional caption
//     "edges": [[a, b], [a, b, strength, width],   // index form
//               { "a": 0, "b": 1, "strength": 1, "width": 1,
//                 "type": "kind", "directed": true }] }  // object form
//               // object endpoints may be node ids or indices
export function parseNativeJSON(obj) {
  if (!obj || !Array.isArray(obj.nodes) || !Array.isArray(obj.edges)) {
    throw new Error("native JSON: top-level nodes/edges arrays are required");
  }
  const nodes = obj.nodes.map((raw, i) => {
    const n = raw && typeof raw === "object" ? raw : {};
    const id = n.id !== undefined ? String(n.id) : String(i);
    return {
      id,
      label: n.label !== undefined ? String(n.label) : id,
      community: num(n.community, 0),
      color: anyColor(n.color),
      radius: n.radius !== undefined ? num(n.radius, undefined) : undefined,
      x: n.x !== undefined ? num(n.x, undefined) : undefined,
      y: n.y !== undefined ? num(n.y, undefined) : undefined,
      ...imageFields(n.image, id),
    };
  });
  const byId = new Map(nodes.map((n, i) => [n.id, i]));
  const resolve = (v) => {
    if (typeof v === "number" && Number.isInteger(v) && v >= 0 && v < nodes.length && !byId.has(String(v))) {
      return v;
    }
    const hit = byId.get(String(v));
    if (hit === undefined) throw new Error(`native JSON: unknown node reference ${JSON.stringify(v)}`);
    return hit;
  };
  const defaultDirected = truthy(obj.directed);
  const edges = obj.edges.map((raw) => {
    if (Array.isArray(raw)) {
      return {
        a: resolve(raw[0]), b: resolve(raw[1]),
        strength: raw.length >= 3 ? num(raw[2], 1) : 1,
        width: raw.length >= 4 ? num(raw[3], 1) : 1,
        type: null, directed: defaultDirected,
      };
    }
    const e = raw && typeof raw === "object" ? raw : {};
    const a = e.a !== undefined ? e.a : e.source;
    const b = e.b !== undefined ? e.b : e.target;
    return {
      a: resolve(a), b: resolve(b),
      strength: num(e.strength !== undefined ? e.strength : e.weight, 1),
      width: num(e.width, 1),
      type: e.type !== undefined && e.type !== null ? String(e.type) : null,
      directed: e.directed !== undefined ? truthy(e.directed) : defaultDirected,
    };
  });
  const model = finalize(nodes, edges);
  model.meta = sanitizeMeta(obj.meta);
  return model;
}

// ---------------- XML helpers (GEXF / GraphML) ----------------

function parseXml(text, format) {
  const doc = new DOMParser().parseFromString(text, "application/xml");
  const err = doc.querySelector("parsererror");
  if (err) throw new Error(`${format}: XML parse error: ${err.textContent.slice(0, 120)}`);
  return doc;
}

/// All descendant elements with the given local name (namespace
/// prefix agnostic: matches viz:size, size, ... alike).
function byTag(root, tag) {
  return [...root.getElementsByTagName("*")].filter((el) => el.localName === tag);
}

function childTags(el, tag) {
  return [...el.children].filter((c) => c.localName === tag);
}

// ---------------- GEXF ----------------
//
// Gephi's format. Mappings: edge weight -> strength; viz:size ->
// radius (size / 2); viz:color -> node color; viz:position -> x/y;
// a node attribute titled group/community/modularity_class ->
// community; a node attribute titled image -> image marker; edge
// attributes titled type/width/weight map likewise. Edge
// directedness: the edge's type="directed" attribute, else the
// graph's defaultedgetype.
export function parseGEXF(text) {
  const doc = parseXml(text, "GEXF");
  const graph = byTag(doc, "graph")[0];
  if (!graph) throw new Error("GEXF: no <graph> element");
  const graphDirected = (graph.getAttribute("defaultedgetype") || "undirected") === "directed";

  // Attribute declarations: id -> title, per class.
  const titles = { node: new Map(), edge: new Map() };
  for (const attrs of byTag(doc, "attributes")) {
    const cls = attrs.getAttribute("class") || "node";
    if (!titles[cls]) continue;
    for (const a of childTags(attrs, "attribute")) {
      titles[cls].set(a.getAttribute("id"), a.getAttribute("title") || a.getAttribute("id"));
    }
  }
  const attvalues = (el, cls) => {
    const out = new Map();
    for (const wrap of childTags(el, "attvalues")) {
      for (const av of childTags(wrap, "attvalue")) {
        const title = titles[cls].get(av.getAttribute("for")) || av.getAttribute("for");
        out.set(title, av.getAttribute("value"));
      }
    }
    return out;
  };

  const communityOf = new Map();
  const communityIndex = (v) => {
    if (!communityOf.has(v)) communityOf.set(v, communityOf.size);
    return communityOf.get(v);
  };

  const nodes = [];
  const byId = new Map();
  for (const el of byTag(doc, "node")) {
    const id = el.getAttribute("id");
    const at = attvalues(el, "node");
    let community = 0;
    for (const key of ["group", "community", "modularity_class"]) {
      if (at.has(key)) { community = communityIndex(at.get(key)); break; }
    }
    let color = null, radius, x, y;
    for (const viz of el.children) {
      if (viz.localName === "color") {
        color = [
          num(viz.getAttribute("r"), 0) / 255,
          num(viz.getAttribute("g"), 0) / 255,
          num(viz.getAttribute("b"), 0) / 255,
        ];
      } else if (viz.localName === "size") {
        radius = num(viz.getAttribute("value"), NaN) / 2;
        if (!Number.isFinite(radius)) radius = undefined;
      } else if (viz.localName === "position") {
        x = num(viz.getAttribute("x"), undefined);
        y = num(viz.getAttribute("y"), undefined);
      }
    }
    byId.set(id, nodes.length);
    nodes.push({
      id: String(id),
      label: el.getAttribute("label") || String(id),
      community,
      color,
      radius,
      x, y,
      ...imageFields(at.has("image") ? at.get("image") : undefined, String(id)),
    });
  }
  const resolve = (v) => {
    const hit = byId.get(String(v));
    if (hit === undefined) throw new Error(`GEXF: unknown node id ${JSON.stringify(v)}`);
    return hit;
  };
  const edges = [];
  for (const el of byTag(doc, "edge")) {
    const at = attvalues(el, "edge");
    const t = el.getAttribute("type");
    edges.push({
      a: resolve(el.getAttribute("source")),
      b: resolve(el.getAttribute("target")),
      strength: num(el.getAttribute("weight") ?? at.get("weight"), 1),
      width: num(at.get("width"), 1),
      type: at.has("type") ? String(at.get("type")) : null,
      directed: t ? t === "directed" : graphDirected,
    });
  }
  return finalize(nodes, edges);
}

// ---------------- GraphML ----------------
//
// Keys resolve through their attr.name (falling back to the key id
// itself). Node data names: label, color ("#rrggbb" or "r,g,b"),
// radius, size (-> radius / 2), x, y, image, community/group.
// Edge data names: weight, width, type, directed. The graph's
// edgedefault gives the default directedness.
export function parseGraphML(text) {
  const doc = parseXml(text, "GraphML");
  const keyName = new Map();
  for (const k of byTag(doc, "key")) {
    keyName.set(k.getAttribute("id"), k.getAttribute("attr.name") || k.getAttribute("id"));
  }
  const graph = byTag(doc, "graph")[0];
  if (!graph) throw new Error("GraphML: no <graph> element");
  const graphDirected = (graph.getAttribute("edgedefault") || "undirected") === "directed";
  const dataOf = (el) => {
    const out = new Map();
    for (const d of childTags(el, "data")) {
      out.set(keyName.get(d.getAttribute("key")) || d.getAttribute("key"), d.textContent.trim());
    }
    return out;
  };
  const communityOf = new Map();
  const communityIndex = (v) => {
    if (!communityOf.has(v)) communityOf.set(v, communityOf.size);
    return communityOf.get(v);
  };
  const nodes = [];
  const byId = new Map();
  for (const el of byTag(graph, "node")) {
    const id = el.getAttribute("id");
    const d = dataOf(el);
    let radius;
    if (d.has("radius")) radius = num(d.get("radius"), undefined);
    else if (d.has("size")) { const s = num(d.get("size"), NaN); radius = Number.isFinite(s) ? s / 2 : undefined; }
    let community = 0;
    for (const key of ["community", "group"]) {
      if (d.has(key)) { community = communityIndex(d.get(key)); break; }
    }
    byId.set(id, nodes.length);
    nodes.push({
      id: String(id),
      label: d.get("label") || String(id),
      community,
      color: d.has("color") ? anyColor(d.get("color")) : null,
      radius,
      x: d.has("x") ? num(d.get("x"), undefined) : undefined,
      y: d.has("y") ? num(d.get("y"), undefined) : undefined,
      ...imageFields(d.has("image") ? d.get("image") : undefined, String(id)),
    });
  }
  const resolve = (v) => {
    const hit = byId.get(String(v));
    if (hit === undefined) throw new Error(`GraphML: unknown node id ${JSON.stringify(v)}`);
    return hit;
  };
  const edges = [];
  for (const el of byTag(graph, "edge")) {
    const d = dataOf(el);
    edges.push({
      a: resolve(el.getAttribute("source")),
      b: resolve(el.getAttribute("target")),
      strength: num(d.get("weight"), 1),
      width: num(d.get("width"), 1),
      type: d.has("type") ? d.get("type") : null,
      directed: d.has("directed") ? truthy(d.get("directed")) : graphDirected,
    });
  }
  return finalize(nodes, edges);
}

// ---------------- CSV edge list ----------------
//
// Header row required: source,target[,weight,width,type,directed]
// (from/to are accepted for source/target). Nodes are inferred in
// first-appearance order; a node's label is its id. Quoted fields
// (with "" escapes) are supported.
export function parseCSV(text) {
  const rows = parseCsvRows(text).filter((r) => r.length > 1 || (r.length === 1 && r[0].trim() !== ""));
  if (rows.length < 2) throw new Error("CSV: a header row and at least one edge row are required");
  const header = rows[0].map((h) => h.trim().toLowerCase());
  const col = (...names) => header.findIndex((h) => names.includes(h));
  const ciSource = col("source", "from");
  const ciTarget = col("target", "to");
  if (ciSource < 0 || ciTarget < 0) throw new Error("CSV: header must contain source and target columns");
  const ciWeight = col("weight", "strength");
  const ciWidth = col("width");
  const ciType = col("type");
  const ciDirected = col("directed");
  const nodes = [];
  const byId = new Map();
  const nodeIndex = (id) => {
    if (!byId.has(id)) {
      byId.set(id, nodes.length);
      nodes.push({
        id, label: id, community: 0, color: null,
        radius: undefined, x: undefined, y: undefined, image: undefined, imageUrl: null,
      });
    }
    return byId.get(id);
  };
  const cell = (row, ci) => (ci >= 0 && ci < row.length ? row[ci].trim() : "");
  const edges = [];
  for (const row of rows.slice(1)) {
    const s = cell(row, ciSource), t = cell(row, ciTarget);
    if (s === "" || t === "") continue;
    edges.push({
      a: nodeIndex(s), b: nodeIndex(t),
      strength: ciWeight >= 0 && cell(row, ciWeight) !== "" ? num(cell(row, ciWeight), 1) : 1,
      width: ciWidth >= 0 && cell(row, ciWidth) !== "" ? num(cell(row, ciWidth), 1) : 1,
      type: ciType >= 0 && cell(row, ciType) !== "" ? cell(row, ciType) : null,
      directed: ciDirected >= 0 ? truthy(cell(row, ciDirected)) : false,
    });
  }
  return finalize(nodes, edges);
}

function parseCsvRows(text) {
  const rows = [];
  let row = [], field = "", quoted = false;
  const src = String(text).replace(/^\uFEFF/, "");
  for (let i = 0; i < src.length; i++) {
    const c = src[i];
    if (quoted) {
      if (c === '"') {
        if (src[i + 1] === '"') { field += '"'; i++; }
        else quoted = false;
      } else field += c;
    } else if (c === '"') {
      quoted = true;
    } else if (c === ",") {
      row.push(field); field = "";
    } else if (c === "\n" || c === "\r") {
      if (c === "\r" && src[i + 1] === "\n") i++;
      row.push(field); field = "";
      rows.push(row); row = [];
    } else field += c;
  }
  if (field !== "" || row.length > 0) { row.push(field); rows.push(row); }
  return rows;
}

// ---------------- dispatch ----------------

/// Parse graph data in any supported format. The format is chosen
/// by file extension when a name is given, else by sniffing the
/// content.
export function parseGraphText(text, filename = "") {
  const name = filename.toLowerCase();
  if (name.endsWith(".gexf")) return parseGEXF(text);
  if (name.endsWith(".graphml") || name.endsWith(".xml")) {
    return text.includes("<gexf") ? parseGEXF(text) : parseGraphML(text);
  }
  if (name.endsWith(".csv")) return parseCSV(text);
  if (name.endsWith(".json")) return parseNativeJSON(JSON.parse(text));
  const head = text.slice(0, 512).trimStart();
  if (head.startsWith("{")) return parseNativeJSON(JSON.parse(text));
  if (head.startsWith("<")) {
    return text.includes("<gexf") ? parseGEXF(text) : parseGraphML(text);
  }
  return parseCSV(text);
}

/// Index of an edge's type in model.types, or -1 when untyped —
/// the value the renderer's edge attribute buffer carries.
export function typeIndexOf(model, edge) {
  return edge.type == null ? -1 : model.types.indexOf(edge.type);
}
