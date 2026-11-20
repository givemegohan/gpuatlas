// Node-image atlas generation for the gpuatlas wgpu viewer (chunk 2).
//
// Ported from the WebGL2 version (../app/main.js): a 4096x4096 canvas
// holding a 32x32 grid of 128 px icons (1,024 distinct icons). Each
// icon is a filled disc in a deterministic hue with a white border
// ring and a two-letter code. Node i uses icon (i % 1024) unless the
// node object carries an explicit `image` cell index — one atlas
// cannot hold 30,000 unique images, so icons repeat across nodes;
// paging atlas slots by visibility is future work.

export const ATLAS_GRID = 32;
export const ATLAS_CELL = 128;
export const ATLAS_COUNT = ATLAS_GRID * ATLAS_GRID;
export const ATLAS_DIM = ATLAS_GRID * ATLAS_CELL; // 4096

/// Draw the atlas on a 2D canvas. Returns { bytes, canvas }: the
/// RGBA pixels (rows top-first) for upload to the GPU, and the canvas
/// itself (verification pages read individual cells back from it).
export function buildAtlas() {
  const cv = document.createElement("canvas");
  cv.width = ATLAS_DIM;
  cv.height = ATLAS_DIM;
  const ctx = cv.getContext("2d");
  const letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  for (let i = 0; i < ATLAS_COUNT; i++) {
    const col = i % ATLAS_GRID, row = Math.floor(i / ATLAS_GRID);
    const cx = col * ATLAS_CELL + ATLAS_CELL / 2;
    const cy = row * ATLAS_CELL + ATLAS_CELL / 2;
    const hue = (i * 137.508) % 360;
    ctx.beginPath();
    ctx.arc(cx, cy, 56, 0, Math.PI * 2);
    ctx.fillStyle = `hsl(${hue}, 65%, 48%)`;
    ctx.fill();
    ctx.lineWidth = 9;
    ctx.strokeStyle = "#ffffff";
    ctx.stroke();
    const code = letters[Math.floor(i / 26) % 26] + letters[i % 26];
    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 46px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(code, cx, cy + 2);
  }
  const img = ctx.getImageData(0, 0, ATLAS_DIM, ATLAS_DIM);
  return { bytes: new Uint8Array(img.data.buffer, img.data.byteOffset, img.data.byteLength), canvas: cv };
}

/// Per-node atlas cell indices: node.image when the data carries an
/// explicit cell, else i % ATLAS_COUNT.
export function imageCellsFor(nodes) {
  const cells = new Uint32Array(nodes.length);
  for (let i = 0; i < nodes.length; i++) {
    cells[i] = nodes[i].image !== undefined ? nodes[i].image : i % ATLAS_COUNT;
  }
  return cells;
}

/// The two-letter code drawn in cell `i` (matches buildAtlas).
export function cellCode(i) {
  const letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  return letters[Math.floor(i / 26) % 26] + letters[i % 26];
}

// ---------------- real images ----------------
//
// Real images are baked into their atlas cells as FULL SQUARES
// (cover-fit, no clip, no ring — see renderImageCellPixelsSquare).
// The circular look is applied at render time instead: the node
// fragment shader's `crop_circle` uniform (on by default) clips
// the sampled cell to a disc and draws the white ring in-shader,
// reproducing the shape the generated icons carry baked-in.
// Turning the crop off shows images as plain squares. That mode
// is display-only: overlap handling always works on the radius
// alone, whatever the crop setting. (renderImageCellPixels below
// is the historical circular bake, kept for reference/tests.)

/// Load an image element for `url`. Resolves to the element, or
/// to null on any failure (network, decode, timeout) — callers
/// fall back to the generated icon. crossOrigin="anonymous" is
/// tried first so CORS-enabled hosts yield an untainted canvas;
/// if that fails the plain load is tried, and a tainted source is
/// still rejected later, when the cell renderer reads the pixels
/// back (getImageData throws on a tainted canvas).
export function loadImageElement(url, timeoutMs = 12000) {
  const attempt = (anonymous) => new Promise((resolve) => {
    const img = new Image();
    if (anonymous) img.crossOrigin = "anonymous";
    const timer = setTimeout(() => { img.onload = img.onerror = null; resolve(null); }, timeoutMs);
    img.onload = () => { clearTimeout(timer); resolve(img); };
    img.onerror = () => { clearTimeout(timer); resolve(null); };
    img.src = url;
  });
  return attempt(true).then((img) => img || attempt(false));
}

let cellCanvas = null;

/// Render a loaded image into one atlas cell's pixels: cover-fit,
/// clipped to a circle of radius 60 (the generated icons' disc
/// spans ~60.5 including their ring), with the same white ring
/// stroked on top. Returns a Uint8ClampedArray of
/// ATLAS_CELL*ATLAS_CELL*4 bytes, or null when the source is
/// unusable (zero size, or a canvas tainted by a non-CORS
/// cross-origin image — getImageData throws there).
export function renderImageCellPixels(img) {
  if (!img || !img.naturalWidth || !img.naturalHeight) return null;
  if (!cellCanvas) {
    cellCanvas = document.createElement("canvas");
    cellCanvas.width = ATLAS_CELL;
    cellCanvas.height = ATLAS_CELL;
  }
  const ctx = cellCanvas.getContext("2d", { willReadFrequently: true });
  const c = ATLAS_CELL / 2;
  ctx.clearRect(0, 0, ATLAS_CELL, ATLAS_CELL);
  ctx.save();
  ctx.beginPath();
  ctx.arc(c, c, 60, 0, Math.PI * 2);
  ctx.clip();
  const scale = Math.max(ATLAS_CELL / img.naturalWidth, ATLAS_CELL / img.naturalHeight);
  const w = img.naturalWidth * scale, h = img.naturalHeight * scale;
  ctx.drawImage(img, c - w / 2, c - h / 2, w, h);
  ctx.restore();
  ctx.beginPath();
  ctx.arc(c, c, 56, 0, Math.PI * 2);
  ctx.lineWidth = 9;
  ctx.strokeStyle = "#ffffff";
  ctx.stroke();
  try {
    return ctx.getImageData(0, 0, ATLAS_CELL, ATLAS_CELL).data;
  } catch {
    return null;
  }
}

/// Render a loaded image into one atlas cell's pixels as a full
/// square: cover-fit with NO circular clip and NO ring. The
/// circle treatment lives in the renderer's fragment shader
/// (the `crop_circle` uniform), so toggling between circular and
/// square node images never needs a re-bake. Returns a
/// Uint8ClampedArray of ATLAS_CELL*ATLAS_CELL*4 bytes (alpha 255
/// everywhere), or null when the source is unusable (zero size,
/// or a canvas tainted by a non-CORS cross-origin image —
/// getImageData throws there).
export function renderImageCellPixelsSquare(img) {
  if (!img || !img.naturalWidth || !img.naturalHeight) return null;
  if (!cellCanvas) {
    cellCanvas = document.createElement("canvas");
    cellCanvas.width = ATLAS_CELL;
    cellCanvas.height = ATLAS_CELL;
  }
  const ctx = cellCanvas.getContext("2d", { willReadFrequently: true });
  const c = ATLAS_CELL / 2;
  ctx.clearRect(0, 0, ATLAS_CELL, ATLAS_CELL);
  const scale = Math.max(ATLAS_CELL / img.naturalWidth, ATLAS_CELL / img.naturalHeight);
  const w = img.naturalWidth * scale, h = img.naturalHeight * scale;
  ctx.drawImage(img, c - w / 2, c - h / 2, w, h);
  try {
    return ctx.getImageData(0, 0, ATLAS_CELL, ATLAS_CELL).data;
  } catch {
    return null;
  }
}

/// Copy one rendered cell (from renderImageCellPixels or
/// renderImageCellPixelsSquare) into an atlas pixel buffer laid
/// out like buildAtlas' bytes (ATLAS_DIM square, rows top-first).
export function writeCellPixels(bytes, cell, pixels) {
  const col = (cell % ATLAS_GRID) * ATLAS_CELL;
  const row = Math.floor(cell / ATLAS_GRID) * ATLAS_CELL;
  const rowBytes = ATLAS_CELL * 4;
  for (let y = 0; y < ATLAS_CELL; y++) {
    const dst = ((row + y) * ATLAS_DIM + col) * 4;
    bytes.set(pixels.subarray(y * rowBytes, (y + 1) * rowBytes), dst);
  }
}
