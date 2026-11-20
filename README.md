# gpuatlas — GPU-resident network graph viewer (Rust + WebGPU)

gpuatlas renders and simulates network graphs of up to ~30,000 nodes in the
browser. The force layout (ForceAtlas-2-style repulsion, attraction and
gravity) runs entirely on the GPU: node positions live in GPU buffers and
are never read back to the CPU per frame. The implementation is Rust +
wgpu (WebGPU), compiled to WebAssembly with a small JavaScript front end.

## Live demo

- Viewer with the bundled benchmark graph (30,000 nodes / 105,260 edges):
  https://muse.givemegohan.xyz/gpuatlas/
- "Faces" sample — a small fictional community whose nodes carry real
  avatar images: https://muse.givemegohan.xyz/gpuatlas/?data=faces/faces-data.json

## Features

- GPU-resident direct N-body layout (measured 60 fps at 30,000 nodes on a
  desktop GPU)
- Per-node radius; per-edge strength and width (linked or independent);
  negative edges acting as repulsion, with an option to hide them
- Node images with level of detail: nodes switch between plain discs and
  images based on their on-screen size (threshold is adjustable)
- Real images on nodes: `image` URLs in the data are loaded into a texture
  atlas at load time; images are circular-cropped with a white ring by
  default, and can optionally be shown square (display only — simulation
  and overlap handling always use circles)
- Edge type coloring with a legend; arrowheads for directed edges
- Measured overlap resolution: node circles are pushed apart until the
  measured maximum overlap is at most 0.001 (the measured value is shown)
- Settle mode (compute without drawing until the layout converges), FPS
  cap, parameter presets
- Lossless SVG export of the current layout (every node and edge included;
  images are recorded as atlas cell references)
- Four input formats: native JSON, GEXF, GraphML, CSV edge list

## Quick start (demo page)

WebGPU does not work from `file://` URLs, so serve the folder over HTTP:

```
# in this folder (the one containing this README)
python3 -m http.server 8000
# then open http://localhost:8000/index.html
```

You need a browser with WebGPU enabled (recent Chrome / Edge). The bundled
benchmark graph loads by default. To view your own data, use the
"ファイルを開く" (open file) button or drag & drop a file onto the page;
`samples/` contains a small graph in three formats. A different data file
can also be selected with the `?data=<url>` query parameter, e.g.
`index.html?data=faces/faces-data.json`.

## Build from source

Requirements: a Rust toolchain and
[wasm-pack](https://rustwasm.github.io/wasm-pack/).

```
./build.sh
```

This produces `pkg/` (the `.wasm` binary and the JavaScript glue) via
`wasm-pack build --release --target web`, and also builds the native
headless benchmark (`target/release/bench`). `build.sh` sets
`--remap-path-prefix` so the shipped `.wasm` contains no local absolute
paths. `pkg/` is a build artifact and is not committed to this repository.

## Usage as an npm package

The `pkg/` directory produced by the build is a ready-to-publish npm
package (`gpuatlas`). After installing it:

```js
import init, { create_viewer, create_headless } from "gpuatlas";

await init(); // loads the .wasm
// create_viewer(canvas, ...) starts the interactive viewer;
// create_headless(...) runs the simulation without a canvas.
```

`index.html` in this repository is the reference integration: it wires the
viewer to the control bar, the file loader (`data.mjs`) and the image
atlas (`atlas.mjs`).

## Controls

- Drag: pan / Mouse wheel: zoom around the cursor / Pinch: zoom + pan
  around the midpoint
- "全体表示" (fit view): frames the whole current layout; the layout's
  centroid and extent are measured on the GPU. After each data load the
  view re-fits automatically at ~1 s and ~3.5 s while the layout
  contracts; any manual camera move cancels the auto-fit for that data
- +/− buttons: zoom around the screen center; the % readout is relative
  to the fitted zoom (100% = fitted)
- Sliders and presets for Scaling / Gravity / speed; node radius scale;
  edge width linked to |strength| or independent; hide negative edges
- Image LOD threshold; circular crop on/off for node images
- FPS cap (unlimited / 60 / 30 / 15); settle-until-converged mode; SVG
  export; overlap resolution (one-shot or per-frame)
- Background color; edge color modes (fixed / endpoint color / type
  color); arrowheads for directed edges

## Input data formats

Files are loaded via the open-file button or drag & drop. All formats are
converted to an internal model at load time, so runtime speed is
unaffected.

### 1. Native JSON

```json
{
  "nodes": [
    { "id": 0, "label": "A", "community": 0 },
    { "id": 1, "label": "B", "community": 0 },
    { "id": 2, "label": "C", "community": 1 }
  ],
  "edges": [
    [0, 1],
    [0, 2, 2.0],
    { "a": 1, "b": 2, "strength": 1, "width": 2, "type": "work", "directed": true }
  ]
}
```

- Node fields (all optional): `id` (default: sequential), `label`,
  `community` (color group index), `color` (`[r, g, b]` in 0..1 or
  `"#rrggbb"`; takes precedence over the community color), `radius`
  (default: derived from degree), `x` / `y` (initial position; default:
  random), `image` (a number selects a generated icon cell, a string is a
  real image URL, `true` assigns a generated icon deterministically)
- Edges are either arrays `[source, target, strength, width]` (last two
  optional) or objects `{a, b, strength, width, type, directed}`
  (`source`/`target` are accepted instead of `a`/`b`; endpoints may be
  node `id`s)
- `strength` is the attraction weight; negative values repel (the force
  saturates with distance). `width` is the on-screen width in px. `type`
  names the edge type, used by the type-color mode and the legend.
  `directed: true` marks a directed edge (direction is visual only; the
  force computation treats edges as undirected). A top-level
  `"directed": true` sets the default for all edges
- An optional top-level `meta` object (`title`, `description`,
  `attribution`) is shown as a caption on the page
- The bundled `bench-data.json` (30,000 nodes) is a large example of this
  format; it is generated deterministically by `gen-data.mjs`

### 2. GEXF (Gephi's standard format)

- Edge `weight` → strength, `viz:size` → radius (half the value),
  `viz:color` → node color, `viz:position` → initial position
- Node attribute `group` / `community` / `modularity_class` → color group
- Node attribute `image` → real image URL (same as a string `image` in
  native JSON)
- Edge attributes `type` / `width`; graph `defaultedgetype` and edge
  `type="directed"` → directed

### 3. GraphML

Mapped by `attr.name`. Nodes: `label`, `color` (`#rrggbb` or `r,g,b`),
`radius`, `size` (radius is half the value), `x`, `y`, `image`,
`community` / `group`. Edges: `weight`, `width`, `type`, `directed`. The
graph's `edgedefault` provides the directed default.

### 4. CSV edge list

A header row is required:

```csv
source,target,weight,width,type,directed
A,B,1,2,work,true
B,C,-1,,conflict,
```

- `source` / `target` (`from` / `to` also accepted) are required; nodes
  are created in order of appearance and named by their label
- `weight`, `width`, `type`, `directed` are optional columns

## Repository layout

```
README.md            this file
LICENSE              MIT license
Cargo.toml           Rust package (library + native bench binary)
src/                 Rust sources and WGSL shaders
build.sh             release build (wasm-pack -> pkg/)
index.html           demo / viewer page
data.mjs             input conversion (JSON / GEXF / GraphML / CSV)
atlas.mjs            node image atlas (generated icons + real images)
bench-data.json      benchmark graph loaded by default (30,000 nodes)
gen-data.mjs         generator for bench-data.json (seeded, deterministic)
samples/             small sample graph (24 nodes / 62 edges) in 3 formats
faces/               avatar sample data (see Attribution below)
make_faces_sample.mjs  generator for the faces sample
```

## Limitations

- Node images are circular by default. Real images are stored square in
  the atlas; the circular crop and white ring are applied while drawing.
  Turning off "画像を円で切り抜く" (circular crop) shows images square.
  This is display-only: polygonal node shapes are not supported, and the
  simulation and overlap handling always use circles (radii) for every
  node, images included
- Real images are shown only while they fit in the atlas (1,024 cells).
  Images beyond the cap and images that fail to load fall back to
  generated icons (a note is shown when the cap is exceeded). For data
  loaded from a local file, relative image URLs cannot be resolved and
  also fall back to generated icons, as do cross-origin images whose
  pixels cannot be read due to CORS
- Export is SVG only; GEXF / GraphML export is not supported (input
  compatibility was prioritized)
- One simulation method (GPU direct N-body). Approximate methods such as
  Barnes-Hut are not included: direct computation measured fast enough.
  Spread is adjusted with the presets and sliders instead
- No 3D view
- Edge direction affects drawing only; forces treat edges as undirected
- The overlap "guarantee" is by measurement: correction runs until the
  measured maximum overlap is ≤ 0.001. In locally dense configurations
  the 200-round cap can be reached, in which case the residual value is
  displayed
- In SVG export, node radii are in world units and edge widths follow
  the on-screen px rule; images are recorded as atlas cell references,
  not embedded

## Sample data and attribution

- `samples/` and `bench-data.json` are synthetic data generated by this
  project (`gen-data.mjs`, mulberry32, seed 42).
- `faces/` is a fictional community dataset (54 members in 4 groups, 162
  directed follow edges) generated by `make_faces_sample.mjs`. The avatar
  images are **Miniavs by Webpixels**, licensed under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), generated
  via the [DiceBear API](https://www.dicebear.com/styles/miniavs). In
  this repository, `faces/faces-data.json` references the DiceBear API
  URLs directly; running `make_faces_sample.mjs` downloads the same
  images into `faces/avatars/` for self-hosting (the live demo is
  self-hosted that way). If you redistribute the avatar PNGs, keep this
  attribution with them (it is also recorded in `faces/faces-data.json`
  under `meta.attribution`).

## License

MIT — see [LICENSE](LICENSE).
