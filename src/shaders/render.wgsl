// Rendering for the gpuatlas wgpu viewer.
//
// Positions never leave the GPU: both vertex shaders read the current
// position storage buffer directly. WebGPU has no gl_PointSize, so
// nodes are drawn as instanced quads (6 vertices from a corner buffer,
// one instance per node) with a soft disc in the fragment shader.
//
// Chunk-1 data model (FEATURE_REQUESTS.md items 1-5):
//   * per-node radius (binding 5): the node disc diameter scales with
//     the node's radius and the radius_scale setting.
//   * per-edge strength and width (binding 4, one vec2 per undirected
//     edge): edges are drawn as instanced quads (two triangles), since
//     WebGPU line primitives are always 1 px. The width is in device
//     pixels (screen space), so an edge keeps its thickness at every
//     zoom level. When `link_width` is set, the rendered width is
//     derived from the strength instead: width_px =
//     clamp(|strength|, 0.25, 16). Otherwise the per-edge width value
//     is used as-is.
//   * negative edges (strength < 0) can be hidden: when
//     `hide_negative` is set, their quads are collapsed to a single
//     clipped-away position in the vertex shader.
//
// Chunk-2 additions (FEATURE_REQUESTS.md items 6 and 8):
//   * node-image LOD: each node carries an atlas cell index
//     (binding 6). When `images_on` is set and the node's on-screen
//     diameter (point_px * radius * radius_scale) is at least
//     `lod_px`, the fragment shader samples the node's icon from the
//     atlas texture (bindings 7 and 8) inside the circle clip;
//     below the threshold the node stays a community-color disc.
//     `lod_px` is a user setting (slider 4-64 px, default 14), not a
//     hard-coded constant. The atlas is a 4096x4096 texture with a
//     32x32 grid of 128 px icons, generated on a 2D canvas exactly as
//     in the WebGL2 version; node i uses cell i % 1024 unless the
//     loader supplied an explicit cell index.
//   * circular crop (later addition): real-image cells are baked
//     as full squares; the `crop_circle` camera uniform (default
//     on) applies the disc clip and white ring in the fragment
//     shader. Off, images render as plain squares — display only;
//     overlap handling always uses the radius alone.
//
// Camera mapping matches the WebGL2 version:
//   clip = (p - center) * view_scale,  view_scale = (2*zoom/w, 2*zoom/h)

struct Camera {
    center: vec2<f32>,
    view_scale: vec2<f32>,
    viewport: vec2<f32>, // device pixels
    point_px: f32,       // node diameter in device px for radius == 1
    radius_scale: f32,   // user multiplier on every node's radius
    link_width: f32,     // 1.0: edge width derived from |strength|
    hide_negative: f32,  // 1.0: do not draw strength < 0 edges
    lod_px: f32,         // image LOD threshold (on-screen diameter, px)
    images_on: f32,      // 1.0: node images enabled (subject to LOD)
    edge_mode: f32,      // 0 fixed color, 1 endpoint color, 2 type color
    edge_color_r: f32,   // fixed edge color (mode 0; fallback in mode 2)
    edge_color_g: f32,
    edge_color_b: f32,
    arrows_on: f32,      // 1.0: draw arrowheads on directed edges
    arrows_all: f32,     // 1.0: treat every edge as directed
    crop_circle: f32,    // 1.0: clip node images to a circle (default);
                         // 0.0: images fill the whole quad as squares
                         // (display only — overlap stays radius-based)
    _pad7: f32,
}

@group(0) @binding(0) var<storage, read> r_pos: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> r_colors: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> r_edges: array<u32>;
@group(0) @binding(3) var<uniform> cam: Camera;
// Per undirected edge: (strength, width, directed, type_index).
// Aligned with r_edges pairs. directed is 1.0 when the edge is
// directed from its first endpoint to its second (render-only; the
// force model stays symmetric). type_index >= 0 selects a palette
// color in edge mode 2; -1 means untyped.
@group(0) @binding(4) var<storage, read> r_edge_attrs: array<vec4<f32>>;
// Per node radius (world units; 1.0 = the pre-chunk-1 node size).
@group(0) @binding(5) var<storage, read> r_radii: array<f32>;
// Per node atlas cell index for the image LOD layer (chunk 2).
@group(0) @binding(6) var<storage, read> r_cells: array<u32>;
// Node-image atlas (4096x4096, 32x32 cells) and its sampler.
@group(0) @binding(7) var atlas: texture_2d<f32>;
@group(0) @binding(8) var atlas_samp: sampler;

// Categorical palette for edge mode 2 (type colors). Twelve
// colors (d3 category10 + two Tableau colors); data.mjs carries
// the same table as hex strings for the on-page legend — keep the
// two in sync.
const TYPE_PALETTE: array<vec3<f32>, 12> = array<vec3<f32>, 12>(
    vec3<f32>(0.122, 0.467, 0.706), // #1f77b4
    vec3<f32>(1.000, 0.498, 0.055), // #ff7f0e
    vec3<f32>(0.173, 0.627, 0.173), // #2ca02c
    vec3<f32>(0.839, 0.153, 0.157), // #d62728
    vec3<f32>(0.580, 0.404, 0.741), // #9467bd
    vec3<f32>(0.549, 0.337, 0.294), // #8c564b
    vec3<f32>(0.890, 0.467, 0.761), // #e377c2
    vec3<f32>(0.498, 0.498, 0.498), // #7f7f7f
    vec3<f32>(0.737, 0.741, 0.133), // #bcbd22
    vec3<f32>(0.090, 0.745, 0.812), // #17becf
    vec3<f32>(0.306, 0.475, 0.655), // #4e79a7
    vec3<f32>(0.949, 0.557, 0.169), // #f28e2b
);

// The color of edge instance `ii` under the current edge color
// mode. Mode 1 uses the source (first) endpoint's node color.
fn edge_rgb(ii: u32, attr: vec4<f32>) -> vec3<f32> {
    if (cam.edge_mode > 1.5) {
        if (attr.w >= 0.0) {
            return TYPE_PALETTE[u32(attr.w) % 12u];
        }
        return vec3<f32>(cam.edge_color_r, cam.edge_color_g, cam.edge_color_b);
    }
    if (cam.edge_mode > 0.5) {
        return r_colors[r_edges[ii * 2u]].rgb;
    }
    return vec3<f32>(cam.edge_color_r, cam.edge_color_g, cam.edge_color_b);
}

// The rendered width of an edge in device px (chunk-1 rules).
fn edge_width_px(attr: vec4<f32>) -> f32 {
    if (cam.link_width > 0.5) {
        return clamp(abs(attr.x), 0.25, 16.0);
    }
    return attr.y;
}

struct EdgeOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec3<f32>,
}

@vertex
fn edge_vs(@builtin(instance_index) ii: u32,
           @location(0) corner: vec2<f32>) -> EdgeOut {
    var out: EdgeOut;
    let attr = r_edge_attrs[ii];
    let strength = attr.x;
    out.color = edge_rgb(ii, attr);
    if (cam.hide_negative > 0.5 && strength < 0.0) {
        // Collapse the quad to a position outside the clip volume.
        out.clip = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        return out;
    }
    let width_px = edge_width_px(attr);
    let pa = r_pos[r_edges[ii * 2u]].xy;
    let pb = r_pos[r_edges[ii * 2u + 1u]].xy;
    // Work in device pixels so the width is screen-space constant.
    let ca = (pa - cam.center) * cam.view_scale;
    let cb = (pb - cam.center) * cam.view_scale;
    let pa_px = (ca * 0.5 + 0.5) * cam.viewport;
    let pb_px = (cb * 0.5 + 0.5) * cam.viewport;
    let axis = pb_px - pa_px;
    let len = length(axis);
    var normal = vec2<f32>(0.0, 0.0);
    if (len > 0.0) {
        normal = vec2<f32>(-axis.y, axis.x) / len;
    }
    // corner.x in {-1, 1} selects the endpoint; corner.y offsets
    // perpendicular to the edge by half the width.
    let t = (corner.x + 1.0) * 0.5;
    let p_px = mix(pa_px, pb_px, t) + normal * (corner.y * width_px * 0.5);
    out.clip = vec4<f32>(p_px / cam.viewport * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

@fragment
fn edge_fs(in: EdgeOut) -> @location(0) vec4<f32> {
    // Alpha matches the historical fixed color (0.16) in every mode.
    return vec4<f32>(in.color, 0.16);
}

// ---- arrowheads for directed edges (usability chunk) ----
//
// A separate instanced draw (3 vertices per edge instance, no
// vertex buffer): vertex 0 is the tip, 1 and 2 the base corners.
// The head sits at the target (second) endpoint, pulled back by
// the target node's on-screen radius so the tip touches the disc
// edge instead of hiding under it. Its length follows the edge's
// rendered width (3.5x, clamped to 7..42 device px). Edges that
// are not directed — or all edges when arrows are off — collapse
// outside the clip volume. Hidden negative edges lose their heads
// together with their quads.
struct ArrowOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec3<f32>,
}

@vertex
fn arrow_vs(@builtin(vertex_index) vid: u32,
            @builtin(instance_index) ii: u32) -> ArrowOut {
    var out: ArrowOut;
    let attr = r_edge_attrs[ii];
    out.color = edge_rgb(ii, attr);
    let directed = attr.z > 0.5 || cam.arrows_all > 0.5;
    if (cam.arrows_on < 0.5 || !directed
        || (cam.hide_negative > 0.5 && attr.x < 0.0)) {
        out.clip = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        return out;
    }
    let a = r_edges[ii * 2u];
    let b = r_edges[ii * 2u + 1u];
    let pa = r_pos[a].xy;
    let pb = r_pos[b].xy;
    let ca = (pa - cam.center) * cam.view_scale;
    let cb = (pb - cam.center) * cam.view_scale;
    let pa_px = (ca * 0.5 + 0.5) * cam.viewport;
    let pb_px = (cb * 0.5 + 0.5) * cam.viewport;
    let axis = pb_px - pa_px;
    let len = length(axis);
    if (len < 1.0) {
        out.clip = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        return out;
    }
    let dir = axis / len;
    let normal = vec2<f32>(-dir.y, dir.x);
    let head_len = clamp(edge_width_px(attr) * 3.5, 7.0, 42.0);
    let head_half = head_len * 0.32;
    let target_r_px = cam.point_px * r_radii[b] * cam.radius_scale * 0.5;
    let tip = pb_px - dir * (target_r_px + 1.0);
    let base = tip - dir * head_len;
    var p_px = tip;
    if (vid == 1u) {
        p_px = base + normal * head_half;
    } else if (vid == 2u) {
        p_px = base - normal * head_half;
    }
    out.clip = vec4<f32>(p_px / cam.viewport * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

@fragment
fn arrow_fs(in: ArrowOut) -> @location(0) vec4<f32> {
    // Nearly opaque: a head is a small marker and must stay
    // findable at a glance, unlike the translucent edge quads.
    return vec4<f32>(in.color, 0.85);
}

struct NodeOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) local: vec2<f32>,
    @location(2) @interpolate(flat) cell: u32,
    @location(3) use_image: f32,
    // The crop flag travels as a varying (like use_image): the
    // Camera uniform's bind-group entry is vertex-stage only, so
    // the fragment shader must not read `cam` directly.
    @location(4) @interpolate(flat) crop_circle: f32,
}

@vertex
fn node_vs(@builtin(instance_index) ii: u32,
           @location(0) corner: vec2<f32>) -> NodeOut {
    let p = r_pos[ii].xy;
    var out: NodeOut;
    let center_clip = (p - cam.center) * cam.view_scale;
    // corner is in [-1, 1]; 1.0 corresponds to the disc radius.
    // NDC spans 2 per viewport dimension, so a diameter of D device
    // px is D / viewport in the offset units used here.
    let diam = max(cam.point_px * r_radii[ii] * cam.radius_scale, 1.0);
    let offset = corner * diam / cam.viewport;
    out.clip = vec4<f32>(center_clip + offset, 0.0, 1.0);
    out.color = r_colors[ii].rgb;
    out.local = corner;
    out.cell = r_cells[ii];
    // Image LOD: this node's own on-screen diameter decides, so
    // large nodes switch to their icon before small ones do.
    out.use_image = select(0.0, 1.0, cam.images_on > 0.5 && diam >= cam.lod_px);
    out.crop_circle = cam.crop_circle;
    return out;
}

@fragment
fn node_fs(in: NodeOut) -> @location(0) vec4<f32> {
    let d = length(in.local);
    if (in.use_image > 0.5) {
        // Sample this node's atlas cell. The atlas rows are stored
        // top-first (canvas order) and texture v = 0 is the top
        // row, while local.y = +1 is the top of the quad on
        // screen, so the y axis is flipped to keep the icon
        // upright.
        let cell = vec2<f32>(f32(in.cell % 32u), f32(in.cell / 32u));
        let uv = (cell + vec2<f32>(in.local.x * 0.5 + 0.5,
                                   1.0 - (in.local.y * 0.5 + 0.5))) / 32.0;
        // textureSampleLevel, not textureSample: the implicit-
        // derivative form is illegal inside this per-fragment
        // (non-uniform) branch, and the atlas has a single mip
        // level anyway.
        let tex = textureSampleLevel(atlas, atlas_samp, uv, 0.0);
        if (in.crop_circle > 0.5) {
            // Circular crop (default). Real-image cells are baked
            // as full squares, so the disc edge and the white ring
            // are drawn here: the ring band matches the canvas-
            // baked geometry (stroke at r = 56/64, width 9/64 in
            // local units, outer edge at 60.5/64) and the alpha
            // mask clips at that same outer edge. The ring is
            // composited OVER the image with its own alpha, as
            // the canvas bake strokes it on top — a plain color
            // mix would lose the ring wherever the source art is
            // transparent (avatar illustrations). Generated icons
            // carry the identical ring and disc alpha baked in, so
            // this treatment lands on an already-matching shape
            // for them.
            let clip = 1.0 - smoothstep(0.925, 0.955, d);
            let band = smoothstep(0.795, 0.820, d)
                * (1.0 - smoothstep(0.930, 0.950, d));
            let a_img = tex.a * clip;
            let a_ring = band * clip;
            let a = a_ring + a_img * (1.0 - a_ring);
            if (a <= 0.0) {
                discard;
            }
            let rgb = (vec3<f32>(1.0, 1.0, 1.0) * a_ring
                + tex.rgb * a_img * (1.0 - a_ring)) / a;
            return vec4<f32>(rgb, a);
        }
        // Square mode (crop off, display only): plain sample over
        // the whole quad — real images show as full squares.
        // Generated icons keep their disc look because their cells
        // are transparent outside the baked disc.
        if (tex.a <= 0.0) {
            discard;
        }
        return tex;
    }
    let a = smoothstep(1.0, 0.72, d);
    if (a <= 0.0) {
        discard;
    }
    return vec4<f32>(in.color, a);
}
