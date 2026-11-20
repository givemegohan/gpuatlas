// ForceAtlas-2-style compute passes (WGSL).
//
// The force math extends the WebGL2 version (gpuatlas/app) with the
// chunk-1 data model (per-edge strength; see FEATURE_REQUESTS.md):
//   repulsion:  F += delta * (scaling * mi * mj / max(|delta|^2, 0.25))
//   attraction: F += (pj - pi) * s_k        (linear, per-CSR-entry
//               strength s_k; a negative strength therefore pushes the
//               pair apart along the edge — a "negative edge")
//   gravity:    F += -pi * (gravity * mi * 0.1)
//   integrate:  factor = speed / (1 + sqrt(|F - F_prev|))
//               disp = F * factor, clamped to max_step; pos += disp
//
// A negative edge in the plain linear form would push with a force
// that grows linearly with distance, so a repelled pair accelerates
// away without bound (measured: a strength -6 pair placed 20 units
// apart ended ~9,300 units apart after 300 iterations, wrecking the
// layout scale). Negative strengths therefore use a saturating form
// in the integrate pass: the force magnitude along the edge is
// |s| * d / max(d, 1), i.e. constant |s| beyond 1 unit. The pair is
// still repelled, but the force cannot run away. Positive strengths
// keep the exact linear form, so default-strength layouts (and the
// recorded benchmarks) are unchanged.
//
// Buffers:
//   positions are vec4<f32> = (x, y, mass, 1). Two position buffers are
//   ping-ponged: a pass reads `pos_in` and the integrate pass writes
//   `pos_out`; the host swaps the two bind groups each iteration.
//
// The repulsion pass uses workgroup shared-memory tiling: each workgroup
// cooperatively loads a tile of 256 positions once and every invocation
// reuses it for its 256 pair interactions. The naive version (one global
// load per pair, as in the fragment-shader version) reads the whole
// position array once per node (~480 KB x 30,000); the tiled version
// reads it once per workgroup (~480 KB x 118), a ~256x reduction in
// global memory traffic.

struct Params {
    count: u32,
    scaling: f32,
    gravity: f32,
    speed: f32,
    max_step: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0) var<storage, read> pos_in: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> forces: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> pos_out: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> prev_forces: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> offsets: array<u32>;
@group(0) @binding(5) var<storage, read> neighbors: array<u32>;
@group(0) @binding(6) var<uniform> params: Params;
// Per-CSR-entry edge strength, aligned with `neighbors` (binding 5).
// Both directions of an undirected edge carry the same strength.
@group(0) @binding(7) var<storage, read> edge_strength: array<f32>;

const WG: u32 = 256u;

var<workgroup> tile: array<vec4<f32>, 256>;

@compute @workgroup_size(256)
fn repulsion(@builtin(global_invocation_id) gid: vec3<u32>,
             @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = gid.x;
    let n = params.count;
    let valid = i < n;
    // Out-of-range invocations still participate in barriers and loads.
    let self_pos = select(vec4<f32>(0.0), pos_in[i], valid);
    var f = vec2<f32>(0.0, 0.0);
    let tiles = (n + WG - 1u) / WG;
    for (var t = 0u; t < tiles; t++) {
        let j = t * WG + lid.x;
        // Out-of-range tile slots get mass 0, which contributes zero force.
        tile[lid.x] = select(vec4<f32>(0.0), pos_in[j], j < n);
        workgroupBarrier();
        if (valid) {
            for (var k = 0u; k < WG; k++) {
                let o = tile[k];
                let delta = self_pos.xy - o.xy;
                let d2 = max(dot(delta, delta), 0.25);
                // The self pair (j == i) has delta == 0, so it adds zero
                // and needs no explicit skip (same result as the WebGL2
                // version, which skipped it explicitly).
                f += delta * (params.scaling * self_pos.z * o.z / d2);
            }
        }
        workgroupBarrier();
    }
    if (valid) {
        forces[i] = vec4<f32>(f, 0.0, 0.0);
    }
}

@compute @workgroup_size(256)
fn integrate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) {
        return;
    }
    let self_pos = pos_in[i];
    let pi = self_pos.xy;
    let mi = self_pos.z;
    var f = forces[i].xy;
    let start = offsets[i];
    let end = offsets[i + 1u];
    for (var k = start; k < end; k++) {
        let j = neighbors[k];
        let delta_e = pos_in[j].xy - pi;
        let s = edge_strength[k];
        if (s < 0.0) {
            // Saturating repulsion along the edge (see header note):
            // magnitude |s| * d / max(d, 1), direction away from j.
            f += delta_e * (s / max(length(delta_e), 1.0));
        } else {
            f += delta_e * s;
        }
    }
    f += -pi * (params.gravity * mi * 0.1);
    let prev = prev_forces[i].xy;
    let swing = length(f - prev);
    let factor = params.speed / (1.0 + sqrt(swing));
    var disp = f * factor;
    let dl = length(disp);
    if (dl > params.max_step) {
        disp *= params.max_step / dl;
    }
    pos_out[i] = vec4<f32>(pi + disp, mi, 1.0);
    prev_forces[i] = vec4<f32>(f, 0.0, 0.0);
}

// Untiled variant: every invocation reads the position array directly
// from global memory (the same access pattern as the WebGL2 fragment
// shader, but in a compute pipeline). Kept as an A/B reference: on real
// GPUs the tiled version above should win by ~256x less global traffic,
// while on software Vulkan the barrier/shared-memory emulation can
// cost more than it saves.
@compute @workgroup_size(256)
fn repulsion_direct(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = params.count;
    if (i >= n) {
        return;
    }
    let self_pos = pos_in[i];
    var f = vec2<f32>(0.0, 0.0);
    for (var j = 0u; j < n; j++) {
        let o = pos_in[j];
        let delta = self_pos.xy - o.xy;
        let d2 = max(dot(delta, delta), 0.25);
        f += delta * (params.scaling * self_pos.z * o.z / d2);
    }
    forces[i] = vec4<f32>(f, 0.0, 0.0);
}
