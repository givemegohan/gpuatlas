// Overlap measure / resolve passes (chunk 3, FEATURE_REQUESTS.md
// item 7): the consumers of the hashed spatial grid built by
// overlap_build.wgsl.
//
// Guarantee-oriented counterpart to the force layout: the FA2-style
// passes in compute.wgsl ignore the per-node radii (chunk-1 data
// model), so a settled layout can still have discs on top of each
// other. These passes remove overlaps *measurably*: the measure
// kernel reduces the exact maximum overlap and the number of
// overlapping pairs on the GPU, and the host loop in sim.rs repeats
// correction rounds until the measured maximum overlap is at or
// below the tolerance (or a round cap is hit, which is reported,
// not hidden).
//
// Correction (`resolve_step`) is Jacobi-style: every node
// accumulates pushes from its own position snapshot, so no atomics
// are needed and the result is deterministic. For an overlapping
// pair the push magnitude is the overlap (r_i + r_j - d), split by
// inverse mass (the lighter node moves more), and the accumulated
// correction is applied scaled by `relax` (2.0 — an
// over-relaxation; see OVERLAP_RELAX in sim.rs for the measured
// reason) and clamped to one cell size per round so a node buried
// in a dense cluster cannot be flung across the layout in a
// single round.
// Exactly coincident nodes (d == 0, no connecting axis) are
// separated along a deterministic direction derived from the
// unordered index pair: the lower index moves along +u, the higher
// along -u.
//
// Measurement (`measure`) counts each pair overlapping by more
// than the tolerance once (from the lower index) and reduces the
// exact maximum overlap with atomicMax over the f32 bit pattern,
// which is order-preserving for the non-negative values overlaps
// take.
//
// Probing: with cell size = 2 * (max radius * radius_scale), any
// pair closer than r_i + r_j lies in the same or an adjacent cell,
// so probing the 3x3 neighborhood of a node's cell is exhaustive.
// Buckets may contain entries from foreign cells (hash collisions);
// entries carry their exact cell (`cell_of`) and probes skip
// mismatches, so collisions cost time, never correctness.
//
// Effective radius everywhere here is radii[i] * radius_scale, the
// same value that scales the discs on screen (render.wgsl), so the
// guarantee covers what the viewer actually draws.
//
// This module has its own bind group (8 storage buffers, the
// WebGPU per-stage guarantee) separate from the grid-build group
// in overlap_build.wgsl (6); a combined group of 9 storage buffers
// exceeded max_storage_buffers_per_shader_stage and was rejected.

struct OverlapParams {
    count: u32,
    buckets: u32, // power of two; used as a mask
    cell_size: f32,
    relax: f32,
    radius_scale: f32,
    tol: f32, // pair-count threshold (measure only)
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0) var<storage, read> pos_in: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> pos_out: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> radii: array<f32>;
@group(0) @binding(3) var<storage, read> grid_counts: array<u32>;
@group(0) @binding(4) var<storage, read> grid_starts: array<u32>;
@group(0) @binding(5) var<storage, read> grid_entries: array<u32>;
@group(0) @binding(6) var<storage, read> cell_of: array<vec2<i32>>;
/// stats[0] = bitcast<u32> of the exact maximum overlap,
/// stats[1] = number of pairs overlapping by more than `tol`
/// (the guarantee threshold: a converged state has count 0 while
/// its exact maximum, also reported, may be anywhere in (0, tol]).
/// Zeroed by the host before each measure.
@group(0) @binding(7) var<storage, read_write> stats: array<atomic<u32>>;
@group(0) @binding(8) var<uniform> oparams: OverlapParams;

fn hash_cell(c: vec2<i32>) -> u32 {
    var h = u32(c.x) * 374761393u + u32(c.y) * 668265263u;
    h = (h ^ (h >> 13u)) * 1274126177u;
    return (h ^ (h >> 16u)) & (oparams.buckets - 1u);
}

@compute @workgroup_size(256)
fn measure(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= oparams.count) {
        return;
    }
    let pi = pos_in[i].xy;
    let ri = radii[i] * oparams.radius_scale;
    let ci = cell_of[i];
    var local_max = 0.0;
    var local_pairs = 0u;
    for (var dx = -1; dx <= 1; dx++) {
        for (var dy = -1; dy <= 1; dy++) {
            let c = ci + vec2<i32>(dx, dy);
            let b = hash_cell(c);
            let start = grid_starts[b];
            let end = start + grid_counts[b];
            for (var s = start; s < end; s++) {
                let j = grid_entries[s];
                // Exact cell match: hash collisions bring foreign
                // entries into the bucket; they are not neighbors.
                if (cell_of[j].x != c.x || cell_of[j].y != c.y) {
                    continue;
                }
                // Count each pair once, from the lower index (the
                // maximum is the same seen from either endpoint).
                if (j <= i) {
                    continue;
                }
                let rj = radii[j] * oparams.radius_scale;
                let ov = ri + rj - distance(pi, pos_in[j].xy);
                if (ov > 0.0) {
                    local_max = max(local_max, ov);
                    if (ov > oparams.tol) {
                        local_pairs += 1u;
                    }
                }
            }
        }
    }
    if (local_max > 0.0) {
        atomicMax(&stats[0], bitcast<u32>(local_max));
    }
    if (local_pairs > 0u) {
        atomicAdd(&stats[1], local_pairs);
    }
}

@compute @workgroup_size(256)
fn resolve_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= oparams.count) {
        return;
    }
    let self_pos = pos_in[i];
    let pi = self_pos.xy;
    let mi = self_pos.z;
    let ri = radii[i] * oparams.radius_scale;
    let ci = cell_of[i];
    var corr = vec2<f32>(0.0, 0.0);
    for (var dx = -1; dx <= 1; dx++) {
        for (var dy = -1; dy <= 1; dy++) {
            let c = ci + vec2<i32>(dx, dy);
            let b = hash_cell(c);
            let start = grid_starts[b];
            let end = start + grid_counts[b];
            for (var s = start; s < end; s++) {
                let j = grid_entries[s];
                if (cell_of[j].x != c.x || cell_of[j].y != c.y) {
                    continue;
                }
                if (j == i) {
                    continue;
                }
                let other = pos_in[j];
                let rj = radii[j] * oparams.radius_scale;
                let rsum = ri + rj;
                let delta = pi - other.xy;
                let d2 = dot(delta, delta);
                if (d2 >= rsum * rsum) {
                    continue;
                }
                var dir = vec2<f32>(0.0, 0.0);
                var ov = 0.0;
                if (d2 > 1e-12) {
                    let d = sqrt(d2);
                    dir = delta / d;
                    ov = rsum - d;
                } else {
                    // Coincident: no axis exists, so derive one
                    // deterministically from the unordered pair.
                    // The lower index takes +u, the higher -u, so
                    // both endpoints agree on the separation axis.
                    let lo = min(i, j);
                    let hi = max(i, j);
                    var h = lo * 374761393u + hi * 668265263u;
                    h = (h ^ (h >> 13u)) * 1274126177u;
                    h = h ^ (h >> 16u);
                    let ang = f32(h) * (6.28318530718 / 4294967296.0);
                    let u = vec2<f32>(cos(ang), sin(ang));
                    dir = select(-u, u, i == lo);
                    ov = rsum;
                }
                // Split the push by inverse mass: the lighter node
                // covers more of the overlap. Both endpoints apply
                // their share, so the pair's separation grows by
                // `relax` times the overlap per round (with
                // relax = 2 an isolated pair overshoots by one
                // overlap-width and is done in a single round).
                let inv_mi = 1.0 / mi;
                let inv_mj = 1.0 / other.z;
                let w = inv_mi / (inv_mi + inv_mj);
                corr += dir * (ov * w);
            }
        }
    }
    var applied = corr * oparams.relax;
    // Per-round displacement cap (one cell = 2 * max effective
    // radius): keeps a node buried in a dense cluster from being
    // flung in a single round; expansion continues over rounds.
    let cl = length(applied);
    if (cl > oparams.cell_size) {
        applied *= oparams.cell_size / cl;
    }
    pos_out[i] = vec4<f32>(pi + applied, self_pos.z, 1.0);
}
