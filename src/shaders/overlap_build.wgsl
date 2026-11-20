// Overlap-resolution grid build (chunk 3, FEATURE_REQUESTS.md
// item 7): the four passes that (re)build the hashed spatial grid
// consumed by the measure / resolve passes in overlap.wgsl.
//
// This is a separate shader module with its own bind group because
// WebGPU guarantees only 8 storage buffers per shader stage
// (max_storage_buffers_per_shader_stage); the build group needs 6
// and the consume group needs 8, and a single combined group of 9
// was rejected at layout creation. See RESULTS.md chunk 3.
//
// Grid layout (counting sort by hashed cell):
//   grid_clear: zero the bucket counts.
//   grid_count: each node records its cell in `cell_of` and bumps
//     its bucket count.
//   grid_scan:  exclusive prefix sums of the counts into
//     `grid_starts` (one workgroup), seeding `grid_cursors`.
//   grid_fill:  each node atomically claims the next slot in its
//     bucket and writes its index into `grid_entries`.
// Cell size = 2 * (max radius * radius_scale), set by the host in
// the shared uniform block, so any overlapping pair lies in the
// same or an adjacent cell. Cells are hashed into a fixed
// power-of-two bucket table; collisions only cost extra distance
// checks downstream, because entries carry their exact cell.

struct OverlapParams {
    count: u32,
    buckets: u32, // power of two; used as a mask
    cell_size: f32,
    relax: f32,
    radius_scale: f32,
    tol: f32, // pair-count threshold (measure only; see overlap.wgsl)
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0) var<storage, read> pos_in: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> grid_counts: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> grid_starts: array<u32>;
@group(0) @binding(3) var<storage, read_write> grid_cursors: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> grid_entries: array<u32>;
@group(0) @binding(5) var<storage, read_write> cell_of: array<vec2<i32>>;
@group(0) @binding(6) var<uniform> oparams: OverlapParams;

const WG: u32 = 256u;

fn cell_of_pos(p: vec2<f32>) -> vec2<i32> {
    return vec2<i32>(floor(p / oparams.cell_size));
}

fn hash_cell(c: vec2<i32>) -> u32 {
    var h = u32(c.x) * 374761393u + u32(c.y) * 668265263u;
    h = (h ^ (h >> 13u)) * 1274126177u;
    return (h ^ (h >> 16u)) & (oparams.buckets - 1u);
}

@compute @workgroup_size(256)
fn grid_clear(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < oparams.buckets) {
        atomicStore(&grid_counts[gid.x], 0u);
    }
}

@compute @workgroup_size(256)
fn grid_count(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= oparams.count) {
        return;
    }
    let c = cell_of_pos(pos_in[i].xy);
    cell_of[i] = c;
    atomicAdd(&grid_counts[hash_cell(c)], 1u);
}

var<workgroup> scan_partial: array<u32, 256>;

@compute @workgroup_size(256)
fn grid_scan(@builtin(local_invocation_id) lid: vec3<u32>) {
    let buckets = oparams.buckets;
    let chunk = (buckets + WG - 1u) / WG;
    let lo = lid.x * chunk;
    let hi = min(lo + chunk, buckets);
    var sum = 0u;
    for (var b = lo; b < hi; b++) {
        sum += atomicLoad(&grid_counts[b]);
    }
    scan_partial[lid.x] = sum;
    workgroupBarrier();
    // In-place scan of the 256 partial sums (each value is read
    // before its slot is overwritten).
    if (lid.x == 0u) {
        var run = 0u;
        for (var t = 0u; t < WG; t++) {
            let s = scan_partial[t];
            scan_partial[t] = run;
            run += s;
        }
    }
    workgroupBarrier();
    var run = scan_partial[lid.x];
    for (var b = lo; b < hi; b++) {
        grid_starts[b] = run;
        atomicStore(&grid_cursors[b], run);
        run += atomicLoad(&grid_counts[b]);
    }
}

@compute @workgroup_size(256)
fn grid_fill(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= oparams.count) {
        return;
    }
    let slot = atomicAdd(&grid_cursors[hash_cell(cell_of[i])], 1u);
    grid_entries[slot] = i;
}
