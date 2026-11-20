// Layout bounds reduction (camera fit): centroid and maximum
// distance from the centroid over the live positions, computed
// entirely on the GPU in three passes over one bind group. The
// host reads back only the 16-byte stats block — positions never
// leave the GPU for a camera fit.
//
//   bounds_sum    one workgroup per 256 nodes reduces its slice
//                 of position sums into `partials` (workgroup tree
//                 reduction in shared memory).
//   bounds_reduce a single workgroup sums the partials and writes
//                 the centroid into stats[0..1] (and zeroes the
//                 max slot for the next pass).
//   bounds_max    every node computes its distance to the
//                 centroid; the exact maximum is reduced with
//                 atomicMax over the f32 bit pattern, which is
//                 order-preserving for the non-negative values a
//                 distance takes (the same trick as the overlap
//                 measure pass).
//
// The three passes run as separate compute passes in one encoder,
// so cross-pass storage visibility needs no explicit barriers.

struct BoundsParams {
    count: u32,
    /// Workgroups dispatched for bounds_sum (= partial slots).
    groups: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> pos: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> partials: array<vec2<f32>>;
/// stats[0] = bitcast<f32> of centroid.x, stats[1] = bitcast<f32>
/// of centroid.y (written by bounds_reduce), stats[2] =
/// bitcast<f32> of the exact maximum distance from the centroid
/// (bounds_max), stats[3] unused.
@group(0) @binding(2) var<storage, read_write> stats: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> bparams: BoundsParams;

var<workgroup> sdata: array<vec2<f32>, 256>;

fn reduce_shared(lid: u32) {
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (lid < s) {
            sdata[lid] += sdata[lid + s];
        }
        workgroupBarrier();
    }
}

@compute @workgroup_size(256)
fn bounds_sum(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    var p = vec2<f32>(0.0, 0.0);
    if (gid.x < bparams.count) {
        p = pos[gid.x].xy;
    }
    sdata[lid.x] = p;
    reduce_shared(lid.x);
    if (lid.x == 0u) {
        partials[wid.x] = sdata[0];
    }
}

@compute @workgroup_size(256)
fn bounds_reduce(@builtin(local_invocation_id) lid: vec3<u32>) {
    var acc = vec2<f32>(0.0, 0.0);
    for (var i = lid.x; i < bparams.groups; i += 256u) {
        acc += partials[i];
    }
    sdata[lid.x] = acc;
    reduce_shared(lid.x);
    if (lid.x == 0u) {
        let c = sdata[0] / f32(bparams.count);
        atomicStore(&stats[0], bitcast<u32>(c.x));
        atomicStore(&stats[1], bitcast<u32>(c.y));
        atomicStore(&stats[2], 0u);
        atomicStore(&stats[3], 0u);
    }
}

@compute @workgroup_size(256)
fn bounds_max(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= bparams.count) {
        return;
    }
    let c = vec2<f32>(
        bitcast<f32>(atomicLoad(&stats[0])),
        bitcast<f32>(atomicLoad(&stats[1])),
    );
    let d = distance(pos[gid.x].xy, c);
    atomicMax(&stats[2], bitcast<u32>(d));
}
