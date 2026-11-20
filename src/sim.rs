//! Core simulation and rendering for the gpuatlas wgpu viewer.
//!
//! Platform-agnostic: the same code drives the WebGPU build (wasm32)
//! and the native headless benchmark. All graph state lives in GPU
//! storage buffers; positions are never read back per frame.
//!
//! The force model extends the WebGL2 version (gpuatlas/app):
//! direct N-body repulsion with a workgroup-tiled compute shader,
//! CSR attraction with per-edge strength (negative = repulsive, in a
//! saturating form; see shaders/compute.wgsl), gravity, and
//! swinging-damped integration. Rendering draws edges as instanced
//! quads with per-edge width and nodes with per-node radius
//! (chunk-1 data model, FEATURE_REQUESTS.md items 1-5). A separate
//! overlap-resolution pass (chunk 3, shaders/overlap.wgsl) removes
//! disc overlaps measurably: it repeats spatial-grid correction
//! rounds until the GPU-measured maximum overlap is at or below
//! [`OVERLAP_TOLERANCE`] (or reports that the round cap was hit).

use bytemuck::{Pod, Zeroable};

/// Integration constants, tuned in the WebGL2 version and kept identical.
pub const SPEED: f32 = 0.1;
pub const MAX_STEP: f32 = 25.0;
pub const DEFAULT_SCALING: f32 = 0.5;
pub const DEFAULT_GRAVITY: f32 = 1.0;

/// Node-image atlas geometry (chunk 2): a square texture of
/// ATLAS_DIM x ATLAS_DIM texels holding a ATLAS_GRID x ATLAS_GRID
/// grid of icons, generated exactly as in the WebGL2 version.
pub const ATLAS_GRID: u32 = 32;
pub const ATLAS_DIM: u32 = 4096;
/// Default image-LOD threshold (on-screen diameter in device px).
pub const DEFAULT_LOD_PX: f32 = 14.0;

/// Overlap resolution (chunk 3): the guarantee tolerance, in world
/// units. `resolve_overlaps` stops as soon as the measured maximum
/// overlap is at or below this value.
pub const OVERLAP_TOLERANCE: f32 = 1e-3;
/// Overlap resolution (chunk 3): round cap for `resolve_overlaps`.
/// Hitting the cap with residual overlap above the tolerance is a
/// reported outcome, not a silent failure.
pub const OVERLAP_MAX_ROUNDS: u32 = 200;
/// Relaxation factor applied to each round's accumulated pairwise
/// corrections (Jacobi update). This is deliberately an
/// over-relaxation: in a dense cluster a node's pushes from its
/// many neighbors largely cancel as vectors, so the classic
/// under-relaxation (0.5) removes overlap only diffusively —
/// measured on the chunk-3 test state (2,000 settled nodes, radii
/// x4) it was still at max 0.43 after 600 rounds, while 2.0
/// converges in 75 rounds (1.0: >525, 1.5: 425; see RESULTS.md).
/// The overshoot is safe for this pass: nothing inside the pass
/// pulls discs back together, an isolated pair simply ends up
/// separated by one extra overlap-width, and the per-round
/// displacement clamp bounds every node's move. Convergence is
/// never assumed from this constant — it is measured every round.
pub const OVERLAP_RELAX: f32 = 2.0;
/// Resolution rounds run per frame when the viewer's per-frame
/// overlap resolution is enabled (chunk 3).
pub const OVERLAP_PER_FRAME_ROUNDS: u32 = 2;
/// Bucket count of the hashed spatial grid (power of two).
pub const OVERLAP_BUCKETS: u32 = 32768;

const WORKGROUP: u32 = 256;

/// Uniform block for the overlap passes; layout must match
/// `struct OverlapParams` in shaders/overlap.wgsl and
/// shaders/overlap_build.wgsl (8 words).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct OverlapParams {
    count: u32,
    buckets: u32,
    cell_size: f32,
    relax: f32,
    radius_scale: f32,
    /// Pair-count threshold for the measure pass (the tolerance of
    /// the run doing the measuring). Unused by the other passes.
    tol: f32,
    _pad1: f32,
    _pad2: f32,
}

/// Uniform block for the layout-bounds passes; layout must match
/// `struct BoundsParams` in shaders/bounds.wgsl (4 words).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BoundsParams {
    count: u32,
    /// Workgroups dispatched for the sum pass (= partial slots).
    groups: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Layout bounds of the live positions (camera fit): the centroid
/// and the exact maximum distance from it, reduced on the GPU by
/// the passes in shaders/bounds.wgsl and read back as three
/// floats. Positions themselves never leave the GPU for this.
#[derive(Clone, Copy, Debug)]
pub struct LayoutBounds {
    pub cx: f32,
    pub cy: f32,
    pub radius: f32,
}

/// Outcome of a measured overlap-resolution run (chunk 3). The
/// before/after values are GPU reductions over all node pairs, so
/// `converged` is a statement about a measurement, not a hope.
/// Pair counts count pairs overlapping by more than the run's
/// tolerance (see `measure_overlap`).
#[derive(Clone, Copy, Debug)]
pub struct OverlapReport {
    /// Exact maximum overlap (world units) before the first round.
    pub before_max: f32,
    /// Pairs overlapping by more than the tolerance, before.
    pub before_pairs: u32,
    /// Exact maximum overlap after the last round.
    pub after_max: f32,
    /// Pairs overlapping by more than the tolerance, after.
    pub after_pairs: u32,
    /// Correction rounds actually run.
    pub rounds: u32,
    /// True when `after_max` is at or below the tolerance.
    pub converged: bool,
}

/// Uniform block for the force passes; layout must match
/// `struct Params` in shaders/compute.wgsl (8 words).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
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

/// Render settings passed to the render shaders each frame. The
/// layout must match `struct Camera` in shaders/render.wgsl exactly
/// (20 floats, 80 bytes; scalar fields only after the three vec2s,
/// so the Rust and WGSL offsets agree one-to-one).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Camera {
    pub center: [f32; 2],
    pub view_scale: [f32; 2],
    /// Viewport size in device pixels.
    pub viewport: [f32; 2],
    /// Node disc diameter in device px for a node of radius 1.
    pub point_px: f32,
    /// Multiplier applied to every node's radius.
    pub radius_scale: f32,
    /// 1.0: edge width is derived from |strength| (see render.wgsl).
    pub link_width: f32,
    /// 1.0: edges with strength < 0 are not drawn.
    pub hide_negative: f32,
    /// Image LOD threshold: nodes whose on-screen diameter reaches
    /// this many device px draw their atlas icon (chunk 2).
    pub lod_px: f32,
    /// 1.0: node images are enabled (still subject to the LOD).
    pub images_on: f32,
    /// Edge color mode (usability chunk): 0 = fixed color
    /// (`edge_color`), 1 = source-endpoint node color, 2 = per-type
    /// palette color (untyped edges fall back to the fixed color).
    pub edge_mode: f32,
    /// Fixed edge color (mode 0, and the fallback in mode 2).
    pub edge_color: [f32; 3],
    /// 1.0: draw arrowheads on directed edges.
    pub arrows_on: f32,
    /// 1.0: treat every edge as directed for arrowheads (used when
    /// the data marks no edge as directed and the user still asks
    /// for arrows).
    pub arrows_all: f32,
    /// 1.0: node images are clipped to a circle in the fragment
    /// shader (default; real-image cells are baked as full squares
    /// and the ring is drawn in-shader). 0.0: images fill the
    /// whole node quad as squares. Display-only — the overlap
    /// passes always work on the radius alone.
    pub crop_circle: f32,
    pub _pad: [f32; 1],
}

const _: () = assert!(std::mem::size_of::<Camera>() == 80);

/// The historical fixed edge color (the WebGL2 version's constant),
/// used as the default for `Camera::edge_color`.
pub const DEFAULT_EDGE_COLOR: [f32; 3] = [0.62, 0.68, 0.78];

impl Camera {
    pub fn new(
        center: [f32; 2],
        view_scale: [f32; 2],
        viewport: [f32; 2],
        point_px: f32,
    ) -> Camera {
        Camera {
            center,
            view_scale,
            viewport,
            point_px,
            radius_scale: 1.0,
            link_width: 0.0,
            hide_negative: 0.0,
            lod_px: DEFAULT_LOD_PX,
            images_on: 0.0,
            edge_mode: 0.0,
            edge_color: DEFAULT_EDGE_COLOR,
            arrows_on: 0.0,
            arrows_all: 0.0,
            crop_circle: 1.0,
            _pad: [0.0; 1],
        }
    }
}

/// Outcome of a settle run (usability chunk): step-only iterations
/// in batches, with the mean per-iteration displacement measured
/// from position readbacks between batches. `converged` is true
/// only when the measured mean fell to the threshold; a cap hit
/// reports the residual value instead.
#[derive(Clone, Copy, Debug)]
pub struct SettleReport {
    /// Iterations actually run.
    pub iterations: u32,
    /// Mean per-iteration displacement (world units) over the last
    /// batch, measured from readbacks.
    pub final_mean_displacement: f32,
    /// True when `final_mean_displacement` is at or below the
    /// threshold the run was given.
    pub converged: bool,
}

/// Host-side graph data, uploaded once at construction.
pub struct GraphData<'a> {
    /// 4 floats per node: x, y, mass, 1.
    pub positions: &'a [f32],
    /// 4 floats per node: r, g, b, 1.
    pub colors: &'a [f32],
    /// Per-node radius (chunk-1 data model), length N. Rendering
    /// scales the node disc by this and the overlap-resolution pass
    /// (chunk 3) separates discs by it; the force model itself does
    /// not use it.
    pub radii: &'a [f32],
    /// CSR offsets, length N + 1.
    pub offsets: &'a [u32],
    /// CSR neighbor indices (both directions), length 2E.
    pub neighbors: &'a [u32],
    /// Per-CSR-entry edge strength, aligned with `neighbors`,
    /// length 2E. May be negative (a repulsive "negative edge").
    pub edge_strengths: &'a [f32],
    /// Undirected edge endpoint index pairs, length 2E.
    pub edge_pairs: &'a [u32],
    /// Per undirected edge attributes, length 4E floats:
    /// (strength, width, directed, type_index). Rendering uses them
    /// for quad width / linked width / negative visibility (chunk
    /// 1), arrowheads (directed: 1.0 marks the edge a -> b as
    /// directed) and the per-type color palette (type_index >= 0;
    /// -1 = untyped) (usability chunk). The strength copy must match
    /// `edge_strengths`. Directedness is render-only: the force
    /// model's CSR stays symmetric.
    pub edge_attrs: &'a [f32],
    /// Per-node atlas cell index for the image LOD layer (chunk 2),
    /// length N. The page loader fills `i % (ATLAS_GRID^2)` unless a
    /// node carries an explicit cell index.
    pub image_cells: &'a [u32],
    /// Atlas RGBA pixels (chunk 2), ATLAS_DIM x ATLAS_DIM x 4 bytes,
    /// rows top-first as drawn on the generating canvas. Required
    /// when rendering is enabled; the native benchmark passes `None`.
    pub atlas: Option<&'a [u8]>,
}

pub struct Sim {
    device: wgpu::Device,
    queue: wgpu::Queue,
    count: u32,
    workgroups: u32,
    /// Number of undirected edges (edge instances drawn per frame).
    edge_count: u32,
    params: Params,
    params_buf: wgpu::Buffer,
    camera_buf: wgpu::Buffer,
    pos: [wgpu::Buffer; 2],
    forces: wgpu::Buffer,
    prev_forces: wgpu::Buffer,
    cur: usize,
    compute_bg: [wgpu::BindGroup; 2],
    repulsion_pipeline: wgpu::ComputePipeline,
    repulsion_direct_pipeline: wgpu::ComputePipeline,
    integrate_pipeline: wgpu::ComputePipeline,
    direct_repulsion: bool,
    // Overlap resolution (chunk 3): the radii buffer (also bound by
    // the render pipelines), the hashed-grid state, and the five
    // passes (clear/count/scan/fill build the grid; measure and
    // resolve_step consume it). Bind groups come in the same two
    // ping-pong variants as the force passes. Buffers the host
    // never reads or writes after construction are underscore-
    // prefixed: those fields exist to keep the buffers alive for
    // the bind groups that reference them (the same convention is
    // used for the layout-bounds partials below).
    _radii_buf: wgpu::Buffer,
    max_radius: f32,
    overlap_params_buf: wgpu::Buffer,
    _grid_counts: wgpu::Buffer,
    _grid_starts: wgpu::Buffer,
    _grid_cursors: wgpu::Buffer,
    _grid_entries: wgpu::Buffer,
    _cell_of: wgpu::Buffer,
    overlap_stats: wgpu::Buffer,
    overlap_bg: [wgpu::BindGroup; 2],
    overlap_build_bg: [wgpu::BindGroup; 2],
    grid_clear_pipeline: wgpu::ComputePipeline,
    grid_count_pipeline: wgpu::ComputePipeline,
    grid_scan_pipeline: wgpu::ComputePipeline,
    grid_fill_pipeline: wgpu::ComputePipeline,
    measure_pipeline: wgpu::ComputePipeline,
    resolve_pipeline: wgpu::ComputePipeline,
    // Layout bounds (camera fit): per-workgroup partial sums, the
    // 16-byte stats block (centroid + max radius as f32 bit
    // patterns), and the three reduction passes. Bind groups come
    // in the same two ping-pong variants as the other passes.
    _bounds_partials: wgpu::Buffer,
    bounds_stats: wgpu::Buffer,
    bounds_bg: [wgpu::BindGroup; 2],
    bounds_sum_pipeline: wgpu::ComputePipeline,
    bounds_reduce_pipeline: wgpu::ComputePipeline,
    bounds_max_pipeline: wgpu::ComputePipeline,
    render: Option<RenderState>,
    /// Clear (background) color for the render pass (usability
    /// chunk); defaults to the historical dark #101216.
    clear_color: wgpu::Color,
    // Host-side copies kept for the SVG export (chunk 2): the GPU
    // buffers are write-only from the host's perspective, and an
    // export must not lose nodes or edges, so the attributes are
    // retained here. Positions are read back at export time only.
    export_colors: Vec<[f32; 3]>,
    export_radii: Vec<f32>,
    /// (a, b, strength, width) per undirected edge.
    export_edges: Vec<(u32, u32, f32, f32)>,
    export_cells: Vec<u32>,
}

struct RenderState {
    render_bg: [wgpu::BindGroup; 2],
    edge_pipeline: wgpu::RenderPipeline,
    node_pipeline: wgpu::RenderPipeline,
    /// Arrowheads for directed edges (usability chunk): one
    /// instanced triangle per edge, generated from the vertex and
    /// instance indices alone (no vertex buffer).
    arrow_pipeline: wgpu::RenderPipeline,
    corner_buf: wgpu::Buffer,
}

fn storage_entry(
    binding: u32,
    stage: wgpu::ShaderStages,
    read_only: bool,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: stage,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32, stage: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: stage,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// wgpu cannot bind a zero-size buffer, so graph arrays that may be
/// empty (an edgeless graph has no CSR entries and no edge pairs)
/// are padded to one element. The padding is never read: CSR ranges
/// are empty and the edge draw count is zero.
fn nonempty(bytes: &[u8]) -> &[u8] {
    if bytes.is_empty() {
        &[0u8; 4]
    } else {
        bytes
    }
}

/// Same padding for the edge-attribute array, whose element is a
/// vec4: the render pipelines index it unconditionally, so the
/// binding must cover at least one full element (16 bytes) even
/// when the edge draw count is zero — a 4-byte pad fails
/// validation and invalidates the whole render pass (an edgeless
/// graph rendered nothing at all; found by test7.html).
fn nonempty16(bytes: &[u8]) -> &[u8] {
    if bytes.len() < 16 {
        &[0u8; 16]
    } else {
        bytes
    }
}

fn buf(
    device: &wgpu::Device,
    label: &str,
    contents: &[u8],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents,
        usage,
    })
}

/// Drive a `map_async` read request on `slice` to completion and
/// report its outcome; the caller then reads the mapped range.
/// On wasm the request completes through the event loop, so this
/// simply awaits it; natively the caller owns the device, so poll
/// until the callback fires. Shared by every readback path
/// (overlap measure, layout bounds, offscreen render, positions).
async fn await_map(device: &wgpu::Device, slice: wgpu::BufferSlice<'_>) -> Result<(), String> {
    let (tx, rx) = futures_channel::oneshot::channel::<Result<(), wgpu::BufferAsyncError>>();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    #[cfg(not(target_arch = "wasm32"))]
    {
        let mut rx = rx;
        loop {
            match rx.try_recv() {
                Ok(Some(Ok(()))) => break,
                Ok(Some(Err(e))) => return Err(format!("map_async failed: {e:?}")),
                // Ok(None): no value yet, keep driving the device.
                Ok(None) => {
                    let _ = device.poll(wgpu::PollType::Poll);
                }
                // Err: the sender was dropped without sending.
                Err(_) => return Err("map_async channel closed".to_string()),
            }
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = device;
        match rx.await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(format!("map_async failed: {e:?}")),
            Err(_) => return Err("map_async channel closed".to_string()),
        }
    }
    Ok(())
}

impl Sim {
    /// `surface_format` enables the render pipelines when `Some`
    /// (web viewer); the native benchmark passes `None`.
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        data: &GraphData,
        scaling: f32,
        gravity: f32,
        surface_format: Option<wgpu::TextureFormat>,
    ) -> Sim {
        let count = data.offsets.len() as u32 - 1;
        assert_eq!(data.positions.len() as u32, count * 4);
        assert_eq!(data.colors.len() as u32, count * 4);
        assert_eq!(data.radii.len() as u32, count);
        let csr_len = *data.offsets.last().unwrap();
        assert_eq!(data.neighbors.len() as u32, csr_len);
        assert_eq!(data.edge_strengths.len() as u32, csr_len);
        let edge_count = data.edge_pairs.len() as u32 / 2;
        assert_eq!(data.edge_attrs.len() as u32, edge_count * 4);
        assert_eq!(data.image_cells.len() as u32, count);
        let workgroups = count.div_ceil(WORKGROUP);
        let storage = wgpu::BufferUsages::STORAGE;
        let init = storage | wgpu::BufferUsages::COPY_DST;

        let pos_a = buf(
            &device,
            "positions A",
            bytemuck::cast_slice(data.positions),
            init | wgpu::BufferUsages::COPY_SRC,
        );
        let pos_b = buf(
            &device,
            "positions B",
            bytemuck::cast_slice(data.positions),
            init | wgpu::BufferUsages::COPY_SRC,
        );
        let empty = vec![0u8; count as usize * 16];
        let forces = buf(&device, "forces", &empty, storage);
        let prev_forces = buf(&device, "prev forces", &empty, storage);
        let offsets_buf = buf(
            &device,
            "csr offsets",
            bytemuck::cast_slice(data.offsets),
            init,
        );
        let neighbors_buf = buf(
            &device,
            "csr neighbors",
            nonempty(bytemuck::cast_slice(data.neighbors)),
            init,
        );
        let colors_buf = buf(&device, "colors", bytemuck::cast_slice(data.colors), init);
        let edges_buf = buf(
            &device,
            "edge pairs",
            nonempty(bytemuck::cast_slice(data.edge_pairs)),
            init,
        );
        let radii_buf = buf(&device, "radii", bytemuck::cast_slice(data.radii), init);
        let edge_strength_buf = buf(
            &device,
            "edge strengths",
            nonempty(bytemuck::cast_slice(data.edge_strengths)),
            init,
        );
        let edge_attr_buf = buf(
            &device,
            "edge attrs",
            nonempty16(bytemuck::cast_slice(data.edge_attrs)),
            init,
        );
        let cells_buf = buf(
            &device,
            "image cells",
            bytemuck::cast_slice(data.image_cells),
            init,
        );

        let params = Params {
            count,
            scaling,
            gravity,
            speed: SPEED,
            max_step: MAX_STEP,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        };
        let params_buf = buf(
            &device,
            "params",
            bytemuck::bytes_of(&params),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let camera_buf = buf(
            &device,
            "camera",
            bytemuck::bytes_of(&Camera::new([0.0, 0.0], [1.0, 1.0], [1.0, 1.0], 3.0)),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        // ---- compute pipelines ----
        let compute_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compute layout"),
            entries: &[
                storage_entry(0, wgpu::ShaderStages::COMPUTE, true),
                storage_entry(1, wgpu::ShaderStages::COMPUTE, false),
                storage_entry(2, wgpu::ShaderStages::COMPUTE, false),
                storage_entry(3, wgpu::ShaderStages::COMPUTE, false),
                storage_entry(4, wgpu::ShaderStages::COMPUTE, true),
                storage_entry(5, wgpu::ShaderStages::COMPUTE, true),
                uniform_entry(6, wgpu::ShaderStages::COMPUTE),
                storage_entry(7, wgpu::ShaderStages::COMPUTE, true),
            ],
        });
        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("compute pipeline layout"),
                bind_group_layouts: &[Some(&compute_layout)],
                immediate_size: 0,
            });
        let compute_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fa2 compute"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/compute.wgsl").into()),
        });
        let make_compute = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&compute_pipeline_layout),
                module: &compute_module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let repulsion_pipeline = make_compute("repulsion");
        let repulsion_direct_pipeline = make_compute("repulsion_direct");
        let integrate_pipeline = make_compute("integrate");

        let pos = [pos_a, pos_b];
        let make_compute_bg = |read: usize, write: usize| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("compute bind group"),
                layout: &compute_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: pos[read].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: forces.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: pos[write].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: prev_forces.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: offsets_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: neighbors_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: edge_strength_buf.as_entire_binding(),
                    },
                ],
            })
        };
        let compute_bg = [make_compute_bg(0, 1), make_compute_bg(1, 0)];

        // ---- overlap resolution (chunk 3) ----
        // Hashed spatial grid + measurement/reduction state. The
        // grid is rebuilt by the passes on every round, so the
        // buffers are plain storage with no host-side contents.
        let max_radius = data.radii.iter().copied().fold(0.0f32, f32::max);
        let overlap_params = OverlapParams {
            count,
            buckets: OVERLAP_BUCKETS,
            cell_size: (2.0 * max_radius).max(1e-6),
            relax: OVERLAP_RELAX,
            radius_scale: 1.0,
            tol: OVERLAP_TOLERANCE,
            _pad1: 0.0,
            _pad2: 0.0,
        };
        let overlap_params_buf = buf(
            &device,
            "overlap params",
            bytemuck::bytes_of(&overlap_params),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let raw = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let empty_u32 = |n: u32| vec![0u8; n as usize * 4];
        let grid_counts = buf(&device, "grid counts", &empty_u32(OVERLAP_BUCKETS), raw);
        let grid_starts = buf(&device, "grid starts", &empty_u32(OVERLAP_BUCKETS), raw);
        let grid_cursors = buf(&device, "grid cursors", &empty_u32(OVERLAP_BUCKETS), raw);
        let grid_entries = buf(&device, "grid entries", &empty_u32(count), raw);
        let cell_of = buf(&device, "cell of", &vec![0u8; count as usize * 8], raw);
        let overlap_stats = buf(
            &device,
            "overlap stats",
            &[0u8; 16],
            raw | wgpu::BufferUsages::COPY_SRC,
        );
        // Two bind groups (grid build / grid consume), not one:
        // WebGPU guarantees only 8 storage buffers per shader
        // stage, and the union of both groups' buffers is 9 — a
        // combined layout was rejected at creation (see RESULTS.md
        // chunk 3). Build needs 6 storage buffers, consume needs 8.
        let overlap_build_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("overlap build layout"),
                entries: &[
                    storage_entry(0, wgpu::ShaderStages::COMPUTE, true),
                    storage_entry(1, wgpu::ShaderStages::COMPUTE, false),
                    storage_entry(2, wgpu::ShaderStages::COMPUTE, false),
                    storage_entry(3, wgpu::ShaderStages::COMPUTE, false),
                    storage_entry(4, wgpu::ShaderStages::COMPUTE, false),
                    storage_entry(5, wgpu::ShaderStages::COMPUTE, false),
                    uniform_entry(6, wgpu::ShaderStages::COMPUTE),
                ],
            });
        let overlap_build_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("overlap build pipeline layout"),
                bind_group_layouts: &[Some(&overlap_build_layout)],
                immediate_size: 0,
            });
        let overlap_build_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("overlap grid build"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/overlap_build.wgsl").into()),
        });
        let make_build = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&overlap_build_pipeline_layout),
                module: &overlap_build_module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let grid_clear_pipeline = make_build("grid_clear");
        let grid_count_pipeline = make_build("grid_count");
        let grid_scan_pipeline = make_build("grid_scan");
        let grid_fill_pipeline = make_build("grid_fill");
        let make_overlap_build_bg = |read: usize| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("overlap build bind group"),
                layout: &overlap_build_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: pos[read].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: grid_counts.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: grid_starts.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: grid_cursors.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: grid_entries.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: cell_of.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: overlap_params_buf.as_entire_binding(),
                    },
                ],
            })
        };
        let overlap_build_bg = [make_overlap_build_bg(0), make_overlap_build_bg(1)];

        let overlap_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("overlap layout"),
            entries: &[
                storage_entry(0, wgpu::ShaderStages::COMPUTE, true),
                storage_entry(1, wgpu::ShaderStages::COMPUTE, false),
                storage_entry(2, wgpu::ShaderStages::COMPUTE, true),
                storage_entry(3, wgpu::ShaderStages::COMPUTE, true),
                storage_entry(4, wgpu::ShaderStages::COMPUTE, true),
                storage_entry(5, wgpu::ShaderStages::COMPUTE, true),
                storage_entry(6, wgpu::ShaderStages::COMPUTE, true),
                storage_entry(7, wgpu::ShaderStages::COMPUTE, false),
                uniform_entry(8, wgpu::ShaderStages::COMPUTE),
            ],
        });
        let overlap_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("overlap pipeline layout"),
                bind_group_layouts: &[Some(&overlap_layout)],
                immediate_size: 0,
            });
        let overlap_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("overlap resolution"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/overlap.wgsl").into()),
        });
        let make_overlap = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&overlap_pipeline_layout),
                module: &overlap_module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let measure_pipeline = make_overlap("measure");
        let resolve_pipeline = make_overlap("resolve_step");
        let make_overlap_bg = |read: usize, write: usize| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("overlap bind group"),
                layout: &overlap_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: pos[read].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: pos[write].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: radii_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: grid_counts.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: grid_starts.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: grid_entries.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: cell_of.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: overlap_stats.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: overlap_params_buf.as_entire_binding(),
                    },
                ],
            })
        };
        let overlap_bg = [make_overlap_bg(0, 1), make_overlap_bg(1, 0)];

        // ---- layout bounds (camera fit) ----
        // Three reduction passes over the live positions (sum ->
        // centroid -> max distance), stats read back as 16 bytes.
        // Only 3 storage buffers, well inside the per-stage limit.
        let bounds_params = BoundsParams {
            count,
            groups: workgroups,
            _pad0: 0,
            _pad1: 0,
        };
        let bounds_params_buf = buf(
            &device,
            "bounds params",
            bytemuck::bytes_of(&bounds_params),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let bounds_partials = buf(
            &device,
            "bounds partials",
            &vec![0u8; workgroups as usize * 8],
            raw,
        );
        let bounds_stats = buf(
            &device,
            "bounds stats",
            &[0u8; 16],
            raw | wgpu::BufferUsages::COPY_SRC,
        );
        let bounds_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bounds layout"),
            entries: &[
                storage_entry(0, wgpu::ShaderStages::COMPUTE, true),
                storage_entry(1, wgpu::ShaderStages::COMPUTE, false),
                storage_entry(2, wgpu::ShaderStages::COMPUTE, false),
                uniform_entry(3, wgpu::ShaderStages::COMPUTE),
            ],
        });
        let bounds_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("bounds pipeline layout"),
                bind_group_layouts: &[Some(&bounds_layout)],
                immediate_size: 0,
            });
        let bounds_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("layout bounds"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/bounds.wgsl").into()),
        });
        let make_bounds = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&bounds_pipeline_layout),
                module: &bounds_module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let bounds_sum_pipeline = make_bounds("bounds_sum");
        let bounds_reduce_pipeline = make_bounds("bounds_reduce");
        let bounds_max_pipeline = make_bounds("bounds_max");
        let make_bounds_bg = |read: usize| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bounds bind group"),
                layout: &bounds_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: pos[read].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: bounds_partials.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: bounds_stats.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: bounds_params_buf.as_entire_binding(),
                    },
                ],
            })
        };
        let bounds_bg = [make_bounds_bg(0), make_bounds_bg(1)];

        // Host-side copies for the SVG export (chunk 2).
        let export_colors: Vec<[f32; 3]> = (0..count as usize)
            .map(|i| {
                [
                    data.colors[i * 4],
                    data.colors[i * 4 + 1],
                    data.colors[i * 4 + 2],
                ]
            })
            .collect();
        let export_radii: Vec<f32> = data.radii.to_vec();
        let export_edges: Vec<(u32, u32, f32, f32)> = (0..edge_count as usize)
            .map(|i| {
                (
                    data.edge_pairs[i * 2],
                    data.edge_pairs[i * 2 + 1],
                    data.edge_attrs[i * 4],
                    data.edge_attrs[i * 4 + 1],
                )
            })
            .collect();
        let export_cells: Vec<u32> = data.image_cells.to_vec();

        // ---- render pipelines (optional) ----
        let render = surface_format.map(|format| {
            // Node-image atlas (chunk 2): uploaded once, sampled by
            // the node fragment shader above the LOD threshold.
            let atlas_pixels = data
                .atlas
                .expect("rendering requires atlas pixels (chunk-2 image LOD)");
            assert_eq!(
                atlas_pixels.len() as u64,
                ATLAS_DIM as u64 * ATLAS_DIM as u64 * 4,
                "atlas pixel buffer must be ATLAS_DIM x ATLAS_DIM RGBA"
            );
            let atlas_tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("node image atlas"),
                size: wgpu::Extent3d {
                    width: ATLAS_DIM,
                    height: ATLAS_DIM,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &atlas_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                atlas_pixels,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(ATLAS_DIM * 4),
                    rows_per_image: Some(ATLAS_DIM),
                },
                wgpu::Extent3d {
                    width: ATLAS_DIM,
                    height: ATLAS_DIM,
                    depth_or_array_layers: 1,
                },
            );
            let atlas_view = atlas_tex.create_view(&wgpu::TextureViewDescriptor::default());
            // LINEAR sampling, no mipmaps: minified sampling must not
            // bleed colors from neighboring cells (as in the WebGL2
            // version).
            let atlas_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("atlas sampler"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                ..Default::default()
            });
            let render_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("render layout"),
                entries: &[
                    storage_entry(0, wgpu::ShaderStages::VERTEX, true),
                    storage_entry(1, wgpu::ShaderStages::VERTEX, true),
                    storage_entry(2, wgpu::ShaderStages::VERTEX, true),
                    uniform_entry(3, wgpu::ShaderStages::VERTEX),
                    storage_entry(4, wgpu::ShaderStages::VERTEX, true),
                    storage_entry(5, wgpu::ShaderStages::VERTEX, true),
                    storage_entry(6, wgpu::ShaderStages::VERTEX, true),
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 8,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });
            let render_pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("render pipeline layout"),
                    bind_group_layouts: &[Some(&render_layout)],
                    immediate_size: 0,
                });
            let render_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("render"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/render.wgsl").into()),
            });
            let target = [Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })];
            // Edges are instanced quads (two triangles per edge) so
            // they can have real thickness; WebGPU line primitives
            // are always 1 px. The corner buffer below is shared with
            // the node pipeline.
            let corner_layout = [Some(wgpu::VertexBufferLayout {
                array_stride: 8,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![0 => Float32x2],
            })];
            let edge_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("edges"),
                layout: Some(&render_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &render_module,
                    entry_point: Some("edge_vs"),
                    compilation_options: Default::default(),
                    buffers: &corner_layout,
                },
                fragment: Some(wgpu::FragmentState {
                    module: &render_module,
                    entry_point: Some("edge_fs"),
                    compilation_options: Default::default(),
                    targets: &target,
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });
            // Arrowheads (usability chunk): a separate pipeline
            // with no vertex buffers — the triangle is generated
            // from @builtin(vertex_index) per edge instance, so the
            // edge quad path stays untouched.
            let arrow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("arrows"),
                layout: Some(&render_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &render_module,
                    entry_point: Some("arrow_vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &render_module,
                    entry_point: Some("arrow_fs"),
                    compilation_options: Default::default(),
                    targets: &target,
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });
            let node_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("nodes"),
                layout: Some(&render_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &render_module,
                    entry_point: Some("node_vs"),
                    compilation_options: Default::default(),
                    buffers: &corner_layout,
                },
                fragment: Some(wgpu::FragmentState {
                    module: &render_module,
                    entry_point: Some("node_fs"),
                    compilation_options: Default::default(),
                    targets: &target,
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });
            let corners: [f32; 12] = [
                -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0,
            ];
            let corner_buf = buf(
                &device,
                "quad corners",
                bytemuck::cast_slice(&corners),
                wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            );
            let make_render_bg = |which: usize| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("render bind group"),
                    layout: &render_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: pos[which].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: colors_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: edges_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: camera_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: edge_attr_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: radii_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: cells_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 7,
                            resource: wgpu::BindingResource::TextureView(&atlas_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 8,
                            resource: wgpu::BindingResource::Sampler(&atlas_sampler),
                        },
                    ],
                })
            };
            RenderState {
                render_bg: [make_render_bg(0), make_render_bg(1)],
                edge_pipeline,
                node_pipeline,
                arrow_pipeline,
                corner_buf,
            }
        });

        Sim {
            device,
            queue,
            count,
            workgroups,
            edge_count,
            params,
            params_buf,
            camera_buf,
            pos,
            forces,
            prev_forces,
            cur: 0,
            compute_bg,
            repulsion_pipeline,
            repulsion_direct_pipeline,
            integrate_pipeline,
            direct_repulsion: false,
            _radii_buf: radii_buf,
            max_radius,
            overlap_params_buf,
            _grid_counts: grid_counts,
            _grid_starts: grid_starts,
            _grid_cursors: grid_cursors,
            _grid_entries: grid_entries,
            _cell_of: cell_of,
            overlap_stats,
            overlap_bg,
            overlap_build_bg,
            grid_clear_pipeline,
            grid_count_pipeline,
            grid_scan_pipeline,
            grid_fill_pipeline,
            measure_pipeline,
            resolve_pipeline,
            _bounds_partials: bounds_partials,
            bounds_stats,
            bounds_bg,
            bounds_sum_pipeline,
            bounds_reduce_pipeline,
            bounds_max_pipeline,
            render,
            clear_color: wgpu::Color {
                r: 0.063,
                g: 0.071,
                b: 0.086,
                a: 1.0,
            },
            export_colors,
            export_radii,
            export_edges,
            export_cells,
        }
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn count(&self) -> u32 {
        self.count
    }

    pub fn set_scaling(&mut self, scaling: f32) {
        self.params.scaling = scaling;
        self.write_params();
    }

    pub fn set_gravity(&mut self, gravity: f32) {
        self.params.gravity = gravity;
        self.write_params();
    }

    /// Integration speed (the FA2 SPEED constant, exposed so the
    /// page's parameter presets can bundle it with scaling/gravity).
    pub fn set_speed(&mut self, speed: f32) {
        self.params.speed = speed;
        self.write_params();
    }

    /// Background (clear) color of the render pass, components in
    /// 0..1, used as-is (the historical default 0.063/0.071/0.086
    /// corresponds to the #101216 the page and the SVG export use).
    pub fn set_clear_color(&mut self, r: f64, g: f64, b: f64) {
        self.clear_color = wgpu::Color { r, g, b, a: 1.0 };
    }

    fn write_params(&self) {
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&self.params));
    }

    /// Replace both position buffers (used by the shuffle control) and
    /// clear the force history, matching the WebGL2 version's
    /// re-upload behavior.
    pub fn reset_positions(&mut self, positions: &[f32]) {
        assert_eq!(positions.len() as u32, self.count * 4);
        let bytes = bytemuck::cast_slice(positions);
        self.queue.write_buffer(&self.pos[0], 0, bytes);
        self.queue.write_buffer(&self.pos[1], 0, bytes);
        let zeros = vec![0u8; self.count as usize * 16];
        self.queue.write_buffer(&self.forces, 0, &zeros);
        self.queue.write_buffer(&self.prev_forces, 0, &zeros);
        self.cur = 0;
    }

    /// Select the untiled repulsion variant (direct global loads, no
    /// workgroup shared memory / barriers). On some software Vulkan
    /// implementations the tiled version's barriers cost more than the
    /// memory traffic they save; the A/B numbers are recorded in
    /// RESULTS.md.
    pub fn set_direct_repulsion(&mut self, direct: bool) {
        self.direct_repulsion = direct;
    }

    /// One full iteration: repulsion pass, then attraction + gravity +
    /// integration pass, submitted as one command buffer. The position
    /// buffers swap roles afterwards.
    pub fn step(&mut self) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fa2 step"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("repulsion"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &self.compute_bg[self.cur], &[]);
            if self.direct_repulsion {
                pass.set_pipeline(&self.repulsion_direct_pipeline);
            } else {
                pass.set_pipeline(&self.repulsion_pipeline);
            }
            pass.dispatch_workgroups(self.workgroups, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("integrate"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &self.compute_bg[self.cur], &[]);
            pass.set_pipeline(&self.integrate_pipeline);
            pass.dispatch_workgroups(self.workgroups, 1, 1);
        }
        self.queue.submit([encoder.finish()]);
        self.cur = 1 - self.cur;
    }

    /// Settle mode (usability chunk): run step-only iterations in
    /// batches of `batch`, measuring the mean per-iteration
    /// displacement from position readbacks between batches, until
    /// it falls to `threshold` (world units per iteration) or
    /// `max_iterations` iterations have run. Nothing is rendered.
    /// The report carries the measured final value either way, so a
    /// cap hit is visible as such. Readbacks make this an
    /// on-demand operation, never a frame-loop one.
    pub async fn settle(
        &mut self,
        batch: u32,
        max_iterations: u32,
        threshold: f32,
    ) -> Result<SettleReport, String> {
        let batch = batch.max(1);
        let mut prev = self.readback_positions().await?;
        let mut total = 0u32;
        let mut mean = f32::INFINITY;
        while total < max_iterations {
            let n = batch.min(max_iterations - total);
            for _ in 0..n {
                self.step();
            }
            total += n;
            let cur = self.readback_positions().await?;
            let mut sum = 0.0f64;
            for i in 0..self.count as usize {
                let dx = (cur[i * 4] - prev[i * 4]) as f64;
                let dy = (cur[i * 4 + 1] - prev[i * 4 + 1]) as f64;
                sum += (dx * dx + dy * dy).sqrt();
            }
            mean = (sum / self.count as f64 / n as f64) as f32;
            prev = cur;
            if mean <= threshold {
                break;
            }
        }
        Ok(SettleReport {
            iterations: total,
            final_mean_displacement: mean,
            converged: mean <= threshold,
        })
    }

    // ---- overlap resolution (chunk 3) ----

    /// Refresh the overlap uniform block for a given radius scale
    /// and pair-count threshold. The grid cell size tracks the
    /// largest effective radius: cell = 2 * max_radius *
    /// radius_scale guarantees that any overlapping pair is in the
    /// same or an adjacent cell.
    fn write_overlap_params(&self, radius_scale: f32, tol: f32) {
        let params = OverlapParams {
            count: self.count,
            buckets: OVERLAP_BUCKETS,
            cell_size: (2.0 * self.max_radius * radius_scale).max(1e-6),
            relax: OVERLAP_RELAX,
            radius_scale,
            tol,
            _pad1: 0.0,
            _pad2: 0.0,
        };
        self.queue
            .write_buffer(&self.overlap_params_buf, 0, bytemuck::bytes_of(&params));
    }

    /// Encode the four grid-build passes (clear, count, scan, fill)
    /// for the current position buffer. Each pass gets its own
    /// compute pass so dispatch ordering and memory visibility are
    /// unambiguous.
    fn encode_grid_build(&self, encoder: &mut wgpu::CommandEncoder) {
        let bg = &self.overlap_build_bg[self.cur];
        let bucket_wg = OVERLAP_BUCKETS.div_ceil(WORKGROUP);
        let passes: [(&wgpu::ComputePipeline, u32); 4] = [
            (&self.grid_clear_pipeline, bucket_wg),
            (&self.grid_count_pipeline, self.workgroups),
            (&self.grid_scan_pipeline, 1),
            (&self.grid_fill_pipeline, self.workgroups),
        ];
        for (i, (pipeline, wg)) in passes.iter().enumerate() {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(["grid clear", "grid count", "grid scan", "grid fill"][i]),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, bg, &[]);
            pass.set_pipeline(pipeline);
            pass.dispatch_workgroups(*wg, 1, 1);
        }
    }

    /// Run `rounds` correction rounds (grid rebuild + Jacobi push
    /// apart per round), without measuring. This is the per-frame
    /// mode's building block and the drain-timing unit; the
    /// guaranteed path is [`Sim::resolve_overlaps`], which measures
    /// between rounds. Positions ping-pong exactly as in
    /// [`Sim::step`]; the force history is left untouched (the next
    /// force step recomputes forces from the corrected positions).
    pub fn resolve_iterations(&mut self, rounds: u32, radius_scale: f32) {
        if rounds == 0 {
            return;
        }
        self.write_overlap_params(radius_scale, OVERLAP_TOLERANCE);
        for _ in 0..rounds {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("overlap round"),
                });
            self.encode_grid_build(&mut encoder);
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("overlap resolve"),
                    timestamp_writes: None,
                });
                pass.set_bind_group(0, &self.overlap_bg[self.cur], &[]);
                pass.set_pipeline(&self.resolve_pipeline);
                pass.dispatch_workgroups(self.workgroups, 1, 1);
            }
            self.queue.submit([encoder.finish()]);
            self.cur = 1 - self.cur;
        }
    }

    /// Measure the overlap state over all pairs via the GPU
    /// reduction in the measure pass: (exact maximum overlap in
    /// world units, number of pairs overlapping by more than
    /// `tolerance`). On-demand only (it reads 16 bytes back);
    /// never called from the frame loop.
    pub async fn measure_overlap(
        &self,
        radius_scale: f32,
        tolerance: f32,
    ) -> Result<(f32, u32), String> {
        self.write_overlap_params(radius_scale, tolerance);
        self.queue.write_buffer(&self.overlap_stats, 0, &[0u8; 16]);
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("overlap stats staging"),
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("overlap measure"),
            });
        self.encode_grid_build(&mut encoder);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("overlap measure pass"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &self.overlap_bg[self.cur], &[]);
            pass.set_pipeline(&self.measure_pipeline);
            pass.dispatch_workgroups(self.workgroups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.overlap_stats, 0, &staging, 0, 16);
        self.queue.submit([encoder.finish()]);

        await_map(&self.device, staging.slice(..)).await?;
        let slice = staging.slice(..);
        let range = slice
            .get_mapped_range()
            .map_err(|e| format!("get_mapped_range failed: {e:?}"))?;
        let words: [u32; 2] = bytemuck::cast_slice(&range[0..8]).try_into().unwrap();
        let out = (f32::from_bits(words[0]), words[1]);
        drop(range);
        staging.unmap();
        Ok(out)
    }

    /// Camera fit support: reduce the live positions to their
    /// layout bounds on the GPU — centroid and the exact maximum
    /// distance from it (shaders/bounds.wgsl: workgroup partial
    /// sums, a single-workgroup centroid pass, then an atomicMax
    /// distance pass) — and read back just the 16-byte stats
    /// block. Positions are never copied to the host for this.
    /// On-demand only; never called from the frame loop. Same
    /// presenting-device readback caveat as
    /// [`Sim::measure_overlap`].
    pub async fn layout_bounds(&self) -> Result<LayoutBounds, String> {
        if self.count == 0 {
            return Ok(LayoutBounds {
                cx: 0.0,
                cy: 0.0,
                radius: 0.0,
            });
        }
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bounds stats staging"),
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("layout bounds"),
            });
        // Separate compute passes so each stage's writes are
        // visible to the next (partials -> centroid -> max).
        let bg = &self.bounds_bg[self.cur];
        let passes: [(&wgpu::ComputePipeline, u32); 3] = [
            (&self.bounds_sum_pipeline, self.workgroups),
            (&self.bounds_reduce_pipeline, 1),
            (&self.bounds_max_pipeline, self.workgroups),
        ];
        for (i, (pipeline, wg)) in passes.iter().enumerate() {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(["bounds sum", "bounds reduce", "bounds max"][i]),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, bg, &[]);
            pass.set_pipeline(pipeline);
            pass.dispatch_workgroups(*wg, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.bounds_stats, 0, &staging, 0, 16);
        self.queue.submit([encoder.finish()]);

        await_map(&self.device, staging.slice(..)).await?;
        let slice = staging.slice(..);
        let range = slice
            .get_mapped_range()
            .map_err(|e| format!("get_mapped_range failed: {e:?}"))?;
        let words: [u32; 3] = bytemuck::cast_slice(&range[0..12]).try_into().unwrap();
        let out = LayoutBounds {
            cx: f32::from_bits(words[0]),
            cy: f32::from_bits(words[1]),
            radius: f32::from_bits(words[2]),
        };
        drop(range);
        staging.unmap();
        Ok(out)
    }

    /// The guarantee path (chunk 3): measure, then run correction
    /// rounds, re-measuring after every round, until the measured
    /// maximum overlap is at or below `tolerance` or `max_rounds`
    /// rounds have run. The returned [`OverlapReport`] carries the
    /// measured before/after values either way, so a cap hit with
    /// residual overlap is visible in the result (and in the UI),
    /// never silently presented as success.
    pub async fn resolve_overlaps(
        &mut self,
        max_rounds: u32,
        tolerance: f32,
        radius_scale: f32,
    ) -> Result<OverlapReport, String> {
        let (before_max, before_pairs) = self.measure_overlap(radius_scale, tolerance).await?;
        let mut rounds = 0u32;
        let mut cur_max = before_max;
        let mut cur_pairs = before_pairs;
        while cur_max > tolerance && rounds < max_rounds {
            self.resolve_iterations(1, radius_scale);
            rounds += 1;
            let (m, p) = self.measure_overlap(radius_scale, tolerance).await?;
            cur_max = m;
            cur_pairs = p;
        }
        Ok(OverlapReport {
            before_max,
            before_pairs,
            after_max: cur_max,
            after_pairs: cur_pairs,
            rounds,
            converged: cur_max <= tolerance,
        })
    }

    /// Draw edges and nodes into `view` using the current positions.
    pub fn render(&self, view: &wgpu::TextureView, camera: &Camera) {
        let render = self.render.as_ref().expect("render pipelines not built");
        self.queue
            .write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(camera));
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, &render.render_bg[self.cur], &[]);
            pass.set_vertex_buffer(0, render.corner_buf.slice(..));
            pass.set_pipeline(&render.edge_pipeline);
            pass.draw(0..6, 0..self.edge_count);
            // Arrowheads sit between edges and nodes: the node
            // discs stay on top. The arrow vertex shader collapses
            // every instance whose edge is not directed (or when
            // arrows are off), so this draw is unconditional.
            pass.set_pipeline(&render.arrow_pipeline);
            pass.draw(0..3, 0..self.edge_count);
            pass.set_pipeline(&render.node_pipeline);
            pass.draw(0..6, 0..self.count);
        }
        self.queue.submit([encoder.finish()]);
    }

    /// Render one frame into an offscreen RGBA8 texture and read the
    /// pixels back (row padding stripped). Verification only: this is
    /// how chunk-1 rendering behavior (radius, edge width, negative
    /// visibility) is checked in environments where the presented
    /// canvas cannot be captured. Requires the render pipelines
    /// (Sim built with a surface format of Rgba8Unorm-compatible
    /// usage; the texture format used here is Rgba8Unorm).
    pub async fn render_rgba(
        &self,
        width: u32,
        height: u32,
        camera: &Camera,
    ) -> Result<Vec<u8>, String> {
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen render target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.render(&view, camera);

        // Rows in a copy must be padded to 256 bytes.
        let unpadded = width * 4;
        let padded = unpadded.div_ceil(256) * 256;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("offscreen readback staging"),
            size: (padded * height) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("offscreen copy"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([encoder.finish()]);

        await_map(&self.device, staging.slice(..)).await?;
        let slice = staging.slice(..);
        let range = slice
            .get_mapped_range()
            .map_err(|e| format!("get_mapped_range failed: {e:?}"))?;
        let mut out = Vec::with_capacity((unpadded * height) as usize);
        for row in 0..height as usize {
            let start = row * padded as usize;
            out.extend_from_slice(&range[start..start + unpadded as usize]);
        }
        drop(range);
        staging.unmap();
        Ok(out)
    }

    /// Copy the current positions out of the GPU. Verification only;
    /// never called from the frame loop.
    pub async fn readback_positions(&self) -> Result<Vec<f32>, String> {
        let size = self.count as u64 * 16;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback"),
            });
        encoder.copy_buffer_to_buffer(&self.pos[self.cur], 0, &staging, 0, size);
        self.queue.submit([encoder.finish()]);

        await_map(&self.device, staging.slice(..)).await?;
        let slice = staging.slice(..);
        let range = slice
            .get_mapped_range()
            .map_err(|e| format!("get_mapped_range failed: {e:?}"))?;
        let out: Vec<f32> = bytemuck::cast_slice(&range).to_vec();
        drop(range);
        staging.unmap();
        Ok(out)
    }

    /// Lossless SVG export of the current layout (chunk 2,
    /// FEATURE_REQUESTS.md item 8). Every node becomes one <circle>
    /// and every undirected edge one <line>, regardless of the
    /// current visibility toggles or node sizes — nothing is dropped.
    /// Negative edges are exported like all others and marked with
    /// data-negative="1". Stroke widths follow the current width
    /// rules (linked: clamp(|strength|, 0.25, 16); unlinked: the
    /// per-edge width), and the raw strength/width are kept in data
    /// attributes. When `images_enabled` is set, each circle also
    /// carries data-image with its atlas cell index; raster images
    /// are deliberately NOT embedded (the cell index is the data).
    /// Node radii in the SVG are the data-model radii in world units
    /// (times `radius_scale`), the same values that scale the
    /// on-screen discs. The y axis is flipped so the document matches
    /// the on-screen appearance. Positions are read back here, at
    /// export time only; the frame loop stays readback-free.
    pub async fn export_svg(
        &self,
        images_enabled: bool,
        link_width: bool,
        radius_scale: f32,
    ) -> Result<String, String> {
        let pos = self.readback_positions().await?;
        let n = self.count as usize;
        // Exported coordinates: x as-is, y flipped (SVG y is down,
        // the layout's y is up on screen).
        let px = |i: usize| pos[i * 4];
        let py = |i: usize| -pos[i * 4 + 1];
        let mut minx = f32::INFINITY;
        let mut miny = f32::INFINITY;
        let mut maxx = f32::NEG_INFINITY;
        let mut maxy = f32::NEG_INFINITY;
        let mut max_r = 0.0f32;
        for (i, &r) in self.export_radii.iter().enumerate() {
            minx = minx.min(px(i));
            maxx = maxx.max(px(i));
            miny = miny.min(py(i));
            maxy = maxy.max(py(i));
            max_r = max_r.max(r * radius_scale);
        }
        if !minx.is_finite() || !max_r.is_finite() {
            return Err("layout contains non-finite positions".to_string());
        }
        let span = (maxx - minx).max(maxy - miny).max(1.0);
        let margin = max_r + span * 0.02 + 1.0;
        let vx = minx - margin;
        let vy = miny - margin;
        let vw = (maxx - minx) + 2.0 * margin;
        let vh = (maxy - miny) + 2.0 * margin;

        let mut s = String::with_capacity(1024 + n * 128 + self.export_edges.len() * 160);
        s.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        s.push_str(&format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"{} {} {} {}\" width=\"{}\" height=\"{}\">\n",
            fmt_num(vx),
            fmt_num(vy),
            fmt_num(vw),
            fmt_num(vh),
            vw.round().max(1.0) as i64,
            vh.round().max(1.0) as i64,
        ));
        s.push_str(&format!(
            "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"#101216\"/>\n",
            fmt_num(vx),
            fmt_num(vy),
            fmt_num(vw),
            fmt_num(vh),
        ));
        // Edges under nodes, in the viewer's edge color.
        s.push_str("<g fill=\"none\" stroke=\"#9eadc7\" stroke-opacity=\"0.35\">\n");
        for &(a, b, strength, width) in &self.export_edges {
            let (a, b) = (a as usize, b as usize);
            let w = if link_width {
                strength.abs().clamp(0.25, 16.0)
            } else {
                width
            };
            s.push_str(&format!(
                "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke-width=\"{}\" data-strength=\"{}\" data-width=\"{}\"",
                fmt_num(px(a)),
                fmt_num(py(a)),
                fmt_num(px(b)),
                fmt_num(py(b)),
                fmt_num(w),
                fmt_num(strength),
                fmt_num(width),
            ));
            if strength < 0.0 {
                s.push_str(" data-negative=\"1\"");
            }
            s.push_str("/>\n");
        }
        s.push_str("</g>\n<g>\n");
        for i in 0..n {
            let [r, g, b] = self.export_colors[i];
            let hex = |c: f32| (c.clamp(0.0, 1.0) * 255.0).round() as u32;
            s.push_str(&format!(
                "<circle cx=\"{}\" cy=\"{}\" r=\"{}\" fill=\"#{:02x}{:02x}{:02x}\"",
                fmt_num(px(i)),
                fmt_num(py(i)),
                fmt_num(self.export_radii[i] * radius_scale),
                hex(r),
                hex(g),
                hex(b),
            ));
            if images_enabled {
                s.push_str(&format!(" data-image=\"{}\"", self.export_cells[i]));
            }
            s.push_str("/>\n");
        }
        s.push_str("</g>\n</svg>\n");
        Ok(s)
    }
}

/// Shortest round-trip formatting for f32 values in the SVG export:
/// integers print without a decimal point, everything else with the
/// minimal digits that reproduce the exact float.
fn fmt_num(v: f32) -> String {
    format!("{v}")
}
