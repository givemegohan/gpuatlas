//! Native headless benchmark for the gpuatlas wgpu simulation core.
//!
//! Usage: cargo run --release --bin bench [data.json] [limit] [iterations]
//! Defaults: ../bench-data.json, full data, 30 timed iterations.
//!
//! Needs a Vulkan or GL adapter. In this VM there is no system Vulkan
//! ICD; point VK_ICD_FILENAMES at the SwiftShader ICD bundled with
//! Playwright's Chromium (and LD_LIBRARY_PATH at its directory) to run
//! on the same software renderer the browser uses.

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    bench_impl::run();
}

#[cfg(not(target_arch = "wasm32"))]
mod bench_impl {
    use gpuatlas::sim::{GraphData, Sim, DEFAULT_GRAVITY, DEFAULT_SCALING};
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct BenchData {
        nodes: Vec<Node>,
        edges: Vec<[u32; 2]>,
    }

    #[derive(Deserialize)]
    struct Node {
        #[allow(dead_code)]
        id: u32,
        community: u32,
    }

    /// mulberry32 with the exact bit semantics of the JS version used
    /// by the WebGL2 viewer (i32 wrapping ops, logical shifts).
    struct Mulberry32 {
        a: i32,
    }

    impl Mulberry32 {
        fn new(seed: u32) -> Self {
            Mulberry32 { a: seed as i32 }
        }
        fn next(&mut self) -> f32 {
            self.a = self.a.wrapping_add(0x6d2b79f5u32 as i32);
            let a = self.a;
            let mut t = (a ^ (((a as u32) >> 15) as i32)).wrapping_mul(1 | a);
            t = t.wrapping_add((t ^ (((t as u32) >> 7) as i32)).wrapping_mul(61 | t)) ^ t;
            (((t ^ (((t as u32) >> 14) as i32)) as u32) as f64 / 4294967296.0) as f32
        }
    }

    /// Golden-angle HSL palette, ported from the WebGL2 version.
    fn community_color(c: u32) -> [f32; 3] {
        let h = ((c as f64 * 137.508) % 360.0) / 360.0;
        let s = 0.65f64;
        let l = 0.55f64;
        let q = if l < 0.5 {
            l * (1.0 + s)
        } else {
            l + s - l * s
        };
        let p = 2.0 * l - q;
        let tc = |mut t: f64| -> f64 {
            if t < 0.0 {
                t += 1.0;
            }
            if t > 1.0 {
                t -= 1.0;
            }
            if t < 1.0 / 6.0 {
                return p + (q - p) * 6.0 * t;
            }
            if t < 1.0 / 2.0 {
                return q;
            }
            if t < 2.0 / 3.0 {
                return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
            }
            p
        };
        [
            tc(h + 1.0 / 3.0) as f32,
            tc(h) as f32,
            tc(h - 1.0 / 3.0) as f32,
        ]
    }

    pub fn run() {
        let args: Vec<String> = std::env::args().collect();
        let path = args
            .get(1)
            .map(String::as_str)
            .unwrap_or("../bench-data.json");
        let limit: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
        let iters: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(30);

        let text = std::fs::read_to_string(path).expect("read bench data");
        let parsed: BenchData = serde_json::from_str(&text).expect("parse bench data");
        let mut nodes = parsed.nodes;
        let mut edges = parsed.edges;
        if limit > 0 && limit < nodes.len() {
            nodes.truncate(limit);
            edges.retain(|e| (e[0] as usize) < limit && (e[1] as usize) < limit);
        }
        let n = nodes.len();

        // CSR adjacency, both directions (same as the web build).
        let mut degree = vec![0u32; n];
        for e in &edges {
            degree[e[0] as usize] += 1;
            degree[e[1] as usize] += 1;
        }
        let mut offsets = vec![0u32; n + 1];
        for i in 0..n {
            offsets[i + 1] = offsets[i] + degree[i];
        }
        let mut neighbors = vec![0u32; offsets[n] as usize];
        let mut cursor = offsets[..n].to_vec();
        for e in &edges {
            let (a, b) = (e[0] as usize, e[1] as usize);
            neighbors[cursor[a] as usize] = e[1];
            cursor[a] += 1;
            neighbors[cursor[b] as usize] = e[0];
            cursor[b] += 1;
        }

        // Initial positions: mulberry32(1234), disc radius 100.
        let mut rng = Mulberry32::new(1234);
        let mut positions = vec![0f32; n * 4];
        for i in 0..n {
            let r = 100.0 * rng.next().sqrt();
            let th = rng.next() * std::f32::consts::TAU;
            positions[i * 4] = r * th.cos();
            positions[i * 4 + 1] = r * th.sin();
            positions[i * 4 + 2] = degree[i] as f32 + 1.0;
            positions[i * 4 + 3] = 1.0;
        }
        let mut colors = vec![0f32; n * 4];
        for (i, node) in nodes.iter().enumerate() {
            let [r, g, b] = community_color(node.community);
            colors[i * 4] = r;
            colors[i * 4 + 1] = g;
            colors[i * 4 + 2] = b;
            colors[i * 4 + 3] = 1.0;
        }
        let mut edge_pairs = Vec::with_capacity(edges.len() * 2);
        for e in &edges {
            edge_pairs.push(e[0]);
            edge_pairs.push(e[1]);
        }
        // Chunk-1 attributes at their defaults, so the native numbers
        // stay comparable with the pre-chunk-1 runs: radius from the
        // documented default formula, strength 1, width 1.
        let radii: Vec<f32> = degree
            .iter()
            .map(|d| 0.35 * (*d as f32 + 1.0).sqrt())
            .collect();
        let edge_strengths = vec![1.0f32; neighbors.len()];
        let mut edge_attrs = Vec::with_capacity(edges.len() * 4);
        for _ in &edges {
            edge_attrs.push(1.0f32); // strength
            edge_attrs.push(1.0f32); // width
            edge_attrs.push(0.0f32); // directed (render-only)
            edge_attrs.push(-1.0f32); // type index (untyped)
        }

        let mut instance_desc = wgpu::InstanceDescriptor::new_without_display_handle();
        instance_desc.backends = wgpu::Backends::VULKAN | wgpu::Backends::GL;
        let instance = wgpu::Instance::new(instance_desc);
        let adapter = match pollster::block_on(
            instance.request_adapter(&wgpu::RequestAdapterOptions::default()),
        ) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("NO ADAPTER: {e:?}");
                std::process::exit(2);
            }
        };
        let info = adapter.get_info();
        println!(
            "adapter: {} (backend {:?}, type {:?})",
            info.name, info.backend, info.device_type
        );
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("bench device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
        }))
        .expect("request_device");

        // Chunk-2 fields: the native bench never renders, so the
        // atlas is not needed; cell indices use the default rule.
        let image_cells: Vec<u32> = (0..n as u32).map(|i| i % 1024).collect();
        let data = GraphData {
            positions: &positions,
            colors: &colors,
            radii: &radii,
            offsets: &offsets,
            neighbors: &neighbors,
            edge_strengths: &edge_strengths,
            edge_pairs: &edge_pairs,
            edge_attrs: &edge_attrs,
            image_cells: &image_cells,
            atlas: None,
        };
        let mut sim = Sim::new(device, queue, &data, DEFAULT_SCALING, DEFAULT_GRAVITY, None);
        let direct = std::env::var("REPULSION")
            .map(|v| v == "direct")
            .unwrap_or(false);
        sim.set_direct_repulsion(direct);
        println!(
            "repulsion variant: {}",
            if direct { "direct" } else { "tiled" }
        );

        for _ in 0..3 {
            sim.step();
            let _ = sim.device().poll(wgpu::PollType::wait_indefinitely());
        }
        let mut times = Vec::new();
        for _ in 0..iters {
            let t0 = std::time::Instant::now();
            sim.step();
            let _ = sim.device().poll(wgpu::PollType::wait_indefinitely());
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = times[times.len() / 2];
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        println!(
            "N={} edges={} timed_iterations={} median {:.1} ms/iter ({:.2} iter/s), mean {:.1} ms/iter",
            n,
            edges.len(),
            iters,
            median,
            1000.0 / median,
            mean
        );

        // Sanity: run a few more iterations and check the layout moved
        // and stayed finite.
        for _ in 0..5 {
            sim.step();
        }
        let _ = sim.device().poll(wgpu::PollType::wait_indefinitely());
        let pos = pollster::block_on(sim.readback_positions()).expect("readback");
        let mut mr = 0f64;
        let mut finite = true;
        for i in 0..n {
            let x = pos[i * 4] as f64;
            let y = pos[i * 4 + 1] as f64;
            if !x.is_finite() || !y.is_finite() {
                finite = false;
            }
            mr += (x * x + y * y).sqrt();
        }
        println!(
            "sanity after +5 iterations: finite={} mean_radius={:.1}",
            finite,
            mr / n as f64
        );
    }
}
