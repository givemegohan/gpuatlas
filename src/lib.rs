//! gpuatlas wgpu viewer: a GPU-resident ForceAtlas-2-style layout engine
//! and renderer for 30,000-node networks, written in Rust on wgpu.
//!
//! The platform-agnostic core lives in [`sim`]. This file adds the
//! WebAssembly bindings used by the browser build; the native
//! headless benchmark lives in `src/bin/bench.rs`.

pub mod sim;

#[cfg(target_arch = "wasm32")]
mod web {
    use crate::sim::{
        Camera, GraphData, Sim, ATLAS_DIM, DEFAULT_EDGE_COLOR, DEFAULT_GRAVITY, DEFAULT_LOD_PX,
        DEFAULT_SCALING, OVERLAP_MAX_ROUNDS, OVERLAP_PER_FRAME_ROUNDS, OVERLAP_TOLERANCE,
    };
    use std::sync::{Arc, Mutex};
    use wasm_bindgen::prelude::*;

    /// Result of a single overlap measurement (chunk 3): the exact
    /// maximum overlap over all pairs and the count of pairs
    /// overlapping by more than the tolerance (1e-3), both from the
    /// GPU reduction in the measure pass.
    #[wasm_bindgen]
    pub struct OverlapInfo {
        pub max_overlap: f32,
        pub pairs: u32,
    }

    /// Outcome of a `resolve_overlaps` run (chunk 3). `converged`
    /// is true only when the measured `after_max` is at or below
    /// the tolerance; a round-cap hit leaves it false with the
    /// residual values in plain sight.
    #[wasm_bindgen]
    pub struct OverlapReport {
        pub before_max: f32,
        pub before_pairs: u32,
        pub after_max: f32,
        pub after_pairs: u32,
        pub rounds: u32,
        pub converged: bool,
    }

    impl From<crate::sim::OverlapReport> for OverlapReport {
        fn from(r: crate::sim::OverlapReport) -> Self {
            OverlapReport {
                before_max: r.before_max,
                before_pairs: r.before_pairs,
                after_max: r.after_max,
                after_pairs: r.after_pairs,
                rounds: r.rounds,
                converged: r.converged,
            }
        }
    }

    /// Outcome of a `settle` run (usability chunk): iterations run
    /// and the measured mean per-iteration displacement of the last
    /// batch. `converged` is true only when that value reached the
    /// threshold the run was given.
    #[wasm_bindgen]
    pub struct SettleReport {
        pub iterations: u32,
        pub final_mean_displacement: f32,
        pub converged: bool,
    }

    /// Layout bounds of the live positions (camera fit): centroid
    /// and the maximum distance from it, reduced on the GPU and
    /// read back as three floats.
    #[wasm_bindgen]
    #[derive(Clone, Copy)]
    pub struct LayoutBounds {
        pub cx: f32,
        pub cy: f32,
        pub radius: f32,
    }

    impl From<crate::sim::LayoutBounds> for LayoutBounds {
        fn from(b: crate::sim::LayoutBounds) -> Self {
            LayoutBounds {
                cx: b.cx,
                cy: b.cy,
                radius: b.radius,
            }
        }
    }

    impl From<crate::sim::SettleReport> for SettleReport {
        fn from(r: crate::sim::SettleReport) -> Self {
            SettleReport {
                iterations: r.iterations,
                final_mean_displacement: r.final_mean_displacement,
                converged: r.converged,
            }
        }
    }

    /// Browser viewer: owns the simulation, the canvas surface, and
    /// its configuration. JavaScript drives it once per
    /// requestAnimationFrame via [`Viewer::frame`], mirroring how the
    /// WebGL2 version was driven.
    #[wasm_bindgen]
    pub struct Viewer {
        sim: Sim,
        surface: wgpu::Surface<'static>,
        config: wgpu::SurfaceConfiguration,
        gpu_error: Arc<Mutex<Option<String>>>,
        surface_status: String,
        // Chunk-1 render settings (see shaders/render.wgsl).
        radius_scale: f32,
        link_width: bool,
        hide_negative: bool,
        // Chunk-2 render settings: image LOD threshold (device px)
        // and the node-images toggle.
        lod_px: f32,
        images_on: bool,
        // Circular crop for node images (default on). Off renders
        // images as full squares; display-only, the simulation and
        // overlap handling are unaffected.
        crop_circle: bool,
        // Chunk-3 setting: run a few overlap-resolution rounds
        // every frame so the layout settles overlap-free while it
        // simulates. Off by default (it costs GPU time per frame);
        // the guaranteed path is the resolve_overlaps button.
        resolve_per_frame: bool,
        // Usability-chunk render settings: edge color mode (0
        // fixed, 1 endpoint, 2 type), the fixed edge color, and
        // the arrowhead toggles.
        edge_mode: f32,
        edge_color: [f32; 3],
        arrows_on: bool,
        arrows_all: bool,
    }

    fn err<E: std::fmt::Debug>(e: E) -> JsValue {
        JsValue::from_str(&format!("{e:?}"))
    }

    /// Base camera shared by the HeadlessSim offscreen renders:
    /// the same mapping as `Viewer::frame`, with dpr = 1.
    fn headless_camera(width: u32, height: u32, cx: f32, cy: f32, zoom: f32) -> Camera {
        let w = width as f32;
        let h = height as f32;
        Camera::new(
            [cx, cy],
            [2.0 * zoom / w, 2.0 * zoom / h],
            [w, h],
            (1.15 * zoom).clamp(1.5, 256.0),
        )
    }

    /// Validate the chunk-2 atlas arguments shared by both
    /// constructors: RGBA pixels for an ATLAS_DIM x ATLAS_DIM atlas.
    fn atlas_vec(atlas_rgba: &js_sys::Uint8Array) -> Result<Vec<u8>, JsValue> {
        let v = atlas_rgba.to_vec();
        let expect = (ATLAS_DIM * ATLAS_DIM * 4) as usize;
        if v.len() != expect {
            return Err(JsValue::from_str(&format!(
                "atlas_rgba must be {expect} bytes (ATLAS_DIM^2 RGBA), got {}",
                v.len()
            )));
        }
        Ok(v)
    }

    /// Create a viewer bound to `canvas`. Graph arrays are prepared by
    /// the caller (see index.html): positions/colors are 4 floats per
    /// node, `radii` is 1 float per node, `offsets`/`neighbors` form
    /// the CSR adjacency (both directions), `edge_strengths` holds 1
    /// float per CSR entry (aligned with `neighbors`), `edge_pairs`
    /// holds 2 endpoint indices per undirected edge, and `edge_attrs`
    /// holds (strength, width, directed, type_index) per undirected
    /// edge. Chunk 2 adds
    /// `image_cells` (1 atlas cell index per node) and `atlas_rgba`
    /// (the ATLAS_DIM x ATLAS_DIM RGBA atlas pixels).
    // The argument list is the JS-facing API (one typed array per
    // graph array); bundling it would change the public signature.
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen]
    pub async fn create_viewer(
        canvas: web_sys::HtmlCanvasElement,
        positions: js_sys::Float32Array,
        colors: js_sys::Float32Array,
        radii: js_sys::Float32Array,
        offsets: js_sys::Uint32Array,
        neighbors: js_sys::Uint32Array,
        edge_strengths: js_sys::Float32Array,
        edge_pairs: js_sys::Uint32Array,
        edge_attrs: js_sys::Float32Array,
        image_cells: js_sys::Uint32Array,
        atlas_rgba: js_sys::Uint8Array,
    ) -> Result<Viewer, JsValue> {
        console_error_panic_hook::set_once();

        let positions = positions.to_vec();
        let colors = colors.to_vec();
        let radii = radii.to_vec();
        let offsets = offsets.to_vec();
        let neighbors = neighbors.to_vec();
        let edge_strengths = edge_strengths.to_vec();
        let edge_pairs = edge_pairs.to_vec();
        let edge_attrs = edge_attrs.to_vec();
        let image_cells = image_cells.to_vec();
        let atlas = atlas_vec(&atlas_rgba)?;

        let mut instance_desc = wgpu::InstanceDescriptor::new_without_display_handle();
        instance_desc.backends = wgpu::Backends::BROWSER_WEBGPU;
        let instance = wgpu::Instance::new(instance_desc);
        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
            .map_err(err)?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .map_err(err)?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("gpuatlas device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            })
            .await
            .map_err(err)?;

        // Record every uncaptured GPU error (validation failures from
        // buffer creation, copies, passes) for diagnosis from JS.
        let gpu_error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        {
            let sink = Arc::clone(&gpu_error);
            device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
                *sink.lock().unwrap() = Some(format!("{e}"));
            }));
        }

        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats[0];
        let width = canvas.width().max(1);
        let height = canvas.height().max(1);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            color_space: wgpu::SurfaceColorSpace::Auto,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

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
            atlas: Some(&atlas),
        };
        let sim = Sim::new(
            device,
            queue,
            &data,
            DEFAULT_SCALING,
            DEFAULT_GRAVITY,
            Some(format),
        );
        Ok(Viewer {
            sim,
            surface,
            config,
            gpu_error,
            surface_status: "init".to_string(),
            radius_scale: 1.0,
            link_width: false,
            hide_negative: false,
            lod_px: DEFAULT_LOD_PX,
            images_on: true,
            crop_circle: true,
            resolve_per_frame: false,
            edge_mode: 0.0,
            edge_color: DEFAULT_EDGE_COLOR,
            arrows_on: false,
            arrows_all: false,
        })
    }

    #[wasm_bindgen]
    impl Viewer {
        /// Most recent uncaptured GPU error, if any (diagnostics).
        pub fn gpu_error(&self) -> Option<String> {
            self.gpu_error.lock().unwrap().clone()
        }

        /// Status of the most recent get_current_texture call
        /// (diagnostics for presentation problems).
        pub fn surface_status(&self) -> String {
            self.surface_status.clone()
        }

        pub fn node_count(&self) -> u32 {
            self.sim.count()
        }

        pub fn set_scaling(&mut self, scaling: f32) {
            self.sim.set_scaling(scaling);
        }

        pub fn set_gravity(&mut self, gravity: f32) {
            self.sim.set_gravity(gravity);
        }

        /// Select the untiled repulsion variant (A/B measurement).
        pub fn set_direct_repulsion(&mut self, direct: bool) {
            self.sim.set_direct_repulsion(direct);
        }

        /// Chunk-1 render settings: multiplier on every node's radius.
        pub fn set_radius_scale(&mut self, scale: f32) {
            self.radius_scale = scale;
        }

        /// Chunk-1 render settings: derive edge width from |strength|.
        pub fn set_link_width(&mut self, linked: bool) {
            self.link_width = linked;
        }

        /// Chunk-1 render settings: hide edges with strength < 0.
        pub fn set_hide_negative(&mut self, hide: bool) {
            self.hide_negative = hide;
        }

        /// Chunk-2 render settings: image LOD threshold — nodes
        /// whose on-screen diameter reaches this many device px draw
        /// their atlas icon instead of the plain disc.
        pub fn set_lod_px(&mut self, px: f32) {
            self.lod_px = px;
        }

        /// Chunk-2 render settings: master toggle for node images.
        pub fn set_images_enabled(&mut self, on: bool) {
            self.images_on = on;
        }

        /// Node-image circular crop (default on). Off renders
        /// images as full squares filling the node quad. Display
        /// only: the simulation and overlap resolution keep using
        /// the radius alone either way.
        pub fn set_crop_circle(&mut self, on: bool) {
            self.crop_circle = on;
        }

        /// Chunk-3 setting: run [`OVERLAP_PER_FRAME_ROUNDS`]
        /// overlap-resolution rounds every frame (after the force
        /// step when running, and also while paused, so a static
        /// layout still settles to overlap-free). Off by default.
        pub fn set_resolve_per_frame(&mut self, on: bool) {
            self.resolve_per_frame = on;
        }

        /// Usability chunk: integration speed, bundled with
        /// scaling/gravity by the page's parameter presets.
        pub fn set_speed(&mut self, speed: f32) {
            self.sim.set_speed(speed);
        }

        /// Usability chunk: background (clear) color, components
        /// in 0..1 used as-is (see `Sim::set_clear_color`).
        pub fn set_background(&mut self, r: f32, g: f32, b: f32) {
            self.sim.set_clear_color(r as f64, g as f64, b as f64);
        }

        /// Usability chunk: edge color mode — 0 fixed color, 1
        /// source-endpoint node color, 2 per-type palette color.
        pub fn set_edge_mode(&mut self, mode: u32) {
            self.edge_mode = mode as f32;
        }

        /// Usability chunk: the fixed edge color (mode 0, and the
        /// fallback for untyped edges in mode 2), components 0..1.
        pub fn set_edge_color(&mut self, r: f32, g: f32, b: f32) {
            self.edge_color = [r, g, b];
        }

        /// Usability chunk: arrowheads on directed edges. `all`
        /// treats every edge as directed (the page sets it when the
        /// data marks no edge as directed).
        pub fn set_arrows(&mut self, on: bool, all: bool) {
            self.arrows_on = on;
            self.arrows_all = all;
        }

        /// Usability chunk: run `n` step-only iterations (no
        /// rendering). Synchronous; the building block under
        /// `settle` and the page's verification hooks.
        pub fn step_iterations(&mut self, n: u32) {
            for _ in 0..n {
                self.sim.step();
            }
        }

        /// Usability chunk: settle mode — step-only batches of 20
        /// iterations, measuring the mean per-iteration
        /// displacement from readbacks between batches, until it
        /// reaches `threshold` or `max_iterations` iterations have
        /// run. Same presenting-device readback caveat as
        /// `readback_positions`.
        pub async fn settle(
            &mut self,
            max_iterations: u32,
            threshold: f32,
        ) -> Result<SettleReport, JsValue> {
            let report = self
                .sim
                .settle(20, max_iterations, threshold)
                .await
                .map_err(err)?;
            Ok(report.into())
        }

        /// Chunk-3: measure the current maximum overlap and
        /// overlapping-pair count (GPU reduction; exact, over all
        /// pairs). Uses the viewer's current radius scale, i.e. the
        /// radii as drawn. Same presenting-device readback caveat as
        /// `readback_positions`: in Chromium + SwiftShader this
        /// fails on a presenting device; use `HeadlessSim` there.
        pub async fn measure_overlap(&self) -> Result<OverlapInfo, JsValue> {
            let (max_overlap, pairs) = self
                .sim
                .measure_overlap(self.radius_scale, OVERLAP_TOLERANCE)
                .await
                .map_err(err)?;
            Ok(OverlapInfo { max_overlap, pairs })
        }

        /// Camera fit: layout bounds (centroid + max radius) of
        /// the live positions from the GPU reduction; only three
        /// floats are read back. Same presenting-device readback
        /// caveat as `readback_positions`.
        pub async fn layout_bounds(&self) -> Result<LayoutBounds, JsValue> {
            let b = self.sim.layout_bounds().await.map_err(err)?;
            Ok(b.into())
        }

        /// Chunk-3: the guarantee path. Repeat measured correction
        /// rounds until the measured maximum overlap is at or below
        /// [`OVERLAP_TOLERANCE`] (1e-3 world units) or
        /// [`OVERLAP_MAX_ROUNDS`] rounds have run; the report says
        /// which of the two happened. Same readback caveat as
        /// `measure_overlap`.
        pub async fn resolve_overlaps(&mut self) -> Result<OverlapReport, JsValue> {
            let report = self
                .sim
                .resolve_overlaps(OVERLAP_MAX_ROUNDS, OVERLAP_TOLERANCE, self.radius_scale)
                .await
                .map_err(err)?;
            Ok(report.into())
        }

        /// Lossless SVG export of the current layout (chunk 2),
        /// using the viewer's current render settings (radius scale,
        /// width linking, images toggle). See `Sim::export_svg`.
        /// Note the readback caveat on `readback_positions`: in
        /// Chromium + SwiftShader this fails on a presenting device;
        /// it works on real browsers and via `HeadlessSim`.
        pub async fn export_svg(&self) -> Result<String, JsValue> {
            self.sim
                .export_svg(self.images_on, self.link_width, self.radius_scale)
                .await
                .map_err(err)
        }

        /// Replace all positions (shuffle control).
        pub fn reset_positions(&mut self, positions: js_sys::Float32Array) {
            self.sim.reset_positions(&positions.to_vec());
        }

        /// Reconfigure the surface after the canvas attribute size
        /// changed (JS owns canvas sizing, as in the WebGL2 version).
        pub fn resize(&mut self, width: u32, height: u32) {
            if width == 0 || height == 0 {
                return;
            }
            if width != self.config.width || height != self.config.height {
                self.config.width = width;
                self.config.height = height;
                self.surface.configure(self.sim.device(), &self.config);
            }
        }

        /// Advance one iteration (when `running`) and draw one frame.
        /// Camera math matches the WebGL2 version:
        /// view_scale = (2*zoom/w, 2*zoom/h),
        /// point_px = clamp(1.15 * zoom * dpr, 1.5, 256).
        pub fn frame(&mut self, cx: f32, cy: f32, zoom: f32, dpr: f32, running: bool) {
            if running {
                self.sim.step();
            }
            if self.resolve_per_frame {
                self.sim
                    .resolve_iterations(OVERLAP_PER_FRAME_ROUNDS, self.radius_scale);
            }
            let frame = self.surface.get_current_texture();
            self.surface_status = match &frame {
                wgpu::CurrentSurfaceTexture::Success(_) => "Success",
                wgpu::CurrentSurfaceTexture::Suboptimal(_) => "Suboptimal",
                wgpu::CurrentSurfaceTexture::Timeout => "Timeout",
                wgpu::CurrentSurfaceTexture::Occluded => "Occluded",
                wgpu::CurrentSurfaceTexture::Outdated => "Outdated",
                wgpu::CurrentSurfaceTexture::Lost => "Lost",
                wgpu::CurrentSurfaceTexture::Validation => "Validation",
            }
            .to_string();
            let texture = match frame {
                wgpu::CurrentSurfaceTexture::Success(t)
                | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
                wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                    self.surface.configure(self.sim.device(), &self.config);
                    return;
                }
                _ => return,
            };
            let view = texture
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            let w = self.config.width as f32;
            let h = self.config.height as f32;
            let mut camera = Camera::new(
                [cx, cy],
                [2.0 * zoom / w, 2.0 * zoom / h],
                [w, h],
                (1.15 * zoom * dpr).clamp(1.5, 256.0),
            );
            camera.radius_scale = self.radius_scale;
            camera.link_width = if self.link_width { 1.0 } else { 0.0 };
            camera.hide_negative = if self.hide_negative { 1.0 } else { 0.0 };
            camera.lod_px = self.lod_px;
            camera.images_on = if self.images_on { 1.0 } else { 0.0 };
            camera.crop_circle = if self.crop_circle { 1.0 } else { 0.0 };
            camera.edge_mode = self.edge_mode;
            camera.edge_color = self.edge_color;
            camera.arrows_on = if self.arrows_on { 1.0 } else { 0.0 };
            camera.arrows_all = if self.arrows_all { 1.0 } else { 0.0 };
            self.sim.render(&view, &camera);
            self.sim.queue().present(texture);
        }

        /// Verification hook: copy current positions (4 floats per
        /// node) back to JS. Not used by the frame loop.
        ///
        /// NOTE: in Chromium + SwiftShader, mapAsync on a device that
        /// has presented a surface always fails ("A valid external
        /// Instance reference no longer exists" — bisected with raw
        /// WebGPU: configure-only and compute-only devices map fine,
        /// presenting devices do not). Use `HeadlessSim` for in-browser
        /// verification; this method works on non-presenting devices
        /// and in the native build.
        pub async fn readback_positions(&self) -> Result<js_sys::Float32Array, JsValue> {
            let v = self.sim.readback_positions().await.map_err(err)?;
            Ok(js_sys::Float32Array::from(v.as_slice()))
        }
    }

    /// A simulation on its own device with no surface. Runs the exact
    /// same shaders and data as the viewer, so its readback verifies
    /// the layout math in-browser (see the note on `Viewer`'s
    /// readback about presenting devices and mapAsync).
    #[wasm_bindgen]
    pub struct HeadlessSim {
        sim: Sim,
        // Circular crop for node images in the offscreen renders
        // (default on, as in the viewer). Stateful like
        // `Sim::set_clear_color` rather than another render_rgba_*
        // parameter list.
        crop_circle: bool,
    }

    #[allow(clippy::too_many_arguments)] // JS-facing signature, as in create_viewer
    #[wasm_bindgen]
    pub async fn create_headless(
        positions: js_sys::Float32Array,
        colors: js_sys::Float32Array,
        radii: js_sys::Float32Array,
        offsets: js_sys::Uint32Array,
        neighbors: js_sys::Uint32Array,
        edge_strengths: js_sys::Float32Array,
        edge_pairs: js_sys::Uint32Array,
        edge_attrs: js_sys::Float32Array,
        image_cells: js_sys::Uint32Array,
        atlas_rgba: js_sys::Uint8Array,
    ) -> Result<HeadlessSim, JsValue> {
        console_error_panic_hook::set_once();
        let positions = positions.to_vec();
        let colors = colors.to_vec();
        let radii = radii.to_vec();
        let offsets = offsets.to_vec();
        let neighbors = neighbors.to_vec();
        let edge_strengths = edge_strengths.to_vec();
        let edge_pairs = edge_pairs.to_vec();
        let edge_attrs = edge_attrs.to_vec();
        let image_cells = image_cells.to_vec();
        let atlas = atlas_vec(&atlas_rgba)?;

        let mut instance_desc = wgpu::InstanceDescriptor::new_without_display_handle();
        instance_desc.backends = wgpu::Backends::BROWSER_WEBGPU;
        let instance = wgpu::Instance::new(instance_desc);
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .map_err(err)?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("gpuatlas headless device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            })
            .await
            .map_err(err)?;
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
            atlas: Some(&atlas),
        };
        // Rgba8Unorm builds the render pipelines too, so the
        // offscreen verification render (render_rgba) works; nothing
        // is ever presented from this device.
        let sim = Sim::new(
            device,
            queue,
            &data,
            DEFAULT_SCALING,
            DEFAULT_GRAVITY,
            Some(wgpu::TextureFormat::Rgba8Unorm),
        );
        Ok(HeadlessSim {
            sim,
            crop_circle: true,
        })
    }

    #[wasm_bindgen]
    impl HeadlessSim {
        /// Run `n` iterations back to back (no rendering).
        pub fn step_iterations(&mut self, n: u32) {
            for _ in 0..n {
                self.sim.step();
            }
        }

        pub async fn readback_positions(&self) -> Result<js_sys::Float32Array, JsValue> {
            let v = self.sim.readback_positions().await.map_err(err)?;
            Ok(js_sys::Float32Array::from(v.as_slice()))
        }

        /// Render the current state into an offscreen RGBA8 image
        /// and return the raw pixels (width * height * 4 bytes).
        /// Camera math matches `Viewer::frame` with dpr = 1. Used by
        /// test.html to verify chunk-1 rendering behavior (radius,
        /// edge width, negative-edge visibility) in environments
        /// where the presented canvas cannot be captured.
        #[allow(clippy::too_many_arguments)] // JS-facing signature, as in create_viewer
        pub async fn render_rgba(
            &self,
            width: u32,
            height: u32,
            cx: f32,
            cy: f32,
            zoom: f32,
            radius_scale: f32,
            link_width: bool,
            hide_negative: bool,
        ) -> Result<js_sys::Uint8Array, JsValue> {
            let mut camera = headless_camera(width, height, cx, cy, zoom);
            camera.radius_scale = radius_scale;
            camera.link_width = if link_width { 1.0 } else { 0.0 };
            camera.hide_negative = if hide_negative { 1.0 } else { 0.0 };
            let v = self
                .sim
                .render_rgba(width, height, &camera)
                .await
                .map_err(err)?;
            Ok(js_sys::Uint8Array::from(v.as_slice()))
        }

        /// Like `render_rgba`, but with the chunk-2 image-LOD
        /// settings (threshold in device px, images toggle). Used by
        /// test2.html to verify the LOD layer offscreen.
        #[allow(clippy::too_many_arguments)]
        pub async fn render_rgba_lod(
            &self,
            width: u32,
            height: u32,
            cx: f32,
            cy: f32,
            zoom: f32,
            radius_scale: f32,
            link_width: bool,
            hide_negative: bool,
            lod_px: f32,
            images_on: bool,
        ) -> Result<js_sys::Uint8Array, JsValue> {
            let mut camera = headless_camera(width, height, cx, cy, zoom);
            camera.radius_scale = radius_scale;
            camera.link_width = if link_width { 1.0 } else { 0.0 };
            camera.hide_negative = if hide_negative { 1.0 } else { 0.0 };
            camera.lod_px = lod_px;
            camera.images_on = if images_on { 1.0 } else { 0.0 };
            let v = self
                .sim
                .render_rgba(width, height, &camera)
                .await
                .map_err(err)?;
            Ok(js_sys::Uint8Array::from(v.as_slice()))
        }

        /// Chunk-3: run `n` overlap-correction rounds (grid rebuild
        /// + push-apart per round) without measuring. The drain
        /// timing unit for the resolution cost, and the building
        /// block of the per-frame mode.
        pub fn resolve_iterations(&mut self, n: u32, radius_scale: f32) {
            self.sim.resolve_iterations(n, radius_scale);
        }

        /// Usability chunk: background (clear) color for the
        /// offscreen renders, components in 0..1 used as-is.
        pub fn set_clear_color(&mut self, r: f32, g: f32, b: f32) {
            self.sim.set_clear_color(r as f64, g as f64, b as f64);
        }

        /// Node-image circular crop for the offscreen renders
        /// (default on). Off renders images as full squares;
        /// display only, exactly as in the viewer.
        pub fn set_crop_circle(&mut self, on: bool) {
            self.crop_circle = on;
        }

        /// Usability chunk: settle mode on the headless sim — the
        /// same code path as `Viewer::settle`, verifiable in
        /// environments where a presenting device cannot read back.
        pub async fn settle(
            &mut self,
            max_iterations: u32,
            threshold: f32,
        ) -> Result<SettleReport, JsValue> {
            let report = self
                .sim
                .settle(20, max_iterations, threshold)
                .await
                .map_err(err)?;
            Ok(report.into())
        }

        /// Like `render_rgba_lod`, but with the usability chunk's
        /// full render settings (edge color mode and fixed color,
        /// arrowhead toggles). Used by test4.html to verify the
        /// directed/type/background rendering offscreen.
        #[allow(clippy::too_many_arguments)]
        pub async fn render_rgba_full(
            &self,
            width: u32,
            height: u32,
            cx: f32,
            cy: f32,
            zoom: f32,
            radius_scale: f32,
            link_width: bool,
            hide_negative: bool,
            lod_px: f32,
            images_on: bool,
            edge_mode: u32,
            edge_r: f32,
            edge_g: f32,
            edge_b: f32,
            arrows_on: bool,
            arrows_all: bool,
        ) -> Result<js_sys::Uint8Array, JsValue> {
            let mut camera = headless_camera(width, height, cx, cy, zoom);
            camera.radius_scale = radius_scale;
            camera.link_width = if link_width { 1.0 } else { 0.0 };
            camera.hide_negative = if hide_negative { 1.0 } else { 0.0 };
            camera.lod_px = lod_px;
            camera.images_on = if images_on { 1.0 } else { 0.0 };
            camera.crop_circle = if self.crop_circle { 1.0 } else { 0.0 };
            camera.edge_mode = edge_mode as f32;
            camera.edge_color = [edge_r, edge_g, edge_b];
            camera.arrows_on = if arrows_on { 1.0 } else { 0.0 };
            camera.arrows_all = if arrows_all { 1.0 } else { 0.0 };
            let v = self
                .sim
                .render_rgba(width, height, &camera)
                .await
                .map_err(err)?;
            Ok(js_sys::Uint8Array::from(v.as_slice()))
        }

        /// Camera fit: layout bounds (centroid + max radius) of
        /// the live positions from the GPU reduction; only three
        /// floats are read back.
        pub async fn layout_bounds(&self) -> Result<LayoutBounds, JsValue> {
            let b = self.sim.layout_bounds().await.map_err(err)?;
            Ok(b.into())
        }

        /// Chunk-3: measure the exact maximum overlap and
        /// overlapping-pair count over all pairs (GPU reduction).
        pub async fn measure_overlap(&self, radius_scale: f32) -> Result<OverlapInfo, JsValue> {
            let (max_overlap, pairs) = self
                .sim
                .measure_overlap(radius_scale, OVERLAP_TOLERANCE)
                .await
                .map_err(err)?;
            Ok(OverlapInfo { max_overlap, pairs })
        }

        /// Chunk-3: the guarantee path — measured correction rounds
        /// until the measured maximum overlap is at or below
        /// `tolerance` or `max_rounds` rounds have run. The report
        /// carries the measured before/after values either way.
        pub async fn resolve_overlaps(
            &mut self,
            max_rounds: u32,
            tolerance: f32,
            radius_scale: f32,
        ) -> Result<OverlapReport, JsValue> {
            let report = self
                .sim
                .resolve_overlaps(max_rounds, tolerance, radius_scale)
                .await
                .map_err(err)?;
            Ok(report.into())
        }

        /// Lossless SVG export of the current layout (chunk 2) with
        /// explicit settings; the same code path the viewer's export
        /// button uses. See `Sim::export_svg`.
        pub async fn export_svg(
            &self,
            images_enabled: bool,
            link_width: bool,
            radius_scale: f32,
        ) -> Result<String, JsValue> {
            self.sim
                .export_svg(images_enabled, link_width, radius_scale)
                .await
                .map_err(err)
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub use web::{
    create_headless, create_viewer, HeadlessSim, LayoutBounds, OverlapInfo, OverlapReport,
    SettleReport, Viewer,
};
