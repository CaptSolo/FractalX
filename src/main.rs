//! FractalX — fractal explorer prototype.
//! Milestone 1: GPU Mandelbrot with smooth pan/zoom, iteration and palette controls.

// Windows release builds: GUI subsystem, so no console window opens behind
// the app. Debug builds keep the console for println/panic output.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod attractor;
mod deep;
mod export;
mod ifs;
mod lsystem;
mod mandelbrot;
mod palette;
mod terrain;

use eframe::egui;
use eframe::wgpu;
use mandelbrot::{RenderResources, Uniforms};

/// Chaos-game points added per frame while an IFS render is filling in.
const IFS_POINTS_PER_FRAME: u64 = 1_000_000;
/// Coarsest rung of the Mandelbrot resolution ladder (1/4 resolution).
const LADDER_START_LEVEL: u32 = 2;
/// Above this iteration cap, a rung renders via chunked compute (this many
/// iterations per frame) instead of one long fragment dispatch.
const CHUNK_ITERS: u32 = 2048;

/// Once units-per-point drops below this, f32 in the shader is out of bits
/// and rendering switches to the perturbation path.
const PERTURB_THRESHOLD: f64 = 1e-7;
/// Hard zoom floor: below this even f32 pixel deltas underflow (~1e30 zoom).
const MIN_UNITS_PER_POINT: f64 = 1e-32;

/// Which fractal family is rendered, plus its family-specific rule.
#[derive(Clone, PartialEq, Default, serde::Serialize, serde::Deserialize)]
#[serde(tag = "family", rename_all = "snake_case")]
enum FractalRule {
    #[default]
    Mandelbrot,
    Multibrot {
        power: u32,
    },
    Julia {
        /// The fixed constant c; the pixel seeds z instead.
        c: [f64; 2],
    },
    Ifs {
        maps: Vec<ifs::AffineMap>,
        points: u64,
    },
    LSystem {
        axiom: String,
        rules: Vec<lsystem::Rule>,
        angle_deg: f64,
        generations: u32,
    },
    Attractor {
        kind: attractor::Kind,
        /// Map parameters a, b, c, d.
        params: [f64; 4],
        points: u64,
    },
    Terrain(terrain::Params),
}

impl FractalRule {
    fn is_escape_time(&self) -> bool {
        !matches!(
            self,
            FractalRule::Ifs { .. }
                | FractalRule::LSystem { .. }
                | FractalRule::Attractor { .. }
                | FractalRule::Terrain(_)
        )
    }

    fn display_name(&self) -> &'static str {
        match self {
            FractalRule::Mandelbrot => "Mandelbrot",
            FractalRule::Multibrot { .. } => "Multibrot",
            FractalRule::Julia { .. } => "Julia",
            FractalRule::Ifs { .. } => "IFS (chaos game)",
            FractalRule::LSystem { .. } => "L-system",
            FractalRule::Attractor { .. } => "Strange attractor",
            FractalRule::Terrain(p) if p.clouds => "Clouds",
            FractalRule::Terrain(_) => "Terrain",
        }
    }

    /// Default viewport (center, units-per-point) for this family.
    fn home_view(&self) -> ([f64; 2], f64) {
        match self {
            FractalRule::Mandelbrot => ([-0.5, 0.0], 0.004),
            FractalRule::Multibrot { .. } => ([0.0, 0.0], 0.004),
            FractalRule::Julia { .. } => ([0.0, 0.0], 0.004),
            FractalRule::Ifs { .. } => ([0.5, 0.43], 0.0016),
            // Placeholder; L-system views are fitted to the drawing's bounds.
            FractalRule::LSystem { .. } => ([0.0, 0.0], 0.01),
            // Placeholder; attractor views are fitted to the orbit's bounds.
            FractalRule::Attractor { .. } => ([0.0, 0.0], 0.006),
            // ~8 base-octave noise features across a ~800pt canvas.
            FractalRule::Terrain(_) => ([0.0, 0.0], 0.01),
        }
    }
}

/// Complete view state — the "bookmark": it fully determines a render.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct ViewState {
    /// Complex-plane coordinates of the canvas center, arbitrary precision.
    /// (Doubles as the world-space center for IFS, where f64 suffices.)
    #[serde(with = "center_serde")]
    center: deep::BigComplex,
    /// Complex units per screen point (zoom level).
    units_per_point: f64,
    /// Escape-time iteration cap (unused by IFS, which has its own `points`).
    max_iter: u32,
    palette_freq: f32,
    palette_phase: f32,
    /// Absent in older bookmarks, which used the (now `Classic`) palette.
    #[serde(default)]
    palette: palette::Palette,
    /// Absent in v1/v2 bookmarks, which are always Mandelbrot.
    #[serde(default)]
    rule: FractalRule,
}

impl Default for ViewState {
    fn default() -> Self {
        Self {
            center: deep::BigComplex::from_f64(-0.5, 0.0),
            units_per_point: 0.004,
            max_iter: 300,
            palette_freq: 1.0,
            palette_phase: 0.0,
            palette: palette::Palette::Classic,
            rule: FractalRule::Mandelbrot,
        }
    }
}

/// Center serialization: decimal strings (v2 bookmarks). Deserialization also
/// accepts the v1 format, a plain [f64; 2].
mod center_serde {
    use crate::deep::BigComplex;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(c: &BigComplex, s: S) -> Result<S::Ok, S::Error> {
        c.to_decimal().serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<BigComplex, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Repr {
            Strings([String; 2]),
            Numbers([f64; 2]),
        }
        match Repr::deserialize(d)? {
            Repr::Strings([re, im]) => {
                // Generous parse precision; the app re-clamps to the zoom level.
                let bits = re.len().max(im.len()) * 4 + 64;
                BigComplex::from_decimal(&re, &im, bits).map_err(serde::de::Error::custom)
            }
            Repr::Numbers([re, im]) => Ok(BigComplex::from_f64(re, im)),
        }
    }
}

/// Versioned wrapper for the JSON embedded in exported PNGs.
#[derive(serde::Serialize, serde::Deserialize)]
struct Bookmark {
    app: String,
    version: u32,
    view: ViewState,
}

/// CPU-side cache describing the reference orbit currently on the GPU.
struct OrbitCache {
    /// Reference point the orbit was computed at.
    reference: deep::BigComplex,
    len: u32,
    escaped: bool,
    /// `max_iter` the orbit was computed for.
    for_max_iter: u32,
}

/// Progressive IFS render: the chaos game fills the histogram over multiple
/// frames (`done` of the target points so far); the tone-map is cheap and
/// re-runs on palette changes without touching the histogram.
struct IfsCache {
    // Histogram key (a change invalidates everything)
    center: [f64; 2],
    units_per_pixel: f64,
    size: [usize; 2],
    maps: Vec<ifs::AffineMap>,
    // Tone-map key
    palette: (palette::Palette, f32, f32),

    game: ifs::ChaosGame,
    done: u64,
    hist: Vec<u32>,
    texture: egui::TextureHandle,
}

/// Hover-linked Julia preview: a small corner overlay showing the Julia set
/// for the c under the cursor (Mandelbrot family only). Re-rendered only
/// when c, the palette, or the pane size changes — a fixed-iteration render
/// this small is a sub-millisecond GPU job.
struct JuliaPane {
    /// c/palette of the current texture contents (c is NaN until first render).
    rendered_c: [f32; 2],
    rendered_palette: (palette::Palette, f32, f32),
    size_px: u32,
    data_texture: wgpu::Texture,
    color_texture: wgpu::Texture,
    texture_id: egui::TextureId,
}

/// Julia-pane edge length in screen points.
const JULIA_PANE_POINTS: f32 = 200.0;
/// Iteration cap for the preview (fixed: it never deep-zooms).
const JULIA_PANE_ITERS: u32 = 300;

/// Where the Julia pane sits: bottom-right corner of the canvas.
fn julia_pane_rect(canvas: egui::Rect) -> egui::Rect {
    let size = egui::vec2(JULIA_PANE_POINTS, JULIA_PANE_POINTS);
    egui::Rect::from_min_size(canvas.right_bottom() - size - egui::vec2(12.0, 12.0), size)
}

/// Progressive strange-attractor render, mirroring `IfsCache`: deterministic
/// orbits fill the histogram over frames; the tone-map re-runs on palette
/// changes without re-iterating.
struct AttractorCache {
    // Histogram key (a change invalidates everything)
    kind: attractor::Kind,
    params: [f64; 4],
    center: [f64; 2],
    units_per_pixel: f64,
    size: [usize; 2],
    // Tone-map key
    palette: (palette::Palette, f32, f32),

    orbits: attractor::Orbits,
    done: u64,
    hist: Vec<u32>,
    texture: egui::TextureHandle,
}

/// Cached terrain/clouds render: a single deterministic full-frame pass,
/// re-run when the parameters, view, size, or palette change.
struct TerrainCache {
    params: terrain::Params,
    center: [f64; 2],
    units_per_pixel: f64,
    size: [usize; 2],
    palette: (palette::Palette, f32, f32),
    texture: egui::TextureHandle,
}

/// Cached L-system render: turtle segments (recomputed only when the rule
/// changes) plus the last rasterized image (recomputed when the view, size,
/// or palette changes — rasterizing is cheap next to expansion).
struct LsysCache {
    // Segment key
    axiom: String,
    rules: Vec<lsystem::Rule>,
    angle_deg: f64,
    generations: u32,
    // Image key (`size == [0, 0]` marks "never rasterized")
    center: [f64; 2],
    units_per_pixel: f64,
    size: [usize; 2],
    palette: (palette::Palette, f32, f32),

    segments: Vec<lsystem::Segment>,
    /// Generations actually expanded — less than requested when the
    /// `MAX_SYMBOLS` cap cut expansion short (the UI warns).
    generations_done: u32,
    /// World bbox of `segments`, cached for the zoom-out limit.
    world_bounds: Option<[f64; 4]>,
    texture: egui::TextureHandle,
}

/// Progressive Mandelbrot render: a resolution ladder. While the view
/// changes, only the coarsest level is rendered (cheap); once it settles,
/// each frame re-renders one level finer until full resolution. The data
/// (iteration) and color textures are separate so palette changes only
/// re-colorize.
/// An in-progress chunked rung: per-pixel iteration state on the GPU plus
/// the number of dispatches left. `ceil(max_iter / CHUNK_ITERS)` dispatches
/// guarantee every pixel resolves — no completion readback needed.
struct ChunkJob {
    state: wgpu::Buffer,
    dispatches_left: u32,
}

struct MandelProgressive {
    /// Iteration-uniform bytes + full-resolution pixel size: the view.
    key: ([u8; std::mem::size_of::<Uniforms>()], [u32; 2]),
    palette: (palette::Palette, f32, f32),
    /// Current rung: divisor is 1 << level, 0 = full resolution.
    level: u32,
    /// Chunked iteration in progress for the current rung.
    job: Option<ChunkJob>,
    /// World anchor of the rendered texture, for pan reprojection.
    center: deep::BigComplex,
    units_per_point: f64,
    data_texture: wgpu::Texture,
    color_texture: wgpu::Texture,
    texture_id: egui::TextureId,
    texture_size: [u32; 2],
}

/// A small delete/remove button with a painted ✕ — drawn with line segments
/// (like egui's own window close button) because the bundled fonts render
/// cross glyphs as tofu.
fn cross_button(ui: &mut egui::Ui) -> egui::Response {
    let response = ui.add(egui::Button::new("").min_size(egui::vec2(20.0, 16.0)));
    let rect = response.rect;
    let half = (rect.width().min(rect.height()) * 0.22).max(3.0);
    let c = rect.center();
    let stroke = ui.style().interact(&response).fg_stroke;
    ui.painter().line_segment(
        [c + egui::vec2(-half, -half), c + egui::vec2(half, half)],
        stroke,
    );
    ui.painter().line_segment(
        [c + egui::vec2(half, -half), c + egui::vec2(-half, half)],
        stroke,
    );
    response
}

/// One saved view in the journal: a thumbnail PNG on disk with the bookmark
/// embedded (the same format as full exports — the journal is just a folder
/// of small bookmark-PNGs).
struct JournalEntry {
    path: std::path::PathBuf,
    family: &'static str,
    texture: egui::TextureHandle,
}

/// Text buffers for exact coordinate entry (Mandelbrot). While `dirty`, the
/// user owns the strings; otherwise they mirror the current view each frame.
#[derive(Default)]
struct CoordEditor {
    re: String,
    im: String,
    zoom: String,
    dirty: bool,
}

struct App {
    view: ViewState,
    coord_edit: CoordEditor,
    /// Last known canvas size in points; export reproduces this framing.
    canvas_size: egui::Vec2,
    export_width: u32,
    export_height: u32,
    status: Option<String>,
    orbit: Option<OrbitCache>,
    ifs_cache: Option<IfsCache>,
    lsys_cache: Option<LsysCache>,
    attr_cache: Option<AttractorCache>,
    terrain_cache: Option<TerrainCache>,
    mandel_prog: Option<MandelProgressive>,
    show_julia_pane: bool,
    /// Pinned Julia-preview c (J toggles); None = track the cursor.
    julia_pin: Option<[f64; 2]>,
    julia_pane: Option<JuliaPane>,
    /// None until first scanned; then the gallery, newest first.
    journal: Option<Vec<JournalEntry>>,
    /// Bookmark mode: the main canvas shows the journal gallery instead of
    /// the fractal.
    journal_mode: bool,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let render_state = cc
            .wgpu_render_state
            .as_ref()
            .expect("wgpu render state (run with the wgpu backend)");
        render_state
            .renderer
            .write()
            .callback_resources
            .insert(RenderResources::new(&render_state.device));
        Self {
            view: ViewState::default(),
            coord_edit: CoordEditor::default(),
            canvas_size: egui::vec2(980.0, 800.0),
            export_width: 2560,
            export_height: 1440,
            status: None,
            orbit: None,
            ifs_cache: None,
            lsys_cache: None,
            attr_cache: None,
            terrain_cache: None,
            mandel_prog: None,
            show_julia_pane: true,
            julia_pin: None,
            julia_pane: None,
            journal: None,
            journal_mode: false,
        }
    }

    /// Interactive zoom-out limit. Escape-time sets and IFS attractors live
    /// at unit scale; L-system drawings (unit turtle steps) can span
    /// thousands of world units, so their limit scales with the drawing:
    /// zoomed all the way out it still covers ~100 screen points.
    fn max_units_per_point(&self) -> f64 {
        match &self.view.rule {
            FractalRule::LSystem { .. } => self
                .lsys_cache
                .as_ref()
                .and_then(|c| c.world_bounds)
                .map(|[min_x, min_y, max_x, max_y]| {
                    ((max_x - min_x).max(max_y - min_y) / 100.0).max(0.1)
                })
                .unwrap_or(0.1),
            _ => 0.1,
        }
    }

    fn perturbation_active(&self) -> bool {
        matches!(self.view.rule, FractalRule::Mandelbrot)
            && self.view.units_per_point < PERTURB_THRESHOLD
    }

    /// Fit the viewport to the IFS attractor's bounding box.
    fn fit_ifs_view(&mut self) {
        let FractalRule::Ifs { maps, .. } = &self.view.rule else {
            return;
        };
        if let Some([min_x, min_y, max_x, max_y]) = ifs::attractor_bbox(maps, 20_000) {
            let w = (max_x - min_x).max(1e-6);
            let h = (max_y - min_y).max(1e-6);
            self.view.center =
                deep::BigComplex::from_f64((min_x + max_x) * 0.5, (min_y + max_y) * 0.5);
            let upp_w = w / self.canvas_size.x.max(1.0) as f64;
            let upp_h = h / self.canvas_size.y.max(1.0) as f64;
            self.view.units_per_point = upp_w.max(upp_h) * 1.1;
        }
    }

    /// Fit the viewport to the strange attractor's bounding box.
    fn fit_attractor_view(&mut self) {
        let FractalRule::Attractor { kind, params, .. } = &self.view.rule else {
            return;
        };
        if let Some([min_x, min_y, max_x, max_y]) = attractor::bbox(*kind, *params, 20_000) {
            let w = (max_x - min_x).max(1e-6);
            let h = (max_y - min_y).max(1e-6);
            self.view.center =
                deep::BigComplex::from_f64((min_x + max_x) * 0.5, (min_y + max_y) * 0.5);
            let upp_w = w / self.canvas_size.x.max(1.0) as f64;
            let upp_h = h / self.canvas_size.y.max(1.0) as f64;
            self.view.units_per_point = upp_w.max(upp_h) * 1.1;
        }
    }

    /// Fit the viewport to the L-system drawing's bounding box.
    fn fit_lsystem_view(&mut self) {
        let FractalRule::LSystem {
            axiom,
            rules,
            angle_deg,
            generations,
        } = &self.view.rule
        else {
            return;
        };
        let (segs, _) = lsystem::segments(axiom, rules, *angle_deg, *generations);
        if let Some([min_x, min_y, max_x, max_y]) = lsystem::bounds(&segs) {
            let w = (max_x - min_x).max(1e-6);
            let h = (max_y - min_y).max(1e-6);
            self.view.center =
                deep::BigComplex::from_f64((min_x + max_x) * 0.5, (min_y + max_y) * 0.5);
            let upp_w = w / self.canvas_size.x.max(1.0) as f64;
            let upp_h = h / self.canvas_size.y.max(1.0) as f64;
            self.view.units_per_point = upp_w.max(upp_h) * 1.1;
        }
    }

    /// Keep the reference orbit valid for the current view; recompute on the
    /// CPU and upload when the view left its neighborhood or needs more
    /// iterations. Runs before the UI so the frame always renders with a
    /// valid orbit.
    fn maintain_orbit(&mut self, frame: &eframe::Frame) {
        if !self.perturbation_active() {
            return;
        }
        let prec = deep::precision_for(self.view.units_per_point);
        if self.view.center.re.precision() < prec {
            self.view.center.set_precision(prec);
        }

        let needs_recompute = match &self.orbit {
            None => true,
            Some(cache) => {
                let [dx, dy] = self.view.center.sub_to_f64(&cache.reference);
                let half_w = self.view.units_per_point * self.canvas_size.x as f64 * 0.5;
                let half_h = self.view.units_per_point * self.canvas_size.y as f64 * 0.5;
                let drifted = dx.abs() > half_w * 0.5 || dy.abs() > half_h * 0.5;
                let too_short = self.view.max_iter > cache.for_max_iter && !cache.escaped;
                drifted || too_short
            }
        };
        if !needs_recompute {
            return;
        }

        let Some(render_state) = frame.wgpu_render_state.as_ref() else {
            return;
        };
        let orbit = deep::reference_orbit(&self.view.center, self.view.max_iter, prec);
        let mut renderer = render_state.renderer.write();
        if let Some(resources) = renderer.callback_resources.get_mut::<RenderResources>() {
            resources.upload_orbit(&render_state.device, &render_state.queue, &orbit.points);
            self.orbit = Some(OrbitCache {
                reference: self.view.center.clone(),
                len: orbit.points.len() as u32,
                escaped: orbit.escaped,
                for_max_iter: self.view.max_iter,
            });
        }
    }

    /// Uniforms for a render whose vertical framing matches the live canvas,
    /// with width following the target's aspect ratio.
    fn uniforms_for_size(&self, width: f64, height: f64) -> Uniforms {
        let half_h = self.view.units_per_point * self.canvas_size.y as f64 * 0.5;
        let half_w = half_h * width / height;
        let center = self.view.center.to_f64();

        let (use_perturb, dc_offset, ref_len) = match &self.orbit {
            Some(cache) if self.perturbation_active() => {
                let [dx, dy] = self.view.center.sub_to_f64(&cache.reference);
                (1, [dx as f32, dy as f32], cache.len)
            }
            _ => (0, [0.0, 0.0], 0),
        };

        let (formula, power, julia_c) = match self.view.rule {
            FractalRule::Multibrot { power } => {
                (mandelbrot::FORMULA_MULTIBROT, power.max(2), [0.0; 2])
            }
            FractalRule::Julia { c } => {
                (mandelbrot::FORMULA_JULIA, 2, [c[0] as f32, c[1] as f32])
            }
            _ => (mandelbrot::FORMULA_MANDELBROT, 2, [0.0; 2]),
        };

        Uniforms {
            center: [center[0] as f32, center[1] as f32],
            half_extent: [half_w as f32, half_h as f32],
            dc_offset,
            julia_c,
            max_iter: self.view.max_iter,
            ref_len,
            use_perturb,
            formula,
            power,
            _pad: 0,
        }
    }

    /// A horizontal strip sampling the palette over one color cycle,
    /// with the frequency/phase sliders applied — live editor feedback.
    fn palette_preview(&self, ui: &mut egui::Ui) {
        let (rect, _) = ui.allocate_exact_size(
            egui::vec2(ui.available_width(), 14.0),
            egui::Sense::hover(),
        );
        if !ui.is_rect_visible(rect) {
            return;
        }
        const STEPS: usize = 64;
        let step_w = rect.width() / STEPS as f32;
        for i in 0..STEPS {
            let t = (i as f32 + 0.5) / STEPS as f32;
            let [r, g, b] =
                self.view
                    .palette
                    .eval(t, self.view.palette_freq, self.view.palette_phase);
            let x = rect.left() + i as f32 * step_w;
            ui.painter().rect_filled(
                egui::Rect::from_min_max(
                    egui::pos2(x, rect.top()),
                    // Slight overlap hides seams from rounding.
                    egui::pos2(x + step_w + 0.5, rect.bottom()),
                ),
                0.0,
                egui::Color32::from_rgb(r, g, b),
            );
        }
        ui.painter().rect_stroke(
            rect,
            2.0,
            ui.visuals().window_stroke(),
            egui::StrokeKind::Outside,
        );
    }

    fn palette_uniforms(&self) -> mandelbrot::PaletteUniforms {
        let [a, b, c, d] = self.view.palette.coeffs();
        mandelbrot::PaletteUniforms {
            a,
            b,
            c,
            d,
            freq: self.view.palette_freq,
            phase: self.view.palette_phase,
            _pad: [0.0; 2],
        }
    }

    /// Render the current view offscreen at the given size — the shared core
    /// of PNG export and journal thumbnails.
    fn render_pixels(&self, frame: &eframe::Frame, w: u32, h: u32) -> Result<Vec<u8>, String> {
        Ok(match &self.view.rule {
                FractalRule::Mandelbrot
                | FractalRule::Multibrot { .. }
                | FractalRule::Julia { .. } => {
                    let render_state = frame
                        .wgpu_render_state
                        .as_ref()
                        .ok_or("no wgpu render state")?;
                    let uniforms = self.uniforms_for_size(w as f64, h as f64);
                    let renderer = render_state.renderer.read();
                    let resources: &RenderResources = renderer
                        .callback_resources
                        .get()
                        .ok_or("render resources missing")?;
                    resources.render_offscreen(
                        &render_state.device,
                        &render_state.queue,
                        &uniforms,
                        &self.palette_uniforms(),
                        w,
                        h,
                    )
                }
                FractalRule::Ifs { maps, points } => {
                    // Same vertical framing as the canvas; scale the point
                    // budget with the pixel count so density is preserved.
                    let world_h = self.view.units_per_point * self.canvas_size.y as f64;
                    let view = ifs::IfsView {
                        center: self.view.center.to_f64(),
                        units_per_pixel: world_h / h as f64,
                    };
                    let canvas_px =
                        (self.canvas_size.x * self.canvas_size.y).max(1.0) as f64;
                    let scale = ((w * h) as f64 / canvas_px).max(1.0);
                    let points = ((*points as f64 * scale) as u64).min(200_000_000);
                    let hist =
                        ifs::chaos_histogram(maps, points, view, w as usize, h as usize);
                    ifs::tonemap_rgba(
                        &hist,
                        self.view.palette,
                        self.view.palette_freq,
                        self.view.palette_phase,
                    )
                }
                FractalRule::Attractor {
                    kind,
                    params,
                    points,
                } => {
                    // Same vertical framing as the canvas; point budget
                    // scaled to pixel count like IFS.
                    let world_h = self.view.units_per_point * self.canvas_size.y as f64;
                    let view = ifs::IfsView {
                        center: self.view.center.to_f64(),
                        units_per_pixel: world_h / h as f64,
                    };
                    let canvas_px =
                        (self.canvas_size.x * self.canvas_size.y).max(1.0) as f64;
                    let scale = ((w * h) as f64 / canvas_px).max(1.0);
                    let points = ((*points as f64 * scale) as u64).min(500_000_000);
                    let hist = attractor::histogram(
                        *kind,
                        *params,
                        points,
                        view,
                        w as usize,
                        h as usize,
                    );
                    ifs::tonemap_rgba(
                        &hist,
                        self.view.palette,
                        self.view.palette_freq,
                        self.view.palette_phase,
                    )
                }
                FractalRule::Terrain(params) => {
                    // Same vertical framing as the canvas.
                    let world_h = self.view.units_per_point * self.canvas_size.y as f64;
                    let view = ifs::IfsView {
                        center: self.view.center.to_f64(),
                        units_per_pixel: world_h / h as f64,
                    };
                    terrain::render_rgba(
                        params,
                        view,
                        w as usize,
                        h as usize,
                        self.view.palette,
                        self.view.palette_freq,
                        self.view.palette_phase,
                    )
                }
                FractalRule::LSystem {
                    axiom,
                    rules,
                    angle_deg,
                    generations,
                } => {
                    // Same vertical framing as the canvas.
                    let world_h = self.view.units_per_point * self.canvas_size.y as f64;
                    let view = lsystem::View {
                        center: self.view.center.to_f64(),
                        units_per_pixel: world_h / h as f64,
                    };
                    let (segs, _) = lsystem::segments(axiom, rules, *angle_deg, *generations);
                    lsystem::rasterize_rgba(
                        &segs,
                        view,
                        w as usize,
                        h as usize,
                        self.view.palette,
                        self.view.palette_freq,
                        self.view.palette_phase,
                    )
                }
            })
    }

    /// The current view as versioned bookmark JSON.
    fn bookmark_json(&self) -> Result<String, String> {
        let bookmark = Bookmark {
            app: "fractalx".to_owned(),
            version: 3,
            view: self.view.clone(),
        };
        serde_json::to_string(&bookmark).map_err(|e| e.to_string())
    }

    fn export_png(&mut self, frame: &eframe::Frame) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("PNG image", &["png"])
            .set_file_name("fractalx.png")
            .save_file()
        else {
            return;
        };

        let result = (|| -> Result<String, String> {
            let (w, h) = (self.export_width.max(16), self.export_height.max(16));
            let pixels = self.render_pixels(frame, w, h)?;
            export::save_png(&path, w, h, &pixels, &self.bookmark_json()?)?;
            Ok(format!("Exported {w}x{h} to {}", path.display()))
        })();

        self.status = Some(result.unwrap_or_else(|e| format!("Export failed: {e}")));
    }

    fn load_png_bookmark(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("PNG image", &["png"])
            .pick_file()
        else {
            return;
        };

        let result = export::load_bookmark_json(&path).and_then(|json| {
            serde_json::from_str::<Bookmark>(&json).map_err(|e| e.to_string())
        });
        self.status = Some(match result {
            Ok(bookmark) => {
                self.apply_bookmark(bookmark);
                format!("Restored view from {}", path.display())
            }
            Err(e) => format!("Load failed: {e}"),
        });
    }

    fn apply_bookmark(&mut self, bookmark: Bookmark) {
        self.view = bookmark.view;
        self.orbit = None; // force fresh caches
        self.ifs_cache = None;
        self.coord_edit.dirty = false;
    }

    /// Directory holding journal entries (bookmark-PNG thumbnails), in the
    /// platform's per-user app-data location.
    fn journal_dir() -> Result<std::path::PathBuf, String> {
        use std::path::PathBuf;
        let base = if cfg!(target_os = "macos") {
            std::env::var_os("HOME")
                .map(|h| PathBuf::from(h).join("Library/Application Support"))
        } else if cfg!(target_os = "windows") {
            std::env::var_os("APPDATA").map(PathBuf::from)
        } else {
            std::env::var_os("XDG_DATA_HOME")
                .map(PathBuf::from)
                .or_else(|| {
                    std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share"))
                })
        };
        base.map(|b| b.join("FractalX").join("journal"))
            .ok_or_else(|| "no home directory".to_owned())
    }

    /// Scan the journal directory into entries, newest first (timestamp
    /// filenames sort chronologically). Unreadable files are skipped.
    fn load_journal(ctx: &egui::Context) -> Vec<JournalEntry> {
        let Ok(dir) = Self::journal_dir() else {
            return vec![];
        };
        let Ok(read) = std::fs::read_dir(&dir) else {
            return vec![]; // not created yet: empty journal
        };
        let mut paths: Vec<_> = read
            .filter_map(|e| Some(e.ok()?.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "png"))
            .collect();
        paths.sort();
        paths.reverse();
        paths
            .into_iter()
            .filter_map(|path| {
                let (w, h, rgba) = export::read_png_rgba(&path).ok()?;
                let json = export::load_bookmark_json(&path).ok()?;
                let bookmark: Bookmark = serde_json::from_str(&json).ok()?;
                let texture = ctx.load_texture(
                    format!("journal:{}", path.display()),
                    egui::ColorImage::from_rgba_unmultiplied(
                        [w as usize, h as usize],
                        &rgba,
                    ),
                    egui::TextureOptions::LINEAR,
                );
                Some(JournalEntry {
                    path,
                    family: bookmark.view.rule.display_name(),
                    texture,
                })
            })
            .collect()
    }

    /// Render a thumbnail of the current view and save it (with the bookmark
    /// embedded) as a new journal entry.
    fn save_journal_entry(&mut self, frame: &eframe::Frame, ctx: &egui::Context) {
        let result = (|| -> Result<String, String> {
            // Thumbnail follows the canvas aspect so reopening reproduces
            // the framing the thumbnail shows.
            let w = 256u32;
            let aspect = (self.canvas_size.y / self.canvas_size.x.max(1.0)).clamp(0.25, 2.0);
            let h = ((w as f32 * aspect) as u32).max(16);
            let pixels = self.render_pixels(frame, w, h)?;

            let dir = Self::journal_dir()?;
            std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
            let millis = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| e.to_string())?
                .as_millis();
            let path = dir.join(format!("bm-{millis}.png"));
            export::save_png(&path, w, h, &pixels, &self.bookmark_json()?)?;

            let texture = ctx.load_texture(
                format!("journal:{}", path.display()),
                egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &pixels),
                egui::TextureOptions::LINEAR,
            );
            self.journal
                .get_or_insert_with(Vec::new)
                .insert(
                    0,
                    JournalEntry {
                        path,
                        family: self.view.rule.display_name(),
                        texture,
                    },
                );
            Ok("Saved to journal".to_owned())
        })();
        self.status = Some(result.unwrap_or_else(|e| format!("Journal save failed: {e}")));
    }

    /// Parse the coordinate editor and jump the view there.
    fn apply_coords(&mut self) {
        let result = (|| -> Result<(), String> {
            let zoom: f64 = self
                .coord_edit
                .zoom
                .trim()
                .parse()
                .map_err(|_| format!("bad zoom {:?}", self.coord_edit.zoom.trim()))?;
            if !zoom.is_finite() || zoom <= 0.0 {
                return Err("zoom must be positive".into());
            }
            let upp = (0.004 / zoom).clamp(MIN_UNITS_PER_POINT, 0.1);

            let (re, im) = (self.coord_edit.re.trim(), self.coord_edit.im.trim());
            // Precision to hold every typed digit, at least the zoom's needs.
            let bits = (re.len().max(im.len()) * 4 + 64).max(deep::precision_for(upp));
            let center = deep::BigComplex::from_decimal(re, im, bits)?;

            self.view.center = center;
            self.view.units_per_point = upp;
            self.orbit = None;
            Ok(())
        })();
        match result {
            Ok(()) => {
                self.coord_edit.dirty = false;
                self.status = Some("Jumped to entered coordinates".into());
            }
            Err(e) => self.status = Some(format!("Invalid coordinates: {e}")),
        }
    }

    /// Switch to the given IFS preset: rule and viewport.
    fn apply_preset(&mut self, preset: &ifs::Preset) {
        let points = match &self.view.rule {
            FractalRule::Ifs { points, .. } => *points,
            _ => 1_000_000,
        };
        self.view.rule = FractalRule::Ifs {
            maps: preset.maps.to_vec(),
            points,
        };
        self.view.center = deep::BigComplex::from_f64(preset.center[0], preset.center[1]);
        self.view.units_per_point = preset.units_per_point;
    }

    /// Switch to the given attractor preset and fit the viewport to it.
    fn apply_attractor_preset(&mut self, preset: &attractor::Preset) {
        let points = match &self.view.rule {
            FractalRule::Attractor { points, .. } => *points,
            _ => 5_000_000,
        };
        self.view.rule = FractalRule::Attractor {
            kind: preset.kind,
            params: preset.params,
            points,
        };
        self.fit_attractor_view();
    }

    /// Switch to the given L-system preset and fit the viewport to it.
    fn apply_lsystem_preset(&mut self, preset: &lsystem::Preset) {
        self.view.rule = FractalRule::LSystem {
            axiom: preset.axiom.into(),
            rules: preset.rules_vec(),
            angle_deg: preset.angle_deg,
            generations: preset.generations,
        };
        self.fit_lsystem_view();
    }

    fn controls(&mut self, ui: &mut egui::Ui, frame: &eframe::Frame) {
        ui.add_space(12.0);
        ui.horizontal(|ui| {
            ui.heading("FractalX");
            ui.small(concat!("v", env!("CARGO_PKG_VERSION")));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                egui::widgets::global_theme_preference_switch(ui);
            });
        });
        ui.add_space(16.0);

        // Family selection: a discriminant-only copy drives the combo box.
        let mut selected = std::mem::discriminant(&self.view.rule);
        let choices: [FractalRule; 7] = [
            FractalRule::Mandelbrot,
            FractalRule::Multibrot { power: 3 },
            FractalRule::Julia { c: [-0.8, 0.156] },
            FractalRule::Ifs {
                maps: vec![],
                points: 0,
            },
            FractalRule::LSystem {
                axiom: String::new(),
                rules: vec![],
                angle_deg: 0.0,
                generations: 0,
            },
            FractalRule::Attractor {
                kind: attractor::Kind::Clifford,
                params: [0.0; 4],
                points: 0,
            },
            FractalRule::Terrain(terrain::Params::default()),
        ];
        egui::ComboBox::from_label("Family")
            .selected_text(self.view.rule.display_name())
            .show_ui(ui, |ui| {
                for choice in &choices {
                    ui.selectable_value(
                        &mut selected,
                        std::mem::discriminant(choice),
                        choice.display_name(),
                    );
                }
            });
        if selected != std::mem::discriminant(&self.view.rule) {
            let chosen = choices
                .into_iter()
                .find(|c| std::mem::discriminant(c) == selected)
                .expect("selected comes from choices");
            if matches!(chosen, FractalRule::Ifs { .. }) {
                self.apply_preset(&ifs::PRESETS[0]);
            } else if matches!(chosen, FractalRule::LSystem { .. }) {
                self.apply_lsystem_preset(&lsystem::PRESETS[0]);
            } else if matches!(chosen, FractalRule::Attractor { .. }) {
                self.apply_attractor_preset(&attractor::PRESETS[0]);
            } else {
                let (center, upp) = chosen.home_view();
                self.view.rule = chosen;
                self.view.center = deep::BigComplex::from_f64(center[0], center[1]);
                self.view.units_per_point = upp;
                self.ifs_cache = None;
                self.lsys_cache = None;
                self.attr_cache = None;
                self.terrain_cache = None;
            }
            self.orbit = None;
            self.coord_edit.dirty = false;
        }
        ui.add_space(8.0);

        match &mut self.view.rule {
            FractalRule::Mandelbrot => {
                ui.label("Max iterations");
                ui.add(
                    egui::Slider::new(&mut self.view.max_iter, 50..=100_000).logarithmic(true),
                );
                ui.add_space(4.0);
                ui.checkbox(&mut self.show_julia_pane, "Julia preview (hover)");
                ui.small("J pins / unpins the preview point");
            }
            FractalRule::Multibrot { power } => {
                ui.label("Max iterations");
                ui.add(
                    egui::Slider::new(&mut self.view.max_iter, 50..=100_000).logarithmic(true),
                );
                ui.label("Power");
                ui.add(egui::Slider::new(power, 2..=8));
            }
            FractalRule::Julia { c } => {
                ui.label("Max iterations");
                ui.add(
                    egui::Slider::new(&mut self.view.max_iter, 50..=100_000).logarithmic(true),
                );
                ui.label("c  (z → z² + c)");
                ui.horizontal(|ui| {
                    ui.label("re");
                    ui.add(egui::DragValue::new(&mut c[0]).speed(0.001).max_decimals(5));
                    ui.label("im");
                    ui.add(egui::DragValue::new(&mut c[1]).speed(0.001).max_decimals(5));
                });
                ui.add_space(4.0);
                ui.label("Presets");
                ui.horizontal_wrapped(|ui| {
                    for (name, preset) in [
                        ("Dendrite", [-0.8, 0.156]),
                        ("Rabbit", [-0.123, 0.745]),
                        ("Siegel", [-0.391, -0.587]),
                        ("Spiral", [0.285, 0.01]),
                        ("Dust", [-0.4, 0.6]),
                        ("Basilica", [-1.0, 0.0]),
                        ("San Marco", [-0.75, 0.0]),
                        ("Airplane", [-1.755, 0.0]),
                        ("Galaxy", [-0.7269, 0.1889]),
                        ("Dragon", [-0.835, -0.2321]),
                        ("Lightning", [0.0, 1.0]),
                    ] {
                        if ui.small_button(name).clicked() {
                            *c = preset;
                        }
                    }
                });
            }
            FractalRule::Ifs { maps, points } => {
                ui.label("Points");
                ui.add(
                    egui::Slider::new(points, 10_000..=20_000_000)
                        .logarithmic(true),
                );
                ui.add_space(8.0);

                ui.label("Presets");
                let mut chosen: Option<&ifs::Preset> = None;
                ui.horizontal_wrapped(|ui| {
                    for preset in ifs::PRESETS {
                        if ui.small_button(preset.name).clicked() {
                            chosen = Some(preset);
                        }
                    }
                });

                ui.add_space(8.0);
                ui.label("Affine maps  (x' = a·x + b·y + e,  y' = c·x + d·y + f)");
                let mut remove: Option<usize> = None;
                for (i, m) in maps.iter_mut().enumerate() {
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        ui.monospace(format!("{i}"));
                        for (label, v) in [
                            ("a", &mut m.a),
                            ("b", &mut m.b),
                            ("c", &mut m.c),
                            ("d", &mut m.d),
                        ] {
                            ui.label(label);
                            ui.add(egui::DragValue::new(v).speed(0.01).max_decimals(3));
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.monospace(" ");
                        for (label, v) in [("e", &mut m.e), ("f", &mut m.f)] {
                            ui.label(label);
                            ui.add(egui::DragValue::new(v).speed(0.01).max_decimals(3));
                        }
                        ui.label("w");
                        ui.add(
                            egui::DragValue::new(&mut m.weight)
                                .speed(0.01)
                                .range(0.0..=100.0)
                                .max_decimals(3),
                        );
                        if cross_button(ui).clicked() {
                            remove = Some(i);
                        }
                    });
                }
                if let Some(i) = remove {
                    maps.remove(i);
                }
                if ui.small_button("+ Add map").clicked() {
                    maps.push(ifs::AffineMap {
                        a: 0.5,
                        b: 0.0,
                        c: 0.0,
                        d: 0.5,
                        e: 0.0,
                        f: 0.0,
                        weight: 1.0,
                    });
                }

                if let Some(preset) = chosen {
                    self.apply_preset(preset);
                }
            }
            FractalRule::LSystem {
                axiom,
                rules,
                angle_deg,
                generations,
            } => {
                // Snapshot to detect edits: the drawing's world-space size
                // changes with the rule, so any edit refits the viewport.
                let before = (axiom.clone(), rules.clone(), *angle_deg, *generations);
                ui.label("Generations");
                ui.add(egui::Slider::new(generations, 0..=20));
                ui.horizontal(|ui| {
                    ui.label("Angle");
                    ui.add(
                        egui::DragValue::new(angle_deg)
                            .speed(0.5)
                            .range(0.0..=180.0)
                            .suffix("°"),
                    );
                });
                ui.add_space(8.0);

                if let Some(cache) = &self.lsys_cache {
                    if cache.generations_done < *generations {
                        ui.label(format!(
                            "Symbol cap reached — showing generation {} of {}.",
                            cache.generations_done, generations
                        ));
                    }
                }
                ui.separator();

                ui.label("Presets");
                let mut chosen: Option<&lsystem::Preset> = None;
                ui.horizontal_wrapped(|ui| {
                    for preset in lsystem::PRESETS {
                        if ui.small_button(preset.name).clicked() {
                            chosen = Some(preset);
                        }
                    }
                });

                ui.add_space(8.0);
                ui.label("Axiom");
                ui.add(
                    egui::TextEdit::singleline(axiom).font(egui::TextStyle::Monospace),
                );
                ui.label("Rules");
                let mut remove: Option<usize> = None;
                for (i, r) in rules.iter_mut().enumerate() {
                    ui.horizontal(|ui| {
                        let mut sym = r.symbol.to_string();
                        let resp = ui.add(
                            egui::TextEdit::singleline(&mut sym)
                                .font(egui::TextStyle::Monospace)
                                .desired_width(16.0),
                        );
                        if resp.changed() {
                            if let Some(c) = sym.chars().last() {
                                r.symbol = c;
                            }
                        }
                        ui.label("→");
                        ui.add(
                            egui::TextEdit::singleline(&mut r.replacement)
                                .font(egui::TextStyle::Monospace)
                                .desired_width(f32::INFINITY),
                        );
                        if cross_button(ui).clicked() {
                            remove = Some(i);
                        }
                    });
                }
                if let Some(i) = remove {
                    rules.remove(i);
                }
                if ui.small_button("+ Add rule").clicked() {
                    rules.push(lsystem::Rule {
                        symbol: 'F',
                        replacement: "F".into(),
                    });
                }
                ui.add_space(4.0);
                ui.small("F G draw · f g move · + - turn · [ ] branch");

                let edited =
                    before != (axiom.clone(), rules.clone(), *angle_deg, *generations);
                if let Some(preset) = chosen {
                    self.apply_lsystem_preset(preset);
                } else if edited {
                    self.fit_lsystem_view();
                }
            }
            FractalRule::Attractor {
                kind,
                params,
                points,
            } => {
                ui.label("Points");
                ui.add(
                    egui::Slider::new(points, 100_000..=50_000_000)
                        .logarithmic(true),
                );
                ui.add_space(8.0);

                ui.label("Presets");
                let mut chosen: Option<&attractor::Preset> = None;
                ui.horizontal_wrapped(|ui| {
                    for preset in attractor::PRESETS {
                        if ui.small_button(preset.name).clicked() {
                            chosen = Some(preset);
                        }
                    }
                });

                ui.add_space(8.0);
                let before = (*kind, *params);
                ui.label(format!("{} map  (a b c d)", kind.name()));
                ui.horizontal(|ui| {
                    for p in params.iter_mut() {
                        ui.add(egui::DragValue::new(p).speed(0.005).max_decimals(3));
                    }
                });
                // Parameter edits re-fit the view: the attractor's extent
                // moves with its parameters.
                let edited = before != (*kind, *params);
                if let Some(preset) = chosen {
                    self.apply_attractor_preset(preset);
                } else if edited {
                    self.fit_attractor_view();
                }
            }
            FractalRule::Terrain(params) => {
                ui.horizontal(|ui| {
                    ui.label("Seed");
                    ui.add_sized(
                        [96.0, ui.spacing().interact_size.y],
                        egui::DragValue::new(&mut params.seed).speed(1),
                    );
                    if ui.small_button("Random").clicked() {
                        params.seed = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map(|d| d.as_nanos() as u64)
                            .unwrap_or(1);
                    }
                });
                ui.label("Roughness  (H)");
                ui.add(egui::Slider::new(&mut params.hurst, 0.05..=1.0));
                ui.small(format!("fractal dimension D = 3 − H ≈ {:.2}", 3.0 - params.hurst));
                ui.label("Octaves");
                ui.add(egui::Slider::new(&mut params.octaves, 1..=12));
                ui.add_space(4.0);
                ui.checkbox(&mut params.clouds, "Clouds (turbulence)");
            }
        }
        ui.add_space(8.0);
        ui.separator();
        ui.add_space(4.0);

        ui.strong("Palette");
        ui.add_space(4.0);
        egui::ComboBox::from_id_salt("palette-preset")
            .selected_text(self.view.palette.name())
            .show_ui(ui, |ui| {
                for p in palette::Palette::ALL {
                    ui.selectable_value(&mut self.view.palette, p, p.name());
                }
            });
        ui.add_space(4.0);
        self.palette_preview(ui);
        ui.label("Palette frequency");
        ui.add(egui::Slider::new(&mut self.view.palette_freq, 0.1..=8.0).logarithmic(true));
        ui.label("Palette phase");
        ui.add(egui::Slider::new(&mut self.view.palette_phase, 0.0..=1.0));
        egui::CollapsingHeader::new("Edit coefficients")
            .id_salt("palette-editor")
            .show(ui, |ui| {
                // Editing any coefficient turns the palette into `Custom`
                // seeded with the current preset's values.
                let mut rows = self.view.palette.coeffs();
                let mut changed = false;
                for (label, row) in [("a", 0), ("b", 1), ("c", 2), ("d", 3)] {
                    ui.horizontal(|ui| {
                        ui.monospace(label);
                        for channel in 0..3 {
                            changed |= ui
                                .add(
                                    egui::DragValue::new(&mut rows[row][channel])
                                        .speed(0.01)
                                        .max_decimals(3),
                                )
                                .changed();
                        }
                    });
                }
                ui.small("color = a + b·cos(2π(c·x + d)) per RGB channel");
                if changed {
                    self.view.palette =
                        palette::Palette::Custom(palette::Coeffs::from_rows(rows));
                }
            });

        ui.add_space(12.0);
        ui.separator();
        ui.add_space(4.0);
        ui.strong("Position");
        ui.add_space(4.0);
        let zoom = 0.004 / self.view.units_per_point;
        let digits = (zoom.log10().max(0.0) as usize) + 6;
        let [re, im] = self.view.center.to_decimal_digits(digits);
        if self.view.rule.is_escape_time() {
            {
                // Editable position: the fields mirror the view until typed
                // in, then hold the user's text until applied or reverted.
                if !self.coord_edit.dirty {
                    self.coord_edit.re = re;
                    self.coord_edit.im = im;
                    self.coord_edit.zoom = format!("{zoom:.3e}");
                }
                let ed = &mut self.coord_edit;
                let mut apply = false;
                for (label, buf) in
                    [("re", &mut ed.re), ("im", &mut ed.im), ("zoom", &mut ed.zoom)]
                {
                    ui.horizontal(|ui| {
                        ui.monospace(format!("{label:<4}"));
                        let r = ui.add(
                            egui::TextEdit::singleline(buf)
                                .font(egui::TextStyle::Monospace)
                                .desired_width(f32::INFINITY),
                        );
                        if r.changed() {
                            ed.dirty = true;
                        }
                        if r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                            apply = true;
                        }
                    });
                }
                // Always present (disabled while the fields mirror the view)
                // so the layout doesn't jump when typing starts.
                let dirty = self.coord_edit.dirty;
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(dirty, egui::Button::new("Go").small())
                        .clicked()
                    {
                        apply = true;
                    }
                    if ui
                        .add_enabled(dirty, egui::Button::new("Revert").small())
                        .clicked()
                    {
                        self.coord_edit.dirty = false;
                    }
                });
                if apply {
                    self.apply_coords();
                }
            }
        } else {
            ui.monospace(format!("x  {re}"));
            ui.monospace(format!("y  {im}"));
            ui.monospace(format!("zoom  {zoom:.3e}"));
        }

        ui.add_space(8.0);
        if ui.button("Reset view").clicked() {
            match &self.view.rule {
                FractalRule::Ifs { .. } => self.fit_ifs_view(),
                FractalRule::LSystem { .. } => self.fit_lsystem_view(),
                FractalRule::Attractor { .. } => self.fit_attractor_view(),
                _ => {
                    let (center, upp) = self.view.rule.home_view();
                    self.view.center = deep::BigComplex::from_f64(center[0], center[1]);
                    self.view.units_per_point = upp;
                }
            }
            self.orbit = None;
        }

        let density_progress = match (&self.view.rule, &self.ifs_cache, &self.attr_cache) {
            (FractalRule::Ifs { points, .. }, Some(cache), _) => Some((cache.done, *points)),
            (FractalRule::Attractor { points, .. }, _, Some(cache)) => {
                Some((cache.done, *points))
            }
            _ => None,
        };
        if let Some((done, points)) = density_progress {
            if done < points {
                ui.add_space(4.0);
                ui.add(
                    egui::ProgressBar::new(done as f32 / points.max(1) as f32)
                        .show_percentage(),
                );
            }
        }
        if self.perturbation_active() {
            ui.add_space(4.0);
            let pts = self.orbit.as_ref().map_or(0, |o| o.len);
            ui.small(format!("Deep zoom: perturbation ({pts} ref pts)"));
        }
        if matches!(self.view.rule, FractalRule::Mandelbrot)
            && self.view.units_per_point <= MIN_UNITS_PER_POINT * 1.01
        {
            ui.add_space(4.0);
            ui.colored_label(ui.visuals().warn_fg_color, "Zoom limit reached (~1e30).");
        }
        if self.view.rule.is_escape_time()
            && !matches!(self.view.rule, FractalRule::Mandelbrot)
            && zoom > 3.0e4
        {
            ui.add_space(4.0);
            ui.colored_label(
                ui.visuals().warn_fg_color,
                "Beyond f32 precision — deep zoom is Mandelbrot-only.",
            );
        }

        ui.add_space(12.0);
        ui.separator();
        ui.add_space(4.0);
        ui.strong("Export");
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.add(
                egui::DragValue::new(&mut self.export_width)
                    .range(16..=16384)
                    .speed(16),
            );
            ui.label("×");
            ui.add(
                egui::DragValue::new(&mut self.export_height)
                    .range(16..=16384)
                    .speed(16),
            );
            ui.label("px");
        });
        ui.add_space(4.0);
        if ui.button("Export PNG…").clicked() {
            self.export_png(frame);
        }
        if ui.button("Open PNG bookmark…").clicked() {
            self.load_png_bookmark();
        }
        if let Some(status) = &self.status {
            ui.add_space(4.0);
            ui.small(status.clone());
        }

        ui.add_space(12.0);
        ui.separator();
        ui.add_space(4.0);
        ui.strong("Journal");
        ui.add_space(4.0);
        // Disabled while the gallery covers the canvas: the button captures
        // the current fractal view, which isn't on screen in bookmark mode.
        if ui
            .add_enabled(
                !self.journal_mode,
                egui::Button::new("Save view to journal"),
            )
            .clicked()
        {
            self.save_journal_entry(frame, ui.ctx());
        }
        let label = if self.journal_mode {
            "Back to fractal"
        } else {
            "Bookmarks…"
        };
        if ui.button(label).clicked() {
            self.journal_mode = !self.journal_mode;
        }

        ui.add_space(12.0);
        ui.separator();
        ui.add_space(4.0);
        ui.small("Drag to pan · scroll or pinch to zoom");
        ui.add_space(12.0);
    }

    /// Bookmark mode: the journal gallery rendered in the main canvas area —
    /// a scrollable wrapped grid of clickable thumbnails.
    fn journal_gallery(&mut self, ui: &mut egui::Ui) {
        if self.journal.is_none() {
            self.journal = Some(Self::load_journal(ui.ctx()));
        }

        let mut open: Option<usize> = None;
        let mut delete: Option<usize> = None;
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                egui::Frame::NONE
                    .inner_margin(egui::Margin::same(16))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.heading("Journal");
                            ui.add_space(8.0);
                            ui.small("click a view to open it · Esc to go back");
                        });
                        ui.add_space(12.0);

                        let entries = self.journal.as_ref().expect("just ensured");
                        if entries.is_empty() {
                            ui.label("No saved views yet — use “Save view to journal”.");
                            return;
                        }
                        const THUMB_W: f32 = 200.0;
                        // Top-aligned wrap: cells must not be vertically
                        // centered per row, or unequal heights stagger them.
                        let layout = egui::Layout::left_to_right(egui::Align::Min)
                            .with_main_wrap(true);
                        ui.with_layout(layout, |ui| {
                            ui.spacing_mut().item_spacing = egui::vec2(12.0, 16.0);
                            for (i, entry) in entries.iter().enumerate() {
                                let tex_size = entry.texture.size_vec2();
                                let size = egui::vec2(
                                    THUMB_W,
                                    THUMB_W * tex_size.y / tex_size.x.max(1.0),
                                );
                                ui.allocate_ui(egui::vec2(THUMB_W, size.y + 28.0), |ui| {
                                    ui.spacing_mut().item_spacing.y = 6.0;
                                    ui.vertical(|ui| {
                                        if ui
                                            .add(
                                                egui::Image::new((
                                                    entry.texture.id(),
                                                    size,
                                                ))
                                                .sense(egui::Sense::click()),
                                            )
                                            .on_hover_cursor(
                                                egui::CursorIcon::PointingHand,
                                            )
                                            .on_hover_text("Open this view")
                                            .clicked()
                                        {
                                            open = Some(i);
                                        }
                                        ui.horizontal(|ui| {
                                            ui.small(entry.family);
                                            ui.with_layout(
                                                egui::Layout::right_to_left(
                                                    egui::Align::Center,
                                                ),
                                                |ui| {
                                                    if cross_button(ui)
                                                        .on_hover_text(
                                                            "Delete this entry",
                                                        )
                                                        .clicked()
                                                    {
                                                        delete = Some(i);
                                                    }
                                                },
                                            );
                                        });
                                    });
                                });
                            }
                        });
                    });
            });

        if let Some(i) = open {
            self.journal_mode = false;
            let path = self.journal.as_ref().expect("checked")[i].path.clone();
            let result = export::load_bookmark_json(&path).and_then(|json| {
                serde_json::from_str::<Bookmark>(&json).map_err(|e| e.to_string())
            });
            self.status = Some(match result {
                Ok(bookmark) => {
                    self.apply_bookmark(bookmark);
                    "Opened journal view".to_owned()
                }
                Err(e) => format!("Journal load failed: {e}"),
            });
        }
        if let Some(i) = delete {
            let entry = self.journal.as_mut().expect("checked").remove(i);
            std::fs::remove_file(&entry.path).ok();
        }
    }

    fn canvas(&mut self, ui: &mut egui::Ui, frame: &eframe::Frame) {
        let (rect, response) =
            ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());

        let prec = deep::precision_for(self.view.units_per_point);

        // Pan: screen x right = +re, screen y down = -im. Deltas are small
        // enough for f64; only the accumulated center needs high precision.
        if response.dragged() {
            let d = response.drag_delta();
            let upp = self.view.units_per_point;
            self.view
                .center
                .offset(-d.x as f64 * upp, d.y as f64 * upp, prec);
        }

        // Zoom (scroll wheel / trackpad pinch), anchored at the pointer.
        if response.hovered() {
            let (scroll, pinch) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
            let factor = pinch as f64 * (scroll as f64 * 0.005).exp();
            if (factor - 1.0).abs() > 1e-9 {
                if let Some(pos) = response.hover_pos() {
                    // Keep the complex point under the pointer fixed.
                    let off = pos - rect.center();
                    let upp = self.view.units_per_point;
                    let new_upp =
                        (upp / factor).clamp(MIN_UNITS_PER_POINT, self.max_units_per_point());
                    self.view.center.offset(
                        off.x as f64 * (upp - new_upp),
                        -off.y as f64 * (upp - new_upp),
                        deep::precision_for(new_upp),
                    );
                    self.view.units_per_point = new_upp;
                }
            }
        }

        self.canvas_size = rect.size();
        let dragging = response.dragged();
        match &self.view.rule {
            FractalRule::Ifs { .. } => self.paint_ifs(ui, rect),
            FractalRule::LSystem { .. } => self.paint_lsystem(ui, rect),
            FractalRule::Attractor { .. } => self.paint_attractor(ui, rect),
            FractalRule::Terrain(_) => self.paint_terrain(ui, rect),
            _ => self.paint_mandelbrot(ui, frame, rect, dragging),
        }

        if self.show_julia_pane && matches!(self.view.rule, FractalRule::Mandelbrot) {
            // c under the cursor; off-canvas (or over the pane itself, so
            // that mousing toward the pane doesn't drag c along) the pane
            // keeps its last c.
            let hover_c = response
                .hover_pos()
                .filter(|pos| !julia_pane_rect(rect).contains(*pos))
                .map(|pos| {
                    let off = pos - rect.center();
                    let upp = self.view.units_per_point;
                    let center = self.view.center.to_f64();
                    [center[0] + off.x as f64 * upp, center[1] - off.y as f64 * upp]
                });
            // J pins the preview to the current c (and unpins again), so a
            // find can be kept while the cursor moves on. Skipped while a
            // text field owns the keyboard.
            if !ui.ctx().egui_wants_keyboard_input()
                && ui.input(|i| i.key_pressed(egui::Key::J))
            {
                self.julia_pin = match self.julia_pin {
                    Some(_) => None,
                    None => hover_c.or_else(|| {
                        // Cursor off-canvas: pin what the pane is showing.
                        self.julia_pane.as_ref().and_then(|p| {
                            p.rendered_c[0]
                                .is_finite()
                                .then(|| [p.rendered_c[0] as f64, p.rendered_c[1] as f64])
                        })
                    }),
                };
            }
            let c = self.julia_pin.or(hover_c);
            self.paint_julia_pane(ui, frame, rect, c);
        }
    }

    /// Render (when needed) and draw the hover-linked Julia preview in the
    /// bottom-right corner of the canvas.
    fn paint_julia_pane(
        &mut self,
        ui: &mut egui::Ui,
        frame: &eframe::Frame,
        rect: egui::Rect,
        hover_c: Option<[f64; 2]>,
    ) {
        let Some(render_state) = frame.wgpu_render_state.as_ref() else {
            return;
        };
        let c = match (hover_c, &self.julia_pane) {
            (Some(c), _) => [c[0] as f32, c[1] as f32],
            (None, Some(pane)) if pane.rendered_c[0].is_finite() => pane.rendered_c,
            _ => return, // nothing hovered yet: no pane to show
        };
        let palette = (
            self.view.palette,
            self.view.palette_freq,
            self.view.palette_phase,
        );
        let palette_uniforms = self.palette_uniforms();
        let ppp = ui.ctx().pixels_per_point();
        let size_px = ((JULIA_PANE_POINTS * ppp) as u32).clamp(64, 1024);

        let device = &render_state.device;
        let queue = &render_state.queue;
        let mut renderer = render_state.renderer.write();

        if self.julia_pane.as_ref().is_none_or(|p| p.size_px != size_px) {
            let (data_texture, color_texture) = RenderResources::create_targets(
                device,
                size_px,
                size_px,
                wgpu::TextureUsages::empty(),
            );
            let color_view = color_texture.create_view(&Default::default());
            let texture_id =
                renderer.register_native_texture(device, &color_view, wgpu::FilterMode::Linear);
            if let Some(old) = self.julia_pane.take() {
                renderer.free_texture(&old.texture_id);
            }
            self.julia_pane = Some(JuliaPane {
                rendered_c: [f32::NAN; 2], // forces the first render
                rendered_palette: palette,
                size_px,
                data_texture,
                color_texture,
                texture_id,
            });
        }

        let pane = self.julia_pane.as_mut().expect("just ensured");
        if pane.rendered_c != c || pane.rendered_palette != palette {
            if let Some(resources) = renderer.callback_resources.get::<RenderResources>() {
                let uniforms = Uniforms {
                    center: [0.0, 0.0],
                    half_extent: [1.7, 1.7],
                    dc_offset: [0.0, 0.0],
                    julia_c: c,
                    max_iter: JULIA_PANE_ITERS,
                    ref_len: 0,
                    use_perturb: 0,
                    formula: mandelbrot::FORMULA_JULIA,
                    power: 2,
                    _pad: 0,
                };
                let data_view = pane.data_texture.create_view(&Default::default());
                let color_view = pane.color_texture.create_view(&Default::default());
                resources.render_data(device, queue, &uniforms, &data_view);
                resources.colorize(device, queue, &data_view, &palette_uniforms, &color_view);
                pane.rendered_c = c;
                pane.rendered_palette = palette;
            }
        }
        drop(renderer);

        let pane = self.julia_pane.as_ref().expect("just ensured");
        let pane_rect = julia_pane_rect(rect);
        ui.painter().image(
            pane.texture_id,
            pane_rect,
            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            egui::Color32::WHITE,
        );
        let stroke = if self.julia_pin.is_some() {
            ui.visuals().selection.stroke
        } else {
            ui.visuals().window_stroke()
        };
        ui.painter()
            .rect_stroke(pane_rect, 2.0, stroke, egui::StrokeKind::Outside);
        if self.julia_pin.is_some() {
            ui.painter().text(
                pane_rect.right_top() - egui::vec2(2.0, 4.0),
                egui::Align2::RIGHT_BOTTOM,
                "pinned · J",
                egui::FontId::proportional(11.0),
                stroke.color,
            );
        }

        // Click the pane (or its corner button) to open this c as a full
        // Julia view. The displayed c is the one that jumps.
        let displayed_c = [pane.rendered_c[0] as f64, pane.rendered_c[1] as f64];
        let pane_response = ui
            .interact(
                pane_rect,
                ui.id().with("julia-pane"),
                egui::Sense::click(),
            )
            .on_hover_cursor(egui::CursorIcon::PointingHand)
            .on_hover_text("Open as Julia view");
        let button_rect = egui::Rect::from_min_size(
            pane_rect.left_top() + egui::vec2(4.0, 4.0),
            egui::vec2(80.0, 18.0),
        );
        let button_clicked = ui
            .put(button_rect, egui::Button::new("Julia view").small())
            .clicked();
        if (pane_response.clicked() || button_clicked) && displayed_c[0].is_finite() {
            self.open_julia_view(displayed_c);
        }
    }

    /// Promote the previewed c to the full Julia family at its home view.
    fn open_julia_view(&mut self, c: [f64; 2]) {
        self.view.rule = FractalRule::Julia { c };
        let (center, upp) = self.view.rule.home_view();
        self.view.center = deep::BigComplex::from_f64(center[0], center[1]);
        self.view.units_per_point = upp;
        self.orbit = None;
        self.julia_pin = None;
        self.coord_edit.dirty = false;
    }

    /// Progressive Mandelbrot: render into a texture via a resolution ladder.
    /// A view change restarts at 1/4 resolution (cheap enough to keep any
    /// interaction fluid); each following frame re-renders one level finer.
    /// A palette-only change re-colorizes the existing data texture without
    /// re-iterating (and without restarting the ladder).
    fn paint_mandelbrot(
        &mut self,
        ui: &mut egui::Ui,
        frame: &eframe::Frame,
        rect: egui::Rect,
        dragging: bool,
    ) {
        let Some(render_state) = frame.wgpu_render_state.as_ref() else {
            return;
        };

        // While actively panning (same zoom), don't re-render at all: paint
        // the existing texture shifted by the pan delta. The ladder restarts
        // on release, so dragging shows the sharp image sliding instead of a
        // coarse re-render.
        if dragging {
            if let Some(prog) = &self.mandel_prog {
                if prog.units_per_point == self.view.units_per_point {
                    let [dx, dy] = self.view.center.sub_to_f64(&prog.center);
                    let upp = prog.units_per_point;
                    let off = egui::vec2((-dx / upp) as f32, (dy / upp) as f32);
                    ui.painter().rect_filled(rect, 0.0, egui::Color32::BLACK);
                    ui.painter().image(
                        prog.texture_id,
                        rect.translate(off),
                        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                        egui::Color32::WHITE,
                    );
                    return;
                }
            }
        }
        let ppp = ui.ctx().pixels_per_point();
        let full = [
            ((rect.width() * ppp) as u32).clamp(16, 8192),
            ((rect.height() * ppp) as u32).clamp(16, 8192),
        ];
        let uniforms = self.uniforms_for_size(rect.width() as f64, rect.height() as f64);
        let palette = (
            self.view.palette,
            self.view.palette_freq,
            self.view.palette_phase,
        );
        let mut key_bytes = [0u8; std::mem::size_of::<Uniforms>()];
        key_bytes.copy_from_slice(bytemuck::bytes_of(&uniforms));
        let key = (key_bytes, full);

        // What to do this frame. Same view: advance the chunk job if one is
        // running, else climb one rung, else recolor on palette change.
        // New view: restart at the coarsest rung.
        enum Action {
            StartRung(u32),
            AdvanceJob,
            Recolor,
            Nothing,
        }
        let action = match &self.mandel_prog {
            Some(p) if p.key == key => {
                if p.job.is_some() {
                    Action::AdvanceJob
                } else if p.level > 0 {
                    Action::StartRung(p.level - 1)
                } else if p.palette != palette {
                    Action::Recolor
                } else {
                    Action::Nothing
                }
            }
            _ => Action::StartRung(LADDER_START_LEVEL),
        };

        if !matches!(action, Action::Nothing) {
            let device = &render_state.device;
            let queue = &render_state.queue;
            let mut renderer = render_state.renderer.write();

            if let Action::StartRung(level) = action {
                let div = 1u32 << level;
                let size = [(full[0] / div).max(1), (full[1] / div).max(1)];
                let reuse = self
                    .mandel_prog
                    .as_ref()
                    .is_some_and(|p| p.texture_size == size);
                if !reuse {
                    let (data_texture, color_texture) = RenderResources::create_targets(
                        device,
                        size[0],
                        size[1],
                        wgpu::TextureUsages::empty(),
                    );
                    let color_view = color_texture.create_view(&Default::default());
                    let texture_id = renderer.register_native_texture(
                        device,
                        &color_view,
                        wgpu::FilterMode::Linear,
                    );
                    if let Some(old) = self.mandel_prog.take() {
                        renderer.free_texture(&old.texture_id);
                    }
                    self.mandel_prog = Some(MandelProgressive {
                        key,
                        palette,
                        level,
                        job: None,
                        center: self.view.center.clone(),
                        units_per_point: self.view.units_per_point,
                        data_texture,
                        color_texture,
                        texture_id,
                        texture_size: size,
                    });
                }
                let prog = self.mandel_prog.as_mut().expect("just ensured");
                prog.key = key;
                prog.level = level;
                prog.center = self.view.center.clone();
                prog.units_per_point = self.view.units_per_point;
                // High iteration caps run chunked so no dispatch stalls.
                prog.job = if uniforms.max_iter > CHUNK_ITERS {
                    Some(ChunkJob {
                        state: RenderResources::create_state_buffer(
                            device,
                            size[0] as u64 * size[1] as u64,
                        ),
                        dispatches_left: uniforms.max_iter.div_ceil(CHUNK_ITERS),
                    })
                } else {
                    None
                };
            }

            let prog = self.mandel_prog.as_mut().expect("exists for all actions");
            prog.palette = palette;
            let data_view = prog.data_texture.create_view(&Default::default());
            let color_view = prog.color_texture.create_view(&Default::default());
            if let Some(resources) = renderer.callback_resources.get::<RenderResources>() {
                match (&action, &mut prog.job) {
                    (Action::Recolor, _) => {}
                    (_, Some(job)) => {
                        let first = matches!(action, Action::StartRung(_));
                        resources.dispatch_chunk(
                            device,
                            queue,
                            &uniforms,
                            &mandelbrot::ChunkParams {
                                size: prog.texture_size,
                                chunk_iters: CHUNK_ITERS,
                                reset: first as u32,
                            },
                            &job.state,
                            &data_view,
                        );
                        job.dispatches_left -= 1;
                        if job.dispatches_left == 0 {
                            prog.job = None;
                        }
                    }
                    (_, None) => {
                        resources.render_data(device, queue, &uniforms, &data_view);
                    }
                }
                resources.colorize(
                    device,
                    queue,
                    &data_view,
                    &self.palette_uniforms(),
                    &color_view,
                );
            }
        }

        if let Some(prog) = &self.mandel_prog {
            ui.painter().image(
                prog.texture_id,
                rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );
            if prog.level > 0 || prog.job.is_some() {
                ui.ctx().request_repaint();
            }
        }
    }

    /// Progressive IFS: the chaos game accumulates a per-frame batch of
    /// points into the cached histogram until the target count is reached
    /// (the image "develops"). Palette changes only re-tone-map.
    fn paint_ifs(&mut self, ui: &mut egui::Ui, rect: egui::Rect) {
        let FractalRule::Ifs { maps, points } = &self.view.rule else {
            return;
        };
        let (maps, target) = (maps.clone(), *points);

        let ppp = ui.ctx().pixels_per_point() as f64;
        let w = ((rect.width() as f64 * ppp) as usize).clamp(16, 4096);
        let h = ((rect.height() as f64 * ppp) as usize).clamp(16, 4096);
        let center = self.view.center.to_f64();
        let units_per_pixel = self.view.units_per_point / ppp;
        let palette = (
            self.view.palette,
            self.view.palette_freq,
            self.view.palette_phase,
        );

        // Anything that changes the histogram restarts the accumulation.
        // (Lowering the target below what's done also restarts, to stay
        // deterministic.)
        let hist_valid = self.ifs_cache.as_ref().is_some_and(|c| {
            c.center == center
                && c.units_per_pixel == units_per_pixel
                && c.size == [w, h]
                && c.maps == maps
                && c.done <= target
        });
        if !hist_valid {
            let image = egui::ColorImage::filled([w, h], egui::Color32::BLACK);
            let texture =
                ui.ctx()
                    .load_texture("ifs-render", image, egui::TextureOptions::LINEAR);
            self.ifs_cache = Some(IfsCache {
                center,
                units_per_pixel,
                size: [w, h],
                maps: maps.clone(),
                palette,
                game: ifs::ChaosGame::new(),
                done: 0,
                hist: vec![0u32; w * h],
                texture,
            });
        }

        let cache = self.ifs_cache.as_mut().expect("just ensured");
        let mut retonemap = cache.palette != palette;
        if cache.done < target {
            let batch = IFS_POINTS_PER_FRAME.min(target - cache.done);
            cache.game.advance(
                &maps,
                ifs::IfsView {
                    center,
                    units_per_pixel,
                },
                w,
                h,
                &mut cache.hist,
                batch,
            );
            cache.done += batch;
            retonemap = true;
        }
        if retonemap {
            let rgba = ifs::tonemap_rgba(&cache.hist, palette.0, palette.1, palette.2);
            cache.texture.set(
                egui::ColorImage::from_rgba_unmultiplied([w, h], &rgba),
                egui::TextureOptions::LINEAR,
            );
            cache.palette = palette;
        }

        ui.painter().image(
            cache.texture.id(),
            rect,
            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            egui::Color32::WHITE,
        );
        if cache.done < target {
            ui.ctx().request_repaint();
        }
    }

    /// Progressive strange attractor, mirroring `paint_ifs`: deterministic
    /// orbits accumulate a per-frame point batch into the cached histogram;
    /// palette changes only re-tone-map.
    fn paint_attractor(&mut self, ui: &mut egui::Ui, rect: egui::Rect) {
        let FractalRule::Attractor {
            kind,
            params,
            points,
        } = &self.view.rule
        else {
            return;
        };
        let (kind, params, target) = (*kind, *params, *points);

        let ppp = ui.ctx().pixels_per_point() as f64;
        let w = ((rect.width() as f64 * ppp) as usize).clamp(16, 4096);
        let h = ((rect.height() as f64 * ppp) as usize).clamp(16, 4096);
        let center = self.view.center.to_f64();
        let units_per_pixel = self.view.units_per_point / ppp;
        let palette = (
            self.view.palette,
            self.view.palette_freq,
            self.view.palette_phase,
        );

        let hist_valid = self.attr_cache.as_ref().is_some_and(|c| {
            c.kind == kind
                && c.params == params
                && c.center == center
                && c.units_per_pixel == units_per_pixel
                && c.size == [w, h]
                && c.done <= target
        });
        if !hist_valid {
            let image = egui::ColorImage::filled([w, h], egui::Color32::BLACK);
            let texture =
                ui.ctx()
                    .load_texture("attractor-render", image, egui::TextureOptions::LINEAR);
            self.attr_cache = Some(AttractorCache {
                kind,
                params,
                center,
                units_per_pixel,
                size: [w, h],
                palette,
                orbits: attractor::Orbits::new(),
                done: 0,
                hist: vec![0u32; w * h],
                texture,
            });
        }

        let cache = self.attr_cache.as_mut().expect("just ensured");
        let mut retonemap = cache.palette != palette;
        if cache.done < target {
            let batch = IFS_POINTS_PER_FRAME.min(target - cache.done);
            cache.orbits.advance(
                kind,
                params,
                ifs::IfsView {
                    center,
                    units_per_pixel,
                },
                w,
                h,
                &mut cache.hist,
                batch,
            );
            cache.done += batch;
            retonemap = true;
        }
        if retonemap {
            let rgba = ifs::tonemap_rgba(&cache.hist, palette.0, palette.1, palette.2);
            cache.texture.set(
                egui::ColorImage::from_rgba_unmultiplied([w, h], &rgba),
                egui::TextureOptions::LINEAR,
            );
            cache.palette = palette;
        }

        ui.painter().image(
            cache.texture.id(),
            rect,
            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            egui::Color32::WHITE,
        );
        if cache.done < target {
            ui.ctx().request_repaint();
        }
    }

    /// Terrain/clouds: one deterministic full-frame render (rayon-parallel),
    /// re-run only when parameters, view, size, or palette change.
    fn paint_terrain(&mut self, ui: &mut egui::Ui, rect: egui::Rect) {
        let FractalRule::Terrain(params) = &self.view.rule else {
            return;
        };
        let params = *params;

        let ppp = ui.ctx().pixels_per_point() as f64;
        let w = ((rect.width() as f64 * ppp) as usize).clamp(16, 4096);
        let h = ((rect.height() as f64 * ppp) as usize).clamp(16, 4096);
        let center = self.view.center.to_f64();
        let units_per_pixel = self.view.units_per_point / ppp;
        let palette = (
            self.view.palette,
            self.view.palette_freq,
            self.view.palette_phase,
        );

        let valid = self.terrain_cache.as_ref().is_some_and(|c| {
            c.params == params
                && c.center == center
                && c.units_per_pixel == units_per_pixel
                && c.size == [w, h]
                && c.palette == palette
        });
        if !valid {
            let rgba = terrain::render_rgba(
                &params,
                ifs::IfsView {
                    center,
                    units_per_pixel,
                },
                w,
                h,
                palette.0,
                palette.1,
                palette.2,
            );
            let image = egui::ColorImage::from_rgba_unmultiplied([w, h], &rgba);
            match &mut self.terrain_cache {
                Some(cache) => {
                    cache.texture.set(image, egui::TextureOptions::LINEAR);
                    cache.params = params;
                    cache.center = center;
                    cache.units_per_pixel = units_per_pixel;
                    cache.size = [w, h];
                    cache.palette = palette;
                }
                None => {
                    let texture = ui.ctx().load_texture(
                        "terrain-render",
                        image,
                        egui::TextureOptions::LINEAR,
                    );
                    self.terrain_cache = Some(TerrainCache {
                        params,
                        center,
                        units_per_pixel,
                        size: [w, h],
                        palette,
                        texture,
                    });
                }
            }
        }

        let cache = self.terrain_cache.as_ref().expect("just ensured");
        ui.painter().image(
            cache.texture.id(),
            rect,
            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            egui::Color32::WHITE,
        );
    }

    /// Cached L-system render: segments rebuild only when the rule changes;
    /// the image re-rasterizes when the view, size, or palette changes.
    fn paint_lsystem(&mut self, ui: &mut egui::Ui, rect: egui::Rect) {
        let FractalRule::LSystem {
            axiom,
            rules,
            angle_deg,
            generations,
        } = &self.view.rule
        else {
            return;
        };
        let (axiom, rules, angle_deg, generations) =
            (axiom.clone(), rules.clone(), *angle_deg, *generations);

        let ppp = ui.ctx().pixels_per_point() as f64;
        let w = ((rect.width() as f64 * ppp) as usize).clamp(16, 4096);
        let h = ((rect.height() as f64 * ppp) as usize).clamp(16, 4096);
        let center = self.view.center.to_f64();
        let units_per_pixel = self.view.units_per_point / ppp;
        let palette = (
            self.view.palette,
            self.view.palette_freq,
            self.view.palette_phase,
        );

        let segs_valid = self.lsys_cache.as_ref().is_some_and(|c| {
            c.axiom == axiom
                && c.rules == rules
                && c.angle_deg == angle_deg
                && c.generations == generations
        });
        if !segs_valid {
            let (segments, generations_done) =
                lsystem::segments(&axiom, &rules, angle_deg, generations);
            let world_bounds = lsystem::bounds(&segments);
            let image = egui::ColorImage::filled([w, h], egui::Color32::BLACK);
            let texture =
                ui.ctx()
                    .load_texture("lsystem-render", image, egui::TextureOptions::LINEAR);
            self.lsys_cache = Some(LsysCache {
                axiom,
                rules,
                angle_deg,
                generations,
                center,
                units_per_pixel,
                size: [0, 0], // forces the first rasterize below
                palette,
                segments,
                generations_done,
                world_bounds,
                texture,
            });
        }

        let cache = self.lsys_cache.as_mut().expect("just ensured");
        if cache.center != center
            || cache.units_per_pixel != units_per_pixel
            || cache.size != [w, h]
            || cache.palette != palette
        {
            let rgba = lsystem::rasterize_rgba(
                &cache.segments,
                lsystem::View {
                    center,
                    units_per_pixel,
                },
                w,
                h,
                palette.0,
                palette.1,
                palette.2,
            );
            cache.texture.set(
                egui::ColorImage::from_rgba_unmultiplied([w, h], &rgba),
                egui::TextureOptions::LINEAR,
            );
            cache.center = center;
            cache.units_per_pixel = units_per_pixel;
            cache.size = [w, h];
            cache.palette = palette;
        }

        ui.painter().image(
            cache.texture.id(),
            rect,
            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            egui::Color32::WHITE,
        );
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let frame: &eframe::Frame = frame;
        self.maintain_orbit(frame);

        // Exact width: content (e.g. focused full-width text fields) must
        // never widen the panel over the canvas.
        // Asymmetric: roomy on the text side, snug where the scrollbar
        // meets the canvas edge.
        let panel_frame = egui::Frame::side_top_panel(ui.style()).inner_margin(egui::Margin {
            left: 16,
            right: 4,
            top: 8,
            bottom: 8,
        });
        egui::Panel::left("controls")
            .resizable(false)
            .exact_size(260.0)
            .frame(panel_frame)
            .show(ui, |ui| {
                // Scroll instead of clipping when the window is too short
                // for the full control stack. Solid (not floating) bars:
                // floating bars overlay the content on hover; solid ones get
                // their own reserved column, plus a gap before the content.
                ui.spacing_mut().scroll = egui::style::ScrollStyle::solid();
                ui.spacing_mut().scroll.bar_inner_margin = 12.0;
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    // Always reserve the bar column so the content width
                    // doesn't shift when scrolling becomes necessary.
                    .scroll_bar_visibility(
                        egui::scroll_area::ScrollBarVisibility::AlwaysVisible,
                    )
                    .show(ui, |ui| self.controls(ui, frame));
            });

        // Esc leaves bookmark mode.
        if self.journal_mode
            && !ui.ctx().egui_wants_keyboard_input()
            && ui.input(|i| i.key_pressed(egui::Key::Escape))
        {
            self.journal_mode = false;
        }

        let canvas_frame = if self.journal_mode {
            egui::Frame::central_panel(ui.style())
        } else {
            egui::Frame::NONE
        };
        egui::CentralPanel::default()
            .frame(canvas_frame)
            .show(ui, |ui| {
                if self.journal_mode {
                    self.journal_gallery(ui);
                } else {
                    self.canvas(ui, frame);
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v2_bookmark_without_rule_loads_as_mandelbrot() {
        let json = r#"{"app":"fractalx","version":2,"view":{
            "center":["-0.75","0.1"],"units_per_point":1e-10,
            "max_iter":5000,"palette_freq":1.0,"palette_phase":0.0}}"#;
        let b: Bookmark = serde_json::from_str(json).unwrap();
        assert!(matches!(b.view.rule, FractalRule::Mandelbrot));
        assert_eq!(b.view.max_iter, 5000);
        assert_eq!(b.view.palette, palette::Palette::Classic);
    }

    #[test]
    fn v3_ifs_bookmark_round_trips() {
        let view = ViewState {
            rule: FractalRule::Ifs {
                maps: ifs::PRESETS[1].maps.to_vec(),
                points: 2_000_000,
            },
            center: deep::BigComplex::from_f64(0.25, 5.0),
            units_per_point: 0.0145,
            ..ViewState::default()
        };
        let json = serde_json::to_string(&Bookmark {
            app: "fractalx".into(),
            version: 3,
            view,
        })
        .unwrap();
        let back: Bookmark = serde_json::from_str(&json).unwrap();
        match back.view.rule {
            FractalRule::Ifs { maps, points } => {
                assert_eq!(points, 2_000_000);
                assert_eq!(maps, ifs::PRESETS[1].maps.to_vec());
            }
            _ => panic!("rule family lost in round trip"),
        }
    }

    #[test]
    fn julia_and_attractor_bookmarks_round_trip() {
        for (rule, tag) in [
            (FractalRule::Julia { c: [-0.8, 0.156] }, r#""family":"julia""#),
            (
                FractalRule::Attractor {
                    kind: attractor::Kind::DeJong,
                    params: [-2.7, -0.09, -0.86, -2.2],
                    points: 5_000_000,
                },
                r#""family":"attractor""#,
            ),
        ] {
            let view = ViewState {
                rule: rule.clone(),
                ..ViewState::default()
            };
            let json = serde_json::to_string(&Bookmark {
                app: "fractalx".into(),
                version: 3,
                view,
            })
            .unwrap();
            assert!(json.contains(tag), "{json}");
            let back: Bookmark = serde_json::from_str(&json).unwrap();
            assert!(back.view.rule == rule, "rule lost in round trip: {json}");
        }
    }

    #[test]
    fn terrain_bookmark_round_trips() {
        let rule = FractalRule::Terrain(terrain::Params {
            seed: 42,
            hurst: 0.65,
            octaves: 9,
            clouds: true,
        });
        let view = ViewState {
            rule: rule.clone(),
            ..ViewState::default()
        };
        let json = serde_json::to_string(&Bookmark {
            app: "fractalx".into(),
            version: 3,
            view,
        })
        .unwrap();
        assert!(json.contains(r#""family":"terrain""#), "{json}");
        let back: Bookmark = serde_json::from_str(&json).unwrap();
        assert!(back.view.rule == rule, "rule lost in round trip: {json}");
    }

    #[test]
    fn lsystem_bookmark_round_trips() {
        let preset = &lsystem::PRESETS[0];
        let view = ViewState {
            rule: FractalRule::LSystem {
                axiom: preset.axiom.into(),
                rules: preset.rules_vec(),
                angle_deg: preset.angle_deg,
                generations: preset.generations,
            },
            ..ViewState::default()
        };
        let json = serde_json::to_string(&Bookmark {
            app: "fractalx".into(),
            version: 3,
            view,
        })
        .unwrap();
        assert!(json.contains(r#""family":"l_system""#), "{json}");
        let back: Bookmark = serde_json::from_str(&json).unwrap();
        match back.view.rule {
            FractalRule::LSystem {
                axiom,
                rules,
                angle_deg,
                generations,
            } => {
                assert_eq!(axiom, preset.axiom);
                assert_eq!(rules, preset.rules_vec());
                assert_eq!(angle_deg, preset.angle_deg);
                assert_eq!(generations, preset.generations);
            }
            _ => panic!("rule family lost in round trip"),
        }
    }
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 700.0])
            .with_title("FractalX — Fractal Explorer"),
        ..Default::default()
    };
    eframe::run_native(
        "FractalX",
        options,
        Box::new(|cc| Ok(Box::new(App::new(cc)))),
    )
}
