//! FractalX — fractal explorer prototype.
//! Milestone 1: GPU Mandelbrot with smooth pan/zoom, iteration and palette controls.

mod deep;
mod export;
mod mandelbrot;

use eframe::egui;
use mandelbrot::{MandelbrotCallback, RenderResources, Uniforms};

/// Once units-per-point drops below this, f32 in the shader is out of bits
/// and rendering switches to the perturbation path.
const PERTURB_THRESHOLD: f64 = 1e-7;
/// Hard zoom floor: below this even f32 pixel deltas underflow (~1e30 zoom).
const MIN_UNITS_PER_POINT: f64 = 1e-32;

/// Complete view state — the "bookmark": it fully determines a render.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct ViewState {
    /// Complex-plane coordinates of the canvas center, arbitrary precision.
    #[serde(with = "center_serde")]
    center: deep::BigComplex,
    /// Complex units per screen point (zoom level).
    units_per_point: f64,
    max_iter: u32,
    palette_freq: f32,
    palette_phase: f32,
}

impl Default for ViewState {
    fn default() -> Self {
        Self {
            center: deep::BigComplex::from_f64(-0.5, 0.0),
            units_per_point: 0.004,
            max_iter: 300,
            palette_freq: 1.0,
            palette_phase: 0.0,
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

struct App {
    view: ViewState,
    /// Last known canvas size in points; export reproduces this framing.
    canvas_size: egui::Vec2,
    export_width: u32,
    export_height: u32,
    status: Option<String>,
    orbit: Option<OrbitCache>,
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
            .insert(RenderResources::new(
                &render_state.device,
                render_state.target_format,
            ));
        Self {
            view: ViewState::default(),
            canvas_size: egui::vec2(980.0, 800.0),
            export_width: 2560,
            export_height: 1440,
            status: None,
            orbit: None,
        }
    }

    fn perturbation_active(&self) -> bool {
        self.view.units_per_point < PERTURB_THRESHOLD
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

        Uniforms {
            center: [center[0] as f32, center[1] as f32],
            half_extent: [half_w as f32, half_h as f32],
            dc_offset,
            max_iter: self.view.max_iter,
            ref_len,
            use_perturb,
            palette_freq: self.view.palette_freq,
            palette_phase: self.view.palette_phase,
            _pad: 0.0,
        }
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
            let render_state = frame
                .wgpu_render_state
                .as_ref()
                .ok_or("no wgpu render state")?;
            let (w, h) = (self.export_width.max(16), self.export_height.max(16));
            let uniforms = self.uniforms_for_size(w as f64, h as f64);

            let renderer = render_state.renderer.read();
            let resources: &RenderResources = renderer
                .callback_resources
                .get()
                .ok_or("render resources missing")?;
            let pixels = resources.render_offscreen(
                &render_state.device,
                &render_state.queue,
                &uniforms,
                w,
                h,
            );
            drop(renderer);

            let bookmark = Bookmark {
                app: "fractalx".to_owned(),
                version: 2,
                view: self.view.clone(),
            };
            let json = serde_json::to_string(&bookmark).map_err(|e| e.to_string())?;
            export::save_png(&path, w, h, &pixels, &json)?;
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
                self.view = bookmark.view;
                self.orbit = None; // force a fresh reference orbit
                format!("Restored view from {}", path.display())
            }
            Err(e) => format!("Load failed: {e}"),
        });
    }

    fn controls(&mut self, ui: &mut egui::Ui, frame: &eframe::Frame) {
        ui.heading("Mandelbrot");
        ui.add_space(8.0);

        ui.label("Max iterations");
        ui.add(
            egui::Slider::new(&mut self.view.max_iter, 50..=100_000)
                .logarithmic(true),
        );
        ui.add_space(8.0);

        ui.label("Palette frequency");
        ui.add(egui::Slider::new(&mut self.view.palette_freq, 0.1..=8.0).logarithmic(true));
        ui.label("Palette phase");
        ui.add(egui::Slider::new(&mut self.view.palette_phase, 0.0..=1.0));
        ui.add_space(12.0);

        if ui.button("Reset view").clicked() {
            self.view = ViewState {
                palette_freq: self.view.palette_freq,
                palette_phase: self.view.palette_phase,
                max_iter: self.view.max_iter,
                ..ViewState::default()
            };
            self.orbit = None;
        }

        ui.add_space(12.0);
        ui.separator();
        ui.add_space(4.0);
        let zoom = 0.004 / self.view.units_per_point;
        let digits = (zoom.log10().max(0.0) as usize) + 6;
        let [re, im] = self.view.center.to_decimal_digits(digits);
        ui.monospace(format!("re  {re}"));
        ui.monospace(format!("im  {im}"));
        ui.monospace(format!("zoom  {zoom:.3e}"));
        if self.perturbation_active() {
            ui.add_space(4.0);
            let pts = self.orbit.as_ref().map_or(0, |o| o.len);
            ui.small(format!("Deep zoom: perturbation ({pts} ref pts)"));
        }
        if self.view.units_per_point <= MIN_UNITS_PER_POINT * 1.01 {
            ui.add_space(4.0);
            ui.colored_label(ui.visuals().warn_fg_color, "Zoom limit reached (~1e30).");
        }

        ui.add_space(12.0);
        ui.separator();
        ui.add_space(4.0);
        ui.heading("Export");
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
        ui.small("Drag to pan · scroll or pinch to zoom");
    }

    fn canvas(&mut self, ui: &mut egui::Ui) {
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
                    let new_upp = (upp / factor).clamp(MIN_UNITS_PER_POINT, 0.1);
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
        let uniforms = self.uniforms_for_size(rect.width() as f64, rect.height() as f64);

        ui.painter().add(eframe::egui_wgpu::Callback::new_paint_callback(
            rect,
            MandelbrotCallback { uniforms },
        ));
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        self.maintain_orbit(frame);

        egui::Panel::left("controls")
            .resizable(false)
            .default_size(220.0)
            .show(ui, |ui| self.controls(ui, frame));

        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ui, |ui| self.canvas(ui));
    }
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_title("FractalX — Fractal Explorer"),
        ..Default::default()
    };
    eframe::run_native(
        "FractalX",
        options,
        Box::new(|cc| Ok(Box::new(App::new(cc)))),
    )
}
