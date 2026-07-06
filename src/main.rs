//! FractalX — fractal explorer prototype.
//! Milestone 1: GPU Mandelbrot with smooth pan/zoom, iteration and palette controls.

mod export;
mod mandelbrot;

use eframe::egui;
use mandelbrot::{MandelbrotCallback, RenderResources, Uniforms};

/// Complete view state — the "bookmark": it fully determines a render.
#[derive(Clone, Copy, serde::Serialize, serde::Deserialize)]
struct ViewState {
    /// Complex-plane coordinates of the canvas center. f64 so panning stays
    /// stable ahead of the shader's f32 limit.
    center: [f64; 2],
    /// Complex units per screen point (zoom level).
    units_per_point: f64,
    max_iter: u32,
    palette_freq: f32,
    palette_phase: f32,
}

impl Default for ViewState {
    fn default() -> Self {
        Self {
            center: [-0.5, 0.0],
            units_per_point: 0.004,
            max_iter: 300,
            palette_freq: 1.0,
            palette_phase: 0.0,
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

struct App {
    view: ViewState,
    /// Last known canvas size in points; export reproduces this framing.
    canvas_size: egui::Vec2,
    export_width: u32,
    export_height: u32,
    status: Option<String>,
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
        }
    }

    /// Uniforms for a render whose vertical framing matches the live canvas,
    /// with width following the target's aspect ratio.
    fn uniforms_for_size(&self, width: f64, height: f64) -> Uniforms {
        let half_h = self.view.units_per_point * self.canvas_size.y as f64 * 0.5;
        let half_w = half_h * width / height;
        Uniforms {
            center: [self.view.center[0] as f32, self.view.center[1] as f32],
            half_extent: [half_w as f32, half_h as f32],
            max_iter: self.view.max_iter,
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
                version: 1,
                view: self.view,
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
            egui::Slider::new(&mut self.view.max_iter, 50..=5000)
                .logarithmic(true),
        );
        ui.add_space(8.0);

        ui.label("Palette frequency");
        ui.add(egui::Slider::new(&mut self.view.palette_freq, 0.1..=8.0).logarithmic(true));
        ui.label("Palette phase");
        ui.add(egui::Slider::new(&mut self.view.palette_phase, 0.0..=1.0));
        ui.add_space(12.0);

        if ui.button("Reset view").clicked() {
            let (freq, phase, iters) = (
                self.view.palette_freq,
                self.view.palette_phase,
                self.view.max_iter,
            );
            self.view = ViewState {
                palette_freq: freq,
                palette_phase: phase,
                max_iter: iters,
                ..ViewState::default()
            };
        }

        ui.add_space(12.0);
        ui.separator();
        ui.add_space(4.0);
        let zoom = 0.004 / self.view.units_per_point;
        ui.monospace(format!("re  {:+.12}", self.view.center[0]));
        ui.monospace(format!("im  {:+.12}", self.view.center[1]));
        ui.monospace(format!("zoom  {:.3e}", zoom));
        if zoom > 3.0e4 {
            ui.add_space(4.0);
            ui.colored_label(
                ui.visuals().warn_fg_color,
                "Beyond f32 precision — pixelation expected. Deep zoom comes later.",
            );
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

        // Pan: screen x right = +re, screen y down = -im.
        if response.dragged() {
            let d = response.drag_delta();
            self.view.center[0] -= d.x as f64 * self.view.units_per_point;
            self.view.center[1] += d.y as f64 * self.view.units_per_point;
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
                    let new_upp = upp / factor;
                    self.view.center[0] += off.x as f64 * (upp - new_upp);
                    self.view.center[1] -= off.y as f64 * (upp - new_upp);
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
