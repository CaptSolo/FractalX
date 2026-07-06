# FractalX

A fast, native fractal explorer for macOS, written in Rust with GPU rendering
(wgpu/Metal) and an egui interface.

Zoom into the Mandelbrot set down to **~10³⁰×** magnification — far beyond
double precision — and export any view as a PNG that carries its exact
coordinates inside, so every image can be reopened right where it was taken.

![FractalX exploring the Mandelbrot set](assets/full_ui_view.png)

*Self-similarity in action: a minibrot with its own halo of filaments, found by
zooming in.*

![A minibrot found at depth](assets/fractal_fragment.png)

## Features

- **GPU-rendered Mandelbrot explorer** — smooth pan (drag) and zoom
  (scroll/pinch, anchored at the pointer), iteration depth up to 100,000,
  adjustable cyclic color palette.
- **Deep zoom via perturbation theory** — a single reference orbit is computed
  on the CPU in arbitrary-precision arithmetic; the GPU iterates only each
  pixel's tiny delta in f32, with rebasing. Precision scales automatically
  with zoom depth. Works to ~10³⁰×, where f32 pixel deltas finally underflow.
- **PNG export with embedded bookmarks** — renders offscreen at any resolution
  (up to 16K), independent of the window. The complete view state (center
  coordinates at full precision, zoom, iterations, palette) is embedded in the
  PNG as an `iTXt` metadata chunk. *Open PNG bookmark…* jumps back to exactly
  that view.
- **Reproducible by construction** — a bookmark fully determines a render;
  there is no hidden state.

## Building

Requires a Rust toolchain (stable) on macOS.

```sh
cargo run --release
```

Debug builds work but render noticeably slower; use `--release` for exploring.

Run the test suite (includes headless GPU tests, so a Metal-capable machine is
needed):

```sh
cargo test
```

## Usage

| Action | Input |
|---|---|
| Pan | drag the canvas |
| Zoom | scroll wheel or trackpad pinch (anchored at pointer) |
| Iterations / palette | sliders in the left panel |
| Export image | set resolution, *Export PNG…* |
| Reopen an exported view | *Open PNG bookmark…* |
| Back to the full set | *Reset view* |

When you zoom past ~3×10⁴×, the renderer switches to the perturbation path
automatically (shown in the panel). If detail dissolves into flat color at
extreme depth, raise the iteration slider — deep locations often need tens of
thousands of iterations.

## How deep zoom works

Standard f32 GPU math runs out of bits at about 10⁴–10⁵× zoom, and Metal has
no shader f64. FractalX uses the modern deep-zoom technique instead:

1. The view center is tracked in arbitrary-precision floats
   ([dashu](https://crates.io/crates/dashu)), with precision growing with zoom
   (pixel scale + 64 guard bits).
2. One **reference orbit** is iterated at that precision on the CPU.
3. Each pixel iterates only its **delta** from the reference in f32 on the
   GPU (`δ′ = δ(2Z + δ) + δc`), rebasing to the orbit start whenever the delta
   outgrows the reference — which also handles the classic glitch cases.

The reference orbit is cached and only recomputed when the view drifts away
from it, so panning stays smooth even at extreme depth.

## Project status & roadmap

Early but functional prototype — part of a larger concept for exploring
self-similarity (escape-time fractals, IFS/L-systems, and statistical
fractals); see [CONCEPT.md](CONCEPT.md) for the full vision and current
implementation status.

Planned next: progressive/cancellable rendering, a hover-linked Julia
companion pane, a bookmarks journal with thumbnails, palette editor, more
formulas (Burning Ship, Multibrot, custom expressions).

## Tech

Rust · [wgpu](https://wgpu.rs) (Metal) · [egui/eframe](https://github.com/emilk/egui) ·
[dashu](https://crates.io/crates/dashu) arbitrary precision · WGSL shaders
