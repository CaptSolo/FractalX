# FractalX

Desktop fractal explorer (macOS-first): Rust + wgpu + egui (eframe). Vision,
scope, and implementation status live in `CONCEPT.md` — §10 tracks what is
built vs. pending; keep it updated when a milestone lands.

## Commands

- Build: `cargo build` · Run: `cargo run --release` (debug builds render slowly)
- Test: `cargo test` — includes headless GPU tests (need a Metal adapter; no
  GPU-less CI)
- The binary is `fractalx`; check liveness with `pgrep -x fractalx`

## Architecture

- `src/main.rs` — app shell, `ViewState` (the "bookmark": fully determines a
  render), interaction, orbit cache maintenance, export/load UI.
- `src/mandelbrot.rs` — wgpu pipelines (live + RGBA8 export), reference-orbit
  storage buffer, offscreen render with readback. Uniform struct layout must
  match `src/shaders/mandelbrot.wgsl` byte-for-byte.
- `src/deep.rs` — arbitrary-precision (dashu) center coordinates and the
  perturbation reference orbit. Precision auto-scales: zoom bits + 64 guard bits.
- `src/export.rs` — PNG with the bookmark embedded as an iTXt chunk.
- Spec rule: fractal families are pluggable modules; adding a family must not
  touch core code (`main.rs` state/scheduling, export).

## Deep zoom (perturbation)

- Plain f32 shader path below ~3e4 zoom; perturbation path beyond
  (`PERTURB_THRESHOLD`): CPU computes one high-precision reference orbit, GPU
  iterates per-pixel f32 deltas with Zhuoran-style rebasing. Hard floor
  `MIN_UNITS_PER_POINT` ≈ 1e-32 (f32 delta underflow).
- The orbit recomputes only when the view drifts > half a screen from the
  reference or needs more iterations — keep it that way; per-frame recompute
  janks panning.
- Correctness oracle: `perturbation_matches_plain_path` renders the same view
  through both paths; ~2.6% boundary-pixel jitter is normal, broad divergence
  means a real bug.

## Gotchas

- **eframe 0.35 diverges from older egui examples** (and from training data):
  `App::ui(&mut self, ui, frame)` not `App::update`, `egui::Panel::left` not
  `SidePanel`, wgpu 29 renames (`multiview_mask`, `immediate_size`,
  `bind_group_layouts: &[Some(..)]`, `PollType::wait_indefinitely()`). When
  unsure, read the vendored sources in `~/.cargo/registry/src/`.
- **naga return analysis**: a WGSL function whose body ends in an
  always-returning `loop` still needs a trailing unreachable `return`.
- **dashu base conversion is ~1 ulp inexact**: bookmark decimal round trips are
  intentionally not bit-exact; the 64 guard bits absorb it. Don't tighten the
  round-trip test to equality.
- **Bookmark compatibility contract**: v1 bookmarks (`center: [f64;2]`) and the
  legacy `selfsame-bookmark` PNG keyword must keep loading (`center_serde`,
  `LEGACY_BOOKMARK_KEYWORD`).
- Screenshots of the running app are unavailable to agents (no screen-recording
  permission); verify rendering via the headless GPU tests instead.

## Workflow

- The user commits; don't run `git commit` unless explicitly asked.
