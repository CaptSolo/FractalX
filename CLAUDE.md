# FractalX

Desktop fractal explorer (macOS-first): Rust + wgpu + egui (eframe). Vision,
scope, and implementation status live in `CONCEPT.md` — §10 tracks what is
built vs. pending; keep it updated when a milestone lands.

## Commands

- Build: `cargo build` · Run: `cargo run --release` (debug builds render slowly)
- Test: `cargo test` — includes headless GPU tests (need a Metal adapter; no
  GPU-less CI). `cargo test --release bench_chaos_speedup -- --ignored
  --nocapture` runs the chaos-game benchmark.
- The binary is `fractalx`; check liveness with `pgrep -x fractalx`

## Architecture

- `src/main.rs` — app shell, `ViewState`/`FractalRule` (the "bookmark": fully
  determines a render), interaction, progressive-render scheduling (resolution
  ladder, chunk jobs, pan reprojection, IFS batching), orbit cache, export/load
  UI, coordinate entry.
- `src/mandelbrot.rs` — escape-time wgpu pipelines. Two-pass: data pass writes
  smooth iteration counts to an R32Float texture (fragment for low `max_iter`,
  chunked compute above `CHUNK_ITERS`), color pass maps data through the
  palette (palette changes re-run only this). Struct layouts must match
  `src/shaders/mandelbrot.wgsl` byte-for-byte.
- `src/deep.rs` — arbitrary-precision (dashu) center coordinates and the
  perturbation reference orbit. Precision auto-scales: zoom bits + 64 guard bits.
- `src/ifs.rs` — IFS chaos game: 16 fixed-seed walkers over rayon, atomic
  scatter-adds into a shared histogram; log tone-map through the shared palette.
- `src/lsystem.rs` — L-systems: rewrite expansion (capped at `MAX_SYMBOLS` —
  returns the last complete generation, never a truncated string), turtle
  interpretation (`F G f g + - [ ]`), CPU rasterization (Liang–Barsky clip +
  DDA) colored by arc position through the shared palette. Fully
  deterministic (no RNG). Bookmark family tag: `l_system`.
- `src/palette.rs` — named cosine-gradient presets (`a + b·cos(2π(c·x + d))`);
  one coefficient table drives both the WGSL color pass and the CPU IFS
  tone-map. `Classic` must stay bit-identical to the original palette
  (tested) — old bookmarks default to it via `#[serde(default)]`.
- `src/export.rs` — PNG with the bookmark embedded as an iTXt chunk.
- Formulas: Mandelbrot / Tricorn / Multibrot share one shader iteration core
  (`step_plain` switch); adding a formula touches the shader switch, the
  `FractalRule` enum, and `uniforms_for_size` only.
- Spec rule: fractal families are pluggable modules; adding a family must not
  touch core code (`main.rs` state/scheduling, export).

## Rendering invariants (load-bearing — don't break casually)

- **Chunked completion needs no readback**: `ceil(max_iter / CHUNK_ITERS)`
  dispatches provably resolve every pixel; `paint_mandelbrot` counts
  dispatches down instead of asking the GPU.
- **Chaos-game determinism** rests on: fixed `LANES = 16` (independent of
  thread count), fixed per-lane seeds, and `lane_share` being a pure function
  of the cumulative total (batch-split invariance). Changing any of these
  changes rendered output for identical bookmarks.
- **Perturbation is Mandelbrot-only** by design (the z² delta algebra doesn't
  carry to Tricorn/Multibrot); other formulas warn in the UI past ~3e4 zoom.
- The reference orbit recomputes only when the view drifts > half a screen
  from the reference or needs more iterations — per-frame recompute janks
  panning.

## Deep zoom (perturbation)

- Plain f32 shader path below ~3e4 zoom; perturbation path beyond
  (`PERTURB_THRESHOLD`): CPU computes one high-precision reference orbit, GPU
  iterates per-pixel f32 deltas with Zhuoran-style rebasing. Hard floor
  `MIN_UNITS_PER_POINT` ≈ 1e-32 (f32 delta underflow).

## Gotchas

- **eframe 0.35 diverges from older egui examples** (and from training data):
  `App::ui(&mut self, ui, frame)` not `App::update`, `egui::Panel::left` not
  `SidePanel`, wgpu 29 renames (`multiview_mask`, `immediate_size`,
  `bind_group_layouts: &[Some(..)]`, `PollType::wait_indefinitely()`). When
  unsure, read the vendored sources in `~/.cargo/registry/src/`.
- **Cross-pipeline float jitter is normal**: Metal schedules float ops
  differently in fragment vs compute variants of the same WGSL, so ~1% of
  boundary pixels escape one iteration apart. Tests tolerate it
  (`perturbation_matches_plain_path`, `chunked_compute_matches_fragment_pass`)
  — don't tighten them to exact equality across pipelines. Chunk *resumption*
  (multi-dispatch vs single dispatch, same pipeline) IS bit-exact and tested
  as such.
- **Panels grow to content unless pinned**: `egui::Panel` `default_size` +
  `resizable(false)` still lets content widen the panel — a focused
  `desired_width(f32::INFINITY)` TextEdit expands it over the canvas. The
  controls panel uses `exact_size` for this reason; keep it.
- **naga return analysis**: a WGSL function whose body ends in an
  always-returning `loop` still needs a trailing unreachable `return`.
- **dashu base conversion is ~1 ulp inexact**: bookmark decimal round trips are
  intentionally not bit-exact; the 64 guard bits absorb it. Don't tighten the
  round-trip test to equality.
- **Bookmark compatibility contract**: v1 (`center: [f64;2]`) and v2 (no
  `rule` field → Mandelbrot) bookmarks and the legacy `selfsame-bookmark` PNG
  keyword must keep loading (`center_serde`, `#[serde(default)] rule`,
  `LEGACY_BOOKMARK_KEYWORD`). Removed family tags (e.g. `burning_ship`) fail
  to load with a clear error — acceptable.
- Screenshots of the running app are unavailable to agents (no screen-recording
  permission); verify rendering via the headless GPU tests instead.

## Workflow

- The user commits; don't run `git commit` unless explicitly asked.
- After launching the app for the user, don't poll to verify it's running.
