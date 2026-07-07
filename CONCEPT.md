# Fractal Explorer — Concept Specification

**Status:** In development — prototype (see §10)
**Date:** 2026-07-06
**Name:** *FractalX*

---

## 1. Vision

A desktop application for exploring fractals and self-similarity — not just *rendering* fractals, but making the **idea of self-similarity tangible**: how simple rules, iterated, produce infinite structure, and how the same patterns recur across scales and across very different generating systems.

The app should serve two moods that often live in the same person:

- **The artist/explorer** — wander, zoom, discover striking regions, tune colors, export beautiful images.
- **The mathematician** — control parameters precisely, compare systems, measure (dimension, orbits, convergence), and understand *why* the structure looks the way it does.

The unifying theme is **iteration made visible**. Every fractal family in scope is "a rule applied repeatedly"; the app should always let the user see the rule, step the iteration, and connect the rule to the picture.

## 2. Audience

| Audience | What they need |
|---|---|
| Casual explorers & artists | Instant gratification: smooth zooming, presets/gallery, great default palettes, easy high-res export. |
| Math enthusiasts / researchers | Precise numeric parameter entry, custom formulas, deep zoom beyond double precision, orbit/dimension analysis, reproducible saved states. |
| The author (hobby project) | A codebase that is fun to grow: each fractal family is a plugin-like module; new ideas are cheap to add. |

Non-audience (for now): classroom deployment, mobile, collaborative/online features.

## 3. Scope

Three families, chosen to show self-similarity from three different angles:

### 3.1 Escape-time fractals (implicit self-similarity)
- Mandelbrot set, Julia sets, Burning Ship, Multibrot, and **user-defined complex formulas** (small expression language, compiled to GPU shader).
- Smooth/continuous coloring, interior coloring, orbit traps.
- **Deep zoom**: perturbation theory — arbitrary-precision reference orbit on the CPU, per-pixel f32 delta iteration with rebasing on the GPU — for zooms to ~10^30. (Metal has no shader f64, so f32+perturbation carries all depth beyond the plain path's ~10^4.)
- Live **Mandelbrot ↔ Julia duality**: hover a point in the Mandelbrot set, see its Julia set update in a companion pane.

### 3.2 IFS & L-systems (explicit self-similarity)
- Iterated Function Systems: affine maps edited visually (drag/rotate/scale the child copies) — Sierpinski triangle, Barnsley fern, Koch-like sets, user-created systems. Render via chaos game (points) and via deterministic set iteration (shapes), switchable.
- L-systems: rule editor (axiom + productions + turtle interpretation), stepwise expansion — the user can watch generation *n* become generation *n+1*.
- This family is the pedagogical heart: the self-similar structure is literally the rule, and the editor makes that visible.

### 3.3 Natural / statistical self-similarity
- Fractal terrain and clouds (midpoint displacement, fractional Brownian motion / noise octaves) with a "roughness ↔ dimension" control.
- Random walks and Brownian paths.
- **Box-counting dimension tool**: estimate fractal dimension of anything rendered in the app *or an imported image* (e.g., a coastline photo), with the log–log fit shown.

Out of scope initially: 3D escape-time fractals (Mandelbulb), strange attractors, flame fractals, animation/video rendering. Listed as candidate extensions (§8).

## 4. Core concepts & UX

### 4.1 The Canvas
One large, always-interactive viewport. Pan/zoom with mouse and trackpad gestures; zoom is smooth and progressive (coarse preview refines in place — never a frozen screen). Rendering happens on the GPU wherever possible; long computations refine progressively and are cancellable by any interaction.

### 4.2 The Rule Panel
Whatever is on canvas, its **generating rule is always one panel away**: the formula for escape-time sets, the affine maps for IFS, the productions for L-systems, the noise parameters for terrain. Editing the rule updates the canvas live. Parameters accept both slider-style manipulation and exact numeric/text entry.

### 4.3 Iteration control
A universal "iteration depth" affordance, meaningful per family: max iterations for escape-time, point count / recursion depth for IFS, generation count for L-systems, octaves for noise. A **step mode** animates iterations one at a time — the single most important feature for building intuition.

### 4.4 Bookmarks & the Journal
Any view can be saved as a **bookmark**: complete state (family, rule, parameters, viewport, palette) in a small human-readable file. Bookmarks form a browsable journal/gallery with thumbnails. Every exported image embeds its bookmark, so any picture can be reopened *exactly* — reproducibility for the researcher, "how did I get here?" for the explorer. The app ships with a starter gallery of curated locations.

### 4.5 Color & export
Palette editor (gradient stops, cyclic palettes, palette import), applied post-hoc without recomputation where possible. Export: PNG at arbitrary resolution (tiled rendering for poster-size output), with embedded state.

### 4.6 Measurement & insight tools
- Orbit visualizer: click a point in an escape-time fractal, see its orbit traced.
- Box-counting dimension (any render or imported image).
- Zoom-scale readout ("you are at 10¹²× — the original view would span the solar system").
- Side-by-side compare of two bookmarks.

## 5. Architecture sketch

**Chosen stack: Rust + wgpu + egui (eframe), macOS-first.** Rationale: native performance for CPU-side arbitrary-precision math, first-class GPU compute for rendering, single-binary distribution on macOS (primary platform), and a language well-suited to a long-lived hobby codebase. Arbitrary precision is `dashu` (pure Rust — no C toolchain dependency; its decimal↔binary conversion is ~1 ulp, absorbed by 64 guard bits of precision headroom). Alternatives considered: C++/OpenGL (more deep-zoom reference material, worse ergonomics), rug/MPFR (faster bignums, GMP build dependency).

```
┌────────────────────────────────────────────┐
│ UI shell (panels, journal, palette editor) │
├────────────────────────────────────────────┤
│ Core: state model, bookmarks, export,      │
│ progressive-render scheduler, undo         │
├──────────┬──────────────┬──────────────────┤
│ Escape-  │ IFS /        │ Natural /        │
│ time     │ L-systems    │ statistical      │
│ module   │ module       │ module           │
├──────────┴──────────────┴──────────────────┤
│ Render backends: GPU (wgpu compute/frag),  │
│ CPU (rayon), arbitrary-precision (dashu)   │
└────────────────────────────────────────────┘
```

Key contracts:
- **Fractal module interface**: each family implements `parameters() / render(viewport, quality) / serialize()`. Adding a family never touches core code.
- **Progressive rendering**: every renderer must produce a usable image quickly and refine; the scheduler owns cancellation.
- **State is data**: a bookmark fully determines a render. No hidden state.

## 6. Principles

1. **Never block interaction.** A coarse image now beats a perfect image in two seconds.
2. **The rule is always visible.** No fractal appears without a path to its generator.
3. **Everything is reproducible.** Any image reopens to the exact state that made it.
4. **Depth is optional.** The first five minutes require zero math; the measurement tools are there when curiosity strikes.
5. **Small core, pluggable families.** The hobby thrives on cheap experiments.

## 7. Success criteria (concept-level)

- A newcomer finds a personally striking Mandelbrot region and exports a wallpaper within 10 minutes, unaided.
- The Barnsley fern can be *built from scratch* in the IFS editor by dragging four maps, and the user can see why it's a fern.
- A zoom to 10³⁰ renders correctly and can be shared as a bookmark file that reopens identically.
- Box-counting on an imported coastline image yields a plausible dimension with a visible log–log fit.

## 8. Candidate extensions (explicitly deferred)

3D fractals (Mandelbulb/Mandelbox via ray-marching) · strange attractors & bifurcation diagrams · flame fractals · zoom-animation/video export · scripting API (Lua/Python) · palette sharing community formats.

## 9. Open questions

- How far to take the custom-formula language (full expression parser → shader codegen is a project in itself)?
- Should the journal be per-project files or a single library database?

Resolved: macOS-only first (cross-platform later via wgpu/egui). UI toolkit: egui via eframe.

## 10. Implementation status

Built so far:

- **Mandelbrot explorer** — GPU-rendered (WGSL fragment shader), smooth pan/zoom anchored at the pointer, iteration slider to 100k, cyclic palette controls (§4.1, partially §4.5).
- **PNG export with embedded bookmark** (§4.4/§4.5) — offscreen GPU render at arbitrary resolution; the complete view state travels in an iTXt chunk (versioned JSON, keyword `fractalx-bookmark`) and any exported PNG reopens to its exact view.
- **Deep zoom to ~10^30** (§3.1) — perturbation with arbitrary-precision reference orbit (dashu), f32 delta iteration with rebasing on the GPU; orbit recomputed only when the view leaves its neighborhood. Verified by GPU tests (perturbation path must match the plain path where both are valid; a 10^-14 boundary view must show structure).

- **IFS module, first cut** (§3.2) — family selector in the UI; chaos-game rendering on the CPU (density histogram → log tone-map through the shared palette, cached so palette tweaks don't re-run the game); presets (Sierpinski, Barnsley fern, Heighway dragon); numeric affine-map editor with add/remove; viewport fitting to the attractor's bounding box. Bookmarks are v3 (family-tagged rule; v1/v2 still load as Mandelbrot). IFS PNG export renders on the CPU at full resolution with the point budget scaled to pixel count.
- **Progressive rendering, first layer** (§4.1) — Mandelbrot renders through a resolution ladder (view changes restart at 1/4 resolution so interaction never stalls; each following frame climbs one rung to full resolution); the IFS chaos game accumulates a fixed point batch per frame, so the image "develops" and huge point counts never block the UI. Both render into textures; a view change simply abandons stale work (generation-by-key model). While actively panning (same zoom), the existing sharp texture is reprojected (drawn shifted) instead of re-rendered; the ladder restarts on release.
- **Palette decoupled from iteration** — the Mandelbrot render is two passes: iteration counts land in an R32Float data texture, a second cheap pass maps them through the palette. Palette sliders re-colorize instantly without re-iterating (no ladder restart); groundwork for a palette editor.
- **Parallel chaos game** — 16 fixed-seed walkers spread over cores via rayon, scatter-adding into a shared histogram with atomics. Deterministic by construction (fixed lane count, cumulative work splitting): identical output for any batch split on any thread count, verified by tests. ~2–3× faster (scatter/atomic-contention bound).
- **More escape-time formulas** — Tricorn (Mandelbar, z̄²+c) and Multibrot (power 2–8) as variants of one shader iteration core. Deep zoom (perturbation) remains Mandelbrot-only — the z² algebra doesn't carry over — with a UI warning past f32 precision. Bookmark family tags: `tricorn`, `multibrot`.
- **Palette presets** — named cosine-gradient palettes (Classic, Sunset, Fire, Electric, Pastel, Grayscale) selectable in the UI, shared by the escape-time color pass and the IFS tone-map; the frequency/phase sliders modulate any preset. One coefficient formula (`a + b·cos(2π(c·x + d))`) drives both the shader and the CPU mirror; `Classic` reproduces the original palette exactly (tested). The preset travels in bookmarks (`#[serde(default)]` — older bookmarks load as Classic); switching presets only re-runs the cheap color pass / tone-map.
- **L-systems** (§3.2) — third fractal family: axiom + rewrite rules expanded
  (capped at 16M symbols so runaway growth can't hang the UI) and interpreted
  as turtle graphics (`F G f g + - [ ]`); segments rasterized on the CPU
  (Liang–Barsky clip + DDA), colored by arc position through the shared
  palette. Nine presets: Koch snowflake/island, dragon, Hilbert, Gosper, and
  Lévy C curves, Sierpinski arrowhead, plant, and bush.
  Full rule editing in the UI (axiom, per-symbol rules, angle,
  generations); any rule edit (and *Reset view*) refits the viewport to the
  drawing's bounds, and the zoom-out limit scales with the drawing. A notice
  shows when the symbol cap cuts expansion short of the requested
  generations. Segments cache until the rule changes; pan/zoom/palette only
  re-rasterize. Bookmark family tag: `l_system`.
- **Julia sets** (§3.1) — fourth escape-time formula: the pixel seeds `z`
  and `c` is a fixed constant (chosen numerically or via eleven classic
  presets — Basilica, San Marco, Airplane, Douady Rabbit, the c=i dendrite,
  and more). Shares the full two-pass/chunked/progressive
  pipeline; f32-precision warning past ~3e4 zoom like the other non-Mandelbrot
  formulas. Bookmark family tag: `julia`.
- **Strange attractors** — Clifford and de Jong maps rendered as density
  plots through the IFS histogram/tone-map pipeline, but fully deterministic
  (no RNG: 16 fixed-seed orbits, cumulative work splitting — batch-split and
  thread-count invariant, tested). Four presets, numeric a/b/c/d editing
  (edits refit the viewport), progressive accumulation, CPU export with the
  point budget scaled to pixel count. Bookmark family tag: `attractor`.
- **Hover-linked Julia preview** (§3.1 duality) — while exploring the
  Mandelbrot set, a corner overlay renders the Julia set for the c under the
  cursor in real time (small fixed-iteration GPU render, re-run only when c
  or the palette changes; keeps its last c when the cursor leaves the
  canvas). Sidebar checkbox to show/hide; `J` pins/unpins the preview's c
  (accented border + label while pinned); clicking the pane (or its "Julia
  view" button) promotes the displayed c to the full Julia family view.
- **Iteration chunking** (§4.1 second layer) — above 2048 iterations, a ladder rung renders via a compute shader that advances every pixel by 2048 iterations per frame, persisting per-pixel state in a storage buffer (`ceil(max_iter/chunk)` dispatches guarantee completion — no readback). No single dispatch can stall the GPU at 100k iterations. Chunk resumption is bit-exact vs. a single dispatch (tested); fragment and compute variants differ by driver-level float jitter on ~1% boundary pixels (tolerated in tests).

Not yet started: background-thread reference-orbit computation (recompute still hitches one frame at extreme depth); visual (drag-handle) editing of IFS maps — the §3.2 signature feature; deterministic shape-iteration IFS view; bookmarks journal; palette editor; custom formula expressions; natural/statistical module; measurement tools.
