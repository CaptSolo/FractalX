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
- **Progressive rendering, first layer** (§4.1) — Mandelbrot renders through a resolution ladder (view changes restart at 1/4 resolution so interaction never stalls; each following frame climbs one rung to full resolution); the IFS chaos game accumulates a fixed point batch per frame, so the image "develops" and huge point counts never block the UI. Both render into textures; a view change simply abandons stale work (generation-by-key model).

Not yet started: iteration chunking for escape-time (§4.1's second layer — a full-res pass at very high `max_iter` is still one long GPU dispatch; needs per-pixel state in storage buffers + compute shader); background-thread reference-orbit computation (recompute still hitches one frame at extreme depth); visual (drag-handle) editing of IFS maps — the §3.2 signature feature; deterministic shape-iteration IFS view; L-systems; Julia companion pane; bookmarks journal; palette editor; other escape-time formulas; natural/statistical module; measurement tools.
