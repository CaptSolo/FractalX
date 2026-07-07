//! Iterated Function Systems: affine maps rendered via the chaos game.
//!
//! CPU pipeline (no GPU involvement): run the chaos game into a density
//! histogram at the target pixel size, then tone-map the histogram through
//! the same cyclic palette the escape-time shader uses. The histogram is
//! cached separately from the tone-mapped image so palette tweaks don't
//! re-run the chaos game.

/// One affine map: (x, y) -> (a x + b y + e, c x + d y + f).
#[derive(Clone, Copy, PartialEq, Debug, serde::Serialize, serde::Deserialize)]
pub struct AffineMap {
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub d: f32,
    pub e: f32,
    pub f: f32,
    /// Relative probability of picking this map in the chaos game.
    pub weight: f32,
}

/// A named preset: maps plus a sensible initial viewport
/// (center, complex-units-per-point for a ~800pt canvas).
pub struct Preset {
    pub name: &'static str,
    pub maps: &'static [AffineMap],
    pub center: [f64; 2],
    pub units_per_point: f64,
}

const fn map(a: f32, b: f32, c: f32, d: f32, e: f32, f: f32, weight: f32) -> AffineMap {
    AffineMap {
        a,
        b,
        c,
        d,
        e,
        f,
        weight,
    }
}

pub const PRESETS: &[Preset] = &[
    Preset {
        name: "Sierpinski triangle",
        maps: &[
            map(0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0),
            map(0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 1.0),
            map(0.5, 0.0, 0.0, 0.5, 0.25, 0.433, 1.0),
        ],
        center: [0.5, 0.43],
        units_per_point: 0.0016,
    },
    Preset {
        name: "Barnsley fern",
        maps: &[
            map(0.0, 0.0, 0.0, 0.16, 0.0, 0.0, 0.01),
            map(0.85, 0.04, -0.04, 0.85, 0.0, 1.6, 0.85),
            map(0.2, -0.26, 0.23, 0.22, 0.0, 1.6, 0.07),
            map(-0.15, 0.28, 0.26, 0.24, 0.0, 0.44, 0.07),
        ],
        center: [0.25, 5.0],
        units_per_point: 0.0145,
    },
    Preset {
        name: "Heighway dragon",
        maps: &[
            map(0.5, -0.5, 0.5, 0.5, 0.0, 0.0, 1.0),
            map(-0.5, -0.5, 0.5, -0.5, 1.0, 0.0, 1.0),
        ],
        center: [0.5, 0.25],
        units_per_point: 0.0025,
    },
];

/// Viewport into IFS space (world units per pixel, world center).
#[derive(Clone, Copy, PartialEq)]
pub struct IfsView {
    pub center: [f64; 2],
    pub units_per_pixel: f64,
}

/// Number of independent chaos-game walkers. Fixed (not tied to the thread
/// count) so results are identical on any machine; rayon spreads the walkers
/// over available cores.
const LANES: usize = 16;

/// A walker's cumulative share of `total` plotted points. A pure function of
/// the cumulative total, so any batch split advances each walker through the
/// same sequence — histograms are batch-split invariant.
fn lane_share(total: u64, lane: usize) -> u64 {
    (total + (LANES - 1 - lane) as u64) / LANES as u64
}

/// View `&mut [u32]` as atomics for lock-free scatter increments.
/// Sound: `AtomicU32` has the same layout as `u32`, and the exclusive borrow
/// guarantees nothing else touches the data during the parallel phase.
fn as_atomic(hist: &mut [u32]) -> &[std::sync::atomic::AtomicU32] {
    unsafe { &*(hist as *mut [u32] as *const [std::sync::atomic::AtomicU32]) }
}

/// One independent chaos-game walker (its own RNG stream and position).
struct Walker {
    rng: fastrand::Rng,
    x: f32,
    y: f32,
    /// Iterations still to skip before plotting (attractor settle-in).
    warmup: u32,
}

impl Walker {
    fn new(lane: usize) -> Self {
        Self {
            rng: fastrand::Rng::with_seed(0x5eed_f2ac_7a15_0001 + lane as u64),
            x: 0.0,
            y: 0.0,
            warmup: 32,
        }
    }

    fn advance(
        &mut self,
        maps: &[AffineMap],
        total_weight: f32,
        view: IfsView,
        width: usize,
        height: usize,
        hist: &[std::sync::atomic::AtomicU32],
        points: u64,
    ) {
        use std::sync::atomic::Ordering::Relaxed;

        // World -> pixel transform (y up in world, down in pixels).
        let upp = view.units_per_pixel;
        let ox = view.center[0] - upp * width as f64 * 0.5;
        let oy = view.center[1] + upp * height as f64 * 0.5;

        let mut plotted = 0u64;
        while plotted < points {
            let mut pick = self.rng.f32() * total_weight;
            let mut chosen = &maps[0];
            for m in maps {
                let w = m.weight.max(0.0);
                if pick < w {
                    chosen = m;
                    break;
                }
                pick -= w;
            }
            let nx = chosen.a * self.x + chosen.b * self.y + chosen.e;
            let ny = chosen.c * self.x + chosen.d * self.y + chosen.f;
            self.x = nx;
            self.y = ny;
            if !self.x.is_finite() || !self.y.is_finite() {
                // Diverging system (user-editable maps); restart the point.
                (self.x, self.y) = (0.0, 0.0);
                plotted += 1; // count it so degenerate systems terminate
                continue;
            }
            if self.warmup > 0 {
                self.warmup -= 1;
                continue;
            }
            plotted += 1;

            let px = (self.x as f64 - ox) / upp;
            let py = (oy - self.y as f64) / upp;
            if px >= 0.0 && py >= 0.0 && (px as usize) < width && (py as usize) < height {
                hist[py as usize * width + px as usize].fetch_add(1, Relaxed);
            }
        }
    }
}

/// Resumable, parallel chaos game: a fixed set of independent walkers filling
/// one histogram (rayon over walkers, atomic scatter adds). Deterministic:
/// fixed seeds, a fixed lane count, and cumulative work-splitting make any
/// batch split — on any number of threads — produce the identical histogram.
pub struct ChaosGame {
    walkers: Vec<Walker>,
    /// Cumulative points requested so far.
    total: u64,
}

impl ChaosGame {
    pub fn new() -> Self {
        Self {
            walkers: (0..LANES).map(Walker::new).collect(),
            total: 0,
        }
    }

    /// Advance by `points` plotted points, accumulating into `hist`
    /// (`width*height`, same view for every batch).
    pub fn advance(
        &mut self,
        maps: &[AffineMap],
        view: IfsView,
        width: usize,
        height: usize,
        hist: &mut [u32],
        points: u64,
    ) {
        use rayon::prelude::*;

        debug_assert_eq!(hist.len(), width * height);
        let total_weight: f32 = maps.iter().map(|m| m.weight.max(0.0)).sum();
        if maps.is_empty() || total_weight <= 0.0 {
            return;
        }

        let old_total = self.total;
        self.total += points;
        let new_total = self.total;
        let atomic_hist = as_atomic(hist);

        self.walkers
            .par_iter_mut()
            .enumerate()
            .for_each(|(lane, walker)| {
                let n = lane_share(new_total, lane) - lane_share(old_total, lane);
                walker.advance(maps, total_weight, view, width, height, atomic_hist, n);
            });
    }
}

impl Default for ChaosGame {
    fn default() -> Self {
        Self::new()
    }
}

/// One-shot chaos game (used by export and tests).
pub fn chaos_histogram(
    maps: &[AffineMap],
    points: u64,
    view: IfsView,
    width: usize,
    height: usize,
) -> Vec<u32> {
    let mut hist = vec![0u32; width * height];
    ChaosGame::new().advance(maps, view, width, height, &mut hist, points);
    hist
}

/// Approximate bounding box of the attractor (min x, min y, max x, max y),
/// for fitting the viewport. None if the system is empty or diverges.
pub fn attractor_bbox(maps: &[AffineMap], samples: u64) -> Option<[f64; 4]> {
    let total_weight: f32 = maps.iter().map(|m| m.weight.max(0.0)).sum();
    if maps.is_empty() || total_weight <= 0.0 {
        return None;
    }
    let mut rng = fastrand::Rng::with_seed(0x5eed_b0b0_0000_0002);
    let (mut x, mut y) = (0.0f32, 0.0f32);
    let (mut min_x, mut min_y, mut max_x, mut max_y) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
    let mut plotted = 0u64;
    for i in 0..samples + 32 {
        let mut pick = rng.f32() * total_weight;
        let mut chosen = &maps[0];
        for m in maps {
            let w = m.weight.max(0.0);
            if pick < w {
                chosen = m;
                break;
            }
            pick -= w;
        }
        let nx = chosen.a * x + chosen.b * y + chosen.e;
        let ny = chosen.c * x + chosen.d * y + chosen.f;
        x = nx;
        y = ny;
        if !x.is_finite() || !y.is_finite() {
            (x, y) = (0.0, 0.0);
            continue;
        }
        if i < 32 {
            continue;
        }
        min_x = min_x.min(x as f64);
        min_y = min_y.min(y as f64);
        max_x = max_x.max(x as f64);
        max_y = max_y.max(y as f64);
        plotted += 1;
    }
    (plotted > 0).then_some([min_x, min_y, max_x, max_y])
}

/// Tone-map a histogram to RGBA: log-scaled density through the palette,
/// black background.
pub fn tonemap_rgba(
    hist: &[u32],
    palette: crate::palette::Palette,
    palette_freq: f32,
    palette_phase: f32,
) -> Vec<u8> {
    let max = hist.iter().copied().max().unwrap_or(0).max(1) as f32;
    let inv_log_max = 1.0 / (1.0 + max).ln();
    let mut rgba = Vec::with_capacity(hist.len() * 4);
    for &n in hist {
        if n == 0 {
            rgba.extend_from_slice(&[0, 0, 0, 255]);
        } else {
            let t = (1.0 + n as f32).ln() * inv_log_max;
            let [r, g, b] = palette.eval(t, palette_freq, palette_phase);
            rgba.extend_from_slice(&[r, g, b, 255]);
        }
    }
    rgba
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sierpinski_view() -> IfsView {
        IfsView {
            center: [0.5, 0.43],
            units_per_pixel: 1.2 / 128.0,
        }
    }

    #[test]
    fn sierpinski_fills_triangle_not_holes() {
        let maps = PRESETS[0].maps;
        let hist = chaos_histogram(maps, 200_000, sierpinski_view(), 128, 128);

        let density_at = |wx: f64, wy: f64| {
            let v = sierpinski_view();
            let px = ((wx - (v.center[0] - v.units_per_pixel * 64.0)) / v.units_per_pixel) as usize;
            let py = (((v.center[1] + v.units_per_pixel * 64.0) - wy) / v.units_per_pixel) as usize;
            // 3x3 neighborhood sum
            (px.saturating_sub(1)..=px + 1)
                .flat_map(|x| (py.saturating_sub(1)..=py + 1).map(move |y| (x, y)))
                .map(|(x, y)| hist.get(y * 128 + x).copied().unwrap_or(0))
                .sum::<u32>()
        };

        // Corners of the triangle are on the attractor.
        assert!(density_at(0.0, 0.0) > 0, "bottom-left corner empty");
        assert!(density_at(1.0, 0.0) > 0, "bottom-right corner empty");
        assert!(density_at(0.5, 0.866) > 0, "top corner empty");
        // The center of the central hole is not.
        assert_eq!(density_at(0.5, 0.29), 0, "central hole has points");
    }

    #[test]
    fn batched_advance_equals_one_shot() {
        let maps = PRESETS[0].maps;
        let view = sierpinski_view();
        let one_shot = chaos_histogram(maps, 100_000, view, 64, 64);

        let mut game = ChaosGame::new();
        let mut hist = vec![0u32; 64 * 64];
        for batch in [30_000u64, 50_000, 20_000] {
            game.advance(maps, view, 64, 64, &mut hist, batch);
        }
        assert_eq!(hist, one_shot, "batch split changed the result");
    }

    // Dev benchmark (not run by default): parallel vs single-thread speedup.
    // Run with: cargo test --release bench_chaos_speedup -- --ignored --nocapture
    // Speedup is scatter-bound: ~3x when the attractor fills the frame,
    // ~2x when concentrated (atomic contention on hot pixels).
    #[test]
    #[ignore]
    fn bench_chaos_speedup() {
        let view = IfsView { center: [0.5, 0.43], units_per_pixel: 1.2 / 2160.0 };
        let t = std::time::Instant::now();
        let _ = chaos_histogram(PRESETS[0].maps, 20_000_000, view, 3840, 2160);
        let par = t.elapsed();
        let single = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap().install(|| {
            let t = std::time::Instant::now();
            let _ = chaos_histogram(PRESETS[0].maps, 20_000_000, view, 3840, 2160);
            t.elapsed()
        });
        println!("parallel: {par:?}  single-thread: {single:?}  speedup: {:.1}x",
            single.as_secs_f64() / par.as_secs_f64());
    }

    #[test]
    fn result_independent_of_thread_count() {
        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| chaos_histogram(PRESETS[1].maps, 100_000, sierpinski_view(), 64, 64))
        };
        assert_eq!(run(1), run(8), "histogram depends on thread count");
    }

    #[test]
    fn chaos_game_is_deterministic() {
        let maps = PRESETS[1].maps; // fern
        let view = IfsView {
            center: [0.25, 5.0],
            units_per_pixel: 11.0 / 64.0,
        };
        let a = chaos_histogram(maps, 50_000, view, 64, 64);
        let b = chaos_histogram(maps, 50_000, view, 64, 64);
        assert_eq!(a, b);
        assert!(a.iter().any(|&n| n > 0));
    }

    #[test]
    fn degenerate_systems_yield_empty_histograms() {
        // No maps, zero weights, and a diverging map must not hang or panic.
        let view = sierpinski_view();
        assert!(chaos_histogram(&[], 1000, view, 16, 16).iter().all(|&n| n == 0));
        let zero_w = [map(0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0)];
        assert!(chaos_histogram(&zero_w, 1000, view, 16, 16).iter().all(|&n| n == 0));
        let diverging = [map(10.0, 0.0, 0.0, 10.0, 1.0, 1.0, 1.0)];
        let _ = chaos_histogram(&diverging, 1000, view, 16, 16); // must terminate
    }

    #[test]
    fn tonemap_maps_zero_to_black_and_scales() {
        let rgba = tonemap_rgba(&[0, 1, 100], crate::palette::Palette::Classic, 1.0, 0.0);
        assert_eq!(&rgba[0..4], &[0, 0, 0, 255]);
        assert_ne!(&rgba[4..7], &[0, 0, 0]);
        assert_eq!(rgba.len(), 12);
    }
}

