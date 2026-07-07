//! Strange attractors: iterated nonlinear 2D maps rendered as density plots.
//!
//! Same CPU pipeline as the IFS chaos game (and it reuses `ifs::tonemap_rgba`
//! for coloring): iterate the map, scatter-add points into a histogram, cache
//! the histogram so palette tweaks don't re-run the orbit. Unlike the chaos
//! game there is no randomness at all — each of the fixed lanes runs one
//! deterministic orbit from its own fixed seed point (the attractor swallows
//! every start after a short warmup), so output is identical on any machine
//! and for any batch split.

use crate::ifs::IfsView;

/// Which map is iterated.
#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    /// x' = sin(a·y) + c·cos(a·x),  y' = sin(b·x) + d·cos(b·y)
    Clifford,
    /// x' = sin(a·y) − cos(b·x),    y' = sin(c·x) − cos(d·y)
    DeJong,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Clifford => "Clifford",
            Kind::DeJong => "de Jong",
        }
    }

    fn step(self, p: [f64; 4], x: f64, y: f64) -> (f64, f64) {
        let [a, b, c, d] = p;
        match self {
            Kind::Clifford => ((a * y).sin() + c * (a * x).cos(), (b * x).sin() + d * (b * y).cos()),
            Kind::DeJong => ((a * y).sin() - (b * x).cos(), (c * x).sin() - (d * y).cos()),
        }
    }
}

/// Fixed lane count (matches the chaos game's reasoning: independent of the
/// thread count so results are machine-independent).
const LANES: usize = 16;
/// Iterations discarded per lane before plotting (settle onto the attractor).
const WARMUP: u32 = 100;

/// A lane's cumulative share of `total` plotted points; pure function of the
/// total, so histograms are batch-split invariant.
fn lane_share(total: u64, lane: usize) -> u64 {
    (total + (LANES - 1 - lane) as u64) / LANES as u64
}

/// See `ifs::as_atomic` — same soundness argument.
fn as_atomic(hist: &mut [u32]) -> &[std::sync::atomic::AtomicU32] {
    unsafe { &*(hist as *mut [u32] as *const [std::sync::atomic::AtomicU32]) }
}

/// One deterministic orbit (fixed starting point per lane).
struct Lane {
    x: f64,
    y: f64,
    warmup: u32,
}

impl Lane {
    fn new(lane: usize) -> Self {
        // Distinct fixed seeds on a small circle around the origin; the
        // attractor absorbs them all, but distinct orbits decorrelate lanes.
        let t = lane as f64 / LANES as f64 * std::f64::consts::TAU;
        Self {
            x: 0.1 * t.cos(),
            y: 0.1 * t.sin(),
            warmup: WARMUP,
        }
    }

    fn advance(
        &mut self,
        kind: Kind,
        params: [f64; 4],
        view: IfsView,
        width: usize,
        height: usize,
        hist: &[std::sync::atomic::AtomicU32],
        points: u64,
    ) {
        use std::sync::atomic::Ordering::Relaxed;

        let upp = view.units_per_pixel;
        let ox = view.center[0] - upp * width as f64 * 0.5;
        let oy = view.center[1] + upp * height as f64 * 0.5;

        let mut plotted = 0u64;
        while plotted < points {
            (self.x, self.y) = kind.step(params, self.x, self.y);
            if !self.x.is_finite() || !self.y.is_finite() {
                // Sin/cos keep these maps bounded, but guard user parameters.
                (self.x, self.y) = (0.0, 0.0);
                plotted += 1;
                continue;
            }
            if self.warmup > 0 {
                self.warmup -= 1;
                continue;
            }
            plotted += 1;

            let px = (self.x - ox) / upp;
            let py = (oy - self.y) / upp;
            if px >= 0.0 && py >= 0.0 && (px as usize) < width && (py as usize) < height {
                hist[py as usize * width + px as usize].fetch_add(1, Relaxed);
            }
        }
    }
}

/// Resumable, parallel attractor orbits filling one histogram; deterministic
/// for any batch split on any thread count (fixed lanes and seeds, cumulative
/// work splitting — same construction as `ifs::ChaosGame`).
pub struct Orbits {
    lanes: Vec<Lane>,
    total: u64,
}

impl Orbits {
    pub fn new() -> Self {
        Self {
            lanes: (0..LANES).map(Lane::new).collect(),
            total: 0,
        }
    }

    /// Advance by `points` plotted points, accumulating into `hist`
    /// (`width*height`, same view for every batch).
    pub fn advance(
        &mut self,
        kind: Kind,
        params: [f64; 4],
        view: IfsView,
        width: usize,
        height: usize,
        hist: &mut [u32],
        points: u64,
    ) {
        use rayon::prelude::*;

        debug_assert_eq!(hist.len(), width * height);
        let old_total = self.total;
        self.total += points;
        let new_total = self.total;
        let atomic_hist = as_atomic(hist);

        self.lanes.par_iter_mut().enumerate().for_each(|(lane, l)| {
            let n = lane_share(new_total, lane) - lane_share(old_total, lane);
            l.advance(kind, params, view, width, height, atomic_hist, n);
        });
    }
}

impl Default for Orbits {
    fn default() -> Self {
        Self::new()
    }
}

/// One-shot histogram (used by export and tests).
pub fn histogram(
    kind: Kind,
    params: [f64; 4],
    points: u64,
    view: IfsView,
    width: usize,
    height: usize,
) -> Vec<u32> {
    let mut hist = vec![0u32; width * height];
    Orbits::new().advance(kind, params, view, width, height, &mut hist, points);
    hist
}

/// Bounding box of the attractor (min x, min y, max x, max y) from a sample
/// orbit, for fitting the viewport.
pub fn bbox(kind: Kind, params: [f64; 4], samples: u64) -> Option<[f64; 4]> {
    let mut lane = Lane::new(0);
    let (mut min_x, mut min_y, mut max_x, mut max_y) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
    let mut plotted = 0u64;
    for _ in 0..samples + WARMUP as u64 {
        (lane.x, lane.y) = kind.step(params, lane.x, lane.y);
        if !lane.x.is_finite() || !lane.y.is_finite() {
            (lane.x, lane.y) = (0.0, 0.0);
            continue;
        }
        if lane.warmup > 0 {
            lane.warmup -= 1;
            continue;
        }
        min_x = min_x.min(lane.x);
        min_y = min_y.min(lane.y);
        max_x = max_x.max(lane.x);
        max_y = max_y.max(lane.y);
        plotted += 1;
    }
    (plotted > 0).then_some([min_x, min_y, max_x, max_y])
}

/// A named parameter set.
pub struct Preset {
    pub name: &'static str,
    pub kind: Kind,
    pub params: [f64; 4],
}

pub const PRESETS: &[Preset] = &[
    Preset {
        name: "Clifford A",
        kind: Kind::Clifford,
        params: [-1.4, 1.6, 1.0, 0.7],
    },
    Preset {
        name: "Clifford B",
        kind: Kind::Clifford,
        params: [-1.7, 1.3, -0.1, -1.2],
    },
    Preset {
        name: "de Jong A",
        kind: Kind::DeJong,
        params: [-2.7, -0.09, -0.86, -2.2],
    },
    Preset {
        name: "de Jong B",
        kind: Kind::DeJong,
        params: [1.4, -2.3, 2.4, -2.1],
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    fn test_view(kind: Kind, params: [f64; 4]) -> IfsView {
        let b = bbox(kind, params, 10_000).unwrap();
        IfsView {
            center: [(b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5],
            units_per_pixel: ((b[2] - b[0]).max(b[3] - b[1]) * 1.1) / 128.0,
        }
    }

    #[test]
    fn orbits_are_deterministic() {
        let p = &PRESETS[0];
        let view = test_view(p.kind, p.params);
        let a = histogram(p.kind, p.params, 200_000, view, 128, 128);
        let b = histogram(p.kind, p.params, 200_000, view, 128, 128);
        assert_eq!(a, b);
        assert!(a.iter().map(|&n| n as u64).sum::<u64>() > 100_000);
    }

    #[test]
    fn batched_advance_equals_one_shot() {
        let p = &PRESETS[2];
        let view = test_view(p.kind, p.params);
        let one = histogram(p.kind, p.params, 100_000, view, 128, 128);

        let mut hist = vec![0u32; 128 * 128];
        let mut orbits = Orbits::new();
        for batch in [1, 999, 30_000, 69_000] {
            orbits.advance(p.kind, p.params, view, 128, 128, &mut hist, batch);
        }
        assert_eq!(one, hist);
    }

    #[test]
    fn presets_have_finite_nondegenerate_bounds() {
        for p in PRESETS {
            let b = bbox(p.kind, p.params, 10_000).unwrap_or_else(|| panic!("{} empty", p.name));
            assert!(b.iter().all(|v| v.is_finite()), "{} unbounded", p.name);
            assert!(b[2] - b[0] > 0.1 && b[3] - b[1] > 0.1, "{} degenerate", p.name);
        }
    }
}
