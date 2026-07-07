//! Fractal terrain and clouds: statistical self-similarity via fractional
//! Brownian motion (fBm) — octaves of seeded gradient (Perlin) noise over an
//! infinite plane.
//!
//! Everything derives from a hash of (seed, lattice point), so rendering is
//! deterministic for a bookmark and any part of the plane can be evaluated
//! at any zoom without stored state. The Hurst exponent `H` is the
//! roughness ↔ dimension control: octave amplitudes fall off as `2^-H`, and
//! the corresponding surface has fractal dimension `D = 3 − H`.
//!
//! Terrain mode colors height through the shared palette with screen-space
//! hill-shading; clouds mode maps turbulence (sum of |noise|) instead.

use crate::ifs::IfsView;
use crate::palette::Palette;

/// Terrain parameters (part of the bookmark).
#[derive(Clone, Copy, PartialEq, Debug, serde::Serialize, serde::Deserialize)]
pub struct Params {
    pub seed: u64,
    /// Hurst exponent in (0, 1]: octave gain is `2^-hurst`.
    pub hurst: f64,
    pub octaves: u32,
    /// Clouds (turbulence, unshaded) instead of shaded terrain.
    pub clouds: bool,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            seed: 1,
            hurst: 0.9,
            octaves: 8,
            clouds: false,
        }
    }
}

/// SplitMix64-style hash of a seeded lattice point.
fn hash(seed: u64, ix: i64, iy: i64) -> u64 {
    let mut h = seed
        ^ (ix as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (iy as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h ^= h >> 30;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^ (h >> 31)
}

/// Unit gradient at a lattice point.
fn gradient(seed: u64, ix: i64, iy: i64) -> (f64, f64) {
    // 53 uniform bits → angle.
    let angle = (hash(seed, ix, iy) >> 11) as f64 / (1u64 << 53) as f64
        * std::f64::consts::TAU;
    (angle.cos(), angle.sin())
}

fn fade(t: f64) -> f64 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// 2D Perlin gradient noise, roughly in [-1, 1].
fn perlin(seed: u64, x: f64, y: f64) -> f64 {
    let (x0, y0) = (x.floor(), y.floor());
    let (ix, iy) = (x0 as i64, y0 as i64);
    let (fx, fy) = (x - x0, y - y0);

    let dot = |gx: i64, gy: i64| {
        let (dx, dy) = (fx - (gx - ix) as f64, fy - (gy - iy) as f64);
        let (gxv, gyv) = gradient(seed, gx, gy);
        dx * gxv + dy * gyv
    };
    let (u, v) = (fade(fx), fade(fy));
    let top = dot(ix, iy) + u * (dot(ix + 1, iy) - dot(ix, iy));
    let bot = dot(ix, iy + 1) + u * (dot(ix + 1, iy + 1) - dot(ix, iy + 1));
    // Perlin's theoretical 2D range is ±√2/2; rescale toward [-1, 1].
    (top + v * (bot - top)) * std::f64::consts::SQRT_2
}

/// fBm height in [0, 1]: octaves of Perlin noise, gain `2^-hurst`.
pub fn height(params: &Params, x: f64, y: f64) -> f64 {
    let gain = (-params.hurst).exp2();
    let (mut sum, mut amp, mut norm, mut freq) = (0.0, 1.0, 0.0, 1.0);
    for octave in 0..params.octaves.max(1) {
        // Distinct lattice seed per octave so octaves don't align.
        let n = perlin(params.seed.wrapping_add(octave as u64), x * freq, y * freq);
        sum += amp * if params.clouds { n.abs() } else { n };
        norm += amp;
        amp *= gain;
        freq *= 2.0;
    }
    if params.clouds {
        (sum / norm).clamp(0.0, 1.0)
    } else {
        (0.5 + 0.5 * sum / norm).clamp(0.0, 1.0)
    }
}

/// Render the field to RGBA: height through the palette, with screen-space
/// hill-shading in terrain mode (clouds are unshaded).
pub fn render_rgba(
    params: &Params,
    view: IfsView,
    w: usize,
    h: usize,
    palette: Palette,
    palette_freq: f32,
    palette_phase: f32,
) -> Vec<u8> {
    use rayon::prelude::*;

    let upp = view.units_per_pixel;
    let ox = view.center[0] - upp * w as f64 * 0.5;
    let oy = view.center[1] + upp * h as f64 * 0.5;

    // Height field first (parallel over rows), then color + shade from
    // neighbor differences.
    let mut heights = vec![0.0f64; w * h];
    heights.par_chunks_mut(w).enumerate().for_each(|(py, row)| {
        let y = oy - (py as f64 + 0.5) * upp;
        for (px, cell) in row.iter_mut().enumerate() {
            let x = ox + (px as f64 + 0.5) * upp;
            *cell = height(params, x, y);
        }
    });

    let mut rgba = vec![0u8; w * h * 4];
    rgba.par_chunks_mut(w * 4).enumerate().for_each(|(py, row)| {
        for px in 0..w {
            let i = py * w + px;
            let t = heights[i] as f32;
            let [mut r, mut g, mut b] = palette.eval(t, palette_freq, palette_phase);
            if !params.clouds {
                // Light from the north-west, slopes from screen neighbors.
                let right = heights[i + usize::from(px + 1 < w)];
                let below = heights[if py + 1 < h { i + w } else { i }];
                let shade =
                    (1.0 + 60.0 * ((heights[i] - right) + (heights[i] - below))).clamp(0.5, 1.4);
                r = ((r as f64 * shade).min(255.0)) as u8;
                g = ((g as f64 * shade).min(255.0)) as u8;
                b = ((b as f64 * shade).min(255.0)) as u8;
            }
            row[px * 4..px * 4 + 4].copy_from_slice(&[r, g, b, 255]);
        }
    });
    rgba
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_view() -> IfsView {
        IfsView {
            center: [0.3, -1.7],
            units_per_pixel: 0.05,
        }
    }

    #[test]
    fn heights_stay_in_unit_range() {
        let params = Params::default();
        for i in 0..500 {
            let (x, y) = (i as f64 * 0.37 - 91.0, i as f64 * 0.61 + 40.0);
            let v = height(&params, x, y);
            assert!((0.0..=1.0).contains(&v), "height {v} at ({x}, {y})");
        }
    }

    #[test]
    fn render_is_deterministic_and_seed_dependent() {
        let params = Params::default();
        let a = render_rgba(&params, test_view(), 64, 64, Palette::Classic, 1.0, 0.0);
        let b = render_rgba(&params, test_view(), 64, 64, Palette::Classic, 1.0, 0.0);
        assert_eq!(a, b);

        let other = Params {
            seed: 2,
            ..params
        };
        let c = render_rgba(&other, test_view(), 64, 64, Palette::Classic, 1.0, 0.0);
        assert_ne!(a, c, "different seeds must render differently");
    }

    #[test]
    fn hurst_controls_roughness() {
        // Mean absolute neighbor difference (high-frequency energy) must
        // grow as H falls — the roughness ↔ dimension control.
        let roughness = |hurst: f64| {
            let params = Params {
                hurst,
                ..Params::default()
            };
            let mut sum = 0.0;
            let mut n = 0u32;
            for i in 0..2000 {
                let (x, y) = (i as f64 * 0.013, i as f64 * 0.007 + 5.0);
                sum += (height(&params, x + 0.01, y) - height(&params, x, y)).abs();
                n += 1;
            }
            sum / n as f64
        };
        assert!(
            roughness(0.2) > roughness(1.0) * 1.2,
            "low H must be rougher"
        );
    }

    #[test]
    fn clouds_differ_from_terrain() {
        let terrain = Params::default();
        let clouds = Params {
            clouds: true,
            ..terrain
        };
        let a = render_rgba(&terrain, test_view(), 32, 32, Palette::Classic, 1.0, 0.0);
        let b = render_rgba(&clouds, test_view(), 32, 32, Palette::Classic, 1.0, 0.0);
        assert_ne!(a, b);
    }
}
