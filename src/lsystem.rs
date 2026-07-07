//! L-systems: parallel string rewriting interpreted as turtle graphics.
//!
//! An L-system is an axiom string plus rewrite rules, expanded for a number
//! of generations and then read as turtle commands: `F`/`G` draw a unit step,
//! `f`/`g` move without drawing, `+`/`-` turn by the system's angle, `[`/`]`
//! push/pop the turtle state (branching). Every other symbol is a placeholder
//! that only drives the rewriting. The resulting line segments live in world
//! coordinates (y up) and are rasterized on the CPU, colored by arc position
//! through the shared palette — deterministic by construction (no RNG).

use crate::palette::Palette;

/// Expansion stops before the symbol string outgrows this many bytes (keeps
/// the UI responsive if the user cranks generations on a fast-growing
/// system). Memory scales at ~32 bytes per drawn symbol (one `Segment`), so
/// this cap bounds a render at roughly 500 MB transient; pan re-rasterizes
/// every frame, so much beyond this segment count would also drop frames.
pub const MAX_SYMBOLS: usize = 16_000_000;

/// One rewrite rule: every occurrence of `symbol` becomes `replacement`.
#[derive(Clone, PartialEq, Debug, serde::Serialize, serde::Deserialize)]
pub struct Rule {
    pub symbol: char,
    pub replacement: String,
}

/// A line segment in world coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Segment {
    pub x0: f64,
    pub y0: f64,
    pub x1: f64,
    pub y1: f64,
}

/// World-space window onto the drawing, mirroring `ifs::IfsView`.
#[derive(Clone, Copy)]
pub struct View {
    pub center: [f64; 2],
    pub units_per_pixel: f64,
}

/// Rewrite `axiom` for `generations` rounds. If a round would exceed `cap`
/// symbols, the previous (complete) generation is returned instead — never a
/// partially rewritten string, so output stays deterministic. Returns the
/// symbols and the number of rounds actually applied (lets the UI warn when
/// the cap cuts the expansion short).
pub fn expand(axiom: &str, rules: &[Rule], generations: u32, cap: usize) -> (String, u32) {
    let mut s: String = axiom.into();
    for done in 0..generations {
        let mut next = String::with_capacity(s.len() * 2);
        for ch in s.chars() {
            match rules.iter().find(|r| r.symbol == ch) {
                Some(r) => next.push_str(&r.replacement),
                None => next.push(ch),
            }
            // Byte length: O(1), and equal to the symbol count for the
            // ASCII alphabets L-systems use.
            if next.len() > cap {
                return (s, done);
            }
        }
        s = next;
    }
    (s, generations)
}

/// Interpret an expanded string as turtle commands (unit step, starting at
/// the origin heading up). Unbalanced `]` are ignored.
pub fn turtle_segments(symbols: &str, angle_deg: f64) -> Vec<Segment> {
    let ang = angle_deg.to_radians();
    let mut pos = [0.0f64, 0.0f64];
    let mut heading = std::f64::consts::FRAC_PI_2;
    let mut stack: Vec<([f64; 2], f64)> = Vec::new();
    let mut segs = Vec::new();
    for ch in symbols.chars() {
        match ch {
            'F' | 'G' | 'f' | 'g' => {
                let next = [pos[0] + heading.cos(), pos[1] + heading.sin()];
                if ch.is_uppercase() {
                    segs.push(Segment {
                        x0: pos[0],
                        y0: pos[1],
                        x1: next[0],
                        y1: next[1],
                    });
                }
                pos = next;
            }
            '+' => heading += ang,
            '-' => heading -= ang,
            '[' => stack.push((pos, heading)),
            ']' => {
                if let Some((p, h)) = stack.pop() {
                    pos = p;
                    heading = h;
                }
            }
            _ => {}
        }
    }
    segs
}

/// Expand and interpret in one step; also returns the generations actually
/// applied (less than requested when `MAX_SYMBOLS` cuts expansion short).
pub fn segments(
    axiom: &str,
    rules: &[Rule],
    angle_deg: f64,
    generations: u32,
) -> (Vec<Segment>, u32) {
    let (symbols, done) = expand(axiom, rules, generations, MAX_SYMBOLS);
    (turtle_segments(&symbols, angle_deg), done)
}

/// Bounding box `[min_x, min_y, max_x, max_y]` over all segment endpoints.
pub fn bounds(segs: &[Segment]) -> Option<[f64; 4]> {
    let mut it = segs.iter();
    let first = it.next()?;
    let mut b = [
        first.x0.min(first.x1),
        first.y0.min(first.y1),
        first.x0.max(first.x1),
        first.y0.max(first.y1),
    ];
    for s in it {
        b[0] = b[0].min(s.x0).min(s.x1);
        b[1] = b[1].min(s.y0).min(s.y1);
        b[2] = b[2].max(s.x0).max(s.x1);
        b[3] = b[3].max(s.y0).max(s.y1);
    }
    Some(b)
}

/// Liang–Barsky segment/rect clip; `None` if fully outside.
fn clip(
    mut x0: f64,
    mut y0: f64,
    mut x1: f64,
    mut y1: f64,
    max_x: f64,
    max_y: f64,
) -> Option<(f64, f64, f64, f64)> {
    let (dx, dy) = (x1 - x0, y1 - y0);
    let (mut t0, mut t1) = (0.0f64, 1.0f64);
    for (p, q) in [
        (-dx, x0),
        (dx, max_x - x0),
        (-dy, y0),
        (dy, max_y - y0),
    ] {
        if p == 0.0 {
            if q < 0.0 {
                return None;
            }
        } else {
            let r = q / p;
            if p < 0.0 {
                if r > t1 {
                    return None;
                }
                t0 = t0.max(r);
            } else {
                if r < t0 {
                    return None;
                }
                t1 = t1.min(r);
            }
        }
    }
    if t0 > t1 {
        return None;
    }
    (x1, y1) = (x0 + t1 * dx, y0 + t1 * dy);
    (x0, y0) = (x0 + t0 * dx, y0 + t0 * dy);
    Some((x0, y0, x1, y1))
}

/// Rasterize segments to an opaque RGBA image: black background, each segment
/// colored by its position along the curve (0..1) through the palette.
pub fn rasterize_rgba(
    segs: &[Segment],
    view: View,
    w: usize,
    h: usize,
    palette: Palette,
    palette_freq: f32,
    palette_phase: f32,
) -> Vec<u8> {
    let mut rgba = vec![0u8; w * h * 4];
    for px in rgba.chunks_exact_mut(4) {
        px[3] = 255;
    }
    let inv_n = 1.0 / segs.len().max(1) as f32;
    let to_px = |x: f64, y: f64| {
        (
            (x - view.center[0]) / view.units_per_pixel + w as f64 * 0.5,
            (view.center[1] - y) / view.units_per_pixel + h as f64 * 0.5,
        )
    };
    for (i, s) in segs.iter().enumerate() {
        let [r, g, b] = palette.eval(i as f32 * inv_n, palette_freq, palette_phase);
        let (x0, y0) = to_px(s.x0, s.y0);
        let (x1, y1) = to_px(s.x1, s.y1);
        let Some((x0, y0, x1, y1)) = clip(x0, y0, x1, y1, w as f64 - 1.0, h as f64 - 1.0)
        else {
            continue;
        };
        // DDA in pixel space; step count is bounded by the clip rect.
        let steps = (x1 - x0).abs().max((y1 - y0).abs()).ceil() as usize + 1;
        for k in 0..steps {
            let t = k as f64 / steps.max(1) as f64;
            let x = (x0 + t * (x1 - x0)).round() as usize;
            let y = (y0 + t * (y1 - y0)).round() as usize;
            if x < w && y < h {
                let o = (y * w + x) * 4;
                rgba[o] = r;
                rgba[o + 1] = g;
                rgba[o + 2] = b;
            }
        }
    }
    rgba
}

/// A named starter system.
pub struct Preset {
    pub name: &'static str,
    pub axiom: &'static str,
    pub rules: &'static [(char, &'static str)],
    pub angle_deg: f64,
    pub generations: u32,
}

impl Preset {
    pub fn rules_vec(&self) -> Vec<Rule> {
        self.rules
            .iter()
            .map(|&(symbol, replacement)| Rule {
                symbol,
                replacement: replacement.into(),
            })
            .collect()
    }
}

pub const PRESETS: &[Preset] = &[
    Preset {
        name: "Koch snowflake",
        axiom: "F--F--F",
        rules: &[('F', "F+F--F+F")],
        angle_deg: 60.0,
        generations: 4,
    },
    Preset {
        name: "Dragon curve",
        axiom: "F",
        rules: &[('F', "F+G"), ('G', "F-G")],
        angle_deg: 90.0,
        generations: 12,
    },
    Preset {
        name: "Sierpinski arrowhead",
        axiom: "F",
        rules: &[('F', "G-F-G"), ('G', "F+G+F")],
        angle_deg: 60.0,
        generations: 7,
    },
    Preset {
        name: "Plant",
        axiom: "X",
        rules: &[('X', "F+[[X]-X]-F[-FX]+X"), ('F', "FF")],
        angle_deg: 25.0,
        generations: 6,
    },
    Preset {
        name: "Hilbert curve",
        axiom: "A",
        rules: &[('A', "-BF+AFA+FB-"), ('B', "+AF-BFB-FA+")],
        angle_deg: 90.0,
        generations: 6,
    },
    Preset {
        name: "Gosper curve",
        axiom: "F",
        rules: &[('F', "F+G++G-F--FF-G+"), ('G', "-F+GG++G+F--F-G")],
        angle_deg: 60.0,
        generations: 4,
    },
    Preset {
        name: "Lévy C curve",
        axiom: "F",
        rules: &[('F', "+F--F+")],
        angle_deg: 45.0,
        generations: 12,
    },
    Preset {
        name: "Koch island",
        axiom: "F-F-F-F",
        rules: &[('F', "F-F+F+FF-F-F+F")],
        angle_deg: 90.0,
        generations: 3,
    },
    Preset {
        name: "Bush",
        axiom: "F",
        rules: &[('F', "FF-[-F+F+F]+[+F-F-F]")],
        angle_deg: 22.5,
        generations: 4,
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    fn rule(symbol: char, replacement: &str) -> Rule {
        Rule {
            symbol,
            replacement: replacement.into(),
        }
    }

    #[test]
    fn expansion_rewrites_and_passes_through() {
        let rules = [rule('F', "F+F--F+F")];
        assert_eq!(
            expand("F--F", &rules, 1, MAX_SYMBOLS),
            ("F+F--F+F--F+F--F+F".into(), 1)
        );
        // Symbols without a rule survive unchanged.
        assert_eq!(expand("X-Y", &[], 3, MAX_SYMBOLS), ("X-Y".into(), 3));
    }

    #[test]
    fn expansion_respects_symbol_cap() {
        // Doubles every generation; a tight cap must return a complete
        // earlier generation, not a truncated string, and report how many
        // generations actually ran.
        let rules = [rule('F', "FF")];
        let (s, done) = expand("F", &rules, 30, 1000);
        assert!(s.len() <= 1000);
        assert!(s.len().is_power_of_two());
        assert!(s.chars().all(|c| c == 'F'));
        // 2^done symbols: the cap of 1000 stops after generation 9 (512).
        assert_eq!(done, 9);
        assert_eq!(s.len(), 1 << done);
    }

    #[test]
    fn capped_expansion_of_fast_growing_system_completes_quickly() {
        // Regression: the cap check must be O(1) per symbol — a rescan of
        // the output string per appended symbol made expansion quadratic,
        // hanging the UI at high generation counts.
        for p in PRESETS {
            let (s, done) = expand(p.axiom, &p.rules_vec(), 16, MAX_SYMBOLS);
            assert!(s.len() <= MAX_SYMBOLS);
            assert!(done <= 16);
        }
    }

    #[test]
    fn square_walk_returns_to_start() {
        let segs = turtle_segments("F+F+F+F", 90.0);
        assert_eq!(segs.len(), 4);
        let last = segs.last().unwrap();
        assert!((last.x1 - segs[0].x0).abs() < 1e-9);
        assert!((last.y1 - segs[0].y0).abs() < 1e-9);
    }

    #[test]
    fn brackets_branch_and_restore() {
        // F [ +F ] F: third segment continues from the first, not the branch.
        let segs = turtle_segments("F[+F]F", 90.0);
        assert_eq!(segs.len(), 3);
        assert_eq!((segs[2].x0, segs[2].y0), (segs[0].x1, segs[0].y1));
        // Lowercase moves without drawing.
        assert_eq!(turtle_segments("fFf", 90.0).len(), 1);
    }

    #[test]
    fn rasterize_draws_colored_pixels_on_black() {
        let (segs, _) = segments("F+F+F+F", &[], 90.0, 0);
        let b = bounds(&segs).unwrap();
        let view = View {
            center: [(b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5],
            units_per_pixel: 2.0 / 64.0,
        };
        let rgba = rasterize_rgba(&segs, view, 64, 64, Palette::Classic, 1.0, 0.0);
        assert_eq!(rgba.len(), 64 * 64 * 4);
        let lit = rgba
            .chunks_exact(4)
            .filter(|px| px[0] != 0 || px[1] != 0 || px[2] != 0)
            .count();
        // A ~32px square outline: significant but sparse coverage.
        assert!(lit > 60 && lit < 1000, "lit = {lit}");
        assert!(rgba.chunks_exact(4).all(|px| px[3] == 255));
        // Deterministic.
        assert_eq!(
            rgba,
            rasterize_rgba(&segs, view, 64, 64, Palette::Classic, 1.0, 0.0)
        );
    }

    #[test]
    fn segments_outside_view_are_clipped_away() {
        let segs = [Segment {
            x0: 100.0,
            y0: 100.0,
            x1: 200.0,
            y1: 100.0,
        }];
        let view = View {
            center: [0.0, 0.0],
            units_per_pixel: 0.1,
        };
        let rgba = rasterize_rgba(&segs, view, 32, 32, Palette::Classic, 1.0, 0.0);
        assert!(rgba
            .chunks_exact(4)
            .all(|px| px[0] == 0 && px[1] == 0 && px[2] == 0));
    }

    #[test]
    fn presets_produce_nonempty_bounded_geometry() {
        for p in PRESETS {
            let (segs, done) = segments(p.axiom, &p.rules_vec(), p.angle_deg, p.generations);
            assert_eq!(done, p.generations, "{} preset hits the cap", p.name);
            assert!(!segs.is_empty(), "{} drew nothing", p.name);
            let b = bounds(&segs).unwrap();
            assert!(b.iter().all(|v| v.is_finite()), "{} unbounded", p.name);
        }
    }
}
