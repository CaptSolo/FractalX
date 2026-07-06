//! High-precision math for deep zoom: arbitrary-precision center coordinates
//! and the perturbation reference orbit.
//!
//! Deep zoom uses standard perturbation theory: one reference orbit is
//! iterated at high precision on the CPU; the GPU then iterates only each
//! pixel's small delta from that orbit in f32 (see `mandelbrot.wgsl`).

use std::str::FromStr;

/// Binary bigfloat with half-away rounding (matches dashu's decimal type so
/// base conversions keep one rounding mode throughout).
pub type FBig = dashu::float::FBig<dashu::float::round::mode::HalfAway, 2>;

/// Extra mantissa bits beyond the zoom depth, as safety margin.
const GUARD_BITS: usize = 64;

/// Mantissa bits needed to resolve pixels at the given scale.
pub fn precision_for(units_per_point: f64) -> usize {
    let zoom_bits = (-units_per_point.log2()).ceil().max(0.0) as usize;
    zoom_bits + GUARD_BITS
}

/// An arbitrary-precision point in the complex plane.
#[derive(Clone, PartialEq)]
pub struct BigComplex {
    pub re: FBig,
    pub im: FBig,
}

impl BigComplex {
    pub fn from_f64(re: f64, im: f64) -> Self {
        Self {
            re: FBig::try_from(re).expect("finite"),
            im: FBig::try_from(im).expect("finite"),
        }
    }

    /// Parse from decimal strings (bookmark format).
    pub fn from_decimal(re: &str, im: &str, precision: usize) -> Result<Self, String> {
        let parse = |s: &str| -> Result<FBig, String> {
            Ok(dashu::float::DBig::from_str(s)
                .map_err(|e| format!("bad coordinate {s:?}: {e}"))?
                .with_base_and_precision::<2>(precision)
                .value())
        };
        Ok(Self {
            re: parse(re)?,
            im: parse(im)?,
        })
    }

    /// Decimal strings with enough digits to reproduce the view.
    ///
    /// dashu's base conversion can be off by 1 ulp, so the round trip is not
    /// bit-exact — but with [`GUARD_BITS`] of margin the error sits ~2^-64
    /// below pixel size, far past visible.
    pub fn to_decimal(&self) -> [String; 2] {
        let bits = self.re.precision().max(self.im.precision()).max(64);
        // bits * log10(2), rounded up, plus margin
        self.to_decimal_digits((bits as f64 * 0.30103).ceil() as usize + 4)
    }

    /// Decimal strings truncated to `digits` significant digits (for display).
    pub fn to_decimal_digits(&self, digits: usize) -> [String; 2] {
        let fmt = |x: &FBig| {
            x.clone()
                .with_base_and_precision::<10>(digits)
                .value()
                .to_string()
        };
        [fmt(&self.re), fmt(&self.im)]
    }

    /// Add an offset that is small enough to be exact in f64.
    pub fn offset(&mut self, dre: f64, dim: f64, precision: usize) {
        let dre = FBig::try_from(dre).expect("finite");
        let dim = FBig::try_from(dim).expect("finite");
        self.re = (&self.re + dre).with_precision(precision).value();
        self.im = (&self.im + dim).with_precision(precision).value();
    }

    /// Raise (or lower) the stored precision.
    pub fn set_precision(&mut self, precision: usize) {
        self.re = self.re.clone().with_precision(precision).value();
        self.im = self.im.clone().with_precision(precision).value();
    }

    /// Difference `self - other`, which must fit in f64 (used for the
    /// center-to-reference offset, always within a few view extents).
    pub fn sub_to_f64(&self, other: &Self) -> [f64; 2] {
        [
            (&self.re - &other.re).to_f64().value(),
            (&self.im - &other.im).to_f64().value(),
        ]
    }

    pub fn to_f64(&self) -> [f64; 2] {
        [self.re.to_f64().value(), self.im.to_f64().value()]
    }
}

/// Reference orbit for perturbation: Z_0 = 0, Z_{n+1} = Z_n² + C, stored as
/// f32 pairs for the GPU. Stops after escape (|Z|² > 4) or `max_iter`.
pub struct ReferenceOrbit {
    pub points: Vec<[f32; 2]>,
    /// True if the orbit escaped before `max_iter` (so extending `max_iter`
    /// would not lengthen it).
    pub escaped: bool,
}

pub fn reference_orbit(c: &BigComplex, max_iter: u32, precision: usize) -> ReferenceOrbit {
    let cre = c.re.clone().with_precision(precision).value();
    let cim = c.im.clone().with_precision(precision).value();
    let mut zr = FBig::ZERO.with_precision(precision).value();
    let mut zi = FBig::ZERO.with_precision(precision).value();

    let mut points = Vec::with_capacity(max_iter as usize + 1);
    points.push([0.0f32, 0.0f32]);
    let mut escaped = false;

    for _ in 0..max_iter {
        let zr2 = zr.sqr();
        let zi2 = zi.sqr();
        let zrzi = &zr * &zi;
        let new_zi = (&zrzi + &zrzi + &cim).with_precision(precision).value();
        zr = (zr2 - zi2 + &cre).with_precision(precision).value();
        zi = new_zi;

        let fr = zr.to_f64().value();
        let fi = zi.to_f64().value();
        points.push([fr as f32, fi as f32]);

        if fr * fr + fi * fi > 4.0 {
            escaped = true;
            break;
        }
    }

    ReferenceOrbit { points, escaped }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orbit_matches_f64_iteration_at_shallow_depth() {
        let c = BigComplex::from_f64(-0.1, 0.65);
        let orbit = reference_orbit(&c, 100, 128);

        // Same iteration in plain f64.
        let (mut zr, mut zi) = (0.0f64, 0.0f64);
        for (n, p) in orbit.points.iter().enumerate().take(50) {
            assert!(
                (p[0] as f64 - zr).abs() < 1e-6 && (p[1] as f64 - zi).abs() < 1e-6,
                "orbit diverges from f64 at iteration {n}"
            );
            let t = zr * zr - zi * zi + (-0.1);
            zi = 2.0 * zr * zi + 0.65;
            zr = t;
        }
    }

    #[test]
    fn orbit_escapes_for_exterior_point() {
        let c = BigComplex::from_f64(1.0, 1.0);
        let orbit = reference_orbit(&c, 1000, 64);
        assert!(orbit.escaped);
        assert!(orbit.points.len() < 20);
    }

    #[test]
    fn interior_orbit_runs_to_max_iter() {
        let c = BigComplex::from_f64(-0.5, 0.0); // inside the main cardioid
        let orbit = reference_orbit(&c, 500, 64);
        assert!(!orbit.escaped);
        assert_eq!(orbit.points.len(), 501);
    }

    #[test]
    fn decimal_round_trip_preserves_deep_coordinates() {
        // A coordinate needing far more precision than f64.
        let prec = 200;
        let mut c = BigComplex::from_f64(-0.75, 0.1);
        c.set_precision(prec);
        // Nudge by a tiny amount only high precision can hold.
        c.offset(1e-45, -3e-46, prec);

        let [re, im] = c.to_decimal();
        let back = BigComplex::from_decimal(&re, &im, prec).unwrap();
        let [dre, dim] = c.sub_to_f64(&back);
        // Round trip may be off by ~1 ulp at 200 bits (~1e-61); anything
        // approaching pixel scale at this precision (~1e-41) is a real bug.
        assert!(dre.abs() < 1e-50 && dim.abs() < 1e-50, "lost {dre:e} {dim:e}");
        // And it must actually preserve the deep nudge (1e-45), which f64 cannot.
        let coarse = BigComplex::from_f64(-0.75, 0.1);
        let [nre, _] = back.sub_to_f64(&coarse);
        assert!((nre - 1e-45).abs() < 1e-50, "deep nudge lost: {nre:e}");
    }
}
