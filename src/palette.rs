//! Shared color palettes: named cosine-gradient presets.
//!
//! Every palette is `a + b*cos(2*pi*(c*x + d))` per RGB channel, where
//! `x = t * freq + phase` (the user's frequency/phase sliders modulate any
//! preset). `Classic` reproduces the original hard-coded palette exactly.
//! The GPU color pass and the CPU IFS tone-map both evaluate this formula
//! from the same coefficients.

/// User-editable cosine-gradient coefficients: one row per `a`/`b`/`c`/`d`,
/// RGB columns.
#[derive(Clone, Copy, PartialEq, Debug, serde::Serialize, serde::Deserialize)]
pub struct Coeffs {
    pub a: [f32; 3],
    pub b: [f32; 3],
    pub c: [f32; 3],
    pub d: [f32; 3],
}

impl Coeffs {
    /// From the vec4-padded rows `coeffs()` returns (padding dropped).
    pub fn from_rows(rows: [[f32; 4]; 4]) -> Self {
        let row = |r: [f32; 4]| [r[0], r[1], r[2]];
        Self {
            a: row(rows[0]),
            b: row(rows[1]),
            c: row(rows[2]),
            d: row(rows[3]),
        }
    }
}

/// A palette: a named preset, or `Custom` coefficients from the editor.
/// Presets serialize by name in bookmarks (absent in older bookmarks, which
/// default to `Classic`); `Custom` carries its coefficients.
#[derive(Clone, Copy, PartialEq, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Palette {
    #[default]
    Classic,
    Sunset,
    Fire,
    Electric,
    Pastel,
    Grayscale,
    Custom(Coeffs),
}

const fn pad(v: [f32; 3]) -> [f32; 4] {
    [v[0], v[1], v[2], 0.0]
}

impl Palette {
    /// The named presets (the combo box list; `Custom` is entered by editing).
    pub const ALL: [Palette; 6] = [
        Palette::Classic,
        Palette::Sunset,
        Palette::Fire,
        Palette::Electric,
        Palette::Pastel,
        Palette::Grayscale,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Palette::Classic => "Classic",
            Palette::Sunset => "Sunset",
            Palette::Fire => "Fire",
            Palette::Electric => "Electric",
            Palette::Pastel => "Pastel",
            Palette::Grayscale => "Grayscale",
            Palette::Custom(_) => "Custom",
        }
    }

    /// Coefficient rows `[a, b, c, d]`, each padded to 4 floats so they can
    /// be uploaded directly as `vec4<f32>` uniform fields.
    pub const fn coeffs(self) -> [[f32; 4]; 4] {
        match self {
            Palette::Classic => [
                [0.5, 0.5, 0.5, 0.0],
                [0.5, 0.5, 0.5, 0.0],
                [1.0, 1.0, 1.0, 0.0],
                [0.0, 0.33, 0.67, 0.0],
            ],
            Palette::Sunset => [
                [0.5, 0.5, 0.5, 0.0],
                [0.5, 0.5, 0.5, 0.0],
                [1.0, 1.0, 1.0, 0.0],
                [0.0, 0.10, 0.20, 0.0],
            ],
            Palette::Fire => [
                [0.5, 0.5, 0.5, 0.0],
                [0.5, 0.5, 0.5, 0.0],
                [1.0, 0.7, 0.4, 0.0],
                [0.0, 0.15, 0.20, 0.0],
            ],
            Palette::Electric => [
                [0.5, 0.5, 0.5, 0.0],
                [0.5, 0.5, 0.5, 0.0],
                [2.0, 1.0, 0.0, 0.0],
                [0.5, 0.20, 0.25, 0.0],
            ],
            Palette::Pastel => [
                [0.8, 0.5, 0.4, 0.0],
                [0.2, 0.4, 0.2, 0.0],
                [2.0, 1.0, 1.0, 0.0],
                [0.0, 0.25, 0.25, 0.0],
            ],
            Palette::Grayscale => [
                [0.5, 0.5, 0.5, 0.0],
                [0.5, 0.5, 0.5, 0.0],
                [1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            Palette::Custom(k) => [pad(k.a), pad(k.b), pad(k.c), pad(k.d)],
        }
    }

    /// CPU evaluation, mirror of the shader's `palette()`.
    pub fn eval(self, t: f32, freq: f32, phase: f32) -> [u8; 3] {
        let [a, b, c, d] = self.coeffs();
        let x = t * freq + phase;
        let mut rgb = [0u8; 3];
        for i in 0..3 {
            let v = a[i] + b[i] * (6.2831853 * (c[i] * x + d[i])).cos();
            rgb[i] = (v.clamp(0.0, 1.0) * 255.0) as u8;
        }
        rgb
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `Classic` must reproduce the original hard-coded palette
    /// (0.5 + 0.5*cos(2*pi*(t*freq + offset + phase))) so existing bookmarks
    /// keep rendering identically.
    #[test]
    fn classic_matches_original_formula() {
        for &(t, freq, phase) in &[(0.0f32, 1.0f32, 0.0f32), (0.37, 2.5, 0.6), (5.1, 0.3, 0.9)] {
            let mut expected = [0u8; 3];
            for (i, base) in [0.0f32, 0.33, 0.67].iter().enumerate() {
                let v = 0.5 + 0.5 * (6.2831853 * (t * freq + base + phase)).cos();
                expected[i] = (v.clamp(0.0, 1.0) * 255.0) as u8;
            }
            assert_eq!(Palette::Classic.eval(t, freq, phase), expected);
        }
    }

    #[test]
    fn serializes_by_snake_case_name() {
        assert_eq!(serde_json::to_string(&Palette::Classic).unwrap(), r#""classic""#);
        let p: Palette = serde_json::from_str(r#""grayscale""#).unwrap();
        assert_eq!(p, Palette::Grayscale);
    }

    #[test]
    fn custom_coefficients_round_trip() {
        let custom = Palette::Custom(Coeffs {
            a: [0.6, 0.5, 0.4],
            b: [0.3, 0.4, 0.5],
            c: [1.0, 2.0, 0.5],
            d: [0.1, 0.2, 0.3],
        });
        let json = serde_json::to_string(&custom).unwrap();
        let back: Palette = serde_json::from_str(&json).unwrap();
        assert_eq!(back, custom);
        // Coefficients drive rendering exactly as edited.
        assert_eq!(custom.coeffs()[2], [1.0, 2.0, 0.5, 0.0]);
    }
}
