// Escape-time Mandelbrot renderer, split into two passes:
//
//  1. Data pass (`fs_data`): iterates each pixel and writes the smooth
//     (continuous) iteration count into an R32Float texture; interior pixels
//     write -1. Two iteration paths:
//      - plain: c computed directly in f32 (shallow zoom)
//      - perturbation: pixel iterates its delta from a high-precision
//        reference orbit (computed on CPU), with rebasing when the delta
//        dominates. Keeps f32 sufficient down to ~1e-30 scales.
//  2. Color pass (`fs_color`): maps the data texture through the cyclic
//     palette. Palette changes only re-run this cheap pass.
//
// Both passes draw a fullscreen triangle over the target.

struct Uniforms {
    center: vec2<f32>,      // complex canvas center (plain path only)
    half_extent: vec2<f32>, // complex half width/height of the visible region
    dc_offset: vec2<f32>,   // canvas center minus reference point (perturbation)
    max_iter: u32,
    ref_len: u32,           // number of valid entries in ref_orbit
    use_perturb: u32,       // 0 = plain path, 1 = perturbation path
    _pad: u32,
};

@group(0) @binding(0)
var<uniform> u: Uniforms;

// Reference orbit Z_n (Z_0 = 0); only used when use_perturb == 1.
@group(0) @binding(1)
var<storage, read> ref_orbit: array<vec2<f32>>;

struct VertexOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOut {
    // Fullscreen triangle: covers the viewport with 3 vertices.
    var out: VertexOut;
    let x = f32(i32(vi & 1u) * 4 - 1); // -1, 3, -1
    let y = f32(i32(vi & 2u) * 2 - 1); // -1, -1, 3
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(x, -y) * 0.5 + 0.5; // y flipped: uv.y grows downward
    return out;
}

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

const BAILOUT: f32 = 256.0; // large bailout for smooth coloring
const INTERIOR: f32 = -1.0;

// Smooth (continuous) iteration count at escape.
fn smooth_iter(iter: f32, z2: f32) -> f32 {
    let log_zn = log(z2) * 0.5;
    let nu = log(log_zn / log(2.0)) / log(2.0);
    return iter + 1.0 - nu;
}

// Plain f32 escape-time iteration.
fn iterate_plain(c: vec2<f32>) -> f32 {
    var z = vec2<f32>(0.0, 0.0);
    var i: u32 = 0u;
    loop {
        if i >= u.max_iter {
            return INTERIOR;
        }
        z = cmul(z, z) + c;
        i = i + 1u;
        if dot(z, z) > BAILOUT {
            return smooth_iter(f32(i), dot(z, z));
        }
    }
    return INTERIOR; // unreachable; satisfies return analysis
}

// Perturbation iteration with rebasing (Zhuoran's method):
// delta' = delta*(2*Z + delta) + dc; rebase to orbit start when the full
// value falls below the delta or the reference orbit runs out.
fn iterate_perturb(dc: vec2<f32>) -> f32 {
    var dz = vec2<f32>(0.0, 0.0);
    var m: u32 = 0u; // index into the reference orbit
    var i: u32 = 0u;
    loop {
        if i >= u.max_iter {
            return INTERIOR;
        }
        let zref = ref_orbit[m];
        dz = cmul(dz, 2.0 * zref + dz) + dc;
        m = m + 1u;
        i = i + 1u;

        let z = ref_orbit[m] + dz; // full value at iteration i
        let z2 = dot(z, z);
        if z2 > BAILOUT {
            return smooth_iter(f32(i), z2);
        }
        // Rebase when the reference no longer dominates, or at orbit end.
        if m + 1u >= u.ref_len || z2 < dot(dz, dz) {
            dz = z;
            m = 0u;
        }
    }
    return INTERIOR; // unreachable; satisfies return analysis
}

@fragment
fn fs_data(in: VertexOut) -> @location(0) vec4<f32> {
    // uv (0,0) = top-left of canvas. Complex plane: x right, y up.
    let off = vec2<f32>(
        (in.uv.x - 0.5) * 2.0 * u.half_extent.x,
        (0.5 - in.uv.y) * 2.0 * u.half_extent.y,
    );
    var v: f32;
    if u.use_perturb == 1u {
        v = iterate_perturb(u.dc_offset + off);
    } else {
        v = iterate_plain(u.center + off);
    }
    return vec4<f32>(v, 0.0, 0.0, 1.0);
}

// ---- Color pass ------------------------------------------------------------

struct PaletteUniforms {
    freq: f32,
    phase: f32,
    _pad: vec2<f32>,
};

@group(0) @binding(0)
var data_tex: texture_2d<f32>;
@group(0) @binding(1)
var<uniform> pal: PaletteUniforms;

fn palette(t: f32) -> vec3<f32> {
    // Cyclic cosine palette.
    let phase = vec3<f32>(0.0, 0.33, 0.67) + pal.phase;
    return 0.5 + 0.5 * cos(6.2831853 * (t * pal.freq + phase));
}

@fragment
fn fs_color(in: VertexOut) -> @location(0) vec4<f32> {
    // Target and data texture are the same size: 1:1 texel mapping.
    let v = textureLoad(data_tex, vec2<i32>(in.pos.xy), 0).x;
    if v < 0.0 {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0); // interior
    }
    let t = v / 64.0; // color cycle length in iterations
    return vec4<f32>(palette(t), 1.0);
}
