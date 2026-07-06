// Escape-time Mandelbrot renderer.
// Drawn as a fullscreen triangle over the callback viewport; uv covers the
// canvas rect in [0,1] with y down (screen convention).
//
// Two paths:
//  - plain: c computed directly in f32 (shallow zoom)
//  - perturbation: pixel iterates its delta from a high-precision reference
//    orbit (computed on CPU), with rebasing when the delta dominates. This
//    keeps f32 sufficient down to ~1e-30 scales.

struct Uniforms {
    center: vec2<f32>,      // complex canvas center (plain path only)
    half_extent: vec2<f32>, // complex half width/height of the visible region
    dc_offset: vec2<f32>,   // canvas center minus reference point (perturbation)
    max_iter: u32,
    ref_len: u32,           // number of valid entries in ref_orbit
    use_perturb: u32,       // 0 = plain path, 1 = perturbation path
    palette_freq: f32,
    palette_phase: f32,
    _pad: f32,
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

fn palette(t: f32) -> vec3<f32> {
    // Cyclic cosine palette.
    let phase = vec3<f32>(0.0, 0.33, 0.67) + u.palette_phase;
    return 0.5 + 0.5 * cos(6.2831853 * (t * u.palette_freq + phase));
}

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

const BAILOUT: f32 = 256.0; // large bailout for smooth coloring

fn color(iter: f32, z2: f32) -> vec4<f32> {
    if iter < 0.0 {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0); // interior
    }
    // Smooth (continuous) iteration count.
    let log_zn = log(z2) * 0.5;
    let nu = log(log_zn / log(2.0)) / log(2.0);
    let smooth_i = iter + 1.0 - nu;
    let t = smooth_i / 64.0; // color cycle length in iterations
    return vec4<f32>(palette(t), 1.0);
}

// Plain f32 escape-time iteration.
fn iterate_plain(c: vec2<f32>) -> vec4<f32> {
    var z = vec2<f32>(0.0, 0.0);
    var i: u32 = 0u;
    loop {
        if i >= u.max_iter {
            return color(-1.0, 0.0);
        }
        z = cmul(z, z) + c;
        i = i + 1u;
        if dot(z, z) > BAILOUT {
            return color(f32(i), dot(z, z));
        }
    }
    return color(-1.0, 0.0); // unreachable; satisfies return analysis
}

// Perturbation iteration with rebasing (Zhuoran's method):
// delta' = delta*(2*Z + delta) + dc; rebase to orbit start when the full
// value falls below the delta or the reference orbit runs out.
fn iterate_perturb(dc: vec2<f32>) -> vec4<f32> {
    var dz = vec2<f32>(0.0, 0.0);
    var m: u32 = 0u; // index into the reference orbit
    var i: u32 = 0u;
    loop {
        if i >= u.max_iter {
            return color(-1.0, 0.0);
        }
        let zref = ref_orbit[m];
        dz = cmul(dz, 2.0 * zref + dz) + dc;
        m = m + 1u;
        i = i + 1u;

        let z = ref_orbit[m] + dz; // full value at iteration i
        let z2 = dot(z, z);
        if z2 > BAILOUT {
            return color(f32(i), z2);
        }
        // Rebase when the reference no longer dominates, or at orbit end.
        if m + 1u >= u.ref_len || z2 < dot(dz, dz) {
            dz = z;
            m = 0u;
        }
    }
    return color(-1.0, 0.0); // unreachable; satisfies return analysis
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    // uv (0,0) = top-left of canvas. Complex plane: x right, y up.
    let off = vec2<f32>(
        (in.uv.x - 0.5) * 2.0 * u.half_extent.x,
        (0.5 - in.uv.y) * 2.0 * u.half_extent.y,
    );
    if u.use_perturb == 1u {
        return iterate_perturb(u.dc_offset + off);
    }
    return iterate_plain(u.center + off);
}
