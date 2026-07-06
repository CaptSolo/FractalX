// Escape-time renderer (Mandelbrot, Burning Ship, Multibrot), split into:
//
//  1. Data pass: iterates each pixel and writes the smooth (continuous)
//     iteration count into an R32Float texture; interior pixels write -1.
//     Two implementations sharing one iteration core:
//       - `fs_data` (fragment): whole image in one draw — used when
//         max_iter is small enough to finish in a frame.
//       - `cs_chunk` (compute): advances every pixel by a bounded number of
//         iterations per dispatch, persisting per-pixel state — used at high
//         max_iter so no single dispatch stalls the GPU.
//  2. Color pass (`fs_color`): maps the data texture through the cyclic
//     palette. Palette changes only re-run this cheap pass.
//
// Iteration paths: plain f32 (shallow zoom) or perturbation against a
// high-precision CPU reference orbit with rebasing (deep zoom; classic
// Mandelbrot only — the z² algebra doesn't carry to the other formulas).

struct Uniforms {
    center: vec2<f32>,      // complex canvas center (plain path only)
    half_extent: vec2<f32>, // complex half width/height of the visible region
    dc_offset: vec2<f32>,   // canvas center minus reference point (perturbation)
    max_iter: u32,
    ref_len: u32,           // number of valid entries in ref_orbit
    use_perturb: u32,       // 0 = plain path, 1 = perturbation path
    formula: u32,           // 0 = Mandelbrot, 1 = Burning Ship, 2 = Multibrot
    power: u32,             // Multibrot exponent (>= 2)
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
const RUNNING: f32 = -2.0; // provisional: chunked pixel not yet resolved

// Smooth (continuous) iteration count at escape.
fn smooth_iter(iter: f32, z2: f32) -> f32 {
    let log_zn = log(z2) * 0.5;
    let nu = log(log_zn / log(2.0)) / log(2.0);
    return iter + 1.0 - nu;
}

// One plain-path step: z' = F(z) + c for the selected formula.
fn step_plain(z: vec2<f32>, c: vec2<f32>) -> vec2<f32> {
    switch u.formula {
        case 1u: { // Burning Ship: fold into the first quadrant, then square
            let a = abs(z);
            return cmul(a, a) + c;
        }
        case 2u: { // Multibrot: z^power + c (integer power, repeated mul)
            var w = z;
            for (var k = 1u; k < u.power; k = k + 1u) {
                w = cmul(w, z);
            }
            return w + c;
        }
        default: {
            return cmul(z, z) + c;
        }
    }
}

// Resumable per-pixel iteration state (also the chunked compute cell).
struct IterState {
    z: vec2<f32>,  // plain: z; perturbation: delta from the reference
    i: u32,
    m: u32,        // perturbation: index into the reference orbit
    done: u32,
    result: f32,
};

fn fresh_state() -> IterState {
    return IterState(vec2<f32>(0.0, 0.0), 0u, 0u, 0u, RUNNING);
}

// Advance a pixel by up to `budget` iterations. `cc` is c (plain) or the
// pixel's dc (perturbation). Sets done/result on escape or max_iter.
fn iterate(s: ptr<function, IterState>, cc: vec2<f32>, budget: u32) {
    var spent = 0u;
    loop {
        if (*s).i >= u.max_iter {
            (*s).done = 1u;
            (*s).result = INTERIOR;
            return;
        }
        if spent >= budget {
            return; // budget exhausted; caller persists state
        }
        if u.use_perturb == 1u {
            // delta' = delta*(2*Z + delta) + dc, with rebasing (Zhuoran).
            let zref = ref_orbit[(*s).m];
            (*s).z = cmul((*s).z, 2.0 * zref + (*s).z) + cc;
            (*s).m = (*s).m + 1u;
            (*s).i = (*s).i + 1u;
            spent = spent + 1u;

            let z = ref_orbit[(*s).m] + (*s).z; // full value
            let z2 = dot(z, z);
            if z2 > BAILOUT {
                (*s).done = 1u;
                (*s).result = smooth_iter(f32((*s).i), z2);
                return;
            }
            // Rebase when the reference stops dominating, or at orbit end.
            if (*s).m + 1u >= u.ref_len || z2 < dot((*s).z, (*s).z) {
                (*s).z = z;
                (*s).m = 0u;
            }
        } else {
            (*s).z = step_plain((*s).z, cc);
            (*s).i = (*s).i + 1u;
            spent = spent + 1u;
            let z2 = dot((*s).z, (*s).z);
            if z2 > BAILOUT {
                (*s).done = 1u;
                (*s).result = smooth_iter(f32((*s).i), z2);
                return;
            }
        }
    }
}

// c (plain) or dc (perturbation) for a pixel at uv in [0,1].
fn pixel_cc(uv: vec2<f32>) -> vec2<f32> {
    // uv (0,0) = top-left of canvas. Complex plane: x right, y up.
    let off = vec2<f32>(
        (uv.x - 0.5) * 2.0 * u.half_extent.x,
        (0.5 - uv.y) * 2.0 * u.half_extent.y,
    );
    if u.use_perturb == 1u {
        return u.dc_offset + off;
    }
    return u.center + off;
}

@fragment
fn fs_data(in: VertexOut) -> @location(0) vec4<f32> {
    var s = fresh_state();
    iterate(&s, pixel_cc(in.uv), u.max_iter);
    return vec4<f32>(s.result, 0.0, 0.0, 1.0);
}

// ---- Chunked compute pass --------------------------------------------------

struct ChunkParams {
    size: vec2<u32>,   // data texture size in texels
    chunk_iters: u32,  // iteration budget per dispatch
    reset: u32,        // 1 = (re)initialize state on this dispatch
};

@group(0) @binding(2)
var<storage, read_write> cells: array<IterState>;
@group(0) @binding(3)
var<uniform> cp: ChunkParams;
@group(0) @binding(4)
var data_out: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn cs_chunk(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= cp.size.x || gid.y >= cp.size.y {
        return;
    }
    let idx = gid.y * cp.size.x + gid.x;

    var s: IterState;
    if cp.reset == 1u {
        s = fresh_state();
    } else {
        s = cells[idx];
        if s.done == 1u {
            return; // final value already in the texture
        }
    }

    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(cp.size);
    iterate(&s, pixel_cc(uv), cp.chunk_iters);

    // RUNNING renders as interior (black) until the pixel resolves.
    textureStore(data_out, vec2<i32>(gid.xy), vec4<f32>(s.result, 0.0, 0.0, 1.0));
    cells[idx] = s;
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
        return vec4<f32>(0.0, 0.0, 0.0, 1.0); // interior or still running
    }
    let t = v / 64.0; // color cycle length in iterations
    return vec4<f32>(palette(t), 1.0);
}
