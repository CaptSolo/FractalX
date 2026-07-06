// Escape-time Mandelbrot renderer.
// Drawn as a fullscreen triangle over the callback viewport; uv covers the
// canvas rect in [0,1] with y down (screen convention).

struct Uniforms {
    center: vec2<f32>,      // complex-plane coordinates of the canvas center
    half_extent: vec2<f32>, // complex half width/height of the visible region
    max_iter: u32,
    palette_freq: f32,
    palette_phase: f32,
    _pad: f32,
};

@group(0) @binding(0)
var<uniform> u: Uniforms;

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

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    // uv (0,0) = top-left of canvas. Complex plane: x right, y up.
    let c = u.center + vec2<f32>(
        (in.uv.x - 0.5) * 2.0 * u.half_extent.x,
        (0.5 - in.uv.y) * 2.0 * u.half_extent.y,
    );

    var z = vec2<f32>(0.0, 0.0);
    var i: u32 = 0u;
    let bailout = 256.0; // large bailout for smooth coloring
    loop {
        if i >= u.max_iter { break; }
        let zz = vec2<f32>(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
        z = zz;
        if dot(z, z) > bailout { break; }
        i = i + 1u;
    }

    if i >= u.max_iter {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0); // interior
    }

    // Smooth (continuous) iteration count.
    let log_zn = log(dot(z, z)) * 0.5;
    let nu = log(log_zn / log(2.0)) / log(2.0);
    let smooth_i = f32(i) + 1.0 - nu;
    let t = smooth_i / 64.0; // color cycle length in iterations

    return vec4<f32>(palette(t), 1.0);
}
