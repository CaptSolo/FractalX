//! GPU Mandelbrot renderer, drawn into the egui canvas via a wgpu paint callback.

use eframe::egui_wgpu::wgpu;

/// Shader uniforms. Layout must match `mandelbrot.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniforms {
    pub center: [f32; 2],
    pub half_extent: [f32; 2],
    pub dc_offset: [f32; 2],
    pub max_iter: u32,
    pub ref_len: u32,
    pub use_perturb: u32,
    pub palette_freq: f32,
    pub palette_phase: f32,
    pub _pad: f32,
}

/// Wgpu resources shared across frames, stored in the egui renderer's
/// `callback_resources` type map.
///
/// All rendering goes to offscreen Rgba8UnormSrgb textures: the live view
/// paints a progressively-refined texture (resolution ladder in `main.rs`),
/// and PNG export reads one back.
pub struct RenderResources {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    /// Perturbation reference orbit (vec2<f32> per iteration).
    orbit_buffer: wgpu::Buffer,
}

fn create_orbit_buffer(device: &wgpu::Device, points: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reference orbit"),
        size: (points.max(2) * 8) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn create_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform_buffer: &wgpu::Buffer,
    orbit_buffer: &wgpu::Buffer,
    label: &str,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: orbit_buffer.as_entire_binding(),
            },
        ],
    })
}

fn build_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("mandelbrot pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
            targets: &[Some(format.into())],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

impl RenderResources {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mandelbrot"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/mandelbrot.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mandelbrot bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let orbit_buffer = create_orbit_buffer(device, 2);

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mandelbrot pl"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = build_pipeline(
            device,
            &shader,
            &pipeline_layout,
            wgpu::TextureFormat::Rgba8UnormSrgb,
        );

        Self {
            pipeline,
            bind_group_layout,
            orbit_buffer,
        }
    }

    /// Upload a new reference orbit, growing the GPU buffer if needed.
    pub fn upload_orbit(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        points: &[[f32; 2]],
    ) {
        let needed = (points.len().max(2) * 8) as u64;
        if self.orbit_buffer.size() < needed {
            self.orbit_buffer = create_orbit_buffer(device, points.len());
        }
        queue.write_buffer(&self.orbit_buffer, 0, bytemuck::cast_slice(points));
    }

    /// Render `uniforms` into the given Rgba8UnormSrgb texture view and
    /// submit. Non-blocking; used by both the live resolution ladder and
    /// export.
    pub fn render_to_texture(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        uniforms: &Uniforms,
        target: &wgpu::TextureView,
    ) {
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mandelbrot uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buffer, 0, bytemuck::bytes_of(uniforms));
        let bind_group = create_bind_group(
            device,
            &self.bind_group_layout,
            &uniform_buffer,
            &self.orbit_buffer,
            "mandelbrot bg",
        );

        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("mandelbrot pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        queue.submit([encoder.finish()]);
    }

    /// Render `uniforms` into a `width` x `height` offscreen texture and read
    /// back tightly packed RGBA8 pixels. Blocks until the GPU finishes.
    pub fn render_offscreen(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        uniforms: &Uniforms,
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("export target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        self.render_to_texture(device, queue, uniforms, &view);

        // Buffer rows must be padded to 256 bytes for texture-to-buffer copies.
        let bytes_per_row = (width * 4).next_multiple_of(256);
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("export readback"),
            size: bytes_per_row as u64 * height as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([encoder.finish()]);

        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("map readback buffer"));
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("gpu poll");

        let mapped = slice.get_mapped_range();
        let mut pixels = Vec::with_capacity((width * height * 4) as usize);
        for row in 0..height {
            let start = (row * bytes_per_row) as usize;
            pixels.extend_from_slice(&mapped[start..start + (width * 4) as usize]);
        }
        pixels
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Headless GPU round trip: render offscreen, check the image is a
    /// plausible Mandelbrot (interior black, exterior colored).
    #[test]
    fn offscreen_render_produces_mandelbrot() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&Default::default()))
            .expect("no gpu adapter");
        let (device, queue) =
            pollster::block_on(adapter.request_device(&Default::default())).expect("device");

        let resources = RenderResources::new(&device);
        let (w, h) = (128u32, 96u32);
        let uniforms = Uniforms {
            center: [-0.5, 0.0],
            half_extent: [1.6, 1.2],
            dc_offset: [0.0, 0.0],
            max_iter: 200,
            ref_len: 0,
            use_perturb: 0,
            palette_freq: 1.0,
            palette_phase: 0.0,
            _pad: 0.0,
        };
        let pixels = resources.render_offscreen(&device, &queue, &uniforms, w, h);
        assert_eq!(pixels.len(), (w * h * 4) as usize);

        // Center pixel (-0.5, 0) is inside the set: black.
        let center = ((h / 2 * w + w / 2) * 4) as usize;
        assert_eq!(&pixels[center..center + 3], &[0, 0, 0]);

        // Top-left corner (far outside) escapes immediately: not black.
        assert!(pixels[..3].iter().any(|&c| c > 0));

        // All pixels fully opaque.
        assert!(pixels.chunks_exact(4).all(|p| p[3] == 255));
    }

    /// The perturbation path must reproduce the plain path where both are
    /// valid (shallow zoom).
    #[test]
    fn perturbation_matches_plain_path() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&Default::default()))
            .expect("no gpu adapter");
        let (device, queue) =
            pollster::block_on(adapter.request_device(&Default::default())).expect("device");

        let mut resources = RenderResources::new(&device);
        let (w, h) = (96u32, 96u32);
        let center = crate::deep::BigComplex::from_f64(-0.65, 0.35);
        let max_iter = 300u32;

        let base = Uniforms {
            center: [-0.65, 0.35],
            half_extent: [0.02, 0.02],
            dc_offset: [0.0, 0.0],
            max_iter,
            ref_len: 0,
            use_perturb: 0,
            palette_freq: 1.0,
            palette_phase: 0.0,
            _pad: 0.0,
        };
        let plain = resources.render_offscreen(&device, &queue, &base, w, h);

        let orbit = crate::deep::reference_orbit(&center, max_iter, 128);
        resources.upload_orbit(&device, &queue, &orbit.points);
        let perturbed = resources.render_offscreen(
            &device,
            &queue,
            &Uniforms {
                use_perturb: 1,
                ref_len: orbit.points.len() as u32,
                ..base
            },
            w,
            h,
        );

        // Smooth coloring amplifies tiny float differences right at the set
        // boundary: ~2.6% of pixels differ in practice, all boundary pixels.
        // A systematic defect (e.g. an off-by-one in the delta iteration)
        // would shift most of the image, so 5% separates noise from bugs.
        let differing = plain
            .chunks_exact(4)
            .zip(perturbed.chunks_exact(4))
            .filter(|(a, b)| {
                a.iter()
                    .zip(b.iter())
                    .any(|(x, y)| x.abs_diff(*y) > 8)
            })
            .count();
        let total = (w * h) as usize;
        assert!(
            differing < total / 20,
            "perturbation diverges from plain path: {differing}/{total} pixels differ"
        );
    }

    /// Deep zoom render (1e-14 scale — far beyond f32) must produce visible
    /// structure, not a uniform block.
    #[test]
    fn deep_zoom_renders_structure() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&Default::default()))
            .expect("no gpu adapter");
        let (device, queue) =
            pollster::block_on(adapter.request_device(&Default::default())).expect("device");

        let mut resources = RenderResources::new(&device);
        let (w, h) = (64u32, 64u32);

        // A point on the boundary (Misiurewicz-like), viewed at 1e-14 scale.
        let center = crate::deep::BigComplex::from_decimal(
            "-0.74364388703715870475",
            "0.13182590420531197035",
            160,
        )
        .unwrap();
        let max_iter = 50_000u32;
        let orbit = crate::deep::reference_orbit(&center, max_iter, 160);
        resources.upload_orbit(&device, &queue, &orbit.points);

        let pixels = resources.render_offscreen(
            &device,
            &queue,
            &Uniforms {
                center: [0.0, 0.0], // unused on the perturbation path
                half_extent: [1e-14, 1e-14],
                dc_offset: [0.0, 0.0],
                max_iter,
                ref_len: orbit.points.len() as u32,
                use_perturb: 1,
                palette_freq: 1.0,
                palette_phase: 0.0,
                _pad: 0.0,
            },
            w,
            h,
        );

        let distinct: std::collections::HashSet<[u8; 4]> = pixels
            .chunks_exact(4)
            .map(|p| [p[0], p[1], p[2], p[3]])
            .collect();
        assert!(
            distinct.len() > 16,
            "expected rich structure at deep zoom, got {} distinct colors",
            distinct.len()
        );
    }
}

