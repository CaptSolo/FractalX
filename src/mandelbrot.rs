//! GPU Mandelbrot renderer, drawn into the egui canvas via a wgpu paint callback.

use eframe::egui_wgpu::{self, wgpu};

/// Shader uniforms. Layout must match `mandelbrot.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniforms {
    pub center: [f32; 2],
    pub half_extent: [f32; 2],
    pub max_iter: u32,
    pub palette_freq: f32,
    pub palette_phase: f32,
    pub _pad: f32,
}

/// Wgpu resources shared across frames, stored in the egui renderer's
/// `callback_resources` type map.
pub struct RenderResources {
    pipeline: wgpu::RenderPipeline,
    /// Same shader targeting Rgba8UnormSrgb, for offscreen PNG export.
    export_pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    uniform_buffer: wgpu::Buffer,
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
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mandelbrot"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/mandelbrot.wgsl").into()),
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mandelbrot uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mandelbrot bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mandelbrot bg"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mandelbrot pl"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = build_pipeline(device, &shader, &pipeline_layout, target_format);
        let export_pipeline = build_pipeline(
            device,
            &shader,
            &pipeline_layout,
            wgpu::TextureFormat::Rgba8UnormSrgb,
        );

        Self {
            pipeline,
            export_pipeline,
            bind_group_layout,
            bind_group,
            uniform_buffer,
        }
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
        // Fresh uniform buffer so the live view's buffer is untouched.
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("export uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buffer, 0, bytemuck::bytes_of(uniforms));
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("export bg"),
            layout: &self.bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

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

        // Buffer rows must be padded to 256 bytes for texture-to-buffer copies.
        let bytes_per_row = (width * 4).next_multiple_of(256);
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("export readback"),
            size: bytes_per_row as u64 * height as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("export pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
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
            pass.set_pipeline(&self.export_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
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

        let resources = RenderResources::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
        let (w, h) = (128u32, 96u32);
        let uniforms = Uniforms {
            center: [-0.5, 0.0],
            half_extent: [1.6, 1.2],
            max_iter: 200,
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
}

/// Per-frame paint callback carrying the uniforms for this view.
pub struct MandelbrotCallback {
    pub uniforms: Uniforms,
}

impl egui_wgpu::CallbackTrait for MandelbrotCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let resources: &RenderResources = callback_resources.get().unwrap();
        queue.write_buffer(
            &resources.uniform_buffer,
            0,
            bytemuck::bytes_of(&self.uniforms),
        );
        Vec::new()
    }

    fn paint(
        &self,
        _info: eframe::egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &egui_wgpu::CallbackResources,
    ) {
        let resources: &RenderResources = callback_resources.get().unwrap();
        render_pass.set_pipeline(&resources.pipeline);
        render_pass.set_bind_group(0, &resources.bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }
}
