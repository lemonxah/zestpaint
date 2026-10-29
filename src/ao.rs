use std::sync::Arc;

use glam::Vec3;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryCommandBufferAbstract,
    RenderPassBeginInfo,
};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::format::Format;
use vulkano::image::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo};
use vulkano::image::view::ImageView;
use vulkano::image::{ImageCreateInfo, ImageType, ImageUsage};
use vulkano::pipeline::graphics::color_blend::{
    AttachmentBlend, BlendFactor, BlendOp, ColorBlendAttachmentState, ColorBlendState,
};
use vulkano::pipeline::graphics::depth_stencil::{DepthState, DepthStencilState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
    PipelineShaderStageCreateInfo,
};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, Subpass};
use vulkano::sync::GpuFuture;

use crate::environment::{build_compute, GpuContext};
use crate::layers::TextureSet;
use crate::mesh::Vertex3D;
use crate::renderer::SceneRenderer;

const SHADOW_RES: u32 = 1024;
const DIRECTION_COUNT: u32 = 64;

/// Bake one ambient-occlusion map per texture set by testing visibility
/// against depth maps rendered from many directions on a sphere (weighted by
/// surface orientation). The whole scene occludes; each set accumulates only
/// its own models' texels. Results land in `set.baked_ao`.
pub fn bake(
    gpu: &GpuContext,
    renderer: &SceneRenderer,
    sets: &mut [TextureSet],
    resolution: u32,
) -> bool {
    // Hidden sub-models neither occlude nor receive AO.
    let models: Vec<_> = renderer
        .models()
        .iter()
        .filter(|m| sets.get(m.set_index).is_some_and(|s| s.visible))
        .collect();
    if models.is_empty() || sets.is_empty() {
        return false;
    }
    let models = models.as_slice();

    // Scene bounding sphere in world space.
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for model in models {
        let (lo, hi) = model.bounds;
        for i in 0..8 {
            let corner = Vec3::new(
                if i & 1 == 0 { lo.x } else { hi.x },
                if i & 2 == 0 { lo.y } else { hi.y },
                if i & 4 == 0 { lo.z } else { hi.z },
            );
            let p = model.transform.transform_point3(corner);
            min = min.min(p);
            max = max.max(p);
        }
    }
    if !min.is_finite() {
        return false;
    }
    let center = (min + max) * 0.5;
    let radius = ((max - min).length() * 0.5).max(1e-4);

    let device = gpu.queue.device().clone();

    // -- resources -----------------------------------------------------------
    let shadow_depth = ImageView::new_default(
        crate::vram::image(
            gpu.memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::D32_SFLOAT,
                extent: [SHADOW_RES, SHADOW_RES, 1],
                usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT | ImageUsage::SAMPLED,
                ..Default::default()
            },
            crate::vram::Kind::Bake,
        )
        .unwrap(),
    )
    .unwrap();

    // Per set: an accumulator of (visibility·weight, weight) and the final AO.
    let accums: Vec<Arc<ImageView>> = (0..sets.len())
        .map(|_| {
            ImageView::new_default(
                crate::vram::image(
                    gpu.memory_allocator.clone(),
                    ImageCreateInfo {
                        image_type: ImageType::Dim2d,
                        format: Format::R16G16_SFLOAT,
                        extent: [resolution, resolution, 1],
                        usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::STORAGE,
                        ..Default::default()
                    },
                    crate::vram::Kind::Bake,
                )
                .unwrap(),
            )
            .unwrap()
        })
        .collect();

    let ao_outs: Vec<Arc<ImageView>> = (0..sets.len())
        .map(|_| {
            ImageView::new_default(
                crate::vram::image(
                    gpu.memory_allocator.clone(),
                    ImageCreateInfo {
                        image_type: ImageType::Dim2d,
                        format: Format::R16G16B16A16_SFLOAT,
                        extent: [resolution, resolution, 1],
                        usage: ImageUsage::STORAGE
                            | ImageUsage::SAMPLED
                            | ImageUsage::TRANSFER_SRC,
                        ..Default::default()
                    },
                    crate::vram::Kind::Bake,
                )
                .unwrap(),
            )
            .unwrap()
        })
        .collect();

    let depth_sampler = Sampler::new(
        device.clone(),
        SamplerCreateInfo {
            mag_filter: Filter::Nearest,
            min_filter: Filter::Nearest,
            address_mode: [SamplerAddressMode::ClampToEdge; 3],
            ..Default::default()
        },
    )
    .unwrap();

    // -- pipelines -----------------------------------------------------------
    let depth_pass = vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            depth: {
                format: Format::D32_SFLOAT,
                samples: 1,
                load_op: Clear,
                store_op: Store,
            },
        },
        pass: {
            color: [],
            depth_stencil: {depth},
        },
    )
    .unwrap();

    let accum_pass = vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            accum: {
                format: Format::R16G16_SFLOAT,
                samples: 1,
                load_op: Load,
                store_op: Store,
            },
        },
        pass: {
            color: [accum],
            depth_stencil: {},
        },
    )
    .unwrap();

    let depth_pipeline = {
        let vs = depth_vs::load(device.clone()).unwrap().entry_point("main").unwrap();
        let vertex_input_state = Vertex3D::per_vertex().definition(&vs).unwrap();
        let stages = [PipelineShaderStageCreateInfo::new(vs)];
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();
        let subpass = Subpass::from(depth_pass.clone(), 0).unwrap();
        GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(vertex_input_state),
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState::default()),
                rasterization_state: Some(RasterizationState::default()),
                multisample_state: Some(MultisampleState::default()),
                depth_stencil_state: Some(DepthStencilState {
                    depth: Some(DepthState::simple()),
                    ..Default::default()
                }),
                dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                subpass: Some(subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )
        .unwrap()
    };

    let accum_pipeline = {
        let vs = accum_vs::load(device.clone()).unwrap().entry_point("main").unwrap();
        let fs = accum_fs::load(device.clone()).unwrap().entry_point("main").unwrap();
        let vertex_input_state = Vertex3D::per_vertex().definition(&vs).unwrap();
        let stages =
            [PipelineShaderStageCreateInfo::new(vs), PipelineShaderStageCreateInfo::new(fs)];
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();
        let subpass = Subpass::from(accum_pass.clone(), 0).unwrap();
        // Additive accumulation.
        let blend = AttachmentBlend {
            src_color_blend_factor: BlendFactor::One,
            dst_color_blend_factor: BlendFactor::One,
            color_blend_op: BlendOp::Add,
            src_alpha_blend_factor: BlendFactor::One,
            dst_alpha_blend_factor: BlendFactor::One,
            alpha_blend_op: BlendOp::Add,
        };
        GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(vertex_input_state),
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState::default()),
                rasterization_state: Some(RasterizationState::default()),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    subpass.num_color_attachments(),
                    ColorBlendAttachmentState {
                        blend: Some(blend),
                        ..Default::default()
                    },
                )),
                dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                subpass: Some(subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )
        .unwrap()
    };

    let finalize = build_compute(&device, finalize_cs::load(device.clone()).unwrap());

    let depth_framebuffer = Framebuffer::new(
        depth_pass.clone(),
        FramebufferCreateInfo {
            attachments: vec![shadow_depth.clone()],
            ..Default::default()
        },
    )
    .unwrap();
    let accum_framebuffers: Vec<Arc<Framebuffer>> = accums
        .iter()
        .map(|accum| {
            Framebuffer::new(
                accum_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![accum.clone()],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect();

    // -- bake ----------------------------------------------------------------
    let mut builder = AutoCommandBufferBuilder::primary(
        gpu.command_buffer_allocator.clone(),
        gpu.queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    for accum in &accums {
        builder
            .clear_color_image(vulkano::command_buffer::ClearColorImageInfo {
                clear_value: vulkano::format::ClearColorValue::Float([0.0; 4]),
                ..vulkano::command_buffer::ClearColorImageInfo::image(accum.image().clone())
            })
            .unwrap();
    }

    let golden_angle = std::f32::consts::PI * (3.0 - 5f32.sqrt());
    for i in 0..DIRECTION_COUNT {
        // Fibonacci sphere direction.
        let y = 1.0 - 2.0 * (i as f32 + 0.5) / DIRECTION_COUNT as f32;
        let r = (1.0 - y * y).max(0.0).sqrt();
        let phi = i as f32 * golden_angle;
        let dir = Vec3::new(r * phi.cos(), y, r * phi.sin()).normalize();

        // Ortho camera looking at the scene from `dir`.
        let up = if dir.y.abs() > 0.99 { Vec3::X } else { Vec3::Y };
        let eye = center + dir * radius * 2.0;
        let view = glam::camera::rh::view::look_at_mat4(eye, center, up);
        let proj = glam::camera::rh::proj::vulkan::orthographic(
            -radius * 1.05,
            radius * 1.05,
            -radius * 1.05,
            radius * 1.05,
            radius * 0.5,
            radius * 3.5,
        );
        let light_vp = proj * view;

        // Depth from this direction.
        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some(1.0f32.into())],
                    ..RenderPassBeginInfo::framebuffer(depth_framebuffer.clone())
                },
                Default::default(),
            )
            .unwrap()
            .set_viewport(
                0,
                [Viewport {
                    offset: [0.0, 0.0],
                    extent: [SHADOW_RES as f32, SHADOW_RES as f32],
                    depth_range: 0.0..=1.0,
                }]
                .into_iter()
                .collect(),
            )
            .unwrap()
            .bind_pipeline_graphics(depth_pipeline.clone())
            .unwrap();
        for model in models {
            builder
                .push_constants(
                    depth_pipeline.layout().clone(),
                    0,
                    depth_vs::Pc {
                        mvp: (light_vp * model.transform).to_cols_array_2d(),
                    },
                )
                .unwrap()
                .bind_vertex_buffers(0, model.vertex_buffer.clone())
                .unwrap()
                .bind_index_buffer(model.index_buffer.clone())
                .unwrap();
            unsafe {
                builder.draw_indexed(model.index_count, 1, 0, 0, 0).unwrap();
            }
        }
        builder.end_render_pass(Default::default()).unwrap();

        // Accumulate visibility in UV space, one pass per texture set so each
        // set's texels land in its own accumulator.
        let ds = DescriptorSet::new(
            gpu.descriptor_set_allocator.clone(),
            accum_pipeline.layout().set_layouts()[0].clone(),
            [WriteDescriptorSet::image_view_sampler(
                0,
                shadow_depth.clone(),
                depth_sampler.clone(),
            )],
            [],
        )
        .unwrap();
        for (set_index, framebuffer) in accum_framebuffers.iter().enumerate() {
            let set_models: Vec<_> =
                models.iter().filter(|m| m.set_index == set_index).collect();
            if set_models.is_empty() {
                continue;
            }
            builder
                .begin_render_pass(
                    RenderPassBeginInfo {
                        clear_values: vec![None],
                        ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                    },
                    Default::default(),
                )
                .unwrap()
                .set_viewport(
                    0,
                    [Viewport {
                        offset: [0.0, 0.0],
                        extent: [resolution as f32, resolution as f32],
                        depth_range: 0.0..=1.0,
                    }]
                    .into_iter()
                    .collect(),
                )
                .unwrap()
                .bind_pipeline_graphics(accum_pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    accum_pipeline.layout().clone(),
                    0,
                    ds.clone(),
                )
                .unwrap();
            for model in &set_models {
                builder
                    .push_constants(
                        accum_pipeline.layout().clone(),
                        0,
                        accum_fs::Pc {
                            light_vp: light_vp.to_cols_array_2d(),
                            model: model.transform.to_cols_array_2d(),
                            light_dir: dir.extend(0.0).to_array(),
                            params: [radius * 0.005, 0.0, 0.0, 0.0],
                        },
                    )
                    .unwrap()
                    .bind_vertex_buffers(0, model.vertex_buffer.clone())
                    .unwrap()
                    .bind_index_buffer(model.index_buffer.clone())
                    .unwrap();
                unsafe {
                    builder.draw_indexed(model.index_count, 1, 0, 0, 0).unwrap();
                }
            }
            builder.end_render_pass(Default::default()).unwrap();
        }
    }

    // Normalize each accumulator into its set's final AO texture.
    builder.bind_pipeline_compute(finalize.clone()).unwrap();
    for (accum, ao_out) in accums.iter().zip(&ao_outs) {
        let ds = DescriptorSet::new(
            gpu.descriptor_set_allocator.clone(),
            finalize.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::image_view(0, ao_out.clone()),
                WriteDescriptorSet::image_view(1, accum.clone()),
            ],
            [],
        )
        .unwrap();
        builder
            .bind_descriptor_sets(PipelineBindPoint::Compute, finalize.layout().clone(), 0, ds)
            .unwrap()
            .push_constants(finalize.layout().clone(), 0, finalize_cs::Pc { size: resolution })
            .unwrap();
        unsafe {
            builder
                .dispatch([resolution / 8 + 1, resolution / 8 + 1, 1])
                .unwrap();
        }
    }

    builder
        .build()
        .unwrap()
        .execute(gpu.queue.clone())
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    for (set, ao_out) in sets.iter_mut().zip(ao_outs) {
        set.baked_ao = Some(ao_out);
    }
    true
}

mod depth_vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
#version 450
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;
layout(push_constant) uniform Pc { mat4 mvp; } pc;

void main() {
    gl_Position = pc.mvp * vec4(position, 1.0);
}
",
    }
}

mod accum_vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
#version 450
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;

layout(push_constant) uniform Pc {
    mat4 light_vp;
    mat4 model;
    vec4 light_dir;
    vec4 params;
} pc;

layout(location = 0) out vec3 v_world_pos;
layout(location = 1) out vec3 v_normal;

void main() {
    vec4 world_pos = pc.model * vec4(position, 1.0);
    v_world_pos = world_pos.xyz;
    v_normal = mat3(pc.model) * normal;
    gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
",
    }
}

mod accum_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
#version 450
layout(location = 0) in vec3 v_world_pos;
layout(location = 1) in vec3 v_normal;
layout(location = 0) out vec2 o_accum;

layout(set = 0, binding = 0) uniform sampler2D shadow_map;
layout(push_constant) uniform Pc {
    mat4 light_vp;
    mat4 model;
    vec4 light_dir;
    vec4 params; // x: world-space bias offset
} pc;

void main() {
    vec3 n = normalize(v_normal);
    float weight = dot(n, pc.light_dir.xyz);
    if (weight <= 0.0) {
        // Direction is behind the surface; contributes nothing.
        o_accum = vec2(0.0);
        return;
    }

    vec3 sample_pos = v_world_pos + n * pc.params.x;
    vec4 clip = pc.light_vp * vec4(sample_pos, 1.0);
    vec3 ndc = clip.xyz / clip.w;
    float visible = 1.0;
    if (abs(ndc.x) <= 1.0 && abs(ndc.y) <= 1.0) {
        vec2 suv = ndc.xy * 0.5 + 0.5;
        float stored = texture(shadow_map, suv).r;
        visible = ndc.z <= stored + 2e-3 ? 1.0 : 0.0;
    }
    o_accum = vec2(visible * weight, weight);
}
",
    }
}

mod finalize_cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
#version 450
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 0, rgba16f) uniform writeonly image2D ao_out;
layout(set = 0, binding = 1, rg16f) uniform readonly image2D accum;
layout(push_constant) uniform Pc { uint size; } pc;

void main() {
    uvec2 id = gl_GlobalInvocationID.xy;
    if (id.x >= pc.size || id.y >= pc.size) return;
    ivec2 p = ivec2(id);
    vec2 a = imageLoad(accum, p).xy;
    float ao = a.y > 1e-4 ? clamp(a.x / a.y, 0.0, 1.0) : 1.0;
    imageStore(ao_out, p, vec4(ao, a.y > 1e-4 ? 1.0 : 0.0, 0.0, 1.0));
}
",
    }
}
