use std::sync::Arc;

use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryCommandBufferAbstract,
    RenderPassBeginInfo,
};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{ImageCreateInfo, ImageType, ImageUsage};
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
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

/// Bake world position, world normal, and curvature maps for every visible
/// texture set — the mesh maps that drive smart-mask generators.
pub fn bake_mesh_maps(
    gpu: &GpuContext,
    renderer: &SceneRenderer,
    sets: &mut [TextureSet],
    resolution: u32,
) -> bool {
    let models: Vec<_> = renderer
        .models()
        .iter()
        .filter(|m| sets.get(m.set_index).is_some_and(|s| s.visible))
        .collect();
    if models.is_empty() {
        return false;
    }

    let device = gpu.queue.device().clone();

    let make_image = |format: Format| {
        ImageView::new_default(
            crate::vram::image(
                gpu.memory_allocator.clone(),
                ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format,
                    extent: [resolution, resolution, 1],
                    // TRANSFER_SRC: project save reads the bakes back.
                    usage: ImageUsage::COLOR_ATTACHMENT
                        | ImageUsage::SAMPLED
                        | ImageUsage::STORAGE
                        | ImageUsage::TRANSFER_SRC,
                    ..Default::default()
                },
                crate::vram::Kind::Bake,
            )
            .unwrap(),
        )
        .unwrap()
    };

    let render_pass = vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            position: {
                format: Format::R32G32B32A32_SFLOAT,
                samples: 1,
                load_op: Clear,
                store_op: Store,
            },
            normal: {
                format: Format::R16G16B16A16_SFLOAT,
                samples: 1,
                load_op: Clear,
                store_op: Store,
            },
        },
        pass: {
            color: [position, normal],
            depth_stencil: {},
        },
    )
    .unwrap();

    let pipeline = {
        let vs = bake_vs::load(device.clone()).unwrap().entry_point("main").unwrap();
        let fs = bake_fs::load(device.clone()).unwrap().entry_point("main").unwrap();
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
        let subpass = Subpass::from(render_pass.clone(), 0).unwrap();
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
                    ColorBlendAttachmentState::default(),
                )),
                dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                subpass: Some(subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )
        .unwrap()
    };

    let curvature_pipeline = build_compute(&device, curvature_cs::load(device.clone()).unwrap());

    let mut builder = AutoCommandBufferBuilder::primary(
        gpu.command_buffer_allocator.clone(),
        gpu.queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    let mut results: Vec<(usize, Arc<ImageView>, Arc<ImageView>, Arc<ImageView>)> = Vec::new();
    for set_index in 0..sets.len() {
        if !sets[set_index].visible {
            continue;
        }
        let set_models: Vec<_> = models.iter().filter(|m| m.set_index == set_index).collect();
        if set_models.is_empty() {
            continue;
        }

        let position = make_image(Format::R32G32B32A32_SFLOAT);
        let normal = make_image(Format::R16G16B16A16_SFLOAT);
        let curvature = make_image(Format::R16G16B16A16_SFLOAT);

        let framebuffer = Framebuffer::new(
            render_pass.clone(),
            FramebufferCreateInfo {
                attachments: vec![position.clone(), normal.clone()],
                ..Default::default()
            },
        )
        .unwrap();

        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![
                        Some([0.0, 0.0, 0.0, 0.0].into()),
                        Some([0.0, 0.0, 0.0, 0.0].into()),
                    ],
                    ..RenderPassBeginInfo::framebuffer(framebuffer)
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
            .bind_pipeline_graphics(pipeline.clone())
            .unwrap();
        for model in &set_models {
            builder
                .push_constants(
                    pipeline.layout().clone(),
                    0,
                    bake_vs::Pc { model: model.transform.to_cols_array_2d() },
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

        // Curvature from the freshly baked position/normal maps.
        let ds = DescriptorSet::new(
            gpu.descriptor_set_allocator.clone(),
            curvature_pipeline.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::image_view(0, curvature.clone()),
                WriteDescriptorSet::image_view(1, position.clone()),
                WriteDescriptorSet::image_view(2, normal.clone()),
            ],
            [],
        )
        .unwrap();
        builder
            .bind_pipeline_compute(curvature_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                curvature_pipeline.layout().clone(),
                0,
                ds,
            )
            .unwrap()
            .push_constants(
                curvature_pipeline.layout().clone(),
                0,
                curvature_cs::Pc { size: resolution },
            )
            .unwrap();
        unsafe {
            builder
                .dispatch([resolution / 8 + 1, resolution / 8 + 1, 1])
                .unwrap();
        }

        results.push((set_index, position, normal, curvature));
    }

    if results.is_empty() {
        return false;
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

    for (set_index, position, normal, curvature) in results {
        let set = &mut sets[set_index];
        set.baked_position = Some(position);
        set.baked_normal = Some(normal);
        set.baked_curvature = Some(curvature);
    }
    true
}

/// Generate a smart mask for one texture set from its baked mesh maps.
/// `amount` widens the affected band, `contrast` sharpens the transition.
pub fn generate_mask(
    gpu: &GpuContext,
    set: &TextureSet,
    kind: crate::layers::MaskKind,
    amount: f32,
    contrast: f32,
    resolution: u32,
) -> Option<Arc<ImageView>> {
    let curvature = set.baked_curvature.clone()?;
    let ao = set.baked_ao.clone();

    let device = gpu.queue.device().clone();
    let mask = ImageView::new_default(
        crate::vram::image(
            gpu.memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R16G16B16A16_SFLOAT,
                extent: [resolution, resolution, 1],
                // COLOR_ATTACHMENT and the transfer bits let a generated mask
                // be painted on and inverted afterwards, like a hand-made one.
                usage: ImageUsage::STORAGE
                    | ImageUsage::SAMPLED
                    | ImageUsage::COLOR_ATTACHMENT
                    | ImageUsage::TRANSFER_SRC
                    | ImageUsage::TRANSFER_DST,
                ..Default::default()
            },
            crate::vram::Kind::Layer,
        )
        .unwrap(),
    )
    .unwrap();

    // AO input is optional; a 1x1 white stand-in means "no AO influence".
    let ao_view = ao.unwrap_or_else(|| {
        crate::renderer::upload_texture(
            &gpu.queue,
            &gpu.memory_allocator,
            &gpu.command_buffer_allocator,
            1,
            1,
            &[255, 255, 255, 255],
            Format::R8G8B8A8_UNORM,
        )
    });

    let pipeline = build_compute(&device, mask_cs::load(device.clone()).unwrap());
    let sampler = vulkano::image::sampler::Sampler::new(
        device,
        vulkano::image::sampler::SamplerCreateInfo::simple_repeat_linear_no_mipmap(),
    )
    .unwrap();

    let ds = DescriptorSet::new(
        gpu.descriptor_set_allocator.clone(),
        pipeline.layout().set_layouts()[0].clone(),
        [
            WriteDescriptorSet::image_view(0, mask.clone()),
            WriteDescriptorSet::image_view(1, curvature),
            WriteDescriptorSet::image_view_sampler(2, ao_view, sampler),
        ],
        [],
    )
    .unwrap();

    let mut builder = AutoCommandBufferBuilder::primary(
        gpu.command_buffer_allocator.clone(),
        gpu.queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();
    builder
        .bind_pipeline_compute(pipeline.clone())
        .unwrap()
        .bind_descriptor_sets(PipelineBindPoint::Compute, pipeline.layout().clone(), 0, ds)
        .unwrap()
        .push_constants(
            pipeline.layout().clone(),
            0,
            mask_cs::Pc {
                size: resolution,
                kind: match kind {
                    crate::layers::MaskKind::EdgeWear => 0,
                    crate::layers::MaskKind::CavityDirt => 1,
                },
                amount,
                contrast,
            },
        )
        .unwrap();
    unsafe {
        builder
            .dispatch([resolution / 8 + 1, resolution / 8 + 1, 1])
            .unwrap();
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

    Some(mask)
}

mod mask_cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
#version 450
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 0, rgba16f) uniform writeonly image2D mask_out;
layout(set = 0, binding = 1, rgba16f) uniform readonly image2D curvature_map;
layout(set = 0, binding = 2) uniform sampler2D ao_map;
layout(push_constant) uniform Pc { uint size; uint kind; float amount; float contrast; } pc;

void main() {
    uvec2 id = gl_GlobalInvocationID.xy;
    if (id.x >= pc.size || id.y >= pc.size) return;
    ivec2 p = ivec2(id);
    vec2 uv = (vec2(id) + 0.5) / float(pc.size);

    vec4 c = imageLoad(curvature_map, p);
    float curv = c.x;
    float coverage = c.y;

    float m = 0.0;
    if (pc.kind == 0u) {
        // Edge wear: convex curvature, band widened by amount.
        float threshold = mix(0.15, 0.005, clamp(pc.amount, 0.0, 1.0));
        m = smoothstep(threshold * 0.5, threshold, curv);
    } else {
        // Cavity dirt: concave curvature plus occluded areas.
        float threshold = mix(0.15, 0.005, clamp(pc.amount, 0.0, 1.0));
        float cavity = smoothstep(threshold * 0.5, threshold, -curv);
        float occluded = 1.0 - texture(ao_map, uv).r;
        m = clamp(cavity + occluded * pc.amount, 0.0, 1.0);
    }
    // Contrast around the midpoint.
    m = clamp((m - 0.5) * max(pc.contrast, 0.01) + 0.5, 0.0, 1.0);
    imageStore(mask_out, p, vec4(m * coverage, 1.0, 0.0, 1.0));
}
",
    }
}

mod bake_vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
#version 450
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;
layout(push_constant) uniform Pc { mat4 model; } pc;

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

mod bake_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
#version 450
layout(location = 0) in vec3 v_world_pos;
layout(location = 1) in vec3 v_normal;
layout(location = 0) out vec4 o_position;
layout(location = 1) out vec4 o_normal;

void main() {
    o_position = vec4(v_world_pos, 1.0);
    o_normal = vec4(normalize(v_normal), 1.0);
}
",
    }
}

mod curvature_cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
#version 450
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 0, rgba16f) uniform writeonly image2D curvature_out;
layout(set = 0, binding = 1, rgba32f) uniform readonly image2D position_map;
layout(set = 0, binding = 2, rgba16f) uniform readonly image2D normal_map;
layout(push_constant) uniform Pc { uint size; } pc;

void main() {
    uvec2 id = gl_GlobalInvocationID.xy;
    if (id.x >= pc.size || id.y >= pc.size) return;
    ivec2 p = ivec2(id);

    vec4 pos_c = imageLoad(position_map, p);
    if (pos_c.w < 0.5) {
        imageStore(curvature_out, p, vec4(0.0));
        return;
    }
    vec3 n_c = imageLoad(normal_map, p).xyz;

    // Mean curvature estimate: how much neighbor normals bend toward or away
    // from the center along the surface direction. Positive = convex edge.
    float curv = 0.0;
    float weight = 0.0;
    const ivec2 offsets[8] = ivec2[8](
        ivec2(-1, 0), ivec2(1, 0), ivec2(0, -1), ivec2(0, 1),
        ivec2(-1, -1), ivec2(1, 1), ivec2(-1, 1), ivec2(1, -1)
    );
    for (int i = 0; i < 8; i++) {
        ivec2 q = clamp(p + offsets[i] * 2, ivec2(0), ivec2(int(pc.size) - 1));
        vec4 pos_n = imageLoad(position_map, q);
        if (pos_n.w < 0.5) continue;
        vec3 dp = pos_n.xyz - pos_c.xyz;
        float len = length(dp);
        if (len < 1e-6) continue;
        vec3 n_n = imageLoad(normal_map, q).xyz;
        curv += dot(n_n - n_c, dp / len);
        weight += 1.0;
    }
    curv = weight > 0.0 ? curv / weight : 0.0;
    imageStore(curvature_out, p, vec4(curv, 1.0, 0.0, 1.0));
}
",
    }
}
