use std::sync::Arc;

use glam::{Mat4, Vec3};
use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferToImageInfo,
    PrimaryCommandBufferAbstract, RenderPassBeginInfo,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::Queue;
use vulkano::format::Format;
use vulkano::image::sampler::{Sampler, SamplerAddressMode, SamplerCreateInfo};
use vulkano::image::view::ImageView;
use vulkano::image::{ImageCreateInfo, ImageType, ImageUsage};
use vulkano::memory::allocator::{
    AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator,
};
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::depth_stencil::{CompareOp, DepthState, DepthStencilState};
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
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::sync::GpuFuture;

use crate::environment::Environment;
use crate::layers::TextureSet;
use crate::mesh::Vertex3D;
use crate::scene::{set_indices, ImageData, MapSlot, ModelData};

const DEPTH_FORMAT: Format = Format::D32_SFLOAT;

/// Material values the UI can force onto every model.
pub struct MaterialOverride {
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
}

pub struct RenderParams {
    pub camera_pos: Vec3,
    pub view_proj: Mat4,
    /// World → view alone (matcaps look up by the view-space normal).
    pub view: Mat4,
    pub exposure: f32,
    pub env_intensity: f32,
    pub key_light: f32,
    /// World-space direction the key light comes *from*. Rotating it is the
    /// quickest way to read normal detail on a surface.
    pub key_light_dir: Vec3,
    pub ao_strength: f32,
    pub uv_checker: bool,
    pub material_override: Option<MaterialOverride>,
    /// Texture sets to outline (`outline_width` thick, in pixels).
    pub selected: Vec<usize>,
    /// Texture sets the pointer is over in the texture-set list: outlined
    /// in a cooler colour.
    pub hovered: Vec<usize>,
    pub outline_width: f32,
    /// Overlay this shader mask on the model, tinted: bright where the mask
    /// is white, dimmed where it is black. Sets without that mask show black.
    pub mask_view: Option<crate::layers::ShaderMask>,
}

pub(crate) struct GpuModel {
    pub(crate) vertex_buffer: Subbuffer<[Vertex3D]>,
    pub(crate) index_buffer: Subbuffer<[u32]>,
    pub(crate) index_count: u32,
    /// Model-space positions and the triangle list, kept on the CPU to sort
    /// transparent triangles by depth.
    positions: Vec<Vec3>,
    indices: Arc<[u32]>,
    /// Material texture per slot: [base color, roughness, metallic, normal, ao].
    /// Base colour, roughness, metallic, normal, AO, emission (map × mask).
    pub(crate) textures: [Arc<ImageView>; 6],
    pub(crate) transform: Mat4,
    pub(crate) bounds: (Vec3, Vec3),
    /// Which texture set this model paints into.
    pub(crate) set_index: usize,
    pub(crate) base_color: [f32; 4],
    pub(crate) metallic: f32,
    pub(crate) roughness: f32,
    /// [coat weight, coat roughness, sheen weight, sheen roughness]
    surface: [f32; 4],
    /// [emission rgb, specular weight]
    pub(crate) emission: [f32; 4],
    pub(crate) factors: [f32; 4],
    /// [material opacity, transparent flag, 0, 0]
    pub(crate) alpha: [f32; 4],
}

/// How an imported map in `slot` is stored: its block encoding, and the
/// RGBA8 format used when the device cannot sample that.
/// `cutout`: the material is not blended, so the shader cuts its base
/// colour's alpha out at 0.5.
fn map_encoding(slot: MapSlot, cutout: bool) -> (crate::texcompress::MapEncoding, Format) {
    use crate::texcompress::MapEncoding;
    match slot {
        MapSlot::BaseColor if cutout => (MapEncoding::Cutout, Format::R8G8B8A8_SRGB),
        MapSlot::BaseColor | MapSlot::Emission => (MapEncoding::Color, Format::R8G8B8A8_SRGB),
        MapSlot::Normal => (MapEncoding::Normal, Format::R8G8B8A8_UNORM),
        _ => (MapEncoding::Scalar, Format::R8G8B8A8_UNORM),
    }
}

/// Index into GpuModel::textures for a material map slot (Height maps are
/// converted to normals before reaching the GPU).
fn slot_texture_index(slot: MapSlot) -> Option<usize> {
    match slot {
        MapSlot::BaseColor => Some(0),
        MapSlot::Roughness => Some(1),
        MapSlot::Metallic => Some(2),
        MapSlot::Normal => Some(3),
        MapSlot::Ao => Some(4),
        // Takes the material's combined map × mask (`emission_texture`).
        MapSlot::Emission => Some(5),
        MapSlot::Height | MapSlot::Opacity | MapSlot::EmissionMask => None,
    }
}

pub struct SceneRenderer {
    queue: Arc<Queue>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    ubo_allocator: SubbufferAllocator,
    render_pass: Arc<RenderPass>,
    pipeline: Arc<GraphicsPipeline>,
    /// Same shaders with alpha blending and no depth writes, for the
    /// transparent pass. It draws triangles in `transparent_order`.
    transparent_pipeline: Arc<GraphicsPipeline>,
    /// The blended pieces' triangles sorted far to near, kept while the
    /// camera and the set of pieces stay the same.
    transparent_order: Option<TransparentOrder>,
    /// Selection outline, as Blender draws it: the selected (and hovered)
    /// pieces are drawn into `select_mask` twice — through everything
    /// (x-ray) and depth-tested (visible) — and `outline_post` traces the
    /// outside of the x-ray silhouette, bright beside visible parts and faint
    /// where the piece is hidden.
    select_pass: Arc<RenderPass>,
    select_xray: Arc<GraphicsPipeline>,
    select_visible: Arc<GraphicsPipeline>,
    outline_post: Arc<vulkano::pipeline::ComputePipeline>,
    select_mask: Option<Arc<ImageView>>,
    skybox_pipeline: Arc<GraphicsPipeline>,
    sampler: Arc<Sampler>,
    white_texture: Arc<ImageView>,
    /// "Mask is zero everywhere" for sets without the overlaid mask.
    black_texture: Arc<ImageView>,
    flat_normal_texture: Arc<ImageView>,
    depth_view: Option<Arc<ImageView>>,
    environment: Environment,
    models: Vec<GpuModel>,
}

impl SceneRenderer {
    pub fn new(
        queue: Arc<Queue>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        output_format: Format,
        environment: Environment,
    ) -> Self {
        let device = queue.device().clone();

        let render_pass = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    format: output_format,
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },
                depth: {
                    format: DEPTH_FORMAT,
                    samples: 1,
                    load_op: Clear,
                    store_op: Store, // sampled later for paint occlusion
                },
            },
            pass: {
                color: [color],
                depth_stencil: {depth},
            },
        )
        .unwrap();

        let pipeline = {
            let vs = vs::load(device.clone()).unwrap().entry_point("main").unwrap();
            let fs = fs::load(device.clone()).unwrap().entry_point("main").unwrap();
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


        // Selection mask: colour = which pieces cover a pixel (R visible
        // selected, G any selected, B visible hovered, A any hovered),
        // combined with Max blending; depth is the scene's, only read.
        let select_pass = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                mask: {
                    format: Format::R8G8B8A8_UNORM,
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },
                depth: {
                    format: DEPTH_FORMAT,
                    samples: 1,
                    load_op: Load,
                    store_op: Store, // still sampled for paint occlusion
                },
            },
            pass: {
                color: [mask],
                depth_stencil: {depth},
            },
        )
        .unwrap();
        let select_pipeline = |depth: Option<DepthState>| {
            let vs = select_vs::load(device.clone()).unwrap().entry_point("main").unwrap();
            let fs = select_fs::load(device.clone()).unwrap().entry_point("main").unwrap();
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
            let subpass = Subpass::from(select_pass.clone(), 0).unwrap();
            let max = vulkano::pipeline::graphics::color_blend::AttachmentBlend {
                src_color_blend_factor: vulkano::pipeline::graphics::color_blend::BlendFactor::One,
                dst_color_blend_factor: vulkano::pipeline::graphics::color_blend::BlendFactor::One,
                color_blend_op: vulkano::pipeline::graphics::color_blend::BlendOp::Max,
                src_alpha_blend_factor: vulkano::pipeline::graphics::color_blend::BlendFactor::One,
                dst_alpha_blend_factor: vulkano::pipeline::graphics::color_blend::BlendFactor::One,
                alpha_blend_op: vulkano::pipeline::graphics::color_blend::BlendOp::Max,
            };
            GraphicsPipeline::new(
                device.clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages: stages.into_iter().collect(),
                    vertex_input_state: Some(vertex_input_state),
                    input_assembly_state: Some(InputAssemblyState::default()),
                    viewport_state: Some(ViewportState::default()),
                    // The visible pass redraws surfaces already in the depth
                    // buffer; a nudge toward the camera keeps them passing.
                    rasterization_state: Some(RasterizationState {
                        depth_bias: depth.map(|_| vulkano::pipeline::graphics::rasterization::DepthBiasState {
                            constant_factor: -1.0,
                            clamp: 0.0,
                            slope_factor: -1.0,
                        }),
                        ..Default::default()
                    }),
                    multisample_state: Some(MultisampleState::default()),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        subpass.num_color_attachments(),
                        ColorBlendAttachmentState { blend: Some(max), ..Default::default() },
                    )),
                    depth_stencil_state: Some(DepthStencilState { depth, ..Default::default() }),
                    dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                    subpass: Some(subpass.into()),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                },
            )
            .unwrap()
        };
        let select_xray = select_pipeline(None);
        let select_visible =
            select_pipeline(Some(DepthState { write_enable: false, compare_op: CompareOp::LessOrEqual }));
        let outline_post = crate::environment::build_compute(&device, outline_cs::load(device.clone()).unwrap());

        let transparent_pipeline = {
            let vs = vs::load(device.clone()).unwrap().entry_point("main").unwrap();
            let fs = fs::load(device.clone()).unwrap().entry_point("main").unwrap();
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
            let blend = vulkano::pipeline::graphics::color_blend::AttachmentBlend {
                src_color_blend_factor:
                    vulkano::pipeline::graphics::color_blend::BlendFactor::SrcAlpha,
                dst_color_blend_factor:
                    vulkano::pipeline::graphics::color_blend::BlendFactor::OneMinusSrcAlpha,
                color_blend_op: vulkano::pipeline::graphics::color_blend::BlendOp::Add,
                src_alpha_blend_factor:
                    vulkano::pipeline::graphics::color_blend::BlendFactor::One,
                dst_alpha_blend_factor:
                    vulkano::pipeline::graphics::color_blend::BlendFactor::OneMinusSrcAlpha,
                alpha_blend_op: vulkano::pipeline::graphics::color_blend::BlendOp::Add,
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
                    // Test against opaque depth but don't write, so stacked
                    // transparent surfaces all draw (sorted back-to-front).
                    depth_stencil_state: Some(DepthStencilState {
                        depth: Some(DepthState {
                            write_enable: false,
                            compare_op: CompareOp::Less,
                        }),
                        ..Default::default()
                    }),
                    dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                    subpass: Some(subpass.into()),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                },
            )
            .unwrap()
        };

        let skybox_pipeline = {
            let vs = sky_vs::load(device.clone()).unwrap().entry_point("main").unwrap();
            let fs = sky_fs::load(device.clone()).unwrap().entry_point("main").unwrap();
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
                    // Fullscreen triangle generated from gl_VertexIndex; no vertex input.
                    vertex_input_state: Some(Default::default()),
                    input_assembly_state: Some(InputAssemblyState::default()),
                    viewport_state: Some(ViewportState::default()),
                    rasterization_state: Some(RasterizationState::default()),
                    multisample_state: Some(MultisampleState::default()),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        subpass.num_color_attachments(),
                        ColorBlendAttachmentState::default(),
                    )),
                    // Drawn at depth 1.0 after the models; passes only where nothing rendered.
                    depth_stencil_state: Some(DepthStencilState {
                        depth: Some(DepthState {
                            write_enable: false,
                            compare_op: CompareOp::LessOrEqual,
                        }),
                        ..Default::default()
                    }),
                    dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                    subpass: Some(subpass.into()),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                },
            )
            .unwrap()
        };

        let sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: vulkano::image::sampler::Filter::Linear,
                min_filter: vulkano::image::sampler::Filter::Linear,
                // Imported maps carry mip chains; single-level images ignore this.
                mipmap_mode: vulkano::image::sampler::SamplerMipmapMode::Linear,
                lod: 0.0..=vulkano::image::sampler::LOD_CLAMP_NONE,
                address_mode: [SamplerAddressMode::Repeat; 3],
                ..Default::default()
            },
        )
        .unwrap();

        let ubo_allocator = SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        let white_texture = upload_texture(
            &queue,
            &memory_allocator,
            &command_buffer_allocator,
            1,
            1,
            &[255, 255, 255, 255],
            Format::R8G8B8A8_SRGB,
        );
        let black_texture = upload_texture(
            &queue,
            &memory_allocator,
            &command_buffer_allocator,
            1,
            1,
            &[0, 0, 0, 255],
            Format::R8G8B8A8_UNORM,
        );
        let flat_normal_texture = upload_texture(
            &queue,
            &memory_allocator,
            &command_buffer_allocator,
            1,
            1,
            &[128, 128, 255, 255],
            Format::R8G8B8A8_UNORM,
        );
        Self {
            queue,
            memory_allocator,
            command_buffer_allocator,
            descriptor_set_allocator,
            ubo_allocator,
            render_pass,
            pipeline,
            transparent_pipeline,
            transparent_order: None,
            select_pass,
            select_xray,
            select_visible,
            outline_post,
            select_mask: None,
            skybox_pipeline,
            sampler,
            white_texture,
            black_texture,
            flat_normal_texture,
            depth_view: None,
            environment,
            models: Vec::new(),
        }
    }

    pub fn set_environment(&mut self, environment: Environment) {
        self.environment = environment;
    }

    pub(crate) fn environment(&self) -> &Environment {
        &self.environment
    }

    pub(crate) fn flat_normal_texture(&self) -> Arc<ImageView> {
        self.flat_normal_texture.clone()
    }

    pub(crate) fn white_texture(&self) -> Arc<ImageView> {
        self.white_texture.clone()
    }

    /// Push updated OpenPBR-style surface parameters for one texture set.
    pub fn set_material_surface(&mut self, set_index: usize, p: crate::scene::SurfaceParams) {
        for model in self.models.iter_mut().filter(|m| m.set_index == set_index) {
            model.surface =
                [p.coat_weight, p.coat_roughness, p.sheen_weight, p.sheen_roughness];
            model.emission = [p.emission[0], p.emission[1], p.emission[2], p.specular_weight];
            model.alpha = [p.opacity, if p.transparent { 1.0 } else { 0.0 }, 0.0, 0.0];
        }
    }

    /// Swap one material map of every model in a texture set (`None`
    /// restores the neutral fallback). Height is not a GPU slot — callers
    /// convert it to a normal map first.
    pub fn set_material_map(
        &mut self,
        set_index: usize,
        slot: MapSlot,
        image: Option<&ImageData>,
    ) {
        let Some(tex_index) = slot_texture_index(slot) else { return };
        let fallback = if slot == MapSlot::Normal {
            self.flat_normal_texture.clone()
        } else {
            self.white_texture.clone()
        };
        let blended = self.models.iter().any(|m| m.set_index == set_index && m.alpha[1] > 0.5);
        let (encoding, format) = map_encoding(slot, !blended);
        let texture = match image {
            Some(img) => upload_material_map(
                &self.queue,
                &self.memory_allocator,
                &self.command_buffer_allocator,
                img,
                encoding,
                format,
            ),
            None => fallback,
        };
        for model in self.models.iter_mut().filter(|m| m.set_index == set_index) {
            model.textures[tex_index] = texture.clone();
            if image.is_some() {
                // A texture takes over from the scalar factor.
                match slot {
                    MapSlot::BaseColor => model.base_color = [1.0; 4],
                    MapSlot::Roughness => model.roughness = 1.0,
                    MapSlot::Metallic => model.metallic = 1.0,
                    _ => {}
                }
            }
        }
    }

    /// Depth of the last rendered frame (camera space), for paint occlusion.
    pub(crate) fn depth_view(&self) -> Option<Arc<ImageView>> {
        self.depth_view.clone()
    }

    pub(crate) fn models(&self) -> &[GpuModel] {
        &self.models
    }

    /// Replace the scene contents, uploading vertex/index/texture data to the GPU.
    /// Replace one model's vertices (same count and order) — a blendshape
    /// change. Frames in flight keep the old buffer alive.
    pub fn update_vertices(&mut self, index: usize, vertices: &[Vertex3D]) {
        let Some(model) = self.models.get_mut(index) else { return };
        model.vertex_buffer = Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo { usage: BufferUsage::VERTEX_BUFFER, ..Default::default() },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertices.iter().copied(),
        )
        .unwrap();
        let mesh = crate::mesh::MeshData { vertices: vertices.to_vec(), indices: Vec::new() };
        model.bounds = mesh.bounds();
        model.positions = vertices.iter().map(|v| Vec3::from(v.position)).collect();
        self.transparent_order = None;
    }

    pub fn set_models(&mut self, models: &[ModelData]) {
        self.transparent_order = None;
        let indices = set_indices(models);
        // Parts of one object share their material's images (see
        // `ImageData::rgba`): upload each once, not once per part.
        let mut uploaded: std::collections::HashMap<(usize, crate::texcompress::MapEncoding), Arc<ImageView>> =
            Default::default();
        self.models = models
            .iter()
            .zip(indices)
            .map(|(model, set_index)| {
                let vertex_buffer = Buffer::from_iter(
                    self.memory_allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::VERTEX_BUFFER,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                            | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                    model.mesh.vertices.iter().copied(),
                )
                .unwrap();
                let index_buffer = Buffer::from_iter(
                    self.memory_allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::INDEX_BUFFER,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                            | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                    model.mesh.indices.iter().copied(),
                )
                .unwrap();

                let mut upload = |img: &Option<ImageData>, slot: MapSlot, fallback: &Arc<ImageView>| {
                    let (encoding, format) = map_encoding(slot, !model.material.transparent);
                    match img {
                        Some(img) => uploaded
                            .entry((Arc::as_ptr(&img.rgba) as usize, encoding))
                            .or_insert_with(|| {
                                upload_material_map(
                                    &self.queue,
                                    &self.memory_allocator,
                                    &self.command_buffer_allocator,
                                    img,
                                    encoding,
                                    format,
                                )
                            })
                            .clone(),
                        None => fallback.clone(),
                    }
                };
                let mat = &model.material;
                let textures = [
                    upload(&mat.base_color_image, MapSlot::BaseColor, &self.white_texture),
                    upload(&mat.roughness_image, MapSlot::Roughness, &self.white_texture),
                    upload(&mat.metallic_image, MapSlot::Metallic, &self.white_texture),
                    upload(&mat.effective_normal_image(), MapSlot::Normal, &self.flat_normal_texture),
                    upload(&mat.ao_image, MapSlot::Ao, &self.white_texture),
                    upload(&mat.emission_texture(), MapSlot::Emission, &self.white_texture),
                ];

                GpuModel {
                    index_count: model.mesh.indices.len() as u32,
                    positions: model.mesh.vertices.iter().map(|v| Vec3::from(v.position)).collect(),
                    indices: model.mesh.indices.as_slice().into(),
                    vertex_buffer,
                    index_buffer,
                    textures,
                    transform: model.transform,
                    bounds: model.mesh.bounds(),
                    set_index,
                    base_color: model.material.base_color,
                    metallic: model.material.metallic,
                    roughness: model.material.roughness,
                    surface: [
                        mat.coat_weight,
                        mat.coat_roughness,
                        mat.sheen_weight,
                        mat.sheen_roughness,
                    ],
                    emission: [
                        mat.emission[0],
                        mat.emission[1],
                        mat.emission[2],
                        mat.specular_weight,
                    ],
                    factors: [
                        mat.base_color[0],
                        mat.base_color[1],
                        mat.base_color[2],
                        0.0,
                    ],
                    alpha: [
                        mat.opacity,
                        if mat.transparent { 1.0 } else { 0.0 },
                        0.0,
                        0.0,
                    ],
                }
            })
            .collect();
    }

    /// Draw the scene into `target`, chained after `before_future`.
    /// Record one model's draw with the currently bound pipeline.
    fn record_model_draw(
        &self,
        builder: &mut AutoCommandBufferBuilder<vulkano::command_buffer::PrimaryAutoCommandBuffer>,
        model: &GpuModel,
        set: &TextureSet,
        params: &RenderParams,
        coverage: Coverage,
    ) {
        let ds = self.model_descriptor_set(model, set, params, coverage);
        builder
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                ds,
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

    /// The scene shader's inputs for one model: its uniforms, material maps,
    /// the set's composites and the environment.
    fn model_descriptor_set(
        &self,
        model: &GpuModel,
        set: &TextureSet,
        params: &RenderParams,
        coverage: Coverage,
    ) -> Arc<DescriptorSet> {
        let layers = &set.layers;
        let matcap_layers = layers.matcap_layers();
        let ao_map = set.baked_ao.clone().unwrap_or_else(|| self.white_texture.clone());
        // Fallback maps are white, so the slider is harmless with no bake.
        let ao_strength = params.ao_strength;
        let (base_color, metallic, roughness) = match &params.material_override {
            Some(o) => (o.base_color, o.metallic, o.roughness),
            None => (model.base_color, model.metallic, model.roughness),
        };
        let ubo = vs::Ubo {
            mvp: (params.view_proj * model.transform).to_cols_array_2d(),
            model: model.transform.to_cols_array_2d(),
            camera_pos: params.camera_pos.extend(1.0).to_array(),
            base_color,
            params: [metallic, roughness, params.exposure, params.env_intensity],
            params2: [
                self.environment.prefiltered_mips as f32,
                params.key_light,
                ao_strength,
                if params.uv_checker { 1.0 } else { 0.0 },
            ],
            key_dir: params.key_light_dir.normalize_or_zero().extend(0.0).to_array(),
            surface: model.surface,
            emission: model.emission,
            // z: the material's textures live in the layers (see
            // `TextureSet::material_drawn_by_layers`); its alpha still cuts out.
            alpha: [model.alpha[0], coverage as u32 as f32, if set.material_drawn_by_layers() { 1.0 } else { 0.0 }, model.alpha[3]],
            mask_view: match params.mask_view {
                Some(kind) => {
                    let t = kind.tint().map(|c| c as f32 / 255.0);
                    [t[0], t[1], t[2], 1.0]
                }
                None => [0.0; 4],
            },
            view: params.view.to_cols_array_2d(),
            matcaps: std::array::from_fn(|i| matcap_layers.get(i).map_or([0.0; 4], |(m, _)| matcap_params(m))),
        };
        // Matcap layers, bottom to top; unused slots are off (and black).
        let mut matcap_tex = Vec::with_capacity(MAX_MATCAPS);
        let mut matcap_mask = Vec::with_capacity(MAX_MATCAPS);
        for i in 0..MAX_MATCAPS {
            let (tex, mask) = match matcap_layers.get(i) {
                Some((m, mask)) => (m.image.as_ref().map(|(_, view)| view.clone()).unwrap(), (*mask).clone()),
                None => (self.black_texture.clone(), self.black_texture.clone()),
            };
            matcap_tex.push((tex, self.sampler.clone()));
            matcap_mask.push((mask, layers.sampler.clone()));
        }
        let mask_preview = params
            .mask_view
            .and_then(|kind| layers.shader_mask(kind).cloned())
            .unwrap_or_else(|| self.black_texture.clone());
        let ubo_buffer = self.ubo_allocator.allocate_sized::<vs::Ubo>().unwrap();
        *ubo_buffer.write().unwrap() = ubo;

        let layout = self.pipeline.layout().set_layouts()[0].clone();
        DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            layout,
            [
                WriteDescriptorSet::buffer(0, ubo_buffer),
                WriteDescriptorSet::image_view_sampler(
                1,
                model.textures[0].clone(),
                self.sampler.clone(),
                ),
                WriteDescriptorSet::image_view_sampler(
                2,
                self.environment.irradiance_cube.clone(),
                self.environment.sampler.clone(),
                ),
                WriteDescriptorSet::image_view_sampler(
                3,
                self.environment.prefiltered_cube.clone(),
                self.environment.sampler.clone(),
                ),
                WriteDescriptorSet::image_view_sampler(
                4,
                layers.composite_basecolor.clone(),
                layers.sampler.clone(),
                ),
                WriteDescriptorSet::image_view_sampler(
                5,
                layers.composite_roughness.clone(),
                layers.sampler.clone(),
                ),
                WriteDescriptorSet::image_view_sampler(
                6,
                layers.composite_metallic.clone(),
                layers.sampler.clone(),
                ),
                WriteDescriptorSet::image_view_sampler(
                7,
                layers.normal_map.clone(),
                layers.sampler.clone(),
                ),
                WriteDescriptorSet::image_view_sampler(8, ao_map, self.sampler.clone()),
                WriteDescriptorSet::image_view_sampler(
                9,
                model.textures[1].clone(),
                self.sampler.clone(),
                ),
                WriteDescriptorSet::image_view_sampler(
                10,
                model.textures[2].clone(),
                self.sampler.clone(),
                ),
                WriteDescriptorSet::image_view_sampler(
                11,
                model.textures[3].clone(),
                self.sampler.clone(),
                ),
                WriteDescriptorSet::image_view_sampler(
                12,
                model.textures[4].clone(),
                self.sampler.clone(),
                ),
                WriteDescriptorSet::image_view_sampler(
                13,
                layers.composite_emissive.clone(),
                layers.sampler.clone(),
                ),
                WriteDescriptorSet::image_view_sampler(
                14,
                layers.composite_opacity.clone(),
                layers.sampler.clone(),
                ),
                WriteDescriptorSet::image_view_sampler(15, mask_preview, layers.sampler.clone()),
                WriteDescriptorSet::image_view_sampler_array(16, 0, matcap_tex),
                WriteDescriptorSet::image_view_sampler_array(17, 0, matcap_mask),
                WriteDescriptorSet::image_view_sampler(18, model.textures[5].clone(), self.sampler.clone()),
            ],
            [],
        )
        .unwrap()
    }

    /// Draw the selected and hovered pieces into the selection mask, then
    /// trace their outlines onto `target`. Runs after the scene pass, on its
    /// depth buffer.
    fn record_selection_outline(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<vulkano::command_buffer::PrimaryAutoCommandBuffer>,
        target: Arc<ImageView>,
        params: &RenderParams,
        sets: &[TextureSet],
    ) {
        let extent = target.image().extent();
        if self.select_mask.as_ref().is_none_or(|m| m.image().extent() != extent) {
            let image = crate::vram::image(
                self.memory_allocator.clone(),
                ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format: Format::R8G8B8A8_UNORM,
                    extent,
                    usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::STORAGE,
                    ..Default::default()
                },
                crate::vram::Kind::Viewport,
            )
            .unwrap();
            self.select_mask = Some(ImageView::new_default(image).unwrap());
        }
        let mask = self.select_mask.clone().unwrap();
        let framebuffer = Framebuffer::new(
            self.select_pass.clone(),
            FramebufferCreateInfo { attachments: vec![mask.clone(), self.depth_view.clone().unwrap()], ..Default::default() },
        )
        .unwrap();
        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some([0.0f32; 4].into()), None],
                    ..RenderPassBeginInfo::framebuffer(framebuffer)
                },
                Default::default(),
            )
            .unwrap()
            .set_viewport(
                0,
                [Viewport { offset: [0.0, 0.0], extent: [extent[0] as f32, extent[1] as f32], depth_range: 0.0..=1.0 }]
                    .into_iter()
                    .collect(),
            )
            .unwrap();
        for (pipeline, visible) in [(self.select_xray.clone(), false), (self.select_visible.clone(), true)] {
            builder.bind_pipeline_graphics(pipeline.clone()).unwrap();
            for model in &self.models {
                let selected = params.selected.contains(&model.set_index);
                let hovered = params.hovered.contains(&model.set_index);
                let Some(set) = sets.get(model.set_index).filter(|s| s.visible) else { continue };
                if !(selected || hovered) {
                    continue;
                }
                let on = |b: bool| if b { 1.0 } else { 0.0 };
                let channel = if visible {
                    [on(selected), 0.0, on(hovered), 0.0]
                } else {
                    [0.0, on(selected), 0.0, on(hovered)]
                };
                // Coverage as the scene pass computes it, so hair cards and
                // lace outline their strands, not their quads.
                let ds = DescriptorSet::new(
                    self.descriptor_set_allocator.clone(),
                    pipeline.layout().set_layouts()[0].clone(),
                    [
                        WriteDescriptorSet::image_view_sampler(0, model.textures[0].clone(), self.sampler.clone()),
                        WriteDescriptorSet::image_view_sampler(1, set.layers.composite_opacity.clone(), set.layers.sampler.clone()),
                    ],
                    [],
                )
                .unwrap();
                builder
                    .bind_descriptor_sets(PipelineBindPoint::Graphics, pipeline.layout().clone(), 0, ds)
                    .unwrap()
                    .push_constants(
                        pipeline.layout().clone(),
                        0,
                        select_vs::Pc {
                            mvp: (params.view_proj * model.transform).to_cols_array_2d(),
                            channel,
                            alpha: [model.alpha[0], 0.0, 0.0, 0.0],
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
        }
        builder.end_render_pass(Default::default()).unwrap();

        let ds = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            self.outline_post.layout().set_layouts()[0].clone(),
            [WriteDescriptorSet::image_view(0, target), WriteDescriptorSet::image_view(1, mask)],
            [],
        )
        .unwrap();
        builder
            .bind_pipeline_compute(self.outline_post.clone())
            .unwrap()
            .bind_descriptor_sets(PipelineBindPoint::Compute, self.outline_post.layout().clone(), 0, ds)
            .unwrap()
            .push_constants(
                self.outline_post.layout().clone(),
                0,
                outline_cs::Pc { size: [extent[0], extent[1], 0, 0], params: [params.outline_width, 0.0, 0.0, 0.0] },
            )
            .unwrap();
        unsafe {
            builder.dispatch([extent[0].div_ceil(8), extent[1].div_ceil(8), 1]).unwrap();
        }
    }

    pub fn render(
        &mut self,
        before_future: Box<dyn GpuFuture>,
        target: Arc<ImageView>,
        params: &RenderParams,
        sets: &mut [TextureSet],
    ) -> Box<dyn GpuFuture> {
        let extent = target.image().extent();

        let depth_stale =
            self.depth_view.as_ref().is_none_or(|d| d.image().extent() != extent);
        if depth_stale {
            self.depth_view = Some(
                ImageView::new_default(
                    crate::vram::image(
                        self.memory_allocator.clone(),
                        ImageCreateInfo {
                            image_type: ImageType::Dim2d,
                            format: DEPTH_FORMAT,
                            extent,
                            usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT | ImageUsage::SAMPLED,
                            ..Default::default()
                        },
                        crate::vram::Kind::Viewport,
                    )
                    .unwrap(),
                )
                .unwrap(),
            );
        }

        let framebuffer = Framebuffer::new(
            self.render_pass.clone(),
            FramebufferCreateInfo {
                attachments: vec![target.clone(), self.depth_view.clone().unwrap()],
                ..Default::default()
            },
        )
        .unwrap();

        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        // Flatten every dirty layer stack first so the scene pass samples
        // fresh composites. Hidden ones wait (dirty) until shown: their
        // composites are released (`LayerStack::set_resident`).
        for set in sets.iter_mut().filter(|s| s.visible) {
            set.layers.record_composite(&mut builder);
        }

        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![
                        Some([0.13, 0.13, 0.15, 1.0].into()),
                        Some(1.0f32.into()),
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
                    extent: [extent[0] as f32, extent[1] as f32],
                    depth_range: 0.0..=1.0,
                }]
                .into_iter()
                .collect(),
            )
            .unwrap()
            .bind_pipeline_graphics(self.pipeline.clone())
            .unwrap();

        // Opaque pass (front-to-back order is irrelevant with depth testing).
        // Blended materials draw their solid texels here too, so those
        // occlude properly; only the see-through rest is left to the sorted
        // pass below.
        for model in &self.models {
            let Some(set) = sets.get(model.set_index) else { continue };
            if !set.visible {
                continue;
            }
            let coverage = if model.alpha[1] > 0.5 { Coverage::SolidPart } else { Coverage::Cutout };
            self.record_model_draw(&mut builder, model, set, params, coverage);
        }
        // Skybox last: fails the depth test wherever geometry already rendered.
        let sky_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            self.skybox_pipeline.layout().set_layouts()[0].clone(),
            [WriteDescriptorSet::image_view_sampler(
                0,
                self.environment.env_cube.clone(),
                self.environment.sampler.clone(),
            )],
            [],
        )
        .unwrap();
        builder
            .bind_pipeline_graphics(self.skybox_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.skybox_pipeline.layout().clone(),
                0,
                sky_set,
            )
            .unwrap()
            .push_constants(
                self.skybox_pipeline.layout().clone(),
                0,
                sky_fs::Pc {
                    inv_view_proj: params.view_proj.inverse().to_cols_array_2d(),
                    camera_pos: params.camera_pos.extend(1.0).to_array(),
                    params: [params.exposure, params.env_intensity, 0.0, 0.0],
                },
            )
            .unwrap();
        unsafe {
            builder.draw(3, 1, 0, 0).unwrap();
        }

        // Transparent pass: after the skybox so glass/sheer blends over it,
        // depth-tested against opaques but not written, triangle by
        // triangle far to near (see `TransparentOrder`).
        let transparent: Vec<usize> = (0..self.models.len())
            .filter(|&i| {
                let m = &self.models[i];
                m.alpha[1] > 0.5 && m.index_count > 0 && sets.get(m.set_index).is_some_and(|s| s.visible)
            })
            .collect();
        if !transparent.is_empty() {
            let order = match self.transparent_order.take() {
                Some(o) if o.camera == params.camera_pos && o.models == transparent => o,
                _ => TransparentOrder::sort(&self.models, params.camera_pos, transparent, &self.memory_allocator),
            };
            builder
                .bind_pipeline_graphics(self.transparent_pipeline.clone())
                .unwrap()
                .bind_index_buffer(order.indices.clone())
                .unwrap();
            let mut descriptor_sets: std::collections::HashMap<usize, Arc<DescriptorSet>> = Default::default();
            for &(m, first, count) in &order.runs {
                let model = &self.models[m];
                let ds = descriptor_sets
                    .entry(m)
                    .or_insert_with(|| self.model_descriptor_set(model, &sets[model.set_index], params, Coverage::SeeThrough))
                    .clone();
                builder
                    .bind_descriptor_sets(PipelineBindPoint::Graphics, self.pipeline.layout().clone(), 0, ds)
                    .unwrap()
                    .bind_vertex_buffers(0, model.vertex_buffer.clone())
                    .unwrap();
                unsafe {
                    builder.draw_indexed(count, 1, first, 0, 0).unwrap();
                }
            }
            self.transparent_order = Some(order);
        }

        builder.end_render_pass(Default::default()).unwrap();
        if !params.selected.is_empty() || !params.hovered.is_empty() {
            self.record_selection_outline(&mut builder, target.clone(), params, sets);
        }
        let command_buffer = builder.build().unwrap();

        before_future
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .boxed()
    }
}

/// Which texels of a material a draw keeps (the shader's `ubo.alpha.y`).
#[derive(Clone, Copy)]
enum Coverage {
    /// Not blended: alpha below 0.5 is cut out, the rest is opaque.
    Cutout = 0,
    /// A blended material's see-through texels, alpha-blended in the
    /// sorted pass.
    SeeThrough = 1,
    /// A blended material's solid texels (alpha ~1), drawn with the opaques
    /// and writing depth. Sorting by triangle centre fails between big and
    /// small triangles (a waistband over its belt loop), and a mostly solid
    /// garment marked transparent showed it everywhere.
    SolidPart = 2,
}

/// Transparent surfaces don't write depth, so they cover each other in the
/// order they are drawn. Drawing whole pieces far to near still left a
/// piece's own triangles in index order (its inside painted over its
/// outside) and overlapping sheer pieces in the wrong order whenever their
/// centres misled the sort. So every visible blended piece's triangles are
/// sorted together by distance, far first, and drawn as runs of consecutive
/// triangles from the same piece.
struct TransparentOrder {
    camera: Vec3,
    /// The models sorted, by index.
    models: Vec<usize>,
    /// Their triangles' vertex indices (each into its own model's vertex
    /// buffer), far to near.
    indices: Subbuffer<[u32]>,
    /// (model, first index, index count)
    runs: Vec<(usize, u32, u32)>,
}

/// Past this many runs (pieces interleaving triangle by triangle) the draw
/// count costs more than the order is worth: pieces are drawn whole, far to
/// near, each still sorted inside.
const MAX_TRANSPARENT_RUNS: usize = 4096;

impl TransparentOrder {
    fn sort(all: &[GpuModel], camera: Vec3, models: Vec<usize>, allocator: &Arc<StandardMemoryAllocator>) -> Self {
        // (distance key, slot in `models`, triangle). Squared distances are
        // non-negative, so their bits order like the floats; inverted, the
        // farthest sorts first.
        let mut tris: Vec<(u32, u32, u32)> = Vec::new();
        let mut piece_distance = Vec::with_capacity(models.len());
        for (slot, &m) in models.iter().enumerate() {
            let model = &all[m];
            let world: Vec<Vec3> = model.positions.iter().map(|&p| model.transform.transform_point3(p)).collect();
            tris.extend(model.indices.chunks_exact(3).enumerate().map(|(t, tri)| {
                let centre = (world[tri[0] as usize] + world[tri[1] as usize] + world[tri[2] as usize]) / 3.0;
                (!camera.distance_squared(centre).to_bits(), slot as u32, t as u32)
            }));
            let centre = model.transform.transform_point3((model.bounds.0 + model.bounds.1) * 0.5);
            piece_distance.push(!camera.distance_squared(centre).to_bits());
        }
        tris.sort_unstable();
        let mut order = Self::build(all, &models, &tris);
        if order.1.len() > MAX_TRANSPARENT_RUNS {
            tris.sort_by_key(|&(_, slot, _)| piece_distance[slot as usize]);
            order = Self::build(all, &models, &tris);
        }
        let (indices, runs) = order;
        let indices = Buffer::from_iter(
            allocator.clone(),
            BufferCreateInfo { usage: BufferUsage::INDEX_BUFFER, ..Default::default() },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            indices,
        )
        .unwrap();
        TransparentOrder { camera, models, indices, runs }
    }

    fn build(all: &[GpuModel], models: &[usize], tris: &[(u32, u32, u32)]) -> (Vec<u32>, Vec<(usize, u32, u32)>) {
        let mut indices = Vec::with_capacity(tris.len() * 3);
        let mut runs: Vec<(usize, u32, u32)> = Vec::new();
        for &(_, slot, t) in tris {
            let m = models[slot as usize];
            let first = indices.len() as u32;
            let t = t as usize * 3;
            indices.extend_from_slice(&all[m].indices[t..t + 3]);
            match runs.last_mut() {
                Some(run) if run.0 == m => run.2 += 3,
                _ => runs.push((m, first, 3)),
            }
        }
        (indices, runs)
    }
}

/// Matcap layers per set the viewport draws (the shader's array size);
/// any above that are left out.
pub const MAX_MATCAPS: usize = 8;

/// A matcap layer as the viewport shader reads it: strength, blend, on.
fn matcap_params(m: &crate::matcap::Matcap) -> [f32; 4] {
    [m.strength, m.blend.shader_id(), if m.active() { 1.0 } else { 0.0 }, 0.0]
}

/// Upload with a full mip chain, built by successive linear blits — for
/// brush tips, which are usually drawn much smaller than their images.
pub(crate) fn upload_texture_mipmapped(
    queue: &Arc<Queue>,
    memory_allocator: &Arc<StandardMemoryAllocator>,
    command_buffer_allocator: &Arc<StandardCommandBufferAllocator>,
    width: u32,
    height: u32,
    rgba: &[u8],
    format: Format,
) -> Arc<ImageView> {
    use vulkano::command_buffer::{BlitImageInfo, ImageBlit};
    use vulkano::image::{ImageAspects, ImageSubresourceLayers};

    let staging: Subbuffer<[u8]> = Buffer::new_slice(
        memory_allocator.clone(),
        BufferCreateInfo { usage: BufferUsage::TRANSFER_SRC, ..Default::default() },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        rgba.len() as u64,
    )
    .unwrap();
    staging.write().unwrap().copy_from_slice(rgba);

    let mip_levels = vulkano::image::max_mip_levels([width, height, 1]);
    let image = crate::vram::image(
        memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format,
            extent: [width, height, 1],
            mip_levels,
            usage: ImageUsage::TRANSFER_SRC | ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
            ..Default::default()
        },
        crate::vram::Kind::Material,
    )
    .unwrap();

    let mut builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator.clone(),
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();
    builder.copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(staging, image.clone())).unwrap();
    let size = |level: u32| [(width >> level).max(1), (height >> level).max(1), 1];
    let layers = |mip_level: u32| ImageSubresourceLayers { aspects: ImageAspects::COLOR, mip_level, array_layers: 0..1 };
    for level in 1..mip_levels {
        builder
            .blit_image(BlitImageInfo {
                regions: [ImageBlit {
                    src_subresource: layers(level - 1),
                    src_offsets: [[0; 3], size(level - 1)],
                    dst_subresource: layers(level),
                    dst_offsets: [[0; 3], size(level)],
                    ..Default::default()
                }]
                .into(),
                filter: vulkano::image::sampler::Filter::Linear,
                ..BlitImageInfo::images(image.clone(), image.clone())
            })
            .unwrap();
    }
    builder
        .build()
        .unwrap()
        .execute(queue.clone())
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    ImageView::new_default(image).unwrap()
}

/// Upload RGBA8 pixels to a device-local sampled image and wait for
/// completion. Use an SRGB format for color data, UNORM for data maps.
pub(crate) fn upload_texture(
    queue: &Arc<Queue>,
    memory_allocator: &Arc<StandardMemoryAllocator>,
    command_buffer_allocator: &Arc<StandardCommandBufferAllocator>,
    width: u32,
    height: u32,
    rgba: &[u8],
    format: Format,
) -> Arc<ImageView> {
    // One memcpy into the staging buffer (an iterator fill goes byte by byte).
    let staging: Subbuffer<[u8]> = Buffer::new_slice(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        rgba.len() as u64,
    )
    .unwrap();
    staging.write().unwrap().copy_from_slice(rgba);

    let image = crate::vram::image(
        memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format,
            extent: [width, height, 1],
            usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
            ..Default::default()
        },
        crate::vram::Kind::Material,
    )
    .unwrap();

    let mut builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator.clone(),
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();
    builder
        .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(staging, image.clone()))
        .unwrap();
    builder
        .build()
        .unwrap()
        .execute(queue.clone())
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    ImageView::new_default(image).unwrap()
}

/// Upload an imported material map: block-compressed with mips when the
/// device can sample that format (see `texcompress`), else plain RGBA8 in
/// `fallback_format`.
pub(crate) fn upload_material_map(
    queue: &Arc<Queue>,
    memory_allocator: &Arc<StandardMemoryAllocator>,
    command_buffer_allocator: &Arc<StandardCommandBufferAllocator>,
    image: &ImageData,
    encoding: crate::texcompress::MapEncoding,
    fallback_format: Format,
) -> Arc<ImageView> {
    let device = queue.device();
    let sampleable = device.enabled_features().texture_compression_bc
        && device
            .physical_device()
            .format_properties(encoding.format())
            .is_ok_and(|p| p.optimal_tiling_features.intersects(vulkano::format::FormatFeatures::SAMPLED_IMAGE));
    if !sampleable {
        return upload_texture(
            queue,
            memory_allocator,
            command_buffer_allocator,
            image.width,
            image.height,
            &image.rgba,
            fallback_format,
        );
    }
    let compressed = crate::texcompress::compress(image, encoding);
    upload_compressed(queue, memory_allocator, command_buffer_allocator, &compressed)
}

/// Upload pre-compressed blocks, one copy region per mip level.
fn upload_compressed(
    queue: &Arc<Queue>,
    memory_allocator: &Arc<StandardMemoryAllocator>,
    command_buffer_allocator: &Arc<StandardCommandBufferAllocator>,
    compressed: &crate::texcompress::Compressed,
) -> Arc<ImageView> {
    use vulkano::command_buffer::BufferImageCopy;
    use vulkano::image::{ImageAspects, ImageSubresourceLayers};

    let staging: Subbuffer<[u8]> = Buffer::new_slice(
        memory_allocator.clone(),
        BufferCreateInfo { usage: BufferUsage::TRANSFER_SRC, ..Default::default() },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        compressed.bytes() as u64,
    )
    .unwrap();
    let mut regions = Vec::with_capacity(compressed.levels.len());
    {
        let mut writer = staging.write().unwrap();
        let mut offset = 0usize;
        for (level, blocks) in compressed.levels.iter().enumerate() {
            writer[offset..offset + blocks.len()].copy_from_slice(blocks);
            let level = level as u32;
            regions.push(BufferImageCopy {
                buffer_offset: offset as u64,
                image_subresource: ImageSubresourceLayers {
                    aspects: ImageAspects::COLOR,
                    mip_level: level,
                    array_layers: 0..1,
                },
                image_extent: [(compressed.width >> level).max(1), (compressed.height >> level).max(1), 1],
                ..Default::default()
            });
            offset += blocks.len();
        }
    }

    let image = crate::vram::image(
        memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: compressed.format,
            extent: [compressed.width, compressed.height, 1],
            mip_levels: compressed.levels.len() as u32,
            usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
            ..Default::default()
        },
        crate::vram::Kind::Material,
    )
    .unwrap();

    let mut builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator.clone(),
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();
    builder
        .copy_buffer_to_image(CopyBufferToImageInfo { regions: regions.into(), ..CopyBufferToImageInfo::buffer_image(staging, image.clone()) })
        .unwrap();
    builder
        .build()
        .unwrap()
        .execute(queue.clone())
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    ImageView::new_default(image).unwrap()
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
#version 450
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;

layout(set = 0, binding = 0) uniform Ubo {
    mat4 mvp;
    mat4 model;
    vec4 camera_pos;
    vec4 base_color;
    vec4 params;  // x: metallic, y: roughness, z: exposure, w: env intensity
    vec4 params2; // x: prefiltered mip count, y: key light strength
    vec4 key_dir; // xyz: direction the key light comes from
    vec4 surface;  // x: coat weight, y: coat roughness, z: sheen weight, w: sheen roughness
    vec4 emission; // rgb: emission, w: specular weight
    vec4 alpha;    // x: material opacity, y: coverage (0 cut-out, 1 see-through part, 2 solid part), z: material in layers
    vec4 mask_view; // rgb: overlay tint (sRGB), w: 1 = show shader-mask overlay
    mat4 view;      // world → view, for matcap lookups
    vec4 matcaps[8]; // per matcap layer, bottom to top: x strength, y blend (0 add, 1 multiply, 2 replace, 3 screen), z on
} ubo;

layout(location = 0) out vec3 v_world_pos;
layout(location = 1) out vec3 v_normal;
layout(location = 2) out vec2 v_uv;
// Matches select_vs, whose depth-tested pass redraws these surfaces.
invariant gl_Position;

void main() {
    vec4 world_pos = ubo.model * vec4(position, 1.0);
    v_world_pos = world_pos.xyz;
    v_normal = mat3(ubo.model) * normal;
    v_uv = uv;
    gl_Position = ubo.mvp * vec4(position, 1.0);
}
",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
#version 450
layout(location = 0) in vec3 v_world_pos;
layout(location = 1) in vec3 v_normal;
layout(location = 2) in vec2 v_uv;

layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 0) uniform Ubo {
    mat4 mvp;
    mat4 model;
    vec4 camera_pos;
    vec4 base_color;
    vec4 params;
    vec4 params2;
    vec4 key_dir;
    vec4 surface;
    vec4 emission;
    vec4 alpha;
    vec4 mask_view;
    mat4 view;
    vec4 matcaps[8];
} ubo;
layout(set = 0, binding = 1) uniform sampler2D base_color_tex;
layout(set = 0, binding = 2) uniform samplerCube irradiance_map;
layout(set = 0, binding = 3) uniform samplerCube prefiltered_map;
layout(set = 0, binding = 4) uniform sampler2D paint_basecolor; // premultiplied linear
layout(set = 0, binding = 5) uniform sampler2D paint_roughness; // (value*a, a)
layout(set = 0, binding = 6) uniform sampler2D paint_metallic;  // (value*a, a)
layout(set = 0, binding = 7) uniform sampler2D paint_normal;    // xyz tangent normal, w coverage
layout(set = 0, binding = 8) uniform sampler2D ao_map;          // x: baked occlusion
layout(set = 0, binding = 9) uniform sampler2D mat_roughness_tex; // r: roughness
layout(set = 0, binding = 10) uniform sampler2D mat_metallic_tex; // r: metallic
layout(set = 0, binding = 11) uniform sampler2D mat_normal_tex;   // material normal map
layout(set = 0, binding = 12) uniform sampler2D mat_ao_tex;       // r: material AO
layout(set = 0, binding = 13) uniform sampler2D paint_emissive;   // premultiplied linear
layout(set = 0, binding = 14) uniform sampler2D paint_opacity;    // (value*a, a)
layout(set = 0, binding = 15) uniform sampler2D mask_preview;     // r: shader mask over black
layout(set = 0, binding = 16) uniform sampler2D matcap_tex[8];   // sRGB sphere images
layout(set = 0, binding = 17) uniform sampler2D matcap_mask[8];  // r: where each shows
layout(set = 0, binding = 18) uniform sampler2D mat_emission_tex; // sRGB: emission map × mask

const float PI = 3.14159265359;

float d_ggx(float ndoth, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float d = ndoth * ndoth * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

float g_smith(float ndotv, float ndotl, float roughness) {
    float r = roughness + 1.0;
    float k = r * r / 8.0;
    float gv = ndotv / (ndotv * (1.0 - k) + k);
    float gl = ndotl / (ndotl * (1.0 - k) + k);
    return gv * gl;
}

vec3 fresnel_schlick(float vdoth, vec3 f0) {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - vdoth, 0.0, 1.0), 5.0);
}

vec3 shade_directional(
    vec3 n, vec3 v, vec3 l, vec3 albedo, float metallic, float roughness, vec3 light_color
) {
    float ndotl = dot(n, l);
    if (ndotl <= 0.0) return vec3(0.0);
    vec3 h = normalize(v + l);
    float ndotv = max(dot(n, v), 1e-4);
    float ndoth = max(dot(n, h), 0.0);
    float vdoth = max(dot(v, h), 0.0);
    vec3 f0 = mix(vec3(0.04), albedo, metallic);
    vec3 F = fresnel_schlick(vdoth, f0);
    float D = d_ggx(ndoth, roughness);
    float G = g_smith(ndotv, ndotl, roughness);
    vec3 specular = D * G * F / (4.0 * ndotv * ndotl + 1e-4);
    vec3 kd = (vec3(1.0) - F) * (1.0 - metallic);
    return (kd * albedo / PI + specular) * light_color * ndotl;
}

vec3 apply_matcap(vec3 c, vec3 m, vec4 p, float mask) {
    if (p.z < 0.5) return c;
    float s = p.x * clamp(mask, 0.0, 1.0);
    int mode = int(p.y + 0.5);
    if (mode == 0) return c + m * s;
    if (mode == 1) return c * mix(vec3(1.0), m, s);
    if (mode == 2) return mix(c, m, clamp(s, 0.0, 1.0));
    return vec3(1.0) - (vec3(1.0) - c) * (vec3(1.0) - clamp(m * s, 0.0, 1.0));
}

vec3 aces_tonemap(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

// Karis' analytic environment BRDF approximation (no LUT needed).
vec2 env_brdf_approx(float ndotv, float roughness) {
    const vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
    const vec4 c1 = vec4(1.0, 0.0425, 1.04, -0.04);
    vec4 r = roughness * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * ndotv)) * r.x + r.y;
    return vec2(-1.04, 1.04) * a004 + r.zw;
}

// Per-pixel tangent frame from position/UV derivatives (no mesh tangents needed).
vec3 perturb_normal(vec3 n, vec3 view_dir, vec2 uv, vec3 map_n) {
    vec3 dp1 = dFdx(-view_dir);
    vec3 dp2 = dFdy(-view_dir);
    vec2 duv1 = dFdx(uv);
    vec2 duv2 = dFdy(uv);
    vec3 dp2perp = cross(dp2, n);
    vec3 dp1perp = cross(n, dp1);
    vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
    vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;
    float invmax = inversesqrt(max(dot(T, T), dot(B, B)));
    return normalize(mat3(T * invmax, B * invmax, n) * map_n);
}

void main() {
    // Surface alpha = material opacity × base color texture alpha, with the
    // painted opacity channel blending toward its painted value.
    vec2 paint_o = texture(paint_opacity, v_uv).xy;
    float base_alpha = ubo.alpha.x * texture(base_color_tex, v_uv).a;
    float surface_alpha = base_alpha * (1.0 - paint_o.y) + paint_o.x;
    // Cut-out materials keep alpha >= 0.5 as opaque. A blended material is
    // drawn twice: its solid texels with the opaques (writing depth), the
    // rest alpha-blended in the sorted pass; each texel lands in exactly one.
    float coverage = ubo.alpha.y;
    bool blended = coverage > 0.5 && coverage < 1.5;
    const float SOLID = 0.995;
    if (coverage < 0.5 && surface_alpha < 0.5) discard;
    if (coverage > 1.5 && surface_alpha < SOLID) discard;
    if (blended && (surface_alpha < 0.003 || surface_alpha >= SOLID)) discard;

    vec3 n = normalize(v_normal);
    if (!gl_FrontFacing) n = -n;
    vec3 v = normalize(ubo.camera_pos.xyz - v_world_pos);

    // Material normal map first (fallback is a flat map ≈ no-op) …
    // x, y only: BC5 maps have no z, so it is rebuilt (tangent space, z > 0).
    vec2 mat_nxy = texture(mat_normal_tex, v_uv).xy * 2.0 - 1.0;
    vec3 mat_n = vec3(mat_nxy, sqrt(max(1.0 - dot(mat_nxy, mat_nxy), 0.0)));
    n = perturb_normal(n, v, v_uv, normalize(mat_n));
    // … then painted height perturbs on top.
    vec4 nm = texture(paint_normal, v_uv);
    if (nm.w > 0.001) {
        vec3 perturbed = perturb_normal(n, v, v_uv, normalize(nm.xyz));
        n = normalize(mix(n, perturbed, nm.w));
    }

    // When the layers carry the material's own maps, only its factors
    // remain underneath them.
    bool in_layers = ubo.alpha.z > 0.5;
    vec3 albedo = (in_layers ? vec3(1.0) : texture(base_color_tex, v_uv).rgb) * ubo.base_color.rgb;
    float metallic = clamp(ubo.params.x * (in_layers ? 1.0 : texture(mat_metallic_tex, v_uv).r), 0.0, 1.0);
    float roughness = clamp(ubo.params.y * (in_layers ? 1.0 : texture(mat_roughness_tex, v_uv).r), 0.03, 1.0);

    // UV checker debug view.
    if (ubo.params2.w > 0.5) {
        vec2 g = floor(v_uv * 16.0);
        float checker = mod(g.x + g.y, 2.0);
        albedo = mix(vec3(0.25), vec3(0.65), checker);
        vec2 f = fract(v_uv * 16.0);
        if (min(f.x, min(f.y, min(1.0 - f.x, 1.0 - f.y))) < 0.03) {
            albedo = vec3(0.9, 0.4, 0.1);
        }
        metallic = 0.0;
        roughness = 0.6;
    }

    // Painted layers composite over the material (premultiplied alpha).
    if (ubo.params2.w < 0.5) {
        vec4 paint_bc = texture(paint_basecolor, v_uv);
        albedo = albedo * (1.0 - paint_bc.a) + paint_bc.rgb;
        vec2 paint_r = texture(paint_roughness, v_uv).xy;
        roughness = clamp(roughness * (1.0 - paint_r.y) + paint_r.x, 0.03, 1.0);
        vec2 paint_m = texture(paint_metallic, v_uv).xy;
        metallic = clamp(metallic * (1.0 - paint_m.y) + paint_m.x, 0.0, 1.0);
    }

    float exposure = ubo.params.z;
    float env_intensity = ubo.params.w;
    float prefiltered_mips = ubo.params2.x;
    float key_light = ubo.params2.y;

    float ndotv = max(dot(n, v), 1e-4);
    // OpenPBR-style specular weight scales the dielectric F0.
    float spec_weight = clamp(ubo.emission.w, 0.0, 1.0);
    vec3 f0 = mix(vec3(0.04) * spec_weight, albedo, metallic);

    // Image-based lighting.
    vec3 irradiance = texture(irradiance_map, n).rgb;
    vec3 diffuse_ibl = irradiance * albedo * (1.0 - metallic);
    vec3 r = reflect(-v, n);
    vec3 prefiltered =
        textureLod(prefiltered_map, r, roughness * (prefiltered_mips - 1.0)).rgb;
    vec2 ab = env_brdf_approx(ndotv, roughness);
    vec3 specular_ibl = prefiltered * (f0 * ab.x + ab.y);

    // Clear coat: a second dielectric lobe layered on top; the base is
    // attenuated by the energy the coat reflects.
    float coat_weight = ubo.surface.x;
    if (coat_weight > 0.001) {
        float coat_rough = clamp(ubo.surface.y, 0.03, 1.0);
        vec3 coat_pref =
            textureLod(prefiltered_map, r, coat_rough * (prefiltered_mips - 1.0)).rgb;
        vec2 cab = env_brdf_approx(ndotv, coat_rough);
        vec3 coat_ibl = coat_pref * (vec3(0.04) * cab.x + cab.y);
        float coat_fresnel = 0.04 + 0.96 * pow(1.0 - ndotv, 5.0);
        float atten = 1.0 - coat_weight * coat_fresnel;
        diffuse_ibl *= atten;
        specular_ibl = specular_ibl * atten + coat_ibl * coat_weight;
    }

    // Sheen: retro-reflective fuzz at grazing angles (cloth).
    float sheen_weight = ubo.surface.z;
    if (sheen_weight > 0.001) {
        float sheen_rough = clamp(ubo.surface.w, 0.05, 1.0);
        float grazing = pow(1.0 - ndotv, 3.0 + (1.0 - sheen_rough) * 5.0);
        specular_ibl += irradiance * sheen_weight * grazing;
    }

    // Baked and material ambient occlusion attenuate the environment terms.
    float ao = texture(ao_map, v_uv).r * texture(mat_ao_tex, v_uv).r;
    float ao_strength = ubo.params2.z;
    float occlusion = mix(1.0, ao, ao_strength);
    vec3 color = (diffuse_ibl + specular_ibl) * env_intensity * occlusion;

    // Optional extra key light for sculpting the shading.
    color += shade_directional(n, v, normalize(ubo.key_dir.xyz),
                               albedo, metallic, roughness,
                               vec3(1.0, 0.98, 0.95) * 3.0 * key_light);

    // Material emission plus painted emissive (premultiplied linear).
    color += ubo.emission.rgb * texture(mat_emission_tex, v_uv).rgb;
    color += texture(paint_emissive, v_uv).rgb;

    color *= exposure;
    color = aces_tonemap(color);
    // Matcaps (Poiyomi-style), on the display-range colour as in VRChat: a
    // sphere image looked up by the view-space normal, where the mask allows.
    if (ubo.matcaps[0].z > 0.5) {
        vec3 vn = normalize(mat3(ubo.view) * n);
        vec2 muv = vec2(vn.x, -vn.y) * 0.5 + 0.5;
        for (int i = 0; i < 8; i++) {
            if (ubo.matcaps[i].z < 0.5) break;
            color = apply_matcap(color, texture(matcap_tex[i], muv).rgb, ubo.matcaps[i], texture(matcap_mask[i], v_uv).r);
        }
    }
    // Target image is UNORM; encode gamma manually.
    color = pow(color, vec3(1.0 / 2.2));
    // Shader-mask overlay: keep the shading readable but grey and dim, and
    // paint the mask on top in its tint so white/black is unmistakable.
    if (ubo.mask_view.w > 0.5) {
        float m = clamp(texture(mask_preview, v_uv).r, 0.0, 1.0);
        vec3 dim = vec3(dot(color, vec3(0.299, 0.587, 0.114))) * 0.35;
        color = mix(dim, ubo.mask_view.rgb, m);
    }
    f_color = vec4(color, blended ? clamp(surface_alpha, 0.0, 1.0) : 1.0);
}
",
    }
}

mod sky_vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
#version 450
layout(location = 0) out vec2 v_ndc;

void main() {
    // Fullscreen triangle from the vertex index, pinned to the far plane.
    vec2 pos = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2) * 2.0 - 1.0;
    v_ndc = pos;
    gl_Position = vec4(pos, 1.0, 1.0);
}
",
    }
}

mod sky_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
#version 450
layout(location = 0) in vec2 v_ndc;
layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 0) uniform samplerCube env;
layout(push_constant) uniform Pc {
    mat4 inv_view_proj;
    vec4 camera_pos;
    vec4 params; // x: exposure, y: env intensity
} pc;

vec3 aces_tonemap(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

void main() {
    vec4 world = pc.inv_view_proj * vec4(v_ndc, 1.0, 1.0);
    vec3 dir = normalize(world.xyz / world.w - pc.camera_pos.xyz);
    vec3 color = textureLod(env, dir, 0.0).rgb * pc.params.y;
    color *= pc.params.x;
    color = aces_tonemap(color);
    color = pow(color, vec3(1.0 / 2.2));
    f_color = vec4(color, 1.0);
}
",
    }
}

mod select_vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
#version 450
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;

layout(push_constant) uniform Pc {
    mat4 mvp;
    vec4 channel; // which mask channels this piece covers
    vec4 alpha;   // x: material opacity
} pc;

layout(location = 0) out vec2 v_uv;
// Same transform as the scene pass, so the depth-tested pass lands exactly.
invariant gl_Position;

void main() {
    v_uv = uv;
    gl_Position = pc.mvp * vec4(position, 1.0);
}
",
    }
}

mod select_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
#version 450
layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_mask;

layout(set = 0, binding = 0) uniform sampler2D base_color_tex;
layout(set = 0, binding = 1) uniform sampler2D paint_opacity; // (value*a, a)
layout(push_constant) uniform Pc {
    mat4 mvp;
    vec4 channel;
    vec4 alpha;
} pc;

void main() {
    vec2 paint_o = texture(paint_opacity, v_uv).xy;
    float base_alpha = pc.alpha.x * texture(base_color_tex, v_uv).a;
    if (base_alpha * (1.0 - paint_o.y) + paint_o.x < 0.5) discard;
    o_mask = pc.channel;
}
",
    }
}

mod outline_cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
#version 450
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 0, rgba8) uniform image2D target;
layout(set = 0, binding = 1, rgba8) uniform readonly image2D mask;
layout(push_constant) uniform Pc {
    uvec4 size;
    vec4 params; // x: outline radius in pixels
} pc;

// sRGB, written straight into the gamma-encoded viewport image.
const vec3 SELECTED = vec3(1.0, 0.58, 0.1);
const vec3 HOVERED = vec3(0.45, 0.85, 1.0);
const float HIDDEN = 0.4;

void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    ivec2 hi = ivec2(pc.size.xy) - 1;
    if (p.x > hi.x || p.y > hi.y) return;
    vec4 here = imageLoad(mask, p);
    float radius = pc.params.x;
    int r = int(ceil(radius));
    vec4 near = vec4(0.0);
    for (int dy = -r; dy <= r; dy++) {
        for (int dx = -r; dx <= r; dx++) {
            if (float(dx * dx + dy * dy) > radius * radius + 0.5) continue;
            near = max(near, imageLoad(mask, clamp(p + ivec2(dx, dy), ivec2(0), hi)));
        }
    }
    // A ring just outside each piece's whole (x-ray) silhouette, so nothing
    // is drawn inside it: full strength beside a visible part of the piece,
    // faint where the piece is hidden behind something.
    vec4 c = imageLoad(target, p);
    if (here.g < 0.5 && near.g > 0.5) {
        c.rgb = mix(c.rgb, SELECTED, near.r > 0.5 ? 1.0 : HIDDEN);
    } else if (here.a < 0.5 && near.a > 0.5) {
        c.rgb = mix(c.rgb, HOVERED, near.b > 0.5 ? 1.0 : HIDDEN);
    } else {
        return;
    }
    imageStore(target, p, c);
}
",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::ShaderMask;
    use crate::scene::MaterialData;
    use vulkano::command_buffer::CopyImageToBufferInfo;

    const SIZE: u32 = 64;

    fn sphere(name: &str, z: f32, radius: f32) -> ModelData {
        ModelData {
            // uv_sphere winds clockwise seen from outside; imported meshes
            // (glTF, FBX) wind counter-clockwise, so flip it to match them.
            mesh: {
                let mut mesh = crate::mesh::uv_sphere(radius, 24, 48);
                mesh.indices.chunks_exact_mut(3).for_each(|t| t.swap(1, 2));
                mesh
            },
            material: MaterialData::default(),
            transform: Mat4::from_translation(Vec3::new(0.0, 0.0, z)),
            name: name.into(),
            material_name: "M".into(),
            morphs: Default::default(),
            skin: None,
        }
    }

    /// Render `models` head-on from +Z and read the frame back (sRGB bytes).
    /// `setup` gets the texture sets before the frame, `selected` is outlined.
    fn render(models: Vec<ModelData>, selected: Vec<usize>, setup: impl FnOnce(&GpuContextRef, &mut [TextureSet])) -> Option<Vec<[u8; 4]>> {
        let gpu = crate::environment::test_gpu::context()?;
        let mut renderer = SceneRenderer::new(
            gpu.queue.clone(),
            gpu.memory_allocator.clone(),
            gpu.command_buffer_allocator.clone(),
            gpu.descriptor_set_allocator.clone(),
            Format::R8G8B8A8_UNORM,
            crate::environment::Environment::procedural(&gpu),
        );
        renderer.set_models(&models);
        let mut sets: Vec<TextureSet> = crate::scene::set_keys(&models)
            .into_iter()
            .map(|k| TextureSet::new(k.object, k.material, gpu.clone(), 64))
            .collect();
        setup(&gpu, &mut sets);
        let target = ImageView::new_default(
            crate::vram::image(
                gpu.memory_allocator.clone(),
                ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format: Format::R8G8B8A8_UNORM,
                    extent: [SIZE, SIZE, 1],
                    usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC,
                    ..Default::default()
                },
                crate::vram::Kind::Viewport,
            )
            .unwrap(),
        )
        .unwrap();
        let eye = Vec3::new(0.0, 0.0, 4.0);
        let view = glam::camera::rh::view::look_at_mat4(eye, Vec3::ZERO, Vec3::Y);
        let params = RenderParams {
            camera_pos: eye,
            view_proj: glam::camera::rh::proj::vulkan::perspective(0.8, 1.0, 0.1, 100.0) * view,
            view,
            exposure: 1.0,
            env_intensity: 1.0,
            key_light: 1.0,
            key_light_dir: Vec3::new(0.3, 0.5, 1.0),
            ao_strength: 1.0,
            uv_checker: false,
            material_override: None,
            selected,
            hovered: Vec::new(),
            outline_width: 2.0,
            mask_view: None,
        };
        let done = renderer.render(vulkano::sync::now(gpu.queue.device().clone()).boxed(), target.clone(), &params, &mut sets);
        done.then_signal_fence_and_flush().unwrap().wait(None).unwrap();

        let buffer: Subbuffer<[u8]> = Buffer::new_slice(
            gpu.memory_allocator.clone(),
            BufferCreateInfo { usage: BufferUsage::TRANSFER_DST, ..Default::default() },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            (SIZE * SIZE * 4) as u64,
        )
        .unwrap();
        let mut builder = AutoCommandBufferBuilder::primary(
            gpu.command_buffer_allocator.clone(),
            gpu.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        builder.copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(target.image().clone(), buffer.clone())).unwrap();
        builder.build().unwrap().execute(gpu.queue.clone()).unwrap().then_signal_fence_and_flush().unwrap().wait(None).unwrap();
        let px = buffer.read().unwrap();
        Some(px.chunks(4).map(|c| [c[0], c[1], c[2], c[3]]).collect())
    }

    type GpuContextRef = crate::environment::GpuContext;

    fn at(px: &[[u8; 4]], x: u32, y: u32) -> [u8; 4] {
        px[(y * SIZE + x) as usize]
    }

    fn orange(p: [u8; 4]) -> bool {
        p[0] > 200 && p[1] > 100 && p[1] < 190 && p[2] < 80
    }

    /// The outline rings the whole piece just outside its silhouette, with
    /// nothing drawn across its inside; where the piece is hidden behind
    /// another, the ring is still there, but faint.
    #[test]
    fn selection_outline_rings_the_outside_only() {
        let Some(plain) = render(vec![sphere("Ball", 0.0, 1.0)], vec![], |_, _| {}) else {
            eprintln!("no Vulkan device; skipping");
            return;
        };
        let lit = render(vec![sphere("Ball", 0.0, 1.0)], vec![0], |_, _| {}).unwrap();
        let row = SIZE / 2;
        // The sphere's right edge on the middle row: first background pixel.
        let edge = (SIZE / 2..SIZE).find(|&x| at(&plain, x, row) == at(&plain, SIZE - 1, row)).unwrap();
        assert!(orange(at(&lit, edge, row)), "ring just outside the edge: {:?}", at(&lit, edge, row));
        assert_eq!(at(&lit, SIZE / 2, row), at(&plain, SIZE / 2, row), "nothing inside the piece");
        assert_eq!(at(&lit, SIZE - 2, 2), at(&plain, SIZE - 2, 2), "nothing far away");

        // The same ball behind a bigger one: its ring shows faintly through.
        let models = || vec![sphere("Ball", -3.0, 1.0), sphere("Wall", 0.5, 1.4)];
        let hidden_plain = render(models(), vec![], |_, _| {}).unwrap();
        let hidden = render(models(), vec![0], |_, _| {}).unwrap();
        let changed: Vec<(u32, u32)> = (0..SIZE)
            .flat_map(|y| (0..SIZE).map(move |x| (x, y)))
            .filter(|&(x, y)| at(&hidden, x, y) != at(&hidden_plain, x, y))
            .collect();
        assert!(!changed.is_empty(), "a hidden piece still shows its outline");
        assert!(changed.iter().all(|&(x, y)| !orange(at(&hidden, x, y))), "but only faintly");
    }

    /// A matcap layer changes the shading where its mask is white, and each
    /// layer adds its own matcap.
    #[test]
    fn matcap_layers_draw_in_the_viewport() {
        let Some(plain) = render(vec![sphere("Ball", 0.0, 1.0)], vec![], |_, _| {}) else {
            eprintln!("no Vulkan device; skipping");
            return;
        };
        let add_matcap = |gpu: &GpuContextRef, sets: &mut [TextureSet], name: &str, fill: f32| {
            let layers = &mut sets[0].layers;
            layers.add_shader_mask_layer(ShaderMask::Matcap);
            layers.fill_active(Some([fill; 3]), None, None);
            let top = layers.active;
            let image = crate::matcap::builtin(name).unwrap();
            layers.layers[top].set_matcap(gpu, name.into(), image, Some(crate::matcap::MatcapBlend::Replace));
        };
        let gold = render(vec![sphere("Ball", 0.0, 1.0)], vec![], |gpu, sets| add_matcap(gpu, sets, "Gold", 1.0)).unwrap();
        let masked = render(vec![sphere("Ball", 0.0, 1.0)], vec![], |gpu, sets| add_matcap(gpu, sets, "Gold", 0.0)).unwrap();
        let both = render(vec![sphere("Ball", 0.0, 1.0)], vec![], |gpu, sets| {
            add_matcap(gpu, sets, "Gold", 1.0);
            add_matcap(gpu, sets, "Clay", 1.0);
        })
        .unwrap();
        let c = |px: &[[u8; 4]]| at(px, SIZE / 2, SIZE / 2);
        let g = c(&gold);
        assert!(g[0] > g[2] + 40, "gold replaces the shading: {g:?} (plain {:?})", c(&plain));
        assert_eq!(c(&masked), c(&plain), "a black mask hides it");
        assert_ne!(c(&both), g, "the upper layer draws over the lower");
    }

    /// Once the material's texture lives in the "Base material" layer, the
    /// viewport stops drawing the original under it: hiding the layer really
    /// takes the texture away.
    #[test]
    fn hiding_the_base_material_layer_hides_the_texture() {
        let red_ball = || {
            let mut ball = sphere("Ball", 0.0, 1.0);
            ball.material.base_color_image = Some(ImageData { width: 2, height: 2, rgba: [220u8, 20, 20, 255].repeat(4).into() });
            vec![ball]
        };
        let seed = |hide: bool| {
            move |_: &GpuContextRef, sets: &mut [TextureSet]| {
                let image = ImageData { width: 2, height: 2, rgba: [220u8, 20, 20, 255].repeat(4).into() };
                assert!(sets[0].layers.seed_from_material(&[(crate::layers::MapKind::BaseColor, &image, [1.0; 3])]));
                sets[0].material_in_layers = true;
                sets[0].layers.layers[0].visible = !hide;
                sets[0].layers.mark_dirty();
            }
        };
        let Some(plain) = render(red_ball(), vec![], |_, _| {}) else {
            eprintln!("no Vulkan device; skipping");
            return;
        };
        let shown = render(red_ball(), vec![], seed(false)).unwrap();
        let hidden = render(red_ball(), vec![], seed(true)).unwrap();
        let c = |px: &[[u8; 4]]| at(px, SIZE / 2, SIZE / 2);
        let red = |p: [u8; 4]| p[0] as i32 - p[1] as i32 > 60;
        assert!(red(c(&plain)) && red(c(&shown)), "red either way while shown: {:?} {:?}", c(&plain), c(&shown));
        assert!(!red(c(&hidden)), "hidden base layer leaves no red: {:?}", c(&hidden));
    }

    /// A blended piece shows its near side on top of its far side. The
    /// sphere's front (u < 0.5, facing the camera) is red, its back blue;
    /// the back's triangles come later in the index buffer, so drawn in
    /// mesh order they would paint over the front.
    #[test]
    fn transparent_pieces_draw_their_front_over_their_back() {
        let mut ball = sphere("Ball", 0.0, 1.0);
        let rgba: Vec<u8> = (0..64).flat_map(|x| if x < 32 { [220u8, 20, 20, 230] } else { [20, 20, 220, 230] }).collect();
        ball.material.base_color_image = Some(ImageData { width: 64, height: 1, rgba: rgba.into() });
        ball.material.transparent = true;
        let Some(px) = render(vec![ball], vec![], |_, _| {}) else {
            eprintln!("no Vulkan device; skipping");
            return;
        };
        let c = at(&px, SIZE / 2, SIZE / 2);
        assert!(c[0] as i32 - c[2] as i32 > 60, "front (red) on top, got {c:?}");
    }

    /// Sheer pieces in front of each other draw in depth order where they
    /// overlap, even when a piece's centre says otherwise: the blue piece is
    /// a quad behind the red one plus a sliver out near the camera, which
    /// puts its centre in front.
    #[test]
    fn overlapping_transparent_pieces_draw_far_to_near() {
        let tinted = |name: &str, rgb: [u8; 3], quads: &[(f32, f32, f32)]| {
            let mut vertices = Vec::new();
            let mut indices = Vec::new();
            for &(x, z, half) in quads {
                let base = vertices.len() as u32;
                for (dx, dy) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                    vertices.push(Vertex3D { position: [x + dx * half, dy * half, z], normal: [0.0, 0.0, 1.0], uv: [dx * 0.5 + 0.5, dy * 0.5 + 0.5] });
                }
                indices.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
            }
            let mut material = MaterialData::default();
            material.base_color_image = Some(ImageData { width: 1, height: 1, rgba: [rgb[0], rgb[1], rgb[2], 220].to_vec().into() });
            material.transparent = true;
            ModelData {
                mesh: crate::mesh::MeshData { vertices, indices },
                material,
                transform: Mat4::IDENTITY,
                name: name.into(),
                material_name: "M".into(),
                morphs: Default::default(),
                skin: None,
            }
        };
        let red = tinted("Red", [220, 20, 20], &[(0.0, 1.0, 0.6)]);
        let blue = tinted("Blue", [20, 20, 220], &[(0.0, -1.0, 1.0), (0.5, 3.8, 0.01)]);
        let Some(px) = render(vec![red, blue], vec![], |_, _| {}) else {
            eprintln!("no Vulkan device; skipping");
            return;
        };
        let c = at(&px, SIZE / 2, SIZE / 2);
        assert!(c[0] as i32 - c[2] as i32 > 60, "near (red) on top, got {c:?}");
    }

    /// The solid texels of a blended material occlude like an opaque one,
    /// where no triangle order could: a red quad crossed by a slanted blue
    /// one (z = x) whose lower triangle sorts in front of the red but lies
    /// behind it left of centre.
    #[test]
    fn solid_parts_of_blended_pieces_occlude_by_depth() {
        let piece = |name: &str, rgb: [u8; 3], corners: [[f32; 3]; 4]| {
            let uvs = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];
            let vertices = corners
                .iter()
                .zip(uvs)
                .map(|(&position, uv)| Vertex3D { position, normal: [0.0, 0.0, 1.0], uv })
                .collect();
            let mut material = MaterialData::default();
            material.base_color_image = Some(ImageData { width: 1, height: 1, rgba: [rgb[0], rgb[1], rgb[2], 255].to_vec().into() });
            material.transparent = true;
            ModelData {
                mesh: crate::mesh::MeshData { vertices, indices: vec![0, 1, 2, 0, 2, 3] },
                material,
                transform: Mat4::IDENTITY,
                name: name.into(),
                material_name: "M".into(),
                morphs: Default::default(),
                skin: None,
            }
        };
        let red = piece("Red", [220, 20, 20], [[-0.6, -0.6, 0.0], [0.6, -0.6, 0.0], [0.6, 0.6, 0.0], [-0.6, 0.6, 0.0]]);
        let blue = piece("Blue", [20, 20, 220], [[-1.0, -1.0, -1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, -1.0]]);
        let Some(px) = render(vec![red, blue], vec![], |_, _| {}) else {
            eprintln!("no Vulkan device; skipping");
            return;
        };
        // World (-0.2, -0.5): inside the blue quad's lower triangle, where
        // it is at z = -0.2, behind the red.
        let c = at(&px, 28, 41);
        assert!(c[0] as i32 - c[2] as i32 > 60, "red in front, got {c:?}");
        // World (0.3, 0.1): blue at z = 0.3, in front.
        let d = at(&px, 37, 30);
        assert!(d[2] as i32 - d[0] as i32 > 60, "blue in front, got {d:?}");
    }
}
