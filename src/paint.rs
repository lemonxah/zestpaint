use std::sync::Arc;

use glam::{Mat4, Vec2, Vec3};
use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::buffer::BufferUsage;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryCommandBufferAbstract,
    RenderPassBeginInfo,
};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::format::Format;
use vulkano::image::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo};
use vulkano::memory::allocator::MemoryTypeFilter;
use vulkano::pipeline::graphics::color_blend::{
    AttachmentBlend, BlendFactor, BlendOp, ColorBlendAttachmentState, ColorBlendState,
};
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
use vulkano::device::Device;
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::shader::EntryPoint;
use vulkano::sync::GpuFuture;

use crate::environment::GpuContext;
use crate::layers::TextureSet;
use crate::mesh::Vertex3D;
use crate::renderer::SceneRenderer;

/// How far a stroke spreads across geometry once it lands.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum FillMode {
    /// Normal brushing — coverage feathers across polygons.
    Off,
    /// Fill whole triangles, so strokes stop hard on their edges.
    Triangle,
    /// Fill the whole UV chart under the brush.
    UvIsland,
    /// Fill the whole disconnected piece under the brush — the paint bucket.
    Shell,
}

impl FillMode {
    pub const ALL: [FillMode; 4] =
        [FillMode::Off, FillMode::Triangle, FillMode::UvIsland, FillMode::Shell];

    pub fn label(self) -> &'static str {
        match self {
            FillMode::Off => "Brush",
            FillMode::Triangle => "Triangle",
            FillMode::UvIsland => "UV island",
            FillMode::Shell => "Whole piece",
        }
    }

    pub fn is_on(self) -> bool {
        self != FillMode::Off
    }
}

/// How a stroke is laid down.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum StrokeMode {
    /// Follow the cursor.
    Freehand,
    /// Anchor on press, drag to the far end, commit on release.
    Line,
    /// One stamp per click — decals, tattoos, graffiti.
    Stamp,
}

impl StrokeMode {
    pub const ALL: [StrokeMode; 3] =
        [StrokeMode::Freehand, StrokeMode::Line, StrokeMode::Stamp];

    pub fn label(self) -> &'static str {
        match self {
            StrokeMode::Freehand => "Freehand",
            StrokeMode::Line => "Line",
            StrokeMode::Stamp => "Stamp",
        }
    }
}

/// What a stroke writes into.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PaintTarget {
    /// The active layer's material channels.
    Material,
    /// The active layer's mask — reveals or hides what the layer already holds.
    Mask,
}

impl PaintTarget {
    pub fn label(self) -> &'static str {
        match self {
            PaintTarget::Material => "Material",
            PaintTarget::Mask => "Mask",
        }
    }
}

/// What a stroke does to the texels under the brush.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BrushOp {
    /// Deposit the brush's channel values.
    Paint,
    /// Remove coverage (destination-out).
    Erase,
    /// Drag existing texels along the stroke's motion.
    Smudge,
    /// Copy what is under the clone source onto what is under the brush.
    /// Goes through `PaintSystem::clone_stroke`.
    Clone,
}

#[derive(Clone)]
pub struct BrushSettings {
    /// What the stroke writes into: layer channels or the layer mask.
    pub target: PaintTarget,
    /// Freehand, straight line, or a single decal stamp.
    pub stroke_mode: StrokeMode,
    /// Mask value deposited in mask mode: 1 paints the layer in, 0 paints it out.
    pub mask_value: f32,
    /// Paint, erase or smudge. Erase uses the same stamps, falloff, tip and
    /// fill modes as Paint; only the blend changes (destination-out), so it
    /// respects the channel toggles and works on masks too. Smudge goes
    /// through `PaintSystem::smudge` instead of `stroke`.
    pub op: BrushOp,
    /// Grey deposited on a shader-mask layer: 1 = effect on, 0 = off.
    pub shader_mask_value: f32,
    /// Brush radius in viewport pixels.
    pub size: f32,
    /// 0 = soft falloff from center, 1 = hard edge.
    pub hardness: f32,
    /// Alpha deposited per stamp.
    pub flow: f32,
    /// Tip stamp: RGB is the stamp's own colour, alpha is its coverage.
    /// None = plain round brush.
    pub tip: Option<std::sync::Arc<vulkano::image::view::ImageView>>,
    pub tip_name: Option<String>,
    /// The loaded tip carries colour, not just an alpha silhouette.
    pub tip_has_color: bool,
    /// Paint the tip's own colours instead of the picker colour — what makes a
    /// leaf stamp deposit green leaves rather than a leaf-shaped blob.
    pub use_tip_color: bool,
    /// Tip rotation in radians, for aiming decals.
    pub tip_rotation: f32,
    /// Tip width / height, so a stamp keeps its proportions.
    pub tip_aspect: f32,
    /// Stamp the whole tip image in a rectangle (no round clip, no falloff)
    /// instead of a round brush. Set by the stamp tool.
    pub square_tip: bool,
    /// Mirror strokes across the world X=0 plane.
    pub symmetry_x: bool,
    /// How far a stroke spreads: off, whole triangles, whole UV islands, or a
    /// whole disconnected piece.
    pub fill_mode: FillMode,
    /// sRGB color from the picker: the foreground colour, what strokes paint.
    pub color: [f32; 3],
    /// sRGB background colour, swapped in with X (as in Photoshop / GIMP).
    pub background_color: [f32; 3],
    pub roughness_value: f32,
    pub metallic_value: f32,
    /// Signed: positive raises the surface, negative carves into it.
    pub height_value: f32,
    /// sRGB emissive color; intensity multiplier applied on top.
    pub emissive_color: [f32; 3],
    pub emissive_intensity: f32,
    /// 0 = fully transparent (clipped), 1 = opaque.
    pub opacity_value: f32,
    pub paint_color: bool,
    pub paint_roughness: bool,
    pub paint_metallic: bool,
    pub paint_height: bool,
    pub paint_emissive: bool,
    pub paint_opacity: bool,
}

impl BrushSettings {
    /// The layer channels a material stroke writes, per the channel toggles.
    pub fn material_channels(&self) -> Vec<crate::layers::PaintChannel> {
        use crate::layers::PaintChannel;
        [
            (self.paint_color, PaintChannel::BaseColor),
            (self.paint_roughness, PaintChannel::Roughness),
            (self.paint_metallic, PaintChannel::Metallic),
            (self.paint_height, PaintChannel::Height),
            (self.paint_emissive, PaintChannel::Emissive),
            (self.paint_opacity, PaintChannel::Opacity),
        ]
        .into_iter()
        .filter_map(|(on, c)| on.then_some(c))
        .collect()
    }
}

/// Foreground and background colours to start with, and after D: black and
/// white, as in Photoshop and GIMP.
pub const DEFAULT_COLORS: ([f32; 3], [f32; 3]) = ([0.0; 3], [1.0; 3]);

impl Default for BrushSettings {
    fn default() -> Self {
        Self {
            target: PaintTarget::Material,
            stroke_mode: StrokeMode::Freehand,
            mask_value: 1.0,
            op: BrushOp::Paint,
            shader_mask_value: 1.0,
            size: 40.0,
            hardness: 0.5,
            flow: 0.35,
            tip: None,
            tip_name: None,
            tip_has_color: false,
            use_tip_color: false,
            tip_rotation: 0.0,
            tip_aspect: 1.0,
            square_tip: false,
            symmetry_x: false,
            fill_mode: FillMode::Off,
            color: DEFAULT_COLORS.0,
            background_color: DEFAULT_COLORS.1,
            roughness_value: 0.5,
            metallic_value: 0.0,
            height_value: 0.5,
            emissive_color: [1.0, 0.6, 0.2],
            emissive_intensity: 2.0,
            opacity_value: 0.0,
            paint_color: true,
            paint_roughness: false,
            paint_metallic: false,
            paint_height: false,
            paint_emissive: false,
            paint_opacity: false,
        }
    }
}

/// Camera state the brush projection needs for one stroke segment.
pub struct PaintView {
    pub view_proj: Mat4,
    pub camera_pos: Vec3,
    pub viewport_size: Vec2,
}

/// Projects screen-space brush stamps onto the mesh, rasterizing the mesh in
/// UV space so each stamp lands in the active layer's textures.
pub struct PaintSystem {
    gpu: GpuContext,
    /// Shaders of the material-channel passes, for building `variants`.
    stroke_shaders: StrokeShaders,
    /// Material-channel pass and pipelines per set of present attachments
    /// (bit i = pass slot i: colour, roughness, metallic, height, emissive,
    /// opacity). Built on first use.
    variants: std::collections::HashMap<u8, Arc<StrokeVariant>>,
    /// Single-attachment pass used when the brush targets the layer mask.
    mask_render_pass: Arc<RenderPass>,
    mask_pipeline: Arc<GraphicsPipeline>,
    /// Same shader as `mask_pipeline`, blended destination-out.
    mask_erase_pipeline: Arc<GraphicsPipeline>,
    /// Per-channel copies of the layer being smudged: a render pass cannot
    /// sample the image it is writing. Keyed by resolution.
    smudge_scratch: Option<(u32, Vec<Arc<vulkano::image::view::ImageView>>)>,
    smudge_sampler: Arc<Sampler>,
    /// Clone: a camera-view pass writes each pixel's UV (z = 1 where there is
    /// surface), so the clone stage can look up what lies under the source.
    gbuf_render_pass: Arc<RenderPass>,
    gbuf_pipeline: Arc<GraphicsPipeline>,
    /// (viewport size, uv target, depth) — reused while the size holds.
    gbuf: Option<([u32; 2], Arc<vulkano::image::view::ImageView>, Arc<vulkano::image::view::ImageView>)>,
    nearest_sampler: Arc<Sampler>,
    /// Mask-pass mark pipeline (the material one is in each variant); kept
    /// separate because a writable SSBO forces every draw that binds it into
    /// its own render pass instance.
    mask_mark_pipeline: Arc<GraphicsPipeline>,
    ubo_allocator: SubbufferAllocator,
    /// Per-segment stamp positions, read by the fragment stage.
    stamp_allocator: SubbufferAllocator,
    /// One flag per triangle, written by the polygon-fill mark pass.
    marks_allocator: SubbufferAllocator,
    depth_sampler: Arc<Sampler>,
    tip_sampler: Arc<Sampler>,
    /// 1x1 white fallback when no tip image is set (plain round brush).
    white_tip: Arc<vulkano::image::view::ImageView>,
}

impl PaintSystem {
    pub fn new(gpu: GpuContext) -> Self {
        let device = gpu.queue.device().clone();

        // Painting the mask writes one scalar channel, so it needs its own
        // single-attachment pass; everything else about the stroke is identical.
        let mask_render_pass = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                mask: {
                    format: Format::R16G16B16A16_SFLOAT,
                    samples: 1,
                    load_op: Load,
                    store_op: Store,
                },
            },
            pass: {
                color: [mask],
                depth_stencil: {},
            },
        )
        .unwrap();

        let vs = paint_vs::load(device.clone()).unwrap().entry_point("main").unwrap();
        let fs = paint_fs::load(device.clone()).unwrap().entry_point("main").unwrap();
        let mask_fs = paint_mask_fs::load(device.clone()).unwrap().entry_point("main").unwrap();
        let mask_pipeline = build_stroke_pipeline(
            &device,
            &mask_render_pass,
            vs.clone(),
            mask_fs.clone(),
            StrokeBlend::Over,
        );
        let mask_erase_pipeline =
            build_stroke_pipeline(&device, &mask_render_pass, vs.clone(), mask_fs, StrokeBlend::Erase);
        let stroke_shaders = StrokeShaders {
            vs: vs.clone(),
            paint: fs,
            smudge: paint_smudge_fs::load(device.clone()).unwrap().entry_point("main").unwrap(),
            clone: paint_clone_fs::load(device.clone()).unwrap().entry_point("main").unwrap(),
            mark: paint_mark_fs::load(device.clone()).unwrap().entry_point("main").unwrap(),
        };
        let gbuf_render_pass = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                uv: {
                    format: Format::R32G32B32A32_SFLOAT,
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },
                depth: {
                    format: Format::D32_SFLOAT,
                    samples: 1,
                    load_op: Clear,
                    store_op: DontCare,
                },
            },
            pass: {
                color: [uv],
                depth_stencil: {depth},
            },
        )
        .unwrap();
        let gbuf_pipeline = build_gbuf_pipeline(&device, &gbuf_render_pass);
        let mask_mark_pipeline = build_stroke_pipeline(
            &device,
            &mask_render_pass,
            vs,
            paint_mask_mark_fs::load(device.clone()).unwrap().entry_point("main").unwrap(),
            StrokeBlend::Over,
        );

        let ubo_allocator = SubbufferAllocator::new(
            gpu.memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        let stamp_allocator = SubbufferAllocator::new(
            gpu.memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::STORAGE_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        let marks_allocator = SubbufferAllocator::new(
            gpu.memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::STORAGE_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

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
        // Clamp to the edge, not to a border colour: filtering the edge texels
        // against an opaque border painted a frame round every stamp. (The
        // 1x1 white plain-brush tip stays white either way.)
        let tip_sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                // Tips carry mip chains; the paint shader picks the level.
                mipmap_mode: vulkano::image::sampler::SamplerMipmapMode::Linear,
                lod: 0.0..=vulkano::image::sampler::LOD_CLAMP_NONE,
                ..Default::default()
            },
        )
        .unwrap();
        let nearest_sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Nearest,
                min_filter: Filter::Nearest,
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                ..Default::default()
            },
        )
        .unwrap();
        let smudge_sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                ..Default::default()
            },
        )
        .unwrap();
        let white_tip = crate::renderer::upload_texture(
            &gpu.queue,
            &gpu.memory_allocator,
            &gpu.command_buffer_allocator,
            1,
            1,
            &[255, 255, 255, 255],
            vulkano::format::Format::R8G8B8A8_UNORM,
        );

        let mut system = Self {
            gpu,
            stroke_shaders,
            variants: Default::default(),
            mask_render_pass,
            mask_pipeline,
            mask_erase_pipeline,
            smudge_scratch: None,
            smudge_sampler,
            gbuf_render_pass,
            gbuf_pipeline,
            gbuf: None,
            nearest_sampler,
            mask_mark_pipeline,
            ubo_allocator,
            stamp_allocator,
            marks_allocator,
            depth_sampler,
            tip_sampler,
            white_tip,
        };
        // Every channel present: the common case once a layer is in use.
        system.variant(ALL_SLOTS);
        system
    }

    /// The material pass for the attachments in `present`, built once.
    fn variant(&mut self, present: u8) -> Arc<StrokeVariant> {
        let device = self.gpu.queue.device().clone();
        let shaders = &self.stroke_shaders;
        self.variants
            .entry(present)
            .or_insert_with(|| {
                let render_pass = stroke_render_pass(&device, present);
                let build = |fs: &EntryPoint, blend| build_stroke_pipeline(&device, &render_pass, shaders.vs.clone(), fs.clone(), blend);
                Arc::new(StrokeVariant {
                    paint: build(&shaders.paint, StrokeBlend::Over),
                    erase: build(&shaders.paint, StrokeBlend::Erase),
                    smudge: build(&shaders.smudge, StrokeBlend::Replace),
                    clone: build(&shaders.clone, StrokeBlend::Over),
                    mark: build(&shaders.mark, StrokeBlend::Over),
                    render_pass,
                })
            })
            .clone()
    }

    /// The pass for these attachments (see `LayerStack::paint_attachments`)
    /// and the present ones in order; None when there are none at all.
    fn variant_for(
        &mut self,
        attachments: &[Option<Arc<vulkano::image::view::ImageView>>; 6],
    ) -> Option<(Arc<StrokeVariant>, Vec<Arc<vulkano::image::view::ImageView>>)> {
        let present = attachments.iter().enumerate().fold(0u8, |m, (i, a)| m | ((a.is_some() as u8) << i));
        if present == 0 {
            return None;
        }
        Some((self.variant(present), attachments.iter().flatten().cloned().collect()))
    }

    /// Apply a run of brush stamps (viewport pixel positions) to the active
    /// layer of one texture set; only that set's models receive paint.
    /// Executes synchronously; the compositor picks the change up next frame.
    pub fn stroke(
        &mut self,
        renderer: &SceneRenderer,
        set: &mut TextureSet,
        set_index: usize,
        view: &PaintView,
        brush: &BrushSettings,
        stamps: &[Vec2],
        // Texel rect [x, y, w, h] the stroke can possibly touch. Rasterising the
        // whole UV layout for a 40px brush is what made dragging lag, so callers
        // pass the region the brush actually projects onto; None paints it all.
        scissor: Option<[u32; 4]>,
        // One flag per triangle for polygon fill, in the same primitive order
        // the shader indexes. Empty means "not polygon filling".
        marks: &[u32],
    ) {
        let models: Vec<_> = renderer
            .models()
            .iter()
            .filter(|m| m.set_index == set_index)
            .collect();
        if stamps.is_empty() || models.is_empty() {
            return;
        }
        let Some(depth_view) = renderer.depth_view() else {
            return; // nothing rendered yet
        };

        let layers = &mut set.layers;
        // Mask strokes go through a separate one-attachment pass; without a
        // mask on the layer there is nothing to paint into.
        let mask_mode = brush.target == PaintTarget::Mask;
        if !mask_mode {
            layers.ensure_channels(&brush.material_channels());
        }
        let layer = layers.active_layer();
        let (framebuffer, pipeline, attachment_count, mark_pipeline) = if mask_mode {
            let Some(mask) = layer.mask.clone() else {
                return;
            };
            let fb = Framebuffer::new(
                self.mask_render_pass.clone(),
                FramebufferCreateInfo { attachments: vec![mask], ..Default::default() },
            )
            .unwrap();
            let pipeline =
                if brush.op == BrushOp::Erase { &self.mask_erase_pipeline } else { &self.mask_pipeline };
            (fb, pipeline.clone(), 1, self.mask_mark_pipeline.clone())
        } else {
            let attachments = layers.paint_attachments();
            let Some((variant, attachments)) = self.variant_for(&attachments) else {
                return; // nothing on the layer to write and every channel off
            };
            let count = attachments.len();
            let fb = Framebuffer::new(
                variant.render_pass.clone(),
                FramebufferCreateInfo { attachments, ..Default::default() },
            )
            .unwrap();
            let pipeline = if brush.op == BrushOp::Erase { &variant.erase } else { &variant.paint };
            (fb, pipeline.clone(), count, variant.mark.clone())
        };

        let res = layers.resolution;
        let scissor_rect = match scissor {
            Some([x, y, w, h]) if w > 0 && h > 0 => vulkano::pipeline::graphics::viewport::Scissor {
                offset: [x.min(res), y.min(res)],
                extent: [w.min(res - x.min(res)), h.min(res - y.min(res))],
            },
            // No hint: fall back to the whole target.
            _ => vulkano::pipeline::graphics::viewport::Scissor {
                offset: [0, 0],
                extent: [res, res],
            },
        };
        let resolution = layers.resolution as f32;
        let color_linear = srgb_to_linear(brush.color);
        let tip = brush.tip.clone().unwrap_or_else(|| self.white_tip.clone());
        let mirror = Mat4::from_scale(Vec3::new(-1.0, 1.0, 1.0));
        let passes: &[Mat4] = if brush.symmetry_x {
            &[Mat4::IDENTITY, mirror]
        } else {
            &[Mat4::IDENTITY]
        };

        let mut builder = AutoCommandBufferBuilder::primary(
            self.gpu.command_buffer_allocator.clone(),
            self.gpu.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        // Every stamp in this segment is evaluated inside a single fragment
        // pass. The expensive work -- projecting the texel back into the
        // camera, the occlusion fetch and the backface test -- is per texel,
        // not per stamp, so a fast drag no longer rasterises the whole UV
        // layout once per stamp.
        let stamp_buffer =
            self.stamp_allocator.allocate_slice::<[f32; 2]>(stamps.len() as u64).unwrap();
        {
            let mut writer = stamp_buffer.write().unwrap();
            for (dst, stamp) in writer.iter_mut().zip(stamps) {
                *dst = [stamp.x, stamp.y];
            }
        }

        // Polygon fill reads one flag per triangle; the binding must stay valid
        // either way, so keep at least one element.
        // The shader indexes marks by `primitive_offset + gl_PrimitiveID`, so a
        // short buffer would read out of bounds. Rather than risk that, fall
        // back to ordinary brushing.
        let total_triangles: usize = models.iter().map(|m| m.index_count as usize / 3).sum();
        // Triangle mode flags polygons on the GPU, where the depth test can tell
        // a visible polygon from one hidden behind it. The group modes get their
        // flags from the caller, which picks by ray and so is already occlusion
        // correct.
        let gpu_mark = brush.fill_mode == FillMode::Triangle && total_triangles > 0;
        let polygon_fill =
            brush.fill_mode.is_on() && (gpu_mark || marks.len() >= total_triangles);
        if brush.fill_mode.is_on() && !polygon_fill && !marks.is_empty() {
            eprintln!(
                "[paint] fill marks cover {} of {total_triangles} triangles; painting normally",
                marks.len()
            );
        }
        let marks_len = if gpu_mark { total_triangles } else { marks.len() };
        let marks_buffer = self
            .marks_allocator
            .allocate_slice::<u32>(marks_len.max(1) as u64)
            .unwrap();
        {
            let mut writer = marks_buffer.write().unwrap();
            writer.fill(0);
            for (dst, src) in writer.iter_mut().zip(marks) {
                *dst = *src;
            }
        }

        // Mark pass. Each draw gets its own render pass instance: a writable
        // SSBO shared by two draws inside one instance is an unsolvable hazard,
        // and barriers cannot be inserted mid-render-pass.
        if gpu_mark {
            let mut primitive_offset = 0u32;
            for model in &models {
                for pass in passes {
                    let ubo = build_ubo(
                        view,
                        brush,
                        model,
                        pass,
                        color_linear,
                        stamps.len() as u32,
                        primitive_offset,
                        1.0,
                    );
                    let ubo_buffer =
                        self.ubo_allocator.allocate_sized::<paint_vs::Ubo>().unwrap();
                    *ubo_buffer.write().unwrap() = ubo;
                    let set = DescriptorSet::new(
                        self.gpu.descriptor_set_allocator.clone(),
                        mark_pipeline.layout().set_layouts()[0].clone(),
                        [
                            WriteDescriptorSet::buffer(0, ubo_buffer),
                            WriteDescriptorSet::image_view_sampler(
                                1,
                                depth_view.clone(),
                                self.depth_sampler.clone(),
                            ),
                            WriteDescriptorSet::image_view_sampler(
                                2,
                                tip.clone(),
                                self.tip_sampler.clone(),
                            ),
                            WriteDescriptorSet::buffer(3, stamp_buffer.clone()),
                            WriteDescriptorSet::buffer(4, marks_buffer.clone()),
                        ],
                        [],
                    )
                    .unwrap();
                    builder
                        .begin_render_pass(
                            RenderPassBeginInfo {
                                clear_values: vec![None; attachment_count],
                                ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                            },
                            Default::default(),
                        )
                        .unwrap()
                        .set_viewport(
                            0,
                            [Viewport {
                                offset: [0.0, 0.0],
                                extent: [resolution, resolution],
                                depth_range: 0.0..=1.0,
                            }]
                            .into_iter()
                            .collect(),
                        )
                        .unwrap()
                        .set_scissor(0, [scissor_rect].into_iter().collect())
                        .unwrap()
                        .bind_pipeline_graphics(mark_pipeline.clone())
                        .unwrap()
                        .bind_descriptor_sets(
                            PipelineBindPoint::Graphics,
                            mark_pipeline.layout().clone(),
                            0,
                            set,
                        )
                        .unwrap()
                        .bind_vertex_buffers(0, model.vertex_buffer.clone())
                        .unwrap()
                        .bind_index_buffer(model.index_buffer.clone())
                        .unwrap();
                    unsafe {
                        builder.draw_indexed(model.index_count, 1, 0, 0, 0).unwrap();
                    }
                    builder.end_render_pass(Default::default()).unwrap();
                }
                primitive_offset += model.index_count / 3;
            }
        }

        let modes: &[f32] = if polygon_fill { &[2.0] } else { &[0.0] };

        for &mode in modes {
            builder
                .begin_render_pass(
                    RenderPassBeginInfo {
                        clear_values: vec![None; attachment_count],
                        ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                    },
                    Default::default(),
                )
                .unwrap()
                .set_viewport(
                    0,
                    [Viewport {
                        offset: [0.0, 0.0],
                        extent: [resolution, resolution],
                        depth_range: 0.0..=1.0,
                    }]
                    .into_iter()
                    .collect(),
                )
                .unwrap()
                .set_scissor(0, [scissor_rect].into_iter().collect())
                .unwrap()
                .bind_pipeline_graphics(pipeline.clone())
                .unwrap();

            let mut primitive_offset = 0u32;
            for model in &models {
                for pass in passes {
                    let ubo = build_ubo(
                        view,
                        brush,
                        model,
                        pass,
                        color_linear,
                        stamps.len() as u32,
                        primitive_offset,
                        mode,
                    );
                    let ubo_buffer =
                        self.ubo_allocator.allocate_sized::<paint_vs::Ubo>().unwrap();
                    *ubo_buffer.write().unwrap() = ubo;

                    let set = DescriptorSet::new(
                        self.gpu.descriptor_set_allocator.clone(),
                        pipeline.layout().set_layouts()[0].clone(),
                        [
                            WriteDescriptorSet::buffer(0, ubo_buffer),
                            WriteDescriptorSet::image_view_sampler(
                                1,
                                depth_view.clone(),
                                self.depth_sampler.clone(),
                            ),
                            WriteDescriptorSet::image_view_sampler(
                                2,
                                tip.clone(),
                                self.tip_sampler.clone(),
                            ),
                            WriteDescriptorSet::buffer(3, stamp_buffer.clone()),
                            WriteDescriptorSet::buffer(4, marks_buffer.clone()),
                        ],
                        [],
                    )
                    .unwrap();

                    builder
                        .bind_descriptor_sets(
                            PipelineBindPoint::Graphics,
                            pipeline.layout().clone(),
                            0,
                            set,
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
                primitive_offset += model.index_count / 3;
            }

            builder.end_render_pass(Default::default()).unwrap();
        }
        builder
            .build()
            .unwrap()
            .execute(self.gpu.queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        if brush.op == BrushOp::Paint && !mask_mode {
            use crate::layers::PaintChannel as C;
            let active = layers.active;
            let layer = &mut layers.layers[active];
            for (on, ch) in [
                (brush.paint_color, C::BaseColor),
                (brush.paint_roughness, C::Roughness),
                (brush.paint_metallic, C::Metallic),
                (brush.paint_height, C::Height),
                (brush.paint_emissive, C::Emissive),
                (brush.paint_opacity, C::Opacity),
            ] {
                if on {
                    layer.used |= ch.bit();
                }
            }
        }
        layers.mark_dirty();
    }
}

impl PaintSystem {
    /// Drag the active layer's material channels along the stroke's motion.
    ///
    /// `stamps` are (centre, step) pairs in viewport pixels, where step is how
    /// far the brush moved to reach that stamp. Each covered texel pulls its
    /// value from behind the motion, `flow` scaling how hard. Channels whose
    /// toggles are off are written back unchanged. Mask targets are ignored.
    #[allow(clippy::too_many_arguments)]
    pub fn smudge(
        &mut self,
        renderer: &SceneRenderer,
        set: &mut TextureSet,
        set_index: usize,
        view: &PaintView,
        brush: &BrushSettings,
        stamps: &[(Vec2, Vec2)],
        scissor: Option<[u32; 4]>,
    ) {
        let models: Vec<_> =
            renderer.models().iter().filter(|m| m.set_index == set_index).collect();
        if stamps.is_empty() || models.is_empty() || brush.target == PaintTarget::Mask {
            return;
        }
        let Some(depth_view) = renderer.depth_view() else {
            return;
        };
        let layers = &mut set.layers;
        let res = layers.resolution;
        layers.ensure_channels(&brush.material_channels());
        let present = layers.paint_attachments();
        let Some((variant, targets)) = self.variant_for(&present) else { return };

        if self.smudge_scratch.as_ref().is_none_or(|(r, _)| *r != res) {
            let views = SLOT_FORMATS
                .iter()
                .map(|&format| {
                    vulkano::image::view::ImageView::new_default(
                        crate::vram::image(
                            self.gpu.memory_allocator.clone(),
                            vulkano::image::ImageCreateInfo {
                                image_type: vulkano::image::ImageType::Dim2d,
                                format,
                                extent: [res, res, 1],
                                usage: vulkano::image::ImageUsage::SAMPLED
                                    | vulkano::image::ImageUsage::TRANSFER_DST,
                                ..Default::default()
                            },
                            crate::vram::Kind::Scratch,
                        )
                        .unwrap(),
                    )
                    .unwrap()
                })
                .collect();
            self.smudge_scratch = Some((res, views));
        }
        let scratch = self.smudge_scratch.as_ref().unwrap().1.clone();

        // Only the scissored region is copied and written; the shader clamps
        // its source lookups to the same rect so it never reads stale texels.
        let [rx, ry, rw, rh] = match scissor {
            Some([x, y, w, h]) if w > 0 && h > 0 => {
                let (x, y) = (x.min(res - 1), y.min(res - 1));
                [x, y, w.min(res - x), h.min(res - y)]
            }
            _ => [0, 0, res, res],
        };

        let mut builder = AutoCommandBufferBuilder::primary(
            self.gpu.command_buffer_allocator.clone(),
            self.gpu.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        // Absent slots are never written, so their scratch is never read.
        for (src, dst) in present.iter().zip(&scratch).filter_map(|(src, dst)| Some((src.as_ref()?, dst))) {
            let mut info = vulkano::command_buffer::CopyImageInfo::images(
                src.image().clone(),
                dst.image().clone(),
            );
            let region = &mut info.regions[0];
            region.src_offset = [rx, ry, 0];
            region.dst_offset = [rx, ry, 0];
            region.extent = [rw, rh, 1];
            builder.copy_image(info).unwrap();
        }

        let stamp_buffer =
            self.stamp_allocator.allocate_slice::<[f32; 4]>(stamps.len() as u64).unwrap();
        {
            let mut writer = stamp_buffer.write().unwrap();
            for (dst, (centre, step)) in writer.iter_mut().zip(stamps) {
                *dst = [centre.x, centre.y, step.x, step.y];
            }
        }

        let attachment_count = targets.len();
        let framebuffer = Framebuffer::new(
            variant.render_pass.clone(),
            FramebufferCreateInfo { attachments: targets, ..Default::default() },
        )
        .unwrap();
        let pipeline = variant.smudge.clone();
        let mirror = Mat4::from_scale(Vec3::new(-1.0, 1.0, 1.0));
        let passes: &[Mat4] =
            if brush.symmetry_x { &[Mat4::IDENTITY, mirror] } else { &[Mat4::IDENTITY] };
        let color_linear = srgb_to_linear(brush.color);
        let tip = brush.tip.clone().unwrap_or_else(|| self.white_tip.clone());

        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![None; attachment_count],
                    ..RenderPassBeginInfo::framebuffer(framebuffer)
                },
                Default::default(),
            )
            .unwrap()
            .set_viewport(
                0,
                [Viewport {
                    offset: [0.0, 0.0],
                    extent: [res as f32, res as f32],
                    depth_range: 0.0..=1.0,
                }]
                .into_iter()
                .collect(),
            )
            .unwrap()
            .set_scissor(
                0,
                [vulkano::pipeline::graphics::viewport::Scissor {
                    offset: [rx, ry],
                    extent: [rw, rh],
                }]
                .into_iter()
                .collect(),
            )
            .unwrap()
            .bind_pipeline_graphics(pipeline.clone())
            .unwrap()
            .push_constants(
                pipeline.layout().clone(),
                0,
                paint_smudge_fs::Push {
                    rect: [rx as f32, ry as f32, (rx + rw) as f32, (ry + rh) as f32],
                },
            )
            .unwrap();

        for model in &models {
            for pass in passes {
                let ubo = build_ubo(
                    view,
                    brush,
                    model,
                    pass,
                    color_linear,
                    stamps.len() as u32,
                    0,
                    0.0,
                );
                let ubo_buffer = self.ubo_allocator.allocate_sized::<paint_vs::Ubo>().unwrap();
                *ubo_buffer.write().unwrap() = ubo;
                let mut writes = vec![
                    WriteDescriptorSet::buffer(0, ubo_buffer),
                    WriteDescriptorSet::image_view_sampler(
                        1,
                        depth_view.clone(),
                        self.depth_sampler.clone(),
                    ),
                    WriteDescriptorSet::image_view_sampler(2, tip.clone(), self.tip_sampler.clone()),
                    WriteDescriptorSet::buffer(3, stamp_buffer.clone()),
                ];
                for (i, view) in scratch.iter().enumerate() {
                    writes.push(WriteDescriptorSet::image_view_sampler(
                        5 + i as u32,
                        view.clone(),
                        self.smudge_sampler.clone(),
                    ));
                }
                let set = DescriptorSet::new(
                    self.gpu.descriptor_set_allocator.clone(),
                    pipeline.layout().set_layouts()[0].clone(),
                    writes,
                    [],
                )
                .unwrap();
                builder
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        pipeline.layout().clone(),
                        0,
                        set,
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
        builder
            .build()
            .unwrap()
            .execute(self.gpu.queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
        layers.mark_dirty();
    }
}

/// What the clone tool copies from.
pub enum CloneSource {
    /// The material as you see it: base colour = the material's own colour
    /// and texture under the painted stack; other channels from the stack.
    Material,
    /// A flattened shader mask (greyscale), when painting a mask layer.
    ShaderMask(Arc<vulkano::image::view::ImageView>),
}

impl PaintSystem {
    /// Clone: every texel under the brush takes the value found on screen at
    /// its own position + `offset` (viewport px). Writes into the active
    /// layer's enabled channels with ordinary "over" blending, so flow and
    /// falloff work as for painting. Only the active set is a source.
    #[allow(clippy::too_many_arguments)]
    pub fn clone_stroke(
        &mut self,
        renderer: &SceneRenderer,
        set: &mut TextureSet,
        set_index: usize,
        view: &PaintView,
        brush: &BrushSettings,
        stamps: &[Vec2],
        scissor: Option<[u32; 4]>,
        offset: Vec2,
        source: CloneSource,
    ) {
        let models: Vec<_> =
            renderer.models().iter().filter(|m| m.set_index == set_index).collect();
        if stamps.is_empty() || models.is_empty() || brush.target == PaintTarget::Mask {
            return;
        }
        let Some(depth_view) = renderer.depth_view() else {
            return;
        };
        let size = [view.viewport_size.x.max(1.0) as u32, view.viewport_size.y.max(1.0) as u32];
        if self.gbuf.as_ref().is_none_or(|(s, _, _)| *s != size) {
            let make = |format: Format, usage: vulkano::image::ImageUsage| {
                vulkano::image::view::ImageView::new_default(
                    crate::vram::image(
                        self.gpu.memory_allocator.clone(),
                        vulkano::image::ImageCreateInfo {
                            image_type: vulkano::image::ImageType::Dim2d,
                            format,
                            extent: [size[0], size[1], 1],
                            usage,
                            ..Default::default()
                        },
                        crate::vram::Kind::Scratch,
                    )
                    .unwrap(),
                )
                .unwrap()
            };
            let uv = make(
                Format::R32G32B32A32_SFLOAT,
                vulkano::image::ImageUsage::COLOR_ATTACHMENT | vulkano::image::ImageUsage::SAMPLED,
            );
            let depth = make(Format::D32_SFLOAT, vulkano::image::ImageUsage::DEPTH_STENCIL_ATTACHMENT);
            self.gbuf = Some((size, uv, depth));
        }
        let (_, gbuf_uv, gbuf_depth) = self.gbuf.clone().unwrap();

        let mut builder = AutoCommandBufferBuilder::primary(
            self.gpu.command_buffer_allocator.clone(),
            self.gpu.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        // 1. Where is each surface point on screen: UV per viewport pixel.
        let gbuf_fb = Framebuffer::new(
            self.gbuf_render_pass.clone(),
            FramebufferCreateInfo {
                attachments: vec![gbuf_uv.clone(), gbuf_depth],
                ..Default::default()
            },
        )
        .unwrap();
        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some([0.0f32; 4].into()), Some(1.0f32.into())],
                    ..RenderPassBeginInfo::framebuffer(gbuf_fb)
                },
                Default::default(),
            )
            .unwrap()
            .set_viewport(
                0,
                [Viewport {
                    offset: [0.0, 0.0],
                    extent: [size[0] as f32, size[1] as f32],
                    depth_range: 0.0..=1.0,
                }]
                .into_iter()
                .collect(),
            )
            .unwrap()
            .bind_pipeline_graphics(self.gbuf_pipeline.clone())
            .unwrap();
        for model in &models {
            builder
                .push_constants(
                    self.gbuf_pipeline.layout().clone(),
                    0,
                    gbuf_vs::Push { mvp: (view.view_proj * model.transform).to_cols_array_2d() },
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

        // 2. The clone itself, rasterised in UV space like a paint stroke.
        let layers = &mut set.layers;
        let res = layers.resolution;
        layers.ensure_channels(&brush.material_channels());
        let Some((variant, attachments)) = self.variant_for(&layers.paint_attachments()) else { return };
        let attachment_count = attachments.len();
        let framebuffer = Framebuffer::new(
            variant.render_pass.clone(),
            FramebufferCreateInfo { attachments, ..Default::default() },
        )
        .unwrap();
        let scissor_rect = match scissor {
            Some([x, y, w, h]) if w > 0 && h > 0 => {
                let (x, y) = (x.min(res - 1), y.min(res - 1));
                vulkano::pipeline::graphics::viewport::Scissor {
                    offset: [x, y],
                    extent: [w.min(res - x), h.min(res - y)],
                }
            }
            _ => vulkano::pipeline::graphics::viewport::Scissor {
                offset: [0, 0],
                extent: [res, res],
            },
        };
        let stamp_buffer =
            self.stamp_allocator.allocate_slice::<[f32; 2]>(stamps.len() as u64).unwrap();
        {
            let mut writer = stamp_buffer.write().unwrap();
            for (dst, stamp) in writer.iter_mut().zip(stamps) {
                *dst = [stamp.x, stamp.y];
            }
        }
        let tip = brush.tip.clone().unwrap_or_else(|| self.white_tip.clone());
        let (mode, colour_source) = match &source {
            CloneSource::Material => (0.0, layers.composite_basecolor.clone()),
            CloneSource::ShaderMask(view) => (1.0, view.clone()),
        };
        let pipeline = variant.clone.clone();
        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![None; attachment_count],
                    ..RenderPassBeginInfo::framebuffer(framebuffer)
                },
                Default::default(),
            )
            .unwrap()
            // Bind first: the G-buffer pipeline still bound from the pass
            // above has a fixed scissor, so setting one before rebinding fails.
            .bind_pipeline_graphics(pipeline.clone())
            .unwrap()
            .set_viewport(
                0,
                [Viewport {
                    offset: [0.0, 0.0],
                    extent: [res as f32, res as f32],
                    depth_range: 0.0..=1.0,
                }]
                .into_iter()
                .collect(),
            )
            .unwrap()
            .set_scissor(0, [scissor_rect].into_iter().collect())
            .unwrap();
        for model in &models {
            let ubo = build_ubo(
                view,
                brush,
                model,
                &Mat4::IDENTITY,
                srgb_to_linear(brush.color),
                stamps.len() as u32,
                0,
                0.0,
            );
            let ubo_buffer = self.ubo_allocator.allocate_sized::<paint_vs::Ubo>().unwrap();
            *ubo_buffer.write().unwrap() = ubo;
            let set = DescriptorSet::new(
                self.gpu.descriptor_set_allocator.clone(),
                pipeline.layout().set_layouts()[0].clone(),
                [
                    WriteDescriptorSet::buffer(0, ubo_buffer),
                    WriteDescriptorSet::image_view_sampler(
                        1,
                        depth_view.clone(),
                        self.depth_sampler.clone(),
                    ),
                    WriteDescriptorSet::image_view_sampler(2, tip.clone(), self.tip_sampler.clone()),
                    WriteDescriptorSet::buffer(3, stamp_buffer.clone()),
                    WriteDescriptorSet::image_view_sampler(
                        5,
                        gbuf_uv.clone(),
                        self.nearest_sampler.clone(),
                    ),
                    WriteDescriptorSet::image_view_sampler(
                        6,
                        colour_source.clone(),
                        layers.sampler.clone(),
                    ),
                    WriteDescriptorSet::image_view_sampler(
                        7,
                        layers.composite_roughness.clone(),
                        layers.sampler.clone(),
                    ),
                    WriteDescriptorSet::image_view_sampler(
                        8,
                        layers.composite_metallic.clone(),
                        layers.sampler.clone(),
                    ),
                    WriteDescriptorSet::image_view_sampler(
                        9,
                        layers.composite_height.clone(),
                        layers.sampler.clone(),
                    ),
                    WriteDescriptorSet::image_view_sampler(
                        10,
                        layers.composite_emissive.clone(),
                        layers.sampler.clone(),
                    ),
                    WriteDescriptorSet::image_view_sampler(
                        11,
                        layers.composite_opacity.clone(),
                        layers.sampler.clone(),
                    ),
                    WriteDescriptorSet::image_view_sampler(
                        12,
                        model.textures[0].clone(),
                        self.smudge_sampler.clone(),
                    ),
                ],
                [],
            )
            .unwrap();
            builder
                .push_constants(
                    pipeline.layout().clone(),
                    0,
                    paint_clone_fs::Push {
                        offset_mode: [offset.x, offset.y, mode, 0.0],
                        base_factor: model.base_color,
                    },
                )
                .unwrap()
                .bind_descriptor_sets(PipelineBindPoint::Graphics, pipeline.layout().clone(), 0, set)
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
        builder
            .build()
            .unwrap()
            .execute(self.gpu.queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
        layers.mark_dirty();
    }
}

/// Camera-view UV pass for the clone tool: depth-tested, no blending.
fn build_gbuf_pipeline(
    device: &Arc<vulkano::device::Device>,
    render_pass: &Arc<RenderPass>,
) -> Arc<GraphicsPipeline> {
    let vs = gbuf_vs::load(device.clone()).unwrap().entry_point("main").unwrap();
    let fs = gbuf_fs::load(device.clone()).unwrap().entry_point("main").unwrap();
    let vertex_input_state = Vertex3D::per_vertex().definition(&vs).unwrap();
    let stages = [PipelineShaderStageCreateInfo::new(vs), PipelineShaderStageCreateInfo::new(fs)];
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
            depth_stencil_state: Some(
                vulkano::pipeline::graphics::depth_stencil::DepthStencilState {
                    depth: Some(vulkano::pipeline::graphics::depth_stencil::DepthState::simple()),
                    ..Default::default()
                },
            ),
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
}

fn srgb_to_linear(c: [f32; 3]) -> [f32; 3] {
    c.map(|v| {
        if v <= 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    })
}


/// The uniform block both the mark and paint passes fill in.
#[allow(clippy::too_many_arguments)]
fn build_ubo(
    view: &PaintView,
    brush: &BrushSettings,
    model: &crate::renderer::GpuModel,
    pass: &Mat4,
    color_linear: [f32; 3],
    stamp_count: u32,
    primitive_offset: u32,
    mode: f32,
) -> paint_vs::Ubo {
    paint_vs::Ubo {
        view_proj: view.view_proj.to_cols_array_2d(),
        model: (*pass * model.transform).to_cols_array_2d(),
        camera_pos: view.camera_pos.extend(1.0).to_array(),
        brush: [
            if brush.square_tip { 1.0 } else { 0.0 },
            brush.tip_aspect,
            brush.size,
            brush.hardness,
        ],
        color: [color_linear[0], color_linear[1], color_linear[2], brush.flow],
        values: [
            brush.roughness_value,
            brush.metallic_value,
            brush.height_value,
            brush.opacity_value,
        ],
        viewport: [view.viewport_size.x, view.viewport_size.y, 0.0, 0.0],
        emissive: {
            let e = srgb_to_linear(brush.emissive_color);
            [
                e[0] * brush.emissive_intensity,
                e[1] * brush.emissive_intensity,
                e[2] * brush.emissive_intensity,
                0.0,
            ]
        },
        channels: [
            brush.paint_color as u32,
            brush.paint_roughness as u32,
            brush.paint_metallic as u32,
            brush.paint_height as u32,
        ],
        channels2: [
            brush.paint_emissive as u32,
            brush.paint_opacity as u32,
            stamp_count,
            primitive_offset,
        ],
        extra: [
            brush.mask_value,
            mode,
            if brush.use_tip_color { 1.0 } else { 0.0 },
            brush.tip_rotation,
        ],
    }
}

/// How a stroke pipeline combines its output with the layer texel.
#[derive(Clone, Copy, PartialEq, Eq)]
enum StrokeBlend {
    /// Premultiplied "over": paint accumulates.
    Over,
    /// Destination-out: dst *= 1 - coverage.
    Erase,
    /// No blending; the shader computes the final texel itself (smudge).
    Replace,
}

/// Formats of the material pass slots: colour, roughness, metallic, height,
/// emissive, opacity (the paint shaders' output locations 0..5).
const SLOT_FORMATS: [Format; 6] = [
    Format::R16G16B16A16_SFLOAT,
    Format::R16G16_SFLOAT,
    Format::R16G16_SFLOAT,
    Format::R16G16_SFLOAT,
    Format::R16G16B16A16_SFLOAT,
    Format::R16G16_SFLOAT,
];
const ALL_SLOTS: u8 = 0b11_1111;

struct StrokeShaders {
    vs: EntryPoint,
    paint: EntryPoint,
    smudge: EntryPoint,
    clone: EntryPoint,
    mark: EntryPoint,
}

/// One material pass and its pipelines. Layer channels that were never
/// painted have no texture (see `LayerStack::paint_attachments`); their slots
/// are `VK_ATTACHMENT_UNUSED` here, so the shaders' writes to them are
/// discarded rather than needing a full-size image to land in.
struct StrokeVariant {
    render_pass: Arc<RenderPass>,
    paint: Arc<GraphicsPipeline>,
    /// Same shader as `paint`, blended destination-out.
    erase: Arc<GraphicsPipeline>,
    /// Reads the layer from `smudge_scratch` and writes the dragged result.
    smudge: Arc<GraphicsPipeline>,
    clone: Arc<GraphicsPipeline>,
    mark: Arc<GraphicsPipeline>,
}

/// The material pass with attachments only for the slots in `present`.
fn stroke_render_pass(device: &Arc<Device>, present: u8) -> Arc<RenderPass> {
    use vulkano::image::{ImageLayout, SampleCount};
    use vulkano::render_pass::{
        AttachmentDescription, AttachmentLoadOp, AttachmentReference, AttachmentStoreOp, RenderPassCreateInfo,
        SubpassDescription,
    };
    let mut attachments = Vec::new();
    let mut color_attachments = Vec::new();
    for (slot, format) in SLOT_FORMATS.into_iter().enumerate() {
        if present & (1 << slot) == 0 {
            color_attachments.push(None);
            continue;
        }
        color_attachments.push(Some(AttachmentReference {
            attachment: attachments.len() as u32,
            layout: ImageLayout::ColorAttachmentOptimal,
            ..Default::default()
        }));
        attachments.push(AttachmentDescription {
            format,
            samples: SampleCount::Sample1,
            load_op: AttachmentLoadOp::Load,
            store_op: AttachmentStoreOp::Store,
            initial_layout: ImageLayout::ColorAttachmentOptimal,
            final_layout: ImageLayout::ColorAttachmentOptimal,
            ..Default::default()
        });
    }
    RenderPass::new(
        device.clone(),
        RenderPassCreateInfo {
            attachments,
            subpasses: vec![SubpassDescription { color_attachments, ..Default::default() }],
            ..Default::default()
        },
    )
    .unwrap()
}

/// Stroke pipelines differ only in fragment shader, target pass and blend: the
/// vertex stage rasterises the mesh in UV space either way.
fn build_stroke_pipeline(
    device: &Arc<vulkano::device::Device>,
    render_pass: &Arc<RenderPass>,
    vs: vulkano::shader::EntryPoint,
    fs: vulkano::shader::EntryPoint,
    mode: StrokeBlend,
) -> Arc<GraphicsPipeline> {
    let vertex_input_state = Vertex3D::per_vertex().definition(&vs).unwrap();
    let stages = [PipelineShaderStageCreateInfo::new(vs), PipelineShaderStageCreateInfo::new(fs)];
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
    )
    .unwrap();
    let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

    // Premultiplied alpha accumulation into the layer textures.
    // Erasing is destination-out: dst *= 1 - coverage. The shaders already put
    // coverage in .a (zero for disabled channels), so they need no erase mode,
    // and scaling a premultiplied texel keeps it premultiplied.
    let src_factor =
        if mode == StrokeBlend::Erase { BlendFactor::Zero } else { BlendFactor::One };
    let blend = AttachmentBlend {
        src_color_blend_factor: src_factor,
        dst_color_blend_factor: BlendFactor::OneMinusSrcAlpha,
        color_blend_op: BlendOp::Add,
        src_alpha_blend_factor: src_factor,
        dst_alpha_blend_factor: BlendFactor::OneMinusSrcAlpha,
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
            // Unused slots (see `StrokeVariant`) have no format to blend.
            color_blend_state: Some(ColorBlendState {
                attachments: subpass
                    .subpass_desc()
                    .color_attachments
                    .iter()
                    .map(|slot| ColorBlendAttachmentState {
                        blend: (mode != StrokeBlend::Replace && slot.is_some()).then(|| blend.clone()),
                        ..Default::default()
                    })
                    .collect(),
                ..Default::default()
            }),
            dynamic_state: [DynamicState::Viewport, DynamicState::Scissor]
                .into_iter()
                .collect(),
            subpass: Some(subpass.into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .unwrap()
}

mod paint_vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
#version 450
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;

layout(set = 0, binding = 0) uniform Ubo {
    mat4 view_proj;
    mat4 model;
    vec4 camera_pos;
    vec4 brush;    // xy: stamp center px, z: radius px, w: hardness
    vec4 color;    // rgb: linear color, a: flow
    vec4 values;   // x: roughness, y: metallic, z: height (signed), w: opacity
    vec4 viewport; // xy: viewport size px
    vec4 emissive; // rgb: linear emissive color * intensity
    uvec4 channels; // x: color, y: roughness, z: metallic, w: height
    uvec4 channels2; // x: emissive, y: opacity
    vec4 extra;    // x: mask value deposited in mask mode
} ubo;

layout(location = 0) out vec3 v_world_pos;
layout(location = 1) out vec3 v_normal;

void main() {
    vec4 world_pos = ubo.model * vec4(position, 1.0);
    v_world_pos = world_pos.xyz;
    v_normal = mat3(ubo.model) * normal;
    // Rasterize the mesh in its UV layout; fragments land on layer texels.
    gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
",
    }
}

mod paint_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
#version 450
layout(location = 0) in vec3 v_world_pos;
layout(location = 1) in vec3 v_normal;

layout(location = 0) out vec4 o_basecolor;
layout(location = 1) out vec4 o_roughness;
layout(location = 2) out vec4 o_metallic;
layout(location = 3) out vec4 o_height;
layout(location = 4) out vec4 o_emissive;
layout(location = 5) out vec4 o_opacity;

layout(set = 0, binding = 0) uniform Ubo {
    mat4 view_proj;
    mat4 model;
    vec4 camera_pos;
    vec4 brush;
    vec4 color;
    vec4 values;
    vec4 viewport;
    vec4 emissive;
    uvec4 channels;
    uvec4 channels2;
    vec4 extra;
} ubo;
layout(set = 0, binding = 1) uniform sampler2D scene_depth;
layout(set = 0, binding = 2) uniform sampler2D brush_tip;
layout(set = 0, binding = 3) readonly buffer Stamps { vec2 centre[]; } stamps;
// One flag per triangle, filled in on the CPU from the same projection the
// scissor uses. Read-only: a writable SSBO shared by two draws (which is what
// X symmetry produces) is an unsolvable hazard inside a render pass.
layout(set = 0, binding = 4) readonly buffer Marks { uint flag[]; } marks;

vec3 tip_srgb_to_linear(vec3 c) {
    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(vec3(0.04045), c));
}

// Colour and coverage from every stamp in this segment. Coverage combines as a
// union -- acc = 1 - prod(1 - a_i), what sequential premultiplied blending
// converges to -- so one pass equals N passes minus the cost. Returns colour
// already premultiplied by the accumulated coverage.
vec4 stroke_paint(vec2 px, float texel_px) {
    float radius = ubo.brush.z;
    float hardness = ubo.brush.w;
    float use_tip_color = ubo.extra.z;
    float rot = ubo.extra.w;
    float cs = cos(rot);
    float sn = sin(rot);
    // Stamp tool: the whole image, at its own aspect, no circular falloff.
    bool square = ubo.brush.x > 0.5;
    float aspect = max(ubo.brush.y, 1e-3);
    vec2 extent = radius * (aspect >= 1.0 ? vec2(1.0, 1.0 / aspect) : vec2(aspect, 1.0));
    // Tip mip level: tip texels per paint texel, so a big image stamped
    // small averages instead of breaking up into speckle.
    vec2 tip_size = vec2(textureSize(brush_tip, 0));
    vec2 per_px = square ? tip_size * 0.5 / extent : tip_size / (2.0 * radius);
    float lod = log2(max(texel_px * max(per_px.x, per_px.y), 1.0));

    vec3 rgb_acc = vec3(0.0);
    float acc = 0.0;
    uint count = ubo.channels2.z;
    for (uint i = 0u; i < count; i++) {
        vec2 c = stamps.centre[i];
        vec2 delta = px - c;
        // Rotate the lookup, not the stamp, so aiming a decal is free.
        vec2 rotated = vec2(delta.x * cs - delta.y * sn, delta.x * sn + delta.y * cs);
        float falloff;
        vec2 lookup;
        if (square) {
            vec2 q = rotated / extent;
            if (abs(q.x) > 1.0 || abs(q.y) > 1.0) continue;
            falloff = 1.0;
            lookup = q * 0.5 + 0.5;
        } else {
            float d = length(delta);
            if (d > radius) continue;
            falloff = 1.0 - smoothstep(hardness, 1.0, d / radius);
            lookup = rotated / (2.0 * radius) + 0.5;
        }
        vec4 tip = textureLod(brush_tip, lookup, lod);
        float a = ubo.color.a * falloff * tip.a;
        if (a <= 0.0) continue;
        vec3 rgb = mix(ubo.color.rgb, tip_srgb_to_linear(tip.a > 1e-4 ? tip.rgb / tip.a : vec3(0.0)), use_tip_color);
        float w = a * (1.0 - acc);
        rgb_acc += rgb * w;
        acc += w;
    }
    return vec4(rgb_acc, acc);
}

// Shared rejection: off-screen, back-facing, or hidden behind nearer geometry.
// Done once per texel rather than once per stamp.
bool visible(out vec2 px) {
    vec4 clip = ubo.view_proj * vec4(v_world_pos, 1.0);
    if (clip.w <= 0.0) return false;
    vec3 ndc = clip.xyz / clip.w;
    if (abs(ndc.x) > 1.0 || abs(ndc.y) > 1.0) return false;

    vec3 view_dir = normalize(ubo.camera_pos.xyz - v_world_pos);
    if (dot(normalize(v_normal), view_dir) <= 0.05) return false;

    vec2 depth_uv = ndc.xy * 0.5 + 0.5;
    float scene_z = texture(scene_depth, depth_uv).r;
    if (ndc.z > scene_z + 1e-3) return false;

    px = (ndc.xy * 0.5 + 0.5) * ubo.viewport.xy;
    return true;
}

void main() {
    // Screen pixels one paint texel spans (for the tip's mip level), taken
    // here in uniform control flow, before anything discards.
    vec4 clip0 = ubo.view_proj * vec4(v_world_pos, 1.0);
    vec2 screen = (clip0.xy / clip0.w * 0.5 + 0.5) * ubo.viewport.xy;
    float texel_px = max(length(dFdx(screen)), length(dFdy(screen)));
    vec2 px;
    if (!visible(px)) discard;

    uint mode = uint(ubo.extra.y + 0.5);
    uint prim = ubo.channels2.w + uint(gl_PrimitiveID);
    float alpha;

    vec3 rgb;
    if (mode == 2u) {
        // Polygon fill: flagged polygons go in whole, at full strength, so the
        // stroke stops hard on their edges.
        if (marks.flag[prim] == 0u) discard;
        alpha = 1.0;
        rgb = ubo.color.rgb * alpha;
    } else {
        vec4 paint = stroke_paint(px, texel_px);
        alpha = paint.a;
        if (alpha <= 0.001) discard;
        rgb = paint.rgb;
    }

    // Premultiplied outputs; disabled channels write zeros (no-op blend).
    o_basecolor = vec4(rgb, alpha) * float(ubo.channels.x);
    o_roughness = vec4(ubo.values.x * alpha, alpha, 0.0, alpha) * float(ubo.channels.y);
    o_metallic = vec4(ubo.values.y * alpha, alpha, 0.0, alpha) * float(ubo.channels.z);
    o_height = vec4(ubo.values.z * alpha, alpha, 0.0, alpha) * float(ubo.channels.w);
    o_emissive = vec4(ubo.emissive.rgb * alpha, alpha) * float(ubo.channels2.x);
    o_opacity = vec4(ubo.values.w * alpha, alpha, 0.0, alpha) * float(ubo.channels2.y);
}
",
    }
}

/// Mask-mode fragment stage. Identical projection/occlusion/falloff logic to
/// `paint_fs`, writing a single scalar instead of the six material channels.
/// The duplication is deliberate: `vulkano_shaders::shader!` needs one literal
/// source, so the shared part cannot be factored out.
mod paint_mask_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
#version 450
layout(location = 0) in vec3 v_world_pos;
layout(location = 1) in vec3 v_normal;

layout(location = 0) out vec4 o_mask;

layout(set = 0, binding = 0) uniform Ubo {
    mat4 view_proj;
    mat4 model;
    vec4 camera_pos;
    vec4 brush;
    vec4 color;
    vec4 values;
    vec4 viewport;
    vec4 emissive;
    uvec4 channels;
    uvec4 channels2;
    vec4 extra;
} ubo;
layout(set = 0, binding = 1) uniform sampler2D scene_depth;
layout(set = 0, binding = 2) uniform sampler2D brush_tip;
layout(set = 0, binding = 3) readonly buffer Stamps { vec2 centre[]; } stamps;
// One flag per triangle, filled in on the CPU from the same projection the
// scissor uses. Read-only: a writable SSBO shared by two draws (which is what
// X symmetry produces) is an unsolvable hazard inside a render pass.
layout(set = 0, binding = 4) readonly buffer Marks { uint flag[]; } marks;

vec3 tip_srgb_to_linear(vec3 c) {
    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(vec3(0.04045), c));
}

// Colour and coverage from every stamp in this segment. Coverage combines as a
// union -- acc = 1 - prod(1 - a_i), what sequential premultiplied blending
// converges to -- so one pass equals N passes minus the cost. Returns colour
// already premultiplied by the accumulated coverage.
vec4 stroke_paint(vec2 px, float texel_px) {
    float radius = ubo.brush.z;
    float hardness = ubo.brush.w;
    float use_tip_color = ubo.extra.z;
    float rot = ubo.extra.w;
    float cs = cos(rot);
    float sn = sin(rot);
    // Stamp tool: the whole image, at its own aspect, no circular falloff.
    bool square = ubo.brush.x > 0.5;
    float aspect = max(ubo.brush.y, 1e-3);
    vec2 extent = radius * (aspect >= 1.0 ? vec2(1.0, 1.0 / aspect) : vec2(aspect, 1.0));
    // Tip mip level: tip texels per paint texel, so a big image stamped
    // small averages instead of breaking up into speckle.
    vec2 tip_size = vec2(textureSize(brush_tip, 0));
    vec2 per_px = square ? tip_size * 0.5 / extent : tip_size / (2.0 * radius);
    float lod = log2(max(texel_px * max(per_px.x, per_px.y), 1.0));

    vec3 rgb_acc = vec3(0.0);
    float acc = 0.0;
    uint count = ubo.channels2.z;
    for (uint i = 0u; i < count; i++) {
        vec2 c = stamps.centre[i];
        vec2 delta = px - c;
        // Rotate the lookup, not the stamp, so aiming a decal is free.
        vec2 rotated = vec2(delta.x * cs - delta.y * sn, delta.x * sn + delta.y * cs);
        float falloff;
        vec2 lookup;
        if (square) {
            vec2 q = rotated / extent;
            if (abs(q.x) > 1.0 || abs(q.y) > 1.0) continue;
            falloff = 1.0;
            lookup = q * 0.5 + 0.5;
        } else {
            float d = length(delta);
            if (d > radius) continue;
            falloff = 1.0 - smoothstep(hardness, 1.0, d / radius);
            lookup = rotated / (2.0 * radius) + 0.5;
        }
        vec4 tip = textureLod(brush_tip, lookup, lod);
        float a = ubo.color.a * falloff * tip.a;
        if (a <= 0.0) continue;
        vec3 rgb = mix(ubo.color.rgb, tip_srgb_to_linear(tip.a > 1e-4 ? tip.rgb / tip.a : vec3(0.0)), use_tip_color);
        float w = a * (1.0 - acc);
        rgb_acc += rgb * w;
        acc += w;
    }
    return vec4(rgb_acc, acc);
}

// Shared rejection: off-screen, back-facing, or hidden behind nearer geometry.
// Done once per texel rather than once per stamp.
bool visible(out vec2 px) {
    vec4 clip = ubo.view_proj * vec4(v_world_pos, 1.0);
    if (clip.w <= 0.0) return false;
    vec3 ndc = clip.xyz / clip.w;
    if (abs(ndc.x) > 1.0 || abs(ndc.y) > 1.0) return false;

    vec3 view_dir = normalize(ubo.camera_pos.xyz - v_world_pos);
    if (dot(normalize(v_normal), view_dir) <= 0.05) return false;

    vec2 depth_uv = ndc.xy * 0.5 + 0.5;
    float scene_z = texture(scene_depth, depth_uv).r;
    if (ndc.z > scene_z + 1e-3) return false;

    px = (ndc.xy * 0.5 + 0.5) * ubo.viewport.xy;
    return true;
}

void main() {
    // Screen pixels one paint texel spans (for the tip's mip level), taken
    // here in uniform control flow, before anything discards.
    vec4 clip0 = ubo.view_proj * vec4(v_world_pos, 1.0);
    vec2 screen = (clip0.xy / clip0.w * 0.5 + 0.5) * ubo.viewport.xy;
    float texel_px = max(length(dFdx(screen)), length(dFdy(screen)));
    vec2 px;
    if (!visible(px)) discard;

    uint mode = uint(ubo.extra.y + 0.5);
    uint prim = ubo.channels2.w + uint(gl_PrimitiveID);
    float alpha;

    if (mode == 2u) {
        if (marks.flag[prim] == 0u) discard;
        alpha = 1.0;
    } else {
        alpha = stroke_paint(px, texel_px).a;
        if (alpha <= 0.001) discard;
    }

    // Premultiplied (value*a, a), matching the other scalar channels; the
    // compositor reads .r as the layer's coverage multiplier.
    o_mask = vec4(ubo.extra.x * alpha, alpha, 0.0, alpha);
}
",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Building the system validates both stroke pipelines, including the
    /// mask pass's single-attachment layout and shader reflection.
    #[test]
    fn both_stroke_pipelines_build() {
        let Some(gpu) = crate::environment::test_gpu::context() else {
            eprintln!("no Vulkan device; skipping");
            return;
        };
        let mut system = PaintSystem::new(gpu);
        assert_eq!(
            Subpass::from(system.mask_render_pass.clone(), 0).unwrap().num_color_attachments(),
            1,
        );
        // Both fragment stages must reflect the same descriptor layout, since
        // the stroke builds one descriptor set and uses it for either pipeline.
        assert_eq!(
            system.variant(ALL_SLOTS).paint.layout().set_layouts().len(),
            system.mask_pipeline.layout().set_layouts().len(),
        );
    }
}

/// Smudge fragment stage. Same UV-space raster and visibility test as
/// `paint_fs`, but instead of depositing values it pulls each covered texel
/// from behind the brush's motion, reading a copy of the layer.
mod paint_smudge_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
#version 450
layout(location = 0) in vec3 v_world_pos;
layout(location = 1) in vec3 v_normal;

layout(location = 0) out vec4 o_basecolor;
layout(location = 1) out vec4 o_roughness;
layout(location = 2) out vec4 o_metallic;
layout(location = 3) out vec4 o_height;
layout(location = 4) out vec4 o_emissive;
layout(location = 5) out vec4 o_opacity;

layout(set = 0, binding = 0) uniform Ubo {
    mat4 view_proj; mat4 model; vec4 camera_pos; vec4 brush; vec4 color;
    vec4 values; vec4 viewport; vec4 emissive; uvec4 channels; uvec4 channels2; vec4 extra;
} ubo;
layout(set = 0, binding = 1) uniform sampler2D scene_depth;
layout(set = 0, binding = 2) uniform sampler2D brush_tip;
// xy: stamp centre px, zw: how far the brush moved to get there.
layout(set = 0, binding = 3) readonly buffer Stamps { vec4 stamp[]; } stamps;
// Copies of the layer as it was before this segment.
layout(set = 0, binding = 5) uniform sampler2D src_basecolor;
layout(set = 0, binding = 6) uniform sampler2D src_roughness;
layout(set = 0, binding = 7) uniform sampler2D src_metallic;
layout(set = 0, binding = 8) uniform sampler2D src_height;
layout(set = 0, binding = 9) uniform sampler2D src_emissive;
layout(set = 0, binding = 10) uniform sampler2D src_opacity;
// Texel rect [x0, y0, x1, y1] that was copied; lookups stay inside it.
layout(push_constant) uniform Push { vec4 rect; } pc;

vec4 pull(sampler2D tex, ivec2 texel, vec2 src_uv, float amount, uint enabled) {
    vec4 orig = texelFetch(tex, texel, 0);
    if (enabled == 0u) return orig;
    return mix(orig, texture(tex, src_uv), amount);
}

void main() {
    // Screen position of this texel, and how it changes per texel step. The
    // derivatives must come before any discard (uniform control flow).
    vec4 clip = ubo.view_proj * vec4(v_world_pos, 1.0);
    vec2 px_raw = (clip.xy / clip.w * 0.5 + 0.5) * ubo.viewport.xy;
    vec2 dpx_dx = dFdx(px_raw);
    vec2 dpx_dy = dFdy(px_raw);

    if (clip.w <= 0.0) discard;
    vec3 ndc = clip.xyz / clip.w;
    if (abs(ndc.x) > 1.0 || abs(ndc.y) > 1.0) discard;
    vec3 view_dir = normalize(ubo.camera_pos.xyz - v_world_pos);
    if (dot(normalize(v_normal), view_dir) <= 0.05) discard;
    if (ndc.z > texture(scene_depth, ndc.xy * 0.5 + 0.5).r + 1e-3) discard;
    vec2 px = px_raw;

    // Coverage (union, as in paint_fs) and the motion that reaches this texel.
    float radius = ubo.brush.z;
    float hardness = ubo.brush.w;
    float cs = cos(ubo.extra.w);
    float sn = sin(ubo.extra.w);
    float acc = 0.0;
    vec2 push = vec2(0.0);
    uint count = ubo.channels2.z;
    for (uint i = 0u; i < count; i++) {
        vec2 delta = px - stamps.stamp[i].xy;
        float d = length(delta);
        if (d > radius) continue;
        float falloff = 1.0 - smoothstep(hardness, 1.0, d / radius);
        vec2 rotated = vec2(delta.x * cs - delta.y * sn, delta.x * sn + delta.y * cs);
        float a = ubo.color.a * falloff * textureLod(brush_tip, rotated / (2.0 * radius) + 0.5, 0.0).a;
        if (a <= 0.0) continue;
        push += stamps.stamp[i].zw * a;
        acc += a * (1.0 - acc);
    }
    if (acc <= 0.001) discard;

    // Screen-space push -> texel-space offset through the inverse Jacobian.
    mat2 j = mat2(dpx_dx, dpx_dy);
    float det = determinant(j);
    if (abs(det) < 1e-6) discard;
    vec2 src_texel = gl_FragCoord.xy - inverse(j) * push;
    src_texel = clamp(src_texel, pc.rect.xy + 0.5, pc.rect.zw - 0.5);
    vec2 src_uv = src_texel / vec2(textureSize(src_basecolor, 0));
    ivec2 texel = ivec2(gl_FragCoord.xy);

    o_basecolor = pull(src_basecolor, texel, src_uv, acc, ubo.channels.x);
    o_roughness = pull(src_roughness, texel, src_uv, acc, ubo.channels.y);
    o_metallic = pull(src_metallic, texel, src_uv, acc, ubo.channels.z);
    o_height = pull(src_height, texel, src_uv, acc, ubo.channels.w);
    o_emissive = pull(src_emissive, texel, src_uv, acc, ubo.channels2.x);
    o_opacity = pull(src_opacity, texel, src_uv, acc, ubo.channels2.y);
}
",
    }
}

mod gbuf_vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
#version 450
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;
layout(push_constant) uniform Push { mat4 mvp; } pc;
layout(location = 0) out vec2 v_uv;
void main() {
    v_uv = uv;
    gl_Position = pc.mvp * vec4(position, 1.0);
}
",
    }
}

mod gbuf_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
#version 450
layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_uv;
// z = 1 marks surface; cleared pixels are all zero.
void main() { o_uv = vec4(v_uv, 1.0, 1.0); }
",
    }
}

/// Clone fragment stage: `paint_fs`'s projection, visibility and coverage,
/// but the value comes from the surface under `px + offset` instead of the
/// brush settings.
mod paint_clone_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
#version 450
layout(location = 0) in vec3 v_world_pos;
layout(location = 1) in vec3 v_normal;

layout(location = 0) out vec4 o_basecolor;
layout(location = 1) out vec4 o_roughness;
layout(location = 2) out vec4 o_metallic;
layout(location = 3) out vec4 o_height;
layout(location = 4) out vec4 o_emissive;
layout(location = 5) out vec4 o_opacity;

layout(set = 0, binding = 0) uniform Ubo {
    mat4 view_proj; mat4 model; vec4 camera_pos; vec4 brush; vec4 color;
    vec4 values; vec4 viewport; vec4 emissive; uvec4 channels; uvec4 channels2; vec4 extra;
} ubo;
layout(set = 0, binding = 1) uniform sampler2D scene_depth;
layout(set = 0, binding = 2) uniform sampler2D brush_tip;
layout(set = 0, binding = 3) readonly buffer Stamps { vec2 centre[]; } stamps;
layout(set = 0, binding = 5) uniform sampler2D gbuf_uv;     // xy: uv, z: surface
layout(set = 0, binding = 6) uniform sampler2D src_color;   // composite or shader mask
layout(set = 0, binding = 7) uniform sampler2D src_rough;
layout(set = 0, binding = 8) uniform sampler2D src_metal;
layout(set = 0, binding = 9) uniform sampler2D src_height;
layout(set = 0, binding = 10) uniform sampler2D src_emissive;
layout(set = 0, binding = 11) uniform sampler2D src_opacity;
layout(set = 0, binding = 12) uniform sampler2D mat_basecolor; // sRGB-decoded by format
layout(push_constant) uniform Push {
    vec4 offset_mode;  // xy: source offset px, z: 0 material / 1 shader mask
    vec4 base_factor;  // material base colour factor
} pc;

float coverage(vec2 px) {
    float radius = ubo.brush.z;
    float hardness = ubo.brush.w;
    float cs = cos(ubo.extra.w);
    float sn = sin(ubo.extra.w);
    bool square = ubo.brush.x > 0.5;
    float aspect = max(ubo.brush.y, 1e-3);
    vec2 extent = radius * (aspect >= 1.0 ? vec2(1.0, 1.0 / aspect) : vec2(aspect, 1.0));
    float acc = 0.0;
    uint count = ubo.channels2.z;
    for (uint i = 0u; i < count; i++) {
        vec2 delta = px - stamps.centre[i];
        vec2 rotated = vec2(delta.x * cs - delta.y * sn, delta.x * sn + delta.y * cs);
        float falloff;
        vec2 lookup;
        if (square) {
            vec2 q = rotated / extent;
            if (abs(q.x) > 1.0 || abs(q.y) > 1.0) continue;
            falloff = 1.0;
            lookup = q * 0.5 + 0.5;
        } else {
            float d = length(delta);
            if (d > radius) continue;
            falloff = 1.0 - smoothstep(hardness, 1.0, d / radius);
            lookup = rotated / (2.0 * radius) + 0.5;
        }
        float a = ubo.color.a * falloff * textureLod(brush_tip, lookup, 0.0).a;
        acc += a * (1.0 - acc);
    }
    return acc;
}

void main() {
    vec4 clip = ubo.view_proj * vec4(v_world_pos, 1.0);
    if (clip.w <= 0.0) discard;
    vec3 ndc = clip.xyz / clip.w;
    if (abs(ndc.x) > 1.0 || abs(ndc.y) > 1.0) discard;
    vec3 view_dir = normalize(ubo.camera_pos.xyz - v_world_pos);
    if (dot(normalize(v_normal), view_dir) <= 0.05) discard;
    if (ndc.z > texture(scene_depth, ndc.xy * 0.5 + 0.5).r + 1e-3) discard;
    vec2 px = (ndc.xy * 0.5 + 0.5) * ubo.viewport.xy;

    float a = coverage(px);
    if (a <= 0.001) discard;

    vec2 src_px = px + pc.offset_mode.xy;
    if (any(lessThan(src_px, vec2(0.0))) || any(greaterThanEqual(src_px, ubo.viewport.xy))) discard;
    vec4 g = texture(gbuf_uv, src_px / ubo.viewport.xy);
    if (g.z < 0.5) discard; // nothing of this set under the source
    vec2 uv = g.xy;

    vec4 bc;
    if (pc.offset_mode.z > 0.5) {
        // Shader mask: flattened over black, so R is the value outright.
        float v = texture(src_color, uv).r;
        bc = vec4(vec3(v), 1.0) * a;
    } else {
        vec4 paint = texture(src_color, uv);
        vec3 mat = texture(mat_basecolor, uv).rgb * pc.base_factor.rgb;
        bc = vec4(mat * (1.0 - paint.a) + paint.rgb, 1.0) * a;
    }
    o_basecolor = bc * float(ubo.channels.x);
    // Other channels carry the stack's own coverage: cloning from bare
    // material leaves them alone rather than painting zeros.
    vec2 r = texture(src_rough, uv).xy * a;
    vec2 m = texture(src_metal, uv).xy * a;
    vec2 h = texture(src_height, uv).xy * a;
    vec2 o = texture(src_opacity, uv).xy * a;
    o_roughness = vec4(r, 0.0, r.y) * float(ubo.channels.y);
    o_metallic = vec4(m, 0.0, m.y) * float(ubo.channels.z);
    o_height = vec4(h, 0.0, h.y) * float(ubo.channels.w);
    o_emissive = texture(src_emissive, uv) * a * float(ubo.channels2.x);
    o_opacity = vec4(o, 0.0, o.y) * float(ubo.channels2.y);
}
",
    }
}

#[cfg(test)]
mod stroke_tests {
    use super::*;
    use crate::layers::TextureSet;
    use crate::mesh::{MeshData, uv_sphere};
    use crate::renderer::{RenderParams, SceneRenderer};
    use crate::scene::{MaterialData, ModelData};
    use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
    use vulkano::memory::allocator::AllocationCreateInfo;

    /// Minimal headless setup: a sphere, a rendered frame (so the depth buffer
    /// the stroke samples exists), and one texture set to paint into.
    struct Harness {
        gpu: GpuContext,
        renderer: SceneRenderer,
        sets: Vec<TextureSet>,
        paint: PaintSystem,
    }

    fn harness() -> Option<Harness> {
        let models = vec![ModelData {
            mesh: uv_sphere(1.0, 24, 48),
            material: MaterialData::default(),
            transform: Mat4::IDENTITY,
            name: "Sphere".into(),
            material_name: "Default".into(),
            morphs: Default::default(),
            skin: None,
        }];
        harness_with(models)
    }

    /// The same setup around any model list; one texture set per
    /// (object, material) pair, as the app builds them.
    fn harness_with(models: Vec<ModelData>) -> Option<Harness> {
        let gpu = crate::environment::test_gpu::context()?;

        let format = Format::R8G8B8A8_UNORM;
        let environment = crate::environment::Environment::procedural(&gpu);
        let mut renderer = SceneRenderer::new(
            gpu.queue.clone(),
            gpu.memory_allocator.clone(),
            gpu.command_buffer_allocator.clone(),
            gpu.descriptor_set_allocator.clone(),
            format,
            environment,
        );
        renderer.set_models(&models);

        let mut sets: Vec<TextureSet> = crate::scene::set_keys(&models)
            .into_iter()
            .map(|k| TextureSet::new(k.object, k.material, gpu.clone(), 256))
            .collect();

        let target = vulkano::image::view::ImageView::new_default(
            Image::new(
                gpu.memory_allocator.clone(),
                ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format,
                    extent: [256, 256, 1],
                    usage: ImageUsage::COLOR_ATTACHMENT
                        | ImageUsage::SAMPLED
                        | ImageUsage::STORAGE,
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .unwrap(),
        )
        .unwrap();

        // One frame, purely so `depth_view()` exists for stroke occlusion.
        let future = vulkano::sync::now(gpu.queue.device().clone()).boxed();
        let params = RenderParams {
            camera_pos: Vec3::new(0.0, 0.0, 4.0),
            view_proj: view().view_proj,
            view: glam::camera::rh::view::look_at_mat4(Vec3::new(0.0, 0.0, 4.0), Vec3::ZERO, Vec3::Y),
            exposure: 1.0,
            env_intensity: 1.0,
            key_light: 0.0,
            key_light_dir: Vec3::new(0.5, 0.8, 0.4),
            ao_strength: 1.0,
            uv_checker: false,
            material_override: None,
            selected: Vec::new(),
            hovered: Vec::new(),
            outline_width: 3.0,
            // Exercises the mask-overlay binding (black fallback here); only
            // the depth buffer of this frame matters to the stroke tests.
            mask_view: Some(crate::layers::ShaderMask::Emission),
        };
        let done = renderer.render(future, target, &params, &mut sets);
        done.then_signal_fence_and_flush().unwrap().wait(None).unwrap();

        let paint = PaintSystem::new(gpu.clone());
        Some(Harness { gpu, renderer, sets, paint })
    }

    fn view() -> PaintView {
        PaintView {
            view_proj: glam::camera::rh::proj::vulkan::perspective(1.0, 1.0, 0.1, 100.0)
                * glam::camera::rh::view::look_at_mat4(
                    Vec3::new(0.0, 0.0, 4.0),
                    Vec3::ZERO,
                    Vec3::new(0.0, 1.0, 0.0),
                ),
            camera_pos: Vec3::new(0.0, 0.0, 4.0),
            viewport_size: Vec2::new(256.0, 256.0),
        }
    }

    /// Every combination of the stroke toggles must survive a real submission.
    /// X symmetry in particular adds a second mirrored draw, and polygon fill
    /// adds a mark pass that writes an SSBO indexed by primitive.
    #[test]
    fn stroke_survives_every_toggle_combination() {
        let Some(mut h) = harness() else {
            eprintln!("no Vulkan device; skipping");
            return;
        };
        let view = view();
        let stamps = [Vec2::new(128.0, 128.0), Vec2::new(132.0, 130.0)];

        for symmetry in [false, true] {
            for fill_mode in FillMode::ALL {
                // Triangle mode flags polygons on the GPU; the group modes take
                // CPU-supplied flags. Both paths must build and submit.
                let polygon_fill = fill_mode.is_on();
                for (mask, erase) in
                    [(false, false), (true, false), (false, true), (true, true)]
                {
                    for scissor in [None, Some([10u32, 10, 64, 64]), Some([0, 0, 256, 256])] {
                        let mut brush = BrushSettings {
                            op: if erase { BrushOp::Erase } else { BrushOp::Paint },
                            symmetry_x: symmetry,
                            fill_mode,
                            size: 40.0,
                            // Exercise the colour and rotation uniforms too.
                            use_tip_color: polygon_fill,
                            tip_rotation: if symmetry { 0.7 } else { 0.0 },
                            ..BrushSettings::default()
                        };
                        if mask {
                            brush.target = PaintTarget::Mask;
                            h.sets[0].layers.add_mask(0, 0.0);
                        }
                        // Polygon fill needs a flag per triangle; feed it a
                        // full set so the fill branch actually runs.
                        let tri_count: usize = h
                            .renderer
                            .models()
                            .iter()
                            .map(|m| m.index_count as usize / 3)
                            .sum();
                        let marks = match fill_mode {
                            FillMode::Shell | FillMode::UvIsland => vec![1u32; tri_count],
                            // Triangle mode marks on the GPU; Off needs none.
                            _ => Vec::new(),
                        };
                        h.paint.stroke(
                            &h.renderer,
                            &mut h.sets[0],
                            0,
                            &view,
                            &brush,
                            &stamps,
                            scissor,
                            &marks,
                        );
                        h.sets[0].layers.composite_now();
                    }
                }
            }
        }
        let _ = &h.gpu;
    }

    /// Sum of the coverage channel over a whole layer image: alpha for
    /// RGBA16F channels, G for the RG16F scalar ones.
    fn total_alpha(gpu: &GpuContext, view: &Arc<vulkano::image::view::ImageView>) -> f32 {
        coverage(gpu, view).iter().sum()
    }

    /// Coverage-weighted centre of a layer image, in texels.
    fn coverage_centroid(gpu: &GpuContext, view: &Arc<vulkano::image::view::ImageView>) -> Vec2 {
        let res = view.image().extent()[0] as usize;
        let cov = coverage(gpu, view);
        let (mut sum, mut total) = (Vec2::ZERO, 0.0);
        for (i, a) in cov.iter().enumerate() {
            sum += Vec2::new((i % res) as f32, (i / res) as f32) * *a;
            total += a;
        }
        sum / total.max(1e-6)
    }

    /// Per-texel coverage channel of a layer image.
    fn coverage(gpu: &GpuContext, view: &Arc<vulkano::image::view::ImageView>) -> Vec<f32> {
        use vulkano::buffer::{Buffer, BufferCreateInfo};
        use vulkano::command_buffer::CopyImageToBufferInfo;
        let [w, h, _] = view.image().extent();
        let comps = view.image().format().components().iter().filter(|&&c| c > 0).count() as u32;
        let buffer: vulkano::buffer::Subbuffer<[u16]> = Buffer::new_slice(
            gpu.memory_allocator.clone(),
            BufferCreateInfo { usage: BufferUsage::TRANSFER_DST, ..Default::default() },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            (w * h * comps) as u64,
        )
        .unwrap();
        let mut builder = AutoCommandBufferBuilder::primary(
            gpu.command_buffer_allocator.clone(),
            gpu.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        builder
            .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
                view.image().clone(),
                buffer.clone(),
            ))
            .unwrap();
        builder
            .build()
            .unwrap()
            .execute(gpu.queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
        let guard = buffer.read().unwrap();
        guard
            .chunks_exact(comps as usize)
            .map(|t| half::f16::from_bits(t[comps as usize - 1]).to_f32())
            .collect()
    }

    /// Smudging across a dab drags it along: coverage is roughly kept but its
    /// centre moves. Also runs every symmetry/scissor combination through a
    /// real submission.
    #[test]
    fn smudge_moves_paint() {
        let Some(mut h) = harness() else {
            eprintln!("no Vulkan device; skipping");
            return;
        };
        let view = view();
        let dab = BrushSettings { size: 12.0, hardness: 1.0, flow: 1.0, ..BrushSettings::default() };
        h.paint.stroke(
            &h.renderer, &mut h.sets[0], 0, &view, &dab, &[Vec2::new(120.0, 128.0)], None, &[],
        );
        let color = h.sets[0].layers.active_layer().basecolor.clone();
        let before = coverage_centroid(&h.gpu, &color);
        let before_total = total_alpha(&h.gpu, &color);
        assert!(before_total > 10.0, "dab should land, got {before_total}");

        let smudge =
            BrushSettings { op: BrushOp::Smudge, size: 30.0, flow: 0.8, ..BrushSettings::default() };
        let mut prev = Vec2::new(110.0, 128.0);
        for i in 1..=6 {
            let at = Vec2::new(110.0 + 5.0 * i as f32, 128.0);
            h.paint.smudge(&h.renderer, &mut h.sets[0], 0, &view, &smudge, &[(at, at - prev)], None);
            prev = at;
        }
        let after = coverage_centroid(&h.gpu, &color);
        let after_total = total_alpha(&h.gpu, &color);
        assert!(
            before.distance(after) > 2.0,
            "smudge should move the dab: {before} -> {after}"
        );
        assert!(after_total > before_total * 0.3, "smudge lost the paint: {after_total}");

        for symmetry in [false, true] {
            for scissor in [None, Some([10u32, 10, 64, 64]), Some([0, 0, 256, 256])] {
                let brush = BrushSettings {
                    op: BrushOp::Smudge,
                    symmetry_x: symmetry,
                    ..BrushSettings::default()
                };
                h.paint.smudge(
                    &h.renderer,
                    &mut h.sets[0],
                    0,
                    &view,
                    &brush,
                    &[(Vec2::new(128.0, 128.0), Vec2::new(4.0, 1.0))],
                    scissor,
                );
            }
        }
    }

    /// Average unpremultiplied colour over covered texels of an RGBA layer.
    fn mean_colour(gpu: &GpuContext, view: &Arc<vulkano::image::view::ImageView>) -> Vec3 {
        use vulkano::buffer::{Buffer, BufferCreateInfo};
        use vulkano::command_buffer::CopyImageToBufferInfo;
        let [w, h, _] = view.image().extent();
        let buffer: vulkano::buffer::Subbuffer<[u16]> = Buffer::new_slice(
            gpu.memory_allocator.clone(),
            BufferCreateInfo { usage: BufferUsage::TRANSFER_DST, ..Default::default() },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            (w * h * 4) as u64,
        )
        .unwrap();
        let mut builder = AutoCommandBufferBuilder::primary(
            gpu.command_buffer_allocator.clone(),
            gpu.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        builder
            .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
                view.image().clone(),
                buffer.clone(),
            ))
            .unwrap();
        builder
            .build()
            .unwrap()
            .execute(gpu.queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
        let guard = buffer.read().unwrap();
        let (mut sum, mut total) = (Vec3::ZERO, 0.0);
        for t in guard.chunks_exact(4) {
            let f = |i: usize| half::f16::from_bits(t[i]).to_f32();
            sum += Vec3::new(f(0), f(1), f(2));
            total += f(3);
        }
        sum / total.max(1e-6)
    }

    /// Cloning over bare surface next to a red dab copies the red across.
    #[test]
    fn clone_copies_the_source_surface() {
        let Some(mut h) = harness() else {
            eprintln!("no Vulkan device; skipping");
            return;
        };
        let view = view();
        let red = BrushSettings {
            size: 24.0,
            hardness: 1.0,
            flow: 1.0,
            color: [1.0, 0.0, 0.0],
            ..BrushSettings::default()
        };
        h.paint.stroke(&h.renderer, &mut h.sets[0], 0, &view, &red, &[Vec2::new(110.0, 128.0)], None, &[]);
        h.sets[0].layers.composite_now();

        // Clone into a fresh layer so the result is easy to read back.
        h.sets[0].layers.add_layer();
        let clone = BrushSettings { op: BrushOp::Clone, size: 8.0, hardness: 1.0, flow: 1.0, ..red };
        h.paint.clone_stroke(
            &h.renderer,
            &mut h.sets[0],
            0,
            &view,
            &clone,
            &[Vec2::new(150.0, 128.0)],
            None,
            Vec2::new(-40.0, 0.0),
            CloneSource::Material,
        );
        let layer = h.sets[0].layers.active_layer().basecolor.clone();
        assert!(total_alpha(&h.gpu, &layer) > 10.0, "clone should deposit coverage");
        let c = mean_colour(&h.gpu, &layer);
        assert!(c.x > 0.8 && c.y < 0.1 && c.z < 0.1, "cloned colour should be the red dab, got {c}");
    }

    /// A detailed stamp drawn much smaller than its image averages its fine
    /// detail (mip levels) instead of aliasing into speckle: a fine checker
    /// of ink and nothing (4 px squares, ~10x shrunk) lands as an even ~50% tone.
    #[test]
    fn small_stamp_of_fine_detail_averages() {
        let Some(mut h) = harness() else {
            eprintln!("no Vulkan device; skipping");
            return;
        };
        let n = 256u32;
        let checker: Vec<u8> = (0..n * n)
            .flat_map(|i| if ((i % n) / 4 + (i / n) / 4) % 2 == 0 { [0u8, 0, 0, 255] } else { [0u8; 4] })
            .collect();
        let tip = crate::renderer::upload_texture_mipmapped(
            &h.gpu.queue,
            &h.gpu.memory_allocator,
            &h.gpu.command_buffer_allocator,
            n,
            n,
            &checker,
            Format::R8G8B8A8_UNORM,
        );
        let view = view();
        let stamp = BrushSettings {
            square_tip: true,
            tip: Some(tip),
            size: 12.0,
            hardness: 1.0,
            flow: 1.0,
            ..BrushSettings::default()
        };
        h.paint.stroke(&h.renderer, &mut h.sets[0], 0, &view, &stamp, &[Vec2::new(128.0, 128.0)], None, &[]);
        let colour = h.sets[0].layers.active_layer().basecolor.clone();
        let painted: Vec<f32> = coverage(&h.gpu, &colour).into_iter().filter(|a| *a > 0.05).collect();
        assert!(painted.len() > 20, "stamp landed on {} texels", painted.len());
        let solid = painted.iter().filter(|a| **a > 0.85).count() as f32 / painted.len() as f32;
        let mean = painted.iter().sum::<f32>() / painted.len() as f32;
        assert!(solid < 0.05, "{:.0}% of texels are solid ink: the checker aliased", solid * 100.0);
        assert!((0.3..0.7).contains(&mean), "mean coverage {mean}");
    }

    /// A stamp paints only what its image holds: a fully transparent tip,
    /// enlarged well past its own size, must leave no frame around the
    /// rectangle (edge texels were once filtered against an opaque border).
    #[test]
    fn transparent_stamp_leaves_no_border() {
        let Some(mut h) = harness() else {
            eprintln!("no Vulkan device; skipping");
            return;
        };
        let tip = crate::renderer::upload_texture(
            &h.gpu.queue,
            &h.gpu.memory_allocator,
            &h.gpu.command_buffer_allocator,
            8,
            8,
            &[0u8; 8 * 8 * 4],
            Format::R8G8B8A8_UNORM,
        );
        let view = view();
        let at = [Vec2::new(128.0, 128.0)];
        let stamp = BrushSettings {
            square_tip: true,
            tip: Some(tip),
            size: 60.0,
            hardness: 1.0,
            flow: 1.0,
            ..BrushSettings::default()
        };
        h.paint.stroke(&h.renderer, &mut h.sets[0], 0, &view, &stamp, &at, None, &[]);
        let colour = h.sets[0].layers.active_layer().basecolor.clone();
        let cover = total_alpha(&h.gpu, &colour);
        assert!(cover < 0.5, "a transparent stamp painted {cover} texels of coverage");
    }

    /// The stamp tool covers its whole rectangle, not a clipped circle: at
    /// equal size the square footprint holds about 4/pi as many texels.
    #[test]
    fn square_stamp_covers_its_corners() {
        let Some(mut h) = harness() else {
            eprintln!("no Vulkan device; skipping");
            return;
        };
        let view = view();
        let at = [Vec2::new(128.0, 128.0)];
        let round = BrushSettings { size: 20.0, hardness: 1.0, flow: 1.0, ..BrushSettings::default() };
        h.paint.stroke(&h.renderer, &mut h.sets[0], 0, &view, &round, &at, None, &[]);
        let colour = h.sets[0].layers.active_layer().basecolor.clone();
        let round_cover = total_alpha(&h.gpu, &colour);

        h.sets[0].layers.add_layer();
        let square = BrushSettings { square_tip: true, ..round };
        h.paint.stroke(&h.renderer, &mut h.sets[0], 0, &view, &square, &at, None, &[]);
        let colour = h.sets[0].layers.active_layer().basecolor.clone();
        let ratio = total_alpha(&h.gpu, &colour) / round_cover;
        assert!((1.15..1.45).contains(&ratio), "square/round coverage ratio {ratio}");

        // A 2:1 tip squeezes the rectangle to half height.
        h.sets[0].layers.add_layer();
        let wide = BrushSettings { square_tip: true, tip_aspect: 2.0, size: 20.0, hardness: 1.0, flow: 1.0, ..BrushSettings::default() };
        h.paint.stroke(&h.renderer, &mut h.sets[0], 0, &view, &wide, &at, None, &[]);
        let colour = h.sets[0].layers.active_layer().basecolor.clone();
        let wide_ratio = total_alpha(&h.gpu, &colour) / round_cover;
        assert!((0.55..0.75).contains(&wide_ratio), "2:1 stamp/round ratio {wide_ratio}");
    }

    /// A blank layer holds no channel textures; a colour-only stroke gives
    /// it exactly one, through a pass whose other slots are unused.
    #[test]
    fn colour_stroke_allocates_only_the_colour_channel() {
        let Some(mut h) = harness() else {
            eprintln!("no Vulkan device; skipping");
            return;
        };
        let layers = &mut h.sets[0].layers;
        layers.add_layer();
        assert!(layers.is_placeholder(&layers.active_layer().basecolor));
        let brush = BrushSettings { size: 40.0, hardness: 1.0, flow: 1.0, ..BrushSettings::default() };
        assert_eq!(brush.material_channels(), [crate::layers::PaintChannel::BaseColor]);
        let stamps = [Vec2::new(128.0, 128.0)];
        h.paint.stroke(&h.renderer, &mut h.sets[0], 0, &view(), &brush, &stamps, None, &[]);
        let layers = &mut h.sets[0].layers;
        let layer = layers.active_layer();
        assert!(total_alpha(&h.gpu, &layer.basecolor) > 10.0, "the colour channel takes the paint");
        for view in [&layer.roughness, &layer.metallic, &layer.height, &layer.emissive, &layer.opacity_map, &layer.normal] {
            assert!(layers.is_placeholder(view), "switched-off channels stay unallocated");
        }
        assert_eq!(layers.take_upgrades().len(), 1);
    }

    /// Erasing over a stroke with the same stamps removes it, and a disabled
    /// channel is left alone.
    /// An FBX object with two material slots (belt + buckle) is two texture
    /// sets. A stroke straddling both only lands on the active one: the other
    /// material's layer stays empty even though it was under the brush.
    #[test]
    fn stroke_paints_only_the_active_material() {
        // One sphere split into two material parts down the middle, the way
        // the FBX loader splits a mesh by `material_parts`.
        let sphere = uv_sphere(1.0, 24, 48);
        let mut halves: [MeshData; 2] = [
            MeshData { vertices: Vec::new(), indices: Vec::new() },
            MeshData { vertices: Vec::new(), indices: Vec::new() },
        ];
        for tri in sphere.indices.chunks_exact(3) {
            let cx: f32 = tri.iter().map(|&i| sphere.vertices[i as usize].position[0]).sum();
            let half = &mut halves[usize::from(cx >= 0.0)];
            for &i in tri {
                half.indices.push(half.vertices.len() as u32);
                half.vertices.push(sphere.vertices[i as usize]);
            }
        }
        let [left, right] = halves;
        let models = vec![
            ModelData {
                mesh: left,
                material: MaterialData::default(),
                transform: Mat4::IDENTITY,
                name: "Belt".into(),
                material_name: "Leather".into(),
                morphs: Default::default(),
                skin: None,
            },
            ModelData {
                mesh: right,
                material: MaterialData::default(),
                transform: Mat4::IDENTITY,
                name: "Belt".into(),
                material_name: "Buckle".into(),
                morphs: Default::default(),
                skin: None,
            },
        ];
        let Some(mut h) = harness_with(models) else {
            eprintln!("no Vulkan device; skipping");
            return;
        };
        assert_eq!(h.sets.len(), 2, "two materials of one object are two sets");
        let view = view();
        let brush = BrushSettings {
            size: 60.0,
            hardness: 1.0,
            flow: 1.0,
            color: [1.0, 0.0, 0.0],
            ..BrushSettings::default()
        };
        // Dead centre of the viewport: the seam between the two halves.
        let stamps = [Vec2::new(128.0, 128.0)];

        let (buckle, leather) = (1, 0);
        h.paint.stroke(&h.renderer, &mut h.sets[buckle], buckle, &view, &brush, &stamps, None, &[]);

        let painted = h.sets[buckle].layers.active_layer().basecolor.clone();
        let untouched = h.sets[leather].layers.active_layer().basecolor.clone();
        assert!(total_alpha(&h.gpu, &painted) > 10.0, "the active material takes the paint");
        assert_eq!(
            total_alpha(&h.gpu, &untouched),
            0.0,
            "the other material under the brush must not be painted"
        );

        // Both halves share the sphere's UV layout, so an unscoped stroke
        // would have put the same stamp on both. Painting the other set now
        // must land on its own faces (and nowhere on the first set).
        let before = total_alpha(&h.gpu, &painted);
        h.paint.stroke(&h.renderer, &mut h.sets[leather], leather, &view, &brush, &stamps, None, &[]);
        // Its first stroke gave the blank channel a real texture.
        let untouched = h.sets[leather].layers.active_layer().basecolor.clone();
        assert!(total_alpha(&h.gpu, &untouched) > 10.0, "the other material paints once it is active");
        assert_eq!(total_alpha(&h.gpu, &painted), before, "switching sets does not touch the first one");
    }

    #[test]
    fn erase_removes_paint() {
        let Some(mut h) = harness() else {
            eprintln!("no Vulkan device; skipping");
            return;
        };
        let view = view();
        let stamps = [Vec2::new(128.0, 128.0), Vec2::new(140.0, 128.0)];
        let paint = BrushSettings {
            size: 40.0,
            hardness: 1.0,
            flow: 1.0,
            paint_roughness: true,
            ..BrushSettings::default()
        };
        h.paint.stroke(&h.renderer, &mut h.sets[0], 0, &view, &paint, &stamps, None, &[]);
        let layer = h.sets[0].layers.active_layer();
        let (color, rough) = (layer.basecolor.clone(), layer.roughness.clone());
        let painted = total_alpha(&h.gpu, &color);
        let painted_rough = total_alpha(&h.gpu, &rough);
        assert!(painted > 100.0, "stroke should cover texels, got {painted}");

        // Erase colour only: roughness must survive untouched.
        let erase = BrushSettings { op: BrushOp::Erase, paint_roughness: false, ..paint };
        h.paint.stroke(&h.renderer, &mut h.sets[0], 0, &view, &erase, &stamps, None, &[]);
        let erased = total_alpha(&h.gpu, &color);
        assert!(erased < painted * 0.01, "erase left {erased} of {painted}");
        let rough_after = total_alpha(&h.gpu, &rough);
        assert!(
            (rough_after - painted_rough).abs() < painted_rough * 1e-3,
            "disabled channel changed: {painted_rough} -> {rough_after}"
        );
    }
}

mod paint_mark_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
#version 450
layout(location = 0) in vec3 v_world_pos;
layout(location = 1) in vec3 v_normal;

layout(location = 0) out vec4 o_basecolor;
layout(location = 1) out vec4 o_roughness;
layout(location = 2) out vec4 o_metallic;
layout(location = 3) out vec4 o_height;
layout(location = 4) out vec4 o_emissive;
layout(location = 5) out vec4 o_opacity;

layout(set = 0, binding = 0) uniform Ubo {
    mat4 view_proj; mat4 model; vec4 camera_pos; vec4 brush; vec4 color;
    vec4 values; vec4 viewport; vec4 emissive; uvec4 channels; uvec4 channels2; vec4 extra;
} ubo;
layout(set = 0, binding = 1) uniform sampler2D scene_depth;
layout(set = 0, binding = 2) uniform sampler2D brush_tip;
layout(set = 0, binding = 3) readonly buffer Stamps { vec2 centre[]; } stamps;
layout(set = 0, binding = 4) buffer Marks { uint flag[]; } marks;

// Flags polygons the brush touches *and* that are actually visible: the depth
// comparison below is the whole point, since a CPU screen-overlap test cannot
// tell a front-facing polygon behind something from one you can see.
void main() {
    vec4 clip = ubo.view_proj * vec4(v_world_pos, 1.0);
    if (clip.w <= 0.0) discard;
    vec3 ndc = clip.xyz / clip.w;
    if (abs(ndc.x) > 1.0 || abs(ndc.y) > 1.0) discard;
    vec3 view_dir = normalize(ubo.camera_pos.xyz - v_world_pos);
    if (dot(normalize(v_normal), view_dir) <= 0.05) discard;
    vec2 depth_uv = ndc.xy * 0.5 + 0.5;
    if (ndc.z > texture(scene_depth, depth_uv).r + 1e-3) discard;

    vec2 px = (ndc.xy * 0.5 + 0.5) * ubo.viewport.xy;
    float radius = ubo.brush.z;
    float cs = cos(ubo.extra.w);
    float sn = sin(ubo.extra.w);
    uint count = ubo.channels2.z;
    bool hit = false;
    for (uint i = 0u; i < count; i++) {
        vec2 d = px - stamps.centre[i];
        if (length(d) > radius) continue;
        // Respect the tip's own alpha, so a stamp does not flag polygons that
        // only its transparent parts cover. Sampling it also keeps binding 2
        // alive: vulkano strips unused bindings during shader reflection, and
        // writing a stripped binding is a validation error.
        vec2 rotated = vec2(d.x * cs - d.y * sn, d.x * sn + d.y * cs);
        if (textureLod(brush_tip, rotated / (2.0 * radius) + 0.5, 0.0).a > 0.02) {
            hit = true;
            break;
        }
    }
    if (hit) marks.flag[ubo.channels2.w + uint(gl_PrimitiveID)] = 1u;

    // Zeros are a no-op under premultiplied blending, so nothing is painted.
    o_basecolor = vec4(0.0);
    o_roughness = vec4(0.0);
    o_metallic = vec4(0.0);
    o_height = vec4(0.0);
    o_emissive = vec4(0.0);
    o_opacity = vec4(0.0);
}
",
    }
}

mod paint_mask_mark_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
#version 450
layout(location = 0) in vec3 v_world_pos;
layout(location = 1) in vec3 v_normal;
layout(location = 0) out vec4 o_mask;

layout(set = 0, binding = 0) uniform Ubo {
    mat4 view_proj; mat4 model; vec4 camera_pos; vec4 brush; vec4 color;
    vec4 values; vec4 viewport; vec4 emissive; uvec4 channels; uvec4 channels2; vec4 extra;
} ubo;
layout(set = 0, binding = 1) uniform sampler2D scene_depth;
layout(set = 0, binding = 2) uniform sampler2D brush_tip;
layout(set = 0, binding = 3) readonly buffer Stamps { vec2 centre[]; } stamps;
layout(set = 0, binding = 4) buffer Marks { uint flag[]; } marks;

void main() {
    vec4 clip = ubo.view_proj * vec4(v_world_pos, 1.0);
    if (clip.w <= 0.0) discard;
    vec3 ndc = clip.xyz / clip.w;
    if (abs(ndc.x) > 1.0 || abs(ndc.y) > 1.0) discard;
    vec3 view_dir = normalize(ubo.camera_pos.xyz - v_world_pos);
    if (dot(normalize(v_normal), view_dir) <= 0.05) discard;
    vec2 depth_uv = ndc.xy * 0.5 + 0.5;
    if (ndc.z > texture(scene_depth, depth_uv).r + 1e-3) discard;

    vec2 px = (ndc.xy * 0.5 + 0.5) * ubo.viewport.xy;
    float radius = ubo.brush.z;
    float cs = cos(ubo.extra.w);
    float sn = sin(ubo.extra.w);
    uint count = ubo.channels2.z;
    for (uint i = 0u; i < count; i++) {
        vec2 d = px - stamps.centre[i];
        if (length(d) > radius) continue;
        vec2 rotated = vec2(d.x * cs - d.y * sn, d.x * sn + d.y * cs);
        if (textureLod(brush_tip, rotated / (2.0 * radius) + 0.5, 0.0).a > 0.02) {
            marks.flag[ubo.channels2.w + uint(gl_PrimitiveID)] = 1u;
            break;
        }
    }
    o_mask = vec4(0.0);
}
",
    }
}
