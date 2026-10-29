use std::path::Path;
use std::sync::Arc;

use anyhow::{Context as _, Result};
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferToImageInfo,
    PrimaryCommandBufferAbstract,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::Queue;
use vulkano::format::Format;
use vulkano::image::sampler::{
    Filter, Sampler, SamplerAddressMode, SamplerCreateInfo, LOD_CLAMP_NONE,
};
use vulkano::image::view::{ImageView, ImageViewCreateInfo, ImageViewType};
use vulkano::image::{
    Image, ImageAspects, ImageCreateFlags, ImageCreateInfo, ImageSubresourceRange, ImageType,
    ImageUsage,
};
use vulkano::memory::allocator::{
    AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator,
};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
};
use vulkano::shader::ShaderModule;
use vulkano::sync::GpuFuture;

const ENV_SIZE: u32 = 512;
const IRRADIANCE_SIZE: u32 = 32;
const PREFILTERED_SIZE: u32 = 128;
const PREFILTERED_MIPS: u32 = 6; // 128 → 4

/// Device features the app needs beyond the defaults.
///
/// `fragment_stores_and_atomics` lets the polygon-fill mark pass flag touched
/// triangles from the fragment stage, and reading `gl_PrimitiveID` there costs
/// the SPIR-V `Geometry` capability, which in turn requires `geometry_shader`
/// even though no geometry shader is used.
pub fn required_device_features() -> vulkano::device::DeviceFeatures {
    vulkano::device::DeviceFeatures {
        fragment_stores_and_atomics: true,
        geometry_shader: true,
        // Imported maps are BC7/BC4/BC5 (see `texcompress`); every desktop
        // GPU has it.
        texture_compression_bc: true,
        // Stroke passes leave never-painted channel slots unused, which
        // takes a blend state per slot (see `paint::StrokeVariant`).
        independent_blend: true,
        ..vulkano::device::DeviceFeatures::empty()
    }
}


/// One Vulkan device shared by every GPU test.
///
/// Each test used to build its own `VulkanoContext`; run in parallel that
/// exhausts the driver and device creation starts failing outright. Tests are
/// read-mostly against the device itself, so sharing is safe and much faster.
#[cfg(test)]
pub mod test_gpu {
    use super::GpuContext;
    use std::sync::{Arc, OnceLock};
    use vulkano_util::context::{VulkanoConfig, VulkanoContext};

    struct Shared {
        _context: VulkanoContext,
        gpu: GpuContext,
    }

    fn shared() -> &'static Option<Shared> {
        static SHARED: OnceLock<Option<Shared>> = OnceLock::new();
        SHARED.get_or_init(|| {
            let context = VulkanoContext::new(VulkanoConfig {
                device_features: super::required_device_features(),
                ..VulkanoConfig::default()
            });
            let gpu = GpuContext {
                queue: context.graphics_queue().clone(),
                memory_allocator: context.memory_allocator().clone(),
                command_buffer_allocator: Arc::new(
                    vulkano::command_buffer::allocator::StandardCommandBufferAllocator::new(
                        context.device().clone(),
                        Default::default(),
                    ),
                ),
                descriptor_set_allocator: Arc::new(
                    vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator::new(
                        context.device().clone(),
                        Default::default(),
                    ),
                ),
            };
            Some(Shared { _context: context, gpu })
        })
    }

    /// The shared context, or None when there is no usable Vulkan device.
    pub fn context() -> Option<GpuContext> {
        shared().as_ref().map(|s| s.gpu.clone())
    }
}

/// GPU allocator/queue bundle the environment generator needs.
#[derive(Clone)]
pub struct GpuContext {
    pub queue: Arc<Queue>,
    pub memory_allocator: Arc<StandardMemoryAllocator>,
    pub command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pub descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
}

/// Image-based lighting data: environment cubemap plus its diffuse irradiance
/// and GGX-prefiltered specular convolutions.
pub struct Environment {
    pub env_cube: Arc<ImageView>,
    pub irradiance_cube: Arc<ImageView>,
    pub prefiltered_cube: Arc<ImageView>,
    pub prefiltered_mips: u32,
    pub sampler: Arc<Sampler>,
}

impl Environment {
    /// Built-in procedural sky (gradient + sun), so the app lights nicely
    /// without any external assets.
    pub fn procedural(gpu: &GpuContext) -> Self {
        Self::generate(gpu, None)
    }

    pub fn from_hdr(gpu: &GpuContext, path: &Path) -> Result<Self> {
        let img = image::open(path)
            .with_context(|| format!("failed to open {}", path.display()))?
            .to_rgba32f();
        let (w, h) = img.dimensions();
        let equirect = upload_rgba32f(gpu, w, h, img.as_raw());
        Ok(Self::generate(gpu, Some(equirect)))
    }

    fn generate(gpu: &GpuContext, equirect: Option<Arc<ImageView>>) -> Self {
        let device = gpu.queue.device().clone();

        let procedural = equirect.is_none();
        // The equirect binding must always be valid; bind a 1x1 black image
        // when we're generating the procedural sky.
        let equirect =
            equirect.unwrap_or_else(|| upload_rgba32f(gpu, 1, 1, &[0.0, 0.0, 0.0, 1.0]));

        let env_image = create_cube_image(gpu, ENV_SIZE, 1);
        let irradiance_image = create_cube_image(gpu, IRRADIANCE_SIZE, 1);
        let prefiltered_image = create_cube_image(gpu, PREFILTERED_SIZE, PREFILTERED_MIPS);

        let env_cube = cube_view(&env_image);
        let irradiance_cube = cube_view(&irradiance_image);
        let prefiltered_cube = cube_view(&prefiltered_image);

        let sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                mipmap_mode: vulkano::image::sampler::SamplerMipmapMode::Linear,
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                lod: 0.0..=LOD_CLAMP_NONE,
                ..Default::default()
            },
        )
        .unwrap();

        let envgen = build_compute(&device, envgen_cs::load(device.clone()).unwrap());
        let irradiance = build_compute(&device, irradiance_cs::load(device.clone()).unwrap());
        let prefilter = build_compute(&device, prefilter_cs::load(device.clone()).unwrap());

        let mut builder = AutoCommandBufferBuilder::primary(
            gpu.command_buffer_allocator.clone(),
            gpu.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        // Pass 1: fill the environment cubemap (procedural sky or equirect sample).
        {
            let set = DescriptorSet::new(
                gpu.descriptor_set_allocator.clone(),
                envgen.layout().set_layouts()[0].clone(),
                [
                    WriteDescriptorSet::image_view(0, storage_array_view(&env_image, 0)),
                    WriteDescriptorSet::image_view_sampler(1, equirect, sampler.clone()),
                ],
                [],
            )
            .unwrap();
            builder
                .bind_pipeline_compute(envgen.clone())
                .unwrap()
                .bind_descriptor_sets(PipelineBindPoint::Compute, envgen.layout().clone(), 0, set)
                .unwrap()
                .push_constants(
                    envgen.layout().clone(),
                    0,
                    envgen_cs::Pc {
                        size: ENV_SIZE,
                        mode: if procedural { 0 } else { 1 },
                    },
                )
                .unwrap();
            unsafe {
                builder.dispatch([ENV_SIZE / 8, ENV_SIZE / 8, 6]).unwrap();
            }
        }

        // Pass 2: cosine-convolve into the irradiance map.
        {
            let set = DescriptorSet::new(
                gpu.descriptor_set_allocator.clone(),
                irradiance.layout().set_layouts()[0].clone(),
                [
                    WriteDescriptorSet::image_view(0, storage_array_view(&irradiance_image, 0)),
                    WriteDescriptorSet::image_view_sampler(1, env_cube.clone(), sampler.clone()),
                ],
                [],
            )
            .unwrap();
            builder
                .bind_pipeline_compute(irradiance.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    irradiance.layout().clone(),
                    0,
                    set,
                )
                .unwrap()
                .push_constants(
                    irradiance.layout().clone(),
                    0,
                    irradiance_cs::Pc { size: IRRADIANCE_SIZE },
                )
                .unwrap();
            unsafe {
                builder
                    .dispatch([IRRADIANCE_SIZE / 8 + 1, IRRADIANCE_SIZE / 8 + 1, 6])
                    .unwrap();
            }
        }

        // Pass 3: GGX importance-sample each prefiltered mip (roughness rises per mip).
        builder.bind_pipeline_compute(prefilter.clone()).unwrap();
        for mip in 0..PREFILTERED_MIPS {
            let mip_size = (PREFILTERED_SIZE >> mip).max(1);
            let set = DescriptorSet::new(
                gpu.descriptor_set_allocator.clone(),
                prefilter.layout().set_layouts()[0].clone(),
                [
                    WriteDescriptorSet::image_view(0, storage_array_view(&prefiltered_image, mip)),
                    WriteDescriptorSet::image_view_sampler(1, env_cube.clone(), sampler.clone()),
                ],
                [],
            )
            .unwrap();
            builder
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    prefilter.layout().clone(),
                    0,
                    set,
                )
                .unwrap()
                .push_constants(
                    prefilter.layout().clone(),
                    0,
                    prefilter_cs::Pc {
                        size: mip_size,
                        roughness: mip as f32 / (PREFILTERED_MIPS - 1) as f32,
                        sample_count: 256,
                    },
                )
                .unwrap();
            unsafe {
                builder.dispatch([mip_size / 8 + 1, mip_size / 8 + 1, 6]).unwrap();
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

        Self {
            env_cube,
            irradiance_cube,
            prefiltered_cube,
            prefiltered_mips: PREFILTERED_MIPS,
            sampler,
        }
    }
}

pub(crate) fn build_compute(
    device: &Arc<vulkano::device::Device>,
    module: Arc<ShaderModule>,
) -> Arc<ComputePipeline> {
    let stage = PipelineShaderStageCreateInfo::new(module.entry_point("main").unwrap());
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
    )
    .unwrap();
    ComputePipeline::new(
        device.clone(),
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .unwrap()
}

fn create_cube_image(gpu: &GpuContext, size: u32, mip_levels: u32) -> Arc<Image> {
    crate::vram::image(
        gpu.memory_allocator.clone(),
        ImageCreateInfo {
            flags: ImageCreateFlags::CUBE_COMPATIBLE,
            image_type: ImageType::Dim2d,
            format: Format::R16G16B16A16_SFLOAT,
            extent: [size, size, 1],
            array_layers: 6,
            mip_levels,
            usage: ImageUsage::STORAGE | ImageUsage::SAMPLED,
            ..Default::default()
        },
        crate::vram::Kind::Environment,
    )
    .unwrap()
}

fn cube_view(image: &Arc<Image>) -> Arc<ImageView> {
    ImageView::new(
        image.clone(),
        ImageViewCreateInfo {
            view_type: ImageViewType::Cube,
            usage: ImageUsage::SAMPLED,
            ..ImageViewCreateInfo::from_image(image)
        },
    )
    .unwrap()
}

/// 2D-array view of a single mip level, for storage writes from compute.
fn storage_array_view(image: &Arc<Image>, mip: u32) -> Arc<ImageView> {
    ImageView::new(
        image.clone(),
        ImageViewCreateInfo {
            view_type: ImageViewType::Dim2dArray,
            usage: ImageUsage::STORAGE,
            subresource_range: ImageSubresourceRange {
                aspects: ImageAspects::COLOR,
                mip_levels: mip..mip + 1,
                array_layers: 0..6,
            },
            ..ImageViewCreateInfo::from_image(image)
        },
    )
    .unwrap()
}

fn upload_rgba32f(gpu: &GpuContext, width: u32, height: u32, data: &[f32]) -> Arc<ImageView> {
    let staging = Buffer::from_iter(
        gpu.memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        data.iter().copied(),
    )
    .unwrap();

    let image = crate::vram::image(
        gpu.memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::R32G32B32A32_SFLOAT,
            extent: [width, height, 1],
            usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
            ..Default::default()
        },
        crate::vram::Kind::Environment,
    )
    .unwrap();

    let mut builder = AutoCommandBufferBuilder::primary(
        gpu.command_buffer_allocator.clone(),
        gpu.queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();
    builder
        .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(staging, image.clone()))
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

    ImageView::new_default(image).unwrap()
}

// NOTE: the `face_direction` GLSL helper is duplicated in all three compute
// shaders below because vulkano_shaders::shader! only accepts literal strings.

mod envgen_cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
#version 450
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 0, rgba16f) uniform writeonly image2DArray out_faces;
layout(set = 0, binding = 1) uniform sampler2D equirect;
layout(push_constant) uniform Pc { uint size; uint mode; } pc;

const float PI = 3.14159265359;

vec3 face_direction(uint face, vec2 st) {
    if (face == 0u) return normalize(vec3( 1.0, -st.y, -st.x));
    if (face == 1u) return normalize(vec3(-1.0, -st.y,  st.x));
    if (face == 2u) return normalize(vec3( st.x,  1.0,  st.y));
    if (face == 3u) return normalize(vec3( st.x, -1.0, -st.y));
    if (face == 4u) return normalize(vec3( st.x, -st.y,  1.0));
    return normalize(vec3(-st.x, -st.y, -1.0));
}

vec3 procedural_sky(vec3 d) {
    vec3 sun_dir = normalize(vec3(0.5, 0.6, 0.4));
    float horizon = smoothstep(-0.08, 0.22, d.y);
    vec3 sky = mix(vec3(0.85, 0.82, 0.78), vec3(0.30, 0.45, 0.75),
                   pow(clamp(d.y, 0.0, 1.0), 0.55));
    vec3 ground = mix(vec3(0.30, 0.26, 0.22), vec3(0.10, 0.09, 0.08),
                      clamp(-d.y * 3.0, 0.0, 1.0));
    vec3 col = mix(ground, sky, horizon);
    float sd = dot(d, sun_dir);
    col += vec3(1.0, 0.95, 0.85) * 60.0 * smoothstep(0.99955, 0.99985, sd);
    col += vec3(1.0, 0.90, 0.70) * 0.35 * pow(clamp(sd, 0.0, 1.0), 8.0);
    return col;
}

vec2 equirect_uv(vec3 d) {
    return vec2(atan(d.z, d.x) / (2.0 * PI) + 0.5, acos(clamp(d.y, -1.0, 1.0)) / PI);
}

void main() {
    uvec3 id = gl_GlobalInvocationID;
    if (id.x >= pc.size || id.y >= pc.size) return;
    vec2 st = (vec2(id.xy) + 0.5) / float(pc.size) * 2.0 - 1.0;
    vec3 dir = face_direction(id.z, st);
    vec3 col = pc.mode == 0u ? procedural_sky(dir) : texture(equirect, equirect_uv(dir)).rgb;
    imageStore(out_faces, ivec3(id), vec4(col, 1.0));
}
",
    }
}

mod irradiance_cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
#version 450
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 0, rgba16f) uniform writeonly image2DArray out_faces;
layout(set = 0, binding = 1) uniform samplerCube env;
layout(push_constant) uniform Pc { uint size; } pc;

const float PI = 3.14159265359;

vec3 face_direction(uint face, vec2 st) {
    if (face == 0u) return normalize(vec3( 1.0, -st.y, -st.x));
    if (face == 1u) return normalize(vec3(-1.0, -st.y,  st.x));
    if (face == 2u) return normalize(vec3( st.x,  1.0,  st.y));
    if (face == 3u) return normalize(vec3( st.x, -1.0, -st.y));
    if (face == 4u) return normalize(vec3( st.x, -st.y,  1.0));
    return normalize(vec3(-st.x, -st.y, -1.0));
}

void main() {
    uvec3 id = gl_GlobalInvocationID;
    if (id.x >= pc.size || id.y >= pc.size) return;
    vec2 st = (vec2(id.xy) + 0.5) / float(pc.size) * 2.0 - 1.0;
    vec3 n = face_direction(id.z, st);

    vec3 up = abs(n.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 right = normalize(cross(up, n));
    up = cross(n, right);

    vec3 irradiance = vec3(0.0);
    float count = 0.0;
    for (float phi = 0.0; phi < 2.0 * PI; phi += 0.1) {
        for (float theta = 0.0; theta < 0.5 * PI; theta += 0.1) {
            vec3 t = vec3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
            vec3 dir = t.x * right + t.y * up + t.z * n;
            // Clamp fireflies (the sun disk) so the diffuse term stays stable.
            vec3 radiance = min(textureLod(env, dir, 0.0).rgb, vec3(25.0));
            irradiance += radiance * cos(theta) * sin(theta);
            count += 1.0;
        }
    }
    irradiance = PI * irradiance / count;
    imageStore(out_faces, ivec3(id), vec4(irradiance, 1.0));
}
",
    }
}

mod prefilter_cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
#version 450
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 0, rgba16f) uniform writeonly image2DArray out_faces;
layout(set = 0, binding = 1) uniform samplerCube env;
layout(push_constant) uniform Pc { uint size; float roughness; uint sample_count; } pc;

const float PI = 3.14159265359;

vec3 face_direction(uint face, vec2 st) {
    if (face == 0u) return normalize(vec3( 1.0, -st.y, -st.x));
    if (face == 1u) return normalize(vec3(-1.0, -st.y,  st.x));
    if (face == 2u) return normalize(vec3( st.x,  1.0,  st.y));
    if (face == 3u) return normalize(vec3( st.x, -1.0, -st.y));
    if (face == 4u) return normalize(vec3( st.x, -st.y,  1.0));
    return normalize(vec3(-st.x, -st.y, -1.0));
}

float radical_inverse_vdc(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}

vec2 hammersley(uint i, uint n) {
    return vec2(float(i) / float(n), radical_inverse_vdc(i));
}

vec3 importance_sample_ggx(vec2 xi, vec3 n, float roughness) {
    float a = roughness * roughness;
    float phi = 2.0 * PI * xi.x;
    float cos_theta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
    float sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    vec3 h = vec3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
    vec3 up = abs(n.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(up, n));
    vec3 bitangent = cross(n, tangent);
    return normalize(tangent * h.x + bitangent * h.y + n * h.z);
}

void main() {
    uvec3 id = gl_GlobalInvocationID;
    if (id.x >= pc.size || id.y >= pc.size) return;
    vec2 st = (vec2(id.xy) + 0.5) / float(pc.size) * 2.0 - 1.0;
    vec3 n = face_direction(id.z, st);
    vec3 v = n;

    vec3 color = vec3(0.0);
    float weight = 0.0;
    for (uint i = 0u; i < pc.sample_count; i++) {
        vec2 xi = hammersley(i, pc.sample_count);
        vec3 h = importance_sample_ggx(xi, n, pc.roughness);
        vec3 l = normalize(2.0 * dot(v, h) * h - v);
        float ndotl = dot(n, l);
        if (ndotl > 0.0) {
            vec3 radiance = min(textureLod(env, l, 0.0).rgb, vec3(50.0));
            color += radiance * ndotl;
            weight += ndotl;
        }
    }
    imageStore(out_faces, ivec3(id), vec4(color / max(weight, 1e-4), 1.0));
}
",
    }
}
