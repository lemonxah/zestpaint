use std::sync::Arc;

use glam::{Mat4, Vec3};
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::format::Format;
use vulkano::image::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo};
use vulkano::image::view::ImageView;
use vulkano::image::{ImageCreateInfo, ImageType, ImageUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint};
use vulkano::sync::GpuFuture;

use crate::environment::{build_compute, GpuContext};
use crate::layers::TextureSet;
use crate::renderer::SceneRenderer;
use crate::studio::BLOCK_VEC4S;

/// Size of the shader's texture table (descriptor array). Sets are not
/// capped: each one indexes this table, and identical views share a slot.
const MAX_TEXTURES: usize = 1024;
/// vec4s of per-set parameters in the `SetParams` buffer.
const SET_STRIDE: usize = 6;
/// vec4s per triangle in the `Tris` buffer.
const TRI_STRIDE: usize = 9;

#[repr(C)]
#[derive(Clone, Copy)]
struct GpuNode {
    bmin: [f32; 4],
    bmax: [f32; 4],
    /// x: left child (or -1 for leaf), y: right child, z: first tri, w: count
    meta: [i32; 4],
}

/// What one traced frame looks through and at.
pub struct Frame<'a> {
    pub view_proj: Mat4,
    pub camera_pos: Vec3,
    /// Lights, backdrop, background and lens: `studio::gpu_block`.
    pub studio: &'a [[f32; 4]; BLOCK_VEC4S],
    pub bounces: u32,
    pub exposure: f32,
}

/// Progressive samples of one image: the viewport's, or a render's.
#[derive(Default)]
pub struct Accumulator {
    image: Option<Arc<ImageView>>,
    extent: [u32; 3],
    /// Samples per pixel so far.
    pub samples: u32,
}

impl Accumulator {
    fn ensure(&mut self, gpu: &GpuContext, extent: [u32; 3]) -> Arc<ImageView> {
        if self.image.is_none() || self.extent != extent {
            self.image = Some(
                ImageView::new_default(
                    crate::vram::image(
                        gpu.memory_allocator.clone(),
                        ImageCreateInfo {
                            image_type: ImageType::Dim2d,
                            format: Format::R32G32B32A32_SFLOAT,
                            extent,
                            usage: ImageUsage::STORAGE,
                            ..Default::default()
                        },
                        crate::vram::Kind::Viewport,
                    )
                    .unwrap(),
                )
                .unwrap(),
            );
            self.extent = extent;
            self.samples = 0;
        }
        self.image.clone().unwrap()
    }
}

/// How the accumulated samples become pixels.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Output {
    /// Over a checkerboard where the background is transparent.
    Viewport,
    /// Straight (unpremultiplied) alpha, for saving.
    File,
}

/// Progressive GPU path tracer over a CPU-built BVH of the visible scene:
/// a metal/rough GGX surface with the painted channels and normal maps,
/// lit by the environment and the studio's soft lights (sampled directly,
/// with shadows), on an optional backdrop, behind which shows the
/// environment, a colour or nothing (alpha 0).
pub struct PathTracer {
    gpu: GpuContext,
    trace: Arc<ComputePipeline>,
    resolve: Arc<ComputePipeline>,
    sampler: Arc<Sampler>,
    viewport: Accumulator,
    tris: Option<Subbuffer<[f32]>>,
    nodes: Option<Subbuffer<[f32]>>,
    set_params: Option<Subbuffer<[f32]>>,
    /// Deduplicated texture table, padded to `MAX_TEXTURES`: slot 0 white,
    /// slot 1 a flat normal map. Each set's params hold its slots.
    textures: Vec<Arc<ImageView>>,
    tri_count: u32,
}

impl PathTracer {
    pub fn new(gpu: GpuContext) -> Self {
        let device = gpu.queue.device().clone();
        let trace = build_compute(&device, trace_cs::load(device.clone()).unwrap());
        let resolve = build_compute(&device, resolve_cs::load(device.clone()).unwrap());
        let sampler = Sampler::new(
            device,
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                ..Default::default()
            },
        )
        .unwrap();
        Self {
            gpu,
            trace,
            resolve,
            sampler,
            viewport: Accumulator::default(),
            tris: None,
            nodes: None,
            set_params: None,
            textures: Vec::new(),
            tri_count: 0,
        }
    }

    pub fn has_scene(&self) -> bool {
        self.tris.is_some()
    }

    /// Samples per pixel the viewport has gathered.
    pub fn viewport_samples(&self) -> u32 {
        self.viewport.samples
    }

    /// Flatten the visible models into a triangle soup + BVH and capture the
    /// per-set textures the shader samples. Geometry comes from the CPU-side
    /// model list; textures and factors from the renderer's GPU models.
    pub fn rebuild_scene(
        &mut self,
        renderer: &SceneRenderer,
        sets: &[TextureSet],
        models: &[crate::scene::ModelData],
    ) {
        // -- gather world-space triangles ------------------------------------
        struct CpuTri {
            p: [Vec3; 3],
            n: [Vec3; 3],
            uv: [[f32; 2]; 3],
            /// Tangent along +U and the handedness of +V.
            tangent: [f32; 4],
            set: usize,
            centroid: Vec3,
        }
        let mut tris: Vec<CpuTri> = Vec::new();
        let mut set_params = vec![0f32; sets.len().max(1) * SET_STRIDE * 4];
        let white = renderer.white_texture();
        let flat = renderer.flat_normal_texture();
        let mut textures: Vec<Arc<ImageView>> = vec![white.clone(), flat];
        let mut table_full = false;
        // A table slot for `view`; past the table's size, `fallback`.
        let mut slot = |view: &Arc<ImageView>, fallback: i32| -> i32 {
            if let Some(i) = textures.iter().position(|t| Arc::ptr_eq(t, view)) {
                return i as i32;
            }
            if textures.len() == MAX_TEXTURES {
                table_full = true;
                return fallback;
            }
            textures.push(view.clone());
            (textures.len() - 1) as i32
        };
        let bits = |i: i32| f32::from_bits(i as u32);

        // Per-set textures and factors from the uploaded GPU models.
        for gpu_model in renderer.models() {
            let Some(set) = sets.get(gpu_model.set_index) else { continue };
            if !set.visible {
                continue;
            }
            let o = gpu_model.set_index * SET_STRIDE * 4;
            set_params[o] = gpu_model.factors[0];
            set_params[o + 1] = gpu_model.factors[1];
            set_params[o + 2] = gpu_model.factors[2];
            // Coverage, as the raster pass computes it: material opacity
            // (here) × base colour alpha, cut out at 0.5 or blended (o + 7).
            set_params[o + 3] = gpu_model.alpha[0];
            set_params[o + 4] = gpu_model.emission[0];
            set_params[o + 5] = gpu_model.emission[1];
            set_params[o + 6] = gpu_model.emission[2];
            set_params[o + 7] = gpu_model.alpha[1];
            let layers = &set.layers;
            let slots = [
                slot(&gpu_model.textures[0], 0),
                slot(&layers.composite_basecolor, 0),
                slot(&layers.composite_emissive, 0),
                slot(&layers.composite_opacity, 0),
            ];
            for (k, s) in slots.into_iter().enumerate() {
                set_params[o + 8 + k] = bits(s);
            }
            set_params[o + 12] = if set.material_drawn_by_layers() { 1.0 } else { 0.0 };
            set_params[o + 13] = bits(slot(&gpu_model.textures[5], 0));
            set_params[o + 14] = gpu_model.metallic;
            set_params[o + 15] = gpu_model.roughness;
            set_params[o + 16] = bits(slot(&gpu_model.textures[1], 0));
            set_params[o + 17] = bits(slot(&gpu_model.textures[2], 0));
            set_params[o + 18] = bits(slot(&gpu_model.textures[3], 1));
            set_params[o + 19] = bits(slot(&layers.composite_roughness, 0));
            set_params[o + 20] = bits(slot(&layers.composite_metallic, 0));
            set_params[o + 21] = bits(slot(&layers.normal_map, 1));
            set_params[o + 22] = gpu_model.emission[3];
        }
        if table_full {
            eprintln!("path tracer: more than {MAX_TEXTURES} textures; extras render plain");
        }
        textures.resize(MAX_TEXTURES, white);

        // Geometry from the CPU model list.
        let indices_per_model = crate::scene::set_indices(models);
        for (model, set_index) in models.iter().zip(indices_per_model) {
            if !sets.get(set_index).is_some_and(|s| s.visible) {
                continue;
            }
            let normal_matrix = glam::Mat3::from_mat4(model.transform).inverse().transpose();
            for tri in model.mesh.indices.chunks_exact(3) {
                let v = [
                    model.mesh.vertices[tri[0] as usize],
                    model.mesh.vertices[tri[1] as usize],
                    model.mesh.vertices[tri[2] as usize],
                ];
                let p = v.map(|v| model.transform.transform_point3(Vec3::from(v.position)));
                let n = v.map(|v| (normal_matrix * Vec3::from(v.normal)).normalize_or_zero());
                let uv = [v[0].uv, v[1].uv, v[2].uv];
                // Tangent frame from the UVs (MikkTSpace-like, per face).
                let (e1, e2) = (p[1] - p[0], p[2] - p[0]);
                let (d1, d2) = (
                    glam::Vec2::new(uv[1][0] - uv[0][0], uv[1][1] - uv[0][1]),
                    glam::Vec2::new(uv[2][0] - uv[0][0], uv[2][1] - uv[0][1]),
                );
                let det = d1.x * d2.y - d2.x * d1.y;
                let tangent = if det.abs() > 1e-12 {
                    let t = (e1 * d2.y - e2 * d1.y) / det;
                    let b = (e2 * d1.x - e1 * d2.x) / det;
                    let face = e1.cross(e2);
                    let sign = if face.cross(t).dot(b) < 0.0 { -1.0 } else { 1.0 };
                    let t = t.normalize_or_zero();
                    [t.x, t.y, t.z, sign]
                } else {
                    [0.0; 4]
                };
                let centroid = (p[0] + p[1] + p[2]) / 3.0;
                tris.push(CpuTri { p, n, uv, tangent, set: set_index, centroid });
            }
        }
        if tris.is_empty() {
            self.tris = None;
            return;
        }

        // -- median-split BVH ------------------------------------------------
        struct Builder<'a> {
            tris: &'a mut [CpuTri],
            nodes: Vec<GpuNode>,
        }
        impl Builder<'_> {
            fn build(&mut self, first: usize, count: usize) -> i32 {
                let slice = &self.tris[first..first + count];
                let mut bmin = Vec3::splat(f32::INFINITY);
                let mut bmax = Vec3::splat(f32::NEG_INFINITY);
                for t in slice {
                    for p in &t.p {
                        bmin = bmin.min(*p);
                        bmax = bmax.max(*p);
                    }
                }
                let index = self.nodes.len() as i32;
                self.nodes.push(GpuNode {
                    bmin: [bmin.x, bmin.y, bmin.z, 0.0],
                    bmax: [bmax.x, bmax.y, bmax.z, 0.0],
                    meta: [-1, -1, first as i32, count as i32],
                });
                if count <= 4 {
                    return index;
                }
                let extent = bmax - bmin;
                let axis = if extent.x > extent.y && extent.x > extent.z {
                    0
                } else if extent.y > extent.z {
                    1
                } else {
                    2
                };
                self.tris[first..first + count].sort_by(|a, b| {
                    a.centroid[axis].partial_cmp(&b.centroid[axis]).unwrap()
                });
                let half = count / 2;
                let left = self.build(first, half);
                let right = self.build(first + half, count - half);
                self.nodes[index as usize].meta = [left, right, 0, 0];
                index
            }
        }
        let mut builder = Builder { tris: &mut tris, nodes: Vec::new() };
        builder.build(0, builder.tris.len().min(usize::MAX));
        let nodes = std::mem::take(&mut builder.nodes);

        // -- upload ----------------------------------------------------------
        let tri_floats: Vec<f32> = tris
            .iter()
            .flat_map(|t| {
                let mut f = Vec::with_capacity(TRI_STRIDE * 4);
                for i in 0..3 {
                    f.extend_from_slice(&[t.p[i].x, t.p[i].y, t.p[i].z, 0.0]);
                }
                for i in 0..3 {
                    f.extend_from_slice(&[t.n[i].x, t.n[i].y, t.n[i].z, 0.0]);
                }
                f.extend_from_slice(&[t.uv[0][0], t.uv[0][1], t.uv[1][0], t.uv[1][1]]);
                f.extend_from_slice(&[t.uv[2][0], t.uv[2][1], t.set as f32, 0.0]);
                f.extend_from_slice(&t.tangent);
                f
            })
            .collect();
        let node_floats: Vec<f32> = nodes
            .iter()
            .flat_map(|n| {
                let mut f = Vec::with_capacity(12);
                f.extend_from_slice(&n.bmin);
                f.extend_from_slice(&n.bmax);
                f.extend(n.meta.iter().map(|&i| f32::from_bits(i as u32)));
                f
            })
            .collect();

        let upload = |data: Vec<f32>| -> Subbuffer<[f32]> {
            Buffer::from_iter(
                self.gpu.memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                data,
            )
            .unwrap()
        };
        self.tri_count = tris.len() as u32;
        self.tris = Some(upload(tri_floats));
        self.nodes = Some(upload(node_floats));
        self.set_params = Some(upload(set_params));
        self.textures = textures;
        self.viewport.samples = 0;
    }

    /// One more sample per pixel into the viewport, resolved into `target`
    /// (over a checkerboard where the background is transparent).
    pub fn render(
        &mut self,
        before_future: Box<dyn GpuFuture>,
        target: Arc<ImageView>,
        frame: &Frame,
        reset: bool,
        sets: &mut [TextureSet],
        renderer: &SceneRenderer,
    ) -> Box<dyn GpuFuture> {
        let mut accum = std::mem::take(&mut self.viewport);
        if reset {
            accum.samples = 0;
        }
        let future = self.render_into(&mut accum, before_future, target, Output::Viewport, frame, 1, sets, renderer);
        self.viewport = accum;
        future
    }

    /// `samples` more samples per pixel into `accum` (sized to `target`),
    /// resolved into `target` as `output` says.
    #[allow(clippy::too_many_arguments)]
    pub fn render_into(
        &mut self,
        accum: &mut Accumulator,
        before_future: Box<dyn GpuFuture>,
        target: Arc<ImageView>,
        output: Output,
        frame: &Frame,
        samples: u32,
        sets: &mut [TextureSet],
        renderer: &SceneRenderer,
    ) -> Box<dyn GpuFuture> {
        let extent = target.image().extent();
        let accum_view = accum.ensure(&self.gpu, extent);

        let mut builder = AutoCommandBufferBuilder::primary(
            self.gpu.command_buffer_allocator.clone(),
            self.gpu.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        // Fresh paint must be flattened before the tracer samples composites
        // (visible sets only, as in `SceneRenderer::render`).
        for set in sets.iter_mut().filter(|s| s.visible) {
            set.layers.record_composite(&mut builder);
        }

        let (Some(tris), Some(nodes), Some(params)) =
            (self.tris.clone(), self.nodes.clone(), self.set_params.clone())
        else {
            // No scene: nothing to trace.
            return before_future;
        };

        let env = renderer.environment();
        let table = self.textures.iter().map(|t| (t.clone(), self.sampler.clone()));
        let studio = Buffer::from_iter(
            self.gpu.memory_allocator.clone(),
            BufferCreateInfo { usage: BufferUsage::UNIFORM_BUFFER, ..Default::default() },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            frame.studio.iter().flatten().copied().collect::<Vec<f32>>(),
        )
        .unwrap();

        let ds = DescriptorSet::new(
            self.gpu.descriptor_set_allocator.clone(),
            self.trace.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::image_view(0, accum_view.clone()),
                WriteDescriptorSet::buffer(1, tris),
                WriteDescriptorSet::buffer(2, nodes),
                WriteDescriptorSet::buffer(3, params),
                WriteDescriptorSet::image_view_sampler(
                    4,
                    env.env_cube.clone(),
                    env.sampler.clone(),
                ),
                WriteDescriptorSet::image_view_sampler_array(5, 0, table),
                WriteDescriptorSet::buffer(6, studio),
            ],
            [],
        )
        .unwrap();

        let inv = frame.view_proj.inverse();
        builder
            .bind_pipeline_compute(self.trace.clone())
            .unwrap()
            .bind_descriptor_sets(PipelineBindPoint::Compute, self.trace.layout().clone(), 0, ds)
            .unwrap();
        for _ in 0..samples.max(1) {
            builder
                .push_constants(
                    self.trace.layout().clone(),
                    0,
                    trace_cs::Pc {
                        inv_view_proj: inv.to_cols_array_2d(),
                        camera_pos: frame.camera_pos.extend(1.0).to_array(),
                        params: [accum.samples as f32, 0.0, frame.bounces as f32, 0.0],
                        size: [extent[0], extent[1], 0, 0],
                    },
                )
                .unwrap();
            unsafe {
                builder.dispatch([extent[0].div_ceil(8), extent[1].div_ceil(8), 1]).unwrap();
            }
            accum.samples += 1;
        }
        self.record_resolve(&mut builder, accum_view, target, output, frame.exposure, accum.samples);

        let cb = builder.build().unwrap();
        before_future
            .then_execute(self.gpu.queue.clone(), cb)
            .unwrap()
            .boxed()
    }

    fn record_resolve(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        accum: Arc<ImageView>,
        target: Arc<ImageView>,
        output: Output,
        exposure: f32,
        samples: u32,
    ) {
        let extent = target.image().extent();
        let ds = DescriptorSet::new(
            self.gpu.descriptor_set_allocator.clone(),
            self.resolve.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::image_view(0, target),
                WriteDescriptorSet::image_view(1, accum),
            ],
            [],
        )
        .unwrap();
        builder
            .bind_pipeline_compute(self.resolve.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.resolve.layout().clone(),
                0,
                ds,
            )
            .unwrap()
            .push_constants(
                self.resolve.layout().clone(),
                0,
                resolve_cs::Pc {
                    size: [extent[0], extent[1], 0, 0],
                    params: [exposure, samples as f32, if output == Output::File { 1.0 } else { 0.0 }, 0.0],
                },
            )
            .unwrap();
        unsafe {
            builder
                .dispatch([extent[0].div_ceil(8), extent[1].div_ceil(8), 1])
                .unwrap();
        }
    }
}

mod trace_cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r#"
#version 450
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
// rgb: summed radiance; a: summed coverage (alpha).
layout(set = 0, binding = 0, rgba32f) uniform image2D accum;
// Per triangle, 9 vec4s: positions, normals, uvs (+ set), tangent (xyz,
// handedness w).
layout(set = 0, binding = 1) readonly buffer Tris { vec4 tri_data[]; };
layout(set = 0, binding = 2) readonly buffer Nodes { vec4 node_data[]; };
// Per set, 6 vec4s: [0] factors + material opacity, [1] emission + blended
// flag, [2] slots (int bits) of material base colour, painted base colour,
// painted emission, painted opacity; [3] x = 1 when the layers carry the
// material's maps, y = slot of the emission map × mask, z metallic factor,
// w roughness factor; [4] slots of material roughness, metallic and normal
// maps and painted roughness; [5] slots of painted metallic and normals,
// z specular weight.
layout(set = 0, binding = 3) readonly buffer SetParams { vec4 set_params[]; };
layout(set = 0, binding = 4) uniform samplerCube env;
layout(set = 0, binding = 5) uniform sampler2D tex[1024];
// The studio (see `studio::gpu_block`).
layout(set = 0, binding = 6) uniform Studio { vec4 s[26]; } st;
layout(push_constant) uniform Pc {
    mat4 inv_view_proj;
    vec4 camera_pos;
    vec4 params; // x: sample index, z: bounces
    uvec4 size;
} pc;

const float PI = 3.14159265359;
const int TRI = 9;
const int SET = 6;

// --- rng (pcg) --------------------------------------------------------------
uint rng_state;
uint pcg() {
    rng_state = rng_state * 747796405u + 2891336453u;
    uint word = ((rng_state >> ((rng_state >> 28u) + 4u)) ^ rng_state) * 277803737u;
    return (word >> 22u) ^ word;
}
float rand() { return float(pcg()) / 4294967296.0; }

int slot(int set, int v, int c) { return floatBitsToInt(set_params[set * SET + v][c]); }

// --- geometry ---------------------------------------------------------------
// `skip`: the triangle the ray leaves, never hit again (see `trace_ray`).
struct Hit { float t; float u; float v; int tri; int skip; };

vec2 hit_uv(int i, float u, float v) {
    vec4 uva = tri_data[i * TRI + 6];
    vec4 uvb = tri_data[i * TRI + 7];
    return uva.xy * (1.0 - u - v) + uva.zw * u + uvb.xy * v;
}

// Whether a ray stops at this point of triangle i. Coverage matches the
// raster pass: material opacity x base colour alpha, with painted opacity
// over it; cut-out materials stop at 0.5, blended ones stop with probability
// alpha (stochastic transparency converges to the blend). Lashes, brows and
// hair cards are mostly transparent texels on solid quads.
bool stops_here(int i, float u, float v) {
    vec2 tuv = hit_uv(i, u, v);
    int set = int(tri_data[i * TRI + 7].z + 0.5);
    float base = set_params[set * SET].a * textureLod(tex[slot(set, 2, 0)], tuv, 0.0).a;
    vec2 po = textureLod(tex[slot(set, 2, 3)], tuv, 0.0).xy;
    float alpha = base * (1.0 - po.y) + po.x;
    bool blended = set_params[set * SET + 1].a > 0.5;
    return blended ? rand() < alpha : alpha >= 0.5;
}

bool intersect_tri(int i, vec3 ro, vec3 rd, inout Hit hit) {
    if (i == hit.skip) return false;
    vec3 p0 = tri_data[i * TRI + 0].xyz;
    vec3 p1 = tri_data[i * TRI + 1].xyz;
    vec3 p2 = tri_data[i * TRI + 2].xyz;
    vec3 e1 = p1 - p0;
    vec3 e2 = p2 - p0;
    vec3 h = cross(rd, e2);
    float a = dot(e1, h);
    if (abs(a) < 1e-12) return false;
    float f = 1.0 / a;
    vec3 s = ro - p0;
    float u = f * dot(s, h);
    if (u < 0.0 || u > 1.0) return false;
    vec3 q = cross(s, e1);
    float v = f * dot(rd, q);
    if (v < 0.0 || u + v > 1.0) return false;
    float t = f * dot(e2, q);
    if (t > 1e-6 && t < hit.t && stops_here(i, u, v)) {
        hit.t = t; hit.u = u; hit.v = v; hit.tri = i;
        return true;
    }
    return false;
}

bool aabb_hit(vec3 bmin, vec3 bmax, vec3 ro, vec3 inv_rd, float tmax) {
    vec3 t0 = (bmin - ro) * inv_rd;
    vec3 t1 = (bmax - ro) * inv_rd;
    vec3 tsm = min(t0, t1);
    vec3 tbg = max(t0, t1);
    float tn = max(max(tsm.x, tsm.y), tsm.z);
    float tf = min(min(tbg.x, tbg.y), tbg.z);
    return tf >= max(tn, 0.0) && tn < tmax;
}

// Rays leaving a surface start a hair off it (`ro`) and ignore the
// triangle they left (`skip`), so they can't re-hit it. The offset used to
// be 1 mm, which started bounces past any sheer layer lying closer than
// that to the skin, lighting the skin under tight clothing untinted.
// With `any`, stops at the first hit closer than `tmax` (shadow rays).
Hit trace_ray(vec3 ro, vec3 rd, int skip, float tmax, bool any) {
    Hit hit;
    hit.t = tmax; hit.tri = -1; hit.u = 0.0; hit.v = 0.0; hit.skip = skip;
    vec3 inv_rd = 1.0 / rd;
    int stack[48];
    int sp = 0;
    stack[sp++] = 0;
    while (sp > 0) {
        int ni = stack[--sp];
        vec3 bmin = node_data[ni * 3 + 0].xyz;
        vec3 bmax = node_data[ni * 3 + 1].xyz;
        if (!aabb_hit(bmin, bmax, ro, inv_rd, hit.t)) continue;
        ivec4 meta = floatBitsToInt(node_data[ni * 3 + 2]);
        if (meta.x < 0) {
            for (int i = 0; i < meta.w; i++) {
                if (intersect_tri(meta.z + i, ro, rd, hit) && any) return hit;
            }
        } else if (sp < 46) {
            stack[sp++] = meta.x;
            stack[sp++] = meta.y;
        }
    }
    return hit;
}

// --- backdrop ---------------------------------------------------------------
// A floor at the subject's feet, or a sweep: the floor curving up (radius
// R) into a wall `D` behind the subject. Infinite sideways.
// The posing box (a seat), axis-aligned with the subject's facing: half
// 0.28 m wide, 0.22 m deep, from the floor up to its height.
float seat_hit(vec3 ro, vec3 rd, out vec3 n) {
    n = vec3(0.0, 1.0, 0.0);
    if (st.s[25].x < 0.5) return 1e30;
    vec3 F = vec3(st.s[17].x, 0.0, st.s[17].y);
    vec3 R = vec3(F.z, 0.0, -F.x);
    vec3 c = vec3(st.s[25].z, st.s[16].y, st.s[25].w);
    vec3 lo_ = vec3(-0.28, 0.0, -0.22), hi_ = vec3(0.28, st.s[25].y, 0.22);
    vec3 o = vec3(dot(ro - c, R), ro.y - c.y, dot(ro - c, F));
    vec3 d = vec3(dot(rd, R), rd.y, dot(rd, F));
    vec3 inv = 1.0 / d;
    vec3 t0 = (lo_ - o) * inv, t1 = (hi_ - o) * inv;
    vec3 tmin = min(t0, t1), tmax = max(t0, t1);
    float tn = max(max(tmin.x, tmin.y), tmin.z);
    float tf = min(min(tmax.x, tmax.y), tmax.z);
    if (tf < max(tn, 1e-5)) return 1e30;
    float t = tn > 1e-5 ? tn : tf;
    vec3 p = o + d * t;
    vec3 q = (p - (lo_ + hi_) * 0.5) / ((hi_ - lo_) * 0.5);
    vec3 a = abs(q);
    vec3 ln = a.x > a.y && a.x > a.z ? vec3(sign(q.x), 0.0, 0.0) : (a.y > a.z ? vec3(0.0, sign(q.y), 0.0) : vec3(0.0, 0.0, sign(q.z)));
    n = R * ln.x + vec3(0.0, ln.y, 0.0) + F * ln.z;
    return t;
}

float backdrop_set_hit(vec3 ro, vec3 rd, out vec3 n);

float backdrop_hit(vec3 ro, vec3 rd, out vec3 n) {
    vec3 sn;
    float seat = seat_hit(ro, rd, sn);
    float set = backdrop_set_hit(ro, rd, n);
    if (seat < set) {
        n = dot(sn, rd) > 0.0 ? -sn : sn;
        return seat;
    }
    return set;
}

float backdrop_set_hit(vec3 ro, vec3 rd, out vec3 n) {
    n = vec3(0.0, 1.0, 0.0);
    float kind = st.s[16].w;
    if (kind < 0.5) return 1e30;
    vec3 o = st.s[16].xyz;
    vec3 F = vec3(st.s[17].x, 0.0, st.s[17].y);
    float D = st.s[17].z;
    float R = kind > 1.5 ? st.s[17].w : 0.0;
    float oy = ro.y - o.y, dy = rd.y;
    float oz = dot(ro - o, F), dz = dot(rd, F);
    float zc = -D + R;
    float best = 1e30;
    // Floor.
    if (abs(dy) > 1e-8) {
        float t = -oy / dy;
        if (t > 1e-5 && (kind < 1.5 || oz + t * dz >= zc)) { best = t; n = vec3(0.0, 1.0, 0.0); }
    }
    if (kind > 1.5) {
        // Wall.
        if (abs(dz) > 1e-8) {
            float t = (-D - oz) / dz;
            if (t > 1e-5 && t < best && oy + t * dy >= R) { best = t; n = F; }
        }
        // The curve: a quarter cylinder along the sideways axis.
        if (R > 1e-4) {
            float py = oy - R, pz = oz - zc;
            float a = dy * dy + dz * dz;
            float b = 2.0 * (py * dy + pz * dz);
            float c = py * py + pz * pz - R * R;
            float disc = b * b - 4.0 * a * c;
            if (a > 1e-12 && disc >= 0.0) {
                float sq = sqrt(disc);
                for (int k = 0; k < 2; k++) {
                    float t = (-b + (k == 0 ? -sq : sq)) / (2.0 * a);
                    float y = oy + t * dy, z = oz + t * dz;
                    if (t > 1e-5 && t < best && y <= R && z <= zc) {
                        best = t;
                        vec2 inward = normalize(vec2(R - y, zc - z));
                        n = vec3(0.0, inward.x, 0.0) + F * inward.y;
                    }
                }
            }
        }
    }
    if (dot(n, rd) > 0.0) n = -n;
    return best;
}

vec3 env_at(vec3 d) {
    float a = st.s[20].y;
    float c = cos(a), s = sin(a);
    vec3 r = vec3(c * d.x + s * d.z, d.y, -s * d.x + c * d.z);
    return texture(env, r).rgb * st.s[20].x;
}

// --- surfaces ---------------------------------------------------------------
struct Surface {
    vec3 albedo; float metallic;
    float roughness; float spec;
    vec3 n;      // shading normal, facing the incoming ray
    vec3 ng;     // geometric normal, same side
    vec3 emit;
};

Surface mesh_surface(Hit hit, vec3 rd) {
    int i = hit.tri;
    float w = 1.0 - hit.u - hit.v;
    vec3 n = normalize(tri_data[i * TRI + 3].xyz * w + tri_data[i * TRI + 4].xyz * hit.u + tri_data[i * TRI + 5].xyz * hit.v);
    vec3 p0 = tri_data[i * TRI + 0].xyz;
    vec3 ng = normalize(cross(tri_data[i * TRI + 1].xyz - p0, tri_data[i * TRI + 2].xyz - p0));
    if (dot(ng, rd) > 0.0) ng = -ng;
    if (dot(n, ng) < 0.0) n = -n;
    vec2 tuv = hit_uv(i, hit.u, hit.v);
    int set = int(tri_data[i * TRI + 7].z + 0.5);
    vec4 factors = set_params[set * SET];
    vec4 p3 = set_params[set * SET + 3];
    bool in_layers = p3.x > 0.5;

    Surface s;
    s.albedo = (in_layers ? vec3(1.0) : texture(tex[slot(set, 2, 0)], tuv).rgb) * factors.rgb;
    vec4 pbc = texture(tex[slot(set, 2, 1)], tuv);
    s.albedo = s.albedo * (1.0 - pbc.a) + pbc.rgb;
    s.metallic = p3.z * (in_layers ? 1.0 : texture(tex[slot(set, 4, 1)], tuv).r);
    s.roughness = p3.w * (in_layers ? 1.0 : texture(tex[slot(set, 4, 0)], tuv).r);
    vec2 pr = texture(tex[slot(set, 4, 3)], tuv).xy;
    s.roughness = clamp(s.roughness * (1.0 - pr.y) + pr.x, 0.03, 1.0);
    vec2 pm = texture(tex[slot(set, 5, 0)], tuv).xy;
    s.metallic = clamp(s.metallic * (1.0 - pm.y) + pm.x, 0.0, 1.0);
    s.spec = clamp(set_params[set * SET + 5].z, 0.0, 1.0);

    // Normal maps: the material's, then the painted ones over it.
    vec4 tg = tri_data[i * TRI + 8];
    vec3 T = tg.xyz - n * dot(n, tg.xyz);
    if (dot(T, T) > 1e-12) {
        T = normalize(T);
        vec3 B = cross(n, T) * (tg.w < 0.0 ? -1.0 : 1.0);
        vec2 mxy = texture(tex[slot(set, 4, 2)], tuv).xy * 2.0 - 1.0;
        vec3 mn = vec3(mxy, sqrt(max(1.0 - dot(mxy, mxy), 0.0)));
        vec3 n1 = normalize(T * mn.x + B * mn.y + n * mn.z);
        vec4 nm = texture(tex[slot(set, 5, 1)], tuv);
        if (nm.w > 0.001) {
            vec3 pn = normalize(nm.xyz);
            n1 = normalize(mix(n1, normalize(T * pn.x + B * pn.y + n1 * pn.z), nm.w));
        }
        n = n1;
    }
    // Keep the shading normal on the side the ray came from.
    if (dot(n, -rd) < 0.02) n = normalize(n + ng * (0.02 - dot(n, -rd)));
    s.n = n;
    s.ng = ng;
    s.emit = set_params[set * SET + 1].rgb * texture(tex[floatBitsToInt(p3.y)], tuv).rgb
        + texture(tex[slot(set, 2, 2)], tuv).rgb;
    return s;
}

Surface backdrop_surface(vec3 n) {
    Surface s;
    s.albedo = st.s[18].rgb;
    s.metallic = 0.0;
    s.roughness = st.s[18].w;
    s.spec = 0.5;
    s.n = n;
    s.ng = n;
    s.emit = vec3(0.0);
    return s;
}

// --- BRDF: Lambert + GGX (Smith-Schlick), metal/rough --------------------------
vec3 fresnel(vec3 f0, float c) { return f0 + (1.0 - f0) * pow(1.0 - clamp(c, 0.0, 1.0), 5.0); }
float lum(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

void basis(vec3 n, out vec3 t, out vec3 b) {
    vec3 up = abs(n.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    t = normalize(cross(up, n));
    b = cross(n, t);
}

// f · cos for light arriving from L.
vec3 brdf_cos(Surface s, vec3 V, vec3 L) {
    float NoL = dot(s.n, L);
    float NoV = max(dot(s.n, V), 1e-4);
    if (NoL <= 0.0 || dot(s.ng, L) <= 0.0) return vec3(0.0);
    vec3 H = normalize(V + L);
    float NoH = max(dot(s.n, H), 0.0);
    float VoH = max(dot(V, H), 0.0);
    float a = max(s.roughness * s.roughness, 0.002);
    float a2 = a * a;
    float d = NoH * NoH * (a2 - 1.0) + 1.0;
    float D = a2 / (PI * d * d);
    float k = a * 0.5;
    float G = (NoV / (NoV * (1.0 - k) + k)) * (NoL / (NoL * (1.0 - k) + k));
    vec3 f0 = mix(vec3(0.04 * s.spec), s.albedo, s.metallic);
    vec3 F = fresnel(f0, VoH);
    vec3 spec = D * G * F / (4.0 * NoV * NoL);
    vec3 diff = (1.0 - F) * (1.0 - s.metallic) * s.albedo / PI;
    return (diff + spec) * NoL;
}

// Pick a bounce direction; `weight` is f·cos/pdf. False ends the path.
bool sample_brdf(Surface s, vec3 V, out vec3 L, out vec3 weight) {
    float NoV = max(dot(s.n, V), 1e-4);
    vec3 f0 = mix(vec3(0.04 * s.spec), s.albedo, s.metallic);
    vec3 Fv = fresnel(f0, NoV);
    float ws = lum(Fv);
    float wd = lum(s.albedo) * (1.0 - s.metallic) * (1.0 - ws);
    float ps = clamp(ws / max(ws + wd, 1e-4), 0.05, 0.95);
    vec3 t, b;
    basis(s.n, t, b);
    if (rand() < ps) {
        float a = max(s.roughness * s.roughness, 0.002);
        float u1 = rand(), u2 = rand();
        float phi = 2.0 * PI * u1;
        float cos_t = sqrt((1.0 - u2) / (1.0 + (a * a - 1.0) * u2));
        float sin_t = sqrt(max(1.0 - cos_t * cos_t, 0.0));
        vec3 H = normalize(t * cos(phi) * sin_t + b * sin(phi) * sin_t + s.n * cos_t);
        L = reflect(-V, H);
        float NoL = dot(s.n, L);
        if (NoL <= 0.0 || dot(s.ng, L) <= 0.0) return false;
        float VoH = max(dot(V, H), 1e-4);
        float NoH = max(dot(s.n, H), 1e-4);
        float k = a * 0.5;
        float G = (NoV / (NoV * (1.0 - k) + k)) * (NoL / (NoL * (1.0 - k) + k));
        weight = fresnel(f0, VoH) * G * VoH / (NoH * NoV) / ps;
    } else {
        float r1 = rand(), r2 = rand();
        float phi = 2.0 * PI * r1;
        float sq = sqrt(r2);
        L = normalize(t * cos(phi) * sq + b * sin(phi) * sq + s.n * sqrt(1.0 - r2));
        if (dot(s.ng, L) <= 0.0) return false;
        weight = (1.0 - Fv) * (1.0 - s.metallic) * s.albedo / (1.0 - ps);
    }
    return true;
}

// --- lights -----------------------------------------------------------------
// A direction towards light k from p (uniform over the cone it fills), its
// solid-angle pdf and the distance to it.
bool sample_light(int k, vec3 p, out vec3 L, out float pdf, out float dist) {
    vec4 pr = st.s[k * 2];
    vec3 toc = pr.xyz - p;
    float d2 = dot(toc, toc);
    float r = pr.w;
    if (d2 <= r * r) return false;
    float d = sqrt(d2);
    vec3 w = toc / d;
    float sin2 = r * r / d2;
    float cos_max = sqrt(max(1.0 - sin2, 0.0));
    float one_minus = sin2 / (1.0 + cos_max);
    float cos_t = 1.0 - rand() * one_minus;
    float sin_t = sqrt(max(1.0 - cos_t * cos_t, 0.0));
    float phi = 2.0 * PI * rand();
    vec3 t, b;
    basis(w, t, b);
    L = normalize(t * cos(phi) * sin_t + b * sin(phi) * sin_t + w * cos_t);
    pdf = 1.0 / (2.0 * PI * one_minus);
    dist = d * cos_t - sqrt(max(r * r - d2 * sin_t * sin_t, 0.0));
    return true;
}

bool blocked(vec3 p, vec3 L, float dist, int skip) {
    vec3 bn;
    if (backdrop_hit(p, L, bn) < dist) return true;
    return trace_ray(p, L, skip, dist, true).tri >= 0;
}

// One light (picked at random) lighting surface s at p.
vec3 direct(Surface s, vec3 V, vec3 p, int skip) {
    int count = int(st.s[20].w + 0.5);
    if (count == 0) return vec3(0.0);
    int k = min(int(rand() * float(count)), count - 1);
    vec3 L; float pdf; float dist;
    if (!sample_light(k, p, L, pdf, dist)) return vec3(0.0);
    vec3 f = brdf_cos(s, V, L);
    if (f == vec3(0.0) || blocked(p, L, dist, skip)) return vec3(0.0);
    return f * st.s[k * 2 + 1].rgb * float(count) / pdf;
}

// Greenscreen shadow catcher: how much of the light the backdrop would get
// at p is blocked by the subject (0 lit, 1 in full shadow).
float shadow_amount(vec3 p, vec3 n) {
    float total = 0.0, lit = 0.0;
    int count = int(st.s[20].w + 0.5);
    for (int k = 0; k < count; k++) {
        vec3 L; float pdf; float dist;
        if (!sample_light(k, p, L, pdf, dist)) continue;
        float c = max(dot(n, L), 0.0);
        vec4 pr = st.s[k * 2];
        vec3 toc = pr.xyz - p;
        float e = lum(st.s[k * 2 + 1].rgb) * c * PI * pr.w * pr.w / max(dot(toc, toc), 1e-6);
        total += e;
        if (trace_ray(p, L, -1, dist, true).tri < 0) lit += e;
    }
    // The environment, one cosine-weighted direction.
    vec3 t, b;
    basis(n, t, b);
    float r1 = rand(), r2 = rand();
    float phi = 2.0 * PI * r1;
    vec3 L = normalize(t * cos(phi) * sqrt(r2) + b * sin(phi) * sqrt(r2) + n * sqrt(1.0 - r2));
    float e = lum(env_at(L)) * PI;
    total += e;
    if (trace_ray(p, L, -1, 1e30, true).tri < 0) lit += e;
    return total > 1e-8 ? clamp(1.0 - lit / total, 0.0, 1.0) : 0.0;
}

void main() {
    uvec2 id = gl_GlobalInvocationID.xy;
    if (id.x >= pc.size.x || id.y >= pc.size.y) return;
    uint frame = uint(pc.params.x);
    rng_state = id.x * 1973u + id.y * 9277u + frame * 26699u + 1u;
    pcg();

    // Camera ray with sub-pixel jitter; a thin lens for depth of field.
    vec2 uv = (vec2(id) + vec2(rand(), rand())) / vec2(pc.size.xy);
    vec2 ndc = uv * 2.0 - 1.0;
    vec4 world = pc.inv_view_proj * vec4(ndc, 1.0, 1.0);
    vec3 ro = pc.camera_pos.xyz;
    vec3 rd = normalize(world.xyz / world.w - ro);
    float aperture = st.s[21].x;
    if (aperture > 0.0) {
        vec3 fwd = st.s[24].xyz;
        vec3 focus = ro + rd * (st.s[21].y / max(dot(rd, fwd), 1e-4));
        float r = aperture * sqrt(rand());
        float a = 2.0 * PI * rand();
        ro += (st.s[22].xyz * cos(a) + st.s[23].xyz * sin(a)) * r;
        rd = normalize(focus - ro);
    }

    int mode = int(st.s[19].w + 0.5); // 0 environment, 1 colour, 2 transparent
    vec3 radiance = vec3(0.0);
    vec3 throughput = vec3(1.0);
    float alpha = 1.0;
    int bounces = int(pc.params.z);
    int left = -1;
    for (int bounce = 0; bounce <= bounces; bounce++) {
        Hit hit = trace_ray(ro, rd, left, 1e30, false);
        vec3 bn;
        float bt = backdrop_hit(ro, rd, bn);
        bool on_backdrop = bt < hit.t;
        if (!on_backdrop && hit.tri < 0) {
            if (bounce > 0 || mode == 0) {
                radiance += throughput * env_at(rd);
            } else if (mode == 1) {
                radiance += st.s[19].rgb;
            } else {
                alpha = 0.0;
            }
            break;
        }
        vec3 hp = ro + rd * (on_backdrop ? bt : hit.t);
        if (on_backdrop && bounce == 0 && mode == 2) {
            // Green screen: the backdrop isn't in the picture; with the
            // shadow catcher, the subject's shadows on it are (in alpha).
            alpha = st.s[20].z > 0.5 ? shadow_amount(hp + bn * 1e-4, bn) : 0.0;
            break;
        }
        Surface s = on_backdrop ? backdrop_surface(bn) : mesh_surface(hit, rd);
        radiance += throughput * s.emit;
        vec3 V = -rd;
        vec3 p = hp + s.ng * (on_backdrop ? 1e-4 : 1e-5);
        int here = on_backdrop ? -1 : hit.tri;
        radiance += throughput * direct(s, V, p, here);
        vec3 L, weight;
        if (bounce == bounces || !sample_brdf(s, V, L, weight)) break;
        throughput *= weight;
        ro = p;
        rd = L;
        left = here;
        // Russian roulette after the first bounces.
        if (bounce >= 1) {
            float q = max(throughput.r, max(throughput.g, throughput.b));
            if (rand() > q) break;
            throughput /= max(q, 1e-4);
        }
    }
    // Rare paths through tiny bright highlights would speckle the image.
    radiance = min(radiance, vec3(32.0));

    vec4 prev = frame == 0u ? vec4(0.0) : imageLoad(accum, ivec2(id));
    imageStore(accum, ivec2(id), prev + vec4(radiance * alpha, alpha));
}
"#,
    }
}

mod resolve_cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
#version 450
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 0, rgba8) uniform writeonly image2D target;
layout(set = 0, binding = 1, rgba32f) uniform readonly image2D accum;
// params: x exposure, y samples, z 1 = straight alpha for a file,
// 0 = over a checkerboard for the viewport.
layout(push_constant) uniform Pc { uvec4 size; vec4 params; } pc;

vec3 aces_tonemap(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

void main() {
    uvec2 id = gl_GlobalInvocationID.xy;
    if (id.x >= pc.size.x || id.y >= pc.size.y) return;
    vec4 a = imageLoad(accum, ivec2(id));
    float alpha = clamp(a.w / max(pc.params.y, 1.0), 0.0, 1.0);
    // Radiance of the covered samples (unpremultiplied).
    vec3 color = a.w > 1e-6 ? a.rgb / a.w : vec3(0.0);
    color = aces_tonemap(color * pc.params.x);
    color = pow(color, vec3(1.0 / 2.2));
    if (pc.params.z > 0.5) {
        imageStore(target, ivec2(id), vec4(color, alpha));
    } else {
        float check = ((id.x / 12u + id.y / 12u) % 2u) == 0u ? 0.32 : 0.24;
        imageStore(target, ivec2(id), vec4(mix(vec3(check), color, alpha), 1.0));
    }
}
",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{MeshData, Vertex3D};
    use crate::scene::{ImageData, MaterialData, ModelData};
    use vulkano::command_buffer::{CopyImageToBufferInfo, PrimaryCommandBufferAbstract as _};

    fn quad(z: f32, name: &str, material: MaterialData) -> ModelData {
        let v = |x: f32, y: f32, u: f32, w: f32| Vertex3D { position: [x, y, z], normal: [0.0, 0.0, 1.0], uv: [u, w] };
        ModelData {
            mesh: MeshData {
                vertices: vec![v(-1.0, -1.0, 0.0, 1.0), v(1.0, -1.0, 1.0, 1.0), v(1.0, 1.0, 1.0, 0.0), v(-1.0, 1.0, 0.0, 0.0)],
                indices: vec![0, 1, 2, 0, 2, 3],
            },
            material,
            transform: Mat4::IDENTITY,
            name: name.into(),
            material_name: "M".into(),
            morphs: Default::default(),
            skin: None,
        }
    }

    /// Path-traced centre pixel of a white card (texture `card_alpha`
    /// everywhere, optionally a blended material) in front of a red quad.
    fn centre_pixel(card_alpha: u8, transparent: bool) -> Option<[u8; 4]> {
        let card = MaterialData {
            base_color: [1.0; 4],
            base_color_image: Some(ImageData { width: 4, height: 4, rgba: [255, 255, 255, card_alpha].repeat(16).into() }),
            transparent,
            ..MaterialData::default()
        };
        let red = MaterialData { base_color: [1.0, 0.0, 0.0, 1.0], ..MaterialData::default() };
        trace_centre(vec![quad(0.5, "Card", card), quad(0.0, "Wall", red)])
    }

    /// Path-traced centre pixel of `models`, seen head-on from +Z.
    fn trace_centre(models: Vec<ModelData>) -> Option<[u8; 4]> {
        trace_mean(models, 0)
    }

    /// Mean of the `(2r+1)²` pixels around the centre, to average out noise.
    fn trace_mean(models: Vec<ModelData>, r: u32) -> Option<[u8; 4]> {
        trace_studio(models, r, &crate::studio::Studio::default())
    }

    /// The same, in `studio`.
    fn trace_studio(models: Vec<ModelData>, r: u32, studio: &crate::studio::Studio) -> Option<[u8; 4]> {
        let gpu = crate::environment::test_gpu::context()?;
        let environment = crate::environment::Environment::procedural(&gpu);
        let mut renderer = SceneRenderer::new(
            gpu.queue.clone(),
            gpu.memory_allocator.clone(),
            gpu.command_buffer_allocator.clone(),
            gpu.descriptor_set_allocator.clone(),
            Format::R8G8B8A8_UNORM,
            environment,
        );
        renderer.set_models(&models);
        let mut sets: Vec<TextureSet> = crate::scene::set_keys(&models)
            .into_iter()
            .map(|k| TextureSet::new_lazy(k.object, k.material, gpu.clone(), 64))
            .collect();

        let size = 32;
        let target = ImageView::new_default(
            crate::vram::image(
                gpu.memory_allocator.clone(),
                ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format: Format::R8G8B8A8_UNORM,
                    extent: [size, size, 1],
                    usage: ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC,
                    ..Default::default()
                },
                crate::vram::Kind::Viewport,
            )
            .unwrap(),
        )
        .unwrap();
        let eye = Vec3::new(0.0, 0.0, 4.0);
        let view_proj = glam::camera::rh::proj::vulkan::perspective(0.6, 1.0, 0.1, 100.0)
            * glam::camera::rh::view::look_at_mat4(eye, Vec3::ZERO, Vec3::Y);
        let mut tracer = PathTracer::new(gpu.clone());
        tracer.rebuild_scene(&renderer, &sets, &models);
        let subject = crate::studio::Subject { center: Vec3::ZERO, floor: -1.0, facing: glam::Quat::IDENTITY, seat: Vec3::ZERO };
        let lens = crate::studio::Lens { position: eye, right: Vec3::X, up: Vec3::Y, forward: -Vec3::Z, fov_y: 0.6 };
        let block = crate::studio::gpu_block(studio, &subject, 1.0, &lens);
        let frame = Frame { view_proj, camera_pos: eye, studio: &block, bounces: 3, exposure: 1.0 };
        for i in 0..32 {
            let future = vulkano::sync::now(gpu.queue.device().clone()).boxed();
            let done = tracer.render(future, target.clone(), &frame, i == 0, &mut sets, &renderer);
            done.then_signal_fence_and_flush().unwrap().wait(None).unwrap();
        }

        let buffer: Subbuffer<[u8]> = Buffer::new_slice(
            gpu.memory_allocator.clone(),
            BufferCreateInfo { usage: BufferUsage::TRANSFER_DST, ..Default::default() },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            (size * size * 4) as u64,
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
        let mut sum = [0u32; 4];
        for y in size / 2 - r..=size / 2 + r {
            for x in size / 2 - r..=size / 2 + r {
                let i = ((y * size + x) * 4) as usize;
                (0..4).for_each(|c| sum[c] += px[i + c] as u32);
            }
        }
        let n = (2 * r + 1) * (2 * r + 1);
        Some(sum.map(|v| (v / n) as u8))
    }

    fn redness(p: [u8; 4]) -> f32 {
        p[0] as f32 / (p[1] as f32 + p[2] as f32 + 1.0) * 2.0
    }

    /// A card whose texture is transparent is see-through, as in the raster
    /// view (lashes, brows, hair cards); an opaque one still blocks.
    #[test]
    fn transparent_texels_let_rays_through() {
        let Some(solid) = centre_pixel(255, false) else {
            eprintln!("no Vulkan device; skipping");
            return;
        };
        let cutout = centre_pixel(0, false).unwrap();
        let blended = centre_pixel(0, true).unwrap();
        assert!(redness(solid) < 1.5, "opaque white card should hide the wall: {solid:?}");
        assert!(redness(cutout) > 3.0, "cut-out texels should show the red wall: {cutout:?}");
        assert!(redness(blended) > 3.0, "fully transparent blended texels should show the wall: {blended:?}");
    }

    /// Sheer fabric lying on skin filters the light the skin bounces back.
    /// Bounces used to start 1 mm off the surface, so where a sheer layer sat
    /// closer than that the skin under it lit up untinted: white patches
    /// with sharp edges wherever clothing fits tight.
    #[test]
    fn sheer_layers_tint_skin_however_tight_they_fit() {
        let traced = |gap: f32| {
            let sheer = MaterialData {
                base_color: [0.2, 1.0, 0.1, 1.0],
                base_color_image: Some(ImageData { width: 4, height: 4, rgba: [255, 255, 255, 140].repeat(16).into() }),
                transparent: true,
                ..MaterialData::default()
            };
            trace_mean(vec![quad(gap, "Top", sheer), quad(0.0, "Skin", MaterialData::default())], 4)
        };
        let Some(loose) = traced(0.02) else {
            eprintln!("no Vulkan device; skipping");
            return;
        };
        let tight = traced(0.0003).unwrap();
        let green = |p: [u8; 4]| p[1] as f32 / (p[0] as f32 + p[2] as f32 + 1.0);
        assert!(green(tight) > green(loose) * 0.85, "tight {tight:?} should be as green as loose {loose:?}");
    }

    /// Avatars with many outfits have far more texture sets than the old
    /// fixed descriptor arrays held (32); sets past that were dropped, so
    /// most of the clothing vanished in the traced view.
    #[test]
    fn sets_past_the_32nd_are_traced() {
        let mut models: Vec<ModelData> = (0..39)
            .map(|i| {
                let mut m = quad(0.0, &format!("Piece{i}"), MaterialData::default());
                m.transform = Mat4::from_translation(Vec3::new(100.0 + i as f32 * 3.0, 0.0, 0.0));
                m
            })
            .collect();
        let red = MaterialData { base_color: [1.0, 0.0, 0.0, 1.0], ..MaterialData::default() };
        models.push(quad(0.0, "Wall", red));
        let Some(px) = trace_centre(models) else {
            eprintln!("no Vulkan device; skipping");
            return;
        };
        assert!(redness(px) > 3.0, "set 39 (the red wall) should be traced: {px:?}");
    }

    /// Material emission goes through its map × mask: an avatar's eye with a
    /// near-white emission colour and a mask that only lights the iris used
    /// to glow flat white all over, hiding the base colour and paint.
    #[test]
    fn emission_mask_limits_the_glow() {
        let glowing = |mask: u8| {
            let material = MaterialData {
                base_color: [1.0, 0.0, 0.0, 1.0],
                emission: [1.0; 3],
                emission_mask_image: Some(ImageData { width: 2, height: 2, rgba: [mask, mask, mask, 255].repeat(4).into() }),
                ..MaterialData::default()
            };
            trace_centre(vec![quad(0.0, "Eye", material)])
        };
        let Some(masked) = glowing(0) else {
            eprintln!("no Vulkan device; skipping");
            return;
        };
        let lit = glowing(255).unwrap();
        assert!(redness(masked) > 3.0, "a black mask leaves the red base colour: {masked:?}");
        assert!(redness(lit) < 1.5, "a white mask lets the white emission through: {lit:?}");
    }

    /// Trace `models` in `studio` from `eye` looking at `look`, 64 samples,
    /// resolved as for a file (straight alpha); all `size`² RGBA pixels.
    fn trace_image(models: Vec<ModelData>, studio: &crate::studio::Studio, subject: crate::studio::Subject, eye: Vec3, look: Vec3, size: u32) -> Option<Vec<u8>> {
        let gpu = crate::environment::test_gpu::context()?;
        let environment = crate::environment::Environment::procedural(&gpu);
        let mut renderer = SceneRenderer::new(
            gpu.queue.clone(),
            gpu.memory_allocator.clone(),
            gpu.command_buffer_allocator.clone(),
            gpu.descriptor_set_allocator.clone(),
            Format::R8G8B8A8_UNORM,
            environment,
        );
        renderer.set_models(&models);
        let mut sets: Vec<TextureSet> = crate::scene::set_keys(&models)
            .into_iter()
            .map(|k| TextureSet::new_lazy(k.object, k.material, gpu.clone(), 64))
            .collect();
        let target = ImageView::new_default(
            crate::vram::image(
                gpu.memory_allocator.clone(),
                ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format: Format::R8G8B8A8_UNORM,
                    extent: [size, size, 1],
                    usage: ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC,
                    ..Default::default()
                },
                crate::vram::Kind::Viewport,
            )
            .unwrap(),
        )
        .unwrap();
        let view = glam::camera::rh::view::look_at_mat4(eye, look, Vec3::Y);
        let view_proj = glam::camera::rh::proj::vulkan::perspective(0.6, 1.0, 0.05, 100.0) * view;
        let forward = (look - eye).normalize();
        let right = forward.cross(Vec3::Y).normalize();
        let lens = crate::studio::Lens { position: eye, right, up: right.cross(forward), forward, fov_y: 0.6 };
        let block = crate::studio::gpu_block(studio, &subject, 1.0, &lens);
        let frame = Frame { view_proj, camera_pos: eye, studio: &block, bounces: 2, exposure: 1.0 };
        let mut tracer = PathTracer::new(gpu.clone());
        tracer.rebuild_scene(&renderer, &sets, &models);
        let mut accum = Accumulator::default();
        let future = vulkano::sync::now(gpu.queue.device().clone()).boxed();
        tracer
            .render_into(&mut accum, future, target.clone(), Output::File, &frame, 64, &mut sets, &renderer)
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
        assert_eq!(accum.samples, 64);
        let buffer: Subbuffer<[u8]> = Buffer::new_slice(
            gpu.memory_allocator.clone(),
            BufferCreateInfo { usage: BufferUsage::TRANSFER_DST, ..Default::default() },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            (size * size * 4) as u64,
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
        let px = buffer.read().unwrap().to_vec();
        Some(px)
    }

    fn pixel(px: &[u8], size: u32, x: u32, y: u32) -> [u8; 4] {
        let i = ((y * size + x) * 4) as usize;
        [px[i], px[i + 1], px[i + 2], px[i + 3]]
    }

    /// A studio light in front lights a card the dark environment leaves
    /// black, and a glossy card shows its highlight.
    #[test]
    fn studio_lights_light_the_subject() {
        use crate::studio::{Background, Light, Studio, Subject};
        let subject = Subject { center: Vec3::ZERO, floor: -1.0, facing: glam::Quat::IDENTITY, seat: Vec3::ZERO };
        let dark = Studio { environment: 0.0, background: Background::Color, background_color: [0.0; 3], ..Studio::default() };
        let card = || vec![quad(0.0, "Card", MaterialData { base_color: [0.8, 0.8, 0.8, 1.0], ..MaterialData::default() })];
        let eye = Vec3::new(0.0, 0.0, 4.0);
        let Some(unlit) = trace_image(card(), &dark, subject, eye, Vec3::ZERO, 16) else {
            eprintln!("no Vulkan device; skipping");
            return;
        };
        let lit_studio = Studio { lights: vec![Light { azimuth: 0.0, elevation: 20.0, distance: 3.0, size: 0.3, intensity: 1.0, ..Light::default() }], ..dark.clone() };
        let lit = trace_image(card(), &lit_studio, subject, eye, Vec3::ZERO, 16).unwrap();
        let (u, l) = (pixel(&unlit, 16, 8, 8), pixel(&lit, 16, 8, 8));
        assert!(u[0] < 10, "no light, no environment: black {u:?}");
        assert!(l[0] > 120, "the key light lights the card: {l:?}");
        assert_eq!(l[3], 255, "the card covers the pixel");
    }

    /// Transparent background: alpha 0 where nothing is, and with the
    /// shadow catcher the floor is invisible except for the shadow cast on
    /// it, which stays in the alpha.
    #[test]
    fn transparent_background_keeps_only_subject_and_shadow() {
        use crate::studio::{Backdrop, Background, Light, Studio, Subject};
        // A small card high over the floor, lit from straight above.
        let blocker = {
            let mut m = quad(0.0, "Blocker", MaterialData::default());
            m.transform = Mat4::from_translation(Vec3::new(0.0, 1.5, 0.0))
                * Mat4::from_rotation_x(-std::f32::consts::FRAC_PI_2)
                * Mat4::from_scale(Vec3::splat(0.3));
            m
        };
        let subject = Subject { center: Vec3::new(0.0, 1.0, 0.0), floor: 0.0, facing: glam::Quat::IDENTITY, seat: Vec3::ZERO };
        let studio = Studio {
            lights: vec![Light { azimuth: 0.0, elevation: 89.9, distance: 3.0, size: 0.05, intensity: 1.0, ..Light::default() }],
            environment: 0.0,
            background: Background::Transparent,
            backdrop: Backdrop::Floor,
            shadow_catcher: true,
            ..Studio::default()
        };
        let size = 32;
        let eye = Vec3::new(0.0, 1.0, 4.0);
        let Some(px) = trace_image(vec![blocker.clone()], &studio, subject, eye, Vec3::ZERO, size) else {
            eprintln!("no Vulkan device; skipping");
            return;
        };
        let shadow = pixel(&px, size, size / 2, size / 2);
        let lit_floor = pixel(&px, size, size / 2, size - 1);
        let sky = pixel(&px, size, size / 2, 0);
        assert!(shadow[3] > 200 && shadow[0] < 30, "the shadow under the card is kept, black: {shadow:?}");
        assert!(lit_floor[3] < 30, "the lit floor is see-through: {lit_floor:?}");
        assert_eq!(sky[3], 0, "nothing behind: fully transparent {sky:?}");
        let without = Studio { shadow_catcher: false, ..studio };
        let px = trace_image(vec![blocker], &without, subject, eye, Vec3::ZERO, size).unwrap();
        assert_eq!(pixel(&px, size, size / 2, size / 2)[3], 0, "no catcher: the floor and its shadow are gone");
    }

    /// Real-file look check: ZESTSURFACE_TEST_FBX=avatar.fbx and
    /// ZESTSURFACE_RENDER_OUT=dir render the avatar in each built-in pose
    /// named in ZESTSURFACE_RENDER_POSES (comma separated) in the default photo
    /// studio, and once on a transparent background, as PNGs.
    #[test]
    fn render_real_avatar() {
        let (Ok(path), Ok(out)) = (std::env::var("ZESTSURFACE_TEST_FBX"), std::env::var("ZESTSURFACE_RENDER_OUT")) else {
            eprintln!("ZESTSURFACE_TEST_FBX / ZESTSURFACE_RENDER_OUT not set; skipping");
            return;
        };
        let poses = std::env::var("ZESTSURFACE_RENDER_POSES").unwrap_or_else(|_| "Contrapposto".into());
        let Some(gpu) = crate::environment::test_gpu::context() else { return };
        let mut models = crate::fbx_loader::load_with_progress(std::path::Path::new(&path), &|_, _, _| {}).unwrap();
        let skeleton = models.iter().find_map(|m| m.skin.as_ref()).unwrap().skeleton.clone();
        let map = crate::skeleton::humanoid_map(&skeleton);
        let environment = crate::environment::Environment::procedural(&gpu);
        let mut renderer = SceneRenderer::new(
            gpu.queue.clone(),
            gpu.memory_allocator.clone(),
            gpu.command_buffer_allocator.clone(),
            gpu.descriptor_set_allocator.clone(),
            Format::R8G8B8A8_UNORM,
            environment,
        );
        renderer.set_models(&models);
        let mut sets: Vec<TextureSet> = crate::scene::set_keys(&models)
            .into_iter()
            .map(|k| TextureSet::new_lazy(k.object, k.material, gpu.clone(), 64))
            .collect();
        let (w, h) = (512u32, 768u32);
        let target = ImageView::new_default(
            crate::vram::image(
                gpu.memory_allocator.clone(),
                ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format: Format::R8G8B8A8_UNORM,
                    extent: [w, h, 1],
                    usage: ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC,
                    ..Default::default()
                },
                crate::vram::Kind::Viewport,
            )
            .unwrap(),
        )
        .unwrap();
        let mut runs: Vec<(String, crate::studio::Studio)> =
            poses.split(',').map(|p| (p.trim().to_string(), crate::studio::Studio::photo_default())).collect();
        runs.push((
            format!("{} transparent", runs[0].0),
            crate::studio::Studio { background: crate::studio::Background::Transparent, ..crate::studio::Studio::photo_default() },
        ));
        for (name, studio) in &mut runs {
            studio.seat = name.starts_with("Sitting");
        }
        let anim = std::env::var("ZESTSURFACE_RENDER_ANIM").ok().map(|p| crate::poses::load(std::path::Path::new(&p)).unwrap());
        if let Some(source) = &anim {
            eprintln!("{}: {} body bones", source.file_name(), source.body_bones());
            runs.push(("Animation".into(), crate::studio::Studio::photo_default()));
        }
        for (name, studio) in runs {
            let pose_name = name.trim_end_matches(" transparent");
            let pose = match (&anim, pose_name) {
                (Some(source), "Animation") => source.pose(0, source.clips[0].frames.len() / 3),
                _ => crate::poses::builtin().into_iter().find(|p| p.name == pose_name),
            };
            let world = std::sync::Arc::new(skeleton.world(&crate::skeleton::retarget(pose.as_ref(), &skeleton, &map, false, &Default::default())));
            for (i, m) in models.iter_mut().enumerate() {
                if let Some(skin) = &mut m.skin {
                    skin.pose = Some(world.clone());
                }
                m.deform();
                renderer.update_vertices(i, &m.mesh.vertices);
            }
            let (lo, hi) = models.iter().fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(lo, hi), m| {
                let (a, b) = m.mesh.bounds();
                (lo.min(m.transform.transform_point3(a)), hi.max(m.transform.transform_point3(b)))
            });
            let center = (lo + hi) * 0.5;
            let hips = world[map[crate::skeleton::HIPS].unwrap()].w_axis.truncate();
            let thigh = [14usize, 18].iter().filter_map(|&b| map[b]).map(|j| world[j].w_axis.y).fold(f32::MAX, f32::min);
            let subject = crate::studio::Subject {
                center,
                floor: lo.y,
                facing: crate::skeleton::rig_facing(&skeleton, &map),
                seat: Vec3::new(hips.x, thigh - lo.y - 0.07, hips.z),
            };
            let fov = crate::studio::fov_for_focal_length(70.0);
            let eye = center + Vec3::new(1.0, 0.1, 4.0).normalize() * ((hi.y - lo.y) * 0.62 / (fov * 0.5).tan());
            let view = glam::camera::rh::view::look_at_mat4(eye, center, Vec3::Y);
            let view_proj = glam::camera::rh::proj::vulkan::perspective(fov, w as f32 / h as f32, 0.05, 100.0) * view;
            let forward = (center - eye).normalize();
            let right = forward.cross(Vec3::Y).normalize();
            let lens = crate::studio::Lens { position: eye, right, up: right.cross(forward), forward, fov_y: fov };
            let block = crate::studio::gpu_block(&studio, &subject, 1.0, &lens);
            let frame = Frame { view_proj, camera_pos: eye, studio: &block, bounces: studio.bounces, exposure: 1.0 };
            let mut tracer = PathTracer::new(gpu.clone());
            tracer.rebuild_scene(&renderer, &sets, &models);
            let mut accum = Accumulator::default();
            let started = std::time::Instant::now();
            for _ in 0..16 {
                let future = vulkano::sync::now(gpu.queue.device().clone()).boxed();
                tracer
                    .render_into(&mut accum, future, target.clone(), Output::File, &frame, 8, &mut sets, &renderer)
                    .then_signal_fence_and_flush()
                    .unwrap()
                    .wait(None)
                    .unwrap();
            }
            eprintln!("{name}: {} spp in {:.1} s", accum.samples, started.elapsed().as_secs_f32());
            let buffer: Subbuffer<[u8]> = Buffer::new_slice(
                gpu.memory_allocator.clone(),
                BufferCreateInfo { usage: BufferUsage::TRANSFER_DST, ..Default::default() },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_RANDOM_ACCESS,
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
            builder.copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(target.image().clone(), buffer.clone())).unwrap();
            builder.build().unwrap().execute(gpu.queue.clone()).unwrap().then_signal_fence_and_flush().unwrap().wait(None).unwrap();
            let img = image::RgbaImage::from_raw(w, h, buffer.read().unwrap().to_vec()).unwrap();
            img.save(std::path::Path::new(&out).join(format!("{name}.png"))).unwrap();
        }
    }
}
