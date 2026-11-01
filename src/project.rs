//! Project files (`.zestsurface`): the whole document, saved so it can be
//! opened again and painted on exactly where it was left.
//!
//! A project is a zip archive:
//!
//! - `project.json` — the manifest: models, texture sets, every layer's
//!   settings, the camera and display settings, and a table of textures.
//! - `meshes/<n>.bin` — each model's geometry, including UVs, so a re-unwrap
//!   survives a save without the source file.
//! - `images/<hash>.png` — the materials' own map images (lossless PNG).
//! - `textures/<n>.raw` — every layer channel, layer mask and baked map,
//!   byte for byte as the GPU holds it (half or full float). Nothing is
//!   re-encoded, so a saved project reopens with no quantisation at all.
//!   All-zero textures (unused channels, the common case) are not stored.
//!
//! Saving is split in two: `capture` reads the GPU textures back on the
//! main thread (it needs the queue), then `write` compresses and writes on
//! a worker. Opening is the mirror image: `read` on a worker, `restore` on
//! the main thread to upload.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::io::{Cursor, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, Context as _, Result};
use serde::{Deserialize, Serialize};
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, ClearColorImageInfo, CommandBufferUsage, CopyBufferToImageInfo,
    CopyImageToBufferInfo, PrimaryCommandBufferAbstract,
};
use vulkano::format::{ClearColorValue, Format};
use vulkano::image::view::ImageView;
use vulkano::image::{ImageCreateInfo, ImageType, ImageUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::sync::GpuFuture;

use crate::environment::GpuContext;
use crate::layers::{Axis, BlendMode, Layer, MaskKind, Projection, ShaderMask, TextureSet};
use crate::mesh::{MeshData, Vertex3D};
use crate::scene::{ImageData, MaterialData, ModelData};

/// File extension, without the dot.
pub const EXTENSION: &str = "zestsurface";
/// Projects saved while the app was still called ZestPaint.
pub const LEGACY_EXTENSION: &str = "zestpaint";
const FORMAT_TAG: &str = "zestsurface-project";
const LEGACY_FORMAT_TAG: &str = "zestpaint-project";
/// Bumped when the layout changes in a way older builds cannot read.
/// 2: more blend modes, clipping, mask on/off.
/// 3: matcaps are layers (each with its own mask), not two slots per set.
const VERSION: u32 = 3;
const MESH_MAGIC: &[u8; 8] = b"ZPMESH1\0";
const MORPH_MAGIC: &[u8; 8] = b"ZPMORPH1";
const SKIN_MAGIC: &[u8; 8] = b"ZPSKIN1\0";

// -- manifest ----------------------------------------------------------------

#[derive(Serialize, Deserialize)]
struct Manifest {
    format: String,
    version: u32,
    name: String,
    active_set: usize,
    camera: CameraState,
    display: DisplayState,
    models: Vec<ModelEntry>,
    sets: Vec<SetEntry>,
    textures: Vec<TextureEntry>,
    /// The rig the models' skins use, when they are rigged.
    #[serde(default)]
    skeleton: Option<SkeletonEntry>,
    /// The app's pose and studio (lights, backdrop, render) settings, kept
    /// as it writes them.
    #[serde(default)]
    pose: Option<serde_json::Value>,
    #[serde(default)]
    studio: Option<serde_json::Value>,
}

#[derive(Serialize, Deserialize)]
struct SkeletonEntry {
    joints: Vec<JointEntry>,
}

#[derive(Serialize, Deserialize)]
struct JointEntry {
    name: String,
    parent: Option<usize>,
    translation: [f32; 3],
    rotation: [f32; 4],
    scale: [f32; 3],
    /// World transform of what a top joint hangs from.
    #[serde(default)]
    base: Option<[f32; 16]>,
}

impl SkeletonEntry {
    fn of(s: &crate::skeleton::Skeleton) -> Self {
        SkeletonEntry {
            joints: s
                .joints
                .iter()
                .map(|j| JointEntry {
                    name: j.name.clone(),
                    parent: j.parent,
                    translation: j.rest.t.to_array(),
                    rotation: j.rest.r.to_array(),
                    scale: j.rest.s.to_array(),
                    base: (j.base != glam::Mat4::IDENTITY).then(|| j.base.to_cols_array()),
                })
                .collect(),
        }
    }

    fn skeleton(&self) -> Result<crate::skeleton::Skeleton> {
        let mut joints = Vec::with_capacity(self.joints.len());
        for (i, j) in self.joints.iter().enumerate() {
            if j.parent.is_some_and(|p| p >= i) {
                bail!("the skeleton is damaged (joint {i} comes before its parent)");
            }
            joints.push(crate::skeleton::Joint {
                name: j.name.clone(),
                parent: j.parent,
                rest: crate::skeleton::Trs {
                    t: j.translation.into(),
                    r: glam::Quat::from_array(j.rotation),
                    s: j.scale.into(),
                },
                base: j.base.map_or(glam::Mat4::IDENTITY, |m| glam::Mat4::from_cols_array(&m)),
            });
        }
        Ok(crate::skeleton::Skeleton { joints })
    }
}

/// A part's bones: skeleton joint and geometry-to-bone matrix per slot;
/// the per-vertex weights are in `data` (see `encode_skin`).
#[derive(Serialize, Deserialize)]
struct SkinEntry {
    joints: Vec<u32>,
    binds: Vec<[f32; 16]>,
    data: String,
}

/// Where the orbit camera was.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub struct CameraState {
    pub target: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
}

/// Viewport lighting. Not part of the paint, but part of how the work was
/// being looked at, so it comes back with it.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct DisplayState {
    pub exposure: f32,
    pub env_intensity: f32,
    pub key_light: f32,
    pub key_azimuth: f32,
    pub key_elevation: f32,
    pub key_follow_camera: bool,
    pub ao_strength: f32,
    /// The HDRI in use; None for the procedural sky. Referenced, not
    /// embedded — environments are large and live in a shared library.
    #[serde(default)]
    pub env_path: Option<PathBuf>,
}

#[derive(Serialize, Deserialize)]
struct ModelEntry {
    name: String,
    material_name: String,
    transform: [f32; 16],
    /// Archive path of the geometry (undeformed, when it has blendshapes).
    mesh: String,
    material: MaterialEntry,
    /// Blendshapes, in the order of the offsets in `morph_data`.
    #[serde(default)]
    morphs: Vec<MorphEntry>,
    /// Archive path of the blendshape offsets (see `encode_morphs`).
    #[serde(default)]
    morph_data: Option<String>,
    #[serde(default)]
    skin: Option<SkinEntry>,
}

#[derive(Serialize, Deserialize)]
struct MorphEntry {
    name: String,
    weight: f32,
}

#[derive(Serialize, Deserialize)]
struct MaterialEntry {
    base_color: [f32; 4],
    metallic: f32,
    roughness: f32,
    emission: [f32; 3],
    specular_weight: f32,
    coat_weight: f32,
    coat_roughness: f32,
    sheen_weight: f32,
    sheen_roughness: f32,
    opacity: f32,
    transparent: bool,
    /// Archive paths of the map images.
    #[serde(default)]
    base_color_image: Option<String>,
    #[serde(default)]
    roughness_image: Option<String>,
    #[serde(default)]
    metallic_image: Option<String>,
    #[serde(default)]
    normal_image: Option<String>,
    #[serde(default)]
    height_image: Option<String>,
    #[serde(default)]
    ao_image: Option<String>,
    #[serde(default)]
    emission_image: Option<String>,
    #[serde(default)]
    emission_mask_image: Option<String>,
}

#[derive(Serialize, Deserialize)]
struct SetEntry {
    object: String,
    material: String,
    visible: bool,
    height_scale: f32,
    active_layer: usize,
    next_layer_number: u32,
    /// Bounds the projected fills are framed to: centre xyz, longest side.
    #[serde(default)]
    projection_frame: Option<[f32; 4]>,
    #[serde(default)]
    baked_ao: Option<u32>,
    #[serde(default)]
    baked_position: Option<u32>,
    #[serde(default)]
    baked_normal: Option<u32>,
    #[serde(default)]
    baked_curvature: Option<u32>,
    /// Bottom to top, as in the stack.
    layers: Vec<LayerEntry>,
    /// Paint resolution; absent in projects from before it could vary.
    #[serde(default)]
    resolution: Option<u32>,
    /// Matcap and Matcap 2 of format-2 projects, from before matcaps were
    /// layers; read only, to turn them into matcap layers.
    #[serde(default, skip_serializing)]
    matcaps: Vec<MatcapEntry>,
    /// The material's own textures were moved into a "Base material" layer.
    #[serde(default)]
    material_in_layers: bool,
}

/// A matcap layer's matcap (or, in format 2, a set's matcap slot). `image`
/// names a PNG in `images/`.
#[derive(Serialize, Deserialize)]
struct MatcapEntry {
    /// Only format-2 slots had this; a layer's visibility does it now.
    #[serde(default = "yes", skip_serializing)]
    enabled: bool,
    name: String,
    image: Option<String>,
    blend: crate::matcap::MatcapBlend,
    strength: f32,
}

/// Every editable property of a layer. Texture fields index `textures`.
#[derive(Serialize, Deserialize)]
struct LayerEntry {
    id: u64,
    name: String,
    visible: bool,
    opacity: f32,
    blend: BlendMode,
    basecolor: u32,
    roughness: u32,
    metallic: u32,
    normal: u32,
    height: u32,
    emissive: u32,
    opacity_map: u32,
    mask: Option<u32>,
    mask_kind: Option<MaskKind>,
    #[serde(default = "yes")]
    mask_enabled: bool,
    #[serde(default)]
    clip: bool,
    uv_scale: [f32; 2],
    uv_offset: [f32; 2],
    uv_rotation: f32,
    projection: Projection,
    projection_axis: Axis,
    triplanar_sharpness: f32,
    projection_offset: [f32; 3],
    projection_rotation: [f32; 3],
    projection_scale: f32,
    projection_repeat: bool,
    has_fill: bool,
    purpose: Option<ShaderMask>,
    /// On a matcap layer; absent before format 3.
    #[serde(default)]
    matcap: Option<MatcapEntry>,
    swatch: Option<[f32; 3]>,
    used: u8,
}

/// Serde default for fields that were implicitly on before they existed.
fn yes() -> bool {
    true
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
enum TexFormat {
    Rgba16f,
    Rg16f,
    Rgba32f,
}

impl TexFormat {
    fn of(format: Format) -> Result<Self> {
        Ok(match format {
            Format::R16G16B16A16_SFLOAT => TexFormat::Rgba16f,
            Format::R16G16_SFLOAT => TexFormat::Rg16f,
            Format::R32G32B32A32_SFLOAT => TexFormat::Rgba32f,
            other => bail!("cannot save a texture in format {other:?}"),
        })
    }

    fn vulkan(self) -> Format {
        match self {
            TexFormat::Rgba16f => Format::R16G16B16A16_SFLOAT,
            TexFormat::Rg16f => Format::R16G16_SFLOAT,
            TexFormat::Rgba32f => Format::R32G32B32A32_SFLOAT,
        }
    }

    fn texel_bytes(self) -> usize {
        match self {
            TexFormat::Rgba16f => 8,
            TexFormat::Rg16f => 4,
            TexFormat::Rgba32f => 16,
        }
    }
}

#[derive(Serialize, Deserialize)]
struct TextureEntry {
    format: TexFormat,
    size: u32,
    /// Archive path of the raw texels; None = all zero, recreated cleared.
    file: Option<String>,
}

// -- save --------------------------------------------------------------------

/// What the app hands `capture`: the document and the view settings.
pub struct SceneRefs<'a> {
    pub name: String,
    pub models: &'a [ModelData],
    pub sets: &'a [TextureSet],
    /// Visibility to save per set, when it differs from the live flags: in
    /// local view, what was visible before isolating — a project is never
    /// saved stuck in local view.
    pub visibility: Option<&'a [bool]>,
    pub active_set: usize,
    pub camera: CameraState,
    pub display: DisplayState,
    pub pose: Option<serde_json::Value>,
    pub studio: Option<serde_json::Value>,
}

/// A project read back into host memory, ready to be written on any thread.
pub struct Snapshot {
    manifest: Manifest,
    meshes: Vec<Vec<u8>>,
    /// Blendshape offsets per model, for those that have any.
    morphs: Vec<Option<Vec<u8>>>,
    /// Skin weights per model, for rigged ones.
    skins: Vec<Option<Vec<u8>>>,
    images: Vec<(String, ImageData)>,
    textures: Vec<Option<Vec<u8>>>,
}

/// Read the document off the GPU. Main thread; blocks on the readbacks.
pub fn capture(gpu: &GpuContext, scene: SceneRefs) -> Result<Snapshot> {
    let mut textures = TextureCollector::default();
    let mut images = ImageCollector::default();

    let mut meshes = Vec::with_capacity(scene.models.len());
    let mut models = Vec::with_capacity(scene.models.len());
    let mut morph_blobs = Vec::with_capacity(scene.models.len());
    let mut skin_blobs = Vec::with_capacity(scene.models.len());
    let skeleton = scene.models.iter().find_map(|m| m.skin.as_ref()).map(|s| s.skeleton.clone());
    for (i, model) in scene.models.iter().enumerate() {
        let has_morphs = !model.morphs.targets.is_empty();
        // The undeformed mesh; the weights and pose redo the shape on open.
        meshes.push(encode_mesh(&MeshData { vertices: model.rest_vertices().to_vec(), indices: model.mesh.indices.clone() }));
        morph_blobs.push(has_morphs.then(|| encode_morphs(&model.morphs)));
        // Only skins on the saved skeleton (there is one per model file).
        let skin = model.skin.as_ref().filter(|s| skeleton.as_ref().is_some_and(|k| Arc::ptr_eq(k, &s.skeleton)));
        skin_blobs.push(skin.map(|s| encode_skin(&s.influences)));
        let m = &model.material;
        models.push(ModelEntry {
            name: model.name.clone(),
            material_name: model.material_name.clone(),
            transform: model.transform.to_cols_array(),
            mesh: format!("meshes/{i}.bin"),
            morphs: model.morphs.targets.iter().map(|t| MorphEntry { name: t.name.clone(), weight: t.weight }).collect(),
            morph_data: has_morphs.then(|| format!("meshes/{i}.morphs")),
            skin: skin.map(|s| SkinEntry {
                joints: s.joints.clone(),
                binds: s.binds.iter().map(|m| m.to_cols_array()).collect(),
                data: format!("meshes/{i}.skin"),
            }),
            material: MaterialEntry {
                base_color: m.base_color,
                metallic: m.metallic,
                roughness: m.roughness,
                emission: m.emission,
                specular_weight: m.specular_weight,
                coat_weight: m.coat_weight,
                coat_roughness: m.coat_roughness,
                sheen_weight: m.sheen_weight,
                sheen_roughness: m.sheen_roughness,
                opacity: m.opacity,
                transparent: m.transparent,
                base_color_image: images.add(&m.base_color_image),
                roughness_image: images.add(&m.roughness_image),
                metallic_image: images.add(&m.metallic_image),
                normal_image: images.add(&m.normal_image),
                height_image: images.add(&m.height_image),
                ao_image: images.add(&m.ao_image),
                emission_image: images.add(&m.emission_image),
                emission_mask_image: images.add(&m.emission_mask_image),
            },
        });
    }

    let mut sets = Vec::with_capacity(scene.sets.len());
    for (set_index, set) in scene.sets.iter().enumerate() {
        let visible = scene.visibility.and_then(|v| v.get(set_index).copied()).unwrap_or(set.visible);
        let stack = &set.layers;
        let mut layers = Vec::with_capacity(stack.layers.len());
        for l in &stack.layers {
            layers.push(LayerEntry {
                id: l.id,
                name: l.name.clone(),
                visible: l.visible,
                opacity: l.opacity,
                blend: l.blend,
                basecolor: textures.add(gpu, &l.basecolor)?,
                roughness: textures.add(gpu, &l.roughness)?,
                metallic: textures.add(gpu, &l.metallic)?,
                normal: textures.add(gpu, &l.normal)?,
                height: textures.add(gpu, &l.height)?,
                emissive: textures.add(gpu, &l.emissive)?,
                opacity_map: textures.add(gpu, &l.opacity_map)?,
                mask: l.mask.as_ref().map(|m| textures.add(gpu, m)).transpose()?,
                mask_kind: l.mask_kind,
                mask_enabled: l.mask_enabled,
                clip: l.clip,
                uv_scale: l.uv_scale,
                uv_offset: l.uv_offset,
                uv_rotation: l.uv_rotation,
                projection: l.projection,
                projection_axis: l.projection_axis,
                triplanar_sharpness: l.triplanar_sharpness,
                projection_offset: l.projection_offset,
                projection_rotation: l.projection_rotation,
                projection_scale: l.projection_scale,
                projection_repeat: l.projection_repeat,
                has_fill: l.has_fill,
                purpose: l.purpose,
                matcap: l.matcap.as_ref().map(|m| MatcapEntry {
                    enabled: true,
                    name: m.name.clone(),
                    image: images.add(&m.image.as_ref().map(|(img, _)| img.clone())),
                    blend: m.blend,
                    strength: m.strength,
                }),
                swatch: l.swatch,
                used: l.used,
            });
        }
        let mut baked = |view: &Option<Arc<ImageView>>| -> Result<Option<u32>> {
            view.as_ref().map(|v| textures.add(gpu, v)).transpose()
        };
        sets.push(SetEntry {
            object: set.object.clone(),
            material: set.material.clone(),
            visible,
            height_scale: stack.height_scale,
            resolution: Some(stack.resolution),
            matcaps: Vec::new(),
            material_in_layers: set.material_in_layers,
            active_layer: stack.active,
            next_layer_number: stack.next_layer_number(),
            projection_frame: stack
                .projection_frame()
                .map(|(c, size)| [c[0], c[1], c[2], size]),
            baked_ao: baked(&set.baked_ao)?,
            baked_position: baked(&set.baked_position)?,
            baked_normal: baked(&set.baked_normal)?,
            baked_curvature: baked(&set.baked_curvature)?,
            layers,
        });
    }

    Ok(Snapshot {
        manifest: Manifest {
            format: FORMAT_TAG.into(),
            version: VERSION,
            name: scene.name,
            active_set: scene.active_set,
            camera: scene.camera,
            display: scene.display,
            models,
            sets,
            textures: textures.entries,
            skeleton: skeleton.as_deref().map(SkeletonEntry::of),
            pose: scene.pose,
            studio: scene.studio,
        },
        meshes,
        morphs: morph_blobs,
        skins: skin_blobs,
        images: images.images,
        textures: textures.data,
    })
}

/// Textures by identity: a texture shared by two layers is stored once.
#[derive(Default)]
struct TextureCollector {
    seen: HashMap<*const ImageView, u32>,
    entries: Vec<TextureEntry>,
    data: Vec<Option<Vec<u8>>>,
}

impl TextureCollector {
    fn add(&mut self, gpu: &GpuContext, view: &Arc<ImageView>) -> Result<u32> {
        if let Some(&index) = self.seen.get(&Arc::as_ptr(view)) {
            return Ok(index);
        }
        let format = TexFormat::of(view.image().format())?;
        let [w, h, _] = view.image().extent();
        if w != h {
            bail!("texture is {w}x{h}; project textures are square");
        }
        let bytes = read_raw(gpu, view)?;
        let empty = bytes.iter().all(|b| *b == 0);
        let index = self.entries.len() as u32;
        self.entries.push(TextureEntry {
            format,
            size: w,
            file: (!empty).then(|| format!("textures/{index}.raw")),
        });
        self.data.push((!empty).then_some(bytes));
        self.seen.insert(Arc::as_ptr(view), index);
        Ok(index)
    }
}

/// Material images by content: models split from one material (UDIM tiles,
/// multi-primitive objects) carry copies of the same maps.
#[derive(Default)]
struct ImageCollector {
    seen: HashMap<u64, String>,
    images: Vec<(String, ImageData)>,
}

impl ImageCollector {
    fn add(&mut self, image: &Option<ImageData>) -> Option<String> {
        let image = image.as_ref()?;
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        (image.width, image.height).hash(&mut hasher);
        image.rgba.hash(&mut hasher);
        let key = hasher.finish();
        Some(
            self.seen
                .entry(key)
                .or_insert_with(|| {
                    let path = format!("images/{key:016x}.png");
                    self.images.push((path.clone(), image.clone()));
                    path
                })
                .clone(),
        )
    }
}

/// Compress and write a captured project. Safe to run off the main thread.
/// Writes to a temporary file first, so a failed save never leaves a
/// half-written project in place of a good one.
pub fn write(snapshot: Snapshot, path: &Path) -> Result<()> {
    use zip::write::SimpleFileOptions;
    use zip::CompressionMethod;

    let tmp = path.with_extension(format!("{EXTENSION}.tmp"));
    let file = std::fs::File::create(&tmp)
        .with_context(|| format!("could not create {}", tmp.display()))?;
    let mut zip = zip::ZipWriter::new(std::io::BufWriter::new(file));
    let deflate = SimpleFileOptions::default()
        .compression_method(CompressionMethod::Deflated)
        .large_file(true);
    // Raw float texels compress poorly whatever the level; the fastest
    // level gets most of the win (flat and empty regions) at a fraction of
    // the time.
    let fast = deflate.compression_level(Some(1));
    // PNGs are already compressed.
    let stored = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);

    zip.start_file("project.json", deflate)?;
    zip.write_all(serde_json::to_string_pretty(&snapshot.manifest)?.as_bytes())?;

    for (((entry, bytes), morphs), skin) in
        snapshot.manifest.models.iter().zip(&snapshot.meshes).zip(&snapshot.morphs).zip(&snapshot.skins)
    {
        zip.start_file(entry.mesh.as_str(), fast)?;
        zip.write_all(bytes)?;
        if let (Some(path), Some(morphs)) = (&entry.morph_data, morphs) {
            zip.start_file(path.as_str(), fast)?;
            zip.write_all(morphs)?;
        }
        if let (Some(entry), Some(skin)) = (&entry.skin, skin) {
            zip.start_file(entry.data.as_str(), fast)?;
            zip.write_all(skin)?;
        }
    }
    for (name, image) in &snapshot.images {
        let rgba = image::RgbaImage::from_raw(image.width, image.height, image.rgba.to_vec())
            .context("material image has the wrong size")?;
        let mut png = Vec::new();
        rgba.write_to(&mut Cursor::new(&mut png), image::ImageFormat::Png)?;
        zip.start_file(name.as_str(), stored)?;
        zip.write_all(&png)?;
    }
    for (entry, data) in snapshot.manifest.textures.iter().zip(&snapshot.textures) {
        if let (Some(file), Some(bytes)) = (&entry.file, data) {
            zip.start_file(file.as_str(), fast)?;
            zip.write_all(bytes)?;
        }
    }
    let mut out = zip.finish()?;
    out.flush()?;
    drop(out);
    std::fs::rename(&tmp, path)
        .with_context(|| format!("could not move the saved project to {}", path.display()))?;
    Ok(())
}

// -- open --------------------------------------------------------------------

/// A project file read into host memory. Built on a worker thread.
pub struct LoadedProject {
    manifest: Manifest,
    meshes: Vec<MeshData>,
    morphs: Vec<crate::scene::Morphs>,
    skins: Vec<Option<Vec<crate::skeleton::Influence>>>,
    images: HashMap<String, ImageData>,
    textures: Vec<Option<Vec<u8>>>,
}

/// Read and decompress a project file. Safe to run off the main thread.
pub fn read(path: &Path) -> Result<LoadedProject> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("could not open {}", path.display()))?;
    let mut zip = zip::ZipArchive::new(std::io::BufReader::new(file))
        .context("not a ZestSurface project (not a zip archive)")?;
    let read_entry = |zip: &mut zip::ZipArchive<_>, name: &str| -> Result<Vec<u8>> {
        let mut entry = zip
            .by_name(name)
            .with_context(|| format!("project is missing {name}"))?;
        let mut bytes = Vec::with_capacity(entry.size() as usize);
        entry.read_to_end(&mut bytes)?;
        Ok(bytes)
    };

    let manifest: Manifest = serde_json::from_slice(&read_entry(&mut zip, "project.json")?)
        .context("project.json is damaged")?;
    if manifest.format != FORMAT_TAG && manifest.format != LEGACY_FORMAT_TAG {
        bail!("not a ZestSurface project");
    }
    if manifest.version > VERSION {
        bail!(
            "this project was saved by a newer ZestSurface (format {}, this build reads {VERSION})",
            manifest.version
        );
    }

    let mut meshes = Vec::with_capacity(manifest.models.len());
    let mut morphs = Vec::with_capacity(manifest.models.len());
    let mut skins = Vec::with_capacity(manifest.models.len());
    let mut images = HashMap::new();
    for model in &manifest.models {
        let mesh = decode_mesh(&read_entry(&mut zip, &model.mesh)?).with_context(|| format!("{} is damaged", model.mesh))?;
        skins.push(match &model.skin {
            Some(skin) => Some(
                decode_skin(&read_entry(&mut zip, &skin.data)?, mesh.vertices.len(), skin.joints.len())
                    .with_context(|| format!("{} is damaged", skin.data))?,
            ),
            None => None,
        });
        morphs.push(match &model.morph_data {
            Some(path) => {
                let offsets = decode_morphs(&read_entry(&mut zip, path)?, model.morphs.len(), mesh.vertices.len())
                    .with_context(|| format!("{path} is damaged"))?;
                crate::scene::Morphs {
                    base: Arc::new(mesh.vertices.clone()),
                    targets: model
                        .morphs
                        .iter()
                        .zip(offsets)
                        .map(|(m, offsets)| crate::scene::Morph { name: m.name.clone(), weight: m.weight, offsets: Arc::new(offsets) })
                        .collect(),
                }
            }
            None => crate::scene::Morphs::default(),
        });
        meshes.push(mesh);
        let m = &model.material;
        let matcaps = manifest
            .sets
            .iter()
            .flat_map(|s| s.matcaps.iter().chain(s.layers.iter().filter_map(|l| l.matcap.as_ref())))
            .map(|m| &m.image);
        for name in [
            &m.base_color_image,
            &m.roughness_image,
            &m.metallic_image,
            &m.normal_image,
            &m.height_image,
            &m.ao_image,
            &m.emission_image,
            &m.emission_mask_image,
        ]
        .into_iter()
        .chain(matcaps)
        .flatten()
        {
            if images.contains_key(name) {
                continue;
            }
            let decoded = image::load_from_memory(&read_entry(&mut zip, name)?)
                .with_context(|| format!("{name} is damaged"))?
                .to_rgba8();
            let (width, height) = decoded.dimensions();
            images.insert(name.clone(), ImageData { width, height, rgba: decoded.into_raw().into() });
        }
    }

    let mut textures = Vec::with_capacity(manifest.textures.len());
    for entry in &manifest.textures {
        let data = match &entry.file {
            Some(file) => {
                let bytes = read_entry(&mut zip, file)?;
                let expected = (entry.size as usize).pow(2) * entry.format.texel_bytes();
                if bytes.len() != expected {
                    bail!("{file} is damaged ({} bytes, expected {expected})", bytes.len());
                }
                Some(bytes)
            }
            None => None,
        };
        textures.push(data);
    }

    Ok(LoadedProject { manifest, meshes, morphs, skins, images, textures })
}

/// A project rebuilt on the GPU, for the app to adopt.
pub struct Restored {
    pub name: String,
    pub models: Vec<ModelData>,
    pub sets: Vec<TextureSet>,
    pub active_set: usize,
    pub camera: CameraState,
    pub display: DisplayState,
    pub pose: Option<serde_json::Value>,
    pub studio: Option<serde_json::Value>,
}

impl LoadedProject {
    /// The resolution sets without their own were saved at (older projects
    /// had one size for all): the first full-size layer texture.
    pub fn resolution(&self) -> Option<u32> {
        self.manifest
            .sets
            .iter()
            .flat_map(|s| &s.layers)
            .flat_map(|l| [l.basecolor, l.roughness, l.metallic])
            .filter_map(|t| self.manifest.textures.get(t as usize))
            .map(|t| t.size)
            .find(|&size| size > 1)
    }
}

/// Upload a loaded project. Main thread.
pub fn restore(gpu: &GpuContext, project: LoadedProject, resolution: u32) -> Result<Restored> {
    let LoadedProject { manifest, meshes, morphs, skins, images, textures: data } = project;
    if manifest.models.is_empty() {
        bail!("project has no models");
    }
    let skeleton = manifest.skeleton.as_ref().map(|s| s.skeleton()).transpose()?.map(Arc::new);
    let image = |name: &Option<String>| name.as_ref().and_then(|n| images.get(n).cloned());

    let models: Vec<ModelData> = manifest
        .models
        .iter()
        .zip(meshes)
        .zip(morphs)
        .zip(skins)
        .map(|(((entry, mesh), morphs), influences)| {
            let m = &entry.material;
            let skin = match (&skeleton, &entry.skin, influences) {
                (Some(skeleton), Some(skin), Some(influences)) => Some(crate::skeleton::Skin {
                    skeleton: skeleton.clone(),
                    joints: skin.joints.iter().map(|&j| j.min(skeleton.joints.len().saturating_sub(1) as u32)).collect(),
                    binds: skin.binds.iter().map(glam::Mat4::from_cols_array).collect(),
                    influences: Arc::new(influences),
                    rest: Arc::new(mesh.vertices.clone()),
                    pose: None,
                }),
                _ => None,
            };
            let mut model = ModelData {
                mesh,
                material: MaterialData {
                    base_color: m.base_color,
                    metallic: m.metallic,
                    roughness: m.roughness,
                    base_color_image: image(&m.base_color_image),
                    roughness_image: image(&m.roughness_image),
                    metallic_image: image(&m.metallic_image),
                    normal_image: image(&m.normal_image),
                    height_image: image(&m.height_image),
                    ao_image: image(&m.ao_image),
                    emission_image: image(&m.emission_image),
                    emission_mask_image: image(&m.emission_mask_image),
                    emission: m.emission,
                    specular_weight: m.specular_weight,
                    coat_weight: m.coat_weight,
                    coat_roughness: m.coat_roughness,
                    sheen_weight: m.sheen_weight,
                    sheen_roughness: m.sheen_roughness,
                    opacity: m.opacity,
                    transparent: m.transparent,
                },
                transform: glam::Mat4::from_cols_array(&entry.transform),
                name: entry.name.clone(),
                material_name: entry.material_name.clone(),
                morphs,
                skin,
            };
            model.deform();
            model
        })
        .collect();

    // Texture sets follow the models' set order — the renderer indexes sets
    // that way — matched to the saved sets by object and material.
    let mut uploaded: Vec<Option<Arc<ImageView>>> = vec![None; manifest.textures.len()];
    let mut texture = |index: u32| -> Result<Arc<ImageView>> {
        let i = index as usize;
        let Some(entry) = manifest.textures.get(i) else {
            bail!("layer refers to texture {index}, which the project does not have");
        };
        if let Some(view) = &uploaded[i] {
            return Ok(view.clone());
        }
        let view = upload_raw(gpu, entry.format.vulkan(), entry.size, data[i].as_deref())?;
        uploaded[i] = Some(view.clone());
        Ok(view)
    };

    let mut sets = Vec::new();
    for key in crate::scene::set_keys(&models) {
        // Lazy: blank channels stay on 1x1 placeholders, and a set that is
        // never opened never allocates paint textures (see `LayerStack::new_lazy`).
        let saved = manifest.sets.iter().find(|s| s.object == key.object && s.material == key.material);
        let resolution = saved.and_then(|s| s.resolution).unwrap_or(resolution);
        let mut set = TextureSet::new_lazy(key.object.clone(), key.material.clone(), gpu.clone(), resolution);
        let Some(saved) = saved else {
            sets.push(set);
            continue;
        };
        set.visible = saved.visible;
        set.material_in_layers = saved.material_in_layers;
        set.seeded = saved.material_in_layers;
        let matcap = |entry: &MatcapEntry| {
            let mut m = crate::matcap::Matcap { name: entry.name.clone(), blend: entry.blend, strength: entry.strength, image: None };
            if let Some(img) = image(&entry.image) {
                m.set_image(gpu, entry.name.clone(), img);
            }
            m
        };
        // An all-zero channel is a blank one: back onto the placeholder.
        macro_rules! channel {
            ($index:expr) => {{
                let index: u32 = $index;
                let blank = manifest
                    .textures
                    .get(index as usize)
                    .filter(|e| e.file.is_none())
                    .map(|e| e.format.vulkan())
                    .filter(|f| matches!(f, Format::R16G16B16A16_SFLOAT | Format::R16G16_SFLOAT));
                match blank {
                    Some(format) => set.layers.placeholder(format),
                    None => texture(index)?,
                }
            }};
        }
        let mut layers = Vec::with_capacity(saved.layers.len());
        for l in &saved.layers {
            let mask = l.mask.map(&mut texture).transpose()?;
            layers.push(Layer {
                id: l.id,
                name: l.name.clone(),
                visible: l.visible,
                opacity: l.opacity,
                blend: l.blend,
                basecolor: channel!(l.basecolor),
                roughness: channel!(l.roughness),
                metallic: channel!(l.metallic),
                normal: channel!(l.normal),
                height: channel!(l.height),
                emissive: channel!(l.emissive),
                opacity_map: channel!(l.opacity_map),
                mask,
                mask_kind: l.mask_kind,
                mask_enabled: l.mask_enabled,
                clip: l.clip,
                uv_scale: l.uv_scale,
                uv_offset: l.uv_offset,
                uv_rotation: l.uv_rotation,
                projection: l.projection,
                projection_axis: l.projection_axis,
                triplanar_sharpness: l.triplanar_sharpness,
                projection_offset: l.projection_offset,
                projection_rotation: l.projection_rotation,
                projection_scale: l.projection_scale,
                projection_repeat: l.projection_repeat,
                has_fill: l.has_fill,
                purpose: l.purpose,
                matcap: l.matcap.as_ref().map(matcap),
                swatch: l.swatch,
                used: l.used,
            });
        }
        // Format 2 kept two matcaps per set, shown where the Matcap /
        // Matcap 2 mask layers allowed: each becomes its mask layer, now a
        // matcap layer, or (with no mask layer) a new one that is all white.
        let mut unmasked = Vec::new();
        for (entry, kind) in saved.matcaps.iter().zip([ShaderMask::Matcap, ShaderMask::Matcap2]) {
            let look = Some(matcap(entry)).filter(|m| entry.enabled && m.image.is_some());
            match layers.iter_mut().rev().find(|l| l.purpose == Some(kind) && l.matcap.is_none()) {
                Some(layer) => {
                    layer.purpose = Some(ShaderMask::Matcap);
                    layer.matcap = Some(look.unwrap_or_default());
                }
                None => unmasked.extend(look),
            }
        }
        for layer in layers.iter_mut().filter(|l| l.purpose == Some(ShaderMask::Matcap2)) {
            layer.purpose = Some(ShaderMask::Matcap);
        }
        for layer in layers.iter_mut().filter(|l| l.is_matcap() && l.matcap.is_none()) {
            layer.matcap = Some(Default::default());
        }
        for view in layers.iter().flat_map(|l| {
            [&l.basecolor, &l.roughness, &l.metallic, &l.normal, &l.height, &l.emissive, &l.opacity_map]
                .into_iter()
                .chain(l.mask.as_ref())
        }) {
            let [w, _, _] = view.image().extent();
            if w != resolution && !set.layers.is_placeholder(view) {
                bail!("project layers are {w}px; this build paints at {resolution}px");
            }
        }
        set.layers.restore_saved(layers, saved.active_layer, saved.height_scale, saved.next_layer_number);
        for look in unmasked {
            let active = set.layers.active;
            set.layers.add_shader_mask_layer(ShaderMask::Matcap);
            set.layers.fill_active(Some([1.0; 3]), None, None);
            let top = set.layers.active;
            let layer = &mut set.layers.layers[top];
            layer.name = format!("Matcap · {}", look.name);
            layer.matcap = Some(look);
            set.layers.active = active;
        }

        set.baked_ao = saved.baked_ao.map(&mut texture).transpose()?;
        set.baked_position = saved.baked_position.map(&mut texture).transpose()?;
        set.baked_normal = saved.baked_normal.map(&mut texture).transpose()?;
        set.baked_curvature = saved.baked_curvature.map(&mut texture).transpose()?;
        if let (Some(frame), Some(position), Some(normal)) =
            (saved.projection_frame, set.baked_position.clone(), set.baked_normal.clone())
        {
            set.layers
                .set_projection_source(position, normal, [frame[0], frame[1], frame[2]], frame[3]);
        }
        sets.push(set);
    }

    Ok(Restored {
        name: manifest.name,
        active_set: manifest.active_set.min(sets.len().saturating_sub(1)),
        models,
        sets,
        camera: manifest.camera,
        display: manifest.display,
        pose: manifest.pose,
        studio: manifest.studio,
    })
}

// -- GPU transfer ------------------------------------------------------------

/// Copy a texture's texels to host memory, exactly as stored.
fn read_raw(gpu: &GpuContext, view: &Arc<ImageView>) -> Result<Vec<u8>> {
    let [w, h, _] = view.image().extent();
    let len = w as u64 * h as u64 * view.image().format().block_size();
    let buffer: Subbuffer<[u8]> = Buffer::new_slice(
        gpu.memory_allocator.clone(),
        BufferCreateInfo { usage: BufferUsage::TRANSFER_DST, ..Default::default() },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        len,
    )?;
    let mut builder = AutoCommandBufferBuilder::primary(
        gpu.command_buffer_allocator.clone(),
        gpu.queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )?;
    builder.copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
        view.image().clone(),
        buffer.clone(),
    ))?;
    builder
        .build()?
        .execute(gpu.queue.clone())?
        .then_signal_fence_and_flush()?
        .wait(None)?;
    let bytes = buffer.read()?.to_vec();
    Ok(bytes)
}

/// Create a texture and fill it with saved texels (None = cleared to zero).
/// Usage covers every role a restored texture can play: painted into
/// (colour attachment), sampled, snapshotted for undo, and — for the
/// four-channel ones — written by compute (smart masks, bakes).
fn upload_raw(gpu: &GpuContext, format: Format, size: u32, bytes: Option<&[u8]>) -> Result<Arc<ImageView>> {
    let mut usage = ImageUsage::COLOR_ATTACHMENT
        | ImageUsage::SAMPLED
        | ImageUsage::TRANSFER_SRC
        | ImageUsage::TRANSFER_DST;
    if format != Format::R16G16_SFLOAT {
        usage |= ImageUsage::STORAGE;
    }
    let image = crate::vram::image(
        gpu.memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format,
            extent: [size, size, 1],
            usage,
            ..Default::default()
        },
        crate::vram::Kind::Layer,
    )?;
    let mut builder = AutoCommandBufferBuilder::primary(
        gpu.command_buffer_allocator.clone(),
        gpu.queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )?;
    match bytes {
        Some(bytes) => {
            let buffer = Buffer::from_iter(
                gpu.memory_allocator.clone(),
                BufferCreateInfo { usage: BufferUsage::TRANSFER_SRC, ..Default::default() },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_HOST
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                bytes.iter().copied(),
            )?;
            builder.copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(buffer, image.clone()))?;
        }
        None => {
            builder.clear_color_image(ClearColorImageInfo {
                clear_value: ClearColorValue::Float([0.0; 4]),
                ..ClearColorImageInfo::image(image.clone())
            })?;
        }
    }
    builder
        .build()?
        .execute(gpu.queue.clone())?
        .then_signal_fence_and_flush()?
        .wait(None)?;
    Ok(ImageView::new_default(image)?)
}

// -- meshes ------------------------------------------------------------------

/// Little-endian: magic, blendshape count, then per blendshape its offset
/// count and per offset the vertex index, position offset xyz and normal
/// offset xyz.
/// Skin weights: magic, vertex count, then per vertex four u16 slots and
/// four f32 weights, little-endian.
fn encode_skin(influences: &[crate::skeleton::Influence]) -> Vec<u8> {
    let mut out = Vec::with_capacity(12 + influences.len() * 24);
    out.extend_from_slice(SKIN_MAGIC);
    out.extend_from_slice(&(influences.len() as u32).to_le_bytes());
    for inf in influences {
        for s in inf.slots {
            out.extend_from_slice(&s.to_le_bytes());
        }
        for w in inf.weights {
            out.extend_from_slice(&w.to_le_bytes());
        }
    }
    out
}

fn decode_skin(bytes: &[u8], vertex_count: usize, slots: usize) -> Result<Vec<crate::skeleton::Influence>> {
    if bytes.len() < 12 || &bytes[..8] != SKIN_MAGIC {
        bail!("not skin weights");
    }
    let count = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    if count != vertex_count || bytes.len() != 12 + count * 24 {
        bail!("skin weights for {count} vertices, the mesh has {vertex_count}");
    }
    let mut out = Vec::with_capacity(count);
    for chunk in bytes[12..].chunks_exact(24) {
        let mut inf = crate::skeleton::Influence::default();
        for k in 0..4 {
            inf.slots[k] = u16::from_le_bytes(chunk[k * 2..k * 2 + 2].try_into().unwrap());
            inf.weights[k] = f32::from_le_bytes(chunk[8 + k * 4..12 + k * 4].try_into().unwrap());
            if inf.weights[k] > 0.0 && inf.slots[k] as usize >= slots {
                bail!("skin weight refers to bone slot {}, the part has {slots}", inf.slots[k]);
            }
        }
        out.push(inf);
    }
    Ok(out)
}

fn encode_morphs(morphs: &crate::scene::Morphs) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(MORPH_MAGIC);
    out.extend_from_slice(&(morphs.targets.len() as u32).to_le_bytes());
    for t in &morphs.targets {
        out.extend_from_slice(&(t.offsets.len() as u32).to_le_bytes());
        for (v, dp, dn) in t.offsets.iter() {
            out.extend_from_slice(&v.to_le_bytes());
            for f in dp.iter().chain(dn) {
                out.extend_from_slice(&f.to_le_bytes());
            }
        }
    }
    out
}

/// The offsets of `count` blendshapes over a mesh of `vertex_count`.
fn decode_morphs(bytes: &[u8], count: usize, vertex_count: usize) -> Result<Vec<Vec<(u32, [f32; 3], [f32; 3])>>> {
    if bytes.len() < 12 || &bytes[..8] != MORPH_MAGIC {
        bail!("not blendshape data");
    }
    let word = |at: usize| -> Result<u32> {
        Ok(u32::from_le_bytes(bytes.get(at..at + 4).context("truncated")?.try_into().unwrap()))
    };
    if word(8)? as usize != count {
        bail!("blendshape count does not match the project");
    }
    let mut at = 12;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let n = word(at)? as usize;
        at += 4;
        let mut offsets = Vec::with_capacity(n);
        for _ in 0..n {
            let v = word(at)?;
            if v as usize >= vertex_count {
                bail!("vertex out of range");
            }
            let f = |k: usize| -> Result<f32> { Ok(f32::from_bits(word(at + 4 + k * 4)?)) };
            offsets.push((v, [f(0)?, f(1)?, f(2)?], [f(3)?, f(4)?, f(5)?]));
            at += 28;
        }
        out.push(offsets);
    }
    if at != bytes.len() {
        bail!("trailing data");
    }
    Ok(out)
}

/// Little-endian: magic, vertex count, index count, then per vertex
/// position xyz, normal xyz, uv; then the indices.
fn encode_mesh(mesh: &MeshData) -> Vec<u8> {
    let mut out = Vec::with_capacity(16 + mesh.vertices.len() * 32 + mesh.indices.len() * 4);
    out.extend_from_slice(MESH_MAGIC);
    out.extend_from_slice(&(mesh.vertices.len() as u32).to_le_bytes());
    out.extend_from_slice(&(mesh.indices.len() as u32).to_le_bytes());
    for v in &mesh.vertices {
        for f in v.position.iter().chain(&v.normal).chain(&v.uv) {
            out.extend_from_slice(&f.to_le_bytes());
        }
    }
    for i in &mesh.indices {
        out.extend_from_slice(&i.to_le_bytes());
    }
    out
}

fn decode_mesh(bytes: &[u8]) -> Result<MeshData> {
    if bytes.len() < 16 || &bytes[..8] != MESH_MAGIC {
        bail!("not a mesh");
    }
    let word = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
    let (vertex_count, index_count) = (word(8) as usize, word(12) as usize);
    if bytes.len() != 16 + vertex_count * 32 + index_count * 4 {
        bail!("truncated");
    }
    let float = |at: usize| f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
    let vertices = (0..vertex_count)
        .map(|v| {
            let at = 16 + v * 32;
            Vertex3D {
                position: [float(at), float(at + 4), float(at + 8)],
                normal: [float(at + 12), float(at + 16), float(at + 20)],
                uv: [float(at + 24), float(at + 28)],
            }
        })
        .collect();
    let base = 16 + vertex_count * 32;
    let indices: Vec<u32> = (0..index_count).map(|i| word(base + i * 4)).collect();
    if indices.iter().any(|&i| i as usize >= vertex_count) {
        bail!("index out of range");
    }
    Ok(MeshData { vertices, indices })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mesh_round_trips() {
        let mesh = crate::mesh::uv_sphere(1.0, 4, 8);
        let back = decode_mesh(&encode_mesh(&mesh)).unwrap();
        assert_eq!(back.indices, mesh.indices);
        assert_eq!(back.vertices.len(), mesh.vertices.len());
        for (a, b) in back.vertices.iter().zip(&mesh.vertices) {
            assert_eq!(a.position, b.position);
            assert_eq!(a.normal, b.normal);
            assert_eq!(a.uv, b.uv);
        }
    }

    #[test]
    fn damaged_meshes_are_rejected() {
        let bytes = encode_mesh(&crate::mesh::uv_sphere(1.0, 4, 8));
        assert!(decode_mesh(&bytes[..bytes.len() - 1]).is_err());
        assert!(decode_mesh(b"nonsense").is_err());
    }

    /// Paint a layer, save, open: every texel, every setting and the mesh
    /// come back exactly, and empty channels are not stored.
    #[test]
    fn project_round_trips_through_a_file() {
        let Some(gpu) = crate::environment::test_gpu::context() else {
            eprintln!("no Vulkan device; skipping");
            return;
        };
        let res = 64;
        let mut set = TextureSet::new("Body".into(), "Skin".into(), gpu.clone(), res);
        let stack = &mut set.layers;
        stack.fill_active(Some([0.25, 0.5, 0.75]), Some(0.3), None);
        stack.add_mask(0, 0.5);
        stack.layers[0].name = "Base coat".into();
        stack.layers[0].opacity = 0.6;
        stack.layers[0].blend = BlendMode::Multiply;
        stack.layers[0].projection = Projection::Triplanar;
        stack.layers[0].uv_rotation = 1.25;
        stack.add_shader_mask_layer(ShaderMask::Matcap);
        let top = stack.active;
        let chrome = crate::matcap::builtin("Chrome").unwrap();
        stack.layers[top].set_matcap(&gpu, "Chrome".into(), chrome, Some(crate::matcap::MatcapBlend::Multiply));
        stack.layers[top].matcap.as_mut().unwrap().strength = 0.7;
        stack.height_scale = 7.0;

        let sphere = crate::mesh::uv_sphere(1.0, 4, 8);
        let mut models = vec![ModelData {
            mesh: sphere.clone(),
            material: MaterialData { roughness: 0.42, ..MaterialData::default() },
            transform: glam::Mat4::from_translation(glam::vec3(1.0, 2.0, 3.0)),
            name: "Body".into(),
            material_name: "Skin".into(),
            morphs: crate::scene::Morphs {
                base: Arc::new(sphere.vertices.clone()),
                targets: vec![
                    crate::scene::Morph { name: "vrc.v_aa".into(), weight: 0.75, offsets: Arc::new(vec![(3, [0.0, 0.5, 0.0], [0.0; 3])]) },
                    crate::scene::Morph { name: "Blink".into(), weight: 0.0, offsets: Arc::new(vec![(4, [0.1, 0.0, 0.0], [0.0; 3])]) },
                ],
            },
            skin: None,
        }];
        // Rigged: two joints, every vertex on the second, posed turned 90°.
        let skeleton = crate::skeleton::Skeleton {
            joints: vec![
                crate::skeleton::Joint { name: "Root".into(), parent: None, rest: Default::default(), base: glam::Mat4::from_scale(glam::Vec3::splat(0.01)) },
                crate::skeleton::Joint {
                    name: "Spine".into(),
                    parent: Some(0),
                    rest: crate::skeleton::Trs { t: glam::vec3(0.0, 100.0, 0.0), ..Default::default() },
                    base: glam::Mat4::IDENTITY,
                },
            ],
        };
        let rest_world = skeleton.rest_world();
        let mut posed = skeleton.rest_locals();
        posed[1].r = glam::Quat::from_rotation_y(1.0);
        models[0].skin = Some(crate::skeleton::Skin {
            joints: vec![1],
            binds: vec![(models[0].transform.inverse() * rest_world[1]).inverse()],
            influences: Arc::new(vec![crate::skeleton::Influence { slots: [0; 4], weights: [1.0, 0.0, 0.0, 0.0] }; sphere.vertices.len()]),
            rest: Arc::new(sphere.vertices.clone()),
            pose: Some(Arc::new(skeleton.world(&posed))),
            skeleton: Arc::new(skeleton),
        });
        models[0].deform();
        let sets = vec![set];
        let camera = CameraState { target: [0.0, 1.0, 0.0], yaw: 0.3, pitch: 0.2, distance: 5.0 };
        let display = DisplayState {
            exposure: 1.5,
            env_intensity: 0.8,
            key_light: 2.0,
            key_azimuth: 10.0,
            key_elevation: 20.0,
            key_follow_camera: true,
            ao_strength: 0.7,
            env_path: None,
        };
        let snapshot = capture(
            &gpu,
            SceneRefs {
                visibility: None,
                name: "test".into(),
                models: &models,
                sets: &sets,
                active_set: 0,
                camera,
                display: display.clone(),
                pose: None,
                studio: None,
            },
        )
        .unwrap();
        let stored = snapshot.textures.iter().filter(|t| t.is_some()).count();
        // Base colour, roughness and the mask of layer 0; nothing on layer 1.
        assert_eq!(stored, 3, "only non-empty textures are stored");

        let dir = std::env::temp_dir().join(format!("zestsurface-project-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.zestsurface");
        write(snapshot, &path).unwrap();
        let loaded = read(&path).unwrap();
        assert_eq!(loaded.resolution(), Some(res));
        let restored = restore(&gpu, loaded, res).unwrap();
        let old_format = read(&path).unwrap();
        std::fs::remove_dir_all(&dir).ok();

        assert_eq!(restored.camera, camera);
        assert_eq!(restored.display, display);
        assert_eq!(restored.models.len(), 1);
        assert_eq!(restored.models[0].material.roughness, 0.42);
        assert_eq!(restored.models[0].transform, models[0].transform);
        assert_eq!(restored.models[0].mesh.indices, models[0].mesh.indices);
        // Blendshapes come back with their weights, the mesh deformed by them.
        let morphs = &restored.models[0].morphs;
        let shapes: Vec<(&str, f32)> = morphs.targets.iter().map(|t| (t.name.as_str(), t.weight)).collect();
        assert_eq!(shapes, [("vrc.v_aa", 0.75), ("Blink", 0.0)]);
        assert_eq!(morphs.base[3].position, sphere.vertices[3].position, "the undeformed mesh is stored, not the posed one");
        // The skin comes back unposed (the app reapplies the saved pose).
        let skin = restored.models[0].skin.as_ref().expect("the skin is kept");
        assert_eq!(skin.skeleton.joints.len(), 2);
        assert_eq!(skin.skeleton.joints[1].name, "Spine");
        assert_eq!(skin.skeleton.joints[0].base, glam::Mat4::from_scale(glam::Vec3::splat(0.01)));
        assert_eq!(skin.joints, [1]);
        assert_eq!(skin.binds, models[0].skin.as_ref().unwrap().binds);
        assert_eq!(skin.influences.len(), sphere.vertices.len());
        assert!(skin.pose.is_none());
        let mut unposed = models[0].clone();
        unposed.skin.as_mut().unwrap().pose = None;
        unposed.deform();
        assert_eq!(restored.models[0].mesh.vertices[3].position, unposed.mesh.vertices[3].position);
        assert_ne!(restored.models[0].mesh.vertices[3].position, sphere.vertices[3].position);
        assert_ne!(models[0].mesh.vertices[3].position, unposed.mesh.vertices[3].position, "the pose did move it");

        let (a, b) = (&sets[0].layers, &restored.sets[0].layers);
        assert_eq!(b.layers.len(), 2);
        assert_eq!(b.active, a.active);
        assert_eq!(b.height_scale, 7.0);
        assert_eq!(b.next_layer_number(), a.next_layer_number());
        let (la, lb) = (&a.layers[0], &b.layers[0]);
        assert_eq!(lb.name, "Base coat");
        assert_eq!(lb.opacity, 0.6);
        assert_eq!(lb.blend, BlendMode::Multiply);
        assert_eq!(lb.projection, Projection::Triplanar);
        assert_eq!(lb.uv_rotation, 1.25);
        assert_eq!(lb.used, la.used);
        assert_eq!(b.layers[1].purpose, Some(ShaderMask::Matcap));
        assert_eq!(b.layers[1].name, "Matcap · Chrome");
        let m = b.layers[1].matcap.as_ref().unwrap();
        assert_eq!((m.name.as_str(), m.blend, m.strength), ("Chrome", crate::matcap::MatcapBlend::Multiply, 0.7));
        assert_eq!(m.image.as_ref().map(|(img, _)| (img.width, img.height)), Some((256, 256)));
        for (x, y) in [
            (&la.basecolor, &lb.basecolor),
            (&la.roughness, &lb.roughness),
            (la.mask.as_ref().unwrap(), lb.mask.as_ref().unwrap()),
        ] {
            assert_eq!(x.image().format(), y.image().format());
            assert_eq!(read_raw(&gpu, x).unwrap(), read_raw(&gpu, y).unwrap());
        }
        // Never-painted channels come back as the lazy placeholder, and get
        // real (blank) textures once the set is worked on.
        assert!(b.is_placeholder(&lb.metallic));
        let id = lb.id;
        let mut b = restored.sets.into_iter().next().unwrap().layers;
        assert!(b.ensure_allocated().is_empty(), "blank channels stay placeholders until written");
        b.active = 0;
        b.begin_stroke(&[crate::layers::PaintChannel::Metallic]);
        let upgrades = b.take_upgrades();
        assert!(upgrades.iter().any(|(layer, channel, _, _)| *layer == id && *channel == 2));
        let metallic = &b.layers[0].metallic;
        assert_eq!(metallic.image().extent()[0], res);
        assert!(read_raw(&gpu, metallic).unwrap().iter().all(|v| *v == 0));

        // The same file as format 2 wrote matcaps: two slots on the set, the
        // second limited by a Matcap 2 mask layer, the first by nothing.
        let mut old = old_format;
        let saved = &mut old.manifest.sets[0];
        let mut slot2 = saved.layers[1].matcap.take().unwrap();
        saved.layers[1].purpose = Some(ShaderMask::Matcap2);
        let slot1 = MatcapEntry { enabled: true, name: "Gloss".into(), image: slot2.image.clone(), blend: crate::matcap::MatcapBlend::Add, strength: 0.5 };
        slot2.enabled = true;
        saved.matcaps = vec![slot1, slot2];
        let mut migrated = restore(&gpu, old, res).unwrap().sets.into_iter().next().unwrap().layers;
        assert_eq!(migrated.layers.len(), 3, "the unmasked slot gets a layer of its own");
        let masked = &migrated.layers[1];
        assert!(masked.is_matcap());
        assert_eq!(masked.matcap.as_ref().unwrap().name, "Chrome");
        let added = &migrated.layers[2];
        assert!(added.is_matcap());
        assert_eq!(added.name, "Matcap · Gloss");
        assert_eq!(added.matcap.as_ref().unwrap().strength, 0.5);
        assert_eq!(migrated.active, a.active, "migration leaves the selection alone");
        migrated.composite_now();
        let id = migrated.layers[2].id;
        let mask = read_raw(&gpu, migrated.matcap_mask(id).unwrap()).unwrap();
        assert!(mask.chunks(8).all(|t| half::f16::from_le_bytes([t[0], t[1]]).to_f32() > 0.99), "it shows everywhere");
    }

    /// How well a real project's meshes match its model file's blendshapes
    /// (read-only):
    ///   ZESTSURFACE_TEST_PROJECT=a.zestsurface ZESTSURFACE_TEST_FBX=a.fbx \
    ///   cargo test --release shapes_match_real_project -- --ignored --nocapture
    /// Bones for a project saved before ZestSurface kept them, from its model
    /// file: every rigged part matches, every vertex gets weights.
    ///   ZESTSURFACE_TEST_PROJECT=… ZESTSURFACE_TEST_FBX=…
    ///   cargo test --release skins_match_real_project -- --ignored --nocapture
    #[test]
    #[ignore = "needs a real project and model file"]
    fn skins_match_real_project() {
        let (Ok(project), Ok(fbx)) = (std::env::var("ZESTSURFACE_TEST_PROJECT"), std::env::var("ZESTSURFACE_TEST_FBX")) else {
            return;
        };
        let gpu = crate::environment::test_gpu::context().unwrap();
        let loaded = read(std::path::Path::new(&project)).unwrap();
        let res = loaded.resolution().unwrap_or(2048);
        let mut models = restore(&gpu, loaded, res).unwrap().models;
        let source = crate::scene::split_udim_tiles(crate::fbx_loader::load_with_progress(std::path::Path::new(&fbx), &|_, _, _| {}).unwrap());
        let attached = crate::scene::attach_skins(&mut models, &source);
        let rigged = source.iter().filter(|m| m.skin.is_some()).count();
        eprintln!("{} of {} project parts got bones ({rigged} rigged parts in the file)", attached.len(), models.len());
        for m in models.iter().filter(|m| m.skin.is_none()) {
            eprintln!("  no bones: {} · {}", m.name, m.material_name);
        }
        for &i in &attached {
            let skin = models[i].skin.as_ref().unwrap();
            let unweighted = skin.influences.iter().filter(|w| w.weights[0] == 0.0).count();
            if unweighted > 0 {
                eprintln!("  {} · {}: {unweighted} of {} vertices unweighted", models[i].name, models[i].material_name, skin.influences.len());
            }
        }
        assert!(!attached.is_empty());
    }

    #[test]
    #[ignore = "needs a real project and model file"]
    fn shapes_match_real_project() {
        let (Ok(project), Ok(fbx)) = (std::env::var("ZESTSURFACE_TEST_PROJECT"), std::env::var("ZESTSURFACE_TEST_FBX")) else {
            return;
        };
        let gpu = crate::environment::test_gpu::context().unwrap();
        let loaded = read(std::path::Path::new(&project)).unwrap();
        let res = loaded.resolution().unwrap_or(2048);
        let mut models = restore(&gpu, loaded, res).unwrap().models;
        let source = crate::scene::split_udim_tiles(crate::fbx_loader::load_with_progress(std::path::Path::new(&fbx), &|_, _, _| {}).unwrap());
        let attached = crate::scene::attach_morphs(&mut models, &source);
        for i in attached {
            let m = &models[i];
            let src = source.iter().find(|s| s.name == m.name && s.material_name == m.material_name).unwrap();
            let moved = |targets: &[crate::scene::Morph]| -> std::collections::HashSet<u32> {
                targets.iter().flat_map(|t| t.offsets.iter().map(|o| o.0)).collect()
            };
            eprintln!(
                "{} · {}: {} shapes, {} of {} verts move (source: {} of {})",
                m.name,
                m.material_name,
                m.morphs.targets.len(),
                moved(&m.morphs.targets).len(),
                m.mesh.vertices.len(),
                moved(&src.morphs.targets).len(),
                src.mesh.vertices.len()
            );
            // Target vertices that match no undeformed source vertex: which
            // shapes' full-weight positions do they sit at instead?
            let q = |p: [f32; 3]| p.map(|v| (v * 1e5).round() as i64);
            let base: std::collections::HashSet<[i64; 3]> = src.morphs.base.iter().map(|v| q(v.position)).collect();
            let unmatched: Vec<[i64; 3]> = m.mesh.vertices.iter().map(|v| q(v.position)).filter(|p| !base.contains(p)).collect();
            if unmatched.is_empty() {
                continue;
            }
            let mut by_shape: Vec<(usize, &str, f32)> = src
                .morphs
                .targets
                .iter()
                .map(|t| {
                    let bent: std::collections::HashSet<[i64; 3]> = t
                        .offsets
                        .iter()
                        .map(|&(j, dp, _)| {
                            let b = src.morphs.base[j as usize].position;
                            q([b[0] + dp[0], b[1] + dp[1], b[2] + dp[2]])
                        })
                        .collect();
                    (unmatched.iter().filter(|p| bent.contains(*p)).count(), t.name.as_str(), t.weight)
                })
                .filter(|(n, _, _)| *n > 0)
                .collect();
            by_shape.sort_by(|a, b| b.0.cmp(&a.0));
            eprintln!("    {} verts match no undeformed source vertex; at a shape's full weight: {:?}", unmatched.len(), &by_shape[..by_shape.len().min(6)]);
        }
    }
}

/// Where a real project's VRAM goes once opened, and what its imported maps
/// would cost uploaded the old way (plain RGBA8, no mips). Run with
/// `ZESTSURFACE_TEST_PROJECT=<file.zestsurface> cargo test --release
/// vram_of_real_project -- --ignored --nocapture`.
#[cfg(test)]
mod vram_bench {
    use super::*;

    #[test]
    #[ignore]
    fn vram_of_real_project() {
        let Ok(path) = std::env::var("ZESTSURFACE_TEST_PROJECT") else {
            eprintln!("ZESTSURFACE_TEST_PROJECT not set; skipping");
            return;
        };
        let Some(gpu) = crate::environment::test_gpu::context() else { return };
        let mib = |b: u64| b as f64 / (1u64 << 20) as f64;
        let physical = gpu.queue.device().physical_device().clone();
        let start = crate::vram::budget(&physical).map(|b| b.process_usage).unwrap_or(0);

        let loaded = read(Path::new(&path)).unwrap();
        let res = loaded.resolution().unwrap_or(2048);
        let t = std::time::Instant::now();
        let mut restored = restore(&gpu, loaded, res).unwrap();
        eprintln!("restore: {:.1?}, {} models, {} sets", t.elapsed(), restored.models.len(), restored.sets.len());

        let environment = crate::environment::Environment::procedural(&gpu);
        let mut renderer = crate::renderer::SceneRenderer::new(
            gpu.queue.clone(),
            gpu.memory_allocator.clone(),
            gpu.command_buffer_allocator.clone(),
            gpu.descriptor_set_allocator.clone(),
            Format::R8G8B8A8_UNORM,
            environment,
        );
        let t = std::time::Instant::now();
        renderer.set_models(&restored.models);
        eprintln!("map upload (compressing): {:.1?}", t.elapsed());
        // The app keeps composites for the sets you can see.
        for set in &mut restored.sets {
            set.layers.set_resident(set.visible);
        }

        // The old upload: one RGBA8 image per distinct (image, colour space).
        let mut seen = std::collections::HashSet::new();
        let mut rgba8 = 0u64;
        for m in &restored.models {
            let mat = &m.material;
            let maps = [
                (mat.base_color_image.clone(), true),
                (mat.roughness_image.clone(), false),
                (mat.metallic_image.clone(), false),
                (mat.effective_normal_image(), false),
                (mat.ao_image.clone(), false),
                (mat.emission_texture(), true),
            ];
            for (img, srgb) in maps.into_iter() {
                if let Some(img) = img
                    && seen.insert((Arc::as_ptr(&img.rgba) as usize, srgb))
                {
                    rgba8 += img.width as u64 * img.height as u64 * 4;
                }
            }
        }

        let report = crate::vram::report(|_| true);
        for (kind, usage) in &report.kinds {
            eprintln!("{:<16} {:>8.1} MiB  {:>4} images", kind.label(), mib(usage.bytes), usage.count);
        }
        eprintln!("tracked total    {:>8.1} MiB", mib(report.total()));
        eprintln!("imported maps as RGBA8 (old): {:.1} MiB in {} images", mib(rgba8), seen.len());
        if let Some(b) = crate::vram::budget(&physical) {
            eprintln!("driver: process grew {:.1} MiB", mib(b.process_usage.saturating_sub(start)));
        }
    }
}
