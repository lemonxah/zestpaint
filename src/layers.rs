use std::sync::Arc;

use half::f16;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, BlitImageInfo, ClearColorImageInfo, CommandBufferUsage, CopyBufferToImageInfo,
    CopyImageToBufferInfo, PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract,
};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::format::{ClearColorValue, Format};
use vulkano::image::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo};
use vulkano::image::view::ImageView;
use vulkano::image::{ImageCreateInfo, ImageType, ImageUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint};
use vulkano::sync::GpuFuture;

use crate::environment::{build_compute, GpuContext};
use crate::scene::{ImageData, MapSlot};

/// Default ceiling on undo history, in bytes of host memory. Snapshots live in
/// system RAM rather than VRAM, so depth is governed by this budget instead of
/// a fixed step count.
pub const DEFAULT_UNDO_BUDGET: u64 = 4 * 1024 * 1024 * 1024;

/// The budget new stacks start with: Preferences (saved in settings.json)
/// sets it, so sets made later by an import or a project open follow it.
static UNDO_BUDGET: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(DEFAULT_UNDO_BUDGET);

pub fn set_default_undo_budget(bytes: u64) {
    UNDO_BUDGET.store(bytes, std::sync::atomic::Ordering::Relaxed);
}

/// Texels of padding grown outside each UV island. Bilinear filtering only
/// reaches one texel past the edge and there is no mipmapping, so two is
/// ample; every extra pass costs a full-image dispatch per composite. Must
/// stay even so the ping-pong leaves the result in the original image.
const DILATE_PASSES: u32 = 2;

/// A paintable layer channel (used to snapshot only what a stroke touches).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PaintChannel {
    BaseColor,
    Roughness,
    Metallic,
    Height,
    Emissive,
    Opacity,
    /// The layer's mask, painted when the brush targets the mask rather than
    /// the material channels.
    Mask,
}

impl PaintChannel {
    /// Bit in `Layer::used`. Normal has no PaintChannel (fill-only), so it
    /// gets its own bit, `NORMAL_BIT`.
    pub fn bit(self) -> u8 {
        match self {
            PaintChannel::BaseColor => 1 << 0,
            PaintChannel::Roughness => 1 << 1,
            PaintChannel::Metallic => 1 << 2,
            PaintChannel::Height => 1 << 3,
            PaintChannel::Emissive => 1 << 4,
            PaintChannel::Opacity => 1 << 5,
            PaintChannel::Mask => 0,
        }
    }

    /// Index in `CHANNEL_FORMATS` / `Layer::channels` order; None for the mask.
    pub fn channel_index(self) -> Option<usize> {
        match self {
            PaintChannel::BaseColor => Some(0),
            PaintChannel::Roughness => Some(1),
            PaintChannel::Metallic => Some(2),
            PaintChannel::Height => Some(4),
            PaintChannel::Emissive => Some(5),
            PaintChannel::Opacity => Some(6),
            PaintChannel::Mask => None,
        }
    }
}

/// `Layer::used` bit for fill-material normals.
pub const NORMAL_BIT: u8 = 1 << 6;

/// What a shader-mask layer feeds. Avatar toon shaders (Poiyomi, lilToon,
/// UTS) take a greyscale texture per effect — white shows the effect, black
/// hides it. Every layer tagged with the same kind composites into one such
/// texture, kept apart from the material and exported on its own.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, serde::Serialize, serde::Deserialize)]
pub enum ShaderMask {
    Emission,
    Matcap,
    Matcap2,
    RimLight,
    Outline,
    Shadow,
    Specular,
    Glitter,
    Detail,
    Dissolve,
    AudioLink,
    Custom,
}

impl ShaderMask {
    /// The kinds a layer can be made for. `Matcap2` is not among them: it is
    /// only read from older projects, which turn it into a matcap layer.
    pub const ALL: [ShaderMask; 11] = [
        ShaderMask::Emission,
        ShaderMask::Matcap,
        ShaderMask::RimLight,
        ShaderMask::Outline,
        ShaderMask::Shadow,
        ShaderMask::Specular,
        ShaderMask::Glitter,
        ShaderMask::Detail,
        ShaderMask::Dissolve,
        ShaderMask::AudioLink,
        ShaderMask::Custom,
    ];

    pub fn label(self) -> &'static str {
        match self {
            ShaderMask::Emission => "Emission mask",
            ShaderMask::Matcap => "Matcap mask",
            ShaderMask::Matcap2 => "Matcap 2 mask",
            ShaderMask::RimLight => "Rim light mask",
            ShaderMask::Outline => "Outline mask",
            ShaderMask::Shadow => "Shadow mask",
            ShaderMask::Specular => "Specular mask",
            ShaderMask::Glitter => "Glitter mask",
            ShaderMask::Detail => "Detail mask",
            ShaderMask::Dissolve => "Dissolve mask",
            ShaderMask::AudioLink => "AudioLink mask",
            ShaderMask::Custom => "Custom mask",
        }
    }

    /// Short tag for the layer list badge.
    pub fn tag(self) -> &'static str {
        match self {
            ShaderMask::Emission => "EMISSION",
            ShaderMask::Matcap => "MATCAP",
            ShaderMask::Matcap2 => "MATCAP 2",
            ShaderMask::RimLight => "RIM",
            ShaderMask::Outline => "OUTLINE",
            ShaderMask::Shadow => "SHADOW",
            ShaderMask::Specular => "SPECULAR",
            ShaderMask::Glitter => "GLITTER",
            ShaderMask::Detail => "DETAIL",
            ShaderMask::Dissolve => "DISSOLVE",
            ShaderMask::AudioLink => "AUDIOLINK",
            ShaderMask::Custom => "CUSTOM",
        }
    }

    /// Export file suffix: `<name>_<suffix>.png`. Always ends in `mask`,
    /// which keeps Import maps from mistaking one for a map.
    pub fn suffix(self) -> &'static str {
        match self {
            ShaderMask::Emission => "emission_mask",
            ShaderMask::Matcap => "matcap_mask",
            ShaderMask::Matcap2 => "matcap2_mask",
            ShaderMask::RimLight => "rim_light_mask",
            ShaderMask::Outline => "outline_mask",
            ShaderMask::Shadow => "shadow_mask",
            ShaderMask::Specular => "specular_mask",
            ShaderMask::Glitter => "glitter_mask",
            ShaderMask::Detail => "detail_mask",
            ShaderMask::Dissolve => "dissolve_mask",
            ShaderMask::AudioLink => "audio_link_mask",
            ShaderMask::Custom => "custom_mask",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            ShaderMask::Emission => "Where the shader's emission glows",
            ShaderMask::Matcap => "A matcap (shine, metal, rim light…) and where it shows",
            ShaderMask::Matcap2 => "Where the second matcap applies",
            ShaderMask::RimLight => "Where rim lighting shows",
            ShaderMask::Outline => "Outline width/visibility",
            ShaderMask::Shadow => "Shadow strength — black keeps areas lit",
            ShaderMask::Specular => "Where specular highlights show",
            ShaderMask::Glitter => "Where glitter/sparkle shows",
            ShaderMask::Detail => "Where detail textures apply",
            ShaderMask::Dissolve => "Dissolve pattern / order",
            ShaderMask::AudioLink => "Where AudioLink effects react",
            ShaderMask::Custom => "Any other greyscale mask your shader takes",
        }
    }

    /// Badge and viewport-overlay colour (sRGB).
    pub fn tint(self) -> [u8; 3] {
        match self {
            ShaderMask::Emission => [255, 150, 40],
            ShaderMask::Matcap => [80, 200, 255],
            ShaderMask::Matcap2 => [120, 140, 255],
            ShaderMask::RimLight => [255, 240, 120],
            ShaderMask::Outline => [200, 200, 200],
            ShaderMask::Shadow => [150, 110, 220],
            ShaderMask::Specular => [240, 240, 255],
            ShaderMask::Glitter => [255, 120, 220],
            ShaderMask::Detail => [120, 220, 140],
            ShaderMask::Dissolve => [255, 90, 90],
            ShaderMask::AudioLink => [60, 255, 200],
            ShaderMask::Custom => [180, 180, 120],
        }
    }
}

/// A pre-stroke copy of the channels one stroke touched, staged into host
/// memory. Keeping these off the GPU is what lets the history run deep: VRAM
/// only ever holds the live layers.
struct HistoryEntry {
    /// `Layer::id` of the painted layer.
    layer: u64,
    maps: Vec<(PaintChannel, Subbuffer<[u8]>)>,
    bytes: u64,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
pub enum BlendMode {
    Normal,
    Multiply,
    Add,
    Screen,
    Overlay,
    SoftLight,
    HardLight,
    Darken,
    Lighten,
    ColorDodge,
    ColorBurn,
    Difference,
    Subtract,
    /// The component modes: they mix colour properties, so on greyscale
    /// channels (roughness, metallic…) they act as Normal.
    Hue,
    Saturation,
    Color,
    Luminosity,
}

impl BlendMode {
    /// In menu order, grouped the way painting apps group them.
    pub const ALL: [BlendMode; 17] = [
        BlendMode::Normal,
        BlendMode::Multiply,
        BlendMode::Darken,
        BlendMode::ColorBurn,
        BlendMode::Screen,
        BlendMode::Lighten,
        BlendMode::ColorDodge,
        BlendMode::Add,
        BlendMode::Overlay,
        BlendMode::SoftLight,
        BlendMode::HardLight,
        BlendMode::Difference,
        BlendMode::Subtract,
        BlendMode::Hue,
        BlendMode::Saturation,
        BlendMode::Color,
        BlendMode::Luminosity,
    ];

    pub fn label(self) -> &'static str {
        match self {
            BlendMode::Normal => "Normal",
            BlendMode::Multiply => "Multiply",
            BlendMode::Add => "Add",
            BlendMode::Screen => "Screen",
            BlendMode::Overlay => "Overlay",
            BlendMode::SoftLight => "Soft light",
            BlendMode::HardLight => "Hard light",
            BlendMode::Darken => "Darken",
            BlendMode::Lighten => "Lighten",
            BlendMode::ColorDodge => "Colour dodge",
            BlendMode::ColorBurn => "Colour burn",
            BlendMode::Difference => "Difference",
            BlendMode::Subtract => "Subtract",
            BlendMode::Hue => "Hue",
            BlendMode::Saturation => "Saturation",
            BlendMode::Color => "Colour",
            BlendMode::Luminosity => "Luminosity",
        }
    }

    /// Four-letter tag for the layer list.
    pub fn short_label(self) -> &'static str {
        match self {
            BlendMode::Normal => "Norm",
            BlendMode::Multiply => "Mult",
            BlendMode::Add => "Add",
            BlendMode::Screen => "Scrn",
            BlendMode::Overlay => "Ovly",
            BlendMode::SoftLight => "Soft",
            BlendMode::HardLight => "Hard",
            BlendMode::Darken => "Dark",
            BlendMode::Lighten => "Lite",
            BlendMode::ColorDodge => "Dodg",
            BlendMode::ColorBurn => "Burn",
            BlendMode::Difference => "Diff",
            BlendMode::Subtract => "Sub",
            BlendMode::Hue => "Hue",
            BlendMode::Saturation => "Sat",
            BlendMode::Color => "Col",
            BlendMode::Luminosity => "Lum",
        }
    }

    /// Where the menu puts a separator before this mode.
    pub fn starts_group(self) -> bool {
        matches!(
            self,
            BlendMode::Multiply
                | BlendMode::Screen
                | BlendMode::Overlay
                | BlendMode::Difference
                | BlendMode::Hue
        )
    }

    /// Must match `blend1` / `blend3` in the compositor shaders.
    fn shader_id(self) -> u32 {
        match self {
            BlendMode::Normal => 0,
            BlendMode::Multiply => 1,
            BlendMode::Add => 2,
            BlendMode::Screen => 3,
            BlendMode::Overlay => 4,
            BlendMode::SoftLight => 5,
            BlendMode::HardLight => 6,
            BlendMode::Darken => 7,
            BlendMode::Lighten => 8,
            BlendMode::ColorDodge => 9,
            BlendMode::ColorBurn => 10,
            BlendMode::Difference => 11,
            BlendMode::Subtract => 12,
            BlendMode::Hue => 13,
            BlendMode::Saturation => 14,
            BlendMode::Color => 15,
            BlendMode::Luminosity => 16,
        }
    }
}

/// How a fill layer's content is wrapped onto the surface. Everything except
/// `Uv` ignores the mesh's UV layout and places the image by where each texel
/// sits on the model (the baked position/normal maps), so it reads cleanly
/// across seams and island scale differences — balls, cans, rocks.
#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
pub enum Projection {
    /// Through the mesh's own UVs (tiling, offset, rotation).
    Uv,
    /// Three planar projections blended by surface normal. Seamless on almost
    /// anything; the default choice for props.
    Triplanar,
    /// Longitude/latitude around the axis, from the object's centre: one
    /// equirectangular (2:1) image wraps a ball. Pinches at the poles.
    Spherical,
    /// Wrapped around the axis, straight along it: labels on cans and bottles.
    Cylindrical,
    /// Straight through the object along the axis, like a slide projector.
    Planar,
}

impl Projection {
    pub const ALL: [Projection; 5] = [
        Projection::Uv,
        Projection::Triplanar,
        Projection::Spherical,
        Projection::Cylindrical,
        Projection::Planar,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Projection::Uv => "UV",
            Projection::Triplanar => "Triplanar",
            Projection::Spherical => "Spherical",
            Projection::Cylindrical => "Cylindrical",
            Projection::Planar => "Planar",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Projection::Uv => "Follow the mesh's UV layout",
            Projection::Triplanar => "Project from all six sides, blended by surface angle — seamless on most shapes",
            Projection::Spherical => "Wrap one image around a ball (2:1 equirectangular images fit best)",
            Projection::Cylindrical => "Wrap around the axis like a label on a can",
            Projection::Planar => "Shine straight through the object along the axis",
        }
    }

    /// Whether this mode is aimed with `Layer::projection_axis`.
    pub fn uses_axis(self) -> bool {
        matches!(self, Projection::Spherical | Projection::Cylindrical | Projection::Planar)
    }

    fn shader_id(self) -> u32 {
        match self {
            Projection::Uv => 0,
            Projection::Triplanar => 1,
            Projection::Spherical => 2,
            Projection::Cylindrical => 3,
            Projection::Planar => 4,
        }
    }
}

/// World axis a projection is aimed along: the pole of a spherical or
/// cylindrical wrap, the throw direction of a planar one.
#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
pub enum Axis {
    X,
    Y,
    Z,
}

impl Axis {
    pub const ALL: [Axis; 3] = [Axis::X, Axis::Y, Axis::Z];

    pub fn label(self) -> &'static str {
        match self {
            Axis::X => "X",
            Axis::Y => "Y",
            Axis::Z => "Z",
        }
    }

    fn shader_id(self) -> u32 {
        match self {
            Axis::X => 0,
            Axis::Y => 1,
            Axis::Z => 2,
        }
    }
}

/// One paint layer. Values are stored premultiplied-alpha in linear space:
/// base color in RGBA16F, roughness/metallic as (value·α, α) pairs in RG16F.
///
/// Cloning is cheap — textures are shared `Arc`s — which is what document
/// snapshots for undo rely on. That only holds because no edit outside a
/// stroke writes into a texture in place: fills, mask edits and imports put
/// their result in a fresh texture, so an older snapshot keeps the old one.
/// Strokes do paint in place, and carry their own pixel history.
#[derive(Clone)]
pub struct Layer {
    /// Stable identity within its stack, for stroke history to find the
    /// layer again after reordering, deleting and undoing.
    pub id: u64,
    pub name: String,
    pub visible: bool,
    pub opacity: f32,
    pub blend: BlendMode,
    pub basecolor: Arc<ImageView>,
    pub roughness: Arc<ImageView>,
    pub metallic: Arc<ImageView>,
    /// Tangent-space normal from a fill material (premultiplied xyz·α, α).
    /// Fill-only — the brush perturbs the surface through `height` instead,
    /// and the two are combined when the composited normal is derived.
    pub normal: Arc<ImageView>,
    /// Signed height values: positive raises, negative carves.
    pub height: Arc<ImageView>,
    /// Emitted light color (premultiplied linear).
    pub emissive: Arc<ImageView>,
    /// Surface opacity: painted values < 0.5 alpha-clip in the viewport.
    pub opacity_map: Arc<ImageView>,
    /// Optional mask multiplying this layer's contribution (x = mask value).
    pub mask: Option<Arc<ImageView>>,
    /// The mask is applied. Off shows the layer as if it had none, without
    /// losing the mask.
    pub mask_enabled: bool,
    /// Clipped to the layer below: shows only where that layer (the nearest
    /// unclipped one below) has paint.
    pub clip: bool,
    /// How the mask was generated, for the UI and regeneration.
    pub mask_kind: Option<MaskKind>,
    /// UV tiling for this layer's channels: how many times the content repeats
    /// across the 0..1 UV range. Only meaningful for fill layers, whose
    /// channels hold a tileable material; the mask is deliberately *not*
    /// transformed, so retiling the material leaves painted coverage in place.
    pub uv_scale: [f32; 2],
    /// UV shift applied after scaling, for lining tiling up with a seam.
    pub uv_offset: [f32; 2],
    /// Rotation in radians, about the middle of the UV square.
    pub uv_rotation: f32,
    /// How the channels are wrapped onto the model. The UV transform above
    /// still applies on top, to the projected coordinates.
    pub projection: Projection,
    pub projection_axis: Axis,
    /// Triplanar blend falloff: higher keeps each side's projection to the
    /// faces pointing at it, lower smears them together over a wider band.
    pub triplanar_sharpness: f32,
    /// Where the projection sits, in world units from the centre of the
    /// object's bounding box. Moved by the Place tool.
    pub projection_offset: [f32; 3],
    /// Projection orientation, XYZ Euler radians, on top of the axis preset.
    pub projection_rotation: [f32; 3],
    /// Projection size as a multiple of the object's longest side: 1 fits
    /// the image to the whole object, 0.2 makes it a small decal.
    pub projection_scale: f32,
    /// Tile the image past the projection's edges. Off places it exactly
    /// once — a logo or label — and leaves the rest to the layers below.
    pub projection_repeat: bool,
    /// True once a library material has been filled in, which is what makes
    /// the tiling controls meaningful.
    pub has_fill: bool,
    /// Some = a shader-mask layer: greyscale lives in `basecolor`, and the
    /// layer composites into that mask's texture instead of the material.
    pub purpose: Option<ShaderMask>,
    /// On a `ShaderMask::Matcap` layer: the matcap its mask reveals. Each
    /// matcap layer flattens to its own mask, so one set can carry any
    /// number of them. None until a matcap is picked.
    pub matcap: Option<crate::matcap::Matcap>,
    /// Roughly what colour the layer is (linear RGB), for its thumbnail in
    /// the layer list: set when it is filled or painted, None until then.
    /// A first-pass stand-in for a rendered thumbnail.
    pub swatch: Option<[f32; 3]>,
    /// Which channels have had content put in (`PaintChannel::bit`,
    /// `NORMAL_BIT`) — drives the layer list badges. Erasing doesn't clear
    /// it; it answers "has this layer ever held emissive", not "is it empty".
    pub used: u8,
}

/// A layer stack's document state, for undo (see `LayerStack::state`).
#[derive(Clone)]
pub struct StackState {
    layers: Vec<Layer>,
    active: usize,
    height_scale: f32,
    projection_source: Option<ProjectionSource>,
    next_layer_number: u32,
}

/// A layer channel that left its 1x1 placeholder: (layer id, channel index
/// in `CHANNEL_FORMATS` order, the placeholder, its replacement).
pub type Upgrade = (u64, usize, Arc<ImageView>, Arc<ImageView>);

impl StackState {
    /// Point layers still on placeholders at the textures that replaced them,
    /// so a saved state reads the same as the live stack after
    /// `LayerStack::ensure_allocated`.
    pub fn adopt_upgrades(&mut self, upgrades: &[Upgrade]) {
        for layer in &mut self.layers {
            for (id, channel, old, fresh) in upgrades {
                if layer.id == *id {
                    let Some(slot) = layer.channels_mut().into_iter().nth(*channel) else { continue };
                    if Arc::ptr_eq(slot, old) {
                        *slot = fresh.clone();
                    }
                }
            }
        }
    }

    /// The same document: layers, bump strength and projection frame. The
    /// selection is not part of it — picking a layer is not an edit.
    pub fn same_document(&self, other: &StackState) -> bool {
        self.height_scale == other.height_scale
            && self.projection_source.as_ref().map(|s| (Arc::as_ptr(&s.position), s.frame))
                == other.projection_source.as_ref().map(|s| (Arc::as_ptr(&s.position), s.frame))
            && self.layers.len() == other.layers.len()
            && self.layers.iter().zip(&other.layers).all(|(a, b)| a.same_as(b))
    }
}

impl Layer {
    /// A matcap layer: its greyscale is where its own matcap shows.
    pub fn is_matcap(&self) -> bool {
        self.purpose == Some(ShaderMask::Matcap)
    }

    /// Give this matcap layer `image` (sRGB) as its matcap, with `blend` if
    /// given. A layer still under its default name is renamed after it.
    pub fn set_matcap(
        &mut self,
        gpu: &GpuContext,
        name: String,
        image: crate::scene::ImageData,
        blend: Option<crate::matcap::MatcapBlend>,
    ) {
        if self.name == "Matcap" || self.name.starts_with("Matcap · ") {
            self.name = format!("Matcap · {name}");
        }
        let m = self.matcap.get_or_insert_with(Default::default);
        m.set_image(gpu, name, image);
        if let Some(blend) = blend {
            m.blend = blend;
        }
    }

    /// Channel textures in `CHANNEL_FORMATS` order.
    pub fn channels(&self) -> [&Arc<ImageView>; 7] {
        [&self.basecolor, &self.roughness, &self.metallic, &self.normal, &self.height, &self.emissive, &self.opacity_map]
    }

    fn channels_mut(&mut self) -> [&mut Arc<ImageView>; 7] {
        [
            &mut self.basecolor,
            &mut self.roughness,
            &mut self.metallic,
            &mut self.normal,
            &mut self.height,
            &mut self.emissive,
            &mut self.opacity_map,
        ]
    }

    /// Same content and settings: textures by identity, the rest by value.
    pub fn same_as(&self, other: &Layer) -> bool {
        let tex = |a: &Arc<ImageView>, b: &Arc<ImageView>| Arc::ptr_eq(a, b);
        self.id == other.id
            && self.name == other.name
            && self.visible == other.visible
            && self.opacity == other.opacity
            && self.blend == other.blend
            && tex(&self.basecolor, &other.basecolor)
            && tex(&self.roughness, &other.roughness)
            && tex(&self.metallic, &other.metallic)
            && tex(&self.normal, &other.normal)
            && tex(&self.height, &other.height)
            && tex(&self.emissive, &other.emissive)
            && tex(&self.opacity_map, &other.opacity_map)
            && match (&self.mask, &other.mask) {
                (Some(a), Some(b)) => tex(a, b),
                (None, None) => true,
                _ => false,
            }
            && self.mask_kind == other.mask_kind
            && self.mask_enabled == other.mask_enabled
            && self.clip == other.clip
            && self.uv_scale == other.uv_scale
            && self.uv_offset == other.uv_offset
            && self.uv_rotation == other.uv_rotation
            && self.projection == other.projection
            && self.projection_axis == other.projection_axis
            && self.triplanar_sharpness == other.triplanar_sharpness
            && self.projection_offset == other.projection_offset
            && self.projection_rotation == other.projection_rotation
            && self.projection_scale == other.projection_scale
            && self.projection_repeat == other.projection_repeat
            && self.has_fill == other.has_fill
            && self.purpose == other.purpose
            && match (&self.matcap, &other.matcap) {
                (Some(a), Some(b)) => a.same_as(b),
                (None, None) => true,
                _ => false,
            }
            && self.swatch == other.swatch
            && self.used == other.used
    }

    /// The projection volume in world space: maps the unit cube
    /// (-0.5..0.5) the projections are defined in onto the object, given the
    /// object's bounds centre and longest side.
    pub fn projection_placement(&self, centre: [f32; 3], size: f32) -> glam::Mat4 {
        let [rx, ry, rz] = self.projection_rotation;
        glam::Mat4::from_scale_rotation_translation(
            glam::Vec3::splat(size * self.projection_scale.max(1e-4)),
            glam::Quat::from_euler(glam::EulerRot::XYZ, rx, ry, rz),
            glam::Vec3::from(centre) + glam::Vec3::from(self.projection_offset),
        )
    }

    /// Put baked channels in (order as `LayerStack::bake` returns them).
    fn set_channels(&mut self, [bc, rough, metal, height, emissive, opacity, normal]: [Arc<ImageView>; 7]) {
        self.basecolor = bc;
        self.roughness = rough;
        self.metallic = metal;
        self.height = height;
        self.emissive = emissive;
        self.opacity_map = opacity;
        self.normal = normal;
    }

    /// Plain UV placement: no tiling, offset, spin or projection. For
    /// layers whose content has just been baked into UV space.
    fn reset_placement(&mut self) {
        self.uv_scale = [1.0, 1.0];
        self.uv_offset = [0.0, 0.0];
        self.uv_rotation = 0.0;
        self.projection = Projection::Uv;
        self.reset_projection_placement();
        self.has_fill = false;
    }

    /// Back to where a fresh projection starts: centred, upright, covering
    /// the whole object.
    pub fn reset_projection_placement(&mut self) {
        self.projection_offset = [0.0; 3];
        self.projection_rotation = [0.0; 3];
        self.projection_scale = 1.0;
    }
}

/// Procedural mask generators driven by baked mesh maps.
#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
pub enum MaskKind {
    /// Bright on convex edges — scratched/worn raised detail.
    EdgeWear,
    /// Bright in concave crevices (optionally weighted by AO) — dirt/grime.
    CavityDirt,
}

impl MaskKind {
    pub const ALL: [MaskKind; 2] = [MaskKind::EdgeWear, MaskKind::CavityDirt];

    pub fn label(self) -> &'static str {
        match self {
            MaskKind::EdgeWear => "Edge wear",
            MaskKind::CavityDirt => "Cavity dirt",
        }
    }
}

/// The flattened maps a texture set exports, and the maps a file can be
/// imported as. AO is export-only: there is no per-layer occlusion channel.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MapKind {
    BaseColor,
    Roughness,
    Metallic,
    Height,
    Normal,
    Emissive,
    Opacity,
    Ao,
}

impl MapKind {
    pub const ALL: [MapKind; 8] = [
        MapKind::BaseColor,
        MapKind::Roughness,
        MapKind::Metallic,
        MapKind::Height,
        MapKind::Normal,
        MapKind::Emissive,
        MapKind::Opacity,
        MapKind::Ao,
    ];

    pub fn label(self) -> &'static str {
        match self {
            MapKind::BaseColor => "Base color",
            MapKind::Roughness => "Roughness",
            MapKind::Metallic => "Metallic",
            MapKind::Height => "Height",
            MapKind::Normal => "Normal",
            MapKind::Emissive => "Emissive",
            MapKind::Opacity => "Opacity",
            MapKind::Ao => "AO",
        }
    }

    /// File-name suffix used on export (`<set>_<suffix>.png`).
    pub fn suffix(self) -> &'static str {
        match self {
            MapKind::BaseColor => "color",
            MapKind::Roughness => "roughness",
            MapKind::Metallic => "metallic",
            MapKind::Height => "height",
            MapKind::Normal => "normals",
            MapKind::Emissive => "emission",
            MapKind::Opacity => "opacity",
            MapKind::Ao => "ao",
        }
    }

    /// Whether a file can be imported as this map onto a layer.
    pub fn importable(self) -> bool {
        self != MapKind::Ao
    }

    /// Guess a map from a file name. Our own export suffix (the last `_`
    /// token) is checked first so an object called "Metal_Plate" exporting
    /// `Metal_Plate_roughness.png` comes back as roughness; then the usual
    /// naming conventions (Substance, ambientCG, Poly Haven). Masks
    /// (`…_mask`, layer or shader) are not maps: `…_emission_mask` must not
    /// come back as emission.
    pub fn detect(file_name: &str) -> Option<MapKind> {
        let stem = file_name.rsplit_once('.').map_or(file_name, |(s, _)| s).to_lowercase();
        let last = stem.rsplit(['_', '-', ' ', '.']).next().unwrap_or(&stem);
        if last.ends_with("mask") {
            return None;
        }
        if let Some(kind) = MapKind::ALL.into_iter().find(|k| k.suffix() == last) {
            return Some(kind);
        }
        let has = |k: &str| stem.contains(k);
        Some(if has("basecolor") || has("base_color") || has("albedo") || has("diffuse") {
            MapKind::BaseColor
        } else if has("emissi") || has("emission") || has("glow") {
            MapKind::Emissive
        } else if has("opacity") || has("alpha") {
            MapKind::Opacity
        } else if has("rough") {
            MapKind::Roughness
        } else if has("metal") {
            MapKind::Metallic
        } else if has("normal") {
            MapKind::Normal
        } else if has("height") || has("displacement") || has("bump") {
            MapKind::Height
        } else if has("occlusion") || last == "ao" {
            MapKind::Ao
        } else if has("color") || has("col") {
            MapKind::BaseColor
        } else {
            return None;
        })
    }
}

/// Where a layer's mask can be taken from, besides painting and images.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MaskSource {
    /// A copy of another layer's mask.
    LayerMask(usize),
    /// Where another layer has paint: its coverage in any channel.
    LayerCoverage(usize),
    /// The flattened shader mask of this kind (Matcap, Emission…).
    ShaderMask(ShaderMask),
}

/// One paintable sub-model: its own layer stack and baked maps. Models with
/// the same source name (e.g. all "Shirt" primitives) share one set.
pub struct TextureSet {
    /// Object (node) this material belongs to — the tree parent.
    pub object: String,
    /// Material name — the actual paint target.
    pub material: String,
    pub layers: LayerStack,
    pub baked_ao: Option<Arc<ImageView>>,
    /// World-space position per texel (xyz, w = coverage).
    pub baked_position: Option<Arc<ImageView>>,
    /// World-space normal per texel (xyz, w = coverage).
    pub baked_normal: Option<Arc<ImageView>>,
    /// Mean curvature per texel (x, signed: + convex, − concave; y = coverage).
    pub baked_curvature: Option<Arc<ImageView>>,
    /// Whether this material's geometry is drawn in the viewport. Hidden sets
    /// are also excluded from painting and AO baking (so you can hide a
    /// jacket and paint the body underneath).
    pub visible: bool,
    /// The material's own textures were offered to the stack as its base
    /// layer (`LayerStack::seed_from_material`) — done once per set, so an
    /// undo of it sticks.
    pub seeded: bool,
    /// The material's own textures were copied into the stack's "Base
    /// material" layer, so the viewport stops drawing them underneath:
    /// hiding or erasing that layer then really removes them.
    pub material_in_layers: bool,
}

impl TextureSet {
    /// Whether the material's colour/roughness/metallic textures are left
    /// out of shading because the layers carry them. An untouched stack
    /// (the seeding undone) shows the material again.
    pub fn material_drawn_by_layers(&self) -> bool {
        self.material_in_layers && !self.layers.is_pristine()
    }

    pub fn new(object: String, material: String, gpu: GpuContext, resolution: u32) -> Self {
        Self::with_stack(object, material, LayerStack::new(gpu, resolution))
    }

    fn with_stack(object: String, material: String, layers: LayerStack) -> Self {
        Self {
            object,
            material,
            layers,
            baked_ao: None,
            baked_position: None,
            baked_normal: None,
            baked_curvature: None,
            visible: true,
            seeded: false,
            material_in_layers: false,
        }
    }

    /// A set whose paint textures are allocated on first use — see
    /// `LayerStack::new_lazy`. Imports and project loads build these.
    pub fn new_lazy(object: String, material: String, gpu: GpuContext, resolution: u32) -> Self {
        Self::with_stack(object, material, LayerStack::new_lazy(gpu, resolution))
    }

    /// Display / file-prefix label: "Object · Material".
    pub fn label(&self) -> String {
        format!("{} · {}", self.object, self.material)
    }
}

/// The paint document: an ordered stack of layers (bottom → top) plus the
/// GPU compositor that flattens them into the textures the PBR shader samples.
pub struct LayerStack {
    gpu: GpuContext,
    pub layers: Vec<Layer>,
    pub active: usize,
    pub resolution: u32,
    pub composite_basecolor: Arc<ImageView>,
    /// RG16F, like the other scalar composites: (value*a, a).
    pub composite_roughness: Arc<ImageView>,
    pub composite_metallic: Arc<ImageView>,
    pub composite_height: Arc<ImageView>,
    pub composite_emissive: Arc<ImageView>,
    pub composite_opacity: Arc<ImageView>,
    /// Flattened fill-material normals, before the height gradient is folded in.
    pub composite_normal: Arc<ImageView>,
    /// Tangent-space normals the shader samples: the composited fill normals
    /// perturbed by the gradient of the composited height, plus coverage (w).
    pub normal_map: Arc<ImageView>,
    /// Gradient strength used when deriving normals from height.
    pub height_scale: f32,
    pub sampler: Arc<Sampler>,
    /// Repeating sampler for layer channels, so a fill layer scaled past 1
    /// tiles instead of smearing its edge texels.
    tiling_sampler: Arc<Sampler>,
    /// 1x1 white — the no-mask fallback for the compositor binding.
    white: Arc<ImageView>,
    pipeline: Arc<ComputePipeline>,
    /// Composites shader-mask layers into their per-kind textures.
    mask_pipeline: Arc<ComputePipeline>,
    /// One flattened texture per shader-mask kind in use (not Matcap).
    shader_masks: Vec<(ShaderMask, Arc<ImageView>)>,
    /// One flattened mask per matcap layer, by layer id.
    matcap_masks: Vec<(u64, Arc<ImageView>)>,
    normal_pipeline: Arc<ComputePipeline>,
    dilate_pipeline: Arc<ComputePipeline>,
    /// `dilate_pipeline` for the RG16F (scalar) composites.
    dilate_rg_pipeline: Arc<ComputePipeline>,
    /// Ping-pong targets for the dilation passes, one per composite format.
    /// Shared with every allocated stack of this size (see `shared_scratch`).
    dilate_temp: Arc<ImageView>,
    dilate_temp_rg: Arc<ImageView>,
    /// Coverage of the current clip base, for the layers clipped to it
    /// (shared like `dilate_temp`).
    clip_coverage: Arc<ImageView>,
    dirty: bool,
    next_layer_number: u32,
    next_layer_id: u64,
    /// Strokes begun so far, ever: the app's undo timeline watches this to
    /// interleave strokes with its other history entries.
    strokes_begun: u64,
    undo_stack: std::collections::VecDeque<HistoryEntry>,
    redo_stack: Vec<HistoryEntry>,
    /// Host memory the history may occupy before the oldest steps are dropped.
    pub undo_budget: u64,
    /// Pad composites past their UV islands. Skipped while a stroke is in
    /// flight — the seams it fixes are only worth paying for once the stroke
    /// settles, and it is a full-image dispatch per composite.
    pub dilate: bool,
    /// Where every texel sits on the model, for projected fills: the baked
    /// world position and normal maps (w = coverage), plus the frame that
    /// normalises positions — centre xyz, and in w the longest bounding-box
    /// side, so the whole object spans -0.5..0.5. None until mesh maps are
    /// baked; projected layers fall back to UV placement meanwhile.
    projection_source: Option<ProjectionSource>,
    /// Whether the composites and scratch images exist at `resolution`. A
    /// lazy stack (`new_lazy`) has 1x1 transparent stand-ins until first use.
    allocated: bool,
    /// 1x1 cleared RGBA16F / RG16F textures standing in for blank layer
    /// channels. A channel only gets a full-resolution texture when something
    /// is written into it (`real_channel`): a colour-only layer is one
    /// texture, not seven.
    placeholders: [Arc<ImageView>; 2],
    /// Channels that left their placeholder since the last `take_upgrades`.
    pending_upgrades: Vec<Upgrade>,
    /// Cleared 1x1 RGBA16F / RG16F storage images: what a composite is when
    /// no layer can put anything in it (sampled as "unpainted"; bound but
    /// never touched by the flatten, see `composite_needs`).
    blanks: [Arc<ImageView>; 2],
    /// A composite was swapped since `set_resident` last reported it.
    composites_swapped: bool,
}

/// Composite bytes a texel: four RGBA16F and four RG16F images.
const COMPOSITE_TEXEL_BYTES: u64 = 48;

/// `record_layer`'s `present` with every composite real.
const ALL_COMPOSITES: u32 = 0x7f;

/// Which flatten scratch image (see `shared_scratch`).
#[derive(Clone, Copy, PartialEq)]
enum Scratch {
    Dilate,
    DilateRg,
    ClipCoverage,
}

/// A full-resolution scratch image for flattening. It is only touched inside
/// one stack's flatten, and flattens never overlap (they are recorded one
/// after another, and the auto command buffer orders the accesses), so all
/// allocated stacks of a size share it rather than each holding its own. The
/// cache keeps no strong reference: the last stack to let go frees it.
fn shared_scratch(gpu: &GpuContext, resolution: u32, which: Scratch) -> Arc<ImageView> {
    type Key = (usize, u32, Scratch);
    static CACHE: std::sync::Mutex<Vec<(Key, std::sync::Weak<ImageView>)>> = std::sync::Mutex::new(Vec::new());
    let key = (Arc::as_ptr(gpu.queue.device()) as usize, resolution, which);
    let mut cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    cache.retain(|(_, weak)| weak.strong_count() > 0);
    if let Some(view) = cache.iter().find(|(k, _)| *k == key).and_then(|(_, w)| w.upgrade()) {
        return view;
    }
    let format = if which == Scratch::DilateRg { Format::R16G16_SFLOAT } else { Format::R16G16B16A16_SFLOAT };
    let view = create_paint_image(gpu, resolution, format, crate::vram::Kind::Scratch);
    cache.push((key, Arc::downgrade(&view)));
    view
}

/// Channel formats in `Layer::channels` order: basecolor, roughness,
/// metallic, normal, height, emissive, opacity.
const CHANNEL_FORMATS: [Format; 7] = [
    Format::R16G16B16A16_SFLOAT,
    Format::R16G16_SFLOAT,
    Format::R16G16_SFLOAT,
    Format::R16G16B16A16_SFLOAT,
    Format::R16G16_SFLOAT,
    Format::R16G16B16A16_SFLOAT,
    Format::R16G16_SFLOAT,
];

/// GPU objects every stack on a device can share: compiling four compute
/// pipelines per texture set made big imports slow for nothing.
#[derive(Clone)]
struct StackShared {
    sampler: Arc<Sampler>,
    tiling_sampler: Arc<Sampler>,
    white: Arc<ImageView>,
    pipeline: Arc<ComputePipeline>,
    mask_pipeline: Arc<ComputePipeline>,
    normal_pipeline: Arc<ComputePipeline>,
    dilate_pipeline: Arc<ComputePipeline>,
    dilate_rg_pipeline: Arc<ComputePipeline>,
}

fn stack_shared(gpu: &GpuContext) -> StackShared {
    static CACHE: std::sync::Mutex<Vec<(usize, StackShared)>> = std::sync::Mutex::new(Vec::new());
    let device = gpu.queue.device().clone();
    // The cached pipelines keep their device alive, so its address can't be reused.
    let key = Arc::as_ptr(&device) as usize;
    let mut cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, shared)) = cache.iter().find(|(k, _)| *k == key) {
        return shared.clone();
    }
    let sampler = |mode| {
        Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [mode; 3],
                ..Default::default()
            },
        )
        .unwrap()
    };
    let shared = StackShared {
        sampler: sampler(SamplerAddressMode::ClampToEdge),
        // Repeating sampler for layer channels, so a fill layer scaled past 1
        // tiles instead of smearing its edge texels.
        tiling_sampler: sampler(SamplerAddressMode::Repeat),
        white: crate::renderer::upload_texture(
            &gpu.queue,
            &gpu.memory_allocator,
            &gpu.command_buffer_allocator,
            1,
            1,
            &[255, 255, 255, 255],
            Format::R8G8B8A8_UNORM,
        ),
        pipeline: build_compute(&device, composite_cs::load(device.clone()).unwrap()),
        mask_pipeline: build_compute(&device, mask_composite_cs::load(device.clone()).unwrap()),
        normal_pipeline: build_compute(&device, normal_cs::load(device.clone()).unwrap()),
        dilate_pipeline: build_compute(&device, dilate_cs::load(device.clone()).unwrap()),
        dilate_rg_pipeline: build_compute(&device, dilate_rg_cs::load(device.clone()).unwrap()),
    };
    cache.push((key, shared.clone()));
    shared
}

#[derive(Clone)]
struct ProjectionSource {
    position: Arc<ImageView>,
    normal: Arc<ImageView>,
    frame: [f32; 4],
}

impl LayerStack {
    pub fn new(gpu: GpuContext, resolution: u32) -> Self {
        let mut stack = Self::new_lazy(gpu, resolution);
        stack.ensure_allocated();
        stack
    }

    /// A stack with nothing painted yet and (almost) no VRAM: 1x1 transparent
    /// composites and a blank first layer on 1x1 placeholders, which render as
    /// "unpainted". `ensure_allocated` swaps in the full-resolution textures
    /// on first activation or edit. Allocating them up front cost ~0.5 GiB per
    /// texture set, which on a many-part FBX overran VRAM and stalled the
    /// whole desktop.
    pub fn new_lazy(gpu: GpuContext, resolution: u32) -> Self {
        let shared = stack_shared(&gpu);
        let rgba = Format::R16G16B16A16_SFLOAT;
        let blanks = [
            create_paint_image(&gpu, 1, rgba, crate::vram::Kind::Composite),
            create_paint_image(&gpu, 1, Format::R16G16_SFLOAT, crate::vram::Kind::Composite),
        ];
        let (blank, blank_rg) = (blanks[0].clone(), blanks[1].clone());
        let tiny = || blank.clone();
        let tiny_rg = || blank_rg.clone();
        let placeholders = [create_layer_image(&gpu, 1, rgba), create_layer_image(&gpu, 1, Format::R16G16_SFLOAT)];

        let mut stack = Self {
            layers: Vec::new(),
            active: 0,
            resolution,
            composite_basecolor: tiny(),
            composite_roughness: tiny_rg(),
            composite_metallic: tiny_rg(),
            composite_height: tiny_rg(),
            composite_emissive: tiny(),
            composite_opacity: tiny_rg(),
            composite_normal: tiny(),
            normal_map: tiny(),
            height_scale: 4.0,
            sampler: shared.sampler,
            tiling_sampler: shared.tiling_sampler,
            white: shared.white,
            pipeline: shared.pipeline,
            mask_pipeline: shared.mask_pipeline,
            shader_masks: Vec::new(),
            matcap_masks: Vec::new(),
            normal_pipeline: shared.normal_pipeline,
            dilate_pipeline: shared.dilate_pipeline,
            dilate_rg_pipeline: shared.dilate_rg_pipeline,
            dilate_temp: tiny(),
            dilate_temp_rg: tiny_rg(),
            clip_coverage: tiny(),
            dirty: false,
            next_layer_number: 1,
            next_layer_id: 1,
            strokes_begun: 0,
            undo_stack: std::collections::VecDeque::new(),
            redo_stack: Vec::new(),
            undo_budget: UNDO_BUDGET.load(std::sync::atomic::Ordering::Relaxed),
            dilate: true,
            projection_source: None,
            allocated: false,
            placeholders,
            pending_upgrades: Vec::new(),
            blanks,
            composites_swapped: false,
            gpu,
        };
        // Everything the renderer samples starts transparent, in one submit.
        let mut builder = AutoCommandBufferBuilder::primary(
            stack.gpu.command_buffer_allocator.clone(),
            stack.gpu.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        for view in stack.composites().iter().chain(&stack.placeholders) {
            builder
                .clear_color_image(ClearColorImageInfo {
                    clear_value: ClearColorValue::Float([0.0; 4]),
                    ..ClearColorImageInfo::image(view.image().clone())
                })
                .unwrap();
        }
        submit_and_wait(&stack.gpu, builder);
        let blank = CHANNEL_FORMATS.map(|f| stack.placeholder(f));
        stack.push_blank_layer(blank);
        stack
    }

    fn composites(&self) -> [Arc<ImageView>; 8] {
        [
            self.composite_basecolor.clone(),
            self.composite_roughness.clone(),
            self.composite_metallic.clone(),
            self.composite_height.clone(),
            self.composite_emissive.clone(),
            self.composite_opacity.clone(),
            self.composite_normal.clone(),
            self.normal_map.clone(),
        ]
    }

    /// The 1x1 stand-in for a blank channel of `format` (project loading
    /// maps never-painted textures back onto it).
    pub fn placeholder(&self, format: Format) -> Arc<ImageView> {
        let i = if format == Format::R16G16B16A16_SFLOAT { 0 } else { 1 };
        self.placeholders[i].clone()
    }

    pub fn is_placeholder(&self, view: &Arc<ImageView>) -> bool {
        self.placeholders.iter().any(|p| Arc::ptr_eq(p, view))
    }

    /// Whether the full-resolution composites exist (see `new_lazy`).
    #[cfg(test)]
    pub fn is_allocated(&self) -> bool {
        self.allocated
    }

    /// Anything a composite would show: real channel textures, a mask, a fill.
    fn has_content(&self) -> bool {
        self.layers.iter().any(|l| {
            l.mask.is_some() || l.has_fill || l.channels().into_iter().any(|t| !self.is_placeholder(t))
        })
    }

    /// Which composites any material layer can put something in, in
    /// `composite_targets` order (colour, roughness, metallic, height,
    /// emissive, opacity, normal): those with a real texture in that channel
    /// on some layer. The rest stay blank 1x1 images — a set painted only in
    /// colour holds one full-size composite, not eight. Height and normal go
    /// together, since the normal map is derived from both.
    fn composite_needs(&self) -> [bool; 7] {
        let mut needs = [false; 7];
        for layer in self.layers.iter().filter(|l| l.purpose.is_none()) {
            let sources = [
                &layer.basecolor,
                &layer.roughness,
                &layer.metallic,
                &layer.height,
                &layer.emissive,
                &layer.opacity_map,
                &layer.normal,
            ];
            for (need, view) in needs.iter_mut().zip(sources) {
                *need |= !self.is_placeholder(view);
            }
        }
        let bump = needs[3] || needs[6];
        needs[3] = bump;
        needs[6] = bump;
        needs
    }

    fn is_blank(&self, view: &Arc<ImageView>) -> bool {
        self.blanks.iter().any(|b| Arc::ptr_eq(b, view))
    }

    /// Make the stack flattenable, with full-resolution composites for
    /// exactly the channels that have content (see `composite_needs`); the
    /// layers themselves are untouched. Cheap when nothing changed. True
    /// when a composite was swapped, so anything holding the old ones (the
    /// path tracer's texture table) must rebuild.
    fn allocate(&mut self) -> bool {
        let (gpu, res) = (self.gpu.clone(), self.resolution);
        let needs = self.composite_needs();
        let [blank, blank_rg] = self.blanks.clone();
        let mut swapped = false;
        let slots = [
            (&mut self.composite_basecolor, &blank, needs[0]),
            (&mut self.composite_roughness, &blank_rg, needs[1]),
            (&mut self.composite_metallic, &blank_rg, needs[2]),
            (&mut self.composite_height, &blank_rg, needs[3]),
            (&mut self.composite_emissive, &blank, needs[4]),
            (&mut self.composite_opacity, &blank_rg, needs[5]),
            (&mut self.composite_normal, &blank, needs[6]),
            (&mut self.normal_map, &blank, needs[3]),
        ];
        for (slot, blank, need) in slots {
            let is_blank = Arc::ptr_eq(slot, blank);
            if need && is_blank {
                *slot = create_paint_image(&gpu, res, blank.image().format(), crate::vram::Kind::Composite);
                swapped = true;
            } else if !need && !is_blank {
                *slot = blank.clone();
                swapped = true;
            }
        }
        if !self.allocated {
            self.dilate_temp = shared_scratch(&gpu, res, Scratch::Dilate);
            self.dilate_temp_rg = shared_scratch(&gpu, res, Scratch::DilateRg);
            self.clip_coverage = shared_scratch(&gpu, res, Scratch::ClipCoverage);
            self.allocated = true;
            swapped = true;
        }
        if swapped {
            // New composites hold garbage until the next flatten.
            self.dirty = true;
            self.composites_swapped = true;
        }
        swapped
    }

    /// The reverse of `allocate`: back to 1x1 transparent composites, for a
    /// set nobody is looking at. Layers keep their textures; the next flatten
    /// with content allocates and rebuilds the composites.
    pub fn release_composites(&mut self) {
        if !self.allocated {
            return;
        }
        let [blank, blank_rg] = self.blanks.clone();
        for (slot, blank) in [
            (&mut self.composite_basecolor, &blank),
            (&mut self.composite_roughness, &blank_rg),
            (&mut self.composite_metallic, &blank_rg),
            (&mut self.composite_height, &blank_rg),
            (&mut self.composite_emissive, &blank),
            (&mut self.composite_opacity, &blank_rg),
            (&mut self.composite_normal, &blank),
            (&mut self.normal_map, &blank),
            (&mut self.dilate_temp, &blank),
            (&mut self.dilate_temp_rg, &blank_rg),
            (&mut self.clip_coverage, &blank),
        ] {
            *slot = blank.clone();
        }
        // Rebuilt by `record_shader_masks` on the next flatten.
        self.shader_masks.clear();
        self.matcap_masks.clear();
        self.allocated = false;
        self.dirty = true;
    }

    /// Hold full-resolution composites only while the set can be seen or
    /// painted. A hidden set's composites cost ~0.25 GiB at 2048 px; on a
    /// big avatar with dozens of painted sets that filled a 16 GB card and
    /// starved the desktop (other apps could not get window buffers). True
    /// when the composites were swapped, so anything caching them (the path
    /// tracer's texture table) must rebuild.
    pub fn set_resident(&mut self, resident: bool) -> bool {
        if resident && (self.allocated || self.has_content()) {
            self.allocate();
            // Also swaps made by edits since the last call.
            std::mem::take(&mut self.composites_swapped)
        } else if !resident && self.allocated {
            self.release_composites();
            true
        } else {
            false
        }
    }

    /// VRAM this stack holds: its composites when allocated, every layer
    /// texture that left its placeholder, and masks. Textures shared with
    /// undo history count here too.
    pub fn vram_bytes(&self) -> u64 {
        let bytes = |view: &Arc<ImageView>| {
            let [w, h, _] = view.image().extent();
            w as u64 * h as u64 * view.image().format().block_size()
        };
        let composites: u64 = if self.allocated { self.composites().iter().map(bytes).sum() } else { 0 };
        let layers: u64 = self
            .layers
            .iter()
            .flat_map(|l| l.channels().into_iter().chain(l.mask.as_ref()))
            .filter(|t| !self.is_placeholder(t))
            .map(bytes)
            .sum();
        composites + layers
    }

    /// Make the stack paintable: full-resolution composites. Layer channels
    /// stay on their placeholders until something writes them (see
    /// `real_channel`). Cheap when there is nothing to do. Returns the
    /// channels swapped off placeholders since the last call: the app's undo
    /// history compares textures by identity, so it adopts them
    /// (`StackState::adopt_upgrades`) rather than seeing an edit.
    pub fn ensure_allocated(&mut self) -> Vec<Upgrade> {
        self.allocate();
        self.take_upgrades()
    }

    /// Channels that left their placeholders since the last call.
    pub fn take_upgrades(&mut self) -> Vec<Upgrade> {
        std::mem::take(&mut self.pending_upgrades)
    }

    /// Layer `layer`'s channel `c` (`CHANNEL_FORMATS` order) as a real,
    /// full-resolution texture, swapping a cleared one in for a placeholder.
    /// For writes in place: strokes and map uploads.
    fn real_channel(&mut self, layer: usize, c: usize) -> Arc<ImageView> {
        let current = self.layers[layer].channels()[c].clone();
        if !self.is_placeholder(&current) {
            return current;
        }
        let fresh = self.new_cleared_texture(CHANNEL_FORMATS[c]);
        if let Some(slot) = self.layers[layer].channels_mut().into_iter().nth(c) {
            *slot = fresh.clone();
        }
        self.pending_upgrades.push((self.layers[layer].id, c, current, fresh.clone()));
        fresh
    }

    /// Give the active layer real textures for the material channels a
    /// stroke will write (the mask is never a placeholder).
    pub fn ensure_channels(&mut self, channels: &[PaintChannel]) {
        let active = self.active;
        for c in channels {
            if let Some(index) = c.channel_index() {
                self.real_channel(active, index);
            }
        }
    }

    /// The active layer's paint-pass attachments, in pass order (colour,
    /// roughness, metallic, height, emissive, opacity); None where the layer
    /// has no texture for the channel. The stroke leaves those slots unused,
    /// so their writes are discarded.
    pub fn paint_attachments(&self) -> [Option<Arc<ImageView>>; 6] {
        let layer = self.active_layer();
        [&layer.basecolor, &layer.roughness, &layer.metallic, &layer.height, &layer.emissive, &layer.opacity_map]
            .map(|view| (!self.is_placeholder(view)).then(|| view.clone()))
    }

    /// VRAM of this stack's paint textures if it were at `resolution`:
    /// composites plus every real layer channel and mask, scaled.
    pub fn bytes_at(&self, resolution: u32) -> u64 {
        let texels = resolution as u64 * resolution as u64;
        let layers: u64 = self
            .layers
            .iter()
            .flat_map(|l| l.channels().into_iter().chain(l.mask.as_ref()))
            .filter(|t| !self.is_placeholder(t))
            .map(|t| t.image().format().block_size())
            .sum();
        texels * (COMPOSITE_TEXEL_BYTES + layers)
    }

    /// This stack at another resolution: every layer channel and mask
    /// resampled (halving steps when shrinking, so detail averages rather
    /// than aliases), every setting kept. Stroke undo history is not carried
    /// over — the app keeps the old stack in its history instead.
    pub fn resampled(&self, resolution: u32) -> LayerStack {
        let mut out = LayerStack::new_lazy(self.gpu.clone(), resolution);
        let mut builder = AutoCommandBufferBuilder::primary(
            self.gpu.command_buffer_allocator.clone(),
            self.gpu.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        let mut resample = |src: &Arc<ImageView>| record_resample(&self.gpu, &mut builder, src, resolution);
        let mut layers = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let mut copy = layer.clone();
            for (c, format) in CHANNEL_FORMATS.into_iter().enumerate() {
                let src = layer.channels()[c];
                let fresh = if self.is_placeholder(src) { out.placeholder(format) } else { resample(src) };
                if let Some(slot) = copy.channels_mut().into_iter().nth(c) {
                    *slot = fresh;
                }
            }
            copy.mask = layer.mask.as_ref().map(&mut resample);
            layers.push(copy);
        }
        submit_and_wait(&self.gpu, builder);

        out.layers = layers;
        out.active = self.active;
        out.height_scale = self.height_scale;
        out.projection_source = self.projection_source.clone();
        out.next_layer_number = self.next_layer_number;
        out.next_layer_id = self.next_layer_id;
        out.undo_budget = self.undo_budget;
        out.dilate = self.dilate;
        if out.has_content() {
            out.allocate();
        }
        out.dirty = true;
        out
    }

    /// `view` (a mask made at another size, e.g. from bakes) at this stack's
    /// resolution, so strokes can paint into it. Returned as is when it
    /// already matches.
    pub fn fit_to_resolution(&self, view: Arc<ImageView>) -> Arc<ImageView> {
        if view.image().extent()[0] == self.resolution {
            return view;
        }
        let mut builder = AutoCommandBufferBuilder::primary(
            self.gpu.command_buffer_allocator.clone(),
            self.gpu.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        let out = record_resample(&self.gpu, &mut builder, &view, self.resolution);
        submit_and_wait(&self.gpu, builder);
        out
    }

    /// Hand the stack the baked mesh maps projected fills place themselves
    /// by. `centre` and `size` (longest bounding-box side) are the object's
    /// world-space bounds, which the projections are framed to.
    pub fn set_projection_source(
        &mut self,
        position: Arc<ImageView>,
        normal: Arc<ImageView>,
        centre: [f32; 3],
        size: f32,
    ) {
        self.projection_source = Some(ProjectionSource {
            position,
            normal,
            frame: [centre[0], centre[1], centre[2], size.max(1e-6)],
        });
        self.dirty = true;
    }

    pub fn has_projection_source(&self) -> bool {
        self.projection_source.is_some()
    }

    /// The object's world bounds the projections are framed to: centre and
    /// longest side.
    pub fn projection_frame(&self) -> Option<([f32; 3], f32)> {
        self.projection_source
            .as_ref()
            .map(|s| ([s.frame[0], s.frame[1], s.frame[2]], s.frame[3]))
    }

    /// A blank layer: every channel on its placeholder until painted.
    pub fn add_layer(&mut self) {
        self.ensure_allocated();
        let channels = CHANNEL_FORMATS.map(|f| self.placeholder(f));
        self.push_blank_layer(channels);
    }

    fn push_blank_layer(&mut self, [basecolor, roughness, metallic, normal, height, emissive, opacity_map]: [Arc<ImageView>; 7]) {
        let layer = Layer {
            id: self.take_layer_id(),
            name: format!("Layer {}", self.next_layer_number),
            visible: true,
            opacity: 1.0,
            blend: BlendMode::Normal,
            basecolor,
            roughness,
            metallic,
            normal,
            height,
            emissive,
            opacity_map,
            mask: None,
            mask_kind: None,
            mask_enabled: true,
            clip: false,
            uv_scale: [1.0, 1.0],
            uv_offset: [0.0, 0.0],
            uv_rotation: 0.0,
            projection: Projection::Uv,
            projection_axis: Axis::Y,
            triplanar_sharpness: 4.0,
            projection_offset: [0.0; 3],
            projection_rotation: [0.0; 3],
            projection_scale: 1.0,
            projection_repeat: true,
            has_fill: false,
            purpose: None,
            matcap: None,
            swatch: None,
            used: 0,
        };
        self.next_layer_number += 1;
        self.layers.push(layer);
        self.active = self.layers.len() - 1;
        self.dirty = true;
    }

    /// Add a shader-mask layer of `kind` above the active stack and select it.
    pub fn add_shader_mask_layer(&mut self, kind: ShaderMask) {
        self.ensure_allocated();
        self.add_layer();
        let layer = self.layers.last_mut().unwrap();
        layer.purpose = Some(kind);
        if kind == ShaderMask::Matcap {
            layer.name = "Matcap".into();
            layer.matcap = Some(crate::matcap::Matcap::default());
        } else {
            layer.name = kind.label().to_string();
        }
        self.dirty = true;
    }

    /// Flattened texture for one shader mask: greyscale in R (composited over
    /// black, so it is the export value directly). None until a layer of that
    /// kind exists and a composite has run. Matcap layers each have their
    /// own mask; for `Matcap` this is the active layer's, else the topmost.
    pub fn shader_mask(&self, kind: ShaderMask) -> Option<&Arc<ImageView>> {
        if kind == ShaderMask::Matcap {
            let layer = Some(self.active_layer())
                .filter(|l| l.is_matcap())
                .or_else(|| self.layers.iter().rev().find(|l| l.is_matcap()))?;
            return self.matcap_mask(layer.id);
        }
        self.shader_masks.iter().find(|(k, _)| *k == kind).map(|(_, v)| v)
    }

    /// The flattened mask of the matcap layer with this id.
    pub fn matcap_mask(&self, id: u64) -> Option<&Arc<ImageView>> {
        self.layer_index(id)?;
        self.matcap_masks.iter().find(|(i, _)| *i == id).map(|(_, v)| v)
    }

    /// Visible matcap layers that have a matcap and a flattened mask,
    /// bottom to top — what the viewport draws.
    pub fn matcap_layers(&self) -> Vec<(&crate::matcap::Matcap, &Arc<ImageView>)> {
        self.layers
            .iter()
            .filter(|l| l.visible && l.is_matcap())
            .filter_map(|l| Some((l.matcap.as_ref().filter(|m| m.active())?, self.matcap_mask(l.id)?)))
            .collect()
    }

    /// Shader-mask kinds that have at least one layer, in `ShaderMask::ALL` order.
    pub fn shader_mask_kinds(&self) -> Vec<ShaderMask> {
        ShaderMask::ALL
            .into_iter()
            .filter(|k| self.layers.iter().any(|l| l.purpose == Some(*k)))
            .collect()
    }

    pub fn remove_layer(&mut self, index: usize) {
        if self.layers.len() <= 1 || index >= self.layers.len() {
            return;
        }
        self.layers.remove(index);
        self.active = self.active.min(self.layers.len() - 1);
        // Stroke history stays: entries name layers by id, and the app's
        // undo restores the deleted layer before reaching its strokes.
        self.dirty = true;
    }

    /// Move a layer to a new position in the stack (0 = bottom). The active
    /// selection and the undo history follow their layers rather than their
    /// old slots, so reordering does not silently retarget a pending undo.
    pub fn move_layer(&mut self, from: usize, to: usize) {
        let len = self.layers.len();
        if from == to || from >= len || to >= len {
            return;
        }
        let layer = self.layers.remove(from);
        self.layers.insert(to, layer);

        // Everything between the old and new slot shifts one step the other way.
        let remap = |i: usize| -> usize {
            if i == from {
                to
            } else if from < to && i > from && i <= to {
                i - 1
            } else if to < from && i >= to && i < from {
                i + 1
            } else {
                i
            }
        };
        // History entries name layers by id, so only the selection needs
        // to follow the move.
        self.active = remap(self.active);
        self.dirty = true;
    }

    /// Copy layer `index` — every channel, its mask and its settings — into
    /// a new layer directly above it, and select the copy.
    pub fn duplicate_layer(&mut self, index: usize) {
        self.ensure_allocated();
        if index >= self.layers.len() {
            return;
        }
        let copy_of = |stack: &Self, view: &Arc<ImageView>| -> Arc<ImageView> {
            if stack.is_placeholder(view) {
                return view.clone();
            }
            let dst = stack.new_cleared_texture(view.image().format());
            let mut builder = AutoCommandBufferBuilder::primary(
                stack.gpu.command_buffer_allocator.clone(),
                stack.gpu.queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();
            builder
                .copy_image(vulkano::command_buffer::CopyImageInfo::images(
                    view.image().clone(),
                    dst.image().clone(),
                ))
                .unwrap();
            submit_and_wait(&stack.gpu, builder);
            dst
        };
        let id = self.take_layer_id();
        let src = &self.layers[index];
        let layer = Layer {
            id,
            name: format!("{} copy", src.name),
            visible: src.visible,
            opacity: src.opacity,
            blend: src.blend,
            basecolor: copy_of(self, &src.basecolor),
            roughness: copy_of(self, &src.roughness),
            metallic: copy_of(self, &src.metallic),
            normal: copy_of(self, &src.normal),
            height: copy_of(self, &src.height),
            emissive: copy_of(self, &src.emissive),
            opacity_map: copy_of(self, &src.opacity_map),
            mask: src.mask.as_ref().map(|m| copy_of(self, m)),
            mask_kind: src.mask_kind,
            mask_enabled: src.mask_enabled,
            clip: src.clip,
            uv_scale: src.uv_scale,
            uv_offset: src.uv_offset,
            uv_rotation: src.uv_rotation,
            projection: src.projection,
            projection_axis: src.projection_axis,
            triplanar_sharpness: src.triplanar_sharpness,
            projection_offset: src.projection_offset,
            projection_rotation: src.projection_rotation,
            projection_scale: src.projection_scale,
            projection_repeat: src.projection_repeat,
            has_fill: src.has_fill,
            purpose: src.purpose,
            matcap: src.matcap.clone(),
            swatch: src.swatch,
            used: src.used,
        };
        let at = index + 1;
        self.layers.insert(at, layer);
        self.active = at;
        self.dirty = true;
    }

    fn take_layer_id(&mut self) -> u64 {
        let id = self.next_layer_id;
        self.next_layer_id += 1;
        id
    }

    /// Slot of the layer with this id, if it is still in the stack.
    fn layer_index(&self, id: u64) -> Option<usize> {
        self.layers.iter().position(|l| l.id == id)
    }

    /// Strokes begun so far (see `strokes_begun`).
    pub fn strokes_begun(&self) -> u64 {
        self.strokes_begun
    }

    /// Forget undone strokes: a new edit elsewhere in the app's history
    /// makes them unreachable.
    pub fn clear_redo(&mut self) {
        self.redo_stack.clear();
    }

    /// The stack's document state: every layer (textures shared, not
    /// copied), the selection, bump strength and the projection frame.
    pub fn state(&self) -> StackState {
        StackState {
            layers: self.layers.clone(),
            active: self.active,
            height_scale: self.height_scale,
            projection_source: self.projection_source.clone(),
            next_layer_number: self.next_layer_number,
        }
    }

    /// Put a saved state back (undo / redo).
    pub fn restore_state(&mut self, state: StackState) {
        self.layers = state.layers;
        self.active = state.active.min(self.layers.len().saturating_sub(1));
        self.height_scale = state.height_scale;
        self.projection_source = state.projection_source;
        self.next_layer_number = state.next_layer_number;
        self.dirty = true;
    }

    pub fn active_layer(&self) -> &Layer {
        &self.layers[self.active]
    }

    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Whether the composites are stale (used to reset path-trace accumulation).
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Which channel textures of the active layer a stroke will touch.
    fn channel_textures(&self, layer: usize, channels: &[PaintChannel]) -> Vec<(PaintChannel, Arc<ImageView>)> {
        let layer = &self.layers[layer];
        channels
            .iter()
            .filter_map(|&c| {
                let tex = match c {
                    PaintChannel::BaseColor => &layer.basecolor,
                    PaintChannel::Roughness => &layer.roughness,
                    PaintChannel::Metallic => &layer.metallic,
                    PaintChannel::Height => &layer.height,
                    PaintChannel::Emissive => &layer.emissive,
                    PaintChannel::Opacity => &layer.opacity_map,
                    // Only meaningful once the layer actually has a mask.
                    PaintChannel::Mask => layer.mask.as_ref()?,
                };
                Some((c, tex.clone()))
            })
            .collect()
    }

    /// Stage a set of layer textures into host memory.
    fn snapshot_maps(
        &self,
        maps: &[(PaintChannel, Arc<ImageView>)],
    ) -> (Vec<(PaintChannel, Subbuffer<[u8]>)>, u64) {
        let mut snapshots = Vec::with_capacity(maps.len());
        let mut total = 0u64;
        let mut builder = AutoCommandBufferBuilder::primary(
            self.gpu.command_buffer_allocator.clone(),
            self.gpu.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        for (channel, src) in maps {
            let len = self.resolution as u64
                * self.resolution as u64
                * src.image().format().block_size();
            let buffer = Buffer::new_slice::<u8>(
                self.gpu.memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_HOST
                        | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                    ..Default::default()
                },
                len,
            )
            .unwrap();
            builder
                .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
                    src.image().clone(),
                    buffer.clone(),
                ))
                .unwrap();
            total += len;
            snapshots.push((*channel, buffer));
        }
        submit_and_wait(&self.gpu, builder);
        (snapshots, total)
    }

    fn restore_maps(&mut self, layer: usize, maps: &[(PaintChannel, Subbuffer<[u8]>)]) {
        let channels: Vec<PaintChannel> = maps.iter().map(|(c, _)| *c).collect();
        // A snapshot is full-size; a channel since emptied back onto its
        // placeholder (a merge with nothing in it) needs a texture again.
        for c in channels.iter().filter_map(|c| c.channel_index()) {
            self.real_channel(layer, c);
        }
        let targets = self.channel_textures(layer, &channels);
        if targets.len() != maps.len() {
            // A channel vanished under us (a mask was removed); the entry is
            // no longer safe to apply.
            return;
        }
        let mut builder = AutoCommandBufferBuilder::primary(
            self.gpu.command_buffer_allocator.clone(),
            self.gpu.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        for ((_, src), (_, dst)) in maps.iter().zip(&targets) {
            builder
                .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
                    src.clone(),
                    dst.image().clone(),
                ))
                .unwrap();
        }
        submit_and_wait(&self.gpu, builder);
    }

    /// Host memory the history currently holds.
    pub fn history_bytes(&self) -> u64 {
        self.undo_stack.iter().chain(self.redo_stack.iter()).map(|e| e.bytes).sum()
    }

    /// Number of strokes that can still be undone.
    pub fn undo_depth(&self) -> usize {
        self.undo_stack.len()
    }

    /// Drop the oldest steps until the history fits its budget. One step is
    /// always kept, however large the textures are.
    fn trim_history(&mut self) {
        while self.history_bytes() > self.undo_budget && self.undo_stack.len() > 1 {
            self.undo_stack.pop_front();
        }
    }

    /// Capture the pre-stroke state of the channels a stroke will touch.
    /// Pushes onto the undo history (clearing any redo branch).
    pub fn begin_stroke(&mut self, channels: &[PaintChannel]) {
        self.ensure_channels(channels);
        self.allocate();
        if channels.is_empty() {
            return;
        }
        let maps = self.channel_textures(self.active, channels);
        if maps.is_empty() {
            return;
        }
        let (snapshot, bytes) = self.snapshot_maps(&maps);
        let layer = self.layers[self.active].id;
        self.undo_stack.push_back(HistoryEntry { layer, maps: snapshot, bytes });
        self.redo_stack.clear();
        self.strokes_begun += 1;
        self.trim_history();
    }

    /// Step back one stroke. Returns false when history is empty.
    pub fn undo_stroke(&mut self) -> bool {
        let Some(entry) = self.undo_stack.pop_back() else {
            return false;
        };
        let Some(index) = self.layer_index(entry.layer) else {
            return false;
        };
        // Capture the current state so the undo can be redone.
        let channels: Vec<PaintChannel> = entry.maps.iter().map(|(c, _)| *c).collect();
        let current = self.channel_textures(index, &channels);
        let (redo_snapshot, bytes) = self.snapshot_maps(&current);
        self.redo_stack.push(HistoryEntry { layer: entry.layer, maps: redo_snapshot, bytes });

        self.restore_maps(index, &entry.maps);
        self.dirty = true;
        true
    }

    /// Re-apply the most recently undone stroke.
    pub fn redo_stroke(&mut self) -> bool {
        let Some(entry) = self.redo_stack.pop() else {
            return false;
        };
        let Some(index) = self.layer_index(entry.layer) else {
            return false;
        };
        let channels: Vec<PaintChannel> = entry.maps.iter().map(|(c, _)| *c).collect();
        let current = self.channel_textures(index, &channels);
        let (undo_snapshot, bytes) = self.snapshot_maps(&current);
        self.undo_stack.push_back(HistoryEntry { layer: entry.layer, maps: undo_snapshot, bytes });

        self.restore_maps(index, &entry.maps);
        self.dirty = true;
        true
    }

    /// Flood-fill channels of the active layer with uniform values
    /// (premultiplied, full coverage). The basis of "fill layers": combined
    /// with a mask, this gives smart-material-style wear/dirt layers.
    pub fn fill_active(
        &mut self,
        color: Option<[f32; 3]>,
        roughness: Option<f32>,
        metallic: Option<f32>,
    ) {
        self.ensure_allocated();
        let layer = &mut self.layers[self.active];
        if color.is_some() {
            layer.swatch = color;
        }
        for (on, ch) in [
            (color.is_some(), PaintChannel::BaseColor),
            (roughness.is_some(), PaintChannel::Roughness),
            (metallic.is_some(), PaintChannel::Metallic),
        ] {
            if on {
                layer.used |= ch.bit();
            }
        }
        // Fresh textures, never in place: undo snapshots still hold the old.
        let active = self.active;
        if color.is_some() {
            self.layers[active].basecolor = self.new_cleared_texture(Format::R16G16B16A16_SFLOAT);
        }
        if roughness.is_some() {
            self.layers[active].roughness = self.new_cleared_texture(Format::R16G16_SFLOAT);
        }
        if metallic.is_some() {
            self.layers[active].metallic = self.new_cleared_texture(Format::R16G16_SFLOAT);
        }
        let layer = &self.layers[active];
        let mut builder = AutoCommandBufferBuilder::primary(
            self.gpu.command_buffer_allocator.clone(),
            self.gpu.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        if let Some(c) = color {
            builder
                .clear_color_image(ClearColorImageInfo {
                    clear_value: ClearColorValue::Float([c[0], c[1], c[2], 1.0]),
                    ..ClearColorImageInfo::image(layer.basecolor.image().clone())
                })
                .unwrap();
        }
        if let Some(r) = roughness {
            builder
                .clear_color_image(ClearColorImageInfo {
                    clear_value: ClearColorValue::Float([r, 1.0, 0.0, 0.0]),
                    ..ClearColorImageInfo::image(layer.roughness.image().clone())
                })
                .unwrap();
        }
        if let Some(m) = metallic {
            builder
                .clear_color_image(ClearColorImageInfo {
                    clear_value: ClearColorValue::Float([m, 1.0, 0.0, 0.0]),
                    ..ClearColorImageInfo::image(layer.metallic.image().clone())
                })
                .unwrap();
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
        self.dirty = true;
    }

    /// Give a layer a paintable mask, flooded with `value` (0 hides the layer
    /// entirely — the starting point for painting a material in by hand).
    pub fn add_mask(&mut self, index: usize, value: f32) {
        self.ensure_allocated();
        if index >= self.layers.len() {
            return;
        }
        // RGBA16F, matching what the smart-mask baker produces, so generated
        // and hand-painted masks are interchangeable and can be layered.
        let mask = self.new_cleared_texture(Format::R16G16B16A16_SFLOAT);
        self.layers[index].mask = Some(mask);
        self.layers[index].mask_kind = None;
        self.fill_mask(index, value);
    }

    pub fn remove_mask(&mut self, index: usize) {
        if let Some(layer) = self.layers.get_mut(index) {
            layer.mask = None;
            layer.mask_kind = None;
            self.dirty = true;
            // Mask snapshots in the history now point at a dropped image.
            self.undo_stack.retain(|e| !e.maps.iter().any(|(c, _)| *c == PaintChannel::Mask));
            self.redo_stack.retain(|e| !e.maps.iter().any(|(c, _)| *c == PaintChannel::Mask));
        }
    }

    /// Flood a mask to a uniform value. Stored premultiplied as (value·α, α)
    /// like the other scalar channels; the compositor reads `.r`.
    pub fn fill_mask(&mut self, index: usize, value: f32) {
        self.ensure_allocated();
        if self.layers.get(index).is_none_or(|l| l.mask.is_none()) {
            return;
        }
        // A fresh mask, so undo snapshots keep the one being replaced.
        let mask = self.new_cleared_texture(Format::R16G16B16A16_SFLOAT);
        self.layers[index].mask = Some(mask.clone());
        let mut builder = AutoCommandBufferBuilder::primary(
            self.gpu.command_buffer_allocator.clone(),
            self.gpu.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        builder
            .clear_color_image(ClearColorImageInfo {
                clear_value: ClearColorValue::Float([value, 1.0, 0.0, 0.0]),
                ..ClearColorImageInfo::image(mask.image().clone())
            })
            .unwrap();
        submit_and_wait(&self.gpu, builder);
        self.dirty = true;
    }

    /// Flip a mask, turning "painted in" into "painted out". Done over a host
    /// round-trip rather than a compute pass: it is a one-off UI action, and
    /// this way it works on baked smart masks as well as painted ones.
    pub fn invert_mask(&mut self, index: usize) {
        self.ensure_allocated();
        let Some(mask) = self.layers.get(index).and_then(|l| l.mask.clone()) else {
            return;
        };
        let texels = (self.resolution * self.resolution) as usize * 4;
        let readback = Buffer::from_iter(
            self.gpu.memory_allocator.clone(),
            BufferCreateInfo { usage: BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC, ..Default::default() },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            (0..texels).map(|_| f16::ZERO),
        )
        .unwrap();

        let mut builder = AutoCommandBufferBuilder::primary(
            self.gpu.command_buffer_allocator.clone(),
            self.gpu.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        builder
            .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
                mask.image().clone(),
                readback.clone(),
            ))
            .unwrap();
        submit_and_wait(&self.gpu, builder);

        {
            let mut data = readback.write().unwrap();
            for texel in data.chunks_exact_mut(4) {
                let alpha = texel[1].to_f32();
                let value = if alpha > 1e-5 { texel[0].to_f32() / alpha } else { 0.0 };
                texel[0] = f16::from_f32(1.0 - value.clamp(0.0, 1.0));
                texel[1] = f16::ONE;
            }
        }

        let mut builder = AutoCommandBufferBuilder::primary(
            self.gpu.command_buffer_allocator.clone(),
            self.gpu.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        // Into a fresh mask, so undo snapshots keep the original.
        let inverted = self.new_cleared_texture(Format::R16G16B16A16_SFLOAT);
        builder
            .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
                readback,
                inverted.image().clone(),
            ))
            .unwrap();
        self.layers[index].mask = Some(inverted);
        submit_and_wait(&self.gpu, builder);
        self.dirty = true;
    }

    /// Replace a layer's channel textures with the maps of a library material,
    /// resampled to the layer resolution. This is what makes "apply material"
    /// produce a fill layer rather than overwriting the object's material.
    ///
    /// Returns the slots that were actually written. `MapSlot::Ao` is not
    /// among them: there is no per-layer occlusion channel, so a material's AO
    /// is left to the baked mesh AO.
    pub fn fill_layer_from_material(
        &mut self,
        index: usize,
        maps: &[(MapSlot, ImageData)],
    ) -> Vec<MapSlot> {
        if index >= self.layers.len() {
            return Vec::new();
        }
        let res = self.resolution;
        let mut written = Vec::new();
        for (slot, image) in maps {
            let pixels = resample_box(image, res);
            if *slot == MapSlot::BaseColor {
                self.layers[index].swatch = average_colour(&pixels);
            }
            let layer = &self.layers[index];
            let (target, data): (Arc<ImageView>, Vec<f16>) = match slot {
                MapSlot::BaseColor => {
                    let mut out = Vec::with_capacity(pixels.len() * 4);
                    for p in &pixels {
                        // Base colour is authored in sRGB; layers are linear.
                        out.push(f16::from_f32(srgb_to_linear(p[0])));
                        out.push(f16::from_f32(srgb_to_linear(p[1])));
                        out.push(f16::from_f32(srgb_to_linear(p[2])));
                        out.push(f16::ONE);
                    }
                    (layer.basecolor.clone(), out)
                }
                MapSlot::Roughness => (layer.roughness.clone(), scalar_channel(&pixels, |v| v)),
                MapSlot::Metallic => (layer.metallic.clone(), scalar_channel(&pixels, |v| v)),
                // Displacement is stored 0..1 with mid-grey meaning "no
                // offset", while layer height is signed around zero.
                MapSlot::Height => (layer.height.clone(), scalar_channel(&pixels, |v| v - 0.5)),
                MapSlot::Normal => {
                    let mut out = Vec::with_capacity(pixels.len() * 4);
                    for p in &pixels {
                        out.push(f16::from_f32(p[0] * 2.0 - 1.0));
                        out.push(f16::from_f32(p[1] * 2.0 - 1.0));
                        out.push(f16::from_f32(p[2] * 2.0 - 1.0));
                        out.push(f16::ONE);
                    }
                    (layer.normal.clone(), out)
                }
                // Cutout coverage drives the layer's opacity channel, which is
                // what makes net and lace fill as open mesh rather than a solid
                // sheet with a pattern printed on it.
                MapSlot::Opacity => {
                    (layer.opacity_map.clone(), scalar_channel(&pixels, |v| v))
                }
                MapSlot::Emission => {
                    let mut out = Vec::with_capacity(pixels.len() * 4);
                    for p in &pixels {
                        out.extend([srgb_to_linear(p[0]), srgb_to_linear(p[1]), srgb_to_linear(p[2]), 1.0].map(f16::from_f32));
                    }
                    (layer.emissive.clone(), out)
                }
                // A layer has no separate strength: the mask would have to
                // multiply the emission, so it is left to the material.
                MapSlot::Ao | MapSlot::EmissionMask => continue,
            };
            // A fresh texture in the old one's place, so undo snapshots keep
            // the previous contents. Matched by slot, not by identity: blank
            // channels of a format all share one placeholder.
            let fresh = self.new_cleared_texture(target.image().format());
            let layer = &mut self.layers[index];
            let field = match slot {
                MapSlot::BaseColor => &mut layer.basecolor,
                MapSlot::Roughness => &mut layer.roughness,
                MapSlot::Metallic => &mut layer.metallic,
                MapSlot::Height => &mut layer.height,
                MapSlot::Normal => &mut layer.normal,
                MapSlot::Opacity => &mut layer.opacity_map,
                MapSlot::Emission => &mut layer.emissive,
                MapSlot::Ao | MapSlot::EmissionMask => unreachable!(),
            };
            *field = fresh.clone();
            self.upload_texels(&fresh, data);
            written.push(*slot);
            self.layers[index].used |= match slot {
                MapSlot::BaseColor => PaintChannel::BaseColor.bit(),
                MapSlot::Roughness => PaintChannel::Roughness.bit(),
                MapSlot::Metallic => PaintChannel::Metallic.bit(),
                MapSlot::Height => PaintChannel::Height.bit(),
                MapSlot::Normal => NORMAL_BIT,
                MapSlot::Opacity => PaintChannel::Opacity.bit(),
                MapSlot::Emission => PaintChannel::Emissive.bit(),
                MapSlot::Ao | MapSlot::EmissionMask => 0,
            };
        }
        self.layers[index].has_fill = true;
        self.dirty = true;
        written
    }

    /// Fill a layer's base colour with one image, as a placeable fill: the
    /// tiling and projection controls apply, so it can be wrapped around the
    /// model. Unlike a library material the image's alpha is kept as
    /// coverage, so a transparent PNG wraps on as a decal over what is below.
    pub fn fill_layer_from_image(&mut self, index: usize, image: &ImageData) {
        self.ensure_allocated();
        if index >= self.layers.len() {
            return;
        }
        let pixels = resample_box(image, self.resolution);
        let mut out = Vec::with_capacity(pixels.len() * 4);
        for p in &pixels {
            let a = p[3];
            out.extend(
                [srgb_to_linear(p[0]) * a, srgb_to_linear(p[1]) * a, srgb_to_linear(p[2]) * a, a]
                    .map(f16::from_f32),
            );
        }
        // A fresh texture, so undo snapshots keep the previous colour.
        let target = self.new_cleared_texture(Format::R16G16B16A16_SFLOAT);
        self.layers[index].basecolor = target.clone();
        self.upload_texels(&target, out);
        let layer = &mut self.layers[index];
        layer.swatch = average_colour(&pixels).or(layer.swatch);
        layer.used |= PaintChannel::BaseColor.bit();
        layer.has_fill = true;
        self.dirty = true;
    }

    /// Push a new layer holding one imported texture map, laid out in the
    /// mesh's own UVs (no tiling). The image's alpha becomes the layer's
    /// coverage, so a ZestPaint export imports back exactly where it was
    /// painted and leaves everything else to the layers underneath. Existing
    /// layers are never touched. Returns false for maps with no layer channel.
    pub fn import_map_layer(&mut self, name: String, kind: MapKind, image: &ImageData) -> bool {
        self.ensure_allocated();
        if !kind.importable() {
            return false;
        }
        self.add_layer();
        let index = self.active;
        self.layers[index].name = name;
        self.write_map(index, kind, image, [1.0; 3]);
        true
    }

    /// Whether the stack is still the untouched default: one empty "Layer 1".
    pub fn is_pristine(&self) -> bool {
        matches!(self.layers.as_slice(), [l] if l.name == "Layer 1" && l.used == 0 && !l.has_fill && l.purpose.is_none() && l.mask.is_none())
    }

    /// Turn an untouched stack into "Base material" — the material's own
    /// maps (each `(kind, image, factor)`; the factor multiplies the map, as
    /// the material's colour and sliders do) — with an empty layer above it
    /// to paint on, so painting never overwrites the original textures.
    /// False, and nothing changes, unless the stack is pristine and there is
    /// at least one map.
    pub fn seed_from_material(&mut self, maps: &[(MapKind, &ImageData, [f32; 3])]) -> bool {
        let maps: Vec<_> = maps.iter().filter(|(kind, _, _)| kind.importable()).collect();
        if maps.is_empty() || !self.is_pristine() {
            return false;
        }
        self.ensure_allocated();
        self.layers[0].name = "Base material".into();
        for (kind, image, factor) in maps {
            self.write_map(0, *kind, image, *factor);
        }
        self.add_layer();
        self.dirty = true;
        true
    }

    /// Write one texture map into layer `index`'s channel for it, scaled by
    /// `factor` (per RGB channel; scalar maps use the first).
    fn write_map(&mut self, index: usize, kind: MapKind, image: &ImageData, factor: [f32; 3]) {
        let pixels = resample_box(image, self.resolution);
        let [fr, fg, fb] = factor;
        let rgb = |decode: fn(f32) -> f32| {
            let mut out = Vec::with_capacity(pixels.len() * 4);
            for p in &pixels {
                let a = p[3];
                out.extend([decode(p[0]) * fr * a, decode(p[1]) * fg * a, decode(p[2]) * fb * a, a].map(f16::from_f32));
            }
            out
        };
        let scalar = |map: fn(f32) -> f32| {
            let mut out = Vec::with_capacity(pixels.len() * 2);
            for p in &pixels {
                out.extend([map(p[0]) * fr * p[3], p[3]].map(f16::from_f32));
            }
            out
        };
        if kind == MapKind::BaseColor {
            self.layers[index].swatch = average_colour(&pixels).map(|[r, g, b]| [r * fr, g * fg, b * fb]);
        }
        self.layers[index].used |= match kind {
            MapKind::BaseColor => PaintChannel::BaseColor.bit(),
            MapKind::Emissive => PaintChannel::Emissive.bit(),
            MapKind::Normal => NORMAL_BIT,
            MapKind::Roughness => PaintChannel::Roughness.bit(),
            MapKind::Metallic => PaintChannel::Metallic.bit(),
            MapKind::Height => PaintChannel::Height.bit(),
            MapKind::Opacity => PaintChannel::Opacity.bit(),
            _ => 0,
        };
        // `CHANNEL_FORMATS` order: basecolor, roughness, metallic, normal,
        // height, emissive, opacity.
        let (channel, data) = match kind {
            MapKind::BaseColor => (0, rgb(srgb_to_linear)),
            MapKind::Emissive => (5, rgb(srgb_to_linear)),
            MapKind::Normal => (3, rgb(|v| v * 2.0 - 1.0)),
            MapKind::Roughness => (1, scalar(|v| v)),
            MapKind::Metallic => (2, scalar(|v| v)),
            // Exported height is 0..1 with mid-grey neutral; layers are signed.
            MapKind::Height => (4, scalar(|v| (v - 0.5) * 2.0)),
            MapKind::Opacity => (6, scalar(|v| v)),
            MapKind::Ao => unreachable!(),
        };
        let target = self.real_channel(index, channel);
        self.upload_texels(&target, data);
        self.dirty = true;
    }

    /// Copy host-side half-float texels into a layer texture.
    fn upload_texels(&self, target: &Arc<ImageView>, data: Vec<f16>) {
        let buffer = Buffer::from_iter(
            self.gpu.memory_allocator.clone(),
            BufferCreateInfo { usage: BufferUsage::TRANSFER_SRC, ..Default::default() },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            data,
        )
        .unwrap();
        let mut builder = AutoCommandBufferBuilder::primary(
            self.gpu.command_buffer_allocator.clone(),
            self.gpu.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        builder
            .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
                buffer,
                target.image().clone(),
            ))
            .unwrap();
        submit_and_wait(&self.gpu, builder);
    }

    /// Give layer `index` a mask taken from `source` (adding one if it had
    /// none, replacing it if it had). Always a copy: painting the new mask
    /// leaves the source alone. False when the source has nothing to give.
    pub fn mask_from(&mut self, index: usize, source: MaskSource) -> bool {
        self.ensure_allocated();
        if index >= self.layers.len() {
            return false;
        }
        let values: Vec<f32> = match source {
            MaskSource::LayerMask(from) => {
                let Some(mask) = self.layers.get(from).and_then(|l| l.mask.clone()) else {
                    return false;
                };
                self.read_texels(&mask).into_iter().map(|t| t[0]).collect()
            }
            MaskSource::LayerCoverage(from) => {
                let Some(layer) = self.layers.get(from) else { return false };
                // Coverage lives in alpha: .a of the colour channels, .g of
                // the scalar ones. The mask is wherever any channel has paint.
                let views = [
                    &layer.basecolor,
                    &layer.roughness,
                    &layer.metallic,
                    &layer.normal,
                    &layer.height,
                    &layer.emissive,
                    &layer.opacity_map,
                ]
                .map(Arc::clone);
                let mut cover = vec![0.0f32; (self.resolution * self.resolution) as usize];
                for view in views {
                    let alpha = if view.image().format() == Format::R16G16_SFLOAT { 1 } else { 3 };
                    for (c, t) in cover.iter_mut().zip(self.read_texels(&view)) {
                        *c = c.max(t[alpha]);
                    }
                }
                cover
            }
            MaskSource::ShaderMask(kind) => {
                self.composite_now();
                let Some(view) = self.shader_mask(kind).cloned() else { return false };
                self.read_texels(&view).into_iter().map(|t| t[0]).collect()
            }
        };
        let texels = values
            .into_iter()
            .flat_map(|v| [v.clamp(0.0, 1.0), 1.0, 0.0, 0.0].map(f16::from_f32))
            .collect();
        // A fresh mask, so undo snapshots keep the one being replaced.
        let mask = self.new_cleared_texture(Format::R16G16B16A16_SFLOAT);
        self.upload_texels(&mask, texels);
        self.layers[index].mask = Some(mask);
        self.layers[index].mask_kind = None;
        self.dirty = true;
        true
    }

    /// Read a half-float texture back as f32 quads; two-channel (RG16F)
    /// textures come back as (r, g, 0, 0).
    fn read_texels(&self, view: &Arc<ImageView>) -> Vec<[f32; 4]> {
        let components = if view.image().format() == Format::R16G16_SFLOAT { 2 } else { 4 };
        let texels = (self.resolution * self.resolution) as usize * components;
        let readback = Buffer::from_iter(
            self.gpu.memory_allocator.clone(),
            BufferCreateInfo { usage: BufferUsage::TRANSFER_DST, ..Default::default() },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            (0..texels).map(|_| f16::ZERO),
        )
        .unwrap();
        let mut builder = AutoCommandBufferBuilder::primary(
            self.gpu.command_buffer_allocator.clone(),
            self.gpu.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        builder
            .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
                view.image().clone(),
                readback.clone(),
            ))
            .unwrap();
        submit_and_wait(&self.gpu, builder);
        let data = readback.read().unwrap();
        data.chunks_exact(components)
            .map(|c| match c {
                [r, g, b, a] => [r.to_f32(), g.to_f32(), b.to_f32(), a.to_f32()],
                [r, g] => [r.to_f32(), g.to_f32(), 0.0, 0.0],
                _ => unreachable!(),
            })
            .collect()
    }

    /// Bake layer `index`'s mask into its channels (GIMP's Apply mask): the
    /// layer looks the same and no longer has a mask. Its placement (tiling,
    /// projection) is baked in too, since the mask lives in plain UV space —
    /// afterwards it is an ordinary painted layer.
    pub fn apply_mask(&mut self, index: usize) -> bool {
        self.ensure_allocated();
        let Some(layer) = self.layers.get(index).cloned() else { return false };
        if layer.mask.is_none() {
            return false;
        }
        let baked = self.bake(&[(&layer, 1.0, BlendMode::Normal, false)]);
        let layer = &mut self.layers[index];
        layer.set_channels(baked);
        layer.mask = None;
        layer.mask_kind = None;
        layer.mask_enabled = true;
        layer.reset_placement();
        self.dirty = true;
        true
    }

    /// Why layer `index` can't merge into the one below, if it can't.
    pub fn merge_down_blocker(&self, index: usize) -> Option<&'static str> {
        if index == 0 || index >= self.layers.len() {
            return Some("There is no layer below to merge into");
        }
        let (lower, upper) = (&self.layers[index - 1], &self.layers[index]);
        if lower.purpose != upper.purpose {
            return Some("Shader-mask layers only merge with layers of the same mask");
        }
        if lower.is_matcap() {
            return Some("Matcap layers each keep their own matcap; they don't merge");
        }
        if lower.clip && !upper.clip {
            return Some("The layer below is clipped and this one isn't; merging would change what shows");
        }
        None
    }

    /// Merge layer `index` into the layer below it (GIMP's Merge down): the
    /// two become one layer that looks the same. Opacities, masks, clipping
    /// and placement are baked in; the result keeps the lower layer's name
    /// and blend mode.
    pub fn merge_down(&mut self, index: usize) -> bool {
        self.ensure_allocated();
        if self.merge_down_blocker(index).is_some() {
            return false;
        }
        let (lower, upper) = (self.layers[index - 1].clone(), self.layers[index].clone());
        // Both clipped: the upper one already shows only inside the shared
        // base, so it composites onto the lower one without its own clip.
        let upper_clipped = upper.clip && !lower.clip;
        let baked = self.bake(&[
            (&lower, lower.opacity, BlendMode::Normal, false),
            (&upper, upper.opacity, upper.blend, upper_clipped),
        ]);
        let id = self.take_layer_id();
        let mut merged = lower.clone();
        merged.id = id;
        merged.set_channels(baked);
        merged.opacity = 1.0;
        merged.visible = true;
        merged.mask = None;
        merged.mask_kind = None;
        merged.mask_enabled = true;
        merged.clip = lower.clip;
        merged.reset_placement();
        merged.swatch = upper.swatch.or(lower.swatch);
        merged.used = lower.used | upper.used;
        self.layers.remove(index);
        self.layers[index - 1] = merged;
        self.active = index - 1;
        self.dirty = true;
        true
    }

    /// Composite `parts` (bottom to top: layer, opacity, mode, clipped) onto
    /// nothing, and return the result as fresh textures in the layer formats
    /// — base colour, roughness, metallic, height, emissive, opacity, normal.
    fn bake(&self, parts: &[(&Layer, f32, BlendMode, bool)]) -> [Arc<ImageView>; 7] {
        // The composite formats `composite_cs` declares: RG16F scalars.
        let targets: [Arc<ImageView>; 7] = std::array::from_fn(|i| {
            let format = if matches!(i, 1 | 2 | 3 | 5) { Format::R16G16_SFLOAT } else { Format::R16G16B16A16_SFLOAT };
            create_paint_image(&self.gpu, self.resolution, format, crate::vram::Kind::Layer)
        });
        let mut builder = AutoCommandBufferBuilder::primary(
            self.gpu.command_buffer_allocator.clone(),
            self.gpu.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        for view in &targets {
            builder
                .clear_color_image(ClearColorImageInfo {
                    clear_value: ClearColorValue::Float([0.0; 4]),
                    ..ClearColorImageInfo::image(view.image().clone())
                })
                .unwrap();
        }
        builder.bind_pipeline_compute(self.pipeline.clone()).unwrap();
        for (layer, opacity, blend, clipped) in parts {
            self.record_layer(&mut builder, layer, &targets, ALL_COMPOSITES, *opacity, *blend, *clipped);
        }
        submit_and_wait(&self.gpu, builder);

        // A channel no part has a texture for came out blank: keep it on
        // the placeholder rather than holding a full-size zero image.
        // Output order: colour, roughness, metallic, height, emissive,
        // opacity, normal.
        let blank = |i: usize| {
            parts.iter().all(|(l, ..)| {
                let view = [&l.basecolor, &l.roughness, &l.metallic, &l.height, &l.emissive, &l.opacity_map, &l.normal][i];
                self.is_placeholder(view)
            })
        };
        // The composite targets are storage images; layer textures must be
        // paintable (colour attachments). Formats already match.
        targets
            .into_iter()
            .enumerate()
            .map(|(i, view)| {
                if blank(i) {
                    return self.placeholder(view.image().format());
                }
                let out = self.new_cleared_texture(view.image().format());
                let mut builder = AutoCommandBufferBuilder::primary(
                    self.gpu.command_buffer_allocator.clone(),
                    self.gpu.queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();
                builder
                    .copy_image(vulkano::command_buffer::CopyImageInfo::images(
                        view.image().clone(),
                        out.image().clone(),
                    ))
                    .unwrap();
                submit_and_wait(&self.gpu, builder);
                out
            })
            .collect::<Vec<_>>()
        .try_into()
        .unwrap()
    }

    /// Replace a layer's mask with a greyscale image (red channel; white
    /// shows the layer, black hides it), resampled to the stack resolution.
    /// Adds a mask if the layer has none. The inverse of a layer-mask export.
    pub fn set_mask_from_image(&mut self, index: usize, image: &ImageData) {
        self.ensure_allocated();
        if index >= self.layers.len() {
            return;
        }
        let pixels = resample_box(image, self.resolution);
        let mut out = Vec::with_capacity(pixels.len() * 4);
        for p in &pixels {
            out.extend([p[0], 1.0, 0.0, 0.0].map(f16::from_f32));
        }
        // A fresh mask, so undo snapshots keep the one being replaced.
        let mask = self.new_cleared_texture(Format::R16G16B16A16_SFLOAT);
        self.upload_texels(&mask, out);
        self.layers[index].mask = Some(mask);
        self.layers[index].mask_kind = None;
        self.dirty = true;
    }

    /// Put a stack read back from a project file in place of this one's
    /// layers. Stroke history starts empty: it held pixels of the old layers.
    pub fn restore_saved(
        &mut self,
        layers: Vec<Layer>,
        active: usize,
        height_scale: f32,
        next_layer_number: u32,
    ) {
        if layers.is_empty() {
            return;
        }
        self.next_layer_id = layers.iter().map(|l| l.id).max().unwrap_or(0) + 1;
        self.next_layer_number = next_layer_number.max(1);
        self.active = active.min(layers.len() - 1);
        self.layers = layers;
        self.height_scale = height_scale;
        self.undo_stack.clear();
        self.redo_stack.clear();
        self.dirty = true;
    }

    /// The "Layer N" counter, for saving.
    pub fn next_layer_number(&self) -> u32 {
        self.next_layer_number
    }

    /// Read one texel of the flattened base colour (premultiplied linear
    /// RGBA) — what the colour picker samples. Reads whatever the last
    /// composite produced, which is what is on screen.
    pub fn read_composite_basecolor(&self, x: u32, y: u32) -> [f32; 4] {
        if !self.allocated || self.is_blank(&self.composite_basecolor) {
            return [0.0; 4];
        }
        let (x, y) = (x.min(self.resolution - 1), y.min(self.resolution - 1));
        let buffer: Subbuffer<[u16]> = Buffer::new_slice(
            self.gpu.memory_allocator.clone(),
            BufferCreateInfo { usage: BufferUsage::TRANSFER_DST, ..Default::default() },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            4,
        )
        .unwrap();
        let mut info = CopyImageToBufferInfo::image_buffer(
            self.composite_basecolor.image().clone(),
            buffer.clone(),
        );
        info.regions[0].image_offset = [x, y, 0];
        info.regions[0].image_extent = [1, 1, 1];
        let mut builder = AutoCommandBufferBuilder::primary(
            self.gpu.command_buffer_allocator.clone(),
            self.gpu.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        builder.copy_image_to_buffer(info).unwrap();
        submit_and_wait(&self.gpu, builder);
        let texel = buffer.read().unwrap();
        [0, 1, 2, 3].map(|i| f16::from_bits(texel[i]).to_f32())
    }

    /// Flatten immediately and wait — used by export, which reads the
    /// composites right after.
    pub fn composite_now(&mut self) {
        if !self.allocated && !self.has_content() {
            return;
        }
        self.dirty = true;
        let mut builder = AutoCommandBufferBuilder::primary(
            self.gpu.command_buffer_allocator.clone(),
            self.gpu.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        self.record_composite(&mut builder);
        builder
            .build()
            .unwrap()
            .execute(self.gpu.queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
    }

    /// Record the flatten pass into the frame's command buffer if any layer
    /// changed. Runs before the scene pass samples the composites.
    pub fn record_composite(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) {
        if !self.dirty {
            return;
        }
        if !self.allocated && !self.has_content() {
            // Still blank: the 1x1 composites already read as unpainted.
            self.dirty = false;
            return;
        }
        self.allocate();
        self.dirty = false;
        // Composites no layer can reach stay blank and are skipped below.
        let needs = self.composite_needs();
        let present = needs.iter().enumerate().fold(0u32, |m, (i, n)| m | ((*n as u32) << i));

        for view in [
            &self.composite_basecolor,
            &self.composite_roughness,
            &self.composite_metallic,
            &self.composite_height,
            &self.composite_emissive,
            &self.composite_opacity,
            &self.composite_normal,
        ] {
            if self.is_blank(view) {
                continue;
            }
            builder
                .clear_color_image(ClearColorImageInfo {
                    clear_value: ClearColorValue::Float([0.0; 4]),
                    ..ClearColorImageInfo::image(view.image().clone())
                })
                .unwrap();
        }

        let targets = self.composite_targets();
        builder.bind_pipeline_compute(self.pipeline.clone()).unwrap();
        // Shader-mask layers are not material; they flatten separately below.
        // A clipped layer shows through the nearest unclipped layer below it
        // (its base); a hidden base hides what is clipped to it.
        let mut base_visible = false;
        let mut have_base = false;
        for layer in self.layers.iter().filter(|l| l.purpose.is_none() && present != 0) {
            let clipped = layer.clip && have_base;
            if !clipped {
                have_base = true;
                base_visible = layer.visible;
            }
            if !layer.visible || (clipped && !base_visible) {
                continue;
            }
            self.record_layer(builder, layer, &targets, present, layer.opacity, layer.blend, clipped);
        }
        let groups = self.resolution / 8;

        self.record_shader_masks(builder);

        // Pad every composite outward past its UV islands. Without this the
        // shader's bilinear taps near an island edge mix painted texels with
        // never-written zeros, which shows up as dashed outlines along every
        // seam — and the height Sobel below turns that step into a fake ridge.
        // Height is dilated *before* the normals are derived so the gradient
        // sees a continuous field.
        if self.dilate {
        for (view, alpha_in_y) in [
            (&self.composite_basecolor, false),
            (&self.composite_roughness, true),
            (&self.composite_metallic, true),
            (&self.composite_height, true),
            (&self.composite_emissive, false),
            (&self.composite_opacity, true),
            (&self.composite_normal, false),
        ] {
            if !self.is_blank(view) {
                self.record_dilate(builder, view.clone(), alpha_in_y);
            }
        }
        for view in self.shader_masks.iter().map(|(_, v)| v).chain(self.matcap_masks.iter().map(|(_, v)| v)) {
            self.record_dilate(builder, view.clone(), false);
        }
        }

        // Derive tangent-space normals from the flattened height (and fill
        // normals); nothing to derive when no layer has either.
        if self.is_blank(&self.normal_map) {
            return;
        }
        let set = DescriptorSet::new(
            self.gpu.descriptor_set_allocator.clone(),
            self.normal_pipeline.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::image_view(0, self.normal_map.clone()),
                WriteDescriptorSet::image_view(1, self.composite_height.clone()),
                WriteDescriptorSet::image_view(2, self.composite_normal.clone()),
            ],
            [],
        )
        .unwrap();
        builder
            .bind_pipeline_compute(self.normal_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.normal_pipeline.layout().clone(),
                0,
                set,
            )
            .unwrap()
            .push_constants(
                self.normal_pipeline.layout().clone(),
                0,
                normal_cs::Pc {
                    size: self.resolution,
                    scale: self.height_scale,
                },
            )
            .unwrap();
        unsafe {
            builder.dispatch([groups, groups, 1]).unwrap();
        }
    }

    fn composite_targets(&self) -> [Arc<ImageView>; 7] {
        [
            self.composite_basecolor.clone(),
            self.composite_roughness.clone(),
            self.composite_metallic.clone(),
            self.composite_height.clone(),
            self.composite_emissive.clone(),
            self.composite_opacity.clone(),
            self.composite_normal.clone(),
        ]
    }

    /// Blend one layer onto `targets` (base colour, roughness, metallic,
    /// height, emissive, opacity, normal — RG16F for the scalars, RGBA16F
    /// for the rest) with the given
    /// opacity and mode. The compute pipeline must already be bound.
    fn record_layer(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        layer: &Layer,
        targets: &[Arc<ImageView>; 7],
        // Bit i set = `targets[i]` is a real composite; blank ones are bound
        // but never read or written.
        present: u32,
        opacity: f32,
        blend: BlendMode,
        clipped: bool,
    ) {
        let groups = self.resolution / 8;
        // Projected placement needs the baked maps; without them the
        // layer keeps its UV placement rather than vanishing.
        let source = self.projection_source.as_ref();
        let projection = if source.is_some() { layer.projection } else { Projection::Uv };
        let mask = match (&layer.mask, layer.mask_enabled) {
            (Some(mask), true) => mask.clone(),
            _ => self.white.clone(),
        };
        let [bc, rough, metal, height, emissive, opac, normal] = targets.clone();
        let set = DescriptorSet::new(
            self.gpu.descriptor_set_allocator.clone(),
            self.pipeline.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::image_view(0, bc),
                WriteDescriptorSet::image_view(1, rough),
                WriteDescriptorSet::image_view(2, metal),
                WriteDescriptorSet::image_view(3, height),
                WriteDescriptorSet::image_view(4, emissive),
                WriteDescriptorSet::image_view(5, opac),
                WriteDescriptorSet::image_view_sampler(6, layer.basecolor.clone(), self.tiling_sampler.clone()),
                WriteDescriptorSet::image_view_sampler(7, layer.roughness.clone(), self.tiling_sampler.clone()),
                WriteDescriptorSet::image_view_sampler(8, layer.metallic.clone(), self.tiling_sampler.clone()),
                WriteDescriptorSet::image_view_sampler(9, layer.height.clone(), self.tiling_sampler.clone()),
                WriteDescriptorSet::image_view_sampler(10, layer.emissive.clone(), self.tiling_sampler.clone()),
                WriteDescriptorSet::image_view_sampler(11, layer.opacity_map.clone(), self.tiling_sampler.clone()),
                WriteDescriptorSet::image_view_sampler(12, mask, self.sampler.clone()),
                WriteDescriptorSet::image_view(13, normal),
                WriteDescriptorSet::image_view_sampler(14, layer.normal.clone(), self.tiling_sampler.clone()),
                WriteDescriptorSet::image_view_sampler(
                    15,
                    source.map_or_else(|| self.white.clone(), |s| s.position.clone()),
                    self.sampler.clone(),
                ),
                WriteDescriptorSet::image_view_sampler(
                    16,
                    source.map_or_else(|| self.white.clone(), |s| s.normal.clone()),
                    self.sampler.clone(),
                ),
                WriteDescriptorSet::image_view(17, self.clip_coverage.clone()),
            ],
            [],
        )
        .unwrap();
        builder
            .bind_descriptor_sets(PipelineBindPoint::Compute, self.pipeline.layout().clone(), 0, set)
            .unwrap()
            .push_constants(
                self.pipeline.layout().clone(),
                0,
                composite_cs::Pc {
                    size: self.resolution,
                    mode: blend.shader_id(),
                    opacity,
                    rotation: layer.uv_rotation,
                    scale: layer.uv_scale,
                    offset: layer.uv_offset,
                    to_local: source
                        .map_or(glam::Mat4::IDENTITY, |s| {
                            layer
                                .projection_placement([s.frame[0], s.frame[1], s.frame[2]], s.frame[3])
                                .inverse()
                        })
                        .to_cols_array_2d(),
                    projection: projection.shader_id(),
                    axis: layer.projection_axis.shader_id(),
                    sharpness: layer.triplanar_sharpness,
                    repeat: layer.projection_repeat as u32,
                    clip: clipped as u32,
                    present,
                },
            )
            .unwrap();
        unsafe {
            builder.dispatch([groups, groups, 1]).unwrap();
        }
    }

    /// Flatten each shader-mask kind's layers (bottom to top, same blend
    /// modes, opacity and layer masks as material layers) over black.
    fn record_shader_masks(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) {
        let kinds: Vec<ShaderMask> = self.shader_mask_kinds().into_iter().filter(|k| *k != ShaderMask::Matcap).collect();
        self.shader_masks.retain(|(k, _)| kinds.contains(k));
        for kind in &kinds {
            if !self.shader_masks.iter().any(|(k, _)| k == kind) {
                let view = create_paint_image(&self.gpu, self.resolution, Format::R16G16B16A16_SFLOAT, crate::vram::Kind::Composite);
                self.shader_masks.push((*kind, view));
            }
        }
        let ids: Vec<u64> = self.layers.iter().filter(|l| l.is_matcap()).map(|l| l.id).collect();
        self.matcap_masks.retain(|(id, _)| ids.contains(id));
        for id in &ids {
            if !self.matcap_masks.iter().any(|(i, _)| i == id) {
                let view = create_paint_image(&self.gpu, self.resolution, Format::R16G16B16A16_SFLOAT, crate::vram::Kind::Composite);
                self.matcap_masks.push((*id, view));
            }
        }
        // Each target with the layers that flatten into it.
        let targets: Vec<(&Arc<ImageView>, Vec<&Layer>)> = self
            .shader_masks
            .iter()
            .map(|(kind, target)| (target, self.layers.iter().filter(|l| l.purpose == Some(*kind)).collect()))
            .chain(self.matcap_masks.iter().map(|(id, target)| (target, self.layers.iter().filter(|l| l.id == *id).collect())))
            .collect();
        let groups = self.resolution / 8;
        for (target, layers) in targets {
            builder
                .clear_color_image(ClearColorImageInfo {
                    clear_value: ClearColorValue::Float([0.0; 4]),
                    ..ClearColorImageInfo::image(target.image().clone())
                })
                .unwrap();
            builder.bind_pipeline_compute(self.mask_pipeline.clone()).unwrap();
            for layer in layers.into_iter().filter(|l| l.visible) {
                let set = DescriptorSet::new(
                    self.gpu.descriptor_set_allocator.clone(),
                    self.mask_pipeline.layout().set_layouts()[0].clone(),
                    [
                        WriteDescriptorSet::image_view(0, target.clone()),
                        WriteDescriptorSet::image_view_sampler(
                            1,
                            layer.basecolor.clone(),
                            self.tiling_sampler.clone(),
                        ),
                        WriteDescriptorSet::image_view_sampler(
                            2,
                            layer.mask.clone().filter(|_| layer.mask_enabled).unwrap_or_else(|| self.white.clone()),
                            self.sampler.clone(),
                        ),
                    ],
                    [],
                )
                .unwrap();
                builder
                    .bind_descriptor_sets(
                        PipelineBindPoint::Compute,
                        self.mask_pipeline.layout().clone(),
                        0,
                        set,
                    )
                    .unwrap()
                    .push_constants(
                        self.mask_pipeline.layout().clone(),
                        0,
                        mask_composite_cs::Pc {
                            size: self.resolution,
                            mode: layer.blend.shader_id(),
                            opacity: layer.opacity,
                            rotation: layer.uv_rotation,
                            scale: layer.uv_scale,
                            offset: layer.uv_offset,
                        },
                    )
                    .unwrap();
                unsafe {
                    builder.dispatch([groups, groups, 1]).unwrap();
                }
            }
        }
    }

    /// Grow covered texels outward by `DILATE_PASSES` texels, ping-ponging
    /// through the scratch image. An even pass count leaves the result back in
    /// the original image.
    fn record_dilate(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        image: Arc<ImageView>,
        alpha_in_y: bool,
    ) {
        let groups = self.resolution / 8;
        let (pipeline, temp) = if image.image().format() == Format::R16G16_SFLOAT {
            (&self.dilate_rg_pipeline, &self.dilate_temp_rg)
        } else {
            (&self.dilate_pipeline, &self.dilate_temp)
        };
        builder.bind_pipeline_compute(pipeline.clone()).unwrap();
        for pass in 0..DILATE_PASSES {
            let (src, dst) = if pass % 2 == 0 {
                (image.clone(), temp.clone())
            } else {
                (temp.clone(), image.clone())
            };
            let set = DescriptorSet::new(
                self.gpu.descriptor_set_allocator.clone(),
                pipeline.layout().set_layouts()[0].clone(),
                [
                    WriteDescriptorSet::image_view(0, dst),
                    WriteDescriptorSet::image_view(1, src),
                ],
                [],
            )
            .unwrap();
            builder
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    pipeline.layout().clone(),
                    0,
                    set,
                )
                .unwrap()
                .push_constants(
                    pipeline.layout().clone(),
                    0,
                    // Both variants share this push-constant block.
                    dilate_cs::Pc { size: self.resolution, alpha_in_y: alpha_in_y as u32 },
                )
                .unwrap();
            unsafe {
                builder.dispatch([groups, groups, 1]).unwrap();
            }
        }
    }

    fn new_cleared_texture(&self, format: Format) -> Arc<ImageView> {
        let view = create_layer_image(&self.gpu, self.resolution, format);

        let mut builder = AutoCommandBufferBuilder::primary(
            self.gpu.command_buffer_allocator.clone(),
            self.gpu.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        builder
            .clear_color_image(ClearColorImageInfo {
                clear_value: ClearColorValue::Float([0.0; 4]),
                ..ClearColorImageInfo::image(view.image().clone())
            })
            .unwrap();
        builder
            .build()
            .unwrap()
            .execute(self.gpu.queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
        view
    }
}

/// Build, submit and block on a one-shot command buffer.
fn submit_and_wait(
    gpu: &GpuContext,
    builder: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
) {
    builder
        .build()
        .unwrap()
        .execute(gpu.queue.clone())
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();
}

/// Coverage-weighted mean colour of resampled sRGB pixels, in linear RGB.
/// Samples a sparse grid: it only feeds a thumbnail swatch.
fn average_colour(pixels: &[[f32; 4]]) -> Option<[f32; 3]> {
    let step = (pixels.len() / 4096).max(1);
    let mut sum = [0.0f32; 3];
    let mut weight = 0.0f32;
    for p in pixels.iter().step_by(step) {
        let a = p[3];
        for c in 0..3 {
            sum[c] += srgb_to_linear(p[c]) * a;
        }
        weight += a;
    }
    (weight > 1e-3).then(|| sum.map(|v| v / weight))
}

fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
}

/// Pack resampled pixels into the (value·α, α) pairs the scalar channels use.
fn scalar_channel(pixels: &[[f32; 4]], map: impl Fn(f32) -> f32) -> Vec<f16> {
    let mut out = Vec::with_capacity(pixels.len() * 2);
    for p in pixels {
        out.push(f16::from_f32(map(p[0])));
        out.push(f16::ONE);
    }
    out
}

/// Box-filter an RGBA8 image to `res`×`res`, returning 0..1 floats. Averaging
/// the covered source rect keeps 4K source maps from aliasing when the layer
/// resolution is lower; when upscaling each target texel takes one sample.
fn resample_box(img: &ImageData, res: u32) -> Vec<[f32; 4]> {
    let mut out = Vec::with_capacity((res * res) as usize);
    if img.width == 0 || img.height == 0 {
        out.resize((res * res) as usize, [0.0; 4]);
        return out;
    }
    let sx = img.width as f32 / res as f32;
    let sy = img.height as f32 / res as f32;
    for y in 0..res {
        let y0 = ((y as f32 * sy) as u32).min(img.height - 1);
        let y1 = (((y + 1) as f32 * sy).ceil() as u32).clamp(y0 + 1, img.height);
        for x in 0..res {
            let x0 = ((x as f32 * sx) as u32).min(img.width - 1);
            let x1 = (((x + 1) as f32 * sx).ceil() as u32).clamp(x0 + 1, img.width);
            let mut acc = [0f32; 4];
            let mut n = 0f32;
            for yy in y0..y1 {
                let row = (yy * img.width) as usize * 4;
                for xx in x0..x1 {
                    let i = row + xx as usize * 4;
                    acc[0] += img.rgba[i] as f32;
                    acc[1] += img.rgba[i + 1] as f32;
                    acc[2] += img.rgba[i + 2] as f32;
                    acc[3] += img.rgba[i + 3] as f32;
                    n += 1.0;
                }
            }
            let inv = 1.0 / (255.0 * n.max(1.0));
            out.push([acc[0] * inv, acc[1] * inv, acc[2] * inv, acc[3] * inv]);
        }
    }
    out
}

/// Record a linear resample of `src` to `resolution` (halving steps when
/// shrinking, so detail averages rather than aliases) into a new layer image.
fn record_resample(
    gpu: &GpuContext,
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    src: &Arc<ImageView>,
    resolution: u32,
) -> Arc<ImageView> {
    let format = src.image().format();
    let mut current = src.clone();
    let mut size = current.image().extent()[0];
    loop {
        let next_size = if size / 2 >= resolution { size / 2 } else { resolution };
        let next = create_layer_image(gpu, next_size, format);
        builder
            .blit_image(BlitImageInfo {
                filter: Filter::Linear,
                ..BlitImageInfo::images(current.image().clone(), next.image().clone())
            })
            .unwrap();
        current = next;
        size = next_size;
        if size == resolution {
            return current;
        }
    }
}

/// A layer channel / mask texture (uncleared): rendered into by strokes.
fn create_layer_image(gpu: &GpuContext, resolution: u32, format: Format) -> Arc<ImageView> {
    layer_image(gpu, resolution, format, crate::vram::Kind::Layer)
}

fn layer_image(gpu: &GpuContext, resolution: u32, format: Format, kind: crate::vram::Kind) -> Arc<ImageView> {
    ImageView::new_default(
        crate::vram::image(
            gpu.memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format,
                extent: [resolution, resolution, 1],
                usage: ImageUsage::COLOR_ATTACHMENT
                    | ImageUsage::SAMPLED
                    | ImageUsage::TRANSFER_SRC
                    | ImageUsage::TRANSFER_DST,
                ..Default::default()
            },
            kind,
        )
        .unwrap(),
    )
    .unwrap()
}

fn create_paint_image(gpu: &GpuContext, resolution: u32, format: Format, kind: crate::vram::Kind) -> Arc<ImageView> {
    ImageView::new_default(
        crate::vram::image(
            gpu.memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format,
                extent: [resolution, resolution, 1],
                // TRANSFER_SRC is required: export reads the composites back
                // with copy_image_to_buffer.
                usage: ImageUsage::STORAGE
                    | ImageUsage::SAMPLED
                    | ImageUsage::TRANSFER_SRC
                    | ImageUsage::TRANSFER_DST,
                ..Default::default()
            },
            kind,
        )
        .unwrap(),
    )
    .unwrap()
}

mod composite_cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
#version 450
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 0, rgba16f) uniform image2D comp_bc;
layout(set = 0, binding = 1, rg16f) uniform image2D comp_rough;
layout(set = 0, binding = 2, rg16f) uniform image2D comp_metal;
layout(set = 0, binding = 3, rg16f) uniform image2D comp_height;
layout(set = 0, binding = 4, rgba16f) uniform image2D comp_emissive;
layout(set = 0, binding = 5, rg16f) uniform image2D comp_opacity;
layout(set = 0, binding = 6) uniform sampler2D layer_bc;
layout(set = 0, binding = 7) uniform sampler2D layer_rough;
layout(set = 0, binding = 8) uniform sampler2D layer_metal;
layout(set = 0, binding = 9) uniform sampler2D layer_height;
layout(set = 0, binding = 10) uniform sampler2D layer_emissive;
layout(set = 0, binding = 11) uniform sampler2D layer_opacity;
layout(set = 0, binding = 12) uniform sampler2D layer_mask;
layout(set = 0, binding = 13, rgba16f) uniform image2D comp_normal;
layout(set = 0, binding = 14) uniform sampler2D layer_normal;
// Baked world position / normal per texel (w = coverage), for projections.
layout(set = 0, binding = 15) uniform sampler2D proj_position;
layout(set = 0, binding = 16) uniform sampler2D proj_normal;
// Clipping: every unclipped layer writes its coverage here (r); a clipped
// layer above it reads it back and shows only there.
layout(set = 0, binding = 17, rgba16f) uniform image2D clip_coverage;
layout(push_constant) uniform Pc {
    uint size;
    uint mode;
    float opacity;
    float rotation; // radians, about the middle of the UV square
    vec2 scale;     // UV tiling for this layer's channels
    vec2 offset;
    // World → projection volume (the unit cube -0.5..0.5): the layer's
    // placement, framed to the object's bounds. Scale is uniform, so its
    // upper 3x3 also carries normals over once renormalised.
    mat4 to_local;
    uint projection; // 0 UV, 1 triplanar, 2 spherical, 3 cylindrical, 4 planar
    uint axis;       // 0 X, 1 Y, 2 Z
    float sharpness; // triplanar blend exponent
    uint repeat;     // 0: the image appears once, nothing past its edges
    uint clip;       // 0: write clip_coverage, 1: clipped (read it)
    // Bit i: composite binding i (colour, roughness, metallic, height,
    // emissive, opacity, normal) is real. The others are 1x1 blanks no
    // layer can reach; they are left alone.
    uint present;
} pc;

bool has(uint i) { return (pc.present & (1u << i)) != 0u; }

const float PI = 3.14159265359;

// Where to sample the layer's channels: up to three taps (triplanar), each
// with a weight; every other mode uses the first alone.
vec2 t0;
vec2 t1;
vec2 t2;
vec3 tw;

// The layer's placement transform. Rotation and scale both work about the
// middle of the image, so a material spins and zooms in place instead of
// swinging off toward 0,0.
vec2 place(vec2 base) {
    vec2 centred = base - 0.5;
    float cs = cos(pc.rotation);
    float sn = sin(pc.rotation);
    vec2 spun = vec2(centred.x * cs - centred.y * sn, centred.x * sn + centred.y * cs);
    return spun * pc.scale + 0.5 + pc.offset;
}

// A tap past the image's edges, with repeat off, contributes nothing.
float inside(vec2 t) {
    if (pc.repeat != 0u) return 1.0;
    return all(greaterThanEqual(t, vec2(0.0))) && all(lessThanEqual(t, vec2(1.0))) ? 1.0 : 0.0;
}

vec4 tap(sampler2D s) {
    vec4 r = texture(s, t0) * tw.x;
    if (tw.y > 0.0) r += texture(s, t1) * tw.y;
    if (tw.z > 0.0) r += texture(s, t2) * tw.z;
    return r;
}

// Fill t0..t2 / tw for the texel at `uv`. False where no surface was baked
// (between UV islands): those texels get nothing, and dilation pads the
// islands from their own edges instead of from an unrelated projection.
bool project(vec2 uv) {
    tw = vec3(1.0, 0.0, 0.0);
    if (pc.projection == 0u) {
        t0 = place(uv);
        return true;
    }
    // texelFetch, not texture: filtering would blend positions across an
    // island edge with the empty texels beyond it.
    ivec2 sz = textureSize(proj_position, 0);
    ivec2 q = clamp(ivec2(uv * vec2(sz)), ivec2(0), sz - 1);
    vec4 pos = texelFetch(proj_position, q, 0);
    if (pos.w < 0.5) return false;
    vec3 l = (pc.to_local * vec4(pos.xyz, 1.0)).xyz;

    if (pc.projection == 1u) {
        vec3 n = normalize(mat3(pc.to_local) * texelFetch(proj_normal, q, 0).xyz);
        vec3 w = pow(abs(n), vec3(max(pc.sharpness, 1.0)));
        tw = w / max(w.x + w.y + w.z, 1e-6);
        // Each side's image flips with the direction it faces, so patterns
        // and lettering read the right way round on the back too, as seen
        // from outside. Image top is +Y on the sides, -Z from above.
        t0 = place(vec2(n.x >= 0.0 ? -l.z : l.z, -l.y) + 0.5);
        t1 = place(vec2(l.x, n.y >= 0.0 ? l.z : -l.z) + 0.5);
        t2 = place(vec2(n.z >= 0.0 ? l.x : -l.x, -l.y) + 0.5);
        tw *= vec3(inside(t0), inside(t1), inside(t2));
        return true;
    }

    // Frame for the aimed modes: `f` is the axis, `front` the direction the
    // middle of a wrap faces (so the seam lands at the back), `right` the
    // image's +u from there; `up` is the planar image's top.
    vec3 f, front, up;
    if (pc.axis == 0u) {
        f = vec3(1.0, 0.0, 0.0); front = vec3(0.0, 0.0, 1.0); up = vec3(0.0, 1.0, 0.0);
    } else if (pc.axis == 1u) {
        f = vec3(0.0, 1.0, 0.0); front = vec3(0.0, 0.0, 1.0); up = vec3(0.0, 0.0, -1.0);
    } else {
        f = vec3(0.0, 0.0, 1.0); front = vec3(0.0, 1.0, 0.0); up = vec3(0.0, 1.0, 0.0);
    }
    vec3 right = cross(f, front);
    vec2 base;
    if (pc.projection == 2u) {
        vec3 d = l / max(length(l), 1e-6);
        base = vec2(
            atan(dot(d, right), dot(d, front)) / (2.0 * PI) + 0.5,
            acos(clamp(dot(d, f), -1.0, 1.0)) / PI
        );
    } else if (pc.projection == 3u) {
        base = vec2(atan(dot(l, right), dot(l, front)) / (2.0 * PI) + 0.5, 0.5 - dot(l, f));
    } else {
        // Planar: throw along the axis; the image lies across the other two.
        vec3 across = cross(up, f);
        base = vec2(dot(l, across) + 0.5, 0.5 - dot(l, up));
    }
    t0 = place(base);
    tw.x = inside(t0);
    return true;
}

// Blend modes on straight (unpremultiplied) values. Ids match
// BlendMode::shader_id. Separable modes work per channel; 13-16 mix colour
// properties (W3C compositing spec definitions).
float blend1(float b, float s, uint m) {
    if (m == 1u) return b * s;
    if (m == 2u) return b + s;
    if (m == 3u) return b + s - b * s;
    if (m == 4u) return b <= 0.5 ? 2.0 * b * s : 1.0 - 2.0 * (1.0 - b) * (1.0 - s);
    if (m == 5u) {
        float d = b <= 0.25 ? ((16.0 * b - 12.0) * b + 4.0) * b : sqrt(max(b, 0.0));
        return s <= 0.5 ? b - (1.0 - 2.0 * s) * b * (1.0 - b) : b + (2.0 * s - 1.0) * (d - b);
    }
    if (m == 6u) return s <= 0.5 ? 2.0 * b * s : 1.0 - 2.0 * (1.0 - b) * (1.0 - s);
    if (m == 7u) return min(b, s);
    if (m == 8u) return max(b, s);
    if (m == 9u) return b <= 0.0 ? 0.0 : (s >= 1.0 ? 1.0 : min(1.0, b / (1.0 - s)));
    if (m == 10u) return b >= 1.0 ? 1.0 : (s <= 0.0 ? 0.0 : 1.0 - min(1.0, (1.0 - b) / s));
    if (m == 11u) return abs(b - s);
    if (m == 12u) return b - s;
    return s;
}

float lum(vec3 c) { return dot(c, vec3(0.3, 0.59, 0.11)); }

vec3 clip_colour(vec3 c) {
    float l = lum(c);
    float n = min(min(c.r, c.g), c.b);
    float x = max(max(c.r, c.g), c.b);
    if (n < 0.0) c = l + (c - l) * l / max(l - n, 1e-6);
    if (x > 1.0) c = l + (c - l) * (1.0 - l) / max(x - l, 1e-6);
    return c;
}

vec3 set_lum(vec3 c, float l) { return clip_colour(c + (l - lum(c))); }

float sat(vec3 c) { return max(max(c.r, c.g), c.b) - min(min(c.r, c.g), c.b); }

vec3 set_sat(vec3 c, float s) {
    float mx = max(max(c.r, c.g), c.b);
    float mn = min(min(c.r, c.g), c.b);
    return mx > mn ? (c - mn) * s / (mx - mn) : vec3(0.0);
}

vec3 blend3(vec3 b, vec3 s, uint m) {
    if (m == 13u) return set_lum(set_sat(s, sat(b)), lum(b));
    if (m == 14u) return set_lum(set_sat(b, sat(s)), lum(b));
    if (m == 15u) return set_lum(s, lum(b));
    if (m == 16u) return set_lum(b, lum(s));
    vec3 r = vec3(blend1(b.r, s.r, m), blend1(b.g, s.g, m), blend1(b.b, s.b, m));
    // Colour can't go negative (Subtract); it can go past 1 (emission).
    return max(r, vec3(0.0));
}

// Premultiplied-alpha blend; mode math happens on unpremultiplied values and
// falls back to the source color where the backdrop is transparent.
vec4 blend_color(vec4 dst, vec4 src, uint mode) {
    vec3 dc = dst.a > 1e-5 ? dst.rgb / dst.a : vec3(0.0);
    vec3 sc = src.a > 1e-5 ? src.rgb / src.a : vec3(0.0);
    vec3 result = mix(sc, blend3(dc, sc, mode), dst.a);
    float out_a = src.a + dst.a * (1.0 - src.a);
    vec3 out_rgb = result * src.a + dc * dst.a * (1.0 - src.a);
    return vec4(out_rgb, out_a);
}

// Greyscale channels: the separable modes; the colour-component modes have
// no meaning on one value and act as Normal.
vec2 blend_scalar(vec2 dst, vec2 src, uint mode) {
    float dv = dst.y > 1e-5 ? dst.x / dst.y : 0.0;
    float sv = src.y > 1e-5 ? src.x / src.y : 0.0;
    float mixed = mode >= 13u ? sv : blend1(dv, sv, mode);
    float result = mix(sv, mixed, dst.y);
    float out_a = src.y + dst.y * (1.0 - src.y);
    float out_v = result * src.y + dv * dst.y * (1.0 - src.y);
    return vec2(out_v, out_a);
}

void main() {
    uvec2 id = gl_GlobalInvocationID.xy;
    if (id.x >= pc.size || id.y >= pc.size) return;
    ivec2 p = ivec2(id);
    vec2 uv = (vec2(id) + 0.5) / float(pc.size);

    // Channels are sampled through the layer's placement (projection plus
    // transform, repeat sampler), but the mask is read at the untransformed
    // UV: retiling or re-aiming a fill material must not drag the coverage
    // you painted along with it.
    bool covered = project(uv);

    // Layer opacity, the (optional) mask and the clip all scale the
    // premultiplied source.
    float layer_mask_v = covered ? texture(layer_mask, uv).r : 0.0;
    vec4 raw_bc = tap(layer_bc);
    vec2 raw_r = tap(layer_rough).xy;
    vec2 raw_m = tap(layer_metal).xy;
    vec2 raw_h = tap(layer_height).xy;
    vec4 raw_e = tap(layer_emissive);
    vec2 raw_o = tap(layer_opacity).xy;
    vec4 raw_n = tap(layer_normal);
    float clip = 1.0;
    if (pc.clip == 1u) {
        clip = imageLoad(clip_coverage, p).r;
    } else {
        // Where this layer has anything, in any channel, through its mask:
        // what layers clipped to it may show on. Opacity is not part of it,
        // so fading the base doesn't fade what is clipped to it twice.
        float cover = max(max(max(raw_bc.a, raw_r.y), max(raw_m.y, raw_h.y)), max(max(raw_e.a, raw_o.y), raw_n.a));
        imageStore(clip_coverage, p, vec4(clamp(cover, 0.0, 1.0) * layer_mask_v, 0.0, 0.0, 0.0));
    }
    float mask = pc.opacity * layer_mask_v * clip;
    vec4 src_bc = raw_bc * mask;
    vec2 src_r = raw_r * mask;
    vec2 src_m = raw_m * mask;
    vec2 src_h = raw_h * mask;

    if (has(0u)) imageStore(comp_bc, p, blend_color(imageLoad(comp_bc, p), src_bc, pc.mode));
    if (has(1u)) {
        vec2 r = blend_scalar(imageLoad(comp_rough, p).xy, src_r, pc.mode);
        imageStore(comp_rough, p, vec4(r, 0.0, 0.0));
    }
    if (has(2u)) {
        vec2 m = blend_scalar(imageLoad(comp_metal, p).xy, src_m, pc.mode);
        imageStore(comp_metal, p, vec4(m, 0.0, 0.0));
    }
    // Height is signed: only Add and Subtract make sense on it; every other
    // mode lays it on like Normal.
    uint height_mode = (pc.mode == 2u || pc.mode == 12u) ? pc.mode : 0u;
    if (has(3u)) {
        vec2 h = blend_scalar(imageLoad(comp_height, p).xy, src_h, height_mode);
        imageStore(comp_height, p, vec4(h, 0.0, 0.0));
    }

    vec4 src_e = raw_e * mask;
    if (has(4u)) imageStore(comp_emissive, p, blend_color(imageLoad(comp_emissive, p), src_e, pc.mode));
    vec2 src_o = raw_o * mask;
    if (has(5u)) {
        vec2 o = blend_scalar(imageLoad(comp_opacity, p).xy, src_o, pc.mode);
        imageStore(comp_opacity, p, vec4(o, 0.0, 0.0));
    }

    // Fill-material normals always stack as plain alpha-over: multiplying or
    // adding tangent vectors has no sensible meaning. Projected modes take
    // the material's tangent-space vectors as-is, without re-orienting them
    // to the projection, so their lighting direction is approximate; height
    // projects exactly, and the normals derived from it are correct.
    vec4 src_n = raw_n * mask;
    if (has(6u)) imageStore(comp_normal, p, blend_color(imageLoad(comp_normal, p), src_n, 0u));
}
",
    }
}

/// `composite_cs` cut down to one RGBA target: the base-colour path only,
/// used to flatten shader-mask layers (greyscale stored in base colour).
mod mask_composite_cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
#version 450
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 0, rgba16f) uniform image2D comp;
layout(set = 0, binding = 1) uniform sampler2D layer_bc;
layout(set = 0, binding = 2) uniform sampler2D layer_mask;
layout(push_constant) uniform Pc {
    uint size;
    uint mode;
    float opacity;
    float rotation;
    vec2 scale;
    vec2 offset;
} pc;

// Blend modes on straight (unpremultiplied) values. Ids match
// BlendMode::shader_id. Separable modes work per channel; 13-16 mix colour
// properties (W3C compositing spec definitions).
float blend1(float b, float s, uint m) {
    if (m == 1u) return b * s;
    if (m == 2u) return b + s;
    if (m == 3u) return b + s - b * s;
    if (m == 4u) return b <= 0.5 ? 2.0 * b * s : 1.0 - 2.0 * (1.0 - b) * (1.0 - s);
    if (m == 5u) {
        float d = b <= 0.25 ? ((16.0 * b - 12.0) * b + 4.0) * b : sqrt(max(b, 0.0));
        return s <= 0.5 ? b - (1.0 - 2.0 * s) * b * (1.0 - b) : b + (2.0 * s - 1.0) * (d - b);
    }
    if (m == 6u) return s <= 0.5 ? 2.0 * b * s : 1.0 - 2.0 * (1.0 - b) * (1.0 - s);
    if (m == 7u) return min(b, s);
    if (m == 8u) return max(b, s);
    if (m == 9u) return b <= 0.0 ? 0.0 : (s >= 1.0 ? 1.0 : min(1.0, b / (1.0 - s)));
    if (m == 10u) return b >= 1.0 ? 1.0 : (s <= 0.0 ? 0.0 : 1.0 - min(1.0, (1.0 - b) / s));
    if (m == 11u) return abs(b - s);
    if (m == 12u) return b - s;
    return s;
}

float lum(vec3 c) { return dot(c, vec3(0.3, 0.59, 0.11)); }

vec3 clip_colour(vec3 c) {
    float l = lum(c);
    float n = min(min(c.r, c.g), c.b);
    float x = max(max(c.r, c.g), c.b);
    if (n < 0.0) c = l + (c - l) * l / max(l - n, 1e-6);
    if (x > 1.0) c = l + (c - l) * (1.0 - l) / max(x - l, 1e-6);
    return c;
}

vec3 set_lum(vec3 c, float l) { return clip_colour(c + (l - lum(c))); }

float sat(vec3 c) { return max(max(c.r, c.g), c.b) - min(min(c.r, c.g), c.b); }

vec3 set_sat(vec3 c, float s) {
    float mx = max(max(c.r, c.g), c.b);
    float mn = min(min(c.r, c.g), c.b);
    return mx > mn ? (c - mn) * s / (mx - mn) : vec3(0.0);
}

vec3 blend3(vec3 b, vec3 s, uint m) {
    if (m == 13u) return set_lum(set_sat(s, sat(b)), lum(b));
    if (m == 14u) return set_lum(set_sat(b, sat(s)), lum(b));
    if (m == 15u) return set_lum(s, lum(b));
    if (m == 16u) return set_lum(b, lum(s));
    vec3 r = vec3(blend1(b.r, s.r, m), blend1(b.g, s.g, m), blend1(b.b, s.b, m));
    // Colour can't go negative (Subtract); it can go past 1 (emission).
    return max(r, vec3(0.0));
}

// Premultiplied-alpha blend; mode math happens on unpremultiplied values and
// falls back to the source color where the backdrop is transparent.
vec4 blend_color(vec4 dst, vec4 src, uint mode) {
    vec3 dc = dst.a > 1e-5 ? dst.rgb / dst.a : vec3(0.0);
    vec3 sc = src.a > 1e-5 ? src.rgb / src.a : vec3(0.0);
    vec3 result = mix(sc, blend3(dc, sc, mode), dst.a);
    float out_a = src.a + dst.a * (1.0 - src.a);
    vec3 out_rgb = result * src.a + dc * dst.a * (1.0 - src.a);
    return vec4(out_rgb, out_a);
}

void main() {
    uvec2 id = gl_GlobalInvocationID.xy;
    if (id.x >= pc.size || id.y >= pc.size) return;
    ivec2 p = ivec2(id);
    vec2 uv = (vec2(id) + 0.5) / float(pc.size);
    vec2 centred = uv - 0.5;
    float cs = cos(pc.rotation);
    float sn = sin(pc.rotation);
    vec2 spun = vec2(centred.x * cs - centred.y * sn, centred.x * sn + centred.y * cs);
    vec2 tuv = spun * pc.scale + 0.5 + pc.offset;
    float mask = pc.opacity * texture(layer_mask, uv).r;
    vec4 src = texture(layer_bc, tuv) * mask;
    imageStore(comp, p, blend_color(imageLoad(comp, p), src, pc.mode));
}
",
    }
}

mod normal_cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
#version 450
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 0, rgba16f) uniform writeonly image2D normal_map;
layout(set = 0, binding = 1, rg16f) uniform readonly image2D comp_height;
layout(set = 0, binding = 2, rgba16f) uniform readonly image2D comp_normal;
layout(push_constant) uniform Pc { uint size; float scale; } pc;

float height_at(ivec2 p) {
    p = clamp(p, ivec2(0), ivec2(int(pc.size) - 1));
    vec2 h = imageLoad(comp_height, p).xy;
    return h.y > 1e-5 ? h.x / h.y : 0.0;
}

void main() {
    uvec2 id = gl_GlobalInvocationID.xy;
    if (id.x >= pc.size || id.y >= pc.size) return;
    ivec2 p = ivec2(id);

    // Sobel gradient of the height field.
    float tl = height_at(p + ivec2(-1, -1));
    float t  = height_at(p + ivec2( 0, -1));
    float tr = height_at(p + ivec2( 1, -1));
    float l  = height_at(p + ivec2(-1,  0));
    float r  = height_at(p + ivec2( 1,  0));
    float bl = height_at(p + ivec2(-1,  1));
    float b  = height_at(p + ivec2( 0,  1));
    float br = height_at(p + ivec2( 1,  1));
    float dx = (tr + 2.0 * r + br) - (tl + 2.0 * l + bl);
    float dy = (bl + 2.0 * b + br) - (tl + 2.0 * t + tr);

    vec3 n = normalize(vec3(-dx * pc.scale, -dy * pc.scale, 1.0));

    // Fold in the fill-material normal, if any. Whiteout blending: sum the
    // slopes and multiply the flat components, so painted height detail rides
    // on top of the material's normal instead of replacing it.
    vec4 ln = imageLoad(comp_normal, p);
    if (ln.w > 1e-5) {
        vec3 nl = normalize(ln.xyz / ln.w);
        n = normalize(vec3(n.xy + nl.xy, n.z * nl.z));
    }

    // Either source of detail counts as coverage, so a fill layer that only
    // supplies a normal map still perturbs the surface.
    float coverage = max(imageLoad(comp_height, p).y, ln.w);
    imageStore(normal_map, p, vec4(n, clamp(coverage, 0.0, 1.0)));
}
",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu_context() -> Option<GpuContext> {
        crate::environment::test_gpu::context()
    }

    fn solid(size: u32, rgba: [u8; 4]) -> ImageData {
        ImageData {
            width: size,
            height: size,
            rgba: rgba.iter().copied().cycle().take((size * size * 4) as usize).collect::<Vec<u8>>().into(),
        }
    }

    /// Read one texel of a composite back to the host.
    fn read_texel(gpu: &GpuContext, view: &Arc<ImageView>, res: u32) -> [f32; 4] {
        read_texel_at(gpu, view, res, 0, 0)
    }

    /// Read the texel at (x, y).
    fn read_texel_at(gpu: &GpuContext, view: &Arc<ImageView>, res: u32, x: u32, y: u32) -> [f32; 4] {
        let buffer: vulkano::buffer::Subbuffer<[u16]> = Buffer::new_slice(
            gpu.memory_allocator.clone(),
            BufferCreateInfo { usage: BufferUsage::TRANSFER_DST, ..Default::default() },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            (res * res * 4) as u64,
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
        submit_and_wait(gpu, builder);
        let guard = buffer.read().unwrap();
        let i = ((y * res + x) * 4) as usize;
        [
            f16::from_bits(guard[i]).to_f32(),
            f16::from_bits(guard[i + 1]).to_f32(),
            f16::from_bits(guard[i + 2]).to_f32(),
            f16::from_bits(guard[i + 3]).to_f32(),
        ]
    }

    /// Left half one colour, right half another — enough to tell tiling apart.
    fn split_image(size: u32, left: [u8; 4], right: [u8; 4]) -> ImageData {
        let mut rgba = Vec::with_capacity((size * size * 4) as usize);
        for _ in 0..size {
            for x in 0..size {
                rgba.extend_from_slice(if x < size / 2 { &left } else { &right });
            }
        }
        ImageData { width: size, height: size, rgba: rgba.into() }
    }

    /// Clipping, the mask switch, merge down, apply mask and a blend mode,
    /// all checked on the flattened base colour.
    #[test]
    fn clipping_masks_merging_and_blend_modes_composite_as_expected() {
        let Some(gpu) = gpu_context() else {
            eprintln!("no Vulkan device; skipping");
            return;
        };
        let res = 16u32;
        let mut stack = LayerStack::new(gpu, res);
        let texel = |stack: &mut LayerStack, x: u32| {
            stack.composite_now();
            stack.read_texels(&stack.composite_basecolor.clone())[(4 * res + x) as usize]
        };
        // Base: opaque on the left half only.
        let mut half = solid(res, [255, 0, 0, 255]);
        for y in 0..res {
            for x in res / 2..res {
                Arc::make_mut(&mut half.rgba)[((y * res + x) * 4 + 3) as usize] = 0;
            }
        }
        stack.import_map_layer("base".into(), MapKind::BaseColor, &half);
        stack.add_layer();
        stack.fill_active(Some([0.0, 0.0, 1.0]), None, None);
        let top = stack.active;
        stack.layers[top].clip = true;

        let (left, right) = (2, res - 2);
        let l = texel(&mut stack, left);
        assert!(l[2] > 0.99 && l[0] < 0.01, "clipped layer shows over its base: {l:?}");
        assert!(texel(&mut stack, right)[3] < 0.01, "and nowhere else");

        // A hidden mask (value 0) hides the top layer; switched off, it's back.
        stack.add_mask(top, 0.0);
        assert!(texel(&mut stack, left)[0] > 0.99);
        stack.layers[top].mask_enabled = false;
        assert!(texel(&mut stack, left)[2] > 0.99);
        stack.layers[top].mask_enabled = true;
        stack.fill_mask(top, 0.5);

        // Merge down and apply mask both leave the picture as it was.
        let before = [texel(&mut stack, left), texel(&mut stack, right)];
        assert!(stack.merge_down(top));
        assert_eq!(stack.layers.len(), 2);
        let after = [texel(&mut stack, left), texel(&mut stack, right)];
        for (b, a) in before.iter().zip(&after) {
            for c in 0..4 {
                assert!((b[c] - a[c]).abs() < 2e-3, "merge changed {b:?} → {a:?}");
            }
        }
        let merged = stack.active;
        stack.add_mask(merged, 0.25);
        stack.layers[merged].uv_scale = [2.0, 2.0];
        let before = texel(&mut stack, left);
        assert!(stack.apply_mask(merged));
        assert!(stack.layers[merged].mask.is_none());
        assert_eq!(stack.layers[merged].uv_scale, [1.0, 1.0]);
        let after = texel(&mut stack, left);
        for c in 0..4 {
            assert!((before[c] - after[c]).abs() < 2e-3, "apply mask changed {before:?} → {after:?}");
        }

        // Screen: 0.5 over 0.5 gives 0.75.
        let mut stack = LayerStack::new(stack.gpu.clone(), res);
        stack.fill_active(Some([0.5; 3]), None, None);
        stack.add_layer();
        stack.fill_active(Some([0.5; 3]), None, None);
        let top = stack.active;
        stack.layers[top].blend = BlendMode::Screen;
        assert!((texel(&mut stack, left)[0] - 0.75).abs() < 2e-3);
        stack.layers[top].blend = BlendMode::Difference;
        assert!(texel(&mut stack, left)[0].abs() < 2e-3);
    }

    /// A mask can come from another layer's mask, from where another layer
    /// is painted, or from a shader mask — always as an independent copy.
    #[test]
    fn masks_can_be_taken_from_other_layers_and_shader_masks() {
        let Some(gpu) = gpu_context() else {
            eprintln!("no Vulkan device; skipping");
            return;
        };
        let mut stack = LayerStack::new(gpu, 16);
        stack.add_mask(0, 0.25); // layer 0: masked at 0.25, nothing painted
        stack.add_layer(); // layer 1: roughness painted everywhere
        stack.fill_active(None, Some(0.7), None);
        stack.add_layer(); // layer 2: the target, no mask
        assert!(stack.layers[2].mask.is_none());

        let mask_value = |stack: &LayerStack, i: usize| stack.read_texels(stack.layers[i].mask.as_ref().unwrap())[0][0];

        assert!(stack.mask_from(2, MaskSource::LayerMask(0)));
        assert!((mask_value(&stack, 2) - 0.25).abs() < 1e-3);
        assert!(!Arc::ptr_eq(stack.layers[0].mask.as_ref().unwrap(), stack.layers[2].mask.as_ref().unwrap()));

        assert!(stack.mask_from(2, MaskSource::LayerCoverage(1)));
        assert!((mask_value(&stack, 2) - 1.0).abs() < 1e-3);
        assert!(stack.mask_from(2, MaskSource::LayerCoverage(0)));
        assert!(mask_value(&stack, 2).abs() < 1e-3, "nothing painted on layer 0");

        // No such shader mask yet; then one filled white.
        assert!(!stack.mask_from(2, MaskSource::ShaderMask(ShaderMask::Matcap)));
        stack.add_shader_mask_layer(ShaderMask::Matcap);
        stack.fill_active(Some([1.0, 1.0, 1.0]), None, None);
        assert!(stack.mask_from(2, MaskSource::ShaderMask(ShaderMask::Matcap)));
        assert!((mask_value(&stack, 2) - 1.0).abs() < 1e-2);

        // A layer with no mask has none to give.
        assert!(!stack.mask_from(2, MaskSource::LayerMask(1)));
    }

    #[test]
    fn map_kind_detects_exports_and_common_names() {
        for kind in MapKind::ALL {
            let name = format!("Metal_Plate_{}.png", kind.suffix());
            assert_eq!(MapKind::detect(&name), Some(kind), "{name}");
        }
        assert_eq!(MapKind::detect("Fabric_Albedo.jpg"), Some(MapKind::BaseColor));
        assert_eq!(MapKind::detect("wood_Roughness.png"), Some(MapKind::Roughness));
        assert_eq!(MapKind::detect("brick_nor_gl_normal.png"), Some(MapKind::Normal));
        assert_eq!(MapKind::detect("Jacket_Emission.png"), Some(MapKind::Emissive));
        assert_eq!(MapKind::detect("photo.png"), None);
        // Exports from before the suffixes changed still come back.
        assert_eq!(MapKind::detect("Body_Skin_basecolor.png"), Some(MapKind::BaseColor));
        assert_eq!(MapKind::detect("Body_Skin_normal.png"), Some(MapKind::Normal));
        assert_eq!(MapKind::detect("Body_Skin_emissive.png"), Some(MapKind::Emissive));
        // Masks are never maps.
        assert_eq!(MapKind::detect("MyAvatar_emission_mask.png"), None);
        assert_eq!(MapKind::detect("MyAvatar_L02_Rust_mask.png"), None);
        assert_eq!(MapKind::detect("Body_Skin_EmissionMask.png"), None);
    }

    /// An imported map lands on a new layer and only covers where its alpha
    /// does: paint below shows through the transparent half, and the
    /// original layer is not modified.
    #[test]
    fn imported_map_is_a_new_layer_over_existing_paint() {
        let Some(gpu) = gpu_context() else {
            eprintln!("no Vulkan device; skipping");
            return;
        };
        let res = 64;
        let mut stack = LayerStack::new(gpu.clone(), res);
        stack.fill_active(Some([0.0, 0.0, 1.0]), None, None);
        let original = stack.layers[0].basecolor.clone();

        let image = split_image(32, [255, 0, 0, 255], [0, 255, 0, 0]);
        assert!(stack.import_map_layer("Base color · x.png".into(), MapKind::BaseColor, &image));
        assert!(!stack.import_map_layer("AO".into(), MapKind::Ao, &image));
        assert_eq!(stack.layers.len(), 2);
        assert_eq!(stack.active, 1);
        assert!(Arc::ptr_eq(&stack.layers[0].basecolor, &original));

        stack.composite_now();
        let left = read_texel_at(&gpu, &stack.composite_basecolor, res, 4, 4);
        let right = read_texel_at(&gpu, &stack.composite_basecolor, res, res - 4, 4);
        assert!(left[0] > 0.9 && left[2] < 0.01, "opaque import should cover, got {left:?}");
        assert!(right[2] > 0.9 && right[1] < 0.01, "transparent import should show paint below, got {right:?}");

        // The layer underneath still holds its own paint.
        let below = read_texel_at(&gpu, &original, res, 4, 4);
        assert!(below[2] > 0.9 && below[0] < 0.01, "original layer was modified: {below:?}");
    }

    /// The fill-layer + mask chain: a material fills a layer, and the mask
    /// decides how much of it reaches the composite.
    #[test]
    fn fill_layer_is_gated_by_its_mask() {
        let Some(gpu) = gpu_context() else {
            eprintln!("no Vulkan device; skipping");
            return;
        };
        let res = 64;
        let mut stack = LayerStack::new(gpu.clone(), res);

        // Pure red base colour, mid-grey everything else.
        let maps = vec![
            (MapSlot::BaseColor, solid(32, [255, 0, 0, 255])),
            (MapSlot::Roughness, solid(32, [128, 128, 128, 255])),
            (MapSlot::Metallic, solid(32, [0, 0, 0, 255])),
            (MapSlot::Height, solid(32, [128, 128, 128, 255])),
            (MapSlot::Normal, solid(32, [128, 128, 255, 255])),
            (MapSlot::Ao, solid(32, [255, 255, 255, 255])),
        ];
        let written = stack.fill_layer_from_material(0, &maps);
        assert_eq!(written.len(), 5, "AO has no layer channel and must be skipped");
        assert!(!written.contains(&MapSlot::Ao));

        // A black mask hides the fill entirely.
        stack.add_mask(0, 0.0);
        stack.composite_now();
        let hidden = read_texel(&gpu, &stack.composite_basecolor, res);
        assert!(hidden[0] < 0.01, "black mask should hide the layer, got {hidden:?}");

        // A white mask lets it through: sRGB 255 red is 1.0 linear.
        stack.fill_mask(0, 1.0);
        stack.composite_now();
        let shown = read_texel(&gpu, &stack.composite_basecolor, res);
        assert!(shown[0] > 0.9, "white mask should reveal the layer, got {shown:?}");
        assert!(shown[1] < 0.01 && shown[2] < 0.01, "colour should stay red, got {shown:?}");

        // Inverting a white mask hides it again.
        stack.invert_mask(0);
        stack.composite_now();
        let inverted = read_texel(&gpu, &stack.composite_basecolor, res);
        assert!(inverted[0] < 0.01, "inverted white mask should hide, got {inverted:?}");
    }

    /// A shader-mask layer flattens into its own texture, stays out of the
    /// material, honours its layer mask, and disappears with its last layer.
    #[test]
    fn shader_mask_layers_composite_apart_from_material() {
        let Some(gpu) = gpu_context() else {
            eprintln!("no Vulkan device; skipping");
            return;
        };
        let res = 64;
        let mut stack = LayerStack::new(gpu.clone(), res);
        stack.fill_active(Some([0.2, 0.4, 0.6]), None, None);

        stack.add_shader_mask_layer(ShaderMask::Matcap);
        assert_eq!(stack.active_layer().purpose, Some(ShaderMask::Matcap));
        stack.fill_active(Some([1.0; 3]), None, None);
        stack.composite_now();

        let material = read_texel(&gpu, &stack.composite_basecolor, res);
        assert!(
            (material[0] - 0.2).abs() < 0.01 && (material[2] - 0.6).abs() < 0.01,
            "a white mask layer must not leak into base colour, got {material:?}"
        );
        let matcap = read_texel(&gpu, stack.shader_mask(ShaderMask::Matcap).unwrap(), res);
        assert!(matcap[0] > 0.99, "matcap mask should be white, got {matcap:?}");
        assert!(stack.shader_mask(ShaderMask::Emission).is_none());
        assert_eq!(stack.shader_mask_kinds(), vec![ShaderMask::Matcap]);

        // A black layer mask hides the mask layer like any other layer.
        let active = stack.active;
        stack.add_mask(active, 0.0);
        stack.composite_now();
        let hidden = read_texel(&gpu, stack.shader_mask(ShaderMask::Matcap).unwrap(), res);
        assert!(hidden[0] < 0.01, "layer mask should hide it, got {hidden:?}");

        stack.remove_layer(active);
        stack.composite_now();
        assert!(stack.shader_mask(ShaderMask::Matcap).is_none());
    }

    /// A material's own textures become the bottom "Base material" layer of
    /// an untouched stack, scaled by the material's factors, and painting
    /// goes to a fresh layer above; a stack with work in it is left alone.
    #[test]
    fn material_textures_seed_a_base_layer() {
        let Some(gpu) = gpu_context() else {
            eprintln!("no Vulkan device; skipping");
            return;
        };
        let res = 16;
        let mut stack = LayerStack::new(gpu.clone(), res);
        let white = ImageData { width: 4, height: 4, rgba: [255u8, 255, 255, 255].repeat(16).into() };
        let rough = ImageData { width: 4, height: 4, rgba: [128u8, 128, 128, 255].repeat(16).into() };
        assert!(stack.is_pristine());
        assert!(stack.seed_from_material(&[
            (MapKind::BaseColor, &white, [0.5, 0.25, 1.0]),
            (MapKind::Roughness, &rough, [0.5; 3]),
        ]));
        assert_eq!(stack.layers.len(), 2);
        assert_eq!(stack.layers[0].name, "Base material");
        assert_eq!((stack.active, stack.layers[1].name.as_str()), (1, "Layer 2"), "painting goes on the new layer");
        stack.composite_now();
        let bc = read_texel(&gpu, &stack.composite_basecolor, res);
        assert!((bc[0] - 0.5).abs() < 0.01 && (bc[1] - 0.25).abs() < 0.01 && bc[3] > 0.99, "tinted by the factor: {bc:?}");
        let r = read_texel(&gpu, &stack.composite_roughness, res);
        assert!((r[0] - 0.5 * 128.0 / 255.0).abs() < 0.02, "roughness scaled: {r:?}");
        assert!(!stack.seed_from_material(&[(MapKind::BaseColor, &white, [1.0; 3])]), "only an untouched stack");
        assert_eq!(stack.layers.len(), 2);
    }

    /// Every matcap layer flattens to a mask of its own, so a material can
    /// carry any number of matcaps, each showing only where it is painted.
    #[test]
    fn matcap_layers_each_have_their_own_mask() {
        let Some(gpu) = gpu_context() else {
            eprintln!("no Vulkan device; skipping");
            return;
        };
        let res = 32;
        let mut stack = LayerStack::new(gpu.clone(), res);
        stack.add_shader_mask_layer(ShaderMask::Matcap);
        let chrome = stack.active;
        stack.fill_active(Some([1.0; 3]), None, None);
        stack.layers[chrome].set_matcap(&gpu, "Chrome".into(), crate::matcap::builtin("Chrome").unwrap(), None);
        stack.add_shader_mask_layer(ShaderMask::Matcap);
        let shine = stack.active;
        stack.fill_active(Some([0.0; 3]), None, None);
        stack.layers[shine].set_matcap(&gpu, "Shine".into(), crate::matcap::builtin("Shine").unwrap(), None);
        stack.composite_now();

        assert_eq!(stack.layers[chrome].name, "Matcap · Chrome");
        let mask = |stack: &LayerStack, i: usize| read_texel(&gpu, stack.matcap_mask(stack.layers[i].id).unwrap(), res)[0];
        assert!(mask(&stack, chrome) > 0.99 && mask(&stack, shine) < 0.01, "masks are separate");
        let shown: Vec<&str> = stack.matcap_layers().iter().map(|(m, _)| m.name.as_str()).collect();
        assert_eq!(shown, ["Chrome", "Shine"], "bottom to top");
        // The Matcap shader mask is the active matcap layer's.
        assert!(read_texel(&gpu, stack.shader_mask(ShaderMask::Matcap).unwrap(), res)[0] < 0.01);
        // Matcap layers stay out of the material, and don't merge.
        assert!(read_texel(&gpu, &stack.composite_basecolor, res)[3] < 0.01);
        assert!(stack.merge_down_blocker(shine).is_some());

        stack.layers[chrome].visible = false;
        stack.mark_dirty();
        stack.composite_now();
        let shown: Vec<&str> = stack.matcap_layers().iter().map(|(m, _)| m.name.as_str()).collect();
        assert_eq!(shown, ["Shine"], "hidden layers are not drawn");
        stack.remove_layer(shine);
        stack.composite_now();
        assert!(stack.matcap_layers().is_empty());
    }

    /// Wall-clock cost of the per-frame composite, which is what a drag pays
    /// on every mouse move. Reports the minimum of many runs: the mean is
    /// dominated by GPU clock ramp and scheduler noise. Run with:
    ///   cargo test --release bench_composite -- --ignored --nocapture
    #[test]
    #[ignore = "benchmark"]
    fn bench_composite() {
        let Some(gpu) = gpu_context() else { return };
        const RES: u32 = 2048;
        const WARMUP: u32 = 40;
        const RUNS: u32 = 60;

        // One long warm-up first, so clocks are up before anything is timed.
        {
            let mut warm = LayerStack::new(gpu.clone(), RES);
            warm.fill_active(Some([0.5, 0.5, 0.5]), Some(0.5), Some(0.0));
            for _ in 0..WARMUP {
                warm.mark_dirty();
                warm.composite_now();
            }
        }

        for count in [1usize, 3, 5] {
            for dilate in [false, true] {
                let mut stack = LayerStack::new(gpu.clone(), RES);
                for _ in 1..count {
                    stack.add_layer();
                }
                for i in 0..count {
                    stack.active = i;
                    stack.fill_active(Some([0.5, 0.5, 0.5]), Some(0.5), Some(0.0));
                }
                stack.dilate = dilate;
                for _ in 0..WARMUP {
                    stack.mark_dirty();
                    stack.composite_now();
                }
                let mut samples = Vec::with_capacity(RUNS as usize);
                for _ in 0..RUNS {
                    stack.mark_dirty();
                    let t = std::time::Instant::now();
                    stack.composite_now();
                    samples.push(t.elapsed().as_secs_f64() * 1000.0);
                }
                samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
                println!(
                    "composite {RES}px  {count} layer(s)  dilate={dilate:<5}  \
                     min {:6.2} ms   median {:6.2} ms",
                    samples[0],
                    samples[samples.len() / 2],
                );
            }
        }
    }

    /// History depth must follow a memory budget, not a hard step count.
    #[test]
    fn undo_history_is_bounded_by_budget_not_step_count() {
        let Some(gpu) = gpu_context() else {
            eprintln!("no Vulkan device; skipping");
            return;
        };
        let res = 64;
        let mut stack = LayerStack::new(gpu.clone(), res);

        // Far more strokes than the old fixed depth of ten.
        const STROKES: usize = 40;
        for i in 0..STROKES {
            stack.begin_stroke(&[PaintChannel::BaseColor]);
            stack.fill_active(Some([i as f32 / STROKES as f32, 0.0, 0.0]), None, None);
        }
        assert_eq!(stack.undo_depth(), STROKES, "depth should not be capped at a count");
        assert!(stack.history_bytes() > 0, "snapshots should be accounted for");

        stack.composite_now();
        let latest = read_texel(&gpu, &stack.composite_basecolor, res);

        assert!(stack.undo_stroke());
        stack.composite_now();
        let undone = read_texel(&gpu, &stack.composite_basecolor, res);
        assert!(
            (undone[0] - latest[0]).abs() > 0.005,
            "undo should roll the content back: {latest:?} -> {undone:?}"
        );

        assert!(stack.redo_stroke());
        stack.composite_now();
        let redone = read_texel(&gpu, &stack.composite_basecolor, res);
        assert!(
            (redone[0] - latest[0]).abs() < 0.005,
            "redo should restore what undo removed: {redone:?} vs {latest:?}"
        );

        // Undoing everything must be possible, and then bottom out cleanly.
        let mut steps = 0;
        while stack.undo_stroke() {
            steps += 1;
            assert!(steps <= STROKES + 1, "undo should terminate");
        }
        assert!(steps >= STROKES, "every recorded stroke should be undoable, got {steps}");

        // A tight budget trims the oldest, but never leaves nothing to undo.
        stack.undo_budget = 1;
        stack.begin_stroke(&[PaintChannel::BaseColor]);
        stack.begin_stroke(&[PaintChannel::BaseColor]);
        assert_eq!(stack.undo_depth(), 1, "budget should trim to the newest step");
    }

    /// Composites must be padded outside their covered region, or bilinear
    /// taps at UV island edges blend against zeros and draw dashed seams.
    #[test]
    fn composites_are_dilated_past_their_coverage() {
        let Some(gpu) = gpu_context() else {
            eprintln!("no Vulkan device; skipping");
            return;
        };
        let res = 64u32;
        let mut stack = LayerStack::new(gpu.clone(), res);

        // A covered block spanning x,y in 24..40; everything else uncovered.
        const LO: u32 = 24;
        const HI: u32 = 40;
        let mut texels = vec![f16::ZERO; (res * res * 4) as usize];
        for y in LO..HI {
            for x in LO..HI {
                let i = ((y * res + x) * 4) as usize;
                texels[i] = f16::ONE; // red, premultiplied
                texels[i + 3] = f16::ONE; // coverage
            }
        }
        let target = stack.real_channel(0, 0);
        stack.upload_texels(&target, texels);
        stack.add_mask(0, 1.0);
        stack.mark_dirty();
        stack.composite_now();

        let row = (LO + HI) / 2;
        let inside = read_texel_at(&gpu, &stack.composite_basecolor, res, LO, row);
        assert!(inside[3] > 0.5, "the covered block should survive, got {inside:?}");

        // DILATE_PASSES texels of padding: the last one is covered...
        let edge = read_texel_at(&gpu, &stack.composite_basecolor, res, LO - DILATE_PASSES, row);
        assert!(
            edge[3] > 0.01,
            "expected padding {DILATE_PASSES} texels out, got {edge:?}"
        );
        assert!(edge[0] > 0.01, "padding should carry the colour too, got {edge:?}");

        // ...and one beyond it is still empty, so dilation is bounded.
        let beyond =
            read_texel_at(&gpu, &stack.composite_basecolor, res, LO - DILATE_PASSES - 2, row);
        assert!(
            beyond[3] < 0.01,
            "dilation should not run past its pass count, got {beyond:?}"
        );
    }

    /// Per-layer placement: scale, offset and rotation, all about the middle of
    /// the UV square so a material zooms and spins in place.
    #[test]
    fn layer_uv_transform_places_the_fill() {
        let Some(gpu) = gpu_context() else {
            eprintln!("no Vulkan device; skipping");
            return;
        };
        let res = 64u32;
        let mut stack = LayerStack::new(gpu.clone(), res);

        // A blue band at u in [0.62, 0.85); red everywhere else. Every sample
        // below sits several texels clear of an edge.
        let size = 64u32;
        let mut rgba = Vec::with_capacity((size * size * 4) as usize);
        for _ in 0..size {
            for x in 0..size {
                let u = (x as f32 + 0.5) / size as f32;
                let blue = (0.62..0.85).contains(&u);
                rgba.extend_from_slice(if blue {
                    &[0, 0, 255, 255]
                } else {
                    &[255, 0, 0, 255]
                });
            }
        }
        let maps =
            vec![(MapSlot::BaseColor, ImageData { width: size, height: size, rgba: rgba.into() })];
        stack.fill_layer_from_material(0, &maps);
        assert!(stack.layers[0].has_fill, "fill must flag the layer for the controls");
        stack.add_mask(0, 1.0);

        let is_blue = |c: [f32; 4]| c[2] > 0.5 && c[0] < 0.5;
        let is_red = |c: [f32; 4]| c[0] > 0.5 && c[2] < 0.5;

        // Identity: both probes land outside the band.
        stack.composite_now();
        assert!(is_red(read_texel_at(&gpu, &stack.composite_basecolor, res, 35, 32)));
        assert!(is_red(read_texel_at(&gpu, &stack.composite_basecolor, res, 32, 45)));

        // Scaling about the centre pushes the probe outward into the band.
        stack.layers[0].uv_scale = [3.0, 3.0];
        stack.mark_dirty();
        stack.composite_now();
        assert!(
            is_blue(read_texel_at(&gpu, &stack.composite_basecolor, res, 35, 32)),
            "3x scale should reach the band"
        );

        // Offset shifts it there instead.
        stack.layers[0].uv_scale = [1.0, 1.0];
        stack.layers[0].uv_offset = [0.12, 0.0];
        stack.mark_dirty();
        stack.composite_now();
        assert!(
            is_blue(read_texel_at(&gpu, &stack.composite_basecolor, res, 35, 32)),
            "offset should reach the band"
        );

        // A quarter turn maps the vertical probe onto the horizontal band.
        stack.layers[0].uv_offset = [0.0, 0.0];
        stack.layers[0].uv_rotation = -std::f32::consts::FRAC_PI_2;
        stack.mark_dirty();
        stack.composite_now();
        assert!(
            is_blue(read_texel_at(&gpu, &stack.composite_basecolor, res, 32, 45)),
            "rotation should swing the probe into the band"
        );
        // ...and leaves the horizontal probe out of it.
        assert!(
            is_red(read_texel_at(&gpu, &stack.composite_basecolor, res, 35, 32)),
            "rotation should move the other probe off the band"
        );

        // A fresh layer starts unplaced.
        stack.add_layer();
        assert_eq!(stack.layers[1].uv_scale, [1.0, 1.0]);
        assert_eq!(stack.layers[1].uv_rotation, 0.0);
        assert!(!stack.layers[1].has_fill);
    }
    /// Reordering must carry the selection and the undo history with the
    /// layer, not leave them pointing at whatever slid into the old slot.
    #[test]
    fn reordering_keeps_selection_and_history_aligned() {
        let Some(gpu) = gpu_context() else {
            eprintln!("no Vulkan device; skipping");
            return;
        };
        let mut stack = LayerStack::new(gpu, 32);
        stack.add_layer();
        stack.add_layer();
        for (i, name) in ["A", "B", "C"].iter().enumerate() {
            stack.layers[i].name = (*name).to_string();
        }

        // Pin a history entry to the bottom layer, "A".
        stack.active = 0;
        stack.begin_stroke(&[PaintChannel::BaseColor]);
        let a_id = stack.layers[0].id;
        assert_eq!(stack.undo_stack.back().unwrap().layer, a_id);
        let painted = |s: &LayerStack| s.layer_index(s.undo_stack.back().unwrap().layer);

        let order = |s: &LayerStack| -> Vec<String> {
            s.layers.iter().map(|l| l.name.clone()).collect()
        };

        // Bottom to top.
        stack.move_layer(0, 2);
        assert_eq!(order(&stack), ["B", "C", "A"]);
        assert_eq!(stack.active, 2, "selection should follow the moved layer");
        assert_eq!(painted(&stack), Some(2), "history should follow too");

        // And back down again.
        stack.move_layer(2, 0);
        assert_eq!(order(&stack), ["A", "B", "C"]);
        assert_eq!(stack.active, 0);
        assert_eq!(painted(&stack), Some(0));

        // Neighbour swap: what the arrow buttons actually do.
        stack.active = 1;
        stack.move_layer(1, 2);
        assert_eq!(order(&stack), ["A", "C", "B"]);
        assert_eq!(stack.active, 2);

        // Out-of-range and no-op moves leave the stack alone.
        stack.move_layer(0, 0);
        stack.move_layer(0, 99);
        assert_eq!(order(&stack), ["A", "C", "B"]);
    }

    /// The point of reordering: which layer ends up on top.
    #[test]
    fn reordering_changes_which_layer_wins() {
        let Some(gpu) = gpu_context() else {
            eprintln!("no Vulkan device; skipping");
            return;
        };
        let res = 32;
        let mut stack = LayerStack::new(gpu.clone(), res);
        stack.fill_active(Some([1.0, 0.0, 0.0]), None, None);
        stack.add_layer();
        stack.fill_active(Some([0.0, 1.0, 0.0]), None, None);

        stack.composite_now();
        let c = read_texel(&gpu, &stack.composite_basecolor, res);
        assert!(c[1] > 0.9 && c[0] < 0.1, "green is on top, got {c:?}");

        // Lift red above green.
        stack.move_layer(0, 1);
        stack.composite_now();
        let c = read_texel(&gpu, &stack.composite_basecolor, res);
        assert!(c[0] > 0.9 && c[1] < 0.1, "red should win after reorder, got {c:?}");
    }

    /// A fill material's normal map must survive into the sampled normal map,
    /// which is what the per-layer normal channel exists for.
    #[test]
    fn fill_material_normal_reaches_the_normal_map() {
        let Some(gpu) = gpu_context() else {
            eprintln!("no Vulkan device; skipping");
            return;
        };
        let res = 64;
        let mut stack = LayerStack::new(gpu.clone(), res);

        // Tilt hard along +X: sRGB 255 in red decodes to x = +1.
        let maps = vec![(MapSlot::Normal, solid(32, [255, 128, 128, 255]))];
        stack.fill_layer_from_material(0, &maps);
        stack.add_mask(0, 1.0);
        stack.composite_now();

        let n = read_texel(&gpu, &stack.normal_map, res);
        assert!(n[3] > 0.5, "normal coverage should follow the fill, got {n:?}");
        assert!(n[0] > 0.5, "normal should tilt along +X, got {n:?}");
    }

    const RED: [u8; 4] = [255, 0, 0, 255];
    const GREEN: [u8; 4] = [0, 255, 0, 255];
    const BLUE: [u8; 4] = [0, 0, 255, 255];
    const WHITE: [u8; 4] = [255, 255, 255, 255];

    /// Top-left red, top-right green, bottom-left blue, bottom-right white —
    /// tells apart where a projection landed in both directions.
    fn quadrant_image(size: u32) -> ImageData {
        let mut rgba = Vec::with_capacity((size * size * 4) as usize);
        for y in 0..size {
            for x in 0..size {
                let (right, bottom) = (x >= size / 2, y >= size / 2);
                rgba.extend_from_slice(match (right, bottom) {
                    (false, false) => &RED,
                    (true, false) => &GREEN,
                    (false, true) => &BLUE,
                    (true, true) => &WHITE,
                });
            }
        }
        ImageData { width: size, height: size, rgba: rgba.into() }
    }

    /// A stand-in for a baked mesh map: `texel(x, y)` per texel.
    fn mesh_map(stack: &LayerStack, texel: impl Fn(u32, u32) -> [f32; 4]) -> Arc<ImageView> {
        let res = stack.resolution;
        let view = create_paint_image(&stack.gpu, res, Format::R16G16B16A16_SFLOAT, crate::vram::Kind::Bake);
        let mut data = Vec::with_capacity((res * res * 4) as usize);
        for y in 0..res {
            for x in 0..res {
                data.extend(texel(x, y).map(f16::from_f32));
            }
        }
        stack.upload_texels(&view, data);
        view
    }

    fn colour_of(c: [f32; 4]) -> &'static str {
        match (c[0] > 0.5, c[1] > 0.5, c[2] > 0.5) {
            (true, false, false) => "red",
            (false, true, false) => "green",
            (false, false, true) => "blue",
            (true, true, true) => "white",
            _ if c[3] < 0.01 => "empty",
            _ => "other",
        }
    }

    /// Four 16-texel column bands, each holding one surface point (and
    /// normal); returns the colour each band composited to.
    fn project_bands(
        projection: Projection,
        axis: Axis,
        points: [([f32; 3], [f32; 3]); 4],
    ) -> Option<[&'static str; 4]> {
        project_bands_with(projection, axis, points, |_| {})
    }

    /// `project_bands` with the layer adjusted first (placement, repeat).
    fn project_bands_with(
        projection: Projection,
        axis: Axis,
        points: [([f32; 3], [f32; 3]); 4],
        adjust: impl Fn(&mut Layer),
    ) -> Option<[&'static str; 4]> {
        let gpu = gpu_context()?;
        let res = 64;
        let mut stack = LayerStack::new(gpu.clone(), res);
        stack.fill_layer_from_image(0, &quadrant_image(64));
        let band = |x: u32| (x / 16) as usize;
        let position = mesh_map(&stack, |x, _| {
            let p = points[band(x)].0;
            [p[0], p[1], p[2], 1.0]
        });
        let normal = mesh_map(&stack, |x, _| {
            let n = points[band(x)].1;
            [n[0], n[1], n[2], 1.0]
        });
        stack.set_projection_source(position, normal, [0.0; 3], 1.0);
        stack.layers[0].projection = projection;
        stack.layers[0].projection_axis = axis;
        adjust(&mut stack.layers[0]);
        stack.composite_now();
        Some([8, 24, 40, 56].map(|x| {
            colour_of(read_texel_at(&gpu, &stack.composite_basecolor, res, x, 16))
        }))
    }

    /// Spherical and cylindrical wraps place by direction around the axis,
    /// seam at the back: front-left is the image's left, up is its top.
    #[test]
    fn wraps_place_by_direction_around_the_axis() {
        let up = [0.0, 0.0, 1.0];
        let points = [
            ([-0.4, 0.1, 0.3], up),
            ([0.4, 0.1, 0.3], up),
            ([-0.4, -0.1, 0.3], up),
            ([0.4, -0.1, 0.3], up),
        ];
        for mode in [Projection::Spherical, Projection::Cylindrical] {
            let Some(got) = project_bands(mode, Axis::Y, points) else {
                eprintln!("no Vulkan device; skipping");
                return;
            };
            assert_eq!(got, ["red", "green", "blue", "white"], "{mode:?}");
        }
    }

    /// Planar throws straight along the axis, whatever the UVs say: the
    /// bands' UVs run left to right but their positions run right to left.
    #[test]
    fn planar_follows_position_not_uvs() {
        let n = [0.0, 0.0, 1.0];
        let points = [
            ([0.3, 0.2, 0.0], n),
            ([-0.3, 0.2, 0.0], n),
            ([0.3, -0.2, 0.0], n),
            ([-0.3, -0.2, 0.0], n),
        ];
        let Some(got) = project_bands(Projection::Planar, Axis::Z, points) else {
            eprintln!("no Vulkan device; skipping");
            return;
        };
        assert_eq!(got, ["green", "red", "white", "blue"]);
        // Through the UVs instead, the left half of the image lands on the
        // left two bands.
        let uv = project_bands(Projection::Uv, Axis::Z, points).unwrap();
        assert_eq!(&uv[..2], ["red", "red"]);
    }

    /// The placement moves, turns and sizes the projection in world space,
    /// and with repeat off nothing lands past the image's edges.
    #[test]
    fn placement_moves_the_projection_in_world_space() {
        let n = [0.0, 0.0, 1.0];
        // Every band is the same point: the image's top-left quarter.
        let p = ([-0.3, 0.2, 0.0], n);
        let points = [p; 4];
        let run = |adjust: fn(&mut Layer)| {
            project_bands_with(Projection::Planar, Axis::Z, points, adjust).map(|c| c[0])
        };
        let Some(start) = run(|_| {}) else {
            eprintln!("no Vulkan device; skipping");
            return;
        };
        assert_eq!(start, "red");
        // Slide the projection right past the point: it wraps round to the
        // right-hand edge of the image...
        assert_eq!(run(|l| l.projection_offset = [0.4, 0.0, 0.0]).unwrap(), "green");
        // ...or, placed once, misses it altogether.
        assert_eq!(
            run(|l| {
                l.projection_offset = [0.4, 0.0, 0.0];
                l.projection_repeat = false;
            })
            .unwrap(),
            "empty"
        );
        // A half turn about the throw axis puts the point bottom-right.
        assert_eq!(run(|l| l.projection_rotation = [0.0, 0.0, std::f32::consts::PI]).unwrap(), "white");
        // Half size: the point falls outside the image.
        assert_eq!(
            run(|l| {
                l.projection_scale = 0.5;
                l.projection_repeat = false;
            })
            .unwrap(),
            "empty"
        );
    }

    /// Triplanar picks the side by normal and flips it per facing, so the
    /// back reads un-mirrored from outside.
    #[test]
    fn triplanar_picks_and_flips_the_side_by_normal() {
        let points = [
            // Front: x = -0.2 is the viewer's left; y up is the top.
            ([-0.2, 0.2, 0.0], [0.0, 0.0, 1.0]),
            // Back, same point: seen from behind it is on the right.
            ([-0.2, 0.2, 0.0], [0.0, 0.0, -1.0]),
            // Side facing +X: -Z is the viewer's right.
            ([0.0, 0.2, -0.2], [1.0, 0.0, 0.0]),
            // Top: +Z is the image's bottom, +X its right.
            ([0.2, 0.0, 0.2], [0.0, 1.0, 0.0]),
        ];
        let Some(got) = project_bands(Projection::Triplanar, Axis::Y, points) else {
            eprintln!("no Vulkan device; skipping");
            return;
        };
        assert_eq!(got, ["red", "green", "green", "white"]);
    }

    /// Texels with no baked surface (between islands) take nothing from a
    /// projected layer, so dilation can pad islands from their own edges.
    #[test]
    fn projection_leaves_unbaked_texels_empty() {
        let Some(gpu) = gpu_context() else {
            eprintln!("no Vulkan device; skipping");
            return;
        };
        let res = 64;
        let mut stack = LayerStack::new(gpu.clone(), res);
        stack.dilate = false;
        stack.fill_layer_from_image(0, &solid(16, RED));
        let position = mesh_map(&stack, |x, _| [0.0, 0.0, 0.0, if x < 32 { 1.0 } else { 0.0 }]);
        let normal = mesh_map(&stack, |_, _| [0.0, 0.0, 1.0, 1.0]);

        // No source yet: a projected layer keeps its UV placement.
        stack.layers[0].projection = Projection::Triplanar;
        stack.composite_now();
        assert!(read_texel_at(&gpu, &stack.composite_basecolor, res, 48, 8)[3] > 0.9);

        stack.set_projection_source(position, normal, [0.0; 3], 1.0);
        stack.composite_now();
        assert!(read_texel_at(&gpu, &stack.composite_basecolor, res, 8, 8)[3] > 0.9);
        assert!(read_texel_at(&gpu, &stack.composite_basecolor, res, 48, 8)[3] < 0.01);
    }

    /// A document snapshot survives later edits: fills write fresh textures,
    /// so restoring the snapshot brings the old pixels back.
    #[test]
    fn state_snapshot_survives_fills() {
        let Some(gpu) = gpu_context() else {
            eprintln!("no Vulkan device; skipping");
            return;
        };
        let res = 64;
        let mut stack = LayerStack::new(gpu.clone(), res);
        stack.fill_active(Some([1.0, 0.0, 0.0]), None, None);
        stack.add_mask(0, 1.0);
        let before = stack.state();

        stack.fill_active(Some([0.0, 0.0, 1.0]), None, None);
        stack.invert_mask(0);
        stack.add_layer();
        assert!(!before.same_document(&stack.state()));

        stack.restore_state(before.clone());
        assert!(before.same_document(&stack.state()));
        assert_eq!(stack.layers.len(), 1);
        stack.composite_now();
        let c = read_texel(&gpu, &stack.composite_basecolor, res);
        assert!(c[0] > 0.9 && c[2] < 0.1, "red and an uninverted mask should be back, got {c:?}");
    }

    /// Selecting a layer is not an edit; changing one is.
    #[test]
    fn selection_is_not_part_of_the_document() {
        let Some(gpu) = gpu_context() else {
            eprintln!("no Vulkan device; skipping");
            return;
        };
        let mut stack = LayerStack::new(gpu, 16);
        stack.add_layer();
        let before = stack.state();
        stack.active = 0;
        assert!(before.same_document(&stack.state()));
        stack.layers[0].opacity = 0.5;
        assert!(!before.same_document(&stack.state()));
    }

    /// Stroke history names layers by id, so it still finds the painted
    /// layer after the stack is reordered.
    #[test]
    fn stroke_undo_follows_the_layer_through_a_reorder() {
        let Some(gpu) = gpu_context() else {
            eprintln!("no Vulkan device; skipping");
            return;
        };
        let res = 32;
        let mut stack = LayerStack::new(gpu.clone(), res);
        stack.fill_active(Some([1.0, 0.0, 0.0]), None, None);
        stack.add_layer(); // layer 1, on top, empty
        stack.active = 0;
        stack.begin_stroke(&[PaintChannel::BaseColor]);
        assert_eq!(stack.strokes_begun(), 1);
        // "Paint" blue over layer 0 in place, as a stroke does.
        let blue: Vec<f16> = (0..res * res).flat_map(|_| [0.0, 0.0, 1.0, 1.0].map(f16::from_f32)).collect();
        let target = stack.layers[0].basecolor.clone();
        stack.upload_texels(&target, blue);
        // Move the painted layer to the top.
        stack.move_layer(0, 1);
        assert!(stack.undo_stroke());
        stack.composite_now();
        let c = read_texel(&gpu, &stack.composite_basecolor, res);
        assert!(c[0] > 0.9 && c[2] < 0.1, "the stroke should be undone on the moved layer, got {c:?}");
    }

    /// A duplicate is an independent copy above the original: same pixels,
    /// same settings, selected, and later edits to the original don't reach it.
    #[test]
    fn duplicate_layer_copies_pixels_and_settings() {
        let Some(gpu) = gpu_context() else {
            eprintln!("no Vulkan device; skipping");
            return;
        };
        let res = 64;
        let mut stack = LayerStack::new(gpu.clone(), res);
        stack.fill_active(Some([1.0, 0.0, 0.0]), None, None);
        stack.add_mask(0, 1.0);
        stack.layers[0].opacity = 0.75;
        stack.layers[0].projection = Projection::Spherical;
        stack.duplicate_layer(0);

        assert_eq!(stack.layers.len(), 2);
        assert_eq!(stack.active, 1, "the copy is selected");
        let copy = &stack.layers[1];
        assert_eq!(copy.name, "Layer 1 copy");
        assert_eq!(copy.opacity, 0.75);
        assert_eq!(copy.projection, Projection::Spherical);
        assert!(copy.mask.is_some(), "the mask comes along");
        assert_eq!(copy.swatch, Some([1.0, 0.0, 0.0]));

        // Repaint the original blue: the copy, on top, still shows red.
        stack.active = 0;
        stack.fill_active(Some([0.0, 0.0, 1.0]), None, None);
        stack.layers[1].opacity = 1.0;
        stack.composite_now();
        let c = read_texel(&gpu, &stack.composite_basecolor, res);
        assert!(c[0] > 0.9 && c[2] < 0.1, "copy should be independent, got {c:?}");
    }

    /// An image fill keeps its alpha as coverage, so a cut-out wraps on as a
    /// decal over the layers below instead of a solid sheet.
    #[test]
    fn image_fill_keeps_alpha_as_coverage() {
        let Some(gpu) = gpu_context() else {
            eprintln!("no Vulkan device; skipping");
            return;
        };
        let res = 64;
        let mut stack = LayerStack::new(gpu.clone(), res);
        stack.fill_layer_from_image(0, &split_image(16, RED, [0, 0, 255, 0]));
        assert!(stack.layers[0].has_fill, "image fill must enable the placement controls");
        stack.composite_now();
        assert!(read_texel_at(&gpu, &stack.composite_basecolor, res, 8, 8)[3] > 0.9);
        assert!(read_texel_at(&gpu, &stack.composite_basecolor, res, 56, 8)[3] < 0.01);
    }
    /// Changing resolution keeps the paint and every setting: a filled layer
    /// reads the same colour at 2x and at a quarter (halving steps), masks
    /// follow, and never-painted channels stay placeholders.
    #[test]
    fn resampling_keeps_paint_at_the_new_size() {
        let Some(gpu) = gpu_context() else { return };
        let mut stack = LayerStack::new_lazy(gpu.clone(), 64);
        stack.fill_active(Some([0.25, 0.5, 1.0]), None, None);
        stack.add_mask(0, 1.0);
        stack.layers[0].name = "Ink".into();
        assert!(stack.is_placeholder(&stack.layers[0].metallic) || stack.layers[0].metallic.image().extent()[0] == 64);

        for res in [128, 16] {
            let out = stack.resampled(res);
            assert_eq!(out.resolution, res);
            assert!(out.is_allocated());
            let layer = &out.layers[0];
            assert_eq!(layer.name, "Ink");
            assert_eq!(layer.basecolor.image().extent()[0], res);
            assert_eq!(layer.mask.as_ref().unwrap().image().extent()[0], res);
            let texels = crate::export::read_texels(&gpu, &layer.basecolor).unwrap();
            let mid = texels[(res * res / 2 + res / 2) as usize];
            assert!((mid[0] - 0.25).abs() < 0.01 && (mid[2] - 1.0).abs() < 0.01, "{mid:?}");
            let mask = crate::export::read_texels(&gpu, layer.mask.as_ref().unwrap()).unwrap();
            assert!(mask.iter().all(|t| (t[0] - 1.0).abs() < 0.01));
        }

        // A blank lazy set resizes without allocating anything.
        let blank = LayerStack::new_lazy(gpu.clone(), 64).resampled(256);
        assert!(!blank.is_allocated());
        assert!(blank.is_placeholder(&blank.layers[0].basecolor));
    }

    /// Composites exist only for channels some layer has content in: a
    /// colour fill flattens into one full-size composite, and roughness gets
    /// its own only once a layer holds roughness. Height brings the normal
    /// map with it.
    #[test]
    fn composites_follow_the_channels_in_use() {
        let Some(gpu) = gpu_context() else { return };
        let res = 64;
        let mut stack = LayerStack::new_lazy(gpu.clone(), res);
        let full = |v: &Arc<ImageView>| v.image().extent()[0] == res;
        stack.fill_active(Some([1.0, 0.0, 0.0]), None, None);
        stack.composite_now();
        assert!(full(&stack.composite_basecolor));
        assert!(read_texel(&gpu, &stack.composite_basecolor, res)[0] > 0.9);
        for view in [&stack.composite_roughness, &stack.composite_metallic, &stack.composite_height, &stack.normal_map] {
            assert!(!full(view), "unused composites stay 1x1");
        }
        assert!(stack.set_resident(true), "the swap is reported for the path tracer");

        stack.fill_active(None, Some(0.25), None);
        stack.composite_now();
        assert!(full(&stack.composite_roughness));
        let rough = read_texel(&gpu, &stack.composite_roughness, res);
        assert!((rough[0] / rough[1] - 0.25).abs() < 0.01, "got {rough:?}");
        assert!(!full(&stack.composite_metallic));

        stack.begin_stroke(&[PaintChannel::Height]);
        stack.composite_now();
        assert!(full(&stack.composite_height) && full(&stack.composite_normal) && full(&stack.normal_map));
    }

    /// A lazy stack holds no full-size textures until it is used, composites
    /// nothing while blank, and its first edit gives the edited channel (and
    /// only that one) a real texture; saved states adopt the swapped textures
    /// so undo never sees a change.
    #[test]
    fn lazy_stack_allocates_on_first_use() {
        let Some(gpu) = gpu_context() else { return };
        let mut stack = LayerStack::new_lazy(gpu.clone(), 64);
        assert!(!stack.is_allocated());
        assert_eq!(stack.composite_basecolor.image().extent()[0], 1);
        assert!(stack.is_placeholder(&stack.layers[0].basecolor));
        stack.composite_now();
        assert!(!stack.is_allocated(), "a blank stack has nothing to flatten");

        let mut saved = stack.state();
        assert!(stack.ensure_allocated().is_empty(), "allocating composites leaves blank channels alone");
        assert!(stack.is_allocated());
        stack.begin_stroke(&[PaintChannel::BaseColor]);
        let upgrades = stack.take_upgrades();
        assert_eq!(upgrades.len(), 1);
        assert!(stack.is_placeholder(&stack.layers[0].roughness), "unpainted channels stay placeholders");
        assert_eq!(stack.composite_basecolor.image().extent()[0], 64);
        assert_eq!(stack.layers[0].basecolor.image().extent()[0], 64);
        assert!(!saved.layers[0].same_as(&stack.layers[0]));
        saved.adopt_upgrades(&upgrades);
        assert!(stack.state().same_document(&saved));
        assert!(stack.ensure_allocated().is_empty(), "second call is a no-op");
    }
}

mod dilate_cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/shaders/dilate.comp",
        define: [("FMT", "rgba16f")],
    }
}

/// `dilate_cs` for the two-channel (value, coverage) composites.
mod dilate_rg_cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/shaders/dilate.comp",
        define: [("FMT", "rg16f")],
    }
}
