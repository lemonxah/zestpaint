use std::path::Path;
use std::sync::Arc;

use anyhow::{Context as _, Result};
use half::f16;
use image::{ImageBuffer, Luma, LumaA, Rgba};
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyImageToBufferInfo,
    PrimaryCommandBufferAbstract,
};
use vulkano::image::view::ImageView;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::sync::GpuFuture;

use crate::environment::GpuContext;
use crate::layers::{Layer, LayerStack, MapKind, PaintChannel, ShaderMask, TextureSet, NORMAL_BIT};

/// Which layers a layer or layer-mask export covers.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LayerScope {
    All,
    Visible,
    /// The selected layer of each exported set.
    Active,
}

impl LayerScope {
    pub const ALL: [LayerScope; 3] = [LayerScope::All, LayerScope::Visible, LayerScope::Active];

    pub fn label(self) -> &'static str {
        match self {
            LayerScope::All => "All layers",
            LayerScope::Visible => "Visible layers",
            LayerScope::Active => "Selected layer",
        }
    }

    /// Stack slots in scope, bottom to top.
    pub fn layers(self, stack: &LayerStack) -> Vec<usize> {
        match self {
            LayerScope::All => (0..stack.layers.len()).collect(),
            LayerScope::Visible => (0..stack.layers.len()).filter(|&i| stack.layers[i].visible).collect(),
            LayerScope::Active => vec![stack.active],
        }
    }
}

/// What an export writes, besides which maps and sets.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ExportContents {
    /// The flattened stack: one PNG per map, what a game engine takes.
    pub flattened: bool,
    /// One greyscale PNG per shader mask in use.
    pub shader_masks: bool,
    /// Each layer on its own: one PNG per map the layer has content in, with
    /// its coverage in alpha, so it can be imported back as a layer.
    pub layers: bool,
    /// Each layer's mask as a greyscale PNG.
    pub layer_masks: bool,
    pub scope: LayerScope,
    /// Shrink every written PNG to at most this many px a side — paint at
    /// 4K/8K for detail, ship 2K. None keeps the paint resolution.
    pub size: Option<u32>,
}

/// Note shown beside per-layer files in the export preview.
pub const LAYER_NOTE: &str = "layer · coverage in alpha";
/// Note shown beside layer-mask files in the export preview.
pub const LAYER_MASK_NOTE: &str = "layer mask";
/// Note shown beside shader-mask files in the export preview.
pub const SHADER_MASK_NOTE: &str = "shader mask";
pub const MATCAP_NOTE: &str = "matcap image · sRGB";

/// File stems of the `n`th matcap layer (0 = bottom) — mask and image —
/// numbered as Poiyomi numbers its matcaps: matcap, matcap2, matcap3…
fn matcap_files(prefix: &str, n: usize) -> (String, String) {
    let stem = if n == 0 { "matcap".to_string() } else { format!("matcap{}", n + 1) };
    (format!("{prefix}_{stem}_mask.png"), format!("{prefix}_{stem}.png"))
}

/// Export the chosen maps of the chosen texture sets, named after `name`
/// (e.g. `MyAvatar_color.png`; see `set_prefixes`). `sets` and `include`
/// run in parallel; a set whose flag is false is skipped. Returns the file
/// names written.
pub fn export_all(
    gpu: &GpuContext,
    sets: &mut [TextureSet],
    include: &[bool],
    name: &str,
    maps: &[MapKind],
    contents: &ExportContents,
    dir: &Path,
) -> Result<Vec<String>> {
    let prefixes = set_prefixes(sets, include, name);
    let mut written = Vec::new();
    for ((set, _), prefix) in sets.iter_mut().zip(include).zip(&prefixes).filter(|((_, on), _)| **on) {
        if contents.flattened {
            let baked_ao = set.baked_ao.clone();
            written.extend(export_maps(
                gpu,
                &mut set.layers,
                baked_ao.as_ref(),
                maps,
                &prefix,
                dir,
            )?);
        }
        if contents.shader_masks {
            written.extend(export_shader_masks(gpu, &mut set.layers, &prefix, dir)?);
        }
        if contents.layers || contents.layer_masks {
            for index in contents.scope.layers(&set.layers) {
                written.extend(export_layer(
                    gpu,
                    set,
                    prefix,
                    index,
                    maps,
                    contents.layers,
                    contents.layer_masks,
                    dir,
                )?);
            }
        }
    }
    if let Some(size) = contents.size {
        for file in &written {
            downscale_png(&dir.join(file), size).with_context(|| format!("downscaling {file}"))?;
        }
    }
    Ok(written)
}

/// Shrink a written PNG to `size` px a side (Lanczos), keeping its colour
/// type. Alpha-aware: colour is premultiplied while filtering, so the black
/// under uncovered texels doesn't bleed dark fringes into painted edges.
pub fn downscale_png(path: &Path, size: u32) -> Result<()> {
    let img = image::open(path)?;
    let (w, h) = (img.width(), img.height());
    if w.max(h) <= size {
        return Ok(());
    }
    let scale = size as f64 / w.max(h) as f64;
    let (nw, nh) = (((w as f64 * scale).round() as u32).max(1), ((h as f64 * scale).round() as u32).max(1));
    let colour = img.color();
    let alpha = colour.has_alpha();
    let mut texels = img.to_rgba32f();
    if alpha {
        for p in texels.pixels_mut() {
            let a = p.0[3];
            p.0[0] *= a;
            p.0[1] *= a;
            p.0[2] *= a;
        }
    }
    let mut small = image::imageops::resize(&texels, nw, nh, image::imageops::FilterType::Lanczos3);
    for p in small.pixels_mut() {
        let a = p.0[3].clamp(0.0, 1.0);
        for c in &mut p.0[..3] {
            let straight = if !alpha { *c } else if a > 1e-6 { *c / a } else { 0.0 };
            *c = straight.clamp(0.0, 1.0);
        }
        p.0[3] = a;
    }
    let small = image::DynamicImage::ImageRgba32F(small);
    let out: image::DynamicImage = match colour {
        image::ColorType::L8 => small.to_luma8().into(),
        image::ColorType::La8 => small.to_luma_alpha8().into(),
        image::ColorType::Rgb8 => small.to_rgb8().into(),
        image::ColorType::L16 => small.to_luma16().into(),
        image::ColorType::La16 => small.to_luma_alpha16().into(),
        image::ColorType::Rgb16 => small.to_rgb16().into(),
        image::ColorType::Rgba16 => small.to_rgba16().into(),
        _ => small.to_rgba8().into(),
    };
    out.save(path)?;
    Ok(())
}

/// File-name stem for one layer's exports: `<set>_L03_Rust`. The number is
/// the layer's position from the bottom, so a folder sorts in stack order.
pub fn layer_stem(prefix: &str, index: usize, layer: &Layer) -> String {
    let name: String = layer
        .name
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '-' { c } else { '_' })
        .collect();
    format!("{prefix}_L{:02}_{name}", index + 1)
}

/// A layer's texture for one map, if the layer has ever held content there.
/// AO has no layer channel.
fn layer_channel(layer: &Layer, kind: MapKind) -> Option<&Arc<ImageView>> {
    let (bit, view) = match kind {
        MapKind::BaseColor => (PaintChannel::BaseColor.bit(), &layer.basecolor),
        MapKind::Roughness => (PaintChannel::Roughness.bit(), &layer.roughness),
        MapKind::Metallic => (PaintChannel::Metallic.bit(), &layer.metallic),
        MapKind::Height => (PaintChannel::Height.bit(), &layer.height),
        MapKind::Normal => (NORMAL_BIT, &layer.normal),
        MapKind::Emissive => (PaintChannel::Emissive.bit(), &layer.emissive),
        MapKind::Opacity => (PaintChannel::Opacity.bit(), &layer.opacity_map),
        MapKind::Ao => return None,
    };
    // A saved-blank channel comes back as the 1x1 placeholder even if it was
    // once used; there is nothing in it to export.
    (layer.used & bit != 0 && view.image().extent()[0] > 1).then_some(view)
}

/// The files `export_layer` would write.
fn planned_layer_files(
    set: &TextureSet,
    prefix: &str,
    index: usize,
    maps: &[MapKind],
    channels: bool,
    mask: bool,
    files: &mut Vec<(String, &'static str)>,
) {
    let Some(layer) = set.layers.layers.get(index) else { return };
    let stem = layer_stem(prefix, index, layer);
    if channels {
        for &kind in MapKind::ALL.iter().filter(|k| maps.contains(k)) {
            if layer_channel(layer, kind).is_some() {
                files.push((format!("{stem}_{}.png", kind.suffix()), LAYER_NOTE));
            }
        }
    }
    if mask && layer.mask.is_some() {
        files.push((format!("{stem}_mask.png"), LAYER_MASK_NOTE));
    }
}

/// Write one layer on its own: each chosen map it has content in
/// (`channels`), and its mask (`mask`). Colour space and alpha follow the
/// flattened export, with the layer's own coverage as alpha — exactly what
/// Import maps expects, so a layer exported this way imports back as one.
pub fn export_layer(
    gpu: &GpuContext,
    set: &TextureSet,
    prefix: &str,
    index: usize,
    maps: &[MapKind],
    channels: bool,
    mask: bool,
    dir: &Path,
) -> Result<Vec<String>> {
    let Some(layer) = set.layers.layers.get(index) else { return Ok(Vec::new()) };
    let res = set.layers.resolution;
    let stem = layer_stem(prefix, index, layer);
    let mut written = Vec::new();
    if channels {
        for &kind in MapKind::ALL.iter().filter(|k| maps.contains(k)) {
            let Some(view) = layer_channel(layer, kind) else { continue };
            let data = read_texels(gpu, view)?;
            let file = format!("{stem}_{}.png", kind.suffix());
            let path = dir.join(&file);
            let saved = match kind {
                MapKind::BaseColor | MapKind::Emissive => colour_image(&data, res).save(&path),
                MapKind::Height => {
                    scalar_image(&data, res, 0.0, |v| v.clamp(-1.0, 1.0) * 0.5 + 0.5).save(&path)
                }
                MapKind::Normal => {
                    // Premultiplied xyz·α: unpremultiply, keep α as coverage.
                    let mut img = ImageBuffer::<Rgba<u8>, Vec<u8>>::new(res, res);
                    for (px, &[x, y, z, a]) in img.pixels_mut().zip(&data) {
                        let encode = |v: f32| ((v.clamp(-1.0, 1.0) * 0.5 + 0.5) * 255.0).round() as u8;
                        *px = if a > 1e-5 {
                            Rgba([encode(x / a), encode(y / a), encode(z / a), unit_byte(a)])
                        } else {
                            Rgba([128, 128, 255, 0])
                        };
                    }
                    img.save(&path)
                }
                _ => scalar_image(&data, res, 0.0, |v| v).save(&path),
            };
            saved.with_context(|| format!("failed to write {}", path.display()))?;
            written.push(file);
        }
    }
    if mask && let Some(view) = &layer.mask {
        written.push(write_mask(gpu, view, res, &format!("{stem}_mask.png"), dir)?);
    }
    Ok(written)
}

/// A layer mask as greyscale: white shows the layer. The compositor reads
/// the mask's red channel, so that is what is written.
fn write_mask(gpu: &GpuContext, view: &Arc<ImageView>, res: u32, file: &str, dir: &Path) -> Result<String> {
    let texels = read_texels(gpu, view)?;
    let mut img = image::GrayImage::new(res, res);
    for (px, t) in img.pixels_mut().zip(&texels) {
        px.0 = [unit_byte(t[0])];
    }
    let path = dir.join(file);
    img.save(&path).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(file.to_string())
}

/// Premultiplied linear colour → straight sRGB with coverage in alpha.
fn colour_image(data: &[[f32; 4]], res: u32) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    let mut img = ImageBuffer::<Rgba<u8>, Vec<u8>>::new(res, res);
    for (px, &[r, g, b, a]) in img.pixels_mut().zip(data) {
        let (r, g, b) = if a > 1e-5 { (r / a, g / a, b / a) } else { (0.0, 0.0, 0.0) };
        *px = Rgba([encode_srgb(r), encode_srgb(g), encode_srgb(b), unit_byte(a)]);
    }
    img
}

/// Premultiplied (value·α, α) → greyscale value with coverage in alpha.
/// `empty` is the value written where nothing is covered.
fn scalar_image(
    data: &[[f32; 4]],
    res: u32,
    empty: f32,
    encode: fn(f32) -> f32,
) -> ImageBuffer<LumaA<u8>, Vec<u8>> {
    let mut img = ImageBuffer::<LumaA<u8>, Vec<u8>>::new(res, res);
    for (px, &[v, a, _, _]) in img.pixels_mut().zip(data) {
        let value = if a > 1e-5 { v / a } else { empty };
        *px = LumaA([unit_byte(encode(value)), unit_byte(a)]);
    }
    img
}

fn unit_byte(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// Letters, digits, `-` and `_` kept; anything else becomes `_`.
fn safe_file_name(text: &str) -> String {
    text.trim()
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect()
}

/// File-name prefix per texture set (parallel to `sets`) for an export
/// named `name`. One set exported: just the name, `MyAvatar_color.png`.
/// Several: the material is added so they don't overwrite each other,
/// `MyAvatar_Skin_color.png` — and the object too when two exported sets
/// share a material name. An empty name falls back to `Object_Material`.
pub fn set_prefixes(sets: &[TextureSet], include: &[bool], name: &str) -> Vec<String> {
    let name = safe_file_name(name);
    let included: Vec<&TextureSet> =
        sets.iter().zip(include).filter(|(_, on)| **on).map(|(s, _)| s).collect();
    let shared_material = |set: &TextureSet| {
        included.iter().filter(|s| s.material == set.material).count() > 1
    };
    sets.iter()
        .map(|set| {
            let object_material = format!("{}_{}", safe_file_name(&set.object), safe_file_name(&set.material));
            if name.is_empty() {
                object_material
            } else if included.len() <= 1 {
                name.clone()
            } else if shared_material(set) {
                format!("{name}_{object_material}")
            } else {
                format!("{name}_{}", safe_file_name(&set.material))
            }
        })
        .collect()
}

/// The files `export_all` would write, without writing anything: what the
/// export dialog previews and counts. Each name comes with what is in it.
pub fn planned_files(
    sets: &[TextureSet],
    include: &[bool],
    name: &str,
    maps: &[MapKind],
    contents: &ExportContents,
) -> Vec<(String, &'static str)> {
    let prefixes = set_prefixes(sets, include, name);
    let mut files = Vec::new();
    for ((set, _), prefix) in sets.iter().zip(include).zip(&prefixes).filter(|((_, on), _)| **on) {
        for &kind in MapKind::ALL.iter().filter(|k| contents.flattened && maps.contains(k)) {
            if kind == MapKind::Ao && set.baked_ao.is_none() {
                continue;
            }
            let note = match kind {
                MapKind::BaseColor | MapKind::Emissive => "sRGB · coverage in alpha",
                MapKind::Roughness | MapKind::Metallic | MapKind::Opacity => "greyscale · coverage in alpha",
                MapKind::Height => "greyscale · 0.5 neutral",
                MapKind::Normal => "tangent space",
                MapKind::Ao => "greyscale · baked",
            };
            files.push((format!("{prefix}_{}.png", kind.suffix()), note));
        }
        if contents.shader_masks {
            for kind in set.layers.shader_mask_kinds().into_iter().filter(|k| *k != ShaderMask::Matcap) {
                files.push((format!("{prefix}_{}.png", kind.suffix()), SHADER_MASK_NOTE));
            }
            for (n, layer) in set.layers.layers.iter().filter(|l| l.is_matcap()).enumerate() {
                let (mask, image) = matcap_files(prefix, n);
                files.push((mask, SHADER_MASK_NOTE));
                if layer.matcap.as_ref().is_some_and(|m| m.image.is_some()) {
                    files.push((image, MATCAP_NOTE));
                }
            }
        }
        if contents.layers || contents.layer_masks {
            for index in contents.scope.layers(&set.layers) {
                planned_layer_files(set, prefix, index, maps, contents.layers, contents.layer_masks, &mut files);
            }
        }
    }
    files
}

/// One greyscale PNG per shader mask the stack has, and per matcap layer
/// its mask and matcap image. The composite is premultiplied over black, so
/// R already is the mask value.
fn export_shader_masks(
    gpu: &GpuContext,
    layers: &mut LayerStack,
    prefix: &str,
    dir: &Path,
) -> Result<Vec<String>> {
    layers.composite_now();
    let res = layers.resolution;
    let mut written = Vec::new();
    let mut save_mask = |view: &Arc<ImageView>, file: String| -> Result<()> {
        let texels = read_texels(gpu, view)?;
        let mut img = image::GrayImage::new(res, res);
        for (px, t) in img.pixels_mut().zip(&texels) {
            px.0 = [(t[0].clamp(0.0, 1.0) * 255.0).round() as u8];
        }
        img.save(dir.join(&file)).with_context(|| format!("writing {file}"))?;
        written.push(file);
        Ok(())
    };
    for kind in layers.shader_mask_kinds().into_iter().filter(|k| *k != ShaderMask::Matcap) {
        let Some(view) = layers.shader_mask(kind) else { continue };
        save_mask(view, format!("{prefix}_{}.png", kind.suffix()))?;
    }
    let mut images = Vec::new();
    for (n, layer) in layers.layers.iter().filter(|l| l.is_matcap()).enumerate() {
        let (mask, image) = matcap_files(prefix, n);
        if let Some(view) = layers.matcap_mask(layer.id) {
            save_mask(view, mask)?;
        }
        if let Some((data, _)) = layer.matcap.as_ref().and_then(|m| m.image.as_ref()) {
            let img = image::RgbaImage::from_raw(data.width, data.height, data.rgba.to_vec())
                .context("matcap image has the wrong size")?;
            img.save(dir.join(&image)).with_context(|| format!("writing {image}"))?;
            images.push(image);
        }
    }
    written.extend(images);
    Ok(written)
}

/// Flatten one layer stack and write the requested texture maps as PNGs.
fn export_maps(
    gpu: &GpuContext,
    layers: &mut LayerStack,
    baked_ao: Option<&Arc<ImageView>>,
    maps: &[MapKind],
    prefix: &str,
    dir: &Path,
) -> Result<Vec<String>> {
    layers.composite_now();
    let res = layers.resolution;
    // A set never activated or painted has no full-size composites (see
    // `LayerStack::new_lazy`), and a channel no layer uses keeps a 1x1 blank
    // one: either exports as unpainted.
    let read = |view: &Arc<ImageView>| -> Result<Vec<[f32; 4]>> {
        if view.image().extent()[0] == res { read_texels(gpu, view) } else { Ok(vec![[0.0; 4]; (res * res) as usize]) }
    };

    let mut written = Vec::new();
    for &kind in MapKind::ALL.iter().filter(|k| maps.contains(k)) {
        let file = format!("{prefix}_{}.png", kind.suffix());
        let path = dir.join(&file);
        let saved = match kind {
            // Unpremultiply, encode linear → sRGB, keep coverage as alpha.
            MapKind::BaseColor | MapKind::Emissive => {
                let view = if kind == MapKind::BaseColor {
                    &layers.composite_basecolor
                } else {
                    &layers.composite_emissive
                };
                let data = read(view)?;
                let mut img = ImageBuffer::<Rgba<u8>, Vec<u8>>::new(res, res);
                for (i, px) in img.pixels_mut().enumerate() {
                    let [r, g, b, a] = data[i];
                    let (r, g, b) =
                        if a > 1e-5 { (r / a, g / a, b / a) } else { (0.0, 0.0, 0.0) };
                    *px = Rgba([
                        encode_srgb(r),
                        encode_srgb(g),
                        encode_srgb(b),
                        (a.clamp(0.0, 1.0) * 255.0).round() as u8,
                    ]);
                }
                img.save(&path)
            }
            // Scalar channels: grayscale value + coverage alpha. Height is
            // signed, remapped so 0.5 is neutral; uncovered opacity is opaque.
            MapKind::Roughness | MapKind::Metallic | MapKind::Height | MapKind::Opacity => {
                let (view, empty, encode): (_, f32, fn(f32) -> f32) = match kind {
                    MapKind::Roughness => (&layers.composite_roughness, 0.0, |v| v),
                    MapKind::Metallic => (&layers.composite_metallic, 0.0, |v| v),
                    MapKind::Height => {
                        (&layers.composite_height, 0.0, |v| v.clamp(-1.0, 1.0) * 0.5 + 0.5)
                    }
                    _ => (&layers.composite_opacity, 1.0, |v| v),
                };
                let data = read(view)?;
                let mut img = ImageBuffer::<LumaA<u8>, Vec<u8>>::new(res, res);
                for (i, px) in img.pixels_mut().enumerate() {
                    let [v, a, _, _] = data[i];
                    let value = if a > 1e-5 { v / a } else { empty };
                    *px = LumaA([
                        (encode(value).clamp(0.0, 1.0) * 255.0).round() as u8,
                        (a.clamp(0.0, 1.0) * 255.0).round() as u8,
                    ]);
                }
                img.save(&path)
            }
            // Tangent-space normal map (flat normal where nothing was painted).
            MapKind::Normal => {
                let data = read(&layers.normal_map)?;
                let mut img = ImageBuffer::<Rgba<u8>, Vec<u8>>::new(res, res);
                for (i, px) in img.pixels_mut().enumerate() {
                    let [x, y, z, w] = data[i];
                    let encode =
                        |v: f32| ((v.clamp(-1.0, 1.0) * 0.5 + 0.5) * 255.0).round() as u8;
                    *px = if w > 1e-3 {
                        Rgba([encode(x), encode(y), encode(z), 255])
                    } else {
                        Rgba([128, 128, 255, 255])
                    };
                }
                img.save(&path)
            }
            // Baked ambient occlusion, if present (white where no data).
            MapKind::Ao => {
                let Some(ao) = baked_ao else { continue };
                let data = read_texels(gpu, ao)?;
                let mut img = ImageBuffer::<Luma<u8>, Vec<u8>>::new(res, res);
                for (i, px) in img.pixels_mut().enumerate() {
                    let [v, valid, _, _] = data[i];
                    let value = if valid > 0.5 { v } else { 1.0 };
                    *px = Luma([(value.clamp(0.0, 1.0) * 255.0).round() as u8]);
                }
                img.save(&path)
            }
        };
        saved.with_context(|| format!("failed to write {}", path.display()))?;
        written.push(file);
    }

    Ok(written)
}

/// Read a half-float image back to the CPU as f32 quads. Two-channel
/// (RG16F) layer channels come back as (r, g, 0, 0).
pub(crate) fn read_texels(gpu: &GpuContext, view: &Arc<ImageView>) -> Result<Vec<[f32; 4]>> {
    let [w, h, _] = view.image().extent();
    let components = match view.image().format() {
        vulkano::format::Format::R16G16B16A16_SFLOAT => 4,
        vulkano::format::Format::R16G16_SFLOAT => 2,
        other => anyhow::bail!("cannot export a texture in format {other:?}"),
    };
    let count = w as u64 * h as u64 * components as u64;
    let buffer: Subbuffer<[u16]> = Buffer::new_slice(
        gpu.memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        count,
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

    let guard = buffer.read()?;
    let f = |bits: u16| f16::from_bits(bits).to_f32();
    Ok(guard
        .chunks_exact(components)
        .map(|c| match c {
            [r, g, b, a] => [f(*r), f(*g), f(*b), f(*a)],
            [r, g] => [f(*r), f(*g), 0.0, 0.0],
            _ => unreachable!(),
        })
        .collect())
}

fn encode_srgb(v: f32) -> u8 {
    let v = v.clamp(0.0, 1.0);
    let s = if v <= 0.0031308 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0).round() as u8
}

#[cfg(test)]
mod tests {
    /// Export downscale: right size, same colour type, and painted edges keep
    /// their colour (no dark fringe from the transparent black around them).
    #[test]
    fn downscaled_export_keeps_edge_colour() {
        let dir = std::env::temp_dir().join(format!("zp-downscale-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ink.png");
        let img = image::RgbaImage::from_fn(64, 64, |x, y| {
            if (16..48).contains(&x) && (16..48).contains(&y) { image::Rgba([200, 30, 30, 255]) } else { image::Rgba([0, 0, 0, 0]) }
        });
        img.save(&path).unwrap();
        downscale_png(&path, 16).unwrap();
        let small = image::open(&path).unwrap();
        assert_eq!((small.width(), small.height()), (16, 16));
        assert_eq!(small.color(), image::ColorType::Rgba8);
        let small = small.to_rgba8();
        for p in small.pixels().filter(|p| p.0[3] > 40) {
            assert!(p.0[0] > 170 && p.0[1] < 60, "edge texel darkened or shifted: {:?}", p.0);
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    use super::*;

    /// A layer's own maps and its mask export as previewed, with the
    /// layer's values and coverage, and import back as an equal layer.
    #[test]
    fn layers_and_masks_export_as_planned_and_import_back() {
        let Some(gpu) = crate::environment::test_gpu::context() else {
            eprintln!("no Vulkan device; skipping");
            return;
        };
        let res = 32;
        let mut sets = vec![TextureSet::new("Body".into(), "Skin".into(), gpu.clone(), res)];
        let stack = &mut sets[0].layers;
        stack.fill_active(Some([0.25, 0.5, 0.75]), Some(0.4), None);
        stack.add_mask(0, 0.5);
        stack.layers[0].name = "Base coat".into();
        stack.add_layer(); // empty: writes nothing

        let contents = ExportContents {
            size: None,
            flattened: false,
            shader_masks: false,
            layers: true,
            layer_masks: true,
            scope: LayerScope::All,
        };
        let planned: Vec<String> =
            planned_files(&sets, &[true], "MyAvatar", &MapKind::ALL, &contents).into_iter().map(|(f, _)| f).collect();
        assert_eq!(
            planned,
            [
                "MyAvatar_L01_Base_coat_color.png",
                "MyAvatar_L01_Base_coat_roughness.png",
                "MyAvatar_L01_Base_coat_mask.png",
            ]
        );

        let dir = std::env::temp_dir().join(format!("zestsurface-layer-export-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let written = export_all(&gpu, &mut sets, &[true], "MyAvatar", &MapKind::ALL, &contents, &dir).unwrap();
        assert_eq!(written, planned);

        let colour = image::open(dir.join(&planned[0])).unwrap().to_rgba8();
        assert_eq!(colour.get_pixel(5, 5).0, [encode_srgb(0.25), encode_srgb(0.5), encode_srgb(0.75), 255]);
        let rough = image::open(dir.join(&planned[1])).unwrap().to_luma_alpha8();
        assert_eq!(rough.get_pixel(5, 5).0, [unit_byte(0.4), 255]);
        let mask = image::open(dir.join(&planned[2])).unwrap().to_luma8();
        assert_eq!(mask.get_pixel(5, 5).0, [128]);

        // Back in: the imported layer's roughness reads as what was exported.
        let image = crate::library::load_map(&dir.join(&planned[1])).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        let stack = &mut sets[0].layers;
        assert!(stack.import_map_layer("again".into(), MapKind::Roughness, &image));
        let again = read_texels(&gpu, &stack.active_layer().roughness).unwrap();
        assert!((again[5 * res as usize + 5][0] - 0.4).abs() < 0.01);

        // And a mask image loads back onto a layer.
        let mask_image = grey_image(res, 200);
        stack.set_mask_from_image(0, &mask_image);
        let m = read_texels(&gpu, stack.layers[0].mask.as_ref().unwrap()).unwrap();
        assert!((m[0][0] - 200.0 / 255.0).abs() < 0.01);
    }

    /// One set: just the name. Several: the material after it, and the
    /// object too where two sets share a material. No name: Object_Material.
    #[test]
    fn export_names_follow_the_name_field() {
        let Some(gpu) = crate::environment::test_gpu::context() else {
            eprintln!("no Vulkan device; skipping");
            return;
        };
        let set = |object: &str, material: &str| TextureSet::new(object.into(), material.into(), gpu.clone(), 4);
        let mut sets = vec![set("Body", "Skin"), set("Shirt", "Cloth"), set("Pants", "Cloth")];
        sets[0].layers.add_shader_mask_layer(crate::layers::ShaderMask::Matcap);
        let flat = ExportContents {
            size: None,
            flattened: true,
            shader_masks: true,
            layers: false,
            layer_masks: false,
            scope: LayerScope::All,
        };
        let maps = [MapKind::BaseColor, MapKind::Normal];
        let names = |include: &[bool], name: &str| -> Vec<String> {
            planned_files(&sets, include, name, &maps, &flat).into_iter().map(|(f, _)| f).collect()
        };
        assert_eq!(
            names(&[true, false, false], "My Avatar"),
            ["My_Avatar_color.png", "My_Avatar_normals.png", "My_Avatar_matcap_mask.png"]
        );
        assert_eq!(
            names(&[true, true, true], "Av"),
            [
                "Av_Skin_color.png",
                "Av_Skin_normals.png",
                "Av_Skin_matcap_mask.png",
                "Av_Shirt_Cloth_color.png",
                "Av_Shirt_Cloth_normals.png",
                "Av_Pants_Cloth_color.png",
                "Av_Pants_Cloth_normals.png",
            ]
        );
        assert_eq!(names(&[false, true, false], " "), ["Shirt_Cloth_color.png", "Shirt_Cloth_normals.png"]);
    }

    fn grey_image(size: u32, v: u8) -> crate::scene::ImageData {
        crate::scene::ImageData { width: size, height: size, rgba: [v, v, v, 255].repeat((size * size) as usize).into() }
    }
}
