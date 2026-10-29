//! Block compression for imported material maps.
//!
//! An FBX's 4K maps uploaded as RGBA8 cost 64 MiB each; Unity and every game
//! engine keep them block-compressed instead, which is the main reason the
//! same avatar fits there in a fraction of the VRAM. Maps are only ever
//! sampled on the GPU (paint, bakes and export use the CPU `ImageData`), so
//! they can be compressed losslessly as far as editing goes:
//!
//! - colour (base colour, emission): BC7 sRGB, 1 byte a texel, alpha kept;
//!   a cut-out material's base colour keeps its coverage down the mips;
//! - scalars (roughness, metallic, AO; the shader reads `.r`): BC4, 0.5;
//! - normals: BC5 (x, y), 1 byte a texel; the shader rebuilds z.
//!
//! Each map also gets a full mip chain (it had none), so a distant 4K map no
//! longer shimmers; that adds a third, and the total is still 4–8x smaller.
//! Encoding uses Intel's ISPC kernels, split into bands across threads.

use vulkano::format::Format;

use crate::scene::ImageData;

/// How a map is stored on the GPU.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MapEncoding {
    /// sRGB colour with alpha: BC7.
    Color,
    /// Colour whose alpha the shader cuts out at 0.5 (lace, hair cards,
    /// lashes): BC7 like `Color`, with each mip's alpha scaled so the same
    /// share of texels stays above the cut-off. Plain averaging dissolves
    /// fine cut-outs a little way from the camera: a lace panel's threads
    /// average below 0.5 and the whole panel vanishes. (Unity's "Mip Maps
    /// Preserve Coverage".)
    Cutout,
    /// One linear channel (red): BC4.
    Scalar,
    /// Tangent-space normal, x and y: BC5.
    Normal,
}

impl MapEncoding {
    pub fn format(self) -> Format {
        match self {
            MapEncoding::Color | MapEncoding::Cutout => Format::BC7_SRGB_BLOCK,
            MapEncoding::Scalar => Format::BC4_UNORM_BLOCK,
            MapEncoding::Normal => Format::BC5_UNORM_BLOCK,
        }
    }

    fn components(self) -> usize {
        match self {
            MapEncoding::Color | MapEncoding::Cutout => 4,
            MapEncoding::Scalar => 1,
            MapEncoding::Normal => 2,
        }
    }

    fn block_bytes(self) -> usize {
        match self {
            MapEncoding::Scalar => 8,
            MapEncoding::Color | MapEncoding::Cutout | MapEncoding::Normal => 16,
        }
    }
}

/// A compressed map: every mip level's blocks, largest first.
pub struct Compressed {
    pub format: Format,
    pub width: u32,
    pub height: u32,
    pub levels: Vec<Vec<u8>>,
}

impl Compressed {
    pub fn bytes(&self) -> usize {
        self.levels.iter().map(Vec::len).sum()
    }
}

/// Compress `image` (RGBA8) with a full mip chain.
pub fn compress(image: &ImageData, encoding: MapEncoding) -> Compressed {
    let (w, h) = (image.width.max(1), image.height.max(1));
    let n = encoding.components();
    // The channels the encoder takes, tightly packed.
    let mut level: Vec<u8> = if n == 4 {
        image.rgba.to_vec()
    } else {
        let mut packed = Vec::with_capacity(w as usize * h as usize * n);
        for p in image.rgba.chunks_exact(4) {
            packed.extend_from_slice(&p[..n]);
        }
        packed
    };
    let colour = n == 4;
    let opaque = colour && level.chunks_exact(4).all(|p| p[3] == 255);
    let coverage = (encoding == MapEncoding::Cutout && !opaque).then(|| alpha_coverage(&level, 1.0));
    let mut levels = Vec::new();
    let (mut lw, mut lh) = (w, h);
    loop {
        // The chain is built from unscaled levels; only the encoded copy of
        // each is rescaled, so the scales don't compound.
        match coverage {
            Some(target) if !levels.is_empty() => {
                let scaled = scale_alpha(&level, coverage_scale(&level, target));
                levels.push(encode_level(&scaled, lw, lh, encoding, opaque));
            }
            _ => levels.push(encode_level(&level, lw, lh, encoding, opaque)),
        }
        if lw == 1 && lh == 1 {
            break;
        }
        let (next, nw, nh) = downsample(&level, lw, lh, n, colour);
        level = next;
        lw = nw;
        lh = nh;
    }
    Compressed { format: encoding.format(), width: w, height: h, levels }
}

/// Half size, 2x2 box filter (edge texels repeat on odd sizes). Colour is
/// averaged in linear light so mips don't darken; alpha and data linearly.
fn downsample(src: &[u8], w: u32, h: u32, n: usize, srgb: bool) -> (Vec<u8>, u32, u32) {
    let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
    let lut = srgb_lut();
    let mut out = vec![0u8; nw as usize * nh as usize * n];
    for y in 0..nh {
        let (y0, y1) = ((y * 2).min(h - 1), (y * 2 + 1).min(h - 1));
        for x in 0..nw {
            let (x0, x1) = ((x * 2).min(w - 1), (x * 2 + 1).min(w - 1));
            let taps = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)];
            for c in 0..n {
                let at = |(tx, ty): (u32, u32)| src[(ty as usize * w as usize + tx as usize) * n + c];
                let value = if srgb && c < 3 {
                    let linear: f32 = taps.iter().map(|&t| lut[at(t) as usize]).sum::<f32>() / 4.0;
                    linear_to_srgb_u8(linear)
                } else {
                    let sum: u32 = taps.iter().map(|&t| at(t) as u32).sum();
                    ((sum + 2) / 4) as u8
                };
                out[(y as usize * nw as usize + x as usize) * n + c] = value;
            }
        }
    }
    (out, nw, nh)
}

/// The share of RGBA texels whose alpha, times `scale`, passes the 0.5
/// cut-off.
fn alpha_coverage(rgba: &[u8], scale: f32) -> f32 {
    let texels = rgba.len() / 4;
    let passing = rgba.chunks_exact(4).filter(|p| p[3] as f32 * scale >= 127.5).count();
    passing as f32 / texels.max(1) as f32
}

/// The alpha scale that brings `rgba`'s coverage to `target`: coverage
/// only grows with the scale, so bisect it, counting from a histogram.
fn coverage_scale(rgba: &[u8], target: f32) -> f32 {
    let mut histogram = [0usize; 256];
    rgba.chunks_exact(4).for_each(|p| histogram[p[3] as usize] += 1);
    let texels = (rgba.len() / 4).max(1) as f32;
    let coverage = |scale: f32| {
        let passing: usize = (1..256).filter(|&a| a as f32 * scale >= 127.5).map(|a| histogram[a]).sum();
        passing as f32 / texels
    };
    let (mut lo, mut hi) = (0.0f32, 255.0f32);
    for _ in 0..24 {
        let mid = 0.5 * (lo + hi);
        if coverage(mid) < target { lo = mid } else { hi = mid }
    }
    hi
}

fn scale_alpha(rgba: &[u8], scale: f32) -> Vec<u8> {
    let mut out = rgba.to_vec();
    for p in out.chunks_exact_mut(4) {
        p[3] = (p[3] as f32 * scale).round().min(255.0) as u8;
    }
    out
}

fn srgb_lut() -> &'static [f32; 256] {
    static LUT: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
    LUT.get_or_init(|| {
        std::array::from_fn(|i| {
            let v = i as f32 / 255.0;
            if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
        })
    })
}

fn linear_to_srgb_u8(v: f32) -> u8 {
    let s = if v <= 0.0031308 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 };
    (s.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// Blocks for one level. The kernels work on whole 4x4 blocks, so the level
/// is padded (edges repeated) to a multiple of four, then split into bands
/// of block rows that compress on separate threads.
fn encode_level(src: &[u8], w: u32, h: u32, encoding: MapEncoding, opaque: bool) -> Vec<u8> {
    let n = encoding.components();
    let (pw, ph) = (w.div_ceil(4) * 4, h.div_ceil(4) * 4);
    let padded: std::borrow::Cow<[u8]> = if (pw, ph) == (w, h) {
        src.into()
    } else {
        let mut out = vec![0u8; pw as usize * ph as usize * n];
        for y in 0..ph {
            let sy = y.min(h - 1) as usize;
            for x in 0..pw {
                let sx = x.min(w - 1) as usize;
                let (d, s) = ((y as usize * pw as usize + x as usize) * n, (sy * w as usize + sx) * n);
                out[d..d + n].copy_from_slice(&src[s..s + n]);
            }
        }
        out.into()
    };

    let block_rows = (ph / 4) as usize;
    let row_bytes = (pw / 4) as usize * encoding.block_bytes();
    let mut blocks = vec![0u8; block_rows * row_bytes];
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(block_rows).max(1);
    let rows_per_band = block_rows.div_ceil(threads);
    let stride = pw * n as u32;
    let settings = if opaque { intel_tex_2::bc7::opaque_fast_settings() } else { intel_tex_2::bc7::alpha_fast_settings() };
    std::thread::scope(|scope| {
        for (band, out) in blocks.chunks_mut(rows_per_band * row_bytes).enumerate() {
            let first_row = band * rows_per_band * 4;
            let rows = (out.len() / row_bytes * 4) as u32;
            let data = &padded[first_row * stride as usize..(first_row + rows as usize) * stride as usize];
            let settings = &settings;
            scope.spawn(move || match encoding {
                MapEncoding::Color | MapEncoding::Cutout => intel_tex_2::bc7::compress_blocks_into(
                    settings,
                    &intel_tex_2::RgbaSurface { data, width: pw, height: rows, stride },
                    out,
                ),
                MapEncoding::Scalar => {
                    intel_tex_2::bc4::compress_blocks_into(&intel_tex_2::RSurface { data, width: pw, height: rows, stride }, out)
                }
                MapEncoding::Normal => {
                    intel_tex_2::bc5::compress_blocks_into(&intel_tex_2::RgSurface { data, width: pw, height: rows, stride }, out)
                }
            });
        }
    });
    blocks
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 4]) -> ImageData {
        let mut rgba = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                rgba.extend(f(x, y));
            }
        }
        ImageData { width: w, height: h, rgba: rgba.into() }
    }

    /// Sizes and level counts follow the block format, odd sizes included.
    #[test]
    fn level_sizes_match_the_block_format() {
        let img = image(100, 37, |x, y| [x as u8, y as u8, 0, 255]);
        let c = compress(&img, MapEncoding::Color);
        assert_eq!(c.levels.len(), 7, "100x37 down to 1x1");
        assert_eq!(c.levels[0].len(), 25 * 10 * 16);
        assert_eq!(c.levels.last().unwrap().len(), 16);
        let s = compress(&img, MapEncoding::Scalar);
        assert_eq!(s.levels[0].len(), 25 * 10 * 8);
        let n = compress(&img, MapEncoding::Normal);
        assert_eq!(n.levels[0].len(), 25 * 10 * 16);
    }

    /// A lace-like cut-out (one opaque texel in four) keeps a quarter of
    /// its texels past the cut-off at every mip, where plain averaging
    /// leaves none from the first mip on.
    #[test]
    fn cutout_mips_keep_their_coverage() {
        let lace = image(64, 64, |x, y| [255, 255, 255, if x % 2 == 0 && y % 2 == 0 { 255 } else { 0 }]);
        let mut level = lace.rgba.to_vec();
        let (mut w, mut h) = (64, 64);
        let target = alpha_coverage(&level, 1.0);
        assert_eq!(target, 0.25);
        for _ in 0..4 {
            (level, w, h) = downsample(&level, w, h, 4, true);
            assert_eq!(alpha_coverage(&level, 1.0), 0.0, "averaged alone, the lace vanishes");
            let kept = alpha_coverage(&scale_alpha(&level, coverage_scale(&level, target)), 1.0);
            assert!(kept >= target, "{w}x{h}: coverage {kept}");
        }
    }

    /// The box filter averages colour in linear light: black and white
    /// make sRGB ~188, not 128.
    #[test]
    fn colour_mips_average_in_linear_light() {
        let (out, w, h) = downsample(&[0, 0, 0, 255, 255, 255, 255, 255], 2, 1, 4, true);
        assert_eq!((w, h), (1, 1));
        assert!((186..=190).contains(&out[0]), "got {}", out[0]);
        assert_eq!(out[3], 255);
        let (data, ..) = downsample(&[0, 255], 2, 1, 1, false);
        assert_eq!(data[0], 128);
    }
}
