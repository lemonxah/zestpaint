//! Matcaps: a "material capture" sphere image looked up by the view-space
//! normal and blended over the shaded surface, as Poiyomi's matcaps do, so a
//! VRChat look can be built while painting. Each matcap lives on a matcap
//! layer (`Layer::matcap`), whose painted greyscale says where it shows.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use vulkano::format::Format;
use vulkano::image::view::ImageView;

use crate::environment::GpuContext;
use crate::scene::ImageData;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub enum MatcapBlend {
    #[default]
    Add,
    Multiply,
    Replace,
    Screen,
}

impl MatcapBlend {
    pub const ALL: [MatcapBlend; 4] = [MatcapBlend::Add, MatcapBlend::Multiply, MatcapBlend::Replace, MatcapBlend::Screen];

    pub fn label(self) -> &'static str {
        match self {
            MatcapBlend::Add => "Add",
            MatcapBlend::Multiply => "Multiply",
            MatcapBlend::Replace => "Replace",
            MatcapBlend::Screen => "Screen",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|b| b.label().eq_ignore_ascii_case(s))
    }

    /// The mode number the viewport shader switches on.
    pub fn shader_id(self) -> f32 {
        self as u8 as f32
    }
}

/// The matcap of one matcap layer. The layer's visibility turns it on and
/// off; its opacity scales the mask.
#[derive(Clone)]
pub struct Matcap {
    /// Built-in name or file name, for the UI.
    pub name: String,
    /// The sRGB image (saved with the project) and its GPU copy.
    pub image: Option<(ImageData, Arc<ImageView>)>,
    pub blend: MatcapBlend,
    pub strength: f32,
}

impl Default for Matcap {
    fn default() -> Self {
        Self { name: String::new(), image: None, blend: MatcapBlend::Add, strength: 1.0 }
    }
}

impl Matcap {
    /// Same settings and the same image (by identity) — for undo history.
    pub fn same_as(&self, other: &Matcap) -> bool {
        self.name == other.name
            && self.blend == other.blend
            && self.strength == other.strength
            && match (&self.image, &other.image) {
                (Some((a, _)), Some((b, _))) => Arc::ptr_eq(&a.rgba, &b.rgba),
                (None, None) => true,
                _ => false,
            }
    }

    /// Whether the viewport should draw it.
    pub fn active(&self) -> bool {
        self.image.is_some() && self.strength > 0.0
    }

    /// Use `image` (sRGB) as this slot's matcap, uploading it.
    pub fn set_image(&mut self, gpu: &GpuContext, name: String, image: ImageData) {
        let view = crate::renderer::upload_texture(
            &gpu.queue,
            &gpu.memory_allocator,
            &gpu.command_buffer_allocator,
            image.width,
            image.height,
            &image.rgba,
            Format::R8G8B8A8_SRGB,
        );
        self.name = name;
        self.image = Some((image, view));
    }
}

/// Built-in matcaps, generated on demand: (name, what it is for, the blend
/// it is meant for).
pub const BUILTINS: [(&str, &str, MatcapBlend); 15] = [
    ("Shine", "soft highlight on black — glossy skin, lips", MatcapBlend::Add),
    ("Rim light", "bright edges on black — a backlit outline", MatcapBlend::Add),
    ("Chrome", "sky/ground reflection with a horizon line — metal", MatcapBlend::Multiply),
    ("Pearl", "pastel iridescence — pearl, holo, opal", MatcapBlend::Multiply),
    ("Satin", "soft sheen — satin, silk, leather", MatcapBlend::Multiply),
    ("Clay", "warm diffuse clay — checks forms without textures", MatcapBlend::Replace),
    ("Gold", "warm sky/ground reflection — gold jewellery, buckles, chains", MatcapBlend::Multiply),
    ("Copper", "rosy metal reflection — copper, rose gold, bronze", MatcapBlend::Multiply),
    ("Toon shine", "one hard-edged highlight on black — anime hair and eyes", MatcapBlend::Add),
    ("Latex", "sharp highlight and softbox streak on black — latex, PVC, patent leather", MatcapBlend::Add),
    ("Wet", "several tiny sharp glints on black — wet skin, lip gloss, eyes", MatcapBlend::Add),
    ("Holo", "rainbow bands that shift with the view — holographic vinyl, oil film", MatcapBlend::Screen),
    ("Glitter", "scattered sparkles on black that twinkle as it turns", MatcapBlend::Add),
    ("Skin glow", "warm red rim on black — subsurface glow on skin and ears", MatcapBlend::Add),
    ("Velvet", "broad soft rim on black — velvet, suede, fuzz", MatcapBlend::Add),
];

/// Render a built-in matcap (256 px, sRGB), or None for an unknown name.
pub fn builtin(name: &str) -> Option<ImageData> {
    let kind = BUILTINS.iter().position(|(n, _, _)| n.eq_ignore_ascii_case(name))?;
    let size = 256u32;
    let light = normalize([-0.45, 0.6, 0.66]);
    let mut rgba = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            // Sphere normal under this pixel; outside the disc, the rim's.
            let nx = (x as f32 + 0.5) / size as f32 * 2.0 - 1.0;
            let ny = -((y as f32 + 0.5) / size as f32 * 2.0 - 1.0);
            let r2 = nx * nx + ny * ny;
            let n = if r2 < 1.0 { [nx, ny, (1.0 - r2).sqrt()] } else { normalize([nx, ny, 0.0]) };
            let ndl = dot(n, light).max(0.0);
            // Reflection of the view ray (0, 0, -1) about the normal.
            let refl = [2.0 * n[2] * n[0], 2.0 * n[2] * n[1], 2.0 * n[2] * n[2] - 1.0];
            let c: [f32; 3] = match kind {
                0 => {
                    let v = ndl.powf(48.0) + 0.18 * ndl.powf(6.0);
                    [v, v, v]
                }
                1 => {
                    let v = (1.0 - n[2]).powf(3.0);
                    [v, v, v]
                }
                2 => {
                    let up = refl[1];
                    let base = if up >= 0.0 {
                        mix([0.78, 0.82, 0.88], [0.22, 0.32, 0.52], up.sqrt())
                    } else {
                        // A lighter ground than a photo would have, so Multiply
                        // over light skin reads as metal rather than black.
                        mix([0.46, 0.42, 0.38], [0.14, 0.13, 0.13], (-up).sqrt())
                    };
                    let horizon = (1.0 - (up.abs() / 0.03)).max(0.0);
                    let glint = dot(normalize(refl), light).max(0.0).powf(120.0);
                    base.map(|b| b + horizon * 0.6 + glint)
                }
                3 => {
                    let t = n[2] * 1.4 + nx * 0.3;
                    let tint = [0.0f32, 0.33, 0.67].map(|o| 0.86 + 0.14 * (std::f32::consts::TAU * (t + o)).cos());
                    let shade = 0.55 + 0.45 * ndl;
                    let spec = ndl.powf(60.0) * 0.6;
                    tint.map(|c| c * shade + spec)
                }
                4 => {
                    let v = 0.3 + 0.5 * ndl + 0.35 * (1.0 - n[2]).powf(2.0) + 0.25 * ndl.powf(12.0);
                    [v, v, v * 0.98]
                }
                5 => {
                    let v = 0.22 + 0.78 * ndl;
                    [0.86 * v, 0.74 * v, 0.66 * v]
                }
                6 | 7 => {
                    // Chrome's sky/ground split, tinted: gold, or copper.
                    let (sky, zenith, ground, deep) = if kind == 6 {
                        ([1.0, 0.88, 0.58], [0.78, 0.56, 0.24], [0.62, 0.46, 0.22], [0.26, 0.17, 0.07])
                    } else {
                        ([1.0, 0.74, 0.62], [0.74, 0.38, 0.26], [0.58, 0.34, 0.24], [0.22, 0.1, 0.07])
                    };
                    let up = refl[1];
                    let base = if up >= 0.0 { mix(sky, zenith, up.sqrt()) } else { mix(ground, deep, (-up).sqrt()) };
                    let horizon = (1.0 - (up.abs() / 0.04)).max(0.0) * 0.5;
                    let glint = dot(normalize(refl), light).max(0.0).powf(90.0);
                    base.map(|b| b + horizon + glint)
                }
                8 => {
                    let v = smoothstep(0.42, 0.5, ndl.powf(14.0)) * 0.92;
                    [v, v, v]
                }
                9 => {
                    let r = normalize(refl);
                    let streak_dir = normalize([0.55, 0.15, 0.82]);
                    // A tall softbox: the reflection near its column, wide in Y.
                    let d = [r[0] - streak_dir[0], (r[1] - streak_dir[1]) * 0.35];
                    let streak = (1.0 - (d[0] * d[0] + d[1] * d[1]).sqrt() / 0.16).clamp(0.0, 1.0).powf(0.8) * 0.75;
                    let v = ndl.powf(120.0) * 1.2 + streak + 0.22 * (1.0 - n[2]).powf(4.0);
                    [v, v, v]
                }
                10 => {
                    let r = normalize(refl);
                    let glints = [[-0.45, 0.6, 0.66], [0.5, 0.45, 0.74], [-0.2, -0.35, 0.92], [0.15, 0.75, 0.64]];
                    let v: f32 = glints.iter().zip([1.0, 0.7, 0.45, 0.55]).map(|(g, w)| w * smoothstep(0.9975, 0.9992, dot(r, normalize(*g)))).sum();
                    [v, v, v]
                }
                11 => {
                    let t = n[2] * 1.15 + nx * 0.35 - ny * 0.25;
                    let edge = 0.35 + 0.65 * (1.0 - n[2]).powf(0.7);
                    [0.0f32, 0.33, 0.67].map(|o| (0.5 + 0.5 * (std::f32::consts::TAU * (t + o)).cos()) * edge * 0.85)
                }
                12 => {
                    // One possible sparkle per 7 px cell, at a hashed spot.
                    let (cx, cy) = (x / 7, y / 7);
                    let h = hash(cx, cy, 1);
                    if h > 0.82 {
                        let (ox, oy) = (hash(cx, cy, 2) * 7.0, hash(cx, cy, 3) * 7.0);
                        let (dx, dy) = ((x % 7) as f32 + 0.5 - ox, (y % 7) as f32 + 0.5 - oy);
                        let core = (1.0 - (dx * dx + dy * dy).sqrt() / 2.2).max(0.0);
                        let cross = (1.0 - dx.abs().min(dy.abs()) / 0.6).max(0.0) * (1.0 - dx.abs().max(dy.abs()) / 3.5).max(0.0);
                        let v = (core + cross * 0.6) * (0.55 + 0.45 * hash(cx, cy, 4));
                        let tint = 0.85 + 0.15 * hash(cx, cy, 5);
                        [v, v * tint, v * (2.0 - tint)]
                    } else {
                        [0.0; 3]
                    }
                }
                13 => {
                    let rim = (1.0 - n[2]).powf(2.2);
                    let soft = ndl.powf(30.0) * 0.1;
                    [0.62 * rim + soft, 0.2 * rim + soft, 0.13 * rim + soft]
                }
                14 => {
                    let rim = (1.0 - n[2]).powf(1.4) * 0.5;
                    [rim, rim * 0.96, rim * 1.02]
                }
                _ => unreachable!("one arm per built-in"),
            };
            let byte = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
            rgba.extend_from_slice(&[byte(c[0]), byte(c[1]), byte(c[2]), 255]);
        }
    }
    Some(ImageData { width: size, height: size, rgba: rgba.into() })
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Stable 0..1 noise for a cell, so a built-in renders the same every time.
fn hash(x: u32, y: u32, seed: u32) -> f32 {
    let mut h = x.wrapping_mul(0x8da6_b343) ^ y.wrapping_mul(0xd816_3841) ^ seed.wrapping_mul(0xcb1a_b31f);
    h ^= h >> 13;
    h = h.wrapping_mul(0x5bd1_e995);
    h ^= h >> 15;
    h as f32 / u32::MAX as f32
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let l = dot(v, v).sqrt().max(1e-6);
    v.map(|c| c / l)
}

fn mix(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [0, 1, 2].map(|i| a[i] + (b[i] - a[i]) * t.clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `ZESTSURFACE_MATCAP_DIR=/some/dir cargo test dump_builtins -- --ignored`
    /// writes the built-ins as PNGs, to look at them.
    #[test]
    #[ignore]
    fn dump_builtins() {
        let dir = std::path::PathBuf::from(std::env::var("ZESTSURFACE_MATCAP_DIR").expect("set ZESTSURFACE_MATCAP_DIR"));
        for (name, _, _) in BUILTINS {
            let img = builtin(name).unwrap();
            image::RgbaImage::from_raw(img.width, img.height, img.rgba.to_vec()).unwrap().save(dir.join(format!("{name}.png"))).unwrap();
        }
    }

    #[test]
    fn builtins_render_and_read_as_intended() {
        for (name, _, _) in BUILTINS {
            let img = builtin(name).unwrap();
            assert_eq!((img.width, img.height, img.rgba.len()), (256, 256, 256 * 256 * 4));
        }
        let px = |img: &ImageData, x: u32, y: u32| img.rgba[((y * img.width + x) * 4) as usize];
        // Rim light: dark facing the camera, bright at the silhouette.
        let rim = builtin("rim light").unwrap();
        assert!(px(&rim, 128, 128) < 10 && px(&rim, 128, 1) > 140);
        // Shine: its highlight sits up and to the left.
        let shine = builtin("Shine").unwrap();
        assert!(px(&shine, 90, 70) > px(&shine, 170, 190));
        assert!(builtin("nope").is_none());
        assert_eq!(MatcapBlend::parse("multiply"), Some(MatcapBlend::Multiply));
    }
}
