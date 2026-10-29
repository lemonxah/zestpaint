//! Procedural splatter stamps and flat parametric materials.
//!
//! Neither CC0 catalogue has blood, paint splats or drips: searching the index
//! for blood/splat/spatter/drip finds only "Leaking001" (a moisture streak
//! decal). Splatter is a few blobs, rays and droplets combined with a smooth
//! union, so generate it — any seed gives a new, clean, resolution-independent
//! stamp. Used by `matfetch splatter` and by the zestpaint assistant.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Result, bail};

use crate::download::Manifest;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Style {
    /// An impact splat: lumpy central pool, radiating spikes, satellite drops.
    Splat,
    /// A pool near the top with runs dripping downward (stamp "down" is +v).
    Drip,
    /// A mist of fine droplets with a few larger ones.
    Spray,
    /// A handful of separate round drops.
    Droplets,
    /// A dragged streak, like a smeared handprint without the hand.
    Smear,
}

impl Style {
    pub const ALL: [Style; 5] = [Style::Splat, Style::Drip, Style::Spray, Style::Droplets, Style::Smear];

    pub fn name(self) -> &'static str {
        match self {
            Style::Splat => "splat",
            Style::Drip => "drip",
            Style::Spray => "spray",
            Style::Droplets => "droplets",
            Style::Smear => "smear",
        }
    }

    pub fn parse(s: &str) -> Result<Style> {
        Style::ALL
            .into_iter()
            .find(|st| st.name().eq_ignore_ascii_case(s))
            .ok_or_else(|| anyhow::anyhow!("unknown splatter style {s:?} (expected splat|drip|spray|droplets|smear)"))
    }
}

/// splitmix64: tiny, seedable, and good enough to scatter droplets.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn f(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.f()
    }
}

#[derive(Clone, Copy)]
enum Prim {
    Circle { cx: f32, cy: f32, r: f32 },
    /// A capsule whose radius tapers linearly from ra (at a) to rb (at b).
    Taper { ax: f32, ay: f32, bx: f32, by: f32, ra: f32, rb: f32 },
    /// A circle whose edge is perturbed by a sum of angular harmonics.
    Lumpy { cx: f32, cy: f32, r: f32, harmonics: [(f32, f32, f32); 6] },
}

impl Prim {
    fn bounds(&self) -> (f32, f32, f32, f32) {
        match *self {
            Prim::Circle { cx, cy, r } => (cx - r, cy - r, cx + r, cy + r),
            Prim::Taper { ax, ay, bx, by, ra, rb } => {
                let r = ra.max(rb);
                (ax.min(bx) - r, ay.min(by) - r, ax.max(bx) + r, ay.max(by) + r)
            }
            Prim::Lumpy { cx, cy, r, harmonics } => {
                let amp: f32 = harmonics.iter().map(|h| h.1.abs()).sum();
                let r = r * (1.0 + amp);
                (cx - r, cy - r, cx + r, cy + r)
            }
        }
    }

    fn sdf(&self, px: f32, py: f32) -> f32 {
        match *self {
            Prim::Circle { cx, cy, r } => ((px - cx).powi(2) + (py - cy).powi(2)).sqrt() - r,
            Prim::Taper { ax, ay, bx, by, ra, rb } => {
                let (pax, pay) = (px - ax, py - ay);
                let (bax, bay) = (bx - ax, by - ay);
                let denom = bax * bax + bay * bay;
                let h = if denom > 0.0 { ((pax * bax + pay * bay) / denom).clamp(0.0, 1.0) } else { 0.0 };
                let (dx, dy) = (pax - bax * h, pay - bay * h);
                (dx * dx + dy * dy).sqrt() - (ra + (rb - ra) * h)
            }
            Prim::Lumpy { cx, cy, r, harmonics } => {
                let (dx, dy) = (px - cx, py - cy);
                let theta = dy.atan2(dx);
                let wobble: f32 = harmonics.iter().map(|&(k, a, phase)| a * (k * theta + phase).sin()).sum();
                (dx * dx + dy * dy).sqrt() - r * (1.0 + wobble)
            }
        }
    }
}

fn lumpy(rng: &mut Rng, cx: f32, cy: f32, r: f32, roughness: f32) -> Prim {
    let mut harmonics = [(0.0, 0.0, 0.0); 6];
    for (i, h) in harmonics.iter_mut().enumerate() {
        let k = (2 + i * 2) as f32 + rng.range(0.0, 1.0).floor();
        *h = (k, rng.range(-1.0, 1.0) * roughness / (1.0 + i as f32 * 0.6), rng.range(0.0, std::f32::consts::TAU));
    }
    Prim::Lumpy { cx, cy, r, harmonics }
}

fn layout(style: Style, seed: u64, density: f32) -> Vec<Prim> {
    let mut rng = Rng(seed ^ 0x5EED_0F_B100D);
    let d = density.clamp(0.1, 3.0);
    let count = |n: f32| ((n * d).round() as usize).max(1);
    let tau = std::f32::consts::TAU;
    let mut out = Vec::new();

    match style {
        Style::Splat => {
            let (cx, cy) = (0.5 + rng.range(-0.03, 0.03), 0.5 + rng.range(-0.03, 0.03));
            let r0 = rng.range(0.11, 0.15);
            out.push(lumpy(&mut rng, cx, cy, r0, 0.18));
            for _ in 0..count(14.0) {
                let a = rng.range(0.0, tau);
                let len = r0 * rng.range(1.2, 2.6);
                let (ex, ey) = (cx + a.cos() * len, cy + a.sin() * len);
                let tip = r0 * rng.range(0.04, 0.12);
                out.push(Prim::Taper { ax: cx, ay: cy, bx: ex, by: ey, ra: r0 * rng.range(0.25, 0.45), rb: tip * 0.5 });
                if rng.f() < 0.7 {
                    let gap = len + tip * rng.range(1.5, 4.0);
                    out.push(Prim::Circle { cx: cx + a.cos() * gap, cy: cy + a.sin() * gap, r: tip });
                }
            }
            for _ in 0..count(40.0) {
                let a = rng.range(0.0, tau);
                let dist = r0 * rng.range(1.3, 3.2);
                let r = r0 * 0.09 * (1.0 - (dist / (r0 * 3.4))).max(0.15) * rng.range(0.4, 1.2);
                out.push(Prim::Circle { cx: cx + a.cos() * dist, cy: cy + a.sin() * dist, r });
            }
        }
        Style::Drip => {
            let (cx, cy) = (0.5, rng.range(0.18, 0.24));
            let r0 = rng.range(0.1, 0.13);
            out.push(lumpy(&mut rng, cx, cy, r0, 0.12));
            for _ in 0..count(7.0) {
                let x = cx + rng.range(-0.9, 0.9) * r0;
                let w = r0 * rng.range(0.08, 0.2);
                let len = rng.range(0.15, 0.72 - cy);
                let end = (cy + len).min(0.9);
                out.push(Prim::Taper { ax: x, ay: cy, bx: x + rng.range(-0.01, 0.01), by: end, ra: w, rb: w * 0.7 });
                // Gravity collects a bead at the bottom of each run.
                out.push(Prim::Circle { cx: x, cy: end, r: w * rng.range(1.2, 1.6) });
            }
            for _ in 0..count(10.0) {
                let a = rng.range(0.0, tau);
                let dist = r0 * rng.range(1.2, 2.0);
                out.push(Prim::Circle { cx: cx + a.cos() * dist, cy: (cy + a.sin() * dist * 0.6).max(0.03), r: r0 * rng.range(0.03, 0.08) });
            }
        }
        Style::Spray => {
            for i in 0..count(260.0) {
                // Box-Muller: a radial gaussian concentrates the mist at the centre.
                let (u1, u2) = (rng.f().max(1e-6), rng.f());
                let rad = (-2.0 * u1.ln()).sqrt() * 0.13;
                let a = u2 * tau;
                let big = i % 23 == 0;
                let r = if big { rng.range(0.012, 0.025) } else { rng.range(0.002, 0.008) };
                out.push(Prim::Circle { cx: 0.5 + a.cos() * rad, cy: 0.5 + a.sin() * rad, r });
            }
        }
        Style::Droplets => {
            for _ in 0..count(9.0) {
                let r = rng.range(0.03, 0.08);
                let (x, y) = (rng.range(0.15, 0.85), rng.range(0.15, 0.85));
                out.push(lumpy(&mut rng, x, y, r, 0.06));
            }
        }
        Style::Smear => {
            let a = rng.range(-0.5, 0.5);
            let (dx, dy) = (a.cos(), a.sin());
            let steps = 18;
            for i in 0..steps {
                let t = i as f32 / (steps - 1) as f32;
                let along = -0.33 + 0.66 * t;
                let wob = (t * 9.0 + rng.range(0.0, 0.5)).sin() * 0.015;
                let (x, y) = (0.5 + dx * along - dy * wob, 0.5 + dy * along + dx * wob);
                // Thick where the hand landed, thinning out as it drags.
                let r = 0.09 * (1.0 - t * 0.65) * rng.range(0.85, 1.1);
                out.push(lumpy(&mut rng, x, y, r, 0.1));
            }
            for _ in 0..count(6.0) {
                let off = rng.range(-0.06, 0.06);
                let start = rng.range(-0.3, -0.1);
                let end = start + rng.range(0.3, 0.6);
                out.push(Prim::Taper {
                    ax: 0.5 + dx * start - dy * off,
                    ay: 0.5 + dy * start + dx * off,
                    bx: 0.5 + dx * end - dy * off,
                    by: 0.5 + dy * end + dx * off,
                    ra: 0.012,
                    rb: 0.002,
                });
            }
        }
    }
    out
}

/// Polynomial smooth minimum: merges blobs with a fillet instead of a crease,
/// which is what makes a cluster of circles read as one liquid body.
fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = (0.5 + 0.5 * (b - a) / k).clamp(0.0, 1.0);
    b + (a - b) * h - k * h * (1.0 - h)
}

/// Rasterise a splatter to (opacity, height). Opacity is 255 = paint, matching
/// the brush tip sampler; height is a dome that is thickest in the pool.
pub fn render(style: Style, seed: u64, density: f32, size: u32) -> (image::GrayImage, image::GrayImage) {
    let prims = layout(style, seed, density);
    let n = size as usize;
    let k = 0.012f32;
    let mut field = vec![1.0f32; n * n];
    let px = 1.0 / size as f32;
    for p in &prims {
        let (x0, y0, x1, y1) = p.bounds();
        let m = k * 2.0 + 2.0 * px;
        let ix0 = (((x0 - m) * size as f32).floor().max(0.0)) as usize;
        let iy0 = (((y0 - m) * size as f32).floor().max(0.0)) as usize;
        let ix1 = (((x1 + m) * size as f32).ceil() as usize).min(n);
        let iy1 = (((y1 + m) * size as f32).ceil() as usize).min(n);
        for y in iy0..iy1 {
            let v = (y as f32 + 0.5) * px;
            for x in ix0..ix1 {
                let u = (x as f32 + 0.5) * px;
                let i = y * n + x;
                field[i] = smin(field[i], p.sdf(u, v), k);
            }
        }
    }

    // Keep the stamp off its border: a splat that touches the edge of the tip
    // shows a hard square cut when stamped.
    let edge_fade = |u: f32| ((u.min(1.0 - u)) / 0.02).clamp(0.0, 1.0);
    let mut opacity = image::GrayImage::new(size, size);
    let mut height = image::GrayImage::new(size, size);
    for y in 0..n {
        let v = (y as f32 + 0.5) * px;
        for x in 0..n {
            let u = (x as f32 + 0.5) * px;
            let d = field[y * n + x];
            let cov = ((-d) / px).clamp(0.0, 1.0) * edge_fade(u) * edge_fade(v);
            opacity.put_pixel(x as u32, y as u32, image::Luma([(cov * 255.0) as u8]));
            let t = ((-d) / 0.05).clamp(0.0, 1.0);
            let dome = (1.0 - (1.0 - t) * (1.0 - t)).sqrt();
            height.put_pixel(x as u32, y as u32, image::Luma([(dome * cov * 255.0) as u8]));
        }
    }
    (opacity, height)
}

/// Generate one splatter stamp into `<root>/brushes/<name>/`.
pub fn generate_one(root: &Path, name: &str, style: Style, seed: u64, density: f32, size: u32) -> Result<PathBuf> {
    let folder = crate::catalog::sanitize(name);
    if folder.is_empty() {
        bail!("brush name {name:?} has no usable characters");
    }
    let dir = root.join("brushes").join(&folder);
    std::fs::create_dir_all(&dir)?;
    let (opacity, height) = render(style, seed, density, size);
    opacity.save(dir.join("opacity.png"))?;
    height.save(dir.join("height.png"))?;

    let manifest = Manifest {
        id: format!("proc:{folder}"),
        name: name.to_string(),
        source: "procedural".into(),
        kind: "brush".into(),
        license: "CC0".into(),
        authors: vec!["zestpaint matfetch".into()],
        page: String::new(),
        variant: format!("{size}px"),
        tags: vec!["splatter".into(), style.name().into(), "procedural".into()],
        maps: HashMap::from([
            ("opacity".to_string(), "opacity.png".to_string()),
            ("height".to_string(), "height.png".to_string()),
        ]),
    };
    let mut value = serde_json::to_value(&manifest)?;
    value["description"] =
        serde_json::Value::String(format!("Procedural {} splatter (seed {seed}, density {density:.2}).", style.name()));
    std::fs::write(dir.join("material.json"), serde_json::to_vec_pretty(&value)?)?;
    Ok(dir)
}

/// The CLI's batch: three seeds of every style.
pub fn generate(root: &Path, size: u32) -> Result<usize> {
    let mut n = 0;
    for style in Style::ALL {
        for seed in 1..=3u64 {
            let name = format!("splatter_{}_{seed}", style.name());
            generate_one(root, &name, style, seed, 1.0, size)?;
            eprintln!("[splatter] {name} ({size}px)");
            n += 1;
        }
    }
    Ok(n)
}

/// Parameters for a flat procedural material: uniform colour/roughness/metal
/// with optional low-amplitude value noise in colour and height so it does not
/// look like a plastic sticker.
pub struct FlatMaterial<'a> {
    pub name: &'a str,
    /// sRGB 0..255.
    pub basecolor: [u8; 3],
    pub roughness: f32,
    pub metallic: f32,
    /// 0 = perfectly flat, 1 = strong mottling.
    pub variation: f32,
    pub tags: Vec<String>,
    pub description: String,
}

fn value_noise(seed: u64, u: f32, v: f32, freq: u32) -> f32 {
    // Integer-frequency lattice with wrap-around, so the result tiles.
    let hash = |x: u32, y: u32| {
        let mut r = Rng(seed ^ ((x as u64) << 32 | y as u64).wrapping_mul(0x2545_F491_4F6C_DD1D));
        r.f()
    };
    let (fu, fv) = (u * freq as f32, v * freq as f32);
    let (x0, y0) = (fu.floor() as u32 % freq, fv.floor() as u32 % freq);
    let (x1, y1) = ((x0 + 1) % freq, (y0 + 1) % freq);
    let (tx, ty) = (fu.fract(), fv.fract());
    let s = |t: f32| t * t * (3.0 - 2.0 * t);
    let (sx, sy) = (s(tx), s(ty));
    let a = hash(x0, y0) + (hash(x1, y0) - hash(x0, y0)) * sx;
    let b = hash(x0, y1) + (hash(x1, y1) - hash(x0, y1)) * sx;
    a + (b - a) * sy
}

/// Write a flat material into `<root>/materials/<name>/`.
pub fn generate_flat_material(root: &Path, m: &FlatMaterial, size: u32) -> Result<PathBuf> {
    let folder = crate::catalog::sanitize(m.name);
    if folder.is_empty() {
        bail!("material name {:?} has no usable characters", m.name);
    }
    let dir = root.join("materials").join(&folder);
    std::fs::create_dir_all(&dir)?;
    let var = m.variation.clamp(0.0, 1.0);
    let seed = folder.bytes().fold(0u64, |h, b| h.wrapping_mul(31).wrapping_add(b as u64));

    let mut basecolor = image::RgbImage::new(size, size);
    let mut roughness = image::GrayImage::new(size, size);
    let mut height = image::GrayImage::new(size, size);
    for y in 0..size {
        let v = (y as f32 + 0.5) / size as f32;
        for x in 0..size {
            let u = (x as f32 + 0.5) / size as f32;
            let n = 0.6 * value_noise(seed, u, v, 8) + 0.4 * value_noise(seed ^ 1, u, v, 32);
            let shade = 1.0 + (n - 0.5) * 0.35 * var;
            let c = m.basecolor.map(|ch| (ch as f32 * shade).clamp(0.0, 255.0) as u8);
            basecolor.put_pixel(x, y, image::Rgb(c));
            let r = (m.roughness + (n - 0.5) * 0.2 * var).clamp(0.0, 1.0);
            roughness.put_pixel(x, y, image::Luma([(r * 255.0) as u8]));
            height.put_pixel(x, y, image::Luma([((0.5 + (n - 0.5) * var) * 255.0) as u8]));
        }
    }
    basecolor.save(dir.join("basecolor.png"))?;
    roughness.save(dir.join("roughness.png"))?;
    let mut maps = HashMap::from([
        ("basecolor".to_string(), "basecolor.png".to_string()),
        ("roughness".to_string(), "roughness.png".to_string()),
    ]);
    if var > 0.0 {
        height.save(dir.join("height.png"))?;
        maps.insert("height".into(), "height.png".into());
    }
    if m.metallic > 0.0 {
        let metal = image::GrayImage::from_pixel(size, size, image::Luma([(m.metallic.clamp(0.0, 1.0) * 255.0) as u8]));
        metal.save(dir.join("metallic.png"))?;
        maps.insert("metallic".into(), "metallic.png".into());
    }

    let manifest = Manifest {
        id: format!("proc:{folder}"),
        name: m.name.to_string(),
        source: "procedural".into(),
        kind: "material".into(),
        license: "CC0".into(),
        authors: vec!["zestpaint matfetch".into()],
        page: String::new(),
        variant: format!("{size}px"),
        tags: m.tags.clone(),
        maps,
    };
    let mut value = serde_json::to_value(&manifest)?;
    value["description"] = serde_json::Value::String(m.description.clone());
    std::fs::write(dir.join("material.json"), serde_json::to_vec_pretty(&value)?)?;
    Ok(dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_style_covers_something_and_stays_inside() {
        for style in Style::ALL {
            let (op, _) = render(style, 7, 1.0, 128);
            let covered = op.pixels().filter(|p| p.0[0] > 128).count();
            assert!(covered > 40, "{style:?} covers only {covered} px");
            let border = (0..128).any(|i| op.get_pixel(i, 0).0[0] > 0 || op.get_pixel(0, i).0[0] > 0);
            assert!(!border, "{style:?} touches the stamp border");
        }
    }

    #[test]
    fn seeds_differ() {
        let (a, _) = render(Style::Splat, 1, 1.0, 64);
        let (b, _) = render(Style::Splat, 2, 1.0, 64);
        assert_ne!(a.as_raw(), b.as_raw());
    }
}
