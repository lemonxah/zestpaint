//! Procedural lace and net materials.
//!
//! No CC0 catalogue publishes lace — searching ambientCG for lace, tulle,
//! chiffon, organza, embroidery, doily or filigree returns nothing, and Poly
//! Haven has no lace either. The hard part of lace is not the colour, it is the
//! *cutout*: you need an opacity map with real holes. That is exactly what a
//! generator is good at, so build it rather than hunt for it.
//!
//! Everything here is evaluated analytically per texel and tiles seamlessly:
//! the net lines are integer-frequency, and motifs sit on a lattice that
//! divides the tile exactly.

use std::collections::HashMap;
use std::f32::consts::TAU;
use std::path::Path;

use anyhow::Result;

use crate::download::Manifest;

/// What sits on top of the net ground.
#[derive(Clone, Copy)]
enum Motif {
    None,
    /// A ring of petal outlines around a centre dot.
    Flower { petals: u32, radius: f32, thread: f32 },
    /// Concentric rings.
    Rings { count: u32, radius: f32, thread: f32 },
    /// Ring plus a solid bar, reading as a scalloped edge band.
    Scallop { radius: f32, thread: f32 },
    /// Punched holes through otherwise solid fabric.
    Eyelet { radius: f32 },
}

struct Lace {
    name: &'static str,
    description: &'static str,
    tags: &'static [&'static str],
    /// Diagonal net frequency; integer keeps the tile seamless. 0 = no net.
    net_freq: f32,
    /// Half-thickness of a net thread, in UV units.
    net_width: f32,
    /// Motif lattice cells across the tile; must divide the tile exactly.
    cells: u32,
    motif: Motif,
    /// Eyelet fabric is solid with holes punched out, the inverse of lace.
    inverted: bool,
}

fn patterns() -> Vec<Lace> {
    vec![
        Lace {
            name: "lace_floral",
            description: "Floral lace: fine diamond ground with flower motifs.",
            tags: &["lace", "fabric", "sheer", "lingerie", "floral", "procedural"],
            net_freq: 28.0,
            net_width: 0.0018,
            cells: 4,
            motif: Motif::Flower { petals: 6, radius: 0.085, thread: 0.0038 },
            inverted: false,
        },
        Lace {
            name: "lace_geometric",
            description: "Geometric lace: diamond ground with concentric rings.",
            tags: &["lace", "fabric", "sheer", "lingerie", "geometric", "procedural"],
            net_freq: 32.0,
            net_width: 0.0016,
            cells: 6,
            motif: Motif::Rings { count: 2, radius: 0.055, thread: 0.0032 },
            inverted: false,
        },
        Lace {
            name: "lace_scallop",
            description: "Scalloped lace trim: banded ground with arc motifs.",
            tags: &["lace", "fabric", "sheer", "lingerie", "trim", "scallop", "procedural"],
            net_freq: 24.0,
            net_width: 0.0020,
            cells: 5,
            motif: Motif::Scallop { radius: 0.075, thread: 0.0042 },
            inverted: false,
        },
        Lace {
            name: "lace_tulle",
            description: "Plain tulle: fine hexagonal-ish netting, no motifs.",
            tags: &["tulle", "net", "fabric", "sheer", "lingerie", "veil", "procedural"],
            net_freq: 44.0,
            net_width: 0.0012,
            cells: 1,
            motif: Motif::None,
            inverted: false,
        },
        Lace {
            name: "lace_eyelet",
            description: "Broderie anglaise: solid fabric with punched eyelet holes.",
            tags: &["eyelet", "broderie", "fabric", "lingerie", "cotton", "procedural"],
            net_freq: 0.0,
            net_width: 0.0,
            cells: 8,
            motif: Motif::Eyelet { radius: 0.030 },
            inverted: true,
        },
    ]
}

/// Signed distance to the nearest net thread. Lines run along both diagonals,
/// at integer frequency so opposite tile edges line up.
fn net_distance(u: f32, v: f32, freq: f32, width: f32) -> f32 {
    if freq <= 0.0 {
        return f32::MAX;
    }
    let diag = std::f32::consts::SQRT_2;
    let s = (u + v) * freq;
    let t = (u - v) * freq;
    let ds = (s - s.round()).abs() / (freq * diag);
    let dt = (t - t.round()).abs() / (freq * diag);
    ds.min(dt) - width
}

fn ring(px: f32, py: f32, cx: f32, cy: f32, radius: f32, thread: f32) -> f32 {
    let d = ((px - cx).powi(2) + (py - cy).powi(2)).sqrt();
    (d - radius).abs() - thread
}

fn disc(px: f32, py: f32, cx: f32, cy: f32, radius: f32) -> f32 {
    ((px - cx).powi(2) + (py - cy).powi(2)).sqrt() - radius
}

/// Signed distance to a motif centred on `(cx, cy)`.
fn motif_distance(motif: Motif, px: f32, py: f32, cx: f32, cy: f32) -> f32 {
    match motif {
        Motif::None => f32::MAX,
        Motif::Flower { petals, radius, thread } => {
            let mut d = ring(px, py, cx, cy, radius, thread);
            for k in 0..petals {
                let a = k as f32 / petals as f32 * TAU;
                let (sx, sy) = (cx + a.cos() * radius * 0.62, cy + a.sin() * radius * 0.62);
                d = d.min(ring(px, py, sx, sy, radius * 0.26, thread));
            }
            d.min(disc(px, py, cx, cy, thread * 1.8))
        }
        Motif::Rings { count, radius, thread } => {
            let mut d = f32::MAX;
            for k in 0..count.max(1) {
                let r = radius * (k + 1) as f32 / count.max(1) as f32;
                d = d.min(ring(px, py, cx, cy, r, thread));
            }
            d.min(disc(px, py, cx, cy, thread * 1.6))
        }
        Motif::Scallop { radius, thread } => {
            // Arc plus a chord, which reads as a scalloped band edge.
            let arc = ring(px, py, cx, cy, radius, thread);
            let bar = (py - cy).abs() - thread;
            arc.min(bar)
        }
        Motif::Eyelet { radius } => disc(px, py, cx, cy, radius),
    }
}

/// Distance to the nearest lace thread at a point, combining net and motifs.
/// Motifs on neighbouring cells are included so the pattern wraps.
fn lace_distance(lace: &Lace, u: f32, v: f32) -> f32 {
    let mut d = net_distance(u, v, lace.net_freq, lace.net_width);
    if !matches!(lace.motif, Motif::None) {
        let cells = lace.cells.max(1) as f32;
        let (cu, cv) = ((u * cells).floor(), (v * cells).floor());
        for dy in -1..=1 {
            for dx in -1..=1 {
                let cx = (cu + dx as f32 + 0.5) / cells;
                let cy = (cv + dy as f32 + 0.5) / cells;
                d = d.min(motif_distance(lace.motif, u, v, cx, cy));
            }
        }
    }
    d
}

/// Rasterise one pattern into its maps.
fn render(lace: &Lace, size: u32) -> LaceMaps {
    let n = (size * size) as usize;
    let mut coverage = vec![0f32; n];
    let aa = 1.0 / size as f32;

    // Rows are independent; a 2K tile with nine neighbour cells per texel is
    // enough work to be worth splitting.
    let threads = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4).min(16);
    let rows_per = size.div_ceil(threads as u32);
    std::thread::scope(|scope| {
        for (chunk_index, chunk) in coverage.chunks_mut((rows_per * size) as usize).enumerate() {
            let y0 = chunk_index as u32 * rows_per;
            scope.spawn(move || {
                for (i, out) in chunk.iter_mut().enumerate() {
                    let y = y0 + i as u32 / size;
                    let x = i as u32 % size;
                    let u = (x as f32 + 0.5) / size as f32;
                    let v = (y as f32 + 0.5) / size as f32;
                    let d = lace_distance(lace, u, v);
                    // Inside the thread -> 1, feathering out over about a texel.
                    let mut c = ((-d) / aa).clamp(0.0, 1.0);
                    if lace.inverted {
                        c = 1.0 - c;
                    }
                    *out = c;
                }
            });
        }
    });

    // Thread relief: a dome across each strand, so the weave catches light.
    let mut height = vec![0f32; n];
    for (i, &c) in coverage.iter().enumerate() {
        height[i] = c * c * (3.0 - 2.0 * c) * 0.5 + c * 0.5;
    }
    LaceMaps { size, coverage, height }
}

struct LaceMaps {
    size: u32,
    coverage: Vec<f32>,
    height: Vec<f32>,
}

/// Sobel the height field into a tangent-space normal map (OpenGL convention).
fn normal_from_height(maps: &LaceMaps, strength: f32) -> image::RgbImage {
    let size = maps.size;
    let at = |x: i32, y: i32| -> f32 {
        let xi = x.rem_euclid(size as i32) as u32;
        let yi = y.rem_euclid(size as i32) as u32;
        maps.height[(yi * size + xi) as usize]
    };
    let mut out = image::RgbImage::new(size, size);
    for y in 0..size as i32 {
        for x in 0..size as i32 {
            let dx = (at(x + 1, y - 1) + 2.0 * at(x + 1, y) + at(x + 1, y + 1))
                - (at(x - 1, y - 1) + 2.0 * at(x - 1, y) + at(x - 1, y + 1));
            let dy = (at(x - 1, y + 1) + 2.0 * at(x, y + 1) + at(x + 1, y + 1))
                - (at(x - 1, y - 1) + 2.0 * at(x, y - 1) + at(x + 1, y - 1));
            let (nx, ny, nz) = (-dx * strength, -dy * strength, 1.0);
            let len = (nx * nx + ny * ny + nz * nz).sqrt();
            let enc = |v: f32| (((v / len) * 0.5 + 0.5) * 255.0).clamp(0.0, 255.0) as u8;
            out.put_pixel(x as u32, y as u32, image::Rgb([enc(nx), enc(ny), enc(nz)]));
        }
    }
    out
}

/// Generate every lace pattern into `<root>/materials/<name>/`.
pub fn generate(root: &Path, size: u32) -> Result<usize> {
    let all = patterns();
    for lace in &all {
        let dir = root.join("materials").join(lace.name);
        std::fs::create_dir_all(&dir)?;
        let maps = render(lace, size);

        // Opacity carries the cutout: this is what makes it lace and not a
        // printed pattern.
        let mut opacity = image::GrayImage::new(size, size);
        let mut height = image::GrayImage::new(size, size);
        let mut basecolor = image::RgbImage::new(size, size);
        let mut roughness = image::GrayImage::new(size, size);
        for y in 0..size {
            for x in 0..size {
                let i = (y * size + x) as usize;
                let c = maps.coverage[i];
                opacity.put_pixel(x, y, image::Luma([(c * 255.0) as u8]));
                height.put_pixel(x, y, image::Luma([(maps.height[i] * 255.0) as u8]));
                // Slightly off-white thread; the cutout comes from opacity.
                basecolor.put_pixel(x, y, image::Rgb([242, 238, 232]));
                // Thread is matte, gaps do not matter once clipped.
                roughness.put_pixel(x, y, image::Luma([196]));
            }
        }
        opacity.save(dir.join("opacity.png"))?;
        height.save(dir.join("height.png"))?;
        basecolor.save(dir.join("basecolor.png"))?;
        roughness.save(dir.join("roughness.png"))?;
        normal_from_height(&maps, 6.0).save(dir.join("normal.png"))?;

        let manifest = Manifest {
            id: format!("proc:{}", lace.name),
            name: lace.name.replace('_', " "),
            source: "procedural".into(),
            kind: "material".into(),
            license: "CC0".into(),
            authors: vec!["zestpaint matfetch".into()],
            page: String::new(),
            variant: format!("{size}px"),
            tags: lace.tags.iter().map(|s| s.to_string()).collect(),
            maps: HashMap::from([
                ("basecolor".to_string(), "basecolor.png".to_string()),
                ("opacity".to_string(), "opacity.png".to_string()),
                ("height".to_string(), "height.png".to_string()),
                ("normal".to_string(), "normal.png".to_string()),
                ("roughness".to_string(), "roughness.png".to_string()),
            ]),
        };
        let mut value = serde_json::to_value(&manifest)?;
        value["description"] = serde_json::Value::String(lace.description.into());
        std::fs::write(dir.join("material.json"), serde_json::to_vec_pretty(&value)?)?;
        eprintln!("[lace] {} ({}px)", lace.name, size);
    }
    Ok(all.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Lace is only useful if it actually has holes: a pattern that came out
    /// fully opaque or fully empty would be silently useless. Rendered at a
    /// realistic size, since the threads are deliberately about one texel wide
    /// at 1K and vanish below that.
    #[test]
    fn every_pattern_has_both_thread_and_holes() {
        for lace in patterns() {
            let maps = render(&lace, 512);
            let solid = maps.coverage.iter().filter(|c| **c > 0.5).count();
            let holes = maps.coverage.iter().filter(|c| **c < 0.1).count();
            let total = maps.coverage.len();
            assert!(
                solid * 100 / total >= 2,
                "{}: only {}% thread",
                lace.name,
                solid * 100 / total
            );
            assert!(
                holes * 100 / total >= 10,
                "{}: only {}% holes — not a cutout",
                lace.name,
                holes * 100 / total
            );
        }
    }

    /// Seams show up instantly on a garment, so the tile must wrap.
    #[test]
    fn patterns_tile_seamlessly() {
        for lace in patterns() {
            for step in 0..64 {
                let t = step as f32 / 64.0 + 0.003;
                // Opposite edges must agree, both horizontally and vertically.
                let left = lace_distance(&lace, 0.0005, t);
                let right = lace_distance(&lace, 1.0005, t);
                assert!(
                    (left - right).abs() < 1e-4,
                    "{}: horizontal seam at v={t}: {left} vs {right}",
                    lace.name
                );
                let top = lace_distance(&lace, t, 0.0005);
                let bottom = lace_distance(&lace, t, 1.0005);
                assert!(
                    (top - bottom).abs() < 1e-4,
                    "{}: vertical seam at u={t}: {top} vs {bottom}",
                    lace.name
                );
            }
        }
    }
}
