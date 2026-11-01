//! Procedural stamp brushes.
//!
//! Neither Poly Haven nor ambientCG publishes stitch or seam alphas — a search
//! of ambientCG for stitch/seam/thread/zipper/rivet returns nothing, and Poly
//! Haven has no brush category at all. Since those are exactly the stamps you
//! want for painting seams onto a garment, generate them instead: the shapes
//! are simple capsule/arc arrangements, and generating them means they come out
//! clean, tileable and at any resolution.

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;

use crate::download::Manifest;

/// A stamp is a set of rounded line segments in UV space, rasterised to a
/// coverage mask (`opacity`) and a domed profile (`height`).
struct Stamp {
    name: &'static str,
    description: &'static str,
    tags: &'static [&'static str],
    /// (ax, ay, bx, by, radius) in 0..1 UV. A segment with a == b is a dot.
    segments: Vec<(f32, f32, f32, f32, f32)>,
}


/// Deterministic pseudo-random in 0..1, so patterns like velcro look scattered
/// without needing a seeded RNG (which would break reproducible output).
fn hash01(i: u32) -> f32 {
    let x = i.wrapping_mul(2_654_435_761).rotate_left(13).wrapping_mul(1_597_334_677);
    ((x >> 8) & 0xFFFF) as f32 / 65535.0
}

/// Trace an ellipse as a closed run of short segments.
fn ellipse(
    cx: f32,
    cy: f32,
    rx: f32,
    ry: f32,
    radius: f32,
    steps: usize,
) -> Vec<(f32, f32, f32, f32, f32)> {
    (0..steps)
        .map(|i| {
            let a0 = i as f32 / steps as f32 * std::f32::consts::TAU;
            let a1 = (i + 1) as f32 / steps as f32 * std::f32::consts::TAU;
            (cx + rx * a0.cos(), cy + ry * a0.sin(), cx + rx * a1.cos(), cy + ry * a1.sin(), radius)
        })
        .collect()
}

fn dash_row(y: f32, count: usize, len: f32, slant: f32, radius: f32) -> Vec<(f32, f32, f32, f32, f32)> {
    (0..count)
        .map(|i| {
            let cx = (i as f32 + 0.5) / count as f32;
            (cx - len / 2.0, y - slant / 2.0, cx + len / 2.0, y + slant / 2.0, radius)
        })
        .collect()
}

fn stamps() -> Vec<Stamp> {
    let mut out = Vec::new();

    out.push(Stamp {
        name: "stitch_straight",
        description: "Classic saddle stitch: a single row of slanted dashes.",
        tags: &["stitch", "seam", "sewing", "fabric", "procedural"],
        segments: dash_row(0.5, 9, 0.062, 0.05, 0.018),
    });

    out.push(Stamp {
        name: "stitch_double",
        description: "Twin-needle topstitch: two parallel rows of dashes.",
        tags: &["stitch", "seam", "topstitch", "sewing", "procedural"],
        segments: {
            let mut s = dash_row(0.37, 9, 0.062, 0.04, 0.016);
            s.extend(dash_row(0.63, 9, 0.062, 0.04, 0.016));
            s
        },
    });

    out.push(Stamp {
        name: "stitch_zigzag",
        description: "Zigzag / overlock stitch running along the seam.",
        tags: &["stitch", "zigzag", "overlock", "sewing", "procedural"],
        segments: {
            let n = 12;
            (0..n)
                .map(|i| {
                    let x0 = i as f32 / n as f32;
                    let x1 = (i + 1) as f32 / n as f32;
                    let (y0, y1) = if i % 2 == 0 { (0.34, 0.66) } else { (0.66, 0.34) };
                    (x0, y0, x1, y1, 0.017)
                })
                .collect()
        },
    });

    out.push(Stamp {
        name: "stitch_cross",
        description: "Cross-stitch: a row of X shapes.",
        tags: &["stitch", "cross", "embroidery", "sewing", "procedural"],
        segments: {
            let n = 8;
            let mut s = Vec::new();
            for i in 0..n {
                let cx = (i as f32 + 0.5) / n as f32;
                let h = 0.042;
                s.push((cx - h, 0.5 - h, cx + h, 0.5 + h, 0.015));
                s.push((cx - h, 0.5 + h, cx + h, 0.5 - h, 0.015));
            }
            s
        },
    });

    out.push(Stamp {
        name: "stitch_chain",
        description: "Chain stitch: interlocking rounded links.",
        tags: &["stitch", "chain", "embroidery", "sewing", "procedural"],
        segments: {
            // Loops are wider than their spacing so consecutive links overlap
            // and read as a chain rather than a ladder.
            let n = 6;
            let mut s = Vec::new();
            for i in 0..n {
                let cx = (i as f32 + 0.5) / n as f32;
                s.extend(ellipse(cx, 0.5, 0.115, 0.085, 0.013, 24));
            }
            s
        },
    });

    out.push(Stamp {
        name: "stitch_blanket",
        description: "Blanket / buttonhole stitch: a spine with perpendicular legs.",
        tags: &["stitch", "blanket", "edge", "sewing", "procedural"],
        segments: {
            let n = 9;
            let mut s = vec![(0.0, 0.34, 1.0, 0.34, 0.015)];
            for i in 0..n {
                let cx = (i as f32 + 0.5) / n as f32;
                s.push((cx, 0.34, cx, 0.68, 0.015));
            }
            s
        },
    });

    out.push(Stamp {
        name: "seam_rivets",
        description: "Row of round rivets or studs, for jeans and leatherwork.",
        tags: &["rivet", "stud", "seam", "hardware", "procedural"],
        segments: (0..6)
            .map(|i| {
                let cx = (i as f32 + 0.5) / 6.0;
                (cx, 0.5, cx, 0.5, 0.055)
            })
            .collect(),
    });

    out.push(Stamp {
        name: "seam_zipper",
        description: "Zipper teeth: interlocking bars either side of a centre line.",
        tags: &["zipper", "seam", "hardware", "clothing", "procedural"],
        segments: {
            let n = 14;
            let mut s = Vec::new();
            for i in 0..n {
                let cx = (i as f32 + 0.5) / n as f32;
                let (y0, y1) = if i % 2 == 0 { (0.5, 0.30) } else { (0.5, 0.70) };
                s.push((cx, y0, cx, y1, 0.020));
            }
            s
        },
    });

    out.push(Stamp {
        name: "stitch_running",
        description: "Fine running stitch: small evenly spaced dashes.",
        tags: &["stitch", "running", "hand", "sewing", "procedural"],
        segments: dash_row(0.5, 16, 0.030, 0.0, 0.011),
    });

    out.push(Stamp {
        name: "stitch_serge",
        description: "Serged / overlock edge: zigzag wrapped over a rail.",
        tags: &["stitch", "serge", "overlock", "edge", "sewing", "procedural"],
        segments: {
            let n = 16;
            let mut s = vec![(0.0, 0.62, 1.0, 0.62, 0.013)];
            for i in 0..n {
                let x0 = i as f32 / n as f32;
                let x1 = (i + 1) as f32 / n as f32;
                let (y0, y1) = if i % 2 == 0 { (0.62, 0.36) } else { (0.36, 0.62) };
                s.push((x0, y0, x1, y1, 0.012));
            }
            s
        },
    });

    out.push(Stamp {
        name: "stitch_ladder",
        description: "Ladder stitch: two rails joined by rungs.",
        tags: &["stitch", "ladder", "rung", "sewing", "procedural"],
        segments: {
            let n = 12;
            let mut s = vec![
                (0.0, 0.38, 1.0, 0.38, 0.012),
                (0.0, 0.62, 1.0, 0.62, 0.012),
            ];
            for i in 0..n {
                let cx = (i as f32 + 0.5) / n as f32;
                s.push((cx, 0.38, cx, 0.62, 0.011));
            }
            s
        },
    });

    out.push(Stamp {
        name: "seam_hem",
        description: "Folded hem: a fold line with twin topstitching.",
        tags: &["hem", "seam", "topstitch", "clothing", "procedural"],
        segments: {
            let mut s = vec![(0.0, 0.5, 1.0, 0.5, 0.020)];
            s.extend(dash_row(0.36, 14, 0.034, 0.0, 0.010));
            s.extend(dash_row(0.64, 14, 0.034, 0.0, 0.010));
            s
        },
    });

    out.push(Stamp {
        name: "seam_piping",
        description: "Corded piping trim: a round cord between two stitch lines.",
        tags: &["piping", "cord", "trim", "seam", "clothing", "procedural"],
        segments: {
            let mut s = vec![(0.0, 0.5, 1.0, 0.5, 0.055)];
            s.extend(dash_row(0.30, 16, 0.028, 0.0, 0.009));
            s.extend(dash_row(0.70, 16, 0.028, 0.0, 0.009));
            s
        },
    });

    out.push(Stamp {
        name: "seam_grommet",
        description: "Lacing grommets: metal eyelet rings along an edge.",
        tags: &["grommet", "eyelet", "lacing", "corset", "hardware", "procedural"],
        segments: {
            let n = 5;
            let mut s = Vec::new();
            for i in 0..n {
                let cx = (i as f32 + 0.5) / n as f32;
                s.extend(ellipse(cx, 0.5, 0.062, 0.062, 0.022, 28));
            }
            s
        },
    });

    out.push(Stamp {
        name: "seam_button",
        description: "Buttons: rimmed disc with four thread holes.",
        tags: &["button", "fastener", "clothing", "hardware", "procedural"],
        segments: {
            let n = 4;
            let mut s = Vec::new();
            for i in 0..n {
                let cx = (i as f32 + 0.5) / n as f32;
                s.extend(ellipse(cx, 0.5, 0.085, 0.085, 0.016, 32));
                for (dx, dy) in [(-0.03, -0.03), (0.03, -0.03), (-0.03, 0.03), (0.03, 0.03)] {
                    s.push((cx + dx, 0.5 + dy, cx + dx, 0.5 + dy, 0.014));
                }
            }
            s
        },
    });

    out.push(Stamp {
        name: "seam_snap",
        description: "Press studs: concentric metal rings.",
        tags: &["snap", "press stud", "fastener", "hardware", "procedural"],
        segments: {
            let n = 4;
            let mut s = Vec::new();
            for i in 0..n {
                let cx = (i as f32 + 0.5) / n as f32;
                s.extend(ellipse(cx, 0.5, 0.090, 0.090, 0.014, 32));
                s.extend(ellipse(cx, 0.5, 0.042, 0.042, 0.012, 24));
            }
            s
        },
    });

    out.push(Stamp {
        name: "seam_zipper_coil",
        description: "Coil zipper: continuous spiral teeth between two tapes.",
        tags: &["zipper", "coil", "seam", "hardware", "clothing", "procedural"],
        segments: {
            let n = 26;
            let mut s = vec![
                (0.0, 0.24, 1.0, 0.24, 0.020),
                (0.0, 0.76, 1.0, 0.76, 0.020),
            ];
            for i in 0..n {
                let x0 = i as f32 / n as f32;
                let x1 = (i + 1) as f32 / n as f32;
                // Alternating short diagonals read as a coil rather than teeth.
                let (y0, y1) = if i % 2 == 0 { (0.40, 0.60) } else { (0.60, 0.40) };
                s.push((x0, y0, x1, y1, 0.016));
            }
            s
        },
    });

    out.push(Stamp {
        name: "seam_velcro",
        description: "Hook-and-loop tape: dense scattered hooks.",
        tags: &["velcro", "hook", "loop", "fastener", "clothing", "procedural"],
        segments: {
            let mut s = Vec::new();
            for i in 0..160u32 {
                // Decorrelated streams: consecutive indices through one hash
                // line up into diagonal bands.
                let x = hash01(i.wrapping_mul(7_919).wrapping_add(13));
                let y = 0.18 + hash01(i.wrapping_mul(104_729).wrapping_add(7)) * 0.64;
                let a = hash01(i.wrapping_mul(1_299_721).wrapping_add(3))
                    * std::f32::consts::TAU;
                let len = 0.020;
                s.push((x, y, x + a.cos() * len, y + a.sin() * len, 0.007));
            }
            s
        },
    });

    out
}

/// Distance from p to the segment a-b, minus the radius.
fn sd_capsule(px: f32, py: f32, ax: f32, ay: f32, bx: f32, by: f32, r: f32) -> f32 {
    let (pax, pay) = (px - ax, py - ay);
    let (bax, bay) = (bx - ax, by - ay);
    let denom = bax * bax + bay * bay;
    let h = if denom > 0.0 { ((pax * bax + pay * bay) / denom).clamp(0.0, 1.0) } else { 0.0 };
    let (dx, dy) = (pax - bax * h, pay - bay * h);
    (dx * dx + dy * dy).sqrt() - r
}

/// Render one stamp. Segments are evaluated at x-1, x and x+1 so the result
/// tiles horizontally and can be dragged along a seam without a visible join.
fn render(stamp: &Stamp, size: u32) -> (image::GrayImage, image::GrayImage) {
    let mut opacity = image::GrayImage::new(size, size);
    let mut height = image::GrayImage::new(size, size);
    // Roughly one pixel of antialiasing, in UV units.
    let aa = 1.0 / size as f32;

    for y in 0..size {
        let v = (y as f32 + 0.5) / size as f32;
        for x in 0..size {
            let u = (x as f32 + 0.5) / size as f32;
            let mut best = f32::MAX;
            let mut best_r = 1.0f32;
            for &(ax, ay, bx, by, r) in &stamp.segments {
                for wrap in [-1.0f32, 0.0, 1.0] {
                    let d = sd_capsule(u, v, ax + wrap, ay, bx + wrap, by, r);
                    if d < best {
                        best = d;
                        best_r = r;
                    }
                }
            }
            // Coverage: 1 inside the shape, feathering out over ~1px.
            let cov = ((-best) / aa).clamp(0.0, 1.0);
            opacity.put_pixel(x, y, image::Luma([(cov * 255.0) as u8]));

            // Height: a dome across the segment's cross-section, so the stamp
            // reads as raised thread rather than a flat sticker.
            let t = ((-best) / best_r).clamp(0.0, 1.0);
            let dome = (1.0 - (1.0 - t) * (1.0 - t)).sqrt();
            height.put_pixel(x, y, image::Luma([(dome * cov * 255.0) as u8]));
        }
    }
    (opacity, height)
}

/// Generate every procedural stamp into `<root>/brushes/<name>/`.
pub fn generate(root: &Path, size: u32) -> Result<usize> {
    let all = stamps();
    for stamp in &all {
        let dir = root.join("brushes").join(stamp.name);
        std::fs::create_dir_all(&dir)?;
        let (opacity, height) = render(stamp, size);
        opacity.save(dir.join("opacity.png"))?;
        height.save(dir.join("height.png"))?;

        let manifest = Manifest {
            id: format!("proc:{}", stamp.name),
            name: stamp.name.replace('_', " "),
            source: "procedural".into(),
            kind: "brush".into(),
            license: "CC0".into(),
            authors: vec!["zestsurface matfetch".into()],
            page: String::new(),
            variant: format!("{size}px"),
            tags: stamp.tags.iter().map(|s| s.to_string()).collect(),
            maps: HashMap::from([
                ("opacity".to_string(), "opacity.png".to_string()),
                ("height".to_string(), "height.png".to_string()),
            ]),
        };
        let mut value = serde_json::to_value(&manifest)?;
        value["description"] = serde_json::Value::String(stamp.description.into());
        std::fs::write(dir.join("material.json"), serde_json::to_vec_pretty(&value)?)?;
        eprintln!("[brush] {} ({}px)", stamp.name, size);
    }
    Ok(all.len())
}
