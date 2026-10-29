//! Built-in brushes: the round brushes every paint app starts with (hard,
//! medium, soft, airbrush) and a few procedurally generated textured tips,
//! so there is something to paint with before any brush set is imported.

use egui::{Color32, CornerRadius, Sense, Stroke, StrokeKind, Ui};

use super::AppState;
use crate::theme;

/// Edge length of generated tips, in texels.
const TIP_SIZE: u32 = 128;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TipShape {
    /// No tip image: the plain round brush, softened by hardness alone.
    Round,
    /// Bell-curve falloff, softer in the middle than smoothstep.
    Gaussian,
    /// A disc with grainy coverage, like chalk on rough paper.
    Chalk,
    /// Scattered droplets.
    Spatter,
    /// Many small bristle ends in a loose cluster.
    Bristle,
}

pub struct Preset {
    pub name: &'static str,
    pub shape: TipShape,
    pub hardness: f32,
    pub flow: f32,
}

pub const PRESETS: [Preset; 8] = [
    Preset { name: "Hard round", shape: TipShape::Round, hardness: 1.0, flow: 1.0 },
    Preset { name: "Medium round", shape: TipShape::Round, hardness: 0.5, flow: 0.6 },
    Preset { name: "Soft round", shape: TipShape::Round, hardness: 0.0, flow: 0.4 },
    Preset { name: "Airbrush", shape: TipShape::Round, hardness: 0.0, flow: 0.08 },
    Preset { name: "Gaussian soft", shape: TipShape::Gaussian, hardness: 1.0, flow: 0.35 },
    Preset { name: "Chalk", shape: TipShape::Chalk, hardness: 1.0, flow: 0.7 },
    Preset { name: "Spatter", shape: TipShape::Spatter, hardness: 1.0, flow: 0.9 },
    Preset { name: "Bristle", shape: TipShape::Bristle, hardness: 1.0, flow: 0.5 },
];

/// Small deterministic RNG, so generated tips are the same every run.
struct Rng(u32);

impl Rng {
    fn next(&mut self) -> f32 {
        // xorshift32
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        (self.0 >> 8) as f32 / (1u32 << 24) as f32
    }
}

/// Hash noise at a lattice point, 0..1.
fn lattice(x: i32, y: i32) -> f32 {
    let mut h = (x as u32).wrapping_mul(374_761_393) ^ (y as u32).wrapping_mul(668_265_263);
    h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
    (h ^ (h >> 16)) as f32 / u32::MAX as f32
}

/// Smooth value noise, 0..1.
fn value_noise(x: f32, y: f32) -> f32 {
    let (xi, yi) = (x.floor() as i32, y.floor() as i32);
    let (fx, fy) = (x - x.floor(), y - y.floor());
    let s = |t: f32| t * t * (3.0 - 2.0 * t);
    let (sx, sy) = (s(fx), s(fy));
    let top = lattice(xi, yi) * (1.0 - sx) + lattice(xi + 1, yi) * sx;
    let bottom = lattice(xi, yi + 1) * (1.0 - sx) + lattice(xi + 1, yi + 1) * sx;
    top * (1.0 - sy) + bottom * sy
}

/// Coverage for a generated shape, 255 = full paint. None for `Round`,
/// which needs no tip image at all.
pub fn generate(shape: TipShape) -> Option<Vec<u8>> {
    let n = TIP_SIZE as usize;
    let mut cov = vec![0f32; n * n];
    let at = |i: usize| {
        let (x, y) = ((i % n) as f32 + 0.5, (i / n) as f32 + 0.5);
        // -1..1 across the tip.
        (x / n as f32 * 2.0 - 1.0, y / n as f32 * 2.0 - 1.0)
    };
    match shape {
        TipShape::Round => return None,
        TipShape::Gaussian => {
            for (i, c) in cov.iter_mut().enumerate() {
                let (x, y) = at(i);
                let r2 = x * x + y * y;
                // sigma 0.38 of the radius; zero at the rim so it never clips.
                let g = (-r2 / (2.0 * 0.38 * 0.38)).exp();
                let rim = (-1.0f32 / (2.0 * 0.38 * 0.38)).exp();
                *c = ((g - rim) / (1.0 - rim)).max(0.0);
            }
        }
        TipShape::Chalk => {
            for (i, c) in cov.iter_mut().enumerate() {
                let (x, y) = at(i);
                let r = (x * x + y * y).sqrt();
                let disc = 1.0 - smoothstep(0.75, 0.98, r);
                let grain = value_noise(x * 22.0 + 7.0, y * 22.0 + 3.0) * 0.6
                    + value_noise(x * 60.0, y * 60.0) * 0.4;
                *c = disc * smoothstep(0.35, 0.6, grain);
            }
        }
        TipShape::Spatter | TipShape::Bristle => {
            let (count, small, big, spread) = match shape {
                TipShape::Spatter => (34, 0.03, 0.16, 0.85),
                _ => (90, 0.025, 0.06, 0.8),
            };
            let mut rng = Rng(if shape == TipShape::Spatter { 0x9E37_79B9 } else { 0x85EB_CA6B });
            for _ in 0..count {
                // Denser towards the middle.
                let a = rng.next() * std::f32::consts::TAU;
                let d = rng.next().sqrt() * spread;
                let (cx, cy) = (a.cos() * d, a.sin() * d);
                let radius = small + (big - small) * rng.next().powi(2);
                let strength = 0.6 + 0.4 * rng.next();
                for (i, c) in cov.iter_mut().enumerate() {
                    let (x, y) = at(i);
                    let r = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();
                    let dot = (1.0 - smoothstep(radius * 0.7, radius, r)) * strength;
                    *c = c.max(dot);
                }
            }
        }
    }
    Some(cov.iter().map(|c| (c.clamp(0.0, 1.0) * 255.0).round() as u8).collect())
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// What a preset looks like, as the thumbnail and the dock preview show it:
/// its tip, or for round brushes the falloff the hardness gives.
pub fn preview(preset: &Preset) -> egui::ColorImage {
    let n = TIP_SIZE as usize;
    let coverage = generate(preset.shape).unwrap_or_else(|| round_coverage(preset.hardness));
    let rgba: Vec<u8> = coverage.iter().flat_map(|&c| [255, 255, 255, c]).collect();
    egui::ColorImage::from_rgba_unmultiplied([n, n], &rgba)
}

/// The plain round brush's coverage: the same smoothstep the paint shader
/// applies, `1 - smoothstep(hardness, 1, d / radius)`.
pub fn round_coverage(hardness: f32) -> Vec<u8> {
    let n = TIP_SIZE as usize;
    (0..n * n)
        .map(|i| {
            let (x, y) = ((i % n) as f32 + 0.5, (i / n) as f32 + 0.5);
            let (x, y) = (x / n as f32 * 2.0 - 1.0, y / n as f32 * 2.0 - 1.0);
            let d = (x * x + y * y).sqrt();
            let f = if d > 1.0 { 0.0 } else { 1.0 - smoothstep(hardness.min(0.999), 1.0, d) };
            (f * 255.0).round() as u8
        })
        .collect()
}

impl AppState {
    /// Switch the brush to a built-in preset: tip, hardness and flow.
    pub(super) fn apply_preset(&mut self, index: usize) {
        let Some(preset) = PRESETS.get(index) else { return };
        match generate(preset.shape) {
            Some(coverage) => {
                let rgba: Vec<u8> = coverage.iter().flat_map(|&c| [255, 255, 255, c]).collect();
                if let Some(image) = image::RgbaImage::from_raw(TIP_SIZE, TIP_SIZE, rgba) {
                    self.set_tip_image(&image, preset.name.to_string());
                }
            }
            None => {
                self.brush.tip = None;
                self.brush.tip_aspect = 1.0;
                self.tip_preview_image = None;
                self.tip_preview = None;
                self.brush.tip_has_color = false;
                self.brush.use_tip_color = false;
                self.brush.tip_name = Some(preset.name.to_string());
            }
        }
        self.brush.hardness = preset.hardness;
        self.brush.flow = preset.flow;
        self.ui.preset = Some(index);
        self.ui.status = format!("Brush: {}", preset.name);
    }

    /// The row of built-in brush thumbnails under the tip block.
    pub(super) fn preset_strip(&mut self, ui: &mut Ui) {
        if self.ui.preset_thumbs.is_empty() {
            self.ui.preset_thumbs = PRESETS
                .iter()
                .map(|p| ui.ctx().load_texture(format!("preset_{}", p.name), preview(p), egui::TextureOptions::LINEAR))
                .collect();
        }
        // The active preset, unless the tip or hardness has since moved away.
        let active = self.ui.preset.filter(|&i| {
            let p = &PRESETS[i];
            self.brush.tip_name.as_deref() == Some(p.name) && (p.shape != TipShape::Round || self.brush.hardness == p.hardness)
        });
        let mut picked = None;
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = egui::vec2(4.0, 4.0);
            for (i, (preset, texture)) in PRESETS.iter().zip(&self.ui.preset_thumbs).enumerate() {
                let (rect, response) = ui.allocate_exact_size(egui::vec2(52.0, 52.0), Sense::click());
                let on = active == Some(i);
                ui.painter().rect(
                    rect,
                    CornerRadius::same(theme::RADIUS_CONTROL),
                    theme::BG_FIELD,
                    Stroke::new(if on { 1.5_f32 } else { 1.0_f32 }, if on { theme::ACCENT } else if response.hovered() { theme::TEXT_DIM } else { theme::LINE_CONTROL }),
                    StrokeKind::Inside,
                );
                egui::Image::new(texture).tint(Color32::from_gray(235)).paint_at(ui, rect.shrink(7.0));
                let response = response.on_hover_text(format!(
                    "{} · hardness {:.2} · flow {:.2}",
                    preset.name, preset.hardness, preset.flow
                ));
                if response.clicked() {
                    picked = Some(i);
                }
            }
        });
        if let Some(i) = picked {
            self.apply_preset(i);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_tips_have_paint_and_stay_inside_the_disc() {
        let n = TIP_SIZE as usize;
        for shape in [TipShape::Gaussian, TipShape::Chalk, TipShape::Spatter, TipShape::Bristle] {
            let cov = generate(shape).expect("generated shape");
            assert_eq!(cov.len(), n * n);
            assert!(cov.iter().any(|&c| c > 200), "{shape:?} should have solid paint somewhere");
            // Corners lie outside the round brush's disc: nothing there.
            for corner in [0, n - 1, n * (n - 1), n * n - 1] {
                assert_eq!(cov[corner], 0, "{shape:?} paints outside the disc");
            }
        }
        assert!(generate(TipShape::Round).is_none(), "round brushes need no tip image");
    }

    #[test]
    fn round_preview_matches_hardness() {
        let n = TIP_SIZE as usize;
        let centre = n / 2 * n + n / 2;
        // Halfway out along the row through the centre.
        let half = n / 2 * n + n / 2 + n / 4;
        let hard = round_coverage(1.0);
        let soft = round_coverage(0.0);
        assert!(hard[centre] > 250 && soft[centre] > 250);
        assert!(hard[half] > 250, "a hard brush is solid to its edge");
        assert!(soft[half] < 200, "a soft brush has faded by half its radius");
    }

    #[test]
    fn presets_are_deterministic() {
        assert_eq!(generate(TipShape::Spatter), generate(TipShape::Spatter));
        assert_eq!(generate(TipShape::Chalk), generate(TipShape::Chalk));
    }
}
