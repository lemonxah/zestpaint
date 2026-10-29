//! Left dock: how you paint. The tip, the paint channels (or the shader-mask
//! box while a mask layer is active) and the fill-mode footer.

use egui::{RichText, Ui};

use super::widgets::{self, SliderRow};
use super::{AppState, Tool};
use crate::layers::ShaderMask;
use crate::paint::PaintTarget;
use crate::theme;

/// Width of the mono value field at the end of a channel row.
const VALUE_WIDTH: f32 = 40.0;
/// Width of the checkbox + channel name column.
const CHECK_WIDTH: f32 = 96.0;

impl AppState {
    pub(super) fn brush_dock(&mut self, ctx: &egui::Context) {
        if !self.ui.panels.brush {
            return;
        }
        egui::SidePanel::left("brush_dock")
            .exact_width(268.0)
            .resizable(false)
            .frame(
                egui::Frame::new()
                    .fill(theme::BG_PANEL)
                    .inner_margin(egui::Margin::same(theme::DOCK_PADDING as i8))
                    .stroke(egui::Stroke::new(1.0_f32, theme::LINE)),
            )
            .show(ctx, |ui| {
                // Footer first, so it pins to the bottom.
                egui::TopBottomPanel::bottom("brush_footer")
                    .frame(egui::Frame::new().inner_margin(egui::Margin::symmetric(0, 8)))
                    .show_separator_line(true)
                    .show_inside(ui, |ui| self.fill_mode_footer(ui));

                egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                    ui.horizontal(|ui| {
                        let title = match self.tool {
                            Tool::Erase => "Eraser",
                            Tool::Smudge => "Smudge",
                            Tool::Stamp => "Stamp",
                            Tool::Clone => "Clone",
                            _ => "Brush",
                        };
                        ui.label(theme::section_header(title));
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.menu_button(crate::icons::MORE, |ui| {
                                ui.label(theme::section_header("Built-in brushes"));
                                for (i, preset) in super::brush_presets::PRESETS.iter().enumerate() {
                                    if ui.button(preset.name).clicked() {
                                        self.apply_preset(i);
                                        ui.close_menu();
                                    }
                                }
                                ui.separator();
                                if ui.button("Reset brush settings").clicked() {
                                    let tip = self.brush.tip.clone();
                                    let name = self.brush.tip_name.clone();
                                    self.brush = crate::paint::BrushSettings::default();
                                    self.brush.tip = tip;
                                    self.brush.tip_name = name;
                                    ui.close_menu();
                                }
                                ui.add_enabled(false, egui::Button::new("Save preset…"))
                                    .on_disabled_hover_text("Coming soon");
                            })
                            .response
                            .on_hover_text("Brush presets");
                        });
                    });
                    ui.add_space(6.0);
                    self.tip_block(ui);
                    ui.add_space(4.0);
                    self.preset_strip(ui);
                    widgets::divider(ui);

                    let shader_mask = (self.brush.target == PaintTarget::Material)
                        .then(|| self.active_shader_mask())
                        .flatten();
                    match shader_mask {
                        Some(kind) => shader_mask_box(ui, kind, &mut self.brush.shader_mask_value),
                        None => self.paint_channels(ui),
                    }

                    widgets::divider(ui);
                    let layers = &mut self.texture_sets[self.active_set].layers;
                    if SliderRow::new("Bump strength", &mut layers.height_scale, 0.5..=20.0)
                        .hint("How strongly painted height turns into surface normals")
                        .label_width(110.0)
                        .show(ui)
                        .changed()
                    {
                        layers.mark_dirty();
                    }
                });
            });
    }

    fn tip_block(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            // 64 px preview: the tip itself, or a soft round dot.
            let (rect, _) = ui.allocate_exact_size(egui::vec2(64.0, 64.0), egui::Sense::hover());
            ui.painter().rect_filled(rect, egui::CornerRadius::same(6), theme::BG_FIELD);
            if self.tip_preview.is_none()
                && let Some(image) = self.tip_preview_image.clone()
            {
                self.tip_preview = Some(ui.ctx().load_texture("stamp_preview", image, egui::TextureOptions::LINEAR));
            }
            match (&self.brush.tip, &self.tip_preview) {
                (Some(_), Some(texture)) => {
                    egui::Image::new(texture).paint_at(ui, rect.shrink(6.0));
                }
                _ => {
                    // The round brush: its falloff at the current hardness,
                    // redrawn only when hardness moves.
                    let hardness = self.brush.hardness;
                    let stale = self.ui.round_preview.as_ref().is_none_or(|(h, _)| (*h - hardness).abs() > 0.005);
                    if stale {
                        let n = 128;
                        let rgba: Vec<u8> = super::brush_presets::round_coverage(hardness)
                            .iter()
                            .flat_map(|&c| [255, 255, 255, c])
                            .collect();
                        let image = egui::ColorImage::from_rgba_unmultiplied([n, n], &rgba);
                        let texture = ui.ctx().load_texture("round_preview", image, egui::TextureOptions::LINEAR);
                        self.ui.round_preview = Some((hardness, texture));
                    }
                    if let Some((_, texture)) = &self.ui.round_preview {
                        egui::Image::new(texture).paint_at(ui, rect.shrink(8.0));
                    }
                }
            }
            ui.vertical(|ui| {
                let name = self.brush.tip_name.clone().unwrap_or_else(|| "Round".into());
                ui.label(RichText::new(name).font(theme::semibold(theme::BODY_SIZE)));
                ui.horizontal_wrapped(|ui| {
                    if ui.button("Browse…").on_hover_text("Open the brush library").clicked() {
                        self.open_library(true);
                    }
                    if ui.button("Load…").on_hover_text("Greyscale tip image").clicked() {
                        self.load_brush_tip();
                    }
                    if ui
                        .add_enabled(self.brush.tip.is_some() || self.brush.tip_name.is_some(), egui::Button::new("Reset"))
                        .on_hover_text("Back to the plain round brush")
                        .clicked()
                    {
                        self.brush.tip = None;
                        self.brush.tip_name = None;
                        self.brush.tip_aspect = 1.0;
                        self.tip_preview_image = None;
                        self.tip_preview = None;
                        self.brush.tip_has_color = false;
                        self.brush.use_tip_color = false;
                    }
                });
            });
        });
        ui.add_enabled(
            self.brush.tip_has_color,
            egui::Checkbox::new(&mut self.brush.use_tip_color, "Use tip colours"),
        )
        .on_hover_text("Stamp the tip's own colours — green leaves stay green instead of becoming leaf-shaped paint")
        .on_disabled_hover_text("This tip is greyscale");
    }

    fn paint_channels(&mut self, ui: &mut Ui) {
        let b = &mut self.brush;
        let on = [b.paint_color, b.paint_roughness, b.paint_metallic, b.paint_height, b.paint_emissive, b.paint_opacity]
            .iter()
            .filter(|on| **on)
            .count();
        widgets::section(ui, "Paint channels", Some(&format!("{on} of 6 on")));

        ui.horizontal(|ui| {
            widgets::fixed(ui, CHECK_WIDTH, |ui| ui.checkbox(&mut b.paint_color, "Color"));
            ui.add_enabled_ui(b.paint_color, |ui| {
                ui.color_edit_button_rgb(&mut b.color);
                let [r, g, bl] = b.color.map(|c| (c.clamp(0.0, 1.0) * 255.0).round() as u8);
                ui.label(RichText::new(format!("#{r:02X}{g:02X}{bl:02X}")).monospace().color(theme::TEXT_MUTED));
            });
        });
        channel_slider(ui, &mut b.paint_roughness, "Roughness", &mut b.roughness_value, 0.0..=1.0, None);
        channel_slider(ui, &mut b.paint_metallic, "Metallic", &mut b.metallic_value, 0.0..=1.0, None);
        channel_slider(ui, &mut b.paint_height, "Height", &mut b.height_value, -1.0..=1.0, Some("Positive raises, negative carves"));
        ui.horizontal(|ui| {
            widgets::fixed(ui, CHECK_WIDTH, |ui| ui.checkbox(&mut b.paint_emissive, "Emissive"));
            ui.add_enabled_ui(b.paint_emissive, |ui| {
                ui.color_edit_button_rgb(&mut b.emissive_color).on_hover_text("Glow colour");
                let gap = ui.spacing().item_spacing.x;
                let width = ui.available_width() - VALUE_WIDTH - gap;
                widgets::slider(ui, &mut b.emissive_intensity, 0.0..=10.0, false, width.max(30.0))
                    .on_hover_text("Intensity");
                ui.add_sized(
                    [VALUE_WIDTH, 20.0],
                    egui::DragValue::new(&mut b.emissive_intensity).range(0.0..=10.0).speed(0.05).fixed_decimals(1),
                );
            });
        });
        channel_slider(ui, &mut b.paint_opacity, "Opacity", &mut b.opacity_value, 0.0..=1.0, Some("0 cuts the surface away, 1 is opaque"));
    }

    fn fill_mode_footer(&mut self, ui: &mut Ui) {
        if self.brush.fill_mode.is_on() {
            self.ensure_mesh_groups();
        }
        let mode = self.brush.fill_mode.label();
        let (text, sizes) = match self.mesh_groups.as_ref() {
            Some(groups) if self.brush.fill_mode.is_on() => {
                let biggest: Vec<String> = groups.shell_sizes.iter().take(4).map(|n| n.to_string()).collect();
                (
                    format!("mesh has {} piece(s), {} UV island(s)", groups.shell_count, groups.island_count),
                    format!(
                        "Largest pieces (triangles): {}{}",
                        biggest.join(", "),
                        if groups.shell_sizes.len() > 4 { ", …" } else { "" }
                    ),
                )
            }
            _ => ("Triangle, UV island and Whole piece fill by mesh connectivity".into(), String::new()),
        };
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing.x = 4.0;
            ui.label(RichText::new("Fill mode:").color(theme::TEXT_MUTED));
            ui.label(RichText::new(mode).color(theme::TEXT));
            ui.label(RichText::new(format!("· {text}")).color(theme::TEXT_MUTED));
        })
        .response
        .on_hover_text(format!("{sizes}\nPieces are welded by position, so a UV seam does not split one"));
    }
}

/// Checkbox, name, slider and value for one scalar channel; the slider dims
/// when the channel is off.
fn channel_slider(
    ui: &mut Ui,
    on: &mut bool,
    name: &str,
    value: &mut f32,
    range: std::ops::RangeInclusive<f32>,
    hint: Option<&str>,
) {
    ui.horizontal(|ui| {
        let check = widgets::fixed(ui, CHECK_WIDTH, |ui| ui.checkbox(on, name));
        if let Some(hint) = hint {
            check.on_hover_text(hint);
        }
        ui.add_enabled_ui(*on, |ui| {
            let gap = ui.spacing().item_spacing.x;
            let width = ui.available_width() - VALUE_WIDTH - gap;
            widgets::slider(ui, value, range.clone(), false, width.max(40.0));
            ui.add_sized(
                [VALUE_WIDTH, 20.0],
                egui::DragValue::new(value).range(range).speed(0.005).fixed_decimals(2),
            );
        });
    });
}

/// While a shader-mask layer is active the material channels do not apply:
/// a framed box in the kind's colour with just the grey value instead.
fn shader_mask_box(ui: &mut Ui, kind: ShaderMask, value: &mut f32) {
    let [r, g, b] = kind.tint();
    let colour = egui::Color32::from_rgb(r, g, b);
    egui::Frame::new()
        .stroke(egui::Stroke::new(1.5_f32, colour))
        .corner_radius(egui::CornerRadius::same(theme::RADIUS_ROW))
        .inner_margin(egui::Margin::same(10))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                widgets::badge(ui, kind.tag(), egui::Color32::BLACK, colour);
                ui.label(RichText::new(format!("Painting {}", kind.label().to_lowercase())).font(theme::semibold(theme::BODY_SIZE)));
            });
            ui.label(RichText::new(kind.description()).small().color(theme::TEXT_MUTED));
            ui.add_space(4.0);
            widgets::segmented(ui, value, &[(1.0, "White — show"), (0.0, "Black — hide")]);
            SliderRow::new("Grey", value, 0.0..=1.0).label_width(40.0).show(ui);
            ui.label(
                RichText::new("Material channel settings don't apply to mask layers")
                    .small()
                    .color(theme::TEXT_DIM),
            );
        });
}
