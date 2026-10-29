//! The unified Library window: one OS window (so it can live on a second
//! monitor) with Materials and Brushes & stamps tabs, a sources list, a
//! thumbnail grid and a details panel.

use egui::{Color32, CornerRadius, RichText, Sense, Stroke, StrokeKind, Ui};

use super::widgets;
use super::{filter_assets, upload_thumbnails, AppState};
use crate::icons;
use crate::library::LibraryMaterial;
use crate::scene::MapSlot;
use crate::theme;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LibraryTab {
    Materials,
    Brushes,
}

/// Which slice of a library the sources list picked.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Source {
    All,
    PolyHaven,
    AmbientCg,
    Mine,
    Generated,
    Imported,
    Images,
}

impl Source {
    fn label(self) -> &'static str {
        match self {
            Source::All => "All",
            Source::PolyHaven => "Poly Haven",
            Source::AmbientCg => "ambientCG",
            Source::Mine => "My materials",
            Source::Generated => "Generated",
            Source::Imported => "Imported sets",
            Source::Images => "Images",
        }
    }

    fn of(item: &LibraryMaterial, tab: LibraryTab) -> Source {
        let source = item.source.as_deref().unwrap_or("");
        match tab {
            LibraryTab::Materials => match source {
                "polyhaven" => Source::PolyHaven,
                "ambientcg" => Source::AmbientCg,
                "" | "user" => Source::Mine,
                _ => Source::Generated,
            },
            LibraryTab::Brushes => match source {
                "imported" => Source::Imported,
                _ => Source::Images,
            },
        }
    }

    fn list(tab: LibraryTab) -> &'static [Source] {
        match tab {
            LibraryTab::Materials => &[Source::All, Source::PolyHaven, Source::AmbientCg, Source::Mine, Source::Generated],
            LibraryTab::Brushes => &[Source::All, Source::Imported, Source::Images],
        }
    }
}

const COLUMNS: usize = 5;

impl AppState {
    pub(super) fn library_window_ui(&mut self, ctx: &egui::Context) {
        let brush = self.ui.library_tab == LibraryTab::Brushes;

        // Scan lazily, and keep a thumbnail worker per library.
        if brush {
            if !self.ui.brushes_scanned {
                self.ui.brushes = crate::library::scan_brushes();
                self.ui.brushes_scanned = true;
            }
            let loader = self.ui.brush_loader.get_or_insert_with(|| crate::library::ThumbnailLoader::new(2));
            upload_thumbnails(ctx, loader, &mut self.ui.brush_thumbs);
        } else {
            if !self.ui.library_scanned {
                self.ui.library = crate::library::scan();
                self.ui.library_scanned = true;
            }
            let loader = self.ui.library_loader.get_or_insert_with(|| crate::library::ThumbnailLoader::new(2));
            upload_thumbnails(ctx, loader, &mut self.ui.library_thumbs);
        }
        let tab = self.ui.library_tab;
        let source = self.ui.library_source[tab as usize];

        let mut rescan = false;
        let mut activate: Option<usize> = None;

        // Title bar: tabs, and what is being painted into.
        egui::TopBottomPanel::top("lib_title")
            .exact_height(46.0)
            .frame(egui::Frame::new().fill(theme::BG_CHROME).inner_margin(egui::Margin::symmetric(14, 8)))
            .show(ctx, |ui| {
                ui.horizontal_centered(|ui| {
                    ui.label(RichText::new("Library").font(theme::semibold(13.0)).color(theme::TEXT_STRONG));
                    ui.add_space(8.0);
                    widgets::segmented(
                        ui,
                        &mut self.ui.library_tab,
                        &[(LibraryTab::Materials, "Materials"), (LibraryTab::Brushes, "Brushes & stamps")],
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let set = &self.texture_sets[self.active_set];
                        ui.label(RichText::new(format!("{} › {}", set.object, set.material)).color(theme::TEXT));
                        ui.label(RichText::new("Painting into").color(theme::TEXT_MUTED));
                    });
                });
            });

        // Toolbar: filter, tag chips, count, Clear, Rescan (+ imports).
        let (items, filter_text) = if brush {
            (&self.ui.brushes, self.ui.brush_filter.clone())
        } else {
            (&self.ui.library, self.ui.library_filter.clone())
        };
        let matching: Vec<usize> = filter_assets(items, &filter_text)
            .into_iter()
            .filter(|&i| source == Source::All || Source::of(&items[i], tab) == source)
            .collect();
        let total = items.len();
        let counts: Vec<(Source, usize)> = Source::list(tab)
            .iter()
            .map(|s| {
                let n = match s {
                    Source::All => items.len(),
                    _ => items.iter().filter(|i| Source::of(i, tab) == *s).count(),
                };
                (*s, n)
            })
            .collect();

        egui::TopBottomPanel::top("lib_toolbar")
            .frame(
                egui::Frame::new()
                    .fill(theme::BG_BAR)
                    .inner_margin(egui::Margin::symmetric(14, 8))
                    .stroke(Stroke::new(1.0_f32, theme::LINE)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    let (filter, hint, chips): (&mut String, &str, &[&str]) = if brush {
                        (&mut self.ui.brush_filter, "name or tag — stitch, zipper, seam, leak, wrinkle", &["stitch", "zipper", "seam", "leak", "wrinkle"])
                    } else {
                        (&mut self.ui.library_filter, "name or tag — fabric, lace, net, wood, metal", &["fabric", "lace", "net", "wood", "metal"])
                    };
                    ui.label(RichText::new(icons::SEARCH).color(theme::TEXT_DIM));
                    ui.add(egui::TextEdit::singleline(filter).hint_text(hint).desired_width(280.0));
                    for chip in chips {
                        let on = filter.split_whitespace().any(|t| t.eq_ignore_ascii_case(chip));
                        let button = egui::Button::new(RichText::new(*chip).color(if on { theme::TEXT_STRONG } else { theme::TEXT_MUTED }))
                            .fill(if on { theme::BG_SELECTED } else { Color32::TRANSPARENT })
                            .stroke(Stroke::new(1.0_f32, if on { theme::ACCENT } else { theme::LINE_CONTROL }))
                            .corner_radius(CornerRadius::same(12));
                        if ui.add(button).clicked() {
                            // Toggle the tag as a filter term.
                            let mut terms: Vec<String> = filter.split_whitespace().map(str::to_string).collect();
                            if on {
                                terms.retain(|t| !t.eq_ignore_ascii_case(chip));
                            } else {
                                terms.push(chip.to_string());
                            }
                            *filter = terms.join(" ");
                        }
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("Rescan").clicked() {
                            rescan = true;
                        }
                        if ui.button("Clear").clicked() {
                            filter.clear();
                        }
                        ui.label(RichText::new(format!("{} of {total}", matching.len())).monospace().color(theme::TEXT_MUTED));
                    });
                });
                if brush {
                    ui.horizontal(|ui| {
                        if ui
                            .button(format!("{} Import brush set…", icons::OPEN))
                            .on_hover_text("Photoshop .abr, GIMP .gbr/.gih, or plain alpha images. Each stamp becomes its own library entry.")
                            .clicked()
                        {
                            rescan |= self.import_brush_files();
                        }
                        if ui.button(format!("{} Import folder…", icons::OPEN)).on_hover_text("Import every brush file in a folder").clicked() {
                            rescan |= self.import_brush_folder();
                        }
                        ui.label(RichText::new(&self.ui.brush_import_status).color(theme::TEXT_MUTED));
                    });
                }
            });

        if rescan {
            if brush {
                self.ui.brushes = crate::library::scan_brushes();
                self.ui.brush_selected = None;
                self.ui.brush_thumbs.clear();
                if let Some(l) = self.ui.brush_loader.as_mut() {
                    l.reset();
                }
            } else {
                self.ui.library = crate::library::scan();
                self.ui.library_selected = None;
                self.ui.library_thumbs.clear();
                if let Some(l) = self.ui.library_loader.as_mut() {
                    l.reset();
                }
            }
            return;
        }

        // Sources, with the folder in the footer.
        egui::SidePanel::left("lib_sources")
            .exact_width(220.0)
            .resizable(false)
            .frame(
                egui::Frame::new()
                    .fill(theme::BG_PANEL)
                    .inner_margin(egui::Margin::same(10))
                    .stroke(Stroke::new(1.0_f32, theme::LINE)),
            )
            .show(ctx, |ui| {
                ui.label(theme::section_header("Sources"));
                ui.add_space(4.0);
                for (s, n) in &counts {
                    if *s == Source::Generated && *n == 0 {
                        continue;
                    }
                    let on = source == *s;
                    let width = ui.available_width();
                    let (rect, response) = ui.allocate_exact_size(egui::vec2(width, 30.0), Sense::click());
                    if on {
                        ui.painter().rect_filled(rect, CornerRadius::same(theme::RADIUS_ROW), theme::BG_SEG_ON);
                    } else if response.hovered() {
                        ui.painter().rect_filled(rect, CornerRadius::same(theme::RADIUS_ROW), Color32::from_white_alpha(8));
                    }
                    ui.painter().text(
                        rect.left_center() + egui::vec2(10.0, 0.0),
                        egui::Align2::LEFT_CENTER,
                        s.label(),
                        egui::FontId::proportional(theme::BODY_SIZE),
                        if on { theme::TEXT_STRONG } else { theme::TEXT },
                    );
                    ui.painter().text(
                        rect.right_center() - egui::vec2(10.0, 0.0),
                        egui::Align2::RIGHT_CENTER,
                        n.to_string(),
                        egui::FontId::monospace(11.0),
                        theme::TEXT_DIM,
                    );
                    if response.clicked() {
                        self.ui.library_source[tab as usize] = *s;
                    }
                }
                ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                    ui.label(RichText::new("Fill it with matfetch.").small().color(theme::TEXT_DIM));
                    let dir = if brush { crate::library::brush_dir() } else { crate::library::library_dir() };
                    ui.add(egui::Label::new(RichText::new(format!("Folder: {}", dir.display())).monospace().small().color(theme::TEXT_DIM)).wrap());
                });
            });

        // Details for the selection.
        egui::SidePanel::right("lib_details")
            .exact_width(300.0)
            .resizable(false)
            .frame(
                egui::Frame::new()
                    .fill(theme::BG_PANEL)
                    .inner_margin(egui::Margin::same(14))
                    .stroke(Stroke::new(1.0_f32, theme::LINE)),
            )
            .show(ctx, |ui| {
                let (items, thumbs, selected, prefix) = if brush {
                    (&self.ui.brushes, &self.ui.brush_thumbs, self.ui.brush_selected, "brush_")
                } else {
                    (&self.ui.library, &self.ui.library_thumbs, self.ui.library_selected, "mat_")
                };
                let Some(item) = selected.and_then(|i| items.get(i)) else {
                    ui.label(RichText::new("Select an entry to preview it.").color(theme::TEXT_MUTED));
                    return;
                };
                let size = ui.available_width();
                match thumbs.get(&format!("{prefix}{}", item.name)) {
                    Some(tex) => {
                        ui.add(egui::Image::new((tex.id(), egui::vec2(size, size))).corner_radius(CornerRadius::same(8)));
                    }
                    None => {
                        let (rect, _) = ui.allocate_exact_size(egui::vec2(size, size), Sense::hover());
                        ui.painter().rect_filled(rect, CornerRadius::same(8), theme::BG_FIELD);
                    }
                }
                ui.add_space(8.0);
                ui.label(RichText::new(&item.display).font(theme::semibold(theme::TITLE_SIZE)).color(theme::TEXT_STRONG));
                ui.add_space(4.0);
                ui.label(RichText::new(if brush { "Tip" } else { "Maps" }).color(theme::TEXT_MUTED));
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing.x = 4.0;
                    for (slot, _) in &item.maps {
                        widgets::badge(ui, slot_tag(*slot), theme::TEXT_MUTED, theme::BG_BUTTON).on_hover_text(slot.label());
                    }
                });
                if !item.tags.is_empty() {
                    ui.add_space(4.0);
                    ui.label(RichText::new("Tags").color(theme::TEXT_MUTED));
                    ui.label(item.tags.join(", "));
                }
                ui.add_space(4.0);
                ui.label(RichText::new("Source").color(theme::TEXT_MUTED));
                ui.add(egui::Label::new(RichText::new(item.dir.display().to_string()).monospace().small().color(theme::TEXT_DIM)).wrap())
                    .on_hover_text("Folder these maps came from");

                ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                    let hint = if brush {
                        "Loads this stamp as the brush tip."
                    } else {
                        "Adds a new layer with a black mask. Paint the mask to reveal it."
                    };
                    ui.label(RichText::new(hint).color(theme::TEXT_MUTED));
                    let action = if brush { "Use as brush tip" } else { "Apply as fill layer" };
                    if widgets::wide(ui, widgets::primary_button(action)).clicked() {
                        activate = selected;
                    }
                });
            });

        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(theme::BG_APP).inner_margin(egui::Margin::same(16)))
            .show(ctx, |ui| {
                let (items, thumbs, loader, selected, prefix) = if brush {
                    (&self.ui.brushes, &self.ui.brush_thumbs, self.ui.brush_loader.as_mut().unwrap(), &mut self.ui.brush_selected, "brush_")
                } else {
                    (&self.ui.library, &self.ui.library_thumbs, self.ui.library_loader.as_mut().unwrap(), &mut self.ui.library_selected, "mat_")
                };
                grid(ui, items, &matching, thumbs, loader, prefix, selected, &mut activate);
            });

        if let Some(i) = activate {
            if brush {
                self.use_library_brush(i);
            } else {
                self.apply_library_material(i);
            }
        }
    }
}

/// The short tag a map slot shows as.
fn slot_tag(slot: MapSlot) -> &'static str {
    match slot {
        MapSlot::BaseColor => "COL",
        MapSlot::Roughness => "RGH",
        MapSlot::Metallic => "MTL",
        MapSlot::Normal => "NRM",
        MapSlot::Height => "HGT",
        MapSlot::Ao => "AO",
        MapSlot::Emission => "EMI",
        MapSlot::EmissionMask => "EMK",
        MapSlot::Opacity => "OPA",
    }
}

/// A virtualised five-column thumbnail grid. Only rows on screen request
/// thumbnails, which keeps a few thousand entries scrolling smoothly.
#[allow(clippy::too_many_arguments)]
fn grid(
    ui: &mut Ui,
    items: &[LibraryMaterial],
    matching: &[usize],
    thumbs: &std::collections::HashMap<String, egui::TextureHandle>,
    loader: &mut crate::library::ThumbnailLoader,
    prefix: &str,
    selected: &mut Option<usize>,
    activate: &mut Option<usize>,
) {
    let gap = 14.0;
    let cell_w = ((ui.available_width() - gap * (COLUMNS as f32 - 1.0)) / COLUMNS as f32).max(60.0);
    let thumb = cell_w - 12.0;
    let cell_h = thumb + 40.0;
    let rows = matching.len().div_ceil(COLUMNS);
    egui::ScrollArea::vertical()
        .id_salt(prefix)
        .auto_shrink([false, false])
        .show_rows(ui, cell_h + gap, rows, |ui, range| {
            for row in range {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = gap;
                    for col in 0..COLUMNS {
                        let Some(&idx) = matching.get(row * COLUMNS + col) else { break };
                        let item = &items[idx];
                        let key = format!("{prefix}{}", item.name);
                        if !thumbs.contains_key(&key)
                            && let Some(src) = item.preview_source()
                        {
                            loader.request(&key, src);
                        }
                        let (rect, response) = ui.allocate_exact_size(egui::vec2(cell_w, cell_h), Sense::click());
                        let is_selected = *selected == Some(idx);
                        let painter = ui.painter();
                        if is_selected {
                            painter.rect(rect, CornerRadius::same(theme::RADIUS_ROW + 1), theme::BG_SELECTED, Stroke::new(1.5_f32, theme::ACCENT), StrokeKind::Inside);
                        } else if response.hovered() {
                            painter.rect_filled(rect, CornerRadius::same(theme::RADIUS_ROW + 1), Color32::from_white_alpha(6));
                        }
                        let image_rect = egui::Rect::from_min_size(rect.min + egui::vec2(6.0, 6.0), egui::vec2(thumb, thumb));
                        match thumbs.get(&key) {
                            Some(tex) => {
                                egui::Image::new((tex.id(), image_rect.size()))
                                    .corner_radius(CornerRadius::same(6))
                                    .paint_at(ui, image_rect);
                            }
                            None => {
                                ui.painter().rect_filled(image_rect, CornerRadius::same(6), theme::BG_FIELD);
                            }
                        }
                        let text = ui.painter().layout(
                            item.display.clone(),
                            egui::FontId::proportional(theme::BODY_SIZE),
                            if is_selected { theme::TEXT_STRONG } else { theme::TEXT },
                            thumb,
                        );
                        let text_pos = egui::pos2(image_rect.left(), image_rect.bottom() + 8.0);
                        ui.painter().with_clip_rect(rect).galley(text_pos, text, theme::TEXT);
                        let response = response.on_hover_text(&item.display);
                        if response.clicked() {
                            *selected = Some(idx);
                        }
                        if response.double_clicked() {
                            *activate = Some(idx);
                        }
                    }
                });
                ui.add_space(gap);
            }
        });
}
