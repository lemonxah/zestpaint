//! Export maps and Import maps dialogs. Naming, colour space and alpha rules
//! are `crate::export`'s, unchanged; these only choose what goes where.

use egui::{RichText, Ui};

use super::overlays::ToastAction;
use super::widgets::{self, Tone};
use super::AppState;
use crate::export::{ExportContents, LayerScope};
use crate::layers::MapKind;
use crate::theme;

/// A saved choice of maps for the export dialog.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ExportPreset {
    AllPbr,
    VrchatPoiyomi,
    /// Every layer on its own, nothing flattened.
    LayersOnly,
    /// Every layer mask and shader mask, nothing else.
    MasksOnly,
    /// Flattened maps, every layer, every mask.
    Everything,
    LastUsed,
}

impl ExportPreset {
    const ALL: [ExportPreset; 6] = [
        ExportPreset::AllPbr,
        ExportPreset::VrchatPoiyomi,
        ExportPreset::LayersOnly,
        ExportPreset::MasksOnly,
        ExportPreset::Everything,
        ExportPreset::LastUsed,
    ];

    fn label(self) -> &'static str {
        match self {
            ExportPreset::AllPbr => "All PBR maps",
            ExportPreset::VrchatPoiyomi => "VRChat · Poiyomi",
            ExportPreset::LayersOnly => "Layers only",
            ExportPreset::MasksOnly => "Masks only",
            ExportPreset::Everything => "Everything",
            ExportPreset::LastUsed => "Last used",
        }
    }
}

/// The export dialog's "Last used" settings.
struct LastUsed {
    maps: Vec<MapKind>,
    contents: ExportContents,
    folder: String,
}

/// A settings file in the app's config dir (see `crate::config`).
pub(super) fn config_file(name: &str) -> std::path::PathBuf {
    crate::config::file(name)
}

/// Where export presets live.
fn presets_path() -> std::path::PathBuf {
    config_file("export_presets.json")
}

/// A map from its export suffix, including the ones saved in presets
/// before the suffixes were renamed.
fn kind_from_suffix(suffix: &str) -> Option<MapKind> {
    match suffix {
        "basecolor" => Some(MapKind::BaseColor),
        "normal" => Some(MapKind::Normal),
        "emissive" => Some(MapKind::Emissive),
        _ => MapKind::ALL.into_iter().find(|k| k.suffix() == suffix),
    }
}

/// The "Last used" preset from disk. Files from before layer exports
/// existed read as flattened maps only.
fn load_last_used() -> Option<LastUsed> {
    let text = std::fs::read_to_string(presets_path()).ok()?;
    let value: serde_json::Value = serde_json::from_str(&text).ok()?;
    let last = value.get("last_used")?;
    let maps = last
        .get("maps")?
        .as_array()?
        .iter()
        .filter_map(|m| m.as_str().and_then(kind_from_suffix))
        .collect();
    let flag = |key: &str, default: bool| last.get(key).and_then(|v| v.as_bool()).unwrap_or(default);
    let scope = match last.get("layer_scope").and_then(|v| v.as_str()) {
        Some("visible") => LayerScope::Visible,
        Some("active") => LayerScope::Active,
        _ => LayerScope::All,
    };
    let contents = ExportContents {
        flattened: flag("flattened", true),
        shader_masks: flag("shader_masks", false),
        layers: flag("layers", false),
        layer_masks: flag("layer_masks", false),
        scope,
        size: last.get("size").and_then(|v| v.as_u64()).map(|s| s as u32),
    };
    let folder = last.get("folder").and_then(|v| v.as_str()).unwrap_or_default().to_string();
    Some(LastUsed { maps, contents, folder })
}

fn save_last_used(maps: &[MapKind], contents: &ExportContents, folder: &str) {
    let value = serde_json::json!({
        "last_used": {
            "maps": maps.iter().map(|k| k.suffix()).collect::<Vec<_>>(),
            "flattened": contents.flattened,
            "size": contents.size,
            "shader_masks": contents.shader_masks,
            "layers": contents.layers,
            "layer_masks": contents.layer_masks,
            "layer_scope": match contents.scope {
                LayerScope::All => "all",
                LayerScope::Visible => "visible",
                LayerScope::Active => "active",
            },
            "folder": folder,
        }
    });
    let path = presets_path();
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let _ = std::fs::write(path, serde_json::to_string_pretty(&value).unwrap_or_default());
}

pub(super) fn dialog_frame() -> egui::Frame {
    egui::Frame::new()
        .fill(theme::BG_PANEL)
        .stroke(egui::Stroke::new(1.0_f32, theme::LINE_INNER))
        .corner_radius(egui::CornerRadius::same(theme::RADIUS_POPOVER + 2))
        .inner_margin(egui::Margin::same(16))
}

/// Dialog title row with a close button; true when closed.
fn title_row(ui: &mut Ui, title: &str, right: impl FnOnce(&mut Ui)) -> bool {
    let mut close = false;
    ui.horizontal(|ui| {
        ui.label(RichText::new(title).font(theme::semibold(theme::TITLE_SIZE)).color(theme::TEXT_STRONG));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            close = ui.add(egui::Button::new(crate::icons::REMOVE).frame(false)).clicked();
            right(ui);
        });
    });
    widgets::divider(ui);
    close
}

impl AppState {
    /// What exported files are named after: the dialog's Name field, or
    /// the project / model name while that is left empty.
    pub(super) fn export_name(&self) -> String {
        match self.ui.export_name.trim() {
            "" => self.ui.model_name.clone(),
            name => name.to_string(),
        }
    }

    /// What the export dialog is set to write, besides maps and sets.
    pub(super) fn export_contents(&self) -> ExportContents {
        ExportContents {
            flattened: self.ui.export_flattened,
            shader_masks: self.ui.export_shader_masks,
            layers: self.ui.export_layers,
            layer_masks: self.ui.export_layer_masks,
            scope: self.ui.export_layer_scope,
            size: self.ui.export_size,
        }
    }

    fn set_export_contents(&mut self, contents: ExportContents) {
        self.ui.export_flattened = contents.flattened;
        self.ui.export_shader_masks = contents.shader_masks;
        self.ui.export_layers = contents.layers;
        self.ui.export_layer_masks = contents.layer_masks;
        self.ui.export_layer_scope = contents.scope;
        self.ui.export_size = contents.size;
    }

    fn apply_export_preset(&mut self, preset: ExportPreset) {
        self.ui.export_preset = preset;
        let only = |flattened, shader_masks, layers, layer_masks| ExportContents {
            flattened,
            shader_masks,
            layers,
            layer_masks,
            scope: LayerScope::All,
            size: self.ui.export_size,
        };
        match preset {
            ExportPreset::AllPbr => {
                self.ui.export_maps = MapKind::ALL.to_vec();
                self.set_export_contents(only(true, false, false, false));
            }
            // Poiyomi takes colour (with alpha cutout), normal and emission,
            // and reads the rest of its effects from greyscale masks.
            ExportPreset::VrchatPoiyomi => {
                self.ui.export_maps = vec![MapKind::BaseColor, MapKind::Normal, MapKind::Emissive];
                self.set_export_contents(only(true, true, false, false));
            }
            ExportPreset::LayersOnly => {
                self.ui.export_maps = MapKind::ALL.to_vec();
                self.set_export_contents(only(false, false, true, false));
            }
            ExportPreset::MasksOnly => {
                self.set_export_contents(only(false, true, false, true));
            }
            ExportPreset::Everything => {
                self.ui.export_maps = MapKind::ALL.to_vec();
                self.set_export_contents(only(true, true, true, true));
            }
            ExportPreset::LastUsed => match load_last_used() {
                Some(last) => {
                    self.ui.export_maps = last.maps;
                    self.set_export_contents(last.contents);
                    if !last.folder.is_empty() {
                        self.ui.export_dir = last.folder;
                    }
                }
                None => self.ui.status = "No last-used export settings yet".into(),
            },
        }
    }

    pub(super) fn export_dialog(&mut self, ctx: &egui::Context) {
        if !self.ui.export_open {
            return;
        }
        self.ui.export_sets.resize(self.texture_sets.len(), true);
        let mut close = false;
        let mut go = false;
        let modal = egui::Modal::new("export_maps".into()).frame(dialog_frame()).show(ctx, |ui| {
            ui.set_width(640.0);
            let mut preset = self.ui.export_preset;
            close = title_row(ui, "Export", |ui| {
                egui::ComboBox::from_id_salt("export_preset")
                    .selected_text(preset.label())
                    .show_ui(ui, |ui| {
                        for p in ExportPreset::ALL {
                            ui.selectable_value(&mut preset, p, p.label());
                        }
                    });
                ui.label(RichText::new("Preset").color(theme::TEXT_MUTED));
            });
            if preset != self.ui.export_preset {
                self.apply_export_preset(preset);
            }

            // Export size, asked up front: paint big for detail, ship smaller.
            // The document keeps its resolution either way.
            let largest = self.texture_sets.iter().map(|s| s.layers.resolution).max().unwrap_or(2048);
            if self.ui.export_size.is_some_and(|s| s >= largest) {
                self.ui.export_size = None;
            }
            ui.label(theme::section_header("Export size"));
            ui.horizontal(|ui| {
                let native = format!("{largest} px (as painted)");
                let mut options: Vec<(Option<u32>, String)> = vec![(None, native)];
                options.extend([4096, 2048, 1024, 512].into_iter().filter(|&s| s < largest).map(|s| (Some(s), format!("{s} px"))));
                let options: Vec<(Option<u32>, &str)> = options.iter().map(|(v, t)| (*v, t.as_str())).collect();
                widgets::segmented(ui, &mut self.ui.export_size, &options);
            });
            ui.label(
                RichText::new(if largest > 2048 {
                    "Downscaled with Lanczos, alpha-aware. Painting at 4K/8K and exporting 2K keeps lines finer than painting at 2K."
                } else {
                    "Smaller sizes are downscaled (Lanczos, alpha-aware); raise a set's Resolution for finer painting."
                })
                .small()
                .color(theme::TEXT_DIM),
            );
            widgets::divider(ui);

            ui.columns(2, |cols| {
                // Texture sets.
                let ui = &mut cols[0];
                ui.horizontal(|ui| {
                    ui.label(theme::section_header("Texture sets"));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.add(egui::Button::new("Active").frame(false)).clicked() {
                            self.ui.export_sets.fill(false);
                            self.ui.export_sets[self.active_set] = true;
                        }
                        if ui.add(egui::Button::new("None").frame(false)).clicked() {
                            self.ui.export_sets.fill(false);
                        }
                        if ui.add(egui::Button::new("All").frame(false)).clicked() {
                            self.ui.export_sets.fill(true);
                        }
                    });
                });
                egui::ScrollArea::vertical().id_salt("export_sets").max_height(200.0).show(ui, |ui| {
                    for (set, on) in self.texture_sets.iter().zip(&mut self.ui.export_sets) {
                        ui.horizontal(|ui| {
                            ui.checkbox(on, "");
                            ui.spacing_mut().item_spacing.x = 2.0;
                            ui.label(RichText::new(format!("{} ›", set.object)).color(theme::TEXT_MUTED));
                            ui.label(&set.material);
                        });
                    }
                });

                // Maps, two to a row.
                let ui = &mut cols[1];
                ui.horizontal(|ui| {
                    ui.label(theme::section_header("Maps"));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.add(egui::Button::new("None").frame(false)).clicked() {
                            self.ui.export_maps.clear();
                        }
                        if ui.add(egui::Button::new("All").frame(false)).clicked() {
                            self.ui.export_maps = MapKind::ALL.to_vec();
                        }
                    });
                });
                egui::Grid::new("export_map_grid").num_columns(2).spacing([16.0, 6.0]).show(ui, |ui| {
                    for (i, kind) in MapKind::ALL.into_iter().enumerate() {
                        let mut on = self.ui.export_maps.contains(&kind);
                        let label = if kind == MapKind::Ao { "AO (baked)" } else { kind.label() };
                        if ui.checkbox(&mut on, label).changed() {
                            self.ui.export_maps.retain(|k| *k != kind);
                            if on {
                                self.ui.export_maps.push(kind);
                            }
                        }
                        if i % 2 == 1 {
                            ui.end_row();
                        }
                    }
                });
                widgets::divider(ui);
                let found: usize = self
                    .texture_sets
                    .iter()
                    .zip(&self.ui.export_sets)
                    .filter(|(_, on)| **on)
                    .map(|(s, _)| s.layers.shader_mask_kinds().len())
                    .sum();
                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.ui.export_shader_masks, "Shader masks")
                        .on_hover_text("Every Matcap/Emission/… mask the set has, as greyscale");
                    ui.label(RichText::new(format!("one PNG each · {found} found")).color(theme::TEXT_MUTED));
                });
            });

            widgets::divider(ui);
            ui.label(theme::section_header("Contents"));
            ui.horizontal_wrapped(|ui| {
                ui.checkbox(&mut self.ui.export_flattened, "Flattened maps")
                    .on_hover_text("The whole stack flattened: one PNG per map, what a game engine takes");
                ui.checkbox(&mut self.ui.export_layers, "Layers")
                    .on_hover_text(
                        "Each layer on its own: one PNG per checked map it has content in, coverage in \
                         alpha. Import maps brings each back as a layer.",
                    );
                ui.checkbox(&mut self.ui.export_layer_masks, "Layer masks")
                    .on_hover_text("Each layer's mask as greyscale (white shows the layer)");
                ui.add_enabled_ui(self.ui.export_layers || self.ui.export_layer_masks, |ui| {
                    egui::ComboBox::from_id_salt("export_layer_scope")
                        .selected_text(self.ui.export_layer_scope.label())
                        .show_ui(ui, |ui| {
                            for scope in LayerScope::ALL {
                                ui.selectable_value(&mut self.ui.export_layer_scope, scope, scope.label());
                            }
                        });
                });
            });

            widgets::divider(ui);
            ui.horizontal(|ui| {
                widgets::fixed(ui, 56.0, |ui| ui.label(RichText::new("Name").color(theme::TEXT_MUTED)));
                let model_name = self.ui.model_name.clone();
                ui.add(
                    egui::TextEdit::singleline(&mut self.ui.export_name)
                        .hint_text(model_name)
                        .font(egui::TextStyle::Monospace)
                        .desired_width(ui.available_width() - 90.0),
                )
                .on_hover_text(
                    "Every file starts with this: Name_color.png, Name_normals.png, Name_matcap_mask.png… \
                     With several texture sets, each set's material is added after it.",
                );
            });
            ui.horizontal(|ui| {
                widgets::fixed(ui, 56.0, |ui| ui.label(RichText::new("Folder").color(theme::TEXT_MUTED)));
                let width = ui.available_width() - 90.0;
                ui.add(
                    egui::TextEdit::singleline(&mut self.ui.export_dir)
                        .hint_text("Pick a folder…")
                        .font(egui::TextStyle::Monospace)
                        .desired_width(width),
                );
                if ui.button("Browse…").clicked()
                    && let Some(dir) = rfd::FileDialog::new().pick_folder()
                {
                    self.ui.export_dir = dir.display().to_string();
                }
            });

            // What will be written.
            let contents = self.export_contents();
            let files = crate::export::planned_files(
                &self.texture_sets,
                &self.ui.export_sets,
                &self.export_name(),
                &self.ui.export_maps,
                &contents,
            );
            egui::Frame::new()
                .fill(theme::BG_FIELD)
                .corner_radius(egui::CornerRadius::same(theme::RADIUS_ROW))
                .inner_margin(egui::Margin::same(10))
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    if files.is_empty() {
                        ui.label(RichText::new("Nothing selected").monospace().color(theme::TEXT_DIM));
                    }
                    for (name, note) in files.iter().take(4) {
                        ui.horizontal(|ui| {
                            ui.label(RichText::new(name).monospace());
                            ui.label(RichText::new(*note).monospace().color(theme::TEXT_DIM));
                        });
                    }
                    if files.len() > 4 {
                        ui.label(RichText::new(format!("+ {} more", files.len() - 4)).monospace().color(theme::TEXT_DIM));
                    }
                });

            widgets::divider(ui);
            let sets = self.ui.export_sets.iter().filter(|on| **on).count();
            let count = |note: &str| files.iter().filter(|(_, n)| *n == note).count();
            let masks = count(crate::export::SHADER_MASK_NOTE) + count(crate::export::LAYER_MASK_NOTE);
            let layer_maps = count(crate::export::LAYER_NOTE);
            let maps = files.len() - masks - layer_maps;
            let summary = match (maps, layer_maps) {
                (_, 0) => format!("{sets} sets · {maps} maps + {masks} masks = "),
                (0, _) => format!("{sets} sets · {layer_maps} layer maps + {masks} masks = "),
                _ => format!("{sets} sets · {maps} maps + {layer_maps} layer maps + {masks} masks = "),
            };
            ui.horizontal(|ui| {
                ui.label(RichText::new(summary).color(theme::TEXT_MUTED));
                ui.label(RichText::new(format!("{} files", files.len())).color(theme::TEXT_STRONG));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    go = ui
                        .add_enabled(
                            !files.is_empty(),
                            widgets::primary_button(format!(
                                "Export {} files at {} px",
                                files.len(),
                                self.ui.export_size.unwrap_or(largest)
                            )),
                        )
                        .clicked();
                    if ui.button("Cancel").clicked() {
                        close = true;
                    }
                });
            });
        });
        if close || modal.should_close() {
            self.ui.export_open = false;
        }
        if go {
            self.export_maps();
        }
    }

    /// Export to the dialog's folder (asking for one if it is empty).
    pub(super) fn export_maps(&mut self) {
        let dir = if self.ui.export_dir.trim().is_empty() {
            let Some(dir) = rfd::FileDialog::new().pick_folder() else {
                return;
            };
            self.ui.export_dir = dir.display().to_string();
            dir
        } else {
            let raw = self.ui.export_dir.trim();
            let expanded = match raw.strip_prefix("~/") {
                Some(rest) => std::env::var("HOME").map(|h| std::path::PathBuf::from(h).join(rest)).unwrap_or(raw.into()),
                None => raw.into(),
            };
            if let Err(e) = std::fs::create_dir_all(&expanded) {
                self.ui.status = format!("Export failed: could not create {}: {e}", expanded.display());
                self.push_toast(Tone::Error, self.ui.status.clone(), None);
                return;
            }
            expanded
        };
        self.ui.export_open = false;
        let contents = self.export_contents();
        let name = self.export_name();
        save_last_used(&self.ui.export_maps, &contents, &self.ui.export_dir);
        match crate::export::export_all(
            &self.gpu,
            &mut self.texture_sets,
            &self.ui.export_sets,
            &name,
            &self.ui.export_maps,
            &contents,
            &dir,
        ) {
            Ok(files) => {
                self.ui.status = format!("Exported {} files to {}", files.len(), dir.display());
                self.push_toast(Tone::Ok, self.ui.status.clone(), Some(ToastAction::OpenFolder(dir)));
            }
            Err(e) => {
                self.ui.status = "Export failed".into();
                self.push_toast(Tone::Error, "Export failed", Some(ToastAction::Details(format!("{e:#}"))));
            }
        }
    }

    pub(super) fn import_dialog(&mut self, ctx: &egui::Context) {
        if !self.ui.import_open {
            return;
        }
        self.ui.import_set = self.ui.import_set.min(self.texture_sets.len() - 1);
        let mut close = false;
        let mut go = false;
        let mut add = false;
        let modal = egui::Modal::new("import_maps".into()).frame(dialog_frame()).show(ctx, |ui| {
            ui.set_width(520.0);
            close = title_row(ui, "Import maps", |_| {});
            ui.horizontal(|ui| {
                widgets::fixed(ui, 56.0, |ui| ui.label(RichText::new("Onto").color(theme::TEXT_MUTED)));
                egui::ComboBox::from_id_salt("import_set")
                    .width(ui.available_width())
                    .selected_text(format!(
                        "{} › {}",
                        self.texture_sets[self.ui.import_set].object, self.texture_sets[self.ui.import_set].material
                    ))
                    .show_ui(ui, |ui| {
                        for (i, set) in self.texture_sets.iter().enumerate() {
                            ui.selectable_value(&mut self.ui.import_set, i, format!("{} › {}", set.object, set.material));
                        }
                    });
            });
            ui.add_space(4.0);
            egui::ScrollArea::vertical().id_salt("import_files").max_height(260.0).show(ui, |ui| {
                for (i, (path, kind)) in self.ui.import_files.iter_mut().enumerate() {
                    let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
                    ui.horizontal(|ui| {
                        let colour = if kind.is_some() { theme::TEXT } else { theme::WARN_FG };
                        ui.label(RichText::new(name).monospace().color(colour));
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            let text = match kind {
                                Some(k) => RichText::new(k.label()),
                                None => RichText::new("Choose map…").color(theme::WARN_FG),
                            };
                            egui::ComboBox::from_id_salt(("import_kind", i))
                                .width(150.0)
                                .selected_text(text)
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(kind, None, "Skip");
                                    for k in MapKind::ALL.into_iter().filter(|k| k.importable()) {
                                        ui.selectable_value(kind, Some(k), k.label());
                                    }
                                });
                        });
                    });
                }
                if self.ui.import_files.is_empty() {
                    ui.label(RichText::new("No files yet — Add files…").color(theme::TEXT_DIM));
                }
            });
            ui.label(
                RichText::new("Each map becomes a new layer on top. Existing layers are not touched.")
                    .color(theme::TEXT_MUTED),
            );
            widgets::divider(ui);
            let count = self.ui.import_files.iter().filter(|(_, k)| k.is_some()).count();
            ui.horizontal(|ui| {
                add = ui.add(egui::Button::new("Add files…").frame(false)).clicked();
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    go = ui
                        .add_enabled(count > 0, widgets::primary_button(format!("Import {count} maps")))
                        .clicked();
                    if ui.button("Cancel").clicked() {
                        close = true;
                    }
                });
            });
        });
        if add {
            self.add_import_files();
        }
        if go {
            self.ui.import_open = false;
            self.import_maps();
        } else if close || modal.should_close() {
            self.ui.import_open = false;
            self.ui.import_files.clear();
        }
    }

    /// File ▸ Import maps…: pick files, then confirm them in the dialog.
    pub(super) fn pick_import_maps(&mut self) {
        self.ui.import_files.clear();
        self.ui.import_set = self.active_set;
        self.add_import_files();
        if !self.ui.import_files.is_empty() {
            self.ui.import_open = true;
        }
    }

    /// Add picked files, each with the map its name suggests (editable).
    fn add_import_files(&mut self) {
        let Some(paths) = rfd::FileDialog::new()
            .add_filter("Texture maps", &["png", "jpg", "jpeg", "tga", "webp", "tif", "tiff"])
            .pick_files()
        else {
            return;
        };
        for p in paths {
            if self.ui.import_files.iter().any(|(q, _)| *q == p) {
                continue;
            }
            let name = p.file_name().unwrap_or_default().to_string_lossy().to_string();
            let kind = MapKind::detect(&name).filter(|k| k.importable());
            self.ui.import_files.push((p, kind));
        }
    }
}
