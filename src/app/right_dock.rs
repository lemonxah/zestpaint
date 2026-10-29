//! Right dock: what you paint into. Texture sets, the layer stack, and the
//! active layer's properties as Layer / Mask / Projection / Material tabs.

use egui::{Color32, CornerRadius, RichText, Sense, Stroke, StrokeKind, Ui};

use super::commands::Cmd;
use super::widgets::{self, SliderRow};
use super::{AppState, PropsTab, Tool};
use crate::icons;
use crate::layers::{Axis, BlendMode, Layer, LayerStack, MaskSource, Projection, ShaderMask};
use crate::scene::MapSlot;
use crate::theme;

/// What a click in the texture-set list asks for; applied after the list
/// is drawn.
enum PieceAction {
    /// Select these (replace, or toggle into the selection with `extend`).
    Select(Vec<usize>, bool),
    SetVisible(Vec<usize>, bool),
    ShowOnly(Vec<usize>),
    IsolateSets(Vec<usize>),
    FrameSets(Vec<usize>),
    Isolate,
    Hide,
    UnhideAll,
    Frame,
    Delete(Vec<usize>),
}

const ROW_HEIGHT: f32 = 50.0;

/// What the "Mask from" picker chose.
#[derive(Clone, Copy)]
enum MaskChoice {
    Image,
    Source(MaskSource),
}

/// The "Mask from" picker's entries, for layer `target`: an image file,
/// another layer's mask or painted area, or a shader mask.
fn mask_source_menu(ui: &mut Ui, stack: &LayerStack, target: usize) -> Option<MaskChoice> {
    let mut chosen = None;
    let mut entry = |ui: &mut Ui, label: String, choice: MaskChoice| {
        if ui.add(egui::Button::new(label).frame(false)).clicked() {
            chosen = Some(choice);
            ui.close_menu();
        }
    };
    ui.set_min_width(220.0);
    entry(ui, format!("{}  Image file…", icons::OPEN), MaskChoice::Image);
    // Top of the stack first, like the layer list.
    let others: Vec<usize> = (0..stack.layers.len()).rev().filter(|&i| i != target).collect();
    let masked: Vec<usize> = others.iter().copied().filter(|&i| stack.layers[i].mask.is_some()).collect();
    if !masked.is_empty() {
        ui.separator();
        ui.label(RichText::new("Copy another layer's mask").color(theme::TEXT_DIM).small());
        for i in masked {
            entry(ui, stack.layers[i].name.clone(), MaskChoice::Source(MaskSource::LayerMask(i)));
        }
    }
    let painted: Vec<usize> = others.iter().copied().filter(|&i| stack.layers[i].used != 0).collect();
    if !painted.is_empty() {
        ui.separator();
        ui.label(RichText::new("Where a layer is painted").color(theme::TEXT_DIM).small());
        for i in painted {
            entry(ui, stack.layers[i].name.clone(), MaskChoice::Source(MaskSource::LayerCoverage(i)));
        }
    }
    let kinds = stack.shader_mask_kinds();
    if !kinds.is_empty() {
        ui.separator();
        ui.label(RichText::new("Shader masks").color(theme::TEXT_DIM).small());
        for kind in kinds {
            entry(ui, kind.label().to_string(), MaskChoice::Source(MaskSource::ShaderMask(kind)));
        }
    }
    chosen
}

impl AppState {
    pub(super) fn right_dock(&mut self, ctx: &egui::Context) {
        let p = self.ui.panels;
        if !(p.texture_sets || p.layers || p.properties) {
            return;
        }
        egui::SidePanel::right("right_dock")
            .default_width(330.0)
            .width_range(320.0..=340.0)
            .resizable(false)
            .frame(
                egui::Frame::new()
                    .fill(theme::BG_PANEL)
                    .inner_margin(egui::Margin::symmetric(theme::DOCK_PADDING as i8, 8))
                    .stroke(Stroke::new(1.0_f32, theme::LINE)),
            )
            .show(ctx, |ui| {
                let dock_height = ui.available_height();
                if p.texture_sets {
                    self.texture_sets_section(ui);
                    widgets::divider(ui);
                }
                if p.layers {
                    self.layers_section(ui, dock_height * 0.4);
                    widgets::divider(ui);
                }
                if p.properties {
                    self.properties_tabs(ui);
                }
            });
    }

    fn texture_sets_section(&mut self, ui: &mut Ui) {
        let mut open_model = false;
        let header = ui.horizontal(|ui| {
            let chevron = if self.ui.texture_sets_open { icons::MOVE_DOWN } else { icons::CHEVRON_RIGHT };
            let toggle = ui.add(
                egui::Button::new(RichText::new(format!("{chevron}  ")).color(theme::TEXT_MUTED)).frame(false),
            );
            if toggle.clicked() {
                self.ui.texture_sets_open = !self.ui.texture_sets_open;
            }
            // Two tabs share the panel: the set list and the blendshapes.
            let shapes = self.models.iter().any(|m| !m.morphs.targets.is_empty());
            for (blendshapes, title) in [(false, "Texture sets"), (true, "Blendshapes")] {
                let on = self.ui.show_blendshapes == blendshapes;
                let mut text = theme::section_header(title);
                if !on {
                    text = text.color(theme::TEXT_DIM);
                }
                let tab = ui.add(egui::Label::new(text).sense(Sense::click()));
                if on {
                    let r = tab.rect;
                    ui.painter().hline(r.x_range(), r.bottom() + 2.0, Stroke::new(2.0_f32, theme::ACCENT));
                }
                let tab = if blendshapes && !shapes { tab.on_hover_text("This model has no blendshapes") } else { tab };
                if tab.clicked() {
                    self.ui.show_blendshapes = blendshapes;
                    self.ui.texture_sets_open = true;
                }
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                open_model = ui
                    .add(egui::Button::new(RichText::new(format!("{}  Open model", icons::OPEN)).color(theme::TEXT_MUTED)).frame(false))
                    .on_hover_text("Open an .fbx / .gltf  (Ctrl O)")
                    .clicked();
            });
        });
        let _ = header;
        if open_model {
            self.open_model_dialog();
        }
        if !self.ui.texture_sets_open {
            return;
        }
        if self.ui.show_blendshapes {
            self.blendshapes_ui(ui);
            return;
        }

        self.pieces_toolbar(ui);

        // Objects in load order, each with its materials (texture sets).
        let mut objects: Vec<(String, Vec<usize>)> = Vec::new();
        for (i, set) in self.texture_sets.iter().enumerate() {
            match objects.iter_mut().find(|(o, _)| *o == set.object) {
                Some((_, ids)) => ids.push(i),
                None => objects.push((set.object.clone(), vec![i])),
            }
        }
        let mut action = None;
        egui::ScrollArea::vertical().id_salt("texture_sets").max_height(280.0).show(ui, |ui| {
            for (object, set_ids) in objects {
                if let Some(a) = self.object_row(ui, &object, &set_ids) {
                    action = Some(a);
                }
                for &i in &set_ids {
                    if let Some(a) = self.set_row(ui, i) {
                        action = Some(a);
                    }
                }
            }
        });
        if let Some(action) = action {
            self.run_piece_action(action);
        }
        self.resolution_picker(ui);
    }

    /// The active set's matcap layers, top first: one click selects one
    /// (its matcap is edited in Layer properties), and a button adds one.
    fn matcaps_ui(&mut self, ui: &mut Ui) {
        let layers = &self.texture_sets[self.active_set].layers;
        let mut select = None;
        let mut any = false;
        for (i, layer) in layers.layers.iter().enumerate().rev().filter(|(_, l)| l.is_matcap()) {
            any = true;
            let detail = match layer.matcap.as_ref().filter(|m| m.image.is_some()) {
                Some(m) => format!("{} · {} {:.0}%", m.name, m.blend.label(), m.strength * 100.0),
                None => "no matcap picked".into(),
            };
            let text = RichText::new(format!("{}  —  {detail}", layer.name))
                .color(if layer.visible { theme::TEXT } else { theme::TEXT_DIM });
            if ui.selectable_label(layers.active == i, text).on_hover_text("Select this matcap layer").clicked() {
                select = Some(i);
            }
        }
        if !any {
            ui.label(RichText::new("No matcap layers on this material yet.").small().color(theme::TEXT_DIM));
        }
        if let Some(i) = select {
            self.texture_sets[self.active_set].layers.active = i;
            self.ui.props_tab = PropsTab::Layer;
        }
        if ui
            .button(format!("{} New matcap layer", icons::ADD))
            .on_hover_text("A layer holding a matcap; paint white where it shows")
            .clicked()
        {
            self.add_shader_mask(ShaderMask::Matcap);
        }
        ui.label(
            RichText::new(
                "Each matcap layer has its own mask, drawn bottom to top over the shading. \
                 Raster view only (view-dependent; not in path traced mode).",
            )
            .small()
            .color(theme::TEXT_DIM),
        );
    }

    /// Paint resolution of the active set, with what each size costs.
    fn resolution_picker(&mut self, ui: &mut Ui) {
        let set = self.active_set;
        let current = self.texture_sets[set].layers.resolution;
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        let mut pick = None;
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label(RichText::new("Resolution").color(theme::TEXT_MUTED))
                .on_hover_text("Texels per side of this set's paint layers. Higher = finer detail (tattoos, linework), but every size step uses 4x the GPU memory.");
            egui::ComboBox::from_id_salt("set_resolution")
                .selected_text(format!("{current} px"))
                .show_ui(ui, |ui| {
                    for res in super::RESOLUTIONS {
                        let (bytes, too_big) = self.resolution_cost(set, res);
                        let text = format!("{res} px ({}K) · ≈{:.1} GiB", res / 1024, gib(bytes));
                        let item = ui.add_enabled(!too_big, egui::Button::new(text).selected(res == current).frame(false));
                        let item = if too_big {
                            item.on_disabled_hover_text("Too much for this GPU with the current layers — it would freeze the desktop")
                        } else {
                            item
                        };
                        if item.clicked() {
                            pick = Some(res);
                            ui.close_menu();
                        }
                    }
                });
            let (bytes, _) = self.resolution_cost(set, current);
            let total = self.paint_vram_bytes();
            let vram = self.vram_bytes();
            let colour = if vram > 0 && total > vram / 4 * 3 { theme::WARN_FG } else { theme::TEXT_DIM };
            ui.label(RichText::new(format!("≈{:.1} GiB · all {:.1}/{:.0}", gib(bytes), gib(total), gib(vram))).small().color(colour))
                .on_hover_text(format!(
                    "This set ≈{:.1} GiB at {current} px. All paint textures ≈{:.1} GiB (undo keeps {:.1} GiB of earlier resolutions) of {:.1} GiB on this GPU. Sizes that would pass 75% are refused.",
                    gib(bytes),
                    gib(total),
                    gib(self.history_vram_bytes()),
                    gib(vram)
                ));
        });
        if current >= 8192 {
            ui.label(
                RichText::new("8K: each stroke's undo snapshot is ~0.5 GB, so fewer steps fit in undo history")
                    .small()
                    .color(theme::TEXT_DIM),
            );
        }
        if let Some(res) = pick
            && let Err(e) = self.set_texture_resolution(set, res)
        {
            self.ui.status = format!("{e:#}");
            self.push_toast(widgets::Tone::Error, format!("{e:#}"), None);
        }
    }

    /// Selection count and the Blender view actions, as buttons.
    /// Blendshape sliders, per object, with a name filter and a reset.
    fn blendshapes_ui(&mut self, ui: &mut Ui) {
        let groups = self.blendshape_groups();
        if groups.is_empty() {
            if let Some(job) = &self.shape_import {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(RichText::new(format!("Reading {}…", job.file_name())).color(theme::TEXT_MUTED));
                });
                return;
            }
            ui.label(
                RichText::new(if self.project_path.is_some() {
                    "No blendshapes in this project — it was saved before ZestPaint kept them. Take them from the model file; your painting stays."
                } else {
                    "This model has no blendshapes."
                })
                .color(theme::TEXT_MUTED),
            );
            let beside = self.shape_source_beside_project();
            if let Some(path) = &beside {
                let name = path.file_name().unwrap_or_default().to_string_lossy().into_owned();
                if ui.add(widgets::primary_button(&format!("Load blendshapes from {name}"))).clicked() {
                    self.start_shape_import(path);
                }
            }
            if self.project_path.is_some() && ui.button("Load blendshapes from another file…").clicked() {
                let mut dialog = rfd::FileDialog::new().add_filter("3D models", &["fbx", "glb", "gltf"]);
                if let Some(dir) = self.project_path.as_ref().and_then(|p| p.parent()) {
                    dialog = dialog.set_directory(dir);
                }
                if let Some(path) = dialog.pick_file() {
                    self.start_shape_import(&path);
                }
            }
            return;
        }
        let mut reset = false;
        ui.horizontal(|ui| {
            ui.add(
                egui::TextEdit::singleline(&mut self.ui.blendshape_filter)
                    .hint_text(format!("{}  Filter shapes", icons::SEARCH))
                    .desired_width(ui.available_width() - 80.0),
            );
            reset = ui.button("Reset all").on_hover_text("Every blendshape back to 0").clicked();
        });
        let filter = self.ui.blendshape_filter.trim().to_lowercase();
        let mut changes: Vec<(String, String, f32)> = Vec::new();
        egui::ScrollArea::vertical().id_salt("blendshapes").max_height(320.0).show(ui, |ui| {
            for (object, shapes) in &groups {
                let shown: Vec<&(String, f32)> =
                    shapes.iter().filter(|(n, _)| filter.is_empty() || n.to_lowercase().contains(&filter)).collect();
                if shown.is_empty() {
                    continue;
                }
                let on = shapes.iter().filter(|(_, w)| *w != 0.0).count();
                ui.label(
                    RichText::new(format!("{object}  ·  {} shapes{}", shapes.len(), if on > 0 { format!(", {on} on") } else { String::new() }))
                        .font(theme::semibold(theme::BODY_SIZE))
                        .color(theme::TEXT_MUTED),
                );
                let label_width = (ui.available_width() * 0.45).max(80.0);
                for (name, weight) in shown {
                    let mut w = *weight;
                    let row = ui.push_id((object, name), |ui| {
                        widgets::SliderRow::new(name, &mut w, 0.0..=1.0).hint(name).label_width(label_width).show(ui)
                    });
                    if row.inner.double_clicked() {
                        w = 0.0;
                    }
                    if w != *weight {
                        changes.push((object.clone(), name.clone(), w));
                    }
                }
                ui.add_space(4.0);
            }
        });
        if reset {
            for (object, shapes) in &groups {
                for (name, _) in shapes {
                    changes.push((object.clone(), name.clone(), 0.0));
                }
            }
        }
        for (object, name, w) in changes {
            self.set_blendshape(&object, &name, w);
        }
    }

    fn pieces_toolbar(&mut self, ui: &mut Ui) {
        let mut action = None;
        ui.horizontal(|ui| {
            let n = self.selection.len();
            let text = match n {
                0 => RichText::new("Click to select · Shift/Ctrl adds").small().color(theme::TEXT_DIM),
                1 => RichText::new("1 selected").small().color(theme::ACCENT),
                n => RichText::new(format!("{n} selected")).small().color(theme::ACCENT),
            };
            ui.label(text);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.small_button("Show all").on_hover_text("Unhide everything  (Alt H)").clicked() {
                    action = Some(PieceAction::UnhideAll);
                }
                if ui.small_button("Hide").on_hover_text("Hide the selection, or the active piece  (H)").clicked() {
                    action = Some(PieceAction::Hide);
                }
                let local = self.isolate_backup.is_some();
                let isolate = egui::Button::new(
                    RichText::new(if local { "Exit isolate" } else { "Isolate" }).color(if local { theme::ON_ACCENT } else { theme::TEXT }),
                )
                .small()
                .fill(if local { theme::ACCENT } else { theme::BG_BUTTON });
                if ui
                    .add(isolate)
                    .on_hover_text(if local {
                        "Leave local view: restore visibility and camera  (/)"
                    } else {
                        "Show only the selection and frame it  (/)"
                    })
                    .clicked()
                {
                    action = Some(PieceAction::Isolate);
                }
                if ui.small_button("Frame").on_hover_text("Frame the selection  (F · Home frames all)").clicked() {
                    action = Some(PieceAction::Frame);
                }
            });
        });
        ui.add_space(2.0);
        if let Some(action) = action {
            self.run_piece_action(action);
        }
    }

    /// An eye toggle with a real hit target: 24 px, hover background.
    fn eye_button(ui: &mut Ui, centre: egui::Pos2, visible: bool, tip: &str) -> egui::Response {
        let rect = egui::Rect::from_center_size(centre, egui::vec2(24.0, 22.0));
        let response = ui.interact(rect, ui.id().with(("eye", centre.x as i32, centre.y as i32)), Sense::click());
        if response.hovered() {
            ui.painter().rect_filled(rect, CornerRadius::same(theme::RADIUS_CONTROL), Color32::from_white_alpha(18));
        }
        let (icon, colour) = if visible { (icons::VISIBLE, theme::TEXT) } else { (icons::HIDDEN, theme::TEXT_DIM.gamma_multiply(0.7)) };
        ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, icon, egui::FontId::proportional(14.0), colour);
        response.on_hover_text(tip)
    }

    /// An object (FBX node) heading: click selects all its pieces.
    fn object_row(&mut self, ui: &mut Ui, object: &str, set_ids: &[usize]) -> Option<PieceAction> {
        let width = ui.available_width();
        let (rect, response) = ui.allocate_exact_size(egui::vec2(width, 24.0), Sense::click());
        let any_visible = set_ids.iter().any(|&i| self.texture_sets[i].visible);
        let all_selected = set_ids.iter().all(|i| self.selection.contains(i));
        let radius = CornerRadius::same(theme::RADIUS_ROW);
        if all_selected {
            ui.painter().rect_filled(rect, radius, theme::ACCENT.gamma_multiply(0.10));
        } else if response.hovered() {
            ui.painter().rect_filled(rect, radius, Color32::from_white_alpha(8));
        }
        let colour = if any_visible { theme::TEXT_MUTED } else { theme::TEXT_DIM.gamma_multiply(0.6) };
        ui.painter().text(
            egui::pos2(rect.left() + 6.0, rect.center().y),
            egui::Align2::LEFT_CENTER,
            object,
            theme::semibold(theme::BODY_SIZE),
            colour,
        );
        let eye = Self::eye_button(ui, egui::pos2(rect.right() - 16.0, rect.center().y), any_visible, "Show/hide the whole object");
        if response.hovered() || eye.hovered() {
            self.outliner_hover = set_ids.to_vec();
        }
        let mut action = None;
        if eye.clicked() {
            action = Some(PieceAction::SetVisible(set_ids.to_vec(), !any_visible));
        } else if response.double_clicked() {
            action = Some(PieceAction::FrameSets(set_ids.to_vec()));
        } else if response.clicked() {
            let extend = ui.input(|i| i.modifiers.shift || i.modifiers.command);
            action = Some(PieceAction::Select(set_ids.to_vec(), extend));
        }
        let response = response.on_hover_text("Click: select this object's pieces · Shift/Ctrl+click: add · double-click: frame · right-click: more");
        response.context_menu(|ui| {
            if let Some(a) = Self::piece_menu(ui, set_ids) {
                action = Some(a);
            }
        });
        action
    }

    /// One texture set: the paint target is outlined and marked with a
    /// brush, selected rows are tinted with an accent bar, hidden ones dim.
    fn set_row(&mut self, ui: &mut Ui, i: usize) -> Option<PieceAction> {
        let active = i == self.active_set;
        let picked = self.selection.contains(&i);
        let visible = self.texture_sets[i].visible;
        let width = ui.available_width();
        let (rect, response) = ui.allocate_exact_size(egui::vec2(width, 28.0), Sense::click());
        if self.ui.scroll_to_set == Some(i) {
            response.scroll_to_me(Some(egui::Align::Center));
            self.ui.scroll_to_set = None;
        }
        let radius = CornerRadius::same(theme::RADIUS_ROW);
        let painter = ui.painter();
        if picked {
            painter.rect_filled(rect, radius, theme::ACCENT.gamma_multiply(0.16));
            let bar = egui::Rect::from_min_size(rect.min + egui::vec2(2.0, 5.0), egui::vec2(3.0, rect.height() - 10.0));
            painter.rect_filled(bar, CornerRadius::same(2), theme::ACCENT);
        } else if response.hovered() {
            painter.rect_filled(rect, radius, Color32::from_white_alpha(10));
        }
        if active {
            painter.rect_stroke(rect, radius, Stroke::new(1.5_f32, theme::ACCENT), StrokeKind::Inside);
        }
        let fade = |c: Color32| if visible { c } else { c.gamma_multiply(0.35) };
        painter.circle_filled(egui::pos2(rect.left() + 22.0, rect.center().y), 6.0, fade(self.material_colour(i)));
        let name_colour = if !visible {
            theme::TEXT_DIM.gamma_multiply(0.6)
        } else if picked || active {
            theme::TEXT_STRONG
        } else {
            theme::TEXT
        };
        painter.text(
            egui::pos2(rect.left() + 36.0, rect.center().y),
            egui::Align2::LEFT_CENTER,
            &self.texture_sets[i].material,
            egui::FontId::proportional(theme::BODY_SIZE),
            name_colour,
        );
        let count = self.texture_sets[i].layers.layers.len();
        let res = self.texture_sets[i].layers.resolution;
        // The size only when it isn't the default 2K.
        let size = if res == super::PAINT_RESOLUTION { String::new() } else { format!("{}K · ", res / 1024) };
        let count_right = rect.right() - 34.0;
        painter.text(
            egui::pos2(count_right, rect.center().y),
            egui::Align2::RIGHT_CENTER,
            format!("{size}{count} layer{}", if count == 1 { "" } else { "s" }),
            egui::FontId::proportional(11.5),
            fade(theme::TEXT_DIM),
        );
        if active {
            painter.text(
                egui::pos2(count_right - 52.0, rect.center().y),
                egui::Align2::RIGHT_CENTER,
                icons::PAINT,
                egui::FontId::proportional(12.0),
                theme::ACCENT,
            );
        }

        // The eye toggles the whole selection when this row is part of it.
        let group: Vec<usize> = if picked && self.selection.len() > 1 { self.selection.clone() } else { vec![i] };
        let tip = if group.len() > 1 { "Show/hide the selected pieces" } else { "Show/hide this piece" };
        let eye = Self::eye_button(ui, egui::pos2(rect.right() - 16.0, rect.center().y), visible, tip);
        if response.hovered() || eye.hovered() {
            self.outliner_hover = vec![i];
        }
        let mut action = None;
        if eye.clicked() {
            action = Some(PieceAction::SetVisible(group, !visible));
        } else if response.double_clicked() {
            action = Some(PieceAction::FrameSets(vec![i]));
        } else if response.clicked() {
            let extend = ui.input(|i| i.modifiers.shift || i.modifiers.command);
            action = Some(PieceAction::Select(vec![i], extend));
        }
        let response = response.on_hover_text(if active {
            "Painting this piece · Shift/Ctrl+click: add to selection · double-click: frame · right-click: more"
        } else {
            "Click: select and paint this · Shift/Ctrl+click: add to selection · double-click: frame · right-click: more"
        });
        let targets = if picked { self.selection.clone() } else { vec![i] };
        response.context_menu(|ui| {
            if let Some(a) = Self::piece_menu(ui, &targets) {
                action = Some(a);
            }
        });
        action
    }

    fn piece_menu(ui: &mut Ui, sets: &[usize]) -> Option<PieceAction> {
        let mut action = None;
        let mut item = |ui: &mut Ui, text: &str, key: &str, a: PieceAction| {
            if ui.add(egui::Button::new(text).shortcut_text(key)).clicked() {
                action = Some(a);
                ui.close_menu();
            }
        };
        item(ui, "Isolate", "/", PieceAction::IsolateSets(sets.to_vec()));
        item(ui, "Frame", "F", PieceAction::FrameSets(sets.to_vec()));
        ui.separator();
        item(ui, "Hide", "H", PieceAction::SetVisible(sets.to_vec(), false));
        item(ui, "Show only these", "Shift H", PieceAction::ShowOnly(sets.to_vec()));
        item(ui, "Show all", "Alt H", PieceAction::UnhideAll);
        ui.separator();
        let delete = if sets.len() == 1 { "Delete piece".to_string() } else { format!("Delete {} pieces", sets.len()) };
        item(ui, &delete, "", PieceAction::Delete(sets.to_vec()));
        action
    }

    fn run_piece_action(&mut self, action: PieceAction) {
        match action {
            PieceAction::Select(sets, extend) => {
                if extend {
                    for set in sets {
                        self.toggle_selected(set, true);
                    }
                } else {
                    self.selection = sets.clone();
                    if !sets.contains(&self.active_set) {
                        self.set_active_set(sets[0]);
                    }
                }
            }
            PieceAction::SetVisible(sets, on) => {
                for set in sets {
                    self.texture_sets[set].visible = on;
                }
                self.pt_scene_stale = true;
            }
            PieceAction::ShowOnly(sets) => {
                self.selection = sets;
                self.hide_unselected();
            }
            PieceAction::IsolateSets(sets) => {
                if self.isolate_backup.is_some() {
                    self.toggle_isolate();
                }
                self.selection = sets;
                self.toggle_isolate();
            }
            PieceAction::FrameSets(sets) => self.frame_sets(&sets),
            PieceAction::Isolate => self.toggle_isolate(),
            PieceAction::Hide => self.hide_selected(),
            PieceAction::UnhideAll => self.unhide_all(),
            PieceAction::Frame => self.frame_selection(),
            PieceAction::Delete(sets) => self.delete_pieces(&sets),
        }
    }

    fn layers_section(&mut self, ui: &mut Ui, max_list_height: f32) {
        let mut run = None;
        ui.horizontal(|ui| {
            ui.label(theme::section_header("Layers"));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.spacing_mut().item_spacing.x = 4.0;
                let layers = &self.texture_sets[self.active_set].layers;
                if ui
                    .add_enabled(layers.layers.len() > 1, egui::Button::new(icons::DELETE).frame(false))
                    .on_hover_text("Delete layer  Del")
                    .on_disabled_hover_text("The last layer can't be deleted")
                    .clicked()
                {
                    run = Some(Cmd::DeleteLayer);
                }
                if ui.add(egui::Button::new(icons::CLONE).frame(false)).on_hover_text("Duplicate layer  Ctrl J").clicked() {
                    run = Some(Cmd::Duplicate);
                }
                ui.menu_button(format!("Mask {}", icons::MOVE_DOWN), |ui| {
                    ui.label(RichText::new("Shader mask").font(theme::semibold(theme::BODY_SIZE))).on_hover_text(
                        "Greyscale masks for avatar toon shaders (Poiyomi, lilToon…): white shows \
                         the effect, black hides it. Each exports as its own PNG.",
                    );
                    for kind in ShaderMask::ALL {
                        let [r, g, b] = kind.tint();
                        if ui
                            .add(egui::Button::new(RichText::new(format!("● {}", kind.label())).color(Color32::from_rgb(r, g, b))).frame(false))
                            .on_hover_text(kind.description())
                            .clicked()
                        {
                            run = Some(Cmd::NewShaderMask(kind));
                            ui.close_menu();
                        }
                    }
                })
                .response
                .on_hover_text("New shader-mask layer");
                if ui
                    .add(egui::Button::new(format!("{} Matcap", icons::ADD)).frame(false))
                    .on_hover_text("New matcap layer: pick a matcap, paint white where it shows")
                    .clicked()
                {
                    run = Some(Cmd::NewShaderMask(ShaderMask::Matcap));
                }
                if ui
                    .add(egui::Button::new(format!("{} Fill", icons::ADD)).frame(false))
                    .on_hover_text("New layer filled with an image, ready to wrap onto the model")
                    .clicked()
                {
                    run = Some(Cmd::NewFillImage);
                }
                if ui
                    .add(egui::Button::new(format!("{} Paint", icons::ADD)).frame(false))
                    .on_hover_text("New paint layer  Ctrl Shift N")
                    .clicked()
                {
                    run = Some(Cmd::NewPaintLayer);
                }
            });
        });

        let mut select = None;
        let mut mask_choice = None;
        let mut toggle_visible = None;
        let mut reorder = None;
        let mut commit_rename = None;
        let renaming = self.ui.renaming.clone();
        let rename_text_fresh = std::mem::take(&mut self.ui.rename_fresh);
        let mut rename_text = renaming.as_ref().map(|(_, t)| t.clone()).unwrap_or_default();
        let mut rename_done = false;
        egui::ScrollArea::vertical().id_salt("layer_list").max_height(max_list_height).show(ui, |ui| {
            let layers = &self.texture_sets[self.active_set].layers;
            let count = layers.layers.len();
            // Top of the stack first: a row composites over the rows below it.
            for i in (0..count).rev() {
                let layer = &layers.layers[i];
                let selected = i == layers.active;
                let width = ui.available_width();
                let (rect, response) = ui.allocate_exact_size(egui::vec2(width, ROW_HEIGHT), Sense::click());
                let painter = ui.painter();
                let radius = CornerRadius::same(theme::RADIUS_ROW);
                let dragging_over = response.dnd_hover_payload::<usize>().is_some_and(|from| *from != i);
                if selected {
                    painter.rect(rect, radius, theme::BG_SELECTED, Stroke::new(1.5_f32, theme::ACCENT), StrokeKind::Inside);
                } else if response.hovered() {
                    painter.rect_filled(rect, radius, Color32::from_white_alpha(8));
                }
                if dragging_over {
                    painter.hline(rect.x_range(), rect.top(), Stroke::new(2.0_f32, theme::ACCENT));
                }
                if let Some(from) = response.dnd_release_payload::<usize>() {
                    reorder = Some((*from, i));
                }

                let mut child = ui.new_child(
                    egui::UiBuilder::new()
                        .max_rect(rect.shrink2(egui::vec2(6.0, 5.0)))
                        .layout(egui::Layout::left_to_right(egui::Align::Center)),
                );
                child.spacing_mut().item_spacing.x = 6.0;
                child
                    .dnd_drag_source(egui::Id::new(("layer_drag", i)), i, |ui| {
                        ui.label(RichText::new(icons::GRIP).color(theme::TEXT_DIM));
                    })
                    .response
                    .on_hover_text("Drag to reorder (also Layer ▸ Move up / down)");
                let eye = if layer.visible { icons::VISIBLE } else { icons::HIDDEN };
                if child
                    .add(egui::Button::new(RichText::new(eye).color(theme::TEXT_MUTED)).frame(false))
                    .on_hover_text("Toggle visibility")
                    .clicked()
                {
                    toggle_visible = Some(i);
                }
                if layer.clip && i > 0 {
                    child
                        .label(RichText::new("↳").color(theme::ACCENT))
                        .on_hover_text("Clipped: shows only where the layer below has paint");
                }
                widgets::swatch(&mut child, 34.0, layer_thumb_colour(layer));
                if layer.mask.is_some() {
                    let thumb = mask_thumb(&mut child, 34.0, layer.mask_enabled).on_hover_text(format!(
                        "Layer mask ({}){} — Shift-click to turn it {}",
                        layer.mask_kind.map(|k| k.label()).unwrap_or("painted"),
                        if layer.mask_enabled { "" } else { ", off" },
                        if layer.mask_enabled { "off" } else { "on" },
                    ));
                    if thumb.clicked() && thumb.ctx.input(|inp| inp.modifiers.shift) {
                        select = Some(i);
                        run = Some(Cmd::ToggleMaskEnabled);
                    }
                }
                child.vertical(|ui| {
                    ui.spacing_mut().item_spacing.y = 3.0;
                    match &renaming {
                        Some((r, _)) if *r == i => {
                            let edit = ui.add(egui::TextEdit::singleline(&mut rename_text).desired_width(140.0));
                            // Focus once, when renaming starts: asking every
                            // frame would take it back from Enter and Esc.
                            if !edit.has_focus() && !edit.lost_focus() && rename_text_fresh {
                                edit.request_focus();
                            }
                            if edit.lost_focus() {
                                rename_done = true;
                                if !ui.input(|inp| inp.key_pressed(egui::Key::Escape)) {
                                    commit_rename = Some((i, rename_text.clone()));
                                }
                            }
                        }
                        _ => {
                            let name = ui.add(
                                egui::Label::new(RichText::new(&layer.name).font(theme::semibold(theme::BODY_SIZE)).color(theme::TEXT_STRONG))
                                    .truncate()
                                    .sense(Sense::click()),
                            );
                            if name.double_clicked() {
                                self.ui.rename_request = Some(i);
                            } else if name.clicked() {
                                select = Some(i);
                            }
                        }
                    }
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 3.0;
                        layer_badges(ui, layer);
                    });
                });
                // Blend and opacity, top right, mono.
                let blend = match layer.purpose {
                    Some(_) => "Mask",
                    None => layer.blend.short_label(),
                };
                let text = format!("{blend}\n{:.0}", layer.opacity * 100.0);
                ui.painter().text(
                    egui::pos2(rect.right() - 8.0, rect.center().y),
                    egui::Align2::RIGHT_CENTER,
                    text,
                    egui::FontId::monospace(10.5),
                    theme::TEXT_DIM,
                );
                if response.clicked() {
                    select = Some(i);
                }
                if response.double_clicked() {
                    self.ui.rename_request = Some(i);
                }
                // Layer commands act on the selected layer, so pick this
                // row first, then run.
                let has_mask = layer.mask.is_some();
                response.context_menu(|ui| {
                    let label = if has_mask { "Replace mask from" } else { "Add mask from" };
                    ui.menu_button(label, |ui| {
                        if let Some(choice) = mask_source_menu(ui, layers, i) {
                            select = Some(i);
                            mask_choice = Some(choice);
                        }
                    });
                    let mut item = |ui: &mut Ui, enabled: bool, label: &str, cmd: Cmd| {
                        if ui.add_enabled(enabled, egui::Button::new(label)).clicked() {
                            select = Some(i);
                            run = Some(cmd);
                            ui.close_menu();
                        }
                    };
                    item(ui, true, "Export layer…", Cmd::ExportLayer);
                    item(ui, has_mask, "Export mask…", Cmd::ExportLayerMask);
                    item(ui, true, "Load mask from image…", Cmd::LoadMaskImage);
                    ui.separator();
                    item(ui, true, "Duplicate", Cmd::Duplicate);
                    item(ui, layers.merge_down_blocker(i).is_none(), "Merge down", Cmd::MergeDown);
                    item(ui, i > 0, if layer.clip { "Unclip" } else { "Clip to layer below" }, Cmd::ToggleClip);
                    item(ui, count > 1, "Delete", Cmd::DeleteLayer);
                });
            }
        });

        // Shader masks in this set, one click to jump to each.
        let kinds = self.texture_sets[self.active_set].layers.shader_mask_kinds();
        if !kinds.is_empty() {
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new("Shader masks:").color(theme::TEXT_MUTED));
                for kind in kinds {
                    let [r, g, b] = kind.tint();
                    let link = ui.add(egui::Link::new(RichText::new(kind.label().trim_end_matches(" mask")).color(Color32::from_rgb(r, g, b)).underline()));
                    if link.on_hover_text(format!("Go to {}", kind.label())).clicked() {
                        let layers = &mut self.texture_sets[self.active_set].layers;
                        if let Some(i) = layers.layers.iter().rposition(|l| l.purpose == Some(kind)) {
                            select = Some(i);
                        }
                    }
                }
            });
        }

        let layers = &mut self.texture_sets[self.active_set].layers;
        if let Some(i) = select {
            layers.active = i;
        }
        if let Some(i) = toggle_visible {
            layers.layers[i].visible = !layers.layers[i].visible;
            layers.mark_dirty();
        }
        if let Some((from, to)) = reorder {
            layers.move_layer(from, to);
        }
        if let Some((i, name)) = commit_rename {
            let name = name.trim();
            if !name.is_empty() && i < layers.layers.len() {
                layers.layers[i].name = name.to_string();
            }
        }
        if rename_done {
            self.ui.renaming = None;
        } else if let Some((i, _)) = renaming {
            self.ui.renaming = Some((i, rename_text));
        }
        if let Some(i) = self.ui.rename_request.take() {
            let layers = &self.texture_sets[self.active_set].layers;
            if i < layers.layers.len() {
                self.ui.renaming = Some((i, layers.layers[i].name.clone()));
                self.ui.rename_fresh = true;
            }
        }
        if let Some(cmd) = run {
            let ctx = ui.ctx().clone();
            self.run_command(cmd, &ctx);
        }
        if let Some(choice) = mask_choice {
            self.apply_mask_choice(choice);
        }
    }

    /// Give the selected layer the mask the picker chose.
    fn apply_mask_choice(&mut self, choice: MaskChoice) {
        match choice {
            MaskChoice::Image => self.load_mask_from_image(),
            MaskChoice::Source(source) => {
                let layers = &mut self.texture_sets[self.active_set].layers;
                let active = layers.active;
                let from = match source {
                    MaskSource::LayerMask(i) => format!("the mask of '{}'", layers.layers[i].name),
                    MaskSource::LayerCoverage(i) => format!("where '{}' is painted", layers.layers[i].name),
                    MaskSource::ShaderMask(kind) => kind.label().to_string(),
                };
                if layers.mask_from(active, source) {
                    self.ui.status = format!("Mask of '{}' made from {from}", layers.active_layer().name);
                    self.ui.props_tab = PropsTab::Mask;
                } else {
                    self.ui.status = format!("Nothing to make a mask from in {from}");
                }
            }
        }
    }

    fn properties_tabs(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 18.0;
            for (tab, label) in [
                (PropsTab::Layer, "Layer"),
                (PropsTab::Mask, "Mask"),
                (PropsTab::Projection, "Projection"),
                (PropsTab::Material, "Material"),
            ] {
                let on = self.ui.props_tab == tab;
                let response = ui.add(
                    egui::Label::new(RichText::new(label).color(if on { theme::TEXT_STRONG } else { theme::TEXT_MUTED }))
                        .sense(Sense::click()),
                );
                if on {
                    let r = response.rect;
                    ui.painter().hline(r.x_range().expand(6.0), r.bottom() + 6.0, Stroke::new(2.0_f32, theme::ACCENT));
                }
                if response.clicked() {
                    self.ui.props_tab = tab;
                }
            }
        });
        ui.add_space(6.0);
        widgets::divider(ui);
        // Only the tab area scrolls.
        egui::ScrollArea::vertical().id_salt("props").auto_shrink([false, false]).show(ui, |ui| {
            match self.ui.props_tab {
                PropsTab::Layer => self.layer_tab(ui),
                PropsTab::Mask => self.mask_tab(ui),
                PropsTab::Projection => self.texture_projection_ui(ui),
                PropsTab::Material => self.material_tab(ui),
            }
        });
    }

    fn layer_tab(&mut self, ui: &mut Ui) {
        let layers = &mut self.texture_sets[self.active_set].layers;
        let active = layers.active;
        let layer = &mut layers.layers[active];
        let mut changed = false;

        ui.label(RichText::new("Name").color(theme::TEXT_MUTED));
        ui.add(egui::TextEdit::singleline(&mut layer.name).desired_width(f32::INFINITY));
        ui.add_space(6.0);

        ui.columns(2, |cols| {
            cols[0].label(RichText::new("Blend").color(theme::TEXT_MUTED));
            egui::ComboBox::from_id_salt("layer_blend")
                .width(cols[0].available_width())
                .selected_text(layer.blend.label())
                .show_ui(&mut cols[0], |ui| {
                    for mode in BlendMode::ALL {
                        if mode.starts_group() {
                            ui.separator();
                        }
                        changed |= ui.selectable_value(&mut layer.blend, mode, mode.label()).changed();
                    }
                });
            cols[1].label(RichText::new("Opacity").color(theme::TEXT_MUTED));
            cols[1].horizontal(|ui| {
                let width = ui.available_width() - 44.0;
                changed |= widgets::slider(ui, &mut layer.opacity, 0.0..=1.0, false, width).changed();
                changed |= ui
                    .add(egui::DragValue::new(&mut layer.opacity).range(0.0..=1.0).speed(0.005).fixed_decimals(2))
                    .changed();
            });
        });
        if active > 0 {
            changed |= ui
                .checkbox(&mut layers.layers[active].clip, "Clip to layer below")
                .on_hover_text("Show only where the layer below has paint (Ctrl Alt G)")
                .changed();
        }
        if changed {
            layers.mark_dirty();
        }
        ui.add_space(8.0);

        match layers.layers[active].purpose {
            Some(kind) => {
                if kind == ShaderMask::Matcap
                    && let Some(error) = matcap_editor(ui, &mut layers.layers[active], &self.gpu)
                {
                    self.ui.status = error;
                }
                ui.label(RichText::new("Mask for").color(theme::TEXT_MUTED));
                let mut chosen = kind;
                egui::ComboBox::from_id_salt("shader_mask_kind")
                    .width(ui.available_width())
                    .selected_text(kind.label())
                    .show_ui(ui, |ui| {
                        for k in ShaderMask::ALL {
                            ui.selectable_value(&mut chosen, k, k.label()).on_hover_text(k.description());
                        }
                    });
                if chosen != kind {
                    let layer = &mut layers.layers[active];
                    if layer.name == kind.label() {
                        layer.name = chosen.label().to_string();
                    }
                    layer.purpose = Some(chosen);
                    if chosen == ShaderMask::Matcap && layer.matcap.is_none() {
                        layer.matcap = Some(Default::default());
                    }
                    layers.mark_dirty();
                }
                ui.columns(2, |cols| {
                    if widgets::wide(&mut cols[0], egui::Button::new("Fill white")).on_hover_text("Effect everywhere").clicked() {
                        layers.fill_active(Some([1.0; 3]), None, None);
                    }
                    if widgets::wide(&mut cols[1], egui::Button::new("Fill black")).on_hover_text("Effect nowhere").clicked() {
                        layers.fill_active(Some([0.0; 3]), None, None);
                    }
                });
                ui.label(
                    RichText::new(format!(
                        "{}. Paint white where the effect shows, black where it doesn't; it exports as its own greyscale PNG.",
                        kind.description()
                    ))
                    .color(theme::TEXT_MUTED),
                );
            }
            None => {
                if ui
                    .add(egui::Button::new("Fill layer from brush values"))
                    .clicked()
                {
                    self.fill_from_brush_values();
                }
                ui.label(
                    RichText::new("Floods this layer with the channels switched on in the Brush panel.")
                        .color(theme::TEXT_MUTED),
                );
            }
        }
    }

    fn mask_tab(&mut self, ui: &mut Ui) {
        let layers = &mut self.texture_sets[self.active_set].layers;
        let active = layers.active;
        let has_mask = layers.layers[active].mask.is_some();
        if has_mask {
            let enabled = layers.layers[active].mask_enabled;
            ui.horizontal(|ui| {
                mask_thumb(ui, 64.0, enabled);
                ui.vertical(|ui| {
                    ui.label(RichText::new("Layer mask").font(theme::semibold(theme::BODY_SIZE)));
                    if widgets::wide(ui, widgets::primary_button("Paint the mask"))
                        .on_hover_text("Switch the brush to the mask")
                        .clicked()
                    {
                        if !self.tool.uses_brush() {
                            self.tool = Tool::Paint;
                        }
                        self.brush.target = crate::paint::PaintTarget::Mask;
                    }
                });
            });
            ui.add_space(4.0);
            let layers = &mut self.texture_sets[self.active_set].layers;
            ui.columns(4, |cols| {
                if widgets::wide(&mut cols[0], egui::Button::new("Fill")).on_hover_text("Reveal everywhere").clicked() {
                    layers.fill_mask(active, 1.0);
                }
                if widgets::wide(&mut cols[1], egui::Button::new("Clear")).on_hover_text("Hide everywhere").clicked() {
                    layers.fill_mask(active, 0.0);
                }
                if widgets::wide(&mut cols[2], egui::Button::new("Invert")).clicked() {
                    layers.invert_mask(active);
                }
                if widgets::wide(&mut cols[3], widgets::danger_button("Remove")).clicked() {
                    layers.remove_mask(active);
                }
            });
            let layers = &mut self.texture_sets[self.active_set].layers;
            if layers.layers.get(active).is_some_and(|l| l.mask.is_some()) {
                ui.horizontal(|ui| {
                    if ui
                        .checkbox(&mut layers.layers[active].mask_enabled, "Enabled")
                        .on_hover_text("Off shows the layer without its mask, keeping the mask (Shift-click its icon)")
                        .changed()
                    {
                        layers.mark_dirty();
                    }
                    if ui
                        .button("Apply mask")
                        .on_hover_text("Bake the mask into the layer and remove it — the layer looks the same")
                        .clicked()
                    {
                        layers.apply_mask(active);
                    }
                });
            }
        } else {
            ui.label(RichText::new("This layer has no mask.").color(theme::TEXT_MUTED));
            let big = |text: &str, sub: &str| {
                egui::Button::new(
                    RichText::new(format!("{text}\n")).font(theme::semibold(theme::BODY_SIZE)),
                )
                .shortcut_text(sub.to_string())
            };
            if widgets::wide(ui, big("Add black mask", "hidden, paint it in")).clicked() {
                layers.add_mask(active, 0.0);
            }
            if widgets::wide(ui, big("Add white mask", "visible, paint it out")).clicked() {
                layers.add_mask(active, 1.0);
            }
        }
        let layers = &self.texture_sets[self.active_set].layers;
        let label = if layers.active_layer().mask.is_some() { "Replace mask from…" } else { "Add mask from…" };
        let mut choice = None;
        ui.menu_button(format!("{label} {}", icons::MOVE_DOWN), |ui| {
            choice = mask_source_menu(ui, layers, layers.active);
        })
        .response
        .on_hover_text("An image file, another layer's mask or painted area, or a shader mask");
        if let Some(choice) = choice {
            self.apply_mask_choice(choice);
        }
        ui.checkbox(&mut self.ui.mask_overlay, "Show mask on model").on_hover_text(
            "While a mask layer is selected, tint the model where the mask is white and dim it where it is black",
        );

        widgets::divider(ui);
        let baked = self.texture_sets[self.active_set].baked_curvature.is_some();
        ui.horizontal(|ui| {
            ui.label(theme::section_header("Smart mask"));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if baked {
                    ui.label(RichText::new("Mesh maps ready").color(theme::OK_FG));
                } else {
                    ui.label(RichText::new("Bakes mesh maps on Generate").color(theme::TEXT_DIM));
                }
            });
        });
        let kinds = crate::layers::MaskKind::ALL.map(|k| (k, k.label()));
        widgets::segmented(ui, &mut self.ui.mask_kind, &kinds);
        SliderRow::new("Amount", &mut self.ui.mask_amount, 0.0..=1.0).show(ui);
        SliderRow::new("Contrast", &mut self.ui.mask_contrast, 0.1..=4.0).show(ui);
        ui.columns(2, |cols| {
            if widgets::wide(&mut cols[0], egui::Button::new("Generate")).clicked() {
                self.generate_smart_mask();
            }
            let layers = &mut self.texture_sets[self.active_set].layers;
            let has_mask = layers.layers[layers.active].mask.is_some();
            if cols[1].add_enabled_ui(has_mask, |ui| widgets::wide(ui, egui::Button::new("Remove"))).inner.clicked() {
                let active = layers.active;
                layers.layers[active].mask = None;
                layers.layers[active].mask_kind = None;
                layers.mark_dirty();
            }
        });
    }

    fn material_tab(&mut self, ui: &mut Ui) {
        let set = &self.texture_sets[self.active_set];
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing.x = 0.0;
            ui.label(RichText::new(format!("{} · ", set.object)).color(theme::TEXT_MUTED));
            ui.label(RichText::new(&set.material).font(theme::semibold(theme::BODY_SIZE)));
            ui.label(RichText::new(" — base material under the layers").color(theme::TEXT_MUTED));
        });
        ui.add_space(4.0);

        let mut load_slot = None;
        let mut clear_slot = None;
        for slot in MapSlot::ALL {
            let state = self.active_material_map_info(slot).unwrap_or_else(|| "—".into());
            let colour = self.map_slot_colour(slot);
            ui.horizontal(|ui| {
                widgets::swatch(ui, 22.0, colour);
                widgets::fixed(ui, 90.0, |ui| ui.label(slot.label()));
                ui.label(RichText::new(state).monospace().color(theme::TEXT_MUTED));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.add(egui::Button::new(icons::REMOVE).frame(false)).on_hover_text("Remove map").clicked() {
                        clear_slot = Some(slot);
                    }
                    if ui.add(egui::Button::new(icons::OPEN).frame(false)).on_hover_text("Load image file").clicked() {
                        load_slot = Some(slot);
                    }
                });
            });
        }
        if let Some(slot) = load_slot {
            self.load_material_map(slot);
        }
        if let Some(slot) = clear_slot {
            self.clear_material_map(slot);
        }

        widgets::divider(ui);
        widgets::section(ui, "Surface", None);
        self.surface_params_ui(ui);

        widgets::divider(ui);
        widgets::section(ui, "Matcaps", None);
        self.matcaps_ui(ui);

        widgets::divider(ui);
        widgets::section(ui, "Bake & UVs", None);
        ui.columns(2, |cols| {
            if widgets::wide(&mut cols[0], egui::Button::new(format!("{} Bake AO", icons::BAKE))).clicked() {
                self.bake_ao();
            }
            if widgets::wide(&mut cols[1], egui::Button::new(format!("{} Bake mesh maps", icons::BAKE)))
                .on_hover_text("Position, normal and curvature — needed for smart masks and projections")
                .clicked()
            {
                self.bake_mesh_maps();
            }
        });
        SliderRow::new("AO strength", &mut self.ui.ao_strength, 0.0..=1.0).show(ui);
        let unwrap = format!("Auto-unwrap UVs ({})…", self.texture_sets[self.active_set].material);
        if widgets::wide(ui, widgets::warn_button(unwrap)).clicked() {
            self.ui.confirm_unwrap = true;
        }
        ui.label(RichText::new("Resets this set's layers. Other sets keep their paint.").color(theme::WARN_FG).small());

        widgets::divider(ui);
        widgets::section(ui, "Save to library", None);
        ui.horizontal(|ui| {
            let width = ui.available_width() - 60.0;
            let field = ui.add(
                egui::TextEdit::singleline(&mut self.ui.library_save_name)
                    .hint_text("Material name")
                    .desired_width(width),
            );
            if self.ui.focus_save_name {
                field.request_focus();
                self.ui.focus_save_name = false;
            }
            if ui.button("Save").clicked() {
                self.save_material_to_library();
            }
        });
        ui.label(
            RichText::new(crate::library::library_dir().display().to_string())
                .small()
                .color(theme::TEXT_DIM),
        );
    }

    /// The colour a material shows as in the texture-set tree: its base
    /// colour map's average, else its base colour factor.
    fn material_colour(&self, set: usize) -> Color32 {
        let ts = &self.texture_sets[set];
        let model = self.models.iter().find(|m| m.name == ts.object && m.material_name == ts.material);
        match model {
            Some(m) => match m.material.base_color_image.as_ref() {
                Some(img) => sample_average(img),
                None => {
                    let f = m.material.base_color;
                    widgets::linear_to_color32([f[0], f[1], f[2]])
                }
            },
            None => theme::TEXT_DIM,
        }
    }

    fn map_slot_colour(&self, slot: MapSlot) -> Color32 {
        let ts = &self.texture_sets[self.active_set];
        let Some(model) = self.models.iter().find(|m| m.name == ts.object && m.material_name == ts.material) else {
            return theme::BG_FIELD;
        };
        let m = &model.material;
        let image = match slot {
            MapSlot::BaseColor => m.base_color_image.as_ref(),
            MapSlot::Roughness => m.roughness_image.as_ref(),
            MapSlot::Metallic => m.metallic_image.as_ref(),
            MapSlot::Normal => m.normal_image.as_ref(),
            MapSlot::Height => m.height_image.as_ref(),
            MapSlot::Ao => m.ao_image.as_ref(),
            MapSlot::Emission => m.emission_image.as_ref(),
            MapSlot::EmissionMask => m.emission_mask_image.as_ref(),
            MapSlot::Opacity => None,
        };
        match image {
            Some(img) => sample_average(img),
            None if slot == MapSlot::Ao && ts.baked_ao.is_some() => Color32::from_gray(210),
            None => theme::BG_FIELD,
        }
    }

    /// Projection tab: fill the layer with an image, wrap it, place it.
    pub(super) fn texture_projection_ui(&mut self, ui: &mut Ui) {
        let set = &self.texture_sets[self.active_set];
        if set.layers.active_layer().purpose.is_some() {
            ui.label(RichText::new("Shader-mask layers are painted, not projected.").color(theme::TEXT_MUTED));
            return;
        }
        let frame_size = set.layers.projection_frame().map_or(1.0, |(_, size)| size);
        let mut fill_image = false;
        let mut from_library = false;
        let mut place_tool = false;
        let mut wants_projection = false;
        let mut changed = false;

        ui.columns(2, |cols| {
            fill_image = widgets::wide(&mut cols[0], egui::Button::new("Fill with image…"))
                .on_hover_text("Put a texture file on this layer's colour. Transparent PNGs keep their alpha.")
                .clicked();
            from_library = widgets::wide(&mut cols[1], egui::Button::new("From library…"))
                .on_hover_text("Pick a library material and apply it as a fill layer")
                .clicked();
        });

        let layers = &mut self.texture_sets[self.active_set].layers;
        let active = layers.active;
        let layer = &mut layers.layers[active];
        // Placement only means something for a filled image or material: on
        // a hand-painted layer it would drag the brushwork around.
        if !layer.has_fill {
            ui.label(
                RichText::new("Fill the layer with an image or a library material to wrap it onto the model.")
                    .color(theme::TEXT_MUTED),
            );
        } else {
            ui.add_space(4.0);
            ui.columns(2, |cols| {
                cols[0].label(RichText::new("Projection").color(theme::TEXT_MUTED));
                let before = layer.projection;
                egui::ComboBox::from_id_salt("projection_mode")
                    .width(cols[0].available_width())
                    .selected_text(layer.projection.label())
                    .show_ui(&mut cols[0], |ui| {
                        for mode in Projection::ALL {
                            ui.selectable_value(&mut layer.projection, mode, mode.label())
                                .on_hover_text(mode.description());
                        }
                    })
                    .response
                    .on_hover_text(layer.projection.description());
                if layer.projection != before {
                    // Aim each mode the way it is most often wanted: wraps stand
                    // upright, a planar projection faces the front.
                    match layer.projection {
                        Projection::Spherical | Projection::Cylindrical => layer.projection_axis = Axis::Y,
                        Projection::Planar => layer.projection_axis = Axis::Z,
                        Projection::Uv | Projection::Triplanar => {}
                    }
                    changed = true;
                    wants_projection = layer.projection != Projection::Uv;
                }
                if layer.projection.uses_axis() {
                    cols[1].label(RichText::new("Axis").color(theme::TEXT_MUTED)).on_hover_text(match layer.projection {
                        Projection::Planar => "Direction the image is thrown through the object",
                        _ => "Pole the image wraps around",
                    });
                    let axes = Axis::ALL.map(|a| (a, a.label()));
                    changed |= widgets::segmented(&mut cols[1], &mut layer.projection_axis, &axes);
                }
            });
            if layer.projection == Projection::Triplanar {
                changed |= SliderRow::new("Sharpness", &mut layer.triplanar_sharpness, 1.0..=32.0)
                    .log()
                    .decimals(1)
                    .hint("Higher keeps each side's image to the faces pointing at it")
                    .show(ui)
                    .changed();
            }

            if layer.projection != Projection::Uv {
                ui.add_space(4.0);
                let placing = self.tool == Tool::Place;
                let place = if placing {
                    widgets::primary_button("Placing in viewport   P")
                } else {
                    widgets::accent_outline_button("Place in viewport   P")
                };
                place_tool = widgets::wide(ui, place)
                    .on_hover_text("Drag the projection over the model · Shift+drag rotate · Ctrl+scroll size · Shift+scroll spin")
                    .clicked();

                if widgets::section_action(ui, "Placement", "Reset") {
                    layer.reset_projection_placement();
                    changed = true;
                }
                let field_width = (ui.available_width() - theme::SLIDER_LABEL_WIDTH - 16.0) / 3.0;
                ui.horizontal(|ui| {
                    widgets::fixed(ui, theme::SLIDER_LABEL_WIDTH - 8.0, |ui| ui.label("Position"));
                    for i in 0..3 {
                        changed |= widgets::axis_drag(ui, &mut layer.projection_offset[i], i, (frame_size * 0.002) as f64, "", field_width).changed();
                    }
                });
                ui.horizontal(|ui| {
                    widgets::fixed(ui, theme::SLIDER_LABEL_WIDTH - 8.0, |ui| ui.label("Rotation"));
                    for i in 0..3 {
                        let mut degrees = layer.projection_rotation[i].to_degrees();
                        if widgets::axis_drag(ui, &mut degrees, i, 0.5, "°", field_width).changed() {
                            layer.projection_rotation[i] = super::wrap_angle(degrees.to_radians());
                            changed = true;
                        }
                    }
                });
                changed |= SliderRow::new("Size", &mut layer.projection_scale, 0.02..=4.0)
                    .log()
                    .hint("1 fits the image to the whole object")
                    .show(ui)
                    .changed();
                changed |= ui
                    .checkbox(&mut layer.projection_repeat, "Repeat (off places the image once)")
                    .changed();
            }

            if widgets::section_action(ui, "Tiling", "Reset") {
                layer.uv_scale = [1.0, 1.0];
                layer.uv_offset = [0.0, 0.0];
                layer.uv_rotation = 0.0;
                layer.triplanar_sharpness = 4.0;
                changed = true;
            }
            let half = (ui.available_width() - theme::SLIDER_LABEL_WIDTH - 34.0) / 2.0;
            let mut scale = layer.uv_scale;
            ui.horizontal(|ui| {
                widgets::fixed(ui, theme::SLIDER_LABEL_WIDTH - 8.0, |ui| ui.label("Repeat")).on_hover_text(
                    if layer.projection == Projection::Uv {
                        "How many times the image repeats across the UVs"
                    } else {
                        "How many times the image repeats across the projection"
                    },
                );
                let u = ui.add_sized([half, 22.0], egui::DragValue::new(&mut scale[0]).speed(0.02).range(0.05..=64.0).fixed_decimals(2));
                let link_icon = if self.ui.tiling_linked { icons::LINK } else { icons::UNLINK };
                if ui
                    .add(egui::Button::new(RichText::new(link_icon).color(if self.ui.tiling_linked { theme::ACCENT } else { theme::TEXT_DIM })).frame(false))
                    .on_hover_text("Link U and V")
                    .clicked()
                {
                    self.ui.tiling_linked = !self.ui.tiling_linked;
                }
                let v = ui.add_sized([half, 22.0], egui::DragValue::new(&mut scale[1]).speed(0.02).range(0.05..=64.0).fixed_decimals(2));
                if u.changed() || v.changed() {
                    if self.ui.tiling_linked {
                        // Whichever axis moved drives the other.
                        let driver = if u.changed() { scale[0] } else { scale[1] };
                        scale = [driver, driver];
                    }
                    layer.uv_scale = scale;
                    changed = true;
                }
            });
            ui.horizontal(|ui| {
                widgets::fixed(ui, theme::SLIDER_LABEL_WIDTH - 8.0, |ui| ui.label("Offset"));
                changed |= ui.add_sized([half, 22.0], egui::DragValue::new(&mut layer.uv_offset[0]).speed(0.005).fixed_decimals(2)).changed();
                ui.add_space(22.0);
                changed |= ui.add_sized([half, 22.0], egui::DragValue::new(&mut layer.uv_offset[1]).speed(0.005).fixed_decimals(2)).changed();
            });
            let mut degrees = layer.uv_rotation.to_degrees();
            if SliderRow::new("Spin", &mut degrees, -180.0..=180.0)
                .decimals(0)
                .suffix("°")
                .hint("Spins the image about its middle")
                .show(ui)
                .changed()
            {
                layer.uv_rotation = degrees.to_radians();
                changed = true;
            }
        }

        if changed {
            self.texture_sets[self.active_set].layers.mark_dirty();
        }
        if fill_image {
            self.fill_active_layer_from_image_dialog();
        }
        if from_library {
            self.open_library(false);
        }
        if place_tool {
            if self.tool == Tool::Place {
                self.select_tool(Tool::Orbit);
            } else {
                self.select_tool(Tool::Place);
            }
        }
        // Projections place the image by the baked mesh maps; bake them when
        // one is first picked rather than making the user find the button.
        // Only on the change, so a set that cannot bake (hidden) doesn't
        // retry every frame.
        if wants_projection && !self.texture_sets[self.active_set].layers.has_projection_source() {
            self.bake_mesh_maps();
        }
        let set = &self.texture_sets[self.active_set];
        let layer = set.layers.active_layer();
        if layer.has_fill && layer.projection != Projection::Uv && !set.layers.has_projection_source() {
            widgets::chip(
                ui,
                &format!("{} Projection needs mesh maps: show this material and press Bake mesh maps", icons::WARN),
                widgets::Tone::Warn,
            );
        }
    }
}

/// A matcap layer's matcap: the image (a built-in or a file), its blend and
/// strength. Returns a status message when a file fails to load.
fn matcap_editor(ui: &mut Ui, layer: &mut Layer, gpu: &crate::environment::GpuContext) -> Option<String> {
    use crate::matcap::{MatcapBlend, BUILTINS};
    let mut pick = None;
    let mut load = false;
    let m = layer.matcap.get_or_insert_with(Default::default);
    ui.horizontal(|ui| {
        ui.label(RichText::new("Matcap").color(theme::TEXT_MUTED));
        let current = if m.image.is_some() { m.name.clone() } else { "Pick one…".into() };
        egui::ComboBox::from_id_salt("matcap_image").selected_text(current).width(ui.available_width()).show_ui(ui, |ui| {
            for (name, what, _) in BUILTINS {
                if ui.selectable_label(m.name == name && m.image.is_some(), name).on_hover_text(what).clicked() {
                    pick = Some(name);
                }
            }
            ui.separator();
            if ui.button("Load image…").clicked() {
                load = true;
                ui.close_menu();
            }
        });
    });
    if m.image.is_some() {
        let options: Vec<(MatcapBlend, &str)> = MatcapBlend::ALL.iter().map(|b| (*b, b.label())).collect();
        widgets::segmented(ui, &mut m.blend, &options);
        SliderRow::new("Strength", &mut m.strength, 0.0..=2.0).show(ui);
    } else {
        ui.label(RichText::new("Pick a matcap to see this layer.").small().color(theme::TEXT_DIM));
    }
    ui.add_space(6.0);
    if let Some(name) = pick
        && let Some(image) = crate::matcap::builtin(name)
    {
        let blend = BUILTINS.iter().find(|b| b.0 == name).map(|b| b.2);
        layer.set_matcap(gpu, name.to_string(), image, blend);
    }
    if load && let Some(path) = rfd::FileDialog::new().add_filter("Images", &["png", "jpg", "jpeg", "tga", "bmp", "webp"]).pick_file() {
        match crate::library::load_map(&path) {
            Ok(image) => {
                let name = path.file_stem().unwrap_or_default().to_string_lossy().to_string();
                layer.set_matcap(gpu, name, image, None);
            }
            Err(e) => return Some(format!("Matcap load failed: {e:#}")),
        }
    }
    None
}

/// What a layer is for and what it holds, at a glance: its shader-mask kind,
/// or the channels it has content in (emissive stands out).
fn layer_badges(ui: &mut Ui, layer: &Layer) {
    use crate::layers::{PaintChannel as C, NORMAL_BIT};
    if let Some(kind) = layer.purpose {
        let [r, g, b] = kind.tint();
        let colour = Color32::from_rgb(r, g, b);
        widgets::badge(ui, kind.tag(), colour, colour.linear_multiply(0.18)).on_hover_text(kind.description());
        return;
    }
    for (bit, tag, name) in [
        (C::BaseColor.bit(), "COL", "Base colour"),
        (C::Roughness.bit(), "RGH", "Roughness"),
        (C::Metallic.bit(), "MTL", "Metallic"),
        (C::Height.bit(), "HGT", "Height"),
        (NORMAL_BIT, "NRM", "Normal"),
        (C::Opacity.bit(), "OPA", "Opacity"),
        (C::Emissive.bit(), "EMIT", "Emissive — this layer glows"),
    ] {
        if layer.used & bit != 0 {
            let (fg, bg) = if tag == "EMIT" {
                (Color32::from_rgb(255, 150, 40), Color32::from_rgb(58, 38, 18))
            } else {
                (theme::TEXT_MUTED, theme::BG_BUTTON)
            };
            widgets::badge(ui, tag, fg, bg).on_hover_text(name);
        }
    }
}

/// The swatch a layer shows in the list.
fn layer_thumb_colour(layer: &Layer) -> Color32 {
    match (layer.purpose, layer.swatch) {
        (Some(_), Some(c)) => widgets::linear_to_color32(c),
        (Some(_), None) => Color32::from_gray(20),
        (None, Some(c)) => widgets::linear_to_color32(c),
        (None, None) => theme::BG_FIELD,
    }
}

/// Stand-in for a mask thumbnail: a black tile with a white blob.
/// A mask's icon; crossed out in red while the mask is turned off.
fn mask_thumb(ui: &mut Ui, size: f32, enabled: bool) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(egui::vec2(size, size), Sense::click());
    let painter = ui.painter();
    painter.rect(rect, CornerRadius::same(4), Color32::BLACK, Stroke::new(1.0_f32, theme::LINE_CONTROL), StrokeKind::Inside);
    let centre = rect.center() - rect.size() * 0.12;
    for i in 0..8 {
        let t = i as f32 / 8.0;
        painter.circle_filled(centre, size * 0.3 * (1.0 - t), Color32::from_white_alpha((60.0 + 195.0 * t) as u8));
    }
    if !enabled {
        let red = Stroke::new(2.0_f32, Color32::from_rgb(220, 60, 60));
        let r = rect.shrink(3.0);
        painter.line_segment([r.left_top(), r.right_bottom()], red);
        painter.line_segment([r.right_top(), r.left_bottom()], red);
    }
    response
}

/// Mean colour of an sRGB image from a sparse grid of samples.
fn sample_average(img: &crate::scene::ImageData) -> Color32 {
    if img.width == 0 || img.height == 0 {
        return theme::BG_FIELD;
    }
    let mut sum = [0u32; 3];
    let mut n = 0u32;
    for gy in 0..8 {
        for gx in 0..8 {
            let x = (gx * 2 + 1) * img.width / 16;
            let y = (gy * 2 + 1) * img.height / 16;
            let i = ((y * img.width + x) * 4) as usize;
            if i + 2 < img.rgba.len() {
                for c in 0..3 {
                    sum[c] += img.rgba[i + c] as u32;
                }
                n += 1;
            }
        }
    }
    let n = n.max(1);
    Color32::from_rgb((sum[0] / n) as u8, (sum[1] / n) as u8, (sum[2] / n) as u8)
}
