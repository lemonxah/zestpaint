//! The window chrome: menu bar, per-tool context bar, tool rail and status
//! bar. None of it owns logic; every control calls the same functions the
//! old toolbar and panels did.

use egui::{CornerRadius, RichText, Ui};

use super::commands::{Cmd, Item, menus};
use super::widgets::{self, Tone};
use super::{AppState, PropsTab, Tool};
use crate::icons;
use crate::layers::Projection;
use crate::paint::{FillMode, PaintTarget, StrokeMode};
use crate::theme;

/// The rail's groups, top to bottom.
pub const TOOL_RAIL: [&[Tool]; 4] = [
    &[Tool::Select],
    &[
        Tool::Paint,
        Tool::Erase,
        Tool::Smudge,
        Tool::Clone,
        Tool::Stamp,
    ],
    &[Tool::Picker, Tool::Place],
    &[Tool::Orbit],
];

pub fn tool_name(tool: Tool) -> &'static str {
    match tool {
        Tool::Paint => "Paint",
        Tool::Erase => "Erase",
        Tool::Smudge => "Smudge",
        Tool::Clone => "Clone",
        Tool::Stamp => "Stamp",
        Tool::Picker => "Pick",
        Tool::Place => "Place projection",
        Tool::Select => "Select",
        Tool::Orbit => "Orbit",
    }
}

pub fn tool_key(tool: Tool) -> &'static str {
    match tool {
        Tool::Paint => "B",
        Tool::Erase => "E",
        Tool::Smudge => "G",
        Tool::Clone => "C",
        Tool::Stamp => "T",
        Tool::Picker => "I",
        Tool::Place => "P",
        Tool::Select => "S",
        Tool::Orbit => "R",
    }
}

fn tool_icon(tool: Tool) -> &'static str {
    match tool {
        Tool::Paint => icons::PAINT,
        Tool::Erase => icons::ERASE,
        Tool::Smudge => icons::SMUDGE,
        Tool::Clone => icons::CLONE,
        Tool::Stamp => icons::STAMP,
        Tool::Picker => icons::PICKER,
        Tool::Place => icons::PLACE,
        Tool::Select => icons::POINTER,
        Tool::Orbit => icons::ORBIT,
    }
}

/// The gesture line for the active tool (status bar centre).
pub fn tool_hint(tool: Tool) -> &'static str {
    match tool {
        Tool::Paint => {
            "LMB paint · [ ] size · Alt+drag orbit · Ctrl+scroll size · RMB/MMB pan · Scroll zoom"
        }
        Tool::Erase => {
            "LMB erase · [ ] size · Alt+drag orbit · Ctrl+scroll size · RMB/MMB pan · Scroll zoom"
        }
        Tool::Smudge => "LMB smudge · strength = flow · Alt+drag orbit · Ctrl+scroll size",
        Tool::Stamp => "LMB place stamp · Shift+scroll rotate · Ctrl+scroll size · Alt+drag orbit",
        Tool::Clone => {
            "Ctrl+LMB set source · LMB paint the copy · Ctrl+scroll size · Alt+drag orbit"
        }
        Tool::Picker => "LMB pick base colour (drag to scrub) · returns to your brush on release",
        Tool::Orbit => {
            "LMB orbit · click select (Shift adds) · RMB/MMB pan · Scroll zoom · / isolate · F frame"
        }
        Tool::Select => {
            "LMB select (Shift adds) · A all · Alt+A none · H hide · Shift+H hide others · Alt+H unhide · / isolate · F frame · Home all"
        }
        Tool::Place => "Drag move · Shift+drag rotate · Ctrl+scroll resize · Shift+scroll spin",
    }
}

impl AppState {
    /// Switch tool, keeping the panels that follow it in step.
    pub(super) fn select_tool(&mut self, tool: Tool) {
        self.tool = tool;
        if tool == Tool::Place {
            self.ui.props_tab = PropsTab::Projection;
            self.ui.panels.properties = true;
        }
    }

    pub(super) fn menu_bar(&mut self, ctx: &egui::Context) {
        let mut run = None;
        egui::TopBottomPanel::top("menu_bar")
            .exact_height(34.0)
            .frame(
                egui::Frame::new()
                    .fill(theme::BG_CHROME)
                    .inner_margin(egui::Margin::symmetric(10, 0)),
            )
            .show(ctx, |ui| {
                ui.horizontal_centered(|ui| {
                    // Logo: the app icon and the name.
                    if self.ui.logo.is_none() {
                        self.ui.logo = crate::icons::app_icon(64).map(|(rgba, size)| {
                            let image =
                                egui::ColorImage::from_rgba_unmultiplied([size as usize; 2], &rgba);
                            ui.ctx()
                                .load_texture("app_logo", image, egui::TextureOptions::LINEAR)
                        });
                    }
                    match &self.ui.logo {
                        Some(logo) => {
                            ui.add(egui::Image::new((logo.id(), egui::vec2(20.0, 20.0))));
                        }
                        None => {
                            let (rect, _) = ui
                                .allocate_exact_size(egui::vec2(18.0, 18.0), egui::Sense::hover());
                            ui.painter()
                                .rect_filled(rect, CornerRadius::same(4), theme::ACCENT);
                        }
                    }
                    egui::menu::bar(ui, |ui| {
                        for (title, items) in menus() {
                            ui.menu_button(title, |ui| {
                                ui.set_min_width(240.0);
                                if title == "Bake" {
                                    ui.label(
                                        RichText::new(self.texture_sets[self.active_set].label())
                                            .color(theme::TEXT_DIM)
                                            .small(),
                                    );
                                    ui.separator();
                                }
                                for item in items {
                                    match item {
                                        Item::Cmd(cmd) => {
                                            if self.menu_item(ui, cmd) {
                                                run = Some(cmd);
                                            }
                                        }
                                        Item::Separator => {
                                            ui.separator();
                                        }
                                        Item::Recent => self.recent_menu(ui),
                                        Item::Sub(label, cmds) => {
                                            ui.menu_button(label, |ui| {
                                                for cmd in cmds {
                                                    if self.menu_item(ui, cmd) {
                                                        run = Some(cmd);
                                                    }
                                                }
                                            });
                                        }
                                    }
                                }
                            });
                        }
                    });

                    // Centre: what is open.
                    let centre = format!(
                        "{} — {}",
                        self.ui.model_name, self.texture_sets[self.active_set].material
                    );
                    let full = ui.max_rect();
                    ui.painter().text(
                        full.center(),
                        egui::Align2::CENTER_CENTER,
                        centre,
                        egui::FontId::proportional(12.0),
                        theme::TEXT_MUTED,
                    );

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .add(egui::Button::new(icons::REDO).frame(false))
                            .on_hover_text("Redo  Ctrl Shift Z")
                            .clicked()
                        {
                            run = Some(Cmd::Redo);
                        }
                        if ui
                            .add(egui::Button::new(icons::UNDO).frame(false))
                            .on_hover_text("Undo  Ctrl Z")
                            .clicked()
                        {
                            run = Some(Cmd::Undo);
                        }
                        ui.add_space(6.0);
                        let search = egui::Button::new(
                            RichText::new(format!(
                                "{}  Search commands                Ctrl K",
                                icons::SEARCH
                            ))
                            .color(theme::TEXT_DIM),
                        )
                        .fill(theme::BG_FIELD)
                        .stroke(egui::Stroke::new(1.0_f32, theme::LINE_CONTROL));
                        if ui
                            .add(search)
                            .on_hover_text("Command search  Ctrl K")
                            .clicked()
                        {
                            run = Some(Cmd::CommandSearch);
                        }
                    });
                });
            });
        if let Some(cmd) = run {
            self.run_command(cmd, ctx);
        }
    }

    /// One menu row: label, a tick for toggles, the shortcut on the right.
    /// Planned commands are disabled with a "Coming soon" tooltip.
    fn menu_item(&self, ui: &mut Ui, cmd: Cmd) -> bool {
        let mut label = match cmd.icon() {
            Some(icon) => format!("{icon}  {}", cmd.label(self)),
            None => cmd.label(self),
        };
        if let Cmd::NewShaderMask(kind) = cmd {
            label = format!("● {label}");
            let [r, g, b] = kind.tint();
            let text = RichText::new(label).color(egui::Color32::from_rgb(r, g, b));
            return ui
                .add(egui::Button::new(text).frame(false))
                .on_hover_text(kind.description())
                .clicked();
        }
        if let Some(on) = cmd.checked(self) {
            label = format!("{} {label}", if on { icons::CHECK } else { "  " });
        }
        let mut button = egui::Button::new(label).shortcut_text(cmd.shortcut_text());
        if cmd.planned() {
            button = button.shortcut_text("Soon");
        }
        let response = ui.add_enabled(cmd.enabled(self), button);
        let response = if cmd.planned() {
            response.on_disabled_hover_text("Coming soon")
        } else {
            response
        };
        if response.clicked() {
            ui.close_menu();
            return true;
        }
        false
    }

    /// Foreground and background colours, Photoshop-style: the front square
    /// is what the brush paints, the back one waits its turn. Click either to
    /// change it; the arrow (X) swaps them, the mini pair (D) resets to black
    /// and white.
    fn colour_pair(&mut self, ui: &mut Ui) {
        use egui::{Color32, Rect, Sense, Stroke, StrokeKind, pos2, vec2};
        let (rect, _) = ui.allocate_exact_size(vec2(40.0, 40.0), Sense::hover());
        let side = 25.0;
        let front = Rect::from_min_size(rect.min, vec2(side, side));
        let back = Rect::from_min_size(rect.max - vec2(side, side), vec2(side, side));
        let swap = Rect::from_min_size(pos2(rect.max.x - 13.0, rect.min.y), vec2(13.0, 13.0));
        let reset = Rect::from_min_size(pos2(rect.min.x, rect.max.y - 13.0), vec2(13.0, 13.0));
        let to32 = |c: [f32; 3]| {
            let [r, g, b] = c.map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8);
            Color32::from_rgb(r, g, b)
        };

        let painter = ui.painter();
        for (r, c) in [
            (back, self.brush.background_color),
            (front, self.brush.color),
        ] {
            painter.rect(
                r,
                CornerRadius::same(3),
                to32(c),
                Stroke::new(2.0_f32, theme::BG_BAR),
                StrokeKind::Outside,
            );
            painter.rect_stroke(
                r,
                CornerRadius::same(3),
                Stroke::new(1.0_f32, theme::LINE_CONTROL),
                StrokeKind::Inside,
            );
        }
        // Swap: a corner arrow with a head at each end.
        let stroke = Stroke::new(1.2_f32, theme::TEXT_MUTED);
        let (a, corner, b) = (
            pos2(swap.min.x + 2.0, swap.min.y + 3.0),
            pos2(swap.max.x - 3.0, swap.min.y + 3.0),
            pos2(swap.max.x - 3.0, swap.max.y - 2.0),
        );
        painter.line_segment([a, corner], stroke);
        painter.line_segment([corner, b], stroke);
        for (tip, dir) in [(a, vec2(1.0, 0.0)), (b, vec2(0.0, -1.0))] {
            let side = vec2(-dir.y, dir.x);
            painter.line_segment([tip, tip + dir * 3.0 + side * 3.0], stroke);
            painter.line_segment([tip, tip + dir * 3.0 - side * 3.0], stroke);
        }
        // Reset: a tiny black-over-white pair.
        let mini = 7.0;
        painter.rect_filled(
            Rect::from_min_size(reset.max - vec2(mini, mini), vec2(mini, mini)),
            CornerRadius::same(1),
            Color32::WHITE,
        );
        painter.rect(
            Rect::from_min_size(reset.min, vec2(mini, mini)),
            CornerRadius::same(1),
            Color32::BLACK,
            Stroke::new(1.0_f32, theme::TEXT_MUTED),
            StrokeKind::Inside,
        );

        // Later interactions sit on top, so the front square wins the overlap.
        let id = ui.id().with("colour_pair");
        let back_r = ui
            .interact(back, id.with("back"), Sense::click())
            .on_hover_text("Background colour — click to change, X to swap");
        let front_r = ui
            .interact(front, id.with("front"), Sense::click())
            .on_hover_text("Paint colour — click to change");
        let swap_r = ui
            .interact(swap, id.with("swap"), Sense::click())
            .on_hover_text("Swap colours  X");
        let reset_r = ui
            .interact(reset, id.with("reset"), Sense::click())
            .on_hover_text("Black and white  D");
        if swap_r.clicked() {
            self.swap_colours();
        }
        if reset_r.clicked() {
            self.reset_colours();
        }
        for (response, background) in [(&front_r, false), (&back_r, true)] {
            let popup = ui.make_persistent_id(("rail_colour", background));
            if response.clicked() {
                ui.memory_mut(|m| m.toggle_popup(popup));
            }
            let colour = if background {
                &mut self.brush.background_color
            } else {
                &mut self.brush.color
            };
            egui::popup::popup_above_or_below_widget(
                ui,
                popup,
                response,
                egui::AboveOrBelow::Above,
                egui::PopupCloseBehavior::CloseOnClickOutside,
                |ui| {
                    let mut c = to32(*colour);
                    if egui::color_picker::color_picker_color32(
                        ui,
                        &mut c,
                        egui::color_picker::Alpha::Opaque,
                    ) {
                        *colour = [c.r(), c.g(), c.b()].map(|v| v as f32 / 255.0);
                    }
                },
            );
        }
    }

    /// Swap the foreground and background colours (X).
    pub(super) fn swap_colours(&mut self) {
        std::mem::swap(&mut self.brush.color, &mut self.brush.background_color);
    }

    /// Back to black on white (D).
    pub(super) fn reset_colours(&mut self) {
        (self.brush.color, self.brush.background_color) = crate::paint::DEFAULT_COLORS;
    }

    pub(super) fn context_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("context_bar")
            .min_height(44.0)
            .frame(
                egui::Frame::new()
                    .fill(theme::BG_BAR)
                    .inner_margin(egui::Margin::symmetric(12, 7))
                    .stroke(egui::Stroke::new(1.0_f32, theme::LINE)),
            )
            .show(ctx, |ui| {
                // Wraps to a second line on a narrow window instead of clipping.
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing.x = 8.0;
                    let (rect, _) = ui.allocate_exact_size(egui::vec2(8.0, 8.0), egui::Sense::hover());
                    ui.painter().rect_filled(rect, CornerRadius::same(2), theme::ACCENT);
                    ui.label(RichText::new(tool_name(self.tool)).font(theme::semibold(13.0)).color(theme::TEXT_STRONG));
                    widgets::key_badge(ui, tool_key(self.tool));
                    ui.separator();
                    match self.tool {
                        Tool::Paint | Tool::Erase | Tool::Smudge | Tool::Clone | Tool::Stamp => self.brush_context(ui),
                        Tool::Picker => {
                            let [r, g, b] = self.brush.color.map(|c| (c.clamp(0.0, 1.0) * 255.0) as u8);
                            widgets::swatch(ui, 22.0, egui::Color32::from_rgb(r, g, b));
                            ui.label(
                                RichText::new("Samples base colour from the model · drag to scrub · release to return to the previous brush")
                                    .color(theme::TEXT_MUTED),
                            );
                        }
                        Tool::Select => {
                            for (cmd, key) in [
                                (Cmd::SelectAll, "A"),
                                (Cmd::Hide, "H"),
                                (Cmd::HideOthers, "Shift H"),
                                (Cmd::UnhideAll, "Alt H"),
                                (Cmd::Isolate, "/"),
                                (Cmd::FrameSelected, "F"),
                            ] {
                                let text = format!("{}   {key}", cmd.label(self));
                                if ui.add_enabled(cmd.enabled(self), egui::Button::new(text)).clicked() {
                                    self.run_command(cmd, ui.ctx());
                                }
                            }
                        }
                        Tool::Place => self.place_context(ui),
                        Tool::Orbit => {
                            ui.label(
                                RichText::new("Drag to orbit the camera · RMB/MMB pan · scroll zoom")
                                    .color(theme::TEXT_MUTED),
                            );
                        }
                    }
                });
            });
    }

    fn brush_context(&mut self, ui: &mut Ui) {
        ui.label(RichText::new("Into").color(theme::TEXT_MUTED));
        let targets = [PaintTarget::Material, PaintTarget::Mask].map(|t| (t, t.label()));
        widgets::segmented(ui, &mut self.brush.target, &targets);
        if self.brush.target == PaintTarget::Mask {
            let has_mask = self.texture_sets[self.active_set]
                .layers
                .active_layer()
                .mask
                .is_some();
            if has_mask {
                widgets::segmented(
                    ui,
                    &mut self.brush.mask_value,
                    &[(1.0, "Reveal"), (0.0, "Hide")],
                );
            } else {
                widgets::chip(
                    ui,
                    &format!("{} Layer has no mask", icons::WARN),
                    Tone::Warn,
                );
                if ui.add(widgets::warn_button("Add mask")).clicked() {
                    self.ui.props_tab = PropsTab::Mask;
                    self.ui.panels.properties = true;
                }
            }
        }
        if self.tool == Tool::Clone {
            match self.clone_source {
                Some(_) => widgets::chip(ui, &format!("{} Source set", icons::CHECK), Tone::Ok),
                None => widgets::chip(
                    ui,
                    &format!("{} Ctrl+click the model to set a source", icons::WARN),
                    Tone::Warn,
                ),
            };
            ui.checkbox(&mut self.ui.clone_aligned, "Aligned")
                .on_hover_text(
                    "On: the source follows the brush at a fixed offset across strokes. \
                 Off: every stroke starts again from the source.",
                );
        }
        if self.tool == Tool::Stamp {
            let name = self
                .brush
                .tip_name
                .clone()
                .unwrap_or_else(|| "No stamp image".into());
            ui.label(RichText::new(name).monospace().color(theme::TEXT));
            if ui
                .button("Browse…")
                .on_hover_text("Open the brush library")
                .clicked()
            {
                self.open_library(true);
            }
            if ui.button("Load image…").clicked() {
                self.load_brush_tip();
            }
        } else {
            ui.label(RichText::new("Stroke").color(theme::TEXT_MUTED));
            let modes = StrokeMode::ALL.map(|m| (m, m.label()));
            widgets::segmented_hints(
                ui,
                &mut self.brush.stroke_mode,
                &modes,
                &[
                    "Follows the cursor",
                    "Anchor on press, commit on release",
                    "One decal per click",
                ],
            );
        }
        compact_slider(ui, "Size", &mut self.brush.size, 2.0..=200.0, 0, " px");
        compact_slider(ui, "Hardness", &mut self.brush.hardness, 0.0..=1.0, 2, "");
        let flow = if self.tool == Tool::Smudge {
            "Strength"
        } else {
            "Flow"
        };
        compact_slider(ui, flow, &mut self.brush.flow, 0.01..=1.0, 2, "");
        let mut degrees = self.brush.tip_rotation.to_degrees();
        if compact_slider(ui, "Angle", &mut degrees, -180.0..=180.0, 0, "°").changed() {
            self.brush.tip_rotation = degrees.to_radians();
        }
        ui.checkbox(&mut self.brush.symmetry_x, "X symmetry")
            .on_hover_text("Mirror strokes across the X = 0 plane");
        if self.tool != Tool::Stamp {
            ui.label(RichText::new("Fill").color(theme::TEXT_MUTED));
            egui::ComboBox::from_id_salt("fill_mode")
                .width(110.0)
                .selected_text(self.brush.fill_mode.label())
                .show_ui(ui, |ui| {
                    for (mode, hint) in FillMode::ALL.into_iter().zip([
                        "Brush feathers normally",
                        "Stops on polygon edges",
                        "Fills a whole UV chart",
                        "Paint bucket: one click floods a disconnected mesh",
                    ]) {
                        ui.selectable_value(&mut self.brush.fill_mode, mode, mode.label())
                            .on_hover_text(hint);
                    }
                });
        }
    }

    fn place_context(&mut self, ui: &mut Ui) {
        let layers = &mut self.texture_sets[self.active_set].layers;
        let active = layers.active;
        let layer = &mut layers.layers[active];
        ui.label(RichText::new("Layer").color(theme::TEXT_MUTED));
        ui.label(&layer.name);
        if !layer.has_fill || layer.projection == Projection::Uv {
            widgets::chip(
                ui,
                &format!(
                    "{} Pick a projection other than UV in the Projection tab",
                    icons::WARN
                ),
                Tone::Warn,
            );
        } else {
            ui.label(RichText::new("Projection").color(theme::TEXT_MUTED));
            let axis = if layer.projection.uses_axis() {
                format!(" · {}", layer.projection_axis.label())
            } else {
                String::new()
            };
            ui.label(format!("{}{axis}", layer.projection.label()));
            let mut changed = ui
                .checkbox(&mut layer.projection_repeat, "Repeat")
                .changed();
            if ui.button("Numeric fields…").clicked() {
                self.ui.props_tab = PropsTab::Projection;
                self.ui.panels.properties = true;
            }
            if ui.button("Reset").clicked() {
                layer.reset_projection_placement();
                changed = true;
            }
            if changed {
                layers.mark_dirty();
            }
        }
    }

    pub(super) fn tool_rail(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("tool_rail")
            .exact_width(52.0)
            .resizable(false)
            .frame(
                egui::Frame::new()
                    .fill(theme::BG_BAR)
                    .inner_margin(egui::Margin::symmetric(6, 8))
                    .stroke(egui::Stroke::new(1.0_f32, theme::LINE)),
            )
            .show(ctx, |ui| {
                ui.vertical_centered(|ui| {
                    ui.spacing_mut().item_spacing.y = 4.0;
                    for (g, group) in TOOL_RAIL.iter().enumerate() {
                        if g > 0 {
                            widgets::divider(ui);
                        }
                        for &tool in *group {
                            let response = widgets::tool_button(
                                ui,
                                tool_icon(tool),
                                tool_key(tool),
                                self.tool == tool,
                            )
                            .on_hover_text(format!(
                                "{}  {}",
                                tool_name(tool),
                                tool_key(tool)
                            ));
                            if response.clicked() {
                                self.select_tool(tool);
                            }
                        }
                    }
                    // The current paint colour, at the foot of the rail.
                    ui.with_layout(egui::Layout::bottom_up(egui::Align::Center), |ui| {
                        self.colour_pair(ui);
                    });
                });
            });
    }

    pub(super) fn status_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("status_bar")
            .exact_height(28.0)
            .frame(
                egui::Frame::new()
                    .fill(theme::BG_CHROME)
                    .inner_margin(egui::Margin::symmetric(12, 0))
                    .stroke(egui::Stroke::new(1.0_f32, theme::LINE)),
            )
            .show(ctx, |ui| {
                ui.horizontal_centered(|ui| {
                    // Last result.
                    let tone = status_tone(&self.ui.status);
                    let (fg, _) = tone.colours();
                    ui.label(
                        RichText::new(format!("{} {}", tone.icon(), self.ui.status)).color(fg),
                    )
                    .on_hover_text(&self.ui.status);

                    // The newest background task.
                    if let Some(task) = self.tasks().into_iter().last() {
                        ui.add_space(12.0);
                        ui.label(RichText::new(&task.title).color(theme::TEXT));
                        let bar = match task.fraction {
                            Some(f) => egui::ProgressBar::new(f),
                            None => egui::ProgressBar::new(0.0).animate(true),
                        };
                        ui.add(
                            bar.desired_width(90.0)
                                .desired_height(6.0)
                                .fill(theme::ACCENT),
                        );
                        if let Some(f) = task.fraction {
                            ui.label(
                                RichText::new(format!("{:.0}%", f * 100.0))
                                    .monospace()
                                    .color(theme::TEXT_MUTED),
                            );
                        }
                        if task.cancellable && ui.small_button("Cancel").clicked() {
                            self.cancel_task(task.id);
                        }
                    }

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            RichText::new(if self.pt_mode {
                                "Path traced"
                            } else {
                                "Raster"
                            })
                            .monospace()
                            .color(theme::TEXT_MUTED),
                        );
                        ui.add_space(12.0);
                        self.vram_label(ui);
                        ui.add_space(12.0);
                        let layers = &self.texture_sets[self.active_set].layers;
                        let gb = |b: u64| b as f64 / (1024.0 * 1024.0 * 1024.0);
                        let undo = format!(
                            "Undo {} steps · {:.1} / {:.0} GB",
                            self.ui.history.len(),
                            gb(layers.history_bytes()),
                            gb(layers.undo_budget)
                        );
                        if ui
                            .add(
                                egui::Button::new(
                                    RichText::new(undo).monospace().color(theme::TEXT_MUTED),
                                )
                                .frame(false),
                            )
                            .on_hover_text(
                                "Stroke snapshots live in system RAM · click to change the budget",
                            )
                            .clicked()
                        {
                            self.ui.prefs_open = true;
                        }
                        // Tool hint, centred in what is left.
                        ui.centered_and_justified(|ui| {
                            ui.label(RichText::new(tool_hint(self.tool)).color(theme::TEXT_MUTED));
                        });
                    });
                });
            });
    }
}

/// A context-bar slider: label, short slider, mono value.
fn compact_slider(
    ui: &mut Ui,
    label: &str,
    value: &mut f32,
    range: std::ops::RangeInclusive<f32>,
    decimals: usize,
    suffix: &str,
) -> egui::Response {
    ui.label(RichText::new(label).color(theme::TEXT_MUTED));
    let mut response = widgets::slider(ui, value, range.clone(), false, 90.0);
    let speed = (range.end() - range.start()) / 300.0;
    let field = ui.add_sized(
        [44.0, 20.0],
        egui::DragValue::new(value)
            .range(range)
            .speed(speed)
            .fixed_decimals(decimals)
            .suffix(suffix),
    );
    if field.changed() {
        response.mark_changed();
    }
    response
}

/// How a status message reads: errors and warnings are worded as such
/// throughout the app, so the tone can be told from the text.
pub fn status_tone(status: &str) -> Tone {
    let s = status.to_lowercase();
    if s.contains("failed") || s.contains("error") || s.contains("could not") {
        Tone::Error
    } else if s.contains("needs ")
        || s.contains("no ")
        || s.starts_with("select ")
        || s.contains("already")
        || s.contains("nothing")
    {
        Tone::Warn
    } else if s.is_empty() || s.starts_with("default sphere") {
        Tone::Neutral
    } else {
        Tone::Ok
    }
}
