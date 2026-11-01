//! Small modal and floating windows: the Ctrl K command palette, the
//! auto-unwrap confirmation, preferences (startup, undo budget), the shortcut list,
//! tool gestures and About.

use egui::{Key, RichText};

use super::commands::{all_menu_commands, panel_controls, Cmd, PanelTarget};
use super::widgets;
use super::AppState;
use crate::theme;

/// Command palette state.
#[derive(Default)]
pub struct Palette {
    pub open: bool,
    pub query: String,
    pub selected: usize,
}

/// One palette hit: a command to run, or a panel control to jump to.
enum Hit {
    Command(Cmd),
    Control(PanelTarget),
}

/// Subsequence match: every query character appears in order. Lower is
/// better; None means no match. Rewards contiguous runs and early hits.
fn fuzzy_score(query: &str, text: &str) -> Option<i32> {
    if query.is_empty() {
        return Some(0);
    }
    let text = text.to_lowercase();
    let mut score = 0;
    let mut last: Option<usize> = None;
    let mut from = 0;
    for qc in query.to_lowercase().chars().filter(|c| !c.is_whitespace()) {
        let pos = text[from..].find(qc)? + from;
        score += match last {
            Some(l) if pos == l + 1 => 0,
            Some(l) => (pos - l) as i32,
            None => pos as i32,
        };
        last = Some(pos);
        from = pos + qc.len_utf8();
    }
    Some(score)
}

impl AppState {
    pub(super) fn windows(&mut self, ctx: &egui::Context) {
        self.palette(ctx);
        self.confirm_unwrap_window(ctx);

        if self.ui.prefs_open {
            let mut open = true;
            egui::Window::new("Preferences").open(&mut open).collapsible(false).resizable(false).show(ctx, |ui| {
                ui.set_width(320.0);
                widgets::section(ui, "Startup", None);
                let mut startup = self.settings.startup;
                egui::ComboBox::from_id_salt("prefs_startup").selected_text(startup.label()).show_ui(ui, |ui| {
                    for option in crate::config::Startup::ALL {
                        ui.selectable_value(&mut startup, option, option.label());
                    }
                });
                if startup != self.settings.startup {
                    self.settings.startup = startup;
                    self.settings.save();
                }
                widgets::section(ui, "Undo", None);
                let layers = &mut self.texture_sets[self.active_set].layers;
                let mut budget_gb = (layers.undo_budget as f64 / (1024.0 * 1024.0 * 1024.0)) as f32;
                if widgets::SliderRow::new("Budget", &mut budget_gb, 0.5..=32.0)
                    .decimals(1)
                    .suffix(" GB")
                    .hint("Stroke snapshots are kept in system RAM; the oldest are dropped only once this is reached")
                    .show(ui)
                    .changed()
                {
                    let bytes = (budget_gb as f64 * 1024.0 * 1024.0 * 1024.0) as u64;
                    for set in &mut self.texture_sets {
                        set.layers.undo_budget = bytes;
                    }
                    crate::layers::set_default_undo_budget(bytes);
                    self.settings.undo_budget_gb = budget_gb;
                    self.settings.save();
                }
                let layers = &self.texture_sets[self.active_set].layers;
                ui.label(
                    RichText::new(format!(
                        "{} steps held · {:.2} GB used",
                        layers.undo_depth(),
                        layers.history_bytes() as f64 / (1024.0 * 1024.0 * 1024.0)
                    ))
                    .color(theme::TEXT_MUTED),
                );
                ui.add_space(6.0);
                ui.label(
                    RichText::new(format!("Saved in {}", crate::config::Settings::path().display()))
                        .small()
                        .color(theme::TEXT_DIM),
                );
            });
            self.ui.prefs_open = open;
        }

        if self.ui.shortcuts_open {
            let mut open = true;
            egui::Window::new("Keyboard shortcuts").open(&mut open).collapsible(false).show(ctx, |ui| {
                egui::ScrollArea::vertical().max_height(480.0).show(ui, |ui| {
                    egui::Grid::new("shortcut_grid").num_columns(3).spacing([18.0, 4.0]).striped(true).show(ui, |ui| {
                        for (menu, cmd) in all_menu_commands() {
                            let keys = cmd.shortcut_text();
                            if keys.is_empty() || cmd.planned() {
                                continue;
                            }
                            ui.label(RichText::new(keys).monospace().color(theme::ACCENT));
                            ui.label(cmd.label(self));
                            ui.label(RichText::new(menu).color(theme::TEXT_DIM));
                            ui.end_row();
                        }
                        for (keys, what) in [
                            ("Ctrl Y", "Redo"),
                            ("[ / ]", "Brush size down / up (Shift: faster)"),
                            ("Alt+LMB", "Orbit while a brush tool is active"),
                            ("Ctrl+Scroll", "Brush size / projection size"),
                            ("Shift+Scroll", "Tip rotation / projection spin"),
                            ("Ctrl+LMB", "Set clone source"),
                        ] {
                            ui.label(RichText::new(keys).monospace().color(theme::ACCENT));
                            ui.label(what);
                            ui.label(RichText::new("Viewport").color(theme::TEXT_DIM));
                            ui.end_row();
                        }
                    });
                });
            });
            self.ui.shortcuts_open = open;
        }

        if self.ui.gestures_open {
            let mut open = true;
            egui::Window::new("Tool gestures").open(&mut open).collapsible(false).resizable(false).show(ctx, |ui| {
                for tool in super::shell::TOOL_RAIL.iter().flat_map(|g| g.iter()) {
                    ui.label(RichText::new(super::shell::tool_name(*tool)).font(theme::semibold(theme::BODY_SIZE)));
                    ui.label(RichText::new(super::shell::tool_hint(*tool)).color(theme::TEXT_MUTED));
                    ui.add_space(4.0);
                }
                widgets::divider(ui);
                super::overlays::gesture_sheet(ui);
            });
            self.ui.gestures_open = open;
        }

        if self.ui.about_open {
            let mut open = true;
            egui::Window::new("About ZestSurface").open(&mut open).collapsible(false).resizable(false).show(ctx, |ui| {
                ui.label(RichText::new("ZestSurface").font(theme::semibold(theme::TITLE_SIZE)));
                ui.label(format!("Version {}", env!("CARGO_PKG_VERSION")));
                ui.label(RichText::new("A 3D texture painter in Rust on Vulkan, with an egui interface.").color(theme::TEXT_MUTED));
                ui.label(
                    RichText::new("Fonts: IBM Plex Sans & Mono (SIL OFL 1.1) · Icons: Lucide (ISC)")
                        .small()
                        .color(theme::TEXT_DIM),
                );
            });
            self.ui.about_open = open;
        }
    }

    fn confirm_unwrap_window(&mut self, ctx: &egui::Context) {
        if !self.ui.confirm_unwrap {
            return;
        }
        let set = self.texture_sets[self.active_set].label();
        let modal = egui::Modal::new("confirm_unwrap".into()).show(ctx, |ui| {
            ui.set_width(360.0);
            ui.label(RichText::new("Auto-unwrap UVs?").font(theme::semibold(theme::TITLE_SIZE)));
            ui.add_space(4.0);
            ui.label(format!("Re-unwraps {set} with new UVs."));
            widgets::chip(ui, "This set's layers reset. Other sets keep their paint.", widgets::Tone::Warn);
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui.add(widgets::warn_button("Unwrap and reset layers")).clicked() {
                    self.ui.confirm_unwrap = false;
                    self.unwrap_uvs();
                }
                if ui.button("Cancel").clicked() {
                    self.ui.confirm_unwrap = false;
                }
            });
        });
        if modal.should_close() {
            self.ui.confirm_unwrap = false;
        }
    }

    fn palette(&mut self, ctx: &egui::Context) {
        if !self.ui.palette.open {
            return;
        }
        // Score every command and panel control against the query.
        let query = self.ui.palette.query.trim().to_string();
        let mut hits: Vec<(i32, String, String, &'static str, Hit, bool)> = Vec::new();
        for (menu, cmd) in all_menu_commands() {
            let label = cmd.label(self);
            if let Some(score) = fuzzy_score(&query, &label) {
                hits.push((score, label, menu.to_string(), cmd.shortcut_text(), Hit::Command(cmd), cmd.enabled(self)));
            }
        }
        for (label, target) in panel_controls() {
            if let Some(score) = fuzzy_score(&query, label) {
                let place = match target {
                    PanelTarget::Props(tab) => format!("Properties › {}", tab.label()),
                    PanelTarget::BrushDock => "Brush panel".into(),
                    PanelTarget::Display => "Display & lighting".into(),
                };
                hits.push((score + 1, label.to_string(), place, "Tab", Hit::Control(*target), true));
            }
        }
        hits.sort_by_key(|h| h.0);
        hits.truncate(12);
        let selected = self.ui.palette.selected.min(hits.len().saturating_sub(1));

        let (up, down, enter, tab) = ctx.input(|i| {
            (i.key_pressed(Key::ArrowUp), i.key_pressed(Key::ArrowDown), i.key_pressed(Key::Enter), i.key_pressed(Key::Tab))
        });
        let mut chosen: Option<usize> = None;
        let modal = egui::Modal::new("command_palette".into())
            .frame(
                egui::Frame::new()
                    .fill(theme::BG_PANEL)
                    .stroke(egui::Stroke::new(1.0_f32, theme::LINE_INNER))
                    .corner_radius(egui::CornerRadius::same(theme::RADIUS_POPOVER + 2))
                    .inner_margin(egui::Margin::same(12)),
            )
            .show(ctx, |ui| {
                ui.set_width(520.0);
                let edit = ui.add(
                    egui::TextEdit::singleline(&mut self.ui.palette.query)
                        .hint_text("Search commands and settings…")
                        .desired_width(f32::INFINITY)
                        .lock_focus(true),
                );
                edit.request_focus();
                if edit.changed() {
                    self.ui.palette.selected = 0;
                }
                ui.add_space(6.0);
                for (i, (_, label, place, keys, hit, enabled)) in hits.iter().enumerate() {
                    let on = i == selected;
                    let frame = egui::Frame::new()
                        .fill(if on { theme::BG_SELECTED } else { egui::Color32::TRANSPARENT })
                        .stroke(if on { egui::Stroke::new(1.0_f32, theme::ACCENT) } else { egui::Stroke::NONE })
                        .corner_radius(egui::CornerRadius::same(theme::RADIUS_CONTROL))
                        .inner_margin(egui::Margin::symmetric(8, 4));
                    let row = frame.show(ui, |ui| {
                        ui.horizontal(|ui| {
                            let colour = if *enabled { theme::TEXT } else { theme::TEXT_DIM };
                            ui.label(RichText::new(label).color(colour));
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                ui.label(RichText::new(place).color(theme::TEXT_DIM));
                                if !keys.is_empty() {
                                    widgets::key_badge(ui, keys);
                                }
                            });
                        });
                    });
                    let response = ui.interact(row.response.rect, ui.id().with(("palette_row", i)), egui::Sense::click());
                    if response.clicked() {
                        chosen = Some(i);
                    }
                    if matches!(hit, Hit::Command(c) if c.planned()) {
                        response.on_hover_text("Coming soon");
                    }
                }
                if hits.is_empty() {
                    ui.label(RichText::new("No matches").color(theme::TEXT_DIM));
                }
                ui.add_space(4.0);
                ui.label(RichText::new("↵ run · Tab go to its panel · Esc close").small().color(theme::TEXT_DIM));
            });

        if up {
            self.ui.palette.selected = selected.saturating_sub(1);
        }
        if down {
            self.ui.palette.selected = (selected + 1).min(hits.len().saturating_sub(1));
        }
        if enter || tab {
            chosen = Some(selected);
        }
        if modal.should_close() {
            self.ui.palette.open = false;
            return;
        }
        if let Some(i) = chosen
            && let Some((_, _, _, _, hit, _)) = hits.into_iter().nth(i)
        {
            self.ui.palette.open = false;
            match hit {
                Hit::Command(cmd) if !tab => self.run_command(cmd, ctx),
                // Tab on a command: open its menu's panel where there is one.
                Hit::Command(_) => {}
                Hit::Control(target) => match target {
                    PanelTarget::Props(t) => {
                        self.ui.props_tab = t;
                        self.ui.panels.properties = true;
                    }
                    PanelTarget::BrushDock => self.ui.panels.brush = true,
                    PanelTarget::Display => {
                        self.ui.display_open = true;
                        self.ui.display_opened_at = ctx.input(|i| i.time);
                    }
                },
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fuzzy_score;

    #[test]
    fn fuzzy_matches_in_order_and_prefers_tight_runs() {
        assert!(fuzzy_score("exp", "Export maps…").is_some());
        assert!(fuzzy_score("xpe", "Export maps…").is_none());
        let tight = fuzzy_score("bake", "Bake AO").unwrap();
        let loose = fuzzy_score("bake", "Brush library window").or(Some(i32::MAX)).unwrap();
        assert!(tight < loose);
        assert_eq!(fuzzy_score("", "anything"), Some(0));
    }
}
