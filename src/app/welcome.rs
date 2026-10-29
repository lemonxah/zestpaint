//! The welcome screen at launch, like Blender's splash: open a model or a
//! project, pick up a recent project, or carry on with the default sphere,
//! and choose what launching does next time (`settings.json`'s `startup`).
//! Help ▸ Welcome screen brings it back.

use std::path::{Path, PathBuf};

use egui::{Align2, RichText, Sense, Ui};

use super::AppState;
use super::widgets;
use crate::config::Startup;
use crate::{icons, theme};

/// Recent projects listed.
const RECENT_SHOWN: usize = 8;

enum Action {
    OpenModel,
    OpenProject,
    Recent(PathBuf),
    Close,
}

/// The modal's id (its egui area).
const ID: &str = "welcome";
/// Content width; the columns are fixed so nothing can stretch the dialog.
const WIDTH: f32 = 620.0;
const START_WIDTH: f32 = 190.0;
const GAP: f32 = 28.0;

impl AppState {
    pub(super) fn welcome_screen(&mut self, ctx: &egui::Context) {
        if !self.ui.welcome_open {
            return;
        }
        let recent: Vec<PathBuf> =
            self.ui.recent_projects.iter().filter(|p| p.exists()).take(RECENT_SHOWN).cloned().collect();
        let mut startup = self.settings.startup;
        let action = show(ctx, &recent, &mut startup);
        if startup != self.settings.startup {
            self.settings.startup = startup;
            self.settings.save();
        }
        match action {
            Some(Action::OpenModel) => {
                self.ui.welcome_open = false;
                self.open_model_dialog();
            }
            Some(Action::OpenProject) => {
                self.ui.welcome_open = false;
                self.open_project_dialog();
            }
            Some(Action::Recent(path)) => {
                self.ui.welcome_open = false;
                self.open_project(path);
            }
            Some(Action::Close) => self.ui.welcome_open = false,
            None => {}
        }
    }
}

/// Draw the screen; what was picked, if anything (Esc and a click outside
/// close it).
fn show(ctx: &egui::Context, recent: &[PathBuf], startup: &mut Startup) -> Option<Action> {
    let mut action = None;
    let modal = egui::Modal::new(ID.into()).frame(super::dialogs::dialog_frame()).show(ctx, |ui| {
        ui.set_width(WIDTH);
        ui.horizontal(|ui| {
            ui.label(RichText::new("ZestPaint").font(theme::semibold(22.0)).color(theme::TEXT_STRONG));
            ui.label(RichText::new(format!("v{}", env!("CARGO_PKG_VERSION"))).color(theme::TEXT_DIM));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.add(egui::Button::new(icons::REMOVE).frame(false)).on_hover_text("Close (Esc)").clicked() {
                    action = Some(Action::Close);
                }
            });
        });
        ui.add_space(10.0);

        ui.horizontal_top(|ui| {
            ui.vertical(|ui| {
                ui.set_width(START_WIDTH);
                widgets::section(ui, "START", None);
                let wide = egui::vec2(START_WIDTH, 30.0);
                if ui
                    .add(widgets::primary_button(format!("{}  Open model…", icons::OPEN)).min_size(wide))
                    .on_hover_text("FBX or glTF (Ctrl O)")
                    .clicked()
                {
                    action = Some(Action::OpenModel);
                }
                if ui
                    .add(egui::Button::new(format!("{}  Open project…", icons::OPEN)).min_size(wide))
                    .on_hover_text("A .zestpaint project (Ctrl Shift O)")
                    .clicked()
                {
                    action = Some(Action::OpenProject);
                }
                if ui
                    .add(egui::Button::new("Default sphere").min_size(wide))
                    .on_hover_text("Close this and try the brushes on a sphere")
                    .clicked()
                {
                    action = Some(Action::Close);
                }
            });
            // A gap, not `ui.separator()`: in a horizontal row a separator
            // grows to the available height, which in a modal is unbounded.
            ui.add_space(GAP);
            ui.vertical(|ui| {
                ui.set_width(WIDTH - START_WIDTH - GAP - ui.spacing().item_spacing.x * 2.0);
                widgets::section(ui, "RECENT PROJECTS", None);
                if recent.is_empty() {
                    ui.label(RichText::new("Projects you open or save show up here.").color(theme::TEXT_DIM));
                }
                for path in recent {
                    if recent_row(ui, path) {
                        action = Some(Action::Recent(path.clone()));
                    }
                }
            });
        });

        ui.add_space(10.0);
        ui.separator();
        ui.horizontal(|ui| {
            ui.label(RichText::new("When ZestPaint starts").color(theme::TEXT_MUTED));
            egui::ComboBox::from_id_salt("welcome_startup").selected_text(startup.label()).show_ui(ui, |ui| {
                for option in Startup::ALL {
                    ui.selectable_value(startup, option, option.label());
                }
            });
        });
        ui.label(
            RichText::new(format!("Saved in {}", tilde(&crate::config::Settings::path())))
                .small()
                .color(theme::TEXT_DIM),
        );
    });
    if modal.should_close() && action.is_none() {
        action = Some(Action::Close);
    }
    action
}

/// One recent project: its name, the folder it is in and when it was last
/// saved. True when clicked.
fn recent_row(ui: &mut Ui, path: &Path) -> bool {
    let (rect, response) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 40.0), Sense::click());
    let painter = ui.painter();
    if response.hovered() {
        painter.rect_filled(rect, egui::CornerRadius::same(theme::RADIUS_ROW), theme::BG_BUTTON);
    }
    let name = path.file_stem().unwrap_or_default().to_string_lossy();
    let folder = path.parent().map(tilde).unwrap_or_default();
    let left = rect.left() + 10.0;
    painter.text(
        egui::pos2(left, rect.top() + 5.0),
        Align2::LEFT_TOP,
        name,
        theme::semibold(theme::BODY_SIZE + 1.0),
        theme::TEXT_STRONG,
    );
    if let Some(when) = saved_ago(path) {
        painter.text(
            egui::pos2(rect.right() - 10.0, rect.top() + 6.0),
            Align2::RIGHT_TOP,
            when,
            egui::FontId::proportional(theme::BODY_SIZE - 1.0),
            theme::TEXT_DIM,
        );
    }
    // Long folders keep their tail, which is the telling part.
    let max_chars = ((rect.width() - 20.0) / 6.0) as usize;
    let chars: Vec<char> = folder.chars().collect();
    let folder: String = if chars.len() > max_chars {
        std::iter::once('…').chain(chars[chars.len() - max_chars.saturating_sub(1)..].iter().copied()).collect()
    } else {
        folder
    };
    painter.text(
        egui::pos2(left, rect.bottom() - 5.0),
        Align2::LEFT_BOTTOM,
        folder,
        egui::FontId::proportional(theme::BODY_SIZE - 1.0),
        theme::TEXT_DIM,
    );
    response.on_hover_text(path.display().to_string()).on_hover_cursor(egui::CursorIcon::PointingHand).clicked()
}

/// "today", "yesterday", "5 days ago", … from the file's modified time.
fn saved_ago(path: &Path) -> Option<String> {
    let modified = std::fs::metadata(path).and_then(|m| m.modified()).ok()?;
    let secs = modified.elapsed().map(|d| d.as_secs()).unwrap_or(0);
    Some(match secs / 86_400 {
        _ if secs < 3_600 => format!("{} min ago", (secs / 60).max(1)),
        0 => format!("{} h ago", secs / 3_600),
        1 => "yesterday".into(),
        days if days < 60 => format!("{days} days ago"),
        days => format!("{} months ago", days / 30),
    })
}

/// A path with the home folder written as `~`.
fn tilde(path: &Path) -> String {
    match std::env::var_os("HOME").map(PathBuf::from) {
        Some(home) if path.starts_with(&home) => {
            format!("~/{}", path.strip_prefix(&home).unwrap_or(path).display())
        }
        _ => path.display().to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Laid out headless, the screen keeps its fixed width and a modest
    /// height: no element may stretch it (a vertical separator once made it
    /// taller than the window, its content off screen).
    #[test]
    fn welcome_screen_fits_the_window() {
        let ctx = egui::Context::default();
        crate::icons::install(&ctx);
        crate::theme::apply(&ctx);
        let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1280.0, 800.0));
        let recent: Vec<PathBuf> = (0..RECENT_SHOWN).map(|i| PathBuf::from(format!("/tmp/projects/a{i}.zestpaint"))).collect();
        let mut startup = Startup::Welcome;
        for _ in 0..3 {
            let input = egui::RawInput { screen_rect: Some(screen), ..Default::default() };
            let _ = ctx.run(input, |ctx| {
                show(ctx, &recent, &mut startup);
            });
        }
        let rect = ctx.memory(|m| m.area_rect(egui::Id::new(ID))).expect("the modal was laid out");
        assert!(rect.height() < 600.0, "too tall: {rect:?}");
        assert!(rect.width() < WIDTH + 60.0, "too wide: {rect:?}");
        assert!(screen.contains_rect(rect), "off screen: {rect:?}");
    }
}
