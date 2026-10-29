//! Overlays on the 3D viewport: only things about the view itself — where
//! you are, how it renders, display and lighting, the axis gizmo, the
//! empty-state prompt — plus the background-task cards and toasts.

use std::time::{Duration, Instant};

use egui::{Align2, CornerRadius, RichText, Stroke, Ui};
use glam::Vec3;

use super::widgets::{self, SliderRow, Tone};
use super::AppState;
use crate::icons;
use crate::theme;

/// A finished-work notice in the corner. Success fades after a few seconds;
/// errors stay until dismissed.
pub struct Toast {
    pub text: String,
    pub tone: Tone,
    pub born: Instant,
    pub action: Option<ToastAction>,
    pub expanded: bool,
}

pub enum ToastAction {
    OpenFolder(std::path::PathBuf),
    /// Longer text revealed by a Details button.
    Details(String),
}

const TOAST_LIFETIME: Duration = Duration::from_secs(6);

/// One running background job, as the cards and the status bar show it.
pub struct TaskView {
    pub id: u64,
    pub title: String,
    pub step: String,
    pub done: usize,
    pub total: usize,
    pub fraction: Option<f32>,
    /// Only where stopping is safe (the result is simply discarded).
    pub cancellable: bool,
}

/// Task id of the model import.
pub const IMPORT_TASK: u64 = 1;
/// Task id of a project save or open.
pub const PROJECT_TASK: u64 = 2;

impl AppState {
    pub(super) fn push_toast(&mut self, tone: Tone, text: impl Into<String>, action: Option<ToastAction>) {
        self.ui.toasts.push(Toast { text: text.into(), tone, born: Instant::now(), action, expanded: false });
    }

    /// Every background job, oldest first. Bakes and exports run on the GPU
    /// queue synchronously and so never appear here.
    pub(super) fn tasks(&self) -> Vec<TaskView> {
        let mut tasks = Vec::new();
        if let Some(job) = self.import.as_ref() {
            let p = job.progress.lock().ok().map(|p| p.clone()).unwrap_or_default();
            let name = job.path.file_name().unwrap_or_default().to_string_lossy().to_string();
            tasks.push(TaskView {
                id: IMPORT_TASK,
                title: format!("Importing {name}"),
                step: p.label.clone(),
                done: p.done,
                total: p.total,
                fraction: p.fraction(),
                cancellable: true,
            });
        }
        if let Some(job) = self.project_job.as_ref() {
            let name = job.path().file_name().unwrap_or_default().to_string_lossy().to_string();
            let (title, step) = if job.is_saving() {
                (format!("Saving {name}"), "Compressing layers")
            } else {
                (format!("Opening {name}"), "Reading layers")
            };
            tasks.push(TaskView {
                id: PROJECT_TASK,
                title,
                step: step.into(),
                done: 0,
                total: 0,
                fraction: None,
                // A half-written save or half-read project can't be dropped
                // safely mid-way; it finishes on its own.
                cancellable: false,
            });
        }
        tasks
    }

    pub(super) fn cancel_task(&mut self, id: u64) {
        if id == IMPORT_TASK && let Some(job) = self.import.take() {
            // The worker can't be interrupted mid-parse; let it finish on its
            // own and drop what it returns.
            drop(job);
            self.ui.status = "Import cancelled".into();
        }
    }

    pub(super) fn viewport_overlays(&mut self, ctx: &egui::Context, rect: egui::Rect) {
        let margin = 12.0;
        let chip_frame = || {
            egui::Frame::new()
                .fill(theme::BG_PANEL.gamma_multiply(0.94))
                .stroke(Stroke::new(1.0_f32, theme::LINE))
                .corner_radius(CornerRadius::same(theme::RADIUS_ROW))
                .inner_margin(egui::Margin::symmetric(10, 6))
        };

        // Top-left: Object › Set · Layer, and the local-view badge.
        let mut leave_local_view = false;
        egui::Area::new("vp_breadcrumb".into())
            .fixed_pos(rect.min + egui::vec2(margin, margin))
            .show(ctx, |ui| {
                chip_frame().show(ui, |ui| {
                    ui.horizontal(|ui| {
                        let set = &self.texture_sets[self.active_set];
                        ui.label(RichText::new(&set.object).color(theme::TEXT_MUTED));
                        ui.label(RichText::new(icons::CHEVRON_RIGHT).color(theme::TEXT_DIM));
                        ui.label(RichText::new(&set.material).font(theme::semibold(theme::BODY_SIZE)).color(theme::TEXT_STRONG));
                        ui.label(RichText::new(format!("· {}", set.layers.active_layer().name)).color(theme::TEXT_MUTED));
                        if self.isolate_backup.is_some() {
                            ui.separator();
                            leave_local_view = ui
                                .add(egui::Button::new(RichText::new("Local view   /").color(theme::ACCENT)).frame(false))
                                .on_hover_text("Only the isolated pieces are shown. Click or press / to leave.")
                                .clicked();
                        }
                    });
                });
            });
        if leave_local_view {
            self.toggle_isolate();
        }

        // Top-right: render switch, spp, view toggles.
        let toggles = egui::Area::new("vp_toggles".into())
            .pivot(Align2::RIGHT_TOP)
            .fixed_pos(rect.right_top() + egui::vec2(-margin, margin))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    let mut pt = self.pt_mode;
                    if widgets::segmented(ui, &mut pt, &[(false, "Raster"), (true, "Path traced")]) {
                        if pt && !self.pt_mode {
                            self.pt_scene_stale = true;
                        }
                        self.pt_mode = pt;
                    }
                    if self.pt_mode {
                        widgets::chip(ui, &format!("{} spp", self.pathtracer.viewport_samples()), Tone::Neutral);
                    }
                    if ui
                        .add(egui::Button::new(format!("{}  Studio", icons::IMAGE)).selected(self.ui.studio_open))
                        .on_hover_text("Pose the avatar, light it and render (F10). Render: F12")
                        .clicked()
                    {
                        self.ui.studio_open = !self.ui.studio_open;
                        if self.ui.studio_open && !self.pt_mode {
                            self.pt_mode = true;
                            self.pt_scene_stale = true;
                        }
                    }
                    chip_frame().inner_margin(egui::Margin::same(3)).show(ui, |ui| {
                        ui.spacing_mut().item_spacing.x = 2.0;
                        if widgets::icon_toggle(ui, icons::UV_CHECKER, self.ui.uv_checker).on_hover_text("UV checker").clicked() {
                            self.ui.uv_checker = !self.ui.uv_checker;
                        }
                        if widgets::icon_toggle(ui, icons::MASK_VIEW, self.ui.mask_overlay).on_hover_text("Show mask on model").clicked() {
                            self.ui.mask_overlay = !self.ui.mask_overlay;
                        }
                        if widgets::icon_toggle(ui, icons::OVERRIDE, self.ui.override_material).on_hover_text("Override material").clicked() {
                            self.ui.override_material = !self.ui.override_material;
                        }
                        if widgets::icon_toggle(ui, icons::ENVIRONMENT, self.ui.display_open).on_hover_text("Display & lighting").clicked() {
                            self.ui.display_open = !self.ui.display_open;
                            self.ui.display_opened_at = ctx.input(|i| i.time);
                        }
                    });
                });
            });

        if self.ui.display_open {
            self.display_popover(ctx, toggles.response.rect);
        }

        // Bottom-left: axis gizmo.
        self.axis_gizmo(ctx, egui::pos2(rect.left() + 44.0, rect.bottom() - 44.0));

        // Bottom-centre: the empty state, while the default sphere is up.
        if self.ui.default_scene {
            egui::Area::new("vp_empty".into())
                .pivot(Align2::CENTER_BOTTOM)
                .fixed_pos(rect.center_bottom() + egui::vec2(0.0, -margin - 10.0))
                .show(ctx, |ui| {
                    chip_frame().inner_margin(egui::Margin::symmetric(14, 8)).show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("Default sphere — open a model to start").color(theme::TEXT));
                            if ui.add(widgets::primary_button("Open model…")).clicked() {
                                self.open_model_dialog();
                            }
                            widgets::key_badge(ui, "Ctrl O");
                            // Pick up where you left off.
                            let last = self.ui.recent_projects.iter().find(|p| p.exists()).cloned();
                            if let Some(last) = last {
                                let name = last.file_stem().unwrap_or_default().to_string_lossy().to_string();
                                if ui
                                    .button(format!("Open {name}"))
                                    .on_hover_text(last.display().to_string())
                                    .clicked()
                                {
                                    self.open_project(last);
                                }
                            }
                        });
                    });
                });
        }

        // Bottom-right: task cards above toasts.
        self.ui.toasts.retain(|t| t.tone == Tone::Error || t.born.elapsed() < TOAST_LIFETIME);
        let tasks = self.tasks();
        if tasks.is_empty() && self.ui.toasts.is_empty() {
            return;
        }
        let mut cancel = None;
        let mut dismiss = None;
        let mut open_folder = None;
        // Sit above the empty-state chip while it shows, so they never overlap.
        let lift = if self.ui.default_scene { 56.0 } else { 0.0 };
        egui::Area::new("vp_tasks".into())
            .pivot(Align2::RIGHT_BOTTOM)
            .fixed_pos(rect.right_bottom() + egui::vec2(-margin, -margin - lift))
            .show(ctx, |ui| {
                ui.set_width(300.0);
                for task in &tasks {
                    chip_frame().inner_margin(egui::Margin::same(10)).show(ui, |ui| {
                        ui.set_width(280.0);
                        ui.horizontal(|ui| {
                            ui.label(RichText::new(&task.title).font(theme::semibold(theme::BODY_SIZE)));
                            if task.cancellable {
                                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                    if ui.small_button("Cancel").clicked() {
                                        cancel = Some(task.id);
                                    }
                                });
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label(RichText::new(&task.step).color(theme::TEXT_MUTED));
                            if task.total > 0 {
                                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                    ui.label(RichText::new(format!("{} / {}", task.done, task.total)).monospace().color(theme::TEXT_DIM));
                                });
                            }
                        });
                        let bar = match task.fraction {
                            Some(f) => egui::ProgressBar::new(f),
                            // Parsing gives no counts, so animate rather than lie.
                            None => egui::ProgressBar::new(0.0).animate(true),
                        };
                        ui.add(bar.desired_height(6.0).fill(theme::ACCENT));
                    });
                    ui.add_space(6.0);
                }
                for (i, toast) in self.ui.toasts.iter_mut().enumerate() {
                    let (fg, bg) = toast.tone.colours();
                    egui::Frame::new()
                        .fill(bg)
                        .stroke(Stroke::new(1.0_f32, fg.linear_multiply(0.35)))
                        .corner_radius(CornerRadius::same(theme::RADIUS_ROW))
                        .inner_margin(egui::Margin::same(10))
                        .show(ui, |ui| {
                            ui.set_width(280.0);
                            ui.horizontal(|ui| {
                                ui.label(RichText::new(toast.tone.icon()).color(fg));
                                ui.add(egui::Label::new(RichText::new(&toast.text).color(fg)).wrap());
                                ui.with_layout(egui::Layout::right_to_left(egui::Align::TOP), |ui| {
                                    if ui.add(egui::Button::new(RichText::new(icons::REMOVE).color(fg)).frame(false)).clicked() {
                                        dismiss = Some(i);
                                    }
                                });
                            });
                            match &toast.action {
                                Some(ToastAction::OpenFolder(dir)) => {
                                    if ui.small_button("Open folder").clicked() {
                                        open_folder = Some(dir.clone());
                                    }
                                }
                                Some(ToastAction::Details(details)) => {
                                    if ui.small_button(if toast.expanded { "Hide details" } else { "Details" }).clicked() {
                                        toast.expanded = !toast.expanded;
                                    }
                                    if toast.expanded {
                                        ui.label(RichText::new(details).monospace().small().color(fg));
                                    }
                                }
                                None => {}
                            }
                        });
                    ui.add_space(6.0);
                }
            });
        if let Some(id) = cancel {
            self.cancel_task(id);
        }
        if let Some(i) = dismiss {
            self.ui.toasts.remove(i);
        }
        if let Some(dir) = open_folder {
            let _ = std::process::Command::new("xdg-open").arg(&dir).spawn();
        }
        // Keep fading toasts and animating bars.
        ctx.request_repaint_after(Duration::from_millis(250));
    }

    fn display_popover(&mut self, ctx: &egui::Context, anchor: egui::Rect) {
        let mut close = false;
        let area = egui::Area::new("vp_display".into())
            .order(egui::Order::Foreground)
            .pivot(Align2::RIGHT_TOP)
            .fixed_pos(anchor.right_bottom() + egui::vec2(0.0, 6.0))
            .show(ctx, |ui| {
                egui::Frame::new()
                    .fill(theme::BG_PANEL)
                    .stroke(Stroke::new(1.0_f32, theme::LINE_INNER))
                    .corner_radius(CornerRadius::same(theme::RADIUS_POPOVER))
                    .inner_margin(egui::Margin::same(14))
                    .show(ui, |ui| {
                        ui.set_width(272.0);
                        ui.horizontal(|ui| {
                            ui.label(theme::section_header("Display & lighting"));
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if ui.add(egui::Button::new(icons::REMOVE).frame(false)).clicked() {
                                    close = true;
                                }
                            });
                        });
                        ui.add_space(4.0);
                        self.environment_picker(ui);
                        SliderRow::new("Exposure", &mut self.ui.exposure, 0.1..=4.0).log().show(ui);
                        SliderRow::new("Env intensity", &mut self.ui.env_intensity, 0.0..=3.0).show(ui);
                        SliderRow::new("Key light", &mut self.ui.key_light, 0.0..=2.0).show(ui);
                        ui.add_enabled_ui(self.ui.key_light > 0.0, |ui| {
                            ui.checkbox(&mut self.ui.key_follow_camera, "Light follows camera")
                                .on_hover_text("Orbit to sweep the highlight across the surface");
                            ui.add_enabled_ui(!self.ui.key_follow_camera, |ui| {
                                SliderRow::new("Rotation", &mut self.ui.key_azimuth, -180.0..=180.0).decimals(0).suffix("°").show(ui);
                                SliderRow::new("Height", &mut self.ui.key_elevation, -89.0..=89.0).decimals(0).suffix("°").show(ui);
                            });
                        });
                        widgets::divider(ui);
                        ui.checkbox(&mut self.ui.uv_checker, "UV checker");
                        ui.checkbox(&mut self.ui.mask_overlay, "Show active mask on model");
                        ui.checkbox(&mut self.ui.override_material, "Override material");
                        ui.add_enabled_ui(self.ui.override_material, |ui| {
                            ui.horizontal(|ui| {
                                ui.add_space(22.0);
                                ui.color_edit_button_rgb(&mut self.ui.base_color);
                                ui.label(
                                    RichText::new(format!("Metal {:.2} · Rough {:.2}", self.ui.metallic, self.ui.roughness))
                                        .color(theme::TEXT_MUTED),
                                );
                            });
                            SliderRow::new("Metallic", &mut self.ui.metallic, 0.0..=1.0).show(ui);
                            SliderRow::new("Roughness", &mut self.ui.roughness, 0.0..=1.0).show(ui);
                        });
                        widgets::divider(ui);
                        gesture_sheet(ui);
                    });
            });
        // Closes on its X, Esc, or a click outside (but not the click that
        // opened it, which lands on the toggle).
        let clicked_outside = ctx.input(|i| {
            i.pointer.any_click()
                && i.time - self.ui.display_opened_at > 0.2
                && i.pointer.interact_pos().is_some_and(|p| !area.response.rect.contains(p) && !anchor.contains(p))
        });
        if close || clicked_outside || ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            self.ui.display_open = false;
        }
    }

    fn environment_picker(&mut self, ui: &mut Ui) {
        let mut load = false;
        let mut chosen: Option<Option<std::path::PathBuf>> = None;
        ui.horizontal(|ui| {
            widgets::fixed(ui, theme::SLIDER_LABEL_WIDTH, |ui| ui.label("Environment"));
            egui::ComboBox::from_id_salt("environment")
                .width(ui.available_width())
                .selected_text(self.ui.env_name.clone())
                .show_ui(ui, |ui| {
                    if ui.selectable_label(self.ui.env_path.is_none(), "Procedural sky").clicked() {
                        chosen = Some(None);
                    }
                    for path in self.ui.hdri_history.clone() {
                        let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
                        if ui.selectable_label(self.ui.env_path.as_ref() == Some(&path), name).clicked() {
                            chosen = Some(Some(path));
                        }
                    }
                });
        });
        // A bounded row: an unbounded right-to-left layout in a popover would
        // claim all the height below it.
        ui.allocate_ui_with_layout(
            egui::vec2(ui.available_width(), 24.0),
            egui::Layout::right_to_left(egui::Align::Center),
            |ui| load = ui.button("Load HDRI…").clicked(),
        );
        if load {
            self.open_hdri_dialog();
        }
        match chosen {
            Some(None) if self.ui.env_path.is_some() => {
                let env = crate::environment::Environment::procedural(&self.gpu);
                self.renderer.set_environment(env);
                self.ui.env_path = None;
                self.ui.env_name = "Procedural sky".into();
                self.pt_scene_stale = true;
            }
            Some(Some(path)) if self.ui.env_path.as_ref() != Some(&path) => self.load_hdri(&path),
            _ => {}
        }
    }

    fn axis_gizmo(&self, ctx: &egui::Context, origin: egui::Pos2) {
        let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Middle, "vp_gizmo".into()));
        let view = self.camera.view_matrix();
        let mut axes = [
            (Vec3::X, theme::AXIS_X, "X"),
            (Vec3::Y, theme::AXIS_Y, "Y"),
            (Vec3::Z, theme::AXIS_Z, "Z"),
        ]
        .map(|(axis, colour, label)| (view.transform_vector3(axis), colour, label));
        // Far axes first, so the near ones draw over them.
        axes.sort_by(|a, b| a.0.z.total_cmp(&b.0.z));
        for (dir, colour, label) in axes {
            let tip = origin + egui::vec2(dir.x, -dir.y) * 26.0;
            let faded = if dir.z < -0.2 { colour.gamma_multiply(0.55) } else { colour };
            painter.line_segment([origin, tip], Stroke::new(1.5_f32, faded));
            painter.text(
                origin + egui::vec2(dir.x, -dir.y) * 34.0,
                Align2::CENTER_CENTER,
                label,
                egui::FontId::monospace(10.0),
                faded,
            );
        }
    }
}

/// Camera gestures, as a two-column mono sheet.
pub fn gesture_sheet(ui: &mut Ui) {
    egui::Grid::new(ui.next_auto_id()).num_columns(2).spacing([16.0, 2.0]).show(ui, |ui| {
        for (keys, action) in [
            ("Alt+LMB", "Orbit (brush tools)"),
            ("LMB", "Orbit (Orbit tool)"),
            ("RMB / MMB", "Pan"),
            ("Scroll", "Zoom"),
            ("Ctrl+Scroll", "Brush size"),
            ("Shift+Scroll", "Tip / stamp rotation"),
        ] {
            ui.label(RichText::new(keys).monospace().color(theme::TEXT_MUTED));
            ui.label(RichText::new(action).color(theme::TEXT_MUTED));
            ui.end_row();
        }
    });
}
