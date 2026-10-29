//! Save / open project files, and the single-layer exports and mask import
//! that go with editing a project's layers outside the app.
//!
//! The file format is `crate::project`'s. Saving reads the GPU back on the
//! main thread (quick) and compresses on a worker (slow); opening reads and
//! decompresses on a worker and uploads on the main thread.

use std::path::{Path, PathBuf};
use std::thread::JoinHandle;

use super::overlays::ToastAction;
use super::widgets::Tone;
use super::{AppState, PAINT_RESOLUTION};
use crate::project::{self, CameraState, DisplayState, LoadedProject};

/// A project save or open running on a worker thread.
pub enum ProjectJob {
    Saving { path: PathBuf, handle: JoinHandle<anyhow::Result<()>> },
    Loading { path: PathBuf, handle: JoinHandle<anyhow::Result<LoadedProject>> },
}

impl ProjectJob {
    pub fn path(&self) -> &Path {
        match self {
            ProjectJob::Saving { path, .. } | ProjectJob::Loading { path, .. } => path,
        }
    }

    pub fn is_saving(&self) -> bool {
        matches!(self, ProjectJob::Saving { .. })
    }

    fn is_finished(&self) -> bool {
        match self {
            ProjectJob::Saving { handle, .. } => handle.is_finished(),
            ProjectJob::Loading { handle, .. } => handle.is_finished(),
        }
    }
}

/// Entries kept in File ▸ Open recent.
const MAX_RECENT: usize = 10;

fn recent_file() -> PathBuf {
    super::dialogs::config_file("recent_projects.json")
}

/// The recent-projects list from disk, most recent first.
pub fn load_recent() -> Vec<PathBuf> {
    std::fs::read_to_string(recent_file())
        .ok()
        .and_then(|text| serde_json::from_str::<Vec<PathBuf>>(&text).ok())
        .unwrap_or_default()
}

fn save_recent(list: &[PathBuf]) {
    let path = recent_file();
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let _ = std::fs::write(path, serde_json::to_string_pretty(list).unwrap_or_default());
}

fn file_name(path: &Path) -> String {
    path.file_name().unwrap_or_default().to_string_lossy().to_string()
}

/// `name.zestpaint`, whatever extension (if any) was typed.
fn with_project_extension(path: PathBuf) -> PathBuf {
    if path.extension().is_some_and(|e| e.eq_ignore_ascii_case(project::EXTENSION)) {
        path
    } else {
        let mut name = path.file_name().unwrap_or_default().to_os_string();
        name.push(".");
        name.push(project::EXTENSION);
        path.with_file_name(name)
    }
}

impl AppState {
    /// Put a project at the top of the recent list (and on disk).
    fn remember_recent(&mut self, path: &Path) {
        let path = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        let list = &mut self.ui.recent_projects;
        list.retain(|p| *p != path);
        list.insert(0, path);
        list.truncate(MAX_RECENT);
        save_recent(list);
    }

    /// The File ▸ Open recent submenu.
    pub(super) fn recent_menu(&mut self, ui: &mut egui::Ui) {
        let mut open = None;
        let mut clear = false;
        let enabled = !self.ui.recent_projects.is_empty();
        ui.add_enabled_ui(enabled, |ui| {
            ui.menu_button(format!("{}  Open recent", crate::icons::OPEN), |ui| {
                ui.set_min_width(260.0);
                for path in &self.ui.recent_projects {
                    let exists = path.exists();
                    let name = path.file_stem().unwrap_or_default().to_string_lossy().to_string();
                    let folder = path.parent().map(|p| p.display().to_string()).unwrap_or_default();
                    let text = if exists {
                        egui::RichText::new(name)
                    } else {
                        egui::RichText::new(format!("{name} (missing)")).color(crate::theme::TEXT_DIM)
                    };
                    let response = ui.add(egui::Button::new(text).frame(false)).on_hover_text(&folder);
                    if response.clicked() {
                        open = Some(path.clone());
                        ui.close_menu();
                    }
                }
                ui.separator();
                if ui.button("Clear recent projects").clicked() {
                    clear = true;
                    ui.close_menu();
                }
            })
            .response
            .on_disabled_hover_text("No projects opened or saved yet");
        });
        if clear {
            self.ui.recent_projects.clear();
            save_recent(&[]);
        }
        if let Some(path) = open {
            if path.exists() {
                self.open_project(path);
            } else {
                // Moved or deleted: drop it so the list stays useful.
                self.ui.recent_projects.retain(|p| *p != path);
                save_recent(&self.ui.recent_projects);
                self.ui.status = format!("{} no longer exists — removed from recent projects", path.display());
                self.push_toast(Tone::Warn, self.ui.status.clone(), None);
            }
        }
    }

    /// A save or open is in flight; a second one would race it.
    fn project_busy(&mut self) -> bool {
        if let Some(job) = &self.project_job {
            let what = if job.is_saving() { "Saving" } else { "Opening" };
            self.ui.status = format!("{what} {} — wait for it to finish", file_name(job.path()));
            return true;
        }
        false
    }

    /// Save to the project's file, or ask where (`save_as`, or never saved).
    pub(super) fn save_project(&mut self, save_as: bool) {
        if self.project_busy() {
            return;
        }
        let path = match (&self.project_path, save_as) {
            (Some(path), false) => path.clone(),
            _ => {
                let mut dialog = rfd::FileDialog::new()
                    .add_filter("ZestPaint project", &[project::EXTENSION])
                    .set_file_name(format!("{}.{}", self.ui.model_name, project::EXTENSION));
                if let Some(dir) = self.project_path.as_ref().and_then(|p| p.parent()) {
                    dialog = dialog.set_directory(dir);
                }
                let Some(path) = dialog.save_file() else { return };
                with_project_extension(path)
            }
        };

        let name = path.file_stem().unwrap_or_default().to_string_lossy().to_string();
        let snapshot = project::capture(
            &self.gpu,
            project::SceneRefs {
                name: name.clone(),
                models: &self.models,
                sets: &self.texture_sets,
                visibility: self.isolate_backup.as_deref(),
                active_set: self.active_set,
                camera: CameraState {
                    target: self.camera.target.to_array(),
                    yaw: self.camera.yaw,
                    pitch: self.camera.pitch,
                    distance: self.camera.distance,
                },
                display: DisplayState {
                    exposure: self.ui.exposure,
                    env_intensity: self.ui.env_intensity,
                    key_light: self.ui.key_light,
                    key_azimuth: self.ui.key_azimuth,
                    key_elevation: self.ui.key_elevation,
                    key_follow_camera: self.ui.key_follow_camera,
                    ao_strength: self.ui.ao_strength,
                    env_path: self.ui.env_path.clone(),
                },
                pose: self.pose_json(),
                studio: self.studio_json(),
            },
        );
        let snapshot = match snapshot {
            Ok(s) => s,
            Err(e) => {
                self.ui.status = "Save failed".into();
                self.push_toast(Tone::Error, "Save failed", Some(ToastAction::Details(format!("{e:#}"))));
                return;
            }
        };
        let job_path = path.clone();
        let handle = std::thread::spawn(move || project::write(snapshot, &job_path));
        self.ui.status = format!("Saving {}…", file_name(&path));
        // Adopt the name now: further saves go to the same file.
        self.project_path = Some(path.clone());
        self.ui.model_name = name;
        self.project_job = Some(ProjectJob::Saving { path, handle });
    }

    pub(super) fn open_project_dialog(&mut self) {
        if self.project_busy() {
            return;
        }
        if self.import.is_some() {
            self.ui.status = "A model import is running — wait for it to finish".into();
            return;
        }
        let Some(path) = rfd::FileDialog::new()
            .add_filter("ZestPaint project", &[project::EXTENSION])
            .pick_file()
        else {
            return;
        };
        self.open_project(path);
    }

    /// Start reading a project on a worker; `poll_project_job` adopts it.
    pub(super) fn open_project(&mut self, path: PathBuf) {
        if self.project_busy() {
            return;
        }
        let job_path = path.clone();
        let handle = std::thread::spawn(move || project::read(&job_path));
        self.ui.status = format!("Opening {}…", file_name(&path));
        self.project_job = Some(ProjectJob::Loading { path, handle });
    }

    /// Finish a save or open once its worker is done. Called every frame.
    pub(super) fn poll_project_job(&mut self) {
        if !self.project_job.as_ref().is_some_and(|j| j.is_finished()) {
            return;
        }
        let Some(job) = self.project_job.take() else { return };
        match job {
            ProjectJob::Saving { path, handle } => {
                match handle.join().unwrap_or_else(|_| Err(anyhow::anyhow!("save thread panicked"))) {
                    Ok(()) => {
                        self.remember_recent(&path);
                        let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                        self.ui.status = format!(
                            "Saved {} ({:.1} MB)",
                            file_name(&path),
                            size as f64 / (1024.0 * 1024.0)
                        );
                        let folder = path.parent().map(Path::to_path_buf);
                        self.push_toast(Tone::Ok, self.ui.status.clone(), folder.map(ToastAction::OpenFolder));
                    }
                    Err(e) => {
                        self.ui.status = "Save failed".into();
                        self.push_toast(
                            Tone::Error,
                            format!("Save failed: {}", file_name(&path)),
                            Some(ToastAction::Details(format!("{e:#}"))),
                        );
                    }
                }
            }
            ProjectJob::Loading { path, handle } => {
                let loaded = handle.join().unwrap_or_else(|_| Err(anyhow::anyhow!("open thread panicked")));
                if let Err(e) = loaded.and_then(|p| self.adopt_project(&path, p)) {
                    self.ui.status = "Open failed".into();
                    self.push_toast(
                        Tone::Error,
                        format!("Could not open {}", file_name(&path)),
                        Some(ToastAction::Details(format!("{e:#}"))),
                    );
                }
            }
        }
    }

    /// Replace the document with a loaded project: models, texture sets
    /// with every layer, bakes, camera and lighting.
    fn adopt_project(&mut self, path: &Path, loaded: LoadedProject) -> anyhow::Result<()> {
        // Sets carry their own resolution; older projects saved one size
        // for all, which `resolution()` reads back.
        let default = loaded.resolution().unwrap_or(PAINT_RESOLUTION);
        let restored = project::restore(&self.gpu, loaded, default)?;

        self.renderer.set_models(&restored.models);
        self.models = restored.models;
        self.adopt_pose_and_studio(restored.pose, restored.studio);
        self.texture_sets = restored.sets;
        self.active_set = restored.active_set;
        // Materials' surface sliders live on the models; the renderer was
        // given them with the models above.
        self.mesh_groups = None;
        self.paint_projection = None;
        self.selection.clear();
        self.isolate_backup = None;
        self.isolate_camera = None;
        self.fill_lock = None;
        self.clone_source = None;
        self.clone_offset = None;
        self.pt_scene_stale = true;

        let c = restored.camera;
        self.camera.target = glam::Vec3::from(c.target);
        self.camera.yaw = c.yaw;
        self.camera.pitch = c.pitch;
        self.camera.distance = c.distance;

        let d = restored.display;
        self.ui.exposure = d.exposure;
        self.ui.env_intensity = d.env_intensity;
        self.ui.key_light = d.key_light;
        self.ui.key_azimuth = d.key_azimuth;
        self.ui.key_elevation = d.key_elevation;
        self.ui.key_follow_camera = d.key_follow_camera;
        self.ui.ao_strength = d.ao_strength;
        let mut note = String::new();
        match d.env_path {
            Some(env) if env.exists() => self.load_hdri(&env),
            Some(env) => note = format!(" — environment {} not found", env.display()),
            None => {
                if self.ui.env_path.is_some() {
                    self.renderer.set_environment(crate::environment::Environment::procedural(&self.gpu));
                    self.ui.env_path = None;
                    self.ui.env_name = "Procedural sky".into();
                }
            }
        }

        self.ui.model_name = restored.name;
        self.ui.export_name.clear();
        self.ui.default_scene = false;
        self.project_path = Some(path.to_path_buf());
        // The project keeps its own visibility.
        self.model_source = None;
        self.remember_recent(path);
        // A new document: the old history doesn't apply to it.
        self.reset_history();
        let layers: usize = self.texture_sets.iter().map(|s| s.layers.layers.len()).sum();
        self.ui.status = format!(
            "Opened {} ({} texture sets, {layers} layers){note}",
            file_name(path),
            self.texture_sets.len()
        );
        self.push_toast(Tone::Ok, self.ui.status.clone(), None);
        Ok(())
    }

    /// Export one layer of the active set — its maps, its mask, or both —
    /// to a folder the user picks.
    pub(super) fn export_layer_dialog(&mut self, index: usize, channels: bool, mask: bool) {
        let set = &self.texture_sets[self.active_set];
        let Some(layer) = set.layers.layers.get(index) else { return };
        if mask && !channels && layer.mask.is_none() {
            self.ui.status = format!("'{}' has no mask to export", layer.name);
            return;
        }
        let mut dialog = rfd::FileDialog::new();
        if !self.ui.export_dir.trim().is_empty() {
            dialog = dialog.set_directory(self.ui.export_dir.trim());
        }
        let Some(dir) = dialog.pick_folder() else { return };
        self.ui.export_dir = dir.display().to_string();
        let name = layer.name.clone();
        // Every map the layer has: one layer, so no reason to leave any out.
        let maps = crate::layers::MapKind::ALL;
        // Named like a full export of every set, so a layer exported on its
        // own sits beside the rest without clashing.
        let all = vec![true; self.texture_sets.len()];
        let prefixes = crate::export::set_prefixes(&self.texture_sets, &all, &self.export_name());
        let prefix = &prefixes[self.active_set];
        match crate::export::export_layer(&self.gpu, set, prefix, index, &maps, channels, mask, &dir) {
            Ok(files) if files.is_empty() => {
                self.ui.status = format!("'{name}' has nothing painted to export yet");
                self.push_toast(Tone::Warn, self.ui.status.clone(), None);
            }
            Ok(files) => {
                self.ui.status = format!("Exported '{name}' ({} files) to {}", files.len(), dir.display());
                self.push_toast(Tone::Ok, self.ui.status.clone(), Some(ToastAction::OpenFolder(dir)));
            }
            Err(e) => {
                self.ui.status = "Export failed".into();
                self.push_toast(Tone::Error, "Export failed", Some(ToastAction::Details(format!("{e:#}"))));
            }
        }
    }

    /// Replace the active layer's mask with a greyscale image — the way back
    /// in for a mask exported and edited elsewhere.
    pub(super) fn load_mask_from_image(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Images", &["png", "jpg", "jpeg", "tga", "bmp", "webp", "tif", "tiff"])
            .pick_file()
        else {
            return;
        };
        match crate::library::load_map(&path) {
            Ok(image) => {
                let layers = &mut self.texture_sets[self.active_set].layers;
                let active = layers.active;
                layers.set_mask_from_image(active, &image);
                self.ui.status =
                    format!("Mask of '{}' loaded from {}", layers.active_layer().name, file_name(&path));
                self.ui.props_tab = super::PropsTab::Mask;
            }
            Err(e) => self.ui.status = format!("Mask load failed: {e:#}"),
        }
    }
}
