//! The Studio window: pose a rigged avatar (built-in photography poses,
//! frames of Mixamo / Quaternius / CMU animations, saved poses, small
//! adjustments per bone), light it (classic setups of soft lights), put it
//! on a set (floor or sweep, environment / colour / transparent
//! background), choose a lens, and render a still to save, with a
//! transparent background when it is going to be composited later.
//!
//! The viewport previews all of it in path-traced mode; a render traces the
//! same scene offscreen at the chosen size, a batch of samples a frame.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use egui::{RichText, Ui};
use glam::{Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyImageToBufferInfo, PrimaryCommandBufferAbstract as _};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{ImageCreateInfo, ImageType, ImageUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::sync::GpuFuture;

use super::widgets::{self, Tone};
use super::AppState;
use crate::renderer::RenderParams;
use crate::pathtracer::{Accumulator, Frame, Output};
use crate::poses::PoseSource;
use crate::skeleton::{self, BONES, HumanMap, HumanPose, Skeleton, Tweaks};
use crate::studio::{self, Backdrop, Background, Lens, Studio, Subject};
use crate::{icons, theme};

/// Paths traced per frame while rendering, so the window stays responsive.
const PATHS_PER_FRAME: u64 = 4_000_000;
/// The Studio window's content width.
const STUDIO_WIDTH: f32 = 340.0;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(super) enum StudioTab {
    #[default]
    Pose,
    Lights,
    Set,
    Camera,
    Render,
}

/// The pose, as projects save it.
#[derive(Serialize, Deserialize, Default, Clone)]
#[serde(default)]
struct SavedPose {
    pose: Option<HumanPose>,
    mirror: bool,
    tweaks: Tweaks,
}

#[derive(Default)]
pub(super) struct PoseState {
    current: Option<HumanPose>,
    mirror: bool,
    tweaks: Tweaks,
    /// Animation files read this session.
    sources: Vec<PoseSource>,
    loading: Option<(PathBuf, std::thread::JoinHandle<anyhow::Result<PoseSource>>)>,
    /// Picked frame: (source, clip, frame).
    scrub: Option<(usize, usize, usize)>,
    /// Built-in poses (no file) and saved ones.
    library: Vec<(Option<PathBuf>, HumanPose)>,
    library_loaded: bool,
    save_name: String,
    tweak_bone: usize,
    /// The skeleton the models share and its humanoid bones.
    rig: Option<(Arc<Skeleton>, HumanMap)>,
}

pub(super) struct RenderJob {
    accum: Accumulator,
    target: Arc<ImageView>,
    texture: Option<egui::TextureId>,
    width: u32,
    height: u32,
    samples: u32,
    view_proj: Mat4,
    camera_pos: Vec3,
    block: [[f32; 4]; studio::BLOCK_VEC4S],
    bounces: u32,
    exposure: f32,
    started: Instant,
    transparent: bool,
}

pub(super) struct RenderResult {
    image: image::RgbaImage,
    texture: egui::TextureHandle,
    seconds: f32,
    samples: u32,
    transparent: bool,
    saved_to: Option<PathBuf>,
}

impl AppState {
    // --- rig and pose ----------------------------------------------------

    /// The skeleton the models' skins share, and its humanoid bones.
    pub(super) fn rig(&mut self) -> Option<(Arc<Skeleton>, HumanMap)> {
        let skeleton = self.models.iter().find_map(|m| m.skin.as_ref()).map(|s| s.skeleton.clone())?;
        if !self.pose.rig.as_ref().is_some_and(|(s, _)| Arc::ptr_eq(s, &skeleton)) {
            let map = skeleton::humanoid_map(&skeleton);
            self.pose.rig = Some((skeleton, map));
        }
        self.pose.rig.clone()
    }

    /// Put the models in the current pose (or back to rest).
    pub(super) fn apply_pose(&mut self) {
        let Some((skeleton, map)) = self.rig() else { return };
        let posed = self.pose.current.is_some() || !self.pose.tweaks.is_empty();
        let world = posed.then(|| {
            let locals = skeleton::retarget(self.pose.current.as_ref(), &skeleton, &map, self.pose.mirror, &self.pose.tweaks);
            Arc::new(skeleton.world(&locals))
        });
        let mut moved = Vec::new();
        for (i, m) in self.models.iter_mut().enumerate() {
            if let Some(skin) = &mut m.skin
                && Arc::ptr_eq(&skin.skeleton, &skeleton)
            {
                skin.pose = world.clone();
                moved.push(i);
            }
        }
        // Bending a few hundred thousand vertices: spread over the cores.
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).max(1);
        let chunk = self.models.len().div_ceil(threads).max(1);
        std::thread::scope(|scope| {
            for part in self.models.chunks_mut(chunk) {
                scope.spawn(move || {
                    for m in part.iter_mut().filter(|m| m.skin.is_some()) {
                        m.deform();
                    }
                });
            }
        });
        for i in moved {
            self.renderer.update_vertices(i, &self.models[i].mesh.vertices);
        }
        self.pt_scene_stale = true;
        self.mesh_groups = None;
        self.paint_projection = None;
    }

    fn set_pose(&mut self, pose: Option<HumanPose>) {
        self.pose.current = pose;
        self.apply_pose();
    }

    /// The pose for a project to save.
    pub(super) fn pose_json(&self) -> Option<serde_json::Value> {
        let saved = SavedPose { pose: self.pose.current.clone(), mirror: self.pose.mirror, tweaks: self.pose.tweaks.clone() };
        (saved.pose.is_some() || !saved.tweaks.is_empty()).then(|| serde_json::to_value(saved).ok()).flatten()
    }

    pub(super) fn studio_json(&self) -> Option<serde_json::Value> {
        serde_json::to_value(&self.studio).ok()
    }

    /// Take a loaded project's pose and studio.
    pub(super) fn adopt_pose_and_studio(&mut self, pose: Option<serde_json::Value>, studio: Option<serde_json::Value>) {
        let saved: SavedPose = pose.and_then(|v| serde_json::from_value(v).ok()).unwrap_or_default();
        self.pose.current = saved.pose;
        self.pose.mirror = saved.mirror;
        self.pose.tweaks = saved.tweaks;
        self.pose.scrub = None;
        self.pose.rig = None;
        self.studio = studio.and_then(|v| serde_json::from_value(v).ok()).unwrap_or_default();
        self.apply_pose();
    }

    /// A new model: no pose carried over.
    pub(super) fn reset_pose(&mut self) {
        self.pose.current = None;
        self.pose.tweaks.clear();
        self.pose.mirror = false;
        self.pose.scrub = None;
        self.pose.rig = None;
    }

    fn poll_pose_import(&mut self) {
        if !self.pose.loading.as_ref().is_some_and(|(_, h)| h.is_finished()) {
            return;
        }
        let (path, handle) = self.pose.loading.take().unwrap();
        let name = path.file_name().unwrap_or_default().to_string_lossy().into_owned();
        match handle.join() {
            Ok(Ok(source)) => {
                let clips = source.clips.len();
                self.ui.status = format!(
                    "{name}: {clips} clip{}, {}/22 body bones",
                    if clips == 1 { "" } else { "s" },
                    source.body_bones()
                );
                self.pose.sources.push(source);
                let s = self.pose.sources.len() - 1;
                self.pose.scrub = Some((s, 0, 0));
                let pose = self.pose.sources[s].pose(0, 0);
                self.set_pose(pose);
            }
            Ok(Err(e)) => self.push_toast(Tone::Error, format!("Could not read {name}"), Some(super::overlays::ToastAction::Details(format!("{e:#}")))),
            Err(_) => self.push_toast(Tone::Error, format!("Reading {name} crashed"), None),
        }
    }

    fn import_pose_file(&mut self) {
        if self.pose.loading.is_some() {
            return;
        }
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Animations and poses", &["fbx", "glb", "gltf", "bvh"])
            .pick_file()
        else {
            return;
        };
        let job = path.clone();
        self.pose.loading = Some((path, std::thread::spawn(move || crate::poses::load(&job))));
    }

    fn reload_pose_library(&mut self) {
        self.pose.library = crate::poses::builtin().into_iter().map(|p| (None, p)).collect();
        self.pose.library.extend(crate::poses::saved().into_iter().map(|(path, p)| (Some(path), p)));
        self.pose.library_loaded = true;
    }

    /// The current pose of the avatar, as a rig-neutral pose to save.
    fn sample_current_pose(&mut self, name: &str) -> Option<HumanPose> {
        let (skeleton, map) = self.rig()?;
        let locals = skeleton::retarget(self.pose.current.as_ref(), &skeleton, &map, self.pose.mirror, &self.pose.tweaks);
        Some(HumanPose::sample(name, "saved", &map, &skeleton.rest_world(), &skeleton.world(&locals)))
    }

    // --- the subject and the frame ---------------------------------------------

    /// Where the visible models stand, and which way the rig faces.
    pub(super) fn subject(&mut self) -> Subject {
        let sets = crate::scene::set_indices(&self.models);
        let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for (m, set) in self.models.iter().zip(sets) {
            if !self.texture_sets.get(set).is_some_and(|s| s.visible) {
                continue;
            }
            let (a, b) = m.mesh.bounds();
            for corner in [a, b, Vec3::new(a.x, b.y, a.z), Vec3::new(b.x, a.y, b.z)] {
                let p = m.transform.transform_point3(corner);
                lo = lo.min(p);
                hi = hi.max(p);
            }
        }
        if lo.x > hi.x {
            (lo, hi) = (Vec3::splat(-1.0), Vec3::splat(1.0));
        }
        let center = (lo + hi) * 0.5;
        let mut subject = Subject { center, floor: lo.y, facing: Quat::IDENTITY, seat: Vec3::new(center.x, 0.45, center.z) };
        if let Some((skeleton, map)) = self.rig() {
            subject.facing = skeleton::rig_facing(&skeleton, &map);
            // The seat goes under the posed hips, up to the thighs.
            let posed = self.models.iter().find_map(|m| m.skin.as_ref().and_then(|s| s.pose.clone()));
            let world = posed.map_or_else(|| skeleton.rest_world(), |w| w.to_vec());
            if let Some(h) = map[skeleton::HIPS] {
                let hips = world[h].w_axis.truncate();
                let thigh = [14usize, 18].iter().filter_map(|&b| map[b]).map(|j| world[j].w_axis.y).fold(hips.y, f32::min);
                subject.seat = Vec3::new(hips.x, (thigh - lo.y - 0.07).max(0.1), hips.z);
            }
        }
        subject
    }

    fn lens(&self, fov_y: f32) -> Lens {
        let view = self.camera.view_matrix();
        let right = Vec3::new(view.x_axis.x, view.y_axis.x, view.z_axis.x);
        let up = Vec3::new(view.x_axis.y, view.y_axis.y, view.z_axis.y);
        let forward = (self.camera.target - self.camera.position()).normalize_or(-Vec3::Z);
        Lens { position: self.camera.position(), right, up, forward, fov_y }
    }

    /// Path-trace the viewport (or step a running render), after `before`.
    pub(super) fn path_trace(&mut self, before: Box<dyn GpuFuture>, params: &RenderParams) -> Box<dyn GpuFuture> {
        if self.pt_scene_stale || !self.pathtracer.has_scene() {
            self.pathtracer.rebuild_scene(&self.renderer, &self.texture_sets, &self.models);
            self.pt_subject = self.subject();
            self.pt_scene_stale = false;
        }
        if let Some(mut job) = self.render_job.take() {
            // A render has the GPU to itself; the viewport waits.
            let left = job.samples.saturating_sub(job.accum.samples);
            let per_frame = (PATHS_PER_FRAME / (job.width as u64 * job.height as u64).max(1)).clamp(1, 32) as u32;
            let frame = Frame { view_proj: job.view_proj, camera_pos: job.camera_pos, studio: &job.block, bounces: job.bounces, exposure: job.exposure };
            let future = self.pathtracer.render_into(
                &mut job.accum,
                before,
                job.target.clone(),
                Output::File,
                &frame,
                per_frame.min(left.max(1)),
                &mut self.texture_sets,
                &self.renderer,
            );
            self.render_job = Some(job);
            return future;
        }
        let fov = self.camera.fov_y;
        let block = studio::gpu_block(&self.studio, &self.pt_subject, params.env_intensity, &self.lens(fov));
        let reset = params.view_proj != self.pt_last_view
            || block != self.pt_last_block
            || self.texture_sets.iter().any(|s| s.visible && s.layers.is_dirty());
        self.pt_last_view = params.view_proj;
        self.pt_last_block = block;
        let frame = Frame { view_proj: params.view_proj, camera_pos: params.camera_pos, studio: &block, bounces: self.studio.bounces, exposure: params.exposure };
        self.pathtracer.render(before, self.viewport.scene_image.clone(), &frame, reset, &mut self.texture_sets, &self.renderer)
    }

    /// Field of view and view-projection for an image of `aspect`, showing
    /// what the frame guide in the viewport outlines.
    fn render_camera(&self, aspect: f32) -> (f32, Mat4) {
        let vp_aspect = self.viewport.size[0] as f32 / self.viewport.size[1].max(1) as f32;
        let fov = if aspect > vp_aspect {
            2.0 * ((self.camera.fov_y * 0.5).tan() * vp_aspect / aspect).atan()
        } else {
            self.camera.fov_y
        };
        let near = (self.camera.distance * 0.01).max(0.001);
        let far = self.camera.distance * 100.0;
        let proj = glam::camera::rh::proj::vulkan::perspective(fov, aspect, near, far);
        (fov, proj * self.camera.view_matrix())
    }

    pub(super) fn start_render(&mut self) {
        if self.render_job.is_some() {
            return;
        }
        let (w, h) = (self.studio.render_width.clamp(16, 8192), self.studio.render_height.clamp(16, 8192));
        let target = ImageView::new_default(
            crate::vram::image(
                self.gpu.memory_allocator.clone(),
                ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format: Format::R8G8B8A8_UNORM,
                    extent: [w, h, 1],
                    usage: ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC | ImageUsage::SAMPLED,
                    ..Default::default()
                },
                crate::vram::Kind::Viewport,
            )
            .expect("render target"),
        )
        .unwrap();
        self.pt_scene_stale = true;
        if self.pt_scene_stale || !self.pathtracer.has_scene() {
            self.pathtracer.rebuild_scene(&self.renderer, &self.texture_sets, &self.models);
            self.pt_subject = self.subject();
            self.pt_scene_stale = false;
        }
        let (fov, view_proj) = self.render_camera(w as f32 / h as f32);
        let block = studio::gpu_block(&self.studio, &self.pt_subject, self.ui.env_intensity, &self.lens(fov));
        self.render_job = Some(RenderJob {
            accum: Accumulator::default(),
            target,
            texture: None,
            width: w,
            height: h,
            samples: self.studio.render_samples.clamp(1, 16384),
            view_proj,
            camera_pos: self.camera.position(),
            block,
            bounces: self.studio.bounces,
            exposure: self.ui.exposure,
            started: Instant::now(),
            transparent: self.studio.background == Background::Transparent,
        });
        self.ui.status = format!("Rendering {w} × {h}…");
    }

    /// Finish a render whose samples are all in: read it back.
    fn poll_render(&mut self, gui: &mut egui_winit_vulkano::Gui) {
        let Some(job) = &self.render_job else { return };
        if job.accum.samples < job.samples {
            return;
        }
        let job = self.render_job.take().unwrap();
        if let Some(id) = job.texture {
            gui.unregister_user_image(id);
        }
        let seconds = job.started.elapsed().as_secs_f32();
        match read_image(&self.gpu, &job.target, job.width, job.height) {
            Ok(image) => {
                let size = [image.width() as usize, image.height() as usize];
                let colour = egui::ColorImage::from_rgba_unmultiplied(size, image.as_raw());
                let texture = gui.context().load_texture("render_result", colour, egui::TextureOptions::LINEAR);
                self.ui.status = format!("Rendered {} × {} in {seconds:.1} s", job.width, job.height);
                self.render_result = Some(RenderResult { image, texture, seconds, samples: job.samples, transparent: job.transparent, saved_to: None });
                self.ui.render_result_open = true;
            }
            Err(e) => self.push_toast(Tone::Error, "Render failed", Some(super::overlays::ToastAction::Details(format!("{e:#}")))),
        }
    }

    fn save_render(&mut self) {
        let Some(result) = &mut self.render_result else { return };
        let mut dialog = rfd::FileDialog::new().add_filter("PNG image", &["png"]).set_file_name(format!("{}_render.png", self.ui.model_name));
        if let Some(dir) = self.project_path.as_ref().and_then(|p| p.parent()) {
            dialog = dialog.set_directory(dir);
        }
        let Some(mut path) = dialog.save_file() else { return };
        if path.extension().is_none() {
            path.set_extension("png");
        }
        match result.image.save(&path) {
            Ok(()) => {
                self.ui.status = format!("Saved {}", path.display());
                result.saved_to = Some(path);
            }
            Err(e) => self.push_toast(Tone::Error, "Could not save the render", Some(super::overlays::ToastAction::Details(e.to_string()))),
        }
    }

    // --- UI ------------------------------------------------------------------

    pub(super) fn studio_ui(&mut self, gui: &mut egui_winit_vulkano::Gui) {
        self.poll_pose_import();
        self.poll_render(gui);
        let ctx = gui.context();
        if let Some(job) = &mut self.render_job
            && job.texture.is_none()
        {
            job.texture = Some(gui.register_user_image_view(job.target.clone(), Default::default()));
        }
        if self.ui.studio_open {
            let mut open = true;
            egui::Window::new(format!("{}  Studio", icons::ENVIRONMENT))
                .id("studio".into())
                .open(&mut open)
                .default_pos(ctx.screen_rect().right_top() + egui::vec2(-720.0, 60.0))
                // Fixed width: rows that fill the width would otherwise
                // widen the window a little each frame while dragging.
                .resizable(false)
                .show(&ctx, |ui| {
                    ui.set_width(STUDIO_WIDTH);
                    let mut tab = self.ui.studio_tab;
                    widgets::segmented(
                        ui,
                        &mut tab,
                        &[
                            (StudioTab::Pose, "Pose"),
                            (StudioTab::Lights, "Lights"),
                            (StudioTab::Set, "Set"),
                            (StudioTab::Camera, "Camera"),
                            (StudioTab::Render, "Render"),
                        ],
                    );
                    self.ui.studio_tab = tab;
                    if !self.pt_mode {
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("The viewport is in raster mode.").color(theme::TEXT_MUTED));
                            if ui.button("Preview path traced").clicked() {
                                self.pt_mode = true;
                                self.pt_scene_stale = true;
                            }
                        });
                    }
                    if self.studio == Studio::default() {
                        if ui
                            .add(widgets::accent_outline_button("Set up a photo studio"))
                            .on_hover_text("Three-point lighting on a grey studio sweep; change anything after")
                            .clicked()
                        {
                            self.studio = Studio::photo_default();
                        }
                    }
                    widgets::divider(ui);
                    egui::ScrollArea::vertical().id_salt("studio_scroll").max_height(ctx.screen_rect().height() * 0.7).auto_shrink([false, true]).show(ui, |ui| {
                        ui.set_width(STUDIO_WIDTH - 14.0);
                        match tab {
                            StudioTab::Pose => self.pose_tab(ui),
                            StudioTab::Lights => self.lights_tab(ui),
                            StudioTab::Set => self.set_tab(ui),
                            StudioTab::Camera => self.camera_tab(ui),
                            StudioTab::Render => self.render_tab(ui),
                        }
                    });
                });
            self.ui.studio_open = open;
        }
        self.render_progress_window(&ctx);
        self.render_result_window(&ctx);
    }

    fn pose_tab(&mut self, ui: &mut Ui) {
        if !self.pose.library_loaded {
            self.reload_pose_library();
        }
        let Some((skeleton, map)) = self.rig() else {
            ui.label(RichText::new("This model has no skeleton, so it can't be posed.").color(theme::TEXT_MUTED));
            if self.project_path.is_some() {
                ui.label(
                    RichText::new("Projects saved before ZestSurface kept bones can take them from the model file; the painting stays.")
                        .color(theme::TEXT_DIM),
                );
                if let Some(job) = &self.shape_import {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label(RichText::new(format!("Reading {}…", job.file_name())).color(theme::TEXT_MUTED));
                    });
                } else if ui.add(widgets::primary_button("Load bones from the model file…")).clicked() {
                    let mut dialog = rfd::FileDialog::new().add_filter("3D models", &["fbx", "glb", "gltf"]);
                    if let Some(dir) = self.project_path.as_ref().and_then(|p| p.parent()) {
                        dialog = dialog.set_directory(dir);
                    }
                    if let Some(path) = dialog.pick_file() {
                        self.start_shape_import(&path);
                    }
                }
            }
            return;
        };
        let body = map[..22].iter().filter(|b| b.is_some()).count();
        let fingers = map[22..].iter().filter(|b| b.is_some()).count();
        widgets::section(ui, "SKELETON", Some(&format!("{} bones", skeleton.joints.len())));
        ui.label(RichText::new(format!("Humanoid: {body}/22 body bones, {fingers}/30 finger bones")).color(theme::TEXT_MUTED));
        let missing: Vec<&str> = (0..22).filter(|&b| map[b].is_none() && b != 3).map(|b| BONES[b]).collect();
        if !missing.is_empty() {
            ui.label(RichText::new(format!("Not found: {}", missing.join(", "))).color(theme::WARN_FG).small());
        }

        ui.horizontal(|ui| {
            if ui.add(egui::Button::new("Rest pose")).on_hover_text("Back to the pose the model was made in").clicked() {
                self.pose.scrub = None;
                self.pose.tweaks.clear();
                self.set_pose(None);
            }
            let mut mirror = self.pose.mirror;
            if ui.checkbox(&mut mirror, "Mirror").on_hover_text("Swap left and right").changed() {
                self.pose.mirror = mirror;
                self.apply_pose();
            }
        });

        widgets::section(ui, "POSES", None);
        let current = self.pose.current.as_ref().map(|p| p.name.clone());
        let mut pick = None;
        let mut delete = None;
        egui::Grid::new("pose_library").num_columns(2).spacing([6.0, 4.0]).show(ui, |ui| {
            for (i, (path, pose)) in self.pose.library.iter().enumerate() {
                let on = current.as_deref() == Some(pose.name.as_str()) && self.pose.scrub.is_none();
                let r = ui.add_sized([150.0, 24.0], egui::Button::new(&pose.name).selected(on));
                if r.on_hover_text(&pose.source).clicked() {
                    pick = Some(i);
                }
                if path.is_some() && ui.small_button(icons::DELETE).on_hover_text("Delete this saved pose").clicked() {
                    delete = Some(i);
                }
                if i % 2 == 1 || path.is_some() {
                    ui.end_row();
                }
            }
        });
        if let Some(i) = pick {
            self.pose.scrub = None;
            let pose = self.pose.library[i].1.clone();
            self.set_pose(Some(pose));
        }
        if let Some(i) = delete
            && let Some(path) = &self.pose.library[i].0
        {
            let _ = std::fs::remove_file(path);
            self.reload_pose_library();
        }

        if !self.studio.seat
            && self.pose.current.as_ref().is_some_and(|p| p.name.to_lowercase().contains("sit"))
            && ui.button("Add a posing box to sit on").clicked()
        {
            self.studio.seat = true;
        }

        widgets::section(ui, "FROM ANIMATIONS", None);
        ui.horizontal(|ui| {
            if self.pose.loading.is_some() {
                ui.spinner();
                ui.label(RichText::new("Reading…").color(theme::TEXT_MUTED));
            } else if ui.button(format!("{}  Import animation…", icons::OPEN)).on_hover_text("FBX (Mixamo), glTF/GLB (Quaternius), BVH (CMU mocap)").clicked() {
                self.import_pose_file();
            }
        });
        let mut new_scrub = None;
        let mut drop_source = None;
        for (s, source) in self.pose.sources.iter().enumerate() {
            let (cur_clip, cur_frame) = match self.pose.scrub {
                Some((cs, c, f)) if cs == s => (c, f),
                _ => (0, 0),
            };
            ui.horizontal(|ui| {
                ui.label(RichText::new(source.file_name()).color(theme::TEXT_STRONG));
                if ui.small_button(icons::REMOVE).on_hover_text("Close this file").clicked() {
                    drop_source = Some(s);
                }
            });
            let mut clip = cur_clip;
            if source.clips.len() > 1 {
                egui::ComboBox::from_id_salt(("pose_clip", s))
                    .width(260.0)
                    .selected_text(&source.clips[clip.min(source.clips.len() - 1)].name)
                    .show_ui(ui, |ui| {
                        for (c, cl) in source.clips.iter().enumerate() {
                            ui.selectable_value(&mut clip, c, &cl.name);
                        }
                    });
            }
            let frames = source.clips.get(clip).map_or(1, |c| c.frames.len());
            let mut frame = if clip == cur_clip { cur_frame } else { 0 } as f32;
            if frames > 1 {
                let spf = source.clips[clip].seconds_per_frame;
                let r = ui.add(
                    egui::Slider::new(&mut frame, 0.0..=(frames - 1) as f32)
                        .step_by(1.0)
                        .custom_formatter(move |f, _| format!("{:.2} s", f as f32 * spf)),
                );
                if r.changed() || clip != cur_clip || self.pose.scrub.is_none_or(|(cs, _, _)| cs != s) && r.drag_started() {
                    new_scrub = Some((s, clip, frame as usize));
                }
            } else if clip != cur_clip {
                new_scrub = Some((s, clip, 0));
            }
            if self.pose.scrub.is_none_or(|(cs, _, _)| cs != s) && ui.small_button("Use this file's pose").clicked() {
                new_scrub = Some((s, clip, frame as usize));
            }
        }
        if let Some(s) = drop_source {
            self.pose.sources.remove(s);
            self.pose.scrub = None;
        }
        if let Some((s, c, f)) = new_scrub {
            self.pose.scrub = Some((s, c, f));
            let pose = self.pose.sources[s].pose(c, f);
            self.set_pose(pose);
        }
        ui.label(
            RichText::new(
                "Mixamo: pick an animation, download as FBX “Without Skin”. Quaternius’ Universal Animation Library (CC0) and CMU’s BVH mocap work too.",
            )
            .small()
            .color(theme::TEXT_DIM),
        );

        widgets::section(ui, "ADJUST", None);
        let body: Vec<usize> = (0..22).filter(|&b| map[b].is_some()).collect();
        if !body.contains(&self.pose.tweak_bone) {
            self.pose.tweak_bone = body.first().copied().unwrap_or(0);
        }
        let mut bone = self.pose.tweak_bone;
        egui::ComboBox::from_id_salt("tweak_bone").width(160.0).selected_text(BONES[bone]).show_ui(ui, |ui| {
            for &b in &body {
                let label = if self.pose.tweaks.contains_key(BONES[b]) { format!("{} •", BONES[b]) } else { BONES[b].to_string() };
                ui.selectable_value(&mut bone, b, label);
            }
        });
        self.pose.tweak_bone = bone;
        let mut t = self.pose.tweaks.get(BONES[bone]).copied().unwrap_or([0.0; 3]);
        let mut changed = false;
        for (k, label) in ["Nod", "Turn", "Tilt"].iter().enumerate() {
            changed |= widgets::SliderRow::new(label, &mut t[k], -90.0..=90.0).decimals(0).suffix("°").show(ui).changed();
        }
        if changed {
            if t == [0.0; 3] {
                self.pose.tweaks.remove(BONES[bone]);
            } else {
                self.pose.tweaks.insert(BONES[bone].to_string(), t);
            }
            self.apply_pose();
        }
        if !self.pose.tweaks.is_empty() && ui.button("Reset adjustments").clicked() {
            self.pose.tweaks.clear();
            self.apply_pose();
        }

        widgets::section(ui, "SAVE POSE", None);
        ui.horizontal(|ui| {
            ui.add(egui::TextEdit::singleline(&mut self.pose.save_name).hint_text("Name").desired_width(180.0));
            let name = self.pose.save_name.trim().to_string();
            if ui.add_enabled(!name.is_empty(), egui::Button::new("Save")).on_hover_text("Into the pose library, with the adjustments").clicked()
                && let Some(pose) = self.sample_current_pose(&name)
            {
                match crate::poses::save(&pose) {
                    Ok(path) => {
                        self.ui.status = format!("Saved pose {}", path.display());
                        self.pose.save_name.clear();
                        self.reload_pose_library();
                    }
                    Err(e) => self.push_toast(Tone::Error, "Could not save the pose", Some(super::overlays::ToastAction::Details(format!("{e:#}")))),
                }
            }
        });
    }

    fn lights_tab(&mut self, ui: &mut Ui) {
        widgets::section(ui, "SETUPS", None);
        let mut chosen = None;
        egui::Grid::new("light_presets").num_columns(2).spacing([6.0, 4.0]).show(ui, |ui| {
            for (i, (name, about, _)) in studio::presets().iter().enumerate() {
                if ui.add_sized([150.0, 24.0], egui::Button::new(*name)).on_hover_text(*about).clicked() {
                    chosen = Some(i);
                }
                if i % 2 == 1 {
                    ui.end_row();
                }
            }
        });
        if let Some(i) = chosen {
            self.studio.lights = studio::presets().remove(i).2;
            if self.studio.environment > 0.5 && !self.studio.lights.is_empty() {
                // Studio lights over a full-strength sky wash out.
                self.studio.environment = 0.25;
            }
        }
        widgets::section(ui, "LIGHTS", Some(&format!("{}/{}", self.studio.lights.len(), studio::MAX_LIGHTS)));
        let mut remove = None;
        for (i, l) in self.studio.lights.iter_mut().enumerate() {
            let id = ui.make_persistent_id(("light", i));
            egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, false)
                .show_header(ui, |ui| {
                    ui.checkbox(&mut l.enabled, "");
                    ui.add(egui::TextEdit::singleline(&mut l.name).desired_width(110.0));
                    let mut rgb = l.color;
                    if ui.color_edit_button_rgb(&mut rgb).changed() {
                        l.color = rgb;
                    }
                    if ui.small_button(icons::DELETE).on_hover_text("Remove this light").clicked() {
                        remove = Some(i);
                    }
                })
                .body(|ui| {
                    widgets::SliderRow::new("Around", &mut l.azimuth, -180.0..=180.0)
                        .decimals(0)
                        .suffix("°")
                        .hint("0° in front of the subject, 90° at their left, ±180° behind")
                        .show(ui);
                    widgets::SliderRow::new("Height", &mut l.elevation, -60.0..=89.0).decimals(0).suffix("°").show(ui);
                    widgets::SliderRow::new("Distance", &mut l.distance, 0.4..=10.0).decimals(1).suffix(" m").show(ui);
                    widgets::SliderRow::new("Size", &mut l.size, 0.02..=2.0).decimals(2).suffix(" m").hint("Bigger lights give softer shadows").show(ui);
                    widgets::SliderRow::new("Strength", &mut l.intensity, 0.0..=6.0).decimals(2).show(ui);
                });
        }
        if let Some(i) = remove {
            self.studio.lights.remove(i);
        }
        if self.studio.lights.len() < studio::MAX_LIGHTS && ui.button(format!("{}  Add light", icons::ADD)).clicked() {
            let n = self.studio.lights.len() + 1;
            self.studio.lights.push(studio::Light { name: format!("Light {n}"), ..Default::default() });
        }
        widgets::section(ui, "ENVIRONMENT", None);
        widgets::SliderRow::new("Strength", &mut self.studio.environment, 0.0..=2.0)
            .decimals(2)
            .hint("Light from the environment (HDRI or sky), on top of the viewport's environment intensity")
            .show(ui);
        widgets::SliderRow::new("Rotation", &mut self.studio.environment_rotation, -180.0..=180.0).decimals(0).suffix("°").show(ui);
    }

    fn set_tab(&mut self, ui: &mut Ui) {
        widgets::section(ui, "BACKDROP", None);
        let mut backdrop = self.studio.backdrop;
        widgets::segmented(ui, &mut backdrop, &Backdrop::ALL.map(|b| (b, b.label())));
        self.studio.backdrop = backdrop;
        if backdrop != Backdrop::None {
            ui.horizontal(|ui| {
                ui.label(RichText::new("Colour").color(theme::TEXT_MUTED));
                ui.color_edit_button_rgb(&mut self.studio.backdrop_color);
            });
            widgets::SliderRow::new("Roughness", &mut self.studio.backdrop_roughness, 0.03..=1.0).decimals(2).hint("Low for a glossy floor that reflects").show(ui);
            if backdrop == Backdrop::Sweep {
                widgets::SliderRow::new("Wall behind", &mut self.studio.backdrop_distance, 0.3..=8.0).decimals(1).suffix(" m").show(ui);
                widgets::SliderRow::new("Curve", &mut self.studio.backdrop_curve, 0.0..=3.0).decimals(1).suffix(" m").show(ui);
            }
        }
        ui.checkbox(&mut self.studio.seat, "Posing box").on_hover_text("A box to sit on, under the hips");
        if self.studio.seat {
            let mut auto = self.studio.seat_height <= 0.0;
            if ui.checkbox(&mut auto, "Height fits the pose").changed() {
                self.studio.seat_height = if auto { 0.0 } else { self.pt_subject.seat.y.max(0.1) };
            }
            if !auto {
                widgets::SliderRow::new("Height", &mut self.studio.seat_height, 0.1..=1.2).decimals(2).suffix(" m").show(ui);
            }
            widgets::SliderRow::new("Back", &mut self.studio.seat_back, -0.3..=0.4).decimals(2).suffix(" m").hint("How far behind the hips the box's middle is").show(ui);
        }
        widgets::section(ui, "BACKGROUND", None);
        let mut bg = self.studio.background;
        egui::ComboBox::from_id_salt("studio_bg").width(220.0).selected_text(bg.label()).show_ui(ui, |ui| {
            for option in Background::ALL {
                ui.selectable_value(&mut bg, option, option.label());
            }
        });
        self.studio.background = bg;
        match bg {
            Background::Color => {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Colour").color(theme::TEXT_MUTED));
                    ui.color_edit_button_rgb(&mut self.studio.background_color);
                    if ui.button("Chroma green").on_hover_text("#00B140, for keying").clicked() {
                        self.studio.background_color = [0.0, 0.694, 0.251];
                    }
                    if ui.button("Chroma blue").on_hover_text("#0047BB").clicked() {
                        self.studio.background_color = [0.0, 0.278, 0.733];
                    }
                });
                if backdrop != Backdrop::None {
                    ui.label(RichText::new("The backdrop stands in front of the colour; set it to None to see only the colour.").small().color(theme::TEXT_DIM));
                }
            }
            Background::Transparent => {
                ui.label(
                    RichText::new("Renders save as PNG with alpha: only the avatar, to place over any background later. The viewport shows a checkerboard behind it.")
                        .small()
                        .color(theme::TEXT_DIM),
                );
                if backdrop != Backdrop::None {
                    ui.checkbox(&mut self.studio.shadow_catcher, "Keep shadows on the backdrop")
                        .on_hover_text("The backdrop is invisible, but the avatar's shadows on it stay in the alpha (a shadow catcher)");
                    ui.label(RichText::new("The backdrop still bounces light onto the avatar.").small().color(theme::TEXT_DIM));
                }
            }
            Background::Environment => {}
        }
    }

    fn camera_tab(&mut self, ui: &mut Ui) {
        widgets::section(ui, "LENS", None);
        let mut mm = studio::focal_length(self.camera.fov_y);
        if widgets::SliderRow::new("Focal length", &mut mm, 14.0..=200.0)
            .decimals(0)
            .suffix(" mm")
            .log()
            .hint("Full-frame equivalent. 85–135 mm flatters portraits; step back to frame")
            .show(ui)
            .changed()
        {
            self.camera.fov_y = studio::fov_for_focal_length(mm);
        }
        ui.horizontal(|ui| {
            for preset in [24.0, 35.0, 50.0, 85.0, 135.0] {
                if ui.small_button(format!("{preset:.0}")).clicked() {
                    // Keep the subject the same size: move back as the lens lengthens.
                    let before = (self.camera.fov_y * 0.5).tan();
                    self.camera.fov_y = studio::fov_for_focal_length(preset);
                    self.camera.distance *= before / (self.camera.fov_y * 0.5).tan();
                }
            }
        });
        ui.checkbox(&mut self.studio.depth_of_field, "Depth of field");
        if self.studio.depth_of_field {
            widgets::SliderRow::new("f-stop", &mut self.studio.f_stop, 0.95..=22.0).decimals(1).log().hint("Smaller numbers blur more").show(ui);
            widgets::SliderRow::new("Focus", &mut self.studio.focus_offset, -3.0..=3.0)
                .decimals(2)
                .suffix(" m")
                .hint("Past (+) or before (−) the middle of the subject")
                .show(ui);
        }
        widgets::section(ui, "EXPOSURE", None);
        widgets::SliderRow::new("Exposure", &mut self.ui.exposure, 0.1..=4.0).decimals(2).log().show(ui);
        let mut bounces = self.studio.bounces as f32;
        if widgets::SliderRow::new("Bounces", &mut bounces, 1.0..=8.0).decimals(0).hint("More: brighter shadows and deeper sheer layers, slower").show(ui).changed() {
            self.studio.bounces = bounces as u32;
        }
        if ui.button("Frame the subject").clicked() {
            let s = self.subject();
            self.camera.target = s.center;
        }
    }

    fn render_tab(&mut self, ui: &mut Ui) {
        widgets::section(ui, "IMAGE", None);
        let presets: [(&str, u32, u32); 6] = [
            ("1920 × 1080", 1920, 1080),
            ("1080 × 1920 (portrait)", 1080, 1920),
            ("2048 × 2048", 2048, 2048),
            ("1080 × 1350 (4:5)", 1080, 1350),
            ("3840 × 2160 (4K)", 3840, 2160),
            ("2160 × 3840 (4K portrait)", 2160, 3840),
        ];
        let (w, h) = (self.studio.render_width, self.studio.render_height);
        let label = presets.iter().find(|p| (p.1, p.2) == (w, h)).map_or(format!("{w} × {h}"), |p| p.0.to_string());
        egui::ComboBox::from_id_salt("render_size").width(220.0).selected_text(label).show_ui(ui, |ui| {
            for (name, pw, ph) in presets {
                if ui.selectable_label((pw, ph) == (w, h), name).clicked() {
                    self.studio.render_width = pw;
                    self.studio.render_height = ph;
                }
            }
            if ui.selectable_label(false, "Viewport shape").clicked() {
                let [vw, vh] = self.viewport.size;
                let scale = 1920.0 / vw.max(vh).max(1) as f32;
                self.studio.render_width = (vw as f32 * scale) as u32;
                self.studio.render_height = (vh as f32 * scale) as u32;
            }
        });
        ui.horizontal(|ui| {
            ui.add(egui::DragValue::new(&mut self.studio.render_width).range(16..=8192).suffix(" px"));
            ui.label("×");
            ui.add(egui::DragValue::new(&mut self.studio.render_height).range(16..=8192).suffix(" px"));
        });
        let mut samples = self.studio.render_samples as f32;
        if widgets::SliderRow::new("Samples", &mut samples, 16.0..=4096.0)
            .decimals(0)
            .log()
            .hint("Per pixel. 256 is clean for most; more for depth of field and glossy floors")
            .show(ui)
            .changed()
        {
            self.studio.render_samples = samples as u32;
        }
        ui.label(
            RichText::new(match self.studio.background {
                Background::Transparent => "Background: transparent (PNG with alpha)",
                Background::Color => "Background: colour",
                Background::Environment => "Background: environment",
            })
            .color(theme::TEXT_MUTED),
        );
        ui.label(RichText::new("The viewport outlines what the image will show.").small().color(theme::TEXT_DIM));
        ui.add_space(6.0);
        let busy = self.render_job.is_some();
        if ui.add_enabled(!busy, widgets::primary_button(format!("{}  Render", icons::IMAGE)).min_size(egui::vec2(160.0, 30.0))).clicked() {
            self.start_render();
        }
        if self.render_result.is_some() && ui.button("Show last render").clicked() {
            self.ui.render_result_open = true;
        }
    }

    fn render_progress_window(&mut self, ctx: &egui::Context) {
        let Some(job) = &self.render_job else { return };
        let mut cancel = false;
        let done = job.accum.samples.min(job.samples);
        let fraction = done as f32 / job.samples.max(1) as f32;
        let elapsed = job.started.elapsed().as_secs_f32();
        let eta = if done > 0 { elapsed / fraction - elapsed } else { 0.0 };
        egui::Window::new("Rendering").collapsible(false).resizable(false).anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0]).show(ctx, |ui| {
            ui.set_width(360.0);
            if let Some(id) = job.texture {
                let aspect = job.width as f32 / job.height as f32;
                let size = if aspect >= 1.0 { egui::vec2(360.0, 360.0 / aspect) } else { egui::vec2(260.0 * aspect, 260.0) };
                ui.vertical_centered(|ui| ui.image(egui::load::SizedTexture::new(id, size)));
            }
            ui.add(egui::ProgressBar::new(fraction).text(format!("{done} / {} samples", job.samples)));
            ui.horizontal(|ui| {
                ui.label(RichText::new(format!("{} × {} · {elapsed:.0} s · about {eta:.0} s left", job.width, job.height)).color(theme::TEXT_MUTED));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    cancel = ui.button("Cancel").clicked();
                });
            });
        });
        if cancel {
            self.render_job = None;
            self.ui.status = "Render cancelled".into();
        }
        ctx.request_repaint();
    }

    fn render_result_window(&mut self, ctx: &egui::Context) {
        if self.render_result.is_none() {
            return;
        }
        let mut open = self.ui.render_result_open;
        let mut save = false;
        let mut again = false;
        let result = self.render_result.as_ref().unwrap();
        egui::Window::new("Render").open(&mut open).default_size([720.0, 560.0]).resizable(true).show(ctx, |ui| {
            let avail = ui.available_size() - egui::vec2(0.0, 48.0);
            let [w, h] = result.texture.size();
            let scale = (avail.x / w as f32).min(avail.y / h as f32).min(1.0).max(0.05);
            let size = egui::vec2(w as f32, h as f32) * scale;
            let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
            if result.transparent {
                // A checkerboard shows where it is see-through.
                let cell = 12.0;
                let painter = ui.painter_at(rect);
                let (cols, rows) = ((size.x / cell).ceil() as i32, (size.y / cell).ceil() as i32);
                painter.rect_filled(rect, 0.0, egui::Color32::from_gray(82));
                for y in 0..rows {
                    for x in (y % 2..cols).step_by(2) {
                        let min = rect.min + egui::vec2(x as f32 * cell, y as f32 * cell);
                        painter.rect_filled(egui::Rect::from_min_size(min, egui::vec2(cell, cell)).intersect(rect), 0.0, egui::Color32::from_gray(61));
                    }
                }
            }
            ui.painter().image(result.texture.id(), rect, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), egui::Color32::WHITE);
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new(format!("{w} × {h} · {} samples · {:.1} s{}", result.samples, result.seconds, if result.transparent { " · transparent" } else { "" }))
                        .color(theme::TEXT_MUTED),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    save = ui.add(widgets::primary_button(format!("{}  Save PNG…", icons::EXPORT))).clicked();
                    again = ui.button("Render again").clicked();
                });
            });
            if let Some(path) = &result.saved_to {
                ui.label(RichText::new(format!("Saved to {}", path.display())).small().color(theme::TEXT_DIM));
            }
        });
        self.ui.render_result_open = open;
        if save {
            self.save_render();
        }
        if again {
            self.start_render();
        }
    }

    /// In path-traced mode with the studio open, dim what the render won't
    /// show and outline the rest.
    pub(super) fn studio_frame_guide(&self, ctx: &egui::Context, rect: egui::Rect) {
        if !self.ui.studio_open || !self.pt_mode {
            return;
        }
        let aspect = self.studio.render_width as f32 / self.studio.render_height.max(1) as f32;
        let vp = rect.width() / rect.height().max(1.0);
        let size = if aspect > vp { egui::vec2(rect.width(), rect.width() / aspect) } else { egui::vec2(rect.height() * aspect, rect.height()) };
        let frame = egui::Rect::from_center_size(rect.center(), size);
        let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Background, "studio_guide".into())).with_clip_rect(rect);
        let shade = egui::Color32::from_black_alpha(120);
        for r in [
            egui::Rect::from_min_max(rect.min, egui::pos2(rect.max.x, frame.min.y)),
            egui::Rect::from_min_max(egui::pos2(rect.min.x, frame.max.y), rect.max),
            egui::Rect::from_min_max(egui::pos2(rect.min.x, frame.min.y), egui::pos2(frame.min.x, frame.max.y)),
            egui::Rect::from_min_max(egui::pos2(frame.max.x, frame.min.y), egui::pos2(rect.max.x, frame.max.y)),
        ] {
            if r.width() > 0.5 && r.height() > 0.5 {
                painter.rect_filled(r, 0.0, shade);
            }
        }
        painter.rect_stroke(frame, 0.0, egui::Stroke::new(1.0_f32, theme::ACCENT.gamma_multiply(0.7)), egui::StrokeKind::Inside);
    }
}

/// Copy a render's pixels to the CPU.
fn read_image(gpu: &crate::environment::GpuContext, target: &Arc<ImageView>, w: u32, h: u32) -> anyhow::Result<image::RgbaImage> {
    let buffer: Subbuffer<[u8]> = Buffer::new_slice(
        gpu.memory_allocator.clone(),
        BufferCreateInfo { usage: BufferUsage::TRANSFER_DST, ..Default::default() },
        AllocationCreateInfo { memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_RANDOM_ACCESS, ..Default::default() },
        (w * h * 4) as u64,
    )?;
    let mut builder = AutoCommandBufferBuilder::primary(gpu.command_buffer_allocator.clone(), gpu.queue.queue_family_index(), CommandBufferUsage::OneTimeSubmit)?;
    builder.copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(target.image().clone(), buffer.clone()))?;
    builder.build()?.execute(gpu.queue.clone())?.then_signal_fence_and_flush()?.wait(None)?;
    let pixels = buffer.read()?.to_vec();
    image::RgbaImage::from_raw(w, h, pixels).ok_or_else(|| anyhow::anyhow!("the render has the wrong size"))
}
