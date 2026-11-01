//! Blendshapes for a document that has none yet: a project saved before
//! ZestSurface kept them takes them from its model file, matched onto the
//! existing meshes so the painting stays.

use std::path::{Path, PathBuf};

use super::AppState;
use crate::scene::ModelData;

/// A model file being read for its blendshapes.
pub(super) struct ShapeImport {
    path: PathBuf,
    handle: std::thread::JoinHandle<anyhow::Result<Vec<ModelData>>>,
}

impl ShapeImport {
    pub(super) fn file_name(&self) -> String {
        self.path.file_name().unwrap_or_default().to_string_lossy().into_owned()
    }
}

impl AppState {
    /// The model file beside the open project with the same name, if any.
    pub(super) fn shape_source_beside_project(&self) -> Option<PathBuf> {
        let project = self.project_path.as_ref()?;
        ["fbx", "glb", "gltf"].into_iter().map(|ext| project.with_extension(ext)).find(|p| p.is_file())
    }

    /// Read `path` on a worker; `poll_shape_import` attaches its shapes.
    pub(super) fn start_shape_import(&mut self, path: &Path) {
        if self.shape_import.is_some() {
            return;
        }
        let job = path.to_path_buf();
        let handle = std::thread::spawn(move || {
            let is_fbx = job.extension().is_some_and(|e| e.eq_ignore_ascii_case("fbx"));
            let models = if is_fbx {
                crate::fbx_loader::load_with_progress(&job, &|_, _, _| {})
            } else {
                crate::gltf_loader::load_with_progress(&job, &|_, _, _| {})
            }?;
            // Same splitting as an import, so part names line up.
            Ok(crate::scene::split_udim_tiles(models))
        });
        self.shape_import = Some(ShapeImport { path: path.to_path_buf(), handle });
        self.ui.status = format!("Reading blendshapes from {}…", path.file_name().unwrap_or_default().to_string_lossy());
    }

    /// Called every frame: attach the shapes once the file is read.
    pub(super) fn poll_shape_import(&mut self) {
        if !self.shape_import.as_ref().is_some_and(|j| j.handle.is_finished()) {
            return;
        }
        let job = self.shape_import.take().unwrap();
        let name = job.file_name();
        let source = match job.handle.join() {
            Ok(Ok(models)) => models,
            Ok(Err(e)) => {
                self.ui.status = format!("Could not read {name}: {e:#}");
                return;
            }
            Err(_) => {
                self.ui.status = format!("Reading {name} crashed");
                return;
            }
        };
        let attached = crate::scene::attach_morphs(&mut self.models, &source);
        for &i in &attached {
            self.renderer.update_vertices(i, &self.models[i].mesh.vertices);
        }
        // Bones too: a project saved before ZestSurface kept them can be posed.
        let rigged = crate::scene::attach_skins(&mut self.models, &source);
        if !rigged.is_empty() {
            self.apply_pose();
        }
        let shapes: usize = self.blendshape_groups().iter().map(|(_, s)| s.len()).sum();
        self.pt_scene_stale = true;
        self.mesh_groups = None;
        self.paint_projection = None;
        let bones = if rigged.is_empty() { String::new() } else { format!(", bones on {} pieces", rigged.len()) };
        self.ui.status = if attached.is_empty() && rigged.is_empty() {
            format!("{name} has no blendshapes or bones for these pieces (object and material names must match)")
        } else {
            format!("{shapes} blendshapes from {name} on {} pieces{bones} — save to keep them in the project", attached.len())
        };
    }
}
