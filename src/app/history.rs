//! One undo history for the whole document, whichever panel, window or tool
//! made the change.
//!
//! Rather than every control recording its own undo step, the document is
//! snapshotted once a frame — cheaply: layers share their textures by `Arc`
//! (see `crate::layers::Layer`) — and any difference from the last committed
//! snapshot becomes one step, once the gesture making it is over (mouse up,
//! text field left). A slider drag or a typed name is therefore one step, and
//! nothing new is missed when a control is added.
//!
//! Strokes paint pixels in place, so they keep their own pixel snapshots in
//! each `LayerStack`; this timeline orders them among everything else. The
//! two edits too big to snapshot every frame — a material's map images and a
//! UV re-unwrap — record explicit steps.
//!
//! Not part of the document, and so not undone: brush and tool settings, the
//! camera, display & lighting, panel layout, and which set or layer is
//! selected. Opening a model starts a new document and a new history.

use std::sync::Arc;

use vulkano::image::view::ImageView;

use super::AppState;
use crate::layers::{StackState, TextureSet};
use crate::scene::{MapSlot, MaterialData, ModelData};

/// Steps kept. Old ones only hold texture references (and, for strokes,
/// the host-RAM snapshots each stack already budgets), so this is generous.
const MAX_STEPS: usize = 400;

/// One texture set's document state.
#[derive(Clone)]
struct SetState {
    stack: StackState,
    visible: bool,
    baked: [Option<Arc<ImageView>>; 4],
    /// The material's surface sliders, from its first model.
    surface: Option<Surface>,
}

/// The OpenPBR-style sliders plus opacity, compared by value.
#[derive(Clone, Copy, PartialEq, Debug)]
struct Surface {
    emission: [f32; 3],
    specular: f32,
    coat: f32,
    coat_roughness: f32,
    sheen: f32,
    sheen_roughness: f32,
    opacity: f32,
    transparent: bool,
}

impl Surface {
    fn of(m: &MaterialData) -> Self {
        Surface {
            emission: m.emission,
            specular: m.specular_weight,
            coat: m.coat_weight,
            coat_roughness: m.coat_roughness,
            sheen: m.sheen_weight,
            sheen_roughness: m.sheen_roughness,
            opacity: m.opacity,
            transparent: m.transparent,
        }
    }

    fn apply(self, m: &mut MaterialData) {
        m.emission = self.emission;
        m.specular_weight = self.specular;
        m.coat_weight = self.coat;
        m.coat_roughness = self.coat_roughness;
        m.sheen_weight = self.sheen;
        m.sheen_roughness = self.sheen_roughness;
        m.opacity = self.opacity;
        m.transparent = self.transparent;
    }
}

/// The whole document.
#[derive(Clone)]
pub struct DocState {
    sets: Vec<SetState>,
    isolate: Option<Vec<bool>>,
}

/// A material's maps and the factors that go with them.
#[derive(Clone)]
pub struct MaterialMaps {
    /// In `MapSlot::ALL` order.
    images: [Option<crate::scene::ImageData>; 8],
    base_color: [f32; 4],
    /// Assigning an emission map can switch the emission colour on.
    emission: [f32; 3],
    roughness: f32,
    metallic: f32,
}

enum Step {
    /// Any document change: the state before it.
    Doc(DocState),
    /// One stroke on `set`, and the document before it (strokes also touch
    /// layer metadata: channel badges, swatch).
    Stroke { set: usize, doc: DocState },
    /// A map slot change on `set`'s material: its maps before.
    Maps { set: usize, maps: MaterialMaps },
    /// A UV re-unwrap of `set`: the models and texture set before it.
    Unwrap { set: usize, models: Vec<ModelData>, texture_set: Box<TextureSet> },
    /// A paint-resolution change of `set`: the texture set before it.
    Resize { set: usize, texture_set: Box<TextureSet> },
    /// Pieces deleted: the models before, and the removed sets at their old
    /// indices (ascending), so undo puts every later index back where the
    /// steps below this one expect it.
    Delete { models: Vec<ModelData>, sets: Vec<(usize, TextureSet)> },
    /// A delete undone: the models without those pieces, and where they sit.
    Undelete { models: Vec<ModelData>, indices: Vec<usize> },
}

#[derive(Default)]
pub struct History {
    undo: Vec<Step>,
    redo: Vec<Step>,
    /// The document as of the last recorded step.
    committed: Option<DocState>,
    /// `strokes_begun` per set at the last look.
    strokes_seen: Vec<u64>,
}

impl History {
    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    pub fn len(&self) -> usize {
        self.undo.len()
    }
}

fn same_doc(a: &DocState, b: &DocState) -> bool {
    a.isolate == b.isolate
        && a.sets.len() == b.sets.len()
        && a.sets.iter().zip(&b.sets).all(|(x, y)| {
            x.visible == y.visible
                && x.surface == y.surface
                && x.baked.iter().zip(&y.baked).all(|(p, q)| match (p, q) {
                    (Some(p), Some(q)) => Arc::ptr_eq(p, q),
                    (None, None) => true,
                    _ => false,
                })
                && same_stack(&x.stack, &y.stack)
        })
}

/// `StackState` comparison goes through a scratch-free path: a stack that
/// matches one state matches another identical one.
fn same_stack(a: &StackState, b: &StackState) -> bool {
    a.same_document(b)
}

impl AppState {
    fn doc_state(&self) -> DocState {
        let sets = self
            .texture_sets
            .iter()
            .map(|set| SetState {
                stack: set.layers.state(),
                visible: set.visible,
                baked: [
                    set.baked_ao.clone(),
                    set.baked_position.clone(),
                    set.baked_normal.clone(),
                    set.baked_curvature.clone(),
                ],
                surface: self
                    .models
                    .iter()
                    .find(|m| m.name == set.object && m.material_name == set.material)
                    .map(|m| Surface::of(&m.material)),
            })
            .collect();
        DocState { sets, isolate: self.isolate_backup.clone() }
    }

    fn apply_doc(&mut self, doc: &DocState) {
        if doc.sets.len() != self.texture_sets.len() {
            return;
        }
        for (i, state) in doc.sets.iter().enumerate() {
            let set = &mut self.texture_sets[i];
            set.layers.restore_state(state.stack.clone());
            set.visible = state.visible;
            let [ao, position, normal, curvature] = state.baked.clone();
            set.baked_ao = ao;
            set.baked_position = position;
            set.baked_normal = normal;
            set.baked_curvature = curvature;
            if let Some(surface) = state.surface {
                let (object, material) = (set.object.clone(), set.material.clone());
                let mut params = None;
                for model in self.models.iter_mut().filter(|m| m.name == object && m.material_name == material) {
                    surface.apply(&mut model.material);
                    params = Some(model.material.surface_params());
                }
                if let Some(params) = params {
                    self.renderer.set_material_surface(i, params);
                }
            }
        }
        self.isolate_backup = doc.isolate.clone();
        // Visibility and layers both feed the path tracer and paint caches.
        self.pt_scene_stale = true;
        self.paint_projection = None;
        self.mesh_groups = None;
    }

    fn push_step(&mut self, step: Step) {
        let h = &mut self.ui.history;
        h.undo.push(step);
        if h.undo.len() > MAX_STEPS {
            h.undo.remove(0);
        }
        h.redo.clear();
        // Undone strokes are unreachable once something new is done.
        for set in &mut self.texture_sets {
            set.layers.clear_redo();
        }
    }

    /// Start over: a new document (model opened).
    pub(super) fn reset_history(&mut self) {
        self.ui.history = History::default();
        self.ui.history.committed = Some(self.doc_state());
        self.ui.history.strokes_seen = self.texture_sets.iter().map(|s| s.layers.strokes_begun()).collect();
    }

    /// Once a frame: record whatever changed since the last step, unless a
    /// gesture is still making the change (`busy`).
    pub(super) fn track_history(&mut self, busy: bool) {
        if self.ui.history.committed.is_none() || self.ui.history.strokes_seen.len() != self.texture_sets.len() {
            self.reset_history();
            return;
        }
        if busy {
            return;
        }
        let current = self.doc_state();
        let mut recorded = false;
        // Strokes first: each new one is a step, carrying the document as it
        // was before it (its metadata changes are part of the stroke).
        for i in 0..self.texture_sets.len() {
            let now = self.texture_sets[i].layers.strokes_begun();
            let seen = self.ui.history.strokes_seen[i];
            for _ in seen..now {
                let doc = self.ui.history.committed.clone().unwrap();
                self.push_step(Step::Stroke { set: i, doc });
                recorded = true;
            }
            self.ui.history.strokes_seen[i] = now;
        }
        if !recorded {
            let committed = self.ui.history.committed.as_ref().unwrap();
            if !same_doc(committed, &current) {
                let before = committed.clone();
                self.push_step(Step::Doc(before));
                recorded = true;
            }
        }
        if recorded {
            self.ui.history.committed = Some(current);
        }
    }

    /// `set`'s stack swapped placeholder textures for real ones (see
    /// `LayerStack::ensure_allocated`): make every saved state use them too,
    /// so the swap is not an edit and undo never brings a placeholder back
    /// under a layer that strokes paint in place.
    pub(super) fn adopt_upgrades(&mut self, set: usize, upgrades: &[crate::layers::Upgrade]) {
        let history = &mut self.ui.history;
        let docs = history.committed.iter_mut().chain(history.undo.iter_mut().chain(history.redo.iter_mut()).filter_map(|step| match step {
            Step::Doc(doc) | Step::Stroke { doc, .. } => Some(doc),
            _ => None,
        }));
        for doc in docs {
            if let Some(state) = doc.sets.get_mut(set) {
                state.stack.adopt_upgrades(upgrades);
            }
        }
    }

    /// Record a map-slot change on `set` before making it.
    pub(super) fn record_maps(&mut self, set: usize) {
        self.track_history(false);
        let maps = self.material_maps(set);
        if let Some(maps) = maps {
            self.push_step(Step::Maps { set, maps });
        }
    }

    /// Record a UV re-unwrap before it replaces `set` and its models' UVs:
    /// takes the old texture set and keeps a copy of the models.
    pub(super) fn record_unwrap(&mut self, set: usize, mut old_set: TextureSet, old_models: Vec<ModelData>) {
        // Composites are rebuilt from the layers if this is undone.
        old_set.layers.release_composites();
        self.push_step(Step::Unwrap { set, models: old_models, texture_set: Box::new(old_set) });
    }

    /// Paint textures the history keeps alive on the GPU: the whole old sets
    /// held by resolution changes, re-unwraps and deleted pieces.
    pub(super) fn history_vram_bytes(&self) -> u64 {
        let h = &self.ui.history;
        h.undo
            .iter()
            .chain(&h.redo)
            .filter_map(|step| match step {
                Step::Resize { texture_set, .. } | Step::Unwrap { texture_set, .. } => Some(set_vram_bytes(texture_set)),
                Step::Delete { sets, .. } => Some(sets.iter().map(|(_, s)| set_vram_bytes(s)).sum()),
                _ => None,
            })
            .sum()
    }

    /// Record a resolution change: keeps the whole old set (its layers and
    /// stroke history at the old size), so undo swaps it straight back.
    pub(super) fn record_resize(&mut self, set: usize, mut old_set: TextureSet) {
        // Composites are rebuilt from the layers if this is undone.
        old_set.layers.release_composites();
        self.push_step(Step::Resize { set, texture_set: Box::new(old_set) });
    }

    /// Record deleted pieces (see `Step::Delete`) after taking them out.
    pub(super) fn record_delete(&mut self, old_models: Vec<ModelData>, sets: Vec<(usize, TextureSet)>) {
        self.push_step(Step::Delete { models: old_models, sets });
    }

    /// Take the sets at `indices` (ascending) out of the document, composites
    /// released: deleted pieces only need their layers back on undo.
    pub(super) fn remove_sets(&mut self, indices: &[usize]) -> Vec<(usize, TextureSet)> {
        let mut removed = Vec::with_capacity(indices.len());
        for &i in indices.iter().rev() {
            if i < self.texture_sets.len() {
                let mut set = self.texture_sets.remove(i);
                set.layers.release_composites();
                removed.push((i, set));
            }
        }
        removed.reverse();
        removed
    }

    /// After pieces were deleted or restored: rebuild what hangs off the set
    /// list, keeping `active` (object, material) as the paint target if it is
    /// still there, else the first visible piece.
    pub(super) fn pieces_changed(&mut self, active: Option<(String, String)>) {
        self.renderer.set_models(&self.models);
        let count = self.texture_sets.len();
        self.active_set = active
            .and_then(|(object, material)| self.texture_sets.iter().position(|s| s.object == object && s.material == material))
            .or_else(|| self.texture_sets.iter().position(|s| s.visible))
            .unwrap_or(0);
        self.selection.retain(|&i| i < count);
        if self.isolate_backup.as_ref().is_some_and(|b| b.len() != count) {
            self.isolate_backup = None;
            self.isolate_camera = None;
        }
        self.outliner_hover.clear();
        self.paint_projection = None;
        self.mesh_groups = None;
        self.pt_scene_stale = true;
    }

    /// The paint target's (object, material), to find it again after the
    /// set list changes.
    pub(super) fn active_key(&self) -> Option<(String, String)> {
        self.texture_sets.get(self.active_set).map(|s| (s.object.clone(), s.material.clone()))
    }

    /// After an explicitly recorded change: the result is the new baseline.
    pub(super) fn settle_history(&mut self) {
        self.ui.history.committed = Some(self.doc_state());
        self.ui.history.strokes_seen = self.texture_sets.iter().map(|s| s.layers.strokes_begun()).collect();
    }

    fn material_maps(&self, set: usize) -> Option<MaterialMaps> {
        let ts = self.texture_sets.get(set)?;
        let m = &self.models.iter().find(|m| m.name == ts.object && m.material_name == ts.material)?.material;
        Some(MaterialMaps {
            images: [
                m.base_color_image.clone(),
                m.roughness_image.clone(),
                m.metallic_image.clone(),
                m.normal_image.clone(),
                m.height_image.clone(),
                m.ao_image.clone(),
                m.emission_image.clone(),
                m.emission_mask_image.clone(),
            ],
            base_color: m.base_color,
            emission: m.emission,
            roughness: m.roughness,
            metallic: m.metallic,
        })
    }

    /// Put a material's maps back, through the same path that assigns them.
    fn restore_material_maps(&mut self, set: usize, maps: MaterialMaps) {
        let previous = self.active_set;
        self.active_set = set;
        for (slot, image) in MapSlot::ALL.into_iter().zip(maps.images) {
            self.apply_material_map(slot, image);
        }
        let ts = &self.texture_sets[set];
        let (object, material) = (ts.object.clone(), ts.material.clone());
        let mut params = None;
        for model in self.models.iter_mut().filter(|m| m.name == object && m.material_name == material) {
            model.material.base_color = maps.base_color;
            model.material.roughness = maps.roughness;
            model.material.metallic = maps.metallic;
            model.material.emission = maps.emission;
            params = Some(model.material.surface_params());
        }
        if let Some(params) = params {
            self.renderer.set_material_surface(set, params);
        }
        self.active_set = previous;
    }

    /// Swap one step out of a stack: returns the step that reverses it.
    fn reverse(&mut self, step: Step, forward: bool) -> Step {
        match step {
            Step::Doc(doc) => {
                let now = self.doc_state();
                self.apply_doc(&doc);
                Step::Doc(now)
            }
            Step::Stroke { set, doc } => {
                let now = self.doc_state();
                if let Some(ts) = self.texture_sets.get_mut(set) {
                    if forward {
                        ts.layers.redo_stroke();
                    } else {
                        ts.layers.undo_stroke();
                    }
                }
                self.apply_doc(&doc);
                Step::Stroke { set, doc: now }
            }
            Step::Maps { set, maps } => {
                let now = self.material_maps(set);
                self.restore_material_maps(set, maps.clone());
                Step::Maps { set, maps: now.unwrap_or(maps) }
            }
            Step::Unwrap { set, mut models, texture_set } => {
                std::mem::swap(&mut self.models, &mut models);
                let mut texture_set = *texture_set;
                if set < self.texture_sets.len() {
                    std::mem::swap(&mut self.texture_sets[set], &mut texture_set);
                }
                self.renderer.set_models(&self.models);
                self.paint_projection = None;
                self.mesh_groups = None;
                self.pt_scene_stale = true;
                Step::Unwrap { set, models, texture_set: Box::new(texture_set) }
            }
            Step::Resize { set, texture_set } => {
                let mut texture_set = *texture_set;
                if set < self.texture_sets.len() {
                    std::mem::swap(&mut self.texture_sets[set], &mut texture_set);
                }
                self.pt_scene_stale = true;
                Step::Resize { set, texture_set: Box::new(texture_set) }
            }
            Step::Delete { mut models, sets } => {
                let active = self.active_key();
                std::mem::swap(&mut self.models, &mut models);
                let indices = sets.iter().map(|(i, _)| *i).collect();
                for (i, set) in sets {
                    let at = i.min(self.texture_sets.len());
                    self.texture_sets.insert(at, set);
                }
                self.selection.clear();
                self.pieces_changed(active);
                Step::Undelete { models, indices }
            }
            Step::Undelete { mut models, indices } => {
                let active = self.active_key();
                std::mem::swap(&mut self.models, &mut models);
                let sets = self.remove_sets(&indices);
                self.selection.clear();
                self.pieces_changed(active);
                Step::Delete { models, sets }
            }
        }
    }

    pub(super) fn undo(&mut self) {
        // Anything still uncommitted (a name being typed) is a step first.
        self.track_history(false);
        let Some(step) = self.ui.history.undo.pop() else {
            self.ui.status = "Nothing to undo".into();
            return;
        };
        let label = step_label(&step);
        let back = self.reverse(step, false);
        self.ui.history.redo.push(back);
        self.settle_history();
        self.ui.status = format!("Undo {label}");
    }

    pub(super) fn redo(&mut self) {
        self.track_history(false);
        let Some(step) = self.ui.history.redo.pop() else {
            self.ui.status = "Nothing to redo".into();
            return;
        };
        let label = step_label(&step);
        let back = self.reverse(step, true);
        self.ui.history.undo.push(back);
        self.settle_history();
        self.ui.status = format!("Redo {label}");
    }
}

/// GPU memory of one set's paint textures.
pub(super) fn set_vram_bytes(set: &TextureSet) -> u64 {
    set.layers.vram_bytes()
}

fn step_label(step: &Step) -> &'static str {
    match step {
        Step::Doc(_) => "edit",
        Step::Stroke { .. } => "stroke",
        Step::Maps { .. } => "material maps",
        Step::Unwrap { .. } => "UV unwrap",
        Step::Resize { .. } => "resolution change",
        // Undelete sits on the redo stack: redoing it deletes again.
        Step::Delete { .. } | Step::Undelete { .. } => "delete pieces",
    }
}
