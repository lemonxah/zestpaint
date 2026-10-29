//! Which pieces were hidden in each model file, so opening the same .fbx /
//! .gltf again picks up where you left off. Projects keep visibility
//! themselves; this is for working straight from the model file.
//! Stored in `~/.config/zestpaint/hidden_pieces.json`: model path → hidden sets,
//! each named "Object · Material".

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use super::AppState;

fn store_path() -> PathBuf {
    crate::config::file("hidden_pieces.json")
}

fn key(model: &Path) -> String {
    std::fs::canonicalize(model).unwrap_or_else(|_| model.to_path_buf()).to_string_lossy().into_owned()
}

fn load_from(store: &Path) -> BTreeMap<String, Vec<String>> {
    std::fs::read_to_string(store).ok().and_then(|text| serde_json::from_str(&text).ok()).unwrap_or_default()
}

/// The pieces hidden last time `model` was open.
fn hidden_in(store: &Path, model: &Path) -> Vec<String> {
    load_from(store).remove(&key(model)).unwrap_or_default()
}

/// Record the pieces now hidden in `model` (none: forget it).
fn remember_in(store: &Path, model: &Path, hidden: &[String]) -> anyhow::Result<()> {
    let mut all = load_from(store);
    if hidden.is_empty() {
        all.remove(&key(model));
    } else {
        all.insert(key(model), hidden.to_vec());
    }
    if let Some(dir) = store.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(store, serde_json::to_string_pretty(&all)?)?;
    Ok(())
}

impl AppState {
    /// Hide what was hidden the last time this model file was open; returns
    /// how many pieces that hid.
    pub(super) fn restore_hidden_pieces(&mut self, model: &Path) -> usize {
        let hidden = hidden_in(&store_path(), model);
        for set in &mut self.texture_sets {
            if hidden.contains(&set.label()) {
                set.visible = false;
            }
        }
        self.model_source = Some(model.to_path_buf());
        self.remembered_hidden = self.hidden_now();
        self.remembered_hidden.len()
    }

    /// Hidden sets, by index — the pre-isolation state while in local view,
    /// which is what the user chose.
    fn hidden_now(&self) -> Vec<usize> {
        let visible: Vec<bool> = match &self.isolate_backup {
            Some(backup) => backup.clone(),
            None => self.texture_sets.iter().map(|s| s.visible).collect(),
        };
        visible.iter().enumerate().filter(|(_, v)| !**v).map(|(i, _)| i).collect()
    }

    /// Called every frame: write the hidden pieces down when they change.
    pub(super) fn remember_hidden_pieces(&mut self) {
        let Some(model) = self.model_source.clone() else { return };
        let hidden = self.hidden_now();
        if hidden == self.remembered_hidden {
            return;
        }
        let labels: Vec<String> = hidden.iter().filter_map(|&i| self.texture_sets.get(i)).map(|s| s.label()).collect();
        if let Err(e) = remember_in(&store_path(), &model, &labels) {
            eprintln!("could not remember hidden pieces: {e:#}");
        }
        self.remembered_hidden = hidden;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hidden_pieces_are_remembered_per_model() {
        let dir = std::env::temp_dir().join(format!("zestpaint-pieces-{}", std::process::id()));
        let store = dir.join("hidden_pieces.json");
        let (a, b) = (dir.join("a.fbx"), dir.join("b.fbx"));
        remember_in(&store, &a, &["Jacket · Jacket".into(), "Hair · Hair".into()]).unwrap();
        remember_in(&store, &b, &["Socks · Socks".into()]).unwrap();
        assert_eq!(hidden_in(&store, &a), ["Jacket · Jacket", "Hair · Hair"]);
        assert_eq!(hidden_in(&store, &b), ["Socks · Socks"]);
        remember_in(&store, &a, &[]).unwrap();
        assert!(hidden_in(&store, &a).is_empty(), "showing everything forgets the model");
        assert_eq!(hidden_in(&store, &b), ["Socks · Socks"]);
        std::fs::remove_dir_all(&dir).ok();
    }
}
