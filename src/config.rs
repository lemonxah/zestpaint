//! The settings folder, `$XDG_CONFIG_HOME/zestpaint` (else
//! `~/.config/zestpaint`), and `settings.json`, the editor's own
//! preferences. The other files there belong to their features: the
//! assistant's `assistant.json` and `comfy_workflow.json`,
//! `recent_projects.json`, `export_presets.json`, `hidden_pieces.json` and
//! the MCP bridge's `mcp-bridge.json`. They used to live in `~/.zestpaint`; [`migrate_legacy`]
//! moves them over once. The libraries stay in `~/.zestpaint/library`: they
//! are the user's assets, not settings.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

fn home() -> PathBuf {
    std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."))
}

pub fn dir() -> PathBuf {
    match std::env::var_os("XDG_CONFIG_HOME").filter(|v| !v.is_empty()) {
        Some(base) => PathBuf::from(base).join("zestpaint"),
        None => home().join(".config/zestpaint"),
    }
}

/// A file in the settings folder.
pub fn file(name: &str) -> PathBuf {
    dir().join(name)
}

/// The settings files that used to live in `~/.zestpaint`.
const LEGACY_FILES: [&str; 6] = [
    "assistant.json",
    "comfy_workflow.json",
    "recent_projects.json",
    "export_presets.json",
    "hidden_pieces.json",
    "mcp-bridge.json",
];

/// Move the settings files from `~/.zestpaint` into the settings folder,
/// each only when the folder has none of that name yet.
pub fn migrate_legacy() {
    migrate_from(&home().join(".zestpaint"), &dir());
}

fn migrate_from(legacy: &Path, target: &Path) {
    for name in LEGACY_FILES {
        let (from, to) = (legacy.join(name), target.join(name));
        if !from.is_file() || to.exists() {
            continue;
        }
        if std::fs::create_dir_all(target).is_err() {
            return;
        }
        // A rename keeps the owner-only permissions of the key and token
        // files; across filesystems, copy (which keeps them too) and remove.
        if std::fs::rename(&from, &to).is_err() && std::fs::copy(&from, &to).is_ok() {
            let _ = std::fs::remove_file(&from);
        }
    }
}

/// What the editor shows when it starts without a file to open.
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug, Default)]
#[serde(rename_all = "snake_case")]
pub enum Startup {
    /// The welcome screen: open a model or project, or pick a recent one.
    #[default]
    Welcome,
    /// Reopen the most recent project that still exists (the welcome screen
    /// when there is none).
    LastProject,
    /// Straight to the default sphere.
    DefaultScene,
}

impl Startup {
    pub const ALL: [Startup; 3] = [Startup::Welcome, Startup::LastProject, Startup::DefaultScene];

    pub fn label(self) -> &'static str {
        match self {
            Startup::Welcome => "Show the welcome screen",
            Startup::LastProject => "Reopen the last project",
            Startup::DefaultScene => "Start with the default sphere",
        }
    }
}

/// `settings.json`. Missing fields take their defaults, so the file can be
/// edited by hand and older files keep working.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct Settings {
    pub startup: Startup,
    /// Undo history budget per texture set, in GB of system RAM.
    pub undo_budget_gb: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Settings { startup: Startup::default(), undo_budget_gb: crate::layers::DEFAULT_UNDO_BUDGET as f32 / GB }
    }
}

const GB: f32 = (1u64 << 30) as f32;

impl Settings {
    pub fn path() -> PathBuf {
        file("settings.json")
    }

    /// The saved settings. A missing file is written with the defaults, so
    /// there is always one to edit; one that doesn't parse is left alone.
    pub fn load() -> Self {
        Self::load_from(&Self::path())
    }

    fn load_from(path: &Path) -> Self {
        match std::fs::read_to_string(path) {
            Ok(text) => serde_json::from_str(&text).unwrap_or_else(|e| {
                eprintln!("{}: {e}; using the defaults", path.display());
                Settings::default()
            }),
            Err(_) => {
                let settings = Settings::default();
                settings.save_to(path);
                settings
            }
        }
    }

    pub fn save(&self) {
        self.save_to(&Self::path());
    }

    fn save_to(&self, path: &Path) {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(text) = serde_json::to_string_pretty(self) {
            if let Err(e) = std::fs::write(path, text + "\n") {
                eprintln!("could not save {}: {e}", path.display());
            }
        }
    }

    pub fn undo_budget_bytes(&self) -> u64 {
        (self.undo_budget_gb.clamp(0.5, 32.0) as f64 * GB as f64) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("zestpaint-config-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Old files move into the new folder; ones already there win.
    #[test]
    fn legacy_files_move_once() {
        let root = scratch("migrate");
        let (legacy, target) = (root.join("old"), root.join("new"));
        std::fs::create_dir_all(&legacy).unwrap();
        std::fs::write(legacy.join("recent_projects.json"), "[\"a\"]").unwrap();
        std::fs::write(legacy.join("assistant.json"), "old").unwrap();
        std::fs::create_dir_all(&target).unwrap();
        std::fs::write(target.join("assistant.json"), "new").unwrap();
        migrate_from(&legacy, &target);
        assert_eq!(std::fs::read_to_string(target.join("recent_projects.json")).unwrap(), "[\"a\"]");
        assert!(!legacy.join("recent_projects.json").exists());
        assert_eq!(std::fs::read_to_string(target.join("assistant.json")).unwrap(), "new");
        assert!(legacy.join("assistant.json").exists(), "the clash is left for the user");
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A missing file is created with the defaults; a hand-edited one with
    /// only some fields fills in the rest; a broken one is not overwritten.
    #[test]
    fn settings_round_trip() {
        let root = scratch("settings");
        let path = root.join("settings.json");
        assert_eq!(Settings::load_from(&path), Settings::default());
        assert!(path.exists(), "defaults written for editing");

        std::fs::write(&path, r#"{ "startup": "last_project" }"#).unwrap();
        let loaded = Settings::load_from(&path);
        assert_eq!(loaded.startup, Startup::LastProject);
        assert_eq!(loaded.undo_budget_gb, Settings::default().undo_budget_gb);

        std::fs::write(&path, "{ not json").unwrap();
        assert_eq!(Settings::load_from(&path), Settings::default());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "{ not json");
        let _ = std::fs::remove_dir_all(&root);
    }
}
