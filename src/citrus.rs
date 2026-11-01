//! Citrus engine exchange: `citrus.json` in, `citrus_return.json` + maps out.
//! The JSON field names are a shared contract with the Citrus project.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CitrusManifest {
    pub object_id: String,
    pub object_name: String,
    pub slot_count: usize,
    pub return_dir: PathBuf,
    /// Starting maps (the applied material's textures) to import as layers.
    #[serde(default)]
    pub textures: Vec<ReturnMap>,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct CitrusReturn {
    pub object_id: String,
    pub maps: Vec<ReturnMap>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ReturnMap {
    /// Material slot index (0..`slot_count`), in the manifest's order.
    pub slot: usize,
    pub kind: String,
    pub file: String,
}

pub fn read_manifest(path: &Path) -> anyhow::Result<CitrusManifest> {
    let text = std::fs::read_to_string(path)?;
    Ok(serde_json::from_str(&text)?)
}

pub fn write_return(dir: &Path, ret: &CitrusReturn) -> anyhow::Result<()> {
    std::fs::write(dir.join("citrus_return.json"), serde_json::to_string_pretty(ret)?)?;
    Ok(())
}
