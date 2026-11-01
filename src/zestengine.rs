//! ZestEngine engine exchange: `zestengine.json` in, `zestengine_return.json` + maps out.
//! The JSON field names are a shared contract with the ZestEngine project.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ZestEngineManifest {
    pub object_id: String,
    pub object_name: String,
    pub slot_count: usize,
    pub return_dir: PathBuf,
    /// Starting maps (the applied material's textures) to import as layers.
    #[serde(default)]
    pub textures: Vec<ReturnMap>,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct ZestEngineReturn {
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

pub fn read_manifest(path: &Path) -> anyhow::Result<ZestEngineManifest> {
    let text = std::fs::read_to_string(path)?;
    Ok(serde_json::from_str(&text)?)
}

pub fn write_return(dir: &Path, ret: &ZestEngineReturn) -> anyhow::Result<()> {
    std::fs::write(dir.join("zestengine_return.json"), serde_json::to_string_pretty(ret)?)?;
    Ok(())
}
