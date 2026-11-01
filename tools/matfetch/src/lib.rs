//! matfetch's catalog/download/generator core, shared by the `matfetch` CLI and
//! the zestsurface assistant (which searches and installs assets on request).

pub mod brushes;
pub mod catalog;
pub mod download;
pub mod http;
pub mod lace;
pub mod splatter;

use std::path::{Path, PathBuf};

use anyhow::Result;

use catalog::Index;

pub fn index_path(out: &Path) -> PathBuf {
    out.join(".matfetch/index.json")
}

/// The cached index, syncing both sites first when there is none.
pub fn load_index(out: &Path, agent: &ureq::Agent) -> Result<Index> {
    let path = index_path(out);
    if let Ok(text) = std::fs::read_to_string(&path)
        && let Ok(index) = serde_json::from_str::<Index>(&text)
        && !index.assets.is_empty()
    {
        return Ok(index);
    }
    eprintln!("[matfetch] no cached index, syncing first...");
    let index = catalog::sync(agent)?;
    save_index(out, &index)?;
    Ok(index)
}

pub fn save_index(out: &Path, index: &Index) -> Result<()> {
    let path = index_path(out);
    std::fs::create_dir_all(path.parent().unwrap())?;
    std::fs::write(&path, serde_json::to_vec(index)?)?;
    eprintln!("[matfetch] index: {} assets -> {}", index.assets.len(), path.display());
    Ok(())
}
