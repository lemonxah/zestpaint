use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result};

use crate::scene::{ImageData, MapSlot};

/// A material folder in the on-disk library, with whichever maps it provides.
pub struct LibraryMaterial {
    /// Folder name — stable, and the thumbnail cache key.
    pub name: String,
    /// Friendly name from `material.json`, falling back to the folder name.
    pub display: String,
    /// Tags from `material.json`, used by the library filter.
    pub tags: Vec<String>,
    /// Where it came from, per `material.json` ("polyhaven", "ambientcg",
    /// "imported", "procedural"); None for folders without a manifest, such
    /// as materials saved from the app.
    pub source: Option<String>,
    pub dir: PathBuf,
    /// (slot, file) pairs discovered by name matching.
    pub maps: Vec<(MapSlot, PathBuf)>,
}

impl LibraryMaterial {
    /// The map a preview thumbnail is built from. Materials show their colour;
    /// generated stamps have no colour map, so they fall back to coverage and
    /// then relief — which is what a brush actually looks like.
    pub fn preview_source(&self) -> Option<&PathBuf> {
        self.pick(&[MapSlot::BaseColor, MapSlot::Opacity, MapSlot::Height])
    }

    /// The greyscale map to use as a brush tip: coverage first, then relief.
    pub fn tip_source(&self) -> Option<&PathBuf> {
        self.pick(&[MapSlot::Opacity, MapSlot::Height, MapSlot::BaseColor])
    }

    fn pick(&self, order: &[MapSlot]) -> Option<&PathBuf> {
        order
            .iter()
            .find_map(|want| self.maps.iter().find(|(s, _)| s == want))
            .or_else(|| self.maps.first())
            .map(|(_, p)| p)
    }

    /// Whether this material matches a whitespace-separated filter.
    pub fn matches(&self, terms: &[String]) -> bool {
        if terms.is_empty() {
            return true;
        }
        terms.iter().all(|t| {
            self.display.to_lowercase().contains(t)
                || self.name.to_lowercase().contains(t)
                || self.tags.iter().any(|tag| tag.contains(t))
        })
    }
}

/// Name, tags and source recorded by `matfetch` alongside the maps.
fn read_manifest(dir: &Path) -> (Option<String>, Vec<String>, Option<String>) {
    let Ok(text) = std::fs::read_to_string(dir.join("material.json")) else {
        return (None, Vec::new(), None);
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) else {
        return (None, Vec::new(), None);
    };
    let source = value.get("source").and_then(|v| v.as_str()).map(str::to_string);
    let name = value.get("name").and_then(|v| v.as_str()).map(str::to_string);
    let tags = value
        .get("tags")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter().filter_map(|t| t.as_str()).map(|t| t.to_lowercase()).collect()
        })
        .unwrap_or_default();
    (name, tags, source)
}

/// Folder-safe name for an imported asset.
pub fn sanitize(name: &str) -> String {
    let out: String = name
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect();
    let trimmed = out.trim_matches('_').to_string();
    if trimmed.is_empty() { "brush".to_string() } else { trimmed }
}

fn home_dir() -> PathBuf {
    std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."))
}

/// `~/.zestsurface`.
pub fn data_root() -> PathBuf {
    home_dir().join(".zestsurface")
}

/// Rename `~/.zestpaint/library` to `~/.zestsurface/library` the first time
/// the new directory has nothing in it. A library already there wins.
pub fn migrate_library() {
    let new_lib = data_root().join("library");
    let old_lib = home_dir().join(".zestpaint/library");
    if !old_lib.exists() || std::fs::read_dir(&new_lib).ok().and_then(|mut d| d.next()).is_some() {
        return;
    }
    if new_lib.exists() {
        let _ = std::fs::remove_dir_all(&new_lib);
    }
    if let Some(parent) = new_lib.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if std::fs::rename(&old_lib, &new_lib).is_err() {
        let _ = std::os::unix::fs::symlink(&old_lib, &new_lib);
    }
}

/// Root of the user's material library: `~/.zestsurface/library/materials`.
/// Drop CC0 material folders here (ambientCG / Poly Haven zips extract into
/// exactly the naming this scanner understands).
pub fn library_dir() -> PathBuf {
    data_root().join("library/materials")
}

/// Root of the brush/stamp library: `~/.zestsurface/library/brushes`. Structurally
/// identical to a material folder — decals and atlases carry full PBR sets,
/// generated stamps carry just coverage and relief — so the same scanner reads
/// both.
pub fn brush_dir() -> PathBuf {
    data_root().join("library/brushes")
}

/// Scan the material library.
pub fn scan() -> Vec<LibraryMaterial> {
    scan_dir(&library_dir())
}

/// Scan the brush/stamp library.
pub fn scan_brushes() -> Vec<LibraryMaterial> {
    scan_dir(&brush_dir())
}

/// Scan a library root for asset folders.
fn scan_dir(root: &Path) -> Vec<LibraryMaterial> {
    let Ok(entries) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    let mut materials: Vec<LibraryMaterial> = entries
        .flatten()
        .filter(|e| e.path().is_dir())
        .filter_map(|e| {
            let dir = e.path();
            let name = e.file_name().to_string_lossy().to_string();
            let maps = scan_material_dir(&dir);
            if maps.is_empty() {
                return None;
            }
            let (display, tags, source) = read_manifest(&dir);
            let display = display.unwrap_or_else(|| name.clone());
            Some(LibraryMaterial { name, display, tags, source, dir, maps })
        })
        .collect();
    materials.sort_by(|a, b| a.display.to_lowercase().cmp(&b.display.to_lowercase()));
    materials
}

/// Match image files in a folder to material slots by common naming
/// conventions (ambientCG, Poly Haven, cgbookcase, Substance exports).
fn scan_material_dir(dir: &Path) -> Vec<(MapSlot, PathBuf)> {
    let mut maps: Vec<(MapSlot, PathBuf)> = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return maps;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(ext) = path.extension().and_then(|e| e.to_str()) else { continue };
        if !matches!(ext.to_ascii_lowercase().as_str(), "png" | "jpg" | "jpeg" | "tga" | "webp") {
            continue;
        }
        let stem = path.file_stem().unwrap_or_default().to_string_lossy().to_lowercase();
        let slot = if stem.contains("emissi") || stem.contains("emission") {
            if stem.contains("mask") || stem.contains("strength") { MapSlot::EmissionMask } else { MapSlot::Emission }
        } else if stem.contains("basecolor")
            || stem.contains("base_color")
            || stem.contains("albedo")
            || stem.contains("diffuse")
            || (stem.contains("color") && !stem.contains("colormask"))
        {
            MapSlot::BaseColor
        } else if stem.contains("opacity") || stem.contains("alpha") {
            MapSlot::Opacity
        } else if stem.contains("rough") {
            MapSlot::Roughness
        } else if stem.contains("metal") {
            MapSlot::Metallic
        } else if stem.contains("normal") && !stem.contains("normaldx") && !stem.contains("_dx") {
            MapSlot::Normal
        } else if stem.contains("height") || stem.contains("displacement") || stem.contains("bump")
        {
            MapSlot::Height
        } else if stem.contains("ambientocclusion")
            || stem.contains("occlusion")
            || stem.ends_with("_ao")
            || stem.contains("_ao_")
            || stem.ends_with("ao")
        {
            MapSlot::Ao
        } else {
            continue;
        };
        // First match per slot wins.
        if !maps.iter().any(|(s, _)| *s == slot) {
            maps.push((slot, path));
        }
    }
    maps
}

/// Load one library map file into CPU image data.
pub fn load_map(path: &Path) -> Result<ImageData> {
    let img = image::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let rgba = img.to_rgba8();
    let (width, height) = rgba.dimensions();
    Ok(ImageData { width, height, rgba: rgba.into_raw().into() })
}

/// Save a material's maps as PNGs into a new library folder.
pub fn save_material(name: &str, maps: &[(MapSlot, &ImageData)]) -> Result<PathBuf> {
    let safe: String = name
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect();
    let dir = library_dir().join(&safe);
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("failed to create {}", dir.display()))?;
    for (slot, img) in maps {
        let file = match slot {
            MapSlot::BaseColor => "basecolor.png",
            MapSlot::Roughness => "roughness.png",
            MapSlot::Metallic => "metallic.png",
            MapSlot::Normal => "normal.png",
            MapSlot::Height => "height.png",
            MapSlot::Ao => "ao.png",
            MapSlot::Emission => "emission.png",
            MapSlot::EmissionMask => "emission_mask.png",
            MapSlot::Opacity => "opacity.png",
        };
        let buffer: image::ImageBuffer<image::Rgba<u8>, _> =
            image::ImageBuffer::from_raw(img.width, img.height, img.rgba.to_vec())
                .context("bad image data")?;
        let path = dir.join(file);
        buffer.save(&path).with_context(|| format!("failed to write {}", path.display()))?;
    }
    Ok(dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::MapSlot;

    /// The canonical filenames `matfetch` writes must all map to a slot. This
    /// is the contract between the downloader and the loader: if either side
    /// renames a map, materials silently lose channels.
    #[test]
    fn canonical_matfetch_filenames_all_resolve() {
        let dir = std::env::temp_dir()
            .join(format!("zestsurface-lib-{}-{:?}", std::process::id(), std::thread::current().id()));
        std::fs::create_dir_all(&dir).unwrap();

        let expected = [
            ("basecolor.jpg", MapSlot::BaseColor),
            ("roughness.jpg", MapSlot::Roughness),
            ("metallic.jpg", MapSlot::Metallic),
            ("normal.jpg", MapSlot::Normal),
            ("height.jpg", MapSlot::Height),
            ("ao.jpg", MapSlot::Ao),
            ("opacity.jpg", MapSlot::Opacity),
        ];
        for (file, _) in expected {
            std::fs::write(dir.join(file), b"").unwrap();
        }
        // Non-map files in the folder must be ignored, not misfiled.
        std::fs::write(dir.join("material.json"), b"{}").unwrap();

        let found = scan_material_dir(&dir);
        std::fs::remove_dir_all(&dir).ok();

        assert_eq!(found.len(), expected.len(), "every canonical map should resolve: {found:?}");
        for (file, slot) in expected {
            let hit = found
                .iter()
                .find(|(s, _)| *s == slot)
                .unwrap_or_else(|| panic!("{file} did not resolve to its slot"));
            assert_eq!(
                hit.1.file_name().unwrap().to_string_lossy(),
                file,
                "slot resolved to the wrong file"
            );
        }
    }

    /// Opacity must not be swallowed by the base-colour branch, which matches
    /// loosely on "color".
    #[test]
    fn opacity_is_not_confused_with_base_colour() {
        let dir = std::env::temp_dir()
            .join(format!("zestsurface-lib2-{}-{:?}", std::process::id(), std::thread::current().id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("opacity.png"), b"").unwrap();
        let found = scan_material_dir(&dir);
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].0, MapSlot::Opacity);
    }
}

// ------------------------------------------------------------- thumbnails

/// Edge length of the cached library preview thumbnails.
pub const THUMB_SIZE: u32 = 128;

fn thumb_dir() -> PathBuf {
    library_dir().parent().map(|p| p.join(".thumbs")).unwrap_or_else(|| PathBuf::from(".thumbs"))
}

/// Decode a preview at thumbnail size, caching the result on disk. The library
/// holds thousands of 2K materials, so full decodes must happen once, ever.
fn build_thumbnail(key: &str, source: &Path) -> Result<ImageData> {
    let cache = thumb_dir().join(format!("{key}.png"));
    if let Ok(img) = image::open(&cache) {
        let rgba = img.to_rgba8();
        let (width, height) = rgba.dimensions();
        return Ok(ImageData { width, height, rgba: rgba.into_raw().into() });
    }
    let img = image::open(source)
        .with_context(|| format!("failed to open {}", source.display()))?
        .thumbnail(THUMB_SIZE, THUMB_SIZE);
    let _ = std::fs::create_dir_all(thumb_dir());
    let _ = img.save(&cache);
    let rgba = img.to_rgba8();
    let (width, height) = rgba.dimensions();
    Ok(ImageData { width, height, rgba: rgba.into_raw().into() })
}

/// Builds preview thumbnails off the UI thread, so scrolling a library of a
/// few thousand materials never blocks on a JPEG decode.
pub struct ThumbnailLoader {
    requests: std::sync::mpsc::Sender<(String, PathBuf)>,
    results: std::sync::mpsc::Receiver<(String, ImageData)>,
    requested: std::collections::HashSet<String>,
}

impl ThumbnailLoader {
    pub fn new(workers: usize) -> Self {
        let (req_tx, req_rx) = std::sync::mpsc::channel::<(String, PathBuf)>();
        let (res_tx, res_rx) = std::sync::mpsc::channel();
        let shared = std::sync::Arc::new(std::sync::Mutex::new(req_rx));
        for _ in 0..workers.max(1) {
            let shared = std::sync::Arc::clone(&shared);
            let res_tx = res_tx.clone();
            std::thread::spawn(move || {
                loop {
                    // The guard is held across recv, so workers hand off one
                    // job at a time — fine, the decode itself is the slow part.
                    let job = shared.lock().unwrap().recv();
                    let Ok((key, path)) = job else { break };
                    match build_thumbnail(&key, &path) {
                        Ok(image) => {
                            if res_tx.send((key, image)).is_err() {
                                break;
                            }
                        }
                        Err(e) => eprintln!("[library] thumbnail {key}: {e:#}"),
                    }
                }
            });
        }
        Self { requests: req_tx, results: res_rx, requested: std::collections::HashSet::new() }
    }

    /// Queue a thumbnail once; repeat calls for the same key are ignored.
    pub fn request(&mut self, key: &str, source: &Path) {
        if self.requested.insert(key.to_string()) {
            let _ = self.requests.send((key.to_string(), source.to_path_buf()));
        }
    }

    /// Thumbnails finished since the last call.
    pub fn drain(&self) -> Vec<(String, ImageData)> {
        self.results.try_iter().collect()
    }

    /// Forget what has been requested, so a rescan reloads previews.
    pub fn reset(&mut self) {
        self.requested.clear();
    }
}
