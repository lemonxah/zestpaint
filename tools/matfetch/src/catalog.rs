use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context as _, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::http;

/// Canonical map slots. These filenames are exactly what `src/library.rs`
/// keyword-matches, so normalising to them makes every downloaded folder load
/// in the app regardless of which site it came from.
pub const BASECOLOR: &str = "basecolor";
pub const NORMAL: &str = "normal";
pub const ROUGHNESS: &str = "roughness";
pub const METALLIC: &str = "metallic";
pub const HEIGHT: &str = "height";
pub const AO: &str = "ao";
pub const OPACITY: &str = "opacity";

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct MapFile {
    pub slot: String,
    pub url: String,
    pub bytes: u64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Payload {
    /// ambientCG ships one zip per resolution/format.
    Zip { url: String },
    /// Poly Haven exposes each map as its own file.
    Maps { files: Vec<MapFile> },
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Variant {
    pub bytes: u64,
    #[serde(flatten)]
    pub payload: Payload,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Asset {
    /// Stable id, e.g. `ph:aerial_asphalt_01` or `acg:Ground107`.
    pub id: String,
    pub source: String,
    /// material | decal | atlas | brush
    pub kind: String,
    pub name: String,
    /// Folder name on disk, deduplicated across sources.
    pub dir: String,
    pub category: String,
    pub tags: Vec<String>,
    pub authors: Vec<String>,
    pub license: String,
    pub page: String,
    pub thumb: String,
    /// Keyed `"<res>-<fmt>"`, e.g. `"2k-jpg"`.
    pub variants: BTreeMap<String, Variant>,
}

impl Asset {
    /// Materials go in the app's material library; decals/atlases/brushes are
    /// stamps and live in the brush library.
    pub fn is_brush(&self) -> bool {
        self.kind != "material"
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct Index {
    pub generated: u64,
    pub assets: Vec<Asset>,
}

impl Index {
    pub fn get(&self, id: &str) -> Option<&Asset> {
        self.assets.iter().find(|a| a.id == id)
    }
}

pub const RESOLUTIONS: &[&str] = &["1k", "2k", "4k", "8k"];
pub const FORMATS: &[&str] = &["jpg", "png"];

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Fetch both catalogs and merge them into a single index.
pub fn sync(agent: &ureq::Agent) -> Result<Index> {
    let mut assets = Vec::new();
    eprintln!("[sync] Poly Haven textures...");
    assets.extend(sync_polyhaven(agent).context("Poly Haven sync")?);
    eprintln!("[sync] ambientCG assets...");
    assets.extend(sync_ambientcg(agent).context("ambientCG sync")?);

    assets.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    assign_dirs(&mut assets);
    Ok(Index { generated: now(), assets })
}

/// Give every asset a unique on-disk folder name. Both sites use distinct
/// naming conventions so collisions are rare, but they must be impossible.
fn assign_dirs(assets: &mut [Asset]) {
    let mut taken: HashSet<String> = HashSet::new();
    for a in assets.iter_mut() {
        let base = sanitize(&a.dir);
        let mut name = base.clone();
        if taken.contains(&name) {
            let suffix = if a.source == "polyhaven" { "ph" } else { "acg" };
            name = format!("{base}_{suffix}");
            let mut n = 2;
            while taken.contains(&name) {
                name = format!("{base}_{suffix}{n}");
                n += 1;
            }
        }
        taken.insert(name.clone());
        a.dir = name;
    }
}

pub fn sanitize(s: &str) -> String {
    let out: String = s
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect();
    out.trim_matches('_').to_string()
}

// ---------------------------------------------------------------- Poly Haven

/// Poly Haven file keys -> (canonical slot, preference). Lower preference wins
/// when a map is offered in several conventions (OpenGL normals over DirectX,
/// true displacement over bump).
fn ph_slot(key: &str) -> Option<(&'static str, u8)> {
    Some(match key {
        "Diffuse" => (BASECOLOR, 0),
        "nor_gl" | "normal_gl" => (NORMAL, 0),
        "nor_dx" | "normal_dx" => (NORMAL, 1),
        "Rough" => (ROUGHNESS, 0),
        "Metal" => (METALLIC, 0),
        "Displacement" => (HEIGHT, 0),
        "Bump" => (HEIGHT, 1),
        "AO" => (AO, 0),
        // Deliberately skipped: arm/rough_ao (packed), spec/spec_ior/ref,
        // anisotropy_*, col1/col2 (colour variants), and the blend/gltf/mtlx/page
        // bundles, which are scene files rather than maps.
        _ => return None,
    })
}

#[derive(Deserialize)]
struct PhAsset {
    name: String,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    categories: Vec<String>,
    #[serde(default)]
    authors: HashMap<String, Value>,
    #[serde(default)]
    thumbnail_url: String,
    #[serde(default)]
    category: Option<String>,
}

fn sync_polyhaven(agent: &ureq::Agent) -> Result<Vec<Asset>> {
    let listing: HashMap<String, PhAsset> =
        http::get_json(agent, "https://api.polyhaven.com/assets?type=textures")?;
    let slugs: Vec<String> = {
        let mut v: Vec<String> = listing.keys().cloned().collect();
        v.sort();
        v
    };
    eprintln!("[sync] Poly Haven: {} textures, fetching file lists", slugs.len());

    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let out: Mutex<Vec<Asset>> = Mutex::new(Vec::new());

    std::thread::scope(|scope| {
        for _ in 0..8 {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(slug) = slugs.get(i) else { break };
                    let url = format!("https://api.polyhaven.com/files/{slug}");
                    let files: HashMap<String, Value> = match http::get_json(agent, &url) {
                        Ok(v) => v,
                        Err(e) => {
                            eprintln!("[sync] {slug}: {e:#}");
                            continue;
                        }
                    };
                    let meta = &listing[slug];
                    if let Some(asset) = ph_build(slug, meta, &files) {
                        out.lock().unwrap().push(asset);
                    }
                    let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                    if d % 100 == 0 {
                        eprintln!("[sync] Poly Haven {d}/{}", slugs.len());
                    }
                }
            });
        }
    });

    Ok(out.into_inner().unwrap())
}

fn ph_build(slug: &str, meta: &PhAsset, files: &HashMap<String, Value>) -> Option<Asset> {
    let mut variants: BTreeMap<String, Variant> = BTreeMap::new();

    for res in RESOLUTIONS {
        for fmt in FORMATS {
            // slot -> (preference, MapFile)
            let mut chosen: HashMap<&str, (u8, MapFile)> = HashMap::new();
            for (key, node) in files {
                let Some((slot, pref)) = ph_slot(key) else { continue };
                let Some(file) = node.get(res).and_then(|r| r.get(fmt)) else { continue };
                let (Some(url), Some(size)) = (
                    file.get("url").and_then(Value::as_str),
                    file.get("size").and_then(Value::as_u64),
                ) else {
                    continue;
                };
                let entry = MapFile { slot: slot.to_string(), url: url.to_string(), bytes: size };
                match chosen.get(slot) {
                    Some((existing, _)) if *existing <= pref => {}
                    _ => {
                        chosen.insert(slot, (pref, entry));
                    }
                }
            }
            if chosen.is_empty() {
                continue;
            }
            let mut list: Vec<MapFile> = chosen.into_values().map(|(_, f)| f).collect();
            list.sort_by(|a, b| a.slot.cmp(&b.slot));
            let bytes = list.iter().map(|f| f.bytes).sum();
            variants
                .insert(format!("{res}-{fmt}"), Variant { bytes, payload: Payload::Maps { files: list } });
        }
    }
    if variants.is_empty() {
        return None;
    }

    let mut tags = meta.tags.clone();
    tags.extend(meta.categories.iter().cloned());
    tags.sort();
    tags.dedup();

    Some(Asset {
        id: format!("ph:{slug}"),
        source: "polyhaven".into(),
        kind: "material".into(),
        name: meta.name.clone(),
        dir: slug.to_string(),
        category: meta
            .category
            .clone()
            .or_else(|| meta.categories.first().cloned())
            .unwrap_or_default(),
        tags,
        authors: meta.authors.keys().cloned().collect(),
        license: "CC0".into(),
        page: format!("https://polyhaven.com/a/{slug}"),
        thumb: if meta.thumbnail_url.is_empty() {
            format!("https://cdn.polyhaven.com/asset_img/thumbs/{slug}.png?width=256&height=256")
        } else {
            meta.thumbnail_url.clone()
        },
        variants,
    })
}

// ------------------------------------------------------------------ ambientCG

/// ambientCG zip member suffix -> (canonical slot, preference).
pub fn acg_slot(stem: &str) -> Option<(&'static str, u8)> {
    let suffix = stem.rsplit('_').next().unwrap_or(stem);
    Some(match suffix {
        "Color" => (BASECOLOR, 0),
        "NormalGL" => (NORMAL, 0),
        "NormalDX" => (NORMAL, 1),
        "Roughness" => (ROUGHNESS, 0),
        "Metalness" => (METALLIC, 0),
        "Displacement" => (HEIGHT, 0),
        "AmbientOcclusion" => (AO, 0),
        "Opacity" => (OPACITY, 0),
        _ => return None,
    })
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct AcgAsset {
    asset_id: String,
    #[serde(default)]
    display_name: Option<String>,
    #[serde(default)]
    display_category: Option<String>,
    #[serde(default)]
    data_type: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    short_link: Option<String>,
    #[serde(default)]
    preview_image: Option<HashMap<String, String>>,
    #[serde(default)]
    download_folders: Value,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct AcgPage {
    number_of_results: usize,
    found_assets: Vec<AcgAsset>,
}

fn sync_ambientcg(agent: &ureq::Agent) -> Result<Vec<Asset>> {
    let mut assets = Vec::new();
    for kind in ["Material", "Decal", "Atlas", "Brush"] {
        let mut offset = 0usize;
        loop {
            let url = format!(
                "https://ambientcg.com/api/v2/full_json?type={kind}&limit=200&offset={offset}\
                 &include=downloadData,previewData"
            );
            let page: AcgPage = http::get_json(agent, &url)?;
            if page.found_assets.is_empty() {
                break;
            }
            for a in &page.found_assets {
                if let Some(asset) = acg_build(a) {
                    assets.push(asset);
                }
            }
            offset += 200;
            if offset >= page.number_of_results {
                break;
            }
        }
        eprintln!("[sync] ambientCG {kind}: {} total so far", assets.len());
    }
    Ok(assets)
}

/// `downloadFolders` nests dicts inside lists inconsistently between asset
/// types, so collect every download descriptor by walking the whole subtree.
fn collect_downloads(node: &Value, out: &mut Vec<(String, String, u64)>) {
    match node {
        Value::Object(map) => {
            let attr = map.get("attribute").and_then(Value::as_str);
            let path = map
                .get("fullDownloadPath")
                .or_else(|| map.get("downloadLink"))
                .and_then(Value::as_str);
            if let (Some(attr), Some(path)) = (attr, path) {
                let size = map.get("size").and_then(Value::as_u64).unwrap_or(0);
                out.push((attr.to_string(), path.to_string(), size));
                return;
            }
            for v in map.values() {
                collect_downloads(v, out);
            }
        }
        Value::Array(items) => {
            for v in items {
                collect_downloads(v, out);
            }
        }
        _ => {}
    }
}

fn acg_build(a: &AcgAsset) -> Option<Asset> {
    let mut downloads = Vec::new();
    collect_downloads(&a.download_folders, &mut downloads);

    let mut variants: BTreeMap<String, Variant> = BTreeMap::new();
    for (attr, url, size) in downloads {
        // Attributes look like "2K-JPG"; ignore anything else (PSD, USD, ...).
        let Some((res, fmt)) = attr.split_once('-') else { continue };
        let (res, fmt) = (res.to_lowercase(), fmt.to_lowercase());
        if !RESOLUTIONS.contains(&res.as_str()) || !FORMATS.contains(&fmt.as_str()) {
            continue;
        }
        variants
            .insert(format!("{res}-{fmt}"), Variant { bytes: size, payload: Payload::Zip { url } });
    }
    if variants.is_empty() {
        return None;
    }

    let kind = a.data_type.as_deref().unwrap_or("Material").to_lowercase();
    let thumb = a
        .preview_image
        .as_ref()
        .and_then(|m| {
            m.get("512-PNG")
                .or_else(|| m.get("256-PNG"))
                .or_else(|| m.get("1024-PNG"))
                .cloned()
        })
        .unwrap_or_default();

    Some(Asset {
        id: format!("acg:{}", a.asset_id),
        source: "ambientcg".into(),
        kind,
        name: a.display_name.clone().unwrap_or_else(|| a.asset_id.clone()),
        dir: a.asset_id.clone(),
        category: a.display_category.clone().unwrap_or_default(),
        tags: a.tags.clone(),
        authors: vec!["ambientCG".into()],
        license: "CC0".into(),
        page: a
            .short_link
            .clone()
            .unwrap_or_else(|| format!("https://ambientcg.com/a/{}", a.asset_id)),
        thumb,
        variants,
    })
}
