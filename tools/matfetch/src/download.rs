use std::collections::HashMap;
use std::io::{BufReader, Read as _};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use anyhow::{Context as _, Result, bail};
use serde::{Deserialize, Serialize};

use crate::catalog::{Asset, Index, Payload, acg_slot};
use crate::http;

/// Written into every downloaded folder: doubles as the "already have it"
/// marker and as the attribution record for the CC0 sources.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Manifest {
    pub id: String,
    pub name: String,
    pub source: String,
    pub kind: String,
    pub license: String,
    pub authors: Vec<String>,
    pub page: String,
    pub variant: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub maps: HashMap<String, String>,
}

#[derive(Serialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum State {
    Queued,
    Running,
    Done,
    Failed,
}

#[derive(Serialize, Clone, Debug)]
pub struct Progress {
    pub state: State,
    pub done_bytes: u64,
    pub total_bytes: u64,
    pub error: Option<String>,
}

pub type Shared = Arc<Mutex<HashMap<String, Progress>>>;

/// Where an asset lands: materials feed the app's material library, everything
/// else (decals, atlases, brushes) is a stamp.
pub fn target_dir(root: &Path, asset: &Asset) -> PathBuf {
    let sub = if asset.is_brush() { "brushes" } else { "materials" };
    root.join(sub).join(&asset.dir)
}

/// Returns the variant already on disk, if any.
pub fn installed(root: &Path, asset: &Asset) -> Option<String> {
    let path = target_dir(root, asset).join("material.json");
    let text = std::fs::read_to_string(path).ok()?;
    let m: Manifest = serde_json::from_str(&text).ok()?;
    Some(m.variant)
}

fn fmt_of(variant: &str) -> &str {
    variant.split_once('-').map(|(_, f)| f).unwrap_or("jpg")
}

/// Download one asset at `variant` into the library, normalising map filenames.
pub fn fetch_one(
    agent: &ureq::Agent,
    root: &Path,
    asset: &Asset,
    variant: &str,
    shared: Option<&Shared>,
) -> Result<PathBuf> {
    let dest = target_dir(root, asset);
    let v = asset
        .variants
        .get(variant)
        .with_context(|| format!("{} has no {variant} variant", asset.id))?;

    let bump = |n: u64| {
        if let Some(s) = shared
            && let Some(p) = s.lock().unwrap().get_mut(&asset.id)
        {
            p.done_bytes += n;
        }
    };

    let staging = root.join(".matfetch/staging").join(&asset.dir);
    if staging.exists() {
        std::fs::remove_dir_all(&staging).ok();
    }
    std::fs::create_dir_all(&staging)?;

    let maps: HashMap<String, String> = match &v.payload {
        Payload::Maps { files } => {
            let mut out = HashMap::new();
            for f in files {
                let ext = fmt_of(variant);
                let filename = format!("{}.{ext}", f.slot);
                let n = http::download_to(agent, &f.url, &staging.join(&filename))?;
                bump(n);
                out.insert(f.slot.clone(), filename);
            }
            out
        }
        Payload::Zip { url } => {
            let zip_path = root.join(".matfetch/staging").join(format!("{}.zip", asset.dir));
            let n = http::download_to(agent, url, &zip_path)?;
            bump(n);
            let out = extract_zip(&zip_path, &staging)
                .with_context(|| format!("extract {}", zip_path.display()))?;
            std::fs::remove_file(&zip_path).ok();
            out
        }
    };

    if maps.is_empty() {
        std::fs::remove_dir_all(&staging).ok();
        bail!("{}: no usable maps in {variant}", asset.id);
    }

    let manifest = Manifest {
        id: asset.id.clone(),
        name: asset.name.clone(),
        source: asset.source.clone(),
        kind: asset.kind.clone(),
        license: asset.license.clone(),
        authors: asset.authors.clone(),
        page: asset.page.clone(),
        variant: variant.to_string(),
        tags: asset.tags.clone(),
        maps,
    };
    std::fs::write(staging.join("material.json"), serde_json::to_vec_pretty(&manifest)?)?;

    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if dest.exists() {
        std::fs::remove_dir_all(&dest).ok();
    }
    std::fs::rename(&staging, &dest)
        .with_context(|| format!("move into {}", dest.display()))?;
    Ok(dest)
}

/// Pull the maps we care about out of an ambientCG zip, renaming each to its
/// canonical slot name. Preview renders, USD/MTLX scene files and duplicate
/// normal conventions are dropped.
fn extract_zip(zip_path: &Path, dest: &Path) -> Result<HashMap<String, String>> {
    let file = std::fs::File::open(zip_path)?;
    let mut archive = zip::ZipArchive::new(BufReader::new(file))?;

    // slot -> (preference, entry index, extension)
    let mut chosen: HashMap<&'static str, (u8, usize, String)> = HashMap::new();
    let mut images: Vec<(usize, String, String)> = Vec::new();
    for i in 0..archive.len() {
        let entry = archive.by_index(i)?;
        if !entry.is_file() {
            continue;
        }
        let name = entry.name().to_string();
        let base = name.rsplit('/').next().unwrap_or(&name);
        let Some((stem, ext)) = base.rsplit_once('.') else { continue };
        let ext = ext.to_ascii_lowercase();
        if !matches!(ext.as_str(), "png" | "jpg" | "jpeg" | "tif" | "tiff") {
            continue;
        }
        if stem.to_ascii_uppercase().contains("PREVIEW") {
            continue;
        }
        images.push((i, ext.clone(), stem.to_ascii_lowercase()));
        let Some((slot, pref)) = acg_slot(stem) else { continue };
        match chosen.get(slot) {
            Some((existing, _, _)) if *existing <= pref => {}
            _ => {
                chosen.insert(slot, (pref, i, ext));
            }
        }
    }

    // ambientCG's Brush assets are not PBR sets: some are a single unsuffixed
    // grayscale PNG, others a pair of TIFFs (`_masked` and `_raw`). Fall back to
    // treating the best lone image as the stamp profile, preferring the masked
    // cut-out. Decals and atlases also ship a bare preview PNG, but those have
    // real maps alongside it, so this never fires for them.
    if chosen.is_empty()
        && let Some((index, ext, _)) = images
            .iter()
            .find(|(_, _, stem)| stem.contains("masked"))
            .or_else(|| images.first())
    {
        chosen.insert(crate::catalog::HEIGHT, (0, *index, ext.clone()));
    }

    let mut maps = HashMap::new();
    for (slot, (_, index, ext)) in chosen {
        let mut entry = archive.by_index(index)?;
        // Nothing downstream reads TIFF, so transcode those to PNG on the way in.
        let filename = if matches!(ext.as_str(), "tif" | "tiff") {
            let mut raw = Vec::new();
            entry.read_to_end(&mut raw)?;
            let name = format!("{slot}.png");
            image::load_from_memory(&raw)
                .with_context(|| format!("decode {slot} tiff"))?
                .save(dest.join(&name))?;
            name
        } else {
            let name = format!("{slot}.{ext}");
            let mut out = std::fs::File::create(dest.join(&name))?;
            std::io::copy(&mut entry, &mut out)?;
            name
        };
        maps.insert(slot.to_string(), filename);
    }
    Ok(maps)
}

// ------------------------------------------------------------- worker pool

struct Queue {
    items: Mutex<Vec<(String, String)>>,
    cv: Condvar,
}

/// A pool of download workers fed from a shared queue. Concurrency is kept
/// modest on purpose: these are free CC0 hosts paying the bandwidth bill.
pub struct Pool {
    queue: Arc<Queue>,
    pub shared: Shared,
    root: PathBuf,
    index: Arc<Index>,
    pending: Arc<AtomicUsize>,
}

impl Pool {
    pub fn new(root: PathBuf, index: Arc<Index>, jobs: usize) -> Arc<Self> {
        let pool = Arc::new(Pool {
            queue: Arc::new(Queue { items: Mutex::new(Vec::new()), cv: Condvar::new() }),
            shared: Arc::new(Mutex::new(HashMap::new())),
            root,
            index,
            pending: Arc::new(AtomicUsize::new(0)),
        });
        for _ in 0..jobs.max(1) {
            let p = Arc::clone(&pool);
            std::thread::spawn(move || p.worker());
        }
        pool
    }

    pub fn enqueue(&self, id: &str, variant: &str) {
        let Some(asset) = self.index.get(id) else { return };
        {
            let mut st = self.shared.lock().unwrap();
            if let Some(p) = st.get(id)
                && matches!(p.state, State::Queued | State::Running)
            {
                return;
            }
            let total = asset.variants.get(variant).map(|v| v.bytes).unwrap_or(0);
            st.insert(
                id.to_string(),
                Progress { state: State::Queued, done_bytes: 0, total_bytes: total, error: None },
            );
        }
        self.pending.fetch_add(1, Ordering::SeqCst);
        self.queue.items.lock().unwrap().push((id.to_string(), variant.to_string()));
        self.queue.cv.notify_one();
    }

    pub fn pending(&self) -> usize {
        self.pending.load(Ordering::SeqCst)
    }

    fn worker(&self) {
        let agent = http::agent();
        loop {
            let (id, variant) = {
                let mut items = self.queue.items.lock().unwrap();
                while items.is_empty() {
                    items = self.queue.cv.wait(items).unwrap();
                }
                items.remove(0)
            };
            let Some(asset) = self.index.get(&id) else {
                self.pending.fetch_sub(1, Ordering::SeqCst);
                continue;
            };
            if let Some(p) = self.shared.lock().unwrap().get_mut(&id) {
                p.state = State::Running;
            }
            let result = fetch_one(&agent, &self.root, asset, &variant, Some(&self.shared));
            let mut st = self.shared.lock().unwrap();
            if let Some(p) = st.get_mut(&id) {
                match result {
                    Ok(_) => {
                        p.state = State::Done;
                        p.done_bytes = p.total_bytes;
                    }
                    Err(e) => {
                        p.state = State::Failed;
                        p.error = Some(format!("{e:#}"));
                    }
                }
            }
            drop(st);
            self.pending.fetch_sub(1, Ordering::SeqCst);
        }
    }
}

pub fn human(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = bytes as f64;
    let mut i = 0;
    while v >= 1024.0 && i < UNITS.len() - 1 {
        v /= 1024.0;
        i += 1;
    }
    format!("{v:.1} {}", UNITS[i])
}

/// Attribution file for the whole library, since both sites appreciate credit
/// even though CC0 does not require it.
pub fn write_credits(root: &Path, index: &Index) -> Result<()> {
    let mut lines = vec![
        "# Credits".to_string(),
        String::new(),
        "All assets here are CC0 (public domain). Credit is not required, but".to_string(),
        "these are the people and projects that made them.".to_string(),
        String::new(),
    ];
    let mut by_author: HashMap<String, usize> = HashMap::new();
    for a in &index.assets {
        if installed(root, a).is_some() {
            for author in &a.authors {
                *by_author.entry(author.clone()).or_default() += 1;
            }
        }
    }
    let mut authors: Vec<_> = by_author.into_iter().collect();
    authors.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    for (author, n) in authors {
        lines.push(format!("- {author} — {n} assets"));
    }
    lines.push(String::new());
    lines.push("Sources: https://polyhaven.com  ·  https://ambientcg.com".into());
    std::fs::write(root.join("CREDITS.md"), lines.join("\n"))?;
    Ok(())
}
