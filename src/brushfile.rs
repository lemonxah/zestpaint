//! Readers for the brush-set formats free packs are distributed in.
//!
//! Importing expands a set into the brush library — one folder per stamp — so
//! everything downstream (the browser, previews, tip loading) treats an
//! imported brush exactly like a generated one.
//!
//! Coverage convention throughout: **255 = full paint**, matching what the
//! tip sampler in `paint.rs` expects.

use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result, bail};

/// Extensions the importer accepts. Plain images go through the `image` crate;
/// the rest are brush *sets* holding many stamps.
pub const SUPPORTED: &[&str] = &[
    "abr", "gbr", "gih", // brush sets
    "png", "jpg", "jpeg", "tga", "bmp", "webp", "tif", "tiff", // single alphas
];

/// One stamp: greyscale coverage.
pub struct BrushAlpha {
    pub name: String,
    pub width: u32,
    pub height: u32,
    /// One byte per texel, 255 = full paint.
    pub gray: Vec<u8>,
}

impl BrushAlpha {
    fn is_sane(&self) -> bool {
        self.width > 0
            && self.height > 0
            && self.gray.len() == (self.width as usize * self.height as usize)
    }
}

/// Big-endian cursor that refuses to read past the end.
struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }
    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.remaining() < n {
            bail!("truncated: wanted {n} bytes, {} left", self.remaining());
        }
        let out = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_be_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_be_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_be_bytes(self.take(4)?.try_into().unwrap()))
    }
}

/// Read every stamp out of a brush file.
pub fn load(path: &Path) -> Result<Vec<BrushAlpha>> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let stem = path.file_stem().unwrap_or_default().to_string_lossy().to_string();

    match ext.as_str() {
        "gbr" => Ok(vec![parse_gbr(&mut Reader::new(&std::fs::read(path)?), &stem)?]),
        "gih" => parse_gih(&std::fs::read(path)?, &stem),
        "abr" => parse_abr(&std::fs::read(path)?, &stem),
        _ => {
            // Plain image: the alpha channel if it has one, else luminance.
            let img = image::open(path).with_context(|| format!("open {}", path.display()))?;
            let rgba = img.to_rgba8();
            let (width, height) = rgba.dimensions();
            let opaque = rgba.pixels().all(|p| p.0[3] == 255);
            let gray = rgba
                .pixels()
                .map(|p| {
                    if opaque {
                        // Luma601, the usual greyscale weighting.
                        ((p.0[0] as u32 * 299 + p.0[1] as u32 * 587 + p.0[2] as u32 * 114) / 1000)
                            as u8
                    } else {
                        p.0[3]
                    }
                })
                .collect();
            Ok(vec![BrushAlpha { name: stem, width, height, gray }])
        }
    }
}

// ------------------------------------------------------------------ GIMP GBR

/// GIMP single brush. Layout verified against GIMP 3.0's shipped brushes:
/// big-endian header_size / version / width / height / bytes-per-pixel, then
/// `GIMP` magic and spacing from v2, then a NUL-padded name filling the header.
fn parse_gbr(r: &mut Reader, fallback: &str) -> Result<BrushAlpha> {
    let header_size = r.u32()? as usize;
    let version = r.u32()?;
    let width = r.u32()?;
    let height = r.u32()?;
    let channels = r.u32()?;

    if !(1..=3).contains(&version) {
        bail!("unsupported GBR version {version}");
    }
    if width == 0 || height == 0 || width > 16384 || height > 16384 {
        bail!("implausible GBR size {width}x{height}");
    }

    let mut name = fallback.to_string();
    if version >= 2 {
        let magic = r.take(4)?;
        if magic != b"GIMP" {
            bail!("bad GBR magic");
        }
        let _spacing = r.u32()?;
        // Whatever is left of the header is the name.
        if header_size > r.pos {
            let raw = r.take(header_size - r.pos)?;
            let trimmed: Vec<u8> = raw.iter().copied().take_while(|b| *b != 0).collect();
            if !trimmed.is_empty() {
                name = String::from_utf8_lossy(&trimmed).to_string();
            }
        }
    } else if header_size > r.pos {
        r.take(header_size - r.pos)?;
    }

    let count = width as usize * height as usize;
    let gray = match channels {
        // Greyscale mask: the value *is* the coverage.
        1 => r.take(count)?.to_vec(),
        // RGBA brush: coverage lives in alpha.
        4 => r.take(count * 4)?.chunks_exact(4).map(|p| p[3]).collect(),
        other => bail!("unsupported GBR channel count {other}"),
    };
    Ok(BrushAlpha { name, width, height, gray })
}

// ------------------------------------------------------------------ GIMP GIH

/// GIMP brush pipe: a name line, a parameter line (`N ncells:N ...`), then N
/// concatenated GBR brushes.
fn parse_gih(data: &[u8], stem: &str) -> Result<Vec<BrushAlpha>> {
    let mut lines = data.splitn(3, |b| *b == b'\n');
    let name = lines.next().context("GIH missing name line")?;
    let params = lines.next().context("GIH missing parameter line")?;
    let body_start = name.len() + params.len() + 2;

    let params = String::from_utf8_lossy(params);
    let count: usize = params
        .split_whitespace()
        .next()
        .and_then(|n| n.parse().ok())
        .context("GIH cell count unreadable")?;
    let set_name = String::from_utf8_lossy(name).trim().to_string();

    let mut reader = Reader::new(&data[body_start.min(data.len())..]);
    let mut out = Vec::new();
    for i in 0..count {
        let label = format!("{set_name} {:02}", i + 1);
        match parse_gbr(&mut reader, &label) {
            Ok(mut brush) => {
                // Every cell in a pipe shares one name; disambiguate them.
                brush.name = label;
                out.push(brush);
            }
            Err(e) => {
                eprintln!("[brushfile] {stem}: stopped after {} cells: {e:#}", out.len());
                break;
            }
        }
    }
    if out.is_empty() {
        bail!("no usable cells in GIH");
    }
    Ok(out)
}

// ------------------------------------------------------------ Photoshop ABR

/// Photoshop brush set.
///
/// Versions 1 and 2 follow Adobe's published layout. Version 6 and up are *not*
/// publicly documented — only the outer `8BIM` block structure is. Rather than
/// hard-code a guessed field order for the sampled-brush payload, the header is
/// located by scanning for a bounds/depth/compression run that validates, which
/// fails loudly instead of emitting garbage.
fn parse_abr(data: &[u8], stem: &str) -> Result<Vec<BrushAlpha>> {
    let mut r = Reader::new(data);
    let version = r.u16()?;
    match version {
        1 | 2 => {
            let count = r.u16()? as usize;
            let mut out = Vec::new();
            for i in 0..count {
                let Ok(kind) = r.u16() else { break };
                let Ok(size) = r.u32() else { break };
                let Ok(block) = r.take(size as usize) else { break };
                // 1 = computed (a parametric dab, nothing to extract), 2 = sampled.
                if kind != 2 {
                    continue;
                }
                match parse_abr_sampled_v1(block, version, &format!("{stem} {:02}", i + 1)) {
                    Ok(b) => out.push(b),
                    Err(e) => eprintln!("[brushfile] {stem}: brush {i}: {e:#}"),
                }
            }
            if out.is_empty() {
                bail!("no sampled brushes found");
            }
            Ok(out)
        }
        6..=10 => {
            let _subversion = r.u16()?;
            let mut out = Vec::new();
            while r.remaining() > 12 {
                let tag = r.take(4)?;
                if tag != b"8BIM" {
                    break;
                }
                let key = r.take(4)?.to_vec();
                let length = r.u32()? as usize;
                let block = r.take(length.min(r.remaining()))?;
                if &key == b"samp" {
                    out.extend(parse_abr_samp(block, stem));
                }
            }
            if out.is_empty() {
                bail!(
                    "no sampled brushes recovered — ABR v{version} is undocumented and this \
                     file's layout was not recognised"
                );
            }
            Ok(out)
        }
        other => bail!("unsupported ABR version {other}"),
    }
}

/// Sampled brush body for ABR v1/v2.
fn parse_abr_sampled_v1(block: &[u8], version: u16, name: &str) -> Result<BrushAlpha> {
    let mut r = Reader::new(block);
    let _misc = r.u32()?;
    let _spacing = r.u16()?;
    if version == 2 {
        // UTF-16BE name, length in characters.
        let chars = r.u32()? as usize;
        r.take(chars * 2)?;
    }
    let _antialias = r.u8()?;
    // Short bounds, then the authoritative long bounds.
    r.take(8)?;
    let top = r.i32()?;
    let left = r.i32()?;
    let bottom = r.i32()?;
    let right = r.i32()?;
    let depth = r.u16()?;
    let compression = r.u8()?;
    decode_bitmap(&mut r, top, left, bottom, right, depth, compression, name)
}

/// Walk the `samp` block, which holds one length-prefixed brush after another.
fn parse_abr_samp(block: &[u8], stem: &str) -> Vec<BrushAlpha> {
    let mut r = Reader::new(block);
    let mut out = Vec::new();
    while r.remaining() > 8 {
        let Ok(size) = r.u32() else { break };
        let size = size as usize;
        if size == 0 || size > r.remaining() {
            break;
        }
        let Ok(entry) = r.take(size) else { break };
        // Entries are padded out to a four-byte boundary.
        let pad = (4 - (size % 4)) % 4;
        if pad > 0 {
            let _ = r.take(pad);
        }
        match parse_abr_sampled_v6(entry, &format!("{stem} {:02}", out.len() + 1)) {
            Ok(b) => out.push(b),
            Err(e) => eprintln!("[brushfile] {stem}: entry {}: {e:#}", out.len()),
        }
    }
    out
}

/// Sampled brush body for ABR v6+.
///
/// The fields between the brush id and the bitmap header vary between
/// subversions and are not documented, so locate the header by validation:
/// four plausible bounds, a known depth, a known compression flag, and enough
/// bytes left to hold the result.
fn parse_abr_sampled_v6(entry: &[u8], name: &str) -> Result<BrushAlpha> {
    let id_len = u32::from_be_bytes(entry.get(0..4).context("entry too short")?.try_into()?);
    let after_id = 4 + id_len as usize;
    if after_id >= entry.len() {
        bail!("brush id runs past the entry");
    }

    for start in after_id..entry.len().saturating_sub(19) {
        let mut r = Reader::new(&entry[start..]);
        let (Ok(top), Ok(left), Ok(bottom), Ok(right)) = (r.i32(), r.i32(), r.i32(), r.i32())
        else {
            continue;
        };
        let (Ok(depth), Ok(compression)) = (r.u16(), r.u8()) else { continue };
        let (w, h) = (right - left, bottom - top);
        if w <= 0 || h <= 0 || w > 16384 || h > 16384 {
            continue;
        }
        if !matches!(depth, 8 | 16) || compression > 1 {
            continue;
        }
        // Raw data must fit exactly-ish; RLE must at least have room for its
        // per-row length table.
        let needed = if compression == 0 {
            w as usize * h as usize * (depth as usize / 8)
        } else {
            h as usize * 2
        };
        if r.remaining() < needed {
            continue;
        }
        if let Ok(brush) = decode_bitmap(&mut r, top, left, bottom, right, depth, compression, name)
            && brush.is_sane()
        {
            return Ok(brush);
        }
    }
    bail!("could not locate a bitmap header")
}

/// Decode a brush bitmap, raw or PackBits-compressed, into 8-bit coverage.
#[allow(clippy::too_many_arguments)]
fn decode_bitmap(
    r: &mut Reader,
    top: i32,
    left: i32,
    bottom: i32,
    right: i32,
    depth: u16,
    compression: u8,
    name: &str,
) -> Result<BrushAlpha> {
    let width = (right - left).max(0) as usize;
    let height = (bottom - top).max(0) as usize;
    if width == 0 || height == 0 {
        bail!("empty bitmap bounds");
    }
    let bytes_per = (depth as usize / 8).max(1);
    let row_bytes = width * bytes_per;

    let raw: Vec<u8> = if compression == 0 {
        r.take(row_bytes * height)?.to_vec()
    } else {
        // Photoshop RLE: all row byte-counts first, then the packed rows.
        let mut lengths = Vec::with_capacity(height);
        for _ in 0..height {
            lengths.push(r.u16()? as usize);
        }
        let mut out = Vec::with_capacity(row_bytes * height);
        for len in lengths {
            let packed = r.take(len)?;
            let row = unpack_bits(packed, row_bytes)?;
            out.extend_from_slice(&row);
        }
        out
    };

    // 16-bit brushes keep the high byte; the tip only needs 8 bits.
    let gray: Vec<u8> = if bytes_per == 1 {
        raw
    } else {
        raw.chunks_exact(bytes_per).map(|c| c[0]).collect()
    };

    let brush = BrushAlpha { name: name.to_string(), width: width as u32, height: height as u32, gray };
    if !brush.is_sane() {
        bail!("decoded {} bytes for a {width}x{height} brush", brush.gray.len());
    }
    Ok(brush)
}

/// PackBits, as Photoshop uses it for one scanline.
fn unpack_bits(packed: &[u8], expected: usize) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(expected);
    let mut i = 0;
    while i < packed.len() && out.len() < expected {
        let n = packed[i] as i8;
        i += 1;
        if n >= 0 {
            let run = n as usize + 1;
            let end = (i + run).min(packed.len());
            out.extend_from_slice(&packed[i..end]);
            i = end;
        } else if n != -128 {
            let run = (1 - n as i32) as usize;
            let Some(&byte) = packed.get(i) else { break };
            i += 1;
            out.extend(std::iter::repeat_n(byte, run));
        }
    }
    out.resize(expected, 0);
    Ok(out)
}

// ---------------------------------------------------------------- importing

/// Expand a brush file into the brush library, one folder per stamp.
/// Returns the names written.
pub fn import(path: &Path) -> Result<Vec<String>> {
    let brushes = load(path)?;
    let set = crate::library::sanitize(
        &path.file_stem().unwrap_or_default().to_string_lossy(),
    );
    let root = crate::library::brush_dir();
    let mut written = Vec::new();

    for (i, brush) in brushes.iter().enumerate() {
        if !brush.is_sane() {
            continue;
        }
        let folder = if brushes.len() == 1 {
            set.clone()
        } else {
            format!("{set}_{:03}", i + 1)
        };
        let dir = root.join(&folder);
        std::fs::create_dir_all(&dir)?;

        let img = image::GrayImage::from_raw(brush.width, brush.height, brush.gray.clone())
            .context("brush data did not match its dimensions")?;
        // Coverage only: a stamp has no colour, and `tip_source` prefers opacity.
        img.save(dir.join("opacity.png"))?;

        let manifest = serde_json::json!({
            "id": format!("import:{folder}"),
            "name": brush.name,
            "source": "imported",
            "kind": "brush",
            "license": "unknown — see wherever you got the set",
            "authors": [],
            "page": "",
            "variant": format!("{}x{}", brush.width, brush.height),
            "tags": ["imported", "brush", "stamp"],
            "maps": { "opacity": "opacity.png" },
        });
        std::fs::write(dir.join("material.json"), serde_json::to_vec_pretty(&manifest)?)?;
        written.push(folder);
    }
    if written.is_empty() {
        bail!("nothing usable in {}", path.display());
    }
    Ok(written)
}

/// Every brush file directly inside a folder, for bulk imports.
pub fn files_in(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut out: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && p.extension()
                    .and_then(|e| e.to_str())
                    .map(|e| SUPPORTED.contains(&e.to_ascii_lowercase().as_str()))
                    .unwrap_or(false)
        })
        .collect();
    out.sort();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// GIMP ships real .gbr/.gih files; parse whatever is installed. This is
    /// the only brush format here that can be checked against genuine data.
    #[test]
    fn parses_installed_gimp_brushes() {
        let mut roots: Vec<PathBuf> = Vec::new();
        for base in ["/usr/share/gimp/3.0/brushes", "/usr/share/gimp/2.0/brushes"] {
            let p = PathBuf::from(base);
            if p.is_dir() {
                roots.push(p);
            }
        }
        if roots.is_empty() {
            eprintln!("no GIMP brushes installed; skipping");
            return;
        }

        let mut checked = 0;
        for root in roots {
            let Ok(dirs) = std::fs::read_dir(&root) else { continue };
            for dir in dirs.flatten() {
                for file in files_in(&dir.path()) {
                    let ext =
                        file.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
                    if ext != "gbr" && ext != "gih" {
                        continue;
                    }
                    let brushes = load(&file)
                        .unwrap_or_else(|e| panic!("{} failed to parse: {e:#}", file.display()));
                    assert!(!brushes.is_empty(), "{} produced no stamps", file.display());
                    for b in &brushes {
                        assert!(b.is_sane(), "{}: {}x{} vs {} bytes", file.display(), b.width, b.height, b.gray.len());
                        // ZESTPAINT_BRUSH_DUMP=/some/dir writes the decoded
                        // stamps out, for eyeballing a parser change.
                        if let Ok(dump) = std::env::var("ZESTPAINT_BRUSH_DUMP") {
                            let dir = PathBuf::from(&dump);
                            let _ = std::fs::create_dir_all(&dir);
                            let safe = crate::library::sanitize(&b.name);
                            if let Some(img) =
                                image::GrayImage::from_raw(b.width, b.height, b.gray.clone())
                            {
                                let _ = img.save(dir.join(format!("{safe}.png")));
                            }
                        }
                        assert!(!b.name.trim().is_empty(), "{}: unnamed stamp", file.display());
                        // A brush that is uniformly zero would stamp nothing.
                        assert!(
                            b.gray.iter().any(|v| *v > 0),
                            "{}: stamp is entirely empty",
                            file.display()
                        );
                    }
                    checked += 1;
                }
            }
        }
        assert!(checked > 0, "found GIMP brush dirs but parsed nothing");
        eprintln!("parsed {checked} GIMP brush files");
    }

    /// PackBits is shared by both ABR paths, so pin its behaviour directly.
    #[test]
    fn packbits_round_trips_literals_and_runs() {
        // 2 literals, then a run of 4 x 0xAB.
        let packed = [1u8, 0x10, 0x20, 253, 0xAB];
        let out = unpack_bits(&packed, 6).unwrap();
        assert_eq!(out, vec![0x10, 0x20, 0xAB, 0xAB, 0xAB, 0xAB]);

        // Short input is padded, never panics.
        assert_eq!(unpack_bits(&[1, 0x10], 4).unwrap(), vec![0x10, 0, 0, 0]);
        // 0x80 is a no-op marker.
        assert_eq!(unpack_bits(&[128, 0, 0x55], 2).unwrap(), vec![0x55, 0]);
    }

    /// A malformed file must be an error, never a panic or a bogus stamp.
    #[test]
    fn rejects_rubbish_without_panicking() {
        assert!(parse_abr(&[0, 99, 0, 1], "x").is_err(), "unknown version must fail");
        assert!(parse_abr(&[], "x").is_err());
        assert!(parse_gbr(&mut Reader::new(&[0, 0, 0, 4]), "x").is_err());
        // Plausible header, no pixel data.
        let mut bad = Vec::new();
        bad.extend_from_slice(&28u32.to_be_bytes());
        bad.extend_from_slice(&2u32.to_be_bytes());
        bad.extend_from_slice(&64u32.to_be_bytes());
        bad.extend_from_slice(&64u32.to_be_bytes());
        bad.extend_from_slice(&1u32.to_be_bytes());
        bad.extend_from_slice(b"GIMP");
        bad.extend_from_slice(&1u32.to_be_bytes());
        assert!(parse_gbr(&mut Reader::new(&bad), "x").is_err(), "missing pixels must fail");
    }
}
