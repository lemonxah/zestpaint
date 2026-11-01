//! Image plumbing for the assistant: API image blocks, SVG rasterising,
//! background removal, and writing a stamp into the brush library.

use std::collections::HashMap;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use anyhow::{Context as _, Result, bail};
use base64::Engine as _;
use image::{Rgba, RgbaImage};
use serde_json::{Value, json};

/// Downscale so the longer edge is at most `max`.
pub fn fit(img: &RgbaImage, max: u32) -> RgbaImage {
    let (w, h) = img.dimensions();
    if w.max(h) <= max {
        return img.clone();
    }
    let scale = max as f32 / w.max(h) as f32;
    image::imageops::resize(
        img,
        ((w as f32 * scale).round() as u32).max(1),
        ((h as f32 * scale).round() as u32).max(1),
        image::imageops::FilterType::Triangle,
    )
}

/// Composite over a solid background (straight alpha).
pub fn flatten(img: &RgbaImage, bg: [u8; 3]) -> image::RgbImage {
    image::RgbImage::from_fn(img.width(), img.height(), |x, y| {
        let p = img.get_pixel(x, y).0;
        let a = p[3] as f32 / 255.0;
        image::Rgb([0, 1, 2].map(|i| (p[i] as f32 * a + bg[i] as f32 * (1.0 - a)).round() as u8))
    })
}

/// A JPEG image block — viewport captures and textures are photos, and JPEG
/// keeps them a fraction of the PNG size.
pub fn jpeg_block(img: &RgbaImage, max: u32) -> Result<Value> {
    let rgb = flatten(&fit(img, max), [0, 0, 0]);
    let mut bytes = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut bytes, 85).encode_image(&rgb)?;
    Ok(block("image/jpeg", &bytes))
}

/// A PNG block of a stamp, flattened onto white so the design reads the way a
/// tattoo or splat reads on skin.
pub fn stamp_block(img: &RgbaImage, max: u32) -> Result<Value> {
    let rgb = flatten(&fit(img, max), [255, 255, 255]);
    let mut bytes = Vec::new();
    rgb.write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)?;
    Ok(block("image/png", &bytes))
}

fn block(media_type: &str, bytes: &[u8]) -> Value {
    json!({
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": base64::engine::general_purpose::STANDARD.encode(bytes),
        }
    })
}

/// Turn a design on a white page into ink with alpha ("colour to alpha"):
/// alpha is how far the darkest channel is from white, and colour is
/// un-blended from white, so anti-aliased edges and coloured ink survive.
pub fn white_to_alpha(img: &RgbaImage) -> RgbaImage {
    let mut out = img.clone();
    for p in out.pixels_mut() {
        let [r, g, b, a] = p.0.map(|c| c as f32 / 255.0);
        let alpha = (1.0 - r.min(g).min(b)).clamp(0.0, 1.0);
        if alpha < 1.0 / 255.0 {
            *p = Rgba([0, 0, 0, 0]);
            continue;
        }
        let un = |c: f32| ((c - (1.0 - alpha)) / alpha).clamp(0.0, 1.0);
        *p = Rgba([un(r), un(g), un(b), alpha * a].map(|c| (c * 255.0).round() as u8));
    }
    out
}

/// Crop to the non-transparent content plus a margin, so a stamp's
/// footprint is the design and not a mostly-empty page.
pub fn trim(img: &RgbaImage, margin: u32) -> RgbaImage {
    let (w, h) = img.dimensions();
    let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0, 0);
    for (x, y, p) in img.enumerate_pixels() {
        if p.0[3] > 8 {
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
        }
    }
    if x1 < x0 {
        return img.clone();
    }
    let x0 = x0.saturating_sub(margin);
    let y0 = y0.saturating_sub(margin);
    let x1 = (x1 + margin).min(w - 1);
    let y1 = (y1 + margin).min(h - 1);
    image::imageops::crop_imm(img, x0, y0, x1 - x0 + 1, y1 - y0 + 1).to_image()
}

fn fontdb() -> Arc<resvg::usvg::fontdb::Database> {
    static DB: OnceLock<Arc<resvg::usvg::fontdb::Database>> = OnceLock::new();
    DB.get_or_init(|| {
        let mut db = resvg::usvg::fontdb::Database::new();
        db.load_system_fonts();
        Arc::new(db)
    })
    .clone()
}

/// Rasterise SVG source so its longer edge is `size` pixels.
pub fn render_svg(svg: &str, size: u32) -> Result<RgbaImage> {
    let options = resvg::usvg::Options { fontdb: fontdb(), ..Default::default() };
    let tree = resvg::usvg::Tree::from_str(svg, &options).context("invalid SVG")?;
    let view = tree.size();
    if view.width() <= 0.0 || view.height() <= 0.0 {
        bail!("SVG has no size (give it a viewBox)");
    }
    let scale = size as f32 / view.width().max(view.height());
    let (w, h) = ((view.width() * scale).ceil() as u32, (view.height() * scale).ceil() as u32);
    let mut pixmap = resvg::tiny_skia::Pixmap::new(w.max(1), h.max(1)).context("SVG too large")?;
    resvg::render(&tree, resvg::tiny_skia::Transform::from_scale(scale, scale), &mut pixmap.as_mut());
    let mut out = RgbaImage::new(pixmap.width(), pixmap.height());
    for (dst, src) in out.pixels_mut().zip(pixmap.pixels()) {
        let c = src.demultiply();
        *dst = Rgba([c.red(), c.green(), c.blue(), c.alpha()]);
    }
    Ok(out)
}

/// Write an RGBA stamp into `<root>/brushes/<name>/`. Colour lives in the
/// RGB channels and coverage in alpha, which is what the tip loader reads.
pub fn save_stamp(
    root: &Path,
    name: &str,
    img: &RgbaImage,
    tags: Vec<String>,
    source: &str,
    description: String,
) -> Result<PathBuf> {
    let folder = crate::library::sanitize(name);
    let dir = root.join("brushes").join(&folder);
    std::fs::create_dir_all(&dir)?;
    img.save(dir.join("basecolor.png"))?;
    let manifest = matfetch::download::Manifest {
        id: format!("ai:{folder}"),
        name: name.to_string(),
        source: source.to_string(),
        kind: "brush".into(),
        license: "generated".into(),
        authors: vec!["zestsurface assistant".into()],
        page: String::new(),
        variant: format!("{}x{}", img.width(), img.height()),
        tags,
        maps: HashMap::from([("basecolor".to_string(), "basecolor.png".to_string())]),
    };
    let mut value = serde_json::to_value(&manifest)?;
    value["description"] = Value::String(description);
    std::fs::write(dir.join("material.json"), serde_json::to_vec_pretty(&value)?)?;
    Ok(dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn white_to_alpha_keeps_ink_and_drops_paper() {
        let mut img = RgbaImage::from_pixel(2, 1, Rgba([255, 255, 255, 255]));
        img.put_pixel(1, 0, Rgba([0, 0, 0, 255]));
        let out = white_to_alpha(&img);
        assert_eq!(out.get_pixel(0, 0).0[3], 0);
        assert_eq!(out.get_pixel(1, 0).0, [0, 0, 0, 255]);
        // Half-strength red ink on white keeps its hue.
        let pink = RgbaImage::from_pixel(1, 1, Rgba([255, 128, 128, 255]));
        let p = white_to_alpha(&pink).get_pixel(0, 0).0;
        assert!(p[0] > 240 && p[1] < 10 && (126..=130).contains(&p[3]), "{p:?}");
    }

    #[test]
    fn renders_svg_at_requested_size() {
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 100">
            <circle cx="50" cy="50" r="40" fill="#000"/></svg>"##;
        let img = render_svg(svg, 400).unwrap();
        assert_eq!(img.dimensions(), (400, 200));
        assert_eq!(img.get_pixel(100, 100).0[3], 255);
        assert_eq!(img.get_pixel(350, 100).0[3], 0);
        let trimmed = trim(&img, 0);
        assert!(trimmed.width() < 170 && trimmed.width() > 150, "{}", trimmed.width());
    }
}
