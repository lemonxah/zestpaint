//! UI icons, from the Lucide icon font (ISC — see
//! `assets/fonts/LUCIDE-LICENSE.txt`).
//!
//! egui's bundled fonts do not cover what a paint UI needs. The fullwidth `＋`
//! and `－` and the paintbrush `🖌` are absent from all four bundled faces, and
//! `▲` `▼` `→` exist only in Hack — which egui puts in the *monospace* family
//! only, so they render as tofu boxes in ordinary proportional text. Rather
//! than hunt for emoji that happen to be covered, use a real icon font.
//!
//! Every constant below is a Unicode private-use codepoint, so icons can never
//! collide with real text.

use std::sync::Arc;

use egui::{FontData, FontDefinitions, FontFamily};

/// Register the icon font. Call once, after the egui context exists.
pub fn install(ctx: &egui::Context) {
    let mut fonts = FontDefinitions::default();
    crate::theme::install_fonts(&mut fonts);
    fonts.font_data.insert(
        "lucide".to_owned(),
        Arc::new(FontData::from_static(include_bytes!("../assets/fonts/lucide.ttf"))),
    );

    for family in [
        FontFamily::Proportional,
        FontFamily::Monospace,
        FontFamily::Name(crate::theme::SANS_SEMIBOLD.into()),
    ] {
        let chain = fonts.families.entry(family).or_default();
        // Appended, not prepended: the private-use range never overlaps real
        // text, so the normal typeface keeps priority for everything else.
        chain.push("lucide".to_owned());
        // Hack carries arrows and geometric shapes Ubuntu-Light lacks. egui
        // leaves it out of the proportional chain, which is what turns `→`
        // into a box in labels; as a last resort it beats tofu.
        if !chain.iter().any(|f| f == "Hack") {
            chain.push("Hack".to_owned());
        }
    }
    ctx.set_fonts(fonts);
}

// Layer stack
/// plus
pub const ADD: &str = "\u{e13d}";
/// trash-2
pub const DELETE: &str = "\u{e18e}";
/// chevron-down
pub const MOVE_DOWN: &str = "\u{e06d}";
/// eye
pub const VISIBLE: &str = "\u{e0ba}";
/// eye-off
pub const HIDDEN: &str = "\u{e0bb}";

// Tools
/// rotate-3d
pub const ORBIT: &str = "\u{e2ea}";
/// brush
pub const PAINT: &str = "\u{e1d3}";
/// eraser
pub const ERASE: &str = "\u{e28f}";
/// hand (smudge pushes paint around)
pub const SMUDGE: &str = "\u{e1d7}";
/// pipette
pub const PICKER: &str = "\u{e13b}";
/// stamp
pub const STAMP: &str = "\u{e3bb}";
/// copy (clone)
pub const CLONE: &str = "\u{e09e}";

// Actions
/// folder-open
pub const OPEN: &str = "\u{e247}";
/// download
pub const EXPORT: &str = "\u{e0b2}";
/// x
pub const REMOVE: &str = "\u{e1b2}";
/// sun
pub const ENVIRONMENT: &str = "\u{e178}";
/// wand-sparkles
pub const BAKE: &str = "\u{e357}";
/// palette
pub const MATERIAL: &str = "\u{e1dd}";

// Shell
/// move — the Place tool
pub const PLACE: &str = "\u{e121}";
/// search
pub const SEARCH: &str = "\u{e151}";
/// undo
pub const UNDO: &str = "\u{e19b}";
/// redo
pub const REDO: &str = "\u{e143}";
/// layout-grid — UV checker view toggle
pub const UV_CHECKER: &str = "\u{e0ff}";
/// contrast — Show mask on model toggle
pub const MASK_VIEW: &str = "\u{e09d}";
/// circle — Override material toggle
pub const OVERRIDE: &str = "\u{e076}";
/// grip-vertical — layer drag handle
pub const GRIP: &str = "\u{e0eb}";
/// check
pub const CHECK: &str = "\u{e06c}";
/// alert-triangle
pub const WARN: &str = "\u{e193}";
/// circle-x
pub const ERROR: &str = "\u{e084}";
/// link
pub const LINK: &str = "\u{e102}";
/// unlink
pub const UNLINK: &str = "\u{e19c}";
/// chevron-right
pub const CHEVRON_RIGHT: &str = "\u{e06f}";
/// ellipsis — presets menu
pub const MORE: &str = "\u{e0b6}";
/// mouse-pointer-2 — the Select tool
pub const POINTER: &str = "\u{e1c3}";
/// image
pub const IMAGE: &str = "\u{e0f6}";
/// library
pub const LIBRARY: &str = "\u{e100}";
/// keyboard
pub const KEYBOARD: &str = "\u{e284}";
/// settings
pub const SETTINGS: &str = "\u{e154}";
/// sparkles
pub const ASSISTANT: &str = "\u{e412}";
/// panel-right-close
pub const PANEL_CLOSE: &str = "\u{e432}";

#[cfg(test)]
mod tests {
    /// Guards against a silent regression to boxes: every icon must be a
    /// single private-use codepoint that the bundled font actually draws.
    #[test]
    fn every_icon_is_a_real_glyph_in_the_font() {
        let font = include_bytes!("../assets/fonts/lucide.ttf");
        let face = ttf_parser::Face::parse(font, 0).expect("lucide.ttf should parse");
        let icons = [
            ("ADD", super::ADD),
            ("DELETE", super::DELETE),
            ("MOVE_DOWN", super::MOVE_DOWN),
            ("VISIBLE", super::VISIBLE),
            ("HIDDEN", super::HIDDEN),
            ("ORBIT", super::ORBIT),
            ("PAINT", super::PAINT),
            ("ERASE", super::ERASE),
            ("SMUDGE", super::SMUDGE),
            ("PICKER", super::PICKER),
            ("STAMP", super::STAMP),
            ("CLONE", super::CLONE),
            ("OPEN", super::OPEN),
            ("EXPORT", super::EXPORT),
            ("REMOVE", super::REMOVE),
            ("ENVIRONMENT", super::ENVIRONMENT),
            ("BAKE", super::BAKE),
            ("MATERIAL", super::MATERIAL),
            ("PLACE", super::PLACE),
            ("SEARCH", super::SEARCH),
            ("UNDO", super::UNDO),
            ("REDO", super::REDO),
            ("UV_CHECKER", super::UV_CHECKER),
            ("MASK_VIEW", super::MASK_VIEW),
            ("OVERRIDE", super::OVERRIDE),
            ("GRIP", super::GRIP),
            ("CHECK", super::CHECK),
            ("WARN", super::WARN),
            ("ERROR", super::ERROR),
            ("LINK", super::LINK),
            ("UNLINK", super::UNLINK),
            ("CHEVRON_RIGHT", super::CHEVRON_RIGHT),
            ("MORE", super::MORE),
            ("POINTER", super::POINTER),
            ("IMAGE", super::IMAGE),
            ("LIBRARY", super::LIBRARY),
            ("KEYBOARD", super::KEYBOARD),
            ("SETTINGS", super::SETTINGS),
            ("ASSISTANT", super::ASSISTANT),
            ("PANEL_CLOSE", super::PANEL_CLOSE),
        ];
        for (name, glyph) in icons {
            let mut chars = glyph.chars();
            let ch = chars.next().unwrap_or('\0');
            assert!(chars.next().is_none(), "{name} must be a single codepoint");
            assert!(
                ('\u{e000}'..='\u{f8ff}').contains(&ch),
                "{name} must live in the private-use area, got U+{:04X}",
                ch as u32
            );
            let id = face
                .glyph_index(ch)
                .unwrap_or_else(|| panic!("{name} (U+{:04X}) is not in lucide.ttf", ch as u32));
            // A mapped but empty glyph would still draw as nothing at all.
            assert!(
                face.glyph_bounding_box(id).is_some(),
                "{name} maps to an empty glyph"
            );
        }
    }

    /// The vector lockup still has to parse. About and the README show the
    /// master JPEG instead.
    #[test]
    fn logo_svg_rasterises() {
        let svg = include_str!("../assets/brand/zestsurface-logo.svg");
        let image = crate::assistant::images::render_svg(svg, 128).expect("logo svg");
        assert!(image.width() > 64, "logo width {}", image.width());
        assert!(image.height() > 64, "logo height {}", image.height());
    }
}

/// The ZestSurface app icon (`assets/icon/zestsurface.svg`, pre-rendered) as
/// straight RGBA at 64 or 256 px: (pixels, side).
pub fn app_icon(size: u32) -> Option<(Vec<u8>, u32)> {
    let png: &[u8] = match size {
        64 => include_bytes!("../assets/icon/zestsurface-64.png"),
        _ => include_bytes!("../assets/icon/zestsurface-256.png"),
    };
    let image = image::load_from_memory(png).ok()?.to_rgba8();
    let side = image.width();
    Some((image.into_raw(), side))
}
