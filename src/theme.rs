//! Design tokens for the whole UI, and the egui style built from them. Every
//! colour, radius and spacing the interface uses comes from here, so the look
//! is changed in one place rather than through hex literals in panel code.

use egui::{Color32, CornerRadius, FontFamily, FontId, Stroke, TextStyle};

const fn hex(rgb: u32) -> Color32 {
    Color32::from_rgb((rgb >> 16) as u8, (rgb >> 8) as u8, rgb as u8)
}

// Backgrounds, darkest first.
/// Menu bar, status bar.
pub const BG_CHROME: Color32 = hex(0x0B0C0E);
/// Window background behind the docks.
pub const BG_APP: Color32 = hex(0x111214);
/// Context bar, tool rail, library toolbar.
pub const BG_BAR: Color32 = hex(0x17181C);
/// Docks, dialogs, popovers.
pub const BG_PANEL: Color32 = hex(0x1A1B1F);
/// 3D viewport clear colour.
pub const BG_VIEWPORT: Color32 = hex(0x1E2024);
/// Inputs, selects, segmented-control troughs.
pub const BG_FIELD: Color32 = hex(0x0F1012);
/// Secondary buttons.
pub const BG_BUTTON: Color32 = hex(0x26282D);
/// Selected segment.
pub const BG_SEG_ON: Color32 = hex(0x2E3036);
/// Selected row or tile (paired with an accent border).
pub const BG_SELECTED: Color32 = hex(0x25261F);

// Lines.
/// Region borders.
pub const LINE: Color32 = hex(0x24262B);
/// Inner dividers.
pub const LINE_INNER: Color32 = hex(0x2A2C31);
/// Button borders.
pub const LINE_CONTROL: Color32 = hex(0x33363C);

// Text.
pub const TEXT: Color32 = hex(0xE4E5E9);
/// Active segment or tab label.
pub const TEXT_STRONG: Color32 = hex(0xF2F3F5);
/// Labels, section headers.
pub const TEXT_MUTED: Color32 = hex(0xA3A6AF);
/// Secondary info, shortcuts.
pub const TEXT_DIM: Color32 = hex(0x8D909A);

// Accent and states.
/// Active tool, selection border, primary button, slider fill, progress.
pub const ACCENT: Color32 = hex(0xF5C84B);
/// Text on the accent.
pub const ON_ACCENT: Color32 = hex(0x1A1500);
pub const WARN_BG: Color32 = hex(0x3A2612);
pub const WARN_FG: Color32 = hex(0xFFB366);
pub const OK_BG: Color32 = hex(0x16241A);
pub const OK_FG: Color32 = hex(0x9FD8A2);
pub const ERR_BG: Color32 = hex(0x2A1716);
pub const ERR_FG: Color32 = hex(0xFF9B94);

// Axes: the gizmo, and the left edge of X/Y/Z number fields.
pub const AXIS_X: Color32 = hex(0xE5534B);
pub const AXIS_Y: Color32 = hex(0x57AB5A);
pub const AXIS_Z: Color32 = hex(0x539BF5);

// Metrics.
pub const RADIUS_CONTROL: u8 = 5;
pub const RADIUS_ROW: u8 = 7;
pub const RADIUS_POPOVER: u8 = 10;
pub const DOCK_PADDING: f32 = 12.0;
/// Label column of a slider row.
pub const SLIDER_LABEL_WIDTH: f32 = 84.0;
/// Value column of a slider row.
pub const SLIDER_VALUE_WIDTH: f32 = 36.0;

// Type sizes.
pub const BODY_SIZE: f32 = 12.0;
pub const SECTION_SIZE: f32 = 11.0;
pub const TITLE_SIZE: f32 = 15.0;
pub const BADGE_SIZE: f32 = 9.5;

/// Font family names registered by `install_fonts`.
pub const SANS_SEMIBOLD: &str = "plex-sans-semibold";

/// Register IBM Plex Sans (body + semibold) and IBM Plex Mono as the UI fonts,
/// keeping egui's defaults behind them for glyphs Plex lacks (and the icon
/// font, which callers add themselves).
pub fn install_fonts(fonts: &mut egui::FontDefinitions) {
    let faces: [(&str, &'static [u8]); 3] = [
        ("plex-sans", include_bytes!("../assets/fonts/IBMPlexSans-Regular.ttf")),
        (SANS_SEMIBOLD, include_bytes!("../assets/fonts/IBMPlexSans-SemiBold.ttf")),
        ("plex-mono", include_bytes!("../assets/fonts/IBMPlexMono-Regular.ttf")),
    ];
    for (name, bytes) in faces {
        fonts
            .font_data
            .insert(name.to_owned(), std::sync::Arc::new(egui::FontData::from_static(bytes)));
    }
    fonts
        .families
        .entry(FontFamily::Proportional)
        .or_default()
        .insert(0, "plex-sans".to_owned());
    fonts
        .families
        .entry(FontFamily::Monospace)
        .or_default()
        .insert(0, "plex-mono".to_owned());
    fonts.families.insert(
        FontFamily::Name(SANS_SEMIBOLD.into()),
        vec!["plex-sans-semibold".to_owned(), "plex-sans".to_owned()],
    );
}

/// The semibold UI face at `size`.
pub fn semibold(size: f32) -> FontId {
    FontId::new(size, FontFamily::Name(SANS_SEMIBOLD.into()))
}

/// Uppercase, muted, semibold section header ("PAINT CHANNELS").
pub fn section_header(text: &str) -> egui::RichText {
    egui::RichText::new(text.to_uppercase())
        .font(semibold(SECTION_SIZE))
        .color(TEXT_MUTED)
        .extra_letter_spacing(SECTION_SIZE * 0.06)
}

/// Apply the tokens to an egui context: colours, radii, spacing and type.
pub fn apply(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();
    style.text_styles = [
        (TextStyle::Small, FontId::proportional(10.5)),
        (TextStyle::Body, FontId::proportional(BODY_SIZE)),
        (TextStyle::Button, FontId::proportional(BODY_SIZE)),
        (TextStyle::Monospace, FontId::monospace(BODY_SIZE)),
        (TextStyle::Heading, semibold(TITLE_SIZE)),
    ]
    .into();

    let spacing = &mut style.spacing;
    spacing.item_spacing = egui::vec2(8.0, 6.0);
    spacing.button_padding = egui::vec2(8.0, 4.0);
    spacing.interact_size.y = 24.0;
    spacing.slider_width = 120.0;
    spacing.window_margin = egui::Margin::same(DOCK_PADDING as i8);

    let v = &mut style.visuals;
    *v = egui::Visuals::dark();
    v.override_text_color = Some(TEXT);
    v.panel_fill = BG_PANEL;
    v.window_fill = BG_PANEL;
    v.extreme_bg_color = BG_FIELD;
    v.faint_bg_color = BG_BAR;
    v.code_bg_color = BG_FIELD;
    v.window_stroke = Stroke::new(1.0_f32, LINE);
    v.window_corner_radius = CornerRadius::same(RADIUS_POPOVER + 2);
    v.menu_corner_radius = CornerRadius::same(RADIUS_POPOVER);
    v.hyperlink_color = ACCENT;
    v.warn_fg_color = WARN_FG;
    v.error_fg_color = ERR_FG;
    v.selection.bg_fill = ACCENT.linear_multiply(0.35);
    v.selection.stroke = Stroke::new(1.0_f32, ACCENT);
    v.slider_trailing_fill = true;

    let radius = CornerRadius::same(RADIUS_CONTROL);
    let w = &mut v.widgets;
    w.noninteractive.bg_fill = BG_PANEL;
    w.noninteractive.weak_bg_fill = BG_PANEL;
    w.noninteractive.bg_stroke = Stroke::new(1.0_f32, LINE_INNER);
    w.noninteractive.fg_stroke = Stroke::new(1.0_f32, TEXT_MUTED);
    w.noninteractive.corner_radius = radius;

    w.inactive.bg_fill = BG_FIELD;
    w.inactive.weak_bg_fill = BG_BUTTON;
    w.inactive.bg_stroke = Stroke::new(1.0_f32, LINE_CONTROL);
    w.inactive.fg_stroke = Stroke::new(1.0_f32, TEXT);
    w.inactive.corner_radius = radius;

    w.hovered.bg_fill = BG_SEG_ON;
    w.hovered.weak_bg_fill = BG_SEG_ON;
    w.hovered.bg_stroke = Stroke::new(1.0_f32, TEXT_DIM);
    w.hovered.fg_stroke = Stroke::new(1.0_f32, TEXT_STRONG);
    w.hovered.corner_radius = radius;

    w.active.bg_fill = BG_SEG_ON;
    w.active.weak_bg_fill = BG_SEG_ON;
    w.active.bg_stroke = Stroke::new(1.0_f32, ACCENT);
    w.active.fg_stroke = Stroke::new(1.0_f32, TEXT_STRONG);
    w.active.corner_radius = radius;

    w.open.bg_fill = BG_SEG_ON;
    w.open.weak_bg_fill = BG_SEG_ON;
    w.open.bg_stroke = Stroke::new(1.0_f32, LINE_CONTROL);
    w.open.fg_stroke = Stroke::new(1.0_f32, TEXT_STRONG);
    w.open.corner_radius = radius;

    // egui keeps one style per light/dark theme and follows the system
    // theme by default; the design is dark-only, so pin it and set both.
    ctx.set_theme(egui::ThemePreference::Dark);
    ctx.set_style_of(egui::Theme::Light, style.clone());
    ctx.set_style_of(egui::Theme::Dark, style);
}
