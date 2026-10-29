//! Small widgets the redesigned panels are built from: segmented controls,
//! label | slider | value rows, section headers, badges, chips, axis fields
//! and tool buttons. All colours come from `crate::theme`.

use egui::{Align2, Color32, CornerRadius, FontId, Response, Sense, Stroke, StrokeKind, Ui};

use crate::theme;

/// A segmented choice: a dark trough holding one segment per option, the
/// selected one raised. Returns true when the selection changed.
pub fn segmented<T: PartialEq + Copy>(ui: &mut Ui, value: &mut T, options: &[(T, &str)]) -> bool {
    segmented_hints(ui, value, options, &[])
}

/// `segmented` with a hover text per option (parallel to `options`; missing
/// entries get none).
pub fn segmented_hints<T: PartialEq + Copy>(
    ui: &mut Ui,
    value: &mut T,
    options: &[(T, &str)],
    hints: &[&str],
) -> bool {
    let mut changed = false;
    egui::Frame::new()
        .fill(theme::BG_FIELD)
        .corner_radius(CornerRadius::same(theme::RADIUS_CONTROL + 1))
        .inner_margin(egui::Margin::same(2))
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing.x = 2.0;
            ui.horizontal(|ui| {
                for (i, (option, label)) in options.iter().enumerate() {
                    let on = *value == *option;
                    let text = egui::RichText::new(*label).color(if on {
                        theme::TEXT_STRONG
                    } else {
                        theme::TEXT_MUTED
                    });
                    let button = egui::Button::new(text)
                        .fill(if on { theme::BG_SEG_ON } else { Color32::TRANSPARENT })
                        .stroke(Stroke::NONE)
                        .corner_radius(CornerRadius::same(theme::RADIUS_CONTROL))
                        .min_size(egui::vec2(0.0, 20.0));
                    let mut response = ui.add(button);
                    if let Some(hint) = hints.get(i) {
                        response = response.on_hover_text(*hint);
                    }
                    if response.clicked() && !on {
                        *value = *option;
                        changed = true;
                    }
                }
            });
        });
    changed
}

/// A slider drawn to the design: thin trough, accent fill, round knob.
/// Logarithmic sliders map the track to the log of the range.
pub fn slider(ui: &mut Ui, value: &mut f32, range: std::ops::RangeInclusive<f32>, log: bool, width: f32) -> Response {
    let (lo, hi) = (*range.start(), *range.end());
    let to_t = |v: f32| -> f32 {
        if log && lo > 0.0 {
            ((v.max(lo)).ln() - lo.ln()) / (hi.ln() - lo.ln())
        } else {
            (v - lo) / (hi - lo)
        }
    };
    let from_t = |t: f32| -> f32 {
        let t = t.clamp(0.0, 1.0);
        if log && lo > 0.0 { (lo.ln() + t * (hi.ln() - lo.ln())).exp() } else { lo + t * (hi - lo) }
    };
    let height = 18.0;
    let (rect, mut response) = ui.allocate_exact_size(egui::vec2(width, height), Sense::click_and_drag());
    let knob = 6.5;
    let track = rect.shrink2(egui::vec2(knob, 0.0));
    if (response.dragged() || response.clicked())
        && let Some(pos) = response.interact_pointer_pos()
    {
        let t = (pos.x - track.left()) / track.width().max(1.0);
        let next = from_t(t);
        if next != *value {
            *value = next;
            response.mark_changed();
        }
    }
    if response.hovered() {
        // Fine adjustment with the wheel, like a spin box.
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll != 0.0 {
            let next = from_t(to_t(*value) + scroll * 0.0015);
            if next != *value {
                *value = next;
                response.mark_changed();
            }
        }
    }
    if ui.is_rect_visible(rect) {
        let enabled = ui.is_enabled();
        let fill = if enabled { theme::ACCENT } else { theme::TEXT_DIM };
        let painter = ui.painter();
        let y = rect.center().y;
        let groove = egui::Rect::from_min_max(
            egui::pos2(track.left(), y - 2.0),
            egui::pos2(track.right(), y + 2.0),
        );
        painter.rect_filled(groove, CornerRadius::same(2), theme::BG_SEG_ON);
        let x = track.left() + to_t(*value).clamp(0.0, 1.0) * track.width();
        painter.rect_filled(
            egui::Rect::from_min_max(groove.min, egui::pos2(x, groove.max.y)),
            CornerRadius::same(2),
            fill,
        );
        painter.circle(egui::pos2(x, y), knob, fill, Stroke::new(1.5_f32, theme::BG_PANEL));
        if response.has_focus() {
            painter.circle_stroke(egui::pos2(x, y), knob + 2.0, Stroke::new(1.0_f32, theme::ACCENT));
        }
    }
    response
}

/// One label | slider | value row, the design's three-column grid. The
/// value is an editable mono field.
pub struct SliderRow<'a> {
    label: &'a str,
    value: &'a mut f32,
    range: std::ops::RangeInclusive<f32>,
    decimals: usize,
    suffix: &'a str,
    log: bool,
    hint: Option<&'a str>,
    label_width: f32,
}

impl<'a> SliderRow<'a> {
    pub fn new(label: &'a str, value: &'a mut f32, range: std::ops::RangeInclusive<f32>) -> Self {
        Self {
            label,
            value,
            range,
            decimals: 2,
            suffix: "",
            log: false,
            hint: None,
            label_width: theme::SLIDER_LABEL_WIDTH,
        }
    }

    pub fn decimals(mut self, decimals: usize) -> Self {
        self.decimals = decimals;
        self
    }

    pub fn suffix(mut self, suffix: &'a str) -> Self {
        self.suffix = suffix;
        self
    }

    pub fn log(mut self) -> Self {
        self.log = true;
        self
    }

    pub fn hint(mut self, hint: &'a str) -> Self {
        self.hint = Some(hint);
        self
    }

    pub fn label_width(mut self, width: f32) -> Self {
        self.label_width = width;
        self
    }

    pub fn show(self, ui: &mut Ui) -> Response {
        let range = self.range.clone();
        let inner = ui.horizontal(|ui| {
            let colour = if ui.is_enabled() { theme::TEXT } else { theme::TEXT_DIM };
            let label = fixed(ui, self.label_width, |ui| {
                ui.add(egui::Label::new(egui::RichText::new(self.label).color(colour)).truncate())
            });
            if let Some(hint) = self.hint {
                label.on_hover_text(hint);
            }
            let value_width = theme::SLIDER_VALUE_WIDTH + 8.0 + 7.0 * self.suffix.chars().count() as f32;
            let gap = ui.spacing().item_spacing.x;
            let slider_width = (ui.available_width() - value_width - gap).max(40.0);
            let mut response = slider(ui, self.value, range.clone(), self.log, slider_width);
            let speed = (range.end() - range.start()) / 300.0;
            let field = ui.add_sized(
                [value_width, 20.0],
                egui::DragValue::new(self.value)
                    .range(range)
                    .speed(speed)
                    .fixed_decimals(self.decimals)
                    .suffix(self.suffix)
                    .custom_formatter(|v, _| format!("{v:.prec$}", prec = self.decimals)),
            );
            if field.changed() {
                response.mark_changed();
            }
            response
        });
        inner.inner
    }
}

/// Lay `add` out left-aligned in a fixed-width column (`add_sized` centres,
/// which breaks label columns).
pub fn fixed<R>(ui: &mut Ui, width: f32, add: impl FnOnce(&mut Ui) -> R) -> R {
    ui.allocate_ui_with_layout(
        egui::vec2(width, 22.0),
        egui::Layout::left_to_right(egui::Align::Center),
        |ui| {
            ui.set_min_width(width);
            add(ui)
        },
    )
    .inner
}

/// A button spanning the available width, its label centred.
pub fn wide(ui: &mut Ui, button: egui::Button<'_>) -> Response {
    let width = ui.available_width();
    ui.allocate_ui_with_layout(
        egui::vec2(width, 28.0),
        egui::Layout::centered_and_justified(egui::Direction::TopDown),
        |ui| ui.add(button),
    )
    .inner
}

/// "PAINT CHANNELS"-style header, with optional right-aligned muted text.
pub fn section(ui: &mut Ui, title: &str, right: Option<&str>) {
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        ui.label(theme::section_header(title));
        if let Some(right) = right {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(egui::RichText::new(right).color(theme::TEXT_DIM));
            });
        }
    });
    ui.add_space(2.0);
}

/// Section header with a right-aligned text button ("Reset"). True on click.
pub fn section_action(ui: &mut Ui, title: &str, action: &str) -> bool {
    ui.add_space(4.0);
    let clicked = ui
        .horizontal(|ui| {
            ui.label(theme::section_header(title));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.add(egui::Button::new(egui::RichText::new(action).color(theme::TEXT_MUTED)).frame(false))
                    .clicked()
            })
            .inner
        })
        .inner;
    ui.add_space(2.0);
    clicked
}

/// A thin divider in the inner line colour.
pub fn divider(ui: &mut Ui) {
    let width = ui.available_width();
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, 9.0), Sense::hover());
    ui.painter().hline(rect.x_range(), rect.center().y, Stroke::new(1.0_f32, theme::LINE_INNER));
}

/// A small mono tag ("COL", "EMISSION").
pub fn badge(ui: &mut Ui, text: &str, fg: Color32, bg: Color32) -> Response {
    let galley = ui.painter().layout_no_wrap(
        text.to_owned(),
        FontId::monospace(theme::BADGE_SIZE),
        fg,
    );
    let size = galley.size() + egui::vec2(8.0, 3.0);
    let (rect, response) = ui.allocate_exact_size(size, Sense::hover());
    ui.painter().rect_filled(rect, CornerRadius::same(3), bg);
    ui.painter().galley(rect.center() - galley.size() / 2.0, galley, fg);
    response
}

/// How a chip or status line reads.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tone {
    Ok,
    Warn,
    Error,
    Neutral,
}

impl Tone {
    pub fn colours(self) -> (Color32, Color32) {
        match self {
            Tone::Ok => (theme::OK_FG, theme::OK_BG),
            Tone::Warn => (theme::WARN_FG, theme::WARN_BG),
            Tone::Error => (theme::ERR_FG, theme::ERR_BG),
            Tone::Neutral => (theme::TEXT_MUTED, theme::BG_BUTTON),
        }
    }

    pub fn icon(self) -> &'static str {
        match self {
            Tone::Ok => crate::icons::CHECK,
            Tone::Warn => crate::icons::WARN,
            Tone::Error => crate::icons::ERROR,
            Tone::Neutral => crate::icons::CHEVRON_RIGHT,
        }
    }
}

/// A rounded, tinted chip of text ("Layer has no mask").
pub fn chip(ui: &mut Ui, text: &str, tone: Tone) -> Response {
    let (fg, bg) = tone.colours();
    egui::Frame::new()
        .fill(bg)
        .corner_radius(CornerRadius::same(theme::RADIUS_CONTROL))
        .inner_margin(egui::Margin::symmetric(8, 3))
        .show(ui, |ui| ui.label(egui::RichText::new(text).color(fg)))
        .response
}

/// A number field with an X/Y/Z-coloured left edge.
pub fn axis_drag(ui: &mut Ui, value: &mut f32, axis: usize, speed: f64, suffix: &str, width: f32) -> Response {
    let colour = [theme::AXIS_X, theme::AXIS_Y, theme::AXIS_Z][axis.min(2)];
    let response = ui.add_sized(
        [width, 22.0],
        egui::DragValue::new(value).speed(speed).fixed_decimals(if suffix == "°" { 0 } else { 2 }).suffix(suffix),
    );
    let rect = response.rect;
    ui.painter().rect_filled(
        egui::Rect::from_min_max(rect.min, egui::pos2(rect.min.x + 2.0, rect.max.y)),
        CornerRadius { nw: 3, sw: 3, ne: 0, se: 0 },
        colour,
    );
    response
}

/// 40x40 tool-rail button: icon with the key letter underneath.
pub fn tool_button(ui: &mut Ui, icon: &str, key: &str, active: bool) -> Response {
    let (rect, response) = ui.allocate_exact_size(egui::vec2(40.0, 40.0), Sense::click());
    let painter = ui.painter();
    let radius = CornerRadius::same(theme::RADIUS_ROW);
    if active {
        painter.rect(rect, radius, Color32::from_white_alpha(18), Stroke::new(1.5_f32, theme::ACCENT), StrokeKind::Inside);
    } else if response.hovered() {
        painter.rect_filled(rect, radius, Color32::from_white_alpha(10));
    }
    let colour = if active { theme::ACCENT } else { theme::TEXT_MUTED };
    painter.text(rect.center() - egui::vec2(0.0, 5.0), Align2::CENTER_CENTER, icon, FontId::proportional(17.0), colour);
    painter.text(
        rect.center() + egui::vec2(0.0, 12.0),
        Align2::CENTER_CENTER,
        key,
        FontId::monospace(9.0),
        if active { theme::ACCENT } else { theme::TEXT_DIM },
    );
    response
}

/// A square icon toggle (viewport view toggles).
pub fn icon_toggle(ui: &mut Ui, icon: &str, on: bool) -> Response {
    let (rect, response) = ui.allocate_exact_size(egui::vec2(28.0, 26.0), Sense::click());
    let radius = CornerRadius::same(theme::RADIUS_CONTROL);
    if on {
        ui.painter().rect_filled(rect, radius, theme::BG_SEG_ON);
    } else if response.hovered() {
        ui.painter().rect_filled(rect, radius, Color32::from_white_alpha(10));
    }
    ui.painter().text(
        rect.center(),
        Align2::CENTER_CENTER,
        icon,
        FontId::proportional(15.0),
        if on { theme::ACCENT } else { theme::TEXT_MUTED },
    );
    response
}

/// Filled accent button (the one primary action in a group).
pub fn primary_button(text: impl Into<String>) -> egui::Button<'static> {
    egui::Button::new(egui::RichText::new(text.into()).color(theme::ON_ACCENT).font(theme::semibold(theme::BODY_SIZE)))
        .fill(theme::ACCENT)
        .stroke(Stroke::NONE)
}

/// Outlined accent button ("Place in viewport").
pub fn accent_outline_button(text: impl Into<String>) -> egui::Button<'static> {
    egui::Button::new(egui::RichText::new(text.into()).color(theme::TEXT_STRONG))
        .fill(theme::BG_SELECTED)
        .stroke(Stroke::new(1.0_f32, theme::ACCENT))
}

/// Button in warn colours, for actions that throw work away.
pub fn warn_button(text: impl Into<String>) -> egui::Button<'static> {
    egui::Button::new(egui::RichText::new(text.into()).color(theme::WARN_FG))
        .fill(theme::WARN_BG)
        .stroke(Stroke::new(1.0_f32, theme::WARN_FG.linear_multiply(0.4)))
}

/// Button in error colours ("Remove").
pub fn danger_button(text: impl Into<String>) -> egui::Button<'static> {
    egui::Button::new(egui::RichText::new(text.into()).color(theme::ERR_FG))
        .fill(theme::ERR_BG)
        .stroke(Stroke::new(1.0_f32, theme::ERR_FG.linear_multiply(0.35)))
}

/// A mono key badge ("B", "Ctrl K").
pub fn key_badge(ui: &mut Ui, key: &str) -> Response {
    let galley = ui.painter().layout_no_wrap(key.to_owned(), FontId::monospace(10.5), theme::TEXT_DIM);
    let size = galley.size() + egui::vec2(10.0, 4.0);
    let (rect, response) = ui.allocate_exact_size(size, Sense::hover());
    ui.painter().rect_stroke(rect, CornerRadius::same(4), Stroke::new(1.0_f32, theme::LINE_CONTROL), StrokeKind::Inside);
    ui.painter().galley(rect.center() - galley.size() / 2.0, galley, theme::TEXT_DIM);
    response
}

/// A filled colour square (layer thumbnail stand-in, map-slot swatch).
pub fn swatch(ui: &mut Ui, size: f32, colour: Color32) -> Response {
    let (rect, response) = ui.allocate_exact_size(egui::vec2(size, size), Sense::click());
    ui.painter().rect(rect, CornerRadius::same(4), colour, Stroke::new(1.0_f32, theme::LINE_CONTROL), StrokeKind::Inside);
    response
}

/// Linear RGB 0..1 to an sRGB `Color32`.
pub fn linear_to_color32(rgb: [f32; 3]) -> Color32 {
    let encode = |v: f32| {
        let v = v.clamp(0.0, 1.0);
        let s = if v <= 0.003_130_8 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 };
        (s * 255.0).round() as u8
    };
    Color32::from_rgb(encode(rgb[0]), encode(rgb[1]), encode(rgb[2]))
}
