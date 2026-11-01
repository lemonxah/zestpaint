//! Assistant dock: the AI chat, and the editor side of its tools.
//!
//! The conversation runs on a worker thread (`crate::assistant::worker`).
//! Tool calls that change the document arrive here as `EditorCall`s and run
//! on the UI thread, inside `layout`, like any other UI action — so they land
//! in undo history the same way. Captures (the viewport image sent with a
//! message, or `capture_viewport`) wait until just after the frame is
//! presented, when the GPU is idle and the image holds the finished frame.

use std::sync::mpsc;

use egui::{Color32, CornerRadius, RichText, Stroke, Ui};
use glam::Vec2;
use serde_json::{Value, json};

use super::widgets;
use super::{AppState, Tool};
use crate::assistant::{self, Event, Settings, ToolOutput, worker};
use crate::icons;
use crate::layers::{BlendMode, MaskKind};
use crate::paint::PaintTarget;
use crate::scene::MapSlot;
use crate::theme;

enum ToolState {
    Running,
    Ok,
    Failed,
}

enum Entry {
    User { text: String, attachments: usize },
    Assistant(String),
    Thinking(String),
    Tool {
        id: String,
        name: String,
        detail: String,
        state: ToolState,
        summary: String,
        preview: Option<image::RgbaImage>,
        texture: Option<egui::TextureHandle>,
    },
    Notice { text: String, error: bool },
}

/// What is waiting for the post-present capture.
enum Capture {
    /// The user pressed Send: capture, then start the turn.
    Send(String),
    /// A `capture_viewport` tool call, or a tool answering with a screenshot
    /// of its result (`note` leads the reply).
    Tool { textures: bool, reply: mpsc::Sender<ToolOutput>, note: Option<String> },
}

pub(super) struct AssistantPanel {
    settings: Settings,
    settings_path: std::path::PathBuf,
    show_settings: bool,
    input: String,
    /// Restored into the input if the request fails outright.
    last_prompt: String,
    entries: Vec<Entry>,
    /// The API conversation. Held by the worker while a turn runs.
    messages: Vec<Value>,
    turn: Option<worker::Turn>,
    captures: Vec<Capture>,
    usage: (u64, u64, u64),
    comfy_status: String,
    /// Claude Code's way in (HTTP MCP endpoint + `zestsurface mcp` bridge), when on.
    bridge: Option<assistant::bridge::Bridge>,
    bridge_status: String,
    /// An `import_model` call waiting for its import to finish.
    import_reply: Option<mpsc::Sender<ToolOutput>>,
}

impl AssistantPanel {
    pub(super) fn new() -> Self {
        let settings_path = super::dialogs::config_file("assistant.json");
        let settings = Settings::load(&settings_path);
        let mut panel = Self {
            settings,
            settings_path,
            show_settings: false,
            input: String::new(),
            last_prompt: String::new(),
            entries: Vec::new(),
            messages: Vec::new(),
            turn: None,
            captures: Vec::new(),
            usage: (0, 0, 0),
            comfy_status: String::new(),
            bridge: None,
            bridge_status: String::new(),
            import_reply: None,
        };
        panel.set_bridge(panel.settings.claude_code_bridge);
        panel
    }

    fn set_bridge(&mut self, on: bool) {
        // Dropping the old one closes its ports and removes its discovery file.
        self.bridge = None;
        self.bridge_status.clear();
        if on {
            if self.settings.mcp_token.is_empty() {
                self.settings.mcp_token = assistant::bridge::new_token();
                if let Err(e) = self.settings.save(&self.settings_path) {
                    self.bridge_status = format!("Could not save the MCP token: {e:#}");
                }
            }
            let s = &self.settings;
            match assistant::bridge::Bridge::start(self.settings_path.clone(), s.mcp_port, &s.mcp_token) {
                Ok(bridge) => self.bridge = Some(bridge),
                Err(e) => self.bridge_status = format!("Bridge failed to start: {e:#}"),
            }
        }
    }

    /// Start or stop the MCP server and remember the choice for next launch.
    fn toggle_mcp(&mut self) -> anyhow::Result<()> {
        self.settings.claude_code_bridge = self.bridge.is_none();
        self.set_bridge(self.settings.claude_code_bridge);
        self.settings.save(&self.settings_path)
    }

    /// Serving MCP over HTTP (the bridge alone only reaches `zestsurface mcp`).
    fn mcp_serving(&self) -> bool {
        self.bridge.as_ref().is_some_and(|b| b.http.is_ok())
    }

    /// The one-time Claude Code registration for the HTTP endpoint.
    fn mcp_add_command(&self) -> String {
        format!(
            "claude mcp add --scope user --transport http zestsurface {} --header \"Authorization: Bearer {}\"",
            assistant::bridge::http_url(self.settings.mcp_port),
            self.settings.mcp_token
        )
    }

    fn busy(&self) -> bool {
        self.turn.is_some() || self.captures.iter().any(|c| matches!(c, Capture::Send(_)))
    }

    fn push_text(&mut self, text: &str, thinking: bool) {
        match (self.entries.last_mut(), thinking) {
            (Some(Entry::Assistant(s)), false) | (Some(Entry::Thinking(s)), true) => s.push_str(text),
            _ => self.entries.push(if thinking { Entry::Thinking(text.into()) } else { Entry::Assistant(text.into()) }),
        }
    }
}

/// sRGB hex → linear RGB, for fills (which composite in linear space).
fn hex_linear(hex: &str) -> anyhow::Result<[f32; 3]> {
    Ok(assistant::tools::hex_to_srgb(hex)?.map(super::srgb_component_to_linear))
}

fn unit(v: &Value) -> Option<f32> {
    v.as_f64().map(|f| (f as f32).clamp(0.0, 1.0))
}

fn find_named(items: &[crate::library::LibraryMaterial], name: &str) -> Option<usize> {
    let want = name.trim().to_lowercase();
    let sanitized = crate::library::sanitize(name).to_lowercase();
    items
        .iter()
        .position(|m| m.name.to_lowercase() == want || m.name.to_lowercase() == sanitized)
        .or_else(|| items.iter().position(|m| m.display.to_lowercase() == want))
        .or_else(|| items.iter().position(|m| m.name.to_lowercase().contains(&sanitized)))
}

impl AppState {
    // ---------------------------------------------------------------- events

    /// Drain the worker's events and answer editor calls. Runs at the top of
    /// every frame.
    pub(super) fn poll_assistant(&mut self) {
        let mut calls = Vec::new();
        let mut library_changed = false;
        let mut events: Vec<Event> = Vec::new();
        if let Some(turn) = self.assistant.turn.as_ref() {
            events.extend(turn.events.try_iter());
        }
        if let Some(bridge) = self.assistant.bridge.as_ref() {
            events.extend(bridge.events.try_iter());
        }
        {
            let panel = &mut self.assistant;
            for event in events {
                match event {
                    Event::Text(t) => panel.push_text(&t, false),
                    Event::Thinking(t) => panel.push_text(&t, true),
                    Event::ToolStart { id, name, detail } => panel.entries.push(Entry::Tool {
                        id,
                        name,
                        detail,
                        state: ToolState::Running,
                        summary: String::new(),
                        preview: None,
                        texture: None,
                    }),
                    Event::ToolDone { id, output } => {
                        for entry in panel.entries.iter_mut().rev() {
                            if let Entry::Tool { id: eid, state, summary, preview, .. } = entry
                                && *eid == id
                            {
                                *state = if output.is_error { ToolState::Failed } else { ToolState::Ok };
                                *summary = output.summary();
                                *preview = output.preview.clone();
                                break;
                            }
                        }
                    }
                    Event::ServerTool { name, detail } => panel.entries.push(Entry::Tool {
                        id: String::new(),
                        name,
                        detail,
                        state: ToolState::Ok,
                        summary: String::new(),
                        preview: None,
                        texture: None,
                    }),
                    Event::Editor(call) => calls.push(call),
                    Event::LibraryChanged => library_changed = true,
                    Event::Usage { input, output, cache_read } => {
                        panel.usage.0 += input;
                        panel.usage.1 += output;
                        panel.usage.2 += cache_read;
                    }
                    Event::Notice(text) => panel.entries.push(Entry::Notice { text, error: false }),
                    Event::Error(text) => {
                        panel.entries.push(Entry::Notice { text, error: true });
                        if panel.input.is_empty() {
                            panel.input = panel.last_prompt.clone();
                        }
                    }
                }
            }
        }
        if library_changed {
            self.rescan_libraries_now();
        }
        for call in calls {
            // Tools that change what the viewport shows answer with a
            // screenshot of the result, taken after this frame is drawn.
            if matches!(call.name.as_str(), "place_stamp" | "set_view" | "set_matcap") {
                let result = match call.name.as_str() {
                    "place_stamp" => self.tool_place_stamp(&call.input),
                    "set_view" => self.tool_set_view(&call.input),
                    _ => self.tool_set_matcap(&call.input),
                };
                match result {
                    Ok(note) if call.input["capture"].as_bool().unwrap_or(true) => {
                        self.assistant.captures.push(Capture::Tool { textures: false, reply: call.reply, note: Some(format!("{note} Screenshot:")) })
                    }
                    Ok(note) => {
                        let _ = call.reply.send(ToolOutput::text(note));
                    }
                    Err(e) => {
                        let _ = call.reply.send(ToolOutput::error(format!("{e:#}")));
                    }
                }
                continue;
            }
            if call.name == "import_model" {
                match self.begin_assistant_import(&call.input) {
                    Ok(()) => self.assistant.import_reply = Some(call.reply),
                    Err(e) => {
                        let _ = call.reply.send(ToolOutput::error(format!("{e:#}")));
                    }
                }
                continue;
            }
            if call.name == "capture_viewport" {
                let textures = call.input["include_textures"].as_bool().unwrap_or(false);
                self.assistant.captures.push(Capture::Tool { textures, reply: call.reply, note: None });
                continue;
            }
            let out = self.run_editor_tool(&call.name, &call.input).unwrap_or_else(|e| ToolOutput::error(format!("{e:#}")));
            let _ = call.reply.send(out);
        }
        // The turn is over once the worker has returned the conversation.
        if self.assistant.turn.as_ref().is_some_and(|t| t.handle.is_finished()) {
            let turn = self.assistant.turn.take().unwrap();
            // Late events (the final error/notice) are still in the channel.
            for event in turn.events.try_iter() {
                match event {
                    Event::Text(t) => self.assistant.push_text(&t, false),
                    Event::Notice(text) => self.assistant.entries.push(Entry::Notice { text, error: false }),
                    Event::Error(text) => {
                        self.assistant.entries.push(Entry::Notice { text, error: true });
                        if self.assistant.input.is_empty() {
                            self.assistant.input = self.assistant.last_prompt.clone();
                        }
                    }
                    _ => {}
                }
            }
            match turn.handle.join() {
                Ok(messages) => self.assistant.messages = messages,
                Err(_) => self.assistant.entries.push(Entry::Notice { text: "The assistant worker crashed.".into(), error: true }),
            }
        }
    }

    fn rescan_libraries_now(&mut self) {
        self.ui.library = crate::library::scan();
        self.ui.library_scanned = true;
        self.ui.library_selected = None;
        self.ui.library_thumbs.clear();
        if let Some(l) = self.ui.library_loader.as_mut() {
            l.reset();
        }
        self.ui.brushes = crate::library::scan_brushes();
        self.ui.brushes_scanned = true;
        self.ui.brush_selected = None;
        self.ui.brush_thumbs.clear();
        if let Some(l) = self.ui.brush_loader.as_mut() {
            l.reset();
        }
    }

    /// `import_image`: a local image as a stamp, a map layer, a fill layer or
    /// one of the material's own map slots.
    fn tool_import_image(&mut self, input: &Value) -> anyhow::Result<ToolOutput> {
        let raw = input["path"].as_str().unwrap_or("").trim();
        let path = match raw.strip_prefix("~/") {
            Some(rest) => std::env::var_os("HOME").map(std::path::PathBuf::from).unwrap_or_default().join(rest),
            None => std::path::PathBuf::from(raw),
        };
        anyhow::ensure!(path.is_file(), "no such file: {}", path.display());
        let file = path.file_name().unwrap_or_default().to_string_lossy().to_string();
        let name = input["name"].as_str().map(str::trim).filter(|n| !n.is_empty()).unwrap_or(&file).to_string();
        let kind = input["kind"].as_str().unwrap_or("base_color");
        let set_label = self.texture_sets[self.active_set].label();
        match input["as"].as_str().unwrap_or("") {
            "stamp" => {
                self.apply_brush_tip(&path, name.clone());
                anyhow::ensure!(self.brush.tip_name.as_deref() == Some(name.as_str()), "{}", self.ui.status);
                self.select_tool(Tool::Stamp);
                Ok(ToolOutput::text(format!(
                    "'{name}' is the stamp tip now (aspect {:.2}), stamp tool selected. Set colour/channels/size with set_brush, then place_stamp.",
                    self.brush.tip_aspect
                )))
            }
            "layer" => {
                let map = match kind {
                    "base_color" => crate::layers::MapKind::BaseColor,
                    "roughness" => crate::layers::MapKind::Roughness,
                    "metallic" => crate::layers::MapKind::Metallic,
                    "height" => crate::layers::MapKind::Height,
                    "normal" => crate::layers::MapKind::Normal,
                    "emissive" => crate::layers::MapKind::Emissive,
                    "opacity" => crate::layers::MapKind::Opacity,
                    other => anyhow::bail!("a layer can't hold a {other} map"),
                };
                let image = crate::library::load_map(&path)?;
                let layers = &mut self.texture_sets[self.active_set].layers;
                anyhow::ensure!(layers.import_map_layer(name.clone(), map, &image), "could not import {file} as a {kind} layer");
                let mut text = format!("Imported '{file}' as a {kind} layer '{name}' on {set_label} (UV space, the whole texture).");
                // A cutout material shows in-between opacity as solid (>= 0.5)
                // or hides it (< 0.5): sheer maps need the blended pass.
                if map == crate::layers::MapKind::Opacity
                    && has_partial_opacity(&image)
                    && self.set_material_surface(self.active_set, None, Some(true))
                {
                    text.push_str(" Its in-between values are see-through, so the material now uses the blended pass (Transparent).");
                }
                Ok(ToolOutput::text(text))
            }
            "fill_layer" => {
                let image = crate::library::load_map(&path)?;
                let layers = &mut self.texture_sets[self.active_set].layers;
                layers.add_layer();
                let target = layers.active;
                layers.layers[target].name = name.clone();
                layers.fill_layer_from_image(target, &image);
                Ok(ToolOutput::text(format!(
                    "'{name}' fills a new layer on {set_label}; set_layer can change its tiling, or its projection for wrapping it onto the model."
                )))
            }
            "material_map" => {
                let slot = match kind {
                    "base_color" => MapSlot::BaseColor,
                    "roughness" => MapSlot::Roughness,
                    "metallic" => MapSlot::Metallic,
                    "normal" => MapSlot::Normal,
                    "height" => MapSlot::Height,
                    "ao" => MapSlot::Ao,
                    "emissive" => MapSlot::Emission,
                    "emission_mask" => MapSlot::EmissionMask,
                    other => anyhow::bail!("materials have no {other} slot"),
                };
                let image = crate::library::load_map(&path)?;
                self.record_maps(self.active_set);
                self.apply_material_map(slot, Some(image));
                self.settle_history();
                Ok(ToolOutput::text(format!("'{file}' is now the {kind} map of {set_label}'s material (undoable).")))
            }
            other => anyhow::bail!("`as` must be stamp, layer, fill_layer or material_map, not {other:?}"),
        }
    }

    /// `set_material`: the blended-pass flag and opacity factor of the active
    /// set's material, or of each of `sets`.
    fn tool_set_material(&mut self, input: &Value) -> anyhow::Result<String> {
        let transparent = input["transparent"].as_bool();
        let opacity = input["opacity"].as_f64().map(|o| (o as f32).clamp(0.0, 1.0));
        anyhow::ensure!(transparent.is_some() || opacity.is_some(), "give transparent and/or opacity");
        let sets = match input.get("sets") {
            Some(_) => self.resolve_sets(input)?,
            None => vec![self.active_set],
        };
        let mut lines = Vec::new();
        for set in sets {
            self.set_material_surface(set, opacity, transparent);
            let label = self.texture_sets[set].label();
            let (o, t) = self.material_blend(set).unwrap_or((1.0, false));
            lines.push(format!("{label}: {} pass, opacity {o:.2}", if t { "blended (Transparent)" } else { "cutout" }));
        }
        Ok(format!("{} (undoable).", lines.join("; ")))
    }

    /// The (opacity, transparent) of `set`'s material.
    fn material_blend(&self, set: usize) -> Option<(f32, bool)> {
        let ts = self.texture_sets.get(set)?;
        let m = &self.models.iter().find(|m| m.name == ts.object && m.material_name == ts.material)?.material;
        Some((m.opacity, m.transparent))
    }

    /// Set the opacity factor and/or blended flag of every model of `set`'s
    /// material, as the Material tab's Surface section does (history picks
    /// the change up). Returns whether anything changed.
    fn set_material_surface(&mut self, set: usize, opacity: Option<f32>, transparent: Option<bool>) -> bool {
        let Some(ts) = self.texture_sets.get(set) else { return false };
        let (object, material) = (ts.object.clone(), ts.material.clone());
        let mut changed = false;
        let mut params = None;
        for model in self.models.iter_mut().filter(|m| m.name == object && m.material_name == material) {
            let m = &mut model.material;
            if let Some(o) = opacity.filter(|o| *o != m.opacity) {
                m.opacity = o;
                changed = true;
            }
            if let Some(t) = transparent.filter(|t| *t != m.transparent) {
                m.transparent = t;
                changed = true;
            }
            params.get_or_insert(m.surface_params());
        }
        if let (true, Some(p)) = (changed, params) {
            self.renderer.set_material_surface(set, p);
            self.pt_scene_stale = true;
        }
        changed
    }

    /// `place_stamp`: one stamp of the current tip at a viewport point given
    /// as fractions (0..1, from the top left of capture_viewport's image).
    fn tool_place_stamp(&mut self, input: &Value) -> anyhow::Result<String> {
        anyhow::ensure!(!self.pt_mode, "painting is off in path-traced mode: set_render_mode raster first");
        let (Some(x), Some(y)) = (input["x"].as_f64(), input["y"].as_f64()) else {
            anyhow::bail!("x and y (0..1 of the viewport) are required");
        };
        anyhow::ensure!((0.0..=1.0).contains(&x) && (0.0..=1.0).contains(&y), "x and y are fractions of the viewport, 0..1");
        anyhow::ensure!(self.brush.tip.is_some(), "no stamp tip: import_image as=stamp, use_library_brush or create_stamp_from_svg first");
        if let Some(size) = input["size"].as_f64() {
            self.brush.size = (size as f32).clamp(1.0, 2000.0);
        }
        if let Some(deg) = input["rotation_degrees"].as_f64() {
            self.brush.tip_rotation = (deg as f32).to_radians();
        }
        if let Some(opacity) = input["opacity"].as_f64() {
            self.brush.flow = (opacity as f32).clamp(0.0, 1.0);
        }
        let px = Vec2::new(x as f32 * self.viewport.size[0] as f32, y as f32 * self.viewport.size[1] as f32);
        let active = self.active_set;
        match self.pick_set(px) {
            Some(set) if set == active => {}
            Some(set) => anyhow::bail!(
                "({x:.3}, {y:.3}) is on '{}', but paint goes to the active set '{}': select_texture_set it, or isolate the target with view_pieces",
                self.texture_sets[set].label(),
                self.texture_sets[active].label()
            ),
            None => anyhow::bail!("nothing of the model is at ({x:.3}, {y:.3}); capture_viewport and aim at the surface"),
        }
        if self.tool != Tool::Stamp {
            self.select_tool(Tool::Stamp);
        }
        let before = self.texture_sets[active].layers.strokes_begun();
        self.place_stamp(px);
        anyhow::ensure!(self.texture_sets[active].layers.strokes_begun() > before, "{}", self.ui.status);
        let layers = &self.texture_sets[active].layers;
        Ok(format!(
            "Stamped '{}' at ({x:.3}, {y:.3}) on {} › layer '{}' (size {:.0}, rotation {:.0}°, opacity {:.2}). Undo reverts it.",
            self.brush.tip_name.as_deref().unwrap_or("tip"),
            self.texture_sets[active].label(),
            layers.active_layer().name,
            self.brush.size,
            self.brush.tip_rotation.to_degrees(),
            self.brush.flow,
        ))
    }

    /// `set_matcap`: add or change a matcap layer on the active set, or add
    /// one to each of `sets`.
    fn tool_set_matcap(&mut self, input: &Value) -> anyhow::Result<String> {
        use crate::matcap::{MatcapBlend, BUILTINS};
        let named = self.resolve_sets(input)?;
        let many = !named.is_empty();
        let sets = if many { named } else { vec![self.active_set] };
        let look = if let Some(name) = input["builtin"].as_str() {
            let (canonical, _, blend) = *BUILTINS.iter().find(|b| b.0.eq_ignore_ascii_case(name)).ok_or_else(|| {
                let names: Vec<&str> = BUILTINS.iter().map(|b| b.0).collect();
                anyhow::anyhow!("no built-in matcap {name:?}; choose from {names:?}")
            })?;
            Some((canonical.to_string(), crate::matcap::builtin(canonical).unwrap(), Some(blend)))
        } else if let Some(path) = input["image_path"].as_str() {
            let path = std::path::Path::new(path.trim());
            let image = crate::library::load_map(path)?;
            Some((path.file_stem().unwrap_or_default().to_string_lossy().to_string(), image, None))
        } else {
            None
        };
        let blend = match input["blend"].as_str() {
            Some(b) => Some(MatcapBlend::parse(b).ok_or_else(|| anyhow::anyhow!("blend is add, multiply, replace or screen"))?),
            None => None,
        };
        let fill = match input["fill"].as_str() {
            None => None,
            Some("white") => Some(1.0),
            Some("black") => Some(0.0),
            Some(other) => anyhow::bail!("fill is white or black, not {other:?}"),
        };
        let new_layer = input["new_layer"].as_bool().unwrap_or(false);

        let mut lines = Vec::new();
        for set in sets {
            let label = self.texture_sets[set].label();
            let layers = &mut self.texture_sets[set].layers;
            let created;
            let index = match input["layer"].as_u64() {
                Some(_) if many => anyhow::bail!("`layer` picks a layer of the active set; leave it out when giving `sets`"),
                Some(i) => {
                    let i = i as usize;
                    anyhow::ensure!(layers.layers.get(i).is_some_and(|l| l.is_matcap()), "layer {i} is not a matcap layer");
                    created = false;
                    i
                }
                None if !many && !new_layer && layers.active_layer().is_matcap() => {
                    created = false;
                    layers.active
                }
                None => {
                    layers.add_shader_mask_layer(crate::layers::ShaderMask::Matcap);
                    created = true;
                    layers.active
                }
            };
            layers.active = index;
            // A new layer the user hasn't painted yet shows everywhere.
            if let Some(v) = fill.or(created.then_some(1.0)) {
                layers.fill_active(Some([v; 3]), None, None);
            }
            let layer = &mut layers.layers[index];
            if let Some((name, image, natural)) = &look {
                layer.set_matcap(&self.gpu, name.clone(), image.clone(), *natural);
            }
            let m = layer.matcap.get_or_insert_with(Default::default);
            if let Some(blend) = blend {
                m.blend = blend;
            }
            if let Some(strength) = input["strength"].as_f64() {
                m.strength = (strength as f32).clamp(0.0, 2.0);
            }
            if let Some(name) = input["name"].as_str() {
                layer.name = name.to_string();
            }
            if let Some(visible) = input["visible"].as_bool() {
                layer.visible = visible;
            }
            let m = layer.matcap.as_ref().unwrap();
            let state = match &m.image {
                Some(_) => format!("{}, {}, strength {:.2}", m.name, m.blend.label(), m.strength),
                None => "no matcap picked yet (give builtin or image_path)".into(),
            };
            lines.push(format!(
                "{label}: {} layer {index} '{}' — {state}{}{}.",
                if created { "new matcap" } else { "matcap" },
                layer.name,
                if layer.visible { "" } else { ", hidden" },
                match fill.or(created.then_some(1.0)) {
                    Some(v) if v > 0.5 => ", mask filled white",
                    Some(_) => ", mask filled black",
                    None => "",
                },
            ));
            layers.mark_dirty();
        }
        // The screenshot should show the matcaps, not a mask overlay.
        self.ui.mask_overlay = false;
        let mut out = lines.join("\n");
        if self.pt_mode {
            out.push_str("\nNote: matcaps only show in raster mode (set_render_mode raster).");
        }
        Ok(out)
    }

    /// `set_view`: aim the camera — a preset side, explicit angles, zoom, and
    /// framing pieces.
    fn tool_set_view(&mut self, input: &Value) -> anyhow::Result<String> {
        use std::f32::consts::{FRAC_PI_2, PI};
        if let Some(target) = input.get("frame") {
            let sets = match target {
                Value::String(s) if s == "all" => (0..self.texture_sets.len()).filter(|&i| self.texture_sets[i].visible).collect(),
                Value::String(s) if s == "selection" => self.selection_or_active(),
                Value::Array(_) => self.resolve_sets(&json!({"sets": target}))?,
                _ => self.resolve_sets(&json!({"sets": [target]}))?,
            };
            self.frame_sets(&sets);
        }
        if let Some(preset) = input["preset"].as_str() {
            let (yaw, pitch) = match preset {
                "front" => (0.0, 0.0),
                "back" => (PI, 0.0),
                "right" => (FRAC_PI_2, 0.0),
                "left" => (-FRAC_PI_2, 0.0),
                "top" => (0.0, 1.54),
                "bottom" => (0.0, -1.54),
                "three_quarter" => (0.6, 0.4),
                other => anyhow::bail!("unknown preset {other:?}"),
            };
            self.camera.yaw = yaw;
            self.camera.pitch = pitch;
        }
        if let Some(yaw) = input["yaw_degrees"].as_f64() {
            self.camera.yaw = (yaw as f32).to_radians();
        }
        if let Some(pitch) = input["pitch_degrees"].as_f64() {
            self.camera.pitch = (pitch as f32).to_radians().clamp(-1.54, 1.54);
        }
        if let Some(zoom) = input["zoom"].as_f64() {
            anyhow::ensure!(zoom > 0.0, "zoom must be positive");
            self.camera.distance = (self.camera.distance / zoom as f32).clamp(0.01, 1e6);
        }
        self.paint_projection = None;
        Ok(format!(
            "Camera: yaw {:.0}°, pitch {:.0}°, distance {:.3}.",
            self.camera.yaw.to_degrees(),
            self.camera.pitch.to_degrees(),
            self.camera.distance
        ))
    }

    fn begin_assistant_import(&mut self, input: &Value) -> anyhow::Result<()> {
        let path = std::path::PathBuf::from(input["path"].as_str().unwrap_or("").trim());
        anyhow::ensure!(path.is_file(), "no such file: {}", path.display());
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
        anyhow::ensure!(matches!(ext.as_str(), "fbx" | "gltf" | "glb"), "only .fbx, .gltf and .glb models can be imported");
        anyhow::ensure!(self.import.is_none() && self.assistant.import_reply.is_none(), "an import is already running");
        self.start_model_import(path);
        Ok(())
    }

    /// The import an `import_model` call started has finished.
    pub(super) fn answer_assistant_import(&mut self, result: Result<String, String>) {
        let Some(reply) = self.assistant.import_reply.take() else { return };
        let out = match result {
            Ok(status) => {
                let mut out = ToolOutput::text(format!("{status}. This replaced the previous document; layers start empty."));
                out.content.push(json!({"type": "text", "text": serde_json::to_string_pretty(&self.editor_state()).unwrap_or_default()}));
                out
            }
            Err(e) => ToolOutput::error(e),
        };
        let _ = reply.send(out);
    }

    // -------------------------------------------------------------- captures

    /// Run waiting captures. Called right after the frame is presented and
    /// waited on, so the scene image and the composites are finished.
    pub(super) fn service_assistant_captures(&mut self) {
        if self.assistant.captures.is_empty() {
            return;
        }
        for capture in std::mem::take(&mut self.assistant.captures) {
            match capture {
                Capture::Send(text) => {
                    let mut content = Vec::new();
                    let mut attached = 0;
                    if self.assistant.settings.attach_viewport {
                        match self.capture_attachments(false) {
                            Ok(blocks) => {
                                attached += blocks.iter().filter(|b| b["type"] == "image").count();
                                content.extend(blocks);
                            }
                            Err(e) => self.assistant.entries.push(Entry::Notice { text: format!("Viewport capture failed: {e:#}"), error: true }),
                        }
                    }
                    if self.assistant.settings.attach_textures {
                        match self.texture_attachments() {
                            Ok(blocks) => {
                                attached += blocks.iter().filter(|b| b["type"] == "image").count();
                                content.extend(blocks);
                            }
                            Err(e) => self.assistant.entries.push(Entry::Notice { text: format!("Texture capture failed: {e:#}"), error: true }),
                        }
                    }
                    content.push(json!({"type": "text", "text": text}));
                    if let Some(Entry::User { attachments, .. }) =
                        self.assistant.entries.iter_mut().rev().find(|e| matches!(e, Entry::User { .. }))
                    {
                        *attachments = attached;
                    }
                    self.start_assistant_turn(content);
                }
                Capture::Tool { textures, reply, note } => {
                    let out = match self.capture_attachments(textures) {
                        Ok(mut blocks) => {
                            if let Some(note) = note {
                                blocks.insert(0, json!({"type": "text", "text": note}));
                            }
                            ToolOutput { content: blocks, is_error: false, preview: None }
                        }
                        Err(e) => ToolOutput::error(format!("Capture failed: {e:#}")),
                    };
                    let _ = reply.send(out);
                }
            }
        }
    }

    /// The viewport (and optionally the textures) as labelled image blocks.
    fn capture_attachments(&mut self, textures: bool) -> anyhow::Result<Vec<Value>> {
        let shot = self.read_viewport()?;
        let set = &self.texture_sets[self.active_set];
        let mut blocks = vec![
            json!({"type": "text", "text": format!(
                "[Viewport capture: {}x{}, {} mode, active texture set '{} / {}', active layer '{}']",
                shot.width(), shot.height(),
                if self.pt_mode { "path-traced" } else { "raster" },
                set.object, set.material, set.layers.active_layer().name,
            )}),
            assistant::images::jpeg_block(&shot, 1280)?,
        ];
        if textures {
            blocks.extend(self.texture_attachments()?);
        }
        Ok(blocks)
    }

    /// The active set's material base colour texture and the composite of
    /// its painted layers, in UV space.
    fn texture_attachments(&mut self) -> anyhow::Result<Vec<Value>> {
        let mut blocks = Vec::new();
        let (object, material) = {
            let set = &self.texture_sets[self.active_set];
            (set.object.clone(), set.material.clone())
        };
        let base = self
            .models
            .iter()
            .find(|m| m.name == object && m.material_name == material)
            .and_then(|m| m.material.base_color_image.as_ref())
            .and_then(|img| image::RgbaImage::from_raw(img.width, img.height, img.rgba.to_vec()));
        if let Some(base) = base {
            blocks.push(json!({"type": "text", "text": format!("[Material base colour texture of '{object} / {material}' (UV space)]")}));
            blocks.push(assistant::images::jpeg_block(&base, 1024)?);
        }

        let layers = &self.texture_sets[self.active_set].layers;
        if layers.composite_basecolor.image().extent()[0] == layers.resolution {
            let texels = crate::export::read_texels(&self.gpu, &layers.composite_basecolor)?;
            let res = layers.resolution;
            let rgba: Vec<u8> = texels
                .iter()
                .flat_map(|[r, g, b, a]| {
                    // Premultiplied linear → straight sRGB.
                    let un = |c: f32| if *a > 1e-4 { c / a } else { 0.0 };
                    let enc = |c: f32| {
                        let c = c.clamp(0.0, 1.0);
                        let s = if c <= 0.0031308 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 };
                        (s * 255.0).round() as u8
                    };
                    [enc(un(*r)), enc(un(*g)), enc(un(*b)), (a.clamp(0.0, 1.0) * 255.0).round() as u8]
                })
                .collect();
            if let Some(img) = image::RgbaImage::from_raw(res, res, rgba) {
                blocks.push(json!({"type": "text", "text": "[Painted layers composite, base colour (UV space; black = unpainted)]"}));
                blocks.push(assistant::images::jpeg_block(&img, 1024)?);
            }
        }
        Ok(blocks)
    }

    /// Read the offscreen scene image back (R8G8B8A8, already sRGB-encoded).
    fn read_viewport(&self) -> anyhow::Result<image::RgbaImage> {
        use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
        use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyImageToBufferInfo, PrimaryCommandBufferAbstract as _};
        use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
        use vulkano::sync::GpuFuture as _;

        let image = self.viewport.scene_image.image().clone();
        let [w, h, _] = image.extent();
        let buffer: Subbuffer<[u8]> = Buffer::new_slice(
            self.gpu.memory_allocator.clone(),
            BufferCreateInfo { usage: BufferUsage::TRANSFER_DST, ..Default::default() },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            w as u64 * h as u64 * 4,
        )?;
        let mut builder = AutoCommandBufferBuilder::primary(
            self.gpu.command_buffer_allocator.clone(),
            self.gpu.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;
        builder.copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(image, buffer.clone()))?;
        builder.build()?.execute(self.gpu.queue.clone())?.then_signal_fence_and_flush()?.wait(None)?;
        let mut pixels = buffer.read()?.to_vec();
        for p in pixels.chunks_exact_mut(4) {
            p[3] = 255;
        }
        image::RgbaImage::from_raw(w, h, pixels).ok_or_else(|| anyhow::anyhow!("bad capture size"))
    }

    fn start_assistant_turn(&mut self, content: Vec<Value>) {
        let messages = std::mem::take(&mut self.assistant.messages);
        self.assistant.turn = Some(worker::start_turn(self.assistant.settings.clone(), messages, content));
    }

    fn send_assistant_message(&mut self) {
        let text = self.assistant.input.trim().to_string();
        if text.is_empty() || self.assistant.busy() {
            return;
        }
        self.assistant.input.clear();
        self.assistant.last_prompt = text.clone();
        self.assistant.entries.push(Entry::User { text: text.clone(), attachments: 0 });
        if self.assistant.settings.attach_viewport || self.assistant.settings.attach_textures {
            self.assistant.captures.push(Capture::Send(text));
        } else {
            self.start_assistant_turn(vec![json!({"type": "text", "text": text})]);
        }
    }

    // ---------------------------------------------------------- editor tools

    fn run_editor_tool(&mut self, name: &str, input: &Value) -> anyhow::Result<ToolOutput> {
        match name {
            "get_editor_state" => Ok(ToolOutput::json(&self.editor_state())),
            "view_pieces" => self.view_pieces(input),
            "import_image" => self.tool_import_image(input),
            "set_material" => self.tool_set_material(input).map(ToolOutput::text),
            "set_render_mode" => {
                match input["mode"].as_str().unwrap_or("") {
                    "raster" => self.pt_mode = false,
                    "path_traced" if !self.pt_mode => {
                        self.pt_mode = true;
                        self.pt_scene_stale = true;
                    }
                    "path_traced" => {}
                    other => anyhow::bail!("unknown mode {other:?}"),
                }
                Ok(ToolOutput::text(if self.pt_mode {
                    "Path traced. It refines over a few seconds — wait before capture_viewport. Painting is off in this mode."
                } else {
                    "Raster (paintable)."
                }))
            }
            "set_texture_resolution" => {
                let set = match input.get("sets") {
                    Some(_) => *self.resolve_sets(input)?.first().ok_or_else(|| anyhow::anyhow!("no texture set given"))?,
                    None => self.active_set,
                };
                let res = input["resolution"].as_u64().unwrap_or(0) as u32;
                self.set_texture_resolution(set, res)?;
                Ok(ToolOutput::text(self.ui.status.clone()))
            }
            "select_texture_set" => {
                let index = if let Some(i) = input["index"].as_u64() {
                    i as usize
                } else {
                    let want = input["name"].as_str().unwrap_or("").trim().to_lowercase();
                    self.texture_sets
                        .iter()
                        .position(|s| format!("{} / {}", s.object, s.material).to_lowercase() == want)
                        .or_else(|| self.texture_sets.iter().position(|s| s.material.to_lowercase() == want))
                        .or_else(|| self.texture_sets.iter().position(|s| s.object.to_lowercase() == want))
                        .or_else(|| {
                            self.texture_sets.iter().position(|s| {
                                format!("{} / {}", s.object, s.material).to_lowercase().contains(&want)
                            })
                        })
                        .ok_or_else(|| anyhow::anyhow!("no texture set matches {want:?}"))?
                };
                anyhow::ensure!(index < self.texture_sets.len(), "index out of range");
                self.set_active_set(index);
                let s = &self.texture_sets[index];
                Ok(ToolOutput::text(format!("Active texture set: {} / {}", s.object, s.material)))
            }
            "list_library" => {
                let brushes = input["kind"].as_str() == Some("brush");
                self.ensure_library_scanned();
                let items = if brushes { &self.ui.brushes } else { &self.ui.library };
                let terms: Vec<String> =
                    input["query"].as_str().unwrap_or("").to_lowercase().split_whitespace().map(str::to_string).collect();
                let limit = input["limit"].as_u64().unwrap_or(25).clamp(1, 100) as usize;
                let matching: Vec<&crate::library::LibraryMaterial> = items.iter().filter(|m| m.matches(&terms)).collect();
                let list: Vec<Value> = matching
                    .iter()
                    .take(limit)
                    .map(|m| {
                        json!({
                            "name": m.name,
                            "display": m.display,
                            "tags": m.tags.iter().take(8).collect::<Vec<_>>(),
                            "source": m.source,
                            "maps": m.maps.iter().map(|(s, _)| format!("{s:?}")).collect::<Vec<_>>(),
                        })
                    })
                    .collect();
                Ok(ToolOutput::json(&json!({"total_matches": matching.len(), "installed": items.len(), "results": list})))
            }
            "apply_library_material" => {
                self.ensure_library_scanned();
                let name = input["name"].as_str().unwrap_or("");
                let i = find_named(&self.ui.library, name).ok_or_else(|| anyhow::anyhow!("no installed material named {name:?} (use list_library)"))?;
                let layers_before = self.texture_sets[self.active_set].layers.layers.len();
                self.apply_library_material(i);
                anyhow::ensure!(
                    self.texture_sets[self.active_set].layers.layers.len() > layers_before,
                    "{}",
                    self.ui.status
                );
                if !self.tool.uses_brush() {
                    self.select_tool(Tool::Paint);
                }
                Ok(ToolOutput::text(format!(
                    "{} The brush now paints this layer's mask (white reveals).",
                    self.ui.status
                )))
            }
            "use_library_brush" => {
                self.ensure_library_scanned();
                let name = input["name"].as_str().unwrap_or("");
                let i = find_named(&self.ui.brushes, name).ok_or_else(|| anyhow::anyhow!("no installed brush named {name:?} (use list_library kind=brush)"))?;
                self.use_library_brush(i);
                anyhow::ensure!(self.brush.tip_name.as_deref() == Some(self.ui.brushes[i].display.as_str()), "{}", self.ui.status);
                Ok(ToolOutput::text(self.ui.status.clone()))
            }
            "add_layer" => {
                let layers = &mut self.texture_sets[self.active_set].layers;
                layers.add_layer();
                if let Some(name) = input["name"].as_str().filter(|n| !n.trim().is_empty()) {
                    let active = layers.active;
                    layers.layers[active].name = name.trim().to_string();
                }
                self.brush.target = PaintTarget::Material;
                Ok(ToolOutput::text(format!("Added layer {} '{}' (active).", layers.active, layers.active_layer().name)))
            }
            "set_layer" => {
                let layers = &mut self.texture_sets[self.active_set].layers;
                let i = input["index"].as_u64().map(|i| i as usize).unwrap_or(layers.active);
                anyhow::ensure!(i < layers.layers.len(), "no layer {i}");
                let layer = &mut layers.layers[i];
                if let Some(n) = input["name"].as_str() {
                    layer.name = n.to_string();
                }
                if let Some(o) = unit(&input["opacity"]) {
                    layer.opacity = o;
                }
                if let Some(b) = input["blend"].as_str() {
                    layer.blend = BlendMode::ALL
                        .into_iter()
                        .find(|m| m.label().eq_ignore_ascii_case(b) || format!("{m:?}").eq_ignore_ascii_case(&b.replace(' ', "")))
                        .ok_or_else(|| anyhow::anyhow!("unknown blend mode {b:?}"))?;
                }
                if let Some(v) = input["visible"].as_bool() {
                    layer.visible = v;
                }
                if let Some(s) = input["uv_scale"].as_f64() {
                    layer.uv_scale = [s as f32; 2];
                }
                if let Some(r) = input["uv_rotation_degrees"].as_f64() {
                    layer.uv_rotation = (r as f32).to_radians();
                }
                if input["select"].as_bool() == Some(true) {
                    layers.active = i;
                }
                layers.mark_dirty();
                let l = &layers.layers[i];
                Ok(ToolOutput::text(format!(
                    "Layer {i} '{}': opacity {:.2}, {}, {}",
                    l.name,
                    l.opacity,
                    l.blend.label(),
                    if l.visible { "visible" } else { "hidden" }
                )))
            }
            "layer_action" => self.layer_action(input),
            "fill_layer" => {
                let color = input["color"].as_str().map(hex_linear).transpose()?;
                let rough = unit(&input["roughness"]);
                let metal = unit(&input["metallic"]);
                anyhow::ensure!(color.is_some() || rough.is_some() || metal.is_some(), "give at least one of color, roughness, metallic");
                let layers = &mut self.texture_sets[self.active_set].layers;
                layers.fill_active(color, rough, metal);
                Ok(ToolOutput::text(format!("Filled layer '{}'.", layers.active_layer().name)))
            }
            "smart_mask" => {
                self.ui.mask_kind = match input["kind"].as_str() {
                    Some("edge_wear") => MaskKind::EdgeWear,
                    Some("cavity_dirt") => MaskKind::CavityDirt,
                    other => anyhow::bail!("unknown smart mask {other:?}"),
                };
                self.ui.mask_amount = unit(&input["amount"]).unwrap_or(0.5);
                self.ui.mask_contrast = unit(&input["contrast"]).unwrap_or(0.5);
                if self.texture_sets[self.active_set].baked_curvature.is_none() {
                    self.bake_mesh_maps();
                }
                let before = self.texture_sets[self.active_set].layers.active_layer().mask.clone();
                self.generate_smart_mask();
                let after = &self.texture_sets[self.active_set].layers.active_layer().mask;
                let changed = match (&before, after) {
                    (Some(a), Some(b)) => !std::sync::Arc::ptr_eq(a, b),
                    (None, Some(_)) => true,
                    _ => false,
                };
                anyhow::ensure!(changed, "{}", self.ui.status);
                Ok(ToolOutput::text(self.ui.status.clone()))
            }
            "set_brush" => self.set_brush_from(input),
            "bake" => {
                match input["what"].as_str() {
                    Some("ao") => self.bake_ao(),
                    Some("mesh_maps") => self.bake_mesh_maps(),
                    _ => {
                        self.bake_ao();
                        self.bake_mesh_maps();
                    }
                }
                Ok(ToolOutput::text(format!("Baked. {}", self.ui.status)))
            }
            "undo" | "redo" => {
                let steps = input["steps"].as_u64().unwrap_or(1).clamp(1, 20);
                for _ in 0..steps {
                    if name == "undo" {
                        self.undo();
                    } else {
                        self.redo();
                    }
                }
                Ok(ToolOutput::text(format!("{name} ×{steps}. {}", self.ui.status)))
            }
            other => anyhow::bail!("unknown editor tool {other}"),
        }
    }

    /// Texture sets named by `view_pieces`: indices, "Object / Material",
    /// a material name, or an object name (all its materials).
    fn resolve_sets(&self, input: &Value) -> anyhow::Result<Vec<usize>> {
        let mut out = Vec::new();
        for item in input["sets"].as_array().into_iter().flatten() {
            let found: Vec<usize> = if let Some(i) = item.as_u64() {
                anyhow::ensure!((i as usize) < self.texture_sets.len(), "set index {i} out of range");
                vec![i as usize]
            } else {
                let want = item.as_str().unwrap_or("").trim().to_lowercase();
                let sets = &self.texture_sets;
                let all = |f: &dyn Fn(&crate::layers::TextureSet) -> bool| -> Vec<usize> {
                    (0..sets.len()).filter(|&i| f(&sets[i])).collect()
                };
                [
                    all(&|s| format!("{} / {}", s.object, s.material).to_lowercase() == want),
                    all(&|s| s.material.to_lowercase() == want),
                    all(&|s| s.object.to_lowercase() == want),
                    all(&|s| format!("{} / {}", s.object, s.material).to_lowercase().contains(&want)),
                ]
                .into_iter()
                .find(|hits| !hits.is_empty())
                .ok_or_else(|| anyhow::anyhow!("no texture set matches {want:?}"))?
            };
            for i in found {
                if !out.contains(&i) {
                    out.push(i);
                }
            }
        }
        Ok(out)
    }

    fn view_pieces(&mut self, input: &Value) -> anyhow::Result<ToolOutput> {
        let action = input["action"].as_str().unwrap_or("");
        let sets = self.resolve_sets(input)?;
        let needs_sets = matches!(action, "select" | "isolate" | "hide" | "show_only" | "delete");
        anyhow::ensure!(!needs_sets || !sets.is_empty(), "{action} needs `sets`");
        let pick = |app: &mut Self| {
            app.selection = sets.clone();
            if !sets.contains(&app.active_set) {
                app.set_active_set(sets[0]);
            }
        };
        match action {
            "select" => pick(self),
            "isolate" => {
                if self.isolate_backup.is_some() {
                    self.toggle_isolate();
                }
                pick(self);
                self.toggle_isolate();
            }
            "leave_isolation" => {
                if self.isolate_backup.is_some() {
                    self.toggle_isolate();
                }
            }
            "hide" => {
                self.selection = sets.clone();
                self.hide_selected();
            }
            "show_only" => {
                pick(self);
                self.hide_unselected();
            }
            "unhide_all" => self.unhide_all(),
            "delete" => {
                self.delete_pieces(&sets);
                anyhow::ensure!(self.ui.status.starts_with("Deleted"), "{}", self.ui.status);
            }
            "frame" => {
                let target = if sets.is_empty() { self.selection_or_active() } else { sets.clone() };
                self.frame_sets(&target);
            }
            other => anyhow::bail!("unknown action {other}"),
        }
        let visible: Vec<String> =
            self.texture_sets.iter().filter(|s| s.visible).map(|s| format!("{} / {}", s.object, s.material)).collect();
        Ok(ToolOutput::json(&json!({
            "status": self.ui.status,
            "local_view": self.isolate_backup.is_some(),
            "active_set": self.texture_sets[self.active_set].label(),
            "selected": self.selection.iter().map(|&i| self.texture_sets[i].label()).collect::<Vec<_>>(),
            "visible": visible,
        })))
    }

    fn ensure_library_scanned(&mut self) {
        if !self.ui.library_scanned {
            self.ui.library = crate::library::scan();
            self.ui.library_scanned = true;
        }
        if !self.ui.brushes_scanned {
            self.ui.brushes = crate::library::scan_brushes();
            self.ui.brushes_scanned = true;
        }
    }

    fn layer_action(&mut self, input: &Value) -> anyhow::Result<ToolOutput> {
        let layers = &mut self.texture_sets[self.active_set].layers;
        let i = input["index"].as_u64().map(|i| i as usize).unwrap_or(layers.active);
        let count = layers.layers.len();
        anyhow::ensure!(i < count, "no layer {i}");
        let action = input["action"].as_str().unwrap_or("");
        let has_mask = layers.layers[i].mask.is_some();
        match action {
            "delete" => {
                anyhow::ensure!(count > 1, "cannot delete the only layer");
                layers.remove_layer(i);
            }
            "duplicate" => layers.duplicate_layer(i),
            "move_up" => {
                anyhow::ensure!(i + 1 < count, "already on top");
                layers.move_layer(i, i + 1);
            }
            "move_down" => {
                anyhow::ensure!(i > 0, "already at the bottom");
                layers.move_layer(i, i - 1);
            }
            "merge_down" => {
                if let Some(why) = layers.merge_down_blocker(i) {
                    anyhow::bail!("cannot merge down: {why}");
                }
                anyhow::ensure!(layers.merge_down(i), "merge down failed");
            }
            "toggle_clip" => {
                layers.layers[i].clip = !layers.layers[i].clip;
                layers.mark_dirty();
            }
            "add_black_mask" => layers.add_mask(i, 0.0),
            "add_white_mask" => layers.add_mask(i, 1.0),
            "invert_mask" => {
                anyhow::ensure!(has_mask, "layer has no mask");
                layers.invert_mask(i);
            }
            "remove_mask" => layers.remove_mask(i),
            "apply_mask" => anyhow::ensure!(layers.apply_mask(i), "nothing to apply"),
            "fill_mask_white" | "fill_mask_black" => {
                let v = if action == "fill_mask_white" { 1.0 } else { 0.0 };
                if has_mask {
                    layers.fill_mask(i, v);
                } else {
                    layers.add_mask(i, v);
                }
            }
            other => anyhow::bail!("unknown action {other:?}"),
        }
        let summary: Vec<String> = layers
            .layers
            .iter()
            .enumerate()
            .rev()
            .map(|(k, l)| format!("{k}: {}{}", l.name, if k == layers.active { " (active)" } else { "" }))
            .collect();
        Ok(ToolOutput::text(format!("{action} done. Layers top→bottom: {}", summary.join(", "))))
    }

    fn set_brush_from(&mut self, input: &Value) -> anyhow::Result<ToolOutput> {
        if let Some(tool) = input["tool"].as_str() {
            self.select_tool(match tool {
                "paint" => Tool::Paint,
                "erase" => Tool::Erase,
                "stamp" => Tool::Stamp,
                "smudge" => Tool::Smudge,
                "orbit" => Tool::Orbit,
                other => anyhow::bail!("unknown tool {other:?}"),
            });
        }
        let b = &mut self.brush;
        match input["target"].as_str() {
            Some("mask") => b.target = PaintTarget::Mask,
            Some("material") => b.target = PaintTarget::Material,
            Some(other) => anyhow::bail!("unknown target {other:?}"),
            None => {}
        }
        if let Some(v) = unit(&input["mask_value"]) {
            b.mask_value = v;
        }
        if let Some(s) = input["size"].as_f64() {
            b.size = (s as f32).clamp(2.0, 200.0);
        }
        if let Some(v) = unit(&input["hardness"]) {
            b.hardness = v;
        }
        if let Some(v) = unit(&input["flow"]) {
            b.flow = v;
        }
        if let Some(c) = input["color"].as_str() {
            b.color = assistant::tools::hex_to_srgb(c)?;
        }
        if let Some(v) = unit(&input["roughness"]) {
            b.roughness_value = v;
        }
        if let Some(v) = unit(&input["metallic"]) {
            b.metallic_value = v;
        }
        if let Some(v) = input["height"].as_f64() {
            b.height_value = (v as f32).clamp(-1.0, 1.0);
        }
        if let Some(list) = input["channels"].as_array() {
            let has = |c: &str| list.iter().any(|v| v.as_str() == Some(c));
            b.paint_color = has("color");
            b.paint_roughness = has("roughness");
            b.paint_metallic = has("metallic");
            b.paint_height = has("height");
            b.paint_emissive = has("emissive");
            b.paint_opacity = has("opacity");
        }
        if input["clear_tip"].as_bool() == Some(true) {
            b.tip = None;
            b.tip_name = None;
            b.tip_has_color = false;
            b.use_tip_color = false;
            self.tip_preview_image = None;
            self.tip_preview = None;
        }
        let b = &mut self.brush;
        if let Some(v) = input["use_tip_color"].as_bool() {
            b.use_tip_color = v && b.tip_has_color;
        }
        if let Some(d) = input["tip_rotation_degrees"].as_f64() {
            b.tip_rotation = (d as f32).to_radians();
        }
        if let Some(v) = input["symmetry_x"].as_bool() {
            b.symmetry_x = v;
        }
        Ok(ToolOutput::json(&json!({"brush": self.brush_state()})))
    }

    fn brush_state(&self) -> Value {
        let b = &self.brush;
        let hex = |c: [f32; 3]| {
            let [r, g, bl] = c.map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8);
            format!("#{r:02x}{g:02x}{bl:02x}")
        };
        let channels: Vec<&str> = [
            (b.paint_color, "color"),
            (b.paint_roughness, "roughness"),
            (b.paint_metallic, "metallic"),
            (b.paint_height, "height"),
            (b.paint_emissive, "emissive"),
            (b.paint_opacity, "opacity"),
        ]
        .into_iter()
        .filter_map(|(on, n)| on.then_some(n))
        .collect();
        json!({
            "tool": super::shell::tool_name(self.tool),
            "target": b.target.label(),
            "mask_value": b.mask_value,
            "tip": b.tip_name,
            "tip_has_color": b.tip_has_color,
            "use_tip_color": b.use_tip_color,
            "tip_rotation_degrees": b.tip_rotation.to_degrees(),
            "size": b.size,
            "hardness": b.hardness,
            "flow": b.flow,
            "color": hex(b.color),
            "roughness": b.roughness_value,
            "metallic": b.metallic_value,
            "height": b.height_value,
            "channels": channels,
            "symmetry_x": b.symmetry_x,
        })
    }

    fn editor_state(&self) -> Value {
        let sets: Vec<Value> = self
            .texture_sets
            .iter()
            .enumerate()
            .map(|(i, s)| {
                json!({
                    "index": i, "object": s.object, "material": s.material,
                    "visible": s.visible, "active": i == self.active_set,
                    "resolution": s.layers.resolution,
                    "selected": self.selection.contains(&i),
                    "transparent": self.material_blend(i).is_some_and(|b| b.1),
                })
            })
            .collect();
        let set = &self.texture_sets[self.active_set];
        let layers: Vec<Value> = set
            .layers
            .layers
            .iter()
            .enumerate()
            .map(|(i, l)| {
                json!({
                    "index": i,
                    "name": l.name,
                    "visible": l.visible,
                    "opacity": l.opacity,
                    "blend": l.blend.label(),
                    "mask": l.mask.as_ref().map(|_| l.mask_kind.map(|k| k.label()).unwrap_or("painted")),
                    "mask_enabled": l.mask_enabled,
                    "clip": l.clip,
                    "fill_layer": l.has_fill,
                    "painted": l.used != 0,
                    "uv_scale": l.uv_scale,
                    "shader_mask": l.purpose.map(|k| k.label()),
                    "matcap": l.matcap.as_ref().filter(|_| l.is_matcap()).map(|m| json!({
                        "image": m.image.as_ref().map(|_| m.name.clone()),
                        "blend": m.blend.label(),
                        "strength": m.strength,
                    })),
                })
            })
            .collect();
        json!({
            "model": self.ui.model_name,
            "render_mode": if self.pt_mode { "path traced (painting disabled)" } else { "raster" },
            "texture_sets": sets,
            "active_set": {
                "index": self.active_set,
                "name": format!("{} / {}", set.object, set.material),
                "layers_bottom_to_top": layers,
                "active_layer": set.layers.active,
                "baked": {"mesh_maps": set.baked_curvature.is_some(), "ao": set.baked_ao.is_some()},
                "resolution": set.layers.resolution,
            },
            "brush": self.brush_state(),
            "vram": self.vram_json(),
        })
    }

    // -------------------------------------------------------------------- UI

    /// The always-visible strip on the right edge whose button pops the
    /// assistant open and closed.
    pub(super) fn assistant_rail(&mut self, ctx: &egui::Context) {
        egui::SidePanel::right("assistant_rail")
            .exact_width(52.0)
            .resizable(false)
            .frame(
                egui::Frame::new()
                    .fill(theme::BG_BAR)
                    .inner_margin(egui::Margin::symmetric(6, 8))
                    .stroke(Stroke::new(1.0_f32, theme::LINE)),
            )
            .show(ctx, |ui| {
                ui.vertical_centered(|ui| {
                    let open = self.ui.panels.assistant;
                    let response = widgets::tool_button(ui, icons::ASSISTANT, "F9", open)
                        .on_hover_text(if open { "Hide assistant  F9" } else { "Show assistant  F9" });
                    // Working with the panel shut: an accent dot on the button.
                    if self.assistant.busy() && !open {
                        let r = response.rect;
                        ui.painter().circle_filled(r.right_top() + egui::vec2(-6.0, 6.0), 4.0, theme::ACCENT);
                    }
                    if response.clicked() {
                        self.ui.panels.assistant = !open;
                    }
                });
            });
    }

    pub(super) fn assistant_dock(&mut self, ctx: &egui::Context) {
        if !self.ui.panels.assistant {
            return;
        }
        egui::SidePanel::right("assistant_dock")
            .default_width(380.0)
            .width_range(300.0..=640.0)
            .resizable(true)
            .frame(
                egui::Frame::new()
                    .fill(theme::BG_PANEL)
                    .inner_margin(egui::Margin::same(theme::DOCK_PADDING as i8))
                    .stroke(Stroke::new(1.0_f32, theme::LINE)),
            )
            .show(ctx, |ui| {
                self.assistant_header(ui);
                if self.assistant.show_settings {
                    self.assistant_settings(ui);
                }
                egui::TopBottomPanel::bottom("assistant_input")
                    .frame(egui::Frame::new().inner_margin(egui::Margin::symmetric(0, 8)))
                    .show_separator_line(true)
                    .show_inside(ui, |ui| self.assistant_input(ui));
                self.assistant_history(ui);
            });
    }

    fn assistant_header(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.label(theme::section_header("Assistant"));
            if self.assistant.busy() {
                ui.spinner();
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button(icons::PANEL_CLOSE).on_hover_text("Hide assistant  F9").clicked() {
                    self.ui.panels.assistant = false;
                }
                if ui.button(icons::SETTINGS).on_hover_text("Assistant settings").clicked() {
                    self.assistant.show_settings = !self.assistant.show_settings;
                }
                let serving = self.assistant.mcp_serving();
                let (icon, color) = match &self.assistant.bridge {
                    Some(b) if b.http.is_ok() => (icons::LINK, theme::OK_FG),
                    Some(_) => (icons::WARN, theme::WARN_FG),
                    None => (icons::UNLINK, theme::TEXT_MUTED),
                };
                let hover = match &self.assistant.bridge {
                    Some(b) => match &b.http {
                        Ok(port) => format!("MCP server running at {}. Click to stop.", assistant::bridge::http_url(*port)),
                        Err(e) => format!("MCP server half up: {e}. Click to stop."),
                    },
                    None => "Start the MCP server so Claude Code can drive the editor".into(),
                };
                if ui.button(RichText::new(icon).color(color)).on_hover_text(hover).clicked() {
                    if let Err(e) = self.assistant.toggle_mcp() {
                        self.ui.status = format!("Could not save assistant settings: {e:#}");
                    }
                    // Show the registration command the first time it starts.
                    if !serving && self.assistant.bridge.is_some() {
                        self.assistant.show_settings = true;
                    }
                }
                let idle = !self.assistant.busy();
                if ui.add_enabled(idle, egui::Button::new(icons::ADD)).on_hover_text("New chat").clicked() {
                    self.assistant.entries.clear();
                    self.assistant.messages.clear();
                    self.assistant.usage = (0, 0, 0);
                }
            });
        });
        widgets::divider(ui);
    }

    fn assistant_settings(&mut self, ui: &mut Ui) {
        let s = &mut self.assistant.settings;
        let mut changed = false;
        egui::Grid::new("assistant_settings").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
            ui.label("Claude key");
            changed |= ui
                .add(egui::TextEdit::singleline(&mut s.api_key).password(true).hint_text("uses $ANTHROPIC_API_KEY").desired_width(f32::INFINITY))
                .changed();
            ui.end_row();
            ui.label("");
            let env = std::env::var("ANTHROPIC_API_KEY").is_ok_and(|k| !k.is_empty());
            ui.label(
                RichText::new(match (s.api_key.trim().is_empty(), env) {
                    (false, _) => "Using the key above (stored in ~/.config/zestsurface/assistant.json)",
                    (true, true) => "Using $ANTHROPIC_API_KEY",
                    (true, false) => "No key yet — get one at console.anthropic.com, or use Claude Code (below)",
                })
                .small()
                .color(theme::TEXT_DIM),
            );
            ui.end_row();
            ui.label("Gemini key");
            changed |= ui
                .add(
                    egui::TextEdit::singleline(&mut s.gemini_api_key)
                        .password(true)
                        .hint_text("uses $GEMINI_API_KEY")
                        .desired_width(f32::INFINITY),
                )
                .changed();
            ui.end_row();
            ui.label("");
            let gemini_env = ["GEMINI_API_KEY", "GOOGLE_API_KEY"].iter().any(|v| std::env::var(v).is_ok_and(|k| !k.is_empty()));
            ui.label(
                RichText::new(match (s.gemini_api_key.trim().is_empty(), gemini_env) {
                    (false, _) => "Using the key above — for Gemini models and Nano Banana images",
                    (true, true) => "Using $GEMINI_API_KEY",
                    (true, false) => "Optional — from aistudio.google.com; Nano Banana images need billing on",
                })
                .small()
                .color(theme::TEXT_DIM),
            );
            ui.end_row();
            ui.label("Model");
            egui::ComboBox::from_id_salt("assistant_model").selected_text(s.model.clone()).show_ui(ui, |ui| {
                let first_gemini = assistant::MODELS.iter().position(|m| assistant::gemini::is_gemini(m));
                for (i, m) in assistant::MODELS.into_iter().enumerate() {
                    if Some(i) == first_gemini {
                        ui.separator();
                    }
                    changed |= ui.selectable_value(&mut s.model, m.to_string(), m).changed();
                }
            });
            ui.end_row();
            ui.label("Images");
            ui.horizontal(|ui| {
                let label = assistant::IMAGE_BACKENDS.iter().find(|(id, _)| *id == s.image_backend).map_or("Auto", |b| b.1);
                egui::ComboBox::from_id_salt("assistant_images").selected_text(label).show_ui(ui, |ui| {
                    for (id, label) in assistant::IMAGE_BACKENDS {
                        changed |= ui.selectable_value(&mut s.image_backend, id.to_string(), label).changed();
                    }
                });
                if s.images_via_gemini() {
                    egui::ComboBox::from_id_salt("assistant_image_model").selected_text(s.image_model.clone()).show_ui(ui, |ui| {
                        for m in assistant::IMAGE_MODELS {
                            changed |= ui.selectable_value(&mut s.image_model, m.to_string(), m).changed();
                        }
                    });
                }
            });
            ui.end_row();
            ui.label("Effort");
            egui::ComboBox::from_id_salt("assistant_effort").selected_text(s.effort.clone()).show_ui(ui, |ui| {
                for e in assistant::EFFORTS {
                    changed |= ui.selectable_value(&mut s.effort, e.to_string(), e).changed();
                }
            });
            ui.end_row();
            ui.label("ComfyUI");
            changed |= ui.add(egui::TextEdit::singleline(&mut s.comfy_url).desired_width(f32::INFINITY)).changed();
            ui.end_row();
            ui.label("Checkpoint");
            ui.horizontal(|ui| {
                changed |= ui
                    .add(egui::TextEdit::singleline(&mut s.comfy_checkpoint).hint_text("first installed").desired_width(150.0))
                    .changed();
                if ui.button("Test").on_hover_text("List the server's checkpoints").clicked() {
                    self.assistant.comfy_status = match assistant::comfy::checkpoints(&s.comfy_url) {
                        Ok(list) if list.is_empty() => "Connected, but no checkpoints installed".into(),
                        Ok(list) => format!("Connected: {}", list.join(", ")),
                        Err(e) => format!("{e:#}"),
                    };
                }
            });
            ui.end_row();
        });
        if !self.assistant.comfy_status.is_empty() {
            ui.label(RichText::new(&self.assistant.comfy_status).small().color(theme::TEXT_MUTED));
        }
        ui.label(
            RichText::new(format!(
                "Custom ComfyUI graph (API format, %prompt% %negative% %seed% %width% %height%): {}",
                assistant::tools::workflow_path().display()
            ))
            .small()
            .color(theme::TEXT_DIM),
        );
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label("Claude Code MCP server");
            let running = self.assistant.bridge.is_some();
            let label = if running { "Stop" } else { "Start" };
            if ui
                .button(label)
                .on_hover_text("Serve the editor's tools to Claude Code over HTTP on this machine. Uses your Claude Code login — no API key. Remembered for next launch.")
                .clicked()
            {
                if let Err(e) = self.assistant.toggle_mcp() {
                    self.ui.status = format!("Could not save assistant settings: {e:#}");
                }
            }
        });
        let (status, color) = match &self.assistant.bridge {
            Some(bridge) => match &bridge.http {
                Ok(port) => (format!("Running at {}", assistant::bridge::http_url(*port)), theme::OK_FG),
                Err(e) => (format!("HTTP endpoint failed: {e}. Only `zestsurface mcp` (stdio) can connect."), theme::WARN_FG),
            },
            None if !self.assistant.bridge_status.is_empty() => (self.assistant.bridge_status.clone(), theme::WARN_FG),
            None => ("Stopped".into(), theme::TEXT_MUTED),
        };
        ui.label(RichText::new(status).small().color(color));
        ui.horizontal(|ui| {
            ui.label(RichText::new("Port").small().color(theme::TEXT_MUTED));
            let mut port = self.assistant.settings.mcp_port;
            let edit = ui.add(egui::DragValue::new(&mut port).range(1024..=65535).speed(0.0));
            if port != self.assistant.settings.mcp_port {
                self.assistant.settings.mcp_port = port;
                changed = true;
            }
            // Restart on the new port once the edit is done, not per keystroke.
            if edit.lost_focus() && self.assistant.bridge.as_ref().is_some_and(|b| b.http != Ok(port)) {
                self.assistant.set_bridge(true);
            }
        });
        if self.assistant.mcp_serving() {
            let cmd = self.assistant.mcp_add_command();
            ui.label(RichText::new("Register once with Claude Code (run in a terminal):").small().color(theme::TEXT_MUTED));
            ui.horizontal(|ui| {
                ui.label(RichText::new("claude mcp add --transport http zestsurface …").small().monospace().color(theme::TEXT_DIM))
                    .on_hover_text(&cmd);
                if ui.small_button("Copy").on_hover_text("Copy the full command (it contains the access token)").clicked() {
                    ui.ctx().copy_text(cmd.clone());
                }
            });
            if ui
                .small_button("New token")
                .on_hover_text("Revoke the current token. Claude Code must be registered again with the new command.")
                .clicked()
            {
                self.assistant.settings.mcp_token.clear();
                self.assistant.set_bridge(true);
                changed = true;
            }
        }
        if changed && let Err(e) = self.assistant.settings.save(&self.assistant.settings_path) {
            self.ui.status = format!("Could not save assistant settings: {e:#}");
        }
        widgets::divider(ui);
    }

    fn assistant_input(&mut self, ui: &mut Ui) {
        let mut changed = false;
        ui.horizontal(|ui| {
            let s = &mut self.assistant.settings;
            changed |= ui.checkbox(&mut s.attach_viewport, "Send view").on_hover_text("Attach a viewport capture to each message").changed();
            changed |= ui
                .checkbox(&mut s.attach_textures, "Send textures")
                .on_hover_text("Attach the active texture set's base colour texture and painted layers")
                .changed();
        });
        if changed {
            let _ = self.assistant.settings.save(&self.assistant.settings_path);
        }
        let id = ui.make_persistent_id("assistant_prompt");
        // Enter sends, Shift+Enter is a new line: take the key before the
        // text field turns it into a newline.
        let enter = ui.memory(|m| m.has_focus(id))
            && ui.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Enter));
        let response = ui.add(
            egui::TextEdit::multiline(&mut self.assistant.input)
                .id(id)
                .hint_text("Ask for anything… e.g. \"add some blood splatter on the jacket\"")
                .desired_rows(3)
                .desired_width(f32::INFINITY),
        );
        ui.horizontal(|ui| {
            let (i, o, c) = self.assistant.usage;
            if i + o + c > 0 {
                ui.label(RichText::new(format!("{}k in · {}k cached · {}k out", i / 1000, c / 1000, o / 1000)).small().color(theme::TEXT_DIM));
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if self.assistant.busy() {
                    if ui.button("Stop").clicked()
                        && let Some(turn) = self.assistant.turn.as_ref()
                    {
                        turn.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
                    }
                } else {
                    let send = ui.add_enabled(!self.assistant.input.trim().is_empty(), egui::Button::new("Send"));
                    if send.clicked() || enter {
                        self.send_assistant_message();
                        // Hand the keyboard back so tool shortcuts and undo
                        // history work while the assistant edits.
                        response.surrender_focus();
                    }
                }
            });
        });
    }

    fn assistant_history(&mut self, ui: &mut Ui) {
        egui::ScrollArea::vertical().auto_shrink([false, false]).stick_to_bottom(true).show(ui, |ui| {
            if self.assistant.entries.is_empty() {
                ui.add_space(8.0);
                ui.label(
                    RichText::new(
                        "Ask the assistant to set up layers, find or generate textures and brushes, design tattoos, or fix something you can see. \
                         It sees your viewport and textures when you send a message.",
                    )
                    .color(theme::TEXT_MUTED),
                );
            }
            let ctx = ui.ctx().clone();
            for entry in self.assistant.entries.iter_mut() {
                match entry {
                    Entry::User { text, attachments } => {
                        egui::Frame::new()
                            .fill(theme::BG_BUTTON)
                            .corner_radius(CornerRadius::same(theme::RADIUS_ROW))
                            .inner_margin(egui::Margin::symmetric(10, 6))
                            .show(ui, |ui| {
                                ui.set_width(ui.available_width());
                                ui.label(RichText::new(text.as_str()).color(theme::TEXT_STRONG));
                                if *attachments > 0 {
                                    ui.label(
                                        RichText::new(format!("{}  {attachments} image(s) attached", icons::IMAGE)).small().color(theme::TEXT_DIM),
                                    );
                                }
                            });
                    }
                    Entry::Assistant(text) => {
                        ui.label(RichText::new(text.trim()).color(theme::TEXT));
                    }
                    Entry::Thinking(text) => {
                        egui::CollapsingHeader::new(RichText::new("Thinking").small().color(theme::TEXT_DIM))
                            .id_salt(text.as_ptr())
                            .show(ui, |ui| {
                                ui.label(RichText::new(text.trim()).small().color(theme::TEXT_DIM));
                            });
                    }
                    Entry::Tool { name, detail, state, summary, preview, texture, .. } => {
                        let (icon, color) = match state {
                            ToolState::Running => (icons::MORE, theme::TEXT_MUTED),
                            ToolState::Ok => (icons::CHECK, theme::OK_FG),
                            ToolState::Failed => (icons::WARN, theme::ERR_FG),
                        };
                        ui.horizontal_wrapped(|ui| {
                            ui.label(RichText::new(icon).color(color));
                            ui.label(RichText::new(name.as_str()).small().color(theme::TEXT_MUTED).monospace());
                            if !detail.is_empty() {
                                ui.label(RichText::new(detail.as_str()).small().color(theme::TEXT_DIM));
                            }
                        });
                        if matches!(state, ToolState::Failed) && !summary.is_empty() {
                            ui.label(RichText::new(summary.as_str()).small().color(theme::ERR_FG));
                        }
                        if let Some(img) = preview.as_ref() {
                            let tex = texture.get_or_insert_with(|| {
                                // Previews are shown on white, like the stamp
                                // reads on skin.
                                let flat = assistant::images::flatten(img, [255, 255, 255]);
                                ctx.load_texture(
                                    format!("assistant_preview_{}", img.as_ptr() as usize),
                                    egui::ColorImage::from_rgb([flat.width() as usize, flat.height() as usize], flat.as_raw()),
                                    Default::default(),
                                )
                            });
                            let size = tex.size_vec2();
                            let scale = (160.0 / size.x.max(size.y)).min(1.0);
                            ui.add(egui::Image::new((tex.id(), size * scale)).corner_radius(CornerRadius::same(4)));
                        }
                    }
                    Entry::Notice { text, error } => {
                        let (fg, bg) = if *error { (theme::ERR_FG, theme::ERR_BG) } else { (theme::TEXT_MUTED, Color32::TRANSPARENT) };
                        egui::Frame::new().fill(bg).corner_radius(CornerRadius::same(4)).inner_margin(egui::Margin::symmetric(8, 4)).show(ui, |ui| {
                            ui.label(RichText::new(text.as_str()).small().color(fg));
                        });
                    }
                }
                ui.add_space(6.0);
            }
        });
    }
}

/// Whether an opacity map has see-through values that a cutout material
/// can't show: more than a sliver of texels strictly between clear and solid.
fn has_partial_opacity(image: &crate::scene::ImageData) -> bool {
    let texels = image.rgba.chunks_exact(4);
    let total = texels.len().max(1);
    let partial = texels.filter(|p| (8..=247).contains(&((p[0] as u32 * p[3] as u32) / 255))).count();
    partial * 200 > total
}
