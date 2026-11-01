//! Tool definitions sent to Claude, and the tools that run on the worker
//! thread (disk and network only — no document state).
//!
//! Editor tools (`EDITOR_TOOLS`) are only *declared* here; they run on the UI
//! thread in `app/assistant_dock.rs`, which owns the document.

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, OnceLock, mpsc};

use anyhow::{Context as _, Result, bail};
use serde_json::{Value, json};

use super::{Event, Settings, ToolOutput, comfy, gemini, images, library_root};

/// Tools executed on the UI thread against the open document.
pub const EDITOR_TOOLS: &[&str] = &[
    "get_editor_state",
    "capture_viewport",
    "select_texture_set",
    "list_library",
    "apply_library_material",
    "use_library_brush",
    "add_layer",
    "set_layer",
    "layer_action",
    "fill_layer",
    "smart_mask",
    "set_brush",
    "bake",
    "undo",
    "redo",
    "import_model",
    "view_pieces",
    "set_texture_resolution",
    "set_render_mode",
    "import_image",
    "place_stamp",
    "set_view",
    "set_matcap",
    "set_material",
];

pub fn is_editor_tool(name: &str) -> bool {
    EDITOR_TOOLS.contains(&name)
}

fn tool(name: &str, description: &str, properties: Value, required: &[&str]) -> Value {
    json!({
        "name": name,
        "description": description,
        "input_schema": {"type": "object", "properties": properties, "required": required},
        // Streamed requests: let inputs (an SVG can be long) stream as they
        // are generated. The worker validates every parsed input itself.
        "eager_input_streaming": true,
    })
}

/// Every tool, in a fixed order so the request prefix stays cacheable.
pub fn definitions() -> Value {
    let color = json!({"type": "string", "description": "sRGB hex colour, e.g. \"#6a0a0a\""});
    let unit = |d: &str| json!({"type": "number", "minimum": 0, "maximum": 1, "description": d});
    let layer_index = json!({"type": "integer", "description": "Layer index from get_editor_state (0 = bottom). Defaults to the active layer."});
    json!([
        tool(
            "get_editor_state",
            "Read the open document: texture sets (object/material, visibility, which is active), the active set's layer stack (index, name, opacity, blend, visibility, mask, clip, fill/tiling) with the active layer, the brush (tool, tip, size, colour, channels, target) and which mesh maps are baked. Call this before editing anything you have not just seen.",
            json!({}),
            &[],
        ),
        tool(
            "capture_viewport",
            "Take a screenshot of the 3D viewport as the user currently sees it, optionally with the active texture set's material base colour texture and painted-layer composite. Use it to check your work after a change, or to look at something the user points out.",
            json!({"include_textures": {"type": "boolean", "description": "Also return the active set's texture maps (default false)."}}),
            &[],
        ),
        tool(
            "select_texture_set",
            "Make a texture set (one object+material, e.g. 'Jacket / Leather') the active paint target. Matches by 'object', 'material' or 'object / material', case-insensitive, or by index.",
            json!({"name": {"type": "string"}, "index": {"type": "integer"}}),
            &[],
        ),
        tool(
            "list_library",
            "Search the user's INSTALLED asset library (materials or brush stamps on disk), by words in the name and tags. Returns folder names to pass to apply_library_material / use_library_brush. To find assets that are not installed yet, use search_online_assets.",
            json!({
                "kind": {"type": "string", "enum": ["material", "brush"]},
                "query": {"type": "string", "description": "Space-separated words; empty lists everything."},
                "limit": {"type": "integer", "description": "Default 25."}
            }),
            &["kind"],
        ),
        tool(
            "apply_library_material",
            "Add a new FILL LAYER from an installed library material (all its maps: colour, roughness, metal, normal, height, opacity) on top of the active texture set, with a BLACK mask, and switch the brush to paint that mask. Nothing is visible until the mask is painted or generated (smart_mask, layer_action fill_mask_white) — tell the user that.",
            json!({"name": {"type": "string", "description": "Folder name from list_library."}}),
            &["name"],
        ),
        tool(
            "use_library_brush",
            "Load an installed brush stamp as the brush tip. Colour stamps (tattoos, decals) keep their own colours; greyscale stamps paint with the brush colour.",
            json!({"name": {"type": "string", "description": "Folder name from list_library (kind brush) or from a tool that created a brush."}}),
            &["name"],
        ),
        tool(
            "add_layer",
            "Add an empty paint layer on top of the active texture set and select it.",
            json!({"name": {"type": "string"}}),
            &[],
        ),
        tool(
            "set_layer",
            "Change a layer's properties. Only the fields given are changed.",
            json!({
                "index": layer_index,
                "name": {"type": "string"},
                "opacity": unit("Layer opacity."),
                "blend": {"type": "string", "enum": ["Normal", "Multiply", "Darken", "Color burn", "Screen", "Lighten", "Color dodge", "Add", "Overlay", "Soft light", "Hard light", "Difference", "Subtract", "Hue", "Saturation", "Color", "Luminosity"]},
                "visible": {"type": "boolean"},
                "select": {"type": "boolean", "description": "Make it the active layer."},
                "uv_scale": {"type": "number", "description": "Fill-layer tiling repeats across the UV square."},
                "uv_rotation_degrees": {"type": "number"}
            }),
            &[],
        ),
        tool(
            "layer_action",
            "One structural action on a layer.",
            json!({
                "index": layer_index,
                "action": {"type": "string", "enum": [
                    "delete", "duplicate", "move_up", "move_down", "merge_down", "toggle_clip",
                    "add_black_mask", "add_white_mask", "invert_mask", "remove_mask", "apply_mask",
                    "fill_mask_white", "fill_mask_black"
                ]}
            }),
            &["action"],
        ),
        tool(
            "fill_layer",
            "Flood the ACTIVE layer with uniform values (only the ones given). For a tinted overlay or a solid-colour layer.",
            json!({"color": color, "roughness": unit("Roughness."), "metallic": unit("Metallic.")}),
            &[],
        ),
        tool(
            "smart_mask",
            "Generate a procedural mask on the active layer from baked mesh maps (bakes them first if needed, which takes a few seconds): edge_wear = bright on convex edges, cavity_dirt = bright in crevices (dirt, grime, dried blood in creases).",
            json!({
                "kind": {"type": "string", "enum": ["edge_wear", "cavity_dirt"]},
                "amount": unit("Coverage, default 0.5."),
                "contrast": unit("Edge sharpness, default 0.5.")
            }),
            &["kind"],
        ),
        tool(
            "set_brush",
            "Configure the brush. Only the fields given are changed. The user does the actual painting/stamping in the viewport.",
            json!({
                "tool": {"type": "string", "enum": ["paint", "erase", "stamp", "smudge", "orbit"], "description": "stamp places the whole tip image once per click (decals, tattoos, splats)."},
                "target": {"type": "string", "enum": ["material", "mask"], "description": "Paint layer channels, or the active layer's mask."},
                "mask_value": unit("In mask mode: 1 reveals the layer, 0 hides it."),
                "size": {"type": "number", "minimum": 2, "maximum": 200, "description": "Radius in viewport pixels."},
                "hardness": unit("0 soft, 1 hard edge."),
                "flow": unit("Opacity per stamp."),
                "color": color,
                "roughness": unit("Roughness value painted."),
                "metallic": unit("Metallic value painted."),
                "height": {"type": "number", "minimum": -1, "maximum": 1, "description": "Height painted: positive raises, negative carves."},
                "channels": {"type": "array", "items": {"type": "string", "enum": ["color", "roughness", "metallic", "height", "emissive", "opacity"]}, "description": "Exactly which channels the brush paints."},
                "use_tip_color": {"type": "boolean", "description": "Paint a colour tip's own colours instead of the brush colour."},
                "tip_rotation_degrees": {"type": "number"},
                "clear_tip": {"type": "boolean", "description": "Back to a plain round brush."},
                "symmetry_x": {"type": "boolean"}
            }),
            &[],
        ),
        tool(
            "bake",
            "Bake mesh maps (curvature/position/normal, needed by smart masks) and/or ambient occlusion for all texture sets.",
            json!({"what": {"type": "string", "enum": ["mesh_maps", "ao", "all"]}}),
            &["what"],
        ),
        tool(
            "import_image",
            "Bring a local image file into the editor. as=stamp: make it the stamp tip (a decal, tattoo, logo or overlay with transparency; then place_stamp). as=layer: a new layer holding it as one map (kind), laid over the whole UV layout — for a texture painted or baked elsewhere for this model. An opacity layer with in-between values switches the material to the blended pass (see set_material) so they show as see-through. as=fill_layer: a new layer filled with the image (tileable textures; adjust tiling/projection with set_layer). as=material_map: replace one of the active material's own map slots (kind) — undoable. Applies to the active texture set.",
            json!({
                "path": {"type": "string", "description": "Absolute path (or ~/...) of a png/jpg/tga/webp/tif image."},
                "as": {"type": "string", "enum": ["stamp", "layer", "fill_layer", "material_map"]},
                "kind": {"type": "string", "enum": ["base_color", "roughness", "metallic", "normal", "height", "emissive", "emission_mask", "opacity", "ao"], "description": "Which map, for as=layer / material_map. Default base_color. emission_mask (material_map only): where the material's emission glows (white) or not (black); it multiplies the emissive map and the Emission colour."},
                "name": {"type": "string", "description": "Layer or stamp name; defaults to the file name."}
            }),
            &["path", "as"],
        ),
        tool(
            "place_stamp",
            "Stamp the current tip (import_image as=stamp, use_library_brush, create_stamp_from_svg…) onto the model at a point of the viewport, exactly like the user clicking with the stamp tool: x, y are fractions (0..1) of capture_viewport's image from its top-left. Paints the active set's active layer with the current brush settings (set_brush: colour, channels, target, use_tip_color). Refuses points that miss the active set. Answers with a screenshot of the result to check placement; undo to retry. Aim with set_view first so the target area faces the camera.",
            json!({
                "x": {"type": "number", "minimum": 0, "maximum": 1},
                "y": {"type": "number", "minimum": 0, "maximum": 1},
                "size": {"type": "number", "description": "Stamp radius in viewport pixels (half its width). Defaults to the current brush size."},
                "rotation_degrees": {"type": "number", "description": "Tip rotation. Defaults to the current one."},
                "opacity": {"type": "number", "minimum": 0, "maximum": 1, "description": "Stamp strength (the brush flow, default 0.35 — use 1 for a solid decal/tattoo). Stays set for later stamps."},
                "capture": {"type": "boolean", "description": "Return a screenshot after stamping (default true)."}
            }),
            &["x", "y"],
        ),
        tool(
            "set_material",
            "Change a texture set's material surface settings. transparent: the blended pass (sheer fabric, glass, lashes) — painted opacity then shows as real see-through at every level; off, the material is drawn as a cutout (below 0.5 opacity is hidden, 0.5 and above is fully solid). opacity: the material's own opacity, multiplying everything painted (keep 1 unless the whole piece should fade). Undoable. Applies to the active set unless `sets` is given.",
            json!({
                "transparent": {"type": "boolean", "description": "Blended (true) or cutout (false) rendering."},
                "opacity": {"type": "number", "minimum": 0, "maximum": 1, "description": "Material opacity factor."},
                "sets": {"type": "array", "items": {"type": ["string", "integer"]}, "description": "Texture sets (indices, 'Object / Material', a material or an object name); defaults to the active set."}
            }),
            &[],
        ),
        tool(
            "set_matcap",
            "Add or change a matcap layer: a layer holding a matcap (a sphere image looked up by the view-space normal and blended over the shading — gloss, metal, rim light, pearl…) whose painted greyscale says where it shows (white = on). A material can have any number; they draw bottom to top, as Poiyomi's matcaps do. With no `layer`, it changes the active layer when that is a matcap layer, else adds a new one; `new_layer` always adds one. `sets` adds a new matcap layer to each named texture set at once (e.g. every Metal material). A new layer's mask starts all white (whole material); `fill` resets a mask to white or black, and the user paints it to limit where it shows. builtin picks a generated matcap with its natural blend; image_path loads a matcap image file. Raster view only. Returns a screenshot.",
            json!({
                "builtin": {"type": "string", "enum": crate::matcap::BUILTINS.iter().map(|b| b.0).collect::<Vec<_>>()},
                "image_path": {"type": "string", "description": "Absolute path of a matcap image (a lit sphere filling a square)."},
                "blend": {"type": "string", "enum": ["add", "multiply", "replace", "screen"]},
                "strength": {"type": "number", "minimum": 0, "maximum": 2},
                "layer": {"type": "integer", "description": "Index of a matcap layer of the active set to change."},
                "new_layer": {"type": "boolean", "description": "Add a new matcap layer even if the active layer is one."},
                "sets": {"type": "array", "items": {}, "description": "Texture sets (indices, 'Object / Material', a material or an object name) to each get a new matcap layer."},
                "fill": {"type": "string", "enum": ["white", "black"], "description": "Fill the layer's mask: white shows it on the whole material, black nowhere."},
                "name": {"type": "string", "description": "Rename the layer."},
                "visible": {"type": "boolean"},
                "capture": {"type": "boolean", "description": "Return a screenshot (default true)."}
            }),
            &[],
        ),
        tool(
            "set_view",
            "Aim the viewport camera, then return a screenshot. preset: which side the camera looks from (front = the model's +Z side, the way glTF/FBX characters face; back; left/right = -X/+X; top; bottom; three_quarter). frame: \"all\", \"selection\", or texture-set names/indices to centre and fit. yaw/pitch override the preset; zoom > 1 moves closer (applied after framing).",
            json!({
                "preset": {"type": "string", "enum": ["front", "back", "left", "right", "top", "bottom", "three_quarter"]},
                "frame": {"description": "\"all\", \"selection\", or a list of texture-set names/indices"},
                "yaw_degrees": {"type": "number"},
                "pitch_degrees": {"type": "number", "minimum": -88, "maximum": 88},
                "zoom": {"type": "number", "exclusiveMinimum": 0},
                "capture": {"type": "boolean", "description": "Return a screenshot (default true)."}
            }),
            &[],
        ),
        tool(
            "set_render_mode",
            "Switch the viewport between raster (fast, paintable) and path_traced (physically based preview; painting is disabled). To judge final looks, switch to path_traced, wait a few seconds, then capture_viewport; switch back to raster before painting.",
            json!({"mode": {"type": "string", "enum": ["raster", "path_traced"]}}),
            &["mode"],
        ),
        tool(
            "set_texture_resolution",
            "Change a texture set's paint resolution (texels per side) — 4096 or 8192 for fine detail such as tattoo linework on a small UV area. Existing layers are resampled (enlarged paint gets slightly softer; new paint is crisp). Undoable. Refused when the GPU memory estimate is too high; each step up uses 4x the memory, so prefer 4096 unless 8192 is really needed. Larger stamps (e.g. 1024–2048 px) are worth it at higher resolutions.",
            json!({
                "resolution": {"type": "integer", "enum": [1024, 2048, 4096, 8192]},
                "sets": {"type": "array", "items": {"type": ["string", "integer"]}, "description": "One texture set (index or name); defaults to the active set."}
            }),
            &["resolution"],
        ),
        tool(
            "view_pieces",
            "Select, isolate, hide, frame or delete pieces of the model (texture sets), like Blender's local view. isolate = show only these and frame them (the user's / key; call again with leave_isolation to restore). Selecting also makes the first piece the paint target. Hidden pieces don't occlude painting or baking. delete removes the pieces and their mesh parts from the document, paint and all (undoable; only when the user asks — prefer hide otherwise).",
            json!({
                "action": {"type": "string", "enum": ["select", "isolate", "leave_isolation", "hide", "show_only", "unhide_all", "frame", "delete"]},
                "sets": {"type": "array", "items": {"type": ["string", "integer"]}, "description": "Texture set indices, \"Object / Material\" names, material names, or object names (= all its materials)."}
            }),
            &["action"],
        ),
        tool(
            "import_model",
            "Open a 3D model (.fbx, .gltf, .glb) from disk as a NEW document, replacing the current model and all its paint (unsaved work is lost — ask first if anything is painted). Waits for the import and returns the editor state with the new texture sets.",
            json!({"path": {"type": "string", "description": "Absolute path of the model file."}}),
            &["path"],
        ),
        tool("undo", "Undo the last document change(s).", json!({"steps": {"type": "integer", "minimum": 1, "maximum": 20}}), &[]),
        tool("redo", "Redo undone change(s).", json!({"steps": {"type": "integer", "minimum": 1, "maximum": 20}}), &[]),
        tool(
            "search_online_assets",
            "Search the catalogue of ~3000 free CC0 assets from Poly Haven and ambientCG (PBR materials, decals, atlases, brush alphas) — whether installed or not. Note: there is NO blood, gore, splatter, tattoo or lace in it; generate those instead.",
            json!({
                "query": {"type": "string", "description": "Space-separated words matched against names, tags and categories."},
                "kind": {"type": "string", "enum": ["material", "decal", "atlas", "brush"]},
                "limit": {"type": "integer", "description": "Default 15."}
            }),
            &["query"],
        ),
        tool(
            "download_asset",
            "Download a catalogue asset (id from search_online_assets) into the library. Materials become usable with apply_library_material, everything else with use_library_brush, under the returned folder name.",
            json!({"id": {"type": "string"}, "resolution": {"type": "string", "enum": ["1k", "2k", "4k"], "description": "Default 2k."}}),
            &["id"],
        ),
        tool(
            "generate_splatter_brush",
            "Procedurally generate a liquid splatter stamp brush (blood, paint, mud, slime) into the brush library and return a preview. Styles: splat (impact with spikes and droplets), drip (pool with runs dripping down — stamp 'down' is the image's bottom), spray (fine mist), droplets (separate drops), smear (dragged streak). Make several seeds for variety.",
            json!({
                "name": {"type": "string", "description": "Brush folder name, e.g. blood_splat_1."},
                "style": {"type": "string", "enum": ["splat", "drip", "spray", "droplets", "smear"]},
                "seed": {"type": "integer", "description": "Different seeds give different shapes."},
                "density": {"type": "number", "minimum": 0.2, "maximum": 3, "description": "Amount of droplets/runs, default 1."}
            }),
            &["name", "style"],
        ),
        tool(
            "create_material",
            "Create a flat procedural PBR material in the library (uniform colour, roughness, metallic, optional subtle mottling in colour/roughness/height). E.g. fresh blood ≈ #5a0000 roughness 0.15, dried blood ≈ #2e0805 roughness 0.6, ink, slime, paint. Then use apply_library_material.",
            json!({
                "name": {"type": "string"},
                "color": color,
                "roughness": unit("Roughness."),
                "metallic": unit("Metallic, default 0."),
                "variation": unit("Mottling strength, default 0.3."),
                "tags": {"type": "array", "items": {"type": "string"}}
            }),
            &["name", "color", "roughness"],
        ),
        tool(
            "create_stamp_from_svg",
            "Draw a design as SVG and save it as a stamp brush (tattoos, emblems, lettering, patterns, decals). Draw on a TRANSPARENT background (no full-page white rect) with a viewBox. Black-only designs become a monochrome stamp painted with the brush colour (ink colour adjustable); multi-colour designs keep their colours. Returns a preview so you can check and iterate.",
            json!({
                "name": {"type": "string"},
                "svg": {"type": "string", "description": "Complete SVG document with xmlns and viewBox."},
                "size": {"type": "integer", "minimum": 256, "maximum": 2048, "description": "Longer edge in pixels, default 1024."}
            }),
            &["name", "svg"],
        ),
        tool(
            "generate_image_stamp",
            "Generate a raster image and save it as a stamp brush — for detailed or shaded designs (realistic tattoos, painted decals) beyond what SVG can do. Runs on Google's Nano Banana (Gemini image models) when the user has a Gemini key, else on their local ComfyUI (Stable Diffusion); the assistant settings choose. Prompt for an isolated design on a plain white background (e.g. 'tattoo flash design of ..., black and grey, clean linework, isolated on plain white background'); the white is removed. Takes 10-60 s. If it fails (no key, ComfyUI not running), offer SVG instead.",
            json!({
                "name": {"type": "string"},
                "prompt": {"type": "string"},
                "negative": {"type": "string", "description": "Default excludes photos of skin, body parts, backgrounds, text and frames."},
                "seed": {"type": "integer"},
                "width": {"type": "integer", "description": "Default 1024 (multiple of 64)."},
                "height": {"type": "integer", "description": "Default 1024 (multiple of 64)."},
                "steps": {"type": "integer", "description": "Default 28."},
                "remove_white_background": {"type": "boolean", "description": "Default true."}
            }),
            &["name", "prompt"],
        ),
        tool(
            "import_image_from_url",
            "Download an image from a direct http(s) image URL (found with web search) into the library, as a stamp brush or as a basecolor-only material. Tell the user the source and its licence; prefer CC0/public-domain sources.",
            json!({
                "url": {"type": "string"},
                "name": {"type": "string"},
                "as": {"type": "string", "enum": ["brush", "material"]},
                "remove_white_background": {"type": "boolean", "description": "For brushes: turn a white page transparent. Default false."}
            }),
            &["url", "name", "as"],
        ),
        {"type": "web_search_20260209", "name": "web_search", "max_uses": 5},
        {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": 5},
    ])
}

/// What the worker tools need besides their input.
pub struct WorkerCtx<'a> {
    pub settings: &'a Settings,
    pub cancel: &'a Arc<AtomicBool>,
    pub events: &'a mpsc::Sender<Event>,
}

pub fn run_worker_tool(name: &str, input: &Value, ctx: &WorkerCtx) -> ToolOutput {
    let result = match name {
        "search_online_assets" => search_online_assets(input),
        "download_asset" => download_asset(input, ctx),
        "generate_splatter_brush" => generate_splatter(input, ctx),
        "create_material" => create_material(input, ctx),
        "create_stamp_from_svg" => stamp_from_svg(input, ctx),
        "generate_image_stamp" => image_stamp(input, ctx),
        "import_image_from_url" => import_from_url(input, ctx),
        other => Err(anyhow::anyhow!("unknown tool {other}")),
    };
    result.unwrap_or_else(|e| ToolOutput::error(format!("{e:#}")))
}

fn str_arg<'a>(input: &'a Value, key: &str) -> Result<&'a str> {
    input[key]
        .as_str()
        .filter(|s| !s.trim().is_empty())
        .with_context(|| format!("missing '{key}'"))
}

fn index() -> Result<Arc<matfetch::catalog::Index>> {
    static INDEX: OnceLock<Mutex<Option<Arc<matfetch::catalog::Index>>>> = OnceLock::new();
    let slot = INDEX.get_or_init(|| Mutex::new(None));
    let mut guard = slot.lock().unwrap();
    if let Some(index) = guard.as_ref() {
        return Ok(index.clone());
    }
    let index = Arc::new(matfetch::load_index(&library_root(), &matfetch::http::agent())?);
    *guard = Some(index.clone());
    Ok(index)
}

fn search_online_assets(input: &Value) -> Result<ToolOutput> {
    let terms: Vec<String> = str_arg(input, "query")?.to_lowercase().split_whitespace().map(str::to_string).collect();
    let kind = input["kind"].as_str();
    let limit = input["limit"].as_u64().unwrap_or(15).clamp(1, 50) as usize;
    let index = index()?;
    let root = library_root();

    let mut scored: Vec<(usize, usize, &matfetch::catalog::Asset)> = index
        .assets
        .iter()
        .filter(|a| kind.is_none_or(|k| a.kind == k))
        .filter_map(|a| {
            let name = a.name.to_lowercase();
            let category = a.category.to_lowercase();
            let mut hits = 0;
            let mut score = 0;
            for t in &terms {
                let mut s = 0;
                if name.contains(t.as_str()) {
                    s += 3;
                }
                if a.tags.iter().any(|tag| tag == t) {
                    s += 3;
                } else if a.tags.iter().any(|tag| tag.contains(t.as_str())) {
                    s += 1;
                }
                if category.contains(t.as_str()) {
                    s += 2;
                }
                if s > 0 {
                    hits += 1;
                    score += s;
                }
            }
            (hits > 0).then_some((hits, score, a))
        })
        .collect();
    scored.sort_by(|x, y| y.0.cmp(&x.0).then(y.1.cmp(&x.1)));

    let results: Vec<Value> = scored
        .iter()
        .take(limit)
        .map(|(hits, _, a)| {
            json!({
                "id": a.id,
                "name": a.name,
                "kind": a.kind,
                "category": a.category,
                "tags": a.tags.iter().take(10).collect::<Vec<_>>(),
                "installed_folder": matfetch::download::installed(&root, a).map(|_| a.dir.clone()),
                "matched_terms": format!("{hits}/{}", terms.len()),
            })
        })
        .collect();
    Ok(ToolOutput::json(&json!({"total_matches": scored.len(), "results": results})))
}

fn download_asset(input: &Value, ctx: &WorkerCtx) -> Result<ToolOutput> {
    let id = str_arg(input, "id")?;
    let res = input["resolution"].as_str().unwrap_or("2k");
    let index = index()?;
    let asset = index.get(id).with_context(|| format!("no asset with id {id}"))?;
    let variant = [format!("{res}-jpg"), format!("{res}-png")]
        .into_iter()
        .find(|v| asset.variants.contains_key(v))
        .or_else(|| asset.variants.keys().find(|k| k.starts_with(res)).cloned())
        .or_else(|| asset.variants.keys().next().cloned())
        .context("asset has no downloadable variant")?;
    let root = library_root();
    let dir = matfetch::download::fetch_one(&matfetch::http::agent(), &root, asset, &variant, None)?;
    let _ = ctx.events.send(Event::LibraryChanged);
    let kind = if asset.is_brush() { "brush" } else { "material" };
    Ok(ToolOutput::json(&json!({
        "downloaded": asset.name,
        "folder": dir.file_name().map(|f| f.to_string_lossy().to_string()),
        "library_kind": kind,
        "variant": variant,
        "license": asset.license,
    })))
}

fn generate_splatter(input: &Value, ctx: &WorkerCtx) -> Result<ToolOutput> {
    let name = str_arg(input, "name")?;
    let style = matfetch::splatter::Style::parse(str_arg(input, "style")?)?;
    let seed = input["seed"].as_u64().unwrap_or_else(|| {
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1)
    });
    let density = input["density"].as_f64().unwrap_or(1.0) as f32;
    let dir = matfetch::splatter::generate_one(&library_root(), name, style, seed, density, 1024)?;
    let _ = ctx.events.send(Event::LibraryChanged);

    let opacity = image::open(dir.join("opacity.png"))?.to_luma8();
    let preview = image::RgbaImage::from_fn(opacity.width(), opacity.height(), |x, y| {
        image::Rgba([90, 0, 0, opacity.get_pixel(x, y).0[0]])
    });
    let folder = dir.file_name().unwrap_or_default().to_string_lossy().to_string();
    Ok(ToolOutput::text(format!(
        "Created greyscale splatter brush '{folder}' (style {}, seed {seed}). It paints with the brush colour; load it with use_library_brush. Preview (shown in dark red):",
        style.name()
    ))
    .with_image(images::stamp_block(&preview, 384)?)
    .with_preview(images::fit(&preview, 256)))
}

fn parse_hex(s: &str) -> Result<[u8; 3]> {
    let h = s.trim().trim_start_matches('#');
    if h.len() != 6 {
        bail!("colour must be #RRGGBB, got {s:?}");
    }
    let v = u32::from_str_radix(h, 16).with_context(|| format!("bad colour {s:?}"))?;
    Ok([(v >> 16) as u8, (v >> 8) as u8, v as u8])
}

pub fn hex_to_srgb(s: &str) -> Result<[f32; 3]> {
    Ok(parse_hex(s)?.map(|c| c as f32 / 255.0))
}

fn create_material(input: &Value, ctx: &WorkerCtx) -> Result<ToolOutput> {
    let name = str_arg(input, "name")?;
    let tags = input["tags"]
        .as_array()
        .map(|a| a.iter().filter_map(|t| t.as_str().map(str::to_lowercase)).collect())
        .unwrap_or_default();
    let m = matfetch::splatter::FlatMaterial {
        name,
        basecolor: parse_hex(str_arg(input, "color")?)?,
        roughness: input["roughness"].as_f64().unwrap_or(0.5) as f32,
        metallic: input["metallic"].as_f64().unwrap_or(0.0) as f32,
        variation: input["variation"].as_f64().unwrap_or(0.3) as f32,
        tags,
        description: "Created by the zestsurface assistant".into(),
    };
    let dir = matfetch::splatter::generate_flat_material(&library_root(), &m, 1024)?;
    let _ = ctx.events.send(Event::LibraryChanged);
    let folder = dir.file_name().unwrap_or_default().to_string_lossy().to_string();
    Ok(ToolOutput::text(format!("Created material '{folder}'. Apply it with apply_library_material.")))
}

/// Shared tail of the image-producing tools: save, announce, preview.
fn finish_stamp(name: &str, img: image::RgbaImage, source: &str, description: String, ctx: &WorkerCtx) -> Result<ToolOutput> {
    let img = images::trim(&images::fit(&img, 2048), 4);
    if img.pixels().all(|p| p.0[3] < 8) {
        bail!("the image came out empty/transparent");
    }
    let colored = img.pixels().any(|p| {
        let [r, g, b, a] = p.0.map(i32::from);
        a > 64 && ((r - g).abs() > 24 || (g - b).abs() > 24 || (r - b).abs() > 24)
    });
    let dir = images::save_stamp(&library_root(), name, &img, vec![source.into(), "assistant".into()], source, description)?;
    let _ = ctx.events.send(Event::LibraryChanged);
    let folder = dir.file_name().unwrap_or_default().to_string_lossy().to_string();
    Ok(ToolOutput::text(format!(
        "Saved {} stamp brush '{folder}' ({}x{}). Load it with use_library_brush, then set_brush tool=stamp. Preview on white:",
        if colored { "colour" } else { "monochrome" },
        img.width(),
        img.height()
    ))
    .with_image(images::stamp_block(&img, 512)?)
    .with_preview(images::fit(&img, 256)))
}

fn stamp_from_svg(input: &Value, ctx: &WorkerCtx) -> Result<ToolOutput> {
    let name = str_arg(input, "name")?;
    let svg = str_arg(input, "svg")?;
    let size = input["size"].as_u64().unwrap_or(1024).clamp(256, 2048) as u32;
    let img = images::render_svg(svg, size)?;
    finish_stamp(name, img, "svg", "Drawn as SVG by the zestsurface assistant".into(), ctx)
}

fn image_stamp(input: &Value, ctx: &WorkerCtx) -> Result<ToolOutput> {
    let name = str_arg(input, "name")?;
    let prompt = str_arg(input, "prompt")?;
    if ctx.settings.images_via_gemini() {
        let key = ctx.settings.resolved_gemini_key().context("Nano Banana needs a Gemini API key (assistant settings, or $GEMINI_API_KEY)")?;
        let (w, h) = (input["width"].as_u64().unwrap_or(1024), input["height"].as_u64().unwrap_or(1024));
        let aspect = match w as f64 / h.max(1) as f64 {
            r if r > 1.5 => "16:9",
            r if r > 1.15 => "4:3",
            r if r < 0.67 => "9:16",
            r if r < 0.87 => "3:4",
            _ => "1:1",
        };
        let model = &ctx.settings.image_model;
        let mut img = gemini::generate_image(&ctx.settings.gemini_api_base, &key, model, prompt, aspect, ctx.cancel)?;
        if input["remove_white_background"].as_bool().unwrap_or(true) {
            img = images::white_to_alpha(&img);
        }
        return finish_stamp(name, img, "nano-banana", format!("{model}: {prompt}"), ctx);
    }
    let job = comfy::Job {
        base_url: &ctx.settings.comfy_url,
        checkpoint: &ctx.settings.comfy_checkpoint,
        prompt,
        negative: input["negative"].as_str().unwrap_or(
            "photo, skin, arm, body, person, background scene, frame, border, text, watermark, signature, blurry, lowres",
        ),
        seed: input["seed"].as_u64().unwrap_or_else(|| {
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
        }),
        width: (input["width"].as_u64().unwrap_or(1024).clamp(256, 2048) as u32) / 64 * 64,
        height: (input["height"].as_u64().unwrap_or(1024).clamp(256, 2048) as u32) / 64 * 64,
        steps: input["steps"].as_u64().unwrap_or(28).clamp(4, 80) as u32,
    };
    let workflow = workflow_path();
    let mut img = comfy::generate(&job, &workflow, ctx.cancel)?;
    if input["remove_white_background"].as_bool().unwrap_or(true) {
        img = images::white_to_alpha(&img);
    }
    finish_stamp(name, img, "comfyui", format!("ComfyUI: {prompt}"), ctx)
}

pub fn workflow_path() -> PathBuf {
    crate::config::file("comfy_workflow.json")
}

/// Hosts a web page must not be able to point the downloader at.
fn is_local_host(host: &str) -> bool {
    let host = host.trim_start_matches('[').trim_end_matches(']').to_lowercase();
    if host == "localhost" || host.ends_with(".localhost") || host.ends_with(".local") {
        return true;
    }
    match host.parse::<std::net::IpAddr>() {
        Ok(std::net::IpAddr::V4(ip)) => ip.is_loopback() || ip.is_private() || ip.is_link_local() || ip.is_unspecified(),
        Ok(std::net::IpAddr::V6(ip)) => ip.is_loopback() || ip.is_unspecified(),
        Err(_) => false,
    }
}

fn import_from_url(input: &Value, ctx: &WorkerCtx) -> Result<ToolOutput> {
    let url = str_arg(input, "url")?;
    let name = str_arg(input, "name")?;
    let rest = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))
        .context("only http(s) URLs can be imported")?;
    let host = rest.split(['/', '?', '#']).next().unwrap_or("");
    let host = host.rsplit('@').next().unwrap_or(host);
    let host = if host.starts_with('[') { host.split(']').next().unwrap_or(host) } else { host.split(':').next().unwrap_or(host) };
    if is_local_host(host) {
        bail!("refusing to fetch from a local/private address");
    }

    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(std::time::Duration::from_secs(60)))
        .build()
        .into();
    let bytes = agent
        .get(url)
        .header("User-Agent", matfetch::http::UA)
        .call()
        .with_context(|| format!("GET {url}"))?
        .body_mut()
        .with_config()
        .limit(40 << 20)
        .read_to_vec()?;
    let img = image::load_from_memory(&bytes)
        .context("that URL is not an image (use web_fetch on the page to find the direct image link)")?
        .to_rgba8();

    match str_arg(input, "as")? {
        "brush" => {
            let img = if input["remove_white_background"].as_bool().unwrap_or(false) { images::white_to_alpha(&img) } else { img };
            finish_stamp(name, img, "web", format!("Imported from {url}"), ctx)
        }
        "material" => {
            let folder = crate::library::sanitize(name);
            let dir = library_root().join("materials").join(&folder);
            std::fs::create_dir_all(&dir)?;
            let img = images::fit(&img, 4096);
            img.save(dir.join("basecolor.png"))?;
            let manifest = json!({
                "id": format!("web:{folder}"), "name": name, "source": "web", "kind": "material",
                "license": "unknown — check the source", "authors": [], "page": url,
                "variant": format!("{}x{}", img.width(), img.height()), "tags": ["web"],
                "maps": {"basecolor": "basecolor.png"},
            });
            std::fs::write(dir.join("material.json"), serde_json::to_vec_pretty(&manifest)?)?;
            let _ = ctx.events.send(Event::LibraryChanged);
            Ok(ToolOutput::text(format!("Saved material '{folder}' (basecolor only). Apply it with apply_library_material."))
                .with_image(images::jpeg_block(&img, 512)?)
                .with_preview(images::fit(&img, 256)))
        }
        other => bail!("'as' must be brush or material, got {other}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn definitions_are_wellformed() {
        let defs = definitions();
        let defs = defs.as_array().unwrap();
        let names: Vec<&str> = defs.iter().filter_map(|d| d["name"].as_str()).collect();
        for t in EDITOR_TOOLS {
            assert!(names.contains(t), "{t} not declared");
        }
        let mut sorted = names.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), names.len(), "duplicate tool names");
        for d in defs.iter().filter(|d| d.get("input_schema").is_some()) {
            assert_eq!(d["input_schema"]["type"], "object");
        }
    }

    #[test]
    fn local_hosts_are_refused() {
        for h in ["localhost", "127.0.0.1", "10.1.2.3", "192.168.0.5", "[::1]", "printer.local"] {
            assert!(is_local_host(h), "{h}");
        }
        assert!(!is_local_host("upload.wikimedia.org"));
    }
}

/// Runs the disk-writing tools for real. Ignored by default: run it with HOME
/// pointed at a scratch dir (holding a copy of the matfetch index) so it never
/// touches the real library.
#[cfg(test)]
mod smoke {
    use super::*;

    #[test]
    #[ignore]
    fn worker_tools_end_to_end() {
        let settings = Settings::default();
        let cancel = Arc::new(AtomicBool::new(false));
        let (tx, rx) = mpsc::channel();
        let ctx = WorkerCtx { settings: &settings, cancel: &cancel, events: &tx };
        let out_dir = std::env::var("ZESTSURFACE_SMOKE_OUT").map(PathBuf::from).ok();

        let found = run_worker_tool("search_online_assets", &json!({"query": "leaking stain"}), &ctx);
        assert!(!found.is_error, "{:?}", found.content);
        assert!(found.summary().contains("total_matches") || found.content[0]["text"].as_str().unwrap().contains("Leaking"));

        let splat = run_worker_tool("generate_splatter_brush", &json!({"name": "blood_splat_test", "style": "splat", "seed": 3}), &ctx);
        assert!(!splat.is_error, "{:?}", splat.content);
        assert_eq!(splat.content[1]["type"], "image");

        let mat = run_worker_tool("create_material", &json!({"name": "Fresh Blood", "color": "#5a0000", "roughness": 0.15}), &ctx);
        assert!(!mat.is_error, "{:?}", mat.content);

        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
            <path d="M100 30 C60 0 10 40 100 170 C190 40 140 0 100 30Z" fill="none" stroke="#111" stroke-width="8"/>
            <path d="M100 60 L100 140 M70 100 L130 100" stroke="#b01020" stroke-width="10" stroke-linecap="round"/></svg>"##;
        let tattoo = run_worker_tool("create_stamp_from_svg", &json!({"name": "heart_cross_tattoo", "svg": svg}), &ctx);
        assert!(!tattoo.is_error, "{:?}", tattoo.content);
        assert!(tattoo.summary().contains("colour stamp"), "{}", tattoo.summary());

        let bad = run_worker_tool("import_image_from_url", &json!({"url": "http://127.0.0.1:8188/x.png", "name": "x", "as": "brush"}), &ctx);
        assert!(bad.is_error);

        // The new folders must be what the app's own scanners pick up.
        let brushes = crate::library::scan_brushes();
        let tat = brushes.iter().find(|b| b.name == "heart_cross_tattoo").expect("tattoo scanned");
        assert!(tat.tip_source().unwrap().ends_with("basecolor.png"));
        assert!(brushes.iter().any(|b| b.name == "blood_splat_test"));
        assert!(crate::library::scan().iter().any(|m| m.name == "Fresh_Blood"));
        assert!(rx.try_iter().filter(|e| matches!(e, Event::LibraryChanged)).count() >= 3);

        if let Some(dir) = out_dir {
            for (name, out) in [("splat", &splat), ("tattoo", &tattoo)] {
                out.preview.as_ref().unwrap().save(dir.join(format!("{name}.png"))).unwrap();
            }
        }
    }
}
