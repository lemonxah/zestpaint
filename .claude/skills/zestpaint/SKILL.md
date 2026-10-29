---
name: zestpaint
description: Drive the ZestPaint 3D texture painting editor through its MCP tools — texture sets, layers, masks, fill materials, brushes, stamps, tattoos, blood/liquid splatter, decals, CC0 material downloads, smart masks, baking, viewport screenshots. Use whenever the user asks to paint, texture, decorate or set up anything in ZestPaint (or "the painter", "the editor", "my model/avatar"), e.g. "add a tattoo on the arm", "make the jacket leather", "blood splatter on the shirt", "find a fabric texture", "what does it look like now".
---

# ZestPaint via MCP

ZestPaint is the user's Substance-Painter-style texture painter (Rust/Vulkan). The `zestpaint` MCP server forwards tool calls into the **running** editor — the same tools its built-in assistant uses. You set things up; the user does the brush strokes and stamp clicks in the viewport.

## Before anything

1. Call `get_editor_state`. If it errors with "isn't running", call `open_zestpaint` (pass `project` for a `.zestpaint` file), then retry.
2. If the `zestpaint` tools aren't available at all, the MCP server isn't registered or isn't running. The editor serves MCP itself: the plug button in the Assistant panel header starts it, and Assistant ▸ settings (gear) shows a Copy button for the one-time `claude mcp add --scope user --transport http zestpaint http://127.0.0.1:47821/mcp --header "Authorization: Bearer …"` — tell the user to run that, then restart Claude Code. The older stdio route still works too: `claude mcp add --scope user zestpaint -- <path-to>/zestpaint mcp` (inside the repo, `.mcp.json` registers it via `cargo run --release -- mcp`); only that route has `open_zestpaint`.
3. Call `capture_viewport` to see what the user sees (`include_textures: true` adds the active set's base colour and painted-layer composite in UV space). Do this whenever "this", "here" or "that" is ambiguous, and after your changes to check them.

## Working loop

- **Opening a model.** `import_model` with an absolute .fbx/.gltf/.glb path replaces the whole document (ask first if anything is painted). It returns the new texture sets.
- **Pieces and local view.** `view_pieces` isolates (Blender's `/`: show only these, frame them), hides, shows only, selects or frames pieces by index, "Object / Material", material or object name. Isolate the piece being painted when other parts are in the way — hidden pieces don't occlude painting or bakes — and `leave_isolation` afterwards (it restores visibility and the camera). The user's keys: click / Shift+click select, A / Alt+A, H / Shift+H / Alt+H, `/`, F, Home.
- **Pick the texture set first.** A model is split into texture sets (object + material). Layers, masks and painting all belong to the active set — `select_texture_set` by name or index from `get_editor_state`.
- **Layers stack bottom → top.** `add_layer` for paint/stamps; `apply_library_material` makes a fill layer with a **black mask** (invisible until the mask is painted, generated with `smart_mask`, or `layer_action fill_mask_white`).
- **Brush setup is yours, the click is theirs.** `set_brush` (tool, target material/mask, channels, colour, size, …) then tell the user exactly where to paint or click. `tool=stamp` places the whole tip once per click (decals, tattoos, splats). In-viewport: Shift+scroll rotates the tip, `[` `]` or Ctrl+scroll resizes.
- **Everything is undoable** (`undo` / `redo`, or Ctrl+Z for the user). Prefer undoing a bad step over piling fixes on top.
- **Verify** with `capture_viewport` after visible changes, and look at the image previews that generation tools return; iterate when they're off.
- Calls that change the document run on the editor's UI thread one at a time. Don't fire many in parallel when order matters.

- **Matcaps are layers.** Each matcap layer holds one matcap plus its own greyscale mask (white = shows); a material can have any number, drawn bottom to top. `set_matcap` with a `builtin` (each sets its natural blend; see the tool's enum) or an `image_path`, plus `blend`/`strength`: it changes the active matcap layer or adds one (`new_layer: true` forces a new one), and `sets: [...]` adds one to several texture sets at once (e.g. Chrome on every Metal material). New layers start all white (whole material); `fill: "black"` empties the mask so the user can paint where it shows — set the brush up for them (the mask is painted like any shader mask: White — show / Black — hide). It returns a screenshot; matcaps are view-dependent, so check from another angle with `set_view`. Raster view only. `get_editor_state` lists each layer's `matcap`.
- **Judging final looks.** `set_render_mode path_traced`, wait a few seconds, `capture_viewport`; switch back to `raster` before the user paints (painting is off while path traced).

- **Placing overlays yourself (decals, tattoos, logos, splats).** The loop is: aim → stamp → look → undo and retry if off.
  1. Get the image in: `import_image` with `as: "stamp"` (a local PNG with transparency), or `use_library_brush` / `create_stamp_from_svg`.
  2. `add_layer` for it, and `set_brush` for colour/channels (`use_tip_color: true` keeps the image's own colours).
  3. `view_pieces isolate` the target piece if clothing covers it, then `set_view` (preset `back`/`front`/…, `frame` the piece, `zoom`). It returns a screenshot.
  4. `place_stamp` with x/y as fractions of that screenshot (0..1 from the top-left), `size` in viewport px (radius), `rotation_degrees`, `opacity` (1 for a solid decal). It returns a screenshot of the result; `undo` and adjust if it is off. Points that miss the active piece are refused with the reason.
- **Whole maps.** `import_image as=layer kind=…` lays a map painted or baked elsewhere over the whole UV layout as a new layer; `as=material_map` replaces the material's own slot (undoable); `as=fill_layer` for tileable textures. Check with `capture_viewport` (`include_textures: true` shows the UV-space result).

## Recipes

**Detail and resolution.** Sets paint at 2048 px unless changed. For fine linework (tattoos, small text, lace), check the set's `resolution` in `get_editor_state` and raise it with `set_texture_resolution` (4096 usually; 8192 only when needed — each step is 4x the GPU memory, and the tool refuses sizes that would overload the GPU). Existing paint is resampled; the change is undoable. Render stamps big enough for the target size (≈1024–2048 px at 4K), since a stamp shrunk a lot onto the skin loses its thin lines.

**Tattoo.** Design with `create_stamp_from_svg` (bold clean linework, transparent background, black ink unless colour is asked for; viewBox square, fill the frame) — or `generate_image_stamp` for shaded/realistic art (Nano Banana when a Gemini key is set in the assistant settings, else the user's ComfyUI). Check the returned preview, refine if needed. Then `add_layer` named after the design → `use_library_brush` with the stamp → `set_brush` tool=stamp, target=material, channels=["color"] (add roughness ≈0.45 for slightly glossy ink), `use_tip_color=true` for colour designs or colour `#151515` for monochrome. Suggest layer opacity 0.85–0.95 and Multiply blend (`set_layer`) for healed ink. Tell the user to click on the skin where it goes.

**Blood / liquid splatter.** `create_material` (fresh blood ≈ `#4a0000`–`#6a0a0a`, roughness 0.1–0.2; dried ≈ `#2e0805`, roughness 0.55) → `apply_library_material` (fill layer, black mask, brush switched to the mask) → `generate_splatter_brush` with a couple of styles/seeds → `use_library_brush` → `set_brush` tool=stamp, target=mask, mask_value=1 → user clicks where splats go. Blood pooled in creases: `smart_mask` cavity_dirt on that layer instead. Drips run along the stamp's bottom edge — orient with tip rotation.

**Materials and textures.** `search_online_assets` (free CC0 catalogue) → `download_asset` → `apply_library_material` / `use_library_brush`. `list_library` shows what's already installed. Only if the catalogue has nothing: find an image with web search and `import_image_from_url` using a **direct** image URL; tell the user the source and licence, prefer CC0 / public domain.

**Emission.** A material's emission is its Emission colour (Properties ▸ Material ▸ Surface) × its Emission map × its Emission mask (white glows, black doesn't). Set the maps with `import_image as=material_map kind=emissive` / `kind=emission_mask`. If a piece renders as one flat bright colour that paint and fills can't change, its material emission is swamping it: give it a mask, or have the user set the Emission colour to black. Painted glow goes on a layer's emissive channel (brush channel `emissive`, or `import_image as=layer kind=emissive`) and respects that layer's mask.

**Wear and masks.** `smart_mask` on a fill layer for edge wear, cavity dirt and similar; it needs mesh maps — run `bake` first if `get_editor_state` shows them unbaked.

## Reporting back

Be brief: what you set up (layer names, brush settings) and the user's next physical action (which tool is active, where to click, which keys adjust it).
