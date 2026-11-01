# ZestSurface UI redesign — handover for Claude Code

You are implementing a re-layout of ZestSurface's desktop UI (a Linux 3D texture painter in the spirit of Substance Painter). The screens in `screens/` are the target. **This is a layout and interaction change, not a feature change**: every control that exists today must still exist afterwards, and no rendering, painting, baking or export behaviour changes unless this document says so.

Read this whole file before touching code. Then read the repo to learn which UI toolkit ZestSurface uses (for example egui or Dear ImGui) and map the spec onto it. Do not port the HTML in `design-source/`. It is a mockup that shows exact colours, spacing and states, and nothing more.

---

## 0. Ground rules

1. **Nothing gets lost.** Section 8 lists every existing control and its new location. Tick each one off as you move it. If a control has no clear home, stop and ask; do not drop it.
2. **Move UI, don't rewrite logic.** Panels should call the same functions they call today. If one of those functions is welded to the old panel code, pull it out into a shared function first, as its own commit, before moving its UI.
3. **Work in phases (section 9).** Each phase must build, run and keep all existing features working before the next one starts. One phase per PR or commit series.
4. **Ask before you invent.** Items tagged **NEW** are small additions approved for this redesign. Items tagged **PLANNED** get space in the UI only: show them disabled with a "Coming soon" tooltip, and do not implement them.
5. **Proposed shortcuts are provisional.** Shortcuts marked † in section 7 are new. Check them against the existing keymap and report any clash instead of overriding a binding.
6. Keep all user-facing strings in one place if the codebase already has a pattern for that.

---

## 1. Screen index (`screens/`)

All screens are 2× PNGs. The workspace frame is 1600×960 logical px.

| File | Shows |
|---|---|
| `01-workspace-paint-tool.png` | Default workspace: Paint tool, Edge wear layer, Layer tab, Display & lighting popover open |
| `02-workspace-display-popover-closed.png` | Same with the popover closed (the normal resting state) |
| `03-layer-mask-tab-into-mask.png` | "Into: Mask" with Reveal/Hide, Mask tab for a layer that has a mask (mask ops + smart mask) |
| `04-no-mask-warning-and-add-mask.png` | Into: Mask on a layer *without* a mask: inline warning in the context bar, and Add black/white in the Mask tab |
| `05-shader-mask-layer.png` | Shader-mask layer selected: the left dock swaps the channels for the White/Black/Grey box; Layer tab shows "Mask for" |
| `06-material-tab-maps-surface-bake.png` | Material tab: map slots, Surface (OpenPBR), Bake & UVs, Save to library |
| `07-place-tool-projection-tab.png` | Place tool (P): context bar for the projection, Projection tab with placement and tiling |
| `08-clone-tool-context-bar.png` | Clone: source warning + Aligned in the context bar |
| `09-stamp-tool-context-bar.png` | Stamp: stamp image + Browse/Load in the context bar |
| `10-select-tool-path-traced.png` | Select: Hide / Hide others / Unhide all / Isolate; Path traced mode with spp counter |
| `11-pick-tool.png` | Pick: sampled swatch + explanation |
| `12-library-materials.png` | Unified library window, Materials tab |
| `13-library-brushes.png` | Unified library window, Brushes & stamps tab (import buttons appear) |
| `14-menu-map-and-command-search.png` | Every menu, the shader-mask submenu, and Ctrl K command search |
| `15-export-import-progress.png` | Export maps, Import maps, background-task cards and toasts |

The `design-source/*.dc.html` files are the mockup sources. Open them in a text editor to read exact values.

---

## 2. Design tokens

Define these once, as theme constants or a style struct, and use them everywhere. Don't scatter hex literals through the code.

### Colour
| Token | Hex | Use |
|---|---|---|
| `bg.chrome` | `#0B0C0E` | Menu bar, status bar |
| `bg.app` | `#111214` | Window background behind docks |
| `bg.bar` | `#17181C` | Context bar, tool rail, library toolbar |
| `bg.panel` | `#1A1B1F` | Docks, dialogs, popovers |
| `bg.viewport` | `#1E2024` | 3D viewport clear colour |
| `bg.field` | `#0F1012` | Inputs, selects, segmented-control troughs |
| `bg.button` | `#26282D` | Secondary buttons |
| `bg.seg.on` | `#2E3036` | Selected segment |
| `bg.selected` | `#25261F` | Selected row / tile (pair with accent border) |
| `line` | `#24262B` / `#2A2C31` | Region borders / inner dividers |
| `line.control` | `#33363C` | Button borders |
| `text` | `#E4E5E9` | Primary text |
| `text.strong` | `#F2F3F5` | Active segment / tab label |
| `text.muted` | `#A3A6AF` | Labels, section headers |
| `text.dim` | `#8D909A` | Secondary info, shortcuts |
| `accent` | `#F5C84B` | Active tool, selection border, primary button, slider fill, progress |
| `on.accent` | `#1A1500` | Text on accent |
| `warn.bg` / `warn.fg` | `#3A2612` / `#FFB366` | Inline warnings (no clone source, no mask), destructive-ish actions (Auto-unwrap) |
| `ok.bg` / `ok.fg` | `#16241A` / `#9FD8A2` | Success status / toast |
| `err.bg` / `err.fg` | `#2A1716` / `#FF9B94` | Error toast, Remove mask |
| `axis.x/y/z` | `#E5534B` / `#57AB5A` / `#539BF5` | Gizmo, and the left edge of X/Y/Z number fields |

Shader-mask kind colours (tags, submenu dots, the frame while painting a mask): Emission `#F0883E`, Matcap `#8FB3FF`, Matcap 2 `#6F8FE0`, Rim light `#7FD6FF`, Outline `#C9CBD1`, Shadow `#7C6FD6`, Specular `#F2F2A0`, Glitter `#F59AD8`, Detail `#9AD07A`, Dissolve `#E5534B`, AudioLink `#4EC9B0`, Custom `#8D909A`. If the app already has colours for these kinds, keep the existing ones.

### Type
- UI font: **IBM Plex Sans** 400/500/600. Numbers, shortcuts and file names: **IBM Plex Mono** 400/500. Both are OFL-licensed; bundle them with the app.
- Sizes: body 12 px; section headers 11 px, semibold, uppercase, 0.06 em tracking, `text.muted`; dialog titles 15 px semibold; badges 9.5 px mono.

### Metrics
- Radii: 5–6 px for controls, 7–8 px for rows and tool buttons, 10–12 px for popovers and dialogs.
- Control heights: 24–28 px in docks, 30–34 px in dialogs. Tool buttons 40×40. Row height for sliders 26–32 px.
- Slider rows use a three-column grid: label (≈84 px) | slider (fills) | value (≈36 px, mono, right-aligned).
- Spacing: 4/6/8/12/14/16 px. Dock padding 12 px.

---

## 3. Window layout

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ MENU BAR 34px  logo · File Edit View Layer Bake Library Window Help · file │ Search (Ctrl K) · undo/redo
├─────────────────────────────────────────────────────────────────────────────┤
│ CONTEXT BAR ≥44px  [Tool name + key] | options for the active tool …        │
├──┬──────────┬──────────────────────────────────────────┬────────────────────┤
│R │ LEFT     │ VIEWPORT                                 │ RIGHT DOCK 320–340 │
│A │ DOCK     │  top-left: Object › Set · Layer          │ Texture sets (coll.)│
│I │ 268px    │  top-right: Raster|Path · view toggles · │ Layers + toolbar   │
│L │ Brush    │    Display & lighting popover            │ Tabs: Layer│Mask│  │
│  │ tip +    │  bottom-left: axis gizmo                 │  Projection│Material│
│52│ channels │  bottom-centre: empty-state chip         │ (scrolls)          │
├──┴──────────┴──────────────────────────────────────────┴────────────────────┤
│ STATUS BAR 28px  last result · task progress+Cancel · tool hint · undo mem · render │
└─────────────────────────────────────────────────────────────────────────────┘
```

The organising rule: the **left side is how you paint** (tool, brush, channels). The **right side is what you paint into** (texture set, layer, mask, projection, base material). The viewport only carries overlays about the view itself.

- Docks keep fixed widths; the viewport takes the remaining space.
- Only the right dock's tab area scrolls. The layer list gets its own scroll if it grows past about 40% of the dock's height.
- Window ▸ Brush / Texture sets / Layers / Properties toggle each panel (F5–F8†), and Window ▸ Reset layout restores the default. If the toolkit supports docking, make panels dockable. Otherwise, fixed panels are fine for this pass.

---

## 4. Region specs

### 4.1 Menu bar
- Menus and their contents exactly as in `14-menu-map-and-command-search.png` (also listed in section 7).
- Centre: `<model file> — <active set>`.
- Right: **command search** button (opens the palette; see 4.8), then Undo and Redo icon buttons.

### 4.2 Context bar (replaces the old toolbar)
Left: an accent square, the tool name and a key badge. Then a divider, then the options for the active tool:

| Tool | Context bar contents |
|---|---|
| Paint / Erase / Smudge / Clone / Stamp | **Into** [Material \| Mask]; when Mask: [Reveal \| Hide]; when Mask and the layer has no mask: orange chip "Layer has no mask" + **Add mask** (opens the Mask tab). **Stroke** [Freehand \| Line \| Stamp]. Size (2–200 px), Hardness (0–1), Flow (0.01–1; labelled **Strength** for Smudge), Angle (−180…180°), X symmetry, Fill [Brush, Triangle, UV island, Whole piece] |
| + Clone | Source status chip ("Ctrl+click the model to set a source" in warn colours, or "Source set" in ok colours) + Aligned |
| + Stamp | Stamp image name + Browse… + Load image… |
| Pick | Sampled swatch + "Samples base colour… release to return to the previous brush" |
| Select | Hide (H), Hide others (Shift H), Unhide all (Alt H), Isolate (/) buttons with their keys |
| Place | Layer name, projection mode · axis, Repeat, "Numeric fields…" (jumps to the Projection tab), Reset |
| Orbit | One-line explanation |

The context bar wraps to a second line when the window is narrow. Don't clip it.

### 4.3 Tool rail (52 px)
Groups separated by thin dividers: **Paint B, Erase E, Smudge G, Clone C, Stamp T** │ **Pick I, Place P** │ **Select S, Orbit R**. Each button is 40×40 with an icon and the key letter underneath, plus a tooltip "Name  Key". Active state: accent border and icon, 7% white fill. At the bottom: a large swatch for the current paint colour that opens the colour picker.

### 4.4 Left dock: Brush (268 px)
- Header: the tool's brush name (Brush / Eraser / Smudge / Clone / Stamp) and a "⋯" presets menu.
- **Tip** block: a 64 px preview, the tip name, Browse… / Load… / Reset, and "Use tip colours" (disabled for greyscale tips).
- **Paint channels**: one row per channel (checkbox, name, value control). Color uses a swatch and its hex; Roughness, Metallic and Height use sliders; Emissive uses a swatch plus an intensity slider; Opacity uses a slider. Disabled channels dim their controls to `text.dim`. The header shows "N of 6 on". Then Bump strength.
- **When the active layer is a shader mask**, replace the channels with a framed box in that kind's colour: "Painting <Kind> mask", [White — show] [Black — hide] and a Grey slider.
- Footer: the fill-mode line, e.g. "Fill mode: Brush · mesh has 1 piece, 2 UV islands".

### 4.5 Viewport overlays
- **Top-left:** breadcrumb chip `Object › Set · Layer`.
- **Top-right:** [Raster | Path traced]. In Path traced, add a `N spp` chip. Then icon toggles: UV checker, Show mask on model, Override material, and **Display & lighting** (sun icon), which opens the popover.
- **Display & lighting popover (272 px):** Environment select (Procedural sky / loaded HDRIs) + Load HDRI…, Exposure 0.1–4, Env intensity 0–3, Key light 0–2, Light follows camera (disables Rotation/Height while on), Rotation −180…180°, Height −89…89°. Then the same three view toggles as checkboxes, with the override material's swatch and metal/rough values. Then the camera gesture cheat sheet. The popover closes on its X, Esc, or a click outside.
- **Bottom-left:** axis gizmo.
- **Bottom-centre:** the empty-state chip, shown only while the default sphere is loaded: "Default sphere — open a model to start" + **Open model…** + `Ctrl O`.
- Background-task cards and toasts stack in the **bottom-right** corner (see 4.9).
- Keep all existing viewport drawing: brush cursor, stamp preview, clone marker, line rubber band, projection wireframe.

### 4.6 Right dock
1. **Texture sets** (collapsible): an "Open model…" link in the header. A tree of objects, each with its materials (UDIM sets show as `Material [1002]`). Eyes on objects and materials. The selected set gets an accent border and `bg.selected`, and shows "N layers".
2. **Layers**: a toolbar with **+ Paint**, **+ Fill** (Fill from image…), **Mask ▾** (the 12 shader-mask kinds, colour-dotted), Duplicate **NEW**, Delete (disabled when one layer is left).
   - Each row has a drag handle (reorder by drag **NEW**; also Layer ▸ Move up/down), an eye, a 34 px layer thumbnail **NEW**, a mask thumbnail if the layer has a mask, the name, channel badges (COL RGH MTL HGT NRM OPA EMIT; shader masks get a coloured kind tag instead), and blend + opacity on the right in mono.
   - Selected row: accent border, `bg.selected`. Double-clicking the name renames it **NEW**.
   - Under the list: "Shader masks:" followed by links that jump to each mask layer.
   - If live thumbnails are expensive, render them small and only on change, or use a flat swatch of the layer's average colour for the first pass.
3. **Properties**, as tabs (remember the last tab):
   - **Layer**: Name field **NEW** (rename), Blend [Normal, Multiply, Add], Opacity, "Fill layer from brush values". For shader-mask layers: "Mask for" (12 kinds), Fill white, Fill black, and a help line.
   - **Mask**: if the layer has a mask, show the mask thumbnail, **Paint the mask** (primary; sets Into = Mask), Fill / Clear / Invert / Remove, and Show mask on model. Then **Smart mask**: a mesh-maps status line, [Edge wear | Cavity dirt], Amount 0–1, Contrast 0.1–4, Generate, Remove. With no mask, show two large buttons: "Add black mask · hidden, paint it in" and "Add white mask · visible, paint it out".
   - **Projection**: Fill with image… / From library…, Projection select [UV, Triplanar, Spherical, Cylindrical, Planar], Axis X/Y/Z (Spherical, Cylindrical and Planar only), Blend sharpness 1–32 (Triplanar only), **Place in viewport P** (switches to the Place tool). **Placement** (not for UV): Position XYZ, Rotation XYZ (axis-coloured fields), Size 0.02–4, Repeat, Reset. **Tiling**: U/V repeat 0.05–64 with a link toggle, U/V offset, Spin −180…180°, Reset. Before the layer has an image or material, show only the fill buttons and a hint. If the projection needs mesh maps while the material is hidden, show the existing status hint inline.
   - **Material**: "Object · Set — base material under the layers". Map slots Base color, Roughness, Metallic, Normal, Height, AO, each with a swatch, resolution or "—", load and remove. **Surface**: Transparent, Opacity, Emission colour, Specular, Coat, Coat roughness 0.03–1, Sheen, Sheen roughness 0.05–1. **Bake & UVs**: Bake AO, Bake mesh maps, AO strength 0–1, and "Auto-unwrap UVs (<Set>)…" in warn colours with the note "Resets this set's layers. Other sets keep their paint." This action should confirm before it runs. **Save to library**: a name field and Save.

### 4.7 Status bar (28 px)
Left to right:
- The last result, with an icon: ok/warn/error colours, the same messages the current status line shows.
- The current background task: label, a 90 px progress bar, %, and Cancel. Hide this when idle.
- The active tool's gesture hint, centred (the strings from the current toolbar hint).
- Undo memory "Undo N steps · used / budget GB". Clicking it opens Edit ▸ Undo budget…
- Render mode.

### 4.8 Command search (Ctrl K) **NEW**
A modal palette with fuzzy search over every menu command, plus the panel controls listed in section 8. Each result shows its menu or panel on the right. ↵ runs the command. Tab jumps to the panel that holds the control and focuses it. Esc closes. If time is short, the first version may search only the menu commands.

### 4.9 Dialogs, tasks, toasts
- **Export maps** (`15`): a header with Preset [All PBR maps, VRChat · Poiyomi, Last used] **NEW** and a close button. Two columns: Texture sets (All / None / Active) and Maps (All / None; the 8 maps as a two-column grid, plus Shader masks "one PNG each · N found"). Folder field + Browse…, a file-name preview (first 4 files + "N more"), and a footer with the file count formula and **Export N files** (disabled at 0). Naming, colour space and alpha rules are unchanged.
- **Import maps**: an "Onto" set select, then one row per file showing the file name and a map-type select. Unmatched files appear in warn colours with "Choose map…". Then the help line, Add files…, and Import N maps (counting only matched files).
- **Background tasks** **NEW**: model import, bakes and exports report progress through one task API. Each task gets a card in the bottom-right (title, sub-step, count, bar, Cancel where cancelling is safe), and the status bar mirrors the newest one. Bakes and exports must stop blocking the UI if that's practical. If it isn't, show the progress modally and say so in the PR.
- **Toasts**: success (with an action such as Open folder) and error (with Details). Toasts disappear after about 6 s. Errors stay until dismissed.

### 4.10 Library window (`12`, `13`)
One OS window replaces the separate material and brush browsers. It has tabs **Materials | Brushes & stamps**.
- Toolbar: a filter box (keep the current placeholder hints), quick tag chips, an "N of M" count, Clear, Rescan. The Brushes tab adds Import brush set… and Import folder….
- Left sources list: All / Poly Haven / ambientCG / My materials, or All / Imported sets / Images. A footer shows the folder path and mentions `matfetch`.
- A 5-column thumbnail grid. The selected tile gets an accent border.
- Details panel: preview, name, maps (or tip type), tags, source path, and the primary action (**Apply as fill layer** or **Use as brush tip**) with a one-line explanation.
- Opening it from Tip ▸ Browse… or Stamp ▸ Browse… selects the Brushes tab. Library ▸ Browse materials… selects Materials.
- Keep it a separate OS window so it can live on a second monitor.

---

## 5. Interaction rules
- The context bar, the left dock title and the status-bar hint follow the active tool. Pick still returns to the previous brush tool on release.
- Selecting a layer updates the viewport breadcrumb, the Properties tab contents, and the left dock (shader-mask box vs channels).
- Choosing Place (P), or "Place in viewport", selects the Projection tab.
- "Add mask" in the context-bar warning, and Mask ▸ Paint the mask, set Into = Mask.
- Disabled states (unchanged): Delete with one layer left, Export with nothing ticked, Use tip colours for greyscale tips, and the light rotation/height sliders while Light follows camera is on.
- Every control has a tooltip that shows its shortcut when it has one.
- Keyboard focus must be visible on every control, using an accent outline.

---

## 6. NEW and PLANNED
**NEW (implement):** command search; layer duplicate; rename field and double-click rename; drag-reorder; layer thumbnails; unified library window; export presets (store them in the existing config dir); background-task progress with Cancel; toasts; Bake all; Preferences entry (it can just host Undo budget for now); Window panel toggles and Reset layout.

**PLANNED (disabled placeholders only):** Open project…, Save project, Group into folder, 2D texture view.

Also add undo support for layer add, delete and reorder, and for projection/tiling edits, **only if** the existing undo system makes that straightforward. Otherwise list it in the PR as follow-up work.

---

## 7. Menus and shortcuts
† = a new shortcut. Check it for clashes before binding it.

- **File:** Open model… Ctrl O · Open project… Ctrl Shift O† (PLANNED) · Save project Ctrl S† (PLANNED) · Import maps… Ctrl Shift I† · Export maps… Ctrl Shift E† · Load HDRI environment… · Quit Ctrl Q†
- **Edit:** Undo Ctrl Z · Redo Ctrl Shift Z / Ctrl Y · Undo budget… · Keyboard shortcuts… F1† · Preferences… Ctrl ,†
- **View:** Render ● Raster / Path traced · Display & lighting… · UV checker · Show mask on model · Override material · Hide selection H · Hide others Shift H · Unhide all Alt H · Isolate /
- **Layer:** New paint layer Ctrl Shift N† · New fill from image… · New from library material… · New shader mask ▸ (12 kinds) · Duplicate Ctrl J† · Rename F2† · Group into folder Ctrl G† (PLANNED) · Move up Ctrl ]† · Move down Ctrl [† · Delete Del† · Add black mask · Add white mask · Invert mask · Remove mask · Fill from brush values
- **Bake:** (header shows the active set) Bake AO · Bake mesh maps · Bake all · Auto-unwrap UVs (<Set>)…
- **Library:** Browse materials… M† · Browse brushes… Shift B† · Save active material… · Import brush set… · Import folder… · Rescan libraries · Open library folder
- **Window:** Brush F5† · Texture sets F6† · Layers F7† · Properties F8† · Material library window · Brush library window · 2D texture view (PLANNED) · Reset layout
- **Help:** Tool gestures · Keyboard shortcuts F1 · About ZestSurface
- **Global:** Ctrl K command search†; tool keys B E G C T I P S R unchanged.

---

## 8. Feature relocation checklist (nothing may be lost)

| Existing control | New home |
|---|---|
| 8 tool toggle buttons + gesture hint | Tool rail; hint → status bar |
| Raster / Path traced switch, spp counter | Viewport top-right |
| Exposure, Env intensity, Key light, Light follows camera, Light rotation, Light height | Display & lighting popover |
| Load HDRI environment… | Popover + File menu |
| UV checker, Override material (+ colour/metal/rough) | Viewport toggles + popover + View menu |
| Show mask on model | Viewport toggle + Mask tab + View menu |
| Camera cheat sheet | Popover footer + Help ▸ Tool gestures |
| Open model… | File menu, Texture sets header, empty-state chip |
| Texture sets tree, eyes, UDIM sets | Right dock › Texture sets |
| Select tool hide/isolate keys | Context bar (Select) + View menu |
| Brush heading per tool | Left dock title |
| Stroke, Paint into, Mask stroke Reveal/Hide | Context bar |
| Size, Hardness, Flow, Tip angle/Rotation, X symmetry, Fill mode | Context bar |
| Tip label, Browse…, Load…, Reset, Use tip colours | Left dock › Tip |
| 7 paint channels + Bump strength | Left dock › Paint channels |
| Fill-mode pieces / islands line | Left dock footer |
| Stamp: Load stamp image…, Browse…, rotation | Context bar (Stamp); rotation = Angle |
| Clone: status line, Aligned | Context bar (Clone) |
| Shader-mask White/Black/Grey box | Left dock (replaces channels) |
| Layer add menu (Paint, Fill from image…, 12 shader masks) | Layers toolbar + Layer menu |
| Delete layer | Layers toolbar + Layer menu |
| Row eye, move up/down, name, badges, MASKED tag | Layer row (drag handle + menu for move; mask thumbnail replaces MASKED) |
| "Masks:" go-to row | "Shader masks:" links under the list |
| Opacity, Blend, Fill from brush values | Properties › Layer |
| Undo readout, Undo budget slider | Status bar + Edit ▸ Undo budget… |
| Layer mask Add black/white, Fill, Clear, Invert, Remove, Paint the mask | Properties › Mask |
| Smart mask kind, Amount, Contrast, Generate, Remove | Properties › Mask |
| Shader mask "Mask for", Fill white/black | Properties › Layer (shader layers) |
| Texture projection (all fields), Place tool | Properties › Projection + Place tool (P) |
| Material map slots, Surface (OpenPBR) | Properties › Material |
| Bake AO, AO strength, Bake mesh maps, Auto-unwrap | Properties › Material › Bake & UVs + Bake menu |
| Library: Browse materials…, Save material + name | Library menu, Library window, Properties › Material |
| Material/Brush browser windows | Unified Library window |
| Export maps…, Import maps… dialogs | File menu → redesigned dialogs |
| Status line | Status bar (left) + toasts |
| Model import progress window | Background-task card |
| Orange inline warnings | Context-bar chips (warn colours) |

---

## 9. Implementation phases

1. **Theme and tokens.** Add the token constants and fonts, and apply them to the existing UI with no layout change yet. *Accept:* the app looks re-skinned and works exactly as before.
2. **Shell layout.** Build the menu bar, context bar, tool rail, the three docks, the viewport and the status bar as empty regions. Move the existing panels into them temporarily. *Accept:* every feature is still reachable.
3. **Tools and context bar.** Wire up the tool rail and the per-tool context bar. Move brush settings out of the old panel and remove the old toolbar. *Accept:* every row of section 4.2 works; shortcuts unchanged.
4. **Left dock.** Tip, channels, the shader-mask box, and the fill-mode footer.
5. **Right dock.** Texture sets, the layer list (thumbnails, badges, drag reorder, duplicate, rename), and the four Properties tabs. Delete the old long left column. *Accept:* every row of section 8 is ticked off.
6. **Viewport overlays.** Breadcrumb, render switch, view toggles, the Display & lighting popover, the empty-state chip.
7. **Menus and command search.** All menus from section 7, then Ctrl K.
8. **Tasks, toasts and dialogs.** The task/progress API with Cancel, toasts, the redesigned Export/Import dialogs, and export presets.
9. **Library window.** Merge the two browsers into one tabbed window.

After each phase: build, run, open a glTF and an FBX, paint, mask, project, bake and export. Compare the result against the matching screen in `screens/`, and report anything you couldn't match and why.

---

## 10. Out of scope
Rendering changes, the path tracer's material support, project save/load, layer folders, the 2D UV view, manual UV editing, and normal re-orientation in projected fills. Leave those as they are.
