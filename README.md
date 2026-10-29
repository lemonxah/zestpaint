# ZestPaint

A 3D texture-painting app in the spirit of Substance Painter, built from scratch
in Rust on Vulkan (via [vulkano]) with an [egui] interface.

## Current features

- **Two viewport modes** — real-time raster, or a progressive **path-traced
  preview** (compute shader over a CPU-built BVH: metal/rough GGX with the
  painted channels and normal maps, environment and studio lights with
  soft shadows, multiple bounces). Painting happens in raster mode.
- **Studio: pose, light and render the avatar** (View ▸ Studio, F10) —
  rigged FBX/glTF models keep their skeleton and skin weights, and can be
  posed: built-in photography poses (contrapposto, hands on hips, arms
  crossed, wave, look over the shoulder, walking, sitting, …), any frame of
  an animation file (Mixamo FBX, Quaternius glTF/GLB, CMU BVH mocap), or
  poses you saved, with per-bone nod/turn/tilt adjustments and mirroring.
  Bones are matched by name and place in the hierarchy (Mixamo, Unity/CATS,
  Unreal, VRoid, Daz names all map), and poses retarget across rest poses (a
  T-pose clip on an A-pose avatar) and facing. Light with classic setups
  (three-point, Rembrandt, butterfly, split, rim, soft box) of soft round
  lights placed around the subject; stand it on a floor or a studio sweep;
  show the environment, a colour (chroma green/blue) or nothing behind it.
  The Camera tab sets the focal length and depth of field. **Render (F12)**
  traces a still at any size and sample count, shows it, and saves it as
  PNG; with a transparent background it has alpha, so only the avatar (and
  optionally its shadow on the floor, a shadow catcher) is in the image, to
  place over a background later. The pose and the studio are saved in the
  project.

- **PBR viewport** — Cook-Torrance GGX metallic-roughness shading with
  image-based lighting (irradiance + GGX-prefiltered specular cubemaps),
  ACES tone mapping, and a skybox background, extended with OpenPBR-style
  surface slots: clear coat, sheen, emission, and specular weight per
  material (a pragmatic subset of the OpenPBR 1.1 slot set, not a certified
  implementation).
- **Environments** — built-in procedural sky; load any equirectangular `.hdr`
  and it is convolved into IBL cubemaps on the GPU at load time.
- **Model loading** — `.gltf` / `.glb` (node hierarchies, per-primitive
  materials, base color textures) and `.fbx` via [ufbx] (any FBX version,
  units/axes normalized, polygons triangulated, base-color textures imported
  whether embedded in the file or referenced on disk). The camera auto-frames
  loaded models.
- **Texture sets per material** — Substance-style: the paint target is a
  *material*, not an object. The left panel shows a tree of objects with
  their materials (an FBX body mesh with Body/Face/Eyes materials becomes
  three independent sets, each with its own layers, maps, and bakes). Select
  a material to paint just its faces; eyes on both levels show/hide a whole
  object or a single material (hidden geometry is also skipped by painting
  and AO baking).
- **Material maps** — every material has six file-loadable slots: base
  color, roughness, metallic, normal, height, and AO. FBX and glTF imports
  fill them automatically where the source file provides them (including
  glTF's packed metallic-roughness, which is split); any slot can be
  replaced or removed from an image file in the Material section. Height
  maps auto-derive a normal map when no explicit normal map is present, and
  material AO multiplies with baked AO.
- **Projection painting** — screen-space brush stamps are projected onto the
  mesh and rasterized in UV space into the active layer, with depth-occlusion
  and backface rejection so you can't paint through the model. Size, hardness,
  flow, loadable grayscale **brush tips**, and **X-symmetry** mirroring.
  Paints base color, roughness, metallic, height, **emissive**, and
  **opacity** channels independently or together.
- **Transparency** — per-material opacity and a "Transparent (alpha blend)"
  flag (auto-detected from FBX transparency and glTF `alphaMode: BLEND`).
  Transparent materials render in a sorted back-to-front blended pass over
  the skybox with depth writes off; opaque materials treat low alpha as a
  cutout. Final alpha = material opacity × base-color texture alpha ×
  painted opacity.
- **Height & normal painting** — a signed height channel (raise or carve) per
  layer; a compute pass derives a tangent-space normal map from the composited
  height (Sobel), applied in the shader via a derivative-based tangent frame.
  "Bump strength" controls the gradient scale.
- **Layer stack** — multiple layers with visibility, opacity, and blend modes
  (Normal / Multiply / Add), flattened by a GPU compositor each time something
  changes. Painted layers composite over the model's own material.
- **AO baking** — GPU baker that renders scene depth from 64 fibonacci-sphere
  directions and accumulates cosine-weighted visibility per texel, one AO map
  per texture set (the whole scene occludes, so clothes shadow the body); the
  result attenuates the environment lighting (strength slider) and exports.
- **UV tools** — auto-unwrap via [xatlas] targets only the active texture
  set: that one object gets a fresh full-resolution atlas (fixing bad or
  overlapping UVs) while every other sub-model's UVs, paint, and bakes stay
  untouched. A UV checker debug view helps judge texel density and seams.
- **Undo history** — Ctrl+Z / Ctrl+Shift+Z step through one history for the
  whole document: strokes (pixel snapshots of just the channels touched,
  within a RAM budget) interleaved with every other edit, which is caught by
  diffing a cheap per-frame document snapshot — a slider drag or a typed name
  is one step.
- **Smart masks & fill layers** — bake mesh maps (world position, normal,
  curvature), then generate edge-wear or cavity-dirt masks per layer with
  amount/contrast controls; "Fill from brush values" floods a layer so a
  fill + mask behaves like a simple smart material.
- **Projected fills** — fill layers (library materials, or any image via
  Layers ▸ + Fill, or Projection ▸ Fill with image…) can ignore the UVs and wrap onto the model:
  triplanar (seamless on most shapes), spherical (one 2:1 image around a
  ball), cylindrical (labels on cans) or planar, aimed along X/Y/Z, with the
  usual tiling/offset/rotation on top. The Place tool (P) moves, turns and
  sizes the projection right on the model. Mesh maps bake automatically the
  first time a projection is picked.
- **UDIM** — multi-tile UVs split into one texture set per tile (1001, 1002,
  …) at import, each painting at full resolution.
- **Material library** — `~/.zestpaint/library/materials/<name>/` folders of
  maps (ambientCG / Poly Haven naming) can be applied to any material or
  saved back from the current material.
- **Export** — flattens every texture set and writes its maps under one
  name you type (default: the project name): `<Name>_color.png` sRGB +
  coverage alpha, `_roughness` / `_metallic` / `_opacity` grayscale +
  coverage alpha, `_height` 0.5-neutral grayscale, `_normals`
  tangent-space, `_emission`, `_ao` when baked, and shader masks as
  `_matcap_mask`, `_emission_mask`… With several texture sets the material
  is added (`<Name>_<Material>_color.png`), and the object too when two
  sets share a material name.
  The export dialog can also write **each layer on its own**
  (`<Name>_L03_<Layer>_<map>.png`, coverage in alpha, so Import
  maps brings it back as a layer) and **each layer mask** (`…_mask.png`,
  greyscale), for all, visible or selected layers — with presets for
  Layers only, Masks only and Everything. Right-click a layer to export just
  that layer or its mask; Layer ▸ Load mask from image… reads a mask back.
- **Project files** (`.zestpaint`) — File ▸ Save project (Ctrl S), Save as
  (Ctrl Shift S), Open project (Ctrl Shift O), Open recent. A project holds
  everything needed to carry on painting: meshes (with any re-unwrapped UVs),
  material maps and sliders, every layer with all its settings, masks, shader
  masks, baked AO/position/normal/curvature, camera and lighting, the
  skeleton and skin weights, the pose and the studio. Layer data
  is stored losslessly as raw half floats in a zip; empty channels take no
  space. `zestpaint file.zestpaint` opens one from the command line.

## Interface

The window follows one rule: **the left side is how you paint, the right side
is what you paint into**, and the viewport only carries overlays about the
view itself (see `docs/zestpaint-ui-handover/` for the design).

- **Menu bar** — File, Edit, View, Layer, Bake, Library, Window, Help, plus
  **Ctrl K** command search over every command and the main panel settings.
- **Context bar** — the active tool's options (Into Material/Mask, stroke,
  size, hardness, flow, angle, symmetry, fill mode; clone source, stamp
  image, Select's hide/isolate, Place's projection).
- **Tool rail** — Paint B, Erase E, Smudge G, Clone C, Stamp T · Pick I,
  Place P · Select S, Orbit R, and the current paint colour.
- **Left dock** — brush tip and the built-in brushes (hard / medium / soft
  round, airbrush, gaussian, chalk, spatter, bristle — also in the ⋯ menu),
  paint channels (or the shader-mask box), bump strength, fill-mode info.
- **Right dock** — texture sets, the layer stack (thumbnails, badges, drag
  to reorder, double-click to rename, duplicate), and Layer / Mask /
  Projection / Material property tabs.
- **Viewport overlays** — breadcrumb, Raster/Path traced, view toggles, the
  Display & lighting popover, axis gizmo, background-task cards and toasts.
- **Library window** — one separate OS window with Materials and
  Brushes & stamps tabs.

| Input | Action |
| --- | --- |
| Brush tools: LMB drag | paint / erase / smudge / clone |
| Alt + LMB drag | orbit (brush tools) |
| Orbit tool: LMB drag | orbit |
| Select / Orbit tool: click, Shift+click | select a piece, add/remove it |
| A, Alt A | select all visible pieces, deselect |
| H, Shift H, Alt H | hide selection, hide others, unhide all |
| / (main or numpad) | local view: isolate the selection and frame it; again to leave |
| F (or numpad .), Home | frame selection, frame all |

Each texture set has its own paint resolution (Texture sets panel ▸
Resolution: 1K–8K, with a GPU-memory estimate; sizes that would overload the
GPU are refused). Changing it resamples the layers and is undoable; projects
save it per set.
The status bar's VRAM readout is what the driver says ZestPaint holds; hover
it for the breakdown (layer textures, set composites, imported maps, bakes…,
and how much only undo still keeps). Memory is only spent where there is
content: a layer gets a texture per channel when that channel is first
painted, a set gets a full-size composite only for channels some layer uses,
and imported maps are stored BC7/BC4/BC5-compressed with mipmaps. Large
textures have their own GPU allocations, so freed memory goes straight back
to the driver for other apps.
Paint big, ship smaller: the Export dialog asks for the export size (as
painted, or 4096/2048/1024/512) and downscales the PNGs (Lanczos,
alpha-aware) without touching the document.
| RMB / MMB drag | pan |
| Scroll | zoom |
| Ctrl+Scroll / Shift+Scroll | brush size / tip rotation (Place: size / spin) |
| [ / ] (Shift: faster) | brush size down / up |
| Place tool: drag / Shift+drag | move / rotate the projection |
| Ctrl Z, Ctrl Shift Z (Ctrl Y) | undo / redo (any document edit) |
| Ctrl O, Ctrl Shift I, Ctrl Shift E | open model, import maps, export maps |
| Ctrl S, Ctrl Shift S, Ctrl Shift O | save project, save as, open project |
| Ctrl Shift N, Ctrl J, F2, Del | new paint layer, duplicate, rename, delete |
| Ctrl ] / Ctrl [ | move layer up / down |
| M, Shift B | material / brush library |
| F5–F8 | toggle Brush, Texture sets, Layers, Properties |
| F1, Ctrl , | keyboard shortcuts, preferences |
| F10, F12 | Studio window, render an image |

## Startup and settings

Launched without a file, ZestPaint shows a welcome screen (Help ▸ Welcome
screen brings it back): open a model or project, or pick a recent project.
It can instead reopen the last project or go straight to the default sphere;
choose on the welcome screen or in Preferences. `zestpaint file.zestpaint`
always opens that file.

Settings live in `~/.config/zestpaint/` (or `$XDG_CONFIG_HOME/zestpaint/`):

- `settings.json`: `startup` (`"welcome"`, `"last_project"` or
  `"default_scene"`) and `undo_budget_gb`; written with the defaults on first
  launch, safe to edit by hand.
- `assistant.json`, `comfy_workflow.json`, `recent_projects.json`,
  `export_presets.json`, `hidden_pieces.json`, `mcp-bridge.json`: kept by their
  features. Older versions kept these in `~/.zestpaint/`; they move over on the
  first launch. The material and brush libraries stay in `~/.zestpaint/library/`,
  and saved poses in `~/.zestpaint/library/poses/` (one JSON file each,
  rig-neutral: they fit any humanoid).

Poses to import (Studio ▸ Pose ▸ Import animation…): Mixamo (free with an
Adobe account; download as FBX "Without Skin"; usable in your own work but
not redistributable), Quaternius' Universal Animation Library 1 and 2 (CC0,
glTF/FBX), CMU's motion capture library (BVH, free for any use). Scrub to
the frame you want and save it as a pose.

## Building

```sh
cargo run
```

Requires a Vulkan-capable GPU and driver. First build compiles shaders through
`vulkano-shaders`.

## Claude Code (MCP)

Claude Code can drive the editor with the same tools as the built-in assistant
(no Anthropic API key needed — it uses your Claude Code login).

- The running app listens on a loopback port and writes it, with a random
  token, to `~/.config/zestpaint/mcp-bridge.json` (owner-only). Toggle it under
  Assistant ▸ settings ▸ *Let Claude Code drive the editor*.
- `zestpaint mcp` is the stdio MCP server Claude Code launches; it forwards
  each tool call to the running app, and its `open_zestpaint` tool starts the
  app when it isn't running.
- Inside this repo `.mcp.json` registers it (`cargo run --release -- mcp`).
  To use it from anywhere, copy the command from the assistant settings, e.g.
  `claude mcp add --scope user zestpaint -- /path/to/zestpaint mcp`.
- Besides the assistant's tools it gets `import_model` (open an .fbx/.gltf
  as a new document), `view_pieces` (select / isolate / hide / frame
  pieces, like the keys above), `set_texture_resolution` and
  `set_render_mode` (raster / path traced, to judge final looks),
  `import_image` (a local file as stamp, map layer, fill layer or material
  map), `set_view` (camera presets/framing) and `place_stamp` (stamp at a
  viewport point) — the last two answer with a screenshot, so Claude can
  place overlays and check them itself — and `set_matcap`.

Materials have two Poiyomi-style matcap slots (Properties ▸ Material ▸
Matcaps): a built-in (Shine, Rim light, Chrome, Pearl, Satin, Clay) or any
matcap image, Add / Multiply / Replace / Screen, strength, limited by the
Matcap / Matcap 2 shader-mask layers when present. Raster view only; saved
with the project and undoable.
- `.claude/skills/zestpaint/` is a Claude Code skill with the editor's model
  and recipes (tattoos, splatter, CC0 materials). Copy it to
  `~/.claude/skills/` to have it outside the repo too.

## Architecture

| Module | Role |
| --- | --- |
| `app.rs` | winit application shell, egui panels, viewport, input routing |
| `renderer.rs` | scene render pass, PBR + skybox pipelines |
| `environment.rs` | HDRI/procedural env → cubemap, irradiance, prefilter (compute) |
| `layers.rs` | layer stack document model + GPU compositor + height→normal derivation |
| `paint.rs` | UV-space brush projection pipeline (color/roughness/metallic/height) |
| `ao.rs` | multi-direction depth-visibility AO baker |
| `bakers.rs` | position/normal/curvature mesh maps + smart-mask generators |
| `pathtracer.rs` | progressive compute path tracer (CPU BVH, GGX, studio lights, backdrop, alpha) |
| `studio.rs` | studio settings: lights and presets, backdrop, background, lens |
| `skeleton.rs` | skeletons, skins (linear blend skinning), humanoid bone mapping, retargeting |
| `poses.rs` | pose sources: FBX/glTF/BVH clips, built-in poses, the pose library |
| `library.rs` | on-disk material library (ambientCG-style naming) |
| `uv_tools.rs` | xatlas auto-unwrap integration |
| `export.rs` | GPU readback and PNG encoding (flattened maps, single layers, masks) |
| `project.rs` | `.zestpaint` project files: zip of JSON manifest + raw textures |
| `gltf_loader.rs` | glTF import to CPU mesh/material data |
| `fbx_loader.rs` | FBX import via ufbx (triangulation, unit/axis normalization) |
| `camera.rs`, `mesh.rs`, `scene.rs` | orbit camera, vertex/mesh types, CPU scene data |

The 3D scene renders into an offscreen image that egui displays as a texture in
the central panel — the same viewport-in-dock pattern Substance uses. Layer
textures store premultiplied alpha in linear space (RGBA16F color, RG16F
value+coverage for scalar channels); the compositor flattens them into
composite textures the PBR shader samples over each model's own material.

## Known limitations / next steps

- The OpenPBR-style shading is a subset (coat/sheen/emission/specular); full
  OpenPBR 1.1 (transmission, subsurface, thin-film, anisotropy) is not
  implemented.
- The path tracer has metal/rough GGX, normal maps and emission, but not
  coat, sheen, transmission or subsurface; studio lights are spheres, not
  rectangular softboxes, and aren't seen directly by the camera.
- Posing is CPU linear blend skinning (no dual quaternions or corrective
  shapes); bakes made while posed use the posed mesh. Bones are adjusted
  with sliders, not with a gizmo in the viewport.
- Undo covers the document — strokes, every layer/mask/projection/tiling
  edit, fills, bakes, visibility, material surface and map slots, UV unwrap —
  from any panel or window. Brush/tool settings, camera, display & lighting
  and layout are not undone, and opening a model starts a fresh history.
- Bakes and exports run on the GPU queue synchronously and block the UI
  while they run (no progress card or Cancel for them yet).
- Re-unwrapping a set resets that set's layers (paint can't survive a UV
  change); other sets are unaffected.
- Manual UV island editing and thickness baking are future work.

## Open-source material sources

Drop CC0 material folders into `~/.zestpaint/library/materials/` — the
scanner understands ambientCG / Poly Haven / cgbookcase file naming.
[ambientCG](https://ambientcg.com) (2000+ CC0 PBR materials) and
[Poly Haven](https://polyhaven.com) (CC0 textures + HDRIs, which also work
with the HDRI loader) are the recommended sources. For a future full OpenPBR
implementation, the Apache-2.0 [OpenPBR spec + MaterialX reference](https://github.com/AcademySoftwareFoundation/OpenPBR)
is the path to follow.

[vulkano]: https://github.com/vulkano-rs/vulkano
[egui]: https://github.com/emilk/egui
[xatlas]: https://github.com/jpcy/xatlas
[ufbx]: https://github.com/ufbx/ufbx
