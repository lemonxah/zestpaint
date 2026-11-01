use std::sync::Arc;

use egui::load::SizedTexture;
use egui::{ImageSource, Sense};
use egui_winit_vulkano::{Gui, GuiConfig};
use glam::{Mat4, Vec3};
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{ImageCreateInfo, ImageType, ImageUsage};
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano_util::context::{VulkanoConfig, VulkanoContext};
use vulkano_util::renderer::DEFAULT_IMAGE_FORMAT;
use vulkano_util::window::{VulkanoWindows, WindowDescriptor};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::window::WindowId;

use glam::Vec2;

use crate::camera::OrbitCamera;
use crate::environment::{Environment, GpuContext};
use crate::fbx_loader;
use crate::gltf_loader;
mod assistant_dock;
mod blendshapes;
mod brush_dock;
mod brush_presets;
mod commands;
mod dialogs;
mod history;
mod library_window;
mod overlays;
mod piece_memory;
mod project_io;
mod right_dock;
mod shell;
mod vram_readout;
mod studio_panel;
mod welcome;
mod widgets;
mod windows;

use crate::layers::{PaintChannel, Projection, ShaderMask, TextureSet};
use crate::mesh::uv_sphere;
use crate::paint::{BrushSettings, PaintSystem, PaintView};
use crate::renderer::{MaterialOverride, RenderParams, SceneRenderer};
use crate::scene::{set_keys, ImageData, MapSlot, MaterialData, ModelData};

/// Texture resolution painting operates at (per channel).
const PAINT_RESOLUTION: u32 = 2048;
/// Paint resolutions a texture set can be switched to.
const RESOLUTIONS: [u32; 4] = [1024, 2048, 4096, 8192];

/// Texels of slack added around a computed paint region, covering the brush's
/// soft edge and the dilation that runs afterwards.
const SCISSOR_MARGIN: u32 = 6;

/// Where in the texture a stroke segment can land.
enum PaintRegion {
    /// Nothing under the brush — skip the draw entirely.
    Nothing,
    /// Bounds could not be established; rasterise the whole layout.
    Everything,
    /// Texel rect [x, y, w, h].
    Rect([u32; 4]),
}

/// A triangle reduced to the two boxes the scissor search needs.
#[derive(Clone)]
struct ProjectedTri {
    screen_lo: Vec2,
    screen_hi: Vec2,
    uv_lo: Vec2,
    uv_hi: Vec2,
}

/// A model load running off the UI thread, so a large FBX does not freeze the
/// window. Only the CPU-side parse happens on the worker; the GPU upload is
/// applied on the main thread when it finishes.
struct ModelImport {
    path: std::path::PathBuf,
    progress: Arc<std::sync::Mutex<crate::scene::ImportProgress>>,
    handle: Option<std::thread::JoinHandle<anyhow::Result<Vec<ModelData>>>>,
}

/// Connectivity of the active set's geometry. Independent of the camera, so it
/// is cached separately from the projection and only rebuilt when the set or
/// the loaded models change.
struct MeshGroups {
    set: usize,
    /// Shell id per global primitive index — welded by position, so a UV seam
    /// does not split a piece in two.
    shell: Vec<u32>,
    /// UV chart id per global primitive index.
    island: Vec<u32>,
    shell_count: usize,
    island_count: usize,
    /// Triangles per shell, largest first.
    shell_sizes: Vec<usize>,
}

/// Screen projection of the active set, reused across the segments of one
/// stroke — the camera cannot move mid-drag, so this is computed once and then
/// every segment is a cheap box test.
struct PaintProjection {
    view_proj: Mat4,
    set: usize,
    viewport: Vec2,
    /// X-symmetry draws a second, mirrored pass that lands elsewhere on screen;
    /// its triangles must be in here too or the mirrored half gets clipped.
    symmetry_x: bool,
    tris: Vec<ProjectedTri>,
    /// Some geometry sits behind the camera, so screen bounds are unreliable.
    degenerate: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Tool {
    Orbit,
    Paint,
    /// The paint brush with destination-out blending: same size, tip, stroke
    /// and fill modes, channel toggles and mask target as Paint.
    Erase,
    /// Drag existing paint along the stroke, with the brush's size, falloff,
    /// tip and channel toggles; flow is the strength.
    Smudge,
    /// Eyedropper: click the model to take its base colour into the brush,
    /// then hand back to the brush tool it was picked up from.
    Picker,
    /// One image per click: the whole tip image at its own aspect ratio,
    /// previewed under the cursor. Shift+scroll rotates.
    Stamp,
    /// Ctrl+click sets a source; painting then copies what is under the
    /// source onto what is under the brush.
    Clone,
    /// Click geometry to select it, then hide or isolate with Blender's keys.
    Select,
    /// Move the active layer's texture projection over the model: drag to
    /// slide it, Shift+drag to turn it, Ctrl+scroll to size it.
    Place,
}

impl Tool {
    /// Tools that drive the brush: the left button strokes, the brush cursor
    /// and Ctrl+scroll sizing apply, and the brush panel is shown.
    fn uses_brush(self) -> bool {
        matches!(self, Tool::Paint | Tool::Erase | Tool::Smudge | Tool::Stamp | Tool::Clone)
    }

    fn brush_op(self) -> crate::paint::BrushOp {
        match self {
            Tool::Erase => crate::paint::BrushOp::Erase,
            Tool::Smudge => crate::paint::BrushOp::Smudge,
            Tool::Clone => crate::paint::BrushOp::Clone,
            _ => crate::paint::BrushOp::Paint,
        }
    }
}

/// A browser living in its own OS window, with its own egui context and
/// swapchain. Real windows rather than floating panels, so they can be moved to
/// another monitor and sized independently of the main window.
struct BrowserWindow {
    id: WindowId,
    gui: Gui,
}

pub struct App {
    context: VulkanoContext,
    windows: VulkanoWindows,
    gui: Option<Gui>,
    primary: Option<WindowId>,
    browsers: Vec<BrowserWindow>,
    state: AppState,
}

/// Everything the egui layout closure touches. Kept separate from `gui` so the
/// closure's disjoint field capture doesn't conflict with the `&mut Gui` borrow.
struct AppState {
    gpu: GpuContext,
    renderer: SceneRenderer,
    camera: OrbitCamera,
    viewport: ViewportState,
    /// CPU copy of the scene, kept for re-unwrapping and re-upload.
    models: Vec<ModelData>,
    /// One paintable texture set per named sub-model (body, shirt, …).
    texture_sets: Vec<TextureSet>,
    active_set: usize,
    zestengine: Option<crate::zestengine::ZestEngineManifest>,
    paint: PaintSystem,
    brush: BrushSettings,
    tool: Tool,
    /// The brush tool the picker returns to after a pick.
    picker_return: Tool,
    /// Clone source: a point on the model (set with Ctrl+click), so it stays
    /// put on the surface when the camera moves.
    clone_source: Option<CloneAnchor>,
    /// Aligned mode: source minus brush, as a world-space offset, fixed by
    /// the first stroke after the source is set.
    clone_offset: Option<Vec3>,
    /// The source's screen offset for the stroke in progress. The camera
    /// cannot move mid-drag, so it is worked out once per stroke.
    clone_stroke_offset: Option<Vec2>,
    /// Downscaled tip image for the stamp cursor preview, and its texture.
    tip_preview_image: Option<egui::ColorImage>,
    tip_preview: Option<egui::TextureHandle>,
    /// Viewport-pixel position of the previous stamp within the active stroke.
    stroke_last: Option<Vec2>,
    /// Line mode: where the drag began, in viewport pixels and on screen.
    line_anchor: Option<(Vec2, egui::Pos2)>,
    /// A model import running on a worker thread.
    import: Option<ModelImport>,
    /// The project file this document was opened from or last saved to.
    project_path: Option<std::path::PathBuf>,
    /// A project save or open running on a worker thread.
    project_job: Option<project_io::ProjectJob>,
    /// Piece locked in at the start of a fill stroke. Dragging onto another
    /// piece mid-stroke must not start filling that one too.
    fill_lock: Option<u32>,
    /// Texture sets picked in the viewport (Select/Orbit tool click, Shift
    /// to add) or the outliner. Empty means "the active set" for the
    /// visibility keys.
    selection: Vec<usize>,
    /// Visibility to restore when isolate is switched off.
    isolate_backup: Option<Vec<bool>>,
    /// The model file the document came from, when it was opened directly
    /// (not from a project): its hidden pieces are remembered for next time.
    model_source: Option<std::path::PathBuf>,
    /// Hidden sets as last written for `model_source`.
    remembered_hidden: Vec<usize>,
    /// A model file being read for its blendshapes.
    shape_import: Option<blendshapes::ShapeImport>,
    /// The camera before entering local view, restored on leaving it.
    isolate_camera: Option<OrbitCamera>,
    /// Sets under the pointer in the texture-set list, outlined in the
    /// viewport so you can see which piece a row is.
    outliner_hover: Vec<usize>,
    /// Numpad . was pressed: it is caught before egui (with NumLock off it
    /// arrives as Delete, which deletes a layer).
    frame_selection_pending: bool,
    paint_projection: Option<PaintProjection>,
    mesh_groups: Option<MeshGroups>,
    pathtracer: crate::pathtracer::PathTracer,
    /// Path-traced viewport (vs raster).
    pt_mode: bool,
    /// Geometry/texture bindings need a rebuild before the next PT frame.
    pt_scene_stale: bool,
    pt_last_view: Mat4,
    /// The studio block the viewport's samples were traced with: a change
    /// starts them over.
    pt_last_block: [[f32; 4]; crate::studio::BLOCK_VEC4S],
    /// Where the subject stood when the traced scene was built.
    pt_subject: crate::studio::Subject,
    /// Lights, backdrop, background, lens and render size.
    studio: crate::studio::Studio,
    pose: studio_panel::PoseState,
    render_job: Option<studio_panel::RenderJob>,
    render_result: Option<studio_panel::RenderResult>,
    /// Last VRAM readout (status bar), retaken about once a second.
    vram: Option<vram_readout::VramSnapshot>,
    /// `settings.json` (startup, undo budget).
    settings: crate::config::Settings,
    ui: UiControls,
    /// The AI chat and its running turn.
    assistant: assistant_dock::AssistantPanel,
}

struct ViewportState {
    memory_allocator: Arc<StandardMemoryAllocator>,
    scene_image: Arc<ImageView>,
    texture_id: Option<egui::TextureId>,
    size: [u32; 2],
}

struct UiControls {
    override_material: bool,
    base_color: [f32; 3],
    metallic: f32,
    roughness: f32,
    exposure: f32,
    env_intensity: f32,
    key_light: f32,
    /// Key light aim, in degrees. Azimuth spins it around the model, elevation
    /// lifts it — grazing angles are what make normal detail readable.
    key_azimuth: f32,
    key_elevation: f32,
    /// Keep the key light pinned to the camera as you orbit.
    key_follow_camera: bool,
    ao_strength: f32,
    uv_checker: bool,
    mask_kind: crate::layers::MaskKind,
    mask_amount: f32,
    mask_contrast: f32,
    library: Vec<crate::library::LibraryMaterial>,
    library_scanned: bool,
    library_save_name: String,
    /// Library browser window state.
    library_open: bool,
    library_filter: String,
    library_selected: Option<usize>,
    library_thumbs: std::collections::HashMap<String, egui::TextureHandle>,
    library_loader: Option<crate::library::ThumbnailLoader>,
    /// Brush browser state, mirroring the material one.
    brushes: Vec<crate::library::LibraryMaterial>,
    brushes_scanned: bool,
    /// Which tab the library window shows, and the source picked per tab.
    library_tab: library_window::LibraryTab,
    library_source: [library_window::Source; 2],
    brush_filter: String,
    brush_selected: Option<usize>,
    brush_thumbs: std::collections::HashMap<String, egui::TextureHandle>,
    brush_loader: Option<crate::library::ThumbnailLoader>,
    brush_import_status: String,
    /// Keep fill-layer tiling square unless the user unlinks the axes.
    tiling_linked: bool,
    /// Tint the model with the active shader-mask layer's mask.
    mask_overlay: bool,
    /// The app icon, for the menu bar; uploaded on first use.
    logo: Option<egui::TextureHandle>,
    /// Full lockup for the About dialog, rasterised from the logo SVG once.
    about_logo: Option<egui::TextureHandle>,
    /// A piece picked in the viewport, for the texture-set list to scroll to.
    scroll_to_set: Option<usize>,
    /// Clone keeps one source offset across strokes (Photoshop's "Aligned").
    clone_aligned: bool,
    /// Export: also write one greyscale PNG per shader mask in use.
    export_shader_masks: bool,
    /// Export dialog: the flattened maps, each layer on its own, each
    /// layer's mask, and which layers those two cover.
    export_flattened: bool,
    export_layers: bool,
    export_layer_masks: bool,
    export_layer_scope: crate::export::LayerScope,
    /// Export downscale target (None = paint resolution).
    export_size: Option<u32>,
    /// Export dialog: which maps, and which texture sets (parallel to
    /// `texture_sets`, resynced whenever the dialog draws).
    export_open: bool,
    export_maps: Vec<crate::layers::MapKind>,
    export_sets: Vec<bool>,
    /// Import dialog: picked files with their (editable) detected map, and the
    /// texture set they will be layered onto.
    import_files: Vec<(std::path::PathBuf, Option<crate::layers::MapKind>)>,
    import_set: usize,
    import_open: bool,
    /// Export dialog folder (typed or browsed) and preset.
    export_dir: String,
    export_preset: dialogs::ExportPreset,
    /// Export dialog Name field; empty means the project / model name.
    export_name: String,
    /// Project files opened or saved, most recent first (File ▸ Open recent).
    recent_projects: Vec<std::path::PathBuf>,
    status: String,
    /// Redesigned shell.
    props_tab: PropsTab,
    panels: Panels,
    texture_sets_open: bool,
    /// The texture-set panel shows blendshapes instead of the set list.
    show_blendshapes: bool,
    /// Filters the blendshape list by name.
    blendshape_filter: String,
    /// Layer being renamed in the list, and the text so far.
    renaming: Option<(usize, String)>,
    /// A double-click asked to rename this layer (applied after the list).
    rename_request: Option<usize>,
    /// Renaming just started: give the field focus this frame.
    rename_fresh: bool,
    palette: windows::Palette,
    confirm_unwrap: bool,
    prefs_open: bool,
    shortcuts_open: bool,
    gestures_open: bool,
    about_open: bool,
    welcome_open: bool,
    studio_open: bool,
    studio_tab: studio_panel::StudioTab,
    render_result_open: bool,
    /// Ask the Save-to-library name field for focus next frame.
    focus_save_name: bool,
    display_open: bool,
    /// When the popover opened, so the opening click doesn't close it.
    display_opened_at: f64,
    /// The startup sphere is still the scene (shows the empty-state chip).
    default_scene: bool,
    /// File the scene came from, for the menu bar.
    model_name: String,
    env_name: String,
    env_path: Option<std::path::PathBuf>,
    hdri_history: Vec<std::path::PathBuf>,
    toasts: Vec<overlays::Toast>,
    /// Undo / redo for the whole document.
    history: history::History,
    /// Built-in brush last picked, and the strip's thumbnails.
    preset: Option<usize>,
    preset_thumbs: Vec<egui::TextureHandle>,
    /// The round brush's preview, for the hardness it was drawn at.
    round_preview: Option<(f32, egui::TextureHandle)>,
}

/// The right dock's Properties tabs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PropsTab {
    Layer,
    Mask,
    Projection,
    Material,
}

impl PropsTab {
    fn label(self) -> &'static str {
        match self {
            PropsTab::Layer => "Layer",
            PropsTab::Mask => "Mask",
            PropsTab::Projection => "Projection",
            PropsTab::Material => "Material",
        }
    }
}

/// Which panels are showing (Window menu, F5–F8).
#[derive(Clone, Copy)]
struct Panels {
    brush: bool,
    texture_sets: bool,
    layers: bool,
    properties: bool,
    assistant: bool,
}

impl Default for Panels {
    fn default() -> Self {
        Self { brush: true, texture_sets: true, layers: true, properties: true, assistant: false }
    }
}

impl App {
    pub fn new() -> Self {
        let context = VulkanoContext::new(VulkanoConfig {
            device_features: crate::environment::required_device_features(),
            ..VulkanoConfig::default()
        });

        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            context.device().clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        ));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            context.device().clone(),
            Default::default(),
        ));
        let gpu = GpuContext {
            queue: context.graphics_queue().clone(),
            memory_allocator: context.memory_allocator().clone(),
            command_buffer_allocator,
            descriptor_set_allocator,
        };

        let environment = Environment::procedural(&gpu);
        let mut renderer = SceneRenderer::new(
            gpu.queue.clone(),
            gpu.memory_allocator.clone(),
            gpu.command_buffer_allocator.clone(),
            gpu.descriptor_set_allocator.clone(),
            DEFAULT_IMAGE_FORMAT,
            environment,
        );

        let settings = crate::config::Settings::load();
        crate::layers::set_default_undo_budget(settings.undo_budget_bytes());

        // Default scene: a sphere to look at before any model is opened.
        let models = vec![ModelData {
            mesh: uv_sphere(1.0, 64, 128),
            material: MaterialData::default(),
            transform: Mat4::IDENTITY,
            name: "Sphere".into(),
            material_name: "Default".into(),
            morphs: Default::default(),
            skin: None,
        }];
        renderer.set_models(&models);

        let viewport_size = [1280, 720];
        let viewport = ViewportState {
            scene_image: create_scene_image(context.memory_allocator(), viewport_size),
            memory_allocator: context.memory_allocator().clone(),
            texture_id: None,
            size: viewport_size,
        };

        Self {
            context,
            windows: VulkanoWindows::default(),
            gui: None,
            primary: None,
            browsers: Vec::new(),
            state: AppState {
                texture_sets: vec![TextureSet::new(
                    "Sphere".into(),
                    "Default".into(),
                    gpu.clone(),
                    PAINT_RESOLUTION,
                )],
                active_set: 0,
                zestengine: None,
                models,
                paint: PaintSystem::new(gpu.clone()),
                brush: BrushSettings::default(),
                // Start by picking pieces; painting is one key (B) away.
                tool: Tool::Select,
                picker_return: Tool::Paint,
                clone_source: None,
                clone_offset: None,
                clone_stroke_offset: None,
                tip_preview_image: None,
                tip_preview: None,
                stroke_last: None,
                line_anchor: None,
                import: None,
                project_path: None,
                project_job: None,
                fill_lock: None,
                selection: Vec::new(),
                isolate_backup: None,
                model_source: None,
                remembered_hidden: Vec::new(),
                shape_import: None,
                isolate_camera: None,
                outliner_hover: Vec::new(),
                frame_selection_pending: false,
                paint_projection: None,
                mesh_groups: None,
                pathtracer: crate::pathtracer::PathTracer::new(gpu.clone()),
                pt_mode: false,
                pt_scene_stale: true,
                pt_last_view: Mat4::IDENTITY,
                pt_last_block: [[0.0; 4]; crate::studio::BLOCK_VEC4S],
                pt_subject: crate::studio::Subject { center: Vec3::ZERO, floor: -1.0, facing: glam::Quat::IDENTITY, seat: Vec3::ZERO },
                studio: crate::studio::Studio::default(),
                pose: Default::default(),
                render_job: None,
                render_result: None,
                vram: None,
                settings,
                gpu,
                renderer,
                camera: OrbitCamera::default(),
                viewport,
                ui: UiControls {
                    override_material: false,
                    base_color: [0.8, 0.8, 0.8],
                    metallic: 0.0,
                    roughness: 0.5,
                    exposure: 1.0,
                    env_intensity: 1.0,
                    key_light: 0.0,
                    key_azimuth: 35.0,
                    key_elevation: 40.0,
                    key_follow_camera: false,
                    ao_strength: 1.0,
                    uv_checker: false,
                    mask_kind: crate::layers::MaskKind::EdgeWear,
                    mask_amount: 0.5,
                    mask_contrast: 1.0,
                    library: Vec::new(),
                    library_scanned: false,
                    library_save_name: String::new(),
                    // ZESTSURFACE_OPEN_BROWSER=material|brush|both opens a browser
                    // window at startup, so the secondary-window path can be
                    // exercised without clicking.
                    library_open: matches!(
                        std::env::var("ZESTSURFACE_OPEN_BROWSER").as_deref(),
                        Ok("material") | Ok("brush") | Ok("both")
                    ),
                    library_filter: String::new(),
                    library_selected: None,
                    library_thumbs: std::collections::HashMap::new(),
                    library_loader: None,
                    brushes: Vec::new(),
                    brushes_scanned: false,
                    library_tab: if std::env::var("ZESTSURFACE_OPEN_BROWSER").as_deref() == Ok("brush") {
                        library_window::LibraryTab::Brushes
                    } else {
                        library_window::LibraryTab::Materials
                    },
                    library_source: [library_window::Source::All; 2],
                    brush_filter: String::new(),
                    brush_selected: None,
                    brush_thumbs: std::collections::HashMap::new(),
                    brush_loader: None,
                    brush_import_status: String::new(),
                    tiling_linked: true,
                    mask_overlay: true,
                    logo: None,
                    about_logo: None,
                    scroll_to_set: None,
                    clone_aligned: true,
                    export_shader_masks: true,
                    export_flattened: true,
                    export_layers: false,
                    export_layer_masks: false,
                    export_layer_scope: crate::export::LayerScope::All,
                    export_size: None,
                    export_open: false,
                    export_maps: crate::layers::MapKind::ALL.to_vec(),
                    export_sets: Vec::new(),
                    import_files: Vec::new(),
                    import_set: 0,
                    import_open: false,
                    export_dir: String::new(),
                    export_preset: dialogs::ExportPreset::AllPbr,
                    export_name: String::new(),
                    recent_projects: project_io::load_recent(),
                    status: "Default sphere — open a model to start".into(),
                    props_tab: PropsTab::Layer,
                    panels: Panels::default(),
                    texture_sets_open: true,
                    show_blendshapes: false,
                    blendshape_filter: String::new(),
                    renaming: None,
                    rename_request: None,
                    rename_fresh: false,
                    palette: windows::Palette::default(),
                    confirm_unwrap: false,
                    prefs_open: false,
                    shortcuts_open: false,
                    gestures_open: false,
                    about_open: false,
                    welcome_open: false,
                    studio_open: false,
                    studio_tab: Default::default(),
                    render_result_open: false,
                    focus_save_name: false,
                    display_open: false,
                    display_opened_at: 0.0,
                    default_scene: true,
                    model_name: "sphere".into(),
                    env_name: "Procedural sky".into(),
                    env_path: None,
                    hdri_history: Vec::new(),
                    toasts: Vec::new(),
                    history: history::History::default(),
                    preset: None,
                    preset_thumbs: Vec::new(),
                    round_preview: None,
                },
                assistant: assistant_dock::AssistantPanel::new(),
            },
        }
    }
}


impl App {
    /// What launching shows: `project` (from the command line) when given,
    /// else what `settings.json`'s `startup` says. A project is read on a
    /// worker like File ▸ Open project and adopted once it is ready.
    pub fn start(&mut self, project: Option<std::path::PathBuf>, zestengine_manifest: Option<std::path::PathBuf>) {
        let state = &mut self.state;
        if let Some(manifest_path) = zestengine_manifest {
            match crate::zestengine::read_manifest(&manifest_path) {
                Ok(manifest) => state.zestengine = Some(manifest),
                Err(e) => eprintln!("reading zestengine manifest {}: {e:#}", manifest_path.display()),
            }
        }
        if let Some(path) = project {
            let is_model = path
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| matches!(e.to_ascii_lowercase().as_str(), "gltf" | "glb" | "fbx"));
            if is_model {
                state.start_model_import(path);
            } else {
                state.open_project(path);
            }
            return;
        }
        match state.settings.startup {
            crate::config::Startup::DefaultScene => {}
            crate::config::Startup::LastProject => {
                match state.ui.recent_projects.iter().find(|p| p.exists()).cloned() {
                    Some(last) => state.open_project(last),
                    None => state.ui.welcome_open = true,
                }
            }
            crate::config::Startup::Welcome => state.ui.welcome_open = true,
        }
    }
}

/// Upload any thumbnails that finished decoding since the last frame.
fn zestengine_kind(kind: crate::layers::MapKind) -> &'static str {
    use crate::layers::MapKind;
    match kind {
        MapKind::BaseColor => "BaseColor",
        MapKind::Roughness => "Roughness",
        MapKind::Metallic => "Metallic",
        MapKind::Height => "Height",
        MapKind::Normal => "Normal",
        MapKind::Emissive => "Emissive",
        MapKind::Opacity => "Opacity",
        MapKind::Ao => "Ao",
    }
}

fn kind_from_str(s: &str) -> Option<crate::layers::MapKind> {
    use crate::layers::MapKind;
    Some(match s {
        "BaseColor" => MapKind::BaseColor,
        "Roughness" => MapKind::Roughness,
        "Metallic" => MapKind::Metallic,
        "Height" => MapKind::Height,
        "Normal" => MapKind::Normal,
        "Emissive" => MapKind::Emissive,
        "Opacity" => MapKind::Opacity,
        "Ao" => MapKind::Ao,
        _ => return None,
    })
}

fn upload_thumbnails(
    ctx: &egui::Context,
    loader: &crate::library::ThumbnailLoader,
    thumbs: &mut std::collections::HashMap<String, egui::TextureHandle>,
) {
    for (key, image) in loader.drain() {
        let colour = egui::ColorImage::from_rgba_unmultiplied(
            [image.width as usize, image.height as usize],
            &image.rgba,
        );
        let handle = ctx.load_texture(&key, colour, egui::TextureOptions::LINEAR);
        thumbs.insert(key, handle);
    }
}

/// Indices of the assets matching a whitespace-separated filter.
fn filter_assets(items: &[crate::library::LibraryMaterial], filter: &str) -> Vec<usize> {
    let terms: Vec<String> = filter.split_whitespace().map(|t| t.to_lowercase()).collect();
    items
        .iter()
        .enumerate()
        .filter(|(_, m)| m.matches(&terms))
        .map(|(i, _)| i)
        .collect()
}


/// Texel region a segment's stamps can reach.
///
/// Conservative by construction: a painted fragment lies inside some triangle,
/// so that triangle's screen box must overlap the brush box, so its UV box is
/// included. Widening the result is harmless; narrowing it would clip paint.
fn region_from_projection(
    proj: &PaintProjection,
    stamps: &[Vec2],
    radius: f32,
    resolution: u32,
) -> PaintRegion {
    if proj.degenerate || proj.tris.is_empty() || stamps.is_empty() {
        return PaintRegion::Everything;
    }

    let mut lo = Vec2::splat(f32::MAX);
    let mut hi = Vec2::splat(f32::MIN);
    for s in stamps {
        lo = lo.min(*s - Vec2::splat(radius));
        hi = hi.max(*s + Vec2::splat(radius));
    }

    let mut uv_lo = Vec2::splat(f32::MAX);
    let mut uv_hi = Vec2::splat(f32::MIN);
    for t in &proj.tris {
        if t.screen_hi.x < lo.x
            || t.screen_lo.x > hi.x
            || t.screen_hi.y < lo.y
            || t.screen_lo.y > hi.y
        {
            continue;
        }
        uv_lo = uv_lo.min(t.uv_lo);
        uv_hi = uv_hi.max(t.uv_hi);
    }
    if uv_lo.x > uv_hi.x {
        return PaintRegion::Nothing;
    }

    let res = resolution as i64;
    let margin = SCISSOR_MARGIN as i64;
    let x0 = ((uv_lo.x * resolution as f32).floor() as i64 - margin).clamp(0, res);
    let y0 = ((uv_lo.y * resolution as f32).floor() as i64 - margin).clamp(0, res);
    let x1 = ((uv_hi.x * resolution as f32).ceil() as i64 + margin).clamp(0, res);
    let y1 = ((uv_hi.y * resolution as f32).ceil() as i64 + margin).clamp(0, res);
    if x1 <= x0 || y1 <= y0 {
        return PaintRegion::Nothing;
    }
    PaintRegion::Rect([x0 as u32, y0 as u32, (x1 - x0) as u32, (y1 - y0) as u32])
}


/// Flag every triangle whose group was picked.
fn marks_for_groups(groups: &[u32], wanted: &std::collections::HashSet<u32>) -> Vec<u32> {
    groups.iter().map(|g| u32::from(wanted.contains(g))).collect()
}

/// What a pointer drag should do to the camera.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum CameraGesture {
    Orbit,
    Pan,
}

/// Route a drag to the camera, or to nothing when the brush owns it.
///
/// The rule that matters: while the paint tool is active the left button always
/// belongs to the brush, in *every* stroke mode. Getting this wrong made Line
/// and Stamp mode orbit the camera while drawing.
fn camera_gesture(
    brush_owns_drag: bool,
    primary: bool,
    panning: bool,
) -> Option<CameraGesture> {
    if panning {
        // Right/middle drag pans in every tool, painting or not.
        Some(CameraGesture::Pan)
    } else if primary && !brush_owns_drag {
        Some(CameraGesture::Orbit)
    } else {
        None
    }
}


/// Möller–Trumbore. Returns the ray parameter of the hit.
fn ray_hits_triangle(
    origin: Vec3,
    dir: Vec3,
    a: Vec3,
    b: Vec3,
    c: Vec3,
) -> Option<f32> {
    ray_hit_barycentric(origin, dir, a, b, c).map(|(t, _, _)| t)
}

/// Möller–Trumbore with the barycentrics: (t, u, v), where the hit point is
/// a·(1-u-v) + b·u + c·v.
fn ray_hit_barycentric(
    origin: Vec3,
    dir: Vec3,
    a: Vec3,
    b: Vec3,
    c: Vec3,
) -> Option<(f32, f32, f32)> {
    let (e1, e2) = (b - a, c - a);
    let p = dir.cross(e2);
    let det = e1.dot(p);
    if det.abs() < 1e-8 {
        return None;
    }
    let inv = 1.0 / det;
    let t = origin - a;
    let u = t.dot(p) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = t.cross(e1);
    let v = dir.dot(q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let dist = e2.dot(q) * inv;
    (dist > 1e-5).then_some((dist, u, v))
}

/// What a ray pick landed on.
struct SurfaceHit {
    set: usize,
    model: usize,
    uv: Vec2,
    /// Hit point and geometric normal, world space.
    world: Vec3,
    normal: Vec3,
}

/// World point -> viewport px, the same convention the paint shaders and
/// `pick_surface` use (ndc * 0.5 + 0.5, y down). None behind the camera or
/// outside the viewport.
fn project_to_viewport(view_proj: Mat4, size: Vec2, p: Vec3) -> Option<Vec2> {
    let clip = view_proj * p.extend(1.0);
    if clip.w <= 1e-6 {
        return None;
    }
    let ndc = Vec2::new(clip.x, clip.y) / clip.w;
    let px = (ndc * 0.5 + Vec2::splat(0.5)) * size;
    (px.x >= 0.0 && px.y >= 0.0 && px.x < size.x && px.y < size.y).then_some(px)
}

/// A clone source pinned to the surface.
#[derive(Clone, Copy)]
struct CloneAnchor {
    world: Vec3,
    normal: Vec3,
}

/// Bilinear, repeating lookup in an sRGB RGBA8 image; returns linear RGB.
/// UV v runs down the image, as the viewport shader samples it.
fn sample_srgb_image(image: &crate::scene::ImageData, uv: Vec2) -> Vec3 {
    let (w, h) = (image.width.max(1), image.height.max(1));
    if image.rgba.len() < (w * h * 4) as usize {
        return Vec3::ONE;
    }
    let x = (uv.x - uv.x.floor()) * w as f32 - 0.5;
    let y = (uv.y - uv.y.floor()) * h as f32 - 0.5;
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let texel = |tx: f32, ty: f32| {
        let tx = (tx as i64).rem_euclid(w as i64) as u32;
        let ty = (ty as i64).rem_euclid(h as i64) as u32;
        let i = ((ty * w + tx) * 4) as usize;
        Vec3::new(
            srgb_component_to_linear(image.rgba[i] as f32 / 255.0),
            srgb_component_to_linear(image.rgba[i + 1] as f32 / 255.0),
            srgb_component_to_linear(image.rgba[i + 2] as f32 / 255.0),
        )
    };
    let top = texel(x0, y0).lerp(texel(x0 + 1.0, y0), fx);
    let bottom = texel(x0, y0 + 1.0).lerp(texel(x0 + 1.0, y0 + 1.0), fx);
    top.lerp(bottom, fy)
}


/// The Blender view keys pressed this frame.
struct ViewKeys {
    hide: bool,
    hide_others: bool,
    unhide: bool,
    isolate: bool,
    select_all: bool,
    deselect_all: bool,
    frame_selected: bool,
    frame_all: bool,
}

/// Apply Blender's isolate toggle to a list of visibility flags.
///
/// First press stores what was visible and shows only `selected`; the second
/// press puts everything back exactly as it was. Returns the backup to keep
/// (`None` once restored).
fn toggle_isolation(
    visible: &mut [bool],
    selected: &[usize],
    backup: Option<Vec<bool>>,
) -> Option<Vec<bool>> {
    if let Some(previous) = backup {
        for (flag, was) in visible.iter_mut().zip(previous) {
            *flag = was;
        }
        return None;
    }
    if selected.is_empty() {
        return None;
    }
    let saved: Vec<bool> = visible.to_vec();
    for (i, flag) in visible.iter_mut().enumerate() {
        *flag = selected.contains(&i);
    }
    Some(saved)
}

fn create_scene_image(
    memory_allocator: &Arc<StandardMemoryAllocator>,
    size: [u32; 2],
) -> Arc<ImageView> {
    ImageView::new_default(
        crate::vram::image(
            memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: DEFAULT_IMAGE_FORMAT,
                extent: [size[0].max(1), size[1].max(1), 1],
                // STORAGE lets the path tracer resolve straight into it.
                // TRANSFER_SRC: the assistant reads captures back.
                usage: ImageUsage::SAMPLED
                    | ImageUsage::COLOR_ATTACHMENT
                    | ImageUsage::STORAGE
                    | ImageUsage::TRANSFER_SRC,
                ..Default::default()
            },
            crate::vram::Kind::Viewport,
        )
        .unwrap(),
    )
    .unwrap()
}

impl ViewportState {
    /// Make sure the offscreen scene image matches the panel size and is
    /// registered with egui; returns the texture id to draw.
    fn ensure(&mut self, gui: &mut Gui, desired: [u32; 2]) -> egui::TextureId {
        let desired = [desired[0].max(16), desired[1].max(16)];
        if self.texture_id.is_none() || desired != self.size {
            if let Some(id) = self.texture_id.take() {
                gui.unregister_user_image(id);
            }
            self.scene_image = create_scene_image(&self.memory_allocator, desired);
            self.size = desired;
            self.texture_id =
                Some(gui.register_user_image_view(self.scene_image.clone(), Default::default()));
        }
        self.texture_id.unwrap()
    }
}

impl AppState {
    fn layout(&mut self, gui: &mut Gui) {
        let ctx = gui.context();
        self.poll_import();
        self.poll_project_job();
        self.poll_shape_import();
        self.poll_assistant();
        self.ensure_active_set_allocated();
        self.update_set_residency();
        self.remember_hidden_pieces();
        // Rebuilt by the texture-set list as it draws (if it is showing).
        self.outliner_hover.clear();

        // Ctrl+Z / Ctrl+Shift+Z (or Ctrl+Y): stroke undo/redo history.
        let (undo_pressed, redo_pressed) = ctx.input(|i| {
            let z = i.key_pressed(egui::Key::Z) && i.modifiers.command;
            let y = i.key_pressed(egui::Key::Y) && i.modifiers.command;
            (z && !i.modifiers.shift, (z && i.modifiers.shift) || y)
        });
        let tool_before = self.tool;
        // Tool shortcuts. Guarded on keyboard focus so typing in a filter box
        // does not switch tools out from under you.
        if !ctx.wants_keyboard_input() {
            let (swap_colours, reset_colours) = ctx.input(|i| {
                let plain = !i.modifiers.command && !i.modifiers.alt && !i.modifiers.shift;
                (plain && i.key_pressed(egui::Key::X), plain && i.key_pressed(egui::Key::D))
            });
            if swap_colours {
                self.swap_colours();
            }
            if reset_colours {
                self.reset_colours();
            }
            let (paint, erase, smudge, picker, orbit, select, stamp, clone, place) = ctx.input(|i| {
                let plain = !i.modifiers.command && !i.modifiers.alt && !i.modifiers.shift;
                (
                    plain && i.key_pressed(egui::Key::B),
                    plain && i.key_pressed(egui::Key::E),
                    plain && i.key_pressed(egui::Key::G),
                    plain && i.key_pressed(egui::Key::I),
                    plain && i.key_pressed(egui::Key::R),
                    plain && i.key_pressed(egui::Key::S),
                    plain && i.key_pressed(egui::Key::T),
                    plain && i.key_pressed(egui::Key::C),
                    plain && i.key_pressed(egui::Key::P),
                )
            });
            if place {
                self.select_tool(Tool::Place);
            } else if stamp {
                self.tool = Tool::Stamp;
            } else if clone {
                self.tool = Tool::Clone;
            } else if paint {
                self.tool = Tool::Paint;
            } else if erase {
                self.tool = Tool::Erase;
            } else if smudge {
                self.tool = Tool::Smudge;
            } else if picker {
                self.tool = Tool::Picker;
            } else if orbit {
                self.tool = Tool::Orbit;
            } else if select {
                self.tool = Tool::Select;
            }

            // [ and ] size the brush, Shift for bigger steps. Held keys
            // repeat. Ctrl+[ / Ctrl+] belong to Move layer, so not with Ctrl.
            // With Shift held the logical key can arrive as { or }.
            let (shrink, grow, fast) = ctx.input(|i| {
                let free = !i.modifiers.command && !i.modifiers.alt;
                (
                    free && (i.key_pressed(egui::Key::OpenBracket) || i.key_pressed(egui::Key::OpenCurlyBracket)),
                    free && (i.key_pressed(egui::Key::CloseBracket) || i.key_pressed(egui::Key::CloseCurlyBracket)),
                    i.modifiers.shift,
                )
            });
            if shrink != grow {
                self.brush.size = step_brush_size(self.brush.size, grow, fast);
            }
        }

        // Blender's selection, visibility and view keys, in every tool,
        // whenever nothing has keyboard focus and no button is held.
        let keys = ctx.input(|i| {
            let plain = !i.modifiers.command && !i.modifiers.alt && !i.modifiers.shift;
            let h = i.key_pressed(egui::Key::H) && !i.modifiers.command;
            let a = i.key_pressed(egui::Key::A) && !i.modifiers.command && !i.modifiers.shift;
            ViewKeys {
                hide: h && !i.modifiers.shift && !i.modifiers.alt,
                hide_others: h && i.modifiers.shift,
                unhide: h && i.modifiers.alt,
                // Numpad / arrives as Slash too. No modifier check: some
                // layouts need Shift to type a slash.
                isolate: i.key_pressed(egui::Key::Slash) && !i.modifiers.command,
                select_all: a && !i.modifiers.alt,
                deselect_all: a && i.modifiers.alt,
                frame_selected: plain && i.key_pressed(egui::Key::F),
                frame_all: plain && i.key_pressed(egui::Key::Home),
            }
        });
        let frame_pending = std::mem::take(&mut self.frame_selection_pending);
        if !ctx.wants_keyboard_input() && !ctx.input(|i| i.pointer.any_down()) {
            if keys.hide {
                self.hide_selected();
            } else if keys.hide_others {
                self.hide_unselected();
            } else if keys.unhide {
                self.unhide_all();
            }
            if keys.isolate {
                self.toggle_isolate();
            }
            if keys.select_all {
                self.select_all();
            } else if keys.deselect_all {
                self.deselect_all();
            }
            if keys.frame_selected || frame_pending {
                self.frame_selection();
            } else if keys.frame_all {
                self.frame_all();
            }
        }

        // One history for the whole document. A focused text field keeps
        // Ctrl Z for its own text undo; mid-gesture (a stroke or drag still
        // held) there is nothing settled to step back from.
        let gesture = ctx.input(|i| i.pointer.any_down()) || self.stroke_last.is_some();
        if !ctx.wants_keyboard_input() && !gesture {
            if undo_pressed {
                self.undo();
            } else if redo_pressed {
                self.redo();
            }
        }

        if let Some(cmd) = self.pressed_command(&ctx) {
            self.run_command(cmd, &ctx);
        }

        // Chrome first — top bars, the status bar, then the side docks — so
        // the viewport gets whatever is left.
        self.menu_bar(&ctx);
        self.context_bar(&ctx);
        self.status_bar(&ctx);
        self.tool_rail(&ctx);
        self.brush_dock(&ctx);
        // Declared before the right dock so they sit outermost: the rail
        // with the button that pops the assistant open, then the assistant.
        self.assistant_rail(&ctx);
        self.assistant_dock(&ctx);
        self.right_dock(&ctx);
        self.export_dialog(&ctx);
        self.import_dialog(&ctx);
        self.windows(&ctx);
        self.studio_ui(gui);
        self.welcome_screen(&ctx);

        // Remember which brush to hand back to once the picker has picked.
        if self.tool == Tool::Picker && tool_before != Tool::Picker {
            self.picker_return = if tool_before.uses_brush() { tool_before } else { Tool::Paint };
        }

        let mut viewport_rect = None;
        egui::CentralPanel::default()
            // The clear colour shows only for the frame before the scene
            // image has resized to the panel.
            .frame(egui::Frame::new().fill(crate::theme::BG_VIEWPORT))
            .show(&ctx, |ui| {
                let avail = ui.available_size();
                let desired = [avail.x as u32, avail.y as u32];
                let texture_id = self.viewport.ensure(gui, desired);

                let response = ui.add(
                    egui::Image::new(ImageSource::Texture(SizedTexture::new(texture_id, avail)))
                        .sense(Sense::click_and_drag()),
                );

                let alt = ui.input(|i| i.modifiers.alt);
                // Capture by value: a closure borrowing `self` would block the
                // paint calls below.
                let rect = response.rect;
                viewport_rect = Some(rect);
                let viewport_px =
                    Vec2::new(self.viewport.size[0] as f32, self.viewport.size[1] as f32);
                let to_viewport = move |pos: egui::Pos2| {
                    let local = pos - rect.min;
                    Vec2::new(
                        local.x * viewport_px.x / rect.width().max(1.0),
                        local.y * viewport_px.y / rect.height().max(1.0),
                    )
                };
                let pointer = response.interact_pointer_pos();
                let delta = response.drag_delta();
                let primary = response.dragged_by(egui::PointerButton::Primary);
                let panning = response.dragged_by(egui::PointerButton::Secondary)
                    || response.dragged_by(egui::PointerButton::Middle);
                // Alt turns the paint tool back into an orbit tool.
                let ctrl_held = ui.input(|i| i.modifiers.command);
                // Ctrl+click with the clone tool picks the source, not paint.
                let setting_source = self.tool == Tool::Clone && ctrl_held && !alt;
                if setting_source
                    && (response.clicked() || response.drag_started_by(egui::PointerButton::Primary))
                    && let Some(pos) = pointer
                {
                    self.set_clone_source(to_viewport(pos));
                }
                let paint_tool = self.tool.uses_brush() && !alt && !setting_source;
                // Erase and smudge are the brush doing something else; keep the
                // op in step with the tool before any stroke this frame.
                self.brush.op = self.tool.brush_op();
                let select_tool = self.tool == Tool::Select && !alt;
                let picker_tool = self.tool == Tool::Picker && !alt;

                if picker_tool {
                    if (primary || response.clicked())
                        && let Some(pos) = pointer
                    {
                        self.pick_color_at(to_viewport(pos));
                    }
                    if response.drag_stopped() || response.clicked() {
                        self.tool = self.picker_return;
                    }
                }

                let place_tool = self.tool == Tool::Place && !alt;
                if place_tool {
                    if primary {
                        let shift = ui.input(|i| i.modifiers.shift);
                        self.drag_projection(delta, rect.height(), shift);
                    }
                    self.draw_projection_gizmo(ui, rect);
                }

                // A click (not a drag) picks a piece in the Select tool, and in
                // the Orbit tool too, since dragging is what orbits there.
                let orbit_tool = self.tool == Tool::Orbit && !alt;
                if (select_tool || orbit_tool)
                    && response.clicked()
                    && let Some(pos) = response.interact_pointer_pos()
                {
                    let px = to_viewport(pos);
                    let extend = ui.input(|i| i.modifiers.shift || i.modifiers.command);
                    self.select_at(px, extend);
                }

                if paint_tool {
                    // The left button belongs to the brush here, whatever the
                    // stroke mode: dragging must never also orbit.
                    let stroke_mode = if self.tool == Tool::Stamp {
                        crate::paint::StrokeMode::Stamp
                    } else {
                        self.brush.stroke_mode
                    };
                    match stroke_mode {
                        crate::paint::StrokeMode::Freehand => {
                            if primary {
                                if let Some(pos) = pointer {
                                    self.paint_at(to_viewport(pos));
                                }
                            } else if self.stroke_last.is_some() {
                                self.finish_stroke();
                            }
                        }
                        // Anchor on press, preview while held, commit on release.
                        crate::paint::StrokeMode::Line => {
                            if response.drag_started_by(egui::PointerButton::Primary)
                                && let Some(pos) = pointer
                            {
                                self.line_anchor = Some((to_viewport(pos), pos));
                            }
                            if let Some((anchor_px, anchor_screen)) = self.line_anchor {
                                let cursor = pointer
                                    .or_else(|| ui.input(|i| i.pointer.hover_pos()))
                                    .unwrap_or(anchor_screen);
                                if primary {
                                    ui.painter().line_segment(
                                        [anchor_screen, cursor],
                                        egui::Stroke::new(
                                            1.5_f32,
                                            egui::Color32::from_rgb(255, 190, 90),
                                        ),
                                    );
                                } else {
                                    self.paint_line(anchor_px, to_viewport(cursor));
                                    self.line_anchor = None;
                                }
                            }
                        }
                        // One decal per press, the moment the button goes
                        // down: drag_started only fires once the pointer has
                        // moved past egui's drag threshold, so a plain click
                        // would never stamp.
                        crate::paint::StrokeMode::Stamp => {
                            let pressed = ui.input(|i| i.pointer.button_pressed(egui::PointerButton::Primary));
                            if pressed
                                && response.hovered()
                                && let Some(pos) = ui.input(|i| i.pointer.interact_pos())
                            {
                                self.place_stamp(to_viewport(pos));
                            }
                        }
                    }
                } else {
                    // Orbit tool, or the paint tool with alt held.
                    if self.stroke_last.is_some() {
                        self.finish_stroke();
                    }
                    // Don't leave a half-made line waiting if the gesture was
                    // interrupted by switching tools or holding alt.
                    self.line_anchor = None;
                }

                // Select mode's left click is a pick, not an orbit.
                match camera_gesture(
                    paint_tool || select_tool || picker_tool || place_tool || setting_source,
                    primary,
                    panning,
                ) {
                    Some(CameraGesture::Orbit) => self.camera.orbit(delta.x, delta.y),
                    Some(CameraGesture::Pan) => self.camera.pan(delta.x, delta.y),
                    None => {}
                }
                if response.hovered() {
                    // egui rewrites ctrl+scroll into a *zoom* event, so
                    // smooth_scroll_delta is zero exactly when ctrl is held —
                    // the brush-size gesture has to read zoom_delta instead.
                    let (scroll, scroll_x, zoom, ctrl, shift) = ui.input(|i| {
                        (
                            i.smooth_scroll_delta.y,
                            i.smooth_scroll_delta.x,
                            i.zoom_delta(),
                            i.modifiers.ctrl,
                            i.modifiers.shift,
                        )
                    });
                    if place_tool && (shift || ctrl) {
                        let turn = scroll + scroll_x;
                        let factor = if (zoom - 1.0).abs() > 1e-4 {
                            zoom
                        } else {
                            (turn * 0.004).exp()
                        };
                        self.scroll_projection(shift, turn * 0.004, factor);
                    } else if shift && self.tool.uses_brush() {
                        // Some platforms turn shift+wheel into horizontal
                        // scroll, so take whichever axis moved.
                        let turn = scroll + scroll_x;
                        if turn != 0.0 {
                            let r = self.brush.tip_rotation + turn * 0.004;
                            self.brush.tip_rotation = (r + std::f32::consts::PI)
                                .rem_euclid(std::f32::consts::TAU)
                                - std::f32::consts::PI;
                        }
                    } else if ctrl && self.tool.uses_brush() {
                        // Multiplicative, so the step stays proportional and is
                        // usable at both 2px and 200px.
                        let factor = if (zoom - 1.0).abs() > 1e-4 {
                            zoom
                        } else {
                            (scroll * 0.004).exp()
                        };
                        if (factor - 1.0).abs() > 1e-4 {
                            self.brush.size = (self.brush.size * factor).clamp(2.0, 200.0);
                        }
                    } else if scroll != 0.0 {
                        self.camera.zoom(scroll * 0.02);
                    }
                    // Brush cursor.
                    if self.tool.uses_brush() {
                        if let Some(pos) = ui.input(|i| i.pointer.hover_pos()) {
                            let scale = response.rect.width() / self.viewport.size[0].max(1) as f32;
                            let radius = self.brush.size * scale;
                            if self.tool == Tool::Stamp {
                                self.draw_stamp_preview(ui, pos, radius);
                            } else {
                                ui.painter().circle_stroke(
                                    pos,
                                    radius,
                                    egui::Stroke::new(1.0_f32, egui::Color32::WHITE),
                                );
                            }
                            if self.tool == Tool::Clone {
                                // Where the copy is coming from right now.
                                // Mid-stroke the offset is fixed; otherwise ask the
                                // model where the source sits for this cursor.
                                let source_px = match self.clone_stroke_offset {
                                    Some(offset) if self.stroke_last.is_some() => {
                                        Some(to_viewport(pos) + offset)
                                    }
                                    _ => self.clone_source_px(to_viewport(pos)),
                                };
                                let source =
                                    source_px.map(|src| rect.min + egui::vec2(src.x, src.y) * scale);
                                if let Some(at) = source {
                                    let stroke = egui::Stroke::new(1.5_f32, egui::Color32::from_rgb(90, 220, 255));
                                    ui.painter().circle_stroke(at, radius, stroke);
                                    let arm = 6.0_f32;
                                    ui.painter().line_segment([at - egui::vec2(arm, 0.0), at + egui::vec2(arm, 0.0)], stroke);
                                    ui.painter().line_segment([at - egui::vec2(0.0, arm), at + egui::vec2(0.0, arm)], stroke);
                                }
                            }
                        }
                    }
                }
            });
        if let Some(rect) = viewport_rect {
            self.viewport_overlays(&ctx, rect);
            self.studio_frame_guide(&ctx, rect);
        }

        // Record this frame's document changes once the gesture making them
        // is over: a drag on release, a typed name when the field is left.
        let busy = ctx.input(|i| i.pointer.any_down())
            || ctx.wants_keyboard_input()
            || self.stroke_last.is_some()
            || self.import.is_some()
            || self.project_job.is_some();
        self.track_history(busy);
    }

    /// The active layer's projection, when it is one the Place tool can move:
    /// a projected fill on a set whose mesh maps are baked.
    fn placeable_projection(&self) -> Option<(Vec3, f32)> {
        let layers = &self.texture_sets[self.active_set].layers;
        let layer = layers.active_layer();
        if !layer.has_fill || layer.projection == Projection::Uv || layer.purpose.is_some() {
            return None;
        }
        layers.projection_frame().map(|(centre, size)| (Vec3::from(centre), size))
    }

    /// Slide (or with `rotate`, turn) the active projection by a viewport
    /// drag of `delta` points, in the camera's own screen plane so it moves
    /// with the cursor at the projection's depth.
    fn drag_projection(&mut self, delta: egui::Vec2, view_height: f32, rotate: bool) {
        let Some((centre, size)) = self.placeable_projection() else {
            return;
        };
        if delta == egui::Vec2::ZERO {
            return;
        }
        let view = self.camera.view_matrix();
        let right = Vec3::new(view.x_axis.x, view.y_axis.x, view.z_axis.x);
        let up = Vec3::new(view.x_axis.y, view.y_axis.y, view.z_axis.y);
        let layers = &mut self.texture_sets[self.active_set].layers;
        let active = layers.active;
        let layer = &mut layers.layers[active];
        if rotate {
            let [rx, ry, rz] = layer.projection_rotation;
            let current = glam::Quat::from_euler(glam::EulerRot::XYZ, rx, ry, rz);
            let turn = glam::Quat::from_axis_angle(up, delta.x * 0.01)
                * glam::Quat::from_axis_angle(right, delta.y * 0.01);
            let (x, y, z) = (turn * current).normalize().to_euler(glam::EulerRot::XYZ);
            layer.projection_rotation = [x, y, z];
        } else {
            let at = centre + Vec3::from(layer.projection_offset);
            let depth = (view.transform_point3(at).z).abs().max(size * 0.01);
            let per_point = 2.0 * depth * (self.camera.fov_y * 0.5).tan() / view_height.max(1.0);
            let moved = (right * delta.x - up * delta.y) * per_point;
            layer.projection_offset = (Vec3::from(layer.projection_offset) + moved).to_array();
        }
        layers.mark_dirty();
    }

    /// Ctrl+scroll sizes the active projection by `factor`; Shift+scroll
    /// (`spin`) turns it by `turn` radians about the view direction.
    fn scroll_projection(&mut self, spin: bool, turn: f32, factor: f32) {
        if self.placeable_projection().is_none() {
            return;
        }
        let forward = (self.camera.target - self.camera.position()).normalize_or_zero();
        let layers = &mut self.texture_sets[self.active_set].layers;
        let active = layers.active;
        let layer = &mut layers.layers[active];
        if spin {
            if turn == 0.0 {
                return;
            }
            let [rx, ry, rz] = layer.projection_rotation;
            let current = glam::Quat::from_euler(glam::EulerRot::XYZ, rx, ry, rz);
            let (x, y, z) = (glam::Quat::from_axis_angle(forward, turn) * current)
                .normalize()
                .to_euler(glam::EulerRot::XYZ);
            layer.projection_rotation = [x, y, z];
        } else {
            if (factor - 1.0).abs() < 1e-4 {
                return;
            }
            layer.projection_scale = (layer.projection_scale * factor).clamp(0.02, 4.0);
        }
        layers.mark_dirty();
    }

    /// The projection volume as a wireframe box over the viewport, its axes
    /// in X/Y/Z colours, so the Place tool shows what it is moving.
    fn draw_projection_gizmo(&self, ui: &egui::Ui, rect: egui::Rect) {
        let Some((centre, size)) = self.placeable_projection() else {
            ui.painter().text(
                rect.center_top() + egui::vec2(0.0, 24.0),
                egui::Align2::CENTER_TOP,
                "Select a layer with a projected image (Projection ≠ UV) to place it",
                egui::FontId::proportional(13.0),
                egui::Color32::from_rgb(255, 179, 102),
            );
            return;
        };
        let layer = self.texture_sets[self.active_set].layers.active_layer();
        let placement = layer.projection_placement(centre.to_array(), size);
        let aspect = self.viewport.size[0] as f32 / self.viewport.size[1].max(1) as f32;
        let view_proj = self.camera.view_proj(aspect);
        let to_screen = |p: Vec3| -> Option<egui::Pos2> {
            let clip = view_proj * placement.transform_point3(p).extend(1.0);
            if clip.w <= 1e-6 {
                return None;
            }
            let ndc = Vec2::new(clip.x, clip.y) / clip.w;
            Some(rect.min + egui::vec2((ndc.x * 0.5 + 0.5) * rect.width(), (ndc.y * 0.5 + 0.5) * rect.height()))
        };
        let corner = |i: u32| {
            Vec3::new(
                if i & 1 == 0 { -0.5 } else { 0.5 },
                if i & 2 == 0 { -0.5 } else { 0.5 },
                if i & 4 == 0 { -0.5 } else { 0.5 },
            )
        };
        let outline = egui::Stroke::new(1.25_f32, egui::Color32::from_rgb(245, 200, 75));
        for a in 0..8u32 {
            for bit in [1u32, 2, 4] {
                let b = a | bit;
                if b != a
                    && let (Some(p), Some(q)) = (to_screen(corner(a)), to_screen(corner(b)))
                {
                    ui.painter().line_segment([p, q], outline);
                }
            }
        }
        let axes = [
            (Vec3::X, crate::theme::AXIS_X),
            (Vec3::Y, crate::theme::AXIS_Y),
            (Vec3::Z, crate::theme::AXIS_Z),
        ];
        if let Some(origin) = to_screen(Vec3::ZERO) {
            for (dir, colour) in axes {
                if let Some(tip) = to_screen(dir * 0.5) {
                    ui.painter().line_segment([origin, tip], egui::Stroke::new(2.0_f32, colour));
                }
            }
        }
    }

    /// The stamp as it will land: the tip image at the cursor with the
    /// brush's size, aspect and rotation, half transparent. Without a tip it
    /// is a solid square, so an outline says as much.
    fn draw_stamp_preview(&mut self, ui: &mut egui::Ui, pos: egui::Pos2, radius: f32) {
        let aspect = self.brush.tip_aspect.max(1e-3);
        let half = if aspect >= 1.0 {
            egui::vec2(radius, radius / aspect)
        } else {
            egui::vec2(radius * aspect, radius)
        };
        let rect = egui::Rect::from_center_size(pos, half * 2.0);
        if self.tip_preview.is_none()
            && let Some(image) = self.tip_preview_image.clone()
        {
            self.tip_preview =
                Some(ui.ctx().load_texture("stamp_preview", image, egui::TextureOptions::LINEAR));
        }
        let rotation = self.brush.tip_rotation;
        match &self.tip_preview {
            Some(texture) => {
                let tint = if self.brush.use_tip_color {
                    egui::Color32::from_white_alpha(150)
                } else {
                    let [r, g, b] = self.brush.color.map(|c| (c.clamp(0.0, 1.0) * 255.0) as u8);
                    egui::Color32::from_rgba_unmultiplied(r, g, b, 150)
                };
                // The shader rotates the lookup by +rotation, so the image
                // itself appears turned the other way.
                egui::Image::new(texture)
                    .rotate(-rotation, egui::Vec2::splat(0.5))
                    .tint(tint)
                    .paint_at(ui, rect);
            }
            None => {
                let (s, c) = (-rotation).sin_cos();
                let corners = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)].map(|(x, y)| {
                    let v = egui::vec2(x * half.x, y * half.y);
                    pos + egui::vec2(v.x * c - v.y * s, v.x * s + v.y * c)
                });
                ui.painter().add(egui::Shape::closed_line(
                    corners.to_vec(),
                    egui::Stroke::new(1.0_f32, egui::Color32::WHITE),
                ));
            }
        }
    }

    /// Deposit brush stamps from the previous pointer position to `px`.
    /// One decal stamp at a point — a whole stroke in itself, so it lands in
    /// the undo history as a single step.
    fn place_stamp(&mut self, px: Vec2) {
        self.stroke_last = None;
        self.paint_at(px);
        self.finish_stroke();
    }

    /// A straight run of stamps between two points, committed as one stroke.
    fn paint_line(&mut self, from: Vec2, to: Vec2) {
        self.stroke_last = Some(from);
        self.paint_at(to);
        self.finish_stroke();
    }

    /// Restore full-quality compositing and re-flatten, so the UV padding the
    /// stroke skipped is applied once at the end.
    fn finish_stroke(&mut self) {
        self.stroke_last = None;
        self.fill_lock = None;
        let layers = &mut self.texture_sets[self.active_set].layers;
        layers.dilate = true;
        layers.mark_dirty();
    }

    fn paint_at(&mut self, px: Vec2) {
        let brush = self.effective_brush();
        let fresh = self.stroke_last.is_none();
        // Painting needs the raster depth buffer; PT mode has none.
        if self.pt_mode {
            self.ui.status = "Switch to Raster mode to paint".into();
            return;
        }
        // Can't paint what isn't shown.
        if !self.texture_sets[self.active_set].visible {
            self.ui.status = format!(
                "'{}' is hidden — unhide it to paint",
                self.texture_sets[self.active_set].label()
            );
            return;
        }
        // Dilation is a full-image dispatch per composite; while the stroke is
        // moving it is not worth paying for, so it runs once at stroke end.
        self.texture_sets[self.active_set].layers.dilate = false;

        // A fresh stroke: capture history for exactly the channels it touches.
        if self.stroke_last.is_none() {
            let mut channels = Vec::new();
            if brush.target == crate::paint::PaintTarget::Mask {
                // A mask stroke touches only the mask, whatever the per-channel
                // toggles say.
                channels.push(PaintChannel::Mask);
            } else {
                channels = brush.material_channels();
            }
            self.texture_sets[self.active_set].layers.begin_stroke(&channels);
            // Channels the stroke writes for the first time just got their
            // textures; undo history must see the same layer, not an edit.
            let upgrades = self.texture_sets[self.active_set].layers.take_upgrades();
            if !upgrades.is_empty() {
                self.adopt_upgrades(self.active_set, &upgrades);
            }
            // The layer list's swatch: the colour last painted onto it.
            if brush.target == crate::paint::PaintTarget::Material
                && brush.paint_color
                && brush.op == crate::paint::BrushOp::Paint
                && !brush.use_tip_color
            {
                // A filled layer keeps its image's colour; brushwork on it is
                // usually a touch-up, not what the layer is.
                let layers = &mut self.texture_sets[self.active_set].layers;
                let active = layers.active;
                if !layers.layers[active].has_fill {
                    layers.layers[active].swatch = Some(brush.color.map(srgb_component_to_linear));
                }
            }
        }
        let start = self.stroke_last.unwrap_or(px);
        let spacing = (brush.size * 0.25).max(1.0);
        let dist = start.distance(px);

        let mut stamps = Vec::new();
        if dist < 1e-3 {
            stamps.push(px);
        } else {
            let steps = (dist / spacing).ceil() as u32;
            for i in 1..=steps {
                stamps.push(start.lerp(px, i as f32 / steps as f32));
            }
        }
        self.stroke_last = Some(px);

        let aspect = self.viewport.size[0] as f32 / self.viewport.size[1] as f32;
        let view = PaintView {
            view_proj: self.camera.view_proj(aspect),
            camera_pos: self.camera.position(),
            viewport_size: Vec2::new(
                self.viewport.size[0] as f32,
                self.viewport.size[1] as f32,
            ),
        };
        self.ensure_paint_projection(&view);
        let resolution = self.texture_sets[self.active_set].layers.resolution;
        // A stamp reaches into its corners, past the round brush radius.
        let reach = if brush.square_tip { brush.size * std::f32::consts::SQRT_2 } else { brush.size };
        let scissor = match self.stroke_region(&stamps, reach, resolution) {
            // The brush is over empty space; nothing to rasterise.
            PaintRegion::Nothing => return,
            PaintRegion::Everything => None,
            PaintRegion::Rect(rect) => Some(rect),
        };

        if brush.op == crate::paint::BrushOp::Clone {
            if brush.target == crate::paint::PaintTarget::Mask {
                self.ui.status = "Clone works on layers, not layer masks".into();
                return;
            }
            let Some(anchor) = self.clone_source else {
                self.ui.status = "Ctrl+click the model to set a clone source first".into();
                return;
            };
            if fresh {
                // Aligned: the first stroke fixes the source's offset from
                // the brush on the model; later strokes reuse it from
                // wherever they start.
                if self.ui.clone_aligned
                    && self.clone_offset.is_none()
                    && let Some(hit) = self.pick_active_surface(px)
                {
                    self.clone_offset = Some(anchor.world - hit.world);
                }
                self.clone_stroke_offset = self.clone_source_px(px).map(|src| src - px);
                if self.clone_stroke_offset.is_none() {
                    self.ui.status = "Clone source is off screen — move the view or set a new one".into();
                } else if anchor.normal.dot(self.camera.position() - anchor.world) < 0.0
                    && !self.ui.clone_aligned
                {
                    self.ui.status = "Clone source faces away from the camera — turn the view".into();
                }
            }
            let Some(offset) = self.clone_stroke_offset else { return };
            let layers = &self.texture_sets[self.active_set].layers;
            let clone_source = match layers.active_layer().purpose {
                Some(kind) => match layers.shader_mask(kind) {
                    Some(view) => crate::paint::CloneSource::ShaderMask(view.clone()),
                    None => return,
                },
                None => crate::paint::CloneSource::Material,
            };
            self.paint.clone_stroke(
                &self.renderer,
                &mut self.texture_sets[self.active_set],
                self.active_set,
                &view,
                &brush,
                &stamps,
                scissor,
                offset,
                clone_source,
            );
            return;
        }

        if brush.op == crate::paint::BrushOp::Smudge {
            if brush.target == crate::paint::PaintTarget::Mask {
                self.ui.status = "Smudge works on material channels, not masks".into();
                return;
            }
            // Each stamp carries how far the brush moved to reach it; that
            // motion is what drags the paint.
            let mut prev = start;
            let smudge_stamps: Vec<(Vec2, Vec2)> = stamps
                .iter()
                .map(|&stamp| {
                    let step = stamp - prev;
                    prev = stamp;
                    (stamp, step)
                })
                .collect();
            self.paint.smudge(
                &self.renderer,
                &mut self.texture_sets[self.active_set],
                self.active_set,
                &view,
                &brush,
                &smudge_stamps,
                scissor,
            );
            return;
        }

        let marks = if brush.fill_mode.is_on() {
            self.ensure_mesh_groups();
            let groups = match brush.fill_mode {
                crate::paint::FillMode::Shell => {
                    self.mesh_groups.as_ref().map(|g| g.shell.clone())
                }
                crate::paint::FillMode::UvIsland => {
                    self.mesh_groups.as_ref().map(|g| g.island.clone())
                }
                // Triangle mode stops at the polygons the brush actually covers.
                _ => None,
            };
            match groups {
                Some(groups) => {
                    // Lock on to the piece the stroke started on. Without this,
                    // dragging across a gap would start filling whatever passed
                    // under the cursor next.
                    if self.fill_lock.is_none() {
                        self.fill_lock = stamps.iter().find_map(|stamp| {
                            self.pick_primitive(*stamp)
                                .and_then(|prim| groups.get(prim as usize).copied())
                        });
                    }
                    match self.fill_lock {
                        Some(locked) => {
                            let mut wanted = std::collections::HashSet::new();
                            wanted.insert(locked);
                            marks_for_groups(&groups, &wanted)
                        }
                        // Started off the model: nothing to fill this stroke.
                        None => Vec::new(),
                    }
                }
                // Triangle mode is flagged on the GPU, where the depth test
                // can reject polygons hidden behind the one you can see.
                None => Vec::new(),
            }
        } else {
            Vec::new()
        };

        self.paint.stroke(
            &self.renderer,
            &mut self.texture_sets[self.active_set],
            self.active_set,
            &view,
            &brush,
            &stamps,
            scissor,
            &marks,
        );
    }

    /// Generate the configured smart mask for the active layer.
    fn generate_smart_mask(&mut self) {
        let set = &self.texture_sets[self.active_set];
        let Some(curvature) = set.baked_curvature.as_ref() else {
            self.ui.status = "Bake mesh maps first (Baking & UVs → Bake mesh maps)".into();
            return;
        };
        // The mask shader reads the bakes texel for texel, so it is made at
        // the bakes' size, then fitted to this set's paint resolution.
        let bake_size = curvature.image().extent()[0];
        match crate::bakers::generate_mask(
            &self.gpu,
            set,
            self.ui.mask_kind,
            self.ui.mask_amount,
            self.ui.mask_contrast,
            bake_size,
        ) {
            Some(mask) => {
                let kind = self.ui.mask_kind;
                let layers = &mut self.texture_sets[self.active_set].layers;
                let mask = layers.fit_to_resolution(mask);
                let active = layers.active;
                layers.layers[active].mask = Some(mask);
                layers.layers[active].mask_kind = Some(kind);
                layers.mark_dirty();
                self.ui.status = format!("{} mask generated", kind.label());
            }
            None => {
                self.ui.status = "Mask generation failed: missing bakes".into();
            }
        }
    }

    /// Rebuild the texture-set list from the current models, preserving
    /// nothing — used after loading a new file.
    fn rebuild_texture_sets(&mut self) {
        self.texture_sets = set_keys(&self.models)
            .into_iter()
            .map(|key| {
                TextureSet::new_lazy(key.object, key.material, self.gpu.clone(), PAINT_RESOLUTION)
            })
            .collect();
        self.active_set = 0;
    }

    fn open_model_dialog(&mut self) {
        if self.import.is_some() {
            self.ui.status = "An import is already running".into();
            return;
        }
        let Some(path) = rfd::FileDialog::new()
            .add_filter("3D models", &["gltf", "glb", "fbx"])
            .add_filter("glTF", &["gltf", "glb"])
            .add_filter("FBX", &["fbx"])
            .pick_file()
        else {
            return;
        };
        self.start_model_import(path);
    }

    /// Parse a model on a worker; `poll_import` adopts it when done.
    fn export_to_zestengine(&mut self) {
        let Some(manifest) = self.zestengine.clone() else {
            self.ui.status = "No ZestEngine exchange open".into();
            return;
        };
        let maps = crate::layers::MapKind::ALL.to_vec();
        let contents = crate::export::ExportContents {
            flattened: true,
            shader_masks: false,
            layers: false,
            layer_masks: false,
            scope: crate::export::LayerScope::All,
            size: None,
        };
        let include = vec![true; self.texture_sets.len()];
        let dir = manifest.return_dir.clone();
        let written = match crate::export::export_all(
            &self.gpu,
            &mut self.texture_sets,
            &include,
            "zestengine",
            &maps,
            &contents,
            &dir,
        ) {
            Ok(w) => w,
            Err(e) => {
                self.ui.status = format!("Export to ZestEngine failed: {e:#}");
                return;
            }
        };
        let mut return_maps = Vec::new();
        for (slot, chunk) in written.chunks(maps.len()).enumerate() {
            for (kind, file) in maps.iter().zip(chunk) {
                if *kind == crate::layers::MapKind::Height {
                    continue;
                }
                return_maps.push(crate::zestengine::ReturnMap {
                    slot,
                    kind: zestengine_kind(*kind).to_owned(),
                    file: file.clone(),
                });
            }
        }
        let ret = crate::zestengine::ZestEngineReturn {
            object_id: manifest.object_id.clone(),
            maps: return_maps,
        };
        match crate::zestengine::write_return(&dir, &ret) {
            Ok(()) => {
                self.ui.status = format!("Exported {} maps to ZestEngine", ret.maps.len());
            }
            Err(e) => {
                self.ui.status = format!("Writing zestengine_return.json failed: {e:#}");
            }
        }
    }

    fn apply_zestengine_textures(&mut self) {
        let Some(manifest) = self.zestengine.clone() else { return };
        for map in &manifest.textures {
            let Some(kind) = kind_from_str(&map.kind) else { continue };
            let Some(set) = self.texture_sets.get_mut(map.slot) else { continue };
            let file = manifest.return_dir.join(&map.file);
            match crate::library::load_map(&file) {
                Ok(image) => {
                    let name = format!("ZestEngine · {}", map.kind);
                    set.layers.import_map_layer(name, kind, &image);
                }
                Err(e) => eprintln!("loading zestengine map {}: {e:#}", file.display()),
            }
        }
    }

    fn start_model_import(&mut self, path: std::path::PathBuf) {
        if self.import.is_some() {
            self.ui.status = "An import is already running".into();
            return;
        }

        let progress = Arc::new(std::sync::Mutex::new(crate::scene::ImportProgress {
            label: "Starting".into(),
            ..Default::default()
        }));
        let reporter = Arc::clone(&progress);
        let job_path = path.clone();
        let handle = std::thread::spawn(move || {
            let report = move |label: &str, done: usize, total: usize| {
                if let Ok(mut p) = reporter.lock() {
                    p.label = label.to_string();
                    p.done = done;
                    p.total = total;
                }
            };
            let is_fbx = job_path
                .extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("fbx"));
            if is_fbx {
                fbx_loader::load_with_progress(&job_path, &report)
            } else {
                gltf_loader::load_with_progress(&job_path, &report)
            }
        });

        self.ui.status = format!(
            "Importing {}…",
            path.file_name().unwrap_or_default().to_string_lossy()
        );
        self.import = Some(ModelImport { path, progress, handle: Some(handle) });
    }

    /// Adopt a finished import. Called every frame; cheap until the worker is done.
    fn poll_import(&mut self) {
        let Some(job) = self.import.as_mut() else { return };
        if !job.handle.as_ref().is_some_and(|h| h.is_finished()) {
            return;
        }
        let Some(handle) = job.handle.take() else { return };
        let path = job.path.clone();
        self.import = None;

        let result = match handle.join() {
            Ok(r) => r,
            Err(_) => Err(anyhow::anyhow!("import thread panicked")),
        };
        match result {
            Ok(models) => {
                // Multi-tile UDIM UVs become one texture set per tile.
                let models = crate::scene::split_udim_tiles(models);
                let (center, radius) = scene_bounds(&models);
                let draws = models.len();
                self.renderer.set_models(&models);
                self.mesh_groups = None;
                self.paint_projection = None;
                self.models = models;
                self.reset_pose();
                // Old bakes/paint don't apply to a new mesh.
                self.rebuild_texture_sets();
                self.apply_zestengine_textures();
                self.selection.clear();
                self.isolate_backup = None;
                self.isolate_camera = None;
                self.pt_scene_stale = true;
                self.camera.frame(center, radius);
                let name = path.file_name().unwrap_or_default().to_string_lossy();
                self.ui.status = format!(
                    "Loaded {name} ({draws} draw calls, {} texture sets)",
                    self.texture_sets.len()
                );
                self.ui.default_scene = false;
                self.ui.model_name = path.file_stem().unwrap_or_default().to_string_lossy().to_string();
                // A new document, not the project that was open.
                self.project_path = None;
                let hidden = self.restore_hidden_pieces(&path);
                if hidden > 0 {
                    self.ui.status.push_str(&format!(" · {hidden} hidden as last time (Alt+H shows all)"));
                }
                self.ui.export_name.clear();
                // A new document: the old history doesn't apply to it.
                self.reset_history();
                self.push_toast(widgets::Tone::Ok, self.ui.status.clone(), None);
                self.answer_assistant_import(Ok(self.ui.status.clone()));
            }
            Err(e) => {
                self.ui.status = "Load failed".into();
                self.push_toast(
                    widgets::Tone::Error,
                    format!("Load failed: {}", path.file_name().unwrap_or_default().to_string_lossy()),
                    Some(overlays::ToastAction::Details(format!("{e:#}"))),
                );
                self.answer_assistant_import(Err(format!("Load failed: {e:#}")));
            }
        }
    }

    /// Sliders for the active material's OpenPBR-style surface parameters.
    fn surface_params_ui(&mut self, ui: &mut egui::Ui) {
        let set = &self.texture_sets[self.active_set];
        let (object, material) = (set.object.clone(), set.material.clone());
        let Some(first) = self
            .models
            .iter()
            .position(|m| m.name == object && m.material_name == material)
        else {
            return;
        };

        let mut mat = {
            let m = &self.models[first].material;
            (
                m.emission,
                m.specular_weight,
                m.coat_weight,
                m.coat_roughness,
                m.sheen_weight,
                m.sheen_roughness,
            )
        };
        let mut alpha = {
            let m = &self.models[first].material;
            (m.opacity, m.transparent)
        };
        let mut changed = false;
        // OpenPBR-style surface slots (a pragmatic subset, not certified).
        changed |= ui
            .checkbox(&mut alpha.1, "Transparent (alpha blend)")
            .on_hover_text("Sorted blended pass — for sheer fabric, glass, lashes")
            .changed();
        changed |= widgets::SliderRow::new("Opacity", &mut alpha.0, 0.0..=1.0).show(ui).changed();
        ui.horizontal(|ui| {
            widgets::fixed(ui, crate::theme::SLIDER_LABEL_WIDTH, |ui| ui.label("Emission"));
            changed |= ui.color_edit_button_rgb(&mut mat.0).changed();
        });
        changed |= widgets::SliderRow::new("Specular", &mut mat.1, 0.0..=1.0).show(ui).changed();
        changed |= widgets::SliderRow::new("Coat", &mut mat.2, 0.0..=1.0).show(ui).changed();
        changed |= widgets::SliderRow::new("Coat rough", &mut mat.3, 0.03..=1.0).hint("Coat roughness").show(ui).changed();
        changed |= widgets::SliderRow::new("Sheen", &mut mat.4, 0.0..=1.0).show(ui).changed();
        changed |= widgets::SliderRow::new("Sheen rough", &mut mat.5, 0.05..=1.0).hint("Sheen roughness").show(ui).changed();

        if changed {
            for model in self
                .models
                .iter_mut()
                .filter(|m| m.name == object && m.material_name == material)
            {
                let m = &mut model.material;
                m.emission = mat.0;
                m.specular_weight = mat.1;
                m.coat_weight = mat.2;
                m.coat_roughness = mat.3;
                m.sheen_weight = mat.4;
                m.sheen_roughness = mat.5;
                m.opacity = alpha.0;
                m.transparent = alpha.1;
            }
            let params = self.models[first].material.surface_params();
            self.renderer.set_material_surface(self.active_set, params);
        }
    }
    /// Import one or more brush files. Returns whether the library changed.
    fn import_brush_files(&mut self) -> bool {
        let Some(paths) = rfd::FileDialog::new()
            .add_filter("Brush sets & alphas", crate::brushfile::SUPPORTED)
            .pick_files()
        else {
            return false;
        };
        self.import_brush_paths(&paths)
    }

    /// Import every brush file directly inside a chosen folder.
    fn import_brush_folder(&mut self) -> bool {
        let Some(dir) = rfd::FileDialog::new().pick_folder() else {
            return false;
        };
        let paths = crate::brushfile::files_in(&dir);
        if paths.is_empty() {
            self.ui.brush_import_status = "No brush files in that folder".into();
            return false;
        }
        self.import_brush_paths(&paths)
    }

    fn import_brush_paths(&mut self, paths: &[std::path::PathBuf]) -> bool {
        let mut stamps = 0usize;
        let mut failed = Vec::new();
        for path in paths {
            match crate::brushfile::import(path) {
                Ok(written) => stamps += written.len(),
                Err(e) => {
                    let name =
                        path.file_name().unwrap_or_default().to_string_lossy().to_string();
                    eprintln!("[import] {name}: {e:#}");
                    failed.push(name);
                }
            }
        }
        self.ui.brush_import_status = if failed.is_empty() {
            format!("Imported {stamps} stamps from {} file(s)", paths.len())
        } else {
            format!("Imported {stamps}; {} failed (see console)", failed.len())
        };
        self.ui.status = self.ui.brush_import_status.clone();
        stamps > 0
    }

    /// Load a library stamp as the active brush tip.
    fn use_library_brush(&mut self, index: usize) {
        let Some(brush) = self.ui.brushes.get(index) else { return };
        let Some(path) = brush.tip_source().cloned() else {
            self.ui.status = "That brush has no usable map".into();
            return;
        };
        let label = brush.display.clone();
        self.apply_brush_tip(&path, label);
    }

    fn apply_library_material(&mut self, index: usize) {
        let maps: Vec<(MapSlot, std::path::PathBuf)> = self.ui.library[index]
            .maps
            .iter()
            .map(|(s, p)| (*s, p.clone()))
            .collect();
        let name = self.ui.library[index].name.clone();

        let mut images: Vec<(MapSlot, ImageData)> = Vec::new();
        for (slot, path) in maps {
            match crate::library::load_map(&path) {
                Ok(image) => images.push((slot, image)),
                Err(e) => {
                    self.ui.status = format!("Library map failed: {e:#}");
                    return;
                }
            }
        }
        if images.is_empty() {
            self.ui.status = format!("'{name}' has no usable maps");
            return;
        }

        let layers = &mut self.texture_sets[self.active_set].layers;
        layers.add_layer();
        let target = layers.active;
        layers.layers[target].name = name.clone();
        let written = layers.fill_layer_from_material(target, &images);
        layers.add_mask(target, 0.0);

        // Painting the mask in is the point, so switch the brush over to it.
        self.brush.target = crate::paint::PaintTarget::Mask;
        self.brush.mask_value = 1.0;

        let skipped = images.len() - written.len();
        self.ui.status = format!(
            "'{name}' filled into a new layer ({} maps{}) — paint the mask to reveal it",
            written.len(),
            if skipped > 0 { ", AO skipped" } else { "" },
        );
    }

    fn save_material_to_library(&mut self) {
        let name = self.ui.library_save_name.trim().to_string();
        if name.is_empty() {
            self.ui.status = "Give the material a name first".into();
            return;
        }
        let set = &self.texture_sets[self.active_set];
        let (object, material) = (set.object.clone(), set.material.clone());
        let Some(model) = self
            .models
            .iter()
            .find(|m| m.name == object && m.material_name == material)
        else {
            return;
        };
        let m = &model.material;
        let maps: Vec<(MapSlot, &ImageData)> = [
            (MapSlot::BaseColor, m.base_color_image.as_ref()),
            (MapSlot::Roughness, m.roughness_image.as_ref()),
            (MapSlot::Metallic, m.metallic_image.as_ref()),
            (MapSlot::Normal, m.normal_image.as_ref()),
            (MapSlot::Height, m.height_image.as_ref()),
            (MapSlot::Ao, m.ao_image.as_ref()),
            (MapSlot::Emission, m.emission_image.as_ref()),
            (MapSlot::EmissionMask, m.emission_mask_image.as_ref()),
        ]
        .into_iter()
        .filter_map(|(s, img)| img.map(|i| (s, i)))
        .collect();
        if maps.is_empty() {
            self.ui.status = "Active material has no maps to save".into();
            return;
        }
        match crate::library::save_material(&name, &maps) {
            Ok(dir) => {
                self.ui.status = format!("Saved {} maps to {}", maps.len(), dir.display());
                self.ui.library = crate::library::scan();
            }
            Err(e) => self.ui.status = format!("Save failed: {e:#}"),
        }
    }

    /// Load a grayscale image as the brush tip stamp.
    fn load_brush_tip(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Images", &["png", "jpg", "jpeg", "tga", "bmp", "webp"])
            .pick_file()
        else {
            return;
        };
        let label = path.file_name().unwrap_or_default().to_string_lossy().to_string();
        self.apply_brush_tip(&path, label);
    }

    /// Upload an image as the brush tip: RGB keeps the stamp's own colours,
    /// alpha carries coverage. A stamp with an alpha channel uses it; a fully
    /// opaque one falls back to luminance, which is how greyscale alphas are
    /// authored.
    fn apply_brush_tip(&mut self, path: &std::path::Path, label: String) {
        match image::open(path) {
            Ok(img) => {
                let src = img.to_rgba8();
                self.set_tip_image(&src, label);
            }
            Err(e) => self.ui.status = format!("Tip load failed: {e}"),
        }
    }

    /// Make an RGBA image the brush tip (shared by tip files, library
    /// stamps and the built-in presets).
    fn set_tip_image(&mut self, src: &image::RgbaImage, label: String) {
        let (w, h) = src.dimensions();
        let has_alpha = src.pixels().any(|p| p.0[3] != 255);
        // "Colourful" means the channels actually differ somewhere;
        // a greyscale stamp should not switch the mode on by itself.
        let has_color = src.pixels().any(|p| {
            let (r, g, b) = (p.0[0] as i32, p.0[1] as i32, p.0[2] as i32);
            (r - g).abs() > 8 || (g - b).abs() > 8 || (r - b).abs() > 8
        });
        let rgba: Vec<u8> = src
            .pixels()
            .flat_map(|p| {
                let coverage = if has_alpha {
                    p.0[3]
                } else {
                    ((p.0[0] as u32 * 299 + p.0[1] as u32 * 587 + p.0[2] as u32 * 114)
                        / 1000) as u8
                };
                [p.0[0], p.0[1], p.0[2], coverage]
            })
            .collect();
        // Premultiplied on the GPU, so filtering and mip levels never bleed
        // the colour hidden under transparent pixels into the edges (pale or
        // dark halos round every splat). The shader divides it back out.
        let premultiplied: Vec<u8> = rgba
            .chunks_exact(4)
            .flat_map(|p| {
                let a = p[3] as u32;
                [(p[0] as u32 * a / 255) as u8, (p[1] as u32 * a / 255) as u8, (p[2] as u32 * a / 255) as u8, p[3]]
            })
            .collect();
        let view = crate::renderer::upload_texture_mipmapped(
            &self.gpu.queue,
            &self.gpu.memory_allocator,
            &self.gpu.command_buffer_allocator,
            w,
            h,
            &premultiplied,
            vulkano::format::Format::R8G8B8A8_UNORM,
        );
        self.brush.tip = Some(view);
        self.brush.tip_name = Some(label.clone());
        self.brush.tip_aspect = w as f32 / h.max(1) as f32;
        // Small copy for the stamp cursor; the GPU tip keeps full size.
        let preview = image::RgbaImage::from_raw(w, h, rgba.clone())
            .map(|img| {
                let scale = (256.0 / w.max(h) as f32).min(1.0);
                image::imageops::resize(
                    &img,
                    ((w as f32 * scale) as u32).max(1),
                    ((h as f32 * scale) as u32).max(1),
                    image::imageops::FilterType::Triangle,
                )
            });
        self.tip_preview_image = preview.map(|img| {
            egui::ColorImage::from_rgba_unmultiplied(
                [img.width() as usize, img.height() as usize],
                img.as_raw(),
            )
        });
        self.tip_preview = None;
        self.brush.tip_has_color = has_color;
        // A colourful stamp is almost always wanted for its colours.
        self.brush.use_tip_color = has_color;
        self.ui.status = format!(
            "Brush tip '{label}' loaded ({w}x{h}{})",
            if has_color { ", colour" } else { "" }
        );
    }

    /// Outline thickness in world units, scaled to the scene so it reads the
    /// same on a ring and on a whole body.
    /// World-space rim width for the selection outline: scaled by camera
    /// distance so it reads the same zoomed in on a shoe or out on the body.
    /// Selection outline thickness in viewport pixels.
    fn outline_width(&self) -> f32 {
        3.0
    }

    /// World direction the key light shines from.
    fn key_light_direction(&self) -> Vec3 {
        if self.ui.key_follow_camera {
            // A headlight: pinned to the eye, so orbiting sweeps the highlight
            // across the surface.
            let dir = self.camera.position() - self.camera.target;
            return dir.normalize_or_zero();
        }
        let az = self.ui.key_azimuth.to_radians();
        let el = self.ui.key_elevation.to_radians();
        Vec3::new(el.cos() * az.sin(), el.sin(), el.cos() * az.cos()).normalize_or_zero()
    }

    /// Which texture set a model belongs to.
    fn set_index_of(&self, model: &ModelData) -> Option<usize> {
        self.texture_sets
            .iter()
            .position(|s| s.object == model.name && s.material == model.material_name)
    }

    /// Nearest visible texture set under a viewport pixel.
    fn pick_set(&self, px: Vec2) -> Option<usize> {
        self.pick_surface(px).map(|hit| hit.set)
    }

    /// The nearest visible surface under a viewport pixel: which texture set,
    /// and the interpolated UV there.
    fn pick_surface(&self, px: Vec2) -> Option<SurfaceHit> {
        let aspect = self.viewport.size[0] as f32 / self.viewport.size[1] as f32;
        let view_proj = self.camera.view_proj(aspect);
        let inverse = view_proj.inverse();

        let ndc = Vec2::new(
            px.x / self.viewport.size[0].max(1) as f32 * 2.0 - 1.0,
            px.y / self.viewport.size[1].max(1) as f32 * 2.0 - 1.0,
        );
        let unproject = |depth: f32| {
            let p = inverse * glam::Vec4::new(ndc.x, ndc.y, depth, 1.0);
            if p.w.abs() < 1e-8 { Vec3::ZERO } else { p.truncate() / p.w }
        };
        // Vulkan depth runs 0 (near) to 1 (far).
        let near = unproject(0.0);
        let ray = unproject(1.0) - near;

        let mut best: Option<(f32, SurfaceHit)> = None;
        for (model_index, model) in self.models.iter().enumerate() {
            let Some(set) = self.set_index_of(model) else { continue };
            if !self.texture_sets[set].visible {
                continue;
            }
            // Cheaper to move the ray into model space than the mesh into world.
            let inv_model = model.transform.inverse();
            let local_origin = inv_model.transform_point3(near);
            let local_dir = inv_model.transform_vector3(ray);

            for tri in model.mesh.indices.chunks_exact(3) {
                let (Some(a), Some(b), Some(c)) = (
                    model.mesh.vertices.get(tri[0] as usize),
                    model.mesh.vertices.get(tri[1] as usize),
                    model.mesh.vertices.get(tri[2] as usize),
                ) else {
                    continue;
                };
                if let Some((t, u, v)) = ray_hit_barycentric(
                    local_origin,
                    local_dir,
                    Vec3::from(a.position),
                    Vec3::from(b.position),
                    Vec3::from(c.position),
                ) && best.as_ref().is_none_or(|(bt, _)| t < *bt)
                {
                    let uv = Vec2::from(a.uv) * (1.0 - u - v)
                        + Vec2::from(b.uv) * u
                        + Vec2::from(c.uv) * v;
                    let (pa, pb, pc) =
                        (Vec3::from(a.position), Vec3::from(b.position), Vec3::from(c.position));
                    // The local ray is the world ray mapped through an affine
                    // transform, so the same t lands on the world hit point.
                    let world = near + ray * t;
                    let normal = (inv_model.transpose().transform_vector3((pb - pa).cross(pc - pa)))
                        .normalize_or_zero();
                    best = Some((t, SurfaceHit { set, model: model_index, uv, world, normal }));
                }
            }
        }
        best.map(|(_, hit)| hit)
    }

    /// `pick_surface`, but only when it lands on the set being painted.
    fn pick_active_surface(&self, px: Vec2) -> Option<SurfaceHit> {
        self.pick_surface(px).filter(|hit| hit.set == self.active_set)
    }

    /// Where a world point is in the viewport, in px; None behind the camera.
    fn world_to_viewport(&self, p: Vec3) -> Option<Vec2> {
        let aspect = self.viewport.size[0] as f32 / self.viewport.size[1] as f32;
        let size = Vec2::new(self.viewport.size[0] as f32, self.viewport.size[1] as f32);
        project_to_viewport(self.camera.view_proj(aspect), size, p)
    }

    /// Where the clone source is on screen for a brush at `brush_px`, under
    /// the current camera. Aligned with an offset: the brush's point on the
    /// model plus the offset; otherwise the anchor itself.
    fn clone_source_px(&self, brush_px: Vec2) -> Option<Vec2> {
        let anchor = self.clone_source?;
        let world = match (self.ui.clone_aligned, self.clone_offset) {
            (true, Some(offset)) => self.pick_active_surface(brush_px)?.world + offset,
            _ => anchor.world,
        };
        self.world_to_viewport(world)
    }

    /// Ctrl+click with the clone tool: pin the source to the surface there.
    fn set_clone_source(&mut self, px: Vec2) {
        match self.pick_surface(px) {
            Some(hit) if hit.set == self.active_set => {
                self.clone_source = Some(CloneAnchor { world: hit.world, normal: hit.normal });
                self.clone_offset = None;
                self.clone_stroke_offset = None;
                self.ui.status = "Clone source pinned to the model — now paint".into();
            }
            Some(hit) => {
                self.ui.status = format!(
                    "That's {} — the source has to be on {}, the material being painted",
                    self.texture_sets[hit.set].label(),
                    self.texture_sets[self.active_set].label()
                );
            }
            None => self.ui.status = "Ctrl+click on the model to set a clone source".into(),
        }
    }

    /// Eyedropper: take the base colour you see at `px` — the material's own
    /// colour with the painted stack composited over it, unlit — into the
    /// brush colour.
    fn pick_color_at(&mut self, px: Vec2) {
        let Some(hit) = self.pick_surface(px) else {
            self.ui.status = "Nothing under the cursor to pick".into();
            return;
        };
        let material = &self.models[hit.model].material;
        let mut albedo = Vec3::new(
            material.base_color[0],
            material.base_color[1],
            material.base_color[2],
        );
        if let Some(image) = &material.base_color_image {
            albedo *= sample_srgb_image(image, hit.uv);
        }
        let layers = &self.texture_sets[hit.set].layers;
        let res = layers.resolution as f32;
        let texel = (hit.uv - hit.uv.floor()) * res;
        let paint = layers.read_composite_basecolor(texel.x as u32, texel.y as u32);
        // Same "over" the viewport shader does with the premultiplied paint.
        let linear = albedo * (1.0 - paint[3]) + Vec3::new(paint[0], paint[1], paint[2]);
        self.brush.color = linear.to_array().map(linear_component_to_srgb);
        let [r, g, b] = self.brush.color.map(|c| (c.clamp(0.0, 1.0) * 255.0).round() as u8);
        self.ui.status = format!("Picked #{r:02X}{g:02X}{b:02X}");
    }

    /// Pick the piece under a viewport pixel. `extend` (Shift/Ctrl) toggles
    /// it in the selection; otherwise it replaces the selection, and a click
    /// on empty space clears it, as in Blender.
    fn select_at(&mut self, px: Vec2, extend: bool) {
        match self.pick_set(px) {
            Some(set) => {
                self.toggle_selected(set, extend);
                self.ui.scroll_to_set = Some(set);
                self.ui.status = if self.selection.contains(&set) {
                    format!("Selected {}{}", self.texture_sets[set].label(), self.selection_suffix())
                } else {
                    format!("Deselected {}", self.texture_sets[set].label())
                };
            }
            None if !extend => {
                self.selection.clear();
                self.ui.status = "Nothing under the cursor".into();
            }
            None => {}
        }
    }

    /// Select `set` (alone, or toggled into the selection with `extend`).
    /// A newly selected set also becomes the paint target, which is nearly
    /// always what you wanted next.
    fn toggle_selected(&mut self, set: usize, extend: bool) {
        if !extend {
            self.selection = vec![set];
        } else if let Some(at) = self.selection.iter().position(|&s| s == set) {
            self.selection.remove(at);
            return;
        } else {
            self.selection.push(set);
        }
        self.set_active_set(set);
    }

    fn selection_suffix(&self) -> String {
        match self.selection.len() {
            0 | 1 => String::new(),
            n => format!(" ({n} selected)"),
        }
    }

    /// What the visibility and frame keys act on: the selection, or the
    /// active set when nothing is selected.
    fn selection_or_active(&self) -> Vec<usize> {
        if self.selection.is_empty() { vec![self.active_set] } else { self.selection.clone() }
    }

    fn select_all(&mut self) {
        self.selection = (0..self.texture_sets.len()).filter(|&i| self.texture_sets[i].visible).collect();
        self.ui.status = format!("Selected {} pieces", self.selection.len());
    }

    fn deselect_all(&mut self) {
        self.selection.clear();
        self.ui.status = "Deselected all".into();
    }

    /// Blender's visibility keys, for the selection (or the active set).
    fn hide_selected(&mut self) {
        let sets = self.selection_or_active();
        for &set in &sets {
            self.texture_sets[set].visible = false;
        }
        self.selection.retain(|s| !sets.contains(s));
        self.pt_scene_stale = true;
        self.ui.status = match sets.as_slice() {
            [one] => format!("Hid {}", self.texture_sets[*one].label()),
            many => format!("Hid {} pieces", many.len()),
        };
    }

    fn hide_unselected(&mut self) {
        let keep = self.selection_or_active();
        for (i, set) in self.texture_sets.iter_mut().enumerate() {
            set.visible = keep.contains(&i);
        }
        self.pt_scene_stale = true;
        self.ui.status = "Hid everything else".into();
    }

    fn unhide_all(&mut self) {
        for set in &mut self.texture_sets {
            set.visible = true;
        }
        self.isolate_backup = None;
        self.isolate_camera = None;
        self.pt_scene_stale = true;
        self.ui.status = "Revealed everything".into();
    }

    /// Delete pieces (texture sets) and the mesh parts drawn with them, paint
    /// and all. Undoable: the step keeps the removed sets (composites
    /// released, layers kept) until it leaves the history.
    fn delete_pieces(&mut self, sets: &[usize]) {
        let mut sets: Vec<usize> = sets.iter().copied().filter(|&s| s < self.texture_sets.len()).collect();
        sets.sort_unstable();
        sets.dedup();
        if sets.is_empty() {
            return;
        }
        if sets.len() == self.texture_sets.len() {
            self.ui.status = "Can't delete every piece — open another model instead".into();
            return;
        }
        // Anything pending is its own step, before the delete's.
        self.track_history(false);
        let label = match sets.as_slice() {
            [one] => self.texture_sets[*one].label(),
            many => format!("{} pieces", many.len()),
        };
        let models_before = self.models.clone();
        let doomed: Vec<(String, String)> =
            sets.iter().map(|&i| (self.texture_sets[i].object.clone(), self.texture_sets[i].material.clone())).collect();
        self.models.retain(|m| !doomed.iter().any(|(object, material)| *object == m.name && *material == m.material_name));
        let active = self.active_key();
        let removed = self.remove_sets(&sets);
        // Surviving indices shift down past the removed ones.
        let shift = |i: usize| i - sets.iter().filter(|&&s| s < i).count();
        self.selection = self.selection.iter().filter(|i| !sets.contains(i)).map(|&i| shift(i)).collect();
        if let Some(backup) = &mut self.isolate_backup {
            for &i in sets.iter().rev() {
                if i < backup.len() {
                    backup.remove(i);
                }
            }
        }
        self.pieces_changed(active);
        self.record_delete(models_before, removed);
        self.settle_history();
        self.ui.status = format!("Deleted {label} — Ctrl+Z brings it back");
    }

    /// Toggle local view: show only the selection (or the active set) and
    /// frame it; the second press restores what was visible and the camera.
    fn toggle_isolate(&mut self) {
        let leaving = self.isolate_backup.is_some();
        let sets = self.selection_or_active();
        let mut flags: Vec<bool> = self.texture_sets.iter().map(|s| s.visible).collect();
        let backup = self.isolate_backup.take();
        self.isolate_backup = toggle_isolation(&mut flags, &sets, backup);
        for (set, flag) in self.texture_sets.iter_mut().zip(flags) {
            set.visible = flag;
        }
        self.pt_scene_stale = true;
        if leaving {
            if let Some(camera) = self.isolate_camera.take() {
                self.camera = camera;
            }
            self.ui.status = "Left local view".into();
        } else {
            // Paint lands on the active set, so it must be one you can see.
            if !sets.contains(&self.active_set) {
                self.set_active_set(sets[0]);
            }
            self.isolate_camera = Some(self.camera.clone());
            self.frame_sets(&sets);
            self.ui.status = match sets.as_slice() {
                [one] => format!("Local view: {} — press / to leave", self.texture_sets[*one].label()),
                many => format!("Local view: {} pieces — press / to leave", many.len()),
            };
        }
    }

    /// The GPU's dedicated memory, for sizing warnings.
    fn vram_bytes(&self) -> u64 {
        use vulkano::memory::MemoryHeapFlags;
        let props = self.gpu.queue.device().physical_device().memory_properties();
        props.memory_heaps.iter().filter(|h| h.flags.intersects(MemoryHeapFlags::DEVICE_LOCAL)).map(|h| h.size).max().unwrap_or(0)
    }

    /// Estimated paint-texture memory of the whole document: every
    /// allocated set plus the old sets undo keeps.
    fn paint_vram_bytes(&self) -> u64 {
        self.texture_sets.iter().map(history::set_vram_bytes).sum::<u64>() + self.history_vram_bytes()
    }

    /// Estimated VRAM of `set` at `resolution`, and whether the document
    /// would then be too big: past 75% of the card (the rest is the desktop
    /// and other apps) the driver starts swapping and the whole PC stalls.
    /// The set's current textures count too — undo keeps them.
    fn resolution_cost(&self, set: usize, resolution: u32) -> (u64, bool) {
        let bytes = self.texture_sets[set].layers.bytes_at(resolution);
        let current = self.texture_sets[set].layers.resolution;
        let after = self.paint_vram_bytes() + if resolution == current { 0 } else { bytes };
        let vram = self.vram_bytes();
        (bytes, vram > 0 && resolution != current && after > vram / 4 * 3)
    }

    /// Repaint `set` at another resolution: its layers are resampled (so
    /// existing paint carries over, softer when enlarged) and the old set
    /// goes into undo history whole.
    fn set_texture_resolution(&mut self, set: usize, resolution: u32) -> anyhow::Result<()> {
        anyhow::ensure!(RESOLUTIONS.contains(&resolution), "resolution must be one of {RESOLUTIONS:?}");
        let Some(old) = self.texture_sets.get(set) else { anyhow::bail!("no texture set {set}") };
        if old.layers.resolution == resolution {
            return Ok(());
        }
        let (bytes, too_big) = self.resolution_cost(set, resolution);
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        anyhow::ensure!(
            !too_big,
            "{resolution}px for '{}' needs about {:.1} GiB more, and the document already holds about {:.1} GiB of paint \
             textures (including {:.1} GiB that undo keeps from earlier resolution changes) on a {:.1} GiB GPU — too close to \
             freezing the desktop. Try a smaller size, lower other sets, or merge/delete layers.",
            old.label(),
            gib(bytes),
            gib(self.paint_vram_bytes()),
            gib(self.history_vram_bytes()),
            gib(self.vram_bytes())
        );
        self.track_history(false);
        let old = &self.texture_sets[set];
        let fresh = TextureSet {
            object: old.object.clone(),
            material: old.material.clone(),
            layers: old.layers.resampled(resolution),
            baked_ao: old.baked_ao.clone(),
            baked_position: old.baked_position.clone(),
            baked_normal: old.baked_normal.clone(),
            baked_curvature: old.baked_curvature.clone(),
            visible: old.visible,
            seeded: old.seeded,
            material_in_layers: old.material_in_layers,
        };
        let label = fresh.label();
        let old_set = std::mem::replace(&mut self.texture_sets[set], fresh);
        self.record_resize(set, old_set);
        self.settle_history();
        self.pt_scene_stale = true;
        self.ui.status = format!("'{label}' now paints at {resolution}px (≈{:.1} GiB VRAM) — Ctrl+Z to go back", gib(bytes));
        Ok(())
    }

    /// Point the camera at some texture sets' geometry.
    fn frame_sets(&mut self, sets: &[usize]) {
        let indices = crate::scene::set_indices(&self.models);
        let models = self.models.iter().zip(indices).filter(|(_, i)| sets.contains(i)).map(|(m, _)| m);
        let (min, max) = world_box(models);
        if min.is_finite() {
            self.camera.frame((min + max) * 0.5, ((max - min).length() * 0.5).max(1e-3));
        }
    }

    fn frame_selection(&mut self) {
        let sets = self.selection_or_active();
        self.frame_sets(&sets);
    }

    fn frame_all(&mut self) {
        let visible: Vec<usize> = (0..self.texture_sets.len()).filter(|&i| self.texture_sets[i].visible).collect();
        let sets = if visible.is_empty() { (0..self.texture_sets.len()).collect() } else { visible };
        self.frame_sets(&sets);
    }

    /// Global primitive index of the nearest triangle in the active set under a
    /// viewport pixel, using the same offsets `paint.rs` and `MeshGroups` use.
    ///
    /// Screen-space overlap alone is not enough for the fill modes: a piece
    /// behind the one you clicked is also front-facing and also under the
    /// cursor, so it would flood too. A depth-ordered ray settles it.
    fn pick_primitive(&self, px: Vec2) -> Option<u32> {
        let set = &self.texture_sets[self.active_set];
        let (object, material) = (set.object.clone(), set.material.clone());

        let aspect = self.viewport.size[0] as f32 / self.viewport.size[1] as f32;
        let inverse = self.camera.view_proj(aspect).inverse();
        let ndc = Vec2::new(
            px.x / self.viewport.size[0].max(1) as f32 * 2.0 - 1.0,
            px.y / self.viewport.size[1].max(1) as f32 * 2.0 - 1.0,
        );
        let unproject = |depth: f32| {
            let p = inverse * glam::Vec4::new(ndc.x, ndc.y, depth, 1.0);
            if p.w.abs() < 1e-8 { Vec3::ZERO } else { p.truncate() / p.w }
        };
        let near = unproject(0.0);
        let ray = unproject(1.0) - near;

        let mut best: Option<(f32, u32)> = None;
        let mut base = 0u32;
        for model in self
            .models
            .iter()
            .filter(|m| m.name == object && m.material_name == material)
        {
            let inv_model = model.transform.inverse();
            let origin = inv_model.transform_point3(near);
            let dir = inv_model.transform_vector3(ray);
            for (local, tri) in model.mesh.indices.chunks_exact(3).enumerate() {
                let (Some(a), Some(b), Some(c)) = (
                    model.mesh.vertices.get(tri[0] as usize),
                    model.mesh.vertices.get(tri[1] as usize),
                    model.mesh.vertices.get(tri[2] as usize),
                ) else {
                    continue;
                };
                if let Some(t) = ray_hits_triangle(
                    origin,
                    dir,
                    Vec3::from(a.position),
                    Vec3::from(b.position),
                    Vec3::from(c.position),
                ) && best.is_none_or(|(bt, _)| t < bt)
                {
                    best = Some((t, base + local as u32));
                }
            }
            base += (model.mesh.indices.len() / 3) as u32;
        }
        best.map(|(_, prim)| prim)
    }

    /// Group the active set's triangles into shells and UV islands.
    fn ensure_mesh_groups(&mut self) {
        if self.mesh_groups.as_ref().is_some_and(|g| g.set == self.active_set) {
            return;
        }
        let set = &self.texture_sets[self.active_set];
        let (object, material) = (set.object.clone(), set.material.clone());

        let mut shell = Vec::new();
        let mut island = Vec::new();
        let mut shell_base = 0u32;
        let mut island_base = 0u32;
        for model in self
            .models
            .iter()
            .filter(|m| m.name == object && m.material_name == material)
        {
            // Separate models are separate pieces by definition, so their ids
            // are offset rather than restarted.
            let s = crate::connectivity::shells(&model.mesh);
            let i = crate::connectivity::uv_islands(&model.mesh);
            shell.extend(s.of_triangle.iter().map(|g| g + shell_base));
            island.extend(i.of_triangle.iter().map(|g| g + island_base));
            shell_base += s.count as u32;
            island_base += i.count as u32;
        }

        let shell_groups =
            crate::connectivity::Groups { of_triangle: shell, count: shell_base as usize };
        let shell_sizes = shell_groups.sizes();

        self.mesh_groups = Some(MeshGroups {
            set: self.active_set,
            shell: shell_groups.of_triangle,
            island,
            shell_count: shell_base as usize,
            island_count: island_base as usize,
            shell_sizes,
        });
    }

    /// Project the active set's triangles to screen space, once per camera pose.
    fn ensure_paint_projection(&mut self, view: &PaintView) {
        let symmetry_x = self.brush.symmetry_x;
        let fresh = self.paint_projection.as_ref().is_some_and(|p| {
            p.set == self.active_set
                && p.view_proj == view.view_proj
                && p.viewport == view.viewport_size
                && p.symmetry_x == symmetry_x
        });
        if fresh {
            return;
        }

        let set = &self.texture_sets[self.active_set];
        let (object, material) = (set.object.clone(), set.material.clone());
        let mut tris = Vec::new();
        let mut degenerate = false;

        let mirror = Mat4::from_scale(Vec3::new(-1.0, 1.0, 1.0));
        let passes: &[Mat4] =
            if symmetry_x { &[Mat4::IDENTITY, mirror] } else { &[Mat4::IDENTITY] };

        for (model, pass) in self
            .models
            .iter()
            .filter(|m| m.name == object && m.material_name == material)
            .flat_map(|m| passes.iter().map(move |p| (m, p)))
        {
            let mvp = view.view_proj * *pass * model.transform;
            let screen: Vec<Vec2> = model
                .mesh
                .vertices
                .iter()
                .map(|v| {
                    let p = mvp
                        * glam::Vec4::new(v.position[0], v.position[1], v.position[2], 1.0);
                    if p.w <= 1e-6 {
                        degenerate = true;
                        Vec2::ZERO
                    } else {
                        let ndc = Vec2::new(p.x / p.w, p.y / p.w);
                        (ndc * 0.5 + Vec2::splat(0.5)) * view.viewport_size
                    }
                })
                .collect();

            for tri in model.mesh.indices.chunks_exact(3) {
                let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
                let (Some(&sa), Some(&sb), Some(&sc)) =
                    (screen.get(a), screen.get(b), screen.get(c))
                else {
                    continue;
                };
                let uva = Vec2::from(model.mesh.vertices[a].uv);
                let uvb = Vec2::from(model.mesh.vertices[b].uv);
                let uvc = Vec2::from(model.mesh.vertices[c].uv);
                tris.push(ProjectedTri {
                    screen_lo: sa.min(sb).min(sc),
                    screen_hi: sa.max(sb).max(sc),
                    uv_lo: uva.min(uvb).min(uvc),
                    uv_hi: uva.max(uvb).max(uvc),
                });
            }
        }

        self.paint_projection = Some(PaintProjection {
            view_proj: view.view_proj,
            set: self.active_set,
            viewport: view.viewport_size,
            symmetry_x,
            tris,
            degenerate,
        });
    }

    fn stroke_region(&self, stamps: &[Vec2], radius: f32, resolution: u32) -> PaintRegion {
        match self.paint_projection.as_ref() {
            Some(proj) => region_from_projection(proj, stamps, radius, resolution),
            None => PaintRegion::Everything,
        }
    }

    /// Size description of the active material's map in a given slot.
    fn active_material_map_info(&self, slot: MapSlot) -> Option<String> {
        let set = &self.texture_sets[self.active_set];
        let model = self
            .models
            .iter()
            .find(|m| m.name == set.object && m.material_name == set.material)?;
        let img = match slot {
            MapSlot::BaseColor => model.material.base_color_image.as_ref(),
            MapSlot::Roughness => model.material.roughness_image.as_ref(),
            MapSlot::Metallic => model.material.metallic_image.as_ref(),
            MapSlot::Normal => model.material.normal_image.as_ref(),
            MapSlot::Height => model.material.height_image.as_ref(),
            MapSlot::Ao => model.material.ao_image.as_ref(),
            MapSlot::Emission => model.material.emission_image.as_ref(),
            MapSlot::EmissionMask => model.material.emission_mask_image.as_ref(),
            // MaterialData carries no opacity texture; opacity reaches the
            // surface through a fill layer's opacity channel instead.
            MapSlot::Opacity => None,
        }?;
        Some(format!("{}x{}", img.width, img.height))
    }

    /// Assign an image file to one map slot of the active material.
    fn load_material_map(&mut self, slot: MapSlot) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Images", &["png", "jpg", "jpeg", "tga", "bmp", "webp", "tif", "tiff"])
            .pick_file()
        else {
            return;
        };
        let image = match image::open(&path) {
            Ok(img) => {
                let rgba = img.to_rgba8();
                let (width, height) = rgba.dimensions();
                ImageData { width, height, rgba: rgba.into_raw().into() }
            }
            Err(e) => {
                self.ui.status = format!("Map load failed: {e}");
                return;
            }
        };
        let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
        self.record_maps(self.active_set);
        self.apply_material_map(slot, Some(image));
        self.settle_history();
        self.ui.status = format!(
            "{} map '{name}' assigned to {}",
            slot.label(),
            self.texture_sets[self.active_set].label()
        );
    }

    /// Remove one map slot of the active material.
    fn clear_material_map(&mut self, slot: MapSlot) {
        self.record_maps(self.active_set);
        self.apply_material_map(slot, None);
        self.settle_history();
        self.ui.status = format!(
            "{} map removed from {}",
            slot.label(),
            self.texture_sets[self.active_set].label()
        );
    }

    /// Store the map on the CPU material and push the GPU texture. Height is
    /// converted to a normal map (unless an explicit normal map exists).
    fn apply_material_map(&mut self, slot: MapSlot, image: Option<ImageData>) {
        let set = &self.texture_sets[self.active_set];
        let (object, material) = (set.object.clone(), set.material.clone());
        let mut effective_normal: Option<Option<ImageData>> = None;
        let mut emission: Option<(Option<ImageData>, [f32; 3])> = None;
        for model in self
            .models
            .iter_mut()
            .filter(|m| m.name == object && m.material_name == material)
        {
            let mat = &mut model.material;
            match slot {
                MapSlot::BaseColor => {
                    mat.base_color_image = image.clone();
                    if image.is_some() {
                        mat.base_color = [1.0; 4];
                    }
                }
                MapSlot::Roughness => {
                    mat.roughness_image = image.clone();
                    if image.is_some() {
                        mat.roughness = 1.0;
                    }
                }
                MapSlot::Metallic => {
                    mat.metallic_image = image.clone();
                    if image.is_some() {
                        mat.metallic = 1.0;
                    }
                }
                MapSlot::Normal => mat.normal_image = image.clone(),
                MapSlot::Height => mat.height_image = image.clone(),
                MapSlot::Ao => mat.ao_image = image.clone(),
                MapSlot::Emission | MapSlot::EmissionMask => {
                    if slot == MapSlot::Emission {
                        mat.emission_image = image.clone();
                    } else {
                        mat.emission_mask_image = image.clone();
                    }
                    // A map on a material that doesn't glow yet would show
                    // nothing: start it at full white emission.
                    if image.is_some() && mat.emission.iter().all(|&c| c <= 0.0) {
                        mat.emission = [1.0; 3];
                    }
                    if emission.is_none() {
                        emission = Some((mat.emission_texture(), mat.emission));
                    }
                }
                MapSlot::Opacity => {}
            }
            if matches!(slot, MapSlot::Normal | MapSlot::Height) && effective_normal.is_none() {
                effective_normal = Some(mat.effective_normal_image());
            }
        }

        self.pt_scene_stale = true;
        match slot {
            // Normal and height both feed the GPU normal slot.
            MapSlot::Normal | MapSlot::Height => {
                let normal = effective_normal.flatten();
                self.renderer
                    .set_material_map(self.active_set, MapSlot::Normal, normal.as_ref());
            }
            // Colour map and mask feed one GPU texture (their product).
            MapSlot::Emission | MapSlot::EmissionMask => {
                if let Some((texture, colour)) = emission {
                    self.renderer.set_material_map(self.active_set, MapSlot::Emission, texture.as_ref());
                    let first = self.models.iter().find(|m| m.name == object && m.material_name == material);
                    if let Some(mut params) = first.map(|m| m.material.surface_params()) {
                        params.emission = colour;
                        self.renderer.set_material_surface(self.active_set, params);
                    }
                }
            }
            _ => {
                self.renderer.set_material_map(self.active_set, slot, image.as_ref());
            }
        }
    }

    fn bake_mesh_maps(&mut self) {
        if crate::bakers::bake_mesh_maps(
            &self.gpu,
            &self.renderer,
            &mut self.texture_sets,
            PAINT_RESOLUTION,
        ) {
            // Projected fills place themselves by these maps, framed to each
            // set's own bounds so a projection at 1x covers it once.
            let indices = crate::scene::set_indices(&self.models);
            for (set_index, set) in self.texture_sets.iter_mut().enumerate() {
                let (Some(position), Some(normal)) =
                    (set.baked_position.clone(), set.baked_normal.clone())
                else {
                    continue;
                };
                let (centre, size) = set_bounds(
                    self.models.iter().zip(&indices).filter(|(_, i)| **i == set_index).map(|(m, _)| m),
                );
                set.layers.set_projection_source(position, normal, centre.to_array(), size);
            }
            self.ui.status = "Mesh maps baked (position, normal, curvature)".into();
            self.push_toast(widgets::Tone::Ok, self.ui.status.clone(), None);
        } else {
            self.ui.status = "Mesh map bake failed: no visible models".into();
            self.push_toast(widgets::Tone::Error, self.ui.status.clone(), None);
        }
    }

    /// Pick an image file and fill a new layer with it, ready to tile or wrap.
    fn fill_layer_from_image_dialog(&mut self) {
        let Some((name, image)) = self.pick_fill_image() else {
            return;
        };
        let layers = &mut self.texture_sets[self.active_set].layers;
        layers.add_layer();
        let target = layers.active;
        layers.layers[target].name = name.clone();
        layers.fill_layer_from_image(target, &image);
        self.ui.status =
            format!("'{name}' filled into a new layer — pick a projection to wrap it onto the model");
    }

    /// Pick an image file and put it on the active layer's colour.
    fn fill_active_layer_from_image_dialog(&mut self) {
        let Some((name, image)) = self.pick_fill_image() else {
            return;
        };
        let layers = &mut self.texture_sets[self.active_set].layers;
        let target = layers.active;
        // A default name says nothing; the file name says what is on it.
        if layers.layers[target].name.starts_with("Layer ") {
            layers.layers[target].name = name.clone();
        }
        layers.fill_layer_from_image(target, &image);
        self.ui.status = format!("'{name}' filled into the layer — pick a projection to wrap it");
    }

    fn pick_fill_image(&mut self) -> Option<(String, ImageData)> {
        let path = rfd::FileDialog::new()
            .add_filter("Images", &["png", "jpg", "jpeg", "tga", "bmp", "webp", "tif", "tiff"])
            .pick_file()?;
        match crate::library::load_map(&path) {
            Ok(image) => {
                let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
                Some((name, image))
            }
            Err(e) => {
                self.ui.status = format!("Image load failed: {e:#}");
                None
            }
        }
    }

    fn bake_ao(&mut self) {
        if crate::ao::bake(&self.gpu, &self.renderer, &mut self.texture_sets, PAINT_RESOLUTION) {
            self.ui.status = format!(
                "AO baked for {} texture set(s) (64 directions)",
                self.texture_sets.len()
            );
            self.push_toast(widgets::Tone::Ok, self.ui.status.clone(), None);
        } else {
            self.ui.status = "AO bake failed: no models loaded".into();
            self.push_toast(widgets::Tone::Error, self.ui.status.clone(), None);
        }
    }

    /// Re-unwrap only the active material's geometry. Other sets' UVs,
    /// paint, and bakes are untouched.
    fn unwrap_uvs(&mut self) {
        let set_index = self.active_set;
        // Anything pending is its own step, before the unwrap's.
        self.track_history(false);
        let models_before = self.models.clone();
        match crate::uv_tools::auto_unwrap_set(&mut self.models, set_index) {
            Ok(()) => {
                self.renderer.set_models(&self.models);
                self.mesh_groups = None;
                self.paint_projection = None;
                // Only this set's paint lives in the old UV layout.
                let old = &self.texture_sets[set_index];
                let (object, material, visible) =
                    (old.object.clone(), old.material.clone(), old.visible);
                let mut fresh = TextureSet::new_lazy(
                    object,
                    material,
                    self.gpu.clone(),
                    PAINT_RESOLUTION,
                );
                fresh.visible = visible;
                // The material's textures are laid out on the old UVs.
                fresh.seeded = true;
                let label = fresh.label();
                let old_set = std::mem::replace(&mut self.texture_sets[set_index], fresh);
                self.record_unwrap(set_index, old_set, models_before);
                self.settle_history();
                self.ui.status =
                    format!("Re-unwrapped '{label}' — its layers were reset (others untouched)");
            }
            Err(e) => {
                self.ui.status = format!("Unwrap failed: {e:#}");
            }
        }
    }

    fn open_hdri_dialog(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("HDR environment", &["hdr", "exr"])
            .pick_file()
        else {
            return;
        };
        self.load_hdri(&path);
    }

    /// Load an HDRI as the environment and remember it in the Display &
    /// lighting environment list.
    fn load_hdri(&mut self, path: &std::path::Path) {
        match Environment::from_hdr(&self.gpu, path) {
            Ok(env) => {
                self.renderer.set_environment(env);
                let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
                self.ui.status = format!("Environment: {name}");
                self.ui.env_name = name;
                self.ui.env_path = Some(path.to_path_buf());
                if !self.ui.hdri_history.iter().any(|p| p == path) {
                    self.ui.hdri_history.push(path.to_path_buf());
                }
                self.pt_scene_stale = true;
            }
            Err(e) => {
                self.ui.status = format!("HDRI load failed: {e:#}");
            }
        }
    }

    /// Add a shader-mask layer and go straight to painting it, with the
    /// mask showing on the model.
    fn add_shader_mask(&mut self, kind: ShaderMask) {
        self.texture_sets[self.active_set].layers.add_shader_mask_layer(kind);
        if !self.tool.uses_brush() {
            self.tool = Tool::Paint;
        }
        self.brush.target = crate::paint::PaintTarget::Material;
        self.ui.mask_overlay = true;
        self.ui.props_tab = PropsTab::Layer;
        self.ui.status = if kind == ShaderMask::Matcap {
            "Matcap layer added — pick its matcap in Layer properties, then paint white where it shows (Fill white for everywhere)".into()
        } else {
            format!("{} added — paint white where the effect should show", kind.label())
        };
    }

    /// Flood the active layer with the brush's enabled channel values.
    fn fill_from_brush_values(&mut self) {
        let color = self.brush.paint_color.then_some({
            let c = self.brush.color;
            // Fill in linear space like painted strokes.
            [
                srgb_component_to_linear(c[0]),
                srgb_component_to_linear(c[1]),
                srgb_component_to_linear(c[2]),
            ]
        });
        let rough = self.brush.paint_roughness.then_some(self.brush.roughness_value);
        let metal = self.brush.paint_metallic.then_some(self.brush.metallic_value);
        let layers = &mut self.texture_sets[self.active_set].layers;
        layers.fill_active(color, rough, metal);
        self.ui.status = "Layer filled from brush values".into();
    }

    /// Open the asset library on its brush or material side.
    fn open_library(&mut self, brushes: bool) {
        self.ui.library_open = true;
        self.ui.library_tab = if brushes {
            library_window::LibraryTab::Brushes
        } else {
            library_window::LibraryTab::Materials
        };
    }

    /// Make set `i` the paint target, dropping caches built for the old one.
    /// Texture sets are created lazy (no paint textures) so big imports don't
    /// exhaust VRAM; the one being worked on gets its textures here, each
    /// frame before anything can paint into it.
    fn ensure_active_set_allocated(&mut self) {
        let set = self.active_set;
        let Some(ts) = self.texture_sets.get_mut(set) else { return };
        let upgrades = ts.layers.ensure_allocated();
        if !upgrades.is_empty() {
            self.adopt_upgrades(set, &upgrades);
        }
        self.seed_base_layer(set);
        let upgrades = self.texture_sets[set].layers.take_upgrades();
        if !upgrades.is_empty() {
            self.adopt_upgrades(set, &upgrades);
        }
    }

    /// Give hidden sets' composites back to the GPU and rebuild them for sets
    /// shown again (see `LayerStack::set_resident`). Runs before the frame is
    /// drawn, so the path tracer can rebuild its texture table first.
    fn update_set_residency(&mut self) {
        let active = self.active_set;
        let mut swapped = false;
        for (i, set) in self.texture_sets.iter_mut().enumerate() {
            swapped |= set.layers.set_resident(set.visible || i == active);
        }
        if swapped {
            self.pt_scene_stale = true;
        }
    }

    /// Every object with blendshapes and its shapes with their weights, in
    /// load order (shapes shared by an object's parts listed once).
    pub(super) fn blendshape_groups(&self) -> Vec<(String, Vec<(String, f32)>)> {
        let mut groups: Vec<(String, Vec<(String, f32)>)> = Vec::new();
        for model in self.models.iter().filter(|m| !m.morphs.targets.is_empty()) {
            let at = match groups.iter().position(|(o, _)| *o == model.name) {
                Some(at) => at,
                None => {
                    groups.push((model.name.clone(), Vec::new()));
                    groups.len() - 1
                }
            };
            let shapes = &mut groups[at].1;
            for t in &model.morphs.targets {
                if !shapes.iter().any(|(n, _)| *n == t.name) {
                    shapes.push((t.name.clone(), t.weight));
                }
            }
        }
        groups
    }

    /// Set blendshape `name` of `object` (on every part that has it) and
    /// rebuild the deformed meshes.
    pub(super) fn set_blendshape(&mut self, object: &str, name: &str, weight: f32) {
        let mut changed = false;
        for (i, model) in self.models.iter_mut().enumerate().filter(|(_, m)| m.name == object) {
            let mut hit = false;
            for t in model.morphs.targets.iter_mut().filter(|t| t.name == name && t.weight != weight) {
                t.weight = weight;
                hit = true;
            }
            if hit && model.deform() {
                self.renderer.update_vertices(i, &model.mesh.vertices);
                changed = true;
            }
        }
        if changed {
            // Everything built from the mesh's shape is out of date.
            self.pt_scene_stale = true;
            self.mesh_groups = None;
            self.paint_projection = None;
        }
    }

    /// The first time a set is opened, an untouched stack takes the
    /// material's own textures as "Base material", with an empty layer on
    /// top to paint on.
    fn seed_base_layer(&mut self, set: usize) {
        let ts = &mut self.texture_sets[set];
        if ts.seeded {
            return;
        }
        ts.seeded = true;
        let Some(material) = self
            .models
            .iter()
            .find(|m| m.name == ts.object && m.material_name == ts.material)
            .map(|m| &m.material)
        else {
            return;
        };
        let [r, g, b, _] = material.base_color;
        let maps: Vec<_> = [
            (crate::layers::MapKind::BaseColor, &material.base_color_image, [r, g, b]),
            (crate::layers::MapKind::Roughness, &material.roughness_image, [material.roughness; 3]),
            (crate::layers::MapKind::Metallic, &material.metallic_image, [material.metallic; 3]),
        ]
        .into_iter()
        .filter_map(|(kind, image, factor)| image.as_ref().map(|i| (kind, i, factor)))
        .collect();
        if ts.layers.seed_from_material(&maps) {
            ts.material_in_layers = true;
            self.ui.status = format!(
                "{}: its own textures are now the 'Base material' layer; you paint on Layer 2 above it",
                ts.label()
            );
        }
    }

    fn set_active_set(&mut self, i: usize) {
        if i != self.active_set && i < self.texture_sets.len() {
            self.active_set = i;
            self.paint_projection = None;
            self.mesh_groups = None;
        }
    }

    fn import_maps(&mut self) {
        let files = std::mem::take(&mut self.ui.import_files);
        let set = self.ui.import_set;
        let mut imported = 0;
        let mut failed = Vec::new();
        for (path, kind) in files {
            let Some(kind) = kind else { continue };
            let file = path.file_name().unwrap_or_default().to_string_lossy().to_string();
            match crate::library::load_map(&path) {
                Ok(image) => {
                    let name = format!("{} · {file}", kind.label());
                    if self.texture_sets[set].layers.import_map_layer(name, kind, &image) {
                        imported += 1;
                    }
                }
                Err(e) => failed.push(format!("{file}: {e:#}")),
            }
        }
        if self.active_set != set {
            self.active_set = set;
            self.paint_projection = None;
            self.mesh_groups = None;
        }
        self.ui.status = if failed.is_empty() {
            format!("Imported {imported} map(s) as new layers on {}", self.texture_sets[set].label())
        } else {
            format!("Imported {imported}, failed: {}", failed.join("; "))
        };
    }

    /// The shader mask the active layer paints into, when the brush targets
    /// material (a layer-mask stroke is still a layer-mask stroke).
    fn active_shader_mask(&self) -> Option<ShaderMask> {
        let layers = &self.texture_sets[self.active_set].layers;
        layers.active_layer().purpose
    }

    /// Which shader mask the viewport overlays, if any.
    fn mask_view(&self) -> Option<ShaderMask> {
        if !self.ui.mask_overlay {
            return None;
        }
        self.active_shader_mask()
    }

    /// The brush as the stroke should see it. On a shader-mask layer the
    /// stroke writes one greyscale value into the colour channel and nothing
    /// else, whatever the material toggles say.
    fn effective_brush(&self) -> BrushSettings {
        let mut brush = self.brush.clone();
        match self.tool {
            Tool::Stamp => {
                brush.stroke_mode = crate::paint::StrokeMode::Stamp;
                brush.square_tip = true;
                brush.fill_mode = crate::paint::FillMode::Off;
            }
            // The source offset is in screen space; a mirrored copy of it
            // would land somewhere unrelated.
            Tool::Clone => brush.symmetry_x = false,
            _ => {}
        }
        if brush.target == crate::paint::PaintTarget::Material
            && self.active_shader_mask().is_some()
        {
            let v = linear_component_to_srgb(brush.shader_mask_value);
            brush.color = [v, v, v];
            brush.use_tip_color = false;
            brush.paint_color = true;
            brush.paint_roughness = false;
            brush.paint_metallic = false;
            brush.paint_height = false;
            brush.paint_emissive = false;
            brush.paint_opacity = false;
        }
        brush
    }

    fn render_params(&self) -> RenderParams {
        let aspect = self.viewport.size[0] as f32 / self.viewport.size[1] as f32;
        RenderParams {
            camera_pos: self.camera.position(),
            view_proj: self.camera.view_proj(aspect),
            view: self.camera.view_matrix(),
            exposure: self.ui.exposure,
            env_intensity: self.ui.env_intensity,
            key_light: self.ui.key_light,
            key_light_dir: self.key_light_direction(),
            // Outline the selection in every tool, but not mid-stroke, where it
            // would get in the way of judging paint.
            selected: if self.stroke_last.is_none() { self.selection.clone() } else { Vec::new() },
            hovered: self.outliner_hover.clone(),
            outline_width: self.outline_width(),
            mask_view: self.mask_view(),
            ao_strength: self.ui.ao_strength,
            uv_checker: self.ui.uv_checker,
            material_override: self.ui.override_material.then(|| MaterialOverride {
                base_color: [
                    self.ui.base_color[0],
                    self.ui.base_color[1],
                    self.ui.base_color[2],
                    1.0,
                ],
                metallic: self.ui.metallic,
                roughness: self.ui.roughness,
            }),
        }
    }
}

fn linear_component_to_srgb(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.0031308 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 }
}

fn srgb_component_to_linear(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// One [ / ] press: 10% bigger or smaller, 30% with Shift. Proportional, so
/// a step is as useful at 4 px as at 150 px; it always moves at least 1 px
/// and stays within the size slider's 2–200 px range.
fn step_brush_size(size: f32, grow: bool, fast: bool) -> f32 {
    let factor = if fast { 1.3 } else { 1.1 };
    let next = if grow { (size * factor).max(size + 1.0) } else { (size / factor).min(size - 1.0) };
    next.clamp(2.0, 200.0)
}

/// An angle folded into -π..π, so typed and dragged rotations stay readable.
fn wrap_angle(a: f32) -> f32 {
    (a + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI
}

fn scene_bounds(models: &[ModelData]) -> (Vec3, f32) {
    let (min, max) = world_box(models);
    if !min.is_finite() {
        return (Vec3::ZERO, 1.0);
    }
    ((min + max) * 0.5, ((max - min).length() * 0.5).max(1e-3))
}

/// Centre and longest side of some models' world-space bounding box — the
/// frame a projected fill is laid out in.
fn set_bounds<'a>(models: impl IntoIterator<Item = &'a ModelData>) -> (Vec3, f32) {
    let (min, max) = world_box(models);
    if !min.is_finite() {
        return (Vec3::ZERO, 1.0);
    }
    ((min + max) * 0.5, (max - min).max_element().max(1e-3))
}

fn world_box<'a>(models: impl IntoIterator<Item = &'a ModelData>) -> (Vec3, Vec3) {
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for model in models {
        let (lo, hi) = model.mesh.bounds();
        for i in 0..8 {
            let corner = Vec3::new(
                if i & 1 == 0 { lo.x } else { hi.x },
                if i & 2 == 0 { lo.y } else { hi.y },
                if i & 4 == 0 { lo.z } else { hi.z },
            );
            let p = model.transform.transform_point3(corner);
            min = min.min(p);
            max = max.max(p);
        }
    }
    (min, max)
}

impl App {
    /// Open or close the library window so it matches what the UI asked for.
    fn sync_browser_windows(&mut self, event_loop: &ActiveEventLoop) {
        match (self.state.ui.library_open, self.browsers.is_empty()) {
            (true, true) => self.open_browser_window(event_loop),
            (false, false) => {
                for browser in self.browsers.drain(..) {
                    self.windows.remove_renderer(browser.id);
                }
            }
            _ => {}
        }
    }

    fn open_browser_window(&mut self, event_loop: &ActiveEventLoop) {
        let id = self.windows.create_window(
            event_loop,
            &self.context,
            &WindowDescriptor {
                title: "ZestSurface — Library".into(),
                width: 1280.0,
                height: 800.0,
                ..Default::default()
            },
            |ci| {
                ci.image_format = Format::B8G8R8A8_UNORM;
                ci.min_image_count = ci.min_image_count.max(2);
            },
        );
        let Some(renderer) = self.windows.get_renderer_mut(id) else { return };
        let gui = Gui::new(
            event_loop,
            renderer.surface(),
            renderer.graphics_queue(),
            renderer.swapchain_format(),
            GuiConfig::default(),
        );
        crate::icons::install(&gui.context());
        crate::theme::apply(&gui.context());
        self.browsers.push(BrowserWindow { id, gui });
    }

    /// Draw and present one browser window.
    fn draw_browser(&mut self, index: usize) {
        let Some(browser) = self.browsers.get_mut(index) else { return };
        let Some(renderer) = self.windows.get_renderer_mut(browser.id) else { return };

        browser.gui.immediate_ui(|gui| {
            let ctx = gui.context();
            self.state.library_window_ui(&ctx);
        });

        match renderer.acquire(None, |_| {}) {
            Ok(before) => {
                let after =
                    browser.gui.draw_on_image(before, renderer.swapchain_image_view());
                renderer.present(after, true);
            }
            Err(vulkano::VulkanError::OutOfDate) => renderer.resize(),
            Err(e) => panic!("failed to acquire browser swapchain image: {e}"),
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.windows.create_window(
            event_loop,
            &self.context,
            &WindowDescriptor {
                title: "ZestSurface".into(),
                width: 1600.0,
                height: 900.0,
                ..Default::default()
            },
            |ci| {
                ci.image_format = Format::B8G8R8A8_UNORM;
                ci.min_image_count = ci.min_image_count.max(2);
            },
        );

        let renderer = self.windows.get_primary_renderer_mut().unwrap();
        let gui = Gui::new(
            event_loop,
            renderer.surface(),
            renderer.graphics_queue(),
            renderer.swapchain_format(),
            GuiConfig::default(),
        );
        // Taskbar / title-bar icon where the platform takes it from the
        // window (X11, Windows); Wayland reads it from the .desktop entry.
        if let Some(window) = self.windows.get_primary_window() {
            window.set_window_icon(crate::icons::app_icon(256).and_then(|(rgba, size)| {
                winit::window::Icon::from_rgba(rgba, size, size).ok()
            }));
        }
        crate::icons::install(&gui.context());
        crate::theme::apply(&gui.context());
        self.gui = Some(gui);
        self.primary = self.windows.primary_window_id();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        // A browser window: its own egui context, and closing it just hides the
        // browser rather than quitting the app.
        if let Some(index) = self.browsers.iter().position(|b| b.id == window_id) {
            self.browsers[index].gui.update(&event);
            match event {
                WindowEvent::Resized(_) | WindowEvent::ScaleFactorChanged { .. } => {
                    if let Some(r) = self.windows.get_renderer_mut(window_id) {
                        r.resize();
                    }
                }
                WindowEvent::CloseRequested => {
                    self.state.ui.library_open = false;
                    let browser = self.browsers.remove(index);
                    self.windows.remove_renderer(browser.id);
                }
                WindowEvent::RedrawRequested => {
                    self.draw_browser(index);
                    // Imports and rescans happen in there, so reconcile after.
                    self.sync_browser_windows(event_loop);
                }
                _ => {}
            }
            return;
        }

        if self.primary.is_some_and(|p| p != window_id) {
            return;
        }
        let Some(renderer) = self.windows.get_renderer_mut(window_id) else {
            return;
        };
        let Some(gui) = self.gui.as_mut() else {
            return;
        };

        // Numpad . frames the selection (Blender). Caught before egui: with
        // NumLock off it arrives as Delete, which deletes a layer.
        if let WindowEvent::KeyboardInput { event: key, .. } = &event
            && key.physical_key == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::NumpadDecimal)
            && !gui.context().wants_keyboard_input()
        {
            if key.state.is_pressed() && !key.repeat {
                self.state.frame_selection_pending = true;
            }
            return;
        }

        gui.update(&event);

        match event {
            WindowEvent::Resized(_) | WindowEvent::ScaleFactorChanged { .. } => {
                renderer.resize();
            }
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                gui.immediate_ui(|gui| self.state.layout(gui));

                match renderer.acquire(None, |_| {}) {
                    Ok(before_future) => {
                        let params = self.state.render_params();
                        let state = &mut self.state;
                        let after_scene = if state.pt_mode {
                            // Progressive path tracing into the viewport image
                            // (or a render's next samples).
                            state.path_trace(before_future, &params)
                        } else {
                            state.renderer.render(
                                before_future,
                                state.viewport.scene_image.clone(),
                                &params,
                                &mut state.texture_sets,
                            )
                        };
                        let after_gui =
                            gui.draw_on_image(after_scene, renderer.swapchain_image_view());
                        renderer.present(after_gui, true);
                        // The frame has been waited on: the scene image is
                        // finished, so assistant captures can read it now.
                        self.state.service_assistant_captures();
                    }
                    Err(vulkano::VulkanError::OutOfDate) => {
                        renderer.resize();
                    }
                    Err(e) => panic!("failed to acquire swapchain image: {e}"),
                }
                self.sync_browser_windows(event_loop);
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(renderer) = self.windows.get_primary_renderer() {
            renderer.window().request_redraw();
        }
        let ids: Vec<WindowId> = self.browsers.iter().map(|b| b.id).collect();
        for id in ids {
            if let Some(r) = self.windows.get_renderer(id) {
                r.window().request_redraw();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tri(sx: f32, sy: f32, size: f32, u0: f32, v0: f32, u1: f32, v1: f32) -> ProjectedTri {
        ProjectedTri {
            screen_lo: Vec2::new(sx, sy),
            screen_hi: Vec2::new(sx + size, sy + size),
            uv_lo: Vec2::new(u0, v0),
            uv_hi: Vec2::new(u1, v1),
        }
    }

    fn projection(tris: Vec<ProjectedTri>, degenerate: bool) -> PaintProjection {
        PaintProjection {
            view_proj: Mat4::IDENTITY,
            set: 0,
            viewport: Vec2::new(1280.0, 720.0),
            symmetry_x: false,
            tris,
            degenerate,
        }
    }

    /// Isolate must restore exactly what was visible, not just show everything.
    #[test]
    fn isolate_round_trips_the_previous_visibility() {
        // C was already hidden before isolating.
        let mut visible = [true, true, false, true];
        let backup = toggle_isolation(&mut visible, &[1], None);
        assert_eq!(visible, [false, true, false, false], "only the selection stays");
        assert!(backup.is_some());

        let backup = toggle_isolation(&mut visible, &[1], backup);
        assert_eq!(visible, [true, true, false, true], "C must stay hidden");
        assert!(backup.is_none(), "leaving isolation clears the backup");
    }

    /// Nothing selected and nothing stored means nothing happens.
    #[test]
    fn isolate_without_a_selection_is_a_no_op() {
        let mut visible = [true, true];
        assert!(toggle_isolation(&mut visible, &[], None).is_none());
        assert_eq!(visible, [true, true]);
    }

    /// Picking needs the ray test to be right about hits and misses.
    #[test]
    fn ray_triangle_hits_only_what_it_should() {
        let a = Vec3::new(-1.0, -1.0, 0.0);
        let b = Vec3::new(1.0, -1.0, 0.0);
        let c = Vec3::new(0.0, 1.0, 0.0);
        let dir = Vec3::new(0.0, 0.0, -1.0);

        let hit = ray_hits_triangle(Vec3::new(0.0, 0.0, 5.0), dir, a, b, c);
        assert!(hit.is_some_and(|t| (t - 5.0).abs() < 1e-3), "straight on should hit at 5");

        assert!(
            ray_hits_triangle(Vec3::new(4.0, 0.0, 5.0), dir, a, b, c).is_none(),
            "off to the side must miss"
        );
        assert!(
            ray_hits_triangle(Vec3::new(0.0, 0.0, -5.0), dir, a, b, c).is_none(),
            "pointing away must miss"
        );
        // Parallel to the plane.
        assert!(
            ray_hits_triangle(
                Vec3::new(0.0, 0.0, 5.0),
                Vec3::new(1.0, 0.0, 0.0),
                a,
                b,
                c
            )
            .is_none(),
            "a parallel ray must miss"
        );
    }

    /// The paint bucket floods exactly the picked pieces and nothing else —
    /// crucially not a piece sitting behind the one under the cursor.
    #[test]
    fn group_fill_floods_only_the_picked_pieces() {
        let groups = [0u32, 0, 1, 1, 2];

        let mut wanted = std::collections::HashSet::new();
        wanted.insert(0);
        assert_eq!(marks_for_groups(&groups, &wanted), vec![1, 1, 0, 0, 0]);

        wanted.insert(2);
        assert_eq!(marks_for_groups(&groups, &wanted), vec![1, 1, 0, 0, 1]);

        // Nothing picked means nothing filled.
        assert_eq!(
            marks_for_groups(&groups, &std::collections::HashSet::new()),
            vec![0, 0, 0, 0, 0]
        );
    }

    /// The fill lock: once a stroke picks a piece, every later stamp must keep
    /// filling that same piece even if the cursor wanders onto another one.
    #[test]
    fn fill_lock_keeps_a_stroke_on_one_piece() {
        let groups = [0u32, 0, 1, 1];
        let mut locked = std::collections::HashSet::new();
        locked.insert(0u32);
        let first = marks_for_groups(&groups, &locked);
        assert_eq!(first, vec![1, 1, 0, 0]);

        // Later stamps reuse the locked group, never the one now under the
        // cursor, so the result cannot grow.
        let later = marks_for_groups(&groups, &locked);
        assert_eq!(later, first, "a locked stroke must not spread to piece 1");
    }

    /// The left button belongs to the brush in every stroke mode — this is the
    /// rule that broke Line and Stamp mode into orbiting while drawing.
    #[test]
    fn left_drag_never_orbits_while_painting() {
        assert_eq!(camera_gesture(true, true, false), None, "paint LMB must not orbit");
        assert_eq!(
            camera_gesture(true, false, false),
            None,
            "no buttons down means no camera move"
        );
    }

    /// A clone anchor must land back on the pixel it was picked from:
    /// projecting is the exact inverse of the picking ray, at any depth.
    #[test]
    fn projection_inverts_the_pick_ray() {
        let size = Vec2::new(800.0, 600.0);
        let view_proj = glam::camera::rh::proj::vulkan::perspective(0.8, size.x / size.y, 0.1, 100.0)
            * glam::camera::rh::view::look_at_mat4(Vec3::new(1.0, 2.0, 5.0), Vec3::ZERO, Vec3::Y);
        let inverse = view_proj.inverse();
        for px in [Vec2::new(400.0, 300.0), Vec2::new(10.0, 590.0), Vec2::new(700.0, 40.0)] {
            let ndc = px / size * 2.0 - Vec2::ONE;
            // Same unprojection as pick_surface, part way along the ray.
            let p = inverse * glam::Vec4::new(ndc.x, ndc.y, 0.7, 1.0);
            let world = p.truncate() / p.w;
            let back = project_to_viewport(view_proj, size, world).unwrap();
            assert!(back.distance(px) < 0.01, "{px} -> {back}");
        }
        // Behind the camera is not on screen.
        assert!(project_to_viewport(view_proj, size, Vec3::new(2.0, 4.0, 10.0)).is_none());
    }

    /// The picker hit test interpolates UVs at the hit point.
    #[test]
    fn barycentric_hit_interpolates() {
        let (a, b, c) = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        let (t, u, v) = ray_hit_barycentric(
            Vec3::new(0.25, 0.5, 5.0),
            Vec3::new(0.0, 0.0, -1.0),
            a,
            b,
            c,
        )
        .unwrap();
        assert!((t - 5.0).abs() < 1e-5);
        assert!((u - 0.25).abs() < 1e-5 && (v - 0.5).abs() < 1e-5, "u={u} v={v}");
    }

    /// Picked colours go back into the sRGB picker unchanged.
    #[test]
    fn picker_colour_round_trips_through_linear() {
        for v in [0.0, 0.02, 0.2, 0.5, 0.85, 1.0] {
            let back = linear_component_to_srgb(srgb_component_to_linear(v));
            assert!((back - v).abs() < 1e-4, "{v} -> {back}");
        }
        // A flat sRGB image samples to its own colour, wrapping out-of-range UVs.
        let image = crate::scene::ImageData { width: 2, height: 2, rgba: [200, 100, 50, 255].repeat(4).into() };
        let got = sample_srgb_image(&image, Vec2::new(1.3, -0.7));
        let want = Vec3::new(200.0, 100.0, 50.0).map(|c| srgb_component_to_linear(c / 255.0));
        assert!(got.distance(want) < 1e-4, "{got} vs {want}");
    }

    /// Panning stays available while painting, and orbit needs the brush to be
    /// out of the way (orbit tool, or alt held, both of which clear paint_tool).
    #[test]
    fn bracket_keys_step_brush_size() {
        // Proportional steps, faster with Shift.
        assert!((step_brush_size(100.0, true, false) - 110.0).abs() < 1e-3);
        assert!((step_brush_size(100.0, true, true) - 130.0).abs() < 1e-3);
        assert!(step_brush_size(100.0, false, true) < step_brush_size(100.0, false, false));
        // Small brushes still move a whole pixel per press.
        assert_eq!(step_brush_size(4.0, true, false), 5.0);
        assert_eq!(step_brush_size(4.0, false, false), 3.0);
        // Clamped to the slider's range.
        assert_eq!(step_brush_size(195.0, true, true), 200.0);
        assert_eq!(step_brush_size(2.5, false, false), 2.0);
    }

    #[test]
    fn camera_gestures_route_as_documented() {
        assert_eq!(camera_gesture(true, false, true), Some(CameraGesture::Pan));
        assert_eq!(camera_gesture(true, true, true), Some(CameraGesture::Pan));
        assert_eq!(camera_gesture(false, true, false), Some(CameraGesture::Orbit));
        assert_eq!(camera_gesture(false, false, true), Some(CameraGesture::Pan));
        assert_eq!(camera_gesture(false, false, false), None);
    }

    /// The scissor must cover every triangle the brush overlaps, and nothing
    /// forces it wider than that. Under-covering would visibly clip strokes.
    #[test]
    fn paint_region_covers_only_the_triangles_under_the_brush() {
        // Near triangle maps to the left half of the texture, far one to the right.
        let proj = projection(
            vec![
                tri(0.0, 0.0, 100.0, 0.0, 0.0, 0.5, 0.5),
                tri(600.0, 600.0, 100.0, 0.6, 0.6, 1.0, 1.0),
            ],
            false,
        );

        let PaintRegion::Rect([x, y, w, h]) =
            region_from_projection(&proj, &[Vec2::new(50.0, 50.0)], 10.0, 1024)
        else {
            panic!("expected a rect over the near triangle");
        };
        // Covers the near triangle's UV box (0..0.5 -> 0..512) plus margin.
        assert!(x <= SCISSOR_MARGIN, "left edge should reach the UV origin, got {x}");
        assert!(y <= SCISSOR_MARGIN);
        assert!(x + w >= 512, "must cover u=0.5, got {}", x + w);
        assert!(y + h >= 512);
        // ...but must not have been dragged out to the far triangle at u=1.0.
        assert!(x + w < 1024, "far triangle should not be included, got {}", x + w);
    }

    /// Both triangles overlapped means the union of both UV boxes.
    #[test]
    fn paint_region_unions_everything_it_touches() {
        let proj = projection(
            vec![
                tri(0.0, 0.0, 100.0, 0.0, 0.0, 0.25, 0.25),
                tri(600.0, 600.0, 100.0, 0.75, 0.75, 1.0, 1.0),
            ],
            false,
        );
        let stamps = [Vec2::new(50.0, 50.0), Vec2::new(650.0, 650.0)];
        let PaintRegion::Rect([x, y, w, h]) =
            region_from_projection(&proj, &stamps, 10.0, 1024)
        else {
            panic!("expected a rect spanning both");
        };
        assert_eq!((x, y), (0, 0), "margin should clamp at the texture edge");
        assert_eq!((x + w, y + h), (1024, 1024), "and at the far edge");
    }

    /// Brush over empty space paints nothing, so the draw can be skipped.
    #[test]
    fn paint_region_is_empty_when_the_brush_misses() {
        let proj = projection(vec![tri(0.0, 0.0, 100.0, 0.0, 0.0, 0.5, 0.5)], false);
        assert!(matches!(
            region_from_projection(&proj, &[Vec2::new(900.0, 900.0)], 10.0, 1024),
            PaintRegion::Nothing
        ));
    }

    /// Anything the projection cannot bound must fall back to a full pass,
    /// never to a guess.
    #[test]
    fn paint_region_falls_back_when_bounds_are_unreliable() {
        let good = vec![tri(0.0, 0.0, 100.0, 0.0, 0.0, 0.5, 0.5)];
        let stamp = [Vec2::new(50.0, 50.0)];
        assert!(matches!(
            region_from_projection(&projection(good.clone(), true), &stamp, 10.0, 1024),
            PaintRegion::Everything
        ));
        assert!(matches!(
            region_from_projection(&projection(vec![], false), &stamp, 10.0, 1024),
            PaintRegion::Everything
        ));
        assert!(matches!(
            region_from_projection(&projection(good, false), &[], 10.0, 1024),
            PaintRegion::Everything
        ));
    }

    /// The region search runs every frame of a drag, so it has to be cheap
    /// relative to the GPU work it saves. Run with:
    ///   cargo test --release bench_paint_region -- --ignored --nocapture
    #[test]
    #[ignore = "benchmark"]
    fn bench_paint_region() {
        // A mesh on the scale of one material of a detailed avatar.
        const TRIS: usize = 50_000;
        let tris: Vec<ProjectedTri> = (0..TRIS)
            .map(|i| {
                let f = i as f32 / TRIS as f32;
                tri(f * 1200.0, (f * 7.0).fract() * 700.0, 6.0, f, (f * 3.0).fract(), f + 0.01, (f * 3.0).fract() + 0.01)
            })
            .collect();
        let proj = projection(tris, false);
        let stamps: Vec<Vec2> = (0..12).map(|i| Vec2::new(600.0 + i as f32 * 4.0, 350.0)).collect();

        for _ in 0..200 {
            std::hint::black_box(region_from_projection(&proj, &stamps, 40.0, 2048));
        }
        let mut samples = Vec::new();
        for _ in 0..200 {
            let t = std::time::Instant::now();
            std::hint::black_box(region_from_projection(&proj, &stamps, 40.0, 2048));
            samples.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "region search over {TRIS} tris: min {:.3} ms  median {:.3} ms",
            samples[0],
            samples[samples.len() / 2]
        );
        let PaintRegion::Rect([_, _, w, h]) =
            region_from_projection(&proj, &stamps, 40.0, 2048)
        else {
            panic!("expected a rect");
        };
        let covered = (w as f64 * h as f64) / (2048.0 * 2048.0) * 100.0;
        println!("scissor covers {covered:.1}% of the 2048px target");
    }

    /// A big brush radius must widen the search, not just the stamp centre.
    #[test]
    fn paint_region_accounts_for_brush_radius() {
        let proj = projection(vec![tri(200.0, 200.0, 50.0, 0.2, 0.2, 0.4, 0.4)], false);
        let stamp = [Vec2::new(100.0, 100.0)];
        assert!(
            matches!(region_from_projection(&proj, &stamp, 10.0, 1024), PaintRegion::Nothing),
            "small brush should miss"
        );
        assert!(
            matches!(region_from_projection(&proj, &stamp, 200.0, 1024), PaintRegion::Rect(_)),
            "large brush should reach it"
        );
    }
}
