//! Every menu command, in one table: the menu bar draws from it, the Ctrl K
//! palette searches it, and new keyboard shortcuts dispatch through it, so
//! the three can never disagree about what a command is called or does.

use egui::{Key, KeyboardShortcut, Modifiers};

use super::{AppState, PropsTab, Tool};
use crate::layers::ShaderMask;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Cmd {
    // File
    OpenModel,
    OpenProject,
    SaveProject,
    SaveProjectAs,
    ImportMaps,
    ExportMaps,
    ExportToZestEngine,
    LoadHdri,
    Quit,
    // Edit
    Undo,
    Redo,
    UndoBudget,
    Shortcuts,
    Preferences,
    // View
    RenderRaster,
    RenderPathTraced,
    Studio,
    StudioPose,
    StudioLights,
    StudioSet,
    StudioCamera,
    RenderImage,
    LastRender,
    DisplayLighting,
    UvChecker,
    MaskOnModel,
    OverrideMaterial,
    Hide,
    HideOthers,
    UnhideAll,
    Isolate,
    DeletePieces,
    SelectAll,
    DeselectAll,
    FrameSelected,
    FrameAll,
    // Layer
    NewPaintLayer,
    NewFillImage,
    NewFromLibrary,
    NewShaderMask(ShaderMask),
    Duplicate,
    Rename,
    GroupFolder,
    MoveUp,
    MoveDown,
    MergeDown,
    ToggleClip,
    DeleteLayer,
    AddBlackMask,
    AddWhiteMask,
    InvertMask,
    RemoveMask,
    ToggleMaskEnabled,
    ApplyMask,
    LoadMaskImage,
    ExportLayer,
    ExportLayerMask,
    FillFromBrush,
    // Bake
    BakeAo,
    BakeMeshMaps,
    BakeAll,
    AutoUnwrap,
    // Library
    BrowseMaterials,
    BrowseBrushes,
    SaveMaterial,
    ImportBrushSet,
    ImportBrushFolder,
    RescanLibraries,
    OpenLibraryFolder,
    // Window
    ToggleBrushDock,
    ToggleTextureSets,
    ToggleLayers,
    ToggleProperties,
    ToggleAssistant,
    MaterialLibraryWindow,
    BrushLibraryWindow,
    TextureView2d,
    ResetLayout,
    // Help
    Welcome,
    ToolGestures,
    About,
    // Tools
    Tool(Tool),
    CommandSearch,
}

/// One entry in a menu.
pub enum Item {
    Cmd(Cmd),
    Separator,
    /// A submenu of commands.
    Sub(&'static str, Vec<Cmd>),
    /// File ▸ Open recent: built from the recent-projects list when drawn.
    Recent,
}

/// The menu bar, left to right.
pub fn menus() -> Vec<(&'static str, Vec<Item>)> {
    use Cmd::*;
    use Item::{Cmd as C, Separator as S};
    vec![
        (
            "File",
            vec![
                C(OpenModel),
                C(OpenProject),
                Item::Recent,
                C(SaveProject),
                C(SaveProjectAs),
                S,
                C(ImportMaps),
                C(ExportMaps),
                C(ExportToZestEngine),
                C(ExportLayer),
                C(ExportLayerMask),
                S,
                C(LoadHdri),
                S,
                C(Quit),
            ],
        ),
        (
            "Edit",
            vec![C(Undo), C(Redo), S, C(UndoBudget), C(Shortcuts), C(Preferences)],
        ),
        (
            "View",
            vec![
                C(RenderRaster),
                C(RenderPathTraced),
                S,
                C(DisplayLighting),
                C(UvChecker),
                C(MaskOnModel),
                C(OverrideMaterial),
                S,
                C(SelectAll),
                C(DeselectAll),
                C(Hide),
                C(HideOthers),
                C(UnhideAll),
                C(Isolate),
                C(DeletePieces),
                S,
                C(FrameSelected),
                C(FrameAll),
            ],
        ),
        (
            "Layer",
            vec![
                C(NewPaintLayer),
                C(NewFillImage),
                C(NewFromLibrary),
                Item::Sub("New shader mask", ShaderMask::ALL.iter().map(|k| NewShaderMask(*k)).collect()),
                S,
                C(Duplicate),
                C(Rename),
                C(GroupFolder),
                C(MoveUp),
                C(MoveDown),
                C(MergeDown),
                C(ToggleClip),
                C(DeleteLayer),
                S,
                C(AddBlackMask),
                C(AddWhiteMask),
                C(InvertMask),
                C(ToggleMaskEnabled),
                C(ApplyMask),
                C(RemoveMask),
                C(LoadMaskImage),
                S,
                C(ExportLayer),
                C(ExportLayerMask),
                S,
                C(FillFromBrush),
            ],
        ),
        ("Bake", vec![C(BakeAo), C(BakeMeshMaps), C(BakeAll), S, C(AutoUnwrap)]),
        (
            "Library",
            vec![
                C(BrowseMaterials),
                C(BrowseBrushes),
                S,
                C(SaveMaterial),
                C(ImportBrushSet),
                C(ImportBrushFolder),
                S,
                C(RescanLibraries),
                C(OpenLibraryFolder),
            ],
        ),
        (
            "Window",
            vec![
                C(ToggleBrushDock),
                C(ToggleTextureSets),
                C(ToggleLayers),
                C(ToggleProperties),
                C(ToggleAssistant),
                S,
                C(MaterialLibraryWindow),
                C(BrushLibraryWindow),
                C(TextureView2d),
                S,
                C(ResetLayout),
            ],
        ),
        (
            "Studio",
            vec![C(Studio), S, C(StudioPose), C(StudioLights), C(StudioSet), C(StudioCamera), S, C(RenderImage), C(LastRender)],
        ),
        ("Help", vec![C(Welcome), S, C(ToolGestures), C(Shortcuts), C(About)]),
    ]
}

/// Every command in the menus, flattened, for the palette.
pub fn all_menu_commands() -> Vec<(&'static str, Cmd)> {
    let mut out = Vec::new();
    for (menu, items) in menus() {
        for item in items {
            match item {
                Item::Cmd(cmd) => {
                    if !out.iter().any(|(_, c)| *c == cmd) {
                        out.push((menu, cmd));
                    }
                }
                Item::Sub(_, cmds) => out.extend(cmds.into_iter().map(|c| (menu, c))),
                Item::Separator | Item::Recent => {}
            }
        }
    }
    for tool in super::shell::TOOL_RAIL.iter().flat_map(|g| g.iter()) {
        out.push(("Tools", Cmd::Tool(*tool)));
    }
    out
}

/// Panel controls the palette can jump to (Tab focuses their panel): the
/// label people search for, and where it lives.
pub fn panel_controls() -> &'static [(&'static str, PanelTarget)] {
    use PanelTarget::*;
    &[
        ("Layer name", Props(PropsTab::Layer)),
        ("Layer opacity", Props(PropsTab::Layer)),
        ("Blend mode", Props(PropsTab::Layer)),
        ("Mask for (shader mask kind)", Props(PropsTab::Layer)),
        ("Layer mask: fill / clear / invert", Props(PropsTab::Mask)),
        ("Paint the mask", Props(PropsTab::Mask)),
        ("Smart mask: edge wear / cavity dirt", Props(PropsTab::Mask)),
        ("Projection mode (triplanar, spherical…)", Props(PropsTab::Projection)),
        ("Projection placement: position / rotation / size", Props(PropsTab::Projection)),
        ("Tiling: repeat / offset / spin", Props(PropsTab::Projection)),
        ("Fill with image", Props(PropsTab::Projection)),
        ("Material map slots", Props(PropsTab::Material)),
        ("Surface: opacity, coat, sheen, specular", Props(PropsTab::Material)),
        ("AO strength", Props(PropsTab::Material)),
        ("Save material to library", Props(PropsTab::Material)),
        ("Brush tip", BrushDock),
        ("Paint channels: color / roughness / metallic / height", BrushDock),
        ("Bump strength", BrushDock),
        ("Exposure / environment / key light", Display),
    ]
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PanelTarget {
    Props(PropsTab),
    BrushDock,
    Display,
}

impl Cmd {
    /// Coming-soon commands: shown disabled with a tooltip, never run.
    pub fn planned(self) -> bool {
        matches!(self, Cmd::GroupFolder | Cmd::TextureView2d)
    }

    pub fn label(self, app: &AppState) -> String {
        use Cmd::*;
        match self {
            OpenModel => "Open model…".into(),
            OpenProject => "Open project…".into(),
            SaveProject => "Save project".into(),
            SaveProjectAs => "Save project as…".into(),
            ImportMaps => "Import maps…".into(),
            ExportMaps => "Export maps…".into(),
            ExportToZestEngine => "Export to ZestEngine".into(),
            LoadHdri => "Load HDRI environment…".into(),
            Quit => "Quit".into(),
            Undo => "Undo".into(),
            Redo => "Redo".into(),
            UndoBudget => "Undo budget…".into(),
            Shortcuts => "Keyboard shortcuts…".into(),
            Preferences => "Preferences…".into(),
            RenderRaster => "Render: Raster".into(),
            RenderPathTraced => "Render: Path traced".into(),
            Studio => "Studio window".into(),
            StudioPose => "Pose the avatar…".into(),
            StudioLights => "Lights…".into(),
            StudioSet => "Backdrop & background…".into(),
            StudioCamera => "Camera & lens…".into(),
            RenderImage => "Render image".into(),
            LastRender => "Show last render".into(),
            DisplayLighting => "Display & lighting…".into(),
            UvChecker => "UV checker".into(),
            MaskOnModel => "Show mask on model".into(),
            OverrideMaterial => "Override material".into(),
            Hide => "Hide selection".into(),
            HideOthers => "Hide others".into(),
            UnhideAll => "Unhide all".into(),
            Isolate => {
                if app.isolate_backup.is_some() { "Leave local view".into() } else { "Isolate selection (local view)".into() }
            }
            DeletePieces => match app.selection.len() {
                0 | 1 => "Delete piece".into(),
                n => format!("Delete {n} pieces"),
            },
            SelectAll => "Select all pieces".into(),
            DeselectAll => "Deselect all".into(),
            FrameSelected => "Frame selection".into(),
            FrameAll => "Frame all".into(),
            NewPaintLayer => "New paint layer".into(),
            NewFillImage => "New fill from image…".into(),
            NewFromLibrary => "New from library material…".into(),
            NewShaderMask(kind) => kind.label().into(),
            Duplicate => "Duplicate layer".into(),
            Rename => "Rename layer".into(),
            GroupFolder => "Group into folder".into(),
            MoveUp => "Move layer up".into(),
            MoveDown => "Move layer down".into(),
            MergeDown => "Merge down".into(),
            ToggleClip => "Clip to layer below".into(),
            ToggleMaskEnabled => "Mask enabled".into(),
            ApplyMask => "Apply mask".into(),
            DeleteLayer => "Delete layer".into(),
            AddBlackMask => "Add black mask".into(),
            AddWhiteMask => "Add white mask".into(),
            InvertMask => "Invert mask".into(),
            RemoveMask => "Remove mask".into(),
            LoadMaskImage => "Load mask from image…".into(),
            ExportLayer => "Export layer…".into(),
            ExportLayerMask => "Export layer mask…".into(),
            FillFromBrush => "Fill from brush values".into(),
            BakeAo => "Bake AO".into(),
            BakeMeshMaps => "Bake mesh maps".into(),
            BakeAll => "Bake all".into(),
            AutoUnwrap => format!("Auto-unwrap UVs ({})…", app.texture_sets[app.active_set].material),
            BrowseMaterials => "Browse materials…".into(),
            BrowseBrushes => "Browse brushes…".into(),
            SaveMaterial => "Save active material…".into(),
            ImportBrushSet => "Import brush set…".into(),
            ImportBrushFolder => "Import folder…".into(),
            RescanLibraries => "Rescan libraries".into(),
            OpenLibraryFolder => "Open library folder".into(),
            ToggleBrushDock => "Brush".into(),
            ToggleTextureSets => "Texture sets".into(),
            ToggleLayers => "Layers".into(),
            ToggleProperties => "Properties".into(),
            ToggleAssistant => "Assistant".into(),
            MaterialLibraryWindow => "Material library window".into(),
            BrushLibraryWindow => "Brush library window".into(),
            TextureView2d => "2D texture view".into(),
            ResetLayout => "Reset layout".into(),
            Welcome => "Welcome screen".into(),
            ToolGestures => "Tool gestures".into(),
            About => "About ZestSurface".into(),
            Tool(tool) => format!("Tool: {}", super::shell::tool_name(tool)),
            CommandSearch => "Search commands".into(),
        }
    }

    /// Icon shown before the label in menus, for the commands that have one.
    pub fn icon(self) -> Option<&'static str> {
        use crate::icons;
        use Cmd::*;
        Some(match self {
            OpenModel | ImportMaps | LoadHdri | ImportBrushSet | ImportBrushFolder => icons::OPEN,
            ExportMaps | ExportLayer | ExportLayerMask | ExportToZestEngine | SaveProject | SaveProjectAs => icons::EXPORT,
            OpenProject | LoadMaskImage => icons::OPEN,
            Undo => icons::UNDO,
            Redo => icons::REDO,
            Shortcuts => icons::KEYBOARD,
            Preferences | UndoBudget => icons::SETTINGS,
            DisplayLighting => icons::ENVIRONMENT,
            NewPaintLayer => icons::ADD,
            NewFillImage => icons::IMAGE,
            NewFromLibrary | SaveMaterial => icons::MATERIAL,
            Duplicate => icons::CLONE,
            DeleteLayer => icons::DELETE,
            BakeAo | BakeMeshMaps | BakeAll => icons::BAKE,
            BrowseMaterials | BrowseBrushes | MaterialLibraryWindow | BrushLibraryWindow => icons::LIBRARY,
            _ => return None,
        })
    }

    /// Shortcut text shown beside the command.
    pub fn shortcut_text(self) -> &'static str {
        use Cmd::*;
        match self {
            OpenModel => "Ctrl O",
            OpenProject => "Ctrl Shift O",
            SaveProject => "Ctrl S",
            SaveProjectAs => "Ctrl Shift S",
            ImportMaps => "Ctrl Shift I",
            ExportMaps => "Ctrl Shift E",
            Quit => "Ctrl Q",
            Undo => "Ctrl Z",
            Redo => "Ctrl Shift Z",
            Shortcuts => "F1",
            Studio => "F10",
            RenderImage => "F12",
            Preferences => "Ctrl ,",
            Hide => "H",
            HideOthers => "Shift H",
            UnhideAll => "Alt H",
            Isolate => "/",
            SelectAll => "A",
            DeselectAll => "Alt A",
            FrameSelected => "F",
            FrameAll => "Home",
            NewPaintLayer => "Ctrl Shift N",
            Duplicate => "Ctrl J",
            Rename => "F2",
            GroupFolder => "Ctrl G",
            MoveUp => "Ctrl ]",
            MoveDown => "Ctrl [",
            MergeDown => "Ctrl E",
            ToggleClip => "Ctrl Alt G",
            DeleteLayer => "Del",
            BrowseMaterials => "M",
            BrowseBrushes => "Shift B",
            ToggleBrushDock => "F5",
            ToggleTextureSets => "F6",
            ToggleLayers => "F7",
            ToggleProperties => "F8",
            ToggleAssistant => "F9",
            Tool(tool) => super::shell::tool_key(tool),
            CommandSearch => "Ctrl K",
            _ => "",
        }
    }

    /// A tick beside toggles and radio items.
    pub fn checked(self, app: &AppState) -> Option<bool> {
        use Cmd::*;
        Some(match self {
            RenderRaster => !app.pt_mode,
            RenderPathTraced => app.pt_mode,
            Studio => app.ui.studio_open,
            UvChecker => app.ui.uv_checker,
            ToggleClip => app.texture_sets[app.active_set].layers.active_layer().clip,
            ToggleMaskEnabled => {
                let layer = app.texture_sets[app.active_set].layers.active_layer();
                layer.mask.is_some() && layer.mask_enabled
            }
            MaskOnModel => app.ui.mask_overlay,
            OverrideMaterial => app.ui.override_material,
            ToggleBrushDock => app.ui.panels.brush,
            ToggleTextureSets => app.ui.panels.texture_sets,
            ToggleLayers => app.ui.panels.layers,
            ToggleProperties => app.ui.panels.properties,
            ToggleAssistant => app.ui.panels.assistant,
            _ => return None,
        })
    }

    pub fn enabled(self, app: &AppState) -> bool {
        use Cmd::*;
        if self.planned() {
            return false;
        }
        let layers = &app.texture_sets[app.active_set].layers;
        let layer = layers.active_layer();
        match self {
            Undo => app.ui.history.can_undo(),
            Redo => app.ui.history.can_redo(),
            DeleteLayer => layers.layers.len() > 1,
            MoveUp => layers.active + 1 < layers.layers.len(),
            MoveDown => layers.active > 0,
            AddBlackMask | AddWhiteMask => layer.mask.is_none(),
            InvertMask | RemoveMask | ExportLayerMask | ToggleMaskEnabled | ApplyMask => layer.mask.is_some(),
            MergeDown => layers.merge_down_blocker(layers.active).is_none(),
            ToggleClip => layers.active > 0,
            DeselectAll => !app.selection.is_empty(),
            _ => true,
        }
    }
}

/// The new shortcuts this table owns (tool keys, Ctrl Z/Y and the Select
/// tool's H keys stay where they always were). Checked against the existing
/// keymap: none of these were bound before.
pub fn shortcuts() -> Vec<(KeyboardShortcut, Cmd)> {
    let ctrl = Modifiers::COMMAND;
    let ctrl_shift = Modifiers::COMMAND | Modifiers::SHIFT;
    vec![
        // Shift variants first: consume_shortcut lets extra Shift through.
        (KeyboardShortcut::new(ctrl_shift, Key::I), Cmd::ImportMaps),
        (KeyboardShortcut::new(ctrl_shift, Key::E), Cmd::ExportMaps),
        (KeyboardShortcut::new(ctrl_shift, Key::N), Cmd::NewPaintLayer),
        (KeyboardShortcut::new(ctrl_shift, Key::S), Cmd::SaveProjectAs),
        (KeyboardShortcut::new(ctrl_shift, Key::O), Cmd::OpenProject),
        (KeyboardShortcut::new(ctrl, Key::S), Cmd::SaveProject),
        (KeyboardShortcut::new(ctrl, Key::O), Cmd::OpenModel),
        (KeyboardShortcut::new(ctrl, Key::Q), Cmd::Quit),
        (KeyboardShortcut::new(ctrl, Key::Comma), Cmd::Preferences),
        (KeyboardShortcut::new(ctrl | Modifiers::ALT, Key::G), Cmd::ToggleClip),
        (KeyboardShortcut::new(ctrl, Key::E), Cmd::MergeDown),
        (KeyboardShortcut::new(ctrl, Key::J), Cmd::Duplicate),
        (KeyboardShortcut::new(ctrl, Key::CloseBracket), Cmd::MoveUp),
        (KeyboardShortcut::new(ctrl, Key::OpenBracket), Cmd::MoveDown),
        (KeyboardShortcut::new(ctrl, Key::K), Cmd::CommandSearch),
        (KeyboardShortcut::new(Modifiers::NONE, Key::F1), Cmd::Shortcuts),
        (KeyboardShortcut::new(Modifiers::NONE, Key::F2), Cmd::Rename),
        (KeyboardShortcut::new(Modifiers::NONE, Key::F5), Cmd::ToggleBrushDock),
        (KeyboardShortcut::new(Modifiers::NONE, Key::F6), Cmd::ToggleTextureSets),
        (KeyboardShortcut::new(Modifiers::NONE, Key::F7), Cmd::ToggleLayers),
        (KeyboardShortcut::new(Modifiers::NONE, Key::F8), Cmd::ToggleProperties),
        (KeyboardShortcut::new(Modifiers::NONE, Key::F9), Cmd::ToggleAssistant),
        (KeyboardShortcut::new(Modifiers::NONE, Key::F10), Cmd::Studio),
        (KeyboardShortcut::new(Modifiers::NONE, Key::F12), Cmd::RenderImage),
        (KeyboardShortcut::new(Modifiers::NONE, Key::Delete), Cmd::DeleteLayer),
        (KeyboardShortcut::new(Modifiers::SHIFT, Key::B), Cmd::BrowseBrushes),
        (KeyboardShortcut::new(Modifiers::NONE, Key::M), Cmd::BrowseMaterials),
    ]
}

impl AppState {
    /// The command whose shortcut was pressed this frame, if any. Skipped
    /// while a text field has focus, so typing never triggers commands.
    pub(super) fn pressed_command(&self, ctx: &egui::Context) -> Option<Cmd> {
        if ctx.wants_keyboard_input() {
            // Ctrl K still opens the palette from a text field, and saving
            // works wherever the focus is.
            let save_as = KeyboardShortcut::new(Modifiers::COMMAND | Modifiers::SHIFT, Key::S);
            let save = KeyboardShortcut::new(Modifiers::COMMAND, Key::S);
            let k = KeyboardShortcut::new(Modifiers::COMMAND, Key::K);
            return ctx.input_mut(|i| {
                if i.consume_shortcut(&save_as) {
                    Some(Cmd::SaveProjectAs)
                } else if i.consume_shortcut(&save) {
                    Some(Cmd::SaveProject)
                } else {
                    i.consume_shortcut(&k).then_some(Cmd::CommandSearch)
                }
            });
        }
        // Exact modifiers: Shift B is a different command from B (the Paint
        // tool), and Ctrl Shift N must not also count as a plain Ctrl N.
        shortcuts().into_iter().find_map(|(shortcut, cmd)| {
            ctx.input_mut(|i| {
                let hit = i.modifiers.matches_exact(shortcut.modifiers)
                    && i.key_pressed(shortcut.logical_key);
                if hit {
                    i.consume_key(shortcut.modifiers, shortcut.logical_key);
                }
                hit.then_some(cmd)
            })
        })
    }

    pub(super) fn run_command(&mut self, cmd: Cmd, ctx: &egui::Context) {
        use Cmd::*;
        if !cmd.enabled(self) {
            return;
        }
        match cmd {
            OpenModel => self.open_model_dialog(),
            OpenProject => self.open_project_dialog(),
            SaveProject => self.save_project(false),
            SaveProjectAs => self.save_project(true),
            ImportMaps => self.pick_import_maps(),
            ExportMaps => self.ui.export_open = true,
            ExportToZestEngine => self.export_to_zestengine(),
            LoadHdri => self.open_hdri_dialog(),
            Quit => ctx.send_viewport_cmd(egui::ViewportCommand::Close),
            Undo => self.undo(),
            Redo => self.redo(),
            UndoBudget | Preferences => self.ui.prefs_open = true,
            Shortcuts => self.ui.shortcuts_open = true,
            RenderRaster => self.pt_mode = false,
            RenderPathTraced => {
                if !self.pt_mode {
                    self.pt_mode = true;
                    self.pt_scene_stale = true;
                }
            }
            Studio => {
                self.ui.studio_open = !self.ui.studio_open;
                if self.ui.studio_open && !self.pt_mode {
                    self.pt_mode = true;
                    self.pt_scene_stale = true;
                }
            }
            StudioPose | StudioLights | StudioSet | StudioCamera => {
                self.ui.studio_open = true;
                self.ui.studio_tab = match self_cmd_tab(cmd) {
                    Some(tab) => tab,
                    None => self.ui.studio_tab,
                };
                if !self.pt_mode {
                    self.pt_mode = true;
                    self.pt_scene_stale = true;
                }
            }
            LastRender => self.ui.render_result_open = true,
            RenderImage => {
                self.ui.studio_open = true;
                self.ui.studio_tab = super::studio_panel::StudioTab::Render;
                self.start_render();
            }
            DisplayLighting => self.ui.display_open = true,
            UvChecker => self.ui.uv_checker = !self.ui.uv_checker,
            MaskOnModel => self.ui.mask_overlay = !self.ui.mask_overlay,
            OverrideMaterial => self.ui.override_material = !self.ui.override_material,
            Hide => self.hide_selected(),
            HideOthers => self.hide_unselected(),
            UnhideAll => self.unhide_all(),
            Isolate => self.toggle_isolate(),
            DeletePieces => self.delete_pieces(&self.selection_or_active()),
            SelectAll => self.select_all(),
            DeselectAll => self.deselect_all(),
            FrameSelected => self.frame_selection(),
            FrameAll => self.frame_all(),
            NewPaintLayer => self.texture_sets[self.active_set].layers.add_layer(),
            NewFillImage => self.fill_layer_from_image_dialog(),
            NewFromLibrary => self.open_library(false),
            NewShaderMask(kind) => self.add_shader_mask(kind),
            Duplicate => {
                let layers = &mut self.texture_sets[self.active_set].layers;
                let active = layers.active;
                layers.duplicate_layer(active);
                self.ui.status = format!("Duplicated '{}'", layers.active_layer().name);
            }
            Rename => {
                let layers = &self.texture_sets[self.active_set].layers;
                self.ui.renaming = Some((layers.active, layers.active_layer().name.clone()));
                self.ui.rename_fresh = true;
            }
            MoveUp => {
                let layers = &mut self.texture_sets[self.active_set].layers;
                let i = layers.active;
                layers.move_layer(i, i + 1);
            }
            MoveDown => {
                let layers = &mut self.texture_sets[self.active_set].layers;
                let i = layers.active;
                layers.move_layer(i, i - 1);
            }
            DeleteLayer => {
                let layers = &mut self.texture_sets[self.active_set].layers;
                let active = layers.active;
                layers.remove_layer(active);
            }
            MergeDown => {
                let layers = &mut self.texture_sets[self.active_set].layers;
                let active = layers.active;
                let upper = layers.layers[active].name.clone();
                if layers.merge_down(active) {
                    self.ui.status = format!("Merged '{upper}' into '{}'", layers.active_layer().name);
                }
            }
            ToggleClip => {
                let layers = &mut self.texture_sets[self.active_set].layers;
                let active = layers.active;
                layers.layers[active].clip = !layers.layers[active].clip;
                layers.mark_dirty();
            }
            ToggleMaskEnabled => {
                let layers = &mut self.texture_sets[self.active_set].layers;
                let active = layers.active;
                layers.layers[active].mask_enabled = !layers.layers[active].mask_enabled;
                layers.mark_dirty();
            }
            ApplyMask => {
                let layers = &mut self.texture_sets[self.active_set].layers;
                let active = layers.active;
                if layers.apply_mask(active) {
                    self.ui.status = format!("Mask applied to '{}'", layers.active_layer().name);
                }
            }
            AddBlackMask | AddWhiteMask => {
                let layers = &mut self.texture_sets[self.active_set].layers;
                let active = layers.active;
                layers.add_mask(active, if cmd == AddWhiteMask { 1.0 } else { 0.0 });
                self.ui.props_tab = PropsTab::Mask;
            }
            InvertMask => {
                let layers = &mut self.texture_sets[self.active_set].layers;
                let active = layers.active;
                layers.invert_mask(active);
            }
            RemoveMask => {
                let layers = &mut self.texture_sets[self.active_set].layers;
                let active = layers.active;
                layers.remove_mask(active);
            }
            LoadMaskImage => self.load_mask_from_image(),
            ExportLayer | ExportLayerMask => {
                let active = self.texture_sets[self.active_set].layers.active;
                if cmd == ExportLayer {
                    self.export_layer_dialog(active, true, true);
                } else {
                    self.export_layer_dialog(active, false, true);
                }
            }
            FillFromBrush => self.fill_from_brush_values(),
            BakeAo => self.bake_ao(),
            BakeMeshMaps => self.bake_mesh_maps(),
            BakeAll => {
                self.bake_ao();
                self.bake_mesh_maps();
                self.ui.status = "Baked AO and mesh maps".into();
            }
            AutoUnwrap => self.ui.confirm_unwrap = true,
            BrowseMaterials | MaterialLibraryWindow => self.open_library(false),
            BrowseBrushes | BrushLibraryWindow => self.open_library(true),
            SaveMaterial => {
                self.ui.props_tab = PropsTab::Material;
                self.ui.panels.properties = true;
                self.ui.focus_save_name = true;
            }
            ImportBrushSet => {
                self.import_brush_files();
            }
            ImportBrushFolder => {
                self.import_brush_folder();
            }
            RescanLibraries => {
                self.ui.library_scanned = false;
                self.ui.brushes_scanned = false;
                self.ui.status = "Libraries will rescan".into();
            }
            OpenLibraryFolder => {
                let dir = crate::library::library_dir();
                let _ = std::fs::create_dir_all(&dir);
                match std::process::Command::new("xdg-open").arg(&dir).spawn() {
                    Ok(_) => self.ui.status = format!("Opened {}", dir.display()),
                    Err(e) => self.ui.status = format!("Could not open {}: {e}", dir.display()),
                }
            }
            ToggleBrushDock => self.ui.panels.brush = !self.ui.panels.brush,
            ToggleTextureSets => self.ui.panels.texture_sets = !self.ui.panels.texture_sets,
            ToggleLayers => self.ui.panels.layers = !self.ui.panels.layers,
            ToggleProperties => self.ui.panels.properties = !self.ui.panels.properties,
            ToggleAssistant => self.ui.panels.assistant = !self.ui.panels.assistant,
            ResetLayout => {
                self.ui.panels = super::Panels::default();
                self.ui.props_tab = PropsTab::Layer;
                self.ui.texture_sets_open = true;
                ctx.memory_mut(|m| m.reset_areas());
            }
            ToolGestures => self.ui.gestures_open = true,
            About => self.ui.about_open = true,
            Welcome => self.ui.welcome_open = true,
            Tool(tool) => self.select_tool(tool),
            CommandSearch => {
                self.ui.palette.open = true;
                self.ui.palette.query.clear();
                self.ui.palette.selected = 0;
            }
            GroupFolder | TextureView2d => {}
        }
    }
}

/// The Studio tab a Studio menu command opens.
fn self_cmd_tab(cmd: Cmd) -> Option<super::studio_panel::StudioTab> {
    use super::studio_panel::StudioTab;
    Some(match cmd {
        Cmd::StudioPose => StudioTab::Pose,
        Cmd::StudioLights => StudioTab::Lights,
        Cmd::StudioSet => StudioTab::Set,
        Cmd::StudioCamera => StudioTab::Camera,
        _ => return None,
    })
}
