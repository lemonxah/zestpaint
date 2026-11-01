mod ao;
mod app;
mod assistant;
mod config;
mod bakers;
mod brushfile;
mod camera;
mod zestengine;
mod connectivity;
mod environment;
mod export;
mod fbx_loader;
mod gltf_loader;
mod icons;
mod layers;
mod library;
mod matcap;
mod mesh;
mod paint;
mod pathtracer;
mod poses;
mod project;
mod renderer;
mod scene;
mod skeleton;
mod studio;
mod texcompress;
mod theme;
mod uv_tools;
mod vram;

fn main() -> Result<(), winit::error::EventLoopError> {
    // Before anything reads a settings file (the MCP server reads the
    // bridge's discovery file too).
    config::migrate_legacy();
    library::migrate_library();
    // `zestsurface mcp`: the MCP server Claude Code launches. It forwards tool
    // calls to the running editor and never opens a window itself.
    if std::env::args().nth(1).as_deref() == Some("mcp") {
        if let Err(e) = assistant::mcp::serve_stdio() {
            eprintln!("zestsurface mcp: {e:#}");
            std::process::exit(1);
        }
        return Ok(());
    }
    let mut zestengine_manifest: Option<std::path::PathBuf> = None;
    let mut path: Option<std::path::PathBuf> = None;
    let mut args = std::env::args_os().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--zestengine" {
            zestengine_manifest = args.next().map(Into::into);
        } else if path.is_none() {
            path = Some(arg.into());
        }
    }
    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    let mut app = app::App::new();
    app.start(path, zestengine_manifest);
    event_loop.run_app(&mut app)
}
