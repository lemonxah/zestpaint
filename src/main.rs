mod ao;
mod app;
mod assistant;
mod config;
mod bakers;
mod brushfile;
mod camera;
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
    // `zestpaint mcp`: the MCP server Claude Code launches. It forwards tool
    // calls to the running editor and never opens a window itself.
    if std::env::args().nth(1).as_deref() == Some("mcp") {
        if let Err(e) = assistant::mcp::serve_stdio() {
            eprintln!("zestpaint mcp: {e:#}");
            std::process::exit(1);
        }
        return Ok(());
    }
    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    let mut app = app::App::new();
    // `zestpaint file.zestpaint` opens a project; otherwise the startup
    // setting decides (welcome screen, last project or the sphere).
    app.start(std::env::args_os().nth(1).map(Into::into));
    event_loop.run_app(&mut app)
}
