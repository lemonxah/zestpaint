//! The AI assistant: a chat with Claude or Gemini that operates the editor.
//!
//! A conversation turn runs on a worker thread (`worker::start_turn`) that
//! streams from the Claude API and executes tool calls. Tools that only touch
//! the disk or the network (catalog search, downloads, procedural stamps,
//! SVG, Nano Banana and ComfyUI image generation) run on the worker; tools that change the
//! document are sent to the UI thread as an [`EditorCall`] and answered there,
//! because the GPU state and `AppState` belong to the UI thread.

pub mod api;
pub mod bridge;
pub mod comfy;
pub mod gemini;
pub mod images;
pub mod mcp;
pub mod tools;
pub mod worker;

use std::path::{Path, PathBuf};
use std::sync::mpsc;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

pub const DEFAULT_MODEL: &str = "claude-opus-5-5";
pub const MODELS: [&str; 6] = [
    "claude-opus-5-5",
    "claude-sonnet-5-5",
    "claude-haiku-5-5",
    "gemini-4-argon",
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
];
/// Nano Banana image models (Gemini API), newest first.
pub const IMAGE_MODELS: [&str; 4] = ["gemini-nano-banana-2.1", "gemini-3.1-flash-image", "gemini-3.1-flash-lite-image", "gemini-3-pro-image"];
/// Where `generate_image_stamp` makes its images: "auto" uses Nano Banana
/// when a Gemini key is set, else ComfyUI.
pub const IMAGE_BACKENDS: [(&str, &str); 3] = [("auto", "Auto"), ("gemini", "Nano Banana (Gemini)"), ("comfyui", "ComfyUI (local)")];
pub const EFFORTS: [&str; 5] = ["low", "medium", "high", "xhigh", "max"];

/// Persisted in `~/.config/zestpaint/assistant.json`.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(default)]
pub struct Settings {
    /// Empty means "use $ANTHROPIC_API_KEY".
    pub api_key: String,
    /// API origin; empty means $ANTHROPIC_BASE_URL, else api.anthropic.com.
    /// For proxies and gateways (and the tests' mock server).
    pub api_base: String,
    pub model: String,
    pub effort: String,
    /// Gemini key; empty means $GEMINI_API_KEY, else $GOOGLE_API_KEY.
    pub gemini_api_key: String,
    /// Gemini API origin; empty means generativelanguage.googleapis.com.
    pub gemini_api_base: String,
    /// "auto", "gemini" or "comfyui" (see `IMAGE_BACKENDS`).
    pub image_backend: String,
    /// Nano Banana model for image generation.
    pub image_model: String,
    /// Base URL of a local ComfyUI server, for raster image generation.
    pub comfy_url: String,
    /// Checkpoint for the built-in txt2img graph; empty picks the first one.
    pub comfy_checkpoint: String,
    /// Send a viewport capture with every message.
    pub attach_viewport: bool,
    /// Send the active texture set's base colour with every message.
    pub attach_textures: bool,
    /// Run the MCP server Claude Code drives the editor through (the HTTP
    /// endpoint, and the bridge `zestpaint mcp` forwards to).
    pub claude_code_bridge: bool,
    /// Port of the HTTP MCP endpoint, fixed so Claude Code's registration
    /// keeps working across restarts.
    pub mcp_port: u16,
    /// Bearer token for the HTTP endpoint; made on first start.
    pub mcp_token: String,
}

pub const DEFAULT_MCP_PORT: u16 = 47821;

impl Default for Settings {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            api_base: String::new(),
            model: DEFAULT_MODEL.into(),
            effort: "medium".into(),
            gemini_api_key: String::new(),
            gemini_api_base: String::new(),
            image_backend: "auto".into(),
            image_model: IMAGE_MODELS[0].into(),
            comfy_url: "http://127.0.0.1:8188".into(),
            comfy_checkpoint: String::new(),
            attach_viewport: true,
            attach_textures: true,
            claude_code_bridge: true,
            mcp_port: DEFAULT_MCP_PORT,
            mcp_token: String::new(),
        }
    }
}

impl Settings {
    pub fn load(path: &Path) -> Self {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_default()
    }

    /// The file can hold an API key, so it is created owner-only.
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let text = serde_json::to_string_pretty(self)?;
        #[cfg(unix)]
        {
            use std::io::Write as _;
            use std::os::unix::fs::OpenOptionsExt as _;
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .mode(0o600)
                .open(path)?;
            file.write_all(text.as_bytes())?;
        }
        #[cfg(not(unix))]
        std::fs::write(path, text)?;
        Ok(())
    }

    pub fn messages_url(&self) -> String {
        let base = Some(self.api_base.trim().to_string())
            .filter(|b| !b.is_empty())
            .or_else(|| std::env::var("ANTHROPIC_BASE_URL").ok().filter(|b| !b.trim().is_empty()))
            .unwrap_or_else(|| "https://api.anthropic.com".into());
        format!("{}/v1/messages", base.trim_end_matches('/'))
    }

    /// Where the chosen model's requests go.
    pub fn request_url(&self) -> String {
        if gemini::is_gemini(&self.model) {
            gemini::url(&self.gemini_api_base, &self.model, "streamGenerateContent")
        } else {
            self.messages_url()
        }
    }

    /// The key for the chosen model's API.
    pub fn resolved_key(&self) -> Option<String> {
        if gemini::is_gemini(&self.model) { self.resolved_gemini_key() } else { self.resolved_anthropic_key() }
    }

    /// The Claude key: the settings field, else the environment.
    pub fn resolved_anthropic_key(&self) -> Option<String> {
        let key = self.api_key.trim();
        if !key.is_empty() {
            return Some(key.to_string());
        }
        std::env::var("ANTHROPIC_API_KEY").ok().filter(|k| !k.trim().is_empty())
    }

    /// The Gemini key: the settings field, else $GEMINI_API_KEY / $GOOGLE_API_KEY.
    pub fn resolved_gemini_key(&self) -> Option<String> {
        let key = self.gemini_api_key.trim();
        if !key.is_empty() {
            return Some(key.to_string());
        }
        ["GEMINI_API_KEY", "GOOGLE_API_KEY"]
            .into_iter()
            .find_map(|var| std::env::var(var).ok().filter(|k| !k.trim().is_empty()))
    }

    /// Whether image generation should go to Nano Banana (else ComfyUI).
    pub fn images_via_gemini(&self) -> bool {
        match self.image_backend.as_str() {
            "gemini" => true,
            "comfyui" => false,
            _ => self.resolved_gemini_key().is_some(),
        }
    }
}

/// The real directory holding `materials/` and `brushes/`. The
/// `~/.zestpaint/library` entries may be symlinks onto another filesystem, and
/// matfetch stages downloads next to the root and renames them into place — a
/// rename across filesystems fails, so resolve the links first.
pub fn library_root() -> PathBuf {
    let materials = crate::library::library_dir();
    let _ = std::fs::create_dir_all(&materials);
    let _ = std::fs::create_dir_all(crate::library::brush_dir());
    let real = std::fs::canonicalize(&materials).unwrap_or(materials);
    real.parent().map(Path::to_path_buf).unwrap_or(real)
}

/// What a tool hands back to Claude: text and image content blocks.
#[derive(Clone, Debug)]
pub struct ToolOutput {
    pub content: Vec<Value>,
    pub is_error: bool,
    /// Shown in the chat next to the tool row (a generated stamp, a capture).
    pub preview: Option<image::RgbaImage>,
}

impl ToolOutput {
    pub fn text(text: impl Into<String>) -> Self {
        Self { content: vec![json!({"type": "text", "text": text.into()})], is_error: false, preview: None }
    }

    pub fn error(text: impl Into<String>) -> Self {
        Self { is_error: true, ..Self::text(text) }
    }

    pub fn json(value: &Value) -> Self {
        Self::text(serde_json::to_string_pretty(value).unwrap_or_default())
    }

    pub fn with_image(mut self, block: Value) -> Self {
        self.content.push(block);
        self
    }

    pub fn with_preview(mut self, image: image::RgbaImage) -> Self {
        self.preview = Some(image);
        self
    }

    /// One line for the chat's tool row.
    pub fn summary(&self) -> String {
        let text = self
            .content
            .iter()
            .find_map(|b| b.get("text").and_then(Value::as_str))
            .unwrap_or("");
        let line = text.lines().next().unwrap_or("").trim();
        if line.chars().count() > 120 {
            format!("{}…", line.chars().take(120).collect::<String>())
        } else {
            line.to_string()
        }
    }
}

/// A document-changing tool call, executed on the UI thread.
pub struct EditorCall {
    pub name: String,
    pub input: Value,
    pub reply: mpsc::Sender<ToolOutput>,
}

/// Worker → UI.
pub enum Event {
    /// Visible assistant text, streamed.
    Text(String),
    /// Summarised thinking, streamed.
    Thinking(String),
    ToolStart { id: String, name: String, detail: String },
    ToolDone { id: String, output: ToolOutput },
    /// A server-side tool (web search / fetch) ran.
    ServerTool { name: String, detail: String },
    Editor(EditorCall),
    /// Something was written into the asset library.
    LibraryChanged,
    Usage { input: u64, output: u64, cache_read: u64 },
    Notice(String),
    Error(String),
}
