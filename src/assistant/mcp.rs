//! `zestsurface mcp`: an MCP server on stdio for Claude Code.
//!
//! It holds no document state. Every `tools/call` is forwarded to the running
//! editor over the loopback bridge (`bridge.rs`), so Claude Code drives the
//! open app with the same tools the built-in assistant uses — no API key
//! needed, the Claude Code session does the thinking.
//!
//! stdout carries JSON-RPC only; anything human-readable goes to stderr.

use std::io::{BufRead as _, BufReader, Write as _};
use std::net::{Ipv4Addr, TcpStream};
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result};
use serde_json::{Value, json};

use super::bridge::{self, Discovery};
use super::{ToolOutput, tools, worker};

const PROTOCOLS: [&str; 4] = ["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"];

/// MCP-only: start the editor when it isn't running.
const OPEN_TOOL: &str = "open_zestsurface";

/// For the editor's own HTTP endpoint (`bridge.rs`): it is running by definition.
const IN_APP_INTRO: &str = "These tools operate ZestSurface, a Substance-Painter-style 3D texture painting app running on the user's desktop. Start with get_editor_state. Use capture_viewport to see the model and check your changes. The user does the brush work in the viewport — tell them exactly what to click next.";

const INSTRUCTIONS_INTRO: &str = "These tools operate ZestSurface, a Substance-Painter-style 3D texture painting app running on the user's desktop. Start with get_editor_state; if the editor isn't running, call open_zestsurface. Use capture_viewport to see the model and check your changes. The user does the brush work in the viewport — tell them exactly what to click next.";

pub fn serve_stdio() -> Result<()> {
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout().lock();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let reply = match serde_json::from_str::<Value>(&line) {
            Ok(msg) => handle(&msg),
            Err(e) => Some(json!({"jsonrpc": "2.0", "id": null, "error": {"code": -32700, "message": format!("parse error: {e}")}})),
        };
        if let Some(reply) = reply {
            writeln!(stdout, "{reply}")?;
            stdout.flush()?;
        }
    }
    Ok(())
}

/// One JSON-RPC message → its response (None for notifications).
fn handle(msg: &Value) -> Option<Value> {
    handle_with(msg, None)
}

/// Like [`handle`], but `in_app` runs tool calls directly — the editor's own
/// HTTP endpoint — instead of forwarding them over the bridge, and
/// `open_zestsurface` is left out.
pub(super) fn handle_with(msg: &Value, in_app: Option<&dyn Fn(&str, &Value) -> ToolOutput>) -> Option<Value> {
    let id = msg.get("id")?.clone();
    let params = &msg["params"];
    let result = match msg["method"].as_str().unwrap_or("") {
        "initialize" => {
            let asked = params["protocolVersion"].as_str().unwrap_or("");
            let version = PROTOCOLS.iter().find(|v| **v == asked).copied().unwrap_or(PROTOCOLS[1]);
            Ok(json!({
                "protocolVersion": version,
                "capabilities": {"tools": {"listChanged": false}},
                "serverInfo": {"name": "zestsurface", "version": env!("CARGO_PKG_VERSION")},
                "instructions": format!("{}\n\n{}", if in_app.is_some() { IN_APP_INTRO } else { INSTRUCTIONS_INTRO }, worker::EDITOR_GUIDE),
            }))
        }
        "ping" => Ok(json!({})),
        "tools/list" => Ok(json!({"tools": tool_list(in_app.is_none())})),
        "tools/call" => {
            let name = params["name"].as_str().unwrap_or("");
            let args = params.get("arguments").cloned().unwrap_or_else(|| json!({}));
            let (content, is_error) = if let Some(run) = in_app {
                let out = run(name, &args);
                (out.content, out.is_error)
            } else if name == OPEN_TOOL {
                match open_editor(&args) {
                    Ok(text) => (vec![json!({"type": "text", "text": text})], false),
                    Err(e) => (vec![json!({"type": "text", "text": format!("{e:#}")})], true),
                }
            } else {
                match forward(name, &args) {
                    Ok(reply) => (reply["content"].as_array().cloned().unwrap_or_default(), reply["is_error"].as_bool().unwrap_or(false)),
                    Err(e) => (vec![json!({"type": "text", "text": format!("{e:#}")})], true),
                }
            };
            Ok(json!({"content": content.iter().map(to_mcp_block).collect::<Vec<_>>(), "isError": is_error}))
        }
        other => Err((-32601, format!("method not found: {other}"))),
    };
    Some(match result {
        Ok(result) => json!({"jsonrpc": "2.0", "id": id, "result": result}),
        Err((code, message)) => json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}),
    })
}

/// The assistant's custom tools in MCP shape. The API's server tools (web
/// search / fetch) are left out — Claude Code has its own.
fn tool_list(with_open: bool) -> Vec<Value> {
    const READ_ONLY: &[&str] = &["get_editor_state", "capture_viewport", "list_library", "search_online_assets"];
    let mut list: Vec<Value> = tools::definitions()
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|d| {
            let name = d["name"].as_str()?;
            Some(json!({
                "name": name,
                "description": d["description"],
                "inputSchema": d.get("input_schema")?,
                "annotations": {"readOnlyHint": READ_ONLY.contains(&name)},
            }))
        })
        .collect();
    if !with_open {
        return list;
    }
    list.push(json!({
        "name": OPEN_TOOL,
        "description": "Start the ZestSurface editor if it isn't running (optionally opening a .zestsurface project) and wait until its tools are reachable. Does nothing when it is already running.",
        "inputSchema": {"type": "object", "properties": {
            "project": {"type": "string", "description": "Absolute path of a .zestsurface project to open."}
        }},
    }));
    list
}

/// API content block → MCP content block. Text is the same shape; images move
/// from `source.{media_type,data}` to `{mimeType,data}`.
fn to_mcp_block(block: &Value) -> Value {
    match block["type"].as_str() {
        Some("image") => json!({"type": "image", "data": block["source"]["data"], "mimeType": block["source"]["media_type"]}),
        Some("text") => json!({"type": "text", "text": block["text"]}),
        _ => json!({"type": "text", "text": block.to_string()}),
    }
}

fn discovery() -> Option<Discovery> {
    serde_json::from_str(&std::fs::read_to_string(bridge::discovery_path()).ok()?).ok()
}

fn not_running() -> anyhow::Error {
    anyhow::anyhow!(
        "ZestSurface isn't running, or its Claude Code bridge is off. Call {OPEN_TOOL} to start it (or start it yourself; \
         the bridge is under Assistant ▸ settings), then retry."
    )
}

fn connect_to(found: &Discovery) -> std::io::Result<TcpStream> {
    TcpStream::connect_timeout(&(Ipv4Addr::LOCALHOST, found.port).into(), Duration::from_secs(2))
}

fn running() -> bool {
    discovery().is_some_and(|found| connect_to(&found).is_ok())
}

fn forward(name: &str, args: &Value) -> Result<Value> {
    let found = discovery().ok_or_else(not_running)?;
    forward_to(&found, name, args)
}

pub(super) fn forward_to(found: &Discovery, name: &str, args: &Value) -> Result<Value> {
    let stream = connect_to(found).map_err(|_| not_running())?;
    let mut writer = stream.try_clone()?;
    writeln!(writer, "{}", json!({"token": found.token, "name": name, "input": args}))?;
    writer.flush()?;
    let mut line = String::new();
    BufReader::new(stream).read_line(&mut line)?;
    anyhow::ensure!(!line.trim().is_empty(), "ZestSurface closed the connection (did it quit?)");
    serde_json::from_str(&line).context("bad reply from ZestSurface")
}

fn open_editor(args: &Value) -> Result<String> {
    if running() {
        return Ok("ZestSurface is already running.".into());
    }
    let exe = std::env::current_exe().context("can't locate the zestsurface executable")?;
    let mut cmd = std::process::Command::new(exe);
    if let Some(project) = args["project"].as_str().filter(|p| !p.trim().is_empty()) {
        cmd.arg(project);
    }
    // stdout is the JSON-RPC channel: the editor must not inherit it.
    cmd.stdin(std::process::Stdio::null()).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null());
    // Its own process group, so it outlives this server and Claude Code.
    #[cfg(unix)]
    std::os::unix::process::CommandExt::process_group(&mut cmd, 0);
    let mut child = cmd.spawn().context("failed to start ZestSurface")?;
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(60) {
        if running() {
            return Ok(format!("ZestSurface started (pid {}) and is ready.", child.id()));
        }
        if let Some(status) = child.try_wait()? {
            anyhow::bail!("ZestSurface exited during startup ({status})");
        }
        std::thread::sleep(Duration::from_millis(250));
    }
    anyhow::bail!("ZestSurface started but its bridge didn't come up within 60 s — is the Claude Code bridge enabled in the assistant settings?")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lists_tools_in_mcp_shape() {
        let init = handle(&json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}})).unwrap();
        assert_eq!(init["result"]["protocolVersion"], "2025-06-18");
        assert!(init["result"]["instructions"].as_str().unwrap().contains("texture sets"));
        assert!(handle(&json!({"jsonrpc": "2.0", "method": "notifications/initialized"})).is_none());

        let list = handle(&json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})).unwrap();
        let tools = list["result"]["tools"].as_array().unwrap();
        assert!(tools.iter().all(|t| t["inputSchema"]["type"] == "object"));
        assert!(!tools.iter().any(|t| t["name"] == "web_search"));
        assert!(tools.iter().any(|t| t["name"] == "set_brush"));
        assert!(tools.iter().any(|t| t["name"] == OPEN_TOOL));
    }

    #[test]
    fn image_blocks_convert() {
        let api = json!({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}});
        assert_eq!(to_mcp_block(&api), json!({"type": "image", "data": "AAAA", "mimeType": "image/png"}));
    }
}
