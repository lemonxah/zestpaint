//! The editor side of the Claude Code bridge.
//!
//! The running app listens on a loopback TCP port and answers tool calls from
//! `zestsurface mcp` (see `mcp.rs`), the MCP server Claude Code launches. The
//! port and a random token go in `~/.config/zestsurface/mcp-bridge.json` (owner-only);
//! every request must carry the token, so other local users and web pages
//! can't drive the editor.
//!
//! Wire format: one JSON object per line each way.
//! → `{"token": "...", "name": "add_layer", "input": {...}}`
//! ← `{"content": [...API content blocks...], "is_error": false}`
//!
//! The editor also serves MCP itself over HTTP (streamable HTTP, plain JSON
//! replies) at `http://127.0.0.1:<port>/mcp`, so Claude Code can connect with
//! `claude mcp add --transport http` and nothing has to be launched by hand.
//! That port is fixed (settings) so the registration survives restarts; the
//! request needs `Authorization: Bearer <token>`, and a browser `Origin` or a
//! foreign `Host` is refused (DNS rebinding).
//!
//! Calls run exactly like the in-app assistant's: editor tools go to the UI
//! thread as [`EditorCall`]s through the same [`Event`] channel, worker tools
//! run on the connection's thread.

use std::io::{BufRead as _, BufReader, Read as _, Write as _};
use std::net::{Ipv4Addr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, mpsc};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::tools::{self, WorkerCtx};
use super::{Event, Settings, ToolOutput, worker};

/// What `~/.config/zestsurface/mcp-bridge.json` holds.
#[derive(Serialize, Deserialize)]
pub struct Discovery {
    pub port: u16,
    pub token: String,
    pub pid: u32,
}

pub fn discovery_path() -> PathBuf {
    crate::config::file("mcp-bridge.json")
}

/// Where Claude Code reaches the editor's own MCP endpoint.
pub fn http_url(port: u16) -> String {
    format!("http://127.0.0.1:{port}/mcp")
}

pub struct Bridge {
    pub port: u16,
    /// The HTTP MCP endpoint's port, or why it couldn't listen (the bridge
    /// for `zestsurface mcp` works either way).
    pub http: std::result::Result<u16, String>,
    /// Editor calls and activity for the UI thread, like a turn's events.
    pub events: mpsc::Receiver<Event>,
    stop: Arc<AtomicBool>,
    http_server: Option<(Arc<tiny_http::Server>, std::thread::JoinHandle<()>)>,
    file: PathBuf,
}

impl Bridge {
    /// Listen and publish the discovery file, and serve MCP over HTTP on
    /// `http_port` with `http_token`. `settings_path` is re-read per call so
    /// worker tools see the current ComfyUI settings.
    pub fn start(settings_path: PathBuf, http_port: u16, http_token: &str) -> Result<Self> {
        Self::start_at(discovery_path(), settings_path, http_port, http_token)
    }

    fn start_at(file: PathBuf, settings_path: PathBuf, http_port: u16, http_token: &str) -> Result<Self> {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))?;
        let port = listener.local_addr()?.port();
        let token = new_token();
        write_private(
            &file,
            &serde_json::to_string_pretty(&Discovery { port, token: token.clone(), pid: std::process::id() })?,
        )?;
        let (tx, rx) = mpsc::channel();
        let stop = Arc::new(AtomicBool::new(false));
        let flag = stop.clone();
        let token = Arc::new(token);
        let settings_path = Arc::new(settings_path);
        let (http, http_server) = match serve_http(http_port, http_token, &settings_path, &tx, &stop) {
            Ok((server, thread)) => (Ok(server.server_addr().to_ip().map_or(http_port, |a| a.port())), Some((server, thread))),
            Err(e) => (Err(e), None),
        };
        std::thread::Builder::new().name("mcp-bridge".into()).spawn(move || {
            for stream in listener.incoming() {
                if flag.load(Ordering::Relaxed) {
                    break;
                }
                let Ok(stream) = stream else { continue };
                let (tx, flag, token, settings_path) = (tx.clone(), flag.clone(), token.clone(), settings_path.clone());
                let _ = std::thread::Builder::new()
                    .name("mcp-conn".into())
                    .spawn(move || serve(stream, &token, &settings_path, &tx, &flag));
            }
        })?;
        Ok(Self { port, http, events: rx, stop, http_server, file })
    }
}

impl Drop for Bridge {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        // Wake the accept loops so they see the flag.
        let _ = TcpStream::connect((Ipv4Addr::LOCALHOST, self.port));
        // Join the request loop so ours is the last handle: dropping it closes
        // the listener now, and a restart can bind the same port again.
        if let Some((server, thread)) = self.http_server.take() {
            server.unblock();
            let _ = thread.join();
            drop(server);
        }
        // Leave the file alone if another instance has taken it over since.
        let ours = std::fs::read_to_string(&self.file)
            .ok()
            .and_then(|t| serde_json::from_str::<Discovery>(&t).ok())
            .is_some_and(|d| d.pid == std::process::id() && d.port == self.port);
        if ours {
            let _ = std::fs::remove_file(&self.file);
        }
    }
}

#[derive(Deserialize)]
struct Request {
    token: String,
    name: String,
    #[serde(default)]
    input: Value,
}

fn serve(stream: TcpStream, token: &str, settings_path: &Path, tx: &mpsc::Sender<Event>, stop: &Arc<AtomicBool>) {
    let Ok(mut writer) = stream.try_clone() else { return };
    for line in BufReader::new(stream).lines() {
        let Ok(line) = line else { return };
        if line.trim().is_empty() {
            continue;
        }
        let output = match serde_json::from_str::<Request>(&line) {
            Ok(req) if req.token == token => call(&req.name, &req.input, settings_path, tx, stop),
            Ok(_) => {
                // Wrong token: answer once and hang up.
                let _ = writeln!(writer, "{}", json!({"content": [{"type": "text", "text": "bad bridge token"}], "is_error": true}));
                return;
            }
            Err(e) => ToolOutput::error(format!("bad request: {e}")),
        };
        let reply = json!({"content": output.content, "is_error": output.is_error});
        if writeln!(writer, "{reply}").and_then(|_| writer.flush()).is_err() {
            return;
        }
    }
}

type HttpServer = (Arc<tiny_http::Server>, std::thread::JoinHandle<()>);

/// Listen on `127.0.0.1:port` and answer MCP JSON-RPC posted to `/mcp`, each
/// request on its own thread (editor calls block until the UI thread answers).
fn serve_http(
    port: u16,
    token: &str,
    settings_path: &Arc<PathBuf>,
    tx: &mpsc::Sender<Event>,
    stop: &Arc<AtomicBool>,
) -> std::result::Result<HttpServer, String> {
    if token.len() < 16 {
        return Err("token too short".into());
    }
    // tiny_http closes its listener on its own thread just after a stop, so
    // a quick restart on the same port may need a moment.
    let mut tries = 0;
    let server = loop {
        match tiny_http::Server::http((Ipv4Addr::LOCALHOST, port)) {
            Ok(server) => break server,
            Err(e) => {
                let busy = e.downcast_ref::<std::io::Error>().is_some_and(|e| e.kind() == std::io::ErrorKind::AddrInUse);
                if busy && tries < 10 {
                    tries += 1;
                    std::thread::sleep(std::time::Duration::from_millis(50));
                    continue;
                }
                return Err(if busy { format!("port {port} is in use (another ZestSurface?)") } else { format!("can't listen on port {port}: {e}") });
            }
        }
    };
    let server = Arc::new(server);
    let (accept, token, settings_path, tx, stop) = (server.clone(), Arc::new(token.to_string()), settings_path.clone(), tx.clone(), stop.clone());
    let thread = std::thread::Builder::new()
        .name("mcp-http".into())
        .spawn(move || {
            while let Ok(req) = accept.recv() {
                if stop.load(Ordering::Relaxed) {
                    break;
                }
                let (token, settings_path, tx, stop) = (token.clone(), settings_path.clone(), tx.clone(), stop.clone());
                let _ = std::thread::Builder::new()
                    .name("mcp-http-req".into())
                    .spawn(move || http_request(req, &token, &settings_path, &tx, &stop));
            }
        })
        .map_err(|e| e.to_string())?;
    Ok((server, thread))
}

fn http_request(mut req: tiny_http::Request, token: &str, settings_path: &Path, tx: &mpsc::Sender<Event>, stop: &Arc<AtomicBool>) {
    let headers = req.headers().to_vec();
    let header = |name: &'static str| headers.iter().find(|h| h.field.equiv(name)).map(|h| h.value.as_str().to_string());
    let plain = |code: u16, text: &str| tiny_http::Response::from_string(text).with_status_code(code);
    let local = |host: &str| {
        let host = host.rsplit_once(':').filter(|(_, p)| p.bytes().all(|b| b.is_ascii_digit())).map_or(host, |(h, _)| h);
        matches!(host, "127.0.0.1" | "localhost" | "[::1]")
    };
    let path = req.url().split('?').next().unwrap_or("").to_string();
    let response = if path != "/mcp" {
        plain(404, "not found; the MCP endpoint is /mcp")
    } else if !header("Host").is_some_and(|h| local(&h)) {
        plain(403, "forbidden host")
    } else if header("Origin").is_some_and(|o| !local(o.split_once("://").map_or(o.as_str(), |(_, rest)| rest))) {
        plain(403, "forbidden origin")
    } else if header("Authorization").as_deref() != Some(&format!("Bearer {token}")) {
        plain(401, "missing or wrong bearer token (see ZestSurface ▸ Assistant ▸ settings)")
    } else if *req.method() != tiny_http::Method::Post {
        // No server-initiated stream: every reply comes back on its POST.
        plain(405, "POST only").with_header(tiny_http::Header::from_bytes("Allow", "POST").unwrap())
    } else {
        let mut body = String::new();
        match req.as_reader().take(64 << 20).read_to_string(&mut body) {
            Err(e) => plain(400, &format!("unreadable body: {e}")),
            Ok(_) => {
                let run = |name: &str, input: &Value| call(name, input, settings_path, tx, stop);
                let reply = match serde_json::from_str::<Value>(&body) {
                    Ok(Value::Array(batch)) => {
                        let replies: Vec<Value> = batch.iter().filter_map(|m| super::mcp::handle_with(m, Some(&run))).collect();
                        (!replies.is_empty()).then(|| Value::Array(replies))
                    }
                    Ok(msg) => super::mcp::handle_with(&msg, Some(&run)),
                    Err(e) => Some(json!({"jsonrpc": "2.0", "id": null, "error": {"code": -32700, "message": format!("parse error: {e}")}})),
                };
                match reply {
                    // Notifications and responses are accepted without a body.
                    None => plain(202, ""),
                    Some(reply) => tiny_http::Response::from_string(reply.to_string())
                        .with_header(tiny_http::Header::from_bytes("Content-Type", "application/json").unwrap()),
                }
            }
        }
    };
    let _ = req.respond(response);
}

fn call(name: &str, input: &Value, settings_path: &Path, tx: &mpsc::Sender<Event>, stop: &Arc<AtomicBool>) -> ToolOutput {
    static NEXT_ID: AtomicU64 = AtomicU64::new(1);
    let known = tools::definitions()
        .as_array()
        .is_some_and(|defs| defs.iter().any(|d| d["name"] == name && d.get("input_schema").is_some()));
    if !known {
        return ToolOutput::error(format!("unknown tool {name}"));
    }
    let input = if input.is_object() { input.clone() } else { json!({}) };
    let id = format!("mcp-{}", NEXT_ID.fetch_add(1, Ordering::Relaxed));
    let detail = worker::detail_of(name, &input);
    let detail = if detail.is_empty() { "via Claude Code".to_string() } else { format!("{detail} · via Claude Code") };
    let _ = tx.send(Event::ToolStart { id: id.clone(), name: name.into(), detail });
    let out = if tools::is_editor_tool(name) {
        worker::editor_call(name, &input, tx, stop)
    } else {
        let settings = Settings::load(settings_path);
        tools::run_worker_tool(name, &input, &WorkerCtx { settings: &settings, cancel: stop, events: tx })
    };
    let _ = tx.send(Event::ToolDone { id, output: out.clone() });
    out
}

/// 128 bits from std's OS-seeded hasher keys — no extra dependency for a
/// loopback shared secret.
pub fn new_token() -> String {
    use std::hash::{BuildHasher as _, Hasher as _};
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    (0..2u64)
        .map(|i| {
            let mut h = std::collections::hash_map::RandomState::new().build_hasher();
            h.write_u64(i);
            h.write_u128(nanos);
            h.write_u32(std::process::id());
            format!("{:016x}", h.finish())
        })
        .collect()
}

fn write_private(path: &Path, text: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        let mut file = std::fs::OpenOptions::new().write(true).create(true).truncate(true).mode(0o600).open(path)?;
        file.write_all(text.as_bytes())?;
    }
    #[cfg(not(unix))]
    std::fs::write(path, text)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// A call through the bridge reaches the "UI thread" as an EditorCall and
    /// its answer comes back to the MCP side; a wrong token is refused.
    #[test]
    fn editor_call_round_trip() {
        let dir = std::env::temp_dir().join(format!("zp-bridge-{}", std::process::id()));
        let file = dir.join("mcp-bridge.json");
        let bridge = Bridge::start_at(file.clone(), dir.join("assistant.json"), 0, &new_token()).unwrap();
        let found: Discovery = serde_json::from_str(&std::fs::read_to_string(&file).unwrap()).unwrap();
        assert_eq!(found.port, bridge.port);

        let client = std::thread::spawn(move || super::super::mcp::forward_to(&found, "get_editor_state", &json!({})).unwrap());
        let mut answered = false;
        let mut done = false;
        while !done {
            match bridge.events.recv_timeout(Duration::from_secs(10)).expect("bridge stalled") {
                Event::Editor(call) => {
                    assert_eq!(call.name, "get_editor_state");
                    call.reply.send(ToolOutput::text("STATE-OK")).unwrap();
                    answered = true;
                }
                Event::ToolDone { .. } => done = true,
                _ => {}
            }
        }
        assert!(answered);
        let reply = client.join().unwrap();
        assert_eq!(reply["content"][0]["text"], "STATE-OK");
        assert_eq!(reply["is_error"], false);

        let bad = Discovery { port: bridge.port, token: "nope".into(), pid: 0 };
        let reply = super::super::mcp::forward_to(&bad, "get_editor_state", &json!({})).unwrap();
        assert_eq!(reply["is_error"], true);

        drop(bridge);
        assert!(!file.exists(), "discovery file removed on shutdown");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Claude Code over HTTP: handshake, a tool call answered by the "UI
    /// thread", and refusals for a wrong token and a browser origin.
    #[test]
    fn http_mcp_round_trip() {
        let dir = std::env::temp_dir().join(format!("zp-http-{}", std::process::id()));
        let token = new_token();
        let bridge = Bridge::start_at(dir.join("mcp-bridge.json"), dir.join("assistant.json"), 0, &token).unwrap();
        let url = http_url(*bridge.http.as_ref().unwrap());
        let agent: ureq::Agent = ureq::Agent::config_builder().http_status_as_error(false).build().into();
        let post = {
            let (agent, url) = (agent.clone(), url.clone());
            move |auth: &str, origin: Option<&str>, body: Value| {
                let mut req = agent.post(&url).header("Authorization", auth).header("Accept", "application/json, text/event-stream");
                if let Some(origin) = origin {
                    req = req.header("Origin", origin);
                }
                let mut res = req.send_json(body).unwrap();
                (res.status().as_u16(), res.body_mut().read_to_string().unwrap())
            }
        };
        let bearer = format!("Bearer {token}");

        let (code, body) = post(&bearer, None, json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}}));
        assert_eq!(code, 200, "{body}");
        let init: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(init["result"]["protocolVersion"], "2025-06-18");
        assert_eq!(post(&bearer, None, json!({"jsonrpc": "2.0", "method": "notifications/initialized"})).0, 202);

        let (_, body) = post(&bearer, None, json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}));
        let tools: Value = serde_json::from_str(&body).unwrap();
        let names: Vec<&str> = tools["result"]["tools"].as_array().unwrap().iter().filter_map(|t| t["name"].as_str()).collect();
        assert!(names.contains(&"set_brush"));
        assert!(!names.contains(&"open_zestsurface"), "the editor is already running");

        let call = {
            let (post, bearer) = (post.clone(), bearer.clone());
            std::thread::spawn(move || {
                post(&bearer, None, json!({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "get_editor_state", "arguments": {}}}))
            })
        };
        loop {
            match bridge.events.recv_timeout(Duration::from_secs(10)).expect("http call stalled") {
                Event::Editor(call) => call.reply.send(ToolOutput::text("STATE-OK")).unwrap(),
                Event::ToolDone { .. } => break,
                _ => {}
            }
        }
        let (code, body) = call.join().unwrap();
        assert_eq!(code, 200);
        let reply: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(reply["result"]["content"][0]["text"], "STATE-OK");
        assert_eq!(reply["result"]["isError"], false);

        let ping = json!({"jsonrpc": "2.0", "id": 4, "method": "ping"});
        assert_eq!(post("Bearer nope", None, ping.clone()).0, 401);
        assert_eq!(post(&bearer, Some("https://evil.example"), ping.clone()).0, 403);
        assert_eq!(post(&bearer, Some("http://localhost:3000"), ping).0, 200);

        // Stop then start on the same port, as "New token" and port edits do.
        let port = *bridge.http.as_ref().unwrap();
        drop(bridge);
        let again = Bridge::start_at(dir.join("mcp-bridge.json"), dir.join("assistant.json"), port, &new_token()).unwrap();
        assert_eq!(again.http, Ok(port));
        let _ = std::fs::remove_dir_all(dir);
    }
}
