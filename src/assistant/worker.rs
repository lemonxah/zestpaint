//! One conversation turn on a worker thread: request → stream → run tools →
//! repeat until the model (Claude or Gemini) ends its turn.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::thread::JoinHandle;
use std::time::Duration;

use serde_json::{Value, json};

use super::api::{self, Delta};
use super::tools::{self, WorkerCtx};
use super::{EditorCall, Event, Settings, ToolOutput};

/// Tool rounds per user message before the worker stops and asks.
const MAX_ROUNDS: usize = 40;

const INTRO: &str = "You are the assistant built into ZestSurface, a Substance-Painter-style 3D texture painting app (layers, masks, PBR channels, brushes, stamps). The user is usually painting textures for clothing and avatars (often VR avatars). You operate the editor through tools; the user does the actual brush work in the viewport.";

const ATTACHMENTS: &str = "Images you receive: the user's messages may carry a viewport screenshot (what they see right now) and the active texture set's maps — the material's own base colour texture and the composite of the painted layers (UV layout space; black where nothing is painted). Use them to understand what \"here\", \"this\" or \"fix that\" means, and capture_viewport to check your own changes.";

const STYLE: &str = "Style: be brief and practical. Do the steps rather than describing them, then say in a sentence or two what you did and exactly what the user should do next (which tool, where to click). Ask only when a choice really matters (which texture set when it is ambiguous, a design detail they care about).";

/// How the editor works and the recipes — shared by the in-app assistant and
/// the Claude Code MCP server (as its `instructions`).
pub const EDITOR_GUIDE: &str = r#"How the editor works:
- A model is split into texture sets (one per object + material). Painting, layers and masks belong to the ACTIVE set; select the right one first (select_texture_set).
- Layers composite bottom to top. A fill layer from apply_library_material holds a whole material and starts with a BLACK mask: it shows nothing until the mask is painted white (set_brush target=mask, mask_value=1), generated (smart_mask), or filled (layer_action fill_mask_white).
- The brush paints the enabled channels (color, roughness, metallic, height, emissive, opacity) into the active layer, or into its mask. tool=stamp places the whole tip image once per click — that is how decals, tattoos and splats are placed. The user clicks on the model to place them; you cannot click for them.
- Pieces: view_pieces selects, isolates (Blender local view, the user's / key), hides or frames texture sets. Hidden pieces don't occlude painting or baking, so isolate the piece being worked on when clothing or other parts are in the way, and leave_isolation when done.
- Every edit you make is undoable with Ctrl+Z, and you have undo/redo tools.

Recipes:
- Blood / liquid splatter: create_material (fresh blood ≈ #4a0000–#6a0a0a, roughness 0.1–0.2; dried ≈ #2e0805, roughness 0.55) → apply_library_material (new fill layer, black mask, brush set to mask) → generate_splatter_brush a few seeds/styles → use_library_brush → set_brush tool=stamp, target=mask, mask_value=1 → tell the user to click where the splats go (Shift+scroll rotates, [ ] or Ctrl+scroll resizes). For blood collected in creases, smart_mask cavity_dirt on that layer instead. Drips run down the stamp's bottom edge, so orient them with tip rotation.
- Fine detail: a set paints at 2048 px by default; set_texture_resolution 4096 (or 8192) on the set before detailed work such as tattoos on a small UV area like the back, and make stamps big enough for that size.
- Tattoo: design it with create_stamp_from_svg (clean bold linework, transparent background, black ink unless colour was asked for) or, for shaded/realistic art, generate_image_stamp (Nano Banana or the user's ComfyUI). Look at the preview you get back and iterate if it is off. Then add_layer named after the design → use_library_brush → set_brush tool=stamp, target=material, channels=["color"] (plus roughness ≈0.45 if they want the ink slightly glossier), use_tip_color=true for colour designs, or a near-black ink colour like #151515 for monochrome. Then tell the user to click on the skin where it should go. Suggest layer opacity 0.85–0.95 and a Multiply blend for a healed-ink look.
- Finding textures: search_online_assets first (free CC0 catalogue) → download_asset → apply_library_material / use_library_brush. Only when that has nothing, use web_search to find images, and import_image_from_url with a DIRECT image URL; say where it came from and its licence, and prefer CC0 / public-domain sources."#;

fn system_prompt() -> String {
    format!("{INTRO}\n\n{EDITOR_GUIDE}\n\n{ATTACHMENTS}\n\n{STYLE}")
}

pub struct Turn {
    pub events: mpsc::Receiver<Event>,
    pub cancel: Arc<AtomicBool>,
    /// The updated conversation, returned when the turn ends.
    pub handle: JoinHandle<Vec<Value>>,
}

/// Start a turn: append the user's content to `messages` and run until
/// Claude ends its turn, fails, or is cancelled.
pub fn start_turn(settings: Settings, mut messages: Vec<Value>, user_content: Vec<Value>) -> Turn {
    let (tx, rx) = mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let flag = cancel.clone();
    let handle = std::thread::Builder::new()
        .name("assistant".into())
        .spawn(move || {
            let before = messages.clone();
            // A previous turn that failed after its tool results leaves a user
            // message last; extend it rather than sending two user turns.
            match messages.last_mut() {
                Some(last) if last["role"] == "user" => {
                    if let Some(content) = last["content"].as_array_mut() {
                        content.extend(user_content);
                    }
                }
                _ => messages.push(json!({"role": "user", "content": user_content})),
            }
            let start_len = messages.len();
            if let Err(e) = run(&settings, &mut messages, &tx, &flag) {
                // Nothing came back at all: forget the user message so it can
                // be edited and resent instead of piling up.
                if messages.len() == start_len {
                    messages = before;
                }
                if flag.load(Ordering::Relaxed) {
                    let _ = tx.send(Event::Notice("Stopped.".into()));
                } else {
                    let _ = tx.send(Event::Error(format!("{e:#}")));
                }
            }
            messages
        })
        .expect("spawn assistant worker");
    Turn { events: rx, cancel, handle }
}

fn run(settings: &Settings, messages: &mut Vec<Value>, tx: &mpsc::Sender<Event>, cancel: &Arc<AtomicBool>) -> anyhow::Result<()> {
    let gemini = super::gemini::is_gemini(&settings.model);
    let key = settings.resolved_key().ok_or_else(|| {
        anyhow::anyhow!(if gemini {
            "No Gemini API key: set GEMINI_API_KEY or enter a key in the assistant settings (gear icon)."
        } else {
            "No API key: set ANTHROPIC_API_KEY or enter a key in the assistant settings (gear icon)."
        })
    })?;
    let tools_def = tools::definitions();
    let agent = api::agent();
    let ctx = WorkerCtx { settings, cancel, events: tx };
    let url = settings.request_url();
    let system = system_prompt();

    for _ in 0..MAX_ROUNDS {
        let request = api::Request {
            url: &url,
            key: &key,
            model: &settings.model,
            effort: &settings.effort,
            system: &system,
            tools: &tools_def,
            messages,
        };
        let on_delta = |delta: Delta| match delta {
            Delta::Text(t) => {
                let _ = tx.send(Event::Text(t.to_string()));
            }
            Delta::Thinking(t) => {
                let _ = tx.send(Event::Thinking(t.to_string()));
            }
            Delta::ServerTool { name, input } => {
                let detail = input["query"].as_str().or_else(|| input["url"].as_str()).unwrap_or("").to_string();
                let _ = tx.send(Event::ServerTool { name: name.to_string(), detail });
            }
        };
        let reply = if gemini {
            super::gemini::send(&agent, &request, cancel, on_delta)?
        } else {
            api::send(&agent, &request, cancel, on_delta)?
        };
        let _ = tx.send(Event::Usage {
            input: reply.usage["input_tokens"].as_u64().unwrap_or(0)
                + reply.usage["cache_creation_input_tokens"].as_u64().unwrap_or(0),
            output: reply.usage["output_tokens"].as_u64().unwrap_or(0),
            cache_read: reply.usage["cache_read_input_tokens"].as_u64().unwrap_or(0),
        });

        // The assistant turn goes back exactly as received.
        messages.push(json!({"role": "assistant", "content": reply.content.clone()}));

        match reply.stop_reason.as_str() {
            "tool_use" => {}
            // The server paused a long server-tool loop: resend as-is and it
            // resumes where it left off.
            "pause_turn" => continue,
            "refusal" => {
                let _ = tx.send(Event::Notice("The model declined this request.".into()));
                return Ok(());
            }
            "max_tokens" => {
                let _ = tx.send(Event::Notice("The reply hit the output limit and was cut short.".into()));
                close_open_tool_uses(messages, &reply.content, "The reply was cut off before this tool ran.");
                return Ok(());
            }
            _ => return Ok(()),
        }

        let mut results = Vec::new();
        for block in reply.content.iter().filter(|b| b["type"] == "tool_use") {
            let id = block["id"].as_str().unwrap_or("").to_string();
            let name = block["name"].as_str().unwrap_or("").to_string();
            let output = if cancel.load(Ordering::Relaxed) {
                ToolOutput::error("Cancelled by the user.")
            } else if reply.bad_inputs.contains(&id) {
                ToolOutput::error("INVALID_JSON: the tool input was not valid JSON. Send the call again with a complete, valid input object.")
            } else {
                let _ = tx.send(Event::ToolStart { id: id.clone(), name: name.clone(), detail: detail_of(&name, &block["input"]) });
                let out = if tools::is_editor_tool(&name) {
                    editor_call(&name, &block["input"], tx, cancel)
                } else {
                    tools::run_worker_tool(&name, &block["input"], &ctx)
                };
                let _ = tx.send(Event::ToolDone { id: id.clone(), output: out.clone() });
                out
            };
            results.push(json!({
                "type": "tool_result",
                "tool_use_id": id,
                "content": output.content,
                "is_error": output.is_error,
            }));
        }
        // All results in one user message keeps parallel tool use working.
        messages.push(json!({"role": "user", "content": results}));
        if cancel.load(Ordering::Relaxed) {
            anyhow::bail!("cancelled");
        }
    }
    let _ = tx.send(Event::Notice(format!("Stopped after {MAX_ROUNDS} tool rounds — say 'continue' to keep going.")));
    Ok(())
}

/// Answer every tool_use in a truncated reply so the history stays valid.
fn close_open_tool_uses(messages: &mut Vec<Value>, content: &[Value], why: &str) {
    let results: Vec<Value> = content
        .iter()
        .filter(|b| b["type"] == "tool_use")
        .map(|b| json!({"type": "tool_result", "tool_use_id": b["id"], "content": why, "is_error": true}))
        .collect();
    if !results.is_empty() {
        messages.push(json!({"role": "user", "content": results}));
    }
}

/// Hand the call to the UI thread and wait for its answer.
pub(super) fn editor_call(name: &str, input: &Value, tx: &mpsc::Sender<Event>, cancel: &Arc<AtomicBool>) -> ToolOutput {
    let (reply_tx, reply_rx) = mpsc::channel();
    if tx.send(Event::Editor(EditorCall { name: name.into(), input: input.clone(), reply: reply_tx })).is_err() {
        return ToolOutput::error("The editor is not listening.");
    }
    loop {
        match reply_rx.recv_timeout(Duration::from_millis(200)) {
            Ok(out) => return out,
            Err(mpsc::RecvTimeoutError::Timeout) if cancel.load(Ordering::Relaxed) => {
                return ToolOutput::error("Cancelled by the user.");
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => return ToolOutput::error("The editor dropped the call."),
        }
    }
}

/// A short description of a call for the chat's tool row.
pub(super) fn detail_of(name: &str, input: &Value) -> String {
    let pick = |keys: &[&str]| keys.iter().find_map(|k| input[*k].as_str().map(str::to_string));
    match name {
        "create_stamp_from_svg" | "generate_image_stamp" | "generate_splatter_brush" | "create_material" => {
            let mut s = pick(&["name"]).unwrap_or_default();
            if let Some(p) = pick(&["prompt", "style", "color"]) {
                s = format!("{s} — {p}");
            }
            s
        }
        _ => pick(&["name", "query", "id", "action", "kind", "what", "tool", "url"]).unwrap_or_default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sse(events: &[Value]) -> String {
        events.iter().map(|e| format!("event: {}\ndata: {e}\n\n", e["type"].as_str().unwrap())).collect()
    }

    /// A full turn against a mock Messages API: Claude calls an editor tool,
    /// the "UI" answers it, and the second request must carry the assistant
    /// turn back unchanged (thinking signature included) plus the result.
    #[test]
    fn tool_round_trip_through_the_editor() {
        let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
        let port = server.server_addr().to_ip().unwrap().port();
        let first = sse(&[
            json!({"type": "message_start", "message": {"content": [], "usage": {"input_tokens": 5}}}),
            json!({"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "SIG"}}),
            json!({"type": "content_block_stop", "index": 0}),
            json!({"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "get_editor_state", "input": {}}}),
            json!({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{}"}}),
            json!({"type": "content_block_stop", "index": 1}),
            json!({"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 7}}),
            json!({"type": "message_stop"}),
        ]);
        let second = sse(&[
            json!({"type": "message_start", "message": {"content": [], "usage": {"input_tokens": 9}}}),
            json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Done."}}),
            json!({"type": "content_block_stop", "index": 0}),
            json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}}),
            json!({"type": "message_stop"}),
        ]);
        let bodies = std::thread::spawn(move || {
            let mut bodies = Vec::new();
            for reply in [first, second] {
                let mut req = server.recv().unwrap();
                let mut body = String::new();
                req.as_reader().read_to_string(&mut body).unwrap();
                assert_eq!(req.headers().iter().find(|h| h.field.equiv("x-api-key")).unwrap().value.as_str(), "test-key");
                bodies.push(serde_json::from_str::<Value>(&body).unwrap());
                req.respond(tiny_http::Response::from_string(reply)).unwrap();
            }
            bodies
        });

        let settings = Settings { api_key: "test-key".into(), api_base: format!("http://127.0.0.1:{port}"), ..Settings::default() };
        let turn = start_turn(settings, Vec::new(), vec![json!({"type": "text", "text": "hi"})]);
        let mut text = String::new();
        let mut answered = false;
        loop {
            match turn.events.recv_timeout(Duration::from_secs(10)) {
                Ok(Event::Editor(call)) => {
                    assert_eq!(call.name, "get_editor_state");
                    call.reply.send(ToolOutput::text("STATE-OK")).unwrap();
                    answered = true;
                }
                Ok(Event::Text(t)) => text.push_str(&t),
                Ok(Event::Error(e)) => panic!("{e}"),
                Ok(_) => {}
                // The worker hung up: the turn is over.
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
                Err(mpsc::RecvTimeoutError::Timeout) => panic!("worker stalled"),
            }
        }
        assert!(answered);
        let messages = turn.handle.join().unwrap();
        assert_eq!(text, "Done.");
        assert_eq!(messages.len(), 4, "user, assistant(tool_use), user(result), assistant(text)");

        let bodies = bodies.join().unwrap();
        let first_req = &bodies[0];
        assert_eq!(first_req["model"], "claude-opus-5-5");
        assert_eq!(first_req["stream"], true);
        assert_eq!(first_req["thinking"]["type"], "adaptive");
        assert_eq!(first_req["fallbacks"], "default");
        assert!(first_req["tools"].as_array().unwrap().iter().any(|t| t["type"] == "web_search_20260209"));
        let second_req = &bodies[1]["messages"];
        assert_eq!(second_req[1]["content"][0]["signature"], "SIG");
        assert_eq!(second_req[2]["content"][0]["tool_use_id"], "toolu_1");
        assert_eq!(second_req[2]["content"][0]["content"][0]["text"], "STATE-OK");
    }

    /// The same round trip against a mock Gemini API: the request goes to
    /// streamGenerateContent with the Gemini key, and the second request
    /// carries the function call back with its thought signature, then the
    /// editor's answer as a function response.
    #[test]
    fn gemini_tool_round_trip() {
        let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
        let port = server.server_addr().to_ip().unwrap().port();
        let chunk = |v: Value| format!("data: {v}\r\n\r\n");
        let first = chunk(json!({"candidates": [{"content": {"role": "model", "parts": [
            {"functionCall": {"name": "get_editor_state", "args": {}}, "thoughtSignature": "GSIG"}
        ]}, "finishReason": "STOP"}], "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3}}));
        let second = chunk(json!({"candidates": [{"content": {"role": "model", "parts": [{"text": "Done."}]}, "finishReason": "STOP"}]}));
        let requests = std::thread::spawn(move || {
            let mut seen = Vec::new();
            for reply in [first, second] {
                let mut req = server.recv().unwrap();
                let mut body = String::new();
                req.as_reader().read_to_string(&mut body).unwrap();
                let key = req.headers().iter().find(|h| h.field.equiv("x-goog-api-key")).unwrap().value.to_string();
                seen.push((req.url().to_string(), key, serde_json::from_str::<Value>(&body).unwrap()));
                req.respond(tiny_http::Response::from_string(reply)).unwrap();
            }
            seen
        });

        let settings = Settings {
            model: "gemini-4-argon".into(),
            gemini_api_key: "g-key".into(),
            gemini_api_base: format!("http://127.0.0.1:{port}"),
            ..Settings::default()
        };
        let turn = start_turn(settings, Vec::new(), vec![json!({"type": "text", "text": "hi"})]);
        let mut text = String::new();
        loop {
            match turn.events.recv_timeout(Duration::from_secs(10)) {
                Ok(Event::Editor(call)) => call.reply.send(ToolOutput::text("STATE-OK")).unwrap(),
                Ok(Event::Text(t)) => text.push_str(&t),
                Ok(Event::Error(e)) => panic!("{e}"),
                Ok(_) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
                Err(mpsc::RecvTimeoutError::Timeout) => panic!("worker stalled"),
            }
        }
        let messages = turn.handle.join().unwrap();
        assert_eq!(text, "Done.");
        assert_eq!(messages.len(), 4);

        let seen = requests.join().unwrap();
        assert_eq!(seen[0].0, "/v1beta/models/gemini-4-argon:streamGenerateContent?alt=sse");
        assert_eq!(seen[0].1, "g-key");
        let contents = &seen[1].2["contents"];
        assert_eq!(contents[1]["parts"][0]["thoughtSignature"], "GSIG");
        assert_eq!(contents[2]["parts"][0]["functionResponse"]["name"], "get_editor_state");
        assert_eq!(contents[2]["parts"][0]["functionResponse"]["response"]["result"], "STATE-OK");
    }
}
