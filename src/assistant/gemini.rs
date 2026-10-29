//! Google Gemini (`generateContent`) as a second assistant backend, plus
//! Nano Banana image generation.
//!
//! The conversation is kept in the Claude Messages shape whichever model runs
//! it, so the worker loop and every tool stay the same; this module
//! translates each request into Gemini's `contents` / `functionDeclarations`
//! and the streamed reply back into `text` / `thinking` / `tool_use` blocks.
//! Each block made from a Gemini part keeps that part verbatim under
//! `gemini_part`: Gemini needs its thought signatures echoed back exactly,
//! and the Claude request strips the field (see `api::body`).

use std::io::{BufRead as _, BufReader};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{Context as _, Result, anyhow, bail};
use base64::Engine as _;
use serde_json::{Value, json};

use super::api::{Delta, Reply, Request};

const MAX_OUTPUT_TOKENS: u32 = 65_536;

/// Whether `model` is served by Gemini rather than Claude.
pub fn is_gemini(model: &str) -> bool {
    model.starts_with("gemini")
}

/// Claude-style effort to Gemini's thinking level (Gemini 3 and later take
/// "low" or "high").
fn thinking_level(effort: &str) -> &'static str {
    match effort {
        "low" | "medium" => "low",
        _ => "high",
    }
}

/// Function declarations for the client tools. Claude's server tools (web
/// search, fetch) have no Gemini counterpart that mixes with functions, so
/// they are left out.
fn declarations(tools: &Value) -> Value {
    let decls: Vec<Value> = tools
        .as_array()
        .into_iter()
        .flatten()
        .filter(|t| t.get("type").is_none() && t["name"].is_string())
        .map(|t| {
            json!({
                "name": t["name"],
                "description": t["description"],
                "parametersJsonSchema": t["input_schema"],
            })
        })
        .collect();
    json!([{"functionDeclarations": decls}])
}

/// A Claude content block as Gemini parts (none for blocks Gemini has no use
/// for: Claude's own thinking, server tool traffic).
fn parts_of(block: &Value, names: &std::collections::HashMap<String, (String, Option<Value>)>) -> Vec<Value> {
    if let Some(part) = block.get("gemini_part") {
        // Gemini's own thought summaries are for display; only parts that
        // carry content (and their signatures) go back.
        return if part["thought"] == true { Vec::new() } else { vec![part.clone()] };
    }
    match block["type"].as_str().unwrap_or("") {
        "text" => vec![json!({"text": block["text"]})],
        "image" => vec![json!({"inlineData": {
            "mimeType": block["source"]["media_type"],
            "data": block["source"]["data"],
        }})],
        "tool_use" => vec![json!({"functionCall": {"name": block["name"], "args": block["input"]}})],
        "tool_result" => {
            let id = block["tool_use_id"].as_str().unwrap_or("");
            let (name, call_id) = names.get(id).cloned().unwrap_or_else(|| ("tool".into(), None));
            let blocks: Vec<Value> = match &block["content"] {
                Value::String(s) => vec![json!({"type": "text", "text": s})],
                Value::Array(list) => list.clone(),
                _ => Vec::new(),
            };
            let text: Vec<&str> = blocks.iter().filter_map(|b| b["text"].as_str()).collect();
            let key = if block["is_error"] == true { "error" } else { "result" };
            let mut response = json!({"name": name, "response": {key: text.join("\n")}});
            if let Some(call_id) = call_id {
                response["id"] = call_id;
            }
            // Images a tool returned (screenshots, stamp previews) follow the
            // response as ordinary parts of the same turn.
            let mut parts = vec![json!({"functionResponse": response})];
            parts.extend(blocks.iter().filter(|b| b["type"] == "image").flat_map(|b| parts_of(b, names)));
            parts
        }
        _ => Vec::new(),
    }
}

/// The request body: Claude-shaped history in, Gemini `contents` out.
fn body(req: &Request) -> Value {
    // tool_use id → (function name, Gemini's own call id), for the responses.
    let mut names = std::collections::HashMap::new();
    for message in req.messages {
        for block in message["content"].as_array().into_iter().flatten().filter(|b| b["type"] == "tool_use") {
            let call_id = block["gemini_part"]["functionCall"].get("id").cloned();
            names.insert(block["id"].as_str().unwrap_or("").to_string(), (block["name"].as_str().unwrap_or("").to_string(), call_id));
        }
    }
    let mut contents: Vec<Value> = Vec::new();
    for message in req.messages {
        let role = if message["role"] == "assistant" { "model" } else { "user" };
        let parts: Vec<Value> = match &message["content"] {
            Value::String(s) => vec![json!({"text": s})],
            Value::Array(blocks) => blocks.iter().flat_map(|b| parts_of(b, &names)).collect(),
            _ => Vec::new(),
        };
        if parts.is_empty() {
            continue;
        }
        // Gemini wants turns to alternate; merge any that don't.
        match contents.last_mut() {
            Some(last) if last["role"] == role => last["parts"].as_array_mut().unwrap().extend(parts),
            _ => contents.push(json!({"role": role, "parts": parts})),
        }
    }
    json!({
        "systemInstruction": {"parts": [{"text": req.system}]},
        "contents": contents,
        "tools": declarations(req.tools),
        "generationConfig": {
            "maxOutputTokens": MAX_OUTPUT_TOKENS,
            "thinkingConfig": {"includeThoughts": true, "thinkingLevel": thinking_level(req.effort)},
        },
    })
}

/// `{base}/v1beta/models/{model}:{method}`; base defaults to Google's API.
pub fn url(base: &str, model: &str, method: &str) -> String {
    let base = Some(base.trim()).filter(|b| !b.is_empty()).unwrap_or("https://generativelanguage.googleapis.com");
    format!("{}/v1beta/models/{model}:{method}", base.trim_end_matches('/'))
}

/// POST `body` with retries on rate limits, overload and connection errors.
fn post(agent: &ureq::Agent, url: &str, key: &str, body: &Value) -> Result<ureq::http::Response<ureq::Body>> {
    let mut attempt = 0u32;
    loop {
        attempt += 1;
        let resp = match agent.post(url).header("x-goog-api-key", key).header("content-type", "application/json").send_json(body) {
            Ok(r) => r,
            Err(e) if attempt < 4 => {
                eprintln!("[assistant] {e} — retrying ({attempt}/3)");
                continue;
            }
            Err(e) => return Err(anyhow!(e).context(format!("could not reach {url}"))),
        };
        let status = resp.status().as_u16();
        if status == 200 {
            return Ok(resp);
        }
        let text = resp.into_body().read_to_string().unwrap_or_default();
        let message = serde_json::from_str::<Value>(&text)
            .ok()
            .and_then(|v| v["error"]["message"].as_str().map(str::to_string))
            .unwrap_or(text);
        if matches!(status, 429 | 500..=599) && attempt < 4 {
            eprintln!("[assistant] HTTP {status}: {message} — retrying ({attempt}/3)");
            std::thread::sleep(Duration::from_secs(2u64.pow(attempt)));
            continue;
        }
        bail!(match status {
            400 if message.contains("API key") => format!("Gemini API key rejected: {message}"),
            401 | 403 => format!("Gemini API key rejected (HTTP {status}): {message}"),
            _ => format!("Gemini HTTP {status}: {message}"),
        });
    }
}

/// Send one turn to Gemini and stream the reply, as Claude-shaped blocks.
pub fn send(agent: &ureq::Agent, req: &Request, cancel: &Arc<AtomicBool>, mut on_delta: impl FnMut(Delta)) -> Result<Reply> {
    let resp = post(agent, &format!("{}?alt=sse", req.url), req.key, &body(req))?;
    read_stream(resp.into_body().into_reader(), cancel, &mut on_delta)
}

/// Rebuild the reply from its SSE chunks. Text arrives in pieces, which are
/// joined into one part per run (thought or not); a thought signature can
/// come on any piece, so it is kept on the joined part.
fn read_stream(reader: impl std::io::Read, cancel: &Arc<AtomicBool>, on_delta: &mut impl FnMut(Delta)) -> Result<Reply> {
    let mut parts: Vec<Value> = Vec::new();
    let mut finish = String::new();
    let mut usage = json!({});
    for line in BufReader::new(reader).lines() {
        if cancel.load(Ordering::Relaxed) {
            bail!("cancelled");
        }
        let line = line.context("stream interrupted")?;
        let Some(data) = line.strip_prefix("data:") else { continue };
        let Ok(chunk) = serde_json::from_str::<Value>(data.trim()) else { continue };
        if let Some(reason) = chunk["promptFeedback"]["blockReason"].as_str() {
            finish = format!("BLOCKED:{reason}");
        }
        if let Some(meta) = chunk.get("usageMetadata") {
            usage = meta.clone();
        }
        let candidate = &chunk["candidates"][0];
        if let Some(reason) = candidate["finishReason"].as_str() {
            finish = reason.to_string();
        }
        for part in candidate["content"]["parts"].as_array().into_iter().flatten() {
            if let Some(text) = part["text"].as_str() {
                let thought = part["thought"] == true;
                if thought {
                    on_delta(Delta::Thinking(text));
                } else {
                    on_delta(Delta::Text(text));
                }
                let joins = parts.last().is_some_and(|last| last["text"].is_string() && (last["thought"] == true) == thought);
                if joins {
                    let last = parts.last_mut().unwrap();
                    last["text"] = Value::String(format!("{}{text}", last["text"].as_str().unwrap_or("")));
                    if let Some(sig) = part.get("thoughtSignature") {
                        last["thoughtSignature"] = sig.clone();
                    }
                    continue;
                }
            }
            parts.push(part.clone());
        }
    }
    if finish.is_empty() {
        bail!("the Gemini stream ended before the reply finished");
    }

    let mut content = Vec::new();
    let mut calls = 0;
    for part in parts {
        if let Some(call) = part.get("functionCall") {
            calls += 1;
            // Unique across the conversation: Claude may read it later.
            let id = call["id"].as_str().map(str::to_string).unwrap_or_else(|| {
                let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
                format!("gemini_{t:x}_{calls}")
            });
            let args = Some(call["args"].clone()).filter(Value::is_object).unwrap_or_else(|| json!({}));
            content.push(json!({"type": "tool_use", "id": id, "name": call["name"], "input": args, "gemini_part": part}));
        } else if let Some(text) = part["text"].as_str() {
            let kind = if part["thought"] == true { "thinking" } else { "text" };
            content.push(json!({"type": kind, kind: text, "gemini_part": part}));
        } else if !part.as_object().is_none_or(|o| o.is_empty()) {
            // Anything else (a signature on its own, code parts): keep it so
            // it goes back with the turn.
            content.push(json!({"type": "gemini", "gemini_part": part}));
        }
    }
    let stop_reason = match finish.as_str() {
        _ if calls > 0 => "tool_use",
        "MAX_TOKENS" => "max_tokens",
        "SAFETY" | "PROHIBITED_CONTENT" | "BLOCKLIST" | "SPII" | "RECITATION" | "IMAGE_SAFETY" => "refusal",
        f if f.starts_with("BLOCKED:") => "refusal",
        _ => "end_turn",
    };
    let thoughts = usage["thoughtsTokenCount"].as_u64().unwrap_or(0);
    let usage = json!({
        "input_tokens": usage["promptTokenCount"].as_u64().unwrap_or(0).saturating_sub(usage["cachedContentTokenCount"].as_u64().unwrap_or(0)),
        "cache_read_input_tokens": usage["cachedContentTokenCount"].as_u64().unwrap_or(0),
        "output_tokens": usage["candidatesTokenCount"].as_u64().unwrap_or(0) + thoughts,
    });
    Ok(Reply { content, stop_reason: stop_reason.into(), usage, bad_inputs: Vec::new() })
}

/// Generate one image with a Nano Banana model from a text prompt.
pub fn generate_image(base: &str, key: &str, model: &str, prompt: &str, aspect_ratio: &str, cancel: &Arc<AtomicBool>) -> Result<image::RgbaImage> {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .http_status_as_error(false)
        .timeout_global(Some(Duration::from_secs(180)))
        .build()
        .into();
    let body = json!({
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"responseModalities": ["IMAGE"], "imageConfig": {"aspectRatio": aspect_ratio}},
    });
    let resp = post(&agent, &url(base, model, "generateContent"), key, &body)?;
    if cancel.load(Ordering::Relaxed) {
        bail!("cancelled");
    }
    let reply: Value = serde_json::from_reader(resp.into_body().into_reader()).context("reading the Gemini reply")?;
    let parts = reply["candidates"][0]["content"]["parts"].as_array().cloned().unwrap_or_default();
    let Some(data) = parts.iter().find_map(|p| p["inlineData"]["data"].as_str()) else {
        let said: Vec<&str> = parts.iter().filter_map(|p| p["text"].as_str()).collect();
        let reason = reply["candidates"][0]["finishReason"].as_str().or(reply["promptFeedback"]["blockReason"].as_str()).unwrap_or("no image");
        bail!("{model} returned no image ({reason}){}", if said.is_empty() { String::new() } else { format!(": {}", said.join(" ")) });
    };
    let bytes = base64::engine::general_purpose::STANDARD.decode(data).context("decoding the image")?;
    Ok(image::load_from_memory(&bytes).context("reading the generated image")?.to_rgba8())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request<'a>(messages: &'a [Value], tools: &'a Value) -> Request<'a> {
        Request { url: "", key: "k", model: "gemini-4-argon", effort: "medium", system: "SYS", tools, messages }
    }

    /// Claude-shaped history becomes alternating Gemini turns: tools without
    /// the server ones, images as inline data, tool results as function
    /// responses named after their calls, and Gemini's own parts verbatim.
    #[test]
    fn history_translates_to_gemini_contents() {
        let tools = json!([
            {"name": "add_layer", "description": "Add", "input_schema": {"type": "object", "properties": {}}, "eager_input_streaming": true},
            {"type": "web_search_20260209", "name": "web_search"},
        ]);
        let call = json!({"functionCall": {"name": "add_layer", "args": {"name": "Ink"}, "id": "g1"}, "thoughtSignature": "SIG"});
        let messages = vec![
            json!({"role": "user", "content": [
                {"type": "text", "text": "hi"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "AAA"}},
            ]}),
            json!({"role": "assistant", "content": [
                {"type": "thinking", "thinking": "plan", "gemini_part": {"text": "plan", "thought": true}},
                {"type": "tool_use", "id": "g1", "name": "add_layer", "input": {"name": "Ink"}, "gemini_part": call},
            ]}),
            json!({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "g1", "content": [
                    {"type": "text", "text": "Added."},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "BBB"}},
                ], "is_error": false},
                {"type": "text", "text": "thanks"},
            ]}),
        ];
        let tools_ref = tools.clone();
        let b = body(&request(&messages, &tools_ref));
        assert_eq!(b["systemInstruction"]["parts"][0]["text"], "SYS");
        let decls = b["tools"][0]["functionDeclarations"].as_array().unwrap();
        assert_eq!(decls.len(), 1, "server tools are left out");
        assert_eq!(decls[0]["parametersJsonSchema"]["type"], "object");
        let c = &b["contents"];
        assert_eq!(c[0]["role"], "user");
        assert_eq!(c[0]["parts"][1]["inlineData"]["mimeType"], "image/jpeg");
        assert_eq!(c[1]["role"], "model");
        assert_eq!(c[1]["parts"].as_array().unwrap().len(), 1, "thought summaries are not sent back");
        assert_eq!(c[1]["parts"][0], call, "the call goes back verbatim, signature and all");
        assert_eq!(c[2]["parts"][0]["functionResponse"]["name"], "add_layer");
        assert_eq!(c[2]["parts"][0]["functionResponse"]["id"], "g1");
        assert_eq!(c[2]["parts"][0]["functionResponse"]["response"]["result"], "Added.");
        assert_eq!(c[2]["parts"][1]["inlineData"]["data"], "BBB");
        assert_eq!(c[2]["parts"][2]["text"], "thanks");
        assert_eq!(b["generationConfig"]["thinkingConfig"]["thinkingLevel"], "low");
    }

    #[test]
    fn stream_rebuilds_text_thoughts_and_calls() {
        let sse = [
            json!({"candidates": [{"content": {"role": "model", "parts": [{"text": "Thinking ", "thought": true}]}}]}),
            json!({"candidates": [{"content": {"role": "model", "parts": [{"text": "hard", "thought": true}]}}]}),
            json!({"candidates": [{"content": {"role": "model", "parts": [{"text": "On "}]}}]}),
            json!({"candidates": [{"content": {"role": "model", "parts": [{"text": "it.", "thoughtSignature": "S1"}]}}]}),
            json!({"candidates": [{"content": {"role": "model", "parts": [{"functionCall": {"name": "get_editor_state", "args": {}}, "thoughtSignature": "S2"}]}, "finishReason": "STOP"}],
                   "usageMetadata": {"promptTokenCount": 100, "cachedContentTokenCount": 40, "candidatesTokenCount": 7, "thoughtsTokenCount": 5}}),
        ]
        .iter()
        .map(|c| format!("data: {c}\r\n\r\n"))
        .collect::<String>();
        let (mut text, mut thinking) = (String::new(), String::new());
        let cancel = Arc::new(AtomicBool::new(false));
        let reply = read_stream(sse.as_bytes(), &cancel, &mut |d| match d {
            Delta::Text(t) => text.push_str(t),
            Delta::Thinking(t) => thinking.push_str(t),
            _ => {}
        })
        .unwrap();
        assert_eq!((text.as_str(), thinking.as_str()), ("On it.", "Thinking hard"));
        assert_eq!(reply.stop_reason, "tool_use");
        assert_eq!(reply.content[0]["type"], "thinking");
        assert_eq!(reply.content[1]["text"], "On it.");
        assert_eq!(reply.content[1]["gemini_part"]["thoughtSignature"], "S1");
        assert_eq!(reply.content[2]["type"], "tool_use");
        assert_eq!(reply.content[2]["name"], "get_editor_state");
        assert_eq!(reply.content[2]["gemini_part"]["thoughtSignature"], "S2");
        assert_eq!(reply.usage["input_tokens"], 60);
        assert_eq!(reply.usage["cache_read_input_tokens"], 40);
        assert_eq!(reply.usage["output_tokens"], 12);
    }

    #[test]
    fn blocked_prompts_are_refusals_and_cut_streams_errors() {
        let cancel = Arc::new(AtomicBool::new(false));
        let blocked = format!("data: {}\n\n", json!({"promptFeedback": {"blockReason": "SAFETY"}}));
        assert_eq!(read_stream(blocked.as_bytes(), &cancel, &mut |_| {}).unwrap().stop_reason, "refusal");
        let cut = format!("data: {}\n\n", json!({"candidates": [{"content": {"parts": [{"text": "Hel"}]}}]}));
        assert!(read_stream(cut.as_bytes(), &cancel, &mut |_| {}).is_err());
    }
}
