//! Claude Messages API over raw HTTP (there is no official Rust SDK), with
//! server-sent-event streaming.
//!
//! Content blocks are rebuilt from the stream exactly as the API sent them —
//! thinking signatures, server tool results, fallback markers and all — and
//! passed back unchanged on the next request, which is what keeps thinking
//! valid across tool-use turns.

use std::io::{BufRead as _, BufReader};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{Context as _, Result, anyhow, bail};
use serde_json::{Value, json};

const VERSION: &str = "2023-06-01";
/// Server-side refusal fallback, routed by refusal category.
const FALLBACK_BETA: &str = "server-side-fallback-2026-07-01";
const MAX_TOKENS: u32 = 64_000;

pub struct Request<'a> {
    pub url: &'a str,
    pub key: &'a str,
    pub model: &'a str,
    pub effort: &'a str,
    pub system: &'a str,
    pub tools: &'a Value,
    pub messages: &'a [Value],
}

/// Streamed fragments, for live display.
pub enum Delta<'a> {
    Text(&'a str),
    Thinking(&'a str),
    /// A server tool (web search/fetch) call finished streaming its input.
    ServerTool { name: &'a str, input: &'a Value },
}

pub struct Reply {
    /// Every content block, ready to append as the assistant turn.
    pub content: Vec<Value>,
    pub stop_reason: String,
    pub usage: Value,
    /// Tool-use ids whose streamed input was not valid JSON. Their `input`
    /// was replaced with `{}` so the turn can still be sent back.
    pub bad_inputs: Vec<String>,
}

pub fn agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        // Errors are read and reported, not turned into a bare status code.
        .http_status_as_error(false)
        .timeout_connect(Some(Duration::from_secs(30)))
        // Long thinking gaps are covered by the stream's own ping events.
        .timeout_recv_body(Some(Duration::from_secs(600)))
        .build()
        .into()
}

/// Refusal fallbacks exist for the Opus/Fable lines and Sonnet 5.5, not Haiku.
fn supports_fallback(model: &str) -> bool {
    model.starts_with("claude-opus-5") || model.starts_with("claude-fable") || model == "claude-sonnet-5-5"
}

/// The history without what only Gemini understands: its thought
/// summaries and bare parts, and the raw part kept on each block.
fn claude_messages(messages: &[Value]) -> Vec<Value> {
    messages
        .iter()
        .map(|m| {
            let mut m = m.clone();
            if let Some(blocks) = m["content"].as_array_mut() {
                blocks.retain(|b| b.get("gemini_part").is_none() || b["type"] == "text" || b["type"] == "tool_use");
                for b in blocks.iter_mut() {
                    if let Some(o) = b.as_object_mut() {
                        o.remove("gemini_part");
                    }
                }
            }
            m
        })
        .filter(|m| m["content"].as_array().is_none_or(|c| !c.is_empty()))
        .collect()
}

fn body(req: &Request) -> Value {
    let mut body = json!({
        "model": req.model,
        "max_tokens": MAX_TOKENS,
        "stream": true,
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": req.effort},
        "system": [{"type": "text", "text": req.system}],
        "tools": req.tools,
        "messages": claude_messages(req.messages),
        // Auto-place the cache breakpoint on the last cacheable block, so the
        // growing conversation is re-read from cache each turn.
        "cache_control": {"type": "ephemeral"},
    });
    if supports_fallback(req.model) {
        body["fallbacks"] = json!("default");
    }
    body
}

/// Send one request and stream the reply. Retries rate limits, overload and
/// connection failures, but only before anything has streamed.
pub fn send(
    agent: &ureq::Agent,
    req: &Request,
    cancel: &Arc<AtomicBool>,
    mut on_delta: impl FnMut(Delta),
) -> Result<Reply> {
    let body = body(req);
    let mut attempt = 0u32;
    loop {
        attempt += 1;
        let mut call = agent
            .post(req.url)
            .header("x-api-key", req.key)
            .header("anthropic-version", VERSION)
            .header("content-type", "application/json");
        if supports_fallback(req.model) {
            call = call.header("anthropic-beta", FALLBACK_BETA);
        }
        let resp = match call.send_json(&body) {
            Ok(r) => r,
            Err(e) if attempt < 4 => {
                log_retry(&format!("{e}"), attempt);
                continue;
            }
            Err(e) => return Err(anyhow!(e).context(format!("could not reach {}", req.url))),
        };
        let status = resp.status().as_u16();
        if status != 200 {
            let retry_after = resp
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.parse::<u64>().ok());
            let text = resp.into_body().read_to_string().unwrap_or_default();
            let message = serde_json::from_str::<Value>(&text)
                .ok()
                .and_then(|v| v["error"]["message"].as_str().map(str::to_string))
                .unwrap_or(text);
            let retryable = matches!(status, 408 | 409 | 429 | 500..=599);
            if retryable && attempt < 4 {
                let wait = retry_after.unwrap_or(2u64.pow(attempt)).min(30);
                log_retry(&format!("HTTP {status}: {message}"), attempt);
                std::thread::sleep(Duration::from_secs(wait));
                continue;
            }
            bail!(match status {
                401 => format!("API key rejected (HTTP 401): {message}"),
                _ => format!("HTTP {status}: {message}"),
            });
        }
        return read_stream(resp.into_body().into_reader(), cancel, &mut on_delta);
    }
}

fn log_retry(what: &str, attempt: u32) {
    eprintln!("[assistant] {what} — retrying ({attempt}/3)");
}

/// Rebuild the message from its SSE events.
fn read_stream(
    reader: impl std::io::Read,
    cancel: &Arc<AtomicBool>,
    on_delta: &mut impl FnMut(Delta),
) -> Result<Reply> {
    let mut blocks: Vec<Value> = Vec::new();
    // Tool inputs arrive as JSON fragments, joined per block index.
    let mut partial: Vec<String> = Vec::new();
    let mut stop_reason = String::new();
    let mut usage = json!({});
    let mut bad_inputs = Vec::new();

    for line in BufReader::new(reader).lines() {
        if cancel.load(Ordering::Relaxed) {
            bail!("cancelled");
        }
        let line = line.context("stream interrupted")?;
        let Some(data) = line.strip_prefix("data:") else { continue };
        let event: Value = match serde_json::from_str(data.trim()) {
            Ok(v) => v,
            Err(_) => continue,
        };
        match event["type"].as_str().unwrap_or("") {
            "message_start" => {
                usage = event["message"]["usage"].clone();
                if let Some(content) = event["message"]["content"].as_array() {
                    blocks.extend(content.iter().cloned());
                    partial.resize(blocks.len(), String::new());
                }
            }
            "content_block_start" => {
                let index = event["index"].as_u64().unwrap_or(blocks.len() as u64) as usize;
                if blocks.len() <= index {
                    blocks.resize(index + 1, Value::Null);
                    partial.resize(index + 1, String::new());
                }
                blocks[index] = event["content_block"].clone();
            }
            "content_block_delta" => {
                let index = event["index"].as_u64().unwrap_or(0) as usize;
                let Some(block) = blocks.get_mut(index) else { continue };
                let delta = &event["delta"];
                match delta["type"].as_str().unwrap_or("") {
                    "text_delta" => {
                        let text = delta["text"].as_str().unwrap_or("");
                        append(block, "text", text);
                        on_delta(Delta::Text(text));
                    }
                    "thinking_delta" => {
                        let text = delta["thinking"].as_str().unwrap_or("");
                        append(block, "thinking", text);
                        on_delta(Delta::Thinking(text));
                    }
                    "signature_delta" => {
                        append(block, "signature", delta["signature"].as_str().unwrap_or(""));
                    }
                    "input_json_delta" => {
                        partial[index].push_str(delta["partial_json"].as_str().unwrap_or(""));
                    }
                    "citations_delta" => {
                        if !block["citations"].is_array() {
                            block["citations"] = json!([]);
                        }
                        if let Some(list) = block["citations"].as_array_mut() {
                            list.push(delta["citation"].clone());
                        }
                    }
                    _ => {}
                }
            }
            "content_block_stop" => {
                let index = event["index"].as_u64().unwrap_or(0) as usize;
                let Some(block) = blocks.get_mut(index) else { continue };
                let kind = block["type"].as_str().unwrap_or("").to_string();
                if kind == "tool_use" || kind == "server_tool_use" {
                    let raw = std::mem::take(&mut partial[index]);
                    let parsed = if raw.trim().is_empty() {
                        Some(json!({}))
                    } else {
                        serde_json::from_str::<Value>(&raw).ok().filter(Value::is_object)
                    };
                    match parsed {
                        Some(input) => block["input"] = input,
                        None => {
                            block["input"] = json!({});
                            bad_inputs.push(block["id"].as_str().unwrap_or("").to_string());
                        }
                    }
                    if kind == "server_tool_use" {
                        let name = block["name"].as_str().unwrap_or("").to_string();
                        on_delta(Delta::ServerTool { name: &name, input: &block["input"] });
                    }
                }
            }
            "message_delta" => {
                if let Some(reason) = event["delta"]["stop_reason"].as_str() {
                    stop_reason = reason.to_string();
                }
                if let Some(obj) = event["usage"].as_object() {
                    for (k, v) in obj {
                        usage[k] = v.clone();
                    }
                }
            }
            "error" => {
                bail!(
                    "{}: {}",
                    event["error"]["type"].as_str().unwrap_or("error"),
                    event["error"]["message"].as_str().unwrap_or("stream error")
                );
            }
            _ => {}
        }
    }

    if stop_reason.is_empty() {
        bail!("the stream ended before the message finished");
    }
    blocks.retain(|b| !b.is_null());
    Ok(Reply { content: blocks, stop_reason, usage, bad_inputs })
}

fn append(block: &mut Value, field: &str, text: &str) {
    let mut s = block[field].as_str().unwrap_or("").to_string();
    s.push_str(text);
    block[field] = Value::String(s);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rebuilds_text_thinking_and_tool_blocks() {
        let sse = r#"event: message_start
data: {"type":"message_start","message":{"content":[],"usage":{"input_tokens":10}}}

data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Plan"}}
data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig"}}
data: {"type":"content_block_stop","index":0}
data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}
data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Hi "}}
data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"there"}}
data: {"type":"content_block_stop","index":1}
data: {"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"t1","name":"add_layer","input":{}}}
data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"name\": \"Bl"}}
data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"ood\"}"}}
data: {"type":"content_block_stop","index":2}
data: {"type":"content_block_start","index":3,"content_block":{"type":"tool_use","id":"t2","name":"add_layer","input":{}}}
data: {"type":"content_block_delta","index":3,"delta":{"type":"input_json_delta","partial_json":"{\"name\": "}}
data: {"type":"content_block_stop","index":3}
data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":42}}
data: {"type":"message_stop"}
"#;
        let mut text = String::new();
        let cancel = Arc::new(AtomicBool::new(false));
        let reply = read_stream(sse.as_bytes(), &cancel, &mut |d| {
            if let Delta::Text(t) = d {
                text.push_str(t)
            }
        })
        .unwrap();
        assert_eq!(text, "Hi there");
        assert_eq!(reply.stop_reason, "tool_use");
        assert_eq!(reply.content[0]["thinking"], "Plan");
        assert_eq!(reply.content[0]["signature"], "sig");
        assert_eq!(reply.content[2]["input"]["name"], "Blood");
        assert_eq!(reply.content[3]["input"], json!({}));
        assert_eq!(reply.bad_inputs, vec!["t2".to_string()]);
        assert_eq!(reply.usage["input_tokens"], 10);
        assert_eq!(reply.usage["output_tokens"], 42);
    }

    #[test]
    fn truncated_stream_is_an_error() {
        let sse = "data: {\"type\":\"message_start\",\"message\":{\"content\":[],\"usage\":{}}}\n";
        let cancel = Arc::new(AtomicBool::new(false));
        assert!(read_stream(sse.as_bytes(), &cancel, &mut |_| {}).is_err());
    }

    /// A chat that switched from Gemini to Claude sends Claude only what it
    /// understands: Gemini's thoughts and bare parts dropped, raw parts
    /// stripped from text and tool calls.
    #[test]
    fn gemini_turns_are_cleaned_for_claude() {
        let messages = vec![
            json!({"role": "user", "content": [{"type": "text", "text": "hi"}]}),
            json!({"role": "assistant", "content": [
                {"type": "thinking", "thinking": "hm", "gemini_part": {"text": "hm", "thought": true}},
                {"type": "text", "text": "ok", "gemini_part": {"text": "ok"}},
                {"type": "tool_use", "id": "g1", "name": "undo", "input": {}, "gemini_part": {"functionCall": {}}},
                {"type": "gemini", "gemini_part": {"thoughtSignature": "x"}},
            ]}),
        ];
        let clean = claude_messages(&messages);
        let blocks = clean[1]["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0], json!({"type": "text", "text": "ok"}));
        assert_eq!(blocks[1], json!({"type": "tool_use", "id": "g1", "name": "undo", "input": {}}));
    }
}
