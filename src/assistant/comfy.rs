//! Raster image generation through a local ComfyUI server.
//!
//! Claude writes text, not pixels, so detailed or shaded designs (realistic
//! tattoos, painted decals) come from a diffusion model the user runs
//! themselves. The built-in graph is plain txt2img for SD1.5/SDXL checkpoints;
//! anything fancier (FLUX, LoRAs, ControlNet) goes in
//! `~/.config/zestsurface/comfy_workflow.json` — a workflow exported with "Save (API
//! Format)" whose strings `%prompt%`, `%negative%`, `%seed%`, `%width%` and
//! `%height%` are filled in per request.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result, bail};
use serde_json::{Value, json};

pub struct Job<'a> {
    pub base_url: &'a str,
    pub checkpoint: &'a str,
    pub prompt: &'a str,
    pub negative: &'a str,
    pub seed: u64,
    pub width: u32,
    pub height: u32,
    pub steps: u32,
}

fn agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_connect(Some(Duration::from_secs(5)))
        .timeout_global(Some(Duration::from_secs(60)))
        .build()
        .into()
}

fn get_json(agent: &ureq::Agent, url: &str) -> Result<Value> {
    let text = agent.get(url).call()?.body_mut().with_config().limit(64 << 20).read_to_string()?;
    Ok(serde_json::from_str(&text)?)
}

/// Checkpoints the server has installed.
pub fn checkpoints(base_url: &str) -> Result<Vec<String>> {
    let base = base_url.trim_end_matches('/');
    let info = get_json(&agent(), &format!("{base}/object_info/CheckpointLoaderSimple"))
        .with_context(|| format!("ComfyUI is not answering at {base} — is it running?"))?;
    Ok(info["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
        .as_array()
        .map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect())
        .unwrap_or_default())
}

fn builtin_graph(job: &Job, checkpoint: &str) -> Value {
    json!({
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": job.width, "height": job.height, "batch_size": 1}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": job.prompt, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": job.negative, "clip": ["4", 1]}},
        "3": {"class_type": "KSampler", "inputs": {
            "seed": job.seed, "steps": job.steps, "cfg": 7.0, "sampler_name": "euler_ancestral",
            "scheduler": "normal", "denoise": 1.0,
            "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0]
        }},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "zestsurface", "images": ["8", 0]}}
    })
}

/// Fill a user workflow's placeholders. A string that *is* a placeholder
/// becomes the typed value (seed and sizes must be numbers); one that merely
/// contains `%prompt%`/`%negative%` has it substituted.
fn fill_placeholders(value: &mut Value, job: &Job) {
    match value {
        Value::String(s) => {
            *value = match s.as_str() {
                "%seed%" => json!(job.seed),
                "%width%" => json!(job.width),
                "%height%" => json!(job.height),
                "%steps%" => json!(job.steps),
                _ => Value::String(s.replace("%prompt%", job.prompt).replace("%negative%", job.negative)),
            };
        }
        Value::Array(items) => items.iter_mut().for_each(|v| fill_placeholders(v, job)),
        Value::Object(map) => map.values_mut().for_each(|v| fill_placeholders(v, job)),
        _ => {}
    }
}

/// Queue the job, wait for it, and return the first output image.
pub fn generate(job: &Job, custom_workflow: &Path, cancel: &Arc<AtomicBool>) -> Result<image::RgbaImage> {
    let base = job.base_url.trim_end_matches('/');
    let agent = agent();

    let graph = if let Ok(text) = std::fs::read_to_string(custom_workflow) {
        let mut graph: Value = serde_json::from_str(&text)
            .with_context(|| format!("{} is not valid JSON", custom_workflow.display()))?;
        fill_placeholders(&mut graph, job);
        graph
    } else {
        let checkpoint = if job.checkpoint.is_empty() {
            checkpoints(base)?
                .into_iter()
                .next()
                .context("ComfyUI has no checkpoints installed (models/checkpoints)")?
        } else {
            job.checkpoint.to_string()
        };
        builtin_graph(job, &checkpoint)
    };

    let mut resp = agent
        .post(&format!("{base}/prompt"))
        .send_json(json!({"prompt": graph, "client_id": "zestsurface"}))
        .with_context(|| format!("ComfyUI is not answering at {base} — is it running?"))?;
    let queued: Value = serde_json::from_str(&resp.body_mut().read_to_string()?)?;
    if let Some(err) = queued.get("error") {
        bail!("ComfyUI rejected the workflow: {err}");
    }
    let id = queued["prompt_id"].as_str().context("ComfyUI returned no prompt_id")?.to_string();

    let started = Instant::now();
    let outputs = loop {
        if cancel.load(Ordering::Relaxed) {
            let _ = agent.post(&format!("{base}/interrupt")).send_empty();
            bail!("cancelled");
        }
        if started.elapsed() > Duration::from_secs(600) {
            bail!("ComfyUI took longer than 10 minutes");
        }
        std::thread::sleep(Duration::from_millis(800));
        let history = get_json(&agent, &format!("{base}/history/{id}"))?;
        let entry = &history[&id];
        if entry.is_null() {
            continue;
        }
        if entry["status"]["status_str"].as_str() == Some("error") {
            bail!("ComfyUI failed: {}", entry["status"]["messages"]);
        }
        if let Some(outputs) = entry["outputs"].as_object()
            && !outputs.is_empty()
        {
            break outputs.clone();
        }
    };

    let image = outputs
        .values()
        .find_map(|o| o["images"].as_array().and_then(|a| a.first()).cloned())
        .context("the workflow finished without saving an image")?;
    let bytes = agent
        .get(&format!("{base}/view"))
        .query("filename", image["filename"].as_str().unwrap_or(""))
        .query("subfolder", image["subfolder"].as_str().unwrap_or(""))
        .query("type", image["type"].as_str().unwrap_or("output"))
        .call()?
        .body_mut()
        .with_config()
        .limit(128 << 20)
        .read_to_vec()?;
    Ok(image::load_from_memory(&bytes).context("ComfyUI output is not an image")?.to_rgba8())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholders_take_their_types() {
        let job = Job {
            base_url: "",
            checkpoint: "",
            prompt: "rose tattoo",
            negative: "blurry",
            seed: 42,
            width: 768,
            height: 512,
            steps: 20,
        };
        let mut graph = json!({"1": {"inputs": {"text": "flash, %prompt%", "neg": "%negative%", "seed": "%seed%", "w": "%width%"}}});
        fill_placeholders(&mut graph, &job);
        assert_eq!(graph["1"]["inputs"]["text"], "flash, rose tattoo");
        assert_eq!(graph["1"]["inputs"]["neg"], "blurry");
        assert_eq!(graph["1"]["inputs"]["seed"], 42);
        assert_eq!(graph["1"]["inputs"]["w"], 768);
    }
}
