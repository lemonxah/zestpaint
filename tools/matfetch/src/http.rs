use std::io::Read;
use std::path::Path;
use std::time::Duration;

use anyhow::{Context as _, Result, bail};
use serde::de::DeserializeOwned;

/// Poly Haven's CDN 403s requests without a User-Agent, and it is simply polite
/// to identify a bulk mirror to both hosts.
pub const UA: &str = "zestsurface-matfetch/0.1 (CC0 asset mirror; +https://github.com/zestsurface)";

const RETRIES: u32 = 4;

pub fn agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(900)))
        .build()
        .into()
}

fn backoff(attempt: u32) -> Duration {
    Duration::from_millis(500 * 2u64.pow(attempt))
}

/// GET a JSON document, retrying transient failures.
pub fn get_json<T: DeserializeOwned>(agent: &ureq::Agent, url: &str) -> Result<T> {
    let mut last: Option<anyhow::Error> = None;
    for attempt in 0..RETRIES {
        if attempt > 0 {
            std::thread::sleep(backoff(attempt - 1));
        }
        match agent.get(url).header("User-Agent", UA).call() {
            Ok(mut resp) => {
                // The catalog endpoints return multi-megabyte documents, well past
                // ureq's default read_to_string cap.
                let body = resp
                    .body_mut()
                    .with_config()
                    .limit(512 * 1024 * 1024)
                    .read_to_string();
                match body {
                    Ok(s) => {
                        return serde_json::from_str(&s)
                            .with_context(|| format!("malformed JSON from {url}"));
                    }
                    Err(e) => last = Some(anyhow::Error::new(e)),
                }
            }
            Err(e) => last = Some(anyhow::Error::new(e)),
        }
    }
    Err(last.unwrap_or_else(|| anyhow::anyhow!("no attempts"))).with_context(|| format!("GET {url}"))
}

/// Stream a URL to `dest`, retrying transient failures. Writes to a sibling
/// `.part` file and renames on success so an interrupted run never leaves a
/// truncated file that looks complete.
pub fn download_to(agent: &ureq::Agent, url: &str, dest: &Path) -> Result<u64> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let part = dest.with_extension(format!(
        "{}.part",
        dest.extension().and_then(|e| e.to_str()).unwrap_or("tmp")
    ));
    let mut last: Option<anyhow::Error> = None;
    for attempt in 0..RETRIES {
        if attempt > 0 {
            std::thread::sleep(backoff(attempt - 1));
        }
        match try_download(agent, url, &part) {
            Ok(n) => {
                std::fs::rename(&part, dest)
                    .with_context(|| format!("rename into {}", dest.display()))?;
                return Ok(n);
            }
            Err(e) => {
                let _ = std::fs::remove_file(&part);
                last = Some(e);
            }
        }
    }
    Err(last.unwrap_or_else(|| anyhow::anyhow!("no attempts")))
        .with_context(|| format!("download {url}"))
}

fn try_download(agent: &ureq::Agent, url: &str, part: &Path) -> Result<u64> {
    let mut resp = agent.get(url).header("User-Agent", UA).call()?;
    let status = resp.status();
    if !status.is_success() {
        bail!("HTTP {status}");
    }
    let expected: Option<u64> = resp
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse().ok());

    let mut file = std::fs::File::create(part)?;
    let mut reader = resp.body_mut().as_reader();
    let mut buf = vec![0u8; 128 * 1024];
    let mut total = 0u64;
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        std::io::Write::write_all(&mut file, &buf[..n])?;
        total += n as u64;
    }
    drop(reader);
    std::io::Write::flush(&mut file)?;
    drop(file);

    if let Some(expected) = expected
        && total != expected
    {
        bail!("short read: got {total} bytes, expected {expected}");
    }
    Ok(total)
}
