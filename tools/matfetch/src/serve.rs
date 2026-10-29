use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use serde::Deserialize;
use serde_json::json;
use tiny_http::{Header, Response, Server};

use matfetch::catalog::Index;
use matfetch::download::{self, Pool};

fn json_header() -> Header {
    Header::from_bytes(&b"Content-Type"[..], &b"application/json; charset=utf-8"[..]).unwrap()
}

fn html_header() -> Header {
    Header::from_bytes(&b"Content-Type"[..], &b"text/html; charset=utf-8"[..]).unwrap()
}

#[derive(Deserialize)]
struct DownloadReq {
    ids: Vec<String>,
    variant: String,
}

/// Which assets are already on disk, and at what variant.
fn installed_map(root: &PathBuf, index: &Index) -> HashMap<String, String> {
    index
        .assets
        .iter()
        .filter_map(|a| download::installed(root, a).map(|v| (a.id.clone(), v)))
        .collect()
}

pub fn serve(root: PathBuf, index: Arc<Index>, port: u16, jobs: usize) -> Result<()> {
    let pool = Pool::new(root.clone(), Arc::clone(&index), jobs);
    let server = Arc::new(
        Server::http(("127.0.0.1", port))
            .map_err(|e| anyhow::anyhow!("bind 127.0.0.1:{port}: {e}"))?,
    );

    // The index is large enough that reserialising it per request is wasteful.
    let index_json = Arc::new(serde_json::to_string(&*index)?);

    println!("\n  matfetch gallery  ->  http://127.0.0.1:{port}\n");
    println!("  library root: {}", root.display());
    println!("  {} assets indexed. Ctrl-C to stop.\n", index.assets.len());

    let mut handles = Vec::new();
    for _ in 0..4 {
        let server = Arc::clone(&server);
        let pool = Arc::clone(&pool);
        let index = Arc::clone(&index);
        let index_json = Arc::clone(&index_json);
        let root = root.clone();
        handles.push(std::thread::spawn(move || {
            loop {
                let Ok(mut request) = server.recv() else { break };
                let url = request.url().to_string();
                let path = url.split('?').next().unwrap_or("/").to_string();

                let response = match path.as_str() {
                    "/" | "/index.html" => {
                        let _ = request.respond(
                            Response::from_string(include_str!("ui.html")).with_header(html_header()),
                        );
                        continue;
                    }
                    "/api/index" => index_json.as_str().to_string(),
                    "/api/status" => {
                        let progress = pool.shared.lock().unwrap().clone();
                        json!({
                            "installed": installed_map(&root, &index),
                            "progress": progress,
                            "pending": pool.pending(),
                        })
                        .to_string()
                    }
                    "/api/download" => {
                        let mut body = String::new();
                        let _ = request.as_reader().read_to_string(&mut body);
                        match serde_json::from_str::<DownloadReq>(&body) {
                            Ok(req) => {
                                for id in &req.ids {
                                    pool.enqueue(id, &req.variant);
                                }
                                json!({ "queued": req.ids.len() }).to_string()
                            }
                            Err(e) => json!({ "error": e.to_string() }).to_string(),
                        }
                    }
                    _ => {
                        let _ = request.respond(Response::from_string("not found").with_status_code(404));
                        continue;
                    }
                };
                let _ = request.respond(Response::from_string(response).with_header(json_header()));
            }
        }));
    }
    for h in handles {
        let _ = h.join();
    }
    Ok(())
}
