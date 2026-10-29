//! The status bar's VRAM readout and the same numbers for the assistant.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use egui::RichText;
use serde_json::{Value, json};
use vulkano::image::Image;

use super::AppState;
use crate::theme;
use crate::vram::{self, Kind};

/// How often the numbers are retaken: the walk is cheap, but the readout
/// does not need to flicker at frame rate.
const REFRESH: Duration = Duration::from_secs(1);

pub(super) struct VramSnapshot {
    taken: Instant,
    report: vram::Report,
    budget: Option<vram::Budget>,
}

const MIB: f64 = (1u64 << 20) as f64;
const GIB: f64 = (1u64 << 30) as f64;

impl AppState {
    /// Images the open document points at: every set's layer channels,
    /// masks and bakes. Layer textures outside this set are only alive
    /// because undo history (or a pending job) still holds them.
    fn document_images(&self) -> HashSet<*const Image> {
        let mut out = HashSet::new();
        for set in &self.texture_sets {
            for layer in &set.layers.layers {
                for view in layer.channels().into_iter().chain(layer.mask.as_ref()) {
                    out.insert(Arc::as_ptr(view.image()));
                }
            }
            let bakes = [&set.baked_ao, &set.baked_position, &set.baked_normal, &set.baked_curvature];
            for view in bakes.into_iter().flatten() {
                out.insert(Arc::as_ptr(view.image()));
            }
        }
        out
    }

    fn take_vram_snapshot(&self) -> VramSnapshot {
        let document = self.document_images();
        let report = vram::report(|image| document.contains(&Arc::as_ptr(image)));
        let budget = vram::budget(self.gpu.queue.device().physical_device());
        VramSnapshot { taken: Instant::now(), report, budget }
    }

    fn vram_snapshot(&mut self) -> &VramSnapshot {
        if self.vram.as_ref().is_none_or(|s| s.taken.elapsed() >= REFRESH) {
            self.vram = Some(self.take_vram_snapshot());
        }
        self.vram.as_ref().unwrap()
    }

    /// Status bar: what the driver says this process holds, with the
    /// breakdown on hover.
    pub(super) fn vram_label(&mut self, ui: &mut egui::Ui) {
        let snap = self.vram_snapshot();
        let tracked = snap.report.total();
        let (text, warn) = match snap.budget {
            Some(b) => {
                // Past ~90% of what the driver will give us, other apps (and
                // the desktop's own screenshot overlay) start failing.
                let warn = b.process_usage as f64 > b.budget as f64 * 0.9;
                (format!("VRAM {:.1} / {:.0} GB", b.process_usage as f64 / GIB, b.heap_size as f64 / GIB), warn)
            }
            None => (format!("VRAM {:.1} GB", tracked as f64 / GIB), false),
        };
        let colour = if warn { theme::WARN_FG } else { theme::TEXT_MUTED };
        let tooltip = vram_breakdown(snap);
        ui.label(RichText::new(text).monospace().color(colour)).on_hover_ui(|ui| {
            ui.label(RichText::new(tooltip).monospace());
        });
    }

    /// The editor-state `vram` block, in MiB.
    pub(super) fn vram_json(&self) -> Value {
        let snap = &self.take_vram_snapshot();
        let mib = |b: u64| (b as f64 / MIB).round() as u64;
        let mut kinds = serde_json::Map::new();
        for (kind, usage) in &snap.report.kinds {
            kinds.insert(kind.key().into(), json!({"mib": mib(usage.bytes), "images": usage.count}));
        }
        let outside = snap.report.layer_outside_document;
        json!({
            "process_mib": snap.budget.map(|b| mib(b.process_usage)),
            "budget_mib": snap.budget.map(|b| mib(b.budget)),
            "gpu_mib": snap.budget.map(|b| mib(b.heap_size)),
            "tracked_images_mib": mib(snap.report.total()),
            "by_kind": kinds,
            "layer_textures_held_only_by_undo_mib": mib(outside.bytes),
        })
    }
}

fn vram_breakdown(snap: &VramSnapshot) -> String {
    let gib = |b: u64| b as f64 / GIB;
    let mut lines = Vec::new();
    for kind in Kind::ALL {
        let usage = snap.report.get(kind);
        lines.push(format!("{:<16} {:>6.2} GB  ({} images)", kind.label(), gib(usage.bytes), usage.count));
        if kind == Kind::Layer && snap.report.layer_outside_document.bytes > 0 {
            lines.push(format!("  held by undo   {:>6.2} GB", gib(snap.report.layer_outside_document.bytes)));
        }
    }
    let tracked = snap.report.total();
    match snap.budget {
        Some(b) => {
            let other = b.process_usage.saturating_sub(tracked);
            lines.push(format!("{:<16} {:>6.2} GB", "Buffers & other", gib(other)));
            lines.push(String::new());
            lines.push(format!("This process     {:>6.2} GB", gib(b.process_usage)));
            lines.push(format!("Driver budget    {:>6.2} GB", gib(b.budget)));
            lines.push("(budget = what is left for us after other apps)".into());
        }
        None => lines.push(format!("{:<16} {:>6.2} GB", "Total images", gib(tracked))),
    }
    lines.join("\n")
}
