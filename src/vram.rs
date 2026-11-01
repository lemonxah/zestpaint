//! Where the GPU memory goes, and getting it back.
//!
//! Every image is created through [`image`], which does two things:
//!
//! - **Large images get their own `VkDeviceMemory`.** vulkano's standard
//!   allocator suballocates from 256 MiB blocks and never frees a block until
//!   the allocator itself is dropped, so the driver saw the app's *peak* usage
//!   for the rest of the session: releasing a hidden set's composites or the
//!   textures of a resolution change made room inside ZestSurface but none for
//!   the desktop (screenshot overlays and other apps' windows failed to
//!   allocate). A dedicated allocation is freed the moment its image drops.
//! - **Each image is registered under a [`Kind`]**, so the status bar can say
//!   what the memory is (see [`report`]). Only weak references are kept; dead
//!   entries are pruned when a report is taken.

use std::ffi::c_void;
use std::sync::{Arc, Mutex, Weak};

use vulkano::device::physical::PhysicalDevice;
use vulkano::image::{AllocateImageError, Image, ImageCreateInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryAllocatePreference, MemoryAllocator};
use vulkano::memory::MemoryHeapFlags;
use vulkano::{Validated, Version, VulkanObject};

/// Images at least this big get a dedicated allocation. Below it the block
/// allocator is kept: it is faster, and small images churn little memory.
const DEDICATED_THRESHOLD: u64 = 4 << 20;

/// What an image is for, in the breakdown.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum Kind {
    /// Layer channels and masks, including ones only undo still holds.
    Layer,
    /// Per-set flattened results (and the derived normal map).
    Composite,
    /// Scratch shared between sets: dilation, clip coverage, smudge, G-buffer.
    Scratch,
    /// Imported material maps, brush tips and matcaps.
    Material,
    /// Mesh-map and AO bakes.
    Bake,
    /// Environment HDR and its cubemaps.
    Environment,
    /// Viewport targets: scene image, depth, selection mask, path tracer.
    Viewport,
}

impl Kind {
    pub const ALL: [Kind; 7] =
        [Kind::Layer, Kind::Composite, Kind::Scratch, Kind::Material, Kind::Bake, Kind::Environment, Kind::Viewport];

    pub fn label(self) -> &'static str {
        match self {
            Kind::Layer => "Layer textures",
            Kind::Composite => "Set composites",
            Kind::Scratch => "Scratch",
            Kind::Material => "Imported maps",
            Kind::Bake => "Bakes",
            Kind::Environment => "Environment",
            Kind::Viewport => "Viewport",
        }
    }

    /// Stable key for the editor-state JSON.
    pub fn key(self) -> &'static str {
        match self {
            Kind::Layer => "layer_textures",
            Kind::Composite => "set_composites",
            Kind::Scratch => "scratch",
            Kind::Material => "imported_maps",
            Kind::Bake => "bakes",
            Kind::Environment => "environment",
            Kind::Viewport => "viewport",
        }
    }
}

struct Entry {
    kind: Kind,
    image: Weak<Image>,
    bytes: u64,
}

static REGISTRY: Mutex<Vec<Entry>> = Mutex::new(Vec::new());

/// `Image::new`, with a dedicated allocation for large images and the image
/// registered under `kind`.
pub fn image(
    allocator: Arc<dyn MemoryAllocator>,
    create_info: ImageCreateInfo,
    kind: Kind,
) -> Result<Arc<Image>, Validated<AllocateImageError>> {
    let allocate_preference = if estimated_bytes(&create_info) >= DEDICATED_THRESHOLD {
        MemoryAllocatePreference::AlwaysAllocate
    } else {
        MemoryAllocatePreference::Unknown
    };
    let image = Image::new(allocator, create_info, AllocationCreateInfo { allocate_preference, ..Default::default() })?;
    track(kind, &image);
    Ok(image)
}

/// Register an image made elsewhere.
pub fn track(kind: Kind, image: &Arc<Image>) {
    let bytes = image.memory_requirements().iter().map(|r| r.layout.size()).sum();
    let mut registry = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    registry.push(Entry { kind, image: Arc::downgrade(image), bytes });
}

/// Rough size before allocation: texel blocks × layers, plus a third for a
/// mip chain.
fn estimated_bytes(info: &ImageCreateInfo) -> u64 {
    let [bw, bh, bd] = info.format.block_extent();
    let [w, h, d] = info.extent;
    let blocks = w.div_ceil(bw) as u64 * h.div_ceil(bh) as u64 * d.div_ceil(bd) as u64;
    let base = blocks * info.format.block_size() * info.array_layers as u64;
    if info.mip_levels > 1 { base / 3 * 4 } else { base }
}

/// Live bytes and image count per kind.
#[derive(Clone, Copy, Default, Debug)]
pub struct Usage {
    pub bytes: u64,
    pub count: usize,
}

/// What the tracked images hold right now, per kind, and how much of the
/// `Layer` total is in images for which `in_document` is false (undo history
/// and other copies outside the open document).
pub struct Report {
    pub kinds: Vec<(Kind, Usage)>,
    pub layer_outside_document: Usage,
}

impl Report {
    pub fn total(&self) -> u64 {
        self.kinds.iter().map(|(_, u)| u.bytes).sum()
    }

    pub fn get(&self, kind: Kind) -> Usage {
        self.kinds.iter().find(|(k, _)| *k == kind).map(|(_, u)| *u).unwrap_or_default()
    }
}

pub fn report(in_document: impl Fn(&Arc<Image>) -> bool) -> Report {
    let mut registry = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    registry.retain(|e| e.image.strong_count() > 0);
    let mut kinds: Vec<(Kind, Usage)> = Kind::ALL.iter().map(|k| (*k, Usage::default())).collect();
    let mut outside = Usage::default();
    for entry in registry.iter() {
        let Some(image) = entry.image.upgrade() else { continue };
        if let Some((_, usage)) = kinds.iter_mut().find(|(k, _)| *k == entry.kind) {
            usage.bytes += entry.bytes;
            usage.count += 1;
        }
        if entry.kind == Kind::Layer && !in_document(&image) {
            outside.bytes += entry.bytes;
            outside.count += 1;
        }
    }
    Report { kinds, layer_outside_document: outside }
}

/// The driver's view of device-local memory (`VK_EXT_memory_budget`): what
/// this process holds — tracked images, buffers, allocator slack, the
/// swapchain and the driver's own — and what it may use before the driver
/// starts evicting. `budget` already accounts for other processes' usage.
#[derive(Clone, Copy, Debug)]
pub struct Budget {
    pub process_usage: u64,
    pub budget: u64,
    pub heap_size: u64,
}

pub fn budget(physical: &PhysicalDevice) -> Option<Budget> {
    let instance = physical.instance();
    if instance.api_version() < Version::V1_1 || !physical.supported_extensions().ext_memory_budget {
        return None;
    }
    let mut props_budget = ash::vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
    let mut props = ash::vk::PhysicalDeviceMemoryProperties2::default();
    props.p_next = &mut props_budget as *mut _ as *mut c_void;
    // SAFETY: valid physical device handle from a live instance of at least
    // Vulkan 1.1; the chained struct is supported (checked above) and both
    // outlive the call.
    unsafe { (instance.fns().v1_1.get_physical_device_memory_properties2)(physical.handle(), &mut props) };
    let heaps = &physical.memory_properties().memory_heaps;
    let mut out = Budget { process_usage: 0, budget: 0, heap_size: 0 };
    for (i, heap) in heaps.iter().enumerate() {
        if heap.flags.intersects(MemoryHeapFlags::DEVICE_LOCAL) {
            out.process_usage += props_budget.heap_usage[i];
            out.budget += props_budget.heap_budget[i];
            out.heap_size += heap.size;
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vulkano::format::Format;
    use vulkano::image::{ImageType, ImageUsage};

    fn info(size: u32) -> ImageCreateInfo {
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::R16G16B16A16_SFLOAT,
            extent: [size, size, 1],
            usage: ImageUsage::SAMPLED,
            ..Default::default()
        }
    }

    #[test]
    fn estimates_texel_and_block_formats() {
        assert_eq!(estimated_bytes(&info(1024)), 1024 * 1024 * 8);
        let bc7 = ImageCreateInfo { format: Format::BC7_UNORM_BLOCK, ..info(1024) };
        assert_eq!(estimated_bytes(&bc7), 1024 * 1024);
        let mipped = ImageCreateInfo { mip_levels: 11, ..info(1024) };
        assert_eq!(estimated_bytes(&mipped), 1024 * 1024 * 8 / 3 * 4);
    }

    /// Large images are tracked and their memory goes back to the driver
    /// when they drop: the process's driver-reported usage falls again.
    #[test]
    fn large_images_are_tracked_and_freed() {
        let Some(gpu) = crate::environment::test_gpu::context() else { return };
        let physical = gpu.queue.device().physical_device().clone();
        let image = image(gpu.memory_allocator.clone(), info(8192), Kind::Bake).unwrap();
        let held = report(|_| true).get(Kind::Bake).bytes;
        assert!(held >= 8192 * 8192 * 8, "tracked {held}");
        let Some(before) = budget(&physical) else { return };
        drop(image);
        let after = budget(&physical).unwrap();
        // Other tests share the device, so only require most of it back.
        assert!(
            after.process_usage + 256 * (1 << 20) < before.process_usage,
            "driver still reports {} -> {}",
            before.process_usage,
            after.process_usage
        );
    }
}
