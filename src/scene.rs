//! Scene data shared by the loaders and the app.
use glam::Mat4;

use crate::mesh::MeshData;

/// Reported by the model loaders so a long import can show progress instead of
/// freezing the window. `total == 0` means "no count yet, just a stage name".
pub type ProgressFn<'a> = &'a (dyn Fn(&str, usize, usize) + Send + Sync);

/// Latest state of a running import.
#[derive(Clone, Debug, Default)]
pub struct ImportProgress {
    pub label: String,
    pub done: usize,
    pub total: usize,
}

impl ImportProgress {
    /// 0..1 for the bar, or None while the stage has no count.
    pub fn fraction(&self) -> Option<f32> {
        (self.total > 0).then(|| (self.done as f32 / self.total as f32).clamp(0.0, 1.0))
    }
}

/// CPU-side RGBA8 image, sRGB-encoded (as glTF base color textures are).
#[derive(Clone)]
pub struct ImageData {
    pub width: u32,
    pub height: u32,
    /// Shared: one texture used by many parts (an FBX atlas) is decoded and
    /// held once, and the renderer uploads it once.
    pub rgba: std::sync::Arc<Vec<u8>>,
}

#[derive(Clone)]
pub struct MaterialData {
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    /// sRGB-encoded color texture.
    pub base_color_image: Option<ImageData>,
    /// Linear grayscale maps (value read from the red channel).
    pub roughness_image: Option<ImageData>,
    pub metallic_image: Option<ImageData>,
    /// Tangent-space normal map (linear, xyz in [0,1]).
    pub normal_image: Option<ImageData>,
    /// Height/displacement map; used to derive a normal map when no explicit
    /// normal map is present.
    pub height_image: Option<ImageData>,
    /// Ambient occlusion map (multiplies environment lighting).
    pub ao_image: Option<ImageData>,
    /// sRGB emission colour map, multiplied by `emission`.
    pub emission_image: Option<ImageData>,
    /// Emission strength mask (linear, red channel): where it glows.
    pub emission_mask_image: Option<ImageData>,
    /// OpenPBR-style layered-surface parameters (subset of the 1.1 slot set).
    pub emission: [f32; 3],
    pub specular_weight: f32,
    pub coat_weight: f32,
    pub coat_roughness: f32,
    pub sheen_weight: f32,
    pub sheen_roughness: f32,
    /// Uniform surface opacity (multiplied with texture alpha and paint).
    pub opacity: f32,
    /// Alpha-blend this material (sorted pass) instead of cutout clipping.
    pub transparent: bool,
}

impl Default for MaterialData {
    fn default() -> Self {
        Self {
            base_color: [0.8, 0.8, 0.8, 1.0],
            metallic: 0.0,
            roughness: 0.5,
            base_color_image: None,
            roughness_image: None,
            metallic_image: None,
            normal_image: None,
            height_image: None,
            ao_image: None,
            emission_image: None,
            emission_mask_image: None,
            emission: [0.0; 3],
            specular_weight: 1.0,
            coat_weight: 0.0,
            coat_roughness: 0.1,
            sheen_weight: 0.0,
            sheen_roughness: 0.3,
            opacity: 1.0,
            transparent: false,
        }
    }
}

/// Just the scalar surface parameters, for pushing UI edits to the GPU.
#[derive(Clone, Copy)]
pub struct SurfaceParams {
    pub emission: [f32; 3],
    pub specular_weight: f32,
    pub coat_weight: f32,
    pub coat_roughness: f32,
    pub sheen_weight: f32,
    pub sheen_roughness: f32,
    pub opacity: f32,
    pub transparent: bool,
}

impl MaterialData {
    /// What the GPU multiplies `emission` by: the colour map times the
    /// strength mask, sRGB-encoded. None (a white texture) when there's
    /// neither, so a plain emission colour glows everywhere.
    pub fn emission_texture(&self) -> Option<ImageData> {
        let (map, mask) = (self.emission_image.as_ref(), self.emission_mask_image.as_ref());
        let Some(mask) = mask else { return map.cloned() };
        let (width, height) = map.map_or((mask.width, mask.height), |m| (m.width, m.height));
        let decode: Vec<f32> = (0..256).map(|v| srgb_to_linear(v as f32 / 255.0)).collect();
        let mut rgba = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            for x in 0..width {
                // Nearest texel of the mask at the map's size.
                let mx = (x as u64 * mask.width as u64 / width as u64) as usize;
                let my = (y as u64 * mask.height as u64 / height as u64) as usize;
                let strength = mask.rgba[(my * mask.width as usize + mx) * 4] as f32 / 255.0;
                let colour = map.map_or([255; 3], |m| {
                    let i = (y as usize * width as usize + x as usize) * 4;
                    [m.rgba[i], m.rgba[i + 1], m.rgba[i + 2]]
                });
                for c in colour {
                    let linear = decode[c as usize] * strength;
                    rgba.push((linear_to_srgb(linear) * 255.0).round() as u8);
                }
                rgba.push(255);
            }
        }
        Some(ImageData { width, height, rgba: rgba.into() })
    }

    pub fn surface_params(&self) -> SurfaceParams {
        SurfaceParams {
            emission: self.emission,
            specular_weight: self.specular_weight,
            coat_weight: self.coat_weight,
            coat_roughness: self.coat_roughness,
            sheen_weight: self.sheen_weight,
            sheen_roughness: self.sheen_roughness,
            opacity: self.opacity,
            transparent: self.transparent,
        }
    }

    /// The normal map the renderer should use: the explicit one, or one
    /// derived from the height map.
    pub fn effective_normal_image(&self) -> Option<ImageData> {
        self.normal_image
            .clone()
            .or_else(|| self.height_image.as_ref().map(|h| derive_normal_from_height(h, 4.0)))
    }
}

/// Texture slots a material exposes for file assignment.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MapSlot {
    BaseColor,
    Roughness,
    Metallic,
    Normal,
    Height,
    Ao,
    /// Emission colour (sRGB), times the material's emission colour.
    Emission,
    /// Emission strength (red channel): 0 doesn't glow, 1 glows fully.
    EmissionMask,
    /// Cutout coverage: 1 opaque, 0 clipped away. Library and fill-layer only —
    /// it has no per-material GPU binding, so it is absent from `ALL`.
    Opacity,
}

impl MapSlot {
    /// Slots assignable per material in the UI. `Opacity` is excluded: it is
    /// consumed by fill layers, not by the material texture bindings.
    pub const ALL: [MapSlot; 8] = [
        MapSlot::BaseColor,
        MapSlot::Roughness,
        MapSlot::Metallic,
        MapSlot::Normal,
        MapSlot::Height,
        MapSlot::Ao,
        MapSlot::Emission,
        MapSlot::EmissionMask,
    ];

    pub fn label(self) -> &'static str {
        match self {
            MapSlot::BaseColor => "Base color",
            MapSlot::Roughness => "Roughness",
            MapSlot::Metallic => "Metallic",
            MapSlot::Normal => "Normal",
            MapSlot::Height => "Height",
            MapSlot::Ao => "AO",
            MapSlot::Emission => "Emission",
            MapSlot::EmissionMask => "Emission mask",
            MapSlot::Opacity => "Opacity",
        }
    }
}

/// Split models whose UVs span multiple UDIM tiles (1001, 1002, …) into one
/// model per tile with UVs remapped to 0..1. Each tile then becomes its own
/// texture set ("Material [1002]"), painting at full resolution.
pub fn split_udim_tiles(models: Vec<ModelData>) -> Vec<ModelData> {
    use std::collections::BTreeMap;

    let mut out = Vec::new();
    for model in models {
        // Bucket triangles by the tile of their UV centroid.
        let mut tiles: BTreeMap<(i32, i32), Vec<u32>> = BTreeMap::new();
        for tri in model.mesh.indices.chunks_exact(3) {
            let mut cu = 0.0;
            let mut cv = 0.0;
            for &i in tri {
                let uv = model.mesh.vertices[i as usize].uv;
                cu += uv[0];
                cv += uv[1];
            }
            // Note: our V axis is flipped vs UDIM convention at import time,
            // so the tile row comes from the original V = 1 - v.
            let tu = (cu / 3.0).floor().clamp(-1.0, 9.0) as i32;
            let tv = (1.0 - cv / 3.0 - 1e-4).floor().clamp(-1.0, 99.0) as i32;
            tiles.entry((tu, tv)).or_default().extend_from_slice(tri);
        }

        if tiles.len() <= 1 {
            out.push(model);
            continue;
        }

        for ((tu, tv), tri_indices) in tiles {
            let udim = 1001 + tu.max(0) + 10 * tv.max(0);
            let source_indices = tri_indices.clone();
            // Rebuild as a flat triangle list with tile-local UVs.
            let vertices: Vec<crate::mesh::Vertex3D> = tri_indices
                .iter()
                .map(|&i| {
                    let mut v = model.mesh.vertices[i as usize];
                    v.uv = [v.uv[0] - tu as f32, v.uv[1] + tv as f32];
                    v
                })
                .collect();
            let indices = (0..vertices.len() as u32).collect();
            let morphs = model.morphs.remapped(&source_indices, &vertices);
            let skin = model.skin.as_ref().map(|s| s.remapped(&source_indices, &vertices));
            out.push(ModelData {
                mesh: crate::mesh::MeshData { vertices, indices },
                material: clone_material(&model.material),
                transform: model.transform,
                name: model.name.clone(),
                material_name: format!("{} [{udim}]", model.material_name),
                morphs,
                skin,
            });
        }
    }
    out
}

fn clone_material(m: &MaterialData) -> MaterialData {
    MaterialData {
        base_color: m.base_color,
        metallic: m.metallic,
        roughness: m.roughness,
        base_color_image: m.base_color_image.clone(),
        roughness_image: m.roughness_image.clone(),
        metallic_image: m.metallic_image.clone(),
        normal_image: m.normal_image.clone(),
        height_image: m.height_image.clone(),
        ao_image: m.ao_image.clone(),
        emission_image: m.emission_image.clone(),
        emission_mask_image: m.emission_mask_image.clone(),
        emission: m.emission,
        specular_weight: m.specular_weight,
        coat_weight: m.coat_weight,
        coat_roughness: m.coat_roughness,
        sheen_weight: m.sheen_weight,
        sheen_roughness: m.sheen_roughness,
        opacity: m.opacity,
        transparent: m.transparent,
    }
}

/// Sobel a grayscale height map into a tangent-space normal map.
pub fn derive_normal_from_height(height: &ImageData, strength: f32) -> ImageData {
    let (w, h) = (height.width as i32, height.height as i32);
    let sample = |x: i32, y: i32| -> f32 {
        let x = x.clamp(0, w - 1);
        let y = y.clamp(0, h - 1);
        height.rgba[((y * w + x) * 4) as usize] as f32 / 255.0
    };
    let mut rgba = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let dx = (sample(x + 1, y - 1) + 2.0 * sample(x + 1, y) + sample(x + 1, y + 1))
                - (sample(x - 1, y - 1) + 2.0 * sample(x - 1, y) + sample(x - 1, y + 1));
            let dy = (sample(x - 1, y + 1) + 2.0 * sample(x, y + 1) + sample(x + 1, y + 1))
                - (sample(x - 1, y - 1) + 2.0 * sample(x, y - 1) + sample(x + 1, y - 1));
            let n = glam::Vec3::new(-dx * strength, -dy * strength, 1.0).normalize();
            let o = ((y * w + x) * 4) as usize;
            rgba[o] = ((n.x * 0.5 + 0.5) * 255.0) as u8;
            rgba[o + 1] = ((n.y * 0.5 + 0.5) * 255.0) as u8;
            rgba[o + 2] = ((n.z * 0.5 + 0.5) * 255.0) as u8;
            rgba[o + 3] = 255;
        }
    }
    ImageData { width: height.width, height: height.height, rgba: rgba.into() }
}

/// One drawable unit: the faces of one material of one object.
#[derive(Clone)]
pub struct ModelData {
    pub mesh: MeshData,
    pub material: MaterialData,
    pub transform: Mat4,
    /// Object (node) name from the source file.
    pub name: String,
    /// Material name within that object. Texture sets are keyed per
    /// (object, material) — Substance-style — so a jacket with two materials
    /// paints as two independent sets.
    pub material_name: String,
    /// Blendshapes; when there are any, `mesh.vertices` is the deformed
    /// result of their weights over `morphs.base`.
    pub morphs: Morphs,
    /// Bones and weights, when the part is rigged; in a pose, `mesh.vertices`
    /// are also bent to it (see [`ModelData::deform`]).
    pub skin: Option<crate::skeleton::Skin>,
}

/// One blendshape (morph target) of a model part.
#[derive(Clone)]
pub struct Morph {
    pub name: String,
    /// 0 = off, 1 = the full shape (values outside 0..1 extrapolate).
    pub weight: f32,
    /// (vertex, position offset, normal offset) for each vertex it moves.
    pub offsets: std::sync::Arc<Vec<(u32, [f32; 3], [f32; 3])>>,
}

/// A model part's blendshapes and the undeformed vertices they act on.
#[derive(Clone, Default)]
pub struct Morphs {
    /// Empty when the part has no blendshapes.
    pub base: std::sync::Arc<Vec<crate::mesh::Vertex3D>>,
    pub targets: Vec<Morph>,
}

impl Morphs {
    /// Blendshapes for a part whose vertices were rebuilt: new vertex `j` is
    /// old vertex `sources[j]`, with the attributes in `vertices` (new UVs).
    pub fn remapped(&self, sources: &[u32], vertices: &[crate::mesh::Vertex3D]) -> Morphs {
        if self.targets.is_empty() {
            return Morphs::default();
        }
        let mut copies: Vec<Vec<u32>> = vec![Vec::new(); self.base.len()];
        for (new, &old) in sources.iter().enumerate() {
            if let Some(list) = copies.get_mut(old as usize) {
                list.push(new as u32);
            }
        }
        let base = sources
            .iter()
            .zip(vertices)
            .map(|(&old, now)| crate::mesh::Vertex3D { uv: now.uv, ..self.base[old as usize] })
            .collect();
        let targets = self
            .targets
            .iter()
            .map(|t| Morph {
                offsets: std::sync::Arc::new(
                    t.offsets
                        .iter()
                        .flat_map(|&(old, dp, dn)| copies[old as usize].iter().map(move |&new| (new, dp, dn)))
                        .collect(),
                ),
                ..t.clone()
            })
            .collect();
        Morphs { base: std::sync::Arc::new(base), targets }
    }
}

/// Give parts without blendshapes the shapes of the same part (object and
/// material) in `source` — the model file a project was made from. Vertices
/// are matched by position and normal, so a part whose UVs were redone
/// still matches; a target vertex with no match just doesn't move. Weights
/// come from the source. Returns the indices of the parts that got shapes.
pub fn attach_morphs(target: &mut [ModelData], source: &[ModelData]) -> Vec<usize> {
    let key = |v: &crate::mesh::Vertex3D| {
        let p = v.position.map(f32::to_bits);
        let n = v.normal.map(f32::to_bits);
        [p[0], p[1], p[2], n[0], n[1], n[2]]
    };
    let mut attached = Vec::new();
    for (i, part) in target.iter_mut().enumerate() {
        if !part.morphs.targets.is_empty() {
            continue;
        }
        let Some(src) = source
            .iter()
            .find(|s| s.name == part.name && s.material_name == part.material_name && !s.morphs.targets.is_empty())
        else {
            continue;
        };
        let by_key: std::collections::HashMap<[u32; 6], u32> =
            src.morphs.base.iter().enumerate().rev().map(|(j, v)| (key(v), j as u32)).collect();
        // Each source vertex's copies in the target.
        let mut copies: Vec<Vec<u32>> = vec![Vec::new(); src.morphs.base.len()];
        for (t, v) in part.rest_vertices().iter().enumerate() {
            if let Some(&j) = by_key.get(&key(v)) {
                copies[j as usize].push(t as u32);
            }
        }
        let targets: Vec<Morph> = src
            .morphs
            .targets
            .iter()
            .map(|m| Morph {
                offsets: std::sync::Arc::new(
                    m.offsets.iter().flat_map(|&(j, dp, dn)| copies[j as usize].iter().map(move |&t| (t, dp, dn))).collect(),
                ),
                ..m.clone()
            })
            .filter(|m| !m.offsets.is_empty())
            .collect();
        if targets.is_empty() {
            continue;
        }
        part.morphs = Morphs { base: std::sync::Arc::new(part.rest_vertices().to_vec()), targets };
        part.deform();
        attached.push(i);
    }
    attached
}

impl ModelData {
    /// The vertices with no blendshapes and no pose: what a project stores.
    pub fn rest_vertices(&self) -> &[crate::mesh::Vertex3D] {
        if !self.morphs.targets.is_empty() {
            &self.morphs.base
        } else if let Some(skin) = &self.skin {
            &skin.rest
        } else {
            &self.mesh.vertices
        }
    }

    /// Rebuild `mesh.vertices` from the rest vertices, the blendshape
    /// weights and the pose. False (and nothing done) for a part with
    /// neither blendshapes nor bones.
    pub fn deform(&mut self) -> bool {
        if self.morphs.targets.is_empty() && self.skin.is_none() {
            return false;
        }
        let mut vertices = self.rest_vertices().to_vec();
        let mut bent_normals = false;
        for target in self.morphs.targets.iter().filter(|t| t.weight != 0.0) {
            let w = target.weight;
            for &(i, dp, dn) in target.offsets.iter() {
                let Some(v) = vertices.get_mut(i as usize) else { continue };
                for k in 0..3 {
                    v.position[k] += dp[k] * w;
                    v.normal[k] += dn[k] * w;
                }
                bent_normals |= dn != [0.0; 3];
            }
        }
        if bent_normals {
            for v in &mut vertices {
                v.normal = glam::Vec3::from(v.normal).normalize_or(glam::Vec3::Y).to_array();
            }
        }
        if let Some(skin) = &self.skin {
            skin.deform(&mut vertices, &self.transform);
        }
        self.mesh.vertices = vertices;
        true
    }
}

/// Give parts without bones the skin of the same part (object and
/// material) in `source`, the model file a project was made from. Vertices
/// are matched by rest position and normal, then by nearest position, so a
/// part whose UVs were redone still follows its bones. Returns the indices
/// of the parts that got a skin.
pub fn attach_skins(target: &mut [ModelData], source: &[ModelData]) -> Vec<usize> {
    let key = |v: &crate::mesh::Vertex3D| {
        let p = v.position.map(f32::to_bits);
        let n = v.normal.map(f32::to_bits);
        [p[0], p[1], p[2], n[0], n[1], n[2]]
    };
    let mut attached = Vec::new();
    for (i, part) in target.iter_mut().enumerate() {
        if part.skin.is_some() {
            continue;
        }
        // Files can hold several meshes of one name and material: take the
        // one whose vertices this part's are.
        let rest = part.rest_vertices().to_vec();
        let keys: std::collections::HashSet<[u32; 6]> = rest.iter().map(key).collect();
        let Some(skin) = source
            .iter()
            .filter(|s| s.name == part.name && s.material_name == part.material_name)
            .filter_map(|s| s.skin.as_ref())
            .max_by_key(|skin| skin.rest.iter().filter(|v| keys.contains(&key(v))).count())
        else {
            continue;
        };
        let by_key: std::collections::HashMap<[u32; 6], u32> =
            skin.rest.iter().enumerate().rev().map(|(j, v)| (key(v), j as u32)).collect();
        // Nearest-vertex fallback on a grid sized to the part.
        let (lo, hi) = skin.rest.iter().fold((glam::Vec3::splat(f32::MAX), glam::Vec3::splat(f32::MIN)), |(lo, hi), v| {
            let p = glam::Vec3::from(v.position);
            (lo.min(p), hi.max(p))
        });
        let cell = ((hi - lo).max_element() / 64.0).max(1e-6);
        let cell_of = |p: glam::Vec3| ((p - lo) / cell).floor().as_ivec3();
        let mut grid: std::collections::HashMap<glam::IVec3, Vec<u32>> = std::collections::HashMap::new();
        for (j, v) in skin.rest.iter().enumerate() {
            grid.entry(cell_of(glam::Vec3::from(v.position))).or_default().push(j as u32);
        }
        let nearest = |p: glam::Vec3| -> Option<u32> {
            let c = cell_of(p);
            for r in 1..6 {
                let mut best: Option<(f32, u32)> = None;
                for dx in -r..=r {
                    for dy in -r..=r {
                        for dz in -r..=r {
                            for &j in grid.get(&(c + glam::IVec3::new(dx, dy, dz))).into_iter().flatten() {
                                let d = (glam::Vec3::from(skin.rest[j as usize].position) - p).length_squared();
                                if best.is_none_or(|b| d < b.0) {
                                    best = Some((d, j));
                                }
                            }
                        }
                    }
                }
                if let Some(b) = best {
                    return Some(b.1);
                }
            }
            // Far from everything: the closest vertex of all.
            (0..skin.rest.len() as u32).min_by(|&a, &b| {
                let d = |j: u32| (glam::Vec3::from(skin.rest[j as usize].position) - p).length_squared();
                d(a).total_cmp(&d(b))
            })
        };
        let influences: Vec<crate::skeleton::Influence> = rest
            .iter()
            .map(|v| {
                by_key
                    .get(&key(v))
                    .copied()
                    .or_else(|| nearest(glam::Vec3::from(v.position)))
                    .map_or_else(Default::default, |j| skin.influences[j as usize])
            })
            .collect();
        part.skin = Some(crate::skeleton::Skin {
            influences: std::sync::Arc::new(influences),
            rest: std::sync::Arc::new(rest),
            pose: None,
            ..skin.clone()
        });
        attached.push(i);
    }
    attached
}

/// Identity of one texture set: an object/material pair.
#[derive(Clone, PartialEq, Eq)]
pub struct SetKey {
    pub object: String,
    pub material: String,
}

/// Unique texture-set keys for a scene, in first-appearance order.
pub fn set_keys(models: &[ModelData]) -> Vec<SetKey> {
    let mut keys: Vec<SetKey> = Vec::new();
    for model in models {
        let key = SetKey {
            object: model.name.clone(),
            material: model.material_name.clone(),
        };
        if !keys.contains(&key) {
            keys.push(key);
        }
    }
    keys
}

/// Which texture set each model belongs to, matching `set_keys` order.
pub fn set_indices(models: &[ModelData]) -> Vec<usize> {
    let keys = set_keys(models);
    models
        .iter()
        .map(|m| {
            keys.iter()
                .position(|k| k.object == m.name && k.material == m.material_name)
                .unwrap()
        })
        .collect()
}

#[cfg(test)]
mod morph_tests {
    use super::*;
    use crate::mesh::Vertex3D;
    use std::sync::Arc;

    fn part() -> ModelData {
        let v = |x: f32| Vertex3D { position: [x, 0.0, 0.0], normal: [0.0, 0.0, 1.0], uv: [x, 0.0] };
        let vertices = vec![v(0.0), v(1.0), v(2.0)];
        ModelData {
            mesh: crate::mesh::MeshData { vertices: vertices.clone(), indices: vec![0, 1, 2] },
            material: MaterialData::default(),
            transform: Mat4::IDENTITY,
            name: "Face".into(),
            material_name: "Skin".into(),
            skin: None,
            morphs: Morphs {
                base: Arc::new(vertices),
                targets: vec![
                    Morph { name: "Smile".into(), weight: 0.0, offsets: Arc::new(vec![(1, [0.0, 1.0, 0.0], [0.0; 3])]) },
                    Morph { name: "Blink".into(), weight: 0.0, offsets: Arc::new(vec![(1, [0.0, 0.0, 2.0], [0.0; 3]), (2, [1.0, 0.0, 0.0], [0.0; 3])]) },
                ],
            },
        }
    }

    /// Weights blend their offsets over the base, from scratch every time.
    #[test]
    fn weights_deform_the_base() {
        let mut m = part();
        m.morphs.targets[0].weight = 0.5;
        m.morphs.targets[1].weight = 1.0;
        assert!(m.deform());
        assert_eq!(m.mesh.vertices[1].position, [1.0, 0.5, 2.0]);
        assert_eq!(m.mesh.vertices[2].position, [3.0, 0.0, 0.0]);
        m.morphs.targets[1].weight = 0.0;
        m.deform();
        assert_eq!(m.mesh.vertices[1].position, [1.0, 0.5, 0.0], "not accumulated across applies");
        assert_eq!(m.mesh.vertices[0].position, [0.0, 0.0, 0.0]);
    }

    /// Shapes from the model file attach to a project's part by name and
    /// vertex position, even with vertices in another order.
    #[test]
    fn shapes_attach_from_the_source_file() {
        let source = part();
        let mut project = part();
        project.morphs = Morphs::default();
        project.mesh.vertices.reverse();
        let mut other = part();
        other.name = "Hair".into();
        other.morphs = Morphs::default();
        let mut target = vec![project, other];
        assert_eq!(attach_morphs(&mut target, &[source]), [0]);
        let blink: Vec<u32> = target[0].morphs.targets[1].offsets.iter().map(|o| o.0).collect();
        assert_eq!(blink, [1, 0], "old vertices 1 and 2 are now 1 and 0");
        assert!(target[1].morphs.targets.is_empty(), "other objects are left alone");
    }

    /// Rebuilding the vertices (UDIM split, re-unwrap) carries every offset
    /// to each copy of its vertex, with the new UVs in the base.
    #[test]
    fn remapping_follows_vertex_copies() {
        let m = part();
        let new_vertices: Vec<Vertex3D> =
            [1u32, 1, 2].iter().enumerate().map(|(j, &old)| Vertex3D { uv: [9.0, j as f32], ..m.morphs.base[old as usize] }).collect();
        let remapped = m.morphs.remapped(&[1, 1, 2], &new_vertices);
        assert_eq!(remapped.base.len(), 3);
        assert_eq!(remapped.base[1].uv, [9.0, 1.0]);
        let smile: Vec<u32> = remapped.targets[0].offsets.iter().map(|o| o.0).collect();
        assert_eq!(smile, [0, 1], "both copies of vertex 1 move");
        let blink: Vec<u32> = remapped.targets[1].offsets.iter().map(|o| o.0).collect();
        assert_eq!(blink, [0, 1, 2]);
    }
}

fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
}

fn linear_to_srgb(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.0031308 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 }
}

#[cfg(test)]
mod emission_tests {
    use super::*;

    fn image(w: u32, h: u32, px: [u8; 4]) -> ImageData {
        ImageData { width: w, height: h, rgba: px.repeat((w * h) as usize).into() }
    }

    #[test]
    fn emission_texture_is_map_times_mask() {
        let mut m = MaterialData::default();
        assert!(m.emission_texture().is_none(), "no maps: plain colour, white texture");

        m.emission_image = Some(image(4, 4, [255, 128, 0, 255]));
        assert_eq!(&m.emission_texture().unwrap().rgba[..4], &[255, 128, 0, 255], "map alone passes through");

        // Left half masked off, right half on, at a different resolution.
        let mut mask = image(2, 1, [0, 0, 0, 255]);
        mask.rgba = [0, 0, 0, 255, 255, 255, 255, 255].to_vec().into();
        m.emission_mask_image = Some(mask);
        let t = m.emission_texture().unwrap();
        assert_eq!((t.width, t.height), (4, 4), "combined at the map's size");
        assert_eq!(&t.rgba[..4], &[0, 0, 0, 255], "masked texel is dark");
        assert_eq!(&t.rgba[12..16], &[255, 128, 0, 255], "unmasked texel keeps the map colour");

        m.emission_image = None;
        let t = m.emission_texture().unwrap();
        assert_eq!((t.width, t.height), (2, 1), "mask alone keeps its size");
        assert_eq!(&t.rgba[4..8], &[255, 255, 255, 255]);
    }
}
