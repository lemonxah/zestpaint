use std::path::Path;

use anyhow::{Context as _, Result, bail};
use glam::Mat4;

use crate::mesh::{MeshData, Vertex3D};
use crate::scene::{ImageData, MaterialData, ModelData, Morph, Morphs};
use crate::skeleton::{Influence, Joint, Skeleton, Skin, Trs};

/// Load a model, reporting stage and item counts so a caller can show a bar.
/// Pass `&|_, _, _| {}` when progress is not needed.
pub fn load_with_progress(
    path: &Path,
    progress: crate::scene::ProgressFn,
) -> Result<Vec<ModelData>> {
    progress("Reading file", 0, 0);
    let (doc, buffers, images) = gltf::import(path)
        .with_context(|| format!("failed to import {}", path.display()))?;

    let scene = doc
        .default_scene()
        .or_else(|| doc.scenes().next())
        .context("glTF file contains no scenes")?;

    let mut models = Vec::new();
    let roots: Vec<_> = scene.nodes().collect();
    let rig = GltfRig::build(&doc, &scene, &buffers);
    for (i, node) in roots.iter().enumerate() {
        progress("Building meshes", i, roots.len());
        visit_node(node, Mat4::IDENTITY, &buffers, &images, rig.as_ref(), &mut models)?;
    }
    if models.is_empty() {
        bail!("glTF scene contains no triangle meshes");
    }
    Ok(models)
}

fn visit_node(
    node: &gltf::Node,
    parent: Mat4,
    buffers: &[gltf::buffer::Data],
    images: &[gltf::image::Data],
    rig: Option<&GltfRig>,
    out: &mut Vec<ModelData>,
) -> Result<()> {
    let transform = parent * Mat4::from_cols_array_2d(&node.transform().matrix());

    if let Some(mesh) = node.mesh() {
        for prim in mesh.primitives() {
            if prim.mode() != gltf::mesh::Mode::Triangles {
                continue;
            }
            let reader = prim.reader(|b| Some(&buffers[b.index()].0));

            let Some(positions) = reader.read_positions() else { continue };
            let positions: Vec<[f32; 3]> = positions.collect();
            let normals: Vec<[f32; 3]> = match reader.read_normals() {
                Some(n) => n.collect(),
                None => compute_flat_normals(&positions),
            };
            let uvs: Vec<[f32; 2]> = match reader.read_tex_coords(0) {
                Some(t) => t.into_f32().collect(),
                None => vec![[0.0, 0.0]; positions.len()],
            };

            let vertices: Vec<Vertex3D> = positions
                .iter()
                .enumerate()
                .map(|(i, p)| Vertex3D {
                    position: *p,
                    normal: normals.get(i).copied().unwrap_or([0.0, 1.0, 0.0]),
                    uv: uvs.get(i).copied().unwrap_or([0.0, 0.0]),
                })
                .collect();

            let indices: Vec<u32> = match reader.read_indices() {
                Some(idx) => idx.into_u32().collect(),
                None => (0..vertices.len() as u32).collect(),
            };

            let name = node
                .name()
                .or_else(|| mesh.name())
                .map(str::to_owned)
                .unwrap_or_else(|| format!("Mesh {}", mesh.index()));
            let material = prim.material();
            let material_name = material
                .name()
                .map(str::to_owned)
                .unwrap_or_else(|| match material.index() {
                    Some(i) => format!("Material {i}"),
                    None => "Default".into(),
                });
            let morphs = morph_targets(&reader, &mesh, node, &vertices);
            let skin = match (rig, node.skin()) {
                (Some(rig), Some(skin)) => rig.skin(&skin, &reader, buffers, &vertices),
                _ => None,
            };
            let mut model = ModelData {
                mesh: MeshData { vertices, indices },
                material: load_material(&material, images),
                transform,
                name,
                material_name,
                morphs,
                skin,
            };
            model.deform();
            out.push(model);
        }
    }

    for child in node.children() {
        visit_node(&child, transform, buffers, images, rig, out)?;
    }
    Ok(())
}

/// The file's skeleton: every joint a skin uses and the nodes above it.
struct GltfRig {
    skeleton: std::sync::Arc<Skeleton>,
    /// Skeleton joint by node index.
    joint_of: std::collections::HashMap<usize, usize>,
}

/// Each node's parent and world matrix in `scene`.
pub(crate) fn node_tree(doc: &gltf::Document, scene: &gltf::Scene) -> (Vec<Option<usize>>, Vec<Mat4>, Vec<usize>) {
    let count = doc.nodes().len();
    let mut parent = vec![None; count];
    let mut world = vec![Mat4::IDENTITY; count];
    let mut depth = vec![0usize; count];
    let mut stack: Vec<(gltf::Node, Mat4, Option<usize>, usize)> = scene.nodes().map(|n| (n, Mat4::IDENTITY, None, 0)).collect();
    while let Some((node, up, p, d)) = stack.pop() {
        let i = node.index();
        let w = up * Mat4::from_cols_array_2d(&node.transform().matrix());
        parent[i] = p;
        world[i] = w;
        depth[i] = d;
        stack.extend(node.children().map(|c| (c, w, Some(i), d + 1)));
    }
    (parent, world, depth)
}

fn inverse_binds(skin: &gltf::Skin, buffers: &[gltf::buffer::Data]) -> Vec<Mat4> {
    let count = skin.joints().len();
    skin.reader(|b| Some(&buffers[b.index()].0))
        .read_inverse_bind_matrices()
        .map(|m| m.map(|c| Mat4::from_cols_array_2d(&c)).collect())
        .unwrap_or_else(|| vec![Mat4::IDENTITY; count])
}

impl GltfRig {
    fn build(doc: &gltf::Document, scene: &gltf::Scene, buffers: &[gltf::buffer::Data]) -> Option<GltfRig> {
        let (parent, node_world, depth) = node_tree(doc, scene);
        // Rest world of each joint: the one that leaves the mesh as stored,
        // mesh node · inverse bind⁻¹.
        let mut bind_world: std::collections::HashMap<usize, Mat4> = std::collections::HashMap::new();
        for node in doc.nodes() {
            let Some(skin) = node.skin() else { continue };
            let mesh_world = node_world[node.index()];
            for (joint, ibm) in skin.joints().zip(inverse_binds(&skin, buffers)) {
                bind_world.entry(joint.index()).or_insert_with(|| mesh_world * ibm.inverse());
            }
        }
        if bind_world.is_empty() {
            return None;
        }
        let mut included = vec![false; parent.len()];
        for &j in bind_world.keys() {
            let mut at = Some(j);
            while let Some(i) = at {
                if included[i] {
                    break;
                }
                included[i] = true;
                at = parent[i];
            }
        }
        let mut order: Vec<usize> = (0..parent.len()).filter(|&i| included[i]).collect();
        order.sort_by_key(|&i| depth[i]);
        let names: Vec<String> = doc.nodes().map(|n| n.name().map(str::to_owned).unwrap_or_else(|| format!("Node {}", n.index()))).collect();
        let mut joint_of = std::collections::HashMap::new();
        let mut joints: Vec<Joint> = Vec::new();
        let mut world: Vec<Mat4> = Vec::new();
        for &i in &order {
            let p = parent[i].and_then(|p| joint_of.get(&p).copied());
            let parent_world = p.map_or(Mat4::IDENTITY, |p| world[p]);
            let w = bind_world.get(&i).copied().unwrap_or_else(|| parent_world * node_world[parent[i].unwrap_or(i)].inverse() * node_world[i]);
            let w = if parent[i].is_none() && !bind_world.contains_key(&i) { node_world[i] } else { w };
            joint_of.insert(i, joints.len());
            joints.push(Joint { name: names[i].clone(), parent: p, rest: Trs::from_matrix(&(parent_world.inverse() * w)), base: Mat4::IDENTITY });
            world.push(w);
        }
        Some(GltfRig { skeleton: std::sync::Arc::new(Skeleton { joints }), joint_of })
    }

    fn skin<'a, 's, F>(
        &self,
        skin: &gltf::Skin,
        reader: &gltf::mesh::Reader<'a, 's, F>,
        buffers: &[gltf::buffer::Data],
        vertices: &[Vertex3D],
    ) -> Option<Skin>
    where
        F: Clone + Fn(gltf::Buffer<'a>) -> Option<&'s [u8]>,
    {
        let joints: Vec<u32> = skin.joints().map(|j| self.joint_of.get(&j.index()).copied().unwrap_or(0) as u32).collect();
        let binds = inverse_binds(skin, buffers);
        let indices: Vec<[u16; 4]> = reader.read_joints(0)?.into_u16().collect();
        let weights: Vec<[f32; 4]> = reader.read_weights(0)?.into_f32().collect();
        let mut pairs = Vec::with_capacity(4);
        let influences = (0..vertices.len())
            .map(|v| {
                pairs.clear();
                if let (Some(j), Some(w)) = (indices.get(v), weights.get(v)) {
                    pairs.extend((0..4).filter(|&k| (j[k] as usize) < joints.len()).map(|k| (j[k], w[k])));
                }
                Influence::strongest(&mut pairs)
            })
            .collect();
        Some(Skin {
            skeleton: self.skeleton.clone(),
            joints,
            binds,
            influences: std::sync::Arc::new(influences),
            rest: std::sync::Arc::new(vertices.to_vec()),
            pose: None,
        })
    }
}

/// A primitive's morph targets, named from the mesh's `extras.targetNames`
/// (the de-facto convention; unnamed ones become "Shape N") and weighted as
/// the node, else the mesh, says.
fn morph_targets<'a, 's, F>(
    reader: &gltf::mesh::Reader<'a, 's, F>,
    mesh: &gltf::Mesh,
    node: &gltf::Node,
    vertices: &[Vertex3D],
) -> Morphs
where
    F: Clone + Fn(gltf::Buffer<'a>) -> Option<&'s [u8]>,
{
    let names: Vec<String> = mesh
        .extras()
        .as_ref()
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(raw.get()).ok())
        .and_then(|v| v["targetNames"].as_array().cloned())
        .map(|list| list.iter().map(|n| n.as_str().unwrap_or("").to_string()).collect())
        .unwrap_or_default();
    let weights = node.weights().or_else(|| mesh.weights()).unwrap_or(&[]);
    let targets: Vec<Morph> = reader
        .read_morph_targets()
        .enumerate()
        .filter_map(|(i, (positions, normals, _))| {
            let dp: Vec<[f32; 3]> = positions.map(|p| p.collect()).unwrap_or_default();
            let dn: Vec<[f32; 3]> = normals.map(|n| n.collect()).unwrap_or_default();
            let offsets: Vec<(u32, [f32; 3], [f32; 3])> = (0..vertices.len())
                .filter_map(|v| {
                    let p = dp.get(v).copied().unwrap_or([0.0; 3]);
                    let n = dn.get(v).copied().unwrap_or([0.0; 3]);
                    (p != [0.0; 3] || n != [0.0; 3]).then_some((v as u32, p, n))
                })
                .collect();
            let name = names.get(i).filter(|n| !n.is_empty()).cloned().unwrap_or_else(|| format!("Shape {i}"));
            (!offsets.is_empty()).then(|| Morph {
                name,
                weight: weights.get(i).copied().unwrap_or(0.0),
                offsets: std::sync::Arc::new(offsets),
            })
        })
        .collect();
    if targets.is_empty() {
        Morphs::default()
    } else {
        Morphs { base: std::sync::Arc::new(vertices.to_vec()), targets }
    }
}

fn load_material(material: &gltf::Material, images: &[gltf::image::Data]) -> MaterialData {
    let pbr = material.pbr_metallic_roughness();
    let base_color_image = pbr
        .base_color_texture()
        .and_then(|info| images.get(info.texture().source().index()))
        .and_then(to_rgba8);

    // glTF packs roughness in G and metallic in B of one texture; split them
    // into the grayscale maps our material model uses.
    let (roughness_image, metallic_image) = match pbr
        .metallic_roughness_texture()
        .and_then(|info| images.get(info.texture().source().index()))
        .and_then(to_rgba8)
    {
        Some(packed) => (
            Some(extract_channel(&packed, 1)),
            Some(extract_channel(&packed, 2)),
        ),
        None => (None, None),
    };

    let normal_image = material
        .normal_texture()
        .and_then(|info| images.get(info.texture().source().index()))
        .and_then(to_rgba8);
    let ao_image = material
        .occlusion_texture()
        .and_then(|info| images.get(info.texture().source().index()))
        .and_then(to_rgba8);
    // glTF's emissive factor is a real multiplier of its texture.
    let emission_image = material
        .emissive_texture()
        .and_then(|info| images.get(info.texture().source().index()))
        .and_then(to_rgba8);

    MaterialData {
        base_color: pbr.base_color_factor(),
        metallic: pbr.metallic_factor(),
        roughness: pbr.roughness_factor(),
        base_color_image,
        roughness_image,
        metallic_image,
        normal_image,
        height_image: None, // glTF has no standard height map
        ao_image,
        emission: material.emissive_factor(),
        emission_image,
        opacity: pbr.base_color_factor()[3],
        transparent: matches!(material.alpha_mode(), gltf::material::AlphaMode::Blend),
        ..Default::default()
    }
}

/// Copy one channel of an RGBA image into a grayscale RGBA image.
fn extract_channel(img: &ImageData, channel: usize) -> ImageData {
    let mut rgba = Vec::with_capacity(img.rgba.len());
    for px in img.rgba.chunks_exact(4) {
        let v = px[channel];
        rgba.extend_from_slice(&[v, v, v, 255]);
    }
    ImageData { width: img.width, height: img.height, rgba: rgba.into() }
}

fn to_rgba8(data: &gltf::image::Data) -> Option<ImageData> {
    use gltf::image::Format;
    let (width, height) = (data.width, data.height);
    let px = (width * height) as usize;
    let rgba = match data.format {
        Format::R8G8B8A8 => data.pixels.clone(),
        Format::R8G8B8 => {
            let mut v = Vec::with_capacity(px * 4);
            for c in data.pixels.chunks_exact(3) {
                v.extend_from_slice(&[c[0], c[1], c[2], 255]);
            }
            v
        }
        Format::R8 => {
            let mut v = Vec::with_capacity(px * 4);
            for &g in &data.pixels {
                v.extend_from_slice(&[g, g, g, 255]);
            }
            v
        }
        Format::R8G8 => {
            let mut v = Vec::with_capacity(px * 4);
            for c in data.pixels.chunks_exact(2) {
                v.extend_from_slice(&[c[0], c[1], 0, 255]);
            }
            v
        }
        // 16-bit and float formats are rare for base color; skip the texture
        // rather than failing the whole load.
        _ => return None,
    };
    Some(ImageData { width, height, rgba: rgba.into() })
}

fn compute_flat_normals(positions: &[[f32; 3]]) -> Vec<[f32; 3]> {
    use glam::Vec3;
    let mut normals = vec![[0.0, 1.0, 0.0]; positions.len()];
    for tri in (0..positions.len().saturating_sub(2)).step_by(3) {
        let a = Vec3::from(positions[tri]);
        let b = Vec3::from(positions[tri + 1]);
        let c = Vec3::from(positions[tri + 2]);
        let n = (b - a).cross(c - a).normalize_or_zero().to_array();
        normals[tri] = n;
        normals[tri + 1] = n;
        normals[tri + 2] = n;
    }
    normals
}
