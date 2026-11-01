use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Context as _, Result};
use glam::{Mat4, Vec4};

use crate::mesh::{MeshData, Vertex3D};
use crate::scene::{ImageData, MaterialData, ModelData, Morph, Morphs};
use crate::skeleton::{Influence, Joint, Skeleton, Skin, Trs};

/// Load an FBX file via ufbx. Geometry is normalized to meters / right-handed
/// Y-up so FBX exports (usually centimeters, sometimes Z-up) match the glTF
/// convention the rest of the app uses. Each mesh node becomes one ModelData
/// named after the node, which is what texture-set grouping keys on.
/// Load a model, reporting stage and item counts so a caller can show a bar.
/// Pass `&|_, _, _| {}` when progress is not needed.
pub fn load_with_progress(
    path: &Path,
    progress: crate::scene::ProgressFn,
) -> Result<Vec<ModelData>> {
    let opts = ufbx::LoadOpts {
        generate_missing_normals: true,
        space_conversion: ufbx::SpaceConversion::TransformRoot,
        target_axes: ufbx::axes_right_handed_y_up(),
        target_unit_meters: 1.0,
        ..Default::default()
    };
    let path_str = path.to_str().context("path is not valid UTF-8")?;
    // Parsing is the long pole and gives no callbacks, so name the stage first.
    progress("Reading file", 0, 0);
    let scene = ufbx::load_file(path_str, opts)
        .map_err(|e| anyhow::anyhow!("ufbx failed to load {}: {:?}", path.display(), e.description))?;

    let rig = FbxRig::build(&scene);
    let mut models = Vec::new();
    // Materials and textures are shared between nodes: convert and decode
    // each once (clones share the pixels).
    let mut materials: HashMap<u32, MaterialData> = HashMap::new();
    let textures = TextureCache::default();
    let node_count = scene.nodes.as_ref().len();
    for (node_index, node) in scene.nodes.as_ref().iter().enumerate() {
        progress("Building meshes", node_index, node_count);
        let Some(mesh) = &node.mesh else { continue };
        if mesh.num_triangles == 0 {
            continue;
        }

        let transform = matrix_to_mat4(&node.geometry_to_world);
        let name = if !node.element.name.is_empty() {
            node.element.name.to_string()
        } else if !mesh.element.name.is_empty() {
            mesh.element.name.to_string()
        } else {
            format!("Mesh {node_index}")
        };
        let fbx_dir = path.parent().unwrap_or(Path::new("."));

        // One ModelData per material part, so each material of an object is
        // its own texture set. Meshes without material assignments get one
        // synthetic part covering every face.
        let all_faces: Vec<u32> = (0..mesh.faces.len() as u32).collect();
        let parts: Vec<(u32, &[u32])> = if mesh.material_parts.is_empty() {
            vec![(0, all_faces.as_slice())]
        } else {
            mesh.material_parts
                .as_ref()
                .iter()
                .filter(|p| p.num_triangles > 0)
                .map(|p| (p.index, p.face_indices.as_ref()))
                .collect()
        };

        let shapes = blend_shapes(mesh);
        let mut tri_buf = vec![0u32; mesh.max_face_triangles.max(1) * 3];
        for (material_index, face_indices) in parts {
            let mut vertices = Vec::new();
            let mut indices = Vec::new();
            // The control point each vertex came from, for its blendshapes.
            let mut control_points = Vec::new();
            // Corners with identical position/normal/UV become one vertex —
            // unless they are different control points, which blendshapes
            // can pull apart (closed lips, eyelids).
            let mut seen: HashMap<[u32; 9], u32> = HashMap::new();
            for &face_index in face_indices {
                let face = mesh.faces[face_index as usize];
                if face.num_indices < 3 {
                    continue;
                }
                let tri_count = mesh.triangulate_face(&mut tri_buf, face) as usize;
                for &index in &tri_buf[..tri_count * 3] {
                    let index = index as usize;
                    let p = &mesh.vertex_position[index];
                    let n = if mesh.vertex_normal.exists {
                        let n = &mesh.vertex_normal[index];
                        [n.x as f32, n.y as f32, n.z as f32]
                    } else {
                        [0.0, 1.0, 0.0]
                    };
                    let uv = if mesh.vertex_uv.exists {
                        let uv = &mesh.vertex_uv[index];
                        // FBX UV origin is bottom-left; our textures are top-left.
                        [uv.x as f32, 1.0 - uv.y as f32]
                    } else {
                        [0.0, 0.0]
                    };
                    let position = [p.x as f32, p.y as f32, p.z as f32];
                    let control_point = mesh.vertex_indices[index];
                    let key = [
                        control_point,
                        position[0].to_bits(),
                        position[1].to_bits(),
                        position[2].to_bits(),
                        n[0].to_bits(),
                        n[1].to_bits(),
                        n[2].to_bits(),
                        uv[0].to_bits(),
                        uv[1].to_bits(),
                    ];
                    let next = vertices.len() as u32;
                    let index = *seen.entry(key).or_insert(next);
                    if index == next {
                        vertices.push(Vertex3D { position, normal: n, uv });
                        control_points.push(control_point);
                    }
                    indices.push(index);
                }
            }
            if vertices.is_empty() {
                continue;
            }

            let fbx_material = node.materials.as_ref().get(material_index as usize);
            let material = fbx_material
                .map(|m| {
                    materials
                        .entry(m.element.element_id)
                        .or_insert_with(|| convert_material(m, fbx_dir, &textures))
                        .clone()
                })
                .unwrap_or_default();
            let material_name = fbx_material
                .map(|m| m.element.name.to_string())
                .filter(|n| !n.is_empty())
                .unwrap_or_else(|| format!("Material {material_index}"));

            // Each shape's offsets for the vertices of this part; shapes
            // that don't touch the part are left out.
            let targets: Vec<Morph> = shapes
                .iter()
                .filter_map(|(shape_name, weight, by_point)| {
                    let offsets: Vec<_> = control_points
                        .iter()
                        .enumerate()
                        .filter_map(|(v, cp)| by_point.get(cp).map(|&(dp, dn)| (v as u32, dp, dn)))
                        .collect();
                    (!offsets.is_empty()).then(|| Morph { name: shape_name.clone(), weight: *weight, offsets: std::sync::Arc::new(offsets) })
                })
                .collect();
            let morphs = if targets.is_empty() {
                Morphs::default()
            } else {
                Morphs { base: std::sync::Arc::new(vertices.clone()), targets }
            };
            let skin = rig.as_ref().and_then(|rig| rig.skin(mesh, &control_points, &vertices));
            let mut model = ModelData {
                mesh: MeshData { vertices, indices },
                material,
                transform,
                name: name.clone(),
                material_name,
                morphs,
                skin,
            };
            model.deform();
            models.push(model);
        }
    }

    if models.is_empty() {
        bail!("FBX file contains no triangle meshes");
    }
    Ok(models)
}

/// The file's skeleton: every bone a skin uses and the nodes above it.
struct FbxRig {
    skeleton: std::sync::Arc<Skeleton>,
    /// Skeleton joint by node (`typed_id`).
    joint_of: HashMap<u32, usize>,
}

impl FbxRig {
    fn build(scene: &ufbx::Scene) -> Option<FbxRig> {
        // Rest (bind) world matrix of each bone a cluster uses: the one that
        // leaves the mesh as stored, mesh node · geometry-to-bone⁻¹.
        let mut bind_world: HashMap<u32, Mat4> = HashMap::new();
        for mesh in scene.meshes.as_ref().iter() {
            let Some(instance) = mesh.element.instances.as_ref().first() else { continue };
            let mesh_world = matrix_to_mat4(&instance.geometry_to_world);
            for skin in mesh.skin_deformers.as_ref().iter() {
                for cluster in skin.clusters.as_ref().iter() {
                    if let Some(bone) = &cluster.bone_node {
                        bind_world.entry(bone.element.typed_id).or_insert_with(|| mesh_world * matrix_to_mat4(&cluster.geometry_to_bone).inverse());
                    }
                }
            }
        }
        if bind_world.is_empty() {
            return None;
        }
        let nodes = scene.nodes.as_ref();
        let mut included = vec![false; nodes.len()];
        for &id in bind_world.keys() {
            let mut node = Some(&nodes[id as usize]);
            while let Some(n) = node {
                if n.is_root || included[n.element.typed_id as usize] {
                    break;
                }
                included[n.element.typed_id as usize] = true;
                node = n.parent.as_ref().map(|p| &nodes[p.element.typed_id as usize]);
            }
        }
        let mut order: Vec<usize> = (0..nodes.len()).filter(|&i| included[i]).collect();
        order.sort_by_key(|&i| nodes[i].node_depth);
        let mut joint_of = HashMap::new();
        let mut joints: Vec<Joint> = Vec::new();
        let mut world: Vec<Mat4> = Vec::new();
        for &i in &order {
            let node = &nodes[i];
            let parent = node.parent.as_ref().and_then(|p| joint_of.get(&p.element.typed_id).copied());
            let base = match (parent, &node.parent) {
                (None, Some(p)) => matrix_to_mat4(&p.node_to_world),
                _ => Mat4::IDENTITY,
            };
            let parent_world = parent.map_or(base, |p| world[p]);
            let w = bind_world.get(&node.element.typed_id).copied().unwrap_or_else(|| parent_world * matrix_to_mat4(&node.node_to_parent));
            joint_of.insert(node.element.typed_id, joints.len());
            joints.push(Joint {
                name: node.element.name.to_string(),
                parent,
                rest: Trs::from_matrix(&(parent_world.inverse() * w)),
                base,
            });
            world.push(w);
        }
        Some(FbxRig { skeleton: std::sync::Arc::new(Skeleton { joints }), joint_of })
    }

    /// The skin of one part: `control_points` are its vertices' sources.
    fn skin(&self, mesh: &ufbx::Mesh, control_points: &[u32], vertices: &[Vertex3D]) -> Option<Skin> {
        let deformer = mesh.skin_deformers.as_ref().first()?;
        let clusters = deformer.clusters.as_ref();
        let mut slot_of: HashMap<u32, u16> = HashMap::new();
        let mut joints = Vec::new();
        let mut binds = Vec::new();
        let mut pairs = Vec::with_capacity(8);
        let influences = control_points
            .iter()
            .map(|&cp| {
                pairs.clear();
                if let Some(sv) = deformer.vertices.as_ref().get(cp as usize) {
                    for k in 0..sv.num_weights {
                        let Some(weight) = deformer.weights.as_ref().get((sv.weight_begin + k) as usize) else { continue };
                        let cluster = &clusters[weight.cluster_index as usize];
                        let Some(joint) = cluster.bone_node.as_ref().and_then(|b| self.joint_of.get(&b.element.typed_id)) else { continue };
                        let slot = *slot_of.entry(weight.cluster_index).or_insert_with(|| {
                            joints.push(*joint as u32);
                            binds.push(matrix_to_mat4(&cluster.geometry_to_bone));
                            (joints.len() - 1) as u16
                        });
                        pairs.push((slot, weight.weight as f32));
                    }
                }
                Influence::strongest(&mut pairs)
            })
            .collect::<Vec<_>>();
        if joints.is_empty() {
            return None;
        }
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

/// Per-control-point offsets: (position offset, normal offset).
type PointOffsets = HashMap<u32, ([f32; 3], [f32; 3])>;

/// A mesh's blendshape channels: (name, weight as saved in the file,
/// offsets by control point). A channel with in-between shapes uses its
/// full-weight target.
fn blend_shapes(mesh: &ufbx::Mesh) -> Vec<(String, f32, PointOffsets)> {
    let mut out: Vec<(String, f32, PointOffsets)> = Vec::new();
    for deformer in mesh.blend_deformers.as_ref().iter() {
        for channel in deformer.channels.as_ref().iter() {
            let shape = match &channel.target_shape {
                Some(shape) => shape,
                None => match channel.keyframes.as_ref().last() {
                    Some(key) => &key.shape,
                    None => continue,
                },
            };
            let mut by_point = PointOffsets::new();
            for (i, &cp) in shape.offset_vertices.as_ref().iter().enumerate() {
                let w = shape.offset_weights.as_ref().get(i).copied().unwrap_or(1.0) as f32;
                let p = shape.position_offsets.as_ref().get(i).map_or([0.0; 3], |v| [v.x as f32 * w, v.y as f32 * w, v.z as f32 * w]);
                let n = shape.normal_offsets.as_ref().get(i).map_or([0.0; 3], |v| [v.x as f32 * w, v.y as f32 * w, v.z as f32 * w]);
                if p != [0.0; 3] || n != [0.0; 3] {
                    by_point.insert(cp, (p, n));
                }
            }
            out.push((channel.element.name.to_string(), channel.weight as f32, by_point));
        }
    }
    out
}

/// Decoded textures by source — an embedded blob's element or a file path.
/// Exporters often give each material its own texture element pointing at
/// one shared atlas, so the element id alone would miss most repeats.
type TextureCache = RefCell<HashMap<String, Option<ImageData>>>;

fn convert_material(material: &ufbx::Material, fbx_dir: &Path, textures: &TextureCache) -> MaterialData {
    let pbr = &material.pbr;
    let base = if pbr.base_color.has_value {
        vec4_to_array(pbr.base_color.value_vec4)
    } else if material.fbx.diffuse_color.has_value {
        vec4_to_array(material.fbx.diffuse_color.value_vec4)
    } else {
        [0.8, 0.8, 0.8, 1.0]
    };

    let tex = |map: &ufbx::MaterialMap| -> Option<crate::scene::ImageData> {
        map.texture.as_ref().and_then(|t| load_texture_image(t, fbx_dir, textures))
    };

    // Surface opacity: PBR opacity slot first, else classic FBX transparency.
    let opacity = if pbr.opacity.has_value {
        (pbr.opacity.value_vec4.x as f32).clamp(0.0, 1.0)
    } else if material.fbx.transparency_factor.has_value {
        (1.0 - material.fbx.transparency_factor.value_vec4.x as f32).clamp(0.0, 1.0)
    } else {
        1.0
    };

    let base_color_image =
        tex(&pbr.base_color).or_else(|| tex(&material.fbx.diffuse_color));
    let roughness_image = tex(&pbr.roughness);
    let metallic_image = tex(&pbr.metalness);
    let normal_image = tex(&pbr.normal_map)
        .or_else(|| tex(&material.fbx.normal_map))
        .or_else(|| tex(&material.fbx.bump));
    let height_image = tex(&pbr.displacement_map);
    let ao_image = tex(&pbr.ambient_occlusion);
    // Emission = colour (× its map) × strength (× its mask), as Blender's
    // Emission Color and Emission Strength sockets export.
    let emission_image = tex(&pbr.emission_color).or_else(|| tex(&material.fbx.emission_color));
    let emission_mask_image = tex(&pbr.emission_factor).or_else(|| tex(&material.fbx.emission_factor));

    MaterialData {
        // When a texture is present its factor often defaults to black/white
        // noise from exporters; keep the factor only when there's no texture.
        base_color: if base_color_image.is_some() { [1.0; 4] } else { base },
        metallic: if metallic_image.is_some() {
            1.0
        } else if pbr.metalness.has_value {
            pbr.metalness.value_vec4.x as f32
        } else {
            0.0
        },
        roughness: if roughness_image.is_some() {
            1.0
        } else if pbr.roughness.has_value {
            pbr.roughness.value_vec4.x as f32
        } else {
            0.6
        },
        base_color_image,
        roughness_image,
        metallic_image,
        normal_image,
        height_image,
        // A linked socket's own value is just its unused default, so a map or
        // mask takes over from it (as with the base colour).
        emission: if pbr.emission_color.has_value || emission_image.is_some() {
            let f = if emission_mask_image.is_some() {
                1.0
            } else if pbr.emission_factor.has_value {
                pbr.emission_factor.value_vec4.x as f32
            } else {
                1.0
            };
            let e = if emission_image.is_some() { [1.0; 3] } else { {
                let e = pbr.emission_color.value_vec4;
                [e.x as f32, e.y as f32, e.z as f32]
            } };
            e.map(|v| v * f)
        } else {
            [0.0; 3]
        },
        emission_image,
        emission_mask_image,
        ao_image,
        specular_weight: if pbr.specular_factor.has_value {
            pbr.specular_factor.value_vec4.x as f32
        } else {
            1.0
        },
        coat_weight: if pbr.coat_factor.has_value {
            pbr.coat_factor.value_vec4.x as f32
        } else {
            0.0
        },
        coat_roughness: if pbr.coat_roughness.has_value {
            pbr.coat_roughness.value_vec4.x as f32
        } else {
            0.1
        },
        sheen_weight: if pbr.sheen_factor.has_value {
            pbr.sheen_factor.value_vec4.x as f32
        } else {
            0.0
        },
        sheen_roughness: if pbr.sheen_roughness.has_value {
            pbr.sheen_roughness.value_vec4.x as f32
        } else {
            0.3
        },
        opacity,
        transparent: opacity < 0.999,
    }
}

/// Resolve a ufbx texture to pixels: embedded content first (texture blob,
/// then its video's blob), then files on disk (absolute path, path relative
/// to the FBX, then just the file name next to the FBX — the common case for
/// models shipped in a folder with their textures).
fn load_texture_image(texture: &ufbx::Texture, fbx_dir: &Path, cache: &TextureCache) -> Option<ImageData> {
    let embedded: Option<&[u8]> = if !texture.content.is_empty() {
        Some(&texture.content)
    } else {
        texture
            .video
            .as_ref()
            .filter(|v| !v.content.is_empty())
            .map(|v| &*v.content)
    };

    // Resolve the source first: it is the cache key. An embedded blob is
    // keyed by its address — textures sharing one video share the blob.
    let (key, file) = if let Some(bytes) = embedded {
        (format!("embedded:{:p}", bytes.as_ptr()), None)
    } else {
        let mut candidates = Vec::new();
        if !texture.absolute_filename.is_empty() {
            candidates.push(std::path::PathBuf::from(texture.absolute_filename.to_string()));
        }
        if !texture.relative_filename.is_empty() {
            let rel = texture.relative_filename.to_string().replace('\\', "/");
            candidates.push(fbx_dir.join(&rel));
            if let Some(name) = std::path::Path::new(&rel).file_name() {
                candidates.push(fbx_dir.join(name));
            }
        }
        let path = candidates.into_iter().find(|p| p.is_file())?;
        (format!("file:{}", path.display()), Some(path))
    };
    if let Some(hit) = cache.borrow().get(&key) {
        return hit.clone();
    }

    let decoded = match (embedded, &file) {
        (Some(bytes), _) => image::load_from_memory(bytes).ok(),
        (None, Some(path)) => image::open(path).ok(),
        (None, None) => None,
    };
    let image = decoded.map(|decoded| {
        let rgba = decoded.to_rgba8();
        let (width, height) = rgba.dimensions();
        ImageData { width, height, rgba: rgba.into_raw().into() }
    });
    cache.borrow_mut().insert(key, image.clone());
    image
}

fn vec4_to_array(v: ufbx::Vec4) -> [f32; 4] {
    [v.x as f32, v.y as f32, v.z as f32, v.w as f32]
}

#[cfg(test)]
mod tests {
    /// Smoke test against a real file: ZESTSURFACE_TEST_FBX=/path/to/file.fbx
    #[test]
    fn load_real_fbx() {
        let Ok(path) = std::env::var("ZESTSURFACE_TEST_FBX") else {
            eprintln!("ZESTSURFACE_TEST_FBX not set; skipping");
            return;
        };
        let models = super::load_with_progress(std::path::Path::new(&path), &|_, _, _| {})
            .expect("FBX should load");
        for m in &models {
            let (lo, hi) = m.mesh.bounds();
            let tex = match &m.material.base_color_image {
                Some(img) => format!("texture {}x{}", img.width, img.height),
                None => "no texture".into(),
            };
            let _ = (lo, hi);
            let moved = m.morphs.targets.iter().filter(|t| t.weight != 0.0).count();
            eprintln!(
                "{} · {}: {} verts, {} tris, {tex}, {} blendshapes ({moved} on){}",
                m.name,
                m.material_name,
                m.mesh.vertices.len(),
                m.mesh.indices.len() / 3,
                m.morphs.targets.len(),
                m.morphs.targets.first().map(|t| format!(", e.g. {:?} moves {} verts", t.name, t.offsets.len())).unwrap_or_default(),
            );
        }
        assert!(!models.is_empty());
    }
}

/// ufbx matrices are column-major 3x4 (rotation/scale columns + translation).
pub(crate) fn matrix_to_mat4(m: &ufbx::Matrix) -> Mat4 {
    Mat4::from_cols(
        Vec4::new(m.m00 as f32, m.m10 as f32, m.m20 as f32, 0.0),
        Vec4::new(m.m01 as f32, m.m11 as f32, m.m21 as f32, 0.0),
        Vec4::new(m.m02 as f32, m.m12 as f32, m.m22 as f32, 0.0),
        Vec4::new(m.m03 as f32, m.m13 as f32, m.m23 as f32, 1.0),
    )
}

#[cfg(test)]
mod rig_tests {
    /// Real-file check: ZESTSURFACE_TEST_FBX=/path/to/avatar.fbx. Prints the
    /// humanoid bones found and poses the mesh with every built-in pose,
    /// which must keep the mesh together (no vertex flung far off).
    #[test]
    fn rig_real_fbx() {
        let Ok(path) = std::env::var("ZESTSURFACE_TEST_FBX") else {
            eprintln!("ZESTSURFACE_TEST_FBX not set; skipping");
            return;
        };
        let mut models = super::load_with_progress(std::path::Path::new(&path), &|_, _, _| {}).unwrap();
        let Some(skin) = models.iter().find_map(|m| m.skin.clone()) else {
            panic!("no skin in {path}");
        };
        let s = &skin.skeleton;
        let names: Vec<String> = s.joints.iter().map(|j| j.name.clone()).collect();
        let parents: Vec<Option<usize>> = s.joints.iter().map(|j| j.parent).collect();
        let map = crate::skeleton::map_humanoid(&names, &parents);
        eprintln!("{} joints, {} skinned parts", s.joints.len(), models.iter().filter(|m| m.skin.is_some()).count());
        for (b, j) in map.iter().enumerate() {
            eprintln!("  {:<14} {}", crate::skeleton::BONES[b], j.map_or("—".into(), |j| names[j].clone()));
        }
        let bounds = |models: &[crate::scene::ModelData]| {
            models.iter().fold((glam::Vec3::splat(f32::MAX), glam::Vec3::splat(f32::MIN)), |(lo, hi), m| {
                m.mesh.vertices.iter().fold((lo, hi), |(lo, hi), v| {
                    let p = m.transform.transform_point3(glam::Vec3::from(v.position));
                    (lo.min(p), hi.max(p))
                })
            })
        };
        let rest = bounds(&models);
        eprintln!("rest bounds {rest:?}");
        for pose in crate::poses::builtin() {
            let locals = crate::skeleton::retarget(Some(&pose), s, &map, false, &Default::default());
            let world = std::sync::Arc::new(s.world(&locals));
            for m in models.iter_mut() {
                if let Some(skin) = &mut m.skin {
                    skin.pose = Some(world.clone());
                }
                m.deform();
            }
            let posed = bounds(&models);
            eprintln!("{:<20} bounds {posed:?}", pose.name);
            let size = (rest.1 - rest.0).length();
            assert!((posed.1 - posed.0).length() < size * 1.6, "{} flings vertices: {posed:?}", pose.name);
        }
    }
}
