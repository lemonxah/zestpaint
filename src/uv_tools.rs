use anyhow::{bail, Result};
use xatlas_rs_v2::{ChartOptions, IndexData, MeshData, MeshDecl, PackOptions, Xatlas};

use crate::mesh::{MeshData as CpuMesh, Vertex3D};
use crate::scene::{set_indices, ModelData};

/// Re-unwrap ONE texture set (sub-model) with xatlas, leaving every other
/// model untouched — so re-unwrapping a jacket doesn't invalidate the paint
/// on the body. The set gets its own atlas covering the full UV space.
/// Positions/normals are preserved; only UVs (and seam-duplicated vertices)
/// change.
pub fn auto_unwrap_set(models: &mut [ModelData], set: usize) -> Result<()> {
    if models.is_empty() {
        bail!("no models loaded");
    }
    let indices = set_indices(models);
    let group: Vec<&mut ModelData> = models
        .iter_mut()
        .zip(&indices)
        .filter(|(_, i)| **i == set)
        .map(|(m, _)| m)
        .collect();
    if group.is_empty() {
        bail!("texture set has no meshes");
    }
    unwrap_group(group)
}

/// Pack one group of models into a single non-overlapping atlas.
fn unwrap_group(mut models: Vec<&mut ModelData>) -> Result<()> {

    // Owned copies of the attribute arrays so the atlas borrows these, not
    // `models` — we mutate `models` while the atlas is still alive.
    let mut positions: Vec<Vec<f32>> = Vec::new();
    let mut normals: Vec<Vec<f32>> = Vec::new();
    let mut uvs: Vec<Vec<f32>> = Vec::new();
    let mut indices: Vec<Vec<u32>> = Vec::new();
    for model in models.iter() {
        positions.push(model.mesh.vertices.iter().flat_map(|v| v.position).collect());
        normals.push(model.mesh.vertices.iter().flat_map(|v| v.normal).collect());
        uvs.push(model.mesh.vertices.iter().flat_map(|v| v.uv).collect());
        indices.push(model.mesh.indices.clone());
    }

    let mut decls = Vec::new();
    for i in 0..models.len() {
        decls.push(MeshDecl {
            vertex_position_data: MeshData::Contiguous(&positions[i]),
            vertex_normal_data: Some(MeshData::Contiguous(&normals[i])),
            vertex_uv_data: Some(MeshData::Contiguous(&uvs[i])),
            index_data: Some(IndexData::U32(&indices[i])),
            face_count: indices[i].len() as u32 / 3,
            ..Default::default()
        });
    }

    let mut atlas = Xatlas::new();
    for decl in &decls {
        if let Err(e) = atlas.add_mesh(decl) {
            bail!("xatlas rejected mesh: {e:?}");
        }
    }
    atlas.generate(&ChartOptions::default(), &PackOptions::default());

    let (aw, ah) = (atlas.width() as f32, atlas.height() as f32);
    if aw < 1.0 || ah < 1.0 {
        bail!("xatlas produced an empty atlas");
    }

    let out_meshes = atlas.meshes();
    if out_meshes.len() != models.len() {
        bail!("xatlas returned {} meshes for {} inputs", out_meshes.len(), models.len());
    }

    for (model, out) in models.iter_mut().zip(out_meshes) {
        let old = &model.mesh.vertices;
        let vertices: Vec<Vertex3D> = out
            .vertex_array
            .iter()
            .map(|v| {
                let src = &old[v.xref as usize];
                Vertex3D {
                    position: src.position,
                    normal: src.normal,
                    // xatlas reports UVs in atlas texels.
                    uv: [v.uv[0] / aw, v.uv[1] / ah],
                }
            })
            .collect();
        let indices = out.index_array.to_vec();
        let sources: Vec<u32> = out.vertex_array.iter().map(|v| v.xref).collect();
        model.morphs = model.morphs.remapped(&sources, &vertices);
        model.mesh = CpuMesh { vertices, indices };
    }

    Ok(())
}
