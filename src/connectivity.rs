//! Grouping a mesh's triangles into connected pieces.
//!
//! Two groupings, from the same machinery but different notions of "joined":
//!
//! * **Shells** — welded by position, so a garment split along a UV seam still
//!   counts as one piece. This is what "separate meshes inside one mesh" means.
//! * **UV islands** — joined by vertex index, which is exactly what a seam
//!   breaks. Useful for filling one chart at a time.
//!
//! Welding matches positions bit-for-bit rather than within a tolerance:
//! exporters duplicate the *same* position when they split a vertex for a UV or
//! normal seam, so exact matching handles the real case without the boundary
//! problems that quantising into a grid introduces.

use std::collections::HashMap;

use crate::mesh::MeshData;

/// Union-find over vertices, used to gather triangles into pieces.
struct DisjointSet {
    parent: Vec<u32>,
}

impl DisjointSet {
    fn new(n: usize) -> Self {
        Self { parent: (0..n as u32).collect() }
    }

    fn find(&mut self, mut i: u32) -> u32 {
        while self.parent[i as usize] != i {
            // Path halving: keeps finds near-constant without recursion.
            let grandparent = self.parent[self.parent[i as usize] as usize];
            self.parent[i as usize] = grandparent;
            i = grandparent;
        }
        i
    }

    fn union(&mut self, a: u32, b: u32) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            self.parent[ra as usize] = rb;
        }
    }
}

/// Which group each triangle belongs to, and how many groups there are.
#[derive(Clone, Debug, Default)]
pub struct Groups {
    /// Group id per triangle, indexed the same way `gl_PrimitiveID` is.
    pub of_triangle: Vec<u32>,
    pub count: usize,
}

impl Groups {
    /// Triangles per group, largest first — a quick read on how a mesh is built.
    pub fn sizes(&self) -> Vec<usize> {
        let mut sizes = vec![0usize; self.count];
        for g in &self.of_triangle {
            if let Some(slot) = sizes.get_mut(*g as usize) {
                *slot += 1;
            }
        }
        sizes.sort_unstable_by(|a, b| b.cmp(a));
        sizes
    }
}

/// Assign every triangle a group id from a per-vertex equivalence.
fn group_by(mesh: &MeshData, vertex_key: &[u32]) -> Groups {
    let tri_count = mesh.indices.len() / 3;
    if tri_count == 0 {
        return Groups::default();
    }
    let mut set = DisjointSet::new(vertex_key.len().max(1));
    for tri in mesh.indices.chunks_exact(3) {
        let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let (Some(&ka), Some(&kb), Some(&kc)) =
            (vertex_key.get(a), vertex_key.get(b), vertex_key.get(c))
        else {
            continue;
        };
        set.union(ka, kb);
        set.union(kb, kc);
    }

    // Renumber roots to a dense 0..count range so ids can index a Vec.
    let mut dense: HashMap<u32, u32> = HashMap::new();
    let mut of_triangle = Vec::with_capacity(tri_count);
    for tri in mesh.indices.chunks_exact(3) {
        let root = vertex_key
            .get(tri[0] as usize)
            .map(|k| set.find(*k))
            .unwrap_or(u32::MAX);
        let next = dense.len() as u32;
        of_triangle.push(*dense.entry(root).or_insert(next));
    }
    Groups { of_triangle, count: dense.len() }
}

/// Disconnected pieces of the mesh, welded by position.
pub fn shells(mesh: &MeshData) -> Groups {
    // Map every distinct position to one representative vertex.
    let mut welded: HashMap<[u32; 3], u32> = HashMap::with_capacity(mesh.vertices.len());
    let key: Vec<u32> = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let bits = [
                v.position[0].to_bits(),
                v.position[1].to_bits(),
                v.position[2].to_bits(),
            ];
            *welded.entry(bits).or_insert(i as u32)
        })
        .collect();
    group_by(mesh, &key)
}

/// UV charts: joined by vertex index, which a seam split breaks apart.
pub fn uv_islands(mesh: &MeshData) -> Groups {
    let key: Vec<u32> = (0..mesh.vertices.len() as u32).collect();
    group_by(mesh, &key)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{uv_sphere, Vertex3D};

    fn vertex(x: f32, y: f32, u: f32) -> Vertex3D {
        Vertex3D { position: [x, y, 0.0], normal: [0.0, 0.0, 1.0], uv: [u, 0.0] }
    }

    /// Triangles that share an edge are one piece; ones that don't are two.
    #[test]
    fn separates_disconnected_triangles() {
        let joined = MeshData {
            vertices: vec![
                vertex(0.0, 0.0, 0.0),
                vertex(1.0, 0.0, 0.0),
                vertex(0.0, 1.0, 0.0),
                vertex(1.0, 1.0, 0.0),
            ],
            indices: vec![0, 1, 2, 1, 3, 2],
        };
        assert_eq!(shells(&joined).count, 1, "a shared edge means one shell");

        let apart = MeshData {
            vertices: vec![
                vertex(0.0, 0.0, 0.0),
                vertex(1.0, 0.0, 0.0),
                vertex(0.0, 1.0, 0.0),
                vertex(9.0, 9.0, 0.0),
                vertex(9.5, 9.0, 0.0),
                vertex(9.0, 9.5, 0.0),
            ],
            indices: vec![0, 1, 2, 3, 4, 5],
        };
        let g = shells(&apart);
        assert_eq!(g.count, 2, "detached triangles are separate shells");
        assert_ne!(g.of_triangle[0], g.of_triangle[1]);
        assert_eq!(g.sizes(), vec![1, 1]);
    }

    /// The case that makes welding necessary: an exporter duplicates a vertex
    /// to split a UV seam. Geometrically it is one piece; in UV space it is two.
    #[test]
    fn a_uv_seam_splits_islands_but_not_shells() {
        let mesh = MeshData {
            vertices: vec![
                vertex(0.0, 0.0, 0.0),
                vertex(1.0, 0.0, 0.0),
                vertex(0.0, 1.0, 0.0),
                // Same positions as 1 and 2, different UVs — a seam.
                vertex(1.0, 0.0, 0.9),
                vertex(0.0, 1.0, 0.9),
                vertex(1.0, 1.0, 0.9),
            ],
            indices: vec![0, 1, 2, 3, 5, 4],
        };
        assert_eq!(shells(&mesh).count, 1, "welded positions keep it one shell");
        assert_eq!(uv_islands(&mesh).count, 2, "the seam makes two UV islands");
    }

    /// A real generated mesh must come out as one piece.
    #[test]
    fn a_sphere_is_a_single_shell() {
        let sphere = uv_sphere(1.0, 16, 32);
        let g = shells(&sphere);
        assert_eq!(g.count, 1, "a sphere is one connected surface");
        assert_eq!(g.of_triangle.len(), sphere.indices.len() / 3);
        assert!(g.of_triangle.iter().all(|id| *id == 0));
    }

    /// Degenerate input must not panic.
    #[test]
    fn handles_empty_and_malformed_meshes() {
        let empty = MeshData { vertices: Vec::new(), indices: Vec::new() };
        assert_eq!(shells(&empty).count, 0);
        assert_eq!(uv_islands(&empty).count, 0);

        // Indices pointing past the vertex list.
        let bogus = MeshData { vertices: vec![vertex(0.0, 0.0, 0.0)], indices: vec![0, 5, 9] };
        let g = shells(&bogus);
        assert_eq!(g.of_triangle.len(), 1);
    }
}
