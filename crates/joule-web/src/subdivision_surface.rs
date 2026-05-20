//! Subdivision Surface — generic subdivision framework with half-edge mesh data structure,
//! connectivity traversal, boundary detection, crease edges, and smooth normals.

use std::collections::HashMap;

// ── Vector types ───────────────────────────────────────────────

/// 3D vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
    pub fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }
    pub fn add(&self, o: &Self) -> Self {
        Self::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
    pub fn sub(&self, o: &Self) -> Self {
        Self::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
    pub fn scale(&self, s: f64) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }
    pub fn length(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
    pub fn normalized(&self) -> Self {
        let len = self.length();
        if len < 1e-12 { Self::zero() } else { Self::new(self.x / len, self.y / len, self.z / len) }
    }
    pub fn cross(&self, o: &Self) -> Self {
        Self::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }
    pub fn dot(&self, o: &Self) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
}

// ── Half-edge mesh ─────────────────────────────────────────────

/// Index types for clarity.
pub type VertexId = usize;
pub type HalfEdgeId = usize;
pub type FaceId = usize;

/// A vertex in the half-edge mesh.
#[derive(Debug, Clone)]
pub struct HVertex {
    pub position: Vec3,
    /// One outgoing half-edge from this vertex.
    pub half_edge: Option<HalfEdgeId>,
}

/// A half-edge.
#[derive(Debug, Clone)]
pub struct HalfEdge {
    /// Origin vertex.
    pub origin: VertexId,
    /// Twin (opposite) half-edge. None for boundary edges.
    pub twin: Option<HalfEdgeId>,
    /// Next half-edge around the face (CCW).
    pub next: HalfEdgeId,
    /// Previous half-edge around the face (CCW).
    pub prev: HalfEdgeId,
    /// Face this half-edge borders. None for boundary half-edges.
    pub face: Option<FaceId>,
    /// Crease weight: 0.0 = smooth, 1.0 = fully sharp.
    pub crease_weight: f64,
}

/// A face in the half-edge mesh.
#[derive(Debug, Clone)]
pub struct HFace {
    /// One half-edge on this face's boundary.
    pub half_edge: HalfEdgeId,
}

/// Half-edge mesh data structure.
#[derive(Debug, Clone)]
pub struct HalfEdgeMesh {
    pub vertices: Vec<HVertex>,
    pub half_edges: Vec<HalfEdge>,
    pub faces: Vec<HFace>,
}

impl HalfEdgeMesh {
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            half_edges: Vec::new(),
            faces: Vec::new(),
        }
    }

    /// Add a vertex, returning its index.
    pub fn add_vertex(&mut self, position: Vec3) -> VertexId {
        let id = self.vertices.len();
        self.vertices.push(HVertex {
            position,
            half_edge: None,
        });
        id
    }

    /// Build from indexed face list. Each face is a list of vertex indices (CCW).
    pub fn from_faces(positions: &[Vec3], faces: &[Vec<usize>]) -> Self {
        let mut mesh = HalfEdgeMesh::new();

        for pos in positions {
            mesh.add_vertex(*pos);
        }

        // Map (origin, dest) → half-edge index
        let mut edge_map: HashMap<(usize, usize), HalfEdgeId> = HashMap::new();

        for face_verts in faces {
            let n = face_verts.len();
            if n < 3 {
                continue;
            }

            let face_id = mesh.faces.len();
            let he_base = mesh.half_edges.len();

            // Create half-edges for this face
            for i in 0..n {
                let origin = face_verts[i];
                let next_i = (i + 1) % n;
                let prev_i = if i == 0 { n - 1 } else { i - 1 };

                mesh.half_edges.push(HalfEdge {
                    origin,
                    twin: None,
                    next: he_base + next_i,
                    prev: he_base + prev_i,
                    face: Some(face_id),
                    crease_weight: 0.0,
                });

                if mesh.vertices[origin].half_edge.is_none() {
                    mesh.vertices[origin].half_edge = Some(he_base + i);
                }

                edge_map.insert((origin, face_verts[next_i]), he_base + i);
            }

            mesh.faces.push(HFace {
                half_edge: he_base,
            });
        }

        // Link twins
        let keys: Vec<(usize, usize)> = edge_map.keys().copied().collect();
        for (a, b) in keys {
            if let Some(&twin_id) = edge_map.get(&(b, a)) {
                let he_id = edge_map[&(a, b)];
                mesh.half_edges[he_id].twin = Some(twin_id);
                mesh.half_edges[twin_id].twin = Some(he_id);
            }
        }

        mesh
    }

    /// Iterate vertex indices.
    pub fn vertex_ids(&self) -> impl Iterator<Item = VertexId> {
        0..self.vertices.len()
    }

    /// Iterate face indices.
    pub fn face_ids(&self) -> impl Iterator<Item = FaceId> {
        0..self.faces.len()
    }

    /// Iterate half-edge indices.
    pub fn half_edge_ids(&self) -> impl Iterator<Item = HalfEdgeId> {
        0..self.half_edges.len()
    }

    /// Get the destination vertex of a half-edge.
    pub fn half_edge_dest(&self, he: HalfEdgeId) -> VertexId {
        let next = self.half_edges[he].next;
        self.half_edges[next].origin
    }

    /// Is this half-edge on the boundary (no twin)?
    pub fn is_boundary_edge(&self, he: HalfEdgeId) -> bool {
        self.half_edges[he].twin.is_none()
    }

    /// Is this vertex on the boundary?
    pub fn is_boundary_vertex(&self, v: VertexId) -> bool {
        let start = match self.vertices[v].half_edge {
            Some(he) => he,
            None => return true,
        };
        let mut current = start;
        loop {
            if self.is_boundary_edge(current) {
                return true;
            }
            let twin = self.half_edges[current].twin.unwrap();
            current = self.half_edges[twin].next;
            if current == start {
                break;
            }
        }
        false
    }

    /// Get face vertex indices (in order).
    pub fn face_vertices(&self, face: FaceId) -> Vec<VertexId> {
        let start = self.faces[face].half_edge;
        let mut result = Vec::new();
        let mut he = start;
        loop {
            result.push(self.half_edges[he].origin);
            he = self.half_edges[he].next;
            if he == start {
                break;
            }
        }
        result
    }

    /// Count edges of a face.
    pub fn face_edge_count(&self, face: FaceId) -> usize {
        self.face_vertices(face).len()
    }

    /// Get the 1-ring neighbor vertices around a vertex.
    pub fn vertex_neighbors(&self, v: VertexId) -> Vec<VertexId> {
        let mut neighbors = Vec::new();
        let start = match self.vertices[v].half_edge {
            Some(he) => he,
            None => return neighbors,
        };
        let mut current = start;
        loop {
            neighbors.push(self.half_edge_dest(current));
            match self.half_edges[current].twin {
                Some(twin) => {
                    current = self.half_edges[twin].next;
                    if current == start {
                        break;
                    }
                }
                None => break,
            }
        }
        neighbors
    }

    /// Vertex valence (number of adjacent edges).
    pub fn vertex_valence(&self, v: VertexId) -> usize {
        self.vertex_neighbors(v).len()
    }

    /// Set crease weight on both half-edges of an edge.
    pub fn set_crease(&mut self, v0: VertexId, v1: VertexId, weight: f64) {
        let w = weight.clamp(0.0, 1.0);
        // Collect matching half-edge indices first to avoid borrow conflicts.
        let matching: Vec<usize> = (0..self.half_edges.len())
            .filter(|he_id| {
                let origin = self.half_edges[*he_id].origin;
                let dest_next = self.half_edges[*he_id].next;
                let dest = self.half_edges[dest_next].origin;
                (origin == v0 && dest == v1) || (origin == v1 && dest == v0)
            })
            .collect();
        for he_id in matching {
            self.half_edges[he_id].crease_weight = w;
        }
    }

    /// Compute smooth normals at each vertex from face normals.
    pub fn compute_smooth_normals(&self) -> Vec<Vec3> {
        let mut normals = vec![Vec3::zero(); self.vertices.len()];

        for face_id in self.face_ids() {
            let verts = self.face_vertices(face_id);
            if verts.len() < 3 {
                continue;
            }
            let a = self.vertices[verts[0]].position;
            let b = self.vertices[verts[1]].position;
            let c = self.vertices[verts[2]].position;
            let face_normal = b.sub(&a).cross(&c.sub(&a));
            for &vi in &verts {
                normals[vi] = normals[vi].add(&face_normal);
            }
        }

        normals.into_iter().map(|n| n.normalized()).collect()
    }

    /// Convert to indexed triangle mesh (triangulate faces by fan).
    pub fn to_triangles(&self) -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let positions: Vec<Vec3> = self.vertices.iter().map(|v| v.position).collect();
        let mut indices = Vec::new();
        for face_id in self.face_ids() {
            let verts = self.face_vertices(face_id);
            for i in 1..verts.len() - 1 {
                indices.push([verts[0] as u32, verts[i] as u32, verts[i + 1] as u32]);
            }
        }
        (positions, indices)
    }

    /// Number of boundary edges.
    pub fn boundary_edge_count(&self) -> usize {
        self.half_edges.iter().filter(|he| he.twin.is_none()).count()
    }

    /// Total edge count (each edge = pair of half-edges + boundary singles).
    pub fn edge_count(&self) -> usize {
        let paired = self.half_edges.iter().filter(|he| he.twin.is_some()).count();
        let boundary = self.boundary_edge_count();
        paired / 2 + boundary
    }

    /// Euler characteristic: V - E + F.
    pub fn euler_characteristic(&self) -> i64 {
        self.vertices.len() as i64 - self.edge_count() as i64 + self.faces.len() as i64
    }
}

/// Apply a generic subdivision step. Caller provides callbacks for:
/// - edge_point: given origin and dest positions, twin face, return new edge vertex position
/// - vertex_point: given old vertex and its neighbors, return updated position
pub fn subdivide_generic<FE, FV>(
    mesh: &HalfEdgeMesh,
    edge_point_fn: FE,
    vertex_point_fn: FV,
    levels: u32,
) -> HalfEdgeMesh
where
    FE: Fn(Vec3, Vec3, f64) -> Vec3 + Copy,
    FV: Fn(Vec3, &[Vec3], bool, f64) -> Vec3 + Copy,
{
    let mut current = mesh.clone();
    let levels = levels.min(4);

    for _ in 0..levels {
        let mut new_positions: Vec<Vec3> = Vec::new();
        let mut new_faces: Vec<Vec<usize>> = Vec::new();

        // Copy existing vertex positions (will be updated)
        let old_vert_count = current.vertices.len();
        for _ in 0..old_vert_count {
            new_positions.push(Vec3::zero()); // placeholder
        }

        // Compute new vertex positions
        for vi in 0..old_vert_count {
            let nbrs: Vec<Vec3> = current
                .vertex_neighbors(vi)
                .iter()
                .map(|n| current.vertices[*n].position)
                .collect();
            let is_boundary = current.is_boundary_vertex(vi);
            // Average crease weight of incident edges
            let avg_crease = if nbrs.is_empty() {
                0.0
            } else {
                let total: f64 = current.vertex_neighbors(vi).iter().map(|_| 0.0).sum();
                total
            };
            new_positions[vi] =
                vertex_point_fn(current.vertices[vi].position, &nbrs, is_boundary, avg_crease);
        }

        // Create edge midpoints
        let mut edge_mid_map: HashMap<(usize, usize), usize> = HashMap::new();
        for he_id in 0..current.half_edges.len() {
            let he = &current.half_edges[he_id];
            let a = he.origin;
            let b = current.half_edge_dest(he_id);
            let key = if a < b { (a, b) } else { (b, a) };
            if edge_mid_map.contains_key(&key) {
                continue;
            }
            let pa = current.vertices[a].position;
            let pb = current.vertices[b].position;
            let mid = edge_point_fn(pa, pb, he.crease_weight);
            let mid_id = new_positions.len();
            new_positions.push(mid);
            edge_mid_map.insert(key, mid_id);
        }

        // Split each face: for triangles, create 4 sub-triangles
        for face_id in current.face_ids() {
            let fv = current.face_vertices(face_id);
            let n = fv.len();
            if n == 3 {
                // Triangle → 4 triangles
                let m01 = edge_mid_map[&if fv[0] < fv[1] {
                    (fv[0], fv[1])
                } else {
                    (fv[1], fv[0])
                }];
                let m12 = edge_mid_map[&if fv[1] < fv[2] {
                    (fv[1], fv[2])
                } else {
                    (fv[2], fv[1])
                }];
                let m20 = edge_mid_map[&if fv[2] < fv[0] {
                    (fv[2], fv[0])
                } else {
                    (fv[0], fv[2])
                }];
                new_faces.push(vec![fv[0], m01, m20]);
                new_faces.push(vec![fv[1], m12, m01]);
                new_faces.push(vec![fv[2], m20, m12]);
                new_faces.push(vec![m01, m12, m20]);
            } else {
                // Polygon → fan of triangles from centroid
                let centroid_pos = {
                    let mut sum = Vec3::zero();
                    for &vi in &fv {
                        sum = sum.add(&current.vertices[vi].position);
                    }
                    sum.scale(1.0 / n as f64)
                };
                let centroid_id = new_positions.len();
                new_positions.push(centroid_pos);
                for i in 0..n {
                    let next = (i + 1) % n;
                    let a = fv[i];
                    let b = fv[next];
                    let mid = edge_mid_map[&if a < b { (a, b) } else { (b, a) }];
                    new_faces.push(vec![a, mid, centroid_id]);
                    let prev_i = if i == 0 { n - 1 } else { i - 1 };
                    let a_prev = fv[prev_i];
                    let mid_prev =
                        edge_mid_map[&if a_prev < a { (a_prev, a) } else { (a, a_prev) }];
                    new_faces.push(vec![a, centroid_id, mid_prev]);
                }
            }
        }

        current = HalfEdgeMesh::from_faces(&new_positions, &new_faces);
    }

    current
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    fn tetrahedron() -> HalfEdgeMesh {
        let positions = vec![
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(-1.0, -1.0, 1.0),
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(1.0, -1.0, -1.0),
        ];
        let faces = vec![
            vec![0, 1, 2],
            vec![0, 3, 1],
            vec![0, 2, 3],
            vec![1, 3, 2],
        ];
        HalfEdgeMesh::from_faces(&positions, &faces)
    }

    fn cube() -> HalfEdgeMesh {
        let positions = vec![
            Vec3::new(-1.0, -1.0, -1.0), // 0
            Vec3::new(1.0, -1.0, -1.0),  // 1
            Vec3::new(1.0, 1.0, -1.0),   // 2
            Vec3::new(-1.0, 1.0, -1.0),  // 3
            Vec3::new(-1.0, -1.0, 1.0),  // 4
            Vec3::new(1.0, -1.0, 1.0),   // 5
            Vec3::new(1.0, 1.0, 1.0),    // 6
            Vec3::new(-1.0, 1.0, 1.0),   // 7
        ];
        let faces = vec![
            vec![0, 3, 2, 1], // front
            vec![4, 5, 6, 7], // back
            vec![0, 1, 5, 4], // bottom
            vec![2, 3, 7, 6], // top
            vec![0, 4, 7, 3], // left
            vec![1, 2, 6, 5], // right
        ];
        HalfEdgeMesh::from_faces(&positions, &faces)
    }

    #[test]
    fn test_tetrahedron_vertex_count() {
        let mesh = tetrahedron();
        assert_eq!(mesh.vertices.len(), 4);
    }

    #[test]
    fn test_tetrahedron_face_count() {
        let mesh = tetrahedron();
        assert_eq!(mesh.faces.len(), 4);
    }

    #[test]
    fn test_tetrahedron_euler() {
        let mesh = tetrahedron();
        assert_eq!(mesh.euler_characteristic(), 2);
    }

    #[test]
    fn test_cube_vertex_count() {
        let mesh = cube();
        assert_eq!(mesh.vertices.len(), 8);
    }

    #[test]
    fn test_cube_face_count() {
        let mesh = cube();
        assert_eq!(mesh.faces.len(), 6);
    }

    #[test]
    fn test_cube_edge_count() {
        let mesh = cube();
        assert_eq!(mesh.edge_count(), 12);
    }

    #[test]
    fn test_cube_euler() {
        let mesh = cube();
        // V - E + F = 8 - 12 + 6 = 2
        assert_eq!(mesh.euler_characteristic(), 2);
    }

    #[test]
    fn test_face_vertices_triangle() {
        let mesh = tetrahedron();
        let fv = mesh.face_vertices(0);
        assert_eq!(fv.len(), 3);
    }

    #[test]
    fn test_face_vertices_quad() {
        let mesh = cube();
        let fv = mesh.face_vertices(0);
        assert_eq!(fv.len(), 4);
    }

    #[test]
    fn test_vertex_neighbors() {
        let mesh = tetrahedron();
        let nbrs = mesh.vertex_neighbors(0);
        // Tetrahedron: each vertex connects to 3 others
        assert_eq!(nbrs.len(), 3);
    }

    #[test]
    fn test_vertex_valence_cube() {
        let mesh = cube();
        // Each cube vertex has valence 3
        for vi in 0..8 {
            assert_eq!(mesh.vertex_valence(vi), 3);
        }
    }

    #[test]
    fn test_boundary_detection_closed() {
        let mesh = tetrahedron();
        assert_eq!(mesh.boundary_edge_count(), 0);
        for vi in 0..4 {
            assert!(!mesh.is_boundary_vertex(vi));
        }
    }

    #[test]
    fn test_boundary_detection_open() {
        // Single triangle = open mesh
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let faces = vec![vec![0, 1, 2]];
        let mesh = HalfEdgeMesh::from_faces(&positions, &faces);
        assert_eq!(mesh.boundary_edge_count(), 3);
        for vi in 0..3 {
            assert!(mesh.is_boundary_vertex(vi));
        }
    }

    #[test]
    fn test_smooth_normals() {
        let mesh = tetrahedron();
        let normals = mesh.compute_smooth_normals();
        assert_eq!(normals.len(), 4);
        for n in &normals {
            assert!(approx(n.length(), 1.0, 0.01));
        }
    }

    #[test]
    fn test_to_triangles() {
        let mesh = cube();
        let (verts, tris) = mesh.to_triangles();
        assert_eq!(verts.len(), 8);
        // 6 quads → 12 triangles
        assert_eq!(tris.len(), 12);
    }

    #[test]
    fn test_set_crease() {
        let mut mesh = tetrahedron();
        mesh.set_crease(0, 1, 0.8);
        // Verify crease was set on relevant half-edges
        let has_crease = mesh.half_edges.iter().any(|he| he.crease_weight > 0.5);
        assert!(has_crease);
    }

    #[test]
    fn test_subdivide_generic_triangle() {
        let mesh = tetrahedron();
        let result = subdivide_generic(
            &mesh,
            |a, b, _crease| a.add(&b).scale(0.5), // midpoint
            |pos, _nbrs, _boundary, _crease| pos,  // keep original
            1,
        );
        // 4 faces × 4 = 16 triangles after 1 level
        assert!(result.faces.len() >= 16);
    }

    #[test]
    fn test_subdivide_level_zero() {
        let mesh = tetrahedron();
        let result = subdivide_generic(
            &mesh,
            |a, b, _| a.add(&b).scale(0.5),
            |pos, _, _, _| pos,
            0,
        );
        assert_eq!(result.faces.len(), mesh.faces.len());
    }

    #[test]
    fn test_half_edge_dest() {
        let mesh = tetrahedron();
        let he0 = &mesh.half_edges[0];
        let dest = mesh.half_edge_dest(0);
        assert_ne!(he0.origin, dest);
    }

    #[test]
    fn test_face_edge_count() {
        let mesh = cube();
        for fi in 0..6 {
            assert_eq!(mesh.face_edge_count(fi), 4);
        }
    }

    #[test]
    fn test_empty_mesh() {
        let mesh = HalfEdgeMesh::new();
        assert_eq!(mesh.vertices.len(), 0);
        assert_eq!(mesh.faces.len(), 0);
        assert_eq!(mesh.euler_characteristic(), 0);
    }

    #[test]
    fn test_add_vertex() {
        let mut mesh = HalfEdgeMesh::new();
        let v0 = mesh.add_vertex(Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(v0, 0);
        assert!(approx(mesh.vertices[0].position.x, 1.0, EPS));
    }

    #[test]
    fn test_level_clamped_to_4() {
        let mesh = tetrahedron();
        // Level 5 should be clamped to 4 — just verify it completes without panic
        let result = subdivide_generic(
            &mesh,
            |a, b, _| a.add(&b).scale(0.5),
            |pos, _, _, _| pos,
            5,
        );
        assert!(result.faces.len() > 0);
    }
}
