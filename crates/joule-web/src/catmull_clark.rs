//! Catmull-Clark Subdivision — quad-dominant subdivision with face points, edge points,
//! vertex points, boundary rules, crease weights, and iterative refinement.

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
}

// ── CCMesh (indexed polygon mesh) ──────────────────────────────

/// A polygon mesh for Catmull-Clark subdivision.
#[derive(Debug, Clone, PartialEq)]
pub struct CCMesh {
    pub positions: Vec<Vec3>,
    /// Each face is a list of vertex indices (CCW winding).
    pub faces: Vec<Vec<u32>>,
    /// Crease weights per edge. Key: (min_vertex, max_vertex).
    pub creases: HashMap<(u32, u32), f64>,
}

impl CCMesh {
    pub fn new(positions: Vec<Vec3>, faces: Vec<Vec<u32>>) -> Self {
        Self {
            positions,
            faces,
            creases: HashMap::new(),
        }
    }

    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    pub fn face_count(&self) -> usize {
        self.faces.len()
    }

    /// Set crease weight on an edge.
    pub fn set_crease(&mut self, v0: u32, v1: u32, weight: f64) {
        let key = if v0 < v1 { (v0, v1) } else { (v1, v0) };
        self.creases.insert(key, weight.clamp(0.0, 1.0));
    }

    /// Get crease weight for an edge.
    pub fn crease_weight(&self, v0: u32, v1: u32) -> f64 {
        let key = if v0 < v1 { (v0, v1) } else { (v1, v0) };
        self.creases.get(&key).copied().unwrap_or(0.0)
    }

    /// Compute the centroid of a face.
    pub fn face_centroid(&self, face_idx: usize) -> Vec3 {
        let face = &self.faces[face_idx];
        let n = face.len() as f64;
        let mut sum = Vec3::zero();
        for &vi in face {
            sum = sum.add(&self.positions[vi as usize]);
        }
        sum.scale(1.0 / n)
    }

    /// Get edges as sorted vertex pairs for a face.
    pub fn face_edges(&self, face_idx: usize) -> Vec<(u32, u32)> {
        let face = &self.faces[face_idx];
        let n = face.len();
        let mut edges = Vec::with_capacity(n);
        for i in 0..n {
            let a = face[i];
            let b = face[(i + 1) % n];
            edges.push(if a < b { (a, b) } else { (b, a) });
        }
        edges
    }

    /// Build adjacency: edge → adjacent face indices.
    fn edge_faces(&self) -> HashMap<(u32, u32), Vec<usize>> {
        let mut map: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
        for (fi, face) in self.faces.iter().enumerate() {
            let n = face.len();
            for i in 0..n {
                let a = face[i];
                let b = face[(i + 1) % n];
                let key = if a < b { (a, b) } else { (b, a) };
                map.entry(key).or_default().push(fi);
            }
        }
        map
    }

    /// Build adjacency: vertex → adjacent face indices.
    fn vertex_faces(&self) -> HashMap<u32, Vec<usize>> {
        let mut map: HashMap<u32, Vec<usize>> = HashMap::new();
        for (fi, face) in self.faces.iter().enumerate() {
            for &vi in face {
                map.entry(vi).or_default().push(fi);
            }
        }
        map
    }

    /// Build adjacency: vertex → adjacent edges.
    fn vertex_edges(&self) -> HashMap<u32, Vec<(u32, u32)>> {
        let mut map: HashMap<u32, Vec<(u32, u32)>> = HashMap::new();
        for face in &self.faces {
            let n = face.len();
            for i in 0..n {
                let a = face[i];
                let b = face[(i + 1) % n];
                let key = if a < b { (a, b) } else { (b, a) };
                map.entry(a).or_default().push(key);
                map.entry(b).or_default().push(key);
            }
        }
        // Deduplicate
        for edges in map.values_mut() {
            edges.sort();
            edges.dedup();
        }
        map
    }

    /// Is an edge on the boundary (only one adjacent face)?
    fn is_boundary_edge(&self, edge: (u32, u32), edge_faces: &HashMap<(u32, u32), Vec<usize>>) -> bool {
        edge_faces.get(&edge).map_or(true, |f| f.len() < 2)
    }

    /// Convert to triangles (fan triangulation of each face).
    pub fn to_triangles(&self) -> Vec<[u32; 3]> {
        let mut tris = Vec::new();
        for face in &self.faces {
            for i in 1..face.len() - 1 {
                tris.push([face[0], face[i] as u32, face[i + 1] as u32]);
            }
        }
        tris
    }

    /// Compute smooth vertex normals.
    pub fn compute_normals(&self) -> Vec<Vec3> {
        let mut normals = vec![Vec3::zero(); self.positions.len()];
        for face in &self.faces {
            if face.len() < 3 {
                continue;
            }
            let a = self.positions[face[0] as usize];
            let b = self.positions[face[1] as usize];
            let c = self.positions[face[2] as usize];
            let fn_vec = b.sub(&a).cross(&c.sub(&a));
            for &vi in face {
                normals[vi as usize] = normals[vi as usize].add(&fn_vec);
            }
        }
        normals.into_iter().map(|n| n.normalized()).collect()
    }
}

/// Perform one level of Catmull-Clark subdivision.
pub fn subdivide_once(mesh: &CCMesh) -> CCMesh {
    let ef = mesh.edge_faces();
    let vf = mesh.vertex_faces();
    let ve = mesh.vertex_edges();

    let old_vert_count = mesh.positions.len();
    let mut new_positions: Vec<Vec3> = Vec::new();

    // ── 1. Face points ─────────────────────────────────────────
    let face_point_base = old_vert_count;
    let mut face_points = Vec::with_capacity(mesh.faces.len());
    for fi in 0..mesh.faces.len() {
        let fp = mesh.face_centroid(fi);
        face_points.push(fp);
    }

    // ── 2. Edge points ─────────────────────────────────────────
    let mut edge_point_map: HashMap<(u32, u32), usize> = HashMap::new();
    let mut edge_point_positions: Vec<Vec3> = Vec::new();

    let edge_keys: Vec<(u32, u32)> = ef.keys().copied().collect();
    for edge in &edge_keys {
        let pa = mesh.positions[edge.0 as usize];
        let pb = mesh.positions[edge.1 as usize];
        let adj_faces = &ef[edge];
        let crease = mesh.crease_weight(edge.0, edge.1);

        let ep = if adj_faces.len() < 2 {
            // Boundary edge: midpoint
            pa.add(&pb).scale(0.5)
        } else {
            let smooth = {
                let fp_avg = adj_faces
                    .iter()
                    .fold(Vec3::zero(), |acc, &fi| acc.add(&face_points[fi]))
                    .scale(1.0 / adj_faces.len() as f64);
                pa.add(&pb).add(&fp_avg).add(&fp_avg).scale(0.25)
            };
            let sharp = pa.add(&pb).scale(0.5);
            // Blend based on crease weight
            sharp.scale(crease).add(&smooth.scale(1.0 - crease))
        };

        let idx = face_point_base + mesh.faces.len() + edge_point_positions.len();
        edge_point_map.insert(*edge, idx);
        edge_point_positions.push(ep);
    }

    // ── 3. Vertex points ───────────────────────────────────────
    let mut vertex_points = Vec::with_capacity(old_vert_count);
    for vi in 0..old_vert_count {
        let vi32 = vi as u32;
        let adj_face_ids = vf.get(&vi32);
        let adj_edges = ve.get(&vi32);

        let n_faces = adj_face_ids.map_or(0, |f| f.len());
        let n_edges = adj_edges.map_or(0, |e| e.len());

        if n_faces == 0 || n_edges == 0 {
            vertex_points.push(mesh.positions[vi]);
            continue;
        }

        // Check if boundary vertex
        let is_boundary = adj_edges
            .map(|edges| edges.iter().any(|e| mesh.is_boundary_edge(*e, &ef)))
            .unwrap_or(false);

        // Compute max crease weight on incident edges
        let max_crease: f64 = adj_edges
            .map(|edges| {
                edges
                    .iter()
                    .map(|e| mesh.crease_weight(e.0, e.1))
                    .fold(0.0_f64, f64::max)
            })
            .unwrap_or(0.0);

        let orig = mesh.positions[vi];

        if is_boundary {
            // Boundary vertex: average of boundary edge midpoints
            let boundary_edges: Vec<_> = adj_edges
                .unwrap()
                .iter()
                .filter(|e| mesh.is_boundary_edge(**e, &ef))
                .collect();
            if boundary_edges.len() >= 2 {
                let mut mid_sum = Vec3::zero();
                for e in &boundary_edges {
                    let mid = mesh.positions[e.0 as usize]
                        .add(&mesh.positions[e.1 as usize])
                        .scale(0.5);
                    mid_sum = mid_sum.add(&mid);
                }
                let mid_avg = mid_sum.scale(1.0 / boundary_edges.len() as f64);
                vertex_points.push(orig.scale(0.5).add(&mid_avg.scale(0.5)));
            } else {
                vertex_points.push(orig);
            }
        } else {
            // Interior vertex: Q/n + 2R/n + S(n-3)/n
            let n = n_faces as f64;
            let face_avg = adj_face_ids
                .unwrap()
                .iter()
                .fold(Vec3::zero(), |acc, &fi| acc.add(&face_points[fi]))
                .scale(1.0 / n);
            let edge_avg = adj_edges
                .unwrap()
                .iter()
                .fold(Vec3::zero(), |acc, e| {
                    let mid = mesh.positions[e.0 as usize]
                        .add(&mesh.positions[e.1 as usize])
                        .scale(0.5);
                    acc.add(&mid)
                })
                .scale(1.0 / n_edges as f64);

            let smooth = face_avg
                .scale(1.0 / n)
                .add(&edge_avg.scale(2.0 / n))
                .add(&orig.scale((n - 3.0) / n));

            // Blend with original based on crease
            let vp = orig.scale(max_crease).add(&smooth.scale(1.0 - max_crease));
            vertex_points.push(vp);
        }
    }

    // ── 4. Build new positions array ───────────────────────────
    // [0..old_vert_count) = vertex points
    // [old_vert_count..old_vert_count+face_count) = face points
    // [old_vert_count+face_count..) = edge points
    new_positions.extend_from_slice(&vertex_points);
    new_positions.extend_from_slice(&face_points);
    new_positions.extend_from_slice(&edge_point_positions);

    // ── 5. Build new faces ─────────────────────────────────────
    let mut new_faces: Vec<Vec<u32>> = Vec::new();

    for (fi, face) in mesh.faces.iter().enumerate() {
        let fp_idx = (face_point_base + fi) as u32;
        let n = face.len();

        for i in 0..n {
            let v_cur = face[i];
            let v_next = face[(i + 1) % n];
            let v_prev = face[if i == 0 { n - 1 } else { i - 1 }];

            let edge_next = if v_cur < v_next {
                (v_cur, v_next)
            } else {
                (v_next, v_cur)
            };
            let edge_prev = if v_prev < v_cur {
                (v_prev, v_cur)
            } else {
                (v_cur, v_prev)
            };

            let ep_next = edge_point_map[&edge_next] as u32;
            let ep_prev = edge_point_map[&edge_prev] as u32;

            // New quad: vertex_point → edge_point(prev) → face_point → edge_point(next)
            // Wait, standard CC: for vertex i in face, quad is:
            // [vertex_point[i], edge_point(i, i+1), face_point, edge_point(i-1, i)]
            new_faces.push(vec![v_cur, ep_next, fp_idx, ep_prev]);
        }
    }

    // Propagate reduced crease weights
    let mut new_creases = HashMap::new();
    for (&edge, &weight) in &mesh.creases {
        if weight > 0.0 {
            let reduced = (weight - 1.0 / mesh.faces.len() as f64).max(0.0);
            if reduced > 0.0 {
                // Crease propagates to sub-edges through edge point
                if let Some(&ep_idx) = edge_point_map.get(&edge) {
                    let ep32 = ep_idx as u32;
                    let k0 = if edge.0 < ep32 { (edge.0, ep32) } else { (ep32, edge.0) };
                    let k1 = if edge.1 < ep32 { (edge.1, ep32) } else { (ep32, edge.1) };
                    new_creases.insert(k0, reduced);
                    new_creases.insert(k1, reduced);
                }
            }
        }
    }

    CCMesh {
        positions: new_positions,
        faces: new_faces,
        creases: new_creases,
    }
}

/// Perform multiple levels of Catmull-Clark subdivision.
pub fn subdivide(mesh: &CCMesh, levels: u32) -> CCMesh {
    let levels = levels.min(4);
    let mut current = mesh.clone();
    for _ in 0..levels {
        current = subdivide_once(&current);
    }
    current
}

/// Build a cube CCMesh for testing.
pub fn cube() -> CCMesh {
    let positions = vec![
        Vec3::new(-1.0, -1.0, -1.0),
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(1.0, 1.0, -1.0),
        Vec3::new(-1.0, 1.0, -1.0),
        Vec3::new(-1.0, -1.0, 1.0),
        Vec3::new(1.0, -1.0, 1.0),
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::new(-1.0, 1.0, 1.0),
    ];
    let faces = vec![
        vec![0, 3, 2, 1],
        vec![4, 5, 6, 7],
        vec![0, 1, 5, 4],
        vec![2, 3, 7, 6],
        vec![0, 4, 7, 3],
        vec![1, 2, 6, 5],
    ];
    CCMesh::new(positions, faces)
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-4;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_cube_creation() {
        let c = cube();
        assert_eq!(c.vertex_count(), 8);
        assert_eq!(c.face_count(), 6);
    }

    #[test]
    fn test_face_centroid() {
        let c = cube();
        let centroid = c.face_centroid(0);
        // Face 0: [0,3,2,1] → avg of those 4 corners
        assert!(centroid.z < 0.0, "front face centroid should have negative z");
    }

    #[test]
    fn test_face_edges() {
        let c = cube();
        let edges = c.face_edges(0);
        assert_eq!(edges.len(), 4);
    }

    #[test]
    fn test_subdivide_once_face_count() {
        let c = cube();
        let sub = subdivide_once(&c);
        // Each quad → 4 quads: 6 × 4 = 24
        assert_eq!(sub.face_count(), 24);
    }

    #[test]
    fn test_subdivide_once_vertex_count() {
        let c = cube();
        let sub = subdivide_once(&c);
        // 8 vertex points + 6 face points + 12 edge points = 26
        assert_eq!(sub.vertex_count(), 26);
    }

    #[test]
    fn test_subdivide_once_all_quads() {
        let c = cube();
        let sub = subdivide_once(&c);
        for face in &sub.faces {
            assert_eq!(face.len(), 4, "CC subdivision should produce quads");
        }
    }

    #[test]
    fn test_subdivide_two_levels() {
        let c = cube();
        let sub = subdivide(&c, 2);
        // Level 1: 24 faces. Level 2: 24 × 4 = 96 faces
        assert_eq!(sub.face_count(), 96);
    }

    #[test]
    fn test_subdivide_zero_levels() {
        let c = cube();
        let sub = subdivide(&c, 0);
        assert_eq!(sub.face_count(), c.face_count());
    }

    #[test]
    fn test_vertices_converge_to_sphere() {
        // After enough CC subdivision, a cube should approximate a sphere
        let c = cube();
        let sub = subdivide(&c, 3);
        let distances: Vec<f64> = sub.positions.iter().map(|p| p.length()).collect();
        let avg_dist = distances.iter().sum::<f64>() / distances.len() as f64;
        // All vertices should be close to the average distance
        for d in &distances {
            assert!(
                (*d - avg_dist).abs() < 0.5,
                "vertex distance {d} too far from avg {avg_dist}"
            );
        }
    }

    #[test]
    fn test_crease_weight() {
        let mut c = cube();
        c.set_crease(0, 1, 0.9);
        assert!(approx(c.crease_weight(0, 1), 0.9, EPS));
        assert!(approx(c.crease_weight(1, 0), 0.9, EPS)); // symmetric
    }

    #[test]
    fn test_crease_preserves_sharpness() {
        let mut c = cube();
        // Crease top edges
        c.set_crease(2, 3, 1.0);
        c.set_crease(3, 7, 1.0);
        c.set_crease(6, 7, 1.0);
        c.set_crease(2, 6, 1.0);
        let sub = subdivide(&c, 1);
        // Creased vertices should be closer to originals
        let v2_orig = c.positions[2];
        let v2_sub = sub.positions[2];
        let dist = v2_sub.sub(&v2_orig).length();
        // With full crease, vertex shouldn't move much
        assert!(dist < 1.5, "creased vertex moved too far: {dist}");
    }

    #[test]
    fn test_to_triangles() {
        let c = cube();
        let tris = c.to_triangles();
        // 6 quads → 12 triangles
        assert_eq!(tris.len(), 12);
    }

    #[test]
    fn test_compute_normals() {
        let c = cube();
        let normals = c.compute_normals();
        assert_eq!(normals.len(), 8);
        for n in &normals {
            assert!(n.length() > 0.5);
        }
    }

    #[test]
    fn test_triangle_mesh_subdivision() {
        // Tetrahedron (all triangles)
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
        let mesh = CCMesh::new(positions, faces);
        let sub = subdivide_once(&mesh);
        // Each triangle → 3 quads: 4 × 3 = 12
        assert_eq!(sub.face_count(), 12);
    }

    #[test]
    fn test_subdivide_level_clamped() {
        let c = cube();
        // Level 5 → clamped to 4
        let sub = subdivide(&c, 5);
        // Level 4: 6 × 4^4 = 6 × 256 = 1536
        assert_eq!(sub.face_count(), 1536);
    }

    #[test]
    fn test_face_indices_in_bounds() {
        let c = cube();
        let sub = subdivide_once(&c);
        let max_v = sub.vertex_count() as u32;
        for face in &sub.faces {
            for &vi in face {
                assert!(vi < max_v, "face vertex index {vi} >= max {max_v}");
            }
        }
    }

    #[test]
    fn test_crease_clamp() {
        let mut c = cube();
        c.set_crease(0, 1, 2.5);
        assert!(approx(c.crease_weight(0, 1), 1.0, EPS));
        c.set_crease(0, 1, -0.5);
        assert!(approx(c.crease_weight(0, 1), 0.0, EPS));
    }

    #[test]
    fn test_single_face() {
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let faces = vec![vec![0, 1, 2, 3]];
        let mesh = CCMesh::new(positions, faces);
        let sub = subdivide_once(&mesh);
        assert_eq!(sub.face_count(), 4);
    }

    #[test]
    fn test_new_vertices_finite() {
        let c = cube();
        let sub = subdivide(&c, 2);
        for p in &sub.positions {
            assert!(p.x.is_finite());
            assert!(p.y.is_finite());
            assert!(p.z.is_finite());
        }
    }

    #[test]
    fn test_normals_after_subdivision() {
        let c = cube();
        let sub = subdivide(&c, 1);
        let normals = sub.compute_normals();
        assert_eq!(normals.len(), sub.vertex_count());
    }
}
