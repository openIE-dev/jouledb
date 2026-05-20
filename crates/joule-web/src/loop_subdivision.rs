//! Loop Subdivision — subdivision for triangle meshes. Each triangle becomes 4 triangles.
//! Edge vertices use β-weighted averages, vertex vertices use Warren's weights.
//! Boundary edge rules, crease handling, and extraordinary vertex support.

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

// ── Triangle mesh ──────────────────────────────────────────────

/// A triangle mesh for Loop subdivision.
#[derive(Debug, Clone, PartialEq)]
pub struct TriMesh {
    pub positions: Vec<Vec3>,
    pub triangles: Vec<[u32; 3]>,
    /// Crease weights per edge. Key: (min_vertex, max_vertex).
    pub creases: HashMap<(u32, u32), f64>,
}

impl TriMesh {
    pub fn new(positions: Vec<Vec3>, triangles: Vec<[u32; 3]>) -> Self {
        Self {
            positions,
            triangles,
            creases: HashMap::new(),
        }
    }

    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    pub fn triangle_count(&self) -> usize {
        self.triangles.len()
    }

    /// Set crease weight on an edge.
    pub fn set_crease(&mut self, v0: u32, v1: u32, weight: f64) {
        let key = if v0 < v1 { (v0, v1) } else { (v1, v0) };
        self.creases.insert(key, weight.clamp(0.0, 1.0));
    }

    pub fn crease_weight(&self, v0: u32, v1: u32) -> f64 {
        let key = if v0 < v1 { (v0, v1) } else { (v1, v0) };
        self.creases.get(&key).copied().unwrap_or(0.0)
    }

    /// Compute smooth normals.
    pub fn compute_normals(&self) -> Vec<Vec3> {
        let mut normals = vec![Vec3::zero(); self.positions.len()];
        for tri in &self.triangles {
            let a = self.positions[tri[0] as usize];
            let b = self.positions[tri[1] as usize];
            let c = self.positions[tri[2] as usize];
            let face_n = b.sub(&a).cross(&c.sub(&a));
            for &vi in tri {
                normals[vi as usize] = normals[vi as usize].add(&face_n);
            }
        }
        normals.into_iter().map(|n| n.normalized()).collect()
    }
}

/// Edge key: sorted vertex pair.
fn edge_key(a: u32, b: u32) -> (u32, u32) {
    if a < b { (a, b) } else { (b, a) }
}

/// Warren's beta weight for a vertex of valence n.
pub fn warren_beta(n: usize) -> f64 {
    use std::f64::consts::PI;
    if n == 3 {
        3.0 / 16.0
    } else {
        let n_f = n as f64;
        let inner = 3.0 / 8.0 + (2.0 * PI / n_f).cos() / 4.0;
        (1.0 / n_f) * (5.0 / 8.0 - inner * inner)
    }
}

/// Build adjacency: edge → list of triangles containing that edge.
fn build_edge_triangles(mesh: &TriMesh) -> HashMap<(u32, u32), Vec<usize>> {
    let mut map: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (ti, tri) in mesh.triangles.iter().enumerate() {
        for i in 0..3 {
            let a = tri[i];
            let b = tri[(i + 1) % 3];
            map.entry(edge_key(a, b)).or_default().push(ti);
        }
    }
    map
}

/// Build adjacency: vertex → set of neighbor vertices.
fn build_vertex_neighbors(mesh: &TriMesh) -> HashMap<u32, Vec<u32>> {
    let mut map: HashMap<u32, Vec<u32>> = HashMap::new();
    for tri in &mesh.triangles {
        for i in 0..3 {
            let a = tri[i];
            let b = tri[(i + 1) % 3];
            map.entry(a).or_default().push(b);
            map.entry(b).or_default().push(a);
        }
    }
    for nbrs in map.values_mut() {
        nbrs.sort();
        nbrs.dedup();
    }
    map
}

/// Find the opposite vertex in a triangle across from edge (a, b).
fn opposite_vertex(tri: &[u32; 3], a: u32, b: u32) -> u32 {
    for &v in tri {
        if v != a && v != b {
            return v;
        }
    }
    tri[0] // fallback
}

/// Perform one level of Loop subdivision.
pub fn subdivide_once(mesh: &TriMesh) -> TriMesh {
    let edge_tris = build_edge_triangles(mesh);
    let vert_nbrs = build_vertex_neighbors(mesh);
    let old_vert_count = mesh.positions.len();

    // ── 1. Compute new edge vertices ───────────────────────────
    let mut edge_vert_map: HashMap<(u32, u32), u32> = HashMap::new();
    let mut new_positions = mesh.positions.clone(); // start with copies of old positions

    let edge_keys_list: Vec<(u32, u32)> = edge_tris.keys().copied().collect();
    for ek in &edge_keys_list {
        let adj = &edge_tris[ek];
        let pa = mesh.positions[ek.0 as usize];
        let pb = mesh.positions[ek.1 as usize];
        let crease = mesh.crease_weight(ek.0, ek.1);
        let is_boundary = adj.len() < 2;

        let ep = if is_boundary {
            // Boundary: simple midpoint
            pa.add(&pb).scale(0.5)
        } else {
            // Interior: 3/8 * (a + b) + 1/8 * (c + d)
            let c = mesh.positions[opposite_vertex(&mesh.triangles[adj[0]], ek.0, ek.1) as usize];
            let d = mesh.positions[opposite_vertex(&mesh.triangles[adj[1]], ek.0, ek.1) as usize];
            let smooth = pa
                .add(&pb)
                .scale(3.0 / 8.0)
                .add(&c.add(&d).scale(1.0 / 8.0));
            let sharp = pa.add(&pb).scale(0.5);
            sharp.scale(crease).add(&smooth.scale(1.0 - crease))
        };

        let new_idx = new_positions.len() as u32;
        new_positions.push(ep);
        edge_vert_map.insert(*ek, new_idx);
    }

    // ── 2. Update old vertex positions ─────────────────────────
    for vi in 0..old_vert_count {
        let vi32 = vi as u32;
        let nbrs = match vert_nbrs.get(&vi32) {
            Some(n) => n,
            None => continue,
        };
        let n = nbrs.len();
        if n == 0 {
            continue;
        }

        // Check if boundary vertex
        let is_boundary = nbrs.iter().any(|nb| {
            let ek = edge_key(vi32, *nb);
            edge_tris.get(&ek).map_or(true, |t| t.len() < 2)
        });

        // Max crease weight on incident edges
        let max_crease: f64 = nbrs
            .iter()
            .map(|nb| mesh.crease_weight(vi32, *nb))
            .fold(0.0_f64, f64::max);

        let orig = mesh.positions[vi];

        if is_boundary {
            // Boundary: (1/8) * (a + b) + (3/4) * orig, where a, b are boundary neighbors
            let boundary_nbrs: Vec<u32> = nbrs
                .iter()
                .filter(|&&nb| {
                    let ek = edge_key(vi32, nb);
                    edge_tris.get(&ek).map_or(true, |t| t.len() < 2)
                })
                .copied()
                .collect();
            if boundary_nbrs.len() >= 2 {
                let a = mesh.positions[boundary_nbrs[0] as usize];
                let b = mesh.positions[boundary_nbrs[1] as usize];
                new_positions[vi] = a.add(&b).scale(1.0 / 8.0).add(&orig.scale(3.0 / 4.0));
            }
        } else {
            // Interior: Warren's formula
            let beta = warren_beta(n);
            let nbr_sum = nbrs
                .iter()
                .fold(Vec3::zero(), |acc, &nb| {
                    acc.add(&mesh.positions[nb as usize])
                });
            let smooth = orig
                .scale(1.0 - n as f64 * beta)
                .add(&nbr_sum.scale(beta));
            // Blend with crease
            new_positions[vi] = orig.scale(max_crease).add(&smooth.scale(1.0 - max_crease));
        }
    }

    // ── 3. Build new triangles ─────────────────────────────────
    let mut new_triangles = Vec::with_capacity(mesh.triangles.len() * 4);

    for tri in &mesh.triangles {
        let m01 = edge_vert_map[&edge_key(tri[0], tri[1])];
        let m12 = edge_vert_map[&edge_key(tri[1], tri[2])];
        let m20 = edge_vert_map[&edge_key(tri[2], tri[0])];

        // 4 sub-triangles
        new_triangles.push([tri[0], m01, m20]);
        new_triangles.push([tri[1], m12, m01]);
        new_triangles.push([tri[2], m20, m12]);
        new_triangles.push([m01, m12, m20]); // center triangle
    }

    // Propagate creases with reduced weight
    let mut new_creases = HashMap::new();
    for (&ek, &w) in &mesh.creases {
        if w > 0.0 {
            let reduced = (w - 0.1).max(0.0);
            if reduced > 0.0 {
                if let Some(&mid) = edge_vert_map.get(&ek) {
                    new_creases.insert(edge_key(ek.0, mid), reduced);
                    new_creases.insert(edge_key(ek.1, mid), reduced);
                }
            }
        }
    }

    TriMesh {
        positions: new_positions,
        triangles: new_triangles,
        creases: new_creases,
    }
}

/// Perform multiple levels of Loop subdivision.
pub fn subdivide(mesh: &TriMesh, levels: u32) -> TriMesh {
    let levels = levels.min(4);
    let mut current = mesh.clone();
    for _ in 0..levels {
        current = subdivide_once(&current);
    }
    current
}

/// Build a tetrahedron TriMesh for testing.
pub fn tetrahedron() -> TriMesh {
    let positions = vec![
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::new(-1.0, -1.0, 1.0),
        Vec3::new(-1.0, 1.0, -1.0),
        Vec3::new(1.0, -1.0, -1.0),
    ];
    let triangles = vec![
        [0, 1, 2],
        [0, 3, 1],
        [0, 2, 3],
        [1, 3, 2],
    ];
    TriMesh::new(positions, triangles)
}

/// Build an octahedron TriMesh.
pub fn octahedron() -> TriMesh {
    let positions = vec![
        Vec3::new(0.0, 1.0, 0.0),   // 0: top
        Vec3::new(1.0, 0.0, 0.0),   // 1
        Vec3::new(0.0, 0.0, 1.0),   // 2
        Vec3::new(-1.0, 0.0, 0.0),  // 3
        Vec3::new(0.0, 0.0, -1.0),  // 4
        Vec3::new(0.0, -1.0, 0.0),  // 5: bottom
    ];
    let triangles = vec![
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 4],
        [0, 4, 1],
        [5, 2, 1],
        [5, 3, 2],
        [5, 4, 3],
        [5, 1, 4],
    ];
    TriMesh::new(positions, triangles)
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const EPS: f64 = 1e-4;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_tetrahedron_creation() {
        let m = tetrahedron();
        assert_eq!(m.vertex_count(), 4);
        assert_eq!(m.triangle_count(), 4);
    }

    #[test]
    fn test_octahedron_creation() {
        let m = octahedron();
        assert_eq!(m.vertex_count(), 6);
        assert_eq!(m.triangle_count(), 8);
    }

    #[test]
    fn test_subdivide_once_triangle_count() {
        let m = tetrahedron();
        let sub = subdivide_once(&m);
        // 4 triangles × 4 = 16
        assert_eq!(sub.triangle_count(), 16);
    }

    #[test]
    fn test_subdivide_once_vertex_count() {
        let m = tetrahedron();
        let sub = subdivide_once(&m);
        // 4 original + 6 edge midpoints = 10
        assert_eq!(sub.vertex_count(), 10);
    }

    #[test]
    fn test_subdivide_two_levels() {
        let m = tetrahedron();
        let sub = subdivide(&m, 2);
        // Level 1: 16 tris. Level 2: 64 tris
        assert_eq!(sub.triangle_count(), 64);
    }

    #[test]
    fn test_subdivide_zero_levels() {
        let m = tetrahedron();
        let sub = subdivide(&m, 0);
        assert_eq!(sub.triangle_count(), 4);
    }

    #[test]
    fn test_octahedron_subdivision() {
        let m = octahedron();
        let sub = subdivide_once(&m);
        assert_eq!(sub.triangle_count(), 32);
    }

    #[test]
    fn test_warren_beta_valence_3() {
        let beta = warren_beta(3);
        assert!(approx(beta, 3.0 / 16.0, EPS));
    }

    #[test]
    fn test_warren_beta_valence_6() {
        let beta = warren_beta(6);
        // For regular vertex (valence 6): β = 1/n * (5/8 - (3/8 + cos(2π/n)/4)²)
        let inner = 3.0 / 8.0 + (2.0 * PI / 6.0).cos() / 4.0;
        let expected = (1.0 / 6.0) * (5.0 / 8.0 - inner * inner);
        assert!(approx(beta, expected, EPS));
    }

    #[test]
    fn test_warren_beta_positive() {
        for n in 3..20 {
            let beta = warren_beta(n);
            assert!(beta > 0.0, "beta should be positive for valence {n}");
        }
    }

    #[test]
    fn test_converge_to_sphere() {
        let m = octahedron();
        let sub = subdivide(&m, 3);
        // After subdivision, vertices should be roughly equidistant from origin
        let distances: Vec<f64> = sub.positions.iter().map(|p| p.length()).collect();
        let avg = distances.iter().sum::<f64>() / distances.len() as f64;
        for d in &distances {
            assert!(
                (*d - avg).abs() < 0.3,
                "distance {d} too far from average {avg}"
            );
        }
    }

    #[test]
    fn test_crease_weight() {
        let mut m = tetrahedron();
        m.set_crease(0, 1, 0.7);
        assert!(approx(m.crease_weight(0, 1), 0.7, EPS));
        assert!(approx(m.crease_weight(1, 0), 0.7, EPS));
    }

    #[test]
    fn test_crease_preserves_edge() {
        let mut m = tetrahedron();
        m.set_crease(0, 1, 1.0);
        let sub = subdivide_once(&m);
        // Edge midpoint should be exactly at midpoint of original positions
        let mid_expected = m.positions[0].add(&m.positions[1]).scale(0.5);
        // Find the edge vertex: it is at index 4 (first new vertex after 4 originals)
        // The exact index depends on iteration order, so find it by proximity
        let mut found = false;
        for i in 4..sub.positions.len() {
            let dist = sub.positions[i].sub(&mid_expected).length();
            if dist < 0.01 {
                found = true;
                break;
            }
        }
        assert!(found, "should find edge vertex near midpoint for creased edge");
    }

    #[test]
    fn test_indices_in_bounds() {
        let m = tetrahedron();
        let sub = subdivide_once(&m);
        let max_v = sub.vertex_count() as u32;
        for tri in &sub.triangles {
            assert!(tri[0] < max_v);
            assert!(tri[1] < max_v);
            assert!(tri[2] < max_v);
        }
    }

    #[test]
    fn test_all_positions_finite() {
        let m = octahedron();
        let sub = subdivide(&m, 2);
        for p in &sub.positions {
            assert!(p.x.is_finite());
            assert!(p.y.is_finite());
            assert!(p.z.is_finite());
        }
    }

    #[test]
    fn test_normals() {
        let m = tetrahedron();
        let sub = subdivide_once(&m);
        let normals = sub.compute_normals();
        assert_eq!(normals.len(), sub.vertex_count());
        for n in &normals {
            assert!(n.length() > 0.5);
        }
    }

    #[test]
    fn test_single_triangle() {
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let triangles = vec![[0, 1, 2]];
        let m = TriMesh::new(positions, triangles);
        let sub = subdivide_once(&m);
        assert_eq!(sub.triangle_count(), 4);
        assert_eq!(sub.vertex_count(), 6); // 3 original + 3 edge mids
    }

    #[test]
    fn test_level_clamped() {
        let m = tetrahedron();
        let sub = subdivide(&m, 5); // clamped to 4
        // 4 × 4^4 = 1024
        assert_eq!(sub.triangle_count(), 1024);
    }

    #[test]
    fn test_crease_clamp() {
        let mut m = tetrahedron();
        m.set_crease(0, 1, 5.0);
        assert!(approx(m.crease_weight(0, 1), 1.0, EPS));
        m.set_crease(0, 1, -2.0);
        assert!(approx(m.crease_weight(0, 1), 0.0, EPS));
    }

    #[test]
    fn test_extraordinary_vertex() {
        // Octahedron has valence-4 vertices (extraordinary for Loop, which expects 6)
        let m = octahedron();
        let sub = subdivide_once(&m);
        // Should still produce valid mesh
        assert!(sub.triangle_count() > 0);
        for tri in &sub.triangles {
            assert!(tri[0] != tri[1] && tri[1] != tri[2] && tri[0] != tri[2]);
        }
    }
}
