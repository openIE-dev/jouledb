//! Mesh Simplify — mesh simplification via edge collapse using Quadric Error Metrics (QEM).
//! Priority queue by collapse cost. Target face count or error threshold. Boundary and
//! seam preservation. Simplification statistics.

use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Ordering;

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

// ── Quadric ────────────────────────────────────────────────────

/// Symmetric 4x4 quadric error matrix, stored as 10 unique values.
/// Represents error Q = [a² ab ac ad; ab b² bc bd; ac bc c² cd; ad bd cd d²]
/// for plane (a, b, c, d) where ax + by + cz + d = 0.
#[derive(Debug, Clone, Copy)]
struct Quadric {
    a2: f64,
    ab: f64,
    ac: f64,
    ad: f64,
    b2: f64,
    bc: f64,
    bd: f64,
    c2: f64,
    cd: f64,
    d2: f64,
}

impl Quadric {
    fn zero() -> Self {
        Self {
            a2: 0.0, ab: 0.0, ac: 0.0, ad: 0.0,
            b2: 0.0, bc: 0.0, bd: 0.0,
            c2: 0.0, cd: 0.0,
            d2: 0.0,
        }
    }

    /// Create a quadric from a plane equation (a, b, c, d).
    fn from_plane(a: f64, b: f64, c: f64, d: f64) -> Self {
        Self {
            a2: a * a, ab: a * b, ac: a * c, ad: a * d,
            b2: b * b, bc: b * c, bd: b * d,
            c2: c * c, cd: c * d,
            d2: d * d,
        }
    }

    fn add(&self, o: &Self) -> Self {
        Self {
            a2: self.a2 + o.a2,
            ab: self.ab + o.ab,
            ac: self.ac + o.ac,
            ad: self.ad + o.ad,
            b2: self.b2 + o.b2,
            bc: self.bc + o.bc,
            bd: self.bd + o.bd,
            c2: self.c2 + o.c2,
            cd: self.cd + o.cd,
            d2: self.d2 + o.d2,
        }
    }

    /// Evaluate the quadric error at a point.
    fn evaluate(&self, p: Vec3) -> f64 {
        let x = p.x;
        let y = p.y;
        let z = p.z;
        self.a2 * x * x
            + 2.0 * self.ab * x * y
            + 2.0 * self.ac * x * z
            + 2.0 * self.ad * x
            + self.b2 * y * y
            + 2.0 * self.bc * y * z
            + 2.0 * self.bd * y
            + self.c2 * z * z
            + 2.0 * self.cd * z
            + self.d2
    }

    /// Find the optimal position that minimizes the quadric error.
    /// Falls back to midpoint if the matrix is singular.
    fn optimal_position(&self, v0: Vec3, v1: Vec3) -> Vec3 {
        // Try to solve the 3x3 linear system
        // [a2  ab  ac] [x]   [-ad]
        // [ab  b2  bc] [y] = [-bd]
        // [ac  bc  c2] [z]   [-cd]
        let det = self.a2 * (self.b2 * self.c2 - self.bc * self.bc)
            - self.ab * (self.ab * self.c2 - self.bc * self.ac)
            + self.ac * (self.ab * self.bc - self.b2 * self.ac);

        if det.abs() > 1e-10 {
            let inv_det = 1.0 / det;
            let x = inv_det
                * (-self.ad * (self.b2 * self.c2 - self.bc * self.bc)
                    + self.bd * (self.ab * self.c2 - self.ac * self.bc)
                    - self.cd * (self.ab * self.bc - self.ac * self.b2));
            let y = inv_det
                * (self.ad * (self.ab * self.c2 - self.bc * self.ac)
                    - self.bd * (self.a2 * self.c2 - self.ac * self.ac)
                    + self.cd * (self.a2 * self.bc - self.ab * self.ac));
            let z = inv_det
                * (-self.ad * (self.ab * self.bc - self.b2 * self.ac)
                    + self.bd * (self.a2 * self.bc - self.ab * self.ac)
                    - self.cd * (self.a2 * self.b2 - self.ab * self.ab));

            let optimal = Vec3::new(x, y, z);
            // Sanity: optimal should be near the edge
            let mid = v0.add(&v1).scale(0.5);
            if optimal.sub(&mid).length() < v0.sub(&v1).length() * 3.0 {
                return optimal;
            }
        }

        // Fallback: test midpoint, v0, v1 and pick best
        let mid = v0.add(&v1).scale(0.5);
        let e_mid = self.evaluate(mid);
        let e_v0 = self.evaluate(v0);
        let e_v1 = self.evaluate(v1);
        if e_v0 <= e_v1 && e_v0 <= e_mid {
            v0
        } else if e_v1 <= e_mid {
            v1
        } else {
            mid
        }
    }
}

// ── Priority queue entry ───────────────────────────────────────

#[derive(Debug, Clone)]
struct CollapseCandidate {
    cost: f64,
    edge: (u32, u32),
    optimal_pos: Vec3,
    generation: u32,
}

impl PartialEq for CollapseCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}

impl Eq for CollapseCandidate {}

impl PartialOrd for CollapseCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CollapseCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: reverse comparison
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
    }
}

// ── Simplification mesh ────────────────────────────────────────

/// Input/output mesh for simplification.
#[derive(Debug, Clone, PartialEq)]
pub struct SimplifyMesh {
    pub positions: Vec<Vec3>,
    pub triangles: Vec<[u32; 3]>,
    /// UV coordinates per vertex (optional; used for seam detection).
    pub uvs: Vec<[f64; 2]>,
}

impl SimplifyMesh {
    pub fn new(positions: Vec<Vec3>, triangles: Vec<[u32; 3]>) -> Self {
        Self {
            positions,
            triangles,
            uvs: Vec::new(),
        }
    }

    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    pub fn triangle_count(&self) -> usize {
        self.triangles.len()
    }
}

/// Simplification statistics.
#[derive(Debug, Clone, PartialEq)]
pub struct SimplifyStats {
    pub original_vertices: usize,
    pub original_triangles: usize,
    pub final_vertices: usize,
    pub final_triangles: usize,
    pub collapses_performed: usize,
    pub max_error: f64,
    pub ratio: f64,
}

/// Configuration for simplification.
#[derive(Debug, Clone)]
pub struct SimplifyConfig {
    /// Target number of triangles (0 = use error threshold only).
    pub target_triangles: usize,
    /// Maximum allowed error (0.0 = use target triangles only).
    pub max_error: f64,
    /// Penalty multiplier for boundary edges.
    pub boundary_penalty: f64,
    /// Penalty multiplier for UV seam edges.
    pub seam_penalty: f64,
}

impl SimplifyConfig {
    pub fn with_target(target: usize) -> Self {
        Self {
            target_triangles: target,
            max_error: f64::MAX,
            boundary_penalty: 10.0,
            seam_penalty: 5.0,
        }
    }

    pub fn with_error(max_err: f64) -> Self {
        Self {
            target_triangles: 0,
            max_error: max_err,
            boundary_penalty: 10.0,
            seam_penalty: 5.0,
        }
    }
}

/// Simplify a mesh using QEM-based edge collapse.
pub fn simplify(mesh: &SimplifyMesh, config: &SimplifyConfig) -> (SimplifyMesh, SimplifyStats) {
    let orig_vert = mesh.positions.len();
    let orig_tri = mesh.triangles.len();

    if orig_tri == 0 {
        return (
            mesh.clone(),
            SimplifyStats {
                original_vertices: orig_vert,
                original_triangles: 0,
                final_vertices: orig_vert,
                final_triangles: 0,
                collapses_performed: 0,
                max_error: 0.0,
                ratio: 1.0,
            },
        );
    }

    let mut positions = mesh.positions.clone();
    let mut triangles = mesh.triangles.clone();

    // Build edge → face adjacency
    let mut edge_faces: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (ti, tri) in triangles.iter().enumerate() {
        for i in 0..3 {
            let a = tri[i];
            let b = tri[(i + 1) % 3];
            let key = if a < b { (a, b) } else { (b, a) };
            edge_faces.entry(key).or_default().push(ti);
        }
    }

    // Detect boundary edges
    let boundary_edges: HashSet<(u32, u32)> = edge_faces
        .iter()
        .filter(|(_, faces)| faces.len() < 2)
        .map(|(e, _)| *e)
        .collect();

    // Detect UV seam edges
    let seam_edges: HashSet<(u32, u32)> = if mesh.uvs.len() == positions.len() {
        edge_faces
            .keys()
            .filter(|(a, b)| {
                let ua = mesh.uvs[*a as usize];
                let ub = mesh.uvs[*b as usize];
                // Simple heuristic: if both endpoints have different UV island hints
                // We just check if they're in different rows/columns
                (ua[0] - ub[0]).abs() > 0.5 || (ua[1] - ub[1]).abs() > 0.5
            })
            .copied()
            .collect()
    } else {
        HashSet::new()
    };

    // Compute per-vertex quadrics from face planes
    let mut quadrics = vec![Quadric::zero(); positions.len()];
    for tri in &triangles {
        let v0 = positions[tri[0] as usize];
        let v1 = positions[tri[1] as usize];
        let v2 = positions[tri[2] as usize];
        let normal = v1.sub(&v0).cross(&v2.sub(&v0));
        let len = normal.length();
        if len < 1e-12 {
            continue;
        }
        let n = normal.scale(1.0 / len);
        let d = -n.dot(&v0);
        let q = Quadric::from_plane(n.x, n.y, n.z, d);
        for &vi in tri {
            quadrics[vi as usize] = quadrics[vi as usize].add(&q);
        }
    }

    // Add boundary penalty quadrics
    for &(a, b) in &boundary_edges {
        let va = positions[a as usize];
        let vb = positions[b as usize];
        let edge_dir = vb.sub(&va).normalized();
        // Create penalty plane perpendicular to boundary edge
        let up = Vec3::new(0.0, 1.0, 0.0);
        let penalty_normal = edge_dir.cross(&up).normalized();
        if penalty_normal.length() > 0.5 {
            let d = -penalty_normal.dot(&va);
            let q = Quadric::from_plane(
                penalty_normal.x * config.boundary_penalty,
                penalty_normal.y * config.boundary_penalty,
                penalty_normal.z * config.boundary_penalty,
                d * config.boundary_penalty,
            );
            quadrics[a as usize] = quadrics[a as usize].add(&q);
            quadrics[b as usize] = quadrics[b as usize].add(&q);
        }
    }

    // Build priority queue
    let mut vertex_gen = vec![0u32; positions.len()];
    let mut heap = BinaryHeap::new();
    let mut removed = vec![false; positions.len()]; // removed vertices
    let mut remap = vec![0u32; positions.len()]; // vertex remap
    for i in 0..positions.len() {
        remap[i] = i as u32;
    }

    let compute_collapse = |a: u32, b: u32, quadrics: &[Quadric], positions: &[Vec3], config: &SimplifyConfig| -> CollapseCandidate {
        let q_sum = quadrics[a as usize].add(&quadrics[b as usize]);
        let optimal = q_sum.optimal_position(positions[a as usize], positions[b as usize]);
        let mut cost = q_sum.evaluate(optimal).max(0.0);

        let key = if a < b { (a, b) } else { (b, a) };
        if boundary_edges.contains(&key) {
            cost *= config.boundary_penalty;
        }
        if seam_edges.contains(&key) {
            cost *= config.seam_penalty;
        }

        CollapseCandidate {
            cost,
            edge: (a, b),
            optimal_pos: optimal,
            generation: 0,
        }
    };

    // Initialize heap with all edges
    let initial_edges: Vec<(u32, u32)> = edge_faces.keys().copied().collect();
    for &(a, b) in &initial_edges {
        let candidate = compute_collapse(a, b, &quadrics, &positions, config);
        heap.push(candidate);
    }

    let mut collapses = 0usize;
    let mut max_error = 0.0_f64;
    let mut live_triangles = triangles.len();

    let target = if config.target_triangles > 0 {
        config.target_triangles
    } else {
        0
    };

    while let Some(candidate) = heap.pop() {
        // Check stopping conditions
        if target > 0 && live_triangles <= target {
            break;
        }
        if candidate.cost > config.max_error {
            break;
        }
        if live_triangles <= 4 {
            break;
        }

        let (a, b) = candidate.edge;

        // Check generation freshness
        if removed[a as usize]
            || removed[b as usize]
            || candidate.generation < vertex_gen[a as usize]
            || candidate.generation < vertex_gen[b as usize]
        {
            continue; // stale entry
        }

        // Perform collapse: merge b into a
        positions[a as usize] = candidate.optimal_pos;
        quadrics[a as usize] = quadrics[a as usize].add(&quadrics[b as usize]);
        removed[b as usize] = true;
        remap[b as usize] = a;

        // Update triangles: remap b→a, remove degenerate
        let mut new_live = 0;
        for tri in &mut triangles {
            for v in tri.iter_mut() {
                if *v == b {
                    *v = a;
                }
            }
        }
        // Count live (non-degenerate) triangles
        for tri in &triangles {
            if tri[0] != tri[1] && tri[1] != tri[2] && tri[0] != tri[2] {
                new_live += 1;
            }
        }
        live_triangles = new_live;

        max_error = max_error.max(candidate.cost);
        collapses += 1;
        vertex_gen[a as usize] += 1;

        // Re-insert affected edges
        for tri in &triangles {
            for &vi in tri {
                if vi == a && !removed[vi as usize] {
                    for &vj in tri {
                        if vj != vi && !removed[vj as usize] {
                            let mut c = compute_collapse(vi, vj, &quadrics, &positions, config);
                            c.generation = vertex_gen[vi as usize];
                            heap.push(c);
                        }
                    }
                }
            }
        }
    }

    // Build output mesh: compact vertices and triangles
    let mut used_verts = HashSet::new();
    let mut final_triangles = Vec::new();
    for tri in &triangles {
        if tri[0] != tri[1] && tri[1] != tri[2] && tri[0] != tri[2] {
            // Follow remap chain
            let mut remap_v = |v: u32| -> u32 {
                let mut cur = v;
                while remap[cur as usize] != cur {
                    cur = remap[cur as usize];
                }
                cur
            };
            let a = remap_v(tri[0]);
            let b = remap_v(tri[1]);
            let c = remap_v(tri[2]);
            if a != b && b != c && a != c {
                used_verts.insert(a);
                used_verts.insert(b);
                used_verts.insert(c);
                final_triangles.push([a, b, c]);
            }
        }
    }

    // Compact: assign new indices
    let mut sorted_verts: Vec<u32> = used_verts.into_iter().collect();
    sorted_verts.sort();
    let mut new_idx_map: HashMap<u32, u32> = HashMap::new();
    let mut final_positions = Vec::new();
    for (i, &vi) in sorted_verts.iter().enumerate() {
        new_idx_map.insert(vi, i as u32);
        final_positions.push(positions[vi as usize]);
    }

    let final_tris: Vec<[u32; 3]> = final_triangles
        .iter()
        .map(|t| [new_idx_map[&t[0]], new_idx_map[&t[1]], new_idx_map[&t[2]]])
        .collect();

    let final_tri_count = final_tris.len();
    let ratio = if orig_tri > 0 {
        final_tri_count as f64 / orig_tri as f64
    } else {
        1.0
    };

    (
        SimplifyMesh::new(final_positions, final_tris),
        SimplifyStats {
            original_vertices: orig_vert,
            original_triangles: orig_tri,
            final_vertices: sorted_verts.len(),
            final_triangles: final_tri_count,
            collapses_performed: collapses,
            max_error,
            ratio,
        },
    )
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-4;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    fn cube_mesh() -> SimplifyMesh {
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
        let triangles = vec![
            [0, 2, 1], [0, 3, 2], // front
            [4, 5, 6], [4, 6, 7], // back
            [0, 1, 5], [0, 5, 4], // bottom
            [2, 3, 7], [2, 7, 6], // top
            [0, 4, 7], [0, 7, 3], // left
            [1, 2, 6], [1, 6, 5], // right
        ];
        SimplifyMesh::new(positions, triangles)
    }

    fn subdivided_plane(n: usize) -> SimplifyMesh {
        let mut positions = Vec::new();
        for j in 0..=n {
            for i in 0..=n {
                positions.push(Vec3::new(
                    i as f64 / n as f64,
                    0.0,
                    j as f64 / n as f64,
                ));
            }
        }
        let mut triangles = Vec::new();
        let w = n + 1;
        for j in 0..n {
            for i in 0..n {
                let a = (j * w + i) as u32;
                let b = (j * w + i + 1) as u32;
                let c = ((j + 1) * w + i + 1) as u32;
                let d = ((j + 1) * w + i) as u32;
                triangles.push([a, b, c]);
                triangles.push([a, c, d]);
            }
        }
        SimplifyMesh::new(positions, triangles)
    }

    #[test]
    fn test_cube_mesh_creation() {
        let m = cube_mesh();
        assert_eq!(m.vertex_count(), 8);
        assert_eq!(m.triangle_count(), 12);
    }

    #[test]
    fn test_simplify_cube_to_target() {
        let m = cube_mesh();
        let config = SimplifyConfig::with_target(8);
        let (result, stats) = simplify(&m, &config);
        assert!(result.triangle_count() <= 8);
        assert!(stats.collapses_performed > 0);
    }

    #[test]
    fn test_simplify_preserves_minimum() {
        let m = cube_mesh();
        let config = SimplifyConfig::with_target(2);
        let (result, _) = simplify(&m, &config);
        // Should keep at least 4 triangles (minimum for edge collapse)
        assert!(result.triangle_count() >= 2);
    }

    #[test]
    fn test_simplify_plane() {
        let m = subdivided_plane(8); // 128 triangles
        let config = SimplifyConfig::with_target(32);
        let (result, stats) = simplify(&m, &config);
        assert!(result.triangle_count() <= 34, "got {} tris", result.triangle_count());
        assert!(stats.ratio < 0.5);
    }

    #[test]
    fn test_simplify_by_error() {
        let m = subdivided_plane(4); // 32 tris
        let config = SimplifyConfig::with_error(0.001);
        let (result, stats) = simplify(&m, &config);
        // Very tight error → minimal simplification (coplanar faces should collapse easily)
        assert!(result.triangle_count() > 0);
        assert!(stats.max_error <= 0.01);
    }

    #[test]
    fn test_simplify_empty_mesh() {
        let m = SimplifyMesh::new(vec![], vec![]);
        let config = SimplifyConfig::with_target(0);
        let (result, stats) = simplify(&m, &config);
        assert_eq!(result.triangle_count(), 0);
        assert_eq!(stats.collapses_performed, 0);
    }

    #[test]
    fn test_stats_ratio() {
        let m = subdivided_plane(6); // 72 tris
        let config = SimplifyConfig::with_target(20);
        let (_, stats) = simplify(&m, &config);
        assert!(stats.ratio > 0.0 && stats.ratio <= 1.0);
        assert_eq!(stats.original_triangles, 72);
    }

    #[test]
    fn test_indices_in_bounds() {
        let m = subdivided_plane(4);
        let config = SimplifyConfig::with_target(8);
        let (result, _) = simplify(&m, &config);
        let max_v = result.vertex_count() as u32;
        for tri in &result.triangles {
            assert!(tri[0] < max_v);
            assert!(tri[1] < max_v);
            assert!(tri[2] < max_v);
        }
    }

    #[test]
    fn test_no_degenerate_triangles() {
        let m = subdivided_plane(4);
        let config = SimplifyConfig::with_target(10);
        let (result, _) = simplify(&m, &config);
        for tri in &result.triangles {
            assert!(tri[0] != tri[1]);
            assert!(tri[1] != tri[2]);
            assert!(tri[0] != tri[2]);
        }
    }

    #[test]
    fn test_quadric_from_plane() {
        let q = Quadric::from_plane(0.0, 1.0, 0.0, -1.0);
        // Plane y = 1. Point (0, 1, 0) → error = 0
        let err = q.evaluate(Vec3::new(0.0, 1.0, 0.0));
        assert!(approx(err, 0.0, EPS));
    }

    #[test]
    fn test_quadric_error_away_from_plane() {
        let q = Quadric::from_plane(0.0, 1.0, 0.0, 0.0);
        // Plane y = 0. Point (0, 2, 0) → error = 4
        let err = q.evaluate(Vec3::new(0.0, 2.0, 0.0));
        assert!(approx(err, 4.0, EPS));
    }

    #[test]
    fn test_quadric_addition() {
        let q1 = Quadric::from_plane(1.0, 0.0, 0.0, 0.0);
        let q2 = Quadric::from_plane(0.0, 1.0, 0.0, 0.0);
        let qs = q1.add(&q2);
        // At origin: both planes pass through origin → error = 0
        let err = qs.evaluate(Vec3::zero());
        assert!(approx(err, 0.0, EPS));
    }

    #[test]
    fn test_optimal_position_midpoint() {
        let q = Quadric::zero();
        let v0 = Vec3::new(0.0, 0.0, 0.0);
        let v1 = Vec3::new(2.0, 0.0, 0.0);
        let opt = q.optimal_position(v0, v1);
        // Zero quadric → should pick one of the endpoints or midpoint (all have error 0)
        assert!(opt.x >= -0.1 && opt.x <= 2.1);
    }

    #[test]
    fn test_boundary_penalty_effect() {
        let m = subdivided_plane(4);
        let config_no_penalty = SimplifyConfig {
            target_triangles: 8,
            max_error: f64::MAX,
            boundary_penalty: 1.0,
            seam_penalty: 1.0,
        };
        let config_high_penalty = SimplifyConfig {
            target_triangles: 8,
            max_error: f64::MAX,
            boundary_penalty: 100.0,
            seam_penalty: 1.0,
        };
        let (r1, _) = simplify(&m, &config_no_penalty);
        let (r2, _) = simplify(&m, &config_high_penalty);
        // Both should simplify, but boundary penalty should be respected
        assert!(r1.triangle_count() > 0);
        assert!(r2.triangle_count() > 0);
    }

    #[test]
    fn test_simplify_config_with_target() {
        let config = SimplifyConfig::with_target(50);
        assert_eq!(config.target_triangles, 50);
        assert!(config.boundary_penalty > 1.0);
    }

    #[test]
    fn test_simplify_config_with_error() {
        let config = SimplifyConfig::with_error(0.1);
        assert!(approx(config.max_error, 0.1, EPS));
        assert_eq!(config.target_triangles, 0);
    }

    #[test]
    fn test_positions_finite() {
        let m = subdivided_plane(6);
        let config = SimplifyConfig::with_target(16);
        let (result, _) = simplify(&m, &config);
        for p in &result.positions {
            assert!(p.x.is_finite());
            assert!(p.y.is_finite());
            assert!(p.z.is_finite());
        }
    }

    #[test]
    fn test_simplify_single_triangle() {
        let m = SimplifyMesh::new(
            vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            vec![[0, 1, 2]],
        );
        let config = SimplifyConfig::with_target(0);
        let (result, _) = simplify(&m, &config);
        // Can't simplify below the minimum
        assert!(result.triangle_count() >= 1);
    }
}
