//! Navigation mesh — polygon-based navmesh, point-in-polygon, A* on polygon
//! graph, string pulling (funnel algorithm), dynamic obstacles, agent radius,
//! off-mesh links.
//!
//! Replaces JavaScript navmesh libraries (nav2d, yuka navmesh) with a pure-Rust
//! navigation mesh for 2D game worlds.

use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Ordering;

// ── 2D Vector ───────────────────────────────────────────────────

/// Simple 2D vector / point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    pub fn new(x: f64, y: f64) -> Self { Self { x, y } }

    pub fn dist(&self, other: Vec2) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }

    pub fn lerp(&self, other: Vec2, t: f64) -> Vec2 {
        Vec2::new(self.x + (other.x - self.x) * t, self.y + (other.y - self.y) * t)
    }
}

// ── Polygon ─────────────────────────────────────────────────────

/// A convex polygon in the navmesh.
#[derive(Debug, Clone)]
pub struct NavPolygon {
    pub id: usize,
    /// Vertices in CCW order.
    pub vertices: Vec<Vec2>,
    /// IDs of adjacent polygons, one per edge.
    pub neighbors: Vec<Option<usize>>,
    /// Whether this polygon is blocked (dynamic obstacle).
    pub blocked: bool,
}

impl NavPolygon {
    pub fn new(id: usize, vertices: Vec<Vec2>) -> Self {
        let edge_count = vertices.len();
        Self {
            id,
            vertices,
            neighbors: vec![None; edge_count],
            blocked: false,
        }
    }

    /// Centroid of the polygon.
    pub fn centroid(&self) -> Vec2 {
        let n = self.vertices.len() as f64;
        let sx: f64 = self.vertices.iter().map(|v| v.x).sum();
        let sy: f64 = self.vertices.iter().map(|v| v.y).sum();
        Vec2::new(sx / n, sy / n)
    }

    /// Point-in-polygon using ray casting.
    pub fn contains(&self, p: Vec2) -> bool {
        let mut inside = false;
        let n = self.vertices.len();
        let mut j = n - 1;
        for i in 0..n {
            let vi = self.vertices[i];
            let vj = self.vertices[j];
            if ((vi.y > p.y) != (vj.y > p.y))
                && (p.x < (vj.x - vi.x) * (p.y - vi.y) / (vj.y - vi.y) + vi.x)
            {
                inside = !inside;
            }
            j = i;
        }
        inside
    }

    /// Get the shared edge between this polygon and a neighbor.
    /// Returns the two endpoints of the shared edge.
    pub fn shared_edge(&self, neighbor_id: usize) -> Option<(Vec2, Vec2)> {
        let n = self.vertices.len();
        for i in 0..n {
            if self.neighbors[i] == Some(neighbor_id) {
                let a = self.vertices[i];
                let b = self.vertices[(i + 1) % n];
                return Some((a, b));
            }
        }
        None
    }
}

// ── Off-mesh link ───────────────────────────────────────────────

/// A link between two non-adjacent polygons (e.g., jump, teleport).
#[derive(Debug, Clone)]
pub struct OffMeshLink {
    pub from_poly: usize,
    pub to_poly: usize,
    pub from_point: Vec2,
    pub to_point: Vec2,
    pub cost: f64,
    pub bidirectional: bool,
}

// ── NavMesh ─────────────────────────────────────────────────────

/// Navigation mesh: a graph of convex polygons.
pub struct NavMesh {
    pub polygons: Vec<NavPolygon>,
    pub off_mesh_links: Vec<OffMeshLink>,
}

impl NavMesh {
    pub fn new() -> Self {
        Self {
            polygons: Vec::new(),
            off_mesh_links: Vec::new(),
        }
    }

    /// Add a polygon and return its id.
    pub fn add_polygon(&mut self, vertices: Vec<Vec2>) -> usize {
        let id = self.polygons.len();
        self.polygons.push(NavPolygon::new(id, vertices));
        id
    }

    /// Connect two polygons along an edge. `edge_a` is the edge index in polygon `a`.
    pub fn connect(&mut self, a: usize, edge_a: usize, b: usize, edge_b: usize) {
        self.polygons[a].neighbors[edge_a] = Some(b);
        self.polygons[b].neighbors[edge_b] = Some(a);
    }

    /// Add an off-mesh link.
    pub fn add_off_mesh_link(&mut self, link: OffMeshLink) {
        self.off_mesh_links.push(link);
    }

    /// Mark a polygon as blocked (dynamic obstacle).
    pub fn set_blocked(&mut self, poly_id: usize, blocked: bool) {
        self.polygons[poly_id].blocked = blocked;
    }

    /// Find which polygon contains a point.
    pub fn find_polygon(&self, p: Vec2) -> Option<usize> {
        self.polygons
            .iter()
            .find(|poly| !poly.blocked && poly.contains(p))
            .map(|poly| poly.id)
    }

    /// Find which polygon contains a point, considering agent radius.
    /// The point must be at least `radius` away from all polygon edges.
    pub fn find_polygon_with_radius(&self, p: Vec2, radius: f64) -> Option<usize> {
        self.polygons.iter().find(|poly| {
            if poly.blocked || !poly.contains(p) { return false; }
            // Check distance to each edge.
            let n = poly.vertices.len();
            for i in 0..n {
                let a = poly.vertices[i];
                let b = poly.vertices[(i + 1) % n];
                if point_to_segment_dist(p, a, b) < radius {
                    return false;
                }
            }
            true
        }).map(|poly| poly.id)
    }

    /// A* pathfinding on the polygon graph. Returns list of polygon ids.
    pub fn find_path_polys(&self, start: Vec2, goal: Vec2) -> Option<Vec<usize>> {
        let start_poly = self.find_polygon(start)?;
        let goal_poly = self.find_polygon(goal)?;

        if start_poly == goal_poly {
            return Some(vec![start_poly]);
        }

        let mut open = BinaryHeap::new();
        let mut g_score: HashMap<usize, f64> = HashMap::new();
        let mut came_from: HashMap<usize, usize> = HashMap::new();
        let mut closed: HashSet<usize> = HashSet::new();

        g_score.insert(start_poly, 0.0);
        let h = self.polygons[start_poly].centroid().dist(self.polygons[goal_poly].centroid());
        open.push(NavHeapEntry { cost: h, poly_id: start_poly });

        while let Some(NavHeapEntry { poly_id: current, .. }) = open.pop() {
            if current == goal_poly {
                return Some(reconstruct(came_from, current));
            }
            if !closed.insert(current) { continue; }

            let current_g = g_score[&current];
            let poly = &self.polygons[current];

            // Regular neighbors.
            for neighbor_id in poly.neighbors.iter().filter_map(|n| *n) {
                if closed.contains(&neighbor_id) { continue; }
                if self.polygons[neighbor_id].blocked { continue; }
                let edge_cost = poly.centroid().dist(self.polygons[neighbor_id].centroid());
                let tentative_g = current_g + edge_cost;
                if tentative_g < *g_score.get(&neighbor_id).unwrap_or(&f64::INFINITY) {
                    g_score.insert(neighbor_id, tentative_g);
                    came_from.insert(neighbor_id, current);
                    let h = self.polygons[neighbor_id].centroid().dist(self.polygons[goal_poly].centroid());
                    open.push(NavHeapEntry { cost: tentative_g + h, poly_id: neighbor_id });
                }
            }

            // Off-mesh links.
            for link in &self.off_mesh_links {
                let neighbor_id = if link.from_poly == current {
                    link.to_poly
                } else if link.bidirectional && link.to_poly == current {
                    link.from_poly
                } else {
                    continue;
                };
                if closed.contains(&neighbor_id) || self.polygons[neighbor_id].blocked { continue; }
                let tentative_g = current_g + link.cost;
                if tentative_g < *g_score.get(&neighbor_id).unwrap_or(&f64::INFINITY) {
                    g_score.insert(neighbor_id, tentative_g);
                    came_from.insert(neighbor_id, current);
                    let h = self.polygons[neighbor_id].centroid().dist(self.polygons[goal_poly].centroid());
                    open.push(NavHeapEntry { cost: tentative_g + h, poly_id: neighbor_id });
                }
            }
        }
        None
    }

    /// Full path: A* on polygons + string pulling for smooth waypoints.
    pub fn find_path(&self, start: Vec2, goal: Vec2) -> Option<Vec<Vec2>> {
        let poly_path = self.find_path_polys(start, goal)?;
        if poly_path.len() == 1 {
            return Some(vec![start, goal]);
        }

        // Build portal edges.
        let mut portals = Vec::new();
        portals.push((start, start)); // degenerate portal at start
        for i in 0..poly_path.len() - 1 {
            let poly = &self.polygons[poly_path[i]];
            if let Some((a, b)) = poly.shared_edge(poly_path[i + 1]) {
                portals.push((a, b));
            } else {
                // Off-mesh link — use the link endpoints.
                for link in &self.off_mesh_links {
                    if link.from_poly == poly_path[i] && link.to_poly == poly_path[i + 1] {
                        portals.push((link.from_point, link.to_point));
                        break;
                    } else if link.bidirectional && link.to_poly == poly_path[i] && link.from_poly == poly_path[i + 1] {
                        portals.push((link.to_point, link.from_point));
                        break;
                    }
                }
            }
        }
        portals.push((goal, goal)); // degenerate portal at goal

        Some(funnel_algorithm(&portals))
    }
}

impl Default for NavMesh {
    fn default() -> Self { Self::new() }
}

// ── Funnel algorithm (string pulling) ───────────────────────────

fn funnel_algorithm(portals: &[(Vec2, Vec2)]) -> Vec<Vec2> {
    if portals.is_empty() { return Vec::new(); }
    if portals.len() == 1 { return vec![portals[0].0]; }

    let mut path = vec![portals[0].0];
    let mut apex = portals[0].0;
    let mut left = portals[0].0;
    let mut right = portals[0].0;
    let mut apex_idx = 0;
    let mut left_idx = 0;
    let mut right_idx = 0;

    for i in 1..portals.len() {
        let (pl, pr) = portals[i];

        // Update right.
        if cross_2d(apex, right, pr) <= 0.0 {
            if apex == right || cross_2d(apex, left, pr) > 0.0 {
                right = pr;
                right_idx = i;
            } else {
                path.push(left);
                apex = left;
                apex_idx = left_idx;
                left = apex;
                right = apex;
                left_idx = apex_idx;
                right_idx = apex_idx;
                // Restart scan.
                continue;
            }
        }

        // Update left.
        if cross_2d(apex, left, pl) >= 0.0 {
            if apex == left || cross_2d(apex, right, pl) < 0.0 {
                left = pl;
                left_idx = i;
            } else {
                path.push(right);
                apex = right;
                apex_idx = right_idx;
                left = apex;
                right = apex;
                left_idx = apex_idx;
                right_idx = apex_idx;
                continue;
            }
        }
    }

    let last = portals.last().unwrap().0;
    if path.last() != Some(&last) {
        path.push(last);
    }
    path
}

fn cross_2d(o: Vec2, a: Vec2, b: Vec2) -> f64 {
    (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
}

// ── Helpers ─────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct NavHeapEntry {
    cost: f64,
    poly_id: usize,
}

impl PartialEq for NavHeapEntry {
    fn eq(&self, other: &Self) -> bool { self.cost == other.cost }
}
impl Eq for NavHeapEntry {}
impl PartialOrd for NavHeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl Ord for NavHeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other.cost.partial_cmp(&self.cost).unwrap_or(Ordering::Equal)
    }
}

fn reconstruct(came_from: HashMap<usize, usize>, mut current: usize) -> Vec<usize> {
    let mut path = vec![current];
    while let Some(&prev) = came_from.get(&current) {
        path.push(prev);
        current = prev;
    }
    path.reverse();
    path
}

fn point_to_segment_dist(p: Vec2, a: Vec2, b: Vec2) -> f64 {
    let ab_x = b.x - a.x;
    let ab_y = b.y - a.y;
    let ap_x = p.x - a.x;
    let ap_y = p.y - a.y;
    let ab_sq = ab_x * ab_x + ab_y * ab_y;
    if ab_sq < 1e-12 { return p.dist(a); }
    let t = ((ap_x * ab_x + ap_y * ab_y) / ab_sq).clamp(0.0, 1.0);
    let proj = Vec2::new(a.x + t * ab_x, a.y + t * ab_y);
    p.dist(proj)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_mesh() -> NavMesh {
        // Two adjacent triangles forming a square:
        // Poly 0: (0,0), (10,0), (10,10) — lower right triangle
        // Poly 1: (0,0), (10,10), (0,10) — upper left triangle
        let mut mesh = NavMesh::new();
        mesh.add_polygon(vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(10.0, 0.0),
            Vec2::new(10.0, 10.0),
        ]);
        mesh.add_polygon(vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(10.0, 10.0),
            Vec2::new(0.0, 10.0),
        ]);
        // Edge 2 of poly 0 (10,0→10,10 ... actually edge between (0,0)→(10,10))
        // Shared edge is (0,0)→(10,10).
        // Poly 0 edge 2: (10,10)→(0,0), poly 1 edge 0: (0,0)→(10,10)
        mesh.connect(0, 2, 1, 0);
        mesh
    }

    #[test]
    fn point_in_polygon() {
        let mesh = simple_mesh();
        // (5, 2) is in poly 0 (lower right triangle).
        assert!(mesh.polygons[0].contains(Vec2::new(5.0, 2.0)));
        // (2, 8) is in poly 1 (upper left triangle).
        assert!(mesh.polygons[1].contains(Vec2::new(2.0, 8.0)));
    }

    #[test]
    fn find_polygon() {
        let mesh = simple_mesh();
        assert_eq!(mesh.find_polygon(Vec2::new(5.0, 2.0)), Some(0));
        assert_eq!(mesh.find_polygon(Vec2::new(2.0, 8.0)), Some(1));
        assert_eq!(mesh.find_polygon(Vec2::new(20.0, 20.0)), None);
    }

    #[test]
    fn path_same_polygon() {
        let mesh = simple_mesh();
        let path = mesh.find_path(Vec2::new(5.0, 2.0), Vec2::new(8.0, 3.0)).unwrap();
        assert_eq!(path.len(), 2);
    }

    #[test]
    fn path_across_polygons() {
        let mesh = simple_mesh();
        let polys = mesh.find_path_polys(Vec2::new(5.0, 2.0), Vec2::new(2.0, 8.0)).unwrap();
        assert_eq!(polys, vec![0, 1]);
    }

    #[test]
    fn path_with_string_pulling() {
        let mesh = simple_mesh();
        let path = mesh.find_path(Vec2::new(5.0, 2.0), Vec2::new(2.0, 8.0)).unwrap();
        assert!(!path.is_empty());
        assert_eq!(path.first().unwrap().x, 5.0);
        assert_eq!(path.last().unwrap().x, 2.0);
    }

    #[test]
    fn dynamic_obstacle() {
        let mut mesh = simple_mesh();
        mesh.set_blocked(1, true);
        assert!(mesh.find_polygon(Vec2::new(2.0, 8.0)).is_none());
        let path = mesh.find_path_polys(Vec2::new(5.0, 2.0), Vec2::new(2.0, 8.0));
        assert!(path.is_none());
    }

    #[test]
    fn off_mesh_link() {
        let mut mesh = NavMesh::new();
        mesh.add_polygon(vec![
            Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0),
            Vec2::new(10.0, 10.0), Vec2::new(0.0, 10.0),
        ]);
        mesh.add_polygon(vec![
            Vec2::new(50.0, 50.0), Vec2::new(60.0, 50.0),
            Vec2::new(60.0, 60.0), Vec2::new(50.0, 60.0),
        ]);
        mesh.add_off_mesh_link(OffMeshLink {
            from_poly: 0,
            to_poly: 1,
            from_point: Vec2::new(5.0, 5.0),
            to_point: Vec2::new(55.0, 55.0),
            cost: 10.0,
            bidirectional: true,
        });
        let polys = mesh.find_path_polys(Vec2::new(5.0, 5.0), Vec2::new(55.0, 55.0)).unwrap();
        assert_eq!(polys, vec![0, 1]);
    }

    #[test]
    fn agent_radius_clearance() {
        let mesh = simple_mesh();
        // A point very close to edge should fail with large radius.
        let corner = Vec2::new(0.5, 0.1);
        // With tiny radius, should be found.
        assert!(mesh.find_polygon_with_radius(corner, 0.01).is_some());
        // With large radius, too close to edges.
        assert!(mesh.find_polygon_with_radius(corner, 1.0).is_none());
    }

    #[test]
    fn centroid() {
        let mesh = simple_mesh();
        let c = mesh.polygons[0].centroid();
        // Triangle (0,0), (10,0), (10,10) → centroid (6.67, 3.33)
        assert!((c.x - 20.0 / 3.0).abs() < 0.1);
        assert!((c.y - 10.0 / 3.0).abs() < 0.1);
    }

    #[test]
    fn shared_edge() {
        let mesh = simple_mesh();
        let edge = mesh.polygons[0].shared_edge(1);
        assert!(edge.is_some());
    }

    #[test]
    fn vec2_operations() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(3.0, 4.0);
        assert!((a.dist(b) - 5.0).abs() < 1e-9);
        let mid = a.lerp(b, 0.5);
        assert!((mid.x - 1.5).abs() < 1e-9);
        assert!((mid.y - 2.0).abs() < 1e-9);
    }

    #[test]
    fn no_path_disconnected() {
        let mut mesh = NavMesh::new();
        mesh.add_polygon(vec![
            Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0),
            Vec2::new(10.0, 10.0), Vec2::new(0.0, 10.0),
        ]);
        mesh.add_polygon(vec![
            Vec2::new(50.0, 50.0), Vec2::new(60.0, 50.0),
            Vec2::new(60.0, 60.0), Vec2::new(50.0, 60.0),
        ]);
        // No connection, no off-mesh link.
        let path = mesh.find_path_polys(Vec2::new(5.0, 5.0), Vec2::new(55.0, 55.0));
        assert!(path.is_none());
    }
}
