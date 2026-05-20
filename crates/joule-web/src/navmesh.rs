//! Navigation mesh for pathfinding — convex polygon regions with adjacency,
//! point-in-polygon queries, funnel algorithm (string-pulling), area costs,
//! off-mesh links (jump points, ladders), navmesh from triangle soup.
//!
//! Replaces JavaScript navmesh libraries (navmesh, yuka NavMesh) with a
//! pure-Rust navigation mesh for games and simulations.

use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Ordering;

// ── Vec2 ────────────────────────────────────────────────────────

/// 2D vector for navigation math.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    pub const ZERO: Vec2 = Vec2 { x: 0.0, y: 0.0 };

    pub fn new(x: f64, y: f64) -> Self { Self { x, y } }

    pub fn length(&self) -> f64 { (self.x * self.x + self.y * self.y).sqrt() }

    pub fn dist(&self, other: Vec2) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }

    pub fn sub(&self, other: Vec2) -> Vec2 {
        Vec2 { x: self.x - other.x, y: self.y - other.y }
    }

    pub fn add(&self, other: Vec2) -> Vec2 {
        Vec2 { x: self.x + other.x, y: self.y + other.y }
    }

    pub fn scale(&self, s: f64) -> Vec2 {
        Vec2 { x: self.x * s, y: self.y * s }
    }

    pub fn cross(&self, other: Vec2) -> f64 {
        self.x * other.y - self.y * other.x
    }

    pub fn dot(&self, other: Vec2) -> f64 {
        self.x * other.x + self.y * other.y
    }
}

// ── Surface type and costs ──────────────────────────────────────

/// Surface type affecting movement cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SurfaceType {
    Normal,
    Grass,
    Swamp,
    Road,
    Custom(u32),
}

/// Get traversal cost multiplier for a surface type.
pub fn surface_cost(surface: SurfaceType) -> f64 {
    match surface {
        SurfaceType::Normal => 1.0,
        SurfaceType::Grass => 1.0,
        SurfaceType::Swamp => 3.0,
        SurfaceType::Road => 0.5,
        SurfaceType::Custom(_) => 1.0,
    }
}

// ── Off-mesh link ───────────────────────────────────────────────

/// Off-mesh link types (jump points, ladders, teleports).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LinkType {
    Jump,
    Ladder,
    Teleport,
}

/// An off-mesh link connecting two polygons outside normal adjacency.
#[derive(Debug, Clone)]
pub struct OffMeshLink {
    pub from_poly: usize,
    pub to_poly: usize,
    pub from_pos: Vec2,
    pub to_pos: Vec2,
    pub link_type: LinkType,
    pub cost: f64,
    pub bidirectional: bool,
}

// ── Portal (shared edge) ────────────────────────────────────────

/// A portal is a shared edge between two adjacent polygons.
#[derive(Debug, Clone, Copy)]
pub struct Portal {
    pub left: Vec2,
    pub right: Vec2,
}

// ── NavPoly ─────────────────────────────────────────────────────

/// A convex polygon region in the navigation mesh.
#[derive(Debug, Clone)]
pub struct NavPoly {
    pub id: usize,
    pub vertices: Vec<Vec2>,
    pub surface: SurfaceType,
}

impl NavPoly {
    /// Compute the centroid of the polygon.
    pub fn centroid(&self) -> Vec2 {
        if self.vertices.is_empty() {
            return Vec2::ZERO;
        }
        let n = self.vertices.len() as f64;
        let sum_x: f64 = self.vertices.iter().map(|v| v.x).sum();
        let sum_y: f64 = self.vertices.iter().map(|v| v.y).sum();
        Vec2::new(sum_x / n, sum_y / n)
    }

    /// Compute the area of the polygon using the shoelace formula.
    pub fn area(&self) -> f64 {
        let n = self.vertices.len();
        if n < 3 {
            return 0.0;
        }
        let mut sum = 0.0;
        for i in 0..n {
            let j = (i + 1) % n;
            sum += self.vertices[i].x * self.vertices[j].y;
            sum -= self.vertices[j].x * self.vertices[i].y;
        }
        sum.abs() / 2.0
    }

    /// Point-in-polygon test using cross product method for convex polygons.
    pub fn contains(&self, point: Vec2) -> bool {
        let n = self.vertices.len();
        if n < 3 {
            return false;
        }
        let mut positive = 0;
        let mut negative = 0;
        for i in 0..n {
            let j = (i + 1) % n;
            let edge = self.vertices[j].sub(self.vertices[i]);
            let to_point = point.sub(self.vertices[i]);
            let cross = edge.cross(to_point);
            if cross > 1e-10 {
                positive += 1;
            } else if cross < -1e-10 {
                negative += 1;
            }
            // On edge counts as inside
        }
        positive == 0 || negative == 0
    }

    /// Find closest point on polygon boundary to a given point.
    pub fn closest_point_on_boundary(&self, point: Vec2) -> Vec2 {
        let n = self.vertices.len();
        if n == 0 {
            return Vec2::ZERO;
        }
        let mut best = self.vertices[0];
        let mut best_dist = f64::MAX;
        for i in 0..n {
            let j = (i + 1) % n;
            let cp = closest_point_on_segment(point, self.vertices[i], self.vertices[j]);
            let d = point.dist(cp);
            if d < best_dist {
                best_dist = d;
                best = cp;
            }
        }
        best
    }
}

/// Closest point on line segment a-b to point p.
fn closest_point_on_segment(p: Vec2, a: Vec2, b: Vec2) -> Vec2 {
    let ab = b.sub(a);
    let len_sq = ab.dot(ab);
    if len_sq < 1e-12 {
        return a;
    }
    let t = (p.sub(a).dot(ab) / len_sq).clamp(0.0, 1.0);
    a.add(ab.scale(t))
}

// ── NavMesh ─────────────────────────────────────────────────────

/// Navigation mesh composed of convex polygons.
#[derive(Debug, Clone)]
pub struct NavMesh {
    pub polygons: Vec<NavPoly>,
    /// Adjacency: polygon id -> list of (neighbor_id, portal)
    adjacency: HashMap<usize, Vec<(usize, Portal)>>,
    /// Off-mesh links
    off_mesh_links: Vec<OffMeshLink>,
}

impl NavMesh {
    pub fn new() -> Self {
        Self {
            polygons: Vec::new(),
            adjacency: HashMap::new(),
            off_mesh_links: Vec::new(),
        }
    }

    /// Add a convex polygon to the mesh.
    pub fn add_polygon(&mut self, vertices: Vec<Vec2>, surface: SurfaceType) -> usize {
        let id = self.polygons.len();
        self.polygons.push(NavPoly { id, vertices, surface });
        id
    }

    /// Connect two polygons with a shared edge (portal).
    pub fn connect(&mut self, poly_a: usize, poly_b: usize, portal: Portal) {
        self.adjacency.entry(poly_a).or_default().push((poly_b, portal));
        let reverse_portal = Portal { left: portal.right, right: portal.left };
        self.adjacency.entry(poly_b).or_default().push((poly_a, reverse_portal));
    }

    /// Add an off-mesh link.
    pub fn add_off_mesh_link(&mut self, link: OffMeshLink) {
        let bidir = link.bidirectional;
        let reverse = if bidir {
            Some(OffMeshLink {
                from_poly: link.to_poly,
                to_poly: link.from_poly,
                from_pos: link.to_pos,
                to_pos: link.from_pos,
                link_type: link.link_type,
                cost: link.cost,
                bidirectional: true,
            })
        } else {
            None
        };
        self.off_mesh_links.push(link);
        if let Some(r) = reverse {
            self.off_mesh_links.push(r);
        }
    }

    /// Find which polygon contains the given point.
    pub fn find_polygon(&self, point: Vec2) -> Option<usize> {
        for poly in &self.polygons {
            if poly.contains(point) {
                return Some(poly.id);
            }
        }
        None
    }

    /// Find the nearest polygon to a point (by centroid distance).
    pub fn nearest_polygon(&self, point: Vec2) -> Option<usize> {
        if self.polygons.is_empty() {
            return None;
        }
        let mut best_id = 0;
        let mut best_dist = f64::MAX;
        for poly in &self.polygons {
            // Try point-in-polygon first
            if poly.contains(point) {
                return Some(poly.id);
            }
            let cp = poly.closest_point_on_boundary(point);
            let d = point.dist(cp);
            if d < best_dist {
                best_dist = d;
                best_id = poly.id;
            }
        }
        Some(best_id)
    }

    /// Get neighbors of a polygon (adjacency + off-mesh links).
    pub fn neighbors(&self, poly_id: usize) -> Vec<(usize, f64, Option<Portal>)> {
        let mut result = Vec::new();
        if let Some(adj) = self.adjacency.get(&poly_id) {
            for (neighbor, portal) in adj {
                let cost = self.edge_cost(poly_id, *neighbor, Some(*portal));
                result.push((*neighbor, cost, Some(*portal)));
            }
        }
        for link in &self.off_mesh_links {
            if link.from_poly == poly_id {
                result.push((link.to_poly, link.cost, None));
            }
        }
        result
    }

    /// Compute traversal cost between two adjacent polygons.
    fn edge_cost(&self, from: usize, to: usize, portal: Option<Portal>) -> f64 {
        let from_centroid = self.polygons[from].centroid();
        let to_centroid = self.polygons[to].centroid();
        let dist = match portal {
            Some(p) => {
                let mid = p.left.add(p.right).scale(0.5);
                from_centroid.dist(mid) + mid.dist(to_centroid)
            }
            None => from_centroid.dist(to_centroid),
        };
        let to_cost = surface_cost(self.polygons[to].surface);
        dist * to_cost
    }

    /// A* pathfinding across polygons. Returns polygon path.
    pub fn find_poly_path(&self, start: Vec2, goal: Vec2) -> Option<Vec<usize>> {
        let start_poly = self.nearest_polygon(start)?;
        let goal_poly = self.nearest_polygon(goal)?;

        if start_poly == goal_poly {
            return Some(vec![start_poly]);
        }

        let goal_centroid = self.polygons[goal_poly].centroid();

        #[derive(Debug)]
        struct Node {
            poly_id: usize,
            f_cost: f64,
        }
        impl PartialEq for Node {
            fn eq(&self, other: &Self) -> bool { self.poly_id == other.poly_id }
        }
        impl Eq for Node {}
        impl Ord for Node {
            fn cmp(&self, other: &Self) -> Ordering {
                other.f_cost.partial_cmp(&self.f_cost)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| self.poly_id.cmp(&other.poly_id))
            }
        }
        impl PartialOrd for Node {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        let mut open = BinaryHeap::new();
        let mut g_cost: HashMap<usize, f64> = HashMap::new();
        let mut came_from: HashMap<usize, usize> = HashMap::new();
        let mut closed: HashSet<usize> = HashSet::new();

        g_cost.insert(start_poly, 0.0);
        let h = self.polygons[start_poly].centroid().dist(goal_centroid);
        open.push(Node { poly_id: start_poly, f_cost: h });

        while let Some(current) = open.pop() {
            if current.poly_id == goal_poly {
                let mut path = vec![goal_poly];
                let mut cur = goal_poly;
                while let Some(&prev) = came_from.get(&cur) {
                    path.push(prev);
                    cur = prev;
                }
                path.reverse();
                return Some(path);
            }

            if !closed.insert(current.poly_id) {
                continue;
            }

            let current_g = g_cost[&current.poly_id];

            for (neighbor, cost, _portal) in self.neighbors(current.poly_id) {
                if closed.contains(&neighbor) {
                    continue;
                }
                let new_g = current_g + cost;
                let existing_g = g_cost.get(&neighbor).copied().unwrap_or(f64::MAX);
                if new_g < existing_g {
                    g_cost.insert(neighbor, new_g);
                    came_from.insert(neighbor, current.poly_id);
                    let h = self.polygons[neighbor].centroid().dist(goal_centroid);
                    open.push(Node { poly_id: neighbor, f_cost: new_g + h });
                }
            }
        }

        None
    }

    /// Get the portal between two adjacent polygons.
    pub fn get_portal(&self, from: usize, to: usize) -> Option<Portal> {
        if let Some(adj) = self.adjacency.get(&from) {
            for (neighbor, portal) in adj {
                if *neighbor == to {
                    return Some(*portal);
                }
            }
        }
        None
    }

    /// String-pulling / funnel algorithm to produce a smooth path through portals.
    pub fn funnel_path(&self, start: Vec2, goal: Vec2, poly_path: &[usize]) -> Vec<Vec2> {
        if poly_path.is_empty() {
            return vec![start];
        }
        if poly_path.len() == 1 {
            return vec![start, goal];
        }

        // Collect portals
        let mut portals: Vec<Portal> = Vec::new();
        for i in 0..poly_path.len() - 1 {
            if let Some(portal) = self.get_portal(poly_path[i], poly_path[i + 1]) {
                portals.push(portal);
            }
        }

        if portals.is_empty() {
            return vec![start, goal];
        }

        // Simple funnel algorithm
        let mut path = vec![start];
        let mut apex = start;
        let mut left = portals[0].left;
        let mut right = portals[0].right;
        let mut left_idx: usize = 0;
        let mut right_idx: usize = 0;
        let mut i = 1;

        while i < portals.len() {
            let new_left = portals[i].left;
            let new_right = portals[i].right;

            // Update right
            if tri_area2(apex, right, new_right) <= 0.0 {
                if apex.dist(right) < 1e-10 || tri_area2(apex, left, new_right) > 0.0 {
                    right = new_right;
                    right_idx = i;
                } else {
                    path.push(left);
                    apex = left;
                    let restart = left_idx + 1;
                    left_idx = restart;
                    right_idx = restart;
                    if restart < portals.len() {
                        left = portals[restart].left;
                        right = portals[restart].right;
                    }
                    i = restart;
                    continue;
                }
            }

            // Update left
            if tri_area2(apex, left, new_left) >= 0.0 {
                if apex.dist(left) < 1e-10 || tri_area2(apex, right, new_left) < 0.0 {
                    left = new_left;
                    left_idx = i;
                } else {
                    path.push(right);
                    apex = right;
                    let restart = right_idx + 1;
                    left_idx = restart;
                    right_idx = restart;
                    if restart < portals.len() {
                        left = portals[restart].left;
                        right = portals[restart].right;
                    }
                    i = restart;
                    continue;
                }
            }

            i += 1;
        }

        path.push(goal);
        path
    }

    /// Full pathfind: find polygon path then smooth with funnel algorithm.
    pub fn find_path(&self, start: Vec2, goal: Vec2) -> Option<Vec<Vec2>> {
        let poly_path = self.find_poly_path(start, goal)?;
        Some(self.funnel_path(start, goal, &poly_path))
    }
}

/// Signed area of triangle * 2 (for funnel algorithm).
fn tri_area2(a: Vec2, b: Vec2, c: Vec2) -> f64 {
    (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)
}

// ── Triangle soup to navmesh ────────────────────────────────────

/// A triangle defined by three vertices.
#[derive(Debug, Clone)]
pub struct Triangle {
    pub v0: Vec2,
    pub v1: Vec2,
    pub v2: Vec2,
}

/// Build a NavMesh from triangle soup by detecting shared edges.
pub fn navmesh_from_triangles(triangles: &[Triangle], surface: SurfaceType) -> NavMesh {
    let mut mesh = NavMesh::new();

    // Add each triangle as a polygon
    for tri in triangles {
        mesh.add_polygon(vec![tri.v0, tri.v1, tri.v2], surface);
    }

    // Detect shared edges and create adjacency
    let n = triangles.len();
    for i in 0..n {
        let edges_i = triangle_edges(&triangles[i]);
        for j in (i + 1)..n {
            let edges_j = triangle_edges(&triangles[j]);
            for ei in &edges_i {
                for ej in &edges_j {
                    if edges_match(ei, ej) {
                        let portal = Portal { left: ei.0, right: ei.1 };
                        mesh.connect(i, j, portal);
                    }
                }
            }
        }
    }

    mesh
}

/// Get edges of a triangle as pairs.
fn triangle_edges(tri: &Triangle) -> [(Vec2, Vec2); 3] {
    [(tri.v0, tri.v1), (tri.v1, tri.v2), (tri.v2, tri.v0)]
}

/// Check if two edges match (same endpoints, either direction).
fn edges_match(a: &(Vec2, Vec2), b: &(Vec2, Vec2)) -> bool {
    let eps = 1e-6;
    (a.0.dist(b.0) < eps && a.1.dist(b.1) < eps)
        || (a.0.dist(b.1) < eps && a.1.dist(b.0) < eps)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn square_poly(x: f64, y: f64, size: f64) -> Vec<Vec2> {
        vec![
            Vec2::new(x, y),
            Vec2::new(x + size, y),
            Vec2::new(x + size, y + size),
            Vec2::new(x, y + size),
        ]
    }

    fn build_two_square_mesh() -> NavMesh {
        let mut mesh = NavMesh::new();
        // Two adjacent squares sharing edge at x=1
        mesh.add_polygon(square_poly(0.0, 0.0, 1.0), SurfaceType::Normal);
        mesh.add_polygon(square_poly(1.0, 0.0, 1.0), SurfaceType::Normal);
        mesh.connect(0, 1, Portal {
            left: Vec2::new(1.0, 0.0),
            right: Vec2::new(1.0, 1.0),
        });
        mesh
    }

    #[test]
    fn test_vec2_basic() {
        let v = Vec2::new(3.0, 4.0);
        assert!((v.length() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_vec2_dist() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(3.0, 4.0);
        assert!((a.dist(b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_vec2_cross() {
        let a = Vec2::new(1.0, 0.0);
        let b = Vec2::new(0.0, 1.0);
        assert!((a.cross(b) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_poly_centroid() {
        let poly = NavPoly {
            id: 0,
            vertices: square_poly(0.0, 0.0, 2.0),
            surface: SurfaceType::Normal,
        };
        let c = poly.centroid();
        assert!((c.x - 1.0).abs() < 1e-10);
        assert!((c.y - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_poly_area() {
        let poly = NavPoly {
            id: 0,
            vertices: square_poly(0.0, 0.0, 2.0),
            surface: SurfaceType::Normal,
        };
        assert!((poly.area() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_poly_contains_inside() {
        let poly = NavPoly {
            id: 0,
            vertices: square_poly(0.0, 0.0, 2.0),
            surface: SurfaceType::Normal,
        };
        assert!(poly.contains(Vec2::new(1.0, 1.0)));
    }

    #[test]
    fn test_poly_contains_outside() {
        let poly = NavPoly {
            id: 0,
            vertices: square_poly(0.0, 0.0, 2.0),
            surface: SurfaceType::Normal,
        };
        assert!(!poly.contains(Vec2::new(3.0, 3.0)));
    }

    #[test]
    fn test_surface_costs() {
        assert!((surface_cost(SurfaceType::Normal) - 1.0).abs() < 1e-10);
        assert!((surface_cost(SurfaceType::Swamp) - 3.0).abs() < 1e-10);
        assert!((surface_cost(SurfaceType::Road) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_find_polygon() {
        let mesh = build_two_square_mesh();
        assert_eq!(mesh.find_polygon(Vec2::new(0.5, 0.5)), Some(0));
        assert_eq!(mesh.find_polygon(Vec2::new(1.5, 0.5)), Some(1));
        assert_eq!(mesh.find_polygon(Vec2::new(5.0, 5.0)), None);
    }

    #[test]
    fn test_nearest_polygon() {
        let mesh = build_two_square_mesh();
        // Point far right is closest to polygon 1
        assert_eq!(mesh.nearest_polygon(Vec2::new(3.0, 0.5)), Some(1));
    }

    #[test]
    fn test_poly_path_same_polygon() {
        let mesh = build_two_square_mesh();
        let path = mesh.find_poly_path(Vec2::new(0.2, 0.2), Vec2::new(0.8, 0.8));
        assert_eq!(path, Some(vec![0]));
    }

    #[test]
    fn test_poly_path_adjacent() {
        let mesh = build_two_square_mesh();
        let path = mesh.find_poly_path(Vec2::new(0.5, 0.5), Vec2::new(1.5, 0.5));
        assert_eq!(path, Some(vec![0, 1]));
    }

    #[test]
    fn test_poly_path_three_polys() {
        let mut mesh = NavMesh::new();
        mesh.add_polygon(square_poly(0.0, 0.0, 1.0), SurfaceType::Normal);
        mesh.add_polygon(square_poly(1.0, 0.0, 1.0), SurfaceType::Normal);
        mesh.add_polygon(square_poly(2.0, 0.0, 1.0), SurfaceType::Normal);
        mesh.connect(0, 1, Portal {
            left: Vec2::new(1.0, 0.0),
            right: Vec2::new(1.0, 1.0),
        });
        mesh.connect(1, 2, Portal {
            left: Vec2::new(2.0, 0.0),
            right: Vec2::new(2.0, 1.0),
        });
        let path = mesh.find_poly_path(Vec2::new(0.5, 0.5), Vec2::new(2.5, 0.5));
        assert_eq!(path, Some(vec![0, 1, 2]));
    }

    #[test]
    fn test_funnel_path_single_poly() {
        let mesh = build_two_square_mesh();
        let path = mesh.funnel_path(
            Vec2::new(0.2, 0.2),
            Vec2::new(0.8, 0.8),
            &[0],
        );
        assert_eq!(path.len(), 2);
    }

    #[test]
    fn test_full_pathfind() {
        let mesh = build_two_square_mesh();
        let path = mesh.find_path(Vec2::new(0.5, 0.5), Vec2::new(1.5, 0.5));
        assert!(path.is_some());
        let path = path.unwrap();
        assert!(path.len() >= 2);
        assert!((path[0].x - 0.5).abs() < 1e-10);
        assert!((path.last().unwrap().x - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_off_mesh_link() {
        let mut mesh = NavMesh::new();
        mesh.add_polygon(square_poly(0.0, 0.0, 1.0), SurfaceType::Normal);
        mesh.add_polygon(square_poly(5.0, 0.0, 1.0), SurfaceType::Normal);
        mesh.add_off_mesh_link(OffMeshLink {
            from_poly: 0,
            to_poly: 1,
            from_pos: Vec2::new(0.5, 0.5),
            to_pos: Vec2::new(5.5, 0.5),
            link_type: LinkType::Jump,
            cost: 2.0,
            bidirectional: false,
        });
        let neighbors = mesh.neighbors(0);
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].0, 1);
    }

    #[test]
    fn test_bidirectional_off_mesh_link() {
        let mut mesh = NavMesh::new();
        mesh.add_polygon(square_poly(0.0, 0.0, 1.0), SurfaceType::Normal);
        mesh.add_polygon(square_poly(5.0, 0.0, 1.0), SurfaceType::Normal);
        mesh.add_off_mesh_link(OffMeshLink {
            from_poly: 0,
            to_poly: 1,
            from_pos: Vec2::new(0.5, 0.5),
            to_pos: Vec2::new(5.5, 0.5),
            link_type: LinkType::Ladder,
            cost: 2.0,
            bidirectional: true,
        });
        let n0 = mesh.neighbors(0);
        let n1 = mesh.neighbors(1);
        assert!(n0.iter().any(|(id, _, _)| *id == 1));
        assert!(n1.iter().any(|(id, _, _)| *id == 0));
    }

    #[test]
    fn test_triangle_soup() {
        let triangles = vec![
            Triangle {
                v0: Vec2::new(0.0, 0.0),
                v1: Vec2::new(1.0, 0.0),
                v2: Vec2::new(0.5, 1.0),
            },
            Triangle {
                v0: Vec2::new(1.0, 0.0),
                v1: Vec2::new(0.5, 1.0),
                v2: Vec2::new(1.5, 1.0),
            },
        ];
        let mesh = navmesh_from_triangles(&triangles, SurfaceType::Normal);
        assert_eq!(mesh.polygons.len(), 2);
        let neighbors = mesh.neighbors(0);
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].0, 1);
    }

    #[test]
    fn test_swamp_higher_cost() {
        let mut mesh = NavMesh::new();
        mesh.add_polygon(square_poly(0.0, 0.0, 1.0), SurfaceType::Normal);
        mesh.add_polygon(square_poly(1.0, 0.0, 1.0), SurfaceType::Swamp);
        mesh.connect(0, 1, Portal {
            left: Vec2::new(1.0, 0.0),
            right: Vec2::new(1.0, 1.0),
        });
        let neighbors = mesh.neighbors(0);
        let swamp_cost = neighbors[0].1;
        // Swamp cost should be 3x the base distance
        assert!(swamp_cost > 1.0);
    }

    #[test]
    fn test_no_path_disconnected() {
        let mut mesh = NavMesh::new();
        mesh.add_polygon(square_poly(0.0, 0.0, 1.0), SurfaceType::Normal);
        mesh.add_polygon(square_poly(10.0, 10.0, 1.0), SurfaceType::Normal);
        // No connection
        let path = mesh.find_poly_path(Vec2::new(0.5, 0.5), Vec2::new(10.5, 10.5));
        assert!(path.is_none());
    }

    #[test]
    fn test_closest_point_on_boundary() {
        let poly = NavPoly {
            id: 0,
            vertices: square_poly(0.0, 0.0, 2.0),
            surface: SurfaceType::Normal,
        };
        let cp = poly.closest_point_on_boundary(Vec2::new(1.0, 3.0));
        assert!((cp.x - 1.0).abs() < 1e-6);
        assert!((cp.y - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_closest_point_on_segment() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(2.0, 0.0);
        let cp = closest_point_on_segment(Vec2::new(1.0, 1.0), a, b);
        assert!((cp.x - 1.0).abs() < 1e-6);
        assert!((cp.y - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_triangle_contains() {
        let poly = NavPoly {
            id: 0,
            vertices: vec![
                Vec2::new(0.0, 0.0),
                Vec2::new(2.0, 0.0),
                Vec2::new(1.0, 2.0),
            ],
            surface: SurfaceType::Normal,
        };
        assert!(poly.contains(Vec2::new(1.0, 0.5)));
        assert!(!poly.contains(Vec2::new(3.0, 3.0)));
    }

    #[test]
    fn test_empty_mesh() {
        let mesh = NavMesh::new();
        assert_eq!(mesh.find_polygon(Vec2::new(0.0, 0.0)), None);
        assert_eq!(mesh.nearest_polygon(Vec2::new(0.0, 0.0)), None);
    }

    #[test]
    fn test_link_types() {
        assert_eq!(LinkType::Jump, LinkType::Jump);
        assert_ne!(LinkType::Jump, LinkType::Ladder);
        assert_ne!(LinkType::Ladder, LinkType::Teleport);
    }
}
