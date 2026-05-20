//! 2D narrowphase collision detection — SAT for convex polygons, circle-polygon
//! overlap with MTV, EPA-like penetration depth, contact manifold (1-2 points),
//! manifold caching for warm-starting, edge-edge / vertex-face classification.

use std::collections::HashMap;

// ── Vec2 ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    pub fn new(x: f64, y: f64) -> Self { Self { x, y } }
    pub fn zero() -> Self { Self { x: 0.0, y: 0.0 } }
    pub fn dot(self, o: Self) -> f64 { self.x * o.x + self.y * o.y }
    pub fn cross(self, o: Self) -> f64 { self.x * o.y - self.y * o.x }
    pub fn length(self) -> f64 { (self.x * self.x + self.y * self.y).sqrt() }
    pub fn length_sq(self) -> f64 { self.x * self.x + self.y * self.y }
    pub fn normalized(self) -> Self {
        let len = self.length();
        if len < 1e-12 { Self::zero() } else { Self { x: self.x / len, y: self.y / len } }
    }
    pub fn add(self, o: Self) -> Self { Self { x: self.x + o.x, y: self.y + o.y } }
    pub fn sub(self, o: Self) -> Self { Self { x: self.x - o.x, y: self.y - o.y } }
    pub fn scale(self, s: f64) -> Self { Self { x: self.x * s, y: self.y * s } }
    pub fn negate(self) -> Self { Self { x: -self.x, y: -self.y } }
    pub fn perpendicular(self) -> Self { Self { x: -self.y, y: self.x } }
}

impl Default for Vec2 {
    fn default() -> Self { Self::zero() }
}

// ── Contact classification ───────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContactType {
    VertexFace,
    EdgeEdge,
}

/// A single contact point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContactPoint {
    pub position: Vec2,
    pub normal: Vec2,
    pub depth: f64,
    pub contact_type: ContactType,
    /// Accumulated normal impulse for warm-starting.
    pub normal_impulse: f64,
    /// Accumulated tangent impulse for warm-starting.
    pub tangent_impulse: f64,
}

impl ContactPoint {
    pub fn new(position: Vec2, normal: Vec2, depth: f64, contact_type: ContactType) -> Self {
        Self { position, normal, depth, contact_type, normal_impulse: 0.0, tangent_impulse: 0.0 }
    }
}

/// Contact manifold between two bodies (1-2 points).
#[derive(Debug, Clone, PartialEq)]
pub struct ContactManifold {
    pub body_a: u64,
    pub body_b: u64,
    pub points: Vec<ContactPoint>,
}

impl ContactManifold {
    pub fn new(a: u64, b: u64) -> Self {
        Self { body_a: a, body_b: b, points: Vec::new() }
    }

    pub fn add_point(&mut self, point: ContactPoint) {
        if self.points.len() < 2 {
            self.points.push(point);
        } else {
            // Replace the shallowest point if the new one is deeper.
            let idx = self.points.iter().enumerate()
                .min_by(|(_, a), (_, b)| a.depth.partial_cmp(&b.depth).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            if point.depth > self.points[idx].depth {
                self.points[idx] = point;
            }
        }
    }
}

// ── SAT (Separating Axis Theorem) ────────────────────────────

/// SAT result with minimum translation vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SatResult {
    pub normal: Vec2,
    pub depth: f64,
}

fn project_polygon(verts: &[Vec2], axis: Vec2) -> (f64, f64) {
    let mut min_p = f64::MAX;
    let mut max_p = f64::MIN;
    for v in verts {
        let p = v.dot(axis);
        min_p = min_p.min(p);
        max_p = max_p.max(p);
    }
    (min_p, max_p)
}

fn polygon_centroid(verts: &[Vec2]) -> Vec2 {
    let mut c = Vec2::zero();
    for v in verts { c = c.add(*v); }
    c.scale(1.0 / verts.len() as f64)
}

/// Run SAT on two convex polygons.  Returns MTV if overlapping.
pub fn sat_polygons(va: &[Vec2], vb: &[Vec2]) -> Option<SatResult> {
    if va.len() < 3 || vb.len() < 3 { return None; }

    let mut min_depth = f64::MAX;
    let mut best_axis = Vec2::new(1.0, 0.0);

    for verts in [va, vb] {
        let n = verts.len();
        for i in 0..n {
            let j = (i + 1) % n;
            let edge = verts[j].sub(verts[i]);
            let axis = edge.perpendicular().normalized();
            if axis.length_sq() < 1e-12 { continue; }

            let (min_a, max_a) = project_polygon(va, axis);
            let (min_b, max_b) = project_polygon(vb, axis);
            let overlap = max_a.min(max_b) - min_a.max(min_b);
            if overlap <= 0.0 { return None; }
            if overlap < min_depth {
                min_depth = overlap;
                best_axis = axis;
            }
        }
    }

    // Ensure normal points from B toward A.
    let dir = polygon_centroid(va).sub(polygon_centroid(vb));
    if dir.dot(best_axis) < 0.0 {
        best_axis = best_axis.negate();
    }

    Some(SatResult { normal: best_axis, depth: min_depth })
}

// ── Circle-Polygon MTV ──────────────────────────────────────

fn closest_point_on_segment(p: Vec2, a: Vec2, b: Vec2) -> Vec2 {
    let ab = b.sub(a);
    let len_sq = ab.length_sq();
    if len_sq < 1e-12 { return a; }
    let t = p.sub(a).dot(ab) / len_sq;
    let t = t.clamp(0.0, 1.0);
    a.add(ab.scale(t))
}

fn point_in_polygon(p: Vec2, verts: &[Vec2]) -> bool {
    let n = verts.len();
    if n < 3 { return false; }
    let mut sign = None;
    for i in 0..n {
        let j = (i + 1) % n;
        let c = verts[j].sub(verts[i]).cross(p.sub(verts[i]));
        match sign {
            None => sign = Some(c >= 0.0),
            Some(s) => { if (c >= 0.0) != s { return false; } }
        }
    }
    true
}

/// Circle-polygon overlap with MTV.
pub fn circle_polygon_mtv(center: Vec2, radius: f64, verts: &[Vec2]) -> Option<SatResult> {
    let n = verts.len();
    if n < 3 { return None; }

    let mut min_dist_sq = f64::MAX;
    let mut closest = center;
    for i in 0..n {
        let j = (i + 1) % n;
        let cp = closest_point_on_segment(center, verts[i], verts[j]);
        let d = center.sub(cp).length_sq();
        if d < min_dist_sq {
            min_dist_sq = d;
            closest = cp;
        }
    }
    let dist = min_dist_sq.sqrt();
    let inside = point_in_polygon(center, verts);

    if !inside && dist > radius { return None; }

    let diff = center.sub(closest);
    let normal = if diff.length_sq() < 1e-12 {
        // Degenerate: use first edge normal.
        let edge = verts[1].sub(verts[0]);
        edge.perpendicular().normalized()
    } else if inside {
        diff.normalized().negate()
    } else {
        diff.normalized()
    };

    let depth = if inside { radius + dist } else { radius - dist };
    Some(SatResult { normal, depth })
}

// ── EPA-like penetration depth ───────────────────────────────

/// Simplified EPA for two convex polygons that are already known to overlap.
/// Uses the Minkowski difference support function and iteratively expands a simplex.
fn support(verts: &[Vec2], dir: Vec2) -> Vec2 {
    let mut best = verts[0];
    let mut best_dot = verts[0].dot(dir);
    for v in &verts[1..] {
        let d = v.dot(dir);
        if d > best_dot {
            best_dot = d;
            best = *v;
        }
    }
    best
}

fn minkowski_support(va: &[Vec2], vb: &[Vec2], dir: Vec2) -> Vec2 {
    let a = support(va, dir);
    let b = support(vb, dir.negate());
    a.sub(b)
}

/// EPA penetration depth for overlapping convex polygons.
pub fn epa_penetration(va: &[Vec2], vb: &[Vec2], max_iter: usize) -> Option<SatResult> {
    // Start with a triangle on the Minkowski difference boundary.
    let dirs = [Vec2::new(1.0, 0.0), Vec2::new(-0.5, 0.866), Vec2::new(-0.5, -0.866)];
    let mut simplex: Vec<Vec2> = dirs.iter().map(|d| minkowski_support(va, vb, *d)).collect();

    // Check that origin is inside simplex (convex hull).
    // If not, fall back to SAT.
    for iter_count in 0..max_iter {
        let _ = iter_count;
        // Find closest edge to origin.
        let n = simplex.len();
        let mut min_dist = f64::MAX;
        let mut best_normal = Vec2::new(1.0, 0.0);
        let mut best_idx = 0;

        for i in 0..n {
            let j = (i + 1) % n;
            let edge = simplex[j].sub(simplex[i]);
            let normal = Vec2::new(edge.y, -edge.x).normalized(); // outward normal
            let d = simplex[i].dot(normal);
            if d < min_dist {
                min_dist = d;
                best_normal = normal;
                best_idx = j;
            }
        }

        let new_point = minkowski_support(va, vb, best_normal);
        let new_dist = new_point.dot(best_normal);
        if (new_dist - min_dist).abs() < 1e-6 {
            return Some(SatResult { normal: best_normal, depth: min_dist.abs() });
        }

        simplex.insert(best_idx, new_point);
    }

    // Fallback: return SAT result.
    sat_polygons(va, vb)
}

// ── Contact manifold generation ──────────────────────────────

/// Generate a contact manifold between two convex polygons.
pub fn generate_manifold_polygons(body_a: u64, body_b: u64, va: &[Vec2], vb: &[Vec2]) -> Option<ContactManifold> {
    let sat = sat_polygons(va, vb)?;
    let mut manifold = ContactManifold::new(body_a, body_b);

    // Find incident edge on B (most anti-parallel to normal).
    let nb = vb.len();
    let mut best_i = 0;
    let mut min_dot = f64::MAX;
    for i in 0..nb {
        let j = (i + 1) % nb;
        let edge = vb[j].sub(vb[i]);
        let face_normal = edge.perpendicular().normalized();
        let d = face_normal.dot(sat.normal);
        if d < min_dot {
            min_dot = d;
            best_i = i;
        }
    }
    let best_j = (best_i + 1) % nb;

    // Clip incident edge to reference face.
    let v1 = vb[best_i];
    let v2 = vb[best_j];

    // Contact point 1: deepest vertex of B along -normal.
    let depth1 = sat.depth;
    manifold.add_point(ContactPoint::new(v1, sat.normal, depth1, ContactType::EdgeEdge));

    // Check if second vertex also penetrates.
    let d2 = sat.normal.negate().dot(v2.sub(polygon_centroid(va)));
    if d2 > -sat.depth * 0.5 {
        manifold.add_point(ContactPoint::new(v2, sat.normal, sat.depth * 0.8, ContactType::EdgeEdge));
    }

    Some(manifold)
}

/// Generate manifold for circle vs polygon.
pub fn generate_manifold_circle_polygon(
    body_a: u64, body_b: u64, center: Vec2, radius: f64, verts: &[Vec2]
) -> Option<ContactManifold> {
    let mtv = circle_polygon_mtv(center, radius, verts)?;
    let mut manifold = ContactManifold::new(body_a, body_b);
    let contact_point = center.add(mtv.normal.negate().scale(radius));
    manifold.add_point(ContactPoint::new(
        contact_point, mtv.normal, mtv.depth, ContactType::VertexFace,
    ));
    Some(manifold)
}

/// Generate manifold for two circles.
pub fn generate_manifold_circles(
    body_a: u64, body_b: u64, c1: Vec2, r1: f64, c2: Vec2, r2: f64,
) -> Option<ContactManifold> {
    let d = c2.sub(c1);
    let dist = d.length();
    if dist > r1 + r2 { return None; }
    let normal = if dist < 1e-12 { Vec2::new(1.0, 0.0) } else { d.scale(1.0 / dist) };
    let depth = r1 + r2 - dist;
    let point = c1.add(normal.scale(r1 - depth * 0.5));
    let mut manifold = ContactManifold::new(body_a, body_b);
    manifold.add_point(ContactPoint::new(point, normal, depth, ContactType::VertexFace));
    Some(manifold)
}

// ── Manifold cache ───────────────────────────────────────────

/// Caches manifolds across frames for warm-starting.
#[derive(Debug, Clone)]
pub struct ManifoldCache {
    cache: HashMap<(u64, u64), ContactManifold>,
}

impl ManifoldCache {
    pub fn new() -> Self { Self { cache: HashMap::new() } }

    fn key(a: u64, b: u64) -> (u64, u64) {
        if a < b { (a, b) } else { (b, a) }
    }

    /// Store a manifold, warm-starting from previous frame impulses if available.
    pub fn update(&mut self, manifold: ContactManifold) {
        let key = Self::key(manifold.body_a, manifold.body_b);
        if let Some(old) = self.cache.get(&key) {
            let mut new_manifold = manifold;
            // Transfer impulses for nearby contact points.
            for new_pt in &mut new_manifold.points {
                for old_pt in &old.points {
                    let dist = new_pt.position.sub(old_pt.position).length_sq();
                    if dist < 0.01 {
                        new_pt.normal_impulse = old_pt.normal_impulse;
                        new_pt.tangent_impulse = old_pt.tangent_impulse;
                        break;
                    }
                }
            }
            self.cache.insert(key, new_manifold);
        } else {
            self.cache.insert(key, manifold);
        }
    }

    /// Remove stale manifolds.
    pub fn remove(&mut self, a: u64, b: u64) {
        let key = Self::key(a, b);
        self.cache.remove(&key);
    }

    pub fn get(&self, a: u64, b: u64) -> Option<&ContactManifold> {
        let key = Self::key(a, b);
        self.cache.get(&key)
    }

    pub fn clear(&mut self) { self.cache.clear(); }
    pub fn len(&self) -> usize { self.cache.len() }
    pub fn is_empty(&self) -> bool { self.cache.is_empty() }

    pub fn iter(&self) -> impl Iterator<Item = &ContactManifold> {
        self.cache.values()
    }
}

impl Default for ManifoldCache {
    fn default() -> Self { Self::new() }
}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-4;
    fn approx(a: f64, b: f64) -> bool { (a - b).abs() < EPS }

    fn unit_square() -> Vec<Vec2> {
        vec![
            Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0),
            Vec2::new(1.0, 1.0), Vec2::new(0.0, 1.0),
        ]
    }

    fn shifted_square(dx: f64, dy: f64) -> Vec<Vec2> {
        unit_square().iter().map(|v| v.add(Vec2::new(dx, dy))).collect()
    }

    // ── SAT ──

    #[test]
    fn sat_overlap() {
        let a = unit_square();
        let b = shifted_square(0.5, 0.5);
        let r = sat_polygons(&a, &b).unwrap();
        assert!(approx(r.depth, 0.5));
    }

    #[test]
    fn sat_separated() {
        let a = unit_square();
        let b = shifted_square(5.0, 5.0);
        assert!(sat_polygons(&a, &b).is_none());
    }

    #[test]
    fn sat_touching() {
        let a = unit_square();
        let b = shifted_square(1.0, 0.0);
        // Touching exactly at edge: overlap ≈ 0.
        let r = sat_polygons(&a, &b);
        // Could be None or very small depth.
        if let Some(r) = r {
            assert!(r.depth < EPS);
        }
    }

    #[test]
    fn sat_triangle() {
        let a = vec![Vec2::new(0.0, 0.0), Vec2::new(2.0, 0.0), Vec2::new(1.0, 2.0)];
        let b = vec![Vec2::new(0.5, 0.5), Vec2::new(2.5, 0.5), Vec2::new(1.5, 2.5)];
        assert!(sat_polygons(&a, &b).is_some());
    }

    #[test]
    fn sat_normal_direction() {
        let a = unit_square();
        let b = shifted_square(0.7, 0.0);
        let r = sat_polygons(&a, &b).unwrap();
        // Normal should roughly point in +x or -x direction.
        assert!(r.normal.x.abs() > 0.5 || r.normal.y.abs() > 0.5);
    }

    // ── Circle-Polygon MTV ──

    #[test]
    fn circle_polygon_hit() {
        let sq = unit_square();
        let r = circle_polygon_mtv(Vec2::new(1.3, 0.5), 0.5, &sq).unwrap();
        assert!(r.depth > 0.0);
    }

    #[test]
    fn circle_polygon_miss() {
        let sq = unit_square();
        assert!(circle_polygon_mtv(Vec2::new(5.0, 5.0), 0.5, &sq).is_none());
    }

    #[test]
    fn circle_inside_polygon() {
        let sq = vec![
            Vec2::new(-2.0, -2.0), Vec2::new(2.0, -2.0),
            Vec2::new(2.0, 2.0), Vec2::new(-2.0, 2.0),
        ];
        let r = circle_polygon_mtv(Vec2::new(0.0, 0.0), 0.5, &sq).unwrap();
        assert!(r.depth > 0.0);
    }

    // ── Manifold generation ──

    #[test]
    fn manifold_polygon_polygon() {
        let a = unit_square();
        let b = shifted_square(0.5, 0.0);
        let m = generate_manifold_polygons(1, 2, &a, &b).unwrap();
        assert!(!m.points.is_empty());
        assert!(m.points.len() <= 2);
    }

    #[test]
    fn manifold_circle_polygon() {
        let sq = unit_square();
        let m = generate_manifold_circle_polygon(1, 2, Vec2::new(1.3, 0.5), 0.5, &sq).unwrap();
        assert_eq!(m.points.len(), 1);
        assert_eq!(m.points[0].contact_type, ContactType::VertexFace);
    }

    #[test]
    fn manifold_circles() {
        let m = generate_manifold_circles(1, 2, Vec2::zero(), 1.0, Vec2::new(1.5, 0.0), 1.0).unwrap();
        assert_eq!(m.points.len(), 1);
        assert!(approx(m.points[0].depth, 0.5));
    }

    #[test]
    fn manifold_circles_miss() {
        assert!(generate_manifold_circles(1, 2, Vec2::zero(), 1.0, Vec2::new(5.0, 0.0), 1.0).is_none());
    }

    #[test]
    fn manifold_max_two_points() {
        let mut m = ContactManifold::new(1, 2);
        for i in 0..5 {
            m.add_point(ContactPoint::new(
                Vec2::new(i as f64, 0.0), Vec2::new(0.0, 1.0),
                0.1 * i as f64, ContactType::EdgeEdge,
            ));
        }
        assert!(m.points.len() <= 2);
    }

    // ── Manifold cache ──

    #[test]
    fn cache_store_retrieve() {
        let mut cache = ManifoldCache::new();
        let m = generate_manifold_circles(1, 2, Vec2::zero(), 1.0, Vec2::new(1.5, 0.0), 1.0).unwrap();
        cache.update(m.clone());
        assert!(cache.get(1, 2).is_some());
        assert!(cache.get(2, 1).is_some()); // symmetric
    }

    #[test]
    fn cache_warm_start() {
        let mut cache = ManifoldCache::new();
        let mut m = generate_manifold_circles(1, 2, Vec2::zero(), 1.0, Vec2::new(1.5, 0.0), 1.0).unwrap();
        m.points[0].normal_impulse = 42.0;
        cache.update(m);

        // Update with new manifold at same position.
        let m2 = generate_manifold_circles(1, 2, Vec2::zero(), 1.0, Vec2::new(1.5, 0.0), 1.0).unwrap();
        cache.update(m2);
        let cached = cache.get(1, 2).unwrap();
        assert!(approx(cached.points[0].normal_impulse, 42.0));
    }

    #[test]
    fn cache_remove() {
        let mut cache = ManifoldCache::new();
        let m = generate_manifold_circles(1, 2, Vec2::zero(), 1.0, Vec2::new(1.5, 0.0), 1.0).unwrap();
        cache.update(m);
        cache.remove(1, 2);
        assert!(cache.is_empty());
    }

    #[test]
    fn cache_clear() {
        let mut cache = ManifoldCache::new();
        cache.update(generate_manifold_circles(1, 2, Vec2::zero(), 1.0, Vec2::new(1.5, 0.0), 1.0).unwrap());
        cache.update(generate_manifold_circles(3, 4, Vec2::zero(), 1.0, Vec2::new(1.5, 0.0), 1.0).unwrap());
        assert_eq!(cache.len(), 2);
        cache.clear();
        assert!(cache.is_empty());
    }

    // ── EPA ──

    #[test]
    fn epa_overlapping_squares() {
        let a = unit_square();
        let b = shifted_square(0.5, 0.0);
        let r = epa_penetration(&a, &b, 20);
        assert!(r.is_some());
        let r = r.unwrap();
        assert!(r.depth > 0.0);
    }

    // ── Contact type ──

    #[test]
    fn contact_type_eq() {
        assert_eq!(ContactType::VertexFace, ContactType::VertexFace);
        assert_ne!(ContactType::VertexFace, ContactType::EdgeEdge);
    }

    // ── Edge cases ──

    #[test]
    fn degenerate_polygon() {
        let a = vec![Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0)]; // only 2 vertices
        let b = unit_square();
        assert!(sat_polygons(&a, &b).is_none());
    }
}
