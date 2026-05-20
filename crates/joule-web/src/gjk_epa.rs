//! GJK (Gilbert-Johnson-Keerthi) convex collision detection + EPA (Expanding
//! Polytope Algorithm) for penetration depth and contact normal. Support
//! functions for sphere, box, capsule, and convex hull. Configurable
//! iteration limits and epsilon.

// ── Vec3 ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };

    pub fn new(x: f64, y: f64, z: f64) -> Self { Self { x, y, z } }
    pub fn dot(self, r: Self) -> f64 { self.x * r.x + self.y * r.y + self.z * r.z }

    pub fn cross(self, r: Self) -> Self {
        Self {
            x: self.y * r.z - self.z * r.y,
            y: self.z * r.x - self.x * r.z,
            z: self.x * r.y - self.y * r.x,
        }
    }

    pub fn length_sq(self) -> f64 { self.dot(self) }
    pub fn length(self) -> f64 { self.length_sq().sqrt() }

    pub fn normalized(self) -> Self {
        let l = self.length();
        if l < 1e-12 { Self::ZERO } else { self * (1.0 / l) }
    }

    pub fn triple_product(a: Self, b: Self, c: Self) -> Self {
        // (A x B) x C = B*(C.A) - A*(C.B)
        b * c.dot(a) - a * c.dot(b)
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;
    fn add(self, r: Self) -> Self { Self { x: self.x + r.x, y: self.y + r.y, z: self.z + r.z } }
}
impl std::ops::Sub for Vec3 {
    type Output = Self;
    fn sub(self, r: Self) -> Self { Self { x: self.x - r.x, y: self.y - r.y, z: self.z - r.z } }
}
impl std::ops::Mul<f64> for Vec3 {
    type Output = Self;
    fn mul(self, s: f64) -> Self { Self { x: self.x * s, y: self.y * s, z: self.z * s } }
}
impl std::ops::Neg for Vec3 {
    type Output = Self;
    fn neg(self) -> Self { Self { x: -self.x, y: -self.y, z: -self.z } }
}

// ── Convex Shape Trait ───────────────────────────────────────

/// A convex shape that can provide a support point in a given direction.
pub trait ConvexSupport {
    /// Return the point on (or in) the shape that is farthest in direction `d`.
    fn support(&self, d: Vec3) -> Vec3;
}

// ── Built-in shapes ──────────────────────────────────────────

/// Sphere at center with radius.
#[derive(Debug, Clone, Copy)]
pub struct GjkSphere {
    pub center: Vec3,
    pub radius: f64,
}

impl ConvexSupport for GjkSphere {
    fn support(&self, d: Vec3) -> Vec3 {
        self.center + d.normalized() * self.radius
    }
}

/// Axis-aligned box defined by center and half-extents.
#[derive(Debug, Clone, Copy)]
pub struct GjkBox {
    pub center: Vec3,
    pub half_extents: Vec3,
}

impl ConvexSupport for GjkBox {
    fn support(&self, d: Vec3) -> Vec3 {
        Vec3::new(
            self.center.x + if d.x >= 0.0 { self.half_extents.x } else { -self.half_extents.x },
            self.center.y + if d.y >= 0.0 { self.half_extents.y } else { -self.half_extents.y },
            self.center.z + if d.z >= 0.0 { self.half_extents.z } else { -self.half_extents.z },
        )
    }
}

/// Capsule defined by two endpoints and radius.
#[derive(Debug, Clone, Copy)]
pub struct GjkCapsule {
    pub a: Vec3,
    pub b: Vec3,
    pub radius: f64,
}

impl ConvexSupport for GjkCapsule {
    fn support(&self, d: Vec3) -> Vec3 {
        let da = d.dot(self.a);
        let db = d.dot(self.b);
        let base = if da >= db { self.a } else { self.b };
        base + d.normalized() * self.radius
    }
}

/// Convex hull from a set of vertices.
#[derive(Debug, Clone)]
pub struct GjkConvexHull {
    pub vertices: Vec<Vec3>,
}

impl ConvexSupport for GjkConvexHull {
    fn support(&self, d: Vec3) -> Vec3 {
        let mut best = self.vertices[0];
        let mut best_dot = d.dot(best);
        for v in &self.vertices[1..] {
            let dd = d.dot(*v);
            if dd > best_dot {
                best_dot = dd;
                best = *v;
            }
        }
        best
    }
}

// ── GJK Configuration ────────────────────────────────────────

/// Configuration for GJK/EPA algorithms.
#[derive(Debug, Clone, Copy)]
pub struct GjkConfig {
    pub gjk_max_iterations: usize,
    pub epa_max_iterations: usize,
    pub epa_tolerance: f64,
}

impl Default for GjkConfig {
    fn default() -> Self {
        Self {
            gjk_max_iterations: 64,
            epa_max_iterations: 64,
            epa_tolerance: 1e-6,
        }
    }
}

// ── GJK Result ───────────────────────────────────────────────

/// Result of a GJK intersection test.
#[derive(Debug, Clone)]
pub struct GjkResult {
    pub intersecting: bool,
    pub simplex: Vec<Vec3>,
    pub iterations: usize,
}

/// Result from EPA: penetration depth and contact normal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpaResult {
    pub normal: Vec3,
    pub depth: f64,
}

// ── Minkowski support ────────────────────────────────────────

fn minkowski_support(a: &dyn ConvexSupport, b: &dyn ConvexSupport, d: Vec3) -> Vec3 {
    a.support(d) - b.support(-d)
}

// ── GJK ──────────────────────────────────────────────────────

/// Run GJK to test if two convex shapes intersect.
pub fn gjk_intersect(
    a: &dyn ConvexSupport,
    b: &dyn ConvexSupport,
    config: &GjkConfig,
) -> GjkResult {
    let mut d = Vec3::new(1.0, 0.0, 0.0);
    let mut simplex: Vec<Vec3> = Vec::new();

    let s = minkowski_support(a, b, d);
    simplex.push(s);
    d = -s;

    for iteration in 0..config.gjk_max_iterations {
        if d.length_sq() < 1e-20 {
            return GjkResult { intersecting: true, simplex, iterations: iteration };
        }
        let new_pt = minkowski_support(a, b, d);
        if new_pt.dot(d) < 0.0 {
            return GjkResult { intersecting: false, simplex, iterations: iteration };
        }
        simplex.push(new_pt);

        if do_simplex(&mut simplex, &mut d) {
            return GjkResult { intersecting: true, simplex, iterations: iteration };
        }
    }

    GjkResult { intersecting: false, simplex, iterations: config.gjk_max_iterations }
}

fn do_simplex(simplex: &mut Vec<Vec3>, d: &mut Vec3) -> bool {
    match simplex.len() {
        2 => line_case(simplex, d),
        3 => triangle_case(simplex, d),
        4 => tetrahedron_case(simplex, d),
        _ => false,
    }
}

fn line_case(simplex: &mut Vec<Vec3>, d: &mut Vec3) -> bool {
    let a = simplex[1]; // newest
    let b = simplex[0];
    let ab = b - a;
    let ao = -a;

    if ab.dot(ao) > 0.0 {
        *d = Vec3::triple_product(ab, ao, ab);
        if d.length_sq() < 1e-20 {
            // Degenerate: origin on the line
            *d = ab.cross(Vec3::new(1.0, 0.0, 0.0));
            if d.length_sq() < 1e-20 {
                *d = ab.cross(Vec3::new(0.0, 1.0, 0.0));
            }
        }
    } else {
        *simplex = vec![a];
        *d = ao;
    }
    false
}

fn triangle_case(simplex: &mut Vec<Vec3>, d: &mut Vec3) -> bool {
    let a = simplex[2]; // newest
    let b = simplex[1];
    let c = simplex[0];
    let ab = b - a;
    let ac = c - a;
    let ao = -a;
    let abc = ab.cross(ac);

    if abc.cross(ac).dot(ao) > 0.0 {
        if ac.dot(ao) > 0.0 {
            *simplex = vec![c, a];
            *d = Vec3::triple_product(ac, ao, ac);
        } else {
            *simplex = vec![b, a];
            return line_case(simplex, d);
        }
    } else if ab.cross(abc).dot(ao) > 0.0 {
        *simplex = vec![b, a];
        return line_case(simplex, d);
    } else {
        // Inside the triangle prism
        if abc.dot(ao) > 0.0 {
            *d = abc;
        } else {
            *simplex = vec![b, c, a];
            *d = -abc;
        }
    }
    false
}

fn tetrahedron_case(simplex: &mut Vec<Vec3>, d: &mut Vec3) -> bool {
    let a = simplex[3]; // newest
    let b = simplex[2];
    let c = simplex[1];
    let dd_pt = simplex[0];

    let ab = b - a;
    let ac = c - a;
    let ad = dd_pt - a;
    let ao = -a;

    let abc = ab.cross(ac);
    let acd = ac.cross(ad);
    let adb = ad.cross(ab);

    if abc.dot(ao) > 0.0 {
        *simplex = vec![c, b, a];
        return triangle_case(simplex, d);
    }
    if acd.dot(ao) > 0.0 {
        *simplex = vec![dd_pt, c, a];
        return triangle_case(simplex, d);
    }
    if adb.dot(ao) > 0.0 {
        *simplex = vec![b, dd_pt, a];
        return triangle_case(simplex, d);
    }

    true // origin is inside tetrahedron
}

// ── EPA ──────────────────────────────────────────────────────

/// EPA face (triangle with indices into vertex list + precomputed normal).
#[derive(Debug, Clone)]
struct EpaFace {
    a: usize,
    b: usize,
    c: usize,
    normal: Vec3,
    distance: f64,
}

/// Run EPA on the result of a successful GJK to find penetration depth.
pub fn epa_penetration(
    a: &dyn ConvexSupport,
    b: &dyn ConvexSupport,
    gjk_result: &GjkResult,
    config: &GjkConfig,
) -> Option<EpaResult> {
    if !gjk_result.intersecting || gjk_result.simplex.len() < 4 {
        return None;
    }

    let mut vertices: Vec<Vec3> = gjk_result.simplex.clone();
    let mut faces: Vec<EpaFace> = Vec::new();

    // Initial tetrahedron faces (ensure outward normals)
    let face_indices = [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]];
    for fi in &face_indices {
        let normal = compute_face_normal(&vertices, fi[0], fi[1], fi[2]);
        let distance = normal.dot(vertices[fi[0]]);
        if distance < 0.0 {
            // Flip winding
            faces.push(EpaFace {
                a: fi[0], b: fi[2], c: fi[1],
                normal: -normal, distance: -distance,
            });
        } else {
            faces.push(EpaFace {
                a: fi[0], b: fi[1], c: fi[2],
                normal, distance,
            });
        }
    }

    for _iteration in 0..config.epa_max_iterations {
        // Find face closest to origin
        let (closest_idx, closest_face) = match faces.iter().enumerate()
            .min_by(|a, b| a.1.distance.partial_cmp(&b.1.distance).unwrap_or(std::cmp::Ordering::Equal))
        {
            Some((i, f)) => (i, f.clone()),
            None => return None,
        };

        let support = minkowski_support(a, b, closest_face.normal);
        let dist = closest_face.normal.dot(support);

        if dist - closest_face.distance < config.epa_tolerance {
            return Some(EpaResult {
                normal: closest_face.normal,
                depth: closest_face.distance,
            });
        }

        // Find all faces that can "see" the new support point
        let new_vtx = vertices.len();
        vertices.push(support);

        let mut edges: Vec<(usize, usize)> = Vec::new();
        let mut i = 0;
        while i < faces.len() {
            let face = &faces[i];
            if face.normal.dot(support - vertices[face.a]) > 0.0 {
                // This face is visible — collect edges and remove
                add_edge(&mut edges, face.a, face.b);
                add_edge(&mut edges, face.b, face.c);
                add_edge(&mut edges, face.c, face.a);
                faces.swap_remove(i);
            } else {
                i += 1;
            }
        }

        // Create new faces from the horizon edges to the new point
        for &(ea, eb) in &edges {
            let normal = compute_face_normal(&vertices, ea, eb, new_vtx);
            let distance = normal.dot(vertices[ea]);
            if distance < 0.0 {
                faces.push(EpaFace {
                    a: ea, b: new_vtx, c: eb,
                    normal: -normal, distance: -distance,
                });
            } else {
                faces.push(EpaFace {
                    a: ea, b: eb, c: new_vtx,
                    normal, distance,
                });
            }
        }

        // Ignore the closest_idx variable since faces were modified
        let _ = closest_idx;
    }

    // Fallback: return best face found
    faces.iter()
        .min_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal))
        .map(|f| EpaResult { normal: f.normal, depth: f.distance })
}

fn compute_face_normal(vertices: &[Vec3], a: usize, b: usize, c: usize) -> Vec3 {
    let ab = vertices[b] - vertices[a];
    let ac = vertices[c] - vertices[a];
    ab.cross(ac).normalized()
}

fn add_edge(edges: &mut Vec<(usize, usize)>, a: usize, b: usize) {
    // If reverse edge exists, remove it (shared edge between two visible faces)
    if let Some(pos) = edges.iter().position(|&(ea, eb)| ea == b && eb == a) {
        edges.swap_remove(pos);
    } else {
        edges.push((a, b));
    }
}

// ── Convenience: full GJK+EPA ────────────────────────────────

/// Full collision test: GJK for intersection, then EPA for penetration info.
pub fn collision_test(
    a: &dyn ConvexSupport,
    b: &dyn ConvexSupport,
) -> Option<EpaResult> {
    let config = GjkConfig::default();
    let gjk = gjk_intersect(a, b, &config);
    if !gjk.intersecting {
        return None;
    }
    epa_penetration(a, b, &gjk, &config)
}

/// Closest distance between two non-intersecting convex shapes (GJK only).
pub fn gjk_distance(
    a: &dyn ConvexSupport,
    b: &dyn ConvexSupport,
) -> f64 {
    let config = GjkConfig::default();
    let gjk = gjk_intersect(a, b, &config);
    if gjk.intersecting {
        return 0.0;
    }
    // Approximate distance from the simplex
    match gjk.simplex.len() {
        0 => f64::MAX,
        1 => gjk.simplex[0].length(),
        2 => {
            let a_pt = gjk.simplex[0];
            let b_pt = gjk.simplex[1];
            closest_point_on_line_to_origin(a_pt, b_pt).length()
        }
        _ => {
            let a_pt = gjk.simplex[0];
            let b_pt = gjk.simplex[1];
            let c_pt = gjk.simplex[2];
            closest_point_on_triangle_to_origin(a_pt, b_pt, c_pt).length()
        }
    }
}

fn closest_point_on_line_to_origin(a: Vec3, b: Vec3) -> Vec3 {
    let ab = b - a;
    let t = (-a).dot(ab) / ab.length_sq();
    let t = t.clamp(0.0, 1.0);
    a + ab * t
}

fn closest_point_on_triangle_to_origin(a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    let ab = b - a;
    let ac = c - a;
    let ao = -a;

    let d1 = ab.dot(ao);
    let d2 = ac.dot(ao);
    if d1 <= 0.0 && d2 <= 0.0 { return a; }

    let bo = -b;
    let d3 = ab.dot(bo);
    let d4 = ac.dot(bo);
    if d3 >= 0.0 && d4 <= d3 { return b; }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return a + ab * v;
    }

    let co = -c;
    let d5 = ab.dot(co);
    let d6 = ac.dot(co);
    if d6 >= 0.0 && d5 <= d6 { return c; }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return a + ac * w;
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + (c - b) * w;
    }

    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    a + ab * v + ac * w
}

// ══════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-4;

    fn approx(a: f64, b: f64) -> bool { (a - b).abs() < EPS }
    fn v3_approx(a: Vec3, b: Vec3) -> bool { approx(a.x, b.x) && approx(a.y, b.y) && approx(a.z, b.z) }

    #[test]
    fn test_sphere_support() {
        let s = GjkSphere { center: Vec3::ZERO, radius: 2.0 };
        let sup = s.support(Vec3::new(1.0, 0.0, 0.0));
        assert!(v3_approx(sup, Vec3::new(2.0, 0.0, 0.0)));
    }

    #[test]
    fn test_box_support() {
        let b = GjkBox { center: Vec3::ZERO, half_extents: Vec3::new(1.0, 2.0, 3.0) };
        let sup = b.support(Vec3::new(1.0, 1.0, 1.0));
        assert!(v3_approx(sup, Vec3::new(1.0, 2.0, 3.0)));
    }

    #[test]
    fn test_box_support_negative() {
        let b = GjkBox { center: Vec3::ZERO, half_extents: Vec3::new(1.0, 2.0, 3.0) };
        let sup = b.support(Vec3::new(-1.0, -1.0, -1.0));
        assert!(v3_approx(sup, Vec3::new(-1.0, -2.0, -3.0)));
    }

    #[test]
    fn test_capsule_support() {
        let c = GjkCapsule {
            a: Vec3::new(0.0, -1.0, 0.0),
            b: Vec3::new(0.0, 1.0, 0.0),
            radius: 0.5,
        };
        let sup = c.support(Vec3::new(0.0, 1.0, 0.0));
        assert!(v3_approx(sup, Vec3::new(0.0, 1.5, 0.0)));
    }

    #[test]
    fn test_convex_hull_support() {
        let hull = GjkConvexHull {
            vertices: vec![
                Vec3::new(-1.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
            ],
        };
        let sup = hull.support(Vec3::new(1.0, 0.0, 0.0));
        assert!(v3_approx(sup, Vec3::new(1.0, 0.0, 0.0)));
    }

    #[test]
    fn test_gjk_spheres_intersecting() {
        let a = GjkSphere { center: Vec3::ZERO, radius: 1.0 };
        let b = GjkSphere { center: Vec3::new(1.0, 0.0, 0.0), radius: 1.0 };
        let config = GjkConfig::default();
        let result = gjk_intersect(&a, &b, &config);
        assert!(result.intersecting);
    }

    #[test]
    fn test_gjk_spheres_separated() {
        let a = GjkSphere { center: Vec3::ZERO, radius: 1.0 };
        let b = GjkSphere { center: Vec3::new(5.0, 0.0, 0.0), radius: 1.0 };
        let config = GjkConfig::default();
        let result = gjk_intersect(&a, &b, &config);
        assert!(!result.intersecting);
    }

    #[test]
    fn test_gjk_boxes_intersecting() {
        let a = GjkBox { center: Vec3::ZERO, half_extents: Vec3::new(1.0, 1.0, 1.0) };
        let b = GjkBox { center: Vec3::new(1.5, 0.0, 0.0), half_extents: Vec3::new(1.0, 1.0, 1.0) };
        let config = GjkConfig::default();
        let result = gjk_intersect(&a, &b, &config);
        assert!(result.intersecting);
    }

    #[test]
    fn test_gjk_boxes_separated() {
        let a = GjkBox { center: Vec3::ZERO, half_extents: Vec3::new(1.0, 1.0, 1.0) };
        let b = GjkBox { center: Vec3::new(5.0, 0.0, 0.0), half_extents: Vec3::new(1.0, 1.0, 1.0) };
        let config = GjkConfig::default();
        let result = gjk_intersect(&a, &b, &config);
        assert!(!result.intersecting);
    }

    #[test]
    fn test_gjk_sphere_box_intersect() {
        let a = GjkSphere { center: Vec3::ZERO, radius: 1.0 };
        let b = GjkBox { center: Vec3::new(0.5, 0.0, 0.0), half_extents: Vec3::new(1.0, 1.0, 1.0) };
        let config = GjkConfig::default();
        let result = gjk_intersect(&a, &b, &config);
        assert!(result.intersecting);
    }

    #[test]
    fn test_gjk_capsule_capsule() {
        let a = GjkCapsule { a: Vec3::new(0.0, -1.0, 0.0), b: Vec3::new(0.0, 1.0, 0.0), radius: 0.5 };
        let b = GjkCapsule { a: Vec3::new(0.8, -1.0, 0.0), b: Vec3::new(0.8, 1.0, 0.0), radius: 0.5 };
        let config = GjkConfig::default();
        let result = gjk_intersect(&a, &b, &config);
        assert!(result.intersecting);
    }

    #[test]
    fn test_epa_spheres() {
        let a = GjkSphere { center: Vec3::ZERO, radius: 1.0 };
        let b = GjkSphere { center: Vec3::new(1.0, 0.0, 0.0), radius: 1.0 };
        let result = collision_test(&a, &b);
        assert!(result.is_some());
        let epa = result.unwrap();
        assert!(epa.depth > 0.0);
        // Penetration depth should be ~1.0 (2 radii - 1.0 distance)
        assert!(approx(epa.depth, 1.0));
    }

    #[test]
    fn test_epa_boxes() {
        let a = GjkBox { center: Vec3::ZERO, half_extents: Vec3::new(1.0, 1.0, 1.0) };
        let b = GjkBox { center: Vec3::new(1.5, 0.0, 0.0), half_extents: Vec3::new(1.0, 1.0, 1.0) };
        let result = collision_test(&a, &b);
        assert!(result.is_some());
        let epa = result.unwrap();
        assert!(epa.depth > 0.0);
        // Overlap = 2.0 - 1.5 = 0.5
        assert!(approx(epa.depth, 0.5));
    }

    #[test]
    fn test_gjk_distance_separated_spheres() {
        let a = GjkSphere { center: Vec3::ZERO, radius: 1.0 };
        let b = GjkSphere { center: Vec3::new(5.0, 0.0, 0.0), radius: 1.0 };
        let d = gjk_distance(&a, &b);
        // Distance should be approximately 3.0 (5.0 - 1.0 - 1.0)
        assert!(d > 2.0);
    }

    #[test]
    fn test_gjk_distance_intersecting() {
        let a = GjkSphere { center: Vec3::ZERO, radius: 2.0 };
        let b = GjkSphere { center: Vec3::new(1.0, 0.0, 0.0), radius: 2.0 };
        let d = gjk_distance(&a, &b);
        assert!(approx(d, 0.0));
    }

    #[test]
    fn test_collision_test_no_collision() {
        let a = GjkSphere { center: Vec3::ZERO, radius: 1.0 };
        let b = GjkSphere { center: Vec3::new(10.0, 0.0, 0.0), radius: 1.0 };
        let result = collision_test(&a, &b);
        assert!(result.is_none());
    }

    #[test]
    fn test_triple_product() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        let c = Vec3::new(0.0, 0.0, 1.0);
        let tp = Vec3::triple_product(a, b, c);
        // (a x b) x c = b*(c.a) - a*(c.b) = b*0 - a*0 = 0
        assert!(v3_approx(tp, Vec3::ZERO));
    }

    #[test]
    fn test_minkowski_support() {
        let a = GjkSphere { center: Vec3::new(1.0, 0.0, 0.0), radius: 1.0 };
        let b = GjkSphere { center: Vec3::new(-1.0, 0.0, 0.0), radius: 1.0 };
        let sup = minkowski_support(&a, &b, Vec3::new(1.0, 0.0, 0.0));
        // a.support(+x) = (2,0,0), b.support(-x) = (-2,0,0), diff = (4,0,0)
        assert!(v3_approx(sup, Vec3::new(4.0, 0.0, 0.0)));
    }

    #[test]
    fn test_gjk_config_default() {
        let c = GjkConfig::default();
        assert_eq!(c.gjk_max_iterations, 64);
        assert_eq!(c.epa_max_iterations, 64);
        assert!(approx(c.epa_tolerance, 1e-6));
    }

    #[test]
    fn test_gjk_hull_vs_hull() {
        let a = GjkConvexHull {
            vertices: vec![
                Vec3::new(-1.0, -1.0, -1.0),
                Vec3::new(1.0, -1.0, -1.0),
                Vec3::new(0.0, 1.0, -1.0),
                Vec3::new(0.0, 0.0, 1.0),
            ],
        };
        let b = GjkConvexHull {
            vertices: vec![
                Vec3::new(-0.5, -0.5, -0.5),
                Vec3::new(0.5, -0.5, -0.5),
                Vec3::new(0.0, 0.5, -0.5),
                Vec3::new(0.0, 0.0, 0.5),
            ],
        };
        let config = GjkConfig::default();
        let result = gjk_intersect(&a, &b, &config);
        assert!(result.intersecting);
    }

    #[test]
    fn test_closest_point_on_line_to_origin_basic() {
        let cp = closest_point_on_line_to_origin(
            Vec3::new(-1.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
        );
        // Closest point to origin on the line y=1 between x=-1 and x=1 is (0,1,0)
        assert!(v3_approx(cp, Vec3::new(0.0, 1.0, 0.0)));
    }

    #[test]
    fn test_epa_normal_direction() {
        let a = GjkSphere { center: Vec3::ZERO, radius: 1.0 };
        let b = GjkSphere { center: Vec3::new(0.5, 0.0, 0.0), radius: 1.0 };
        let result = collision_test(&a, &b);
        assert!(result.is_some());
        let epa = result.unwrap();
        // Normal should be roughly along X axis
        assert!(epa.normal.x.abs() > 0.5);
    }
}
