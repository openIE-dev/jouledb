// Signed Distance 2D — SDF primitives for UI shapes, CSG operations, rendering

use std::f32::consts::PI;

/// 2D point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };

    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub fn length(self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y
    }

    pub fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }

    pub fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }

    pub fn scale(self, s: f32) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
        }
    }

    pub fn abs(self) -> Self {
        Self {
            x: self.x.abs(),
            y: self.y.abs(),
        }
    }
}

// --- SDF Primitives ---

/// SDF of a circle centered at origin.
pub fn sdf_circle(p: Vec2, radius: f32) -> f32 {
    p.length() - radius
}

/// SDF of an axis-aligned box centered at origin with half-extents.
pub fn sdf_box(p: Vec2, half_extents: Vec2) -> f32 {
    let d = p.abs().sub(half_extents);
    let outside = Vec2::new(d.x.max(0.0), d.y.max(0.0)).length();
    let inside = d.x.max(d.y).min(0.0);
    outside + inside
}

/// SDF of a rounded rectangle (box with corner radius).
pub fn sdf_rounded_rect(p: Vec2, half_extents: Vec2, radius: f32) -> f32 {
    let r = radius.min(half_extents.x).min(half_extents.y);
    let shrunk = Vec2::new(half_extents.x - r, half_extents.y - r);
    sdf_box(p, shrunk) - r
}

/// SDF of a line segment from `a` to `b`.
pub fn sdf_segment(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let pa = p.sub(a);
    let ba = b.sub(a);
    let h = (pa.dot(ba) / ba.dot(ba)).clamp(0.0, 1.0);
    let closest = a.add(ba.scale(h));
    p.sub(closest).length()
}

/// SDF of a triangle with vertices a, b, c.
pub fn sdf_triangle(p: Vec2, a: Vec2, b: Vec2, c: Vec2) -> f32 {
    let edges = [(a, b), (b, c), (c, a)];
    let mut min_dist = f32::MAX;
    let mut sign = 1.0f32;

    for (e0, e1) in &edges {
        let d = sdf_segment(p, *e0, *e1);
        if d < min_dist {
            min_dist = d;
        }
    }

    // Winding test for inside/outside
    let cross = |a_pt: Vec2, b_pt: Vec2, pt: Vec2| -> f32 {
        (b_pt.x - a_pt.x) * (pt.y - a_pt.y) - (b_pt.y - a_pt.y) * (pt.x - a_pt.x)
    };

    let c1 = cross(a, b, p);
    let c2 = cross(b, c, p);
    let c3 = cross(c, a, p);

    if (c1 >= 0.0 && c2 >= 0.0 && c3 >= 0.0) || (c1 <= 0.0 && c2 <= 0.0 && c3 <= 0.0) {
        sign = -1.0;
    }

    sign * min_dist
}

/// SDF of an ellipse centered at origin (approximate).
pub fn sdf_ellipse(p: Vec2, radii: Vec2) -> f32 {
    // Approximate: normalize to unit circle, compute distance, scale back
    let np = Vec2::new(p.x / radii.x, p.y / radii.y);
    let d = np.length() - 1.0;
    // Approximate correction using gradient length
    let grad = Vec2::new(p.x / (radii.x * radii.x), p.y / (radii.y * radii.y));
    let grad_len = grad.length();
    if grad_len < 1e-10 {
        return -(radii.x.min(radii.y));
    }
    d * np.length() / grad_len * (if d < 0.0 { 1.0 } else { 1.0 })
}

/// SDF of a ring (annulus) centered at origin.
pub fn sdf_ring(p: Vec2, outer_radius: f32, inner_radius: f32) -> f32 {
    let thickness = (outer_radius - inner_radius) * 0.5;
    let mid_radius = (outer_radius + inner_radius) * 0.5;
    (p.length() - mid_radius).abs() - thickness
}

/// SDF of a circular arc centered at origin.
/// `aperture` is half the arc angle in radians.
pub fn sdf_arc(p: Vec2, radius: f32, aperture: f32, thickness: f32) -> f32 {
    let angle = p.y.atan2(p.x).abs();
    if angle < aperture {
        ((p.length() - radius).abs() - thickness * 0.5).max(0.0)
    } else {
        // Distance to arc endpoints
        let tip = Vec2::new(radius * aperture.cos(), radius * aperture.sin());
        let d1 = p.sub(tip).length();
        let tip2 = Vec2::new(tip.x, -tip.y);
        let d2 = p.sub(tip2).length();
        d1.min(d2) - thickness * 0.5
    }
}

/// SDF of a pie/sector shape centered at origin.
pub fn sdf_pie(p: Vec2, radius: f32, aperture: f32) -> f32 {
    let angle = p.y.atan2(p.x).abs();
    if angle < aperture {
        p.length() - radius
    } else {
        // Distance to edge
        let edge_dir = Vec2::new(aperture.cos(), aperture.sin());
        let proj = p.x * edge_dir.x + p.y.abs() * edge_dir.y;
        let proj_clamped = proj.clamp(0.0, radius);
        let closest = edge_dir.scale(proj_clamped);
        let q = Vec2::new(p.x, p.y.abs());
        q.sub(closest).length()
    }
}

/// SDF of a cross shape centered at origin.
pub fn sdf_cross(p: Vec2, arm_length: f32, arm_width: f32) -> f32 {
    let half_w = arm_width * 0.5;
    let pa = p.abs();
    // Union of horizontal and vertical bars
    let h_bar = sdf_box(
        Vec2::new(pa.x, pa.y),
        Vec2::new(arm_length, half_w),
    );
    let v_bar = sdf_box(
        Vec2::new(pa.x, pa.y),
        Vec2::new(half_w, arm_length),
    );
    h_bar.min(v_bar)
}

/// SDF of a regular polygon centered at origin.
pub fn sdf_regular_polygon(p: Vec2, radius: f32, sides: u32) -> f32 {
    if sides < 3 {
        return sdf_circle(p, radius);
    }
    let n = sides as f32;
    let angle_step = PI * 2.0 / n;
    let angle = p.y.atan2(p.x);
    // Angle within current sector
    let sector_angle = ((angle % angle_step) + angle_step) % angle_step - angle_step * 0.5;
    let r = p.length();
    // Distance to edge of polygon
    let cos_a = (angle_step * 0.5).cos();
    r * sector_angle.abs().cos() - radius * cos_a
}

/// SDF of a star shape.
pub fn sdf_star(p: Vec2, outer_radius: f32, inner_radius: f32, points: u32) -> f32 {
    if points < 2 {
        return sdf_circle(p, outer_radius);
    }
    let n = points as f32;
    let angle_step = PI / n;
    let angle = p.y.atan2(p.x).abs();
    let sector = ((angle / angle_step) as u32) % 2;
    let sector_angle = angle % angle_step;

    let r = p.length();
    let target_r = if sector == 0 {
        // Interpolate from outer to inner
        let t = sector_angle / angle_step;
        outer_radius * (1.0 - t) + inner_radius * t
    } else {
        let t = sector_angle / angle_step;
        inner_radius * (1.0 - t) + outer_radius * t
    };

    r - target_r
}

// --- CSG Operations ---

/// Union of two SDFs: min(a, b).
pub fn sdf_union(a: f32, b: f32) -> f32 {
    a.min(b)
}

/// Intersection of two SDFs: max(a, b).
pub fn sdf_intersection(a: f32, b: f32) -> f32 {
    a.max(b)
}

/// Subtraction: a minus b = max(a, -b).
pub fn sdf_subtraction(a: f32, b: f32) -> f32 {
    a.max(-b)
}

/// Smooth union (polynomial smooth min).
pub fn sdf_smooth_union(a: f32, b: f32, k: f32) -> f32 {
    if k <= 0.0 {
        return a.min(b);
    }
    let h = (0.5 + 0.5 * (a - b) / k).clamp(0.0, 1.0);
    a * (1.0 - h) + b * h - k * h * (1.0 - h)
}

/// Smooth intersection (polynomial smooth max).
pub fn sdf_smooth_intersection(a: f32, b: f32, k: f32) -> f32 {
    if k <= 0.0 {
        return a.max(b);
    }
    let h = (0.5 - 0.5 * (a - b) / k).clamp(0.0, 1.0);
    a * (1.0 - h) + b * h + k * h * (1.0 - h)
}

// --- Rendering Helpers ---

/// Convert SDF distance to pixel coverage (anti-aliased).
/// `pixel_width` is the width of one pixel in SDF units.
pub fn sdf_coverage(distance: f32, pixel_width: f32) -> f32 {
    let half = pixel_width * 0.5;
    (0.5 - distance / pixel_width).clamp(0.0, 1.0)
        * (if distance < -half {
            1.0
        } else if distance > half {
            0.0
        } else {
            1.0
        })
}

/// Simpler coverage: smoothstep.
pub fn sdf_smooth_coverage(distance: f32, edge_width: f32) -> f32 {
    let t = ((-distance / edge_width) + 0.5).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Stroke from SDF: outline at distance d with given half-width.
pub fn sdf_stroke(distance: f32, half_width: f32) -> f32 {
    distance.abs() - half_width
}

/// Shadow: offset the sample point and blur.
pub fn sdf_shadow(distance: f32, blur_radius: f32) -> f32 {
    if blur_radius <= 0.0 {
        return if distance < 0.0 { 1.0 } else { 0.0 };
    }
    sdf_smooth_coverage(distance, blur_radius)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec2_length() {
        let v = Vec2::new(3.0, 4.0);
        assert!((v.length() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_vec2_dot() {
        let a = Vec2::new(1.0, 0.0);
        let b = Vec2::new(0.0, 1.0);
        assert!(a.dot(b).abs() < 1e-6);
    }

    #[test]
    fn test_sdf_circle_inside() {
        assert!(sdf_circle(Vec2::ZERO, 5.0) < 0.0);
    }

    #[test]
    fn test_sdf_circle_outside() {
        assert!(sdf_circle(Vec2::new(10.0, 0.0), 5.0) > 0.0);
    }

    #[test]
    fn test_sdf_circle_on_edge() {
        let d = sdf_circle(Vec2::new(5.0, 0.0), 5.0);
        assert!(d.abs() < 1e-6);
    }

    #[test]
    fn test_sdf_box_inside() {
        let d = sdf_box(Vec2::ZERO, Vec2::new(5.0, 5.0));
        assert!(d < 0.0);
    }

    #[test]
    fn test_sdf_box_outside() {
        let d = sdf_box(Vec2::new(10.0, 0.0), Vec2::new(5.0, 5.0));
        assert!(d > 0.0);
    }

    #[test]
    fn test_sdf_box_corner_distance() {
        let d = sdf_box(Vec2::new(8.0, 8.0), Vec2::new(5.0, 5.0));
        let expected = Vec2::new(3.0, 3.0).length();
        assert!((d - expected).abs() < 1e-4);
    }

    #[test]
    fn test_sdf_rounded_rect() {
        let d = sdf_rounded_rect(Vec2::ZERO, Vec2::new(10.0, 10.0), 3.0);
        assert!(d < 0.0);
        // On a corner, distance should account for rounding
        let corner = sdf_rounded_rect(Vec2::new(10.0, 10.0), Vec2::new(10.0, 10.0), 3.0);
        assert!(corner > 0.0);
    }

    #[test]
    fn test_sdf_segment() {
        let d = sdf_segment(Vec2::new(0.0, 5.0), Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0));
        assert!((d - 5.0).abs() < 1e-4);
    }

    #[test]
    fn test_sdf_triangle_inside() {
        let a = Vec2::new(0.0, 10.0);
        let b = Vec2::new(-10.0, -10.0);
        let c = Vec2::new(10.0, -10.0);
        let d = sdf_triangle(Vec2::ZERO, a, b, c);
        assert!(d < 0.0);
    }

    #[test]
    fn test_sdf_triangle_outside() {
        let a = Vec2::new(0.0, 10.0);
        let b = Vec2::new(-10.0, -10.0);
        let c = Vec2::new(10.0, -10.0);
        let d = sdf_triangle(Vec2::new(20.0, 20.0), a, b, c);
        assert!(d > 0.0);
    }

    #[test]
    fn test_sdf_ring() {
        let d = sdf_ring(Vec2::new(7.5, 0.0), 10.0, 5.0);
        assert!(d < 0.0); // inside ring
        let d2 = sdf_ring(Vec2::ZERO, 10.0, 5.0);
        assert!(d2 > 0.0); // center hole
    }

    #[test]
    fn test_sdf_cross() {
        let d = sdf_cross(Vec2::new(3.0, 0.0), 10.0, 4.0);
        assert!(d < 0.0);
        let d2 = sdf_cross(Vec2::new(8.0, 8.0), 10.0, 4.0);
        assert!(d2 > 0.0);
    }

    #[test]
    fn test_sdf_regular_polygon_center() {
        let d = sdf_regular_polygon(Vec2::ZERO, 10.0, 6);
        assert!(d < 0.0);
    }

    #[test]
    fn test_sdf_star_center() {
        let d = sdf_star(Vec2::ZERO, 10.0, 5.0, 5);
        assert!(d < 0.0);
    }

    #[test]
    fn test_union() {
        assert!((sdf_union(3.0, 5.0) - 3.0).abs() < 1e-6);
        assert!((sdf_union(-2.0, 1.0) - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_intersection() {
        assert!((sdf_intersection(3.0, 5.0) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_subtraction() {
        // a - b: inside a but outside b
        let a = -5.0; // deep inside a
        let b = -3.0; // deep inside b
        let d = sdf_subtraction(a, b);
        assert!((d - 3.0).abs() < 1e-6); // should be positive (subtracted)
    }

    #[test]
    fn test_smooth_union() {
        let a = 2.0;
        let b = 3.0;
        let su = sdf_smooth_union(a, b, 1.0);
        // Smooth union should be <= min(a, b)
        assert!(su <= a + 1e-6);
    }

    #[test]
    fn test_smooth_union_zero_k() {
        let su = sdf_smooth_union(2.0, 3.0, 0.0);
        assert!((su - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_smooth_intersection() {
        let si = sdf_smooth_intersection(2.0, 3.0, 1.0);
        assert!(si >= 3.0 - 1e-6);
    }

    #[test]
    fn test_stroke() {
        let s = sdf_stroke(-2.0, 1.0);
        assert!((s - 1.0).abs() < 1e-6); // |−2| − 1 = 1
        let s2 = sdf_stroke(0.5, 1.0);
        assert!((s2 - (-0.5)).abs() < 1e-6); // |0.5| − 1 = −0.5
    }

    #[test]
    fn test_shadow_sharp() {
        let s = sdf_shadow(-1.0, 0.0);
        assert!((s - 1.0).abs() < 1e-6);
        let s2 = sdf_shadow(1.0, 0.0);
        assert!(s2.abs() < 1e-6);
    }

    #[test]
    fn test_smooth_coverage_inside() {
        let c = sdf_smooth_coverage(-10.0, 1.0);
        assert!((c - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_smooth_coverage_outside() {
        let c = sdf_smooth_coverage(10.0, 1.0);
        assert!(c < 1e-4);
    }

    #[test]
    fn test_smooth_coverage_edge() {
        let c = sdf_smooth_coverage(0.0, 1.0);
        assert!((c - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_sdf_ellipse_on_axis() {
        let d = sdf_ellipse(Vec2::new(5.0, 0.0), Vec2::new(5.0, 3.0));
        assert!(d.abs() < 0.5); // Approximate — close to edge
    }

    #[test]
    fn test_sdf_pie_inside() {
        let d = sdf_pie(Vec2::new(3.0, 0.0), 5.0, PI * 0.5);
        assert!(d < 0.0);
    }

    #[test]
    fn test_vec2_operations() {
        let a = Vec2::new(1.0, 2.0);
        let b = Vec2::new(3.0, 4.0);
        let s = a.add(b);
        assert!((s.x - 4.0).abs() < 1e-6);
        assert!((s.y - 6.0).abs() < 1e-6);
        let d = a.sub(b);
        assert!((d.x - (-2.0)).abs() < 1e-6);
        let sc = a.scale(3.0);
        assert!((sc.x - 3.0).abs() < 1e-6);
    }
}
