// Photon mapping for caustics and indirect illumination.
// kd-tree storage, nearest-neighbor gather, kernel density estimation.

use std::fmt;

const PI: f64 = std::f64::consts::PI;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self { Self { x, y, z } }
    pub fn zero() -> Self { Self { x: 0.0, y: 0.0, z: 0.0 } }
    pub fn dot(self, o: Self) -> f64 { self.x * o.x + self.y * o.y + self.z * o.z }
    pub fn length_squared(self) -> f64 { self.dot(self) }
    pub fn length(self) -> f64 { self.length_squared().sqrt() }
    pub fn normalized(self) -> Self {
        let l = self.length();
        if l < 1e-15 { Self::zero() } else { self * (1.0 / l) }
    }
    pub fn index(self, axis: usize) -> f64 {
        match axis { 0 => self.x, 1 => self.y, _ => self.z }
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

impl fmt::Display for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.4}, {:.4}, {:.4})", self.x, self.y, self.z)
    }
}

/// RGB power carried by a photon.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    pub r: f64,
    pub g: f64,
    pub b: f64,
}

impl Color {
    pub fn new(r: f64, g: f64, b: f64) -> Self { Self { r, g, b } }
    pub fn black() -> Self { Self { r: 0.0, g: 0.0, b: 0.0 } }
    pub fn luminance(self) -> f64 { 0.2126 * self.r + 0.7152 * self.g + 0.0722 * self.b }
}

impl std::ops::Add for Color {
    type Output = Self;
    fn add(self, r: Self) -> Self { Self { r: self.r + r.r, g: self.g + r.g, b: self.b + r.b } }
}
impl std::ops::Mul<f64> for Color {
    type Output = Self;
    fn mul(self, s: f64) -> Self { Self { r: self.r * s, g: self.g * s, b: self.b * s } }
}
impl std::ops::Mul for Color {
    type Output = Self;
    fn mul(self, o: Self) -> Self { Self { r: self.r * o.r, g: self.g * o.g, b: self.b * o.b } }
}

/// Photon type flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhotonType {
    Caustic,
    Global,
}

/// A single photon stored in the map.
#[derive(Debug, Clone)]
pub struct Photon {
    pub position: Vec3,
    pub direction: Vec3,
    pub power: Color,
    pub photon_type: PhotonType,
}

impl Photon {
    pub fn new(position: Vec3, direction: Vec3, power: Color, photon_type: PhotonType) -> Self {
        Self { position, direction: direction.normalized(), power, photon_type }
    }
}

/// Simple LCG random for photon emission.
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self { Self { state: seed.wrapping_add(1) } }
    pub fn next_f64(&mut self) -> f64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let bits = (self.state >> 11) as f64;
        bits / (1u64 << 53) as f64
    }
}

/// Point light source for photon emission.
#[derive(Debug, Clone)]
pub struct PointLight {
    pub position: Vec3,
    pub power: Color,
}

/// Emit photons uniformly from a point light.
pub fn emit_photons(light: &PointLight, count: usize, rng: &mut Rng) -> Vec<Photon> {
    let power_per_photon = light.power * (1.0 / count as f64);
    let mut photons = Vec::with_capacity(count);
    for _ in 0..count {
        let dir = sample_sphere(rng);
        photons.push(Photon::new(light.position, dir, power_per_photon, PhotonType::Global));
    }
    photons
}

fn sample_sphere(rng: &mut Rng) -> Vec3 {
    let z = 1.0 - 2.0 * rng.next_f64();
    let r = (1.0 - z * z).max(0.0).sqrt();
    let phi = 2.0 * PI * rng.next_f64();
    Vec3::new(r * phi.cos(), r * phi.sin(), z)
}

// ─── kd-tree for photon storage ───

/// kd-tree node stored in a flat array (balanced, median split).
#[derive(Debug, Clone)]
struct KdNode {
    photon_idx: usize,
    split_axis: u8,
    left: Option<usize>,
    right: Option<usize>,
}

/// Photon map backed by a kd-tree.
pub struct PhotonMap {
    photons: Vec<Photon>,
    nodes: Vec<KdNode>,
    root: Option<usize>,
}

impl PhotonMap {
    /// Build a photon map from a list of photons.
    pub fn build(mut photons: Vec<Photon>) -> Self {
        if photons.is_empty() {
            return Self { photons, nodes: Vec::new(), root: None };
        }
        let mut indices: Vec<usize> = (0..photons.len()).collect();
        let mut nodes = Vec::with_capacity(photons.len());
        let root = Self::build_node(&photons, &mut indices, 0, photons.len(), 0, &mut nodes);
        // Reorder photons to match index order in nodes
        let _ = &mut photons;
        Self { photons, nodes, root: Some(root) }
    }

    fn build_node(
        photons: &[Photon],
        indices: &mut [usize],
        start: usize,
        end: usize,
        depth: usize,
        nodes: &mut Vec<KdNode>,
    ) -> usize {
        let count = end - start;
        let axis = depth % 3;

        if count == 1 {
            let node_idx = nodes.len();
            nodes.push(KdNode {
                photon_idx: indices[start],
                split_axis: axis as u8,
                left: None,
                right: None,
            });
            return node_idx;
        }

        // Sort by axis and pick median
        let slice = &mut indices[start..end];
        slice.sort_by(|&a, &b| {
            let va = photons[a].position.index(axis);
            let vb = photons[b].position.index(axis);
            va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mid = start + count / 2;
        let node_idx = nodes.len();
        nodes.push(KdNode {
            photon_idx: indices[mid],
            split_axis: axis as u8,
            left: None,
            right: None,
        });

        let left = if mid > start {
            Some(Self::build_node(photons, indices, start, mid, depth + 1, nodes))
        } else {
            None
        };

        let right = if mid + 1 < end {
            Some(Self::build_node(photons, indices, mid + 1, end, depth + 1, nodes))
        } else {
            None
        };

        nodes[node_idx].left = left;
        nodes[node_idx].right = right;
        node_idx
    }

    pub fn len(&self) -> usize {
        self.photons.len()
    }

    pub fn is_empty(&self) -> bool {
        self.photons.is_empty()
    }

    /// Find the N nearest photons to a query point within max_radius.
    pub fn gather(
        &self,
        query: Vec3,
        max_count: usize,
        max_radius: f64,
    ) -> Vec<(usize, f64)> {
        let mut results: Vec<(usize, f64)> = Vec::new();
        let mut max_dist_sq = max_radius * max_radius;
        if let Some(root) = self.root {
            self.gather_recursive(root, query, max_count, &mut max_dist_sq, &mut results);
        }
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    fn gather_recursive(
        &self,
        node_idx: usize,
        query: Vec3,
        max_count: usize,
        max_dist_sq: &mut f64,
        results: &mut Vec<(usize, f64)>,
    ) {
        let node = &self.nodes[node_idx];
        let photon = &self.photons[node.photon_idx];
        let diff = query - photon.position;
        let dist_sq = diff.length_squared();

        if dist_sq < *max_dist_sq {
            if results.len() < max_count {
                results.push((node.photon_idx, dist_sq));
                if results.len() == max_count {
                    // Find max dist in results and update max_dist_sq
                    let max_d = results.iter().map(|r| r.1).fold(0.0f64, |a, b| a.max(b));
                    *max_dist_sq = max_d;
                }
            } else {
                // Replace furthest
                let mut worst_idx = 0;
                let mut worst_dist = 0.0f64;
                for (i, r) in results.iter().enumerate() {
                    if r.1 > worst_dist {
                        worst_dist = r.1;
                        worst_idx = i;
                    }
                }
                if dist_sq < worst_dist {
                    results[worst_idx] = (node.photon_idx, dist_sq);
                    let max_d = results.iter().map(|r| r.1).fold(0.0f64, |a, b| a.max(b));
                    *max_dist_sq = max_d;
                }
            }
        }

        let axis = node.split_axis as usize;
        let delta = query.index(axis) - photon.position.index(axis);
        let delta_sq = delta * delta;

        let (near, far) = if delta < 0.0 {
            (node.left, node.right)
        } else {
            (node.right, node.left)
        };

        if let Some(near_idx) = near {
            self.gather_recursive(near_idx, query, max_count, max_dist_sq, results);
        }
        if let Some(far_idx) = far {
            if delta_sq < *max_dist_sq {
                self.gather_recursive(far_idx, query, max_count, max_dist_sq, results);
            }
        }
    }

    pub fn get_photon(&self, idx: usize) -> &Photon {
        &self.photons[idx]
    }
}

/// Kernel filter types for density estimation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KernelFilter {
    /// Box filter (constant weight 1).
    Box,
    /// Cone filter with parameter k (usually 1).
    Cone { k: f64 },
    /// Gaussian filter with sigma = alpha * max_radius.
    Gaussian { alpha: f64 },
}

/// Estimate radiance at a point using gathered photons.
pub fn radiance_estimate(
    map: &PhotonMap,
    point: Vec3,
    normal: Vec3,
    max_photons: usize,
    max_radius: f64,
    filter: KernelFilter,
) -> Color {
    let gathered = map.gather(point, max_photons, max_radius);
    if gathered.is_empty() {
        return Color::black();
    }

    // Find actual max distance for the gathered set
    let actual_max_dist_sq = gathered.iter().map(|g| g.1).fold(0.0f64, |a, b| a.max(b));
    let actual_radius = if actual_max_dist_sq.sqrt() < 1e-15 {
        max_radius
    } else {
        actual_max_dist_sq.sqrt()
    };
    if actual_radius < 1e-15 {
        return Color::black();
    }

    let mut flux = Color::black();
    let n = normal.normalized();

    for &(idx, dist_sq) in &gathered {
        let photon = map.get_photon(idx);
        // Only count photons on the correct hemisphere
        if photon.direction.dot(n) > 0.0 {
            continue; // photon coming from behind
        }

        let dist = dist_sq.sqrt();
        let weight = match filter {
            KernelFilter::Box => 1.0,
            KernelFilter::Cone { k } => {
                let dp = dist / actual_radius;
                (1.0 - dp / k).max(0.0)
            }
            KernelFilter::Gaussian { alpha } => {
                let sigma = alpha * actual_radius;
                let sigma_sq = sigma * sigma;
                (-dist_sq / (2.0 * sigma_sq)).exp()
            }
        };

        flux = flux + photon.power * weight;
    }

    // Normalize by area
    let area = PI * actual_radius * actual_radius;
    flux * (1.0 / area)
}

/// Build a caustic photon map: only store photons that followed specular->diffuse paths.
pub fn build_caustic_map(photons: Vec<Photon>) -> PhotonMap {
    let caustic_photons: Vec<Photon> = photons
        .into_iter()
        .filter(|p| p.photon_type == PhotonType::Caustic)
        .collect();
    PhotonMap::build(caustic_photons)
}

/// Build a global photon map for indirect illumination.
pub fn build_global_map(photons: Vec<Photon>) -> PhotonMap {
    let global_photons: Vec<Photon> = photons
        .into_iter()
        .filter(|p| p.photon_type == PhotonType::Global)
        .collect();
    PhotonMap::build(global_photons)
}

/// Mark photon as caustic type.
pub fn mark_caustic(photon: &mut Photon) {
    photon.photon_type = PhotonType::Caustic;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool { (a - b).abs() < eps }

    #[test]
    fn test_photon_creation() {
        let p = Photon::new(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(0.0, -1.0, 0.0),
            Color::new(0.5, 0.5, 0.5),
            PhotonType::Global,
        );
        assert!(approx_eq(p.direction.length(), 1.0, 1e-9));
        assert_eq!(p.photon_type, PhotonType::Global);
    }

    #[test]
    fn test_emit_photons_count() {
        let light = PointLight { position: Vec3::zero(), power: Color::new(100.0, 100.0, 100.0) };
        let mut rng = Rng::new(42);
        let photons = emit_photons(&light, 1000, &mut rng);
        assert_eq!(photons.len(), 1000);
    }

    #[test]
    fn test_emit_photons_power_conservation() {
        let light = PointLight { position: Vec3::zero(), power: Color::new(10.0, 20.0, 30.0) };
        let mut rng = Rng::new(42);
        let photons = emit_photons(&light, 500, &mut rng);
        let total_r: f64 = photons.iter().map(|p| p.power.r).sum();
        let total_g: f64 = photons.iter().map(|p| p.power.g).sum();
        let total_b: f64 = photons.iter().map(|p| p.power.b).sum();
        assert!(approx_eq(total_r, 10.0, 1e-6));
        assert!(approx_eq(total_g, 20.0, 1e-6));
        assert!(approx_eq(total_b, 30.0, 1e-6));
    }

    #[test]
    fn test_emit_photons_directions_normalized() {
        let light = PointLight { position: Vec3::zero(), power: Color::new(1.0, 1.0, 1.0) };
        let mut rng = Rng::new(77);
        let photons = emit_photons(&light, 100, &mut rng);
        for p in &photons {
            assert!(approx_eq(p.direction.length(), 1.0, 1e-9));
        }
    }

    #[test]
    fn test_photon_map_empty() {
        let map = PhotonMap::build(vec![]);
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);
        let results = map.gather(Vec3::zero(), 10, 1.0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_photon_map_single() {
        let photons = vec![
            Photon::new(Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, -1.0, 0.0), Color::new(1.0, 0.0, 0.0), PhotonType::Global),
        ];
        let map = PhotonMap::build(photons);
        assert_eq!(map.len(), 1);
        let results = map.gather(Vec3::new(1.0, 0.0, 0.0), 5, 1.0);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_photon_map_gather_nearest() {
        let photons = vec![
            Photon::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, -1.0, 0.0), Color::new(1.0, 0.0, 0.0), PhotonType::Global),
            Photon::new(Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, -1.0, 0.0), Color::new(0.0, 1.0, 0.0), PhotonType::Global),
            Photon::new(Vec3::new(10.0, 0.0, 0.0), Vec3::new(0.0, -1.0, 0.0), Color::new(0.0, 0.0, 1.0), PhotonType::Global),
        ];
        let map = PhotonMap::build(photons);
        // Query near origin, should get first two
        let results = map.gather(Vec3::zero(), 2, 5.0);
        assert_eq!(results.len(), 2);
        // Closest should be photon at origin (dist=0)
        assert!(approx_eq(results[0].1, 0.0, 1e-9));
    }

    #[test]
    fn test_photon_map_gather_radius_limit() {
        let photons = vec![
            Photon::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, -1.0, 0.0), Color::new(1.0, 0.0, 0.0), PhotonType::Global),
            Photon::new(Vec3::new(100.0, 0.0, 0.0), Vec3::new(0.0, -1.0, 0.0), Color::new(0.0, 1.0, 0.0), PhotonType::Global),
        ];
        let map = PhotonMap::build(photons);
        let results = map.gather(Vec3::zero(), 10, 5.0);
        assert_eq!(results.len(), 1); // only nearby photon
    }

    #[test]
    fn test_radiance_estimate_box_filter() {
        let dir = Vec3::new(0.0, -1.0, 0.0); // photon going down
        let normal = Vec3::new(0.0, 1.0, 0.0); // surface facing up
        let photons = vec![
            Photon::new(Vec3::new(0.0, 0.0, 0.0), dir, Color::new(1.0, 0.0, 0.0), PhotonType::Global),
            Photon::new(Vec3::new(0.1, 0.0, 0.0), dir, Color::new(1.0, 0.0, 0.0), PhotonType::Global),
        ];
        let map = PhotonMap::build(photons);
        let rad = radiance_estimate(&map, Vec3::zero(), normal, 10, 1.0, KernelFilter::Box);
        assert!(rad.r > 0.0);
        assert!(rad.r.is_finite());
    }

    #[test]
    fn test_radiance_estimate_cone_filter() {
        let dir = Vec3::new(0.0, -1.0, 0.0);
        let normal = Vec3::new(0.0, 1.0, 0.0);
        let photons = vec![
            Photon::new(Vec3::zero(), dir, Color::new(2.0, 0.0, 0.0), PhotonType::Global),
            Photon::new(Vec3::new(0.5, 0.0, 0.0), dir, Color::new(2.0, 0.0, 0.0), PhotonType::Global),
        ];
        let map = PhotonMap::build(photons);
        let rad = radiance_estimate(&map, Vec3::zero(), normal, 10, 2.0, KernelFilter::Cone { k: 1.0 });
        assert!(rad.r > 0.0);
    }

    #[test]
    fn test_radiance_estimate_gaussian_filter() {
        let dir = Vec3::new(0.0, -1.0, 0.0);
        let normal = Vec3::new(0.0, 1.0, 0.0);
        let photons = vec![
            Photon::new(Vec3::zero(), dir, Color::new(5.0, 5.0, 5.0), PhotonType::Global),
        ];
        let map = PhotonMap::build(photons);
        let rad = radiance_estimate(&map, Vec3::zero(), normal, 10, 1.0, KernelFilter::Gaussian { alpha: 0.918 });
        assert!(rad.r > 0.0);
    }

    #[test]
    fn test_radiance_ignores_backfacing() {
        let dir = Vec3::new(0.0, 1.0, 0.0); // photon going UP (behind surface)
        let normal = Vec3::new(0.0, 1.0, 0.0);
        let photons = vec![
            Photon::new(Vec3::zero(), dir, Color::new(10.0, 10.0, 10.0), PhotonType::Global),
        ];
        let map = PhotonMap::build(photons);
        let rad = radiance_estimate(&map, Vec3::zero(), normal, 10, 1.0, KernelFilter::Box);
        // Photon direction dots positive with normal -> ignored
        assert!(approx_eq(rad.r, 0.0, 1e-9));
    }

    #[test]
    fn test_caustic_map_filters() {
        let photons = vec![
            Photon::new(Vec3::zero(), Vec3::new(0.0, -1.0, 0.0), Color::new(1.0, 0.0, 0.0), PhotonType::Caustic),
            Photon::new(Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, -1.0, 0.0), Color::new(0.0, 1.0, 0.0), PhotonType::Global),
        ];
        let caustic_map = build_caustic_map(photons);
        assert_eq!(caustic_map.len(), 1);
    }

    #[test]
    fn test_global_map_filters() {
        let photons = vec![
            Photon::new(Vec3::zero(), Vec3::new(0.0, -1.0, 0.0), Color::new(1.0, 0.0, 0.0), PhotonType::Caustic),
            Photon::new(Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, -1.0, 0.0), Color::new(0.0, 1.0, 0.0), PhotonType::Global),
            Photon::new(Vec3::new(2.0, 0.0, 0.0), Vec3::new(0.0, -1.0, 0.0), Color::new(0.0, 0.0, 1.0), PhotonType::Global),
        ];
        let global_map = build_global_map(photons);
        assert_eq!(global_map.len(), 2);
    }

    #[test]
    fn test_mark_caustic() {
        let mut p = Photon::new(Vec3::zero(), Vec3::new(0.0, -1.0, 0.0), Color::new(1.0, 1.0, 1.0), PhotonType::Global);
        assert_eq!(p.photon_type, PhotonType::Global);
        mark_caustic(&mut p);
        assert_eq!(p.photon_type, PhotonType::Caustic);
    }

    #[test]
    fn test_kdtree_many_photons() {
        let mut rng = Rng::new(42);
        let photons: Vec<Photon> = (0..500).map(|_| {
            let pos = Vec3::new(
                rng.next_f64() * 100.0 - 50.0,
                rng.next_f64() * 100.0 - 50.0,
                rng.next_f64() * 100.0 - 50.0,
            );
            Photon::new(pos, Vec3::new(0.0, -1.0, 0.0), Color::new(0.01, 0.01, 0.01), PhotonType::Global)
        }).collect();
        let map = PhotonMap::build(photons);
        assert_eq!(map.len(), 500);

        let results = map.gather(Vec3::zero(), 20, 30.0);
        assert!(results.len() <= 20);
        // Results should be sorted by distance
        for w in results.windows(2) {
            assert!(w[0].1 <= w[1].1 + 1e-12);
        }
    }

    #[test]
    fn test_gather_respects_max_count() {
        let photons: Vec<Photon> = (0..50).map(|i| {
            Photon::new(
                Vec3::new(i as f64 * 0.01, 0.0, 0.0),
                Vec3::new(0.0, -1.0, 0.0),
                Color::new(1.0, 1.0, 1.0),
                PhotonType::Global,
            )
        }).collect();
        let map = PhotonMap::build(photons);
        let results = map.gather(Vec3::zero(), 5, 100.0);
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_color_luminance() {
        let c = Color::new(1.0, 1.0, 1.0);
        assert!(approx_eq(c.luminance(), 1.0, 1e-4));
    }

    #[test]
    fn test_color_ops() {
        let a = Color::new(0.5, 0.3, 0.1);
        let b = Color::new(0.2, 0.4, 0.6);
        let sum = a + b;
        assert!(approx_eq(sum.r, 0.7, 1e-9));
        let prod = a * b;
        assert!(approx_eq(prod.r, 0.1, 1e-9));
        let scaled = a * 2.0;
        assert!(approx_eq(scaled.r, 1.0, 1e-9));
    }

    #[test]
    fn test_sample_sphere_on_unit_sphere() {
        let mut rng = Rng::new(99);
        for _ in 0..100 {
            let d = sample_sphere(&mut rng);
            assert!(approx_eq(d.length(), 1.0, 1e-9));
        }
    }

    #[test]
    fn test_radiance_no_photons() {
        let map = PhotonMap::build(vec![]);
        let rad = radiance_estimate(&map, Vec3::zero(), Vec3::new(0.0, 1.0, 0.0), 10, 1.0, KernelFilter::Box);
        assert!(approx_eq(rad.r, 0.0, 1e-9));
        assert!(approx_eq(rad.g, 0.0, 1e-9));
        assert!(approx_eq(rad.b, 0.0, 1e-9));
    }
}
