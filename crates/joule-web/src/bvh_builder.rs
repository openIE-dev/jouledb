// Bounding Volume Hierarchy construction for ray tracing acceleration.
// Top-down BVH with binned SAH split, flattened to linear array.

use std::fmt;

const DEFAULT_MAX_LEAF_SIZE: usize = 4;
const SAH_TRAVERSAL_COST: f64 = 1.0;
const SAH_INTERSECT_COST: f64 = 1.0;
const NUM_BINS: usize = 12;

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
        Self { x: 0.0, y: 0.0, z: 0.0 }
    }

    pub fn splat(v: f64) -> Self {
        Self { x: v, y: v, z: v }
    }

    pub fn component_min(self, o: Self) -> Self {
        Self { x: self.x.min(o.x), y: self.y.min(o.y), z: self.z.min(o.z) }
    }

    pub fn component_max(self, o: Self) -> Self {
        Self { x: self.x.max(o.x), y: self.y.max(o.y), z: self.z.max(o.z) }
    }

    pub fn index(self, axis: usize) -> f64 {
        match axis {
            0 => self.x,
            1 => self.y,
            _ => self.z,
        }
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

impl fmt::Display for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.3}, {:.3}, {:.3})", self.x, self.y, self.z)
    }
}

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub fn empty() -> Self {
        Self { min: Vec3::splat(f64::INFINITY), max: Vec3::splat(f64::NEG_INFINITY) }
    }

    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    pub fn surface_area(&self) -> f64 {
        let d = self.max - self.min;
        if d.x < 0.0 || d.y < 0.0 || d.z < 0.0 {
            return 0.0;
        }
        2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
    }

    pub fn union(&self, other: &Aabb) -> Aabb {
        Aabb {
            min: self.min.component_min(other.min),
            max: self.max.component_max(other.max),
        }
    }

    pub fn expand_point(&self, p: Vec3) -> Aabb {
        Aabb {
            min: self.min.component_min(p),
            max: self.max.component_max(p),
        }
    }

    pub fn centroid(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    pub fn longest_axis(&self) -> usize {
        let d = self.max - self.min;
        if d.x >= d.y && d.x >= d.z { 0 }
        else if d.y >= d.z { 1 }
        else { 2 }
    }

    pub fn is_valid(&self) -> bool {
        self.min.x <= self.max.x && self.min.y <= self.max.y && self.min.z <= self.max.z
    }
}

/// A primitive with a bounding box and centroid for BVH construction.
#[derive(Debug, Clone)]
pub struct BvhPrimitive {
    pub bounds: Aabb,
    pub centroid: Vec3,
    pub index: usize,
}

impl BvhPrimitive {
    pub fn new(bounds: Aabb, index: usize) -> Self {
        Self { centroid: bounds.centroid(), bounds, index }
    }
}

/// Flat BVH node in cache-friendly linear layout.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearBvhNode {
    pub bounds: Aabb,
    /// For interior: index of second child. First child is always at offset+1.
    /// For leaf: first primitive index.
    pub offset: u32,
    /// Number of primitives (0 = interior node).
    pub prim_count: u16,
    /// Split axis (0=x, 1=y, 2=z) for interior nodes.
    pub axis: u8,
}

/// BVH statistics.
#[derive(Debug, Clone, PartialEq)]
pub struct BvhStats {
    pub node_count: usize,
    pub leaf_count: usize,
    pub interior_count: usize,
    pub max_depth: usize,
    pub total_primitives: usize,
    pub sah_cost: f64,
}

/// Tree node used during construction (before flattening).
enum BuildNode {
    Interior {
        bounds: Aabb,
        axis: usize,
        left: Box<BuildNode>,
        right: Box<BuildNode>,
    },
    Leaf {
        bounds: Aabb,
        first_prim: usize,
        count: usize,
    },
}

/// SAH bin for binned SAH evaluation.
struct SahBin {
    bounds: Aabb,
    count: usize,
}

impl SahBin {
    fn empty() -> Self {
        Self { bounds: Aabb::empty(), count: 0 }
    }
}

/// Build a BVH from a list of primitives using binned SAH.
pub fn build_bvh(primitives: &mut Vec<BvhPrimitive>, max_leaf: usize) -> (Vec<LinearBvhNode>, Vec<usize>) {
    if primitives.is_empty() {
        return (vec![], vec![]);
    }

    let leaf_size = if max_leaf == 0 { DEFAULT_MAX_LEAF_SIZE } else { max_leaf };
    let mut ordered_prims: Vec<usize> = Vec::with_capacity(primitives.len());
    let root = build_recursive(primitives, 0, primitives.len(), leaf_size, &mut ordered_prims);

    let mut nodes = Vec::new();
    flatten_bvh(&root, &mut nodes);

    (nodes, ordered_prims)
}

fn build_recursive(
    prims: &mut Vec<BvhPrimitive>,
    start: usize,
    end: usize,
    max_leaf: usize,
    ordered: &mut Vec<usize>,
) -> BuildNode {
    let count = end - start;

    // Compute total bounds
    let mut bounds = Aabb::empty();
    for p in &prims[start..end] {
        bounds = bounds.union(&p.bounds);
    }

    if count <= max_leaf {
        let first = ordered.len();
        for p in &prims[start..end] {
            ordered.push(p.index);
        }
        return BuildNode::Leaf { bounds, first_prim: first, count };
    }

    // Compute centroid bounds
    let mut centroid_bounds = Aabb::empty();
    for p in &prims[start..end] {
        centroid_bounds = centroid_bounds.expand_point(p.centroid);
    }

    let axis = centroid_bounds.longest_axis();
    let extent = centroid_bounds.max.index(axis) - centroid_bounds.min.index(axis);

    // Degenerate case: all centroids the same
    if extent < 1e-12 {
        let first = ordered.len();
        for p in &prims[start..end] {
            ordered.push(p.index);
        }
        return BuildNode::Leaf { bounds, first_prim: first, count };
    }

    // Binned SAH
    let mut bins: Vec<SahBin> = (0..NUM_BINS).map(|_| SahBin::empty()).collect();
    let min_c = centroid_bounds.min.index(axis);

    for p in &prims[start..end] {
        let offset = (p.centroid.index(axis) - min_c) / extent;
        let b = ((offset * NUM_BINS as f64) as usize).min(NUM_BINS - 1);
        bins[b].count += 1;
        bins[b].bounds = bins[b].bounds.union(&p.bounds);
    }

    // Evaluate SAH for each split position
    let mut best_cost = f64::INFINITY;
    let mut best_split = 0;

    let parent_sa = bounds.surface_area();
    if parent_sa < 1e-15 {
        let first = ordered.len();
        for p in &prims[start..end] {
            ordered.push(p.index);
        }
        return BuildNode::Leaf { bounds, first_prim: first, count };
    }
    let inv_parent_sa = 1.0 / parent_sa;

    for split in 1..NUM_BINS {
        let mut left_bounds = Aabb::empty();
        let mut left_count = 0usize;
        for b in &bins[..split] {
            left_bounds = left_bounds.union(&b.bounds);
            left_count += b.count;
        }

        let mut right_bounds = Aabb::empty();
        let mut right_count = 0usize;
        for b in &bins[split..] {
            right_bounds = right_bounds.union(&b.bounds);
            right_count += b.count;
        }

        if left_count == 0 || right_count == 0 {
            continue;
        }

        let cost = SAH_TRAVERSAL_COST +
            SAH_INTERSECT_COST * (
                left_count as f64 * left_bounds.surface_area() +
                right_count as f64 * right_bounds.surface_area()
            ) * inv_parent_sa;

        if cost < best_cost {
            best_cost = cost;
            best_split = split;
        }
    }

    let leaf_cost = SAH_INTERSECT_COST * count as f64;
    if leaf_cost <= best_cost && count <= max_leaf * 2 {
        let first = ordered.len();
        for p in &prims[start..end] {
            ordered.push(p.index);
        }
        return BuildNode::Leaf { bounds, first_prim: first, count };
    }

    // Partition primitives
    let threshold = min_c + extent * (best_split as f64 / NUM_BINS as f64);
    let mut mid = start;
    let mut i = start;
    while i < end {
        if prims[i].centroid.index(axis) < threshold {
            prims.swap(i, mid);
            mid += 1;
        }
        i += 1;
    }

    // Ensure non-degenerate partition
    if mid == start || mid == end {
        mid = start + count / 2;
    }

    let left = build_recursive(prims, start, mid, max_leaf, ordered);
    let right = build_recursive(prims, mid, end, max_leaf, ordered);

    BuildNode::Interior { bounds, axis, left: Box::new(left), right: Box::new(right) }
}

fn flatten_bvh(node: &BuildNode, nodes: &mut Vec<LinearBvhNode>) -> usize {
    let my_index = nodes.len();
    match node {
        BuildNode::Leaf { bounds, first_prim, count } => {
            nodes.push(LinearBvhNode {
                bounds: *bounds,
                offset: *first_prim as u32,
                prim_count: *count as u16,
                axis: 0,
            });
        }
        BuildNode::Interior { bounds, axis, left, right } => {
            nodes.push(LinearBvhNode {
                bounds: *bounds,
                offset: 0, // placeholder
                prim_count: 0,
                axis: *axis as u8,
            });
            flatten_bvh(left, nodes);
            let right_offset = flatten_bvh(right, nodes);
            nodes[my_index].offset = right_offset as u32;
        }
    }
    my_index
}

/// Compute BVH statistics.
pub fn bvh_stats(nodes: &[LinearBvhNode], total_prims: usize) -> BvhStats {
    if nodes.is_empty() {
        return BvhStats {
            node_count: 0, leaf_count: 0, interior_count: 0,
            max_depth: 0, total_primitives: total_prims, sah_cost: 0.0,
        };
    }

    let mut leaf_count = 0usize;
    let mut interior_count = 0usize;
    let mut max_depth = 0usize;
    let mut sah_cost = 0.0;

    let root_sa = nodes[0].bounds.surface_area();
    let inv_root_sa = if root_sa > 1e-15 { 1.0 / root_sa } else { 1.0 };

    fn walk(
        nodes: &[LinearBvhNode], idx: usize, depth: usize,
        inv_root_sa: f64,
        leaf_count: &mut usize, interior_count: &mut usize,
        max_depth: &mut usize, sah_cost: &mut f64,
    ) {
        if idx >= nodes.len() {
            return;
        }
        let node = &nodes[idx];
        if depth > *max_depth {
            *max_depth = depth;
        }
        if node.prim_count > 0 {
            *leaf_count += 1;
            let sa_ratio = node.bounds.surface_area() * inv_root_sa;
            *sah_cost += SAH_INTERSECT_COST * node.prim_count as f64 * sa_ratio;
        } else {
            *interior_count += 1;
            *sah_cost += SAH_TRAVERSAL_COST * node.bounds.surface_area() * inv_root_sa;
            walk(nodes, idx + 1, depth + 1, inv_root_sa, leaf_count, interior_count, max_depth, sah_cost);
            walk(nodes, node.offset as usize, depth + 1, inv_root_sa, leaf_count, interior_count, max_depth, sah_cost);
        }
    }

    walk(nodes, 0, 0, inv_root_sa, &mut leaf_count, &mut interior_count, &mut max_depth, &mut sah_cost);

    BvhStats {
        node_count: leaf_count + interior_count,
        leaf_count,
        interior_count,
        max_depth,
        total_primitives: total_prims,
        sah_cost,
    }
}

/// Refit BVH bounds from primitives (update bounds without rebuild).
pub fn refit_bvh(nodes: &mut [LinearBvhNode], primitives: &[Aabb], ordered_indices: &[usize]) {
    if nodes.is_empty() {
        return;
    }
    refit_recursive(nodes, 0, primitives, ordered_indices);
}

fn refit_recursive(nodes: &mut [LinearBvhNode], idx: usize, prims: &[Aabb], ordered: &[usize]) -> Aabb {
    if idx >= nodes.len() {
        return Aabb::empty();
    }

    let prim_count = nodes[idx].prim_count;
    if prim_count > 0 {
        let first = nodes[idx].offset as usize;
        let mut bounds = Aabb::empty();
        for i in 0..prim_count as usize {
            let pi = ordered[first + i];
            if pi < prims.len() {
                bounds = bounds.union(&prims[pi]);
            }
        }
        nodes[idx].bounds = bounds;
        bounds
    } else {
        let right_idx = nodes[idx].offset as usize;
        let left_bounds = refit_recursive(nodes, idx + 1, prims, ordered);
        let right_bounds = refit_recursive(nodes, right_idx, prims, ordered);
        let bounds = left_bounds.union(&right_bounds);
        nodes[idx].bounds = bounds;
        bounds
    }
}

/// Simple BVH traversal for a ray (returns indices of potentially intersecting primitives).
pub fn traverse_bvh(
    nodes: &[LinearBvhNode],
    ray_origin: Vec3,
    ray_dir: Vec3,
    t_min: f64,
    t_max: f64,
) -> Vec<(usize, usize)> {
    let mut result = Vec::new();
    if nodes.is_empty() {
        return result;
    }

    let inv_dir = Vec3::new(
        if ray_dir.x.abs() > 1e-15 { 1.0 / ray_dir.x } else { f64::INFINITY.copysign(ray_dir.x) },
        if ray_dir.y.abs() > 1e-15 { 1.0 / ray_dir.y } else { f64::INFINITY.copysign(ray_dir.y) },
        if ray_dir.z.abs() > 1e-15 { 1.0 / ray_dir.z } else { f64::INFINITY.copysign(ray_dir.z) },
    );

    let mut stack = vec![0usize];
    while let Some(idx) = stack.pop() {
        if idx >= nodes.len() {
            continue;
        }
        let node = &nodes[idx];

        if !ray_intersects_aabb(&node.bounds, ray_origin, inv_dir, t_min, t_max) {
            continue;
        }

        if node.prim_count > 0 {
            let first = node.offset as usize;
            result.push((first, node.prim_count as usize));
        } else {
            stack.push(node.offset as usize);
            stack.push(idx + 1);
        }
    }
    result
}

fn ray_intersects_aabb(aabb: &Aabb, origin: Vec3, inv_dir: Vec3, t_min: f64, t_max: f64) -> bool {
    let t0x = (aabb.min.x - origin.x) * inv_dir.x;
    let t1x = (aabb.max.x - origin.x) * inv_dir.x;
    let (t0x, t1x) = if inv_dir.x < 0.0 { (t1x, t0x) } else { (t0x, t1x) };

    let t0y = (aabb.min.y - origin.y) * inv_dir.y;
    let t1y = (aabb.max.y - origin.y) * inv_dir.y;
    let (t0y, t1y) = if inv_dir.y < 0.0 { (t1y, t0y) } else { (t0y, t1y) };

    let t0z = (aabb.min.z - origin.z) * inv_dir.z;
    let t1z = (aabb.max.z - origin.z) * inv_dir.z;
    let (t0z, t1z) = if inv_dir.z < 0.0 { (t1z, t0z) } else { (t0z, t1z) };

    let t_enter = t0x.max(t0y).max(t0z).max(t_min);
    let t_exit = t1x.min(t1y).min(t1z).min(t_max);
    t_enter <= t_exit
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_prim(cx: f64, cy: f64, cz: f64, size: f64, index: usize) -> BvhPrimitive {
        let half = size * 0.5;
        let bounds = Aabb::new(
            Vec3::new(cx - half, cy - half, cz - half),
            Vec3::new(cx + half, cy + half, cz + half),
        );
        BvhPrimitive::new(bounds, index)
    }

    #[test]
    fn test_aabb_empty() {
        let a = Aabb::empty();
        assert!(a.min.x == f64::INFINITY);
        assert!(a.max.x == f64::NEG_INFINITY);
    }

    #[test]
    fn test_aabb_surface_area() {
        let a = Aabb::new(Vec3::zero(), Vec3::new(2.0, 3.0, 4.0));
        let sa = a.surface_area();
        // 2*(2*3 + 3*4 + 4*2) = 2*(6+12+8) = 52
        assert!((sa - 52.0).abs() < 1e-9);
    }

    #[test]
    fn test_aabb_union() {
        let a = Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(0.0, 0.0, 0.0));
        let b = Aabb::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
        let u = a.union(&b);
        assert_eq!(u.min, Vec3::new(-1.0, -1.0, -1.0));
        assert_eq!(u.max, Vec3::new(1.0, 1.0, 1.0));
    }

    #[test]
    fn test_aabb_centroid() {
        let a = Aabb::new(Vec3::new(2.0, 4.0, 6.0), Vec3::new(4.0, 6.0, 8.0));
        let c = a.centroid();
        assert!((c.x - 3.0).abs() < 1e-9);
        assert!((c.y - 5.0).abs() < 1e-9);
        assert!((c.z - 7.0).abs() < 1e-9);
    }

    #[test]
    fn test_aabb_longest_axis() {
        let a = Aabb::new(Vec3::zero(), Vec3::new(1.0, 5.0, 2.0));
        assert_eq!(a.longest_axis(), 1);
    }

    #[test]
    fn test_build_empty() {
        let mut prims = vec![];
        let (nodes, ordered) = build_bvh(&mut prims, 4);
        assert!(nodes.is_empty());
        assert!(ordered.is_empty());
    }

    #[test]
    fn test_build_single() {
        let mut prims = vec![make_prim(0.0, 0.0, 0.0, 1.0, 0)];
        let (nodes, ordered) = build_bvh(&mut prims, 4);
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].prim_count, 1);
        assert_eq!(ordered.len(), 1);
        assert_eq!(ordered[0], 0);
    }

    #[test]
    fn test_build_two_prims() {
        let mut prims = vec![
            make_prim(-5.0, 0.0, 0.0, 1.0, 0),
            make_prim(5.0, 0.0, 0.0, 1.0, 1),
        ];
        let (nodes, ordered) = build_bvh(&mut prims, 1);
        // Should have interior + 2 leaves = 3 nodes
        assert!(nodes.len() >= 2);
        assert_eq!(ordered.len(), 2);
    }

    #[test]
    fn test_build_many_prims() {
        let mut prims: Vec<BvhPrimitive> = (0..20).map(|i| {
            make_prim(i as f64 * 3.0, 0.0, 0.0, 1.0, i)
        }).collect();
        let (nodes, ordered) = build_bvh(&mut prims, 4);
        assert!(!nodes.is_empty());
        assert_eq!(ordered.len(), 20);
    }

    #[test]
    fn test_bvh_stats_single() {
        let mut prims = vec![make_prim(0.0, 0.0, 0.0, 1.0, 0)];
        let (nodes, _) = build_bvh(&mut prims, 4);
        let stats = bvh_stats(&nodes, 1);
        assert_eq!(stats.leaf_count, 1);
        assert_eq!(stats.interior_count, 0);
        assert_eq!(stats.node_count, 1);
    }

    #[test]
    fn test_bvh_stats_multi() {
        let mut prims: Vec<BvhPrimitive> = (0..16).map(|i| {
            make_prim(i as f64 * 3.0, 0.0, 0.0, 1.0, i)
        }).collect();
        let (nodes, _) = build_bvh(&mut prims, 2);
        let stats = bvh_stats(&nodes, 16);
        assert!(stats.leaf_count >= 2);
        assert!(stats.interior_count >= 1);
        assert!(stats.max_depth >= 1);
        assert_eq!(stats.total_primitives, 16);
    }

    #[test]
    fn test_bvh_root_bounds_enclose_all() {
        let mut prims: Vec<BvhPrimitive> = (0..10).map(|i| {
            make_prim(i as f64 * 5.0, i as f64 * 2.0, 0.0, 1.0, i)
        }).collect();
        let (nodes, _) = build_bvh(&mut prims, 4);
        let root_bounds = nodes[0].bounds;
        // All primitive centroids should be within root bounds
        for p in &prims {
            assert!(p.centroid.x >= root_bounds.min.x - 1e-9);
            assert!(p.centroid.x <= root_bounds.max.x + 1e-9);
        }
    }

    #[test]
    fn test_ordered_indices_complete() {
        let n = 25;
        let mut prims: Vec<BvhPrimitive> = (0..n).map(|i| {
            make_prim(i as f64 * 2.0, (i as f64).sin() * 5.0, 0.0, 1.0, i)
        }).collect();
        let (_, ordered) = build_bvh(&mut prims, 4);
        assert_eq!(ordered.len(), n);
        let mut sorted = ordered.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), n);
    }

    #[test]
    fn test_refit_bvh() {
        let mut prims = vec![
            make_prim(-5.0, 0.0, 0.0, 1.0, 0),
            make_prim(5.0, 0.0, 0.0, 1.0, 1),
        ];
        let (mut nodes, ordered) = build_bvh(&mut prims, 1);
        // Move primitives
        let new_bounds = vec![
            Aabb::new(Vec3::new(-100.0, -1.0, -1.0), Vec3::new(-99.0, 1.0, 1.0)),
            Aabb::new(Vec3::new(99.0, -1.0, -1.0), Vec3::new(100.0, 1.0, 1.0)),
        ];
        refit_bvh(&mut nodes, &new_bounds, &ordered);
        let root = &nodes[0];
        assert!(root.bounds.min.x <= -99.0);
        assert!(root.bounds.max.x >= 99.0);
    }

    #[test]
    fn test_traverse_bvh_hit() {
        let mut prims: Vec<BvhPrimitive> = (0..8).map(|i| {
            make_prim(i as f64 * 10.0, 0.0, 0.0, 2.0, i)
        }).collect();
        let (nodes, _ordered) = build_bvh(&mut prims, 2);
        // Ray aiming at first primitive centered at (0,0,0)
        let results = traverse_bvh(&nodes, Vec3::new(0.0, 0.0, -5.0), Vec3::new(0.0, 0.0, 1.0), 0.0, 100.0);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_traverse_bvh_miss() {
        let mut prims: Vec<BvhPrimitive> = (0..8).map(|i| {
            make_prim(i as f64 * 10.0, 0.0, 0.0, 1.0, i)
        }).collect();
        let (nodes, _) = build_bvh(&mut prims, 2);
        // Ray far away from all prims
        let results = traverse_bvh(
            &nodes,
            Vec3::new(0.0, 1000.0, -5.0),
            Vec3::new(0.0, 0.0, 1.0),
            0.0,
            100.0,
        );
        assert!(results.is_empty());
    }

    #[test]
    fn test_identical_centroids() {
        // All prims at same position -> should become a single leaf
        let mut prims: Vec<BvhPrimitive> = (0..5).map(|i| {
            make_prim(0.0, 0.0, 0.0, 1.0, i)
        }).collect();
        let (nodes, ordered) = build_bvh(&mut prims, 2);
        assert!(!nodes.is_empty());
        assert_eq!(ordered.len(), 5);
    }

    #[test]
    fn test_linear_node_leaf_flag() {
        let mut prims = vec![make_prim(0.0, 0.0, 0.0, 1.0, 0)];
        let (nodes, _) = build_bvh(&mut prims, 4);
        assert!(nodes[0].prim_count > 0); // leaf
    }

    #[test]
    fn test_aabb_expand_point() {
        let a = Aabb::new(Vec3::zero(), Vec3::new(1.0, 1.0, 1.0));
        let b = a.expand_point(Vec3::new(5.0, -2.0, 0.5));
        assert!((b.max.x - 5.0).abs() < 1e-9);
        assert!((b.min.y - (-2.0)).abs() < 1e-9);
    }

    #[test]
    fn test_bvh_stats_empty() {
        let stats = bvh_stats(&[], 0);
        assert_eq!(stats.node_count, 0);
        assert_eq!(stats.max_depth, 0);
    }

    #[test]
    fn test_aabb_is_valid() {
        assert!(Aabb::new(Vec3::zero(), Vec3::new(1.0, 1.0, 1.0)).is_valid());
        assert!(!Aabb::new(Vec3::new(2.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0)).is_valid());
    }

    #[test]
    fn test_aabb_degenerate_surface_area() {
        let a = Aabb::new(Vec3::zero(), Vec3::new(1.0, 0.0, 1.0)); // flat
        let sa = a.surface_area();
        // 2*(1*0 + 0*1 + 1*1) = 2
        assert!((sa - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_build_preserves_all_prims_large() {
        let n = 100;
        let mut prims: Vec<BvhPrimitive> = (0..n).map(|i| {
            let x = (i as f64 * 7.3).sin() * 50.0;
            let y = (i as f64 * 3.1).cos() * 50.0;
            let z = (i as f64 * 1.7).sin() * 50.0;
            make_prim(x, y, z, 1.0, i)
        }).collect();
        let (nodes, ordered) = build_bvh(&mut prims, 4);
        assert!(!nodes.is_empty());
        assert_eq!(ordered.len(), n);
        let mut sorted = ordered.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), n);
    }
}
