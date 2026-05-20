//! 3D k-d tree index for `Point3` keys.
//!
//! A static, balanced kd-tree over 3D points. Used as the primary spatial
//! index for "nearest object", "k-nearest neighbors", and "points in box"
//! queries — the operations the cascade picks before falling back to LLM
//! ("which graspable object is nearest the gripper?", "what's in the
//! reach envelope?").
//!
//! ## Why a separate index (not behind `Index`)
//!
//! The existing `Index` trait is byte-key oriented. A kd-tree's whole
//! advantage is comparing one f64 axis at a time and never serializing the
//! point — encoding to bytes on every comparison would obliterate the
//! cache-friendliness. So this lives as a native typed index, like
//! `MinHashLshIndex` but with `Point3` keys instead of byte slices.
//!
//! ## Build / query model
//!
//! - **Static**: built once from a `Vec<(Point3, V)>`, then queried.
//!   Insertions after `build` are O(N) re-balances — fine for the
//!   write-once / read-many scenes a robot perceives.
//! - **Median-split** on the longest axis at each level. Keeps the tree
//!   well-balanced even on non-uniform clouds (typical of LiDAR returns).
//! - **k-NN** uses a bounded max-heap; **range** uses a bbox descent.
//!
//! ## Energy story
//!
//! `flowG::Spatial3dOp::KnnQuery3d` and `RangeQuery3d` are classified as
//! `L4Communication` (memory-bound). One probe is `O(log N)` cache misses,
//! ~500 pJ on Apple Silicon. Compare to a single LLM token at ~1–10 mJ —
//! that's the cascade's 4–7 orders-of-magnitude advantage in a single op.

use crate::types::spatial::{Bbox3, Point3};

// ============================================================================
// Tree
// ============================================================================

/// 3D k-d tree mapping `Point3` → `V`.
///
/// Internally stored as a flat `Vec<Node<V>>` with index references —
/// no per-node allocation, no `Box`, friendly to the prefetcher.
#[derive(Debug, Clone)]
pub struct KdTree3<V> {
    nodes: Vec<Node<V>>,
    /// Index of the root, or `None` if empty.
    root: Option<usize>,
}

#[derive(Debug, Clone)]
struct Node<V> {
    point: Point3,
    value: V,
    /// Splitting axis: 0 = x, 1 = y, 2 = z.
    axis: u8,
    left: Option<usize>,
    right: Option<usize>,
}

impl<V: Clone> KdTree3<V> {
    /// Build a kd-tree from a set of `(point, value)` pairs.
    ///
    /// O(N log² N) build time using median-of-three pivoting on the
    /// longest-spread axis at each level. Empty input → empty tree.
    pub fn build(points: Vec<(Point3, V)>) -> Self {
        if points.is_empty() {
            return Self { nodes: Vec::new(), root: None };
        }

        let mut nodes = Vec::with_capacity(points.len());
        // Working buffer: indices into `points`. Points are moved out as
        // they're consumed by `build_recursive`.
        let mut points = points;
        let mut idx: Vec<usize> = (0..points.len()).collect();
        let root = build_recursive(&mut points, &mut idx[..], &mut nodes);
        Self { nodes, root: Some(root) }
    }

    /// Number of points in the tree.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Find the single nearest point to `query`.
    ///
    /// Returns `None` if the tree is empty.
    pub fn nearest(&self, query: Point3) -> Option<(&Point3, &V, f64)> {
        let root = self.root?;
        let mut best: Option<(usize, f64)> = None;
        nearest_recursive(&self.nodes, root, query, &mut best);
        best.map(|(i, d2)| (&self.nodes[i].point, &self.nodes[i].value, d2.sqrt()))
    }

    /// Find the `k` nearest neighbors of `query`, sorted by ascending distance.
    ///
    /// If `k >= self.len()`, returns all points.
    pub fn knn(&self, query: Point3, k: usize) -> Vec<(&Point3, &V, f64)> {
        if k == 0 || self.is_empty() {
            return Vec::new();
        }
        let Some(root) = self.root else { return Vec::new() };

        // Bounded max-heap of (squared distance, node index).
        // We use a Vec sorted by squared distance descending; the worst
        // candidate is always at index 0 for cheap displacement.
        // For small k (≤ 64, the typical robotics case), this is faster
        // than std::BinaryHeap because of cache locality and no allocation.
        let mut heap: Vec<(f64, usize)> = Vec::with_capacity(k);
        knn_recursive(&self.nodes, root, query, k, &mut heap);

        // Sort ascending by distance.
        heap.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        heap.into_iter()
            .map(|(d2, i)| {
                let n = &self.nodes[i];
                (&n.point, &n.value, d2.sqrt())
            })
            .collect()
    }

    /// Find all points inside `bbox` (inclusive).
    pub fn range(&self, bbox: Bbox3) -> Vec<(&Point3, &V)> {
        let mut hits = Vec::new();
        if let Some(root) = self.root {
            range_recursive(&self.nodes, root, &bbox, &mut hits);
        }
        hits.into_iter()
            .map(|i| {
                let n = &self.nodes[i];
                (&n.point, &n.value)
            })
            .collect()
    }
}

// ============================================================================
// Build
// ============================================================================

fn build_recursive<V>(
    points: &mut Vec<(Point3, V)>,
    idx: &mut [usize],
    nodes: &mut Vec<Node<V>>,
) -> usize
where
    V: Clone,
{
    debug_assert!(!idx.is_empty());

    // Pick splitting axis = axis of widest spread on this slice.
    let axis = widest_axis(points, idx);

    // Sort by that axis and pick the median.
    idx.sort_by(|&a, &b| {
        let pa = coord(&points[a].0, axis);
        let pb = coord(&points[b].0, axis);
        pa.partial_cmp(&pb).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mid = idx.len() / 2;
    let median_pi = idx[mid];

    // Reserve a slot for this node, then recurse to fill children first
    // so left/right indices are stable when we write them in.
    let my_index = nodes.len();
    nodes.push(Node {
        // Placeholder — overwritten below. We need the slot reserved so
        // children can use ascending indices.
        point: Point3::ORIGIN,
        // SAFETY: we always overwrite this slot before returning.
        // Use a clone of the median value as a placeholder; the median
        // will be moved in via direct assignment after the children
        // are built.
        value: points[median_pi].1.clone(),
        axis,
        left: None,
        right: None,
    });

    let left = if mid > 0 {
        Some(build_recursive(points, &mut idx[..mid], nodes))
    } else {
        None
    };
    let right = if mid + 1 < idx.len() {
        Some(build_recursive(points, &mut idx[mid + 1..], nodes))
    } else {
        None
    };

    // Now fill in the actual point + value for our slot.
    let n = &mut nodes[my_index];
    n.point = points[median_pi].0;
    n.value = points[median_pi].1.clone();
    n.left = left;
    n.right = right;

    my_index
}

fn widest_axis<V>(points: &[(Point3, V)], idx: &[usize]) -> u8 {
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for &i in idx {
        let p = &points[i].0;
        for (a, c) in [p.x, p.y, p.z].iter().enumerate() {
            if *c < min[a] {
                min[a] = *c;
            }
            if *c > max[a] {
                max[a] = *c;
            }
        }
    }
    let spreads = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
    if spreads[0] >= spreads[1] && spreads[0] >= spreads[2] {
        0
    } else if spreads[1] >= spreads[2] {
        1
    } else {
        2
    }
}

fn coord(p: &Point3, axis: u8) -> f64 {
    match axis {
        0 => p.x,
        1 => p.y,
        _ => p.z,
    }
}

// ============================================================================
// Query
// ============================================================================

fn dist2(a: Point3, b: Point3) -> f64 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    let dz = a.z - b.z;
    dx * dx + dy * dy + dz * dz
}

fn nearest_recursive<V>(
    nodes: &[Node<V>],
    cur: usize,
    query: Point3,
    best: &mut Option<(usize, f64)>,
) {
    let node = &nodes[cur];
    let d2 = dist2(node.point, query);

    let better = match best {
        Some((_, bd)) => d2 < *bd,
        None => true,
    };
    if better {
        *best = Some((cur, d2));
    }

    let diff = coord(&query, node.axis) - coord(&node.point, node.axis);
    let (near, far) = if diff < 0.0 {
        (node.left, node.right)
    } else {
        (node.right, node.left)
    };

    if let Some(n) = near {
        nearest_recursive(nodes, n, query, best);
    }

    // Only descend the far side if the splitting plane is closer than
    // the current best — the standard kd-tree pruning rule.
    let prune = match best {
        Some((_, bd)) => diff * diff < *bd,
        None => true,
    };
    if prune {
        if let Some(f) = far {
            nearest_recursive(nodes, f, query, best);
        }
    }
}

fn knn_recursive<V>(
    nodes: &[Node<V>],
    cur: usize,
    query: Point3,
    k: usize,
    heap: &mut Vec<(f64, usize)>,
) {
    let node = &nodes[cur];
    let d2 = dist2(node.point, query);

    push_bounded(heap, k, d2, cur);

    let diff = coord(&query, node.axis) - coord(&node.point, node.axis);
    let (near, far) = if diff < 0.0 {
        (node.left, node.right)
    } else {
        (node.right, node.left)
    };

    if let Some(n) = near {
        knn_recursive(nodes, n, query, k, heap);
    }

    let worst = if heap.len() == k { heap[0].0 } else { f64::INFINITY };
    if diff * diff < worst {
        if let Some(f) = far {
            knn_recursive(nodes, f, query, k, heap);
        }
    }
}

/// Bounded max-heap (slot 0 = worst). Push if there's room or this beats
/// the worst, then re-establish the worst at slot 0 by linear scan.
fn push_bounded(heap: &mut Vec<(f64, usize)>, k: usize, d2: f64, idx: usize) {
    if heap.len() < k {
        heap.push((d2, idx));
        // Find new worst.
        let worst_pos = heap
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.0.partial_cmp(&b.1.0).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        heap.swap(0, worst_pos);
    } else if d2 < heap[0].0 {
        heap[0] = (d2, idx);
        // Find new worst.
        let worst_pos = heap
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.0.partial_cmp(&b.1.0).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        heap.swap(0, worst_pos);
    }
}

fn range_recursive<V>(
    nodes: &[Node<V>],
    cur: usize,
    bbox: &Bbox3,
    hits: &mut Vec<usize>,
) {
    let node = &nodes[cur];
    if bbox.contains(node.point) {
        hits.push(cur);
    }

    // Descend left if the bbox extends below the splitting plane.
    let lo = coord(&bbox.min, node.axis);
    let hi = coord(&bbox.max, node.axis);
    let split = coord(&node.point, node.axis);

    if lo <= split {
        if let Some(l) = node.left {
            range_recursive(nodes, l, bbox, hits);
        }
    }
    if hi >= split {
        if let Some(r) = node.right {
            range_recursive(nodes, r, bbox, hits);
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn p(x: f64, y: f64, z: f64) -> Point3 {
        Point3::new(x, y, z)
    }

    #[test]
    fn empty_tree() {
        let t: KdTree3<u32> = KdTree3::build(Vec::new());
        assert!(t.is_empty());
        assert_eq!(t.len(), 0);
        assert!(t.nearest(p(0.0, 0.0, 0.0)).is_none());
        assert!(t.knn(p(0.0, 0.0, 0.0), 5).is_empty());
        assert!(t.range(Bbox3::point(p(0.0, 0.0, 0.0))).is_empty());
    }

    #[test]
    fn single_point() {
        let t = KdTree3::build(vec![(p(1.0, 2.0, 3.0), 42u32)]);
        let (pt, &v, d) = t.nearest(p(1.0, 2.0, 3.0)).unwrap();
        assert_eq!(*pt, p(1.0, 2.0, 3.0));
        assert_eq!(v, 42);
        assert_eq!(d, 0.0);
    }

    #[test]
    fn nearest_picks_correct_point() {
        let pts = vec![
            (p(0.0, 0.0, 0.0), "origin"),
            (p(10.0, 0.0, 0.0), "x10"),
            (p(0.0, 10.0, 0.0), "y10"),
            (p(0.0, 0.0, 10.0), "z10"),
            (p(5.0, 5.0, 5.0), "diag"),
        ];
        let t = KdTree3::build(pts);

        // Query near origin → "origin".
        let (_, &v, _) = t.nearest(p(0.5, 0.5, 0.5)).unwrap();
        assert_eq!(v, "origin");

        // Query near (10, 0, 0) → "x10".
        let (_, &v, _) = t.nearest(p(9.0, 0.5, 0.0)).unwrap();
        assert_eq!(v, "x10");

        // Query near diag.
        let (_, &v, _) = t.nearest(p(5.5, 5.5, 5.5)).unwrap();
        assert_eq!(v, "diag");
    }

    #[test]
    fn knn_returns_sorted_by_distance() {
        let pts: Vec<_> = (0..20)
            .map(|i| (p(i as f64, 0.0, 0.0), i as u32))
            .collect();
        let t = KdTree3::build(pts);

        let result = t.knn(p(5.0, 0.0, 0.0), 5);
        assert_eq!(result.len(), 5);
        // Distances must be ascending.
        for w in result.windows(2) {
            assert!(w[0].2 <= w[1].2, "distances not sorted: {:?}", result);
        }
        // The nearest should be exactly at (5,0,0) — distance 0.
        assert_eq!(result[0].2, 0.0);
        // Result IDs should be {3,4,5,6,7} in some distance-sorted order.
        let mut ids: Vec<u32> = result.iter().map(|&(_, &v, _)| v).collect();
        ids.sort();
        assert_eq!(ids, vec![3, 4, 5, 6, 7]);
    }

    #[test]
    fn knn_k_larger_than_size_returns_all() {
        let pts = vec![
            (p(0.0, 0.0, 0.0), 1u32),
            (p(1.0, 0.0, 0.0), 2u32),
            (p(2.0, 0.0, 0.0), 3u32),
        ];
        let t = KdTree3::build(pts);
        let result = t.knn(p(0.0, 0.0, 0.0), 100);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn range_query_returns_points_in_box() {
        let pts: Vec<_> = (0..10)
            .flat_map(|x| (0..10).map(move |y| (p(x as f64, y as f64, 0.0), (x, y))))
            .collect();
        let t = KdTree3::build(pts);

        let bbox = Bbox3::new(p(2.0, 3.0, -1.0), p(4.0, 5.0, 1.0));
        let mut hits: Vec<(i32, i32)> = t.range(bbox).into_iter().map(|(_, &v)| v).collect();
        hits.sort();
        // Inclusive on all sides → x ∈ [2,4], y ∈ [3,5] → 3×3 = 9 points.
        assert_eq!(hits.len(), 9);
        assert_eq!(hits[0], (2, 3));
        assert_eq!(hits[8], (4, 5));
    }

    #[test]
    fn range_query_empty_when_disjoint() {
        let pts = vec![
            (p(0.0, 0.0, 0.0), 1u32),
            (p(1.0, 1.0, 1.0), 2u32),
        ];
        let t = KdTree3::build(pts);
        let bbox = Bbox3::new(p(100.0, 100.0, 100.0), p(200.0, 200.0, 200.0));
        assert!(t.range(bbox).is_empty());
    }

    #[test]
    fn nearest_matches_brute_force_on_random() {
        // Pseudo-random but deterministic — no rng dependency.
        let pts: Vec<_> = (0..500)
            .map(|i| {
                let x = ((i * 9301 + 49297) % 233280) as f64 / 233.0;
                let y = ((i * 5731 + 12345) % 233280) as f64 / 233.0;
                let z = ((i * 3571 + 67890) % 233280) as f64 / 233.0;
                (p(x, y, z), i as u32)
            })
            .collect();
        let pts_clone = pts.clone();
        let t = KdTree3::build(pts);

        // Query 50 random points and confirm against O(N) brute force.
        for q in 0..50 {
            let qx = ((q * 1103 + 12345) % 233280) as f64 / 233.0;
            let qy = ((q * 2087 + 65535) % 233280) as f64 / 233.0;
            let qz = ((q * 4093 + 11111) % 233280) as f64 / 233.0;
            let qp = p(qx, qy, qz);

            let (_, &kdv, kd_dist) = t.nearest(qp).unwrap();

            let (brute_v, brute_d2) = pts_clone
                .iter()
                .map(|(pt, v)| (*v, dist2(*pt, qp)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap();

            // Distances must agree (squared, then sqrt'd by kd-tree).
            assert!(
                (kd_dist - brute_d2.sqrt()).abs() < 1e-9,
                "kd dist {} vs brute dist {}",
                kd_dist,
                brute_d2.sqrt()
            );
            // The chosen value must agree (or be a tie).
            assert_eq!(kdv, brute_v, "kd picked {} brute picked {}", kdv, brute_v);
        }
    }

    #[test]
    fn knn_matches_brute_force_on_random() {
        let pts: Vec<_> = (0..200)
            .map(|i| {
                let x = ((i * 9301 + 49297) % 233280) as f64 / 233.0;
                let y = ((i * 5731 + 12345) % 233280) as f64 / 233.0;
                let z = ((i * 3571 + 67890) % 233280) as f64 / 233.0;
                (p(x, y, z), i as u32)
            })
            .collect();
        let pts_clone = pts.clone();
        let t = KdTree3::build(pts);

        let qp = p(500.0, 500.0, 500.0);
        let kd_result = t.knn(qp, 10);
        let kd_ids: Vec<u32> = kd_result.iter().map(|&(_, &v, _)| v).collect();

        let mut brute: Vec<(u32, f64)> = pts_clone
            .iter()
            .map(|(pt, v)| (*v, dist2(*pt, qp)))
            .collect();
        brute.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let brute_ids: Vec<u32> = brute.iter().take(10).map(|(v, _)| *v).collect();

        assert_eq!(kd_ids, brute_ids);
    }
}
