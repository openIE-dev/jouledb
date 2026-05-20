//! 3D octree spatial index.
//!
//! A static, point-keyed octree over a fixed world bounding box. Used for
//! hierarchical / LOD queries the kd-tree and R-tree don't cover well:
//!
//! - **LOD descent**: walk to a target depth and read aggregate statistics
//!   per cell (occupancy, count, mean position) — the "is there *anything*
//!   in this region" query that drives splat / NeRF traversal.
//! - **Hierarchical filtering**: prune entire subtrees by cell-level
//!   predicates (occupancy threshold, semantic label, time bucket).
//! - **Cache-friendly streaming**: octree codes (Morton order) put nearby
//!   cells next to each other on disk, so range queries hit contiguous
//!   memory — critical for the splat / point cloud workload.
//!
//! The R-tree (`rtree3.rs`) wins for objects with shape; the kd-tree
//! (`kdtree3.rs`) wins for unbounded point clouds. The octree wins for
//! **bounded scenes with hierarchical structure** — exactly the robotics
//! and AR/VR cases the cascade targets.
//!
//! ## Design choices
//!
//! - **Static, bounded**: built once from a `Vec<(Point3, V)>` plus a
//!   world bbox. Out-of-bounds points are dropped (caller's responsibility
//!   to size the world correctly). This matches how perception systems
//!   work — you know the room, the warehouse, the scene volume.
//! - **Fixed max depth + bucket size**: subdivide a cell when it has more
//!   than `max_bucket` points, but stop at `max_depth` regardless.
//!   Prevents pathological deep recursion on coincident points.
//! - **Flat node storage** in a `Vec<Node>`, same shape as the kd-tree
//!   and R-tree. Children are an `Option<[u32; 8]>` rather than `Box`-ed.
//!
//! ## Energy story
//!
//! The cascade picks `OctreeQuery` (L4Communication, ~500 pJ) for spatial
//! lookups before falling back to LLM. A LOD descent reads exactly
//! `O(log_8 N)` cells — for a million-point scene, that's 7 cache lines.

use crate::types::spatial::{Bbox3, Point3};

// ============================================================================
// Tuning constants
// ============================================================================

/// Default maximum points in a leaf before splitting.
pub const DEFAULT_MAX_BUCKET: usize = 16;

/// Default maximum depth — 2^10 = 1024 cells per axis = 1.07 B leaves max.
/// Plenty for any realistic scene; the bucket limit kicks in long before.
pub const DEFAULT_MAX_DEPTH: u8 = 10;

// ============================================================================
// Tree
// ============================================================================

/// 3D octree mapping `Point3` → `V`.
#[derive(Debug, Clone)]
pub struct Octree3<V> {
    nodes: Vec<Node<V>>,
    /// World bbox — all queries are clipped to this.
    world: Bbox3,
    /// Index of the root, or `None` if empty.
    root: Option<usize>,
    max_bucket: usize,
    max_depth: u8,
}

#[derive(Debug, Clone)]
struct Node<V> {
    /// This cell's bbox (a sub-octant of `world`).
    bounds: Bbox3,
    /// Depth (root = 0).
    depth: u8,
    /// Either a leaf with points, or an internal with 8 children.
    kind: NodeKind<V>,
}

#[derive(Debug, Clone)]
enum NodeKind<V> {
    Leaf(Vec<(Point3, V)>),
    /// Eight children indexed by octant code (0..8). See `octant_of`.
    Internal([u32; 8]),
}

impl<V: Clone> Octree3<V> {
    /// Build an octree with default tuning.
    pub fn build(world: Bbox3, points: Vec<(Point3, V)>) -> Self {
        Self::build_with(world, points, DEFAULT_MAX_BUCKET, DEFAULT_MAX_DEPTH)
    }

    /// Build an octree with explicit `max_bucket` and `max_depth`.
    ///
    /// Out-of-bounds points are silently dropped.
    pub fn build_with(
        world: Bbox3,
        points: Vec<(Point3, V)>,
        max_bucket: usize,
        max_depth: u8,
    ) -> Self {
        let in_bounds: Vec<(Point3, V)> = points
            .into_iter()
            .filter(|(p, _)| world.contains(*p))
            .collect();

        if in_bounds.is_empty() {
            return Self {
                nodes: Vec::new(),
                world,
                root: None,
                max_bucket,
                max_depth,
            };
        }

        let mut nodes = Vec::new();
        let root = build_node(&mut nodes, world, 0, in_bounds, max_bucket, max_depth);
        Self {
            nodes,
            world,
            root: Some(root),
            max_bucket,
            max_depth,
        }
    }

    /// Root bounding box covering all points in the tree.
    pub fn world(&self) -> Bbox3 {
        self.world
    }

    /// Returns `true` if the tree contains no points.
    pub fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    /// Total number of points stored.
    pub fn len(&self) -> usize {
        self.nodes
            .iter()
            .filter_map(|n| match &n.kind {
                NodeKind::Leaf(v) => Some(v.len()),
                NodeKind::Internal(_) => None,
            })
            .sum()
    }

    /// All points inside `query` (inclusive).
    pub fn range(&self, query: Bbox3) -> Vec<(&Point3, &V)> {
        let mut hits = Vec::new();
        if let Some(root) = self.root {
            range_recursive(&self.nodes, root, &query, &mut hits);
        }
        hits
    }

    /// Single nearest neighbor of `query` within `max_dist` (inclusive).
    ///
    /// Returns `None` if no point is within `max_dist`. The bound is
    /// mandatory because unbounded NN on an octree degenerates without a
    /// best-first heap — for unbounded NN, use `KdTree3::nearest`.
    pub fn nearest_within(&self, query: Point3, max_dist: f64) -> Option<(&Point3, &V, f64)> {
        let root = self.root?;
        let mut best: Option<(usize, usize, f64)> = None; // (node_idx, leaf_idx, d²)
        let max_d2 = max_dist * max_dist;
        nearest_recursive(&self.nodes, root, query, max_d2, &mut best);
        best.map(|(node_i, leaf_i, d2)| {
            let n = &self.nodes[node_i];
            let leaf = match &n.kind {
                NodeKind::Leaf(v) => v,
                NodeKind::Internal(_) => unreachable!("nearest result must be a leaf"),
            };
            let (p, v) = &leaf[leaf_i];
            (p, v, d2.sqrt())
        })
    }

    /// Walk the tree at `target_depth` and return the bounds + point count
    /// of every cell at that depth (or its leaf if shallower).
    ///
    /// Used for LOD rendering and "is there anything in this region"
    /// hierarchical pre-filtering.
    pub fn lod_cells(&self, target_depth: u8) -> Vec<(Bbox3, usize)> {
        let mut out = Vec::new();
        if let Some(root) = self.root {
            lod_recursive(&self.nodes, root, target_depth, &mut out);
        }
        out
    }
}

// ============================================================================
// Build
// ============================================================================

fn build_node<V: Clone>(
    nodes: &mut Vec<Node<V>>,
    bounds: Bbox3,
    depth: u8,
    points: Vec<(Point3, V)>,
    max_bucket: usize,
    max_depth: u8,
) -> usize {
    // Stop conditions: small enough, or too deep.
    if points.len() <= max_bucket || depth >= max_depth {
        let idx = nodes.len();
        nodes.push(Node {
            bounds,
            depth,
            kind: NodeKind::Leaf(points),
        });
        return idx;
    }

    // Reserve our slot up front so child indices are stable.
    let my_idx = nodes.len();
    nodes.push(Node {
        bounds,
        depth,
        kind: NodeKind::Leaf(Vec::new()), // placeholder
    });

    // Split into 8 octants.
    let center = Point3::new(
        (bounds.min.x + bounds.max.x) * 0.5,
        (bounds.min.y + bounds.max.y) * 0.5,
        (bounds.min.z + bounds.max.z) * 0.5,
    );

    let mut buckets: [Vec<(Point3, V)>; 8] =
        [vec![], vec![], vec![], vec![], vec![], vec![], vec![], vec![]];
    for (p, v) in points {
        let oct = octant_of(p, center);
        buckets[oct as usize].push((p, v));
    }

    let mut children = [u32::MAX; 8];
    for (oct, bucket) in buckets.into_iter().enumerate() {
        let oct = oct as u8;
        let child_bounds = octant_bounds(bounds, center, oct);
        if bucket.is_empty() {
            // Empty leaf — still allocate so the array is dense. The cost
            // is one Node entry per empty octant, bounded by 8^max_depth in
            // the worst case but in practice ≪ that.
            let idx = nodes.len();
            nodes.push(Node {
                bounds: child_bounds,
                depth: depth + 1,
                kind: NodeKind::Leaf(Vec::new()),
            });
            children[oct as usize] = idx as u32;
        } else {
            let idx = build_node(
                nodes,
                child_bounds,
                depth + 1,
                bucket,
                max_bucket,
                max_depth,
            );
            children[oct as usize] = idx as u32;
        }
    }

    nodes[my_idx].kind = NodeKind::Internal(children);
    my_idx
}

/// Octant code: bit 0 = +x, bit 1 = +y, bit 2 = +z. 0..8.
fn octant_of(p: Point3, center: Point3) -> u8 {
    let mut o = 0u8;
    if p.x >= center.x {
        o |= 0b001;
    }
    if p.y >= center.y {
        o |= 0b010;
    }
    if p.z >= center.z {
        o |= 0b100;
    }
    o
}

fn octant_bounds(parent: Bbox3, center: Point3, oct: u8) -> Bbox3 {
    let min_x = if oct & 0b001 != 0 { center.x } else { parent.min.x };
    let max_x = if oct & 0b001 != 0 { parent.max.x } else { center.x };
    let min_y = if oct & 0b010 != 0 { center.y } else { parent.min.y };
    let max_y = if oct & 0b010 != 0 { parent.max.y } else { center.y };
    let min_z = if oct & 0b100 != 0 { center.z } else { parent.min.z };
    let max_z = if oct & 0b100 != 0 { parent.max.z } else { center.z };
    Bbox3::new(
        Point3::new(min_x, min_y, min_z),
        Point3::new(max_x, max_y, max_z),
    )
}

// ============================================================================
// Query
// ============================================================================

fn range_recursive<'a, V>(
    nodes: &'a [Node<V>],
    cur: usize,
    query: &Bbox3,
    hits: &mut Vec<(&'a Point3, &'a V)>,
) {
    let node = &nodes[cur];
    if !node.bounds.intersects(query) {
        return;
    }
    match &node.kind {
        NodeKind::Leaf(points) => {
            for (p, v) in points {
                if query.contains(*p) {
                    hits.push((p, v));
                }
            }
        }
        NodeKind::Internal(children) => {
            for &c in children {
                if c != u32::MAX {
                    range_recursive(nodes, c as usize, query, hits);
                }
            }
        }
    }
}

fn dist2(a: Point3, b: Point3) -> f64 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    let dz = a.z - b.z;
    dx * dx + dy * dy + dz * dz
}

/// Squared distance from `query` to the nearest point on `bounds`. Zero if
/// `query` is inside.
fn min_dist2_to_bbox(query: Point3, bounds: &Bbox3) -> f64 {
    let cx = query.x.clamp(bounds.min.x, bounds.max.x);
    let cy = query.y.clamp(bounds.min.y, bounds.max.y);
    let cz = query.z.clamp(bounds.min.z, bounds.max.z);
    dist2(query, Point3::new(cx, cy, cz))
}

fn nearest_recursive<V>(
    nodes: &[Node<V>],
    cur: usize,
    query: Point3,
    max_d2: f64,
    best: &mut Option<(usize, usize, f64)>,
) {
    let node = &nodes[cur];

    // Prune: if the closest possible point in this subtree is farther than
    // either the current best or the user's max_dist, skip.
    let cell_min = min_dist2_to_bbox(query, &node.bounds);
    let prune_at = match best {
        Some((_, _, d2)) => d2.min(max_d2),
        None => max_d2,
    };
    if cell_min > prune_at {
        return;
    }

    match &node.kind {
        NodeKind::Leaf(points) => {
            for (i, (p, _)) in points.iter().enumerate() {
                let d2 = dist2(*p, query);
                if d2 > max_d2 {
                    continue;
                }
                let better = match best {
                    Some((_, _, bd)) => d2 < *bd,
                    None => true,
                };
                if better {
                    *best = Some((cur, i, d2));
                }
            }
        }
        NodeKind::Internal(children) => {
            // Visit children nearest-first for better pruning.
            let mut order: Vec<(usize, f64)> = children
                .iter()
                .filter_map(|&c| {
                    if c == u32::MAX {
                        None
                    } else {
                        let cn = &nodes[c as usize];
                        Some((c as usize, min_dist2_to_bbox(query, &cn.bounds)))
                    }
                })
                .collect();
            order.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            for (child_idx, _) in order {
                nearest_recursive(nodes, child_idx, query, max_d2, best);
            }
        }
    }
}

fn lod_recursive<V>(
    nodes: &[Node<V>],
    cur: usize,
    target_depth: u8,
    out: &mut Vec<(Bbox3, usize)>,
) {
    let node = &nodes[cur];
    let count = subtree_count(nodes, cur);

    // Stop at the target depth, OR earlier if this is a leaf.
    if node.depth >= target_depth {
        if count > 0 {
            out.push((node.bounds, count));
        }
        return;
    }
    match &node.kind {
        NodeKind::Leaf(_) => {
            if count > 0 {
                out.push((node.bounds, count));
            }
        }
        NodeKind::Internal(children) => {
            for &c in children {
                if c != u32::MAX {
                    lod_recursive(nodes, c as usize, target_depth, out);
                }
            }
        }
    }
}

fn subtree_count<V>(nodes: &[Node<V>], cur: usize) -> usize {
    match &nodes[cur].kind {
        NodeKind::Leaf(points) => points.len(),
        NodeKind::Internal(children) => children
            .iter()
            .filter(|&&c| c != u32::MAX)
            .map(|&c| subtree_count(nodes, c as usize))
            .sum(),
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

    fn world() -> Bbox3 {
        Bbox3::new(p(0.0, 0.0, 0.0), p(100.0, 100.0, 100.0))
    }

    #[test]
    fn empty_octree() {
        let t: Octree3<u32> = Octree3::build(world(), Vec::new());
        assert!(t.is_empty());
        assert_eq!(t.len(), 0);
        assert!(t.range(world()).is_empty());
        assert!(t.nearest_within(p(50.0, 50.0, 50.0), 1000.0).is_none());
    }

    #[test]
    fn single_point() {
        let t = Octree3::build(world(), vec![(p(50.0, 50.0, 50.0), 42u32)]);
        assert_eq!(t.len(), 1);
        let r = t.range(world());
        assert_eq!(r.len(), 1);
        assert_eq!(*r[0].1, 42);
    }

    #[test]
    fn out_of_bounds_dropped() {
        let t = Octree3::build(
            world(),
            vec![
                (p(50.0, 50.0, 50.0), 1u32),
                (p(-1.0, 50.0, 50.0), 2u32), // outside on x
                (p(50.0, 200.0, 50.0), 3u32), // outside on y
            ],
        );
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn range_query_returns_points_in_box() {
        let pts: Vec<_> = (0..10)
            .flat_map(|x| {
                (0..10).flat_map(move |y| {
                    (0..10).map(move |z| {
                        (
                            p(x as f64 * 10.0, y as f64 * 10.0, z as f64 * 10.0),
                            (x, y, z),
                        )
                    })
                })
            })
            .collect();
        let t = Octree3::build(world(), pts);
        assert_eq!(t.len(), 1000);

        // Query a small region — points at multiples of 10, query [25..55]³
        // should match {30, 40, 50} on each axis = 27 points.
        let q = Bbox3::new(p(25.0, 25.0, 25.0), p(55.0, 55.0, 55.0));
        let hits = t.range(q);
        assert_eq!(hits.len(), 27);
    }

    #[test]
    fn nearest_within_finds_closest() {
        let pts = vec![
            (p(10.0, 10.0, 10.0), "a"),
            (p(50.0, 50.0, 50.0), "b"),
            (p(90.0, 90.0, 90.0), "c"),
        ];
        let t = Octree3::build(world(), pts);

        let (_, &v, d) = t.nearest_within(p(11.0, 10.0, 10.0), 100.0).unwrap();
        assert_eq!(v, "a");
        assert!((d - 1.0).abs() < 1e-9);

        let (_, &v, _) = t.nearest_within(p(48.0, 50.0, 50.0), 100.0).unwrap();
        assert_eq!(v, "b");

        // Too tight a radius → None.
        assert!(t.nearest_within(p(50.0, 50.0, 50.0), 0.5).is_some()); // (50,50,50) is exact
        assert!(t.nearest_within(p(60.0, 60.0, 60.0), 1.0).is_none());
    }

    #[test]
    fn nearest_within_matches_brute_force_on_random() {
        let pts: Vec<_> = (0..500)
            .map(|i| {
                let x = ((i * 9301 + 49297) % 99000) as f64 / 1000.0;
                let y = ((i * 5731 + 12345) % 99000) as f64 / 1000.0;
                let z = ((i * 3571 + 67890) % 99000) as f64 / 1000.0;
                (p(x, y, z), i as u32)
            })
            .collect();
        let pts_clone = pts.clone();
        let t = Octree3::build(world(), pts);

        for q in 0..30 {
            let qx = ((q * 1103 + 12345) % 99000) as f64 / 1000.0;
            let qy = ((q * 2087 + 65535) % 99000) as f64 / 1000.0;
            let qz = ((q * 4093 + 11111) % 99000) as f64 / 1000.0;
            let qp = p(qx, qy, qz);

            let (_, &octv, oct_d) = t.nearest_within(qp, 1000.0).unwrap();
            let (brute_v, brute_d2) = pts_clone
                .iter()
                .map(|(pt, v)| (*v, dist2(*pt, qp)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap();
            // Distances must agree.
            assert!(
                (oct_d - brute_d2.sqrt()).abs() < 1e-9,
                "octree dist {} vs brute dist {}",
                oct_d,
                brute_d2.sqrt()
            );
            assert_eq!(octv, brute_v);
        }
    }

    #[test]
    fn lod_cells_at_root_returns_one() {
        let pts: Vec<_> = (0..50).map(|i| (p(i as f64, 50.0, 50.0), i)).collect();
        let t = Octree3::build(world(), pts);
        let cells = t.lod_cells(0);
        assert_eq!(cells.len(), 1);
        assert_eq!(cells[0].1, 50);
    }

    #[test]
    fn lod_cells_at_depth_partitions_count() {
        // 100 points spread across all 8 octants.
        let pts: Vec<_> = (0..100)
            .map(|i| {
                let x = (i % 100) as f64;
                let y = ((i * 7) % 100) as f64;
                let z = ((i * 13) % 100) as f64;
                (p(x, y, z), i)
            })
            .collect();
        let total = pts.len();
        let t = Octree3::build(world(), pts);

        // At depth 1 we should have at most 8 cells, and their counts must
        // sum to the total point count.
        let cells = t.lod_cells(1);
        assert!(cells.len() <= 8);
        let sum: usize = cells.iter().map(|(_, c)| c).sum();
        assert_eq!(sum, total);
    }

    #[test]
    fn range_matches_brute_force_on_random() {
        let pts: Vec<_> = (0..400)
            .map(|i| {
                let x = ((i * 9301 + 49297) % 99000) as f64 / 1000.0;
                let y = ((i * 5731 + 12345) % 99000) as f64 / 1000.0;
                let z = ((i * 3571 + 67890) % 99000) as f64 / 1000.0;
                (p(x, y, z), i as u32)
            })
            .collect();
        let pts_clone = pts.clone();
        let t = Octree3::build(world(), pts);

        for q in 0..20 {
            let qx = ((q * 1103 + 12345) % 80000) as f64 / 1000.0;
            let qy = ((q * 2087 + 65535) % 80000) as f64 / 1000.0;
            let qz = ((q * 4093 + 11111) % 80000) as f64 / 1000.0;
            let bbox = Bbox3::new(p(qx, qy, qz), p(qx + 20.0, qy + 20.0, qz + 20.0));

            let mut tree_ids: Vec<u32> =
                t.range(bbox).iter().map(|(_, v)| **v).collect();
            tree_ids.sort();
            let mut brute_ids: Vec<u32> = pts_clone
                .iter()
                .filter(|(pt, _)| bbox.contains(*pt))
                .map(|(_, v)| *v)
                .collect();
            brute_ids.sort();

            assert_eq!(tree_ids, brute_ids, "octree range diverged on query {}", q);
        }
    }
}
