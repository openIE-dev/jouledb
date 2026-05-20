//! 3D R-tree index for `Bbox3` keys.
//!
//! A bulk-loaded, in-memory R-tree over axis-aligned bounding boxes in 3D.
//! Used as the spatial index for "what overlaps this region", "what's inside
//! this volume", "ray vs scene" pre-pass, and the spatial-relation predicates
//! the query planner pushes (`OVERLAPS`, `CONTAINS`, `WITHIN`).
//!
//! ## Why R-tree (and not just a kd-tree of centroids)
//!
//! kd-tree-3 (`kdtree3.rs`) indexes points. R-tree indexes **extents** —
//! the natural key for objects with shape: meshes, gaussian splats,
//! collision volumes, scene-graph nodes with non-zero size. Without an
//! R-tree, every "does this collide with anything" query degenerates into
//! a kd-tree of centroids + a per-object box test, which is O(N) when the
//! query box is large.
//!
//! ## Variant: STR (Sort-Tile-Recursive) bulk load
//!
//! Build is O(N log N) via STR packing — sort by X, partition into vertical
//! slabs, sort each slab by Y, partition into rows, then sort each row by Z
//! and pack `M` boxes per leaf. This gives a near-optimal tree without the
//! incremental-insert split overhead. Insertions after build are O(log N)
//! single-path inserts that can degrade balance — for write-heavy workloads
//! we'd rebalance periodically, but the typical robotics scene is built once
//! per perception cycle and queried many times.
//!
//! ## Energy story
//!
//! `flowG::Spatial3dOp::SceneNeighborQuery` and `RayAabbIntersect` (when
//! batched over a scene) are L4Communication ops in the cascade. One R-tree
//! probe is `O(log_M N)` cache lines — ~500 pJ on Apple Silicon. The cascade
//! picks this before falling back to LLM ("which objects are within reach
//! of the gripper?" → R-tree probe, not GPT call).

use crate::types::spatial::{Bbox3, Point3};

// ============================================================================
// Tuning constants
// ============================================================================

/// Maximum entries per node. 16 keeps each node under one cache line on
/// 64-bit (16 × bbox(48) + 16 × payload(16) ≈ 1 KiB — fits in L1d easily).
const M: usize = 16;

/// Minimum entries per node — STR packing fills leaves to `M`, so this is
/// only relevant for the future incremental-insert path.
#[allow(dead_code)]
const MIN_FILL: usize = M / 2;

// ============================================================================
// Tree
// ============================================================================

/// 3D R-tree mapping `Bbox3` → `V`.
///
/// Internally a flat `Vec<Node<V>>` with index references — same shape as
/// `KdTree3` for cache friendliness. Leaves and internals are distinguished
/// by the `is_leaf` field rather than separate types.
#[derive(Debug, Clone)]
pub struct RTree3<V> {
    nodes: Vec<Node<V>>,
    /// Index of the root, or `None` if empty.
    root: Option<usize>,
}

#[derive(Debug, Clone)]
struct Node<V> {
    /// Tight bbox enclosing all entries in this subtree.
    mbr: Bbox3,
    is_leaf: bool,
    /// For leaves: payloads (one per entry).
    /// For internals: empty.
    leaf_values: Vec<(Bbox3, V)>,
    /// For internals: child node indices.
    /// For leaves: empty.
    children: Vec<usize>,
}

impl<V: Clone> RTree3<V> {
    /// Build an R-tree from a set of `(bbox, value)` pairs via STR packing.
    ///
    /// O(N log N) build, near-optimal locality. Empty input → empty tree.
    pub fn build(entries: Vec<(Bbox3, V)>) -> Self {
        if entries.is_empty() {
            return Self { nodes: Vec::new(), root: None };
        }

        let mut nodes: Vec<Node<V>> = Vec::new();

        // ── Phase 1: pack into leaves via STR ──────────────────────────
        //
        // STR (Leutenegger et al., 1997): sort by X, partition into
        // ⌈√(N/M)⌉ vertical slabs; sort each slab by Y, partition into
        // ⌈⁴√(N/M)⌉ rows per slab; sort each row by Z and pack M per leaf.
        // For 3D we add one more level, but the standard 2-level slab/row
        // approach already gives near-optimal trees in practice.
        let leaves = pack_leaves(entries, &mut nodes);

        // ── Phase 2: recursively pack internals up to the root ─────────
        let root = if leaves.len() == 1 {
            leaves[0]
        } else {
            pack_internals(leaves, &mut nodes)
        };

        Self { nodes, root: Some(root) }
    }

    /// Number of leaf entries (≠ number of nodes).
    pub fn len(&self) -> usize {
        match self.root {
            None => 0,
            Some(_) => self.nodes.iter()
                .filter(|n| n.is_leaf)
                .map(|n| n.leaf_values.len())
                .sum(),
        }
    }

    /// Returns `true` if the tree contains no entries.
    pub fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    /// All entries whose bbox intersects `query`.
    pub fn intersects(&self, query: Bbox3) -> Vec<(&Bbox3, &V)> {
        let mut hits = Vec::new();
        if let Some(root) = self.root {
            intersects_recursive(&self.nodes, root, &query, &mut hits);
        }
        hits
    }

    /// All entries whose bbox contains `point` (inclusive).
    pub fn contains_point(&self, point: Point3) -> Vec<(&Bbox3, &V)> {
        let mut hits = Vec::new();
        if let Some(root) = self.root {
            contains_point_recursive(&self.nodes, root, point, &mut hits);
        }
        hits
    }

    /// All entries whose bbox is fully inside `query`.
    pub fn within(&self, query: Bbox3) -> Vec<(&Bbox3, &V)> {
        let mut hits = Vec::new();
        if let Some(root) = self.root {
            within_recursive(&self.nodes, root, &query, &mut hits);
        }
        hits
    }
}

// ============================================================================
// STR bulk load
// ============================================================================

fn pack_leaves<V: Clone>(
    mut entries: Vec<(Bbox3, V)>,
    nodes: &mut Vec<Node<V>>,
) -> Vec<usize> {
    let n = entries.len();
    let leaf_count = n.div_ceil(M);
    // Number of vertical slabs ≈ ⌈√(leaf_count)⌉. Round up so each slab
    // ends up with ≤ ⌈√(leaf_count)⌉ leaves' worth of entries.
    let slab_count = (leaf_count as f64).sqrt().ceil() as usize;
    let slab_count = slab_count.max(1);
    let slab_size = n.div_ceil(slab_count) * M / M.max(1); // entries per slab, rounded
    let slab_size = ((n + slab_count - 1) / slab_count).max(1);

    // Sort by centroid X.
    entries.sort_by(|a, b| centroid_x(&a.0).partial_cmp(&centroid_x(&b.0)).unwrap());

    let mut leaves = Vec::with_capacity(leaf_count);

    for slab in entries.chunks_mut(slab_size) {
        // Sort this slab by centroid Y.
        slab.sort_by(|a, b| centroid_y(&a.0).partial_cmp(&centroid_y(&b.0)).unwrap());

        // Partition slab into rows of M entries (each row = one leaf).
        // For 3D we'd do another sort+partition by Z within each row, but
        // since each row is already ≤ M we just sort by Z and pack.
        for row in slab.chunks_mut(M) {
            row.sort_by(|a, b| centroid_z(&a.0).partial_cmp(&centroid_z(&b.0)).unwrap());

            let leaf_values: Vec<(Bbox3, V)> = row.to_vec();
            let mbr = compute_mbr(leaf_values.iter().map(|(b, _)| *b));
            let leaf_idx = nodes.len();
            nodes.push(Node {
                mbr,
                is_leaf: true,
                leaf_values,
                children: Vec::new(),
            });
            leaves.push(leaf_idx);
        }
    }

    leaves
}

fn pack_internals<V: Clone>(
    mut current_level: Vec<usize>,
    nodes: &mut Vec<Node<V>>,
) -> usize {
    while current_level.len() > 1 {
        let n = current_level.len();
        let next_count = n.div_ceil(M);
        let slab_count = (next_count as f64).sqrt().ceil() as usize;
        let slab_count = slab_count.max(1);
        let slab_size = ((n + slab_count - 1) / slab_count).max(1);

        // Sort by MBR centroid X.
        current_level.sort_by(|&a, &b| {
            centroid_x(&nodes[a].mbr)
                .partial_cmp(&centroid_x(&nodes[b].mbr))
                .unwrap()
        });

        let mut next_level = Vec::with_capacity(next_count);

        // We need to slice current_level into slabs but keep using the same
        // backing storage; collect into a temporary then iterate.
        let level_snapshot: Vec<usize> = current_level.clone();
        for slab in level_snapshot.chunks(slab_size) {
            let mut slab_vec = slab.to_vec();
            // Sort slab by MBR centroid Y.
            slab_vec.sort_by(|&a, &b| {
                centroid_y(&nodes[a].mbr)
                    .partial_cmp(&centroid_y(&nodes[b].mbr))
                    .unwrap()
            });

            for row in slab_vec.chunks(M) {
                let mut row_vec = row.to_vec();
                row_vec.sort_by(|&a, &b| {
                    centroid_z(&nodes[a].mbr)
                        .partial_cmp(&centroid_z(&nodes[b].mbr))
                        .unwrap()
                });

                let mbr = compute_mbr(row_vec.iter().map(|&i| nodes[i].mbr));
                let internal_idx = nodes.len();
                nodes.push(Node {
                    mbr,
                    is_leaf: false,
                    leaf_values: Vec::new(),
                    children: row_vec,
                });
                next_level.push(internal_idx);
            }
        }

        current_level = next_level;
    }

    current_level[0]
}

fn centroid_x(b: &Bbox3) -> f64 {
    (b.min.x + b.max.x) * 0.5
}
fn centroid_y(b: &Bbox3) -> f64 {
    (b.min.y + b.max.y) * 0.5
}
fn centroid_z(b: &Bbox3) -> f64 {
    (b.min.z + b.max.z) * 0.5
}

fn compute_mbr(boxes: impl Iterator<Item = Bbox3>) -> Bbox3 {
    let mut iter = boxes;
    let first = iter.next().expect("compute_mbr: empty iterator");
    let mut min = first.min;
    let mut max = first.max;
    for b in iter {
        if b.min.x < min.x {
            min.x = b.min.x;
        }
        if b.min.y < min.y {
            min.y = b.min.y;
        }
        if b.min.z < min.z {
            min.z = b.min.z;
        }
        if b.max.x > max.x {
            max.x = b.max.x;
        }
        if b.max.y > max.y {
            max.y = b.max.y;
        }
        if b.max.z > max.z {
            max.z = b.max.z;
        }
    }
    Bbox3::new(min, max)
}

// ============================================================================
// Query
// ============================================================================

fn intersects_recursive<'a, V>(
    nodes: &'a [Node<V>],
    cur: usize,
    query: &Bbox3,
    hits: &mut Vec<(&'a Bbox3, &'a V)>,
) {
    let node = &nodes[cur];
    if !node.mbr.intersects(query) {
        return;
    }
    if node.is_leaf {
        for (b, v) in &node.leaf_values {
            if b.intersects(query) {
                hits.push((b, v));
            }
        }
    } else {
        for &c in &node.children {
            intersects_recursive(nodes, c, query, hits);
        }
    }
}

fn contains_point_recursive<'a, V>(
    nodes: &'a [Node<V>],
    cur: usize,
    point: Point3,
    hits: &mut Vec<(&'a Bbox3, &'a V)>,
) {
    let node = &nodes[cur];
    if !node.mbr.contains(point) {
        return;
    }
    if node.is_leaf {
        for (b, v) in &node.leaf_values {
            if b.contains(point) {
                hits.push((b, v));
            }
        }
    } else {
        for &c in &node.children {
            contains_point_recursive(nodes, c, point, hits);
        }
    }
}

fn within_recursive<'a, V>(
    nodes: &'a [Node<V>],
    cur: usize,
    query: &Bbox3,
    hits: &mut Vec<(&'a Bbox3, &'a V)>,
) {
    let node = &nodes[cur];
    // If the subtree's MBR doesn't even overlap the query, prune.
    if !node.mbr.intersects(query) {
        return;
    }
    if node.is_leaf {
        for (b, v) in &node.leaf_values {
            // Fully inside = query contains both corners of b.
            if query.contains(b.min) && query.contains(b.max) {
                hits.push((b, v));
            }
        }
    } else {
        for &c in &node.children {
            within_recursive(nodes, c, query, hits);
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
    fn b(min: (f64, f64, f64), max: (f64, f64, f64)) -> Bbox3 {
        Bbox3::new(p(min.0, min.1, min.2), p(max.0, max.1, max.2))
    }

    #[test]
    fn empty_tree() {
        let t: RTree3<u32> = RTree3::build(Vec::new());
        assert!(t.is_empty());
        assert_eq!(t.len(), 0);
        assert!(t.intersects(b((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))).is_empty());
    }

    #[test]
    fn single_box() {
        let entries = vec![(b((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), 42u32)];
        let t = RTree3::build(entries);
        assert_eq!(t.len(), 1);
        let hits = t.intersects(b((0.5, 0.5, 0.5), (2.0, 2.0, 2.0)));
        assert_eq!(hits.len(), 1);
        assert_eq!(*hits[0].1, 42);
    }

    #[test]
    fn intersects_basic() {
        let entries = vec![
            (b((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), 1u32),
            (b((2.0, 2.0, 2.0), (3.0, 3.0, 3.0)), 2u32),
            (b((0.5, 0.5, 0.5), (2.5, 2.5, 2.5)), 3u32),
        ];
        let t = RTree3::build(entries);

        // Query that should hit boxes 1 and 3 (both overlap [0,1]³).
        let q = b((0.0, 0.0, 0.0), (1.5, 1.5, 1.5));
        let mut ids: Vec<u32> = t.intersects(q).iter().map(|&(_, v)| *v).collect();
        ids.sort();
        assert_eq!(ids, vec![1, 3]);
    }

    #[test]
    fn intersects_empty_query() {
        let entries = vec![
            (b((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), 1u32),
            (b((10.0, 10.0, 10.0), (11.0, 11.0, 11.0)), 2u32),
        ];
        let t = RTree3::build(entries);
        let q = b((100.0, 100.0, 100.0), (200.0, 200.0, 200.0));
        assert!(t.intersects(q).is_empty());
    }

    #[test]
    fn contains_point_basic() {
        let entries = vec![
            (b((0.0, 0.0, 0.0), (10.0, 10.0, 10.0)), "outer"),
            (b((1.0, 1.0, 1.0), (2.0, 2.0, 2.0)), "inner_a"),
            (b((5.0, 5.0, 5.0), (6.0, 6.0, 6.0)), "inner_b"),
            (b((20.0, 20.0, 20.0), (30.0, 30.0, 30.0)), "far"),
        ];
        let t = RTree3::build(entries);

        // (1.5, 1.5, 1.5) is inside outer + inner_a.
        let mut ids: Vec<&str> = t
            .contains_point(p(1.5, 1.5, 1.5))
            .iter()
            .map(|&(_, &v)| v)
            .collect();
        ids.sort();
        assert_eq!(ids, vec!["inner_a", "outer"]);

        // (5.5, 5.5, 5.5) is inside outer + inner_b.
        let mut ids: Vec<&str> = t
            .contains_point(p(5.5, 5.5, 5.5))
            .iter()
            .map(|&(_, &v)| v)
            .collect();
        ids.sort();
        assert_eq!(ids, vec!["inner_b", "outer"]);

        // (25, 25, 25) is inside far only.
        let ids: Vec<&str> = t
            .contains_point(p(25.0, 25.0, 25.0))
            .iter()
            .map(|&(_, &v)| v)
            .collect();
        assert_eq!(ids, vec!["far"]);

        // (50, 50, 50) is inside nothing.
        assert!(t.contains_point(p(50.0, 50.0, 50.0)).is_empty());
    }

    #[test]
    fn within_basic() {
        let entries = vec![
            (b((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), "small"),
            (b((0.0, 0.0, 0.0), (10.0, 10.0, 10.0)), "big"),
            (b((2.0, 2.0, 2.0), (3.0, 3.0, 3.0)), "mid"),
        ];
        let t = RTree3::build(entries);

        // Query [0..5]³ — `small` and `mid` are fully inside; `big` is not.
        let q = b((0.0, 0.0, 0.0), (5.0, 5.0, 5.0));
        let mut ids: Vec<&str> = t.within(q).iter().map(|&(_, &v)| v).collect();
        ids.sort();
        assert_eq!(ids, vec!["mid", "small"]);
    }

    #[test]
    fn build_handles_many_boxes() {
        // 1000 unit cubes on a 10×10×10 grid.
        let entries: Vec<_> = (0..10)
            .flat_map(|x| {
                (0..10).flat_map(move |y| {
                    (0..10).map(move |z| {
                        (
                            b(
                                (x as f64, y as f64, z as f64),
                                (x as f64 + 0.5, y as f64 + 0.5, z as f64 + 0.5),
                            ),
                            (x, y, z),
                        )
                    })
                })
            })
            .collect();
        let t = RTree3::build(entries);
        assert_eq!(t.len(), 1000);

        // Query [2..4]³ — cube `(x,y,z)` spans `[x, x+0.5]³`, so it
        // intersects iff `x ∈ {2, 3, 4}` (cube at x=1 ends at 1.5; cube
        // at x=4 starts exactly at the query max, which is inclusive).
        // → 3 × 3 × 3 = 27 cubes.
        let q = b((2.0, 2.0, 2.0), (4.0, 4.0, 4.0));
        let hits = t.intersects(q);
        assert_eq!(hits.len(), 27, "expected 27 cubes in [2..4]³");
    }

    #[test]
    fn intersects_matches_brute_force_on_random() {
        // 300 pseudo-random boxes.
        let entries: Vec<_> = (0..300)
            .map(|i| {
                let x = ((i * 9301 + 49297) % 233280) as f64 / 1000.0;
                let y = ((i * 5731 + 12345) % 233280) as f64 / 1000.0;
                let z = ((i * 3571 + 67890) % 233280) as f64 / 1000.0;
                let sz = 5.0 + ((i * 1009) % 30) as f64;
                (b((x, y, z), (x + sz, y + sz, z + sz)), i as u32)
            })
            .collect();
        let entries_clone = entries.clone();
        let t = RTree3::build(entries);

        // 20 random query boxes.
        for q in 0..20 {
            let qx = ((q * 1103 + 12345) % 233280) as f64 / 1000.0;
            let qy = ((q * 2087 + 65535) % 233280) as f64 / 1000.0;
            let qz = ((q * 4093 + 11111) % 233280) as f64 / 1000.0;
            let qb = b((qx, qy, qz), (qx + 50.0, qy + 50.0, qz + 50.0));

            let mut tree_ids: Vec<u32> =
                t.intersects(qb).iter().map(|&(_, v)| *v).collect();
            tree_ids.sort();

            let mut brute_ids: Vec<u32> = entries_clone
                .iter()
                .filter(|(bb, _)| bb.intersects(&qb))
                .map(|(_, v)| *v)
                .collect();
            brute_ids.sort();

            assert_eq!(
                tree_ids, brute_ids,
                "rtree vs brute force diverged on query {}",
                q
            );
        }
    }
}
