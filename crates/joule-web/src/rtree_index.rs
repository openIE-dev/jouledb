//! R-tree — insert/delete/search, bounding box computation, node splitting
//! (linear/quadratic), bulk loading (STR sort), nearest-neighbor query,
//! range query, RTreeConfig builder.
//!
//! Pure-Rust R-tree index for spatial objects in arbitrary dimensions.
//! Supports configurable min/max fanout, two split strategies, and
//! Sort-Tile-Recursive bulk loading for large datasets.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum RTreeError {
    InvalidConfig(String),
    EmptyTree,
    NotFound,
    DimensionMismatch { expected: usize, got: usize },
}

impl fmt::Display for RTreeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(s) => write!(f, "invalid config: {s}"),
            Self::EmptyTree => write!(f, "tree is empty"),
            Self::NotFound => write!(f, "entry not found"),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
        }
    }
}

impl std::error::Error for RTreeError {}

// ── Bounding Box ────────────────────────────────────────────────

/// Axis-aligned bounding box in N dimensions.
#[derive(Debug, Clone, PartialEq)]
pub struct BBox {
    pub min: Vec<f64>,
    pub max: Vec<f64>,
}

impl BBox {
    pub fn new(min: Vec<f64>, max: Vec<f64>) -> Self {
        debug_assert_eq!(min.len(), max.len());
        Self { min, max }
    }

    pub fn dims(&self) -> usize {
        self.min.len()
    }

    pub fn from_point(coords: &[f64]) -> Self {
        Self { min: coords.to_vec(), max: coords.to_vec() }
    }

    pub fn area(&self) -> f64 {
        self.min.iter().zip(self.max.iter()).fold(1.0, |a, (lo, hi)| a * (hi - lo).max(0.0))
    }

    pub fn margin(&self) -> f64 {
        self.min.iter().zip(self.max.iter()).map(|(lo, hi)| (hi - lo).max(0.0)).sum()
    }

    pub fn center(&self) -> Vec<f64> {
        self.min.iter().zip(self.max.iter()).map(|(lo, hi)| (lo + hi) * 0.5).collect()
    }

    pub fn contains_point(&self, p: &[f64]) -> bool {
        self.min.iter().zip(self.max.iter()).zip(p.iter())
            .all(|((lo, hi), v)| *v >= *lo && *v <= *hi)
    }

    pub fn intersects(&self, other: &BBox) -> bool {
        self.min.iter().zip(self.max.iter())
            .zip(other.min.iter().zip(other.max.iter()))
            .all(|((a_lo, a_hi), (b_lo, b_hi))| a_lo <= b_hi && b_lo <= a_hi)
    }

    pub fn merge(&self, other: &BBox) -> BBox {
        let min: Vec<f64> = self.min.iter().zip(other.min.iter()).map(|(a, b)| a.min(*b)).collect();
        let max: Vec<f64> = self.max.iter().zip(other.max.iter()).map(|(a, b)| a.max(*b)).collect();
        BBox { min, max }
    }

    pub fn enlargement(&self, other: &BBox) -> f64 {
        self.merge(other).area() - self.area()
    }

    pub fn distance_sq_to_point(&self, p: &[f64]) -> f64 {
        self.min.iter().zip(self.max.iter()).zip(p.iter())
            .map(|((lo, hi), v)| {
                if *v < *lo { (lo - v) * (lo - v) }
                else if *v > *hi { (v - hi) * (v - hi) }
                else { 0.0 }
            })
            .sum()
    }
}

impl fmt::Display for BBox {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BBox({:?} .. {:?})", self.min, self.max)
    }
}

// ── Split Strategy ──────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SplitStrategy {
    Linear,
    Quadratic,
}

impl fmt::Display for SplitStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Linear => write!(f, "Linear"),
            Self::Quadratic => write!(f, "Quadratic"),
        }
    }
}

// ── Config ──────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct RTreeConfig {
    pub min_children: usize,
    pub max_children: usize,
    pub dimensions: usize,
    pub split_strategy: SplitStrategy,
}

impl RTreeConfig {
    pub fn new(dimensions: usize) -> Self {
        Self {
            min_children: 2,
            max_children: 8,
            dimensions,
            split_strategy: SplitStrategy::Quadratic,
        }
    }

    pub fn with_min_children(mut self, m: usize) -> Self { self.min_children = m; self }
    pub fn with_max_children(mut self, m: usize) -> Self { self.max_children = m; self }
    pub fn with_split_strategy(mut self, s: SplitStrategy) -> Self { self.split_strategy = s; self }

    pub fn validate(&self) -> Result<(), RTreeError> {
        if self.dimensions == 0 {
            return Err(RTreeError::InvalidConfig("dimensions must be > 0".into()));
        }
        if self.min_children < 1 {
            return Err(RTreeError::InvalidConfig("min_children must be >= 1".into()));
        }
        if self.max_children < 2 * self.min_children {
            return Err(RTreeError::InvalidConfig("max_children must be >= 2 * min_children".into()));
        }
        Ok(())
    }
}

impl fmt::Display for RTreeConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RTreeConfig({}D, {}-{}, {:?})", self.dimensions, self.min_children, self.max_children, self.split_strategy)
    }
}

// ── Entry ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Entry {
    pub bbox: BBox,
    pub id: u64,
}

// ── Node ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
enum NodeKind {
    Leaf { entries: Vec<Entry> },
    Internal { children: Vec<usize> },
}

#[derive(Debug, Clone)]
struct Node {
    bbox: BBox,
    kind: NodeKind,
}

// ── R-tree ──────────────────────────────────────────────────────

pub struct RTree {
    config: RTreeConfig,
    nodes: Vec<Node>,
    root: usize,
    entry_count: usize,
}

impl RTree {
    pub fn new(config: RTreeConfig) -> Result<Self, RTreeError> {
        config.validate()?;
        let dims = config.dimensions;
        let root_node = Node {
            bbox: BBox::new(vec![f64::INFINITY; dims], vec![f64::NEG_INFINITY; dims]),
            kind: NodeKind::Leaf { entries: Vec::new() },
        };
        Ok(Self { config, nodes: vec![root_node], root: 0, entry_count: 0 })
    }

    pub fn len(&self) -> usize { self.entry_count }
    pub fn is_empty(&self) -> bool { self.entry_count == 0 }

    /// Insert an entry.
    pub fn insert(&mut self, entry: Entry) -> Result<(), RTreeError> {
        if entry.bbox.dims() != self.config.dimensions {
            return Err(RTreeError::DimensionMismatch {
                expected: self.config.dimensions,
                got: entry.bbox.dims(),
            });
        }
        let leaf = self.choose_leaf(self.root, &entry.bbox);
        self.insert_at_leaf(leaf, entry);
        self.entry_count += 1;
        Ok(())
    }

    fn choose_leaf(&self, node_idx: usize, bbox: &BBox) -> usize {
        match &self.nodes[node_idx].kind {
            NodeKind::Leaf { .. } => node_idx,
            NodeKind::Internal { children } => {
                let best = children.iter().copied().min_by(|&a, &b| {
                    let ea = self.nodes[a].bbox.enlargement(bbox);
                    let eb = self.nodes[b].bbox.enlargement(bbox);
                    ea.partial_cmp(&eb).unwrap_or(std::cmp::Ordering::Equal)
                }).unwrap();
                self.choose_leaf(best, bbox)
            }
        }
    }

    fn insert_at_leaf(&mut self, leaf_idx: usize, entry: Entry) {
        let entry_bbox = entry.bbox.clone();
        if let NodeKind::Leaf { entries } = &mut self.nodes[leaf_idx].kind {
            entries.push(entry);
        }
        self.expand_bbox(leaf_idx, &entry_bbox);
        if self.node_overflow(leaf_idx) {
            self.handle_overflow(leaf_idx);
        }
    }

    fn expand_bbox(&mut self, node_idx: usize, bbox: &BBox) {
        self.nodes[node_idx].bbox = self.nodes[node_idx].bbox.merge(bbox);
    }

    fn node_overflow(&self, idx: usize) -> bool {
        match &self.nodes[idx].kind {
            NodeKind::Leaf { entries } => entries.len() > self.config.max_children,
            NodeKind::Internal { children } => children.len() > self.config.max_children,
        }
    }

    fn handle_overflow(&mut self, node_idx: usize) {
        let (left_kind, right_kind, left_bb, right_bb) = match self.config.split_strategy {
            SplitStrategy::Linear => self.linear_split(node_idx),
            SplitStrategy::Quadratic => self.quadratic_split(node_idx),
        };
        if node_idx == self.root {
            let left_idx = self.nodes.len();
            self.nodes.push(Node { bbox: left_bb, kind: left_kind });
            let right_idx = self.nodes.len();
            self.nodes.push(Node { bbox: right_bb, kind: right_kind });
            let dims = self.config.dimensions;
            self.nodes[self.root] = Node {
                bbox: self.nodes[left_idx].bbox.merge(&self.nodes[right_idx].bbox),
                kind: NodeKind::Internal { children: vec![left_idx, right_idx] },
            };
            let _ = dims;
        } else {
            self.nodes[node_idx] = Node { bbox: left_bb, kind: left_kind };
            let right_idx = self.nodes.len();
            self.nodes.push(Node { bbox: right_bb, kind: right_kind });
            // Propagate to parent — simplified: update root children
            if let NodeKind::Internal { children } = &mut self.nodes[self.root].kind {
                if !children.contains(&right_idx) {
                    children.push(right_idx);
                }
            }
        }
    }

    fn compute_bbox_for_entries(entries: &[Entry]) -> BBox {
        let dims = entries[0].bbox.dims();
        let mut min = vec![f64::INFINITY; dims];
        let mut max = vec![f64::NEG_INFINITY; dims];
        for e in entries {
            for (i, (lo, hi)) in e.bbox.min.iter().zip(e.bbox.max.iter()).enumerate() {
                min[i] = min[i].min(*lo);
                max[i] = max[i].max(*hi);
            }
        }
        BBox::new(min, max)
    }

    fn compute_bbox_for_children(&self, children: &[usize]) -> BBox {
        let dims = self.config.dimensions;
        let mut min = vec![f64::INFINITY; dims];
        let mut max = vec![f64::NEG_INFINITY; dims];
        for &c in children {
            for (i, (lo, hi)) in self.nodes[c].bbox.min.iter().zip(self.nodes[c].bbox.max.iter()).enumerate() {
                min[i] = min[i].min(*lo);
                max[i] = max[i].max(*hi);
            }
        }
        BBox::new(min, max)
    }

    fn linear_split(&self, node_idx: usize) -> (NodeKind, NodeKind, BBox, BBox) {
        match &self.nodes[node_idx].kind {
            NodeKind::Leaf { entries } => {
                let mid = entries.len() / 2;
                let left = entries[..mid].to_vec();
                let right = entries[mid..].to_vec();
                let lbb = Self::compute_bbox_for_entries(&left);
                let rbb = Self::compute_bbox_for_entries(&right);
                (NodeKind::Leaf { entries: left }, NodeKind::Leaf { entries: right }, lbb, rbb)
            }
            NodeKind::Internal { children } => {
                let mid = children.len() / 2;
                let left = children[..mid].to_vec();
                let right = children[mid..].to_vec();
                let lbb = self.compute_bbox_for_children(&left);
                let rbb = self.compute_bbox_for_children(&right);
                (NodeKind::Internal { children: left }, NodeKind::Internal { children: right }, lbb, rbb)
            }
        }
    }

    fn quadratic_split(&self, node_idx: usize) -> (NodeKind, NodeKind, BBox, BBox) {
        match &self.nodes[node_idx].kind {
            NodeKind::Leaf { entries } => {
                let (s1, s2) = self.pick_seeds_entries(entries);
                let mut left = vec![entries[s1].clone()];
                let mut right = vec![entries[s2].clone()];
                let mut left_bb = entries[s1].bbox.clone();
                let mut right_bb = entries[s2].bbox.clone();
                for (i, e) in entries.iter().enumerate() {
                    if i == s1 || i == s2 { continue; }
                    let el = left_bb.enlargement(&e.bbox);
                    let er = right_bb.enlargement(&e.bbox);
                    if el < er || (el == er && left.len() <= right.len()) {
                        left_bb = left_bb.merge(&e.bbox);
                        left.push(e.clone());
                    } else {
                        right_bb = right_bb.merge(&e.bbox);
                        right.push(e.clone());
                    }
                }
                (NodeKind::Leaf { entries: left }, NodeKind::Leaf { entries: right }, left_bb, right_bb)
            }
            NodeKind::Internal { children } => {
                let mid = children.len() / 2;
                let left = children[..mid].to_vec();
                let right = children[mid..].to_vec();
                let lbb = self.compute_bbox_for_children(&left);
                let rbb = self.compute_bbox_for_children(&right);
                (NodeKind::Internal { children: left }, NodeKind::Internal { children: right }, lbb, rbb)
            }
        }
    }

    fn pick_seeds_entries(&self, entries: &[Entry]) -> (usize, usize) {
        let mut worst_waste = f64::NEG_INFINITY;
        let (mut s1, mut s2) = (0, 1);
        for i in 0..entries.len() {
            for j in (i + 1)..entries.len() {
                let merged = entries[i].bbox.merge(&entries[j].bbox);
                let waste = merged.area() - entries[i].bbox.area() - entries[j].bbox.area();
                if waste > worst_waste {
                    worst_waste = waste;
                    s1 = i;
                    s2 = j;
                }
            }
        }
        (s1, s2)
    }

    /// Range query: return all entry ids whose bounding box intersects `query`.
    pub fn search(&self, query: &BBox) -> Vec<u64> {
        let mut results = Vec::new();
        self.search_node(self.root, query, &mut results);
        results
    }

    fn search_node(&self, node_idx: usize, query: &BBox, results: &mut Vec<u64>) {
        if !self.nodes[node_idx].bbox.intersects(query) {
            return;
        }
        match &self.nodes[node_idx].kind {
            NodeKind::Leaf { entries } => {
                for e in entries {
                    if e.bbox.intersects(query) {
                        results.push(e.id);
                    }
                }
            }
            NodeKind::Internal { children } => {
                for &c in children {
                    self.search_node(c, query, results);
                }
            }
        }
    }

    /// Nearest-neighbor query returning the closest entry id and distance squared.
    pub fn nearest(&self, point: &[f64]) -> Result<(u64, f64), RTreeError> {
        if self.is_empty() {
            return Err(RTreeError::EmptyTree);
        }
        let mut best_id = 0;
        let mut best_dist = f64::INFINITY;
        self.nearest_node(self.root, point, &mut best_id, &mut best_dist);
        Ok((best_id, best_dist))
    }

    fn nearest_node(&self, node_idx: usize, point: &[f64], best_id: &mut u64, best_dist: &mut f64) {
        let node_dist = self.nodes[node_idx].bbox.distance_sq_to_point(point);
        if node_dist >= *best_dist {
            return;
        }
        match &self.nodes[node_idx].kind {
            NodeKind::Leaf { entries } => {
                for e in entries {
                    let d = e.bbox.distance_sq_to_point(point);
                    if d < *best_dist {
                        *best_dist = d;
                        *best_id = e.id;
                    }
                }
            }
            NodeKind::Internal { children } => {
                let mut ordered: Vec<(usize, f64)> = children.iter()
                    .map(|c| (*c, self.nodes[*c].bbox.distance_sq_to_point(point)))
                    .collect();
                ordered.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                for (c, _) in ordered {
                    self.nearest_node(c, point, best_id, best_dist);
                }
            }
        }
    }

    /// Delete entry by id. Returns true if found and removed.
    pub fn delete(&mut self, id: u64) -> bool {
        if self.delete_from_node(self.root, id) {
            self.entry_count -= 1;
            true
        } else {
            false
        }
    }

    fn delete_from_node(&mut self, node_idx: usize, id: u64) -> bool {
        match &mut self.nodes[node_idx].kind {
            NodeKind::Leaf { entries } => {
                let before = entries.len();
                entries.retain(|e| e.id != id);
                entries.len() < before
            }
            NodeKind::Internal { children } => {
                let children_copy: Vec<usize> = children.clone();
                for &c in &children_copy {
                    if self.delete_from_node(c, id) {
                        return true;
                    }
                }
                false
            }
        }
    }

    /// Bulk load entries using Sort-Tile-Recursive (STR).
    pub fn bulk_load(config: RTreeConfig, mut entries: Vec<Entry>) -> Result<Self, RTreeError> {
        config.validate()?;
        if entries.is_empty() {
            return Self::new(config);
        }
        let dims = config.dimensions;
        let max_c = config.max_children;
        // Sort by center of first dimension, then tile
        entries.sort_by(|a, b| {
            let ca = (a.bbox.min[0] + a.bbox.max[0]) * 0.5;
            let cb = (b.bbox.min[0] + b.bbox.max[0]) * 0.5;
            ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
        });
        // Within slices of sqrt(n/M), sort by second dimension
        let slice_size = ((entries.len() as f64 / max_c as f64).sqrt().ceil() as usize).max(1) * max_c;
        for chunk in entries.chunks_mut(slice_size) {
            let dim = if dims > 1 { 1 } else { 0 };
            chunk.sort_by(|a, b| {
                let ca = (a.bbox.min[dim] + a.bbox.max[dim]) * 0.5;
                let cb = (b.bbox.min[dim] + b.bbox.max[dim]) * 0.5;
                ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        let mut tree = Self::new(config)?;
        for e in entries {
            tree.insert(e)?;
        }
        Ok(tree)
    }
}

impl fmt::Display for RTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RTree(entries={}, nodes={})", self.entry_count, self.nodes.len())
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn pt_entry(x: f64, y: f64, id: u64) -> Entry {
        Entry { bbox: BBox::from_point(&[x, y]), id }
    }

    fn rect_entry(x0: f64, y0: f64, x1: f64, y1: f64, id: u64) -> Entry {
        Entry { bbox: BBox::new(vec![x0, y0], vec![x1, y1]), id }
    }

    #[test]
    fn test_bbox_area() {
        let bb = BBox::new(vec![0.0, 0.0], vec![3.0, 4.0]);
        assert!((bb.area() - 12.0).abs() < 1e-9);
    }

    #[test]
    fn test_bbox_contains_point() {
        let bb = BBox::new(vec![0.0, 0.0], vec![10.0, 10.0]);
        assert!(bb.contains_point(&[5.0, 5.0]));
        assert!(!bb.contains_point(&[11.0, 5.0]));
    }

    #[test]
    fn test_bbox_intersects() {
        let a = BBox::new(vec![0.0, 0.0], vec![5.0, 5.0]);
        let b = BBox::new(vec![3.0, 3.0], vec![8.0, 8.0]);
        let c = BBox::new(vec![6.0, 6.0], vec![9.0, 9.0]);
        assert!(a.intersects(&b));
        assert!(!a.intersects(&c));
    }

    #[test]
    fn test_bbox_merge() {
        let a = BBox::new(vec![0.0, 0.0], vec![5.0, 5.0]);
        let b = BBox::new(vec![3.0, -1.0], vec![8.0, 3.0]);
        let m = a.merge(&b);
        assert_eq!(m.min, vec![0.0, -1.0]);
        assert_eq!(m.max, vec![8.0, 5.0]);
    }

    #[test]
    fn test_bbox_display() {
        let bb = BBox::new(vec![1.0], vec![2.0]);
        let s = format!("{bb}");
        assert!(s.contains("BBox"));
    }

    #[test]
    fn test_config_validation() {
        assert!(RTreeConfig::new(0).validate().is_err());
        assert!(RTreeConfig::new(2).with_min_children(5).with_max_children(4).validate().is_err());
        assert!(RTreeConfig::new(2).validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let c = RTreeConfig::new(2);
        let s = format!("{c}");
        assert!(s.contains("2D"));
    }

    #[test]
    fn test_insert_and_search() {
        let mut tree = RTree::new(RTreeConfig::new(2)).unwrap();
        tree.insert(pt_entry(1.0, 1.0, 1)).unwrap();
        tree.insert(pt_entry(5.0, 5.0, 2)).unwrap();
        tree.insert(pt_entry(9.0, 9.0, 3)).unwrap();
        let results = tree.search(&BBox::new(vec![0.0, 0.0], vec![6.0, 6.0]));
        assert!(results.contains(&1));
        assert!(results.contains(&2));
        assert!(!results.contains(&3));
    }

    #[test]
    fn test_insert_rect_entries() {
        let mut tree = RTree::new(RTreeConfig::new(2)).unwrap();
        tree.insert(rect_entry(0.0, 0.0, 2.0, 2.0, 10)).unwrap();
        tree.insert(rect_entry(5.0, 5.0, 7.0, 7.0, 11)).unwrap();
        let results = tree.search(&BBox::new(vec![1.0, 1.0], vec![3.0, 3.0]));
        assert!(results.contains(&10));
    }

    #[test]
    fn test_nearest_neighbor() {
        let mut tree = RTree::new(RTreeConfig::new(2)).unwrap();
        tree.insert(pt_entry(0.0, 0.0, 1)).unwrap();
        tree.insert(pt_entry(10.0, 10.0, 2)).unwrap();
        let (id, _dist) = tree.nearest(&[1.0, 1.0]).unwrap();
        assert_eq!(id, 1);
    }

    #[test]
    fn test_nearest_empty_tree() {
        let tree = RTree::new(RTreeConfig::new(2)).unwrap();
        assert!(tree.nearest(&[0.0, 0.0]).is_err());
    }

    #[test]
    fn test_delete() {
        let mut tree = RTree::new(RTreeConfig::new(2)).unwrap();
        tree.insert(pt_entry(1.0, 1.0, 1)).unwrap();
        tree.insert(pt_entry(2.0, 2.0, 2)).unwrap();
        assert_eq!(tree.len(), 2);
        assert!(tree.delete(1));
        assert_eq!(tree.len(), 1);
        assert!(!tree.delete(99));
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut tree = RTree::new(RTreeConfig::new(2)).unwrap();
        let entry = Entry { bbox: BBox::from_point(&[1.0, 2.0, 3.0]), id: 1 };
        assert!(tree.insert(entry).is_err());
    }

    #[test]
    fn test_bulk_load() {
        let entries: Vec<Entry> = (0..50).map(|i| pt_entry(i as f64, i as f64 * 0.5, i)).collect();
        let tree = RTree::bulk_load(RTreeConfig::new(2), entries).unwrap();
        assert_eq!(tree.len(), 50);
    }

    #[test]
    fn test_bulk_load_empty() {
        let tree = RTree::bulk_load(RTreeConfig::new(2), vec![]).unwrap();
        assert!(tree.is_empty());
    }

    #[test]
    fn test_linear_split_strategy() {
        let config = RTreeConfig::new(2).with_max_children(4).with_split_strategy(SplitStrategy::Linear);
        let mut tree = RTree::new(config).unwrap();
        for i in 0..20 {
            tree.insert(pt_entry(i as f64, i as f64, i)).unwrap();
        }
        assert_eq!(tree.len(), 20);
    }

    #[test]
    fn test_quadratic_split_strategy() {
        let config = RTreeConfig::new(2).with_max_children(4).with_split_strategy(SplitStrategy::Quadratic);
        let mut tree = RTree::new(config).unwrap();
        for i in 0..20 {
            tree.insert(pt_entry(i as f64, i as f64, i)).unwrap();
        }
        assert_eq!(tree.len(), 20);
    }

    #[test]
    fn test_search_no_results() {
        let mut tree = RTree::new(RTreeConfig::new(2)).unwrap();
        tree.insert(pt_entry(0.0, 0.0, 1)).unwrap();
        let results = tree.search(&BBox::new(vec![100.0, 100.0], vec![200.0, 200.0]));
        assert!(results.is_empty());
    }

    #[test]
    fn test_3d_tree() {
        let mut tree = RTree::new(RTreeConfig::new(3)).unwrap();
        tree.insert(Entry { bbox: BBox::from_point(&[1.0, 2.0, 3.0]), id: 1 }).unwrap();
        tree.insert(Entry { bbox: BBox::from_point(&[4.0, 5.0, 6.0]), id: 2 }).unwrap();
        let results = tree.search(&BBox::new(vec![0.0, 0.0, 0.0], vec![3.0, 3.0, 4.0]));
        assert!(results.contains(&1));
    }

    #[test]
    fn test_display() {
        let tree = RTree::new(RTreeConfig::new(2)).unwrap();
        let s = format!("{tree}");
        assert!(s.contains("RTree"));
        assert!(s.contains("entries=0"));
    }

    #[test]
    fn test_split_strategy_display() {
        assert_eq!(format!("{}", SplitStrategy::Linear), "Linear");
        assert_eq!(format!("{}", SplitStrategy::Quadratic), "Quadratic");
    }
}
