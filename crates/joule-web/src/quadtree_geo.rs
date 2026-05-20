//! Quadtree — point quadtree, region quadtree, insert/query/delete,
//! bounding box range search, nearest-neighbor, depth-limited subdivision,
//! QuadtreeConfig builder.
//!
//! Pure-Rust spatial index for 2D geospatial data with configurable
//! capacity per node, maximum depth, and both point and region variants.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum QuadtreeError {
    InvalidConfig(String),
    OutOfBounds { x: f64, y: f64 },
    EmptyTree,
    NotFound(u64),
}

impl fmt::Display for QuadtreeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(s) => write!(f, "invalid config: {s}"),
            Self::OutOfBounds { x, y } => write!(f, "point ({x}, {y}) out of bounds"),
            Self::EmptyTree => write!(f, "tree is empty"),
            Self::NotFound(id) => write!(f, "entry {id} not found"),
        }
    }
}

impl std::error::Error for QuadtreeError {}

// ── Rect ────────────────────────────────────────────────────────

/// Axis-aligned rectangle in 2D.
#[derive(Debug, Clone, PartialEq)]
pub struct Rect {
    pub x_min: f64,
    pub y_min: f64,
    pub x_max: f64,
    pub y_max: f64,
}

impl Rect {
    pub fn new(x_min: f64, y_min: f64, x_max: f64, y_max: f64) -> Self {
        Self { x_min, y_min, x_max, y_max }
    }

    pub fn width(&self) -> f64 { self.x_max - self.x_min }
    pub fn height(&self) -> f64 { self.y_max - self.y_min }
    pub fn area(&self) -> f64 { self.width() * self.height() }
    pub fn center_x(&self) -> f64 { (self.x_min + self.x_max) * 0.5 }
    pub fn center_y(&self) -> f64 { (self.y_min + self.y_max) * 0.5 }

    pub fn contains_point(&self, x: f64, y: f64) -> bool {
        x >= self.x_min && x <= self.x_max && y >= self.y_min && y <= self.y_max
    }

    pub fn intersects(&self, other: &Rect) -> bool {
        self.x_min <= other.x_max && self.x_max >= other.x_min
            && self.y_min <= other.y_max && self.y_max >= other.y_min
    }

    pub fn contains_rect(&self, other: &Rect) -> bool {
        self.x_min <= other.x_min && self.x_max >= other.x_max
            && self.y_min <= other.y_min && self.y_max >= other.y_max
    }

    /// Distance squared from a point to the nearest edge of this rect.
    pub fn distance_sq_to_point(&self, x: f64, y: f64) -> f64 {
        let dx = if x < self.x_min { self.x_min - x }
                 else if x > self.x_max { x - self.x_max }
                 else { 0.0 };
        let dy = if y < self.y_min { self.y_min - y }
                 else if y > self.y_max { y - self.y_max }
                 else { 0.0 };
        dx * dx + dy * dy
    }

    fn nw(&self) -> Rect {
        Rect::new(self.x_min, self.center_y(), self.center_x(), self.y_max)
    }
    fn ne(&self) -> Rect {
        Rect::new(self.center_x(), self.center_y(), self.x_max, self.y_max)
    }
    fn sw(&self) -> Rect {
        Rect::new(self.x_min, self.y_min, self.center_x(), self.center_y())
    }
    fn se(&self) -> Rect {
        Rect::new(self.center_x(), self.y_min, self.x_max, self.center_y())
    }
}

impl fmt::Display for Rect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Rect({:.4}, {:.4} .. {:.4}, {:.4})", self.x_min, self.y_min, self.x_max, self.y_max)
    }
}

// ── Point entry ─────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct PointEntry {
    pub x: f64,
    pub y: f64,
    pub id: u64,
}

impl fmt::Display for PointEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Point({}, {} id={})", self.x, self.y, self.id)
    }
}

// ── Config ──────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct QuadtreeConfig {
    pub capacity: usize,
    pub max_depth: usize,
    pub bounds: Rect,
}

impl QuadtreeConfig {
    pub fn new(bounds: Rect) -> Self {
        Self { capacity: 8, max_depth: 20, bounds }
    }

    pub fn with_capacity(mut self, c: usize) -> Self { self.capacity = c; self }
    pub fn with_max_depth(mut self, d: usize) -> Self { self.max_depth = d; self }
    pub fn with_bounds(mut self, b: Rect) -> Self { self.bounds = b; self }

    pub fn validate(&self) -> Result<(), QuadtreeError> {
        if self.capacity == 0 {
            return Err(QuadtreeError::InvalidConfig("capacity must be > 0".into()));
        }
        if self.max_depth == 0 {
            return Err(QuadtreeError::InvalidConfig("max_depth must be > 0".into()));
        }
        if self.bounds.width() <= 0.0 || self.bounds.height() <= 0.0 {
            return Err(QuadtreeError::InvalidConfig("bounds must have positive area".into()));
        }
        Ok(())
    }
}

impl fmt::Display for QuadtreeConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "QuadtreeConfig(cap={}, depth={}, {})", self.capacity, self.max_depth, self.bounds)
    }
}

// ── Quadtree node ───────────────────────────────────────────────

struct QtNode {
    bounds: Rect,
    points: Vec<PointEntry>,
    children: Option<[Box<QtNode>; 4]>,  // NW, NE, SW, SE
    depth: usize,
}

impl QtNode {
    fn new(bounds: Rect, depth: usize) -> Self {
        Self { bounds, points: Vec::new(), children: None, depth }
    }

    fn is_leaf(&self) -> bool { self.children.is_none() }

    fn subdivide(&mut self) {
        let nw = Box::new(QtNode::new(self.bounds.nw(), self.depth + 1));
        let ne = Box::new(QtNode::new(self.bounds.ne(), self.depth + 1));
        let sw = Box::new(QtNode::new(self.bounds.sw(), self.depth + 1));
        let se = Box::new(QtNode::new(self.bounds.se(), self.depth + 1));
        self.children = Some([nw, ne, sw, se]);
    }

    fn quadrant_for(&self, x: f64, y: f64) -> usize {
        let cx = self.bounds.center_x();
        let cy = self.bounds.center_y();
        if x < cx {
            if y >= cy { 0 } else { 2 }  // NW or SW
        } else {
            if y >= cy { 1 } else { 3 }  // NE or SE
        }
    }
}

// ── Quadtree ────────────────────────────────────────────────────

pub struct Quadtree {
    root: QtNode,
    config: QuadtreeConfig,
    count: usize,
}

impl Quadtree {
    pub fn new(config: QuadtreeConfig) -> Result<Self, QuadtreeError> {
        config.validate()?;
        let root = QtNode::new(config.bounds.clone(), 0);
        Ok(Self { root, config, count: 0 })
    }

    pub fn len(&self) -> usize { self.count }
    pub fn is_empty(&self) -> bool { self.count == 0 }
    pub fn bounds(&self) -> &Rect { &self.config.bounds }

    /// Insert a point.
    pub fn insert(&mut self, x: f64, y: f64, id: u64) -> Result<(), QuadtreeError> {
        if !self.config.bounds.contains_point(x, y) {
            return Err(QuadtreeError::OutOfBounds { x, y });
        }
        let cap = self.config.capacity;
        let max_d = self.config.max_depth;
        Self::insert_into(&mut self.root, PointEntry { x, y, id }, cap, max_d);
        self.count += 1;
        Ok(())
    }

    fn insert_into(node: &mut QtNode, entry: PointEntry, capacity: usize, max_depth: usize) {
        if node.is_leaf() {
            if node.points.len() < capacity || node.depth >= max_depth {
                node.points.push(entry);
                return;
            }
            node.subdivide();
            let drained: Vec<PointEntry> = node.points.drain(..).collect();
            for p in drained {
                let q = node.quadrant_for(p.x, p.y);
                Self::insert_into(&mut node.children.as_mut().unwrap()[q], p, capacity, max_depth);
            }
        }
        let q = node.quadrant_for(entry.x, entry.y);
        Self::insert_into(&mut node.children.as_mut().unwrap()[q], entry, capacity, max_depth);
    }

    /// Range search: return all entries within `query` rect.
    pub fn query_range(&self, query: &Rect) -> Vec<PointEntry> {
        let mut results = Vec::new();
        Self::query_range_node(&self.root, query, &mut results);
        results
    }

    fn query_range_node(node: &QtNode, query: &Rect, results: &mut Vec<PointEntry>) {
        if !node.bounds.intersects(query) {
            return;
        }
        for p in &node.points {
            if query.contains_point(p.x, p.y) {
                results.push(p.clone());
            }
        }
        if let Some(children) = &node.children {
            for child in children.iter() {
                Self::query_range_node(child, query, results);
            }
        }
    }

    /// Nearest-neighbor search.
    pub fn nearest(&self, x: f64, y: f64) -> Result<(PointEntry, f64), QuadtreeError> {
        if self.is_empty() {
            return Err(QuadtreeError::EmptyTree);
        }
        let mut best: Option<PointEntry> = None;
        let mut best_dist = f64::INFINITY;
        Self::nearest_node(&self.root, x, y, &mut best, &mut best_dist);
        match best {
            Some(p) => Ok((p, best_dist)),
            None => Err(QuadtreeError::EmptyTree),
        }
    }

    fn nearest_node(node: &QtNode, x: f64, y: f64, best: &mut Option<PointEntry>, best_dist: &mut f64) {
        if node.bounds.distance_sq_to_point(x, y) >= *best_dist {
            return;
        }
        for p in &node.points {
            let d = (p.x - x) * (p.x - x) + (p.y - y) * (p.y - y);
            if d < *best_dist {
                *best_dist = d;
                *best = Some(p.clone());
            }
        }
        if let Some(children) = &node.children {
            let mut ordered: Vec<(usize, f64)> = children.iter().enumerate()
                .map(|(i, c)| (i, c.bounds.distance_sq_to_point(x, y)))
                .collect();
            ordered.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            for (i, _) in ordered {
                Self::nearest_node(&children[i], x, y, best, best_dist);
            }
        }
    }

    /// Delete entry by id. Returns true if found and removed.
    pub fn delete(&mut self, id: u64) -> bool {
        if Self::delete_from_node(&mut self.root, id) {
            self.count -= 1;
            true
        } else {
            false
        }
    }

    fn delete_from_node(node: &mut QtNode, id: u64) -> bool {
        let before = node.points.len();
        node.points.retain(|p| p.id != id);
        if node.points.len() < before {
            return true;
        }
        if let Some(children) = &mut node.children {
            for child in children.iter_mut() {
                if Self::delete_from_node(child, id) {
                    return true;
                }
            }
        }
        false
    }

    /// Count all entries (recursive verification).
    pub fn count_all(&self) -> usize {
        Self::count_node(&self.root)
    }

    fn count_node(node: &QtNode) -> usize {
        let mut total = node.points.len();
        if let Some(children) = &node.children {
            for child in children.iter() {
                total += Self::count_node(child);
            }
        }
        total
    }

    /// Get the maximum depth used in the tree.
    pub fn max_depth_used(&self) -> usize {
        Self::depth_node(&self.root)
    }

    fn depth_node(node: &QtNode) -> usize {
        if let Some(children) = &node.children {
            children.iter().map(|c| Self::depth_node(c)).max().unwrap_or(node.depth)
        } else {
            node.depth
        }
    }
}

impl fmt::Display for Quadtree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Quadtree(entries={}, bounds={})", self.count, self.config.bounds)
    }
}

// ── Region Quadtree ─────────────────────────────────────────────

/// A region quadtree that subdivides space into uniform cells, each
/// storing an optional value.
pub struct RegionQuadtree {
    root: RegionNode,
    bounds: Rect,
    max_depth: usize,
}

struct RegionNode {
    value: Option<u64>,
    children: Option<[Box<RegionNode>; 4]>,
}

impl RegionNode {
    fn new() -> Self { Self { value: None, children: None } }

    fn is_leaf(&self) -> bool { self.children.is_none() }
}

impl RegionQuadtree {
    pub fn new(bounds: Rect, max_depth: usize) -> Self {
        Self { root: RegionNode::new(), bounds, max_depth }
    }

    /// Set the value at position (x, y).
    pub fn set(&mut self, x: f64, y: f64, value: u64) -> bool {
        if !self.bounds.contains_point(x, y) {
            return false;
        }
        Self::set_node(&mut self.root, &self.bounds, x, y, value, 0, self.max_depth);
        true
    }

    fn set_node(node: &mut RegionNode, bounds: &Rect, x: f64, y: f64, value: u64, depth: usize, max_depth: usize) {
        if depth >= max_depth {
            node.value = Some(value);
            return;
        }
        if node.children.is_none() {
            node.children = Some([
                Box::new(RegionNode::new()),
                Box::new(RegionNode::new()),
                Box::new(RegionNode::new()),
                Box::new(RegionNode::new()),
            ]);
        }
        let cx = bounds.center_x();
        let cy = bounds.center_y();
        let (idx, child_bounds) = if x < cx {
            if y >= cy { (0, bounds.nw()) } else { (2, bounds.sw()) }
        } else {
            if y >= cy { (1, bounds.ne()) } else { (3, bounds.se()) }
        };
        Self::set_node(&mut node.children.as_mut().unwrap()[idx], &child_bounds, x, y, value, depth + 1, max_depth);
    }

    /// Get the value at position (x, y).
    pub fn get(&self, x: f64, y: f64) -> Option<u64> {
        if !self.bounds.contains_point(x, y) {
            return None;
        }
        Self::get_node(&self.root, &self.bounds, x, y, 0, self.max_depth)
    }

    fn get_node(node: &RegionNode, bounds: &Rect, x: f64, y: f64, depth: usize, max_depth: usize) -> Option<u64> {
        if depth >= max_depth || node.is_leaf() {
            return node.value;
        }
        let cx = bounds.center_x();
        let cy = bounds.center_y();
        let (idx, child_bounds) = if x < cx {
            if y >= cy { (0, bounds.nw()) } else { (2, bounds.sw()) }
        } else {
            if y >= cy { (1, bounds.ne()) } else { (3, bounds.se()) }
        };
        Self::get_node(&node.children.as_ref().unwrap()[idx], &child_bounds, x, y, depth + 1, max_depth)
    }
}

impl fmt::Display for RegionQuadtree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RegionQuadtree(bounds={}, max_depth={})", self.bounds, self.max_depth)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn world_bounds() -> Rect {
        Rect::new(-180.0, -90.0, 180.0, 90.0)
    }

    #[test]
    fn test_rect_basics() {
        let r = Rect::new(0.0, 0.0, 10.0, 5.0);
        assert!((r.area() - 50.0).abs() < 1e-9);
        assert!(r.contains_point(5.0, 2.5));
        assert!(!r.contains_point(11.0, 0.0));
    }

    #[test]
    fn test_rect_intersects() {
        let a = Rect::new(0.0, 0.0, 5.0, 5.0);
        let b = Rect::new(3.0, 3.0, 8.0, 8.0);
        let c = Rect::new(6.0, 6.0, 9.0, 9.0);
        assert!(a.intersects(&b));
        assert!(!a.intersects(&c));
    }

    #[test]
    fn test_rect_display() {
        let r = Rect::new(1.0, 2.0, 3.0, 4.0);
        let s = format!("{r}");
        assert!(s.contains("Rect"));
    }

    #[test]
    fn test_config_validation() {
        let r = world_bounds();
        assert!(QuadtreeConfig::new(r.clone()).with_capacity(0).validate().is_err());
        assert!(QuadtreeConfig::new(r.clone()).with_max_depth(0).validate().is_err());
        assert!(QuadtreeConfig::new(r).validate().is_ok());
    }

    #[test]
    fn test_config_display() {
        let c = QuadtreeConfig::new(Rect::new(0.0, 0.0, 10.0, 10.0));
        let s = format!("{c}");
        assert!(s.contains("QuadtreeConfig"));
    }

    #[test]
    fn test_insert_and_count() {
        let config = QuadtreeConfig::new(world_bounds());
        let mut qt = Quadtree::new(config).unwrap();
        for i in 0..100 {
            qt.insert(i as f64 - 50.0, (i as f64) * 0.5 - 25.0, i).unwrap();
        }
        assert_eq!(qt.len(), 100);
        assert_eq!(qt.count_all(), 100);
    }

    #[test]
    fn test_insert_out_of_bounds() {
        let config = QuadtreeConfig::new(Rect::new(0.0, 0.0, 10.0, 10.0));
        let mut qt = Quadtree::new(config).unwrap();
        assert!(qt.insert(100.0, 100.0, 1).is_err());
    }

    #[test]
    fn test_range_query() {
        let config = QuadtreeConfig::new(Rect::new(0.0, 0.0, 100.0, 100.0));
        let mut qt = Quadtree::new(config).unwrap();
        qt.insert(10.0, 10.0, 1).unwrap();
        qt.insert(50.0, 50.0, 2).unwrap();
        qt.insert(90.0, 90.0, 3).unwrap();
        let results = qt.query_range(&Rect::new(0.0, 0.0, 30.0, 30.0));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 1);
    }

    #[test]
    fn test_nearest() {
        let config = QuadtreeConfig::new(Rect::new(0.0, 0.0, 100.0, 100.0));
        let mut qt = Quadtree::new(config).unwrap();
        qt.insert(10.0, 10.0, 1).unwrap();
        qt.insert(50.0, 50.0, 2).unwrap();
        let (entry, _dist) = qt.nearest(12.0, 11.0).unwrap();
        assert_eq!(entry.id, 1);
    }

    #[test]
    fn test_nearest_empty() {
        let config = QuadtreeConfig::new(world_bounds());
        let qt = Quadtree::new(config).unwrap();
        assert!(qt.nearest(0.0, 0.0).is_err());
    }

    #[test]
    fn test_delete() {
        let config = QuadtreeConfig::new(Rect::new(0.0, 0.0, 100.0, 100.0));
        let mut qt = Quadtree::new(config).unwrap();
        qt.insert(10.0, 10.0, 1).unwrap();
        qt.insert(50.0, 50.0, 2).unwrap();
        assert!(qt.delete(1));
        assert_eq!(qt.len(), 1);
        assert!(!qt.delete(99));
    }

    #[test]
    fn test_depth_limited_subdivision() {
        let config = QuadtreeConfig::new(Rect::new(0.0, 0.0, 100.0, 100.0))
            .with_capacity(1)
            .with_max_depth(3);
        let mut qt = Quadtree::new(config).unwrap();
        for i in 0..50 {
            qt.insert(i as f64 * 2.0, i as f64 * 2.0, i).unwrap();
        }
        assert!(qt.max_depth_used() <= 3);
    }

    #[test]
    fn test_display() {
        let config = QuadtreeConfig::new(world_bounds());
        let qt = Quadtree::new(config).unwrap();
        let s = format!("{qt}");
        assert!(s.contains("Quadtree"));
    }

    #[test]
    fn test_point_entry_display() {
        let e = PointEntry { x: 1.0, y: 2.0, id: 3 };
        let s = format!("{e}");
        assert!(s.contains("Point"));
    }

    #[test]
    fn test_region_quadtree_set_get() {
        let mut rq = RegionQuadtree::new(Rect::new(0.0, 0.0, 100.0, 100.0), 4);
        rq.set(25.0, 25.0, 42);
        assert_eq!(rq.get(25.0, 25.0), Some(42));
    }

    #[test]
    fn test_region_quadtree_out_of_bounds() {
        let mut rq = RegionQuadtree::new(Rect::new(0.0, 0.0, 10.0, 10.0), 4);
        assert!(!rq.set(100.0, 100.0, 1));
        assert_eq!(rq.get(100.0, 100.0), None);
    }

    #[test]
    fn test_region_quadtree_display() {
        let rq = RegionQuadtree::new(world_bounds(), 8);
        let s = format!("{rq}");
        assert!(s.contains("RegionQuadtree"));
    }

    #[test]
    fn test_rect_distance_sq() {
        let r = Rect::new(0.0, 0.0, 10.0, 10.0);
        assert!((r.distance_sq_to_point(5.0, 5.0)).abs() < 1e-9);
        assert!((r.distance_sq_to_point(13.0, 0.0) - 9.0).abs() < 1e-9);
    }

    #[test]
    fn test_rect_contains_rect() {
        let outer = Rect::new(0.0, 0.0, 100.0, 100.0);
        let inner = Rect::new(10.0, 10.0, 50.0, 50.0);
        assert!(outer.contains_rect(&inner));
        assert!(!inner.contains_rect(&outer));
    }
}
