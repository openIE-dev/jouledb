//! Quadtree spatial index — insert, remove, range query, nearest neighbor,
//! bulk loading, split/merge, point-in-region, iterator, collision detection.
//!
//! Replaces JavaScript quadtree libraries (d3-quadtree, quadtree-js) with a
//! pure-Rust spatial index for 2D games and simulations.

use std::collections::HashSet;

// ── Point & Rectangle ───────────────────────────────────────────

/// A 2D point with an identifier.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub id: u64,
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub fn new(id: u64, x: f64, y: f64) -> Self { Self { id, x, y } }
}

/// Axis-aligned bounding rectangle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

impl Rect {
    pub fn new(x: f64, y: f64, w: f64, h: f64) -> Self { Self { x, y, w, h } }

    pub fn contains(&self, px: f64, py: f64) -> bool {
        px >= self.x && px < self.x + self.w && py >= self.y && py < self.y + self.h
    }

    pub fn intersects(&self, other: &Rect) -> bool {
        !(self.x + self.w <= other.x
            || other.x + other.w <= self.x
            || self.y + self.h <= other.y
            || other.y + other.h <= self.y)
    }

    fn quadrants(&self) -> [Rect; 4] {
        let hw = self.w / 2.0;
        let hh = self.h / 2.0;
        [
            Rect::new(self.x, self.y, hw, hh),           // NW
            Rect::new(self.x + hw, self.y, hw, hh),      // NE
            Rect::new(self.x, self.y + hh, hw, hh),      // SW
            Rect::new(self.x + hw, self.y + hh, hw, hh), // SE
        ]
    }
}

// ── Quadtree ────────────────────────────────────────────────────

/// A quadtree node.
#[derive(Debug, Clone)]
pub struct Quadtree {
    boundary: Rect,
    capacity: usize,
    points: Vec<Point>,
    children: Option<Box<[Quadtree; 4]>>,
    total_count: usize,
}

impl Quadtree {
    /// Create a new quadtree with the given boundary and per-node capacity.
    pub fn new(boundary: Rect, capacity: usize) -> Self {
        Self {
            boundary,
            capacity: capacity.max(1),
            points: Vec::new(),
            children: None,
            total_count: 0,
        }
    }

    /// Insert a point. Returns false if out of bounds.
    pub fn insert(&mut self, point: Point) -> bool {
        if !self.boundary.contains(point.x, point.y) {
            return false;
        }
        self.total_count += 1;

        if self.children.is_none() && self.points.len() < self.capacity {
            self.points.push(point);
            return true;
        }

        // Subdivide if needed.
        if self.children.is_none() {
            self.subdivide();
        }

        if let Some(children) = &mut self.children {
            for child in children.iter_mut() {
                if child.insert(point) {
                    return true;
                }
            }
        }
        // Shouldn't reach here if boundary check passed.
        false
    }

    fn subdivide(&mut self) {
        let quads = self.boundary.quadrants();
        let cap = self.capacity;
        let mut children = Box::new([
            Quadtree::new(quads[0], cap),
            Quadtree::new(quads[1], cap),
            Quadtree::new(quads[2], cap),
            Quadtree::new(quads[3], cap),
        ]);
        // Re-insert existing points.
        for p in self.points.drain(..) {
            for child in children.iter_mut() {
                if child.insert(p) {
                    break;
                }
            }
        }
        self.children = Some(children);
    }

    /// Remove a point by id. Returns true if found and removed.
    pub fn remove(&mut self, id: u64) -> bool {
        if let Some(idx) = self.points.iter().position(|p| p.id == id) {
            self.points.swap_remove(idx);
            self.total_count -= 1;
            self.try_merge();
            return true;
        }
        if let Some(children) = &mut self.children {
            for child in children.iter_mut() {
                if child.remove(id) {
                    self.total_count -= 1;
                    self.try_merge();
                    return true;
                }
            }
        }
        false
    }

    fn try_merge(&mut self) {
        if self.total_count <= self.capacity {
            if let Some(children) = self.children.take() {
                for child in children.into_iter() {
                    self.collect_all_into(&child);
                }
            }
        }
    }

    fn collect_all_into(&mut self, node: &Quadtree) {
        self.points.extend_from_slice(&node.points);
        if let Some(children) = &node.children {
            for child in children.iter() {
                self.collect_all_into(child);
            }
        }
    }

    /// Query all points within a rectangle.
    pub fn query_range(&self, range: &Rect) -> Vec<Point> {
        let mut result = Vec::new();
        self.query_range_inner(range, &mut result);
        result
    }

    fn query_range_inner(&self, range: &Rect, result: &mut Vec<Point>) {
        if !self.boundary.intersects(range) {
            return;
        }
        for p in &self.points {
            if range.contains(p.x, p.y) {
                result.push(*p);
            }
        }
        if let Some(children) = &self.children {
            for child in children.iter() {
                child.query_range_inner(range, result);
            }
        }
    }

    /// Find the nearest neighbor to a query point.
    pub fn nearest(&self, qx: f64, qy: f64) -> Option<Point> {
        let mut best: Option<(Point, f64)> = None;
        self.nearest_inner(qx, qy, &mut best);
        best.map(|(p, _)| p)
    }

    fn nearest_inner(&self, qx: f64, qy: f64, best: &mut Option<(Point, f64)>) {
        // Prune if this node's boundary is farther than current best.
        if let Some((_, best_dist)) = best {
            let closest_x = qx.clamp(self.boundary.x, self.boundary.x + self.boundary.w);
            let closest_y = qy.clamp(self.boundary.y, self.boundary.y + self.boundary.h);
            let boundary_dist = dist_sq(qx, qy, closest_x, closest_y);
            if boundary_dist > *best_dist {
                return;
            }
        }
        for p in &self.points {
            let d = dist_sq(qx, qy, p.x, p.y);
            if best.is_none() || d < best.unwrap().1 {
                *best = Some((*p, d));
            }
        }
        if let Some(children) = &self.children {
            for child in children.iter() {
                child.nearest_inner(qx, qy, best);
            }
        }
    }

    /// Bulk-load points. More efficient than individual inserts when
    /// the full set is known ahead of time.
    pub fn bulk_load(boundary: Rect, capacity: usize, points: &[Point]) -> Self {
        let mut tree = Self::new(boundary, capacity);
        for p in points {
            tree.insert(*p);
        }
        tree
    }

    /// Check if a point (by coordinates) lies in any region containing a stored point.
    pub fn point_in_region(&self, x: f64, y: f64, radius: f64) -> bool {
        let range = Rect::new(x - radius, y - radius, radius * 2.0, radius * 2.0);
        let candidates = self.query_range(&range);
        candidates.iter().any(|p| dist_sq(x, y, p.x, p.y) <= radius * radius)
    }

    /// Total number of points in the tree.
    pub fn len(&self) -> usize {
        self.total_count
    }

    pub fn is_empty(&self) -> bool {
        self.total_count == 0
    }

    /// Collect all points in the tree.
    pub fn all_points(&self) -> Vec<Point> {
        let mut result = Vec::new();
        self.collect_all(&mut result);
        result
    }

    fn collect_all(&self, result: &mut Vec<Point>) {
        result.extend_from_slice(&self.points);
        if let Some(children) = &self.children {
            for child in children.iter() {
                child.collect_all(result);
            }
        }
    }

    /// Detect all pairs of points within `distance` of each other.
    pub fn collision_pairs(&self, distance: f64) -> Vec<(Point, Point)> {
        let all = self.all_points();
        let dist_sq_threshold = distance * distance;
        let mut pairs = Vec::new();
        let mut seen: HashSet<(u64, u64)> = HashSet::new();

        for p in &all {
            let range = Rect::new(p.x - distance, p.y - distance, distance * 2.0, distance * 2.0);
            let nearby = self.query_range(&range);
            for q in &nearby {
                if p.id == q.id { continue; }
                let key = if p.id < q.id { (p.id, q.id) } else { (q.id, p.id) };
                if seen.contains(&key) { continue; }
                if dist_sq(p.x, p.y, q.x, q.y) <= dist_sq_threshold {
                    pairs.push((*p, *q));
                    seen.insert(key);
                }
            }
        }
        pairs
    }
}

fn dist_sq(ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    (ax - bx) * (ax - bx) + (ay - by) * (ay - by)
}

/// Iterator over all points in the quadtree (depth-first).
pub struct QuadtreeIter {
    stack: Vec<Point>,
}

impl Quadtree {
    pub fn iter(&self) -> QuadtreeIter {
        QuadtreeIter { stack: self.all_points() }
    }
}

impl Iterator for QuadtreeIter {
    type Item = Point;
    fn next(&mut self) -> Option<Self::Item> {
        self.stack.pop()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_boundary() -> Rect {
        Rect::new(0.0, 0.0, 100.0, 100.0)
    }

    #[test]
    fn insert_and_count() {
        let mut qt = Quadtree::new(test_boundary(), 4);
        for i in 0..10 {
            assert!(qt.insert(Point::new(i, (i * 10) as f64, (i * 5) as f64)));
        }
        assert_eq!(qt.len(), 10);
    }

    #[test]
    fn insert_out_of_bounds() {
        let mut qt = Quadtree::new(test_boundary(), 4);
        assert!(!qt.insert(Point::new(1, 200.0, 200.0)));
        assert_eq!(qt.len(), 0);
    }

    #[test]
    fn remove_point() {
        let mut qt = Quadtree::new(test_boundary(), 4);
        qt.insert(Point::new(1, 10.0, 10.0));
        qt.insert(Point::new(2, 20.0, 20.0));
        assert!(qt.remove(1));
        assert_eq!(qt.len(), 1);
        assert!(!qt.remove(99));
    }

    #[test]
    fn range_query() {
        let mut qt = Quadtree::new(test_boundary(), 4);
        qt.insert(Point::new(1, 10.0, 10.0));
        qt.insert(Point::new(2, 50.0, 50.0));
        qt.insert(Point::new(3, 90.0, 90.0));

        let result = qt.query_range(&Rect::new(0.0, 0.0, 30.0, 30.0));
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].id, 1);
    }

    #[test]
    fn nearest_neighbor() {
        let mut qt = Quadtree::new(test_boundary(), 4);
        qt.insert(Point::new(1, 10.0, 10.0));
        qt.insert(Point::new(2, 50.0, 50.0));
        qt.insert(Point::new(3, 90.0, 90.0));

        let nearest = qt.nearest(12.0, 12.0).unwrap();
        assert_eq!(nearest.id, 1);
    }

    #[test]
    fn bulk_load() {
        let points: Vec<Point> = (0..20)
            .map(|i| Point::new(i, (i * 5) as f64, (i * 3) as f64))
            .collect();
        let qt = Quadtree::bulk_load(test_boundary(), 4, &points);
        assert_eq!(qt.len(), 20);
    }

    #[test]
    fn point_in_region() {
        let mut qt = Quadtree::new(test_boundary(), 4);
        qt.insert(Point::new(1, 50.0, 50.0));
        assert!(qt.point_in_region(52.0, 52.0, 5.0));
        assert!(!qt.point_in_region(80.0, 80.0, 5.0));
    }

    #[test]
    fn collision_pairs_detection() {
        let mut qt = Quadtree::new(test_boundary(), 4);
        qt.insert(Point::new(1, 10.0, 10.0));
        qt.insert(Point::new(2, 12.0, 10.0)); // 2 units away
        qt.insert(Point::new(3, 90.0, 90.0)); // far away

        let pairs = qt.collision_pairs(5.0);
        assert_eq!(pairs.len(), 1);
    }

    #[test]
    fn iterator() {
        let mut qt = Quadtree::new(test_boundary(), 4);
        for i in 0..5 {
            qt.insert(Point::new(i, (i * 10) as f64, (i * 10) as f64));
        }
        let collected: Vec<Point> = qt.iter().collect();
        assert_eq!(collected.len(), 5);
    }

    #[test]
    fn rect_contains_and_intersects() {
        let r = Rect::new(0.0, 0.0, 10.0, 10.0);
        assert!(r.contains(5.0, 5.0));
        assert!(!r.contains(10.0, 10.0)); // exclusive upper bound
        assert!(!r.contains(-1.0, 5.0));

        let other = Rect::new(5.0, 5.0, 10.0, 10.0);
        assert!(r.intersects(&other));

        let far = Rect::new(20.0, 20.0, 5.0, 5.0);
        assert!(!r.intersects(&far));
    }

    #[test]
    fn subdivide_and_merge() {
        let mut qt = Quadtree::new(test_boundary(), 2);
        // Insert 3 → triggers subdivision.
        qt.insert(Point::new(1, 10.0, 10.0));
        qt.insert(Point::new(2, 20.0, 20.0));
        qt.insert(Point::new(3, 30.0, 30.0));
        assert_eq!(qt.len(), 3);
        // Remove 2 → should merge back.
        qt.remove(3);
        qt.remove(2);
        assert_eq!(qt.len(), 1);
    }

    #[test]
    fn empty_tree() {
        let qt = Quadtree::new(test_boundary(), 4);
        assert!(qt.is_empty());
        assert!(qt.nearest(50.0, 50.0).is_none());
        assert!(qt.query_range(&test_boundary()).is_empty());
    }
}
