//! Interval tree — efficient interval storage and query.
//!
//! Supports insert, point query, overlap query, deletion, min/max endpoints,
//! merging overlapping intervals, and sweep line operations.

use std::fmt;

// ── Interval ────────────────────────────────────────────────────────────────

/// A closed interval [low, high].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Interval {
    pub low: i64,
    pub high: i64,
}

impl Interval {
    pub fn new(low: i64, high: i64) -> Self {
        assert!(low <= high, "Interval low must be <= high");
        Self { low, high }
    }

    /// Check if this interval contains a point.
    pub fn contains_point(&self, point: i64) -> bool {
        self.low <= point && point <= self.high
    }

    /// Check if this interval overlaps with another.
    pub fn overlaps(&self, other: &Self) -> bool {
        self.low <= other.high && other.low <= self.high
    }

    /// Length of the interval.
    pub fn length(&self) -> i64 {
        self.high - self.low
    }
}

impl fmt::Display for Interval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {}]", self.low, self.high)
    }
}

// ── IntervalTree (augmented BST) ────────────────────────────────────────────

#[derive(Debug)]
struct Node {
    interval: Interval,
    max_high: i64,
    left: Option<Box<Node>>,
    right: Option<Box<Node>>,
}

impl Node {
    fn new(interval: Interval) -> Self {
        Self {
            max_high: interval.high,
            interval,
            left: None,
            right: None,
        }
    }

    fn update_max(&mut self) {
        self.max_high = self.interval.high;
        if let Some(l) = &self.left {
            self.max_high = self.max_high.max(l.max_high);
        }
        if let Some(r) = &self.right {
            self.max_high = self.max_high.max(r.max_high);
        }
    }
}

/// Augmented BST-based interval tree.
#[derive(Debug)]
pub struct IntervalTree {
    root: Option<Box<Node>>,
    size: usize,
}

impl IntervalTree {
    pub fn new() -> Self {
        Self {
            root: None,
            size: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Insert an interval.
    pub fn insert(&mut self, interval: Interval) {
        self.root = Some(Self::insert_node(self.root.take(), interval));
        self.size += 1;
    }

    fn insert_node(node: Option<Box<Node>>, interval: Interval) -> Box<Node> {
        match node {
            None => Box::new(Node::new(interval)),
            Some(mut n) => {
                if interval.low <= n.interval.low {
                    n.left = Some(Self::insert_node(n.left.take(), interval));
                } else {
                    n.right = Some(Self::insert_node(n.right.take(), interval));
                }
                n.update_max();
                n
            }
        }
    }

    /// Find all intervals containing the given point.
    pub fn query_point(&self, point: i64) -> Vec<Interval> {
        let mut result = Vec::new();
        Self::query_point_node(&self.root, point, &mut result);
        result
    }

    fn query_point_node(node: &Option<Box<Node>>, point: i64, result: &mut Vec<Interval>) {
        if let Some(n) = node {
            if n.max_high < point {
                return; // No interval in this subtree can contain the point
            }

            // Check left subtree
            Self::query_point_node(&n.left, point, result);

            // Check current interval
            if n.interval.contains_point(point) {
                result.push(n.interval);
            }

            // Check right subtree only if point could be >= low of some right node
            if point >= n.interval.low {
                Self::query_point_node(&n.right, point, result);
            }
        }
    }

    /// Find all intervals overlapping with the given query interval.
    pub fn query_overlap(&self, query: &Interval) -> Vec<Interval> {
        let mut result = Vec::new();
        Self::query_overlap_node(&self.root, query, &mut result);
        result
    }

    fn query_overlap_node(
        node: &Option<Box<Node>>,
        query: &Interval,
        result: &mut Vec<Interval>,
    ) {
        if let Some(n) = node {
            if n.max_high < query.low {
                return;
            }

            Self::query_overlap_node(&n.left, query, result);

            if n.interval.overlaps(query) {
                result.push(n.interval);
            }

            if query.high >= n.interval.low {
                Self::query_overlap_node(&n.right, query, result);
            }
        }
    }

    /// Delete an interval. Returns true if found and removed.
    pub fn delete(&mut self, interval: &Interval) -> bool {
        let (new_root, found) = Self::delete_node(self.root.take(), interval);
        self.root = new_root;
        if found {
            self.size -= 1;
        }
        found
    }

    fn delete_node(
        node: Option<Box<Node>>,
        target: &Interval,
    ) -> (Option<Box<Node>>, bool) {
        match node {
            None => (None, false),
            Some(mut n) => {
                if n.interval == *target {
                    // Found — replace with in-order successor or predecessor
                    match (n.left.take(), n.right.take()) {
                        (None, None) => (None, true),
                        (Some(left), None) => (Some(left), true),
                        (None, Some(right)) => (Some(right), true),
                        (Some(left), Some(right)) => {
                            // Find min of right subtree
                            let (min_interval, new_right) = Self::remove_min(right);
                            let mut replacement = Box::new(Node::new(min_interval));
                            replacement.left = Some(left);
                            replacement.right = new_right;
                            replacement.update_max();
                            (Some(replacement), true)
                        }
                    }
                } else if target.low <= n.interval.low {
                    let (new_left, found) = Self::delete_node(n.left.take(), target);
                    n.left = new_left;
                    n.update_max();
                    (Some(n), found)
                } else {
                    let (new_right, found) = Self::delete_node(n.right.take(), target);
                    n.right = new_right;
                    n.update_max();
                    (Some(n), found)
                }
            }
        }
    }

    fn remove_min(mut node: Box<Node>) -> (Interval, Option<Box<Node>>) {
        if node.left.is_none() {
            let interval = node.interval;
            (interval, node.right.take())
        } else {
            let left = node.left.take().unwrap();
            let (min, new_left) = Self::remove_min(left);
            node.left = new_left;
            node.update_max();
            (min, Some(node))
        }
    }

    /// Minimum low endpoint across all intervals.
    pub fn min_low(&self) -> Option<i64> {
        Self::leftmost(&self.root).map(|n| n.interval.low)
    }

    fn leftmost(node: &Option<Box<Node>>) -> Option<&Node> {
        node.as_ref().map(|n| {
            if n.left.is_some() {
                Self::leftmost(&n.left).unwrap()
            } else {
                n.as_ref()
            }
        })
    }

    /// Maximum high endpoint across all intervals.
    pub fn max_high(&self) -> Option<i64> {
        self.root.as_ref().map(|n| n.max_high)
    }

    /// Collect all intervals in sorted order (by low endpoint).
    pub fn all_intervals(&self) -> Vec<Interval> {
        let mut result = Vec::new();
        Self::inorder(&self.root, &mut result);
        result
    }

    fn inorder(node: &Option<Box<Node>>, result: &mut Vec<Interval>) {
        if let Some(n) = node {
            Self::inorder(&n.left, result);
            result.push(n.interval);
            Self::inorder(&n.right, result);
        }
    }
}

impl Default for IntervalTree {
    fn default() -> Self {
        Self::new()
    }
}

// ── Merge overlapping intervals (sweep-line) ────────────────────────────────

/// Merge a list of intervals, combining all overlapping or adjacent ones.
pub fn merge_intervals(intervals: &[Interval]) -> Vec<Interval> {
    if intervals.is_empty() {
        return Vec::new();
    }
    let mut sorted: Vec<Interval> = intervals.to_vec();
    sorted.sort_by_key(|iv| (iv.low, iv.high));

    let mut merged = vec![sorted[0]];
    for iv in &sorted[1..] {
        let last = merged.last_mut().unwrap();
        if iv.low <= last.high + 1 {
            last.high = last.high.max(iv.high);
        } else {
            merged.push(*iv);
        }
    }
    merged
}

/// Sweep-line: compute the total covered length of a set of intervals.
pub fn sweep_line_coverage(intervals: &[Interval]) -> i64 {
    let merged = merge_intervals(intervals);
    merged.iter().map(|iv| iv.length()).sum()
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interval_basics() {
        let iv = Interval::new(1, 10);
        assert!(iv.contains_point(5));
        assert!(iv.contains_point(1));
        assert!(iv.contains_point(10));
        assert!(!iv.contains_point(0));
        assert!(!iv.contains_point(11));
        assert_eq!(iv.length(), 9);
    }

    #[test]
    fn test_interval_overlap() {
        let a = Interval::new(1, 5);
        let b = Interval::new(3, 8);
        let c = Interval::new(6, 10);
        assert!(a.overlaps(&b));
        assert!(!a.overlaps(&c));
        assert!(b.overlaps(&c));
    }

    #[test]
    fn test_insert_and_point_query() {
        let mut tree = IntervalTree::new();
        tree.insert(Interval::new(1, 10));
        tree.insert(Interval::new(5, 15));
        tree.insert(Interval::new(20, 30));

        let hits = tree.query_point(7);
        assert_eq!(hits.len(), 2);
        assert!(hits.contains(&Interval::new(1, 10)));
        assert!(hits.contains(&Interval::new(5, 15)));
    }

    #[test]
    fn test_point_query_no_match() {
        let mut tree = IntervalTree::new();
        tree.insert(Interval::new(1, 5));
        tree.insert(Interval::new(10, 15));
        let hits = tree.query_point(7);
        assert!(hits.is_empty());
    }

    #[test]
    fn test_overlap_query() {
        let mut tree = IntervalTree::new();
        tree.insert(Interval::new(1, 5));
        tree.insert(Interval::new(10, 20));
        tree.insert(Interval::new(15, 25));
        tree.insert(Interval::new(30, 40));

        let query = Interval::new(12, 18);
        let hits = tree.query_overlap(&query);
        assert_eq!(hits.len(), 2);
        assert!(hits.contains(&Interval::new(10, 20)));
        assert!(hits.contains(&Interval::new(15, 25)));
    }

    #[test]
    fn test_delete() {
        let mut tree = IntervalTree::new();
        tree.insert(Interval::new(1, 10));
        tree.insert(Interval::new(5, 15));
        assert_eq!(tree.len(), 2);

        assert!(tree.delete(&Interval::new(1, 10)));
        assert_eq!(tree.len(), 1);
        assert!(tree.query_point(3).is_empty());
        assert!(!tree.query_point(7).is_empty());
    }

    #[test]
    fn test_delete_nonexistent() {
        let mut tree = IntervalTree::new();
        tree.insert(Interval::new(1, 10));
        assert!(!tree.delete(&Interval::new(99, 100)));
        assert_eq!(tree.len(), 1);
    }

    #[test]
    fn test_min_max() {
        let mut tree = IntervalTree::new();
        tree.insert(Interval::new(5, 10));
        tree.insert(Interval::new(1, 3));
        tree.insert(Interval::new(8, 20));
        assert_eq!(tree.min_low(), Some(1));
        assert_eq!(tree.max_high(), Some(20));
    }

    #[test]
    fn test_empty_tree() {
        let tree = IntervalTree::new();
        assert!(tree.is_empty());
        assert_eq!(tree.min_low(), None);
        assert_eq!(tree.max_high(), None);
        assert!(tree.query_point(0).is_empty());
    }

    #[test]
    fn test_merge_intervals() {
        let intervals = vec![
            Interval::new(1, 3),
            Interval::new(2, 6),
            Interval::new(8, 10),
            Interval::new(15, 18),
        ];
        let merged = merge_intervals(&intervals);
        assert_eq!(merged, vec![Interval::new(1, 6), Interval::new(8, 10), Interval::new(15, 18)]);
    }

    #[test]
    fn test_merge_adjacent() {
        let intervals = vec![
            Interval::new(1, 5),
            Interval::new(6, 10),
        ];
        let merged = merge_intervals(&intervals);
        assert_eq!(merged, vec![Interval::new(1, 10)]);
    }

    #[test]
    fn test_sweep_line_coverage() {
        let intervals = vec![
            Interval::new(1, 5),
            Interval::new(3, 8),
            Interval::new(10, 15),
        ];
        assert_eq!(sweep_line_coverage(&intervals), 12); // [1,8] = 7, [10,15] = 5
    }

    #[test]
    fn test_all_intervals_sorted() {
        let mut tree = IntervalTree::new();
        tree.insert(Interval::new(10, 20));
        tree.insert(Interval::new(1, 5));
        tree.insert(Interval::new(5, 15));
        let all = tree.all_intervals();
        assert_eq!(all.len(), 3);
        assert!(all[0].low <= all[1].low);
        assert!(all[1].low <= all[2].low);
    }

    #[test]
    fn test_interval_display() {
        let iv = Interval::new(3, 7);
        assert_eq!(format!("{}", iv), "[3, 7]");
    }
}
