//! Spatial hashing for broad-phase collision detection — grid-based spatial hash,
//! insert/remove/update, query region, query neighbors, cell size optimization,
//! object tracking across cells, collision pair generation.

use std::collections::{HashMap, HashSet};

// ── Vec2 ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    pub fn new(x: f64, y: f64) -> Self { Self { x, y } }
    pub fn zero() -> Self { Self { x: 0.0, y: 0.0 } }
    pub fn add(self, o: Self) -> Self { Self { x: self.x + o.x, y: self.y + o.y } }
    pub fn sub(self, o: Self) -> Self { Self { x: self.x - o.x, y: self.y - o.y } }
    pub fn length(self) -> f64 { (self.x * self.x + self.y * self.y).sqrt() }
}

impl Default for Vec2 {
    fn default() -> Self { Self::zero() }
}

// ── AABB ─────────────────────────────────────────────────────

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AABB {
    pub min: Vec2,
    pub max: Vec2,
}

impl AABB {
    pub fn new(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Self {
        Self {
            min: Vec2::new(min_x, min_y),
            max: Vec2::new(max_x, max_y),
        }
    }

    pub fn from_center_size(cx: f64, cy: f64, w: f64, h: f64) -> Self {
        let hw = w * 0.5;
        let hh = h * 0.5;
        Self::new(cx - hw, cy - hh, cx + hw, cy + hh)
    }

    pub fn from_circle(cx: f64, cy: f64, r: f64) -> Self {
        Self::new(cx - r, cy - r, cx + r, cy + r)
    }

    pub fn width(&self) -> f64 { self.max.x - self.min.x }
    pub fn height(&self) -> f64 { self.max.y - self.min.y }

    pub fn center(&self) -> Vec2 {
        Vec2::new(
            (self.min.x + self.max.x) * 0.5,
            (self.min.y + self.max.y) * 0.5,
        )
    }

    pub fn intersects(&self, other: &AABB) -> bool {
        self.min.x <= other.max.x && self.max.x >= other.min.x &&
        self.min.y <= other.max.y && self.max.y >= other.min.y
    }

    pub fn contains_point(&self, p: Vec2) -> bool {
        p.x >= self.min.x && p.x <= self.max.x &&
        p.y >= self.min.y && p.y <= self.max.y
    }

    pub fn merged(&self, other: &AABB) -> AABB {
        AABB::new(
            self.min.x.min(other.min.x),
            self.min.y.min(other.min.y),
            self.max.x.max(other.max.x),
            self.max.y.max(other.max.y),
        )
    }
}

// ── Cell Key ─────────────────────────────────────────────────

type CellKey = (i32, i32);

fn cell_key(x: f64, y: f64, cell_size: f64) -> CellKey {
    ((x / cell_size).floor() as i32, (y / cell_size).floor() as i32)
}

fn aabb_cells(aabb: &AABB, cell_size: f64) -> Vec<CellKey> {
    let min_c = cell_key(aabb.min.x, aabb.min.y, cell_size);
    let max_c = cell_key(aabb.max.x, aabb.max.y, cell_size);
    let mut keys = Vec::new();
    for cx in min_c.0..=max_c.0 {
        for cy in min_c.1..=max_c.1 {
            keys.push((cx, cy));
        }
    }
    keys
}

// ── SpatialHash ──────────────────────────────────────────────

/// A tracked object in the spatial hash.
#[derive(Debug, Clone)]
struct TrackedObject {
    aabb: AABB,
    /// Cells this object currently occupies.
    cells: Vec<CellKey>,
}

/// Grid-based spatial hash for efficient broad-phase collision detection.
#[derive(Debug, Clone)]
pub struct SpatialHash {
    cell_size: f64,
    /// Map from cell key to set of object IDs in that cell.
    grid: HashMap<CellKey, Vec<usize>>,
    /// Map from object ID to tracked object data.
    objects: HashMap<usize, TrackedObject>,
    /// Number of objects currently stored.
    count: usize,
}

impl SpatialHash {
    /// Create a spatial hash with the given cell size.
    pub fn new(cell_size: f64) -> Self {
        assert!(cell_size > 0.0, "cell_size must be positive");
        Self {
            cell_size,
            grid: HashMap::new(),
            objects: HashMap::new(),
            count: 0,
        }
    }

    /// Cell size.
    pub fn cell_size(&self) -> f64 { self.cell_size }

    /// Number of stored objects.
    pub fn len(&self) -> usize { self.count }

    /// Whether the hash is empty.
    pub fn is_empty(&self) -> bool { self.count == 0 }

    /// Number of non-empty cells.
    pub fn cell_count(&self) -> usize {
        self.grid.values().filter(|v| !v.is_empty()).count()
    }

    /// Insert an object with the given ID and bounding box.
    pub fn insert(&mut self, id: usize, aabb: AABB) {
        let cells = aabb_cells(&aabb, self.cell_size);
        for &key in &cells {
            self.grid.entry(key).or_default().push(id);
        }
        self.objects.insert(id, TrackedObject { aabb, cells });
        self.count += 1;
    }

    /// Remove an object by ID.
    pub fn remove(&mut self, id: usize) -> bool {
        if let Some(tracked) = self.objects.remove(&id) {
            for key in &tracked.cells {
                if let Some(cell) = self.grid.get_mut(key) {
                    cell.retain(|x| *x != id);
                    if cell.is_empty() {
                        self.grid.remove(key);
                    }
                }
            }
            self.count -= 1;
            true
        } else {
            false
        }
    }

    /// Update an object's bounding box (re-hash if cells changed).
    pub fn update(&mut self, id: usize, new_aabb: AABB) {
        let new_cells = aabb_cells(&new_aabb, self.cell_size);

        if let Some(tracked) = self.objects.get(&id) {
            let old_cells = &tracked.cells;

            // Check if cells actually changed
            if old_cells.len() == new_cells.len()
                && old_cells.iter().all(|c| new_cells.contains(c))
            {
                // Same cells, just update the AABB
                if let Some(t) = self.objects.get_mut(&id) {
                    t.aabb = new_aabb;
                }
                return;
            }

            // Remove from old cells
            let old_cells_clone = old_cells.clone();
            for key in &old_cells_clone {
                if let Some(cell) = self.grid.get_mut(key) {
                    cell.retain(|x| *x != id);
                    if cell.is_empty() {
                        self.grid.remove(key);
                    }
                }
            }
        }

        // Add to new cells
        for &key in &new_cells {
            self.grid.entry(key).or_default().push(id);
        }

        if let Some(t) = self.objects.get_mut(&id) {
            t.aabb = new_aabb;
            t.cells = new_cells;
        }
    }

    /// Query all object IDs whose bounding boxes intersect the given region.
    pub fn query_region(&self, region: &AABB) -> Vec<usize> {
        let cells = aabb_cells(region, self.cell_size);
        let mut candidates = HashSet::new();
        for key in &cells {
            if let Some(ids) = self.grid.get(key) {
                for &id in ids {
                    candidates.insert(id);
                }
            }
        }
        // Fine check: verify actual AABB overlap
        candidates.into_iter().filter(|id| {
            if let Some(tracked) = self.objects.get(id) {
                tracked.aabb.intersects(region)
            } else {
                false
            }
        }).collect()
    }

    /// Query all neighbors of a given object (objects in the same or adjacent cells).
    pub fn query_neighbors(&self, id: usize) -> Vec<usize> {
        let tracked = match self.objects.get(&id) {
            Some(t) => t,
            None => return Vec::new(),
        };
        let aabb = tracked.aabb;
        let mut result = self.query_region(&aabb);
        result.retain(|x| *x != id);
        result
    }

    /// Query objects near a point within a given radius.
    pub fn query_point(&self, point: Vec2, radius: f64) -> Vec<usize> {
        let region = AABB::from_circle(point.x, point.y, radius);
        self.query_region(&region)
    }

    /// Generate all unique collision pairs (broad phase).
    /// Each pair (a, b) has a < b and appears exactly once.
    pub fn collision_pairs(&self) -> Vec<(usize, usize)> {
        let mut pairs = HashSet::new();
        for ids in self.grid.values() {
            for i in 0..ids.len() {
                for j in (i + 1)..ids.len() {
                    let a = ids[i].min(ids[j]);
                    let b = ids[i].max(ids[j]);
                    // Fine-check AABB overlap
                    let overlap = match (self.objects.get(&a), self.objects.get(&b)) {
                        (Some(ta), Some(tb)) => ta.aabb.intersects(&tb.aabb),
                        _ => false,
                    };
                    if overlap {
                        pairs.insert((a, b));
                    }
                }
            }
        }
        pairs.into_iter().collect()
    }

    /// Clear all objects from the hash.
    pub fn clear(&mut self) {
        self.grid.clear();
        self.objects.clear();
        self.count = 0;
    }

    /// Get the AABB for a stored object.
    pub fn get_aabb(&self, id: usize) -> Option<&AABB> {
        self.objects.get(&id).map(|t| &t.aabb)
    }

    /// Check if an object ID exists in the hash.
    pub fn contains(&self, id: usize) -> bool {
        self.objects.contains_key(&id)
    }

    /// All stored object IDs.
    pub fn object_ids(&self) -> Vec<usize> {
        self.objects.keys().copied().collect()
    }
}

// ── Cell Size Optimization ───────────────────────────────────

/// Estimate an optimal cell size given a collection of AABBs.
///
/// Uses the average object size * 2 as a heuristic (objects should span ~1-2 cells on average).
pub fn optimal_cell_size(aabbs: &[AABB]) -> f64 {
    if aabbs.is_empty() {
        return 1.0;
    }
    let total_size: f64 = aabbs.iter()
        .map(|a| a.width().max(a.height()))
        .sum();
    let avg = total_size / aabbs.len() as f64;
    (avg * 2.0).max(0.1)
}

/// Compute the average number of objects per cell (load factor).
pub fn load_factor(hash: &SpatialHash) -> f64 {
    let cells = hash.cell_count();
    if cells == 0 { return 0.0; }
    hash.len() as f64 / cells as f64
}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_query() {
        let mut sh = SpatialHash::new(10.0);
        sh.insert(0, AABB::new(0.0, 0.0, 5.0, 5.0));
        sh.insert(1, AABB::new(3.0, 3.0, 8.0, 8.0));
        sh.insert(2, AABB::new(50.0, 50.0, 55.0, 55.0));

        let near_origin = sh.query_region(&AABB::new(0.0, 0.0, 6.0, 6.0));
        assert!(near_origin.contains(&0));
        assert!(near_origin.contains(&1));
        assert!(!near_origin.contains(&2));
    }

    #[test]
    fn remove_object() {
        let mut sh = SpatialHash::new(10.0);
        sh.insert(0, AABB::new(0.0, 0.0, 5.0, 5.0));
        assert_eq!(sh.len(), 1);
        assert!(sh.remove(0));
        assert_eq!(sh.len(), 0);
        assert!(!sh.remove(0));
    }

    #[test]
    fn update_object() {
        let mut sh = SpatialHash::new(10.0);
        sh.insert(0, AABB::new(0.0, 0.0, 5.0, 5.0));
        // Move to a different cell
        sh.update(0, AABB::new(50.0, 50.0, 55.0, 55.0));
        let near_origin = sh.query_region(&AABB::new(0.0, 0.0, 6.0, 6.0));
        assert!(!near_origin.contains(&0));
        let far = sh.query_region(&AABB::new(49.0, 49.0, 56.0, 56.0));
        assert!(far.contains(&0));
    }

    #[test]
    fn collision_pairs_basic() {
        let mut sh = SpatialHash::new(10.0);
        sh.insert(0, AABB::new(0.0, 0.0, 5.0, 5.0));
        sh.insert(1, AABB::new(3.0, 3.0, 8.0, 8.0));
        sh.insert(2, AABB::new(50.0, 50.0, 55.0, 55.0));

        let pairs = sh.collision_pairs();
        assert!(pairs.contains(&(0, 1)));
        assert!(!pairs.contains(&(0, 2)));
        assert!(!pairs.contains(&(1, 2)));
    }

    #[test]
    fn no_self_pairs() {
        let mut sh = SpatialHash::new(10.0);
        sh.insert(0, AABB::new(0.0, 0.0, 5.0, 5.0));
        let pairs = sh.collision_pairs();
        assert!(pairs.is_empty());
    }

    #[test]
    fn query_neighbors() {
        let mut sh = SpatialHash::new(10.0);
        sh.insert(0, AABB::new(0.0, 0.0, 5.0, 5.0));
        sh.insert(1, AABB::new(3.0, 3.0, 8.0, 8.0));
        sh.insert(2, AABB::new(50.0, 50.0, 55.0, 55.0));

        let neighbors = sh.query_neighbors(0);
        assert!(neighbors.contains(&1));
        assert!(!neighbors.contains(&2));
        assert!(!neighbors.contains(&0)); // Should not include self
    }

    #[test]
    fn query_point() {
        let mut sh = SpatialHash::new(10.0);
        sh.insert(0, AABB::new(0.0, 0.0, 5.0, 5.0));
        sh.insert(1, AABB::new(50.0, 50.0, 55.0, 55.0));

        let near = sh.query_point(Vec2::new(3.0, 3.0), 5.0);
        assert!(near.contains(&0));
        assert!(!near.contains(&1));
    }

    #[test]
    fn clear_all() {
        let mut sh = SpatialHash::new(10.0);
        sh.insert(0, AABB::new(0.0, 0.0, 5.0, 5.0));
        sh.insert(1, AABB::new(10.0, 10.0, 15.0, 15.0));
        sh.clear();
        assert_eq!(sh.len(), 0);
        assert!(sh.is_empty());
        assert_eq!(sh.cell_count(), 0);
    }

    #[test]
    fn get_aabb() {
        let mut sh = SpatialHash::new(10.0);
        let aabb = AABB::new(1.0, 2.0, 3.0, 4.0);
        sh.insert(0, aabb);
        let retrieved = sh.get_aabb(0).unwrap();
        assert_eq!(*retrieved, aabb);
        assert!(sh.get_aabb(99).is_none());
    }

    #[test]
    fn contains_check() {
        let mut sh = SpatialHash::new(10.0);
        sh.insert(42, AABB::new(0.0, 0.0, 1.0, 1.0));
        assert!(sh.contains(42));
        assert!(!sh.contains(99));
    }

    #[test]
    fn object_ids() {
        let mut sh = SpatialHash::new(10.0);
        sh.insert(5, AABB::new(0.0, 0.0, 1.0, 1.0));
        sh.insert(10, AABB::new(5.0, 5.0, 6.0, 6.0));
        let ids = sh.object_ids();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&5));
        assert!(ids.contains(&10));
    }

    #[test]
    fn optimal_cell_size_calc() {
        let aabbs = vec![
            AABB::new(0.0, 0.0, 4.0, 4.0),
            AABB::new(0.0, 0.0, 6.0, 6.0),
            AABB::new(0.0, 0.0, 2.0, 2.0),
        ];
        let cs = optimal_cell_size(&aabbs);
        // Avg max dimension: (4+6+2)/3 = 4, optimal = 8
        assert!((cs - 8.0).abs() < 0.1);
    }

    #[test]
    fn optimal_cell_size_empty() {
        assert!((optimal_cell_size(&[]) - 1.0).abs() < 0.01);
    }

    #[test]
    fn load_factor_calc() {
        let mut sh = SpatialHash::new(10.0);
        sh.insert(0, AABB::new(0.0, 0.0, 5.0, 5.0));
        sh.insert(1, AABB::new(0.0, 0.0, 5.0, 5.0)); // Same cell
        let lf = load_factor(&sh);
        assert!(lf >= 1.0);
    }

    #[test]
    fn aabb_operations() {
        let a = AABB::new(0.0, 0.0, 10.0, 10.0);
        let b = AABB::new(5.0, 5.0, 15.0, 15.0);
        assert!(a.intersects(&b));
        let m = a.merged(&b);
        assert_eq!(m, AABB::new(0.0, 0.0, 15.0, 15.0));
        assert!(a.contains_point(Vec2::new(5.0, 5.0)));
        assert!(!a.contains_point(Vec2::new(15.0, 5.0)));
    }

    #[test]
    fn aabb_from_center() {
        let a = AABB::from_center_size(5.0, 5.0, 10.0, 10.0);
        assert!((a.min.x - 0.0).abs() < 0.01);
        assert!((a.max.x - 10.0).abs() < 0.01);
    }

    #[test]
    fn large_grid_stress() {
        let mut sh = SpatialHash::new(10.0);
        for i in 0..100 {
            let x = (i % 10) as f64 * 5.0;
            let y = (i / 10) as f64 * 5.0;
            sh.insert(i, AABB::new(x, y, x + 6.0, y + 6.0));
        }
        assert_eq!(sh.len(), 100);
        let pairs = sh.collision_pairs();
        // Adjacent objects (spaced 5.0, width 6.0) overlap by 1.0, so pairs exist
        assert!(!pairs.is_empty());
    }

    #[test]
    fn update_same_cell() {
        let mut sh = SpatialHash::new(100.0);
        sh.insert(0, AABB::new(0.0, 0.0, 5.0, 5.0));
        // Move within the same cell
        sh.update(0, AABB::new(1.0, 1.0, 6.0, 6.0));
        assert!(sh.contains(0));
        let aabb = sh.get_aabb(0).unwrap();
        assert!((aabb.min.x - 1.0).abs() < 0.01);
    }

    #[test]
    fn no_overlap_no_pairs() {
        let mut sh = SpatialHash::new(5.0);
        sh.insert(0, AABB::new(0.0, 0.0, 2.0, 2.0));
        sh.insert(1, AABB::new(3.0, 3.0, 4.0, 4.0)); // Same cell, but AABBs don't overlap
        let pairs = sh.collision_pairs();
        assert!(!pairs.iter().any(|p| *p == (0, 1)));
    }
}
