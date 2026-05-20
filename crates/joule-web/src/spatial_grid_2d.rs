//! Uniform spatial grid for 2D — configurable cell size, insert/remove/update
//! entities with AABB bounds.  Queries: by cell, by AABB overlap, by radius,
//! ray-march (DDA), neighbor cells.  Grid statistics.

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
    pub fn scale(self, s: f64) -> Self { Self { x: self.x * s, y: self.y * s } }
    pub fn length_sq(self) -> f64 { self.x * self.x + self.y * self.y }
    pub fn length(self) -> f64 { self.length_sq().sqrt() }
}

impl Default for Vec2 {
    fn default() -> Self { Self::zero() }
}

// ── AABB ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AABB {
    pub min: Vec2,
    pub max: Vec2,
}

impl AABB {
    pub fn new(min: Vec2, max: Vec2) -> Self { Self { min, max } }

    pub fn overlaps(&self, other: &AABB) -> bool {
        self.min.x <= other.max.x && self.max.x >= other.min.x
            && self.min.y <= other.max.y && self.max.y >= other.min.y
    }

    pub fn contains(&self, p: Vec2) -> bool {
        p.x >= self.min.x && p.x <= self.max.x && p.y >= self.min.y && p.y <= self.max.y
    }

    pub fn center(&self) -> Vec2 { self.min.add(self.max).scale(0.5) }
}

// ── Entity ID ────────────────────────────────────────────────

pub type EntityId = u64;

// ── Cell key ─────────────────────────────────────────────────

type CellKey = (i64, i64);

// ── Spatial Grid ─────────────────────────────────────────────

/// Uniform 2D spatial grid for fast spatial queries.
#[derive(Debug, Clone)]
pub struct SpatialGrid2D {
    cell_size: f64,
    inv_cell: f64,
    /// entity → AABB
    entities: HashMap<EntityId, AABB>,
    /// entity → set of cells it occupies
    entity_cells: HashMap<EntityId, Vec<CellKey>>,
    /// cell → list of entities
    cells: HashMap<CellKey, Vec<EntityId>>,
}

impl SpatialGrid2D {
    pub fn new(cell_size: f64) -> Self {
        assert!(cell_size > 0.0, "cell size must be positive");
        Self {
            cell_size,
            inv_cell: 1.0 / cell_size,
            entities: HashMap::new(),
            entity_cells: HashMap::new(),
            cells: HashMap::new(),
        }
    }

    /// Cell size.
    pub fn cell_size(&self) -> f64 { self.cell_size }

    /// Total entities in the grid.
    pub fn entity_count(&self) -> usize { self.entities.len() }

    /// Number of occupied cells.
    pub fn occupied_cell_count(&self) -> usize { self.cells.len() }

    /// Insert an entity with the given AABB.
    pub fn insert(&mut self, id: EntityId, aabb: AABB) {
        self.remove(id); // ensure clean state
        self.entities.insert(id, aabb);
        let keys = self.cells_for_aabb(&aabb);
        for &key in &keys {
            self.cells.entry(key).or_default().push(id);
        }
        self.entity_cells.insert(id, keys);
    }

    /// Remove an entity.
    pub fn remove(&mut self, id: EntityId) -> bool {
        if self.entities.remove(&id).is_none() { return false; }
        if let Some(keys) = self.entity_cells.remove(&id) {
            for key in &keys {
                if let Some(vec) = self.cells.get_mut(key) {
                    vec.retain(|e| *e != id);
                    if vec.is_empty() {
                        self.cells.remove(key);
                    }
                }
            }
        }
        true
    }

    /// Update an entity's AABB.
    pub fn update(&mut self, id: EntityId, aabb: AABB) {
        self.remove(id);
        self.insert(id, aabb);
    }

    /// Get all entities in a specific cell.
    pub fn entities_in_cell(&self, cx: i64, cy: i64) -> Vec<EntityId> {
        self.cells.get(&(cx, cy)).cloned().unwrap_or_default()
    }

    /// Query: all entities whose AABB overlaps the given AABB.
    pub fn query_aabb(&self, query: &AABB) -> Vec<EntityId> {
        let keys = self.cells_for_aabb(query);
        let mut seen = HashSet::new();
        let mut result = Vec::new();
        for key in &keys {
            if let Some(ids) = self.cells.get(key) {
                for &id in ids {
                    if seen.insert(id) {
                        if let Some(ent_aabb) = self.entities.get(&id) {
                            if ent_aabb.overlaps(query) {
                                result.push(id);
                            }
                        }
                    }
                }
            }
        }
        result
    }

    /// Query: all entities within `radius` of `center` (distance check on AABB center).
    pub fn query_radius(&self, center: Vec2, radius: f64) -> Vec<EntityId> {
        let query_aabb = AABB::new(
            Vec2::new(center.x - radius, center.y - radius),
            Vec2::new(center.x + radius, center.y + radius),
        );
        let candidates = self.query_aabb(&query_aabb);
        let r2 = radius * radius;
        candidates.into_iter().filter(|id| {
            if let Some(aabb) = self.entities.get(&id) {
                let c = aabb.center();
                c.sub(center).length_sq() <= r2
            } else {
                false
            }
        }).collect()
    }

    /// Ray-march through the grid using DDA (Digital Differential Analyzer).
    /// Returns entity IDs intersected along the ray, up to `max_dist`.
    pub fn ray_march(&self, origin: Vec2, direction: Vec2, max_dist: f64) -> Vec<EntityId> {
        let dir_len = direction.length();
        if dir_len < 1e-12 { return Vec::new(); }
        let dir = Vec2::new(direction.x / dir_len, direction.y / dir_len);

        let mut cx = (origin.x * self.inv_cell).floor() as i64;
        let mut cy = (origin.y * self.inv_cell).floor() as i64;

        let step_x: i64 = if dir.x >= 0.0 { 1 } else { -1 };
        let step_y: i64 = if dir.y >= 0.0 { 1 } else { -1 };

        let t_delta_x = if dir.x.abs() > 1e-12 { self.cell_size / dir.x.abs() } else { f64::MAX };
        let t_delta_y = if dir.y.abs() > 1e-12 { self.cell_size / dir.y.abs() } else { f64::MAX };

        let next_x = if dir.x >= 0.0 { (cx + 1) as f64 * self.cell_size } else { cx as f64 * self.cell_size };
        let next_y = if dir.y >= 0.0 { (cy + 1) as f64 * self.cell_size } else { cy as f64 * self.cell_size };

        let mut t_max_x = if dir.x.abs() > 1e-12 { (next_x - origin.x) / dir.x } else { f64::MAX };
        let mut t_max_y = if dir.y.abs() > 1e-12 { (next_y - origin.y) / dir.y } else { f64::MAX };

        let mut seen = HashSet::new();
        let mut result = Vec::new();
        let mut t = 0.0_f64;

        while t < max_dist {
            if let Some(ids) = self.cells.get(&(cx, cy)) {
                for &id in ids {
                    if seen.insert(id) {
                        result.push(id);
                    }
                }
            }

            if t_max_x < t_max_y {
                t = t_max_x;
                t_max_x += t_delta_x;
                cx += step_x;
            } else {
                t = t_max_y;
                t_max_y += t_delta_y;
                cy += step_y;
            }
        }
        result
    }

    /// Get all entities in the 8 neighboring cells around (cx, cy) (not including the cell itself).
    pub fn neighbors(&self, cx: i64, cy: i64) -> Vec<EntityId> {
        let mut seen = HashSet::new();
        let mut result = Vec::new();
        for dx in -1..=1_i64 {
            for dy in -1..=1_i64 {
                if dx == 0 && dy == 0 { continue; }
                if let Some(ids) = self.cells.get(&(cx + dx, cy + dy)) {
                    for &id in ids {
                        if seen.insert(id) { result.push(id); }
                    }
                }
            }
        }
        result
    }

    /// Get the cell key for a point.
    pub fn cell_for_point(&self, p: Vec2) -> (i64, i64) {
        ((p.x * self.inv_cell).floor() as i64, (p.y * self.inv_cell).floor() as i64)
    }

    // ── Statistics ──

    /// Max entities in any single cell.
    pub fn max_entities_per_cell(&self) -> usize {
        self.cells.values().map(|v| v.len()).max().unwrap_or(0)
    }

    /// Average entities per occupied cell.
    pub fn avg_entities_per_cell(&self) -> f64 {
        if self.cells.is_empty() { return 0.0; }
        let total: usize = self.cells.values().map(|v| v.len()).sum();
        total as f64 / self.cells.len() as f64
    }

    /// Fraction of cells that are occupied (requires knowing total grid extent).
    /// Returns occupied / total_occupied (just the count for reporting).
    pub fn occupancy(&self) -> usize {
        self.cells.len()
    }

    // ── Internal ──

    fn cells_for_aabb(&self, aabb: &AABB) -> Vec<CellKey> {
        let cx0 = (aabb.min.x * self.inv_cell).floor() as i64;
        let cy0 = (aabb.min.y * self.inv_cell).floor() as i64;
        let cx1 = (aabb.max.x * self.inv_cell).floor() as i64;
        let cy1 = (aabb.max.y * self.inv_cell).floor() as i64;
        let mut keys = Vec::new();
        for cx in cx0..=cx1 {
            for cy in cy0..=cy1 {
                keys.push((cx, cy));
            }
        }
        keys
    }
}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn aabb(x1: f64, y1: f64, x2: f64, y2: f64) -> AABB {
        AABB::new(Vec2::new(x1, y1), Vec2::new(x2, y2))
    }

    #[test]
    fn insert_and_count() {
        let mut g = SpatialGrid2D::new(10.0);
        g.insert(1, aabb(0.0, 0.0, 5.0, 5.0));
        g.insert(2, aabb(20.0, 20.0, 25.0, 25.0));
        assert_eq!(g.entity_count(), 2);
    }

    #[test]
    fn remove_entity() {
        let mut g = SpatialGrid2D::new(10.0);
        g.insert(1, aabb(0.0, 0.0, 5.0, 5.0));
        assert!(g.remove(1));
        assert_eq!(g.entity_count(), 0);
        assert!(!g.remove(1)); // already removed
    }

    #[test]
    fn update_entity() {
        let mut g = SpatialGrid2D::new(10.0);
        g.insert(1, aabb(0.0, 0.0, 5.0, 5.0));
        g.update(1, aabb(50.0, 50.0, 55.0, 55.0));
        let result = g.query_aabb(&aabb(0.0, 0.0, 10.0, 10.0));
        assert!(!result.contains(&1));
        let result = g.query_aabb(&aabb(49.0, 49.0, 56.0, 56.0));
        assert!(result.contains(&1));
    }

    #[test]
    fn query_aabb_overlap() {
        let mut g = SpatialGrid2D::new(10.0);
        g.insert(1, aabb(0.0, 0.0, 5.0, 5.0));
        g.insert(2, aabb(3.0, 3.0, 8.0, 8.0));
        g.insert(3, aabb(50.0, 50.0, 55.0, 55.0));
        let result = g.query_aabb(&aabb(2.0, 2.0, 6.0, 6.0));
        assert!(result.contains(&1));
        assert!(result.contains(&2));
        assert!(!result.contains(&3));
    }

    #[test]
    fn query_aabb_no_match() {
        let mut g = SpatialGrid2D::new(10.0);
        g.insert(1, aabb(0.0, 0.0, 5.0, 5.0));
        let result = g.query_aabb(&aabb(100.0, 100.0, 110.0, 110.0));
        assert!(result.is_empty());
    }

    #[test]
    fn query_radius() {
        let mut g = SpatialGrid2D::new(10.0);
        g.insert(1, aabb(0.0, 0.0, 2.0, 2.0));   // center (1,1)
        g.insert(2, aabb(50.0, 50.0, 52.0, 52.0)); // center (51,51)
        let result = g.query_radius(Vec2::new(0.0, 0.0), 5.0);
        assert!(result.contains(&1));
        assert!(!result.contains(&2));
    }

    #[test]
    fn query_radius_empty() {
        let g = SpatialGrid2D::new(10.0);
        let result = g.query_radius(Vec2::new(0.0, 0.0), 100.0);
        assert!(result.is_empty());
    }

    #[test]
    fn entities_in_cell() {
        let mut g = SpatialGrid2D::new(10.0);
        g.insert(1, aabb(0.0, 0.0, 5.0, 5.0));
        g.insert(2, aabb(1.0, 1.0, 3.0, 3.0));
        let result = g.entities_in_cell(0, 0);
        assert!(result.contains(&1));
        assert!(result.contains(&2));
    }

    #[test]
    fn ray_march_basic() {
        let mut g = SpatialGrid2D::new(10.0);
        g.insert(1, aabb(15.0, -2.0, 18.0, 2.0));
        let result = g.ray_march(Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0), 30.0);
        assert!(result.contains(&1));
    }

    #[test]
    fn ray_march_miss() {
        let mut g = SpatialGrid2D::new(10.0);
        g.insert(1, aabb(15.0, 15.0, 18.0, 18.0));
        let result = g.ray_march(Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0), 30.0);
        assert!(!result.contains(&1));
    }

    #[test]
    fn ray_march_zero_direction() {
        let g = SpatialGrid2D::new(10.0);
        let result = g.ray_march(Vec2::new(0.0, 0.0), Vec2::zero(), 30.0);
        assert!(result.is_empty());
    }

    #[test]
    fn neighbors_query() {
        let mut g = SpatialGrid2D::new(10.0);
        g.insert(1, aabb(15.0, 5.0, 18.0, 8.0));   // cell (1, 0)
        g.insert(2, aabb(-5.0, -5.0, -2.0, -2.0));  // cell (-1, -1)
        g.insert(3, aabb(5.0, 5.0, 8.0, 8.0));      // cell (0, 0)
        let result = g.neighbors(0, 0);
        assert!(result.contains(&1)); // adjacent cell
    }

    #[test]
    fn cell_for_point() {
        let g = SpatialGrid2D::new(10.0);
        assert_eq!(g.cell_for_point(Vec2::new(5.0, 5.0)), (0, 0));
        assert_eq!(g.cell_for_point(Vec2::new(15.0, 25.0)), (1, 2));
        assert_eq!(g.cell_for_point(Vec2::new(-5.0, -5.0)), (-1, -1));
    }

    #[test]
    fn statistics_max_per_cell() {
        let mut g = SpatialGrid2D::new(10.0);
        g.insert(1, aabb(0.0, 0.0, 1.0, 1.0));
        g.insert(2, aabb(0.5, 0.5, 1.5, 1.5));
        g.insert(3, aabb(0.0, 0.0, 0.5, 0.5));
        assert!(g.max_entities_per_cell() >= 2);
    }

    #[test]
    fn statistics_avg() {
        let mut g = SpatialGrid2D::new(10.0);
        g.insert(1, aabb(0.0, 0.0, 1.0, 1.0));
        g.insert(2, aabb(50.0, 50.0, 51.0, 51.0));
        let avg = g.avg_entities_per_cell();
        assert!(avg >= 1.0);
    }

    #[test]
    fn occupied_cell_count() {
        let mut g = SpatialGrid2D::new(10.0);
        assert_eq!(g.occupied_cell_count(), 0);
        g.insert(1, aabb(0.0, 0.0, 1.0, 1.0));
        assert!(g.occupied_cell_count() >= 1);
    }

    #[test]
    fn large_entity_spans_cells() {
        let mut g = SpatialGrid2D::new(10.0);
        g.insert(1, aabb(0.0, 0.0, 25.0, 25.0)); // spans 3x3 cells
        assert!(g.occupied_cell_count() >= 4);
    }

    #[test]
    fn aabb_overlap_check() {
        let a = aabb(0.0, 0.0, 5.0, 5.0);
        let b = aabb(3.0, 3.0, 8.0, 8.0);
        assert!(a.overlaps(&b));
    }

    #[test]
    fn aabb_no_overlap_check() {
        let a = aabb(0.0, 0.0, 1.0, 1.0);
        let b = aabb(5.0, 5.0, 6.0, 6.0);
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn duplicate_insert_replaces() {
        let mut g = SpatialGrid2D::new(10.0);
        g.insert(1, aabb(0.0, 0.0, 5.0, 5.0));
        g.insert(1, aabb(50.0, 50.0, 55.0, 55.0));
        assert_eq!(g.entity_count(), 1);
        let result = g.query_aabb(&aabb(0.0, 0.0, 10.0, 10.0));
        assert!(!result.contains(&1));
    }
}
