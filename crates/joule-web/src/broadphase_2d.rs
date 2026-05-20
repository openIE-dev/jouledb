//! 2D broadphase collision detection — spatial hashing, sweep-and-prune,
//! grid-based dynamic cell assignment.  Returns pairs of potentially colliding body IDs.
//! Static bodies skip re-hashing.  Pair cache avoids redundant narrowphase checks.

use std::collections::{HashMap, HashSet};

// ── Types ────────────────────────────────────────────────────

pub type BodyId = u64;

/// Axis-aligned bounding box for broadphase.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AABB {
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
}

impl AABB {
    pub fn new(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Self {
        Self { min_x, min_y, max_x, max_y }
    }

    pub fn overlaps(&self, other: &AABB) -> bool {
        self.min_x <= other.max_x && self.max_x >= other.min_x
            && self.min_y <= other.max_y && self.max_y >= other.min_y
    }
}

/// Whether a proxy is static or dynamic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProxyKind {
    Static,
    Dynamic,
}

/// Internal proxy stored per body.
#[derive(Debug, Clone)]
struct Proxy {
    id: BodyId,
    aabb: AABB,
    kind: ProxyKind,
}

// ── Ordered pair ─────────────────────────────────────────────

/// Canonical pair: always (min, max) so (a,b) == (b,a).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Pair(pub BodyId, pub BodyId);

impl Pair {
    pub fn new(a: BodyId, b: BodyId) -> Self {
        if a < b { Self(a, b) } else { Self(b, a) }
    }
}

// ── Spatial Hash Broadphase ──────────────────────────────────

/// Spatial-hash broadphase that buckets AABBs into grid cells.
#[derive(Debug)]
pub struct SpatialHashBroadphase {
    cell_size: f64,
    inv_cell: f64,
    proxies: HashMap<BodyId, Proxy>,
    /// cell_key → set of body IDs in that cell.
    cells: HashMap<(i64, i64), Vec<BodyId>>,
    /// Cached pairs from the last `compute_pairs` call.
    pair_cache: HashSet<Pair>,
}

impl SpatialHashBroadphase {
    pub fn new(cell_size: f64) -> Self {
        assert!(cell_size > 0.0);
        Self {
            cell_size,
            inv_cell: 1.0 / cell_size,
            proxies: HashMap::new(),
            cells: HashMap::new(),
            pair_cache: HashSet::new(),
        }
    }

    /// Insert or update a proxy.
    pub fn insert(&mut self, id: BodyId, aabb: AABB, kind: ProxyKind) {
        self.remove_from_cells(id);
        self.proxies.insert(id, Proxy { id, aabb, kind });
        self.add_to_cells(id, &aabb);
    }

    /// Update the AABB for a dynamic body.  Static bodies should use `insert`.
    pub fn update(&mut self, id: BodyId, aabb: AABB) {
        if let Some(proxy) = self.proxies.get(&id) {
            let kind = proxy.kind;
            self.remove_from_cells(id);
            self.proxies.insert(id, Proxy { id, aabb, kind });
            self.add_to_cells(id, &aabb);
        }
    }

    /// Remove a body from the broadphase.
    pub fn remove(&mut self, id: BodyId) {
        self.remove_from_cells(id);
        self.proxies.remove(&id);
    }

    /// Compute all overlapping pairs.  Results are cached.
    pub fn compute_pairs(&mut self) -> &HashSet<Pair> {
        self.pair_cache.clear();
        // Collect cell keys to avoid borrow issue
        let keys: Vec<(i64, i64)> = self.cells.keys().copied().collect();
        for key in &keys {
            let ids: Vec<BodyId> = match self.cells.get(key) {
                Some(v) => v.clone(),
                None => continue,
            };
            let n = ids.len();
            for i in 0..n {
                for j in (i + 1)..n {
                    let a = ids[i];
                    let b = ids[j];
                    let pair = Pair::new(a, b);
                    if self.pair_cache.contains(&pair) { continue; }
                    // AABB overlap check
                    let pa = match self.proxies.get(&a) { Some(p) => &p.aabb, None => continue };
                    let pb = match self.proxies.get(&b) { Some(p) => &p.aabb, None => continue };
                    if pa.overlaps(pb) {
                        self.pair_cache.insert(pair);
                    }
                }
            }
        }
        &self.pair_cache
    }

    /// Return the number of proxies.
    pub fn len(&self) -> usize { self.proxies.len() }
    pub fn is_empty(&self) -> bool { self.proxies.is_empty() }

    /// Number of occupied cells.
    pub fn cell_count(&self) -> usize { self.cells.len() }

    // ── internal ──

    fn cell_key(&self, x: f64, y: f64) -> (i64, i64) {
        ((x * self.inv_cell).floor() as i64, (y * self.inv_cell).floor() as i64)
    }

    fn add_to_cells(&mut self, id: BodyId, aabb: &AABB) {
        let (cx0, cy0) = self.cell_key(aabb.min_x, aabb.min_y);
        let (cx1, cy1) = self.cell_key(aabb.max_x, aabb.max_y);
        for cx in cx0..=cx1 {
            for cy in cy0..=cy1 {
                self.cells.entry((cx, cy)).or_default().push(id);
            }
        }
    }

    fn remove_from_cells(&mut self, id: BodyId) {
        // Collect keys that contain this id
        let keys: Vec<(i64, i64)> = self.cells.keys().copied().collect();
        for key in keys {
            if let Some(vec) = self.cells.get_mut(&key) {
                vec.retain(|x| *x != id);
                if vec.is_empty() {
                    self.cells.remove(&key);
                }
            }
        }
    }
}

// ── Sweep-and-Prune ──────────────────────────────────────────

/// One-axis sweep-and-prune broadphase (X-axis).
#[derive(Debug)]
pub struct SweepAndPrune {
    proxies: Vec<(BodyId, AABB)>,
}

impl SweepAndPrune {
    pub fn new() -> Self { Self { proxies: Vec::new() } }

    pub fn clear(&mut self) { self.proxies.clear(); }

    pub fn insert(&mut self, id: BodyId, aabb: AABB) {
        self.proxies.push((id, aabb));
    }

    /// Sort on X-axis and sweep to find overlapping pairs.
    pub fn compute_pairs(&mut self) -> Vec<Pair> {
        self.proxies.sort_by(|a, b| a.1.min_x.partial_cmp(&b.1.min_x).unwrap());
        let mut pairs = Vec::new();
        let n = self.proxies.len();
        for i in 0..n {
            let (id_a, ref aabb_a) = self.proxies[i];
            for j in (i + 1)..n {
                let (id_b, ref aabb_b) = self.proxies[j];
                if aabb_b.min_x > aabb_a.max_x { break; }
                // Overlap on X confirmed by sweep; check Y.
                if aabb_a.min_y <= aabb_b.max_y && aabb_a.max_y >= aabb_b.min_y {
                    pairs.push(Pair::new(id_a, id_b));
                }
            }
        }
        pairs
    }

    pub fn len(&self) -> usize { self.proxies.len() }
    pub fn is_empty(&self) -> bool { self.proxies.is_empty() }
}

impl Default for SweepAndPrune {
    fn default() -> Self { Self::new() }
}

// ── Grid Broadphase ──────────────────────────────────────────

/// Simple uniform-grid broadphase with separate tracking for static / dynamic.
#[derive(Debug)]
pub struct GridBroadphase {
    cell_size: f64,
    inv_cell: f64,
    statics: HashMap<BodyId, AABB>,
    dynamics: HashMap<BodyId, AABB>,
    static_cells: HashMap<(i64, i64), Vec<BodyId>>,
    dynamic_cells: HashMap<(i64, i64), Vec<BodyId>>,
}

impl GridBroadphase {
    pub fn new(cell_size: f64) -> Self {
        assert!(cell_size > 0.0);
        Self {
            cell_size,
            inv_cell: 1.0 / cell_size,
            statics: HashMap::new(),
            dynamics: HashMap::new(),
            static_cells: HashMap::new(),
            dynamic_cells: HashMap::new(),
        }
    }

    pub fn insert_static(&mut self, id: BodyId, aabb: AABB) {
        self.statics.insert(id, aabb);
        self.add_to_map(&aabb, id, true);
    }

    pub fn insert_dynamic(&mut self, id: BodyId, aabb: AABB) {
        self.dynamics.insert(id, aabb);
        self.add_to_map(&aabb, id, false);
    }

    /// Update a dynamic body AABB.  Statics don't need updating.
    pub fn update_dynamic(&mut self, id: BodyId, aabb: AABB) {
        self.remove_dynamic_from_cells(id);
        self.dynamics.insert(id, aabb);
        self.add_to_map(&aabb, id, false);
    }

    pub fn remove(&mut self, id: BodyId) {
        if self.statics.remove(&id).is_some() {
            Self::remove_from_map(id, &mut self.static_cells);
        }
        if self.dynamics.remove(&id).is_some() {
            self.remove_dynamic_from_cells(id);
        }
    }

    /// Returns pairs: dynamic-dynamic and dynamic-static.
    pub fn compute_pairs(&self) -> Vec<Pair> {
        let mut seen = HashSet::new();
        let mut pairs = Vec::new();

        let keys: Vec<(i64, i64)> = self.dynamic_cells.keys().copied().collect();
        for key in &keys {
            let dyn_ids: &Vec<BodyId> = match self.dynamic_cells.get(key) {
                Some(v) => v,
                None => continue,
            };
            // dynamic-dynamic
            for i in 0..dyn_ids.len() {
                for j in (i + 1)..dyn_ids.len() {
                    let p = Pair::new(dyn_ids[i], dyn_ids[j]);
                    if seen.insert(p) {
                        if let (Some(a), Some(b)) = (self.dynamics.get(&dyn_ids[i]), self.dynamics.get(&dyn_ids[j])) {
                            if a.overlaps(b) { pairs.push(p); }
                        }
                    }
                }
            }
            // dynamic-static
            if let Some(stat_ids) = self.static_cells.get(key) {
                for &d in dyn_ids.iter() {
                    for &s in stat_ids.iter() {
                        let p = Pair::new(d, s);
                        if seen.insert(p) {
                            if let (Some(a), Some(b)) = (self.dynamics.get(&d), self.statics.get(&s)) {
                                if a.overlaps(b) { pairs.push(p); }
                            }
                        }
                    }
                }
            }
        }
        pairs
    }

    pub fn len(&self) -> usize { self.statics.len() + self.dynamics.len() }
    pub fn is_empty(&self) -> bool { self.statics.is_empty() && self.dynamics.is_empty() }

    // ── internal ──

    fn cell_key(&self, x: f64, y: f64) -> (i64, i64) {
        ((x * self.inv_cell).floor() as i64, (y * self.inv_cell).floor() as i64)
    }

    fn add_to_map(&mut self, aabb: &AABB, id: BodyId, is_static: bool) {
        let (cx0, cy0) = self.cell_key(aabb.min_x, aabb.min_y);
        let (cx1, cy1) = self.cell_key(aabb.max_x, aabb.max_y);
        for cx in cx0..=cx1 {
            for cy in cy0..=cy1 {
                if is_static {
                    self.static_cells.entry((cx, cy)).or_default().push(id);
                } else {
                    self.dynamic_cells.entry((cx, cy)).or_default().push(id);
                }
            }
        }
    }

    fn remove_dynamic_from_cells(&mut self, id: BodyId) {
        Self::remove_from_map(id, &mut self.dynamic_cells);
    }

    fn remove_from_map(id: BodyId, map: &mut HashMap<(i64, i64), Vec<BodyId>>) {
        let keys: Vec<(i64, i64)> = map.keys().copied().collect();
        for key in keys {
            if let Some(vec) = map.get_mut(&key) {
                vec.retain(|x| *x != id);
                if vec.is_empty() {
                    map.remove(&key);
                }
            }
        }
    }
}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn aabb(x1: f64, y1: f64, x2: f64, y2: f64) -> AABB {
        AABB::new(x1, y1, x2, y2)
    }

    // ── AABB ──

    #[test]
    fn aabb_overlap_test() {
        assert!(aabb(0.0, 0.0, 2.0, 2.0).overlaps(&aabb(1.0, 1.0, 3.0, 3.0)));
    }

    #[test]
    fn aabb_no_overlap_test() {
        assert!(!aabb(0.0, 0.0, 1.0, 1.0).overlaps(&aabb(2.0, 2.0, 3.0, 3.0)));
    }

    // ── Pair ──

    #[test]
    fn pair_canonical() {
        assert_eq!(Pair::new(3, 1), Pair::new(1, 3));
        assert_eq!(Pair::new(5, 5), Pair(5, 5));
    }

    // ── Spatial Hash ──

    #[test]
    fn spatial_hash_basic() {
        let mut bp = SpatialHashBroadphase::new(10.0);
        bp.insert(1, aabb(0.0, 0.0, 5.0, 5.0), ProxyKind::Dynamic);
        bp.insert(2, aabb(3.0, 3.0, 8.0, 8.0), ProxyKind::Dynamic);
        let pairs = bp.compute_pairs();
        assert!(pairs.contains(&Pair::new(1, 2)));
    }

    #[test]
    fn spatial_hash_no_overlap() {
        let mut bp = SpatialHashBroadphase::new(10.0);
        bp.insert(1, aabb(0.0, 0.0, 1.0, 1.0), ProxyKind::Dynamic);
        bp.insert(2, aabb(50.0, 50.0, 51.0, 51.0), ProxyKind::Dynamic);
        let pairs = bp.compute_pairs();
        assert!(pairs.is_empty());
    }

    #[test]
    fn spatial_hash_remove() {
        let mut bp = SpatialHashBroadphase::new(10.0);
        bp.insert(1, aabb(0.0, 0.0, 5.0, 5.0), ProxyKind::Dynamic);
        bp.insert(2, aabb(3.0, 3.0, 8.0, 8.0), ProxyKind::Dynamic);
        bp.remove(1);
        let pairs = bp.compute_pairs();
        assert!(pairs.is_empty());
    }

    #[test]
    fn spatial_hash_update() {
        let mut bp = SpatialHashBroadphase::new(10.0);
        bp.insert(1, aabb(0.0, 0.0, 1.0, 1.0), ProxyKind::Dynamic);
        bp.insert(2, aabb(5.0, 5.0, 6.0, 6.0), ProxyKind::Dynamic);
        assert!(bp.compute_pairs().is_empty());
        bp.update(1, aabb(4.5, 4.5, 5.5, 5.5));
        assert!(!bp.compute_pairs().is_empty());
    }

    #[test]
    fn spatial_hash_many_bodies() {
        let mut bp = SpatialHashBroadphase::new(5.0);
        for i in 0..20 {
            let x = (i as f64) * 0.5;
            bp.insert(i, aabb(x, x, x + 1.0, x + 1.0), ProxyKind::Dynamic);
        }
        let pairs = bp.compute_pairs();
        assert!(!pairs.is_empty());
    }

    #[test]
    fn spatial_hash_cell_count() {
        let mut bp = SpatialHashBroadphase::new(10.0);
        bp.insert(1, aabb(0.0, 0.0, 1.0, 1.0), ProxyKind::Dynamic);
        assert!(bp.cell_count() >= 1);
    }

    #[test]
    fn spatial_hash_static_bodies() {
        let mut bp = SpatialHashBroadphase::new(10.0);
        bp.insert(1, aabb(0.0, 0.0, 5.0, 5.0), ProxyKind::Static);
        bp.insert(2, aabb(3.0, 3.0, 8.0, 8.0), ProxyKind::Dynamic);
        let pairs = bp.compute_pairs();
        assert!(pairs.contains(&Pair::new(1, 2)));
    }

    // ── Sweep-and-Prune ──

    #[test]
    fn sap_basic() {
        let mut sap = SweepAndPrune::new();
        sap.insert(1, aabb(0.0, 0.0, 3.0, 3.0));
        sap.insert(2, aabb(2.0, 2.0, 5.0, 5.0));
        let pairs = sap.compute_pairs();
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0], Pair::new(1, 2));
    }

    #[test]
    fn sap_no_overlap() {
        let mut sap = SweepAndPrune::new();
        sap.insert(1, aabb(0.0, 0.0, 1.0, 1.0));
        sap.insert(2, aabb(5.0, 5.0, 6.0, 6.0));
        assert!(sap.compute_pairs().is_empty());
    }

    #[test]
    fn sap_x_overlap_y_miss() {
        let mut sap = SweepAndPrune::new();
        sap.insert(1, aabb(0.0, 0.0, 3.0, 1.0));
        sap.insert(2, aabb(2.0, 5.0, 5.0, 6.0));
        assert!(sap.compute_pairs().is_empty());
    }

    #[test]
    fn sap_multiple() {
        let mut sap = SweepAndPrune::new();
        sap.insert(1, aabb(0.0, 0.0, 2.0, 2.0));
        sap.insert(2, aabb(1.0, 0.0, 3.0, 2.0));
        sap.insert(3, aabb(2.5, 0.0, 4.0, 2.0));
        let pairs = sap.compute_pairs();
        assert!(pairs.contains(&Pair::new(1, 2)));
        assert!(pairs.contains(&Pair::new(2, 3)));
    }

    // ── Grid Broadphase ──

    #[test]
    fn grid_dynamic_pair() {
        let mut g = GridBroadphase::new(10.0);
        g.insert_dynamic(1, aabb(0.0, 0.0, 5.0, 5.0));
        g.insert_dynamic(2, aabb(3.0, 3.0, 8.0, 8.0));
        let pairs = g.compute_pairs();
        assert!(pairs.contains(&Pair::new(1, 2)));
    }

    #[test]
    fn grid_dynamic_static_pair() {
        let mut g = GridBroadphase::new(10.0);
        g.insert_static(1, aabb(0.0, 0.0, 5.0, 5.0));
        g.insert_dynamic(2, aabb(3.0, 3.0, 8.0, 8.0));
        let pairs = g.compute_pairs();
        assert!(pairs.contains(&Pair::new(1, 2)));
    }

    #[test]
    fn grid_no_static_static() {
        let mut g = GridBroadphase::new(10.0);
        g.insert_static(1, aabb(0.0, 0.0, 5.0, 5.0));
        g.insert_static(2, aabb(3.0, 3.0, 8.0, 8.0));
        // Static-static pairs are never generated.
        let pairs = g.compute_pairs();
        assert!(pairs.is_empty());
    }

    #[test]
    fn grid_update_dynamic() {
        let mut g = GridBroadphase::new(10.0);
        g.insert_dynamic(1, aabb(0.0, 0.0, 1.0, 1.0));
        g.insert_dynamic(2, aabb(5.0, 5.0, 6.0, 6.0));
        assert!(g.compute_pairs().is_empty());
        g.update_dynamic(1, aabb(4.5, 4.5, 5.5, 5.5));
        assert!(!g.compute_pairs().is_empty());
    }

    #[test]
    fn grid_remove() {
        let mut g = GridBroadphase::new(10.0);
        g.insert_dynamic(1, aabb(0.0, 0.0, 5.0, 5.0));
        g.insert_dynamic(2, aabb(3.0, 3.0, 8.0, 8.0));
        g.remove(1);
        assert!(g.compute_pairs().is_empty());
    }

    #[test]
    fn grid_len() {
        let mut g = GridBroadphase::new(10.0);
        g.insert_static(1, aabb(0.0, 0.0, 1.0, 1.0));
        g.insert_dynamic(2, aabb(0.0, 0.0, 1.0, 1.0));
        assert_eq!(g.len(), 2);
    }
}
