//! World chunk management system.
//!
//! Manages spatial chunks identified by (x, z) integer coordinates. Provides
//! active-radius loading around a camera position, priority-ordered loading
//! (nearest first), per-chunk LOD, neighbor queries, load/unload callbacks,
//! and a ring buffer of recently unloaded chunks for fast reload.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// ── Chunk coordinates ──────────────────────────────────────────

/// Integer (x, z) position identifying a chunk in the world grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChunkCoord {
    pub x: i32,
    pub z: i32,
}

impl ChunkCoord {
    pub fn new(x: i32, z: i32) -> Self {
        Self { x, z }
    }

    /// Squared distance to another chunk coordinate.
    pub fn distance_sq(&self, other: &ChunkCoord) -> i64 {
        let dx = (self.x as i64) - (other.x as i64);
        let dz = (self.z as i64) - (other.z as i64);
        dx * dx + dz * dz
    }

    /// Manhattan distance to another chunk coordinate.
    pub fn manhattan(&self, other: &ChunkCoord) -> i32 {
        (self.x - other.x).abs() + (self.z - other.z).abs()
    }

    /// Orthogonal neighbors (N, S, E, W).
    pub fn neighbors(&self) -> [ChunkCoord; 4] {
        [
            ChunkCoord::new(self.x, self.z - 1),
            ChunkCoord::new(self.x, self.z + 1),
            ChunkCoord::new(self.x - 1, self.z),
            ChunkCoord::new(self.x + 1, self.z),
        ]
    }

    /// 8-connected neighbors (orthogonal + diagonal).
    pub fn neighbors_8(&self) -> [ChunkCoord; 8] {
        [
            ChunkCoord::new(self.x - 1, self.z - 1),
            ChunkCoord::new(self.x, self.z - 1),
            ChunkCoord::new(self.x + 1, self.z - 1),
            ChunkCoord::new(self.x - 1, self.z),
            ChunkCoord::new(self.x + 1, self.z),
            ChunkCoord::new(self.x - 1, self.z + 1),
            ChunkCoord::new(self.x, self.z + 1),
            ChunkCoord::new(self.x + 1, self.z + 1),
        ]
    }
}

impl fmt::Display for ChunkCoord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.z)
    }
}

// ── Chunk state ────────────────────────────────────────────────

/// Lifecycle state of a chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChunkState {
    Unloaded,
    Loading,
    Loaded,
    Unloading,
}

impl fmt::Display for ChunkState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChunkState::Unloaded => write!(f, "unloaded"),
            ChunkState::Loading => write!(f, "loading"),
            ChunkState::Loaded => write!(f, "loaded"),
            ChunkState::Unloading => write!(f, "unloading"),
        }
    }
}

// ── Level of detail ────────────────────────────────────────────

/// Level-of-detail tier for a chunk based on distance from camera.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum LodLevel {
    /// Full detail — closest ring.
    Full,
    /// Half detail — mid range.
    Half,
    /// Quarter detail — far range.
    Quarter,
    /// Minimal — distant, barely visible.
    Minimal,
}

impl LodLevel {
    /// Compute LOD from squared distance and radius.
    pub fn from_distance_sq(dist_sq: i64, radius: i32) -> Self {
        let r = radius as i64;
        let quarter = (r * r) / 4;
        let half = (r * r) / 2;
        let three_quarter = (r * r * 3) / 4;
        if dist_sq <= quarter {
            LodLevel::Full
        } else if dist_sq <= half {
            LodLevel::Half
        } else if dist_sq <= three_quarter {
            LodLevel::Quarter
        } else {
            LodLevel::Minimal
        }
    }

    /// Detail factor: 1.0 for Full, 0.5 for Half, etc.
    pub fn detail_factor(&self) -> f64 {
        match self {
            LodLevel::Full => 1.0,
            LodLevel::Half => 0.5,
            LodLevel::Quarter => 0.25,
            LodLevel::Minimal => 0.125,
        }
    }
}

// ── Chunk event ────────────────────────────────────────────────

/// Events emitted by the chunk manager.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChunkEvent {
    LoadRequested(ChunkCoord),
    Loaded(ChunkCoord),
    UnloadRequested(ChunkCoord),
    Unloaded(ChunkCoord),
}

// ── Chunk entry ────────────────────────────────────────────────

/// A single chunk with its state and optional payload.
#[derive(Debug, Clone)]
pub struct Chunk<T: Clone> {
    pub coord: ChunkCoord,
    pub state: ChunkState,
    pub lod: LodLevel,
    pub data: Option<T>,
}

impl<T: Clone> Chunk<T> {
    pub fn new(coord: ChunkCoord) -> Self {
        Self {
            coord,
            state: ChunkState::Unloaded,
            lod: LodLevel::Full,
            data: None,
        }
    }
}

// ── Unload ring buffer ─────────────────────────────────────────

/// Ring buffer caching recently unloaded chunk data for fast reload.
#[derive(Debug)]
struct UnloadCache<T: Clone> {
    entries: VecDeque<(ChunkCoord, T)>,
    capacity: usize,
}

impl<T: Clone> UnloadCache<T> {
    fn new(capacity: usize) -> Self {
        Self {
            entries: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    fn insert(&mut self, coord: ChunkCoord, data: T) {
        if self.entries.len() >= self.capacity {
            self.entries.pop_front();
        }
        self.entries.push_back((coord, data));
    }

    fn take(&mut self, coord: &ChunkCoord) -> Option<T> {
        if let Some(idx) = self.entries.iter().position(|(c, _)| c == coord) {
            self.entries.remove(idx).map(|(_, d)| d)
        } else {
            None
        }
    }

    fn contains(&self, coord: &ChunkCoord) -> bool {
        self.entries.iter().any(|(c, _)| c == coord)
    }

    fn len(&self) -> usize {
        self.entries.len()
    }
}

// ── Chunk manager ──────────────────────────────────────────────

/// Manages loading, unloading, and LOD of world chunks around a camera.
#[derive(Debug)]
pub struct ChunkManager<T: Clone> {
    chunks: HashMap<ChunkCoord, Chunk<T>>,
    camera: ChunkCoord,
    radius: i32,
    unload_cache: UnloadCache<T>,
    events: Vec<ChunkEvent>,
    load_queue: Vec<ChunkCoord>,
}

impl<T: Clone + fmt::Debug> ChunkManager<T> {
    /// Create a new chunk manager with given active radius and unload cache size.
    pub fn new(radius: i32, cache_capacity: usize) -> Self {
        Self {
            chunks: HashMap::new(),
            camera: ChunkCoord::new(0, 0),
            radius: radius.max(1),
            unload_cache: UnloadCache::new(cache_capacity),
            events: Vec::new(),
            load_queue: Vec::new(),
        }
    }

    /// Current camera chunk coordinate.
    pub fn camera(&self) -> ChunkCoord {
        self.camera
    }

    /// Active chunk radius.
    pub fn radius(&self) -> i32 {
        self.radius
    }

    /// Move the camera to a new chunk coordinate and update which chunks
    /// should be loaded or unloaded. Returns events generated.
    pub fn update_camera(&mut self, new_pos: ChunkCoord) -> Vec<ChunkEvent> {
        self.events.clear();
        self.camera = new_pos;

        // Determine which coords should be active.
        let desired = self.desired_coords();

        // Mark chunks outside radius for unloading.
        let current_coords: Vec<ChunkCoord> = self.chunks.keys().copied().collect();
        for coord in &current_coords {
            if !desired.contains(coord) {
                self.begin_unload(*coord);
            }
        }

        // Build a load queue sorted by distance (nearest first).
        let mut to_load: Vec<ChunkCoord> = desired
            .into_iter()
            .filter(|c| !self.chunks.contains_key(c))
            .collect();
        let cam = self.camera;
        to_load.sort_by_key(|c| c.distance_sq(&cam));
        self.load_queue = to_load.clone();

        for coord in &to_load {
            self.begin_load(*coord);
        }

        self.events.clone()
    }

    /// Set of chunk coords that should be loaded for current camera + radius.
    fn desired_coords(&self) -> HashSet<ChunkCoord> {
        let mut set = HashSet::new();
        let r = self.radius;
        let r_sq = (r as i64) * (r as i64);
        for dx in -r..=r {
            for dz in -r..=r {
                let coord = ChunkCoord::new(self.camera.x + dx, self.camera.z + dz);
                if coord.distance_sq(&self.camera) <= r_sq {
                    set.insert(coord);
                }
            }
        }
        set
    }

    fn begin_load(&mut self, coord: ChunkCoord) {
        let mut chunk = Chunk::new(coord);
        chunk.state = ChunkState::Loading;
        chunk.lod = LodLevel::from_distance_sq(coord.distance_sq(&self.camera), self.radius);
        self.chunks.insert(coord, chunk);
        self.events.push(ChunkEvent::LoadRequested(coord));
    }

    fn begin_unload(&mut self, coord: ChunkCoord) {
        if let Some(chunk) = self.chunks.get_mut(&coord) {
            if chunk.state == ChunkState::Loaded || chunk.state == ChunkState::Loading {
                chunk.state = ChunkState::Unloading;
                self.events.push(ChunkEvent::UnloadRequested(coord));
            }
        }
    }

    /// Complete loading of a chunk, providing its data payload.
    pub fn finish_load(&mut self, coord: ChunkCoord, data: T) -> Option<ChunkEvent> {
        // Check if data is in unload cache first, then use provided data
        let _ = self.unload_cache.take(&coord);
        if let Some(chunk) = self.chunks.get_mut(&coord) {
            if chunk.state == ChunkState::Loading {
                chunk.state = ChunkState::Loaded;
                chunk.data = Some(data);
                chunk.lod =
                    LodLevel::from_distance_sq(coord.distance_sq(&self.camera), self.radius);
                return Some(ChunkEvent::Loaded(coord));
            }
        }
        None
    }

    /// Try to fast-reload a chunk from the unload cache.
    pub fn try_reload_from_cache(&mut self, coord: ChunkCoord) -> bool {
        if let Some(data) = self.unload_cache.take(&coord) {
            let mut chunk = Chunk::new(coord);
            chunk.state = ChunkState::Loaded;
            chunk.data = Some(data);
            chunk.lod = LodLevel::from_distance_sq(coord.distance_sq(&self.camera), self.radius);
            self.chunks.insert(coord, chunk);
            true
        } else {
            false
        }
    }

    /// Complete unloading of a chunk, caching its data.
    pub fn finish_unload(&mut self, coord: ChunkCoord) -> Option<ChunkEvent> {
        if let Some(chunk) = self.chunks.remove(&coord) {
            if let Some(data) = chunk.data {
                self.unload_cache.insert(coord, data);
            }
            return Some(ChunkEvent::Unloaded(coord));
        }
        None
    }

    /// Get chunk state for a coordinate.
    pub fn chunk_state(&self, coord: &ChunkCoord) -> ChunkState {
        self.chunks
            .get(coord)
            .map(|c| c.state)
            .unwrap_or(ChunkState::Unloaded)
    }

    /// Get LOD for a chunk.
    pub fn chunk_lod(&self, coord: &ChunkCoord) -> Option<LodLevel> {
        self.chunks.get(coord).map(|c| c.lod)
    }

    /// Get chunk data reference.
    pub fn chunk_data(&self, coord: &ChunkCoord) -> Option<&T> {
        self.chunks.get(coord).and_then(|c| c.data.as_ref())
    }

    /// Get mutable chunk data reference.
    pub fn chunk_data_mut(&mut self, coord: &ChunkCoord) -> Option<&mut T> {
        self.chunks.get_mut(coord).and_then(|c| c.data.as_mut())
    }

    /// Loaded neighbor chunks (orthogonal) for a given coordinate.
    pub fn loaded_neighbors(&self, coord: &ChunkCoord) -> Vec<ChunkCoord> {
        coord
            .neighbors()
            .iter()
            .filter(|n| self.chunk_state(n) == ChunkState::Loaded)
            .copied()
            .collect()
    }

    /// 8-connected loaded neighbors.
    pub fn loaded_neighbors_8(&self, coord: &ChunkCoord) -> Vec<ChunkCoord> {
        coord
            .neighbors_8()
            .iter()
            .filter(|n| self.chunk_state(n) == ChunkState::Loaded)
            .copied()
            .collect()
    }

    /// Number of chunks currently tracked (any state).
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Number of chunks in Loaded state.
    pub fn loaded_count(&self) -> usize {
        self.chunks
            .values()
            .filter(|c| c.state == ChunkState::Loaded)
            .count()
    }

    /// Number of chunks pending load.
    pub fn loading_count(&self) -> usize {
        self.chunks
            .values()
            .filter(|c| c.state == ChunkState::Loading)
            .count()
    }

    /// Current pending load queue length.
    pub fn load_queue_len(&self) -> usize {
        self.load_queue.len()
    }

    /// Unload cache size.
    pub fn cache_size(&self) -> usize {
        self.unload_cache.len()
    }

    /// Whether a coord is in the unload cache.
    pub fn is_cached(&self, coord: &ChunkCoord) -> bool {
        self.unload_cache.contains(coord)
    }

    /// All coords currently tracked.
    pub fn all_coords(&self) -> Vec<ChunkCoord> {
        self.chunks.keys().copied().collect()
    }

    /// All loaded coords.
    pub fn loaded_coords(&self) -> Vec<ChunkCoord> {
        self.chunks
            .values()
            .filter(|c| c.state == ChunkState::Loaded)
            .map(|c| c.coord)
            .collect()
    }

    /// Recalculate LODs for all loaded chunks relative to current camera.
    pub fn refresh_lods(&mut self) {
        let cam = self.camera;
        let r = self.radius;
        for chunk in self.chunks.values_mut() {
            chunk.lod = LodLevel::from_distance_sq(chunk.coord.distance_sq(&cam), r);
        }
    }

    /// Change the active radius. Does not trigger load/unload — call update_camera after.
    pub fn set_radius(&mut self, radius: i32) {
        self.radius = radius.max(1);
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coord_distance_sq() {
        let a = ChunkCoord::new(0, 0);
        let b = ChunkCoord::new(3, 4);
        assert_eq!(a.distance_sq(&b), 25);
    }

    #[test]
    fn coord_manhattan() {
        let a = ChunkCoord::new(1, 2);
        let b = ChunkCoord::new(4, 6);
        assert_eq!(a.manhattan(&b), 7);
    }

    #[test]
    fn coord_neighbors_4() {
        let c = ChunkCoord::new(5, 5);
        let ns = c.neighbors();
        let set: HashSet<ChunkCoord> = ns.into_iter().collect();
        assert!(set.contains(&ChunkCoord::new(5, 4)));
        assert!(set.contains(&ChunkCoord::new(5, 6)));
        assert!(set.contains(&ChunkCoord::new(4, 5)));
        assert!(set.contains(&ChunkCoord::new(6, 5)));
        assert_eq!(set.len(), 4);
    }

    #[test]
    fn coord_neighbors_8() {
        let c = ChunkCoord::new(0, 0);
        let ns = c.neighbors_8();
        assert_eq!(ns.len(), 8);
        let set: HashSet<ChunkCoord> = ns.into_iter().collect();
        assert!(set.contains(&ChunkCoord::new(-1, -1)));
        assert!(set.contains(&ChunkCoord::new(1, 1)));
    }

    #[test]
    fn coord_display() {
        assert_eq!(ChunkCoord::new(3, -7).to_string(), "(3, -7)");
    }

    #[test]
    fn lod_from_distance() {
        // radius 10 → r_sq = 100
        assert_eq!(LodLevel::from_distance_sq(0, 10), LodLevel::Full);
        assert_eq!(LodLevel::from_distance_sq(24, 10), LodLevel::Full);
        assert_eq!(LodLevel::from_distance_sq(26, 10), LodLevel::Half);
        assert_eq!(LodLevel::from_distance_sq(50, 10), LodLevel::Half);
        assert_eq!(LodLevel::from_distance_sq(51, 10), LodLevel::Quarter);
        assert_eq!(LodLevel::from_distance_sq(76, 10), LodLevel::Minimal);
    }

    #[test]
    fn lod_detail_factors() {
        assert!((LodLevel::Full.detail_factor() - 1.0).abs() < 1e-9);
        assert!((LodLevel::Half.detail_factor() - 0.5).abs() < 1e-9);
        assert!((LodLevel::Quarter.detail_factor() - 0.25).abs() < 1e-9);
        assert!((LodLevel::Minimal.detail_factor() - 0.125).abs() < 1e-9);
    }

    #[test]
    fn new_manager_empty() {
        let mgr: ChunkManager<String> = ChunkManager::new(3, 8);
        assert_eq!(mgr.chunk_count(), 0);
        assert_eq!(mgr.loaded_count(), 0);
        assert_eq!(mgr.radius(), 3);
    }

    #[test]
    fn update_camera_creates_load_requests() {
        let mut mgr: ChunkManager<u32> = ChunkManager::new(1, 4);
        let evts = mgr.update_camera(ChunkCoord::new(0, 0));
        // radius 1 circle: (0,0), (1,0), (-1,0), (0,1), (0,-1) = 5 chunks
        let load_reqs = evts
            .iter()
            .filter(|e| matches!(e, ChunkEvent::LoadRequested(_)))
            .count();
        assert_eq!(load_reqs, 5);
        assert_eq!(mgr.loading_count(), 5);
    }

    #[test]
    fn finish_load_transitions_to_loaded() {
        let mut mgr: ChunkManager<u32> = ChunkManager::new(1, 4);
        mgr.update_camera(ChunkCoord::new(0, 0));
        let c = ChunkCoord::new(0, 0);
        let evt = mgr.finish_load(c, 42);
        assert_eq!(evt, Some(ChunkEvent::Loaded(c)));
        assert_eq!(mgr.chunk_state(&c), ChunkState::Loaded);
        assert_eq!(mgr.chunk_data(&c), Some(&42));
    }

    #[test]
    fn finish_unload_removes_chunk_and_caches() {
        let mut mgr: ChunkManager<String> = ChunkManager::new(1, 4);
        mgr.update_camera(ChunkCoord::new(0, 0));
        let c = ChunkCoord::new(0, 0);
        mgr.finish_load(c, "hello".to_string());
        // Simulate moving camera far away so (0,0) is out of range
        mgr.update_camera(ChunkCoord::new(100, 100));
        let evt = mgr.finish_unload(c);
        assert_eq!(evt, Some(ChunkEvent::Unloaded(c)));
        assert_eq!(mgr.chunk_state(&c), ChunkState::Unloaded);
        assert!(mgr.is_cached(&c));
    }

    #[test]
    fn reload_from_cache() {
        let mut mgr: ChunkManager<u32> = ChunkManager::new(1, 4);
        mgr.update_camera(ChunkCoord::new(0, 0));
        let c = ChunkCoord::new(0, 0);
        mgr.finish_load(c, 99);
        mgr.update_camera(ChunkCoord::new(100, 100));
        mgr.finish_unload(c);
        assert!(mgr.is_cached(&c));
        assert!(mgr.try_reload_from_cache(c));
        assert_eq!(mgr.chunk_data(&c), Some(&99));
        assert!(!mgr.is_cached(&c));
    }

    #[test]
    fn cache_evicts_oldest() {
        let mut mgr: ChunkManager<u32> = ChunkManager::new(1, 2);
        mgr.update_camera(ChunkCoord::new(0, 0));
        let coords: Vec<ChunkCoord> = vec![
            ChunkCoord::new(0, 0),
            ChunkCoord::new(1, 0),
            ChunkCoord::new(0, 1),
        ];
        for c in &coords {
            mgr.finish_load(*c, 1);
        }
        mgr.update_camera(ChunkCoord::new(100, 100));
        for c in &coords {
            mgr.finish_unload(*c);
        }
        // Cache capacity 2 → oldest (0,0) was evicted
        assert_eq!(mgr.cache_size(), 2);
        assert!(!mgr.is_cached(&coords[0]));
        assert!(mgr.is_cached(&coords[1]));
        assert!(mgr.is_cached(&coords[2]));
    }

    #[test]
    fn loaded_neighbors_query() {
        let mut mgr: ChunkManager<u32> = ChunkManager::new(2, 4);
        mgr.update_camera(ChunkCoord::new(0, 0));
        let center = ChunkCoord::new(0, 0);
        mgr.finish_load(center, 1);
        mgr.finish_load(ChunkCoord::new(1, 0), 2);
        mgr.finish_load(ChunkCoord::new(-1, 0), 3);
        let ns = mgr.loaded_neighbors(&center);
        assert_eq!(ns.len(), 2);
    }

    #[test]
    fn chunk_data_mut_access() {
        let mut mgr: ChunkManager<Vec<u8>> = ChunkManager::new(1, 4);
        mgr.update_camera(ChunkCoord::new(0, 0));
        let c = ChunkCoord::new(0, 0);
        mgr.finish_load(c, vec![1, 2, 3]);
        if let Some(data) = mgr.chunk_data_mut(&c) {
            data.push(4);
        }
        assert_eq!(mgr.chunk_data(&c), Some(&vec![1, 2, 3, 4]));
    }

    #[test]
    fn camera_movement_unloads_far_chunks() {
        let mut mgr: ChunkManager<u32> = ChunkManager::new(1, 4);
        mgr.update_camera(ChunkCoord::new(0, 0));
        let c = ChunkCoord::new(0, 0);
        mgr.finish_load(c, 1);
        assert_eq!(mgr.loaded_count(), 1);
        let evts = mgr.update_camera(ChunkCoord::new(50, 50));
        let unload_reqs: Vec<_> = evts
            .iter()
            .filter(|e| matches!(e, ChunkEvent::UnloadRequested(_)))
            .collect();
        assert!(!unload_reqs.is_empty());
    }

    #[test]
    fn refresh_lods() {
        let mut mgr: ChunkManager<u32> = ChunkManager::new(5, 4);
        mgr.update_camera(ChunkCoord::new(0, 0));
        let far = ChunkCoord::new(4, 0);
        mgr.finish_load(far, 1);
        let near = ChunkCoord::new(0, 0);
        mgr.finish_load(near, 2);
        mgr.refresh_lods();
        let near_lod = mgr.chunk_lod(&near).unwrap();
        let far_lod = mgr.chunk_lod(&far).unwrap();
        assert!(near_lod.detail_factor() >= far_lod.detail_factor());
    }

    #[test]
    fn set_radius() {
        let mut mgr: ChunkManager<u32> = ChunkManager::new(2, 4);
        mgr.set_radius(5);
        assert_eq!(mgr.radius(), 5);
        mgr.set_radius(0);
        assert_eq!(mgr.radius(), 1); // clamped to 1
    }

    #[test]
    fn all_coords_and_loaded_coords() {
        let mut mgr: ChunkManager<u32> = ChunkManager::new(1, 4);
        mgr.update_camera(ChunkCoord::new(0, 0));
        let c = ChunkCoord::new(0, 0);
        mgr.finish_load(c, 1);
        assert!(mgr.all_coords().len() >= 1);
        assert_eq!(mgr.loaded_coords().len(), 1);
    }

    #[test]
    fn load_queue_nearest_first() {
        let mut mgr: ChunkManager<u32> = ChunkManager::new(2, 4);
        mgr.update_camera(ChunkCoord::new(0, 0));
        // load_queue should have nearest chunks first
        assert!(mgr.load_queue_len() > 0);
    }

    #[test]
    fn chunk_state_display() {
        assert_eq!(ChunkState::Loaded.to_string(), "loaded");
        assert_eq!(ChunkState::Unloaded.to_string(), "unloaded");
        assert_eq!(ChunkState::Loading.to_string(), "loading");
        assert_eq!(ChunkState::Unloading.to_string(), "unloading");
    }

    #[test]
    fn cannot_finish_load_on_already_loaded() {
        let mut mgr: ChunkManager<u32> = ChunkManager::new(1, 4);
        mgr.update_camera(ChunkCoord::new(0, 0));
        let c = ChunkCoord::new(0, 0);
        mgr.finish_load(c, 1);
        // Second finish_load should return None (already Loaded, not Loading)
        let evt = mgr.finish_load(c, 2);
        assert_eq!(evt, None);
        // Data unchanged
        assert_eq!(mgr.chunk_data(&c), Some(&1));
    }

    #[test]
    fn negative_coordinates_work() {
        let mut mgr: ChunkManager<u32> = ChunkManager::new(1, 4);
        let evts = mgr.update_camera(ChunkCoord::new(-100, -200));
        assert!(!evts.is_empty());
        let c = ChunkCoord::new(-100, -200);
        assert_eq!(mgr.chunk_state(&c), ChunkState::Loading);
    }
}
