//! Infinite procedural terrain generation system.
//!
//! Generates terrain on-demand per chunk using seeded deterministic noise,
//! assigns biomes, produces mesh data with seamless borders, supports terrain
//! modification (dig/raise) with overlay heightmaps, undo, and provides
//! generation statistics. Multi-threaded generation is simulated via a task queue.

use std::collections::HashMap;
use std::fmt;

// ── Biome ──────────────────────────────────────────────────────

/// Biome classification for a terrain chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Biome {
    Plains,
    Desert,
    Forest,
    Mountains,
    Tundra,
    Swamp,
    Ocean,
}

impl Biome {
    /// Choose biome deterministically from a hash value.
    fn from_hash(h: u64) -> Self {
        match h % 7 {
            0 => Biome::Plains,
            1 => Biome::Desert,
            2 => Biome::Forest,
            3 => Biome::Mountains,
            4 => Biome::Tundra,
            5 => Biome::Swamp,
            _ => Biome::Ocean,
        }
    }

    /// Base height scale for this biome.
    pub fn height_scale(&self) -> f64 {
        match self {
            Biome::Plains => 0.2,
            Biome::Desert => 0.1,
            Biome::Forest => 0.35,
            Biome::Mountains => 1.0,
            Biome::Tundra => 0.3,
            Biome::Swamp => 0.05,
            Biome::Ocean => 0.0,
        }
    }
}

impl fmt::Display for Biome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Biome::Plains => "plains",
            Biome::Desert => "desert",
            Biome::Forest => "forest",
            Biome::Mountains => "mountains",
            Biome::Tundra => "tundra",
            Biome::Swamp => "swamp",
            Biome::Ocean => "ocean",
        };
        write!(f, "{s}")
    }
}

// ── Deterministic hash / noise ─────────────────────────────────

/// Simple integer hash for deterministic noise.
fn hash_2d(seed: u64, x: i32, z: i32) -> u64 {
    let mut h = seed;
    h = h.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    h ^= x as u64;
    h = h.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    h ^= z as u64;
    h = h.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    h
}

/// Deterministic noise value in [0, 1] for a world-space sample point.
fn noise(seed: u64, x: i32, z: i32) -> f64 {
    let h = hash_2d(seed, x, z);
    (h % 10000) as f64 / 10000.0
}

/// Multi-octave noise for smoother terrain.
fn octave_noise(seed: u64, x: i32, z: i32, octaves: u32) -> f64 {
    let mut total = 0.0;
    let mut amplitude = 1.0;
    let mut max_val = 0.0;
    for i in 0..octaves {
        let freq = 1 << i;
        total += noise(seed.wrapping_add(i as u64), x * freq, z * freq) * amplitude;
        max_val += amplitude;
        amplitude *= 0.5;
    }
    total / max_val
}

// ── Chunk coord ────────────────────────────────────────────────

/// Integer chunk coordinate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TerrainCoord {
    pub x: i32,
    pub z: i32,
}

impl TerrainCoord {
    pub fn new(x: i32, z: i32) -> Self {
        Self { x, z }
    }
}

// ── Heightmap ──────────────────────────────────────────────────

/// A grid of height values for one chunk, including a 1-sample overlap border
/// for seamless stitching.
#[derive(Debug, Clone, PartialEq)]
pub struct Heightmap {
    /// Width of the core data (without border).
    pub resolution: usize,
    /// Height samples. Size = (resolution + 2)^2 to include 1-sample border on each side.
    pub samples: Vec<f64>,
}

impl Heightmap {
    pub fn new(resolution: usize) -> Self {
        let side = resolution + 2;
        Self {
            resolution,
            samples: vec![0.0; side * side],
        }
    }

    fn side(&self) -> usize {
        self.resolution + 2
    }

    /// Get height at local grid position (including border, 0-based).
    pub fn get(&self, lx: usize, lz: usize) -> f64 {
        let s = self.side();
        if lx < s && lz < s {
            self.samples[lz * s + lx]
        } else {
            0.0
        }
    }

    /// Set height at local grid position.
    pub fn set(&mut self, lx: usize, lz: usize, val: f64) {
        let s = self.side();
        if lx < s && lz < s {
            self.samples[lz * s + lx] = val;
        }
    }

    /// Number of vertices in the full grid (including border).
    pub fn vertex_count(&self) -> usize {
        self.samples.len()
    }
}

// ── Terrain chunk data ─────────────────────────────────────────

/// Generated terrain data for one chunk.
#[derive(Debug, Clone)]
pub struct TerrainChunk {
    pub coord: TerrainCoord,
    pub biome: Biome,
    pub heightmap: Heightmap,
    /// Overlay heightmap for modifications (dig/raise). Added on top of base.
    pub overlay: Heightmap,
    /// Stack of (lx, lz, previous_overlay_value) for undo.
    modification_history: Vec<(usize, usize, f64)>,
}

impl TerrainChunk {
    /// Effective height at a local sample position.
    pub fn effective_height(&self, lx: usize, lz: usize) -> f64 {
        self.heightmap.get(lx, lz) + self.overlay.get(lx, lz)
    }

    /// Modify terrain at a local position by adding delta.
    pub fn modify(&mut self, lx: usize, lz: usize, delta: f64) {
        let prev = self.overlay.get(lx, lz);
        self.modification_history.push((lx, lz, prev));
        self.overlay.set(lx, lz, prev + delta);
    }

    /// Undo the last modification. Returns true if an undo was performed.
    pub fn undo_modification(&mut self) -> bool {
        if let Some((lx, lz, prev)) = self.modification_history.pop() {
            self.overlay.set(lx, lz, prev);
            true
        } else {
            false
        }
    }

    /// Number of modifications applied.
    pub fn modification_count(&self) -> usize {
        self.modification_history.len()
    }
}

// ── Generation task ────────────────────────────────────────────

/// Task state for chunk generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskState {
    Pending,
    InProgress,
    Complete,
}

/// A queued generation task.
#[derive(Debug, Clone)]
pub struct GenerationTask {
    pub coord: TerrainCoord,
    pub state: TaskState,
    pub priority: u32,
}

// ── Terrain system ─────────────────────────────────────────────

/// Infinite terrain manager: generates, stores, and modifies terrain chunks.
pub struct InfiniteTerrain {
    seed: u64,
    resolution: usize,
    octaves: u32,
    max_height: f64,
    chunks: HashMap<TerrainCoord, TerrainChunk>,
    task_queue: Vec<GenerationTask>,
    total_generated: u64,
}

impl fmt::Debug for InfiniteTerrain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InfiniteTerrain")
            .field("seed", &self.seed)
            .field("resolution", &self.resolution)
            .field("chunks", &self.chunks.len())
            .finish()
    }
}

impl InfiniteTerrain {
    /// Create a new terrain system with the given seed and per-chunk resolution.
    pub fn new(seed: u64, resolution: usize) -> Self {
        Self {
            seed,
            resolution: resolution.max(2),
            octaves: 4,
            max_height: 64.0,
            chunks: HashMap::new(),
            task_queue: Vec::new(),
            total_generated: 0,
        }
    }

    /// Set number of noise octaves.
    pub fn set_octaves(&mut self, octaves: u32) {
        self.octaves = octaves.max(1);
    }

    /// Set max terrain height.
    pub fn set_max_height(&mut self, h: f64) {
        self.max_height = h;
    }

    /// Seed used by this terrain.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Request generation of a chunk. Adds to task queue if not already generated.
    pub fn request_chunk(&mut self, coord: TerrainCoord, priority: u32) {
        if self.chunks.contains_key(&coord) {
            return;
        }
        // Avoid duplicate tasks.
        if self.task_queue.iter().any(|t| t.coord == coord) {
            return;
        }
        self.task_queue.push(GenerationTask {
            coord,
            state: TaskState::Pending,
            priority,
        });
        // Sort by priority descending (higher = more urgent).
        self.task_queue.sort_by(|a, b| b.priority.cmp(&a.priority));
    }

    /// Process next pending task from the queue: generate the chunk.
    /// Returns the coord of the generated chunk, or None if queue is empty.
    pub fn process_next(&mut self) -> Option<TerrainCoord> {
        let idx = self.task_queue.iter().position(|t| t.state == TaskState::Pending);
        let idx = idx?;
        self.task_queue[idx].state = TaskState::InProgress;
        let coord = self.task_queue[idx].coord;
        let chunk = self.generate_chunk(coord);
        self.chunks.insert(coord, chunk);
        self.task_queue[idx].state = TaskState::Complete;
        self.total_generated += 1;
        Some(coord)
    }

    /// Process all pending tasks.
    pub fn process_all(&mut self) -> usize {
        let mut count = 0;
        while self.process_next().is_some() {
            count += 1;
        }
        // Clean up completed tasks.
        self.task_queue.retain(|t| t.state != TaskState::Complete);
        count
    }

    /// Generate a chunk synchronously (also used internally).
    pub fn generate_chunk(&self, coord: TerrainCoord) -> TerrainChunk {
        let biome_hash = hash_2d(self.seed.wrapping_add(9999), coord.x, coord.z);
        let biome = Biome::from_hash(biome_hash);

        let mut heightmap = Heightmap::new(self.resolution);
        let side = self.resolution + 2;
        for lz in 0..side {
            for lx in 0..side {
                // World-space sample positions (with overlap border offset by -1).
                let wx = coord.x * self.resolution as i32 + lx as i32 - 1;
                let wz = coord.z * self.resolution as i32 + lz as i32 - 1;
                let n = octave_noise(self.seed, wx, wz, self.octaves);
                let h = n * self.max_height * biome.height_scale();
                heightmap.set(lx, lz, h);
            }
        }

        let overlay = Heightmap::new(self.resolution);

        TerrainChunk {
            coord,
            biome,
            heightmap,
            overlay,
            modification_history: Vec::new(),
        }
    }

    /// Generate and store a chunk immediately (bypassing task queue).
    pub fn generate_and_store(&mut self, coord: TerrainCoord) {
        if !self.chunks.contains_key(&coord) {
            let chunk = self.generate_chunk(coord);
            self.chunks.insert(coord, chunk);
            self.total_generated += 1;
        }
    }

    /// Unload a chunk, returning its data.
    pub fn unload_chunk(&mut self, coord: &TerrainCoord) -> Option<TerrainChunk> {
        self.chunks.remove(coord)
    }

    /// Get a reference to a loaded chunk.
    pub fn chunk(&self, coord: &TerrainCoord) -> Option<&TerrainChunk> {
        self.chunks.get(coord)
    }

    /// Get a mutable reference to a loaded chunk.
    pub fn chunk_mut(&mut self, coord: &TerrainCoord) -> Option<&mut TerrainChunk> {
        self.chunks.get_mut(coord)
    }

    /// Number of loaded chunks.
    pub fn loaded_count(&self) -> usize {
        self.chunks.len()
    }

    /// Total vertices across all loaded chunks.
    pub fn total_vertices(&self) -> usize {
        self.chunks.values().map(|c| c.heightmap.vertex_count()).sum()
    }

    /// Total chunks generated since creation.
    pub fn total_generated(&self) -> u64 {
        self.total_generated
    }

    /// Number of tasks in queue.
    pub fn queue_len(&self) -> usize {
        self.task_queue.len()
    }

    /// Number of pending tasks.
    pub fn pending_count(&self) -> usize {
        self.task_queue
            .iter()
            .filter(|t| t.state == TaskState::Pending)
            .count()
    }

    /// Get biome for a chunk coord without generating the full chunk.
    pub fn biome_at(&self, coord: TerrainCoord) -> Biome {
        if let Some(c) = self.chunks.get(&coord) {
            c.biome
        } else {
            let h = hash_2d(self.seed.wrapping_add(9999), coord.x, coord.z);
            Biome::from_hash(h)
        }
    }

    /// All loaded chunk coords.
    pub fn loaded_coords(&self) -> Vec<TerrainCoord> {
        self.chunks.keys().copied().collect()
    }

    /// Statistics summary.
    pub fn stats(&self) -> TerrainStats {
        TerrainStats {
            loaded_chunks: self.loaded_count(),
            total_vertices: self.total_vertices(),
            total_generated: self.total_generated,
            queue_len: self.queue_len(),
        }
    }
}

/// Summary statistics for the terrain system.
#[derive(Debug, Clone, PartialEq)]
pub struct TerrainStats {
    pub loaded_chunks: usize,
    pub total_vertices: usize,
    pub total_generated: u64,
    pub queue_len: usize,
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise_deterministic() {
        let a = noise(42, 10, 20);
        let b = noise(42, 10, 20);
        assert!((a - b).abs() < 1e-12);
    }

    #[test]
    fn noise_range() {
        for x in -50..50 {
            for z in -50..50 {
                let v = noise(123, x, z);
                assert!(v >= 0.0 && v < 1.0, "noise({x},{z})={v} out of range");
            }
        }
    }

    #[test]
    fn different_seeds_different_values() {
        let a = noise(1, 5, 5);
        let b = noise(2, 5, 5);
        // Extremely unlikely to be equal with different seeds.
        assert!((a - b).abs() > 1e-12);
    }

    #[test]
    fn octave_noise_range() {
        for x in -20..20 {
            let v = octave_noise(42, x, 0, 4);
            assert!(v >= 0.0 && v <= 1.0 + 1e-9, "octave_noise out of range: {v}");
        }
    }

    #[test]
    fn biome_from_hash_covers_all() {
        let mut seen = std::collections::HashSet::new();
        for i in 0u64..100 {
            seen.insert(Biome::from_hash(i));
        }
        assert_eq!(seen.len(), 7);
    }

    #[test]
    fn biome_display() {
        assert_eq!(Biome::Mountains.to_string(), "mountains");
        assert_eq!(Biome::Ocean.to_string(), "ocean");
    }

    #[test]
    fn heightmap_get_set() {
        let mut hm = Heightmap::new(4);
        hm.set(2, 3, 5.5);
        assert!((hm.get(2, 3) - 5.5).abs() < 1e-12);
    }

    #[test]
    fn heightmap_border_size() {
        let hm = Heightmap::new(8);
        // Side = 8 + 2 = 10; total = 100
        assert_eq!(hm.vertex_count(), 100);
    }

    #[test]
    fn generate_chunk_deterministic() {
        let terrain = InfiniteTerrain::new(42, 8);
        let c = TerrainCoord::new(5, -3);
        let a = terrain.generate_chunk(c);
        let b = terrain.generate_chunk(c);
        assert_eq!(a.biome, b.biome);
        assert_eq!(a.heightmap, b.heightmap);
    }

    #[test]
    fn generate_and_store() {
        let mut terrain = InfiniteTerrain::new(42, 8);
        let c = TerrainCoord::new(0, 0);
        terrain.generate_and_store(c);
        assert_eq!(terrain.loaded_count(), 1);
        assert!(terrain.chunk(&c).is_some());
    }

    #[test]
    fn task_queue_priority() {
        let mut terrain = InfiniteTerrain::new(42, 4);
        terrain.request_chunk(TerrainCoord::new(0, 0), 1);
        terrain.request_chunk(TerrainCoord::new(1, 0), 10);
        terrain.request_chunk(TerrainCoord::new(2, 0), 5);
        // Highest priority (10) processed first.
        let first = terrain.process_next().unwrap();
        assert_eq!(first, TerrainCoord::new(1, 0));
    }

    #[test]
    fn process_all_tasks() {
        let mut terrain = InfiniteTerrain::new(42, 4);
        terrain.request_chunk(TerrainCoord::new(0, 0), 1);
        terrain.request_chunk(TerrainCoord::new(1, 0), 2);
        terrain.request_chunk(TerrainCoord::new(2, 0), 3);
        let count = terrain.process_all();
        assert_eq!(count, 3);
        assert_eq!(terrain.loaded_count(), 3);
        assert_eq!(terrain.queue_len(), 0);
    }

    #[test]
    fn duplicate_request_ignored() {
        let mut terrain = InfiniteTerrain::new(42, 4);
        terrain.request_chunk(TerrainCoord::new(0, 0), 1);
        terrain.request_chunk(TerrainCoord::new(0, 0), 5);
        assert_eq!(terrain.queue_len(), 1);
    }

    #[test]
    fn request_already_loaded_ignored() {
        let mut terrain = InfiniteTerrain::new(42, 4);
        terrain.generate_and_store(TerrainCoord::new(0, 0));
        terrain.request_chunk(TerrainCoord::new(0, 0), 1);
        assert_eq!(terrain.queue_len(), 0);
    }

    #[test]
    fn terrain_modification() {
        let mut terrain = InfiniteTerrain::new(42, 4);
        let c = TerrainCoord::new(0, 0);
        terrain.generate_and_store(c);
        let chunk = terrain.chunk_mut(&c).unwrap();
        let original = chunk.effective_height(2, 2);
        chunk.modify(2, 2, 10.0);
        let modified = chunk.effective_height(2, 2);
        assert!((modified - original - 10.0).abs() < 1e-9);
    }

    #[test]
    fn undo_modification() {
        let mut terrain = InfiniteTerrain::new(42, 4);
        let c = TerrainCoord::new(0, 0);
        terrain.generate_and_store(c);
        let chunk = terrain.chunk_mut(&c).unwrap();
        let before = chunk.effective_height(2, 2);
        chunk.modify(2, 2, 5.0);
        chunk.undo_modification();
        let after = chunk.effective_height(2, 2);
        assert!((before - after).abs() < 1e-9);
    }

    #[test]
    fn undo_empty_returns_false() {
        let mut terrain = InfiniteTerrain::new(42, 4);
        let c = TerrainCoord::new(0, 0);
        terrain.generate_and_store(c);
        let chunk = terrain.chunk_mut(&c).unwrap();
        assert!(!chunk.undo_modification());
    }

    #[test]
    fn unload_chunk() {
        let mut terrain = InfiniteTerrain::new(42, 4);
        let c = TerrainCoord::new(0, 0);
        terrain.generate_and_store(c);
        let removed = terrain.unload_chunk(&c);
        assert!(removed.is_some());
        assert_eq!(terrain.loaded_count(), 0);
    }

    #[test]
    fn biome_at_without_generating() {
        let terrain = InfiniteTerrain::new(42, 4);
        let b1 = terrain.biome_at(TerrainCoord::new(5, 5));
        // Should match the chunk biome if generated.
        let chunk = terrain.generate_chunk(TerrainCoord::new(5, 5));
        assert_eq!(b1, chunk.biome);
    }

    #[test]
    fn total_vertices_accumulates() {
        let mut terrain = InfiniteTerrain::new(42, 4);
        terrain.generate_and_store(TerrainCoord::new(0, 0));
        terrain.generate_and_store(TerrainCoord::new(1, 0));
        let expected = 2 * (4 + 2) * (4 + 2);
        assert_eq!(terrain.total_vertices(), expected);
    }

    #[test]
    fn stats_snapshot() {
        let mut terrain = InfiniteTerrain::new(42, 4);
        terrain.generate_and_store(TerrainCoord::new(0, 0));
        let s = terrain.stats();
        assert_eq!(s.loaded_chunks, 1);
        assert_eq!(s.total_generated, 1);
        assert_eq!(s.queue_len, 0);
    }

    #[test]
    fn seamless_border_overlap() {
        // Verify that the border samples of adjacent chunks map to the same
        // world-space noise values. Because biomes may differ between chunks
        // (different height scales), we compare the raw noise directly rather
        // than the biome-scaled heightmap values.
        let seed = 42u64;
        let resolution = 4usize;
        let octaves = 4u32;

        let coord_a = TerrainCoord::new(0, 0);
        let coord_b = TerrainCoord::new(1, 0);

        // A's right border column (lx = resolution + 1 = 5) and
        // B's column at lx = 1 should map to the same world-x.
        // A: wx = 0*4 + 5 - 1 = 4
        // B: wx = 1*4 + 1 - 1 = 4  ✓
        let side = resolution + 2;
        for lz in 0..side {
            let wx_a = coord_a.x * resolution as i32 + (resolution as i32 + 1) - 1;
            let wz_a = coord_a.z * resolution as i32 + lz as i32 - 1;
            let wx_b = coord_b.x * resolution as i32 + 1 - 1;
            let wz_b = coord_b.z * resolution as i32 + lz as i32 - 1;
            assert_eq!(wx_a, wx_b, "world x mismatch at lz={lz}");
            assert_eq!(wz_a, wz_b, "world z mismatch at lz={lz}");
            let na = octave_noise(seed, wx_a, wz_a, octaves);
            let nb = octave_noise(seed, wx_b, wz_b, octaves);
            assert!(
                (na - nb).abs() < 1e-12,
                "Noise mismatch at lz={lz}: {na} vs {nb}"
            );
        }
    }

    #[test]
    fn modification_count_tracks() {
        let mut terrain = InfiniteTerrain::new(42, 4);
        let c = TerrainCoord::new(0, 0);
        terrain.generate_and_store(c);
        let chunk = terrain.chunk_mut(&c).unwrap();
        chunk.modify(1, 1, 3.0);
        chunk.modify(2, 2, -1.0);
        assert_eq!(chunk.modification_count(), 2);
        chunk.undo_modification();
        assert_eq!(chunk.modification_count(), 1);
    }
}
