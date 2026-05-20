//! Garbage collector — mark-sweep with tri-color marking, generational GC
//! (young/old), write barriers, root set management, allocation, GC
//! statistics (pauses, reclaimed bytes), heap visualization.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// ── Object Header ──────────────────────────────────────────────────────────

/// Tri-color marking state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Color {
    /// Not yet discovered.
    White,
    /// Discovered but not fully scanned.
    Gray,
    /// Fully scanned — reachable.
    Black,
}

/// Generational space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Generation {
    Young,
    Old,
}

/// The header of a heap object.
#[derive(Debug, Clone)]
pub struct ObjectHeader {
    pub id: u64,
    pub size_bytes: usize,
    pub color: Color,
    pub generation: Generation,
    /// How many collections this object has survived.
    pub survive_count: u32,
    /// IDs of objects this object references.
    pub references: Vec<u64>,
    /// Object type tag for debugging.
    pub type_tag: String,
}

impl ObjectHeader {
    fn new(id: u64, size_bytes: usize, type_tag: impl Into<String>) -> Self {
        Self {
            id,
            size_bytes,
            color: Color::White,
            generation: Generation::Young,
            survive_count: 0,
            references: Vec::new(),
            type_tag: type_tag.into(),
        }
    }
}

// ── GC Configuration ───────────────────────────────────────────────────────

/// Configuration for the garbage collector.
#[derive(Debug, Clone)]
pub struct GcConfig {
    /// Number of surviving collections before promotion to old gen.
    pub promotion_threshold: u32,
    /// Maximum heap size in bytes.
    pub max_heap_bytes: usize,
    /// Young-gen allocation budget before triggering a minor GC.
    pub young_budget_bytes: usize,
    /// Full heap utilization threshold (0.0–1.0) to trigger major GC.
    pub major_gc_threshold: f64,
}

impl Default for GcConfig {
    fn default() -> Self {
        Self {
            promotion_threshold: 3,
            max_heap_bytes: 1024 * 1024, // 1 MiB
            young_budget_bytes: 64 * 1024,
            major_gc_threshold: 0.75,
        }
    }
}

// ── GC Statistics ──────────────────────────────────────────────────────────

/// Statistics about garbage collection activity.
#[derive(Debug, Clone, Default)]
pub struct GcStats {
    pub minor_collections: u64,
    pub major_collections: u64,
    pub total_allocated_bytes: u64,
    pub total_reclaimed_bytes: u64,
    pub total_objects_allocated: u64,
    pub total_objects_reclaimed: u64,
    pub current_heap_bytes: usize,
    pub current_object_count: usize,
    pub young_object_count: usize,
    pub old_object_count: usize,
    pub promotions: u64,
    pub write_barriers: u64,
    pub peak_heap_bytes: usize,
}

impl GcStats {
    /// Heap utilization as a fraction (0.0–1.0).
    pub fn heap_utilization(&self, max_bytes: usize) -> f64 {
        if max_bytes == 0 {
            return 0.0;
        }
        self.current_heap_bytes as f64 / max_bytes as f64
    }
}

// ── GC Error ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GcError {
    OutOfMemory { requested: usize, available: usize },
    ObjectNotFound(u64),
    InvalidReference { from: u64, to: u64 },
    HeapCorruption(String),
}

impl fmt::Display for GcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfMemory {
                requested,
                available,
            } => write!(f, "OOM: requested {requested}, available {available}"),
            Self::ObjectNotFound(id) => write!(f, "object not found: {id}"),
            Self::InvalidReference { from, to } => {
                write!(f, "invalid reference: {from} -> {to}")
            }
            Self::HeapCorruption(msg) => write!(f, "heap corruption: {msg}"),
        }
    }
}

// ── Write Barrier ──────────────────────────────────────────────────────────

/// Write barrier record: an old-gen object has a reference to a young-gen object.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RememberedRef {
    old_id: u64,
    young_id: u64,
}

// ── Garbage Collector ──────────────────────────────────────────────────────

/// A generational mark-sweep garbage collector.
pub struct GarbageCollector {
    /// All objects, keyed by ID.
    objects: HashMap<u64, ObjectHeader>,
    /// Root set — IDs directly reachable from the mutator.
    roots: HashSet<u64>,
    /// Remembered set for write barriers (old → young references).
    remembered_set: HashSet<RememberedRef>,
    /// Allocation counter for young-gen budget.
    young_allocated_bytes: usize,
    /// Next object ID.
    next_id: u64,
    /// Configuration.
    config: GcConfig,
    /// Statistics.
    stats: GcStats,
}

impl GarbageCollector {
    pub fn new(config: GcConfig) -> Self {
        Self {
            objects: HashMap::new(),
            roots: HashSet::new(),
            remembered_set: HashSet::new(),
            young_allocated_bytes: 0,
            next_id: 1,
            config,
            stats: GcStats::default(),
        }
    }

    pub fn config(&self) -> &GcConfig {
        &self.config
    }

    pub fn stats(&self) -> &GcStats {
        &self.stats
    }

    pub fn root_count(&self) -> usize {
        self.roots.len()
    }

    pub fn object_count(&self) -> usize {
        self.objects.len()
    }

    // ── Root set management ────────────────────────────────────────────

    /// Add a root.
    pub fn add_root(&mut self, id: u64) {
        self.roots.insert(id);
    }

    /// Remove a root.
    pub fn remove_root(&mut self, id: u64) {
        self.roots.remove(&id);
    }

    /// Check if an ID is a root.
    pub fn is_root(&self, id: u64) -> bool {
        self.roots.contains(&id)
    }

    // ── Allocation ─────────────────────────────────────────────────────

    /// Allocate a new object. Returns the object ID.
    /// May trigger GC if the young-gen budget is exceeded.
    pub fn allocate(
        &mut self,
        size_bytes: usize,
        type_tag: impl Into<String>,
    ) -> Result<u64, GcError> {
        // Check if we need a minor GC.
        if self.young_allocated_bytes + size_bytes > self.config.young_budget_bytes {
            self.minor_collect();
        }

        // Check overall heap limit.
        if self.stats.current_heap_bytes + size_bytes > self.config.max_heap_bytes {
            // Try a major GC.
            self.major_collect();
            if self.stats.current_heap_bytes + size_bytes > self.config.max_heap_bytes {
                return Err(GcError::OutOfMemory {
                    requested: size_bytes,
                    available: self.config.max_heap_bytes.saturating_sub(self.stats.current_heap_bytes),
                });
            }
        }

        let id = self.next_id;
        self.next_id += 1;
        let obj = ObjectHeader::new(id, size_bytes, type_tag);
        self.objects.insert(id, obj);
        self.young_allocated_bytes += size_bytes;
        self.stats.current_heap_bytes += size_bytes;
        self.stats.total_allocated_bytes += size_bytes as u64;
        self.stats.total_objects_allocated += 1;
        self.stats.current_object_count = self.objects.len();

        if self.stats.current_heap_bytes > self.stats.peak_heap_bytes {
            self.stats.peak_heap_bytes = self.stats.current_heap_bytes;
        }

        self.update_gen_counts();
        Ok(id)
    }

    /// Add a reference from one object to another.
    /// This enforces the write barrier.
    pub fn add_reference(&mut self, from: u64, to: u64) -> Result<(), GcError> {
        // Validate both objects exist.
        if !self.objects.contains_key(&from) {
            return Err(GcError::ObjectNotFound(from));
        }
        if !self.objects.contains_key(&to) {
            return Err(GcError::InvalidReference { from, to });
        }

        // Write barrier: if `from` is old and `to` is young, remember it.
        let from_gen = self.objects[&from].generation;
        let to_gen = self.objects[&to].generation;
        if from_gen == Generation::Old && to_gen == Generation::Young {
            self.remembered_set.insert(RememberedRef {
                old_id: from,
                young_id: to,
            });
            self.stats.write_barriers += 1;
        }

        let obj = self.objects.get_mut(&from).unwrap();
        if !obj.references.contains(&to) {
            obj.references.push(to);
        }
        Ok(())
    }

    /// Remove a reference.
    pub fn remove_reference(&mut self, from: u64, to: u64) -> Result<(), GcError> {
        let obj = self
            .objects
            .get_mut(&from)
            .ok_or(GcError::ObjectNotFound(from))?;
        obj.references.retain(|r| *r != to);
        // Clean remembered set.
        self.remembered_set.retain(|r| !(r.old_id == from && r.young_id == to));
        Ok(())
    }

    // ── Mark phase (tri-color) ─────────────────────────────────────────

    /// Mark phase: tri-color marking starting from the given root IDs.
    fn mark(&mut self, root_ids: &HashSet<u64>) {
        // Reset all to white.
        for obj in self.objects.values_mut() {
            obj.color = Color::White;
        }

        // Gray worklist.
        let mut worklist: VecDeque<u64> = VecDeque::new();

        // Paint roots gray.
        for &id in root_ids {
            if let Some(obj) = self.objects.get_mut(&id) {
                obj.color = Color::Gray;
                worklist.push_back(id);
            }
        }

        // Process gray objects.
        while let Some(id) = worklist.pop_front() {
            // Collect references before borrowing mutably.
            let refs: Vec<u64> = self
                .objects
                .get(&id)
                .map(|o| o.references.clone())
                .unwrap_or_default();

            // Mark children gray if white.
            for &child_id in &refs {
                if let Some(child) = self.objects.get_mut(&child_id) {
                    if child.color == Color::White {
                        child.color = Color::Gray;
                        worklist.push_back(child_id);
                    }
                }
            }

            // Mark self black.
            if let Some(obj) = self.objects.get_mut(&id) {
                obj.color = Color::Black;
            }
        }
    }

    /// Sweep phase: remove all white objects and return reclaimed bytes.
    fn sweep(&mut self) -> (usize, usize) {
        let mut reclaimed_bytes = 0usize;
        let mut reclaimed_count = 0usize;

        let to_remove: Vec<u64> = self
            .objects
            .iter()
            .filter(|(_, obj)| obj.color == Color::White)
            .map(|(&id, _)| id)
            .collect();

        for id in &to_remove {
            if let Some(obj) = self.objects.remove(id) {
                reclaimed_bytes += obj.size_bytes;
                reclaimed_count += 1;
            }
            self.roots.remove(id);
        }

        // Clean remembered set.
        self.remembered_set
            .retain(|r| self.objects.contains_key(&r.old_id) && self.objects.contains_key(&r.young_id));

        (reclaimed_bytes, reclaimed_count)
    }

    // ── Minor collection (young gen only) ──────────────────────────────

    /// Collect young generation objects.
    pub fn minor_collect(&mut self) -> (usize, usize) {
        self.stats.minor_collections += 1;

        // Roots for minor GC: actual roots + old-gen objects that reference young-gen.
        let mut minor_roots = self.roots.clone();
        for rr in &self.remembered_set {
            minor_roots.insert(rr.old_id);
        }
        // Also include all old-gen objects as roots (they survive minor GC).
        let old_ids: Vec<u64> = self
            .objects
            .iter()
            .filter(|(_, obj)| obj.generation == Generation::Old)
            .map(|(&id, _)| id)
            .collect();
        for id in &old_ids {
            minor_roots.insert(*id);
        }

        self.mark(&minor_roots);

        // Before sweep, promote surviving young objects.
        let threshold = self.config.promotion_threshold;
        let surviving_young: Vec<u64> = self
            .objects
            .iter()
            .filter(|(_, obj)| {
                obj.generation == Generation::Young && obj.color != Color::White
            })
            .map(|(&id, _)| id)
            .collect();

        for id in &surviving_young {
            if let Some(obj) = self.objects.get_mut(id) {
                obj.survive_count += 1;
                if obj.survive_count >= threshold {
                    obj.generation = Generation::Old;
                    self.stats.promotions += 1;
                }
            }
        }

        let (bytes, count) = self.sweep();
        self.stats.total_reclaimed_bytes += bytes as u64;
        self.stats.total_objects_reclaimed += count as u64;
        self.stats.current_heap_bytes = self.stats.current_heap_bytes.saturating_sub(bytes);
        self.stats.current_object_count = self.objects.len();
        self.young_allocated_bytes = 0;
        self.update_gen_counts();
        (bytes, count)
    }

    // ── Major collection (all generations) ─────────────────────────────

    /// Full collection of all generations.
    pub fn major_collect(&mut self) -> (usize, usize) {
        self.stats.major_collections += 1;
        self.mark(&self.roots.clone());
        let (bytes, count) = self.sweep();
        self.stats.total_reclaimed_bytes += bytes as u64;
        self.stats.total_objects_reclaimed += count as u64;
        self.stats.current_heap_bytes = self.stats.current_heap_bytes.saturating_sub(bytes);
        self.stats.current_object_count = self.objects.len();
        self.young_allocated_bytes = 0;
        self.update_gen_counts();
        (bytes, count)
    }

    /// Update young/old counts in stats.
    fn update_gen_counts(&mut self) {
        let mut young = 0usize;
        let mut old = 0usize;
        for obj in self.objects.values() {
            match obj.generation {
                Generation::Young => young += 1,
                Generation::Old => old += 1,
            }
        }
        self.stats.young_object_count = young;
        self.stats.old_object_count = old;
    }

    // ── Queries ────────────────────────────────────────────────────────

    /// Get an object's header.
    pub fn get_object(&self, id: u64) -> Option<&ObjectHeader> {
        self.objects.get(&id)
    }

    /// Check if an object is alive.
    pub fn is_alive(&self, id: u64) -> bool {
        self.objects.contains_key(&id)
    }

    /// Get all live object IDs.
    pub fn live_objects(&self) -> Vec<u64> {
        let mut ids: Vec<u64> = self.objects.keys().copied().collect();
        ids.sort();
        ids
    }

    // ── Heap visualization ─────────────────────────────────────────────

    /// Produce a textual heap dump.
    pub fn heap_dump(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "Heap: {} objects, {} bytes",
            self.objects.len(),
            self.stats.current_heap_bytes
        ));
        let mut sorted: Vec<_> = self.objects.values().collect();
        sorted.sort_by_key(|o| o.id);
        for obj in sorted {
            let root_marker = if self.roots.contains(&obj.id) {
                " [root]"
            } else {
                ""
            };
            lines.push(format!(
                "  #{}: {} ({} bytes, {:?}, {:?}, survived {}){} -> {:?}",
                obj.id,
                obj.type_tag,
                obj.size_bytes,
                obj.generation,
                obj.color,
                obj.survive_count,
                root_marker,
                obj.references,
            ));
        }
        lines.join("\n")
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> GcConfig {
        GcConfig {
            promotion_threshold: 2,
            max_heap_bytes: 10_000,
            young_budget_bytes: 2_000,
            major_gc_threshold: 0.75,
        }
    }

    #[test]
    fn allocate_object() {
        let mut gc = GarbageCollector::new(test_config());
        let id = gc.allocate(100, "string").unwrap();
        assert!(gc.is_alive(id));
        assert_eq!(gc.object_count(), 1);
    }

    #[test]
    fn unreachable_objects_collected() {
        let mut gc = GarbageCollector::new(test_config());
        let a = gc.allocate(100, "a").unwrap();
        let _b = gc.allocate(100, "b").unwrap(); // unreachable
        gc.add_root(a);
        let (bytes, count) = gc.major_collect();
        assert_eq!(count, 1);
        assert_eq!(bytes, 100);
        assert!(gc.is_alive(a));
    }

    #[test]
    fn transitive_reachability() {
        let mut gc = GarbageCollector::new(test_config());
        let a = gc.allocate(50, "a").unwrap();
        let b = gc.allocate(50, "b").unwrap();
        let c = gc.allocate(50, "c").unwrap();
        gc.add_root(a);
        gc.add_reference(a, b).unwrap();
        gc.add_reference(b, c).unwrap();
        gc.major_collect();
        assert!(gc.is_alive(a));
        assert!(gc.is_alive(b));
        assert!(gc.is_alive(c));
    }

    #[test]
    fn cycle_collection() {
        let mut gc = GarbageCollector::new(test_config());
        let a = gc.allocate(50, "a").unwrap();
        let b = gc.allocate(50, "b").unwrap();
        gc.add_reference(a, b).unwrap();
        gc.add_reference(b, a).unwrap();
        // Neither is a root → both should be collected.
        let (_, count) = gc.major_collect();
        assert_eq!(count, 2);
    }

    #[test]
    fn root_management() {
        let mut gc = GarbageCollector::new(test_config());
        let id = gc.allocate(100, "obj").unwrap();
        gc.add_root(id);
        assert!(gc.is_root(id));
        gc.remove_root(id);
        assert!(!gc.is_root(id));
    }

    #[test]
    fn minor_collection_young_only() {
        let mut gc = GarbageCollector::new(test_config());
        let root = gc.allocate(50, "root").unwrap();
        gc.add_root(root);
        // Allocate junk.
        for i in 0..5 {
            gc.allocate(50, format!("junk{i}")).unwrap();
        }
        let before = gc.object_count();
        gc.minor_collect();
        assert!(gc.object_count() < before);
        assert!(gc.is_alive(root));
    }

    #[test]
    fn promotion_to_old_gen() {
        let mut gc = GarbageCollector::new(test_config());
        let id = gc.allocate(50, "obj").unwrap();
        gc.add_root(id);
        // Survive enough minor GCs to promote.
        for _ in 0..3 {
            gc.minor_collect();
        }
        let obj = gc.get_object(id).unwrap();
        assert_eq!(obj.generation, Generation::Old);
    }

    #[test]
    fn write_barrier() {
        let mut gc = GarbageCollector::new(GcConfig {
            promotion_threshold: 1,
            ..test_config()
        });
        let old = gc.allocate(50, "old").unwrap();
        gc.add_root(old);
        gc.minor_collect(); // promote to old

        let young = gc.allocate(50, "young").unwrap();
        gc.add_reference(old, young).unwrap();
        assert!(gc.stats().write_barriers > 0);

        // The young object should survive minor GC via remembered set.
        gc.minor_collect();
        assert!(gc.is_alive(young));
    }

    #[test]
    fn oom_error() {
        let mut gc = GarbageCollector::new(GcConfig {
            max_heap_bytes: 200,
            young_budget_bytes: 200,
            ..test_config()
        });
        let a = gc.allocate(100, "a").unwrap();
        gc.add_root(a);
        let b = gc.allocate(100, "b").unwrap();
        gc.add_root(b);
        let result = gc.allocate(100, "c");
        assert!(matches!(result, Err(GcError::OutOfMemory { .. })));
    }

    #[test]
    fn stats_tracking() {
        let mut gc = GarbageCollector::new(test_config());
        gc.allocate(100, "a").unwrap();
        gc.allocate(100, "b").unwrap();
        assert_eq!(gc.stats().total_objects_allocated, 2);
        assert_eq!(gc.stats().total_allocated_bytes, 200);
        gc.major_collect();
        assert_eq!(gc.stats().total_objects_reclaimed, 2);
        assert_eq!(gc.stats().total_reclaimed_bytes, 200);
    }

    #[test]
    fn heap_utilization() {
        let mut gc = GarbageCollector::new(test_config());
        gc.allocate(5000, "big").unwrap();
        let util = gc.stats().heap_utilization(gc.config().max_heap_bytes);
        assert!((util - 0.5).abs() < 0.01);
    }

    #[test]
    fn peak_heap_tracked() {
        let mut gc = GarbageCollector::new(test_config());
        let a = gc.allocate(500, "a").unwrap();
        gc.add_root(a);
        let _b = gc.allocate(500, "b").unwrap();
        assert_eq!(gc.stats().peak_heap_bytes, 1000);
        gc.major_collect(); // reclaim b
        assert_eq!(gc.stats().peak_heap_bytes, 1000); // peak unchanged
        assert_eq!(gc.stats().current_heap_bytes, 500);
    }

    #[test]
    fn live_objects_sorted() {
        let mut gc = GarbageCollector::new(test_config());
        let a = gc.allocate(10, "a").unwrap();
        let b = gc.allocate(10, "b").unwrap();
        let c = gc.allocate(10, "c").unwrap();
        gc.add_root(a);
        gc.add_root(b);
        gc.add_root(c);
        let live = gc.live_objects();
        assert_eq!(live, vec![a, b, c]);
    }

    #[test]
    fn heap_dump_format() {
        let mut gc = GarbageCollector::new(test_config());
        let id = gc.allocate(64, "node").unwrap();
        gc.add_root(id);
        let dump = gc.heap_dump();
        assert!(dump.contains("node"));
        assert!(dump.contains("[root]"));
    }

    #[test]
    fn remove_reference() {
        let mut gc = GarbageCollector::new(test_config());
        let a = gc.allocate(50, "a").unwrap();
        let b = gc.allocate(50, "b").unwrap();
        gc.add_root(a);
        gc.add_reference(a, b).unwrap();
        gc.remove_reference(a, b).unwrap();
        gc.major_collect();
        // b should be collected since the reference was removed.
        assert!(!gc.is_alive(b));
    }

    #[test]
    fn invalid_reference_error() {
        let mut gc = GarbageCollector::new(test_config());
        let a = gc.allocate(50, "a").unwrap();
        let result = gc.add_reference(a, 999);
        assert!(matches!(result, Err(GcError::InvalidReference { .. })));
    }

    #[test]
    fn gc_error_display() {
        let e = GcError::OutOfMemory {
            requested: 100,
            available: 50,
        };
        assert!(e.to_string().contains("OOM"));
    }

    #[test]
    fn gen_counts_after_collection() {
        let mut gc = GarbageCollector::new(GcConfig {
            promotion_threshold: 1,
            ..test_config()
        });
        let id = gc.allocate(50, "obj").unwrap();
        gc.add_root(id);
        assert_eq!(gc.stats().young_object_count, 1);
        gc.minor_collect(); // promote
        assert_eq!(gc.stats().old_object_count, 1);
        assert_eq!(gc.stats().young_object_count, 0);
    }

    #[test]
    fn color_starts_white() {
        let mut gc = GarbageCollector::new(test_config());
        let id = gc.allocate(10, "x").unwrap();
        assert_eq!(gc.get_object(id).unwrap().color, Color::White);
    }
}
