//! Extent-based allocation — extent (offset, length), free list management,
//! allocation strategies (first-fit, best-fit), coalescing adjacent free
//! extents, fragmentation metrics, extent map persistence.

use serde::{Deserialize, Serialize};

// ── Errors ───────────────────────────────────────────────────────────────────

/// Errors returned by extent allocator operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExtentError {
    /// No free extent large enough for the requested size.
    NoSpace(u64),
    /// The extent to free is not recognized as allocated.
    NotAllocated { offset: u64, length: u64 },
    /// Double free detected.
    DoubleFree { offset: u64 },
    /// Invalid extent parameters.
    InvalidExtent(String),
    /// Serialization/deserialization error.
    SerdeError(String),
}

impl std::fmt::Display for ExtentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoSpace(size) => write!(f, "no free extent of size {size}"),
            Self::NotAllocated { offset, length } => {
                write!(f, "extent at offset {offset} length {length} is not allocated")
            }
            Self::DoubleFree { offset } => write!(f, "double free at offset {offset}"),
            Self::InvalidExtent(msg) => write!(f, "invalid extent: {msg}"),
            Self::SerdeError(msg) => write!(f, "serde error: {msg}"),
        }
    }
}

impl std::error::Error for ExtentError {}

// ── Extent ───────────────────────────────────────────────────────────────────

/// An extent representing a contiguous region: [offset, offset + length).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Extent {
    /// Starting offset.
    pub offset: u64,
    /// Length in units (bytes, blocks, etc.).
    pub length: u64,
}

impl Extent {
    /// Create a new extent.
    pub fn new(offset: u64, length: u64) -> Self {
        Self { offset, length }
    }

    /// End offset (exclusive).
    pub fn end(&self) -> u64 {
        self.offset + self.length
    }

    /// Whether this extent is adjacent to (or overlaps with) another.
    pub fn is_adjacent(&self, other: &Extent) -> bool {
        self.end() == other.offset || other.end() == self.offset
    }

    /// Whether this extent overlaps with another.
    pub fn overlaps(&self, other: &Extent) -> bool {
        self.offset < other.end() && other.offset < self.end()
    }

    /// Merge two adjacent or overlapping extents into one.
    pub fn merge(&self, other: &Extent) -> Option<Extent> {
        if !self.is_adjacent(other) && !self.overlaps(other) {
            return None;
        }
        let start = self.offset.min(other.offset);
        let end = self.end().max(other.end());
        Some(Extent::new(start, end - start))
    }

    /// Whether this extent contains the given offset.
    pub fn contains_offset(&self, off: u64) -> bool {
        off >= self.offset && off < self.end()
    }
}

impl std::fmt::Display for Extent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}..{})", self.offset, self.end())
    }
}

impl PartialOrd for Extent {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Extent {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.offset
            .cmp(&other.offset)
            .then_with(|| self.length.cmp(&other.length))
    }
}

// ── Allocation Strategy ──────────────────────────────────────────────────────

/// Strategy for choosing a free extent during allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocStrategy {
    /// Use the first free extent that fits.
    FirstFit,
    /// Use the smallest free extent that fits.
    BestFit,
}

// ── Fragmentation Metrics ────────────────────────────────────────────────────

/// Fragmentation statistics for the allocator.
#[derive(Debug, Clone, Default)]
pub struct FragmentationMetrics {
    /// Number of free extents.
    pub free_extent_count: usize,
    /// Total free space.
    pub total_free: u64,
    /// Largest free extent.
    pub largest_free: u64,
    /// Smallest free extent.
    pub smallest_free: u64,
    /// Average free extent size.
    pub average_free: f64,
    /// Fragmentation ratio (1.0 - largest_free / total_free).
    /// 0.0 = no fragmentation, 1.0 = fully fragmented.
    pub fragmentation_ratio: f64,
    /// Number of allocated extents.
    pub allocated_extent_count: usize,
    /// Total allocated space.
    pub total_allocated: u64,
}

// ── Extent Map (Persistence) ─────────────────────────────────────────────────

/// Serializable extent map for persistence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtentMap {
    pub free_extents: Vec<Extent>,
    pub allocated_extents: Vec<Extent>,
    pub total_capacity: u64,
}

// ── Extent Allocator ─────────────────────────────────────────────────────────

/// Extent-based allocator managing a contiguous address space.
#[derive(Debug)]
pub struct ExtentAllocator {
    /// Sorted list of free extents.
    free_list: Vec<Extent>,
    /// Sorted list of allocated extents.
    allocated: Vec<Extent>,
    /// Total capacity.
    total_capacity: u64,
    /// Allocation strategy.
    strategy: AllocStrategy,
    /// Statistics.
    alloc_count: u64,
    free_count: u64,
    coalesce_count: u64,
}

impl ExtentAllocator {
    /// Create a new allocator with the given total capacity and strategy.
    pub fn new(total_capacity: u64, strategy: AllocStrategy) -> Self {
        let free_list = if total_capacity > 0 {
            vec![Extent::new(0, total_capacity)]
        } else {
            Vec::new()
        };
        Self {
            free_list,
            allocated: Vec::new(),
            total_capacity,
            strategy,
            alloc_count: 0,
            free_count: 0,
            coalesce_count: 0,
        }
    }

    /// Allocate an extent of the given size.
    pub fn allocate(&mut self, size: u64) -> Result<Extent, ExtentError> {
        if size == 0 {
            return Err(ExtentError::InvalidExtent("cannot allocate zero size".into()));
        }

        let idx = match self.strategy {
            AllocStrategy::FirstFit => self.find_first_fit(size),
            AllocStrategy::BestFit => self.find_best_fit(size),
        };

        let idx = idx.ok_or(ExtentError::NoSpace(size))?;
        let free_extent = self.free_list[idx];

        let allocated = Extent::new(free_extent.offset, size);

        if free_extent.length == size {
            self.free_list.remove(idx);
        } else {
            self.free_list[idx] = Extent::new(
                free_extent.offset + size,
                free_extent.length - size,
            );
        }

        // Insert into allocated list in sorted order.
        let insert_pos = self
            .allocated
            .binary_search_by_key(&allocated.offset, |e| e.offset)
            .unwrap_or_else(|i| i);
        self.allocated.insert(insert_pos, allocated);
        self.alloc_count += 1;

        Ok(allocated)
    }

    fn find_first_fit(&self, size: u64) -> Option<usize> {
        self.free_list.iter().position(|e| e.length >= size)
    }

    fn find_best_fit(&self, size: u64) -> Option<usize> {
        let mut best: Option<(usize, u64)> = None;
        for (i, e) in self.free_list.iter().enumerate() {
            if e.length >= size {
                match best {
                    None => best = Some((i, e.length)),
                    Some((_, best_len)) if e.length < best_len => {
                        best = Some((i, e.length));
                    }
                    _ => {}
                }
            }
        }
        best.map(|(i, _)| i)
    }

    /// Free a previously allocated extent.
    pub fn free(&mut self, extent: Extent) -> Result<(), ExtentError> {
        let pos = self
            .allocated
            .binary_search_by_key(&extent.offset, |e| e.offset)
            .map_err(|_| ExtentError::NotAllocated {
                offset: extent.offset,
                length: extent.length,
            })?;

        if self.allocated[pos].length != extent.length {
            return Err(ExtentError::NotAllocated {
                offset: extent.offset,
                length: extent.length,
            });
        }

        self.allocated.remove(pos);

        // Insert into free list in sorted order.
        let insert_pos = self
            .free_list
            .binary_search_by_key(&extent.offset, |e| e.offset)
            .unwrap_or_else(|i| i);
        self.free_list.insert(insert_pos, extent);
        self.free_count += 1;

        // Coalesce adjacent free extents.
        self.coalesce_at(insert_pos);

        Ok(())
    }

    /// Coalesce free extents around the given index.
    fn coalesce_at(&mut self, idx: usize) {
        if self.free_list.is_empty() {
            return;
        }

        // Merge with next.
        while idx + 1 < self.free_list.len() {
            let current = self.free_list[idx];
            let next = self.free_list[idx + 1];
            if let Some(merged) = current.merge(&next) {
                self.free_list[idx] = merged;
                self.free_list.remove(idx + 1);
                self.coalesce_count += 1;
            } else {
                break;
            }
        }

        // Merge with previous.
        if idx > 0 {
            let prev = self.free_list[idx - 1];
            let current = self.free_list[idx];
            if let Some(merged) = prev.merge(&current) {
                self.free_list[idx - 1] = merged;
                self.free_list.remove(idx);
                self.coalesce_count += 1;
            }
        }
    }

    /// Run a full coalesce pass over the free list.
    pub fn coalesce_all(&mut self) -> u64 {
        let before = self.free_list.len();
        let mut i = 0;
        while i + 1 < self.free_list.len() {
            let current = self.free_list[i];
            let next = self.free_list[i + 1];
            if let Some(merged) = current.merge(&next) {
                self.free_list[i] = merged;
                self.free_list.remove(i + 1);
                self.coalesce_count += 1;
            } else {
                i += 1;
            }
        }
        (before - self.free_list.len()) as u64
    }

    /// Compute fragmentation metrics.
    pub fn fragmentation(&self) -> FragmentationMetrics {
        let free_count = self.free_list.len();
        let total_free: u64 = self.free_list.iter().map(|e| e.length).sum();
        let largest = self.free_list.iter().map(|e| e.length).max().unwrap_or(0);
        let smallest = self.free_list.iter().map(|e| e.length).min().unwrap_or(0);
        let average = if free_count > 0 {
            total_free as f64 / free_count as f64
        } else {
            0.0
        };
        let frag = if total_free > 0 {
            1.0 - (largest as f64 / total_free as f64)
        } else {
            0.0
        };
        let total_alloc: u64 = self.allocated.iter().map(|e| e.length).sum();

        FragmentationMetrics {
            free_extent_count: free_count,
            total_free,
            largest_free: largest,
            smallest_free: smallest,
            average_free: average,
            fragmentation_ratio: frag,
            allocated_extent_count: self.allocated.len(),
            total_allocated: total_alloc,
        }
    }

    /// Total capacity.
    pub fn total_capacity(&self) -> u64 {
        self.total_capacity
    }

    /// Total free space.
    pub fn total_free(&self) -> u64 {
        self.free_list.iter().map(|e| e.length).sum()
    }

    /// Total allocated space.
    pub fn total_allocated(&self) -> u64 {
        self.allocated.iter().map(|e| e.length).sum()
    }

    /// Number of free extents.
    pub fn free_extent_count(&self) -> usize {
        self.free_list.len()
    }

    /// Number of allocated extents.
    pub fn allocated_extent_count(&self) -> usize {
        self.allocated.len()
    }

    /// Current allocation strategy.
    pub fn strategy(&self) -> AllocStrategy {
        self.strategy
    }

    /// Update the allocation strategy.
    pub fn set_strategy(&mut self, strategy: AllocStrategy) {
        self.strategy = strategy;
    }

    /// Export the extent map for persistence.
    pub fn to_extent_map(&self) -> ExtentMap {
        ExtentMap {
            free_extents: self.free_list.clone(),
            allocated_extents: self.allocated.clone(),
            total_capacity: self.total_capacity,
        }
    }

    /// Restore from a persisted extent map.
    pub fn from_extent_map(map: ExtentMap, strategy: AllocStrategy) -> Result<Self, ExtentError> {
        // Validate that free + allocated = total capacity.
        let total_free: u64 = map.free_extents.iter().map(|e| e.length).sum();
        let total_alloc: u64 = map.allocated_extents.iter().map(|e| e.length).sum();
        if total_free + total_alloc != map.total_capacity {
            return Err(ExtentError::InvalidExtent(format!(
                "free ({total_free}) + allocated ({total_alloc}) != capacity ({})",
                map.total_capacity
            )));
        }
        Ok(Self {
            free_list: map.free_extents,
            allocated: map.allocated_extents,
            total_capacity: map.total_capacity,
            strategy,
            alloc_count: 0,
            free_count: 0,
            coalesce_count: 0,
        })
    }

    /// Serialize the extent map to JSON.
    pub fn to_json(&self) -> Result<String, ExtentError> {
        serde_json::to_string(&self.to_extent_map())
            .map_err(|e| ExtentError::SerdeError(e.to_string()))
    }

    /// Deserialize an extent map from JSON.
    pub fn from_json(json: &str, strategy: AllocStrategy) -> Result<Self, ExtentError> {
        let map: ExtentMap =
            serde_json::from_str(json).map_err(|e| ExtentError::SerdeError(e.to_string()))?;
        Self::from_extent_map(map, strategy)
    }

    /// Allocation count.
    pub fn alloc_count(&self) -> u64 {
        self.alloc_count
    }

    /// Free count.
    pub fn free_count(&self) -> u64 {
        self.free_count
    }

    /// Coalesce count.
    pub fn coalesce_count(&self) -> u64 {
        self.coalesce_count
    }

    /// Iterate free extents.
    pub fn free_extents(&self) -> &[Extent] {
        &self.free_list
    }

    /// Iterate allocated extents.
    pub fn allocated_extents(&self) -> &[Extent] {
        &self.allocated
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extent_basics() {
        let e = Extent::new(10, 20);
        assert_eq!(e.offset, 10);
        assert_eq!(e.length, 20);
        assert_eq!(e.end(), 30);
        assert!(e.contains_offset(10));
        assert!(e.contains_offset(29));
        assert!(!e.contains_offset(30));
    }

    #[test]
    fn extent_adjacent() {
        let a = Extent::new(0, 10);
        let b = Extent::new(10, 10);
        assert!(a.is_adjacent(&b));
        assert!(b.is_adjacent(&a));
    }

    #[test]
    fn extent_overlap() {
        let a = Extent::new(0, 10);
        let b = Extent::new(5, 10);
        assert!(a.overlaps(&b));
    }

    #[test]
    fn extent_merge() {
        let a = Extent::new(0, 10);
        let b = Extent::new(10, 20);
        let merged = a.merge(&b).unwrap();
        assert_eq!(merged, Extent::new(0, 30));
    }

    #[test]
    fn extent_no_merge_gap() {
        let a = Extent::new(0, 10);
        let b = Extent::new(20, 10);
        assert!(a.merge(&b).is_none());
    }

    #[test]
    fn first_fit_allocation() {
        let mut alloc = ExtentAllocator::new(1000, AllocStrategy::FirstFit);
        let e1 = alloc.allocate(100).unwrap();
        assert_eq!(e1, Extent::new(0, 100));
        let e2 = alloc.allocate(200).unwrap();
        assert_eq!(e2, Extent::new(100, 200));
        assert_eq!(alloc.total_allocated(), 300);
        assert_eq!(alloc.total_free(), 700);
    }

    #[test]
    fn best_fit_allocation() {
        let mut alloc = ExtentAllocator::new(1000, AllocStrategy::FirstFit);
        let e1 = alloc.allocate(300).unwrap();
        let e2 = alloc.allocate(200).unwrap();
        // Free e1, leaving a 300-size gap at front.
        alloc.free(e1).unwrap();
        // Now switch to best-fit.
        alloc.set_strategy(AllocStrategy::BestFit);
        // Allocate 150 — best fit should use the 300-gap (smallest fitting).
        let e3 = alloc.allocate(150).unwrap();
        assert_eq!(e3, Extent::new(0, 150));
        assert_eq!(alloc.allocated_extent_count(), 2);
        // e2 at 300..500, e3 at 0..150.
        assert_eq!(e2, Extent::new(300, 200));
    }

    #[test]
    fn allocate_no_space() {
        let mut alloc = ExtentAllocator::new(100, AllocStrategy::FirstFit);
        alloc.allocate(80).unwrap();
        let result = alloc.allocate(50);
        assert_eq!(result, Err(ExtentError::NoSpace(50)));
    }

    #[test]
    fn allocate_zero_size() {
        let mut alloc = ExtentAllocator::new(100, AllocStrategy::FirstFit);
        let result = alloc.allocate(0);
        assert!(result.is_err());
    }

    #[test]
    fn free_and_coalesce() {
        let mut alloc = ExtentAllocator::new(1000, AllocStrategy::FirstFit);
        let e1 = alloc.allocate(100).unwrap();
        let e2 = alloc.allocate(100).unwrap();
        let e3 = alloc.allocate(100).unwrap();
        alloc.free(e1).unwrap();
        alloc.free(e3).unwrap();
        // Free list: [0..100], [200..300], [300..1000].
        // e3 and remaining should coalesce.
        assert_eq!(alloc.free_extent_count(), 2);
        alloc.free(e2).unwrap();
        // All should coalesce into one extent [0..1000].
        assert_eq!(alloc.free_extent_count(), 1);
        assert_eq!(alloc.total_free(), 1000);
    }

    #[test]
    fn free_not_allocated() {
        let mut alloc = ExtentAllocator::new(1000, AllocStrategy::FirstFit);
        let result = alloc.free(Extent::new(500, 100));
        assert!(result.is_err());
    }

    #[test]
    fn fragmentation_metrics() {
        let mut alloc = ExtentAllocator::new(1000, AllocStrategy::FirstFit);
        let e1 = alloc.allocate(100).unwrap();
        alloc.allocate(100).unwrap();
        let e3 = alloc.allocate(100).unwrap();
        alloc.free(e1).unwrap();
        alloc.free(e3).unwrap();
        let frag = alloc.fragmentation();
        assert_eq!(frag.free_extent_count, 2);
        assert_eq!(frag.total_free, 900);
        assert!(frag.fragmentation_ratio > 0.0);
        assert_eq!(frag.allocated_extent_count, 1);
    }

    #[test]
    fn extent_map_persistence() {
        let mut alloc = ExtentAllocator::new(1000, AllocStrategy::FirstFit);
        alloc.allocate(100).unwrap();
        alloc.allocate(200).unwrap();

        let map = alloc.to_extent_map();
        assert_eq!(map.total_capacity, 1000);
        assert_eq!(map.allocated_extents.len(), 2);

        let restored = ExtentAllocator::from_extent_map(map, AllocStrategy::BestFit).unwrap();
        assert_eq!(restored.total_allocated(), 300);
        assert_eq!(restored.total_free(), 700);
    }

    #[test]
    fn json_roundtrip() {
        let mut alloc = ExtentAllocator::new(500, AllocStrategy::FirstFit);
        alloc.allocate(50).unwrap();
        alloc.allocate(100).unwrap();

        let json = alloc.to_json().unwrap();
        let restored = ExtentAllocator::from_json(&json, AllocStrategy::FirstFit).unwrap();
        assert_eq!(restored.total_allocated(), 150);
        assert_eq!(restored.total_free(), 350);
    }

    #[test]
    fn coalesce_all() {
        let mut alloc = ExtentAllocator::new(1000, AllocStrategy::FirstFit);
        let e1 = alloc.allocate(100).unwrap();
        let e2 = alloc.allocate(100).unwrap();
        let e3 = alloc.allocate(100).unwrap();
        // Free in non-adjacent order.
        alloc.free(e1).unwrap();
        alloc.free(e3).unwrap();
        alloc.free(e2).unwrap();
        // Should have coalesced during free calls.
        assert_eq!(alloc.free_extent_count(), 1);
    }

    #[test]
    fn extent_display() {
        let e = Extent::new(10, 20);
        assert_eq!(format!("{e}"), "[10..30)");
    }

    #[test]
    fn allocator_stats() {
        let mut alloc = ExtentAllocator::new(1000, AllocStrategy::FirstFit);
        let e = alloc.allocate(100).unwrap();
        alloc.free(e).unwrap();
        assert_eq!(alloc.alloc_count(), 1);
        assert_eq!(alloc.free_count(), 1);
    }

    #[test]
    fn extent_ordering() {
        let mut extents = vec![
            Extent::new(30, 10),
            Extent::new(10, 20),
            Extent::new(10, 10),
        ];
        extents.sort();
        assert_eq!(extents[0], Extent::new(10, 10));
        assert_eq!(extents[1], Extent::new(10, 20));
        assert_eq!(extents[2], Extent::new(30, 10));
    }
}
