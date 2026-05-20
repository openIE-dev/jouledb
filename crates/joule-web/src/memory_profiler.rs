//! Memory profiling — allocation tracking, leak detection, allocation histogram,
//! top allocators, memory timeline, high-water mark, and allocation rate.

use std::collections::HashMap;

// ── Allocation ───────────────────────────────────────────────────

/// A single memory allocation.
#[derive(Debug, Clone)]
pub struct Allocation {
    pub id: u64,
    pub size: usize,
    pub location: String,
    pub timestamp_us: u64,
    pub freed: bool,
    pub freed_at_us: Option<u64>,
}

impl Allocation {
    pub fn new(id: u64, size: usize, location: &str, timestamp_us: u64) -> Self {
        Self {
            id,
            size,
            location: location.to_string(),
            timestamp_us,
            freed: false,
            freed_at_us: None,
        }
    }

    /// Duration this allocation was live (in microseconds), if freed.
    pub fn lifetime_us(&self) -> Option<u64> {
        self.freed_at_us.map(|ft| ft.saturating_sub(self.timestamp_us))
    }
}

// ── Allocation Bucket ────────────────────────────────────────────

/// A bucket in the allocation size histogram.
#[derive(Debug, Clone)]
pub struct AllocationBucket {
    pub min_size: usize,
    pub max_size: usize,
    pub count: usize,
    pub total_bytes: usize,
}

// ── Allocator Summary ────────────────────────────────────────────

/// Summary of allocations from a single location.
#[derive(Debug, Clone)]
pub struct AllocatorSummary {
    pub location: String,
    pub allocation_count: usize,
    pub total_bytes: usize,
    pub avg_size: f64,
    pub max_size: usize,
    pub leaked_count: usize,
    pub leaked_bytes: usize,
}

// ── Timeline Point ───────────────────────────────────────────────

/// A point in the memory usage timeline.
#[derive(Debug, Clone)]
pub struct TimelinePoint {
    pub timestamp_us: u64,
    pub live_bytes: usize,
    pub live_count: usize,
    pub total_allocated: usize,
    pub total_freed: usize,
}

// ── Leak Info ────────────────────────────────────────────────────

/// Information about a detected memory leak.
#[derive(Debug, Clone)]
pub struct LeakInfo {
    pub allocation_id: u64,
    pub size: usize,
    pub location: String,
    pub allocated_at_us: u64,
    pub age_us: u64,
}

// ── Memory Profiler ──────────────────────────────────────────────

/// Memory profiler that tracks allocations, detects leaks, and computes statistics.
pub struct MemoryProfiler {
    allocations: HashMap<u64, Allocation>,
    next_id: u64,
    total_allocated_bytes: usize,
    total_freed_bytes: usize,
    high_water_mark_bytes: usize,
    current_live_bytes: usize,
    current_live_count: usize,
    timeline: Vec<TimelinePoint>,
}

impl MemoryProfiler {
    pub fn new() -> Self {
        Self {
            allocations: HashMap::new(),
            next_id: 0,
            total_allocated_bytes: 0,
            total_freed_bytes: 0,
            high_water_mark_bytes: 0,
            current_live_bytes: 0,
            current_live_count: 0,
            timeline: Vec::new(),
        }
    }

    /// Record a new allocation. Returns the allocation id.
    pub fn allocate(&mut self, size: usize, location: &str, timestamp_us: u64) -> u64 {
        let id = self.next_id;
        self.next_id += 1;

        self.allocations
            .insert(id, Allocation::new(id, size, location, timestamp_us));

        self.total_allocated_bytes += size;
        self.current_live_bytes += size;
        self.current_live_count += 1;

        if self.current_live_bytes > self.high_water_mark_bytes {
            self.high_water_mark_bytes = self.current_live_bytes;
        }

        self.record_timeline_point(timestamp_us);
        id
    }

    /// Record a deallocation. Returns true if the allocation was found.
    pub fn free(&mut self, id: u64, timestamp_us: u64) -> bool {
        if let Some(alloc) = self.allocations.get_mut(&id) {
            if alloc.freed {
                return false; // double free
            }
            alloc.freed = true;
            alloc.freed_at_us = Some(timestamp_us);
            self.total_freed_bytes += alloc.size;
            self.current_live_bytes = self.current_live_bytes.saturating_sub(alloc.size);
            self.current_live_count = self.current_live_count.saturating_sub(1);
            self.record_timeline_point(timestamp_us);
            true
        } else {
            false
        }
    }

    fn record_timeline_point(&mut self, timestamp_us: u64) {
        self.timeline.push(TimelinePoint {
            timestamp_us,
            live_bytes: self.current_live_bytes,
            live_count: self.current_live_count,
            total_allocated: self.total_allocated_bytes,
            total_freed: self.total_freed_bytes,
        });
    }

    /// Current live (unfreed) bytes.
    pub fn live_bytes(&self) -> usize {
        self.current_live_bytes
    }

    /// Current live (unfreed) allocation count.
    pub fn live_count(&self) -> usize {
        self.current_live_count
    }

    /// Total bytes ever allocated.
    pub fn total_allocated(&self) -> usize {
        self.total_allocated_bytes
    }

    /// Total bytes freed.
    pub fn total_freed(&self) -> usize {
        self.total_freed_bytes
    }

    /// High-water mark (peak live bytes).
    pub fn high_water_mark(&self) -> usize {
        self.high_water_mark_bytes
    }

    /// Detect leaks: allocations that have not been freed.
    /// `current_time_us` is used to compute the age of each leak.
    pub fn detect_leaks(&self, current_time_us: u64) -> Vec<LeakInfo> {
        let mut leaks: Vec<LeakInfo> = self
            .allocations
            .values()
            .filter(|a| !a.freed)
            .map(|a| LeakInfo {
                allocation_id: a.id,
                size: a.size,
                location: a.location.clone(),
                allocated_at_us: a.timestamp_us,
                age_us: current_time_us.saturating_sub(a.timestamp_us),
            })
            .collect();
        leaks.sort_by(|a, b| b.size.cmp(&a.size).then_with(|| a.allocation_id.cmp(&b.allocation_id)));
        leaks
    }

    /// Build a size histogram with the given bucket boundaries.
    /// Each boundary defines the upper limit (exclusive) of a bucket.
    pub fn allocation_histogram(&self, boundaries: &[usize]) -> Vec<AllocationBucket> {
        let mut buckets = Vec::new();
        let mut sorted_bounds = boundaries.to_vec();
        sorted_bounds.sort();

        let mut prev = 0;
        for &bound in &sorted_bounds {
            buckets.push(AllocationBucket {
                min_size: prev,
                max_size: bound,
                count: 0,
                total_bytes: 0,
            });
            prev = bound;
        }
        // Overflow bucket
        buckets.push(AllocationBucket {
            min_size: prev,
            max_size: usize::MAX,
            count: 0,
            total_bytes: 0,
        });

        for alloc in self.allocations.values() {
            for bucket in buckets.iter_mut() {
                if alloc.size >= bucket.min_size && alloc.size < bucket.max_size {
                    bucket.count += 1;
                    bucket.total_bytes += alloc.size;
                    break;
                }
            }
        }

        buckets
    }

    /// Top N allocators by total bytes allocated, grouped by location.
    pub fn top_allocators(&self, n: usize) -> Vec<AllocatorSummary> {
        let mut by_location: HashMap<&str, Vec<&Allocation>> = HashMap::new();
        for alloc in self.allocations.values() {
            by_location.entry(alloc.location.as_str()).or_default().push(alloc);
        }

        let mut summaries: Vec<AllocatorSummary> = by_location
            .into_iter()
            .map(|(location, allocs)| {
                let allocation_count = allocs.len();
                let total_bytes: usize = allocs.iter().map(|a| a.size).sum();
                let max_size = allocs.iter().map(|a| a.size).max().unwrap_or(0);
                let leaked: Vec<&&Allocation> = allocs.iter().filter(|a| !a.freed).collect();
                let leaked_count = leaked.len();
                let leaked_bytes: usize = leaked.iter().map(|a| a.size).sum();
                let avg_size = if allocation_count > 0 {
                    total_bytes as f64 / allocation_count as f64
                } else {
                    0.0
                };

                AllocatorSummary {
                    location: location.to_string(),
                    allocation_count,
                    total_bytes,
                    avg_size,
                    max_size,
                    leaked_count,
                    leaked_bytes,
                }
            })
            .collect();

        summaries.sort_by(|a, b| b.total_bytes.cmp(&a.total_bytes).then_with(|| a.location.cmp(&b.location)));
        summaries.truncate(n);
        summaries
    }

    /// Get the memory timeline.
    pub fn timeline(&self) -> &[TimelinePoint] {
        &self.timeline
    }

    /// Compute allocation rate (bytes per microsecond) over a time window.
    pub fn allocation_rate(&self, start_us: u64, end_us: u64) -> f64 {
        if end_us <= start_us {
            return 0.0;
        }
        let bytes_in_window: usize = self
            .allocations
            .values()
            .filter(|a| a.timestamp_us >= start_us && a.timestamp_us < end_us)
            .map(|a| a.size)
            .sum();
        bytes_in_window as f64 / (end_us - start_us) as f64
    }

    /// Total number of allocations tracked.
    pub fn allocation_count(&self) -> usize {
        self.allocations.len()
    }

    /// Get an allocation by id.
    pub fn get_allocation(&self, id: u64) -> Option<&Allocation> {
        self.allocations.get(&id)
    }
}

impl Default for MemoryProfiler {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_profiler() -> MemoryProfiler {
        let mut p = MemoryProfiler::new();
        p.allocate(100, "vec_new", 1000);
        p.allocate(200, "string_new", 2000);
        p.allocate(50, "vec_new", 3000);
        p
    }

    #[test]
    fn test_allocate() {
        let mut p = MemoryProfiler::new();
        let id = p.allocate(100, "test", 1000);
        assert_eq!(id, 0);
        assert_eq!(p.live_bytes(), 100);
        assert_eq!(p.live_count(), 1);
    }

    #[test]
    fn test_free() {
        let mut p = MemoryProfiler::new();
        let id = p.allocate(100, "test", 1000);
        assert!(p.free(id, 2000));
        assert_eq!(p.live_bytes(), 0);
        assert_eq!(p.live_count(), 0);
    }

    #[test]
    fn test_free_unknown() {
        let mut p = MemoryProfiler::new();
        assert!(!p.free(999, 1000));
    }

    #[test]
    fn test_double_free() {
        let mut p = MemoryProfiler::new();
        let id = p.allocate(100, "test", 1000);
        assert!(p.free(id, 2000));
        assert!(!p.free(id, 3000)); // double free
    }

    #[test]
    fn test_total_allocated_and_freed() {
        let mut p = setup_profiler();
        assert_eq!(p.total_allocated(), 350);
        p.free(0, 4000);
        assert_eq!(p.total_freed(), 100);
        assert_eq!(p.live_bytes(), 250);
    }

    #[test]
    fn test_high_water_mark() {
        let mut p = MemoryProfiler::new();
        let a = p.allocate(100, "a", 1000);
        let _b = p.allocate(200, "b", 2000);
        assert_eq!(p.high_water_mark(), 300);
        p.free(a, 3000);
        assert_eq!(p.high_water_mark(), 300); // unchanged
        assert_eq!(p.live_bytes(), 200);
    }

    #[test]
    fn test_detect_leaks() {
        let mut p = setup_profiler();
        p.free(0, 4000);
        let leaks = p.detect_leaks(5000);
        assert_eq!(leaks.len(), 2); // id=1 and id=2 not freed
        assert_eq!(leaks[0].size, 200); // sorted by size desc
    }

    #[test]
    fn test_detect_no_leaks() {
        let mut p = MemoryProfiler::new();
        let id = p.allocate(100, "a", 1000);
        p.free(id, 2000);
        let leaks = p.detect_leaks(3000);
        assert!(leaks.is_empty());
    }

    #[test]
    fn test_leak_age() {
        let mut p = MemoryProfiler::new();
        p.allocate(100, "a", 1000);
        let leaks = p.detect_leaks(5000);
        assert_eq!(leaks[0].age_us, 4000);
    }

    #[test]
    fn test_allocation_histogram() {
        let p = setup_profiler();
        let hist = p.allocation_histogram(&[64, 128, 256]);
        // 50 bytes -> [0, 64) bucket
        // 100 bytes -> [64, 128) bucket
        // 200 bytes -> [128, 256) bucket
        assert_eq!(hist.len(), 4); // 3 explicit + overflow
        assert_eq!(hist[0].count, 1); // [0, 64): 50
        assert_eq!(hist[1].count, 1); // [64, 128): 100
        assert_eq!(hist[2].count, 1); // [128, 256): 200
        assert_eq!(hist[3].count, 0); // [256, MAX): none
    }

    #[test]
    fn test_top_allocators() {
        let p = setup_profiler();
        let top = p.top_allocators(10);
        assert_eq!(top.len(), 2); // vec_new and string_new
        // vec_new: 100 + 50 = 150, string_new: 200
        assert_eq!(top[0].location, "string_new");
        assert_eq!(top[0].total_bytes, 200);
        assert_eq!(top[1].location, "vec_new");
        assert_eq!(top[1].allocation_count, 2);
    }

    #[test]
    fn test_top_allocators_limit() {
        let p = setup_profiler();
        let top = p.top_allocators(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_allocator_leak_tracking() {
        let mut p = MemoryProfiler::new();
        let a = p.allocate(100, "src", 1000);
        p.allocate(200, "src", 2000);
        p.free(a, 3000);
        let top = p.top_allocators(10);
        let src = top.iter().find(|t| t.location == "src").unwrap();
        assert_eq!(src.leaked_count, 1);
        assert_eq!(src.leaked_bytes, 200);
    }

    #[test]
    fn test_timeline() {
        let p = setup_profiler();
        let tl = p.timeline();
        assert_eq!(tl.len(), 3);
        assert_eq!(tl[0].live_bytes, 100);
        assert_eq!(tl[1].live_bytes, 300);
        assert_eq!(tl[2].live_bytes, 350);
    }

    #[test]
    fn test_allocation_rate() {
        let p = setup_profiler();
        // 3 allocs between 1000 and 4000: 350 bytes / 3000 us
        let rate = p.allocation_rate(1000, 4000);
        assert!((rate - 350.0 / 3000.0).abs() < 0.001);
    }

    #[test]
    fn test_allocation_rate_zero_window() {
        let p = setup_profiler();
        assert!((p.allocation_rate(1000, 1000) - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_allocation_lifetime() {
        let mut p = MemoryProfiler::new();
        let id = p.allocate(100, "test", 1000);
        p.free(id, 3000);
        let alloc = p.get_allocation(id).unwrap();
        assert_eq!(alloc.lifetime_us(), Some(2000));
    }

    #[test]
    fn test_allocation_lifetime_not_freed() {
        let mut p = MemoryProfiler::new();
        let id = p.allocate(100, "test", 1000);
        let alloc = p.get_allocation(id).unwrap();
        assert_eq!(alloc.lifetime_us(), None);
    }

    #[test]
    fn test_allocation_count() {
        let p = setup_profiler();
        assert_eq!(p.allocation_count(), 3);
    }

    #[test]
    fn test_empty_profiler() {
        let p = MemoryProfiler::new();
        assert_eq!(p.live_bytes(), 0);
        assert_eq!(p.live_count(), 0);
        assert_eq!(p.high_water_mark(), 0);
        assert!(p.detect_leaks(0).is_empty());
    }

    #[test]
    fn test_allocator_avg_size() {
        let p = setup_profiler();
        let top = p.top_allocators(10);
        let vec_alloc = top.iter().find(|t| t.location == "vec_new").unwrap();
        assert!((vec_alloc.avg_size - 75.0).abs() < 0.01);
    }

    #[test]
    fn test_histogram_overflow_bucket() {
        let mut p = MemoryProfiler::new();
        p.allocate(1000, "big", 1);
        let hist = p.allocation_histogram(&[64, 128]);
        assert_eq!(hist.last().unwrap().count, 1);
        assert_eq!(hist.last().unwrap().total_bytes, 1000);
    }
}
