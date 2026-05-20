//! Memory allocator simulation — first-fit/best-fit/worst-fit strategies,
//! alloc/free, coalescing, fragmentation metrics, memory map, statistics.

use std::collections::HashMap;

// ── Strategy ────────────────────────────────────────────────────────────────

/// Allocation strategy for finding free blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocStrategy {
    FirstFit,
    BestFit,
    WorstFit,
}

// ── Block ───────────────────────────────────────────────────────────────────

/// A contiguous memory block (free or allocated).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block {
    pub offset: usize,
    pub size: usize,
    pub free: bool,
    /// Label for allocated blocks.
    pub label: Option<String>,
}

// ── Allocation Handle ───────────────────────────────────────────────────────

/// A handle representing an allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AllocHandle(pub u64);

// ── Error ───────────────────────────────────────────────────────────────────

/// Allocator errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AllocError {
    OutOfMemory { requested: usize, largest_free: usize },
    InvalidHandle(AllocHandle),
    DoubleFree(AllocHandle),
    InvalidSize(String),
}

impl std::fmt::Display for AllocError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AllocError::OutOfMemory { requested, largest_free } => {
                write!(f, "OOM: requested {requested}, largest free {largest_free}")
            }
            AllocError::InvalidHandle(h) => write!(f, "invalid handle: {}", h.0),
            AllocError::DoubleFree(h) => write!(f, "double free: {}", h.0),
            AllocError::InvalidSize(msg) => write!(f, "invalid size: {msg}"),
        }
    }
}

// ── Statistics ──────────────────────────────────────────────────────────────

/// Allocation statistics.
#[derive(Debug, Clone, PartialEq)]
pub struct AllocStats {
    pub total_size: usize,
    pub allocated_bytes: usize,
    pub free_bytes: usize,
    pub num_allocations: usize,
    pub num_free_blocks: usize,
    pub total_allocs_ever: u64,
    pub total_frees_ever: u64,
    pub fragmentation_ratio: f64,
    pub largest_free_block: usize,
}

// ── MemoryAllocator ─────────────────────────────────────────────────────────

/// Simulated memory allocator with configurable strategies.
#[derive(Debug)]
pub struct MemoryAllocator {
    blocks: Vec<Block>,
    strategy: AllocStrategy,
    total_size: usize,
    handles: HashMap<AllocHandle, usize>,
    next_handle: u64,
    alloc_count: u64,
    free_count: u64,
}

impl MemoryAllocator {
    /// Create a new allocator with the given total memory size and strategy.
    pub fn new(total_size: usize, strategy: AllocStrategy) -> Self {
        let blocks = vec![Block {
            offset: 0,
            size: total_size,
            free: true,
            label: None,
        }];
        Self {
            blocks,
            strategy,
            total_size,
            handles: HashMap::new(),
            next_handle: 1,
            alloc_count: 0,
            free_count: 0,
        }
    }

    /// Change the allocation strategy.
    pub fn set_strategy(&mut self, strategy: AllocStrategy) {
        self.strategy = strategy;
    }

    /// Allocate `size` bytes with an optional label.
    pub fn alloc(&mut self, size: usize, label: Option<&str>) -> Result<AllocHandle, AllocError> {
        if size == 0 {
            return Err(AllocError::InvalidSize("cannot allocate 0 bytes".into()));
        }

        let block_idx = self.find_free_block(size)?;

        let block_offset = self.blocks[block_idx].offset;
        let block_size = self.blocks[block_idx].size;

        // If the block is larger than requested, split it
        if block_size > size {
            self.blocks[block_idx] = Block {
                offset: block_offset,
                size,
                free: false,
                label: label.map(|l| l.to_string()),
            };
            self.blocks.insert(
                block_idx + 1,
                Block {
                    offset: block_offset + size,
                    size: block_size - size,
                    free: true,
                    label: None,
                },
            );
        } else {
            self.blocks[block_idx].free = false;
            self.blocks[block_idx].label = label.map(|l| l.to_string());
        }

        let handle = AllocHandle(self.next_handle);
        self.next_handle += 1;
        self.handles.insert(handle, block_offset);
        self.alloc_count += 1;

        Ok(handle)
    }

    /// Find a free block index based on the current strategy.
    fn find_free_block(&self, size: usize) -> Result<usize, AllocError> {
        let candidates: Vec<(usize, usize)> = self
            .blocks
            .iter()
            .enumerate()
            .filter(|(_, b)| b.free && b.size >= size)
            .map(|(i, b)| (i, b.size))
            .collect();

        if candidates.is_empty() {
            let largest = self
                .blocks
                .iter()
                .filter(|b| b.free)
                .map(|b| b.size)
                .max()
                .unwrap_or(0);
            return Err(AllocError::OutOfMemory {
                requested: size,
                largest_free: largest,
            });
        }

        let idx = match self.strategy {
            AllocStrategy::FirstFit => candidates[0].0,
            AllocStrategy::BestFit => {
                candidates
                    .iter()
                    .min_by_key(|(_, sz)| *sz)
                    .unwrap()
                    .0
            }
            AllocStrategy::WorstFit => {
                candidates
                    .iter()
                    .max_by_key(|(_, sz)| *sz)
                    .unwrap()
                    .0
            }
        };

        Ok(idx)
    }

    /// Free a previously allocated block.
    pub fn free(&mut self, handle: AllocHandle) -> Result<(), AllocError> {
        let offset = self
            .handles
            .remove(&handle)
            .ok_or(AllocError::InvalidHandle(handle))?;

        let block_idx = self
            .blocks
            .iter()
            .position(|b| b.offset == offset && !b.free)
            .ok_or(AllocError::DoubleFree(handle))?;

        self.blocks[block_idx].free = true;
        self.blocks[block_idx].label = None;
        self.free_count += 1;

        self.coalesce();
        Ok(())
    }

    /// Merge adjacent free blocks.
    fn coalesce(&mut self) {
        let mut i = 0;
        while i + 1 < self.blocks.len() {
            if self.blocks[i].free && self.blocks[i + 1].free {
                let merged_size = self.blocks[i].size + self.blocks[i + 1].size;
                self.blocks[i].size = merged_size;
                self.blocks.remove(i + 1);
                // Don't increment i — check the merged block again
            } else {
                i += 1;
            }
        }
    }

    /// Get allocation statistics.
    pub fn stats(&self) -> AllocStats {
        let allocated_bytes: usize = self
            .blocks
            .iter()
            .filter(|b| !b.free)
            .map(|b| b.size)
            .sum();
        let free_bytes = self.total_size - allocated_bytes;
        let num_free_blocks = self.blocks.iter().filter(|b| b.free).count();
        let num_allocations = self.blocks.iter().filter(|b| !b.free).count();
        let largest_free = self
            .blocks
            .iter()
            .filter(|b| b.free)
            .map(|b| b.size)
            .max()
            .unwrap_or(0);

        // Fragmentation: 1 - (largest_free / total_free)
        let fragmentation = if free_bytes > 0 {
            1.0 - (largest_free as f64 / free_bytes as f64)
        } else {
            0.0
        };

        AllocStats {
            total_size: self.total_size,
            allocated_bytes,
            free_bytes,
            num_allocations,
            num_free_blocks,
            total_allocs_ever: self.alloc_count,
            total_frees_ever: self.free_count,
            fragmentation_ratio: fragmentation,
            largest_free_block: largest_free,
        }
    }

    /// Get the memory map (all blocks in order).
    pub fn memory_map(&self) -> Vec<Block> {
        self.blocks.clone()
    }

    /// Render a visual memory map string.
    pub fn render_map(&self, width: usize) -> String {
        let mut output = String::new();
        for block in &self.blocks {
            let chars = std::cmp::max(1, (block.size as f64 / self.total_size as f64 * width as f64) as usize);
            let ch = if block.free { '.' } else { '#' };
            for _ in 0..chars {
                output.push(ch);
            }
        }
        output
    }

    /// Total size of the allocator.
    pub fn total_size(&self) -> usize {
        self.total_size
    }

    /// Current strategy.
    pub fn strategy(&self) -> AllocStrategy {
        self.strategy
    }

    /// Number of active allocations.
    pub fn active_allocations(&self) -> usize {
        self.handles.len()
    }

    /// Get the size of an allocation by handle.
    pub fn allocation_size(&self, handle: AllocHandle) -> Option<usize> {
        let offset = self.handles.get(&handle)?;
        self.blocks
            .iter()
            .find(|b| b.offset == *offset && !b.free)
            .map(|b| b.size)
    }

    /// Get the label of an allocation by handle.
    pub fn allocation_label(&self, handle: AllocHandle) -> Option<String> {
        let offset = self.handles.get(&handle)?;
        self.blocks
            .iter()
            .find(|b| b.offset == *offset && !b.free)
            .and_then(|b| b.label.clone())
    }

    /// Attempt to resize an allocation (in-place if possible).
    pub fn realloc(
        &mut self,
        handle: AllocHandle,
        new_size: usize,
    ) -> Result<AllocHandle, AllocError> {
        if new_size == 0 {
            return Err(AllocError::InvalidSize("cannot realloc to 0".into()));
        }

        let offset = *self
            .handles
            .get(&handle)
            .ok_or(AllocError::InvalidHandle(handle))?;

        let block_idx = self
            .blocks
            .iter()
            .position(|b| b.offset == offset && !b.free)
            .ok_or(AllocError::InvalidHandle(handle))?;

        let current_size = self.blocks[block_idx].size;
        let label = self.blocks[block_idx].label.clone();

        if new_size <= current_size {
            // Shrink: split off remainder
            if new_size < current_size {
                self.blocks[block_idx].size = new_size;
                self.blocks.insert(
                    block_idx + 1,
                    Block {
                        offset: offset + new_size,
                        size: current_size - new_size,
                        free: true,
                        label: None,
                    },
                );
                self.coalesce();
            }
            Ok(handle)
        } else {
            // Try to expand in-place if next block is free and large enough
            let can_expand = if block_idx + 1 < self.blocks.len() {
                let next = &self.blocks[block_idx + 1];
                next.free && current_size + next.size >= new_size
            } else {
                false
            };

            if can_expand {
                let next_size = self.blocks[block_idx + 1].size;
                let extra_needed = new_size - current_size;
                if extra_needed == next_size {
                    self.blocks.remove(block_idx + 1);
                } else {
                    self.blocks[block_idx + 1].offset += extra_needed;
                    self.blocks[block_idx + 1].size -= extra_needed;
                }
                self.blocks[block_idx].size = new_size;
                Ok(handle)
            } else {
                // Must move: free old, alloc new
                self.free(handle)?;
                self.alloc(new_size, label.as_deref())
            }
        }
    }

    /// Reset the allocator to its initial state.
    pub fn reset(&mut self) {
        self.blocks = vec![Block {
            offset: 0,
            size: self.total_size,
            free: true,
            label: None,
        }];
        self.handles.clear();
        self.alloc_count = 0;
        self.free_count = 0;
        self.next_handle = 1;
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_alloc() {
        let mut alloc = MemoryAllocator::new(1024, AllocStrategy::FirstFit);
        let h = alloc.alloc(256, Some("block1")).unwrap();
        assert_eq!(alloc.allocation_size(h), Some(256));
        assert_eq!(alloc.allocation_label(h), Some("block1".into()));
    }

    #[test]
    fn test_alloc_and_free() {
        let mut alloc = MemoryAllocator::new(1024, AllocStrategy::FirstFit);
        let h = alloc.alloc(512, None).unwrap();
        alloc.free(h).unwrap();
        let stats = alloc.stats();
        assert_eq!(stats.free_bytes, 1024);
        assert_eq!(stats.allocated_bytes, 0);
    }

    #[test]
    fn test_oom() {
        let mut alloc = MemoryAllocator::new(100, AllocStrategy::FirstFit);
        let result = alloc.alloc(200, None);
        assert!(matches!(result, Err(AllocError::OutOfMemory { .. })));
    }

    #[test]
    fn test_zero_alloc() {
        let mut alloc = MemoryAllocator::new(100, AllocStrategy::FirstFit);
        let result = alloc.alloc(0, None);
        assert!(matches!(result, Err(AllocError::InvalidSize(_))));
    }

    #[test]
    fn test_coalescing() {
        let mut alloc = MemoryAllocator::new(1024, AllocStrategy::FirstFit);
        let h1 = alloc.alloc(256, None).unwrap();
        let h2 = alloc.alloc(256, None).unwrap();
        let h3 = alloc.alloc(256, None).unwrap();
        alloc.free(h1).unwrap();
        alloc.free(h3).unwrap();
        alloc.free(h2).unwrap();
        // All three freed blocks should coalesce
        let stats = alloc.stats();
        assert_eq!(stats.num_free_blocks, 1);
        assert_eq!(stats.free_bytes, 1024);
    }

    #[test]
    fn test_fragmentation() {
        let mut alloc = MemoryAllocator::new(1024, AllocStrategy::FirstFit);
        let h1 = alloc.alloc(256, None).unwrap();
        let _h2 = alloc.alloc(256, None).unwrap();
        let h3 = alloc.alloc(256, None).unwrap();
        alloc.free(h1).unwrap();
        alloc.free(h3).unwrap();
        let stats = alloc.stats();
        assert!(stats.fragmentation_ratio > 0.0);
        assert_eq!(stats.num_free_blocks, 2);
    }

    #[test]
    fn test_best_fit() {
        let mut alloc = MemoryAllocator::new(1024, AllocStrategy::BestFit);
        let h1 = alloc.alloc(100, Some("a")).unwrap();
        let _h2 = alloc.alloc(300, Some("b")).unwrap();
        let h3 = alloc.alloc(200, Some("c")).unwrap();
        alloc.free(h1).unwrap();
        alloc.free(h3).unwrap();
        // Free blocks: 100 @ offset 0, 200 @ offset 400, 424 @ offset 600
        // Best fit for 150 should pick the 200-byte block
        let h4 = alloc.alloc(150, Some("d")).unwrap();
        assert_eq!(alloc.allocation_size(h4), Some(150));
    }

    #[test]
    fn test_worst_fit() {
        let mut alloc = MemoryAllocator::new(1024, AllocStrategy::WorstFit);
        let h1 = alloc.alloc(100, None).unwrap();
        let _h2 = alloc.alloc(100, None).unwrap();
        alloc.free(h1).unwrap();
        // Free blocks: 100 at offset 0, 824 at offset 200
        // Worst fit for 50 should pick the 824-byte block
        let h3 = alloc.alloc(50, None).unwrap();
        // The 50-byte alloc should be at offset 200 (the start of the larger free block)
        let offset = *alloc.handles.get(&h3).unwrap();
        assert_eq!(offset, 200);
    }

    #[test]
    fn test_multiple_allocs_fill() {
        let mut alloc = MemoryAllocator::new(100, AllocStrategy::FirstFit);
        let _h1 = alloc.alloc(30, None).unwrap();
        let _h2 = alloc.alloc(30, None).unwrap();
        let _h3 = alloc.alloc(30, None).unwrap();
        // Only 10 bytes left
        let result = alloc.alloc(20, None);
        assert!(matches!(result, Err(AllocError::OutOfMemory { .. })));
    }

    #[test]
    fn test_memory_map() {
        let mut alloc = MemoryAllocator::new(1024, AllocStrategy::FirstFit);
        alloc.alloc(512, Some("half")).unwrap();
        let map = alloc.memory_map();
        assert_eq!(map.len(), 2);
        assert!(!map[0].free);
        assert!(map[1].free);
    }

    #[test]
    fn test_render_map() {
        let mut alloc = MemoryAllocator::new(100, AllocStrategy::FirstFit);
        alloc.alloc(50, None).unwrap();
        let rendered = alloc.render_map(10);
        assert!(rendered.contains('#'));
        assert!(rendered.contains('.'));
    }

    #[test]
    fn test_stats_counters() {
        let mut alloc = MemoryAllocator::new(1024, AllocStrategy::FirstFit);
        let h1 = alloc.alloc(100, None).unwrap();
        let _h2 = alloc.alloc(100, None).unwrap();
        alloc.free(h1).unwrap();
        let stats = alloc.stats();
        assert_eq!(stats.total_allocs_ever, 2);
        assert_eq!(stats.total_frees_ever, 1);
        assert_eq!(stats.num_allocations, 1);
    }

    #[test]
    fn test_realloc_shrink() {
        let mut alloc = MemoryAllocator::new(1024, AllocStrategy::FirstFit);
        let h = alloc.alloc(200, Some("shrinkable")).unwrap();
        let h2 = alloc.realloc(h, 100).unwrap();
        assert_eq!(h, h2); // Same handle for in-place shrink
        assert_eq!(alloc.allocation_size(h2), Some(100));
    }

    #[test]
    fn test_realloc_grow_inplace() {
        let mut alloc = MemoryAllocator::new(1024, AllocStrategy::FirstFit);
        let h = alloc.alloc(200, None).unwrap();
        // Next block is free and large, so should expand in-place
        let h2 = alloc.realloc(h, 400).unwrap();
        assert_eq!(h, h2);
        assert_eq!(alloc.allocation_size(h2), Some(400));
    }

    #[test]
    fn test_realloc_must_move() {
        let mut alloc = MemoryAllocator::new(1024, AllocStrategy::FirstFit);
        let h1 = alloc.alloc(256, None).unwrap();
        let _h2 = alloc.alloc(256, None).unwrap();
        // h1 can't grow in place because h2 is adjacent
        let h3 = alloc.realloc(h1, 512).unwrap();
        assert_ne!(h1, h3);
        assert_eq!(alloc.allocation_size(h3), Some(512));
    }

    #[test]
    fn test_invalid_handle() {
        let mut alloc = MemoryAllocator::new(1024, AllocStrategy::FirstFit);
        let result = alloc.free(AllocHandle(999));
        assert!(matches!(result, Err(AllocError::InvalidHandle(_))));
    }

    #[test]
    fn test_reset() {
        let mut alloc = MemoryAllocator::new(1024, AllocStrategy::FirstFit);
        alloc.alloc(100, None).unwrap();
        alloc.alloc(200, None).unwrap();
        alloc.reset();
        let stats = alloc.stats();
        assert_eq!(stats.free_bytes, 1024);
        assert_eq!(stats.num_allocations, 0);
        assert_eq!(stats.total_allocs_ever, 0);
    }

    #[test]
    fn test_active_allocations() {
        let mut alloc = MemoryAllocator::new(1024, AllocStrategy::FirstFit);
        let h1 = alloc.alloc(100, None).unwrap();
        let _h2 = alloc.alloc(100, None).unwrap();
        assert_eq!(alloc.active_allocations(), 2);
        alloc.free(h1).unwrap();
        assert_eq!(alloc.active_allocations(), 1);
    }

    #[test]
    fn test_set_strategy() {
        let mut alloc = MemoryAllocator::new(1024, AllocStrategy::FirstFit);
        assert_eq!(alloc.strategy(), AllocStrategy::FirstFit);
        alloc.set_strategy(AllocStrategy::BestFit);
        assert_eq!(alloc.strategy(), AllocStrategy::BestFit);
    }
}
