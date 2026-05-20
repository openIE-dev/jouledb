//! Memory pool allocator — fixed-size block pool with alloc/free by handle,
//! pool growth policy, fragmentation tracking, statistics (allocated/free/peak),
//! reset/clear, and a type-safe wrapper.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────────────────────

/// Errors returned by pool operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PoolError {
    /// Pool exhausted — no free blocks available.
    OutOfMemory,
    /// Invalid handle (already freed, out of range, or generation mismatch).
    InvalidHandle,
    /// Pool cannot grow (fixed policy).
    CannotGrow,
}

impl fmt::Display for PoolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfMemory => write!(f, "pool out of memory"),
            Self::InvalidHandle => write!(f, "invalid block handle"),
            Self::CannotGrow => write!(f, "pool cannot grow"),
        }
    }
}

// ── Growth Policy ────────────────────────────────────────────────────────────

/// Controls whether and how the pool grows when exhausted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrowthPolicy {
    /// Fixed capacity — never grow.
    Fixed,
    /// Double the pool when exhausted.
    Double,
    /// Grow by a fixed number of blocks.
    Linear(usize),
}

// ── Handle ───────────────────────────────────────────────────────────────────

/// A handle to an allocated block. Includes generation to detect use-after-free.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockHandle {
    index: usize,
    generation: u32,
}

impl BlockHandle {
    /// Block index within the pool.
    pub fn index(&self) -> usize {
        self.index
    }

    /// Generation counter (incremented on each alloc at this slot).
    pub fn generation(&self) -> u32 {
        self.generation
    }
}

// ── Block metadata ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct BlockMeta {
    allocated: bool,
    generation: u32,
}

// ── Pool Statistics ──────────────────────────────────────────────────────────

/// Statistics for the memory pool.
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    pub total_blocks: usize,
    pub allocated_blocks: usize,
    pub free_blocks: usize,
    pub peak_allocated: usize,
    pub total_allocs: u64,
    pub total_frees: u64,
    pub growth_events: u64,
    pub block_size: usize,
}

impl PoolStats {
    /// Fragmentation ratio: fraction of the pool that is free but interspersed
    /// among allocated blocks (0.0 = no fragmentation, 1.0 = worst case).
    pub fn fragmentation_ratio(&self) -> f64 {
        if self.total_blocks == 0 || self.free_blocks == 0 {
            return 0.0;
        }
        // Simple metric: free/total when there are allocations.
        if self.allocated_blocks == 0 {
            return 0.0;
        }
        self.free_blocks as f64 / self.total_blocks as f64
    }
}

// ── MemoryPool ───────────────────────────────────────────────────────────────

/// Fixed-size block memory pool with generational handles.
pub struct MemoryPool {
    block_size: usize,
    storage: Vec<u8>,
    meta: Vec<BlockMeta>,
    free_list: Vec<usize>,
    growth_policy: GrowthPolicy,
    allocated_count: usize,
    peak_allocated: usize,
    total_allocs: u64,
    total_frees: u64,
    growth_events: u64,
}

impl MemoryPool {
    /// Create a new pool with the given block size and initial capacity (number of blocks).
    pub fn new(block_size: usize, initial_capacity: usize, growth_policy: GrowthPolicy) -> Self {
        assert!(block_size > 0, "block size must be > 0");
        assert!(initial_capacity > 0, "initial capacity must be > 0");

        let storage = vec![0u8; block_size * initial_capacity];
        let meta: Vec<BlockMeta> = (0..initial_capacity)
            .map(|_| BlockMeta {
                allocated: false,
                generation: 0,
            })
            .collect();
        let free_list: Vec<usize> = (0..initial_capacity).rev().collect();

        Self {
            block_size,
            storage,
            meta,
            free_list,
            growth_policy,
            allocated_count: 0,
            peak_allocated: 0,
            total_allocs: 0,
            total_frees: 0,
            growth_events: 0,
        }
    }

    /// Create a fixed-capacity pool.
    pub fn fixed(block_size: usize, capacity: usize) -> Self {
        Self::new(block_size, capacity, GrowthPolicy::Fixed)
    }

    /// Create a doubling-growth pool.
    pub fn doubling(block_size: usize, initial_capacity: usize) -> Self {
        Self::new(block_size, initial_capacity, GrowthPolicy::Double)
    }

    /// Block size in bytes.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Total number of blocks (allocated + free).
    pub fn total_blocks(&self) -> usize {
        self.meta.len()
    }

    /// Number of currently allocated blocks.
    pub fn allocated_count(&self) -> usize {
        self.allocated_count
    }

    /// Number of free blocks.
    pub fn free_count(&self) -> usize {
        self.free_list.len()
    }

    /// Peak number of simultaneously allocated blocks.
    pub fn peak_allocated(&self) -> usize {
        self.peak_allocated
    }

    /// Allocate a block, returning a handle.
    pub fn alloc(&mut self) -> Result<BlockHandle, PoolError> {
        if self.free_list.is_empty() {
            self.try_grow()?;
        }

        let index = self.free_list.pop().ok_or(PoolError::OutOfMemory)?;
        let block = &mut self.meta[index];
        block.allocated = true;
        block.generation += 1;
        let handle = BlockHandle {
            index,
            generation: block.generation,
        };

        self.allocated_count += 1;
        if self.allocated_count > self.peak_allocated {
            self.peak_allocated = self.allocated_count;
        }
        self.total_allocs += 1;

        // Zero the block.
        let offset = index * self.block_size;
        self.storage[offset..offset + self.block_size].fill(0);

        Ok(handle)
    }

    /// Free a block by handle.
    pub fn free(&mut self, handle: BlockHandle) -> Result<(), PoolError> {
        self.validate_handle(&handle)?;

        self.meta[handle.index].allocated = false;
        self.free_list.push(handle.index);
        self.allocated_count -= 1;
        self.total_frees += 1;
        Ok(())
    }

    /// Read from an allocated block.
    pub fn read(&self, handle: &BlockHandle) -> Result<&[u8], PoolError> {
        self.validate_handle(handle)?;
        let offset = handle.index * self.block_size;
        Ok(&self.storage[offset..offset + self.block_size])
    }

    /// Write to an allocated block. Data is truncated to block_size.
    pub fn write(&mut self, handle: &BlockHandle, data: &[u8]) -> Result<usize, PoolError> {
        self.validate_handle(handle)?;
        let offset = handle.index * self.block_size;
        let len = data.len().min(self.block_size);
        self.storage[offset..offset + len].copy_from_slice(&data[..len]);
        Ok(len)
    }

    /// Check if a handle is currently valid (allocated and correct generation).
    pub fn is_valid(&self, handle: &BlockHandle) -> bool {
        self.validate_handle(handle).is_ok()
    }

    /// Get pool statistics.
    pub fn stats(&self) -> PoolStats {
        PoolStats {
            total_blocks: self.meta.len(),
            allocated_blocks: self.allocated_count,
            free_blocks: self.free_list.len(),
            peak_allocated: self.peak_allocated,
            total_allocs: self.total_allocs,
            total_frees: self.total_frees,
            growth_events: self.growth_events,
            block_size: self.block_size,
        }
    }

    /// Compute fragmentation: count the number of free/alloc transitions
    /// in the block array (higher = more fragmented).
    pub fn fragmentation_transitions(&self) -> usize {
        if self.meta.len() <= 1 {
            return 0;
        }
        let mut transitions = 0;
        for i in 1..self.meta.len() {
            if self.meta[i].allocated != self.meta[i - 1].allocated {
                transitions += 1;
            }
        }
        transitions
    }

    /// Reset the pool: free all blocks without deallocating storage.
    pub fn reset(&mut self) {
        self.free_list.clear();
        for (i, block) in self.meta.iter_mut().enumerate() {
            block.allocated = false;
            block.generation += 1; // invalidate all outstanding handles
            self.free_list.push(i);
        }
        self.free_list.reverse();
        self.allocated_count = 0;
        self.storage.fill(0);
    }

    /// Clear and shrink back to a new pool with the given capacity.
    pub fn clear(&mut self, new_capacity: usize) {
        assert!(new_capacity > 0);
        self.storage = vec![0u8; self.block_size * new_capacity];
        self.meta = (0..new_capacity)
            .map(|_| BlockMeta {
                allocated: false,
                generation: 0,
            })
            .collect();
        self.free_list = (0..new_capacity).rev().collect();
        self.allocated_count = 0;
    }

    // ── Internal ─────────────────────────────────────────────────────

    fn validate_handle(&self, handle: &BlockHandle) -> Result<(), PoolError> {
        if handle.index >= self.meta.len() {
            return Err(PoolError::InvalidHandle);
        }
        let block = &self.meta[handle.index];
        if !block.allocated || block.generation != handle.generation {
            return Err(PoolError::InvalidHandle);
        }
        Ok(())
    }

    fn try_grow(&mut self) -> Result<(), PoolError> {
        let additional = match self.growth_policy {
            GrowthPolicy::Fixed => return Err(PoolError::OutOfMemory),
            GrowthPolicy::Double => self.meta.len(),
            GrowthPolicy::Linear(n) => n,
        };

        let old_count = self.meta.len();
        let new_count = old_count + additional;

        self.storage.resize(new_count * self.block_size, 0);
        for _ in 0..additional {
            self.meta.push(BlockMeta {
                allocated: false,
                generation: 0,
            });
        }
        // Add new blocks to free list in reverse order for LIFO behavior.
        for i in (old_count..new_count).rev() {
            self.free_list.push(i);
        }
        self.growth_events += 1;
        Ok(())
    }
}

impl fmt::Debug for MemoryPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemoryPool")
            .field("block_size", &self.block_size)
            .field("total_blocks", &self.meta.len())
            .field("allocated", &self.allocated_count)
            .field("free", &self.free_list.len())
            .field("peak", &self.peak_allocated)
            .finish()
    }
}

// ── TypedPool ────────────────────────────────────────────────────────────────

/// Type-safe wrapper over MemoryPool for fixed-size values that implement
/// Copy and have a known size.
pub struct TypedPool<T: Copy> {
    inner: MemoryPool,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Copy> TypedPool<T> {
    /// Create a new typed pool.
    pub fn new(capacity: usize, growth_policy: GrowthPolicy) -> Self {
        let block_size = std::mem::size_of::<T>().max(1);
        Self {
            inner: MemoryPool::new(block_size, capacity, growth_policy),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Allocate and store a value.
    pub fn alloc(&mut self, value: T) -> Result<BlockHandle, PoolError> {
        let handle = self.inner.alloc()?;
        let bytes =
            unsafe { std::slice::from_raw_parts(&value as *const T as *const u8, std::mem::size_of::<T>()) };
        self.inner.write(&handle, bytes)?;
        Ok(handle)
    }

    /// Read a value by handle.
    pub fn read(&self, handle: &BlockHandle) -> Result<T, PoolError> {
        let bytes = self.inner.read(handle)?;
        let mut val = std::mem::MaybeUninit::<T>::uninit();
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                val.as_mut_ptr() as *mut u8,
                std::mem::size_of::<T>(),
            );
            Ok(val.assume_init())
        }
    }

    /// Free a block.
    pub fn free(&mut self, handle: BlockHandle) -> Result<(), PoolError> {
        self.inner.free(handle)
    }

    /// Pool statistics.
    pub fn stats(&self) -> PoolStats {
        self.inner.stats()
    }

    /// Reset the pool.
    pub fn reset(&mut self) {
        self.inner.reset();
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alloc_and_free() {
        let mut pool = MemoryPool::fixed(64, 4);
        let h = pool.alloc().unwrap();
        assert_eq!(pool.allocated_count(), 1);
        assert_eq!(pool.free_count(), 3);
        pool.free(h).unwrap();
        assert_eq!(pool.allocated_count(), 0);
        assert_eq!(pool.free_count(), 4);
    }

    #[test]
    fn test_read_write() {
        let mut pool = MemoryPool::fixed(16, 4);
        let h = pool.alloc().unwrap();
        pool.write(&h, b"hello world!!!!").unwrap();
        let data = pool.read(&h).unwrap();
        assert_eq!(&data[..15], b"hello world!!!!");
    }

    #[test]
    fn test_write_truncation() {
        let mut pool = MemoryPool::fixed(4, 2);
        let h = pool.alloc().unwrap();
        let written = pool.write(&h, b"abcdefgh").unwrap();
        assert_eq!(written, 4);
        let data = pool.read(&h).unwrap();
        assert_eq!(data, b"abcd");
    }

    #[test]
    fn test_out_of_memory_fixed() {
        let mut pool = MemoryPool::fixed(8, 2);
        pool.alloc().unwrap();
        pool.alloc().unwrap();
        assert_eq!(pool.alloc(), Err(PoolError::OutOfMemory));
    }

    #[test]
    fn test_double_growth() {
        let mut pool = MemoryPool::doubling(8, 2);
        pool.alloc().unwrap();
        pool.alloc().unwrap();
        // Should trigger growth.
        let h = pool.alloc().unwrap();
        assert!(pool.is_valid(&h));
        assert_eq!(pool.total_blocks(), 4);
        assert_eq!(pool.stats().growth_events, 1);
    }

    #[test]
    fn test_linear_growth() {
        let mut pool = MemoryPool::new(8, 2, GrowthPolicy::Linear(3));
        pool.alloc().unwrap();
        pool.alloc().unwrap();
        pool.alloc().unwrap(); // triggers growth by 3
        assert_eq!(pool.total_blocks(), 5);
    }

    #[test]
    fn test_generation_use_after_free() {
        let mut pool = MemoryPool::fixed(8, 2);
        let h1 = pool.alloc().unwrap();
        pool.free(h1).unwrap();
        // h1 is now invalid even if the slot gets reused.
        assert!(!pool.is_valid(&h1));
        let h2 = pool.alloc().unwrap();
        // h2 has a new generation.
        assert!(pool.is_valid(&h2));
        assert_ne!(h1.generation(), h2.generation());
    }

    #[test]
    fn test_double_free_rejected() {
        let mut pool = MemoryPool::fixed(8, 2);
        let h = pool.alloc().unwrap();
        pool.free(h).unwrap();
        assert_eq!(pool.free(h), Err(PoolError::InvalidHandle));
    }

    #[test]
    fn test_peak_allocated() {
        let mut pool = MemoryPool::fixed(8, 10);
        let h1 = pool.alloc().unwrap();
        let h2 = pool.alloc().unwrap();
        let h3 = pool.alloc().unwrap();
        assert_eq!(pool.peak_allocated(), 3);
        pool.free(h2).unwrap();
        pool.free(h1).unwrap();
        pool.free(h3).unwrap();
        assert_eq!(pool.peak_allocated(), 3); // peak is retained
        assert_eq!(pool.allocated_count(), 0);
    }

    #[test]
    fn test_stats() {
        let mut pool = MemoryPool::fixed(16, 4);
        let h = pool.alloc().unwrap();
        pool.free(h).unwrap();
        let stats = pool.stats();
        assert_eq!(stats.total_blocks, 4);
        assert_eq!(stats.total_allocs, 1);
        assert_eq!(stats.total_frees, 1);
        assert_eq!(stats.block_size, 16);
    }

    #[test]
    fn test_fragmentation_transitions() {
        let mut pool = MemoryPool::fixed(8, 6);
        let h0 = pool.alloc().unwrap();
        let _h1 = pool.alloc().unwrap();
        let h2 = pool.alloc().unwrap();
        let _h3 = pool.alloc().unwrap();
        // Free alternating blocks to create fragmentation.
        pool.free(h0).unwrap();
        pool.free(h2).unwrap();
        let transitions = pool.fragmentation_transitions();
        assert!(transitions > 0);
    }

    #[test]
    fn test_reset() {
        let mut pool = MemoryPool::fixed(8, 4);
        let h1 = pool.alloc().unwrap();
        let _h2 = pool.alloc().unwrap();
        pool.reset();
        assert_eq!(pool.allocated_count(), 0);
        assert_eq!(pool.free_count(), 4);
        // Old handles should be invalid.
        assert!(!pool.is_valid(&h1));
    }

    #[test]
    fn test_clear() {
        let mut pool = MemoryPool::doubling(8, 4);
        pool.alloc().unwrap();
        pool.alloc().unwrap();
        pool.clear(2);
        assert_eq!(pool.total_blocks(), 2);
        assert_eq!(pool.allocated_count(), 0);
    }

    #[test]
    fn test_alloc_zeroes_block() {
        let mut pool = MemoryPool::fixed(8, 2);
        let h = pool.alloc().unwrap();
        pool.write(&h, &[0xFF; 8]).unwrap();
        pool.free(h).unwrap();
        // Re-alloc same slot.
        let h2 = pool.alloc().unwrap();
        let data = pool.read(&h2).unwrap();
        assert_eq!(data, &[0u8; 8]);
    }

    #[test]
    fn test_typed_pool() {
        let mut pool: TypedPool<u64> = TypedPool::new(4, GrowthPolicy::Fixed);
        let h = pool.alloc(42u64).unwrap();
        let val = pool.read(&h).unwrap();
        assert_eq!(val, 42u64);
        pool.free(h).unwrap();
    }

    #[test]
    fn test_typed_pool_struct() {
        #[derive(Debug, Clone, Copy, PartialEq)]
        struct Point {
            x: f64,
            y: f64,
        }
        let mut pool: TypedPool<Point> = TypedPool::new(4, GrowthPolicy::Double);
        let p = Point { x: 1.0, y: 2.0 };
        let h = pool.alloc(p).unwrap();
        let read_back = pool.read(&h).unwrap();
        assert_eq!(read_back, p);
    }

    #[test]
    fn test_typed_pool_reset() {
        let mut pool: TypedPool<i32> = TypedPool::new(4, GrowthPolicy::Fixed);
        let h = pool.alloc(100).unwrap();
        pool.reset();
        assert_eq!(pool.stats().allocated_blocks, 0);
        assert_eq!(pool.read(&h), Err(PoolError::InvalidHandle));
    }

    #[test]
    fn test_fragmentation_ratio() {
        let pool = MemoryPool::fixed(8, 4);
        let stats = pool.stats();
        // All free, no allocations => fragmentation = 0.
        assert_eq!(stats.fragmentation_ratio(), 0.0);
    }
}
