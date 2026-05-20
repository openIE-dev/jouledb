//! Block device abstraction — fixed-size blocks, read/write at block offset,
//! block cache (LRU), write-behind buffering, device statistics (reads/writes/
//! cache hits), block bitmap for allocation.

use std::collections::HashMap;

// ── Errors ───────────────────────────────────────────────────────────────────

/// Errors returned by block device operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockDeviceError {
    /// Block number is out of range.
    OutOfRange(u64),
    /// Block is not allocated.
    NotAllocated(u64),
    /// No free blocks available.
    NoFreeBlocks,
    /// Data length does not match block size.
    SizeMismatch { expected: usize, actual: usize },
    /// Cache error.
    CacheError(String),
}

impl std::fmt::Display for BlockDeviceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OutOfRange(n) => write!(f, "block {n} out of range"),
            Self::NotAllocated(n) => write!(f, "block {n} not allocated"),
            Self::NoFreeBlocks => write!(f, "no free blocks available"),
            Self::SizeMismatch { expected, actual } => {
                write!(f, "block size mismatch: expected {expected}, got {actual}")
            }
            Self::CacheError(msg) => write!(f, "cache error: {msg}"),
        }
    }
}

impl std::error::Error for BlockDeviceError {}

// ── Device Statistics ────────────────────────────────────────────────────────

/// Block device I/O statistics.
#[derive(Debug, Clone, Default)]
pub struct DeviceStats {
    /// Total read operations (including cache hits).
    pub total_reads: u64,
    /// Total write operations.
    pub total_writes: u64,
    /// Cache hits.
    pub cache_hits: u64,
    /// Cache misses.
    pub cache_misses: u64,
    /// Blocks allocated.
    pub blocks_allocated: u64,
    /// Blocks freed.
    pub blocks_freed: u64,
    /// Write-behind flushes performed.
    pub write_behind_flushes: u64,
    /// Total blocks on device.
    pub total_blocks: u64,
    /// Free blocks.
    pub free_blocks: u64,
    /// Block size in bytes.
    pub block_size: usize,

    /// Cache hit rate as a ratio [0.0, 1.0].
    pub cache_hit_rate: f64,
}

// ── LRU Cache ────────────────────────────────────────────────────────────────

/// A simple LRU cache for blocks keyed by block number.
#[derive(Debug)]
struct LruBlockCache {
    /// block_num -> (data, lru_counter).
    entries: HashMap<u64, (Vec<u8>, u64)>,
    /// Maximum number of entries.
    capacity: usize,
    /// Monotonic counter for LRU ordering.
    counter: u64,
}

impl LruBlockCache {
    fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::new(),
            capacity,
            counter: 0,
        }
    }

    fn get(&mut self, block_num: u64) -> Option<Vec<u8>> {
        self.counter += 1;
        let counter = self.counter;
        if let Some(entry) = self.entries.get_mut(&block_num) {
            entry.1 = counter;
            return Some(entry.0.clone());
        }
        None
    }

    fn put(&mut self, block_num: u64, data: Vec<u8>) {
        self.counter += 1;
        let counter = self.counter;

        if self.entries.len() >= self.capacity && !self.entries.contains_key(&block_num) {
            self.evict_one();
        }

        self.entries.insert(block_num, (data, counter));
    }

    fn evict_one(&mut self) {
        if self.entries.is_empty() {
            return;
        }
        // Find the entry with the smallest LRU counter.
        let victim = self
            .entries
            .iter()
            .min_by_key(|(_, (_, c))| *c)
            .map(|(&k, _)| k);
        if let Some(key) = victim {
            self.entries.remove(&key);
        }
    }

    fn invalidate(&mut self, block_num: u64) {
        self.entries.remove(&block_num);
    }

    fn clear(&mut self) {
        self.entries.clear();
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn contains(&self, block_num: u64) -> bool {
        self.entries.contains_key(&block_num)
    }
}

// ── Block Bitmap ─────────────────────────────────────────────────────────────

/// Bitmap tracking which blocks are allocated.
#[derive(Debug, Clone)]
pub struct BlockBitmap {
    bits: Vec<bool>,
}

impl BlockBitmap {
    /// Create a bitmap for `total_blocks`, all initially free.
    pub fn new(total_blocks: usize) -> Self {
        Self {
            bits: vec![false; total_blocks],
        }
    }

    /// Mark a block as allocated.
    pub fn set(&mut self, block_num: u64) -> Result<(), BlockDeviceError> {
        let idx = block_num as usize;
        if idx >= self.bits.len() {
            return Err(BlockDeviceError::OutOfRange(block_num));
        }
        self.bits[idx] = true;
        Ok(())
    }

    /// Mark a block as free.
    pub fn clear(&mut self, block_num: u64) -> Result<(), BlockDeviceError> {
        let idx = block_num as usize;
        if idx >= self.bits.len() {
            return Err(BlockDeviceError::OutOfRange(block_num));
        }
        self.bits[idx] = false;
        Ok(())
    }

    /// Check if a block is allocated.
    pub fn is_allocated(&self, block_num: u64) -> bool {
        let idx = block_num as usize;
        idx < self.bits.len() && self.bits[idx]
    }

    /// Find the next free block starting from `hint`.
    pub fn find_free(&self, hint: u64) -> Option<u64> {
        let start = hint as usize;
        // Search from hint to end.
        for i in start..self.bits.len() {
            if !self.bits[i] {
                return Some(i as u64);
            }
        }
        // Wrap around.
        for i in 0..start.min(self.bits.len()) {
            if !self.bits[i] {
                return Some(i as u64);
            }
        }
        None
    }

    /// Count of allocated blocks.
    pub fn allocated_count(&self) -> usize {
        self.bits.iter().filter(|&&b| b).count()
    }

    /// Count of free blocks.
    pub fn free_count(&self) -> usize {
        self.bits.len() - self.allocated_count()
    }

    /// Total blocks.
    pub fn total(&self) -> usize {
        self.bits.len()
    }
}

// ── Write-Behind Buffer ──────────────────────────────────────────────────────

/// Write-behind buffer that defers writes until flushed.
#[derive(Debug)]
struct WriteBehindBuffer {
    /// Pending writes: block_num -> data.
    pending: HashMap<u64, Vec<u8>>,
    /// Maximum pending writes before auto-flush.
    max_pending: usize,
}

impl WriteBehindBuffer {
    fn new(max_pending: usize) -> Self {
        Self {
            pending: HashMap::new(),
            max_pending,
        }
    }

    fn add(&mut self, block_num: u64, data: Vec<u8>) {
        self.pending.insert(block_num, data);
    }

    fn is_full(&self) -> bool {
        self.pending.len() >= self.max_pending
    }

    fn drain(&mut self) -> Vec<(u64, Vec<u8>)> {
        std::mem::take(&mut self.pending).into_iter().collect()
    }

    fn get(&self, block_num: u64) -> Option<&Vec<u8>> {
        self.pending.get(&block_num)
    }

    fn len(&self) -> usize {
        self.pending.len()
    }
}

// ── Block Device ─────────────────────────────────────────────────────────────

/// A simulated block device with caching and write-behind buffering.
#[derive(Debug)]
pub struct BlockDevice {
    /// Block size in bytes.
    block_size: usize,
    /// Total number of blocks.
    total_blocks: u64,
    /// The actual block storage.
    storage: HashMap<u64, Vec<u8>>,
    /// Block allocation bitmap.
    bitmap: BlockBitmap,
    /// LRU read cache.
    cache: LruBlockCache,
    /// Write-behind buffer.
    write_buffer: WriteBehindBuffer,
    /// Next allocation hint.
    alloc_hint: u64,
    /// Statistics.
    stats: DeviceStats,
}

impl BlockDevice {
    /// Create a new block device.
    ///
    /// - `block_size`: size of each block in bytes.
    /// - `total_blocks`: total number of blocks on the device.
    /// - `cache_capacity`: max blocks in the LRU cache.
    /// - `write_buffer_capacity`: max pending writes before auto-flush.
    pub fn new(
        block_size: usize,
        total_blocks: u64,
        cache_capacity: usize,
        write_buffer_capacity: usize,
    ) -> Self {
        Self {
            block_size,
            total_blocks,
            storage: HashMap::new(),
            bitmap: BlockBitmap::new(total_blocks as usize),
            cache: LruBlockCache::new(cache_capacity),
            write_buffer: WriteBehindBuffer::new(write_buffer_capacity),
            alloc_hint: 0,
            stats: DeviceStats {
                total_blocks,
                free_blocks: total_blocks,
                block_size,
                ..Default::default()
            },
        }
    }

    /// Allocate a block, returning its block number.
    pub fn allocate_block(&mut self) -> Result<u64, BlockDeviceError> {
        let block_num = self
            .bitmap
            .find_free(self.alloc_hint)
            .ok_or(BlockDeviceError::NoFreeBlocks)?;

        self.bitmap.set(block_num)?;
        self.alloc_hint = block_num + 1;
        if self.alloc_hint >= self.total_blocks {
            self.alloc_hint = 0;
        }
        self.stats.blocks_allocated += 1;
        self.stats.free_blocks = self.bitmap.free_count() as u64;
        Ok(block_num)
    }

    /// Free a block.
    pub fn free_block(&mut self, block_num: u64) -> Result<(), BlockDeviceError> {
        if block_num >= self.total_blocks {
            return Err(BlockDeviceError::OutOfRange(block_num));
        }
        if !self.bitmap.is_allocated(block_num) {
            return Err(BlockDeviceError::NotAllocated(block_num));
        }
        self.bitmap.clear(block_num)?;
        self.storage.remove(&block_num);
        self.cache.invalidate(block_num);
        self.stats.blocks_freed += 1;
        self.stats.free_blocks = self.bitmap.free_count() as u64;
        Ok(())
    }

    /// Write data to a block.  The data must be exactly `block_size` bytes.
    pub fn write(&mut self, block_num: u64, data: Vec<u8>) -> Result<(), BlockDeviceError> {
        if block_num >= self.total_blocks {
            return Err(BlockDeviceError::OutOfRange(block_num));
        }
        if data.len() != self.block_size {
            return Err(BlockDeviceError::SizeMismatch {
                expected: self.block_size,
                actual: data.len(),
            });
        }

        // Update cache.
        self.cache.put(block_num, data.clone());
        // Buffer the write.
        self.write_buffer.add(block_num, data);
        self.stats.total_writes += 1;

        // Auto-flush if buffer is full.
        if self.write_buffer.is_full() {
            self.flush()?;
        }

        Ok(())
    }

    /// Read data from a block.
    pub fn read(&mut self, block_num: u64) -> Result<Vec<u8>, BlockDeviceError> {
        if block_num >= self.total_blocks {
            return Err(BlockDeviceError::OutOfRange(block_num));
        }

        self.stats.total_reads += 1;

        // Check write-behind buffer first (most recent data).
        if let Some(data) = self.write_buffer.get(block_num).cloned() {
            self.stats.cache_hits += 1;
            self.update_hit_rate();
            return Ok(data);
        }

        // Check cache.
        if let Some(data) = self.cache.get(block_num) {
            self.stats.cache_hits += 1;
            self.update_hit_rate();
            return Ok(data.clone());
        }

        // Read from storage.
        self.stats.cache_misses += 1;
        self.update_hit_rate();
        match self.storage.get(&block_num) {
            Some(data) => {
                self.cache.put(block_num, data.clone());
                Ok(data.clone())
            }
            None => {
                if self.bitmap.is_allocated(block_num) {
                    // Block allocated but never written: return zeros.
                    let data = vec![0u8; self.block_size];
                    self.cache.put(block_num, data.clone());
                    Ok(data)
                } else {
                    Err(BlockDeviceError::NotAllocated(block_num))
                }
            }
        }
    }

    /// Flush write-behind buffer to storage.
    pub fn flush(&mut self) -> Result<usize, BlockDeviceError> {
        let writes = self.write_buffer.drain();
        let count = writes.len();
        for (block_num, data) in writes {
            self.storage.insert(block_num, data);
        }
        self.stats.write_behind_flushes += 1;
        Ok(count)
    }

    fn update_hit_rate(&mut self) {
        let total = self.stats.cache_hits + self.stats.cache_misses;
        self.stats.cache_hit_rate = if total > 0 {
            self.stats.cache_hits as f64 / total as f64
        } else {
            0.0
        };
    }

    /// Block size in bytes.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Total blocks on the device.
    pub fn total_blocks(&self) -> u64 {
        self.total_blocks
    }

    /// Free blocks on the device.
    pub fn free_blocks(&self) -> u64 {
        self.bitmap.free_count() as u64
    }

    /// Allocated blocks.
    pub fn allocated_blocks(&self) -> u64 {
        self.bitmap.allocated_count() as u64
    }

    /// Get device statistics.
    pub fn stats(&self) -> &DeviceStats {
        &self.stats
    }

    /// Number of cached blocks.
    pub fn cached_blocks(&self) -> usize {
        self.cache.len()
    }

    /// Pending writes in the write-behind buffer.
    pub fn pending_writes(&self) -> usize {
        self.write_buffer.len()
    }

    /// Clear the cache.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Whether a block is allocated.
    pub fn is_allocated(&self, block_num: u64) -> bool {
        self.bitmap.is_allocated(block_num)
    }

    /// Access the block bitmap.
    pub fn bitmap(&self) -> &BlockBitmap {
        &self.bitmap
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_device() -> BlockDevice {
        BlockDevice::new(512, 100, 16, 8)
    }

    #[test]
    fn allocate_and_write_read() {
        let mut dev = make_device();
        let blk = dev.allocate_block().unwrap();
        let data = vec![42u8; 512];
        dev.write(blk, data.clone()).unwrap();
        let read_data = dev.read(blk).unwrap();
        assert_eq!(read_data, data);
    }

    #[test]
    fn read_unwritten_allocated_block() {
        let mut dev = make_device();
        let blk = dev.allocate_block().unwrap();
        // Reading an allocated but unwritten block should return zeros.
        dev.flush().unwrap();
        let data = dev.read(blk).unwrap();
        assert_eq!(data, vec![0u8; 512]);
    }

    #[test]
    fn read_unallocated_block() {
        let mut dev = make_device();
        let result = dev.read(50);
        assert_eq!(result, Err(BlockDeviceError::NotAllocated(50)));
    }

    #[test]
    fn write_out_of_range() {
        let mut dev = make_device();
        let data = vec![0u8; 512];
        let result = dev.write(200, data);
        assert_eq!(result, Err(BlockDeviceError::OutOfRange(200)));
    }

    #[test]
    fn write_size_mismatch() {
        let mut dev = make_device();
        let blk = dev.allocate_block().unwrap();
        let result = dev.write(blk, vec![0u8; 100]);
        assert_eq!(
            result,
            Err(BlockDeviceError::SizeMismatch {
                expected: 512,
                actual: 100,
            })
        );
    }

    #[test]
    fn free_block() {
        let mut dev = make_device();
        let blk = dev.allocate_block().unwrap();
        assert!(dev.is_allocated(blk));
        dev.free_block(blk).unwrap();
        assert!(!dev.is_allocated(blk));
    }

    #[test]
    fn free_unallocated_block() {
        let mut dev = make_device();
        let result = dev.free_block(50);
        assert_eq!(result, Err(BlockDeviceError::NotAllocated(50)));
    }

    #[test]
    fn no_free_blocks() {
        let mut dev = BlockDevice::new(512, 2, 4, 4);
        dev.allocate_block().unwrap();
        dev.allocate_block().unwrap();
        let result = dev.allocate_block();
        assert_eq!(result, Err(BlockDeviceError::NoFreeBlocks));
    }

    #[test]
    fn cache_hit_on_second_read() {
        let mut dev = make_device();
        let blk = dev.allocate_block().unwrap();
        let data = vec![1u8; 512];
        dev.write(blk, data).unwrap();
        dev.flush().unwrap();

        // First read from buffer/cache.
        dev.read(blk).unwrap();
        // Second read should be a cache hit.
        dev.read(blk).unwrap();
        assert!(dev.stats().cache_hits >= 2);
    }

    #[test]
    fn write_behind_flush() {
        let mut dev = make_device();
        let blk = dev.allocate_block().unwrap();
        let data = vec![7u8; 512];
        dev.write(blk, data.clone()).unwrap();
        assert!(dev.pending_writes() > 0);
        let flushed = dev.flush().unwrap();
        assert!(flushed > 0);
        assert_eq!(dev.pending_writes(), 0);
    }

    #[test]
    fn write_behind_auto_flush() {
        // Buffer capacity = 2, so 3rd write should auto-flush.
        let mut dev = BlockDevice::new(64, 100, 16, 2);
        let b1 = dev.allocate_block().unwrap();
        let b2 = dev.allocate_block().unwrap();
        let b3 = dev.allocate_block().unwrap();
        dev.write(b1, vec![1u8; 64]).unwrap();
        dev.write(b2, vec![2u8; 64]).unwrap();
        dev.write(b3, vec![3u8; 64]).unwrap();
        assert!(dev.stats().write_behind_flushes >= 1);
    }

    #[test]
    fn bitmap_basics() {
        let mut bm = BlockBitmap::new(10);
        assert_eq!(bm.free_count(), 10);
        assert_eq!(bm.allocated_count(), 0);
        bm.set(3).unwrap();
        assert!(bm.is_allocated(3));
        assert_eq!(bm.free_count(), 9);
        bm.clear(3).unwrap();
        assert!(!bm.is_allocated(3));
    }

    #[test]
    fn bitmap_find_free() {
        let mut bm = BlockBitmap::new(5);
        bm.set(0).unwrap();
        bm.set(1).unwrap();
        assert_eq!(bm.find_free(0), Some(2));
    }

    #[test]
    fn bitmap_find_free_wrap() {
        let mut bm = BlockBitmap::new(5);
        bm.set(3).unwrap();
        bm.set(4).unwrap();
        assert_eq!(bm.find_free(3), Some(0));
    }

    #[test]
    fn device_stats() {
        let mut dev = make_device();
        let blk = dev.allocate_block().unwrap();
        dev.write(blk, vec![0u8; 512]).unwrap();
        dev.read(blk).unwrap();
        let s = dev.stats();
        assert_eq!(s.total_writes, 1);
        assert_eq!(s.total_reads, 1);
        assert_eq!(s.blocks_allocated, 1);
    }

    #[test]
    fn clear_cache() {
        let mut dev = make_device();
        let blk = dev.allocate_block().unwrap();
        dev.write(blk, vec![0u8; 512]).unwrap();
        assert!(dev.cached_blocks() > 0);
        dev.clear_cache();
        assert_eq!(dev.cached_blocks(), 0);
    }

    #[test]
    fn lru_eviction() {
        // Cache capacity = 2.
        let mut dev = BlockDevice::new(64, 100, 2, 100);
        let b1 = dev.allocate_block().unwrap();
        let b2 = dev.allocate_block().unwrap();
        let b3 = dev.allocate_block().unwrap();
        dev.write(b1, vec![1u8; 64]).unwrap();
        dev.write(b2, vec![2u8; 64]).unwrap();
        dev.write(b3, vec![3u8; 64]).unwrap();
        // Cache should have evicted the oldest (b1).
        assert!(dev.cached_blocks() <= 2);
    }

    #[test]
    fn error_display() {
        let e = BlockDeviceError::NoFreeBlocks;
        assert_eq!(e.to_string(), "no free blocks available");
        let e = BlockDeviceError::OutOfRange(42);
        assert!(e.to_string().contains("42"));
    }
}
