//! Page buffer pool — fixed page-size buffers with page table (page_id to frame),
//! LRU eviction, pin/unpin counting, dirty page tracking, flush dirty pages,
//! and buffer hit rate statistics.

use std::collections::HashMap;

// ── Errors ───────────────────────────────────────────────────────────────────

/// Errors returned by buffer pool operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BufferPoolError {
    /// No free frames and all frames are pinned (cannot evict).
    NoFreeFrames,
    /// Page ID not found in the buffer pool.
    PageNotFound(u64),
    /// The page is still pinned and cannot be evicted.
    PagePinned(u64),
    /// Frame index out of bounds.
    InvalidFrame(usize),
}

impl std::fmt::Display for BufferPoolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoFreeFrames => write!(f, "no free frames available"),
            Self::PageNotFound(id) => write!(f, "page {id} not found"),
            Self::PagePinned(id) => write!(f, "page {id} is pinned"),
            Self::InvalidFrame(idx) => write!(f, "invalid frame index {idx}"),
        }
    }
}

// ── Frame ────────────────────────────────────────────────────────────────────

#[derive(Debug)]
struct Frame {
    page_id: Option<u64>,
    data: Vec<u8>,
    pin_count: u32,
    dirty: bool,
    /// Position in LRU order (lower = more recently used).
    lru_counter: u64,
}

// ── Statistics ───────────────────────────────────────────────────────────────

/// Buffer pool statistics.
#[derive(Debug, Clone, Default)]
pub struct BufferPoolStats {
    pub total_frames: usize,
    pub used_frames: usize,
    pub free_frames: usize,
    pub dirty_frames: usize,
    pub pinned_frames: usize,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub flushes: u64,
    pub page_size: usize,
}

impl BufferPoolStats {
    /// Hit rate as a fraction in [0.0, 1.0].
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            return 0.0;
        }
        self.hits as f64 / total as f64
    }
}

// ── Flush result ─────────────────────────────────────────────────────────────

/// A dirty page flushed from the pool.
#[derive(Debug, Clone)]
pub struct FlushedPage {
    pub page_id: u64,
    pub data: Vec<u8>,
}

// ── BufferPool ───────────────────────────────────────────────────────────────

/// Fixed-size page buffer pool with LRU eviction, pin counting, and dirty tracking.
pub struct BufferPool {
    frames: Vec<Frame>,
    page_table: HashMap<u64, usize>,
    page_size: usize,
    lru_clock: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
    flushes: u64,
}

impl BufferPool {
    /// Create a new buffer pool with `num_frames` frames of `page_size` bytes each.
    pub fn new(num_frames: usize, page_size: usize) -> Self {
        assert!(num_frames > 0, "buffer pool must have > 0 frames");
        assert!(page_size > 0, "page size must be > 0");

        let frames = (0..num_frames)
            .map(|_| Frame {
                page_id: None,
                data: vec![0u8; page_size],
                pin_count: 0,
                dirty: false,
                lru_counter: 0,
            })
            .collect();

        Self {
            frames,
            page_table: HashMap::with_capacity(num_frames),
            page_size,
            lru_clock: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
            flushes: 0,
        }
    }

    /// Page size in bytes.
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// Total number of frames.
    pub fn num_frames(&self) -> usize {
        self.frames.len()
    }

    /// Fetch a page into the pool and pin it. Returns the page data.
    /// If the page is already in the pool, increments its pin count (buffer hit).
    /// If not, allocates a frame (evicting if necessary) and the caller must
    /// fill it via `write_page`.
    pub fn fetch_page(&mut self, page_id: u64) -> Result<&[u8], BufferPoolError> {
        if let Some(&frame_idx) = self.page_table.get(&page_id) {
            // Hit — pin and touch LRU.
            self.hits += 1;
            self.lru_clock += 1;
            self.frames[frame_idx].pin_count += 1;
            self.frames[frame_idx].lru_counter = self.lru_clock;
            return Ok(&self.frames[frame_idx].data);
        }

        // Miss — find a frame.
        self.misses += 1;
        let frame_idx = self.find_free_frame()?;

        // Clear old mapping if this frame was occupied.
        if let Some(old_page_id) = self.frames[frame_idx].page_id {
            self.page_table.remove(&old_page_id);
        }

        // Set up new page.
        self.lru_clock += 1;
        self.frames[frame_idx].page_id = Some(page_id);
        self.frames[frame_idx].pin_count = 1;
        self.frames[frame_idx].dirty = false;
        self.frames[frame_idx].lru_counter = self.lru_clock;
        self.frames[frame_idx].data.fill(0);
        self.page_table.insert(page_id, frame_idx);

        Ok(&self.frames[frame_idx].data)
    }

    /// Fetch a page and load it with the given data. Pins the page.
    pub fn fetch_page_with_data(
        &mut self,
        page_id: u64,
        data: &[u8],
    ) -> Result<(), BufferPoolError> {
        if let Some(&frame_idx) = self.page_table.get(&page_id) {
            self.hits += 1;
            self.lru_clock += 1;
            self.frames[frame_idx].pin_count += 1;
            self.frames[frame_idx].lru_counter = self.lru_clock;
            return Ok(());
        }

        self.misses += 1;
        let frame_idx = self.find_free_frame()?;

        if let Some(old_page_id) = self.frames[frame_idx].page_id {
            self.page_table.remove(&old_page_id);
        }

        self.lru_clock += 1;
        let len = data.len().min(self.page_size);
        self.frames[frame_idx].data[..len].copy_from_slice(&data[..len]);
        if len < self.page_size {
            self.frames[frame_idx].data[len..].fill(0);
        }
        self.frames[frame_idx].page_id = Some(page_id);
        self.frames[frame_idx].pin_count = 1;
        self.frames[frame_idx].dirty = false;
        self.frames[frame_idx].lru_counter = self.lru_clock;
        self.page_table.insert(page_id, frame_idx);
        Ok(())
    }

    /// Write data to a page already in the pool. Marks the page as dirty.
    pub fn write_page(&mut self, page_id: u64, data: &[u8]) -> Result<(), BufferPoolError> {
        let frame_idx = *self
            .page_table
            .get(&page_id)
            .ok_or(BufferPoolError::PageNotFound(page_id))?;
        let len = data.len().min(self.page_size);
        self.frames[frame_idx].data[..len].copy_from_slice(&data[..len]);
        self.frames[frame_idx].dirty = true;
        Ok(())
    }

    /// Read the data of a page in the pool.
    pub fn read_page(&self, page_id: u64) -> Result<&[u8], BufferPoolError> {
        let frame_idx = *self
            .page_table
            .get(&page_id)
            .ok_or(BufferPoolError::PageNotFound(page_id))?;
        Ok(&self.frames[frame_idx].data)
    }

    /// Unpin a page (decrement pin count). Only unpinned pages can be evicted.
    pub fn unpin_page(&mut self, page_id: u64, dirty: bool) -> Result<(), BufferPoolError> {
        let frame_idx = *self
            .page_table
            .get(&page_id)
            .ok_or(BufferPoolError::PageNotFound(page_id))?;
        let frame = &mut self.frames[frame_idx];
        if frame.pin_count == 0 {
            return Ok(()); // already unpinned
        }
        frame.pin_count -= 1;
        if dirty {
            frame.dirty = true;
        }
        Ok(())
    }

    /// Check if a page is in the pool.
    pub fn contains_page(&self, page_id: u64) -> bool {
        self.page_table.contains_key(&page_id)
    }

    /// Get the pin count for a page.
    pub fn pin_count(&self, page_id: u64) -> Option<u32> {
        self.page_table
            .get(&page_id)
            .map(|idx| self.frames[*idx].pin_count)
    }

    /// Check if a page is dirty.
    pub fn is_dirty(&self, page_id: u64) -> Option<bool> {
        self.page_table
            .get(&page_id)
            .map(|idx| self.frames[*idx].dirty)
    }

    /// Mark a page as dirty.
    pub fn mark_dirty(&mut self, page_id: u64) -> Result<(), BufferPoolError> {
        let frame_idx = *self
            .page_table
            .get(&page_id)
            .ok_or(BufferPoolError::PageNotFound(page_id))?;
        self.frames[frame_idx].dirty = true;
        Ok(())
    }

    /// Flush all dirty pages, returning their data. Clears dirty flags.
    pub fn flush_all_dirty(&mut self) -> Vec<FlushedPage> {
        let mut flushed = Vec::new();
        for frame in &mut self.frames {
            if frame.dirty {
                if let Some(page_id) = frame.page_id {
                    flushed.push(FlushedPage {
                        page_id,
                        data: frame.data.clone(),
                    });
                    frame.dirty = false;
                    self.flushes += 1;
                }
            }
        }
        flushed
    }

    /// Flush a specific dirty page.
    pub fn flush_page(&mut self, page_id: u64) -> Result<Option<FlushedPage>, BufferPoolError> {
        let frame_idx = *self
            .page_table
            .get(&page_id)
            .ok_or(BufferPoolError::PageNotFound(page_id))?;
        let frame = &mut self.frames[frame_idx];
        if !frame.dirty {
            return Ok(None);
        }
        let flushed = FlushedPage {
            page_id,
            data: frame.data.clone(),
        };
        frame.dirty = false;
        self.flushes += 1;
        Ok(Some(flushed))
    }

    /// Number of dirty pages currently in the pool.
    pub fn dirty_count(&self) -> usize {
        self.frames
            .iter()
            .filter(|f| f.dirty && f.page_id.is_some())
            .count()
    }

    /// Number of pinned pages.
    pub fn pinned_count(&self) -> usize {
        self.frames
            .iter()
            .filter(|f| f.pin_count > 0 && f.page_id.is_some())
            .count()
    }

    /// Number of used frames (containing a page).
    pub fn used_count(&self) -> usize {
        self.page_table.len()
    }

    /// Number of free frames (no page loaded).
    pub fn free_count(&self) -> usize {
        self.frames.len() - self.page_table.len()
    }

    /// All page IDs currently in the pool.
    pub fn page_ids(&self) -> Vec<u64> {
        self.page_table.keys().copied().collect()
    }

    /// Statistics snapshot.
    pub fn stats(&self) -> BufferPoolStats {
        BufferPoolStats {
            total_frames: self.frames.len(),
            used_frames: self.page_table.len(),
            free_frames: self.frames.len() - self.page_table.len(),
            dirty_frames: self.dirty_count(),
            pinned_frames: self.pinned_count(),
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
            flushes: self.flushes,
            page_size: self.page_size,
        }
    }

    /// Reset statistics counters.
    pub fn reset_stats(&mut self) {
        self.hits = 0;
        self.misses = 0;
        self.evictions = 0;
        self.flushes = 0;
    }

    /// Evict a specific (unpinned) page from the pool.
    pub fn evict_page(&mut self, page_id: u64) -> Result<Option<FlushedPage>, BufferPoolError> {
        let frame_idx = *self
            .page_table
            .get(&page_id)
            .ok_or(BufferPoolError::PageNotFound(page_id))?;
        if self.frames[frame_idx].pin_count > 0 {
            return Err(BufferPoolError::PagePinned(page_id));
        }

        let frame = &mut self.frames[frame_idx];
        let flushed = if frame.dirty {
            Some(FlushedPage {
                page_id,
                data: frame.data.clone(),
            })
        } else {
            None
        };

        frame.page_id = None;
        frame.pin_count = 0;
        frame.dirty = false;
        frame.data.fill(0);
        self.page_table.remove(&page_id);
        self.evictions += 1;
        Ok(flushed)
    }

    // ── Internal ─────────────────────────────────────────────────────

    /// Find a free frame or evict the LRU unpinned frame.
    fn find_free_frame(&mut self) -> Result<usize, BufferPoolError> {
        // Look for an empty frame first.
        for (i, frame) in self.frames.iter().enumerate() {
            if frame.page_id.is_none() {
                return Ok(i);
            }
        }

        // Evict LRU unpinned frame.
        let victim = self
            .frames
            .iter()
            .enumerate()
            .filter(|(_, f)| f.pin_count == 0 && f.page_id.is_some())
            .min_by_key(|(_, f)| f.lru_counter)
            .map(|(i, _)| i);

        match victim {
            Some(idx) => {
                let old_page_id = self.frames[idx].page_id;
                if let Some(pid) = old_page_id {
                    self.page_table.remove(&pid);
                }
                self.frames[idx].dirty = false;
                self.frames[idx].page_id = None;
                self.evictions += 1;
                Ok(idx)
            }
            None => Err(BufferPoolError::NoFreeFrames),
        }
    }
}

impl std::fmt::Debug for BufferPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufferPool")
            .field("num_frames", &self.frames.len())
            .field("page_size", &self.page_size)
            .field("used", &self.page_table.len())
            .field("hits", &self.hits)
            .field("misses", &self.misses)
            .finish()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fetch_and_read() {
        let mut pool = BufferPool::new(4, 64);
        pool.fetch_page_with_data(1, &[0xAB; 64]).unwrap();
        let data = pool.read_page(1).unwrap();
        assert_eq!(data[0], 0xAB);
        assert_eq!(data[63], 0xAB);
    }

    #[test]
    fn test_write_page() {
        let mut pool = BufferPool::new(4, 16);
        pool.fetch_page(1).unwrap();
        pool.write_page(1, &[1, 2, 3, 4]).unwrap();
        let data = pool.read_page(1).unwrap();
        assert_eq!(&data[..4], &[1, 2, 3, 4]);
        assert!(pool.is_dirty(1).unwrap());
    }

    #[test]
    fn test_hit_on_refetch() {
        let mut pool = BufferPool::new(4, 16);
        pool.fetch_page(1).unwrap();
        pool.fetch_page(1).unwrap(); // hit
        assert_eq!(pool.stats().hits, 1);
        assert_eq!(pool.stats().misses, 1);
    }

    #[test]
    fn test_pin_unpin() {
        let mut pool = BufferPool::new(4, 16);
        pool.fetch_page(1).unwrap(); // pin_count = 1
        assert_eq!(pool.pin_count(1), Some(1));
        pool.unpin_page(1, false).unwrap();
        assert_eq!(pool.pin_count(1), Some(0));
    }

    #[test]
    fn test_dirty_tracking() {
        let mut pool = BufferPool::new(4, 16);
        pool.fetch_page(1).unwrap();
        assert!(!pool.is_dirty(1).unwrap());
        pool.unpin_page(1, true).unwrap();
        assert!(pool.is_dirty(1).unwrap());
        assert_eq!(pool.dirty_count(), 1);
    }

    #[test]
    fn test_flush_dirty() {
        let mut pool = BufferPool::new(4, 16);
        pool.fetch_page_with_data(1, &[0xFF; 16]).unwrap();
        pool.write_page(1, &[0xAA; 16]).unwrap();
        pool.fetch_page_with_data(2, &[0xBB; 16]).unwrap();
        // Only page 1 is dirty.
        let flushed = pool.flush_all_dirty();
        assert_eq!(flushed.len(), 1);
        assert_eq!(flushed[0].page_id, 1);
        assert_eq!(flushed[0].data[0], 0xAA);
        assert_eq!(pool.dirty_count(), 0);
    }

    #[test]
    fn test_lru_eviction() {
        let mut pool = BufferPool::new(2, 16);
        pool.fetch_page(1).unwrap();
        pool.unpin_page(1, false).unwrap();
        pool.fetch_page(2).unwrap();
        pool.unpin_page(2, false).unwrap();
        // Access page 1 to make it recently used.
        pool.fetch_page(1).unwrap();
        pool.unpin_page(1, false).unwrap();
        // Fetch page 3 — should evict page 2 (LRU).
        pool.fetch_page(3).unwrap();
        assert!(pool.contains_page(1));
        assert!(!pool.contains_page(2));
        assert!(pool.contains_page(3));
    }

    #[test]
    fn test_no_evict_pinned() {
        let mut pool = BufferPool::new(2, 16);
        pool.fetch_page(1).unwrap(); // pinned
        pool.fetch_page(2).unwrap(); // pinned
        // Both pinned — cannot evict.
        let result = pool.fetch_page(3);
        assert!(matches!(result, Err(BufferPoolError::NoFreeFrames)));
    }

    #[test]
    fn test_evict_page() {
        let mut pool = BufferPool::new(4, 16);
        pool.fetch_page_with_data(1, &[0xCC; 16]).unwrap();
        pool.write_page(1, &[0xDD; 16]).unwrap();
        pool.unpin_page(1, true).unwrap();
        let flushed = pool.evict_page(1).unwrap();
        assert!(flushed.is_some());
        assert_eq!(flushed.unwrap().data[0], 0xDD);
        assert!(!pool.contains_page(1));
    }

    #[test]
    fn test_evict_pinned_fails() {
        let mut pool = BufferPool::new(4, 16);
        pool.fetch_page(1).unwrap(); // pinned
        assert!(matches!(pool.evict_page(1), Err(BufferPoolError::PagePinned(1))));
    }

    #[test]
    fn test_page_not_found() {
        let pool = BufferPool::new(4, 16);
        assert!(matches!(pool.read_page(99), Err(BufferPoolError::PageNotFound(99))));
    }

    #[test]
    fn test_hit_rate() {
        let mut pool = BufferPool::new(4, 16);
        pool.fetch_page(1).unwrap(); // miss
        pool.fetch_page(1).unwrap(); // hit
        pool.fetch_page(1).unwrap(); // hit
        let stats = pool.stats();
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_rate() - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_flush_specific_page() {
        let mut pool = BufferPool::new(4, 16);
        pool.fetch_page_with_data(1, &[0xEE; 16]).unwrap();
        pool.mark_dirty(1).unwrap();
        let flushed = pool.flush_page(1).unwrap();
        assert!(flushed.is_some());
        assert!(!pool.is_dirty(1).unwrap());
    }

    #[test]
    fn test_flush_clean_page() {
        let mut pool = BufferPool::new(4, 16);
        pool.fetch_page(1).unwrap();
        let flushed = pool.flush_page(1).unwrap();
        assert!(flushed.is_none());
    }

    #[test]
    fn test_used_and_free_count() {
        let mut pool = BufferPool::new(4, 16);
        assert_eq!(pool.free_count(), 4);
        pool.fetch_page(1).unwrap();
        pool.fetch_page(2).unwrap();
        assert_eq!(pool.used_count(), 2);
        assert_eq!(pool.free_count(), 2);
    }

    #[test]
    fn test_page_ids() {
        let mut pool = BufferPool::new(4, 16);
        pool.fetch_page(10).unwrap();
        pool.fetch_page(20).unwrap();
        let mut ids = pool.page_ids();
        ids.sort();
        assert_eq!(ids, vec![10, 20]);
    }

    #[test]
    fn test_reset_stats() {
        let mut pool = BufferPool::new(4, 16);
        pool.fetch_page(1).unwrap();
        pool.fetch_page(1).unwrap();
        pool.reset_stats();
        assert_eq!(pool.stats().hits, 0);
        assert_eq!(pool.stats().misses, 0);
    }

    #[test]
    fn test_multiple_pins() {
        let mut pool = BufferPool::new(4, 16);
        pool.fetch_page(1).unwrap(); // pin 1
        pool.fetch_page(1).unwrap(); // pin 2
        pool.fetch_page(1).unwrap(); // pin 3
        assert_eq!(pool.pin_count(1), Some(3));
        pool.unpin_page(1, false).unwrap();
        pool.unpin_page(1, false).unwrap();
        assert_eq!(pool.pin_count(1), Some(1));
    }
}
