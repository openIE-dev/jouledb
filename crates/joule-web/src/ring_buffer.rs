//! Ring buffer — fixed-capacity circular buffer.
//!
//! Supports push/pop front and back, overwrite-on-full mode, peek,
//! drain, iteration, bulk read/write, and watermark tracking.

use std::fmt;

// ── RingBuffer ──────────────────────────────────────────────────────────────

/// Fixed-capacity circular buffer with optional overwrite-on-full behavior.
pub struct RingBuffer<T> {
    buf: Vec<Option<T>>,
    capacity: usize,
    head: usize, // index of first element
    len: usize,
    overwrite: bool,
    high_watermark: usize,
    total_written: u64,
    total_read: u64,
}

impl<T: fmt::Debug> fmt::Debug for RingBuffer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RingBuffer")
            .field("capacity", &self.capacity)
            .field("len", &self.len)
            .field("overwrite", &self.overwrite)
            .field("high_watermark", &self.high_watermark)
            .finish()
    }
}

impl<T> RingBuffer<T> {
    /// Create a new ring buffer with the given capacity.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "Ring buffer capacity must be > 0");
        let mut buf = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            buf.push(None);
        }
        Self {
            buf,
            capacity,
            head: 0,
            len: 0,
            overwrite: false,
            high_watermark: 0,
            total_written: 0,
            total_read: 0,
        }
    }

    /// Create a ring buffer that overwrites the oldest element when full.
    pub fn with_overwrite(capacity: usize) -> Self {
        let mut rb = Self::new(capacity);
        rb.overwrite = true;
        rb
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn is_full(&self) -> bool {
        self.len == self.capacity
    }

    /// High watermark — the maximum fill level ever reached.
    pub fn high_watermark(&self) -> usize {
        self.high_watermark
    }

    /// Total items ever written.
    pub fn total_written(&self) -> u64 {
        self.total_written
    }

    /// Total items ever read/popped.
    pub fn total_read(&self) -> u64 {
        self.total_read
    }

    fn tail_index(&self) -> usize {
        (self.head + self.len) % self.capacity
    }

    fn update_watermark(&mut self) {
        if self.len > self.high_watermark {
            self.high_watermark = self.len;
        }
    }

    /// Push an item to the back. Returns Err(item) if full and not in overwrite mode.
    pub fn push_back(&mut self, item: T) -> Result<(), T> {
        if self.is_full() {
            if self.overwrite {
                // Overwrite head (oldest)
                let idx = self.head;
                self.buf[idx] = Some(item);
                self.head = (self.head + 1) % self.capacity;
                self.total_written += 1;
                return Ok(());
            }
            return Err(item);
        }
        let idx = self.tail_index();
        self.buf[idx] = Some(item);
        self.len += 1;
        self.total_written += 1;
        self.update_watermark();
        Ok(())
    }

    /// Push an item to the front.
    pub fn push_front(&mut self, item: T) -> Result<(), T> {
        if self.is_full() {
            if self.overwrite {
                let tail = if self.len == 0 {
                    self.head
                } else {
                    (self.head + self.len - 1) % self.capacity
                };
                self.head = if self.head == 0 {
                    self.capacity - 1
                } else {
                    self.head - 1
                };
                self.buf[self.head] = Some(item);
                // Drop the old tail
                self.buf[tail] = None;
                // Actually, we replaced the tail and moved head back
                // len stays the same since we replaced
                self.total_written += 1;
                return Ok(());
            }
            return Err(item);
        }
        self.head = if self.head == 0 {
            self.capacity - 1
        } else {
            self.head - 1
        };
        self.buf[self.head] = Some(item);
        self.len += 1;
        self.total_written += 1;
        self.update_watermark();
        Ok(())
    }

    /// Pop from the front (oldest item).
    pub fn pop_front(&mut self) -> Option<T> {
        if self.is_empty() {
            return None;
        }
        let item = self.buf[self.head].take();
        self.head = (self.head + 1) % self.capacity;
        self.len -= 1;
        self.total_read += 1;
        item
    }

    /// Pop from the back (newest item).
    pub fn pop_back(&mut self) -> Option<T> {
        if self.is_empty() {
            return None;
        }
        let idx = (self.head + self.len - 1) % self.capacity;
        let item = self.buf[idx].take();
        self.len -= 1;
        self.total_read += 1;
        item
    }

    /// Peek at the front without removing.
    pub fn peek_front(&self) -> Option<&T> {
        if self.is_empty() {
            return None;
        }
        self.buf[self.head].as_ref()
    }

    /// Peek at the back without removing.
    pub fn peek_back(&self) -> Option<&T> {
        if self.is_empty() {
            return None;
        }
        let idx = (self.head + self.len - 1) % self.capacity;
        self.buf[idx].as_ref()
    }

    /// Get element at logical index (0 = front).
    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len {
            return None;
        }
        let physical = (self.head + index) % self.capacity;
        self.buf[physical].as_ref()
    }

    /// Drain all elements in front-to-back order.
    pub fn drain(&mut self) -> Vec<T> {
        let mut result = Vec::with_capacity(self.len);
        while let Some(item) = self.pop_front() {
            result.push(item);
        }
        result
    }

    /// Clear all elements.
    pub fn clear(&mut self) {
        while self.pop_front().is_some() {}
    }

    /// Iterate over elements in front-to-back order.
    pub fn iter(&self) -> RingBufferIter<'_, T> {
        RingBufferIter {
            buf: self,
            index: 0,
        }
    }

    /// Reset watermark tracking.
    pub fn reset_watermark(&mut self) {
        self.high_watermark = self.len;
    }

    /// Reset all statistics.
    pub fn reset_stats(&mut self) {
        self.high_watermark = self.len;
        self.total_written = 0;
        self.total_read = 0;
    }
}

impl<T: Clone> RingBuffer<T> {
    /// Bulk write — push all items to the back.
    pub fn write_bulk(&mut self, items: &[T]) -> usize {
        let mut written = 0;
        for item in items {
            if self.push_back(item.clone()).is_ok() {
                written += 1;
            } else {
                break;
            }
        }
        written
    }

    /// Bulk read — pop up to `count` items from the front.
    pub fn read_bulk(&mut self, count: usize) -> Vec<T> {
        let mut result = Vec::with_capacity(count.min(self.len));
        for _ in 0..count {
            match self.pop_front() {
                Some(item) => result.push(item),
                None => break,
            }
        }
        result
    }
}

// ── Iterator ────────────────────────────────────────────────────────────────

pub struct RingBufferIter<'a, T> {
    buf: &'a RingBuffer<T>,
    index: usize,
}

impl<'a, T> Iterator for RingBufferIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.buf.get(self.index)?;
        self.index += 1;
        Some(item)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.buf.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a, T> ExactSizeIterator for RingBufferIter<'a, T> {}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_pop_back() {
        let mut rb = RingBuffer::new(3);
        rb.push_back(1).unwrap();
        rb.push_back(2).unwrap();
        rb.push_back(3).unwrap();
        assert_eq!(rb.pop_back(), Some(3));
        assert_eq!(rb.pop_back(), Some(2));
        assert_eq!(rb.pop_back(), Some(1));
        assert!(rb.is_empty());
    }

    #[test]
    fn test_push_pop_front() {
        let mut rb = RingBuffer::new(3);
        rb.push_front(1).unwrap();
        rb.push_front(2).unwrap();
        rb.push_front(3).unwrap();
        assert_eq!(rb.pop_front(), Some(3));
        assert_eq!(rb.pop_front(), Some(2));
        assert_eq!(rb.pop_front(), Some(1));
    }

    #[test]
    fn test_full_returns_err() {
        let mut rb = RingBuffer::new(2);
        rb.push_back(1).unwrap();
        rb.push_back(2).unwrap();
        assert!(rb.push_back(3).is_err());
    }

    #[test]
    fn test_overwrite_mode() {
        let mut rb = RingBuffer::with_overwrite(3);
        rb.push_back(1).unwrap();
        rb.push_back(2).unwrap();
        rb.push_back(3).unwrap();
        rb.push_back(4).unwrap(); // overwrites 1
        assert_eq!(rb.pop_front(), Some(2));
        assert_eq!(rb.pop_front(), Some(3));
        assert_eq!(rb.pop_front(), Some(4));
    }

    #[test]
    fn test_peek() {
        let mut rb = RingBuffer::new(5);
        rb.push_back(10).unwrap();
        rb.push_back(20).unwrap();
        rb.push_back(30).unwrap();
        assert_eq!(rb.peek_front(), Some(&10));
        assert_eq!(rb.peek_back(), Some(&30));
    }

    #[test]
    fn test_get() {
        let mut rb = RingBuffer::new(5);
        rb.push_back(100).unwrap();
        rb.push_back(200).unwrap();
        rb.push_back(300).unwrap();
        assert_eq!(rb.get(0), Some(&100));
        assert_eq!(rb.get(1), Some(&200));
        assert_eq!(rb.get(2), Some(&300));
        assert_eq!(rb.get(3), None);
    }

    #[test]
    fn test_drain() {
        let mut rb = RingBuffer::new(5);
        rb.push_back(1).unwrap();
        rb.push_back(2).unwrap();
        rb.push_back(3).unwrap();
        let drained = rb.drain();
        assert_eq!(drained, vec![1, 2, 3]);
        assert!(rb.is_empty());
    }

    #[test]
    fn test_iterator() {
        let mut rb = RingBuffer::new(5);
        rb.push_back(10).unwrap();
        rb.push_back(20).unwrap();
        rb.push_back(30).unwrap();
        let items: Vec<_> = rb.iter().copied().collect();
        assert_eq!(items, vec![10, 20, 30]);
        assert_eq!(rb.iter().len(), 3);
    }

    #[test]
    fn test_wraparound() {
        let mut rb = RingBuffer::new(3);
        rb.push_back(1).unwrap();
        rb.push_back(2).unwrap();
        rb.push_back(3).unwrap();
        rb.pop_front(); // remove 1
        rb.pop_front(); // remove 2
        rb.push_back(4).unwrap();
        rb.push_back(5).unwrap();
        let items: Vec<_> = rb.iter().copied().collect();
        assert_eq!(items, vec![3, 4, 5]);
    }

    #[test]
    fn test_bulk_write_read() {
        let mut rb = RingBuffer::new(5);
        let written = rb.write_bulk(&[10, 20, 30, 40, 50, 60]);
        assert_eq!(written, 5); // capacity limited
        let read = rb.read_bulk(3);
        assert_eq!(read, vec![10, 20, 30]);
        assert_eq!(rb.len(), 2);
    }

    #[test]
    fn test_watermark() {
        let mut rb = RingBuffer::new(10);
        rb.push_back(1).unwrap();
        rb.push_back(2).unwrap();
        rb.push_back(3).unwrap();
        assert_eq!(rb.high_watermark(), 3);
        rb.pop_front();
        rb.pop_front();
        assert_eq!(rb.high_watermark(), 3); // still 3
        assert_eq!(rb.len(), 1);
    }

    #[test]
    fn test_stats() {
        let mut rb = RingBuffer::new(10);
        rb.push_back(1).unwrap();
        rb.push_back(2).unwrap();
        rb.pop_front();
        assert_eq!(rb.total_written(), 2);
        assert_eq!(rb.total_read(), 1);
        rb.reset_stats();
        assert_eq!(rb.total_written(), 0);
        assert_eq!(rb.total_read(), 0);
    }

    #[test]
    fn test_clear() {
        let mut rb = RingBuffer::new(5);
        rb.push_back(1).unwrap();
        rb.push_back(2).unwrap();
        rb.clear();
        assert!(rb.is_empty());
        assert_eq!(rb.len(), 0);
    }

    #[test]
    fn test_capacity() {
        let rb: RingBuffer<i32> = RingBuffer::new(7);
        assert_eq!(rb.capacity(), 7);
        assert!(rb.is_empty());
        assert!(!rb.is_full());
    }
}
