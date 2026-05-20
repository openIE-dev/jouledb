//! Triple-Buffered Query Results
//!
//! Provides predictable latency for async query results by using
//! a triple-buffered system where results are always read from N-2 frames ago.
//!
//! Inspired by Phantom Engine's frame-based buffering strategy.

use std::collections::VecDeque;
use std::sync::{
    Arc, RwLock,
    atomic::{AtomicU64, Ordering},
};

/// Triple-buffered query result system
///
/// Ensures predictable latency: queries submitted in frame N
/// are read from frame N-2, eliminating stutter from async timing.
pub struct TripleBufferedQueries<T> {
    /// Three buffers for frame-based storage
    buffers: [Arc<RwLock<VecDeque<T>>>; 3],
    /// Current frame index (atomic for thread safety)
    frame_index: Arc<AtomicU64>,
}

impl<T> TripleBufferedQueries<T> {
    /// Create a new triple-buffered query system
    pub fn new() -> Self {
        Self {
            buffers: [
                Arc::new(RwLock::new(VecDeque::new())),
                Arc::new(RwLock::new(VecDeque::new())),
                Arc::new(RwLock::new(VecDeque::new())),
            ],
            frame_index: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Submit query result (called by async operation)
    ///
    /// # Arguments
    /// * `frame` - Frame number when query was submitted
    /// * `result` - Query result to store
    pub fn submit_result(&self, frame: u64, result: T) {
        let idx = (frame % 3) as usize;
        let mut buffer = self.buffers[idx]
            .write()
            .expect("lock poisoned: triple buffer write");
        buffer.push_back(result);
    }

    /// Read authoritative result (always from 2 frames ago)
    ///
    /// Returns the most recent result from the read frame.
    pub fn read_result(&self, frame: u64) -> Option<T>
    where
        T: Clone,
    {
        let read_frame = frame.saturating_sub(2);
        let idx = (read_frame % 3) as usize;
        let buffer = self.buffers[idx]
            .read()
            .expect("lock poisoned: triple buffer read");
        buffer.back().cloned()
    }

    /// Read all results from the read frame
    pub fn read_all_results(&self, frame: u64) -> Vec<T>
    where
        T: Clone,
    {
        let read_frame = frame.saturating_sub(2);
        let idx = (read_frame % 3) as usize;
        let buffer = self.buffers[idx]
            .read()
            .expect("lock poisoned: triple buffer read");
        buffer.iter().cloned().collect()
    }

    /// Advance frame (call at start of each frame)
    ///
    /// Returns the new frame number.
    /// Clears the buffer from 3 frames ago (now safe to reuse).
    pub fn advance_frame(&self) -> u64 {
        let frame = self.frame_index.fetch_add(1, Ordering::SeqCst);

        // Clear buffer from 3 frames ago (now safe to reuse)
        let clear_idx = ((frame + 1) % 3) as usize;
        let mut buffer = self.buffers[clear_idx]
            .write()
            .expect("lock poisoned: triple buffer write");
        buffer.clear();

        frame
    }

    /// Get current frame number
    pub fn current_frame(&self) -> u64 {
        self.frame_index.load(Ordering::Relaxed)
    }

    /// Clear all buffers
    pub fn clear(&self) {
        for buffer in &self.buffers {
            buffer
                .write()
                .expect("lock poisoned: triple buffer write")
                .clear();
        }
        self.frame_index.store(0, Ordering::Release);
    }
}

impl<T> Default for TripleBufferedQueries<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Query handle for tracking async queries
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueryHandle {
    /// Frame when query was submitted
    frame: u64,
    /// Query ID
    query_id: u64,
}

impl QueryHandle {
    /// Create a new query handle
    pub fn new(frame: u64, query_id: u64) -> Self {
        Self { frame, query_id }
    }

    /// Get the frame number
    pub fn frame(&self) -> u64 {
        self.frame
    }

    /// Get the query ID
    pub fn query_id(&self) -> u64 {
        self.query_id
    }
}

/// Triple-buffered query manager
///
/// Manages async queries with predictable latency using triple buffering.
pub struct TripleBufferedQueryManager<T> {
    /// Triple buffer for results
    buffer: Arc<TripleBufferedQueries<T>>,
    /// Next query ID
    next_query_id: Arc<AtomicU64>,
}

impl<T> TripleBufferedQueryManager<T> {
    /// Create a new query manager
    pub fn new() -> Self {
        Self {
            buffer: Arc::new(TripleBufferedQueries::new()),
            next_query_id: Arc::new(AtomicU64::new(1)),
        }
    }

    /// Submit a query (returns handle for polling)
    ///
    /// The actual query execution should call `submit_result` when done.
    pub fn submit_query(&self, frame: u64) -> QueryHandle {
        let query_id = self.next_query_id.fetch_add(1, Ordering::SeqCst);
        QueryHandle::new(frame, query_id)
    }

    /// Submit query result
    pub fn submit_result(&self, handle: QueryHandle, result: T) {
        self.buffer.submit_result(handle.frame(), result);
    }

    /// Poll query result (returns result from 2 frames ago)
    pub fn poll_query(&self, _handle: QueryHandle, current_frame: u64) -> Option<T>
    where
        T: Clone,
    {
        self.buffer.read_result(current_frame)
    }

    /// Advance frame (call at start of each frame)
    pub fn advance_frame(&self) -> u64 {
        self.buffer.advance_frame()
    }

    /// Get current frame
    pub fn current_frame(&self) -> u64 {
        self.buffer.current_frame()
    }

    /// Get the underlying buffer (for direct access)
    pub fn buffer(&self) -> Arc<TripleBufferedQueries<T>> {
        Arc::clone(&self.buffer)
    }
}

impl<T> Default for TripleBufferedQueryManager<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triple_buffer() {
        let buffer = TripleBufferedQueries::new();

        // Frame 0: Submit result to buffer[0]
        buffer.submit_result(0, "result0".to_string());

        // Frame 1: Advance, submit result to buffer[1]
        buffer.advance_frame();
        buffer.submit_result(1, "result1".to_string());

        // Frame 2: Advance, submit to buffer[2]
        buffer.advance_frame();
        buffer.submit_result(2, "result2".to_string());

        // Frame 3: Advance to frame 3
        // advance_frame returns old frame (2), but current is now 3
        // read_result(3) reads from frame 3-2=1, buffer[1%3]=buffer[1]
        let _frame2 = buffer.advance_frame(); // returns 2
        let result = buffer.read_result(buffer.current_frame());
        assert_eq!(result, Some("result1".to_string()));

        // Frame 4: Advance, should read from frame 4-2=2, buffer[2%3]=buffer[2]
        buffer.advance_frame();
        let result = buffer.read_result(buffer.current_frame());
        assert_eq!(result, Some("result2".to_string()));
    }

    #[test]
    fn test_query_manager() {
        let manager = TripleBufferedQueryManager::new();

        let frame0 = manager.current_frame();
        let handle = manager.submit_query(frame0);

        manager.submit_result(handle, "result".to_string());

        // Advance frames
        manager.advance_frame();
        manager.advance_frame();

        let current = manager.current_frame();
        let result = manager.poll_query(handle, current);
        assert_eq!(result, Some("result".to_string()));

        // Use handle to verify it's correct
        assert_eq!(handle.frame(), frame0);
    }
}
