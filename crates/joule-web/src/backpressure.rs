//! Backpressure strategies — drop, buffer, sample, batch, adaptive rate control.
//!
//! Pure Rust backpressure mechanisms for flow control between producers
//! and consumers: drop-newest, drop-oldest, block-producer, watermark
//! buffering, sampling, window batching, and adaptive rate adjustment.

use std::collections::VecDeque;

// ── Strategy ───────────────────────────────────────────────────

/// Which backpressure strategy to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    DropNewest,
    DropOldest,
    BlockProducer,
}

// ── Backpressure Buffer ────────────────────────────────────────

/// Buffer with a configurable backpressure strategy.
#[derive(Debug)]
pub struct BackpressureBuffer<T> {
    buffer: VecDeque<T>,
    capacity: usize,
    strategy: Strategy,
    dropped_count: u64,
    total_pushed: u64,
}

impl<T> BackpressureBuffer<T> {
    pub fn new(capacity: usize, strategy: Strategy) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity: capacity.max(1),
            strategy,
            dropped_count: 0,
            total_pushed: 0,
        }
    }

    /// Push an item. Returns the dropped item (if any), or Err if blocked.
    pub fn push(&mut self, item: T) -> Result<Option<T>, T> {
        self.total_pushed += 1;
        if self.buffer.len() < self.capacity {
            self.buffer.push_back(item);
            return Ok(None);
        }
        match self.strategy {
            Strategy::DropNewest => {
                self.dropped_count += 1;
                Ok(Some(item)) // Drop the incoming item.
            }
            Strategy::DropOldest => {
                let old = self.buffer.pop_front();
                self.buffer.push_back(item);
                self.dropped_count += 1;
                Ok(old)
            }
            Strategy::BlockProducer => Err(item),
        }
    }

    pub fn pop(&mut self) -> Option<T> {
        self.buffer.pop_front()
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub fn is_full(&self) -> bool {
        self.buffer.len() >= self.capacity
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn dropped_count(&self) -> u64 {
        self.dropped_count
    }

    pub fn total_pushed(&self) -> u64 {
        self.total_pushed
    }
}

// ── Watermark Buffer ───────────────────────────────────────────

/// Buffer with high/low watermarks for flow control signals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlowSignal {
    /// Producer should continue.
    Continue,
    /// Producer should pause (buffer above high watermark).
    Pause,
    /// Producer can resume (buffer below low watermark).
    Resume,
}

#[derive(Debug)]
pub struct WatermarkBuffer<T> {
    buffer: VecDeque<T>,
    capacity: usize,
    high_watermark: usize,
    low_watermark: usize,
    paused: bool,
}

impl<T> WatermarkBuffer<T> {
    pub fn new(capacity: usize, low_pct: f64, high_pct: f64) -> Self {
        let cap = capacity.max(1);
        let low = ((cap as f64 * low_pct) as usize).max(1);
        let high = ((cap as f64 * high_pct) as usize).min(cap);
        Self {
            buffer: VecDeque::with_capacity(cap),
            capacity: cap,
            high_watermark: high,
            low_watermark: low,
            paused: false,
        }
    }

    /// Push an item and return the flow control signal.
    pub fn push(&mut self, item: T) -> FlowSignal {
        if self.buffer.len() >= self.capacity {
            return FlowSignal::Pause;
        }
        self.buffer.push_back(item);
        if self.buffer.len() >= self.high_watermark && !self.paused {
            self.paused = true;
            FlowSignal::Pause
        } else {
            FlowSignal::Continue
        }
    }

    /// Pop an item and return the flow control signal.
    pub fn pop(&mut self) -> (Option<T>, FlowSignal) {
        let item = self.buffer.pop_front();
        if self.paused && self.buffer.len() <= self.low_watermark {
            self.paused = false;
            (item, FlowSignal::Resume)
        } else {
            (item, FlowSignal::Continue)
        }
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub fn is_paused(&self) -> bool {
        self.paused
    }

    pub fn high_watermark(&self) -> usize {
        self.high_watermark
    }

    pub fn low_watermark(&self) -> usize {
        self.low_watermark
    }
}

// ── Sampler ────────────────────────────────────────────────────

/// Samples every Nth item, dropping the rest.
#[derive(Debug)]
pub struct Sampler<T> {
    interval: u64,
    count: u64,
    dropped: u64,
    last: Option<T>,
}

impl<T: Clone> Sampler<T> {
    pub fn new(interval: u64) -> Self {
        Self {
            interval: interval.max(1),
            count: 0,
            dropped: 0,
            last: None,
        }
    }

    /// Submit an item. Returns Some if this item is sampled.
    pub fn submit(&mut self, item: T) -> Option<T> {
        self.count += 1;
        self.last = Some(item.clone());
        if self.count % self.interval == 0 {
            Some(item)
        } else {
            self.dropped += 1;
            None
        }
    }

    pub fn total_submitted(&self) -> u64 {
        self.count
    }

    pub fn total_dropped(&self) -> u64 {
        self.dropped
    }

    pub fn last(&self) -> Option<&T> {
        self.last.as_ref()
    }
}

// ── Window Batcher ─────────────────────────────────────────────

/// Collects items into fixed-size windows (batches).
#[derive(Debug)]
pub struct WindowBatcher<T> {
    window_size: usize,
    current: Vec<T>,
    batches_emitted: u64,
}

impl<T> WindowBatcher<T> {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size: window_size.max(1),
            current: Vec::with_capacity(window_size),
            batches_emitted: 0,
        }
    }

    /// Push an item. Returns a complete batch when window is full.
    pub fn push(&mut self, item: T) -> Option<Vec<T>> {
        self.current.push(item);
        if self.current.len() >= self.window_size {
            self.batches_emitted += 1;
            let batch = std::mem::replace(
                &mut self.current,
                Vec::with_capacity(self.window_size),
            );
            Some(batch)
        } else {
            None
        }
    }

    /// Flush any partial batch.
    pub fn flush(&mut self) -> Option<Vec<T>> {
        if self.current.is_empty() {
            None
        } else {
            self.batches_emitted += 1;
            Some(std::mem::replace(
                &mut self.current,
                Vec::with_capacity(self.window_size),
            ))
        }
    }

    pub fn pending(&self) -> usize {
        self.current.len()
    }

    pub fn batches_emitted(&self) -> u64 {
        self.batches_emitted
    }
}

// ── Adaptive Rate ──────────────────────────────────────────────

/// Adaptive rate controller: adjusts throughput based on success/failure feedback.
#[derive(Debug)]
pub struct AdaptiveRate {
    /// Current rate limit (items per window).
    rate: f64,
    min_rate: f64,
    max_rate: f64,
    /// Multiplicative increase factor on success.
    increase_factor: f64,
    /// Multiplicative decrease factor on failure.
    decrease_factor: f64,
    /// Items allowed this window.
    window_budget: u64,
    /// Items consumed this window.
    window_used: u64,
    /// Total adjustments made.
    adjustments: u64,
}

impl AdaptiveRate {
    pub fn new(initial_rate: f64, min_rate: f64, max_rate: f64) -> Self {
        Self {
            rate: initial_rate.clamp(min_rate, max_rate),
            min_rate,
            max_rate,
            increase_factor: 1.1,
            decrease_factor: 0.5,
            window_budget: initial_rate as u64,
            window_used: 0,
            adjustments: 0,
        }
    }

    pub fn with_factors(mut self, increase: f64, decrease: f64) -> Self {
        self.increase_factor = increase.max(1.0);
        self.decrease_factor = decrease.clamp(0.01, 1.0);
        self
    }

    pub fn rate(&self) -> f64 {
        self.rate
    }

    /// Try to acquire one permit in the current window.
    pub fn try_acquire(&mut self) -> bool {
        if self.window_used < self.window_budget {
            self.window_used += 1;
            true
        } else {
            false
        }
    }

    /// Report success — increase rate.
    pub fn on_success(&mut self) {
        self.rate = (self.rate * self.increase_factor).min(self.max_rate);
        self.adjustments += 1;
    }

    /// Report failure — decrease rate.
    pub fn on_failure(&mut self) {
        self.rate = (self.rate * self.decrease_factor).max(self.min_rate);
        self.adjustments += 1;
    }

    /// Reset window for the next period.
    pub fn next_window(&mut self) {
        self.window_budget = self.rate as u64;
        self.window_used = 0;
    }

    pub fn window_remaining(&self) -> u64 {
        self.window_budget.saturating_sub(self.window_used)
    }

    pub fn adjustments(&self) -> u64 {
        self.adjustments
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drop_newest() {
        let mut buf = BackpressureBuffer::new(2, Strategy::DropNewest);
        assert!(buf.push(1).unwrap().is_none());
        assert!(buf.push(2).unwrap().is_none());
        // Full — drop incoming.
        let dropped = buf.push(3).unwrap();
        assert_eq!(dropped, Some(3));
        assert_eq!(buf.dropped_count(), 1);
        assert_eq!(buf.pop(), Some(1)); // Original items preserved.
    }

    #[test]
    fn test_drop_oldest() {
        let mut buf = BackpressureBuffer::new(2, Strategy::DropOldest);
        buf.push(1).unwrap();
        buf.push(2).unwrap();
        let dropped = buf.push(3).unwrap();
        assert_eq!(dropped, Some(1)); // Oldest dropped.
        assert_eq!(buf.pop(), Some(2));
        assert_eq!(buf.pop(), Some(3));
    }

    #[test]
    fn test_block_producer() {
        let mut buf = BackpressureBuffer::new(2, Strategy::BlockProducer);
        buf.push(1).unwrap();
        buf.push(2).unwrap();
        let result = buf.push(3);
        assert!(result.is_err()); // Blocked.
        assert_eq!(result.unwrap_err(), 3);
    }

    #[test]
    fn test_watermark_signals() {
        let mut buf = WatermarkBuffer::new(10, 0.3, 0.7);
        // High watermark at 7, low watermark at 3.
        for i in 0..6 {
            assert_eq!(buf.push(i), FlowSignal::Continue);
        }
        // 7th item triggers pause.
        assert_eq!(buf.push(99), FlowSignal::Pause);
        assert!(buf.is_paused());
        // Pop down toward low watermark.
        for _ in 0..3 {
            let (_, sig) = buf.pop();
            assert_eq!(sig, FlowSignal::Continue);
        }
        // 4th pop brings len to 3 (== low watermark), triggers resume.
        let (_, sig) = buf.pop();
        assert_eq!(sig, FlowSignal::Resume);
    }

    #[test]
    fn test_sampler() {
        let mut s = Sampler::new(3);
        assert!(s.submit(1).is_none());
        assert!(s.submit(2).is_none());
        assert_eq!(s.submit(3), Some(3)); // Every 3rd item.
        assert!(s.submit(4).is_none());
        assert!(s.submit(5).is_none());
        assert_eq!(s.submit(6), Some(6));
        assert_eq!(s.total_dropped(), 4);
    }

    #[test]
    fn test_window_batcher() {
        let mut b = WindowBatcher::new(3);
        assert!(b.push(1).is_none());
        assert!(b.push(2).is_none());
        let batch = b.push(3).unwrap();
        assert_eq!(batch, vec![1, 2, 3]);
        assert_eq!(b.batches_emitted(), 1);
    }

    #[test]
    fn test_window_batcher_flush() {
        let mut b = WindowBatcher::new(5);
        b.push(1);
        b.push(2);
        let partial = b.flush().unwrap();
        assert_eq!(partial, vec![1, 2]);
        assert!(b.flush().is_none()); // No pending.
    }

    #[test]
    fn test_adaptive_rate_increase() {
        let mut ar = AdaptiveRate::new(10.0, 1.0, 100.0);
        let initial = ar.rate();
        ar.on_success();
        assert!(ar.rate() > initial);
    }

    #[test]
    fn test_adaptive_rate_decrease() {
        let mut ar = AdaptiveRate::new(10.0, 1.0, 100.0);
        let initial = ar.rate();
        ar.on_failure();
        assert!(ar.rate() < initial);
    }

    #[test]
    fn test_adaptive_rate_bounds() {
        let mut ar = AdaptiveRate::new(10.0, 5.0, 20.0);
        for _ in 0..100 {
            ar.on_success();
        }
        assert!(ar.rate() <= 20.0);
        for _ in 0..100 {
            ar.on_failure();
        }
        assert!(ar.rate() >= 5.0);
    }

    #[test]
    fn test_adaptive_rate_window() {
        let mut ar = AdaptiveRate::new(3.0, 1.0, 100.0);
        ar.next_window();
        assert!(ar.try_acquire());
        assert!(ar.try_acquire());
        assert!(ar.try_acquire());
        assert!(!ar.try_acquire()); // Budget exhausted.
        ar.next_window();
        assert!(ar.try_acquire()); // Fresh window.
    }

    #[test]
    fn test_backpressure_total_pushed() {
        let mut buf = BackpressureBuffer::new(2, Strategy::DropNewest);
        buf.push(1).unwrap();
        buf.push(2).unwrap();
        buf.push(3).unwrap();
        assert_eq!(buf.total_pushed(), 3);
    }
}
