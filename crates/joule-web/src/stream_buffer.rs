//! Stream buffering — windowed buffers (tumbling/sliding/session), watermark
//! tracking, late event handling, flush policies, ordered output, statistics.
//!
//! Replaces JS stream processing libraries (RxJS windowTime, Highland.js) with
//! a pure-Rust stream buffer engine supporting tumbling, sliding, and session
//! windows with configurable flush policies and watermark-based late event handling.

use std::collections::VecDeque;

// ── Errors ─────────────────────────────────────────────────────

/// Stream buffer domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamBufferError {
    /// Buffer is closed and cannot accept events.
    BufferClosed,
    /// Event timestamp is before the watermark (late event).
    LateEvent { event_ts: u64, watermark: u64 },
    /// Buffer overflow.
    BufferOverflow { capacity: usize },
    /// Invalid window config.
    InvalidConfig(String),
}

impl std::fmt::Display for StreamBufferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BufferClosed => write!(f, "buffer is closed"),
            Self::LateEvent { event_ts, watermark } => {
                write!(f, "late event: ts={event_ts} < watermark={watermark}")
            }
            Self::BufferOverflow { capacity } => {
                write!(f, "buffer overflow: capacity {capacity}")
            }
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
        }
    }
}

impl std::error::Error for StreamBufferError {}

// ── Window Type ───────────────────────────────────────────────

/// Type of window for buffering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowType {
    /// Fixed-size, non-overlapping windows.
    Tumbling,
    /// Fixed-size, overlapping windows with a slide interval.
    Sliding,
    /// Dynamic windows that close after an inactivity gap.
    Session,
}

// ── Flush Policy ──────────────────────────────────────────────

/// When to flush the buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlushPolicy {
    /// Flush after N events.
    Count(usize),
    /// Flush after a time interval (ms).
    Time(u64),
    /// Flush after total payload size exceeds threshold.
    Size(usize),
}

// ── Late Event Policy ─────────────────────────────────────────

/// How to handle events that arrive after the watermark.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LateEventPolicy {
    /// Drop late events silently.
    Drop,
    /// Accept late events into a separate side output.
    SideOutput,
    /// Accept late events if within an allowed lateness window.
    AllowedLateness(u64),
}

impl Default for LateEventPolicy {
    fn default() -> Self {
        Self::Drop
    }
}

// ── Timestamped Event ─────────────────────────────────────────

/// An event with an associated timestamp.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimestampedEvent {
    pub timestamp_ms: u64,
    pub payload: String,
    pub key: Option<String>,
}

impl TimestampedEvent {
    pub fn new(timestamp_ms: u64, payload: impl Into<String>) -> Self {
        Self {
            timestamp_ms,
            payload: payload.into(),
            key: None,
        }
    }

    pub fn with_key(mut self, key: impl Into<String>) -> Self {
        self.key = Some(key.into());
        self
    }
}

// ── Window ────────────────────────────────────────────────────

/// A window of events.
#[derive(Debug, Clone)]
pub struct Window {
    pub id: u64,
    pub start_ms: u64,
    pub end_ms: u64,
    pub events: Vec<TimestampedEvent>,
    pub closed: bool,
}

impl Window {
    fn new(id: u64, start_ms: u64, end_ms: u64) -> Self {
        Self {
            id,
            start_ms,
            end_ms,
            events: Vec::new(),
            closed: false,
        }
    }

    fn contains(&self, ts: u64) -> bool {
        ts >= self.start_ms && ts < self.end_ms
    }

    fn event_count(&self) -> usize {
        self.events.len()
    }
}

// ── Buffer Stats ──────────────────────────────────────────────

/// Statistics for the stream buffer.
#[derive(Debug, Clone, Default)]
pub struct BufferStats {
    pub total_events: u64,
    pub total_flushed_windows: u64,
    pub total_late_events: u64,
    pub total_dropped: u64,
    pub total_side_output: u64,
    pub current_buffer_size: usize,
    pub max_buffer_size: usize,
}

// ── Stream Buffer ─────────────────────────────────────────────

/// Stream buffer with windowing, watermarks, and flush policies.
#[derive(Debug)]
pub struct StreamBuffer {
    window_type: WindowType,
    window_duration_ms: u64,
    slide_interval_ms: u64,
    session_gap_ms: u64,
    flush_policy: Option<FlushPolicy>,
    late_policy: LateEventPolicy,
    watermark: u64,
    windows: Vec<Window>,
    flushed: Vec<Window>,
    side_output: Vec<TimestampedEvent>,
    next_window_id: u64,
    closed: bool,
    capacity: Option<usize>,
    stats: BufferStats,
    /// Pending events (not yet assigned to windows for sliding).
    pending: VecDeque<TimestampedEvent>,
}

impl StreamBuffer {
    /// Create a tumbling window buffer.
    pub fn tumbling(window_duration_ms: u64) -> Self {
        Self {
            window_type: WindowType::Tumbling,
            window_duration_ms,
            slide_interval_ms: window_duration_ms,
            session_gap_ms: 0,
            flush_policy: None,
            late_policy: LateEventPolicy::default(),
            watermark: 0,
            windows: Vec::new(),
            flushed: Vec::new(),
            side_output: Vec::new(),
            next_window_id: 1,
            closed: false,
            capacity: None,
            stats: BufferStats::default(),
            pending: VecDeque::new(),
        }
    }

    /// Create a sliding window buffer.
    pub fn sliding(window_duration_ms: u64, slide_interval_ms: u64) -> Self {
        Self {
            window_type: WindowType::Sliding,
            window_duration_ms,
            slide_interval_ms,
            session_gap_ms: 0,
            flush_policy: None,
            late_policy: LateEventPolicy::default(),
            watermark: 0,
            windows: Vec::new(),
            flushed: Vec::new(),
            side_output: Vec::new(),
            next_window_id: 1,
            closed: false,
            capacity: None,
            stats: BufferStats::default(),
            pending: VecDeque::new(),
        }
    }

    /// Create a session window buffer.
    pub fn session(gap_ms: u64) -> Self {
        Self {
            window_type: WindowType::Session,
            window_duration_ms: 0,
            slide_interval_ms: 0,
            session_gap_ms: gap_ms,
            flush_policy: None,
            late_policy: LateEventPolicy::default(),
            watermark: 0,
            windows: Vec::new(),
            flushed: Vec::new(),
            side_output: Vec::new(),
            next_window_id: 1,
            closed: false,
            capacity: None,
            stats: BufferStats::default(),
            pending: VecDeque::new(),
        }
    }

    pub fn with_flush_policy(mut self, policy: FlushPolicy) -> Self {
        self.flush_policy = Some(policy);
        self
    }

    pub fn with_late_policy(mut self, policy: LateEventPolicy) -> Self {
        self.late_policy = policy;
        self
    }

    pub fn with_capacity(mut self, cap: usize) -> Self {
        self.capacity = Some(cap);
        self
    }

    /// Update the watermark (event time progress indicator).
    pub fn advance_watermark(&mut self, watermark_ms: u64) {
        if watermark_ms > self.watermark {
            self.watermark = watermark_ms;
        }
        // Close windows that end at or before the watermark.
        for window in &mut self.windows {
            if !window.closed && window.end_ms <= self.watermark {
                window.closed = true;
            }
        }
    }

    pub fn watermark(&self) -> u64 {
        self.watermark
    }

    /// Close the buffer — no more events accepted.
    pub fn close(&mut self) {
        self.closed = true;
        for window in &mut self.windows {
            window.closed = true;
        }
    }

    pub fn is_closed(&self) -> bool {
        self.closed
    }

    // ── Ingest ────────────────────────────────────────────────

    /// Ingest an event into the buffer.
    pub fn ingest(&mut self, event: TimestampedEvent) -> Result<(), StreamBufferError> {
        if self.closed {
            return Err(StreamBufferError::BufferClosed);
        }

        if let Some(cap) = self.capacity {
            let total_buffered: usize = self.windows.iter().map(|w| w.event_count()).sum();
            if total_buffered >= cap {
                self.stats.total_dropped += 1;
                return Err(StreamBufferError::BufferOverflow { capacity: cap });
            }
        }

        // Check late event.
        if event.timestamp_ms < self.watermark {
            match self.late_policy {
                LateEventPolicy::Drop => {
                    self.stats.total_late_events += 1;
                    self.stats.total_dropped += 1;
                    return Err(StreamBufferError::LateEvent {
                        event_ts: event.timestamp_ms,
                        watermark: self.watermark,
                    });
                }
                LateEventPolicy::SideOutput => {
                    self.stats.total_late_events += 1;
                    self.stats.total_side_output += 1;
                    self.side_output.push(event);
                    return Ok(());
                }
                LateEventPolicy::AllowedLateness(allowed) => {
                    if self.watermark.saturating_sub(event.timestamp_ms) > allowed {
                        self.stats.total_late_events += 1;
                        self.stats.total_dropped += 1;
                        return Err(StreamBufferError::LateEvent {
                            event_ts: event.timestamp_ms,
                            watermark: self.watermark,
                        });
                    }
                    // Within allowed lateness — continue to assign.
                }
            }
        }

        self.stats.total_events += 1;

        match self.window_type {
            WindowType::Tumbling => self.ingest_tumbling(event),
            WindowType::Sliding => self.ingest_sliding(event),
            WindowType::Session => self.ingest_session(event),
        }

        self.check_flush_policy();
        self.update_buffer_stats();
        Ok(())
    }

    fn ingest_tumbling(&mut self, event: TimestampedEvent) {
        let ts = event.timestamp_ms;
        // Find or create the window for this timestamp.
        let window_start = (ts / self.window_duration_ms) * self.window_duration_ms;
        let window_end = window_start + self.window_duration_ms;

        let existing = self
            .windows
            .iter_mut()
            .find(|w| w.start_ms == window_start && !w.closed);

        if let Some(window) = existing {
            window.events.push(event);
        } else {
            let id = self.next_window_id;
            self.next_window_id += 1;
            let mut window = Window::new(id, window_start, window_end);
            window.events.push(event);
            self.windows.push(window);
        }
    }

    fn ingest_sliding(&mut self, event: TimestampedEvent) {
        let ts = event.timestamp_ms;
        // An event belongs to all windows whose [start, start + duration) contains ts.
        // Windows start at multiples of slide_interval.
        let earliest_start = ts.saturating_sub(self.window_duration_ms.saturating_sub(1));
        let first_window_start =
            (earliest_start / self.slide_interval_ms) * self.slide_interval_ms;
        let last_window_start = (ts / self.slide_interval_ms) * self.slide_interval_ms;

        let mut start = first_window_start;
        while start <= last_window_start {
            let end = start + self.window_duration_ms;
            if ts >= start && ts < end {
                let existing = self
                    .windows
                    .iter_mut()
                    .find(|w| w.start_ms == start && !w.closed);
                if let Some(window) = existing {
                    window.events.push(event.clone());
                } else {
                    let id = self.next_window_id;
                    self.next_window_id += 1;
                    let mut window = Window::new(id, start, end);
                    window.events.push(event.clone());
                    self.windows.push(window);
                }
            }
            start += self.slide_interval_ms;
        }
    }

    fn ingest_session(&mut self, event: TimestampedEvent) {
        let ts = event.timestamp_ms;
        // Find a session window whose end + gap encompasses this event.
        let matching = self.windows.iter_mut().find(|w| {
            !w.closed && ts >= w.start_ms && ts <= w.end_ms + self.session_gap_ms
        });

        if let Some(window) = matching {
            window.events.push(event);
            // Extend the session window.
            if ts + self.session_gap_ms > window.end_ms {
                window.end_ms = ts + self.session_gap_ms;
            }
        } else {
            let id = self.next_window_id;
            self.next_window_id += 1;
            let mut window = Window::new(id, ts, ts + self.session_gap_ms);
            window.events.push(event);
            self.windows.push(window);
        }
    }

    // ── Flush ─────────────────────────────────────────────────

    fn check_flush_policy(&mut self) {
        if let Some(policy) = self.flush_policy {
            match policy {
                FlushPolicy::Count(n) => {
                    let to_flush: Vec<usize> = self
                        .windows
                        .iter()
                        .enumerate()
                        .filter(|(_, w)| !w.closed && w.event_count() >= n)
                        .map(|(i, _)| i)
                        .collect();
                    for &idx in to_flush.iter().rev() {
                        self.windows[idx].closed = true;
                    }
                }
                FlushPolicy::Time(interval_ms) => {
                    for window in &mut self.windows {
                        if !window.closed && !window.events.is_empty() {
                            let first_ts = window.events[0].timestamp_ms;
                            let last_ts = window.events.last().unwrap().timestamp_ms;
                            if last_ts.saturating_sub(first_ts) >= interval_ms {
                                window.closed = true;
                            }
                        }
                    }
                }
                FlushPolicy::Size(max_size) => {
                    for window in &mut self.windows {
                        if !window.closed {
                            let total: usize =
                                window.events.iter().map(|e| e.payload.len()).sum();
                            if total >= max_size {
                                window.closed = true;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Drain all closed windows, sorted by start time.
    pub fn drain_closed(&mut self) -> Vec<Window> {
        let mut closed = Vec::new();
        let mut remaining = Vec::new();
        for window in self.windows.drain(..) {
            if window.closed {
                closed.push(window);
            } else {
                remaining.push(window);
            }
        }
        self.windows = remaining;
        closed.sort_by_key(|w| w.start_ms);
        self.stats.total_flushed_windows += closed.len() as u64;
        self.flushed.extend(closed.clone());
        closed
    }

    /// Force-close and drain all windows.
    pub fn flush_all(&mut self) -> Vec<Window> {
        for window in &mut self.windows {
            window.closed = true;
        }
        self.drain_closed()
    }

    fn update_buffer_stats(&mut self) {
        let current: usize = self
            .windows
            .iter()
            .filter(|w| !w.closed)
            .map(|w| w.event_count())
            .sum();
        self.stats.current_buffer_size = current;
        if current > self.stats.max_buffer_size {
            self.stats.max_buffer_size = current;
        }
    }

    // ── Queries ──────────────────────────────────────────────

    /// Get current (open) windows.
    pub fn open_windows(&self) -> Vec<&Window> {
        self.windows.iter().filter(|w| !w.closed).collect()
    }

    /// Get all flushed windows.
    pub fn flushed_windows(&self) -> &[Window] {
        &self.flushed
    }

    /// Side output (late events).
    pub fn side_output(&self) -> &[TimestampedEvent] {
        &self.side_output
    }

    /// Buffer stats.
    pub fn stats(&self) -> &BufferStats {
        &self.stats
    }

    /// Total events in all open windows.
    pub fn buffered_count(&self) -> usize {
        self.windows
            .iter()
            .filter(|w| !w.closed)
            .map(|w| w.event_count())
            .sum()
    }

    /// Number of open windows.
    pub fn open_window_count(&self) -> usize {
        self.windows.iter().filter(|w| !w.closed).count()
    }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tumbling_window_basic() {
        let mut buf = StreamBuffer::tumbling(100);
        buf.ingest(TimestampedEvent::new(10, "a")).unwrap();
        buf.ingest(TimestampedEvent::new(50, "b")).unwrap();
        buf.ingest(TimestampedEvent::new(110, "c")).unwrap();
        // Two windows: [0,100) and [100,200).
        assert_eq!(buf.open_window_count(), 2);
    }

    #[test]
    fn test_tumbling_window_flush() {
        let mut buf = StreamBuffer::tumbling(100);
        buf.ingest(TimestampedEvent::new(10, "a")).unwrap();
        buf.ingest(TimestampedEvent::new(50, "b")).unwrap();
        // Advance watermark past window end.
        buf.advance_watermark(100);
        let closed = buf.drain_closed();
        assert_eq!(closed.len(), 1);
        assert_eq!(closed[0].events.len(), 2);
    }

    #[test]
    fn test_sliding_window() {
        let mut buf = StreamBuffer::sliding(100, 50);
        buf.ingest(TimestampedEvent::new(60, "a")).unwrap();
        // Event at 60 belongs to windows [0,100) and [50,150).
        assert_eq!(buf.open_window_count(), 2);
    }

    #[test]
    fn test_session_window() {
        let mut buf = StreamBuffer::session(100);
        buf.ingest(TimestampedEvent::new(10, "a")).unwrap();
        buf.ingest(TimestampedEvent::new(50, "b")).unwrap();
        // Both within gap — same session.
        assert_eq!(buf.open_window_count(), 1);
        assert_eq!(buf.open_windows()[0].events.len(), 2);
    }

    #[test]
    fn test_session_window_gap() {
        let mut buf = StreamBuffer::session(50);
        buf.ingest(TimestampedEvent::new(10, "a")).unwrap();
        buf.ingest(TimestampedEvent::new(200, "b")).unwrap();
        // Gap > 50ms — separate sessions.
        assert_eq!(buf.open_window_count(), 2);
    }

    #[test]
    fn test_watermark_late_event_drop() {
        let mut buf = StreamBuffer::tumbling(100);
        buf.advance_watermark(200);
        let result = buf.ingest(TimestampedEvent::new(50, "late"));
        assert!(matches!(result, Err(StreamBufferError::LateEvent { .. })));
        assert_eq!(buf.stats().total_late_events, 1);
    }

    #[test]
    fn test_late_event_side_output() {
        let mut buf = StreamBuffer::tumbling(100)
            .with_late_policy(LateEventPolicy::SideOutput);
        buf.advance_watermark(200);
        buf.ingest(TimestampedEvent::new(50, "late")).unwrap();
        assert_eq!(buf.side_output().len(), 1);
        assert_eq!(buf.stats().total_side_output, 1);
    }

    #[test]
    fn test_allowed_lateness() {
        let mut buf = StreamBuffer::tumbling(100)
            .with_late_policy(LateEventPolicy::AllowedLateness(50));
        buf.advance_watermark(200);
        // 170 is within allowed lateness of 50 (watermark 200 - 170 = 30 < 50).
        buf.ingest(TimestampedEvent::new(170, "ok")).unwrap();
        // 100 is beyond allowed lateness (200 - 100 = 100 > 50).
        let result = buf.ingest(TimestampedEvent::new(100, "too late"));
        assert!(matches!(result, Err(StreamBufferError::LateEvent { .. })));
    }

    #[test]
    fn test_flush_policy_count() {
        let mut buf = StreamBuffer::tumbling(1000)
            .with_flush_policy(FlushPolicy::Count(3));
        buf.ingest(TimestampedEvent::new(10, "a")).unwrap();
        buf.ingest(TimestampedEvent::new(20, "b")).unwrap();
        buf.ingest(TimestampedEvent::new(30, "c")).unwrap();
        let closed = buf.drain_closed();
        assert_eq!(closed.len(), 1);
        assert_eq!(closed[0].events.len(), 3);
    }

    #[test]
    fn test_flush_policy_size() {
        let mut buf = StreamBuffer::tumbling(1000)
            .with_flush_policy(FlushPolicy::Size(10));
        buf.ingest(TimestampedEvent::new(10, "abcde")).unwrap();
        buf.ingest(TimestampedEvent::new(20, "fghij")).unwrap();
        let closed = buf.drain_closed();
        assert_eq!(closed.len(), 1);
    }

    #[test]
    fn test_capacity_overflow() {
        let mut buf = StreamBuffer::tumbling(100).with_capacity(2);
        buf.ingest(TimestampedEvent::new(10, "a")).unwrap();
        buf.ingest(TimestampedEvent::new(20, "b")).unwrap();
        let result = buf.ingest(TimestampedEvent::new(30, "c"));
        assert!(matches!(result, Err(StreamBufferError::BufferOverflow { .. })));
    }

    #[test]
    fn test_close_buffer() {
        let mut buf = StreamBuffer::tumbling(100);
        buf.ingest(TimestampedEvent::new(10, "a")).unwrap();
        buf.close();
        let result = buf.ingest(TimestampedEvent::new(20, "b"));
        assert!(matches!(result, Err(StreamBufferError::BufferClosed)));
    }

    #[test]
    fn test_flush_all() {
        let mut buf = StreamBuffer::tumbling(100);
        buf.ingest(TimestampedEvent::new(10, "a")).unwrap();
        buf.ingest(TimestampedEvent::new(110, "b")).unwrap();
        let all = buf.flush_all();
        assert_eq!(all.len(), 2);
        assert_eq!(buf.open_window_count(), 0);
    }

    #[test]
    fn test_ordered_output() {
        let mut buf = StreamBuffer::tumbling(100);
        buf.ingest(TimestampedEvent::new(250, "c")).unwrap();
        buf.ingest(TimestampedEvent::new(50, "a")).unwrap();
        buf.ingest(TimestampedEvent::new(150, "b")).unwrap();
        let all = buf.flush_all();
        // Ordered by start time.
        assert!(all[0].start_ms <= all[1].start_ms);
        assert!(all[1].start_ms <= all[2].start_ms);
    }

    #[test]
    fn test_stats_tracking() {
        let mut buf = StreamBuffer::tumbling(100);
        buf.ingest(TimestampedEvent::new(10, "a")).unwrap();
        buf.ingest(TimestampedEvent::new(20, "b")).unwrap();
        assert_eq!(buf.stats().total_events, 2);
        buf.flush_all();
        buf.drain_closed();
    }

    #[test]
    fn test_buffered_count() {
        let mut buf = StreamBuffer::tumbling(100);
        buf.ingest(TimestampedEvent::new(10, "a")).unwrap();
        buf.ingest(TimestampedEvent::new(20, "b")).unwrap();
        assert_eq!(buf.buffered_count(), 2);
    }

    #[test]
    fn test_event_with_key() {
        let event = TimestampedEvent::new(100, "data").with_key("user-1");
        assert_eq!(event.key, Some("user-1".to_string()));
    }

    #[test]
    fn test_session_extends_window() {
        let mut buf = StreamBuffer::session(100);
        buf.ingest(TimestampedEvent::new(10, "a")).unwrap();
        buf.ingest(TimestampedEvent::new(80, "b")).unwrap();
        // Window should extend: end = 80 + 100 = 180.
        let windows = buf.open_windows();
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].end_ms, 180);
    }
}
