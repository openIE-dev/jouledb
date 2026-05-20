//! Network flow control.
//!
//! Provides credit-based flow control with per-stream and per-connection
//! windows, window update messages, backpressure signaling, window
//! auto-tuning based on throughput, stall detection (zero window),
//! and configurable initial window sizes.

use std::collections::BTreeMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Flow control domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FlowControlError {
    /// Stream not found.
    StreamNotFound(u64),
    /// Duplicate stream ID.
    DuplicateStream(u64),
    /// Send window exhausted.
    SendWindowExhausted { stream_id: u64, available: u64 },
    /// Connection send window exhausted.
    ConnectionWindowExhausted { available: u64 },
    /// Receive buffer overflow.
    ReceiveBufferOverflow { stream_id: u64, capacity: u64 },
    /// Zero window stall detected.
    ZeroWindowStall(u64),
}

impl fmt::Display for FlowControlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StreamNotFound(id) => write!(f, "stream not found: {id}"),
            Self::DuplicateStream(id) => write!(f, "duplicate stream: {id}"),
            Self::SendWindowExhausted { stream_id, available } => {
                write!(f, "send window exhausted on stream {stream_id} (avail={available})")
            }
            Self::ConnectionWindowExhausted { available } => {
                write!(f, "connection send window exhausted (avail={available})")
            }
            Self::ReceiveBufferOverflow { stream_id, capacity } => {
                write!(f, "receive buffer overflow on stream {stream_id} (cap={capacity})")
            }
            Self::ZeroWindowStall(id) => write!(f, "zero window stall on stream {id}"),
        }
    }
}

impl std::error::Error for FlowControlError {}

// ── Window Update ───────────────────────────────────────────────

/// A window update message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WindowUpdate {
    pub stream_id: Option<u64>,
    pub increment: u64,
}

impl WindowUpdate {
    /// Create a stream-level window update.
    pub fn stream(stream_id: u64, increment: u64) -> Self {
        Self { stream_id: Some(stream_id), increment }
    }

    /// Create a connection-level window update.
    pub fn connection(increment: u64) -> Self {
        Self { stream_id: None, increment }
    }

    /// Whether this is a connection-level update.
    pub fn is_connection_level(&self) -> bool {
        self.stream_id.is_none()
    }
}

impl fmt::Display for WindowUpdate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.stream_id {
            Some(id) => write!(f, "WindowUpdate(stream={id}, +{} bytes)", self.increment),
            None => write!(f, "WindowUpdate(connection, +{} bytes)", self.increment),
        }
    }
}

// ── Backpressure Signal ─────────────────────────────────────────

/// Backpressure status for a stream or connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackpressureStatus {
    /// Normal flow — no backpressure.
    Normal,
    /// Approaching limit — slow down.
    Warning,
    /// Window exhausted — must stop.
    Blocked,
}

impl fmt::Display for BackpressureStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Normal => write!(f, "normal"),
            Self::Warning => write!(f, "warning"),
            Self::Blocked => write!(f, "blocked"),
        }
    }
}

// ── Flow Stats ──────────────────────────────────────────────────

/// Flow control statistics.
#[derive(Debug, Clone, Default)]
pub struct FlowStats {
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub window_updates_sent: u64,
    pub window_updates_received: u64,
    pub stalls: u64,
    pub auto_tunes: u64,
}

impl fmt::Display for FlowStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "sent={}B recv={}B updates_sent={} stalls={}",
            self.bytes_sent,
            self.bytes_received,
            self.window_updates_sent,
            self.stalls,
        )
    }
}

// ── Stream Window ───────────────────────────────────────────────

/// Flow control state for a single stream.
#[derive(Debug, Clone)]
struct StreamWindow {
    send_window: u64,
    recv_window: u64,
    initial_recv_window: u64,
    bytes_consumed: u64,
    bytes_buffered: u64,
    stalled: bool,
    throughput_bytes: u64,
    throughput_samples: u32,
}

impl StreamWindow {
    fn new(send_window: u64, recv_window: u64) -> Self {
        Self {
            send_window,
            recv_window,
            initial_recv_window: recv_window,
            bytes_consumed: 0,
            bytes_buffered: 0,
            stalled: false,
            throughput_bytes: 0,
            throughput_samples: 0,
        }
    }

    fn backpressure(&self) -> BackpressureStatus {
        if self.send_window == 0 {
            BackpressureStatus::Blocked
        } else if self.send_window < self.initial_recv_window / 4 {
            BackpressureStatus::Warning
        } else {
            BackpressureStatus::Normal
        }
    }
}

// ── Flow Config ─────────────────────────────────────────────────

/// Configuration for the flow controller.
#[derive(Debug, Clone)]
pub struct FlowConfig {
    pub initial_stream_send_window: u64,
    pub initial_stream_recv_window: u64,
    pub connection_send_window: u64,
    pub connection_recv_window: u64,
    pub auto_tune: bool,
    pub auto_tune_threshold: u32,
    pub auto_tune_multiplier: f64,
    pub max_window: u64,
    pub warning_threshold_fraction: f64,
}

impl Default for FlowConfig {
    fn default() -> Self {
        Self {
            initial_stream_send_window: 65536,
            initial_stream_recv_window: 65536,
            connection_send_window: 1_048_576,
            connection_recv_window: 1_048_576,
            auto_tune: false,
            auto_tune_threshold: 10,
            auto_tune_multiplier: 1.5,
            max_window: 16_777_216,
            warning_threshold_fraction: 0.25,
        }
    }
}

impl FlowConfig {
    pub fn with_stream_windows(mut self, send: u64, recv: u64) -> Self {
        self.initial_stream_send_window = send;
        self.initial_stream_recv_window = recv;
        self
    }

    pub fn with_connection_windows(mut self, send: u64, recv: u64) -> Self {
        self.connection_send_window = send;
        self.connection_recv_window = recv;
        self
    }

    pub fn with_auto_tune(mut self, enable: bool) -> Self {
        self.auto_tune = enable;
        self
    }

    pub fn with_max_window(mut self, max: u64) -> Self {
        self.max_window = max;
        self
    }
}

// ── Flow Controller ─────────────────────────────────────────────

/// Network flow controller managing per-stream and per-connection windows.
pub struct FlowController {
    config: FlowConfig,
    streams: BTreeMap<u64, StreamWindow>,
    conn_send_window: u64,
    conn_recv_window: u64,
    conn_bytes_consumed: u64,
    pending_updates: Vec<WindowUpdate>,
    stats: FlowStats,
}

impl FlowController {
    pub fn new(config: FlowConfig) -> Self {
        let send = config.connection_send_window;
        let recv = config.connection_recv_window;
        Self {
            config,
            streams: BTreeMap::new(),
            conn_send_window: send,
            conn_recv_window: recv,
            conn_bytes_consumed: 0,
            pending_updates: Vec::new(),
            stats: FlowStats::default(),
        }
    }

    /// Register a new stream.
    pub fn register_stream(&mut self, stream_id: u64) -> Result<(), FlowControlError> {
        if self.streams.contains_key(&stream_id) {
            return Err(FlowControlError::DuplicateStream(stream_id));
        }
        let sw = StreamWindow::new(
            self.config.initial_stream_send_window,
            self.config.initial_stream_recv_window,
        );
        self.streams.insert(stream_id, sw);
        Ok(())
    }

    /// Unregister a stream.
    pub fn unregister_stream(&mut self, stream_id: u64) -> Result<(), FlowControlError> {
        self.streams.remove(&stream_id)
            .ok_or(FlowControlError::StreamNotFound(stream_id))?;
        Ok(())
    }

    /// Record data being sent on a stream. Decrements both stream and connection windows.
    pub fn on_send(&mut self, stream_id: u64, bytes: u64) -> Result<(), FlowControlError> {
        // Check connection window.
        if bytes > self.conn_send_window {
            return Err(FlowControlError::ConnectionWindowExhausted {
                available: self.conn_send_window,
            });
        }

        let sw = self.streams.get_mut(&stream_id)
            .ok_or(FlowControlError::StreamNotFound(stream_id))?;

        // Check stream window.
        if bytes > sw.send_window {
            return Err(FlowControlError::SendWindowExhausted {
                stream_id,
                available: sw.send_window,
            });
        }

        sw.send_window -= bytes;
        self.conn_send_window -= bytes;
        self.stats.bytes_sent += bytes;

        // Detect zero-window stall.
        if sw.send_window == 0 {
            sw.stalled = true;
            self.stats.stalls += 1;
        }

        Ok(())
    }

    /// Record data being received on a stream. Fills receive buffer.
    pub fn on_receive(&mut self, stream_id: u64, bytes: u64) -> Result<(), FlowControlError> {
        let sw = self.streams.get_mut(&stream_id)
            .ok_or(FlowControlError::StreamNotFound(stream_id))?;

        if bytes > sw.recv_window {
            return Err(FlowControlError::ReceiveBufferOverflow {
                stream_id,
                capacity: sw.recv_window,
            });
        }

        sw.recv_window -= bytes;
        sw.bytes_buffered += bytes;
        self.conn_recv_window = self.conn_recv_window.saturating_sub(bytes);
        self.stats.bytes_received += bytes;

        Ok(())
    }

    /// Consume buffered data and potentially generate window updates.
    pub fn consume(&mut self, stream_id: u64, bytes: u64) -> Result<Vec<WindowUpdate>, FlowControlError> {
        let config = &self.config;
        let sw = self.streams.get_mut(&stream_id)
            .ok_or(FlowControlError::StreamNotFound(stream_id))?;

        let actual = bytes.min(sw.bytes_buffered);
        sw.bytes_buffered -= actual;
        sw.bytes_consumed += actual;
        sw.throughput_bytes += actual;
        sw.throughput_samples += 1;

        let mut updates = Vec::new();

        // Reopen the receive window.
        sw.recv_window += actual;
        if actual > 0 {
            updates.push(WindowUpdate::stream(stream_id, actual));
        }

        // Connection-level window update.
        self.conn_recv_window += actual;
        if actual > 0 {
            updates.push(WindowUpdate::connection(actual));
        }

        // Auto-tune: if throughput is high, grow the window.
        if config.auto_tune && sw.throughput_samples >= config.auto_tune_threshold {
            let avg_throughput = sw.throughput_bytes / sw.throughput_samples as u64;
            if avg_throughput > sw.initial_recv_window / 2 {
                let new_window = ((sw.initial_recv_window as f64 * config.auto_tune_multiplier) as u64)
                    .min(config.max_window);
                if new_window > sw.initial_recv_window {
                    let increment = new_window - sw.initial_recv_window;
                    sw.recv_window += increment;
                    sw.initial_recv_window = new_window;
                    updates.push(WindowUpdate::stream(stream_id, increment));
                    self.stats.auto_tunes += 1;
                }
            }
            sw.throughput_bytes = 0;
            sw.throughput_samples = 0;
        }

        self.stats.window_updates_sent += updates.len() as u64;
        Ok(updates)
    }

    /// Apply a received window update (peer grants more send credit).
    pub fn apply_window_update(&mut self, update: &WindowUpdate) -> Result<(), FlowControlError> {
        self.stats.window_updates_received += 1;
        match update.stream_id {
            Some(stream_id) => {
                let sw = self.streams.get_mut(&stream_id)
                    .ok_or(FlowControlError::StreamNotFound(stream_id))?;
                sw.send_window = sw.send_window.saturating_add(update.increment);
                if sw.stalled && sw.send_window > 0 {
                    sw.stalled = false;
                }
                Ok(())
            }
            None => {
                self.conn_send_window = self.conn_send_window.saturating_add(update.increment);
                Ok(())
            }
        }
    }

    /// Get backpressure status for a stream.
    pub fn backpressure(&self, stream_id: u64) -> Result<BackpressureStatus, FlowControlError> {
        let sw = self.streams.get(&stream_id)
            .ok_or(FlowControlError::StreamNotFound(stream_id))?;
        Ok(sw.backpressure())
    }

    /// Check if a stream has a zero-window stall.
    pub fn is_stalled(&self, stream_id: u64) -> Result<bool, FlowControlError> {
        let sw = self.streams.get(&stream_id)
            .ok_or(FlowControlError::StreamNotFound(stream_id))?;
        Ok(sw.stalled)
    }

    /// Get stream send window.
    pub fn stream_send_window(&self, stream_id: u64) -> Result<u64, FlowControlError> {
        let sw = self.streams.get(&stream_id)
            .ok_or(FlowControlError::StreamNotFound(stream_id))?;
        Ok(sw.send_window)
    }

    /// Get stream receive window.
    pub fn stream_recv_window(&self, stream_id: u64) -> Result<u64, FlowControlError> {
        let sw = self.streams.get(&stream_id)
            .ok_or(FlowControlError::StreamNotFound(stream_id))?;
        Ok(sw.recv_window)
    }

    /// Connection-level send window.
    pub fn connection_send_window(&self) -> u64 {
        self.conn_send_window
    }

    /// Connection-level receive window.
    pub fn connection_recv_window(&self) -> u64 {
        self.conn_recv_window
    }

    /// Number of registered streams.
    pub fn stream_count(&self) -> usize {
        self.streams.len()
    }

    /// Get flow statistics.
    pub fn stats(&self) -> &FlowStats {
        &self.stats
    }
}

impl fmt::Display for FlowController {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FlowController(streams={}, conn_send={}, conn_recv={})",
            self.streams.len(),
            self.conn_send_window,
            self.conn_recv_window,
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_fc() -> FlowController {
        FlowController::new(FlowConfig::default())
    }

    #[test]
    fn register_and_unregister_stream() {
        let mut fc = default_fc();
        fc.register_stream(1).unwrap();
        assert_eq!(fc.stream_count(), 1);
        fc.unregister_stream(1).unwrap();
        assert_eq!(fc.stream_count(), 0);
    }

    #[test]
    fn duplicate_stream_rejected() {
        let mut fc = default_fc();
        fc.register_stream(1).unwrap();
        let err = fc.register_stream(1).unwrap_err();
        assert!(matches!(err, FlowControlError::DuplicateStream(1)));
    }

    #[test]
    fn send_decrements_windows() {
        let mut fc = default_fc();
        fc.register_stream(1).unwrap();
        let before_stream = fc.stream_send_window(1).unwrap();
        let before_conn = fc.connection_send_window();
        fc.on_send(1, 1000).unwrap();
        assert_eq!(fc.stream_send_window(1).unwrap(), before_stream - 1000);
        assert_eq!(fc.connection_send_window(), before_conn - 1000);
    }

    #[test]
    fn send_window_exhausted() {
        let config = FlowConfig::default().with_stream_windows(100, 65536);
        let mut fc = FlowController::new(config);
        fc.register_stream(1).unwrap();
        fc.on_send(1, 80).unwrap();
        let err = fc.on_send(1, 50).unwrap_err();
        assert!(matches!(err, FlowControlError::SendWindowExhausted { .. }));
    }

    #[test]
    fn connection_window_exhausted() {
        let config = FlowConfig::default().with_connection_windows(100, 1_048_576);
        let mut fc = FlowController::new(config);
        fc.register_stream(1).unwrap();
        let err = fc.on_send(1, 200).unwrap_err();
        assert!(matches!(err, FlowControlError::ConnectionWindowExhausted { .. }));
    }

    #[test]
    fn receive_decrements_recv_window() {
        let mut fc = default_fc();
        fc.register_stream(1).unwrap();
        let before = fc.stream_recv_window(1).unwrap();
        fc.on_receive(1, 500).unwrap();
        assert_eq!(fc.stream_recv_window(1).unwrap(), before - 500);
    }

    #[test]
    fn receive_buffer_overflow() {
        let config = FlowConfig::default().with_stream_windows(65536, 100);
        let mut fc = FlowController::new(config);
        fc.register_stream(1).unwrap();
        let err = fc.on_receive(1, 200).unwrap_err();
        assert!(matches!(err, FlowControlError::ReceiveBufferOverflow { .. }));
    }

    #[test]
    fn consume_generates_window_updates() {
        let mut fc = default_fc();
        fc.register_stream(1).unwrap();
        fc.on_receive(1, 1000).unwrap();
        let updates = fc.consume(1, 1000).unwrap();
        // Should have at least stream and connection updates.
        assert!(updates.len() >= 2);
        assert!(updates.iter().any(|u| u.stream_id == Some(1)));
        assert!(updates.iter().any(|u| u.is_connection_level()));
    }

    #[test]
    fn apply_window_update_stream() {
        let config = FlowConfig::default().with_stream_windows(100, 65536);
        let mut fc = FlowController::new(config);
        fc.register_stream(1).unwrap();
        fc.on_send(1, 80).unwrap();
        assert_eq!(fc.stream_send_window(1).unwrap(), 20);
        fc.apply_window_update(&WindowUpdate::stream(1, 50)).unwrap();
        assert_eq!(fc.stream_send_window(1).unwrap(), 70);
    }

    #[test]
    fn apply_window_update_connection() {
        let config = FlowConfig::default().with_connection_windows(1000, 1_048_576);
        let mut fc = FlowController::new(config);
        fc.register_stream(1).unwrap();
        fc.on_send(1, 500).unwrap();
        fc.apply_window_update(&WindowUpdate::connection(300)).unwrap();
        assert_eq!(fc.connection_send_window(), 800);
    }

    #[test]
    fn zero_window_stall_detection() {
        let config = FlowConfig::default().with_stream_windows(100, 65536);
        let mut fc = FlowController::new(config);
        fc.register_stream(1).unwrap();
        fc.on_send(1, 100).unwrap();
        assert!(fc.is_stalled(1).unwrap());
        assert_eq!(fc.backpressure(1).unwrap(), BackpressureStatus::Blocked);
    }

    #[test]
    fn stall_clears_on_window_update() {
        let config = FlowConfig::default().with_stream_windows(100, 65536);
        let mut fc = FlowController::new(config);
        fc.register_stream(1).unwrap();
        fc.on_send(1, 100).unwrap();
        assert!(fc.is_stalled(1).unwrap());
        fc.apply_window_update(&WindowUpdate::stream(1, 50)).unwrap();
        assert!(!fc.is_stalled(1).unwrap());
    }

    #[test]
    fn backpressure_warning() {
        let config = FlowConfig::default().with_stream_windows(1000, 65536);
        let mut fc = FlowController::new(config);
        fc.register_stream(1).unwrap();
        // Use most of the window.
        fc.on_send(1, 900).unwrap();
        // 100 remaining out of 1000 initial = 10% < 25% threshold.
        assert_eq!(fc.backpressure(1).unwrap(), BackpressureStatus::Warning);
    }

    #[test]
    fn backpressure_normal() {
        let mut fc = default_fc();
        fc.register_stream(1).unwrap();
        assert_eq!(fc.backpressure(1).unwrap(), BackpressureStatus::Normal);
    }

    #[test]
    fn auto_tune_grows_window() {
        let config = FlowConfig::default()
            .with_stream_windows(65536, 100)
            .with_auto_tune(true)
            .with_max_window(10_000);
        let mut fc = FlowController::new(config);
        fc.register_stream(1).unwrap();
        // Simulate high throughput: receive and consume repeatedly.
        for _ in 0..12 {
            fc.on_receive(1, 80).unwrap();
            let _ = fc.consume(1, 80).unwrap();
        }
        // Window should have grown.
        assert!(fc.stream_recv_window(1).unwrap() > 100);
    }

    #[test]
    fn stats_tracking() {
        let mut fc = default_fc();
        fc.register_stream(1).unwrap();
        fc.on_send(1, 500).unwrap();
        fc.on_receive(1, 300).unwrap();
        assert_eq!(fc.stats().bytes_sent, 500);
        assert_eq!(fc.stats().bytes_received, 300);
    }

    #[test]
    fn window_update_display() {
        let u = WindowUpdate::stream(5, 1024);
        let s = format!("{u}");
        assert!(s.contains("stream=5"));
        assert!(s.contains("1024"));
    }

    #[test]
    fn flow_controller_display() {
        let fc = default_fc();
        let s = format!("{fc}");
        assert!(s.contains("FlowController"));
    }

    #[test]
    fn config_builder() {
        let config = FlowConfig::default()
            .with_stream_windows(1000, 2000)
            .with_connection_windows(5000, 6000)
            .with_auto_tune(true)
            .with_max_window(100_000);
        assert_eq!(config.initial_stream_send_window, 1000);
        assert_eq!(config.initial_stream_recv_window, 2000);
        assert_eq!(config.connection_send_window, 5000);
        assert_eq!(config.connection_recv_window, 6000);
        assert!(config.auto_tune);
        assert_eq!(config.max_window, 100_000);
    }
}
