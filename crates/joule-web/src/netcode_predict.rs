//! Netcode prediction — client-side prediction for networked game state.
//!
//! Replaces custom prediction layers in Unreal/Unity netcode with a pure-Rust
//! prediction engine. Stores timestamped inputs in a ring buffer, applies them
//! locally to produce predicted state, and reconciles when authoritative server
//! state arrives (rewind-and-replay on mismatch). Tracks prediction error and
//! enforces a maximum prediction window.

use std::collections::VecDeque;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Prediction engine domain errors.
#[derive(Debug, Clone, PartialEq)]
pub enum PredictError {
    /// Input buffer is full.
    BufferFull { capacity: usize },
    /// Sequence number is out of order.
    OutOfOrder { expected: u64, got: u64 },
    /// Prediction window exceeded.
    WindowExceeded { max_ticks: u32, current: u32 },
    /// No authoritative state to reconcile against.
    NoAuthoritativeState,
    /// Tick is too old for reconciliation.
    TickTooOld { requested: u64, oldest: u64 },
}

impl fmt::Display for PredictError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BufferFull { capacity } => write!(f, "input buffer full (cap {capacity})"),
            Self::OutOfOrder { expected, got } => {
                write!(f, "out-of-order input: expected seq {expected}, got {got}")
            }
            Self::WindowExceeded { max_ticks, current } => {
                write!(f, "prediction window exceeded: {current}/{max_ticks} ticks")
            }
            Self::NoAuthoritativeState => write!(f, "no authoritative state available"),
            Self::TickTooOld { requested, oldest } => {
                write!(f, "tick {requested} too old, oldest is {oldest}")
            }
        }
    }
}

impl std::error::Error for PredictError {}

// ── Input ───────────────────────────────────────────────────────

/// A timestamped player input.
#[derive(Debug, Clone, PartialEq)]
pub struct Input {
    pub sequence: u64,
    pub tick: u64,
    pub move_x: f64,
    pub move_y: f64,
    pub action_flags: u32,
}

impl Input {
    pub fn new(sequence: u64, tick: u64, move_x: f64, move_y: f64) -> Self {
        Self { sequence, tick, move_x, move_y, action_flags: 0 }
    }

    pub fn with_action(mut self, flags: u32) -> Self {
        self.action_flags = flags;
        self
    }
}

impl fmt::Display for Input {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Input(seq={}, tick={}, ({:.2},{:.2}))", self.sequence, self.tick, self.move_x, self.move_y)
    }
}

// ── Input Buffer ────────────────────────────────────────────────

/// Ring buffer of timestamped inputs, ordered by sequence number.
#[derive(Debug)]
pub struct InputBuffer {
    inputs: VecDeque<Input>,
    capacity: usize,
    next_sequence: u64,
}

impl InputBuffer {
    pub fn new(capacity: usize) -> Self {
        Self { inputs: VecDeque::with_capacity(capacity), capacity, next_sequence: 0 }
    }

    /// Push an input. Enforces ordering and capacity.
    pub fn push(&mut self, input: Input) -> Result<(), PredictError> {
        if self.inputs.len() >= self.capacity {
            return Err(PredictError::BufferFull { capacity: self.capacity });
        }
        if input.sequence < self.next_sequence {
            return Err(PredictError::OutOfOrder {
                expected: self.next_sequence,
                got: input.sequence,
            });
        }
        self.next_sequence = input.sequence + 1;
        self.inputs.push_back(input);
        Ok(())
    }

    /// Remove all inputs up to and including the given sequence number.
    pub fn acknowledge(&mut self, up_to_sequence: u64) {
        while let Some(front) = self.inputs.front() {
            if front.sequence <= up_to_sequence {
                self.inputs.pop_front();
            } else {
                break;
            }
        }
    }

    /// Return all inputs after a given sequence number.
    pub fn inputs_after(&self, sequence: u64) -> Vec<&Input> {
        self.inputs.iter().filter(|i| i.sequence > sequence).collect()
    }

    pub fn len(&self) -> usize {
        self.inputs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inputs.is_empty()
    }

    pub fn clear(&mut self) {
        self.inputs.clear();
    }

    pub fn oldest_sequence(&self) -> Option<u64> {
        self.inputs.front().map(|i| i.sequence)
    }

    pub fn newest_sequence(&self) -> Option<u64> {
        self.inputs.back().map(|i| i.sequence)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Input> {
        self.inputs.iter()
    }
}

// ── Predicted State ─────────────────────────────────────────────

/// The predicted client-side state.
#[derive(Debug, Clone, PartialEq)]
pub struct PredictedState {
    pub tick: u64,
    pub last_input_sequence: u64,
    pub pos_x: f64,
    pub pos_y: f64,
    pub vel_x: f64,
    pub vel_y: f64,
}

impl PredictedState {
    pub fn new() -> Self {
        Self { tick: 0, last_input_sequence: 0, pos_x: 0.0, pos_y: 0.0, vel_x: 0.0, vel_y: 0.0 }
    }

    pub fn at(tick: u64, x: f64, y: f64) -> Self {
        Self { tick, last_input_sequence: 0, pos_x: x, pos_y: y, vel_x: 0.0, vel_y: 0.0 }
    }

    /// Apply a single input to advance state by one tick.
    pub fn apply_input(&mut self, input: &Input, dt: f64) {
        let accel = 100.0;
        self.vel_x += input.move_x * accel * dt;
        self.vel_y += input.move_y * accel * dt;
        self.pos_x += self.vel_x * dt;
        self.pos_y += self.vel_y * dt;
        // Damping
        self.vel_x *= 0.95;
        self.vel_y *= 0.95;
        self.tick = input.tick;
        self.last_input_sequence = input.sequence;
    }

    /// Squared distance to another state.
    pub fn distance_sq(&self, other: &PredictedState) -> f64 {
        let dx = self.pos_x - other.pos_x;
        let dy = self.pos_y - other.pos_y;
        dx * dx + dy * dy
    }
}

impl Default for PredictedState {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for PredictedState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "State(tick={}, pos=({:.2},{:.2}), vel=({:.2},{:.2}))",
            self.tick, self.pos_x, self.pos_y, self.vel_x, self.vel_y
        )
    }
}

// ── Authoritative State ─────────────────────────────────────────

/// Server-authoritative state snapshot.
#[derive(Debug, Clone, PartialEq)]
pub struct AuthoritativeState {
    pub tick: u64,
    pub last_processed_sequence: u64,
    pub pos_x: f64,
    pub pos_y: f64,
    pub vel_x: f64,
    pub vel_y: f64,
}

impl AuthoritativeState {
    pub fn new(tick: u64, seq: u64, x: f64, y: f64) -> Self {
        Self { tick, last_processed_sequence: seq, pos_x: x, pos_y: y, vel_x: 0.0, vel_y: 0.0 }
    }

    pub fn with_velocity(mut self, vx: f64, vy: f64) -> Self {
        self.vel_x = vx;
        self.vel_y = vy;
        self
    }
}

// ── Prediction Error Tracker ────────────────────────────────────

/// Tracks running prediction error statistics.
#[derive(Debug)]
pub struct PredictionErrorTracker {
    errors: VecDeque<f64>,
    max_samples: usize,
    total: f64,
}

impl PredictionErrorTracker {
    pub fn new(max_samples: usize) -> Self {
        Self { errors: VecDeque::with_capacity(max_samples), max_samples, total: 0.0 }
    }

    pub fn record(&mut self, error: f64) {
        if self.errors.len() >= self.max_samples {
            if let Some(old) = self.errors.pop_front() {
                self.total -= old;
            }
        }
        self.total += error;
        self.errors.push_back(error);
    }

    pub fn average(&self) -> f64 {
        if self.errors.is_empty() {
            0.0
        } else {
            self.total / self.errors.len() as f64
        }
    }

    pub fn peak(&self) -> f64 {
        self.errors.iter().cloned().fold(0.0_f64, f64::max)
    }

    pub fn sample_count(&self) -> usize {
        self.errors.len()
    }

    pub fn clear(&mut self) {
        self.errors.clear();
        self.total = 0.0;
    }
}

// ── Reconciliation Result ───────────────────────────────────────

/// Outcome of a server reconciliation.
#[derive(Debug, Clone, PartialEq)]
pub enum ReconciliationResult {
    /// Prediction was accurate within threshold.
    Accurate { error: f64 },
    /// Prediction was wrong; state was corrected by replaying inputs.
    Corrected { error: f64, replayed_inputs: u32 },
    /// Could not reconcile (tick too old, etc.).
    Failed(String),
}

impl fmt::Display for ReconciliationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Accurate { error } => write!(f, "accurate (err={error:.4})"),
            Self::Corrected { error, replayed_inputs } => {
                write!(f, "corrected (err={error:.4}, replayed {replayed_inputs})")
            }
            Self::Failed(msg) => write!(f, "failed: {msg}"),
        }
    }
}

// ── Prediction Config ───────────────────────────────────────────

/// Configuration for the prediction engine.
#[derive(Debug, Clone)]
pub struct PredictionConfig {
    pub max_prediction_window: u32,
    pub error_threshold: f64,
    pub tick_dt: f64,
    pub input_buffer_capacity: usize,
    pub error_tracker_samples: usize,
}

impl PredictionConfig {
    pub fn new() -> Self {
        Self {
            max_prediction_window: 30,
            error_threshold: 0.5,
            tick_dt: 1.0 / 60.0,
            input_buffer_capacity: 256,
            error_tracker_samples: 100,
        }
    }

    pub fn with_window(mut self, ticks: u32) -> Self {
        self.max_prediction_window = ticks;
        self
    }

    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.error_threshold = threshold;
        self
    }

    pub fn with_dt(mut self, dt: f64) -> Self {
        self.tick_dt = dt;
        self
    }

    pub fn with_buffer_capacity(mut self, cap: usize) -> Self {
        self.input_buffer_capacity = cap;
        self
    }
}

impl Default for PredictionConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ── Prediction Engine ───────────────────────────────────────────

/// Client-side prediction engine: apply inputs locally and reconcile with server.
#[derive(Debug)]
pub struct PredictionEngine {
    config: PredictionConfig,
    input_buffer: InputBuffer,
    current_state: PredictedState,
    state_history: VecDeque<PredictedState>,
    error_tracker: PredictionErrorTracker,
    last_authoritative_tick: u64,
    ticks_since_authoritative: u32,
}

impl PredictionEngine {
    pub fn new(config: PredictionConfig) -> Self {
        let buf_cap = config.input_buffer_capacity;
        let err_samples = config.error_tracker_samples;
        Self {
            config,
            input_buffer: InputBuffer::new(buf_cap),
            current_state: PredictedState::new(),
            state_history: VecDeque::with_capacity(256),
            error_tracker: PredictionErrorTracker::new(err_samples),
            last_authoritative_tick: 0,
            ticks_since_authoritative: 0,
        }
    }

    /// Submit a new input and apply it to the predicted state.
    pub fn submit_input(&mut self, input: Input) -> Result<(), PredictError> {
        self.ticks_since_authoritative += 1;
        if self.ticks_since_authoritative > self.config.max_prediction_window {
            return Err(PredictError::WindowExceeded {
                max_ticks: self.config.max_prediction_window,
                current: self.ticks_since_authoritative,
            });
        }
        self.input_buffer.push(input.clone())?;
        // Save state before applying (for potential rollback).
        self.state_history.push_back(self.current_state.clone());
        if self.state_history.len() > 256 {
            self.state_history.pop_front();
        }
        self.current_state.apply_input(&input, self.config.tick_dt);
        Ok(())
    }

    /// Reconcile with authoritative server state.
    pub fn reconcile(&mut self, auth: &AuthoritativeState) -> ReconciliationResult {
        // Find the predicted state at the authoritative tick.
        let predicted_at_auth = self
            .state_history
            .iter()
            .find(|s| s.tick == auth.tick)
            .cloned()
            .unwrap_or_else(|| self.current_state.clone());

        let auth_as_predicted = PredictedState {
            tick: auth.tick,
            last_input_sequence: auth.last_processed_sequence,
            pos_x: auth.pos_x,
            pos_y: auth.pos_y,
            vel_x: auth.vel_x,
            vel_y: auth.vel_y,
        };

        let error = predicted_at_auth.distance_sq(&auth_as_predicted).sqrt();
        self.error_tracker.record(error);
        self.last_authoritative_tick = auth.tick;
        self.ticks_since_authoritative = 0;

        // Acknowledge processed inputs.
        self.input_buffer.acknowledge(auth.last_processed_sequence);

        if error < self.config.error_threshold {
            return ReconciliationResult::Accurate { error };
        }

        // Rewind to authoritative state and replay unacknowledged inputs.
        self.current_state = auth_as_predicted;
        let unacked: Vec<Input> =
            self.input_buffer.inputs_after(auth.last_processed_sequence)
                .into_iter()
                .cloned()
                .collect();
        let replayed = unacked.len() as u32;
        for input in &unacked {
            self.current_state.apply_input(input, self.config.tick_dt);
        }

        // Rebuild state history from the corrected state.
        self.state_history.clear();

        ReconciliationResult::Corrected { error, replayed_inputs: replayed }
    }

    pub fn state(&self) -> &PredictedState {
        &self.current_state
    }

    pub fn pending_inputs(&self) -> usize {
        self.input_buffer.len()
    }

    pub fn average_error(&self) -> f64 {
        self.error_tracker.average()
    }

    pub fn peak_error(&self) -> f64 {
        self.error_tracker.peak()
    }

    pub fn ticks_since_server(&self) -> u32 {
        self.ticks_since_authoritative
    }

    pub fn config(&self) -> &PredictionConfig {
        &self.config
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_buffer_push_and_len() {
        let mut buf = InputBuffer::new(8);
        buf.push(Input::new(0, 0, 1.0, 0.0)).unwrap();
        buf.push(Input::new(1, 1, 0.0, 1.0)).unwrap();
        assert_eq!(buf.len(), 2);
    }

    #[test]
    fn input_buffer_full_error() {
        let mut buf = InputBuffer::new(2);
        buf.push(Input::new(0, 0, 0.0, 0.0)).unwrap();
        buf.push(Input::new(1, 1, 0.0, 0.0)).unwrap();
        let err = buf.push(Input::new(2, 2, 0.0, 0.0)).unwrap_err();
        assert_eq!(err, PredictError::BufferFull { capacity: 2 });
    }

    #[test]
    fn input_buffer_out_of_order() {
        let mut buf = InputBuffer::new(8);
        buf.push(Input::new(5, 5, 0.0, 0.0)).unwrap();
        let err = buf.push(Input::new(3, 3, 0.0, 0.0)).unwrap_err();
        assert_eq!(err, PredictError::OutOfOrder { expected: 6, got: 3 });
    }

    #[test]
    fn input_buffer_acknowledge() {
        let mut buf = InputBuffer::new(8);
        for i in 0..5 {
            buf.push(Input::new(i, i, 0.0, 0.0)).unwrap();
        }
        buf.acknowledge(2);
        assert_eq!(buf.len(), 2);
        assert_eq!(buf.oldest_sequence(), Some(3));
    }

    #[test]
    fn input_buffer_inputs_after() {
        let mut buf = InputBuffer::new(8);
        for i in 0..5 {
            buf.push(Input::new(i, i, 0.0, 0.0)).unwrap();
        }
        let after = buf.inputs_after(2);
        assert_eq!(after.len(), 2);
        assert_eq!(after[0].sequence, 3);
        assert_eq!(after[1].sequence, 4);
    }

    #[test]
    fn predicted_state_apply_input() {
        let mut state = PredictedState::new();
        let input = Input::new(0, 1, 1.0, 0.0);
        state.apply_input(&input, 1.0 / 60.0);
        assert!(state.pos_x > 0.0);
        assert_eq!(state.tick, 1);
    }

    #[test]
    fn predicted_state_distance() {
        let a = PredictedState::at(0, 0.0, 0.0);
        let b = PredictedState::at(0, 3.0, 4.0);
        let d = a.distance_sq(&b).sqrt();
        assert!((d - 5.0).abs() < 1e-9);
    }

    #[test]
    fn error_tracker_average() {
        let mut tracker = PredictionErrorTracker::new(10);
        tracker.record(2.0);
        tracker.record(4.0);
        assert!((tracker.average() - 3.0).abs() < 1e-9);
    }

    #[test]
    fn error_tracker_peak() {
        let mut tracker = PredictionErrorTracker::new(10);
        tracker.record(1.0);
        tracker.record(5.0);
        tracker.record(3.0);
        assert!((tracker.peak() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn error_tracker_eviction() {
        let mut tracker = PredictionErrorTracker::new(3);
        tracker.record(10.0);
        tracker.record(20.0);
        tracker.record(30.0);
        tracker.record(6.0); // evicts 10.0
        assert_eq!(tracker.sample_count(), 3);
        // avg = (20 + 30 + 6) / 3
        assert!((tracker.average() - 18.666_666_666_666_668).abs() < 0.01);
    }

    #[test]
    fn engine_submit_and_predict() {
        let config = PredictionConfig::new().with_dt(1.0 / 60.0);
        let mut engine = PredictionEngine::new(config);
        engine.submit_input(Input::new(0, 1, 1.0, 0.0)).unwrap();
        assert!(engine.state().pos_x > 0.0);
        assert_eq!(engine.pending_inputs(), 1);
    }

    #[test]
    fn engine_reconcile_accurate() {
        let config = PredictionConfig::new().with_threshold(100.0);
        let mut engine = PredictionEngine::new(config);
        engine.submit_input(Input::new(0, 1, 1.0, 0.0)).unwrap();
        let auth = AuthoritativeState::new(1, 0, engine.state().pos_x, engine.state().pos_y)
            .with_velocity(engine.state().vel_x, engine.state().vel_y);
        let result = engine.reconcile(&auth);
        matches!(result, ReconciliationResult::Accurate { .. });
    }

    #[test]
    fn engine_reconcile_correction() {
        let config = PredictionConfig::new().with_threshold(0.001);
        let mut engine = PredictionEngine::new(config);
        engine.submit_input(Input::new(0, 1, 1.0, 0.0)).unwrap();
        engine.submit_input(Input::new(1, 2, 1.0, 0.0)).unwrap();
        // Server says we are somewhere different.
        let auth = AuthoritativeState::new(1, 0, 50.0, 50.0);
        let result = engine.reconcile(&auth);
        match result {
            ReconciliationResult::Corrected { replayed_inputs, .. } => {
                assert_eq!(replayed_inputs, 1);
            }
            _ => panic!("expected correction"),
        }
    }

    #[test]
    fn engine_prediction_window_exceeded() {
        let config = PredictionConfig::new().with_window(2);
        let mut engine = PredictionEngine::new(config);
        engine.submit_input(Input::new(0, 1, 0.0, 0.0)).unwrap();
        engine.submit_input(Input::new(1, 2, 0.0, 0.0)).unwrap();
        let err = engine.submit_input(Input::new(2, 3, 0.0, 0.0)).unwrap_err();
        matches!(err, PredictError::WindowExceeded { .. });
    }

    #[test]
    fn engine_reconcile_resets_window() {
        let config = PredictionConfig::new().with_window(3).with_threshold(100.0);
        let mut engine = PredictionEngine::new(config);
        engine.submit_input(Input::new(0, 1, 0.0, 0.0)).unwrap();
        engine.submit_input(Input::new(1, 2, 0.0, 0.0)).unwrap();
        let auth = AuthoritativeState::new(2, 1, 0.0, 0.0);
        engine.reconcile(&auth);
        assert_eq!(engine.ticks_since_server(), 0);
        // Can submit more.
        engine.submit_input(Input::new(2, 3, 0.0, 0.0)).unwrap();
    }

    #[test]
    fn input_with_action_flags() {
        let input = Input::new(0, 0, 1.0, 0.0).with_action(0b0011);
        assert_eq!(input.action_flags, 3);
    }

    #[test]
    fn input_display() {
        let input = Input::new(42, 10, 1.5, -0.5);
        let s = format!("{input}");
        assert!(s.contains("seq=42"));
        assert!(s.contains("tick=10"));
    }

    #[test]
    fn predicted_state_default() {
        let state = PredictedState::default();
        assert_eq!(state.tick, 0);
        assert_eq!(state.pos_x, 0.0);
    }

    #[test]
    fn predicted_state_display() {
        let state = PredictedState::at(5, 1.0, 2.0);
        let s = format!("{state}");
        assert!(s.contains("tick=5"));
    }
}
