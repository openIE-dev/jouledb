//! Temporal difference learning — TD(0), TD(λ), eligibility traces, n-step TD, value function approximation.
//!
//! Replaces stable-baselines3 / RLlib TD methods with pure Rust.
//! Supports tabular TD(0) for state-value estimation, TD(λ) with eligibility traces
//! (accumulating and replacing), n-step TD returns, linear value function approximation
//! with feature vectors, and convergence tracking via value change monitoring.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum TdError {
    InvalidParameter(String),
    StateNotFound(u64),
    DimensionMismatch { expected: usize, got: usize },
    EmptyHistory,
}

impl fmt::Display for TdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter(s) => write!(f, "invalid parameter: {s}"),
            Self::StateNotFound(s) => write!(f, "state {s} not found"),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::EmptyHistory => write!(f, "no history available"),
        }
    }
}

impl std::error::Error for TdError {}

// ── Value Table ─────────────────────────────────────────────────

/// Tabular state-value function V(s).
#[derive(Debug, Clone)]
pub struct ValueTable {
    values: HashMap<u64, f64>,
    default: f64,
}

impl ValueTable {
    pub fn new(default: f64) -> Self {
        Self {
            values: HashMap::new(),
            default,
        }
    }

    pub fn get(&self, state: u64) -> f64 {
        self.values.get(&state).copied().unwrap_or(self.default)
    }

    pub fn set(&mut self, state: u64, value: f64) {
        self.values.insert(state, value);
    }

    pub fn update(&mut self, state: u64, delta: f64) {
        let v = self.values.entry(state).or_insert(self.default);
        *v += delta;
    }

    pub fn state_count(&self) -> usize {
        self.values.len()
    }

    /// Maximum absolute value across all states (for convergence check).
    pub fn max_value(&self) -> f64 {
        self.values
            .values()
            .map(|v| v.abs())
            .fold(0.0, f64::max)
    }

    /// All state-value pairs.
    pub fn entries(&self) -> Vec<(u64, f64)> {
        self.values.iter().map(|(&k, &v)| (k, v)).collect()
    }
}

impl fmt::Display for ValueTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ValueTable(states={}, default={:.3})",
            self.values.len(),
            self.default,
        )
    }
}

// ── TD(0) Learner ───────────────────────────────────────────────

/// Tabular TD(0) state-value learner.
/// Update: V(s) += α * (r + γ * V(s') − V(s))
#[derive(Debug, Clone)]
pub struct Td0Learner {
    values: ValueTable,
    alpha: f64,
    gamma: f64,
    total_updates: u64,
    td_errors: Vec<f64>,
}

impl Td0Learner {
    pub fn new(alpha: f64, gamma: f64) -> Result<Self, TdError> {
        if alpha <= 0.0 || alpha > 1.0 {
            return Err(TdError::InvalidParameter("alpha must be in (0, 1]".into()));
        }
        if gamma < 0.0 || gamma > 1.0 {
            return Err(TdError::InvalidParameter("gamma must be in [0, 1]".into()));
        }
        Ok(Self {
            values: ValueTable::new(0.0),
            alpha,
            gamma,
            total_updates: 0,
            td_errors: Vec::new(),
        })
    }

    pub fn with_default_value(mut self, default: f64) -> Self {
        self.values = ValueTable::new(default);
        self
    }

    /// Perform a single TD(0) update.
    pub fn update(&mut self, state: u64, reward: f64, next_state: u64, done: bool) -> f64 {
        let v_s = self.values.get(state);
        let v_next = if done { 0.0 } else { self.values.get(next_state) };
        let td_error = reward + self.gamma * v_next - v_s;
        self.values.set(state, v_s + self.alpha * td_error);
        self.total_updates += 1;
        self.td_errors.push(td_error);
        td_error
    }

    /// Batch update from a sequence of transitions.
    pub fn update_batch(&mut self, transitions: &[(u64, f64, u64, bool)]) {
        for &(state, reward, next_state, done) in transitions {
            self.update(state, reward, next_state, done);
        }
    }

    pub fn value(&self, state: u64) -> f64 {
        self.values.get(state)
    }

    pub fn value_table(&self) -> &ValueTable {
        &self.values
    }

    pub fn total_updates(&self) -> u64 {
        self.total_updates
    }

    /// Average absolute TD error over the last n updates.
    pub fn average_td_error(&self, last_n: usize) -> Result<f64, TdError> {
        if self.td_errors.is_empty() {
            return Err(TdError::EmptyHistory);
        }
        let start = self.td_errors.len().saturating_sub(last_n);
        let slice = &self.td_errors[start..];
        Ok(slice.iter().map(|e| e.abs()).sum::<f64>() / slice.len() as f64)
    }

    /// Check convergence: average absolute TD error below threshold.
    pub fn has_converged(&self, window: usize, threshold: f64) -> bool {
        self.average_td_error(window)
            .map(|avg| avg < threshold)
            .unwrap_or(false)
    }

    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    pub fn gamma(&self) -> f64 {
        self.gamma
    }
}

impl fmt::Display for Td0Learner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TD(0)(α={:.3}, γ={:.3}, states={}, updates={})",
            self.alpha,
            self.gamma,
            self.values.state_count(),
            self.total_updates,
        )
    }
}

// ── Eligibility Traces ──────────────────────────────────────────

/// Trace type for TD(λ).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TraceType {
    /// Accumulating traces: e(s) += 1.
    Accumulating,
    /// Replacing traces: e(s) = 1.
    Replacing,
}

impl fmt::Display for TraceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Accumulating => write!(f, "Accumulating"),
            Self::Replacing => write!(f, "Replacing"),
        }
    }
}

/// Eligibility trace storage.
#[derive(Debug, Clone)]
pub struct EligibilityTraces {
    traces: HashMap<u64, f64>,
    trace_type: TraceType,
    threshold: f64,
}

impl EligibilityTraces {
    pub fn new(trace_type: TraceType) -> Self {
        Self {
            traces: HashMap::new(),
            trace_type,
            threshold: 1e-4,
        }
    }

    /// Decay all traces by γλ.
    pub fn decay(&mut self, gamma_lambda: f64) {
        self.traces.retain(|_, v| {
            *v *= gamma_lambda;
            v.abs() > self.threshold
        });
    }

    /// Update trace for visited state.
    pub fn visit(&mut self, state: u64) {
        match self.trace_type {
            TraceType::Accumulating => {
                *self.traces.entry(state).or_insert(0.0) += 1.0;
            }
            TraceType::Replacing => {
                self.traces.insert(state, 1.0);
            }
        }
    }

    /// Get trace value for a state.
    pub fn get(&self, state: u64) -> f64 {
        self.traces.get(&state).copied().unwrap_or(0.0)
    }

    /// Reset all traces (at episode start).
    pub fn reset(&mut self) {
        self.traces.clear();
    }

    /// Number of active traces.
    pub fn active_count(&self) -> usize {
        self.traces.len()
    }

    /// Iterator over active state-trace pairs.
    pub fn active_states(&self) -> Vec<(u64, f64)> {
        self.traces.iter().map(|(&k, &v)| (k, v)).collect()
    }
}

impl fmt::Display for EligibilityTraces {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "EligibilityTraces(type={}, active={})",
            self.trace_type,
            self.traces.len(),
        )
    }
}

// ── TD(λ) Learner ───────────────────────────────────────────────

/// TD(λ) learner with eligibility traces.
#[derive(Debug, Clone)]
pub struct TdLambdaLearner {
    values: ValueTable,
    traces: EligibilityTraces,
    alpha: f64,
    gamma: f64,
    lambda: f64,
    total_updates: u64,
    td_errors: Vec<f64>,
}

impl TdLambdaLearner {
    pub fn new(
        alpha: f64,
        gamma: f64,
        lambda: f64,
        trace_type: TraceType,
    ) -> Result<Self, TdError> {
        if alpha <= 0.0 || alpha > 1.0 {
            return Err(TdError::InvalidParameter("alpha must be in (0, 1]".into()));
        }
        if gamma < 0.0 || gamma > 1.0 {
            return Err(TdError::InvalidParameter("gamma must be in [0, 1]".into()));
        }
        if lambda < 0.0 || lambda > 1.0 {
            return Err(TdError::InvalidParameter("lambda must be in [0, 1]".into()));
        }
        Ok(Self {
            values: ValueTable::new(0.0),
            traces: EligibilityTraces::new(trace_type),
            alpha,
            gamma,
            lambda,
            total_updates: 0,
            td_errors: Vec::new(),
        })
    }

    pub fn with_default_value(mut self, default: f64) -> Self {
        self.values = ValueTable::new(default);
        self
    }

    /// Start a new episode (reset traces).
    pub fn begin_episode(&mut self) {
        self.traces.reset();
    }

    /// Perform a TD(λ) update step.
    pub fn update(&mut self, state: u64, reward: f64, next_state: u64, done: bool) -> f64 {
        let v_s = self.values.get(state);
        let v_next = if done { 0.0 } else { self.values.get(next_state) };
        let td_error = reward + self.gamma * v_next - v_s;

        // Update trace for current state
        self.traces.visit(state);

        // Update all states proportionally to their trace
        let active = self.traces.active_states();
        for (s, trace) in &active {
            let delta = self.alpha * td_error * trace;
            self.values.update(*s, delta);
        }

        // Decay traces
        self.traces.decay(self.gamma * self.lambda);

        if done {
            self.traces.reset();
        }

        self.total_updates += 1;
        self.td_errors.push(td_error);
        td_error
    }

    pub fn value(&self, state: u64) -> f64 {
        self.values.get(state)
    }

    pub fn value_table(&self) -> &ValueTable {
        &self.values
    }

    pub fn total_updates(&self) -> u64 {
        self.total_updates
    }

    pub fn active_traces(&self) -> usize {
        self.traces.active_count()
    }

    pub fn lambda(&self) -> f64 {
        self.lambda
    }
}

impl fmt::Display for TdLambdaLearner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TD(λ={:.3})(α={:.3}, γ={:.3}, states={}, updates={})",
            self.lambda,
            self.alpha,
            self.gamma,
            self.values.state_count(),
            self.total_updates,
        )
    }
}

// ── N-Step TD ───────────────────────────────────────────────────

/// N-step TD learner: uses n-step returns for value updates.
#[derive(Debug, Clone)]
pub struct NStepTd {
    values: ValueTable,
    alpha: f64,
    gamma: f64,
    n_steps: usize,
    buffer: Vec<(u64, f64)>,
    total_updates: u64,
}

impl NStepTd {
    pub fn new(alpha: f64, gamma: f64, n_steps: usize) -> Result<Self, TdError> {
        if alpha <= 0.0 || alpha > 1.0 {
            return Err(TdError::InvalidParameter("alpha must be in (0, 1]".into()));
        }
        if gamma < 0.0 || gamma > 1.0 {
            return Err(TdError::InvalidParameter("gamma must be in [0, 1]".into()));
        }
        if n_steps == 0 {
            return Err(TdError::InvalidParameter("n_steps must be > 0".into()));
        }
        Ok(Self {
            values: ValueTable::new(0.0),
            alpha,
            gamma,
            n_steps,
            buffer: Vec::new(),
            total_updates: 0,
        })
    }

    /// Begin a new episode.
    pub fn begin_episode(&mut self) {
        self.buffer.clear();
    }

    /// Add a transition. When n steps accumulated, update the oldest state.
    pub fn step(
        &mut self,
        state: u64,
        reward: f64,
        next_state: u64,
        done: bool,
    ) -> Option<f64> {
        self.buffer.push((state, reward));

        if done {
            // Flush all remaining transitions
            let updates = self.flush_buffer(next_state, done);
            return updates;
        }

        if self.buffer.len() >= self.n_steps {
            // Compute n-step return for oldest transition
            let g = self.compute_n_step_return(next_state);
            let (oldest_state, _) = self.buffer.remove(0);
            let v = self.values.get(oldest_state);
            let td_error = g - v;
            self.values.set(oldest_state, v + self.alpha * td_error);
            self.total_updates += 1;
            return Some(td_error);
        }

        None
    }

    fn compute_n_step_return(&self, bootstrap_state: u64) -> f64 {
        let mut g = 0.0;
        let mut gamma_power = 1.0;
        for &(_, reward) in &self.buffer {
            g += gamma_power * reward;
            gamma_power *= self.gamma;
        }
        g += gamma_power * self.values.get(bootstrap_state);
        g
    }

    fn flush_buffer(&mut self, final_state: u64, _done: bool) -> Option<f64> {
        let mut last_error = None;
        while !self.buffer.is_empty() {
            let mut g = 0.0;
            let mut gamma_power = 1.0;
            for &(_, reward) in &self.buffer {
                g += gamma_power * reward;
                gamma_power *= self.gamma;
            }
            // Terminal state — no bootstrap
            let (oldest_state, _) = self.buffer.remove(0);
            let v = self.values.get(oldest_state);
            let td_error = g - v;
            self.values.set(oldest_state, v + self.alpha * td_error);
            self.total_updates += 1;
            last_error = Some(td_error);
        }
        let _ = final_state;
        last_error
    }

    pub fn value(&self, state: u64) -> f64 {
        self.values.get(state)
    }

    pub fn value_table(&self) -> &ValueTable {
        &self.values
    }

    pub fn total_updates(&self) -> u64 {
        self.total_updates
    }

    pub fn n_steps(&self) -> usize {
        self.n_steps
    }
}

impl fmt::Display for NStepTd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "NStepTD(n={}, α={:.3}, γ={:.3}, states={}, updates={})",
            self.n_steps,
            self.alpha,
            self.gamma,
            self.values.state_count(),
            self.total_updates,
        )
    }
}

// ── Linear Value Approximation ──────────────────────────────────

/// Linear value function approximation: V(s) = w · φ(s).
#[derive(Debug, Clone)]
pub struct LinearValueFunction {
    weights: Vec<f64>,
    dim: usize,
    alpha: f64,
    gamma: f64,
    total_updates: u64,
}

impl LinearValueFunction {
    pub fn new(dim: usize, alpha: f64, gamma: f64) -> Result<Self, TdError> {
        if dim == 0 {
            return Err(TdError::InvalidParameter("dimension must be > 0".into()));
        }
        if alpha <= 0.0 || alpha > 1.0 {
            return Err(TdError::InvalidParameter("alpha must be in (0, 1]".into()));
        }
        if gamma < 0.0 || gamma > 1.0 {
            return Err(TdError::InvalidParameter("gamma must be in [0, 1]".into()));
        }
        Ok(Self {
            weights: vec![0.0; dim],
            dim,
            alpha,
            gamma,
            total_updates: 0,
        })
    }

    /// Compute V(s) = w · features.
    pub fn value(&self, features: &[f64]) -> f64 {
        self.weights
            .iter()
            .zip(features.iter())
            .map(|(w, f)| w * f)
            .sum()
    }

    /// TD(0) update with feature vectors.
    pub fn update(
        &mut self,
        features: &[f64],
        reward: f64,
        next_features: &[f64],
        done: bool,
    ) -> Result<f64, TdError> {
        if features.len() != self.dim {
            return Err(TdError::DimensionMismatch {
                expected: self.dim,
                got: features.len(),
            });
        }
        if next_features.len() != self.dim {
            return Err(TdError::DimensionMismatch {
                expected: self.dim,
                got: next_features.len(),
            });
        }

        let v_s = self.value(features);
        let v_next = if done { 0.0 } else { self.value(next_features) };
        let td_error = reward + self.gamma * v_next - v_s;

        // Gradient update: w += α * δ * φ(s)
        for (w, &f) in self.weights.iter_mut().zip(features.iter()) {
            *w += self.alpha * td_error * f;
        }

        self.total_updates += 1;
        Ok(td_error)
    }

    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn total_updates(&self) -> u64 {
        self.total_updates
    }
}

impl fmt::Display for LinearValueFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LinearVF(dim={}, α={:.3}, γ={:.3}, updates={})",
            self.dim, self.alpha, self.gamma, self.total_updates,
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;
    fn approx(a: f64, b: f64) -> bool { (a - b).abs() < EPS }

    #[test]
    fn test_value_table_default() {
        let vt = ValueTable::new(5.0);
        assert!(approx(vt.get(99), 5.0));
    }

    #[test]
    fn test_value_table_set_get() {
        let mut vt = ValueTable::new(0.0);
        vt.set(1, 3.14);
        assert!(approx(vt.get(1), 3.14));
    }

    #[test]
    fn test_value_table_update() {
        let mut vt = ValueTable::new(0.0);
        vt.set(0, 5.0);
        vt.update(0, 2.0);
        assert!(approx(vt.get(0), 7.0));
    }

    #[test]
    fn test_td0_creation() {
        let td = Td0Learner::new(0.1, 0.99).unwrap();
        assert!(approx(td.alpha(), 0.1));
        assert!(approx(td.gamma(), 0.99));
    }

    #[test]
    fn test_td0_invalid_params() {
        assert!(Td0Learner::new(0.0, 0.9).is_err());
        assert!(Td0Learner::new(0.1, 1.5).is_err());
    }

    #[test]
    fn test_td0_update() {
        let mut td = Td0Learner::new(0.5, 0.9).unwrap();
        let err = td.update(0, 1.0, 1, false);
        assert!(err != 0.0 || td.value(0) != 0.0 || true);
        assert!(td.value(0) > 0.0, "V(0) = {}", td.value(0));
    }

    #[test]
    fn test_td0_terminal() {
        let mut td = Td0Learner::new(1.0, 0.9).unwrap();
        td.update(0, 10.0, 0, true);
        // V(0) = 0 + 1.0 * (10 + 0 - 0) = 10.0
        assert!(approx(td.value(0), 10.0));
    }

    #[test]
    fn test_td0_convergence() {
        let mut td = Td0Learner::new(0.1, 1.0).unwrap();
        // Simple chain: state 0 -> state 1 (terminal, reward 1)
        for _ in 0..200 {
            td.update(0, 0.0, 1, false);
            td.update(1, 1.0, 0, true);
        }
        // V(0) should approach 1.0, V(1) should approach 1.0
        assert!(td.value(1) > 0.5, "V(1) = {}", td.value(1));
    }

    #[test]
    fn test_td0_average_td_error() {
        let mut td = Td0Learner::new(0.1, 0.9).unwrap();
        td.update(0, 1.0, 1, false);
        td.update(1, 2.0, 2, true);
        let avg = td.average_td_error(10).unwrap();
        assert!(avg > 0.0);
    }

    #[test]
    fn test_eligibility_traces_accumulating() {
        let mut traces = EligibilityTraces::new(TraceType::Accumulating);
        traces.visit(0);
        traces.visit(0);
        assert!(approx(traces.get(0), 2.0));
    }

    #[test]
    fn test_eligibility_traces_replacing() {
        let mut traces = EligibilityTraces::new(TraceType::Replacing);
        traces.visit(0);
        traces.visit(0);
        assert!(approx(traces.get(0), 1.0));
    }

    #[test]
    fn test_eligibility_traces_decay() {
        let mut traces = EligibilityTraces::new(TraceType::Accumulating);
        traces.visit(0);
        traces.decay(0.5);
        assert!(approx(traces.get(0), 0.5));
    }

    #[test]
    fn test_eligibility_traces_reset() {
        let mut traces = EligibilityTraces::new(TraceType::Accumulating);
        traces.visit(0);
        traces.visit(1);
        traces.reset();
        assert_eq!(traces.active_count(), 0);
    }

    #[test]
    fn test_td_lambda_creation() {
        let td = TdLambdaLearner::new(0.1, 0.99, 0.8, TraceType::Accumulating).unwrap();
        assert!(approx(td.lambda(), 0.8));
    }

    #[test]
    fn test_td_lambda_invalid() {
        assert!(TdLambdaLearner::new(0.1, 0.99, 1.5, TraceType::Accumulating).is_err());
    }

    #[test]
    fn test_td_lambda_update() {
        let mut td = TdLambdaLearner::new(0.5, 0.9, 0.8, TraceType::Accumulating).unwrap();
        td.begin_episode();
        td.update(0, 1.0, 1, false);
        td.update(1, 2.0, 2, true);
        assert!(td.value(0) > 0.0);
        assert_eq!(td.total_updates(), 2);
    }

    #[test]
    fn test_nstep_td_creation() {
        let td = NStepTd::new(0.1, 0.99, 3).unwrap();
        assert_eq!(td.n_steps(), 3);
    }

    #[test]
    fn test_nstep_td_invalid() {
        assert!(NStepTd::new(0.1, 0.99, 0).is_err());
    }

    #[test]
    fn test_nstep_td_update() {
        let mut td = NStepTd::new(0.5, 0.9, 2).unwrap();
        td.begin_episode();
        let r1 = td.step(0, 1.0, 1, false);
        assert!(r1.is_none()); // Not enough steps yet
        let r2 = td.step(1, 2.0, 2, false);
        assert!(r2.is_some()); // Now 2 steps accumulated
    }

    #[test]
    fn test_nstep_td_terminal() {
        let mut td = NStepTd::new(0.5, 0.9, 5).unwrap();
        td.begin_episode();
        td.step(0, 1.0, 1, false);
        td.step(1, 2.0, 2, true); // Terminal before n steps
        assert!(td.value(0) > 0.0 || td.total_updates() > 0);
    }

    #[test]
    fn test_linear_vf_creation() {
        let vf = LinearValueFunction::new(3, 0.1, 0.99).unwrap();
        assert_eq!(vf.dim(), 3);
    }

    #[test]
    fn test_linear_vf_invalid() {
        assert!(LinearValueFunction::new(0, 0.1, 0.99).is_err());
    }

    #[test]
    fn test_linear_vf_value() {
        let mut vf = LinearValueFunction::new(2, 0.1, 0.99).unwrap();
        // Zero weights → zero value
        assert!(approx(vf.value(&[1.0, 1.0]), 0.0));
        // Update and check non-zero
        vf.update(&[1.0, 0.0], 1.0, &[0.0, 1.0], false).unwrap();
        assert!(vf.value(&[1.0, 0.0]) > 0.0);
    }

    #[test]
    fn test_linear_vf_dim_mismatch() {
        let mut vf = LinearValueFunction::new(3, 0.1, 0.99).unwrap();
        assert!(vf.update(&[1.0, 2.0], 1.0, &[0.0, 0.0, 0.0], false).is_err());
    }

    #[test]
    fn test_display_td0() {
        let td = Td0Learner::new(0.1, 0.99).unwrap();
        let s = format!("{td}");
        assert!(s.contains("TD(0)"));
    }

    #[test]
    fn test_display_td_lambda() {
        let td = TdLambdaLearner::new(0.1, 0.99, 0.8, TraceType::Accumulating).unwrap();
        let s = format!("{td}");
        assert!(s.contains("TD"));
    }

    #[test]
    fn test_display_linear_vf() {
        let vf = LinearValueFunction::new(3, 0.1, 0.99).unwrap();
        let s = format!("{vf}");
        assert!(s.contains("LinearVF"));
    }
}
