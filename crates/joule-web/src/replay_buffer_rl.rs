//! Experience replay buffer — uniform sampling, prioritized replay, n-step returns, circular buffer.
//!
//! Replaces stable-baselines3 / RLlib replay buffers with pure Rust.
//! Supports fixed-capacity circular buffer, uniform random sampling, prioritized
//! experience replay (PER) with proportional priorities and importance-sampling
//! weights, n-step return computation, batch sampling, and buffer statistics.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum ReplayError {
    InvalidCapacity,
    InvalidParameter(String),
    EmptyBuffer,
    BatchTooLarge { requested: usize, available: usize },
}

impl fmt::Display for ReplayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCapacity => write!(f, "buffer capacity must be > 0"),
            Self::InvalidParameter(s) => write!(f, "invalid parameter: {s}"),
            Self::EmptyBuffer => write!(f, "buffer is empty"),
            Self::BatchTooLarge { requested, available } => {
                write!(f, "batch size {requested} exceeds available {available}")
            }
        }
    }
}

impl std::error::Error for ReplayError {}

// ── PRNG ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 1 } else { seed } }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() & 0x001F_FFFF_FFFF_FFFF) as f64 / (1u64 << 53) as f64
    }

    fn next_usize(&mut self, upper: usize) -> usize {
        if upper == 0 { return 0; }
        (self.next_u64() % upper as u64) as usize
    }
}

// ── Experience ──────────────────────────────────────────────────

/// A single experience transition.
#[derive(Debug, Clone)]
pub struct Experience {
    pub state: Vec<f64>,
    pub action: usize,
    pub reward: f64,
    pub next_state: Vec<f64>,
    pub done: bool,
}

impl Experience {
    pub fn new(
        state: Vec<f64>,
        action: usize,
        reward: f64,
        next_state: Vec<f64>,
        done: bool,
    ) -> Self {
        Self { state, action, reward, next_state, done }
    }
}

impl fmt::Display for Experience {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Exp(a={}, r={:.3}, done={})",
            self.action, self.reward, self.done,
        )
    }
}

// ── Uniform Replay Buffer ───────────────────────────────────────

/// Fixed-capacity circular replay buffer with uniform sampling.
#[derive(Debug, Clone)]
pub struct UniformReplayBuffer {
    buffer: Vec<Experience>,
    capacity: usize,
    write_pos: usize,
    full: bool,
    total_added: u64,
    rng: Rng,
}

impl UniformReplayBuffer {
    pub fn new(capacity: usize) -> Result<Self, ReplayError> {
        if capacity == 0 {
            return Err(ReplayError::InvalidCapacity);
        }
        Ok(Self {
            buffer: Vec::with_capacity(capacity.min(1024)),
            capacity,
            write_pos: 0,
            full: false,
            total_added: 0,
            rng: Rng::new(42),
        })
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = Rng::new(seed);
        self
    }

    /// Add an experience to the buffer.
    pub fn push(&mut self, exp: Experience) {
        if self.buffer.len() < self.capacity {
            self.buffer.push(exp);
        } else {
            self.buffer[self.write_pos] = exp;
        }
        self.write_pos = (self.write_pos + 1) % self.capacity;
        if self.write_pos == 0 && self.buffer.len() == self.capacity {
            self.full = true;
        }
        self.total_added += 1;
    }

    /// Number of experiences currently stored.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn is_full(&self) -> bool {
        self.full
    }

    pub fn total_added(&self) -> u64 {
        self.total_added
    }

    /// Sample a batch of experiences uniformly at random.
    pub fn sample(&mut self, batch_size: usize) -> Result<Vec<&Experience>, ReplayError> {
        if self.buffer.is_empty() {
            return Err(ReplayError::EmptyBuffer);
        }
        if batch_size > self.buffer.len() {
            return Err(ReplayError::BatchTooLarge {
                requested: batch_size,
                available: self.buffer.len(),
            });
        }
        let mut indices = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            indices.push(self.rng.next_usize(self.buffer.len()));
        }
        Ok(indices.iter().map(|i| &self.buffer[*i]).collect())
    }

    /// Sample a batch, returning cloned experiences.
    pub fn sample_cloned(&mut self, batch_size: usize) -> Result<Vec<Experience>, ReplayError> {
        if self.buffer.is_empty() {
            return Err(ReplayError::EmptyBuffer);
        }
        if batch_size > self.buffer.len() {
            return Err(ReplayError::BatchTooLarge {
                requested: batch_size,
                available: self.buffer.len(),
            });
        }
        let mut batch = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            let idx = self.rng.next_usize(self.buffer.len());
            batch.push(self.buffer[idx].clone());
        }
        Ok(batch)
    }

    /// Clear all experiences.
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.write_pos = 0;
        self.full = false;
    }
}

impl fmt::Display for UniformReplayBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "UniformReplay(size={}, capacity={}, total_added={})",
            self.buffer.len(),
            self.capacity,
            self.total_added,
        )
    }
}

// ── Prioritized Replay Buffer ───────────────────────────────────

/// A node for the sum-tree used in prioritized replay.
#[derive(Debug, Clone)]
struct SumTree {
    tree: Vec<f64>,
    capacity: usize,
}

impl SumTree {
    fn new(capacity: usize) -> Self {
        Self {
            tree: vec![0.0; 2 * capacity],
            capacity,
        }
    }

    fn update(&mut self, idx: usize, priority: f64) {
        let tree_idx = self.capacity + idx;
        let change = priority - self.tree[tree_idx];
        self.tree[tree_idx] = priority;
        let mut parent = tree_idx / 2;
        while parent >= 1 {
            self.tree[parent] += change;
            parent /= 2;
        }
    }

    fn total(&self) -> f64 {
        self.tree[1]
    }

    /// Find the leaf index for a given cumulative value.
    fn find(&self, mut value: f64) -> usize {
        let mut idx = 1;
        while idx < self.capacity {
            let left = 2 * idx;
            let right = left + 1;
            if left >= self.tree.len() {
                break;
            }
            if value <= self.tree[left] {
                idx = left;
            } else {
                value -= self.tree[left];
                idx = right;
            }
        }
        idx - self.capacity
    }

    fn get(&self, idx: usize) -> f64 {
        self.tree[self.capacity + idx]
    }
}

/// Prioritized experience replay buffer with proportional priorities.
#[derive(Debug, Clone)]
pub struct PrioritizedReplayBuffer {
    buffer: Vec<Option<Experience>>,
    tree: SumTree,
    capacity: usize,
    write_pos: usize,
    size: usize,
    alpha: f64,
    beta: f64,
    beta_increment: f64,
    epsilon: f64,
    max_priority: f64,
    total_added: u64,
    rng: Rng,
}

impl PrioritizedReplayBuffer {
    pub fn new(capacity: usize, alpha: f64, beta: f64) -> Result<Self, ReplayError> {
        if capacity == 0 {
            return Err(ReplayError::InvalidCapacity);
        }
        if alpha < 0.0 || alpha > 1.0 {
            return Err(ReplayError::InvalidParameter("alpha must be in [0, 1]".into()));
        }
        if beta < 0.0 || beta > 1.0 {
            return Err(ReplayError::InvalidParameter("beta must be in [0, 1]".into()));
        }
        // Round capacity up to power of 2 for sum tree
        let cap = capacity.next_power_of_two();
        Ok(Self {
            buffer: (0..cap).map(|_| None).collect(),
            tree: SumTree::new(cap),
            capacity: cap,
            write_pos: 0,
            size: 0,
            alpha,
            beta,
            beta_increment: 0.001,
            epsilon: 1e-6,
            max_priority: 1.0,
            total_added: 0,
            rng: Rng::new(42),
        })
    }

    pub fn with_beta_increment(mut self, inc: f64) -> Self {
        self.beta_increment = inc;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = Rng::new(seed);
        self
    }

    /// Add experience with maximum priority.
    pub fn push(&mut self, exp: Experience) {
        let priority = self.max_priority.powf(self.alpha);
        self.buffer[self.write_pos] = Some(exp);
        self.tree.update(self.write_pos, priority);
        self.write_pos = (self.write_pos + 1) % self.capacity;
        if self.size < self.capacity {
            self.size += 1;
        }
        self.total_added += 1;
    }

    /// Sample a prioritized batch. Returns (experiences, indices, is-weights).
    pub fn sample(
        &mut self,
        batch_size: usize,
    ) -> Result<(Vec<Experience>, Vec<usize>, Vec<f64>), ReplayError> {
        if self.size == 0 {
            return Err(ReplayError::EmptyBuffer);
        }
        if batch_size > self.size {
            return Err(ReplayError::BatchTooLarge {
                requested: batch_size,
                available: self.size,
            });
        }

        let total = self.tree.total();
        let segment = total / batch_size as f64;

        // Anneal beta toward 1.0
        self.beta = (self.beta + self.beta_increment).min(1.0);

        let mut experiences = Vec::with_capacity(batch_size);
        let mut indices = Vec::with_capacity(batch_size);
        let mut is_weights = Vec::with_capacity(batch_size);

        let min_prob = if self.size > 0 {
            let mut min_p = f64::MAX;
            for i in 0..self.size {
                let p = self.tree.get(i);
                if p > 0.0 && p < min_p {
                    min_p = p;
                }
            }
            if min_p == f64::MAX { 1.0 / self.size as f64 } else { min_p / total }
        } else {
            1.0
        };
        let max_weight = (self.size as f64 * min_prob).powf(-self.beta);

        for i in 0..batch_size {
            let low = segment * i as f64;
            let high = segment * (i + 1) as f64;
            let value = low + self.rng.next_f64() * (high - low);
            let idx = self.tree.find(value).min(self.size - 1);

            if let Some(ref exp) = self.buffer[idx] {
                let prob = self.tree.get(idx) / total;
                let weight = (self.size as f64 * prob).powf(-self.beta) / max_weight;
                experiences.push(exp.clone());
                indices.push(idx);
                is_weights.push(weight);
            } else {
                // Fallback: use first valid entry
                for j in 0..self.size {
                    if self.buffer[j].is_some() {
                        experiences.push(self.buffer[j].as_ref().unwrap().clone());
                        indices.push(j);
                        is_weights.push(1.0);
                        break;
                    }
                }
            }
        }

        Ok((experiences, indices, is_weights))
    }

    /// Update priorities after learning.
    pub fn update_priorities(&mut self, indices: &[usize], td_errors: &[f64]) {
        for (&idx, &err) in indices.iter().zip(td_errors.iter()) {
            let priority = (err.abs() + self.epsilon).powf(self.alpha);
            self.tree.update(idx, priority);
            if priority > self.max_priority {
                self.max_priority = priority;
            }
        }
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn total_added(&self) -> u64 {
        self.total_added
    }

    pub fn beta(&self) -> f64 {
        self.beta
    }
}

impl fmt::Display for PrioritizedReplayBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PrioritizedReplay(size={}, cap={}, α={:.3}, β={:.3})",
            self.size, self.capacity, self.alpha, self.beta,
        )
    }
}

// ── N-Step Return Buffer ────────────────────────────────────────

/// N-step return buffer that accumulates transitions and computes n-step returns.
#[derive(Debug, Clone)]
pub struct NStepBuffer {
    buffer: Vec<Experience>,
    n_steps: usize,
    gamma: f64,
    pending: Vec<(Vec<f64>, usize, f64)>,
}

impl NStepBuffer {
    pub fn new(n_steps: usize, gamma: f64) -> Result<Self, ReplayError> {
        if n_steps == 0 {
            return Err(ReplayError::InvalidParameter("n_steps must be > 0".into()));
        }
        if gamma < 0.0 || gamma > 1.0 {
            return Err(ReplayError::InvalidParameter("gamma must be in [0, 1]".into()));
        }
        Ok(Self {
            buffer: Vec::new(),
            n_steps,
            gamma,
            pending: Vec::new(),
        })
    }

    /// Add a transition. When n steps are accumulated, an n-step experience is emitted.
    pub fn add(
        &mut self,
        state: Vec<f64>,
        action: usize,
        reward: f64,
        next_state: Vec<f64>,
        done: bool,
    ) -> Option<Experience> {
        self.pending.push((state, action, reward));

        if done || self.pending.len() == self.n_steps {
            let n = self.pending.len();
            let first_state = self.pending[0].0.clone();
            let first_action = self.pending[0].1;

            // Compute n-step return
            let mut g = 0.0;
            let mut gamma_power = 1.0;
            for entry in &self.pending {
                g += gamma_power * entry.2;
                gamma_power *= self.gamma;
            }

            let exp = Experience::new(first_state, first_action, g, next_state, done);
            self.buffer.push(exp.clone());
            self.pending.clear();
            Some(exp)
        } else {
            None
        }
    }

    /// Flush remaining pending transitions (called at episode end).
    pub fn flush(&mut self, final_next_state: Vec<f64>) -> Option<Experience> {
        if self.pending.is_empty() {
            return None;
        }
        let first_state = self.pending[0].0.clone();
        let first_action = self.pending[0].1;
        let mut g = 0.0;
        let mut gamma_power = 1.0;
        for entry in &self.pending {
            g += gamma_power * entry.2;
            gamma_power *= self.gamma;
        }
        let exp = Experience::new(first_state, first_action, g, final_next_state, true);
        self.buffer.push(exp.clone());
        self.pending.clear();
        Some(exp)
    }

    /// All accumulated n-step experiences.
    pub fn experiences(&self) -> &[Experience] {
        &self.buffer
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub fn n_steps(&self) -> usize {
        self.n_steps
    }

    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Clear all stored experiences.
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.pending.clear();
    }
}

impl fmt::Display for NStepBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "NStepBuffer(n={}, γ={:.3}, stored={}, pending={})",
            self.n_steps,
            self.gamma,
            self.buffer.len(),
            self.pending.len(),
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;
    fn approx(a: f64, b: f64) -> bool { (a - b).abs() < EPS }

    fn make_exp(action: usize, reward: f64, done: bool) -> Experience {
        Experience::new(vec![action as f64], action, reward, vec![(action + 1) as f64], done)
    }

    #[test]
    fn test_uniform_buffer_new() {
        let buf = UniformReplayBuffer::new(100).unwrap();
        assert_eq!(buf.capacity(), 100);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_uniform_buffer_zero_capacity() {
        assert!(UniformReplayBuffer::new(0).is_err());
    }

    #[test]
    fn test_uniform_push_and_len() {
        let mut buf = UniformReplayBuffer::new(10).unwrap();
        buf.push(make_exp(0, 1.0, false));
        buf.push(make_exp(1, 2.0, false));
        assert_eq!(buf.len(), 2);
    }

    #[test]
    fn test_uniform_circular_overwrite() {
        let mut buf = UniformReplayBuffer::new(3).unwrap();
        for i in 0..5 {
            buf.push(make_exp(i, i as f64, false));
        }
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.total_added(), 5);
    }

    #[test]
    fn test_uniform_sample() {
        let mut buf = UniformReplayBuffer::new(10).unwrap().with_seed(42);
        for i in 0..10 {
            buf.push(make_exp(i, i as f64, false));
        }
        let batch = buf.sample(5).unwrap();
        assert_eq!(batch.len(), 5);
    }

    #[test]
    fn test_uniform_sample_empty() {
        let mut buf = UniformReplayBuffer::new(10).unwrap();
        assert!(buf.sample(1).is_err());
    }

    #[test]
    fn test_uniform_sample_too_large() {
        let mut buf = UniformReplayBuffer::new(10).unwrap();
        buf.push(make_exp(0, 1.0, false));
        assert!(buf.sample(5).is_err());
    }

    #[test]
    fn test_uniform_clear() {
        let mut buf = UniformReplayBuffer::new(10).unwrap();
        buf.push(make_exp(0, 1.0, false));
        buf.clear();
        assert!(buf.is_empty());
    }

    #[test]
    fn test_prioritized_buffer_new() {
        let buf = PrioritizedReplayBuffer::new(64, 0.6, 0.4).unwrap();
        assert!(buf.is_empty());
    }

    #[test]
    fn test_prioritized_invalid() {
        assert!(PrioritizedReplayBuffer::new(0, 0.6, 0.4).is_err());
        assert!(PrioritizedReplayBuffer::new(64, 1.5, 0.4).is_err());
        assert!(PrioritizedReplayBuffer::new(64, 0.6, -0.1).is_err());
    }

    #[test]
    fn test_prioritized_push_and_sample() {
        let mut buf = PrioritizedReplayBuffer::new(16, 0.6, 0.4).unwrap().with_seed(42);
        for i in 0..10 {
            buf.push(make_exp(i % 3, i as f64, false));
        }
        assert_eq!(buf.len(), 10);
        let (exps, indices, weights) = buf.sample(4).unwrap();
        assert_eq!(exps.len(), 4);
        assert_eq!(indices.len(), 4);
        assert_eq!(weights.len(), 4);
        for &w in &weights {
            assert!(w > 0.0 && w <= 1.0 + 1e-6);
        }
    }

    #[test]
    fn test_prioritized_update_priorities() {
        let mut buf = PrioritizedReplayBuffer::new(16, 0.6, 0.4).unwrap().with_seed(42);
        for i in 0..8 {
            buf.push(make_exp(i % 2, i as f64, false));
        }
        let (_, indices, _) = buf.sample(4).unwrap();
        let td_errors: Vec<f64> = vec![0.1, 0.5, 2.0, 0.01];
        buf.update_priorities(&indices, &td_errors);
        // Should not crash; priorities updated internally
        assert_eq!(buf.len(), 8);
    }

    #[test]
    fn test_prioritized_beta_annealing() {
        let mut buf = PrioritizedReplayBuffer::new(16, 0.6, 0.4)
            .unwrap()
            .with_beta_increment(0.1)
            .with_seed(42);
        for i in 0..8 {
            buf.push(make_exp(i, 1.0, false));
        }
        let initial_beta = buf.beta();
        let _ = buf.sample(2);
        assert!(buf.beta() > initial_beta);
    }

    #[test]
    fn test_nstep_buffer_creation() {
        let buf = NStepBuffer::new(3, 0.99).unwrap();
        assert_eq!(buf.n_steps(), 3);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_nstep_invalid() {
        assert!(NStepBuffer::new(0, 0.99).is_err());
        assert!(NStepBuffer::new(3, 1.5).is_err());
    }

    #[test]
    fn test_nstep_accumulation() {
        let mut buf = NStepBuffer::new(3, 0.9).unwrap();
        let r1 = buf.add(vec![0.0], 0, 1.0, vec![1.0], false);
        assert!(r1.is_none());
        let r2 = buf.add(vec![1.0], 1, 2.0, vec![2.0], false);
        assert!(r2.is_none());
        let r3 = buf.add(vec![2.0], 2, 3.0, vec![3.0], false);
        assert!(r3.is_some());
        let exp = r3.unwrap();
        // n-step return: 1.0 + 0.9*2.0 + 0.81*3.0 = 1 + 1.8 + 2.43 = 5.23
        assert!(approx(exp.reward, 5.23));
    }

    #[test]
    fn test_nstep_done_flushes() {
        let mut buf = NStepBuffer::new(5, 0.9).unwrap();
        let r = buf.add(vec![0.0], 0, 10.0, vec![1.0], true);
        assert!(r.is_some());
        assert!(approx(r.unwrap().reward, 10.0));
    }

    #[test]
    fn test_nstep_flush_pending() {
        let mut buf = NStepBuffer::new(5, 1.0).unwrap();
        buf.add(vec![0.0], 0, 1.0, vec![1.0], false);
        buf.add(vec![1.0], 1, 2.0, vec![2.0], false);
        let flushed = buf.flush(vec![99.0]);
        assert!(flushed.is_some());
        assert!(approx(flushed.unwrap().reward, 3.0));
        assert_eq!(buf.pending_count(), 0);
    }

    #[test]
    fn test_display_uniform() {
        let buf = UniformReplayBuffer::new(100).unwrap();
        let s = format!("{buf}");
        assert!(s.contains("UniformReplay"));
    }

    #[test]
    fn test_display_prioritized() {
        let buf = PrioritizedReplayBuffer::new(64, 0.6, 0.4).unwrap();
        let s = format!("{buf}");
        assert!(s.contains("PrioritizedReplay"));
    }

    #[test]
    fn test_display_nstep() {
        let buf = NStepBuffer::new(3, 0.99).unwrap();
        let s = format!("{buf}");
        assert!(s.contains("NStepBuffer"));
    }
}
