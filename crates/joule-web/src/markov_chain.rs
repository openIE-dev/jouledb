//! Markov chain — transition matrix, stationary distribution, classification.
//!
//! Replaces markovchain.js / Markov-chain / node-markov with pure Rust.
//! Supports transition matrix, state transitions, stationary distribution
//! (power iteration), absorption probabilities, mean first passage time,
//! chain classification (ergodic/absorbing), and text generation.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Domain errors for Markov chains.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MarkovError {
    /// State not found.
    StateNotFound(usize),
    /// Matrix dimensions invalid.
    InvalidDimension { expected: usize, got: usize },
    /// Row probabilities do not sum to 1.
    InvalidRowSum { row: usize, sum: String },
    /// Empty chain.
    EmptyChain,
    /// Power iteration did not converge.
    NotConverged { iterations: usize },
    /// Token not found in vocabulary.
    TokenNotFound(String),
}

impl fmt::Display for MarkovError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StateNotFound(s) => write!(f, "state not found: {s}"),
            Self::InvalidDimension { expected, got } => {
                write!(f, "invalid dimension: expected {expected}, got {got}")
            }
            Self::InvalidRowSum { row, sum } => {
                write!(f, "row {row} probabilities sum to {sum}, expected ~1.0")
            }
            Self::EmptyChain => write!(f, "chain is empty"),
            Self::NotConverged { iterations } => {
                write!(f, "power iteration did not converge after {iterations} iterations")
            }
            Self::TokenNotFound(t) => write!(f, "token not found: {t}"),
        }
    }
}

impl std::error::Error for MarkovError {}

// ── Simple PRNG ─────────────────────────────────────────────────

/// xorshift64 for deterministic sampling.
#[derive(Debug, Clone)]
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 1 } else { seed } }
    }

    fn next_f64(&mut self) -> f64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        (x >> 11) as f64 / ((1u64 << 53) as f64)
    }
}

// ── Transition Matrix ───────────────────────────────────────────

/// A Markov chain with N states and an NxN transition matrix.
#[derive(Debug, Clone)]
pub struct MarkovChain {
    n: usize,
    matrix: Vec<Vec<f64>>,
    state_names: Vec<String>,
    current_state: usize,
    rng: Rng,
    step_count: u64,
}

impl MarkovChain {
    /// Create from a transition matrix. Rows must sum to ~1.0.
    pub fn new(matrix: Vec<Vec<f64>>, seed: u64) -> Result<Self, MarkovError> {
        let n = matrix.len();
        if n == 0 {
            return Err(MarkovError::EmptyChain);
        }
        for (i, row) in matrix.iter().enumerate() {
            if row.len() != n {
                return Err(MarkovError::InvalidDimension { expected: n, got: row.len() });
            }
            let sum: f64 = row.iter().sum();
            if (sum - 1.0).abs() > 0.01 {
                return Err(MarkovError::InvalidRowSum {
                    row: i,
                    sum: format!("{sum:.6}"),
                });
            }
        }
        let names = (0..n).map(|i| format!("S{i}")).collect();
        Ok(Self {
            n,
            matrix,
            state_names: names,
            current_state: 0,
            rng: Rng::new(seed),
            step_count: 0,
        })
    }

    /// Set human-readable state names.
    pub fn with_names(mut self, names: Vec<String>) -> Result<Self, MarkovError> {
        if names.len() != self.n {
            return Err(MarkovError::InvalidDimension { expected: self.n, got: names.len() });
        }
        self.state_names = names;
        Ok(self)
    }

    /// Number of states.
    pub fn state_count(&self) -> usize { self.n }

    /// Current state index.
    pub fn current_state(&self) -> usize { self.current_state }

    /// Name of a state.
    pub fn state_name(&self, idx: usize) -> Option<&str> {
        self.state_names.get(idx).map(|s| s.as_str())
    }

    /// Steps taken so far.
    pub fn step_count(&self) -> u64 { self.step_count }

    /// Set the current state.
    pub fn set_state(&mut self, state: usize) -> Result<(), MarkovError> {
        if state >= self.n {
            return Err(MarkovError::StateNotFound(state));
        }
        self.current_state = state;
        Ok(())
    }

    /// Get the transition probability from state i to state j.
    pub fn transition_prob(&self, from: usize, to: usize) -> Result<f64, MarkovError> {
        if from >= self.n {
            return Err(MarkovError::StateNotFound(from));
        }
        if to >= self.n {
            return Err(MarkovError::StateNotFound(to));
        }
        Ok(self.matrix[from][to])
    }

    /// Take one step: choose next state based on current row probabilities.
    pub fn step(&mut self) -> usize {
        let row = &self.matrix[self.current_state];
        let r = self.rng.next_f64();
        let mut cumulative = 0.0;
        let mut next = self.n - 1;
        for (j, &p) in row.iter().enumerate() {
            cumulative += p;
            if r < cumulative {
                next = j;
                break;
            }
        }
        self.current_state = next;
        self.step_count += 1;
        next
    }

    /// Take n steps, returning the sequence of visited states.
    pub fn walk(&mut self, n: usize) -> Vec<usize> {
        let mut path = Vec::with_capacity(n + 1);
        path.push(self.current_state);
        for _ in 0..n {
            path.push(self.step());
        }
        path
    }

    /// Compute the k-step transition matrix (matrix^k).
    pub fn power_matrix(&self, k: u32) -> Vec<Vec<f64>> {
        let mut result = identity_matrix(self.n);
        let mut base = self.matrix.clone();
        let mut exp = k;
        while exp > 0 {
            if exp % 2 == 1 {
                result = mat_mul(&result, &base);
            }
            base = mat_mul(&base, &base);
            exp /= 2;
        }
        result
    }

    /// Compute the stationary distribution using power iteration.
    pub fn stationary_distribution(&self, max_iter: usize, tolerance: f64) -> Result<Vec<f64>, MarkovError> {
        if self.n == 0 {
            return Err(MarkovError::EmptyChain);
        }
        let mut pi = vec![1.0 / self.n as f64; self.n];
        for _ in 0..max_iter {
            let mut next = vec![0.0; self.n];
            for j in 0..self.n {
                for i in 0..self.n {
                    next[j] += pi[i] * self.matrix[i][j];
                }
            }
            let diff: f64 = pi.iter().zip(next.iter()).map(|(a, b)| (a - b).abs()).sum();
            pi = next;
            if diff < tolerance {
                return Ok(pi);
            }
        }
        Err(MarkovError::NotConverged { iterations: max_iter })
    }

    /// Classify an absorbing state (a state whose self-transition is 1.0).
    pub fn is_absorbing_state(&self, state: usize) -> bool {
        state < self.n && (self.matrix[state][state] - 1.0).abs() < 1e-10
    }

    /// Find all absorbing states.
    pub fn absorbing_states(&self) -> Vec<usize> {
        (0..self.n).filter(|s| self.is_absorbing_state(*s)).collect()
    }

    /// Check if the chain is absorbing (has at least one absorbing state
    /// and every non-absorbing state can reach an absorbing state).
    pub fn is_absorbing_chain(&self) -> bool {
        let absorbing = self.absorbing_states();
        if absorbing.is_empty() {
            return false;
        }
        let non_absorbing: Vec<usize> = (0..self.n)
            .filter(|s| !absorbing.contains(s))
            .collect();
        for &s in &non_absorbing {
            if !self.can_reach_any(s, &absorbing) {
                return false;
            }
        }
        true
    }

    /// Check if the chain is ergodic (irreducible and aperiodic).
    /// Approximated by checking if the power matrix at step n has all positive entries.
    pub fn is_ergodic(&self) -> bool {
        let power = self.power_matrix(self.n as u32 * 2);
        power.iter().all(|row| row.iter().all(|p| *p > 1e-10))
    }

    /// Check if state `from` can reach any state in `targets` via BFS.
    fn can_reach_any(&self, from: usize, targets: &[usize]) -> bool {
        let mut visited = vec![false; self.n];
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(from);
        visited[from] = true;
        while let Some(s) = queue.pop_front() {
            if targets.contains(&s) {
                return true;
            }
            for (j, &p) in self.matrix[s].iter().enumerate() {
                if p > 0.0 && !visited[j] {
                    visited[j] = true;
                    queue.push_back(j);
                }
            }
        }
        false
    }

    /// Mean first passage time from state i to state j using simulation.
    pub fn mean_first_passage_time(&mut self, from: usize, to: usize, trials: usize) -> Result<f64, MarkovError> {
        if from >= self.n || to >= self.n {
            return Err(MarkovError::StateNotFound(if from >= self.n { from } else { to }));
        }
        let mut total_steps = 0u64;
        let max_steps_per_trial = self.n as u64 * 1000;
        for _ in 0..trials {
            self.current_state = from;
            let mut steps = 0u64;
            while self.current_state != to && steps < max_steps_per_trial {
                self.step();
                steps += 1;
            }
            total_steps += steps;
        }
        Ok(total_steps as f64 / trials as f64)
    }

    /// The transition matrix.
    pub fn matrix(&self) -> &Vec<Vec<f64>> {
        &self.matrix
    }
}

// ── Matrix helpers ──────────────────────────────────────────────

fn identity_matrix(n: usize) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0.0; n]; n];
    for i in 0..n {
        m[i][i] = 1.0;
    }
    m
}

fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut c = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

// ── Text Markov Chain ───────────────────────────────────────────

/// A text generator using a first-order Markov chain over tokens.
#[derive(Debug, Clone)]
pub struct TextMarkov {
    transitions: HashMap<String, Vec<(String, u32)>>,
    rng: Rng,
}

impl TextMarkov {
    /// Create a new text Markov model.
    pub fn new(seed: u64) -> Self {
        Self {
            transitions: HashMap::new(),
            rng: Rng::new(seed),
        }
    }

    /// Train on a sequence of tokens.
    pub fn train(&mut self, tokens: &[&str]) {
        for window in tokens.windows(2) {
            let from = window[0].to_string();
            let to = window[1].to_string();
            let entry = self.transitions.entry(from).or_insert_with(Vec::new);
            if let Some(pair) = entry.iter_mut().find(|(t, _)| t == &to) {
                pair.1 += 1;
            } else {
                entry.push((to, 1));
            }
        }
    }

    /// Number of unique tokens seen as sources.
    pub fn vocabulary_size(&self) -> usize {
        self.transitions.len()
    }

    /// Generate text starting from a token, producing `n` tokens.
    pub fn generate(&mut self, start: &str, n: usize) -> Result<Vec<String>, MarkovError> {
        if !self.transitions.contains_key(start) {
            return Err(MarkovError::TokenNotFound(start.to_string()));
        }
        let mut result = Vec::with_capacity(n + 1);
        result.push(start.to_string());
        let mut current = start.to_string();
        for _ in 0..n {
            let candidates = match self.transitions.get(&current) {
                Some(c) => c,
                None => break,
            };
            let total: u32 = candidates.iter().map(|(_, c)| c).sum();
            if total == 0 {
                break;
            }
            let r = (self.rng.next_f64() * total as f64) as u32;
            let mut cumulative = 0u32;
            let mut chosen = candidates[0].0.clone();
            for (token, count) in candidates {
                cumulative += count;
                if r < cumulative {
                    chosen = token.clone();
                    break;
                }
            }
            result.push(chosen.clone());
            current = chosen;
        }
        Ok(result)
    }

    /// Get successors for a token.
    pub fn successors(&self, token: &str) -> Option<&Vec<(String, u32)>> {
        self.transitions.get(token)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn two_state_chain(seed: u64) -> MarkovChain {
        MarkovChain::new(vec![
            vec![0.7, 0.3],
            vec![0.4, 0.6],
        ], seed).unwrap()
    }

    #[test]
    fn test_creation() {
        let mc = two_state_chain(42);
        assert_eq!(mc.state_count(), 2);
        assert_eq!(mc.current_state(), 0);
    }

    #[test]
    fn test_empty_chain() {
        let result = MarkovChain::new(vec![], 42);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_row_length() {
        let result = MarkovChain::new(vec![
            vec![0.5, 0.5],
            vec![1.0],
        ], 42);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_row_sum() {
        let result = MarkovChain::new(vec![
            vec![0.5, 0.3],
            vec![0.4, 0.6],
        ], 42);
        assert!(result.is_err());
    }

    #[test]
    fn test_with_names() {
        let mc = two_state_chain(42)
            .with_names(vec!["Rain".into(), "Sun".into()]).unwrap();
        assert_eq!(mc.state_name(0), Some("Rain"));
        assert_eq!(mc.state_name(1), Some("Sun"));
    }

    #[test]
    fn test_names_wrong_count() {
        let result = two_state_chain(42)
            .with_names(vec!["A".into()]);
        assert!(result.is_err());
    }

    #[test]
    fn test_transition_prob() {
        let mc = two_state_chain(42);
        assert!((mc.transition_prob(0, 1).unwrap() - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_set_state() {
        let mut mc = two_state_chain(42);
        mc.set_state(1).unwrap();
        assert_eq!(mc.current_state(), 1);
    }

    #[test]
    fn test_set_invalid_state() {
        let mut mc = two_state_chain(42);
        assert!(mc.set_state(5).is_err());
    }

    #[test]
    fn test_step_stays_in_bounds() {
        let mut mc = two_state_chain(42);
        for _ in 0..100 {
            let s = mc.step();
            assert!(s < 2);
        }
    }

    #[test]
    fn test_step_count() {
        let mut mc = two_state_chain(42);
        mc.step();
        mc.step();
        mc.step();
        assert_eq!(mc.step_count(), 3);
    }

    #[test]
    fn test_walk_length() {
        let mut mc = two_state_chain(42);
        let path = mc.walk(10);
        assert_eq!(path.len(), 11); // start + 10 steps
    }

    #[test]
    fn test_power_matrix_identity() {
        let mc = two_state_chain(42);
        let p0 = mc.power_matrix(0);
        assert!((p0[0][0] - 1.0).abs() < 1e-10);
        assert!((p0[0][1] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_power_matrix_one() {
        let mc = two_state_chain(42);
        let p1 = mc.power_matrix(1);
        assert!((p1[0][0] - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_stationary_distribution() {
        let mc = two_state_chain(42);
        let pi = mc.stationary_distribution(1000, 1e-10).unwrap();
        // For [0.7, 0.3; 0.4, 0.6], stationary is [4/7, 3/7].
        assert!((pi[0] - 4.0 / 7.0).abs() < 1e-6);
        assert!((pi[1] - 3.0 / 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_absorbing_state() {
        let mc = MarkovChain::new(vec![
            vec![0.5, 0.5, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ], 42).unwrap();
        assert!(!mc.is_absorbing_state(0));
        assert!(mc.is_absorbing_state(1));
        assert!(mc.is_absorbing_state(2));
    }

    #[test]
    fn test_absorbing_chain() {
        let mc = MarkovChain::new(vec![
            vec![0.0, 0.5, 0.5],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ], 42).unwrap();
        assert!(mc.is_absorbing_chain());
    }

    #[test]
    fn test_ergodic_chain() {
        let mc = two_state_chain(42);
        assert!(mc.is_ergodic());
    }

    #[test]
    fn test_non_ergodic_chain() {
        // Two absorbing states — not ergodic.
        let mc = MarkovChain::new(vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ], 42).unwrap();
        assert!(!mc.is_ergodic());
    }

    #[test]
    fn test_mean_first_passage() {
        let mut mc = two_state_chain(42);
        let mfpt = mc.mean_first_passage_time(0, 1, 1000).unwrap();
        // Expected ~1/0.3 ≈ 3.33.
        assert!(mfpt > 1.0 && mfpt < 10.0, "MFPT was {mfpt}");
    }

    #[test]
    fn test_text_markov_train() {
        let mut tm = TextMarkov::new(42);
        tm.train(&["the", "cat", "sat", "the", "cat", "ran"]);
        assert!(tm.vocabulary_size() >= 3);
    }

    #[test]
    fn test_text_markov_generate() {
        let mut tm = TextMarkov::new(42);
        tm.train(&["the", "cat", "sat", "on", "the", "mat"]);
        let result = tm.generate("the", 3).unwrap();
        assert!(!result.is_empty());
        assert_eq!(result[0], "the");
    }

    #[test]
    fn test_text_markov_missing_token() {
        let mut tm = TextMarkov::new(42);
        tm.train(&["a", "b"]);
        assert!(tm.generate("z", 5).is_err());
    }

    #[test]
    fn test_text_markov_successors() {
        let mut tm = TextMarkov::new(42);
        tm.train(&["a", "b", "a", "c", "a", "b"]);
        let succ = tm.successors("a").unwrap();
        // "a" leads to "b" (twice) and "c" (once).
        assert!(succ.len() == 2);
    }
}
