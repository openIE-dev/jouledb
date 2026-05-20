//! Multi-armed bandits — UCB1, Thompson sampling, epsilon-greedy, contextual bandits, regret tracking.
//!
//! Replaces Vowpal Wabbit / custom bandit implementations with pure Rust.
//! Supports UCB1 (upper confidence bound), Thompson sampling with Beta posteriors,
//! epsilon-greedy, decaying epsilon, contextual linear bandits (LinUCB),
//! cumulative and instantaneous regret tracking, arm statistics, and pull history.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum BanditError {
    InvalidParameter(String),
    NoArms,
    ArmOutOfRange { arm: usize, num_arms: usize },
    InsufficientData,
}

impl fmt::Display for BanditError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter(s) => write!(f, "invalid parameter: {s}"),
            Self::NoArms => write!(f, "no arms configured"),
            Self::ArmOutOfRange { arm, num_arms } => {
                write!(f, "arm {arm} out of range ({num_arms} arms)")
            }
            Self::InsufficientData => write!(f, "insufficient data for computation"),
        }
    }
}

impl std::error::Error for BanditError {}

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

    /// Sample from Beta(alpha, beta) using Johnk's algorithm.
    fn sample_beta(&mut self, alpha: f64, beta_param: f64) -> f64 {
        if alpha <= 0.0 || beta_param <= 0.0 {
            return 0.5;
        }
        // Simple beta sampling via gamma ratio
        let x = self.sample_gamma(alpha);
        let y = self.sample_gamma(beta_param);
        if x + y == 0.0 { 0.5 } else { x / (x + y) }
    }

    /// Sample from Gamma(shape, 1) using Marsaglia and Tsang's method.
    fn sample_gamma(&mut self, shape: f64) -> f64 {
        if shape < 1.0 {
            let u = self.next_f64();
            return self.sample_gamma(shape + 1.0) * u.powf(1.0 / shape);
        }
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let mut x: f64;
            let mut v: f64;
            loop {
                // Box-Muller normal
                let u1 = self.next_f64().max(1e-15);
                let u2 = self.next_f64();
                x = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                v = 1.0 + c * x;
                if v > 0.0 {
                    break;
                }
            }
            v = v * v * v;
            let u = self.next_f64();
            if u < 1.0 - 0.0331 * (x * x) * (x * x) {
                return d * v;
            }
            if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                return d * v;
            }
        }
    }
}

// ── Arm Statistics ──────────────────────────────────────────────

/// Statistics for a single bandit arm.
#[derive(Debug, Clone)]
pub struct ArmStats {
    pulls: u64,
    total_reward: f64,
    sum_sq: f64,
}

impl ArmStats {
    fn new() -> Self {
        Self { pulls: 0, total_reward: 0.0, sum_sq: 0.0 }
    }

    fn update(&mut self, reward: f64) {
        self.pulls += 1;
        self.total_reward += reward;
        self.sum_sq += reward * reward;
    }

    pub fn mean(&self) -> f64 {
        if self.pulls == 0 { 0.0 } else { self.total_reward / self.pulls as f64 }
    }

    pub fn variance(&self) -> f64 {
        if self.pulls < 2 {
            return 0.0;
        }
        let mean = self.mean();
        self.sum_sq / self.pulls as f64 - mean * mean
    }

    pub fn pulls(&self) -> u64 {
        self.pulls
    }

    pub fn total_reward(&self) -> f64 {
        self.total_reward
    }
}

impl fmt::Display for ArmStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Arm(pulls={}, mean={:.4}, var={:.4})",
            self.pulls,
            self.mean(),
            self.variance(),
        )
    }
}

// ── Regret Tracker ──────────────────────────────────────────────

/// Tracks cumulative and per-round regret.
#[derive(Debug, Clone)]
pub struct RegretTracker {
    optimal_reward: f64,
    cumulative_regret: f64,
    regret_history: Vec<f64>,
}

impl RegretTracker {
    /// Create with the known optimal expected reward per round.
    pub fn new(optimal_reward: f64) -> Self {
        Self {
            optimal_reward,
            cumulative_regret: 0.0,
            regret_history: Vec::new(),
        }
    }

    /// Record a round's reward and compute regret.
    pub fn record(&mut self, reward: f64) {
        let instant_regret = self.optimal_reward - reward;
        self.cumulative_regret += instant_regret;
        self.regret_history.push(self.cumulative_regret);
    }

    pub fn cumulative_regret(&self) -> f64 {
        self.cumulative_regret
    }

    pub fn average_regret(&self) -> f64 {
        if self.regret_history.is_empty() {
            0.0
        } else {
            self.cumulative_regret / self.regret_history.len() as f64
        }
    }

    pub fn rounds(&self) -> usize {
        self.regret_history.len()
    }

    pub fn history(&self) -> &[f64] {
        &self.regret_history
    }
}

impl fmt::Display for RegretTracker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Regret(cumulative={:.3}, average={:.4}, rounds={})",
            self.cumulative_regret,
            self.average_regret(),
            self.regret_history.len(),
        )
    }
}

// ── Epsilon-Greedy Bandit ───────────────────────────────────────

/// Epsilon-greedy multi-armed bandit.
#[derive(Debug, Clone)]
pub struct EpsilonGreedyBandit {
    arms: Vec<ArmStats>,
    epsilon: f64,
    decay: f64,
    min_epsilon: f64,
    total_rounds: u64,
    rng: Rng,
}

impl EpsilonGreedyBandit {
    pub fn new(num_arms: usize, epsilon: f64) -> Result<Self, BanditError> {
        if num_arms == 0 {
            return Err(BanditError::NoArms);
        }
        if epsilon < 0.0 || epsilon > 1.0 {
            return Err(BanditError::InvalidParameter("epsilon must be in [0, 1]".into()));
        }
        Ok(Self {
            arms: (0..num_arms).map(|_| ArmStats::new()).collect(),
            epsilon,
            decay: 1.0,
            min_epsilon: 0.0,
            total_rounds: 0,
            rng: Rng::new(42),
        })
    }

    pub fn with_decay(mut self, decay: f64, min_epsilon: f64) -> Self {
        self.decay = decay.clamp(0.0, 1.0);
        self.min_epsilon = min_epsilon.clamp(0.0, 1.0);
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = Rng::new(seed);
        self
    }

    fn effective_epsilon(&self) -> f64 {
        (self.epsilon * self.decay.powf(self.total_rounds as f64)).max(self.min_epsilon)
    }

    /// Select an arm.
    pub fn select(&mut self) -> usize {
        let eps = self.effective_epsilon();
        if self.rng.next_f64() < eps {
            self.rng.next_usize(self.arms.len())
        } else {
            let mut best = 0;
            let mut best_mean = self.arms[0].mean();
            for (i, arm) in self.arms.iter().enumerate().skip(1) {
                let m = arm.mean();
                if m > best_mean {
                    best_mean = m;
                    best = i;
                }
            }
            best
        }
    }

    /// Update arm with observed reward.
    pub fn update(&mut self, arm: usize, reward: f64) -> Result<(), BanditError> {
        if arm >= self.arms.len() {
            return Err(BanditError::ArmOutOfRange {
                arm,
                num_arms: self.arms.len(),
            });
        }
        self.arms[arm].update(reward);
        self.total_rounds += 1;
        Ok(())
    }

    pub fn arm_stats(&self, arm: usize) -> Option<&ArmStats> {
        self.arms.get(arm)
    }

    pub fn num_arms(&self) -> usize {
        self.arms.len()
    }

    pub fn total_rounds(&self) -> u64 {
        self.total_rounds
    }
}

impl fmt::Display for EpsilonGreedyBandit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "EpsilonGreedy(arms={}, ε={:.3}, rounds={})",
            self.arms.len(),
            self.effective_epsilon(),
            self.total_rounds,
        )
    }
}

// ── UCB1 Bandit ─────────────────────────────────────────────────

/// UCB1 (Upper Confidence Bound) bandit.
#[derive(Debug, Clone)]
pub struct Ucb1Bandit {
    arms: Vec<ArmStats>,
    exploration_factor: f64,
    total_rounds: u64,
}

impl Ucb1Bandit {
    pub fn new(num_arms: usize) -> Result<Self, BanditError> {
        if num_arms == 0 {
            return Err(BanditError::NoArms);
        }
        Ok(Self {
            arms: (0..num_arms).map(|_| ArmStats::new()).collect(),
            exploration_factor: 2.0,
            total_rounds: 0,
        })
    }

    pub fn with_exploration(mut self, factor: f64) -> Self {
        self.exploration_factor = factor.max(0.0);
        self
    }

    /// UCB1 score for an arm.
    fn ucb_score(&self, arm_idx: usize) -> f64 {
        let arm = &self.arms[arm_idx];
        if arm.pulls == 0 {
            return f64::MAX;
        }
        let exploitation = arm.mean();
        let exploration = (self.exploration_factor * (self.total_rounds as f64).ln()
            / arm.pulls as f64)
            .sqrt();
        exploitation + exploration
    }

    /// Select the arm with highest UCB1 score.
    pub fn select(&self) -> usize {
        let mut best = 0;
        let mut best_score = self.ucb_score(0);
        for i in 1..self.arms.len() {
            let score = self.ucb_score(i);
            if score > best_score {
                best_score = score;
                best = i;
            }
        }
        best
    }

    /// Update arm with observed reward.
    pub fn update(&mut self, arm: usize, reward: f64) -> Result<(), BanditError> {
        if arm >= self.arms.len() {
            return Err(BanditError::ArmOutOfRange {
                arm,
                num_arms: self.arms.len(),
            });
        }
        self.arms[arm].update(reward);
        self.total_rounds += 1;
        Ok(())
    }

    pub fn arm_stats(&self, arm: usize) -> Option<&ArmStats> {
        self.arms.get(arm)
    }

    pub fn num_arms(&self) -> usize {
        self.arms.len()
    }

    pub fn total_rounds(&self) -> u64 {
        self.total_rounds
    }
}

impl fmt::Display for Ucb1Bandit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "UCB1(arms={}, c={:.3}, rounds={})",
            self.arms.len(),
            self.exploration_factor,
            self.total_rounds,
        )
    }
}

// ── Thompson Sampling Bandit ────────────────────────────────────

/// Thompson sampling bandit with Beta(α, β) priors for Bernoulli rewards.
#[derive(Debug, Clone)]
pub struct ThompsonBandit {
    alphas: Vec<f64>,
    betas: Vec<f64>,
    arm_stats: Vec<ArmStats>,
    total_rounds: u64,
    rng: Rng,
}

impl ThompsonBandit {
    pub fn new(num_arms: usize) -> Result<Self, BanditError> {
        if num_arms == 0 {
            return Err(BanditError::NoArms);
        }
        Ok(Self {
            alphas: vec![1.0; num_arms],
            betas: vec![1.0; num_arms],
            arm_stats: (0..num_arms).map(|_| ArmStats::new()).collect(),
            total_rounds: 0,
            rng: Rng::new(42),
        })
    }

    pub fn with_prior(mut self, alpha: f64, beta_param: f64) -> Self {
        for a in &mut self.alphas {
            *a = alpha.max(0.01);
        }
        for b in &mut self.betas {
            *b = beta_param.max(0.01);
        }
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = Rng::new(seed);
        self
    }

    /// Select arm by sampling from posteriors.
    pub fn select(&mut self) -> usize {
        let mut best = 0;
        let mut best_sample = f64::NEG_INFINITY;
        for i in 0..self.alphas.len() {
            let sample = self.rng.sample_beta(self.alphas[i], self.betas[i]);
            if sample > best_sample {
                best_sample = sample;
                best = i;
            }
        }
        best
    }

    /// Update with Bernoulli reward (0 or 1).
    pub fn update(&mut self, arm: usize, reward: f64) -> Result<(), BanditError> {
        if arm >= self.alphas.len() {
            return Err(BanditError::ArmOutOfRange {
                arm,
                num_arms: self.alphas.len(),
            });
        }
        // Treat reward > 0.5 as success for Bernoulli
        if reward > 0.5 {
            self.alphas[arm] += 1.0;
        } else {
            self.betas[arm] += 1.0;
        }
        self.arm_stats[arm].update(reward);
        self.total_rounds += 1;
        Ok(())
    }

    /// Posterior mean for an arm: α / (α + β).
    pub fn posterior_mean(&self, arm: usize) -> f64 {
        if arm >= self.alphas.len() {
            return 0.0;
        }
        self.alphas[arm] / (self.alphas[arm] + self.betas[arm])
    }

    pub fn arm_stats(&self, arm: usize) -> Option<&ArmStats> {
        self.arm_stats.get(arm)
    }

    pub fn num_arms(&self) -> usize {
        self.alphas.len()
    }

    pub fn total_rounds(&self) -> u64 {
        self.total_rounds
    }
}

impl fmt::Display for ThompsonBandit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Thompson(arms={}, rounds={})",
            self.alphas.len(),
            self.total_rounds,
        )
    }
}

// ── Contextual Linear Bandit (LinUCB) ───────────────────────────

/// LinUCB contextual bandit with linear payoff model.
#[derive(Debug, Clone)]
pub struct LinUcbBandit {
    num_arms: usize,
    dim: usize,
    alpha: f64,
    // A_a = d×d matrices (stored as flat Vec<f64> per arm)
    a_matrices: Vec<Vec<f64>>,
    // b_a = d-vectors per arm
    b_vectors: Vec<Vec<f64>>,
    total_rounds: u64,
}

impl LinUcbBandit {
    pub fn new(num_arms: usize, dim: usize, alpha: f64) -> Result<Self, BanditError> {
        if num_arms == 0 {
            return Err(BanditError::NoArms);
        }
        if dim == 0 {
            return Err(BanditError::InvalidParameter("dimension must be > 0".into()));
        }
        let mut a_matrices = Vec::with_capacity(num_arms);
        for _ in 0..num_arms {
            let mut mat = vec![0.0; dim * dim];
            for j in 0..dim {
                mat[j * dim + j] = 1.0; // Identity matrix
            }
            a_matrices.push(mat);
        }
        let b_vectors = vec![vec![0.0; dim]; num_arms];
        Ok(Self {
            num_arms,
            dim,
            alpha,
            a_matrices,
            b_vectors,
            total_rounds: 0,
        })
    }

    /// Solve A * theta = b using simple iterative method for SPD matrices.
    fn solve_linear(&self, a: &[f64], b: &[f64]) -> Vec<f64> {
        let d = self.dim;
        let mut theta = vec![0.0; d];
        // Simple Gauss-Seidel for SPD matrices
        for _ in 0..50 {
            for i in 0..d {
                let mut sum = b[i];
                for j in 0..d {
                    if i != j {
                        sum -= a[i * d + j] * theta[j];
                    }
                }
                let diag = a[i * d + i];
                if diag.abs() > 1e-12 {
                    theta[i] = sum / diag;
                }
            }
        }
        theta
    }

    /// Compute x^T A^{-1} x (approximated via solve).
    fn quadratic_form(&self, a: &[f64], x: &[f64]) -> f64 {
        let a_inv_x = self.solve_linear(a, x);
        x.iter().zip(a_inv_x.iter()).map(|(xi, ai)| xi * ai).sum()
    }

    /// Select arm given context vector.
    pub fn select(&self, context: &[f64]) -> Result<usize, BanditError> {
        if context.len() != self.dim {
            return Err(BanditError::InvalidParameter(format!(
                "expected context dim {}, got {}",
                self.dim,
                context.len(),
            )));
        }

        let mut best_arm = 0;
        let mut best_score = f64::NEG_INFINITY;

        for arm in 0..self.num_arms {
            let theta = self.solve_linear(&self.a_matrices[arm], &self.b_vectors[arm]);
            let exploitation: f64 = theta.iter().zip(context.iter()).map(|(t, c)| t * c).sum();
            let exploration = self.alpha
                * self.quadratic_form(&self.a_matrices[arm], context).abs().sqrt();
            let score = exploitation + exploration;
            if score > best_score {
                best_score = score;
                best_arm = arm;
            }
        }

        Ok(best_arm)
    }

    /// Update arm with observed context and reward.
    pub fn update(&mut self, arm: usize, context: &[f64], reward: f64) -> Result<(), BanditError> {
        if arm >= self.num_arms {
            return Err(BanditError::ArmOutOfRange {
                arm,
                num_arms: self.num_arms,
            });
        }
        if context.len() != self.dim {
            return Err(BanditError::InvalidParameter(format!(
                "expected context dim {}, got {}",
                self.dim,
                context.len(),
            )));
        }

        // A_a += x * x^T
        let a = &mut self.a_matrices[arm];
        for i in 0..self.dim {
            for j in 0..self.dim {
                a[i * self.dim + j] += context[i] * context[j];
            }
        }

        // b_a += reward * x
        let b = &mut self.b_vectors[arm];
        for i in 0..self.dim {
            b[i] += reward * context[i];
        }

        self.total_rounds += 1;
        Ok(())
    }

    pub fn num_arms(&self) -> usize {
        self.num_arms
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn total_rounds(&self) -> u64 {
        self.total_rounds
    }
}

impl fmt::Display for LinUcbBandit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LinUCB(arms={}, dim={}, α={:.3}, rounds={})",
            self.num_arms, self.dim, self.alpha, self.total_rounds,
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
    fn test_arm_stats() {
        let mut arm = ArmStats::new();
        arm.update(1.0);
        arm.update(3.0);
        assert_eq!(arm.pulls(), 2);
        assert!(approx(arm.mean(), 2.0));
    }

    #[test]
    fn test_regret_tracker() {
        let mut rt = RegretTracker::new(1.0);
        rt.record(0.8);
        rt.record(0.5);
        assert!(approx(rt.cumulative_regret(), 0.7));
        assert_eq!(rt.rounds(), 2);
    }

    #[test]
    fn test_epsilon_greedy_creation() {
        let b = EpsilonGreedyBandit::new(5, 0.1).unwrap();
        assert_eq!(b.num_arms(), 5);
    }

    #[test]
    fn test_epsilon_greedy_no_arms() {
        assert!(EpsilonGreedyBandit::new(0, 0.1).is_err());
    }

    #[test]
    fn test_epsilon_greedy_select_update() {
        let mut b = EpsilonGreedyBandit::new(3, 0.0).unwrap();
        b.update(1, 10.0).unwrap();
        let arm = b.select();
        assert_eq!(arm, 1); // Greedy selects best arm
    }

    #[test]
    fn test_epsilon_greedy_explores() {
        let mut b = EpsilonGreedyBandit::new(4, 1.0).unwrap().with_seed(123);
        let mut selections = std::collections::HashSet::new();
        for _ in 0..200 {
            selections.insert(b.select());
        }
        assert!(selections.len() > 1, "should explore multiple arms");
    }

    #[test]
    fn test_ucb1_creation() {
        let b = Ucb1Bandit::new(3).unwrap();
        assert_eq!(b.num_arms(), 3);
    }

    #[test]
    fn test_ucb1_no_arms() {
        assert!(Ucb1Bandit::new(0).is_err());
    }

    #[test]
    fn test_ucb1_selects_unpulled() {
        let b = Ucb1Bandit::new(3).unwrap();
        // All arms unpulled → should select arm 0 (first with MAX score)
        let arm = b.select();
        assert!(arm < 3);
    }

    #[test]
    fn test_ucb1_converges_to_best() {
        let mut b = Ucb1Bandit::new(3).unwrap();
        // Arm 2 always gives 1.0, others give 0.0
        for _ in 0..100 {
            let arm = b.select();
            let reward = if arm == 2 { 1.0 } else { 0.0 };
            b.update(arm, reward).unwrap();
        }
        assert!(b.arm_stats(2).unwrap().pulls() > 50, "best arm should be pulled most");
    }

    #[test]
    fn test_thompson_creation() {
        let b = ThompsonBandit::new(4).unwrap().with_seed(42);
        assert_eq!(b.num_arms(), 4);
    }

    #[test]
    fn test_thompson_no_arms() {
        assert!(ThompsonBandit::new(0).is_err());
    }

    #[test]
    fn test_thompson_update() {
        let mut b = ThompsonBandit::new(2).unwrap().with_seed(42);
        b.update(0, 1.0).unwrap();
        b.update(0, 1.0).unwrap();
        b.update(1, 0.0).unwrap();
        assert!(b.posterior_mean(0) > b.posterior_mean(1));
    }

    #[test]
    fn test_thompson_select_and_learn() {
        let mut b = ThompsonBandit::new(3).unwrap().with_seed(42);
        for _ in 0..200 {
            let arm = b.select();
            let reward = if arm == 1 { 1.0 } else { 0.0 };
            b.update(arm, reward).unwrap();
        }
        assert!(
            b.arm_stats(1).unwrap().pulls() > 50,
            "arm 1 pulls = {}",
            b.arm_stats(1).unwrap().pulls(),
        );
    }

    #[test]
    fn test_thompson_invalid_arm() {
        let mut b = ThompsonBandit::new(2).unwrap();
        assert!(b.update(5, 1.0).is_err());
    }

    #[test]
    fn test_linucb_creation() {
        let b = LinUcbBandit::new(3, 4, 1.0).unwrap();
        assert_eq!(b.num_arms(), 3);
        assert_eq!(b.dim(), 4);
    }

    #[test]
    fn test_linucb_no_arms() {
        assert!(LinUcbBandit::new(0, 4, 1.0).is_err());
    }

    #[test]
    fn test_linucb_zero_dim() {
        assert!(LinUcbBandit::new(3, 0, 1.0).is_err());
    }

    #[test]
    fn test_linucb_select_update() {
        let mut b = LinUcbBandit::new(2, 3, 1.0).unwrap();
        let ctx = vec![1.0, 0.0, 0.5];
        let arm = b.select(&ctx).unwrap();
        assert!(arm < 2);
        b.update(arm, &ctx, 1.0).unwrap();
        assert_eq!(b.total_rounds(), 1);
    }

    #[test]
    fn test_linucb_wrong_dim() {
        let b = LinUcbBandit::new(2, 3, 1.0).unwrap();
        assert!(b.select(&[1.0, 2.0]).is_err());
    }

    #[test]
    fn test_display_epsilon_greedy() {
        let b = EpsilonGreedyBandit::new(3, 0.1).unwrap();
        let s = format!("{b}");
        assert!(s.contains("EpsilonGreedy"));
    }

    #[test]
    fn test_display_ucb1() {
        let b = Ucb1Bandit::new(3).unwrap();
        let s = format!("{b}");
        assert!(s.contains("UCB1"));
    }

    #[test]
    fn test_display_thompson() {
        let b = ThompsonBandit::new(3).unwrap();
        let s = format!("{b}");
        assert!(s.contains("Thompson"));
    }

    #[test]
    fn test_display_linucb() {
        let b = LinUcbBandit::new(2, 3, 1.0).unwrap();
        let s = format!("{b}");
        assert!(s.contains("LinUCB"));
    }
}
