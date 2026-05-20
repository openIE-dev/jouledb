//! Policy gradient methods — REINFORCE, baseline subtraction, advantage estimation, policy network.
//!
//! Replaces PyTorch / TensorFlow policy gradient implementations with pure Rust.
//! Supports softmax policy with linear features, REINFORCE with Monte Carlo returns,
//! baseline subtraction (mean-return and learned value baseline), advantage estimation,
//! policy network with configurable feature dimensions, and reward-to-go computation.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum PolicyGradientError {
    InvalidParameter(String),
    DimensionMismatch { expected: usize, got: usize },
    EmptyTrajectory,
    NoEpisodes,
}

impl fmt::Display for PolicyGradientError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter(s) => write!(f, "invalid parameter: {s}"),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::EmptyTrajectory => write!(f, "trajectory is empty"),
            Self::NoEpisodes => write!(f, "no episodes collected"),
        }
    }
}

impl std::error::Error for PolicyGradientError {}

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
}

// ── Linear Algebra Helpers ──────────────────────────────────────

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn vec_add_scaled(dst: &mut [f64], src: &[f64], scale: f64) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d += scale * s;
    }
}

fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

// ── Trajectory ──────────────────────────────────────────────────

/// A single (state, action, reward) transition.
#[derive(Debug, Clone)]
pub struct Transition {
    pub state: Vec<f64>,
    pub action: usize,
    pub reward: f64,
}

/// A complete episode trajectory.
#[derive(Debug, Clone)]
pub struct Trajectory {
    transitions: Vec<Transition>,
}

impl Trajectory {
    pub fn new() -> Self {
        Self { transitions: Vec::new() }
    }

    pub fn push(&mut self, state: Vec<f64>, action: usize, reward: f64) {
        self.transitions.push(Transition { state, action, reward });
    }

    pub fn len(&self) -> usize {
        self.transitions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.transitions.is_empty()
    }

    pub fn total_reward(&self) -> f64 {
        self.transitions.iter().map(|t| t.reward).sum()
    }

    /// Compute discounted returns (reward-to-go) for each timestep.
    pub fn compute_returns(&self, gamma: f64) -> Vec<f64> {
        let n = self.transitions.len();
        let mut returns = vec![0.0; n];
        if n == 0 {
            return returns;
        }
        returns[n - 1] = self.transitions[n - 1].reward;
        for i in (0..n - 1).rev() {
            returns[i] = self.transitions[i].reward + gamma * returns[i + 1];
        }
        returns
    }

    pub fn transitions(&self) -> &[Transition] {
        &self.transitions
    }
}

impl Default for Trajectory {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for Trajectory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Trajectory(steps={}, total_reward={:.3})",
            self.transitions.len(),
            self.total_reward(),
        )
    }
}

// ── Baseline ────────────────────────────────────────────────────

/// Baseline for variance reduction in policy gradients.
#[derive(Debug, Clone)]
pub enum Baseline {
    /// No baseline — raw returns.
    None,
    /// Constant baseline (mean of returns).
    MeanReturn,
    /// Running mean with exponential smoothing.
    ExponentialMean { value: f64, decay: f64 },
    /// Learned linear value function baseline.
    LinearValue { weights: Vec<f64>, lr: f64 },
}

impl Baseline {
    /// Compute baseline-subtracted advantages from returns and states.
    pub fn compute_advantages(
        &mut self,
        returns: &[f64],
        states: &[Vec<f64>],
    ) -> Vec<f64> {
        match self {
            Self::None => returns.to_vec(),
            Self::MeanReturn => {
                let mean = returns.iter().sum::<f64>() / returns.len().max(1) as f64;
                returns.iter().map(|r| r - mean).collect()
            }
            Self::ExponentialMean { value, decay } => {
                let batch_mean = returns.iter().sum::<f64>() / returns.len().max(1) as f64;
                *value = *decay * *value + (1.0 - *decay) * batch_mean;
                returns.iter().map(|r| r - *value).collect()
            }
            Self::LinearValue { weights, lr } => {
                let advantages: Vec<f64> = returns
                    .iter()
                    .zip(states.iter())
                    .map(|(&ret, state)| {
                        let pred = dot(weights, state);
                        ret - pred
                    })
                    .collect();

                // Update value function weights via gradient descent
                for (i, (state, &adv)) in states.iter().zip(advantages.iter()).enumerate() {
                    let _ = i;
                    vec_add_scaled(weights, state, *lr * adv);
                }

                advantages
            }
        }
    }
}

impl fmt::Display for Baseline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "NoBaseline"),
            Self::MeanReturn => write!(f, "MeanReturn"),
            Self::ExponentialMean { value, decay } => {
                write!(f, "ExponentialMean(val={value:.3}, decay={decay:.3})")
            }
            Self::LinearValue { weights, lr } => {
                write!(f, "LinearValue(dims={}, lr={lr:.4})", weights.len())
            }
        }
    }
}

// ── Softmax Policy Network ─────────────────────────────────────

/// Linear softmax policy: π(a|s) = softmax(W·s + b).
#[derive(Debug, Clone)]
pub struct SoftmaxPolicy {
    weights: Vec<Vec<f64>>,
    bias: Vec<f64>,
    num_features: usize,
    num_actions: usize,
}

impl SoftmaxPolicy {
    /// Create a softmax policy with given feature and action dimensions.
    pub fn new(num_features: usize, num_actions: usize) -> Result<Self, PolicyGradientError> {
        if num_features == 0 || num_actions == 0 {
            return Err(PolicyGradientError::InvalidParameter(
                "features and actions must be > 0".into(),
            ));
        }
        Ok(Self {
            weights: vec![vec![0.0; num_features]; num_actions],
            bias: vec![0.0; num_actions],
            num_features,
            num_actions,
        })
    }

    /// Initialize weights with small random values.
    pub fn with_random_init(mut self, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let scale = 0.01;
        for row in &mut self.weights {
            for w in row.iter_mut() {
                *w = (rng.next_f64() - 0.5) * 2.0 * scale;
            }
        }
        self
    }

    /// Compute action probabilities for a given state.
    pub fn action_probs(&self, state: &[f64]) -> Result<Vec<f64>, PolicyGradientError> {
        if state.len() != self.num_features {
            return Err(PolicyGradientError::DimensionMismatch {
                expected: self.num_features,
                got: state.len(),
            });
        }
        let logits: Vec<f64> = (0..self.num_actions)
            .map(|a| dot(&self.weights[a], state) + self.bias[a])
            .collect();
        Ok(softmax(&logits))
    }

    /// Sample an action from the policy.
    pub fn sample_action(&self, state: &[f64], rng: &mut Rng) -> Result<usize, PolicyGradientError> {
        let probs = self.action_probs(state)?;
        let mut r = rng.next_f64();
        for (i, &p) in probs.iter().enumerate() {
            r -= p;
            if r <= 0.0 {
                return Ok(i);
            }
        }
        Ok(self.num_actions - 1)
    }

    /// Compute log-gradient of log π(a|s) w.r.t. weights.
    /// Returns gradient for weights and bias.
    fn log_grad(&self, state: &[f64], action: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
        let probs = self.action_probs(state).unwrap_or_else(|_| vec![1.0 / self.num_actions as f64; self.num_actions]);
        let mut w_grad = vec![vec![0.0; self.num_features]; self.num_actions];
        let mut b_grad = vec![0.0; self.num_actions];

        for a in 0..self.num_actions {
            let indicator = if a == action { 1.0 } else { 0.0 };
            let diff = indicator - probs[a];
            for j in 0..self.num_features {
                w_grad[a][j] = diff * state[j];
            }
            b_grad[a] = diff;
        }
        (w_grad, b_grad)
    }

    /// Apply a gradient update scaled by advantage.
    pub fn update(&mut self, state: &[f64], action: usize, advantage: f64, lr: f64) {
        let (w_grad, b_grad) = self.log_grad(state, action);
        for a in 0..self.num_actions {
            for j in 0..self.num_features {
                self.weights[a][j] += lr * advantage * w_grad[a][j];
            }
            self.bias[a] += lr * advantage * b_grad[a];
        }
    }

    pub fn num_features(&self) -> usize {
        self.num_features
    }

    pub fn num_actions(&self) -> usize {
        self.num_actions
    }
}

impl fmt::Display for SoftmaxPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SoftmaxPolicy(features={}, actions={})",
            self.num_features, self.num_actions,
        )
    }
}

// ── REINFORCE Agent ─────────────────────────────────────────────

/// REINFORCE policy gradient agent with optional baseline.
#[derive(Debug, Clone)]
pub struct ReinforceAgent {
    policy: SoftmaxPolicy,
    baseline: Baseline,
    gamma: f64,
    lr: f64,
    episodes_trained: u64,
    reward_history: Vec<f64>,
    rng: Rng,
}

impl ReinforceAgent {
    pub fn new(
        num_features: usize,
        num_actions: usize,
        lr: f64,
        gamma: f64,
    ) -> Result<Self, PolicyGradientError> {
        if lr <= 0.0 {
            return Err(PolicyGradientError::InvalidParameter("lr must be > 0".into()));
        }
        if gamma < 0.0 || gamma > 1.0 {
            return Err(PolicyGradientError::InvalidParameter("gamma must be in [0,1]".into()));
        }
        let policy = SoftmaxPolicy::new(num_features, num_actions)?;
        Ok(Self {
            policy: policy.with_random_init(42),
            baseline: Baseline::None,
            gamma,
            lr,
            episodes_trained: 0,
            reward_history: Vec::new(),
            rng: Rng::new(42),
        })
    }

    pub fn with_baseline(mut self, baseline: Baseline) -> Self {
        self.baseline = baseline;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = Rng::new(seed);
        self.policy = SoftmaxPolicy::new(self.policy.num_features(), self.policy.num_actions())
            .unwrap()
            .with_random_init(seed);
        self
    }

    pub fn with_gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
        self
    }

    /// Select an action for the given state features.
    pub fn select_action(&mut self, state: &[f64]) -> Result<usize, PolicyGradientError> {
        self.policy.sample_action(state, &mut self.rng)
    }

    /// Train on a batch of trajectories using REINFORCE.
    pub fn train(&mut self, trajectories: &[Trajectory]) -> Result<f64, PolicyGradientError> {
        if trajectories.is_empty() {
            return Err(PolicyGradientError::NoEpisodes);
        }

        let mut total_loss = 0.0;

        for traj in trajectories {
            if traj.is_empty() {
                continue;
            }
            let returns = traj.compute_returns(self.gamma);
            let states: Vec<Vec<f64>> = traj
                .transitions()
                .iter()
                .map(|t| t.state.clone())
                .collect();

            let advantages = self.baseline.compute_advantages(&returns, &states);

            for (trans, &adv) in traj.transitions().iter().zip(advantages.iter()) {
                self.policy.update(&trans.state, trans.action, adv, self.lr);
                total_loss += adv.abs();
            }

            self.reward_history.push(traj.total_reward());
            self.episodes_trained += 1;
        }

        Ok(total_loss / trajectories.len() as f64)
    }

    /// Get action probabilities for a state (for debugging / visualization).
    pub fn action_probs(&self, state: &[f64]) -> Result<Vec<f64>, PolicyGradientError> {
        self.policy.action_probs(state)
    }

    pub fn episodes_trained(&self) -> u64 {
        self.episodes_trained
    }

    pub fn reward_history(&self) -> &[f64] {
        &self.reward_history
    }

    pub fn average_reward(&self, last_n: usize) -> Result<f64, PolicyGradientError> {
        if self.reward_history.is_empty() {
            return Err(PolicyGradientError::NoEpisodes);
        }
        let start = self.reward_history.len().saturating_sub(last_n);
        let slice = &self.reward_history[start..];
        Ok(slice.iter().sum::<f64>() / slice.len() as f64)
    }

    pub fn policy(&self) -> &SoftmaxPolicy {
        &self.policy
    }

    pub fn gamma(&self) -> f64 {
        self.gamma
    }

    pub fn lr(&self) -> f64 {
        self.lr
    }
}

impl fmt::Display for ReinforceAgent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "REINFORCE(lr={:.4}, γ={:.3}, baseline={}, episodes={})",
            self.lr,
            self.gamma,
            self.baseline,
            self.episodes_trained,
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
    fn test_trajectory_new() {
        let traj = Trajectory::new();
        assert!(traj.is_empty());
        assert_eq!(traj.len(), 0);
    }

    #[test]
    fn test_trajectory_push() {
        let mut traj = Trajectory::new();
        traj.push(vec![1.0, 0.0], 0, 1.0);
        traj.push(vec![0.0, 1.0], 1, 2.0);
        assert_eq!(traj.len(), 2);
        assert!(approx(traj.total_reward(), 3.0));
    }

    #[test]
    fn test_trajectory_returns_no_discount() {
        let mut traj = Trajectory::new();
        traj.push(vec![1.0], 0, 1.0);
        traj.push(vec![1.0], 0, 2.0);
        traj.push(vec![1.0], 0, 3.0);
        let returns = traj.compute_returns(1.0);
        assert!(approx(returns[0], 6.0));
        assert!(approx(returns[1], 5.0));
        assert!(approx(returns[2], 3.0));
    }

    #[test]
    fn test_trajectory_returns_with_discount() {
        let mut traj = Trajectory::new();
        traj.push(vec![1.0], 0, 1.0);
        traj.push(vec![1.0], 0, 1.0);
        traj.push(vec![1.0], 0, 1.0);
        let returns = traj.compute_returns(0.5);
        // G2 = 1.0, G1 = 1 + 0.5*1 = 1.5, G0 = 1 + 0.5*1.5 = 1.75
        assert!(approx(returns[2], 1.0));
        assert!(approx(returns[1], 1.5));
        assert!(approx(returns[0], 1.75));
    }

    #[test]
    fn test_softmax_policy_creation() {
        let p = SoftmaxPolicy::new(4, 3).unwrap();
        assert_eq!(p.num_features(), 4);
        assert_eq!(p.num_actions(), 3);
    }

    #[test]
    fn test_softmax_policy_invalid() {
        assert!(SoftmaxPolicy::new(0, 3).is_err());
        assert!(SoftmaxPolicy::new(4, 0).is_err());
    }

    #[test]
    fn test_softmax_probs_sum_to_one() {
        let p = SoftmaxPolicy::new(3, 4).unwrap().with_random_init(42);
        let probs = p.action_probs(&[1.0, 0.5, -0.3]).unwrap();
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "sum = {sum}");
    }

    #[test]
    fn test_softmax_dim_mismatch() {
        let p = SoftmaxPolicy::new(3, 2).unwrap();
        assert!(p.action_probs(&[1.0, 2.0]).is_err());
    }

    #[test]
    fn test_softmax_sample_valid() {
        let p = SoftmaxPolicy::new(3, 4).unwrap().with_random_init(42);
        let mut rng = Rng::new(99);
        for _ in 0..50 {
            let a = p.sample_action(&[1.0, 0.0, -1.0], &mut rng).unwrap();
            assert!(a < 4);
        }
    }

    #[test]
    fn test_baseline_none() {
        let mut bl = Baseline::None;
        let returns = vec![1.0, 2.0, 3.0];
        let states = vec![vec![1.0], vec![1.0], vec![1.0]];
        let adv = bl.compute_advantages(&returns, &states);
        assert_eq!(adv, returns);
    }

    #[test]
    fn test_baseline_mean_return() {
        let mut bl = Baseline::MeanReturn;
        let returns = vec![1.0, 2.0, 3.0];
        let states = vec![vec![1.0], vec![1.0], vec![1.0]];
        let adv = bl.compute_advantages(&returns, &states);
        assert!(approx(adv[0], -1.0));
        assert!(approx(adv[1], 0.0));
        assert!(approx(adv[2], 1.0));
    }

    #[test]
    fn test_baseline_exponential_mean() {
        let mut bl = Baseline::ExponentialMean { value: 0.0, decay: 0.9 };
        let returns = vec![10.0, 10.0, 10.0];
        let states = vec![vec![1.0], vec![1.0], vec![1.0]];
        let _adv = bl.compute_advantages(&returns, &states);
        if let Baseline::ExponentialMean { value, .. } = &bl {
            assert!(*value > 0.0, "exponential mean should update");
        }
    }

    #[test]
    fn test_reinforce_creation() {
        let agent = ReinforceAgent::new(4, 3, 0.01, 0.99).unwrap();
        assert_eq!(agent.episodes_trained(), 0);
        assert!(approx(agent.gamma(), 0.99));
    }

    #[test]
    fn test_reinforce_invalid_params() {
        assert!(ReinforceAgent::new(4, 3, -0.01, 0.99).is_err());
        assert!(ReinforceAgent::new(4, 3, 0.01, 1.5).is_err());
    }

    #[test]
    fn test_reinforce_select_action() {
        let mut agent = ReinforceAgent::new(3, 4, 0.01, 0.99).unwrap().with_seed(42);
        let action = agent.select_action(&[1.0, 0.0, -1.0]).unwrap();
        assert!(action < 4);
    }

    #[test]
    fn test_reinforce_train() {
        let mut agent = ReinforceAgent::new(2, 2, 0.01, 0.99).unwrap();
        let mut traj = Trajectory::new();
        traj.push(vec![1.0, 0.0], 0, 1.0);
        traj.push(vec![0.0, 1.0], 1, 2.0);
        let loss = agent.train(&[traj]).unwrap();
        assert!(loss >= 0.0);
        assert_eq!(agent.episodes_trained(), 1);
    }

    #[test]
    fn test_reinforce_train_empty() {
        let mut agent = ReinforceAgent::new(2, 2, 0.01, 0.99).unwrap();
        assert!(agent.train(&[]).is_err());
    }

    #[test]
    fn test_reinforce_with_baseline() {
        let mut agent = ReinforceAgent::new(2, 2, 0.01, 0.99)
            .unwrap()
            .with_baseline(Baseline::MeanReturn);
        let mut traj = Trajectory::new();
        traj.push(vec![1.0, 0.0], 0, 1.0);
        traj.push(vec![0.0, 1.0], 1, 5.0);
        let _ = agent.train(&[traj]);
        assert_eq!(agent.episodes_trained(), 1);
    }

    #[test]
    fn test_reinforce_reward_history() {
        let mut agent = ReinforceAgent::new(2, 2, 0.01, 0.99).unwrap();
        let mut t1 = Trajectory::new();
        t1.push(vec![1.0, 0.0], 0, 3.0);
        let mut t2 = Trajectory::new();
        t2.push(vec![0.0, 1.0], 1, 7.0);
        let _ = agent.train(&[t1, t2]);
        assert!(approx(agent.average_reward(10).unwrap(), 5.0));
    }

    #[test]
    fn test_display_trajectory() {
        let mut traj = Trajectory::new();
        traj.push(vec![1.0], 0, 5.0);
        let s = format!("{traj}");
        assert!(s.contains("Trajectory"));
        assert!(s.contains("5.000"));
    }

    #[test]
    fn test_display_reinforce() {
        let agent = ReinforceAgent::new(3, 2, 0.01, 0.95).unwrap();
        let s = format!("{agent}");
        assert!(s.contains("REINFORCE"));
    }
}
