//! Reward shaping — potential-based shaping, intrinsic motivation, curiosity-driven, reward normalization.
//!
//! Replaces custom reward shaping in RLlib / stable-baselines3 with pure Rust.
//! Supports potential-based reward shaping (PBRS), intrinsic motivation via
//! state-visit counts, curiosity-driven exploration via prediction error,
//! reward normalization (running mean/std), reward clipping, and composite
//! reward signals combining extrinsic and intrinsic components.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum RewardShapingError {
    InvalidParameter(String),
    PotentialNotDefined(u64),
    EmptyHistory,
}

impl fmt::Display for RewardShapingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter(s) => write!(f, "invalid parameter: {s}"),
            Self::PotentialNotDefined(s) => write!(f, "potential not defined for state {s}"),
            Self::EmptyHistory => write!(f, "reward history is empty"),
        }
    }
}

impl std::error::Error for RewardShapingError {}

// ── Potential-Based Reward Shaping ──────────────────────────────

/// Potential function Φ(s) for potential-based reward shaping.
/// Shaped reward F(s,a,s') = γ·Φ(s') − Φ(s).
#[derive(Debug, Clone)]
pub struct PotentialFunction {
    potentials: HashMap<u64, f64>,
    default: f64,
}

impl PotentialFunction {
    /// Create with a default potential for unknown states.
    pub fn new(default: f64) -> Self {
        Self {
            potentials: HashMap::new(),
            default,
        }
    }

    /// Set potential for a specific state.
    pub fn set(&mut self, state: u64, potential: f64) {
        self.potentials.insert(state, potential);
    }

    /// Get potential for a state.
    pub fn get(&self, state: u64) -> f64 {
        self.potentials.get(&state).copied().unwrap_or(self.default)
    }

    /// Number of states with defined potentials.
    pub fn defined_count(&self) -> usize {
        self.potentials.len()
    }

    /// Set potentials from a distance-to-goal heuristic.
    /// States closer to goal get higher potential.
    pub fn from_goal_distances(distances: &[(u64, f64)], max_distance: f64) -> Self {
        let mut pf = Self::new(0.0);
        for &(state, dist) in distances {
            pf.set(state, max_distance - dist);
        }
        pf
    }
}

impl fmt::Display for PotentialFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PotentialFunction(states={}, default={:.3})",
            self.potentials.len(),
            self.default,
        )
    }
}

/// Potential-based reward shaping wrapper.
#[derive(Debug, Clone)]
pub struct PotentialShaping {
    potential: PotentialFunction,
    gamma: f64,
    total_shaped: u64,
}

impl PotentialShaping {
    pub fn new(potential: PotentialFunction, gamma: f64) -> Result<Self, RewardShapingError> {
        if gamma < 0.0 || gamma > 1.0 {
            return Err(RewardShapingError::InvalidParameter("gamma must be in [0,1]".into()));
        }
        Ok(Self {
            potential,
            gamma,
            total_shaped: 0,
        })
    }

    /// Compute shaped reward: r + γ·Φ(s') − Φ(s).
    pub fn shape(&mut self, reward: f64, state: u64, next_state: u64) -> f64 {
        let phi_s = self.potential.get(state);
        let phi_s_prime = self.potential.get(next_state);
        let shaping = self.gamma * phi_s_prime - phi_s;
        self.total_shaped += 1;
        reward + shaping
    }

    pub fn total_shaped(&self) -> u64 {
        self.total_shaped
    }

    pub fn potential(&self) -> &PotentialFunction {
        &self.potential
    }
}

impl fmt::Display for PotentialShaping {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PotentialShaping(γ={:.3}, shaped={}, {})",
            self.gamma, self.total_shaped, self.potential,
        )
    }
}

// ── Count-Based Intrinsic Motivation ────────────────────────────

/// Intrinsic motivation based on state visitation counts.
/// Bonus = scale / sqrt(count(s)).
#[derive(Debug, Clone)]
pub struct CountBasedBonus {
    counts: HashMap<u64, u64>,
    scale: f64,
    total_visits: u64,
}

impl CountBasedBonus {
    pub fn new(scale: f64) -> Result<Self, RewardShapingError> {
        if scale < 0.0 {
            return Err(RewardShapingError::InvalidParameter("scale must be >= 0".into()));
        }
        Ok(Self {
            counts: HashMap::new(),
            scale,
            total_visits: 0,
        })
    }

    pub fn with_scale(mut self, scale: f64) -> Self {
        self.scale = scale;
        self
    }

    /// Record a visit to a state and return the intrinsic bonus.
    pub fn visit(&mut self, state: u64) -> f64 {
        let count = self.counts.entry(state).or_insert(0);
        *count += 1;
        self.total_visits += 1;
        self.scale / (*count as f64).sqrt()
    }

    /// Get the bonus for a state without recording a visit.
    pub fn bonus(&self, state: u64) -> f64 {
        let count = self.counts.get(&state).copied().unwrap_or(0);
        if count == 0 {
            self.scale
        } else {
            self.scale / (count as f64).sqrt()
        }
    }

    /// Number of unique states visited.
    pub fn unique_states(&self) -> usize {
        self.counts.len()
    }

    /// Total visits across all states.
    pub fn total_visits(&self) -> u64 {
        self.total_visits
    }

    /// Visit count for a specific state.
    pub fn visit_count(&self, state: u64) -> u64 {
        self.counts.get(&state).copied().unwrap_or(0)
    }
}

impl fmt::Display for CountBasedBonus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CountBasedBonus(scale={:.3}, unique_states={}, total_visits={})",
            self.scale,
            self.counts.len(),
            self.total_visits,
        )
    }
}

// ── Curiosity-Driven Exploration ────────────────────────────────

/// Simple curiosity module: learns to predict next state features,
/// uses prediction error as intrinsic reward.
#[derive(Debug, Clone)]
pub struct CuriosityModule {
    feature_dim: usize,
    /// Linear predictor weights: W such that predicted_next = W * [state; action_onehot]
    weights: Vec<Vec<f64>>,
    lr: f64,
    scale: f64,
    num_actions: usize,
    total_predictions: u64,
    cumulative_error: f64,
}

impl CuriosityModule {
    pub fn new(
        feature_dim: usize,
        num_actions: usize,
        lr: f64,
        scale: f64,
    ) -> Result<Self, RewardShapingError> {
        if feature_dim == 0 || num_actions == 0 {
            return Err(RewardShapingError::InvalidParameter(
                "feature_dim and num_actions must be > 0".into(),
            ));
        }
        if lr <= 0.0 {
            return Err(RewardShapingError::InvalidParameter("lr must be > 0".into()));
        }
        let input_dim = feature_dim + num_actions;
        let weights = vec![vec![0.0; input_dim]; feature_dim];
        Ok(Self {
            feature_dim,
            weights,
            lr,
            scale,
            num_actions,
            total_predictions: 0,
            cumulative_error: 0.0,
        })
    }

    pub fn with_scale(mut self, scale: f64) -> Self {
        self.scale = scale;
        self
    }

    /// Build input vector from state features and action.
    fn build_input(&self, state: &[f64], action: usize) -> Vec<f64> {
        let mut input = Vec::with_capacity(self.feature_dim + self.num_actions);
        input.extend_from_slice(state);
        for a in 0..self.num_actions {
            input.push(if a == action { 1.0 } else { 0.0 });
        }
        input
    }

    /// Predict next state features.
    fn predict(&self, input: &[f64]) -> Vec<f64> {
        self.weights
            .iter()
            .map(|row| row.iter().zip(input.iter()).map(|(w, x)| w * x).sum())
            .collect()
    }

    /// Compute intrinsic reward from prediction error and update the predictor.
    pub fn compute_reward(
        &mut self,
        state: &[f64],
        action: usize,
        next_state: &[f64],
    ) -> f64 {
        let state_clamped = if state.len() >= self.feature_dim {
            &state[..self.feature_dim]
        } else {
            state
        };
        let next_clamped = if next_state.len() >= self.feature_dim {
            &next_state[..self.feature_dim]
        } else {
            next_state
        };

        let input = self.build_input(state_clamped, action);
        let predicted = self.predict(&input);

        // Prediction error (MSE)
        let mut error = 0.0;
        let dim = predicted.len().min(next_clamped.len());
        for i in 0..dim {
            let diff = predicted[i] - next_clamped[i];
            error += diff * diff;
        }
        error /= dim.max(1) as f64;

        // Update predictor weights via gradient descent
        for i in 0..dim {
            let diff = predicted[i] - next_clamped[i];
            for (j, &x) in input.iter().enumerate() {
                self.weights[i][j] -= self.lr * 2.0 * diff * x / dim as f64;
            }
        }

        self.total_predictions += 1;
        self.cumulative_error += error;

        self.scale * error
    }

    pub fn average_error(&self) -> f64 {
        if self.total_predictions == 0 {
            0.0
        } else {
            self.cumulative_error / self.total_predictions as f64
        }
    }

    pub fn total_predictions(&self) -> u64 {
        self.total_predictions
    }
}

impl fmt::Display for CuriosityModule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Curiosity(feat={}, actions={}, lr={:.4}, avg_err={:.4})",
            self.feature_dim,
            self.num_actions,
            self.lr,
            self.average_error(),
        )
    }
}

// ── Reward Normalizer ───────────────────────────────────────────

/// Running mean/variance normalizer for rewards.
#[derive(Debug, Clone)]
pub struct RewardNormalizer {
    mean: f64,
    var: f64,
    count: u64,
    clip: f64,
    epsilon: f64,
}

impl RewardNormalizer {
    pub fn new(clip: f64) -> Self {
        Self {
            mean: 0.0,
            var: 1.0,
            count: 0,
            clip,
            epsilon: 1e-8,
        }
    }

    pub fn with_clip(mut self, clip: f64) -> Self {
        self.clip = clip;
        self
    }

    /// Normalize a reward using running statistics (Welford's online algorithm).
    pub fn normalize(&mut self, reward: f64) -> f64 {
        self.count += 1;
        let delta = reward - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = reward - self.mean;
        self.var += (delta * delta2 - self.var) / self.count as f64;

        let std = (self.var.max(0.0) + self.epsilon).sqrt();
        let normalized = (reward - self.mean) / std;
        normalized.clamp(-self.clip, self.clip)
    }

    /// Normalize without updating statistics.
    pub fn normalize_no_update(&self, reward: f64) -> f64 {
        let std = (self.var.max(0.0) + self.epsilon).sqrt();
        let normalized = (reward - self.mean) / std;
        normalized.clamp(-self.clip, self.clip)
    }

    pub fn mean(&self) -> f64 {
        self.mean
    }

    pub fn std(&self) -> f64 {
        (self.var.max(0.0) + self.epsilon).sqrt()
    }

    pub fn count(&self) -> u64 {
        self.count
    }
}

impl fmt::Display for RewardNormalizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RewardNormalizer(mean={:.3}, std={:.3}, count={}, clip={:.1})",
            self.mean,
            self.std(),
            self.count,
            self.clip,
        )
    }
}

// ── Composite Reward ────────────────────────────────────────────

/// Combines extrinsic reward with one or more intrinsic signals.
#[derive(Debug, Clone)]
pub struct CompositeReward {
    extrinsic_weight: f64,
    count_bonus: Option<(CountBasedBonus, f64)>,
    normalizer: Option<RewardNormalizer>,
    total_shaped: u64,
}

impl CompositeReward {
    pub fn new(extrinsic_weight: f64) -> Self {
        Self {
            extrinsic_weight,
            count_bonus: None,
            normalizer: None,
            total_shaped: 0,
        }
    }

    pub fn with_count_bonus(mut self, bonus: CountBasedBonus, weight: f64) -> Self {
        self.count_bonus = Some((bonus, weight));
        self
    }

    pub fn with_normalizer(mut self, normalizer: RewardNormalizer) -> Self {
        self.normalizer = Some(normalizer);
        self
    }

    /// Compute composite reward for a given state transition.
    pub fn compute(&mut self, extrinsic_reward: f64, state: u64) -> f64 {
        let mut reward = self.extrinsic_weight * extrinsic_reward;

        if let Some((ref mut bonus, weight)) = self.count_bonus {
            reward += weight * bonus.visit(state);
        }

        if let Some(ref mut norm) = self.normalizer {
            reward = norm.normalize(reward);
        }

        self.total_shaped += 1;
        reward
    }

    pub fn total_shaped(&self) -> u64 {
        self.total_shaped
    }
}

impl fmt::Display for CompositeReward {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CompositeReward(ext_w={:.3}, count_bonus={}, normalized={}, shaped={})",
            self.extrinsic_weight,
            self.count_bonus.is_some(),
            self.normalizer.is_some(),
            self.total_shaped,
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
    fn test_potential_function_default() {
        let pf = PotentialFunction::new(5.0);
        assert!(approx(pf.get(999), 5.0));
        assert_eq!(pf.defined_count(), 0);
    }

    #[test]
    fn test_potential_function_set_get() {
        let mut pf = PotentialFunction::new(0.0);
        pf.set(1, 10.0);
        pf.set(2, 20.0);
        assert!(approx(pf.get(1), 10.0));
        assert!(approx(pf.get(2), 20.0));
        assert!(approx(pf.get(3), 0.0));
    }

    #[test]
    fn test_potential_from_distances() {
        let distances = vec![(0, 4.0), (1, 3.0), (2, 2.0), (3, 1.0), (4, 0.0)];
        let pf = PotentialFunction::from_goal_distances(&distances, 4.0);
        assert!(approx(pf.get(0), 0.0));  // 4 - 4
        assert!(approx(pf.get(4), 4.0));  // 4 - 0
    }

    #[test]
    fn test_potential_shaping() {
        let mut pf = PotentialFunction::new(0.0);
        pf.set(0, 0.0);
        pf.set(1, 5.0);
        let mut shaper = PotentialShaping::new(pf, 1.0).unwrap();
        // shape(r=1.0, s=0, s'=1) = 1.0 + 1.0*5.0 - 0.0 = 6.0
        let shaped = shaper.shape(1.0, 0, 1);
        assert!(approx(shaped, 6.0));
    }

    #[test]
    fn test_potential_shaping_invalid_gamma() {
        let pf = PotentialFunction::new(0.0);
        assert!(PotentialShaping::new(pf, 1.5).is_err());
    }

    #[test]
    fn test_count_bonus_first_visit() {
        let mut cb = CountBasedBonus::new(1.0).unwrap();
        let bonus = cb.visit(42);
        assert!(approx(bonus, 1.0)); // 1 / sqrt(1) = 1.0
    }

    #[test]
    fn test_count_bonus_diminishing() {
        let mut cb = CountBasedBonus::new(1.0).unwrap();
        let b1 = cb.visit(0);
        let b2 = cb.visit(0);
        let b3 = cb.visit(0);
        assert!(b1 > b2);
        assert!(b2 > b3);
    }

    #[test]
    fn test_count_bonus_unique_states() {
        let mut cb = CountBasedBonus::new(1.0).unwrap();
        cb.visit(0);
        cb.visit(1);
        cb.visit(2);
        cb.visit(0);
        assert_eq!(cb.unique_states(), 3);
        assert_eq!(cb.total_visits(), 4);
    }

    #[test]
    fn test_count_bonus_invalid_scale() {
        assert!(CountBasedBonus::new(-1.0).is_err());
    }

    #[test]
    fn test_curiosity_module_creation() {
        let cm = CuriosityModule::new(4, 2, 0.01, 1.0).unwrap();
        assert_eq!(cm.total_predictions(), 0);
    }

    #[test]
    fn test_curiosity_invalid() {
        assert!(CuriosityModule::new(0, 2, 0.01, 1.0).is_err());
        assert!(CuriosityModule::new(4, 0, 0.01, 1.0).is_err());
        assert!(CuriosityModule::new(4, 2, -0.01, 1.0).is_err());
    }

    #[test]
    fn test_curiosity_reward_decreases() {
        let mut cm = CuriosityModule::new(2, 2, 0.1, 1.0).unwrap();
        let s1 = vec![1.0, 0.0];
        let s2 = vec![0.0, 1.0];
        let r1 = cm.compute_reward(&s1, 0, &s2);
        // Repeat same transition — error should decrease as predictor learns
        let mut last = r1;
        for _ in 0..20 {
            last = cm.compute_reward(&s1, 0, &s2);
        }
        assert!(last < r1, "curiosity reward should decrease: {last} < {r1}");
    }

    #[test]
    fn test_curiosity_novel_states_high_reward() {
        let mut cm = CuriosityModule::new(2, 2, 0.1, 1.0).unwrap();
        // Train on one transition
        for _ in 0..50 {
            cm.compute_reward(&[1.0, 0.0], 0, &[0.0, 1.0]);
        }
        // Novel transition should have higher error
        let novel = cm.compute_reward(&[0.0, 0.0], 1, &[1.0, 1.0]);
        let familiar = cm.compute_reward(&[1.0, 0.0], 0, &[0.0, 1.0]);
        assert!(novel > familiar, "novel={novel}, familiar={familiar}");
    }

    #[test]
    fn test_reward_normalizer() {
        let mut norm = RewardNormalizer::new(5.0);
        for i in 0..100 {
            norm.normalize(i as f64);
        }
        // After 100 samples, normalized values should be in [-5, 5]
        let n = norm.normalize_no_update(50.0);
        assert!(n >= -5.0 && n <= 5.0);
    }

    #[test]
    fn test_reward_normalizer_clip() {
        let mut norm = RewardNormalizer::new(1.0);
        for i in 0..100 {
            norm.normalize(i as f64);
        }
        let n = norm.normalize_no_update(10000.0);
        assert!(n <= 1.0 + EPS);
    }

    #[test]
    fn test_composite_reward() {
        let cb = CountBasedBonus::new(0.5).unwrap();
        let mut comp = CompositeReward::new(1.0)
            .with_count_bonus(cb, 1.0);
        let r = comp.compute(1.0, 0);
        // 1.0 * 1.0 + 1.0 * 0.5/sqrt(1) = 1.5
        assert!(approx(r, 1.5));
    }

    #[test]
    fn test_composite_with_normalizer() {
        let norm = RewardNormalizer::new(5.0);
        let mut comp = CompositeReward::new(1.0).with_normalizer(norm);
        let _r = comp.compute(10.0, 0);
        assert_eq!(comp.total_shaped(), 1);
    }

    #[test]
    fn test_display_potential() {
        let pf = PotentialFunction::new(0.0);
        let s = format!("{pf}");
        assert!(s.contains("PotentialFunction"));
    }

    #[test]
    fn test_display_count_bonus() {
        let cb = CountBasedBonus::new(1.0).unwrap();
        let s = format!("{cb}");
        assert!(s.contains("CountBasedBonus"));
    }

    #[test]
    fn test_display_curiosity() {
        let cm = CuriosityModule::new(4, 2, 0.01, 1.0).unwrap();
        let s = format!("{cm}");
        assert!(s.contains("Curiosity"));
    }
}
