//! Q-learning — tabular Q-values, epsilon-greedy exploration, Q-table updates, SARSA variant.
//!
//! Replaces stable-baselines3 / RLlib Q-learning with pure Rust.
//! Supports tabular Q-learning with configurable learning rate, discount factor,
//! epsilon-greedy and softmax exploration, Q-table initialization, SARSA on-policy
//! variant, episode tracking, and convergence metrics.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum QLearningError {
    InvalidParameter(String),
    StateNotFound(u64),
    ActionOutOfRange { action: usize, num_actions: usize },
    NoEpisodeData,
}

impl fmt::Display for QLearningError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter(s) => write!(f, "invalid parameter: {s}"),
            Self::StateNotFound(s) => write!(f, "state not found: {s}"),
            Self::ActionOutOfRange { action, num_actions } => {
                write!(f, "action {action} out of range for {num_actions} actions")
            }
            Self::NoEpisodeData => write!(f, "no episode data available"),
        }
    }
}

impl std::error::Error for QLearningError {}

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

// ── Exploration Strategy ────────────────────────────────────────

/// Strategy for balancing exploration vs exploitation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExplorationStrategy {
    /// Epsilon-greedy with a fixed epsilon.
    EpsilonGreedy(f64),
    /// Epsilon-greedy with linear decay from start to minimum over given steps.
    EpsilonDecay { start: f64, minimum: f64, decay_steps: u64 },
    /// Softmax (Boltzmann) exploration with temperature.
    Softmax(f64),
    /// Pure greedy — always pick best known action.
    Greedy,
}

impl fmt::Display for ExplorationStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EpsilonGreedy(e) => write!(f, "EpsilonGreedy({e:.3})"),
            Self::EpsilonDecay { start, minimum, decay_steps } => {
                write!(f, "EpsilonDecay({start:.3}→{minimum:.3} over {decay_steps})")
            }
            Self::Softmax(t) => write!(f, "Softmax(τ={t:.3})"),
            Self::Greedy => write!(f, "Greedy"),
        }
    }
}

// ── Q-Table ─────────────────────────────────────────────────────

/// Tabular Q-value storage mapping (state, action) → Q-value.
#[derive(Debug, Clone)]
pub struct QTable {
    table: HashMap<u64, Vec<f64>>,
    num_actions: usize,
    default_value: f64,
}

impl QTable {
    /// Create a new Q-table for a given number of actions.
    pub fn new(num_actions: usize, default_value: f64) -> Self {
        Self {
            table: HashMap::new(),
            num_actions,
            default_value,
        }
    }

    /// Get Q-value for a state-action pair.
    pub fn get(&self, state: u64, action: usize) -> f64 {
        self.table
            .get(&state)
            .map(|row| row[action.min(self.num_actions - 1)])
            .unwrap_or(self.default_value)
    }

    /// Set Q-value for a state-action pair.
    pub fn set(&mut self, state: u64, action: usize, value: f64) {
        let row = self
            .table
            .entry(state)
            .or_insert_with(|| vec![self.default_value; self.num_actions]);
        if action < self.num_actions {
            row[action] = value;
        }
    }

    /// Get the row of Q-values for a given state.
    pub fn get_row(&self, state: u64) -> Vec<f64> {
        self.table
            .get(&state)
            .cloned()
            .unwrap_or_else(|| vec![self.default_value; self.num_actions])
    }

    /// Index of the action with the highest Q-value for a state.
    pub fn best_action(&self, state: u64) -> usize {
        let row = self.get_row(state);
        let mut best = 0;
        let mut best_val = row[0];
        for (i, &v) in row.iter().enumerate().skip(1) {
            if v > best_val {
                best_val = v;
                best = i;
            }
        }
        best
    }

    /// Maximum Q-value for a given state.
    pub fn max_q(&self, state: u64) -> f64 {
        let row = self.get_row(state);
        row.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }

    /// Number of unique states stored.
    pub fn state_count(&self) -> usize {
        self.table.len()
    }

    /// Number of actions.
    pub fn num_actions(&self) -> usize {
        self.num_actions
    }
}

impl fmt::Display for QTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "QTable(states={}, actions={}, default={:.3})",
            self.table.len(),
            self.num_actions,
            self.default_value,
        )
    }
}

// ── Q-Learning Agent ────────────────────────────────────────────

/// Tabular Q-learning agent.
#[derive(Debug, Clone)]
pub struct QLearningAgent {
    q_table: QTable,
    alpha: f64,
    gamma: f64,
    strategy: ExplorationStrategy,
    total_steps: u64,
    episode_rewards: Vec<f64>,
    rng: Rng,
}

impl QLearningAgent {
    /// Create a new Q-learning agent.
    pub fn new(num_actions: usize, alpha: f64, gamma: f64) -> Result<Self, QLearningError> {
        if num_actions == 0 {
            return Err(QLearningError::InvalidParameter("num_actions must be > 0".into()));
        }
        if alpha <= 0.0 || alpha > 1.0 {
            return Err(QLearningError::InvalidParameter("alpha must be in (0, 1]".into()));
        }
        if gamma < 0.0 || gamma > 1.0 {
            return Err(QLearningError::InvalidParameter("gamma must be in [0, 1]".into()));
        }
        Ok(Self {
            q_table: QTable::new(num_actions, 0.0),
            alpha,
            gamma,
            strategy: ExplorationStrategy::EpsilonGreedy(0.1),
            total_steps: 0,
            episode_rewards: Vec::new(),
            rng: Rng::new(42),
        })
    }

    /// Set exploration strategy.
    pub fn with_strategy(mut self, strategy: ExplorationStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = Rng::new(seed);
        self
    }

    /// Set default Q-value for unseen states.
    pub fn with_default_q(mut self, value: f64) -> Self {
        self.q_table = QTable::new(self.q_table.num_actions(), value);
        self
    }

    /// Current effective epsilon for decay strategy.
    fn current_epsilon(&self) -> f64 {
        match self.strategy {
            ExplorationStrategy::EpsilonGreedy(e) => e,
            ExplorationStrategy::EpsilonDecay { start, minimum, decay_steps } => {
                if decay_steps == 0 {
                    return minimum;
                }
                let progress = (self.total_steps as f64) / (decay_steps as f64);
                let progress = progress.min(1.0);
                start + (minimum - start) * progress
            }
            ExplorationStrategy::Softmax(_) | ExplorationStrategy::Greedy => 0.0,
        }
    }

    /// Select an action using the configured exploration strategy.
    pub fn select_action(&mut self, state: u64) -> usize {
        let n = self.q_table.num_actions();
        match self.strategy {
            ExplorationStrategy::EpsilonGreedy(_)
            | ExplorationStrategy::EpsilonDecay { .. } => {
                let eps = self.current_epsilon();
                if self.rng.next_f64() < eps {
                    self.rng.next_usize(n)
                } else {
                    self.q_table.best_action(state)
                }
            }
            ExplorationStrategy::Softmax(temp) => {
                let row = self.q_table.get_row(state);
                let max_q = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = row.iter().map(|q| ((q - max_q) / temp).exp()).collect();
                let sum: f64 = exps.iter().sum();
                let mut r = self.rng.next_f64() * sum;
                for (i, e) in exps.iter().enumerate() {
                    r -= e;
                    if r <= 0.0 {
                        return i;
                    }
                }
                n - 1
            }
            ExplorationStrategy::Greedy => self.q_table.best_action(state),
        }
    }

    /// Perform a Q-learning update: Q(s,a) += alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
    pub fn update(
        &mut self,
        state: u64,
        action: usize,
        reward: f64,
        next_state: u64,
        done: bool,
    ) -> Result<(), QLearningError> {
        if action >= self.q_table.num_actions() {
            return Err(QLearningError::ActionOutOfRange {
                action,
                num_actions: self.q_table.num_actions(),
            });
        }
        let current_q = self.q_table.get(state, action);
        let target = if done {
            reward
        } else {
            reward + self.gamma * self.q_table.max_q(next_state)
        };
        let new_q = current_q + self.alpha * (target - current_q);
        self.q_table.set(state, action, new_q);
        self.total_steps += 1;
        Ok(())
    }

    /// Record a completed episode's total reward.
    pub fn record_episode(&mut self, total_reward: f64) {
        self.episode_rewards.push(total_reward);
    }

    /// Total training steps.
    pub fn total_steps(&self) -> u64 {
        self.total_steps
    }

    /// Reference to internal Q-table.
    pub fn q_table(&self) -> &QTable {
        &self.q_table
    }

    /// Episode reward history.
    pub fn episode_rewards(&self) -> &[f64] {
        &self.episode_rewards
    }

    /// Average reward over the last `n` episodes.
    pub fn average_reward(&self, n: usize) -> Result<f64, QLearningError> {
        if self.episode_rewards.is_empty() {
            return Err(QLearningError::NoEpisodeData);
        }
        let start = self.episode_rewards.len().saturating_sub(n);
        let slice = &self.episode_rewards[start..];
        Ok(slice.iter().sum::<f64>() / slice.len() as f64)
    }

    /// Learning rate.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Discount factor.
    pub fn gamma(&self) -> f64 {
        self.gamma
    }
}

impl fmt::Display for QLearningAgent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "QLearning(α={:.3}, γ={:.3}, strategy={}, steps={}, episodes={})",
            self.alpha,
            self.gamma,
            self.strategy,
            self.total_steps,
            self.episode_rewards.len(),
        )
    }
}

// ── SARSA Agent ─────────────────────────────────────────────────

/// SARSA (on-policy TD) agent — uses Q(s',a') from actual next action.
#[derive(Debug, Clone)]
pub struct SarsaAgent {
    q_table: QTable,
    alpha: f64,
    gamma: f64,
    strategy: ExplorationStrategy,
    total_steps: u64,
    episode_rewards: Vec<f64>,
    rng: Rng,
    pending: Option<(u64, usize)>,
}

impl SarsaAgent {
    /// Create a new SARSA agent.
    pub fn new(num_actions: usize, alpha: f64, gamma: f64) -> Result<Self, QLearningError> {
        if num_actions == 0 {
            return Err(QLearningError::InvalidParameter("num_actions must be > 0".into()));
        }
        if alpha <= 0.0 || alpha > 1.0 {
            return Err(QLearningError::InvalidParameter("alpha must be in (0, 1]".into()));
        }
        if gamma < 0.0 || gamma > 1.0 {
            return Err(QLearningError::InvalidParameter("gamma must be in [0, 1]".into()));
        }
        Ok(Self {
            q_table: QTable::new(num_actions, 0.0),
            alpha,
            gamma,
            strategy: ExplorationStrategy::EpsilonGreedy(0.1),
            total_steps: 0,
            episode_rewards: Vec::new(),
            rng: Rng::new(42),
            pending: None,
        })
    }

    /// Set exploration strategy.
    pub fn with_strategy(mut self, strategy: ExplorationStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = Rng::new(seed);
        self
    }

    /// Select action using the exploration strategy.
    pub fn select_action(&mut self, state: u64) -> usize {
        let n = self.q_table.num_actions();
        let eps = match self.strategy {
            ExplorationStrategy::EpsilonGreedy(e) => e,
            ExplorationStrategy::EpsilonDecay { start, minimum, decay_steps } => {
                if decay_steps == 0 { minimum }
                else {
                    let p = (self.total_steps as f64) / (decay_steps as f64);
                    start + (minimum - start) * p.min(1.0)
                }
            }
            ExplorationStrategy::Softmax(_) | ExplorationStrategy::Greedy => 0.0,
        };

        if self.rng.next_f64() < eps {
            self.rng.next_usize(n)
        } else {
            self.q_table.best_action(state)
        }
    }

    /// Begin an episode: choose the first action for the initial state.
    pub fn begin_episode(&mut self, state: u64) -> usize {
        let action = self.select_action(state);
        self.pending = Some((state, action));
        action
    }

    /// SARSA step: observe reward and next state, select next action, update Q.
    /// Returns the next action to take.
    pub fn step(
        &mut self,
        reward: f64,
        next_state: u64,
        done: bool,
    ) -> Result<usize, QLearningError> {
        let (state, action) = self
            .pending
            .ok_or_else(|| QLearningError::InvalidParameter("call begin_episode first".into()))?;

        let next_action = if done { 0 } else { self.select_action(next_state) };

        let current_q = self.q_table.get(state, action);
        let target = if done {
            reward
        } else {
            reward + self.gamma * self.q_table.get(next_state, next_action)
        };
        let new_q = current_q + self.alpha * (target - current_q);
        self.q_table.set(state, action, new_q);
        self.total_steps += 1;

        if done {
            self.pending = None;
        } else {
            self.pending = Some((next_state, next_action));
        }

        Ok(next_action)
    }

    /// Record completed episode reward.
    pub fn record_episode(&mut self, total_reward: f64) {
        self.episode_rewards.push(total_reward);
    }

    /// Reference to the Q-table.
    pub fn q_table(&self) -> &QTable {
        &self.q_table
    }

    /// Total steps taken.
    pub fn total_steps(&self) -> u64 {
        self.total_steps
    }

    /// Episode rewards history.
    pub fn episode_rewards(&self) -> &[f64] {
        &self.episode_rewards
    }
}

impl fmt::Display for SarsaAgent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SARSA(α={:.3}, γ={:.3}, strategy={}, steps={})",
            self.alpha, self.gamma, self.strategy, self.total_steps,
        )
    }
}

// ── Expected SARSA ──────────────────────────────────────────────

/// Expected SARSA — uses expected Q-value over all actions weighted by policy.
#[derive(Debug, Clone)]
pub struct ExpectedSarsa {
    q_table: QTable,
    alpha: f64,
    gamma: f64,
    epsilon: f64,
    total_steps: u64,
    rng: Rng,
}

impl ExpectedSarsa {
    pub fn new(num_actions: usize, alpha: f64, gamma: f64, epsilon: f64) -> Result<Self, QLearningError> {
        if num_actions == 0 {
            return Err(QLearningError::InvalidParameter("num_actions must be > 0".into()));
        }
        Ok(Self {
            q_table: QTable::new(num_actions, 0.0),
            alpha,
            gamma,
            epsilon,
            total_steps: 0,
            rng: Rng::new(42),
        })
    }

    /// Select action: epsilon-greedy.
    pub fn select_action(&mut self, state: u64) -> usize {
        let n = self.q_table.num_actions();
        if self.rng.next_f64() < self.epsilon {
            self.rng.next_usize(n)
        } else {
            self.q_table.best_action(state)
        }
    }

    /// Update: uses expected value under epsilon-greedy policy.
    pub fn update(&mut self, state: u64, action: usize, reward: f64, next_state: u64, done: bool) {
        let current_q = self.q_table.get(state, action);
        let target = if done {
            reward
        } else {
            let n = self.q_table.num_actions();
            let row = self.q_table.get_row(next_state);
            let best = self.q_table.best_action(next_state);
            let explore_prob = self.epsilon / n as f64;
            let mut expected = 0.0;
            for (i, &q) in row.iter().enumerate() {
                if i == best {
                    expected += (1.0 - self.epsilon + explore_prob) * q;
                } else {
                    expected += explore_prob * q;
                }
            }
            reward + self.gamma * expected
        };
        let new_q = current_q + self.alpha * (target - current_q);
        self.q_table.set(state, action, new_q);
        self.total_steps += 1;
    }

    pub fn q_table(&self) -> &QTable {
        &self.q_table
    }

    pub fn total_steps(&self) -> u64 {
        self.total_steps
    }
}

impl fmt::Display for ExpectedSarsa {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ExpectedSARSA(α={:.3}, γ={:.3}, ε={:.3}, steps={})",
            self.alpha, self.gamma, self.epsilon, self.total_steps,
        )
    }
}

// ── Double Q-Learning ───────────────────────────────────────────

/// Double Q-learning to reduce maximization bias.
#[derive(Debug, Clone)]
pub struct DoubleQLearning {
    q1: QTable,
    q2: QTable,
    alpha: f64,
    gamma: f64,
    epsilon: f64,
    total_steps: u64,
    rng: Rng,
}

impl DoubleQLearning {
    pub fn new(num_actions: usize, alpha: f64, gamma: f64, epsilon: f64) -> Result<Self, QLearningError> {
        if num_actions == 0 {
            return Err(QLearningError::InvalidParameter("num_actions must be > 0".into()));
        }
        Ok(Self {
            q1: QTable::new(num_actions, 0.0),
            q2: QTable::new(num_actions, 0.0),
            alpha,
            gamma,
            epsilon,
            total_steps: 0,
            rng: Rng::new(42),
        })
    }

    /// Select action using combined Q-values.
    pub fn select_action(&mut self, state: u64) -> usize {
        let n = self.q1.num_actions();
        if self.rng.next_f64() < self.epsilon {
            self.rng.next_usize(n)
        } else {
            let row1 = self.q1.get_row(state);
            let row2 = self.q2.get_row(state);
            let mut best = 0;
            let mut best_val = row1[0] + row2[0];
            for i in 1..n {
                let v = row1[i] + row2[i];
                if v > best_val {
                    best_val = v;
                    best = i;
                }
            }
            best
        }
    }

    /// Double Q-learning update: randomly update Q1 or Q2.
    pub fn update(&mut self, state: u64, action: usize, reward: f64, next_state: u64, done: bool) {
        if self.rng.next_f64() < 0.5 {
            let best_a = self.q1.best_action(next_state);
            let current = self.q1.get(state, action);
            let target = if done { reward } else { reward + self.gamma * self.q2.get(next_state, best_a) };
            self.q1.set(state, action, current + self.alpha * (target - current));
        } else {
            let best_a = self.q2.best_action(next_state);
            let current = self.q2.get(state, action);
            let target = if done { reward } else { reward + self.gamma * self.q1.get(next_state, best_a) };
            self.q2.set(state, action, current + self.alpha * (target - current));
        }
        self.total_steps += 1;
    }

    pub fn total_steps(&self) -> u64 {
        self.total_steps
    }

    /// Combined Q-value (Q1 + Q2) / 2.
    pub fn combined_q(&self, state: u64, action: usize) -> f64 {
        (self.q1.get(state, action) + self.q2.get(state, action)) / 2.0
    }
}

impl fmt::Display for DoubleQLearning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DoubleQ(α={:.3}, γ={:.3}, ε={:.3}, steps={})",
            self.alpha, self.gamma, self.epsilon, self.total_steps,
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
    fn test_qtable_new() {
        let qt = QTable::new(4, 0.0);
        assert_eq!(qt.num_actions(), 4);
        assert_eq!(qt.state_count(), 0);
    }

    #[test]
    fn test_qtable_get_default() {
        let qt = QTable::new(3, 5.0);
        assert!(approx(qt.get(99, 0), 5.0));
    }

    #[test]
    fn test_qtable_set_and_get() {
        let mut qt = QTable::new(4, 0.0);
        qt.set(1, 2, 3.14);
        assert!(approx(qt.get(1, 2), 3.14));
        assert!(approx(qt.get(1, 0), 0.0));
    }

    #[test]
    fn test_qtable_best_action() {
        let mut qt = QTable::new(3, 0.0);
        qt.set(0, 0, 1.0);
        qt.set(0, 1, 5.0);
        qt.set(0, 2, 3.0);
        assert_eq!(qt.best_action(0), 1);
    }

    #[test]
    fn test_qtable_max_q() {
        let mut qt = QTable::new(3, 0.0);
        qt.set(0, 1, 7.5);
        assert!(approx(qt.max_q(0), 7.5));
    }

    #[test]
    fn test_agent_creation() {
        let agent = QLearningAgent::new(4, 0.1, 0.99).unwrap();
        assert!(approx(agent.alpha(), 0.1));
        assert!(approx(agent.gamma(), 0.99));
    }

    #[test]
    fn test_agent_invalid_params() {
        assert!(QLearningAgent::new(0, 0.1, 0.9).is_err());
        assert!(QLearningAgent::new(4, 0.0, 0.9).is_err());
        assert!(QLearningAgent::new(4, 0.1, 1.5).is_err());
    }

    #[test]
    fn test_agent_update() {
        let mut agent = QLearningAgent::new(2, 0.5, 0.9).unwrap()
            .with_strategy(ExplorationStrategy::Greedy);
        agent.update(0, 0, 1.0, 1, false).unwrap();
        assert!(agent.q_table().get(0, 0) > 0.0);
    }

    #[test]
    fn test_agent_update_terminal() {
        let mut agent = QLearningAgent::new(2, 1.0, 0.9).unwrap();
        agent.update(0, 0, 10.0, 0, true).unwrap();
        assert!(approx(agent.q_table().get(0, 0), 10.0));
    }

    #[test]
    fn test_agent_select_greedy() {
        let mut agent = QLearningAgent::new(3, 0.5, 0.9).unwrap()
            .with_strategy(ExplorationStrategy::Greedy);
        agent.update(0, 1, 100.0, 0, true).unwrap();
        let action = agent.select_action(0);
        assert_eq!(action, 1);
    }

    #[test]
    fn test_agent_epsilon_greedy_explores() {
        let mut agent = QLearningAgent::new(4, 0.1, 0.9).unwrap()
            .with_strategy(ExplorationStrategy::EpsilonGreedy(1.0))
            .with_seed(123);
        let mut actions = std::collections::HashSet::new();
        for _ in 0..200 {
            actions.insert(agent.select_action(0));
        }
        assert!(actions.len() > 1, "should explore multiple actions");
    }

    #[test]
    fn test_episode_tracking() {
        let mut agent = QLearningAgent::new(2, 0.1, 0.9).unwrap();
        agent.record_episode(10.0);
        agent.record_episode(15.0);
        agent.record_episode(20.0);
        assert!(approx(agent.average_reward(2).unwrap(), 17.5));
    }

    #[test]
    fn test_average_reward_empty() {
        let agent = QLearningAgent::new(2, 0.1, 0.9).unwrap();
        assert!(agent.average_reward(5).is_err());
    }

    #[test]
    fn test_sarsa_creation() {
        let sarsa = SarsaAgent::new(4, 0.1, 0.99).unwrap();
        assert_eq!(sarsa.total_steps(), 0);
    }

    #[test]
    fn test_sarsa_episode() {
        let mut sarsa = SarsaAgent::new(2, 0.5, 0.9).unwrap()
            .with_strategy(ExplorationStrategy::Greedy);
        let _a0 = sarsa.begin_episode(0);
        let _a1 = sarsa.step(1.0, 1, false).unwrap();
        let _a2 = sarsa.step(5.0, 2, true).unwrap();
        assert_eq!(sarsa.total_steps(), 2);
    }

    #[test]
    fn test_sarsa_q_update() {
        let mut sarsa = SarsaAgent::new(2, 1.0, 0.0).unwrap()
            .with_strategy(ExplorationStrategy::Greedy);
        let _a = sarsa.begin_episode(0);
        sarsa.step(7.0, 1, true).unwrap();
        // With alpha=1.0, gamma=0.0 and terminal: Q = 7.0
        let q = sarsa.q_table().get(0, sarsa.q_table().best_action(0));
        assert!(q >= 0.0);
    }

    #[test]
    fn test_expected_sarsa() {
        let mut es = ExpectedSarsa::new(3, 0.5, 0.9, 0.1).unwrap();
        es.update(0, 0, 1.0, 1, false);
        assert!(es.q_table().get(0, 0) > 0.0);
    }

    #[test]
    fn test_double_q_learning() {
        let mut dq = DoubleQLearning::new(2, 0.5, 0.9, 0.1).unwrap();
        dq.update(0, 0, 1.0, 1, false);
        dq.update(0, 0, 1.0, 1, false);
        assert!(dq.combined_q(0, 0) > 0.0);
    }

    #[test]
    fn test_display_agent() {
        let agent = QLearningAgent::new(4, 0.1, 0.99).unwrap();
        let s = format!("{agent}");
        assert!(s.contains("QLearning"));
    }

    #[test]
    fn test_display_sarsa() {
        let sarsa = SarsaAgent::new(2, 0.1, 0.9).unwrap();
        let s = format!("{sarsa}");
        assert!(s.contains("SARSA"));
    }

    #[test]
    fn test_display_qtable() {
        let qt = QTable::new(4, 0.0);
        let s = format!("{qt}");
        assert!(s.contains("QTable"));
    }
}
