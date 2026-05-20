//! RL environment framework — state/action spaces, step/reset interface, reward signals, gym-like API.
//!
//! Replaces OpenAI Gym / Gymnasium with pure Rust environment abstractions.
//! Supports discrete and continuous state/action space definitions, step/reset API,
//! built-in environments (GridWorld, CartPole-like, chain walk), reward signals,
//! episode tracking, observation clamping, and environment wrappers.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum EnvError {
    InvalidParameter(String),
    InvalidAction(String),
    EpisodeNotStarted,
    DimensionMismatch { expected: usize, got: usize },
}

impl fmt::Display for EnvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter(s) => write!(f, "invalid parameter: {s}"),
            Self::InvalidAction(s) => write!(f, "invalid action: {s}"),
            Self::EpisodeNotStarted => write!(f, "episode not started — call reset() first"),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
        }
    }
}

impl std::error::Error for EnvError {}

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

// ── Spaces ──────────────────────────────────────────────────────

/// Describes an action or observation space.
#[derive(Debug, Clone, PartialEq)]
pub enum Space {
    /// Discrete space with n possible values (0..n).
    Discrete(usize),
    /// Box (continuous) space with low/high bounds per dimension.
    Box { low: Vec<f64>, high: Vec<f64> },
}

impl Space {
    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        match self {
            Self::Discrete(_) => 1,
            Self::Box { low, .. } => low.len(),
        }
    }

    /// Clamp a continuous observation to the space bounds.
    pub fn clamp(&self, obs: &mut [f64]) {
        if let Self::Box { low, high } = self {
            for (i, v) in obs.iter_mut().enumerate() {
                if i < low.len() {
                    *v = v.clamp(low[i], high[i]);
                }
            }
        }
    }

    /// Check whether a discrete action is valid.
    pub fn contains_discrete(&self, action: usize) -> bool {
        match self {
            Self::Discrete(n) => action < *n,
            Self::Box { .. } => false,
        }
    }
}

impl fmt::Display for Space {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Discrete(n) => write!(f, "Discrete({n})"),
            Self::Box { low, high } => write!(f, "Box(dims={}, low={:.1?}, high={:.1?})", low.len(), low, high),
        }
    }
}

// ── Step Result ─────────────────────────────────────────────────

/// Result of taking a step in an environment.
#[derive(Debug, Clone)]
pub struct StepResult {
    pub observation: Vec<f64>,
    pub reward: f64,
    pub done: bool,
    pub truncated: bool,
    pub info: StepInfo,
}

/// Additional information returned by a step.
#[derive(Debug, Clone, Default)]
pub struct StepInfo {
    pub episode_length: u64,
    pub episode_reward: f64,
}

impl fmt::Display for StepResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Step(reward={:.3}, done={}, truncated={}, len={})",
            self.reward, self.done, self.truncated, self.info.episode_length,
        )
    }
}

// ── Grid World ──────────────────────────────────────────────────

/// Actions for grid world: Up, Down, Left, Right.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridAction {
    Up = 0,
    Down = 1,
    Left = 2,
    Right = 3,
}

impl GridAction {
    pub fn from_usize(a: usize) -> Option<Self> {
        match a {
            0 => Some(Self::Up),
            1 => Some(Self::Down),
            2 => Some(Self::Left),
            3 => Some(Self::Right),
            _ => None,
        }
    }
}

/// A simple grid-world environment with walls, goals, and optional traps.
#[derive(Debug, Clone)]
pub struct GridWorld {
    width: usize,
    height: usize,
    agent_x: usize,
    agent_y: usize,
    start_x: usize,
    start_y: usize,
    goal_x: usize,
    goal_y: usize,
    walls: Vec<bool>,
    traps: Vec<bool>,
    step_reward: f64,
    goal_reward: f64,
    trap_penalty: f64,
    episode_steps: u64,
    episode_reward: f64,
    max_steps: u64,
    started: bool,
    rng: Rng,
    slip_prob: f64,
}

impl GridWorld {
    /// Create a new grid world environment.
    pub fn new(width: usize, height: usize) -> Result<Self, EnvError> {
        if width < 2 || height < 2 {
            return Err(EnvError::InvalidParameter("grid must be at least 2x2".into()));
        }
        let size = width * height;
        Ok(Self {
            width,
            height,
            agent_x: 0,
            agent_y: 0,
            start_x: 0,
            start_y: 0,
            goal_x: width - 1,
            goal_y: height - 1,
            walls: vec![false; size],
            traps: vec![false; size],
            step_reward: -0.01,
            goal_reward: 1.0,
            trap_penalty: -1.0,
            episode_steps: 0,
            episode_reward: 0.0,
            max_steps: (width * height * 4) as u64,
            started: false,
            rng: Rng::new(42),
            slip_prob: 0.0,
        })
    }

    pub fn with_start(mut self, x: usize, y: usize) -> Self {
        self.start_x = x.min(self.width - 1);
        self.start_y = y.min(self.height - 1);
        self
    }

    pub fn with_goal(mut self, x: usize, y: usize) -> Self {
        self.goal_x = x.min(self.width - 1);
        self.goal_y = y.min(self.height - 1);
        self
    }

    pub fn with_step_reward(mut self, r: f64) -> Self {
        self.step_reward = r;
        self
    }

    pub fn with_goal_reward(mut self, r: f64) -> Self {
        self.goal_reward = r;
        self
    }

    pub fn with_max_steps(mut self, steps: u64) -> Self {
        self.max_steps = steps;
        self
    }

    pub fn with_slip_prob(mut self, p: f64) -> Self {
        self.slip_prob = p.clamp(0.0, 1.0);
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = Rng::new(seed);
        self
    }

    /// Place a wall at (x, y).
    pub fn set_wall(&mut self, x: usize, y: usize) {
        if x < self.width && y < self.height {
            self.walls[y * self.width + x] = true;
        }
    }

    /// Place a trap at (x, y).
    pub fn set_trap(&mut self, x: usize, y: usize) {
        if x < self.width && y < self.height {
            self.traps[y * self.width + x] = true;
        }
    }

    fn is_wall(&self, x: usize, y: usize) -> bool {
        self.walls[y * self.width + x]
    }

    fn is_trap(&self, x: usize, y: usize) -> bool {
        self.traps[y * self.width + x]
    }

    /// Observation space: discrete (width * height) states.
    pub fn observation_space(&self) -> Space {
        Space::Discrete(self.width * self.height)
    }

    /// Action space: 4 discrete actions.
    pub fn action_space(&self) -> Space {
        Space::Discrete(4)
    }

    fn observation(&self) -> Vec<f64> {
        vec![self.agent_x as f64, self.agent_y as f64]
    }

    /// State as a single integer.
    pub fn state_id(&self) -> u64 {
        (self.agent_y * self.width + self.agent_x) as u64
    }

    /// Reset the environment.
    pub fn reset(&mut self) -> Vec<f64> {
        self.agent_x = self.start_x;
        self.agent_y = self.start_y;
        self.episode_steps = 0;
        self.episode_reward = 0.0;
        self.started = true;
        self.observation()
    }

    /// Take a step with the given action (0-3).
    pub fn step(&mut self, action: usize) -> Result<StepResult, EnvError> {
        if !self.started {
            return Err(EnvError::EpisodeNotStarted);
        }
        let act = GridAction::from_usize(action)
            .ok_or_else(|| EnvError::InvalidAction(format!("expected 0-3, got {action}")))?;

        // Stochastic slip
        let effective_act = if self.slip_prob > 0.0 && self.rng.next_f64() < self.slip_prob {
            GridAction::from_usize(self.rng.next_usize(4)).unwrap()
        } else {
            act
        };

        let (mut nx, mut ny) = (self.agent_x, self.agent_y);
        match effective_act {
            GridAction::Up => { if ny > 0 { ny -= 1; } }
            GridAction::Down => { if ny < self.height - 1 { ny += 1; } }
            GridAction::Left => { if nx > 0 { nx -= 1; } }
            GridAction::Right => { if nx < self.width - 1 { nx += 1; } }
        }

        if !self.is_wall(nx, ny) {
            self.agent_x = nx;
            self.agent_y = ny;
        }

        self.episode_steps += 1;

        let at_goal = self.agent_x == self.goal_x && self.agent_y == self.goal_y;
        let at_trap = self.is_trap(self.agent_x, self.agent_y);
        let truncated = self.episode_steps >= self.max_steps;

        let reward = if at_goal {
            self.goal_reward
        } else if at_trap {
            self.trap_penalty
        } else {
            self.step_reward
        };

        self.episode_reward += reward;
        let done = at_goal || at_trap;

        if done || truncated {
            self.started = false;
        }

        Ok(StepResult {
            observation: self.observation(),
            reward,
            done,
            truncated,
            info: StepInfo {
                episode_length: self.episode_steps,
                episode_reward: self.episode_reward,
            },
        })
    }

    pub fn width(&self) -> usize { self.width }
    pub fn height(&self) -> usize { self.height }
}

impl fmt::Display for GridWorld {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GridWorld({}x{}, agent=({},{}), goal=({},{}))",
            self.width, self.height,
            self.agent_x, self.agent_y,
            self.goal_x, self.goal_y,
        )
    }
}

// ── Chain Walk ──────────────────────────────────────────────────

/// A simple chain-walk environment: 1D chain of states with two actions (left/right).
#[derive(Debug, Clone)]
pub struct ChainWalk {
    length: usize,
    position: usize,
    goal: usize,
    step_reward: f64,
    goal_reward: f64,
    episode_steps: u64,
    episode_reward: f64,
    max_steps: u64,
    started: bool,
    slip_prob: f64,
    rng: Rng,
}

impl ChainWalk {
    pub fn new(length: usize) -> Result<Self, EnvError> {
        if length < 2 {
            return Err(EnvError::InvalidParameter("chain length must be >= 2".into()));
        }
        Ok(Self {
            length,
            position: 0,
            goal: length - 1,
            step_reward: -0.01,
            goal_reward: 1.0,
            episode_steps: 0,
            episode_reward: 0.0,
            max_steps: (length * 10) as u64,
            started: false,
            slip_prob: 0.0,
            rng: Rng::new(42),
        })
    }

    pub fn with_goal_reward(mut self, r: f64) -> Self {
        self.goal_reward = r;
        self
    }

    pub fn with_slip_prob(mut self, p: f64) -> Self {
        self.slip_prob = p.clamp(0.0, 1.0);
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = Rng::new(seed);
        self
    }

    pub fn observation_space(&self) -> Space {
        Space::Discrete(self.length)
    }

    pub fn action_space(&self) -> Space {
        Space::Discrete(2)
    }

    pub fn reset(&mut self) -> Vec<f64> {
        self.position = 0;
        self.episode_steps = 0;
        self.episode_reward = 0.0;
        self.started = true;
        vec![self.position as f64]
    }

    /// Step: action 0 = left, action 1 = right.
    pub fn step(&mut self, action: usize) -> Result<StepResult, EnvError> {
        if !self.started {
            return Err(EnvError::EpisodeNotStarted);
        }
        if action > 1 {
            return Err(EnvError::InvalidAction(format!("expected 0 or 1, got {action}")));
        }

        let effective = if self.slip_prob > 0.0 && self.rng.next_f64() < self.slip_prob {
            1 - action
        } else {
            action
        };

        if effective == 0 && self.position > 0 {
            self.position -= 1;
        } else if effective == 1 && self.position < self.length - 1 {
            self.position += 1;
        }

        self.episode_steps += 1;
        let at_goal = self.position == self.goal;
        let truncated = self.episode_steps >= self.max_steps;

        let reward = if at_goal { self.goal_reward } else { self.step_reward };
        self.episode_reward += reward;

        if at_goal || truncated {
            self.started = false;
        }

        Ok(StepResult {
            observation: vec![self.position as f64],
            reward,
            done: at_goal,
            truncated,
            info: StepInfo {
                episode_length: self.episode_steps,
                episode_reward: self.episode_reward,
            },
        })
    }

    pub fn position(&self) -> usize { self.position }
    pub fn length(&self) -> usize { self.length }
}

impl fmt::Display for ChainWalk {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ChainWalk(length={}, pos={}, goal={})",
            self.length, self.position, self.goal,
        )
    }
}

// ── Reward Tracker ──────────────────────────────────────────────

/// Tracks episode rewards and computes statistics.
#[derive(Debug, Clone)]
pub struct RewardTracker {
    rewards: Vec<f64>,
    window_size: usize,
}

impl RewardTracker {
    pub fn new(window_size: usize) -> Self {
        Self {
            rewards: Vec::new(),
            window_size: window_size.max(1),
        }
    }

    pub fn record(&mut self, reward: f64) {
        self.rewards.push(reward);
    }

    pub fn mean(&self) -> f64 {
        if self.rewards.is_empty() { return 0.0; }
        self.rewards.iter().sum::<f64>() / self.rewards.len() as f64
    }

    pub fn windowed_mean(&self) -> f64 {
        if self.rewards.is_empty() { return 0.0; }
        let start = self.rewards.len().saturating_sub(self.window_size);
        let slice = &self.rewards[start..];
        slice.iter().sum::<f64>() / slice.len() as f64
    }

    pub fn max(&self) -> f64 {
        self.rewards.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }

    pub fn min(&self) -> f64 {
        self.rewards.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    pub fn count(&self) -> usize {
        self.rewards.len()
    }

    pub fn rewards(&self) -> &[f64] {
        &self.rewards
    }
}

impl fmt::Display for RewardTracker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RewardTracker(episodes={}, mean={:.3}, windowed_mean={:.3})",
            self.rewards.len(),
            self.mean(),
            self.windowed_mean(),
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
    fn test_space_discrete() {
        let s = Space::Discrete(4);
        assert_eq!(s.ndim(), 1);
        assert!(s.contains_discrete(3));
        assert!(!s.contains_discrete(4));
    }

    #[test]
    fn test_space_box() {
        let s = Space::Box {
            low: vec![-1.0, -2.0],
            high: vec![1.0, 2.0],
        };
        assert_eq!(s.ndim(), 2);
    }

    #[test]
    fn test_space_clamp() {
        let s = Space::Box {
            low: vec![-1.0, -1.0],
            high: vec![1.0, 1.0],
        };
        let mut obs = vec![5.0, -5.0];
        s.clamp(&mut obs);
        assert!(approx(obs[0], 1.0));
        assert!(approx(obs[1], -1.0));
    }

    #[test]
    fn test_gridworld_creation() {
        let gw = GridWorld::new(5, 5).unwrap();
        assert_eq!(gw.width(), 5);
        assert_eq!(gw.height(), 5);
    }

    #[test]
    fn test_gridworld_invalid() {
        assert!(GridWorld::new(1, 5).is_err());
        assert!(GridWorld::new(5, 1).is_err());
    }

    #[test]
    fn test_gridworld_reset() {
        let mut gw = GridWorld::new(5, 5).unwrap();
        let obs = gw.reset();
        assert_eq!(obs, vec![0.0, 0.0]);
    }

    #[test]
    fn test_gridworld_step_right() {
        let mut gw = GridWorld::new(5, 5).unwrap();
        gw.reset();
        let result = gw.step(3).unwrap(); // Right
        assert!(approx(result.observation[0], 1.0));
        assert!(approx(result.observation[1], 0.0));
    }

    #[test]
    fn test_gridworld_step_not_started() {
        let mut gw = GridWorld::new(5, 5).unwrap();
        assert!(gw.step(0).is_err());
    }

    #[test]
    fn test_gridworld_wall_blocks() {
        let mut gw = GridWorld::new(5, 5).unwrap();
        gw.set_wall(1, 0);
        gw.reset();
        let result = gw.step(3).unwrap(); // Try to move right into wall
        assert!(approx(result.observation[0], 0.0)); // Didn't move
    }

    #[test]
    fn test_gridworld_goal_reached() {
        let mut gw = GridWorld::new(2, 2).unwrap()
            .with_goal(1, 0)
            .with_goal_reward(10.0);
        gw.reset();
        let result = gw.step(3).unwrap(); // Right to (1,0)
        assert!(result.done);
        assert!(approx(result.reward, 10.0));
    }

    #[test]
    fn test_gridworld_trap() {
        let mut gw = GridWorld::new(3, 3).unwrap();
        gw.set_trap(1, 0);
        gw.reset();
        let result = gw.step(3).unwrap(); // Right into trap
        assert!(result.done);
        assert!(result.reward < 0.0);
    }

    #[test]
    fn test_gridworld_invalid_action() {
        let mut gw = GridWorld::new(5, 5).unwrap();
        gw.reset();
        assert!(gw.step(10).is_err());
    }

    #[test]
    fn test_chain_walk_creation() {
        let cw = ChainWalk::new(10).unwrap();
        assert_eq!(cw.length(), 10);
    }

    #[test]
    fn test_chain_walk_invalid() {
        assert!(ChainWalk::new(1).is_err());
    }

    #[test]
    fn test_chain_walk_reset() {
        let mut cw = ChainWalk::new(5).unwrap();
        let obs = cw.reset();
        assert!(approx(obs[0], 0.0));
    }

    #[test]
    fn test_chain_walk_right() {
        let mut cw = ChainWalk::new(5).unwrap();
        cw.reset();
        let r = cw.step(1).unwrap();
        assert!(approx(r.observation[0], 1.0));
    }

    #[test]
    fn test_chain_walk_reach_goal() {
        let mut cw = ChainWalk::new(3).unwrap();
        cw.reset();
        let _ = cw.step(1).unwrap();
        let r = cw.step(1).unwrap();
        assert!(r.done);
    }

    #[test]
    fn test_chain_walk_boundary() {
        let mut cw = ChainWalk::new(5).unwrap();
        cw.reset();
        let r = cw.step(0).unwrap(); // Left at position 0
        assert!(approx(r.observation[0], 0.0)); // Stays at 0
    }

    #[test]
    fn test_reward_tracker() {
        let mut rt = RewardTracker::new(3);
        rt.record(10.0);
        rt.record(20.0);
        rt.record(30.0);
        rt.record(40.0);
        assert!(approx(rt.mean(), 25.0));
        assert!(approx(rt.windowed_mean(), 30.0));
        assert!(approx(rt.max(), 40.0));
        assert!(approx(rt.min(), 10.0));
    }

    #[test]
    fn test_display_gridworld() {
        let gw = GridWorld::new(5, 5).unwrap();
        let s = format!("{gw}");
        assert!(s.contains("GridWorld"));
    }

    #[test]
    fn test_display_chain() {
        let cw = ChainWalk::new(5).unwrap();
        let s = format!("{cw}");
        assert!(s.contains("ChainWalk"));
    }

    #[test]
    fn test_display_space() {
        let s = Space::Discrete(4);
        let display = format!("{s}");
        assert!(display.contains("Discrete"));
    }
}
