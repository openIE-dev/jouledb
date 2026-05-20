//! Multi-agent RL — independent learners, centralized training, communication channels, shared rewards.
//!
//! Replaces PettingZoo / RLlib multi-agent with pure Rust.
//! Supports independent Q-learners, centralized training with decentralized execution
//! (CTDE), inter-agent communication channels, shared and individual reward schemes,
//! team-vs-individual reward mixing, and multi-agent episode tracking.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum MultiAgentError {
    InvalidParameter(String),
    AgentNotFound(usize),
    ActionOutOfRange { agent: usize, action: usize, num_actions: usize },
    NoAgents,
    ChannelFull,
}

impl fmt::Display for MultiAgentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter(s) => write!(f, "invalid parameter: {s}"),
            Self::AgentNotFound(id) => write!(f, "agent {id} not found"),
            Self::ActionOutOfRange { agent, action, num_actions } => {
                write!(f, "agent {agent}: action {action} out of range ({num_actions} actions)")
            }
            Self::NoAgents => write!(f, "no agents configured"),
            Self::ChannelFull => write!(f, "communication channel is full"),
        }
    }
}

impl std::error::Error for MultiAgentError {}

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

// ── Q-Table (local) ─────────────────────────────────────────────

#[derive(Debug, Clone)]
struct QTable {
    table: HashMap<u64, Vec<f64>>,
    num_actions: usize,
}

impl QTable {
    fn new(num_actions: usize) -> Self {
        Self {
            table: HashMap::new(),
            num_actions,
        }
    }

    fn get(&self, state: u64, action: usize) -> f64 {
        self.table
            .get(&state)
            .map(|row| row[action.min(self.num_actions - 1)])
            .unwrap_or(0.0)
    }

    fn set(&mut self, state: u64, action: usize, value: f64) {
        let row = self
            .table
            .entry(state)
            .or_insert_with(|| vec![0.0; self.num_actions]);
        if action < self.num_actions {
            row[action] = value;
        }
    }

    fn best_action(&self, state: u64) -> usize {
        let row = self
            .table
            .get(&state)
            .cloned()
            .unwrap_or_else(|| vec![0.0; self.num_actions]);
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

    fn max_q(&self, state: u64) -> f64 {
        let row = self
            .table
            .get(&state)
            .cloned()
            .unwrap_or_else(|| vec![0.0; self.num_actions]);
        row.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }

    fn state_count(&self) -> usize {
        self.table.len()
    }
}

// ── Agent ───────────────────────────────────────────────────────

/// A single agent in a multi-agent system.
#[derive(Debug, Clone)]
pub struct Agent {
    id: usize,
    q_table: QTable,
    alpha: f64,
    gamma: f64,
    epsilon: f64,
    total_reward: f64,
    steps: u64,
    rng: Rng,
}

impl Agent {
    fn new(id: usize, num_actions: usize, alpha: f64, gamma: f64, epsilon: f64, seed: u64) -> Self {
        Self {
            id,
            q_table: QTable::new(num_actions),
            alpha,
            gamma,
            epsilon,
            total_reward: 0.0,
            steps: 0,
            rng: Rng::new(seed.wrapping_add(id as u64)),
        }
    }

    /// Epsilon-greedy action selection.
    fn select_action(&mut self, state: u64) -> usize {
        if self.rng.next_f64() < self.epsilon {
            self.rng.next_usize(self.q_table.num_actions)
        } else {
            self.q_table.best_action(state)
        }
    }

    /// Q-learning update.
    fn update(&mut self, state: u64, action: usize, reward: f64, next_state: u64, done: bool) {
        let current = self.q_table.get(state, action);
        let target = if done {
            reward
        } else {
            reward + self.gamma * self.q_table.max_q(next_state)
        };
        let new_q = current + self.alpha * (target - current);
        self.q_table.set(state, action, new_q);
        self.total_reward += reward;
        self.steps += 1;
    }
}

impl fmt::Display for Agent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Agent(id={}, states={}, reward={:.3}, steps={})",
            self.id,
            self.q_table.state_count(),
            self.total_reward,
            self.steps,
        )
    }
}

// ── Communication Channel ───────────────────────────────────────

/// A message in the communication channel.
#[derive(Debug, Clone)]
pub struct Message {
    pub sender: usize,
    pub content: Vec<f64>,
    pub timestamp: u64,
}

impl fmt::Display for Message {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Msg(from={}, len={}, t={})",
            self.sender,
            self.content.len(),
            self.timestamp,
        )
    }
}

/// Broadcast communication channel between agents.
#[derive(Debug, Clone)]
pub struct CommChannel {
    messages: Vec<Message>,
    capacity: usize,
    message_dim: usize,
    total_sent: u64,
}

impl CommChannel {
    pub fn new(capacity: usize, message_dim: usize) -> Self {
        Self {
            messages: Vec::new(),
            capacity,
            message_dim,
            total_sent: 0,
        }
    }

    /// Send a message (broadcast to all agents).
    pub fn send(&mut self, sender: usize, content: Vec<f64>) -> Result<(), MultiAgentError> {
        if self.messages.len() >= self.capacity {
            return Err(MultiAgentError::ChannelFull);
        }
        let mut msg_content = content;
        msg_content.resize(self.message_dim, 0.0);
        self.messages.push(Message {
            sender,
            content: msg_content,
            timestamp: self.total_sent,
        });
        self.total_sent += 1;
        Ok(())
    }

    /// Receive all messages (except own).
    pub fn receive(&self, receiver: usize) -> Vec<&Message> {
        self.messages
            .iter()
            .filter(|m| m.sender != receiver)
            .collect()
    }

    /// Receive all messages from a specific sender.
    pub fn receive_from(&self, sender: usize) -> Vec<&Message> {
        self.messages.iter().filter(|m| m.sender == sender).collect()
    }

    /// Clear all messages (call between steps).
    pub fn clear(&mut self) {
        self.messages.clear();
    }

    pub fn pending_count(&self) -> usize {
        self.messages.len()
    }

    pub fn total_sent(&self) -> u64 {
        self.total_sent
    }
}

impl fmt::Display for CommChannel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CommChannel(pending={}, capacity={}, dim={}, total_sent={})",
            self.messages.len(),
            self.capacity,
            self.message_dim,
            self.total_sent,
        )
    }
}

// ── Reward Scheme ───────────────────────────────────────────────

/// How rewards are distributed among agents.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RewardScheme {
    /// Each agent gets only its own reward.
    Individual,
    /// All agents share the team's total reward equally.
    SharedEqual,
    /// Weighted mix of individual and team reward.
    Mixed { team_weight: f64 },
}

impl fmt::Display for RewardScheme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Individual => write!(f, "Individual"),
            Self::SharedEqual => write!(f, "SharedEqual"),
            Self::Mixed { team_weight } => write!(f, "Mixed(team_w={team_weight:.3})"),
        }
    }
}

// ── Multi-Agent System ──────────────────────────────────────────

/// Multi-agent reinforcement learning system.
#[derive(Debug, Clone)]
pub struct MultiAgentSystem {
    agents: Vec<Agent>,
    num_actions: usize,
    reward_scheme: RewardScheme,
    channel: CommChannel,
    episode_count: u64,
    total_steps: u64,
    team_rewards: Vec<f64>,
}

impl MultiAgentSystem {
    /// Create a multi-agent system.
    pub fn new(
        num_agents: usize,
        num_actions: usize,
        alpha: f64,
        gamma: f64,
        epsilon: f64,
    ) -> Result<Self, MultiAgentError> {
        if num_agents == 0 {
            return Err(MultiAgentError::NoAgents);
        }
        if num_actions == 0 {
            return Err(MultiAgentError::InvalidParameter("num_actions must be > 0".into()));
        }
        let agents: Vec<Agent> = (0..num_agents)
            .map(|i| Agent::new(i, num_actions, alpha, gamma, epsilon, 42))
            .collect();
        Ok(Self {
            agents,
            num_actions,
            reward_scheme: RewardScheme::Individual,
            channel: CommChannel::new(num_agents * 10, 4),
            episode_count: 0,
            total_steps: 0,
            team_rewards: Vec::new(),
        })
    }

    pub fn with_reward_scheme(mut self, scheme: RewardScheme) -> Self {
        self.reward_scheme = scheme;
        self
    }

    pub fn with_channel_capacity(mut self, capacity: usize, msg_dim: usize) -> Self {
        self.channel = CommChannel::new(capacity, msg_dim);
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        for (i, agent) in self.agents.iter_mut().enumerate() {
            agent.rng = Rng::new(seed.wrapping_add(i as u64));
        }
        self
    }

    /// Number of agents.
    pub fn num_agents(&self) -> usize {
        self.agents.len()
    }

    /// Select actions for all agents given their individual states.
    pub fn select_actions(&mut self, states: &[u64]) -> Result<Vec<usize>, MultiAgentError> {
        if states.len() != self.agents.len() {
            return Err(MultiAgentError::InvalidParameter(format!(
                "expected {} states, got {}",
                self.agents.len(),
                states.len(),
            )));
        }
        let actions: Vec<usize> = self
            .agents
            .iter_mut()
            .zip(states.iter())
            .map(|(agent, &state)| agent.select_action(state))
            .collect();
        Ok(actions)
    }

    /// Apply reward distribution according to the reward scheme.
    fn distribute_rewards(&self, individual_rewards: &[f64]) -> Vec<f64> {
        let n = individual_rewards.len();
        match self.reward_scheme {
            RewardScheme::Individual => individual_rewards.to_vec(),
            RewardScheme::SharedEqual => {
                let total: f64 = individual_rewards.iter().sum();
                let share = total / n as f64;
                vec![share; n]
            }
            RewardScheme::Mixed { team_weight } => {
                let total: f64 = individual_rewards.iter().sum();
                let team_share = total / n as f64;
                individual_rewards
                    .iter()
                    .map(|r| (1.0 - team_weight) * r + team_weight * team_share)
                    .collect()
            }
        }
    }

    /// Update all agents with their transitions.
    pub fn update(
        &mut self,
        states: &[u64],
        actions: &[usize],
        individual_rewards: &[f64],
        next_states: &[u64],
        dones: &[bool],
    ) -> Result<(), MultiAgentError> {
        let n = self.agents.len();
        if states.len() != n
            || actions.len() != n
            || individual_rewards.len() != n
            || next_states.len() != n
            || dones.len() != n
        {
            return Err(MultiAgentError::InvalidParameter(
                "all arrays must have length equal to num_agents".into(),
            ));
        }

        let rewards = self.distribute_rewards(individual_rewards);

        for i in 0..n {
            if actions[i] >= self.num_actions {
                return Err(MultiAgentError::ActionOutOfRange {
                    agent: i,
                    action: actions[i],
                    num_actions: self.num_actions,
                });
            }
            self.agents[i].update(states[i], actions[i], rewards[i], next_states[i], dones[i]);
        }

        self.total_steps += 1;
        Ok(())
    }

    /// Record an episode, tracking team reward.
    pub fn record_episode(&mut self) {
        let team_reward: f64 = self.agents.iter().map(|a| a.total_reward).sum();
        self.team_rewards.push(team_reward);
        self.episode_count += 1;
    }

    /// Reset agent reward accumulators for a new episode.
    pub fn reset_episode(&mut self) {
        for agent in &mut self.agents {
            agent.total_reward = 0.0;
        }
    }

    /// Send a message from an agent.
    pub fn send_message(
        &mut self,
        sender: usize,
        content: Vec<f64>,
    ) -> Result<(), MultiAgentError> {
        if sender >= self.agents.len() {
            return Err(MultiAgentError::AgentNotFound(sender));
        }
        self.channel.send(sender, content)
    }

    /// Get messages for an agent (excluding own messages).
    pub fn get_messages(&self, receiver: usize) -> Vec<&Message> {
        self.channel.receive(receiver)
    }

    /// Clear communication channel.
    pub fn clear_messages(&mut self) {
        self.channel.clear();
    }

    /// Get agent by ID.
    pub fn agent(&self, id: usize) -> Result<&Agent, MultiAgentError> {
        self.agents.get(id).ok_or(MultiAgentError::AgentNotFound(id))
    }

    /// Best action for a given agent and state.
    pub fn best_action(&self, agent_id: usize, state: u64) -> Result<usize, MultiAgentError> {
        let agent = self.agents.get(agent_id).ok_or(MultiAgentError::AgentNotFound(agent_id))?;
        Ok(agent.q_table.best_action(state))
    }

    pub fn total_steps(&self) -> u64 {
        self.total_steps
    }

    pub fn episode_count(&self) -> u64 {
        self.episode_count
    }

    pub fn team_rewards(&self) -> &[f64] {
        &self.team_rewards
    }

    pub fn channel(&self) -> &CommChannel {
        &self.channel
    }

    /// Average team reward over last n episodes.
    pub fn average_team_reward(&self, last_n: usize) -> f64 {
        if self.team_rewards.is_empty() {
            return 0.0;
        }
        let start = self.team_rewards.len().saturating_sub(last_n);
        let slice = &self.team_rewards[start..];
        slice.iter().sum::<f64>() / slice.len() as f64
    }
}

impl fmt::Display for MultiAgentSystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MultiAgentSystem(agents={}, actions={}, scheme={}, episodes={}, steps={})",
            self.agents.len(),
            self.num_actions,
            self.reward_scheme,
            self.episode_count,
            self.total_steps,
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
    fn test_system_creation() {
        let sys = MultiAgentSystem::new(3, 4, 0.1, 0.99, 0.1).unwrap();
        assert_eq!(sys.num_agents(), 3);
    }

    #[test]
    fn test_system_no_agents() {
        assert!(MultiAgentSystem::new(0, 4, 0.1, 0.99, 0.1).is_err());
    }

    #[test]
    fn test_system_no_actions() {
        assert!(MultiAgentSystem::new(3, 0, 0.1, 0.99, 0.1).is_err());
    }

    #[test]
    fn test_select_actions() {
        let mut sys = MultiAgentSystem::new(3, 4, 0.1, 0.99, 0.0)
            .unwrap()
            .with_seed(42);
        let actions = sys.select_actions(&[0, 0, 0]).unwrap();
        assert_eq!(actions.len(), 3);
        for &a in &actions {
            assert!(a < 4);
        }
    }

    #[test]
    fn test_select_actions_wrong_count() {
        let mut sys = MultiAgentSystem::new(3, 4, 0.1, 0.99, 0.1).unwrap();
        assert!(sys.select_actions(&[0, 0]).is_err());
    }

    #[test]
    fn test_individual_rewards() {
        let mut sys = MultiAgentSystem::new(2, 2, 0.5, 0.9, 0.0)
            .unwrap()
            .with_reward_scheme(RewardScheme::Individual);
        sys.update(&[0, 0], &[0, 1], &[1.0, 2.0], &[1, 1], &[false, false])
            .unwrap();
        let a0 = sys.agent(0).unwrap();
        let a1 = sys.agent(1).unwrap();
        assert!(approx(a0.total_reward, 1.0));
        assert!(approx(a1.total_reward, 2.0));
    }

    #[test]
    fn test_shared_equal_rewards() {
        let mut sys = MultiAgentSystem::new(2, 2, 0.5, 0.9, 0.0)
            .unwrap()
            .with_reward_scheme(RewardScheme::SharedEqual);
        sys.update(&[0, 0], &[0, 1], &[1.0, 3.0], &[1, 1], &[false, false])
            .unwrap();
        let a0 = sys.agent(0).unwrap();
        let a1 = sys.agent(1).unwrap();
        assert!(approx(a0.total_reward, 2.0));
        assert!(approx(a1.total_reward, 2.0));
    }

    #[test]
    fn test_mixed_rewards() {
        let mut sys = MultiAgentSystem::new(2, 2, 0.5, 0.9, 0.0)
            .unwrap()
            .with_reward_scheme(RewardScheme::Mixed { team_weight: 0.5 });
        sys.update(&[0, 0], &[0, 1], &[0.0, 4.0], &[1, 1], &[false, false])
            .unwrap();
        let a0 = sys.agent(0).unwrap();
        let a1 = sys.agent(1).unwrap();
        // Agent 0: 0.5*0.0 + 0.5*2.0 = 1.0
        // Agent 1: 0.5*4.0 + 0.5*2.0 = 3.0
        assert!(approx(a0.total_reward, 1.0));
        assert!(approx(a1.total_reward, 3.0));
    }

    #[test]
    fn test_comm_channel_send_receive() {
        let mut sys = MultiAgentSystem::new(3, 2, 0.1, 0.9, 0.1)
            .unwrap()
            .with_channel_capacity(20, 3);
        sys.send_message(0, vec![1.0, 2.0, 3.0]).unwrap();
        sys.send_message(1, vec![4.0, 5.0, 6.0]).unwrap();

        let msgs_for_2 = sys.get_messages(2);
        assert_eq!(msgs_for_2.len(), 2);

        let msgs_for_0 = sys.get_messages(0);
        assert_eq!(msgs_for_0.len(), 1); // Doesn't see own message
    }

    #[test]
    fn test_comm_channel_clear() {
        let mut sys = MultiAgentSystem::new(2, 2, 0.1, 0.9, 0.1).unwrap();
        sys.send_message(0, vec![1.0]).unwrap();
        sys.clear_messages();
        assert_eq!(sys.get_messages(1).len(), 0);
    }

    #[test]
    fn test_agent_not_found() {
        let sys = MultiAgentSystem::new(2, 2, 0.1, 0.9, 0.1).unwrap();
        assert!(sys.agent(5).is_err());
    }

    #[test]
    fn test_episode_tracking() {
        let mut sys = MultiAgentSystem::new(2, 2, 0.5, 0.9, 0.0).unwrap();
        sys.update(&[0, 0], &[0, 1], &[1.0, 2.0], &[1, 1], &[true, true])
            .unwrap();
        sys.record_episode();
        assert_eq!(sys.episode_count(), 1);
        assert!(approx(sys.team_rewards()[0], 3.0));
    }

    #[test]
    fn test_reset_episode() {
        let mut sys = MultiAgentSystem::new(2, 2, 0.5, 0.9, 0.0).unwrap();
        sys.update(&[0, 0], &[0, 1], &[5.0, 5.0], &[1, 1], &[true, true])
            .unwrap();
        sys.reset_episode();
        assert!(approx(sys.agent(0).unwrap().total_reward, 0.0));
    }

    #[test]
    fn test_best_action() {
        let mut sys = MultiAgentSystem::new(1, 3, 1.0, 0.0, 0.0).unwrap();
        sys.update(&[0], &[2], &[10.0], &[0], &[true]).unwrap();
        let best = sys.best_action(0, 0).unwrap();
        assert_eq!(best, 2);
    }

    #[test]
    fn test_average_team_reward() {
        let mut sys = MultiAgentSystem::new(2, 2, 0.5, 0.9, 0.0).unwrap();
        for r in [2.0, 4.0, 6.0] {
            sys.update(&[0, 0], &[0, 1], &[r, 0.0], &[1, 1], &[true, true]).unwrap();
            sys.record_episode();
            sys.reset_episode();
        }
        assert!(approx(sys.average_team_reward(2), 5.0));
    }

    #[test]
    fn test_action_out_of_range() {
        let mut sys = MultiAgentSystem::new(2, 3, 0.1, 0.9, 0.0).unwrap();
        let result = sys.update(&[0, 0], &[0, 10], &[1.0, 1.0], &[1, 1], &[false, false]);
        assert!(result.is_err());
    }

    #[test]
    fn test_display_system() {
        let sys = MultiAgentSystem::new(3, 4, 0.1, 0.99, 0.1).unwrap();
        let s = format!("{sys}");
        assert!(s.contains("MultiAgentSystem"));
    }

    #[test]
    fn test_display_agent() {
        let agent = Agent::new(0, 4, 0.1, 0.99, 0.1, 42);
        let s = format!("{agent}");
        assert!(s.contains("Agent"));
    }

    #[test]
    fn test_display_channel() {
        let ch = CommChannel::new(10, 4);
        let s = format!("{ch}");
        assert!(s.contains("CommChannel"));
    }

    #[test]
    fn test_display_reward_scheme() {
        let rs = RewardScheme::Mixed { team_weight: 0.5 };
        let s = format!("{rs}");
        assert!(s.contains("Mixed"));
    }
}
