//! NAT hole punching for direct P2P connectivity — NAT type detection,
//! simultaneous open coordination, technique selection based on NAT types,
//! punch attempt tracking, retry logic, and connectivity matrix.

use std::collections::HashMap;
use std::fmt;

// ── NAT Type ────────────────────────────────────────────────────────────────

/// Classification of NAT behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NatType {
    /// No NAT, publicly reachable.
    Open,
    /// Maps same internal endpoint to same external endpoint for all destinations.
    FullCone,
    /// Like FullCone but only allows replies from previously contacted IPs.
    RestrictedCone,
    /// Like RestrictedCone but also restricts by port.
    PortRestricted,
    /// Maps to different external endpoint per destination — hardest to punch.
    Symmetric,
}

impl NatType {
    /// Whether direct hole punching is likely to succeed between two NAT types.
    pub fn punch_likely(a: NatType, b: NatType) -> bool {
        use NatType::*;
        match (a, b) {
            (Open, _) | (_, Open) => true,
            (FullCone, _) | (_, FullCone) => true,
            (RestrictedCone, RestrictedCone) => true,
            (RestrictedCone, PortRestricted) | (PortRestricted, RestrictedCone) => true,
            (PortRestricted, PortRestricted) => true,
            (Symmetric, RestrictedCone) | (RestrictedCone, Symmetric) => true,
            (Symmetric, Symmetric)
            | (Symmetric, PortRestricted)
            | (PortRestricted, Symmetric) => false,
        }
    }
}

impl fmt::Display for NatType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NatType::Open => write!(f, "Open"),
            NatType::FullCone => write!(f, "Full Cone"),
            NatType::RestrictedCone => write!(f, "Restricted Cone"),
            NatType::PortRestricted => write!(f, "Port Restricted"),
            NatType::Symmetric => write!(f, "Symmetric"),
        }
    }
}

// ── PunchState ──────────────────────────────────────────────────────────────

/// State of a hole punch attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PunchState {
    /// Waiting to start.
    Pending,
    /// Sending probe packets.
    Probing,
    /// Received a response, verifying bidirectionality.
    Verifying,
    /// Connection established.
    Connected,
    /// Punch failed after exhausting retries.
    Failed,
}

impl fmt::Display for PunchState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PunchState::Pending => write!(f, "Pending"),
            PunchState::Probing => write!(f, "Probing"),
            PunchState::Verifying => write!(f, "Verifying"),
            PunchState::Connected => write!(f, "Connected"),
            PunchState::Failed => write!(f, "Failed"),
        }
    }
}

// ── PunchTechnique ──────────────────────────────────────────────────────────

/// Technique to use for hole punching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PunchTechnique {
    /// Both sides send simultaneously to each other's predicted endpoint.
    SimultaneousOpen,
    /// Try multiple candidate ports in parallel.
    PortPrediction,
    /// Use a relay as fallback.
    RelayFallback,
}

impl PunchTechnique {
    /// Select technique based on the NAT types of both peers.
    pub fn select(a: NatType, b: NatType) -> Self {
        use NatType::*;
        match (a, b) {
            (Open, _) | (_, Open) => PunchTechnique::SimultaneousOpen,
            (FullCone, _) | (_, FullCone) => PunchTechnique::SimultaneousOpen,
            (Symmetric, Symmetric) => PunchTechnique::RelayFallback,
            (Symmetric, PortRestricted) | (PortRestricted, Symmetric) => {
                PunchTechnique::RelayFallback
            }
            (Symmetric, _) | (_, Symmetric) => PunchTechnique::PortPrediction,
            _ => PunchTechnique::SimultaneousOpen,
        }
    }
}

// ── HolePunchAttempt ────────────────────────────────────────────────────────

/// Tracks a single hole punch attempt between two peers.
#[derive(Debug, Clone)]
pub struct HolePunchAttempt {
    pub peer_a: String,
    pub peer_b: String,
    pub nat_a: NatType,
    pub nat_b: NatType,
    pub state: PunchState,
    pub technique: PunchTechnique,
    pub attempts: u32,
    pub max_attempts: u32,
    pub port_offset: u16,
    pub started_at: u64,
    pub completed_at: Option<u64>,
}

impl HolePunchAttempt {
    pub fn new(
        peer_a: impl Into<String>,
        peer_b: impl Into<String>,
        nat_a: NatType,
        nat_b: NatType,
        now: u64,
    ) -> Self {
        let technique = PunchTechnique::select(nat_a, nat_b);
        Self {
            peer_a: peer_a.into(),
            peer_b: peer_b.into(),
            nat_a,
            nat_b,
            state: PunchState::Pending,
            technique,
            attempts: 0,
            max_attempts: 5,
            port_offset: 0,
            started_at: now,
            completed_at: None,
        }
    }

    pub fn with_max_attempts(mut self, max: u32) -> Self {
        self.max_attempts = max;
        self
    }

    /// Advance the punch attempt one step.
    pub fn step(&mut self, now: u64) {
        match self.state {
            PunchState::Pending => {
                self.state = PunchState::Probing;
                self.attempts = 1;
            }
            PunchState::Probing => {
                // Simulate: probing moves to verifying
                self.state = PunchState::Verifying;
            }
            PunchState::Verifying => {
                // Simulate: if technique allows direct connect
                if self.technique != PunchTechnique::RelayFallback {
                    self.state = PunchState::Connected;
                    self.completed_at = Some(now);
                } else {
                    self.state = PunchState::Failed;
                    self.completed_at = Some(now);
                }
            }
            _ => {}
        }
    }

    /// Retry with an incremented port offset.
    pub fn retry(&mut self) -> bool {
        if self.attempts >= self.max_attempts {
            self.state = PunchState::Failed;
            return false;
        }
        self.attempts += 1;
        self.port_offset += 1;
        self.state = PunchState::Probing;
        true
    }

    /// Whether the punch is finished (connected or failed).
    pub fn is_finished(&self) -> bool {
        matches!(self.state, PunchState::Connected | PunchState::Failed)
    }

    /// Whether the punch succeeded.
    pub fn is_connected(&self) -> bool {
        self.state == PunchState::Connected
    }

    /// Duration if completed.
    pub fn duration(&self) -> Option<u64> {
        self.completed_at.map(|c| c.saturating_sub(self.started_at))
    }
}

impl fmt::Display for HolePunchAttempt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Punch({}<->{}  state={}, technique={:?}, attempt={}/{})",
            self.peer_a, self.peer_b, self.state, self.technique,
            self.attempts, self.max_attempts,
        )
    }
}

// ── HolePunchCoordinator ────────────────────────────────────────────────────

/// Coordinates hole punch attempts between multiple peer pairs.
pub struct HolePunchCoordinator {
    attempts: HashMap<(String, String), HolePunchAttempt>,
    nat_types: HashMap<String, NatType>,
    total_success: u64,
    total_failure: u64,
    current_tick: u64,
}

impl HolePunchCoordinator {
    pub fn new() -> Self {
        Self {
            attempts: HashMap::new(),
            nat_types: HashMap::new(),
            total_success: 0,
            total_failure: 0,
            current_tick: 0,
        }
    }

    /// Advance the internal tick.
    pub fn tick(&mut self, now: u64) {
        self.current_tick = now;
    }

    /// Register the NAT type of a peer.
    pub fn register_nat(&mut self, peer_id: impl Into<String>, nat: NatType) {
        self.nat_types.insert(peer_id.into(), nat);
    }

    /// Get the NAT type of a peer.
    pub fn get_nat(&self, peer_id: &str) -> Option<NatType> {
        self.nat_types.get(peer_id).copied()
    }

    /// Initiate a hole punch between two peers.
    pub fn initiate(
        &mut self,
        peer_a: impl Into<String>,
        peer_b: impl Into<String>,
    ) -> Result<(), String> {
        let a = peer_a.into();
        let b = peer_b.into();
        let nat_a = self.nat_types.get(&a).copied().unwrap_or(NatType::PortRestricted);
        let nat_b = self.nat_types.get(&b).copied().unwrap_or(NatType::PortRestricted);

        let key = if a < b { (a.clone(), b.clone()) } else { (b.clone(), a.clone()) };
        if self.attempts.contains_key(&key) {
            return Err("punch already in progress".into());
        }
        let attempt = HolePunchAttempt::new(a, b, nat_a, nat_b, self.current_tick);
        self.attempts.insert(key, attempt);
        Ok(())
    }

    /// Step all active punch attempts forward.
    pub fn step_all(&mut self) {
        let keys: Vec<_> = self.attempts.keys().cloned().collect();
        for key in keys {
            if let Some(attempt) = self.attempts.get_mut(&key) {
                if !attempt.is_finished() {
                    attempt.step(self.current_tick);
                    if attempt.state == PunchState::Connected {
                        self.total_success += 1;
                    } else if attempt.state == PunchState::Failed {
                        self.total_failure += 1;
                    }
                }
            }
        }
    }

    /// Get the state of a punch attempt.
    pub fn get_attempt(&self, a: &str, b: &str) -> Option<&HolePunchAttempt> {
        let key = if a < b {
            (a.to_string(), b.to_string())
        } else {
            (b.to_string(), a.to_string())
        };
        self.attempts.get(&key)
    }

    /// Build a connectivity matrix (which pairs can potentially connect).
    pub fn connectivity_matrix(&self, peer_ids: &[&str]) -> Vec<Vec<bool>> {
        let n = peer_ids.len();
        let mut matrix = vec![vec![false; n]; n];
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    matrix[i][j] = true;
                    continue;
                }
                let nat_a = self.nat_types.get(peer_ids[i]).copied().unwrap_or(NatType::Symmetric);
                let nat_b = self.nat_types.get(peer_ids[j]).copied().unwrap_or(NatType::Symmetric);
                matrix[i][j] = NatType::punch_likely(nat_a, nat_b);
            }
        }
        matrix
    }

    /// Number of active (unfinished) attempts.
    pub fn active_count(&self) -> usize {
        self.attempts.values().filter(|a| !a.is_finished()).count()
    }

    /// Total successful punches.
    pub fn success_count(&self) -> u64 {
        self.total_success
    }

    /// Total failed punches.
    pub fn failure_count(&self) -> u64 {
        self.total_failure
    }
}

impl Default for HolePunchCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nat_type_display() {
        assert_eq!(format!("{}", NatType::Open), "Open");
        assert_eq!(format!("{}", NatType::Symmetric), "Symmetric");
    }

    #[test]
    fn test_punch_likely_open() {
        assert!(NatType::punch_likely(NatType::Open, NatType::Symmetric));
        assert!(NatType::punch_likely(NatType::Symmetric, NatType::Open));
    }

    #[test]
    fn test_punch_unlikely_symmetric_symmetric() {
        assert!(!NatType::punch_likely(NatType::Symmetric, NatType::Symmetric));
    }

    #[test]
    fn test_punch_likely_restricted_restricted() {
        assert!(NatType::punch_likely(NatType::RestrictedCone, NatType::RestrictedCone));
    }

    #[test]
    fn test_technique_select_open() {
        assert_eq!(
            PunchTechnique::select(NatType::Open, NatType::Symmetric),
            PunchTechnique::SimultaneousOpen,
        );
    }

    #[test]
    fn test_technique_select_symmetric_pair() {
        assert_eq!(
            PunchTechnique::select(NatType::Symmetric, NatType::Symmetric),
            PunchTechnique::RelayFallback,
        );
    }

    #[test]
    fn test_technique_select_symmetric_restricted() {
        assert_eq!(
            PunchTechnique::select(NatType::Symmetric, NatType::RestrictedCone),
            PunchTechnique::PortPrediction,
        );
    }

    #[test]
    fn test_attempt_step_to_connected() {
        let mut a = HolePunchAttempt::new("x", "y", NatType::FullCone, NatType::FullCone, 0);
        a.step(1); // Pending -> Probing
        assert_eq!(a.state, PunchState::Probing);
        a.step(2); // Probing -> Verifying
        assert_eq!(a.state, PunchState::Verifying);
        a.step(3); // Verifying -> Connected
        assert_eq!(a.state, PunchState::Connected);
        assert!(a.is_connected());
    }

    #[test]
    fn test_attempt_relay_fallback_fails() {
        let mut a = HolePunchAttempt::new("x", "y", NatType::Symmetric, NatType::Symmetric, 0);
        a.step(1);
        a.step(2);
        a.step(3);
        assert_eq!(a.state, PunchState::Failed);
    }

    #[test]
    fn test_attempt_retry() {
        let mut a = HolePunchAttempt::new("x", "y", NatType::FullCone, NatType::FullCone, 0)
            .with_max_attempts(3);
        a.step(1);
        assert!(a.retry());
        assert_eq!(a.attempts, 2);
        assert_eq!(a.port_offset, 1);
    }

    #[test]
    fn test_attempt_retry_exhausted() {
        let mut a = HolePunchAttempt::new("x", "y", NatType::FullCone, NatType::FullCone, 0)
            .with_max_attempts(1);
        a.step(1);
        assert!(!a.retry());
        assert_eq!(a.state, PunchState::Failed);
    }

    #[test]
    fn test_attempt_duration() {
        let mut a = HolePunchAttempt::new("x", "y", NatType::Open, NatType::Open, 10);
        assert!(a.duration().is_none());
        a.step(11);
        a.step(12);
        a.step(15);
        assert_eq!(a.duration(), Some(5));
    }

    #[test]
    fn test_coordinator_initiate() {
        let mut coord = HolePunchCoordinator::new();
        coord.register_nat("a", NatType::Open);
        coord.register_nat("b", NatType::FullCone);
        assert!(coord.initiate("a", "b").is_ok());
        assert_eq!(coord.active_count(), 1);
    }

    #[test]
    fn test_coordinator_duplicate_initiate() {
        let mut coord = HolePunchCoordinator::new();
        coord.initiate("a", "b").unwrap();
        assert!(coord.initiate("a", "b").is_err());
        assert!(coord.initiate("b", "a").is_err());
    }

    #[test]
    fn test_coordinator_step_all() {
        let mut coord = HolePunchCoordinator::new();
        coord.register_nat("a", NatType::Open);
        coord.register_nat("b", NatType::Open);
        coord.initiate("a", "b").unwrap();
        coord.step_all(); // Pending -> Probing
        coord.step_all(); // Probing -> Verifying
        coord.step_all(); // Verifying -> Connected
        assert_eq!(coord.success_count(), 1);
        assert_eq!(coord.active_count(), 0);
    }

    #[test]
    fn test_connectivity_matrix() {
        let mut coord = HolePunchCoordinator::new();
        coord.register_nat("a", NatType::Open);
        coord.register_nat("b", NatType::Symmetric);
        coord.register_nat("c", NatType::Symmetric);
        let m = coord.connectivity_matrix(&["a", "b", "c"]);
        assert!(m[0][1]); // Open <-> Symmetric = yes
        assert!(!m[1][2]); // Symmetric <-> Symmetric = no
        assert!(m[0][0]); // self = yes
    }

    #[test]
    fn test_get_nat() {
        let mut coord = HolePunchCoordinator::new();
        coord.register_nat("x", NatType::PortRestricted);
        assert_eq!(coord.get_nat("x"), Some(NatType::PortRestricted));
        assert_eq!(coord.get_nat("y"), None);
    }

    #[test]
    fn test_attempt_display() {
        let a = HolePunchAttempt::new("x", "y", NatType::Open, NatType::Open, 0);
        let s = format!("{}", a);
        assert!(s.contains("x"));
        assert!(s.contains("y"));
    }

    #[test]
    fn test_punch_state_display() {
        assert_eq!(format!("{}", PunchState::Probing), "Probing");
        assert_eq!(format!("{}", PunchState::Connected), "Connected");
    }
}
