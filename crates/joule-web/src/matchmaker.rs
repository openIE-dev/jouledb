//! General matchmaking system — skill-based pool, party grouping, quality scoring.
//!
//! Replaces matchmaking.js / PlayFab-matchmaking with pure Rust.
//! MatchRequest with skill and preferences, MatchPool collecting requests,
//! expanding skill windows over wait time, party/group matching,
//! configurable match size, quality score, cancel, queue position.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatchError {
    PlayerAlreadyQueued(String),
    PlayerNotQueued(String),
    PartyTooLarge { party_size: usize, match_size: usize },
    NoMatchFound,
    InvalidConfig(String),
    PoolEmpty,
}

impl fmt::Display for MatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PlayerAlreadyQueued(p) => write!(f, "already queued: {p}"),
            Self::PlayerNotQueued(p) => write!(f, "not queued: {p}"),
            Self::PartyTooLarge { party_size, match_size } => {
                write!(f, "party size {party_size} exceeds match size {match_size}")
            }
            Self::NoMatchFound => write!(f, "no match found"),
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
            Self::PoolEmpty => write!(f, "match pool is empty"),
        }
    }
}

impl std::error::Error for MatchError {}

// ── Match Request ───────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct MatchRequest {
    pub player_id: String,
    pub skill: f64,
    pub game_mode: String,
    pub party_id: Option<String>,
    pub enqueue_time_ms: u64,
}

impl MatchRequest {
    pub fn new(player_id: &str, skill: f64, game_mode: &str) -> Self {
        Self {
            player_id: player_id.to_string(),
            skill,
            game_mode: game_mode.to_string(),
            party_id: None,
            enqueue_time_ms: 0,
        }
    }

    pub fn with_party(mut self, party_id: &str) -> Self {
        self.party_id = Some(party_id.to_string());
        self
    }

    pub fn with_enqueue_time(mut self, ms: u64) -> Self {
        self.enqueue_time_ms = ms;
        self
    }

    pub fn wait_time(&self, now_ms: u64) -> u64 {
        now_ms.saturating_sub(self.enqueue_time_ms)
    }
}

impl fmt::Display for MatchRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MatchReq({}, skill={:.0})", self.player_id, self.skill)
    }
}

// ── Match Result ────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct MatchResult {
    pub match_id: String,
    pub players: Vec<String>,
    pub average_skill: f64,
    pub quality_score: f64,
    pub game_mode: String,
}

impl fmt::Display for MatchResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Match({}, {} players, quality={:.2})",
            self.match_id,
            self.players.len(),
            self.quality_score
        )
    }
}

// ── Match Config ────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct MatchConfig {
    pub match_size: usize,
    pub base_skill_range: f64,
    pub skill_range_expansion_per_sec: f64,
    pub max_skill_range: f64,
}

impl MatchConfig {
    pub fn new(match_size: usize) -> Self {
        Self {
            match_size,
            base_skill_range: 100.0,
            skill_range_expansion_per_sec: 10.0,
            max_skill_range: 500.0,
        }
    }

    pub fn with_base_skill_range(mut self, range: f64) -> Self {
        self.base_skill_range = range;
        self
    }

    pub fn with_expansion_rate(mut self, rate: f64) -> Self {
        self.skill_range_expansion_per_sec = rate;
        self
    }

    pub fn with_max_skill_range(mut self, max: f64) -> Self {
        self.max_skill_range = max;
        self
    }

    /// Effective skill range for a request given current time.
    pub fn effective_range(&self, wait_ms: u64) -> f64 {
        let expansion = (wait_ms as f64 / 1000.0) * self.skill_range_expansion_per_sec;
        (self.base_skill_range + expansion).min(self.max_skill_range)
    }
}

impl Default for MatchConfig {
    fn default() -> Self {
        Self::new(10)
    }
}

// ── Match Pool ──────────────────────────────────────────────────

#[derive(Debug)]
pub struct MatchPool {
    requests: Vec<MatchRequest>,
    config: MatchConfig,
    next_match_id: u64,
    player_index: HashMap<String, usize>,
}

impl MatchPool {
    pub fn new(config: MatchConfig) -> Self {
        Self {
            requests: Vec::new(),
            config,
            next_match_id: 1,
            player_index: HashMap::new(),
        }
    }

    pub fn queue_size(&self) -> usize {
        self.requests.len()
    }

    pub fn enqueue(&mut self, req: MatchRequest) -> Result<(), MatchError> {
        if self.player_index.contains_key(&req.player_id) {
            return Err(MatchError::PlayerAlreadyQueued(req.player_id.clone()));
        }
        if let Some(ref pid) = req.party_id {
            let party_count = self.party_size(pid) + 1;
            if party_count > self.config.match_size {
                return Err(MatchError::PartyTooLarge {
                    party_size: party_count,
                    match_size: self.config.match_size,
                });
            }
        }
        let idx = self.requests.len();
        self.player_index.insert(req.player_id.clone(), idx);
        self.requests.push(req);
        Ok(())
    }

    pub fn cancel(&mut self, player_id: &str) -> Result<(), MatchError> {
        let idx = self
            .player_index
            .remove(player_id)
            .ok_or_else(|| MatchError::PlayerNotQueued(player_id.to_string()))?;
        self.requests.remove(idx);
        // Rebuild index after removal.
        self.player_index.clear();
        for (i, r) in self.requests.iter().enumerate() {
            self.player_index.insert(r.player_id.clone(), i);
        }
        Ok(())
    }

    pub fn queue_position(&self, player_id: &str) -> Option<usize> {
        self.player_index.get(player_id).map(|i| i + 1)
    }

    pub fn estimated_wait_secs(&self, player_id: &str) -> Option<f64> {
        let pos = self.queue_position(player_id)?;
        let matches_ahead = pos / self.config.match_size;
        Some(matches_ahead as f64 * 30.0)
    }

    fn party_size(&self, party_id: &str) -> usize {
        self.requests
            .iter()
            .filter(|r| r.party_id.as_deref() == Some(party_id))
            .count()
    }

    /// Try to form a match at the given timestamp.
    pub fn try_match(&mut self, now_ms: u64) -> Result<MatchResult, MatchError> {
        if self.requests.len() < self.config.match_size {
            return Err(MatchError::NoMatchFound);
        }

        // Sort by skill for bracket matching.
        self.requests.sort_by(|a, b| a.skill.partial_cmp(&b.skill).unwrap());
        self.rebuild_index();

        // Sliding window: find tightest group of match_size.
        let size = self.config.match_size;
        let mut best_start = None;
        let mut best_range = f64::MAX;

        for start in 0..=self.requests.len() - size {
            let lo = self.requests[start].skill;
            let hi = self.requests[start + size - 1].skill;
            let range = hi - lo;

            // Check each player's effective range covers this spread.
            let anchor = (lo + hi) / 2.0;
            let all_within = self.requests[start..start + size].iter().all(|r| {
                let eff = self.config.effective_range(r.wait_time(now_ms));
                (r.skill - anchor).abs() <= eff
            });

            if all_within && range < best_range {
                // Verify party constraints: all party members must be in the window.
                let parties_ok = self.check_party_fit(start, size);
                if parties_ok {
                    best_range = range;
                    best_start = Some(start);
                }
            }
        }

        let start = best_start.ok_or(MatchError::NoMatchFound)?;
        let matched: Vec<MatchRequest> = self.requests.drain(start..start + size).collect();
        self.rebuild_index();

        let avg = matched.iter().map(|r| r.skill).sum::<f64>() / matched.len() as f64;
        let variance = matched.iter().map(|r| (r.skill - avg).powi(2)).sum::<f64>() / matched.len() as f64;
        let quality = 1.0 / (1.0 + variance.sqrt() / 100.0);

        let mid = format!("match-{}", self.next_match_id);
        self.next_match_id += 1;

        Ok(MatchResult {
            match_id: mid,
            players: matched.iter().map(|r| r.player_id.clone()).collect(),
            average_skill: avg,
            quality_score: quality,
            game_mode: matched[0].game_mode.clone(),
        })
    }

    fn check_party_fit(&self, start: usize, size: usize) -> bool {
        let window: Vec<&MatchRequest> = self.requests[start..start + size].iter().collect();
        let window_ids: std::collections::HashSet<&str> =
            window.iter().map(|r| r.player_id.as_str()).collect();

        for req in &window {
            if let Some(ref pid) = req.party_id {
                // All party members must be in the window.
                let all_in = self
                    .requests
                    .iter()
                    .filter(|r| r.party_id.as_deref() == Some(pid))
                    .all(|r| window_ids.contains(r.player_id.as_str()));
                if !all_in {
                    return false;
                }
            }
        }
        true
    }

    fn rebuild_index(&mut self) {
        self.player_index.clear();
        for (i, r) in self.requests.iter().enumerate() {
            self.player_index.insert(r.player_id.clone(), i);
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn pool(size: usize) -> MatchPool {
        MatchPool::new(MatchConfig::new(size))
    }

    fn req(id: &str, skill: f64) -> MatchRequest {
        MatchRequest::new(id, skill, "ranked")
    }

    #[test]
    fn enqueue_and_cancel() {
        let mut p = pool(2);
        p.enqueue(req("a", 1000.0)).unwrap();
        assert_eq!(p.queue_size(), 1);
        p.cancel("a").unwrap();
        assert_eq!(p.queue_size(), 0);
    }

    #[test]
    fn duplicate_enqueue_error() {
        let mut p = pool(2);
        p.enqueue(req("a", 1000.0)).unwrap();
        let err = p.enqueue(req("a", 1000.0)).unwrap_err();
        assert!(matches!(err, MatchError::PlayerAlreadyQueued(_)));
    }

    #[test]
    fn cancel_not_queued() {
        let mut p = pool(2);
        let err = p.cancel("ghost").unwrap_err();
        assert!(matches!(err, MatchError::PlayerNotQueued(_)));
    }

    #[test]
    fn simple_match() {
        let mut p = pool(2);
        p.enqueue(req("a", 1000.0)).unwrap();
        p.enqueue(req("b", 1050.0)).unwrap();
        let result = p.try_match(0).unwrap();
        assert_eq!(result.players.len(), 2);
        assert_eq!(p.queue_size(), 0);
    }

    #[test]
    fn not_enough_players() {
        let mut p = pool(4);
        p.enqueue(req("a", 1000.0)).unwrap();
        p.enqueue(req("b", 1050.0)).unwrap();
        let err = p.try_match(0).unwrap_err();
        assert!(matches!(err, MatchError::NoMatchFound));
    }

    #[test]
    fn skill_range_expansion() {
        let cfg = MatchConfig::new(2)
            .with_base_skill_range(50.0)
            .with_expansion_rate(50.0);
        let mut p = MatchPool::new(cfg);
        // 400 apart — base range 50 won't match at 1s (range=100, gap/2=200 > 100).
        p.enqueue(req("a", 1000.0).with_enqueue_time(0)).unwrap();
        p.enqueue(req("b", 1400.0).with_enqueue_time(0)).unwrap();
        let err = p.try_match(1000).unwrap_err(); // 1s wait → range 100, gap/2=200 too wide
        assert!(matches!(err, MatchError::NoMatchFound));
        let result = p.try_match(5000).unwrap(); // 5s wait → range 300, gap/2=200 fits
        assert_eq!(result.players.len(), 2);
    }

    #[test]
    fn max_skill_range_cap() {
        let cfg = MatchConfig::new(2).with_max_skill_range(200.0);
        assert_eq!(cfg.effective_range(999_999), 200.0);
    }

    #[test]
    fn quality_score_perfect() {
        let mut p = pool(2);
        p.enqueue(req("a", 1000.0)).unwrap();
        p.enqueue(req("b", 1000.0)).unwrap();
        let result = p.try_match(0).unwrap();
        assert!((result.quality_score - 1.0).abs() < 0.01);
    }

    #[test]
    fn quality_score_lower_for_spread() {
        let mut p = pool(2);
        p.enqueue(req("a", 800.0).with_enqueue_time(0)).unwrap();
        p.enqueue(req("b", 1200.0).with_enqueue_time(0)).unwrap();
        let result = p.try_match(60_000).unwrap();
        assert!(result.quality_score < 0.9);
    }

    #[test]
    fn party_matching() {
        let mut p = pool(4);
        p.enqueue(req("a", 1000.0).with_party("p1")).unwrap();
        p.enqueue(req("b", 1020.0).with_party("p1")).unwrap();
        p.enqueue(req("c", 1010.0)).unwrap();
        p.enqueue(req("d", 1030.0)).unwrap();
        let result = p.try_match(0).unwrap();
        assert!(result.players.contains(&"a".to_string()));
        assert!(result.players.contains(&"b".to_string()));
    }

    #[test]
    fn party_too_large() {
        let mut p = pool(2);
        p.enqueue(req("a", 1000.0).with_party("p1")).unwrap();
        p.enqueue(req("b", 1000.0).with_party("p1")).unwrap();
        let err = p.enqueue(req("c", 1000.0).with_party("p1")).unwrap_err();
        assert!(matches!(err, MatchError::PartyTooLarge { .. }));
    }

    #[test]
    fn queue_position() {
        let mut p = pool(4);
        p.enqueue(req("a", 100.0)).unwrap();
        p.enqueue(req("b", 200.0)).unwrap();
        assert_eq!(p.queue_position("a"), Some(1));
        assert_eq!(p.queue_position("b"), Some(2));
        assert_eq!(p.queue_position("z"), None);
    }

    #[test]
    fn estimated_wait() {
        let mut p = pool(2);
        p.enqueue(req("a", 100.0)).unwrap();
        p.enqueue(req("b", 200.0)).unwrap();
        p.enqueue(req("c", 300.0)).unwrap();
        let wait = p.estimated_wait_secs("c").unwrap();
        assert!(wait > 0.0);
    }

    #[test]
    fn match_result_display() {
        let r = MatchResult {
            match_id: "m1".into(),
            players: vec!["a".into(), "b".into()],
            average_skill: 1000.0,
            quality_score: 0.95,
            game_mode: "ranked".into(),
        };
        let s = r.to_string();
        assert!(s.contains("m1"));
        assert!(s.contains("2 players"));
    }

    #[test]
    fn match_request_display() {
        let r = req("alice", 1500.0);
        assert!(r.to_string().contains("alice"));
    }

    #[test]
    fn config_builder() {
        let cfg = MatchConfig::new(6)
            .with_base_skill_range(200.0)
            .with_expansion_rate(25.0)
            .with_max_skill_range(1000.0);
        assert_eq!(cfg.match_size, 6);
        assert_eq!(cfg.base_skill_range, 200.0);
    }

    #[test]
    fn wait_time_calculation() {
        let r = req("a", 100.0).with_enqueue_time(5000);
        assert_eq!(r.wait_time(8000), 3000);
        assert_eq!(r.wait_time(3000), 0); // saturating
    }

    #[test]
    fn multiple_matches_drain_pool() {
        let mut p = pool(2);
        for i in 0..6 {
            p.enqueue(req(&format!("p{i}"), 1000.0 + i as f64)).unwrap();
        }
        let m1 = p.try_match(0).unwrap();
        let m2 = p.try_match(0).unwrap();
        let m3 = p.try_match(0).unwrap();
        assert_eq!(m1.players.len(), 2);
        assert_eq!(m2.players.len(), 2);
        assert_eq!(m3.players.len(), 2);
        assert_eq!(p.queue_size(), 0);
    }

    #[test]
    fn average_skill_calculated() {
        let mut p = pool(2);
        p.enqueue(req("a", 1000.0)).unwrap();
        p.enqueue(req("b", 1100.0)).unwrap();
        let result = p.try_match(0).unwrap();
        assert!((result.average_skill - 1050.0).abs() < 0.01);
    }
}
