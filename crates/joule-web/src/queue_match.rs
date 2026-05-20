//! Match queue system — FIFO, priority lanes, VIP, age-boost, statistics.
//!
//! Replaces queue.js / PlayFab-queue with pure Rust.
//! QueueEntry with player/enqueue_time/priority, MatchQueue with
//! FIFO and priority lanes, dequeue for match, configurable match size,
//! estimated wait time, queue position, statistics (avg wait, match rate),
//! VIP/priority support, age-based priority boost, capacity limits.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueueError {
    QueueFull { capacity: usize },
    PlayerAlreadyQueued(String),
    PlayerNotQueued(String),
    NotEnoughPlayers { have: usize, need: usize },
    InvalidConfig(String),
}

impl fmt::Display for QueueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::QueueFull { capacity } => write!(f, "queue full (capacity {capacity})"),
            Self::PlayerAlreadyQueued(p) => write!(f, "already queued: {p}"),
            Self::PlayerNotQueued(p) => write!(f, "not queued: {p}"),
            Self::NotEnoughPlayers { have, need } => {
                write!(f, "not enough players: have {have}, need {need}")
            }
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
        }
    }
}

impl std::error::Error for QueueError {}

// ── Priority ────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Priority {
    Normal,
    High,
    VIP,
}

impl fmt::Display for Priority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Normal => write!(f, "Normal"),
            Self::High => write!(f, "High"),
            Self::VIP => write!(f, "VIP"),
        }
    }
}

// ── Queue Entry ─────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct QueueEntry {
    pub player_id: String,
    pub enqueue_time_ms: u64,
    pub priority: Priority,
    pub game_mode: String,
}

impl QueueEntry {
    pub fn new(player_id: &str, game_mode: &str, enqueue_time_ms: u64) -> Self {
        Self {
            player_id: player_id.to_string(),
            enqueue_time_ms,
            priority: Priority::Normal,
            game_mode: game_mode.to_string(),
        }
    }

    pub fn with_priority(mut self, prio: Priority) -> Self {
        self.priority = prio;
        self
    }

    pub fn wait_time_ms(&self, now_ms: u64) -> u64 {
        now_ms.saturating_sub(self.enqueue_time_ms)
    }

    /// Effective priority score: base priority + age boost.
    pub fn effective_score(&self, now_ms: u64, age_boost_per_sec: f64) -> f64 {
        let base = match self.priority {
            Priority::Normal => 0.0,
            Priority::High => 1000.0,
            Priority::VIP => 2000.0,
        };
        let age_secs = self.wait_time_ms(now_ms) as f64 / 1000.0;
        base + age_secs * age_boost_per_sec
    }
}

impl fmt::Display for QueueEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "QueueEntry({}, {})", self.player_id, self.priority)
    }
}

// ── Queue Statistics ────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct QueueStats {
    pub total_enqueued: u64,
    pub total_matched: u64,
    pub total_cancelled: u64,
    pub total_wait_ms: u64,
    pub matches_formed: u64,
}

impl QueueStats {
    pub fn avg_wait_ms(&self) -> f64 {
        if self.total_matched == 0 {
            return 0.0;
        }
        self.total_wait_ms as f64 / self.total_matched as f64
    }

    pub fn match_rate(&self) -> f64 {
        if self.total_enqueued == 0 {
            return 0.0;
        }
        self.total_matched as f64 / self.total_enqueued as f64
    }

    pub fn cancel_rate(&self) -> f64 {
        if self.total_enqueued == 0 {
            return 0.0;
        }
        self.total_cancelled as f64 / self.total_enqueued as f64
    }
}

impl fmt::Display for QueueStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Stats(enqueued={}, matched={}, avg_wait={:.0}ms)",
            self.total_enqueued,
            self.total_matched,
            self.avg_wait_ms()
        )
    }
}

// ── Queue Config ────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct QueueConfig {
    pub match_size: usize,
    pub capacity: usize,
    pub age_boost_per_sec: f64,
    pub game_mode: String,
}

impl QueueConfig {
    pub fn new(game_mode: &str, match_size: usize) -> Self {
        Self {
            match_size,
            capacity: 10_000,
            age_boost_per_sec: 10.0,
            game_mode: game_mode.to_string(),
        }
    }

    pub fn with_capacity(mut self, cap: usize) -> Self {
        self.capacity = cap;
        self
    }

    pub fn with_age_boost(mut self, boost: f64) -> Self {
        self.age_boost_per_sec = boost;
        self
    }
}

// ── Dequeued Match ──────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct DequeuedMatch {
    pub match_id: String,
    pub players: Vec<String>,
    pub game_mode: String,
    pub formed_at_ms: u64,
}

impl fmt::Display for DequeuedMatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Match({}, {} players, {})",
            self.match_id,
            self.players.len(),
            self.game_mode
        )
    }
}

// ── Match Queue ─────────────────────────────────────────────────

#[derive(Debug)]
pub struct MatchQueue {
    entries: Vec<QueueEntry>,
    player_index: HashMap<String, usize>,
    config: QueueConfig,
    stats: QueueStats,
    next_match_id: u64,
}

impl MatchQueue {
    pub fn new(config: QueueConfig) -> Self {
        Self {
            entries: Vec::new(),
            player_index: HashMap::new(),
            config,
            stats: QueueStats::default(),
            next_match_id: 1,
        }
    }

    pub fn size(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn is_full(&self) -> bool {
        self.entries.len() >= self.config.capacity
    }

    pub fn enqueue(&mut self, entry: QueueEntry) -> Result<usize, QueueError> {
        if self.is_full() {
            return Err(QueueError::QueueFull {
                capacity: self.config.capacity,
            });
        }
        if self.player_index.contains_key(&entry.player_id) {
            return Err(QueueError::PlayerAlreadyQueued(entry.player_id.clone()));
        }
        let idx = self.entries.len();
        self.player_index.insert(entry.player_id.clone(), idx);
        self.entries.push(entry);
        self.stats.total_enqueued += 1;
        Ok(idx + 1)
    }

    pub fn cancel(&mut self, player_id: &str) -> Result<(), QueueError> {
        let idx = self
            .player_index
            .remove(player_id)
            .ok_or_else(|| QueueError::PlayerNotQueued(player_id.to_string()))?;
        self.entries.remove(idx);
        self.rebuild_index();
        self.stats.total_cancelled += 1;
        Ok(())
    }

    pub fn position(&self, player_id: &str) -> Option<usize> {
        self.player_index.get(player_id).map(|i| i + 1)
    }

    pub fn estimated_wait_ms(&self, player_id: &str, _now_ms: u64) -> Option<u64> {
        let pos = self.position(player_id)?;
        let matches_ahead = pos / self.config.match_size;
        let avg_wait = self.stats.avg_wait_ms();
        if avg_wait > 0.0 {
            Some((matches_ahead as f64 * avg_wait) as u64)
        } else {
            Some(matches_ahead as u64 * 30_000)
        }
    }

    /// Try to form a match by dequeueing top-priority players.
    pub fn try_dequeue(&mut self, now_ms: u64) -> Result<DequeuedMatch, QueueError> {
        let size = self.config.match_size;
        if self.entries.len() < size {
            return Err(QueueError::NotEnoughPlayers {
                have: self.entries.len(),
                need: size,
            });
        }

        // Sort by effective score descending.
        let boost = self.config.age_boost_per_sec;
        self.entries
            .sort_by(|a, b| {
                b.effective_score(now_ms, boost)
                    .partial_cmp(&a.effective_score(now_ms, boost))
                    .unwrap()
            });
        self.rebuild_index();

        // Take top N.
        let matched: Vec<QueueEntry> = self.entries.drain(0..size).collect();
        self.rebuild_index();

        // Update stats.
        for m in &matched {
            self.stats.total_matched += 1;
            self.stats.total_wait_ms += m.wait_time_ms(now_ms);
        }
        self.stats.matches_formed += 1;

        let mid = format!("qm-{}", self.next_match_id);
        self.next_match_id += 1;

        Ok(DequeuedMatch {
            match_id: mid,
            players: matched.iter().map(|e| e.player_id.clone()).collect(),
            game_mode: self.config.game_mode.clone(),
            formed_at_ms: now_ms,
        })
    }

    pub fn stats(&self) -> &QueueStats {
        &self.stats
    }

    pub fn config(&self) -> &QueueConfig {
        &self.config
    }

    fn rebuild_index(&mut self) {
        self.player_index.clear();
        for (i, e) in self.entries.iter().enumerate() {
            self.player_index.insert(e.player_id.clone(), i);
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> QueueConfig {
        QueueConfig::new("ranked", 2)
    }

    fn entry(id: &str, time: u64) -> QueueEntry {
        QueueEntry::new(id, "ranked", time)
    }

    #[test]
    fn enqueue_and_size() {
        let mut q = MatchQueue::new(cfg());
        q.enqueue(entry("a", 0)).unwrap();
        assert_eq!(q.size(), 1);
    }

    #[test]
    fn enqueue_returns_position() {
        let mut q = MatchQueue::new(cfg());
        let pos = q.enqueue(entry("a", 0)).unwrap();
        assert_eq!(pos, 1);
        let pos2 = q.enqueue(entry("b", 0)).unwrap();
        assert_eq!(pos2, 2);
    }

    #[test]
    fn duplicate_enqueue_error() {
        let mut q = MatchQueue::new(cfg());
        q.enqueue(entry("a", 0)).unwrap();
        let err = q.enqueue(entry("a", 100)).unwrap_err();
        assert!(matches!(err, QueueError::PlayerAlreadyQueued(_)));
    }

    #[test]
    fn cancel_player() {
        let mut q = MatchQueue::new(cfg());
        q.enqueue(entry("a", 0)).unwrap();
        q.cancel("a").unwrap();
        assert_eq!(q.size(), 0);
    }

    #[test]
    fn cancel_not_queued() {
        let mut q = MatchQueue::new(cfg());
        let err = q.cancel("ghost").unwrap_err();
        assert!(matches!(err, QueueError::PlayerNotQueued(_)));
    }

    #[test]
    fn dequeue_match() {
        let mut q = MatchQueue::new(cfg());
        q.enqueue(entry("a", 0)).unwrap();
        q.enqueue(entry("b", 0)).unwrap();
        let m = q.try_dequeue(1000).unwrap();
        assert_eq!(m.players.len(), 2);
        assert_eq!(q.size(), 0);
    }

    #[test]
    fn not_enough_players() {
        let mut q = MatchQueue::new(cfg());
        q.enqueue(entry("a", 0)).unwrap();
        let err = q.try_dequeue(0).unwrap_err();
        assert!(matches!(err, QueueError::NotEnoughPlayers { .. }));
    }

    #[test]
    fn vip_priority() {
        let mut q = MatchQueue::new(QueueConfig::new("ranked", 2).with_age_boost(0.0));
        q.enqueue(entry("normal", 0)).unwrap();
        q.enqueue(entry("vip", 0).with_priority(Priority::VIP)).unwrap();
        q.enqueue(entry("also_normal", 0)).unwrap();
        let m = q.try_dequeue(0).unwrap();
        assert!(m.players.contains(&"vip".to_string()));
    }

    #[test]
    fn age_boost() {
        let mut q = MatchQueue::new(QueueConfig::new("ranked", 2).with_age_boost(100.0));
        q.enqueue(entry("old", 0)).unwrap(); // 10s old at now=10000
        q.enqueue(entry("new", 9000)).unwrap(); // 1s old
        q.enqueue(entry("high", 9000).with_priority(Priority::High)).unwrap();
        let m = q.try_dequeue(10_000).unwrap();
        // "high" has 1000 base + 100 age boost, "old" has 0 + 1000 age boost
        assert!(m.players.contains(&"old".to_string()));
        assert!(m.players.contains(&"high".to_string()));
    }

    #[test]
    fn queue_capacity() {
        let mut q = MatchQueue::new(QueueConfig::new("ranked", 2).with_capacity(2));
        q.enqueue(entry("a", 0)).unwrap();
        q.enqueue(entry("b", 0)).unwrap();
        let err = q.enqueue(entry("c", 0)).unwrap_err();
        assert!(matches!(err, QueueError::QueueFull { .. }));
    }

    #[test]
    fn queue_position() {
        let mut q = MatchQueue::new(cfg());
        q.enqueue(entry("a", 0)).unwrap();
        q.enqueue(entry("b", 0)).unwrap();
        assert_eq!(q.position("a"), Some(1));
        assert_eq!(q.position("b"), Some(2));
        assert_eq!(q.position("z"), None);
    }

    #[test]
    fn estimated_wait() {
        let mut q = MatchQueue::new(cfg());
        q.enqueue(entry("a", 0)).unwrap();
        q.enqueue(entry("b", 0)).unwrap();
        q.enqueue(entry("c", 0)).unwrap();
        // No history, defaults to 30s per match.
        let wait = q.estimated_wait_ms("c", 1000).unwrap();
        assert!(wait > 0);
    }

    #[test]
    fn stats_tracking() {
        let mut q = MatchQueue::new(cfg());
        q.enqueue(entry("a", 0)).unwrap();
        q.enqueue(entry("b", 0)).unwrap();
        q.try_dequeue(5000).unwrap();
        assert_eq!(q.stats().total_enqueued, 2);
        assert_eq!(q.stats().total_matched, 2);
        assert_eq!(q.stats().matches_formed, 1);
    }

    #[test]
    fn avg_wait_stat() {
        let mut q = MatchQueue::new(cfg());
        q.enqueue(entry("a", 0)).unwrap();
        q.enqueue(entry("b", 0)).unwrap();
        q.try_dequeue(4000).unwrap();
        assert!((q.stats().avg_wait_ms() - 4000.0).abs() < 0.01);
    }

    #[test]
    fn match_rate() {
        let mut q = MatchQueue::new(cfg());
        q.enqueue(entry("a", 0)).unwrap();
        q.enqueue(entry("b", 0)).unwrap();
        q.enqueue(entry("c", 0)).unwrap();
        q.try_dequeue(0).unwrap();
        q.cancel("c").unwrap();
        assert!((q.stats().match_rate() - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn cancel_rate() {
        let mut q = MatchQueue::new(cfg());
        q.enqueue(entry("a", 0)).unwrap();
        q.cancel("a").unwrap();
        assert!((q.stats().cancel_rate() - 1.0).abs() < 0.01);
    }

    #[test]
    fn display_entry() {
        let e = entry("alice", 0).with_priority(Priority::VIP);
        assert!(e.to_string().contains("alice"));
        assert!(e.to_string().contains("VIP"));
    }

    #[test]
    fn display_stats() {
        let stats = QueueStats::default();
        assert!(stats.to_string().contains("enqueued=0"));
    }

    #[test]
    fn display_match() {
        let m = DequeuedMatch {
            match_id: "qm-1".into(),
            players: vec!["a".into()],
            game_mode: "ranked".into(),
            formed_at_ms: 0,
        };
        assert!(m.to_string().contains("qm-1"));
    }

    #[test]
    fn effective_score_no_wait() {
        let e = entry("a", 0);
        assert!((e.effective_score(0, 10.0)).abs() < 0.01);
        let v = entry("a", 0).with_priority(Priority::VIP);
        assert!((v.effective_score(0, 10.0) - 2000.0).abs() < 0.01);
    }
}
