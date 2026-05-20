//! Leaderboard/ranking system — scores, ranks, pagination, time windows.
//!
//! Replaces leaderboard.js / Redis-leaderboard with pure Rust.
//! Multiple leaderboards, sort orders, rank computation with ties,
//! pagination, time windows, score bounds, rate limiting,
//! percentile calculation, and top-N queries.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LeaderboardError {
    BoardNotFound(String),
    DuplicateBoard(String),
    PlayerNotFound(String),
    ScoreOutOfBounds { score: i64, min: i64, max: i64 },
    RateLimited { player_id: String, cooldown_remaining_ms: u64 },
    InvalidPageSize(usize),
    InvalidPage { page: usize, max_page: usize },
    EmptyBoard(String),
}

impl fmt::Display for LeaderboardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BoardNotFound(name) => write!(f, "leaderboard not found: {name}"),
            Self::DuplicateBoard(name) => write!(f, "duplicate leaderboard: {name}"),
            Self::PlayerNotFound(id) => write!(f, "player not found: {id}"),
            Self::ScoreOutOfBounds { score, min, max } => {
                write!(f, "score {score} out of bounds [{min}, {max}]")
            }
            Self::RateLimited { player_id, cooldown_remaining_ms } => {
                write!(f, "player {player_id} rate limited ({cooldown_remaining_ms}ms remaining)")
            }
            Self::InvalidPageSize(s) => write!(f, "invalid page size: {s}"),
            Self::InvalidPage { page, max_page } => {
                write!(f, "page {page} exceeds max page {max_page}")
            }
            Self::EmptyBoard(name) => write!(f, "leaderboard {name} is empty"),
        }
    }
}

impl std::error::Error for LeaderboardError {}

// ── Types ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortOrder {
    HighestFirst,
    LowestFirst,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TimeWindow {
    Daily,
    Weekly,
    Monthly,
    AllTime,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeaderboardEntry {
    pub player_id: String,
    pub score: i64,
    pub timestamp_ms: u64,
    pub metadata: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankedEntry {
    pub rank: usize,
    pub player_id: String,
    pub score: i64,
    pub timestamp_ms: u64,
    pub metadata: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Page {
    pub entries: Vec<RankedEntry>,
    pub page: usize,
    pub total_pages: usize,
    pub total_entries: usize,
}

// ── Score Bounds ────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScoreBounds {
    pub min: i64,
    pub max: i64,
}

impl ScoreBounds {
    pub fn new(min: i64, max: i64) -> Self {
        Self { min, max }
    }

    pub fn unbounded() -> Self {
        Self { min: i64::MIN, max: i64::MAX }
    }

    pub fn validate(&self, score: i64) -> Result<(), LeaderboardError> {
        if score < self.min || score > self.max {
            Err(LeaderboardError::ScoreOutOfBounds { score, min: self.min, max: self.max })
        } else {
            Ok(())
        }
    }
}

// ── Rate Limiter ────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct RateLimiter {
    cooldown_ms: u64,
    last_submission: HashMap<String, u64>,
}

impl RateLimiter {
    fn new(cooldown_ms: u64) -> Self {
        Self { cooldown_ms, last_submission: HashMap::new() }
    }

    fn check(&self, player_id: &str, current_time_ms: u64) -> Result<(), LeaderboardError> {
        if self.cooldown_ms == 0 { return Ok(()); }
        if let Some(&last) = self.last_submission.get(player_id) {
            if current_time_ms < last + self.cooldown_ms {
                return Err(LeaderboardError::RateLimited {
                    player_id: player_id.to_string(),
                    cooldown_remaining_ms: (last + self.cooldown_ms) - current_time_ms,
                });
            }
        }
        Ok(())
    }

    fn record(&mut self, player_id: &str, current_time_ms: u64) {
        self.last_submission.insert(player_id.to_string(), current_time_ms);
    }
}

// ── Single Leaderboard ──────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Leaderboard {
    name: String,
    sort_order: SortOrder,
    entries: Vec<LeaderboardEntry>,
    bounds: ScoreBounds,
    rate_limiter: RateLimiter,
}

impl Leaderboard {
    pub fn new(name: &str, sort_order: SortOrder) -> Self {
        Self {
            name: name.to_string(),
            sort_order,
            entries: Vec::new(),
            bounds: ScoreBounds::unbounded(),
            rate_limiter: RateLimiter::new(0),
        }
    }

    pub fn with_bounds(mut self, bounds: ScoreBounds) -> Self {
        self.bounds = bounds;
        self
    }

    pub fn with_rate_limit(mut self, cooldown_ms: u64) -> Self {
        self.rate_limiter = RateLimiter::new(cooldown_ms);
        self
    }

    pub fn name(&self) -> &str { &self.name }
    pub fn sort_order(&self) -> SortOrder { self.sort_order }
    pub fn entry_count(&self) -> usize { self.entries.len() }

    fn sort_entries(&mut self) {
        match self.sort_order {
            SortOrder::HighestFirst => {
                self.entries.sort_by(|a, b| b.score.cmp(&a.score).then(a.timestamp_ms.cmp(&b.timestamp_ms)));
            }
            SortOrder::LowestFirst => {
                self.entries.sort_by(|a, b| a.score.cmp(&b.score).then(a.timestamp_ms.cmp(&b.timestamp_ms)));
            }
        }
    }

    /// Submit a score. Updates only if better (or first submission).
    pub fn submit(&mut self, player_id: &str, score: i64, timestamp_ms: u64, metadata: &str) -> Result<bool, LeaderboardError> {
        self.bounds.validate(score)?;
        self.rate_limiter.check(player_id, timestamp_ms)?;
        self.rate_limiter.record(player_id, timestamp_ms);

        if let Some(existing) = self.entries.iter_mut().find(|e| e.player_id == player_id) {
            let is_better = match self.sort_order {
                SortOrder::HighestFirst => score > existing.score,
                SortOrder::LowestFirst => score < existing.score,
            };
            if is_better {
                existing.score = score;
                existing.timestamp_ms = timestamp_ms;
                existing.metadata = metadata.to_string();
                self.sort_entries();
                return Ok(true);
            }
            return Ok(false);
        }

        self.entries.push(LeaderboardEntry {
            player_id: player_id.to_string(),
            score,
            timestamp_ms,
            metadata: metadata.to_string(),
        });
        self.sort_entries();
        Ok(true)
    }

    /// Force-set a score regardless of whether it's better.
    pub fn set_score(&mut self, player_id: &str, score: i64, timestamp_ms: u64, metadata: &str) -> Result<(), LeaderboardError> {
        self.bounds.validate(score)?;
        if let Some(existing) = self.entries.iter_mut().find(|e| e.player_id == player_id) {
            existing.score = score;
            existing.timestamp_ms = timestamp_ms;
            existing.metadata = metadata.to_string();
        } else {
            self.entries.push(LeaderboardEntry {
                player_id: player_id.to_string(),
                score,
                timestamp_ms,
                metadata: metadata.to_string(),
            });
        }
        self.sort_entries();
        Ok(())
    }

    /// Compute ranks (1-based, with ties sharing rank).
    fn ranked_entries(&self) -> Vec<RankedEntry> {
        if self.entries.is_empty() { return Vec::new(); }
        let mut ranked = Vec::with_capacity(self.entries.len());
        let mut current_rank = 1usize;
        for (i, entry) in self.entries.iter().enumerate() {
            if i > 0 && entry.score != self.entries[i - 1].score {
                current_rank = i + 1;
            }
            ranked.push(RankedEntry {
                rank: current_rank,
                player_id: entry.player_id.clone(),
                score: entry.score,
                timestamp_ms: entry.timestamp_ms,
                metadata: entry.metadata.clone(),
            });
        }
        ranked
    }

    pub fn top_n(&self, n: usize) -> Vec<RankedEntry> {
        let ranked = self.ranked_entries();
        ranked.into_iter().take(n).collect()
    }

    pub fn player_rank(&self, player_id: &str) -> Result<RankedEntry, LeaderboardError> {
        let ranked = self.ranked_entries();
        ranked.into_iter().find(|e| e.player_id == player_id)
            .ok_or_else(|| LeaderboardError::PlayerNotFound(player_id.to_string()))
    }

    pub fn paginate(&self, page: usize, page_size: usize) -> Result<Page, LeaderboardError> {
        if page_size == 0 {
            return Err(LeaderboardError::InvalidPageSize(0));
        }
        let ranked = self.ranked_entries();
        let total = ranked.len();
        let total_pages = if total == 0 { 1 } else { (total + page_size - 1) / page_size };
        if page >= total_pages {
            return Err(LeaderboardError::InvalidPage { page, max_page: total_pages.saturating_sub(1) });
        }
        let start = page * page_size;
        let end = (start + page_size).min(total);
        Ok(Page {
            entries: ranked[start..end].to_vec(),
            page,
            total_pages,
            total_entries: total,
        })
    }

    /// Get entries in a rank range (1-based, inclusive).
    pub fn rank_range(&self, from_rank: usize, to_rank: usize) -> Vec<RankedEntry> {
        let ranked = self.ranked_entries();
        ranked.into_iter().filter(|e| e.rank >= from_rank && e.rank <= to_rank).collect()
    }

    /// Percentile of a player (0.0 = worst, 100.0 = best).
    pub fn percentile(&self, player_id: &str) -> Result<f64, LeaderboardError> {
        if self.entries.is_empty() {
            return Err(LeaderboardError::EmptyBoard(self.name.clone()));
        }
        let rank_entry = self.player_rank(player_id)?;
        let total = self.entries.len() as f64;
        let pct = (total - rank_entry.rank as f64) / total * 100.0;
        Ok(pct)
    }

    pub fn remove_player(&mut self, player_id: &str) -> bool {
        let before = self.entries.len();
        self.entries.retain(|e| e.player_id != player_id);
        self.entries.len() < before
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

// ── Leaderboard Manager ─────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct LeaderboardManager {
    boards: HashMap<String, Leaderboard>,
}

impl LeaderboardManager {
    pub fn new() -> Self {
        Self { boards: HashMap::new() }
    }

    pub fn create(&mut self, board: Leaderboard) -> Result<(), LeaderboardError> {
        let name = board.name.clone();
        if self.boards.contains_key(&name) {
            return Err(LeaderboardError::DuplicateBoard(name));
        }
        self.boards.insert(name, board);
        Ok(())
    }

    pub fn get(&self, name: &str) -> Result<&Leaderboard, LeaderboardError> {
        self.boards.get(name).ok_or_else(|| LeaderboardError::BoardNotFound(name.to_string()))
    }

    pub fn get_mut(&mut self, name: &str) -> Result<&mut Leaderboard, LeaderboardError> {
        self.boards.get_mut(name).ok_or_else(|| LeaderboardError::BoardNotFound(name.to_string()))
    }

    pub fn remove(&mut self, name: &str) -> Result<Leaderboard, LeaderboardError> {
        self.boards.remove(name).ok_or_else(|| LeaderboardError::BoardNotFound(name.to_string()))
    }

    pub fn board_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.boards.keys().cloned().collect();
        names.sort();
        names
    }

    pub fn board_count(&self) -> usize { self.boards.len() }

    /// Create time-windowed boards (daily, weekly, all-time) for a base name.
    pub fn create_windowed(&mut self, base_name: &str, sort_order: SortOrder) -> Result<(), LeaderboardError> {
        let windows = [
            (TimeWindow::Daily, "daily"),
            (TimeWindow::Weekly, "weekly"),
            (TimeWindow::AllTime, "all_time"),
        ];
        for (_, suffix) in &windows {
            let name = format!("{base_name}_{suffix}");
            if self.boards.contains_key(&name) {
                return Err(LeaderboardError::DuplicateBoard(name));
            }
            self.boards.insert(name.clone(), Leaderboard::new(&name, sort_order));
        }
        Ok(())
    }

    /// Submit to all windowed boards for a base name.
    pub fn submit_windowed(
        &mut self,
        base_name: &str,
        player_id: &str,
        score: i64,
        timestamp_ms: u64,
        metadata: &str,
    ) -> Result<Vec<(String, bool)>, LeaderboardError> {
        let suffixes = ["daily", "weekly", "all_time"];
        let mut results = Vec::new();
        for suffix in &suffixes {
            let name = format!("{base_name}_{suffix}");
            if let Some(board) = self.boards.get_mut(&name) {
                let updated = board.submit(player_id, score, timestamp_ms, metadata)?;
                results.push((name, updated));
            }
        }
        Ok(results)
    }

    /// Reset a time-windowed board (e.g., daily reset).
    pub fn reset_window(&mut self, base_name: &str, window: TimeWindow) -> Result<(), LeaderboardError> {
        let suffix = match window {
            TimeWindow::Daily => "daily",
            TimeWindow::Weekly => "weekly",
            TimeWindow::Monthly => "monthly",
            TimeWindow::AllTime => "all_time",
        };
        let name = format!("{base_name}_{suffix}");
        let board = self.boards.get_mut(&name)
            .ok_or_else(|| LeaderboardError::BoardNotFound(name.clone()))?;
        board.clear();
        Ok(())
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn populated_board() -> Leaderboard {
        let mut board = Leaderboard::new("scores", SortOrder::HighestFirst);
        board.submit("alice", 1000, 100, "").unwrap();
        board.submit("bob", 1500, 200, "").unwrap();
        board.submit("carol", 1200, 300, "").unwrap();
        board.submit("dave", 800, 400, "").unwrap();
        board.submit("eve", 1500, 500, "").unwrap(); // tied with bob
        board
    }

    #[test]
    fn submit_and_rank() {
        let board = populated_board();
        let top = board.top_n(5);
        assert_eq!(top[0].rank, 1);
        assert_eq!(top[0].score, 1500);
        // bob and eve tied at 1500, both rank 1
        assert_eq!(top[1].rank, 1);
        assert_eq!(top[2].rank, 3); // carol at 1200
    }

    #[test]
    fn player_rank() {
        let board = populated_board();
        let rank = board.player_rank("carol").unwrap();
        assert_eq!(rank.rank, 3);
        assert_eq!(rank.score, 1200);
    }

    #[test]
    fn player_not_found() {
        let board = populated_board();
        let err = board.player_rank("nobody").unwrap_err();
        assert!(matches!(err, LeaderboardError::PlayerNotFound(_)));
    }

    #[test]
    fn lowest_first() {
        let mut board = Leaderboard::new("speedrun", SortOrder::LowestFirst);
        board.submit("fast", 30, 100, "").unwrap();
        board.submit("slow", 120, 200, "").unwrap();
        board.submit("medium", 60, 300, "").unwrap();
        let top = board.top_n(3);
        assert_eq!(top[0].player_id, "fast");
        assert_eq!(top[0].rank, 1);
        assert_eq!(top[2].player_id, "slow");
    }

    #[test]
    fn submit_updates_if_better() {
        let mut board = Leaderboard::new("test", SortOrder::HighestFirst);
        board.submit("p1", 100, 10, "").unwrap();
        let updated = board.submit("p1", 200, 20, "").unwrap();
        assert!(updated);
        assert_eq!(board.player_rank("p1").unwrap().score, 200);
    }

    #[test]
    fn submit_does_not_update_if_worse() {
        let mut board = Leaderboard::new("test", SortOrder::HighestFirst);
        board.submit("p1", 200, 10, "").unwrap();
        let updated = board.submit("p1", 100, 20, "").unwrap();
        assert!(!updated);
        assert_eq!(board.player_rank("p1").unwrap().score, 200);
    }

    #[test]
    fn score_bounds_validation() {
        let mut board = Leaderboard::new("bounded", SortOrder::HighestFirst)
            .with_bounds(ScoreBounds::new(0, 1000));
        assert!(board.submit("p1", 500, 10, "").is_ok());
        let err = board.submit("p2", 2000, 20, "").unwrap_err();
        assert!(matches!(err, LeaderboardError::ScoreOutOfBounds { .. }));
        let err = board.submit("p3", -1, 30, "").unwrap_err();
        assert!(matches!(err, LeaderboardError::ScoreOutOfBounds { .. }));
    }

    #[test]
    fn rate_limiting() {
        let mut board = Leaderboard::new("rated", SortOrder::HighestFirst)
            .with_rate_limit(10_000); // 10 second cooldown
        board.submit("p1", 100, 1000, "").unwrap();
        let err = board.submit("p1", 200, 5000, "").unwrap_err();
        assert!(matches!(err, LeaderboardError::RateLimited { .. }));
        // After cooldown
        board.submit("p1", 200, 11_000, "").unwrap();
    }

    #[test]
    fn pagination() {
        let board = populated_board();
        let page0 = board.paginate(0, 2).unwrap();
        assert_eq!(page0.entries.len(), 2);
        assert_eq!(page0.page, 0);
        assert_eq!(page0.total_pages, 3);
        assert_eq!(page0.total_entries, 5);
        let page1 = board.paginate(1, 2).unwrap();
        assert_eq!(page1.entries.len(), 2);
        let page2 = board.paginate(2, 2).unwrap();
        assert_eq!(page2.entries.len(), 1);
    }

    #[test]
    fn pagination_invalid_page() {
        let board = populated_board();
        let err = board.paginate(10, 2).unwrap_err();
        assert!(matches!(err, LeaderboardError::InvalidPage { .. }));
    }

    #[test]
    fn pagination_zero_size() {
        let board = populated_board();
        let err = board.paginate(0, 0).unwrap_err();
        assert!(matches!(err, LeaderboardError::InvalidPageSize(0)));
    }

    #[test]
    fn rank_range() {
        let board = populated_board();
        let range = board.rank_range(1, 3);
        // Ranks 1, 1, 3 (two tied at rank 1, carol at rank 3)
        assert_eq!(range.len(), 3);
    }

    #[test]
    fn percentile() {
        let board = populated_board();
        // bob is rank 1 out of 5: (5-1)/5 * 100 = 80%
        let pct = board.percentile("bob").unwrap();
        assert!((pct - 80.0).abs() < 1e-6);
        // dave is rank 5 out of 5: (5-5)/5 * 100 = 0%
        let pct_d = board.percentile("dave").unwrap();
        assert!((pct_d - 0.0).abs() < 1e-6);
    }

    #[test]
    fn percentile_empty() {
        let board = Leaderboard::new("empty", SortOrder::HighestFirst);
        let err = board.percentile("anyone").unwrap_err();
        assert!(matches!(err, LeaderboardError::EmptyBoard(_)));
    }

    #[test]
    fn remove_player() {
        let mut board = populated_board();
        assert!(board.remove_player("alice"));
        assert_eq!(board.entry_count(), 4);
        assert!(!board.remove_player("nobody"));
    }

    #[test]
    fn clear_board() {
        let mut board = populated_board();
        board.clear();
        assert_eq!(board.entry_count(), 0);
    }

    #[test]
    fn top_n_more_than_entries() {
        let board = populated_board();
        let top = board.top_n(100);
        assert_eq!(top.len(), 5);
    }

    #[test]
    fn manager_create_and_get() {
        let mut mgr = LeaderboardManager::new();
        mgr.create(Leaderboard::new("scores", SortOrder::HighestFirst)).unwrap();
        assert!(mgr.get("scores").is_ok());
        assert!(mgr.get("missing").is_err());
    }

    #[test]
    fn manager_duplicate() {
        let mut mgr = LeaderboardManager::new();
        mgr.create(Leaderboard::new("scores", SortOrder::HighestFirst)).unwrap();
        let err = mgr.create(Leaderboard::new("scores", SortOrder::HighestFirst)).unwrap_err();
        assert!(matches!(err, LeaderboardError::DuplicateBoard(_)));
    }

    #[test]
    fn manager_windowed_boards() {
        let mut mgr = LeaderboardManager::new();
        mgr.create_windowed("scores", SortOrder::HighestFirst).unwrap();
        assert_eq!(mgr.board_count(), 3);
        let names = mgr.board_names();
        assert!(names.contains(&"scores_daily".to_string()));
        assert!(names.contains(&"scores_weekly".to_string()));
        assert!(names.contains(&"scores_all_time".to_string()));
    }

    #[test]
    fn submit_windowed() {
        let mut mgr = LeaderboardManager::new();
        mgr.create_windowed("scores", SortOrder::HighestFirst).unwrap();
        let results = mgr.submit_windowed("scores", "p1", 1000, 100, "").unwrap();
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|(_, updated)| *updated));
        // Verify score exists in all boards
        for (name, _) in &results {
            let board = mgr.get(name).unwrap();
            assert_eq!(board.entry_count(), 1);
        }
    }

    #[test]
    fn reset_window() {
        let mut mgr = LeaderboardManager::new();
        mgr.create_windowed("scores", SortOrder::HighestFirst).unwrap();
        mgr.submit_windowed("scores", "p1", 1000, 100, "").unwrap();
        mgr.reset_window("scores", TimeWindow::Daily).unwrap();
        assert_eq!(mgr.get("scores_daily").unwrap().entry_count(), 0);
        assert_eq!(mgr.get("scores_weekly").unwrap().entry_count(), 1); // not reset
    }

    #[test]
    fn manager_remove_board() {
        let mut mgr = LeaderboardManager::new();
        mgr.create(Leaderboard::new("temp", SortOrder::HighestFirst)).unwrap();
        let removed = mgr.remove("temp").unwrap();
        assert_eq!(removed.name(), "temp");
        assert!(mgr.get("temp").is_err());
    }

    #[test]
    fn force_set_score() {
        let mut board = Leaderboard::new("test", SortOrder::HighestFirst);
        board.submit("p1", 200, 10, "").unwrap();
        board.set_score("p1", 50, 20, "reset").unwrap();
        assert_eq!(board.player_rank("p1").unwrap().score, 50);
    }

    #[test]
    fn tie_ordering_by_timestamp() {
        let mut board = Leaderboard::new("test", SortOrder::HighestFirst);
        board.submit("late", 100, 200, "").unwrap();
        board.submit("early", 100, 100, "").unwrap();
        let top = board.top_n(2);
        // Both rank 1 (tied), but earlier timestamp first
        assert_eq!(top[0].player_id, "early");
        assert_eq!(top[1].player_id, "late");
        assert_eq!(top[0].rank, 1);
        assert_eq!(top[1].rank, 1);
    }

    #[test]
    fn metadata_preserved() {
        let mut board = Leaderboard::new("test", SortOrder::HighestFirst);
        board.submit("p1", 100, 10, "level=5").unwrap();
        let entry = board.player_rank("p1").unwrap();
        assert_eq!(entry.metadata, "level=5");
    }

    #[test]
    fn empty_board_top_n() {
        let board = Leaderboard::new("empty", SortOrder::HighestFirst);
        let top = board.top_n(10);
        assert!(top.is_empty());
    }
}
