//! ELO-based skill rating and matchmaking — expected score, K-factor decay, leaderboard.
//!
//! Replaces elo.js / chess.com-rating with pure Rust.
//! Standard ELO expected-score formula, rating update for win/loss/draw,
//! K-factor decay by games played, provisional period, leaderboard with
//! rank tracking, rating history, confidence interval, multi-player extension.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EloError {
    PlayerNotFound(String),
    DuplicatePlayer(String),
    SamePlayer,
    InvalidScore(String),
    NoHistory(String),
}

impl fmt::Display for EloError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PlayerNotFound(id) => write!(f, "player not found: {id}"),
            Self::DuplicatePlayer(id) => write!(f, "duplicate player: {id}"),
            Self::SamePlayer => write!(f, "cannot play against self"),
            Self::InvalidScore(msg) => write!(f, "invalid score: {msg}"),
            Self::NoHistory(id) => write!(f, "no history for: {id}"),
        }
    }
}

impl std::error::Error for EloError {}

// ── Match Outcome ───────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Outcome {
    Win,
    Loss,
    Draw,
}

impl Outcome {
    pub fn score(self) -> f64 {
        match self {
            Self::Win => 1.0,
            Self::Loss => 0.0,
            Self::Draw => 0.5,
        }
    }

    pub fn opposite(self) -> Self {
        match self {
            Self::Win => Self::Loss,
            Self::Loss => Self::Win,
            Self::Draw => Self::Draw,
        }
    }
}

impl fmt::Display for Outcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Win => write!(f, "Win"),
            Self::Loss => write!(f, "Loss"),
            Self::Draw => write!(f, "Draw"),
        }
    }
}

// ── ELO Rating ──────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct EloRating {
    pub player_id: String,
    pub rating: f64,
    pub games_played: u32,
    pub wins: u32,
    pub losses: u32,
    pub draws: u32,
    pub peak_rating: f64,
    history: Vec<f64>,
}

impl EloRating {
    pub fn new(player_id: &str, initial_rating: f64) -> Self {
        Self {
            player_id: player_id.to_string(),
            rating: initial_rating,
            games_played: 0,
            wins: 0,
            losses: 0,
            draws: 0,
            peak_rating: initial_rating,
            history: vec![initial_rating],
        }
    }

    pub fn is_provisional(&self) -> bool {
        self.games_played < 30
    }

    pub fn win_rate(&self) -> f64 {
        if self.games_played == 0 {
            return 0.0;
        }
        self.wins as f64 / self.games_played as f64
    }

    pub fn k_factor(&self) -> f64 {
        if self.games_played < 30 {
            40.0
        } else if self.rating < 2400.0 {
            20.0
        } else {
            10.0
        }
    }

    pub fn history(&self) -> &[f64] {
        &self.history
    }

    /// 95% confidence interval half-width based on games played.
    pub fn confidence_interval(&self) -> f64 {
        if self.games_played == 0 {
            return 400.0;
        }
        // RD approximation: decays with sqrt(games_played).
        let rd = 350.0 / (1.0 + (self.games_played as f64 / 10.0)).sqrt();
        rd * 1.96
    }

    fn record_outcome(&mut self, outcome: Outcome) {
        self.games_played += 1;
        match outcome {
            Outcome::Win => self.wins += 1,
            Outcome::Loss => self.losses += 1,
            Outcome::Draw => self.draws += 1,
        }
    }

    fn apply_delta(&mut self, delta: f64) {
        self.rating += delta;
        if self.rating > self.peak_rating {
            self.peak_rating = self.rating;
        }
        self.history.push(self.rating);
    }
}

impl fmt::Display for EloRating {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}({:.0}, {} games{})",
            self.player_id,
            self.rating,
            self.games_played,
            if self.is_provisional() { ", provisional" } else { "" }
        )
    }
}

// ── ELO Calculator ──────────────────────────────────────────────

/// Pure functions for ELO math.
pub struct EloCalc;

impl EloCalc {
    /// Expected score of player A vs player B.
    pub fn expected_score(rating_a: f64, rating_b: f64) -> f64 {
        1.0 / (1.0 + 10.0_f64.powf((rating_b - rating_a) / 400.0))
    }

    /// Rating change for player given K, expected, and actual score.
    pub fn rating_delta(k: f64, expected: f64, actual: f64) -> f64 {
        k * (actual - expected)
    }

    /// Multi-player ELO: expected score against a field of opponents.
    pub fn expected_score_multi(player_rating: f64, opponent_ratings: &[f64]) -> f64 {
        if opponent_ratings.is_empty() {
            return 0.0;
        }
        let total: f64 = opponent_ratings
            .iter()
            .map(|opp| Self::expected_score(player_rating, *opp))
            .sum();
        total / opponent_ratings.len() as f64
    }

    /// Multi-player rating delta given placement (0 = first, n-1 = last).
    pub fn multi_delta(k: f64, player_rating: f64, opponent_ratings: &[f64], placement: usize) -> f64 {
        let n = opponent_ratings.len() + 1;
        if n <= 1 {
            return 0.0;
        }
        let actual = 1.0 - (placement as f64 / (n - 1) as f64);
        let expected = Self::expected_score_multi(player_rating, opponent_ratings);
        Self::rating_delta(k, expected, actual)
    }
}

// ── ELO Leaderboard ─────────────────────────────────────────────

#[derive(Debug)]
pub struct EloLeaderboard {
    players: HashMap<String, EloRating>,
    default_rating: f64,
}

impl EloLeaderboard {
    pub fn new(default_rating: f64) -> Self {
        Self {
            players: HashMap::new(),
            default_rating,
        }
    }

    pub fn register(&mut self, player_id: &str) -> Result<(), EloError> {
        if self.players.contains_key(player_id) {
            return Err(EloError::DuplicatePlayer(player_id.to_string()));
        }
        self.players.insert(
            player_id.to_string(),
            EloRating::new(player_id, self.default_rating),
        );
        Ok(())
    }

    pub fn get(&self, player_id: &str) -> Result<&EloRating, EloError> {
        self.players
            .get(player_id)
            .ok_or_else(|| EloError::PlayerNotFound(player_id.to_string()))
    }

    pub fn player_count(&self) -> usize {
        self.players.len()
    }

    pub fn record_match(
        &mut self,
        player_a: &str,
        player_b: &str,
        outcome_a: Outcome,
    ) -> Result<(f64, f64), EloError> {
        if player_a == player_b {
            return Err(EloError::SamePlayer);
        }
        if !self.players.contains_key(player_a) {
            return Err(EloError::PlayerNotFound(player_a.to_string()));
        }
        if !self.players.contains_key(player_b) {
            return Err(EloError::PlayerNotFound(player_b.to_string()));
        }

        let ra = self.players[player_a].rating;
        let rb = self.players[player_b].rating;
        let ka = self.players[player_a].k_factor();
        let kb = self.players[player_b].k_factor();

        let ea = EloCalc::expected_score(ra, rb);
        let eb = 1.0 - ea;

        let da = EloCalc::rating_delta(ka, ea, outcome_a.score());
        let db = EloCalc::rating_delta(kb, eb, outcome_a.opposite().score());

        let pa = self.players.get_mut(player_a).unwrap();
        pa.record_outcome(outcome_a);
        pa.apply_delta(da);

        let pb = self.players.get_mut(player_b).unwrap();
        pb.record_outcome(outcome_a.opposite());
        pb.apply_delta(db);

        Ok((da, db))
    }

    pub fn record_multi_match(
        &mut self,
        placements: &[(&str, usize)],
    ) -> Result<Vec<f64>, EloError> {
        // Gather ratings.
        let ratings: Vec<(String, f64, f64)> = placements
            .iter()
            .map(|(id, _)| {
                let p = self
                    .players
                    .get(*id)
                    .ok_or_else(|| EloError::PlayerNotFound(id.to_string()))?;
                Ok((id.to_string(), p.rating, p.k_factor()))
            })
            .collect::<Result<Vec<_>, EloError>>()?;

        let mut deltas = Vec::new();
        for (i, (id, rating, k)) in ratings.iter().enumerate() {
            let opp_ratings: Vec<f64> = ratings
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, (_, r, _))| *r)
                .collect();
            let placement = placements[i].1;
            let delta = EloCalc::multi_delta(*k, *rating, &opp_ratings, placement);
            deltas.push(delta);

            let outcome = if placement == 0 {
                Outcome::Win
            } else if placement == placements.len() - 1 {
                Outcome::Loss
            } else {
                Outcome::Draw
            };

            let p = self.players.get_mut(id.as_str()).unwrap();
            p.record_outcome(outcome);
            p.apply_delta(delta);
        }
        Ok(deltas)
    }

    /// Ranked list sorted by rating descending.
    pub fn rankings(&self) -> Vec<(&str, f64, u32)> {
        let mut ranked: Vec<(&str, f64, u32)> = self
            .players
            .values()
            .map(|p| (p.player_id.as_str(), p.rating, p.games_played))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        ranked
    }

    pub fn rank_of(&self, player_id: &str) -> Result<usize, EloError> {
        let rankings = self.rankings();
        rankings
            .iter()
            .position(|(id, _, _)| *id == player_id)
            .map(|i| i + 1)
            .ok_or_else(|| EloError::PlayerNotFound(player_id.to_string()))
    }
}

impl Default for EloLeaderboard {
    fn default() -> Self {
        Self::new(1200.0)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn board() -> EloLeaderboard {
        let mut b = EloLeaderboard::new(1200.0);
        b.register("alice").unwrap();
        b.register("bob").unwrap();
        b
    }

    #[test]
    fn expected_score_equal() {
        let e = EloCalc::expected_score(1200.0, 1200.0);
        assert!((e - 0.5).abs() < 0.001);
    }

    #[test]
    fn expected_score_favors_higher() {
        let e = EloCalc::expected_score(1600.0, 1200.0);
        assert!(e > 0.9);
    }

    #[test]
    fn expected_score_symmetry() {
        let ea = EloCalc::expected_score(1400.0, 1200.0);
        let eb = EloCalc::expected_score(1200.0, 1400.0);
        assert!((ea + eb - 1.0).abs() < 0.001);
    }

    #[test]
    fn rating_update_win() {
        let mut b = board();
        let (da, db) = b.record_match("alice", "bob", Outcome::Win).unwrap();
        assert!(da > 0.0);
        assert!(db < 0.0);
        assert!(b.get("alice").unwrap().rating > 1200.0);
        assert!(b.get("bob").unwrap().rating < 1200.0);
    }

    #[test]
    fn rating_update_draw() {
        let mut b = board();
        let (da, db) = b.record_match("alice", "bob", Outcome::Draw).unwrap();
        assert!(da.abs() < 1.0);
        assert!(db.abs() < 1.0);
    }

    #[test]
    fn k_factor_decay() {
        let mut r = EloRating::new("test", 1200.0);
        assert_eq!(r.k_factor(), 40.0); // provisional
        r.games_played = 30;
        assert_eq!(r.k_factor(), 20.0); // standard
        r.rating = 2500.0;
        assert_eq!(r.k_factor(), 10.0); // master
    }

    #[test]
    fn provisional_flag() {
        let r = EloRating::new("test", 1200.0);
        assert!(r.is_provisional());
        let mut r2 = r;
        r2.games_played = 30;
        assert!(!r2.is_provisional());
    }

    #[test]
    fn peak_rating_tracked() {
        let mut b = board();
        b.record_match("alice", "bob", Outcome::Win).unwrap();
        let alice = b.get("alice").unwrap();
        assert_eq!(alice.peak_rating, alice.rating);
        b.record_match("alice", "bob", Outcome::Loss).unwrap();
        let alice2 = b.get("alice").unwrap();
        assert!(alice2.peak_rating >= alice2.rating);
    }

    #[test]
    fn rating_history() {
        let mut b = board();
        b.record_match("alice", "bob", Outcome::Win).unwrap();
        b.record_match("alice", "bob", Outcome::Loss).unwrap();
        let h = b.get("alice").unwrap().history();
        assert_eq!(h.len(), 3); // initial + 2 games
    }

    #[test]
    fn leaderboard_rankings() {
        let mut b = board();
        b.record_match("alice", "bob", Outcome::Win).unwrap();
        let ranks = b.rankings();
        assert_eq!(ranks[0].0, "alice");
        assert_eq!(ranks[1].0, "bob");
    }

    #[test]
    fn rank_of_player() {
        let mut b = board();
        b.record_match("alice", "bob", Outcome::Win).unwrap();
        assert_eq!(b.rank_of("alice").unwrap(), 1);
        assert_eq!(b.rank_of("bob").unwrap(), 2);
    }

    #[test]
    fn same_player_error() {
        let mut b = board();
        let err = b.record_match("alice", "alice", Outcome::Win).unwrap_err();
        assert!(matches!(err, EloError::SamePlayer));
    }

    #[test]
    fn player_not_found() {
        let mut b = board();
        let err = b.record_match("alice", "ghost", Outcome::Win).unwrap_err();
        assert!(matches!(err, EloError::PlayerNotFound(_)));
    }

    #[test]
    fn duplicate_register() {
        let mut b = board();
        let err = b.register("alice").unwrap_err();
        assert!(matches!(err, EloError::DuplicatePlayer(_)));
    }

    #[test]
    fn win_rate() {
        let mut b = board();
        b.record_match("alice", "bob", Outcome::Win).unwrap();
        b.record_match("alice", "bob", Outcome::Win).unwrap();
        b.record_match("alice", "bob", Outcome::Loss).unwrap();
        assert!((b.get("alice").unwrap().win_rate() - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn confidence_interval_shrinks() {
        let r0 = EloRating::new("a", 1200.0);
        let mut r30 = EloRating::new("b", 1200.0);
        r30.games_played = 30;
        assert!(r30.confidence_interval() < r0.confidence_interval());
    }

    #[test]
    fn multi_player_expected_score() {
        let e = EloCalc::expected_score_multi(1200.0, &[1200.0, 1200.0, 1200.0]);
        assert!((e - 0.5).abs() < 0.01);
    }

    #[test]
    fn multi_match() {
        let mut b = EloLeaderboard::new(1200.0);
        b.register("a").unwrap();
        b.register("b").unwrap();
        b.register("c").unwrap();
        let deltas = b.record_multi_match(&[("a", 0), ("b", 1), ("c", 2)]).unwrap();
        assert!(deltas[0] > 0.0); // winner gains
        assert!(deltas[2] < 0.0); // loser drops
    }

    #[test]
    fn display_elo_rating() {
        let r = EloRating::new("alice", 1500.0);
        let s = r.to_string();
        assert!(s.contains("alice"));
        assert!(s.contains("provisional"));
    }

    #[test]
    fn outcome_opposite() {
        assert_eq!(Outcome::Win.opposite(), Outcome::Loss);
        assert_eq!(Outcome::Loss.opposite(), Outcome::Win);
        assert_eq!(Outcome::Draw.opposite(), Outcome::Draw);
    }
}
