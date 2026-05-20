//! HDC-powered Gaming and Interactive Entertainment module
//!
//! Provides holographic encoding for:
//! - Player behavior profiling and matchmaking
//! - Game state similarity and session analysis
//! - Cheat/anomaly detection
//! - Item and achievement recommendation

use joule_db_hdc::{BinaryHV, BundleAccumulator};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const DIMENSION: usize = 10000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Genre {
    FPS,
    MOBA,
    RPG,
    Strategy,
    Sports,
    Racing,
    Puzzle,
    Sandbox,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SkillTier {
    Bronze,
    Silver,
    Gold,
    Platinum,
    Diamond,
    Master,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PlayerRole {
    Tank,
    DPS,
    Support,
    Healer,
    Assassin,
    Controller,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SessionOutcome {
    Win,
    Loss,
    Draw,
    Abandoned,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ItemRarity {
    Common,
    Uncommon,
    Rare,
    Epic,
    Legendary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Player {
    pub id: String,
    pub username: String,
    pub skill_tier: SkillTier,
    pub mmr: u32,
    pub preferred_roles: Vec<PlayerRole>,
    pub total_matches: u32,
    pub win_rate: f32,
    pub avg_session_mins: f32,
    pub playtime_hours: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameSession {
    pub id: String,
    pub player_id: String,
    pub genre: Genre,
    pub role: PlayerRole,
    pub outcome: SessionOutcome,
    pub duration_secs: u32,
    pub kills: u32,
    pub deaths: u32,
    pub assists: u32,
    pub score: u32,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameItem {
    pub id: String,
    pub name: String,
    pub rarity: ItemRarity,
    pub level_req: u32,
    pub power: u32,
    pub attributes: HashMap<String, f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Achievement {
    pub id: String,
    pub name: String,
    pub genre: Genre,
    pub difficulty: f32,
    pub completion_rate: f32,
}

joule_db_hdc::define_domain_module! {
    /// HDC encoder for gaming domain data
    pub struct GamingLink {
        seed: 0x6A3E_0001,
        dimension: 10000,
        fields: ["player", "session", "item", "achievement", "role", "genre", "outcome"],
        scalars: ["mmr", "winrate", "kda", "score", "duration", "level", "power", "playtime"],
        enums: {
            genre_vectors: Genre => [Genre::FPS, Genre::MOBA, Genre::RPG, Genre::Strategy, Genre::Sports, Genre::Racing, Genre::Puzzle, Genre::Sandbox],
            tier_vectors: SkillTier => [SkillTier::Bronze, SkillTier::Silver, SkillTier::Gold, SkillTier::Platinum, SkillTier::Diamond, SkillTier::Master],
            role_vectors: PlayerRole => [PlayerRole::Tank, PlayerRole::DPS, PlayerRole::Support, PlayerRole::Healer, PlayerRole::Assassin, PlayerRole::Controller],
            outcome_vectors: SessionOutcome => [SessionOutcome::Win, SessionOutcome::Loss, SessionOutcome::Draw, SessionOutcome::Abandoned],
            rarity_vectors: ItemRarity => [ItemRarity::Common, ItemRarity::Uncommon, ItemRarity::Rare, ItemRarity::Epic, ItemRarity::Legendary]
        },
        dynamic: {
            character_vectors: "character",
            map_vectors: "map"
        },
    }
}

impl GamingLink {
    pub fn encode_player(&self, player: &Player) -> BinaryHV {
        let tier_hv = self.field_vectors["player"].bind(&self.tier_vectors[&player.skill_tier]);
        let mmr_hv = self.encode_scalar("mmr", player.mmr, 5000);
        let wr_hv = self.encode_scalar("winrate", (player.win_rate * 100.0) as u32, 100);
        let playtime_hv = self.encode_scalar("playtime", player.playtime_hours.min(10000), 10000);
        let mut components = vec![tier_hv, mmr_hv, wr_hv, playtime_hv];
        // Encode preferred roles
        for role in &player.preferred_roles {
            components.push(self.field_vectors["role"].bind(&self.role_vectors[role]));
        }
        self.bundle(&components)
    }

    pub fn encode_session(&self, session: &GameSession) -> BinaryHV {
        let genre_hv = self.field_vectors["genre"].bind(&self.genre_vectors[&session.genre]);
        let role_hv = self.field_vectors["role"].bind(&self.role_vectors[&session.role]);
        let outcome_hv =
            self.field_vectors["outcome"].bind(&self.outcome_vectors[&session.outcome]);
        let dur_hv = self.encode_scalar("duration", session.duration_secs, 7200);
        // KDA ratio encoded: (kills + assists) / max(1, deaths)
        let kda = ((session.kills + session.assists) as f32 / (session.deaths.max(1)) as f32 * 10.0)
            as u32;
        let kda_hv = self.encode_scalar("kda", kda.min(100), 100);
        let score_hv = self.encode_scalar("score", session.score.min(10000), 10000);
        self.bundle(&[genre_hv, role_hv, outcome_hv, dur_hv, kda_hv, score_hv])
    }

    pub fn encode_item(&self, item: &GameItem) -> BinaryHV {
        let rarity_hv = self.field_vectors["item"].bind(&self.rarity_vectors[&item.rarity]);
        let level_hv = self.encode_scalar("level", item.level_req.min(100), 100);
        let power_hv = self.encode_scalar("power", item.power.min(10000), 10000);
        self.bundle(&[rarity_hv, level_hv, power_hv])
    }

    pub fn encode_achievement(&self, ach: &Achievement) -> BinaryHV {
        let genre_hv = self.field_vectors["achievement"].bind(&self.genre_vectors[&ach.genre]);
        let diff_hv = self.encode_scalar("level", (ach.difficulty * 100.0) as u32, 100);
        self.bundle(&[genre_hv, diff_hv])
    }
}

/// Matchmaking system using HDC player profiles.
pub struct Matchmaker {
    encoder: GamingLink,
    pool: Vec<(String, BinaryHV)>,
}

impl Matchmaker {
    pub fn new() -> Self {
        Self {
            encoder: GamingLink::new(),
            pool: Vec::new(),
        }
    }

    pub fn add_player(&mut self, player: &Player) {
        let hv = self.encoder.encode_player(player);
        self.pool.push((player.id.clone(), hv));
    }

    pub fn clear(&mut self) {
        self.pool.clear();
    }

    /// Find the best match for a player from the pool.
    pub fn find_match(&self, player: &Player, team_size: usize) -> Vec<(String, f32)> {
        let query_hv = self.encoder.encode_player(player);
        let mut candidates: Vec<(String, f32)> = self
            .pool
            .iter()
            .filter(|(id, _)| *id != player.id)
            .map(|(id, hv)| (id.clone(), query_hv.similarity(hv)))
            .collect();
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(team_size);
        candidates
    }
}

impl Default for Matchmaker {
    fn default() -> Self {
        Self::new()
    }
}

/// Anomaly detector for cheat/bot detection using session behavior profiles.
pub struct BehaviorDetector {
    encoder: GamingLink,
    normal_profile: BundleAccumulator,
    sample_count: usize,
    threshold: f32,
}

impl BehaviorDetector {
    pub fn new(threshold: f32) -> Self {
        Self {
            encoder: GamingLink::new(),
            normal_profile: BundleAccumulator::new(DIMENSION),
            sample_count: 0,
            threshold,
        }
    }

    pub fn learn_normal(&mut self, session: &GameSession) {
        let hv = self.encoder.encode_session(session);
        self.normal_profile.add(&hv);
        self.sample_count += 1;
    }

    /// Returns an anomaly score; higher means more suspicious.
    /// Returns `None` if not enough training data.
    pub fn detect(&self, session: &GameSession) -> Option<f32> {
        if self.sample_count < 5 {
            return None;
        }
        let hv = self.encoder.encode_session(session);
        let normal_hv = self.normal_profile.threshold();
        let sim = hv.similarity(&normal_hv);
        let anomaly_score = 1.0 - sim;
        if anomaly_score > self.threshold {
            Some(anomaly_score)
        } else {
            None
        }
    }
}

impl Default for BehaviorDetector {
    fn default() -> Self {
        Self::new(0.3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_player() -> Player {
        Player {
            id: "p1".to_string(),
            username: "ProGamer".to_string(),
            skill_tier: SkillTier::Gold,
            mmr: 1500,
            preferred_roles: vec![PlayerRole::DPS, PlayerRole::Assassin],
            total_matches: 200,
            win_rate: 0.55,
            avg_session_mins: 30.0,
            playtime_hours: 500,
        }
    }

    fn sample_session() -> GameSession {
        GameSession {
            id: "s1".to_string(),
            player_id: "p1".to_string(),
            genre: Genre::MOBA,
            role: PlayerRole::DPS,
            outcome: SessionOutcome::Win,
            duration_secs: 1800,
            kills: 12,
            deaths: 4,
            assists: 8,
            score: 2500,
            timestamp: 1700000000,
        }
    }

    #[test]
    fn test_player_encoding() {
        let encoder = GamingLink::new();
        let hv = encoder.encode_player(&sample_player());
        assert_eq!(hv.dimension(), DIMENSION);
    }

    #[test]
    fn test_session_encoding() {
        let encoder = GamingLink::new();
        let hv = encoder.encode_session(&sample_session());
        assert_eq!(hv.dimension(), DIMENSION);
    }

    #[test]
    fn test_item_encoding() {
        let encoder = GamingLink::new();
        let item = GameItem {
            id: "i1".to_string(),
            name: "Dragon Sword".to_string(),
            rarity: ItemRarity::Epic,
            level_req: 30,
            power: 850,
            attributes: HashMap::from([("damage".to_string(), 120.0), ("speed".to_string(), 1.5)]),
        };
        let hv = encoder.encode_item(&item);
        assert_eq!(hv.dimension(), DIMENSION);
    }

    #[test]
    fn test_achievement_encoding() {
        let encoder = GamingLink::new();
        let ach = Achievement {
            id: "a1".to_string(),
            name: "First Blood".to_string(),
            genre: Genre::FPS,
            difficulty: 0.3,
            completion_rate: 0.85,
        };
        let hv = encoder.encode_achievement(&ach);
        assert_eq!(hv.dimension(), DIMENSION);
    }

    #[test]
    fn test_similar_players_match() {
        let encoder = GamingLink::new();
        let p1 = sample_player();
        let p2 = Player {
            id: "p2".to_string(),
            username: "Similar".to_string(),
            mmr: 1520,
            win_rate: 0.53,
            ..p1.clone()
        };
        let p3 = Player {
            id: "p3".to_string(),
            username: "Different".to_string(),
            skill_tier: SkillTier::Bronze,
            mmr: 400,
            win_rate: 0.30,
            preferred_roles: vec![PlayerRole::Support],
            ..p1.clone()
        };

        let hv1 = encoder.encode_player(&p1);
        let hv2 = encoder.encode_player(&p2);
        let hv3 = encoder.encode_player(&p3);

        let sim_close = hv1.similarity(&hv2);
        let sim_far = hv1.similarity(&hv3);
        assert!(
            sim_close > sim_far,
            "Similar players should have higher similarity: close={sim_close} far={sim_far}"
        );
    }

    #[test]
    fn test_matchmaker() {
        let mut mm = Matchmaker::new();
        mm.add_player(&Player {
            id: "a".to_string(),
            username: "A".to_string(),
            skill_tier: SkillTier::Gold,
            mmr: 1500,
            preferred_roles: vec![PlayerRole::DPS],
            total_matches: 100,
            win_rate: 0.5,
            avg_session_mins: 25.0,
            playtime_hours: 300,
        });
        mm.add_player(&Player {
            id: "b".to_string(),
            username: "B".to_string(),
            skill_tier: SkillTier::Gold,
            mmr: 1480,
            preferred_roles: vec![PlayerRole::Support],
            total_matches: 80,
            win_rate: 0.52,
            avg_session_mins: 28.0,
            playtime_hours: 250,
        });
        mm.add_player(&Player {
            id: "c".to_string(),
            username: "C".to_string(),
            skill_tier: SkillTier::Bronze,
            mmr: 400,
            preferred_roles: vec![PlayerRole::Tank],
            total_matches: 20,
            win_rate: 0.35,
            avg_session_mins: 15.0,
            playtime_hours: 50,
        });

        let query = Player {
            id: "q".to_string(),
            username: "Q".to_string(),
            skill_tier: SkillTier::Gold,
            mmr: 1510,
            preferred_roles: vec![PlayerRole::DPS],
            total_matches: 120,
            win_rate: 0.54,
            avg_session_mins: 30.0,
            playtime_hours: 400,
        };
        let matches = mm.find_match(&query, 2);
        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn test_behavior_detector_needs_samples() {
        let detector = BehaviorDetector::new(0.3);
        assert!(
            detector.detect(&sample_session()).is_none(),
            "Should return None with < 5 samples"
        );
    }

    #[test]
    fn test_behavior_detector_normal() {
        let mut detector = BehaviorDetector::new(0.3);
        let session = sample_session();
        for _ in 0..10 {
            detector.learn_normal(&session);
        }
        // Same session pattern should not be flagged
        let result = detector.detect(&session);
        assert!(
            result.is_none(),
            "Normal behavior should not be flagged as anomalous"
        );
    }

    #[test]
    fn test_deterministic_encoding() {
        let enc1 = GamingLink::new();
        let enc2 = GamingLink::new();
        let player = sample_player();
        let sim = enc1
            .encode_player(&player)
            .similarity(&enc2.encode_player(&player));
        assert_eq!(sim, 1.0, "Same seed should produce identical encodings");
    }
}
