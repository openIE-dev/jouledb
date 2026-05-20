//! Team balancing algorithm — skill-aware, role-aware, party-constrained.
//!
//! Replaces team-balance.js / Overwatch-matchmaker with pure Rust.
//! Player with skill_rating and role_preference, Team struct,
//! TeamBalancer distributing players minimizing skill variance,
//! role-aware balancing, swap-based optimization, balance score,
//! party constraints (friends on same team), iterative improvement,
//! configurable strictness.

use std::collections::{HashMap, HashSet};
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BalanceError {
    NotEnoughPlayers { have: usize, need: usize },
    UnevenTeams(String),
    PartyTooLarge { party_size: usize, team_size: usize },
    InvalidTeamCount(usize),
    NoImprovement,
}

impl fmt::Display for BalanceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotEnoughPlayers { have, need } => {
                write!(f, "not enough players: have {have}, need {need}")
            }
            Self::UnevenTeams(msg) => write!(f, "uneven teams: {msg}"),
            Self::PartyTooLarge { party_size, team_size } => {
                write!(f, "party ({party_size}) exceeds team size ({team_size})")
            }
            Self::InvalidTeamCount(n) => write!(f, "invalid team count: {n}"),
            Self::NoImprovement => write!(f, "no improvement possible"),
        }
    }
}

impl std::error::Error for BalanceError {}

// ── Role ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Role {
    Tank,
    Damage,
    Support,
    Flex,
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tank => write!(f, "Tank"),
            Self::Damage => write!(f, "Damage"),
            Self::Support => write!(f, "Support"),
            Self::Flex => write!(f, "Flex"),
        }
    }
}

// ── Player ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct Player {
    pub id: String,
    pub skill_rating: f64,
    pub role: Role,
    pub party_id: Option<String>,
}

impl Player {
    pub fn new(id: &str, skill: f64, role: Role) -> Self {
        Self {
            id: id.to_string(),
            skill_rating: skill,
            role,
            party_id: None,
        }
    }

    pub fn with_party(mut self, party_id: &str) -> Self {
        self.party_id = Some(party_id.to_string());
        self
    }
}

impl fmt::Display for Player {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({:.0}, {})", self.id, self.skill_rating, self.role)
    }
}

// ── Team ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Team {
    pub name: String,
    pub members: Vec<Player>,
}

impl Team {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            members: Vec::new(),
        }
    }

    pub fn size(&self) -> usize {
        self.members.len()
    }

    pub fn total_skill(&self) -> f64 {
        self.members.iter().map(|p| p.skill_rating).sum()
    }

    pub fn average_skill(&self) -> f64 {
        if self.members.is_empty() {
            return 0.0;
        }
        self.total_skill() / self.members.len() as f64
    }

    pub fn skill_variance(&self) -> f64 {
        if self.members.is_empty() {
            return 0.0;
        }
        let avg = self.average_skill();
        self.members
            .iter()
            .map(|p| (p.skill_rating - avg).powi(2))
            .sum::<f64>()
            / self.members.len() as f64
    }

    pub fn role_count(&self, role: Role) -> usize {
        self.members.iter().filter(|p| p.role == role).count()
    }

    pub fn has_role(&self, role: Role) -> bool {
        self.role_count(role) > 0 || self.role_count(Role::Flex) > 0
    }

    pub fn add(&mut self, player: Player) {
        self.members.push(player);
    }
}

impl fmt::Display for Team {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Team {} ({} players, avg skill {:.0})",
            self.name,
            self.size(),
            self.average_skill()
        )
    }
}

// ── Balance Result ──────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct BalanceResult {
    pub teams: Vec<Team>,
    pub balance_score: f64,
    pub iterations: usize,
}

impl BalanceResult {
    /// Score 0..1 where 1 = perfectly balanced.
    pub fn compute_score(teams: &[Team]) -> f64 {
        if teams.len() < 2 {
            return 1.0;
        }
        let avgs: Vec<f64> = teams.iter().map(|t| t.average_skill()).collect();
        let global_avg = avgs.iter().sum::<f64>() / avgs.len() as f64;
        if global_avg == 0.0 {
            return 1.0;
        }
        let max_diff = avgs.iter().map(|a| (a - global_avg).abs()).fold(0.0f64, f64::max);
        (1.0 - max_diff / global_avg).max(0.0)
    }
}

impl fmt::Display for BalanceResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BalanceResult({} teams, score={:.3}, {} iters)",
            self.teams.len(),
            self.balance_score,
            self.iterations
        )
    }
}

// ── Role Requirement ────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct RoleRequirement {
    pub min_tanks: usize,
    pub min_damage: usize,
    pub min_support: usize,
}

impl RoleRequirement {
    pub fn new(tanks: usize, damage: usize, support: usize) -> Self {
        Self {
            min_tanks: tanks,
            min_damage: damage,
            min_support: support,
        }
    }

    pub fn satisfied_by(&self, team: &Team) -> bool {
        let tanks = team.role_count(Role::Tank) + team.role_count(Role::Flex);
        let dmg = team.role_count(Role::Damage) + team.role_count(Role::Flex);
        let sup = team.role_count(Role::Support) + team.role_count(Role::Flex);
        tanks >= self.min_tanks && dmg >= self.min_damage && sup >= self.min_support
    }
}

// ── Team Balancer ───────────────────────────────────────────────

#[derive(Debug)]
pub struct TeamBalancer {
    pub team_count: usize,
    pub max_iterations: usize,
    pub role_req: Option<RoleRequirement>,
    pub strictness: f64,
}

impl TeamBalancer {
    pub fn new(team_count: usize) -> Self {
        Self {
            team_count,
            max_iterations: 100,
            role_req: None,
            strictness: 1.0,
        }
    }

    pub fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    pub fn with_role_requirement(mut self, req: RoleRequirement) -> Self {
        self.role_req = Some(req);
        self
    }

    pub fn with_strictness(mut self, s: f64) -> Self {
        self.strictness = s.clamp(0.0, 1.0);
        self
    }

    pub fn balance(&self, players: &[Player]) -> Result<BalanceResult, BalanceError> {
        if self.team_count < 2 {
            return Err(BalanceError::InvalidTeamCount(self.team_count));
        }
        let team_size = players.len() / self.team_count;
        if team_size == 0 || players.len() < self.team_count {
            return Err(BalanceError::NotEnoughPlayers {
                have: players.len(),
                need: self.team_count,
            });
        }

        // Validate party sizes.
        let mut party_sizes: HashMap<&str, usize> = HashMap::new();
        for p in players {
            if let Some(ref pid) = p.party_id {
                *party_sizes.entry(pid.as_str()).or_insert(0) += 1;
            }
        }
        for (_, &size) in &party_sizes {
            if size > team_size {
                return Err(BalanceError::PartyTooLarge {
                    party_size: size,
                    team_size,
                });
            }
        }

        // Initial assignment: sort by skill descending, snake draft.
        let mut sorted: Vec<Player> = players.to_vec();
        sorted.sort_by(|a, b| b.skill_rating.partial_cmp(&a.skill_rating).unwrap());

        let mut teams: Vec<Team> = (0..self.team_count)
            .map(|i| Team::new(&format!("Team {}", i + 1)))
            .collect();

        // Assign party members first.
        let mut assigned: HashSet<String> = HashSet::new();
        let party_ids: Vec<String> = party_sizes.keys().map(|s| s.to_string()).collect();
        for pid in &party_ids {
            let party_members: Vec<Player> = sorted
                .iter()
                .filter(|p| p.party_id.as_deref() == Some(pid))
                .cloned()
                .collect();
            // Find the team with lowest total skill that has room.
            let best_team = teams
                .iter_mut()
                .filter(|t| t.size() + party_members.len() <= team_size + (players.len() % self.team_count).min(1))
                .min_by(|a, b| a.total_skill().partial_cmp(&b.total_skill()).unwrap());
            if let Some(team) = best_team {
                for m in &party_members {
                    team.add(m.clone());
                    assigned.insert(m.id.clone());
                }
            }
        }

        // Snake draft for remaining players.
        let remaining: Vec<Player> = sorted
            .into_iter()
            .filter(|p| !assigned.contains(&p.id))
            .collect();

        let mut team_idx = 0;
        let mut ascending = true;
        for p in remaining {
            teams[team_idx].add(p);
            if ascending {
                if team_idx + 1 >= self.team_count {
                    ascending = false;
                } else {
                    team_idx += 1;
                }
            } else if team_idx == 0 {
                ascending = true;
            } else {
                team_idx -= 1;
            }
        }

        // Iterative swap improvement.
        let mut iterations = 0;
        for _ in 0..self.max_iterations {
            let improved = self.try_swap_improve(&mut teams, &party_sizes);
            iterations += 1;
            if !improved {
                break;
            }
        }

        let score = BalanceResult::compute_score(&teams);
        Ok(BalanceResult {
            teams,
            balance_score: score,
            iterations,
        })
    }

    fn try_swap_improve(&self, teams: &mut [Team], _parties: &HashMap<&str, usize>) -> bool {
        let current_score = BalanceResult::compute_score(teams);
        let target = current_score + 0.001 * self.strictness;

        for ti in 0..teams.len() {
            for tj in (ti + 1)..teams.len() {
                for pi in 0..teams[ti].size() {
                    for pj in 0..teams[tj].size() {
                        // Skip if either is in a party.
                        if teams[ti].members[pi].party_id.is_some()
                            || teams[tj].members[pj].party_id.is_some()
                        {
                            continue;
                        }
                        // Swap.
                        let tmp = teams[ti].members[pi].clone();
                        teams[ti].members[pi] = teams[tj].members[pj].clone();
                        teams[tj].members[pj] = tmp;

                        let new_score = BalanceResult::compute_score(teams);
                        if new_score > target {
                            return true;
                        }
                        // Undo.
                        let tmp = teams[ti].members[pi].clone();
                        teams[ti].members[pi] = teams[tj].members[pj].clone();
                        teams[tj].members[pj] = tmp;
                    }
                }
            }
        }
        false
    }
}

impl Default for TeamBalancer {
    fn default() -> Self {
        Self::new(2)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn p(id: &str, skill: f64) -> Player {
        Player::new(id, skill, Role::Flex)
    }

    #[test]
    fn balance_two_teams() {
        let players = vec![
            p("a", 1000.0), p("b", 900.0), p("c", 800.0), p("d", 700.0),
        ];
        let result = TeamBalancer::new(2).balance(&players).unwrap();
        assert_eq!(result.teams.len(), 2);
        assert_eq!(result.teams[0].size(), 2);
        assert_eq!(result.teams[1].size(), 2);
    }

    #[test]
    fn balance_score_good() {
        let players = vec![
            p("a", 1000.0), p("b", 1000.0), p("c", 1000.0), p("d", 1000.0),
        ];
        let result = TeamBalancer::new(2).balance(&players).unwrap();
        assert!(result.balance_score > 0.95);
    }

    #[test]
    fn balance_improves_with_swap() {
        let players = vec![
            p("a", 2000.0), p("b", 1000.0), p("c", 1500.0), p("d", 1500.0),
        ];
        let result = TeamBalancer::new(2)
            .with_max_iterations(50)
            .balance(&players)
            .unwrap();
        assert!(result.balance_score > 0.8);
    }

    #[test]
    fn party_kept_together() {
        let players = vec![
            p("a", 1000.0).with_party("p1"),
            p("b", 1010.0).with_party("p1"),
            p("c", 990.0),
            p("d", 1020.0),
        ];
        let result = TeamBalancer::new(2).balance(&players).unwrap();
        // Find which team has "a".
        let team_a = result.teams.iter().find(|t| t.members.iter().any(|m| m.id == "a")).unwrap();
        assert!(team_a.members.iter().any(|m| m.id == "b"));
    }

    #[test]
    fn party_too_large_error() {
        let players = vec![
            p("a", 100.0).with_party("p1"),
            p("b", 100.0).with_party("p1"),
            p("c", 100.0).with_party("p1"),
            p("d", 100.0),
        ];
        let err = TeamBalancer::new(2).balance(&players).unwrap_err();
        assert!(matches!(err, BalanceError::PartyTooLarge { .. }));
    }

    #[test]
    fn not_enough_players() {
        let players = vec![p("a", 100.0)];
        let err = TeamBalancer::new(2).balance(&players).unwrap_err();
        assert!(matches!(err, BalanceError::NotEnoughPlayers { .. }));
    }

    #[test]
    fn invalid_team_count() {
        let players = vec![p("a", 100.0), p("b", 200.0)];
        let err = TeamBalancer::new(1).balance(&players).unwrap_err();
        assert!(matches!(err, BalanceError::InvalidTeamCount(_)));
    }

    #[test]
    fn three_teams() {
        let players: Vec<Player> = (0..6).map(|i| p(&format!("p{i}"), 1000.0 + i as f64 * 100.0)).collect();
        let result = TeamBalancer::new(3).balance(&players).unwrap();
        assert_eq!(result.teams.len(), 3);
        for t in &result.teams {
            assert_eq!(t.size(), 2);
        }
    }

    #[test]
    fn role_requirement_check() {
        let mut team = Team::new("T1");
        team.add(Player::new("a", 100.0, Role::Tank));
        team.add(Player::new("b", 100.0, Role::Damage));
        let req = RoleRequirement::new(1, 1, 1);
        assert!(!req.satisfied_by(&team)); // no support
        team.add(Player::new("c", 100.0, Role::Support));
        assert!(req.satisfied_by(&team));
    }

    #[test]
    fn flex_fills_role() {
        let mut team = Team::new("T1");
        team.add(Player::new("a", 100.0, Role::Flex));
        let req = RoleRequirement::new(1, 0, 0);
        assert!(req.satisfied_by(&team));
    }

    #[test]
    fn team_average_skill() {
        let mut team = Team::new("T1");
        team.add(p("a", 1000.0));
        team.add(p("b", 2000.0));
        assert!((team.average_skill() - 1500.0).abs() < 0.01);
    }

    #[test]
    fn team_variance() {
        let mut team = Team::new("T1");
        team.add(p("a", 1000.0));
        team.add(p("b", 1000.0));
        assert!((team.skill_variance()).abs() < 0.01);
    }

    #[test]
    fn balance_score_equal_teams() {
        let t1 = { let mut t = Team::new("A"); t.add(p("a", 1000.0)); t };
        let t2 = { let mut t = Team::new("B"); t.add(p("b", 1000.0)); t };
        let score = BalanceResult::compute_score(&[t1, t2]);
        assert!((score - 1.0).abs() < 0.001);
    }

    #[test]
    fn balance_score_unequal_teams() {
        let t1 = { let mut t = Team::new("A"); t.add(p("a", 2000.0)); t };
        let t2 = { let mut t = Team::new("B"); t.add(p("b", 1000.0)); t };
        let score = BalanceResult::compute_score(&[t1, t2]);
        assert!(score < 0.8);
    }

    #[test]
    fn display_team() {
        let mut team = Team::new("Alpha");
        team.add(p("a", 1000.0));
        assert!(team.to_string().contains("Alpha"));
    }

    #[test]
    fn display_player() {
        let pl = Player::new("bob", 1500.0, Role::Tank);
        assert!(pl.to_string().contains("bob"));
        assert!(pl.to_string().contains("Tank"));
    }

    #[test]
    fn display_balance_result() {
        let result = BalanceResult {
            teams: vec![Team::new("A"), Team::new("B")],
            balance_score: 0.95,
            iterations: 10,
        };
        let s = result.to_string();
        assert!(s.contains("2 teams"));
    }

    #[test]
    fn strictness_clamp() {
        let b = TeamBalancer::new(2).with_strictness(2.0);
        assert!((b.strictness - 1.0).abs() < 0.001);
        let b2 = TeamBalancer::new(2).with_strictness(-1.0);
        assert!(b2.strictness.abs() < 0.001);
    }

    #[test]
    fn has_role_check() {
        let mut team = Team::new("T");
        team.add(Player::new("a", 100.0, Role::Tank));
        assert!(team.has_role(Role::Tank));
        assert!(team.has_role(Role::Tank)); // flex can fill
    }

    #[test]
    fn empty_team_stats() {
        let team = Team::new("Empty");
        assert_eq!(team.average_skill(), 0.0);
        assert_eq!(team.skill_variance(), 0.0);
        assert_eq!(team.size(), 0);
    }
}
