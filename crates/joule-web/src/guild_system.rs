//! Guild/clan system — guilds, ranks, permissions, message board, treasury.
//!
//! Replaces WoW guild API / Discord server management with pure Rust.
//! Guild lifecycle, ranked membership with permissions, message boards,
//! treasury, search, alliances, and activity logging.

use std::collections::{HashMap, HashSet};
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GuildError {
    GuildNotFound(String),
    DuplicateGuild(String),
    MemberNotFound(String),
    AlreadyMember(String),
    GuildFull(String),
    InsufficientRank(String),
    RankNotFound(String),
    CannotDemoteLeader,
    CannotKickLeader,
    AlreadyAllied(String, String),
    NotAllied(String, String),
}

impl fmt::Display for GuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GuildNotFound(id) => write!(f, "guild not found: {id}"),
            Self::DuplicateGuild(id) => write!(f, "duplicate guild: {id}"),
            Self::MemberNotFound(u) => write!(f, "member not found: {u}"),
            Self::AlreadyMember(u) => write!(f, "already a member: {u}"),
            Self::GuildFull(id) => write!(f, "guild is full: {id}"),
            Self::InsufficientRank(u) => write!(f, "insufficient rank: {u}"),
            Self::RankNotFound(r) => write!(f, "rank not found: {r}"),
            Self::CannotDemoteLeader => write!(f, "cannot demote guild leader"),
            Self::CannotKickLeader => write!(f, "cannot kick guild leader"),
            Self::AlreadyAllied(a, b) => write!(f, "already allied: {a} and {b}"),
            Self::NotAllied(a, b) => write!(f, "not allied: {a} and {b}"),
        }
    }
}

impl std::error::Error for GuildError {}

// ── GuildPermission ─────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GuildPermission {
    Invite,
    Kick,
    Promote,
    ManageBoard,
    ManageTreasury,
    ManageAlliances,
}

// ── GuildRank ───────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct GuildRank {
    pub name: String,
    pub level: u8,
    pub permissions: HashSet<GuildPermission>,
}

impl GuildRank {
    pub fn new(name: &str, level: u8) -> Self {
        Self {
            name: name.to_string(),
            level,
            permissions: HashSet::new(),
        }
    }

    pub fn with_permission(mut self, perm: GuildPermission) -> Self {
        self.permissions.insert(perm);
        self
    }

    pub fn has_permission(&self, perm: GuildPermission) -> bool {
        self.permissions.contains(&perm)
    }
}

impl fmt::Display for GuildRank {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (level {})", self.name, self.level)
    }
}

// ── GuildMember ─────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct GuildMember {
    pub user_id: String,
    pub rank_name: String,
    pub joined_at: u64,
}

// ── BoardPost ───────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct BoardPost {
    pub id: u64,
    pub author: String,
    pub content: String,
    pub posted_at: u64,
}

impl fmt::Display for BoardPost {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}: {}", self.posted_at, self.author, self.content)
    }
}

// ── ActivityEntry ───────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ActivityEntry {
    pub description: String,
    pub timestamp: u64,
}

// ── Guild ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Guild {
    pub id: String,
    pub name: String,
    pub tag: String,
    pub leader: String,
    pub max_members: usize,
    ranks: Vec<GuildRank>,
    members: HashMap<String, GuildMember>,
    board: Vec<BoardPost>,
    next_post_id: u64,
    treasury: u64,
    alliances: HashSet<String>,
    activity_log: Vec<ActivityEntry>,
}

impl Guild {
    pub fn new(id: &str, name: &str, tag: &str, leader: &str, max_members: usize, created_at: u64) -> Self {
        let leader_rank = GuildRank::new("Leader", 100)
            .with_permission(GuildPermission::Invite)
            .with_permission(GuildPermission::Kick)
            .with_permission(GuildPermission::Promote)
            .with_permission(GuildPermission::ManageBoard)
            .with_permission(GuildPermission::ManageTreasury)
            .with_permission(GuildPermission::ManageAlliances);
        let member_rank = GuildRank::new("Member", 1);

        let mut members = HashMap::new();
        members.insert(leader.to_string(), GuildMember {
            user_id: leader.to_string(),
            rank_name: "Leader".to_string(),
            joined_at: created_at,
        });

        Self {
            id: id.to_string(),
            name: name.to_string(),
            tag: tag.to_string(),
            leader: leader.to_string(),
            max_members,
            ranks: vec![leader_rank, member_rank],
            members,
            board: Vec::new(),
            next_post_id: 1,
            treasury: 0,
            alliances: HashSet::new(),
            activity_log: Vec::new(),
        }
    }

    pub fn member_count(&self) -> usize {
        self.members.len()
    }

    pub fn is_member(&self, user_id: &str) -> bool {
        self.members.contains_key(user_id)
    }

    pub fn get_member(&self, user_id: &str) -> Option<&GuildMember> {
        self.members.get(user_id)
    }

    fn get_rank(&self, name: &str) -> Option<&GuildRank> {
        self.ranks.iter().find(|r| r.name == name)
    }

    fn member_has_permission(&self, user_id: &str, perm: GuildPermission) -> bool {
        let Some(member) = self.members.get(user_id) else { return false };
        let Some(rank) = self.get_rank(&member.rank_name) else { return false };
        rank.has_permission(perm)
    }

    pub fn join(&mut self, user_id: &str, joined_at: u64) -> Result<(), GuildError> {
        if self.members.contains_key(user_id) {
            return Err(GuildError::AlreadyMember(user_id.to_string()));
        }
        if self.members.len() >= self.max_members {
            return Err(GuildError::GuildFull(self.id.clone()));
        }
        self.members.insert(user_id.to_string(), GuildMember {
            user_id: user_id.to_string(),
            rank_name: "Member".to_string(),
            joined_at,
        });
        self.log_activity(&format!("{user_id} joined"), joined_at);
        Ok(())
    }

    pub fn leave(&mut self, user_id: &str, timestamp: u64) -> Result<(), GuildError> {
        if !self.members.contains_key(user_id) {
            return Err(GuildError::MemberNotFound(user_id.to_string()));
        }
        if user_id == self.leader {
            return Err(GuildError::CannotKickLeader);
        }
        self.members.remove(user_id);
        self.log_activity(&format!("{user_id} left"), timestamp);
        Ok(())
    }

    pub fn kick(&mut self, kicker: &str, target: &str, timestamp: u64) -> Result<(), GuildError> {
        if target == self.leader {
            return Err(GuildError::CannotKickLeader);
        }
        if !self.member_has_permission(kicker, GuildPermission::Kick) {
            return Err(GuildError::InsufficientRank(kicker.to_string()));
        }
        if !self.members.contains_key(target) {
            return Err(GuildError::MemberNotFound(target.to_string()));
        }
        self.members.remove(target);
        self.log_activity(&format!("{target} kicked by {kicker}"), timestamp);
        Ok(())
    }

    pub fn promote(&mut self, promoter: &str, target: &str, new_rank: &str, timestamp: u64) -> Result<(), GuildError> {
        if !self.member_has_permission(promoter, GuildPermission::Promote) {
            return Err(GuildError::InsufficientRank(promoter.to_string()));
        }
        if self.get_rank(new_rank).is_none() {
            return Err(GuildError::RankNotFound(new_rank.to_string()));
        }
        let member = self.members.get_mut(target).ok_or_else(|| GuildError::MemberNotFound(target.to_string()))?;
        member.rank_name = new_rank.to_string();
        self.log_activity(&format!("{target} promoted to {new_rank}"), timestamp);
        Ok(())
    }

    pub fn demote(&mut self, demoter: &str, target: &str, new_rank: &str, timestamp: u64) -> Result<(), GuildError> {
        if target == self.leader {
            return Err(GuildError::CannotDemoteLeader);
        }
        if !self.member_has_permission(demoter, GuildPermission::Promote) {
            return Err(GuildError::InsufficientRank(demoter.to_string()));
        }
        if self.get_rank(new_rank).is_none() {
            return Err(GuildError::RankNotFound(new_rank.to_string()));
        }
        let member = self.members.get_mut(target).ok_or_else(|| GuildError::MemberNotFound(target.to_string()))?;
        member.rank_name = new_rank.to_string();
        self.log_activity(&format!("{target} demoted to {new_rank}"), timestamp);
        Ok(())
    }

    pub fn add_rank(&mut self, rank: GuildRank) {
        self.ranks.push(rank);
    }

    pub fn post_to_board(&mut self, author: &str, content: &str, posted_at: u64) -> Result<u64, GuildError> {
        if !self.is_member(author) {
            return Err(GuildError::MemberNotFound(author.to_string()));
        }
        let id = self.next_post_id;
        self.next_post_id += 1;
        self.board.push(BoardPost { id, author: author.to_string(), content: content.to_string(), posted_at });
        Ok(id)
    }

    pub fn board_posts(&self) -> &[BoardPost] {
        &self.board
    }

    pub fn deposit(&mut self, amount: u64) {
        self.treasury = self.treasury.saturating_add(amount);
    }

    pub fn withdraw(&mut self, amount: u64) -> bool {
        if self.treasury >= amount {
            self.treasury -= amount;
            true
        } else {
            false
        }
    }

    pub fn treasury(&self) -> u64 {
        self.treasury
    }

    pub fn add_alliance(&mut self, guild_id: &str) -> Result<(), GuildError> {
        if !self.alliances.insert(guild_id.to_string()) {
            return Err(GuildError::AlreadyAllied(self.id.clone(), guild_id.to_string()));
        }
        Ok(())
    }

    pub fn remove_alliance(&mut self, guild_id: &str) -> Result<(), GuildError> {
        if !self.alliances.remove(guild_id) {
            return Err(GuildError::NotAllied(self.id.clone(), guild_id.to_string()));
        }
        Ok(())
    }

    pub fn alliances(&self) -> &HashSet<String> {
        &self.alliances
    }

    pub fn activity_log(&self) -> &[ActivityEntry] {
        &self.activity_log
    }

    fn log_activity(&mut self, description: &str, timestamp: u64) {
        self.activity_log.push(ActivityEntry { description: description.to_string(), timestamp });
    }

    pub fn members_list(&self) -> Vec<&GuildMember> {
        self.members.values().collect()
    }
}

impl fmt::Display for Guild {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {} ({}/{})", self.tag, self.name, self.member_count(), self.max_members)
    }
}

// ── GuildManager ────────────────────────────────────────────────

#[derive(Debug, Default)]
pub struct GuildManager {
    guilds: HashMap<String, Guild>,
}

impl GuildManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn create_guild(&mut self, id: &str, name: &str, tag: &str, leader: &str, max_members: usize, created_at: u64) -> Result<(), GuildError> {
        if self.guilds.contains_key(id) {
            return Err(GuildError::DuplicateGuild(id.to_string()));
        }
        self.guilds.insert(id.to_string(), Guild::new(id, name, tag, leader, max_members, created_at));
        Ok(())
    }

    pub fn disband_guild(&mut self, id: &str) -> Result<Guild, GuildError> {
        self.guilds.remove(id).ok_or_else(|| GuildError::GuildNotFound(id.to_string()))
    }

    pub fn get_guild(&self, id: &str) -> Option<&Guild> {
        self.guilds.get(id)
    }

    pub fn get_guild_mut(&mut self, id: &str) -> Option<&mut Guild> {
        self.guilds.get_mut(id)
    }

    pub fn search_by_name(&self, query: &str) -> Vec<&Guild> {
        let lower = query.to_lowercase();
        self.guilds.values().filter(|g| g.name.to_lowercase().contains(&lower)).collect()
    }

    pub fn search_by_tag(&self, tag: &str) -> Option<&Guild> {
        self.guilds.values().find(|g| g.tag.eq_ignore_ascii_case(tag))
    }

    pub fn guild_count(&self) -> usize {
        self.guilds.len()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn guild() -> Guild {
        Guild::new("g1", "Warriors", "WAR", "leader", 50, 1000)
    }

    #[test]
    fn test_create_guild() {
        let g = guild();
        assert_eq!(g.member_count(), 1);
        assert!(g.is_member("leader"));
    }

    #[test]
    fn test_join_guild() {
        let mut g = guild();
        g.join("alice", 1001).unwrap();
        assert_eq!(g.member_count(), 2);
    }

    #[test]
    fn test_join_full() {
        let mut g = Guild::new("g1", "Tiny", "T", "leader", 2, 1000);
        g.join("alice", 1001).unwrap();
        assert!(g.join("bob", 1002).is_err());
    }

    #[test]
    fn test_leave_guild() {
        let mut g = guild();
        g.join("alice", 1001).unwrap();
        g.leave("alice", 1002).unwrap();
        assert_eq!(g.member_count(), 1);
    }

    #[test]
    fn test_leader_cannot_leave() {
        let mut g = guild();
        assert!(g.leave("leader", 2000).is_err());
    }

    #[test]
    fn test_kick_member() {
        let mut g = guild();
        g.join("alice", 1001).unwrap();
        g.kick("leader", "alice", 1002).unwrap();
        assert!(!g.is_member("alice"));
    }

    #[test]
    fn test_kick_without_permission() {
        let mut g = guild();
        g.join("alice", 1001).unwrap();
        g.join("bob", 1002).unwrap();
        assert!(g.kick("alice", "bob", 1003).is_err());
    }

    #[test]
    fn test_promote() {
        let mut g = guild();
        g.join("alice", 1001).unwrap();
        g.add_rank(GuildRank::new("Officer", 50).with_permission(GuildPermission::Invite));
        g.promote("leader", "alice", "Officer", 1002).unwrap();
        assert_eq!(g.get_member("alice").unwrap().rank_name, "Officer");
    }

    #[test]
    fn test_demote() {
        let mut g = guild();
        g.join("alice", 1001).unwrap();
        g.add_rank(GuildRank::new("Officer", 50));
        g.promote("leader", "alice", "Officer", 1002).unwrap();
        g.demote("leader", "alice", "Member", 1003).unwrap();
        assert_eq!(g.get_member("alice").unwrap().rank_name, "Member");
    }

    #[test]
    fn test_cannot_demote_leader() {
        let mut g = guild();
        assert!(g.demote("leader", "leader", "Member", 1001).is_err());
    }

    #[test]
    fn test_board_post() {
        let mut g = guild();
        let id = g.post_to_board("leader", "Welcome!", 1001).unwrap();
        assert_eq!(id, 1);
        assert_eq!(g.board_posts().len(), 1);
    }

    #[test]
    fn test_treasury() {
        let mut g = guild();
        g.deposit(500);
        assert_eq!(g.treasury(), 500);
        assert!(g.withdraw(200));
        assert_eq!(g.treasury(), 300);
        assert!(!g.withdraw(400));
    }

    #[test]
    fn test_alliance() {
        let mut g = guild();
        g.add_alliance("g2").unwrap();
        assert!(g.alliances().contains("g2"));
        g.remove_alliance("g2").unwrap();
        assert!(g.alliances().is_empty());
    }

    #[test]
    fn test_duplicate_alliance() {
        let mut g = guild();
        g.add_alliance("g2").unwrap();
        assert!(g.add_alliance("g2").is_err());
    }

    #[test]
    fn test_activity_log() {
        let mut g = guild();
        g.join("alice", 1001).unwrap();
        g.leave("alice", 1002).unwrap();
        assert_eq!(g.activity_log().len(), 2);
    }

    #[test]
    fn test_guild_manager_create_disband() {
        let mut mgr = GuildManager::new();
        mgr.create_guild("g1", "Warriors", "WAR", "leader", 50, 1000).unwrap();
        assert_eq!(mgr.guild_count(), 1);
        mgr.disband_guild("g1").unwrap();
        assert_eq!(mgr.guild_count(), 0);
    }

    #[test]
    fn test_search_by_name() {
        let mut mgr = GuildManager::new();
        mgr.create_guild("g1", "Warriors", "WAR", "a", 50, 1000).unwrap();
        mgr.create_guild("g2", "Mages", "MAG", "b", 50, 1000).unwrap();
        assert_eq!(mgr.search_by_name("war").len(), 1);
    }

    #[test]
    fn test_search_by_tag() {
        let mut mgr = GuildManager::new();
        mgr.create_guild("g1", "Warriors", "WAR", "a", 50, 1000).unwrap();
        assert!(mgr.search_by_tag("war").is_some());
    }

    #[test]
    fn test_display_guild() {
        let g = guild();
        let s = format!("{g}");
        assert!(s.contains("WAR"));
        assert!(s.contains("Warriors"));
    }

    #[test]
    fn test_rank_display() {
        let r = GuildRank::new("Officer", 50);
        let s = format!("{r}");
        assert!(s.contains("Officer"));
        assert!(s.contains("50"));
    }
}
