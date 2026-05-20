//! Ban/penalty system — progressive punishment, appeals, hardware bans.
//!
//! Replaces ban-management microservices with pure Rust.
//! BanType classification, Ban records with issuance/expiry tracking,
//! BanManager for active ban management, issue/revoke/check operations,
//! progressive punishment with escalating severity, ban appeal tracking,
//! per-player ban history, active ban counts, ban search by player/type/date,
//! ban expiry checking, and hardware_id ban support.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BanError {
    BanNotFound(u64),
    PlayerNotFound(u64),
    AlreadyBanned { player_id: u64, ban_type: BanType },
    AppealNotFound(u64),
    InvalidDuration(String),
}

impl fmt::Display for BanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BanNotFound(id) => write!(f, "ban not found: {id}"),
            Self::PlayerNotFound(id) => write!(f, "player not found: {id}"),
            Self::AlreadyBanned { player_id, ban_type } => {
                write!(f, "player {player_id} already has {ban_type} ban")
            }
            Self::AppealNotFound(id) => write!(f, "appeal not found: {id}"),
            Self::InvalidDuration(msg) => write!(f, "invalid duration: {msg}"),
        }
    }
}

impl std::error::Error for BanError {}

// ── Ban Type ────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BanType {
    Warning,
    TempBan,
    PermBan,
    ShadowBan,
}

impl BanType {
    pub fn severity_level(&self) -> u8 {
        match self {
            Self::Warning => 1,
            Self::TempBan => 2,
            Self::ShadowBan => 3,
            Self::PermBan => 4,
        }
    }
}

impl fmt::Display for BanType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Warning => write!(f, "Warning"),
            Self::TempBan => write!(f, "TempBan"),
            Self::PermBan => write!(f, "PermBan"),
            Self::ShadowBan => write!(f, "ShadowBan"),
        }
    }
}

// ── Ban ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct Ban {
    pub id: u64,
    pub player_id: u64,
    pub ban_type: BanType,
    pub reason: String,
    pub issued_at: u64,
    pub expires_at: Option<u64>,
    pub revoked: bool,
    pub revoked_at: Option<u64>,
    pub hardware_id: Option<String>,
    pub issuer: String,
}

impl Ban {
    pub fn new(id: u64, player_id: u64, ban_type: BanType, reason: &str, issued_at: u64) -> Self {
        Self {
            id,
            player_id,
            ban_type,
            reason: reason.to_string(),
            issued_at,
            expires_at: None,
            revoked: false,
            revoked_at: None,
            hardware_id: None,
            issuer: String::new(),
        }
    }

    pub fn with_expiry(mut self, expires_at: u64) -> Self {
        self.expires_at = Some(expires_at);
        self
    }

    pub fn with_hardware_id(mut self, hw_id: &str) -> Self {
        self.hardware_id = Some(hw_id.to_string());
        self
    }

    pub fn with_issuer(mut self, issuer: &str) -> Self {
        self.issuer = issuer.to_string();
        self
    }

    pub fn is_active(&self, current_time: u64) -> bool {
        if self.revoked {
            return false;
        }
        if let Some(exp) = self.expires_at {
            current_time < exp
        } else {
            true
        }
    }

    pub fn is_expired(&self, current_time: u64) -> bool {
        if let Some(exp) = self.expires_at {
            current_time >= exp
        } else {
            false
        }
    }

    pub fn remaining_seconds(&self, current_time: u64) -> Option<u64> {
        self.expires_at.map(|exp| exp.saturating_sub(current_time))
    }

    pub fn revoke(&mut self, at_time: u64) {
        self.revoked = true;
        self.revoked_at = Some(at_time);
    }

    pub fn duration(&self) -> Option<u64> {
        self.expires_at.map(|exp| exp.saturating_sub(self.issued_at))
    }
}

impl fmt::Display for Ban {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Ban#{} player={} type={} reason={}",
            self.id, self.player_id, self.ban_type, self.reason
        )
    }
}

// ── Appeal ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AppealStatus {
    Pending,
    Accepted,
    Rejected,
}

impl fmt::Display for AppealStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pending => write!(f, "Pending"),
            Self::Accepted => write!(f, "Accepted"),
            Self::Rejected => write!(f, "Rejected"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Appeal {
    pub id: u64,
    pub ban_id: u64,
    pub player_id: u64,
    pub reason: String,
    pub status: AppealStatus,
    pub submitted_at: u64,
    pub resolved_at: Option<u64>,
}

impl Appeal {
    pub fn new(id: u64, ban_id: u64, player_id: u64, reason: &str, submitted_at: u64) -> Self {
        Self {
            id,
            ban_id,
            player_id,
            reason: reason.to_string(),
            status: AppealStatus::Pending,
            submitted_at,
            resolved_at: None,
        }
    }

    pub fn accept(&mut self, at_time: u64) {
        self.status = AppealStatus::Accepted;
        self.resolved_at = Some(at_time);
    }

    pub fn reject(&mut self, at_time: u64) {
        self.status = AppealStatus::Rejected;
        self.resolved_at = Some(at_time);
    }

    pub fn is_pending(&self) -> bool {
        self.status == AppealStatus::Pending
    }
}

impl fmt::Display for Appeal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Appeal#{} ban={} player={} status={}",
            self.id, self.ban_id, self.player_id, self.status
        )
    }
}

// ── Progressive Config ──────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ProgressiveConfig {
    pub escalation: Vec<(BanType, Option<u64>)>,
}

impl Default for ProgressiveConfig {
    fn default() -> Self {
        Self {
            escalation: vec![
                (BanType::Warning, None),
                (BanType::TempBan, Some(3600)),
                (BanType::TempBan, Some(86400)),
                (BanType::TempBan, Some(604800)),
                (BanType::PermBan, None),
            ],
        }
    }
}

impl ProgressiveConfig {
    pub fn step_for(&self, offense_count: usize) -> (BanType, Option<u64>) {
        let idx = offense_count.min(self.escalation.len().saturating_sub(1));
        self.escalation[idx]
    }
}

// ── Ban Manager ─────────────────────────────────────────────────

#[derive(Debug)]
pub struct BanManager {
    bans: Vec<Ban>,
    appeals: Vec<Appeal>,
    progressive: ProgressiveConfig,
    next_ban_id: u64,
    next_appeal_id: u64,
    current_time: u64,
    hardware_bans: HashMap<String, Vec<u64>>,
}

impl BanManager {
    pub fn new() -> Self {
        Self {
            bans: Vec::new(),
            appeals: Vec::new(),
            progressive: ProgressiveConfig::default(),
            next_ban_id: 1,
            next_appeal_id: 1,
            current_time: 0,
            hardware_bans: HashMap::new(),
        }
    }

    pub fn with_progressive(mut self, cfg: ProgressiveConfig) -> Self {
        self.progressive = cfg;
        self
    }

    pub fn set_time(&mut self, t: u64) {
        self.current_time = t;
    }

    pub fn issue_ban(&mut self, player_id: u64, ban_type: BanType, reason: &str, duration: Option<u64>) -> Ban {
        let id = self.next_ban_id;
        self.next_ban_id += 1;
        let expires_at = duration.map(|d| self.current_time + d);
        let ban = Ban::new(id, player_id, ban_type, reason, self.current_time)
            .with_expiry(expires_at.unwrap_or(u64::MAX));
        if ban_type == BanType::PermBan {
            let mut b = Ban::new(id, player_id, ban_type, reason, self.current_time);
            b.expires_at = None;
            self.bans.push(b.clone());
            return b;
        }
        self.bans.push(ban.clone());
        ban
    }

    pub fn issue_ban_with_hardware(&mut self, player_id: u64, ban_type: BanType, reason: &str, duration: Option<u64>, hw_id: &str) -> Ban {
        let mut ban = self.issue_ban(player_id, ban_type, reason, duration);
        ban.hardware_id = Some(hw_id.to_string());
        // Update the ban in storage
        if let Some(stored) = self.bans.iter_mut().find(|b| b.id == ban.id) {
            stored.hardware_id = Some(hw_id.to_string());
        }
        self.hardware_bans
            .entry(hw_id.to_string())
            .or_default()
            .push(ban.id);
        ban
    }

    pub fn issue_progressive_ban(&mut self, player_id: u64, reason: &str) -> Ban {
        let offense_count = self
            .bans
            .iter()
            .filter(|b| b.player_id == player_id)
            .count();
        let (ban_type, duration) = self.progressive.step_for(offense_count);
        self.issue_ban(player_id, ban_type, reason, duration)
    }

    pub fn revoke_ban(&mut self, ban_id: u64) -> Result<(), BanError> {
        let ban = self
            .bans
            .iter_mut()
            .find(|b| b.id == ban_id)
            .ok_or(BanError::BanNotFound(ban_id))?;
        ban.revoke(self.current_time);
        Ok(())
    }

    pub fn is_banned(&self, player_id: u64) -> bool {
        self.bans
            .iter()
            .any(|b| b.player_id == player_id && b.is_active(self.current_time))
    }

    pub fn is_hardware_banned(&self, hw_id: &str) -> bool {
        if let Some(ban_ids) = self.hardware_bans.get(hw_id) {
            ban_ids.iter().any(|bid| {
                self.bans
                    .iter()
                    .any(|b| b.id == *bid && b.is_active(self.current_time))
            })
        } else {
            false
        }
    }

    pub fn active_bans(&self) -> Vec<&Ban> {
        self.bans
            .iter()
            .filter(|b| b.is_active(self.current_time))
            .collect()
    }

    pub fn active_ban_count(&self) -> usize {
        self.active_bans().len()
    }

    pub fn player_bans(&self, player_id: u64) -> Vec<&Ban> {
        self.bans.iter().filter(|b| b.player_id == player_id).collect()
    }

    pub fn player_active_bans(&self, player_id: u64) -> Vec<&Ban> {
        self.bans
            .iter()
            .filter(|b| b.player_id == player_id && b.is_active(self.current_time))
            .collect()
    }

    pub fn bans_by_type(&self, ban_type: BanType) -> Vec<&Ban> {
        self.bans.iter().filter(|b| b.ban_type == ban_type).collect()
    }

    pub fn bans_in_range(&self, from: u64, to: u64) -> Vec<&Ban> {
        self.bans
            .iter()
            .filter(|b| b.issued_at >= from && b.issued_at <= to)
            .collect()
    }

    pub fn expired_bans(&self) -> Vec<&Ban> {
        self.bans
            .iter()
            .filter(|b| b.is_expired(self.current_time) && !b.revoked)
            .collect()
    }

    pub fn submit_appeal(&mut self, ban_id: u64, player_id: u64, reason: &str) -> Result<Appeal, BanError> {
        if !self.bans.iter().any(|b| b.id == ban_id) {
            return Err(BanError::BanNotFound(ban_id));
        }
        let id = self.next_appeal_id;
        self.next_appeal_id += 1;
        let appeal = Appeal::new(id, ban_id, player_id, reason, self.current_time);
        self.appeals.push(appeal.clone());
        Ok(appeal)
    }

    pub fn resolve_appeal(&mut self, appeal_id: u64, accept: bool) -> Result<(), BanError> {
        let appeal = self
            .appeals
            .iter_mut()
            .find(|a| a.id == appeal_id)
            .ok_or(BanError::AppealNotFound(appeal_id))?;
        if accept {
            appeal.accept(self.current_time);
            let ban_id = appeal.ban_id;
            if let Some(ban) = self.bans.iter_mut().find(|b| b.id == ban_id) {
                ban.revoke(self.current_time);
            }
        } else {
            appeal.reject(self.current_time);
        }
        Ok(())
    }

    pub fn pending_appeals(&self) -> Vec<&Appeal> {
        self.appeals.iter().filter(|a| a.is_pending()).collect()
    }

    pub fn player_appeals(&self, player_id: u64) -> Vec<&Appeal> {
        self.appeals
            .iter()
            .filter(|a| a.player_id == player_id)
            .collect()
    }

    pub fn total_bans(&self) -> usize {
        self.bans.len()
    }

    pub fn total_appeals(&self) -> usize {
        self.appeals.len()
    }
}

impl Default for BanManager {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn manager() -> BanManager {
        let mut m = BanManager::new();
        m.set_time(1000);
        m
    }

    #[test]
    fn test_issue_temp_ban() {
        let mut m = manager();
        let ban = m.issue_ban(1, BanType::TempBan, "cheating", Some(3600));
        assert_eq!(ban.player_id, 1);
        assert_eq!(ban.ban_type, BanType::TempBan);
        assert!(ban.is_active(1000));
    }

    #[test]
    fn test_ban_expiry() {
        let mut m = manager();
        m.issue_ban(1, BanType::TempBan, "test", Some(100));
        assert!(m.is_banned(1));
        m.set_time(1200);
        assert!(!m.is_banned(1));
    }

    #[test]
    fn test_perm_ban_no_expiry() {
        let mut m = manager();
        let ban = m.issue_ban(1, BanType::PermBan, "severe", None);
        assert!(ban.is_active(1000));
        assert!(ban.is_active(u64::MAX - 1));
    }

    #[test]
    fn test_revoke_ban() {
        let mut m = manager();
        let ban = m.issue_ban(1, BanType::TempBan, "test", Some(3600));
        m.revoke_ban(ban.id).unwrap();
        assert!(!m.is_banned(1));
    }

    #[test]
    fn test_revoke_nonexistent() {
        let mut m = manager();
        assert!(matches!(m.revoke_ban(999), Err(BanError::BanNotFound(999))));
    }

    #[test]
    fn test_progressive_punishment() {
        let mut m = manager();
        let b1 = m.issue_progressive_ban(1, "offense 1");
        assert_eq!(b1.ban_type, BanType::Warning);
        let b2 = m.issue_progressive_ban(1, "offense 2");
        assert_eq!(b2.ban_type, BanType::TempBan);
    }

    #[test]
    fn test_progressive_escalates() {
        let mut m = manager();
        for i in 0..5 {
            m.issue_progressive_ban(1, &format!("offense {}", i + 1));
        }
        let bans = m.player_bans(1);
        assert_eq!(bans.last().unwrap().ban_type, BanType::PermBan);
    }

    #[test]
    fn test_player_bans() {
        let mut m = manager();
        m.issue_ban(1, BanType::Warning, "w1", None);
        m.issue_ban(1, BanType::TempBan, "t1", Some(100));
        m.issue_ban(2, BanType::Warning, "w2", None);
        assert_eq!(m.player_bans(1).len(), 2);
    }

    #[test]
    fn test_bans_by_type() {
        let mut m = manager();
        m.issue_ban(1, BanType::Warning, "w1", None);
        m.issue_ban(2, BanType::Warning, "w2", None);
        m.issue_ban(3, BanType::TempBan, "t1", Some(100));
        assert_eq!(m.bans_by_type(BanType::Warning).len(), 2);
    }

    #[test]
    fn test_bans_in_range() {
        let mut m = manager();
        m.set_time(100);
        m.issue_ban(1, BanType::Warning, "early", None);
        m.set_time(500);
        m.issue_ban(2, BanType::Warning, "middle", None);
        m.set_time(900);
        m.issue_ban(3, BanType::Warning, "late", None);
        let range = m.bans_in_range(200, 600);
        assert_eq!(range.len(), 1);
    }

    #[test]
    fn test_submit_appeal() {
        let mut m = manager();
        let ban = m.issue_ban(1, BanType::TempBan, "test", Some(3600));
        let appeal = m.submit_appeal(ban.id, 1, "I'm innocent").unwrap();
        assert!(appeal.is_pending());
    }

    #[test]
    fn test_accept_appeal_revokes_ban() {
        let mut m = manager();
        let ban = m.issue_ban(1, BanType::TempBan, "test", Some(3600));
        let appeal = m.submit_appeal(ban.id, 1, "please").unwrap();
        m.resolve_appeal(appeal.id, true).unwrap();
        assert!(!m.is_banned(1));
    }

    #[test]
    fn test_reject_appeal() {
        let mut m = manager();
        let ban = m.issue_ban(1, BanType::TempBan, "test", Some(3600));
        let appeal = m.submit_appeal(ban.id, 1, "please").unwrap();
        m.resolve_appeal(appeal.id, false).unwrap();
        assert!(m.is_banned(1));
        assert!(m.pending_appeals().is_empty());
    }

    #[test]
    fn test_hardware_ban() {
        let mut m = manager();
        m.issue_ban_with_hardware(1, BanType::PermBan, "hwid ban", None, "HWID-ABC123");
        assert!(m.is_hardware_banned("HWID-ABC123"));
        assert!(!m.is_hardware_banned("HWID-OTHER"));
    }

    #[test]
    fn test_active_ban_count() {
        let mut m = manager();
        m.issue_ban(1, BanType::TempBan, "t1", Some(3600));
        m.issue_ban(2, BanType::TempBan, "t2", Some(3600));
        assert_eq!(m.active_ban_count(), 2);
    }

    #[test]
    fn test_ban_display() {
        let b = Ban::new(1, 42, BanType::TempBan, "speed hack", 100);
        let s = format!("{b}");
        assert!(s.contains("player=42"));
        assert!(s.contains("TempBan"));
    }

    #[test]
    fn test_ban_type_severity() {
        assert!(BanType::Warning.severity_level() < BanType::TempBan.severity_level());
        assert!(BanType::TempBan.severity_level() < BanType::PermBan.severity_level());
    }

    #[test]
    fn test_ban_remaining_seconds() {
        let ban = Ban::new(1, 1, BanType::TempBan, "test", 100).with_expiry(200);
        assert_eq!(ban.remaining_seconds(150), Some(50));
        assert_eq!(ban.remaining_seconds(250), Some(0));
    }

    #[test]
    fn test_appeal_display() {
        let a = Appeal::new(1, 5, 42, "not guilty", 500);
        let s = format!("{a}");
        assert!(s.contains("Appeal#1"));
        assert!(s.contains("Pending"));
    }
}
