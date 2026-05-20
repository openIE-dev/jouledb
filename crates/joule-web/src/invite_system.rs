//! Invitation system — invite generation, expiry, single-use tokens, role
//! assignment on accept, invite tracking, bulk invites, and email-based lookup.
//!
//! Replaces invite-codes, nodemailer-based invite flows, and similar JS/TS
//! invitation libraries with a pure-Rust invite lifecycle engine.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── Errors ─────────────────────────────────────────────────────

/// Invite system errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InviteError {
    /// Invite not found.
    NotFound(String),
    /// Invite has expired.
    Expired(String),
    /// Invite already accepted.
    AlreadyAccepted(String),
    /// Invite already revoked.
    AlreadyRevoked(String),
    /// Email already has a pending invite.
    DuplicateEmail(String),
    /// Invalid email format.
    InvalidEmail(String),
    /// Inviter not found.
    InviterNotFound(String),
    /// Max invites per user exceeded.
    InviteLimitExceeded { user_id: String, limit: usize },
}

impl fmt::Display for InviteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound(id) => write!(f, "invite not found: {id}"),
            Self::Expired(id) => write!(f, "invite expired: {id}"),
            Self::AlreadyAccepted(id) => write!(f, "invite already accepted: {id}"),
            Self::AlreadyRevoked(id) => write!(f, "invite already revoked: {id}"),
            Self::DuplicateEmail(email) => write!(f, "pending invite exists for: {email}"),
            Self::InvalidEmail(email) => write!(f, "invalid email: {email}"),
            Self::InviterNotFound(id) => write!(f, "inviter not found: {id}"),
            Self::InviteLimitExceeded { user_id, limit } => {
                write!(f, "invite limit {limit} exceeded for {user_id}")
            }
        }
    }
}

impl std::error::Error for InviteError {}

// ── Types ──────────────────────────────────────────────────────

/// Status of an invitation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum InviteStatus {
    /// Sent and awaiting acceptance.
    Pending,
    /// Accepted by the invitee.
    Accepted,
    /// Expired without acceptance.
    Expired,
    /// Revoked by the inviter or admin.
    Revoked,
}

impl fmt::Display for InviteStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pending => write!(f, "pending"),
            Self::Accepted => write!(f, "accepted"),
            Self::Expired => write!(f, "expired"),
            Self::Revoked => write!(f, "revoked"),
        }
    }
}

/// An invitation record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Invite {
    pub id: String,
    pub token: String,
    pub email: String,
    pub inviter_id: String,
    pub roles: Vec<String>,
    pub status: InviteStatus,
    pub created_at_secs: u64,
    pub expires_at_secs: u64,
    pub accepted_at_secs: Option<u64>,
    pub accepted_by: Option<String>,
    pub metadata: HashMap<String, String>,
}

impl Invite {
    /// Check if invite is expired at the given time.
    pub fn is_expired(&self, now_secs: u64) -> bool {
        now_secs >= self.expires_at_secs
    }

    /// Check if invite is usable (pending and not expired).
    pub fn is_usable(&self, now_secs: u64) -> bool {
        self.status == InviteStatus::Pending && !self.is_expired(now_secs)
    }

    /// Time remaining until expiry, or 0.
    pub fn remaining_secs(&self, now_secs: u64) -> u64 {
        self.expires_at_secs.saturating_sub(now_secs)
    }
}

/// Result of accepting an invite.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcceptResult {
    pub invite_id: String,
    pub email: String,
    pub roles: Vec<String>,
    pub inviter_id: String,
}

/// Configuration for the invite system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InviteConfig {
    /// Default invite lifetime in seconds.
    pub default_ttl_secs: u64,
    /// Max pending invites per inviter (0 = unlimited).
    pub max_invites_per_user: usize,
    /// Whether to allow duplicate emails for pending invites.
    pub allow_duplicate_emails: bool,
}

impl Default for InviteConfig {
    fn default() -> Self {
        Self {
            default_ttl_secs: 7 * 24 * 3600, // 7 days
            max_invites_per_user: 50,
            allow_duplicate_emails: false,
        }
    }
}

// ── Engine ─────────────────────────────────────────────────────

/// The invite system engine.
#[derive(Debug, Clone)]
pub struct InviteEngine {
    config: InviteConfig,
    /// Invites by ID.
    invites: HashMap<String, Invite>,
    /// Token -> invite ID.
    token_index: HashMap<String, String>,
    /// Email -> invite IDs.
    email_index: HashMap<String, Vec<String>>,
    /// Inviter ID -> invite IDs.
    inviter_index: HashMap<String, Vec<String>>,
    /// Counter for ID generation.
    id_counter: u64,
}

impl InviteEngine {
    pub fn new(config: InviteConfig) -> Self {
        Self {
            config,
            invites: HashMap::new(),
            token_index: HashMap::new(),
            email_index: HashMap::new(),
            inviter_index: HashMap::new(),
            id_counter: 0,
        }
    }

    fn next_id(&mut self) -> String {
        self.id_counter += 1;
        format!("inv_{}", self.id_counter)
    }

    fn generate_token(&mut self) -> String {
        self.id_counter += 1;
        // Deterministic token for testing.
        format!(
            "{:016X}{:016X}",
            self.id_counter.wrapping_mul(6364136223846793005),
            self.id_counter.wrapping_mul(1442695040888963407)
        )
    }

    /// Validate email format (basic check).
    fn validate_email(email: &str) -> Result<(), InviteError> {
        if email.is_empty()
            || !email.contains('@')
            || email.starts_with('@')
            || email.ends_with('@')
        {
            return Err(InviteError::InvalidEmail(email.to_string()));
        }
        let parts: Vec<&str> = email.split('@').collect();
        if parts.len() != 2 || parts[1].is_empty() || !parts[1].contains('.') {
            return Err(InviteError::InvalidEmail(email.to_string()));
        }
        Ok(())
    }

    /// Create a new invite.
    pub fn create_invite(
        &mut self,
        email: &str,
        inviter_id: &str,
        roles: Vec<String>,
        now_secs: u64,
        metadata: HashMap<String, String>,
    ) -> Result<Invite, InviteError> {
        Self::validate_email(email)?;

        // Check duplicate email (pending only).
        if !self.config.allow_duplicate_emails {
            if let Some(ids) = self.email_index.get(email) {
                for id in ids {
                    if let Some(inv) = self.invites.get(id) {
                        if inv.status == InviteStatus::Pending && !inv.is_expired(now_secs) {
                            return Err(InviteError::DuplicateEmail(email.to_string()));
                        }
                    }
                }
            }
        }

        // Check per-user invite limit.
        if self.config.max_invites_per_user > 0 {
            let pending_count = self
                .inviter_index
                .get(inviter_id)
                .map(|ids| {
                    ids.iter()
                        .filter(|id| {
                            self.invites
                                .get(*id)
                                .map(|inv| inv.status == InviteStatus::Pending && !inv.is_expired(now_secs))
                                .unwrap_or(false)
                        })
                        .count()
                })
                .unwrap_or(0);

            if pending_count >= self.config.max_invites_per_user {
                return Err(InviteError::InviteLimitExceeded {
                    user_id: inviter_id.to_string(),
                    limit: self.config.max_invites_per_user,
                });
            }
        }

        let id = self.next_id();
        let token = self.generate_token();

        let invite = Invite {
            id: id.clone(),
            token: token.clone(),
            email: email.to_string(),
            inviter_id: inviter_id.to_string(),
            roles,
            status: InviteStatus::Pending,
            created_at_secs: now_secs,
            expires_at_secs: now_secs + self.config.default_ttl_secs,
            accepted_at_secs: None,
            accepted_by: None,
            metadata,
        };

        self.token_index.insert(token, id.clone());
        self.email_index
            .entry(email.to_string())
            .or_default()
            .push(id.clone());
        self.inviter_index
            .entry(inviter_id.to_string())
            .or_default()
            .push(id.clone());
        self.invites.insert(id, invite.clone());

        Ok(invite)
    }

    /// Accept an invite by token.
    pub fn accept_by_token(
        &mut self,
        token: &str,
        acceptor_id: &str,
        now_secs: u64,
    ) -> Result<AcceptResult, InviteError> {
        let invite_id = self
            .token_index
            .get(token)
            .ok_or_else(|| InviteError::NotFound(token.to_string()))?
            .clone();

        self.accept_invite(&invite_id, acceptor_id, now_secs)
    }

    /// Accept an invite by ID.
    pub fn accept_invite(
        &mut self,
        invite_id: &str,
        acceptor_id: &str,
        now_secs: u64,
    ) -> Result<AcceptResult, InviteError> {
        let invite = self
            .invites
            .get_mut(invite_id)
            .ok_or_else(|| InviteError::NotFound(invite_id.to_string()))?;

        match invite.status {
            InviteStatus::Accepted => {
                return Err(InviteError::AlreadyAccepted(invite_id.to_string()));
            }
            InviteStatus::Revoked => {
                return Err(InviteError::AlreadyRevoked(invite_id.to_string()));
            }
            InviteStatus::Expired => {
                return Err(InviteError::Expired(invite_id.to_string()));
            }
            InviteStatus::Pending => {}
        }

        if invite.is_expired(now_secs) {
            invite.status = InviteStatus::Expired;
            return Err(InviteError::Expired(invite_id.to_string()));
        }

        invite.status = InviteStatus::Accepted;
        invite.accepted_at_secs = Some(now_secs);
        invite.accepted_by = Some(acceptor_id.to_string());

        Ok(AcceptResult {
            invite_id: invite_id.to_string(),
            email: invite.email.clone(),
            roles: invite.roles.clone(),
            inviter_id: invite.inviter_id.clone(),
        })
    }

    /// Revoke an invite.
    pub fn revoke_invite(&mut self, invite_id: &str) -> Result<(), InviteError> {
        let invite = self
            .invites
            .get_mut(invite_id)
            .ok_or_else(|| InviteError::NotFound(invite_id.to_string()))?;

        if invite.status == InviteStatus::Accepted {
            return Err(InviteError::AlreadyAccepted(invite_id.to_string()));
        }
        if invite.status == InviteStatus::Revoked {
            return Err(InviteError::AlreadyRevoked(invite_id.to_string()));
        }

        invite.status = InviteStatus::Revoked;
        Ok(())
    }

    /// Look up invites by email.
    pub fn find_by_email(&self, email: &str) -> Vec<&Invite> {
        self.email_index
            .get(email)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.invites.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Look up invite by token.
    pub fn find_by_token(&self, token: &str) -> Option<&Invite> {
        self.token_index
            .get(token)
            .and_then(|id| self.invites.get(id))
    }

    /// Get invite by ID.
    pub fn get_invite(&self, invite_id: &str) -> Option<&Invite> {
        self.invites.get(invite_id)
    }

    /// List invites created by a specific inviter.
    pub fn invites_by_inviter(&self, inviter_id: &str) -> Vec<&Invite> {
        self.inviter_index
            .get(inviter_id)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.invites.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Create bulk invites. Returns the results for each email.
    pub fn bulk_invite(
        &mut self,
        emails: &[String],
        inviter_id: &str,
        roles: Vec<String>,
        now_secs: u64,
    ) -> Vec<Result<Invite, InviteError>> {
        emails
            .iter()
            .map(|email| {
                self.create_invite(
                    email,
                    inviter_id,
                    roles.clone(),
                    now_secs,
                    HashMap::new(),
                )
            })
            .collect()
    }

    /// Mark expired invites as expired.
    pub fn expire_stale(&mut self, now_secs: u64) -> usize {
        let mut count = 0;
        for invite in self.invites.values_mut() {
            if invite.status == InviteStatus::Pending && invite.is_expired(now_secs) {
                invite.status = InviteStatus::Expired;
                count += 1;
            }
        }
        count
    }

    /// Total invites.
    pub fn total_count(&self) -> usize {
        self.invites.len()
    }

    /// Count invites by status.
    pub fn count_by_status(&self, status: &InviteStatus) -> usize {
        self.invites
            .values()
            .filter(|inv| inv.status == *status)
            .count()
    }
}

impl Default for InviteEngine {
    fn default() -> Self {
        Self::new(InviteConfig::default())
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> InviteConfig {
        InviteConfig {
            default_ttl_secs: 3600,
            max_invites_per_user: 5,
            allow_duplicate_emails: false,
        }
    }

    fn create_engine() -> InviteEngine {
        InviteEngine::new(test_config())
    }

    #[test]
    fn test_create_invite() {
        let mut engine = create_engine();
        let invite = engine
            .create_invite(
                "alice@example.com",
                "admin-1",
                vec!["editor".to_string()],
                1000,
                HashMap::new(),
            )
            .unwrap();

        assert_eq!(invite.email, "alice@example.com");
        assert_eq!(invite.inviter_id, "admin-1");
        assert_eq!(invite.roles, vec!["editor"]);
        assert_eq!(invite.status, InviteStatus::Pending);
        assert_eq!(invite.expires_at_secs, 4600);
    }

    #[test]
    fn test_invalid_email() {
        let mut engine = create_engine();
        let cases = vec!["", "noat", "@bad", "bad@", "bad@nodot", "a@@b.com"];
        for email in cases {
            let err = engine
                .create_invite(email, "admin", vec![], 1000, HashMap::new())
                .unwrap_err();
            assert!(matches!(err, InviteError::InvalidEmail(_)), "email={email}");
        }
    }

    #[test]
    fn test_duplicate_email() {
        let mut engine = create_engine();
        engine
            .create_invite("alice@example.com", "admin", vec![], 1000, HashMap::new())
            .unwrap();

        let err = engine
            .create_invite("alice@example.com", "admin", vec![], 1001, HashMap::new())
            .unwrap_err();
        assert!(matches!(err, InviteError::DuplicateEmail(_)));
    }

    #[test]
    fn test_duplicate_email_allowed_after_expiry() {
        let mut engine = create_engine();
        engine
            .create_invite("alice@example.com", "admin", vec![], 1000, HashMap::new())
            .unwrap();

        // After expiry, duplicate should be allowed.
        let result = engine.create_invite(
            "alice@example.com",
            "admin",
            vec![],
            1000 + 3601,
            HashMap::new(),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_accept_by_token() {
        let mut engine = create_engine();
        let invite = engine
            .create_invite(
                "alice@example.com",
                "admin",
                vec!["editor".to_string()],
                1000,
                HashMap::new(),
            )
            .unwrap();

        let result = engine
            .accept_by_token(&invite.token, "alice-user-id", 1500)
            .unwrap();
        assert_eq!(result.email, "alice@example.com");
        assert_eq!(result.roles, vec!["editor"]);

        let inv = engine.get_invite(&invite.id).unwrap();
        assert_eq!(inv.status, InviteStatus::Accepted);
        assert_eq!(inv.accepted_by.as_deref(), Some("alice-user-id"));
    }

    #[test]
    fn test_accept_expired() {
        let mut engine = create_engine();
        let invite = engine
            .create_invite("alice@example.com", "admin", vec![], 1000, HashMap::new())
            .unwrap();

        let err = engine
            .accept_invite(&invite.id, "alice", 100000)
            .unwrap_err();
        assert!(matches!(err, InviteError::Expired(_)));
    }

    #[test]
    fn test_accept_already_accepted() {
        let mut engine = create_engine();
        let invite = engine
            .create_invite("alice@example.com", "admin", vec![], 1000, HashMap::new())
            .unwrap();

        engine.accept_invite(&invite.id, "alice", 1500).unwrap();
        let err = engine
            .accept_invite(&invite.id, "bob", 1600)
            .unwrap_err();
        assert!(matches!(err, InviteError::AlreadyAccepted(_)));
    }

    #[test]
    fn test_revoke_invite() {
        let mut engine = create_engine();
        let invite = engine
            .create_invite("alice@example.com", "admin", vec![], 1000, HashMap::new())
            .unwrap();

        engine.revoke_invite(&invite.id).unwrap();
        let inv = engine.get_invite(&invite.id).unwrap();
        assert_eq!(inv.status, InviteStatus::Revoked);
    }

    #[test]
    fn test_revoke_already_accepted() {
        let mut engine = create_engine();
        let invite = engine
            .create_invite("alice@example.com", "admin", vec![], 1000, HashMap::new())
            .unwrap();

        engine.accept_invite(&invite.id, "alice", 1500).unwrap();
        let err = engine.revoke_invite(&invite.id).unwrap_err();
        assert!(matches!(err, InviteError::AlreadyAccepted(_)));
    }

    #[test]
    fn test_revoke_already_revoked() {
        let mut engine = create_engine();
        let invite = engine
            .create_invite("alice@example.com", "admin", vec![], 1000, HashMap::new())
            .unwrap();

        engine.revoke_invite(&invite.id).unwrap();
        let err = engine.revoke_invite(&invite.id).unwrap_err();
        assert!(matches!(err, InviteError::AlreadyRevoked(_)));
    }

    #[test]
    fn test_accept_revoked() {
        let mut engine = create_engine();
        let invite = engine
            .create_invite("alice@example.com", "admin", vec![], 1000, HashMap::new())
            .unwrap();

        engine.revoke_invite(&invite.id).unwrap();
        let err = engine
            .accept_invite(&invite.id, "alice", 1500)
            .unwrap_err();
        assert!(matches!(err, InviteError::AlreadyRevoked(_)));
    }

    #[test]
    fn test_find_by_email() {
        let mut engine = create_engine();
        engine.config.allow_duplicate_emails = true;

        engine
            .create_invite("alice@example.com", "admin", vec![], 1000, HashMap::new())
            .unwrap();
        engine
            .create_invite("alice@example.com", "admin", vec![], 1001, HashMap::new())
            .unwrap();

        let results = engine.find_by_email("alice@example.com");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_find_by_token() {
        let mut engine = create_engine();
        let invite = engine
            .create_invite("alice@example.com", "admin", vec![], 1000, HashMap::new())
            .unwrap();

        let found = engine.find_by_token(&invite.token).unwrap();
        assert_eq!(found.id, invite.id);
    }

    #[test]
    fn test_invites_by_inviter() {
        let mut engine = create_engine();
        engine
            .create_invite("a@b.com", "admin-1", vec![], 1000, HashMap::new())
            .unwrap();
        engine
            .create_invite("c@d.com", "admin-1", vec![], 1001, HashMap::new())
            .unwrap();
        engine
            .create_invite("e@f.com", "admin-2", vec![], 1002, HashMap::new())
            .unwrap();

        let results = engine.invites_by_inviter("admin-1");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_bulk_invite() {
        let mut engine = create_engine();
        let emails = vec![
            "a@b.com".to_string(),
            "c@d.com".to_string(),
            "bad_email".to_string(),
        ];

        let results = engine.bulk_invite(&emails, "admin", vec!["viewer".to_string()], 1000);
        assert!(results[0].is_ok());
        assert!(results[1].is_ok());
        assert!(results[2].is_err()); // invalid email
    }

    #[test]
    fn test_invite_limit() {
        let mut engine = InviteEngine::new(InviteConfig {
            max_invites_per_user: 2,
            ..test_config()
        });

        engine
            .create_invite("a@b.com", "admin", vec![], 1000, HashMap::new())
            .unwrap();
        engine
            .create_invite("c@d.com", "admin", vec![], 1001, HashMap::new())
            .unwrap();

        let err = engine
            .create_invite("e@f.com", "admin", vec![], 1002, HashMap::new())
            .unwrap_err();
        assert!(matches!(err, InviteError::InviteLimitExceeded { .. }));
    }

    #[test]
    fn test_expire_stale() {
        let mut engine = create_engine();
        engine
            .create_invite("a@b.com", "admin", vec![], 1000, HashMap::new())
            .unwrap();
        engine
            .create_invite("c@d.com", "admin", vec![], 1000, HashMap::new())
            .unwrap();

        let expired = engine.expire_stale(100000);
        assert_eq!(expired, 2);
        assert_eq!(engine.count_by_status(&InviteStatus::Expired), 2);
    }

    #[test]
    fn test_invite_remaining_secs() {
        let mut engine = create_engine();
        let invite = engine
            .create_invite("a@b.com", "admin", vec![], 1000, HashMap::new())
            .unwrap();

        assert_eq!(invite.remaining_secs(2000), 2600);
        assert_eq!(invite.remaining_secs(4600), 0);
        assert_eq!(invite.remaining_secs(5000), 0);
    }

    #[test]
    fn test_invite_is_usable() {
        let mut engine = create_engine();
        let invite = engine
            .create_invite("a@b.com", "admin", vec![], 1000, HashMap::new())
            .unwrap();

        assert!(invite.is_usable(2000));
        assert!(!invite.is_usable(100000));
    }

    #[test]
    fn test_total_count() {
        let mut engine = create_engine();
        assert_eq!(engine.total_count(), 0);
        engine
            .create_invite("a@b.com", "admin", vec![], 1000, HashMap::new())
            .unwrap();
        assert_eq!(engine.total_count(), 1);
    }

    #[test]
    fn test_count_by_status() {
        let mut engine = create_engine();
        engine
            .create_invite("a@b.com", "admin", vec![], 1000, HashMap::new())
            .unwrap();
        let inv = engine
            .create_invite("c@d.com", "admin", vec![], 1000, HashMap::new())
            .unwrap();
        engine.accept_invite(&inv.id, "user-1", 1500).unwrap();

        assert_eq!(engine.count_by_status(&InviteStatus::Pending), 1);
        assert_eq!(engine.count_by_status(&InviteStatus::Accepted), 1);
    }

    #[test]
    fn test_metadata() {
        let mut engine = create_engine();
        let mut meta = HashMap::new();
        meta.insert("team".to_string(), "engineering".to_string());

        let invite = engine
            .create_invite("a@b.com", "admin", vec![], 1000, meta)
            .unwrap();
        assert_eq!(invite.metadata.get("team").unwrap(), "engineering");
    }

    #[test]
    fn test_error_display() {
        let e = InviteError::Expired("inv_1".to_string());
        assert_eq!(e.to_string(), "invite expired: inv_1");
    }

    #[test]
    fn test_status_display() {
        assert_eq!(InviteStatus::Pending.to_string(), "pending");
        assert_eq!(InviteStatus::Accepted.to_string(), "accepted");
    }

    #[test]
    fn test_default_engine() {
        let engine = InviteEngine::default();
        assert_eq!(engine.total_count(), 0);
    }

    #[test]
    fn test_not_found_errors() {
        let mut engine = create_engine();
        assert!(engine.accept_invite("ghost", "user", 1000).is_err());
        assert!(engine.revoke_invite("ghost").is_err());
        assert!(engine.find_by_token("ghost").is_none());
        assert!(engine.get_invite("ghost").is_none());
    }
}
