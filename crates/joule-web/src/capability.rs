//! Capability-based security — capability tokens, attenuation (restrict
//! capabilities), delegation, revocation, capability composition, authority
//! verification, least privilege enforcement, capability audit trail.

use std::collections::{HashMap, HashSet};
use std::fmt;

// ── Resource Types ─────────────────────────────────────────────────────────

/// The kind of resource a capability grants access to.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ResourceKind {
    File,
    Network,
    Process,
    Memory,
    Device,
    Custom(String),
}

impl fmt::Display for ResourceKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::File => write!(f, "file"),
            Self::Network => write!(f, "network"),
            Self::Process => write!(f, "process"),
            Self::Memory => write!(f, "memory"),
            Self::Device => write!(f, "device"),
            Self::Custom(s) => write!(f, "custom({s})"),
        }
    }
}

// ── Access Rights ──────────────────────────────────────────────────────────

/// Individual access rights that can be composed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccessRight {
    Read,
    Write,
    Execute,
    Create,
    Delete,
    Delegate,
    Attenuate,
}

impl fmt::Display for AccessRight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Read => "read",
            Self::Write => "write",
            Self::Execute => "execute",
            Self::Create => "create",
            Self::Delete => "delete",
            Self::Delegate => "delegate",
            Self::Attenuate => "attenuate",
        };
        write!(f, "{s}")
    }
}

// ── Capability Token ───────────────────────────────────────────────────────

/// A unique capability token ID.
pub type CapId = u64;

/// A capability token granting specific rights to a resource.
#[derive(Debug, Clone)]
pub struct Capability {
    pub id: CapId,
    pub resource_kind: ResourceKind,
    pub resource_path: String,
    pub rights: HashSet<AccessRight>,
    /// Who issued this capability.
    pub issuer: String,
    /// Who currently holds this capability.
    pub holder: String,
    /// The parent capability this was derived from (if any).
    pub parent_id: Option<CapId>,
    /// Whether this capability has been revoked.
    pub revoked: bool,
    /// Expiration timestamp (0 = never expires).
    pub expires_at: u64,
    /// Creation timestamp.
    pub created_at: u64,
    /// Free-form label.
    pub label: String,
}

impl Capability {
    /// Check if a specific right is granted.
    pub fn has_right(&self, right: AccessRight) -> bool {
        self.rights.contains(&right)
    }

    /// Check if this capability is valid (not revoked, not expired).
    pub fn is_valid(&self, now: u64) -> bool {
        !self.revoked && (self.expires_at == 0 || now < self.expires_at)
    }

    /// Check if this capability covers the requested access.
    pub fn authorizes(
        &self,
        resource_kind: &ResourceKind,
        resource_path: &str,
        right: AccessRight,
        now: u64,
    ) -> bool {
        self.is_valid(now)
            && self.resource_kind == *resource_kind
            && path_covers(&self.resource_path, resource_path)
            && self.rights.contains(&right)
    }

    /// Readable summary.
    pub fn summary(&self) -> String {
        let rights_str: Vec<String> = self.rights.iter().map(|r| r.to_string()).collect();
        // Sort for deterministic output.
        let mut sorted = rights_str;
        sorted.sort();
        format!(
            "cap#{} {}:{} [{}] holder={} revoked={}",
            self.id,
            self.resource_kind,
            self.resource_path,
            sorted.join(","),
            self.holder,
            self.revoked,
        )
    }
}

/// Check if `granted_path` covers `requested_path`.
/// A path ending with `/*` covers all children.
fn path_covers(granted: &str, requested: &str) -> bool {
    if granted == requested {
        return true;
    }
    if let Some(prefix) = granted.strip_suffix("/*") {
        return requested.starts_with(prefix);
    }
    false
}

// ── Capability Error ───────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapabilityError {
    NotFound(CapId),
    Revoked(CapId),
    Expired { id: CapId, expired_at: u64 },
    InsufficientRights { id: CapId, missing: AccessRight },
    CannotDelegate(CapId),
    CannotAttenuate(CapId),
    AttenuationWidens { right: AccessRight },
    VerificationFailed(String),
    AlreadyRevoked(CapId),
}

impl fmt::Display for CapabilityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound(id) => write!(f, "capability not found: {id}"),
            Self::Revoked(id) => write!(f, "capability revoked: {id}"),
            Self::Expired { id, expired_at } => {
                write!(f, "capability {id} expired at {expired_at}")
            }
            Self::InsufficientRights { id, missing } => {
                write!(f, "capability {id} missing right: {missing}")
            }
            Self::CannotDelegate(id) => write!(f, "capability {id} cannot be delegated"),
            Self::CannotAttenuate(id) => write!(f, "capability {id} cannot be attenuated"),
            Self::AttenuationWidens { right } => {
                write!(f, "attenuation cannot widen: {right}")
            }
            Self::VerificationFailed(msg) => write!(f, "verification failed: {msg}"),
            Self::AlreadyRevoked(id) => write!(f, "capability {id} already revoked"),
        }
    }
}

// ── Audit Event ────────────────────────────────────────────────────────────

/// An event in the capability audit trail.
#[derive(Debug, Clone)]
pub struct CapAuditEvent {
    pub timestamp: u64,
    pub kind: CapAuditKind,
    pub cap_id: CapId,
    pub actor: String,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapAuditKind {
    Created,
    Delegated,
    Attenuated,
    Revoked,
    Exercised,
    Denied,
    Expired,
    Verified,
}

impl fmt::Display for CapAuditKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Created => "created",
            Self::Delegated => "delegated",
            Self::Attenuated => "attenuated",
            Self::Revoked => "revoked",
            Self::Exercised => "exercised",
            Self::Denied => "denied",
            Self::Expired => "expired",
            Self::Verified => "verified",
        };
        write!(f, "{s}")
    }
}

// ── Capability Authority ───────────────────────────────────────────────────

/// Central authority that manages capability tokens.
pub struct CapabilityAuthority {
    capabilities: HashMap<CapId, Capability>,
    audit_trail: Vec<CapAuditEvent>,
    next_id: CapId,
    current_time: u64,
}

impl CapabilityAuthority {
    pub fn new() -> Self {
        Self {
            capabilities: HashMap::new(),
            audit_trail: Vec::new(),
            next_id: 1,
            current_time: 0,
        }
    }

    /// Advance the simulated clock.
    pub fn set_time(&mut self, t: u64) {
        self.current_time = t;
    }

    pub fn current_time(&self) -> u64 {
        self.current_time
    }

    /// Number of capabilities.
    pub fn capability_count(&self) -> usize {
        self.capabilities.len()
    }

    /// Number of active (non-revoked, non-expired) capabilities.
    pub fn active_count(&self) -> usize {
        let now = self.current_time;
        self.capabilities
            .values()
            .filter(|c| c.is_valid(now))
            .count()
    }

    /// Get a capability by ID.
    pub fn get(&self, id: CapId) -> Option<&Capability> {
        self.capabilities.get(&id)
    }

    /// Audit trail.
    pub fn audit_trail(&self) -> &[CapAuditEvent] {
        &self.audit_trail
    }

    /// Record an audit event.
    fn audit(&mut self, kind: CapAuditKind, cap_id: CapId, actor: &str, detail: &str) {
        self.audit_trail.push(CapAuditEvent {
            timestamp: self.current_time,
            kind,
            cap_id,
            actor: actor.to_string(),
            detail: detail.to_string(),
        });
    }

    // ── Create ─────────────────────────────────────────────────────────

    /// Mint a new root capability.
    pub fn create(
        &mut self,
        resource_kind: ResourceKind,
        resource_path: impl Into<String>,
        rights: HashSet<AccessRight>,
        issuer: impl Into<String>,
        holder: impl Into<String>,
        expires_at: u64,
        label: impl Into<String>,
    ) -> CapId {
        let id = self.next_id;
        self.next_id += 1;
        let issuer_str = issuer.into();
        let holder_str = holder.into();
        let resource_path_str = resource_path.into();
        let label_str = label.into();

        self.capabilities.insert(
            id,
            Capability {
                id,
                resource_kind,
                resource_path: resource_path_str,
                rights,
                issuer: issuer_str.clone(),
                holder: holder_str,
                parent_id: None,
                revoked: false,
                expires_at,
                created_at: self.current_time,
                label: label_str,
            },
        );
        self.audit(CapAuditKind::Created, id, &issuer_str, "root capability");
        id
    }

    // ── Delegate ───────────────────────────────────────────────────────

    /// Delegate a capability to a new holder (same rights).
    pub fn delegate(
        &mut self,
        cap_id: CapId,
        new_holder: impl Into<String>,
    ) -> Result<CapId, CapabilityError> {
        let now = self.current_time;
        let cap = self
            .capabilities
            .get(&cap_id)
            .ok_or(CapabilityError::NotFound(cap_id))?;

        if !cap.is_valid(now) {
            if cap.revoked {
                return Err(CapabilityError::Revoked(cap_id));
            }
            return Err(CapabilityError::Expired {
                id: cap_id,
                expired_at: cap.expires_at,
            });
        }

        if !cap.rights.contains(&AccessRight::Delegate) {
            return Err(CapabilityError::CannotDelegate(cap_id));
        }

        let new_holder_str = new_holder.into();
        let new_id = self.next_id;
        self.next_id += 1;

        // Clone the relevant fields before mutating.
        let resource_kind = cap.resource_kind.clone();
        let resource_path = cap.resource_path.clone();
        let rights = cap.rights.clone();
        let issuer = cap.holder.clone();
        let expires_at = cap.expires_at;
        let label = cap.label.clone();

        self.capabilities.insert(
            new_id,
            Capability {
                id: new_id,
                resource_kind,
                resource_path,
                rights,
                issuer: issuer.clone(),
                holder: new_holder_str,
                parent_id: Some(cap_id),
                revoked: false,
                expires_at,
                created_at: self.current_time,
                label,
            },
        );
        self.audit(
            CapAuditKind::Delegated,
            new_id,
            &issuer,
            &format!("delegated from cap#{cap_id}"),
        );
        Ok(new_id)
    }

    // ── Attenuate ──────────────────────────────────────────────────────

    /// Attenuate a capability: create a child with fewer rights.
    /// The new rights must be a subset of the parent's rights.
    pub fn attenuate(
        &mut self,
        cap_id: CapId,
        new_rights: HashSet<AccessRight>,
        new_holder: impl Into<String>,
    ) -> Result<CapId, CapabilityError> {
        let now = self.current_time;
        let cap = self
            .capabilities
            .get(&cap_id)
            .ok_or(CapabilityError::NotFound(cap_id))?;

        if !cap.is_valid(now) {
            return Err(CapabilityError::Revoked(cap_id));
        }

        if !cap.rights.contains(&AccessRight::Attenuate) {
            return Err(CapabilityError::CannotAttenuate(cap_id));
        }

        // New rights must be a subset.
        for right in &new_rights {
            if !cap.rights.contains(right) {
                return Err(CapabilityError::AttenuationWidens { right: *right });
            }
        }

        let new_holder_str = new_holder.into();
        let new_id = self.next_id;
        self.next_id += 1;

        let resource_kind = cap.resource_kind.clone();
        let resource_path = cap.resource_path.clone();
        let issuer = cap.holder.clone();
        let expires_at = cap.expires_at;
        let label = cap.label.clone();

        self.capabilities.insert(
            new_id,
            Capability {
                id: new_id,
                resource_kind,
                resource_path,
                rights: new_rights,
                issuer: issuer.clone(),
                holder: new_holder_str,
                parent_id: Some(cap_id),
                revoked: false,
                expires_at,
                created_at: self.current_time,
                label,
            },
        );
        self.audit(
            CapAuditKind::Attenuated,
            new_id,
            &issuer,
            &format!("attenuated from cap#{cap_id}"),
        );
        Ok(new_id)
    }

    // ── Revoke ─────────────────────────────────────────────────────────

    /// Revoke a capability and all capabilities derived from it.
    pub fn revoke(&mut self, cap_id: CapId) -> Result<u32, CapabilityError> {
        if !self.capabilities.contains_key(&cap_id) {
            return Err(CapabilityError::NotFound(cap_id));
        }

        let cap = &self.capabilities[&cap_id];
        if cap.revoked {
            return Err(CapabilityError::AlreadyRevoked(cap_id));
        }

        // Find all descendants.
        let mut to_revoke = Vec::new();
        let mut queue = vec![cap_id];
        while let Some(id) = queue.pop() {
            to_revoke.push(id);
            // Find children.
            let children: Vec<CapId> = self
                .capabilities
                .iter()
                .filter(|(_, c)| c.parent_id == Some(id) && !c.revoked)
                .map(|(&cid, _)| cid)
                .collect();
            queue.extend(children);
        }

        let actor = self.capabilities[&cap_id].issuer.clone();
        let count = to_revoke.len() as u32;
        for id in &to_revoke {
            if let Some(cap) = self.capabilities.get_mut(id) {
                cap.revoked = true;
            }
            self.audit(
                CapAuditKind::Revoked,
                *id,
                &actor,
                &format!("revoked (cascade from cap#{cap_id})"),
            );
        }

        Ok(count)
    }

    // ── Verify / Exercise ──────────────────────────────────────────────

    /// Verify that a capability authorizes an access.
    pub fn verify(
        &mut self,
        cap_id: CapId,
        resource_kind: &ResourceKind,
        resource_path: &str,
        right: AccessRight,
    ) -> Result<(), CapabilityError> {
        let now = self.current_time;
        let cap = self
            .capabilities
            .get(&cap_id)
            .ok_or(CapabilityError::NotFound(cap_id))?;

        // Extract all fields we need before releasing the borrow on self.
        let is_revoked = cap.revoked;
        let is_valid = cap.is_valid(now);
        let expired_at = cap.expires_at;
        let authorized = cap.authorizes(resource_kind, resource_path, right, now);
        let has_right = cap.rights.contains(&right);
        let actor = cap.holder.clone();
        // `cap` is no longer used -- borrow released.

        if is_revoked {
            self.audit(CapAuditKind::Denied, cap_id, &actor, "revoked");
            return Err(CapabilityError::Revoked(cap_id));
        }

        if !is_valid {
            self.audit(CapAuditKind::Expired, cap_id, &actor, "expired");
            return Err(CapabilityError::Expired {
                id: cap_id,
                expired_at,
            });
        }

        if !authorized {
            if !has_right {
                self.audit(
                    CapAuditKind::Denied,
                    cap_id,
                    &actor,
                    &format!("missing right: {right}"),
                );
                return Err(CapabilityError::InsufficientRights {
                    id: cap_id,
                    missing: right,
                });
            }
            self.audit(
                CapAuditKind::Denied,
                cap_id,
                &actor,
                "resource mismatch",
            );
            return Err(CapabilityError::VerificationFailed(
                "resource does not match".to_string(),
            ));
        }

        self.audit(CapAuditKind::Verified, cap_id, &actor, &format!("{right}"));
        Ok(())
    }

    /// Exercise a capability (verify + log as exercised).
    pub fn exercise(
        &mut self,
        cap_id: CapId,
        resource_kind: &ResourceKind,
        resource_path: &str,
        right: AccessRight,
    ) -> Result<(), CapabilityError> {
        self.verify(cap_id, resource_kind, resource_path, right)?;
        let actor = self.capabilities[&cap_id].holder.clone();
        self.audit(
            CapAuditKind::Exercised,
            cap_id,
            &actor,
            &format!("{right} on {resource_path}"),
        );
        Ok(())
    }

    // ── Composition ────────────────────────────────────────────────────

    /// Compose two capabilities by intersecting their rights.
    /// Both must cover the same resource kind and path.
    pub fn compose(
        &mut self,
        cap_a: CapId,
        cap_b: CapId,
        holder: impl Into<String>,
    ) -> Result<CapId, CapabilityError> {
        let a = self
            .capabilities
            .get(&cap_a)
            .ok_or(CapabilityError::NotFound(cap_a))?;
        let b = self
            .capabilities
            .get(&cap_b)
            .ok_or(CapabilityError::NotFound(cap_b))?;

        if a.resource_kind != b.resource_kind || a.resource_path != b.resource_path {
            return Err(CapabilityError::VerificationFailed(
                "cannot compose: different resources".to_string(),
            ));
        }

        let intersection: HashSet<AccessRight> =
            a.rights.intersection(&b.rights).copied().collect();

        let resource_kind = a.resource_kind.clone();
        let resource_path = a.resource_path.clone();
        let issuer = a.holder.clone();
        let holder_str = holder.into();
        let expires_at = match (a.expires_at, b.expires_at) {
            (0, 0) => 0,
            (0, x) | (x, 0) => x,
            (x, y) => x.min(y),
        };
        let label = format!("composed(#{},#{})", cap_a, cap_b);

        let new_id = self.next_id;
        self.next_id += 1;

        self.capabilities.insert(
            new_id,
            Capability {
                id: new_id,
                resource_kind,
                resource_path,
                rights: intersection,
                issuer: issuer.clone(),
                holder: holder_str,
                parent_id: None,
                revoked: false,
                expires_at,
                created_at: self.current_time,
                label,
            },
        );
        self.audit(
            CapAuditKind::Created,
            new_id,
            &issuer,
            &format!("composed from cap#{cap_a} + cap#{cap_b}"),
        );
        Ok(new_id)
    }

    // ── Least Privilege ────────────────────────────────────────────────

    /// Find the minimum capability that covers the requested access.
    /// Returns the cap with the fewest rights that still authorizes.
    pub fn find_least_privilege(
        &self,
        holder: &str,
        resource_kind: &ResourceKind,
        resource_path: &str,
        right: AccessRight,
    ) -> Option<CapId> {
        let now = self.current_time;
        self.capabilities
            .values()
            .filter(|c| {
                c.holder == holder && c.authorizes(resource_kind, resource_path, right, now)
            })
            .min_by_key(|c| c.rights.len())
            .map(|c| c.id)
    }

    /// List all capabilities held by a principal.
    pub fn capabilities_for(&self, holder: &str) -> Vec<CapId> {
        let mut ids: Vec<CapId> = self
            .capabilities
            .values()
            .filter(|c| c.holder == holder && c.is_valid(self.current_time))
            .map(|c| c.id)
            .collect();
        ids.sort();
        ids
    }

    /// List all children (direct derivations) of a capability.
    pub fn children(&self, cap_id: CapId) -> Vec<CapId> {
        let mut ids: Vec<CapId> = self
            .capabilities
            .values()
            .filter(|c| c.parent_id == Some(cap_id))
            .map(|c| c.id)
            .collect();
        ids.sort();
        ids
    }
}

impl Default for CapabilityAuthority {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn full_rights() -> HashSet<AccessRight> {
        let mut s = HashSet::new();
        s.insert(AccessRight::Read);
        s.insert(AccessRight::Write);
        s.insert(AccessRight::Execute);
        s.insert(AccessRight::Delegate);
        s.insert(AccessRight::Attenuate);
        s
    }

    fn read_only() -> HashSet<AccessRight> {
        let mut s = HashSet::new();
        s.insert(AccessRight::Read);
        s
    }

    #[test]
    fn create_and_verify() {
        let mut auth = CapabilityAuthority::new();
        let cap = auth.create(
            ResourceKind::File,
            "/data/*",
            full_rights(),
            "root",
            "alice",
            0,
            "full access",
        );
        let result = auth.verify(cap, &ResourceKind::File, "/data/file.txt", AccessRight::Read);
        assert!(result.is_ok());
    }

    #[test]
    fn verify_wrong_resource() {
        let mut auth = CapabilityAuthority::new();
        let cap = auth.create(
            ResourceKind::File,
            "/data/specific.txt",
            full_rights(),
            "root",
            "alice",
            0,
            "",
        );
        let result = auth.verify(cap, &ResourceKind::File, "/other.txt", AccessRight::Read);
        assert!(result.is_err());
    }

    #[test]
    fn verify_wrong_right() {
        let mut auth = CapabilityAuthority::new();
        let cap = auth.create(ResourceKind::File, "/data", read_only(), "root", "alice", 0, "");
        let result = auth.verify(cap, &ResourceKind::File, "/data", AccessRight::Write);
        assert!(matches!(
            result,
            Err(CapabilityError::InsufficientRights { .. })
        ));
    }

    #[test]
    fn delegation() {
        let mut auth = CapabilityAuthority::new();
        let cap = auth.create(
            ResourceKind::File,
            "/data/*",
            full_rights(),
            "root",
            "alice",
            0,
            "",
        );
        let delegated = auth.delegate(cap, "bob").unwrap();
        let result = auth.verify(delegated, &ResourceKind::File, "/data/x", AccessRight::Read);
        assert!(result.is_ok());
        assert_eq!(auth.get(delegated).unwrap().holder, "bob");
        assert_eq!(auth.get(delegated).unwrap().parent_id, Some(cap));
    }

    #[test]
    fn cannot_delegate_without_right() {
        let mut auth = CapabilityAuthority::new();
        let cap = auth.create(ResourceKind::File, "/data", read_only(), "root", "alice", 0, "");
        let result = auth.delegate(cap, "bob");
        assert!(matches!(result, Err(CapabilityError::CannotDelegate(_))));
    }

    #[test]
    fn attenuation() {
        let mut auth = CapabilityAuthority::new();
        let cap = auth.create(
            ResourceKind::File,
            "/data",
            full_rights(),
            "root",
            "alice",
            0,
            "",
        );
        let attenuated = auth.attenuate(cap, read_only(), "bob").unwrap();
        let attn_cap = auth.get(attenuated).unwrap();
        assert_eq!(attn_cap.rights.len(), 1);
        assert!(attn_cap.has_right(AccessRight::Read));
    }

    #[test]
    fn attenuation_cannot_widen() {
        let mut auth = CapabilityAuthority::new();
        let mut rights = HashSet::new();
        rights.insert(AccessRight::Read);
        rights.insert(AccessRight::Attenuate);
        let cap = auth.create(ResourceKind::File, "/data", rights, "root", "alice", 0, "");
        let mut wider = HashSet::new();
        wider.insert(AccessRight::Read);
        wider.insert(AccessRight::Write);
        let result = auth.attenuate(cap, wider, "bob");
        assert!(matches!(result, Err(CapabilityError::AttenuationWidens { .. })));
    }

    #[test]
    fn revocation_cascades() {
        let mut auth = CapabilityAuthority::new();
        let root = auth.create(
            ResourceKind::File,
            "/data",
            full_rights(),
            "root",
            "alice",
            0,
            "",
        );
        let child = auth.delegate(root, "bob").unwrap();
        let grandchild = auth.delegate(child, "charlie").unwrap();

        let count = auth.revoke(root).unwrap();
        assert_eq!(count, 3);
        assert!(auth.get(root).unwrap().revoked);
        assert!(auth.get(child).unwrap().revoked);
        assert!(auth.get(grandchild).unwrap().revoked);
    }

    #[test]
    fn double_revoke() {
        let mut auth = CapabilityAuthority::new();
        let cap = auth.create(ResourceKind::File, "/data", read_only(), "root", "alice", 0, "");
        auth.revoke(cap).unwrap();
        let result = auth.revoke(cap);
        assert!(matches!(result, Err(CapabilityError::AlreadyRevoked(_))));
    }

    #[test]
    fn expiration() {
        let mut auth = CapabilityAuthority::new();
        let cap = auth.create(ResourceKind::File, "/data", read_only(), "root", "alice", 100, "");
        auth.set_time(50);
        assert!(auth.verify(cap, &ResourceKind::File, "/data", AccessRight::Read).is_ok());
        auth.set_time(200);
        assert!(auth.verify(cap, &ResourceKind::File, "/data", AccessRight::Read).is_err());
    }

    #[test]
    fn exercise_logs_audit() {
        let mut auth = CapabilityAuthority::new();
        let cap = auth.create(ResourceKind::File, "/data", read_only(), "root", "alice", 0, "");
        auth.exercise(cap, &ResourceKind::File, "/data", AccessRight::Read)
            .unwrap();
        let exercised = auth
            .audit_trail()
            .iter()
            .filter(|e| e.kind == CapAuditKind::Exercised)
            .count();
        assert_eq!(exercised, 1);
    }

    #[test]
    fn composition() {
        let mut auth = CapabilityAuthority::new();
        let mut rw = HashSet::new();
        rw.insert(AccessRight::Read);
        rw.insert(AccessRight::Write);
        let mut rx = HashSet::new();
        rx.insert(AccessRight::Read);
        rx.insert(AccessRight::Execute);

        let a = auth.create(ResourceKind::File, "/data", rw, "root", "alice", 0, "");
        let b = auth.create(ResourceKind::File, "/data", rx, "root", "alice", 0, "");
        let composed = auth.compose(a, b, "alice").unwrap();
        let cap = auth.get(composed).unwrap();
        // Intersection: {Read}.
        assert_eq!(cap.rights.len(), 1);
        assert!(cap.has_right(AccessRight::Read));
    }

    #[test]
    fn compose_different_resources_fails() {
        let mut auth = CapabilityAuthority::new();
        let a = auth.create(ResourceKind::File, "/a", read_only(), "root", "alice", 0, "");
        let b = auth.create(ResourceKind::File, "/b", read_only(), "root", "alice", 0, "");
        let result = auth.compose(a, b, "alice");
        assert!(result.is_err());
    }

    #[test]
    fn find_least_privilege() {
        let mut auth = CapabilityAuthority::new();
        let _broad = auth.create(ResourceKind::File, "/data/*", full_rights(), "root", "alice", 0, "");
        let narrow = auth.create(ResourceKind::File, "/data/*", read_only(), "root", "alice", 0, "");
        let found = auth.find_least_privilege(
            "alice",
            &ResourceKind::File,
            "/data/x",
            AccessRight::Read,
        );
        assert_eq!(found, Some(narrow));
    }

    #[test]
    fn capabilities_for_holder() {
        let mut auth = CapabilityAuthority::new();
        auth.create(ResourceKind::File, "/a", read_only(), "root", "alice", 0, "");
        auth.create(ResourceKind::File, "/b", read_only(), "root", "alice", 0, "");
        auth.create(ResourceKind::File, "/c", read_only(), "root", "bob", 0, "");
        let alice_caps = auth.capabilities_for("alice");
        assert_eq!(alice_caps.len(), 2);
    }

    #[test]
    fn children_of_cap() {
        let mut auth = CapabilityAuthority::new();
        let root = auth.create(
            ResourceKind::File,
            "/data",
            full_rights(),
            "root",
            "alice",
            0,
            "",
        );
        let c1 = auth.delegate(root, "bob").unwrap();
        let c2 = auth.delegate(root, "charlie").unwrap();
        let children = auth.children(root);
        assert!(children.contains(&c1));
        assert!(children.contains(&c2));
    }

    #[test]
    fn path_wildcard_coverage() {
        assert!(path_covers("/data/*", "/data/file.txt"));
        assert!(path_covers("/data/*", "/data/sub/deep"));
        assert!(!path_covers("/data/specific", "/data/other"));
        assert!(path_covers("/data/exact", "/data/exact"));
    }

    #[test]
    fn active_count_after_revoke() {
        let mut auth = CapabilityAuthority::new();
        auth.create(ResourceKind::File, "/a", read_only(), "root", "alice", 0, "");
        let b = auth.create(ResourceKind::File, "/b", read_only(), "root", "alice", 0, "");
        assert_eq!(auth.active_count(), 2);
        auth.revoke(b).unwrap();
        assert_eq!(auth.active_count(), 1);
    }

    #[test]
    fn capability_summary() {
        let mut auth = CapabilityAuthority::new();
        let cap_id = auth.create(ResourceKind::File, "/tmp", read_only(), "root", "alice", 0, "test");
        let cap = auth.get(cap_id).unwrap();
        let summary = cap.summary();
        assert!(summary.contains("file"));
        assert!(summary.contains("/tmp"));
        assert!(summary.contains("alice"));
    }

    #[test]
    fn error_display() {
        let e = CapabilityError::InsufficientRights {
            id: 1,
            missing: AccessRight::Write,
        };
        assert!(e.to_string().contains("write"));
    }

    #[test]
    fn revoked_cap_cannot_delegate() {
        let mut auth = CapabilityAuthority::new();
        let cap = auth.create(ResourceKind::File, "/x", full_rights(), "root", "alice", 0, "");
        auth.revoke(cap).unwrap();
        let result = auth.delegate(cap, "bob");
        assert!(matches!(result, Err(CapabilityError::Revoked(_))));
    }
}
