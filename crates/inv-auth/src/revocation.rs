use std::sync::atomic::{AtomicU64, Ordering};

use dashmap::DashMap;
use tracing::info;

/// An entry in the revocation list, tracking a revoked JWT ID and when it expires.
#[derive(Debug, Clone)]
struct RevocationEntry {
    /// The original token expiration time (unix timestamp).
    /// We keep the entry until this time so we can reject the token
    /// even if someone tries to use it before its natural expiry.
    expires_at: u64,
}

/// Statistics about the revocation list.
#[derive(Debug, Clone)]
pub struct RevocationStats {
    /// Number of currently tracked revoked tokens.
    pub active_entries: usize,
    /// Total number of tokens ever revoked.
    pub total_revoked: u64,
    /// Total number of entries cleaned up after expiry.
    pub total_cleaned: u64,
}

/// A concurrent revocation list for tracking revoked JWT IDs.
///
/// Uses `DashMap` for lock-free concurrent access.
pub struct RevocationService {
    /// Map from JTI (JWT ID) to revocation entry.
    entries: DashMap<String, RevocationEntry>,
    /// Total number of tokens ever revoked (monotonically increasing).
    total_revoked: AtomicU64,
    /// Total number of entries cleaned up.
    total_cleaned: AtomicU64,
}

impl RevocationService {
    /// Create a new, empty revocation service.
    pub fn new() -> Self {
        Self {
            entries: DashMap::new(),
            total_revoked: AtomicU64::new(0),
            total_cleaned: AtomicU64::new(0),
        }
    }

    /// Revoke a token by its JTI (JWT ID).
    ///
    /// The `token_exp` is the original expiration timestamp of the token.
    /// The entry will be kept until cleanup removes it after that time.
    pub fn revoke_token(&self, jti: &str, token_exp: u64) {
        self.entries.insert(
            jti.to_string(),
            RevocationEntry {
                expires_at: token_exp,
            },
        );
        self.total_revoked.fetch_add(1, Ordering::Relaxed);
        info!(jti = jti, "Token revoked");
    }

    /// Check whether a token with the given JTI has been revoked.
    pub fn is_revoked(&self, jti: &str) -> bool {
        self.entries.contains_key(jti)
    }

    /// Remove all entries whose original token has expired.
    ///
    /// Returns the number of entries removed.
    pub fn cleanup_expired(&self) -> usize {
        let now = current_timestamp();
        let before = self.entries.len();

        self.entries.retain(|_jti, entry| entry.expires_at > now);

        let removed = before - self.entries.len();
        if removed > 0 {
            self.total_cleaned
                .fetch_add(removed as u64, Ordering::Relaxed);
            info!(removed = removed, "Cleaned up expired revocation entries");
        }
        removed
    }

    /// Get statistics about the revocation list.
    pub fn stats(&self) -> RevocationStats {
        RevocationStats {
            active_entries: self.entries.len(),
            total_revoked: self.total_revoked.load(Ordering::Relaxed),
            total_cleaned: self.total_cleaned.load(Ordering::Relaxed),
        }
    }
}

impl Default for RevocationService {
    fn default() -> Self {
        Self::new()
    }
}

fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn revoke_and_check() {
        let svc = RevocationService::new();
        let jti = "token-123";
        let exp = current_timestamp() + 3600;

        assert!(!svc.is_revoked(jti));
        svc.revoke_token(jti, exp);
        assert!(svc.is_revoked(jti));
    }

    #[test]
    fn non_revoked_token_passes() {
        let svc = RevocationService::new();
        assert!(!svc.is_revoked("some-random-jti"));
    }

    #[test]
    fn cleanup_removes_expired_entries() {
        let svc = RevocationService::new();
        // Add an already-expired entry
        svc.revoke_token("old-token", 1);
        // Add a still-valid entry
        let future = current_timestamp() + 3600;
        svc.revoke_token("new-token", future);

        assert_eq!(svc.entries.len(), 2);

        let removed = svc.cleanup_expired();
        assert_eq!(removed, 1);
        assert!(!svc.is_revoked("old-token"));
        assert!(svc.is_revoked("new-token"));
    }

    #[test]
    fn stats_are_accurate() {
        let svc = RevocationService::new();
        let future = current_timestamp() + 3600;

        let stats = svc.stats();
        assert_eq!(stats.active_entries, 0);
        assert_eq!(stats.total_revoked, 0);
        assert_eq!(stats.total_cleaned, 0);

        svc.revoke_token("t1", future);
        svc.revoke_token("t2", future);
        svc.revoke_token("t3", 1); // already expired

        let stats = svc.stats();
        assert_eq!(stats.active_entries, 3);
        assert_eq!(stats.total_revoked, 3);

        svc.cleanup_expired();

        let stats = svc.stats();
        assert_eq!(stats.active_entries, 2);
        assert_eq!(stats.total_cleaned, 1);
    }

    #[test]
    fn cleanup_with_no_expired_entries() {
        let svc = RevocationService::new();
        let future = current_timestamp() + 3600;
        svc.revoke_token("t1", future);

        let removed = svc.cleanup_expired();
        assert_eq!(removed, 0);
        assert_eq!(svc.entries.len(), 1);
    }

    #[test]
    fn revoking_same_jti_twice_is_idempotent() {
        let svc = RevocationService::new();
        let future = current_timestamp() + 3600;

        svc.revoke_token("t1", future);
        svc.revoke_token("t1", future);

        assert!(svc.is_revoked("t1"));
        assert_eq!(svc.entries.len(), 1);
        // total_revoked counts each call
        assert_eq!(svc.stats().total_revoked, 2);
    }

    #[test]
    fn concurrent_access_is_safe() {
        use std::sync::Arc;
        use std::thread;

        let svc = Arc::new(RevocationService::new());
        let future = current_timestamp() + 3600;
        let mut handles = vec![];

        for i in 0..10 {
            let svc = Arc::clone(&svc);
            handles.push(thread::spawn(move || {
                let jti = format!("token-{i}");
                svc.revoke_token(&jti, future);
                assert!(svc.is_revoked(&jti));
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(svc.entries.len(), 10);
        assert_eq!(svc.stats().total_revoked, 10);
    }

    #[test]
    fn default_creates_empty_service() {
        let svc = RevocationService::default();
        assert_eq!(svc.stats().active_entries, 0);
        assert_eq!(svc.stats().total_revoked, 0);
    }
}
