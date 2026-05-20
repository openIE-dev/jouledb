//! SMS/OTP multi-factor authentication for admin accounts.
//!
//! Provides MFA challenge creation, verification with lockout, and utility
//! helpers (code generation, phone masking, constant-time comparison).

use std::collections::HashMap;
use std::sync::RwLock;

use serde::{Deserialize, Serialize};

use crate::rbac::Role;

/// A pending SMS MFA challenge for admin accounts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MfaPending {
    /// The 6-digit SMS code.
    pub code: String,
    /// Email of the admin being challenged.
    pub email: String,
    /// Organization.
    pub org: String,
    /// Role (always Admin).
    pub role: Role,
    /// Region restrictions.
    pub regions: Vec<String>,
    /// Expiration (unix seconds). MFA codes are short-lived (5 minutes).
    pub expires_at: u64,
    /// Remaining verification attempts (locks out after 3 failures).
    pub attempts_remaining: u32,
}

/// Result of an MFA verification attempt.
#[derive(Debug)]
pub enum MfaVerifyResult {
    /// Code matched. Contains (email, org, role, regions).
    Success {
        email: String,
        org: String,
        role: Role,
        regions: Vec<String>,
    },
    /// Code expired.
    Expired,
    /// Wrong code, attempts remaining.
    WrongCode { attempts_remaining: u32 },
    /// Locked out (no attempts remaining).
    LockedOut,
    /// Session not found.
    NotFound,
}

/// Service for managing MFA challenges.
pub struct MfaService {
    pending: RwLock<HashMap<String, MfaPending>>,
    /// Emails allowed to authenticate as admin.
    admin_emails: Vec<String>,
    /// Phone number for admin SMS MFA.
    admin_phone: String,
}

impl MfaService {
    /// Create a new MFA service.
    pub fn new(admin_emails: Vec<String>, admin_phone: String) -> Self {
        Self {
            pending: RwLock::new(HashMap::new()),
            admin_emails,
            admin_phone,
        }
    }

    /// Check whether the given email is in the admin allowlist.
    pub fn is_admin_email(&self, email: &str) -> bool {
        self.admin_emails
            .iter()
            .any(|e| e.eq_ignore_ascii_case(email))
    }

    /// Create an MFA challenge for an admin login.
    ///
    /// Returns `(session_id, mfa_code)`. In production, send the code via SMS.
    /// In dev mode, the caller can return it in the response.
    pub fn create_challenge(
        &self,
        email: &str,
        org: &str,
        role: Role,
        regions: Vec<String>,
    ) -> (String, String) {
        let code = generate_mfa_code();
        let session_id = uuid::Uuid::new_v4().to_string();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let pending = MfaPending {
            code: code.clone(),
            email: email.to_string(),
            org: org.to_string(),
            role,
            regions,
            expires_at: now + 300, // 5 minutes
            attempts_remaining: 3,
        };

        self.pending
            .write()
            .unwrap()
            .insert(session_id.clone(), pending);

        (session_id, code)
    }

    /// Verify an MFA code for a given session.
    pub fn verify(&self, session_id: &str, code: &str) -> MfaVerifyResult {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let mut map = self.pending.write().unwrap();
        let Some(pending) = map.get_mut(session_id) else {
            return MfaVerifyResult::NotFound;
        };

        // Check expiration
        if now > pending.expires_at {
            map.remove(session_id);
            return MfaVerifyResult::Expired;
        }

        // Check attempts
        if pending.attempts_remaining == 0 {
            map.remove(session_id);
            return MfaVerifyResult::LockedOut;
        }

        // Constant-time comparison to prevent timing attacks
        if !constant_time_eq(code.as_bytes(), pending.code.as_bytes()) {
            pending.attempts_remaining -= 1;
            let remaining = pending.attempts_remaining;
            tracing::warn!(
                email = %pending.email,
                remaining = remaining,
                "Admin MFA code verification failed"
            );
            return MfaVerifyResult::WrongCode {
                attempts_remaining: remaining,
            };
        }

        // Success: extract data and remove pending entry
        let result = MfaVerifyResult::Success {
            email: pending.email.clone(),
            org: pending.org.clone(),
            role: pending.role,
            regions: pending.regions.clone(),
        };
        map.remove(session_id);
        result
    }

    /// Get the masked admin phone number for display.
    pub fn masked_phone(&self) -> String {
        mask_phone(&self.admin_phone)
    }
}

/// Generate a cryptographically random 6-digit MFA code.
pub fn generate_mfa_code() -> String {
    // Use UUID v4 (128 random bits from OS CSPRNG) and derive a 6-digit code.
    let bytes = uuid::Uuid::new_v4().into_bytes();
    let n = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    format!("{:06}", n % 1_000_000)
}

/// Mask a phone number for display: show first 4 and last 2 chars.
pub fn mask_phone(phone: &str) -> String {
    if phone.len() <= 6 {
        return "***".to_string();
    }
    let prefix = &phone[..4];
    let suffix = &phone[phone.len() - 2..];
    format!("{prefix}***{suffix}")
}

/// Constant-time byte comparison (prevents timing side-channel attacks).
pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_service() -> MfaService {
        MfaService::new(
            vec!["admin@test.com".to_string(), "david@openie.dev".to_string()],
            "+1-941-256-2032".to_string(),
        )
    }

    #[test]
    fn admin_email_check() {
        let svc = test_service();
        assert!(svc.is_admin_email("admin@test.com"));
        assert!(svc.is_admin_email("ADMIN@TEST.COM"));
        assert!(!svc.is_admin_email("nobody@test.com"));
    }

    #[test]
    fn mfa_create_and_verify() {
        let svc = test_service();
        let (session_id, code) =
            svc.create_challenge("admin@test.com", "acme", Role::Admin, vec![]);

        assert!(!session_id.is_empty());
        assert_eq!(code.len(), 6);

        match svc.verify(&session_id, &code) {
            MfaVerifyResult::Success { email, org, .. } => {
                assert_eq!(email, "admin@test.com");
                assert_eq!(org, "acme");
            }
            other => panic!("Expected Success, got {:?}", other),
        }

        // Session consumed -- second verify returns NotFound
        assert!(matches!(
            svc.verify(&session_id, &code),
            MfaVerifyResult::NotFound
        ));
    }

    #[test]
    fn mfa_wrong_code_decrements() {
        let svc = test_service();
        let (session_id, _code) =
            svc.create_challenge("admin@test.com", "acme", Role::Admin, vec![]);

        match svc.verify(&session_id, "000000") {
            MfaVerifyResult::WrongCode {
                attempts_remaining: 2,
            } => {}
            other => panic!("Expected WrongCode with 2 remaining, got {:?}", other),
        }
    }

    #[test]
    fn mfa_lockout_after_3_failures() {
        let svc = test_service();
        let (session_id, _code) =
            svc.create_challenge("admin@test.com", "acme", Role::Admin, vec![]);

        for _ in 0..3 {
            let result = svc.verify(&session_id, "000000");
            assert!(matches!(
                result,
                MfaVerifyResult::WrongCode { .. } | MfaVerifyResult::LockedOut
            ));
        }

        assert!(matches!(
            svc.verify(&session_id, "000000"),
            MfaVerifyResult::NotFound | MfaVerifyResult::LockedOut
        ));
    }

    #[test]
    fn mfa_code_is_six_digits() {
        for _ in 0..100 {
            let code = generate_mfa_code();
            assert_eq!(code.len(), 6);
            assert!(code.chars().all(|c| c.is_ascii_digit()));
        }
    }

    #[test]
    fn mask_phone_works() {
        assert_eq!(mask_phone("+1-941-256-2032"), "+1-9***32");
        assert_eq!(mask_phone("short"), "***");
    }

    #[test]
    fn constant_time_eq_works() {
        assert!(constant_time_eq(b"hello", b"hello"));
        assert!(!constant_time_eq(b"hello", b"world"));
        assert!(!constant_time_eq(b"hello", b"hell"));
    }

    #[test]
    fn not_found_for_unknown_session() {
        let svc = test_service();
        assert!(matches!(
            svc.verify("nonexistent", "123456"),
            MfaVerifyResult::NotFound
        ));
    }

    #[test]
    fn masked_phone_accessor() {
        let svc = test_service();
        let masked = svc.masked_phone();
        assert!(masked.contains("***"));
        assert!(!masked.contains("2032"));
    }
}
