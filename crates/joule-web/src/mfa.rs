//! Multi-factor authentication — TOTP (RFC 6238) with HMAC-SHA1, recovery codes,
//! MFA enrollment, verification flow, backup methods, and rate limiting attempts.
//!
//! Replaces speakeasy, otplib, node-2fa, and similar JS/TS MFA libraries
//! with a pure-Rust implementation suitable for WASM and native targets.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── Errors ─────────────────────────────────────────────────────

/// MFA engine errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MfaError {
    /// User not found.
    UserNotFound(String),
    /// MFA already enrolled for this user.
    AlreadyEnrolled(String),
    /// MFA not enrolled for this user.
    NotEnrolled(String),
    /// Invalid TOTP code.
    InvalidCode,
    /// Code already used (replay protection).
    CodeReused,
    /// Rate limit exceeded.
    RateLimited { retry_after_secs: u64 },
    /// Invalid recovery code.
    InvalidRecoveryCode,
    /// No recovery codes remaining.
    NoRecoveryCodesLeft,
    /// Enrollment not yet verified.
    EnrollmentNotVerified,
    /// Secret too short.
    InvalidSecret(String),
}

impl fmt::Display for MfaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UserNotFound(id) => write!(f, "user not found: {id}"),
            Self::AlreadyEnrolled(id) => write!(f, "already enrolled: {id}"),
            Self::NotEnrolled(id) => write!(f, "not enrolled: {id}"),
            Self::InvalidCode => write!(f, "invalid TOTP code"),
            Self::CodeReused => write!(f, "code already used"),
            Self::RateLimited { retry_after_secs } => {
                write!(f, "rate limited, retry after {retry_after_secs}s")
            }
            Self::InvalidRecoveryCode => write!(f, "invalid recovery code"),
            Self::NoRecoveryCodesLeft => write!(f, "no recovery codes remaining"),
            Self::EnrollmentNotVerified => write!(f, "enrollment not yet verified"),
            Self::InvalidSecret(msg) => write!(f, "invalid secret: {msg}"),
        }
    }
}

impl std::error::Error for MfaError {}

// ── HMAC-SHA1 (minimal, RFC 2104) ─────────────────────────────

/// SHA-1 block/output sizes.
const SHA1_BLOCK_SIZE: usize = 64;
const SHA1_DIGEST_SIZE: usize = 20;

/// Minimal SHA-1 (FIPS 180-4). Only used internally for HMAC-SHA1 in TOTP.
fn sha1(data: &[u8]) -> [u8; SHA1_DIGEST_SIZE] {
    let mut h0: u32 = 0x67452301;
    let mut h1: u32 = 0xEFCDAB89;
    let mut h2: u32 = 0x98BADCFE;
    let mut h3: u32 = 0x10325476;
    let mut h4: u32 = 0xC3D2E1F0;

    // Pre-processing: pad message
    let bit_len = (data.len() as u64) * 8;
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0x00);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    // Process each 512-bit block
    for block in msg.chunks(64) {
        let mut w = [0u32; 80];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[i * 4],
                block[i * 4 + 1],
                block[i * 4 + 2],
                block[i * 4 + 3],
            ]);
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }

        let (mut a, mut b, mut c, mut d, mut e) = (h0, h1, h2, h3, h4);

        for i in 0..80 {
            let (f_val, k) = match i {
                0..=19 => ((b & c) | ((!b) & d), 0x5A827999u32),
                20..=39 => (b ^ c ^ d, 0x6ED9EBA1u32),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1BBCDCu32),
                _ => (b ^ c ^ d, 0xCA62C1D6u32),
            };
            let temp = a
                .rotate_left(5)
                .wrapping_add(f_val)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(w[i]);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = temp;
        }

        h0 = h0.wrapping_add(a);
        h1 = h1.wrapping_add(b);
        h2 = h2.wrapping_add(c);
        h3 = h3.wrapping_add(d);
        h4 = h4.wrapping_add(e);
    }

    let mut digest = [0u8; 20];
    digest[0..4].copy_from_slice(&h0.to_be_bytes());
    digest[4..8].copy_from_slice(&h1.to_be_bytes());
    digest[8..12].copy_from_slice(&h2.to_be_bytes());
    digest[12..16].copy_from_slice(&h3.to_be_bytes());
    digest[16..20].copy_from_slice(&h4.to_be_bytes());
    digest
}

/// HMAC-SHA1 (RFC 2104).
fn hmac_sha1(key: &[u8], message: &[u8]) -> [u8; SHA1_DIGEST_SIZE] {
    let mut padded_key = [0u8; SHA1_BLOCK_SIZE];
    if key.len() > SHA1_BLOCK_SIZE {
        let hashed = sha1(key);
        padded_key[..SHA1_DIGEST_SIZE].copy_from_slice(&hashed);
    } else {
        padded_key[..key.len()].copy_from_slice(key);
    }

    let mut ipad = [0x36u8; SHA1_BLOCK_SIZE];
    let mut opad = [0x5Cu8; SHA1_BLOCK_SIZE];
    for i in 0..SHA1_BLOCK_SIZE {
        ipad[i] ^= padded_key[i];
        opad[i] ^= padded_key[i];
    }

    let mut inner = Vec::with_capacity(SHA1_BLOCK_SIZE + message.len());
    inner.extend_from_slice(&ipad);
    inner.extend_from_slice(message);
    let inner_hash = sha1(&inner);

    let mut outer = Vec::with_capacity(SHA1_BLOCK_SIZE + SHA1_DIGEST_SIZE);
    outer.extend_from_slice(&opad);
    outer.extend_from_slice(&inner_hash);
    sha1(&outer)
}

// ── TOTP (RFC 6238) ───────────────────────────────────────────

/// TOTP parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TotpParams {
    /// Shared secret (raw bytes).
    pub secret: Vec<u8>,
    /// Time step in seconds (default 30).
    pub period_secs: u64,
    /// Number of digits (default 6).
    pub digits: u32,
    /// Window of time steps to accept (default 1 = current +/- 1).
    pub skew: u64,
}

impl TotpParams {
    pub fn new(secret: Vec<u8>) -> Result<Self, MfaError> {
        if secret.len() < 10 {
            return Err(MfaError::InvalidSecret(
                "secret must be at least 10 bytes".to_string(),
            ));
        }
        Ok(Self {
            secret,
            period_secs: 30,
            digits: 6,
            skew: 1,
        })
    }

    /// Generate a TOTP code for a given Unix timestamp.
    pub fn generate(&self, timestamp_secs: u64) -> u32 {
        let counter = timestamp_secs / self.period_secs;
        self.generate_at_counter(counter)
    }

    /// Generate TOTP at a specific counter value.
    fn generate_at_counter(&self, counter: u64) -> u32 {
        let msg = counter.to_be_bytes();
        let hash = hmac_sha1(&self.secret, &msg);

        // Dynamic truncation (RFC 4226 Section 5.4)
        let offset = (hash[19] & 0x0F) as usize;
        let code = ((hash[offset] as u32 & 0x7F) << 24)
            | ((hash[offset + 1] as u32) << 16)
            | ((hash[offset + 2] as u32) << 8)
            | (hash[offset + 3] as u32);

        code % 10u32.pow(self.digits)
    }

    /// Verify a TOTP code within the allowed skew window.
    /// Returns the counter value that matched, or None.
    pub fn verify(&self, code: u32, timestamp_secs: u64) -> Option<u64> {
        let counter = timestamp_secs / self.period_secs;
        let start = counter.saturating_sub(self.skew);
        let end = counter + self.skew;
        for c in start..=end {
            if self.generate_at_counter(c) == code {
                return Some(c);
            }
        }
        None
    }

    /// Generate an otpauth:// URI for QR code enrollment.
    pub fn to_otpauth_uri(&self, issuer: &str, account: &str) -> String {
        let secret_b32 = base32_encode(&self.secret);
        format!(
            "otpauth://totp/{issuer}:{account}?secret={secret_b32}&issuer={issuer}&period={}&digits={}",
            self.period_secs, self.digits
        )
    }
}

/// Simple base32 encoder (RFC 4648, no padding).
fn base32_encode(data: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";
    let mut result = String::new();
    let mut buffer: u64 = 0;
    let mut bits = 0;

    for &byte in data {
        buffer = (buffer << 8) | byte as u64;
        bits += 8;
        while bits >= 5 {
            bits -= 5;
            let idx = ((buffer >> bits) & 0x1F) as usize;
            result.push(ALPHABET[idx] as char);
        }
    }
    if bits > 0 {
        let idx = ((buffer << (5 - bits)) & 0x1F) as usize;
        result.push(ALPHABET[idx] as char);
    }
    result
}

// ── Recovery Codes ────────────────────────────────────────────

/// A set of single-use recovery codes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryCodes {
    /// Hashed codes (simple hash for demo; in production use bcrypt/argon2).
    codes: Vec<RecoveryCodeEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RecoveryCodeEntry {
    /// The code value (stored plaintext here; in prod, store hash).
    code: String,
    used: bool,
}

impl RecoveryCodes {
    /// Generate a set of recovery codes deterministically from a seed.
    pub fn generate(seed: u64, count: usize) -> (Self, Vec<String>) {
        let mut codes = Vec::with_capacity(count);
        let mut plaintext = Vec::with_capacity(count);

        for i in 0..count {
            // Simple deterministic code generation from seed.
            let val = seed.wrapping_mul(6364136223846793005).wrapping_add(i as u64);
            let code = format!("{:08X}", val as u32);
            plaintext.push(code.clone());
            codes.push(RecoveryCodeEntry { code, used: false });
        }

        (Self { codes }, plaintext)
    }

    /// Attempt to use a recovery code. Returns true if valid and not yet used.
    pub fn use_code(&mut self, code: &str) -> bool {
        for entry in &mut self.codes {
            if !entry.used && entry.code == code {
                entry.used = true;
                return true;
            }
        }
        false
    }

    /// Count remaining (unused) codes.
    pub fn remaining(&self) -> usize {
        self.codes.iter().filter(|c| !c.used).count()
    }

    /// Total count.
    pub fn total(&self) -> usize {
        self.codes.len()
    }
}

// ── Rate Limiter ──────────────────────────────────────────────

/// Sliding-window rate limiter for MFA attempts.
#[derive(Debug, Clone)]
pub struct AttemptTracker {
    /// Max attempts within the window.
    max_attempts: u32,
    /// Window size in seconds.
    window_secs: u64,
    /// Map from user ID to list of attempt timestamps.
    attempts: HashMap<String, Vec<u64>>,
}

impl AttemptTracker {
    pub fn new(max_attempts: u32, window_secs: u64) -> Self {
        Self {
            max_attempts,
            window_secs,
            attempts: HashMap::new(),
        }
    }

    /// Record an attempt. Returns Err if rate limited.
    pub fn record_attempt(
        &mut self,
        user_id: &str,
        now_secs: u64,
    ) -> Result<(), MfaError> {
        let window_start = now_secs.saturating_sub(self.window_secs);
        let entries = self.attempts.entry(user_id.to_string()).or_default();
        entries.retain(|t| *t > window_start);

        if entries.len() >= self.max_attempts as usize {
            let oldest = entries.first().copied().unwrap_or(now_secs);
            let retry_after = oldest + self.window_secs - now_secs;
            return Err(MfaError::RateLimited {
                retry_after_secs: retry_after,
            });
        }

        entries.push(now_secs);
        Ok(())
    }

    /// Reset attempts for a user (e.g., after successful auth).
    pub fn reset(&mut self, user_id: &str) {
        self.attempts.remove(user_id);
    }
}

// ── Enrollment State ──────────────────────────────────────────

/// MFA enrollment status.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnrollmentStatus {
    /// Not enrolled.
    None,
    /// Enrollment started, awaiting first code verification.
    Pending,
    /// Fully enrolled and verified.
    Active,
    /// Disabled (e.g., by admin).
    Disabled,
}

/// Per-user MFA state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MfaState {
    pub user_id: String,
    pub status: EnrollmentStatus,
    pub totp: Option<TotpParams>,
    pub recovery: Option<RecoveryCodes>,
    /// Last verified counter (replay protection).
    pub last_counter: Option<u64>,
    /// Backup method (e.g., email, SMS identifier).
    pub backup_method: Option<BackupMethod>,
}

/// Supported backup MFA methods.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackupMethod {
    Email(String),
    Sms(String),
}

// ── MFA Engine ────────────────────────────────────────────────

/// The MFA engine managing enrollment and verification.
#[derive(Debug, Clone)]
pub struct MfaEngine {
    users: HashMap<String, MfaState>,
    rate_limiter: AttemptTracker,
    recovery_code_count: usize,
}

impl MfaEngine {
    pub fn new(max_attempts: u32, window_secs: u64) -> Self {
        Self {
            users: HashMap::new(),
            rate_limiter: AttemptTracker::new(max_attempts, window_secs),
            recovery_code_count: 8,
        }
    }

    /// Begin MFA enrollment. Returns the TOTP params and recovery codes.
    pub fn begin_enrollment(
        &mut self,
        user_id: &str,
        secret: Vec<u8>,
        seed: u64,
    ) -> Result<(TotpParams, Vec<String>), MfaError> {
        if let Some(state) = self.users.get(user_id) {
            if state.status == EnrollmentStatus::Active {
                return Err(MfaError::AlreadyEnrolled(user_id.to_string()));
            }
        }

        let totp = TotpParams::new(secret)?;
        let (recovery, codes) = RecoveryCodes::generate(seed, self.recovery_code_count);

        self.users.insert(
            user_id.to_string(),
            MfaState {
                user_id: user_id.to_string(),
                status: EnrollmentStatus::Pending,
                totp: Some(totp.clone()),
                recovery: Some(recovery),
                last_counter: None,
                backup_method: None,
            },
        );

        Ok((totp, codes))
    }

    /// Verify the first code to complete enrollment.
    pub fn verify_enrollment(
        &mut self,
        user_id: &str,
        code: u32,
        now_secs: u64,
    ) -> Result<(), MfaError> {
        let state = self
            .users
            .get_mut(user_id)
            .ok_or_else(|| MfaError::UserNotFound(user_id.to_string()))?;

        if state.status != EnrollmentStatus::Pending {
            return Err(MfaError::NotEnrolled(user_id.to_string()));
        }

        let totp = state
            .totp
            .as_ref()
            .ok_or_else(|| MfaError::NotEnrolled(user_id.to_string()))?;

        match totp.verify(code, now_secs) {
            Some(counter) => {
                state.status = EnrollmentStatus::Active;
                state.last_counter = Some(counter);
                self.rate_limiter.reset(user_id);
                Ok(())
            }
            None => Err(MfaError::InvalidCode),
        }
    }

    /// Verify a TOTP code for an enrolled user.
    pub fn verify_code(
        &mut self,
        user_id: &str,
        code: u32,
        now_secs: u64,
    ) -> Result<(), MfaError> {
        self.rate_limiter.record_attempt(user_id, now_secs)?;

        let state = self
            .users
            .get_mut(user_id)
            .ok_or_else(|| MfaError::UserNotFound(user_id.to_string()))?;

        if state.status != EnrollmentStatus::Active {
            return Err(MfaError::EnrollmentNotVerified);
        }

        let totp = state
            .totp
            .as_ref()
            .ok_or_else(|| MfaError::NotEnrolled(user_id.to_string()))?;

        match totp.verify(code, now_secs) {
            Some(counter) => {
                // Replay protection: reject if counter was already used.
                if let Some(last) = state.last_counter {
                    if counter <= last {
                        return Err(MfaError::CodeReused);
                    }
                }
                state.last_counter = Some(counter);
                self.rate_limiter.reset(user_id);
                Ok(())
            }
            None => Err(MfaError::InvalidCode),
        }
    }

    /// Use a recovery code as a backup method.
    pub fn use_recovery_code(
        &mut self,
        user_id: &str,
        code: &str,
    ) -> Result<usize, MfaError> {
        let state = self
            .users
            .get_mut(user_id)
            .ok_or_else(|| MfaError::UserNotFound(user_id.to_string()))?;

        if state.status != EnrollmentStatus::Active {
            return Err(MfaError::EnrollmentNotVerified);
        }

        let recovery = state
            .recovery
            .as_mut()
            .ok_or(MfaError::NoRecoveryCodesLeft)?;

        if recovery.remaining() == 0 {
            return Err(MfaError::NoRecoveryCodesLeft);
        }

        if recovery.use_code(code) {
            Ok(recovery.remaining())
        } else {
            Err(MfaError::InvalidRecoveryCode)
        }
    }

    /// Set a backup MFA method.
    pub fn set_backup_method(
        &mut self,
        user_id: &str,
        method: BackupMethod,
    ) -> Result<(), MfaError> {
        let state = self
            .users
            .get_mut(user_id)
            .ok_or_else(|| MfaError::UserNotFound(user_id.to_string()))?;
        state.backup_method = Some(method);
        Ok(())
    }

    /// Get enrollment status.
    pub fn status(&self, user_id: &str) -> EnrollmentStatus {
        self.users
            .get(user_id)
            .map(|s| s.status.clone())
            .unwrap_or(EnrollmentStatus::None)
    }

    /// Disable MFA for a user.
    pub fn disable(&mut self, user_id: &str) -> Result<(), MfaError> {
        let state = self
            .users
            .get_mut(user_id)
            .ok_or_else(|| MfaError::UserNotFound(user_id.to_string()))?;
        state.status = EnrollmentStatus::Disabled;
        Ok(())
    }

    /// Get recovery codes remaining count.
    pub fn recovery_codes_remaining(&self, user_id: &str) -> Result<usize, MfaError> {
        let state = self
            .users
            .get(user_id)
            .ok_or_else(|| MfaError::UserNotFound(user_id.to_string()))?;
        Ok(state
            .recovery
            .as_ref()
            .map(|r| r.remaining())
            .unwrap_or(0))
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_SECRET: &[u8] = b"12345678901234567890"; // 20 bytes

    #[test]
    fn test_sha1_empty() {
        let digest = sha1(b"");
        let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, "da39a3ee5e6b4b0d3255bfef95601890afd80709");
    }

    #[test]
    fn test_sha1_abc() {
        let digest = sha1(b"abc");
        let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, "a9993e364706816aba3e25717850c26c9cd0d89d");
    }

    #[test]
    fn test_hmac_sha1_rfc2202_vector() {
        // RFC 2202 test case 2
        let key = b"Jefe";
        let data = b"what do ya want for nothing?";
        let mac = hmac_sha1(key, data);
        let hex: String = mac.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, "effcdf6ae5eb2fa2d27416d5f184df9c259a7c79");
    }

    #[test]
    fn test_totp_generate_known_vector() {
        // RFC 6238 test vector for SHA-1, time=59, secret="12345678901234567890"
        let totp = TotpParams::new(TEST_SECRET.to_vec()).unwrap();
        let code = totp.generate(59);
        assert_eq!(code, 287082);
    }

    #[test]
    fn test_totp_verify_within_skew() {
        let totp = TotpParams::new(TEST_SECRET.to_vec()).unwrap();
        let timestamp = 59u64;
        let code = totp.generate(timestamp);
        // Verify at exact time
        assert!(totp.verify(code, timestamp).is_some());
        // Verify within skew (1 period ahead)
        assert!(totp.verify(code, timestamp + 30).is_some());
    }

    #[test]
    fn test_totp_reject_outside_skew() {
        let totp = TotpParams::new(TEST_SECRET.to_vec()).unwrap();
        let code = totp.generate(59);
        // 3 periods ahead is outside skew=1
        assert!(totp.verify(code, 59 + 90).is_none());
    }

    #[test]
    fn test_totp_secret_too_short() {
        let err = TotpParams::new(b"short".to_vec()).unwrap_err();
        assert!(matches!(err, MfaError::InvalidSecret(_)));
    }

    #[test]
    fn test_base32_encode() {
        assert_eq!(base32_encode(b""), "");
        assert_eq!(base32_encode(b"f"), "MY");
        assert_eq!(base32_encode(b"fo"), "MZXQ");
        assert_eq!(base32_encode(b"foo"), "MZXW6");
        assert_eq!(base32_encode(b"foob"), "MZXW6YQ");
        assert_eq!(base32_encode(b"fooba"), "MZXW6YTB");
        assert_eq!(base32_encode(b"foobar"), "MZXW6YTBOI");
    }

    #[test]
    fn test_otpauth_uri() {
        let totp = TotpParams::new(TEST_SECRET.to_vec()).unwrap();
        let uri = totp.to_otpauth_uri("JouleApp", "alice@example.com");
        assert!(uri.starts_with("otpauth://totp/JouleApp:alice@example.com?"));
        assert!(uri.contains("secret="));
        assert!(uri.contains("period=30"));
        assert!(uri.contains("digits=6"));
    }

    #[test]
    fn test_recovery_codes_generate() {
        let (recovery, codes) = RecoveryCodes::generate(42, 8);
        assert_eq!(codes.len(), 8);
        assert_eq!(recovery.total(), 8);
        assert_eq!(recovery.remaining(), 8);
    }

    #[test]
    fn test_recovery_code_use() {
        let (mut recovery, codes) = RecoveryCodes::generate(42, 4);
        assert!(recovery.use_code(&codes[0]));
        assert_eq!(recovery.remaining(), 3);
        // Can't reuse
        assert!(!recovery.use_code(&codes[0]));
        assert_eq!(recovery.remaining(), 3);
    }

    #[test]
    fn test_recovery_code_invalid() {
        let (mut recovery, _codes) = RecoveryCodes::generate(42, 4);
        assert!(!recovery.use_code("INVALID_CODE"));
    }

    #[test]
    fn test_rate_limiter() {
        let mut tracker = AttemptTracker::new(3, 60);
        assert!(tracker.record_attempt("u1", 100).is_ok());
        assert!(tracker.record_attempt("u1", 101).is_ok());
        assert!(tracker.record_attempt("u1", 102).is_ok());
        // 4th attempt should be rate limited
        let err = tracker.record_attempt("u1", 103).unwrap_err();
        assert!(matches!(err, MfaError::RateLimited { .. }));
    }

    #[test]
    fn test_rate_limiter_window_expiry() {
        let mut tracker = AttemptTracker::new(2, 10);
        assert!(tracker.record_attempt("u1", 100).is_ok());
        assert!(tracker.record_attempt("u1", 101).is_ok());
        // After window expires
        assert!(tracker.record_attempt("u1", 112).is_ok());
    }

    #[test]
    fn test_rate_limiter_reset() {
        let mut tracker = AttemptTracker::new(1, 60);
        assert!(tracker.record_attempt("u1", 100).is_ok());
        assert!(tracker.record_attempt("u1", 101).is_err());
        tracker.reset("u1");
        assert!(tracker.record_attempt("u1", 102).is_ok());
    }

    #[test]
    fn test_enrollment_flow() {
        let mut engine = MfaEngine::new(5, 60);
        let (totp, _codes) = engine
            .begin_enrollment("alice", TEST_SECRET.to_vec(), 42)
            .unwrap();

        assert_eq!(engine.status("alice"), EnrollmentStatus::Pending);

        // Verify enrollment with correct code
        let timestamp = 1000u64;
        let code = totp.generate(timestamp);
        engine.verify_enrollment("alice", code, timestamp).unwrap();

        assert_eq!(engine.status("alice"), EnrollmentStatus::Active);
    }

    #[test]
    fn test_enrollment_already_active() {
        let mut engine = MfaEngine::new(5, 60);
        let (totp, _) = engine
            .begin_enrollment("alice", TEST_SECRET.to_vec(), 42)
            .unwrap();
        let code = totp.generate(1000);
        engine.verify_enrollment("alice", code, 1000).unwrap();

        let err = engine
            .begin_enrollment("alice", TEST_SECRET.to_vec(), 43)
            .unwrap_err();
        assert!(matches!(err, MfaError::AlreadyEnrolled(_)));
    }

    #[test]
    fn test_verify_code_active_user() {
        let mut engine = MfaEngine::new(5, 60);
        let (totp, _) = engine
            .begin_enrollment("alice", TEST_SECRET.to_vec(), 42)
            .unwrap();

        // Complete enrollment
        let t1 = 1000u64;
        let code1 = totp.generate(t1);
        engine.verify_enrollment("alice", code1, t1).unwrap();

        // Now verify a code at a later time
        let t2 = 1060u64; // 2 periods later
        let code2 = totp.generate(t2);
        engine.verify_code("alice", code2, t2).unwrap();
    }

    #[test]
    fn test_replay_protection() {
        let mut engine = MfaEngine::new(5, 60);
        let (totp, _) = engine
            .begin_enrollment("alice", TEST_SECRET.to_vec(), 42)
            .unwrap();

        let t1 = 1000u64;
        let code1 = totp.generate(t1);
        engine.verify_enrollment("alice", code1, t1).unwrap();

        // Same code (same counter) should be rejected
        let code_same = totp.generate(t1);
        let err = engine.verify_code("alice", code_same, t1).unwrap_err();
        assert!(matches!(err, MfaError::CodeReused));
    }

    #[test]
    fn test_use_recovery_code_via_engine() {
        let mut engine = MfaEngine::new(5, 60);
        let (totp, codes) = engine
            .begin_enrollment("alice", TEST_SECRET.to_vec(), 42)
            .unwrap();

        // Complete enrollment
        let code = totp.generate(1000);
        engine.verify_enrollment("alice", code, 1000).unwrap();

        let remaining = engine.use_recovery_code("alice", &codes[0]).unwrap();
        assert_eq!(remaining, 7);

        // Invalid code
        let err = engine.use_recovery_code("alice", "BADCODE").unwrap_err();
        assert!(matches!(err, MfaError::InvalidRecoveryCode));
    }

    #[test]
    fn test_backup_method() {
        let mut engine = MfaEngine::new(5, 60);
        engine
            .begin_enrollment("alice", TEST_SECRET.to_vec(), 42)
            .unwrap();

        engine
            .set_backup_method("alice", BackupMethod::Email("a@b.com".to_string()))
            .unwrap();

        let state = engine.users.get("alice").unwrap();
        assert_eq!(
            state.backup_method,
            Some(BackupMethod::Email("a@b.com".to_string()))
        );
    }

    #[test]
    fn test_disable_mfa() {
        let mut engine = MfaEngine::new(5, 60);
        let (totp, _) = engine
            .begin_enrollment("alice", TEST_SECRET.to_vec(), 42)
            .unwrap();
        let code = totp.generate(1000);
        engine.verify_enrollment("alice", code, 1000).unwrap();

        engine.disable("alice").unwrap();
        assert_eq!(engine.status("alice"), EnrollmentStatus::Disabled);
    }

    #[test]
    fn test_not_enrolled_status() {
        let engine = MfaEngine::new(5, 60);
        assert_eq!(engine.status("nobody"), EnrollmentStatus::None);
    }

    #[test]
    fn test_recovery_codes_remaining() {
        let mut engine = MfaEngine::new(5, 60);
        let (totp, codes) = engine
            .begin_enrollment("alice", TEST_SECRET.to_vec(), 42)
            .unwrap();
        let code = totp.generate(1000);
        engine.verify_enrollment("alice", code, 1000).unwrap();

        assert_eq!(engine.recovery_codes_remaining("alice").unwrap(), 8);
        engine.use_recovery_code("alice", &codes[0]).unwrap();
        assert_eq!(engine.recovery_codes_remaining("alice").unwrap(), 7);
    }

    #[test]
    fn test_verify_not_enrolled() {
        let mut engine = MfaEngine::new(5, 60);
        let err = engine.verify_code("alice", 123456, 1000).unwrap_err();
        assert!(matches!(err, MfaError::UserNotFound(_)));
    }

    #[test]
    fn test_verify_enrollment_wrong_code() {
        let mut engine = MfaEngine::new(5, 60);
        engine
            .begin_enrollment("alice", TEST_SECRET.to_vec(), 42)
            .unwrap();

        let err = engine.verify_enrollment("alice", 999999, 1000).unwrap_err();
        assert!(matches!(err, MfaError::InvalidCode));
    }

    #[test]
    fn test_error_display() {
        assert_eq!(MfaError::InvalidCode.to_string(), "invalid TOTP code");
        assert_eq!(MfaError::CodeReused.to_string(), "code already used");
        assert_eq!(
            MfaError::RateLimited {
                retry_after_secs: 30
            }
            .to_string(),
            "rate limited, retry after 30s"
        );
    }

    #[test]
    fn test_totp_rfc6238_vector_1000000000() {
        // RFC 6238: time = 1111111109, SHA-1, expect 081804
        let totp = TotpParams::new(TEST_SECRET.to_vec()).unwrap();
        let code = totp.generate(1111111109);
        assert_eq!(code, 081804);
    }

    #[test]
    fn test_totp_rfc6238_vector_2000000000() {
        // RFC 6238: time = 2000000000, SHA-1, expect 279037
        let totp = TotpParams::new(TEST_SECRET.to_vec()).unwrap();
        let code = totp.generate(2000000000);
        assert_eq!(code, 279037);
    }

    #[test]
    fn test_recovery_exhaust_all() {
        let (mut recovery, codes) = RecoveryCodes::generate(99, 3);
        for code in &codes {
            assert!(recovery.use_code(code));
        }
        assert_eq!(recovery.remaining(), 0);
        assert!(!recovery.use_code(&codes[0]));
    }
}
