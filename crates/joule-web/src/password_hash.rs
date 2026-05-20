//! Password hashing — PBKDF2-SHA256 implementation with pure-Rust HMAC-SHA256,
//! configurable iterations, salt generation, hash verification, timing-safe
//! comparison, and password strength estimation (zxcvbn-lite).
//!
//! Replaces bcrypt, argon2, and zxcvbn with a self-contained password security toolkit.

use serde::{Deserialize, Serialize};

// ── Constants ──────────────────────────────────────────────────

/// Default PBKDF2 iteration count (OWASP 2024 recommendation for SHA-256).
pub const DEFAULT_ITERATIONS: u32 = 100_000;

/// Default derived key length in bytes.
pub const DEFAULT_DK_LEN: usize = 32;

/// Default salt length in bytes.
pub const DEFAULT_SALT_LEN: usize = 16;

/// Minimum acceptable password length.
pub const MIN_PASSWORD_LEN: usize = 8;

// ── Errors ─────────────────────────────────────────────────────

/// Password hashing errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PasswordHashError {
    /// Password too short.
    TooShort { min: usize, got: usize },
    /// Empty password.
    EmptyPassword,
    /// Invalid iteration count.
    InvalidIterations,
    /// Salt too short.
    SaltTooShort(usize),
    /// Invalid stored hash format.
    InvalidFormat(String),
    /// Derived key length too long.
    DkLenTooLong,
}

impl std::fmt::Display for PasswordHashError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooShort { min, got } => {
                write!(f, "password too short: {got} chars (min {min})")
            }
            Self::EmptyPassword => write!(f, "password must not be empty"),
            Self::InvalidIterations => write!(f, "iteration count must be >= 1"),
            Self::SaltTooShort(n) => write!(f, "salt too short: {n} bytes (min 8)"),
            Self::InvalidFormat(s) => write!(f, "invalid hash format: {s}"),
            Self::DkLenTooLong => write!(f, "derived key length too long"),
        }
    }
}

impl std::error::Error for PasswordHashError {}

// ── Inline SHA-256 ─────────────────────────────────────────────

const SHA256_H0: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

const SHA256_K: [u32; 64] = [
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
];

fn sha256_process_block(state: &mut [u32; 8], block: &[u8; 64]) {
    let mut w = [0u32; 64];
    for i in 0..16 {
        w[i] = u32::from_be_bytes([block[i*4], block[i*4+1], block[i*4+2], block[i*4+3]]);
    }
    for i in 16..64 {
        let s0 = w[i-15].rotate_right(7) ^ w[i-15].rotate_right(18) ^ (w[i-15] >> 3);
        let s1 = w[i-2].rotate_right(17) ^ w[i-2].rotate_right(19) ^ (w[i-2] >> 10);
        w[i] = w[i-16].wrapping_add(s0).wrapping_add(w[i-7]).wrapping_add(s1);
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    for i in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ ((!e) & g);
        let t1 = h.wrapping_add(s1).wrapping_add(ch).wrapping_add(SHA256_K[i]).wrapping_add(w[i]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let t2 = s0.wrapping_add(maj);
        h = g; g = f; f = e; e = d.wrapping_add(t1);
        d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    state[0] = state[0].wrapping_add(a); state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c); state[3] = state[3].wrapping_add(d);
    state[4] = state[4].wrapping_add(e); state[5] = state[5].wrapping_add(f);
    state[6] = state[6].wrapping_add(g); state[7] = state[7].wrapping_add(h);
}

fn sha256(data: &[u8]) -> [u8; 32] {
    let mut state = SHA256_H0;
    let bit_len = (data.len() as u64) * 8;
    let mut buf = data.to_vec();
    buf.push(0x80);
    while buf.len() % 64 != 56 { buf.push(0x00); }
    buf.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in buf.chunks_exact(64) {
        let block: [u8; 64] = chunk.try_into().unwrap();
        sha256_process_block(&mut state, &block);
    }
    let mut out = [0u8; 32];
    for (i, word) in state.iter().enumerate() {
        out[i*4..(i+1)*4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

fn hmac_sha256(key: &[u8], message: &[u8]) -> [u8; 32] {
    let k = if key.len() > 64 { sha256(key).to_vec() } else { key.to_vec() };
    let mut k_padded = [0u8; 64];
    k_padded[..k.len()].copy_from_slice(&k);
    let mut ipad = [0x36u8; 64];
    let mut opad = [0x5cu8; 64];
    for i in 0..64 { ipad[i] ^= k_padded[i]; opad[i] ^= k_padded[i]; }
    let mut inner = ipad.to_vec();
    inner.extend_from_slice(message);
    let inner_hash = sha256(&inner);
    let mut outer = opad.to_vec();
    outer.extend_from_slice(&inner_hash);
    sha256(&outer)
}

// ── PBKDF2 ─────────────────────────────────────────────────────

/// Derive a key using PBKDF2-HMAC-SHA256.
fn pbkdf2_derive(password: &[u8], salt: &[u8], iterations: u32, dk_len: usize) -> Vec<u8> {
    let num_blocks = (dk_len + 31) / 32;
    let mut dk = Vec::with_capacity(dk_len);

    for block_idx in 1..=num_blocks as u32 {
        let mut salt_i = salt.to_vec();
        salt_i.extend_from_slice(&block_idx.to_be_bytes());
        let mut u = hmac_sha256(password, &salt_i);
        let mut result = u;

        for _ in 1..iterations {
            u = hmac_sha256(password, &u);
            for j in 0..32 {
                result[j] ^= u[j];
            }
        }
        dk.extend_from_slice(&result);
    }
    dk.truncate(dk_len);
    dk
}

// ── Salt Generation ────────────────────────────────────────────

/// Generate a pseudo-random salt from system entropy.
fn generate_salt(len: usize) -> Vec<u8> {
    use std::time::{SystemTime, UNIX_EPOCH};
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let counter = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let stack_val: usize = 0;
    let stack_addr = &stack_val as *const _ as usize;

    let mut seed = Vec::new();
    seed.extend_from_slice(&ts.to_le_bytes());
    seed.extend_from_slice(&counter.to_le_bytes());
    seed.extend_from_slice(&stack_addr.to_le_bytes());
    seed.extend_from_slice(&(std::process::id() as u64).to_le_bytes());

    let mut salt = Vec::with_capacity(len);
    let mut idx = 0u64;
    while salt.len() < len {
        let mut block_seed = seed.clone();
        block_seed.extend_from_slice(&idx.to_le_bytes());
        let hash = sha256(&block_seed);
        let take = (len - salt.len()).min(32);
        salt.extend_from_slice(&hash[..take]);
        idx += 1;
    }
    salt
}

// ── Timing-Safe Comparison ─────────────────────────────────────

/// Constant-time byte comparison to prevent timing attacks.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for i in 0..a.len() {
        diff |= a[i] ^ b[i];
    }
    diff == 0
}

// ── Hex Utilities ──────────────────────────────────────────────

fn bytes_to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn hex_to_bytes(hex: &str) -> Result<Vec<u8>, PasswordHashError> {
    if hex.len() % 2 != 0 {
        return Err(PasswordHashError::InvalidFormat("odd hex length".to_string()));
    }
    let mut bytes = Vec::with_capacity(hex.len() / 2);
    let mut i = 0;
    while i < hex.len() {
        let byte = u8::from_str_radix(&hex[i..i + 2], 16)
            .map_err(|_| PasswordHashError::InvalidFormat("invalid hex".to_string()))?;
        bytes.push(byte);
        i += 2;
    }
    Ok(bytes)
}

// ── Stored Hash ────────────────────────────────────────────────

/// A stored password hash containing salt, iterations, and derived key.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoredHash {
    /// Salt bytes.
    pub salt: Vec<u8>,
    /// PBKDF2 iteration count.
    pub iterations: u32,
    /// Derived key length.
    pub dk_len: usize,
    /// Derived key (hash).
    pub hash: Vec<u8>,
}

impl StoredHash {
    /// Hash a password with default parameters.
    pub fn create(password: &str) -> Result<Self, PasswordHashError> {
        Self::create_with_params(password, DEFAULT_ITERATIONS, DEFAULT_DK_LEN, DEFAULT_SALT_LEN)
    }

    /// Hash a password with custom parameters.
    pub fn create_with_params(
        password: &str,
        iterations: u32,
        dk_len: usize,
        salt_len: usize,
    ) -> Result<Self, PasswordHashError> {
        if password.is_empty() {
            return Err(PasswordHashError::EmptyPassword);
        }
        if iterations < 1 {
            return Err(PasswordHashError::InvalidIterations);
        }
        if salt_len < 8 {
            return Err(PasswordHashError::SaltTooShort(salt_len));
        }
        let salt = generate_salt(salt_len);
        let hash = pbkdf2_derive(password.as_bytes(), &salt, iterations, dk_len);
        Ok(Self { salt, iterations, dk_len, hash })
    }

    /// Verify a password against this stored hash (timing-safe).
    pub fn verify(&self, password: &str) -> bool {
        let derived = pbkdf2_derive(password.as_bytes(), &self.salt, self.iterations, self.dk_len);
        constant_time_eq(&derived, &self.hash)
    }

    /// Serialize to portable string: `$pbkdf2-sha256$iterations$salt_hex$hash_hex`.
    pub fn to_string_repr(&self) -> String {
        format!(
            "$pbkdf2-sha256${}${}${}",
            self.iterations,
            bytes_to_hex(&self.salt),
            bytes_to_hex(&self.hash),
        )
    }

    /// Parse from portable string format.
    pub fn from_string_repr(s: &str) -> Result<Self, PasswordHashError> {
        let parts: Vec<&str> = s.split('$').collect();
        // Expected: ["", "pbkdf2-sha256", iterations, salt_hex, hash_hex]
        if parts.len() != 5 || parts[1] != "pbkdf2-sha256" {
            return Err(PasswordHashError::InvalidFormat(s.to_string()));
        }
        let iterations: u32 = parts[2]
            .parse()
            .map_err(|_| PasswordHashError::InvalidFormat("bad iterations".to_string()))?;
        let salt = hex_to_bytes(parts[3])?;
        let hash = hex_to_bytes(parts[4])?;
        let dk_len = hash.len();
        Ok(Self { salt, iterations, dk_len, hash })
    }

    /// Check if the hash needs rehashing (e.g., iterations increased).
    pub fn needs_rehash(&self, target_iterations: u32) -> bool {
        self.iterations < target_iterations
    }
}

// ── Password Strength Estimation ───────────────────────────────

/// Password strength score (0-4, like zxcvbn).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum StrengthScore {
    /// Very weak (0).
    VeryWeak = 0,
    /// Weak (1).
    Weak = 1,
    /// Fair (2).
    Fair = 2,
    /// Strong (3).
    Strong = 3,
    /// Very strong (4).
    VeryStrong = 4,
}

impl StrengthScore {
    /// Human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::VeryWeak => "very weak",
            Self::Weak => "weak",
            Self::Fair => "fair",
            Self::Strong => "strong",
            Self::VeryStrong => "very strong",
        }
    }
}

/// Detailed password strength analysis.
#[derive(Debug, Clone)]
pub struct StrengthReport {
    /// Overall score.
    pub score: StrengthScore,
    /// Estimated entropy in bits.
    pub entropy_bits: f64,
    /// Feedback suggestions.
    pub feedback: Vec<String>,
    /// Whether it contains uppercase letters.
    pub has_uppercase: bool,
    /// Whether it contains lowercase letters.
    pub has_lowercase: bool,
    /// Whether it contains digits.
    pub has_digits: bool,
    /// Whether it contains special characters.
    pub has_special: bool,
    /// Length of the password.
    pub length: usize,
}

/// Common passwords to check against (top 20).
const COMMON_PASSWORDS: &[&str] = &[
    "password", "123456", "12345678", "qwerty", "abc123",
    "monkey", "1234567", "letmein", "trustno1", "dragon",
    "baseball", "iloveyou", "master", "sunshine", "ashley",
    "bailey", "passw0rd", "shadow", "123123", "654321",
];

/// Estimate password strength (zxcvbn-lite).
pub fn estimate_strength(password: &str) -> StrengthReport {
    let length = password.len();
    let has_lowercase = password.chars().any(|c| c.is_ascii_lowercase());
    let has_uppercase = password.chars().any(|c| c.is_ascii_uppercase());
    let has_digits = password.chars().any(|c| c.is_ascii_digit());
    let has_special = password.chars().any(|c| !c.is_alphanumeric());

    let mut feedback = Vec::new();

    // Check common passwords.
    let is_common = COMMON_PASSWORDS
        .iter()
        .any(|p| p.eq_ignore_ascii_case(password));
    if is_common {
        feedback.push("This is a commonly used password".to_string());
    }

    // Check for repeated characters.
    let has_repeats = password
        .as_bytes()
        .windows(3)
        .any(|w| w[0] == w[1] && w[1] == w[2]);
    if has_repeats {
        feedback.push("Avoid repeated characters".to_string());
    }

    // Check for sequential characters.
    let has_sequential = password.as_bytes().windows(3).any(|w| {
        w[1] == w[0].wrapping_add(1) && w[2] == w[1].wrapping_add(1)
    });
    if has_sequential {
        feedback.push("Avoid sequential characters".to_string());
    }

    // Calculate character set size.
    let mut charset_size = 0u32;
    if has_lowercase { charset_size += 26; }
    if has_uppercase { charset_size += 26; }
    if has_digits { charset_size += 10; }
    if has_special { charset_size += 33; }
    if charset_size == 0 { charset_size = 1; }

    // Estimate entropy.
    let entropy_bits = (length as f64) * (charset_size as f64).log2();

    // Penalty for common passwords.
    let effective_entropy = if is_common { entropy_bits * 0.1 } else { entropy_bits };

    // Score based on entropy.
    let score = if effective_entropy < 20.0 {
        StrengthScore::VeryWeak
    } else if effective_entropy < 35.0 {
        StrengthScore::Weak
    } else if effective_entropy < 50.0 {
        StrengthScore::Fair
    } else if effective_entropy < 70.0 {
        StrengthScore::Strong
    } else {
        StrengthScore::VeryStrong
    };

    // Generate feedback.
    if length < 8 {
        feedback.push("Use at least 8 characters".to_string());
    }
    if !has_uppercase {
        feedback.push("Add uppercase letters".to_string());
    }
    if !has_lowercase {
        feedback.push("Add lowercase letters".to_string());
    }
    if !has_digits {
        feedback.push("Add numbers".to_string());
    }
    if !has_special {
        feedback.push("Add special characters".to_string());
    }

    StrengthReport {
        score,
        entropy_bits,
        feedback,
        has_uppercase,
        has_lowercase,
        has_digits,
        has_special,
        length,
    }
}

/// Validate that a password meets minimum requirements.
pub fn validate_password(password: &str, min_length: usize) -> Result<(), PasswordHashError> {
    if password.is_empty() {
        return Err(PasswordHashError::EmptyPassword);
    }
    if password.len() < min_length {
        return Err(PasswordHashError::TooShort {
            min: min_length,
            got: password.len(),
        });
    }
    Ok(())
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sha256_empty() {
        let digest = sha256(b"");
        let hex = bytes_to_hex(&digest);
        assert_eq!(hex, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    }

    #[test]
    fn test_sha256_hello() {
        let digest = sha256(b"hello");
        let hex = bytes_to_hex(&digest);
        assert_eq!(hex, "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824");
    }

    #[test]
    fn test_hmac_sha256_rfc4231_case1() {
        // Test case from RFC 4231.
        let key = vec![0x0bu8; 20];
        let data = b"Hi There";
        let mac = hmac_sha256(&key, data);
        let hex = bytes_to_hex(&mac);
        assert_eq!(hex, "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7");
    }

    #[test]
    fn test_create_and_verify() {
        let hash = StoredHash::create_with_params("test_password", 1000, 32, 16).unwrap();
        assert!(hash.verify("test_password"));
        assert!(!hash.verify("wrong_password"));
    }

    #[test]
    fn test_empty_password_rejected() {
        let err = StoredHash::create("").unwrap_err();
        assert_eq!(err, PasswordHashError::EmptyPassword);
    }

    #[test]
    fn test_salt_too_short() {
        let err = StoredHash::create_with_params("password", 1000, 32, 4).unwrap_err();
        match err {
            PasswordHashError::SaltTooShort(4) => {}
            _ => panic!("expected SaltTooShort"),
        }
    }

    #[test]
    fn test_string_repr_roundtrip() {
        let hash = StoredHash::create_with_params("roundtrip", 1000, 32, 16).unwrap();
        let repr = hash.to_string_repr();
        assert!(repr.starts_with("$pbkdf2-sha256$"));
        let parsed = StoredHash::from_string_repr(&repr).unwrap();
        assert_eq!(parsed.iterations, hash.iterations);
        assert_eq!(parsed.salt, hash.salt);
        assert_eq!(parsed.hash, hash.hash);
        assert!(parsed.verify("roundtrip"));
    }

    #[test]
    fn test_invalid_format() {
        assert!(StoredHash::from_string_repr("garbage").is_err());
        assert!(StoredHash::from_string_repr("$bcrypt$1000$aa$bb").is_err());
    }

    #[test]
    fn test_needs_rehash() {
        let hash = StoredHash::create_with_params("pw", 1000, 32, 16).unwrap();
        assert!(hash.needs_rehash(10000));
        assert!(!hash.needs_rehash(500));
        assert!(!hash.needs_rehash(1000));
    }

    #[test]
    fn test_timing_safe_comparison() {
        assert!(constant_time_eq(b"hello", b"hello"));
        assert!(!constant_time_eq(b"hello", b"world"));
        assert!(!constant_time_eq(b"hi", b"hello"));
    }

    #[test]
    fn test_unique_salts() {
        let s1 = generate_salt(16);
        let s2 = generate_salt(16);
        assert_ne!(s1, s2);
    }

    #[test]
    fn test_strength_very_weak() {
        let report = estimate_strength("123");
        assert_eq!(report.score, StrengthScore::VeryWeak);
        assert!(report.length == 3);
    }

    #[test]
    fn test_strength_common_password() {
        let report = estimate_strength("password");
        assert!(report.score <= StrengthScore::Weak);
        assert!(report.feedback.iter().any(|f| f.contains("commonly")));
    }

    #[test]
    fn test_strength_strong() {
        let report = estimate_strength("C0mpl3x!P@ssw0rd#2024");
        assert!(report.score >= StrengthScore::Strong);
        assert!(report.has_uppercase);
        assert!(report.has_lowercase);
        assert!(report.has_digits);
        assert!(report.has_special);
    }

    #[test]
    fn test_strength_repeated_chars() {
        let report = estimate_strength("aaa111bbb");
        assert!(report.feedback.iter().any(|f| f.contains("repeated")));
    }

    #[test]
    fn test_strength_sequential() {
        let report = estimate_strength("abcdef123");
        assert!(report.feedback.iter().any(|f| f.contains("sequential")));
    }

    #[test]
    fn test_strength_score_label() {
        assert_eq!(StrengthScore::VeryWeak.label(), "very weak");
        assert_eq!(StrengthScore::Weak.label(), "weak");
        assert_eq!(StrengthScore::Fair.label(), "fair");
        assert_eq!(StrengthScore::Strong.label(), "strong");
        assert_eq!(StrengthScore::VeryStrong.label(), "very strong");
    }

    #[test]
    fn test_validate_password_ok() {
        assert!(validate_password("longpassword", 8).is_ok());
    }

    #[test]
    fn test_validate_password_too_short() {
        let err = validate_password("short", 8).unwrap_err();
        match err {
            PasswordHashError::TooShort { min: 8, got: 5 } => {}
            _ => panic!("expected TooShort"),
        }
    }

    #[test]
    fn test_validate_password_empty() {
        assert_eq!(
            validate_password("", 1).unwrap_err(),
            PasswordHashError::EmptyPassword
        );
    }

    #[test]
    fn test_hex_roundtrip() {
        let data = vec![0xDE, 0xAD, 0xBE, 0xEF];
        let hex = bytes_to_hex(&data);
        assert_eq!(hex, "deadbeef");
        let back = hex_to_bytes(&hex).unwrap();
        assert_eq!(back, data);
    }

    #[test]
    fn test_different_passwords_different_hashes() {
        let h1 = StoredHash::create_with_params("password1", 100, 32, 16).unwrap();
        let h2 = StoredHash::create_with_params("password2", 100, 32, 16).unwrap();
        assert_ne!(h1.hash, h2.hash);
    }

    #[test]
    fn test_error_display() {
        let e = PasswordHashError::TooShort { min: 8, got: 3 };
        assert!(e.to_string().contains("too short"));
        let e2 = PasswordHashError::InvalidFormat("bad".to_string());
        assert!(e2.to_string().contains("bad"));
    }

    #[test]
    fn test_strength_score_ordering() {
        assert!(StrengthScore::VeryWeak < StrengthScore::Weak);
        assert!(StrengthScore::Weak < StrengthScore::Fair);
        assert!(StrengthScore::Fair < StrengthScore::Strong);
        assert!(StrengthScore::Strong < StrengthScore::VeryStrong);
    }
}
