//! PBKDF2 key derivation — pure-Rust PBKDF2-HMAC-SHA256.
//!
//! Replaces Node.js `crypto.pbkdf2` and browser SubtleCrypto `deriveBits`
//! with a zero-dependency key derivation function supporting configurable
//! iterations, salt generation, password hashing, and timing-safe verification.

use serde::{Deserialize, Serialize};
use std::fmt;

// ── Constants ──────────────────────────────────────────────────

/// Default iteration count (100_000 as of OWASP 2024 recommendation for SHA-256).
pub const DEFAULT_ITERATIONS: u32 = 100_000;

/// Default derived key length in bytes.
pub const DEFAULT_DK_LEN: usize = 32;

/// Default salt length in bytes.
pub const DEFAULT_SALT_LEN: usize = 16;

// ── Errors ─────────────────────────────────────────────────────

/// PBKDF2 domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Pbkdf2Error {
    /// Iteration count must be >= 1.
    InvalidIterations,
    /// Derived key length must be >= 1.
    InvalidDkLen,
    /// Empty password.
    EmptyPassword,
    /// Salt too short (min 8 bytes).
    SaltTooShort(usize),
    /// Invalid stored hash format.
    InvalidHashFormat(String),
    /// Derived key too long (max 32 * (2^32 - 1) bytes).
    DkLenTooLong,
}

impl fmt::Display for Pbkdf2Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidIterations => write!(f, "iteration count must be >= 1"),
            Self::InvalidDkLen => write!(f, "derived key length must be >= 1"),
            Self::EmptyPassword => write!(f, "password must not be empty"),
            Self::SaltTooShort(n) => write!(f, "salt too short: {n} bytes (min 8)"),
            Self::InvalidHashFormat(s) => write!(f, "invalid hash format: {s}"),
            Self::DkLenTooLong => write!(f, "derived key length too long"),
        }
    }
}

impl std::error::Error for Pbkdf2Error {}

// ── Inline HMAC-SHA256 ─────────────────────────────────────────
// We inline SHA-256 and HMAC to avoid cross-module deps.

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
pub fn pbkdf2_sha256(password: &[u8], salt: &[u8], iterations: u32, dk_len: usize) -> Result<Vec<u8>, Pbkdf2Error> {
    if iterations < 1 {
        return Err(Pbkdf2Error::InvalidIterations);
    }
    if dk_len < 1 {
        return Err(Pbkdf2Error::InvalidDkLen);
    }
    // Max derived key length: 32 * (2^32 - 1) bytes.
    let num_blocks = (dk_len + 31) / 32;
    if num_blocks > u32::MAX as usize {
        return Err(Pbkdf2Error::DkLenTooLong);
    }

    let mut dk = Vec::with_capacity(dk_len);

    for block_idx in 1..=num_blocks as u32 {
        // U_1 = HMAC(password, salt || INT_32_BE(block_idx))
        let mut salt_i = salt.to_vec();
        salt_i.extend_from_slice(&block_idx.to_be_bytes());
        let mut u = hmac_sha256(password, &salt_i);
        let mut result = u;

        // U_2 .. U_c
        for _ in 1..iterations {
            u = hmac_sha256(password, &u);
            for j in 0..32 {
                result[j] ^= u[j];
            }
        }

        dk.extend_from_slice(&result);
    }

    dk.truncate(dk_len);
    Ok(dk)
}

// ── Salt generation ────────────────────────────────────────────

/// Generate a pseudo-random salt from system entropy sources.
///
/// Uses a combination of timestamp, stack address, and counter for entropy.
/// Not cryptographically secure — in production use OS randomness.
pub fn generate_salt(len: usize) -> Vec<u8> {
    use std::time::{SystemTime, UNIX_EPOCH};

    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let counter = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let stack_entropy: usize = 0;
    let stack_addr = &stack_entropy as *const _ as usize;

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

// ── Password hash storage ──────────────────────────────────────

/// A stored password hash (salt + iterations + derived key).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PasswordHash {
    pub salt: Vec<u8>,
    pub iterations: u32,
    pub dk_len: usize,
    pub hash: Vec<u8>,
}

impl PasswordHash {
    /// Hash a password with default parameters.
    pub fn create(password: &[u8]) -> Result<Self, Pbkdf2Error> {
        Self::create_with_params(password, DEFAULT_ITERATIONS, DEFAULT_DK_LEN, DEFAULT_SALT_LEN)
    }

    /// Hash a password with custom parameters.
    pub fn create_with_params(
        password: &[u8],
        iterations: u32,
        dk_len: usize,
        salt_len: usize,
    ) -> Result<Self, Pbkdf2Error> {
        if password.is_empty() {
            return Err(Pbkdf2Error::EmptyPassword);
        }
        if salt_len < 8 {
            return Err(Pbkdf2Error::SaltTooShort(salt_len));
        }
        let salt = generate_salt(salt_len);
        let hash = pbkdf2_sha256(password, &salt, iterations, dk_len)?;
        Ok(Self { salt, iterations, dk_len, hash })
    }

    /// Verify a password against this stored hash.
    pub fn verify(&self, password: &[u8]) -> Result<bool, Pbkdf2Error> {
        let derived = pbkdf2_sha256(password, &self.salt, self.iterations, self.dk_len)?;
        Ok(constant_time_compare(&derived, &self.hash))
    }

    /// Serialize to a portable string: `$pbkdf2-sha256$iterations$salt_hex$hash_hex`.
    pub fn to_string_repr(&self) -> String {
        let salt_hex: String = self.salt.iter().map(|b| format!("{b:02x}")).collect();
        let hash_hex: String = self.hash.iter().map(|b| format!("{b:02x}")).collect();
        format!("$pbkdf2-sha256${}${}${}", self.iterations, salt_hex, hash_hex)
    }

    /// Parse from the portable string format.
    pub fn from_string_repr(s: &str) -> Result<Self, Pbkdf2Error> {
        let parts: Vec<&str> = s.split('$').collect();
        if parts.len() != 5 || parts[1] != "pbkdf2-sha256" {
            return Err(Pbkdf2Error::InvalidHashFormat(s.to_string()));
        }
        let iterations: u32 = parts[2]
            .parse()
            .map_err(|_| Pbkdf2Error::InvalidHashFormat(s.to_string()))?;
        let salt = hex_to_bytes(parts[3])
            .map_err(|_| Pbkdf2Error::InvalidHashFormat(s.to_string()))?;
        let hash = hex_to_bytes(parts[4])
            .map_err(|_| Pbkdf2Error::InvalidHashFormat(s.to_string()))?;
        Ok(Self {
            salt,
            iterations,
            dk_len: hash.len(),
            hash,
        })
    }
}

impl fmt::Display for PasswordHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_string_repr())
    }
}

// ── Helpers ────────────────────────────────────────────────────

/// Constant-time byte comparison.
pub fn constant_time_compare(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for i in 0..a.len() {
        diff |= a[i] ^ b[i];
    }
    diff == 0
}

fn hex_to_bytes(hex: &str) -> Result<Vec<u8>, ()> {
    if hex.len() % 2 != 0 {
        return Err(());
    }
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).map_err(|_| ()))
        .collect()
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pbkdf2_rfc6070_case1() {
        // RFC 6070 Test Vector 1 (iterations=1)
        let dk = pbkdf2_sha256(b"password", b"salt", 1, 20).unwrap();
        let hex: String = dk.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, "120fb6cffcf8b32c43e7225256c4f837a86548c9");
    }

    #[test]
    fn test_pbkdf2_rfc6070_case2() {
        // RFC 6070 Test Vector 2 (iterations=2)
        let dk = pbkdf2_sha256(b"password", b"salt", 2, 20).unwrap();
        let hex: String = dk.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, "ae4d0c95af6b46d32d0adff928f06dd02a303f8e");
    }

    #[test]
    fn test_pbkdf2_rfc6070_case3() {
        // RFC 6070 Test Vector 3 (iterations=4096)
        let dk = pbkdf2_sha256(b"password", b"salt", 4096, 20).unwrap();
        let hex: String = dk.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, "c5e478d59288c841aa530db6845c4c8d962893a0");
    }

    #[test]
    fn test_pbkdf2_long_dk() {
        // RFC 6070 Test Vector 5 (dk_len=25)
        let dk = pbkdf2_sha256(
            b"passwordPASSWORDpassword",
            b"saltSALTsaltSALTsaltSALTsaltSALTsalt",
            4096,
            25,
        ).unwrap();
        let hex: String = dk.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, "348c89dbcbd32b2f32d814b8116e84cf2b17347ebc1800181c");
    }

    #[test]
    fn test_pbkdf2_with_null_byte() {
        // RFC 6070 Test Vector 6
        let dk = pbkdf2_sha256(b"pass\0word", b"sa\0lt", 4096, 16).unwrap();
        let hex: String = dk.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, "89b69d0516f829893c696226650a8687");
    }

    #[test]
    fn test_invalid_iterations() {
        assert_eq!(pbkdf2_sha256(b"p", b"s", 0, 32), Err(Pbkdf2Error::InvalidIterations));
    }

    #[test]
    fn test_invalid_dk_len() {
        assert_eq!(pbkdf2_sha256(b"p", b"s", 1, 0), Err(Pbkdf2Error::InvalidDkLen));
    }

    #[test]
    fn test_password_hash_create_verify() {
        let ph = PasswordHash::create_with_params(b"my_password", 1000, 32, 16).unwrap();
        assert!(ph.verify(b"my_password").unwrap());
        assert!(!ph.verify(b"wrong_password").unwrap());
    }

    #[test]
    fn test_password_hash_string_roundtrip() {
        let ph = PasswordHash::create_with_params(b"test123", 500, 32, 16).unwrap();
        let s = ph.to_string_repr();
        assert!(s.starts_with("$pbkdf2-sha256$500$"));
        let parsed = PasswordHash::from_string_repr(&s).unwrap();
        assert_eq!(parsed.iterations, ph.iterations);
        assert_eq!(parsed.hash, ph.hash);
        assert_eq!(parsed.salt, ph.salt);
    }

    #[test]
    fn test_password_hash_empty_password() {
        assert_eq!(
            PasswordHash::create(b""),
            Err(Pbkdf2Error::EmptyPassword)
        );
    }

    #[test]
    fn test_salt_too_short() {
        assert_eq!(
            PasswordHash::create_with_params(b"pass", 1000, 32, 4),
            Err(Pbkdf2Error::SaltTooShort(4))
        );
    }

    #[test]
    fn test_generate_salt_length() {
        let salt = generate_salt(32);
        assert_eq!(salt.len(), 32);
    }

    #[test]
    fn test_generate_salt_uniqueness() {
        let s1 = generate_salt(16);
        let s2 = generate_salt(16);
        assert_ne!(s1, s2);
    }

    #[test]
    fn test_constant_time_compare() {
        assert!(constant_time_compare(b"abc", b"abc"));
        assert!(!constant_time_compare(b"abc", b"abd"));
        assert!(!constant_time_compare(b"ab", b"abc"));
    }

    #[test]
    fn test_invalid_hash_format() {
        assert!(PasswordHash::from_string_repr("invalid").is_err());
        assert!(PasswordHash::from_string_repr("$pbkdf2-sha512$100$aa$bb").is_err());
    }
}
