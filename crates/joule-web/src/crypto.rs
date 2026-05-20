//! Cryptographic primitives in pure Rust.
//!
//! Replaces `crypto-js` with zero external crypto dependencies.
//! Implements SHA-256, MD5, HMAC-SHA256, base64, hex, and password hashing.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Errors from cryptographic operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CryptoError {
    /// Invalid hexadecimal input.
    InvalidHex,
    /// Invalid base64 input.
    InvalidBase64,
    /// Invalid length for operation.
    InvalidLength,
}

impl fmt::Display for CryptoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidHex => write!(f, "invalid hex string"),
            Self::InvalidBase64 => write!(f, "invalid base64 string"),
            Self::InvalidLength => write!(f, "invalid length"),
        }
    }
}

impl std::error::Error for CryptoError {}

// ── SHA-256 ─────────────────────────────────────────────────────

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

const SHA256_INIT: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c,
    0x1f83d9ab, 0x5be0cd19,
];

fn sha256_pad(input: &[u8]) -> Vec<u8> {
    let len = input.len();
    let bit_len = (len as u64) * 8;
    let mut padded = input.to_vec();
    padded.push(0x80);
    while (padded.len() % 64) != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());
    padded
}

/// Compute SHA-256 hash of input bytes.
pub fn sha256(input: &[u8]) -> [u8; 32] {
    let padded = sha256_pad(input);
    let mut h = SHA256_INIT;

    for chunk in padded.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                chunk[i * 4],
                chunk[i * 4 + 1],
                chunk[i * 4 + 2],
                chunk[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(SHA256_K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut result = [0u8; 32];
    for (i, val) in h.iter().enumerate() {
        result[i * 4..i * 4 + 4].copy_from_slice(&val.to_be_bytes());
    }
    result
}

/// SHA-256 hash as lowercase hex string.
pub fn sha256_hex(input: &[u8]) -> String {
    hex_encode(&sha256(input))
}

// ── MD5 ─────────────────────────────────────────────────────────

const MD5_S: [u32; 64] = [
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5,
    9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10,
    15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
];

const MD5_K: [u32; 64] = [
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a,
    0xa8304613, 0xfd469501, 0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
    0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340,
    0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8,
    0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
    0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, 0x289b7ec6, 0xeaa127fa,
    0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92,
    0xffeff47d, 0x85845dd1, 0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
    0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391,
];

/// Compute MD5 hash (for compatibility, not security).
pub fn md5(input: &[u8]) -> [u8; 16] {
    let len = input.len();
    let bit_len = (len as u64) * 8;
    let mut data = input.to_vec();
    data.push(0x80);
    while (data.len() % 64) != 56 {
        data.push(0);
    }
    data.extend_from_slice(&bit_len.to_le_bytes());

    let mut a0: u32 = 0x67452301;
    let mut b0: u32 = 0xefcdab89;
    let mut c0: u32 = 0x98badcfe;
    let mut d0: u32 = 0x10325476;

    for chunk in data.chunks(64) {
        let mut m = [0u32; 16];
        for i in 0..16 {
            m[i] = u32::from_le_bytes([
                chunk[i * 4],
                chunk[i * 4 + 1],
                chunk[i * 4 + 2],
                chunk[i * 4 + 3],
            ]);
        }

        let (mut a, mut b, mut c, mut d) = (a0, b0, c0, d0);

        for i in 0..64 {
            let (f, g) = match i {
                0..=15 => ((b & c) | ((!b) & d), i),
                16..=31 => ((d & b) | ((!d) & c), (5 * i + 1) % 16),
                32..=47 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | (!d)), (7 * i) % 16),
            };

            let f = f.wrapping_add(a).wrapping_add(MD5_K[i]).wrapping_add(m[g]);
            a = d;
            d = c;
            c = b;
            b = b.wrapping_add(f.rotate_left(MD5_S[i]));
        }

        a0 = a0.wrapping_add(a);
        b0 = b0.wrapping_add(b);
        c0 = c0.wrapping_add(c);
        d0 = d0.wrapping_add(d);
    }

    let mut result = [0u8; 16];
    result[0..4].copy_from_slice(&a0.to_le_bytes());
    result[4..8].copy_from_slice(&b0.to_le_bytes());
    result[8..12].copy_from_slice(&c0.to_le_bytes());
    result[12..16].copy_from_slice(&d0.to_le_bytes());
    result
}

/// MD5 hash as lowercase hex string.
pub fn md5_hex(input: &[u8]) -> String {
    hex_encode(&md5(input))
}

// ── HMAC-SHA256 ─────────────────────────────────────────────────

/// Compute HMAC-SHA256.
pub fn hmac_sha256(key: &[u8], message: &[u8]) -> [u8; 32] {
    let block_size = 64;

    // If key is longer than block size, hash it first
    let key_prime = if key.len() > block_size {
        sha256(key).to_vec()
    } else {
        key.to_vec()
    };

    // Pad key to block size
    let mut key_pad = vec![0u8; block_size];
    key_pad[..key_prime.len()].copy_from_slice(&key_prime);

    // Inner and outer pads
    let mut i_key_pad = vec![0u8; block_size];
    let mut o_key_pad = vec![0u8; block_size];
    for i in 0..block_size {
        i_key_pad[i] = key_pad[i] ^ 0x36;
        o_key_pad[i] = key_pad[i] ^ 0x5c;
    }

    // Inner hash
    let mut inner = i_key_pad;
    inner.extend_from_slice(message);
    let inner_hash = sha256(&inner);

    // Outer hash
    let mut outer = o_key_pad;
    outer.extend_from_slice(&inner_hash);
    sha256(&outer)
}

/// HMAC-SHA256 as lowercase hex string.
pub fn hmac_sha256_hex(key: &[u8], message: &[u8]) -> String {
    hex_encode(&hmac_sha256(key, message))
}

// ── Hex ─────────────────────────────────────────────────────────

/// Encode bytes as lowercase hex string.
pub fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(HEX_CHARS[(b >> 4) as usize]);
        s.push(HEX_CHARS[(b & 0x0f) as usize]);
    }
    s
}

const HEX_CHARS: [char; 16] = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f',
];

/// Decode a hex string into bytes.
pub fn hex_decode(hex: &str) -> Result<Vec<u8>, CryptoError> {
    if hex.len() % 2 != 0 {
        return Err(CryptoError::InvalidHex);
    }
    let mut bytes = Vec::with_capacity(hex.len() / 2);
    let chars: Vec<char> = hex.chars().collect();
    for pair in chars.chunks(2) {
        let hi = hex_val(pair[0]).ok_or(CryptoError::InvalidHex)?;
        let lo = hex_val(pair[1]).ok_or(CryptoError::InvalidHex)?;
        bytes.push((hi << 4) | lo);
    }
    Ok(bytes)
}

fn hex_val(c: char) -> Option<u8> {
    match c {
        '0'..='9' => Some(c as u8 - b'0'),
        'a'..='f' => Some(c as u8 - b'a' + 10),
        'A'..='F' => Some(c as u8 - b'A' + 10),
        _ => None,
    }
}

// ── Base64 ──────────────────────────────────────────────────────

const B64_CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Encode bytes as standard base64 with padding.
pub fn base64_encode(input: &[u8]) -> String {
    let mut out = String::new();
    let mut i = 0;
    while i + 2 < input.len() {
        let n = ((input[i] as u32) << 16) | ((input[i + 1] as u32) << 8) | (input[i + 2] as u32);
        out.push(B64_CHARS[((n >> 18) & 0x3f) as usize] as char);
        out.push(B64_CHARS[((n >> 12) & 0x3f) as usize] as char);
        out.push(B64_CHARS[((n >> 6) & 0x3f) as usize] as char);
        out.push(B64_CHARS[(n & 0x3f) as usize] as char);
        i += 3;
    }
    let remaining = input.len() - i;
    if remaining == 2 {
        let n = ((input[i] as u32) << 16) | ((input[i + 1] as u32) << 8);
        out.push(B64_CHARS[((n >> 18) & 0x3f) as usize] as char);
        out.push(B64_CHARS[((n >> 12) & 0x3f) as usize] as char);
        out.push(B64_CHARS[((n >> 6) & 0x3f) as usize] as char);
        out.push('=');
    } else if remaining == 1 {
        let n = (input[i] as u32) << 16;
        out.push(B64_CHARS[((n >> 18) & 0x3f) as usize] as char);
        out.push(B64_CHARS[((n >> 12) & 0x3f) as usize] as char);
        out.push('=');
        out.push('=');
    }
    out
}

/// Decode standard base64 (with or without padding).
pub fn base64_decode(input: &str) -> Result<Vec<u8>, CryptoError> {
    let input = input.trim_end_matches('=');
    let mut out = Vec::new();
    let mut buf: u32 = 0;
    let mut bits: u32 = 0;

    for c in input.chars() {
        let val = b64_val(c).ok_or(CryptoError::InvalidBase64)?;
        buf = (buf << 6) | (val as u32);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buf >> bits) as u8);
            buf &= (1 << bits) - 1;
        }
    }
    Ok(out)
}

fn b64_val(c: char) -> Option<u8> {
    match c {
        'A'..='Z' => Some(c as u8 - b'A'),
        'a'..='z' => Some(c as u8 - b'a' + 26),
        '0'..='9' => Some(c as u8 - b'0' + 52),
        '+' => Some(62),
        '/' => Some(63),
        _ => None,
    }
}

// ── Constant-time comparison ────────────────────────────────────

/// Timing-safe byte comparison.
pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

// ── Random (deterministic PRNG — NOT cryptographically secure) ──

/// Simple xorshift64 PRNG state.
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x1234_5678_9abc_def0 } else { seed },
        }
    }

    fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
}

use std::sync::atomic::{AtomicU64, Ordering};
static PRNG_COUNTER: AtomicU64 = AtomicU64::new(0x9e37_79b9_7f4a_7c15);

fn get_prng() -> Xorshift64 {
    let seed = PRNG_COUNTER.fetch_add(0x6a09_e667_bb67_ae85, Ordering::Relaxed);
    Xorshift64::new(seed)
}

/// Generate pseudo-random bytes (NOT cryptographically secure).
pub fn random_bytes(len: usize) -> Vec<u8> {
    let mut rng = get_prng();
    let mut out = Vec::with_capacity(len);
    while out.len() < len {
        let val = rng.next();
        for &b in val.to_le_bytes().iter() {
            if out.len() < len {
                out.push(b);
            }
        }
    }
    out
}

/// Generate random hex string of `len` bytes (2*len hex chars).
pub fn random_hex(len: usize) -> String {
    hex_encode(&random_bytes(len))
}

const URL_SAFE_CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

/// Generate a URL-safe random token of the given byte length.
pub fn generate_token(len: usize) -> String {
    let bytes = random_bytes(len);
    bytes
        .iter()
        .map(|b| URL_SAFE_CHARS[(*b as usize) % URL_SAFE_CHARS.len()] as char)
        .collect()
}

// ── Password hashing ────────────────────────────────────────────

/// Simple PBKDF2-like password hash (1000 iterations of HMAC-SHA256).
/// Returns hex-encoded hash. Not for production security.
pub fn hash_password(password: &str, salt: &[u8]) -> String {
    let mut result = hmac_sha256(password.as_bytes(), salt);
    for _ in 1..1000 {
        result = hmac_sha256(password.as_bytes(), &result);
    }
    hex_encode(&result)
}

/// Verify a password against a previously computed hash.
pub fn verify_password(password: &str, salt: &[u8], hash: &str) -> bool {
    let computed = hash_password(password, salt);
    // Use constant-time comparison
    if let Ok(hash_bytes) = hex_decode(hash) {
        if let Ok(computed_bytes) = hex_decode(&computed) {
            return constant_time_eq(&hash_bytes, &computed_bytes);
        }
    }
    false
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_known_vector_abc() {
        let hash = sha256_hex(b"abc");
        assert_eq!(
            hash,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn sha256_empty() {
        let hash = sha256_hex(b"");
        assert_eq!(
            hash,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn md5_known_vector() {
        // MD5("abc") = 900150983cd24fb0d6963f7d28e17f72
        let hash = md5_hex(b"abc");
        assert_eq!(hash, "900150983cd24fb0d6963f7d28e17f72");
    }

    #[test]
    fn hmac_sha256_rfc4231_test1() {
        // RFC 4231 Test Case 1
        let key = [0x0b; 20];
        let data = b"Hi There";
        let result = hmac_sha256_hex(&key, data);
        assert_eq!(
            result,
            "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"
        );
    }

    #[test]
    fn hex_roundtrip() {
        let original = b"hello world";
        let encoded = hex_encode(original);
        let decoded = hex_decode(&encoded).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn base64_roundtrip() {
        let original = b"hello world, this is a test!";
        let encoded = base64_encode(original);
        let decoded = base64_decode(&encoded).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn constant_time_eq_equal() {
        assert!(constant_time_eq(b"hello", b"hello"));
    }

    #[test]
    fn constant_time_eq_different() {
        assert!(!constant_time_eq(b"hello", b"world"));
        assert!(!constant_time_eq(b"hello", b"hell"));
    }

    #[test]
    fn random_bytes_correct_length() {
        assert_eq!(random_bytes(0).len(), 0);
        assert_eq!(random_bytes(16).len(), 16);
        assert_eq!(random_bytes(100).len(), 100);
    }

    #[test]
    fn generate_token_url_safe() {
        let token = generate_token(32);
        assert_eq!(token.len(), 32);
        for c in token.chars() {
            assert!(
                c.is_ascii_alphanumeric() || c == '-' || c == '_',
                "non-URL-safe char: {c}"
            );
        }
    }

    #[test]
    fn password_hash_verify_roundtrip() {
        let salt = b"random_salt_here";
        let hash = hash_password("mypassword", salt);
        assert!(verify_password("mypassword", salt, &hash));
        assert!(!verify_password("wrong", salt, &hash));
    }

    #[test]
    fn hex_decode_invalid() {
        assert_eq!(hex_decode("zz"), Err(CryptoError::InvalidHex));
        assert_eq!(hex_decode("abc"), Err(CryptoError::InvalidHex)); // odd length
    }

    #[test]
    fn base64_decode_invalid() {
        assert!(base64_decode("!!!").is_err());
    }
}
