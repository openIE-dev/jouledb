//! SHA-256 cryptographic hash — pure-Rust implementation.
//!
//! Replaces Node.js `crypto.createHash('sha256')` and browser SubtleCrypto
//! with a zero-dependency SHA-256 that runs native + WASM.  Includes streaming
//! (update/finalize), hex digest, HMAC-SHA256, constant-time comparison, and
//! hash chaining.

use serde::{Deserialize, Serialize};

// ── Constants ──────────────────────────────────────────────────

/// SHA-256 initial hash values (first 32 bits of fractional parts of square roots of first 8 primes).
const H0: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// SHA-256 round constants (first 32 bits of fractional parts of cube roots of first 64 primes).
const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

// ── Errors ─────────────────────────────────────────────────────

/// SHA-256 domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Sha256Error {
    /// Hasher already finalized.
    AlreadyFinalized,
    /// Invalid hex string.
    InvalidHex(String),
    /// HMAC key error.
    InvalidKeyLength,
}

impl std::fmt::Display for Sha256Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyFinalized => write!(f, "hasher already finalized"),
            Self::InvalidHex(s) => write!(f, "invalid hex: {s}"),
            Self::InvalidKeyLength => write!(f, "invalid key length"),
        }
    }
}

impl std::error::Error for Sha256Error {}

// ── SHA-256 Digest ─────────────────────────────────────────────

/// A 32-byte SHA-256 digest.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Sha256Digest(pub [u8; 32]);

impl Sha256Digest {
    /// Return hex-encoded digest string.
    pub fn to_hex(&self) -> String {
        self.0.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// Parse from hex string.
    pub fn from_hex(hex: &str) -> Result<Self, Sha256Error> {
        if hex.len() != 64 {
            return Err(Sha256Error::InvalidHex(hex.to_string()));
        }
        let mut bytes = [0u8; 32];
        for i in 0..32 {
            bytes[i] = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16)
                .map_err(|_| Sha256Error::InvalidHex(hex.to_string()))?;
        }
        Ok(Self(bytes))
    }

    /// Constant-time comparison to prevent timing attacks.
    pub fn constant_time_eq(&self, other: &Sha256Digest) -> bool {
        constant_time_compare(&self.0, &other.0)
    }
}

impl std::fmt::Display for Sha256Digest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

// ── Streaming Hasher ───────────────────────────────────────────

/// Streaming SHA-256 hasher — call `update()` one or more times, then `finalize()`.
#[derive(Debug, Clone)]
pub struct Sha256Hasher {
    state: [u32; 8],
    buffer: Vec<u8>,
    total_len: u64,
    finalized: bool,
}

impl Default for Sha256Hasher {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha256Hasher {
    /// Create a new SHA-256 hasher.
    pub fn new() -> Self {
        Self {
            state: H0,
            buffer: Vec::with_capacity(64),
            total_len: 0,
            finalized: false,
        }
    }

    /// Feed data into the hasher.
    pub fn update(&mut self, data: &[u8]) -> Result<(), Sha256Error> {
        if self.finalized {
            return Err(Sha256Error::AlreadyFinalized);
        }
        self.total_len += data.len() as u64;
        self.buffer.extend_from_slice(data);

        // Process complete 64-byte blocks.
        while self.buffer.len() >= 64 {
            let block: [u8; 64] = self.buffer[..64].try_into().unwrap();
            process_block(&mut self.state, &block);
            self.buffer.drain(..64);
        }
        Ok(())
    }

    /// Finalize and return the digest.  Consumes the hasher logically (sets finalized flag).
    pub fn finalize(&mut self) -> Result<Sha256Digest, Sha256Error> {
        if self.finalized {
            return Err(Sha256Error::AlreadyFinalized);
        }
        self.finalized = true;

        // Padding.
        let bit_len = self.total_len * 8;
        self.buffer.push(0x80);
        while self.buffer.len() % 64 != 56 {
            self.buffer.push(0x00);
        }
        self.buffer.extend_from_slice(&bit_len.to_be_bytes());

        // Process remaining blocks.
        let chunks: Vec<[u8; 64]> = self
            .buffer
            .chunks_exact(64)
            .map(|c| c.try_into().unwrap())
            .collect();
        for block in &chunks {
            process_block(&mut self.state, block);
        }

        let mut digest = [0u8; 32];
        for (i, word) in self.state.iter().enumerate() {
            digest[i * 4..(i + 1) * 4].copy_from_slice(&word.to_be_bytes());
        }
        Ok(Sha256Digest(digest))
    }
}

// ── One-shot helpers ───────────────────────────────────────────

/// Compute SHA-256 of `data` in one call.
pub fn sha256(data: &[u8]) -> Sha256Digest {
    let mut h = Sha256Hasher::new();
    h.update(data).unwrap();
    h.finalize().unwrap()
}

/// Compute hex digest of `data`.
pub fn sha256_hex(data: &[u8]) -> String {
    sha256(data).to_hex()
}

// ── HMAC-SHA256 ────────────────────────────────────────────────

/// Compute HMAC-SHA256(key, message).
pub fn hmac_sha256(key: &[u8], message: &[u8]) -> Sha256Digest {
    // If key > 64 bytes, hash it.
    let key_hash;
    let k = if key.len() > 64 {
        key_hash = sha256(key);
        &key_hash.0[..]
    } else {
        key
    };

    // Pad key to 64 bytes.
    let mut k_padded = [0u8; 64];
    k_padded[..k.len()].copy_from_slice(k);

    // Inner and outer pads.
    let mut ipad = [0x36u8; 64];
    let mut opad = [0x5cu8; 64];
    for i in 0..64 {
        ipad[i] ^= k_padded[i];
        opad[i] ^= k_padded[i];
    }

    // inner = SHA256(ipad || message)
    let mut inner = Sha256Hasher::new();
    inner.update(&ipad).unwrap();
    inner.update(message).unwrap();
    let inner_digest = inner.finalize().unwrap();

    // outer = SHA256(opad || inner)
    let mut outer = Sha256Hasher::new();
    outer.update(&opad).unwrap();
    outer.update(&inner_digest.0).unwrap();
    outer.finalize().unwrap()
}

// ── Hash chaining ──────────────────────────────────────────────

/// Chain hashes: H(H(H(data) || data2) || data3) ...
pub fn hash_chain(items: &[&[u8]]) -> Sha256Digest {
    if items.is_empty() {
        return sha256(b"");
    }
    let mut current = sha256(items[0]);
    for item in &items[1..] {
        let mut h = Sha256Hasher::new();
        h.update(&current.0).unwrap();
        h.update(item).unwrap();
        current = h.finalize().unwrap();
    }
    current
}

// ── Internal functions ─────────────────────────────────────────

/// Process a single 64-byte block, updating state in-place.
fn process_block(state: &mut [u32; 8], block: &[u8; 64]) {
    // Prepare message schedule.
    let mut w = [0u32; 64];
    for i in 0..16 {
        w[i] = u32::from_be_bytes([
            block[i * 4],
            block[i * 4 + 1],
            block[i * 4 + 2],
            block[i * 4 + 3],
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

    // Working variables.
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;

    for i in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ ((!e) & g);
        let temp1 = h
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(K[i])
            .wrapping_add(w[i]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let temp2 = s0.wrapping_add(maj);

        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(temp1);
        d = c;
        c = b;
        b = a;
        a = temp1.wrapping_add(temp2);
    }

    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
    state[4] = state[4].wrapping_add(e);
    state[5] = state[5].wrapping_add(f);
    state[6] = state[6].wrapping_add(g);
    state[7] = state[7].wrapping_add(h);
}

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

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_string() {
        // NIST test vector: SHA-256("")
        let d = sha256(b"");
        assert_eq!(
            d.to_hex(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn test_abc() {
        // NIST test vector: SHA-256("abc")
        let d = sha256(b"abc");
        assert_eq!(
            d.to_hex(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn test_two_block_message() {
        // SHA-256("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")
        let msg = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
        let d = sha256(msg);
        assert_eq!(
            d.to_hex(),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn test_streaming() {
        let mut h = Sha256Hasher::new();
        h.update(b"abc").unwrap();
        let d = h.finalize().unwrap();
        assert_eq!(d, sha256(b"abc"));
    }

    #[test]
    fn test_streaming_multi_update() {
        let mut h = Sha256Hasher::new();
        h.update(b"a").unwrap();
        h.update(b"b").unwrap();
        h.update(b"c").unwrap();
        let d = h.finalize().unwrap();
        assert_eq!(d, sha256(b"abc"));
    }

    #[test]
    fn test_finalize_twice_errors() {
        let mut h = Sha256Hasher::new();
        h.update(b"test").unwrap();
        h.finalize().unwrap();
        assert_eq!(h.finalize(), Err(Sha256Error::AlreadyFinalized));
    }

    #[test]
    fn test_update_after_finalize_errors() {
        let mut h = Sha256Hasher::new();
        h.finalize().unwrap();
        assert_eq!(h.update(b"x"), Err(Sha256Error::AlreadyFinalized));
    }

    #[test]
    fn test_hex_roundtrip() {
        let d = sha256(b"hello");
        let hex = d.to_hex();
        let d2 = Sha256Digest::from_hex(&hex).unwrap();
        assert_eq!(d, d2);
    }

    #[test]
    fn test_invalid_hex() {
        assert!(Sha256Digest::from_hex("zz").is_err());
        assert!(Sha256Digest::from_hex("abcd").is_err());
    }

    #[test]
    fn test_constant_time_eq() {
        let d1 = sha256(b"hello");
        let d2 = sha256(b"hello");
        let d3 = sha256(b"world");
        assert!(d1.constant_time_eq(&d2));
        assert!(!d1.constant_time_eq(&d3));
    }

    #[test]
    fn test_hmac_sha256() {
        // RFC 4231 Test Case 2
        let key = b"Jefe";
        let data = b"what do ya want for nothing?";
        let mac = hmac_sha256(key, data);
        assert_eq!(
            mac.to_hex(),
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
    }

    #[test]
    fn test_hmac_sha256_long_key() {
        // Key longer than 64 bytes gets hashed first.
        let key = vec![0xAAu8; 131];
        let data = b"Test Using Larger Than Block-Size Key - Hash Key First";
        let mac = hmac_sha256(&key, data);
        assert_eq!(
            mac.to_hex(),
            "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54"
        );
    }

    #[test]
    fn test_hash_chain() {
        let chain = hash_chain(&[b"hello", b"world"]);
        // Manual: SHA256(SHA256("hello") || "world")
        let h1 = sha256(b"hello");
        let mut h = Sha256Hasher::new();
        h.update(&h1.0).unwrap();
        h.update(b"world").unwrap();
        let expected = h.finalize().unwrap();
        assert_eq!(chain, expected);
    }

    #[test]
    fn test_hash_chain_empty() {
        let chain = hash_chain(&[]);
        assert_eq!(chain, sha256(b""));
    }

    #[test]
    fn test_constant_time_compare_different_lengths() {
        assert!(!constant_time_compare(&[1, 2], &[1, 2, 3]));
    }

    #[test]
    fn test_large_message() {
        // SHA-256 of 1 million 'a' characters.
        let data = vec![b'a'; 1_000_000];
        let d = sha256(&data);
        assert_eq!(
            d.to_hex(),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }
}
