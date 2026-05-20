//! HMAC engine — keyed-hash message authentication with SHA-256.
//!
//! Full HMAC-SHA256 implementation with key padding, inner/outer hash,
//! timing-safe verification, key derivation (HKDF-like extract/expand),
//! streaming HMAC, and multi-message authentication.

// ── Inline SHA-256 ──────────────────────────────────────────────────────────

const SHA256_H0: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

const SHA256_K: [u32; 64] = [
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

fn sha256_process_block(state: &mut [u32; 8], block: &[u8; 64]) {
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
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    for i in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ ((!e) & g);
        let t1 = h
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(SHA256_K[i])
            .wrapping_add(w[i]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let t2 = s0.wrapping_add(maj);
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
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

fn sha256(data: &[u8]) -> [u8; 32] {
    let mut state = SHA256_H0;
    let total_len = data.len() as u64;
    let mut buf = data.to_vec();
    buf.push(0x80);
    while buf.len() % 64 != 56 {
        buf.push(0x00);
    }
    buf.extend_from_slice(&(total_len * 8).to_be_bytes());
    for chunk in buf.chunks_exact(64) {
        let block: [u8; 64] = chunk.try_into().unwrap();
        sha256_process_block(&mut state, &block);
    }
    let mut out = [0u8; 32];
    for (i, word) in state.iter().enumerate() {
        out[i * 4..(i + 1) * 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

/// Streaming SHA-256 hasher used internally by HMAC.
struct Sha256Streaming {
    state: [u32; 8],
    buffer: Vec<u8>,
    total_len: u64,
}

impl Sha256Streaming {
    fn new() -> Self {
        Self {
            state: SHA256_H0,
            buffer: Vec::with_capacity(64),
            total_len: 0,
        }
    }

    fn update(&mut self, data: &[u8]) {
        self.total_len += data.len() as u64;
        self.buffer.extend_from_slice(data);
        while self.buffer.len() >= 64 {
            let block: [u8; 64] = self.buffer[..64].try_into().unwrap();
            sha256_process_block(&mut self.state, &block);
            self.buffer.drain(..64);
        }
    }

    fn finalize(mut self) -> [u8; 32] {
        let bit_len = self.total_len * 8;
        self.buffer.push(0x80);
        while self.buffer.len() % 64 != 56 {
            self.buffer.push(0x00);
        }
        self.buffer.extend_from_slice(&bit_len.to_be_bytes());
        for chunk in self.buffer.chunks_exact(64) {
            let block: [u8; 64] = chunk.try_into().unwrap();
            sha256_process_block(&mut self.state, &block);
        }
        let mut out = [0u8; 32];
        for (i, word) in self.state.iter().enumerate() {
            out[i * 4..(i + 1) * 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }
}

// ── Errors ──────────────────────────────────────────────────────────────────

/// HMAC engine errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HmacError {
    /// Empty key provided.
    EmptyKey,
    /// HKDF expand: output too long.
    OutputTooLong { max: usize, requested: usize },
    /// HMAC already finalized.
    AlreadyFinalized,
    /// Invalid hex string.
    InvalidHex(String),
}

impl std::fmt::Display for HmacError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyKey => write!(f, "HMAC key must not be empty"),
            Self::OutputTooLong { max, requested } => {
                write!(f, "HKDF output too long: {requested} bytes (max {max})")
            }
            Self::AlreadyFinalized => write!(f, "HMAC already finalized"),
            Self::InvalidHex(s) => write!(f, "invalid hex: {s}"),
        }
    }
}

impl std::error::Error for HmacError {}

// ── Hex helpers ─────────────────────────────────────────────────────────────

fn bytes_to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn hex_to_bytes(hex: &str) -> Result<Vec<u8>, HmacError> {
    if hex.len() % 2 != 0 {
        return Err(HmacError::InvalidHex(hex.to_string()));
    }
    let mut out = Vec::with_capacity(hex.len() / 2);
    for i in (0..hex.len()).step_by(2) {
        let byte = u8::from_str_radix(&hex[i..i + 2], 16)
            .map_err(|_| HmacError::InvalidHex(hex.to_string()))?;
        out.push(byte);
    }
    Ok(out)
}

// ── Constant-time comparison ────────────────────────────────────────────────

/// Timing-safe byte comparison.
pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for i in 0..a.len() {
        diff |= a[i] ^ b[i];
    }
    diff == 0
}

// ── HMAC Tag ────────────────────────────────────────────────────────────────

/// A 32-byte HMAC-SHA256 tag.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HmacTag {
    bytes: [u8; 32],
}

impl HmacTag {
    /// Create from raw bytes.
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self { bytes }
    }

    /// Return the raw bytes.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.bytes
    }

    /// Hex-encoded tag.
    pub fn to_hex(&self) -> String {
        bytes_to_hex(&self.bytes)
    }

    /// Parse from hex string.
    pub fn from_hex(hex: &str) -> Result<Self, HmacError> {
        let bytes = hex_to_bytes(hex)?;
        if bytes.len() != 32 {
            return Err(HmacError::InvalidHex(hex.to_string()));
        }
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&bytes);
        Ok(Self { bytes: arr })
    }

    /// Constant-time comparison.
    pub fn verify(&self, other: &HmacTag) -> bool {
        constant_time_eq(&self.bytes, &other.bytes)
    }
}

impl std::fmt::Display for HmacTag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

// ── Key preparation ─────────────────────────────────────────────────────────

/// Prepare HMAC key: if > 64 bytes, hash it. Pad to 64 bytes.
fn prepare_key(key: &[u8]) -> [u8; 64] {
    let hashed;
    let k = if key.len() > 64 {
        hashed = sha256(key);
        &hashed[..]
    } else {
        key
    };
    let mut padded = [0u8; 64];
    padded[..k.len()].copy_from_slice(k);
    padded
}

// ── One-shot HMAC ───────────────────────────────────────────────────────────

/// Compute HMAC-SHA256(key, message) in one call.
pub fn hmac_sha256(key: &[u8], message: &[u8]) -> HmacTag {
    let k_padded = prepare_key(key);

    let mut ipad = [0x36u8; 64];
    let mut opad = [0x5cu8; 64];
    for i in 0..64 {
        ipad[i] ^= k_padded[i];
        opad[i] ^= k_padded[i];
    }

    // inner = SHA256(ipad || message)
    let mut inner = Sha256Streaming::new();
    inner.update(&ipad);
    inner.update(message);
    let inner_digest = inner.finalize();

    // outer = SHA256(opad || inner)
    let mut outer = Sha256Streaming::new();
    outer.update(&opad);
    outer.update(&inner_digest);
    HmacTag { bytes: outer.finalize() }
}

/// Verify an HMAC tag in constant time.
pub fn hmac_verify(key: &[u8], message: &[u8], tag: &HmacTag) -> bool {
    let computed = hmac_sha256(key, message);
    computed.verify(tag)
}

// ── Streaming HMAC ──────────────────────────────────────────────────────────

/// Streaming HMAC-SHA256 — call `update()` multiple times, then `finalize()`.
pub struct HmacStreaming {
    inner_hasher: Sha256Streaming,
    opad: [u8; 64],
    finalized: bool,
}

impl HmacStreaming {
    /// Create a new streaming HMAC with the given key.
    pub fn new(key: &[u8]) -> Self {
        let k_padded = prepare_key(key);

        let mut ipad = [0x36u8; 64];
        let mut opad = [0x5cu8; 64];
        for i in 0..64 {
            ipad[i] ^= k_padded[i];
            opad[i] ^= k_padded[i];
        }

        let mut inner_hasher = Sha256Streaming::new();
        inner_hasher.update(&ipad);

        Self {
            inner_hasher,
            opad,
            finalized: false,
        }
    }

    /// Feed data into the HMAC.
    pub fn update(&mut self, data: &[u8]) -> Result<(), HmacError> {
        if self.finalized {
            return Err(HmacError::AlreadyFinalized);
        }
        self.inner_hasher.update(data);
        Ok(())
    }

    /// Finalize and return the HMAC tag.
    pub fn finalize(mut self) -> Result<HmacTag, HmacError> {
        if self.finalized {
            return Err(HmacError::AlreadyFinalized);
        }
        self.finalized = true;

        let inner_digest = self.inner_hasher.finalize();

        let mut outer = Sha256Streaming::new();
        outer.update(&self.opad);
        outer.update(&inner_digest);
        Ok(HmacTag { bytes: outer.finalize() })
    }
}

// ── HKDF (Extract + Expand) ────────────────────────────────────────────────

/// HKDF-Extract: PRK = HMAC-SHA256(salt, IKM).
/// If salt is empty, uses a zero-filled 32-byte salt per RFC 5869.
pub fn hkdf_extract(salt: &[u8], ikm: &[u8]) -> HmacTag {
    let effective_salt = if salt.is_empty() {
        vec![0u8; 32]
    } else {
        salt.to_vec()
    };
    hmac_sha256(&effective_salt, ikm)
}

/// HKDF-Expand: derive `length` bytes of keying material from PRK and info.
/// Maximum output: 255 * 32 = 8160 bytes.
pub fn hkdf_expand(prk: &[u8], info: &[u8], length: usize) -> Result<Vec<u8>, HmacError> {
    let max_output = 255 * 32;
    if length > max_output {
        return Err(HmacError::OutputTooLong {
            max: max_output,
            requested: length,
        });
    }
    if length == 0 {
        return Ok(Vec::new());
    }

    let n = (length + 31) / 32; // ceil(length / 32)
    let mut okm = Vec::with_capacity(n * 32);
    let mut t_prev: Vec<u8> = Vec::new();

    for i in 1..=n {
        let mut input = Vec::new();
        input.extend_from_slice(&t_prev);
        input.extend_from_slice(info);
        input.push(i as u8);

        let t_i = hmac_sha256(prk, &input);
        t_prev = t_i.bytes.to_vec();
        okm.extend_from_slice(&t_prev);
    }

    okm.truncate(length);
    Ok(okm)
}

/// Full HKDF: extract then expand.
pub fn hkdf(
    salt: &[u8],
    ikm: &[u8],
    info: &[u8],
    length: usize,
) -> Result<Vec<u8>, HmacError> {
    let prk = hkdf_extract(salt, ikm);
    hkdf_expand(prk.as_bytes(), info, length)
}

// ── Multi-message HMAC ──────────────────────────────────────────────────────

/// Compute HMAC over multiple message fragments concatenated logically.
pub fn hmac_sha256_multi(key: &[u8], parts: &[&[u8]]) -> HmacTag {
    let k_padded = prepare_key(key);

    let mut ipad = [0x36u8; 64];
    let mut opad = [0x5cu8; 64];
    for i in 0..64 {
        ipad[i] ^= k_padded[i];
        opad[i] ^= k_padded[i];
    }

    let mut inner = Sha256Streaming::new();
    inner.update(&ipad);
    for part in parts {
        inner.update(part);
    }
    let inner_digest = inner.finalize();

    let mut outer = Sha256Streaming::new();
    outer.update(&opad);
    outer.update(&inner_digest);
    HmacTag { bytes: outer.finalize() }
}

// ── Key Derivation Helper ───────────────────────────────────────────────────

/// Derive a fixed-length key using HKDF with optional context label.
pub fn derive_key(
    master_key: &[u8],
    salt: &[u8],
    context: &[u8],
    key_len: usize,
) -> Result<Vec<u8>, HmacError> {
    hkdf(salt, master_key, context, key_len)
}

/// Derive multiple keys from a single master key, each with a distinct label.
pub fn derive_keys(
    master_key: &[u8],
    salt: &[u8],
    labels: &[&[u8]],
    key_len: usize,
) -> Result<Vec<Vec<u8>>, HmacError> {
    let mut keys = Vec::with_capacity(labels.len());
    for label in labels {
        keys.push(derive_key(master_key, salt, label, key_len)?);
    }
    Ok(keys)
}

// ── Truncated HMAC ──────────────────────────────────────────────────────────

/// Compute HMAC-SHA256 and truncate to `len` bytes (min 10 for security).
pub fn hmac_sha256_truncated(key: &[u8], message: &[u8], len: usize) -> Vec<u8> {
    let tag = hmac_sha256(key, message);
    let effective_len = len.min(32).max(10);
    tag.bytes[..effective_len].to_vec()
}

/// Verify a truncated HMAC.
pub fn hmac_verify_truncated(
    key: &[u8],
    message: &[u8],
    truncated_tag: &[u8],
) -> bool {
    let full_tag = hmac_sha256(key, message);
    let compare_len = truncated_tag.len().min(32);
    constant_time_eq(&full_tag.bytes[..compare_len], &truncated_tag[..compare_len])
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sha256_empty() {
        let d = sha256(b"");
        assert_eq!(
            bytes_to_hex(&d),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn test_sha256_abc() {
        let d = sha256(b"abc");
        assert_eq!(
            bytes_to_hex(&d),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn test_hmac_rfc4231_case2() {
        // RFC 4231 Test Case 2: key = "Jefe", data = "what do ya want for nothing?"
        let tag = hmac_sha256(b"Jefe", b"what do ya want for nothing?");
        assert_eq!(
            tag.to_hex(),
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
    }

    #[test]
    fn test_hmac_rfc4231_case1() {
        // RFC 4231 Test Case 1: key = 20 bytes of 0x0b
        let key = vec![0x0bu8; 20];
        let data = b"Hi There";
        let tag = hmac_sha256(&key, data);
        assert_eq!(
            tag.to_hex(),
            "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"
        );
    }

    #[test]
    fn test_hmac_long_key() {
        // Key longer than 64 bytes gets hashed first (RFC 4231 Test Case 6)
        let key = vec![0xAAu8; 131];
        let data = b"Test Using Larger Than Block-Size Key - Hash Key First";
        let tag = hmac_sha256(&key, data);
        assert_eq!(
            tag.to_hex(),
            "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54"
        );
    }

    #[test]
    fn test_hmac_verify_valid() {
        let key = b"secret-key";
        let msg = b"important message";
        let tag = hmac_sha256(key, msg);
        assert!(hmac_verify(key, msg, &tag));
    }

    #[test]
    fn test_hmac_verify_invalid() {
        let key = b"secret-key";
        let msg = b"important message";
        let tag = hmac_sha256(key, msg);
        assert!(!hmac_verify(b"wrong-key", msg, &tag));
        assert!(!hmac_verify(key, b"tampered message", &tag));
    }

    #[test]
    fn test_streaming_hmac() {
        let key = b"stream-key";
        let mut h = HmacStreaming::new(key);
        h.update(b"hello ").unwrap();
        h.update(b"world").unwrap();
        let tag = h.finalize().unwrap();

        let expected = hmac_sha256(key, b"hello world");
        assert!(tag.verify(&expected));
    }

    #[test]
    fn test_streaming_hmac_single_update() {
        let key = b"test-key";
        let mut h = HmacStreaming::new(key);
        h.update(b"data").unwrap();
        let tag = h.finalize().unwrap();
        assert!(tag.verify(&hmac_sha256(key, b"data")));
    }

    #[test]
    fn test_hmac_tag_hex_roundtrip() {
        let tag = hmac_sha256(b"key", b"msg");
        let hex = tag.to_hex();
        let parsed = HmacTag::from_hex(&hex).unwrap();
        assert_eq!(tag, parsed);
    }

    #[test]
    fn test_hmac_tag_from_hex_invalid() {
        assert!(HmacTag::from_hex("zz").is_err());
        assert!(HmacTag::from_hex("abc").is_err()); // odd length
    }

    #[test]
    fn test_hkdf_extract() {
        let salt = b"salt-value";
        let ikm = b"input keying material";
        let prk = hkdf_extract(salt, ikm);
        // PRK should be 32 bytes (HMAC-SHA256 output)
        assert_eq!(prk.as_bytes().len(), 32);
    }

    #[test]
    fn test_hkdf_extract_empty_salt() {
        // Empty salt defaults to 32 zero bytes
        let prk1 = hkdf_extract(b"", b"ikm");
        let prk2 = hkdf_extract(&[0u8; 32], b"ikm");
        assert_eq!(prk1, prk2);
    }

    #[test]
    fn test_hkdf_expand_basic() {
        let prk = hmac_sha256(b"salt", b"ikm");
        let okm = hkdf_expand(prk.as_bytes(), b"info", 42).unwrap();
        assert_eq!(okm.len(), 42);
    }

    #[test]
    fn test_hkdf_expand_too_long() {
        let prk = hmac_sha256(b"salt", b"ikm");
        let result = hkdf_expand(prk.as_bytes(), b"info", 255 * 32 + 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_hkdf_expand_zero_length() {
        let prk = hmac_sha256(b"salt", b"ikm");
        let okm = hkdf_expand(prk.as_bytes(), b"info", 0).unwrap();
        assert!(okm.is_empty());
    }

    #[test]
    fn test_hkdf_full() {
        let key1 = hkdf(b"salt", b"master-secret", b"enc-key", 32).unwrap();
        let key2 = hkdf(b"salt", b"master-secret", b"mac-key", 32).unwrap();
        assert_eq!(key1.len(), 32);
        assert_eq!(key2.len(), 32);
        // Different info should produce different keys
        assert_ne!(key1, key2);
    }

    #[test]
    fn test_hkdf_deterministic() {
        let k1 = hkdf(b"s", b"ikm", b"ctx", 16).unwrap();
        let k2 = hkdf(b"s", b"ikm", b"ctx", 16).unwrap();
        assert_eq!(k1, k2);
    }

    #[test]
    fn test_hmac_multi_message() {
        let key = b"multi-key";
        let tag1 = hmac_sha256_multi(key, &[b"hello ", b"world"]);
        let tag2 = hmac_sha256(key, b"hello world");
        assert!(tag1.verify(&tag2));
    }

    #[test]
    fn test_derive_key() {
        let key = derive_key(b"master", b"salt", b"context", 32).unwrap();
        assert_eq!(key.len(), 32);
    }

    #[test]
    fn test_derive_keys_different_labels() {
        let keys =
            derive_keys(b"master", b"salt", &[b"label-a", b"label-b", b"label-c"], 32)
                .unwrap();
        assert_eq!(keys.len(), 3);
        assert_ne!(keys[0], keys[1]);
        assert_ne!(keys[1], keys[2]);
        assert_ne!(keys[0], keys[2]);
    }

    #[test]
    fn test_truncated_hmac() {
        let key = b"trunc-key";
        let msg = b"message";
        let trunc = hmac_sha256_truncated(key, msg, 16);
        assert_eq!(trunc.len(), 16);

        let full = hmac_sha256(key, msg);
        assert_eq!(&full.as_bytes()[..16], &trunc[..]);
    }

    #[test]
    fn test_truncated_hmac_min_length() {
        // Minimum enforced length is 10
        let trunc = hmac_sha256_truncated(b"key", b"msg", 4);
        assert_eq!(trunc.len(), 10);
    }

    #[test]
    fn test_truncated_hmac_verify() {
        let key = b"key";
        let msg = b"verify this";
        let trunc = hmac_sha256_truncated(key, msg, 16);
        assert!(hmac_verify_truncated(key, msg, &trunc));
        // Tampered message should fail
        assert!(!hmac_verify_truncated(key, b"tampered", &trunc));
    }

    #[test]
    fn test_constant_time_eq_same() {
        let a = [1u8, 2, 3, 4];
        assert!(constant_time_eq(&a, &a));
    }

    #[test]
    fn test_constant_time_eq_different_lengths() {
        assert!(!constant_time_eq(&[1, 2], &[1, 2, 3]));
    }

    #[test]
    fn test_constant_time_eq_different_values() {
        assert!(!constant_time_eq(&[1, 2, 3], &[1, 2, 4]));
    }

    #[test]
    fn test_hmac_empty_message() {
        let tag = hmac_sha256(b"key", b"");
        assert_eq!(tag.as_bytes().len(), 32);
        assert!(hmac_verify(b"key", b"", &tag));
    }
}
