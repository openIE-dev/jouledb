//! SHA-512 cryptographic hash — pure-Rust implementation.
//!
//! Replaces Node.js `crypto.createHash('sha512')` with a zero-dependency SHA-512
//! that runs native + WASM.  Includes SHA-384, SHA-512/256, streaming, hex digest,
//! HMAC-SHA512, and constant-time comparison.

use serde::{Deserialize, Serialize};

// ── Constants ──────────────────────────────────────────────────

/// SHA-512 initial hash values.
const H512: [u64; 8] = [
    0x6a09e667f3bcc908, 0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
    0x510e527fade682d1, 0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
];

/// SHA-384 initial hash values.
const H384: [u64; 8] = [
    0xcbbb9d5dc1059ed8, 0x629a292a367cd507,
    0x9159015a3070dd17, 0x152fecd8f70e5939,
    0x67332667ffc00b31, 0x8eb44a8768581511,
    0xdb0c2e0d64f98fa7, 0x47b5481dbefa4fa4,
];

/// SHA-512/256 initial hash values.
const H512_256: [u64; 8] = [
    0x22312194FC2BF72C, 0x9F555FA3C84C64C2,
    0x2393B86B6F53B151, 0x963877195940EABD,
    0x96283EE2A88EFFE3, 0xBE5E1E2553863992,
    0x2B0199FC2C85B8AA, 0x0EB72DDC81C52CA2,
];

/// SHA-512 round constants.
const K: [u64; 80] = [
    0x428a2f98d728ae22, 0x7137449123ef65cd, 0xb5c0fbcfec4d3b2f, 0xe9b5dba58189dbbc,
    0x3956c25bf348b538, 0x59f111f1b605d019, 0x923f82a4af194f9b, 0xab1c5ed5da6d8118,
    0xd807aa98a3030242, 0x12835b0145706fbe, 0x243185be4ee4b28c, 0x550c7dc3d5ffb4e2,
    0x72be5d74f27b896f, 0x80deb1fe3b1696b1, 0x9bdc06a725c71235, 0xc19bf174cf692694,
    0xe49b69c19ef14ad2, 0xefbe4786384f25e3, 0x0fc19dc68b8cd5b5, 0x240ca1cc77ac9c65,
    0x2de92c6f592b0275, 0x4a7484aa6ea6e483, 0x5cb0a9dcbd41fbd4, 0x76f988da831153b5,
    0x983e5152ee66dfab, 0xa831c66d2db43210, 0xb00327c898fb213f, 0xbf597fc7beef0ee4,
    0xc6e00bf33da88fc2, 0xd5a79147930aa725, 0x06ca6351e003826f, 0x142929670a0e6e70,
    0x27b70a8546d22ffc, 0x2e1b21385c26c926, 0x4d2c6dfc5ac42aed, 0x53380d139d95b3df,
    0x650a73548baf63de, 0x766a0abb3c77b2a8, 0x81c2c92e47edaee6, 0x92722c851482353b,
    0xa2bfe8a14cf10364, 0xa81a664bbc423001, 0xc24b8b70d0f89791, 0xc76c51a30654be30,
    0xd192e819d6ef5218, 0xd69906245565a910, 0xf40e35855771202a, 0x106aa07032bbd1b8,
    0x19a4c116b8d2d0c8, 0x1e376c085141ab53, 0x2748774cdf8eeb99, 0x34b0bcb5e19b48a8,
    0x391c0cb3c5c95a63, 0x4ed8aa4ae3418acb, 0x5b9cca4f7763e373, 0x682e6ff3d6b2b8a3,
    0x748f82ee5defb2fc, 0x78a5636f43172f60, 0x84c87814a1f0ab72, 0x8cc702081a6439ec,
    0x90befffa23631e28, 0xa4506cebde82bde9, 0xbef9a3f7b2c67915, 0xc67178f2e372532b,
    0xca273eceea26619c, 0xd186b8c721c0c207, 0xeada7dd6cde0eb1e, 0xf57d4f7fee6ed178,
    0x06f067aa72176fba, 0x0a637dc5a2c898a6, 0x113f9804bef90dae, 0x1b710b35131c471b,
    0x28db77f523047d84, 0x32caab7b40c72493, 0x3c9ebe0a15c9bebc, 0x431d67c49c100d4c,
    0x4cc5d4becb3e42b6, 0x597f299cfc657e2a, 0x5fcb6fab3ad6faec, 0x6c44198c4a475817,
];

// ── Errors ─────────────────────────────────────────────────────

/// SHA-512 domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Sha512Error {
    AlreadyFinalized,
    InvalidHex(String),
}

impl std::fmt::Display for Sha512Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyFinalized => write!(f, "hasher already finalized"),
            Self::InvalidHex(s) => write!(f, "invalid hex: {s}"),
        }
    }
}

impl std::error::Error for Sha512Error {}

// ── Variants ───────────────────────────────────────────────────

/// Which SHA-512 variant to compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sha512Variant {
    Sha512,
    Sha384,
    Sha512_256,
}

// ── Digest types ───────────────────────────────────────────────

/// A 64-byte SHA-512 digest.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Sha512Digest(pub [u8; 64]);

impl Sha512Digest {
    pub fn to_hex(&self) -> String {
        self.0.iter().map(|b| format!("{b:02x}")).collect()
    }

    pub fn from_hex(hex: &str) -> Result<Self, Sha512Error> {
        if hex.len() != 128 {
            return Err(Sha512Error::InvalidHex(hex.to_string()));
        }
        let mut bytes = [0u8; 64];
        for i in 0..64 {
            bytes[i] = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16)
                .map_err(|_| Sha512Error::InvalidHex(hex.to_string()))?;
        }
        Ok(Self(bytes))
    }

    pub fn constant_time_eq(&self, other: &Self) -> bool {
        constant_time_compare(&self.0, &other.0)
    }
}

impl std::fmt::Display for Sha512Digest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

/// A 48-byte SHA-384 digest.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Sha384Digest(pub [u8; 48]);

impl Sha384Digest {
    pub fn to_hex(&self) -> String {
        self.0.iter().map(|b| format!("{b:02x}")).collect()
    }
}

impl std::fmt::Display for Sha384Digest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

/// A 32-byte SHA-512/256 digest.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Sha512_256Digest(pub [u8; 32]);

impl Sha512_256Digest {
    pub fn to_hex(&self) -> String {
        self.0.iter().map(|b| format!("{b:02x}")).collect()
    }
}

impl std::fmt::Display for Sha512_256Digest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

// ── Streaming Hasher ───────────────────────────────────────────

/// Streaming SHA-512 hasher (also used for SHA-384, SHA-512/256).
#[derive(Debug, Clone)]
pub struct Sha512Hasher {
    state: [u64; 8],
    buffer: Vec<u8>,
    total_len: u128,
    variant: Sha512Variant,
    finalized: bool,
}

impl Sha512Hasher {
    /// Create a new hasher for the given variant.
    pub fn new(variant: Sha512Variant) -> Self {
        let state = match variant {
            Sha512Variant::Sha512 => H512,
            Sha512Variant::Sha384 => H384,
            Sha512Variant::Sha512_256 => H512_256,
        };
        Self {
            state,
            buffer: Vec::with_capacity(128),
            total_len: 0,
            variant,
            finalized: false,
        }
    }

    /// Feed data into the hasher.
    pub fn update(&mut self, data: &[u8]) -> Result<(), Sha512Error> {
        if self.finalized {
            return Err(Sha512Error::AlreadyFinalized);
        }
        self.total_len += data.len() as u128;
        self.buffer.extend_from_slice(data);
        while self.buffer.len() >= 128 {
            let block: [u8; 128] = self.buffer[..128].try_into().unwrap();
            process_block(&mut self.state, &block);
            self.buffer.drain(..128);
        }
        Ok(())
    }

    /// Finalize and return raw 64-byte hash state.
    fn finalize_raw(&mut self) -> Result<[u8; 64], Sha512Error> {
        if self.finalized {
            return Err(Sha512Error::AlreadyFinalized);
        }
        self.finalized = true;

        let bit_len = self.total_len * 8;
        self.buffer.push(0x80);
        while self.buffer.len() % 128 != 112 {
            self.buffer.push(0x00);
        }
        self.buffer.extend_from_slice(&bit_len.to_be_bytes());

        let chunks: Vec<[u8; 128]> = self
            .buffer
            .chunks_exact(128)
            .map(|c| c.try_into().unwrap())
            .collect();
        for block in &chunks {
            process_block(&mut self.state, block);
        }

        let mut digest = [0u8; 64];
        for (i, word) in self.state.iter().enumerate() {
            digest[i * 8..(i + 1) * 8].copy_from_slice(&word.to_be_bytes());
        }
        Ok(digest)
    }

    /// Finalize as SHA-512.
    pub fn finalize_512(&mut self) -> Result<Sha512Digest, Sha512Error> {
        let raw = self.finalize_raw()?;
        Ok(Sha512Digest(raw))
    }

    /// Finalize as SHA-384 (first 48 bytes).
    pub fn finalize_384(&mut self) -> Result<Sha384Digest, Sha512Error> {
        let raw = self.finalize_raw()?;
        let mut out = [0u8; 48];
        out.copy_from_slice(&raw[..48]);
        Ok(Sha384Digest(out))
    }

    /// Finalize as SHA-512/256 (first 32 bytes).
    pub fn finalize_512_256(&mut self) -> Result<Sha512_256Digest, Sha512Error> {
        let raw = self.finalize_raw()?;
        let mut out = [0u8; 32];
        out.copy_from_slice(&raw[..32]);
        Ok(Sha512_256Digest(out))
    }
}

// ── One-shot helpers ───────────────────────────────────────────

/// Compute SHA-512 of `data`.
pub fn sha512(data: &[u8]) -> Sha512Digest {
    let mut h = Sha512Hasher::new(Sha512Variant::Sha512);
    h.update(data).unwrap();
    h.finalize_512().unwrap()
}

/// Compute SHA-384 of `data`.
pub fn sha384(data: &[u8]) -> Sha384Digest {
    let mut h = Sha512Hasher::new(Sha512Variant::Sha384);
    h.update(data).unwrap();
    h.finalize_384().unwrap()
}

/// Compute SHA-512/256 of `data`.
pub fn sha512_256(data: &[u8]) -> Sha512_256Digest {
    let mut h = Sha512Hasher::new(Sha512Variant::Sha512_256);
    h.update(data).unwrap();
    h.finalize_512_256().unwrap()
}

/// Compute HMAC-SHA512(key, message).
pub fn hmac_sha512(key: &[u8], message: &[u8]) -> Sha512Digest {
    let key_hash;
    let k = if key.len() > 128 {
        key_hash = sha512(key);
        &key_hash.0[..]
    } else {
        key
    };

    let mut k_padded = [0u8; 128];
    k_padded[..k.len()].copy_from_slice(k);

    let mut ipad = [0x36u8; 128];
    let mut opad = [0x5cu8; 128];
    for i in 0..128 {
        ipad[i] ^= k_padded[i];
        opad[i] ^= k_padded[i];
    }

    let mut inner = Sha512Hasher::new(Sha512Variant::Sha512);
    inner.update(&ipad).unwrap();
    inner.update(message).unwrap();
    let inner_digest = inner.finalize_512().unwrap();

    let mut outer = Sha512Hasher::new(Sha512Variant::Sha512);
    outer.update(&opad).unwrap();
    outer.update(&inner_digest.0).unwrap();
    outer.finalize_512().unwrap()
}

// ── Internal ───────────────────────────────────────────────────

fn process_block(state: &mut [u64; 8], block: &[u8; 128]) {
    let mut w = [0u64; 80];
    for i in 0..16 {
        w[i] = u64::from_be_bytes([
            block[i * 8],     block[i * 8 + 1], block[i * 8 + 2], block[i * 8 + 3],
            block[i * 8 + 4], block[i * 8 + 5], block[i * 8 + 6], block[i * 8 + 7],
        ]);
    }
    for i in 16..80 {
        let s0 = w[i - 15].rotate_right(1) ^ w[i - 15].rotate_right(8) ^ (w[i - 15] >> 7);
        let s1 = w[i - 2].rotate_right(19) ^ w[i - 2].rotate_right(61) ^ (w[i - 2] >> 6);
        w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
    }

    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;

    for i in 0..80 {
        let s1 = e.rotate_right(14) ^ e.rotate_right(18) ^ e.rotate_right(41);
        let ch = (e & f) ^ ((!e) & g);
        let temp1 = h.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
        let s0 = a.rotate_right(28) ^ a.rotate_right(34) ^ a.rotate_right(39);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let temp2 = s0.wrapping_add(maj);

        h = g; g = f; f = e;
        e = d.wrapping_add(temp1);
        d = c; c = b; b = a;
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
    fn test_sha512_empty() {
        let d = sha512(b"");
        assert_eq!(
            d.to_hex(),
            "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce\
             47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e"
        );
    }

    #[test]
    fn test_sha512_abc() {
        let d = sha512(b"abc");
        assert_eq!(
            d.to_hex(),
            "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a\
             2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f"
        );
    }

    #[test]
    fn test_sha384_empty() {
        let d = sha384(b"");
        assert_eq!(
            d.to_hex(),
            "38b060a751ac96384cd9327eb1b1e36a21fdb71114be07434c0cc7bf63f6e1da\
             274edebfe76f65fbd51ad2f14898b95b"
        );
    }

    #[test]
    fn test_sha384_abc() {
        let d = sha384(b"abc");
        assert_eq!(
            d.to_hex(),
            "cb00753f45a35e8bb5a03d699ac65007272c32ab0eded1631a8b605a43ff5bed\
             8086072ba1e7cc2358baeca134c825a7"
        );
    }

    #[test]
    fn test_sha512_256_empty() {
        let d = sha512_256(b"");
        assert_eq!(
            d.to_hex(),
            "c672b8d1ef56ed28ab87c3622c5114069bdd3ad7b8f9737498d0c01ecef0967a"
        );
    }

    #[test]
    fn test_sha512_256_abc() {
        let d = sha512_256(b"abc");
        assert_eq!(
            d.to_hex(),
            "53048e2681941ef99b2e29b76b4c7dabe4c2d0c634fc6d46e0e2f13107e7af23"
        );
    }

    #[test]
    fn test_streaming_sha512() {
        let mut h = Sha512Hasher::new(Sha512Variant::Sha512);
        h.update(b"a").unwrap();
        h.update(b"bc").unwrap();
        let d = h.finalize_512().unwrap();
        assert_eq!(d, sha512(b"abc"));
    }

    #[test]
    fn test_finalize_twice_errors() {
        let mut h = Sha512Hasher::new(Sha512Variant::Sha512);
        h.finalize_512().unwrap();
        assert!(h.finalize_512().is_err());
    }

    #[test]
    fn test_hmac_sha512() {
        // RFC 4231 Test Case 2
        let key = b"Jefe";
        let data = b"what do ya want for nothing?";
        let mac = hmac_sha512(key, data);
        assert_eq!(
            mac.to_hex(),
            "164b7a7bfcf819e2e395fbe73b56e0a387bd64222e831fd610270cd7ea250554\
             9758bf75c05a994a6d034f65f8f0e6fdcaeab1a34d4a6b4b636e070a38bce737"
        );
    }

    #[test]
    fn test_hex_roundtrip() {
        let d = sha512(b"test");
        let hex = d.to_hex();
        let d2 = Sha512Digest::from_hex(&hex).unwrap();
        assert_eq!(d, d2);
    }

    #[test]
    fn test_constant_time_eq() {
        let d1 = sha512(b"hello");
        let d2 = sha512(b"hello");
        let d3 = sha512(b"world");
        assert!(d1.constant_time_eq(&d2));
        assert!(!d1.constant_time_eq(&d3));
    }

    #[test]
    fn test_sha512_two_block() {
        // "abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu"
        let msg = b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu";
        let d = sha512(msg);
        assert_eq!(
            d.to_hex(),
            "8e959b75dae313da8cf4f72814fc143f8f7779c6eb9f7fa17299aeadb6889018\
             501d289e4900f7e4331b99dec4b5433ac7d329eeb6dd26545e96e55b874be909"
        );
    }
}
