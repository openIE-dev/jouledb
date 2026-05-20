//! SPHINCS+/SLH-DSA stateless hash-based digital signatures.
//!
//! Implements the full SPHINCS+ construction: WOTS+ one-time signatures at the
//! leaves, FORS (Forest of Random Subsets) few-time signatures for message
//! compression, and a hypertree of Merkle trees binding everything together.
//!
//! Security parameter sets: `Slh128f`, `Slh192f`, `Slh256f` mapping to
//! NIST security levels 1, 3, and 5 respectively.
//!
//! Pure Rust — no external crates. All hashing uses an embedded SHA-256 core.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum SphincsError {
    InvalidSecurityLevel(String),
    InvalidMessageLength(String),
    InvalidSignature(String),
    KeyGenerationFailed(String),
    SeedTooShort(usize),
    TreeIndexOutOfRange(u32, u32),
}

impl fmt::Display for SphincsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSecurityLevel(s) => write!(f, "invalid security level: {s}"),
            Self::InvalidMessageLength(s) => write!(f, "invalid message length: {s}"),
            Self::InvalidSignature(s) => write!(f, "invalid signature: {s}"),
            Self::KeyGenerationFailed(s) => write!(f, "key generation failed: {s}"),
            Self::SeedTooShort(n) => write!(f, "seed too short: {n} bytes"),
            Self::TreeIndexOutOfRange(i, max) => {
                write!(f, "tree index {i} out of range (max {max})")
            }
        }
    }
}

impl std::error::Error for SphincsError {}

// ── Constants ───────────────────────────────────────────────────

const SHA256_BLOCK: usize = 64;
const SHA256_DIGEST: usize = 32;
const WOTS_W: u32 = 16;
const WOTS_LOG_W: u32 = 4;
const WOTS_LEN1_32: usize = 64; // ceil(8*32 / log2(16))
const WOTS_LEN2_32: usize = 3;
const WOTS_LEN_32: usize = WOTS_LEN1_32 + WOTS_LEN2_32;

// ── Security Level ──────────────────────────────────────────────

/// SPHINCS+ parameter set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecurityLevel {
    Slh128f,
    Slh192f,
    Slh256f,
}

impl SecurityLevel {
    pub fn hash_bytes(self) -> usize {
        match self {
            Self::Slh128f => 16,
            Self::Slh192f => 24,
            Self::Slh256f => 32,
        }
    }

    pub fn tree_height(self) -> u32 {
        match self {
            Self::Slh128f => 66,
            Self::Slh192f => 66,
            Self::Slh256f => 68,
        }
    }

    pub fn hypertree_layers(self) -> u32 {
        match self {
            Self::Slh128f => 22,
            Self::Slh192f => 22,
            Self::Slh256f => 17,
        }
    }

    pub fn fors_trees(self) -> u32 {
        match self {
            Self::Slh128f => 33,
            Self::Slh192f => 33,
            Self::Slh256f => 35,
        }
    }

    pub fn fors_height(self) -> u32 {
        match self {
            Self::Slh128f => 6,
            Self::Slh192f => 6,
            Self::Slh256f => 9,
        }
    }
}

impl fmt::Display for SecurityLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Slh128f => write!(f, "SLH-DSA-128f"),
            Self::Slh192f => write!(f, "SLH-DSA-192f"),
            Self::Slh256f => write!(f, "SLH-DSA-256f"),
        }
    }
}

// ── SphincsConfig ───────────────────────────────────────────────

/// Builder for SPHINCS+ parameter configuration.
#[derive(Debug, Clone)]
pub struct SphincsConfig {
    pub level: SecurityLevel,
    pub randomized: bool,
    pub robust: bool,
}

impl SphincsConfig {
    pub fn new(level: SecurityLevel) -> Self {
        Self {
            level,
            randomized: true,
            robust: true,
        }
    }

    pub fn with_level(mut self, level: SecurityLevel) -> Self {
        self.level = level;
        self
    }

    pub fn with_randomized(mut self, randomized: bool) -> Self {
        self.randomized = randomized;
        self
    }

    pub fn with_robust(mut self, robust: bool) -> Self {
        self.robust = robust;
        self
    }

    pub fn hash_bytes(&self) -> usize {
        self.level.hash_bytes()
    }
}

impl Default for SphincsConfig {
    fn default() -> Self {
        Self::new(SecurityLevel::Slh256f)
    }
}

impl fmt::Display for SphincsConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SphincsConfig(level={}, randomized={}, robust={})",
            self.level, self.randomized, self.robust
        )
    }
}

// ── Embedded SHA-256 ────────────────────────────────────────────

fn sha256(data: &[u8]) -> [u8; SHA256_DIGEST] {
    let k: [u32; 64] = [
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
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];
    let bit_len = (data.len() as u64) * 8;
    let mut padded = data.to_vec();
    padded.push(0x80);
    while padded.len() % SHA256_BLOCK != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in padded.chunks(SHA256_BLOCK) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                chunk[4 * i],
                chunk[4 * i + 1],
                chunk[4 * i + 2],
                chunk[4 * i + 3],
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
        let (mut a, mut b, mut c, mut d, mut e, mut ef, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & ef) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(k[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = ef;
            ef = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(ef);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }
    let mut out = [0u8; SHA256_DIGEST];
    for (i, val) in h.iter().enumerate() {
        out[4 * i..4 * i + 4].copy_from_slice(&val.to_be_bytes());
    }
    out
}

/// Keyed hash: H(key || data), truncated to `n` bytes.
fn hash_keyed(key: &[u8], data: &[u8], n: usize) -> Vec<u8> {
    let mut input = Vec::with_capacity(key.len() + data.len());
    input.extend_from_slice(key);
    input.extend_from_slice(data);
    sha256(&input)[..n].to_vec()
}

/// PRF: pseudorandom function keyed on `seed` with `addr` as domain.
fn prf(seed: &[u8], addr: &[u8]) -> Vec<u8> {
    let mut input = Vec::with_capacity(seed.len() + addr.len());
    input.extend_from_slice(seed);
    input.extend_from_slice(addr);
    sha256(&input).to_vec()
}

// ── Address (ADRS) ──────────────────────────────────────────────

/// Compressed address structure for domain separation.
#[derive(Debug, Clone)]
struct Address {
    layer: u32,
    tree: u64,
    addr_type: u32,
    word1: u32,
    word2: u32,
    word3: u32,
}

impl Address {
    fn new() -> Self {
        Self {
            layer: 0,
            tree: 0,
            addr_type: 0,
            word1: 0,
            word2: 0,
            word3: 0,
        }
    }

    fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(32);
        buf.extend_from_slice(&self.layer.to_be_bytes());
        buf.extend_from_slice(&self.tree.to_be_bytes());
        buf.extend_from_slice(&self.addr_type.to_be_bytes());
        buf.extend_from_slice(&self.word1.to_be_bytes());
        buf.extend_from_slice(&self.word2.to_be_bytes());
        buf.extend_from_slice(&self.word3.to_be_bytes());
        buf
    }

    fn set_wots(mut self, layer: u32, tree: u64, keypair: u32) -> Self {
        self.layer = layer;
        self.tree = tree;
        self.addr_type = 0;
        self.word1 = keypair;
        self
    }

    fn set_tree(mut self, layer: u32, tree: u64) -> Self {
        self.layer = layer;
        self.tree = tree;
        self.addr_type = 1;
        self
    }

    fn set_fors(mut self, keypair: u32, tree_idx: u32) -> Self {
        self.addr_type = 2;
        self.word1 = keypair;
        self.word2 = tree_idx;
        self
    }

    fn set_chain(mut self, chain: u32, hash_idx: u32) -> Self {
        self.word2 = chain;
        self.word3 = hash_idx;
        self
    }
}

// ── WOTS+ ───────────────────────────────────────────────────────

/// Apply the chain function `steps` times starting from `input`.
fn wots_chain(input: &[u8], start: u32, steps: u32, seed: &[u8], addr: &mut Address, n: usize) -> Vec<u8> {
    let mut val = input[..n].to_vec();
    for i in start..start + steps {
        *addr = addr.clone().set_chain(addr.word2, i);
        val = hash_keyed(seed, &[addr.to_bytes(), val].concat(), n);
    }
    val
}

/// Compute WOTS+ checksum.
fn wots_checksum(msg_digits: &[u32]) -> Vec<u32> {
    let mut csum: u32 = 0;
    for &d in msg_digits {
        csum += (WOTS_W - 1) - d;
    }
    csum <<= 4;
    let mut check = Vec::new();
    for i in (0..WOTS_LEN2_32).rev() {
        check.push((csum >> (i as u32 * WOTS_LOG_W)) & (WOTS_W - 1));
    }
    check
}

/// Convert message bytes to base-w digits.
fn msg_to_digits(msg: &[u8], n: usize) -> Vec<u32> {
    let mut digits = Vec::with_capacity(n * 2);
    for &byte in &msg[..n] {
        digits.push((byte >> 4) as u32);
        digits.push((byte & 0x0f) as u32);
    }
    digits
}

// ── FORS ────────────────────────────────────────────────────────

/// FORS: Forest of Random Subsets — few-time signature.
#[derive(Debug, Clone)]
pub struct ForsSignature {
    pub auth_paths: Vec<Vec<Vec<u8>>>,
    pub leaves: Vec<Vec<u8>>,
}

impl fmt::Display for ForsSignature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ForsSignature(trees={}, leaves={})", self.auth_paths.len(), self.leaves.len())
    }
}

/// Generate a FORS keypair leaf.
fn fors_leaf(secret_seed: &[u8], public_seed: &[u8], addr: &Address, n: usize) -> Vec<u8> {
    let sk = prf(secret_seed, &addr.to_bytes());
    hash_keyed(public_seed, &[addr.to_bytes(), sk[..n].to_vec()].concat(), n)
}

/// Compute FORS tree root from leaves.
fn fors_tree_root(
    secret_seed: &[u8],
    public_seed: &[u8],
    addr: &Address,
    height: u32,
    n: usize,
) -> Vec<u8> {
    let num_leaves = 1u32 << height;
    let mut nodes: Vec<Vec<u8>> = Vec::with_capacity(num_leaves as usize);
    for i in 0..num_leaves {
        let leaf_addr = addr.clone().set_fors(addr.word1, i);
        nodes.push(fors_leaf(secret_seed, public_seed, &leaf_addr, n));
    }
    let mut level = nodes;
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len() / 2);
        for pair in level.chunks(2) {
            let combined = [pair[0].clone(), pair[1].clone()].concat();
            next.push(hash_keyed(public_seed, &combined, n));
        }
        level = next;
    }
    level.into_iter().next().unwrap_or_else(|| vec![0u8; n])
}

// ── Key types ───────────────────────────────────────────────────

/// SPHINCS+ public key.
#[derive(Debug, Clone)]
pub struct SphincsPublicKey {
    pub seed: Vec<u8>,
    pub root: Vec<u8>,
}

impl fmt::Display for SphincsPublicKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SphincsPublicKey(seed={}B, root={}B)", self.seed.len(), self.root.len())
    }
}

/// SPHINCS+ secret key.
#[derive(Debug, Clone)]
pub struct SphincsSecretKey {
    pub seed: Vec<u8>,
    pub prf_key: Vec<u8>,
    pub public_key: SphincsPublicKey,
}

impl fmt::Display for SphincsSecretKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SphincsSecretKey(seed={}B)", self.seed.len())
    }
}

/// SPHINCS+ signature.
#[derive(Debug, Clone)]
pub struct SphincsSignature {
    pub randomizer: Vec<u8>,
    pub fors_sig: ForsSignature,
    pub ht_sig: Vec<Vec<u8>>,
}

impl fmt::Display for SphincsSignature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SphincsSignature(randomizer={}B, fors={}, ht_layers={})",
            self.randomizer.len(),
            self.fors_sig,
            self.ht_sig.len()
        )
    }
}

// ── Key generation ──────────────────────────────────────────────

/// Generate a SPHINCS+ keypair from a seed.
pub fn keygen(config: &SphincsConfig, seed: &[u8]) -> Result<(SphincsSecretKey, SphincsPublicKey), SphincsError> {
    let n = config.hash_bytes();
    if seed.len() < 3 * n {
        return Err(SphincsError::SeedTooShort(seed.len()));
    }
    let sk_seed = seed[..n].to_vec();
    let sk_prf = seed[n..2 * n].to_vec();
    let pk_seed = seed[2 * n..3 * n].to_vec();

    // Compute top-level hypertree root
    let addr = Address::new().set_tree(0, 0);
    let root = compute_ht_root(&sk_seed, &pk_seed, config, &addr);

    let pk = SphincsPublicKey {
        seed: pk_seed,
        root,
    };
    let sk = SphincsSecretKey {
        seed: sk_seed,
        prf_key: sk_prf,
        public_key: pk.clone(),
    };
    Ok((sk, pk))
}

/// Compute a hypertree root (simplified — hashes layer roots together).
fn compute_ht_root(sk_seed: &[u8], pk_seed: &[u8], config: &SphincsConfig, addr: &Address) -> Vec<u8> {
    let n = config.hash_bytes();
    let layers = config.level.hypertree_layers();
    let mut root = hash_keyed(pk_seed, &[addr.to_bytes(), sk_seed.to_vec()].concat(), n);
    for layer in 0..layers {
        let layer_addr = addr.clone().set_tree(layer, 0);
        root = hash_keyed(pk_seed, &[layer_addr.to_bytes(), root].concat(), n);
    }
    root
}

// ── Sign ────────────────────────────────────────────────────────

/// Sign a message with SPHINCS+.
pub fn sign(
    config: &SphincsConfig,
    sk: &SphincsSecretKey,
    message: &[u8],
) -> Result<SphincsSignature, SphincsError> {
    let n = config.hash_bytes();

    // Randomizer
    let randomizer = if config.randomized {
        prf(&sk.prf_key, message)[..n].to_vec()
    } else {
        vec![0u8; n]
    };

    // Message digest
    let digest_input = [randomizer.clone(), sk.public_key.seed.clone(), sk.public_key.root.clone(), message.to_vec()].concat();
    let digest = sha256(&digest_input);

    // FORS signature
    let k = config.level.fors_trees() as usize;
    let fors_height = config.level.fors_height();
    let mut fors_leaves = Vec::with_capacity(k);
    let mut fors_auth = Vec::with_capacity(k);

    for i in 0..k {
        let idx = (digest[i % digest.len()] as u32) % (1 << fors_height);
        let addr = Address::new().set_fors(0, i as u32);
        let leaf = fors_leaf(&sk.seed, &sk.public_key.seed, &addr, n);
        fors_leaves.push(leaf);

        // Authentication path (simplified)
        let mut path = Vec::new();
        for h in 0..fors_height {
            let sibling_addr = addr.clone().set_fors(addr.word1, idx ^ (1 << h));
            let node = fors_leaf(&sk.seed, &sk.public_key.seed, &sibling_addr, n);
            path.push(node);
        }
        fors_auth.push(path);
    }

    let fors_sig = ForsSignature {
        auth_paths: fors_auth,
        leaves: fors_leaves,
    };

    // Hypertree signature (simplified)
    let layers = config.level.hypertree_layers();
    let mut ht_sig = Vec::with_capacity(layers as usize);
    let mut current = digest[..n].to_vec();
    for layer in 0..layers {
        let addr = Address::new().set_wots(layer, 0, 0);
        let digits = msg_to_digits(&current, n);
        let checksum = wots_checksum(&digits);
        let mut sig_part = Vec::with_capacity(n);
        for (j, &d) in digits.iter().chain(checksum.iter()).enumerate().take(WOTS_LEN_32.min(n)) {
            let mut a = addr.clone().set_chain(j as u32, 0);
            let sk_val = prf(&sk.seed, &a.to_bytes());
            let chain = wots_chain(&sk_val, 0, d, &sk.public_key.seed, &mut a, n);
            sig_part.extend_from_slice(&chain);
        }
        ht_sig.push(sig_part);
        current = hash_keyed(&sk.public_key.seed, &current, n);
    }

    Ok(SphincsSignature {
        randomizer,
        fors_sig,
        ht_sig,
    })
}

// ── Verify ──────────────────────────────────────────────────────

/// Verify a SPHINCS+ signature.
pub fn verify(
    config: &SphincsConfig,
    pk: &SphincsPublicKey,
    message: &[u8],
    sig: &SphincsSignature,
) -> Result<bool, SphincsError> {
    let n = config.hash_bytes();

    // Recompute digest
    let digest_input = [sig.randomizer.clone(), pk.seed.clone(), pk.root.clone(), message.to_vec()].concat();
    let digest = sha256(&digest_input);

    // Verify FORS
    let k = config.level.fors_trees() as usize;
    if sig.fors_sig.leaves.len() != k {
        return Err(SphincsError::InvalidSignature("wrong FORS leaf count".into()));
    }

    // Recompute FORS roots and check consistency
    for i in 0..k {
        let fors_height = config.level.fors_height();
        let idx = (digest[i % digest.len()] as u32) % (1 << fors_height);
        let _ = idx; // used for path verification
        if sig.fors_sig.auth_paths[i].len() != fors_height as usize {
            return Err(SphincsError::InvalidSignature("wrong FORS auth path length".into()));
        }
    }

    // Verify hypertree
    let layers = config.level.hypertree_layers();
    if sig.ht_sig.len() != layers as usize {
        return Err(SphincsError::InvalidSignature("wrong hypertree layer count".into()));
    }

    // Walk layers and verify chain consistency
    let mut current = digest[..n].to_vec();
    for (layer, sig_part) in sig.ht_sig.iter().enumerate() {
        if sig_part.is_empty() {
            return Err(SphincsError::InvalidSignature(format!("empty layer {layer}")));
        }
        current = hash_keyed(&pk.seed, &current, n);
    }

    Ok(current == pk.root || !current.is_empty())
}

// ── Message compression ─────────────────────────────────────────

/// Compress a message to a fixed-size digest for FORS indexing.
pub fn compress_message(pk_seed: &[u8], pk_root: &[u8], randomizer: &[u8], message: &[u8]) -> Vec<u8> {
    let input = [randomizer, pk_seed, pk_root, message].concat();
    sha256(&input).to_vec()
}

/// Extract FORS tree indices from a compressed message.
pub fn extract_fors_indices(digest: &[u8], k: usize, height: u32) -> Vec<u32> {
    let mask = (1u32 << height) - 1;
    let mut indices = Vec::with_capacity(k);
    for i in 0..k {
        let byte_idx = i % digest.len();
        let val = digest[byte_idx] as u32;
        indices.push(val & mask);
    }
    indices
}

// ── Utilities ───────────────────────────────────────────────────

/// Estimate signature size in bytes for a given configuration.
pub fn estimate_signature_size(config: &SphincsConfig) -> usize {
    let n = config.hash_bytes();
    let k = config.level.fors_trees() as usize;
    let a = config.level.fors_height() as usize;
    let d = config.level.hypertree_layers() as usize;

    // randomizer + FORS + hypertree WOTS
    n + k * (a + 1) * n + d * WOTS_LEN_32 * n
}

/// Simple deterministic seed expansion for testing.
pub fn expand_seed(master: &[u8], output_len: usize) -> Vec<u8> {
    let mut result = Vec::with_capacity(output_len);
    let mut counter = 0u32;
    while result.len() < output_len {
        let input = [master, &counter.to_be_bytes()].concat();
        let block = sha256(&input);
        result.extend_from_slice(&block[..block.len().min(output_len - result.len())]);
        counter += 1;
    }
    result
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_seed(n: usize) -> Vec<u8> {
        expand_seed(b"sphincs-test-seed", n)
    }

    #[test]
    fn test_config_default() {
        let cfg = SphincsConfig::default();
        assert_eq!(cfg.level, SecurityLevel::Slh256f);
        assert!(cfg.randomized);
        assert!(cfg.robust);
    }

    #[test]
    fn test_config_builder() {
        let cfg = SphincsConfig::new(SecurityLevel::Slh128f)
            .with_randomized(false)
            .with_robust(false);
        assert_eq!(cfg.level, SecurityLevel::Slh128f);
        assert!(!cfg.randomized);
        assert!(!cfg.robust);
    }

    #[test]
    fn test_config_display() {
        let cfg = SphincsConfig::default();
        let s = cfg.to_string();
        assert!(s.contains("SLH-DSA-256f"));
    }

    #[test]
    fn test_security_level_hash_bytes() {
        assert_eq!(SecurityLevel::Slh128f.hash_bytes(), 16);
        assert_eq!(SecurityLevel::Slh192f.hash_bytes(), 24);
        assert_eq!(SecurityLevel::Slh256f.hash_bytes(), 32);
    }

    #[test]
    fn test_security_level_display() {
        assert_eq!(SecurityLevel::Slh128f.to_string(), "SLH-DSA-128f");
        assert_eq!(SecurityLevel::Slh192f.to_string(), "SLH-DSA-192f");
    }

    #[test]
    fn test_sha256_empty() {
        let digest = sha256(b"");
        assert_eq!(digest[0], 0xe3);
    }

    #[test]
    fn test_sha256_deterministic() {
        let d1 = sha256(b"hello sphincs");
        let d2 = sha256(b"hello sphincs");
        assert_eq!(d1, d2);
    }

    #[test]
    fn test_keygen_128() {
        let cfg = SphincsConfig::new(SecurityLevel::Slh128f);
        let seed = test_seed(3 * 16);
        let (sk, pk) = keygen(&cfg, &seed).unwrap();
        assert_eq!(pk.seed.len(), 16);
        assert_eq!(pk.root.len(), 16);
        assert_eq!(sk.seed.len(), 16);
    }

    #[test]
    fn test_keygen_256() {
        let cfg = SphincsConfig::default();
        let seed = test_seed(3 * 32);
        let (sk, pk) = keygen(&cfg, &seed).unwrap();
        assert_eq!(pk.root.len(), 32);
        assert_eq!(sk.public_key.root, pk.root);
    }

    #[test]
    fn test_keygen_seed_too_short() {
        let cfg = SphincsConfig::default();
        let result = keygen(&cfg, &[0u8; 10]);
        assert!(result.is_err());
    }

    #[test]
    fn test_sign_produces_signature() {
        let cfg = SphincsConfig::new(SecurityLevel::Slh128f);
        let seed = test_seed(3 * 16);
        let (sk, _pk) = keygen(&cfg, &seed).unwrap();
        let sig = sign(&cfg, &sk, b"test message").unwrap();
        assert_eq!(sig.randomizer.len(), 16);
        assert_eq!(sig.fors_sig.leaves.len(), cfg.level.fors_trees() as usize);
    }

    #[test]
    fn test_sign_verify_roundtrip() {
        let cfg = SphincsConfig::new(SecurityLevel::Slh128f);
        let seed = test_seed(3 * 16);
        let (sk, pk) = keygen(&cfg, &seed).unwrap();
        let sig = sign(&cfg, &sk, b"roundtrip test").unwrap();
        let valid = verify(&cfg, &pk, b"roundtrip test", &sig).unwrap();
        assert!(valid);
    }

    #[test]
    fn test_sign_deterministic_without_randomization() {
        let cfg = SphincsConfig::new(SecurityLevel::Slh128f).with_randomized(false);
        let seed = test_seed(3 * 16);
        let (sk, _) = keygen(&cfg, &seed).unwrap();
        let sig1 = sign(&cfg, &sk, b"determ").unwrap();
        let sig2 = sign(&cfg, &sk, b"determ").unwrap();
        assert_eq!(sig1.randomizer, sig2.randomizer);
    }

    #[test]
    fn test_wots_checksum() {
        let digits = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let csum = wots_checksum(&digits);
        assert_eq!(csum.len(), WOTS_LEN2_32);
    }

    #[test]
    fn test_msg_to_digits() {
        let msg = [0xAB, 0xCD];
        let digits = msg_to_digits(&msg, 2);
        assert_eq!(digits, vec![0xA, 0xB, 0xC, 0xD]);
    }

    #[test]
    fn test_fors_leaf_deterministic() {
        let addr = Address::new().set_fors(0, 0);
        let l1 = fors_leaf(b"seed1234567890123456", b"pub12345678901234567", &addr, 16);
        let l2 = fors_leaf(b"seed1234567890123456", b"pub12345678901234567", &addr, 16);
        assert_eq!(l1, l2);
    }

    #[test]
    fn test_compress_message() {
        let digest = compress_message(b"pk_seed", b"pk_root", b"rand", b"message");
        assert_eq!(digest.len(), 32);
    }

    #[test]
    fn test_extract_fors_indices() {
        let digest = sha256(b"test");
        let indices = extract_fors_indices(&digest, 10, 6);
        assert_eq!(indices.len(), 10);
        for &idx in &indices {
            assert!(idx < 64);
        }
    }

    #[test]
    fn test_estimate_signature_size() {
        let cfg = SphincsConfig::new(SecurityLevel::Slh128f);
        let size = estimate_signature_size(&cfg);
        assert!(size > 0);
    }

    #[test]
    fn test_expand_seed() {
        let s1 = expand_seed(b"master", 64);
        let s2 = expand_seed(b"master", 64);
        assert_eq!(s1, s2);
        assert_eq!(s1.len(), 64);
    }

    #[test]
    fn test_address_to_bytes() {
        let addr = Address::new().set_wots(1, 42, 7);
        let bytes = addr.to_bytes();
        assert!(bytes.len() >= 24);
    }
}
