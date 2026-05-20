//! SLH-DSA (Stateless Hash-based Digital Signature Algorithm) - FIPS 205
//!
//! Implementation of the NIST-standardized post-quantum signature scheme
//! based purely on hash function security (no lattice assumptions).
//!
//! ## Parameter Sets
//!
//! | Parameter     | Security | Sig Size | Speed    |
//! |---------------|----------|----------|----------|
//! | SLH-DSA-128f  | Level 1  | ~17 KB   | Fast     |
//! | SLH-DSA-128s  | Level 1  | ~7 KB    | Small    |
//! | SLH-DSA-192f  | Level 3  | ~35 KB   | Fast     |
//! | SLH-DSA-192s  | Level 3  | ~16 KB   | Small    |
//! | SLH-DSA-256f  | Level 5  | ~50 KB   | Fast     |
//! | SLH-DSA-256s  | Level 5  | ~30 KB   | Small    |
//!
//! ## Components
//!
//! - **WOTS+** - Winternitz One-Time Signature
//! - **XMSS** - eXtended Merkle Signature Scheme (Merkle tree of WOTS+)
//! - **Hypertree** - Tree of XMSS trees
//! - **FORS** - Forest of Random Subsets (few-time signature)

use super::common::{ConstantTime, SecureZeroingVec, Sha3_256, Shake256};
use super::{PqcError, PqcResult};
use rand::Rng;

// ============================================================================
// SLH-DSA Parameters
// ============================================================================

/// SLH-DSA parameter set
#[derive(Clone, Copy, Debug)]
pub struct SlhDsaParams {
    /// Parameter set name
    pub name: &'static str,
    /// Security parameter n (hash output length in bytes)
    pub n: usize,
    /// Hypertree height h
    pub h: usize,
    /// Number of XMSS tree layers d
    pub d: usize,
    /// FORS trees count k
    pub fors_k: usize,
    /// FORS tree height a
    pub fors_a: usize,
    /// Winternitz parameter w
    pub w: usize,
    /// XMSS tree height (h/d)
    pub hp: usize,
}

// Parameter sets per FIPS 205

/// SLH-DSA-SHAKE-128f (fast)
pub const SLH_DSA_128F_PARAMS: SlhDsaParams = SlhDsaParams {
    name: "SLH-DSA-SHAKE-128f",
    n: 16,
    h: 66,
    d: 22,
    fors_k: 33,
    fors_a: 6,
    w: 16,
    hp: 3,
};

/// SLH-DSA-SHAKE-128s (small)
pub const SLH_DSA_128S_PARAMS: SlhDsaParams = SlhDsaParams {
    name: "SLH-DSA-SHAKE-128s",
    n: 16,
    h: 63,
    d: 7,
    fors_k: 14,
    fors_a: 12,
    w: 16,
    hp: 9,
};

/// SLH-DSA-SHAKE-192f (fast)
pub const SLH_DSA_192F_PARAMS: SlhDsaParams = SlhDsaParams {
    name: "SLH-DSA-SHAKE-192f",
    n: 24,
    h: 66,
    d: 22,
    fors_k: 33,
    fors_a: 8,
    w: 16,
    hp: 3,
};

/// SLH-DSA-SHAKE-192s (small)
pub const SLH_DSA_192S_PARAMS: SlhDsaParams = SlhDsaParams {
    name: "SLH-DSA-SHAKE-192s",
    n: 24,
    h: 63,
    d: 7,
    fors_k: 17,
    fors_a: 14,
    w: 16,
    hp: 9,
};

/// SLH-DSA-SHAKE-256f (fast)
pub const SLH_DSA_256F_PARAMS: SlhDsaParams = SlhDsaParams {
    name: "SLH-DSA-SHAKE-256f",
    n: 32,
    h: 68,
    d: 17,
    fors_k: 35,
    fors_a: 9,
    w: 16,
    hp: 4,
};

/// SLH-DSA-SHAKE-256s (small)
pub const SLH_DSA_256S_PARAMS: SlhDsaParams = SlhDsaParams {
    name: "SLH-DSA-SHAKE-256s",
    n: 32,
    h: 64,
    d: 8,
    fors_k: 22,
    fors_a: 14,
    w: 16,
    hp: 8,
};

impl SlhDsaParams {
    /// Public key size in bytes
    pub const fn public_key_size(&self) -> usize {
        2 * self.n
    }

    /// Secret key size in bytes
    pub const fn secret_key_size(&self) -> usize {
        4 * self.n
    }

    /// WOTS+ signature length
    pub const fn wots_len(&self) -> usize {
        let len1 = (8 * self.n + self.w.ilog2() as usize - 1) / self.w.ilog2() as usize;
        let len2 = (len1 * (self.w - 1)).ilog2() as usize / self.w.ilog2() as usize + 1;
        len1 + len2
    }

    /// Signature size in bytes (approximate)
    pub const fn signature_size(&self) -> usize {
        // sig = randomness || FORS sig || HT sig
        self.n  // randomness
        + self.fors_k * (self.fors_a + 1) * self.n  // FORS
        + self.d * (self.hp * self.n + self.wots_len() * self.n) // hypertree
    }
}

// ============================================================================
// Address Structure (ADRS)
// ============================================================================

/// Address structure for domain separation
#[derive(Clone, Default)]
struct Address {
    data: [u8; 32],
}

impl Address {
    /// Address types
    const WOTS_HASH: u32 = 0;
    const WOTS_PK: u32 = 1;
    const TREE: u32 = 2;
    const FORS_TREE: u32 = 3;
    const FORS_ROOTS: u32 = 4;
    const WOTS_PRF: u32 = 5;
    const FORS_PRF: u32 = 6;

    fn new() -> Self {
        Self { data: [0u8; 32] }
    }

    fn set_layer(&mut self, layer: u32) {
        self.data[0..4].copy_from_slice(&layer.to_be_bytes());
    }

    fn set_tree(&mut self, tree: u64) {
        self.data[4..12].copy_from_slice(&tree.to_be_bytes());
    }

    fn set_type(&mut self, addr_type: u32) {
        self.data[12..16].copy_from_slice(&addr_type.to_be_bytes());
    }

    fn set_keypair(&mut self, keypair: u32) {
        self.data[16..20].copy_from_slice(&keypair.to_be_bytes());
    }

    fn set_chain(&mut self, chain: u32) {
        self.data[20..24].copy_from_slice(&chain.to_be_bytes());
    }

    fn set_hash(&mut self, hash: u32) {
        self.data[24..28].copy_from_slice(&hash.to_be_bytes());
    }

    fn set_tree_height(&mut self, height: u32) {
        self.data[24..28].copy_from_slice(&height.to_be_bytes());
    }

    fn set_tree_index(&mut self, index: u32) {
        self.data[28..32].copy_from_slice(&index.to_be_bytes());
    }

    fn keypair(&self) -> u32 {
        u32::from_be_bytes(self.data[16..20].try_into().unwrap())
    }

    fn set_type_and_clear(&mut self, addr_type: u32) {
        self.data[12..16].copy_from_slice(&addr_type.to_be_bytes());
        self.data[16..32].fill(0);
    }

    fn as_bytes(&self) -> &[u8; 32] {
        &self.data
    }
}

// ============================================================================
// Hash Functions (SHAKE-based)
// ============================================================================

/// Hash function F
fn hash_f(pk_seed: &[u8], adrs: &Address, m: &[u8], n: usize) -> Vec<u8> {
    let mut shake = Shake256::new();
    shake.absorb(pk_seed);
    shake.absorb(adrs.as_bytes());
    shake.absorb(m);
    let mut output = vec![0u8; n];
    shake.squeeze(&mut output);
    output
}

/// Hash function H
fn hash_h(pk_seed: &[u8], adrs: &Address, m1: &[u8], m2: &[u8], n: usize) -> Vec<u8> {
    let mut shake = Shake256::new();
    shake.absorb(pk_seed);
    shake.absorb(adrs.as_bytes());
    shake.absorb(m1);
    shake.absorb(m2);
    let mut output = vec![0u8; n];
    shake.squeeze(&mut output);
    output
}

/// Hash function T (for FORS)
fn hash_t(pk_seed: &[u8], adrs: &Address, m: &[u8], n: usize) -> Vec<u8> {
    let mut shake = Shake256::new();
    shake.absorb(pk_seed);
    shake.absorb(adrs.as_bytes());
    shake.absorb(m);
    let mut output = vec![0u8; n];
    shake.squeeze(&mut output);
    output
}

/// PRF for secret key generation
fn prf(sk_seed: &[u8], pk_seed: &[u8], adrs: &Address, n: usize) -> Vec<u8> {
    let mut shake = Shake256::new();
    shake.absorb(pk_seed);
    shake.absorb(adrs.as_bytes());
    shake.absorb(sk_seed);
    let mut output = vec![0u8; n];
    shake.squeeze(&mut output);
    output
}

/// PRF for message randomization
fn prf_msg(sk_prf: &[u8], opt_rand: &[u8], m: &[u8], n: usize) -> Vec<u8> {
    let mut shake = Shake256::new();
    shake.absorb(sk_prf);
    shake.absorb(opt_rand);
    shake.absorb(m);
    let mut output = vec![0u8; n];
    shake.squeeze(&mut output);
    output
}

/// Hash message to get digest and indices
fn hash_msg(r: &[u8], pk_seed: &[u8], pk_root: &[u8], m: &[u8], params: &SlhDsaParams) -> Vec<u8> {
    let digest_size = (params.fors_k * params.fors_a + 7) / 8
        + (params.h - params.h / params.d + 7) / 8
        + (params.h / params.d + 7) / 8;

    let mut shake = Shake256::new();
    shake.absorb(r);
    shake.absorb(pk_seed);
    shake.absorb(pk_root);
    shake.absorb(m);
    let mut output = vec![0u8; digest_size];
    shake.squeeze(&mut output);
    output
}

// ============================================================================
// WOTS+ (Winternitz One-Time Signature)
// ============================================================================

/// Convert byte string to base-w representation (FIPS 205 Algorithm 4)
fn base_w(x: &[u8], w: usize, out_len: usize) -> Vec<usize> {
    let lg_w = w.ilog2() as usize;
    let mut result = vec![0usize; out_len];
    let mut in_idx = 0;
    let mut bits = 0u32;
    let mut total = 0u32;

    for out in 0..out_len {
        if bits == 0 {
            total = if in_idx < x.len() {
                let v = x[in_idx] as u32;
                in_idx += 1;
                v
            } else {
                0
            };
            bits += 8;
        }
        bits -= lg_w as u32;
        result[out] = ((total >> bits) & (w as u32 - 1)) as usize;
    }

    result
}

/// Compute WOTS+ message digits with checksum (FIPS 205 Algorithms 5-6)
fn wots_msg_base_w(msg: &[u8], params: &SlhDsaParams) -> Vec<usize> {
    let w = params.w;
    let lg_w = w.ilog2() as usize;
    let len1 = 8 * params.n / lg_w;
    let len2 = params.wots_len() - len1;

    // Message part: base-w of message bytes
    let mut digits = base_w(msg, w, len1);

    // Compute checksum
    let mut csum = 0u32;
    for i in 0..len1 {
        csum += (w - 1 - digits[i]) as u32;
    }

    // Left-shift checksum per FIPS 205
    let shift = (8 - ((len2 * lg_w) % 8)) % 8;
    csum <<= shift;

    // Encode checksum as big-endian bytes, then base-w
    let csum_bytes_len = (len2 * lg_w + 7) / 8;
    let csum_bytes: Vec<u8> = (0..csum_bytes_len)
        .map(|i| ((csum >> (8 * (csum_bytes_len - 1 - i))) & 0xFF) as u8)
        .collect();

    let csum_digits = base_w(&csum_bytes, w, len2);
    digits.extend(csum_digits);

    digits
}

/// Compute WOTS+ chain
fn wots_chain(
    x: &[u8],
    start: usize,
    steps: usize,
    pk_seed: &[u8],
    adrs: &mut Address,
    n: usize,
) -> Vec<u8> {
    let mut result = x.to_vec();

    for i in start..(start + steps) {
        adrs.set_hash(i as u32);
        result = hash_f(pk_seed, adrs, &result, n);
    }

    result
}

/// Generate WOTS+ public key
fn wots_pk_gen(
    sk_seed: &[u8],
    pk_seed: &[u8],
    adrs: &mut Address,
    params: &SlhDsaParams,
) -> Vec<u8> {
    let n = params.n;
    let w = params.w;
    let len = params.wots_len();
    let kp = adrs.keypair();

    let mut wots_pk_adrs = adrs.clone();
    wots_pk_adrs.set_type_and_clear(Address::WOTS_PK);
    wots_pk_adrs.set_keypair(kp);

    let mut pk_concat = Vec::with_capacity(len * n);

    for i in 0..len {
        // Clean skADRS for PRF per FIPS 205
        let mut sk_adrs = adrs.clone();
        sk_adrs.set_type_and_clear(Address::WOTS_PRF);
        sk_adrs.set_keypair(kp);
        sk_adrs.set_chain(i as u32);
        let sk = prf(sk_seed, pk_seed, &sk_adrs, n);

        // Clean ADRS for chain: clear stale tree_index (bytes 28-31)
        adrs.set_type(Address::WOTS_HASH);
        adrs.set_keypair(kp);
        adrs.set_chain(i as u32);
        adrs.set_tree_index(0);
        let chain_result = wots_chain(&sk, 0, w - 1, pk_seed, adrs, n);
        pk_concat.extend(chain_result);
    }

    hash_t(pk_seed, &wots_pk_adrs, &pk_concat, n)
}

/// Sign with WOTS+
fn wots_sign(
    msg: &[u8],
    sk_seed: &[u8],
    pk_seed: &[u8],
    adrs: &mut Address,
    params: &SlhDsaParams,
) -> Vec<u8> {
    let n = params.n;
    let len = params.wots_len();
    let kp = adrs.keypair();

    let msg_base_w = wots_msg_base_w(msg, params);

    let mut sig = Vec::with_capacity(len * n);

    for i in 0..len {
        // Clean skADRS for PRF per FIPS 205
        let mut sk_adrs = adrs.clone();
        sk_adrs.set_type_and_clear(Address::WOTS_PRF);
        sk_adrs.set_keypair(kp);
        sk_adrs.set_chain(i as u32);
        let sk = prf(sk_seed, pk_seed, &sk_adrs, n);

        // Clean ADRS for chain: clear stale tree_index
        adrs.set_type(Address::WOTS_HASH);
        adrs.set_keypair(kp);
        adrs.set_chain(i as u32);
        adrs.set_tree_index(0);
        let chain_result = wots_chain(&sk, 0, msg_base_w[i], pk_seed, adrs, n);
        sig.extend(chain_result);
    }

    sig
}

/// Verify WOTS+ signature (compute public key from signature)
fn wots_pk_from_sig(
    sig: &[u8],
    msg: &[u8],
    pk_seed: &[u8],
    adrs: &mut Address,
    params: &SlhDsaParams,
) -> Vec<u8> {
    let n = params.n;
    let w = params.w;
    let len = params.wots_len();
    let kp = adrs.keypair();

    let msg_base_w = wots_msg_base_w(msg, params);

    let mut wots_pk_adrs = adrs.clone();
    wots_pk_adrs.set_type_and_clear(Address::WOTS_PK);
    wots_pk_adrs.set_keypair(kp);

    let mut pk_concat = Vec::with_capacity(len * n);

    for i in 0..len {
        // Clean ADRS for chain: clear stale tree_index
        adrs.set_type(Address::WOTS_HASH);
        adrs.set_keypair(kp);
        adrs.set_chain(i as u32);
        adrs.set_tree_index(0);

        let sig_chunk = &sig[i * n..(i + 1) * n];
        let steps = (w - 1).saturating_sub(msg_base_w[i]);
        let chain_result = wots_chain(sig_chunk, msg_base_w[i], steps, pk_seed, adrs, n);
        pk_concat.extend(chain_result);
    }

    hash_t(pk_seed, &wots_pk_adrs, &pk_concat, n)
}

// ============================================================================
// XMSS (Merkle Tree of WOTS+)
// ============================================================================

/// Build XMSS tree node
fn xmss_node(
    sk_seed: &[u8],
    pk_seed: &[u8],
    i: u32,
    z: u32,
    adrs: &mut Address,
    params: &SlhDsaParams,
) -> Vec<u8> {
    let n = params.n;

    if z == 0 {
        // Leaf: WOTS+ public key
        adrs.set_keypair(i);
        return wots_pk_gen(sk_seed, pk_seed, adrs, params);
    }

    // Internal node: hash of children
    let left = xmss_node(sk_seed, pk_seed, 2 * i, z - 1, adrs, params);
    let right = xmss_node(sk_seed, pk_seed, 2 * i + 1, z - 1, adrs, params);

    // Clean ADRS for TREE type: clear stale WOTS keypair/chain fields
    adrs.set_type(Address::TREE);
    adrs.set_keypair(0);
    adrs.set_chain(0);
    adrs.set_tree_height(z);
    adrs.set_tree_index(i);

    hash_h(pk_seed, adrs, &left, &right, n)
}

/// Sign with XMSS
fn xmss_sign(
    msg: &[u8],
    sk_seed: &[u8],
    pk_seed: &[u8],
    idx: u32,
    adrs: &mut Address,
    params: &SlhDsaParams,
) -> Vec<u8> {
    let n = params.n;
    let hp = params.hp;

    let mut sig = Vec::new();

    // WOTS+ signature
    adrs.set_keypair(idx);
    let wots_sig = wots_sign(msg, sk_seed, pk_seed, adrs, params);
    sig.extend(wots_sig);

    // Authentication path
    for j in 0..hp {
        let sibling_idx = (idx >> j) ^ 1;
        let auth_node = xmss_node(sk_seed, pk_seed, sibling_idx, j as u32, adrs, params);
        sig.extend(auth_node);
    }

    sig
}

/// Compute XMSS root from signature
fn xmss_root_from_sig(
    sig: &[u8],
    msg: &[u8],
    pk_seed: &[u8],
    idx: u32,
    adrs: &mut Address,
    params: &SlhDsaParams,
) -> Vec<u8> {
    let n = params.n;
    let hp = params.hp;
    let wots_len = params.wots_len();

    // Extract WOTS+ signature
    let wots_sig = &sig[..wots_len * n];
    let auth_path = &sig[wots_len * n..];

    // Compute WOTS+ pk
    adrs.set_keypair(idx);
    let mut node = wots_pk_from_sig(wots_sig, msg, pk_seed, adrs, params);

    // Traverse authentication path
    for j in 0..hp {
        let auth_node = &auth_path[j * n..(j + 1) * n];

        // Clean ADRS for TREE type: clear stale WOTS keypair/chain fields
        adrs.set_type(Address::TREE);
        adrs.set_keypair(0);
        adrs.set_chain(0);
        adrs.set_tree_height((j + 1) as u32);
        adrs.set_tree_index(idx >> (j + 1));

        if (idx >> j) & 1 == 0 {
            node = hash_h(pk_seed, adrs, &node, auth_node, n);
        } else {
            node = hash_h(pk_seed, adrs, auth_node, &node, n);
        }
    }

    node
}

// ============================================================================
// FORS (Forest of Random Subsets)
// ============================================================================

/// Generate FORS secret key element
fn fors_sk_gen(sk_seed: &[u8], pk_seed: &[u8], adrs: &mut Address, idx: u32, n: usize) -> Vec<u8> {
    adrs.set_tree_index(idx);
    adrs.set_type(Address::FORS_PRF);
    prf(sk_seed, pk_seed, adrs, n)
}

/// Build FORS tree node
fn fors_node(
    sk_seed: &[u8],
    pk_seed: &[u8],
    i: u32,
    z: u32,
    adrs: &mut Address,
    params: &SlhDsaParams,
) -> Vec<u8> {
    let n = params.n;

    if z == 0 {
        let sk = fors_sk_gen(sk_seed, pk_seed, adrs, i, n);
        adrs.set_type(Address::FORS_TREE);
        adrs.set_tree_height(0);
        adrs.set_tree_index(i);
        return hash_f(pk_seed, adrs, &sk, n);
    }

    let left = fors_node(sk_seed, pk_seed, 2 * i, z - 1, adrs, params);
    let right = fors_node(sk_seed, pk_seed, 2 * i + 1, z - 1, adrs, params);

    adrs.set_type(Address::FORS_TREE);
    adrs.set_tree_height(z);
    adrs.set_tree_index(i);

    hash_h(pk_seed, adrs, &left, &right, n)
}

/// Sign with FORS
fn fors_sign(
    md: &[u8],
    sk_seed: &[u8],
    pk_seed: &[u8],
    adrs: &mut Address,
    params: &SlhDsaParams,
) -> Vec<u8> {
    let n = params.n;
    let k = params.fors_k;
    let a = params.fors_a;

    let mut sig = Vec::new();

    // Extract indices from message digest
    let mut indices = Vec::with_capacity(k);
    for i in 0..k {
        let bit_start = i * a;
        let byte_start = bit_start / 8;
        let bit_offset = bit_start % 8;

        let mut idx = 0u32;
        for b in 0..a {
            let byte_idx = (bit_start + b) / 8;
            let bit_idx = (bit_start + b) % 8;
            if byte_idx < md.len() {
                idx |= (((md[byte_idx] >> bit_idx) & 1) as u32) << b;
            }
        }
        indices.push(idx);
    }

    for i in 0..k {
        let idx = indices[i];

        // Secret key value
        adrs.set_tree_height(0);
        adrs.set_tree_index(i as u32 * (1 << a) + idx);
        let sk = fors_sk_gen(sk_seed, pk_seed, adrs, i as u32 * (1 << a) + idx, n);
        sig.extend(sk);

        // Authentication path
        for j in 0..a {
            let base = i as u32 * (1u32 << a);
            let abs_leaf = base + idx;
            let sibling = (abs_leaf >> j) ^ 1;
            let auth_node = fors_node(sk_seed, pk_seed, sibling, j as u32, adrs, params);
            sig.extend(auth_node);
        }
    }

    sig
}

/// Compute FORS public key from signature
fn fors_pk_from_sig(
    sig: &[u8],
    md: &[u8],
    pk_seed: &[u8],
    adrs: &mut Address,
    params: &SlhDsaParams,
) -> Vec<u8> {
    let n = params.n;
    let k = params.fors_k;
    let a = params.fors_a;

    // Extract indices (same as signing)
    let mut indices = Vec::with_capacity(k);
    for i in 0..k {
        let bit_start = i * a;
        let mut idx = 0u32;
        for b in 0..a {
            let byte_idx = (bit_start + b) / 8;
            let bit_idx = (bit_start + b) % 8;
            if byte_idx < md.len() {
                idx |= (((md[byte_idx] >> bit_idx) & 1) as u32) << b;
            }
        }
        indices.push(idx);
    }

    let mut roots = Vec::with_capacity(k * n);

    for i in 0..k {
        let idx = indices[i];
        let sig_start = i * (n * (a + 1));
        let sk = &sig[sig_start..sig_start + n];

        // Hash leaf
        adrs.set_type(Address::FORS_TREE);
        adrs.set_tree_height(0);
        adrs.set_tree_index(i as u32 * (1 << a) + idx);
        let mut node = hash_f(pk_seed, adrs, sk, n);

        // Compute root from auth path
        for j in 0..a {
            let auth = &sig[sig_start + n * (j + 1)..sig_start + n * (j + 2)];

            adrs.set_tree_height((j + 1) as u32);
            adrs.set_tree_index((i as u32 * (1 << a) + idx) >> (j + 1));

            if (idx >> j) & 1 == 0 {
                node = hash_h(pk_seed, adrs, &node, auth, n);
            } else {
                node = hash_h(pk_seed, adrs, auth, &node, n);
            }
        }

        roots.extend(node);
    }

    // Hash all roots together
    let kp = adrs.keypair();
    adrs.set_type_and_clear(Address::FORS_ROOTS);
    adrs.set_keypair(kp);
    hash_t(pk_seed, adrs, &roots, n)
}

// ============================================================================
// Key Types
// ============================================================================

/// SLH-DSA Public Key
#[derive(Clone)]
pub struct SlhDsaPublicKey {
    data: Vec<u8>,
    params: SlhDsaParams,
}

impl SlhDsaPublicKey {
    /// Create from bytes
    pub fn from_bytes(bytes: &[u8], params: SlhDsaParams) -> PqcResult<Self> {
        if bytes.len() != params.public_key_size() {
            return Err(PqcError::InvalidKey);
        }
        Ok(Self {
            data: bytes.to_vec(),
            params,
        })
    }

    /// Get bytes
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Get seed component
    pub fn pk_seed(&self) -> &[u8] {
        &self.data[..self.params.n]
    }

    /// Get root component
    pub fn pk_root(&self) -> &[u8] {
        &self.data[self.params.n..]
    }

    /// Get parameters
    pub fn params(&self) -> SlhDsaParams {
        self.params
    }
}

/// SLH-DSA Secret Key
#[derive(Clone)]
pub struct SlhDsaSecretKey {
    data: SecureZeroingVec,
    params: SlhDsaParams,
}

impl SlhDsaSecretKey {
    /// Create from bytes
    pub fn from_bytes(bytes: &[u8], params: SlhDsaParams) -> PqcResult<Self> {
        if bytes.len() != params.secret_key_size() {
            return Err(PqcError::InvalidKey);
        }
        Ok(Self {
            data: SecureZeroingVec::from_vec(bytes.to_vec()),
            params,
        })
    }

    /// Get bytes
    pub fn as_bytes(&self) -> &[u8] {
        self.data.as_slice()
    }

    /// Get seed component
    pub fn sk_seed(&self) -> &[u8] {
        &self.data.as_slice()[..self.params.n]
    }

    /// Get PRF key
    pub fn sk_prf(&self) -> &[u8] {
        let n = self.params.n;
        &self.data.as_slice()[n..2 * n]
    }

    /// Get public seed
    pub fn pk_seed(&self) -> &[u8] {
        let n = self.params.n;
        &self.data.as_slice()[2 * n..3 * n]
    }

    /// Get public root
    pub fn pk_root(&self) -> &[u8] {
        let n = self.params.n;
        &self.data.as_slice()[3 * n..]
    }

    /// Get parameters
    pub fn params(&self) -> SlhDsaParams {
        self.params
    }
}

/// SLH-DSA Signature
#[derive(Clone)]
pub struct SlhDsaSignature {
    data: Vec<u8>,
    params: SlhDsaParams,
}

impl SlhDsaSignature {
    /// Create from bytes
    pub fn from_bytes(bytes: &[u8], params: SlhDsaParams) -> PqcResult<Self> {
        Ok(Self {
            data: bytes.to_vec(),
            params,
        })
    }

    /// Get bytes
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Get parameters
    pub fn params(&self) -> SlhDsaParams {
        self.params
    }
}

// ============================================================================
// Core Algorithm
// ============================================================================

/// Core SLH-DSA implementation
pub struct SlhDsaCore;

impl SlhDsaCore {
    /// Key generation
    pub fn keygen(params: SlhDsaParams) -> PqcResult<(SlhDsaPublicKey, SlhDsaSecretKey)> {
        let mut rng = rand::rng();
        let n = params.n;

        let mut sk_seed = vec![0u8; n];
        let mut sk_prf = vec![0u8; n];
        let mut pk_seed = vec![0u8; n];

        rng.fill(&mut sk_seed[..]);
        rng.fill(&mut sk_prf[..]);
        rng.fill(&mut pk_seed[..]);

        Self::keygen_internal(params, &sk_seed, &sk_prf, &pk_seed)
    }

    /// Internal keygen with explicit randomness
    pub fn keygen_internal(
        params: SlhDsaParams,
        sk_seed: &[u8],
        sk_prf: &[u8],
        pk_seed: &[u8],
    ) -> PqcResult<(SlhDsaPublicKey, SlhDsaSecretKey)> {
        let n = params.n;
        let hp = params.hp;
        let d = params.d;

        // Compute root of top-level XMSS tree
        let mut adrs = Address::new();
        adrs.set_layer((d - 1) as u32);
        adrs.set_tree(0);

        let pk_root = xmss_node(sk_seed, pk_seed, 0, hp as u32, &mut adrs, &params);

        // Assemble keys
        let mut pk_bytes = Vec::with_capacity(2 * n);
        pk_bytes.extend_from_slice(pk_seed);
        pk_bytes.extend(pk_root);

        let mut sk_bytes = Vec::with_capacity(4 * n);
        sk_bytes.extend_from_slice(sk_seed);
        sk_bytes.extend_from_slice(sk_prf);
        sk_bytes.extend_from_slice(pk_seed);
        sk_bytes.extend_from_slice(&pk_bytes[n..]);

        Ok((
            SlhDsaPublicKey {
                data: pk_bytes,
                params,
            },
            SlhDsaSecretKey {
                data: SecureZeroingVec::from_vec(sk_bytes),
                params,
            },
        ))
    }

    /// Sign message
    pub fn sign(sk: &SlhDsaSecretKey, message: &[u8]) -> PqcResult<SlhDsaSignature> {
        let params = sk.params;
        let n = params.n;
        let d = params.d;
        let hp = params.hp;
        let fors_k = params.fors_k;
        let fors_a = params.fors_a;

        let sk_seed = sk.sk_seed();
        let sk_prf = sk.sk_prf();
        let pk_seed = sk.pk_seed();
        let pk_root = sk.pk_root();

        // Generate randomness
        let mut rng = rand::rng();
        let mut opt_rand = vec![0u8; n];
        rng.fill(&mut opt_rand[..]);

        // R = PRFmsg(SK.prf, opt_rand, M)
        let r = prf_msg(sk_prf, &opt_rand, message, n);

        // Hash message
        let digest = hash_msg(&r, pk_seed, pk_root, message, &params);

        // Extract FORS indices and tree index
        let md = &digest[..(fors_k * fors_a + 7) / 8];
        let idx_tree_bytes = &digest[(fors_k * fors_a + 7) / 8..];

        let mut idx_tree = 0u64;
        for (i, &b) in idx_tree_bytes.iter().take(8).enumerate() {
            idx_tree |= (b as u64) << (8 * i);
        }
        idx_tree &= (1u64 << (params.h - hp)) - 1;

        let idx_leaf = (idx_tree & ((1 << hp) - 1)) as u32;
        idx_tree >>= hp;

        // Start signature with randomness
        let mut sig = r;

        // FORS signature
        let mut adrs = Address::new();
        adrs.set_layer(0);
        adrs.set_tree(idx_tree);
        adrs.set_keypair(idx_leaf);

        let fors_sig = fors_sign(md, sk_seed, pk_seed, &mut adrs, &params);

        // FORS public key (compute before extending sig to avoid borrow issue)
        let fors_pk = fors_pk_from_sig(&fors_sig, md, pk_seed, &mut adrs, &params);
        sig.extend(fors_sig);

        // Hypertree signature
        let mut root = fors_pk;
        let mut tree = idx_tree;
        let mut leaf = idx_leaf;

        for layer in 0..d {
            adrs.set_layer(layer as u32);
            adrs.set_tree(tree);

            let xmss_sig = xmss_sign(&root, sk_seed, pk_seed, leaf, &mut adrs, &params);

            if layer < d - 1 {
                root = xmss_root_from_sig(&xmss_sig, &root, pk_seed, leaf, &mut adrs, &params);
                leaf = (tree & ((1 << hp) - 1)) as u32;
                tree >>= hp;
            }
            sig.extend(xmss_sig);
        }

        Ok(SlhDsaSignature { data: sig, params })
    }

    /// Verify signature
    pub fn verify(pk: &SlhDsaPublicKey, message: &[u8], sig: &SlhDsaSignature) -> bool {
        let params = pk.params;
        let n = params.n;
        let d = params.d;
        let hp = params.hp;
        let fors_k = params.fors_k;
        let fors_a = params.fors_a;
        let wots_len = params.wots_len();

        let pk_seed = pk.pk_seed();
        let pk_root = pk.pk_root();

        let sig_bytes = sig.as_bytes();
        if sig_bytes.len() < n {
            return false;
        }

        // Extract randomness
        let r = &sig_bytes[..n];

        // Hash message
        let digest = hash_msg(r, pk_seed, pk_root, message, &params);

        // Extract indices
        let md = &digest[..(fors_k * fors_a + 7) / 8];
        let idx_tree_bytes = &digest[(fors_k * fors_a + 7) / 8..];

        let mut idx_tree = 0u64;
        for (i, &b) in idx_tree_bytes.iter().take(8).enumerate() {
            idx_tree |= (b as u64) << (8 * i);
        }
        idx_tree &= (1u64 << (params.h - hp)) - 1;

        let idx_leaf = (idx_tree & ((1 << hp) - 1)) as u32;
        idx_tree >>= hp;

        // FORS signature
        let fors_sig_start = n;
        let fors_sig_len = fors_k * (fors_a + 1) * n;
        if sig_bytes.len() < fors_sig_start + fors_sig_len {
            return false;
        }
        let fors_sig = &sig_bytes[fors_sig_start..fors_sig_start + fors_sig_len];

        let mut adrs = Address::new();
        adrs.set_layer(0);
        adrs.set_tree(idx_tree);
        adrs.set_keypair(idx_leaf);

        let mut root = fors_pk_from_sig(fors_sig, md, pk_seed, &mut adrs, &params);

        // Verify hypertree
        let xmss_sig_len = hp * n + wots_len * n;
        let mut sig_offset = fors_sig_start + fors_sig_len;
        let mut tree = idx_tree;
        let mut leaf = idx_leaf;

        for layer in 0..d {
            if sig_bytes.len() < sig_offset + xmss_sig_len {
                return false;
            }

            let xmss_sig = &sig_bytes[sig_offset..sig_offset + xmss_sig_len];
            sig_offset += xmss_sig_len;

            adrs.set_layer(layer as u32);
            adrs.set_tree(tree);

            root = xmss_root_from_sig(xmss_sig, &root, pk_seed, leaf, &mut adrs, &params);

            if layer < d - 1 {
                leaf = (tree & ((1 << hp) - 1)) as u32;
                tree >>= hp;
            }
        }

        // Compare computed root with public key root
        ConstantTime::ct_eq(&root, pk_root)
    }
}

// ============================================================================
// Type-Safe Variants
// ============================================================================

/// SLH-DSA-SHAKE-128f (fast, Level 1)
pub struct SlhDsa128f;

impl SlhDsa128f {
    /// Parameters
    pub const PARAMS: SlhDsaParams = SLH_DSA_128F_PARAMS;

    /// Generate key pair
    pub fn keygen() -> PqcResult<(SlhDsaPublicKey, SlhDsaSecretKey)> {
        SlhDsaCore::keygen(Self::PARAMS)
    }

    /// Sign message
    pub fn sign(sk: &SlhDsaSecretKey, message: &[u8]) -> PqcResult<SlhDsaSignature> {
        SlhDsaCore::sign(sk, message)
    }

    /// Verify signature
    pub fn verify(pk: &SlhDsaPublicKey, message: &[u8], sig: &SlhDsaSignature) -> bool {
        SlhDsaCore::verify(pk, message, sig)
    }
}

/// SLH-DSA-SHAKE-128s (small, Level 1)
pub struct SlhDsa128s;

impl SlhDsa128s {
    pub const PARAMS: SlhDsaParams = SLH_DSA_128S_PARAMS;

    pub fn keygen() -> PqcResult<(SlhDsaPublicKey, SlhDsaSecretKey)> {
        SlhDsaCore::keygen(Self::PARAMS)
    }

    pub fn sign(sk: &SlhDsaSecretKey, message: &[u8]) -> PqcResult<SlhDsaSignature> {
        SlhDsaCore::sign(sk, message)
    }

    pub fn verify(pk: &SlhDsaPublicKey, message: &[u8], sig: &SlhDsaSignature) -> bool {
        SlhDsaCore::verify(pk, message, sig)
    }
}

/// SLH-DSA-SHAKE-192f (fast, Level 3)
pub struct SlhDsa192f;

impl SlhDsa192f {
    pub const PARAMS: SlhDsaParams = SLH_DSA_192F_PARAMS;

    pub fn keygen() -> PqcResult<(SlhDsaPublicKey, SlhDsaSecretKey)> {
        SlhDsaCore::keygen(Self::PARAMS)
    }

    pub fn sign(sk: &SlhDsaSecretKey, message: &[u8]) -> PqcResult<SlhDsaSignature> {
        SlhDsaCore::sign(sk, message)
    }

    pub fn verify(pk: &SlhDsaPublicKey, message: &[u8], sig: &SlhDsaSignature) -> bool {
        SlhDsaCore::verify(pk, message, sig)
    }
}

/// SLH-DSA-SHAKE-192s (small, Level 3)
pub struct SlhDsa192s;

impl SlhDsa192s {
    pub const PARAMS: SlhDsaParams = SLH_DSA_192S_PARAMS;

    pub fn keygen() -> PqcResult<(SlhDsaPublicKey, SlhDsaSecretKey)> {
        SlhDsaCore::keygen(Self::PARAMS)
    }

    pub fn sign(sk: &SlhDsaSecretKey, message: &[u8]) -> PqcResult<SlhDsaSignature> {
        SlhDsaCore::sign(sk, message)
    }

    pub fn verify(pk: &SlhDsaPublicKey, message: &[u8], sig: &SlhDsaSignature) -> bool {
        SlhDsaCore::verify(pk, message, sig)
    }
}

/// SLH-DSA-SHAKE-256f (fast, Level 5)
pub struct SlhDsa256f;

impl SlhDsa256f {
    pub const PARAMS: SlhDsaParams = SLH_DSA_256F_PARAMS;

    pub fn keygen() -> PqcResult<(SlhDsaPublicKey, SlhDsaSecretKey)> {
        SlhDsaCore::keygen(Self::PARAMS)
    }

    pub fn sign(sk: &SlhDsaSecretKey, message: &[u8]) -> PqcResult<SlhDsaSignature> {
        SlhDsaCore::sign(sk, message)
    }

    pub fn verify(pk: &SlhDsaPublicKey, message: &[u8], sig: &SlhDsaSignature) -> bool {
        SlhDsaCore::verify(pk, message, sig)
    }
}

/// SLH-DSA-SHAKE-256s (small, Level 5)
pub struct SlhDsa256s;

impl SlhDsa256s {
    pub const PARAMS: SlhDsaParams = SLH_DSA_256S_PARAMS;

    pub fn keygen() -> PqcResult<(SlhDsaPublicKey, SlhDsaSecretKey)> {
        SlhDsaCore::keygen(Self::PARAMS)
    }

    pub fn sign(sk: &SlhDsaSecretKey, message: &[u8]) -> PqcResult<SlhDsaSignature> {
        SlhDsaCore::sign(sk, message)
    }

    pub fn verify(pk: &SlhDsaPublicKey, message: &[u8], sig: &SlhDsaSignature) -> bool {
        SlhDsaCore::verify(pk, message, sig)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slh_dsa_128f_sign_verify() {
        let (pk, sk) = SlhDsa128f::keygen().expect("keygen failed");
        let message = b"Test message for SLH-DSA";

        assert_eq!(pk.as_bytes().len(), SLH_DSA_128F_PARAMS.public_key_size());
        assert_eq!(sk.as_bytes().len(), SLH_DSA_128F_PARAMS.secret_key_size());

        let sig = SlhDsa128f::sign(&sk, message).expect("signing failed");
        assert!(
            SlhDsa128f::verify(&pk, message, &sig),
            "verification failed"
        );
    }

    #[test]
    fn test_wots_roundtrip() {
        let params = SLH_DSA_128F_PARAMS;
        let n = params.n;
        let sk_seed = [1u8; 16];
        let pk_seed = [3u8; 16];
        let msg = [0xAB_u8; 16];

        let mut adrs = Address::new();
        adrs.set_layer(0);
        adrs.set_tree(42);
        adrs.set_keypair(5);

        let mut adrs_gen = adrs.clone();
        let pk_gen = wots_pk_gen(&sk_seed, &pk_seed, &mut adrs_gen, &params);

        let mut adrs_sign = adrs.clone();
        let wots_sig = wots_sign(&msg, &sk_seed, &pk_seed, &mut adrs_sign, &params);
        let mut adrs_verify = adrs.clone();
        let pk_from_sig = wots_pk_from_sig(&wots_sig, &msg, &pk_seed, &mut adrs_verify, &params);

        assert_eq!(pk_gen.len(), n);
        assert_eq!(pk_from_sig.len(), n);
        assert_eq!(
            pk_gen, pk_from_sig,
            "WOTS+ pk mismatch: pk_gen != pk_from_sig"
        );
    }

    #[test]
    fn test_xmss_roundtrip() {
        let params = SLH_DSA_128F_PARAMS;
        let hp = params.hp;
        let sk_seed = [1u8; 16];
        let pk_seed = [3u8; 16];
        let msg = [0xCD_u8; 16];

        let mut adrs = Address::new();
        adrs.set_layer(0);
        adrs.set_tree(0);

        let mut adrs_tree = adrs.clone();
        let root = xmss_node(&sk_seed, &pk_seed, 0, hp as u32, &mut adrs_tree, &params);

        // Test at leaf 0
        let mut adrs_sign = adrs.clone();
        let xmss_sig = xmss_sign(&msg, &sk_seed, &pk_seed, 0, &mut adrs_sign, &params);
        let mut adrs_verify = adrs.clone();
        let recovered = xmss_root_from_sig(&xmss_sig, &msg, &pk_seed, 0, &mut adrs_verify, &params);
        assert_eq!(root, recovered, "XMSS root mismatch at leaf 0");

        // Test at leaf 3
        let mut adrs_sign = adrs.clone();
        let xmss_sig = xmss_sign(&msg, &sk_seed, &pk_seed, 3, &mut adrs_sign, &params);
        let mut adrs_verify = adrs.clone();
        let recovered = xmss_root_from_sig(&xmss_sig, &msg, &pk_seed, 3, &mut adrs_verify, &params);
        assert_eq!(root, recovered, "XMSS root mismatch at leaf 3");
    }

    #[test]
    fn test_slh_dsa_wrong_message() {
        let (pk, sk) = SlhDsa128f::keygen().expect("keygen failed");
        let message = b"Original message";
        let wrong_message = b"Wrong message";

        let sig = SlhDsa128f::sign(&sk, message).expect("signing failed");
        assert!(
            !SlhDsa128f::verify(&pk, wrong_message, &sig),
            "should reject wrong message"
        );
    }

    #[test]
    fn test_key_sizes() {
        // 128f
        assert_eq!(SLH_DSA_128F_PARAMS.public_key_size(), 32);
        assert_eq!(SLH_DSA_128F_PARAMS.secret_key_size(), 64);

        // 192f
        assert_eq!(SLH_DSA_192F_PARAMS.public_key_size(), 48);
        assert_eq!(SLH_DSA_192F_PARAMS.secret_key_size(), 96);

        // 256f
        assert_eq!(SLH_DSA_256F_PARAMS.public_key_size(), 64);
        assert_eq!(SLH_DSA_256F_PARAMS.secret_key_size(), 128);
    }

    #[test]
    fn test_wots_len() {
        // For w=16, n=16: len1 = 32, len2 = 3, total = 35
        assert_eq!(SLH_DSA_128F_PARAMS.wots_len(), 35);
    }

    #[test]
    fn test_deterministic_keygen() {
        let sk_seed = [1u8; 16];
        let sk_prf = [2u8; 16];
        let pk_seed = [3u8; 16];

        let (pk1, sk1) =
            SlhDsaCore::keygen_internal(SLH_DSA_128F_PARAMS, &sk_seed, &sk_prf, &pk_seed).unwrap();
        let (pk2, sk2) =
            SlhDsaCore::keygen_internal(SLH_DSA_128F_PARAMS, &sk_seed, &sk_prf, &pk_seed).unwrap();

        assert_eq!(pk1.as_bytes(), pk2.as_bytes());
        assert_eq!(sk1.as_bytes(), sk2.as_bytes());
    }
}
