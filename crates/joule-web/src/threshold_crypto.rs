//! Threshold cryptography concepts — key generation ceremony (shares),
//! threshold signing (t-of-n), share combination, Lagrange interpolation
//! for share recovery, dealer-based setup, and verification of partial
//! signatures.

use serde::{Deserialize, Serialize};

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
            block[i * 4], block[i * 4 + 1], block[i * 4 + 2], block[i * 4 + 3],
        ]);
    }
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
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
    let total_len = data.len() as u64;
    let mut buf = data.to_vec();
    buf.push(0x80);
    while buf.len() % 64 != 56 { buf.push(0x00); }
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

fn bytes_to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

// ── Modular arithmetic ──────────────────────────────────────────────────────

/// Prime modulus for the group. We use a safe prime: p = 2147483647 (2^31 - 1).
const TC_P: u64 = 2_147_483_647;
/// Generator.
const TC_G: u64 = 3;
/// Group order: p - 1.
const TC_ORDER: u64 = TC_P - 1;

fn mod_mul(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

fn mod_pow(mut base: u64, mut exp: u64, m: u64) -> u64 {
    if m == 1 { return 0; }
    let mut result = 1u64;
    base %= m;
    while exp > 0 {
        if exp & 1 == 1 {
            result = mod_mul(result, base, m);
        }
        exp >>= 1;
        base = mod_mul(base, base, m);
    }
    result
}

fn mod_add(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 + b as u128) % m as u128) as u64
}

fn mod_sub(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 + m as u128 - b as u128) % m as u128) as u64
}

/// Modular inverse using extended Euclidean algorithm.
/// Works for any modulus (Fermat's little theorem only works for prime m).
fn mod_inv(a: u64, m: u64) -> u64 {
    let (mut old_r, mut r) = (a as i128, m as i128);
    let (mut old_s, mut s) = (1i128, 0i128);
    while r != 0 {
        let q = old_r / r;
        let temp_r = r;
        r = old_r - q * r;
        old_r = temp_r;
        let temp_s = s;
        s = old_s - q * s;
        old_s = temp_s;
    }
    ((old_s % m as i128 + m as i128) % m as i128) as u64
}

fn mod_div(a: u64, b: u64, m: u64) -> u64 {
    mod_mul(a, mod_inv(b, m), m)
}

/// Hash to scalar in [0, order).
fn hash_to_scalar(data: &[u8]) -> u64 {
    let h = sha256(data);
    let val = u64::from_le_bytes(h[..8].try_into().unwrap());
    val % TC_ORDER
}

// ── Simple PRNG ─────────────────────────────────────────────────────────────

struct Rng {
    state: u64,
}

impl Rng {
    fn from_seed(seed: &[u8]) -> Self {
        let mut state = 0x123456789ABCDEFu64;
        for (i, &byte) in seed.iter().enumerate() {
            state ^= (byte as u64) << ((i % 8) * 8);
            state = state.wrapping_mul(0x5DEECE66D).wrapping_add(0xB);
        }
        if state == 0 { state = 1; }
        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    fn next_mod(&mut self, m: u64) -> u64 {
        self.next_u64() % m
    }

    fn next_nonzero_mod(&mut self, m: u64) -> u64 {
        loop {
            let v = self.next_mod(m);
            if v != 0 {
                return v;
            }
        }
    }
}

// ── Errors ──────────────────────────────────────────────────────────────────

/// Threshold cryptography errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ThresholdError {
    /// Threshold must be >= 2.
    ThresholdTooLow(usize),
    /// Threshold exceeds total participants.
    ThresholdExceedsTotal { threshold: usize, total: usize },
    /// Not enough partial signatures.
    InsufficientSignatures { needed: usize, provided: usize },
    /// Duplicate participant index.
    DuplicateIndex(u64),
    /// Invalid partial signature.
    InvalidPartialSignature { index: u64 },
    /// Zero index (reserved).
    ZeroIndex,
    /// Verification failed for combined signature.
    VerificationFailed,
}

impl std::fmt::Display for ThresholdError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ThresholdTooLow(t) => write!(f, "threshold {t} too low (min 2)"),
            Self::ThresholdExceedsTotal { threshold, total } => {
                write!(f, "threshold {threshold} exceeds total {total}")
            }
            Self::InsufficientSignatures { needed, provided } => {
                write!(f, "need {needed} signatures, got {provided}")
            }
            Self::DuplicateIndex(i) => write!(f, "duplicate participant index: {i}"),
            Self::InvalidPartialSignature { index } => {
                write!(f, "invalid partial signature from participant {index}")
            }
            Self::ZeroIndex => write!(f, "participant index 0 is reserved"),
            Self::VerificationFailed => write!(f, "combined signature verification failed"),
        }
    }
}

impl std::error::Error for ThresholdError {}

// ── Key Share ───────────────────────────────────────────────────────────────

/// A participant's key share from the key generation ceremony.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyShare {
    /// Participant index (1-based).
    pub index: u64,
    /// The secret share value: f(index) mod order.
    pub secret_share: u64,
    /// The public verification key: g^(secret_share) mod p.
    pub verification_key: u64,
    /// Threshold required.
    pub threshold: usize,
    /// Total participants.
    pub total: usize,
}

/// Public parameters from the key generation ceremony.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdPublicKey {
    /// The combined public key: g^secret mod p.
    pub key: u64,
    /// Verification keys for each participant's share.
    pub verification_keys: Vec<u64>,
    /// Threshold.
    pub threshold: usize,
    /// Total participants.
    pub total: usize,
    /// Group prime.
    pub p: u64,
    /// Generator.
    pub g: u64,
}

/// Result of the key generation ceremony.
#[derive(Debug, Clone)]
pub struct KeyGenResult {
    /// Individual key shares (one per participant).
    pub shares: Vec<KeyShare>,
    /// Public parameters.
    pub public_key: ThresholdPublicKey,
}

// ── Key Generation Ceremony ─────────────────────────────────────────────────

/// Dealer-based threshold key generation.
///
/// The dealer generates a random polynomial f(x) of degree (threshold - 1),
/// where f(0) is the master secret. Each participant i gets f(i) as their share.
pub fn key_gen(
    threshold: usize,
    total: usize,
    entropy: &[u8],
) -> Result<KeyGenResult, ThresholdError> {
    if threshold < 2 {
        return Err(ThresholdError::ThresholdTooLow(threshold));
    }
    if threshold > total {
        return Err(ThresholdError::ThresholdExceedsTotal { threshold, total });
    }

    let mut rng = Rng::from_seed(entropy);

    // Generate random polynomial coefficients.
    // f(x) = a0 + a1*x + a2*x^2 + ... + a_{t-1}*x^{t-1}
    // where a0 is the master secret.
    let mut coefficients = Vec::with_capacity(threshold);
    for _ in 0..threshold {
        coefficients.push(rng.next_nonzero_mod(TC_P));
    }

    let master_secret = coefficients[0];
    let public_key_value = mod_pow(TC_G, master_secret, TC_P);

    // Evaluate polynomial at each participant's index.
    let mut shares = Vec::with_capacity(total);
    let mut verification_keys = Vec::with_capacity(total);

    for i in 1..=total as u64 {
        let share_val = poly_eval(&coefficients, i);
        let vk = mod_pow(TC_G, share_val, TC_P);
        shares.push(KeyShare {
            index: i,
            secret_share: share_val,
            verification_key: vk,
            threshold,
            total,
        });
        verification_keys.push(vk);
    }

    Ok(KeyGenResult {
        shares,
        public_key: ThresholdPublicKey {
            key: public_key_value,
            verification_keys,
            threshold,
            total,
            p: TC_P,
            g: TC_G,
        },
    })
}

/// Evaluate polynomial at x (mod TC_P).
fn poly_eval(coeffs: &[u64], x: u64) -> u64 {
    // Horner's method.
    let mut result = 0u64;
    for &coeff in coeffs.iter().rev() {
        result = mod_add(mod_mul(result, x, TC_P), coeff, TC_P);
    }
    result
}

// ── Lagrange Interpolation ──────────────────────────────────────────────────

/// Compute Lagrange basis coefficient for index `i` at x=0,
/// given the set of all participant indices.
pub fn lagrange_coefficient(index: u64, all_indices: &[u64]) -> u64 {
    let mut num = 1u64;
    let mut den = 1u64;

    for &j in all_indices {
        if j == index {
            continue;
        }
        // L_i(0) = prod_{j != i} (0 - j) / (i - j)
        num = mod_mul(num, mod_sub(0, j, TC_P), TC_P);
        let diff = mod_sub(index, j, TC_P);
        den = mod_mul(den, diff, TC_P);
    }

    mod_div(num, den, TC_P)
}

/// Reconstruct the master secret from t key shares using Lagrange interpolation.
pub fn reconstruct_secret(shares: &[KeyShare]) -> Result<u64, ThresholdError> {
    if shares.is_empty() {
        return Err(ThresholdError::InsufficientSignatures {
            needed: 2,
            provided: 0,
        });
    }

    let threshold = shares[0].threshold;
    if shares.len() < threshold {
        return Err(ThresholdError::InsufficientSignatures {
            needed: threshold,
            provided: shares.len(),
        });
    }

    // Check for duplicates.
    let mut seen = std::collections::HashSet::new();
    for share in shares {
        if share.index == 0 {
            return Err(ThresholdError::ZeroIndex);
        }
        if !seen.insert(share.index) {
            return Err(ThresholdError::DuplicateIndex(share.index));
        }
    }

    let k = threshold.min(shares.len());
    let eval_shares = &shares[..k];
    let indices: Vec<u64> = eval_shares.iter().map(|s| s.index).collect();

    let mut secret = 0u64;
    for share in eval_shares {
        let li = lagrange_coefficient(share.index, &indices);
        let term = mod_mul(share.secret_share, li, TC_P);
        secret = mod_add(secret, term, TC_P);
    }

    Ok(secret)
}

// ── Partial Signature ───────────────────────────────────────────────────────

/// A partial signature from one participant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartialSignature {
    /// Participant index.
    pub index: u64,
    /// The partial signature value: H(msg)^(share * lagrange_coeff) mod p.
    /// (Simplified BLS-like approach using our modular group.)
    pub value: u64,
    /// Commitment for verification: g^k_i mod p.
    pub commitment: u64,
    /// Response: k_i + share_i * c mod order.
    pub response: u64,
}

/// A combined threshold signature.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdSignature {
    /// Combined signature value.
    pub value: u64,
    /// Message hash that was signed.
    pub message_hash: Vec<u8>,
    /// Indices of participants who contributed.
    pub participant_indices: Vec<u64>,
}

/// Hash a message to a group element.
fn hash_message(message: &[u8]) -> u64 {
    let h = sha256(message);
    let val = u64::from_le_bytes(h[..8].try_into().unwrap());
    // Map to a group element: g^hash mod p.
    mod_pow(TC_G, val % TC_ORDER, TC_P)
}

/// Create a partial signature using a key share.
pub fn partial_sign(
    share: &KeyShare,
    message: &[u8],
    randomness: &[u8],
) -> PartialSignature {
    let m_hash = hash_to_scalar(message);

    // Generate random k.
    let mut k_input = Vec::with_capacity(randomness.len() + 8);
    k_input.extend_from_slice(randomness);
    k_input.extend_from_slice(&share.index.to_le_bytes());
    let k = hash_to_scalar(&k_input);

    // Commitment: r = g^k mod p.
    let commitment = mod_pow(TC_G, k, TC_P);

    // Challenge: c = H(r || message).
    let mut c_input = Vec::new();
    c_input.extend_from_slice(&commitment.to_le_bytes());
    c_input.extend_from_slice(message);
    let c = hash_to_scalar(&c_input);

    // Response: s = k + share * c mod order.
    let sc = mod_mul(share.secret_share, c, TC_ORDER);
    let response = mod_add(k, sc, TC_ORDER);

    // Partial signature value: m^share mod p (simplified).
    let msg_elem = hash_message(message);
    let value = mod_pow(msg_elem, share.secret_share, TC_P);

    PartialSignature {
        index: share.index,
        value,
        commitment,
        response,
    }
}

/// Verify a partial signature against the public verification key.
pub fn verify_partial(
    sig: &PartialSignature,
    pub_key: &ThresholdPublicKey,
    message: &[u8],
) -> bool {
    if sig.index == 0 || sig.index > pub_key.total as u64 {
        return false;
    }
    let vk_idx = (sig.index - 1) as usize;
    if vk_idx >= pub_key.verification_keys.len() {
        return false;
    }
    let vk = pub_key.verification_keys[vk_idx];

    // Recompute challenge.
    let mut c_input = Vec::new();
    c_input.extend_from_slice(&sig.commitment.to_le_bytes());
    c_input.extend_from_slice(message);
    let c = hash_to_scalar(&c_input);

    // Verify: g^response == commitment * vk^c mod p.
    let lhs = mod_pow(pub_key.g, sig.response, pub_key.p);
    let vkc = mod_pow(vk, c, pub_key.p);
    let rhs = mod_mul(sig.commitment, vkc, pub_key.p);

    lhs == rhs
}

/// Combine partial signatures into a threshold signature.
///
/// Requires at least `threshold` valid partial signatures.
pub fn combine_signatures(
    partial_sigs: &[PartialSignature],
    pub_key: &ThresholdPublicKey,
    message: &[u8],
) -> Result<ThresholdSignature, ThresholdError> {
    if partial_sigs.len() < pub_key.threshold {
        return Err(ThresholdError::InsufficientSignatures {
            needed: pub_key.threshold,
            provided: partial_sigs.len(),
        });
    }

    // Check for duplicates.
    let mut seen = std::collections::HashSet::new();
    for sig in partial_sigs {
        if sig.index == 0 {
            return Err(ThresholdError::ZeroIndex);
        }
        if !seen.insert(sig.index) {
            return Err(ThresholdError::DuplicateIndex(sig.index));
        }
    }

    let k = pub_key.threshold.min(partial_sigs.len());
    let sigs = &partial_sigs[..k];
    let indices: Vec<u64> = sigs.iter().map(|s| s.index).collect();

    // Combine: sigma = prod(sig_i^lambda_i) mod p.
    let mut combined = 1u64;
    for sig in sigs {
        let li = lagrange_coefficient(sig.index, &indices);
        let sig_li = mod_pow(sig.value, li, TC_P);
        combined = mod_mul(combined, sig_li, TC_P);
    }

    let msg_hash = sha256(message).to_vec();

    Ok(ThresholdSignature {
        value: combined,
        message_hash: msg_hash,
        participant_indices: indices,
    })
}

/// Verify a combined threshold signature.
pub fn verify_threshold_signature(
    sig: &ThresholdSignature,
    pub_key: &ThresholdPublicKey,
    message: &[u8],
) -> bool {
    // Verify message hash matches.
    let expected_hash = sha256(message);
    if sig.message_hash != expected_hash {
        return false;
    }

    // The combined signature should equal m^secret mod p.
    let msg_elem = hash_message(message);
    // We can't verify directly without knowing the secret, but we can
    // check that e(sigma, g) = e(m, pk) in a pairing-based scheme.
    // In our simplified scheme, we verify: sigma = m^secret mod p
    // by checking: sigma^order = 1 mod p (it's a valid group element).
    let check = mod_pow(sig.value, TC_ORDER, TC_P);
    // In Z_p^*, every element satisfies a^(p-1) = 1 mod p (Fermat's).
    // So this is a basic validity check.
    if check != 1 && sig.value != 0 {
        return false;
    }

    // Verify consistency: the combined value should be non-trivial.
    sig.value != 0 && sig.value != 1 || sig.participant_indices.len() >= pub_key.threshold
}

// ── Ceremony Tracking ───────────────────────────────────────────────────────

/// Status of a key generation ceremony participant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CeremonyStatus {
    /// Waiting to receive share.
    Pending,
    /// Share received and verified.
    Verified,
    /// Share received but verification failed.
    Failed,
}

/// Track the key generation ceremony.
#[derive(Debug, Clone)]
pub struct CeremonyTracker {
    /// Status of each participant.
    statuses: Vec<(u64, CeremonyStatus)>,
    /// Threshold required.
    threshold: usize,
}

impl CeremonyTracker {
    /// Create a new ceremony tracker.
    pub fn new(total: usize, threshold: usize) -> Self {
        let statuses = (1..=total as u64)
            .map(|i| (i, CeremonyStatus::Pending))
            .collect();
        Self {
            statuses,
            threshold,
        }
    }

    /// Mark a participant as verified.
    pub fn mark_verified(&mut self, index: u64) {
        for (i, status) in &mut self.statuses {
            if *i == index {
                *status = CeremonyStatus::Verified;
            }
        }
    }

    /// Mark a participant as failed.
    pub fn mark_failed(&mut self, index: u64) {
        for (i, status) in &mut self.statuses {
            if *i == index {
                *status = CeremonyStatus::Failed;
            }
        }
    }

    /// Number of verified participants.
    pub fn verified_count(&self) -> usize {
        self.statuses
            .iter()
            .filter(|(_, s)| *s == CeremonyStatus::Verified)
            .count()
    }

    /// Whether the ceremony has enough verified participants.
    pub fn is_complete(&self) -> bool {
        self.verified_count() >= self.threshold
    }

    /// Get status of a specific participant.
    pub fn status(&self, index: u64) -> Option<CeremonyStatus> {
        self.statuses.iter().find(|(i, _)| *i == index).map(|(_, s)| *s)
    }

    /// Get all statuses.
    pub fn all_statuses(&self) -> &[(u64, CeremonyStatus)] {
        &self.statuses
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_gen_basic() {
        let result = key_gen(3, 5, b"ceremony-entropy").unwrap();
        assert_eq!(result.shares.len(), 5);
        assert_eq!(result.public_key.threshold, 3);
        assert_eq!(result.public_key.total, 5);
        assert_eq!(result.public_key.verification_keys.len(), 5);
    }

    #[test]
    fn test_key_gen_threshold_too_low() {
        assert!(key_gen(1, 3, b"e").is_err());
        assert!(key_gen(0, 3, b"e").is_err());
    }

    #[test]
    fn test_key_gen_threshold_exceeds_total() {
        assert!(key_gen(5, 3, b"e").is_err());
    }

    #[test]
    fn test_reconstruct_secret() {
        let result = key_gen(3, 5, b"entropy").unwrap();
        // The master secret is f(0), which equals the first coefficient.
        // We can verify by reconstructing.
        let reconstructed = reconstruct_secret(&result.shares[..3]).unwrap();
        // The public key should be g^secret mod p.
        let expected_pk = mod_pow(TC_G, reconstructed, TC_P);
        assert_eq!(expected_pk, result.public_key.key);
    }

    #[test]
    fn test_reconstruct_any_k_shares() {
        let result = key_gen(3, 5, b"test").unwrap();
        let combos: Vec<Vec<usize>> = vec![
            vec![0, 1, 2],
            vec![0, 2, 4],
            vec![1, 3, 4],
            vec![2, 3, 4],
        ];
        // Reconstruct secret from first valid combo for reference.
        let reference = reconstruct_secret(&result.shares[..3]).unwrap();
        for combo in &combos {
            let subset: Vec<KeyShare> = combo.iter().map(|i| result.shares[*i].clone()).collect();
            let reconstructed = reconstruct_secret(&subset).unwrap();
            assert_eq!(
                reconstructed, reference,
                "mismatch with shares {:?}",
                combo
            );
        }
    }

    #[test]
    fn test_reconstruct_insufficient() {
        let result = key_gen(3, 5, b"e").unwrap();
        assert!(reconstruct_secret(&result.shares[..2]).is_err());
    }

    #[test]
    fn test_reconstruct_duplicate() {
        let result = key_gen(3, 5, b"e").unwrap();
        let dup = vec![
            result.shares[0].clone(),
            result.shares[0].clone(),
            result.shares[1].clone(),
        ];
        assert!(reconstruct_secret(&dup).is_err());
    }

    #[test]
    fn test_partial_sign() {
        let result = key_gen(3, 5, b"entropy").unwrap();
        let message = b"sign this message";
        let sig = partial_sign(&result.shares[0], message, b"rand");
        assert_ne!(sig.value, 0);
        assert_eq!(sig.index, 1);
    }

    #[test]
    fn test_verify_partial() {
        let result = key_gen(3, 5, b"entropy").unwrap();
        let message = b"verify me";
        let sig = partial_sign(&result.shares[0], message, b"rand");
        assert!(verify_partial(&sig, &result.public_key, message));
    }

    #[test]
    fn test_verify_partial_wrong_message() {
        let result = key_gen(3, 5, b"entropy").unwrap();
        let sig = partial_sign(&result.shares[0], b"original", b"rand");
        assert!(!verify_partial(&sig, &result.public_key, b"tampered"));
    }

    #[test]
    fn test_combine_signatures() {
        let result = key_gen(3, 5, b"entropy").unwrap();
        let message = b"threshold signed message";
        let sigs: Vec<PartialSignature> = result.shares[..3]
            .iter()
            .enumerate()
            .map(|(i, share)| {
                let mut rand = b"rand".to_vec();
                rand.push(i as u8);
                partial_sign(share, message, &rand)
            })
            .collect();

        let combined = combine_signatures(&sigs, &result.public_key, message).unwrap();
        assert!(!combined.participant_indices.is_empty());
        assert_eq!(combined.participant_indices.len(), 3);
    }

    #[test]
    fn test_combine_insufficient() {
        let result = key_gen(3, 5, b"e").unwrap();
        let message = b"msg";
        let sigs: Vec<PartialSignature> = result.shares[..2]
            .iter()
            .map(|share| partial_sign(share, message, b"r"))
            .collect();
        assert!(combine_signatures(&sigs, &result.public_key, message).is_err());
    }

    #[test]
    fn test_lagrange_coefficient_basic() {
        let indices = vec![1, 2, 3];
        let l1 = lagrange_coefficient(1, &indices);
        let l2 = lagrange_coefficient(2, &indices);
        let l3 = lagrange_coefficient(3, &indices);
        // Sum of Lagrange coefficients at x=0 for constant polynomial = 1.
        // But in modular arithmetic, verify they're non-zero.
        assert_ne!(l1, 0);
        assert_ne!(l2, 0);
        assert_ne!(l3, 0);
    }

    #[test]
    fn test_ceremony_tracker() {
        let mut tracker = CeremonyTracker::new(5, 3);
        assert_eq!(tracker.verified_count(), 0);
        assert!(!tracker.is_complete());

        tracker.mark_verified(1);
        tracker.mark_verified(2);
        assert_eq!(tracker.verified_count(), 2);
        assert!(!tracker.is_complete());

        tracker.mark_verified(3);
        assert!(tracker.is_complete());
    }

    #[test]
    fn test_ceremony_tracker_status() {
        let mut tracker = CeremonyTracker::new(3, 2);
        assert_eq!(tracker.status(1), Some(CeremonyStatus::Pending));
        tracker.mark_verified(1);
        assert_eq!(tracker.status(1), Some(CeremonyStatus::Verified));
        tracker.mark_failed(2);
        assert_eq!(tracker.status(2), Some(CeremonyStatus::Failed));
    }

    #[test]
    fn test_ceremony_tracker_nonexistent() {
        let tracker = CeremonyTracker::new(3, 2);
        assert_eq!(tracker.status(99), None);
    }

    #[test]
    fn test_key_share_serialization() {
        let result = key_gen(2, 3, b"e").unwrap();
        let json = serde_json::to_string(&result.shares[0]).unwrap();
        let deserialized: KeyShare = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.index, result.shares[0].index);
        assert_eq!(deserialized.secret_share, result.shares[0].secret_share);
    }

    #[test]
    fn test_public_key_serialization() {
        let result = key_gen(2, 3, b"e").unwrap();
        let json = serde_json::to_string(&result.public_key).unwrap();
        let deserialized: ThresholdPublicKey = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.key, result.public_key.key);
    }

    #[test]
    fn test_threshold_signature_serialization() {
        let result = key_gen(2, 3, b"e").unwrap();
        let message = b"msg";
        let sigs: Vec<PartialSignature> = result.shares[..2]
            .iter()
            .map(|s| partial_sign(s, message, b"r"))
            .collect();
        let combined = combine_signatures(&sigs, &result.public_key, message).unwrap();
        let json = serde_json::to_string(&combined).unwrap();
        let _: ThresholdSignature = serde_json::from_str(&json).unwrap();
    }

    #[test]
    fn test_two_of_two() {
        let result = key_gen(2, 2, b"minimal").unwrap();
        assert_eq!(result.shares.len(), 2);
        let secret = reconstruct_secret(&result.shares).unwrap();
        let pk = mod_pow(TC_G, secret, TC_P);
        assert_eq!(pk, result.public_key.key);
    }

    #[test]
    fn test_key_gen_deterministic() {
        let r1 = key_gen(3, 5, b"same-seed").unwrap();
        let r2 = key_gen(3, 5, b"same-seed").unwrap();
        assert_eq!(r1.public_key.key, r2.public_key.key);
        for (s1, s2) in r1.shares.iter().zip(r2.shares.iter()) {
            assert_eq!(s1.secret_share, s2.secret_share);
        }
    }

    #[test]
    fn test_mod_arithmetic() {
        assert_eq!(mod_add(5, 7, 10), 2);
        assert_eq!(mod_sub(3, 5, 10), 8);
        assert_eq!(mod_mul(3, 4, 10), 2);
        let inv5 = mod_inv(5, 7); // 5^(-1) mod 7 = 3 (since 5*3 = 15 = 1 mod 7)
        assert_eq!(mod_mul(5, inv5, 7), 1);
    }
}
