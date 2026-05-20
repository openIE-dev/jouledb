//! Commitment schemes — hash-based commitment (commit = H(value || nonce)),
//! commit/reveal protocol, Pedersen-like commitment (simplified over a prime group),
//! batch commitment, commitment equality proof, and serialization.

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

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() { return false; }
    let mut diff = 0u8;
    for i in 0..a.len() { diff |= a[i] ^ b[i]; }
    diff == 0
}

// ── Errors ──────────────────────────────────────────────────────────────────

/// Commitment scheme errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommitError {
    /// Invalid reveal: commitment does not match.
    RevealMismatch,
    /// Nonce too short.
    NonceTooShort { min: usize, got: usize },
    /// Empty value.
    EmptyValue,
    /// Already revealed.
    AlreadyRevealed,
    /// Not yet committed.
    NotCommitted,
    /// Invalid commitment data.
    InvalidData(String),
}

impl std::fmt::Display for CommitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RevealMismatch => write!(f, "commitment does not match revealed value"),
            Self::NonceTooShort { min, got } => {
                write!(f, "nonce too short: {got} bytes (min {min})")
            }
            Self::EmptyValue => write!(f, "value must not be empty"),
            Self::AlreadyRevealed => write!(f, "commitment already revealed"),
            Self::NotCommitted => write!(f, "no commitment to reveal"),
            Self::InvalidData(s) => write!(f, "invalid data: {s}"),
        }
    }
}

impl std::error::Error for CommitError {}

// ── Hash Commitment ─────────────────────────────────────────────────────────

/// A hash-based commitment: C = H(value || nonce).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HashCommitment {
    /// The commitment hash.
    pub hash: Vec<u8>,
}

impl HashCommitment {
    /// Hex string of the commitment.
    pub fn to_hex(&self) -> String {
        bytes_to_hex(&self.hash)
    }
}

/// Opening information for a hash commitment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HashOpening {
    /// The committed value.
    pub value: Vec<u8>,
    /// The nonce used.
    pub nonce: Vec<u8>,
}

/// Create a hash commitment: C = SHA256(value || nonce).
pub fn hash_commit(value: &[u8], nonce: &[u8]) -> Result<HashCommitment, CommitError> {
    if value.is_empty() {
        return Err(CommitError::EmptyValue);
    }
    if nonce.len() < 16 {
        return Err(CommitError::NonceTooShort {
            min: 16,
            got: nonce.len(),
        });
    }

    let mut data = Vec::with_capacity(value.len() + nonce.len());
    data.extend_from_slice(value);
    data.extend_from_slice(nonce);
    let hash = sha256(&data);

    Ok(HashCommitment {
        hash: hash.to_vec(),
    })
}

/// Verify a hash commitment against a revealed value and nonce.
pub fn hash_verify(
    commitment: &HashCommitment,
    value: &[u8],
    nonce: &[u8],
) -> bool {
    let mut data = Vec::with_capacity(value.len() + nonce.len());
    data.extend_from_slice(value);
    data.extend_from_slice(nonce);
    let expected = sha256(&data);
    constant_time_eq(&commitment.hash, &expected)
}

// ── Commit/Reveal Protocol ──────────────────────────────────────────────────

/// Protocol state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProtocolPhase {
    /// Waiting for commitment.
    AwaitingCommitment,
    /// Commitment received, awaiting reveal.
    AwaitingReveal,
    /// Revealed and verified.
    Verified,
    /// Revealed but verification failed.
    Failed,
}

/// A stateful commit/reveal protocol participant.
#[derive(Debug, Clone)]
pub struct CommitRevealProtocol {
    phase: ProtocolPhase,
    commitment: Option<HashCommitment>,
    opening: Option<HashOpening>,
}

impl CommitRevealProtocol {
    /// Create a new protocol instance.
    pub fn new() -> Self {
        Self {
            phase: ProtocolPhase::AwaitingCommitment,
            commitment: None,
            opening: None,
        }
    }

    /// Current phase.
    pub fn phase(&self) -> ProtocolPhase {
        self.phase
    }

    /// Submit a commitment.
    pub fn submit_commitment(&mut self, commitment: HashCommitment) -> Result<(), CommitError> {
        if self.phase != ProtocolPhase::AwaitingCommitment {
            return Err(CommitError::AlreadyRevealed);
        }
        self.commitment = Some(commitment);
        self.phase = ProtocolPhase::AwaitingReveal;
        Ok(())
    }

    /// Submit the opening (reveal phase).
    pub fn submit_opening(&mut self, opening: HashOpening) -> Result<bool, CommitError> {
        if self.phase != ProtocolPhase::AwaitingReveal {
            if self.phase == ProtocolPhase::AwaitingCommitment {
                return Err(CommitError::NotCommitted);
            }
            return Err(CommitError::AlreadyRevealed);
        }

        let commitment = self.commitment.as_ref().unwrap();
        let valid = hash_verify(commitment, &opening.value, &opening.nonce);

        if valid {
            self.phase = ProtocolPhase::Verified;
        } else {
            self.phase = ProtocolPhase::Failed;
        }
        self.opening = Some(opening);
        Ok(valid)
    }

    /// Get the revealed value (only available after successful verification).
    pub fn revealed_value(&self) -> Option<&[u8]> {
        if self.phase == ProtocolPhase::Verified {
            self.opening.as_ref().map(|o| o.value.as_slice())
        } else {
            None
        }
    }
}

impl Default for CommitRevealProtocol {
    fn default() -> Self {
        Self::new()
    }
}

// ── Pedersen-like Commitment ────────────────────────────────────────────────

/// A simplified Pedersen-like commitment operating over a prime modular group.
///
/// C = g^value * h^blinding mod p
///
/// Uses a small prime group for demonstration. In production, use an
/// elliptic curve group.
///
/// We use p = 2147483647 (Mersenne prime 2^31 - 1) with generators
/// g = 3, h = 5 (both primitive roots mod p).
const PEDERSEN_P: u64 = 2_147_483_647;
const PEDERSEN_G: u64 = 3;
const PEDERSEN_H: u64 = 5;

/// Modular exponentiation: base^exp mod modulus.
fn mod_pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    if modulus == 1 {
        return 0;
    }
    let mut result: u64 = 1;
    base %= modulus;
    while exp > 0 {
        if exp & 1 == 1 {
            result = mod_mul(result, base, modulus);
        }
        exp >>= 1;
        base = mod_mul(base, base, modulus);
    }
    result
}

/// Modular multiplication avoiding overflow via u128.
fn mod_mul(a: u64, b: u64, modulus: u64) -> u64 {
    ((a as u128 * b as u128) % modulus as u128) as u64
}

/// A Pedersen-like commitment value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PedersenCommitment {
    /// The commitment value C = g^v * h^r mod p.
    pub value: u64,
}

/// Opening for a Pedersen commitment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PedersenOpening {
    /// The committed value.
    pub committed_value: u64,
    /// The blinding factor.
    pub blinding: u64,
}

/// Create a Pedersen-like commitment.
pub fn pedersen_commit(value: u64, blinding: u64) -> PedersenCommitment {
    let gv = mod_pow(PEDERSEN_G, value % (PEDERSEN_P - 1), PEDERSEN_P);
    let hr = mod_pow(PEDERSEN_H, blinding % (PEDERSEN_P - 1), PEDERSEN_P);
    let c = mod_mul(gv, hr, PEDERSEN_P);
    PedersenCommitment { value: c }
}

/// Verify a Pedersen commitment against an opening.
pub fn pedersen_verify(
    commitment: &PedersenCommitment,
    opening: &PedersenOpening,
) -> bool {
    let expected = pedersen_commit(opening.committed_value, opening.blinding);
    commitment.value == expected.value
}

/// Homomorphic addition of two Pedersen commitments:
/// C(v1, r1) * C(v2, r2) = C(v1+v2, r1+r2) mod p.
pub fn pedersen_add(c1: &PedersenCommitment, c2: &PedersenCommitment) -> PedersenCommitment {
    let value = mod_mul(c1.value, c2.value, PEDERSEN_P);
    PedersenCommitment { value }
}

// ── Batch Commitment ────────────────────────────────────────────────────────

/// A batch of hash commitments with a single root commitment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchCommitment {
    /// Individual commitment hashes.
    pub commitments: Vec<Vec<u8>>,
    /// Root: hash of all commitments concatenated.
    pub root: Vec<u8>,
}

/// Create a batch commitment for multiple values, each with its own nonce.
pub fn batch_commit(
    values: &[&[u8]],
    nonces: &[&[u8]],
) -> Result<BatchCommitment, CommitError> {
    if values.len() != nonces.len() {
        return Err(CommitError::InvalidData(
            "values and nonces must have same length".to_string(),
        ));
    }

    let mut commitments = Vec::with_capacity(values.len());
    let mut root_data = Vec::new();

    for (value, nonce) in values.iter().zip(nonces.iter()) {
        let c = hash_commit(value, nonce)?;
        root_data.extend_from_slice(&c.hash);
        commitments.push(c.hash);
    }

    let root = sha256(&root_data).to_vec();

    Ok(BatchCommitment {
        commitments,
        root,
    })
}

/// Verify a single item within a batch commitment.
pub fn batch_verify_item(
    batch: &BatchCommitment,
    index: usize,
    value: &[u8],
    nonce: &[u8],
) -> bool {
    if index >= batch.commitments.len() {
        return false;
    }
    let expected_commitment = HashCommitment {
        hash: batch.commitments[index].clone(),
    };
    hash_verify(&expected_commitment, value, nonce)
}

/// Verify the batch root is consistent with all stored commitments.
pub fn batch_verify_root(batch: &BatchCommitment) -> bool {
    let mut root_data = Vec::new();
    for c in &batch.commitments {
        root_data.extend_from_slice(c);
    }
    let expected = sha256(&root_data);
    constant_time_eq(&batch.root, &expected)
}

// ── Commitment Equality Proof ───────────────────────────────────────────────

/// Proof that two hash commitments commit to the same value
/// (revealed via opening both).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EqualityProof {
    /// The common value.
    pub value: Vec<u8>,
    /// Nonce for the first commitment.
    pub nonce1: Vec<u8>,
    /// Nonce for the second commitment.
    pub nonce2: Vec<u8>,
}

/// Verify that two commitments are equal using the equality proof.
pub fn verify_equality(
    c1: &HashCommitment,
    c2: &HashCommitment,
    proof: &EqualityProof,
) -> bool {
    hash_verify(c1, &proof.value, &proof.nonce1)
        && hash_verify(c2, &proof.value, &proof.nonce2)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_nonce() -> Vec<u8> {
        // Deterministic 32-byte nonce for tests.
        let mut nonce = vec![0u8; 32];
        for (i, byte) in nonce.iter_mut().enumerate() {
            *byte = (i as u8).wrapping_mul(7).wrapping_add(13);
        }
        nonce
    }

    fn make_nonce2() -> Vec<u8> {
        let mut nonce = vec![0u8; 32];
        for (i, byte) in nonce.iter_mut().enumerate() {
            *byte = (i as u8).wrapping_mul(11).wrapping_add(37);
        }
        nonce
    }

    #[test]
    fn test_hash_commit_basic() {
        let nonce = make_nonce();
        let c = hash_commit(b"hello", &nonce).unwrap();
        assert_eq!(c.hash.len(), 32);
    }

    #[test]
    fn test_hash_commit_empty_value() {
        let nonce = make_nonce();
        assert!(hash_commit(b"", &nonce).is_err());
    }

    #[test]
    fn test_hash_commit_short_nonce() {
        assert!(hash_commit(b"val", &[1, 2, 3]).is_err());
    }

    #[test]
    fn test_hash_verify_valid() {
        let nonce = make_nonce();
        let c = hash_commit(b"secret", &nonce).unwrap();
        assert!(hash_verify(&c, b"secret", &nonce));
    }

    #[test]
    fn test_hash_verify_wrong_value() {
        let nonce = make_nonce();
        let c = hash_commit(b"secret", &nonce).unwrap();
        assert!(!hash_verify(&c, b"wrong", &nonce));
    }

    #[test]
    fn test_hash_verify_wrong_nonce() {
        let nonce = make_nonce();
        let c = hash_commit(b"secret", &nonce).unwrap();
        let bad_nonce = make_nonce2();
        assert!(!hash_verify(&c, b"secret", &bad_nonce));
    }

    #[test]
    fn test_hash_commitment_hex() {
        let nonce = make_nonce();
        let c = hash_commit(b"data", &nonce).unwrap();
        assert_eq!(c.to_hex().len(), 64);
    }

    #[test]
    fn test_hash_commitment_deterministic() {
        let nonce = make_nonce();
        let c1 = hash_commit(b"data", &nonce).unwrap();
        let c2 = hash_commit(b"data", &nonce).unwrap();
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_hash_different_nonces_different_commits() {
        let n1 = make_nonce();
        let n2 = make_nonce2();
        let c1 = hash_commit(b"same", &n1).unwrap();
        let c2 = hash_commit(b"same", &n2).unwrap();
        assert_ne!(c1, c2);
    }

    #[test]
    fn test_commit_reveal_protocol_success() {
        let nonce = make_nonce();
        let c = hash_commit(b"my value", &nonce).unwrap();

        let mut proto = CommitRevealProtocol::new();
        assert_eq!(proto.phase(), ProtocolPhase::AwaitingCommitment);

        proto.submit_commitment(c).unwrap();
        assert_eq!(proto.phase(), ProtocolPhase::AwaitingReveal);

        let opening = HashOpening {
            value: b"my value".to_vec(),
            nonce: nonce.clone(),
        };
        let valid = proto.submit_opening(opening).unwrap();
        assert!(valid);
        assert_eq!(proto.phase(), ProtocolPhase::Verified);
        assert_eq!(proto.revealed_value(), Some(b"my value".as_ref()));
    }

    #[test]
    fn test_commit_reveal_protocol_failure() {
        let nonce = make_nonce();
        let c = hash_commit(b"real", &nonce).unwrap();

        let mut proto = CommitRevealProtocol::new();
        proto.submit_commitment(c).unwrap();

        let opening = HashOpening {
            value: b"fake".to_vec(),
            nonce,
        };
        let valid = proto.submit_opening(opening).unwrap();
        assert!(!valid);
        assert_eq!(proto.phase(), ProtocolPhase::Failed);
        assert!(proto.revealed_value().is_none());
    }

    #[test]
    fn test_protocol_reveal_before_commit() {
        let mut proto = CommitRevealProtocol::new();
        let opening = HashOpening {
            value: b"x".to_vec(),
            nonce: make_nonce(),
        };
        assert!(proto.submit_opening(opening).is_err());
    }

    #[test]
    fn test_pedersen_commit_verify() {
        let c = pedersen_commit(42, 1000);
        let opening = PedersenOpening {
            committed_value: 42,
            blinding: 1000,
        };
        assert!(pedersen_verify(&c, &opening));
    }

    #[test]
    fn test_pedersen_wrong_value() {
        let c = pedersen_commit(42, 1000);
        let bad_opening = PedersenOpening {
            committed_value: 43,
            blinding: 1000,
        };
        assert!(!pedersen_verify(&c, &bad_opening));
    }

    #[test]
    fn test_pedersen_homomorphic() {
        let v1 = 100u64;
        let r1 = 777u64;
        let v2 = 200u64;
        let r2 = 888u64;

        let c1 = pedersen_commit(v1, r1);
        let c2 = pedersen_commit(v2, r2);
        let c_sum = pedersen_add(&c1, &c2);

        let c_direct = pedersen_commit(v1 + v2, r1 + r2);
        assert_eq!(c_sum, c_direct);
    }

    #[test]
    fn test_batch_commit() {
        let n1 = make_nonce();
        let n2 = make_nonce2();
        let batch = batch_commit(
            &[b"alpha", b"beta"],
            &[n1.as_slice(), n2.as_slice()],
        ).unwrap();
        assert_eq!(batch.commitments.len(), 2);
        assert_eq!(batch.root.len(), 32);
    }

    #[test]
    fn test_batch_verify_item() {
        let n1 = make_nonce();
        let n2 = make_nonce2();
        let batch = batch_commit(
            &[b"alpha", b"beta"],
            &[n1.as_slice(), n2.as_slice()],
        ).unwrap();
        assert!(batch_verify_item(&batch, 0, b"alpha", &n1));
        assert!(batch_verify_item(&batch, 1, b"beta", &n2));
        assert!(!batch_verify_item(&batch, 0, b"wrong", &n1));
    }

    #[test]
    fn test_batch_verify_root() {
        let n1 = make_nonce();
        let n2 = make_nonce2();
        let batch = batch_commit(
            &[b"x", b"y"],
            &[n1.as_slice(), n2.as_slice()],
        ).unwrap();
        assert!(batch_verify_root(&batch));
    }

    #[test]
    fn test_batch_length_mismatch() {
        let n1 = make_nonce();
        let result = batch_commit(
            &[b"a", b"b"],
            &[n1.as_slice()],
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_equality_proof() {
        let n1 = make_nonce();
        let n2 = make_nonce2();
        let c1 = hash_commit(b"same-value", &n1).unwrap();
        let c2 = hash_commit(b"same-value", &n2).unwrap();

        let proof = EqualityProof {
            value: b"same-value".to_vec(),
            nonce1: n1,
            nonce2: n2,
        };
        assert!(verify_equality(&c1, &c2, &proof));
    }

    #[test]
    fn test_equality_proof_fails_for_different() {
        let n1 = make_nonce();
        let n2 = make_nonce2();
        let c1 = hash_commit(b"value-a", &n1).unwrap();
        let c2 = hash_commit(b"value-b", &n2).unwrap();

        let proof = EqualityProof {
            value: b"value-a".to_vec(),
            nonce1: n1,
            nonce2: n2,
        };
        // c2 was committed with "value-b", not "value-a"
        assert!(!verify_equality(&c1, &c2, &proof));
    }

    #[test]
    fn test_commitment_serialization() {
        let nonce = make_nonce();
        let c = hash_commit(b"data", &nonce).unwrap();
        let json = serde_json::to_string(&c).unwrap();
        let c2: HashCommitment = serde_json::from_str(&json).unwrap();
        assert_eq!(c, c2);
    }

    #[test]
    fn test_pedersen_serialization() {
        let c = pedersen_commit(99, 456);
        let json = serde_json::to_string(&c).unwrap();
        let c2: PedersenCommitment = serde_json::from_str(&json).unwrap();
        assert_eq!(c, c2);
    }

    #[test]
    fn test_mod_pow_basic() {
        assert_eq!(mod_pow(2, 10, 1000), 24); // 1024 mod 1000
        assert_eq!(mod_pow(3, 0, 100), 1);
        assert_eq!(mod_pow(5, 1, 100), 5);
    }

    #[test]
    fn test_pedersen_zero_value() {
        let c = pedersen_commit(0, 100);
        let opening = PedersenOpening {
            committed_value: 0,
            blinding: 100,
        };
        assert!(pedersen_verify(&c, &opening));
    }
}
