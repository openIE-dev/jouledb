//! Zero-knowledge proof concepts — Schnorr-like proof simulation (discrete log
//! knowledge), sigma protocol (commit/challenge/response), proof verification,
//! interactive to non-interactive (Fiat-Shamir heuristic), and serialization.

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

// ── Modular arithmetic (simulated group) ────────────────────────────────────

/// We simulate a cyclic group Z_p^* with a safe prime p and generator g.
/// Using p = 2147483647 (Mersenne prime 2^31 - 1), g = 3.
const ZK_P: u64 = 2_147_483_647;
const ZK_G: u64 = 3;
/// Order of the group: p - 1 (for this prime, the group order is p-1).
const ZK_ORDER: u64 = ZK_P - 1;

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

/// Modular addition: (a + b) mod m.
fn mod_add(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 + b as u128) % m as u128) as u64
}

/// Modular subtraction: (a - b) mod m.
fn mod_sub(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 + m as u128 - b as u128) % m as u128) as u64
}

/// Hash-to-scalar: reduce SHA-256 output to a value mod order.
fn hash_to_scalar(data: &[u8]) -> u64 {
    let h = sha256(data);
    let val = u64::from_le_bytes(h[..8].try_into().unwrap());
    val % ZK_ORDER
}

// ── Errors ──────────────────────────────────────────────────────────────────

/// Zero-knowledge proof errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ZkError {
    /// Proof verification failed.
    VerificationFailed,
    /// Invalid parameters.
    InvalidParams(String),
    /// Protocol state error.
    InvalidState(String),
}

impl std::fmt::Display for ZkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::VerificationFailed => write!(f, "zero-knowledge proof verification failed"),
            Self::InvalidParams(s) => write!(f, "invalid parameters: {s}"),
            Self::InvalidState(s) => write!(f, "invalid state: {s}"),
        }
    }
}

impl std::error::Error for ZkError {}

// ── Group Parameters ────────────────────────────────────────────────────────

/// Public parameters for the ZK protocol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupParams {
    /// Prime modulus.
    pub p: u64,
    /// Generator.
    pub g: u64,
    /// Group order.
    pub order: u64,
}

impl GroupParams {
    /// Default parameters using the Mersenne prime.
    pub fn default_params() -> Self {
        Self {
            p: ZK_P,
            g: ZK_G,
            order: ZK_ORDER,
        }
    }
}

impl Default for GroupParams {
    fn default() -> Self {
        Self::default_params()
    }
}

// ── Schnorr-like Proof of Knowledge of Discrete Log ─────────────────────────

/// Prove knowledge of x such that y = g^x mod p.
///
/// This is a sigma protocol: Commit -> Challenge -> Response.

/// Prover's commitment (first message).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchnorrCommitment {
    /// t = g^r mod p (the commitment).
    pub t: u64,
}

/// Verifier's challenge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchnorrChallenge {
    /// Random challenge value.
    pub c: u64,
}

/// Prover's response (third message).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchnorrResponse {
    /// s = r + c*x mod order.
    pub s: u64,
}

/// A complete Schnorr proof (non-interactive via Fiat-Shamir).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchnorrProof {
    /// The commitment t = g^r.
    pub commitment: u64,
    /// The challenge c = H(g, y, t).
    pub challenge: u64,
    /// The response s = r + c*x mod order.
    pub response: u64,
    /// The public key y = g^x.
    pub public_key: u64,
}

/// Generate a Schnorr proof of knowledge of `secret` (the discrete log).
///
/// `secret`: the witness x such that y = g^x mod p.
/// `randomness`: entropy for the commitment random value r.
pub fn schnorr_prove(secret: u64, randomness: &[u8]) -> SchnorrProof {
    let params = GroupParams::default_params();

    // Compute public key: y = g^x mod p.
    let y = mod_pow(params.g, secret % params.order, params.p);

    // Generate random r from randomness.
    let r = hash_to_scalar(randomness);

    // Commitment: t = g^r mod p.
    let t = mod_pow(params.g, r, params.p);

    // Fiat-Shamir challenge: c = H(g || y || t).
    let mut challenge_input = Vec::new();
    challenge_input.extend_from_slice(&params.g.to_le_bytes());
    challenge_input.extend_from_slice(&y.to_le_bytes());
    challenge_input.extend_from_slice(&t.to_le_bytes());
    let c = hash_to_scalar(&challenge_input);

    // Response: s = r + c*x mod order.
    let cx = mod_mul(c, secret % params.order, params.order);
    let s = mod_add(r, cx, params.order);

    SchnorrProof {
        commitment: t,
        challenge: c,
        response: s,
        public_key: y,
    }
}

/// Verify a Schnorr proof.
pub fn schnorr_verify(proof: &SchnorrProof) -> bool {
    let params = GroupParams::default_params();

    // Recompute challenge: c = H(g || y || t).
    let mut challenge_input = Vec::new();
    challenge_input.extend_from_slice(&params.g.to_le_bytes());
    challenge_input.extend_from_slice(&proof.public_key.to_le_bytes());
    challenge_input.extend_from_slice(&proof.commitment.to_le_bytes());
    let c = hash_to_scalar(&challenge_input);

    if c != proof.challenge {
        return false;
    }

    // Verify: g^s == t * y^c mod p.
    let lhs = mod_pow(params.g, proof.response, params.p);
    let y_c = mod_pow(proof.public_key, c, params.p);
    let rhs = mod_mul(proof.commitment, y_c, params.p);

    lhs == rhs
}

// ── Interactive Sigma Protocol ──────────────────────────────────────────────

/// State of the sigma protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SigmaPhase {
    AwaitCommitment,
    AwaitChallenge,
    AwaitResponse,
    Verified,
    Failed,
}

/// Interactive sigma protocol verifier.
#[derive(Debug, Clone)]
pub struct SigmaVerifier {
    params: GroupParams,
    public_key: u64,
    commitment: Option<u64>,
    challenge: Option<u64>,
    phase: SigmaPhase,
}

impl SigmaVerifier {
    /// Create a new verifier for the given public key y = g^x.
    pub fn new(public_key: u64) -> Self {
        Self {
            params: GroupParams::default_params(),
            public_key,
            commitment: None,
            challenge: None,
            phase: SigmaPhase::AwaitCommitment,
        }
    }

    /// Current phase.
    pub fn phase(&self) -> SigmaPhase {
        self.phase
    }

    /// Receive the prover's commitment t.
    pub fn receive_commitment(&mut self, t: u64) -> Result<(), ZkError> {
        if self.phase != SigmaPhase::AwaitCommitment {
            return Err(ZkError::InvalidState("not awaiting commitment".to_string()));
        }
        self.commitment = Some(t);
        self.phase = SigmaPhase::AwaitChallenge;
        Ok(())
    }

    /// Generate a challenge (using Fiat-Shamir for determinism).
    pub fn generate_challenge(&mut self) -> Result<u64, ZkError> {
        if self.phase != SigmaPhase::AwaitChallenge {
            return Err(ZkError::InvalidState("not awaiting challenge".to_string()));
        }
        let t = self.commitment.unwrap();
        let mut input = Vec::new();
        input.extend_from_slice(&self.params.g.to_le_bytes());
        input.extend_from_slice(&self.public_key.to_le_bytes());
        input.extend_from_slice(&t.to_le_bytes());
        let c = hash_to_scalar(&input);
        self.challenge = Some(c);
        self.phase = SigmaPhase::AwaitResponse;
        Ok(c)
    }

    /// Receive the prover's response and verify.
    pub fn receive_response(&mut self, s: u64) -> Result<bool, ZkError> {
        if self.phase != SigmaPhase::AwaitResponse {
            return Err(ZkError::InvalidState("not awaiting response".to_string()));
        }
        let t = self.commitment.unwrap();
        let c = self.challenge.unwrap();

        let lhs = mod_pow(self.params.g, s, self.params.p);
        let y_c = mod_pow(self.public_key, c, self.params.p);
        let rhs = mod_mul(t, y_c, self.params.p);

        let valid = lhs == rhs;
        self.phase = if valid {
            SigmaPhase::Verified
        } else {
            SigmaPhase::Failed
        };
        Ok(valid)
    }
}

/// Interactive sigma protocol prover.
#[derive(Debug, Clone)]
pub struct SigmaProver {
    params: GroupParams,
    secret: u64,
    r: u64,
    commitment: u64,
}

impl SigmaProver {
    /// Create a new prover with secret x and randomness.
    pub fn new(secret: u64, randomness: &[u8]) -> Self {
        let params = GroupParams::default_params();
        let r = hash_to_scalar(randomness);
        let commitment = mod_pow(params.g, r, params.p);
        let order = params.order;
        Self {
            params,
            secret: secret % order,
            r,
            commitment,
        }
    }

    /// Get the commitment to send to the verifier.
    pub fn commitment(&self) -> u64 {
        self.commitment
    }

    /// Respond to a challenge c: s = r + c*x mod order.
    pub fn respond(&self, challenge: u64) -> u64 {
        let cx = mod_mul(challenge, self.secret, self.params.order);
        mod_add(self.r, cx, self.params.order)
    }
}

// ── Non-interactive (Fiat-Shamir) batch proofs ──────────────────────────────

/// A batch of Schnorr proofs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchSchnorrProof {
    pub proofs: Vec<SchnorrProof>,
}

/// Create batch proofs for multiple secrets.
pub fn batch_prove(secrets: &[u64], randomness_seed: &[u8]) -> BatchSchnorrProof {
    let proofs: Vec<SchnorrProof> = secrets
        .iter()
        .enumerate()
        .map(|(i, &secret)| {
            let mut r = Vec::with_capacity(randomness_seed.len() + 4);
            r.extend_from_slice(randomness_seed);
            r.extend_from_slice(&(i as u32).to_le_bytes());
            schnorr_prove(secret, &r)
        })
        .collect();
    BatchSchnorrProof { proofs }
}

/// Verify all proofs in a batch.
pub fn batch_verify(batch: &BatchSchnorrProof) -> bool {
    batch.proofs.iter().all(schnorr_verify)
}

// ── Proof of Equality of Discrete Logs ──────────────────────────────────────

/// Proof that log_g(y1) == log_h(y2), i.e., the same secret x satisfies
/// y1 = g^x and y2 = h^x.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DlogEqualityProof {
    /// t1 = g^r.
    pub t1: u64,
    /// t2 = h^r.
    pub t2: u64,
    /// Challenge.
    pub challenge: u64,
    /// Response s = r + c*x.
    pub response: u64,
    /// Public values.
    pub y1: u64,
    pub y2: u64,
    pub g_val: u64,
    pub h_val: u64,
}

/// Prove equality of discrete logs: log_g(y1) = log_h(y2).
///
/// `secret`: the common exponent x.
/// `g_val`, `h_val`: two generators.
pub fn prove_dlog_equality(
    secret: u64,
    g_val: u64,
    h_val: u64,
    randomness: &[u8],
) -> DlogEqualityProof {
    let p = ZK_P;
    let order = ZK_ORDER;
    let x = secret % order;

    let y1 = mod_pow(g_val, x, p);
    let y2 = mod_pow(h_val, x, p);

    let r = hash_to_scalar(randomness);
    let t1 = mod_pow(g_val, r, p);
    let t2 = mod_pow(h_val, r, p);

    // Fiat-Shamir: c = H(g, h, y1, y2, t1, t2).
    let mut input = Vec::new();
    input.extend_from_slice(&g_val.to_le_bytes());
    input.extend_from_slice(&h_val.to_le_bytes());
    input.extend_from_slice(&y1.to_le_bytes());
    input.extend_from_slice(&y2.to_le_bytes());
    input.extend_from_slice(&t1.to_le_bytes());
    input.extend_from_slice(&t2.to_le_bytes());
    let c = hash_to_scalar(&input);

    let cx = mod_mul(c, x, order);
    let s = mod_add(r, cx, order);

    DlogEqualityProof {
        t1,
        t2,
        challenge: c,
        response: s,
        y1,
        y2,
        g_val,
        h_val,
    }
}

/// Verify a discrete log equality proof.
pub fn verify_dlog_equality(proof: &DlogEqualityProof) -> bool {
    let p = ZK_P;

    // Recompute challenge.
    let mut input = Vec::new();
    input.extend_from_slice(&proof.g_val.to_le_bytes());
    input.extend_from_slice(&proof.h_val.to_le_bytes());
    input.extend_from_slice(&proof.y1.to_le_bytes());
    input.extend_from_slice(&proof.y2.to_le_bytes());
    input.extend_from_slice(&proof.t1.to_le_bytes());
    input.extend_from_slice(&proof.t2.to_le_bytes());
    let c = hash_to_scalar(&input);

    if c != proof.challenge {
        return false;
    }

    // Check: g^s == t1 * y1^c.
    let lhs1 = mod_pow(proof.g_val, proof.response, p);
    let y1c = mod_pow(proof.y1, c, p);
    let rhs1 = mod_mul(proof.t1, y1c, p);

    // Check: h^s == t2 * y2^c.
    let lhs2 = mod_pow(proof.h_val, proof.response, p);
    let y2c = mod_pow(proof.y2, c, p);
    let rhs2 = mod_mul(proof.t2, y2c, p);

    lhs1 == rhs1 && lhs2 == rhs2
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mod_pow_basic() {
        assert_eq!(mod_pow(2, 10, 1000), 24);
        assert_eq!(mod_pow(3, 0, 100), 1);
        assert_eq!(mod_pow(ZK_G, 0, ZK_P), 1);
    }

    #[test]
    fn test_mod_add_sub() {
        assert_eq!(mod_add(5, 7, 10), 2);
        assert_eq!(mod_sub(3, 5, 10), 8); // (3 - 5) mod 10 = 8
    }

    #[test]
    fn test_schnorr_prove_verify() {
        let secret = 12345u64;
        let proof = schnorr_prove(secret, b"randomness");
        assert!(schnorr_verify(&proof));
    }

    #[test]
    fn test_schnorr_different_secrets() {
        let proof1 = schnorr_prove(100, b"rand1");
        let proof2 = schnorr_prove(200, b"rand1");
        assert!(schnorr_verify(&proof1));
        assert!(schnorr_verify(&proof2));
        assert_ne!(proof1.public_key, proof2.public_key);
    }

    #[test]
    fn test_schnorr_deterministic() {
        let p1 = schnorr_prove(42, b"seed");
        let p2 = schnorr_prove(42, b"seed");
        assert_eq!(p1.commitment, p2.commitment);
        assert_eq!(p1.challenge, p2.challenge);
        assert_eq!(p1.response, p2.response);
    }

    #[test]
    fn test_schnorr_tampered_response() {
        let mut proof = schnorr_prove(999, b"rand");
        proof.response = proof.response.wrapping_add(1);
        assert!(!schnorr_verify(&proof));
    }

    #[test]
    fn test_schnorr_tampered_commitment() {
        let mut proof = schnorr_prove(42, b"rand");
        proof.commitment = proof.commitment.wrapping_add(1);
        assert!(!schnorr_verify(&proof));
    }

    #[test]
    fn test_schnorr_tampered_public_key() {
        let mut proof = schnorr_prove(42, b"rand");
        proof.public_key = proof.public_key.wrapping_add(1);
        assert!(!schnorr_verify(&proof));
    }

    #[test]
    fn test_schnorr_serialization() {
        let proof = schnorr_prove(777, b"entropy");
        let json = serde_json::to_string(&proof).unwrap();
        let deserialized: SchnorrProof = serde_json::from_str(&json).unwrap();
        assert!(schnorr_verify(&deserialized));
    }

    #[test]
    fn test_interactive_sigma() {
        let secret = 54321u64;
        let params = GroupParams::default_params();
        let y = mod_pow(params.g, secret % params.order, params.p);

        let prover = SigmaProver::new(secret, b"prover-rand");
        let mut verifier = SigmaVerifier::new(y);

        // Step 1: Prover sends commitment.
        let t = prover.commitment();
        verifier.receive_commitment(t).unwrap();

        // Step 2: Verifier sends challenge.
        let c = verifier.generate_challenge().unwrap();

        // Step 3: Prover sends response.
        let s = prover.respond(c);
        let valid = verifier.receive_response(s).unwrap();

        assert!(valid);
        assert_eq!(verifier.phase(), SigmaPhase::Verified);
    }

    #[test]
    fn test_interactive_sigma_wrong_secret() {
        let real_secret = 100u64;
        let fake_secret = 200u64;
        let params = GroupParams::default_params();
        let y = mod_pow(params.g, real_secret, params.p);

        // Prover uses fake secret.
        let prover = SigmaProver::new(fake_secret, b"rand");
        let mut verifier = SigmaVerifier::new(y);

        verifier.receive_commitment(prover.commitment()).unwrap();
        let c = verifier.generate_challenge().unwrap();
        let s = prover.respond(c);
        let valid = verifier.receive_response(s).unwrap();

        assert!(!valid);
        assert_eq!(verifier.phase(), SigmaPhase::Failed);
    }

    #[test]
    fn test_sigma_verifier_wrong_order() {
        let mut v = SigmaVerifier::new(42);
        // Can't generate challenge before receiving commitment.
        assert!(v.generate_challenge().is_err());
    }

    #[test]
    fn test_batch_prove_verify() {
        let secrets = vec![10, 20, 30, 40, 50];
        let batch = batch_prove(&secrets, b"batch-seed");
        assert_eq!(batch.proofs.len(), 5);
        assert!(batch_verify(&batch));
    }

    #[test]
    fn test_batch_tampered() {
        let secrets = vec![10, 20, 30];
        let mut batch = batch_prove(&secrets, b"seed");
        batch.proofs[1].response = batch.proofs[1].response.wrapping_add(1);
        assert!(!batch_verify(&batch));
    }

    #[test]
    fn test_dlog_equality_proof() {
        let secret = 42u64;
        let g = 3u64;
        let h = 5u64;
        let proof = prove_dlog_equality(secret, g, h, b"rand");
        assert!(verify_dlog_equality(&proof));
    }

    #[test]
    fn test_dlog_equality_wrong_secret() {
        // Create proof for secret=42 but verify with tampered proof.
        let proof = prove_dlog_equality(42, 3, 5, b"rand");
        let mut tampered = proof.clone();
        tampered.response = tampered.response.wrapping_add(1);
        assert!(!verify_dlog_equality(&tampered));
    }

    #[test]
    fn test_dlog_equality_serialization() {
        let proof = prove_dlog_equality(99, 3, 7, b"entropy");
        let json = serde_json::to_string(&proof).unwrap();
        let deserialized: DlogEqualityProof = serde_json::from_str(&json).unwrap();
        assert!(verify_dlog_equality(&deserialized));
    }

    #[test]
    fn test_group_params_default() {
        let params = GroupParams::default();
        assert_eq!(params.p, ZK_P);
        assert_eq!(params.g, ZK_G);
    }

    #[test]
    fn test_hash_to_scalar_deterministic() {
        let s1 = hash_to_scalar(b"input");
        let s2 = hash_to_scalar(b"input");
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_hash_to_scalar_different_inputs() {
        let s1 = hash_to_scalar(b"a");
        let s2 = hash_to_scalar(b"b");
        assert_ne!(s1, s2);
    }

    #[test]
    fn test_schnorr_zero_secret() {
        // Secret = 0 means y = g^0 = 1.
        let proof = schnorr_prove(0, b"rand");
        assert_eq!(proof.public_key, 1);
        assert!(schnorr_verify(&proof));
    }

    #[test]
    fn test_schnorr_large_secret() {
        let proof = schnorr_prove(u64::MAX, b"rand");
        assert!(schnorr_verify(&proof));
    }
}
