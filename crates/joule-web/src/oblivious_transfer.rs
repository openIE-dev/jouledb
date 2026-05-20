//! Oblivious transfer simulation — 1-out-of-2 OT protocol, sender/receiver
//! state machines, XOR-based simple OT, protocol transcript, batch OT, and
//! OT extension concepts.

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

/// XOR two byte slices of equal length.
fn xor_bytes(a: &[u8], b: &[u8]) -> Vec<u8> {
    a.iter().zip(b.iter()).map(|(x, y)| x ^ y).collect()
}

/// Derive a mask from a key and index using SHA-256.
fn derive_mask(key: &[u8], index: u32, length: usize) -> Vec<u8> {
    let mut result = Vec::with_capacity(length);
    let mut counter = 0u32;
    while result.len() < length {
        let mut input = Vec::with_capacity(key.len() + 8);
        input.extend_from_slice(key);
        input.extend_from_slice(&index.to_le_bytes());
        input.extend_from_slice(&counter.to_le_bytes());
        let hash = sha256(&input);
        let remaining = length - result.len();
        let take = remaining.min(32);
        result.extend_from_slice(&hash[..take]);
        counter += 1;
    }
    result.truncate(length);
    result
}

// ── Errors ──────────────────────────────────────────────────────────────────

/// OT protocol errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OtError {
    /// Invalid choice bit (must be 0 or 1).
    InvalidChoice(u8),
    /// Protocol state error.
    InvalidState(String),
    /// Message length mismatch.
    LengthMismatch { expected: usize, got: usize },
    /// Empty message.
    EmptyMessage,
}

impl std::fmt::Display for OtError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidChoice(c) => write!(f, "invalid choice bit: {c} (must be 0 or 1)"),
            Self::InvalidState(s) => write!(f, "invalid protocol state: {s}"),
            Self::LengthMismatch { expected, got } => {
                write!(f, "message length mismatch: expected {expected}, got {got}")
            }
            Self::EmptyMessage => write!(f, "message must not be empty"),
        }
    }
}

impl std::error::Error for OtError {}

// ── Protocol Transcript ─────────────────────────────────────────────────────

/// A single step in the OT protocol transcript.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptEntry {
    /// Step number.
    pub step: usize,
    /// Description of the action.
    pub action: String,
    /// Party performing the action.
    pub party: String,
    /// Data involved (hex-encoded where applicable).
    pub data: String,
}

/// Protocol transcript recording all steps.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Transcript {
    pub entries: Vec<TranscriptEntry>,
}

impl Transcript {
    fn new() -> Self {
        Self { entries: Vec::new() }
    }

    fn record(&mut self, party: &str, action: &str, data: &str) {
        let step = self.entries.len() + 1;
        self.entries.push(TranscriptEntry {
            step,
            action: action.to_string(),
            party: party.to_string(),
            data: data.to_string(),
        });
    }

    /// Number of steps recorded.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ── XOR-based 1-out-of-2 OT ────────────────────────────────────────────────

/// Simulated 1-out-of-2 Oblivious Transfer using a hash-based scheme.
///
/// Protocol overview:
/// 1. Sender has two messages: m0, m1.
/// 2. Receiver has a choice bit b (0 or 1).
/// 3. Sender generates a random key k.
/// 4. Receiver generates a blinding value based on b.
/// 5. Sender encrypts both messages: e0 = m0 XOR H(k, 0), e1 = m1 XOR H(k, 1).
/// 6. Receiver can only decrypt m_b.
///
/// This is a simulation — the random oracle model is approximated with SHA-256.

/// Result of a completed OT.
#[derive(Debug, Clone)]
pub struct OtResult {
    /// The message the receiver obtained.
    pub received_message: Vec<u8>,
    /// The choice bit used.
    pub choice: u8,
    /// Protocol transcript.
    pub transcript: Transcript,
}

/// Sender state.
#[derive(Debug, Clone)]
pub enum SenderState {
    /// Initial: holding messages.
    Init { msg0: Vec<u8>, msg1: Vec<u8> },
    /// Key generated, ready to encrypt.
    KeyGenerated {
        msg0: Vec<u8>,
        msg1: Vec<u8>,
        key: Vec<u8>,
    },
    /// Ciphertexts sent.
    Sent {
        enc0: Vec<u8>,
        enc1: Vec<u8>,
    },
    /// Complete.
    Done,
}

/// Receiver state.
#[derive(Debug, Clone)]
pub enum ReceiverState {
    /// Initial: holding choice bit.
    Init { choice: u8 },
    /// Blinding generated.
    Blinded { choice: u8, blind_key: Vec<u8> },
    /// Message received.
    Received { message: Vec<u8>, choice: u8 },
    /// Complete.
    Done,
}

/// Run a complete simulated 1-out-of-2 OT.
///
/// `msg0` and `msg1` are the sender's messages (must be same length).
/// `choice` is the receiver's bit (0 or 1).
/// `randomness` provides deterministic randomness for the simulation.
pub fn ot_1_of_2(
    msg0: &[u8],
    msg1: &[u8],
    choice: u8,
    randomness: &[u8],
) -> Result<OtResult, OtError> {
    if choice > 1 {
        return Err(OtError::InvalidChoice(choice));
    }
    if msg0.is_empty() || msg1.is_empty() {
        return Err(OtError::EmptyMessage);
    }
    if msg0.len() != msg1.len() {
        return Err(OtError::LengthMismatch {
            expected: msg0.len(),
            got: msg1.len(),
        });
    }

    let msg_len = msg0.len();
    let mut transcript = Transcript::new();

    // Step 1: Sender generates key from randomness.
    let key = sha256(randomness);
    transcript.record("Sender", "Generate key", &bytes_to_hex(&key));

    // Step 2: Receiver generates blinding from choice and randomness.
    let mut blind_input = Vec::with_capacity(randomness.len() + 1);
    blind_input.extend_from_slice(randomness);
    blind_input.push(choice);
    let blind_key = sha256(&blind_input);
    transcript.record("Receiver", "Generate blinding", &bytes_to_hex(&blind_key));

    // Step 3: Sender derives encryption masks.
    let mask0 = derive_mask(&key, 0, msg_len);
    let mask1 = derive_mask(&key, 1, msg_len);
    transcript.record("Sender", "Derive mask0", &bytes_to_hex(&mask0[..8.min(msg_len)]));
    transcript.record("Sender", "Derive mask1", &bytes_to_hex(&mask1[..8.min(msg_len)]));

    // Step 4: Sender encrypts both messages.
    let enc0 = xor_bytes(msg0, &mask0);
    let enc1 = xor_bytes(msg1, &mask1);
    transcript.record("Sender", "Encrypt msg0", &bytes_to_hex(&enc0[..8.min(msg_len)]));
    transcript.record("Sender", "Encrypt msg1", &bytes_to_hex(&enc1[..8.min(msg_len)]));

    // Step 5: Receiver derives the mask for chosen message and decrypts.
    let chosen_mask = derive_mask(&key, choice as u32, msg_len);
    let chosen_enc = if choice == 0 { &enc0 } else { &enc1 };
    let received = xor_bytes(chosen_enc, &chosen_mask);
    transcript.record(
        "Receiver",
        &format!("Decrypt msg{choice}"),
        &bytes_to_hex(&received[..8.min(msg_len)]),
    );

    Ok(OtResult {
        received_message: received,
        choice,
        transcript,
    })
}

// ── Batch OT ────────────────────────────────────────────────────────────────

/// Result of a batch OT.
#[derive(Debug, Clone)]
pub struct BatchOtResult {
    /// One received message per OT instance.
    pub received_messages: Vec<Vec<u8>>,
    /// Choice bits used.
    pub choices: Vec<u8>,
    /// Combined transcript.
    pub transcript: Transcript,
}

/// Execute multiple OTs in batch.
///
/// `pairs` is a list of (msg0, msg1) message pairs.
/// `choices` is a list of choice bits (one per pair).
/// `randomness` is seed material for the batch.
pub fn batch_ot(
    pairs: &[(&[u8], &[u8])],
    choices: &[u8],
    randomness: &[u8],
) -> Result<BatchOtResult, OtError> {
    if pairs.len() != choices.len() {
        return Err(OtError::LengthMismatch {
            expected: pairs.len(),
            got: choices.len(),
        });
    }

    let mut received = Vec::with_capacity(pairs.len());
    let mut transcript = Transcript::new();

    for (i, ((m0, m1), &choice)) in pairs.iter().zip(choices.iter()).enumerate() {
        // Derive per-instance randomness.
        let mut instance_rand = Vec::with_capacity(randomness.len() + 4);
        instance_rand.extend_from_slice(randomness);
        instance_rand.extend_from_slice(&(i as u32).to_le_bytes());
        let instance_seed = sha256(&instance_rand);

        let result = ot_1_of_2(m0, m1, choice, &instance_seed)?;
        received.push(result.received_message);

        transcript.record(
            "Batch",
            &format!("OT instance {i} complete (choice={choice})"),
            &format!("{} steps", result.transcript.len()),
        );
    }

    Ok(BatchOtResult {
        received_messages: received,
        choices: choices.to_vec(),
        transcript,
    })
}

// ── OT Extension (Conceptual) ───────────────────────────────────────────────

/// Parameters for OT extension.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtExtensionParams {
    /// Security parameter (number of base OTs, typically 128).
    pub security_param: usize,
    /// Number of extended OTs to produce.
    pub num_ots: usize,
}

/// Summary of an OT extension run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtExtensionSummary {
    /// Base OTs performed.
    pub base_ots: usize,
    /// Extended OTs produced.
    pub extended_ots: usize,
    /// Expansion factor.
    pub expansion_factor: f64,
    /// Steps in the protocol.
    pub steps: Vec<String>,
}

/// Simulate OT extension: expand `k` base OTs into `n` extended OTs.
///
/// This is a conceptual simulation showing the protocol structure.
/// Real OT extension (e.g., IKNP) uses matrix transposition and correlation-robust hashing.
pub fn ot_extension_simulate(params: &OtExtensionParams) -> OtExtensionSummary {
    let k = params.security_param;
    let n = params.num_ots;

    let mut steps = Vec::new();

    // Phase 1: Base OTs (receiver plays sender, sender plays receiver).
    steps.push(format!("Phase 1: Execute {k} base OTs with swapped roles"));
    steps.push(format!("  Receiver generates {k} random choice bits"));
    steps.push(format!("  Sender generates {k} pairs of random seeds"));

    // Phase 2: Matrix construction.
    steps.push(format!("Phase 2: Construct {n} x {k} binary matrix T"));
    steps.push("  Each row of T is a random {k}-bit string".to_string());
    steps.push("  Columns of T are masked using base OT outputs".to_string());

    // Phase 3: Transfer.
    steps.push(format!("Phase 3: Sender encrypts {n} message pairs using row hashes"));
    steps.push(format!("  For each i in 0..{n}: mask_i = H(i, T[i])"));
    steps.push("  e0_i = m0_i XOR H(i, T[i]), e1_i = m1_i XOR H(i, T[i] XOR s)".to_string());

    // Phase 4: Receiver decrypts.
    steps.push(format!("Phase 4: Receiver decrypts {n} chosen messages"));

    let expansion = if k > 0 { n as f64 / k as f64 } else { 0.0 };

    OtExtensionSummary {
        base_ots: k,
        extended_ots: n,
        expansion_factor: expansion,
        steps,
    }
}

// ── Simple OT Wrapper ───────────────────────────────────────────────────────

/// A simple OT interface for single message selection.
pub struct SimpleOt {
    key: Vec<u8>,
}

impl SimpleOt {
    /// Create with a shared setup key.
    pub fn new(setup_key: &[u8]) -> Self {
        Self {
            key: setup_key.to_vec(),
        }
    }

    /// Sender encrypts two messages.
    pub fn sender_encrypt(
        &self,
        msg0: &[u8],
        msg1: &[u8],
    ) -> Result<(Vec<u8>, Vec<u8>), OtError> {
        if msg0.is_empty() || msg1.is_empty() {
            return Err(OtError::EmptyMessage);
        }
        if msg0.len() != msg1.len() {
            return Err(OtError::LengthMismatch {
                expected: msg0.len(),
                got: msg1.len(),
            });
        }
        let len = msg0.len();
        let mask0 = derive_mask(&self.key, 0, len);
        let mask1 = derive_mask(&self.key, 1, len);
        Ok((xor_bytes(msg0, &mask0), xor_bytes(msg1, &mask1)))
    }

    /// Receiver decrypts the chosen message.
    pub fn receiver_decrypt(
        &self,
        choice: u8,
        ciphertext: &[u8],
    ) -> Result<Vec<u8>, OtError> {
        if choice > 1 {
            return Err(OtError::InvalidChoice(choice));
        }
        let mask = derive_mask(&self.key, choice as u32, ciphertext.len());
        Ok(xor_bytes(ciphertext, &mask))
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xor_bytes() {
        assert_eq!(xor_bytes(&[1, 2, 3], &[4, 5, 6]), vec![5, 7, 5]);
        // XOR with self is zero.
        assert_eq!(xor_bytes(&[0xFF, 0xAA], &[0xFF, 0xAA]), vec![0, 0]);
    }

    #[test]
    fn test_derive_mask_deterministic() {
        let m1 = derive_mask(b"key", 0, 64);
        let m2 = derive_mask(b"key", 0, 64);
        assert_eq!(m1, m2);
    }

    #[test]
    fn test_derive_mask_different_index() {
        let m0 = derive_mask(b"key", 0, 32);
        let m1 = derive_mask(b"key", 1, 32);
        assert_ne!(m0, m1);
    }

    #[test]
    fn test_ot_1_of_2_choice_0() {
        let msg0 = b"message zero!!";
        let msg1 = b"message one!!!";
        let result = ot_1_of_2(msg0, msg1, 0, b"random-seed").unwrap();
        assert_eq!(result.received_message, msg0);
        assert_eq!(result.choice, 0);
    }

    #[test]
    fn test_ot_1_of_2_choice_1() {
        let msg0 = b"first message!";
        let msg1 = b"second message";
        let result = ot_1_of_2(msg0, msg1, 1, b"random-seed").unwrap();
        assert_eq!(result.received_message, msg1);
        assert_eq!(result.choice, 1);
    }

    #[test]
    fn test_ot_invalid_choice() {
        assert!(ot_1_of_2(b"a", b"b", 2, b"r").is_err());
    }

    #[test]
    fn test_ot_empty_message() {
        assert!(ot_1_of_2(b"", b"x", 0, b"r").is_err());
        assert!(ot_1_of_2(b"x", b"", 0, b"r").is_err());
    }

    #[test]
    fn test_ot_length_mismatch() {
        assert!(ot_1_of_2(b"short", b"longer!", 0, b"r").is_err());
    }

    #[test]
    fn test_ot_transcript() {
        let result = ot_1_of_2(b"aaaa", b"bbbb", 0, b"seed").unwrap();
        assert!(!result.transcript.is_empty());
        assert!(result.transcript.len() >= 5);
    }

    #[test]
    fn test_batch_ot() {
        let pairs: Vec<(&[u8], &[u8])> = vec![
            (b"msg0-a", b"msg1-a"),
            (b"msg0-b", b"msg1-b"),
            (b"msg0-c", b"msg1-c"),
        ];
        let choices = vec![0, 1, 0];
        let result = batch_ot(&pairs, &choices, b"batch-seed").unwrap();
        assert_eq!(result.received_messages.len(), 3);
        assert_eq!(result.received_messages[0], b"msg0-a");
        assert_eq!(result.received_messages[1], b"msg1-b");
        assert_eq!(result.received_messages[2], b"msg0-c");
    }

    #[test]
    fn test_batch_ot_mismatch() {
        let pairs: Vec<(&[u8], &[u8])> = vec![(b"aa", b"bb")];
        let choices = vec![0, 1]; // Wrong length.
        assert!(batch_ot(&pairs, &choices, b"r").is_err());
    }

    #[test]
    fn test_ot_extension_simulate() {
        let params = OtExtensionParams {
            security_param: 128,
            num_ots: 10000,
        };
        let summary = ot_extension_simulate(&params);
        assert_eq!(summary.base_ots, 128);
        assert_eq!(summary.extended_ots, 10000);
        assert!(summary.expansion_factor > 1.0);
        assert!(!summary.steps.is_empty());
    }

    #[test]
    fn test_ot_extension_serialization() {
        let summary = ot_extension_simulate(&OtExtensionParams {
            security_param: 64,
            num_ots: 1000,
        });
        let json = serde_json::to_string(&summary).unwrap();
        let deserialized: OtExtensionSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.base_ots, 64);
    }

    #[test]
    fn test_simple_ot_choice_0() {
        let ot = SimpleOt::new(b"shared-setup");
        let (enc0, _enc1) = ot.sender_encrypt(b"hello!", b"world!").unwrap();
        let msg = ot.receiver_decrypt(0, &enc0).unwrap();
        assert_eq!(msg, b"hello!");
    }

    #[test]
    fn test_simple_ot_choice_1() {
        let ot = SimpleOt::new(b"shared-setup");
        let (_enc0, enc1) = ot.sender_encrypt(b"alpha!", b"beta!!").unwrap();
        let msg = ot.receiver_decrypt(1, &enc1).unwrap();
        assert_eq!(msg, b"beta!!");
    }

    #[test]
    fn test_simple_ot_wrong_choice_fails() {
        let ot = SimpleOt::new(b"shared-setup");
        let (enc0, _enc1) = ot.sender_encrypt(b"msg-0!", b"msg-1!").unwrap();
        // Decrypting enc0 with choice=1 should give garbage.
        let wrong = ot.receiver_decrypt(1, &enc0).unwrap();
        assert_ne!(wrong, b"msg-0!");
        assert_ne!(wrong, b"msg-1!");
    }

    #[test]
    fn test_simple_ot_invalid_choice() {
        let ot = SimpleOt::new(b"key");
        assert!(ot.receiver_decrypt(2, b"ct").is_err());
    }

    #[test]
    fn test_simple_ot_empty_message() {
        let ot = SimpleOt::new(b"key");
        assert!(ot.sender_encrypt(b"", b"x").is_err());
    }

    #[test]
    fn test_ot_deterministic() {
        let r1 = ot_1_of_2(b"AAA", b"BBB", 0, b"same-seed").unwrap();
        let r2 = ot_1_of_2(b"AAA", b"BBB", 0, b"same-seed").unwrap();
        assert_eq!(r1.received_message, r2.received_message);
    }

    #[test]
    fn test_batch_ot_all_zeros() {
        let pairs: Vec<(&[u8], &[u8])> = vec![
            (b"aa", b"bb"),
            (b"cc", b"dd"),
        ];
        let choices = vec![0, 0];
        let result = batch_ot(&pairs, &choices, b"seed").unwrap();
        assert_eq!(result.received_messages[0], b"aa");
        assert_eq!(result.received_messages[1], b"cc");
    }

    #[test]
    fn test_batch_ot_all_ones() {
        let pairs: Vec<(&[u8], &[u8])> = vec![
            (b"aa", b"bb"),
            (b"cc", b"dd"),
        ];
        let choices = vec![1, 1];
        let result = batch_ot(&pairs, &choices, b"seed").unwrap();
        assert_eq!(result.received_messages[0], b"bb");
        assert_eq!(result.received_messages[1], b"dd");
    }
}
