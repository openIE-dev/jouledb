//! Webhook signature verification — HMAC-SHA256 signature generation and
//! verification, timestamp validation (replay protection), canonical request
//! format, signature header parsing, and multiple signing secrets (rotation).
//!
//! Replaces `svix-webhooks`, `stripe-webhook`, and similar JS webhook
//! verification libraries with a pure-Rust, timing-safe verifier.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── Errors ───────────────────────────────────────────────────────

/// Webhook verification error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyError {
    /// Missing signature header.
    MissingSignature,
    /// Invalid signature format.
    InvalidSignatureFormat(String),
    /// Signature mismatch.
    SignatureMismatch,
    /// Missing timestamp.
    MissingTimestamp,
    /// Invalid timestamp format.
    InvalidTimestamp(String),
    /// Timestamp too old (replay attack).
    TimestampTooOld { age_secs: u64, max_secs: u64 },
    /// Timestamp in the future.
    TimestampInFuture { ahead_secs: u64 },
    /// No signing secrets configured.
    NoSigningSecrets,
    /// Empty payload.
    EmptyPayload,
}

impl fmt::Display for VerifyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingSignature => write!(f, "missing webhook signature"),
            Self::InvalidSignatureFormat(msg) => {
                write!(f, "invalid signature format: {msg}")
            }
            Self::SignatureMismatch => write!(f, "webhook signature mismatch"),
            Self::MissingTimestamp => write!(f, "missing webhook timestamp"),
            Self::InvalidTimestamp(msg) => write!(f, "invalid timestamp: {msg}"),
            Self::TimestampTooOld { age_secs, max_secs } => {
                write!(f, "timestamp too old: {age_secs}s (max {max_secs}s)")
            }
            Self::TimestampInFuture { ahead_secs } => {
                write!(f, "timestamp {ahead_secs}s in the future")
            }
            Self::NoSigningSecrets => write!(f, "no signing secrets configured"),
            Self::EmptyPayload => write!(f, "empty payload"),
        }
    }
}

impl std::error::Error for VerifyError {}

// ── SHA-256 + HMAC ───────────────────────────────────────────────

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
            block[i * 4], block[i * 4 + 1],
            block[i * 4 + 2], block[i * 4 + 3],
        ]);
    }
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7)
            ^ w[i - 15].rotate_right(18)
            ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17)
            ^ w[i - 2].rotate_right(19)
            ^ (w[i - 2] >> 10);
        w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    for i in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ ((!e) & g);
        let t1 = h.wrapping_add(s1).wrapping_add(ch)
            .wrapping_add(SHA256_K[i]).wrapping_add(w[i]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let t2 = s0.wrapping_add(maj);
        h = g; g = f; f = e;
        e = d.wrapping_add(t1);
        d = c; c = b; b = a;
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

fn sha256(data: &[u8]) -> Vec<u8> {
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
    let mut out = Vec::with_capacity(32);
    for word in &state {
        out.extend_from_slice(&word.to_be_bytes());
    }
    out
}

fn hmac_sha256(key: &[u8], message: &[u8]) -> Vec<u8> {
    let block_size = 64;
    let mut key_padded = if key.len() > block_size {
        sha256(key)
    } else {
        key.to_vec()
    };
    while key_padded.len() < block_size {
        key_padded.push(0x00);
    }

    let mut i_key_pad = vec![0u8; block_size];
    let mut o_key_pad = vec![0u8; block_size];
    for i in 0..block_size {
        i_key_pad[i] = key_padded[i] ^ 0x36;
        o_key_pad[i] = key_padded[i] ^ 0x5c;
    }

    let mut inner = i_key_pad;
    inner.extend_from_slice(message);
    let inner_hash = sha256(&inner);

    let mut outer = o_key_pad;
    outer.extend_from_slice(&inner_hash);
    sha256(&outer)
}

/// Constant-time byte comparison.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

fn bytes_to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn hex_to_bytes(hex: &str) -> Result<Vec<u8>, VerifyError> {
    if hex.len() % 2 != 0 {
        return Err(VerifyError::InvalidSignatureFormat(
            "odd hex length".to_string(),
        ));
    }
    let mut out = Vec::with_capacity(hex.len() / 2);
    for i in (0..hex.len()).step_by(2) {
        let byte = u8::from_str_radix(&hex[i..i + 2], 16)
            .map_err(|_| VerifyError::InvalidSignatureFormat(hex.to_string()))?;
        out.push(byte);
    }
    Ok(out)
}

// ── Canonical Request ────────────────────────────────────────────

/// Build a canonical string for signing.
///
/// Format: `{timestamp}.{payload}`
pub fn canonical_payload(timestamp: u64, payload: &str) -> String {
    format!("{timestamp}.{payload}")
}

/// Build a canonical string with a message ID (Svix-style).
///
/// Format: `{msg_id}.{timestamp}.{payload}`
pub fn canonical_payload_with_id(msg_id: &str, timestamp: u64, payload: &str) -> String {
    format!("{msg_id}.{timestamp}.{payload}")
}

// ── Signature Header Parsing ─────────────────────────────────────

/// Parsed signature header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignatureHeader {
    /// Signature scheme (e.g. "v1").
    pub scheme: String,
    /// The hex-encoded signature.
    pub signature: String,
}

impl SignatureHeader {
    /// Parse a signature header value.
    ///
    /// Supports formats:
    /// - `sha256=<hex>` (GitHub style)
    /// - `v1,<hex>` (Svix/Stripe style)
    /// - `t=<ts>,v1=<sig>` (Stripe style)
    /// - plain hex
    pub fn parse(header: &str) -> Result<Vec<Self>, VerifyError> {
        let trimmed = header.trim();
        if trimmed.is_empty() {
            return Err(VerifyError::MissingSignature);
        }

        let mut signatures = Vec::new();

        // GitHub style: "sha256=<hex>"
        if let Some(hex) = trimmed.strip_prefix("sha256=") {
            signatures.push(SignatureHeader {
                scheme: "sha256".to_string(),
                signature: hex.to_string(),
            });
            return Ok(signatures);
        }

        // Stripe/Svix style: "t=<ts>,v1=<sig>,v1=<sig>"
        if trimmed.contains('=') && trimmed.contains(',') {
            for part in trimmed.split(',') {
                let part = part.trim();
                if let Some((key, value)) = part.split_once('=') {
                    if key != "t" {
                        signatures.push(SignatureHeader {
                            scheme: key.to_string(),
                            signature: value.to_string(),
                        });
                    }
                }
            }
            if !signatures.is_empty() {
                return Ok(signatures);
            }
        }

        // Svix v1 style: "v1,<hex>"
        if let Some(hex) = trimmed.strip_prefix("v1,") {
            signatures.push(SignatureHeader {
                scheme: "v1".to_string(),
                signature: hex.to_string(),
            });
            return Ok(signatures);
        }

        // Plain hex.
        signatures.push(SignatureHeader {
            scheme: "plain".to_string(),
            signature: trimmed.to_string(),
        });
        Ok(signatures)
    }

    /// Format as a header value.
    pub fn to_header(&self) -> String {
        if self.scheme == "sha256" {
            format!("sha256={}", self.signature)
        } else if self.scheme == "plain" {
            self.signature.clone()
        } else {
            format!("{}={}", self.scheme, self.signature)
        }
    }
}

// ── Signing Secret ───────────────────────────────────────────────

/// A signing secret with optional label and creation time.
#[derive(Debug, Clone)]
pub struct SigningSecret {
    /// Secret key bytes.
    pub key: Vec<u8>,
    /// Optional label for identification.
    pub label: String,
    /// Whether this is the current (active) signing secret.
    pub active: bool,
}

impl SigningSecret {
    /// Create a new active signing secret.
    pub fn new(key: &[u8], label: &str) -> Self {
        Self {
            key: key.to_vec(),
            label: label.to_string(),
            active: true,
        }
    }

    /// Create from a string (used as UTF-8 bytes).
    pub fn from_string(secret: &str, label: &str) -> Self {
        Self::new(secret.as_bytes(), label)
    }

    /// Mark as retired (still accepted for verification, not used for signing).
    pub fn retire(mut self) -> Self {
        self.active = false;
        self
    }
}

// ── Webhook Signer ───────────────────────────────────────────────

/// Signs webhook payloads.
#[derive(Debug, Clone)]
pub struct WebhookSigner {
    secrets: Vec<SigningSecret>,
}

impl WebhookSigner {
    /// Create a signer with a single secret.
    pub fn new(secret: SigningSecret) -> Self {
        Self { secrets: vec![secret] }
    }

    /// Create a signer with multiple secrets (for rotation).
    pub fn with_secrets(secrets: Vec<SigningSecret>) -> Result<Self, VerifyError> {
        if secrets.is_empty() {
            return Err(VerifyError::NoSigningSecrets);
        }
        Ok(Self { secrets })
    }

    /// Add a new signing secret.
    pub fn add_secret(&mut self, secret: SigningSecret) {
        self.secrets.push(secret);
    }

    /// Sign a payload, returning the hex signature.
    pub fn sign(&self, timestamp: u64, payload: &str) -> Result<String, VerifyError> {
        let active = self.active_secret()?;
        let canonical = canonical_payload(timestamp, payload);
        let mac = hmac_sha256(&active.key, canonical.as_bytes());
        Ok(bytes_to_hex(&mac))
    }

    /// Sign a payload with a message ID.
    pub fn sign_with_id(
        &self,
        msg_id: &str,
        timestamp: u64,
        payload: &str,
    ) -> Result<String, VerifyError> {
        let active = self.active_secret()?;
        let canonical = canonical_payload_with_id(msg_id, timestamp, payload);
        let mac = hmac_sha256(&active.key, canonical.as_bytes());
        Ok(bytes_to_hex(&mac))
    }

    /// Get the active signing secret.
    fn active_secret(&self) -> Result<&SigningSecret, VerifyError> {
        self.secrets
            .iter()
            .find(|s| s.active)
            .ok_or(VerifyError::NoSigningSecrets)
    }

    /// Number of signing secrets.
    pub fn secret_count(&self) -> usize {
        self.secrets.len()
    }

    /// Labels of all secrets.
    pub fn secret_labels(&self) -> Vec<&str> {
        self.secrets.iter().map(|s| s.label.as_str()).collect()
    }
}

// ── Webhook Verifier ─────────────────────────────────────────────

/// Configuration for webhook verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifyConfig {
    /// Maximum age of timestamp in seconds (replay protection).
    pub max_age_secs: u64,
    /// Maximum clock skew tolerance for future timestamps.
    pub max_future_secs: u64,
    /// Whether to require a timestamp.
    pub require_timestamp: bool,
}

impl Default for VerifyConfig {
    fn default() -> Self {
        Self {
            max_age_secs: 300,
            max_future_secs: 60,
            require_timestamp: true,
        }
    }
}

/// Verifies webhook signatures.
#[derive(Debug, Clone)]
pub struct WebhookVerifier {
    secrets: Vec<SigningSecret>,
    config: VerifyConfig,
}

impl WebhookVerifier {
    /// Create a verifier with a single secret.
    pub fn new(secret: SigningSecret, config: VerifyConfig) -> Self {
        Self {
            secrets: vec![secret],
            config,
        }
    }

    /// Create with multiple secrets for key rotation.
    pub fn with_secrets(
        secrets: Vec<SigningSecret>,
        config: VerifyConfig,
    ) -> Result<Self, VerifyError> {
        if secrets.is_empty() {
            return Err(VerifyError::NoSigningSecrets);
        }
        Ok(Self { secrets, config })
    }

    /// Verify a webhook request.
    ///
    /// - `signature_header`: The signature header value.
    /// - `timestamp`: Epoch seconds from the request header.
    /// - `payload`: The raw request body.
    /// - `now`: Current epoch seconds.
    pub fn verify(
        &self,
        signature_header: &str,
        timestamp: Option<u64>,
        payload: &str,
        now: u64,
    ) -> Result<VerifyResult, VerifyError> {
        // Validate timestamp.
        if self.config.require_timestamp {
            let ts = timestamp.ok_or(VerifyError::MissingTimestamp)?;
            if now > ts {
                let age = now - ts;
                if age > self.config.max_age_secs {
                    return Err(VerifyError::TimestampTooOld {
                        age_secs: age,
                        max_secs: self.config.max_age_secs,
                    });
                }
            } else {
                let ahead = ts - now;
                if ahead > self.config.max_future_secs {
                    return Err(VerifyError::TimestampInFuture { ahead_secs: ahead });
                }
            }
        }

        // Parse signature.
        let parsed = SignatureHeader::parse(signature_header)?;
        if parsed.is_empty() {
            return Err(VerifyError::MissingSignature);
        }

        // Try each secret against each signature.
        let ts = timestamp.unwrap_or(0);
        let canonical = canonical_payload(ts, payload);

        for secret in &self.secrets {
            let expected_mac = hmac_sha256(&secret.key, canonical.as_bytes());
            let expected_hex = bytes_to_hex(&expected_mac);

            for sig in &parsed {
                // Try both raw hex comparison and hex-decoded comparison.
                if constant_time_eq(sig.signature.as_bytes(), expected_hex.as_bytes()) {
                    return Ok(VerifyResult {
                        valid: true,
                        secret_label: secret.label.clone(),
                        scheme: sig.scheme.clone(),
                    });
                }
                // Also try decoding hex and comparing bytes.
                if let Ok(sig_bytes) = hex_to_bytes(&sig.signature) {
                    if constant_time_eq(&sig_bytes, &expected_mac) {
                        return Ok(VerifyResult {
                            valid: true,
                            secret_label: secret.label.clone(),
                            scheme: sig.scheme.clone(),
                        });
                    }
                }
            }
        }

        Err(VerifyError::SignatureMismatch)
    }

    /// Verify using headers map.
    pub fn verify_from_headers(
        &self,
        headers: &HashMap<String, String>,
        payload: &str,
        now: u64,
        sig_header_name: &str,
        ts_header_name: &str,
    ) -> Result<VerifyResult, VerifyError> {
        let sig = headers
            .get(&sig_header_name.to_lowercase())
            .ok_or(VerifyError::MissingSignature)?;

        let ts = if self.config.require_timestamp {
            let ts_str = headers
                .get(&ts_header_name.to_lowercase())
                .ok_or(VerifyError::MissingTimestamp)?;
            Some(
                ts_str
                    .parse::<u64>()
                    .map_err(|_| VerifyError::InvalidTimestamp(ts_str.clone()))?,
            )
        } else {
            headers
                .get(&ts_header_name.to_lowercase())
                .and_then(|s| s.parse::<u64>().ok())
        };

        self.verify(sig, ts, payload, now)
    }
}

/// Result of a successful verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifyResult {
    /// Whether the signature is valid.
    pub valid: bool,
    /// Which signing secret matched.
    pub secret_label: String,
    /// Which signature scheme was used.
    pub scheme: String,
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_secret() -> SigningSecret {
        SigningSecret::from_string("whsec_test_secret_key_123", "primary")
    }

    #[test]
    fn test_sign_and_verify() {
        let secret = test_secret();
        let signer = WebhookSigner::new(secret.clone());
        let verifier = WebhookVerifier::new(secret, VerifyConfig::default());

        let sig = signer.sign(1000, "hello world").unwrap();
        let result = verifier.verify(&sig, Some(1000), "hello world", 1001).unwrap();
        assert!(result.valid);
    }

    #[test]
    fn test_verify_wrong_payload() {
        let secret = test_secret();
        let signer = WebhookSigner::new(secret.clone());
        let verifier = WebhookVerifier::new(secret, VerifyConfig::default());

        let sig = signer.sign(1000, "hello world").unwrap();
        let err = verifier.verify(&sig, Some(1000), "different payload", 1001).unwrap_err();
        assert_eq!(err, VerifyError::SignatureMismatch);
    }

    #[test]
    fn test_verify_wrong_timestamp() {
        let secret = test_secret();
        let signer = WebhookSigner::new(secret.clone());
        let verifier = WebhookVerifier::new(secret, VerifyConfig::default());

        let sig = signer.sign(1000, "payload").unwrap();
        let err = verifier.verify(&sig, Some(999), "payload", 1001).unwrap_err();
        assert_eq!(err, VerifyError::SignatureMismatch);
    }

    #[test]
    fn test_timestamp_too_old() {
        let secret = test_secret();
        let verifier = WebhookVerifier::new(secret, VerifyConfig {
            max_age_secs: 300,
            ..Default::default()
        });

        let err = verifier.verify("fakesig", Some(1000), "payload", 2000).unwrap_err();
        assert!(matches!(err, VerifyError::TimestampTooOld { .. }));
    }

    #[test]
    fn test_timestamp_in_future() {
        let secret = test_secret();
        let verifier = WebhookVerifier::new(secret, VerifyConfig {
            max_future_secs: 60,
            ..Default::default()
        });

        let err = verifier.verify("fakesig", Some(2000), "payload", 1000).unwrap_err();
        assert!(matches!(err, VerifyError::TimestampInFuture { .. }));
    }

    #[test]
    fn test_missing_timestamp() {
        let secret = test_secret();
        let verifier = WebhookVerifier::new(secret, VerifyConfig {
            require_timestamp: true,
            ..Default::default()
        });

        let err = verifier.verify("fakesig", None, "payload", 1000).unwrap_err();
        assert_eq!(err, VerifyError::MissingTimestamp);
    }

    #[test]
    fn test_no_timestamp_required() {
        let secret = test_secret();
        let signer = WebhookSigner::new(secret.clone());
        let verifier = WebhookVerifier::new(secret, VerifyConfig {
            require_timestamp: false,
            ..Default::default()
        });

        let sig = signer.sign(0, "payload").unwrap();
        let result = verifier.verify(&sig, None, "payload", 1000).unwrap();
        assert!(result.valid);
    }

    #[test]
    fn test_github_style_signature() {
        let secret = test_secret();
        let signer = WebhookSigner::new(secret.clone());
        let verifier = WebhookVerifier::new(secret, VerifyConfig {
            require_timestamp: false,
            ..Default::default()
        });

        let sig = signer.sign(0, "payload").unwrap();
        let github_header = format!("sha256={sig}");
        let result = verifier.verify(&github_header, None, "payload", 1000).unwrap();
        assert!(result.valid);
        assert_eq!(result.scheme, "sha256");
    }

    #[test]
    fn test_key_rotation() {
        let old_secret = SigningSecret::from_string("old_key", "old").retire();
        let new_secret = SigningSecret::from_string("new_key", "new");

        // Sign with old key manually.
        let canonical = canonical_payload(1000, "payload");
        let old_sig = bytes_to_hex(&hmac_sha256(b"old_key", canonical.as_bytes()));

        // Verifier has both keys.
        let verifier = WebhookVerifier::with_secrets(
            vec![new_secret, old_secret],
            VerifyConfig::default(),
        ).unwrap();

        let result = verifier.verify(&old_sig, Some(1000), "payload", 1001).unwrap();
        assert!(result.valid);
        assert_eq!(result.secret_label, "old");
    }

    #[test]
    fn test_signer_with_id() {
        let secret = test_secret();
        let signer = WebhookSigner::new(secret);

        let sig = signer.sign_with_id("msg_123", 1000, "payload").unwrap();
        assert!(!sig.is_empty());
    }

    #[test]
    fn test_parse_signature_github() {
        let parsed = SignatureHeader::parse("sha256=abcdef0123456789").unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].scheme, "sha256");
        assert_eq!(parsed[0].signature, "abcdef0123456789");
    }

    #[test]
    fn test_parse_signature_stripe_style() {
        let parsed = SignatureHeader::parse("t=1234,v1=abcdef,v1=fedcba").unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].scheme, "v1");
        assert_eq!(parsed[1].scheme, "v1");
    }

    #[test]
    fn test_parse_signature_empty() {
        let err = SignatureHeader::parse("").unwrap_err();
        assert_eq!(err, VerifyError::MissingSignature);
    }

    #[test]
    fn test_canonical_payload() {
        let c = canonical_payload(1234567890, "{\"event\":\"test\"}");
        assert_eq!(c, "1234567890.{\"event\":\"test\"}");
    }

    #[test]
    fn test_canonical_with_id() {
        let c = canonical_payload_with_id("msg_1", 1000, "data");
        assert_eq!(c, "msg_1.1000.data");
    }

    #[test]
    fn test_verify_from_headers() {
        let secret = test_secret();
        let signer = WebhookSigner::new(secret.clone());
        let verifier = WebhookVerifier::new(secret, VerifyConfig::default());

        let sig = signer.sign(1000, "body data").unwrap();
        let mut headers = HashMap::new();
        headers.insert("x-signature".to_string(), sig);
        headers.insert("x-timestamp".to_string(), "1000".to_string());

        let result = verifier
            .verify_from_headers(&headers, "body data", 1001, "x-signature", "x-timestamp")
            .unwrap();
        assert!(result.valid);
    }

    #[test]
    fn test_no_signing_secrets() {
        let err = WebhookVerifier::with_secrets(vec![], VerifyConfig::default()).unwrap_err();
        assert_eq!(err, VerifyError::NoSigningSecrets);
    }

    #[test]
    fn test_signer_secret_count() {
        let mut signer = WebhookSigner::new(test_secret());
        assert_eq!(signer.secret_count(), 1);
        signer.add_secret(SigningSecret::from_string("extra", "extra"));
        assert_eq!(signer.secret_count(), 2);
    }

    #[test]
    fn test_constant_time_eq() {
        assert!(constant_time_eq(b"abc", b"abc"));
        assert!(!constant_time_eq(b"abc", b"abd"));
        assert!(!constant_time_eq(b"abc", b"ab"));
    }

    #[test]
    fn test_hex_roundtrip() {
        let data = vec![0xde, 0xad, 0xbe, 0xef];
        let hex = bytes_to_hex(&data);
        assert_eq!(hex, "deadbeef");
        let decoded = hex_to_bytes(&hex).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_error_display() {
        let err = VerifyError::TimestampTooOld { age_secs: 600, max_secs: 300 };
        let msg = err.to_string();
        assert!(msg.contains("600"));
        assert!(msg.contains("300"));
    }
}
