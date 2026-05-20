//! Lightweight JWT implementation — header/payload encoding, HMAC-SHA256
//! signing/verification, claims validation (exp, nbf, iss, aud), token
//! parsing, custom claims, and token refresh logic.
//!
//! Replaces jsonwebtoken, jose, and jwt-decode with a self-contained
//! pure-Rust JWT implementation.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

// ── Errors ─────────────────────────────────────────────────────

/// JWT errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JwtError {
    /// Token has wrong number of segments.
    MalformedToken,
    /// Base64 decoding failed.
    Base64DecodeError(String),
    /// JSON parsing failed.
    JsonParseError(String),
    /// Signature verification failed.
    InvalidSignature,
    /// Token has expired.
    TokenExpired { exp: i64, now: i64 },
    /// Token not yet valid (nbf).
    TokenNotYetValid { nbf: i64, now: i64 },
    /// Issuer mismatch.
    IssuerMismatch { expected: String, got: String },
    /// Audience mismatch.
    AudienceMismatch { expected: String, got: String },
    /// Missing required claim.
    MissingClaim(String),
    /// Unsupported algorithm.
    UnsupportedAlgorithm(String),
}

impl std::fmt::Display for JwtError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MalformedToken => write!(f, "malformed JWT token"),
            Self::Base64DecodeError(s) => write!(f, "base64 decode error: {s}"),
            Self::JsonParseError(s) => write!(f, "JSON parse error: {s}"),
            Self::InvalidSignature => write!(f, "invalid JWT signature"),
            Self::TokenExpired { exp, now } => {
                write!(f, "token expired at {exp}, current time {now}")
            }
            Self::TokenNotYetValid { nbf, now } => {
                write!(f, "token not valid before {nbf}, current time {now}")
            }
            Self::IssuerMismatch { expected, got } => {
                write!(f, "issuer mismatch: expected {expected}, got {got}")
            }
            Self::AudienceMismatch { expected, got } => {
                write!(f, "audience mismatch: expected {expected}, got {got}")
            }
            Self::MissingClaim(c) => write!(f, "missing claim: {c}"),
            Self::UnsupportedAlgorithm(a) => write!(f, "unsupported algorithm: {a}"),
        }
    }
}

impl std::error::Error for JwtError {}

// ── Inline SHA-256 + HMAC ──────────────────────────────────────

const SHA256_H0: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

const SHA256_K: [u32; 64] = [
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
];

fn sha256_process_block(state: &mut [u32; 8], block: &[u8; 64]) {
    let mut w = [0u32; 64];
    for i in 0..16 {
        w[i] = u32::from_be_bytes([block[i*4], block[i*4+1], block[i*4+2], block[i*4+3]]);
    }
    for i in 16..64 {
        let s0 = w[i-15].rotate_right(7) ^ w[i-15].rotate_right(18) ^ (w[i-15] >> 3);
        let s1 = w[i-2].rotate_right(17) ^ w[i-2].rotate_right(19) ^ (w[i-2] >> 10);
        w[i] = w[i-16].wrapping_add(s0).wrapping_add(w[i-7]).wrapping_add(s1);
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
    let bit_len = (data.len() as u64) * 8;
    let mut buf = data.to_vec();
    buf.push(0x80);
    while buf.len() % 64 != 56 { buf.push(0x00); }
    buf.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in buf.chunks_exact(64) {
        let block: [u8; 64] = chunk.try_into().unwrap();
        sha256_process_block(&mut state, &block);
    }
    let mut out = [0u8; 32];
    for (i, word) in state.iter().enumerate() {
        out[i*4..(i+1)*4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

fn hmac_sha256(key: &[u8], message: &[u8]) -> [u8; 32] {
    let k = if key.len() > 64 { sha256(key).to_vec() } else { key.to_vec() };
    let mut k_padded = [0u8; 64];
    k_padded[..k.len()].copy_from_slice(&k);
    let mut ipad = [0x36u8; 64];
    let mut opad = [0x5cu8; 64];
    for i in 0..64 { ipad[i] ^= k_padded[i]; opad[i] ^= k_padded[i]; }
    let mut inner = ipad.to_vec();
    inner.extend_from_slice(message);
    let inner_hash = sha256(&inner);
    let mut outer = opad.to_vec();
    outer.extend_from_slice(&inner_hash);
    sha256(&outer)
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() { return false; }
    let mut diff = 0u8;
    for i in 0..a.len() { diff |= a[i] ^ b[i]; }
    diff == 0
}

// ── Base64url ──────────────────────────────────────────────────

fn base64url_encode(data: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut result = String::with_capacity((data.len() + 2) / 3 * 4);
    let mut i = 0;
    while i < data.len() {
        let b0 = data[i] as u32;
        let b1 = if i + 1 < data.len() { data[i + 1] as u32 } else { 0 };
        let b2 = if i + 2 < data.len() { data[i + 2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;

        result.push(CHARS[((triple >> 18) & 0x3F) as usize] as char);
        result.push(CHARS[((triple >> 12) & 0x3F) as usize] as char);

        if i + 1 < data.len() {
            result.push(CHARS[((triple >> 6) & 0x3F) as usize] as char);
        }
        if i + 2 < data.len() {
            result.push(CHARS[(triple & 0x3F) as usize] as char);
        }
        i += 3;
    }
    result
}

fn base64url_decode(input: &str) -> Result<Vec<u8>, JwtError> {
    let lookup = |c: u8| -> Result<u32, JwtError> {
        match c {
            b'A'..=b'Z' => Ok((c - b'A') as u32),
            b'a'..=b'z' => Ok((c - b'a' + 26) as u32),
            b'0'..=b'9' => Ok((c - b'0' + 52) as u32),
            b'-' => Ok(62),
            b'_' => Ok(63),
            _ => Err(JwtError::Base64DecodeError(format!("invalid char: {}", c as char))),
        }
    };

    // Strip padding.
    let input = input.trim_end_matches('=');
    let bytes = input.as_bytes();
    let mut result = Vec::with_capacity(bytes.len() * 3 / 4);

    let mut i = 0;
    while i < bytes.len() {
        let c0 = lookup(bytes[i])?;
        let c1 = if i + 1 < bytes.len() { lookup(bytes[i + 1])? } else { 0 };
        let c2 = if i + 2 < bytes.len() { lookup(bytes[i + 2])? } else { 0 };
        let c3 = if i + 3 < bytes.len() { lookup(bytes[i + 3])? } else { 0 };

        let triple = (c0 << 18) | (c1 << 12) | (c2 << 6) | c3;

        result.push(((triple >> 16) & 0xFF) as u8);
        if i + 2 < bytes.len() {
            result.push(((triple >> 8) & 0xFF) as u8);
        }
        if i + 3 < bytes.len() {
            result.push((triple & 0xFF) as u8);
        }
        i += 4;
    }

    Ok(result)
}

// ── JWT Types ──────────────────────────────────────────────────

/// Supported JWT algorithms.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Algorithm {
    /// HMAC-SHA256.
    HS256,
}

impl Algorithm {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::HS256 => "HS256",
        }
    }

    pub fn from_str_val(s: &str) -> Result<Self, JwtError> {
        match s {
            "HS256" => Ok(Self::HS256),
            other => Err(JwtError::UnsupportedAlgorithm(other.to_string())),
        }
    }
}

/// JWT header.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JwtHeader {
    pub alg: String,
    pub typ: String,
}

impl Default for JwtHeader {
    fn default() -> Self {
        Self {
            alg: "HS256".to_string(),
            typ: "JWT".to_string(),
        }
    }
}

/// JWT claims (registered + custom).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claims {
    /// Subject.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sub: Option<String>,
    /// Issuer.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub iss: Option<String>,
    /// Audience.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aud: Option<String>,
    /// Expiration time (Unix timestamp).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exp: Option<i64>,
    /// Not before (Unix timestamp).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nbf: Option<i64>,
    /// Issued at (Unix timestamp).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub iat: Option<i64>,
    /// JWT ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub jti: Option<String>,
    /// Custom claims.
    #[serde(flatten)]
    pub custom: HashMap<String, Value>,
}

impl Claims {
    /// Create minimal claims with a subject.
    pub fn new(sub: &str) -> Self {
        Self {
            sub: Some(sub.to_string()),
            iss: None,
            aud: None,
            exp: None,
            nbf: None,
            iat: Some(Utc::now().timestamp()),
            jti: None,
            custom: HashMap::new(),
        }
    }

    /// Set expiration to N seconds from now.
    pub fn with_expiry(mut self, secs: i64) -> Self {
        self.exp = Some(Utc::now().timestamp() + secs);
        self
    }

    /// Set issuer.
    pub fn with_issuer(mut self, iss: &str) -> Self {
        self.iss = Some(iss.to_string());
        self
    }

    /// Set audience.
    pub fn with_audience(mut self, aud: &str) -> Self {
        self.aud = Some(aud.to_string());
        self
    }

    /// Set not-before to N seconds from now.
    pub fn with_nbf(mut self, secs: i64) -> Self {
        self.nbf = Some(Utc::now().timestamp() + secs);
        self
    }

    /// Set a custom claim.
    pub fn with_claim(mut self, key: &str, value: Value) -> Self {
        self.custom.insert(key.to_string(), value);
        self
    }

    /// Get a custom claim value.
    pub fn get_claim(&self, key: &str) -> Option<&Value> {
        self.custom.get(key)
    }
}

// ── Validation Options ─────────────────────────────────────────

/// Options for validating a JWT.
#[derive(Debug, Clone)]
pub struct ValidationOptions {
    /// Whether to validate expiration.
    pub validate_exp: bool,
    /// Whether to validate not-before.
    pub validate_nbf: bool,
    /// Expected issuer (None = don't check).
    pub expected_issuer: Option<String>,
    /// Expected audience (None = don't check).
    pub expected_audience: Option<String>,
    /// Clock skew tolerance in seconds.
    pub leeway_secs: i64,
}

impl Default for ValidationOptions {
    fn default() -> Self {
        Self {
            validate_exp: true,
            validate_nbf: true,
            expected_issuer: None,
            expected_audience: None,
            leeway_secs: 0,
        }
    }
}

impl ValidationOptions {
    /// Set expected issuer.
    pub fn with_issuer(mut self, iss: &str) -> Self {
        self.expected_issuer = Some(iss.to_string());
        self
    }

    /// Set expected audience.
    pub fn with_audience(mut self, aud: &str) -> Self {
        self.expected_audience = Some(aud.to_string());
        self
    }

    /// Set clock skew tolerance.
    pub fn with_leeway(mut self, secs: i64) -> Self {
        self.leeway_secs = secs;
        self
    }
}

// ── Decoded Token ──────────────────────────────────────────────

/// A decoded JWT token.
#[derive(Debug, Clone)]
pub struct DecodedToken {
    pub header: JwtHeader,
    pub claims: Claims,
}

// ── JWT Codec ──────────────────────────────────────────────────

/// JWT encoder/decoder with HMAC-SHA256 signing.
pub struct JwtCodec {
    secret: Vec<u8>,
}

impl JwtCodec {
    /// Create a new codec with the given secret key.
    pub fn new(secret: &[u8]) -> Self {
        Self {
            secret: secret.to_vec(),
        }
    }

    /// Encode and sign a JWT from claims.
    pub fn encode(&self, claims: &Claims) -> Result<String, JwtError> {
        let header = JwtHeader::default();
        let header_json = serde_json::to_string(&header)
            .map_err(|e| JwtError::JsonParseError(e.to_string()))?;
        let claims_json = serde_json::to_string(claims)
            .map_err(|e| JwtError::JsonParseError(e.to_string()))?;

        let header_b64 = base64url_encode(header_json.as_bytes());
        let payload_b64 = base64url_encode(claims_json.as_bytes());
        let signing_input = format!("{header_b64}.{payload_b64}");
        let signature = hmac_sha256(&self.secret, signing_input.as_bytes());
        let sig_b64 = base64url_encode(&signature);

        Ok(format!("{signing_input}.{sig_b64}"))
    }

    /// Decode and verify a JWT, returning the claims.
    pub fn decode(
        &self,
        token: &str,
        options: &ValidationOptions,
    ) -> Result<DecodedToken, JwtError> {
        let parts: Vec<&str> = token.split('.').collect();
        if parts.len() != 3 {
            return Err(JwtError::MalformedToken);
        }

        // Verify signature.
        let signing_input = format!("{}.{}", parts[0], parts[1]);
        let expected_sig = hmac_sha256(&self.secret, signing_input.as_bytes());
        let actual_sig = base64url_decode(parts[2])?;
        if !constant_time_eq(&expected_sig, &actual_sig) {
            return Err(JwtError::InvalidSignature);
        }

        // Decode header.
        let header_bytes = base64url_decode(parts[0])?;
        let header_str = std::str::from_utf8(&header_bytes)
            .map_err(|e| JwtError::Base64DecodeError(e.to_string()))?;
        let header: JwtHeader = serde_json::from_str(header_str)
            .map_err(|e| JwtError::JsonParseError(e.to_string()))?;

        // Verify algorithm.
        Algorithm::from_str_val(&header.alg)?;

        // Decode claims.
        let claims_bytes = base64url_decode(parts[1])?;
        let claims_str = std::str::from_utf8(&claims_bytes)
            .map_err(|e| JwtError::Base64DecodeError(e.to_string()))?;
        let claims: Claims = serde_json::from_str(claims_str)
            .map_err(|e| JwtError::JsonParseError(e.to_string()))?;

        // Validate claims.
        let now = Utc::now().timestamp();

        if options.validate_exp {
            if let Some(exp) = claims.exp {
                if now > exp + options.leeway_secs {
                    return Err(JwtError::TokenExpired { exp, now });
                }
            }
        }

        if options.validate_nbf {
            if let Some(nbf) = claims.nbf {
                if now < nbf - options.leeway_secs {
                    return Err(JwtError::TokenNotYetValid { nbf, now });
                }
            }
        }

        if let Some(expected_iss) = &options.expected_issuer {
            match &claims.iss {
                Some(iss) if iss == expected_iss => {}
                Some(iss) => {
                    return Err(JwtError::IssuerMismatch {
                        expected: expected_iss.clone(),
                        got: iss.clone(),
                    });
                }
                None => {
                    return Err(JwtError::MissingClaim("iss".to_string()));
                }
            }
        }

        if let Some(expected_aud) = &options.expected_audience {
            match &claims.aud {
                Some(aud) if aud == expected_aud => {}
                Some(aud) => {
                    return Err(JwtError::AudienceMismatch {
                        expected: expected_aud.clone(),
                        got: aud.clone(),
                    });
                }
                None => {
                    return Err(JwtError::MissingClaim("aud".to_string()));
                }
            }
        }

        Ok(DecodedToken { header, claims })
    }

    /// Decode without verifying the signature (unsafe — for inspection only).
    pub fn decode_unverified(token: &str) -> Result<DecodedToken, JwtError> {
        let parts: Vec<&str> = token.split('.').collect();
        if parts.len() != 3 {
            return Err(JwtError::MalformedToken);
        }

        let header_bytes = base64url_decode(parts[0])?;
        let header_str = std::str::from_utf8(&header_bytes)
            .map_err(|e| JwtError::Base64DecodeError(e.to_string()))?;
        let header: JwtHeader = serde_json::from_str(header_str)
            .map_err(|e| JwtError::JsonParseError(e.to_string()))?;

        let claims_bytes = base64url_decode(parts[1])?;
        let claims_str = std::str::from_utf8(&claims_bytes)
            .map_err(|e| JwtError::Base64DecodeError(e.to_string()))?;
        let claims: Claims = serde_json::from_str(claims_str)
            .map_err(|e| JwtError::JsonParseError(e.to_string()))?;

        Ok(DecodedToken { header, claims })
    }

    /// Refresh a token: decode, update exp/iat, re-sign.
    pub fn refresh(
        &self,
        token: &str,
        new_expiry_secs: i64,
        options: &ValidationOptions,
    ) -> Result<String, JwtError> {
        let decoded = self.decode(token, options)?;
        let mut claims = decoded.claims;
        let now = Utc::now().timestamp();
        claims.iat = Some(now);
        claims.exp = Some(now + new_expiry_secs);
        self.encode(&claims)
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn codec() -> JwtCodec {
        JwtCodec::new(b"super-secret-key-for-testing")
    }

    fn no_exp_opts() -> ValidationOptions {
        ValidationOptions {
            validate_exp: false,
            validate_nbf: false,
            ..Default::default()
        }
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let c = codec();
        let claims = Claims::new("user123").with_expiry(3600);
        let token = c.encode(&claims).unwrap();
        let decoded = c.decode(&token, &ValidationOptions::default()).unwrap();
        assert_eq!(decoded.claims.sub, Some("user123".to_string()));
    }

    #[test]
    fn test_invalid_signature() {
        let c1 = JwtCodec::new(b"key1");
        let c2 = JwtCodec::new(b"key2");
        let claims = Claims::new("user");
        let token = c1.encode(&claims).unwrap();
        let err = c2.decode(&token, &no_exp_opts()).unwrap_err();
        assert_eq!(err, JwtError::InvalidSignature);
    }

    #[test]
    fn test_malformed_token() {
        let c = codec();
        assert_eq!(
            c.decode("not.a.valid.jwt.token", &no_exp_opts()).unwrap_err(),
            JwtError::MalformedToken
        );
        assert_eq!(
            c.decode("onlyone", &no_exp_opts()).unwrap_err(),
            JwtError::MalformedToken
        );
    }

    #[test]
    fn test_expired_token() {
        let c = codec();
        let mut claims = Claims::new("user");
        claims.exp = Some(Utc::now().timestamp() - 100);
        let token = c.encode(&claims).unwrap();
        match c.decode(&token, &ValidationOptions::default()) {
            Err(JwtError::TokenExpired { .. }) => {}
            other => panic!("expected TokenExpired, got {other:?}"),
        }
    }

    #[test]
    fn test_nbf_not_yet_valid() {
        let c = codec();
        let mut claims = Claims::new("user");
        claims.exp = Some(Utc::now().timestamp() + 7200);
        claims.nbf = Some(Utc::now().timestamp() + 3600);
        let token = c.encode(&claims).unwrap();
        match c.decode(&token, &ValidationOptions::default()) {
            Err(JwtError::TokenNotYetValid { .. }) => {}
            other => panic!("expected TokenNotYetValid, got {other:?}"),
        }
    }

    #[test]
    fn test_issuer_validation() {
        let c = codec();
        let claims = Claims::new("user").with_issuer("my-app").with_expiry(3600);
        let token = c.encode(&claims).unwrap();
        let opts = ValidationOptions::default().with_issuer("my-app");
        assert!(c.decode(&token, &opts).is_ok());

        let opts_wrong = ValidationOptions::default().with_issuer("other-app");
        match c.decode(&token, &opts_wrong) {
            Err(JwtError::IssuerMismatch { .. }) => {}
            other => panic!("expected IssuerMismatch, got {other:?}"),
        }
    }

    #[test]
    fn test_audience_validation() {
        let c = codec();
        let claims = Claims::new("user").with_audience("web-client").with_expiry(3600);
        let token = c.encode(&claims).unwrap();
        let opts = ValidationOptions::default().with_audience("web-client");
        assert!(c.decode(&token, &opts).is_ok());

        let opts_wrong = ValidationOptions::default().with_audience("mobile");
        match c.decode(&token, &opts_wrong) {
            Err(JwtError::AudienceMismatch { .. }) => {}
            other => panic!("expected AudienceMismatch, got {other:?}"),
        }
    }

    #[test]
    fn test_custom_claims() {
        let c = codec();
        let claims = Claims::new("user")
            .with_expiry(3600)
            .with_claim("role", Value::String("admin".to_string()))
            .with_claim("level", Value::Number(serde_json::Number::from(5)));
        let token = c.encode(&claims).unwrap();
        let decoded = c.decode(&token, &ValidationOptions::default()).unwrap();
        assert_eq!(
            decoded.claims.get_claim("role"),
            Some(&Value::String("admin".to_string()))
        );
        assert_eq!(
            decoded.claims.get_claim("level"),
            Some(&Value::Number(serde_json::Number::from(5)))
        );
    }

    #[test]
    fn test_decode_unverified() {
        let c = codec();
        let claims = Claims::new("user");
        let token = c.encode(&claims).unwrap();
        let decoded = JwtCodec::decode_unverified(&token).unwrap();
        assert_eq!(decoded.claims.sub, Some("user".to_string()));
        assert_eq!(decoded.header.alg, "HS256");
    }

    #[test]
    fn test_refresh_token() {
        let c = codec();
        let claims = Claims::new("user").with_expiry(3600);
        let token = c.encode(&claims).unwrap();
        let new_token = c.refresh(&token, 7200, &ValidationOptions::default()).unwrap();
        assert_ne!(token, new_token);
        let decoded = c.decode(&new_token, &ValidationOptions::default()).unwrap();
        assert_eq!(decoded.claims.sub, Some("user".to_string()));
    }

    #[test]
    fn test_leeway() {
        let c = codec();
        let mut claims = Claims::new("user");
        claims.exp = Some(Utc::now().timestamp() - 5);
        let token = c.encode(&claims).unwrap();
        // Without leeway: expired.
        assert!(c.decode(&token, &ValidationOptions::default()).is_err());
        // With 10s leeway: ok.
        let opts = ValidationOptions::default().with_leeway(10);
        assert!(c.decode(&token, &opts).is_ok());
    }

    #[test]
    fn test_base64url_roundtrip() {
        let data = b"Hello, World!";
        let encoded = base64url_encode(data);
        let decoded = base64url_decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_base64url_binary() {
        let data: Vec<u8> = (0..=255).collect();
        let encoded = base64url_encode(&data);
        let decoded = base64url_decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_token_three_parts() {
        let c = codec();
        let claims = Claims::new("user");
        let token = c.encode(&claims).unwrap();
        let parts: Vec<&str> = token.split('.').collect();
        assert_eq!(parts.len(), 3);
    }

    #[test]
    fn test_algorithm_from_str() {
        assert_eq!(Algorithm::from_str_val("HS256").unwrap(), Algorithm::HS256);
        assert!(Algorithm::from_str_val("RS256").is_err());
    }

    #[test]
    fn test_missing_issuer_claim() {
        let c = codec();
        let claims = Claims::new("user").with_expiry(3600); // no issuer
        let token = c.encode(&claims).unwrap();
        let opts = ValidationOptions::default().with_issuer("required-iss");
        match c.decode(&token, &opts) {
            Err(JwtError::MissingClaim(c)) => assert_eq!(c, "iss"),
            other => panic!("expected MissingClaim, got {other:?}"),
        }
    }

    #[test]
    fn test_error_display() {
        let e = JwtError::InvalidSignature;
        assert_eq!(e.to_string(), "invalid JWT signature");
        let e2 = JwtError::TokenExpired { exp: 100, now: 200 };
        assert!(e2.to_string().contains("expired"));
    }
}
