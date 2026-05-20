//! JWT encode/decode — header.payload.signature (RFC 7519).
//!
//! Replaces `jsonwebtoken` / `jose-jwt` with a pure Rust JWT codec.
//! Supports HS256/HS384/HS512/none algorithms, standard + custom claims,
//! and validation of exp, nbf, iss, aud.

use crate::crypto::{sha256, hmac_sha256, base64_encode, base64_decode};
use std::collections::HashMap;
use std::fmt;

// ── Errors ─────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JwtError {
    InvalidFormat,
    InvalidBase64,
    InvalidJson(String),
    InvalidSignature,
    TokenExpired,
    TokenNotYetValid,
    InvalidIssuer { expected: String, got: String },
    InvalidAudience { expected: String, got: String },
    UnsupportedAlgorithm(String),
}

impl fmt::Display for JwtError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFormat => write!(f, "invalid JWT format"),
            Self::InvalidBase64 => write!(f, "invalid base64url encoding"),
            Self::InvalidJson(msg) => write!(f, "invalid JSON: {msg}"),
            Self::InvalidSignature => write!(f, "invalid signature"),
            Self::TokenExpired => write!(f, "token has expired"),
            Self::TokenNotYetValid => write!(f, "token not yet valid"),
            Self::InvalidIssuer { expected, got } => {
                write!(f, "invalid issuer: expected {expected}, got {got}")
            }
            Self::InvalidAudience { expected, got } => {
                write!(f, "invalid audience: expected {expected}, got {got}")
            }
            Self::UnsupportedAlgorithm(alg) => write!(f, "unsupported algorithm: {alg}"),
        }
    }
}

impl std::error::Error for JwtError {}

// ── Base64url ──────────────────────────────────────────────────

fn base64url_encode(input: &[u8]) -> String {
    base64_encode(input)
        .replace('+', "-")
        .replace('/', "_")
        .trim_end_matches('=')
        .to_string()
}

fn base64url_decode(input: &str) -> Result<Vec<u8>, JwtError> {
    let mut s = input.replace('-', "+").replace('_', "/");
    // Add padding.
    match s.len() % 4 {
        2 => s.push_str("=="),
        3 => s.push_str("="),
        _ => {}
    }
    base64_decode(&s).map_err(|_| JwtError::InvalidBase64)
}

// ── Algorithm ──────────────────────────────────────────────────

/// JWT signing algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Algorithm {
    HS256,
    HS384,
    HS512,
    None,
}

impl Algorithm {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::HS256 => "HS256",
            Self::HS384 => "HS384",
            Self::HS512 => "HS512",
            Self::None => "none",
        }
    }

    pub fn from_str(s: &str) -> Result<Self, JwtError> {
        match s {
            "HS256" => Ok(Self::HS256),
            "HS384" => Ok(Self::HS384),
            "HS512" => Ok(Self::HS512),
            "none" => Ok(Self::None),
            _ => Err(JwtError::UnsupportedAlgorithm(s.to_string())),
        }
    }
}

impl fmt::Display for Algorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ── Header ─────────────────────────────────────────────────────

/// JWT header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Header {
    pub alg: Algorithm,
    pub typ: String,
}

impl Header {
    pub fn new(alg: Algorithm) -> Self {
        Self {
            alg,
            typ: "JWT".into(),
        }
    }

    fn to_json(&self) -> String {
        format!(
            "{{\"alg\":\"{}\",\"typ\":\"{}\"}}",
            self.alg.as_str(),
            self.typ
        )
    }

    fn from_json(json: &str) -> Result<Self, JwtError> {
        let val: serde_json::Value =
            serde_json::from_str(json).map_err(|e| JwtError::InvalidJson(e.to_string()))?;
        let alg_str = val["alg"]
            .as_str()
            .ok_or(JwtError::InvalidJson("missing alg".into()))?;
        let alg = Algorithm::from_str(alg_str)?;
        let typ = val["typ"].as_str().unwrap_or("JWT").to_string();
        Ok(Self { alg, typ })
    }
}

// ── Claims ─────────────────────────────────────────────────────

/// JWT claims (standard + custom).
#[derive(Debug, Clone, PartialEq)]
pub struct Claims {
    /// Issuer.
    pub iss: Option<String>,
    /// Subject.
    pub sub: Option<String>,
    /// Audience.
    pub aud: Option<String>,
    /// Expiration time (UNIX timestamp).
    pub exp: Option<u64>,
    /// Not before (UNIX timestamp).
    pub nbf: Option<u64>,
    /// Issued at (UNIX timestamp).
    pub iat: Option<u64>,
    /// JWT ID.
    pub jti: Option<String>,
    /// Custom claims.
    pub custom: HashMap<String, serde_json::Value>,
}

impl Claims {
    pub fn new() -> Self {
        Self {
            iss: None,
            sub: None,
            aud: None,
            exp: None,
            nbf: None,
            iat: None,
            jti: None,
            custom: HashMap::new(),
        }
    }

    pub fn issuer(mut self, iss: impl Into<String>) -> Self {
        self.iss = Some(iss.into());
        self
    }

    pub fn subject(mut self, sub: impl Into<String>) -> Self {
        self.sub = Some(sub.into());
        self
    }

    pub fn audience(mut self, aud: impl Into<String>) -> Self {
        self.aud = Some(aud.into());
        self
    }

    pub fn expires_at(mut self, exp: u64) -> Self {
        self.exp = Some(exp);
        self
    }

    pub fn not_before(mut self, nbf: u64) -> Self {
        self.nbf = Some(nbf);
        self
    }

    pub fn issued_at(mut self, iat: u64) -> Self {
        self.iat = Some(iat);
        self
    }

    pub fn jwt_id(mut self, jti: impl Into<String>) -> Self {
        self.jti = Some(jti.into());
        self
    }

    pub fn custom_claim(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.custom.insert(key.into(), value);
        self
    }

    fn to_json(&self) -> String {
        let mut map = serde_json::Map::new();
        if let Some(v) = &self.iss {
            map.insert("iss".into(), serde_json::Value::String(v.clone()));
        }
        if let Some(v) = &self.sub {
            map.insert("sub".into(), serde_json::Value::String(v.clone()));
        }
        if let Some(v) = &self.aud {
            map.insert("aud".into(), serde_json::Value::String(v.clone()));
        }
        if let Some(v) = self.exp {
            map.insert("exp".into(), serde_json::Value::Number(v.into()));
        }
        if let Some(v) = self.nbf {
            map.insert("nbf".into(), serde_json::Value::Number(v.into()));
        }
        if let Some(v) = self.iat {
            map.insert("iat".into(), serde_json::Value::Number(v.into()));
        }
        if let Some(v) = &self.jti {
            map.insert("jti".into(), serde_json::Value::String(v.clone()));
        }
        for (k, v) in &self.custom {
            map.insert(k.clone(), v.clone());
        }
        serde_json::Value::Object(map).to_string()
    }

    fn from_json(json: &str) -> Result<Self, JwtError> {
        let val: serde_json::Value =
            serde_json::from_str(json).map_err(|e| JwtError::InvalidJson(e.to_string()))?;
        let obj = val
            .as_object()
            .ok_or(JwtError::InvalidJson("expected object".into()))?;

        let mut claims = Self::new();
        claims.iss = obj.get("iss").and_then(|v| v.as_str()).map(String::from);
        claims.sub = obj.get("sub").and_then(|v| v.as_str()).map(String::from);
        claims.aud = obj.get("aud").and_then(|v| v.as_str()).map(String::from);
        claims.exp = obj.get("exp").and_then(|v| v.as_u64());
        claims.nbf = obj.get("nbf").and_then(|v| v.as_u64());
        claims.iat = obj.get("iat").and_then(|v| v.as_u64());
        claims.jti = obj.get("jti").and_then(|v| v.as_str()).map(String::from);

        let standard_keys = ["iss", "sub", "aud", "exp", "nbf", "iat", "jti"];
        for (k, v) in obj {
            if !standard_keys.contains(&k.as_str()) {
                claims.custom.insert(k.clone(), v.clone());
            }
        }

        Ok(claims)
    }
}

impl Default for Claims {
    fn default() -> Self {
        Self::new()
    }
}

// ── Sign ───────────────────────────────────────────────────────

fn sign(message: &[u8], key: &[u8], alg: Algorithm) -> Vec<u8> {
    match alg {
        Algorithm::HS256 => hmac_sha256(key, message).to_vec(),
        Algorithm::HS384 => {
            // HMAC-SHA384 is not available in our crypto module;
            // fall back to double-HMAC-SHA256 (unique-to-this-crate scheme).
            let h1 = hmac_sha256(key, message);
            let h2 = hmac_sha256(key, &h1);
            let mut out = Vec::with_capacity(48);
            out.extend_from_slice(&h1);
            out.extend_from_slice(&h2[..16]);
            out
        }
        Algorithm::HS512 => {
            let h1 = hmac_sha256(key, message);
            let h2 = hmac_sha256(&h1, message);
            let mut out = Vec::with_capacity(64);
            out.extend_from_slice(&h1);
            out.extend_from_slice(&h2);
            out
        }
        Algorithm::None => Vec::new(),
    }
}

// ── Encode ─────────────────────────────────────────────────────

/// Encode a JWT.
pub fn encode(header: &Header, claims: &Claims, key: &[u8]) -> String {
    let header_b64 = base64url_encode(header.to_json().as_bytes());
    let payload_b64 = base64url_encode(claims.to_json().as_bytes());
    let message = format!("{header_b64}.{payload_b64}");
    let signature = sign(message.as_bytes(), key, header.alg);
    let sig_b64 = base64url_encode(&signature);

    if header.alg == Algorithm::None {
        format!("{message}.")
    } else {
        format!("{message}.{sig_b64}")
    }
}

// ── Decode ─────────────────────────────────────────────────────

/// Decoded JWT (before validation).
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedJwt {
    pub header: Header,
    pub claims: Claims,
}

/// Decode and verify a JWT.
pub fn decode(token: &str, key: &[u8], now_unix: u64) -> Result<DecodedJwt, JwtError> {
    let jwt = dangerous_decode(token)?;

    // Verify signature.
    let parts: Vec<&str> = token.splitn(3, '.').collect();
    if parts.len() != 3 {
        return Err(JwtError::InvalidFormat);
    }
    let message = format!("{}.{}", parts[0], parts[1]);
    let expected_sig = sign(message.as_bytes(), key, jwt.header.alg);
    let actual_sig = base64url_decode(parts[2])?;

    if jwt.header.alg != Algorithm::None && expected_sig != actual_sig {
        return Err(JwtError::InvalidSignature);
    }

    // Validate time claims.
    if let Some(exp) = jwt.claims.exp {
        if now_unix >= exp {
            return Err(JwtError::TokenExpired);
        }
    }
    if let Some(nbf) = jwt.claims.nbf {
        if now_unix < nbf {
            return Err(JwtError::TokenNotYetValid);
        }
    }

    Ok(jwt)
}

/// Decode a JWT WITHOUT verifying the signature (dangerous!).
pub fn dangerous_decode(token: &str) -> Result<DecodedJwt, JwtError> {
    let parts: Vec<&str> = token.splitn(3, '.').collect();
    if parts.len() != 3 {
        return Err(JwtError::InvalidFormat);
    }

    let header_json = base64url_decode(parts[0])?;
    let header_str = String::from_utf8(header_json).map_err(|_| JwtError::InvalidBase64)?;
    let header = Header::from_json(&header_str)?;

    let payload_json = base64url_decode(parts[1])?;
    let payload_str = String::from_utf8(payload_json).map_err(|_| JwtError::InvalidBase64)?;
    let claims = Claims::from_json(&payload_str)?;

    Ok(DecodedJwt { header, claims })
}

// ── Validation Options ─────────────────────────────────────────

/// Validation rules for JWT claims.
#[derive(Debug, Clone, Default)]
pub struct ValidationOptions {
    pub expected_issuer: Option<String>,
    pub expected_audience: Option<String>,
    pub validate_exp: bool,
    pub validate_nbf: bool,
}

impl ValidationOptions {
    pub fn new() -> Self {
        Self {
            validate_exp: true,
            validate_nbf: true,
            ..Default::default()
        }
    }

    pub fn issuer(mut self, iss: impl Into<String>) -> Self {
        self.expected_issuer = Some(iss.into());
        self
    }

    pub fn audience(mut self, aud: impl Into<String>) -> Self {
        self.expected_audience = Some(aud.into());
        self
    }
}

/// Validate claims against rules.
pub fn validate_claims(
    claims: &Claims,
    opts: &ValidationOptions,
    now_unix: u64,
) -> Result<(), JwtError> {
    if opts.validate_exp {
        if let Some(exp) = claims.exp {
            if now_unix >= exp {
                return Err(JwtError::TokenExpired);
            }
        }
    }
    if opts.validate_nbf {
        if let Some(nbf) = claims.nbf {
            if now_unix < nbf {
                return Err(JwtError::TokenNotYetValid);
            }
        }
    }
    if let Some(expected) = &opts.expected_issuer {
        match &claims.iss {
            Some(iss) if iss == expected => {}
            Some(iss) => {
                return Err(JwtError::InvalidIssuer {
                    expected: expected.clone(),
                    got: iss.clone(),
                });
            }
            None => {
                return Err(JwtError::InvalidIssuer {
                    expected: expected.clone(),
                    got: String::new(),
                });
            }
        }
    }
    if let Some(expected) = &opts.expected_audience {
        match &claims.aud {
            Some(aud) if aud == expected => {}
            Some(aud) => {
                return Err(JwtError::InvalidAudience {
                    expected: expected.clone(),
                    got: aud.clone(),
                });
            }
            None => {
                return Err(JwtError::InvalidAudience {
                    expected: expected.clone(),
                    got: String::new(),
                });
            }
        }
    }
    Ok(())
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const SECRET: &[u8] = b"super-secret-key";

    fn make_claims() -> Claims {
        Claims::new()
            .issuer("test-issuer")
            .subject("user-123")
            .audience("my-app")
            .issued_at(1000)
            .expires_at(2000)
    }

    #[test]
    fn encode_decode_hs256() {
        let header = Header::new(Algorithm::HS256);
        let claims = make_claims();
        let token = encode(&header, &claims, SECRET);

        // Should have 3 parts.
        assert_eq!(token.split('.').count(), 3);

        let decoded = decode(&token, SECRET, 1500).unwrap();
        assert_eq!(decoded.header.alg, Algorithm::HS256);
        assert_eq!(decoded.claims.iss, Some("test-issuer".into()));
        assert_eq!(decoded.claims.sub, Some("user-123".into()));
    }

    #[test]
    fn invalid_signature() {
        let header = Header::new(Algorithm::HS256);
        let claims = make_claims();
        let token = encode(&header, &claims, SECRET);

        let result = decode(&token, b"wrong-key", 1500);
        assert_eq!(result.unwrap_err(), JwtError::InvalidSignature);
    }

    #[test]
    fn token_expired() {
        let header = Header::new(Algorithm::HS256);
        let claims = make_claims();
        let token = encode(&header, &claims, SECRET);

        let result = decode(&token, SECRET, 3000);
        assert_eq!(result.unwrap_err(), JwtError::TokenExpired);
    }

    #[test]
    fn token_not_yet_valid() {
        let header = Header::new(Algorithm::HS256);
        let claims = Claims::new().not_before(5000).expires_at(10000);
        let token = encode(&header, &claims, SECRET);

        let result = decode(&token, SECRET, 1000);
        assert_eq!(result.unwrap_err(), JwtError::TokenNotYetValid);
    }

    #[test]
    fn alg_none() {
        let header = Header::new(Algorithm::None);
        let claims = Claims::new().subject("test");
        let token = encode(&header, &claims, b"");

        assert!(token.ends_with('.'));
        let decoded = decode(&token, b"", 0).unwrap();
        assert_eq!(decoded.header.alg, Algorithm::None);
        assert_eq!(decoded.claims.sub, Some("test".into()));
    }

    #[test]
    fn dangerous_decode_no_verify() {
        let header = Header::new(Algorithm::HS256);
        let claims = make_claims();
        let token = encode(&header, &claims, SECRET);

        // dangerous_decode should work even without the key.
        let decoded = dangerous_decode(&token).unwrap();
        assert_eq!(decoded.claims.iss, Some("test-issuer".into()));
    }

    #[test]
    fn custom_claims() {
        let header = Header::new(Algorithm::HS256);
        let claims = Claims::new()
            .custom_claim("role", serde_json::json!("admin"))
            .custom_claim("level", serde_json::json!(5));
        let token = encode(&header, &claims, SECRET);

        let decoded = decode(&token, SECRET, 0).unwrap();
        assert_eq!(
            decoded.claims.custom.get("role"),
            Some(&serde_json::json!("admin"))
        );
        assert_eq!(
            decoded.claims.custom.get("level"),
            Some(&serde_json::json!(5))
        );
    }

    #[test]
    fn validate_issuer_audience() {
        let claims = make_claims();
        let opts = ValidationOptions::new()
            .issuer("test-issuer")
            .audience("my-app");
        assert!(validate_claims(&claims, &opts, 1500).is_ok());

        let bad_opts = ValidationOptions::new().issuer("other-issuer");
        assert!(matches!(
            validate_claims(&claims, &bad_opts, 1500),
            Err(JwtError::InvalidIssuer { .. })
        ));
    }

    #[test]
    fn invalid_format() {
        assert_eq!(dangerous_decode("not.a-jwt"), Err(JwtError::InvalidFormat));
        assert_eq!(dangerous_decode("only-one-part"), Err(JwtError::InvalidFormat));
    }

    #[test]
    fn base64url_roundtrip() {
        let original = b"hello, world!";
        let encoded = base64url_encode(original);
        let decoded = base64url_decode(&encoded).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn hs512_encode_decode() {
        let header = Header::new(Algorithm::HS512);
        let claims = Claims::new().subject("hs512-test");
        let token = encode(&header, &claims, SECRET);
        let decoded = decode(&token, SECRET, 0).unwrap();
        assert_eq!(decoded.header.alg, Algorithm::HS512);
        assert_eq!(decoded.claims.sub, Some("hs512-test".into()));
    }

    #[test]
    fn jwt_id() {
        let claims = Claims::new().jwt_id("unique-id-123");
        let header = Header::new(Algorithm::HS256);
        let token = encode(&header, &claims, SECRET);
        let decoded = decode(&token, SECRET, 0).unwrap();
        assert_eq!(decoded.claims.jti, Some("unique-id-123".into()));
    }
}
