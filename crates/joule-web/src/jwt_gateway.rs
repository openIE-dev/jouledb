// jwt_gateway.rs — JWT gateway: base64url decode, header/payload parsing,
// claims extraction, audience/issuer/expiry validation, key registry
// with rotation, and claims-based routing rules.

use std::collections::HashMap;

const B64URL: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

fn b64url_encode(data: &[u8]) -> String {
    let mut out = String::with_capacity((data.len() * 4 + 2) / 3);
    for chunk in data.chunks(3) {
        let (b0, b1, b2) = (chunk[0] as u32,
            if chunk.len() > 1 { chunk[1] as u32 } else { 0 },
            if chunk.len() > 2 { chunk[2] as u32 } else { 0 });
        let t = (b0 << 16) | (b1 << 8) | b2;
        out.push(B64URL[((t >> 18) & 0x3F) as usize] as char);
        out.push(B64URL[((t >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 { out.push(B64URL[((t >> 6) & 0x3F) as usize] as char); }
        if chunk.len() > 2 { out.push(B64URL[(t & 0x3F) as usize] as char); }
    }
    out
}

fn b64url_decode_char(c: u8) -> Option<u32> {
    match c {
        b'A'..=b'Z' => Some((c - b'A') as u32),
        b'a'..=b'z' => Some((c - b'a' + 26) as u32),
        b'0'..=b'9' => Some((c - b'0' + 52) as u32),
        b'-' => Some(62), b'_' => Some(63), _ => None,
    }
}

pub fn b64url_decode(input: &str) -> Result<Vec<u8>, JwtError> {
    let mut out = Vec::with_capacity(input.len() * 3 / 4);
    for chunk in input.as_bytes().chunks(4) {
        let v: Vec<u32> = chunk.iter()
            .map(|c| b64url_decode_char(*c).ok_or(JwtError::InvalidBase64))
            .collect::<Result<_, _>>()?;
        if v.len() >= 2 { out.push(((v[0] << 2) | (v[1] >> 4)) as u8); }
        if v.len() >= 3 { out.push((((v[1] & 0x0F) << 4) | (v[2] >> 2)) as u8); }
        if v.len() >= 4 { out.push((((v[2] & 0x03) << 6) | v[3]) as u8); }
    }
    Ok(out)
}

#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    Null, Bool(bool), Number(f64), Str(String),
    Array(Vec<JsonValue>), Object(Vec<(String, JsonValue)>),
}

impl JsonValue {
    pub fn as_str(&self) -> Option<&str> { match self { Self::Str(s) => Some(s), _ => None } }
    pub fn as_f64(&self) -> Option<f64> { match self { Self::Number(n) => Some(*n), _ => None } }
    pub fn as_u64(&self) -> Option<u64> { self.as_f64().map(|n| n as u64) }
    pub fn as_bool(&self) -> Option<bool> { match self { Self::Bool(b) => Some(*b), _ => None } }
    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        match self { Self::Object(p) => p.iter().find(|(k, _)| k == key).map(|(_, v)| v), _ => None }
    }
    pub fn as_array(&self) -> Option<&[JsonValue]> { match self { Self::Array(a) => Some(a), _ => None } }
}

struct JParser<'a> { inp: &'a [u8], pos: usize }

impl<'a> JParser<'a> {
    fn new(s: &'a str) -> Self { Self { inp: s.as_bytes(), pos: 0 } }
    fn ws(&mut self) { while self.pos < self.inp.len() && matches!(self.inp[self.pos], b' '|b'\t'|b'\n'|b'\r') { self.pos += 1; } }
    fn peek(&self) -> Option<u8> { self.inp.get(self.pos).copied() }
    fn next(&mut self) -> Option<u8> { let c = self.inp.get(self.pos).copied()?; self.pos += 1; Some(c) }
    fn expect(&mut self, c: u8) -> Result<(), JwtError> { self.ws(); if self.next() == Some(c) { Ok(()) } else { Err(JwtError::MalformedJson) } }

    fn value(&mut self) -> Result<JsonValue, JwtError> {
        self.ws();
        match self.peek() {
            Some(b'"') => self.string().map(JsonValue::Str),
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b't') | Some(b'f') => self.boolean(),
            Some(b'n') => self.null(),
            Some(c) if c == b'-' || c.is_ascii_digit() => self.number(),
            _ => Err(JwtError::MalformedJson),
        }
    }

    fn string(&mut self) -> Result<String, JwtError> {
        self.expect(b'"')?;
        let mut s = String::new();
        loop {
            match self.next() {
                None => return Err(JwtError::MalformedJson),
                Some(b'"') => return Ok(s),
                Some(b'\\') => match self.next() {
                    Some(b'"') => s.push('"'), Some(b'\\') => s.push('\\'),
                    Some(b'/') => s.push('/'), Some(b'n') => s.push('\n'),
                    Some(b't') => s.push('\t'), Some(b'r') => s.push('\r'),
                    _ => return Err(JwtError::MalformedJson),
                },
                Some(c) => s.push(c as char),
            }
        }
    }

    fn number(&mut self) -> Result<JsonValue, JwtError> {
        let start = self.pos;
        if self.peek() == Some(b'-') { self.pos += 1; }
        while self.pos < self.inp.len() && self.inp[self.pos].is_ascii_digit() { self.pos += 1; }
        if self.pos < self.inp.len() && self.inp[self.pos] == b'.' {
            self.pos += 1;
            while self.pos < self.inp.len() && self.inp[self.pos].is_ascii_digit() { self.pos += 1; }
        }
        let s = std::str::from_utf8(&self.inp[start..self.pos]).map_err(|_| JwtError::MalformedJson)?;
        Ok(JsonValue::Number(s.parse().map_err(|_| JwtError::MalformedJson)?))
    }

    fn boolean(&mut self) -> Result<JsonValue, JwtError> {
        if self.inp[self.pos..].starts_with(b"true") { self.pos += 4; Ok(JsonValue::Bool(true)) }
        else if self.inp[self.pos..].starts_with(b"false") { self.pos += 5; Ok(JsonValue::Bool(false)) }
        else { Err(JwtError::MalformedJson) }
    }

    fn null(&mut self) -> Result<JsonValue, JwtError> {
        if self.inp[self.pos..].starts_with(b"null") { self.pos += 4; Ok(JsonValue::Null) }
        else { Err(JwtError::MalformedJson) }
    }

    fn object(&mut self) -> Result<JsonValue, JwtError> {
        self.expect(b'{')?;
        let mut pairs = Vec::new();
        self.ws();
        if self.peek() == Some(b'}') { self.pos += 1; return Ok(JsonValue::Object(pairs)); }
        loop {
            self.ws(); let key = self.string()?; self.expect(b':')?;
            pairs.push((key, self.value()?));
            self.ws();
            match self.peek() {
                Some(b',') => { self.pos += 1; }
                Some(b'}') => { self.pos += 1; return Ok(JsonValue::Object(pairs)); }
                _ => return Err(JwtError::MalformedJson),
            }
        }
    }

    fn array(&mut self) -> Result<JsonValue, JwtError> {
        self.expect(b'[')?;
        let mut items = Vec::new();
        self.ws();
        if self.peek() == Some(b']') { self.pos += 1; return Ok(JsonValue::Array(items)); }
        loop {
            items.push(self.value()?);
            self.ws();
            match self.peek() {
                Some(b',') => { self.pos += 1; }
                Some(b']') => { self.pos += 1; return Ok(JsonValue::Array(items)); }
                _ => return Err(JwtError::MalformedJson),
            }
        }
    }
}

fn parse_json(input: &str) -> Result<JsonValue, JwtError> { JParser::new(input).value() }

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JwtError {
    InvalidBase64, MalformedJson, MalformedToken,
    MissingClaim(String), ExpiredToken, InvalidAudience, InvalidIssuer, UnknownKeyId,
}

impl std::fmt::Display for JwtError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidBase64 => write!(f, "invalid base64url"),
            Self::MalformedJson => write!(f, "malformed JSON"),
            Self::MalformedToken => write!(f, "malformed JWT token"),
            Self::MissingClaim(c) => write!(f, "missing claim: {c}"),
            Self::ExpiredToken => write!(f, "token expired"),
            Self::InvalidAudience => write!(f, "invalid audience"),
            Self::InvalidIssuer => write!(f, "invalid issuer"),
            Self::UnknownKeyId => write!(f, "unknown key ID"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct JwtHeader {
    pub alg: String,
    pub typ: Option<String>,
    pub kid: Option<String>,
    pub raw: JsonValue,
}

#[derive(Debug, Clone, PartialEq)]
pub struct JwtClaims {
    pub raw: JsonValue,
}

impl JwtClaims {
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.raw.get(key)?.as_str()
    }

    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.raw.get(key)?.as_u64()
    }

    pub fn subject(&self) -> Option<&str> {
        self.get_str("sub")
    }

    pub fn issuer(&self) -> Option<&str> {
        self.get_str("iss")
    }

    pub fn audience(&self) -> Option<&str> {
        self.get_str("aud")
    }

    pub fn expiry(&self) -> Option<u64> {
        self.get_u64("exp")
    }

    pub fn issued_at(&self) -> Option<u64> {
        self.get_u64("iat")
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParsedJwt {
    pub header: JwtHeader,
    pub claims: JwtClaims,
    pub signature_raw: Vec<u8>,
}

/// Parse a JWT token string (header.payload.signature) WITHOUT verifying the signature.
pub fn parse_jwt(token: &str) -> Result<ParsedJwt, JwtError> {
    let parts: Vec<&str> = token.split('.').collect();
    if parts.len() != 3 {
        return Err(JwtError::MalformedToken);
    }

    let header_bytes = b64url_decode(parts[0])?;
    let header_json =
        std::str::from_utf8(&header_bytes).map_err(|_| JwtError::MalformedJson)?;
    let header_val = parse_json(header_json)?;

    let alg = header_val
        .get("alg")
        .and_then(|v| v.as_str())
        .ok_or(JwtError::MissingClaim("alg".into()))?
        .to_string();
    let typ = header_val.get("typ").and_then(|v| v.as_str()).map(String::from);
    let kid = header_val.get("kid").and_then(|v| v.as_str()).map(String::from);

    let payload_bytes = b64url_decode(parts[1])?;
    let payload_json =
        std::str::from_utf8(&payload_bytes).map_err(|_| JwtError::MalformedJson)?;
    let claims_val = parse_json(payload_json)?;

    let sig = b64url_decode(parts[2])?;

    Ok(ParsedJwt {
        header: JwtHeader {
            alg,
            typ,
            kid,
            raw: header_val,
        },
        claims: JwtClaims { raw: claims_val },
        signature_raw: sig,
    })
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

pub fn validate_expiry(claims: &JwtClaims, now_epoch: u64) -> Result<(), JwtError> {
    match claims.expiry() {
        None => Err(JwtError::MissingClaim("exp".into())),
        Some(exp) if now_epoch >= exp => Err(JwtError::ExpiredToken),
        _ => Ok(()),
    }
}

pub fn validate_issuer(claims: &JwtClaims, expected: &str) -> Result<(), JwtError> {
    match claims.issuer() {
        Some(iss) if iss == expected => Ok(()),
        _ => Err(JwtError::InvalidIssuer),
    }
}

pub fn validate_audience(claims: &JwtClaims, expected: &str) -> Result<(), JwtError> {
    match claims.audience() {
        Some(aud) if aud == expected => Ok(()),
        _ => Err(JwtError::InvalidAudience),
    }
}

// ---------------------------------------------------------------------------
// Key registry with rotation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct KeyEntry {
    pub kid: String,
    pub key_data: Vec<u8>,
    pub active: bool,
}

#[derive(Debug, Clone, Default)]
pub struct KeyRegistry {
    keys: Vec<KeyEntry>,
}

impl KeyRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_key(&mut self, kid: &str, key_data: Vec<u8>) {
        self.keys.push(KeyEntry {
            kid: kid.to_string(),
            key_data,
            active: true,
        });
    }

    pub fn rotate(&mut self, old_kid: &str, new_kid: &str, new_data: Vec<u8>) {
        for k in &mut self.keys {
            if k.kid == old_kid {
                k.active = false;
            }
        }
        self.add_key(new_kid, new_data);
    }

    pub fn find(&self, kid: &str) -> Option<&KeyEntry> {
        self.keys.iter().find(|k| k.kid == kid)
    }

    pub fn find_active(&self, kid: &str) -> Option<&KeyEntry> {
        self.keys.iter().find(|k| k.kid == kid && k.active)
    }

    pub fn active_count(&self) -> usize {
        self.keys.iter().filter(|k| k.active).count()
    }

    pub fn revoke(&mut self, kid: &str) {
        for k in &mut self.keys {
            if k.kid == kid {
                k.active = false;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Claims-based routing
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum RoutingCondition {
    ClaimEquals { claim: String, value: String },
    ClaimExists(String),
    ClaimInList { claim: String, values: Vec<String> },
}

impl RoutingCondition {
    pub fn matches(&self, claims: &JwtClaims) -> bool {
        match self {
            Self::ClaimEquals { claim, value } => {
                claims.get_str(claim).map(|v| v == value).unwrap_or(false)
            }
            Self::ClaimExists(claim) => claims.raw.get(claim).is_some(),
            Self::ClaimInList { claim, values } => claims
                .get_str(claim)
                .map(|v| values.iter().any(|expected| expected == v))
                .unwrap_or(false),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RoutingRule {
    pub condition: RoutingCondition,
    pub target: String,
}

/// Evaluate a list of routing rules against claims. Returns the first matching target.
pub fn route_by_claims(rules: &[RoutingRule], claims: &JwtClaims) -> Option<String> {
    rules
        .iter()
        .find(|r| r.condition.matches(claims))
        .map(|r| r.target.clone())
}

// ---------------------------------------------------------------------------
// Helper: build a test JWT string (unverified signature, for testing)
// ---------------------------------------------------------------------------

fn build_test_jwt(header_json: &str, payload_json: &str) -> String {
    let h = b64url_encode(header_json.as_bytes());
    let p = b64url_encode(payload_json.as_bytes());
    let sig = b64url_encode(b"fake-sig");
    format!("{h}.{p}.{sig}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_b64url_roundtrip() {
        let data = b"Hello, JWT!";
        let encoded = b64url_encode(data);
        let decoded = b64url_decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_b64url_empty() {
        let encoded = b64url_encode(b"");
        assert_eq!(encoded, "");
        let decoded = b64url_decode("").unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_b64url_decode_invalid() {
        assert_eq!(b64url_decode("!!!"), Err(JwtError::InvalidBase64));
    }

    #[test]
    fn test_parse_jwt_basic() {
        let token = build_test_jwt(
            r#"{"alg":"HS256","typ":"JWT"}"#,
            r#"{"sub":"user1","iss":"auth.example.com","aud":"api","exp":9999999999}"#,
        );
        let jwt = parse_jwt(&token).unwrap();
        assert_eq!(jwt.header.alg, "HS256");
        assert_eq!(jwt.header.typ.as_deref(), Some("JWT"));
        assert_eq!(jwt.claims.subject(), Some("user1"));
        assert_eq!(jwt.claims.issuer(), Some("auth.example.com"));
        assert_eq!(jwt.claims.audience(), Some("api"));
        assert_eq!(jwt.claims.expiry(), Some(9999999999));
    }

    #[test]
    fn test_parse_jwt_malformed_parts() {
        assert_eq!(parse_jwt("only.two"), Err(JwtError::MalformedToken));
        assert_eq!(parse_jwt("one"), Err(JwtError::MalformedToken));
    }

    #[test]
    fn test_validate_expiry() {
        let ok = parse_jwt(&build_test_jwt(r#"{"alg":"HS256"}"#, r#"{"exp":2000000000}"#)).unwrap();
        assert!(validate_expiry(&ok.claims, 1_000_000_000).is_ok());
        let exp = parse_jwt(&build_test_jwt(r#"{"alg":"HS256"}"#, r#"{"exp":1000}"#)).unwrap();
        assert_eq!(validate_expiry(&exp.claims, 2000), Err(JwtError::ExpiredToken));
        let miss = parse_jwt(&build_test_jwt(r#"{"alg":"HS256"}"#, r#"{"sub":"x"}"#)).unwrap();
        assert_eq!(validate_expiry(&miss.claims, 1000), Err(JwtError::MissingClaim("exp".into())));
    }

    #[test]
    fn test_validate_issuer_and_audience() {
        let t1 = parse_jwt(&build_test_jwt(r#"{"alg":"HS256"}"#, r#"{"iss":"a.com","aud":"api"}"#)).unwrap();
        assert!(validate_issuer(&t1.claims, "a.com").is_ok());
        assert_eq!(validate_issuer(&t1.claims, "b.com"), Err(JwtError::InvalidIssuer));
        assert!(validate_audience(&t1.claims, "api").is_ok());
        assert_eq!(validate_audience(&t1.claims, "other"), Err(JwtError::InvalidAudience));
    }

    #[test]
    fn test_key_registry_and_rotation() {
        let mut reg = KeyRegistry::new();
        reg.add_key("k1", vec![1, 2, 3]);
        reg.add_key("k2", vec![4, 5, 6]);
        assert_eq!(reg.active_count(), 2);
        assert!(reg.find("k1").is_some());
        reg.rotate("k1", "k3", vec![7]);
        assert_eq!(reg.active_count(), 2); // k2, k3
        assert!(reg.find("k1").is_some());
        assert!(reg.find_active("k1").is_none());
        assert!(reg.find_active("k3").is_some());
        reg.revoke("k2");
        assert_eq!(reg.active_count(), 1);
    }

    #[test]
    fn test_routing_conditions() {
        let admin = parse_jwt(&build_test_jwt(r#"{"alg":"HS256"}"#, r#"{"role":"admin","tier":"gold","debug":true}"#)).unwrap();
        let rules = vec![
            RoutingRule { condition: RoutingCondition::ClaimEquals { claim: "role".into(), value: "admin".into() }, target: "admin-be".into() },
            RoutingRule { condition: RoutingCondition::ClaimEquals { claim: "role".into(), value: "user".into() }, target: "user-be".into() },
        ];
        assert_eq!(route_by_claims(&rules, &admin.claims), Some("admin-be".into()));
        let guest = parse_jwt(&build_test_jwt(r#"{"alg":"HS256"}"#, r#"{"role":"guest"}"#)).unwrap();
        assert_eq!(route_by_claims(&rules, &guest.claims), None);
        assert!(RoutingCondition::ClaimInList { claim: "tier".into(), values: vec!["gold".into(), "platinum".into()] }.matches(&admin.claims));
        assert!(RoutingCondition::ClaimExists("debug".into()).matches(&admin.claims));
        assert!(!RoutingCondition::ClaimExists("nope".into()).matches(&admin.claims));
    }

    #[test]
    fn test_jwt_with_kid() {
        let jwt = parse_jwt(&build_test_jwt(r#"{"alg":"RS256","kid":"key-2024"}"#, r#"{"sub":"u"}"#)).unwrap();
        assert_eq!(jwt.header.kid.as_deref(), Some("key-2024"));
    }

    #[test]
    fn test_json_parser_features() {
        let nested = parse_json(r#"{"a":{"b":42}}"#).unwrap();
        assert_eq!(nested.get("a").unwrap().get("b").unwrap().as_f64(), Some(42.0));
        let arr = parse_json(r#"{"items":[1,2,3]}"#).unwrap();
        assert_eq!(arr.get("items").unwrap().as_array().unwrap().len(), 3);
        let mixed = parse_json(r#"{"a":null,"b":true,"c":false}"#).unwrap();
        assert_eq!(mixed.get("a"), Some(&JsonValue::Null));
        assert_eq!(mixed.get("b").unwrap().as_bool(), Some(true));
    }

    #[test]
    fn test_jwt_error_display() {
        assert_eq!(JwtError::ExpiredToken.to_string(), "token expired");
        assert_eq!(JwtError::MissingClaim("exp".into()).to_string(), "missing claim: exp");
    }
}
