//! WebAuthn ceremony model — registration and authentication flows.
//!
//! Replaces `@simplewebauthn/server` with a pure Rust WebAuthn model.
//! Implements credential creation/assertion options, attestation response
//! structures, challenge validation, origin checks, and credential storage.

use std::collections::HashMap;
use std::fmt;

// ── Errors ─────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WebAuthnError {
    ChallengeMismatch,
    OriginMismatch { expected: String, got: String },
    InvalidClientData(String),
    InvalidAttestation(String),
    CredentialNotFound,
    SignatureInvalid,
    InvalidCbor(String),
}

impl fmt::Display for WebAuthnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ChallengeMismatch => write!(f, "challenge mismatch"),
            Self::OriginMismatch { expected, got } => {
                write!(f, "origin mismatch: expected {expected}, got {got}")
            }
            Self::InvalidClientData(msg) => write!(f, "invalid client data: {msg}"),
            Self::InvalidAttestation(msg) => write!(f, "invalid attestation: {msg}"),
            Self::CredentialNotFound => write!(f, "credential not found"),
            Self::SignatureInvalid => write!(f, "signature invalid"),
            Self::InvalidCbor(msg) => write!(f, "invalid CBOR: {msg}"),
        }
    }
}

impl std::error::Error for WebAuthnError {}

// ── Types ──────────────────────────────────────────────────────

/// Relying Party information.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelyingParty {
    pub id: String,
    pub name: String,
}

/// User information for registration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WebAuthnUser {
    pub id: Vec<u8>,
    pub name: String,
    pub display_name: String,
}

/// Public key credential algorithm parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PubKeyCredParam {
    /// "public-key" type.
    pub alg: i32,
}

impl PubKeyCredParam {
    /// ES256 (ECDSA w/ SHA-256).
    pub const ES256: Self = Self { alg: -7 };
    /// RS256 (RSASSA-PKCS1-v1_5 w/ SHA-256).
    pub const RS256: Self = Self { alg: -257 };
    /// EdDSA.
    pub const EDDSA: Self = Self { alg: -8 };
}

/// Attestation conveyance preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttestationConveyance {
    None,
    Indirect,
    Direct,
    Enterprise,
}

impl fmt::Display for AttestationConveyance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::Indirect => write!(f, "indirect"),
            Self::Direct => write!(f, "direct"),
            Self::Enterprise => write!(f, "enterprise"),
        }
    }
}

/// User verification requirement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UserVerification {
    Required,
    Preferred,
    Discouraged,
}

// ── Registration (Creation) ────────────────────────────────────

/// Options for `navigator.credentials.create()`.
#[derive(Debug, Clone)]
pub struct PublicKeyCredentialCreationOptions {
    pub rp: RelyingParty,
    pub user: WebAuthnUser,
    pub challenge: Vec<u8>,
    pub pub_key_cred_params: Vec<PubKeyCredParam>,
    pub timeout_ms: Option<u64>,
    pub attestation: AttestationConveyance,
    pub exclude_credentials: Vec<Vec<u8>>,
}

impl PublicKeyCredentialCreationOptions {
    pub fn new(rp: RelyingParty, user: WebAuthnUser, challenge: Vec<u8>) -> Self {
        Self {
            rp,
            user,
            challenge,
            pub_key_cred_params: vec![PubKeyCredParam::ES256, PubKeyCredParam::RS256],
            timeout_ms: Some(60_000),
            attestation: AttestationConveyance::None,
            exclude_credentials: Vec::new(),
        }
    }

    pub fn timeout(mut self, ms: u64) -> Self {
        self.timeout_ms = Some(ms);
        self
    }

    pub fn attestation(mut self, conveyance: AttestationConveyance) -> Self {
        self.attestation = conveyance;
        self
    }
}

/// Client data collected during a ceremony.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectedClientData {
    pub challenge_b64: String,
    pub origin: String,
    pub cross_origin: bool,
    pub ceremony_type: CeremonyType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CeremonyType {
    Create,
    Get,
}

impl fmt::Display for CeremonyType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Create => write!(f, "webauthn.create"),
            Self::Get => write!(f, "webauthn.get"),
        }
    }
}

/// Authenticator attestation response (from registration).
#[derive(Debug, Clone)]
pub struct AuthenticatorAttestationResponse {
    pub client_data_json: Vec<u8>,
    pub attestation_object: Vec<u8>,
}

/// Parsed attestation object (simplified CBOR concept).
#[derive(Debug, Clone)]
pub struct AttestationObject {
    pub fmt: String,
    pub auth_data: AuthenticatorData,
    pub att_stmt: Vec<u8>,
}

/// Authenticator data.
#[derive(Debug, Clone)]
pub struct AuthenticatorData {
    pub rp_id_hash: [u8; 32],
    pub flags: u8,
    pub sign_count: u32,
    pub credential_id: Option<Vec<u8>>,
    pub credential_public_key: Option<Vec<u8>>,
}

impl AuthenticatorData {
    /// User Present flag (bit 0).
    pub fn user_present(&self) -> bool {
        self.flags & 0x01 != 0
    }

    /// User Verified flag (bit 2).
    pub fn user_verified(&self) -> bool {
        self.flags & 0x04 != 0
    }

    /// Attested credential data present (bit 6).
    pub fn has_attested_credential(&self) -> bool {
        self.flags & 0x40 != 0
    }
}

// ── Authentication (Assertion) ─────────────────────────────────

/// Options for `navigator.credentials.get()`.
#[derive(Debug, Clone)]
pub struct PublicKeyCredentialRequestOptions {
    pub challenge: Vec<u8>,
    pub timeout_ms: Option<u64>,
    pub rp_id: String,
    pub allow_credentials: Vec<Vec<u8>>,
    pub user_verification: UserVerification,
}

impl PublicKeyCredentialRequestOptions {
    pub fn new(challenge: Vec<u8>, rp_id: impl Into<String>) -> Self {
        Self {
            challenge,
            timeout_ms: Some(60_000),
            rp_id: rp_id.into(),
            allow_credentials: Vec::new(),
            user_verification: UserVerification::Preferred,
        }
    }
}

/// Authenticator assertion response (from authentication).
#[derive(Debug, Clone)]
pub struct AuthenticatorAssertionResponse {
    pub client_data_json: Vec<u8>,
    pub authenticator_data: Vec<u8>,
    pub signature: Vec<u8>,
    pub user_handle: Option<Vec<u8>>,
}

// ── Registration Validation ────────────────────────────────────

/// Validate client data during a ceremony.
pub fn validate_client_data(
    client_data: &CollectedClientData,
    expected_challenge_b64: &str,
    expected_origin: &str,
    expected_type: CeremonyType,
) -> Result<(), WebAuthnError> {
    if client_data.ceremony_type != expected_type {
        return Err(WebAuthnError::InvalidClientData(format!(
            "expected type {expected_type}, got {}",
            client_data.ceremony_type
        )));
    }
    if client_data.challenge_b64 != expected_challenge_b64 {
        return Err(WebAuthnError::ChallengeMismatch);
    }
    if client_data.origin != expected_origin {
        return Err(WebAuthnError::OriginMismatch {
            expected: expected_origin.to_string(),
            got: client_data.origin.clone(),
        });
    }
    Ok(())
}

/// Validate that auth data has required flags.
pub fn validate_auth_data_flags(
    auth_data: &AuthenticatorData,
    require_user_verification: bool,
) -> Result<(), WebAuthnError> {
    if !auth_data.user_present() {
        return Err(WebAuthnError::InvalidAttestation(
            "user not present".into(),
        ));
    }
    if require_user_verification && !auth_data.user_verified() {
        return Err(WebAuthnError::InvalidAttestation(
            "user not verified".into(),
        ));
    }
    Ok(())
}

// ── Credential Storage ─────────────────────────────────────────

/// A stored credential.
#[derive(Debug, Clone)]
pub struct StoredCredential {
    pub credential_id: Vec<u8>,
    pub public_key: Vec<u8>,
    pub sign_count: u32,
    pub user_id: Vec<u8>,
    pub user_name: String,
}

/// In-memory credential store.
#[derive(Debug, Clone, Default)]
pub struct CredentialStore {
    credentials: HashMap<Vec<u8>, StoredCredential>,
}

impl CredentialStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn store(&mut self, credential: StoredCredential) {
        self.credentials
            .insert(credential.credential_id.clone(), credential);
    }

    pub fn lookup(&self, credential_id: &[u8]) -> Option<&StoredCredential> {
        self.credentials.get(credential_id)
    }

    pub fn lookup_mut(&mut self, credential_id: &[u8]) -> Option<&mut StoredCredential> {
        self.credentials.get_mut(credential_id)
    }

    /// Get all credentials for a user ID.
    pub fn credentials_for_user(&self, user_id: &[u8]) -> Vec<&StoredCredential> {
        self.credentials
            .values()
            .filter(|c| c.user_id == user_id)
            .collect()
    }

    pub fn remove(&mut self, credential_id: &[u8]) -> Option<StoredCredential> {
        self.credentials.remove(credential_id)
    }

    pub fn len(&self) -> usize {
        self.credentials.len()
    }

    pub fn is_empty(&self) -> bool {
        self.credentials.is_empty()
    }

    /// Update sign count, checking for cloned authenticator (replay).
    pub fn update_sign_count(
        &mut self,
        credential_id: &[u8],
        new_count: u32,
    ) -> Result<(), WebAuthnError> {
        let cred = self
            .credentials
            .get_mut(credential_id)
            .ok_or(WebAuthnError::CredentialNotFound)?;
        if new_count > 0 && new_count <= cred.sign_count {
            return Err(WebAuthnError::SignatureInvalid);
        }
        cred.sign_count = new_count;
        Ok(())
    }
}

// ── Base64url helpers ──────────────────────────────────────────

const B64URL_CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

/// Base64url encode (no padding).
pub fn base64url_encode(input: &[u8]) -> String {
    let mut out = String::new();
    let mut i = 0;
    while i < input.len() {
        let b0 = input[i] as u32;
        let b1 = if i + 1 < input.len() { input[i + 1] as u32 } else { 0 };
        let b2 = if i + 2 < input.len() { input[i + 2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        out.push(B64URL_CHARS[((triple >> 18) & 0x3F) as usize] as char);
        out.push(B64URL_CHARS[((triple >> 12) & 0x3F) as usize] as char);
        if i + 1 < input.len() {
            out.push(B64URL_CHARS[((triple >> 6) & 0x3F) as usize] as char);
        }
        if i + 2 < input.len() {
            out.push(B64URL_CHARS[(triple & 0x3F) as usize] as char);
        }
        i += 3;
    }
    out
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rp() -> RelyingParty {
        RelyingParty {
            id: "example.com".into(),
            name: "Example".into(),
        }
    }

    fn make_user() -> WebAuthnUser {
        WebAuthnUser {
            id: vec![1, 2, 3],
            name: "alice@example.com".into(),
            display_name: "Alice".into(),
        }
    }

    #[test]
    fn creation_options_defaults() {
        let opts = PublicKeyCredentialCreationOptions::new(
            make_rp(),
            make_user(),
            vec![0xAA, 0xBB],
        );
        assert_eq!(opts.challenge, vec![0xAA, 0xBB]);
        assert_eq!(opts.pub_key_cred_params.len(), 2);
        assert_eq!(opts.timeout_ms, Some(60_000));
        assert_eq!(opts.attestation, AttestationConveyance::None);
    }

    #[test]
    fn creation_options_custom() {
        let opts = PublicKeyCredentialCreationOptions::new(
            make_rp(),
            make_user(),
            vec![0xCC],
        )
        .timeout(120_000)
        .attestation(AttestationConveyance::Direct);
        assert_eq!(opts.timeout_ms, Some(120_000));
        assert_eq!(opts.attestation, AttestationConveyance::Direct);
    }

    #[test]
    fn client_data_validation_pass() {
        let cd = CollectedClientData {
            challenge_b64: "dGVzdA".into(),
            origin: "https://example.com".into(),
            cross_origin: false,
            ceremony_type: CeremonyType::Create,
        };
        let result = validate_client_data(
            &cd,
            "dGVzdA",
            "https://example.com",
            CeremonyType::Create,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn client_data_challenge_mismatch() {
        let cd = CollectedClientData {
            challenge_b64: "wrong".into(),
            origin: "https://example.com".into(),
            cross_origin: false,
            ceremony_type: CeremonyType::Create,
        };
        let result = validate_client_data(
            &cd,
            "expected",
            "https://example.com",
            CeremonyType::Create,
        );
        assert_eq!(result, Err(WebAuthnError::ChallengeMismatch));
    }

    #[test]
    fn client_data_origin_mismatch() {
        let cd = CollectedClientData {
            challenge_b64: "ok".into(),
            origin: "https://evil.com".into(),
            cross_origin: false,
            ceremony_type: CeremonyType::Create,
        };
        let result = validate_client_data(
            &cd,
            "ok",
            "https://example.com",
            CeremonyType::Create,
        );
        assert!(matches!(result, Err(WebAuthnError::OriginMismatch { .. })));
    }

    #[test]
    fn auth_data_flags() {
        let ad = AuthenticatorData {
            rp_id_hash: [0; 32],
            flags: 0x05, // user present + user verified
            sign_count: 1,
            credential_id: None,
            credential_public_key: None,
        };
        assert!(ad.user_present());
        assert!(ad.user_verified());
        assert!(!ad.has_attested_credential());
        assert!(validate_auth_data_flags(&ad, true).is_ok());
    }

    #[test]
    fn auth_data_no_user_present() {
        let ad = AuthenticatorData {
            rp_id_hash: [0; 32],
            flags: 0x00,
            sign_count: 0,
            credential_id: None,
            credential_public_key: None,
        };
        assert!(validate_auth_data_flags(&ad, false).is_err());
    }

    #[test]
    fn credential_store_basic() {
        let mut store = CredentialStore::new();
        assert!(store.is_empty());

        store.store(StoredCredential {
            credential_id: vec![1, 2, 3],
            public_key: vec![10, 20],
            sign_count: 0,
            user_id: vec![100],
            user_name: "alice".into(),
        });

        assert_eq!(store.len(), 1);
        assert!(store.lookup(&[1, 2, 3]).is_some());
        assert!(store.lookup(&[9, 9, 9]).is_none());
    }

    #[test]
    fn credential_store_user_lookup() {
        let mut store = CredentialStore::new();
        let user_id = vec![42u8];
        store.store(StoredCredential {
            credential_id: vec![1],
            public_key: vec![],
            sign_count: 0,
            user_id: user_id.clone(),
            user_name: "alice".into(),
        });
        store.store(StoredCredential {
            credential_id: vec![2],
            public_key: vec![],
            sign_count: 0,
            user_id: user_id.clone(),
            user_name: "alice".into(),
        });
        store.store(StoredCredential {
            credential_id: vec![3],
            public_key: vec![],
            sign_count: 0,
            user_id: vec![99],
            user_name: "bob".into(),
        });
        assert_eq!(store.credentials_for_user(&user_id).len(), 2);
    }

    #[test]
    fn sign_count_replay_detection() {
        let mut store = CredentialStore::new();
        store.store(StoredCredential {
            credential_id: vec![1],
            public_key: vec![],
            sign_count: 5,
            user_id: vec![],
            user_name: String::new(),
        });
        // Valid increment.
        assert!(store.update_sign_count(&[1], 6).is_ok());
        // Replay (same count).
        assert!(store.update_sign_count(&[1], 6).is_err());
        // Replay (lower count).
        assert!(store.update_sign_count(&[1], 3).is_err());
    }

    #[test]
    fn base64url_encode_vectors() {
        assert_eq!(base64url_encode(b""), "");
        assert_eq!(base64url_encode(b"f"), "Zg");
        assert_eq!(base64url_encode(b"fo"), "Zm8");
        assert_eq!(base64url_encode(b"foo"), "Zm9v");
        assert_eq!(base64url_encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn request_options() {
        let opts = PublicKeyCredentialRequestOptions::new(
            vec![0xDE, 0xAD],
            "example.com",
        );
        assert_eq!(opts.rp_id, "example.com");
        assert_eq!(opts.challenge, vec![0xDE, 0xAD]);
        assert_eq!(opts.user_verification, UserVerification::Preferred);
    }
}
