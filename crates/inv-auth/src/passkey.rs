//! Passwordless authentication: WebAuthn/FIDO2 passkeys, magic links, and email OTP.
//!
//! This module implements the 2026 passwordless OAuth model:
//! - Email-only identity (no username, no password)
//! - Magic link for first-time enrollment (email → passkey creation)
//! - Passkey/FIDO2 for all subsequent logins (silent biometric assertion)
//! - Email OTP fallback for new devices
//! - PKCE-protected OAuth 2.0 authorization code flow for token issuance

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tracing::info;

use crate::rbac::Role;

// ---------------------------------------------------------------------------
// WebAuthn credential types (server-side representation)
// ---------------------------------------------------------------------------

/// A WebAuthn credential stored on the server.
/// The private key never leaves the user's device; we only store the public key.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PasskeyCredential {
    /// WebAuthn credential ID (base64url-encoded).
    pub credential_id: String,
    /// COSE public key (base64url-encoded).
    pub public_key: String,
    /// Relying Party ID (domain).
    pub rp_id: String,
    /// Monotonically increasing sign counter for clone detection.
    pub sign_count: u64,
    /// AAGUID of the authenticator (e.g., iCloud Keychain, Google Password Manager).
    pub aaguid: String,
    /// When the credential was created.
    pub created_at: u64,
    /// When the credential was last used.
    pub last_used: u64,
    /// Human-readable label (e.g., "iPhone 16 Pro", "MacBook Pro").
    pub device_label: String,
    /// Whether this credential is active.
    pub active: bool,
}

/// A user account identified solely by email.
/// No password, no username — only email + passkey credentials.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PasskeyAccount {
    /// Unique account ID.
    pub id: String,
    /// Email address — the sole identifier.
    pub email: String,
    /// Organization the account belongs to.
    pub org: String,
    /// Role assigned to this account.
    pub role: Role,
    /// Region restrictions (empty = all regions).
    pub regions: Vec<String>,
    /// All registered WebAuthn credentials (multi-device).
    pub credentials: Vec<PasskeyCredential>,
    /// When the account was created.
    pub created_at: u64,
    /// Whether email ownership has been verified.
    pub email_verified: bool,
}

// ---------------------------------------------------------------------------
// Magic link tokens
// ---------------------------------------------------------------------------

/// A signed magic link token for email verification.
/// Encodes { email, nonce, org, exp } signed with HMAC-SHA256.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MagicLinkToken {
    /// The email address being verified.
    pub email: String,
    /// Organization context.
    pub org: String,
    /// Role to assign upon enrollment.
    pub role: Role,
    /// Random nonce (single-use).
    pub nonce: String,
    /// Expiration timestamp (unix seconds).
    pub exp: u64,
    /// HMAC-SHA256 signature.
    pub signature: String,
}

/// Configuration for magic link generation.
#[derive(Debug, Clone)]
pub struct MagicLinkConfig {
    /// HMAC secret for signing magic link tokens.
    pub secret: Vec<u8>,
    /// Time-to-live for magic links (default: 15 minutes).
    pub ttl_secs: u64,
}

impl Default for MagicLinkConfig {
    fn default() -> Self {
        let secret = match std::env::var("INVISIBLE_MAGIC_LINK_SECRET") {
            Ok(s) if !s.is_empty() => s.into_bytes(),
            _ => b"invisible-magic-link-secret-change-in-production".to_vec(),
        };
        Self {
            secret,
            ttl_secs: 900, // 15 minutes
        }
    }
}

// ---------------------------------------------------------------------------
// Email OTP
// ---------------------------------------------------------------------------

/// A one-time password sent via email for device enrollment fallback.
#[derive(Debug, Clone)]
pub struct EmailOtp {
    /// The 6-digit OTP code.
    pub code: String,
    /// Email this OTP was sent to.
    pub email: String,
    /// Organization context.
    pub org: String,
    /// Expiration timestamp (unix seconds).
    pub expires_at: u64,
    /// Number of verification attempts (max 3).
    pub attempts: u32,
    /// Whether this OTP has been consumed.
    pub used: bool,
}

/// Configuration for email OTP.
#[derive(Debug, Clone)]
pub struct OtpConfig {
    /// Time-to-live for OTP codes (default: 5 minutes).
    pub ttl_secs: u64,
    /// Maximum verification attempts before lockout.
    pub max_attempts: u32,
    /// Minimum interval between OTP sends (rate limit, default: 60 seconds).
    pub rate_limit_secs: u64,
}

impl Default for OtpConfig {
    fn default() -> Self {
        Self {
            ttl_secs: 300, // 5 minutes
            max_attempts: 3,
            rate_limit_secs: 60, // 1 per minute
        }
    }
}

// ---------------------------------------------------------------------------
// PKCE (Proof Key for Code Exchange)
// ---------------------------------------------------------------------------

/// An OAuth 2.0 authorization code with PKCE binding.
#[derive(Debug, Clone)]
pub struct AuthorizationCode {
    /// The authorization code value.
    pub code: String,
    /// The email of the authenticated user.
    pub email: String,
    /// Organization.
    pub org: String,
    /// Role.
    pub role: Role,
    /// Regions.
    pub regions: Vec<String>,
    /// PKCE code_challenge (S256).
    pub code_challenge: String,
    /// Expiration (unix seconds). Auth codes are very short-lived (60s).
    pub expires_at: u64,
    /// Whether this code has been exchanged.
    pub used: bool,
}

// ---------------------------------------------------------------------------
// WebAuthn challenge
// ---------------------------------------------------------------------------

/// A WebAuthn challenge for registration or authentication.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebAuthnChallenge {
    /// Random challenge bytes (base64url-encoded).
    pub challenge: String,
    /// Type: "registration" or "authentication".
    pub challenge_type: String,
    /// Relying Party ID.
    pub rp_id: String,
    /// Associated email (for registration) or credential IDs (for authentication).
    pub email: String,
    /// Organization context.
    pub org: String,
    /// Expiration (unix seconds). Challenges are short-lived (5 minutes).
    pub expires_at: u64,
}

// ---------------------------------------------------------------------------
// Passkey service
// ---------------------------------------------------------------------------

/// The core passwordless authentication service.
/// Manages accounts, magic links, OTPs, passkey credentials, and PKCE codes.
pub struct PasskeyService {
    /// Accounts indexed by email (lowercased).
    accounts: Arc<RwLock<HashMap<String, PasskeyAccount>>>,
    /// Active magic link nonces (nonce → MagicLinkToken). Single-use.
    magic_links: Arc<RwLock<HashMap<String, MagicLinkToken>>>,
    /// Active OTP codes (email → EmailOtp).
    otps: Arc<RwLock<HashMap<String, EmailOtp>>>,
    /// Active WebAuthn challenges (challenge_id → WebAuthnChallenge).
    challenges: Arc<RwLock<HashMap<String, WebAuthnChallenge>>>,
    /// Active authorization codes (code → AuthorizationCode). Single-use, 60s TTL.
    auth_codes: Arc<RwLock<HashMap<String, AuthorizationCode>>>,
    /// OTP rate limiter: email → last_sent_timestamp.
    otp_rate_limits: Arc<RwLock<HashMap<String, u64>>>,
    /// Magic link config.
    magic_link_config: MagicLinkConfig,
    /// OTP config.
    otp_config: OtpConfig,
}

impl PasskeyService {
    pub fn new(magic_link_config: MagicLinkConfig, otp_config: OtpConfig) -> Self {
        Self {
            accounts: Arc::new(RwLock::new(HashMap::new())),
            magic_links: Arc::new(RwLock::new(HashMap::new())),
            otps: Arc::new(RwLock::new(HashMap::new())),
            challenges: Arc::new(RwLock::new(HashMap::new())),
            auth_codes: Arc::new(RwLock::new(HashMap::new())),
            otp_rate_limits: Arc::new(RwLock::new(HashMap::new())),
            magic_link_config,
            otp_config,
        }
    }

    /// Create with default config.
    pub fn default_config() -> Self {
        Self::new(MagicLinkConfig::default(), OtpConfig::default())
    }

    // -----------------------------------------------------------------------
    // Account management
    // -----------------------------------------------------------------------

    /// Get an account by email.
    pub fn get_account(&self, email: &str) -> Option<PasskeyAccount> {
        let accounts = self.accounts.read().unwrap();
        accounts.get(&email.to_lowercase()).cloned()
    }

    /// Get an account by credential ID (for discoverable credential flows).
    pub fn get_account_by_credential_id(&self, credential_id: &str) -> Option<PasskeyAccount> {
        let accounts = self.accounts.read().unwrap();
        accounts
            .values()
            .find(|a| {
                a.credentials
                    .iter()
                    .any(|c| c.credential_id == credential_id && c.active)
            })
            .cloned()
    }

    /// Create a new account (email-only, no credentials yet).
    /// Called after magic link verification.
    pub fn create_account(
        &self,
        email: &str,
        org: &str,
        role: Role,
        regions: Vec<String>,
    ) -> PasskeyAccount {
        let now = current_timestamp();
        let account = PasskeyAccount {
            id: uuid::Uuid::new_v4().to_string(),
            email: email.to_lowercase(),
            org: org.to_string(),
            role,
            regions,
            credentials: Vec::new(),
            created_at: now,
            email_verified: true,
        };

        let mut accounts = self.accounts.write().unwrap();
        accounts.insert(email.to_lowercase(), account.clone());

        info!(email = email, org = org, "Passkey account created");
        account
    }

    /// Register a passkey credential for an account.
    pub fn register_credential(
        &self,
        email: &str,
        credential: PasskeyCredential,
    ) -> Result<(), PasskeyError> {
        let mut accounts = self.accounts.write().unwrap();
        let account = accounts
            .get_mut(&email.to_lowercase())
            .ok_or(PasskeyError::AccountNotFound)?;

        // Check for duplicate credential ID
        if account
            .credentials
            .iter()
            .any(|c| c.credential_id == credential.credential_id)
        {
            return Err(PasskeyError::CredentialAlreadyExists);
        }

        info!(
            email = email,
            credential_id = credential.credential_id,
            device = credential.device_label,
            "Passkey credential registered"
        );

        account.credentials.push(credential);
        Ok(())
    }

    /// Update sign counter after successful authentication (clone detection).
    pub fn update_sign_count(
        &self,
        email: &str,
        credential_id: &str,
        new_count: u64,
    ) -> Result<(), PasskeyError> {
        let mut accounts = self.accounts.write().unwrap();
        let account = accounts
            .get_mut(&email.to_lowercase())
            .ok_or(PasskeyError::AccountNotFound)?;

        let cred = account
            .credentials
            .iter_mut()
            .find(|c| c.credential_id == credential_id && c.active)
            .ok_or(PasskeyError::CredentialNotFound)?;

        // Clone detection: new count must be greater than stored count
        if new_count <= cred.sign_count {
            return Err(PasskeyError::PossibleClone {
                stored: cred.sign_count,
                received: new_count,
            });
        }

        cred.sign_count = new_count;
        cred.last_used = current_timestamp();
        Ok(())
    }

    /// Revoke a specific credential.
    pub fn revoke_credential(&self, email: &str, credential_id: &str) -> Result<(), PasskeyError> {
        let mut accounts = self.accounts.write().unwrap();
        let account = accounts
            .get_mut(&email.to_lowercase())
            .ok_or(PasskeyError::AccountNotFound)?;

        let cred = account
            .credentials
            .iter_mut()
            .find(|c| c.credential_id == credential_id)
            .ok_or(PasskeyError::CredentialNotFound)?;

        cred.active = false;
        info!(
            email = email,
            credential_id = credential_id,
            "Passkey credential revoked"
        );
        Ok(())
    }

    /// List all credentials for an account.
    pub fn list_credentials(&self, email: &str) -> Result<Vec<PasskeyCredential>, PasskeyError> {
        let accounts = self.accounts.read().unwrap();
        let account = accounts
            .get(&email.to_lowercase())
            .ok_or(PasskeyError::AccountNotFound)?;
        Ok(account.credentials.clone())
    }

    // -----------------------------------------------------------------------
    // Magic link operations
    // -----------------------------------------------------------------------

    /// Generate a magic link token for email verification.
    /// Returns (token_string, nonce) — the token_string is sent in the email link.
    pub fn create_magic_link(
        &self,
        email: &str,
        org: &str,
        role: Role,
    ) -> Result<MagicLinkToken, PasskeyError> {
        let now = current_timestamp();

        // Generate random nonce
        let nonce = generate_random_hex(32);

        // Compute HMAC signature
        let message = format!(
            "{}:{}:{}:{}",
            email.to_lowercase(),
            org,
            nonce,
            now + self.magic_link_config.ttl_secs
        );
        let signature = hmac_sign(&self.magic_link_config.secret, &message);

        let token = MagicLinkToken {
            email: email.to_lowercase(),
            org: org.to_string(),
            role,
            nonce: nonce.clone(),
            exp: now + self.magic_link_config.ttl_secs,
            signature,
        };

        // Store nonce for single-use enforcement
        let mut links = self.magic_links.write().unwrap();
        links.insert(nonce, token.clone());

        info!(email = email, org = org, "Magic link created");
        Ok(token)
    }

    /// Verify a magic link token and consume it (single-use).
    /// On success, creates the account if it doesn't exist and returns it.
    pub fn verify_magic_link(
        &self,
        email: &str,
        nonce: &str,
        signature: &str,
    ) -> Result<PasskeyAccount, PasskeyError> {
        let now = current_timestamp();

        // Consume the nonce (single-use)
        let token = {
            let mut links = self.magic_links.write().unwrap();
            links.remove(nonce).ok_or(PasskeyError::InvalidMagicLink)?
        };

        // Verify email matches
        if token.email != email.to_lowercase() {
            return Err(PasskeyError::InvalidMagicLink);
        }

        // Check expiration
        if now > token.exp {
            return Err(PasskeyError::MagicLinkExpired);
        }

        // Verify HMAC signature
        let message = format!(
            "{}:{}:{}:{}",
            token.email, token.org, token.nonce, token.exp
        );
        let expected_sig = hmac_sign(&self.magic_link_config.secret, &message);
        if signature != expected_sig {
            return Err(PasskeyError::InvalidMagicLink);
        }

        // Create account if it doesn't exist, or return existing
        let account = match self.get_account(email) {
            Some(existing) => existing,
            None => self.create_account(email, &token.org, token.role, vec![]),
        };

        info!(email = email, "Magic link verified");
        Ok(account)
    }

    // -----------------------------------------------------------------------
    // Email OTP operations
    // -----------------------------------------------------------------------

    /// Generate a 6-digit OTP for the given email.
    /// Returns the OTP code (to be sent via email).
    pub fn create_otp(&self, email: &str, org: &str) -> Result<String, PasskeyError> {
        let now = current_timestamp();
        let email_lower = email.to_lowercase();

        // Rate limit check
        {
            let rates = self.otp_rate_limits.read().unwrap();
            if let Some(&last_sent) = rates.get(&email_lower)
                && now - last_sent < self.otp_config.rate_limit_secs
            {
                return Err(PasskeyError::OtpRateLimited {
                    retry_after_secs: self.otp_config.rate_limit_secs - (now - last_sent),
                });
            }
        }

        // Generate 6-digit code
        let code = generate_otp_code();

        let otp = EmailOtp {
            code: code.clone(),
            email: email_lower.clone(),
            org: org.to_string(),
            expires_at: now + self.otp_config.ttl_secs,
            attempts: 0,
            used: false,
        };

        // Store OTP
        {
            let mut otps = self.otps.write().unwrap();
            otps.insert(email_lower.clone(), otp);
        }

        // Update rate limiter
        {
            let mut rates = self.otp_rate_limits.write().unwrap();
            rates.insert(email_lower, now);
        }

        info!(email = email, "Email OTP created");
        Ok(code)
    }

    /// Verify an OTP code. On success, returns the account.
    pub fn verify_otp(&self, email: &str, code: &str) -> Result<PasskeyAccount, PasskeyError> {
        let now = current_timestamp();
        let email_lower = email.to_lowercase();

        let mut otps = self.otps.write().unwrap();
        let otp = otps.get_mut(&email_lower).ok_or(PasskeyError::InvalidOtp)?;

        // Check if already used
        if otp.used {
            return Err(PasskeyError::OtpAlreadyUsed);
        }

        // Check expiration
        if now > otp.expires_at {
            return Err(PasskeyError::OtpExpired);
        }

        // Increment attempts
        otp.attempts += 1;
        if otp.attempts > self.otp_config.max_attempts {
            otp.used = true; // Lock it out
            return Err(PasskeyError::OtpMaxAttempts);
        }

        // Constant-time comparison
        if !constant_time_eq(code.as_bytes(), otp.code.as_bytes()) {
            return Err(PasskeyError::InvalidOtp);
        }

        // Mark as used
        otp.used = true;
        drop(otps);

        // Return the account
        self.get_account(email).ok_or(PasskeyError::AccountNotFound)
    }

    // -----------------------------------------------------------------------
    // WebAuthn challenge operations
    // -----------------------------------------------------------------------

    /// Create a WebAuthn registration challenge.
    pub fn create_registration_challenge(
        &self,
        email: &str,
        org: &str,
        rp_id: &str,
    ) -> WebAuthnChallenge {
        let now = current_timestamp();
        let challenge_bytes = generate_random_hex(32);

        let challenge = WebAuthnChallenge {
            challenge: challenge_bytes.clone(),
            challenge_type: "registration".to_string(),
            rp_id: rp_id.to_string(),
            email: email.to_lowercase(),
            org: org.to_string(),
            expires_at: now + 300, // 5 minutes
        };

        let mut challenges = self.challenges.write().unwrap();
        challenges.insert(challenge_bytes, challenge.clone());

        challenge
    }

    /// Create a WebAuthn authentication challenge.
    pub fn create_authentication_challenge(
        &self,
        email: &str,
        org: &str,
        rp_id: &str,
    ) -> Result<(WebAuthnChallenge, Vec<String>), PasskeyError> {
        let account = self
            .get_account(email)
            .ok_or(PasskeyError::AccountNotFound)?;

        let credential_ids: Vec<String> = account
            .credentials
            .iter()
            .filter(|c| c.active)
            .map(|c| c.credential_id.clone())
            .collect();

        if credential_ids.is_empty() {
            return Err(PasskeyError::NoCredentials);
        }

        let now = current_timestamp();
        let challenge_bytes = generate_random_hex(32);

        let challenge = WebAuthnChallenge {
            challenge: challenge_bytes.clone(),
            challenge_type: "authentication".to_string(),
            rp_id: rp_id.to_string(),
            email: email.to_lowercase(),
            org: org.to_string(),
            expires_at: now + 300,
        };

        let mut challenges = self.challenges.write().unwrap();
        challenges.insert(challenge_bytes, challenge.clone());

        Ok((challenge, credential_ids))
    }

    /// Consume a WebAuthn challenge (verify it exists and hasn't expired).
    pub fn consume_challenge(&self, challenge_id: &str) -> Result<WebAuthnChallenge, PasskeyError> {
        let now = current_timestamp();

        let mut challenges = self.challenges.write().unwrap();
        let challenge = challenges
            .remove(challenge_id)
            .ok_or(PasskeyError::InvalidChallenge)?;

        if now > challenge.expires_at {
            return Err(PasskeyError::ChallengeExpired);
        }

        Ok(challenge)
    }

    // -----------------------------------------------------------------------
    // PKCE authorization code operations
    // -----------------------------------------------------------------------

    /// Create a PKCE-bound authorization code after successful authentication.
    /// The code_challenge is the S256 hash the client computed from code_verifier.
    pub fn create_auth_code(
        &self,
        email: &str,
        org: &str,
        role: Role,
        regions: &[String],
        code_challenge: &str,
    ) -> String {
        let code = generate_random_hex(32);
        let now = current_timestamp();

        let auth_code = AuthorizationCode {
            code: code.clone(),
            email: email.to_lowercase(),
            org: org.to_string(),
            role,
            regions: regions.to_vec(),
            code_challenge: code_challenge.to_string(),
            expires_at: now + 60, // 60 seconds
            used: false,
        };

        let mut codes = self.auth_codes.write().unwrap();
        codes.insert(code.clone(), auth_code);

        code
    }

    /// Exchange an authorization code for account info, verifying the PKCE code_verifier.
    /// Returns (email, org, role, regions) on success.
    pub fn exchange_auth_code(
        &self,
        code: &str,
        code_verifier: &str,
    ) -> Result<(String, String, Role, Vec<String>), PasskeyError> {
        let now = current_timestamp();

        let mut codes = self.auth_codes.write().unwrap();
        let auth_code = codes.get_mut(code).ok_or(PasskeyError::InvalidAuthCode)?;

        if auth_code.used {
            return Err(PasskeyError::AuthCodeAlreadyUsed);
        }

        if now > auth_code.expires_at {
            return Err(PasskeyError::AuthCodeExpired);
        }

        // Verify PKCE: S256(code_verifier) must equal stored code_challenge
        let computed_challenge = pkce_s256(code_verifier);
        if computed_challenge != auth_code.code_challenge {
            return Err(PasskeyError::PkceVerificationFailed);
        }

        auth_code.used = true;
        let result = (
            auth_code.email.clone(),
            auth_code.org.clone(),
            auth_code.role,
            auth_code.regions.clone(),
        );

        Ok(result)
    }

    // -----------------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------------

    /// Remove expired magic links, OTPs, challenges, and auth codes.
    pub fn cleanup_expired(&self) -> usize {
        let now = current_timestamp();
        let mut cleaned = 0;

        {
            let mut links = self.magic_links.write().unwrap();
            let before = links.len();
            links.retain(|_, v| v.exp > now);
            cleaned += before - links.len();
        }
        {
            let mut otps = self.otps.write().unwrap();
            let before = otps.len();
            otps.retain(|_, v| v.expires_at > now);
            cleaned += before - otps.len();
        }
        {
            let mut challenges = self.challenges.write().unwrap();
            let before = challenges.len();
            challenges.retain(|_, v| v.expires_at > now);
            cleaned += before - challenges.len();
        }
        {
            let mut codes = self.auth_codes.write().unwrap();
            let before = codes.len();
            codes.retain(|_, v| v.expires_at > now);
            cleaned += before - codes.len();
        }

        cleaned
    }

    /// Get statistics about the passkey service state.
    pub fn stats(&self) -> PasskeyStats {
        PasskeyStats {
            total_accounts: self.accounts.read().unwrap().len(),
            active_magic_links: self.magic_links.read().unwrap().len(),
            active_otps: self.otps.read().unwrap().len(),
            active_challenges: self.challenges.read().unwrap().len(),
            active_auth_codes: self.auth_codes.read().unwrap().len(),
        }
    }
}

/// Statistics about the passkey service.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PasskeyStats {
    pub total_accounts: usize,
    pub active_magic_links: usize,
    pub active_otps: usize,
    pub active_challenges: usize,
    pub active_auth_codes: usize,
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum PasskeyError {
    #[error("account not found")]
    AccountNotFound,

    #[error("credential not found")]
    CredentialNotFound,

    #[error("credential already exists")]
    CredentialAlreadyExists,

    #[error("no active credentials for this account")]
    NoCredentials,

    #[error("possible credential clone detected: stored={stored}, received={received}")]
    PossibleClone { stored: u64, received: u64 },

    #[error("invalid magic link")]
    InvalidMagicLink,

    #[error("magic link expired")]
    MagicLinkExpired,

    #[error("invalid OTP code")]
    InvalidOtp,

    #[error("OTP expired")]
    OtpExpired,

    #[error("OTP already used")]
    OtpAlreadyUsed,

    #[error("OTP maximum attempts exceeded")]
    OtpMaxAttempts,

    #[error("OTP rate limited, retry after {retry_after_secs}s")]
    OtpRateLimited { retry_after_secs: u64 },

    #[error("invalid WebAuthn challenge")]
    InvalidChallenge,

    #[error("WebAuthn challenge expired")]
    ChallengeExpired,

    #[error("invalid authorization code")]
    InvalidAuthCode,

    #[error("authorization code already used")]
    AuthCodeAlreadyUsed,

    #[error("authorization code expired")]
    AuthCodeExpired,

    #[error("PKCE verification failed")]
    PkceVerificationFailed,
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

/// Generate a random hex string of the given number of bytes.
fn generate_random_hex(bytes: usize) -> String {
    use rand::RngExt;
    let mut rng = rand::rng();
    let mut buf = vec![0u8; bytes];
    rng.fill(&mut buf);
    hex::encode(buf)
}

/// Generate a 6-digit OTP code.
fn generate_otp_code() -> String {
    use rand::RngExt;
    let mut rng = rand::rng();
    let code: u32 = rng.random_range(100_000..1_000_000);
    format!("{:06}", code)
}

/// HMAC-SHA256 sign a message.
fn hmac_sign(secret: &[u8], message: &str) -> String {
    use hmac::{Hmac, Mac};
    let mut mac = Hmac::<Sha256>::new_from_slice(secret).expect("HMAC key length is valid");
    mac.update(message.as_bytes());
    hex::encode(mac.finalize().into_bytes())
}

/// PKCE S256 challenge: BASE64URL(SHA256(code_verifier)).
pub fn pkce_s256(code_verifier: &str) -> String {
    use base64::Engine;
    let hash = Sha256::digest(code_verifier.as_bytes());
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(hash)
}

/// Generate a PKCE code_verifier (43-128 unreserved characters).
pub fn generate_pkce_verifier() -> String {
    generate_random_hex(32) // 64 hex chars
}

/// Constant-time byte comparison to prevent timing attacks on OTP verification.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut result = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        result |= x ^ y;
    }
    result == 0
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_service() -> PasskeyService {
        PasskeyService::new(
            MagicLinkConfig {
                secret: b"test-magic-link-secret".to_vec(),
                ttl_secs: 900,
            },
            OtpConfig {
                ttl_secs: 300,
                max_attempts: 3,
                rate_limit_secs: 1, // 1 second for tests
            },
        )
    }

    fn test_credential() -> PasskeyCredential {
        PasskeyCredential {
            credential_id: "cred-id-123".to_string(),
            public_key: "pk-base64url".to_string(),
            rp_id: "example.com".to_string(),
            sign_count: 0,
            aaguid: "00000000-0000-0000-0000-000000000000".to_string(),
            created_at: current_timestamp(),
            last_used: current_timestamp(),
            device_label: "Test Device".to_string(),
            active: true,
        }
    }

    // --- Account tests ---

    #[test]
    fn create_account_email_only() {
        let svc = test_service();
        let account = svc.create_account("User@Example.COM", "acme", Role::Customer, vec![]);

        assert_eq!(account.email, "user@example.com"); // lowercased
        assert_eq!(account.org, "acme");
        assert_eq!(account.role, Role::Customer);
        assert!(account.email_verified);
        assert!(account.credentials.is_empty());
    }

    #[test]
    fn get_account_case_insensitive() {
        let svc = test_service();
        svc.create_account("Admin@ACME.com", "acme", Role::Admin, vec![]);

        assert!(svc.get_account("admin@acme.com").is_some());
        assert!(svc.get_account("ADMIN@ACME.COM").is_some());
        assert!(svc.get_account("nobody@acme.com").is_none());
    }

    #[test]
    fn register_and_list_credentials() {
        let svc = test_service();
        svc.create_account("user@test.com", "org1", Role::Customer, vec![]);

        let cred = test_credential();
        svc.register_credential("user@test.com", cred).unwrap();

        let creds = svc.list_credentials("user@test.com").unwrap();
        assert_eq!(creds.len(), 1);
        assert_eq!(creds[0].credential_id, "cred-id-123");
    }

    #[test]
    fn duplicate_credential_rejected() {
        let svc = test_service();
        svc.create_account("user@test.com", "org1", Role::Customer, vec![]);

        let cred = test_credential();
        svc.register_credential("user@test.com", cred.clone())
            .unwrap();

        let result = svc.register_credential("user@test.com", cred);
        assert!(matches!(result, Err(PasskeyError::CredentialAlreadyExists)));
    }

    #[test]
    fn sign_count_clone_detection() {
        let svc = test_service();
        svc.create_account("user@test.com", "org1", Role::Customer, vec![]);

        let cred = test_credential();
        svc.register_credential("user@test.com", cred).unwrap();

        // Normal increment
        svc.update_sign_count("user@test.com", "cred-id-123", 1)
            .unwrap();

        // Replayed/cloned count
        let result = svc.update_sign_count("user@test.com", "cred-id-123", 1);
        assert!(matches!(result, Err(PasskeyError::PossibleClone { .. })));

        // Lower count
        let result = svc.update_sign_count("user@test.com", "cred-id-123", 0);
        assert!(matches!(result, Err(PasskeyError::PossibleClone { .. })));
    }

    #[test]
    fn revoke_credential() {
        let svc = test_service();
        svc.create_account("user@test.com", "org1", Role::Customer, vec![]);

        let cred = test_credential();
        svc.register_credential("user@test.com", cred).unwrap();
        svc.revoke_credential("user@test.com", "cred-id-123")
            .unwrap();

        let creds = svc.list_credentials("user@test.com").unwrap();
        assert!(!creds[0].active);
    }

    #[test]
    fn get_account_by_credential_id() {
        let svc = test_service();
        svc.create_account("user@test.com", "org1", Role::Customer, vec![]);

        let cred = test_credential();
        svc.register_credential("user@test.com", cred).unwrap();

        let account = svc.get_account_by_credential_id("cred-id-123");
        assert!(account.is_some());
        assert_eq!(account.unwrap().email, "user@test.com");

        // Revoke and check
        svc.revoke_credential("user@test.com", "cred-id-123")
            .unwrap();
        assert!(svc.get_account_by_credential_id("cred-id-123").is_none());
    }

    // --- Magic link tests ---

    #[test]
    fn magic_link_flow() {
        let svc = test_service();

        // Create magic link
        let token = svc
            .create_magic_link("new@user.com", "acme", Role::Customer)
            .unwrap();

        // Verify it
        let account = svc
            .verify_magic_link(&token.email, &token.nonce, &token.signature)
            .unwrap();

        assert_eq!(account.email, "new@user.com");
        assert_eq!(account.org, "acme");
        assert!(account.email_verified);
    }

    #[test]
    fn magic_link_single_use() {
        let svc = test_service();

        let token = svc
            .create_magic_link("user@test.com", "acme", Role::Customer)
            .unwrap();

        // First use succeeds
        svc.verify_magic_link(&token.email, &token.nonce, &token.signature)
            .unwrap();

        // Second use fails (nonce consumed)
        let result = svc.verify_magic_link(&token.email, &token.nonce, &token.signature);
        assert!(matches!(result, Err(PasskeyError::InvalidMagicLink)));
    }

    #[test]
    fn magic_link_wrong_email() {
        let svc = test_service();

        let token = svc
            .create_magic_link("user@test.com", "acme", Role::Customer)
            .unwrap();

        let result = svc.verify_magic_link("wrong@email.com", &token.nonce, &token.signature);
        assert!(matches!(result, Err(PasskeyError::InvalidMagicLink)));
    }

    #[test]
    fn magic_link_wrong_signature() {
        let svc = test_service();

        let token = svc
            .create_magic_link("user@test.com", "acme", Role::Customer)
            .unwrap();

        let result = svc.verify_magic_link(&token.email, &token.nonce, "wrong-signature");
        assert!(matches!(result, Err(PasskeyError::InvalidMagicLink)));
    }

    // --- OTP tests ---

    #[test]
    fn otp_flow() {
        let svc = test_service();
        svc.create_account("user@test.com", "acme", Role::Customer, vec![]);

        let code = svc.create_otp("user@test.com", "acme").unwrap();
        assert_eq!(code.len(), 6);

        let account = svc.verify_otp("user@test.com", &code).unwrap();
        assert_eq!(account.email, "user@test.com");
    }

    #[test]
    fn otp_wrong_code() {
        let svc = test_service();
        svc.create_account("user@test.com", "acme", Role::Customer, vec![]);

        svc.create_otp("user@test.com", "acme").unwrap();

        let result = svc.verify_otp("user@test.com", "000000");
        assert!(matches!(result, Err(PasskeyError::InvalidOtp)));
    }

    #[test]
    fn otp_max_attempts() {
        let svc = test_service();
        svc.create_account("user@test.com", "acme", Role::Customer, vec![]);

        svc.create_otp("user@test.com", "acme").unwrap();

        // Exhaust attempts
        for _ in 0..3 {
            let _ = svc.verify_otp("user@test.com", "000000");
        }

        // Next attempt should be locked out
        let result = svc.verify_otp("user@test.com", "000000");
        assert!(matches!(result, Err(PasskeyError::OtpMaxAttempts)));
    }

    #[test]
    fn otp_single_use() {
        let svc = test_service();
        svc.create_account("user@test.com", "acme", Role::Customer, vec![]);

        let code = svc.create_otp("user@test.com", "acme").unwrap();

        // First use succeeds
        svc.verify_otp("user@test.com", &code).unwrap();

        // Second use fails
        let result = svc.verify_otp("user@test.com", &code);
        assert!(matches!(result, Err(PasskeyError::OtpAlreadyUsed)));
    }

    #[test]
    fn otp_rate_limiting() {
        let svc = test_service();
        svc.create_account("user@test.com", "acme", Role::Customer, vec![]);

        // First OTP succeeds
        svc.create_otp("user@test.com", "acme").unwrap();

        // Immediate second attempt should be rate limited
        let result = svc.create_otp("user@test.com", "acme");
        assert!(matches!(result, Err(PasskeyError::OtpRateLimited { .. })));
    }

    // --- PKCE tests ---

    #[test]
    fn pkce_auth_code_flow() {
        let svc = test_service();

        let verifier = generate_pkce_verifier();
        let challenge = pkce_s256(&verifier);

        let code = svc.create_auth_code("user@test.com", "acme", Role::Customer, &[], &challenge);

        let (email, org, role, _regions) = svc.exchange_auth_code(&code, &verifier).unwrap();

        assert_eq!(email, "user@test.com");
        assert_eq!(org, "acme");
        assert_eq!(role, Role::Customer);
    }

    #[test]
    fn pkce_wrong_verifier() {
        let svc = test_service();

        let verifier = generate_pkce_verifier();
        let challenge = pkce_s256(&verifier);

        let code = svc.create_auth_code("user@test.com", "acme", Role::Customer, &[], &challenge);

        let result = svc.exchange_auth_code(&code, "wrong-verifier");
        assert!(matches!(result, Err(PasskeyError::PkceVerificationFailed)));
    }

    #[test]
    fn pkce_code_single_use() {
        let svc = test_service();

        let verifier = generate_pkce_verifier();
        let challenge = pkce_s256(&verifier);

        let code = svc.create_auth_code("user@test.com", "acme", Role::Customer, &[], &challenge);

        // First exchange succeeds
        svc.exchange_auth_code(&code, &verifier).unwrap();

        // Second exchange fails
        let result = svc.exchange_auth_code(&code, &verifier);
        assert!(matches!(result, Err(PasskeyError::AuthCodeAlreadyUsed)));
    }

    // --- WebAuthn challenge tests ---

    #[test]
    fn registration_challenge_flow() {
        let svc = test_service();

        let challenge = svc.create_registration_challenge("user@test.com", "acme", "example.com");

        assert_eq!(challenge.challenge_type, "registration");
        assert_eq!(challenge.rp_id, "example.com");
        assert_eq!(challenge.email, "user@test.com");

        // Consume it
        let consumed = svc.consume_challenge(&challenge.challenge).unwrap();
        assert_eq!(consumed.email, "user@test.com");

        // Double-consume fails
        let result = svc.consume_challenge(&challenge.challenge);
        assert!(matches!(result, Err(PasskeyError::InvalidChallenge)));
    }

    #[test]
    fn authentication_challenge_requires_credentials() {
        let svc = test_service();
        svc.create_account("user@test.com", "acme", Role::Customer, vec![]);

        // No credentials yet
        let result = svc.create_authentication_challenge("user@test.com", "acme", "example.com");
        assert!(matches!(result, Err(PasskeyError::NoCredentials)));

        // Register a credential
        svc.register_credential("user@test.com", test_credential())
            .unwrap();

        // Now it works
        let (challenge, cred_ids) = svc
            .create_authentication_challenge("user@test.com", "acme", "example.com")
            .unwrap();

        assert_eq!(challenge.challenge_type, "authentication");
        assert_eq!(cred_ids, vec!["cred-id-123"]);
    }

    // --- Cleanup tests ---

    #[test]
    fn cleanup_expired_entries() {
        let svc = PasskeyService::new(
            MagicLinkConfig {
                secret: b"test".to_vec(),
                ttl_secs: 0, // expire immediately
            },
            OtpConfig {
                ttl_secs: 0,
                max_attempts: 3,
                rate_limit_secs: 0,
            },
        );

        svc.create_account("user@test.com", "acme", Role::Customer, vec![]);

        // Create items that will expire
        let _ = svc.create_magic_link("user@test.com", "acme", Role::Customer);
        let _ = svc.create_otp("user@test.com", "acme");
        let _ = svc.create_registration_challenge("user@test.com", "acme", "example.com");

        // Wait for expiry
        std::thread::sleep(std::time::Duration::from_millis(1100));

        let cleaned = svc.cleanup_expired();
        assert!(cleaned >= 2); // magic link + challenge at minimum
    }

    // --- PKCE S256 tests ---

    #[test]
    fn pkce_s256_deterministic() {
        let verifier = "test-verifier-string";
        let c1 = pkce_s256(verifier);
        let c2 = pkce_s256(verifier);
        assert_eq!(c1, c2);
    }

    #[test]
    fn pkce_s256_different_verifiers() {
        assert_ne!(pkce_s256("verifier-a"), pkce_s256("verifier-b"));
    }

    // --- Stats ---

    #[test]
    fn stats_reflect_state() {
        let svc = test_service();
        let s = svc.stats();
        assert_eq!(s.total_accounts, 0);

        svc.create_account("user@test.com", "acme", Role::Customer, vec![]);
        let s = svc.stats();
        assert_eq!(s.total_accounts, 1);
    }
}
