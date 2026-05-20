//! SCRAM-SHA-256 authentication (RFC 5802 + RFC 7677).
//!
//! Implements the server side of the SASL SCRAM-SHA-256 mechanism used by
//! PostgreSQL wire protocol v3. This is the standard authentication method
//! for modern PostgreSQL clients.
//!
//! # Stored Credentials
//!
//! The server stores `ScramCredentials` per user (salt, iterations,
//! StoredKey, ServerKey). The plaintext password is never stored.
//!
//! # Protocol Flow
//!
//! 1. Server receives client-first-message (username + client nonce)
//! 2. Server sends server-first-message (combined nonce + salt + iterations)
//! 3. Server receives client-final-message (channel binding + proof)
//! 4. Server verifies proof, sends server-final-message (server signature)

use base64::{Engine, engine::general_purpose::STANDARD as BASE64};
use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};

type HmacSha256 = Hmac<Sha256>;

const DEFAULT_ITERATIONS: u32 = 4096;
const SALT_LENGTH: usize = 16;

// ============================================================================
// Credentials
// ============================================================================

/// SCRAM-SHA-256 credentials stored per user.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ScramCredentials {
    pub salt: Vec<u8>,
    pub iterations: u32,
    pub stored_key: Vec<u8>,
    pub server_key: Vec<u8>,
}

/// Generate SCRAM credentials from a plaintext password.
pub fn generate_credentials(password: &str) -> ScramCredentials {
    let salt: Vec<u8> = (0..SALT_LENGTH).map(|_| rand::random::<u8>()).collect();
    generate_credentials_with_salt(password, &salt, DEFAULT_ITERATIONS)
}

/// Generate SCRAM credentials with a specific salt and iteration count.
pub fn generate_credentials_with_salt(
    password: &str,
    salt: &[u8],
    iterations: u32,
) -> ScramCredentials {
    let salted_password = pbkdf2_hmac_sha256(password.as_bytes(), salt, iterations);
    let client_key = hmac_sha256(&salted_password, b"Client Key");
    let stored_key = sha256(&client_key);
    let server_key = hmac_sha256(&salted_password, b"Server Key");

    ScramCredentials {
        salt: salt.to_vec(),
        iterations,
        stored_key: stored_key.to_vec(),
        server_key: server_key.to_vec(),
    }
}

/// Serialize credentials to a JSON string for storage in user metadata.
pub fn credentials_to_json(creds: &ScramCredentials) -> String {
    serde_json::json!({
        "scram_sha256": {
            "salt": BASE64.encode(&creds.salt),
            "iterations": creds.iterations,
            "stored_key": BASE64.encode(&creds.stored_key),
            "server_key": BASE64.encode(&creds.server_key),
        }
    })
    .to_string()
}

/// Deserialize credentials from a JSON string stored in user metadata.
/// Returns None if the format doesn't match (e.g., old SHA-256-only hash).
pub fn credentials_from_json(json_str: &str) -> Option<ScramCredentials> {
    let v: serde_json::Value = serde_json::from_str(json_str).ok()?;
    let scram = v.get("scram_sha256")?;

    let salt = BASE64.decode(scram.get("salt")?.as_str()?).ok()?;
    let iterations = scram.get("iterations")?.as_u64()? as u32;
    let stored_key = BASE64.decode(scram.get("stored_key")?.as_str()?).ok()?;
    let server_key = BASE64.decode(scram.get("server_key")?.as_str()?).ok()?;

    Some(ScramCredentials {
        salt,
        iterations,
        stored_key,
        server_key,
    })
}

// ============================================================================
// Server State Machine
// ============================================================================

/// SCRAM server authentication error.
#[derive(Debug)]
pub enum ScramError {
    /// Protocol error (malformed message).
    Protocol(String),
    /// Authentication failed (wrong password).
    AuthFailed,
}

impl std::fmt::Display for ScramError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScramError::Protocol(msg) => write!(f, "SCRAM protocol error: {}", msg),
            ScramError::AuthFailed => write!(f, "SCRAM authentication failed"),
        }
    }
}

/// Server-side SCRAM-SHA-256 state machine.
pub struct ScramServer {
    credentials: ScramCredentials,
    server_nonce: String,
    state: ScramState,
    // Saved from client-first for AuthMessage construction
    client_first_bare: String,
    server_first: String,
}

enum ScramState {
    /// Waiting for client-first-message.
    Init,
    /// Sent server-first, waiting for client-final-message.
    WaitingClientFinal,
    /// Authentication complete.
    Done,
}

impl ScramServer {
    /// Create a new SCRAM server authenticator for the given user credentials.
    pub fn new(credentials: &ScramCredentials) -> Self {
        // Generate a random server nonce
        let server_nonce: String = (0..24)
            .map(|_| {
                let idx = rand::random::<u8>() % 62;
                match idx {
                    0..=9 => (b'0' + idx) as char,
                    10..=35 => (b'a' + idx - 10) as char,
                    _ => (b'A' + idx - 36) as char,
                }
            })
            .collect();

        Self {
            credentials: credentials.clone(),
            server_nonce,
            state: ScramState::Init,
            client_first_bare: String::new(),
            server_first: String::new(),
        }
    }

    /// Process the client-first-message and produce the server-first-message.
    ///
    /// Input: `n,,n=username,r=client_nonce` (the full client-first-message)
    /// Output: `r=combined_nonce,s=base64_salt,i=iterations`
    pub fn handle_client_first(&mut self, msg: &str) -> Result<String, ScramError> {
        if !matches!(self.state, ScramState::Init) {
            return Err(ScramError::Protocol("unexpected client-first".into()));
        }

        // Parse: "n,,n=user,r=nonce" or "n,,n=user,r=nonce,..."
        // The "n,," is the GS2 header (no channel binding).
        let client_first_bare = if let Some(bare) = msg.strip_prefix("n,,") {
            bare
        } else if msg.starts_with("n=") {
            // Some clients send just the bare portion
            msg
        } else {
            return Err(ScramError::Protocol(format!(
                "invalid client-first: expected 'n,,' prefix, got '{}'",
                &msg[..msg.len().min(20)]
            )));
        };

        // Extract client nonce from r= attribute
        let client_nonce = parse_attribute(client_first_bare, 'r')
            .ok_or_else(|| ScramError::Protocol("missing r= in client-first".into()))?;

        // Build combined nonce
        let combined_nonce = format!("{}{}", client_nonce, self.server_nonce);

        // Build server-first-message
        let server_first = format!(
            "r={},s={},i={}",
            combined_nonce,
            BASE64.encode(&self.credentials.salt),
            self.credentials.iterations,
        );

        self.client_first_bare = client_first_bare.to_string();
        self.server_first = server_first.clone();
        self.state = ScramState::WaitingClientFinal;

        Ok(server_first)
    }

    /// Process the client-final-message, verify the proof, and produce
    /// the server-final-message.
    ///
    /// Input: `c=biws,r=combined_nonce,p=base64_proof`
    /// Output: `v=base64_server_signature`
    pub fn handle_client_final(&mut self, msg: &str) -> Result<String, ScramError> {
        if !matches!(self.state, ScramState::WaitingClientFinal) {
            return Err(ScramError::Protocol("unexpected client-final".into()));
        }

        // Extract the proof
        let proof_b64 = parse_attribute(msg, 'p')
            .ok_or_else(|| ScramError::Protocol("missing p= in client-final".into()))?;
        let client_proof = BASE64
            .decode(proof_b64)
            .map_err(|e| ScramError::Protocol(format!("invalid base64 proof: {}", e)))?;

        // Build client-final-without-proof (everything before ",p=")
        let client_final_without_proof = msg
            .rfind(",p=")
            .map(|i| &msg[..i])
            .ok_or_else(|| ScramError::Protocol("missing ,p= separator".into()))?;

        // AuthMessage = client-first-bare + "," + server-first + "," + client-final-without-proof
        let auth_message = format!(
            "{},{},{}",
            self.client_first_bare, self.server_first, client_final_without_proof,
        );

        // Verify the client proof
        let client_signature = hmac_sha256(&self.credentials.stored_key, auth_message.as_bytes());

        // ClientKey = ClientProof XOR ClientSignature
        if client_proof.len() != client_signature.len() {
            return Err(ScramError::AuthFailed);
        }
        let recovered_client_key: Vec<u8> = client_proof
            .iter()
            .zip(client_signature.iter())
            .map(|(a, b)| a ^ b)
            .collect();

        // StoredKey should equal SHA256(ClientKey)
        let recovered_stored_key = sha256(&recovered_client_key);
        if recovered_stored_key.as_slice() != self.credentials.stored_key.as_slice() {
            return Err(ScramError::AuthFailed);
        }

        // Compute server signature for the client to verify
        let server_signature = hmac_sha256(&self.credentials.server_key, auth_message.as_bytes());
        let server_final = format!("v={}", BASE64.encode(&server_signature));

        self.state = ScramState::Done;
        Ok(server_final)
    }
}

// ============================================================================
// Crypto Primitives
// ============================================================================

fn pbkdf2_hmac_sha256(password: &[u8], salt: &[u8], iterations: u32) -> Vec<u8> {
    let mut output = vec![0u8; 32];
    pbkdf2::pbkdf2_hmac::<Sha256>(password, salt, iterations, &mut output);
    output
}

fn hmac_sha256(key: &[u8], data: &[u8]) -> Vec<u8> {
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC can take key of any size");
    mac.update(data);
    mac.finalize().into_bytes().to_vec()
}

fn sha256(data: &[u8]) -> Vec<u8> {
    Sha256::digest(data).to_vec()
}

// ============================================================================
// Helpers
// ============================================================================

/// Parse a SCRAM attribute (e.g., extract "nonce" from "r=nonce,s=salt").
fn parse_attribute(msg: &str, attr: char) -> Option<&str> {
    let prefix = format!("{}=", attr);
    for part in msg.split(',') {
        if let Some(value) = part.strip_prefix(&prefix) {
            return Some(value);
        }
    }
    None
}

// ============================================================================
// Client-Side (for testing)
// ============================================================================

/// A minimal SCRAM-SHA-256 client for testing purposes.
#[cfg(test)]
pub struct ScramClient {
    username: String,
    password: String,
    client_nonce: String,
    client_first_bare: String,
    server_first: String,
}

#[cfg(test)]
impl ScramClient {
    pub fn new(username: &str, password: &str) -> Self {
        let client_nonce: String = (0..24)
            .map(|_| {
                let idx = rand::random::<u8>() % 62;
                match idx {
                    0..=9 => (b'0' + idx) as char,
                    10..=35 => (b'a' + idx - 10) as char,
                    _ => (b'A' + idx - 36) as char,
                }
            })
            .collect();

        Self {
            username: username.to_string(),
            password: password.to_string(),
            client_nonce,
            client_first_bare: String::new(),
            server_first: String::new(),
        }
    }

    /// Generate the client-first-message.
    pub fn client_first(&mut self) -> String {
        self.client_first_bare = format!("n={},r={}", self.username, self.client_nonce);
        format!("n,,{}", self.client_first_bare)
    }

    /// Process server-first-message and produce client-final-message.
    pub fn client_final(&mut self, server_first: &str) -> Result<String, String> {
        self.server_first = server_first.to_string();

        let combined_nonce =
            parse_attribute(server_first, 'r').ok_or("missing r= in server-first")?;
        let salt_b64 = parse_attribute(server_first, 's').ok_or("missing s= in server-first")?;
        let iterations_str =
            parse_attribute(server_first, 'i').ok_or("missing i= in server-first")?;

        let salt = BASE64
            .decode(salt_b64)
            .map_err(|e| format!("bad salt: {}", e))?;
        let iterations: u32 = iterations_str
            .parse()
            .map_err(|e| format!("bad iterations: {}", e))?;

        // Verify server nonce starts with our client nonce
        if !combined_nonce.starts_with(&self.client_nonce) {
            return Err("server nonce doesn't contain client nonce".into());
        }

        // Derive keys
        let salted_password = pbkdf2_hmac_sha256(self.password.as_bytes(), &salt, iterations);
        let client_key = hmac_sha256(&salted_password, b"Client Key");
        let stored_key = sha256(&client_key);

        // Channel binding: "biws" = base64("n,,")
        let client_final_without_proof = format!("c=biws,r={}", combined_nonce);

        // AuthMessage
        let auth_message = format!(
            "{},{},{}",
            self.client_first_bare, self.server_first, client_final_without_proof,
        );

        // ClientSignature = HMAC(StoredKey, AuthMessage)
        let client_signature = hmac_sha256(&stored_key, auth_message.as_bytes());

        // ClientProof = ClientKey XOR ClientSignature
        let client_proof: Vec<u8> = client_key
            .iter()
            .zip(client_signature.iter())
            .map(|(a, b)| a ^ b)
            .collect();

        Ok(format!(
            "{},p={}",
            client_final_without_proof,
            BASE64.encode(&client_proof)
        ))
    }

    /// Verify the server-final-message.
    pub fn verify_server_final(
        &self,
        server_final: &str,
        server_first: &str,
    ) -> Result<(), String> {
        let server_sig_b64 =
            parse_attribute(server_final, 'v').ok_or("missing v= in server-final")?;
        let server_sig = BASE64
            .decode(server_sig_b64)
            .map_err(|e| format!("bad server signature: {}", e))?;

        // Recompute expected server signature
        let combined_nonce = parse_attribute(server_first, 'r').unwrap();
        let salt_b64 = parse_attribute(server_first, 's').unwrap();
        let iterations_str = parse_attribute(server_first, 'i').unwrap();

        let salt = BASE64.decode(salt_b64).unwrap();
        let iterations: u32 = iterations_str.parse().unwrap();

        let salted_password = pbkdf2_hmac_sha256(self.password.as_bytes(), &salt, iterations);
        let server_key = hmac_sha256(&salted_password, b"Server Key");

        let client_final_without_proof = format!("c=biws,r={}", combined_nonce);
        let auth_message = format!(
            "{},{},{}",
            self.client_first_bare, server_first, client_final_without_proof,
        );

        let expected_sig = hmac_sha256(&server_key, auth_message.as_bytes());

        if server_sig == expected_sig {
            Ok(())
        } else {
            Err("server signature mismatch".into())
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_credential_generation() {
        let creds = generate_credentials("testpassword");
        assert_eq!(creds.salt.len(), SALT_LENGTH);
        assert_eq!(creds.iterations, DEFAULT_ITERATIONS);
        assert_eq!(creds.stored_key.len(), 32);
        assert_eq!(creds.server_key.len(), 32);
    }

    #[test]
    fn test_credential_json_roundtrip() {
        let creds = generate_credentials("mypassword");
        let json = credentials_to_json(&creds);
        let parsed = credentials_from_json(&json).expect("should parse");
        assert_eq!(parsed.salt, creds.salt);
        assert_eq!(parsed.iterations, creds.iterations);
        assert_eq!(parsed.stored_key, creds.stored_key);
        assert_eq!(parsed.server_key, creds.server_key);
    }

    #[test]
    fn test_credentials_from_json_rejects_old_format() {
        // Old SHA-256 hex hash (not SCRAM JSON)
        let old_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        assert!(credentials_from_json(old_hash).is_none());
    }

    #[test]
    fn test_full_scram_handshake() {
        let password = "correct_password";
        let creds = generate_credentials(password);

        let mut server = ScramServer::new(&creds);
        let mut client = ScramClient::new("testuser", password);

        // Step 1: client-first
        let client_first = client.client_first();

        // Step 2: server-first
        let server_first = server
            .handle_client_first(&client_first)
            .expect("server should accept client-first");

        // Step 3: client-final
        let client_final = client
            .client_final(&server_first)
            .expect("client should produce client-final");

        // Step 4: server-final (verifies proof)
        let server_final = server
            .handle_client_final(&client_final)
            .expect("server should verify proof");

        // Step 5: client verifies server signature
        client
            .verify_server_final(&server_final, &server_first)
            .expect("client should verify server signature");
    }

    #[test]
    fn test_wrong_password_rejected() {
        let creds = generate_credentials("correct_password");

        let mut server = ScramServer::new(&creds);
        let mut client = ScramClient::new("testuser", "wrong_password");

        let client_first = client.client_first();
        let server_first = server.handle_client_first(&client_first).unwrap();
        let client_final = client.client_final(&server_first).unwrap();

        // Server should reject the proof
        let result = server.handle_client_final(&client_final);
        assert!(matches!(result, Err(ScramError::AuthFailed)));
    }

    #[test]
    fn test_deterministic_with_known_salt() {
        let password = "pencil";
        let salt = b"salt_for_testing";
        let creds = generate_credentials_with_salt(password, salt, 4096);

        // Generate again with same params — should produce identical credentials
        let creds2 = generate_credentials_with_salt(password, salt, 4096);
        assert_eq!(creds.stored_key, creds2.stored_key);
        assert_eq!(creds.server_key, creds2.server_key);
    }

    #[test]
    fn test_parse_attribute() {
        assert_eq!(parse_attribute("r=abc,s=def,i=4096", 'r'), Some("abc"));
        assert_eq!(parse_attribute("r=abc,s=def,i=4096", 's'), Some("def"));
        assert_eq!(parse_attribute("r=abc,s=def,i=4096", 'i'), Some("4096"));
        assert_eq!(parse_attribute("r=abc,s=def,i=4096", 'x'), None);
    }

    #[test]
    fn test_invalid_client_first() {
        let creds = generate_credentials("password");
        let mut server = ScramServer::new(&creds);

        // Missing GS2 header
        let result = server.handle_client_first("garbage");
        assert!(matches!(result, Err(ScramError::Protocol(_))));
    }

    #[test]
    fn test_multiple_handshakes_different_results() {
        // Different salt = different credentials
        let creds1 = generate_credentials("same_password");
        let creds2 = generate_credentials("same_password");
        assert_ne!(creds1.salt, creds2.salt);
        assert_ne!(creds1.stored_key, creds2.stored_key);
    }
}
