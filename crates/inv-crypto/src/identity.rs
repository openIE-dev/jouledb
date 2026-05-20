use ed25519_dalek::{SigningKey, VerifyingKey};
use inv_core::NodeId;
use rand_core_06::OsRng;
use serde::{Deserialize, Serialize};
use zeroize::{Zeroize, ZeroizeOnDrop};

/// An Ed25519 keypair for node identity.
/// The private key is zeroized on drop.
#[derive(Zeroize, ZeroizeOnDrop)]
pub struct NodeKeypair {
    #[zeroize(skip)]
    signing_key: SigningKey,
}

impl NodeKeypair {
    /// Generate a new random Ed25519 keypair.
    pub fn generate() -> Self {
        let signing_key = SigningKey::generate(&mut OsRng);
        Self { signing_key }
    }

    /// Reconstruct from secret key bytes (32 bytes).
    pub fn from_secret_bytes(bytes: &[u8; 32]) -> Self {
        let signing_key = SigningKey::from_bytes(bytes);
        Self { signing_key }
    }

    /// The secret key bytes (32 bytes). Handle with care.
    pub fn secret_bytes(&self) -> &[u8; 32] {
        self.signing_key.as_bytes()
    }

    /// The public key bytes (32 bytes).
    pub fn public_key_bytes(&self) -> [u8; 32] {
        *self.signing_key.verifying_key().as_bytes()
    }

    /// The Ed25519 verifying (public) key.
    pub fn verifying_key(&self) -> VerifyingKey {
        self.signing_key.verifying_key()
    }

    /// Derive the NodeId from this keypair's public key.
    pub fn node_id(&self) -> NodeId {
        NodeId::from_public_key(&self.public_key_bytes())
    }

    /// Sign a message with this keypair.
    pub fn sign(&self, message: &[u8]) -> ed25519_dalek::Signature {
        use ed25519_dalek::Signer;
        self.signing_key.sign(message)
    }

    /// Reference to the inner signing key.
    pub fn signing_key(&self) -> &SigningKey {
        &self.signing_key
    }
}

impl std::fmt::Debug for NodeKeypair {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NodeKeypair")
            .field("node_id", &self.node_id())
            .field("secret", &"[REDACTED]")
            .finish()
    }
}

/// Verify a signature against a public key.
pub fn verify_signature(
    public_key: &VerifyingKey,
    message: &[u8],
    signature: &ed25519_dalek::Signature,
) -> bool {
    use ed25519_dalek::Verifier;
    public_key.verify(message, signature).is_ok()
}

/// A time-limited join token for node enrollment.
/// Signed by the org CA's Ed25519 key.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoinToken {
    /// Organization identifier.
    pub org_id: String,
    /// Unix timestamp when this token was issued.
    pub issued_at: u64,
    /// Unix timestamp when this token expires.
    pub expires_at: u64,
    /// Allowed roles for the joining node.
    pub allowed_roles: Vec<String>,
    /// Ed25519 signature over the token payload.
    #[serde(with = "signature_serde")]
    pub signature: ed25519_dalek::Signature,
}

impl JoinToken {
    /// Create and sign a new join token.
    pub fn create(
        keypair: &NodeKeypair,
        org_id: &str,
        allowed_roles: Vec<String>,
        ttl_seconds: u64,
    ) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let mut token = Self {
            org_id: org_id.to_string(),
            issued_at: now,
            expires_at: now + ttl_seconds,
            allowed_roles,
            signature: ed25519_dalek::Signature::from_bytes(&[0u8; 64]),
        };

        let payload = token.signable_payload();
        token.signature = keypair.sign(&payload);
        token
    }

    /// The bytes that are signed/verified.
    /// Length-prefixed fields prevent delimiter injection (e.g. roles
    /// ["ab","c"] vs ["a","bc"] producing identical payloads).
    fn signable_payload(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(self.org_id.len() as u32).to_be_bytes());
        buf.extend_from_slice(self.org_id.as_bytes());
        buf.extend_from_slice(&self.issued_at.to_be_bytes());
        buf.extend_from_slice(&self.expires_at.to_be_bytes());
        buf.extend_from_slice(&(self.allowed_roles.len() as u32).to_be_bytes());
        for role in &self.allowed_roles {
            buf.extend_from_slice(&(role.len() as u32).to_be_bytes());
            buf.extend_from_slice(role.as_bytes());
        }
        buf
    }

    /// Verify the token signature and check expiry.
    pub fn verify(&self, public_key: &VerifyingKey) -> Result<(), JoinTokenError> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        if now > self.expires_at {
            return Err(JoinTokenError::Expired);
        }

        let payload = self.signable_payload();
        if !verify_signature(public_key, &payload, &self.signature) {
            return Err(JoinTokenError::InvalidSignature);
        }

        Ok(())
    }

    /// Encode as a prefixed base64 string: `inv_join_<base64>`.
    pub fn encode(&self) -> String {
        use base64_engine::Engine;
        let json = serde_json::to_vec(self).expect("JoinToken serializes to JSON");
        let b64 = base64_engine::STANDARD.encode(&json);
        format!("inv_join_{b64}")
    }

    /// Decode from the prefixed base64 string.
    pub fn decode(s: &str) -> Result<Self, JoinTokenError> {
        use base64_engine::Engine;
        let b64 = s
            .strip_prefix("inv_join_")
            .ok_or(JoinTokenError::InvalidFormat)?;
        let json = base64_engine::STANDARD
            .decode(b64)
            .map_err(|_| JoinTokenError::InvalidFormat)?;
        serde_json::from_slice(&json).map_err(|_| JoinTokenError::InvalidFormat)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum JoinTokenError {
    #[error("join token has expired")]
    Expired,
    #[error("invalid token signature")]
    InvalidSignature,
    #[error("invalid token format")]
    InvalidFormat,
}

/// Base64 engine alias (using the standard alphabet).
mod base64_engine {
    pub use base64::Engine;
    pub use base64::engine::general_purpose::STANDARD;
}

/// Serde support for ed25519_dalek::Signature (as hex).
mod signature_serde {
    use ed25519_dalek::Signature;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(sig: &Signature, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&hex::encode(sig.to_bytes()))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Signature, D::Error> {
        let hex_str = String::deserialize(d)?;
        let bytes = hex::decode(&hex_str).map_err(serde::de::Error::custom)?;
        let arr: [u8; 64] = bytes
            .try_into()
            .map_err(|_| serde::de::Error::custom("expected 64-byte signature"))?;
        Ok(Signature::from_bytes(&arr))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keypair_generates_valid_node_id() {
        let kp = NodeKeypair::generate();
        let id = kp.node_id();
        let repr = id.to_string_repr();
        assert!(repr.starts_with("inv_"));
    }

    #[test]
    fn keypair_roundtrip_from_secret() {
        let kp = NodeKeypair::generate();
        let secret = *kp.secret_bytes();
        let restored = NodeKeypair::from_secret_bytes(&secret);
        assert_eq!(kp.node_id(), restored.node_id());
    }

    #[test]
    fn sign_and_verify() {
        let kp = NodeKeypair::generate();
        let msg = b"hello invisible infrastructure";
        let sig = kp.sign(msg);
        assert!(verify_signature(&kp.verifying_key(), msg, &sig));
        assert!(!verify_signature(
            &kp.verifying_key(),
            b"wrong message",
            &sig
        ));
    }

    #[test]
    fn join_token_roundtrip() {
        let ca = NodeKeypair::generate();
        let token = JoinToken::create(&ca, "acme-corp", vec!["backbone".into()], 86400);

        let encoded = token.encode();
        assert!(encoded.starts_with("inv_join_"));

        let decoded = JoinToken::decode(&encoded).unwrap();
        decoded.verify(&ca.verifying_key()).unwrap();
    }

    #[test]
    fn join_token_expired() {
        let ca = NodeKeypair::generate();
        let mut token = JoinToken::create(&ca, "acme-corp", vec![], 0);
        // Force expiry in the past
        token.expires_at = 1;
        let payload = token.signable_payload();
        token.signature = ca.sign(&payload);

        assert!(matches!(
            token.verify(&ca.verifying_key()),
            Err(JoinTokenError::Expired)
        ));
    }

    #[test]
    fn join_token_wrong_key() {
        let ca = NodeKeypair::generate();
        let other = NodeKeypair::generate();
        let token = JoinToken::create(&ca, "acme-corp", vec![], 86400);

        assert!(matches!(
            token.verify(&other.verifying_key()),
            Err(JoinTokenError::InvalidSignature)
        ));
    }

    #[test]
    fn debug_redacts_secret() {
        let kp = NodeKeypair::generate();
        let debug = format!("{kp:?}");
        assert!(debug.contains("[REDACTED]"));
        assert!(!debug.contains(&hex::encode(kp.secret_bytes())));
    }
}
