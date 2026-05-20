use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;

/// A node's unique identity in the mesh.
/// Derived from the SHA-256 hash of its Ed25519 public key.
/// Rendered as base32 with an `inv_` prefix.
#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId([u8; 32]);

impl NodeId {
    /// Create a NodeId from the SHA-256 hash of a public key.
    pub fn from_public_key(public_key: &[u8]) -> Self {
        let hash = Sha256::digest(public_key);
        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(&hash);
        Self(bytes)
    }

    /// Create a NodeId from raw 32-byte hash.
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Get the raw 32-byte hash.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Render as the canonical string: `inv_` + lowercase base32 (no padding).
    pub fn to_string_repr(&self) -> String {
        let encoded = base32::encode(base32::Alphabet::Rfc4648Lower { padding: false }, &self.0);
        format!("inv_{encoded}")
    }

    /// Parse from the canonical string representation.
    pub fn from_string_repr(s: &str) -> Option<Self> {
        let encoded = s.strip_prefix("inv_")?;
        let bytes = base32::decode(base32::Alphabet::Rfc4648Lower { padding: false }, encoded)?;
        if bytes.len() != 32 {
            return None;
        }
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&bytes);
        Some(Self(arr))
    }

    /// First 8 bytes as a u64 for compact wire representation.
    pub fn short_id(&self) -> u64 {
        u64::from_be_bytes(self.0[..8].try_into().unwrap())
    }

    /// Reconstruct a partial NodeId from a short_id (first 8 bytes).
    /// Remaining bytes are zeroed. Used for wire protocol dispatch where
    /// only the short_id is available.
    pub fn from_short_id(short: u64) -> Self {
        let mut bytes = [0u8; 32];
        bytes[..8].copy_from_slice(&short.to_be_bytes());
        Self(bytes)
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_string_repr())
    }
}

impl fmt::Debug for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NodeId({})", self.to_string_repr())
    }
}

/// An organization's unique identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OrgId(String);

impl OrgId {
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for OrgId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A workload's unique identifier (UUID v4).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WorkloadId(uuid::Uuid);

impl WorkloadId {
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4())
    }

    pub fn from_uuid(id: uuid::Uuid) -> Self {
        Self(id)
    }

    pub fn as_uuid(&self) -> &uuid::Uuid {
        &self.0
    }
}

impl Default for WorkloadId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for WorkloadId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A region identifier (e.g., "us-east", "eu-west").
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RegionId(String);

impl RegionId {
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for RegionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_id_roundtrip() {
        let pubkey = b"test-public-key-for-node-identity";
        let id = NodeId::from_public_key(pubkey);
        let repr = id.to_string_repr();

        assert!(repr.starts_with("inv_"));
        let parsed = NodeId::from_string_repr(&repr).expect("should parse");
        assert_eq!(id, parsed);
    }

    #[test]
    fn node_id_display() {
        let id = NodeId::from_bytes([0xAB; 32]);
        let display = format!("{id}");
        assert!(display.starts_with("inv_"));
    }

    #[test]
    fn node_id_short_id() {
        let id = NodeId::from_bytes([0xFF; 32]);
        assert_eq!(id.short_id(), u64::MAX);
    }

    #[test]
    fn org_id_basics() {
        let org = OrgId::new("acme-corp");
        assert_eq!(org.as_str(), "acme-corp");
        assert_eq!(format!("{org}"), "acme-corp");
    }

    #[test]
    fn workload_id_unique() {
        let a = WorkloadId::new();
        let b = WorkloadId::new();
        assert_ne!(a, b);
    }
}
