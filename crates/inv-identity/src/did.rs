//! DID Documents using the `did:webvh` method.
//!
//! Provides decentralized identifier documents for mesh nodes, a local
//! resolver, and helpers to create node-specific DIDs.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::IdentityError;

/// A DID Document conforming to the W3C DID Core specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DidDocument {
    /// The DID itself (e.g., `did:webvh:example.com:node123`).
    pub id: String,
    /// The DID of the controller of this document.
    pub controller: String,
    /// Verification methods (public keys) associated with this DID.
    pub verification_methods: Vec<VerificationMethod>,
    /// IDs of verification methods used for authentication.
    pub authentication: Vec<String>,
    /// Services exposed by this DID subject.
    pub services: Vec<DidService>,
    /// When the document was created.
    pub created_at: DateTime<Utc>,
    /// When the document was last updated.
    pub updated_at: DateTime<Utc>,
}

/// A verification method (public key) within a DID Document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationMethod {
    /// Unique identifier for this verification method (e.g., `did:webvh:...:node123#key-1`).
    pub id: String,
    /// The type of verification method.
    pub method_type: VerificationMethodType,
    /// The DID of the controller of this key.
    pub controller: String,
    /// The raw public key bytes.
    pub public_key_bytes: Vec<u8>,
}

/// Types of verification methods supported.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VerificationMethodType {
    /// Ed25519 verification key (W3C 2020 suite).
    Ed25519VerificationKey2020,
    /// ML-DSA-65 post-quantum signature verification key.
    MlDsa65VerificationKey,
    /// X25519 key agreement key.
    X25519KeyAgreement,
    /// Hybrid classical + post-quantum signature key.
    HybridPqSig,
}

/// A service endpoint in a DID Document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DidService {
    /// Unique identifier for this service.
    pub id: String,
    /// The type of service (e.g., "MeshEndpoint", "CredentialService").
    pub service_type: String,
    /// The service endpoint URL or address.
    pub endpoint: String,
}

/// Trait for resolving DIDs to their documents.
pub trait DidResolver {
    /// Resolve a DID string to its document.
    fn resolve(&self, did: &str) -> Result<DidDocument, IdentityError>;
}

/// A local in-memory DID resolver.
#[derive(Debug, Clone, Default)]
pub struct LocalDidResolver {
    documents: HashMap<String, DidDocument>,
}

impl LocalDidResolver {
    /// Create a new empty resolver.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a DID document.
    pub fn register(&mut self, doc: DidDocument) {
        self.documents.insert(doc.id.clone(), doc);
    }

    /// List all registered DIDs.
    pub fn list_dids(&self) -> Vec<&str> {
        self.documents.keys().map(|s| s.as_str()).collect()
    }
}

impl DidResolver for LocalDidResolver {
    fn resolve(&self, did: &str) -> Result<DidDocument, IdentityError> {
        self.documents
            .get(did)
            .cloned()
            .ok_or_else(|| IdentityError::DidResolution(format!("DID not found: {did}")))
    }
}

/// Create a DID document for a mesh node.
///
/// Generates a `did:webvh:{domain}:{node_id}` document with a single
/// Ed25519 verification method and a mesh endpoint service.
pub fn create_node_did(domain: &str, node_id: &str, public_key: &[u8]) -> DidDocument {
    let did = format!("did:webvh:{domain}:{node_id}");
    let key_id = format!("{did}#key-1");
    let now = Utc::now();

    DidDocument {
        id: did.clone(),
        controller: did.clone(),
        verification_methods: vec![VerificationMethod {
            id: key_id.clone(),
            method_type: VerificationMethodType::Ed25519VerificationKey2020,
            controller: did.clone(),
            public_key_bytes: public_key.to_vec(),
        }],
        authentication: vec![key_id],
        services: vec![DidService {
            id: format!("{did}#mesh"),
            service_type: "MeshEndpoint".into(),
            endpoint: format!("https://{domain}/mesh/{node_id}"),
        }],
        created_at: now,
        updated_at: now,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_node_did_structure() {
        let doc = create_node_did("example.com", "node123", &[0xAB; 32]);
        assert_eq!(doc.id, "did:webvh:example.com:node123");
        assert_eq!(doc.controller, doc.id);
        assert_eq!(doc.verification_methods.len(), 1);
        assert_eq!(
            doc.verification_methods[0].method_type,
            VerificationMethodType::Ed25519VerificationKey2020
        );
        assert_eq!(doc.verification_methods[0].public_key_bytes, vec![0xAB; 32]);
        assert_eq!(doc.authentication.len(), 1);
        assert_eq!(doc.services.len(), 1);
        assert_eq!(doc.services[0].service_type, "MeshEndpoint");
    }

    #[test]
    fn did_format() {
        let doc = create_node_did("invisible.dev", "node-abc", &[1, 2, 3]);
        assert!(doc.id.starts_with("did:webvh:invisible.dev:"));
        assert!(doc.services[0].endpoint.contains("invisible.dev"));
    }

    #[test]
    fn local_resolver_register_and_resolve() {
        let mut resolver = LocalDidResolver::new();
        let doc = create_node_did("example.com", "node1", &[0x01; 32]);
        resolver.register(doc.clone());
        let resolved = resolver.resolve("did:webvh:example.com:node1").unwrap();
        assert_eq!(resolved.id, doc.id);
    }

    #[test]
    fn local_resolver_not_found() {
        let resolver = LocalDidResolver::new();
        let err = resolver
            .resolve("did:webvh:example.com:missing")
            .unwrap_err();
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn local_resolver_list_dids() {
        let mut resolver = LocalDidResolver::new();
        resolver.register(create_node_did("a.com", "n1", &[1]));
        resolver.register(create_node_did("a.com", "n2", &[2]));
        let dids = resolver.list_dids();
        assert_eq!(dids.len(), 2);
    }

    #[test]
    fn verification_method_types() {
        let types = [
            VerificationMethodType::Ed25519VerificationKey2020,
            VerificationMethodType::MlDsa65VerificationKey,
            VerificationMethodType::X25519KeyAgreement,
            VerificationMethodType::HybridPqSig,
        ];
        for t in &types {
            let json = serde_json::to_string(t).unwrap();
            let parsed: VerificationMethodType = serde_json::from_str(&json).unwrap();
            assert_eq!(*t, parsed);
        }
    }

    #[test]
    fn did_document_serialization_roundtrip() {
        let doc = create_node_did("test.io", "node42", &[0xFF; 32]);
        let json = serde_json::to_string(&doc).unwrap();
        let parsed: DidDocument = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, doc.id);
        assert_eq!(parsed.verification_methods.len(), 1);
        assert_eq!(parsed.services.len(), 1);
    }

    #[test]
    fn multiple_verification_methods() {
        let mut doc = create_node_did("test.io", "node1", &[1; 32]);
        doc.verification_methods.push(VerificationMethod {
            id: format!("{}#key-2", doc.id),
            method_type: VerificationMethodType::MlDsa65VerificationKey,
            controller: doc.id.clone(),
            public_key_bytes: vec![2; 1952],
        });
        assert_eq!(doc.verification_methods.len(), 2);
    }

    #[test]
    fn did_service_structure() {
        let service = DidService {
            id: "did:webvh:test.io:n1#api".into(),
            service_type: "RestApi".into(),
            endpoint: "https://test.io/api/v1".into(),
        };
        let json = serde_json::to_string(&service).unwrap();
        let parsed: DidService = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.service_type, "RestApi");
    }

    #[test]
    fn resolver_overwrite() {
        let mut resolver = LocalDidResolver::new();
        let doc1 = create_node_did("a.com", "n1", &[1; 32]);
        let mut doc2 = create_node_did("a.com", "n1", &[2; 32]);
        doc2.controller = "updated-controller".into();
        resolver.register(doc1);
        resolver.register(doc2);
        let resolved = resolver.resolve("did:webvh:a.com:n1").unwrap();
        assert_eq!(resolved.controller, "updated-controller");
    }

    #[test]
    fn create_node_did_timestamps() {
        let doc = create_node_did("test.io", "n1", &[1]);
        assert_eq!(doc.created_at, doc.updated_at);
    }

    #[test]
    fn empty_public_key() {
        let doc = create_node_did("test.io", "n1", &[]);
        assert!(doc.verification_methods[0].public_key_bytes.is_empty());
    }
}
