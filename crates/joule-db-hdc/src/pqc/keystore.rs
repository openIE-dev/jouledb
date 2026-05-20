//! PQC Key Storage and Management
//!
//! Provides secure storage for post-quantum cryptographic keys with:
//! - Key generation and storage
//! - Key retrieval and deletion
//! - Metadata management
//! - Encrypted storage option

use super::common::{SecureZeroingVec, Sha3_256, Shake256};
use super::hybrid::{HybridPublicKey, HybridSecretKey};
use super::ml_dsa::{MlDsa65, MlDsaSigningKey, MlDsaVerificationKey};
use super::ml_kem::{MlKem768, MlKemDecapsulationKey, MlKemEncapsulationKey};
use super::{PqcError, PqcResult};
use std::collections::HashMap;

// ============================================================================
// Key Metadata
// ============================================================================

/// Metadata for a stored key
#[derive(Clone, Debug)]
pub struct KeyMetadata {
    /// Unique key identifier
    pub key_id: String,
    /// Key type (e.g., "ML-KEM-768", "ML-DSA-65", "Hybrid")
    pub key_type: KeyType,
    /// Creation timestamp (Unix time)
    pub created_at: u64,
    /// Optional expiration timestamp
    pub expires_at: Option<u64>,
    /// Optional label/description
    pub label: Option<String>,
    /// Usage count
    pub use_count: u64,
    /// Is key rotated (should not be used for new operations)
    pub rotated: bool,
}

/// Key type enumeration
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum KeyType {
    /// ML-KEM key pair
    MlKem,
    /// ML-DSA key pair
    MlDsa,
    /// Hybrid key pair
    Hybrid,
}

impl std::fmt::Display for KeyType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KeyType::MlKem => write!(f, "ML-KEM-768"),
            KeyType::MlDsa => write!(f, "ML-DSA-65"),
            KeyType::Hybrid => write!(f, "Hybrid"),
        }
    }
}

// ============================================================================
// Stored Key
// ============================================================================

/// A stored key with its data and metadata
#[derive(Clone)]
pub struct StoredKey {
    /// Key metadata
    pub metadata: KeyMetadata,
    /// Key data (encrypted or plaintext)
    data: SecureZeroingVec,
    /// Is data encrypted
    encrypted: bool,
}

impl StoredKey {
    /// Get key data (returns encrypted data if encrypted)
    pub fn data(&self) -> &[u8] {
        self.data.as_slice()
    }

    /// Check if encrypted
    pub fn is_encrypted(&self) -> bool {
        self.encrypted
    }

    /// Decrypt key data with password
    pub fn decrypt(&self, password: &[u8]) -> PqcResult<Vec<u8>> {
        if !self.encrypted {
            return Ok(self.data.as_slice().to_vec());
        }

        // Derive decryption key from password
        let key = Self::derive_key(password, &self.metadata.key_id);

        // Decrypt (XOR with keystream)
        let keystream = Shake256::xof(&key, self.data.len());
        let decrypted: Vec<u8> = self
            .data
            .as_slice()
            .iter()
            .zip(keystream.iter())
            .map(|(d, k)| d ^ k)
            .collect();

        Ok(decrypted)
    }

    /// Encrypt key data with password
    fn encrypt_data(data: &[u8], password: &[u8], key_id: &str) -> Vec<u8> {
        let key = Self::derive_key(password, key_id);
        let keystream = Shake256::xof(&key, data.len());

        data.iter()
            .zip(keystream.iter())
            .map(|(d, k)| d ^ k)
            .collect()
    }

    /// Derive encryption key from password
    fn derive_key(password: &[u8], key_id: &str) -> [u8; 32] {
        let mut input = Vec::new();
        input.extend_from_slice(b"PQCKeyStore-KeyDerivation");
        input.extend_from_slice(password);
        input.extend_from_slice(key_id.as_bytes());
        Sha3_256::hash(&input)
    }
}

// ============================================================================
// Key Store
// ============================================================================

/// PQC Key Store for managing cryptographic keys
pub struct PqcKeyStore {
    /// Stored keys indexed by key_id
    keys: HashMap<String, StoredKey>,
    /// Default encryption password (optional)
    default_password: Option<SecureZeroingVec>,
    /// Counter for generating unique IDs
    id_counter: u64,
}

impl PqcKeyStore {
    /// Create new empty key store
    pub fn new() -> Self {
        Self {
            keys: HashMap::new(),
            default_password: None,
            id_counter: 0,
        }
    }

    /// Create key store with default encryption password
    pub fn with_password(password: &[u8]) -> Self {
        Self {
            keys: HashMap::new(),
            default_password: Some(SecureZeroingVec::from_vec(password.to_vec())),
            id_counter: 0,
        }
    }

    /// Set default encryption password
    pub fn set_password(&mut self, password: &[u8]) {
        self.default_password = Some(SecureZeroingVec::from_vec(password.to_vec()));
    }

    /// Clear default password
    pub fn clear_password(&mut self) {
        self.default_password = None;
    }

    /// Generate unique key ID
    fn generate_id(&mut self, prefix: &str) -> String {
        self.id_counter += 1;
        let random_part = Sha3_256::hash(&self.id_counter.to_le_bytes());
        format!("{}-{}", prefix, hex::encode(&random_part[..8]))
    }

    /// Get current time (simplified - returns counter for determinism in tests)
    fn current_time(&self) -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    // ========================================================================
    // ML-KEM Key Management
    // ========================================================================

    /// Generate and store ML-KEM key pair
    pub fn generate_ml_kem(&mut self, label: Option<&str>) -> PqcResult<String> {
        let (ek, dk) = MlKem768::keygen()?;

        let key_id = self.generate_id("mlkem");

        // Combine public and secret key
        let mut key_data = Vec::new();
        key_data.extend_from_slice(ek.as_bytes());
        key_data.extend_from_slice(dk.as_bytes());

        let encrypted = self.default_password.is_some();
        let final_data = if let Some(ref pwd) = self.default_password {
            StoredKey::encrypt_data(&key_data, pwd.as_slice(), &key_id)
        } else {
            key_data
        };

        let metadata = KeyMetadata {
            key_id: key_id.clone(),
            key_type: KeyType::MlKem,
            created_at: self.current_time(),
            expires_at: None,
            label: label.map(String::from),
            use_count: 0,
            rotated: false,
        };

        self.keys.insert(
            key_id.clone(),
            StoredKey {
                metadata,
                data: SecureZeroingVec::from_vec(final_data),
                encrypted,
            },
        );

        Ok(key_id)
    }

    /// Get ML-KEM encapsulation key (public)
    pub fn get_ml_kem_public(&self, key_id: &str) -> PqcResult<MlKemEncapsulationKey> {
        let stored = self.keys.get(key_id).ok_or(PqcError::InvalidKey)?;

        if stored.metadata.key_type != KeyType::MlKem {
            return Err(PqcError::InvalidKey);
        }

        let data = if stored.encrypted {
            let pwd = self.default_password.as_ref().ok_or(PqcError::InvalidKey)?;
            stored.decrypt(pwd.as_slice())?
        } else {
            stored.data.as_slice().to_vec()
        };

        let pk_size = MlKem768::PARAMS.encapsulation_key_size();
        MlKemEncapsulationKey::from_bytes(&data[..pk_size], MlKem768::PARAMS)
    }

    /// Get ML-KEM decapsulation key (secret)
    pub fn get_ml_kem_secret(&mut self, key_id: &str) -> PqcResult<MlKemDecapsulationKey> {
        let stored = self.keys.get_mut(key_id).ok_or(PqcError::InvalidKey)?;

        if stored.metadata.key_type != KeyType::MlKem {
            return Err(PqcError::InvalidKey);
        }

        stored.metadata.use_count += 1;

        let data = if stored.encrypted {
            let pwd = self.default_password.as_ref().ok_or(PqcError::InvalidKey)?;
            stored.decrypt(pwd.as_slice())?
        } else {
            stored.data.as_slice().to_vec()
        };

        let pk_size = MlKem768::PARAMS.encapsulation_key_size();
        MlKemDecapsulationKey::from_bytes(&data[pk_size..], MlKem768::PARAMS)
    }

    // ========================================================================
    // ML-DSA Key Management
    // ========================================================================

    /// Generate and store ML-DSA key pair
    pub fn generate_ml_dsa(&mut self, label: Option<&str>) -> PqcResult<String> {
        let (vk, sk) = MlDsa65::keygen()?;

        let key_id = self.generate_id("mldsa");

        // Combine verification and signing key
        let mut key_data = Vec::new();
        key_data.extend_from_slice(vk.as_bytes());
        key_data.extend_from_slice(sk.as_bytes());

        let encrypted = self.default_password.is_some();
        let final_data = if let Some(ref pwd) = self.default_password {
            StoredKey::encrypt_data(&key_data, pwd.as_slice(), &key_id)
        } else {
            key_data
        };

        let metadata = KeyMetadata {
            key_id: key_id.clone(),
            key_type: KeyType::MlDsa,
            created_at: self.current_time(),
            expires_at: None,
            label: label.map(String::from),
            use_count: 0,
            rotated: false,
        };

        self.keys.insert(
            key_id.clone(),
            StoredKey {
                metadata,
                data: SecureZeroingVec::from_vec(final_data),
                encrypted,
            },
        );

        Ok(key_id)
    }

    /// Get ML-DSA verification key (public)
    pub fn get_ml_dsa_public(&self, key_id: &str) -> PqcResult<MlDsaVerificationKey> {
        let stored = self.keys.get(key_id).ok_or(PqcError::InvalidKey)?;

        if stored.metadata.key_type != KeyType::MlDsa {
            return Err(PqcError::InvalidKey);
        }

        let data = if stored.encrypted {
            let pwd = self.default_password.as_ref().ok_or(PqcError::InvalidKey)?;
            stored.decrypt(pwd.as_slice())?
        } else {
            stored.data.as_slice().to_vec()
        };

        let vk_size = MlDsa65::PARAMS.verification_key_size();
        MlDsaVerificationKey::from_bytes(&data[..vk_size], MlDsa65::PARAMS)
    }

    /// Get ML-DSA signing key (secret)
    pub fn get_ml_dsa_secret(&mut self, key_id: &str) -> PqcResult<MlDsaSigningKey> {
        let stored = self.keys.get_mut(key_id).ok_or(PqcError::InvalidKey)?;

        if stored.metadata.key_type != KeyType::MlDsa {
            return Err(PqcError::InvalidKey);
        }

        stored.metadata.use_count += 1;

        let data = if stored.encrypted {
            let pwd = self.default_password.as_ref().ok_or(PqcError::InvalidKey)?;
            stored.decrypt(pwd.as_slice())?
        } else {
            stored.data.as_slice().to_vec()
        };

        let vk_size = MlDsa65::PARAMS.verification_key_size();
        MlDsaSigningKey::from_bytes(&data[vk_size..], MlDsa65::PARAMS)
    }

    // ========================================================================
    // Hybrid Key Management
    // ========================================================================

    /// Generate and store hybrid key pair
    pub fn generate_hybrid(&mut self, label: Option<&str>) -> PqcResult<String> {
        let (pk, sk) = super::hybrid::HybridKem::keygen()?;

        let key_id = self.generate_id("hybrid");

        // Combine public and secret key
        let mut key_data = pk.to_bytes();
        key_data.extend(sk.to_bytes());

        let encrypted = self.default_password.is_some();
        let final_data = if let Some(ref pwd) = self.default_password {
            StoredKey::encrypt_data(&key_data, pwd.as_slice(), &key_id)
        } else {
            key_data
        };

        let metadata = KeyMetadata {
            key_id: key_id.clone(),
            key_type: KeyType::Hybrid,
            created_at: self.current_time(),
            expires_at: None,
            label: label.map(String::from),
            use_count: 0,
            rotated: false,
        };

        self.keys.insert(
            key_id.clone(),
            StoredKey {
                metadata,
                data: SecureZeroingVec::from_vec(final_data),
                encrypted,
            },
        );

        Ok(key_id)
    }

    /// Get hybrid public key
    pub fn get_hybrid_public(&self, key_id: &str) -> PqcResult<HybridPublicKey> {
        let stored = self.keys.get(key_id).ok_or(PqcError::InvalidKey)?;

        if stored.metadata.key_type != KeyType::Hybrid {
            return Err(PqcError::InvalidKey);
        }

        let data = if stored.encrypted {
            let pwd = self.default_password.as_ref().ok_or(PqcError::InvalidKey)?;
            stored.decrypt(pwd.as_slice())?
        } else {
            stored.data.as_slice().to_vec()
        };

        let pk_size = HybridPublicKey::size();
        HybridPublicKey::from_bytes(&data[..pk_size])
    }

    /// Get hybrid secret key
    pub fn get_hybrid_secret(&mut self, key_id: &str) -> PqcResult<HybridSecretKey> {
        let stored = self.keys.get_mut(key_id).ok_or(PqcError::InvalidKey)?;

        if stored.metadata.key_type != KeyType::Hybrid {
            return Err(PqcError::InvalidKey);
        }

        stored.metadata.use_count += 1;

        let data = if stored.encrypted {
            let pwd = self.default_password.as_ref().ok_or(PqcError::InvalidKey)?;
            stored.decrypt(pwd.as_slice())?
        } else {
            stored.data.as_slice().to_vec()
        };

        let pk_size = HybridPublicKey::size();
        HybridSecretKey::from_bytes(&data[pk_size..])
    }

    // ========================================================================
    // General Key Management
    // ========================================================================

    /// Get key metadata
    pub fn get_metadata(&self, key_id: &str) -> Option<&KeyMetadata> {
        self.keys.get(key_id).map(|k| &k.metadata)
    }

    /// List all key IDs
    pub fn list_keys(&self) -> Vec<&str> {
        self.keys.keys().map(String::as_str).collect()
    }

    /// List keys by type
    pub fn list_keys_by_type(&self, key_type: KeyType) -> Vec<&str> {
        self.keys
            .iter()
            .filter(|(_, k)| k.metadata.key_type == key_type)
            .map(|(id, _)| id.as_str())
            .collect()
    }

    /// Delete a key
    pub fn delete_key(&mut self, key_id: &str) -> bool {
        self.keys.remove(key_id).is_some()
    }

    /// Mark key as rotated (should not be used for new operations)
    pub fn rotate_key(&mut self, key_id: &str) -> PqcResult<()> {
        let stored = self.keys.get_mut(key_id).ok_or(PqcError::InvalidKey)?;
        stored.metadata.rotated = true;
        Ok(())
    }

    /// Set key expiration
    pub fn set_expiration(&mut self, key_id: &str, expires_at: u64) -> PqcResult<()> {
        let stored = self.keys.get_mut(key_id).ok_or(PqcError::InvalidKey)?;
        stored.metadata.expires_at = Some(expires_at);
        Ok(())
    }

    /// Check if key is expired
    pub fn is_expired(&self, key_id: &str) -> bool {
        if let Some(stored) = self.keys.get(key_id) {
            if let Some(expires_at) = stored.metadata.expires_at {
                return self.current_time() > expires_at;
            }
        }
        false
    }

    /// Get number of stored keys
    pub fn key_count(&self) -> usize {
        self.keys.len()
    }

    /// Clear all keys
    pub fn clear(&mut self) {
        self.keys.clear();
    }
}

impl Default for PqcKeyStore {
    fn default() -> Self {
        Self::new()
    }
}

// Simple hex encoding for key IDs
mod hex {
    pub fn encode(data: &[u8]) -> String {
        data.iter().map(|b| format!("{:02x}", b)).collect()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ml_kem_key_management() {
        let mut store = PqcKeyStore::new();

        let key_id = store
            .generate_ml_kem(Some("test-kem"))
            .expect("keygen failed");

        let metadata = store.get_metadata(&key_id).expect("metadata missing");
        assert_eq!(metadata.key_type, KeyType::MlKem);
        assert_eq!(metadata.label.as_deref(), Some("test-kem"));

        let pk = store.get_ml_kem_public(&key_id).expect("get pk failed");
        let sk = store.get_ml_kem_secret(&key_id).expect("get sk failed");

        // Verify roundtrip
        let (ct, ss_enc) = MlKem768::encapsulate(&pk).expect("encapsulate failed");
        let ss_dec = MlKem768::decapsulate(&sk, &ct).expect("decapsulate failed");
        assert_eq!(ss_enc, ss_dec);

        // Check use count increased
        let metadata = store.get_metadata(&key_id).unwrap();
        assert_eq!(metadata.use_count, 1);
    }

    #[test]
    fn test_ml_dsa_key_management() {
        let mut store = PqcKeyStore::new();

        let key_id = store
            .generate_ml_dsa(Some("test-dsa"))
            .expect("keygen failed");

        let vk = store.get_ml_dsa_public(&key_id).expect("get vk failed");
        let sk = store.get_ml_dsa_secret(&key_id).expect("get sk failed");

        let message = b"Test message";
        let sig = MlDsa65::sign(&sk, message).expect("sign failed");
        assert!(MlDsa65::verify(&vk, message, &sig));
    }

    #[test]
    fn test_hybrid_key_management() {
        let mut store = PqcKeyStore::new();

        let key_id = store
            .generate_hybrid(Some("test-hybrid"))
            .expect("keygen failed");

        let pk = store.get_hybrid_public(&key_id).expect("get pk failed");
        let sk = store.get_hybrid_secret(&key_id).expect("get sk failed");

        let (ct, ss_enc) =
            super::super::hybrid::HybridKem::encapsulate(&pk).expect("encapsulate failed");
        let ss_dec =
            super::super::hybrid::HybridKem::decapsulate(&sk, &ct).expect("decapsulate failed");
        assert_eq!(ss_enc, ss_dec);
    }

    #[test]
    fn test_encrypted_storage() {
        let mut store = PqcKeyStore::with_password(b"test-password");

        let key_id = store.generate_ml_kem(None).expect("keygen failed");

        // Key should be encrypted
        let stored = store.keys.get(&key_id).unwrap();
        assert!(stored.encrypted);

        // Should still work
        let pk = store.get_ml_kem_public(&key_id).expect("get pk failed");
        let sk = store.get_ml_kem_secret(&key_id).expect("get sk failed");

        let (ct, ss_enc) = MlKem768::encapsulate(&pk).expect("encapsulate failed");
        let ss_dec = MlKem768::decapsulate(&sk, &ct).expect("decapsulate failed");
        assert_eq!(ss_enc, ss_dec);
    }

    #[test]
    fn test_key_listing() {
        let mut store = PqcKeyStore::new();

        store.generate_ml_kem(None).expect("keygen failed");
        store.generate_ml_kem(None).expect("keygen failed");
        store.generate_ml_dsa(None).expect("keygen failed");
        store.generate_hybrid(None).expect("keygen failed");

        assert_eq!(store.key_count(), 4);
        assert_eq!(store.list_keys_by_type(KeyType::MlKem).len(), 2);
        assert_eq!(store.list_keys_by_type(KeyType::MlDsa).len(), 1);
        assert_eq!(store.list_keys_by_type(KeyType::Hybrid).len(), 1);
    }

    #[test]
    fn test_key_deletion() {
        let mut store = PqcKeyStore::new();

        let key_id = store.generate_ml_kem(None).expect("keygen failed");
        assert_eq!(store.key_count(), 1);

        assert!(store.delete_key(&key_id));
        assert_eq!(store.key_count(), 0);

        assert!(!store.delete_key(&key_id)); // Already deleted
    }

    #[test]
    fn test_key_rotation() {
        let mut store = PqcKeyStore::new();

        let key_id = store.generate_ml_kem(None).expect("keygen failed");

        store.rotate_key(&key_id).expect("rotate failed");

        let metadata = store.get_metadata(&key_id).unwrap();
        assert!(metadata.rotated);
    }

    #[test]
    fn test_wrong_key_type() {
        let mut store = PqcKeyStore::new();

        let kem_id = store.generate_ml_kem(None).expect("keygen failed");
        let dsa_id = store.generate_ml_dsa(None).expect("keygen failed");

        // Try to get DSA key as KEM
        assert!(store.get_ml_kem_public(&dsa_id).is_err());

        // Try to get KEM key as DSA
        assert!(store.get_ml_dsa_public(&kem_id).is_err());
    }
}
