//! Encryption-at-rest system for JouleDB
//!
//! This module provides comprehensive encryption support for data at rest:
//! - AES-256-GCM encryption for page-level encryption (using audited RustCrypto `aes-gcm` crate)
//! - Key management with master key and data encryption keys (DEK)
//! - Key rotation support
//! - Encrypted storage backend wrapper
//! - Key derivation using HKDF
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    EncryptedBackend<S>                          │
//! │  ┌─────────────┐  ┌─────────────────┐  ┌───────────────────┐   │
//! │  │ KeyManager  │  │  AES-256-GCM    │  │ Inner Backend (S) │   │
//! │  │  - Master   │──│  (aes-gcm crate)│──│ - read_page       │   │
//! │  │  - DEKs     │  │  - Encrypt      │  │ - write_page      │   │
//! │  └─────────────┘  └─────────────────┘  └───────────────────┘   │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Security Model
//!
//! - Master key encrypts data encryption keys (DEKs)
//! - Each page is encrypted with a unique random nonce
//! - Key rotation creates new DEKs without re-encrypting all data
//! - HKDF derives subkeys from the master key for different purposes
//! - Uses audited RustCrypto `aes-gcm` crate (NCC Group 2020 audit)
//! - Constant-time operations and AES-NI hardware acceleration

use aes_gcm::aead::{Aead, KeyInit};
use aes_gcm::{Aes256Gcm, Key, Nonce};

use crate::error::StorageError;
use crate::storage::{Page, PageFlags, PageId, PageType, StorageBackend, StorageStats};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

// ============================================================================
// AES-256 Constants and Types
// ============================================================================

/// AES-256 key size in bytes
const AES_256_KEY_SIZE: usize = 32;

/// GCM nonce size in bytes (96 bits recommended)
const GCM_NONCE_SIZE: usize = 12;

/// GCM tag size in bytes (128 bits)
const GCM_TAG_SIZE: usize = 16;

/// Key ID type for tracking which DEK encrypted data
pub type KeyId = u64;

// ============================================================================
// Encryption Configuration
// ============================================================================

/// Configuration for the encryption system
#[derive(Debug, Clone)]
pub struct EncryptionConfig {
    /// Master key (32 bytes for AES-256)
    pub master_key: [u8; AES_256_KEY_SIZE],
    /// Salt for key derivation
    pub salt: [u8; 32],
    /// Whether to encrypt page metadata
    pub encrypt_metadata: bool,
    /// Maximum number of pages per DEK before rotation
    pub pages_per_dek: u64,
    /// Enable automatic key rotation
    pub auto_rotate: bool,
}

impl EncryptionConfig {
    /// Create a new encryption configuration
    pub fn new(master_key: [u8; AES_256_KEY_SIZE], salt: [u8; 32]) -> Self {
        Self {
            master_key,
            salt,
            encrypt_metadata: true,
            pages_per_dek: 1_000_000,
            auto_rotate: true,
        }
    }

    /// Create configuration with custom settings
    pub fn with_settings(
        master_key: [u8; AES_256_KEY_SIZE],
        salt: [u8; 32],
        encrypt_metadata: bool,
        pages_per_dek: u64,
        auto_rotate: bool,
    ) -> Self {
        Self {
            master_key,
            salt,
            encrypt_metadata,
            pages_per_dek,
            auto_rotate,
        }
    }
}

// ============================================================================
// Data Encryption Key
// ============================================================================

/// A data encryption key with metadata
#[derive(Clone)]
pub struct DataEncryptionKey {
    /// Unique key identifier
    pub id: KeyId,
    /// The actual key material (32 bytes)
    pub key: [u8; AES_256_KEY_SIZE],
    /// Creation timestamp (Unix epoch seconds)
    pub created_at: u64,
    /// Number of pages encrypted with this key
    pub pages_encrypted: u64,
    /// Whether this key is active for new encryptions
    pub active: bool,
    /// Encrypted form of this DEK (encrypted with master key)
    pub encrypted_key: Vec<u8>,
}

impl DataEncryptionKey {
    /// Create a new DEK
    fn new(id: KeyId, key: [u8; AES_256_KEY_SIZE], created_at: u64) -> Self {
        Self {
            id,
            key,
            created_at,
            pages_encrypted: 0,
            active: true,
            encrypted_key: Vec::new(),
        }
    }
}

// ============================================================================
// Key Manager
// ============================================================================

/// Manages the lifecycle of encryption keys
pub struct KeyManager {
    /// Configuration
    config: EncryptionConfig,
    /// AES-256-GCM cipher for master key operations
    master_cipher: Aes256Gcm,
    /// All data encryption keys indexed by ID
    deks: HashMap<KeyId, DataEncryptionKey>,
    /// Currently active DEK ID
    active_dek_id: KeyId,
    /// Next DEK ID to allocate
    next_dek_id: KeyId,
    /// Nonce counter for GCM (used for master key operations only)
    nonce_counter: u64,
}

impl KeyManager {
    /// Create a new key manager with the given configuration
    pub fn new(config: EncryptionConfig) -> Self {
        let master_key = Key::<Aes256Gcm>::from_slice(&config.master_key);
        let master_cipher = Aes256Gcm::new(master_key);

        let mut manager = Self {
            config,
            master_cipher,
            deks: HashMap::new(),
            active_dek_id: 0,
            next_dek_id: 1,
            nonce_counter: 0,
        };

        // Create initial DEK
        manager.create_new_dek();

        manager
    }

    /// Create a new data encryption key
    pub fn create_new_dek(&mut self) -> KeyId {
        let id = self.next_dek_id;
        self.next_dek_id += 1;

        // Derive DEK using HKDF
        let dek_key = self.derive_dek_key(id);

        // Get current timestamp (simplified - in production use proper time source)
        let created_at = self.nonce_counter;

        let mut dek = DataEncryptionKey::new(id, dek_key, created_at);

        // Encrypt the DEK with the master key
        dek.encrypted_key = self.encrypt_dek(&dek_key);

        // Deactivate previous active DEK
        if let Some(old_dek) = self.deks.get_mut(&self.active_dek_id) {
            old_dek.active = false;
        }

        self.deks.insert(id, dek);
        self.active_dek_id = id;

        id
    }

    /// Get the currently active DEK
    pub fn active_dek(&self) -> Option<&DataEncryptionKey> {
        self.deks.get(&self.active_dek_id)
    }

    /// Get a DEK by ID
    pub fn get_dek(&self, id: KeyId) -> Option<&DataEncryptionKey> {
        self.deks.get(&id)
    }

    /// Get mutable DEK by ID
    pub fn get_dek_mut(&mut self, id: KeyId) -> Option<&mut DataEncryptionKey> {
        self.deks.get_mut(&id)
    }

    /// Check if key rotation is needed
    pub fn needs_rotation(&self) -> bool {
        if !self.config.auto_rotate {
            return false;
        }

        self.deks
            .get(&self.active_dek_id)
            .map(|dek| dek.pages_encrypted >= self.config.pages_per_dek)
            .unwrap_or(true)
    }

    /// Rotate keys if needed
    pub fn rotate_if_needed(&mut self) -> Option<KeyId> {
        if self.needs_rotation() {
            Some(self.create_new_dek())
        } else {
            None
        }
    }

    /// Generate a unique nonce for GCM using random bytes
    pub fn generate_nonce(&mut self) -> [u8; GCM_NONCE_SIZE] {
        self.nonce_counter += 1;
        let mut nonce = [0u8; GCM_NONCE_SIZE];
        rand::fill(&mut nonce);
        nonce
    }

    /// Increment pages encrypted counter for a DEK
    pub fn increment_pages(&mut self, dek_id: KeyId) {
        if let Some(dek) = self.deks.get_mut(&dek_id) {
            dek.pages_encrypted += 1;
        }
    }

    /// Get all DEK IDs
    pub fn all_dek_ids(&self) -> Vec<KeyId> {
        self.deks.keys().copied().collect()
    }

    /// Get number of DEKs
    pub fn dek_count(&self) -> usize {
        self.deks.len()
    }

    // Internal: Derive a DEK key using HKDF
    fn derive_dek_key(&self, id: KeyId) -> [u8; AES_256_KEY_SIZE] {
        let info = format!("joule_db_dek-{}", id);
        hkdf_expand(&self.config.master_key, &self.config.salt, info.as_bytes())
    }

    // Internal: Encrypt a DEK with the master key
    fn encrypt_dek(&mut self, dek: &[u8; AES_256_KEY_SIZE]) -> Vec<u8> {
        let nonce_bytes = self.generate_nonce();
        let nonce = Nonce::from_slice(&nonce_bytes);

        let ciphertext = self
            .master_cipher
            .encrypt(nonce, dek.as_ref())
            .expect("master key encryption should not fail");

        // Format: [nonce: 12 bytes][ciphertext + tag]
        let mut result = Vec::with_capacity(GCM_NONCE_SIZE + ciphertext.len());
        result.extend_from_slice(&nonce_bytes);
        result.extend_from_slice(&ciphertext);
        result
    }

    /// Decrypt a DEK with the master key (for recovery)
    pub fn decrypt_dek(&self, encrypted_dek: &[u8]) -> Option<[u8; AES_256_KEY_SIZE]> {
        if encrypted_dek.len() < GCM_NONCE_SIZE + GCM_TAG_SIZE + AES_256_KEY_SIZE {
            return None;
        }

        let nonce = Nonce::from_slice(&encrypted_dek[0..GCM_NONCE_SIZE]);
        let ciphertext = &encrypted_dek[GCM_NONCE_SIZE..];

        let plaintext = self.master_cipher.decrypt(nonce, ciphertext).ok()?;

        if plaintext.len() != AES_256_KEY_SIZE {
            return None;
        }

        let mut key = [0u8; AES_256_KEY_SIZE];
        key.copy_from_slice(&plaintext);
        Some(key)
    }
}

// ============================================================================
// HKDF Key Derivation
// ============================================================================

/// HMAC-SHA256 for HKDF
fn hmac_sha256(key: &[u8], message: &[u8]) -> [u8; 32] {
    let block_size = 64;

    // Prepare key
    let mut padded_key = [0u8; 64];
    if key.len() > block_size {
        let hashed = sha256(key);
        padded_key[..32].copy_from_slice(&hashed);
    } else {
        padded_key[..key.len()].copy_from_slice(key);
    }

    // Inner padding
    let mut ipad = [0x36u8; 64];
    for i in 0..64 {
        ipad[i] ^= padded_key[i];
    }

    // Outer padding
    let mut opad = [0x5cu8; 64];
    for i in 0..64 {
        opad[i] ^= padded_key[i];
    }

    // Inner hash
    let mut inner_data = Vec::with_capacity(64 + message.len());
    inner_data.extend_from_slice(&ipad);
    inner_data.extend_from_slice(message);
    let inner_hash = sha256(&inner_data);

    // Outer hash
    let mut outer_data = Vec::with_capacity(64 + 32);
    outer_data.extend_from_slice(&opad);
    outer_data.extend_from_slice(&inner_hash);
    sha256(&outer_data)
}

/// Simple SHA-256 implementation for HKDF key derivation
fn sha256(data: &[u8]) -> [u8; 32] {
    // Initial hash values
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    // Round constants
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    // Pad message
    let msg_len = data.len();
    let bit_len = (msg_len as u64) * 8;

    // Calculate padded length
    let pad_len = if (msg_len % 64) < 56 {
        56 - (msg_len % 64)
    } else {
        120 - (msg_len % 64)
    };

    let mut padded = Vec::with_capacity(msg_len + pad_len + 8);
    padded.extend_from_slice(data);
    padded.push(0x80);
    padded.resize(msg_len + pad_len, 0);
    padded.extend_from_slice(&bit_len.to_be_bytes());

    // Process blocks
    for chunk in padded.chunks(64) {
        let mut w = [0u32; 64];

        // Copy chunk into first 16 words
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                chunk[i * 4],
                chunk[i * 4 + 1],
                chunk[i * 4 + 2],
                chunk[i * 4 + 3],
            ]);
        }

        // Extend words
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        // Initialize working variables
        let mut a = h[0];
        let mut b = h[1];
        let mut c = h[2];
        let mut d = h[3];
        let mut e = h[4];
        let mut f = h[5];
        let mut g = h[6];
        let mut hh = h[7];

        // Main loop
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    // Produce final hash
    let mut result = [0u8; 32];
    for (i, &val) in h.iter().enumerate() {
        result[i * 4..i * 4 + 4].copy_from_slice(&val.to_be_bytes());
    }
    result
}

/// HKDF-Expand to derive keys
fn hkdf_expand(ikm: &[u8; AES_256_KEY_SIZE], salt: &[u8], info: &[u8]) -> [u8; AES_256_KEY_SIZE] {
    // Extract
    let prk = hmac_sha256(salt, ikm);

    // Expand
    let mut okm = [0u8; AES_256_KEY_SIZE];
    let mut t = Vec::new();

    for i in 0..((AES_256_KEY_SIZE + 31) / 32) {
        let mut input = Vec::with_capacity(t.len() + info.len() + 1);
        input.extend_from_slice(&t);
        input.extend_from_slice(info);
        input.push((i + 1) as u8);

        t = hmac_sha256(&prk, &input).to_vec();

        let start = i * 32;
        let end = std::cmp::min(start + 32, AES_256_KEY_SIZE);
        okm[start..end].copy_from_slice(&t[..(end - start)]);
    }

    okm
}

// ============================================================================
// Encrypted Backend Wrapper
// ============================================================================

/// Encrypted page header stored with encrypted data
#[derive(Clone)]
struct EncryptedPageHeader {
    /// Key ID used for encryption
    key_id: KeyId,
    /// Original page ID
    page_id: PageId,
}

impl EncryptedPageHeader {
    fn encode(&self) -> [u8; 16] {
        let mut buf = [0u8; 16];
        buf[0..8].copy_from_slice(&self.key_id.to_le_bytes());
        buf[8..16].copy_from_slice(&self.page_id.to_le_bytes());
        buf
    }

    fn decode(buf: &[u8]) -> Option<Self> {
        if buf.len() < 16 {
            return None;
        }
        Some(Self {
            key_id: u64::from_le_bytes([
                buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
            ]),
            page_id: u64::from_le_bytes([
                buf[8], buf[9], buf[10], buf[11], buf[12], buf[13], buf[14], buf[15],
            ]),
        })
    }
}

/// Storage backend wrapper that encrypts all pages
pub struct EncryptedBackend<S: StorageBackend> {
    /// Inner storage backend
    inner: S,
    /// Key manager for encryption keys
    key_manager: Arc<RwLock<KeyManager>>,
    /// Cache of AES-256-GCM cipher instances for DEKs
    cipher_cache: RwLock<HashMap<KeyId, Aes256Gcm>>,
}

impl<S: StorageBackend> EncryptedBackend<S> {
    /// Create a new encrypted backend
    pub fn new(inner: S, config: EncryptionConfig) -> Self {
        let key_manager = Arc::new(RwLock::new(KeyManager::new(config)));
        Self {
            inner,
            key_manager,
            cipher_cache: RwLock::new(HashMap::new()),
        }
    }

    /// Get the key manager
    pub fn key_manager(&self) -> &Arc<RwLock<KeyManager>> {
        &self.key_manager
    }

    /// Force key rotation
    pub fn rotate_key(&mut self) -> KeyId {
        let mut manager = self.key_manager.write().unwrap();
        manager.create_new_dek()
    }

    /// Get statistics about encryption
    pub fn encryption_stats(&self) -> EncryptionStats {
        let manager = self.key_manager.read().unwrap();
        EncryptionStats {
            total_deks: manager.dek_count(),
            active_dek_id: manager.active_dek().map(|d| d.id).unwrap_or(0),
            pages_with_active_dek: manager.active_dek().map(|d| d.pages_encrypted).unwrap_or(0),
        }
    }

    /// Get or create AES-256-GCM cipher for a DEK
    fn get_cipher(&self, dek_id: KeyId) -> Option<Aes256Gcm> {
        // Check cache first
        {
            let cache = self.cipher_cache.read().unwrap();
            if let Some(cipher) = cache.get(&dek_id) {
                return Some(cipher.clone());
            }
        }

        // Get key from manager and create cipher
        let manager = self.key_manager.read().unwrap();
        let dek = manager.get_dek(dek_id)?;
        let key = Key::<Aes256Gcm>::from_slice(&dek.key);
        let cipher = Aes256Gcm::new(key);

        // Cache it
        {
            let mut cache = self.cipher_cache.write().unwrap();
            cache.insert(dek_id, cipher.clone());
        }

        Some(cipher)
    }

    /// Encrypt a page's data
    fn encrypt_page_data(&self, page: &Page) -> Result<Vec<u8>, StorageError> {
        let mut manager = self.key_manager.write().unwrap();

        // Check if rotation is needed
        manager.rotate_if_needed();

        let dek = manager
            .active_dek()
            .ok_or_else(|| StorageError::Backend("No active encryption key".to_string()))?;

        let dek_id = dek.id;
        let dek_key = dek.key;
        let nonce_bytes = manager.generate_nonce();

        // Increment page counter
        manager.increment_pages(dek_id);

        // Drop the lock before doing crypto operations
        drop(manager);

        // Create cipher for this DEK
        let key = Key::<Aes256Gcm>::from_slice(&dek_key);
        let cipher = Aes256Gcm::new(key);
        let nonce = Nonce::from_slice(&nonce_bytes);

        // Create encrypted header
        let header = EncryptedPageHeader {
            key_id: dek_id,
            page_id: page.id,
        };

        // Prepare plaintext: page type + flags + padding + data
        let mut plaintext = Vec::with_capacity(4 + page.data.len());
        plaintext.push(page.page_type as u8);
        plaintext.extend_from_slice(&page.flags.bits().to_le_bytes());
        plaintext.push(0); // padding
        plaintext.extend_from_slice(&page.data);

        // Encrypt with AAD (associated authenticated data) being the header
        let aad = header.encode();

        // Use aes-gcm crate for encryption
        let ciphertext_with_tag = cipher
            .encrypt(
                nonce,
                aes_gcm::aead::Payload {
                    msg: &plaintext,
                    aad: &aad,
                },
            )
            .map_err(|_| StorageError::Backend("AES-GCM encryption failed".to_string()))?;

        // Format: header (16 bytes) + nonce (12 bytes) + ciphertext + tag
        let mut result = Vec::with_capacity(16 + GCM_NONCE_SIZE + ciphertext_with_tag.len());
        result.extend_from_slice(&aad);
        result.extend_from_slice(&nonce_bytes);
        result.extend_from_slice(&ciphertext_with_tag);

        Ok(result)
    }

    /// Decrypt page data
    fn decrypt_page_data(
        &self,
        encrypted_data: &[u8],
        expected_page_id: PageId,
    ) -> Result<Page, StorageError> {
        if encrypted_data.len() < 16 + GCM_NONCE_SIZE + GCM_TAG_SIZE {
            return Err(StorageError::Corrupted {
                page_id: expected_page_id,
                reason: "Encrypted data too short".to_string(),
            });
        }

        // Parse header
        let header = EncryptedPageHeader::decode(&encrypted_data[..16]).ok_or_else(|| {
            StorageError::Corrupted {
                page_id: expected_page_id,
                reason: "Invalid encrypted header".to_string(),
            }
        })?;

        // Verify page ID matches
        if header.page_id != expected_page_id {
            return Err(StorageError::Corrupted {
                page_id: expected_page_id,
                reason: format!(
                    "Page ID mismatch: expected {}, got {}",
                    expected_page_id, header.page_id
                ),
            });
        }

        // Get cipher for this DEK
        let cipher = self
            .get_cipher(header.key_id)
            .ok_or_else(|| StorageError::Backend(format!("Unknown key ID: {}", header.key_id)))?;

        // Extract nonce and ciphertext
        let nonce_start = 16;
        let nonce_bytes = &encrypted_data[nonce_start..nonce_start + GCM_NONCE_SIZE];
        let nonce = Nonce::from_slice(nonce_bytes);

        let ciphertext_with_tag = &encrypted_data[nonce_start + GCM_NONCE_SIZE..];
        let aad = &encrypted_data[..16];

        // Decrypt using aes-gcm crate
        let plaintext = cipher
            .decrypt(
                nonce,
                aes_gcm::aead::Payload {
                    msg: ciphertext_with_tag,
                    aad,
                },
            )
            .map_err(|_| StorageError::Corrupted {
                page_id: expected_page_id,
                reason: "Decryption failed - authentication tag mismatch".to_string(),
            })?;

        if plaintext.len() < 4 {
            return Err(StorageError::Corrupted {
                page_id: expected_page_id,
                reason: "Decrypted data too short".to_string(),
            });
        }

        // Parse decrypted content
        let page_type = PageType::try_from(plaintext[0]).map_err(|_| StorageError::Corrupted {
            page_id: expected_page_id,
            reason: "Invalid page type".to_string(),
        })?;

        let flags = PageFlags::from_bits_truncate(u16::from_le_bytes([plaintext[1], plaintext[2]]));
        let data = plaintext[4..].to_vec();

        Ok(Page {
            id: expected_page_id,
            page_type,
            flags,
            data,
        })
    }
}

impl<S: StorageBackend> StorageBackend for EncryptedBackend<S> {
    fn read_page(&self, page_id: PageId) -> Result<Option<Page>, StorageError> {
        let encrypted_page = self.inner.read_page(page_id)?;

        match encrypted_page {
            Some(page) => {
                let decrypted = self.decrypt_page_data(&page.data, page_id)?;
                Ok(Some(decrypted))
            }
            None => Ok(None),
        }
    }

    fn write_page(&mut self, page: Page) -> Result<(), StorageError> {
        let page_id = page.id;
        let encrypted_data = self.encrypt_page_data(&page)?;

        // Store as a generic page with encrypted data
        let encrypted_page = Page {
            id: page_id,
            page_type: PageType::Metadata, // All encrypted pages look like metadata externally
            flags: PageFlags::from_bits_truncate(PageFlags::ENCRYPTED),
            data: encrypted_data,
        };

        self.inner.write_page(encrypted_page)
    }

    fn allocate_page(&mut self) -> Result<PageId, StorageError> {
        self.inner.allocate_page()
    }

    fn free_page(&mut self, page_id: PageId) -> Result<(), StorageError> {
        self.inner.free_page(page_id)
    }

    fn sync(&mut self) -> Result<(), StorageError> {
        self.inner.sync()
    }

    fn page_size(&self) -> usize {
        // Account for encryption overhead
        let overhead = 16 + GCM_NONCE_SIZE + GCM_TAG_SIZE + 4;
        self.inner.page_size().saturating_sub(overhead)
    }

    fn stats(&self) -> StorageStats {
        self.inner.stats()
    }
}

/// Statistics about encryption operations
#[derive(Debug, Clone)]
pub struct EncryptionStats {
    /// Total number of DEKs
    pub total_deks: usize,
    /// Currently active DEK ID
    pub active_dek_id: KeyId,
    /// Pages encrypted with active DEK
    pub pages_with_active_dek: u64,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::memory::MemoryBackend;

    fn test_key() -> [u8; 32] {
        [
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
            0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b,
            0x1c, 0x1d, 0x1e, 0x1f,
        ]
    }

    fn test_salt() -> [u8; 32] {
        [
            0xde, 0xad, 0xbe, 0xef, 0xca, 0xfe, 0xba, 0xbe, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab,
            0xcd, 0xef, 0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10, 0x00, 0x11, 0x22, 0x33,
            0x44, 0x55, 0x66, 0x77,
        ]
    }

    #[test]
    fn test_aes_gcm_encrypt_decrypt() {
        let key_bytes = test_key();
        let key = Key::<Aes256Gcm>::from_slice(&key_bytes);
        let cipher = Aes256Gcm::new(key);
        let nonce_bytes: [u8; 12] = [
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c,
        ];
        let nonce = Nonce::from_slice(&nonce_bytes);

        let plaintext = b"Hello, World! This is a test of AES-GCM encryption.";
        let aad = b"additional authenticated data";

        let ciphertext = cipher
            .encrypt(
                nonce,
                aes_gcm::aead::Payload {
                    msg: plaintext.as_ref(),
                    aad: aad.as_ref(),
                },
            )
            .expect("encryption failed");

        // Ciphertext should include tag (16 bytes longer)
        assert_eq!(ciphertext.len(), plaintext.len() + GCM_TAG_SIZE);

        // Decrypt
        let decrypted = cipher
            .decrypt(
                nonce,
                aes_gcm::aead::Payload {
                    msg: ciphertext.as_ref(),
                    aad: aad.as_ref(),
                },
            )
            .expect("decryption failed");

        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_aes_gcm_tamper_detection() {
        let key_bytes = test_key();
        let key = Key::<Aes256Gcm>::from_slice(&key_bytes);
        let cipher = Aes256Gcm::new(key);
        let nonce_bytes: [u8; 12] = [
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c,
        ];
        let nonce = Nonce::from_slice(&nonce_bytes);

        let plaintext = b"Secret message";
        let aad = b"header";

        let mut ciphertext = cipher
            .encrypt(
                nonce,
                aes_gcm::aead::Payload {
                    msg: plaintext.as_ref(),
                    aad: aad.as_ref(),
                },
            )
            .expect("encryption failed");

        // Tamper with ciphertext
        if ciphertext.len() > 3 {
            ciphertext[3] ^= 0xff;
        }

        let result = cipher.decrypt(
            nonce,
            aes_gcm::aead::Payload {
                msg: ciphertext.as_ref(),
                aad: aad.as_ref(),
            },
        );
        assert!(
            result.is_err(),
            "Tampered ciphertext should fail authentication"
        );
    }

    #[test]
    fn test_key_manager_dek_creation() {
        let config = EncryptionConfig::new(test_key(), test_salt());
        let mut manager = KeyManager::new(config);

        assert!(manager.active_dek().is_some());
        let first_id = manager.active_dek().unwrap().id;

        let new_id = manager.create_new_dek();
        assert_ne!(first_id, new_id);
        assert_eq!(manager.active_dek().unwrap().id, new_id);
        assert_eq!(manager.dek_count(), 2);
    }

    #[test]
    fn test_key_manager_rotation() {
        let config = EncryptionConfig::with_settings(
            test_key(),
            test_salt(),
            true,
            10, // Rotate after 10 pages
            true,
        );
        let mut manager = KeyManager::new(config);

        let initial_id = manager.active_dek().unwrap().id;

        // Simulate encrypting 10 pages
        for _ in 0..10 {
            manager.increment_pages(initial_id);
        }

        assert!(manager.needs_rotation());

        let new_id = manager.rotate_if_needed();
        assert!(new_id.is_some());
        assert_ne!(new_id.unwrap(), initial_id);
    }

    #[test]
    fn test_hkdf_derivation() {
        let ikm = test_key();
        let salt = test_salt();

        let key1 = hkdf_expand(&ikm, &salt, b"purpose1");
        let key2 = hkdf_expand(&ikm, &salt, b"purpose2");

        // Different info should produce different keys
        assert_ne!(key1, key2);

        // Same inputs should produce same output
        let key1_again = hkdf_expand(&ikm, &salt, b"purpose1");
        assert_eq!(key1, key1_again);
    }

    #[test]
    fn test_encrypted_backend_roundtrip() {
        let inner = MemoryBackend::new();
        let config = EncryptionConfig::new(test_key(), test_salt());
        let mut backend = EncryptedBackend::new(inner, config);

        // Allocate and write a page
        let page_id = backend.allocate_page().unwrap();
        let original_data = b"This is sensitive data that should be encrypted".to_vec();
        let page = Page::with_data(page_id, PageType::BTreeLeaf, original_data.clone());

        backend.write_page(page).unwrap();

        // Read it back
        let read_page = backend.read_page(page_id).unwrap().unwrap();

        assert_eq!(read_page.id, page_id);
        assert_eq!(read_page.page_type, PageType::BTreeLeaf);
        assert_eq!(read_page.data, original_data);
    }

    #[test]
    fn test_encrypted_backend_key_rotation() {
        let inner = MemoryBackend::new();
        let config = EncryptionConfig::with_settings(
            test_key(),
            test_salt(),
            true,
            5, // Rotate after 5 pages
            true,
        );
        let mut backend = EncryptedBackend::new(inner, config);

        let initial_stats = backend.encryption_stats();
        let initial_dek_id = initial_stats.active_dek_id;

        // Write enough pages to trigger rotation
        for i in 0..10 {
            let page_id = backend.allocate_page().unwrap();
            let page = Page::with_data(page_id, PageType::BTreeLeaf, vec![i as u8; 100]);
            backend.write_page(page).unwrap();
        }

        let final_stats = backend.encryption_stats();

        // Should have rotated at least once
        assert!(final_stats.total_deks > 1);
        assert_ne!(final_stats.active_dek_id, initial_dek_id);

        // All pages should still be readable
        for page_id in 1..=10 {
            let page = backend.read_page(page_id).unwrap();
            assert!(page.is_some());
        }
    }

    #[test]
    fn test_sha256_known_vector() {
        // Test vector: SHA-256("abc")
        let input = b"abc";
        let expected = [
            0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde, 0x5d, 0xae,
            0x22, 0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61,
            0xf2, 0x00, 0x15, 0xad,
        ];

        let result = sha256(input);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_multiple_pages_different_data() {
        let inner = MemoryBackend::new();
        let config = EncryptionConfig::new(test_key(), test_salt());
        let mut backend = EncryptedBackend::new(inner, config);

        // Write multiple pages with different data
        let test_data: Vec<Vec<u8>> = vec![
            b"First page content".to_vec(),
            b"Second page with different data".to_vec(),
            b"Third page - even more unique content here!".to_vec(),
            vec![0u8; 1000],    // Large page with zeros
            (0..255).collect(), // Page with all byte values
        ];

        let mut page_ids = Vec::new();
        for data in &test_data {
            let page_id = backend.allocate_page().unwrap();
            page_ids.push(page_id);
            let page = Page::with_data(page_id, PageType::BTreeLeaf, data.clone());
            backend.write_page(page).unwrap();
        }

        // Verify all pages
        for (i, page_id) in page_ids.iter().enumerate() {
            let page = backend.read_page(*page_id).unwrap().unwrap();
            assert_eq!(page.data, test_data[i], "Page {} data mismatch", i);
        }
    }

    #[test]
    fn test_nonce_uniqueness() {
        let config = EncryptionConfig::new(test_key(), test_salt());
        let mut manager = KeyManager::new(config);

        let mut nonces = std::collections::HashSet::new();

        // Generate many nonces and ensure uniqueness
        for _ in 0..1000 {
            let nonce = manager.generate_nonce();
            assert!(nonces.insert(nonce), "Duplicate nonce generated!");
        }
    }

    #[test]
    fn test_dek_encrypt_decrypt() {
        let config = EncryptionConfig::new(test_key(), test_salt());
        let mut manager = KeyManager::new(config);

        // Get the active DEK's encrypted form
        let dek = manager.active_dek().unwrap();
        let original_key = dek.key;
        let encrypted_dek = dek.encrypted_key.clone();

        // Decrypt should recover the original key
        let decrypted = manager.decrypt_dek(&encrypted_dek);
        assert!(decrypted.is_some(), "DEK decryption failed");
        assert_eq!(decrypted.unwrap(), original_key);
    }

    #[test]
    fn test_different_keys_produce_different_ciphertext() {
        let key1_bytes = [1u8; 32];
        let key2_bytes = [2u8; 32];
        let key1 = Key::<Aes256Gcm>::from_slice(&key1_bytes);
        let key2 = Key::<Aes256Gcm>::from_slice(&key2_bytes);
        let cipher1 = Aes256Gcm::new(key1);
        let cipher2 = Aes256Gcm::new(key2);
        let nonce = Nonce::from_slice(&[0u8; 12]);

        let plaintext = b"same data";

        let ct1 = cipher1.encrypt(nonce, plaintext.as_ref()).unwrap();
        let ct2 = cipher2.encrypt(nonce, plaintext.as_ref()).unwrap();

        // Different keys should produce different ciphertext
        assert_ne!(ct1, ct2);

        // Each cipher should only decrypt its own ciphertext
        assert!(cipher1.decrypt(nonce, ct2.as_ref()).is_err());
        assert!(cipher2.decrypt(nonce, ct1.as_ref()).is_err());
    }
}
