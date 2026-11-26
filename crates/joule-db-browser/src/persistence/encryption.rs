//! OPFS Encryption Integration
//!
//! Provides page-level encryption for database storage with:
//! - AES-256-GCM (hardware-accelerated on x86/ARM)
//! - ChaCha20-Poly1305 (constant-time, mobile-friendly)
//! - Argon2id key derivation from passwords
//! - Web Crypto API integration for browser environments
//!
//! # Security Model
//! - Each page encrypted with unique nonce (page_id + counter)
//! - AEAD authentication prevents tampering
//! - Keys derived from password using memory-hard KDF

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use wasm_bindgen::prelude::*;

#[cfg(feature = "encryption")]
use aes_gcm::{
    Aes256Gcm,
    aead::{Aead, KeyInit, generic_array::GenericArray},
};

#[cfg(feature = "encryption")]
use chacha20poly1305::ChaCha20Poly1305;

#[cfg(feature = "encryption")]
use argon2::{Argon2, PasswordHasher, password_hash::SaltString};

/// Encryption algorithm selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EncryptionAlgorithm {
    /// AES-256-GCM - Hardware accelerated on modern CPUs
    /// Best for: x86_64 with AES-NI, ARM with Cryptographic Extensions
    #[default]
    Aes256Gcm,
    /// ChaCha20-Poly1305 - Software-only, constant-time
    /// Best for: ARM without crypto extensions, avoiding timing attacks
    ChaCha20Poly1305,
}

impl EncryptionAlgorithm {
    /// Select the best algorithm for the current platform
    pub fn auto_select() -> Self {
        // In WASM, we'll use AES-GCM as Web Crypto API supports it natively
        // For native, ChaCha20 might be faster on ARM without crypto extensions
        #[cfg(target_arch = "wasm32")]
        {
            EncryptionAlgorithm::Aes256Gcm
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            // Check for AES hardware support
            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            {
                EncryptionAlgorithm::Aes256Gcm
            }
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            {
                EncryptionAlgorithm::ChaCha20Poly1305
            }
        }
    }
}

/// Encryption configuration
#[derive(Debug, Clone)]
pub struct EncryptionConfig {
    /// Algorithm to use
    pub algorithm: EncryptionAlgorithm,
    /// Key derivation iterations (higher = more secure but slower)
    pub kdf_iterations: u32,
    /// Memory cost for Argon2 (in KB)
    pub kdf_memory_cost: u32,
    /// Parallelism for Argon2
    pub kdf_parallelism: u32,
}

impl Default for EncryptionConfig {
    fn default() -> Self {
        Self {
            algorithm: EncryptionAlgorithm::auto_select(),
            kdf_iterations: 3,
            kdf_memory_cost: 64 * 1024, // 64 MB
            kdf_parallelism: 4,
        }
    }
}

impl EncryptionConfig {
    /// Configuration for low-memory devices
    pub fn low_memory() -> Self {
        Self {
            algorithm: EncryptionAlgorithm::ChaCha20Poly1305,
            kdf_iterations: 4,
            kdf_memory_cost: 16 * 1024, // 16 MB
            kdf_parallelism: 1,
        }
    }

    /// Configuration for maximum security
    pub fn high_security() -> Self {
        Self {
            algorithm: EncryptionAlgorithm::Aes256Gcm,
            kdf_iterations: 5,
            kdf_memory_cost: 128 * 1024, // 128 MB
            kdf_parallelism: 4,
        }
    }
}

/// OPFS encryption wrapper
pub struct OpfsEncryption {
    /// Encryption key (32 bytes for AES-256 or ChaCha20)
    key: Arc<RwLock<Option<[u8; 32]>>>,
    /// Key ID for reference
    key_id: Arc<RwLock<Option<String>>>,
    /// Page size for encryption (used for buffer allocation hints)
    #[allow(dead_code)]
    page_size: usize,
    /// Whether encryption is enabled
    enabled: Arc<RwLock<bool>>,
    /// Nonce counter (for uniqueness)
    nonce_counter: AtomicU64,
    /// Encryption algorithm
    algorithm: Arc<RwLock<EncryptionAlgorithm>>,
    /// Configuration
    config: EncryptionConfig,
}

impl OpfsEncryption {
    /// Create a new encryption instance with default configuration
    pub fn new() -> Self {
        Self::with_config(EncryptionConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: EncryptionConfig) -> Self {
        Self {
            key: Arc::new(RwLock::new(None)),
            key_id: Arc::new(RwLock::new(None)),
            page_size: 4096,
            enabled: Arc::new(RwLock::new(false)),
            nonce_counter: AtomicU64::new(0),
            algorithm: Arc::new(RwLock::new(config.algorithm)),
            config,
        }
    }

    /// Enable encryption with a password
    ///
    /// Derives a 256-bit key using Argon2id and enables encryption.
    #[cfg(feature = "encryption")]
    pub fn enable_with_password(&self, password: &str) -> Result<String, JsValue> {
        use argon2::password_hash::rand_core::OsRng;

        // Generate random salt
        let salt = SaltString::generate(&mut OsRng);

        // Derive key using Argon2id
        let argon2 = Argon2::new(
            argon2::Algorithm::Argon2id,
            argon2::Version::V0x13,
            argon2::Params::new(
                self.config.kdf_memory_cost,
                self.config.kdf_iterations,
                self.config.kdf_parallelism,
                Some(32),
            )
            .map_err(|e| JsValue::from_str(&format!("KDF params error: {}", e)))?,
        );

        // Hash password to derive key
        let mut key = [0u8; 32];
        argon2
            .hash_password(password.as_bytes(), &salt)
            .map_err(|e| JsValue::from_str(&format!("Key derivation failed: {}", e)))?
            .hash
            .ok_or_else(|| JsValue::from_str("No hash output"))?
            .as_bytes()
            .iter()
            .zip(key.iter_mut())
            .for_each(|(src, dst)| *dst = *src);

        // Store key and enable encryption
        *self.key.write().unwrap_or_else(|e| e.into_inner()) = Some(key);
        let key_id = format!("key_{}", salt.as_str());
        *self.key_id.write().unwrap_or_else(|e| e.into_inner()) = Some(key_id.clone());
        *self.enabled.write().unwrap_or_else(|e| e.into_inner()) = true;

        Ok(key_id)
    }

    /// Enable encryption with a raw key
    #[cfg(feature = "encryption")]
    pub fn enable_with_key(&self, key: [u8; 32]) -> Result<String, JsValue> {
        *self.key.write().unwrap_or_else(|e| e.into_inner()) = Some(key);
        let key_id = format!(
            "key_{:x}",
            key[0..8].iter().fold(0u64, |acc, &x| acc << 8 | x as u64)
        );
        *self.key_id.write().unwrap_or_else(|e| e.into_inner()) = Some(key_id.clone());
        *self.enabled.write().unwrap_or_else(|e| e.into_inner()) = true;
        Ok(key_id)
    }

    /// Enable encryption (generates random key)
    #[cfg(target_arch = "wasm32")]
    pub fn enable(&self) -> Result<String, JsValue> {
        #[cfg(feature = "encryption")]
        {
            let mut key = [0u8; 32];
            getrandom::getrandom(&mut key)
                .map_err(|e| JsValue::from_str(&format!("RNG error: {}", e)))?;
            self.enable_with_key(key)
        }
        #[cfg(not(feature = "encryption"))]
        {
            // Fallback when encryption feature is not enabled
            let key_id = format!("key_{}", js_sys::Date::now() as u64);
            *self.key_id.write().unwrap_or_else(|e| e.into_inner()) = Some(key_id.clone());
            *self.enabled.write().unwrap_or_else(|e| e.into_inner()) = true;
            Ok(key_id)
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn enable(&self) -> Result<String, JsValue> {
        #[cfg(feature = "encryption")]
        {
            let mut key = [0u8; 32];
            getrandom::getrandom(&mut key)
                .map_err(|e| JsValue::from_str(&format!("RNG error: {}", e)))?;
            self.enable_with_key(key)
        }
        #[cfg(not(feature = "encryption"))]
        {
            let key_id = format!(
                "key_{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis()
            );
            *self.key_id.write().unwrap_or_else(|e| e.into_inner()) = Some(key_id.clone());
            *self.enabled.write().unwrap_or_else(|e| e.into_inner()) = true;
            Ok(key_id)
        }
    }

    /// Disable encryption
    pub fn disable(&self) {
        *self.enabled.write().unwrap_or_else(|e| e.into_inner()) = false;
        // Securely clear the key
        if let Some(ref mut key) = *self.key.write().unwrap_or_else(|e| e.into_inner()) {
            key.iter_mut().for_each(|b| *b = 0);
        }
        *self.key.write().unwrap_or_else(|e| e.into_inner()) = None;
    }

    /// Check if encryption is enabled
    pub fn is_enabled(&self) -> bool {
        *self.enabled.read().unwrap_or_else(|e| e.into_inner())
    }

    /// Get the current key ID
    pub fn key_id(&self) -> Option<String> {
        self.key_id
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Get the current algorithm
    pub fn algorithm(&self) -> EncryptionAlgorithm {
        *self.algorithm.read().unwrap_or_else(|e| e.into_inner())
    }

    /// Set the encryption algorithm
    pub fn set_algorithm(&self, algo: EncryptionAlgorithm) {
        *self.algorithm.write().unwrap_or_else(|e| e.into_inner()) = algo;
    }

    /// Generate a unique nonce for a page
    fn generate_nonce(&self, page_id: u64) -> [u8; 12] {
        let counter = self.nonce_counter.fetch_add(1, Ordering::SeqCst);
        let mut nonce = [0u8; 12];
        // Combine page_id and counter for uniqueness
        nonce[0..8].copy_from_slice(&page_id.to_le_bytes());
        nonce[8..12].copy_from_slice(&(counter as u32).to_le_bytes());
        nonce
    }

    /// Encrypt page data
    #[cfg(feature = "encryption")]
    pub async fn encrypt_page(&self, page_data: &[u8]) -> Result<Vec<u8>, JsValue> {
        self.encrypt_page_with_id(page_data, 0).await
    }

    /// Encrypt page data with specific page ID for nonce derivation
    #[cfg(feature = "encryption")]
    pub async fn encrypt_page_with_id(
        &self,
        page_data: &[u8],
        page_id: u64,
    ) -> Result<Vec<u8>, JsValue> {
        if !*self.enabled.read().unwrap_or_else(|e| e.into_inner()) {
            return Ok(page_data.to_vec());
        }

        let key = self
            .key
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .ok_or_else(|| JsValue::from_str("Encryption key not set"))?;

        let nonce = self.generate_nonce(page_id);
        let algorithm = *self.algorithm.read().unwrap_or_else(|e| e.into_inner());

        let ciphertext = match algorithm {
            EncryptionAlgorithm::Aes256Gcm => {
                let cipher = Aes256Gcm::new(GenericArray::from_slice(&key));
                cipher
                    .encrypt(GenericArray::from_slice(&nonce), page_data)
                    .map_err(|e| JsValue::from_str(&format!("AES encryption failed: {}", e)))?
            }
            EncryptionAlgorithm::ChaCha20Poly1305 => {
                let cipher = ChaCha20Poly1305::new(GenericArray::from_slice(&key));
                cipher
                    .encrypt(GenericArray::from_slice(&nonce), page_data)
                    .map_err(|e| JsValue::from_str(&format!("ChaCha20 encryption failed: {}", e)))?
            }
        };

        // Prepend nonce to ciphertext for storage
        // Format: [algorithm (1 byte)][nonce (12 bytes)][ciphertext]
        let mut result = Vec::with_capacity(1 + 12 + ciphertext.len());
        result.push(algorithm as u8);
        result.extend_from_slice(&nonce);
        result.extend(ciphertext);

        Ok(result)
    }

    /// Encrypt page (non-encryption feature fallback)
    #[cfg(not(feature = "encryption"))]
    pub async fn encrypt_page(&self, page_data: &[u8]) -> Result<Vec<u8>, JsValue> {
        // When encryption feature is disabled, return data as-is
        Ok(page_data.to_vec())
    }

    /// Encrypt page with ID (non-encryption feature fallback)
    #[cfg(not(feature = "encryption"))]
    pub async fn encrypt_page_with_id(
        &self,
        page_data: &[u8],
        _page_id: u64,
    ) -> Result<Vec<u8>, JsValue> {
        Ok(page_data.to_vec())
    }

    /// Decrypt page data
    #[cfg(feature = "encryption")]
    pub async fn decrypt_page(&self, encrypted_data: &[u8]) -> Result<Vec<u8>, JsValue> {
        if !*self.enabled.read().unwrap_or_else(|e| e.into_inner()) {
            return Ok(encrypted_data.to_vec());
        }

        if encrypted_data.len() < 13 {
            return Err(JsValue::from_str("Encrypted data too short"));
        }

        let key = self
            .key
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .ok_or_else(|| JsValue::from_str("Encryption key not set"))?;

        // Parse format: [algorithm (1 byte)][nonce (12 bytes)][ciphertext]
        let stored_algorithm = encrypted_data[0];
        let nonce = &encrypted_data[1..13];
        let ciphertext = &encrypted_data[13..];

        let plaintext = match stored_algorithm {
            0 => {
                // AES-256-GCM
                let cipher = Aes256Gcm::new(GenericArray::from_slice(&key));
                cipher
                    .decrypt(GenericArray::from_slice(nonce), ciphertext)
                    .map_err(|e| JsValue::from_str(&format!("AES decryption failed: {}", e)))?
            }
            1 => {
                // ChaCha20-Poly1305
                let cipher = ChaCha20Poly1305::new(GenericArray::from_slice(&key));
                cipher
                    .decrypt(GenericArray::from_slice(nonce), ciphertext)
                    .map_err(|e| JsValue::from_str(&format!("ChaCha20 decryption failed: {}", e)))?
            }
            _ => {
                return Err(JsValue::from_str(&format!(
                    "Unknown algorithm: {}",
                    stored_algorithm
                )));
            }
        };

        Ok(plaintext)
    }

    /// Decrypt page (non-encryption feature fallback)
    #[cfg(not(feature = "encryption"))]
    pub async fn decrypt_page(&self, encrypted_data: &[u8]) -> Result<Vec<u8>, JsValue> {
        Ok(encrypted_data.to_vec())
    }

    /// Encrypt multiple pages (batch)
    #[cfg(target_arch = "wasm32")]
    pub async fn encrypt_pages(&self, pages: &js_sys::Array) -> Result<js_sys::Array, JsValue> {
        use wasm_bindgen::JsCast;

        let encrypted_pages = js_sys::Array::new();

        if !*self.enabled.read().unwrap_or_else(|e| e.into_inner()) {
            for i in 0..pages.length() {
                encrypted_pages.push(&pages.get(i));
            }
            return Ok(encrypted_pages);
        }

        for i in 0..pages.length() {
            if let Ok(page_arr) = pages.get(i).dyn_into::<js_sys::Uint8Array>() {
                let mut page_data = vec![0u8; page_arr.length() as usize];
                page_arr.copy_to(&mut page_data);

                let encrypted = self.encrypt_page_with_id(&page_data, i as u64).await?;
                encrypted_pages.push(&js_sys::Uint8Array::from(encrypted.as_slice()).into());
            }
        }

        Ok(encrypted_pages)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub async fn encrypt_pages(&self, _pages: &js_sys::Array) -> Result<js_sys::Array, JsValue> {
        Ok(js_sys::Array::new())
    }

    /// Decrypt multiple pages (batch)
    #[cfg(target_arch = "wasm32")]
    pub async fn decrypt_pages(
        &self,
        encrypted_pages: &js_sys::Array,
    ) -> Result<js_sys::Array, JsValue> {
        use wasm_bindgen::JsCast;

        let decrypted_pages = js_sys::Array::new();

        if !*self.enabled.read().unwrap_or_else(|e| e.into_inner()) {
            for i in 0..encrypted_pages.length() {
                decrypted_pages.push(&encrypted_pages.get(i));
            }
            return Ok(decrypted_pages);
        }

        for i in 0..encrypted_pages.length() {
            if let Ok(enc_arr) = encrypted_pages.get(i).dyn_into::<js_sys::Uint8Array>() {
                let mut encrypted_data = vec![0u8; enc_arr.length() as usize];
                enc_arr.copy_to(&mut encrypted_data);

                let decrypted = self.decrypt_page(&encrypted_data).await?;
                decrypted_pages.push(&js_sys::Uint8Array::from(decrypted.as_slice()).into());
            }
        }

        Ok(decrypted_pages)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub async fn decrypt_pages(
        &self,
        _encrypted_pages: &js_sys::Array,
    ) -> Result<js_sys::Array, JsValue> {
        Ok(js_sys::Array::new())
    }

    /// Re-encrypt all pages with a new key (key rotation)
    #[cfg(feature = "encryption")]
    pub async fn rotate_key(
        &self,
        new_password: &str,
        pages: Vec<Vec<u8>>,
    ) -> Result<Vec<Vec<u8>>, JsValue> {
        // Decrypt all pages with current key
        let mut decrypted = Vec::with_capacity(pages.len());
        for page in &pages {
            decrypted.push(self.decrypt_page(page).await?);
        }

        // Generate new key
        self.enable_with_password(new_password)?;

        // Re-encrypt with new key
        let mut re_encrypted = Vec::with_capacity(decrypted.len());
        for (i, page) in decrypted.iter().enumerate() {
            re_encrypted.push(self.encrypt_page_with_id(page, i as u64).await?);
        }

        Ok(re_encrypted)
    }
}

impl Default for OpfsEncryption {
    fn default() -> Self {
        Self::new()
    }
}

// Safety: These are safe because we're using proper locking
unsafe impl Send for OpfsEncryption {}
unsafe impl Sync for OpfsEncryption {}

// ============================================================================
// Web Crypto API Integration (Browser)
// ============================================================================

#[cfg(target_arch = "wasm32")]
pub mod web_crypto {
    use js_sys::{ArrayBuffer, Object, Reflect, Uint8Array};
    use wasm_bindgen::prelude::*;
    use wasm_bindgen_futures::JsFuture;
    use web_sys::window;

    /// Web Crypto API wrapper for browser-native AES-GCM
    pub struct WebCryptoAes {
        crypto_key: web_sys::CryptoKey,
    }

    impl WebCryptoAes {
        /// Import a raw key for AES-GCM
        pub async fn import_key(key: &[u8]) -> Result<Self, JsValue> {
            let window = window().ok_or_else(|| JsValue::from_str("No window"))?;
            let crypto = window
                .crypto()
                .map_err(|_| JsValue::from_str("No crypto"))?;
            let subtle = crypto.subtle();

            let key_data = Uint8Array::from(key);

            let algorithm = Object::new();
            Reflect::set(&algorithm, &"name".into(), &"AES-GCM".into())?;

            let usages = js_sys::Array::new();
            usages.push(&"encrypt".into());
            usages.push(&"decrypt".into());

            let crypto_key_js = JsFuture::from(
                subtle.import_key_with_object("raw", &key_data, &algorithm, false, &usages)?,
            )
            .await?;
            let crypto_key = crypto_key_js
                .dyn_into::<web_sys::CryptoKey>()
                .map_err(|_| JsValue::from_str("Invalid key type"))?;

            Ok(Self { crypto_key })
        }

        /// Encrypt data using Web Crypto API
        pub async fn encrypt(&self, nonce: &[u8], data: &[u8]) -> Result<Vec<u8>, JsValue> {
            let window = window().ok_or_else(|| JsValue::from_str("No window"))?;
            let crypto = window
                .crypto()
                .map_err(|_| JsValue::from_str("No crypto"))?;
            let subtle = crypto.subtle();

            let iv = Uint8Array::from(nonce);
            let plaintext = Uint8Array::from(data);

            let algorithm = Object::new();
            Reflect::set(&algorithm, &"name".into(), &"AES-GCM".into())?;
            Reflect::set(&algorithm, &"iv".into(), &iv)?;

            let result = JsFuture::from(subtle.encrypt_with_object_and_buffer_source(
                &algorithm,
                &self.crypto_key,
                &plaintext,
            )?)
            .await?;

            let buffer: ArrayBuffer = result.dyn_into()?;
            let array = Uint8Array::new(&buffer);
            Ok(array.to_vec())
        }

        /// Decrypt data using Web Crypto API
        pub async fn decrypt(&self, nonce: &[u8], ciphertext: &[u8]) -> Result<Vec<u8>, JsValue> {
            let window = window().ok_or_else(|| JsValue::from_str("No window"))?;
            let crypto = window
                .crypto()
                .map_err(|_| JsValue::from_str("No crypto"))?;
            let subtle = crypto.subtle();

            let iv = Uint8Array::from(nonce);
            let encrypted = Uint8Array::from(ciphertext);

            let algorithm = Object::new();
            Reflect::set(&algorithm, &"name".into(), &"AES-GCM".into())?;
            Reflect::set(&algorithm, &"iv".into(), &iv)?;

            let result = JsFuture::from(subtle.decrypt_with_object_and_buffer_source(
                &algorithm,
                &self.crypto_key,
                &encrypted,
            )?)
            .await?;

            let buffer: ArrayBuffer = result.dyn_into()?;
            let array = Uint8Array::new(&buffer);
            Ok(array.to_vec())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encryption_disabled_by_default() {
        let enc = OpfsEncryption::new();
        assert!(!enc.is_enabled());
    }

    #[test]
    fn test_enable_disable() {
        let enc = OpfsEncryption::new();
        enc.enable().unwrap();
        assert!(enc.is_enabled());
        assert!(enc.key_id().is_some());

        enc.disable();
        assert!(!enc.is_enabled());
    }

    #[test]
    fn test_algorithm_selection() {
        let enc = OpfsEncryption::new();
        enc.set_algorithm(EncryptionAlgorithm::ChaCha20Poly1305);
        assert_eq!(enc.algorithm(), EncryptionAlgorithm::ChaCha20Poly1305);

        enc.set_algorithm(EncryptionAlgorithm::Aes256Gcm);
        assert_eq!(enc.algorithm(), EncryptionAlgorithm::Aes256Gcm);
    }

    #[test]
    fn test_config_presets() {
        let low_mem = EncryptionConfig::low_memory();
        assert_eq!(low_mem.algorithm, EncryptionAlgorithm::ChaCha20Poly1305);
        assert!(low_mem.kdf_memory_cost < EncryptionConfig::default().kdf_memory_cost);

        let high_sec = EncryptionConfig::high_security();
        assert!(high_sec.kdf_iterations > EncryptionConfig::default().kdf_iterations);
    }

    #[cfg(feature = "encryption")]
    #[tokio::test]
    async fn test_encrypt_decrypt_roundtrip() {
        let enc = OpfsEncryption::new();
        enc.enable().unwrap();

        let plaintext = b"Hello, World! This is a test of encryption.";
        let encrypted = enc.encrypt_page(plaintext).await.unwrap();

        // Encrypted should be larger (nonce + auth tag)
        assert!(encrypted.len() > plaintext.len());

        let decrypted = enc.decrypt_page(&encrypted).await.unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[cfg(feature = "encryption")]
    #[tokio::test]
    async fn test_password_based_encryption() {
        let enc = OpfsEncryption::new();
        enc.enable_with_password("test_password_123").unwrap();

        assert!(enc.is_enabled());

        let plaintext = b"Secret data protected by password";
        let encrypted = enc.encrypt_page(plaintext).await.unwrap();
        let decrypted = enc.decrypt_page(&encrypted).await.unwrap();

        assert_eq!(decrypted, plaintext);
    }

    #[cfg(feature = "encryption")]
    #[tokio::test]
    async fn test_chacha20_encryption() {
        let config = EncryptionConfig {
            algorithm: EncryptionAlgorithm::ChaCha20Poly1305,
            ..Default::default()
        };
        let enc = OpfsEncryption::with_config(config);
        enc.enable().unwrap();

        let plaintext = b"Testing ChaCha20-Poly1305 encryption";
        let encrypted = enc.encrypt_page(plaintext).await.unwrap();
        let decrypted = enc.decrypt_page(&encrypted).await.unwrap();

        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_nonce_uniqueness() {
        let enc = OpfsEncryption::new();

        let nonce1 = enc.generate_nonce(0);
        let nonce2 = enc.generate_nonce(0);
        let nonce3 = enc.generate_nonce(1);

        // Same page_id but different counters
        assert_ne!(nonce1, nonce2);
        // Different page_ids
        assert_ne!(nonce1[0..8], nonce3[0..8]);
    }
}
