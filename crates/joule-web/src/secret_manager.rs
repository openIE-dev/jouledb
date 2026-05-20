//! In-memory secret / credential manager with audit trail, TTL expiry,
//! XOR-based obfuscation (demo), rotation, and envelope encryption pattern.
//!
//! This is *not* a production-grade HSM — it demonstrates the vault API
//! surface with an XOR cipher for illustrative purposes.

use chrono::{DateTime, Utc};
use std::collections::HashMap;
use uuid::Uuid;

// ── Errors ──────────────────────────────────────────────────────

/// Errors from the secret manager.
#[derive(Debug, Clone, PartialEq)]
pub enum SecretError {
    /// The requested secret does not exist.
    NotFound(String),
    /// The secret has expired.
    Expired(String),
    /// Access denied (wrong key or sealed vault).
    AccessDenied(String),
    /// The vault is sealed and cannot serve secrets.
    VaultSealed,
    /// Duplicate key on insert without overwrite.
    AlreadyExists(String),
}

impl std::fmt::Display for SecretError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SecretError::NotFound(k) => write!(f, "secret not found: {}", k),
            SecretError::Expired(k) => write!(f, "secret expired: {}", k),
            SecretError::AccessDenied(msg) => write!(f, "access denied: {}", msg),
            SecretError::VaultSealed => write!(f, "vault is sealed"),
            SecretError::AlreadyExists(k) => write!(f, "secret already exists: {}", k),
        }
    }
}

// ── Audit ───────────────────────────────────────────────────────

/// Audit operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuditAction {
    Read,
    Write,
    Delete,
    Rotate,
    Seal,
    Unseal,
}

impl std::fmt::Display for AuditAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AuditAction::Read => write!(f, "read"),
            AuditAction::Write => write!(f, "write"),
            AuditAction::Delete => write!(f, "delete"),
            AuditAction::Rotate => write!(f, "rotate"),
            AuditAction::Seal => write!(f, "seal"),
            AuditAction::Unseal => write!(f, "unseal"),
        }
    }
}

/// A single audit trail entry.
#[derive(Debug, Clone)]
pub struct AuditEntry {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub action: AuditAction,
    pub key: String,
    pub actor: String,
    pub success: bool,
}

// ── Secret Entry ────────────────────────────────────────────────

/// Internal representation of a stored secret.
#[derive(Debug, Clone)]
struct SecretEntry {
    /// XOR-obfuscated payload.
    ciphertext: Vec<u8>,
    /// XOR key used for this entry.
    obfuscation_key: Vec<u8>,
    /// When the secret was created / last rotated.
    created_at: DateTime<Utc>,
    /// Optional TTL: the secret expires after this instant.
    expires_at: Option<DateTime<Utc>>,
    /// How many times this secret has been rotated.
    version: u32,
    /// Metadata tags.
    tags: HashMap<String, String>,
}

// ── XOR helpers ─────────────────────────────────────────────────

/// Derive a deterministic obfuscation key from a master key and a salt.
fn derive_key(master: &[u8], salt: &[u8], length: usize) -> Vec<u8> {
    let mut key = Vec::with_capacity(length);
    for i in 0..length {
        let m = master[i % master.len()];
        let s = salt[i % salt.len()];
        key.push(m ^ s ^ (i as u8));
    }
    key
}

/// XOR-encrypt / decrypt (symmetric).
fn xor_apply(data: &[u8], key: &[u8]) -> Vec<u8> {
    data.iter()
        .enumerate()
        .map(|(i, b)| b ^ key[i % key.len()])
        .collect()
}

// ── Vault ───────────────────────────────────────────────────────

/// In-memory secret vault with obfuscation, TTL, rotation, and audit.
pub struct SecretManager {
    master_key: Vec<u8>,
    secrets: HashMap<String, SecretEntry>,
    audit_log: Vec<AuditEntry>,
    sealed: bool,
}

impl SecretManager {
    /// Create a new vault with the given master key bytes.
    pub fn new(master_key: &[u8]) -> Self {
        Self {
            master_key: master_key.to_vec(),
            secrets: HashMap::new(),
            audit_log: Vec::new(),
            sealed: false,
        }
    }

    /// Seal the vault — all read/write operations will fail until unsealed.
    pub fn seal(&mut self, actor: &str) {
        self.sealed = true;
        self.audit(AuditAction::Seal, "*", actor, true);
    }

    /// Unseal the vault.
    pub fn unseal(&mut self, master_key: &[u8], actor: &str) -> Result<(), SecretError> {
        if master_key != self.master_key.as_slice() {
            self.audit(AuditAction::Unseal, "*", actor, false);
            return Err(SecretError::AccessDenied("wrong master key".into()));
        }
        self.sealed = false;
        self.audit(AuditAction::Unseal, "*", actor, true);
        Ok(())
    }

    /// Is the vault sealed?
    pub fn is_sealed(&self) -> bool {
        self.sealed
    }

    /// Store a secret. Fails if the vault is sealed or the key already exists.
    pub fn put(
        &mut self,
        key: impl Into<String>,
        plaintext: &[u8],
        actor: &str,
        ttl_seconds: Option<i64>,
    ) -> Result<(), SecretError> {
        if self.sealed {
            return Err(SecretError::VaultSealed);
        }
        let key = key.into();
        if self.secrets.contains_key(&key) {
            self.audit(AuditAction::Write, &key, actor, false);
            return Err(SecretError::AlreadyExists(key));
        }

        let salt = Uuid::new_v4().to_string();
        let obfuscation_key = derive_key(&self.master_key, salt.as_bytes(), plaintext.len().max(16));
        let ciphertext = xor_apply(plaintext, &obfuscation_key);

        let now = Utc::now();
        let expires_at = ttl_seconds.map(|secs| now + chrono::Duration::seconds(secs));

        self.secrets.insert(key.clone(), SecretEntry {
            ciphertext,
            obfuscation_key,
            created_at: now,
            expires_at,
            version: 1,
            tags: HashMap::new(),
        });
        self.audit(AuditAction::Write, &key, actor, true);
        Ok(())
    }

    /// Retrieve a secret's plaintext. Fails if sealed, not found, or expired.
    pub fn get(&mut self, key: &str, actor: &str) -> Result<Vec<u8>, SecretError> {
        if self.sealed {
            return Err(SecretError::VaultSealed);
        }
        let entry = self.secrets.get(key)
            .ok_or_else(|| SecretError::NotFound(key.into()))?;

        if let Some(exp) = entry.expires_at {
            if Utc::now() > exp {
                self.audit(AuditAction::Read, key, actor, false);
                return Err(SecretError::Expired(key.into()));
            }
        }

        let plaintext = xor_apply(&entry.ciphertext, &entry.obfuscation_key);
        self.audit(AuditAction::Read, key, actor, true);
        Ok(plaintext)
    }

    /// Delete a secret.
    pub fn delete(&mut self, key: &str, actor: &str) -> Result<(), SecretError> {
        if self.sealed {
            return Err(SecretError::VaultSealed);
        }
        if self.secrets.remove(key).is_some() {
            self.audit(AuditAction::Delete, key, actor, true);
            Ok(())
        } else {
            self.audit(AuditAction::Delete, key, actor, false);
            Err(SecretError::NotFound(key.into()))
        }
    }

    /// Rotate a secret: replace its value and bump the version counter.
    pub fn rotate(
        &mut self,
        key: &str,
        new_plaintext: &[u8],
        actor: &str,
    ) -> Result<u32, SecretError> {
        if self.sealed {
            return Err(SecretError::VaultSealed);
        }
        let entry = self.secrets.get_mut(key)
            .ok_or_else(|| SecretError::NotFound(key.into()))?;

        let salt = Uuid::new_v4().to_string();
        let new_key = derive_key(&self.master_key, salt.as_bytes(), new_plaintext.len().max(16));
        entry.ciphertext = xor_apply(new_plaintext, &new_key);
        entry.obfuscation_key = new_key;
        entry.version += 1;
        entry.created_at = Utc::now();
        let version = entry.version;
        self.audit(AuditAction::Rotate, key, actor, true);
        Ok(version)
    }

    /// Get the version counter of a secret.
    pub fn version(&self, key: &str) -> Option<u32> {
        self.secrets.get(key).map(|e| e.version)
    }

    /// Set a metadata tag on a secret.
    pub fn set_tag(&mut self, key: &str, tag_key: &str, tag_val: &str) -> Result<(), SecretError> {
        let entry = self.secrets.get_mut(key)
            .ok_or_else(|| SecretError::NotFound(key.into()))?;
        entry.tags.insert(tag_key.to_string(), tag_val.to_string());
        Ok(())
    }

    /// Get a metadata tag.
    pub fn get_tag(&self, key: &str, tag_key: &str) -> Option<&str> {
        self.secrets.get(key)
            .and_then(|e| e.tags.get(tag_key))
            .map(|s| s.as_str())
    }

    /// List all secret keys (does not reveal values).
    pub fn list_keys(&self) -> Vec<&str> {
        let mut keys: Vec<_> = self.secrets.keys().map(|k| k.as_str()).collect();
        keys.sort();
        keys
    }

    /// Number of stored secrets.
    pub fn count(&self) -> usize {
        self.secrets.len()
    }

    /// Check if a secret exists (does not check expiry).
    pub fn exists(&self, key: &str) -> bool {
        self.secrets.contains_key(key)
    }

    // ── Envelope encryption ─────────────────────────────────────

    /// Envelope encrypt: encrypt data with a random data-encryption key (DEK),
    /// then encrypt the DEK with the master key, returning (encrypted_data, wrapped_dek).
    pub fn envelope_encrypt(&self, plaintext: &[u8]) -> (Vec<u8>, Vec<u8>) {
        // Generate random DEK.
        let dek_seed = Uuid::new_v4();
        let dek = dek_seed.as_bytes().to_vec();
        let encrypted_data = xor_apply(plaintext, &dek);
        let wrapped_dek = xor_apply(&dek, &self.master_key);
        (encrypted_data, wrapped_dek)
    }

    /// Envelope decrypt: unwrap the DEK with the master key, then decrypt data.
    pub fn envelope_decrypt(&self, ciphertext: &[u8], wrapped_dek: &[u8]) -> Vec<u8> {
        let dek = xor_apply(wrapped_dek, &self.master_key);
        xor_apply(ciphertext, &dek)
    }

    // ── Audit log ───────────────────────────────────────────────

    /// Full audit log.
    pub fn audit_log(&self) -> &[AuditEntry] {
        &self.audit_log
    }

    /// Audit entries for a specific key.
    pub fn audit_log_for_key(&self, key: &str) -> Vec<&AuditEntry> {
        self.audit_log.iter().filter(|e| e.key == key).collect()
    }

    fn audit(&mut self, action: AuditAction, key: &str, actor: &str, success: bool) {
        self.audit_log.push(AuditEntry {
            id: Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            action,
            key: key.to_string(),
            actor: actor.to_string(),
            success,
        });
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn vault() -> SecretManager {
        SecretManager::new(b"master-key-32-bytes-for-testing!")
    }

    #[test]
    fn put_and_get() {
        let mut v = vault();
        v.put("db_pass", b"s3cret", "admin", None).unwrap();
        let plain = v.get("db_pass", "admin").unwrap();
        assert_eq!(plain, b"s3cret");
    }

    #[test]
    fn get_not_found() {
        let mut v = vault();
        assert!(matches!(v.get("nope", "admin"), Err(SecretError::NotFound(_))));
    }

    #[test]
    fn duplicate_key_rejected() {
        let mut v = vault();
        v.put("k", b"v1", "admin", None).unwrap();
        assert!(matches!(v.put("k", b"v2", "admin", None), Err(SecretError::AlreadyExists(_))));
    }

    #[test]
    fn delete_secret() {
        let mut v = vault();
        v.put("k", b"v", "admin", None).unwrap();
        v.delete("k", "admin").unwrap();
        assert!(!v.exists("k"));
    }

    #[test]
    fn delete_nonexistent() {
        let mut v = vault();
        assert!(matches!(v.delete("nope", "admin"), Err(SecretError::NotFound(_))));
    }

    #[test]
    fn rotate_secret() {
        let mut v = vault();
        v.put("api_key", b"old_key", "admin", None).unwrap();
        let ver = v.rotate("api_key", b"new_key", "admin").unwrap();
        assert_eq!(ver, 2);
        let plain = v.get("api_key", "admin").unwrap();
        assert_eq!(plain, b"new_key");
    }

    #[test]
    fn version_tracking() {
        let mut v = vault();
        v.put("k", b"v1", "admin", None).unwrap();
        assert_eq!(v.version("k"), Some(1));
        v.rotate("k", b"v2", "admin").unwrap();
        assert_eq!(v.version("k"), Some(2));
        v.rotate("k", b"v3", "admin").unwrap();
        assert_eq!(v.version("k"), Some(3));
    }

    #[test]
    fn seal_blocks_operations() {
        let mut v = vault();
        v.put("k", b"v", "admin", None).unwrap();
        v.seal("admin");
        assert!(v.is_sealed());
        assert!(matches!(v.get("k", "admin"), Err(SecretError::VaultSealed)));
        assert!(matches!(v.put("k2", b"v", "admin", None), Err(SecretError::VaultSealed)));
        assert!(matches!(v.delete("k", "admin"), Err(SecretError::VaultSealed)));
    }

    #[test]
    fn unseal_with_correct_key() {
        let mut v = vault();
        v.seal("admin");
        v.unseal(b"master-key-32-bytes-for-testing!", "admin").unwrap();
        assert!(!v.is_sealed());
    }

    #[test]
    fn unseal_with_wrong_key() {
        let mut v = vault();
        v.seal("admin");
        let result = v.unseal(b"wrong-key", "admin");
        assert!(matches!(result, Err(SecretError::AccessDenied(_))));
    }

    #[test]
    fn audit_trail_recorded() {
        let mut v = vault();
        v.put("k", b"v", "admin", None).unwrap();
        let _ = v.get("k", "reader");
        v.delete("k", "admin").unwrap();
        assert_eq!(v.audit_log().len(), 3);
        let log_for_k = v.audit_log_for_key("k");
        assert_eq!(log_for_k.len(), 3);
        assert_eq!(log_for_k[0].action, AuditAction::Write);
        assert_eq!(log_for_k[1].action, AuditAction::Read);
        assert_eq!(log_for_k[2].action, AuditAction::Delete);
    }

    #[test]
    fn list_keys() {
        let mut v = vault();
        v.put("beta", b"x", "admin", None).unwrap();
        v.put("alpha", b"y", "admin", None).unwrap();
        let keys = v.list_keys();
        assert_eq!(keys, vec!["alpha", "beta"]);
    }

    #[test]
    fn count_and_exists() {
        let mut v = vault();
        assert_eq!(v.count(), 0);
        assert!(!v.exists("k"));
        v.put("k", b"v", "admin", None).unwrap();
        assert_eq!(v.count(), 1);
        assert!(v.exists("k"));
    }

    #[test]
    fn tags() {
        let mut v = vault();
        v.put("k", b"v", "admin", None).unwrap();
        v.set_tag("k", "env", "production").unwrap();
        assert_eq!(v.get_tag("k", "env"), Some("production"));
        assert_eq!(v.get_tag("k", "missing"), None);
    }

    #[test]
    fn envelope_encrypt_decrypt() {
        let v = vault();
        let plaintext = b"sensitive payload";
        let (ct, wrapped_dek) = v.envelope_encrypt(plaintext);
        assert_ne!(ct, plaintext.to_vec());
        let recovered = v.envelope_decrypt(&ct, &wrapped_dek);
        assert_eq!(recovered, plaintext.to_vec());
    }

    #[test]
    fn derive_key_determinism() {
        let k1 = derive_key(b"master", b"salt", 32);
        let k2 = derive_key(b"master", b"salt", 32);
        assert_eq!(k1, k2);
    }

    #[test]
    fn xor_roundtrip() {
        let key = b"abcdefgh";
        let data = b"hello world!!";
        let ct = xor_apply(data, key);
        let pt = xor_apply(&ct, key);
        assert_eq!(pt, data.to_vec());
    }

    #[test]
    fn ttl_expiry() {
        let mut v = vault();
        // Insert with TTL of 0 seconds — immediately expired.
        v.put("ephemeral", b"temp", "admin", Some(0)).unwrap();
        // The secret was created at Utc::now() with 0-second TTL,
        // so expires_at == created_at. Any subsequent get at or after that moment should fail.
        // We rely on the tiny passage of real time.
        std::thread::sleep(std::time::Duration::from_millis(5));
        let result = v.get("ephemeral", "admin");
        assert!(matches!(result, Err(SecretError::Expired(_))));
    }

    #[test]
    fn empty_plaintext() {
        let mut v = vault();
        v.put("empty", b"", "admin", None).unwrap();
        let plain = v.get("empty", "admin").unwrap();
        assert!(plain.is_empty());
    }
}
