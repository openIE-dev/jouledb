use std::collections::HashMap;
use std::sync::RwLock;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use x509_parser::prelude::*;

use crate::encryption::{Aes256Gcm, EncryptionError, SymmetricKey};

/// Error type for secret store operations.
#[derive(Debug, thiserror::Error)]
pub enum SecretError {
    #[error("secret not found: {0}")]
    NotFound(String),
    #[error("version {version} not found for secret {name}")]
    VersionNotFound { name: String, version: u32 },
    #[error("encryption error: {0}")]
    Encryption(#[from] EncryptionError),
    #[error("certificate parse error: {0}")]
    CertParse(String),
}

/// A single versioned entry in the secret store.
/// The value is always stored encrypted.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SecretEntry {
    /// The version number (1-based, monotonically increasing).
    pub version: u32,
    /// AES-256-GCM encrypted value.
    pub encrypted_value: Vec<u8>,
    /// Unix timestamp (seconds) when this version was created.
    pub created_at: u64,
    /// Optional Unix timestamp (seconds) when this version expires.
    pub expires_at: Option<u64>,
    /// If this version was created by rotation, the previous version number.
    pub rotated_from: Option<u32>,
}

/// Policy governing automatic secret rotation and version retention.
#[derive(Debug, Clone)]
pub struct RotationPolicy {
    /// How often secrets should be rotated.
    pub rotation_interval: Duration,
    /// Grace period after rotation during which the old version remains valid.
    pub grace_period: Duration,
    /// Maximum number of versions to retain per secret.
    pub max_versions: u32,
}

impl Default for RotationPolicy {
    fn default() -> Self {
        Self {
            rotation_interval: Duration::from_secs(90 * 24 * 3600), // 90 days
            grace_period: Duration::from_secs(7 * 24 * 3600),       // 7 days
            max_versions: 5,
        }
    }
}

/// Metadata about a secret version (without the encrypted payload).
#[derive(Debug, Clone)]
pub struct SecretVersionInfo {
    pub version: u32,
    pub created_at: u64,
    pub expires_at: Option<u64>,
}

/// Encrypted secret store with versioning, rotation, and pruning.
///
/// All secret values are encrypted at rest using AES-256-GCM with the master key.
pub struct SecretStore {
    master_key: SymmetricKey,
    secrets: RwLock<HashMap<String, Vec<SecretEntry>>>,
    rotation_policy: RotationPolicy,
}

impl Default for SecretStore {
    fn default() -> Self {
        Self::new()
    }
}

impl SecretStore {
    /// Create a new secret store with a generated master key and default rotation policy.
    pub fn new() -> Self {
        Self {
            master_key: SymmetricKey::generate(),
            secrets: RwLock::new(HashMap::new()),
            rotation_policy: RotationPolicy::default(),
        }
    }

    /// Create a new secret store with a specific master key and rotation policy.
    pub fn with_policy(master_key: SymmetricKey, policy: RotationPolicy) -> Self {
        Self {
            master_key,
            secrets: RwLock::new(HashMap::new()),
            rotation_policy: policy,
        }
    }

    /// Store a secret value. If the secret already exists, a new version is appended.
    /// The plaintext is encrypted before storage.
    pub fn put(
        &self,
        name: &str,
        plaintext: &[u8],
        expires_at: Option<u64>,
    ) -> Result<u32, SecretError> {
        let cipher = Aes256Gcm::new(&self.master_key);
        let encrypted_value = cipher.encrypt(plaintext, b"")?;

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before UNIX epoch")
            .as_secs();

        let mut secrets = self.secrets.write().unwrap();
        let entries = secrets.entry(name.to_string()).or_default();

        let version = entries.last().map_or(1, |e| e.version + 1);
        let rotated_from = if version > 1 { Some(version - 1) } else { None };

        entries.push(SecretEntry {
            version,
            encrypted_value,
            created_at: now,
            expires_at,
            rotated_from,
        });

        Ok(version)
    }

    /// Retrieve the latest version of a secret, decrypted.
    pub fn get(&self, name: &str) -> Result<Vec<u8>, SecretError> {
        let secrets = self.secrets.read().unwrap();
        let entries = secrets
            .get(name)
            .ok_or_else(|| SecretError::NotFound(name.to_string()))?;
        let entry = entries
            .last()
            .ok_or_else(|| SecretError::NotFound(name.to_string()))?;

        let cipher = Aes256Gcm::new(&self.master_key);
        let plaintext = cipher.decrypt(&entry.encrypted_value, b"")?;
        Ok(plaintext)
    }

    /// Retrieve a specific version of a secret, decrypted.
    pub fn get_version(&self, name: &str, version: u32) -> Result<Vec<u8>, SecretError> {
        let secrets = self.secrets.read().unwrap();
        let entries = secrets
            .get(name)
            .ok_or_else(|| SecretError::NotFound(name.to_string()))?;
        let entry = entries
            .iter()
            .find(|e| e.version == version)
            .ok_or_else(|| SecretError::VersionNotFound {
                name: name.to_string(),
                version,
            })?;

        let cipher = Aes256Gcm::new(&self.master_key);
        let plaintext = cipher.decrypt(&entry.encrypted_value, b"")?;
        Ok(plaintext)
    }

    /// List the names of all secrets in the store.
    pub fn list(&self) -> Vec<String> {
        let secrets = self.secrets.read().unwrap();
        let mut names: Vec<String> = secrets.keys().cloned().collect();
        names.sort();
        names
    }

    /// Return version metadata for all versions of a secret.
    pub fn versions(&self, name: &str) -> Result<Vec<SecretVersionInfo>, SecretError> {
        let secrets = self.secrets.read().unwrap();
        let entries = secrets
            .get(name)
            .ok_or_else(|| SecretError::NotFound(name.to_string()))?;

        Ok(entries
            .iter()
            .map(|e| SecretVersionInfo {
                version: e.version,
                created_at: e.created_at,
                expires_at: e.expires_at,
            })
            .collect())
    }

    /// Delete all versions of a secret.
    pub fn delete(&self, name: &str) -> Result<(), SecretError> {
        let mut secrets = self.secrets.write().unwrap();
        secrets
            .remove(name)
            .ok_or_else(|| SecretError::NotFound(name.to_string()))?;
        Ok(())
    }

    /// Rotate the master key. All existing secret values are decrypted with the old key
    /// and re-encrypted with the new key.
    pub fn rotate_master_key(&mut self, new_key: SymmetricKey) -> Result<(), SecretError> {
        let old_cipher = Aes256Gcm::new(&self.master_key);
        let new_cipher = Aes256Gcm::new(&new_key);

        let mut secrets = self.secrets.write().unwrap();
        for entries in secrets.values_mut() {
            for entry in entries.iter_mut() {
                let plaintext = old_cipher.decrypt(&entry.encrypted_value, b"")?;
                entry.encrypted_value = new_cipher.encrypt(&plaintext, b"")?;
            }
        }

        drop(secrets);
        self.master_key = new_key;
        Ok(())
    }

    /// Check which secrets need rotation based on the rotation policy.
    /// Returns the names of secrets whose latest version is older than the rotation interval.
    pub fn check_rotation_needed(&self) -> Vec<String> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before UNIX epoch")
            .as_secs();

        let secrets = self.secrets.read().unwrap();
        let mut needs_rotation = Vec::new();

        for (name, entries) in secrets.iter() {
            if let Some(latest) = entries.last() {
                let age = now.saturating_sub(latest.created_at);
                if age >= self.rotation_policy.rotation_interval.as_secs() {
                    needs_rotation.push(name.clone());
                }
            }
        }

        needs_rotation.sort();
        needs_rotation
    }

    /// Remove old versions exceeding `max_versions`, keeping the most recent ones.
    pub fn prune_old_versions(&self) -> usize {
        let max = self.rotation_policy.max_versions as usize;
        let mut secrets = self.secrets.write().unwrap();
        let mut pruned = 0;

        for entries in secrets.values_mut() {
            if entries.len() > max {
                let to_remove = entries.len() - max;
                entries.drain(..to_remove);
                pruned += to_remove;
            }
        }

        pruned
    }
}

/// Checks whether an X.509 certificate (DER-encoded) needs renewal.
pub struct CertRenewalChecker {
    /// How far before expiry to trigger renewal.
    pub renewal_threshold: Duration,
}

impl CertRenewalChecker {
    /// Create a checker with a given threshold before expiry.
    pub fn new(renewal_threshold: Duration) -> Self {
        Self { renewal_threshold }
    }

    /// Returns `true` if the certificate expires within the renewal threshold.
    pub fn needs_renewal(&self, cert_der: &[u8]) -> Result<bool, SecretError> {
        let days = self.days_until_expiry(cert_der)?;
        let threshold_days = self.renewal_threshold.as_secs() / 86400;
        Ok(days <= threshold_days as i64)
    }

    /// Returns the number of days until the certificate expires.
    /// A negative value means the certificate has already expired.
    pub fn days_until_expiry(&self, cert_der: &[u8]) -> Result<i64, SecretError> {
        let (_, cert) = X509Certificate::from_der(cert_der)
            .map_err(|e| SecretError::CertParse(format!("invalid X.509 DER: {e}")))?;

        let not_after = cert.validity().not_after.timestamp();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before UNIX epoch")
            .as_secs() as i64;

        let remaining_secs = not_after - now;
        Ok(remaining_secs / 86400)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_store() -> SecretStore {
        let key = SymmetricKey::generate();
        let policy = RotationPolicy {
            rotation_interval: Duration::from_secs(1), // 1 second for testing
            grace_period: Duration::from_secs(1),
            max_versions: 3,
        };
        SecretStore::with_policy(key, policy)
    }

    #[test]
    fn put_get_roundtrip() {
        let store = make_store();
        let secret = b"super-secret-api-key";
        let version = store.put("api-key", secret, None).unwrap();
        assert_eq!(version, 1);

        let retrieved = store.get("api-key").unwrap();
        assert_eq!(retrieved, secret);
    }

    #[test]
    fn encryption_verification() {
        let store = make_store();
        let plaintext = b"do-not-store-in-clear";
        store.put("token", plaintext, None).unwrap();

        let secrets = store.secrets.read().unwrap();
        let entry = &secrets["token"][0];
        // The encrypted value must differ from the plaintext.
        assert_ne!(entry.encrypted_value, plaintext);
        // Encrypted output includes nonce (12) + ciphertext + tag (16), so it must be longer.
        assert!(entry.encrypted_value.len() > plaintext.len());
    }

    #[test]
    fn versioning() {
        let store = make_store();
        let v1 = store.put("db-pass", b"password-v1", None).unwrap();
        let v2 = store.put("db-pass", b"password-v2", None).unwrap();
        assert_eq!(v1, 1);
        assert_eq!(v2, 2);

        let got_v1 = store.get_version("db-pass", 1).unwrap();
        assert_eq!(got_v1, b"password-v1");

        let got_v2 = store.get_version("db-pass", 2).unwrap();
        assert_eq!(got_v2, b"password-v2");

        // get() returns latest
        let latest = store.get("db-pass").unwrap();
        assert_eq!(latest, b"password-v2");
    }

    #[test]
    fn list_secrets() {
        let store = make_store();
        store.put("alpha", b"a", None).unwrap();
        store.put("beta", b"b", None).unwrap();
        store.put("gamma", b"c", None).unwrap();

        let names = store.list();
        assert_eq!(names, vec!["alpha", "beta", "gamma"]);
    }

    #[test]
    fn delete_secret() {
        let store = make_store();
        store.put("ephemeral", b"temp", None).unwrap();
        assert!(store.get("ephemeral").is_ok());

        store.delete("ephemeral").unwrap();
        assert!(matches!(
            store.get("ephemeral"),
            Err(SecretError::NotFound(_))
        ));
    }

    #[test]
    fn delete_not_found() {
        let store = make_store();
        assert!(matches!(
            store.delete("nonexistent"),
            Err(SecretError::NotFound(_))
        ));
    }

    #[test]
    fn rotate_master_key() {
        let mut store = make_store();
        store.put("secret-a", b"value-a", None).unwrap();
        store.put("secret-b", b"value-b", None).unwrap();

        // Capture the ciphertext before rotation.
        let ct_before = {
            let secrets = store.secrets.read().unwrap();
            secrets["secret-a"][0].encrypted_value.clone()
        };

        let new_key = SymmetricKey::generate();
        store.rotate_master_key(new_key).unwrap();

        // Ciphertext must have changed after re-encryption with the new key.
        let ct_after = {
            let secrets = store.secrets.read().unwrap();
            secrets["secret-a"][0].encrypted_value.clone()
        };
        assert_ne!(ct_before, ct_after);

        // Values must still decrypt correctly with the new master key.
        assert_eq!(store.get("secret-a").unwrap(), b"value-a");
        assert_eq!(store.get("secret-b").unwrap(), b"value-b");
    }

    #[test]
    fn prune_old_versions() {
        let store = make_store(); // max_versions = 3
        for i in 1..=6 {
            store.put("key", format!("v{i}").as_bytes(), None).unwrap();
        }

        let pruned = store.prune_old_versions();
        assert_eq!(pruned, 3);

        let versions = store.versions("key").unwrap();
        assert_eq!(versions.len(), 3);
        // Only versions 4, 5, 6 should remain.
        assert_eq!(versions[0].version, 4);
        assert_eq!(versions[2].version, 6);

        // The remaining versions should still decrypt.
        assert_eq!(store.get_version("key", 4).unwrap(), b"v4");
        assert_eq!(store.get_version("key", 6).unwrap(), b"v6");
    }

    #[test]
    fn check_rotation_needed() {
        let key = SymmetricKey::generate();
        let policy = RotationPolicy {
            rotation_interval: Duration::from_secs(1),
            grace_period: Duration::from_secs(1),
            max_versions: 5,
        };
        let store = SecretStore::with_policy(key, policy);
        store.put("old-secret", b"val", None).unwrap();

        // Sleep just over the rotation interval so the secret becomes stale.
        std::thread::sleep(Duration::from_millis(1100));

        let needs = store.check_rotation_needed();
        assert!(needs.contains(&"old-secret".to_string()));
    }

    #[test]
    fn not_found_get() {
        let store = make_store();
        assert!(matches!(
            store.get("missing"),
            Err(SecretError::NotFound(_))
        ));
    }

    #[test]
    fn version_not_found() {
        let store = make_store();
        store.put("exists", b"val", None).unwrap();
        assert!(matches!(
            store.get_version("exists", 99),
            Err(SecretError::VersionNotFound { .. })
        ));
    }

    #[test]
    fn versions_metadata() {
        let store = make_store();
        let expires = 1_800_000_000;
        store.put("meta", b"v1", Some(expires)).unwrap();
        store.put("meta", b"v2", None).unwrap();

        let versions = store.versions("meta").unwrap();
        assert_eq!(versions.len(), 2);
        assert_eq!(versions[0].version, 1);
        assert_eq!(versions[0].expires_at, Some(expires));
        assert_eq!(versions[1].version, 2);
        assert_eq!(versions[1].expires_at, None);
    }

    #[test]
    fn default_rotation_policy() {
        let policy = RotationPolicy::default();
        assert_eq!(policy.rotation_interval.as_secs(), 90 * 24 * 3600);
        assert_eq!(policy.grace_period.as_secs(), 7 * 24 * 3600);
        assert_eq!(policy.max_versions, 5);
    }

    #[test]
    fn cert_renewal_checker_needs_renewal() {
        // Generate a self-signed certificate that expires in 10 days.
        let mut params = rcgen::CertificateParams::default();
        params.not_before = rcgen::date_time_ymd(2026, 1, 1);
        // Set expiry close enough to trigger renewal.
        let now = chrono::Utc::now();
        let expiry = now + chrono::Duration::days(10);
        params.not_after = rcgen::date_time_ymd(
            expiry.format("%Y").to_string().parse().unwrap(),
            expiry.format("%m").to_string().parse::<u32>().unwrap() as u8,
            expiry.format("%d").to_string().parse::<u32>().unwrap() as u8,
        );

        let key_pair = rcgen::KeyPair::generate_for(&rcgen::PKCS_ED25519).unwrap();
        let cert = params.self_signed(&key_pair).unwrap();
        let cert_der = cert.der().to_vec();

        // Threshold of 30 days: cert expiring in ~10 days should need renewal.
        let checker = CertRenewalChecker::new(Duration::from_secs(30 * 86400));
        assert!(checker.needs_renewal(&cert_der).unwrap());

        // Threshold of 5 days: cert expiring in ~10 days should NOT need renewal yet.
        let checker_short = CertRenewalChecker::new(Duration::from_secs(5 * 86400));
        assert!(!checker_short.needs_renewal(&cert_der).unwrap());
    }

    #[test]
    fn cert_days_until_expiry() {
        let mut params = rcgen::CertificateParams::default();
        params.not_before = rcgen::date_time_ymd(2026, 1, 1);
        params.not_after = rcgen::date_time_ymd(2027, 1, 1);

        let key_pair = rcgen::KeyPair::generate_for(&rcgen::PKCS_ED25519).unwrap();
        let cert = params.self_signed(&key_pair).unwrap();
        let cert_der = cert.der().to_vec();

        let checker = CertRenewalChecker::new(Duration::from_secs(30 * 86400));
        let days = checker.days_until_expiry(&cert_der).unwrap();
        // The cert expires on 2027-01-01, so days should be positive (well into the future).
        assert!(days > 0);
    }
}
