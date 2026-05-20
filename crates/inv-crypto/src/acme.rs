//! ACME v2 client for Let's Encrypt certificate provisioning.
//!
//! Uses DNS-01 challenges via Cloudflare API, since nodes are behind Cloudflare proxy
//! (HTTP-01 on port 80 is unreliable). Generates ECDSA P-256 keys for Cloudflare compatibility.
//!
//! # Env vars
//! - `ACME_EMAIL` — contact email for Let's Encrypt account
//! - `CLOUDFLARE_API_TOKEN` — API token with Zone:DNS:Edit permission
//! - `CLOUDFLARE_ZONE_ID` — zone ID for the domain (e.g., openie.sh)
//!
//! If any env var is missing, `AcmeClient::from_env()` returns an unconfigured client
//! and the agent falls back to self-signed certificates.

use std::path::Path;

use base64::Engine;
use sha2::Digest;
use thiserror::Error;
use tracing::{debug, info, warn};

const LETS_ENCRYPT_DIRECTORY: &str = "https://acme-v02.api.letsencrypt.org/directory";
const LETS_ENCRYPT_STAGING: &str = "https://acme-staging-v02.api.letsencrypt.org/directory";

#[derive(Debug, Error)]
pub enum AcmeError {
    #[error("ACME not configured (missing env vars)")]
    NotConfigured,

    #[error("HTTP error: {0}")]
    Http(String),

    #[error("ACME protocol error: {0}")]
    Protocol(String),

    #[error("Cloudflare DNS error: {0}")]
    Dns(String),

    #[error("certificate error: {0}")]
    Certificate(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("timeout waiting for challenge validation")]
    Timeout,
}

/// ACME client for automated certificate provisioning via Let's Encrypt.
pub struct AcmeClient {
    email: Option<String>,
    cf_token: Option<String>,
    cf_zone_id: Option<String>,
    http: reqwest::Client,
    staging: bool,
}

impl AcmeClient {
    /// Create from environment variables.
    pub fn from_env() -> Self {
        let email = std::env::var("ACME_EMAIL").ok();
        let cf_token = std::env::var("CLOUDFLARE_API_TOKEN").ok();
        let cf_zone_id = std::env::var("CLOUDFLARE_ZONE_ID").ok();

        let configured = email.is_some() && cf_token.is_some() && cf_zone_id.is_some();
        if configured {
            info!("ACME client configured (Let's Encrypt + Cloudflare DNS-01)");
        } else {
            debug!("ACME not configured — will use self-signed certificates");
        }

        Self {
            email,
            cf_token,
            cf_zone_id,
            http: reqwest::Client::new(),
            staging: std::env::var("ACME_STAGING").is_ok(),
        }
    }

    /// Whether all required env vars are set.
    pub fn is_configured(&self) -> bool {
        self.email.is_some() && self.cf_token.is_some() && self.cf_zone_id.is_some()
    }

    fn directory_url(&self) -> &str {
        if self.staging {
            LETS_ENCRYPT_STAGING
        } else {
            LETS_ENCRYPT_DIRECTORY
        }
    }

    /// Provision a TLS certificate for the given domains.
    /// Writes cert + key to `cert_path` and `key_path`.
    pub async fn provision(
        &self,
        domains: &[&str],
        cert_path: &Path,
        key_path: &Path,
    ) -> Result<(), AcmeError> {
        if !self.is_configured() {
            return Err(AcmeError::NotConfigured);
        }

        let email = self.email.as_deref().unwrap();
        let cf_token = self.cf_token.as_deref().unwrap();
        let cf_zone_id = self.cf_zone_id.as_deref().unwrap();

        info!(domains = ?domains, "Provisioning Let's Encrypt certificate");

        // Step 1: Fetch ACME directory
        let directory = self.fetch_directory().await?;
        let new_nonce_url = directory["newNonce"]
            .as_str()
            .ok_or_else(|| AcmeError::Protocol("missing newNonce URL".into()))?;
        let new_account_url = directory["newAccount"]
            .as_str()
            .ok_or_else(|| AcmeError::Protocol("missing newAccount URL".into()))?;
        let new_order_url = directory["newOrder"]
            .as_str()
            .ok_or_else(|| AcmeError::Protocol("missing newOrder URL".into()))?;

        // Step 2: Generate account key (ECDSA P-256)
        let account_key = rcgen::KeyPair::generate_for(&rcgen::PKCS_ECDSA_P256_SHA256)
            .map_err(|e| AcmeError::Certificate(e.to_string()))?;
        let jwk = account_key_jwk(&account_key)?;
        let thumbprint = jwk_thumbprint(&jwk)?;

        // Step 3: Get initial nonce
        let nonce = self.get_nonce(new_nonce_url).await?;

        // Step 4: Create/find account
        let payload = serde_json::json!({
            "termsOfServiceAgreed": true,
            "contact": [format!("mailto:{email}")]
        });
        let (resp, nonce) = self
            .signed_request(new_account_url, &payload, &account_key, &jwk, None, &nonce)
            .await?;
        let account_url = resp
            .headers()
            .get("location")
            .and_then(|v| v.to_str().ok())
            .ok_or_else(|| AcmeError::Protocol("no account URL in response".into()))?
            .to_string();
        debug!(account_url = %account_url, "ACME account created/found");

        // Step 5: Create order
        let identifiers: Vec<serde_json::Value> = domains
            .iter()
            .map(|d| serde_json::json!({"type": "dns", "value": d}))
            .collect();
        let payload = serde_json::json!({"identifiers": identifiers});
        let (resp, nonce) = self
            .signed_request(
                new_order_url,
                &payload,
                &account_key,
                &jwk,
                Some(&account_url),
                &nonce,
            )
            .await?;

        let order_url = resp
            .headers()
            .get("location")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();

        let order: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| AcmeError::Http(e.to_string()))?;

        let authz_urls: Vec<String> = order["authorizations"]
            .as_array()
            .ok_or_else(|| AcmeError::Protocol("no authorizations in order".into()))?
            .iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();

        let finalize_url = order["finalize"]
            .as_str()
            .ok_or_else(|| AcmeError::Protocol("no finalize URL".into()))?
            .to_string();

        // Step 6: Process each authorization (DNS-01 challenges)
        let mut nonce = nonce;
        let mut dns_records_to_cleanup: Vec<String> = Vec::new();

        for authz_url in &authz_urls {
            let (resp, new_nonce) = self
                .signed_request(
                    authz_url,
                    &serde_json::Value::String(String::new()),
                    &account_key,
                    &jwk,
                    Some(&account_url),
                    &nonce,
                )
                .await?;
            nonce = new_nonce;

            let authz: serde_json::Value = resp
                .json()
                .await
                .map_err(|e| AcmeError::Http(e.to_string()))?;

            let domain = authz["identifier"]["value"].as_str().unwrap_or("unknown");

            // Find dns-01 challenge
            let challenge = authz["challenges"]
                .as_array()
                .and_then(|cs| cs.iter().find(|c| c["type"] == "dns-01"))
                .ok_or_else(|| AcmeError::Protocol(format!("no dns-01 challenge for {domain}")))?;

            let token = challenge["token"]
                .as_str()
                .ok_or_else(|| AcmeError::Protocol("no token in challenge".into()))?;
            let challenge_url = challenge["url"]
                .as_str()
                .ok_or_else(|| AcmeError::Protocol("no URL in challenge".into()))?;

            // Compute key authorization and DNS TXT value
            let key_auth = format!("{token}.{thumbprint}");
            let digest = sha2::Sha256::digest(key_auth.as_bytes());
            let txt_value = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(digest);

            // Create DNS TXT record via Cloudflare
            let record_name = format!("_acme-challenge.{domain}");
            let record_id = self
                .cf_create_txt_record(cf_token, cf_zone_id, &record_name, &txt_value)
                .await?;
            dns_records_to_cleanup.push(record_id);

            // Wait for DNS propagation
            debug!(domain, "Waiting 10s for DNS propagation");
            tokio::time::sleep(std::time::Duration::from_secs(10)).await;

            // Respond to challenge
            let (_, new_nonce) = self
                .signed_request(
                    challenge_url,
                    &serde_json::json!({}),
                    &account_key,
                    &jwk,
                    Some(&account_url),
                    &nonce,
                )
                .await?;
            nonce = new_nonce;

            // Poll for validation (max 60s)
            for attempt in 0..12 {
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                let (resp, new_nonce) = self
                    .signed_request(
                        authz_url,
                        &serde_json::Value::String(String::new()),
                        &account_key,
                        &jwk,
                        Some(&account_url),
                        &nonce,
                    )
                    .await?;
                nonce = new_nonce;
                let status: serde_json::Value = resp
                    .json()
                    .await
                    .map_err(|e| AcmeError::Http(e.to_string()))?;
                let s = status["status"].as_str().unwrap_or("pending");
                debug!(domain, status = s, attempt, "Authorization poll");
                if s == "valid" {
                    break;
                }
                if s == "invalid" {
                    // Cleanup DNS records before returning error
                    for rid in &dns_records_to_cleanup {
                        let _ = self.cf_delete_txt_record(cf_token, cf_zone_id, rid).await;
                    }
                    return Err(AcmeError::Protocol(format!(
                        "authorization failed for {domain}"
                    )));
                }
                if attempt == 11 {
                    for rid in &dns_records_to_cleanup {
                        let _ = self.cf_delete_txt_record(cf_token, cf_zone_id, rid).await;
                    }
                    return Err(AcmeError::Timeout);
                }
            }
        }

        // Step 7: Generate certificate key and CSR
        let cert_key = rcgen::KeyPair::generate_for(&rcgen::PKCS_ECDSA_P256_SHA256)
            .map_err(|e| AcmeError::Certificate(e.to_string()))?;

        let mut csr_params = rcgen::CertificateParams::new(
            domains.iter().map(|d| d.to_string()).collect::<Vec<_>>(),
        )
        .map_err(|e| AcmeError::Certificate(e.to_string()))?;
        csr_params
            .distinguished_name
            .push(rcgen::DnType::CommonName, domains[0]);
        let csr_der = csr_params
            .serialize_request(&cert_key)
            .map_err(|e| AcmeError::Certificate(e.to_string()))?;
        let csr_b64 = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(csr_der.der());

        // Step 8: Finalize order
        let payload = serde_json::json!({"csr": csr_b64});
        let (resp, mut nonce) = self
            .signed_request(
                &finalize_url,
                &payload,
                &account_key,
                &jwk,
                Some(&account_url),
                &nonce,
            )
            .await?;
        let finalize_resp: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| AcmeError::Http(e.to_string()))?;

        // Step 9: Download certificate (poll order URL until status=valid)
        let cert_url = if let Some(url) = finalize_resp["certificate"].as_str() {
            url.to_string()
        } else {
            // Poll order URL
            let mut cert_url = None;
            for _ in 0..12 {
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                let (resp, new_nonce) = self
                    .signed_request(
                        &order_url,
                        &serde_json::Value::String(String::new()),
                        &account_key,
                        &jwk,
                        Some(&account_url),
                        &nonce,
                    )
                    .await?;
                nonce = new_nonce;
                let order: serde_json::Value = resp
                    .json()
                    .await
                    .map_err(|e| AcmeError::Http(e.to_string()))?;
                if let Some(url) = order["certificate"].as_str() {
                    cert_url = Some(url.to_string());
                    break;
                }
            }
            cert_url.ok_or(AcmeError::Timeout)?
        };

        // Download the certificate chain
        let (resp, _) = self
            .signed_request(
                &cert_url,
                &serde_json::Value::String(String::new()),
                &account_key,
                &jwk,
                Some(&account_url),
                &nonce,
            )
            .await?;
        let cert_pem = resp
            .text()
            .await
            .map_err(|e| AcmeError::Http(e.to_string()))?;

        // Write cert and key
        std::fs::write(cert_path, &cert_pem)?;
        std::fs::write(key_path, cert_key.serialize_pem())?;

        info!(
            domains = ?domains,
            cert_path = %cert_path.display(),
            "Let's Encrypt certificate provisioned"
        );

        // Cleanup DNS records
        let cf_token = self.cf_token.as_deref().unwrap();
        let cf_zone_id = self.cf_zone_id.as_deref().unwrap();
        for record_id in &dns_records_to_cleanup {
            if let Err(e) = self
                .cf_delete_txt_record(cf_token, cf_zone_id, record_id)
                .await
            {
                warn!(error = %e, "Failed to cleanup ACME DNS record");
            }
        }

        Ok(())
    }

    /// Check if an existing cert expires within `days` days.
    pub fn cert_expires_within(cert_path: &Path, days: u32) -> bool {
        let Ok(pem_data) = std::fs::read(cert_path) else {
            return true; // No cert = needs provisioning
        };

        let Ok(Some(cert_der)) = rustls_pemfile::certs(&mut &pem_data[..]).next().transpose()
        else {
            return true;
        };

        let Ok((_, cert)) = x509_parser::parse_x509_certificate(&cert_der) else {
            return true;
        };

        let expires_at = cert.validity().not_after.timestamp();
        let now = chrono::Utc::now().timestamp();
        let remaining_days = (expires_at - now) / 86400;

        remaining_days < days as i64
    }

    /// Check if the cert at the given path is self-signed.
    pub fn is_self_signed(cert_path: &Path) -> bool {
        let Ok(pem_data) = std::fs::read(cert_path) else {
            return true;
        };

        let Ok(Some(cert_der)) = rustls_pemfile::certs(&mut &pem_data[..]).next().transpose()
        else {
            return true;
        };

        let Ok((_, cert)) = x509_parser::parse_x509_certificate(&cert_der) else {
            return true;
        };

        cert.issuer() == cert.subject()
    }

    // ── Internal helpers ──────────────────────────────────────────────────

    async fn fetch_directory(&self) -> Result<serde_json::Value, AcmeError> {
        self.http
            .get(self.directory_url())
            .send()
            .await
            .map_err(|e| AcmeError::Http(e.to_string()))?
            .json()
            .await
            .map_err(|e| AcmeError::Http(e.to_string()))
    }

    async fn get_nonce(&self, url: &str) -> Result<String, AcmeError> {
        let resp = self
            .http
            .head(url)
            .send()
            .await
            .map_err(|e| AcmeError::Http(e.to_string()))?;
        resp.headers()
            .get("replay-nonce")
            .and_then(|v| v.to_str().ok())
            .map(String::from)
            .ok_or_else(|| AcmeError::Protocol("no replay-nonce header".into()))
    }

    async fn signed_request(
        &self,
        url: &str,
        payload: &serde_json::Value,
        key: &rcgen::KeyPair,
        jwk: &serde_json::Value,
        kid: Option<&str>,
        nonce: &str,
    ) -> Result<(reqwest::Response, String), AcmeError> {
        let b64 = base64::engine::general_purpose::URL_SAFE_NO_PAD;

        // Protected header
        let mut protected = serde_json::json!({
            "alg": "ES256",
            "nonce": nonce,
            "url": url,
        });
        if let Some(kid) = kid {
            protected["kid"] = serde_json::Value::String(kid.to_string());
        } else {
            protected["jwk"] = jwk.clone();
        }
        let protected_b64 = b64.encode(serde_json::to_vec(&protected).unwrap());

        // Payload (empty string = POST-as-GET)
        let payload_b64 = if payload.is_string() && payload.as_str() == Some("") {
            String::new()
        } else {
            b64.encode(serde_json::to_vec(payload).unwrap())
        };

        // Sign using aws-lc-rs ECDSA (rcgen's sign() is private)
        let signing_input = format!("{protected_b64}.{payload_b64}");
        let key_der = key.serialize_der();
        let signing_key = aws_lc_rs::signature::EcdsaKeyPair::from_pkcs8(
            &aws_lc_rs::signature::ECDSA_P256_SHA256_FIXED_SIGNING,
            &key_der,
        )
        .map_err(|e| AcmeError::Certificate(format!("key parse error: {e}")))?;
        let sig = signing_key
            .sign(
                &aws_lc_rs::rand::SystemRandom::new(),
                signing_input.as_bytes(),
            )
            .map_err(|e| AcmeError::Certificate(format!("sign error: {e}")))?;
        let sig_b64 = b64.encode(sig.as_ref());

        let body = serde_json::json!({
            "protected": protected_b64,
            "payload": payload_b64,
            "signature": sig_b64,
        });

        let resp = self
            .http
            .post(url)
            .header("content-type", "application/jose+json")
            .json(&body)
            .send()
            .await
            .map_err(|e| AcmeError::Http(e.to_string()))?;

        let new_nonce = resp
            .headers()
            .get("replay-nonce")
            .and_then(|v| v.to_str().ok())
            .unwrap_or(nonce)
            .to_string();

        Ok((resp, new_nonce))
    }

    async fn cf_create_txt_record(
        &self,
        token: &str,
        zone_id: &str,
        name: &str,
        value: &str,
    ) -> Result<String, AcmeError> {
        let url = format!("https://api.cloudflare.com/client/v4/zones/{zone_id}/dns_records");
        let resp = self
            .http
            .post(&url)
            .bearer_auth(token)
            .json(&serde_json::json!({
                "type": "TXT",
                "name": name,
                "content": value,
                "ttl": 60,
            }))
            .send()
            .await
            .map_err(|e| AcmeError::Dns(e.to_string()))?;

        let body: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| AcmeError::Dns(e.to_string()))?;

        if body["success"].as_bool() != Some(true) {
            return Err(AcmeError::Dns(format!(
                "Cloudflare API error: {}",
                body["errors"]
            )));
        }

        body["result"]["id"]
            .as_str()
            .map(String::from)
            .ok_or_else(|| AcmeError::Dns("no record ID in response".into()))
    }

    async fn cf_delete_txt_record(
        &self,
        token: &str,
        zone_id: &str,
        record_id: &str,
    ) -> Result<(), AcmeError> {
        let url =
            format!("https://api.cloudflare.com/client/v4/zones/{zone_id}/dns_records/{record_id}");
        self.http
            .delete(&url)
            .bearer_auth(token)
            .send()
            .await
            .map_err(|e| AcmeError::Dns(e.to_string()))?;
        Ok(())
    }
}

/// Extract the raw ECDSA public key coordinates from an rcgen KeyPair
/// and format as a JWK.
fn account_key_jwk(key: &rcgen::KeyPair) -> Result<serde_json::Value, AcmeError> {
    let b64 = base64::engine::general_purpose::URL_SAFE_NO_PAD;

    // rcgen's public_key_raw() returns the uncompressed point (0x04 || x || y)
    let raw = key.public_key_raw();
    if raw.len() != 65 || raw[0] != 0x04 {
        return Err(AcmeError::Certificate(
            "unexpected public key format (expected uncompressed P-256)".into(),
        ));
    }
    let x = b64.encode(&raw[1..33]);
    let y = b64.encode(&raw[33..65]);

    Ok(serde_json::json!({
        "kty": "EC",
        "crv": "P-256",
        "x": x,
        "y": y,
    }))
}

/// Compute JWK Thumbprint (RFC 7638) for the account key.
fn jwk_thumbprint(jwk: &serde_json::Value) -> Result<String, AcmeError> {
    let b64 = base64::engine::general_purpose::URL_SAFE_NO_PAD;
    // For EC keys, the canonical form is {"crv":"P-256","kty":"EC","x":"...","y":"..."}
    let canonical = format!(
        r#"{{"crv":"P-256","kty":"EC","x":"{}","y":"{}"}}"#,
        jwk["x"].as_str().unwrap_or(""),
        jwk["y"].as_str().unwrap_or(""),
    );
    let digest = sha2::Sha256::digest(canonical.as_bytes());
    Ok(b64.encode(digest))
}

/// Convert a DER-encoded ECDSA signature to raw r||s (64 bytes for P-256).
#[cfg(test)]
fn der_sig_to_raw(der: &[u8]) -> Result<Vec<u8>, AcmeError> {
    // DER: 0x30 <len> 0x02 <r_len> <r> 0x02 <s_len> <s>
    if der.len() < 6 || der[0] != 0x30 {
        return Err(AcmeError::Certificate("invalid DER signature".into()));
    }

    let mut pos = 2; // skip 0x30 <total_len>
    if pos >= der.len() || der[pos] != 0x02 {
        return Err(AcmeError::Certificate("missing r integer".into()));
    }
    pos += 1;
    let r_len = der[pos] as usize;
    pos += 1;
    let r = &der[pos..pos + r_len];
    pos += r_len;

    if pos >= der.len() || der[pos] != 0x02 {
        return Err(AcmeError::Certificate("missing s integer".into()));
    }
    pos += 1;
    let s_len = der[pos] as usize;
    pos += 1;
    let s = &der[pos..pos + s_len];

    // Pad/trim to 32 bytes each
    let mut raw = vec![0u8; 64];
    let r_trimmed = if r.len() > 32 { &r[r.len() - 32..] } else { r };
    let s_trimmed = if s.len() > 32 { &s[s.len() - 32..] } else { s };
    raw[32 - r_trimmed.len()..32].copy_from_slice(r_trimmed);
    raw[64 - s_trimmed.len()..64].copy_from_slice(s_trimmed);

    Ok(raw)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn client_unconfigured_without_env() {
        let client = AcmeClient {
            email: None,
            cf_token: None,
            cf_zone_id: None,
            http: reqwest::Client::new(),
            staging: false,
        };
        assert!(!client.is_configured());
    }

    #[test]
    fn client_configured_with_all_vars() {
        let client = AcmeClient {
            email: Some("admin@openie.sh".into()),
            cf_token: Some("cf-token".into()),
            cf_zone_id: Some("zone-123".into()),
            http: reqwest::Client::new(),
            staging: false,
        };
        assert!(client.is_configured());
    }

    #[test]
    fn staging_uses_staging_directory() {
        let client = AcmeClient {
            email: None,
            cf_token: None,
            cf_zone_id: None,
            http: reqwest::Client::new(),
            staging: true,
        };
        assert_eq!(client.directory_url(), LETS_ENCRYPT_STAGING);
    }

    #[test]
    fn production_uses_production_directory() {
        let client = AcmeClient {
            email: None,
            cf_token: None,
            cf_zone_id: None,
            http: reqwest::Client::new(),
            staging: false,
        };
        assert_eq!(client.directory_url(), LETS_ENCRYPT_DIRECTORY);
    }

    #[test]
    fn cert_expires_within_returns_true_for_missing_file() {
        assert!(AcmeClient::cert_expires_within(
            Path::new("/nonexistent/cert.pem"),
            30
        ));
    }

    #[test]
    fn is_self_signed_returns_true_for_missing_file() {
        assert!(AcmeClient::is_self_signed(Path::new(
            "/nonexistent/cert.pem"
        )));
    }

    #[test]
    fn jwk_thumbprint_deterministic() {
        let jwk = serde_json::json!({
            "kty": "EC",
            "crv": "P-256",
            "x": "f83OJ3D2xF1Bg8vub9tLe1gHMzV76e8Tus9uPHvRVEU",
            "y": "x_FEzRu9m36HLN_tue659LNpXW6pCyStikYjKIWI5a0",
        });
        let tp = jwk_thumbprint(&jwk).unwrap();
        assert!(!tp.is_empty());
        // Thumbprint should be deterministic
        let tp2 = jwk_thumbprint(&jwk).unwrap();
        assert_eq!(tp, tp2);
    }

    #[test]
    fn der_sig_to_raw_valid() {
        // A minimal valid DER ECDSA signature
        let mut der = vec![0x30, 0x06, 0x02, 0x01, 0x01, 0x02, 0x01, 0x02];
        let raw = der_sig_to_raw(&der).unwrap();
        assert_eq!(raw.len(), 64);
        assert_eq!(raw[31], 1); // r = 1
        assert_eq!(raw[63], 2); // s = 2
    }
}
