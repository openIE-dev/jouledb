use jsonwebtoken::{DecodingKey, EncodingKey, Header, Validation, decode, encode};
use serde::{Deserialize, Serialize};

use crate::AuthError;
use crate::rbac::{Permission, Role};
use crate::revocation::RevocationService;

/// Configuration for the token service.
#[derive(Clone)]
pub struct TokenServiceConfig {
    /// HMAC secret for signing tokens.
    pub secret: Vec<u8>,
    /// Access token time-to-live in seconds.
    pub access_token_ttl_secs: u64,
    /// Refresh token time-to-live in seconds.
    pub refresh_token_ttl_secs: u64,
    /// Token issuer (e.g., "invisible").
    pub issuer: String,
}

/// A pair of access + refresh tokens.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenPair {
    pub access_token: String,
    pub refresh_token: String,
    pub expires_in: u64,
    pub token_type: String,
}

/// JWT claims for Invisible Infrastructure tokens.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claims {
    /// Subject (user or agent ID).
    pub sub: String,
    /// Organization ID — all resource access is scoped to this org.
    pub org: String,
    /// Role determines available permissions.
    pub role: Role,
    /// Allowed regions (empty = all regions).
    #[serde(default)]
    pub regions: Vec<String>,
    /// Token type: "access" or "refresh".
    pub token_type: String,
    /// Issuer.
    pub iss: String,
    /// Issued at (unix timestamp).
    pub iat: u64,
    /// Expiration (unix timestamp).
    pub exp: u64,
    /// JWT ID for revocation tracking.
    pub jti: String,
    /// Optional client fingerprint for token binding.
    ///
    /// When present, the token can only be used from a client whose
    /// IP + User-Agent hash matches this fingerprint. Limits stolen token
    /// utility. SHA-256(client_ip + ":" + user_agent).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cnf: Option<String>,
}

impl Claims {
    /// Check if these claims grant a specific permission.
    pub fn has_permission(&self, perm: Permission) -> bool {
        self.role.has_permission(perm)
    }

    /// Verify that the claims are for the expected org.
    pub fn verify_org(&self, expected_org: &str) -> Result<(), AuthError> {
        if self.org == expected_org {
            Ok(())
        } else {
            Err(AuthError::OrgMismatch {
                token_org: self.org.clone(),
                requested_org: expected_org.to_string(),
            })
        }
    }

    /// Check if access to a specific region is allowed.
    /// Empty regions list means all regions are allowed.
    pub fn can_access_region(&self, region: &str) -> bool {
        self.regions.is_empty() || self.regions.iter().any(|r| r == region)
    }
}

/// Service for issuing and validating JWT tokens.
pub struct TokenService {
    encoding_key: EncodingKey,
    decoding_key: DecodingKey,
    config: TokenServiceConfig,
}

impl TokenService {
    pub fn new(config: TokenServiceConfig) -> Self {
        let encoding_key = EncodingKey::from_secret(&config.secret);
        let decoding_key = DecodingKey::from_secret(&config.secret);
        Self {
            encoding_key,
            decoding_key,
            config,
        }
    }

    /// Issue an access + refresh token pair.
    pub fn issue(
        &self,
        org: &str,
        subject: &str,
        role: Role,
        regions: &[String],
    ) -> Result<TokenPair, AuthError> {
        let now = current_timestamp();

        let access_claims = Claims {
            sub: subject.to_string(),
            org: org.to_string(),
            role,
            regions: regions.to_vec(),
            token_type: "access".into(),
            iss: self.config.issuer.clone(),
            iat: now,
            exp: now + self.config.access_token_ttl_secs,
            jti: uuid::Uuid::new_v4().to_string(),
            cnf: None,
        };

        let refresh_claims = Claims {
            sub: subject.to_string(),
            org: org.to_string(),
            role,
            regions: regions.to_vec(),
            token_type: "refresh".into(),
            iss: self.config.issuer.clone(),
            iat: now,
            exp: now + self.config.refresh_token_ttl_secs,
            jti: uuid::Uuid::new_v4().to_string(),
            cnf: None,
        };

        let access_token = encode(&Header::default(), &access_claims, &self.encoding_key)
            .map_err(|e| AuthError::InvalidToken(e.to_string()))?;

        let refresh_token = encode(&Header::default(), &refresh_claims, &self.encoding_key)
            .map_err(|e| AuthError::InvalidToken(e.to_string()))?;

        Ok(TokenPair {
            access_token,
            refresh_token,
            expires_in: self.config.access_token_ttl_secs,
            token_type: "Bearer".into(),
        })
    }

    /// Validate a token and return its claims.
    pub fn validate(&self, token: &str) -> Result<Claims, AuthError> {
        let mut validation = Validation::default();
        validation.set_issuer(&[&self.config.issuer]);
        validation.set_required_spec_claims(&["exp", "iss", "sub"]);
        validation.leeway = 0;

        let token_data = decode::<Claims>(token, &self.decoding_key, &validation).map_err(|e| {
            match e.kind() {
                jsonwebtoken::errors::ErrorKind::ExpiredSignature => AuthError::TokenExpired,
                _ => AuthError::InvalidToken(e.to_string()),
            }
        })?;

        Ok(token_data.claims)
    }

    /// Validate a token and check it against the revocation list.
    ///
    /// This combines JWT validation with revocation checking in a single call.
    pub fn validate_with_revocation(
        &self,
        token: &str,
        revocation: &RevocationService,
    ) -> Result<Claims, AuthError> {
        let claims = self.validate(token)?;
        if revocation.is_revoked(&claims.jti) {
            return Err(AuthError::TokenRevoked);
        }
        Ok(claims)
    }

    /// Refresh an access token using a refresh token.
    pub fn refresh(&self, refresh_token: &str) -> Result<TokenPair, AuthError> {
        let claims = self.validate(refresh_token)?;

        if claims.token_type != "refresh" {
            return Err(AuthError::InvalidToken(
                "expected refresh token".to_string(),
            ));
        }

        self.issue(&claims.org, &claims.sub, claims.role, &claims.regions)
    }
}

fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_service() -> TokenService {
        TokenService::new(TokenServiceConfig {
            secret: b"test-secret-for-jwt-testing".to_vec(),
            access_token_ttl_secs: 3600,
            refresh_token_ttl_secs: 86400,
            issuer: "invisible-test".into(),
        })
    }

    #[test]
    fn issue_and_validate() {
        let svc = test_service();
        let pair = svc.issue("acme", "user-1", Role::Admin, &[]).unwrap();

        let claims = svc.validate(&pair.access_token).unwrap();
        assert_eq!(claims.sub, "user-1");
        assert_eq!(claims.org, "acme");
        assert_eq!(claims.role, Role::Admin);
        assert_eq!(claims.token_type, "access");
        assert_eq!(claims.iss, "invisible-test");
        assert!(!claims.jti.is_empty());
    }

    #[test]
    fn refresh_flow() {
        let svc = test_service();
        let pair = svc
            .issue("acme", "user-1", Role::Operator, &["us-east".into()])
            .unwrap();

        let new_pair = svc.refresh(&pair.refresh_token).unwrap();
        let claims = svc.validate(&new_pair.access_token).unwrap();
        assert_eq!(claims.org, "acme");
        assert_eq!(claims.sub, "user-1");
        assert_eq!(claims.role, Role::Operator);
        assert_eq!(claims.regions, vec!["us-east"]);
    }

    #[test]
    fn cannot_refresh_with_access_token() {
        let svc = test_service();
        let pair = svc.issue("acme", "user-1", Role::Admin, &[]).unwrap();

        let result = svc.refresh(&pair.access_token);
        assert!(result.is_err());
    }

    #[test]
    fn region_access() {
        let svc = test_service();

        // With specific regions
        let pair = svc
            .issue(
                "acme",
                "user-1",
                Role::Admin,
                &["us-east".into(), "eu-west".into()],
            )
            .unwrap();
        let claims = svc.validate(&pair.access_token).unwrap();
        assert!(claims.can_access_region("us-east"));
        assert!(claims.can_access_region("eu-west"));
        assert!(!claims.can_access_region("ap-south"));

        // With empty regions (all allowed)
        let pair = svc.issue("acme", "user-1", Role::Admin, &[]).unwrap();
        let claims = svc.validate(&pair.access_token).unwrap();
        assert!(claims.can_access_region("any-region"));
    }

    #[test]
    fn token_pair_fields() {
        let svc = test_service();
        let pair = svc.issue("acme", "user-1", Role::Admin, &[]).unwrap();

        assert_eq!(pair.token_type, "Bearer");
        assert_eq!(pair.expires_in, 3600);
        assert!(!pair.access_token.is_empty());
        assert!(!pair.refresh_token.is_empty());
        assert_ne!(pair.access_token, pair.refresh_token);
    }

    #[test]
    fn expired_token_rejected() {
        let config = TokenServiceConfig {
            secret: b"test-secret".to_vec(),
            access_token_ttl_secs: 0, // expires immediately
            refresh_token_ttl_secs: 0,
            issuer: "invisible-test".into(),
        };
        let svc = TokenService::new(config);

        let pair = svc.issue("acme", "user-1", Role::Admin, &[]).unwrap();

        // Sleep briefly to ensure expiration
        std::thread::sleep(std::time::Duration::from_millis(1100));

        let result = svc.validate(&pair.access_token);
        assert!(matches!(result, Err(AuthError::TokenExpired)));
    }
}
