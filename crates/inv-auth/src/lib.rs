pub mod apikey;
pub mod attestation_auth;
pub mod audit;
pub mod email;
pub mod jwt;
pub mod mfa;
pub mod oidc;
pub mod passkey;
pub mod rbac;
pub mod revocation;
pub mod session;
pub mod webauthn;

pub use apikey::{ApiKey, ApiKeyCreateResult, ApiKeyService};
pub use attestation_auth::{
    AttestationAuthConfig, AttestationAuthError, AttestationAuthenticator, AttestationToken,
    upgrade_trust_from_attestation,
};
pub use audit::{AuditAction, AuditEntry, AuditLog, AuditOutcome, AuditQuery, AuditSink};
pub use email::{
    DevEmailSender, EmailError, EmailMessage, EmailSender, SmtpConfig, SmtpEmailSender,
};
pub use jwt::{Claims, TokenPair, TokenService, TokenServiceConfig};
pub use mfa::{MfaPending, MfaService, MfaVerifyResult};
pub use oidc::{
    OidcClaims, OidcDiscovery, OidcError, OidcIdentity, OidcProviderConfig, OidcService,
    RoleMapping,
};
pub use passkey::{
    AuthorizationCode, EmailOtp, MagicLinkConfig, MagicLinkToken, OtpConfig, PasskeyAccount,
    PasskeyCredential, PasskeyError, PasskeyService, PasskeyStats, WebAuthnChallenge,
};
pub use rbac::{Permission, Role};
pub use revocation::{RevocationService, RevocationStats};
pub use session::{Session, SessionManager};
pub use webauthn::{AuthenticatorData, CoseAlgorithm, VerifyError, verify_assertion, verify_eddsa};

use thiserror::Error;

/// Authentication and authorization errors.
#[derive(Debug, Error)]
pub enum AuthError {
    #[error("invalid credentials")]
    InvalidCredentials,

    #[error("token expired")]
    TokenExpired,

    #[error("invalid token: {0}")]
    InvalidToken(String),

    #[error("missing authorization header")]
    MissingAuth,

    #[error("insufficient permissions: requires {required}")]
    InsufficientPermissions { required: Permission },

    #[error("org mismatch: token org {token_org} does not match requested org {requested_org}")]
    OrgMismatch {
        token_org: String,
        requested_org: String,
    },

    #[error("API key not found")]
    ApiKeyNotFound,

    #[error("API key expired")]
    ApiKeyExpired,

    #[error("API key revoked")]
    ApiKeyRevoked,

    #[error("session not found")]
    SessionNotFound,

    #[error("token has been revoked")]
    TokenRevoked,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_auth_flow() {
        let config = TokenServiceConfig {
            secret: b"test-secret-key-for-invisible-infra".to_vec(),
            access_token_ttl_secs: 3600,
            refresh_token_ttl_secs: 86400,
            issuer: "invisible".into(),
        };
        let svc = TokenService::new(config);

        // Issue tokens
        let pair = svc
            .issue("acme", "user-1", Role::Operator, &["us-east".into()])
            .unwrap();

        // Validate access token
        let claims = svc.validate(&pair.access_token).unwrap();
        assert_eq!(claims.org, "acme");
        assert_eq!(claims.sub, "user-1");
        assert_eq!(claims.role, Role::Operator);

        // Operator can deploy but cannot manage org
        assert!(claims.has_permission(Permission::WorkloadDeploy));
        assert!(claims.has_permission(Permission::NodeRegister));
        assert!(!claims.has_permission(Permission::OrgManage));

        // Refresh
        let new_pair = svc.refresh(&pair.refresh_token).unwrap();
        let new_claims = svc.validate(&new_pair.access_token).unwrap();
        assert_eq!(new_claims.org, "acme");
    }

    #[test]
    fn admin_has_all_permissions() {
        let config = TokenServiceConfig {
            secret: b"test-secret".to_vec(),
            access_token_ttl_secs: 3600,
            refresh_token_ttl_secs: 86400,
            issuer: "invisible".into(),
        };
        let svc = TokenService::new(config);

        let pair = svc.issue("acme", "admin-1", Role::Admin, &[]).unwrap();
        let claims = svc.validate(&pair.access_token).unwrap();

        for perm in Permission::all() {
            assert!(claims.has_permission(*perm), "admin should have {perm}");
        }
    }

    #[test]
    fn viewer_cannot_write() {
        let config = TokenServiceConfig {
            secret: b"test-secret".to_vec(),
            access_token_ttl_secs: 3600,
            refresh_token_ttl_secs: 86400,
            issuer: "invisible".into(),
        };
        let svc = TokenService::new(config);

        let pair = svc.issue("acme", "viewer-1", Role::Viewer, &[]).unwrap();
        let claims = svc.validate(&pair.access_token).unwrap();

        assert!(claims.has_permission(Permission::NodeRead));
        assert!(claims.has_permission(Permission::WorkloadRead));
        assert!(claims.has_permission(Permission::EnergyRead));
        assert!(!claims.has_permission(Permission::WorkloadDeploy));
        assert!(!claims.has_permission(Permission::NodeRegister));
    }

    #[test]
    fn invalid_token_rejected() {
        let config = TokenServiceConfig {
            secret: b"test-secret".to_vec(),
            access_token_ttl_secs: 3600,
            refresh_token_ttl_secs: 86400,
            issuer: "invisible".into(),
        };
        let svc = TokenService::new(config);

        let result = svc.validate("not-a-valid-token");
        assert!(result.is_err());
    }

    #[test]
    fn wrong_secret_rejected() {
        let config1 = TokenServiceConfig {
            secret: b"secret-one".to_vec(),
            access_token_ttl_secs: 3600,
            refresh_token_ttl_secs: 86400,
            issuer: "invisible".into(),
        };
        let config2 = TokenServiceConfig {
            secret: b"secret-two".to_vec(),
            access_token_ttl_secs: 3600,
            refresh_token_ttl_secs: 86400,
            issuer: "invisible".into(),
        };

        let svc1 = TokenService::new(config1);
        let svc2 = TokenService::new(config2);

        let pair = svc1.issue("acme", "user-1", Role::Admin, &[]).unwrap();
        let result = svc2.validate(&pair.access_token);
        assert!(result.is_err());
    }

    #[test]
    fn org_isolation() {
        let config = TokenServiceConfig {
            secret: b"test-secret".to_vec(),
            access_token_ttl_secs: 3600,
            refresh_token_ttl_secs: 86400,
            issuer: "invisible".into(),
        };
        let svc = TokenService::new(config);

        let pair = svc.issue("acme", "user-1", Role::Admin, &[]).unwrap();
        let claims = svc.validate(&pair.access_token).unwrap();

        // Correct org
        assert!(claims.verify_org("acme").is_ok());

        // Wrong org
        assert!(claims.verify_org("other-org").is_err());
    }

    #[test]
    fn error_display() {
        let err = AuthError::ApiKeyNotFound;
        assert_eq!(err.to_string(), "API key not found");

        let err = AuthError::ApiKeyExpired;
        assert_eq!(err.to_string(), "API key expired");

        let err = AuthError::ApiKeyRevoked;
        assert_eq!(err.to_string(), "API key revoked");

        let err = AuthError::SessionNotFound;
        assert_eq!(err.to_string(), "session not found");

        let err = AuthError::TokenRevoked;
        assert_eq!(err.to_string(), "token has been revoked");
    }

    #[test]
    fn validate_with_revocation() {
        let config = TokenServiceConfig {
            secret: b"test-secret-for-revocation".to_vec(),
            access_token_ttl_secs: 3600,
            refresh_token_ttl_secs: 86400,
            issuer: "invisible".into(),
        };
        let svc = TokenService::new(config);
        let revocation = RevocationService::new();

        // Issue a token
        let pair = svc.issue("acme", "user-1", Role::Operator, &[]).unwrap();

        // Validate it normally
        let claims = svc.validate(&pair.access_token).unwrap();
        assert!(!revocation.is_revoked(&claims.jti));

        // Revoke it
        revocation.revoke_token(&claims.jti, claims.exp);

        // Now validate should still decode, but revocation check catches it
        let claims = svc.validate(&pair.access_token).unwrap();
        assert!(revocation.is_revoked(&claims.jti));

        // Using validate_with_revocation
        let result = svc.validate_with_revocation(&pair.access_token, &revocation);
        assert!(matches!(result, Err(AuthError::TokenRevoked)));
    }
}
