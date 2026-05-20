//! Security headers — HSTS, X-Frame-Options, X-Content-Type-Options,
//! Referrer-Policy, Permissions-Policy, header builder, and per-route overrides.
//!
//! Replaces `helmet`, `koa-helmet`, and similar Node.js security header middleware
//! with a pure-Rust security header builder supporting per-route configuration.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── Errors ─────────────────────────────────────────────────────

/// Security header errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SecurityHeaderError {
    /// Invalid header value.
    InvalidValue { header: String, value: String },
    /// Conflicting configuration.
    Conflict(String),
    /// Unknown header name.
    UnknownHeader(String),
}

impl fmt::Display for SecurityHeaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidValue { header, value } => {
                write!(f, "invalid value for {header}: {value}")
            }
            Self::Conflict(msg) => write!(f, "conflicting header config: {msg}"),
            Self::UnknownHeader(name) => write!(f, "unknown security header: {name}"),
        }
    }
}

impl std::error::Error for SecurityHeaderError {}

// ── HSTS ───────────────────────────────────────────────────────

/// Strict-Transport-Security configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HstsConfig {
    /// Max age in seconds.
    pub max_age_seconds: u64,
    /// Include subdomains.
    pub include_subdomains: bool,
    /// Preload flag.
    pub preload: bool,
}

impl Default for HstsConfig {
    fn default() -> Self {
        Self {
            max_age_seconds: 31_536_000, // 1 year
            include_subdomains: true,
            preload: false,
        }
    }
}

impl HstsConfig {
    pub fn to_header_value(&self) -> String {
        let mut val = format!("max-age={}", self.max_age_seconds);
        if self.include_subdomains {
            val.push_str("; includeSubDomains");
        }
        if self.preload {
            val.push_str("; preload");
        }
        val
    }

    pub fn validate(&self) -> Result<(), SecurityHeaderError> {
        // Preload requires includeSubDomains and max-age >= 31536000
        if self.preload {
            if !self.include_subdomains {
                return Err(SecurityHeaderError::Conflict(
                    "HSTS preload requires includeSubDomains".into(),
                ));
            }
            if self.max_age_seconds < 31_536_000 {
                return Err(SecurityHeaderError::InvalidValue {
                    header: "Strict-Transport-Security".into(),
                    value: format!(
                        "preload requires max-age >= 31536000, got {}",
                        self.max_age_seconds
                    ),
                });
            }
        }
        Ok(())
    }
}

// ── X-Frame-Options ────────────────────────────────────────────

/// X-Frame-Options values.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum XFrameOptions {
    /// Page cannot be displayed in a frame.
    Deny,
    /// Page can only be displayed in a frame on the same origin.
    SameOrigin,
    /// Page can be displayed in a frame on the specified origin.
    AllowFrom(String),
}

impl XFrameOptions {
    pub fn to_header_value(&self) -> String {
        match self {
            Self::Deny => "DENY".to_string(),
            Self::SameOrigin => "SAMEORIGIN".to_string(),
            Self::AllowFrom(origin) => format!("ALLOW-FROM {origin}"),
        }
    }
}

// ── Referrer-Policy ────────────────────────────────────────────

/// Referrer-Policy values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReferrerPolicy {
    NoReferrer,
    NoReferrerWhenDowngrade,
    Origin,
    OriginWhenCrossOrigin,
    SameOrigin,
    StrictOrigin,
    StrictOriginWhenCrossOrigin,
    UnsafeUrl,
}

impl ReferrerPolicy {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::NoReferrer => "no-referrer",
            Self::NoReferrerWhenDowngrade => "no-referrer-when-downgrade",
            Self::Origin => "origin",
            Self::OriginWhenCrossOrigin => "origin-when-cross-origin",
            Self::SameOrigin => "same-origin",
            Self::StrictOrigin => "strict-origin",
            Self::StrictOriginWhenCrossOrigin => "strict-origin-when-cross-origin",
            Self::UnsafeUrl => "unsafe-url",
        }
    }
}

impl fmt::Display for ReferrerPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ── Permissions-Policy ─────────────────────────────────────────

/// A Permissions-Policy feature and its allowlist.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PermissionFeature {
    /// Feature name (e.g., "camera", "microphone", "geolocation").
    pub name: String,
    /// Allowlist.
    pub allowlist: PermissionAllowlist,
}

/// Allowlist for a Permissions-Policy feature.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PermissionAllowlist {
    /// Deny all (empty allowlist).
    None,
    /// Allow self only.
    CSelf,
    /// Allow all origins.
    All,
    /// Specific origins.
    Origins(Vec<String>),
}

impl PermissionAllowlist {
    pub fn to_policy_value(&self) -> String {
        match self {
            Self::None => "()".to_string(),
            Self::CSelf => "(self)".to_string(),
            Self::All => "*".to_string(),
            Self::Origins(origins) => {
                let quoted: Vec<String> = origins.iter().map(|o| format!("\"{o}\"")).collect();
                format!("({})", quoted.join(" "))
            }
        }
    }
}

/// Permissions-Policy builder.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PermissionsPolicy {
    pub features: Vec<PermissionFeature>,
}

impl PermissionsPolicy {
    pub fn new() -> Self {
        Self { features: vec![] }
    }

    /// Add a feature policy.
    pub fn add(&mut self, name: &str, allowlist: PermissionAllowlist) -> &mut Self {
        self.features.push(PermissionFeature {
            name: name.to_string(),
            allowlist,
        });
        self
    }

    /// Deny a feature entirely.
    pub fn deny(&mut self, name: &str) -> &mut Self {
        self.add(name, PermissionAllowlist::None)
    }

    /// Allow a feature for self only.
    pub fn allow_self(&mut self, name: &str) -> &mut Self {
        self.add(name, PermissionAllowlist::CSelf)
    }

    /// Build the Permissions-Policy header value.
    pub fn to_header_value(&self) -> String {
        self.features
            .iter()
            .map(|f| format!("{}={}", f.name, f.allowlist.to_policy_value()))
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// Create a strict default policy denying most features.
    pub fn strict() -> Self {
        let mut p = Self::new();
        p.deny("camera");
        p.deny("microphone");
        p.deny("geolocation");
        p.deny("payment");
        p.deny("usb");
        p.deny("magnetometer");
        p.deny("gyroscope");
        p.deny("accelerometer");
        p
    }
}

// ── Security Header Collection ─────────────────────────────────

/// A named header (name, value).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SecurityHeader {
    pub name: String,
    pub value: String,
}

impl SecurityHeader {
    pub fn new(name: &str, value: &str) -> Self {
        Self {
            name: name.to_string(),
            value: value.to_string(),
        }
    }
}

/// Security header set builder.
#[derive(Debug, Clone)]
pub struct SecurityHeaders {
    pub hsts: Option<HstsConfig>,
    pub x_frame_options: Option<XFrameOptions>,
    pub x_content_type_options: bool,
    pub x_xss_protection: Option<String>,
    pub referrer_policy: Option<ReferrerPolicy>,
    pub permissions_policy: Option<PermissionsPolicy>,
    pub cross_origin_opener_policy: Option<String>,
    pub cross_origin_embedder_policy: Option<String>,
    pub cross_origin_resource_policy: Option<String>,
    pub x_dns_prefetch_control: Option<bool>,
    pub x_download_options: bool,
    pub x_permitted_cross_domain_policies: Option<String>,
    pub custom: Vec<SecurityHeader>,
}

impl Default for SecurityHeaders {
    fn default() -> Self {
        Self::recommended()
    }
}

impl SecurityHeaders {
    /// Empty security headers (nothing set).
    pub fn none() -> Self {
        Self {
            hsts: None,
            x_frame_options: None,
            x_content_type_options: false,
            x_xss_protection: None,
            referrer_policy: None,
            permissions_policy: None,
            cross_origin_opener_policy: None,
            cross_origin_embedder_policy: None,
            cross_origin_resource_policy: None,
            x_dns_prefetch_control: None,
            x_download_options: false,
            x_permitted_cross_domain_policies: None,
            custom: vec![],
        }
    }

    /// Recommended security headers (equivalent to helmet defaults).
    pub fn recommended() -> Self {
        Self {
            hsts: Some(HstsConfig::default()),
            x_frame_options: Some(XFrameOptions::Deny),
            x_content_type_options: true,
            x_xss_protection: Some("0".to_string()),
            referrer_policy: Some(ReferrerPolicy::StrictOriginWhenCrossOrigin),
            permissions_policy: Some(PermissionsPolicy::strict()),
            cross_origin_opener_policy: Some("same-origin".to_string()),
            cross_origin_embedder_policy: None,
            cross_origin_resource_policy: Some("same-origin".to_string()),
            x_dns_prefetch_control: Some(false),
            x_download_options: true,
            x_permitted_cross_domain_policies: Some("none".to_string()),
            custom: vec![],
        }
    }

    // Builder methods

    pub fn hsts(mut self, config: HstsConfig) -> Self {
        self.hsts = Some(config);
        self
    }

    pub fn no_hsts(mut self) -> Self {
        self.hsts = None;
        self
    }

    pub fn frame_options(mut self, opts: XFrameOptions) -> Self {
        self.x_frame_options = Some(opts);
        self
    }

    pub fn no_frame_options(mut self) -> Self {
        self.x_frame_options = None;
        self
    }

    pub fn content_type_nosniff(mut self, enabled: bool) -> Self {
        self.x_content_type_options = enabled;
        self
    }

    pub fn referrer_policy(mut self, policy: ReferrerPolicy) -> Self {
        self.referrer_policy = Some(policy);
        self
    }

    pub fn permissions_policy(mut self, policy: PermissionsPolicy) -> Self {
        self.permissions_policy = Some(policy);
        self
    }

    pub fn cross_origin_opener(mut self, value: &str) -> Self {
        self.cross_origin_opener_policy = Some(value.to_string());
        self
    }

    pub fn cross_origin_embedder(mut self, value: &str) -> Self {
        self.cross_origin_embedder_policy = Some(value.to_string());
        self
    }

    pub fn cross_origin_resource(mut self, value: &str) -> Self {
        self.cross_origin_resource_policy = Some(value.to_string());
        self
    }

    pub fn dns_prefetch_control(mut self, allow: bool) -> Self {
        self.x_dns_prefetch_control = Some(allow);
        self
    }

    pub fn download_options(mut self, noopen: bool) -> Self {
        self.x_download_options = noopen;
        self
    }

    pub fn custom_header(mut self, name: &str, value: &str) -> Self {
        self.custom.push(SecurityHeader::new(name, value));
        self
    }

    /// Build the list of security headers to set on a response.
    pub fn build(&self) -> Vec<SecurityHeader> {
        let mut headers = Vec::new();

        if let Some(hsts) = &self.hsts {
            headers.push(SecurityHeader::new(
                "Strict-Transport-Security",
                &hsts.to_header_value(),
            ));
        }

        if let Some(xfo) = &self.x_frame_options {
            headers.push(SecurityHeader::new(
                "X-Frame-Options",
                &xfo.to_header_value(),
            ));
        }

        if self.x_content_type_options {
            headers.push(SecurityHeader::new(
                "X-Content-Type-Options",
                "nosniff",
            ));
        }

        if let Some(xxss) = &self.x_xss_protection {
            headers.push(SecurityHeader::new("X-XSS-Protection", xxss));
        }

        if let Some(rp) = &self.referrer_policy {
            headers.push(SecurityHeader::new("Referrer-Policy", rp.as_str()));
        }

        if let Some(pp) = &self.permissions_policy {
            headers.push(SecurityHeader::new(
                "Permissions-Policy",
                &pp.to_header_value(),
            ));
        }

        if let Some(coop) = &self.cross_origin_opener_policy {
            headers.push(SecurityHeader::new(
                "Cross-Origin-Opener-Policy",
                coop,
            ));
        }

        if let Some(coep) = &self.cross_origin_embedder_policy {
            headers.push(SecurityHeader::new(
                "Cross-Origin-Embedder-Policy",
                coep,
            ));
        }

        if let Some(corp) = &self.cross_origin_resource_policy {
            headers.push(SecurityHeader::new(
                "Cross-Origin-Resource-Policy",
                corp,
            ));
        }

        if let Some(dns) = self.x_dns_prefetch_control {
            headers.push(SecurityHeader::new(
                "X-DNS-Prefetch-Control",
                if dns { "on" } else { "off" },
            ));
        }

        if self.x_download_options {
            headers.push(SecurityHeader::new("X-Download-Options", "noopen"));
        }

        if let Some(xpcdp) = &self.x_permitted_cross_domain_policies {
            headers.push(SecurityHeader::new(
                "X-Permitted-Cross-Domain-Policies",
                xpcdp,
            ));
        }

        for custom in &self.custom {
            headers.push(custom.clone());
        }

        headers
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), SecurityHeaderError> {
        if let Some(hsts) = &self.hsts {
            hsts.validate()?;
        }
        Ok(())
    }
}

// ── Per-Route Overrides ────────────────────────────────────────

/// Security headers middleware with per-route overrides.
pub struct SecurityHeadersMiddleware {
    default_headers: SecurityHeaders,
    route_overrides: HashMap<String, SecurityHeaders>,
}

impl SecurityHeadersMiddleware {
    pub fn new(default_headers: SecurityHeaders) -> Self {
        Self {
            default_headers,
            route_overrides: HashMap::new(),
        }
    }

    /// Add a route-specific override.
    pub fn add_override(&mut self, path_prefix: &str, headers: SecurityHeaders) {
        self.route_overrides
            .insert(path_prefix.to_string(), headers);
    }

    /// Get the effective headers for a given path.
    pub fn headers_for_path(&self, path: &str) -> Vec<SecurityHeader> {
        // Try exact match first, then longest prefix match
        if let Some(headers) = self.route_overrides.get(path) {
            return headers.build();
        }

        let mut best: Option<(&str, &SecurityHeaders)> = None;
        for (prefix, headers) in &self.route_overrides {
            if path.starts_with(prefix.as_str()) {
                match best {
                    Some((prev, _)) if prefix.len() > prev.len() => {
                        best = Some((prefix, headers));
                    }
                    None => {
                        best = Some((prefix, headers));
                    }
                    _ => {}
                }
            }
        }

        match best {
            Some((_, headers)) => headers.build(),
            None => self.default_headers.build(),
        }
    }

    /// Get the count of route overrides.
    pub fn override_count(&self) -> usize {
        self.route_overrides.len()
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hsts_default() {
        let hsts = HstsConfig::default();
        let val = hsts.to_header_value();
        assert!(val.contains("max-age=31536000"));
        assert!(val.contains("includeSubDomains"));
        assert!(!val.contains("preload"));
    }

    #[test]
    fn test_hsts_with_preload() {
        let hsts = HstsConfig {
            max_age_seconds: 31_536_000,
            include_subdomains: true,
            preload: true,
        };
        let val = hsts.to_header_value();
        assert!(val.contains("preload"));
        assert!(hsts.validate().is_ok());
    }

    #[test]
    fn test_hsts_preload_requires_subdomains() {
        let hsts = HstsConfig {
            max_age_seconds: 31_536_000,
            include_subdomains: false,
            preload: true,
        };
        assert!(matches!(
            hsts.validate(),
            Err(SecurityHeaderError::Conflict(_))
        ));
    }

    #[test]
    fn test_hsts_preload_requires_min_max_age() {
        let hsts = HstsConfig {
            max_age_seconds: 3600,
            include_subdomains: true,
            preload: true,
        };
        assert!(matches!(
            hsts.validate(),
            Err(SecurityHeaderError::InvalidValue { .. })
        ));
    }

    #[test]
    fn test_x_frame_options() {
        assert_eq!(XFrameOptions::Deny.to_header_value(), "DENY");
        assert_eq!(XFrameOptions::SameOrigin.to_header_value(), "SAMEORIGIN");
        assert_eq!(
            XFrameOptions::AllowFrom("https://example.com".into()).to_header_value(),
            "ALLOW-FROM https://example.com"
        );
    }

    #[test]
    fn test_referrer_policy_values() {
        assert_eq!(ReferrerPolicy::NoReferrer.as_str(), "no-referrer");
        assert_eq!(
            ReferrerPolicy::StrictOriginWhenCrossOrigin.as_str(),
            "strict-origin-when-cross-origin"
        );
        assert_eq!(format!("{}", ReferrerPolicy::Origin), "origin");
    }

    #[test]
    fn test_referrer_policy_all_values() {
        let policies = [
            (ReferrerPolicy::NoReferrer, "no-referrer"),
            (ReferrerPolicy::NoReferrerWhenDowngrade, "no-referrer-when-downgrade"),
            (ReferrerPolicy::Origin, "origin"),
            (ReferrerPolicy::OriginWhenCrossOrigin, "origin-when-cross-origin"),
            (ReferrerPolicy::SameOrigin, "same-origin"),
            (ReferrerPolicy::StrictOrigin, "strict-origin"),
            (ReferrerPolicy::StrictOriginWhenCrossOrigin, "strict-origin-when-cross-origin"),
            (ReferrerPolicy::UnsafeUrl, "unsafe-url"),
        ];
        for (policy, expected) in &policies {
            assert_eq!(policy.as_str(), *expected);
        }
    }

    #[test]
    fn test_permission_allowlist_none() {
        assert_eq!(PermissionAllowlist::None.to_policy_value(), "()");
    }

    #[test]
    fn test_permission_allowlist_self() {
        assert_eq!(PermissionAllowlist::CSelf.to_policy_value(), "(self)");
    }

    #[test]
    fn test_permission_allowlist_all() {
        assert_eq!(PermissionAllowlist::All.to_policy_value(), "*");
    }

    #[test]
    fn test_permission_allowlist_origins() {
        let al = PermissionAllowlist::Origins(vec![
            "https://example.com".into(),
            "https://cdn.example.com".into(),
        ]);
        let val = al.to_policy_value();
        assert!(val.contains("\"https://example.com\""));
        assert!(val.contains("\"https://cdn.example.com\""));
    }

    #[test]
    fn test_permissions_policy_builder() {
        let mut pp = PermissionsPolicy::new();
        pp.deny("camera");
        pp.allow_self("microphone");
        pp.add("geolocation", PermissionAllowlist::Origins(vec!["https://maps.example.com".into()]));
        let val = pp.to_header_value();
        assert!(val.contains("camera=()"));
        assert!(val.contains("microphone=(self)"));
        assert!(val.contains("geolocation=(\"https://maps.example.com\")"));
    }

    #[test]
    fn test_permissions_policy_strict() {
        let pp = PermissionsPolicy::strict();
        let val = pp.to_header_value();
        assert!(val.contains("camera=()"));
        assert!(val.contains("microphone=()"));
        assert!(val.contains("geolocation=()"));
        assert!(val.contains("payment=()"));
    }

    #[test]
    fn test_recommended_headers() {
        let headers = SecurityHeaders::recommended();
        let built = headers.build();
        let names: Vec<&str> = built.iter().map(|h| h.name.as_str()).collect();
        assert!(names.contains(&"Strict-Transport-Security"));
        assert!(names.contains(&"X-Frame-Options"));
        assert!(names.contains(&"X-Content-Type-Options"));
        assert!(names.contains(&"Referrer-Policy"));
        assert!(names.contains(&"Permissions-Policy"));
        assert!(names.contains(&"Cross-Origin-Opener-Policy"));
    }

    #[test]
    fn test_recommended_header_values() {
        let headers = SecurityHeaders::recommended();
        let built = headers.build();
        let find = |name: &str| built.iter().find(|h| h.name == name).map(|h| h.value.as_str());
        assert_eq!(find("X-Content-Type-Options"), Some("nosniff"));
        assert_eq!(find("X-Frame-Options"), Some("DENY"));
        assert_eq!(find("X-XSS-Protection"), Some("0"));
        assert_eq!(find("Cross-Origin-Opener-Policy"), Some("same-origin"));
        assert_eq!(find("X-DNS-Prefetch-Control"), Some("off"));
        assert_eq!(find("X-Download-Options"), Some("noopen"));
    }

    #[test]
    fn test_none_headers() {
        let headers = SecurityHeaders::none();
        let built = headers.build();
        assert!(built.is_empty());
    }

    #[test]
    fn test_builder_chain() {
        let headers = SecurityHeaders::none()
            .hsts(HstsConfig {
                max_age_seconds: 3600,
                include_subdomains: false,
                preload: false,
            })
            .frame_options(XFrameOptions::SameOrigin)
            .content_type_nosniff(true)
            .referrer_policy(ReferrerPolicy::NoReferrer);

        let built = headers.build();
        assert_eq!(built.len(), 4);
    }

    #[test]
    fn test_builder_remove() {
        let headers = SecurityHeaders::recommended().no_hsts().no_frame_options();
        let built = headers.build();
        let names: Vec<&str> = built.iter().map(|h| h.name.as_str()).collect();
        assert!(!names.contains(&"Strict-Transport-Security"));
        assert!(!names.contains(&"X-Frame-Options"));
    }

    #[test]
    fn test_custom_headers() {
        let headers = SecurityHeaders::none()
            .custom_header("X-Custom", "value1")
            .custom_header("X-Another", "value2");
        let built = headers.build();
        assert_eq!(built.len(), 2);
        assert_eq!(built[0].name, "X-Custom");
        assert_eq!(built[0].value, "value1");
    }

    #[test]
    fn test_cross_origin_policies() {
        let headers = SecurityHeaders::none()
            .cross_origin_opener("same-origin-allow-popups")
            .cross_origin_embedder("require-corp")
            .cross_origin_resource("cross-origin");
        let built = headers.build();
        assert_eq!(built.len(), 3);
    }

    #[test]
    fn test_validate_recommended() {
        let headers = SecurityHeaders::recommended();
        assert!(headers.validate().is_ok());
    }

    #[test]
    fn test_validate_invalid_hsts() {
        let headers = SecurityHeaders::none().hsts(HstsConfig {
            max_age_seconds: 100,
            include_subdomains: false,
            preload: true,
        });
        assert!(headers.validate().is_err());
    }

    #[test]
    fn test_middleware_default() {
        let mw = SecurityHeadersMiddleware::new(SecurityHeaders::recommended());
        let headers = mw.headers_for_path("/api/data");
        let names: Vec<&str> = headers.iter().map(|h| h.name.as_str()).collect();
        assert!(names.contains(&"X-Content-Type-Options"));
    }

    #[test]
    fn test_middleware_route_override() {
        let mut mw = SecurityHeadersMiddleware::new(SecurityHeaders::recommended());
        // For /api routes, relax frame options
        mw.add_override(
            "/api/",
            SecurityHeaders::none().frame_options(XFrameOptions::SameOrigin),
        );

        let api_headers = mw.headers_for_path("/api/resource");
        let find_api = |name: &str| {
            api_headers
                .iter()
                .find(|h| h.name == name)
                .map(|h| h.value.clone())
        };
        assert_eq!(find_api("X-Frame-Options"), Some("SAMEORIGIN".into()));
        // Should NOT have HSTS (from override, not default)
        assert!(find_api("Strict-Transport-Security").is_none());

        let default_headers = mw.headers_for_path("/web/page");
        let find_default = |name: &str| {
            default_headers
                .iter()
                .find(|h| h.name == name)
                .map(|h| h.value.clone())
        };
        assert_eq!(find_default("X-Frame-Options"), Some("DENY".into()));
    }

    #[test]
    fn test_middleware_exact_match() {
        let mut mw = SecurityHeadersMiddleware::new(SecurityHeaders::recommended());
        mw.add_override(
            "/health",
            SecurityHeaders::none().content_type_nosniff(true),
        );
        let headers = mw.headers_for_path("/health");
        assert_eq!(headers.len(), 1);
        assert_eq!(headers[0].name, "X-Content-Type-Options");
    }

    #[test]
    fn test_middleware_longest_prefix() {
        let mut mw = SecurityHeadersMiddleware::new(SecurityHeaders::none());
        mw.add_override(
            "/api/",
            SecurityHeaders::none().content_type_nosniff(true),
        );
        mw.add_override(
            "/api/admin/",
            SecurityHeaders::none()
                .content_type_nosniff(true)
                .frame_options(XFrameOptions::Deny),
        );

        let admin_headers = mw.headers_for_path("/api/admin/users");
        assert_eq!(admin_headers.len(), 2); // nosniff + x-frame-options

        let api_headers = mw.headers_for_path("/api/data");
        assert_eq!(api_headers.len(), 1); // just nosniff
    }

    #[test]
    fn test_middleware_override_count() {
        let mut mw = SecurityHeadersMiddleware::new(SecurityHeaders::none());
        assert_eq!(mw.override_count(), 0);
        mw.add_override("/a", SecurityHeaders::none());
        mw.add_override("/b", SecurityHeaders::none());
        assert_eq!(mw.override_count(), 2);
    }

    #[test]
    fn test_dns_prefetch_control() {
        let headers = SecurityHeaders::none().dns_prefetch_control(true);
        let built = headers.build();
        assert_eq!(built.len(), 1);
        assert_eq!(built[0].value, "on");

        let headers2 = SecurityHeaders::none().dns_prefetch_control(false);
        let built2 = headers2.build();
        assert_eq!(built2[0].value, "off");
    }

    #[test]
    fn test_download_options() {
        let headers = SecurityHeaders::none().download_options(true);
        let built = headers.build();
        assert_eq!(built.len(), 1);
        assert_eq!(built[0].name, "X-Download-Options");
        assert_eq!(built[0].value, "noopen");
    }

    #[test]
    fn test_error_display() {
        let e = SecurityHeaderError::InvalidValue {
            header: "HSTS".into(),
            value: "bad".into(),
        };
        assert!(e.to_string().contains("HSTS"));
        let e = SecurityHeaderError::Conflict("test".into());
        assert!(e.to_string().contains("test"));
    }

    #[test]
    fn test_security_header_new() {
        let h = SecurityHeader::new("X-Test", "value");
        assert_eq!(h.name, "X-Test");
        assert_eq!(h.value, "value");
    }
}
