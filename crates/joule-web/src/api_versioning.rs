//! API versioning — URL path, header, query param strategies, version negotiation,
//! deprecation warnings, sunset headers, and migration paths.
//!
//! Replaces Express versioning middleware, API gateway version routing, and custom
//! version negotiation with a pure-Rust API versioning framework.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── Errors ─────────────────────────────────────────────────────

/// API versioning errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VersionError {
    /// Version not found or unsupported.
    UnsupportedVersion(String),
    /// Version has been sunset (removed).
    VersionSunset { version: String, sunset_date: String },
    /// No version specified in request.
    NoVersionSpecified,
    /// Invalid version format.
    InvalidFormat(String),
    /// Version negotiation failed.
    NegotiationFailed(String),
    /// Duplicate version registration.
    DuplicateVersion(String),
    /// Route not found for version.
    RouteNotFound { version: String, path: String },
}

impl fmt::Display for VersionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedVersion(v) => write!(f, "unsupported API version: {v}"),
            Self::VersionSunset { version, sunset_date } => {
                write!(f, "API version {version} was sunset on {sunset_date}")
            }
            Self::NoVersionSpecified => write!(f, "no API version specified in request"),
            Self::InvalidFormat(v) => write!(f, "invalid version format: {v}"),
            Self::NegotiationFailed(msg) => write!(f, "version negotiation failed: {msg}"),
            Self::DuplicateVersion(v) => write!(f, "duplicate version: {v}"),
            Self::RouteNotFound { version, path } => {
                write!(f, "route not found: {path} (version {version})")
            }
        }
    }
}

impl std::error::Error for VersionError {}

// ── Version Extraction Strategy ────────────────────────────────

/// Strategy for extracting version from a request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VersionStrategy {
    /// Extract from URL path (e.g., /v1/resource).
    UrlPath { prefix: String },
    /// Extract from header (e.g., Accept-Version: 1).
    Header { name: String },
    /// Extract from query parameter (e.g., ?version=1).
    QueryParam { param: String },
    /// Extract from Accept media type (e.g., application/vnd.api.v1+json).
    MediaType { vendor: String },
}

impl VersionStrategy {
    /// URL path strategy with default "/v" prefix.
    pub fn url_path() -> Self {
        Self::UrlPath { prefix: "/v".to_string() }
    }

    /// Header strategy with default "Accept-Version" header.
    pub fn header() -> Self {
        Self::Header { name: "Accept-Version".to_string() }
    }

    /// Query param strategy with default "version" param.
    pub fn query_param() -> Self {
        Self::QueryParam { param: "version".to_string() }
    }

    /// Media type strategy.
    pub fn media_type(vendor: impl Into<String>) -> Self {
        Self::MediaType { vendor: vendor.into() }
    }

    /// Extract version string from request components.
    pub fn extract(
        &self,
        path: &str,
        headers: &HashMap<String, String>,
        query_params: &HashMap<String, String>,
    ) -> Option<String> {
        match self {
            Self::UrlPath { prefix } => {
                // Look for /v1/, /v2/, etc.
                if let Some(rest) = path.strip_prefix(prefix) {
                    // Take digits until next slash or end
                    let version: String = rest.chars().take_while(|c| c.is_ascii_digit() || *c == '.').collect();
                    if !version.is_empty() {
                        return Some(version);
                    }
                }
                None
            }
            Self::Header { name } => {
                // Case-insensitive header lookup
                let lower = name.to_lowercase();
                headers.iter()
                    .find(|(k, _)| k.to_lowercase() == lower)
                    .map(|(_, v)| v.clone())
            }
            Self::QueryParam { param } => query_params.get(param).cloned(),
            Self::MediaType { vendor } => {
                // Look in Accept header for vendor media type
                let accept = headers.get("Accept").or_else(|| headers.get("accept"))?;
                let pattern = format!("vnd.{vendor}.v");
                if let Some(pos) = accept.find(&pattern) {
                    let after = &accept[pos + pattern.len()..];
                    let version: String = after.chars().take_while(|c| c.is_ascii_digit() || *c == '.').collect();
                    if !version.is_empty() {
                        return Some(version);
                    }
                }
                None
            }
        }
    }
}

// ── API Version ────────────────────────────────────────────────

/// Status of an API version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VersionStatus {
    /// Currently active and supported.
    Active,
    /// Deprecated — still works but will be removed.
    Deprecated,
    /// Sunset — no longer available.
    Sunset,
    /// Beta/preview — not yet stable.
    Beta,
}

impl fmt::Display for VersionStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Active => write!(f, "active"),
            Self::Deprecated => write!(f, "deprecated"),
            Self::Sunset => write!(f, "sunset"),
            Self::Beta => write!(f, "beta"),
        }
    }
}

/// An API version definition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApiVersion {
    /// Version identifier (e.g., "1", "2", "2.1").
    pub version: String,
    /// Status of this version.
    pub status: VersionStatus,
    /// Deprecation date (if deprecated).
    pub deprecation_date: Option<String>,
    /// Sunset date (if sunset).
    pub sunset_date: Option<String>,
    /// Version this one supersedes.
    pub superseded_by: Option<String>,
    /// Release notes or description.
    pub description: Option<String>,
}

impl ApiVersion {
    /// Create a new active version.
    pub fn new(version: impl Into<String>) -> Self {
        Self {
            version: version.into(),
            status: VersionStatus::Active,
            deprecation_date: None,
            sunset_date: None,
            superseded_by: None,
            description: None,
        }
    }

    /// Mark as deprecated.
    pub fn deprecated(mut self, date: impl Into<String>, superseded_by: impl Into<String>) -> Self {
        self.status = VersionStatus::Deprecated;
        self.deprecation_date = Some(date.into());
        self.superseded_by = Some(superseded_by.into());
        self
    }

    /// Mark as sunset.
    pub fn sunset(mut self, date: impl Into<String>) -> Self {
        self.status = VersionStatus::Sunset;
        self.sunset_date = Some(date.into());
        self
    }

    /// Set as beta.
    pub fn beta(mut self) -> Self {
        self.status = VersionStatus::Beta;
        self
    }

    /// Set description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Check if this version is usable (not sunset).
    pub fn is_usable(&self) -> bool {
        !matches!(self.status, VersionStatus::Sunset)
    }
}

// ── Deprecation Warning ────────────────────────────────────────

/// A deprecation warning to include in API responses.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeprecationWarning {
    /// The deprecated version.
    pub version: String,
    /// When it will be sunset.
    pub sunset_date: Option<String>,
    /// Suggested replacement version.
    pub replacement: Option<String>,
    /// Link to migration guide.
    pub migration_link: Option<String>,
}

impl DeprecationWarning {
    /// Generate HTTP headers for this warning.
    pub fn to_headers(&self) -> HashMap<String, String> {
        let mut headers = HashMap::new();
        headers.insert(
            "Deprecation".to_string(),
            format!("version=\"{}\"", self.version),
        );
        if let Some(sunset) = &self.sunset_date {
            headers.insert("Sunset".to_string(), sunset.clone());
        }
        if let Some(link) = &self.migration_link {
            headers.insert(
                "Link".to_string(),
                format!("<{link}>; rel=\"successor-version\""),
            );
        }
        headers
    }
}

// ── Migration Path ─────────────────────────────────────────────

/// A migration path between API versions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MigrationPath {
    /// Source version.
    pub from: String,
    /// Target version.
    pub to: String,
    /// Breaking changes in this migration.
    pub breaking_changes: Vec<String>,
    /// New features available.
    pub new_features: Vec<String>,
    /// Removed features.
    pub removed_features: Vec<String>,
}

// ── Version Router ─────────────────────────────────────────────

/// Result of resolving a version from a request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VersionResolution {
    /// Resolved version string.
    pub version: String,
    /// Deprecation warning (if version is deprecated).
    pub deprecation: Option<DeprecationWarning>,
    /// Additional response headers to include.
    pub headers: HashMap<String, String>,
}

/// API version router managing versions and routing.
#[derive(Debug, Clone)]
pub struct VersionRouter {
    /// Registered versions.
    versions: HashMap<String, ApiVersion>,
    /// Version extraction strategy.
    pub strategy: VersionStrategy,
    /// Default version if none specified.
    pub default_version: Option<String>,
    /// Migration paths between versions.
    migrations: Vec<MigrationPath>,
}

impl VersionRouter {
    /// Create a new version router.
    pub fn new(strategy: VersionStrategy) -> Self {
        Self {
            versions: HashMap::new(),
            strategy,
            default_version: None,
            migrations: Vec::new(),
        }
    }

    /// Set the default version.
    pub fn with_default(mut self, version: impl Into<String>) -> Self {
        self.default_version = Some(version.into());
        self
    }

    /// Register a version.
    pub fn register(&mut self, version: ApiVersion) -> Result<(), VersionError> {
        if self.versions.contains_key(&version.version) {
            return Err(VersionError::DuplicateVersion(version.version));
        }
        self.versions.insert(version.version.clone(), version);
        Ok(())
    }

    /// Get a version definition.
    pub fn get_version(&self, version: &str) -> Option<&ApiVersion> {
        self.versions.get(version)
    }

    /// List all versions (sorted).
    pub fn all_versions(&self) -> Vec<&ApiVersion> {
        let mut versions: Vec<&ApiVersion> = self.versions.values().collect();
        versions.sort_by(|a, b| a.version.cmp(&b.version));
        versions
    }

    /// List active versions.
    pub fn active_versions(&self) -> Vec<&ApiVersion> {
        let mut versions: Vec<&ApiVersion> = self.versions.values()
            .filter(|v| v.status == VersionStatus::Active)
            .collect();
        versions.sort_by(|a, b| a.version.cmp(&b.version));
        versions
    }

    /// Add a migration path.
    pub fn add_migration(&mut self, migration: MigrationPath) {
        self.migrations.push(migration);
    }

    /// Get migration path between two versions.
    pub fn migration_path(&self, from: &str, to: &str) -> Option<&MigrationPath> {
        self.migrations.iter().find(|m| m.from == from && m.to == to)
    }

    /// Resolve a version from request components.
    pub fn resolve(
        &self,
        path: &str,
        headers: &HashMap<String, String>,
        query_params: &HashMap<String, String>,
    ) -> Result<VersionResolution, VersionError> {
        let version_str = self
            .strategy
            .extract(path, headers, query_params)
            .or_else(|| self.default_version.clone())
            .ok_or(VersionError::NoVersionSpecified)?;

        let version = self
            .versions
            .get(&version_str)
            .ok_or_else(|| VersionError::UnsupportedVersion(version_str.clone()))?;

        if matches!(version.status, VersionStatus::Sunset) {
            return Err(VersionError::VersionSunset {
                version: version_str,
                sunset_date: version.sunset_date.clone().unwrap_or_default(),
            });
        }

        let deprecation = if version.status == VersionStatus::Deprecated {
            Some(DeprecationWarning {
                version: version_str.clone(),
                sunset_date: version.sunset_date.clone(),
                replacement: version.superseded_by.clone(),
                migration_link: None,
            })
        } else {
            None
        };

        let mut response_headers = HashMap::new();
        if let Some(ref dep) = deprecation {
            response_headers.extend(dep.to_headers());
        }
        response_headers.insert("API-Version".to_string(), version_str.clone());

        Ok(VersionResolution {
            version: version_str,
            deprecation,
            headers: response_headers,
        })
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_router() -> VersionRouter {
        let mut router = VersionRouter::new(VersionStrategy::url_path())
            .with_default("1");
        router.register(ApiVersion::new("1").with_description("Initial release")).unwrap();
        router.register(ApiVersion::new("2").with_description("Major update")).unwrap();
        router.register(
            ApiVersion::new("0")
                .deprecated("2025-01-01", "1"),
        ).unwrap();
        router
    }

    #[test]
    fn test_url_path_extraction() {
        let strategy = VersionStrategy::url_path();
        let headers = HashMap::new();
        let query = HashMap::new();
        assert_eq!(strategy.extract("/v1/users", &headers, &query), Some("1".to_string()));
        assert_eq!(strategy.extract("/v2/posts", &headers, &query), Some("2".to_string()));
        assert_eq!(strategy.extract("/v2.1/items", &headers, &query), Some("2.1".to_string()));
        assert_eq!(strategy.extract("/users", &headers, &query), None);
    }

    #[test]
    fn test_header_extraction() {
        let strategy = VersionStrategy::header();
        let mut headers = HashMap::new();
        headers.insert("Accept-Version".to_string(), "2".to_string());
        assert_eq!(strategy.extract("/users", &headers, &HashMap::new()), Some("2".to_string()));
    }

    #[test]
    fn test_header_extraction_case_insensitive() {
        let strategy = VersionStrategy::header();
        let mut headers = HashMap::new();
        headers.insert("accept-version".to_string(), "3".to_string());
        assert_eq!(strategy.extract("/users", &headers, &HashMap::new()), Some("3".to_string()));
    }

    #[test]
    fn test_query_param_extraction() {
        let strategy = VersionStrategy::query_param();
        let mut query = HashMap::new();
        query.insert("version".to_string(), "1".to_string());
        assert_eq!(strategy.extract("/users", &HashMap::new(), &query), Some("1".to_string()));
    }

    #[test]
    fn test_media_type_extraction() {
        let strategy = VersionStrategy::media_type("myapi");
        let mut headers = HashMap::new();
        headers.insert("Accept".to_string(), "application/vnd.myapi.v2+json".to_string());
        assert_eq!(strategy.extract("/users", &headers, &HashMap::new()), Some("2".to_string()));
    }

    #[test]
    fn test_resolve_active_version() {
        let router = make_router();
        let result = router.resolve("/v1/users", &HashMap::new(), &HashMap::new()).unwrap();
        assert_eq!(result.version, "1");
        assert!(result.deprecation.is_none());
        assert_eq!(result.headers.get("API-Version").unwrap(), "1");
    }

    #[test]
    fn test_resolve_deprecated_version() {
        let router = make_router();
        let result = router.resolve("/v0/users", &HashMap::new(), &HashMap::new()).unwrap();
        assert_eq!(result.version, "0");
        assert!(result.deprecation.is_some());
        let dep = result.deprecation.unwrap();
        assert_eq!(dep.replacement.as_deref(), Some("1"));
    }

    #[test]
    fn test_resolve_sunset_version() {
        let mut router = VersionRouter::new(VersionStrategy::url_path());
        router.register(ApiVersion::new("9").sunset("2024-01-01")).unwrap();
        let result = router.resolve("/v9/users", &HashMap::new(), &HashMap::new());
        assert!(matches!(result, Err(VersionError::VersionSunset { .. })));
    }

    #[test]
    fn test_resolve_default_version() {
        let router = make_router();
        // No version in path, should use default
        let result = router.resolve("/users", &HashMap::new(), &HashMap::new()).unwrap();
        assert_eq!(result.version, "1");
    }

    #[test]
    fn test_resolve_no_version_no_default() {
        let router = VersionRouter::new(VersionStrategy::url_path());
        let result = router.resolve("/users", &HashMap::new(), &HashMap::new());
        assert!(matches!(result, Err(VersionError::NoVersionSpecified)));
    }

    #[test]
    fn test_resolve_unsupported_version() {
        let router = make_router();
        let result = router.resolve("/v99/users", &HashMap::new(), &HashMap::new());
        assert!(matches!(result, Err(VersionError::UnsupportedVersion(_))));
    }

    #[test]
    fn test_duplicate_version() {
        let mut router = VersionRouter::new(VersionStrategy::url_path());
        router.register(ApiVersion::new("1")).unwrap();
        let err = router.register(ApiVersion::new("1")).unwrap_err();
        assert!(matches!(err, VersionError::DuplicateVersion(_)));
    }

    #[test]
    fn test_all_versions_sorted() {
        let router = make_router();
        let versions: Vec<String> = router.all_versions().iter().map(|v| v.version.clone()).collect();
        assert_eq!(versions, vec!["0", "1", "2"]);
    }

    #[test]
    fn test_active_versions() {
        let router = make_router();
        let active: Vec<String> = router.active_versions().iter().map(|v| v.version.clone()).collect();
        assert_eq!(active, vec!["1", "2"]);
    }

    #[test]
    fn test_version_is_usable() {
        assert!(ApiVersion::new("1").is_usable());
        assert!(ApiVersion::new("1").deprecated("2025-01-01", "2").is_usable());
        assert!(ApiVersion::new("1").beta().is_usable());
        assert!(!ApiVersion::new("1").sunset("2024-01-01").is_usable());
    }

    #[test]
    fn test_deprecation_headers() {
        let warning = DeprecationWarning {
            version: "1".to_string(),
            sunset_date: Some("2025-12-31".to_string()),
            replacement: Some("2".to_string()),
            migration_link: Some("https://docs.example.com/migrate".to_string()),
        };
        let headers = warning.to_headers();
        assert!(headers.get("Deprecation").unwrap().contains("1"));
        assert_eq!(headers.get("Sunset").unwrap(), "2025-12-31");
        assert!(headers.get("Link").unwrap().contains("successor-version"));
    }

    #[test]
    fn test_migration_path() {
        let mut router = make_router();
        router.add_migration(MigrationPath {
            from: "1".to_string(),
            to: "2".to_string(),
            breaking_changes: vec!["removed /legacy endpoint".to_string()],
            new_features: vec!["added pagination".to_string()],
            removed_features: vec!["/legacy".to_string()],
        });
        let path = router.migration_path("1", "2").unwrap();
        assert_eq!(path.breaking_changes.len(), 1);
        assert_eq!(path.new_features.len(), 1);
        assert!(router.migration_path("2", "3").is_none());
    }

    #[test]
    fn test_version_status_display() {
        assert_eq!(format!("{}", VersionStatus::Active), "active");
        assert_eq!(format!("{}", VersionStatus::Deprecated), "deprecated");
        assert_eq!(format!("{}", VersionStatus::Sunset), "sunset");
        assert_eq!(format!("{}", VersionStatus::Beta), "beta");
    }

    #[test]
    fn test_error_display() {
        let err = VersionError::UnsupportedVersion("99".to_string());
        assert!(format!("{err}").contains("99"));
        let err = VersionError::NoVersionSpecified;
        assert!(format!("{err}").contains("no API version"));
    }

    #[test]
    fn test_header_strategy_missing() {
        let strategy = VersionStrategy::header();
        assert!(strategy.extract("/users", &HashMap::new(), &HashMap::new()).is_none());
    }

    #[test]
    fn test_media_type_no_match() {
        let strategy = VersionStrategy::media_type("myapi");
        let mut headers = HashMap::new();
        headers.insert("Accept".to_string(), "application/json".to_string());
        assert!(strategy.extract("/users", &headers, &HashMap::new()).is_none());
    }

    #[test]
    fn test_get_version() {
        let router = make_router();
        let v = router.get_version("1").unwrap();
        assert_eq!(v.status, VersionStatus::Active);
        assert!(router.get_version("99").is_none());
    }

    #[test]
    fn test_beta_version() {
        let v = ApiVersion::new("3").beta();
        assert_eq!(v.status, VersionStatus::Beta);
        assert!(v.is_usable());
    }

    #[test]
    fn test_resolve_with_header_strategy() {
        let mut router = VersionRouter::new(VersionStrategy::header());
        router.register(ApiVersion::new("2")).unwrap();
        let mut headers = HashMap::new();
        headers.insert("Accept-Version".to_string(), "2".to_string());
        let result = router.resolve("/users", &headers, &HashMap::new()).unwrap();
        assert_eq!(result.version, "2");
    }
}
