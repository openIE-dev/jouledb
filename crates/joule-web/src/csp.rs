//! Content Security Policy — directive builder and header generation.
//!
//! Replaces Node.js helmet/csp middleware with a pure-Rust CSP builder
//! that constructs, validates, and evaluates Content-Security-Policy headers.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

// ── Errors ─────────────────────────────────────────────────────

/// CSP domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CspError {
    /// Unknown directive name.
    UnknownDirective(String),
    /// Invalid source value.
    InvalidSource(String),
    /// Duplicate directive.
    DuplicateDirective(String),
    /// Empty policy.
    EmptyPolicy,
    /// Invalid nonce (must be base64).
    InvalidNonce(String),
    /// Invalid hash (must be sha256/sha384/sha512-base64).
    InvalidHash(String),
}

impl fmt::Display for CspError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownDirective(d) => write!(f, "unknown directive: {d}"),
            Self::InvalidSource(s) => write!(f, "invalid source: {s}"),
            Self::DuplicateDirective(d) => write!(f, "duplicate directive: {d}"),
            Self::EmptyPolicy => write!(f, "empty policy"),
            Self::InvalidNonce(n) => write!(f, "invalid nonce: {n}"),
            Self::InvalidHash(h) => write!(f, "invalid hash: {h}"),
        }
    }
}

impl std::error::Error for CspError {}

// ── Directive names ────────────────────────────────────────────

/// Well-known CSP directive names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Directive {
    DefaultSrc,
    ScriptSrc,
    StyleSrc,
    ImgSrc,
    ConnectSrc,
    FontSrc,
    ObjectSrc,
    MediaSrc,
    FrameSrc,
    ChildSrc,
    WorkerSrc,
    ManifestSrc,
    PrefetchSrc,
    FormAction,
    FrameAncestors,
    BaseUri,
    ReportUri,
    ReportTo,
    UpgradeInsecureRequests,
    BlockAllMixedContent,
    Sandbox,
}

impl Directive {
    /// Return the CSP header string for this directive.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::DefaultSrc => "default-src",
            Self::ScriptSrc => "script-src",
            Self::StyleSrc => "style-src",
            Self::ImgSrc => "img-src",
            Self::ConnectSrc => "connect-src",
            Self::FontSrc => "font-src",
            Self::ObjectSrc => "object-src",
            Self::MediaSrc => "media-src",
            Self::FrameSrc => "frame-src",
            Self::ChildSrc => "child-src",
            Self::WorkerSrc => "worker-src",
            Self::ManifestSrc => "manifest-src",
            Self::PrefetchSrc => "prefetch-src",
            Self::FormAction => "form-action",
            Self::FrameAncestors => "frame-ancestors",
            Self::BaseUri => "base-uri",
            Self::ReportUri => "report-uri",
            Self::ReportTo => "report-to",
            Self::UpgradeInsecureRequests => "upgrade-insecure-requests",
            Self::BlockAllMixedContent => "block-all-mixed-content",
            Self::Sandbox => "sandbox",
        }
    }

    /// Parse a directive name from a string.
    pub fn from_str_name(s: &str) -> Result<Self, CspError> {
        match s {
            "default-src" => Ok(Self::DefaultSrc),
            "script-src" => Ok(Self::ScriptSrc),
            "style-src" => Ok(Self::StyleSrc),
            "img-src" => Ok(Self::ImgSrc),
            "connect-src" => Ok(Self::ConnectSrc),
            "font-src" => Ok(Self::FontSrc),
            "object-src" => Ok(Self::ObjectSrc),
            "media-src" => Ok(Self::MediaSrc),
            "frame-src" => Ok(Self::FrameSrc),
            "child-src" => Ok(Self::ChildSrc),
            "worker-src" => Ok(Self::WorkerSrc),
            "manifest-src" => Ok(Self::ManifestSrc),
            "prefetch-src" => Ok(Self::PrefetchSrc),
            "form-action" => Ok(Self::FormAction),
            "frame-ancestors" => Ok(Self::FrameAncestors),
            "base-uri" => Ok(Self::BaseUri),
            "report-uri" => Ok(Self::ReportUri),
            "report-to" => Ok(Self::ReportTo),
            "upgrade-insecure-requests" => Ok(Self::UpgradeInsecureRequests),
            "block-all-mixed-content" => Ok(Self::BlockAllMixedContent),
            "sandbox" => Ok(Self::Sandbox),
            _ => Err(CspError::UnknownDirective(s.to_string())),
        }
    }
}

impl fmt::Display for Directive {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ── Source values ──────────────────────────────────────────────

/// CSP source values.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Source {
    /// 'self'
    CSelf,
    /// 'none'
    None,
    /// 'unsafe-inline'
    UnsafeInline,
    /// 'unsafe-eval'
    UnsafeEval,
    /// 'strict-dynamic'
    StrictDynamic,
    /// 'unsafe-hashes'
    UnsafeHashes,
    /// A nonce: 'nonce-<base64>'
    Nonce(String),
    /// A hash: 'sha256-<base64>', 'sha384-<base64>', 'sha512-<base64>'
    Hash(String),
    /// A host source: *.example.com, https://cdn.example.com
    Host(String),
    /// A scheme source: https:, data:, blob:
    Scheme(String),
    /// Wildcard: *
    Wildcard,
}

impl Source {
    /// Render this source as a CSP header value.
    pub fn to_csp_string(&self) -> String {
        match self {
            Self::CSelf => "'self'".to_string(),
            Self::None => "'none'".to_string(),
            Self::UnsafeInline => "'unsafe-inline'".to_string(),
            Self::UnsafeEval => "'unsafe-eval'".to_string(),
            Self::StrictDynamic => "'strict-dynamic'".to_string(),
            Self::UnsafeHashes => "'unsafe-hashes'".to_string(),
            Self::Nonce(n) => format!("'nonce-{n}'"),
            Self::Hash(h) => format!("'{h}'"),
            Self::Host(h) => h.clone(),
            Self::Scheme(s) => format!("{s}:"),
            Self::Wildcard => "*".to_string(),
        }
    }
}

impl fmt::Display for Source {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_csp_string())
    }
}

// ── CSP Policy Builder ────────────────────────────────────────

/// Content Security Policy builder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CspPolicy {
    directives: BTreeMap<String, Vec<Source>>,
    report_only: bool,
}

impl Default for CspPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl CspPolicy {
    /// Create a new empty CSP policy.
    pub fn new() -> Self {
        Self {
            directives: BTreeMap::new(),
            report_only: false,
        }
    }

    /// Create a strict default policy.
    pub fn strict() -> Self {
        let mut p = Self::new();
        p.add(Directive::DefaultSrc, Source::None);
        p.add(Directive::ScriptSrc, Source::CSelf);
        p.add(Directive::StyleSrc, Source::CSelf);
        p.add(Directive::ImgSrc, Source::CSelf);
        p.add(Directive::ConnectSrc, Source::CSelf);
        p.add(Directive::FontSrc, Source::CSelf);
        p.add(Directive::ObjectSrc, Source::None);
        p.add(Directive::FrameAncestors, Source::None);
        p.add(Directive::BaseUri, Source::CSelf);
        p.add(Directive::FormAction, Source::CSelf);
        p
    }

    /// Set this policy as report-only.
    pub fn report_only(mut self) -> Self {
        self.report_only = true;
        self
    }

    /// Add a source to a directive.
    pub fn add(&mut self, directive: Directive, source: Source) -> &mut Self {
        self.directives
            .entry(directive.as_str().to_string())
            .or_default()
            .push(source);
        self
    }

    /// Set a directive with multiple sources at once.
    pub fn set(&mut self, directive: Directive, sources: Vec<Source>) -> &mut Self {
        self.directives.insert(directive.as_str().to_string(), sources);
        self
    }

    /// Add a report-uri directive.
    pub fn report_uri(&mut self, uri: &str) -> &mut Self {
        self.directives.insert(
            Directive::ReportUri.as_str().to_string(),
            vec![Source::Host(uri.to_string())],
        );
        self
    }

    /// Add a report-to directive.
    pub fn report_to_group(&mut self, group: &str) -> &mut Self {
        self.directives.insert(
            Directive::ReportTo.as_str().to_string(),
            vec![Source::Host(group.to_string())],
        );
        self
    }

    /// Add a nonce to script-src.
    pub fn add_script_nonce(&mut self, nonce: &str) -> &mut Self {
        self.add(Directive::ScriptSrc, Source::Nonce(nonce.to_string()))
    }

    /// Add a nonce to style-src.
    pub fn add_style_nonce(&mut self, nonce: &str) -> &mut Self {
        self.add(Directive::StyleSrc, Source::Nonce(nonce.to_string()))
    }

    /// Enable upgrade-insecure-requests.
    pub fn upgrade_insecure(&mut self) -> &mut Self {
        self.directives.insert(
            Directive::UpgradeInsecureRequests.as_str().to_string(),
            vec![],
        );
        self
    }

    /// Enable block-all-mixed-content.
    pub fn block_mixed_content(&mut self) -> &mut Self {
        self.directives.insert(
            Directive::BlockAllMixedContent.as_str().to_string(),
            vec![],
        );
        self
    }

    /// Get the CSP header name.
    pub fn header_name(&self) -> &str {
        if self.report_only {
            "Content-Security-Policy-Report-Only"
        } else {
            "Content-Security-Policy"
        }
    }

    /// Generate the CSP header value.
    pub fn to_header_value(&self) -> String {
        let mut parts = Vec::new();
        for (directive, sources) in &self.directives {
            if sources.is_empty() {
                parts.push(directive.clone());
            } else {
                let values: Vec<String> = sources.iter().map(|s| s.to_csp_string()).collect();
                parts.push(format!("{} {}", directive, values.join(" ")));
            }
        }
        parts.join("; ")
    }

    /// Parse a CSP header value into a policy.
    pub fn parse(header: &str) -> Result<Self, CspError> {
        let mut policy = Self::new();
        for part in header.split(';') {
            let trimmed = part.trim();
            if trimmed.is_empty() {
                continue;
            }
            let tokens: Vec<&str> = trimmed.split_whitespace().collect();
            if tokens.is_empty() {
                continue;
            }
            let directive_name = tokens[0];
            // Validate directive name.
            let _ = Directive::from_str_name(directive_name)?;
            let sources: Vec<Source> = tokens[1..].iter().map(|t| parse_source(t)).collect();
            policy.directives.insert(directive_name.to_string(), sources);
        }
        Ok(policy)
    }

    /// Check if a source is allowed by a specific directive.
    pub fn allows(&self, directive: Directive, source: &str) -> bool {
        // Check specific directive first, then fall back to default-src.
        let sources = self
            .directives
            .get(directive.as_str())
            .or_else(|| self.directives.get("default-src"));

        match sources {
            None => true, // No policy = allowed.
            Some(srcs) => {
                for s in srcs {
                    match s {
                        Source::Wildcard => return true,
                        Source::None => return false,
                        Source::CSelf => {
                            // In a real impl, compare against page origin.
                            if source == "self" || source == "'self'" {
                                return true;
                            }
                        }
                        Source::Host(h) => {
                            if h.starts_with("*.") {
                                let domain = &h[2..];
                                if source.ends_with(domain) {
                                    return true;
                                }
                            } else if source == h.as_str() {
                                return true;
                            }
                        }
                        Source::Scheme(scheme) => {
                            if source.starts_with(&format!("{scheme}:")) {
                                return true;
                            }
                        }
                        _ => {}
                    }
                }
                false
            }
        }
    }

    /// Get all directives.
    pub fn directives(&self) -> &BTreeMap<String, Vec<Source>> {
        &self.directives
    }

    /// Check if a directive is present.
    pub fn has_directive(&self, directive: Directive) -> bool {
        self.directives.contains_key(directive.as_str())
    }
}

impl fmt::Display for CspPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_header_value())
    }
}

fn parse_source(s: &str) -> Source {
    match s {
        "'self'" => Source::CSelf,
        "'none'" => Source::None,
        "'unsafe-inline'" => Source::UnsafeInline,
        "'unsafe-eval'" => Source::UnsafeEval,
        "'strict-dynamic'" => Source::StrictDynamic,
        "'unsafe-hashes'" => Source::UnsafeHashes,
        "*" => Source::Wildcard,
        _ if s.starts_with("'nonce-") && s.ends_with('\'') => {
            Source::Nonce(s[7..s.len() - 1].to_string())
        }
        _ if (s.starts_with("'sha256-") || s.starts_with("'sha384-") || s.starts_with("'sha512-"))
            && s.ends_with('\'') =>
        {
            Source::Hash(s[1..s.len() - 1].to_string())
        }
        _ if s.ends_with(':') && !s.contains('/') => {
            Source::Scheme(s[..s.len() - 1].to_string())
        }
        _ => Source::Host(s.to_string()),
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strict_policy() {
        let p = CspPolicy::strict();
        let header = p.to_header_value();
        assert!(header.contains("default-src 'none'"));
        assert!(header.contains("script-src 'self'"));
        assert!(header.contains("object-src 'none'"));
    }

    #[test]
    fn test_header_name() {
        let p = CspPolicy::new();
        assert_eq!(p.header_name(), "Content-Security-Policy");
        let p2 = CspPolicy::new().report_only();
        assert_eq!(p2.header_name(), "Content-Security-Policy-Report-Only");
    }

    #[test]
    fn test_add_sources() {
        let mut p = CspPolicy::new();
        p.add(Directive::ScriptSrc, Source::CSelf);
        p.add(Directive::ScriptSrc, Source::Host("https://cdn.example.com".to_string()));
        let header = p.to_header_value();
        assert!(header.contains("script-src 'self' https://cdn.example.com"));
    }

    #[test]
    fn test_nonce() {
        let mut p = CspPolicy::new();
        p.add_script_nonce("abc123");
        let header = p.to_header_value();
        assert!(header.contains("'nonce-abc123'"));
    }

    #[test]
    fn test_parse_simple() {
        let header = "default-src 'self'; script-src 'self' 'unsafe-inline'; img-src *";
        let p = CspPolicy::parse(header).unwrap();
        assert!(p.has_directive(Directive::DefaultSrc));
        assert!(p.has_directive(Directive::ScriptSrc));
        assert!(p.has_directive(Directive::ImgSrc));
    }

    #[test]
    fn test_parse_nonce_and_hash() {
        let header = "script-src 'nonce-abc123' 'sha256-xyz789'";
        let p = CspPolicy::parse(header).unwrap();
        let srcs = p.directives().get("script-src").unwrap();
        assert_eq!(srcs[0], Source::Nonce("abc123".to_string()));
        assert_eq!(srcs[1], Source::Hash("sha256-xyz789".to_string()));
    }

    #[test]
    fn test_upgrade_insecure_requests() {
        let mut p = CspPolicy::new();
        p.upgrade_insecure();
        let header = p.to_header_value();
        assert!(header.contains("upgrade-insecure-requests"));
    }

    #[test]
    fn test_report_uri() {
        let mut p = CspPolicy::new();
        p.report_uri("https://report.example.com/csp");
        let header = p.to_header_value();
        assert!(header.contains("report-uri https://report.example.com/csp"));
    }

    #[test]
    fn test_allows_wildcard() {
        let mut p = CspPolicy::new();
        p.add(Directive::ImgSrc, Source::Wildcard);
        assert!(p.allows(Directive::ImgSrc, "https://anything.com"));
    }

    #[test]
    fn test_allows_none() {
        let mut p = CspPolicy::new();
        p.add(Directive::ObjectSrc, Source::None);
        assert!(!p.allows(Directive::ObjectSrc, "https://evil.com"));
    }

    #[test]
    fn test_allows_host() {
        let mut p = CspPolicy::new();
        p.add(Directive::ScriptSrc, Source::Host("https://cdn.example.com".to_string()));
        assert!(p.allows(Directive::ScriptSrc, "https://cdn.example.com"));
        assert!(!p.allows(Directive::ScriptSrc, "https://evil.com"));
    }

    #[test]
    fn test_allows_wildcard_host() {
        let mut p = CspPolicy::new();
        p.add(Directive::ImgSrc, Source::Host("*.example.com".to_string()));
        assert!(p.allows(Directive::ImgSrc, "cdn.example.com"));
        assert!(!p.allows(Directive::ImgSrc, "evil.com"));
    }

    #[test]
    fn test_fallback_to_default_src() {
        let mut p = CspPolicy::new();
        p.add(Directive::DefaultSrc, Source::CSelf);
        // No explicit script-src, so should fall back to default-src.
        assert!(p.allows(Directive::ScriptSrc, "self"));
    }

    #[test]
    fn test_parse_unknown_directive() {
        let result = CspPolicy::parse("fake-directive 'self'");
        assert!(result.is_err());
    }

    #[test]
    fn test_scheme_source() {
        let mut p = CspPolicy::new();
        p.add(Directive::ImgSrc, Source::Scheme("data".to_string()));
        assert!(p.allows(Directive::ImgSrc, "data:image/png;base64,..."));
    }

    #[test]
    fn test_block_mixed_content() {
        let mut p = CspPolicy::new();
        p.block_mixed_content();
        assert!(p.has_directive(Directive::BlockAllMixedContent));
    }

    #[test]
    fn test_display() {
        let mut p = CspPolicy::new();
        p.add(Directive::DefaultSrc, Source::CSelf);
        let s = format!("{p}");
        assert_eq!(s, "default-src 'self'");
    }
}
