//! URI parser (RFC 3986).
//!
//! Replaces `url` / `http::Uri` with a pure-Rust URI parser.
//! Supports scheme, authority (userinfo@host:port), path, query string,
//! fragment, percent-encoding/decoding, relative reference resolution,
//! URI normalization, and IP literal parsing.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────

/// URI parse errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UriError {
    /// Empty URI.
    Empty,
    /// Invalid scheme.
    InvalidScheme(String),
    /// Invalid percent-encoding.
    InvalidPercentEncoding(String),
    /// Invalid port number.
    InvalidPort(String),
    /// Invalid IP literal.
    InvalidIpLiteral(String),
    /// Invalid host.
    InvalidHost(String),
    /// Relative reference resolution failed.
    ResolutionError(String),
}

impl fmt::Display for UriError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "empty URI"),
            Self::InvalidScheme(s) => write!(f, "invalid scheme: {s}"),
            Self::InvalidPercentEncoding(s) => write!(f, "invalid percent encoding: {s}"),
            Self::InvalidPort(p) => write!(f, "invalid port: {p}"),
            Self::InvalidIpLiteral(s) => write!(f, "invalid IP literal: {s}"),
            Self::InvalidHost(h) => write!(f, "invalid host: {h}"),
            Self::ResolutionError(msg) => write!(f, "resolution error: {msg}"),
        }
    }
}

impl std::error::Error for UriError {}

// ── Percent Encoding ────────────────────────────────────────

/// Percent-encode a string. Encodes all characters except unreserved
/// (ALPHA / DIGIT / "-" / "." / "_" / "~").
pub fn percent_encode(input: &str) -> String {
    let mut out = String::new();
    for byte in input.bytes() {
        if is_unreserved(byte) {
            out.push(byte as char);
        } else {
            out.push('%');
            out.push_str(&format!("{byte:02X}"));
        }
    }
    out
}

/// Percent-decode a string.
pub fn percent_decode(input: &str) -> Result<String, UriError> {
    let mut out = Vec::new();
    let bytes = input.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            if i + 2 >= bytes.len() {
                return Err(UriError::InvalidPercentEncoding(input.to_string()));
            }
            let hi = hex_val(bytes[i + 1])
                .ok_or_else(|| UriError::InvalidPercentEncoding(input.to_string()))?;
            let lo = hex_val(bytes[i + 2])
                .ok_or_else(|| UriError::InvalidPercentEncoding(input.to_string()))?;
            out.push((hi << 4) | lo);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).map_err(|_| UriError::InvalidPercentEncoding(input.to_string()))
}

fn hex_val(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn is_unreserved(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'.' || byte == b'_' || byte == b'~'
}

// ── URI Components ──────────────────────────────────────────

/// A parsed URI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Uri {
    pub scheme: Option<String>,
    pub userinfo: Option<String>,
    pub host: Option<String>,
    pub port: Option<u16>,
    pub path: String,
    pub query: Option<String>,
    pub fragment: Option<String>,
}

impl Uri {
    /// Parse a URI from a string.
    pub fn parse(input: &str) -> Result<Self, UriError> {
        if input.is_empty() {
            return Err(UriError::Empty);
        }

        let mut remaining = input;
        let mut scheme = None;
        let mut userinfo = None;
        let mut host = None;
        let mut port = None;
        let mut query = None;
        let mut fragment = None;

        // Extract fragment
        if let Some(hash_pos) = remaining.rfind('#') {
            fragment = Some(remaining[hash_pos + 1..].to_string());
            remaining = &remaining[..hash_pos];
        }

        // Extract query
        if let Some(q_pos) = remaining.find('?') {
            query = Some(remaining[q_pos + 1..].to_string());
            remaining = &remaining[..q_pos];
        }

        // Extract scheme
        if let Some(colon_pos) = remaining.find(':') {
            let potential_scheme = &remaining[..colon_pos];
            if is_valid_scheme(potential_scheme) {
                scheme = Some(potential_scheme.to_lowercase());
                remaining = &remaining[colon_pos + 1..];
            }
        }

        // Extract authority
        if remaining.starts_with("//") {
            remaining = &remaining[2..];
            // Find end of authority (first / or end)
            let auth_end = remaining.find('/').unwrap_or(remaining.len());
            let authority = &remaining[..auth_end];
            remaining = &remaining[auth_end..];

            // Parse authority: [userinfo@]host[:port]
            let host_port;
            if let Some(at_pos) = authority.rfind('@') {
                userinfo = Some(authority[..at_pos].to_string());
                host_port = &authority[at_pos + 1..];
            } else {
                host_port = authority;
            }

            // Parse host:port
            if host_port.starts_with('[') {
                // IP literal (IPv6)
                let bracket_end = host_port
                    .find(']')
                    .ok_or_else(|| UriError::InvalidIpLiteral(host_port.to_string()))?;
                host = Some(host_port[..bracket_end + 1].to_string());
                let after_bracket = &host_port[bracket_end + 1..];
                if let Some(colon_rest) = after_bracket.strip_prefix(':') {
                    if !colon_rest.is_empty() {
                        port = Some(
                            colon_rest
                                .parse()
                                .map_err(|_| UriError::InvalidPort(colon_rest.to_string()))?,
                        );
                    }
                }
            } else if let Some(colon_pos) = host_port.rfind(':') {
                let h = &host_port[..colon_pos];
                let p = &host_port[colon_pos + 1..];
                host = Some(h.to_string());
                if !p.is_empty() {
                    port = Some(
                        p.parse()
                            .map_err(|_| UriError::InvalidPort(p.to_string()))?,
                    );
                }
            } else if !host_port.is_empty() {
                host = Some(host_port.to_string());
            }
        }

        let path = remaining.to_string();

        Ok(Self { scheme, userinfo, host, port, path, query, fragment })
    }

    /// Whether this URI has an authority component.
    pub fn has_authority(&self) -> bool {
        self.host.is_some()
    }

    /// Get the authority string (host:port with optional userinfo).
    pub fn authority(&self) -> Option<String> {
        self.host.as_ref().map(|h| {
            let mut auth = String::new();
            if let Some(ui) = &self.userinfo {
                auth.push_str(ui);
                auth.push('@');
            }
            auth.push_str(h);
            if let Some(p) = self.port {
                auth.push(':');
                auth.push_str(&p.to_string());
            }
            auth
        })
    }

    /// Reconstruct the URI string.
    pub fn to_string(&self) -> String {
        let mut out = String::new();
        if let Some(scheme) = &self.scheme {
            out.push_str(scheme);
            out.push(':');
        }
        if self.has_authority() {
            out.push_str("//");
            if let Some(ui) = &self.userinfo {
                out.push_str(ui);
                out.push('@');
            }
            if let Some(host) = &self.host {
                out.push_str(host);
            }
            if let Some(port) = self.port {
                out.push(':');
                out.push_str(&port.to_string());
            }
        }
        out.push_str(&self.path);
        if let Some(q) = &self.query {
            out.push('?');
            out.push_str(q);
        }
        if let Some(f) = &self.fragment {
            out.push('#');
            out.push_str(f);
        }
        out
    }

    /// Normalize a URI: lowercase scheme/host, remove default port,
    /// remove dot segments, decode unreserved percent-encoded chars.
    pub fn normalize(&self) -> Self {
        let mut uri = self.clone();
        if let Some(s) = &uri.scheme {
            uri.scheme = Some(s.to_lowercase());
        }
        if let Some(h) = &uri.host {
            uri.host = Some(h.to_lowercase());
        }
        // Remove default ports
        if let (Some(scheme), Some(port)) = (&uri.scheme, uri.port) {
            if (scheme == "http" && port == 80) || (scheme == "https" && port == 443) {
                uri.port = None;
            }
        }
        // Remove dot segments from path
        uri.path = remove_dot_segments(&uri.path);
        uri
    }

    /// Resolve a relative reference against this base URI (RFC 3986 Section 5.3).
    pub fn resolve(&self, reference: &str) -> Result<Self, UriError> {
        let r = Uri::parse(reference)?;

        if r.scheme.is_some() {
            // Reference has scheme — use it almost as-is
            return Ok(Uri {
                scheme: r.scheme,
                userinfo: r.userinfo,
                host: r.host,
                port: r.port,
                path: remove_dot_segments(&r.path),
                query: r.query,
                fragment: r.fragment,
            });
        }

        if r.has_authority() {
            return Ok(Uri {
                scheme: self.scheme.clone(),
                userinfo: r.userinfo,
                host: r.host,
                port: r.port,
                path: remove_dot_segments(&r.path),
                query: r.query,
                fragment: r.fragment,
            });
        }

        if r.path.is_empty() {
            return Ok(Uri {
                scheme: self.scheme.clone(),
                userinfo: self.userinfo.clone(),
                host: self.host.clone(),
                port: self.port,
                path: self.path.clone(),
                query: r.query.or_else(|| self.query.clone()),
                fragment: r.fragment,
            });
        }

        let path = if r.path.starts_with('/') {
            remove_dot_segments(&r.path)
        } else {
            let base = merge_paths(self, &r.path);
            remove_dot_segments(&base)
        };

        Ok(Uri {
            scheme: self.scheme.clone(),
            userinfo: self.userinfo.clone(),
            host: self.host.clone(),
            port: self.port,
            path,
            query: r.query,
            fragment: r.fragment,
        })
    }

    /// Parse query parameters into key-value pairs.
    pub fn query_params(&self) -> Vec<(String, String)> {
        match &self.query {
            None => Vec::new(),
            Some(q) => parse_query_string(q),
        }
    }
}

impl fmt::Display for Uri {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

// ── Helpers ─────────────────────────────────────────────────

fn is_valid_scheme(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    let bytes = s.as_bytes();
    if !bytes[0].is_ascii_alphabetic() {
        return false;
    }
    bytes[1..].iter().all(|b| {
        b.is_ascii_alphanumeric() || *b == b'+' || *b == b'-' || *b == b'.'
    })
}

/// Remove dot segments from a path (RFC 3986 Section 5.2.4).
pub fn remove_dot_segments(path: &str) -> String {
    let mut input = path.to_string();
    let mut output: Vec<String> = Vec::new();

    while !input.is_empty() {
        // A
        if input.starts_with("../") {
            input = input[3..].to_string();
        } else if input.starts_with("./") {
            input = input[2..].to_string();
        }
        // B
        else if input.starts_with("/./") {
            input = format!("/{}", &input[3..]);
        } else if input == "/." {
            input = "/".to_string();
        }
        // C
        else if input.starts_with("/../") {
            input = format!("/{}", &input[4..]);
            output.pop();
        } else if input == "/.." {
            input = "/".to_string();
            output.pop();
        }
        // D
        else if input == "." || input == ".." {
            input.clear();
        }
        // E
        else {
            let seg;
            if input.starts_with('/') {
                let next_slash = input[1..].find('/').map(|p| p + 1).unwrap_or(input.len());
                seg = input[..next_slash].to_string();
                input = input[next_slash..].to_string();
            } else {
                let next_slash = input.find('/').unwrap_or(input.len());
                seg = input[..next_slash].to_string();
                input = input[next_slash..].to_string();
            }
            output.push(seg);
        }
    }

    output.join("")
}

fn merge_paths(base: &Uri, rel_path: &str) -> String {
    if base.has_authority() && base.path.is_empty() {
        format!("/{rel_path}")
    } else {
        match base.path.rfind('/') {
            Some(pos) => format!("{}{rel_path}", &base.path[..=pos]),
            None => rel_path.to_string(),
        }
    }
}

/// Parse a query string into key-value pairs.
pub fn parse_query_string(query: &str) -> Vec<(String, String)> {
    query
        .split('&')
        .filter(|s| !s.is_empty())
        .map(|pair| {
            let mut parts = pair.splitn(2, '=');
            let key = percent_decode(parts.next().unwrap_or("")).unwrap_or_default();
            let value = percent_decode(parts.next().unwrap_or("")).unwrap_or_default();
            (key, value)
        })
        .collect()
}

/// Build a query string from key-value pairs.
pub fn build_query_string(params: &[(&str, &str)]) -> String {
    params
        .iter()
        .map(|(k, v)| format!("{}={}", percent_encode(k), percent_encode(v)))
        .collect::<Vec<_>>()
        .join("&")
}

// ── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_full_uri() {
        let uri = Uri::parse("https://user:pass@example.com:8080/path/to?q=1&r=2#frag").unwrap();
        assert_eq!(uri.scheme, Some("https".into()));
        assert_eq!(uri.userinfo, Some("user:pass".into()));
        assert_eq!(uri.host, Some("example.com".into()));
        assert_eq!(uri.port, Some(8080));
        assert_eq!(uri.path, "/path/to");
        assert_eq!(uri.query, Some("q=1&r=2".into()));
        assert_eq!(uri.fragment, Some("frag".into()));
    }

    #[test]
    fn parse_simple_uri() {
        let uri = Uri::parse("http://example.com/").unwrap();
        assert_eq!(uri.scheme, Some("http".into()));
        assert_eq!(uri.host, Some("example.com".into()));
        assert_eq!(uri.port, None);
        assert_eq!(uri.path, "/");
    }

    #[test]
    fn parse_relative_reference() {
        let uri = Uri::parse("/path/to/resource?key=val").unwrap();
        assert_eq!(uri.scheme, None);
        assert_eq!(uri.host, None);
        assert_eq!(uri.path, "/path/to/resource");
        assert_eq!(uri.query, Some("key=val".into()));
    }

    #[test]
    fn percent_encoding_roundtrip() {
        let encoded = percent_encode("hello world/path?q=1&r=2");
        assert!(encoded.contains("%20"));
        let decoded = percent_decode(&encoded).unwrap();
        assert_eq!(decoded, "hello world/path?q=1&r=2");
    }

    #[test]
    fn percent_decode_invalid() {
        assert!(percent_decode("%ZZ").is_err());
        assert!(percent_decode("%2").is_err());
    }

    #[test]
    fn uri_to_string_roundtrip() {
        let input = "https://user@host.com:443/path?query#frag";
        let uri = Uri::parse(input).unwrap();
        assert_eq!(uri.to_string(), input);
    }

    #[test]
    fn normalize_uri() {
        let uri = Uri::parse("HTTP://Example.COM:80/a/b/../c").unwrap();
        let normalized = uri.normalize();
        assert_eq!(normalized.scheme, Some("http".into()));
        assert_eq!(normalized.host, Some("example.com".into()));
        assert_eq!(normalized.port, None);
        assert_eq!(normalized.path, "/a/c");
    }

    #[test]
    fn resolve_relative() {
        let base = Uri::parse("http://example.com/a/b/c").unwrap();
        let resolved = base.resolve("../d").unwrap();
        assert_eq!(resolved.path, "/a/d");
        assert_eq!(resolved.host, Some("example.com".into()));
    }

    #[test]
    fn resolve_absolute_path() {
        let base = Uri::parse("http://example.com/a/b").unwrap();
        let resolved = base.resolve("/x/y").unwrap();
        assert_eq!(resolved.path, "/x/y");
    }

    #[test]
    fn resolve_with_scheme() {
        let base = Uri::parse("http://example.com/a").unwrap();
        let resolved = base.resolve("https://other.com/b").unwrap();
        assert_eq!(resolved.scheme, Some("https".into()));
        assert_eq!(resolved.host, Some("other.com".into()));
    }

    #[test]
    fn ipv6_host() {
        let uri = Uri::parse("http://[::1]:8080/path").unwrap();
        assert_eq!(uri.host, Some("[::1]".into()));
        assert_eq!(uri.port, Some(8080));
    }

    #[test]
    fn query_params() {
        let uri = Uri::parse("http://x.com/p?a=1&b=hello%20world&c=").unwrap();
        let params = uri.query_params();
        assert_eq!(params.len(), 3);
        assert_eq!(params[0], ("a".into(), "1".into()));
        assert_eq!(params[1], ("b".into(), "hello world".into()));
        assert_eq!(params[2], ("c".into(), "".into()));
    }

    #[test]
    fn build_query() {
        let qs = build_query_string(&[("key", "val ue"), ("foo", "bar")]);
        assert!(qs.contains("key=val%20ue"));
        assert!(qs.contains("foo=bar"));
    }

    #[test]
    fn remove_dot_segments_cases() {
        assert_eq!(remove_dot_segments("/a/b/c/./../../g"), "/a/g");
        assert_eq!(remove_dot_segments("mid/content=5/../6"), "mid/6");
        assert_eq!(remove_dot_segments("/a/b/../c"), "/a/c");
    }

    #[test]
    fn empty_uri_error() {
        assert_eq!(Uri::parse("").unwrap_err(), UriError::Empty);
    }

    #[test]
    fn authority_string() {
        let uri = Uri::parse("http://admin:secret@db.local:5432/mydb").unwrap();
        assert_eq!(uri.authority(), Some("admin:secret@db.local:5432".into()));
    }
}
