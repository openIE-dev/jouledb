//! Cookie jar — parse, store, match, and serialize HTTP cookies.
//!
//! Replaces `tough-cookie`, `cookie`, and `js-cookie` with pure Rust.
//! Parses Set-Cookie headers, handles domain/path matching, secure/httponly/
//! samesite attributes, expiry tracking, specificity sorting, serialization.

use std::fmt;

// ── SameSite ───────────────────────────────────────────────────

/// SameSite cookie attribute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SameSite {
    Strict,
    Lax,
    None,
}

impl SameSite {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Strict => "Strict",
            Self::Lax => "Lax",
            Self::None => "None",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "strict" => Some(Self::Strict),
            "lax" => Some(Self::Lax),
            "none" => Some(Self::None),
            _ => None,
        }
    }
}

// ── Cookie ─────────────────────────────────────────────────────

/// A parsed HTTP cookie.
#[derive(Debug, Clone)]
pub struct Cookie {
    pub name: String,
    pub value: String,
    pub domain: Option<String>,
    pub path: Option<String>,
    pub secure: bool,
    pub http_only: bool,
    pub same_site: Option<SameSite>,
    /// Max-Age in seconds (from Set-Cookie).
    pub max_age: Option<i64>,
    /// Expires as epoch seconds.
    pub expires_epoch_s: Option<u64>,
    /// When the cookie was created (epoch seconds).
    pub created_epoch_s: u64,
}

impl Cookie {
    pub fn new(name: &str, value: &str) -> Self {
        Self {
            name: name.to_string(),
            value: value.to_string(),
            domain: None,
            path: None,
            secure: false,
            http_only: false,
            same_site: None,
            max_age: None,
            expires_epoch_s: None,
            created_epoch_s: 0,
        }
    }

    pub fn domain(mut self, d: &str) -> Self {
        self.domain = Some(d.to_ascii_lowercase());
        self
    }

    pub fn path(mut self, p: &str) -> Self {
        self.path = Some(p.to_string());
        self
    }

    pub fn secure(mut self, s: bool) -> Self {
        self.secure = s;
        self
    }

    pub fn http_only(mut self, h: bool) -> Self {
        self.http_only = h;
        self
    }

    pub fn same_site_attr(mut self, ss: SameSite) -> Self {
        self.same_site = Some(ss);
        self
    }

    pub fn max_age_seconds(mut self, seconds: i64) -> Self {
        self.max_age = Some(seconds);
        self
    }

    pub fn expires(mut self, epoch_s: u64) -> Self {
        self.expires_epoch_s = Some(epoch_s);
        self
    }

    /// Check if the cookie is expired at the given time.
    pub fn is_expired(&self, now_epoch_s: u64) -> bool {
        if let Some(max_age) = self.max_age {
            if max_age <= 0 {
                return true;
            }
            if now_epoch_s > self.created_epoch_s + (max_age as u64) {
                return true;
            }
        }
        if let Some(exp) = self.expires_epoch_s {
            if now_epoch_s > exp {
                return true;
            }
        }
        false
    }

    /// Check if this cookie matches the given domain.
    pub fn matches_domain(&self, request_domain: &str) -> bool {
        let req = request_domain.to_ascii_lowercase();
        match &self.domain {
            None => true, // No domain restriction — host-only.
            Some(d) => {
                if req == *d {
                    return true;
                }
                // Domain cookie: ".example.com" matches "sub.example.com".
                if d.starts_with('.') {
                    req.ends_with(d.as_str()) || req == d[1..]
                } else {
                    req == *d || req.ends_with(&format!(".{}", d))
                }
            }
        }
    }

    /// Check if this cookie matches the given path.
    pub fn matches_path(&self, request_path: &str) -> bool {
        match &self.path {
            None => true,
            Some(p) => {
                if request_path == p {
                    return true;
                }
                if request_path.starts_with(p) {
                    // Cookie path "/foo" should match "/foo/bar" but not "/foobar".
                    if p.ends_with('/') {
                        return true;
                    }
                    let rest = &request_path[p.len()..];
                    return rest.starts_with('/');
                }
                false
            }
        }
    }

    /// Specificity score for sorting: longer path = more specific.
    pub fn specificity(&self) -> usize {
        self.path.as_ref().map_or(0, |p| p.len())
    }

    /// Serialize to a `Cookie:` header value (name=value).
    pub fn to_header_value(&self) -> String {
        format!("{}={}", self.name, self.value)
    }

    /// Serialize to a `Set-Cookie` header value.
    pub fn to_set_cookie(&self) -> String {
        let mut parts = vec![format!("{}={}", self.name, self.value)];
        if let Some(d) = &self.domain {
            parts.push(format!("Domain={}", d));
        }
        if let Some(p) = &self.path {
            parts.push(format!("Path={}", p));
        }
        if self.secure {
            parts.push("Secure".to_string());
        }
        if self.http_only {
            parts.push("HttpOnly".to_string());
        }
        if let Some(ss) = &self.same_site {
            parts.push(format!("SameSite={}", ss.as_str()));
        }
        if let Some(ma) = self.max_age {
            parts.push(format!("Max-Age={}", ma));
        }
        parts.join("; ")
    }

    /// Parse a `Set-Cookie` header value.
    pub fn parse_set_cookie(header: &str, now_epoch_s: u64) -> Option<Self> {
        let mut parts = header.split(';');
        let first = parts.next()?.trim();
        let eq_idx = first.find('=')?;
        let name = first[..eq_idx].trim().to_string();
        let value = first[eq_idx + 1..].trim().to_string();

        if name.is_empty() {
            return None;
        }

        let mut cookie = Cookie::new(&name, &value);
        cookie.created_epoch_s = now_epoch_s;

        for attr in parts {
            let attr = attr.trim();
            if attr.is_empty() {
                continue;
            }
            if let Some(eq) = attr.find('=') {
                let key = attr[..eq].trim().to_ascii_lowercase();
                let val = attr[eq + 1..].trim();
                match key.as_str() {
                    "domain" => {
                        cookie.domain = Some(val.to_ascii_lowercase());
                    }
                    "path" => {
                        cookie.path = Some(val.to_string());
                    }
                    "max-age" => {
                        if let Ok(n) = val.parse::<i64>() {
                            cookie.max_age = Some(n);
                        }
                    }
                    "samesite" => {
                        cookie.same_site = SameSite::from_str(val);
                    }
                    _ => {}
                }
            } else {
                match attr.to_ascii_lowercase().as_str() {
                    "secure" => cookie.secure = true,
                    "httponly" => cookie.http_only = true,
                    _ => {}
                }
            }
        }

        Some(cookie)
    }
}

impl fmt::Display for Cookie {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}={}", self.name, self.value)
    }
}

// ── Cookie jar ─────────────────────────────────────────────────

/// A cookie jar: stores cookies, matches by domain/path, handles expiry.
#[derive(Debug, Default)]
pub struct CookieJar {
    cookies: Vec<Cookie>,
}

impl CookieJar {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or update a cookie (replacing same name+domain+path).
    pub fn insert(&mut self, cookie: Cookie) {
        self.cookies.retain(|c| {
            !(c.name == cookie.name && c.domain == cookie.domain && c.path == cookie.path)
        });
        self.cookies.push(cookie);
    }

    /// Parse and insert a Set-Cookie header.
    pub fn insert_set_cookie(&mut self, header: &str, now_epoch_s: u64) -> bool {
        if let Some(cookie) = Cookie::parse_set_cookie(header, now_epoch_s) {
            self.insert(cookie);
            true
        } else {
            false
        }
    }

    /// Get all cookies matching a request (domain + path), sorted by specificity.
    pub fn get_matching(
        &self,
        domain: &str,
        path: &str,
        is_secure: bool,
        now_epoch_s: u64,
    ) -> Vec<&Cookie> {
        let mut matches: Vec<&Cookie> = self
            .cookies
            .iter()
            .filter(|c| {
                !c.is_expired(now_epoch_s)
                    && c.matches_domain(domain)
                    && c.matches_path(path)
                    && (!c.secure || is_secure)
            })
            .collect();
        // Sort by specificity (most specific first).
        matches.sort_by(|a, b| b.specificity().cmp(&a.specificity()));
        matches
    }

    /// Build a `Cookie:` header value for a request.
    pub fn cookie_header(
        &self,
        domain: &str,
        path: &str,
        is_secure: bool,
        now_epoch_s: u64,
    ) -> String {
        self.get_matching(domain, path, is_secure, now_epoch_s)
            .iter()
            .map(|c| c.to_header_value())
            .collect::<Vec<_>>()
            .join("; ")
    }

    /// Remove expired cookies.
    pub fn evict_expired(&mut self, now_epoch_s: u64) {
        self.cookies.retain(|c| !c.is_expired(now_epoch_s));
    }

    /// Remove a specific cookie by name, domain, path.
    pub fn remove(&mut self, name: &str, domain: Option<&str>, path: Option<&str>) {
        self.cookies.retain(|c| {
            !(c.name == name
                && c.domain.as_deref() == domain
                && c.path.as_deref() == path)
        });
    }

    pub fn len(&self) -> usize {
        self.cookies.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cookies.is_empty()
    }

    pub fn clear(&mut self) {
        self.cookies.clear();
    }

    pub fn iter(&self) -> impl Iterator<Item = &Cookie> {
        self.cookies.iter()
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_set_cookie() {
        let c = Cookie::parse_set_cookie("session=abc123", 1000).unwrap();
        assert_eq!(c.name, "session");
        assert_eq!(c.value, "abc123");
        assert_eq!(c.created_epoch_s, 1000);
    }

    #[test]
    fn parse_full_set_cookie() {
        let header = "id=xyz; Domain=example.com; Path=/app; Secure; HttpOnly; SameSite=Strict; Max-Age=3600";
        let c = Cookie::parse_set_cookie(header, 5000).unwrap();
        assert_eq!(c.name, "id");
        assert_eq!(c.value, "xyz");
        assert_eq!(c.domain.as_deref(), Some("example.com"));
        assert_eq!(c.path.as_deref(), Some("/app"));
        assert!(c.secure);
        assert!(c.http_only);
        assert_eq!(c.same_site, Some(SameSite::Strict));
        assert_eq!(c.max_age, Some(3600));
    }

    #[test]
    fn cookie_expiry_max_age() {
        let c = Cookie::new("a", "1")
            .max_age_seconds(100);
        // Created at epoch 0, so expires at epoch 100.
        assert!(!c.is_expired(50));
        assert!(c.is_expired(101));
    }

    #[test]
    fn cookie_expiry_negative_max_age() {
        let c = Cookie::new("a", "1").max_age_seconds(-1);
        assert!(c.is_expired(0));
    }

    #[test]
    fn cookie_expiry_expires_field() {
        let c = Cookie::new("a", "1").expires(500);
        assert!(!c.is_expired(499));
        assert!(c.is_expired(501));
    }

    #[test]
    fn domain_matching_exact() {
        let c = Cookie::new("a", "1").domain("example.com");
        assert!(c.matches_domain("example.com"));
        assert!(c.matches_domain("sub.example.com"));
        assert!(!c.matches_domain("notexample.com"));
    }

    #[test]
    fn domain_matching_dot_prefix() {
        let c = Cookie::new("a", "1").domain(".example.com");
        assert!(c.matches_domain("example.com"));
        assert!(c.matches_domain("sub.example.com"));
        assert!(!c.matches_domain("other.com"));
    }

    #[test]
    fn path_matching() {
        let c = Cookie::new("a", "1").path("/app");
        assert!(c.matches_path("/app"));
        assert!(c.matches_path("/app/page"));
        assert!(!c.matches_path("/application"));
        assert!(!c.matches_path("/other"));
    }

    #[test]
    fn path_matching_trailing_slash() {
        let c = Cookie::new("a", "1").path("/app/");
        assert!(c.matches_path("/app/page"));
        assert!(c.matches_path("/app/"));
    }

    #[test]
    fn cookie_jar_insert_and_match() {
        let mut jar = CookieJar::new();
        jar.insert(
            Cookie::new("session", "abc")
                .domain("example.com")
                .path("/"),
        );
        jar.insert(
            Cookie::new("pref", "dark")
                .domain("example.com")
                .path("/app"),
        );

        let matches = jar.get_matching("example.com", "/app/page", true, 0);
        assert_eq!(matches.len(), 2);
        // More specific path first.
        assert_eq!(matches[0].name, "pref");
        assert_eq!(matches[1].name, "session");
    }

    #[test]
    fn cookie_jar_header() {
        let mut jar = CookieJar::new();
        jar.insert(Cookie::new("a", "1").domain("x.com").path("/"));
        jar.insert(Cookie::new("b", "2").domain("x.com").path("/"));
        let header = jar.cookie_header("x.com", "/", true, 0);
        assert!(header.contains("a=1"));
        assert!(header.contains("b=2"));
    }

    #[test]
    fn cookie_jar_evict_expired() {
        let mut jar = CookieJar::new();
        jar.insert(Cookie::new("old", "1").max_age_seconds(10));
        jar.insert(Cookie::new("new", "2").max_age_seconds(1000));
        jar.evict_expired(50);
        assert_eq!(jar.len(), 1);
        assert_eq!(jar.iter().next().unwrap().name, "new");
    }

    #[test]
    fn cookie_jar_replace_duplicate() {
        let mut jar = CookieJar::new();
        jar.insert(Cookie::new("k", "v1").domain("x.com").path("/"));
        jar.insert(Cookie::new("k", "v2").domain("x.com").path("/"));
        assert_eq!(jar.len(), 1);
        assert_eq!(jar.iter().next().unwrap().value, "v2");
    }

    #[test]
    fn secure_cookie_not_sent_over_http() {
        let mut jar = CookieJar::new();
        jar.insert(
            Cookie::new("sec", "1")
                .domain("x.com")
                .path("/")
                .secure(true),
        );
        assert_eq!(jar.get_matching("x.com", "/", false, 0).len(), 0);
        assert_eq!(jar.get_matching("x.com", "/", true, 0).len(), 1);
    }

    #[test]
    fn set_cookie_serialization() {
        let c = Cookie::new("sid", "xyz")
            .domain("example.com")
            .path("/")
            .secure(true)
            .http_only(true)
            .same_site_attr(SameSite::Lax)
            .max_age_seconds(3600);
        let s = c.to_set_cookie();
        assert!(s.contains("sid=xyz"));
        assert!(s.contains("Domain=example.com"));
        assert!(s.contains("Secure"));
        assert!(s.contains("HttpOnly"));
        assert!(s.contains("SameSite=Lax"));
        assert!(s.contains("Max-Age=3600"));
    }

    #[test]
    fn insert_from_set_cookie_header() {
        let mut jar = CookieJar::new();
        assert!(jar.insert_set_cookie(
            "token=abc; Domain=api.com; Path=/v1; Secure",
            100,
        ));
        assert_eq!(jar.len(), 1);
        let c = jar.iter().next().unwrap();
        assert_eq!(c.name, "token");
        assert!(c.secure);
    }
}
