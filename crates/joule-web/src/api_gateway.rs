//! API gateway core — route matching, upstream routing, request/response transformation,
//! header manipulation, path rewriting, and per-route timeout configuration.
//!
//! Replaces `express-gateway`, `kong`, and similar JS/Node gateway libraries with a
//! pure-Rust API gateway engine supporting pattern-based routing and upstream dispatch.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── Errors ─────────────────────────────────────────────────────

/// API gateway errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GatewayError {
    /// No route matched the request.
    NoRouteMatch { method: String, path: String },
    /// Upstream not found by name.
    UpstreamNotFound(String),
    /// Duplicate route ID.
    DuplicateRoute(String),
    /// Invalid path pattern.
    InvalidPattern(String),
    /// Request timeout exceeded.
    Timeout { route_id: String, timeout_ms: u64 },
    /// Transformation error.
    TransformError(String),
}

impl fmt::Display for GatewayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoRouteMatch { method, path } => {
                write!(f, "no route match: {method} {path}")
            }
            Self::UpstreamNotFound(name) => write!(f, "upstream not found: {name}"),
            Self::DuplicateRoute(id) => write!(f, "duplicate route: {id}"),
            Self::InvalidPattern(p) => write!(f, "invalid pattern: {p}"),
            Self::Timeout { route_id, timeout_ms } => {
                write!(f, "timeout on route {route_id}: {timeout_ms}ms")
            }
            Self::TransformError(msg) => write!(f, "transform error: {msg}"),
        }
    }
}

impl std::error::Error for GatewayError {}

// ── HTTP Method ────────────────────────────────────────────────

/// HTTP method for route matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HttpMethod {
    Get,
    Post,
    Put,
    Delete,
    Patch,
    Head,
    Options,
    Any,
}

impl HttpMethod {
    /// Check if this method matches a candidate.
    pub fn matches(self, candidate: HttpMethod) -> bool {
        self == HttpMethod::Any || candidate == HttpMethod::Any || self == candidate
    }
}

impl fmt::Display for HttpMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Get => "GET",
            Self::Post => "POST",
            Self::Put => "PUT",
            Self::Delete => "DELETE",
            Self::Patch => "PATCH",
            Self::Head => "HEAD",
            Self::Options => "OPTIONS",
            Self::Any => "*",
        };
        f.write_str(s)
    }
}

// ── Path Pattern ───────────────────────────────────────────────

/// A segment in a path pattern.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PatternSegment {
    /// Exact literal match, e.g. `users`.
    Literal(String),
    /// Named parameter, e.g. `:id`.
    Param(String),
    /// Wildcard matches any single segment.
    Wildcard,
    /// Glob matches zero or more remaining segments.
    Glob,
}

/// Compiled path pattern for route matching.
#[derive(Debug, Clone)]
pub struct PathPattern {
    segments: Vec<PatternSegment>,
    raw: String,
}

impl PathPattern {
    /// Parse a path pattern string.
    pub fn parse(pattern: &str) -> Result<Self, GatewayError> {
        if pattern.is_empty() {
            return Err(GatewayError::InvalidPattern("empty pattern".into()));
        }
        let trimmed = pattern.strip_prefix('/').unwrap_or(pattern);
        if trimmed.is_empty() {
            return Ok(Self { segments: vec![], raw: pattern.to_string() });
        }
        let mut segments = Vec::new();
        for part in trimmed.split('/') {
            let seg = if part == "*" {
                PatternSegment::Wildcard
            } else if part == "**" {
                PatternSegment::Glob
            } else if let Some(name) = part.strip_prefix(':') {
                if name.is_empty() {
                    return Err(GatewayError::InvalidPattern(
                        "empty parameter name".into(),
                    ));
                }
                PatternSegment::Param(name.to_string())
            } else {
                PatternSegment::Literal(part.to_string())
            };
            segments.push(seg);
        }
        Ok(Self { segments, raw: pattern.to_string() })
    }

    /// Match a concrete path against this pattern, returning extracted params.
    pub fn match_path(&self, path: &str) -> Option<HashMap<String, String>> {
        let trimmed = path.strip_prefix('/').unwrap_or(path);
        let parts: Vec<&str> = if trimmed.is_empty() {
            vec![]
        } else {
            trimmed.split('/').collect()
        };

        let mut params = HashMap::new();
        let mut pi = 0;
        let mut si = 0;

        while si < self.segments.len() {
            match &self.segments[si] {
                PatternSegment::Literal(lit) => {
                    if pi >= parts.len() || parts[pi] != lit.as_str() {
                        return None;
                    }
                    pi += 1;
                }
                PatternSegment::Param(name) => {
                    if pi >= parts.len() {
                        return None;
                    }
                    params.insert(name.clone(), parts[pi].to_string());
                    pi += 1;
                }
                PatternSegment::Wildcard => {
                    if pi >= parts.len() {
                        return None;
                    }
                    pi += 1;
                }
                PatternSegment::Glob => {
                    // Glob matches rest of segments, but requires at least one.
                    if pi >= parts.len() {
                        return None;
                    }
                    return Some(params);
                }
            }
            si += 1;
        }

        if pi == parts.len() {
            Some(params)
        } else {
            None
        }
    }
}

// ── Header Manipulation ────────────────────────────────────────

/// Header manipulation action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HeaderAction {
    /// Add a header (appends if exists).
    Add { name: String, value: String },
    /// Set a header (replaces if exists).
    Set { name: String, value: String },
    /// Remove a header by name.
    Remove { name: String },
    /// Rename a header.
    Rename { from: String, to: String },
}

/// Apply header actions to a header map.
pub fn apply_header_actions(
    headers: &mut HashMap<String, Vec<String>>,
    actions: &[HeaderAction],
) {
    for action in actions {
        match action {
            HeaderAction::Add { name, value } => {
                headers
                    .entry(name.clone())
                    .or_insert_with(Vec::new)
                    .push(value.clone());
            }
            HeaderAction::Set { name, value } => {
                headers.insert(name.clone(), vec![value.clone()]);
            }
            HeaderAction::Remove { name } => {
                headers.remove(name);
            }
            HeaderAction::Rename { from, to } => {
                if let Some(vals) = headers.remove(from) {
                    headers.insert(to.clone(), vals);
                }
            }
        }
    }
}

// ── Path Rewriting ─────────────────────────────────────────────

/// Path rewrite rule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathRewrite {
    /// Prefix to strip from the incoming path.
    pub strip_prefix: Option<String>,
    /// Prefix to prepend to the path.
    pub add_prefix: Option<String>,
    /// Exact replacement (if set, replaces entire path).
    pub replace: Option<String>,
}

impl PathRewrite {
    /// Apply the rewrite to a path.
    pub fn apply(&self, path: &str) -> String {
        if let Some(replacement) = &self.replace {
            return replacement.clone();
        }
        let mut result = path.to_string();
        if let Some(prefix) = &self.strip_prefix {
            if let Some(stripped) = result.strip_prefix(prefix.as_str()) {
                result = stripped.to_string();
                if !result.starts_with('/') {
                    result = format!("/{result}");
                }
            }
        }
        if let Some(prefix) = &self.add_prefix {
            let clean = prefix.trim_end_matches('/');
            if result.starts_with('/') {
                result = format!("{clean}{result}");
            } else {
                result = format!("{clean}/{result}");
            }
        }
        result
    }
}

// ── Upstream ───────────────────────────────────────────────────

/// An upstream service target.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Upstream {
    /// Unique name for this upstream.
    pub name: String,
    /// Base URL of the upstream service.
    pub base_url: String,
    /// Weight for load balancing (higher = more traffic).
    pub weight: u32,
    /// Whether this upstream is healthy.
    pub healthy: bool,
}

// ── Route ──────────────────────────────────────────────────────

/// A gateway route definition.
#[derive(Debug, Clone)]
pub struct Route {
    /// Unique route identifier.
    pub id: String,
    /// HTTP method to match.
    pub method: HttpMethod,
    /// Path pattern.
    pub pattern: PathPattern,
    /// Name of the upstream to route to.
    pub upstream: String,
    /// Request timeout in milliseconds.
    pub timeout_ms: u64,
    /// Header actions applied to the request before forwarding.
    pub request_headers: Vec<HeaderAction>,
    /// Header actions applied to the response before returning.
    pub response_headers: Vec<HeaderAction>,
    /// Optional path rewrite.
    pub path_rewrite: Option<PathRewrite>,
    /// Priority (lower number = higher priority).
    pub priority: u32,
    /// Whether the route is enabled.
    pub enabled: bool,
}

// ── Gateway Request / Response ─────────────────────────────────

/// An incoming gateway request.
#[derive(Debug, Clone)]
pub struct GatewayRequest {
    pub method: HttpMethod,
    pub path: String,
    pub headers: HashMap<String, Vec<String>>,
    pub query_params: HashMap<String, String>,
    pub body: Option<Vec<u8>>,
}

/// A matched route result with routing information.
#[derive(Debug, Clone)]
pub struct RouteMatch {
    pub route_id: String,
    pub upstream_name: String,
    pub upstream_url: String,
    pub rewritten_path: String,
    pub path_params: HashMap<String, String>,
    pub timeout_ms: u64,
    pub request_headers: HashMap<String, Vec<String>>,
}

/// A gateway response.
#[derive(Debug, Clone)]
pub struct GatewayResponse {
    pub status: u16,
    pub headers: HashMap<String, Vec<String>>,
    pub body: Option<Vec<u8>>,
}

// ── Gateway ────────────────────────────────────────────────────

/// The API Gateway engine.
#[derive(Debug)]
pub struct ApiGateway {
    routes: Vec<Route>,
    upstreams: HashMap<String, Upstream>,
    global_request_headers: Vec<HeaderAction>,
    global_response_headers: Vec<HeaderAction>,
    default_timeout_ms: u64,
}

impl ApiGateway {
    /// Create a new gateway with default settings.
    pub fn new() -> Self {
        Self {
            routes: Vec::new(),
            upstreams: HashMap::new(),
            global_request_headers: Vec::new(),
            global_response_headers: Vec::new(),
            default_timeout_ms: 30_000,
        }
    }

    /// Set the default timeout for routes that do not specify one.
    pub fn set_default_timeout_ms(&mut self, ms: u64) {
        self.default_timeout_ms = ms;
    }

    /// Add a global request header action.
    pub fn add_global_request_header(&mut self, action: HeaderAction) {
        self.global_request_headers.push(action);
    }

    /// Add a global response header action.
    pub fn add_global_response_header(&mut self, action: HeaderAction) {
        self.global_response_headers.push(action);
    }

    /// Register an upstream.
    pub fn add_upstream(&mut self, upstream: Upstream) {
        self.upstreams.insert(upstream.name.clone(), upstream);
    }

    /// Remove an upstream by name.
    pub fn remove_upstream(&mut self, name: &str) -> bool {
        self.upstreams.remove(name).is_some()
    }

    /// Mark an upstream as healthy or unhealthy.
    pub fn set_upstream_health(&mut self, name: &str, healthy: bool) -> bool {
        if let Some(up) = self.upstreams.get_mut(name) {
            up.healthy = healthy;
            true
        } else {
            false
        }
    }

    /// Add a route to the gateway.
    pub fn add_route(&mut self, route: Route) -> Result<(), GatewayError> {
        if self.routes.iter().any(|r| r.id == route.id) {
            return Err(GatewayError::DuplicateRoute(route.id));
        }
        self.routes.push(route);
        self.routes.sort_by_key(|r| r.priority);
        Ok(())
    }

    /// Remove a route by ID.
    pub fn remove_route(&mut self, id: &str) -> bool {
        let len_before = self.routes.len();
        self.routes.retain(|r| r.id != id);
        self.routes.len() < len_before
    }

    /// Enable or disable a route.
    pub fn set_route_enabled(&mut self, id: &str, enabled: bool) -> bool {
        for r in &mut self.routes {
            if r.id == id {
                r.enabled = enabled;
                return true;
            }
        }
        false
    }

    /// Resolve a request to a route match.
    pub fn resolve(&self, request: &GatewayRequest) -> Result<RouteMatch, GatewayError> {
        for route in &self.routes {
            if !route.enabled {
                continue;
            }
            if !route.method.matches(request.method) {
                continue;
            }
            if let Some(params) = route.pattern.match_path(&request.path) {
                // Look up upstream
                let upstream = self
                    .upstreams
                    .get(&route.upstream)
                    .ok_or_else(|| GatewayError::UpstreamNotFound(route.upstream.clone()))?;

                if !upstream.healthy {
                    continue; // Skip unhealthy upstreams
                }

                // Apply path rewrite
                let rewritten = if let Some(rewrite) = &route.path_rewrite {
                    rewrite.apply(&request.path)
                } else {
                    request.path.clone()
                };

                // Build request headers
                let mut headers = request.headers.clone();
                apply_header_actions(&mut headers, &self.global_request_headers);
                apply_header_actions(&mut headers, &route.request_headers);

                let timeout = if route.timeout_ms > 0 {
                    route.timeout_ms
                } else {
                    self.default_timeout_ms
                };

                return Ok(RouteMatch {
                    route_id: route.id.clone(),
                    upstream_name: upstream.name.clone(),
                    upstream_url: format!(
                        "{}{}",
                        upstream.base_url.trim_end_matches('/'),
                        rewritten
                    ),
                    rewritten_path: rewritten,
                    path_params: params,
                    timeout_ms: timeout,
                    request_headers: headers,
                });
            }
        }

        Err(GatewayError::NoRouteMatch {
            method: request.method.to_string(),
            path: request.path.clone(),
        })
    }

    /// Transform a response using route and global header actions.
    pub fn transform_response(
        &self,
        route_id: &str,
        mut response: GatewayResponse,
    ) -> GatewayResponse {
        // Apply global response headers
        apply_header_actions(&mut response.headers, &self.global_response_headers);

        // Apply route-specific response headers
        if let Some(route) = self.routes.iter().find(|r| r.id == route_id) {
            apply_header_actions(&mut response.headers, &route.response_headers);
        }

        response
    }

    /// Get all registered route IDs.
    pub fn route_ids(&self) -> Vec<String> {
        self.routes.iter().map(|r| r.id.clone()).collect()
    }

    /// Get all upstream names.
    pub fn upstream_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.upstreams.keys().cloned().collect();
        names.sort();
        names
    }

    /// Get the number of registered routes.
    pub fn route_count(&self) -> usize {
        self.routes.len()
    }
}

impl Default for ApiGateway {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_upstream(name: &str, url: &str) -> Upstream {
        Upstream {
            name: name.to_string(),
            base_url: url.to_string(),
            weight: 1,
            healthy: true,
        }
    }

    fn make_route(id: &str, method: HttpMethod, pattern: &str, upstream: &str) -> Route {
        Route {
            id: id.to_string(),
            method,
            pattern: PathPattern::parse(pattern).unwrap(),
            upstream: upstream.to_string(),
            timeout_ms: 5000,
            request_headers: vec![],
            response_headers: vec![],
            path_rewrite: None,
            priority: 100,
            enabled: true,
        }
    }

    fn make_request(method: HttpMethod, path: &str) -> GatewayRequest {
        GatewayRequest {
            method,
            path: path.to_string(),
            headers: HashMap::new(),
            query_params: HashMap::new(),
            body: None,
        }
    }

    // ── Pattern parsing tests ──

    #[test]
    fn parse_literal_pattern() {
        let p = PathPattern::parse("/api/users").unwrap();
        assert_eq!(p.segments.len(), 2);
        assert_eq!(p.segments[0], PatternSegment::Literal("api".into()));
        assert_eq!(p.segments[1], PatternSegment::Literal("users".into()));
    }

    #[test]
    fn parse_param_pattern() {
        let p = PathPattern::parse("/users/:id/posts/:post_id").unwrap();
        assert_eq!(p.segments.len(), 4);
        assert_eq!(p.segments[1], PatternSegment::Param("id".into()));
        assert_eq!(p.segments[3], PatternSegment::Param("post_id".into()));
    }

    #[test]
    fn parse_wildcard_and_glob() {
        let p = PathPattern::parse("/api/*/docs/**").unwrap();
        assert_eq!(p.segments[1], PatternSegment::Wildcard);
        assert_eq!(p.segments[3], PatternSegment::Glob);
    }

    #[test]
    fn parse_empty_pattern_error() {
        assert!(PathPattern::parse("").is_err());
    }

    #[test]
    fn parse_empty_param_name_error() {
        assert!(PathPattern::parse("/users/:").is_err());
    }

    #[test]
    fn parse_root_pattern() {
        let p = PathPattern::parse("/").unwrap();
        assert!(p.segments.is_empty());
        assert!(p.match_path("/").is_some());
    }

    // ── Pattern matching tests ──

    #[test]
    fn match_literal_path() {
        let p = PathPattern::parse("/api/users").unwrap();
        assert!(p.match_path("/api/users").is_some());
        assert!(p.match_path("/api/posts").is_none());
        assert!(p.match_path("/api/users/extra").is_none());
    }

    #[test]
    fn match_params_extracted() {
        let p = PathPattern::parse("/users/:id").unwrap();
        let params = p.match_path("/users/42").unwrap();
        assert_eq!(params.get("id").unwrap(), "42");
    }

    #[test]
    fn match_multiple_params() {
        let p = PathPattern::parse("/orgs/:org/repos/:repo").unwrap();
        let params = p.match_path("/orgs/acme/repos/widget").unwrap();
        assert_eq!(params.get("org").unwrap(), "acme");
        assert_eq!(params.get("repo").unwrap(), "widget");
    }

    #[test]
    fn match_wildcard() {
        let p = PathPattern::parse("/api/*/info").unwrap();
        assert!(p.match_path("/api/anything/info").is_some());
        assert!(p.match_path("/api/info").is_none());
    }

    #[test]
    fn match_glob_catches_rest() {
        let p = PathPattern::parse("/static/**").unwrap();
        assert!(p.match_path("/static/css/main.css").is_some());
        assert!(p.match_path("/static").is_none());
        assert!(p.match_path("/static/a").is_some());
    }

    #[test]
    fn no_match_on_short_path() {
        let p = PathPattern::parse("/a/b/c").unwrap();
        assert!(p.match_path("/a/b").is_none());
    }

    // ── Header action tests ──

    #[test]
    fn header_add() {
        let mut headers = HashMap::new();
        apply_header_actions(&mut headers, &[HeaderAction::Add {
            name: "X-Req-Id".into(),
            value: "abc".into(),
        }]);
        assert_eq!(headers.get("X-Req-Id").unwrap(), &vec!["abc".to_string()]);
    }

    #[test]
    fn header_add_appends() {
        let mut headers = HashMap::new();
        headers.insert("X-Tag".into(), vec!["a".into()]);
        apply_header_actions(&mut headers, &[HeaderAction::Add {
            name: "X-Tag".into(),
            value: "b".into(),
        }]);
        assert_eq!(headers.get("X-Tag").unwrap(), &vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn header_set_replaces() {
        let mut headers = HashMap::new();
        headers.insert("X-Tag".into(), vec!["a".into(), "b".into()]);
        apply_header_actions(&mut headers, &[HeaderAction::Set {
            name: "X-Tag".into(),
            value: "c".into(),
        }]);
        assert_eq!(headers.get("X-Tag").unwrap(), &vec!["c".to_string()]);
    }

    #[test]
    fn header_remove() {
        let mut headers = HashMap::new();
        headers.insert("X-Secret".into(), vec!["hide".into()]);
        apply_header_actions(&mut headers, &[HeaderAction::Remove {
            name: "X-Secret".into(),
        }]);
        assert!(!headers.contains_key("X-Secret"));
    }

    #[test]
    fn header_rename() {
        let mut headers = HashMap::new();
        headers.insert("Old-Name".into(), vec!["val".into()]);
        apply_header_actions(&mut headers, &[HeaderAction::Rename {
            from: "Old-Name".into(),
            to: "New-Name".into(),
        }]);
        assert!(!headers.contains_key("Old-Name"));
        assert_eq!(headers.get("New-Name").unwrap(), &vec!["val".to_string()]);
    }

    // ── Path rewrite tests ──

    #[test]
    fn rewrite_strip_prefix() {
        let rw = PathRewrite {
            strip_prefix: Some("/api/v1".into()),
            add_prefix: None,
            replace: None,
        };
        assert_eq!(rw.apply("/api/v1/users"), "/users");
    }

    #[test]
    fn rewrite_add_prefix() {
        let rw = PathRewrite {
            strip_prefix: None,
            add_prefix: Some("/backend".into()),
            replace: None,
        };
        assert_eq!(rw.apply("/users"), "/backend/users");
    }

    #[test]
    fn rewrite_strip_and_add() {
        let rw = PathRewrite {
            strip_prefix: Some("/v1".into()),
            add_prefix: Some("/v2".into()),
            replace: None,
        };
        assert_eq!(rw.apply("/v1/items"), "/v2/items");
    }

    #[test]
    fn rewrite_replace_entire() {
        let rw = PathRewrite {
            strip_prefix: Some("/should-not-matter".into()),
            add_prefix: Some("/also-not".into()),
            replace: Some("/fixed/path".into()),
        };
        assert_eq!(rw.apply("/anything"), "/fixed/path");
    }

    // ── Gateway tests ──

    #[test]
    fn gateway_add_and_resolve() {
        let mut gw = ApiGateway::new();
        gw.add_upstream(make_upstream("users-svc", "http://users:8080"));
        gw.add_route(make_route("r1", HttpMethod::Get, "/api/users", "users-svc"))
            .unwrap();

        let req = make_request(HttpMethod::Get, "/api/users");
        let m = gw.resolve(&req).unwrap();
        assert_eq!(m.route_id, "r1");
        assert_eq!(m.upstream_url, "http://users:8080/api/users");
    }

    #[test]
    fn gateway_no_match() {
        let gw = ApiGateway::new();
        let req = make_request(HttpMethod::Get, "/nope");
        assert!(matches!(gw.resolve(&req), Err(GatewayError::NoRouteMatch { .. })));
    }

    #[test]
    fn gateway_duplicate_route() {
        let mut gw = ApiGateway::new();
        gw.add_upstream(make_upstream("svc", "http://svc:80"));
        gw.add_route(make_route("r1", HttpMethod::Get, "/a", "svc")).unwrap();
        let err = gw.add_route(make_route("r1", HttpMethod::Post, "/b", "svc")).unwrap_err();
        assert!(matches!(err, GatewayError::DuplicateRoute(_)));
    }

    #[test]
    fn gateway_method_mismatch() {
        let mut gw = ApiGateway::new();
        gw.add_upstream(make_upstream("svc", "http://svc:80"));
        gw.add_route(make_route("r1", HttpMethod::Post, "/data", "svc")).unwrap();

        let req = make_request(HttpMethod::Get, "/data");
        assert!(gw.resolve(&req).is_err());
    }

    #[test]
    fn gateway_any_method_matches() {
        let mut gw = ApiGateway::new();
        gw.add_upstream(make_upstream("svc", "http://svc:80"));
        gw.add_route(make_route("r1", HttpMethod::Any, "/health", "svc")).unwrap();

        for method in [HttpMethod::Get, HttpMethod::Post, HttpMethod::Delete] {
            let req = make_request(method, "/health");
            assert!(gw.resolve(&req).is_ok());
        }
    }

    #[test]
    fn gateway_with_path_rewrite() {
        let mut gw = ApiGateway::new();
        gw.add_upstream(make_upstream("svc", "http://backend:80"));
        let mut route = make_route("r1", HttpMethod::Get, "/api/v1/**", "svc");
        route.path_rewrite = Some(PathRewrite {
            strip_prefix: Some("/api/v1".into()),
            add_prefix: Some("/internal".into()),
            replace: None,
        });
        gw.add_route(route).unwrap();

        let req = make_request(HttpMethod::Get, "/api/v1/items");
        let m = gw.resolve(&req).unwrap();
        assert_eq!(m.rewritten_path, "/internal/items");
        assert_eq!(m.upstream_url, "http://backend:80/internal/items");
    }

    #[test]
    fn gateway_unhealthy_upstream_skipped() {
        let mut gw = ApiGateway::new();
        let mut up = make_upstream("svc", "http://svc:80");
        up.healthy = false;
        gw.add_upstream(up);
        gw.add_route(make_route("r1", HttpMethod::Get, "/test", "svc")).unwrap();

        let req = make_request(HttpMethod::Get, "/test");
        assert!(gw.resolve(&req).is_err());
    }

    #[test]
    fn gateway_set_upstream_health() {
        let mut gw = ApiGateway::new();
        gw.add_upstream(make_upstream("svc", "http://svc:80"));
        assert!(gw.set_upstream_health("svc", false));
        assert!(!gw.set_upstream_health("nonexistent", false));
    }

    #[test]
    fn gateway_remove_route() {
        let mut gw = ApiGateway::new();
        gw.add_upstream(make_upstream("svc", "http://svc:80"));
        gw.add_route(make_route("r1", HttpMethod::Get, "/a", "svc")).unwrap();
        assert!(gw.remove_route("r1"));
        assert!(!gw.remove_route("r1"));
        assert_eq!(gw.route_count(), 0);
    }

    #[test]
    fn gateway_disable_route() {
        let mut gw = ApiGateway::new();
        gw.add_upstream(make_upstream("svc", "http://svc:80"));
        gw.add_route(make_route("r1", HttpMethod::Get, "/a", "svc")).unwrap();
        gw.set_route_enabled("r1", false);

        let req = make_request(HttpMethod::Get, "/a");
        assert!(gw.resolve(&req).is_err());
    }

    #[test]
    fn gateway_priority_ordering() {
        let mut gw = ApiGateway::new();
        gw.add_upstream(make_upstream("svc-a", "http://a:80"));
        gw.add_upstream(make_upstream("svc-b", "http://b:80"));

        let mut r1 = make_route("low", HttpMethod::Get, "/data", "svc-a");
        r1.priority = 200;
        let mut r2 = make_route("high", HttpMethod::Get, "/data", "svc-b");
        r2.priority = 10;

        gw.add_route(r1).unwrap();
        gw.add_route(r2).unwrap();

        let req = make_request(HttpMethod::Get, "/data");
        let m = gw.resolve(&req).unwrap();
        assert_eq!(m.route_id, "high");
    }

    #[test]
    fn gateway_global_request_headers() {
        let mut gw = ApiGateway::new();
        gw.add_upstream(make_upstream("svc", "http://svc:80"));
        gw.add_route(make_route("r1", HttpMethod::Get, "/a", "svc")).unwrap();
        gw.add_global_request_header(HeaderAction::Set {
            name: "X-Gateway".into(),
            value: "joule".into(),
        });

        let req = make_request(HttpMethod::Get, "/a");
        let m = gw.resolve(&req).unwrap();
        assert_eq!(
            m.request_headers.get("X-Gateway").unwrap(),
            &vec!["joule".to_string()]
        );
    }

    #[test]
    fn gateway_per_route_request_headers() {
        let mut gw = ApiGateway::new();
        gw.add_upstream(make_upstream("svc", "http://svc:80"));
        let mut route = make_route("r1", HttpMethod::Get, "/a", "svc");
        route.request_headers.push(HeaderAction::Set {
            name: "X-Route".into(),
            value: "r1".into(),
        });
        gw.add_route(route).unwrap();

        let req = make_request(HttpMethod::Get, "/a");
        let m = gw.resolve(&req).unwrap();
        assert_eq!(
            m.request_headers.get("X-Route").unwrap(),
            &vec!["r1".to_string()]
        );
    }

    #[test]
    fn gateway_transform_response() {
        let mut gw = ApiGateway::new();
        gw.add_upstream(make_upstream("svc", "http://svc:80"));
        let mut route = make_route("r1", HttpMethod::Get, "/a", "svc");
        route.response_headers.push(HeaderAction::Set {
            name: "X-Powered-By".into(),
            value: "joule-web".into(),
        });
        gw.add_route(route).unwrap();
        gw.add_global_response_header(HeaderAction::Remove {
            name: "Server".into(),
        });

        let mut resp_headers = HashMap::new();
        resp_headers.insert("Server".into(), vec!["nginx".into()]);
        let resp = GatewayResponse {
            status: 200,
            headers: resp_headers,
            body: None,
        };

        let transformed = gw.transform_response("r1", resp);
        assert!(!transformed.headers.contains_key("Server"));
        assert_eq!(
            transformed.headers.get("X-Powered-By").unwrap(),
            &vec!["joule-web".to_string()]
        );
    }

    #[test]
    fn gateway_default_timeout_used() {
        let mut gw = ApiGateway::new();
        gw.set_default_timeout_ms(10_000);
        gw.add_upstream(make_upstream("svc", "http://svc:80"));
        let mut route = make_route("r1", HttpMethod::Get, "/a", "svc");
        route.timeout_ms = 0; // use default
        gw.add_route(route).unwrap();

        let req = make_request(HttpMethod::Get, "/a");
        let m = gw.resolve(&req).unwrap();
        assert_eq!(m.timeout_ms, 10_000);
    }

    #[test]
    fn gateway_route_ids_and_upstream_names() {
        let mut gw = ApiGateway::new();
        gw.add_upstream(make_upstream("b-svc", "http://b:80"));
        gw.add_upstream(make_upstream("a-svc", "http://a:80"));
        gw.add_route(make_route("r1", HttpMethod::Get, "/a", "a-svc")).unwrap();
        gw.add_route(make_route("r2", HttpMethod::Get, "/b", "b-svc")).unwrap();

        assert_eq!(gw.route_count(), 2);
        assert_eq!(gw.upstream_names(), vec!["a-svc", "b-svc"]);
    }

    #[test]
    fn gateway_upstream_not_found() {
        let mut gw = ApiGateway::new();
        gw.add_route(make_route("r1", HttpMethod::Get, "/a", "missing")).unwrap();
        let req = make_request(HttpMethod::Get, "/a");
        assert!(matches!(gw.resolve(&req), Err(GatewayError::UpstreamNotFound(_))));
    }

    #[test]
    fn gateway_remove_upstream() {
        let mut gw = ApiGateway::new();
        gw.add_upstream(make_upstream("svc", "http://svc:80"));
        assert!(gw.remove_upstream("svc"));
        assert!(!gw.remove_upstream("svc"));
    }

    #[test]
    fn gateway_params_in_upstream_url() {
        let mut gw = ApiGateway::new();
        gw.add_upstream(make_upstream("svc", "http://svc:80"));
        gw.add_route(make_route("r1", HttpMethod::Get, "/users/:id", "svc"))
            .unwrap();

        let req = make_request(HttpMethod::Get, "/users/99");
        let m = gw.resolve(&req).unwrap();
        assert_eq!(m.path_params.get("id").unwrap(), "99");
        assert_eq!(m.upstream_url, "http://svc:80/users/99");
    }

    #[test]
    fn http_method_display() {
        assert_eq!(HttpMethod::Get.to_string(), "GET");
        assert_eq!(HttpMethod::Post.to_string(), "POST");
        assert_eq!(HttpMethod::Any.to_string(), "*");
    }

    #[test]
    fn gateway_error_display() {
        let e = GatewayError::Timeout {
            route_id: "r1".into(),
            timeout_ms: 5000,
        };
        assert!(e.to_string().contains("5000"));
    }
}
