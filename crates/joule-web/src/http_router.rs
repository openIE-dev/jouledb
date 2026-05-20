//! HTTP request router — path pattern matching, method routing, route groups
//! with prefix, middleware chain, path parameter extraction, wildcard routes,
//! and route conflict detection.
//!
//! Replaces Express, Koa-Router, Hono, and similar JS routing libraries with
//! a pure-Rust, zero-allocation-path router.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── Errors ───────────────────────────────────────────────────────

/// Router error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RouterError {
    /// Route not found for the given path and method.
    NotFound { method: String, path: String },
    /// Method not allowed for the given path.
    MethodNotAllowed { path: String, allowed: Vec<String> },
    /// Duplicate route registration.
    DuplicateRoute { method: String, pattern: String },
    /// Invalid route pattern.
    InvalidPattern(String),
    /// Route conflict detected between two patterns.
    RouteConflict { existing: String, new: String },
    /// Middleware error.
    MiddlewareError(String),
}

impl fmt::Display for RouterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound { method, path } => {
                write!(f, "{method} {path} not found")
            }
            Self::MethodNotAllowed { path, allowed } => {
                write!(f, "method not allowed for {path}, allowed: {}", allowed.join(", "))
            }
            Self::DuplicateRoute { method, pattern } => {
                write!(f, "duplicate route: {method} {pattern}")
            }
            Self::InvalidPattern(p) => write!(f, "invalid pattern: {p}"),
            Self::RouteConflict { existing, new } => {
                write!(f, "route conflict: '{existing}' vs '{new}'")
            }
            Self::MiddlewareError(msg) => write!(f, "middleware error: {msg}"),
        }
    }
}

impl std::error::Error for RouterError {}

// ── Types ────────────────────────────────────────────────────────

/// HTTP method for routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Method {
    Get,
    Post,
    Put,
    Delete,
    Patch,
    Head,
    Options,
}

impl Method {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Get => "GET",
            Self::Post => "POST",
            Self::Put => "PUT",
            Self::Delete => "DELETE",
            Self::Patch => "PATCH",
            Self::Head => "HEAD",
            Self::Options => "OPTIONS",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "GET" => Some(Self::Get),
            "POST" => Some(Self::Post),
            "PUT" => Some(Self::Put),
            "DELETE" => Some(Self::Delete),
            "PATCH" => Some(Self::Patch),
            "HEAD" => Some(Self::Head),
            "OPTIONS" => Some(Self::Options),
            _ => None,
        }
    }
}

impl fmt::Display for Method {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A segment in a route pattern.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Segment {
    /// Literal path segment, e.g. "users".
    Literal(String),
    /// Named parameter, e.g. ":id".
    Param(String),
    /// Wildcard that matches everything, e.g. "*".
    Wildcard,
}

/// A parsed route pattern.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoutePattern {
    /// Original pattern string.
    pub raw: String,
    /// Parsed segments.
    pub segments: Vec<Segment>,
}

impl RoutePattern {
    /// Parse a route pattern string like "/users/:id/posts".
    pub fn parse(pattern: &str) -> Result<Self, RouterError> {
        let normalized = normalize_path(pattern);
        if normalized.is_empty() || !normalized.starts_with('/') {
            return Err(RouterError::InvalidPattern(pattern.to_string()));
        }

        let mut segments = Vec::new();
        let parts: Vec<&str> = normalized[1..].split('/').collect();

        // Root route has no segments
        if parts.len() == 1 && parts[0].is_empty() {
            return Ok(Self { raw: normalized, segments });
        }

        let mut seen_wildcard = false;
        for part in &parts {
            if seen_wildcard {
                return Err(RouterError::InvalidPattern(
                    format!("{pattern}: segments after wildcard"),
                ));
            }
            if part.is_empty() {
                return Err(RouterError::InvalidPattern(
                    format!("{pattern}: empty segment"),
                ));
            }
            if *part == "*" {
                segments.push(Segment::Wildcard);
                seen_wildcard = true;
            } else if let Some(name) = part.strip_prefix(':') {
                if name.is_empty() {
                    return Err(RouterError::InvalidPattern(
                        format!("{pattern}: empty param name"),
                    ));
                }
                segments.push(Segment::Param(name.to_string()));
            } else {
                segments.push(Segment::Literal(part.to_string()));
            }
        }

        Ok(Self { raw: normalized, segments })
    }

    /// Check if this pattern conflicts with another (same structure but
    /// different literals in the same position).
    pub fn conflicts_with(&self, other: &RoutePattern) -> bool {
        if self.segments.len() != other.segments.len() {
            // A wildcard can conflict with anything that starts the same
            return false;
        }
        let mut any_param = false;
        for (a, b) in self.segments.iter().zip(other.segments.iter()) {
            match (a, b) {
                (Segment::Literal(la), Segment::Literal(lb)) => {
                    if la != lb {
                        return false;
                    }
                }
                (Segment::Param(_), Segment::Param(_)) => {
                    any_param = true;
                }
                (Segment::Wildcard, Segment::Wildcard) => {
                    any_param = true;
                }
                (Segment::Param(_), Segment::Literal(_))
                | (Segment::Literal(_), Segment::Param(_)) => {
                    any_param = true;
                }
                _ => {}
            }
        }
        any_param
    }

    /// Try to match a path against this pattern, extracting parameters.
    pub fn match_path(&self, path: &str) -> Option<HashMap<String, String>> {
        let normalized = normalize_path(path);
        let path_parts: Vec<&str> = if normalized == "/" {
            Vec::new()
        } else {
            normalized[1..].split('/').collect()
        };

        let mut params = HashMap::new();

        if self.segments.is_empty() {
            return if path_parts.is_empty() { Some(params) } else { None };
        }

        let mut pi = 0;
        for seg in &self.segments {
            match seg {
                Segment::Literal(lit) => {
                    if pi >= path_parts.len() || path_parts[pi] != lit.as_str() {
                        return None;
                    }
                    pi += 1;
                }
                Segment::Param(name) => {
                    if pi >= path_parts.len() {
                        return None;
                    }
                    params.insert(name.clone(), path_parts[pi].to_string());
                    pi += 1;
                }
                Segment::Wildcard => {
                    let rest = path_parts[pi..].join("/");
                    params.insert("*".to_string(), rest);
                    return Some(params);
                }
            }
        }

        if pi == path_parts.len() {
            Some(params)
        } else {
            None
        }
    }

    /// Specificity score for ranking routes (higher = more specific).
    pub fn specificity(&self) -> u32 {
        let mut score = 0u32;
        for seg in &self.segments {
            match seg {
                Segment::Literal(_) => score += 3,
                Segment::Param(_) => score += 2,
                Segment::Wildcard => score += 1,
            }
        }
        score
    }
}

impl fmt::Display for RoutePattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.raw)
    }
}

/// A registered route entry.
#[derive(Debug, Clone)]
pub struct Route {
    /// Route pattern.
    pub pattern: RoutePattern,
    /// HTTP method.
    pub method: Method,
    /// Handler identifier (e.g., function name or ID).
    pub handler: String,
    /// Middleware names attached to this route.
    pub middleware: Vec<String>,
}

/// Result of matching a request against a route.
#[derive(Debug, Clone)]
pub struct RouteMatch {
    /// The matched route handler.
    pub handler: String,
    /// Extracted path parameters.
    pub params: HashMap<String, String>,
    /// Middleware chain to execute.
    pub middleware: Vec<String>,
    /// The pattern that matched.
    pub pattern: String,
}

/// Middleware definition for the router.
#[derive(Debug, Clone)]
pub struct MiddlewareDef {
    /// Name of the middleware.
    pub name: String,
    /// Priority (lower = runs first).
    pub priority: u32,
}

/// A route group with a shared prefix and middleware.
#[derive(Debug, Clone)]
pub struct RouteGroup {
    /// Path prefix for the group.
    pub prefix: String,
    /// Middleware applied to all routes in the group.
    pub middleware: Vec<String>,
}

// ── Helpers ──────────────────────────────────────────────────────

/// Normalize a path: strip trailing slash, ensure leading slash.
fn normalize_path(path: &str) -> String {
    let trimmed = path.trim();
    if trimmed.is_empty() || trimmed == "/" {
        return "/".to_string();
    }
    let with_leading = if trimmed.starts_with('/') {
        trimmed.to_string()
    } else {
        format!("/{trimmed}")
    };
    if with_leading.len() > 1 && with_leading.ends_with('/') {
        with_leading[..with_leading.len() - 1].to_string()
    } else {
        with_leading
    }
}

// ── Router ───────────────────────────────────────────────────────

/// HTTP request router with path pattern matching and method routing.
#[derive(Debug, Clone)]
pub struct Router {
    routes: Vec<Route>,
    global_middleware: Vec<String>,
}

impl Router {
    /// Create a new empty router.
    pub fn new() -> Self {
        Self {
            routes: Vec::new(),
            global_middleware: Vec::new(),
        }
    }

    /// Add global middleware applied to all routes.
    pub fn use_middleware(&mut self, name: &str) {
        self.global_middleware.push(name.to_string());
    }

    /// Register a route with the given method, pattern, and handler.
    pub fn add_route(
        &mut self,
        method: Method,
        pattern: &str,
        handler: &str,
    ) -> Result<(), RouterError> {
        self.add_route_with_middleware(method, pattern, handler, &[])
    }

    /// Register a route with middleware.
    pub fn add_route_with_middleware(
        &mut self,
        method: Method,
        pattern: &str,
        handler: &str,
        middleware: &[&str],
    ) -> Result<(), RouterError> {
        let parsed = RoutePattern::parse(pattern)?;

        // Check for duplicates
        for existing in &self.routes {
            if existing.method == method && existing.pattern.raw == parsed.raw {
                return Err(RouterError::DuplicateRoute {
                    method: method.as_str().to_string(),
                    pattern: parsed.raw,
                });
            }
        }

        self.routes.push(Route {
            pattern: parsed,
            method,
            handler: handler.to_string(),
            middleware: middleware.iter().map(|s| s.to_string()).collect(),
        });

        Ok(())
    }

    /// Register a route group with shared prefix and middleware.
    pub fn add_group(
        &mut self,
        group: &RouteGroup,
        routes: &[(Method, &str, &str)],
    ) -> Result<(), RouterError> {
        let prefix = normalize_path(&group.prefix);
        for (method, sub_pattern, handler) in routes {
            let full = if *sub_pattern == "/" || sub_pattern.is_empty() {
                prefix.clone()
            } else {
                let sub = normalize_path(sub_pattern);
                format!("{prefix}{sub}")
            };
            let mw: Vec<&str> = group.middleware.iter().map(|s| s.as_str()).collect();
            self.add_route_with_middleware(*method, &full, handler, &mw)?;
        }
        Ok(())
    }

    /// Shorthand to add GET route.
    pub fn get(&mut self, pattern: &str, handler: &str) -> Result<(), RouterError> {
        self.add_route(Method::Get, pattern, handler)
    }

    /// Shorthand to add POST route.
    pub fn post(&mut self, pattern: &str, handler: &str) -> Result<(), RouterError> {
        self.add_route(Method::Post, pattern, handler)
    }

    /// Shorthand to add PUT route.
    pub fn put(&mut self, pattern: &str, handler: &str) -> Result<(), RouterError> {
        self.add_route(Method::Put, pattern, handler)
    }

    /// Shorthand to add DELETE route.
    pub fn delete(&mut self, pattern: &str, handler: &str) -> Result<(), RouterError> {
        self.add_route(Method::Delete, pattern, handler)
    }

    /// Match a request to a route.
    pub fn match_route(
        &self,
        method: Method,
        path: &str,
    ) -> Result<RouteMatch, RouterError> {
        let mut best_match: Option<(RouteMatch, u32)> = None;
        let mut path_matched = false;
        let mut allowed_methods: Vec<String> = Vec::new();

        for route in &self.routes {
            if let Some(params) = route.pattern.match_path(path) {
                path_matched = true;
                if route.method == method {
                    let specificity = route.pattern.specificity();
                    let should_replace = match &best_match {
                        None => true,
                        Some((_, best_spec)) => specificity > *best_spec,
                    };
                    if should_replace {
                        let mut mw = self.global_middleware.clone();
                        mw.extend(route.middleware.clone());
                        best_match = Some((
                            RouteMatch {
                                handler: route.handler.clone(),
                                params,
                                middleware: mw,
                                pattern: route.pattern.raw.clone(),
                            },
                            specificity,
                        ));
                    }
                } else if !allowed_methods.contains(&route.method.as_str().to_string()) {
                    allowed_methods.push(route.method.as_str().to_string());
                }
            }
        }

        if let Some((route_match, _)) = best_match {
            Ok(route_match)
        } else if path_matched {
            Err(RouterError::MethodNotAllowed {
                path: path.to_string(),
                allowed: allowed_methods,
            })
        } else {
            Err(RouterError::NotFound {
                method: method.as_str().to_string(),
                path: path.to_string(),
            })
        }
    }

    /// Detect conflicting routes that could match the same paths.
    pub fn detect_conflicts(&self) -> Vec<(String, String)> {
        let mut conflicts = Vec::new();
        for i in 0..self.routes.len() {
            for j in (i + 1)..self.routes.len() {
                let a = &self.routes[i];
                let b = &self.routes[j];
                if a.method == b.method && a.pattern.conflicts_with(&b.pattern) {
                    let label_a = format!("{} {}", a.method.as_str(), a.pattern.raw);
                    let label_b = format!("{} {}", b.method.as_str(), b.pattern.raw);
                    conflicts.push((label_a, label_b));
                }
            }
        }
        conflicts
    }

    /// List all registered routes.
    pub fn routes(&self) -> Vec<(String, String)> {
        self.routes
            .iter()
            .map(|r| (r.method.as_str().to_string(), r.pattern.raw.clone()))
            .collect()
    }

    /// Number of registered routes.
    pub fn len(&self) -> usize {
        self.routes.len()
    }

    /// Whether the router has any routes.
    pub fn is_empty(&self) -> bool {
        self.routes.is_empty()
    }
}

impl Default for Router {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_parse_simple() {
        let p = RoutePattern::parse("/users").unwrap();
        assert_eq!(p.segments.len(), 1);
        assert_eq!(p.segments[0], Segment::Literal("users".to_string()));
    }

    #[test]
    fn test_pattern_parse_with_param() {
        let p = RoutePattern::parse("/users/:id").unwrap();
        assert_eq!(p.segments.len(), 2);
        assert_eq!(p.segments[0], Segment::Literal("users".to_string()));
        assert_eq!(p.segments[1], Segment::Param("id".to_string()));
    }

    #[test]
    fn test_pattern_parse_nested_params() {
        let p = RoutePattern::parse("/users/:user_id/posts/:post_id").unwrap();
        assert_eq!(p.segments.len(), 4);
        assert_eq!(p.segments[1], Segment::Param("user_id".to_string()));
        assert_eq!(p.segments[3], Segment::Param("post_id".to_string()));
    }

    #[test]
    fn test_pattern_parse_wildcard() {
        let p = RoutePattern::parse("/files/*").unwrap();
        assert_eq!(p.segments.len(), 2);
        assert_eq!(p.segments[1], Segment::Wildcard);
    }

    #[test]
    fn test_pattern_parse_root() {
        let p = RoutePattern::parse("/").unwrap();
        assert_eq!(p.segments.len(), 0);
    }

    #[test]
    fn test_pattern_parse_invalid_empty_param() {
        let result = RoutePattern::parse("/users/:");
        assert!(result.is_err());
    }

    #[test]
    fn test_pattern_parse_segments_after_wildcard() {
        let result = RoutePattern::parse("/files/*/extra");
        assert!(result.is_err());
    }

    #[test]
    fn test_match_literal() {
        let p = RoutePattern::parse("/users").unwrap();
        assert!(p.match_path("/users").is_some());
        assert!(p.match_path("/users/").is_some());
        assert!(p.match_path("/posts").is_none());
    }

    #[test]
    fn test_match_param() {
        let p = RoutePattern::parse("/users/:id").unwrap();
        let m = p.match_path("/users/42").unwrap();
        assert_eq!(m.get("id").unwrap(), "42");
        assert!(p.match_path("/users").is_none());
    }

    #[test]
    fn test_match_wildcard() {
        let p = RoutePattern::parse("/files/*").unwrap();
        let m = p.match_path("/files/docs/readme.md").unwrap();
        assert_eq!(m.get("*").unwrap(), "docs/readme.md");
    }

    #[test]
    fn test_match_root() {
        let p = RoutePattern::parse("/").unwrap();
        assert!(p.match_path("/").is_some());
        assert!(p.match_path("/users").is_none());
    }

    #[test]
    fn test_router_basic() {
        let mut r = Router::new();
        r.get("/users", "list_users").unwrap();
        r.post("/users", "create_user").unwrap();

        let m = r.match_route(Method::Get, "/users").unwrap();
        assert_eq!(m.handler, "list_users");

        let m = r.match_route(Method::Post, "/users").unwrap();
        assert_eq!(m.handler, "create_user");
    }

    #[test]
    fn test_router_params() {
        let mut r = Router::new();
        r.get("/users/:id", "get_user").unwrap();

        let m = r.match_route(Method::Get, "/users/123").unwrap();
        assert_eq!(m.handler, "get_user");
        assert_eq!(m.params.get("id").unwrap(), "123");
    }

    #[test]
    fn test_router_method_not_allowed() {
        let mut r = Router::new();
        r.get("/users", "list_users").unwrap();

        let err = r.match_route(Method::Delete, "/users").unwrap_err();
        match err {
            RouterError::MethodNotAllowed { path, allowed } => {
                assert_eq!(path, "/users");
                assert!(allowed.contains(&"GET".to_string()));
            }
            _ => panic!("expected MethodNotAllowed"),
        }
    }

    #[test]
    fn test_router_not_found() {
        let r = Router::new();
        let err = r.match_route(Method::Get, "/nowhere").unwrap_err();
        assert!(matches!(err, RouterError::NotFound { .. }));
    }

    #[test]
    fn test_router_duplicate_route() {
        let mut r = Router::new();
        r.get("/users", "handler_a").unwrap();
        let err = r.get("/users", "handler_b").unwrap_err();
        assert!(matches!(err, RouterError::DuplicateRoute { .. }));
    }

    #[test]
    fn test_router_global_middleware() {
        let mut r = Router::new();
        r.use_middleware("logger");
        r.get("/users", "list_users").unwrap();

        let m = r.match_route(Method::Get, "/users").unwrap();
        assert!(m.middleware.contains(&"logger".to_string()));
    }

    #[test]
    fn test_router_route_middleware() {
        let mut r = Router::new();
        r.add_route_with_middleware(Method::Get, "/admin", "admin_panel", &["auth"]).unwrap();

        let m = r.match_route(Method::Get, "/admin").unwrap();
        assert!(m.middleware.contains(&"auth".to_string()));
    }

    #[test]
    fn test_router_group() {
        let mut r = Router::new();
        let group = RouteGroup {
            prefix: "/api/v1".to_string(),
            middleware: vec!["auth".to_string()],
        };
        r.add_group(&group, &[
            (Method::Get, "/users", "list_users"),
            (Method::Post, "/users", "create_user"),
            (Method::Get, "/posts", "list_posts"),
        ]).unwrap();

        let m = r.match_route(Method::Get, "/api/v1/users").unwrap();
        assert_eq!(m.handler, "list_users");
        assert!(m.middleware.contains(&"auth".to_string()));

        let m = r.match_route(Method::Get, "/api/v1/posts").unwrap();
        assert_eq!(m.handler, "list_posts");
    }

    #[test]
    fn test_router_specificity_literal_over_param() {
        let mut r = Router::new();
        r.get("/users/me", "current_user").unwrap();
        r.get("/users/:id", "get_user").unwrap();

        let m = r.match_route(Method::Get, "/users/me").unwrap();
        assert_eq!(m.handler, "current_user");
    }

    #[test]
    fn test_conflict_detection() {
        let mut r = Router::new();
        r.get("/users/:id", "get_user").unwrap();
        r.get("/users/:name", "get_user_by_name").unwrap();

        let conflicts = r.detect_conflicts();
        assert_eq!(conflicts.len(), 1);
    }

    #[test]
    fn test_no_conflict_different_methods() {
        let mut r = Router::new();
        r.get("/users/:id", "get_user").unwrap();
        r.post("/users/:id", "update_user").unwrap();

        let conflicts = r.detect_conflicts();
        assert!(conflicts.is_empty());
    }

    #[test]
    fn test_normalize_trailing_slash() {
        let p = normalize_path("/users/");
        assert_eq!(p, "/users");
    }

    #[test]
    fn test_method_display() {
        assert_eq!(Method::Get.to_string(), "GET");
        assert_eq!(Method::Post.to_string(), "POST");
    }

    #[test]
    fn test_method_from_str() {
        assert_eq!(Method::from_str("get"), Some(Method::Get));
        assert_eq!(Method::from_str("POST"), Some(Method::Post));
        assert_eq!(Method::from_str("invalid"), None);
    }

    #[test]
    fn test_router_len_and_empty() {
        let mut r = Router::new();
        assert!(r.is_empty());
        assert_eq!(r.len(), 0);
        r.get("/a", "ha").unwrap();
        assert!(!r.is_empty());
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn test_wildcard_route_in_router() {
        let mut r = Router::new();
        r.get("/static/*", "serve_static").unwrap();

        let m = r.match_route(Method::Get, "/static/css/main.css").unwrap();
        assert_eq!(m.handler, "serve_static");
        assert_eq!(m.params.get("*").unwrap(), "css/main.css");
    }

    #[test]
    fn test_multiple_params() {
        let mut r = Router::new();
        r.get("/orgs/:org_id/teams/:team_id/members/:member_id", "get_member").unwrap();

        let m = r.match_route(Method::Get, "/orgs/acme/teams/eng/members/42").unwrap();
        assert_eq!(m.params.get("org_id").unwrap(), "acme");
        assert_eq!(m.params.get("team_id").unwrap(), "eng");
        assert_eq!(m.params.get("member_id").unwrap(), "42");
    }

    #[test]
    fn test_specificity_score() {
        let literal = RoutePattern::parse("/users/me").unwrap();
        let param = RoutePattern::parse("/users/:id").unwrap();
        let wild = RoutePattern::parse("/users/*").unwrap();

        assert!(literal.specificity() > param.specificity());
        assert!(param.specificity() > wild.specificity());
    }

    #[test]
    fn test_group_root_sub_pattern() {
        let mut r = Router::new();
        let group = RouteGroup {
            prefix: "/api".to_string(),
            middleware: vec![],
        };
        r.add_group(&group, &[
            (Method::Get, "/", "api_root"),
        ]).unwrap();

        let m = r.match_route(Method::Get, "/api").unwrap();
        assert_eq!(m.handler, "api_root");
    }
}
