//! Client-side router with pattern matching, nested routes, and navigation history.
//!
//! Replaces React Router / Vue Router with a pure-Rust implementation featuring
//! parameterized routes, wildcards, guards, and a browser-independent history stack.

use std::collections::HashMap;

// ── Route Context ──

/// Context available to route guards and handlers.
#[derive(Debug, Clone)]
pub struct RouteContext {
    pub current_path: String,
    pub params: HashMap<String, String>,
    pub query: HashMap<String, String>,
}

// ── Route ──

/// A single route definition with optional children and guard.
pub struct Route {
    pub path_pattern: String,
    pub name: Option<String>,
    pub handler_id: u64,
    pub children: Vec<Route>,
    pub guard: Option<Box<dyn Fn(&RouteContext) -> bool>>,
}

impl Route {
    /// Create a new route with the given pattern and handler.
    pub fn new(path_pattern: &str, handler_id: u64) -> Self {
        Self {
            path_pattern: path_pattern.to_string(),
            name: None,
            handler_id,
            children: Vec::new(),
            guard: None,
        }
    }

    /// Set a human-readable name for this route.
    pub fn name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }

    /// Add a child route.
    pub fn child(mut self, child: Route) -> Self {
        self.children.push(child);
        self
    }

    /// Set a navigation guard on this route.
    pub fn guard(mut self, f: impl Fn(&RouteContext) -> bool + 'static) -> Self {
        self.guard = Some(Box::new(f));
        self
    }
}

// ── Route Match ──

/// Result of successfully resolving a path against the router.
#[derive(Debug, Clone)]
pub struct RouteMatch {
    pub route_name: Option<String>,
    pub handler_id: u64,
    pub params: HashMap<String, String>,
    pub query: HashMap<String, String>,
    pub path: String,
}

// ── Router ──

/// Client-side router that matches URL paths to handlers.
pub struct Router {
    routes: Vec<Route>,
    not_found_handler: u64,
    base_path: String,
}

impl Router {
    pub fn new() -> Self {
        Self {
            routes: Vec::new(),
            not_found_handler: 0,
            base_path: String::new(),
        }
    }

    /// Set the handler ID used for unmatched paths.
    pub fn not_found(mut self, handler_id: u64) -> Self {
        self.not_found_handler = handler_id;
        self
    }

    /// Set a base path prefix for all routes (e.g. "/app").
    pub fn base_path(mut self, base: &str) -> Self {
        self.base_path = normalize_path(base).trim_end_matches('/').to_string();
        self
    }

    /// Add a route and return a mutable reference to it for chaining.
    pub fn route(&mut self, path: &str, handler_id: u64) -> &mut Route {
        self.routes.push(Route::new(path, handler_id));
        self.routes.last_mut().unwrap_or_else(|| unreachable!())
    }

    /// Mount a set of sub-routes under a common prefix.
    pub fn nested(&mut self, base: &str, routes: Vec<Route>) {
        let base = base.trim_end_matches('/');
        for mut route in routes {
            route.path_pattern = format!("{}{}", base, route.path_pattern);
            self.routes.push(route);
        }
    }

    /// Resolve a path (with optional query string) against registered routes.
    pub fn resolve(&self, raw_path: &str) -> Option<RouteMatch> {
        let (path_part, query_string) = split_query(raw_path);
        let query = parse_query(query_string);

        // Strip base_path prefix if present.
        let path = if !self.base_path.is_empty() {
            let normalized = normalize_path(path_part);
            if let Some(stripped) = normalized.strip_prefix(&self.base_path) {
                if stripped.is_empty() {
                    "/".to_string()
                } else {
                    stripped.to_string()
                }
            } else {
                return None;
            }
        } else {
            normalize_path(path_part)
        };

        self.resolve_routes(&self.routes, &path, &query)
    }

    fn resolve_routes(
        &self,
        routes: &[Route],
        path: &str,
        query: &HashMap<String, String>,
    ) -> Option<RouteMatch> {
        for route in routes {
            if let Some(params) = match_pattern(&route.path_pattern, path) {
                // Check guard.
                if let Some(guard) = &route.guard {
                    let ctx = RouteContext {
                        current_path: path.to_string(),
                        params: params.clone(),
                        query: query.clone(),
                    };
                    if !guard(&ctx) {
                        continue;
                    }
                }

                // Try children first (more specific match).
                if !route.children.is_empty() {
                    // For children, we need to match against the remaining path.
                    let prefix = route_prefix(&route.path_pattern);
                    let remaining = strip_matched_prefix(path, &prefix);
                    if let Some(child_match) =
                        self.resolve_routes(&route.children, &remaining, query)
                    {
                        // Merge parent params into child match.
                        let mut merged_params = params;
                        merged_params.extend(child_match.params);
                        return Some(RouteMatch {
                            route_name: child_match.route_name,
                            handler_id: child_match.handler_id,
                            params: merged_params,
                            query: query.clone(),
                            path: path.to_string(),
                        });
                    }
                }

                return Some(RouteMatch {
                    route_name: route.name.clone(),
                    handler_id: route.handler_id,
                    params,
                    query: query.clone(),
                    path: path.to_string(),
                });
            }
        }
        None
    }
}

impl Default for Router {
    fn default() -> Self {
        Self::new()
    }
}

// ── Pattern Matching Helpers ──

/// Normalize a path: ensure leading slash, strip trailing slash (unless root).
fn normalize_path(path: &str) -> String {
    let p = if path.starts_with('/') {
        path.to_string()
    } else {
        format!("/{path}")
    };
    if p.len() > 1 && p.ends_with('/') {
        p.trim_end_matches('/').to_string()
    } else {
        p
    }
}

/// Split "path?query" into (path, query).
fn split_query(raw: &str) -> (&str, &str) {
    if let Some(idx) = raw.find('?') {
        (&raw[..idx], &raw[idx + 1..])
    } else {
        (raw, "")
    }
}

/// Parse "key=val&key2=val2" into a HashMap.
fn parse_query(qs: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    if qs.is_empty() {
        return map;
    }
    for pair in qs.split('&') {
        if let Some(idx) = pair.find('=') {
            map.insert(pair[..idx].to_string(), pair[idx + 1..].to_string());
        } else if !pair.is_empty() {
            map.insert(pair.to_string(), String::new());
        }
    }
    map
}

/// Match a route pattern against a path, returning extracted params on success.
fn match_pattern(pattern: &str, path: &str) -> Option<HashMap<String, String>> {
    let pattern = normalize_path(pattern);
    let path = normalize_path(path);

    let pat_segments: Vec<&str> = pattern.split('/').filter(|s| !s.is_empty()).collect();
    let path_segments: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();

    let mut params = HashMap::new();
    let mut pi = 0;

    for (i, seg) in pat_segments.iter().enumerate() {
        if let Some(name) = seg.strip_prefix('*') {
            // Wildcard — consume all remaining segments.
            let remaining: Vec<&str> = path_segments[pi..].to_vec();
            let key = if name.is_empty() { "wildcard" } else { name };
            params.insert(key.to_string(), remaining.join("/"));
            return Some(params);
        } else if let Some(name) = seg.strip_prefix(':') {
            // Parameter segment.
            if pi >= path_segments.len() {
                return None;
            }
            params.insert(name.to_string(), path_segments[pi].to_string());
            pi += 1;
        } else {
            // Literal segment.
            if pi >= path_segments.len() || *seg != path_segments[pi] {
                return None;
            }
            pi += 1;
        }

        // If this is the last pattern segment (and not a wildcard), path must be fully consumed.
        if i == pat_segments.len() - 1 && pi != path_segments.len() {
            return None;
        }
    }

    if pat_segments.is_empty() && path_segments.is_empty() {
        return Some(params);
    }

    if pi == path_segments.len() {
        Some(params)
    } else {
        None
    }
}

/// Get the literal prefix of a pattern (before any : or * segment).
fn route_prefix(pattern: &str) -> String {
    let segments: Vec<&str> = pattern.split('/').filter(|s| !s.is_empty()).collect();
    let mut prefix_parts = Vec::new();
    for seg in &segments {
        if seg.starts_with(':') || seg.starts_with('*') {
            break;
        }
        prefix_parts.push(*seg);
    }
    if prefix_parts.is_empty() {
        "/".to_string()
    } else {
        format!("/{}", prefix_parts.join("/"))
    }
}

/// Strip the matched prefix from a path for child route matching.
fn strip_matched_prefix(path: &str, prefix: &str) -> String {
    let normalized = normalize_path(path);
    let prefix_norm = normalize_path(prefix);
    if let Some(rest) = normalized.strip_prefix(prefix_norm.as_str()) {
        if rest.is_empty() {
            "/".to_string()
        } else {
            rest.to_string()
        }
    } else {
        normalized
    }
}

// ── Route History ──

/// Browser-independent navigation history stack.
#[derive(Debug, Clone)]
pub struct RouteHistory {
    entries: Vec<String>,
    current_index: usize,
}

impl RouteHistory {
    pub fn new(initial_path: &str) -> Self {
        Self {
            entries: vec![initial_path.to_string()],
            current_index: 0,
        }
    }

    /// Push a new path, discarding any forward history.
    pub fn push(&mut self, path: &str) {
        // Truncate forward entries.
        self.entries.truncate(self.current_index + 1);
        self.entries.push(path.to_string());
        self.current_index = self.entries.len() - 1;
    }

    /// Go back one entry; returns the new current path.
    pub fn back(&mut self) -> Option<String> {
        if self.can_go_back() {
            self.current_index -= 1;
            Some(self.entries[self.current_index].clone())
        } else {
            None
        }
    }

    /// Go forward one entry; returns the new current path.
    pub fn forward(&mut self) -> Option<String> {
        if self.can_go_forward() {
            self.current_index += 1;
            Some(self.entries[self.current_index].clone())
        } else {
            None
        }
    }

    pub fn current(&self) -> &str {
        &self.entries[self.current_index]
    }

    pub fn can_go_back(&self) -> bool {
        self.current_index > 0
    }

    pub fn can_go_forward(&self) -> bool {
        self.current_index + 1 < self.entries.len()
    }
}

// ── Navigation Guard ──

/// Trait for navigation guards that can allow or block transitions.
pub trait NavigationGuard {
    fn can_navigate(&self, from: &str, to: &str) -> bool;
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_match() {
        let mut router = Router::new();
        router.route("/users", 1);
        let m = router.resolve("/users").unwrap();
        assert_eq!(m.handler_id, 1);
        assert!(m.params.is_empty());
    }

    #[test]
    fn test_param_extraction() {
        let mut router = Router::new();
        router.route("/users/:id", 2);
        let m = router.resolve("/users/42").unwrap();
        assert_eq!(m.handler_id, 2);
        assert_eq!(m.params.get("id").unwrap(), "42");
    }

    #[test]
    fn test_multiple_params() {
        let mut router = Router::new();
        router.route("/users/:user_id/posts/:post_id", 3);
        let m = router.resolve("/users/7/posts/99").unwrap();
        assert_eq!(m.params.get("user_id").unwrap(), "7");
        assert_eq!(m.params.get("post_id").unwrap(), "99");
    }

    #[test]
    fn test_wildcard() {
        let mut router = Router::new();
        router.route("/files/*path", 4);
        let m = router.resolve("/files/a/b/c").unwrap();
        assert_eq!(m.params.get("path").unwrap(), "a/b/c");
    }

    #[test]
    fn test_no_match_returns_none() {
        let mut router = Router::new();
        router.route("/users", 1);
        assert!(router.resolve("/posts").is_none());
    }

    #[test]
    fn test_trailing_slash_normalization() {
        let mut router = Router::new();
        router.route("/users", 1);
        let m = router.resolve("/users/").unwrap();
        assert_eq!(m.handler_id, 1);
    }

    #[test]
    fn test_query_parsing() {
        let mut router = Router::new();
        router.route("/search", 5);
        let m = router.resolve("/search?q=rust&page=2").unwrap();
        assert_eq!(m.query.get("q").unwrap(), "rust");
        assert_eq!(m.query.get("page").unwrap(), "2");
    }

    #[test]
    fn test_nested_routes() {
        let mut router = Router::new();
        router.nested(
            "/api",
            vec![
                Route::new("/users", 10),
                Route::new("/posts", 11),
            ],
        );
        let m = router.resolve("/api/users").unwrap();
        assert_eq!(m.handler_id, 10);
        let m2 = router.resolve("/api/posts").unwrap();
        assert_eq!(m2.handler_id, 11);
    }

    #[test]
    fn test_base_path_prepend() {
        let mut router = Router::new().base_path("/app");
        router.route("/dashboard", 20);
        let m = router.resolve("/app/dashboard").unwrap();
        assert_eq!(m.handler_id, 20);
        // Without base path prefix, should not match.
        assert!(router.resolve("/dashboard").is_none());
    }

    #[test]
    fn test_history_push_back_forward() {
        let mut history = RouteHistory::new("/");
        assert_eq!(history.current(), "/");
        assert!(!history.can_go_back());

        history.push("/users");
        history.push("/users/42");
        assert_eq!(history.current(), "/users/42");
        assert!(history.can_go_back());

        let prev = history.back().unwrap();
        assert_eq!(prev, "/users");
        assert!(history.can_go_forward());

        let fwd = history.forward().unwrap();
        assert_eq!(fwd, "/users/42");
        assert!(!history.can_go_forward());
    }

    #[test]
    fn test_history_push_truncates_forward() {
        let mut history = RouteHistory::new("/");
        history.push("/a");
        history.push("/b");
        history.back();
        // Now at "/a", forward has "/b".
        history.push("/c");
        // Forward history ("/b") should be gone.
        assert!(!history.can_go_forward());
        assert_eq!(history.current(), "/c");
    }

    #[test]
    fn test_route_guard_blocks() {
        let mut router = Router::new();
        let route = Route::new("/admin", 30).guard(|_ctx| false);
        router.routes.push(route);
        assert!(router.resolve("/admin").is_none());
    }

    #[test]
    fn test_root_path() {
        let mut router = Router::new();
        router.route("/", 100);
        let m = router.resolve("/").unwrap();
        assert_eq!(m.handler_id, 100);
    }
}
