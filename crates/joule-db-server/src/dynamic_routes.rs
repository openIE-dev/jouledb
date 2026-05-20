//! Dynamic API endpoint routing for DEFINE API.
//!
//! Stores user-defined HTTP endpoints and dispatches incoming requests
//! to their stored SQL handlers via the query executor.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Key for routing: (path, HTTP method)
type RouteKey = (String, String);

/// Information about a registered dynamic API endpoint.
#[derive(Debug, Clone)]
pub struct DynamicRoute {
    pub path: String,
    pub method: String,
    pub handler_sql: String,
}

/// Manages user-defined API endpoints registered via DEFINE API.
///
/// Thread-safe: uses `tokio::sync::RwLock` for concurrent read access
/// with exclusive write access during registration.
pub struct DynamicRouteManager {
    routes: RwLock<HashMap<RouteKey, DynamicRoute>>,
}

impl DynamicRouteManager {
    /// Create an empty route manager.
    pub fn new() -> Self {
        Self {
            routes: RwLock::new(HashMap::new()),
        }
    }

    /// Register a dynamic endpoint. Overwrites any existing route with the same path+method.
    pub async fn register(&self, path: String, method: String, handler_sql: String) {
        let key = (path.clone(), method.clone());
        let route = DynamicRoute {
            path,
            method,
            handler_sql,
        };
        self.routes.write().await.insert(key, route);
    }

    /// Look up a route by path and method.
    pub async fn resolve(&self, path: &str, method: &str) -> Option<DynamicRoute> {
        self.routes
            .read()
            .await
            .get(&(path.to_string(), method.to_string()))
            .cloned()
    }

    /// Return the number of registered routes.
    pub async fn len(&self) -> usize {
        self.routes.read().await.len()
    }

    /// Check if there are no registered routes.
    pub async fn is_empty(&self) -> bool {
        self.routes.read().await.is_empty()
    }

    /// List all registered routes.
    pub async fn list(&self) -> Vec<DynamicRoute> {
        self.routes.read().await.values().cloned().collect()
    }
}

/// Create a shared DynamicRouteManager wrapped in an Arc.
pub fn create_route_manager() -> Arc<DynamicRouteManager> {
    Arc::new(DynamicRouteManager::new())
}
