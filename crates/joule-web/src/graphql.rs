//! GraphQL client — query builder, cache, and normalization.
//!
//! Replaces Apollo, urql, and graphql-request with a pure Rust implementation.
//! Handles query construction, variable serialization, response caching,
//! entity normalization, and fragment composition. No HTTP — transport is
//! plugged in externally.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Types ───────────────────────────────────────────────────────

/// GraphQL operation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Operation {
    Query,
    Mutation,
    Subscription,
}

/// A GraphQL request ready for transport.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphqlRequest {
    pub operation: Operation,
    pub query: String,
    pub variables: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_name: Option<String>,
}

impl GraphqlRequest {
    /// Serialize this request as a JSON value suitable for HTTP POST.
    pub fn to_json(&self) -> serde_json::Value {
        let mut map = serde_json::Map::new();
        map.insert("query".to_string(), serde_json::Value::String(self.query.clone()));
        if !self.variables.is_null() {
            map.insert("variables".to_string(), self.variables.clone());
        }
        if let Some(ref name) = self.operation_name {
            map.insert(
                "operationName".to_string(),
                serde_json::Value::String(name.clone()),
            );
        }
        serde_json::Value::Object(map)
    }
}

/// A GraphQL response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphqlResponse {
    pub data: Option<serde_json::Value>,
    pub errors: Option<Vec<GraphqlError>>,
}

/// A GraphQL error.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphqlError {
    pub message: String,
    #[serde(default)]
    pub locations: Vec<Location>,
    #[serde(default)]
    pub path: Vec<PathSegment>,
    pub extensions: Option<serde_json::Value>,
}

/// Source location in a GraphQL document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Location {
    pub line: u32,
    pub column: u32,
}

/// Path segment in a GraphQL error — either a field name or array index.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PathSegment {
    Field(String),
    Index(usize),
}

// ── QueryBuilder ────────────────────────────────────────────────

/// Fluent builder for GraphQL requests.
pub struct QueryBuilder {
    operation: Operation,
    query: String,
    variables: serde_json::Map<String, serde_json::Value>,
    operation_name: Option<String>,
}

impl QueryBuilder {
    /// Start building a query operation.
    pub fn query(q: &str) -> Self {
        Self {
            operation: Operation::Query,
            query: q.to_string(),
            variables: serde_json::Map::new(),
            operation_name: None,
        }
    }

    /// Start building a mutation operation.
    pub fn mutation(m: &str) -> Self {
        Self {
            operation: Operation::Mutation,
            query: m.to_string(),
            variables: serde_json::Map::new(),
            operation_name: None,
        }
    }

    /// Start building a subscription operation.
    pub fn subscription(s: &str) -> Self {
        Self {
            operation: Operation::Subscription,
            query: s.to_string(),
            variables: serde_json::Map::new(),
            operation_name: None,
        }
    }

    /// Add a variable.
    pub fn variable(mut self, name: &str, value: impl Serialize) -> Self {
        self.variables.insert(
            name.to_string(),
            serde_json::to_value(value).unwrap_or(serde_json::Value::Null),
        );
        self
    }

    /// Set the operation name.
    pub fn operation_name(mut self, name: &str) -> Self {
        self.operation_name = Some(name.to_string());
        self
    }

    /// Build the final request.
    pub fn build(self) -> GraphqlRequest {
        let variables = if self.variables.is_empty() {
            serde_json::Value::Null
        } else {
            serde_json::Value::Object(self.variables)
        };
        GraphqlRequest {
            operation: self.operation,
            query: self.query,
            variables,
            operation_name: self.operation_name,
        }
    }
}

// ── Cache ───────────────────────────────────────────────────────

/// A cached GraphQL response entry.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    pub data: serde_json::Value,
    pub fetched_at: DateTime<Utc>,
    pub stale: bool,
}

/// In-memory GraphQL response cache with entity normalization.
pub struct GraphqlCache {
    entries: HashMap<String, CacheEntry>,
    normalized: HashMap<String, serde_json::Value>,
}

impl GraphqlCache {
    /// Create an empty cache.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            normalized: HashMap::new(),
        }
    }

    /// Compute a deterministic cache key from a request (query + variables).
    pub fn cache_key(request: &GraphqlRequest) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        request.query.hash(&mut hasher);
        let vars = serde_json::to_string(&request.variables).unwrap_or_default();
        vars.hash(&mut hasher);
        format!("gql:{:016x}", hasher.finish())
    }

    /// Get a cached entry by key.
    pub fn get(&self, key: &str) -> Option<&CacheEntry> {
        self.entries.get(key)
    }

    /// Store a response in the cache.
    pub fn set(&mut self, key: &str, data: serde_json::Value) {
        self.entries.insert(
            key.to_string(),
            CacheEntry {
                data,
                fetched_at: Utc::now(),
                stale: false,
            },
        );
    }

    /// Invalidate (remove) a single cache entry.
    pub fn invalidate(&mut self, key: &str) {
        self.entries.remove(key);
    }

    /// Invalidate all cache entries.
    pub fn invalidate_all(&mut self) {
        self.entries.clear();
    }

    /// Remove all stale entries. Returns the number evicted.
    pub fn evict_stale(&mut self) -> usize {
        let before = self.entries.len();
        self.entries.retain(|_, entry| !entry.stale);
        before - self.entries.len()
    }

    // ── Normalization ───────────────────────────────────────────

    /// Extract entities with `__typename` and `id` fields from a response
    /// and store them in the normalized cache.
    pub fn normalize_response(&mut self, data: &serde_json::Value) {
        self.walk_and_normalize(data);
    }

    fn walk_and_normalize(&mut self, value: &serde_json::Value) {
        match value {
            serde_json::Value::Object(map) => {
                if let (Some(serde_json::Value::String(typename)), Some(id_val)) =
                    (map.get("__typename"), map.get("id"))
                {
                    let id_str = match id_val {
                        serde_json::Value::String(s) => s.clone(),
                        other => other.to_string(),
                    };
                    let key = format!("{typename}:{id_str}");
                    self.normalized.insert(key, value.clone());
                }
                for v in map.values() {
                    self.walk_and_normalize(v);
                }
            }
            serde_json::Value::Array(arr) => {
                for item in arr {
                    self.walk_and_normalize(item);
                }
            }
            _ => {}
        }
    }

    /// Read a normalized entity by typename and id.
    pub fn read_entity(&self, typename: &str, id: &str) -> Option<&serde_json::Value> {
        self.normalized.get(&format!("{typename}:{id}"))
    }

    /// Update (or insert) a normalized entity.
    pub fn update_entity(&mut self, typename: &str, id: &str, data: serde_json::Value) {
        self.normalized.insert(format!("{typename}:{id}"), data);
    }
}

impl Default for GraphqlCache {
    fn default() -> Self {
        Self::new()
    }
}

// ── Fragments ───────────────────────────────────────────────────

/// A reusable GraphQL fragment.
#[derive(Debug, Clone)]
pub struct GraphqlFragment {
    pub name: String,
    pub on_type: String,
    pub body: String,
}

impl GraphqlFragment {
    /// Create a new fragment.
    pub fn new(name: &str, on_type: &str, body: &str) -> Self {
        Self {
            name: name.to_string(),
            on_type: on_type.to_string(),
            body: body.to_string(),
        }
    }
}

impl fmt::Display for GraphqlFragment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "fragment {} on {} {{ {} }}",
            self.name, self.on_type, self.body
        )
    }
}

use std::fmt;

/// Compose a query string with fragment definitions appended.
pub fn compose_query(query: &str, fragments: &[&GraphqlFragment]) -> String {
    let mut result = query.to_string();
    for frag in fragments {
        result.push('\n');
        result.push_str(&frag.to_string());
    }
    result
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_builder_produces_correct_json() {
        let req = QueryBuilder::query("{ users { id name } }").build();
        let json = req.to_json();
        assert_eq!(json["query"], "{ users { id name } }");
        assert!(json.get("variables").is_none()); // null variables omitted
    }

    #[test]
    fn variables_serialized() {
        let req = QueryBuilder::query("query($id: ID!) { user(id: $id) { name } }")
            .variable("id", "123")
            .build();
        let json = req.to_json();
        assert_eq!(json["variables"]["id"], "123");
    }

    #[test]
    fn mutation_operation() {
        let req = QueryBuilder::mutation("mutation { deleteUser(id: 1) }")
            .operation_name("DeleteUser")
            .build();
        assert_eq!(req.operation, Operation::Mutation);
        let json = req.to_json();
        assert_eq!(json["operationName"], "DeleteUser");
    }

    #[test]
    fn cache_set_get() {
        let mut cache = GraphqlCache::new();
        cache.set("key1", serde_json::json!({"users": []}));
        let entry = cache.get("key1").unwrap();
        assert_eq!(entry.data, serde_json::json!({"users": []}));
        assert!(!entry.stale);
    }

    #[test]
    fn cache_key_deterministic() {
        let req = QueryBuilder::query("{ users { id } }")
            .variable("limit", 10)
            .build();
        let k1 = GraphqlCache::cache_key(&req);
        let k2 = GraphqlCache::cache_key(&req);
        assert_eq!(k1, k2);
    }

    #[test]
    fn invalidate_removes() {
        let mut cache = GraphqlCache::new();
        cache.set("k", serde_json::json!(null));
        assert!(cache.get("k").is_some());
        cache.invalidate("k");
        assert!(cache.get("k").is_none());
    }

    #[test]
    fn normalize_extracts_entities() {
        let mut cache = GraphqlCache::new();
        let data = serde_json::json!({
            "users": [
                {"__typename": "User", "id": "1", "name": "Alice"},
                {"__typename": "User", "id": "2", "name": "Bob"}
            ]
        });
        cache.normalize_response(&data);
        assert!(cache.read_entity("User", "1").is_some());
        assert!(cache.read_entity("User", "2").is_some());
    }

    #[test]
    fn read_entity_by_typename_id() {
        let mut cache = GraphqlCache::new();
        let data = serde_json::json!({"__typename": "Post", "id": "42", "title": "Hello"});
        cache.normalize_response(&data);
        let entity = cache.read_entity("Post", "42").unwrap();
        assert_eq!(entity["title"], "Hello");
    }

    #[test]
    fn update_entity() {
        let mut cache = GraphqlCache::new();
        cache.update_entity("User", "1", serde_json::json!({"name": "Alice"}));
        let e = cache.read_entity("User", "1").unwrap();
        assert_eq!(e["name"], "Alice");

        cache.update_entity("User", "1", serde_json::json!({"name": "Alicia"}));
        let e = cache.read_entity("User", "1").unwrap();
        assert_eq!(e["name"], "Alicia");
    }

    #[test]
    fn fragment_composition() {
        let frag = GraphqlFragment::new("UserFields", "User", "id name email");
        let query = "query { users { ...UserFields } }";
        let composed = compose_query(query, &[&frag]);
        assert!(composed.contains("fragment UserFields on User { id name email }"));
        assert!(composed.starts_with(query));
    }

    #[test]
    fn response_with_errors() {
        let json_str = r#"{"data":null,"errors":[{"message":"Not found","locations":[{"line":1,"column":3}],"path":["user"]}]}"#;
        let resp: GraphqlResponse = serde_json::from_str(json_str).unwrap();
        assert!(resp.data.is_none() || resp.data == Some(serde_json::Value::Null));
        let errors = resp.errors.unwrap();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].message, "Not found");
    }

    #[test]
    fn stale_eviction() {
        let mut cache = GraphqlCache::new();
        cache.set("fresh", serde_json::json!(1));
        cache.set("old", serde_json::json!(2));
        // Mark one as stale
        cache.entries.get_mut("old").unwrap().stale = true;
        let evicted = cache.evict_stale();
        assert_eq!(evicted, 1);
        assert!(cache.get("fresh").is_some());
        assert!(cache.get("old").is_none());
    }

    #[test]
    fn invalidate_all_clears() {
        let mut cache = GraphqlCache::new();
        cache.set("a", serde_json::json!(1));
        cache.set("b", serde_json::json!(2));
        cache.invalidate_all();
        assert!(cache.get("a").is_none());
        assert!(cache.get("b").is_none());
    }
}
