//! Persistence Layer for JouleDB Features
//!
//! Provides durable storage for specialized data structures by
//! persisting them to the B-tree storage engine.
//!
//! ## Key Formats
//!
//! Each feature uses a unique key prefix:
//! - Time Series: `__ts__::{metric}::{partition}::{timestamp}`
//! - Graph Nodes: `__graph__::node::{id}`
//! - Graph Edges: `__graph__::edge::{id}` + adjacency indices
//! - Vector Index: `__vec__::{index}::{id}`
//! - Full-Text: `__ft__::{index}::term::{term}` + `__ft__::{index}::doc::{id}`

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Storage backend trait for persistence
pub trait StorageEngine: Send + Sync {
    /// Get a value by key
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, PersistenceError>;

    /// Put a key-value pair
    fn put(&self, key: &[u8], value: &[u8]) -> Result<(), PersistenceError>;

    /// Delete a key
    fn delete(&self, key: &[u8]) -> Result<bool, PersistenceError>;

    /// Scan keys with a prefix
    fn prefix_scan(&self, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>, PersistenceError>;
}

/// Persistence error
#[derive(Debug, Clone)]
pub enum PersistenceError {
    /// Storage error
    Storage(String),
    /// Serialization error
    Serialization(String),
    /// Not found
    NotFound(String),
}

impl std::fmt::Display for PersistenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PersistenceError::Storage(s) => write!(f, "Storage error: {}", s),
            PersistenceError::Serialization(s) => write!(f, "Serialization error: {}", s),
            PersistenceError::NotFound(s) => write!(f, "Not found: {}", s),
        }
    }
}

impl std::error::Error for PersistenceError {}

/// Result type for persistence operations
pub type PersistResult<T> = Result<T, PersistenceError>;

// ============================================================================
// Time Series Persistence
// ============================================================================

const TS_PREFIX: &[u8] = b"__ts__::";
const TS_META_PREFIX: &[u8] = b"__ts__::meta::";

/// Persistent time series data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedDataPoint {
    pub timestamp: i64,
    pub value: f64,
    pub tags: HashMap<String, String>,
}

/// Time series persistence adapter
pub struct TimeSeriesPersistence<S: StorageEngine> {
    storage: Arc<S>,
}

impl<S: StorageEngine> TimeSeriesPersistence<S> {
    /// Create new time series persistence
    pub fn new(storage: Arc<S>) -> Self {
        Self { storage }
    }

    /// Write a data point
    pub fn write(&self, metric: &str, point: PersistedDataPoint) -> PersistResult<()> {
        let key = self.point_key(metric, point.timestamp);
        let value = serde_json::to_vec(&point)
            .map_err(|e| PersistenceError::Serialization(e.to_string()))?;
        self.storage.put(&key, &value)
    }

    /// Write multiple data points
    pub fn write_batch(&self, metric: &str, points: Vec<PersistedDataPoint>) -> PersistResult<()> {
        for point in points {
            self.write(metric, point)?;
        }
        Ok(())
    }

    /// Query time range
    pub fn query(
        &self,
        metric: &str,
        start: i64,
        end: i64,
    ) -> PersistResult<Vec<PersistedDataPoint>> {
        let prefix = self.metric_prefix(metric);
        let entries = self.storage.prefix_scan(&prefix)?;

        let mut points = Vec::new();
        for (key, value) in entries {
            // Extract timestamp from key
            if let Some(ts) = self.extract_timestamp(&key, &prefix) {
                if ts >= start && ts <= end {
                    let point: PersistedDataPoint = serde_json::from_slice(&value)
                        .map_err(|e| PersistenceError::Serialization(e.to_string()))?;
                    points.push(point);
                }
            }
        }

        points.sort_by_key(|p| p.timestamp);
        Ok(points)
    }

    /// List all metrics
    pub fn list_metrics(&self) -> PersistResult<Vec<String>> {
        let entries = self.storage.prefix_scan(TS_META_PREFIX)?;
        let mut metrics = Vec::new();

        for (key, _) in entries {
            if let Some(name) = self.extract_metric_name(&key) {
                metrics.push(name);
            }
        }

        Ok(metrics)
    }

    /// Register a metric (for listing)
    pub fn register_metric(&self, metric: &str) -> PersistResult<()> {
        let key = [TS_META_PREFIX, metric.as_bytes()].concat();
        self.storage.put(&key, b"1")
    }

    /// Delete a metric and all its data
    pub fn delete_metric(&self, metric: &str) -> PersistResult<()> {
        let prefix = self.metric_prefix(metric);
        let entries = self.storage.prefix_scan(&prefix)?;

        for (key, _) in entries {
            self.storage.delete(&key)?;
        }

        // Delete metadata
        let meta_key = [TS_META_PREFIX, metric.as_bytes()].concat();
        self.storage.delete(&meta_key)?;

        Ok(())
    }

    // Helper methods

    fn metric_prefix(&self, metric: &str) -> Vec<u8> {
        [TS_PREFIX, metric.as_bytes(), b"::"].concat()
    }

    fn point_key(&self, metric: &str, timestamp: i64) -> Vec<u8> {
        let mut key = self.metric_prefix(metric);
        key.extend_from_slice(&timestamp.to_be_bytes());
        key
    }

    fn extract_timestamp(&self, key: &[u8], prefix: &[u8]) -> Option<i64> {
        if key.len() >= prefix.len() + 8 {
            let ts_bytes: [u8; 8] = key[prefix.len()..prefix.len() + 8].try_into().ok()?;
            Some(i64::from_be_bytes(ts_bytes))
        } else {
            None
        }
    }

    fn extract_metric_name(&self, key: &[u8]) -> Option<String> {
        if key.starts_with(TS_META_PREFIX) {
            String::from_utf8(key[TS_META_PREFIX.len()..].to_vec()).ok()
        } else {
            None
        }
    }
}

// ============================================================================
// Graph Persistence
// ============================================================================

const GRAPH_NODE_PREFIX: &[u8] = b"__graph__::node::";
const GRAPH_EDGE_PREFIX: &[u8] = b"__graph__::edge::";
const GRAPH_OUT_PREFIX: &[u8] = b"__graph__::out::";
const GRAPH_IN_PREFIX: &[u8] = b"__graph__::in::";
const GRAPH_LABEL_PREFIX: &[u8] = b"__graph__::label::";
const GRAPH_META: &[u8] = b"__graph__::meta";

/// Persisted graph node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedNode {
    pub id: u64,
    pub labels: Vec<String>,
    pub properties: HashMap<String, serde_json::Value>,
}

/// Persisted graph edge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedEdge {
    pub id: u64,
    pub from: u64,
    pub to: u64,
    pub edge_type: String,
    pub properties: HashMap<String, serde_json::Value>,
}

/// Graph metadata
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct GraphMeta {
    next_node_id: u64,
    next_edge_id: u64,
}

/// Graph persistence adapter
pub struct GraphPersistence<S: StorageEngine> {
    storage: Arc<S>,
}

impl<S: StorageEngine> GraphPersistence<S> {
    /// Create new graph persistence
    pub fn new(storage: Arc<S>) -> Self {
        Self { storage }
    }

    /// Create a node
    pub fn create_node(
        &self,
        labels: Vec<String>,
        properties: HashMap<String, serde_json::Value>,
    ) -> PersistResult<u64> {
        let id = self.next_node_id()?;
        let node = PersistedNode {
            id,
            labels: labels.clone(),
            properties,
        };

        // Store node
        let key = self.node_key(id);
        let value = serde_json::to_vec(&node)
            .map_err(|e| PersistenceError::Serialization(e.to_string()))?;
        self.storage.put(&key, &value)?;

        // Update label indices
        for label in &labels {
            let label_key = self.label_key(label, id);
            self.storage.put(&label_key, &id.to_le_bytes())?;
        }

        Ok(id)
    }

    /// Get a node
    pub fn get_node(&self, id: u64) -> PersistResult<Option<PersistedNode>> {
        let key = self.node_key(id);
        match self.storage.get(&key)? {
            Some(data) => {
                let node = serde_json::from_slice(&data)
                    .map_err(|e| PersistenceError::Serialization(e.to_string()))?;
                Ok(Some(node))
            }
            None => Ok(None),
        }
    }

    /// Delete a node
    pub fn delete_node(&self, id: u64) -> PersistResult<bool> {
        // Get node for labels
        let node = match self.get_node(id)? {
            Some(n) => n,
            None => return Ok(false),
        };

        // Delete edges connected to this node
        let out_edges = self.get_outgoing_edges(id)?;
        for edge in out_edges {
            self.delete_edge(edge.id)?;
        }

        let in_edges = self.get_incoming_edges(id)?;
        for edge in in_edges {
            self.delete_edge(edge.id)?;
        }

        // Delete label indices
        for label in &node.labels {
            let label_key = self.label_key(label, id);
            self.storage.delete(&label_key)?;
        }

        // Delete node
        let key = self.node_key(id);
        self.storage.delete(&key)
    }

    /// Create an edge
    pub fn create_edge(
        &self,
        from: u64,
        to: u64,
        edge_type: String,
        properties: HashMap<String, serde_json::Value>,
    ) -> PersistResult<u64> {
        // Verify nodes exist
        if self.get_node(from)?.is_none() {
            return Err(PersistenceError::NotFound(format!(
                "Node {} not found",
                from
            )));
        }
        if self.get_node(to)?.is_none() {
            return Err(PersistenceError::NotFound(format!("Node {} not found", to)));
        }

        let id = self.next_edge_id()?;
        let edge = PersistedEdge {
            id,
            from,
            to,
            edge_type,
            properties,
        };

        // Store edge
        let key = self.edge_key(id);
        let value = serde_json::to_vec(&edge)
            .map_err(|e| PersistenceError::Serialization(e.to_string()))?;
        self.storage.put(&key, &value)?;

        // Update adjacency indices
        let out_key = self.out_key(from, id);
        self.storage.put(&out_key, &id.to_le_bytes())?;

        let in_key = self.in_key(to, id);
        self.storage.put(&in_key, &id.to_le_bytes())?;

        Ok(id)
    }

    /// Get an edge
    pub fn get_edge(&self, id: u64) -> PersistResult<Option<PersistedEdge>> {
        let key = self.edge_key(id);
        match self.storage.get(&key)? {
            Some(data) => {
                let edge = serde_json::from_slice(&data)
                    .map_err(|e| PersistenceError::Serialization(e.to_string()))?;
                Ok(Some(edge))
            }
            None => Ok(None),
        }
    }

    /// Delete an edge
    pub fn delete_edge(&self, id: u64) -> PersistResult<bool> {
        let edge = match self.get_edge(id)? {
            Some(e) => e,
            None => return Ok(false),
        };

        // Delete adjacency indices
        let out_key = self.out_key(edge.from, id);
        self.storage.delete(&out_key)?;

        let in_key = self.in_key(edge.to, id);
        self.storage.delete(&in_key)?;

        // Delete edge
        let key = self.edge_key(id);
        self.storage.delete(&key)
    }

    /// Get outgoing edges from a node
    pub fn get_outgoing_edges(&self, node_id: u64) -> PersistResult<Vec<PersistedEdge>> {
        let prefix = self.out_prefix(node_id);
        let entries = self.storage.prefix_scan(&prefix)?;

        let mut edges = Vec::new();
        for (_, value) in entries {
            if value.len() >= 8 {
                let edge_id = u64::from_le_bytes(value[..8].try_into().unwrap());
                if let Some(edge) = self.get_edge(edge_id)? {
                    edges.push(edge);
                }
            }
        }

        Ok(edges)
    }

    /// Get incoming edges to a node
    pub fn get_incoming_edges(&self, node_id: u64) -> PersistResult<Vec<PersistedEdge>> {
        let prefix = self.in_prefix(node_id);
        let entries = self.storage.prefix_scan(&prefix)?;

        let mut edges = Vec::new();
        for (_, value) in entries {
            if value.len() >= 8 {
                let edge_id = u64::from_le_bytes(value[..8].try_into().unwrap());
                if let Some(edge) = self.get_edge(edge_id)? {
                    edges.push(edge);
                }
            }
        }

        Ok(edges)
    }

    /// Get nodes by label
    pub fn get_nodes_by_label(&self, label: &str) -> PersistResult<Vec<PersistedNode>> {
        let prefix = self.label_prefix(label);
        let entries = self.storage.prefix_scan(&prefix)?;

        let mut nodes = Vec::new();
        for (_, value) in entries {
            if value.len() >= 8 {
                let node_id = u64::from_le_bytes(value[..8].try_into().unwrap());
                if let Some(node) = self.get_node(node_id)? {
                    nodes.push(node);
                }
            }
        }

        Ok(nodes)
    }

    // Helper methods

    fn node_key(&self, id: u64) -> Vec<u8> {
        [GRAPH_NODE_PREFIX, &id.to_le_bytes()].concat()
    }

    fn edge_key(&self, id: u64) -> Vec<u8> {
        [GRAPH_EDGE_PREFIX, &id.to_le_bytes()].concat()
    }

    fn out_prefix(&self, node_id: u64) -> Vec<u8> {
        [GRAPH_OUT_PREFIX, &node_id.to_le_bytes(), b"::"].concat()
    }

    fn out_key(&self, node_id: u64, edge_id: u64) -> Vec<u8> {
        [&self.out_prefix(node_id)[..], &edge_id.to_le_bytes()].concat()
    }

    fn in_prefix(&self, node_id: u64) -> Vec<u8> {
        [GRAPH_IN_PREFIX, &node_id.to_le_bytes(), b"::"].concat()
    }

    fn in_key(&self, node_id: u64, edge_id: u64) -> Vec<u8> {
        [&self.in_prefix(node_id)[..], &edge_id.to_le_bytes()].concat()
    }

    fn label_prefix(&self, label: &str) -> Vec<u8> {
        [GRAPH_LABEL_PREFIX, label.as_bytes(), b"::"].concat()
    }

    fn label_key(&self, label: &str, node_id: u64) -> Vec<u8> {
        [&self.label_prefix(label)[..], &node_id.to_le_bytes()].concat()
    }

    fn next_node_id(&self) -> PersistResult<u64> {
        let mut meta = self.get_meta()?;
        meta.next_node_id += 1;
        self.save_meta(&meta)?;
        Ok(meta.next_node_id)
    }

    fn next_edge_id(&self) -> PersistResult<u64> {
        let mut meta = self.get_meta()?;
        meta.next_edge_id += 1;
        self.save_meta(&meta)?;
        Ok(meta.next_edge_id)
    }

    fn get_meta(&self) -> PersistResult<GraphMeta> {
        match self.storage.get(GRAPH_META)? {
            Some(data) => serde_json::from_slice(&data)
                .map_err(|e| PersistenceError::Serialization(e.to_string())),
            None => Ok(GraphMeta::default()),
        }
    }

    fn save_meta(&self, meta: &GraphMeta) -> PersistResult<()> {
        let value =
            serde_json::to_vec(meta).map_err(|e| PersistenceError::Serialization(e.to_string()))?;
        self.storage.put(GRAPH_META, &value)
    }
}

// ============================================================================
// Vector Index Persistence
// ============================================================================

const VEC_PREFIX: &[u8] = b"__vec__::";
const VEC_META_PREFIX: &[u8] = b"__vec__::meta::";

/// Persisted vector entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedVector {
    pub id: String,
    pub vector: Vec<f32>,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// Vector index persistence adapter
pub struct VectorPersistence<S: StorageEngine> {
    storage: Arc<S>,
    index_name: String,
}

impl<S: StorageEngine> VectorPersistence<S> {
    /// Create new vector persistence for an index
    pub fn new(storage: Arc<S>, index_name: &str) -> Self {
        Self {
            storage,
            index_name: index_name.to_string(),
        }
    }

    /// Insert a vector
    pub fn insert(
        &self,
        id: &str,
        vector: Vec<f32>,
        metadata: Option<HashMap<String, serde_json::Value>>,
    ) -> PersistResult<()> {
        let entry = PersistedVector {
            id: id.to_string(),
            vector,
            metadata,
        };

        let key = self.vector_key(id);
        let value = serde_json::to_vec(&entry)
            .map_err(|e| PersistenceError::Serialization(e.to_string()))?;
        self.storage.put(&key, &value)
    }

    /// Get a vector
    pub fn get(&self, id: &str) -> PersistResult<Option<PersistedVector>> {
        let key = self.vector_key(id);
        match self.storage.get(&key)? {
            Some(data) => {
                let entry = serde_json::from_slice(&data)
                    .map_err(|e| PersistenceError::Serialization(e.to_string()))?;
                Ok(Some(entry))
            }
            None => Ok(None),
        }
    }

    /// Delete a vector
    pub fn delete(&self, id: &str) -> PersistResult<bool> {
        let key = self.vector_key(id);
        self.storage.delete(&key)
    }

    /// Get all vectors (for rebuilding index)
    pub fn get_all(&self) -> PersistResult<Vec<PersistedVector>> {
        let prefix = self.index_prefix();
        let entries = self.storage.prefix_scan(&prefix)?;

        let mut vectors = Vec::new();
        for (_, value) in entries {
            let entry: PersistedVector = serde_json::from_slice(&value)
                .map_err(|e| PersistenceError::Serialization(e.to_string()))?;
            vectors.push(entry);
        }

        Ok(vectors)
    }

    // Helper methods

    fn index_prefix(&self) -> Vec<u8> {
        [VEC_PREFIX, self.index_name.as_bytes(), b"::"].concat()
    }

    fn vector_key(&self, id: &str) -> Vec<u8> {
        [&self.index_prefix()[..], id.as_bytes()].concat()
    }
}

// ============================================================================
// Full-Text Index Persistence
// ============================================================================

const FT_TERM_PREFIX: &[u8] = b"__ft__::term::";
const FT_DOC_PREFIX: &[u8] = b"__ft__::doc::";
const FT_META_PREFIX: &[u8] = b"__ft__::meta::";

/// Persisted document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedDocument {
    pub id: String,
    pub content: String,
    pub length: usize,
}

/// Persisted posting list entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedPosting {
    pub doc_id: String,
    pub positions: Vec<usize>,
    pub term_frequency: f64,
}

/// Full-text index persistence adapter
pub struct FullTextPersistence<S: StorageEngine> {
    storage: Arc<S>,
    index_name: String,
}

impl<S: StorageEngine> FullTextPersistence<S> {
    /// Create new full-text persistence for an index
    pub fn new(storage: Arc<S>, index_name: &str) -> Self {
        Self {
            storage,
            index_name: index_name.to_string(),
        }
    }

    /// Add a document
    pub fn add_document(&self, id: &str, content: &str) -> PersistResult<()> {
        let doc = PersistedDocument {
            id: id.to_string(),
            content: content.to_string(),
            length: content.split_whitespace().count(),
        };

        let key = self.doc_key(id);
        let value =
            serde_json::to_vec(&doc).map_err(|e| PersistenceError::Serialization(e.to_string()))?;
        self.storage.put(&key, &value)
    }

    /// Get a document
    pub fn get_document(&self, id: &str) -> PersistResult<Option<PersistedDocument>> {
        let key = self.doc_key(id);
        match self.storage.get(&key)? {
            Some(data) => {
                let doc = serde_json::from_slice(&data)
                    .map_err(|e| PersistenceError::Serialization(e.to_string()))?;
                Ok(Some(doc))
            }
            None => Ok(None),
        }
    }

    /// Delete a document
    pub fn delete_document(&self, id: &str) -> PersistResult<bool> {
        let key = self.doc_key(id);
        self.storage.delete(&key)
    }

    /// Add posting for a term
    pub fn add_posting(&self, term: &str, posting: PersistedPosting) -> PersistResult<()> {
        let key = self.term_doc_key(term, &posting.doc_id);
        let value = serde_json::to_vec(&posting)
            .map_err(|e| PersistenceError::Serialization(e.to_string()))?;
        self.storage.put(&key, &value)
    }

    /// Get postings for a term
    pub fn get_postings(&self, term: &str) -> PersistResult<Vec<PersistedPosting>> {
        let prefix = self.term_prefix(term);
        let entries = self.storage.prefix_scan(&prefix)?;

        let mut postings = Vec::new();
        for (_, value) in entries {
            let posting: PersistedPosting = serde_json::from_slice(&value)
                .map_err(|e| PersistenceError::Serialization(e.to_string()))?;
            postings.push(posting);
        }

        Ok(postings)
    }

    /// Get all documents
    pub fn get_all_documents(&self) -> PersistResult<Vec<PersistedDocument>> {
        let prefix = self.doc_prefix();
        let entries = self.storage.prefix_scan(&prefix)?;

        let mut docs = Vec::new();
        for (_, value) in entries {
            let doc: PersistedDocument = serde_json::from_slice(&value)
                .map_err(|e| PersistenceError::Serialization(e.to_string()))?;
            docs.push(doc);
        }

        Ok(docs)
    }

    // Helper methods

    fn doc_prefix(&self) -> Vec<u8> {
        [FT_DOC_PREFIX, self.index_name.as_bytes(), b"::"].concat()
    }

    fn doc_key(&self, id: &str) -> Vec<u8> {
        [&self.doc_prefix()[..], id.as_bytes()].concat()
    }

    fn term_prefix(&self, term: &str) -> Vec<u8> {
        [
            FT_TERM_PREFIX,
            self.index_name.as_bytes(),
            b"::",
            term.as_bytes(),
            b"::",
        ]
        .concat()
    }

    fn term_doc_key(&self, term: &str, doc_id: &str) -> Vec<u8> {
        [&self.term_prefix(term)[..], doc_id.as_bytes()].concat()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::RwLock;

    /// In-memory storage for testing
    struct MemoryStorage {
        data: RwLock<HashMap<Vec<u8>, Vec<u8>>>,
    }

    impl MemoryStorage {
        fn new() -> Self {
            Self {
                data: RwLock::new(HashMap::new()),
            }
        }
    }

    impl StorageEngine for MemoryStorage {
        fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, PersistenceError> {
            Ok(self.data.read().unwrap().get(key).cloned())
        }

        fn put(&self, key: &[u8], value: &[u8]) -> Result<(), PersistenceError> {
            self.data
                .write()
                .unwrap()
                .insert(key.to_vec(), value.to_vec());
            Ok(())
        }

        fn delete(&self, key: &[u8]) -> Result<bool, PersistenceError> {
            Ok(self.data.write().unwrap().remove(key).is_some())
        }

        fn prefix_scan(&self, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>, PersistenceError> {
            let data = self.data.read().unwrap();
            let mut results: Vec<_> = data
                .iter()
                .filter(|(k, _)| k.starts_with(prefix))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            results.sort_by(|a, b| a.0.cmp(&b.0));
            Ok(results)
        }
    }

    #[test]
    fn test_timeseries_persistence() {
        let storage = Arc::new(MemoryStorage::new());
        let ts = TimeSeriesPersistence::new(storage);

        ts.register_metric("cpu").unwrap();

        ts.write(
            "cpu",
            PersistedDataPoint {
                timestamp: 1000,
                value: 50.0,
                tags: HashMap::new(),
            },
        )
        .unwrap();

        ts.write(
            "cpu",
            PersistedDataPoint {
                timestamp: 2000,
                value: 60.0,
                tags: HashMap::new(),
            },
        )
        .unwrap();

        let points = ts.query("cpu", 0, 3000).unwrap();
        assert_eq!(points.len(), 2);
        assert_eq!(points[0].value, 50.0);
        assert_eq!(points[1].value, 60.0);

        let metrics = ts.list_metrics().unwrap();
        assert!(metrics.contains(&"cpu".to_string()));
    }

    #[test]
    fn test_graph_persistence() {
        let storage = Arc::new(MemoryStorage::new());
        let graph = GraphPersistence::new(storage);

        let node1 = graph
            .create_node(
                vec!["Person".to_string()],
                [("name".to_string(), serde_json::json!("Alice"))]
                    .into_iter()
                    .collect(),
            )
            .unwrap();

        let node2 = graph
            .create_node(
                vec!["Person".to_string()],
                [("name".to_string(), serde_json::json!("Bob"))]
                    .into_iter()
                    .collect(),
            )
            .unwrap();

        let edge_id = graph
            .create_edge(node1, node2, "KNOWS".to_string(), HashMap::new())
            .unwrap();

        let node = graph.get_node(node1).unwrap().unwrap();
        assert_eq!(node.labels, vec!["Person"]);

        let edge = graph.get_edge(edge_id).unwrap().unwrap();
        assert_eq!(edge.from, node1);
        assert_eq!(edge.to, node2);

        let out_edges = graph.get_outgoing_edges(node1).unwrap();
        assert_eq!(out_edges.len(), 1);

        let persons = graph.get_nodes_by_label("Person").unwrap();
        assert_eq!(persons.len(), 2);
    }

    #[test]
    fn test_vector_persistence() {
        let storage = Arc::new(MemoryStorage::new());
        let vecs = VectorPersistence::new(storage, "embeddings");

        vecs.insert("doc1", vec![0.1, 0.2, 0.3], None).unwrap();
        vecs.insert(
            "doc2",
            vec![0.4, 0.5, 0.6],
            Some(
                [("title".to_string(), serde_json::json!("Test"))]
                    .into_iter()
                    .collect(),
            ),
        )
        .unwrap();

        let v1 = vecs.get("doc1").unwrap().unwrap();
        assert_eq!(v1.vector, vec![0.1, 0.2, 0.3]);

        let all = vecs.get_all().unwrap();
        assert_eq!(all.len(), 2);

        vecs.delete("doc1").unwrap();
        assert!(vecs.get("doc1").unwrap().is_none());
    }

    #[test]
    fn test_fulltext_persistence() {
        let storage = Arc::new(MemoryStorage::new());
        let ft = FullTextPersistence::new(storage, "search");

        ft.add_document("doc1", "hello world").unwrap();
        ft.add_document("doc2", "world peace").unwrap();

        ft.add_posting(
            "hello",
            PersistedPosting {
                doc_id: "doc1".to_string(),
                positions: vec![0],
                term_frequency: 1.0,
            },
        )
        .unwrap();

        ft.add_posting(
            "world",
            PersistedPosting {
                doc_id: "doc1".to_string(),
                positions: vec![1],
                term_frequency: 1.0,
            },
        )
        .unwrap();

        ft.add_posting(
            "world",
            PersistedPosting {
                doc_id: "doc2".to_string(),
                positions: vec![0],
                term_frequency: 1.0,
            },
        )
        .unwrap();

        let doc = ft.get_document("doc1").unwrap().unwrap();
        assert_eq!(doc.content, "hello world");

        let postings = ft.get_postings("world").unwrap();
        assert_eq!(postings.len(), 2);

        let all_docs = ft.get_all_documents().unwrap();
        assert_eq!(all_docs.len(), 2);
    }
}
