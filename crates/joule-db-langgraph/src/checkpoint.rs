//! Checkpoint store for LangGraph agents
//!
//! This module provides checkpoint storage for LangGraph agents, allowing them
//! to save and restore their state across executions.
//!
//! ## Checkpointing Model
//!
//! Checkpoints are organized by:
//! - **Thread ID** - The conversation/session identifier
//! - **Checkpoint ID** - A unique identifier for each checkpoint
//!
//! Each checkpoint contains:
//! - The serialized agent state (arbitrary JSON)
//! - Metadata (timestamp, parent checkpoint, etc.)
//!
//! ## Semantic Features
//!
//! Unlike traditional checkpoint stores, JouleDB's implementation stores
//! checkpoints as semantic entities, enabling:
//! - Finding similar states across threads
//! - Semantic versioning based on state similarity
//! - Efficient state deduplication

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use joule_db_amorphic::AmorphicStore;

use crate::error::{LangGraphError, LangGraphResult};

/// Metadata for a checkpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// Unique checkpoint ID
    pub checkpoint_id: String,
    /// Thread this checkpoint belongs to
    pub thread_id: String,
    /// Parent checkpoint ID (if any)
    pub parent_id: Option<String>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Step number (optional)
    pub step: Option<u64>,
    /// Custom tags
    pub tags: Vec<String>,
}

impl CheckpointMetadata {
    /// Create new metadata for a checkpoint
    pub fn new(thread_id: impl Into<String>, checkpoint_id: impl Into<String>) -> Self {
        Self {
            checkpoint_id: checkpoint_id.into(),
            thread_id: thread_id.into(),
            parent_id: None,
            created_at: Utc::now(),
            step: None,
            tags: Vec::new(),
        }
    }

    /// Set the parent checkpoint ID
    pub fn with_parent(mut self, parent_id: impl Into<String>) -> Self {
        self.parent_id = Some(parent_id.into());
        self
    }

    /// Set the step number
    pub fn with_step(mut self, step: u64) -> Self {
        self.step = Some(step);
        self
    }

    /// Add tags
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }
}

/// A checkpoint containing agent state and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Checkpoint metadata
    pub metadata: CheckpointMetadata,
    /// The actual agent state (arbitrary JSON)
    pub state: Value,
}

impl Checkpoint {
    /// Create a new checkpoint
    pub fn new(thread_id: &str, checkpoint_id: &str, state: Value) -> Self {
        Self {
            metadata: CheckpointMetadata::new(thread_id, checkpoint_id),
            state,
        }
    }

    /// Get the checkpoint ID
    pub fn checkpoint_id(&self) -> &str {
        &self.metadata.checkpoint_id
    }

    /// Get the thread ID
    pub fn thread_id(&self) -> &str {
        &self.metadata.thread_id
    }
}

/// Checkpoint store backed by AmorphicStore
///
/// Provides checkpoint storage for LangGraph agents with semantic capabilities.
pub struct JouleCheckpointStore {
    /// The underlying AmorphicStore
    store: AmorphicStore,
    /// Index mapping thread_id -> checkpoint_ids (most recent last)
    thread_index: HashMap<String, Vec<String>>,
    /// Index mapping checkpoint_id -> RecordId
    checkpoint_index: HashMap<String, joule_db_amorphic::RecordId>,
}

impl JouleCheckpointStore {
    /// Create a new in-memory checkpoint store
    pub fn new() -> Self {
        Self {
            store: AmorphicStore::new(),
            thread_index: HashMap::new(),
            checkpoint_index: HashMap::new(),
        }
    }

    /// Put a checkpoint
    pub fn put_checkpoint(
        &mut self,
        thread_id: &str,
        checkpoint_id: &str,
        state: &Value,
    ) -> LangGraphResult<()> {
        self.put_checkpoint_with_metadata(thread_id, checkpoint_id, state, None)
    }

    /// Put a checkpoint with custom metadata
    pub fn put_checkpoint_with_metadata(
        &mut self,
        thread_id: &str,
        checkpoint_id: &str,
        state: &Value,
        metadata: Option<CheckpointMetadata>,
    ) -> LangGraphResult<()> {
        // Create checkpoint
        let checkpoint = Checkpoint {
            metadata: metadata.unwrap_or_else(|| CheckpointMetadata::new(thread_id, checkpoint_id)),
            state: state.clone(),
        };

        // Serialize to JSON for storage
        let json = serde_json::to_string(&checkpoint)?;

        // Store in AmorphicStore
        let record_id = self.store.ingest_json(&json)?;

        // Update indices
        self.thread_index
            .entry(thread_id.to_string())
            .or_default()
            .push(checkpoint_id.to_string());

        self.checkpoint_index
            .insert(checkpoint_id.to_string(), record_id);

        Ok(())
    }

    /// Get the latest checkpoint for a thread
    pub fn get_checkpoint(&self, thread_id: &str) -> Option<Checkpoint> {
        let checkpoints = self.thread_index.get(thread_id)?;
        let latest_id = checkpoints.last()?;
        self.get_checkpoint_by_id(latest_id)
    }

    /// Get a specific checkpoint by ID
    pub fn get_checkpoint_by_id(&self, checkpoint_id: &str) -> Option<Checkpoint> {
        let record_id = self.checkpoint_index.get(checkpoint_id)?;
        let record = self.store.get(*record_id)?;

        // Reconstruct the checkpoint from fields
        // The record has the JSON stored, we need to deserialize it
        if let Some(joule_db_amorphic::Value::String(json)) = record.get("_raw") {
            serde_json::from_str(json).ok()
        } else {
            // Try to reconstruct from individual fields
            self.reconstruct_checkpoint_from_record(record)
        }
    }

    /// Reconstruct a checkpoint from an AmorphicRecord
    fn reconstruct_checkpoint_from_record(
        &self,
        record: &joule_db_amorphic::AmorphicRecord,
    ) -> Option<Checkpoint> {
        // Get the metadata object
        let metadata_obj = record.get("metadata")?;
        let metadata_map = match metadata_obj {
            joule_db_amorphic::Value::Object(map) => map,
            _ => return None,
        };

        // Extract checkpoint_id from metadata
        let checkpoint_id = metadata_map
            .get("checkpoint_id")
            .and_then(|v| match v {
                joule_db_amorphic::Value::String(s) => Some(s.clone()),
                _ => None,
            })
            .unwrap_or_else(|| Uuid::new_v4().to_string());

        // Extract thread_id from metadata
        let thread_id = metadata_map
            .get("thread_id")
            .and_then(|v| match v {
                joule_db_amorphic::Value::String(s) => Some(s.clone()),
                _ => None,
            })
            .unwrap_or_default();

        // Extract parent_id from metadata
        let parent_id = metadata_map.get("parent_id").and_then(|v| match v {
            joule_db_amorphic::Value::String(s) => Some(s.clone()),
            joule_db_amorphic::Value::Null => None,
            _ => None,
        });

        // Extract step from metadata
        let step = metadata_map.get("step").and_then(|v| match v {
            joule_db_amorphic::Value::Int(i) => Some(*i as u64),
            joule_db_amorphic::Value::Null => None,
            _ => None,
        });

        // Extract tags from metadata
        let tags = metadata_map
            .get("tags")
            .and_then(|v| match v {
                joule_db_amorphic::Value::Array(arr) => {
                    let strs: Vec<String> = arr
                        .iter()
                        .filter_map(|v| match v {
                            joule_db_amorphic::Value::String(s) => Some(s.clone()),
                            _ => None,
                        })
                        .collect();
                    Some(strs)
                }
                _ => None,
            })
            .unwrap_or_default();

        // Get the state object
        let state = record
            .get("state")
            .map(value_to_json)
            .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));

        // Build metadata
        let mut metadata = CheckpointMetadata::new(&thread_id, &checkpoint_id);
        if let Some(pid) = parent_id {
            metadata = metadata.with_parent(pid);
        }
        if let Some(s) = step {
            metadata = metadata.with_step(s);
        }
        if !tags.is_empty() {
            metadata = metadata.with_tags(tags);
        }

        Some(Checkpoint { metadata, state })
    }

    /// List all checkpoints for a thread
    pub fn list_checkpoints(&self, thread_id: &str) -> Vec<CheckpointMetadata> {
        let Some(checkpoint_ids) = self.thread_index.get(thread_id) else {
            return Vec::new();
        };

        checkpoint_ids
            .iter()
            .filter_map(|id| self.get_checkpoint_by_id(id))
            .map(|cp| cp.metadata)
            .collect()
    }

    /// Find checkpoints with similar state
    pub fn find_similar_checkpoints(&self, state: &Value, k: usize) -> Vec<Checkpoint> {
        // Convert state to a query string for similarity search
        let query = serde_json::to_string(state).unwrap_or_default();

        // Use AmorphicStore's similarity search
        let results = self.store.query_similar_to(&query, k);

        results
            .records()
            .iter()
            .filter_map(|r| self.reconstruct_checkpoint_from_record(r))
            .collect()
    }

    /// Delete a checkpoint
    pub fn delete_checkpoint(&mut self, checkpoint_id: &str) -> LangGraphResult<bool> {
        if let Some(record_id) = self.checkpoint_index.remove(checkpoint_id) {
            self.store.delete(record_id)?;

            // Remove from thread index
            for checkpoints in self.thread_index.values_mut() {
                checkpoints.retain(|id| id != checkpoint_id);
            }

            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Get checkpoint count for a thread
    pub fn checkpoint_count(&self, thread_id: &str) -> usize {
        self.thread_index
            .get(thread_id)
            .map(|v| v.len())
            .unwrap_or(0)
    }

    /// Get total checkpoint count
    pub fn total_checkpoints(&self) -> usize {
        self.checkpoint_index.len()
    }
}

impl Default for JouleCheckpointStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert AmorphicStore Value to serde_json Value
fn value_to_json(v: &joule_db_amorphic::Value) -> serde_json::Value {
    match v {
        joule_db_amorphic::Value::Null => serde_json::Value::Null,
        joule_db_amorphic::Value::Bool(b) => serde_json::Value::Bool(*b),
        joule_db_amorphic::Value::Int(i) => serde_json::Value::Number((*i).into()),
        joule_db_amorphic::Value::Float(f) => serde_json::Number::from_f64(*f)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        joule_db_amorphic::Value::String(s) => serde_json::Value::String(s.clone()),
        joule_db_amorphic::Value::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(value_to_json).collect())
        }
        joule_db_amorphic::Value::Object(obj) => serde_json::Value::Object(
            obj.iter()
                .map(|(k, v)| (k.clone(), value_to_json(v)))
                .collect(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_basic() {
        let mut store = JouleCheckpointStore::new();

        let state = serde_json::json!({
            "step": 1,
            "data": "hello world",
            "nested": {"key": "value"}
        });

        store.put_checkpoint("thread1", "cp1", &state).unwrap();

        let retrieved = store.get_checkpoint("thread1").unwrap();
        assert_eq!(retrieved.checkpoint_id(), "cp1");
        assert_eq!(retrieved.thread_id(), "thread1");
    }

    #[test]
    fn test_multiple_checkpoints() {
        let mut store = JouleCheckpointStore::new();

        store
            .put_checkpoint("thread1", "cp1", &serde_json::json!({"step": 1}))
            .unwrap();
        store
            .put_checkpoint("thread1", "cp2", &serde_json::json!({"step": 2}))
            .unwrap();
        store
            .put_checkpoint("thread1", "cp3", &serde_json::json!({"step": 3}))
            .unwrap();

        assert_eq!(store.checkpoint_count("thread1"), 3);

        // Latest should be cp3
        let latest = store.get_checkpoint("thread1").unwrap();
        assert_eq!(latest.checkpoint_id(), "cp3");

        // Can get specific checkpoint
        let cp1 = store.get_checkpoint_by_id("cp1").unwrap();
        assert_eq!(cp1.checkpoint_id(), "cp1");
    }

    #[test]
    fn test_list_checkpoints() {
        let mut store = JouleCheckpointStore::new();

        store
            .put_checkpoint("thread1", "cp1", &serde_json::json!({}))
            .unwrap();
        store
            .put_checkpoint("thread1", "cp2", &serde_json::json!({}))
            .unwrap();

        let list = store.list_checkpoints("thread1");
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn test_delete_checkpoint() {
        let mut store = JouleCheckpointStore::new();

        store
            .put_checkpoint("thread1", "cp1", &serde_json::json!({}))
            .unwrap();
        assert_eq!(store.total_checkpoints(), 1);

        store.delete_checkpoint("cp1").unwrap();
        assert_eq!(store.total_checkpoints(), 0);
    }

    #[test]
    fn test_metadata() {
        let mut store = JouleCheckpointStore::new();

        let metadata = CheckpointMetadata::new("thread1", "cp1")
            .with_parent("cp0")
            .with_step(5)
            .with_tags(vec!["important".to_string()]);

        store
            .put_checkpoint_with_metadata("thread1", "cp1", &serde_json::json!({}), Some(metadata))
            .unwrap();

        let retrieved = store.get_checkpoint("thread1").unwrap();
        assert_eq!(retrieved.metadata.step, Some(5));
    }

    #[test]
    fn test_multiple_threads() {
        let mut store = JouleCheckpointStore::new();

        store
            .put_checkpoint("thread1", "cp1", &serde_json::json!({"thread": 1}))
            .unwrap();
        store
            .put_checkpoint("thread2", "cp2", &serde_json::json!({"thread": 2}))
            .unwrap();

        assert_eq!(store.checkpoint_count("thread1"), 1);
        assert_eq!(store.checkpoint_count("thread2"), 1);
        assert_eq!(store.total_checkpoints(), 2);
    }
}
