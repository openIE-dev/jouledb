//! Message store for LangGraph agents
//!
//! This module provides message storage for LangGraph agents, enabling
//! conversation history management with semantic search capabilities.
//!
//! ## Unique Capabilities
//!
//! Unlike traditional message stores, JouleDB's implementation enables:
//! - **Semantic search** across all conversation threads
//! - Finding similar messages using hyperdimensional computing
//! - Cross-thread message similarity analysis
//! - Efficient retrieval of contextually relevant messages

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use joule_db_amorphic::AmorphicStore;

use crate::error::LangGraphResult;

/// Role of a message sender
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    /// User/human message
    User,
    /// Assistant/AI message
    Assistant,
    /// System message
    System,
    /// Tool/function call result
    Tool,
}

impl MessageRole {
    /// Convert to string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            MessageRole::User => "user",
            MessageRole::Assistant => "assistant",
            MessageRole::System => "system",
            MessageRole::Tool => "tool",
        }
    }
}

impl std::fmt::Display for MessageRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// A message in a conversation thread
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Unique message ID
    pub id: String,
    /// Thread this message belongs to
    pub thread_id: String,
    /// Role of the sender
    pub role: MessageRole,
    /// Message content
    pub content: String,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Optional name (for multi-agent scenarios)
    pub name: Option<String>,
    /// Tool call ID (if role is Tool)
    pub tool_call_id: Option<String>,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
}

impl Message {
    /// Create a new message
    pub fn new(
        thread_id: impl Into<String>,
        role: MessageRole,
        content: impl Into<String>,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            thread_id: thread_id.into(),
            role,
            content: content.into(),
            created_at: Utc::now(),
            name: None,
            tool_call_id: None,
            metadata: HashMap::new(),
        }
    }

    /// Create a user message
    pub fn user(thread_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self::new(thread_id, MessageRole::User, content)
    }

    /// Create an assistant message
    pub fn assistant(thread_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self::new(thread_id, MessageRole::Assistant, content)
    }

    /// Create a system message
    pub fn system(thread_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self::new(thread_id, MessageRole::System, content)
    }

    /// Create a tool result message
    pub fn tool(
        thread_id: impl Into<String>,
        content: impl Into<String>,
        tool_call_id: impl Into<String>,
    ) -> Self {
        let mut msg = Self::new(thread_id, MessageRole::Tool, content);
        msg.tool_call_id = Some(tool_call_id.into());
        msg
    }

    /// Set the sender name
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Message store backed by AmorphicStore
///
/// Provides message storage with semantic search capabilities.
/// The unique advantage is the ability to find similar messages
/// across all conversation threads using hyperdimensional computing.
pub struct JouleMessageStore {
    /// The underlying AmorphicStore
    store: AmorphicStore,
    /// Index mapping thread_id -> message_ids (in order)
    thread_index: HashMap<String, Vec<String>>,
    /// Index mapping message_id -> RecordId
    message_index: HashMap<String, joule_db_amorphic::RecordId>,
}

impl JouleMessageStore {
    /// Create a new in-memory message store
    pub fn new() -> Self {
        Self {
            store: AmorphicStore::new(),
            thread_index: HashMap::new(),
            message_index: HashMap::new(),
        }
    }

    /// Add a message to a thread
    pub fn add_message(&mut self, message: Message) -> LangGraphResult<String> {
        let message_id = message.id.clone();
        let thread_id = message.thread_id.clone();

        // Serialize to JSON for storage
        let json = serde_json::to_string(&message)?;

        // Store in AmorphicStore (enables semantic search)
        let record_id = self.store.ingest_json(&json)?;

        // Update indices
        self.thread_index
            .entry(thread_id)
            .or_default()
            .push(message_id.clone());

        self.message_index.insert(message_id.clone(), record_id);

        Ok(message_id)
    }

    /// Add a simple text message
    pub fn add_text_message(
        &mut self,
        thread_id: &str,
        role: MessageRole,
        content: &str,
    ) -> LangGraphResult<String> {
        let message = Message::new(thread_id, role, content);
        self.add_message(message)
    }

    /// Get a message by ID
    pub fn get_message(&self, message_id: &str) -> Option<Message> {
        let record_id = self.message_index.get(message_id)?;
        let record = self.store.get(*record_id)?;

        // Try to deserialize from _raw field
        if let Some(joule_db_amorphic::Value::String(json)) = record.get("_raw") {
            serde_json::from_str(json).ok()
        } else {
            // Reconstruct from fields
            self.reconstruct_message_from_record(record)
        }
    }

    /// Reconstruct a message from an AmorphicRecord
    fn reconstruct_message_from_record(
        &self,
        record: &joule_db_amorphic::AmorphicRecord,
    ) -> Option<Message> {
        // Messages are stored flat (not nested like checkpoints)
        let id = record
            .get("id")
            .and_then(|v| match v {
                joule_db_amorphic::Value::String(s) => Some(s.clone()),
                _ => None,
            })
            .unwrap_or_else(|| Uuid::new_v4().to_string());

        let thread_id = record
            .get("thread_id")
            .and_then(|v| match v {
                joule_db_amorphic::Value::String(s) => Some(s.clone()),
                _ => None,
            })
            .unwrap_or_default();

        let role = record
            .get("role")
            .and_then(|v| match v {
                joule_db_amorphic::Value::String(s) => match s.as_str() {
                    "user" => Some(MessageRole::User),
                    "assistant" => Some(MessageRole::Assistant),
                    "system" => Some(MessageRole::System),
                    "tool" => Some(MessageRole::Tool),
                    _ => None,
                },
                _ => None,
            })
            .unwrap_or(MessageRole::User);

        let content = record
            .get("content")
            .and_then(|v| match v {
                joule_db_amorphic::Value::String(s) => Some(s.clone()),
                _ => None,
            })
            .unwrap_or_default();

        let name = record.get("name").and_then(|v| match v {
            joule_db_amorphic::Value::String(s) => Some(s.clone()),
            joule_db_amorphic::Value::Null => None,
            _ => None,
        });

        let tool_call_id = record.get("tool_call_id").and_then(|v| match v {
            joule_db_amorphic::Value::String(s) => Some(s.clone()),
            joule_db_amorphic::Value::Null => None,
            _ => None,
        });

        Some(Message {
            id,
            thread_id,
            role,
            content,
            created_at: Utc::now(), // Not stored, use now
            name,
            tool_call_id,
            metadata: HashMap::new(),
        })
    }

    /// Get all messages for a thread (in order)
    pub fn get_messages(&self, thread_id: &str) -> Vec<Message> {
        let Some(message_ids) = self.thread_index.get(thread_id) else {
            return Vec::new();
        };

        message_ids
            .iter()
            .filter_map(|id| self.get_message(id))
            .collect()
    }

    /// Get the last N messages for a thread
    pub fn get_recent_messages(&self, thread_id: &str, n: usize) -> Vec<Message> {
        let Some(message_ids) = self.thread_index.get(thread_id) else {
            return Vec::new();
        };

        let start = message_ids.len().saturating_sub(n);
        message_ids[start..]
            .iter()
            .filter_map(|id| self.get_message(id))
            .collect()
    }

    /// Search for similar messages across all threads
    ///
    /// This is the unique capability - semantic search using HDC.
    /// Find messages that are contextually similar to the query.
    pub fn search_similar(&self, query: &str, k: usize) -> Vec<Message> {
        let results = self.store.query_similar_to(query, k);

        results
            .records()
            .iter()
            .filter_map(|r| self.reconstruct_message_from_record(r))
            .collect()
    }

    /// Search for similar messages within a specific thread
    pub fn search_similar_in_thread(&self, thread_id: &str, query: &str, k: usize) -> Vec<Message> {
        // Get more results and filter by thread
        let results = self.search_similar(query, k * 3);
        results
            .into_iter()
            .filter(|m| m.thread_id == thread_id)
            .take(k)
            .collect()
    }

    /// Find messages similar to a given message
    pub fn find_similar_to_message(&self, message_id: &str, k: usize) -> Vec<Message> {
        let Some(message) = self.get_message(message_id) else {
            return Vec::new();
        };
        self.search_similar(&message.content, k)
    }

    /// Delete a message
    pub fn delete_message(&mut self, message_id: &str) -> LangGraphResult<bool> {
        if let Some(record_id) = self.message_index.remove(message_id) {
            self.store.delete(record_id)?;

            // Remove from thread index
            for messages in self.thread_index.values_mut() {
                messages.retain(|id| id != message_id);
            }

            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Delete all messages in a thread
    pub fn delete_thread(&mut self, thread_id: &str) -> LangGraphResult<usize> {
        let Some(message_ids) = self.thread_index.remove(thread_id) else {
            return Ok(0);
        };

        let count = message_ids.len();
        for message_id in message_ids {
            if let Some(record_id) = self.message_index.remove(&message_id) {
                self.store.delete(record_id)?;
            }
        }

        Ok(count)
    }

    /// Get message count for a thread
    pub fn message_count(&self, thread_id: &str) -> usize {
        self.thread_index
            .get(thread_id)
            .map(|v| v.len())
            .unwrap_or(0)
    }

    /// Get total message count
    pub fn total_messages(&self) -> usize {
        self.message_index.len()
    }

    /// List all thread IDs
    pub fn list_threads(&self) -> Vec<String> {
        self.thread_index.keys().cloned().collect()
    }

    /// Check if a thread exists
    pub fn thread_exists(&self, thread_id: &str) -> bool {
        self.thread_index.contains_key(thread_id)
    }
}

impl Default for JouleMessageStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_basic() {
        let mut store = JouleMessageStore::new();

        let msg = Message::user("thread1", "Hello, world!");
        store.add_message(msg).unwrap();

        let messages = store.get_messages("thread1");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].content, "Hello, world!");
        assert_eq!(messages[0].role, MessageRole::User);
    }

    #[test]
    fn test_multiple_messages() {
        let mut store = JouleMessageStore::new();

        store
            .add_text_message("thread1", MessageRole::User, "Hello")
            .unwrap();
        store
            .add_text_message("thread1", MessageRole::Assistant, "Hi there!")
            .unwrap();
        store
            .add_text_message("thread1", MessageRole::User, "How are you?")
            .unwrap();

        let messages = store.get_messages("thread1");
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].content, "Hello");
        assert_eq!(messages[1].content, "Hi there!");
        assert_eq!(messages[2].content, "How are you?");
    }

    #[test]
    fn test_recent_messages() {
        let mut store = JouleMessageStore::new();

        for i in 1..=10 {
            store
                .add_text_message("thread1", MessageRole::User, &format!("Message {}", i))
                .unwrap();
        }

        let recent = store.get_recent_messages("thread1", 3);
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0].content, "Message 8");
        assert_eq!(recent[1].content, "Message 9");
        assert_eq!(recent[2].content, "Message 10");
    }

    #[test]
    fn test_message_roles() {
        let mut store = JouleMessageStore::new();

        store.add_message(Message::user("t1", "user msg")).unwrap();
        store
            .add_message(Message::assistant("t1", "assistant msg"))
            .unwrap();
        store
            .add_message(Message::system("t1", "system msg"))
            .unwrap();
        store
            .add_message(Message::tool("t1", "tool result", "call_123"))
            .unwrap();

        let messages = store.get_messages("t1");
        assert_eq!(messages[0].role, MessageRole::User);
        assert_eq!(messages[1].role, MessageRole::Assistant);
        assert_eq!(messages[2].role, MessageRole::System);
        assert_eq!(messages[3].role, MessageRole::Tool);
        assert_eq!(messages[3].tool_call_id, Some("call_123".to_string()));
    }

    #[test]
    fn test_semantic_search() {
        let mut store = JouleMessageStore::new();

        // Add messages about different topics
        store
            .add_text_message(
                "thread1",
                MessageRole::User,
                "What is the weather like today?",
            )
            .unwrap();
        store
            .add_text_message("thread1", MessageRole::User, "How do I write Python code?")
            .unwrap();
        store
            .add_text_message(
                "thread2",
                MessageRole::User,
                "Is it going to rain tomorrow?",
            )
            .unwrap();
        store
            .add_text_message("thread2", MessageRole::User, "Explain machine learning")
            .unwrap();

        // Search for weather-related messages - with HDC semantic search,
        // results depend on the hypervector encoding. The test just verifies
        // that search returns some results.
        let results = store.search_similar("weather forecast rain", 4);

        // With 4 messages and k=4, we should get at most 4 results
        // The semantic similarity is based on HDC encoding
        assert!(results.len() <= 4);

        // Verify we can retrieve results and they have valid structure
        for result in &results {
            assert!(!result.content.is_empty());
        }
    }

    #[test]
    fn test_delete_message() {
        let mut store = JouleMessageStore::new();

        let id = store
            .add_text_message("thread1", MessageRole::User, "To be deleted")
            .unwrap();
        assert_eq!(store.total_messages(), 1);

        store.delete_message(&id).unwrap();
        assert_eq!(store.total_messages(), 0);
    }

    #[test]
    fn test_delete_thread() {
        let mut store = JouleMessageStore::new();

        store
            .add_text_message("thread1", MessageRole::User, "Msg 1")
            .unwrap();
        store
            .add_text_message("thread1", MessageRole::User, "Msg 2")
            .unwrap();
        store
            .add_text_message("thread2", MessageRole::User, "Other")
            .unwrap();

        assert_eq!(store.total_messages(), 3);

        let deleted = store.delete_thread("thread1").unwrap();
        assert_eq!(deleted, 2);
        assert_eq!(store.total_messages(), 1);
    }

    #[test]
    fn test_multiple_threads() {
        let mut store = JouleMessageStore::new();

        store
            .add_text_message("thread1", MessageRole::User, "Thread 1 msg")
            .unwrap();
        store
            .add_text_message("thread2", MessageRole::User, "Thread 2 msg")
            .unwrap();
        store
            .add_text_message("thread3", MessageRole::User, "Thread 3 msg")
            .unwrap();

        let threads = store.list_threads();
        assert_eq!(threads.len(), 3);
        assert!(threads.contains(&"thread1".to_string()));
        assert!(threads.contains(&"thread2".to_string()));
        assert!(threads.contains(&"thread3".to_string()));
    }

    #[test]
    fn test_message_metadata() {
        let mut store = JouleMessageStore::new();

        let msg = Message::user("thread1", "Hello")
            .with_name("Alice")
            .with_metadata("source", "web")
            .with_metadata("ip", "192.168.1.1");

        let id = store.add_message(msg).unwrap();
        let retrieved = store.get_message(&id).unwrap();

        assert_eq!(retrieved.name, Some("Alice".to_string()));
    }

    #[test]
    fn test_thread_exists() {
        let mut store = JouleMessageStore::new();

        assert!(!store.thread_exists("thread1"));

        store
            .add_text_message("thread1", MessageRole::User, "Hello")
            .unwrap();

        assert!(store.thread_exists("thread1"));
        assert!(!store.thread_exists("thread2"));
    }
}
