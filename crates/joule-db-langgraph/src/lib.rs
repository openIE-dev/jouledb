//! # JouleDB LangGraph Integration
//!
//! This crate provides checkpoint and message stores for LangGraph, powered by
//! JouleDB's AmorphicStore. The unique advantage is **semantic search** across
//! conversation history - something no other checkpoint store can do.
//!
//! ## Features
//!
//! - **CheckpointStore** - Save and restore agent states with semantic versioning
//! - **MessageStore** - Store conversation history with semantic search
//! - **Semantic Search** - Find similar messages across threads using HDC
//! - **Durable** - Optional persistence with crash recovery
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use joule_db_langgraph::{JouleCheckpointStore, JouleMessageStore};
//!
//! // Create stores
//! let checkpoints = JouleCheckpointStore::new();
//! let messages = JouleMessageStore::new();
//!
//! // Save a checkpoint
//! let state = serde_json::json!({"step": 1, "data": "hello"});
//! checkpoints.put_checkpoint("thread1", "cp1", &state);
//!
//! // Add messages
//! messages.add_message("thread1", "user", "Hello, how are you?");
//! messages.add_message("thread1", "assistant", "I'm doing great, thanks!");
//!
//! // Semantic search across all threads!
//! let similar = messages.search_similar("greeting", 5);
//! ```

pub mod checkpoint;
pub mod error;
pub mod messages;

pub use checkpoint::{Checkpoint, CheckpointMetadata, JouleCheckpointStore};
pub use error::{LangGraphError, LangGraphResult};
pub use messages::{JouleMessageStore, Message, MessageRole};
