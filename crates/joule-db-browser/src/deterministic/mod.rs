//! Deterministic Execution Module
//!
//! Browser-native deterministic execution for multi-tab coordination

pub mod coordination;
pub mod replication;
pub mod transaction;

pub use coordination::{CoordinationStrategy, MultiTabCoordinator};
pub use replication::{ReplicationConfig, ServiceWorkerReplication};
pub use transaction::{DeterministicTransaction, DeterministicTransactionManager};
