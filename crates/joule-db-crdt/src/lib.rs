//! Conflict-Free Replicated Data Types (CRDTs) for JouleDB
//!
//! Provides mathematically-guaranteed merge semantics for distributed
//! and disconnected operation — critical for DoD edge deployments where
//! nodes may operate without connectivity for hours or days.
//!
//! ## CRDT Types
//!
//! | Type | Description | Use Case |
//! |------|-------------|----------|
//! | `GCounter` | Grow-only counter | Upvotes, event counts |
//! | `PNCounter` | Positive-negative counter | Inventory, bidirectional |
//! | `LWWRegister` | Last-writer-wins register | Single-value cells |
//! | `MVRegister` | Multi-value register | Concurrent edits preserved |
//! | `ORSet` | Observed-remove set | Tag sets, memberships |
//! | `LWWMap` | LWW register per key | JSON document merge |
//! | `RGA` | Replicated growable array | Ordered lists, text |
//!
//! ## Guarantees
//!
//! All types satisfy:
//! - **Commutativity**: merge(a, b) == merge(b, a)
//! - **Associativity**: merge(merge(a, b), c) == merge(a, merge(b, c))
//! - **Idempotency**: merge(a, a) == a
//!
//! These properties ensure convergence regardless of message ordering,
//! duplication, or network partition duration.

pub mod types;

use serde::{Deserialize, Serialize};
use thiserror::Error;

// Re-export all CRDT types
pub use types::{GCounter, LWWMap, LWWRegister, MVRegister, ORSet, PNCounter};

// ============================================================================
// Errors
// ============================================================================

#[derive(Debug, Error)]
pub enum CrdtError {
    #[error("unknown CRDT type: {0}")]
    UnknownType(String),

    #[error("type mismatch: cannot merge {0} with {1}")]
    TypeMismatch(String, String),

    #[error("serialization error: {0}")]
    Serialization(String),
}

// ============================================================================
// Core CRDT trait
// ============================================================================

/// The fundamental trait all CRDTs implement.
///
/// `merge` must be commutative, associative, and idempotent.
pub trait Crdt: Clone + Send + Sync {
    /// Merge another replica's state into this one.
    /// After merge, both replicas will converge to the same state.
    fn merge(&mut self, other: &Self);

    /// Return the CRDT type name (for serialization/dispatch)
    fn crdt_type(&self) -> &'static str;
}

// ============================================================================
// CRDT type tag (for serialized storage)
// ============================================================================

/// Tag identifying which CRDT type is stored in a column
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CrdtType {
    GCounter,
    PNCounter,
    LWWRegister,
    MVRegister,
    ORSet,
    LWWMap,
}

impl CrdtType {
    pub fn from_str(s: &str) -> Result<Self, CrdtError> {
        match s.to_lowercase().as_str() {
            "gcounter" | "g_counter" => Ok(Self::GCounter),
            "pncounter" | "pn_counter" => Ok(Self::PNCounter),
            "lwwregister" | "lww_register" | "lww" => Ok(Self::LWWRegister),
            "mvregister" | "mv_register" | "mv" => Ok(Self::MVRegister),
            "orset" | "or_set" => Ok(Self::ORSet),
            "lwwmap" | "lww_map" => Ok(Self::LWWMap),
            _ => Err(CrdtError::UnknownType(s.to_string())),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::GCounter => "gcounter",
            Self::PNCounter => "pncounter",
            Self::LWWRegister => "lww_register",
            Self::MVRegister => "mv_register",
            Self::ORSet => "orset",
            Self::LWWMap => "lww_map",
        }
    }
}

// ============================================================================
// Timestamp source (for LWW types)
// ============================================================================

/// A hybrid logical clock timestamp for LWW ordering.
/// Combines wall-clock time with a logical counter to break ties.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct HLCTimestamp {
    /// Wall-clock milliseconds since Unix epoch
    pub millis: u64,
    /// Logical counter (breaks ties when wall clocks are equal)
    pub counter: u32,
    /// Node ID hash (breaks ties when counter is also equal)
    pub node_hash: u32,
}

impl HLCTimestamp {
    /// Create a new timestamp for the current instant on a given node
    pub fn now(node_id: &str) -> Self {
        let millis = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Self {
            millis,
            counter: 0,
            node_hash: simple_hash(node_id),
        }
    }

    /// Advance this clock, ensuring it's always greater than both the local
    /// clock and a received remote timestamp.
    pub fn tick(&mut self, received: Option<&HLCTimestamp>) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        match received {
            Some(remote) => {
                if now > self.millis && now > remote.millis {
                    self.millis = now;
                    self.counter = 0;
                } else if self.millis == remote.millis {
                    self.counter = self.counter.max(remote.counter) + 1;
                } else if self.millis > remote.millis {
                    self.counter += 1;
                } else {
                    self.millis = remote.millis;
                    self.counter = remote.counter + 1;
                }
            }
            None => {
                if now > self.millis {
                    self.millis = now;
                    self.counter = 0;
                } else {
                    self.counter += 1;
                }
            }
        }
    }
}

fn simple_hash(s: &str) -> u32 {
    let mut h: u32 = 0;
    for b in s.bytes() {
        h = h.wrapping_mul(31).wrapping_add(b as u32);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crdt_type_parsing() {
        assert_eq!(CrdtType::from_str("lww").unwrap(), CrdtType::LWWRegister);
        assert_eq!(CrdtType::from_str("orset").unwrap(), CrdtType::ORSet);
        assert_eq!(CrdtType::from_str("gcounter").unwrap(), CrdtType::GCounter);
        assert!(CrdtType::from_str("unknown").is_err());
    }

    #[test]
    fn test_hlc_ordering() {
        let a = HLCTimestamp {
            millis: 100,
            counter: 0,
            node_hash: 1,
        };
        let b = HLCTimestamp {
            millis: 100,
            counter: 1,
            node_hash: 1,
        };
        let c = HLCTimestamp {
            millis: 101,
            counter: 0,
            node_hash: 1,
        };

        assert!(a < b);
        assert!(b < c);
        assert!(a < c);
    }

    #[test]
    fn test_hlc_tick() {
        let mut clock = HLCTimestamp::now("node1");
        let initial = clock.millis;

        clock.tick(None);
        // Should be >= initial (wall clock may advance or counter increments)
        assert!(clock.millis >= initial);
    }
}
