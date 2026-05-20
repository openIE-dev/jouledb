//! Temporal Field Validity — time-bounded field values for rights/licensing.
//!
//! "Can we stream this title in France on March 31, 2026?" → single query.
//!
//! This is an overlay on AmorphicStore, not a modification of the core Value enum.
//! Fields can have multiple temporal versions with territory restrictions.

use crate::{RecordId, Value};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A field value with temporal and territorial validity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalField {
    /// The field value
    pub value: Value,
    /// Valid from (unix milliseconds, 0 = beginning of time)
    pub valid_from: u64,
    /// Valid until (unix milliseconds, u64::MAX = forever)
    pub valid_to: u64,
    /// Territory restrictions (empty = global)
    pub territories: Vec<String>,
}

impl TemporalField {
    /// Create a globally valid field (no time or territory restriction).
    pub fn global(value: Value) -> Self {
        Self {
            value,
            valid_from: 0,
            valid_to: u64::MAX,
            territories: vec![],
        }
    }

    /// Create a time-bounded field.
    pub fn bounded(value: Value, valid_from: u64, valid_to: u64) -> Self {
        Self {
            value,
            valid_from,
            valid_to,
            territories: vec![],
        }
    }

    /// Create a territory-restricted field.
    pub fn territorial(value: Value, valid_from: u64, valid_to: u64, territories: Vec<String>) -> Self {
        Self {
            value,
            valid_from,
            valid_to,
            territories,
        }
    }

    /// Check if this field is valid at the given time and territory.
    pub fn is_valid_at(&self, timestamp: u64, territory: Option<&str>) -> bool {
        let time_valid = timestamp >= self.valid_from && timestamp < self.valid_to;
        let territory_valid = self.territories.is_empty()
            || territory
                .map(|t| self.territories.iter().any(|tt| tt == t))
                .unwrap_or(true);
        time_valid && territory_valid
    }
}

/// Temporal field store — overlay on AmorphicStore for time-bounded fields.
///
/// Keyed by (record_id, field_name), stores multiple temporal versions.
#[derive(Debug, Default)]
pub struct TemporalStore {
    pub(crate) fields: HashMap<(RecordId, String), Vec<TemporalField>>,
}

impl TemporalStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a temporal field value.
    pub fn set(
        &mut self,
        record_id: RecordId,
        field: &str,
        temporal: TemporalField,
    ) {
        self.fields
            .entry((record_id, field.to_string()))
            .or_default()
            .push(temporal);
    }

    /// Query the valid value at a specific time and territory.
    /// Returns the most recently added matching value.
    pub fn query_valid_at(
        &self,
        record_id: RecordId,
        field: &str,
        timestamp: u64,
        territory: Option<&str>,
    ) -> Option<&Value> {
        self.fields
            .get(&(record_id, field.to_string()))
            .and_then(|versions| {
                versions
                    .iter()
                    .rev() // Most recent first
                    .find(|v| v.is_valid_at(timestamp, territory))
                    .map(|v| &v.value)
            })
    }

    /// Check if a content item can be streamed in a territory at a given time.
    /// Convenience wrapper for rights management.
    pub fn can_stream(
        &self,
        content_id: RecordId,
        territory: &str,
        at: u64,
    ) -> bool {
        self.query_valid_at(content_id, "streaming_rights", at, Some(territory))
            .map(|v| match v {
                Value::Bool(b) => *b,
                Value::String(s) => s == "allowed" || s == "true",
                _ => false,
            })
            .unwrap_or(false)
    }

    /// Get all active licenses for a content item.
    pub fn active_licenses(
        &self,
        content_id: RecordId,
        at: u64,
    ) -> Vec<(&str, &TemporalField)> {
        self.fields
            .iter()
            .filter(|((rid, _), _)| *rid == content_id)
            .flat_map(|((_, field), versions)| {
                versions
                    .iter()
                    .filter(|v| v.is_valid_at(at, None))
                    .map(move |v| (field.as_str(), v))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_validity() {
        let mut store = TemporalStore::new();

        // License valid Jan 2025 - Dec 2026, US and UK only
        store.set(
            1,
            "streaming_rights",
            TemporalField::territorial(
                Value::Bool(true),
                1704067200000, // 2024-01-01
                1767225600000, // 2026-01-01
                vec!["US".into(), "UK".into()],
            ),
        );

        // Valid in US in 2025
        assert!(store.can_stream(1, "US", 1735689600000)); // 2025-01-01
        // Valid in UK
        assert!(store.can_stream(1, "UK", 1735689600000));
        // NOT valid in France
        assert!(!store.can_stream(1, "FR", 1735689600000));
        // NOT valid after expiry
        assert!(!store.can_stream(1, "US", 1800000000000)); // 2027
    }

    #[test]
    fn test_multiple_versions() {
        let mut store = TemporalStore::new();

        // Price changes over time
        store.set(
            1,
            "price",
            TemporalField::bounded(Value::Float(9.99), 0, 1735689600000),
        );
        store.set(
            1,
            "price",
            TemporalField::bounded(Value::Float(12.99), 1735689600000, u64::MAX),
        );

        // Before change
        let price = store.query_valid_at(1, "price", 1700000000000, None);
        assert_eq!(price, Some(&Value::Float(9.99)));

        // After change
        let price = store.query_valid_at(1, "price", 1740000000000, None);
        assert_eq!(price, Some(&Value::Float(12.99)));
    }

    #[test]
    fn test_active_licenses() {
        let mut store = TemporalStore::new();
        let now = 1735689600000u64; // 2025-01-01

        store.set(1, "streaming_rights", TemporalField::bounded(Value::Bool(true), 0, u64::MAX));
        store.set(1, "download_rights", TemporalField::bounded(Value::Bool(false), 0, u64::MAX));

        let licenses = store.active_licenses(1, now);
        assert_eq!(licenses.len(), 2);
    }
}
