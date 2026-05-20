//! Temporal versioning for JouleDB
//!
//! Provides time-travel queries over any table by maintaining versioned rows.
//! Each row carries a validity interval `[valid_from, valid_to)` that tracks
//! when it was current. Historical versions are immutable; only the latest
//! version can be mutated.
//!
//! # Query syntax (SQL projection)
//!
//! ```sql
//! -- Point-in-time query
//! SELECT * FROM users AS OF '2025-06-01T00:00:00Z'
//!
//! -- Range query
//! SELECT * FROM users FOR SYSTEM_TIME BETWEEN '2025-01-01' AND '2025-06-01'
//! ```
//!
//! # Design
//!
//! - Each row in a temporal table stores `(valid_from, valid_to)` timestamps.
//! - `valid_to = i64::MAX` means the row is currently active.
//! - UPDATE creates a new version (closes old, opens new).
//! - DELETE closes the current version (sets valid_to = now).
//! - INSERT opens a new version (valid_from = now, valid_to = MAX).
//! - Historical rows are append-only and never modified.

use std::collections::HashMap;
use std::sync::RwLock;

/// Timestamp representation (microseconds since Unix epoch).
pub type Timestamp = i64;

/// Sentinel value meaning "row is currently active" (no end time).
pub const TIMESTAMP_MAX: Timestamp = i64::MAX;

/// A temporal validity interval [from, to).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Validity {
    /// Start of validity (inclusive).
    pub valid_from: Timestamp,
    /// End of validity (exclusive). `TIMESTAMP_MAX` = currently active.
    pub valid_to: Timestamp,
}

impl Validity {
    /// Create a new validity interval.
    pub fn new(from: Timestamp, to: Timestamp) -> Self {
        Self {
            valid_from: from,
            valid_to: to,
        }
    }

    /// Create a currently-active validity starting at `from`.
    pub fn active_from(from: Timestamp) -> Self {
        Self {
            valid_from: from,
            valid_to: TIMESTAMP_MAX,
        }
    }

    /// Whether the row is currently active (not closed).
    pub fn is_active(&self) -> bool {
        self.valid_to == TIMESTAMP_MAX
    }

    /// Whether the row was valid at a specific point in time.
    pub fn contains(&self, timestamp: Timestamp) -> bool {
        self.valid_from <= timestamp && timestamp < self.valid_to
    }

    /// Whether this interval overlaps with [start, end).
    pub fn overlaps(&self, start: Timestamp, end: Timestamp) -> bool {
        self.valid_from < end && start < self.valid_to
    }

    /// Close this validity at the given timestamp.
    pub fn close_at(&mut self, timestamp: Timestamp) {
        self.valid_to = timestamp;
    }
}

/// A temporal clause for queries (parsed from SQL AS OF / FOR SYSTEM_TIME).
#[derive(Debug, Clone, PartialEq)]
pub enum TemporalClause {
    /// Point-in-time: `AS OF <timestamp>`
    AsOf(Timestamp),
    /// Range: `FOR SYSTEM_TIME BETWEEN <start> AND <end>`
    Between(Timestamp, Timestamp),
    /// All versions: `FOR SYSTEM_TIME ALL`
    All,
}

/// A versioned row: the user data plus its validity interval.
#[derive(Debug, Clone)]
pub struct VersionedRow {
    /// Row identifier (primary key or surrogate).
    pub row_id: String,
    /// Validity interval.
    pub validity: Validity,
    /// The row data (column name → serialized value).
    pub data: HashMap<String, serde_json::Value>,
}

/// Temporal table: a versioned store for a single table's row history.
///
/// Maintains all versions of all rows, supporting point-in-time and range queries.
pub struct TemporalTable {
    /// Table name.
    pub name: String,
    /// All row versions, indexed by row_id.
    rows: RwLock<HashMap<String, Vec<VersionedRow>>>,
    /// GC retention: minimum age (in microseconds) before historical versions can be purged.
    pub retention_us: i64,
}

impl TemporalTable {
    /// Create a new temporal table.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            rows: RwLock::new(HashMap::new()),
            retention_us: 30 * 24 * 3600 * 1_000_000, // 30 days default
        }
    }

    /// Create with custom retention period.
    pub fn with_retention(name: &str, retention_us: i64) -> Self {
        Self {
            name: name.to_string(),
            rows: RwLock::new(HashMap::new()),
            retention_us,
        }
    }

    /// Insert a new row (opens a new version at `now`).
    pub fn insert(
        &self,
        row_id: &str,
        data: HashMap<String, serde_json::Value>,
        now: Timestamp,
    ) -> Result<(), String> {
        let mut rows = self
            .rows
            .write()
            .map_err(|_| "lock poisoned".to_string())?;

        let versions = rows.entry(row_id.to_string()).or_default();

        // Check for active version conflict.
        if versions.iter().any(|v| v.validity.is_active()) {
            return Err(format!(
                "row {} already has an active version — update or delete first",
                row_id
            ));
        }

        versions.push(VersionedRow {
            row_id: row_id.to_string(),
            validity: Validity::active_from(now),
            data,
        });

        Ok(())
    }

    /// Update a row: closes the current version and opens a new one.
    pub fn update(
        &self,
        row_id: &str,
        data: HashMap<String, serde_json::Value>,
        now: Timestamp,
    ) -> Result<(), String> {
        let mut rows = self
            .rows
            .write()
            .map_err(|_| "lock poisoned".to_string())?;

        let versions = rows
            .get_mut(row_id)
            .ok_or_else(|| format!("row {} not found", row_id))?;

        // Close the active version.
        let active = versions
            .iter_mut()
            .find(|v| v.validity.is_active())
            .ok_or_else(|| format!("row {} has no active version", row_id))?;

        active.validity.close_at(now);

        // Open new version.
        versions.push(VersionedRow {
            row_id: row_id.to_string(),
            validity: Validity::active_from(now),
            data,
        });

        Ok(())
    }

    /// Delete a row: closes the active version.
    pub fn delete(&self, row_id: &str, now: Timestamp) -> Result<(), String> {
        let mut rows = self
            .rows
            .write()
            .map_err(|_| "lock poisoned".to_string())?;

        let versions = rows
            .get_mut(row_id)
            .ok_or_else(|| format!("row {} not found", row_id))?;

        let active = versions
            .iter_mut()
            .find(|v| v.validity.is_active())
            .ok_or_else(|| format!("row {} has no active version", row_id))?;

        active.validity.close_at(now);
        Ok(())
    }

    /// Query current state (equivalent to `AS OF now`).
    pub fn scan_current(&self) -> Result<Vec<VersionedRow>, String> {
        let rows = self
            .rows
            .read()
            .map_err(|_| "lock poisoned".to_string())?;

        let mut result = Vec::new();
        for versions in rows.values() {
            for v in versions {
                if v.validity.is_active() {
                    result.push(v.clone());
                }
            }
        }
        Ok(result)
    }

    /// Point-in-time query: return rows as they existed at `timestamp`.
    pub fn scan_as_of(&self, timestamp: Timestamp) -> Result<Vec<VersionedRow>, String> {
        let rows = self
            .rows
            .read()
            .map_err(|_| "lock poisoned".to_string())?;

        let mut result = Vec::new();
        for versions in rows.values() {
            for v in versions {
                if v.validity.contains(timestamp) {
                    result.push(v.clone());
                }
            }
        }
        Ok(result)
    }

    /// Range query: return all versions overlapping [start, end).
    pub fn scan_between(
        &self,
        start: Timestamp,
        end: Timestamp,
    ) -> Result<Vec<VersionedRow>, String> {
        let rows = self
            .rows
            .read()
            .map_err(|_| "lock poisoned".to_string())?;

        let mut result = Vec::new();
        for versions in rows.values() {
            for v in versions {
                if v.validity.overlaps(start, end) {
                    result.push(v.clone());
                }
            }
        }
        Ok(result)
    }

    /// Return all versions (FOR SYSTEM_TIME ALL).
    pub fn scan_all_versions(&self) -> Result<Vec<VersionedRow>, String> {
        let rows = self
            .rows
            .read()
            .map_err(|_| "lock poisoned".to_string())?;

        let mut result = Vec::new();
        for versions in rows.values() {
            for v in versions {
                result.push(v.clone());
            }
        }
        Ok(result)
    }

    /// Execute a temporal query based on a TemporalClause.
    pub fn scan_temporal(
        &self,
        clause: &TemporalClause,
    ) -> Result<Vec<VersionedRow>, String> {
        match clause {
            TemporalClause::AsOf(ts) => self.scan_as_of(*ts),
            TemporalClause::Between(start, end) => self.scan_between(*start, *end),
            TemporalClause::All => self.scan_all_versions(),
        }
    }

    /// Garbage collect historical versions older than retention period.
    /// Returns the number of versions purged.
    pub fn gc(&self, now: Timestamp) -> Result<usize, String> {
        let mut rows = self
            .rows
            .write()
            .map_err(|_| "lock poisoned".to_string())?;

        let cutoff = now - self.retention_us;
        let mut purged = 0;

        for versions in rows.values_mut() {
            let before = versions.len();
            versions.retain(|v| {
                // Keep active versions always.
                // Keep historical versions newer than cutoff.
                v.validity.is_active() || v.validity.valid_to > cutoff
            });
            purged += before - versions.len();
        }

        // Remove empty entries.
        rows.retain(|_, v| !v.is_empty());

        Ok(purged)
    }

    /// Total number of row versions stored.
    pub fn version_count(&self) -> usize {
        self.rows
            .read()
            .map(|r| r.values().map(|v| v.len()).sum())
            .unwrap_or(0)
    }

    /// Number of currently active rows.
    pub fn active_count(&self) -> usize {
        self.rows
            .read()
            .map(|r| {
                r.values()
                    .filter(|versions| versions.iter().any(|v| v.validity.is_active()))
                    .count()
            })
            .unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// Timestamp utilities
// ---------------------------------------------------------------------------

/// Parse an ISO 8601 timestamp string to microseconds since epoch.
/// Supports: "2025-01-01", "2025-01-01T12:00:00", "2025-01-01T12:00:00Z"
pub fn parse_timestamp(s: &str) -> Result<Timestamp, String> {
    let s = s.trim().trim_matches('\'').trim_matches('"');

    // Try full ISO 8601 with time
    if s.contains('T') {
        let s = s.trim_end_matches('Z');
        let parts: Vec<&str> = s.split('T').collect();
        if parts.len() != 2 {
            return Err(format!("invalid timestamp: {}", s));
        }
        let date_us = parse_date(parts[0])?;
        let time_us = parse_time(parts[1])?;
        return Ok(date_us + time_us);
    }

    // Date only — midnight
    parse_date(s)
}

fn parse_date(s: &str) -> Result<Timestamp, String> {
    let parts: Vec<&str> = s.split('-').collect();
    if parts.len() != 3 {
        return Err(format!("invalid date: {}", s));
    }
    let year: i64 = parts[0].parse().map_err(|_| format!("invalid year: {}", parts[0]))?;
    let month: i64 = parts[1].parse().map_err(|_| format!("invalid month: {}", parts[1]))?;
    let day: i64 = parts[2].parse().map_err(|_| format!("invalid day: {}", parts[2]))?;

    // Simplified: days since epoch (not handling leap seconds, etc.)
    // Good enough for temporal versioning; real impl would use chrono.
    let days = (year - 1970) * 365 + (year - 1969) / 4 - (year - 1901) / 100 + (year - 1601) / 400;
    let month_days: [i64; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    let m = (month - 1).clamp(0, 11) as usize;
    let leap = if month > 2 && (year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)) {
        1
    } else {
        0
    };
    let total_days = days + month_days[m] + day - 1 + leap;
    Ok(total_days * 86_400 * 1_000_000) // microseconds
}

fn parse_time(s: &str) -> Result<Timestamp, String> {
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() < 2 {
        return Err(format!("invalid time: {}", s));
    }
    let hour: i64 = parts[0].parse().map_err(|_| format!("invalid hour: {}", parts[0]))?;
    let min: i64 = parts[1].parse().map_err(|_| format!("invalid minute: {}", parts[1]))?;
    let sec: i64 = if parts.len() > 2 {
        parts[2].parse().map_err(|_| format!("invalid second: {}", parts[2]))?
    } else {
        0
    };
    Ok((hour * 3600 + min * 60 + sec) * 1_000_000)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn now() -> Timestamp {
        1_000_000_000 // arbitrary fixed "now" for testing
    }

    fn make_data(pairs: &[(&str, &str)]) -> HashMap<String, serde_json::Value> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), serde_json::Value::String(v.to_string())))
            .collect()
    }

    #[test]
    fn test_validity() {
        let v = Validity::active_from(100);
        assert!(v.is_active());
        assert!(v.contains(100));
        assert!(v.contains(200));
        assert!(!v.contains(99));

        let mut v2 = Validity::active_from(100);
        v2.close_at(200);
        assert!(!v2.is_active());
        assert!(v2.contains(100));
        assert!(v2.contains(199));
        assert!(!v2.contains(200));
    }

    #[test]
    fn test_validity_overlaps() {
        let v = Validity::new(100, 200);
        assert!(v.overlaps(150, 250)); // partial overlap right
        assert!(v.overlaps(50, 150));  // partial overlap left
        assert!(v.overlaps(50, 250));  // contains
        assert!(v.overlaps(120, 180)); // contained
        assert!(!v.overlaps(200, 300)); // adjacent, no overlap
        assert!(!v.overlaps(0, 100));   // adjacent, no overlap
    }

    #[test]
    fn test_insert_and_current_scan() {
        let table = TemporalTable::new("users");
        table
            .insert("1", make_data(&[("name", "Alice")]), now())
            .unwrap();
        table
            .insert("2", make_data(&[("name", "Bob")]), now())
            .unwrap();

        let current = table.scan_current().unwrap();
        assert_eq!(current.len(), 2);
    }

    #[test]
    fn test_duplicate_insert_fails() {
        let table = TemporalTable::new("users");
        table
            .insert("1", make_data(&[("name", "Alice")]), now())
            .unwrap();
        let err = table
            .insert("1", make_data(&[("name", "Alice2")]), now())
            .unwrap_err();
        assert!(err.contains("already has an active version"));
    }

    #[test]
    fn test_update_creates_new_version() {
        let table = TemporalTable::new("users");
        let t0 = 1000;
        let t1 = 2000;

        table
            .insert("1", make_data(&[("name", "Alice")]), t0)
            .unwrap();
        table
            .update("1", make_data(&[("name", "Alice Updated")]), t1)
            .unwrap();

        // Current: should see updated version.
        let current = table.scan_current().unwrap();
        assert_eq!(current.len(), 1);
        assert_eq!(
            current[0].data.get("name").unwrap(),
            &serde_json::Value::String("Alice Updated".to_string())
        );

        // AS OF t0: should see original version.
        let historical = table.scan_as_of(t0).unwrap();
        assert_eq!(historical.len(), 1);
        assert_eq!(
            historical[0].data.get("name").unwrap(),
            &serde_json::Value::String("Alice".to_string())
        );

        // Total versions: 2
        assert_eq!(table.version_count(), 2);
    }

    #[test]
    fn test_delete_closes_version() {
        let table = TemporalTable::new("users");
        let t0 = 1000;
        let t1 = 2000;

        table
            .insert("1", make_data(&[("name", "Alice")]), t0)
            .unwrap();
        table.delete("1", t1).unwrap();

        // Current: should see nothing.
        let current = table.scan_current().unwrap();
        assert_eq!(current.len(), 0);

        // AS OF t0: should see Alice.
        let historical = table.scan_as_of(t0).unwrap();
        assert_eq!(historical.len(), 1);

        // AS OF t1: should see nothing (deleted at t1).
        let after = table.scan_as_of(t1).unwrap();
        assert_eq!(after.len(), 0);
    }

    #[test]
    fn test_time_travel_multiple_updates() {
        let table = TemporalTable::new("users");

        table
            .insert("1", make_data(&[("name", "V1")]), 100)
            .unwrap();
        table
            .update("1", make_data(&[("name", "V2")]), 200)
            .unwrap();
        table
            .update("1", make_data(&[("name", "V3")]), 300)
            .unwrap();

        // 3 versions
        assert_eq!(table.version_count(), 3);

        // AS OF 150: V1
        let r = table.scan_as_of(150).unwrap();
        assert_eq!(r[0].data["name"], "V1");

        // AS OF 250: V2
        let r = table.scan_as_of(250).unwrap();
        assert_eq!(r[0].data["name"], "V2");

        // AS OF 350: V3 (current)
        let r = table.scan_as_of(350).unwrap();
        assert_eq!(r[0].data["name"], "V3");
    }

    #[test]
    fn test_scan_between() {
        let table = TemporalTable::new("users");

        table
            .insert("1", make_data(&[("name", "V1")]), 100)
            .unwrap();
        table
            .update("1", make_data(&[("name", "V2")]), 200)
            .unwrap();
        table
            .update("1", make_data(&[("name", "V3")]), 300)
            .unwrap();

        // Between 150 and 250: should get V1 (ends at 200, overlaps) and V2 (starts at 200)
        let r = table.scan_between(150, 250).unwrap();
        assert_eq!(r.len(), 2);

        // Between 0 and 400: should get all 3 versions
        let r = table.scan_between(0, 400).unwrap();
        assert_eq!(r.len(), 3);
    }

    #[test]
    fn test_scan_all_versions() {
        let table = TemporalTable::new("users");

        table
            .insert("1", make_data(&[("name", "V1")]), 100)
            .unwrap();
        table
            .update("1", make_data(&[("name", "V2")]), 200)
            .unwrap();

        let r = table.scan_all_versions().unwrap();
        assert_eq!(r.len(), 2);
    }

    #[test]
    fn test_temporal_clause_dispatch() {
        let table = TemporalTable::new("users");
        table
            .insert("1", make_data(&[("name", "Alice")]), 100)
            .unwrap();
        table
            .update("1", make_data(&[("name", "Bob")]), 200)
            .unwrap();

        // AsOf
        let r = table
            .scan_temporal(&TemporalClause::AsOf(150))
            .unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].data["name"], "Alice");

        // Between
        let r = table
            .scan_temporal(&TemporalClause::Between(0, 300))
            .unwrap();
        assert_eq!(r.len(), 2);

        // All
        let r = table.scan_temporal(&TemporalClause::All).unwrap();
        assert_eq!(r.len(), 2);
    }

    #[test]
    fn test_gc() {
        let table = TemporalTable::with_retention("users", 1000);

        table
            .insert("1", make_data(&[("name", "V1")]), 100)
            .unwrap();
        table
            .update("1", make_data(&[("name", "V2")]), 200)
            .unwrap();
        table
            .update("1", make_data(&[("name", "V3")]), 300)
            .unwrap();

        assert_eq!(table.version_count(), 3);

        // GC at 1500: retention=1000, cutoff=500. V1 closed at 200 < 500 → purged.
        // V2 closed at 300 < 500 → purged. V3 is active → kept.
        let purged = table.gc(1500).unwrap();
        assert_eq!(purged, 2);
        assert_eq!(table.version_count(), 1);

        // Active row should still be there.
        let current = table.scan_current().unwrap();
        assert_eq!(current.len(), 1);
        assert_eq!(current[0].data["name"], "V3");
    }

    #[test]
    fn test_active_count() {
        let table = TemporalTable::new("users");
        table
            .insert("1", make_data(&[("name", "Alice")]), 100)
            .unwrap();
        table
            .insert("2", make_data(&[("name", "Bob")]), 100)
            .unwrap();
        assert_eq!(table.active_count(), 2);

        table.delete("1", 200).unwrap();
        assert_eq!(table.active_count(), 1);
    }

    #[test]
    fn test_parse_timestamp_date_only() {
        let ts = parse_timestamp("2025-01-01").unwrap();
        assert!(ts > 0);
    }

    #[test]
    fn test_parse_timestamp_datetime() {
        let ts = parse_timestamp("2025-06-15T12:30:00Z").unwrap();
        assert!(ts > 0);
        // Should be greater than date-only (which is midnight)
        let ts_date = parse_timestamp("2025-06-15").unwrap();
        assert!(ts > ts_date);
    }

    #[test]
    fn test_parse_timestamp_quoted() {
        let ts1 = parse_timestamp("'2025-01-01'").unwrap();
        let ts2 = parse_timestamp("2025-01-01").unwrap();
        assert_eq!(ts1, ts2);
    }
}
