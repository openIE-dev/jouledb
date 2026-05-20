//! Per-operation energy profiling using the Joule meter.
//!
//! Wraps energy readings to profile individual operations, tracking
//! per-operation energy consumption statistics over time.
//!
//! # Overview
//!
//! The [`JouleMeter`] maintains a map of [`OperationProfile`] entries, one
//! per unique operation name.  Every call to [`JouleMeter::record`] feeds a
//! new energy sample into the corresponding profile, updating the running
//! totals, call count, and min/max bounds.
//!
//! # Usage
//!
//! ```rust,ignore
//! use inv_energy::joule_meter::JouleMeter;
//!
//! let mut meter = JouleMeter::new();
//!
//! // Record energy for two different operations.
//! meter.record("sha256_hash", 0.012);
//! meter.record("aes_encrypt", 0.045);
//! meter.record("sha256_hash", 0.014);
//!
//! // Inspect per-operation statistics.
//! let hash = meter.get_profile("sha256_hash").unwrap();
//! println!("SHA-256 avg energy: {:.4} J", hash.avg_joules());
//!
//! // Find the most energy-intensive operations.
//! for profile in meter.top_consumers(3) {
//!     println!("{}: {:.4} J total", profile.op_name, profile.total_joules);
//! }
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// OperationProfile
// ---------------------------------------------------------------------------

/// Accumulated energy statistics for a single named operation.
///
/// An `OperationProfile` is created automatically by [`JouleMeter::record`]
/// the first time a given operation name is seen.  Subsequent recordings
/// under the same name update the running statistics.
///
/// All energy values are expressed in **joules**.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationProfile {
    /// Human-readable name of the operation being profiled.
    pub op_name: String,
    /// Cumulative energy consumed across all invocations (joules).
    pub total_joules: f64,
    /// Number of times this operation has been recorded.
    pub call_count: u64,
    /// Smallest single-invocation energy reading (joules).
    pub min_joules: f64,
    /// Largest single-invocation energy reading (joules).
    pub max_joules: f64,
}

impl OperationProfile {
    /// Returns the average energy per invocation.
    ///
    /// Computed as `total_joules / call_count`.
    ///
    /// If `call_count` is zero the result is `0.0` to avoid division by zero.
    pub fn avg_joules(&self) -> f64 {
        if self.call_count == 0 {
            0.0
        } else {
            self.total_joules / self.call_count as f64
        }
    }

    /// Returns the energy spread: `max_joules - min_joules`.
    ///
    /// A large spread indicates high variance between invocations.
    /// Returns `0.0` when no recordings have been made.
    pub fn spread(&self) -> f64 {
        if self.call_count == 0 {
            0.0
        } else {
            self.max_joules - self.min_joules
        }
    }
}

// ---------------------------------------------------------------------------
// JouleMeter
// ---------------------------------------------------------------------------

/// Collects per-operation energy profiles over time.
///
/// Each call to [`record`](JouleMeter::record) updates (or creates) the
/// [`OperationProfile`] for the given operation name, maintaining running
/// min/max/total statistics.
///
/// The meter is intentionally **not** thread-safe; wrap it in a `Mutex` or
/// `RwLock` if shared across tasks.
pub struct JouleMeter {
    profiles: HashMap<String, OperationProfile>,
}

impl JouleMeter {
    /// Creates a new, empty `JouleMeter`.
    pub fn new() -> Self {
        Self {
            profiles: HashMap::new(),
        }
    }

    /// Records an energy measurement for the named operation.
    ///
    /// If no profile exists for `op_name` yet, one is created.  Otherwise the
    /// existing profile is updated with the new reading.
    pub fn record(&mut self, op_name: &str, joules: f64) {
        let profile =
            self.profiles
                .entry(op_name.to_string())
                .or_insert_with(|| OperationProfile {
                    op_name: op_name.to_string(),
                    total_joules: 0.0,
                    call_count: 0,
                    min_joules: f64::MAX,
                    max_joules: f64::MIN,
                });

        profile.total_joules += joules;
        profile.call_count += 1;

        if joules < profile.min_joules {
            profile.min_joules = joules;
        }
        if joules > profile.max_joules {
            profile.max_joules = joules;
        }
    }

    /// Returns the profile for `op_name`, if one has been recorded.
    pub fn get_profile(&self, op_name: &str) -> Option<&OperationProfile> {
        self.profiles.get(op_name)
    }

    /// Returns references to all recorded profiles in arbitrary order.
    pub fn list_profiles(&self) -> Vec<&OperationProfile> {
        self.profiles.values().collect()
    }

    /// Returns the number of distinct operations that have been recorded.
    pub fn profile_count(&self) -> usize {
        self.profiles.len()
    }

    /// Sum of `total_joules` across every profile.
    ///
    /// This gives the aggregate energy consumption tracked by this meter
    /// since it was created (or last reset).
    pub fn total_energy(&self) -> f64 {
        self.profiles.values().map(|p| p.total_joules).sum()
    }

    /// Sum of `call_count` across every profile.
    ///
    /// Represents the total number of individual energy recordings made,
    /// regardless of which operation they belong to.
    pub fn total_operations(&self) -> u64 {
        self.profiles.values().map(|p| p.call_count).sum()
    }

    /// Removes all recorded profiles, resetting the meter to its initial
    /// empty state.
    pub fn reset(&mut self) {
        self.profiles.clear();
    }

    /// Returns the top `n` profiles sorted by `total_joules` descending.
    ///
    /// If fewer than `n` profiles exist, all profiles are returned.
    pub fn top_consumers(&self, n: usize) -> Vec<&OperationProfile> {
        let mut sorted: Vec<&OperationProfile> = self.profiles.values().collect();
        sorted.sort_by(|a, b| {
            b.total_joules
                .partial_cmp(&a.total_joules)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.truncate(n);
        sorted
    }
}

impl Default for JouleMeter {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_meter_empty() {
        let meter = JouleMeter::new();
        assert!(meter.list_profiles().is_empty());
        assert_eq!(meter.total_energy(), 0.0);
        assert_eq!(meter.total_operations(), 0);
    }

    #[test]
    fn record_single_operation() {
        let mut meter = JouleMeter::new();
        meter.record("hash", 1.5);

        let profile = meter.get_profile("hash").expect("profile should exist");
        assert_eq!(profile.op_name, "hash");
        assert!((profile.total_joules - 1.5).abs() < 1e-10);
        assert_eq!(profile.call_count, 1);
        assert!((profile.min_joules - 1.5).abs() < 1e-10);
        assert!((profile.max_joules - 1.5).abs() < 1e-10);
    }

    #[test]
    fn record_multiple_calls_same_op() {
        let mut meter = JouleMeter::new();
        meter.record("encrypt", 2.0);
        meter.record("encrypt", 3.0);
        meter.record("encrypt", 1.0);

        let profile = meter.get_profile("encrypt").unwrap();
        assert!((profile.total_joules - 6.0).abs() < 1e-10);
        assert_eq!(profile.call_count, 3);
    }

    #[test]
    fn avg_joules_calculation() {
        let mut meter = JouleMeter::new();
        meter.record("io", 4.0);
        meter.record("io", 6.0);

        let profile = meter.get_profile("io").unwrap();
        assert!((profile.avg_joules() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn min_max_tracking() {
        let mut meter = JouleMeter::new();
        meter.record("sort", 3.0);
        meter.record("sort", 1.0);
        meter.record("sort", 5.0);
        meter.record("sort", 2.0);

        let profile = meter.get_profile("sort").unwrap();
        assert!((profile.min_joules - 1.0).abs() < 1e-10);
        assert!((profile.max_joules - 5.0).abs() < 1e-10);
    }

    #[test]
    fn get_profile_not_found() {
        let meter = JouleMeter::new();
        assert!(meter.get_profile("nonexistent").is_none());
    }

    #[test]
    fn list_profiles() {
        let mut meter = JouleMeter::new();
        meter.record("a", 1.0);
        meter.record("b", 2.0);
        meter.record("c", 3.0);

        let profiles = meter.list_profiles();
        assert_eq!(profiles.len(), 3);
    }

    #[test]
    fn total_energy() {
        let mut meter = JouleMeter::new();
        meter.record("x", 10.0);
        meter.record("y", 20.0);
        meter.record("z", 30.0);

        assert!((meter.total_energy() - 60.0).abs() < 1e-10);
    }

    #[test]
    fn total_operations() {
        let mut meter = JouleMeter::new();
        meter.record("a", 1.0);
        meter.record("a", 2.0);
        meter.record("b", 3.0);

        assert_eq!(meter.total_operations(), 3);
    }

    #[test]
    fn reset_clears_all() {
        let mut meter = JouleMeter::new();
        meter.record("op", 5.0);
        assert!(!meter.list_profiles().is_empty());

        meter.reset();
        assert!(meter.list_profiles().is_empty());
        assert_eq!(meter.total_energy(), 0.0);
    }

    #[test]
    fn top_consumers() {
        let mut meter = JouleMeter::new();
        meter.record("low", 1.0);
        meter.record("mid", 5.0);
        meter.record("high", 10.0);

        let top = meter.top_consumers(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].op_name, "high");
        assert_eq!(top[1].op_name, "mid");
    }

    #[test]
    fn top_consumers_more_than_available() {
        let mut meter = JouleMeter::new();
        meter.record("only", 1.0);

        let top = meter.top_consumers(10);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].op_name, "only");
    }

    #[test]
    fn multiple_operations() {
        let mut meter = JouleMeter::new();
        meter.record("read", 2.0);
        meter.record("write", 3.0);
        meter.record("read", 4.0);

        assert_eq!(meter.list_profiles().len(), 2);
        assert!((meter.get_profile("read").unwrap().total_joules - 6.0).abs() < 1e-10);
        assert!((meter.get_profile("write").unwrap().total_joules - 3.0).abs() < 1e-10);
    }

    #[test]
    fn zero_joules_operation() {
        let mut meter = JouleMeter::new();
        meter.record("noop", 0.0);

        let profile = meter.get_profile("noop").unwrap();
        assert!((profile.total_joules).abs() < 1e-10);
        assert_eq!(profile.call_count, 1);
        assert!((profile.min_joules).abs() < 1e-10);
        assert!((profile.max_joules).abs() < 1e-10);
    }

    #[test]
    fn profile_serialization() {
        let mut meter = JouleMeter::new();
        meter.record("ser_test", 7.5);
        meter.record("ser_test", 2.5);

        let profile = meter.get_profile("ser_test").unwrap();
        let json = serde_json::to_string(profile).expect("serialization should succeed");
        let deserialized: OperationProfile =
            serde_json::from_str(&json).expect("deserialization should succeed");

        assert_eq!(deserialized.op_name, "ser_test");
        assert!((deserialized.total_joules - 10.0).abs() < 1e-10);
        assert_eq!(deserialized.call_count, 2);
        assert!((deserialized.min_joules - 2.5).abs() < 1e-10);
        assert!((deserialized.max_joules - 7.5).abs() < 1e-10);
    }

    #[test]
    fn meter_after_reset_is_empty() {
        let mut meter = JouleMeter::new();
        meter.record("alpha", 1.0);
        meter.record("beta", 2.0);
        meter.reset();

        assert!(meter.get_profile("alpha").is_none());
        assert!(meter.get_profile("beta").is_none());
        assert_eq!(meter.total_operations(), 0);
        assert!((meter.total_energy()).abs() < 1e-10);
        assert!(meter.list_profiles().is_empty());
    }
}
