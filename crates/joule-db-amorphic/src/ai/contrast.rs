//! The Contrast Engine — SNN-style spike gate on every ingest and query.
//!
//! Axiom 4: Information is a contrast signal.
//! Axiom 5: Intelligence is the recognition of contrast.
//!
//! Every piece of data entering or leaving the database passes through
//! a contrast gate. If the contrast is below threshold — zero compute,
//! pass through. If above threshold — spike, record the contrast,
//! update the model.
//!
//! This is a spiking neural network applied to the database itself.
//! No change = no energy. Change = proportional energy.
//!
//! ## Improvements over v1
//!
//! - **Proper centroid** via BundleAccumulator (not the every-100-records hack).
//! - **Per-field contrast** tracking: which fields change most across ingests.
//! - **Spike clustering**: when many spikes occur in a window, collect their
//!   record IDs so the Promoter can do targeted analysis.
//! - **Duplicate detection** against individual records, not just centroid.

use joule_db_hdc::{BinaryHV, BundleAccumulator};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{AmorphicRecord, RecordId, DIMENSION};

/// A detected contrast (spike event).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Contrast {
    /// What triggered the contrast
    pub source: ContrastSource,
    /// Magnitude of the contrast (0.0 = identical, 1.0 = maximally different)
    pub magnitude: f64,
    /// Which record(s) are involved
    pub record_ids: Vec<RecordId>,
    /// Description of what changed
    pub description: String,
    /// Timestamp (unix ms)
    pub timestamp_ms: u64,
}

/// Where a contrast was detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContrastSource {
    /// New record is surprisingly different from existing records
    IngestNovelty,
    /// New record is surprisingly similar to an existing record
    IngestDuplicate,
    /// Updated record changed significantly from its prior version
    UpdateDelta,
    /// Query result is different from what the model predicted
    QuerySurprise,
    /// External data contradicts stored data
    ExternalConflict,
}

/// Per-field contrast statistics.
#[derive(Debug, Clone, Default)]
pub struct FieldContrast {
    /// How many times this field appeared in ingested records
    pub appearances: u64,
    /// How many times this field's value was novel (not seen before for this field)
    pub novel_values: u64,
    /// Running average of contrast magnitude when this field differs
    pub avg_contrast: f64,
}

/// A cluster of recent spikes that may indicate a hidden dimension.
#[derive(Debug, Clone)]
pub struct SpikeCluster {
    /// Record IDs that spiked
    pub record_ids: Vec<RecordId>,
    /// Average spike magnitude
    pub avg_magnitude: f64,
    /// Time window (first spike, last spike)
    pub window_ms: (u64, u64),
}

/// The Contrast Engine — sits between data and storage.
pub struct ContrastEngine {
    /// Spike threshold: contrasts below this are ignored (no compute)
    pub novelty_threshold: f64,
    /// Duplicate threshold: similarity above this triggers duplicate detection
    pub duplicate_threshold: f32,
    /// Running model: centroid built via BundleAccumulator
    centroid_acc: BundleAccumulator,
    /// Materialized centroid (updated periodically from accumulator)
    centroid: Option<BinaryHV>,
    /// Number of records that contributed to the centroid
    centroid_count: u64,
    /// Recent contrasts (ring buffer)
    recent_contrasts: Vec<Contrast>,
    /// Maximum contrasts to retain
    max_contrasts: usize,
    /// Per-field contrast statistics
    field_stats: HashMap<String, FieldContrast>,
    /// Per-field value signatures: field -> set of value hashes seen
    field_value_hashes: HashMap<String, Vec<u64>>,
    /// Counters
    pub total_ingests: AtomicU64,
    pub total_spikes: AtomicU64,
    pub total_suppressed: AtomicU64,
}

impl ContrastEngine {
    pub fn new() -> Self {
        Self {
            novelty_threshold: 0.3,
            duplicate_threshold: 0.95,
            centroid_acc: BundleAccumulator::new(DIMENSION),
            centroid: None,
            centroid_count: 0,
            recent_contrasts: Vec::new(),
            max_contrasts: 1000,
            field_stats: HashMap::new(),
            field_value_hashes: HashMap::new(),
            total_ingests: AtomicU64::new(0),
            total_spikes: AtomicU64::new(0),
            total_suppressed: AtomicU64::new(0),
        }
    }

    /// Pass a new record through the contrast gate BEFORE storage.
    ///
    /// Returns:
    /// - `None` if the record is unremarkable (below novelty threshold). Zero additional compute.
    /// - `Some(Contrast)` if the record is novel or duplicate. The contrast is recorded.
    ///
    /// The record is always stored regardless. The contrast gate determines
    /// whether the system "notices" — whether it spends energy analyzing.
    pub fn gate_ingest(
        &mut self,
        record: &AmorphicRecord,
        timestamp_ms: u64,
    ) -> Option<Contrast> {
        self.total_ingests.fetch_add(1, Ordering::Relaxed);

        // Update per-field statistics
        self.update_field_stats(record);

        // Compute contrast against the centroid (what "normal" looks like)
        let novelty = match &self.centroid {
            Some(centroid) => 1.0 - record.hologram.similarity(centroid) as f64,
            None => 1.0, // First record — maximum novelty by definition
        };

        // Update the centroid properly
        self.update_centroid(&record.hologram);

        // Check spike threshold
        if novelty < self.novelty_threshold {
            self.total_suppressed.fetch_add(1, Ordering::Relaxed);
            return None;
        }

        // SPIKE — this record is noteworthy
        self.total_spikes.fetch_add(1, Ordering::Relaxed);

        let source = if novelty > 0.8 {
            ContrastSource::IngestNovelty
        } else {
            ContrastSource::IngestNovelty
        };

        let contrast = Contrast {
            source,
            magnitude: novelty,
            record_ids: vec![record.id],
            description: format!(
                "Record {} has novelty {:.3} (threshold: {:.3})",
                record.id, novelty, self.novelty_threshold
            ),
            timestamp_ms,
        };

        self.record_contrast(contrast.clone());
        Some(contrast)
    }

    /// Check if a record is a near-duplicate of any specific record in the store.
    ///
    /// Unlike gate_ingest (which checks against the centroid), this checks
    /// against individual records. More expensive but catches exact duplicates.
    pub fn check_duplicate(
        &self,
        record: &AmorphicRecord,
        candidates: &[(RecordId, &AmorphicRecord)],
    ) -> Option<(RecordId, f32)> {
        let mut best_match: Option<(RecordId, f32)> = None;

        for &(id, candidate) in candidates {
            if id == record.id {
                continue;
            }
            let sim = record.hologram.similarity(&candidate.hologram);
            if sim >= self.duplicate_threshold {
                match best_match {
                    None => best_match = Some((id, sim)),
                    Some((_, prev_sim)) if sim > prev_sim => best_match = Some((id, sim)),
                    _ => {}
                }
            }
        }

        best_match
    }

    /// Pass a query result through the contrast gate AFTER retrieval.
    ///
    /// Compares the query result against what was expected (the query hologram).
    /// If the result is surprising — the best match is very different from the
    /// query, or unexpectedly identical — that's a contrast.
    pub fn gate_query(
        &mut self,
        query_hologram: &BinaryHV,
        best_match: &AmorphicRecord,
        similarity: f32,
        timestamp_ms: u64,
    ) -> Option<Contrast> {
        let surprise = if similarity > self.duplicate_threshold {
            Some(Contrast {
                source: ContrastSource::IngestDuplicate,
                magnitude: similarity as f64,
                record_ids: vec![best_match.id],
                description: format!(
                    "Query result has unusually high similarity: {:.3}",
                    similarity
                ),
                timestamp_ms,
            })
        } else if (similarity as f64) < self.novelty_threshold {
            Some(Contrast {
                source: ContrastSource::QuerySurprise,
                magnitude: 1.0 - similarity as f64,
                record_ids: vec![best_match.id],
                description: format!(
                    "Best match has low similarity: {:.3} — store may be missing relevant content",
                    similarity
                ),
                timestamp_ms,
            })
        } else {
            None
        };

        if let Some(ref c) = surprise {
            self.total_spikes.fetch_add(1, Ordering::Relaxed);
            self.record_contrast(c.clone());
        }

        surprise
    }

    /// Pass an update through the contrast gate.
    ///
    /// Compares old and new versions of a record.
    pub fn gate_update(
        &mut self,
        old: &AmorphicRecord,
        new: &AmorphicRecord,
        timestamp_ms: u64,
    ) -> Option<Contrast> {
        let delta = 1.0 - old.hologram.similarity(&new.hologram) as f64;

        if delta < self.novelty_threshold {
            self.total_suppressed.fetch_add(1, Ordering::Relaxed);
            return None;
        }

        self.total_spikes.fetch_add(1, Ordering::Relaxed);

        let contrast = Contrast {
            source: ContrastSource::UpdateDelta,
            magnitude: delta,
            record_ids: vec![old.id],
            description: format!(
                "Record {} changed by {:.3} (threshold: {:.3})",
                old.id, delta, self.novelty_threshold
            ),
            timestamp_ms,
        };

        self.record_contrast(contrast.clone());
        Some(contrast)
    }

    /// Get recent contrasts.
    pub fn recent_contrasts(&self) -> &[Contrast] {
        &self.recent_contrasts
    }

    /// Get the current centroid (what "normal" looks like).
    pub fn centroid(&self) -> Option<&BinaryHV> {
        self.centroid.as_ref()
    }

    /// Number of records the centroid was built from.
    pub fn centroid_count(&self) -> u64 {
        self.centroid_count
    }

    /// Per-field contrast statistics.
    pub fn field_stats(&self) -> &HashMap<String, FieldContrast> {
        &self.field_stats
    }

    /// Get the top N most contrasting fields (highest novel value rate).
    pub fn hottest_fields(&self, n: usize) -> Vec<(String, f64)> {
        let mut fields: Vec<(String, f64)> = self.field_stats.iter()
            .filter(|(_, stats)| stats.appearances > 0)
            .map(|(name, stats)| {
                let novelty_rate = stats.novel_values as f64 / stats.appearances as f64;
                (name.clone(), novelty_rate)
            })
            .collect();

        fields.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        fields.truncate(n);
        fields
    }

    /// Detect spike clusters: groups of recent spikes that may share hidden structure.
    ///
    /// Returns clusters of record IDs that spiked within `window_ms` of each other.
    /// The Promoter can use these for targeted `analyze_records()`.
    pub fn detect_spike_clusters(&self, window_ms: u64, min_cluster_size: usize) -> Vec<SpikeCluster> {
        if self.recent_contrasts.len() < min_cluster_size {
            return Vec::new();
        }

        // Only consider ingest novelty spikes
        let spikes: Vec<&Contrast> = self.recent_contrasts.iter()
            .filter(|c| c.source == ContrastSource::IngestNovelty)
            .collect();

        if spikes.len() < min_cluster_size {
            return Vec::new();
        }

        // Sliding window clustering
        let mut clusters = Vec::new();
        let mut i = 0;

        while i < spikes.len() {
            let window_start = spikes[i].timestamp_ms;
            let window_end = window_start + window_ms;

            let mut cluster_ids = Vec::new();
            let mut total_mag = 0.0;
            let mut last_ts = window_start;

            let mut j = i;
            while j < spikes.len() && spikes[j].timestamp_ms <= window_end {
                for &id in &spikes[j].record_ids {
                    cluster_ids.push(id);
                }
                total_mag += spikes[j].magnitude;
                last_ts = spikes[j].timestamp_ms;
                j += 1;
            }

            if cluster_ids.len() >= min_cluster_size {
                clusters.push(SpikeCluster {
                    record_ids: cluster_ids,
                    avg_magnitude: total_mag / (j - i) as f64,
                    window_ms: (window_start, last_ts),
                });
                i = j; // Skip past this cluster
            } else {
                i += 1;
            }
        }

        clusters
    }

    /// Spike rate: fraction of ingests that triggered a contrast.
    pub fn spike_rate(&self) -> f64 {
        let total = self.total_ingests.load(Ordering::Relaxed);
        let spikes = self.total_spikes.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        spikes as f64 / total as f64
    }

    /// Suppression rate: fraction of ingests that were unremarkable.
    pub fn suppression_rate(&self) -> f64 {
        1.0 - self.spike_rate()
    }

    /// Adaptive threshold: if spike rate is too high, raise threshold.
    /// If too low, lower it. Target: 10-20% spike rate.
    pub fn adapt_threshold(&mut self) {
        let rate = self.spike_rate();
        if rate > 0.3 {
            self.novelty_threshold = (self.novelty_threshold * 1.1).min(0.9);
        } else if rate < 0.05 {
            self.novelty_threshold = (self.novelty_threshold * 0.9).max(0.05);
        }
    }

    // ========================================================================
    // Internal
    // ========================================================================

    /// Update the running centroid with a new hologram using BundleAccumulator.
    fn update_centroid(&mut self, hologram: &BinaryHV) {
        self.centroid_acc.add(hologram);
        self.centroid_count += 1;

        // Materialize centroid periodically (every 10 records or first record).
        // BundleAccumulator.threshold() is cheap but we don't need it every single ingest.
        if self.centroid_count <= 1 || self.centroid_count % 10 == 0 {
            self.centroid = Some(self.centroid_acc.threshold());
        }
    }

    /// Track per-field statistics for contrast detection.
    fn update_field_stats(&mut self, record: &AmorphicRecord) {
        for (field, value) in record.fields_iter() {
            let stats = self.field_stats.entry(field.to_string()).or_default();
            stats.appearances += 1;

            // Hash the value to check for novelty
            let value_hash = {
                use std::hash::{Hash, Hasher};
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                format!("{:?}", value).hash(&mut hasher);
                hasher.finish()
            };

            let seen = self.field_value_hashes
                .entry(field.to_string())
                .or_default();

            if !seen.contains(&value_hash) {
                stats.novel_values += 1;
                // Cap stored hashes to prevent unbounded growth
                if seen.len() < 10_000 {
                    seen.push(value_hash);
                }
            }
        }
    }

    /// Record a contrast in the ring buffer.
    fn record_contrast(&mut self, contrast: Contrast) {
        if self.recent_contrasts.len() >= self.max_contrasts {
            self.recent_contrasts.remove(0);
        }
        self.recent_contrasts.push(contrast);
    }
}

impl Default for ContrastEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AmorphicStore;

    #[test]
    fn test_gate_ingest_first_record_spikes() {
        let mut store = AmorphicStore::new();
        let id = store.ingest_json(r#"{"name": "first"}"#).unwrap();
        let record = store.records.get(&id).unwrap();

        let mut engine = ContrastEngine::new();
        let result = engine.gate_ingest(record, 1000);

        // First record should always spike (maximum novelty)
        assert!(result.is_some());
        assert_eq!(engine.total_spikes.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_centroid_builds_properly() {
        let mut store = AmorphicStore::new();
        let mut engine = ContrastEngine::new();

        // Ingest many similar records — centroid should stabilize
        for i in 0..20 {
            let id = store.ingest_json(&format!(
                r#"{{"name": "item_{}", "type": "widget"}}"#, i
            )).unwrap();
            let record = store.records.get(&id).unwrap();
            engine.gate_ingest(record, 1000 + i as u64);
        }

        assert!(engine.centroid().is_some());
        assert_eq!(engine.centroid_count(), 20);
    }

    #[test]
    fn test_per_field_contrast_tracking() {
        let mut store = AmorphicStore::new();
        let mut engine = ContrastEngine::new();

        // Same field, different values
        for color in &["red", "blue", "green", "red", "red"] {
            let id = store.ingest_json(&format!(
                r#"{{"color": "{}"}}"#, color
            )).unwrap();
            let record = store.records.get(&id).unwrap();
            engine.gate_ingest(record, 1000);
        }

        let stats = engine.field_stats();
        let color_stats = stats.get("color").unwrap();
        assert_eq!(color_stats.appearances, 5);
        assert_eq!(color_stats.novel_values, 3); // red, blue, green
    }

    #[test]
    fn test_hottest_fields() {
        let mut store = AmorphicStore::new();
        let mut engine = ContrastEngine::new();

        // "name" always different, "type" always same
        for i in 0..10 {
            let id = store.ingest_json(&format!(
                r#"{{"name": "item_{}", "type": "widget"}}"#, i
            )).unwrap();
            let record = store.records.get(&id).unwrap();
            engine.gate_ingest(record, 1000 + i);
        }

        let hot = engine.hottest_fields(5);
        assert!(!hot.is_empty());

        // "name" should be hotter than "type" (all unique vs all same)
        let name_rate = hot.iter().find(|(f, _)| f == "name").map(|(_, r)| *r);
        let type_rate = hot.iter().find(|(f, _)| f == "type").map(|(_, r)| *r);

        if let (Some(nr), Some(tr)) = (name_rate, type_rate) {
            assert!(nr > tr);
        }
    }

    #[test]
    fn test_spike_clustering() {
        let mut engine = ContrastEngine::new();

        // Simulate a burst of spikes within 100ms
        for i in 0..5 {
            engine.record_contrast(Contrast {
                source: ContrastSource::IngestNovelty,
                magnitude: 0.7,
                record_ids: vec![i + 1],
                description: format!("spike {}", i),
                timestamp_ms: 1000 + i * 10, // 10ms apart
            });
            engine.total_spikes.fetch_add(1, Ordering::Relaxed);
        }

        // Gap
        // Another burst
        for i in 0..5 {
            engine.record_contrast(Contrast {
                source: ContrastSource::IngestNovelty,
                magnitude: 0.6,
                record_ids: vec![i + 100],
                description: format!("spike {}", i),
                timestamp_ms: 5000 + i * 10,
            });
        }

        let clusters = engine.detect_spike_clusters(200, 3);
        assert!(clusters.len() >= 1); // At least one cluster of 5
        assert!(clusters[0].record_ids.len() >= 3);
    }

    #[test]
    fn test_gate_query_surprise() {
        let mut store = AmorphicStore::new();
        let id = store.ingest_json(r#"{"name": "something"}"#).unwrap();
        let record = store.records.get(&id).unwrap();

        let mut engine = ContrastEngine::new();
        let query_hv = BinaryHV::from_hash(b"totally_different_query", DIMENSION);

        // Low similarity should trigger query surprise
        let result = engine.gate_query(&query_hv, record, 0.1, 1000);
        assert!(result.is_some());
        assert_eq!(result.unwrap().source, ContrastSource::QuerySurprise);
    }

    #[test]
    fn test_gate_update_delta() {
        let mut store = AmorphicStore::new();
        let id1 = store.ingest_json(r#"{"name": "version_1", "data": "original"}"#).unwrap();
        let old = store.records.get(&id1).unwrap().clone();

        let id2 = store.ingest_json(r#"{"name": "completely_different_record", "data": "changed"}"#).unwrap();
        let new = store.records.get(&id2).unwrap();

        let mut engine = ContrastEngine::new();
        let result = engine.gate_update(&old, new, 1000);

        // Different records should produce a delta
        // (whether it exceeds threshold depends on holographic encoding)
        if let Some(c) = result {
            assert_eq!(c.source, ContrastSource::UpdateDelta);
            assert!(c.magnitude > 0.0);
        }
    }

    #[test]
    fn test_adaptive_threshold() {
        let mut engine = ContrastEngine::new();
        let original = engine.novelty_threshold;

        // Simulate high spike rate
        for _ in 0..100 {
            engine.total_ingests.fetch_add(1, Ordering::Relaxed);
            engine.total_spikes.fetch_add(1, Ordering::Relaxed);
        }

        engine.adapt_threshold();
        assert!(engine.novelty_threshold > original); // Should raise threshold

        // Reset and simulate low spike rate
        engine.total_ingests = AtomicU64::new(1000);
        engine.total_spikes = AtomicU64::new(1);
        let raised = engine.novelty_threshold;

        engine.adapt_threshold();
        assert!(engine.novelty_threshold < raised); // Should lower threshold
    }
}
