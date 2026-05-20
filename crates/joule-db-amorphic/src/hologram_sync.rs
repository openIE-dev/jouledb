//! Hologram Delta Sync — XOR-composable edge replication.
//!
//! BinaryHV holograms are XOR-composable:
//! - `old XOR new = delta` (compute what changed)
//! - `old XOR delta = new` (apply the change)
//! - `delta1 XOR delta2 = merged_delta` (merge multiple changes)
//!
//! This enables:
//! - **1.25KB deltas** instead of full record transfers
//! - **Associative merging** — deltas from multiple sources combine without coordination
//! - **Self-inverse** — applying a delta twice cancels it out (undo for free)
//!
//! Also provides catalog-level batch serialization for origin→edge sync.

use joule_db_hdc::BinaryHV;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{AmorphicRecord, RecordId, Value, DIMENSION};

/// A hologram delta: the XOR difference between two record versions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HologramDelta {
    /// Record this delta applies to
    pub record_id: RecordId,
    /// XOR of old and new hologram word arrays
    pub delta_words: Vec<u64>,
    /// Changed metadata fields (only the ones that differ)
    pub metadata_patch: HashMap<String, Value>,
    /// Timestamp of the change (for causal ordering)
    pub timestamp_ms: u64,
    /// Node that created this delta
    pub source_node: String,
}

impl HologramDelta {
    /// Compute delta between two versions of a record.
    pub fn compute(
        old: &AmorphicRecord,
        new: &AmorphicRecord,
        source_node: &str,
        timestamp_ms: u64,
    ) -> Self {
        let old_words = old.hologram.as_words();
        let new_words = new.hologram.as_words();

        // XOR gives the bits that changed
        let delta_words: Vec<u64> = old_words
            .iter()
            .zip(new_words.iter())
            .map(|(a, b)| a ^ b)
            .collect();

        // Find changed metadata fields
        let mut patch = HashMap::new();
        for (key, new_val) in &new.fields {
            match old.fields.get(key) {
                Some(old_val) if old_val == new_val => {} // unchanged
                _ => {
                    patch.insert(key.clone(), new_val.clone());
                }
            }
        }

        HologramDelta {
            record_id: new.id,
            delta_words,
            metadata_patch: patch,
            timestamp_ms,
            source_node: source_node.to_string(),
        }
    }

    /// Apply this delta to a record's hologram (in-place).
    /// XOR is self-inverse: applying twice cancels out.
    pub fn apply(&self, record: &mut AmorphicRecord) {
        let words = record.hologram.as_words_mut();
        for (w, &d) in words.iter_mut().zip(self.delta_words.iter()) {
            *w ^= d;
        }

        // Apply metadata patch
        for (key, value) in &self.metadata_patch {
            record.fields.insert(key.clone(), value.clone());
        }
    }

    /// Merge two deltas into one (XOR is associative + commutative).
    /// The merged delta, when applied to the original, produces the same
    /// result as applying both deltas sequentially.
    pub fn merge(a: &HologramDelta, b: &HologramDelta) -> HologramDelta {
        let delta_words: Vec<u64> = a
            .delta_words
            .iter()
            .zip(b.delta_words.iter())
            .map(|(x, y)| x ^ y)
            .collect();

        // Merge metadata patches (b overwrites a for same keys)
        let mut patch = a.metadata_patch.clone();
        patch.extend(b.metadata_patch.clone());

        HologramDelta {
            record_id: a.record_id,
            delta_words,
            metadata_patch: patch,
            timestamp_ms: a.timestamp_ms.max(b.timestamp_ms),
            source_node: if a.timestamp_ms >= b.timestamp_ms {
                a.source_node.clone()
            } else {
                b.source_node.clone()
            },
        }
    }

    /// Check if this delta is empty (no changes).
    pub fn is_empty(&self) -> bool {
        self.delta_words.iter().all(|&w| w == 0) && self.metadata_patch.is_empty()
    }

    /// Size in bytes of this delta (for bandwidth estimation).
    pub fn wire_size(&self) -> usize {
        // record_id(8) + words(n*8) + timestamp(8) + node_id(~32) + metadata(variable)
        8 + self.delta_words.len() * 8 + 8 + 32 + self.metadata_patch.len() * 64
    }
}

// ============================================================================
// Catalog Sync — batch hologram transfer for origin→edge replication
// ============================================================================

/// Batch of holograms for catalog sync between origin and edge.
/// 100K items × 1.25KB = 125MB (vs ~400MB raw JSON metadata).
#[derive(Debug, Clone)]
pub struct CatalogSync {
    /// Packed hologram data: (record_id, hologram_words)
    pub entries: Vec<CatalogEntry>,
}

/// A single entry in a catalog sync batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogEntry {
    pub record_id: RecordId,
    /// Packed hologram as u64 words
    pub hologram_words: Vec<u64>,
    /// Essential metadata (subset of full fields)
    pub metadata: HashMap<String, Value>,
}

impl CatalogSync {
    /// Create a catalog sync batch from amorphic records.
    pub fn from_records(records: &[AmorphicRecord]) -> Self {
        let entries = records
            .iter()
            .map(|r| CatalogEntry {
                record_id: r.id,
                hologram_words: r.hologram.as_words().to_vec(),
                metadata: r.fields.clone(),
            })
            .collect();
        Self { entries }
    }

    /// Create a catalog sync with only essential metadata fields.
    pub fn from_records_with_fields(
        records: &[AmorphicRecord],
        fields: &[&str],
    ) -> Self {
        let entries = records
            .iter()
            .map(|r| {
                let metadata: HashMap<String, Value> = fields
                    .iter()
                    .filter_map(|&f| r.fields.get(f).map(|v| (f.to_string(), v.clone())))
                    .collect();
                CatalogEntry {
                    record_id: r.id,
                    hologram_words: r.hologram.as_words().to_vec(),
                    metadata,
                }
            })
            .collect();
        Self { entries }
    }

    /// Serialize to compact binary format for wire transfer.
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();

        // Header: magic + entry count
        buf.extend_from_slice(b"CSYN");
        buf.extend_from_slice(&(self.entries.len() as u64).to_le_bytes());

        // Each entry: record_id + word_count + words
        for entry in &self.entries {
            buf.extend_from_slice(&(entry.record_id as u64).to_le_bytes());
            buf.extend_from_slice(&(entry.hologram_words.len() as u32).to_le_bytes());
            for &word in &entry.hologram_words {
                buf.extend_from_slice(&word.to_le_bytes());
            }
        }

        buf
    }

    /// Deserialize from compact binary format.
    pub fn deserialize(data: &[u8]) -> Option<Self> {
        if data.len() < 12 || &data[0..4] != b"CSYN" {
            return None;
        }

        let entry_count =
            u64::from_le_bytes([data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11]])
                as usize;

        let mut entries = Vec::with_capacity(entry_count);
        let mut pos = 12;

        for _ in 0..entry_count {
            if pos + 12 > data.len() {
                return None;
            }

            let record_id = u64::from_le_bytes([
                data[pos],
                data[pos + 1],
                data[pos + 2],
                data[pos + 3],
                data[pos + 4],
                data[pos + 5],
                data[pos + 6],
                data[pos + 7],
            ]) as RecordId;
            pos += 8;

            let word_count = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
                as usize;
            pos += 4;

            let mut words = Vec::with_capacity(word_count);
            for _ in 0..word_count {
                if pos + 8 > data.len() {
                    return None;
                }
                let word = u64::from_le_bytes([
                    data[pos],
                    data[pos + 1],
                    data[pos + 2],
                    data[pos + 3],
                    data[pos + 4],
                    data[pos + 5],
                    data[pos + 6],
                    data[pos + 7],
                ]);
                words.push(word);
                pos += 8;
            }

            entries.push(CatalogEntry {
                record_id,
                hologram_words: words,
                metadata: HashMap::new(), // Metadata serialization would use bincode/serde
            });
        }

        Some(Self { entries })
    }

    /// Total wire size in bytes.
    pub fn wire_size(&self) -> usize {
        12 + self
            .entries
            .iter()
            .map(|e| 12 + e.hologram_words.len() * 8)
            .sum::<usize>()
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ============================================================================
// Client-Side Hologram Cache Export
// ============================================================================

/// Export a compact binary blob of holograms for client-side caching.
/// Clients can do offline similarity search without network access.
///
/// Format: `[magic:4][count:4][dimension:4][(id:8)(words:dim/64*8)]*`
pub fn export_client_cache(records: &[AmorphicRecord]) -> Vec<u8> {
    let mut buf = Vec::new();

    buf.extend_from_slice(b"HCCH"); // Hologram Client Cache
    buf.extend_from_slice(&(records.len() as u32).to_le_bytes());
    buf.extend_from_slice(&(DIMENSION as u32).to_le_bytes());

    for record in records {
        buf.extend_from_slice(&(record.id as u64).to_le_bytes());
        for &word in record.hologram.as_words() {
            buf.extend_from_slice(&word.to_le_bytes());
        }
    }

    buf
}

/// Search within a client cache blob.
/// Returns (record_id, similarity) pairs sorted by similarity descending.
pub fn client_cache_search(
    cache: &[u8],
    query_words: &[u64],
    k: usize,
) -> Vec<(RecordId, f32)> {
    if cache.len() < 12 || &cache[0..4] != b"HCCH" {
        return Vec::new();
    }

    let count = u32::from_le_bytes([cache[4], cache[5], cache[6], cache[7]]) as usize;
    let dim = u32::from_le_bytes([cache[8], cache[9], cache[10], cache[11]]) as usize;
    let words_per_record = (dim + 63) / 64;

    let entry_size = 8 + words_per_record * 8;
    let mut results: Vec<(RecordId, f32)> = Vec::with_capacity(count);

    let mut pos = 12;
    for _ in 0..count {
        if pos + entry_size > cache.len() {
            break;
        }

        let record_id = u64::from_le_bytes([
            cache[pos], cache[pos + 1], cache[pos + 2], cache[pos + 3],
            cache[pos + 4], cache[pos + 5], cache[pos + 6], cache[pos + 7],
        ]) as RecordId;
        pos += 8;

        // Compute Hamming similarity inline (no allocation)
        let mut matching_bits = 0u32;
        let total_bits = dim as u32;
        for i in 0..words_per_record.min(query_words.len()) {
            let word = u64::from_le_bytes([
                cache[pos + i * 8],
                cache[pos + i * 8 + 1],
                cache[pos + i * 8 + 2],
                cache[pos + i * 8 + 3],
                cache[pos + i * 8 + 4],
                cache[pos + i * 8 + 5],
                cache[pos + i * 8 + 6],
                cache[pos + i * 8 + 7],
            ]);
            matching_bits += total_bits - (word ^ query_words[i]).count_ones();
        }
        pos += words_per_record * 8;

        let similarity = matching_bits as f32 / total_bits as f32;
        results.push((record_id, similarity));
    }

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(k);
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_record(id: RecordId, label: &str) -> AmorphicRecord {
        let hologram = BinaryHV::from_hash(label.as_bytes(), DIMENSION);
        AmorphicRecord {
            id,
            hologram,
            fields: {
                let mut m = HashMap::new();
                m.insert("name".to_string(), Value::String(label.to_string()));
                m
            },
            edges: vec![],
            timestamp: None,
        }
    }

    #[test]
    fn test_delta_compute_and_apply() {
        let old = make_record(1, "version_1");
        let mut current = old.clone();

        let new = make_record(1, "version_2");
        let delta = HologramDelta::compute(&old, &new, "node_a", 1000);

        // Delta should not be empty (different holograms)
        assert!(!delta.is_empty());

        // Apply delta to old → should produce new
        delta.apply(&mut current);
        assert!(
            current.hologram.similarity(&new.hologram) > 0.99,
            "Applied delta should reconstruct new hologram"
        );
    }

    #[test]
    fn test_delta_self_inverse() {
        let old = make_record(1, "original");
        let new = make_record(1, "modified");
        let delta = HologramDelta::compute(&old, &new, "node_a", 1000);

        // Apply delta twice → should return to original
        let mut record = old.clone();
        delta.apply(&mut record); // old → new
        delta.apply(&mut record); // new → old (XOR is self-inverse)
        assert!(
            record.hologram.similarity(&old.hologram) > 0.99,
            "Double-apply should return to original"
        );
    }

    #[test]
    fn test_delta_merge_associative() {
        let v1 = make_record(1, "v1");
        let v2 = make_record(1, "v2");
        let v3 = make_record(1, "v3");

        let d12 = HologramDelta::compute(&v1, &v2, "n", 100);
        let d23 = HologramDelta::compute(&v2, &v3, "n", 200);
        let d13_direct = HologramDelta::compute(&v1, &v3, "n", 200);
        let d13_merged = HologramDelta::merge(&d12, &d23);

        // Merged delta should produce same result as direct delta
        let mut from_merged = v1.clone();
        d13_merged.apply(&mut from_merged);

        let mut from_direct = v1.clone();
        d13_direct.apply(&mut from_direct);

        assert!(
            from_merged.hologram.similarity(&from_direct.hologram) > 0.99,
            "Merged delta should equal direct delta"
        );
    }

    #[test]
    fn test_identical_records_empty_delta() {
        let record = make_record(1, "same");
        let delta = HologramDelta::compute(&record, &record, "n", 0);
        assert!(delta.is_empty());
    }

    #[test]
    fn test_catalog_sync_roundtrip() {
        let records = vec![
            make_record(1, "item_a"),
            make_record(2, "item_b"),
            make_record(3, "item_c"),
        ];

        let sync = CatalogSync::from_records(&records);
        assert_eq!(sync.len(), 3);

        let bytes = sync.serialize();
        let restored = CatalogSync::deserialize(&bytes).unwrap();
        assert_eq!(restored.len(), 3);
        assert_eq!(restored.entries[0].record_id, 1);
        assert_eq!(restored.entries[2].record_id, 3);
    }

    #[test]
    fn test_client_cache_export_and_search() {
        let records = vec![
            make_record(1, "action_movie"),
            make_record(2, "comedy_show"),
            make_record(3, "action_film"), // Similar to action_movie
        ];

        let cache = export_client_cache(&records);

        // Search for something similar to "action_movie"
        let query = BinaryHV::from_hash(b"action_movie", DIMENSION);
        let results = client_cache_search(&cache, query.as_words(), 3);

        assert_eq!(results.len(), 3);
        // First result should be the exact match
        assert_eq!(results[0].0, 1); // action_movie
        assert!(results[0].1 > 0.99);
    }
}
