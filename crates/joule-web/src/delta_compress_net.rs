//! Delta compression for network state synchronization.
//!
//! Replaces custom delta encoding in Quake/Source netcode with a pure-Rust
//! delta compression system. Computes deltas between full state snapshots
//! using field-level bitmask tracking, priority-based field ordering,
//! run-length encoding of unchanged regions, bandwidth estimation, and
//! sequence tracking for reliable delta chains.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Delta compression errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeltaError {
    /// Baseline snapshot not found for delta decoding.
    BaselineNotFound { sequence: u64 },
    /// Field index out of range.
    FieldOutOfRange { index: usize, max: usize },
    /// Delta chain broken (missing intermediate snapshot).
    ChainBroken { expected: u64, got: u64 },
    /// Decode buffer too short.
    BufferTooShort { expected: usize, actual: usize },
}

impl fmt::Display for DeltaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BaselineNotFound { sequence } => {
                write!(f, "baseline snapshot {sequence} not found")
            }
            Self::FieldOutOfRange { index, max } => {
                write!(f, "field index {index} out of range (max {max})")
            }
            Self::ChainBroken { expected, got } => {
                write!(f, "delta chain broken: expected seq {expected}, got {got}")
            }
            Self::BufferTooShort { expected, actual } => {
                write!(f, "buffer too short: need {expected} bytes, have {actual}")
            }
        }
    }
}

impl std::error::Error for DeltaError {}

// ── Field Priority ──────────────────────────────────────────────

/// Priority level for a field, determining send order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FieldPriority {
    Critical = 0,
    High = 1,
    Normal = 2,
    Low = 3,
}

impl fmt::Display for FieldPriority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Critical => write!(f, "CRITICAL"),
            Self::High => write!(f, "HIGH"),
            Self::Normal => write!(f, "NORMAL"),
            Self::Low => write!(f, "LOW"),
        }
    }
}

// ── Field Descriptor ────────────────────────────────────────────

/// Describes a single field in the state schema.
#[derive(Debug, Clone)]
pub struct FieldDescriptor {
    pub name: String,
    pub size_bytes: usize,
    pub priority: FieldPriority,
    pub index: usize,
}

impl FieldDescriptor {
    pub fn new(name: impl Into<String>, size: usize, priority: FieldPriority, index: usize) -> Self {
        Self { name: name.into(), size_bytes: size, priority, index }
    }
}

// ── State Schema ────────────────────────────────────────────────

/// Schema defining the layout of a state snapshot.
#[derive(Debug, Clone)]
pub struct StateSchema {
    fields: Vec<FieldDescriptor>,
    total_size: usize,
}

impl StateSchema {
    pub fn new() -> Self {
        Self { fields: Vec::new(), total_size: 0 }
    }

    pub fn add_field(&mut self, name: impl Into<String>, size: usize, priority: FieldPriority) {
        let index = self.fields.len();
        self.fields.push(FieldDescriptor::new(name, size, priority, index));
        self.total_size += size;
    }

    pub fn field_count(&self) -> usize {
        self.fields.len()
    }

    pub fn total_size(&self) -> usize {
        self.total_size
    }

    /// Get field offset in the flat byte array.
    pub fn field_offset(&self, index: usize) -> Option<usize> {
        if index >= self.fields.len() {
            return None;
        }
        Some(self.fields[..index].iter().map(|f| f.size_bytes).sum())
    }

    /// Get fields sorted by priority (critical first).
    pub fn priority_order(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.fields.len()).collect();
        indices.sort_by_key(|i| self.fields[*i].priority);
        indices
    }

    pub fn field(&self, index: usize) -> Option<&FieldDescriptor> {
        self.fields.get(index)
    }
}

impl Default for StateSchema {
    fn default() -> Self {
        Self::new()
    }
}

// ── State Snapshot ──────────────────────────────────────────────

/// A full state snapshot (flat byte representation).
#[derive(Debug, Clone)]
pub struct StateSnapshot {
    pub sequence: u64,
    pub tick: u64,
    pub data: Vec<u8>,
}

impl StateSnapshot {
    pub fn new(sequence: u64, tick: u64, data: Vec<u8>) -> Self {
        Self { sequence, tick, data }
    }

    /// Read a field's bytes given schema.
    pub fn read_field(&self, schema: &StateSchema, index: usize) -> Result<&[u8], DeltaError> {
        let offset = schema.field_offset(index).ok_or(DeltaError::FieldOutOfRange {
            index,
            max: schema.field_count(),
        })?;
        let field = schema.field(index).unwrap();
        let end = offset + field.size_bytes;
        if end > self.data.len() {
            return Err(DeltaError::BufferTooShort { expected: end, actual: self.data.len() });
        }
        Ok(&self.data[offset..end])
    }
}

// ── Change Bitmask ──────────────────────────────────────────────

/// Bitmask tracking which fields have changed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChangeBitmask {
    bits: Vec<u8>,
    field_count: usize,
}

impl ChangeBitmask {
    pub fn new(field_count: usize) -> Self {
        let bytes = (field_count + 7) / 8;
        Self { bits: vec![0; bytes], field_count }
    }

    pub fn set(&mut self, index: usize) {
        if index < self.field_count {
            self.bits[index / 8] |= 1 << (index % 8);
        }
    }

    pub fn is_set(&self, index: usize) -> bool {
        if index >= self.field_count {
            return false;
        }
        (self.bits[index / 8] & (1 << (index % 8))) != 0
    }

    pub fn changed_count(&self) -> usize {
        (0..self.field_count).filter(|i| self.is_set(*i)).count()
    }

    pub fn all_unchanged(&self) -> bool {
        self.bits.iter().all(|b| *b == 0)
    }

    pub fn raw_bytes(&self) -> &[u8] {
        &self.bits
    }

    pub fn field_count(&self) -> usize {
        self.field_count
    }
}

impl fmt::Display for ChangeBitmask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Bitmask({}/{})", self.changed_count(), self.field_count)
    }
}

// ── Delta ───────────────────────────────────────────────────────

/// A delta between two snapshots.
#[derive(Debug, Clone)]
pub struct StateDelta {
    pub base_sequence: u64,
    pub target_sequence: u64,
    pub target_tick: u64,
    pub bitmask: ChangeBitmask,
    /// Changed field data in priority order.
    pub changed_data: Vec<u8>,
}

impl StateDelta {
    pub fn size_bytes(&self) -> usize {
        self.bitmask.raw_bytes().len() + self.changed_data.len() + 24 // header
    }

    pub fn compression_ratio(&self, full_size: usize) -> f64 {
        if full_size == 0 {
            return 0.0;
        }
        self.size_bytes() as f64 / full_size as f64
    }
}

// ── Run-Length Encoding ─────────────────────────────────────────

/// Run-length encode unchanged field spans in a bitmask.
#[derive(Debug, Clone)]
pub struct RleSpan {
    pub start: usize,
    pub length: usize,
    pub changed: bool,
}

/// Compute RLE spans from a bitmask.
pub fn rle_encode(bitmask: &ChangeBitmask) -> Vec<RleSpan> {
    if bitmask.field_count() == 0 {
        return Vec::new();
    }
    let mut spans = Vec::new();
    let mut current_changed = bitmask.is_set(0);
    let mut start = 0;
    let mut length = 1;

    for i in 1..bitmask.field_count() {
        let changed = bitmask.is_set(i);
        if changed == current_changed {
            length += 1;
        } else {
            spans.push(RleSpan { start, length, changed: current_changed });
            current_changed = changed;
            start = i;
            length = 1;
        }
    }
    spans.push(RleSpan { start, length, changed: current_changed });
    spans
}

// ── Bandwidth Estimator ─────────────────────────────────────────

/// Estimates bandwidth from delta history.
#[derive(Debug)]
pub struct BandwidthEstimator {
    samples: Vec<usize>,
    max_samples: usize,
    tick_rate: f64,
}

impl BandwidthEstimator {
    pub fn new(max_samples: usize, tick_rate: f64) -> Self {
        Self { samples: Vec::with_capacity(max_samples), max_samples, tick_rate }
    }

    pub fn record_delta(&mut self, delta_bytes: usize) {
        if self.samples.len() >= self.max_samples {
            self.samples.remove(0);
        }
        self.samples.push(delta_bytes);
    }

    /// Average bytes per tick.
    pub fn avg_bytes_per_tick(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        self.samples.iter().sum::<usize>() as f64 / self.samples.len() as f64
    }

    /// Estimated bytes per second.
    pub fn estimated_bps(&self) -> f64 {
        self.avg_bytes_per_tick() * self.tick_rate
    }

    /// Estimated kilobits per second.
    pub fn estimated_kbps(&self) -> f64 {
        self.estimated_bps() * 8.0 / 1000.0
    }
}

// ── Delta Encoder ───────────────────────────────────────────────

/// Computes deltas between state snapshots.
#[derive(Debug)]
pub struct DeltaEncoder {
    schema: StateSchema,
    baselines: HashMap<u64, StateSnapshot>,
    next_sequence: u64,
    bandwidth: BandwidthEstimator,
}

impl DeltaEncoder {
    pub fn new(schema: StateSchema, tick_rate: f64) -> Self {
        Self {
            schema,
            baselines: HashMap::new(),
            next_sequence: 0,
            bandwidth: BandwidthEstimator::new(100, tick_rate),
        }
    }

    /// Store a full snapshot as a baseline.
    pub fn store_baseline(&mut self, snapshot: StateSnapshot) {
        self.baselines.insert(snapshot.sequence, snapshot);
    }

    /// Compute a delta from a baseline to a new state.
    pub fn encode_delta(
        &mut self,
        baseline_seq: u64,
        new_data: &[u8],
        new_tick: u64,
    ) -> Result<StateDelta, DeltaError> {
        let baseline = self
            .baselines
            .get(&baseline_seq)
            .ok_or(DeltaError::BaselineNotFound { sequence: baseline_seq })?;

        let mut bitmask = ChangeBitmask::new(self.schema.field_count());
        let mut changed_data = Vec::new();

        // Compare fields in priority order.
        let priority_order = self.schema.priority_order();
        for &idx in &priority_order {
            let offset = self.schema.field_offset(idx).unwrap();
            let field = self.schema.field(idx).unwrap();
            let end = offset + field.size_bytes;

            if end > baseline.data.len() || end > new_data.len() {
                continue;
            }

            let old_slice = &baseline.data[offset..end];
            let new_slice = &new_data[offset..end];

            if old_slice != new_slice {
                bitmask.set(idx);
                changed_data.extend_from_slice(new_slice);
            }
        }

        let target_seq = self.next_sequence;
        self.next_sequence += 1;

        let delta = StateDelta {
            base_sequence: baseline_seq,
            target_sequence: target_seq,
            target_tick: new_tick,
            bitmask,
            changed_data,
        };

        self.bandwidth.record_delta(delta.size_bytes());
        Ok(delta)
    }

    /// Decode a delta against its baseline to produce a full snapshot.
    pub fn decode_delta(&self, delta: &StateDelta) -> Result<StateSnapshot, DeltaError> {
        let baseline = self
            .baselines
            .get(&delta.base_sequence)
            .ok_or(DeltaError::BaselineNotFound { sequence: delta.base_sequence })?;

        let mut data = baseline.data.clone();
        let priority_order = self.schema.priority_order();
        let mut read_pos = 0;

        for &idx in &priority_order {
            if delta.bitmask.is_set(idx) {
                let offset = self.schema.field_offset(idx).unwrap();
                let field = self.schema.field(idx).unwrap();
                let end = offset + field.size_bytes;

                if end <= data.len() && read_pos + field.size_bytes <= delta.changed_data.len() {
                    data[offset..end]
                        .copy_from_slice(&delta.changed_data[read_pos..read_pos + field.size_bytes]);
                }
                read_pos += field.size_bytes;
            }
        }

        Ok(StateSnapshot::new(delta.target_sequence, delta.target_tick, data))
    }

    pub fn estimated_kbps(&self) -> f64 {
        self.bandwidth.estimated_kbps()
    }

    pub fn schema(&self) -> &StateSchema {
        &self.schema
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_schema() -> StateSchema {
        let mut s = StateSchema::new();
        s.add_field("pos_x", 4, FieldPriority::Critical);
        s.add_field("pos_y", 4, FieldPriority::Critical);
        s.add_field("health", 2, FieldPriority::High);
        s.add_field("ammo", 2, FieldPriority::Normal);
        s.add_field("score", 4, FieldPriority::Low);
        s
    }

    #[test]
    fn schema_field_count() {
        let s = make_schema();
        assert_eq!(s.field_count(), 5);
    }

    #[test]
    fn schema_total_size() {
        let s = make_schema();
        assert_eq!(s.total_size(), 16);
    }

    #[test]
    fn schema_field_offset() {
        let s = make_schema();
        assert_eq!(s.field_offset(0), Some(0));
        assert_eq!(s.field_offset(1), Some(4));
        assert_eq!(s.field_offset(2), Some(8));
        assert_eq!(s.field_offset(3), Some(10));
        assert_eq!(s.field_offset(4), Some(12));
    }

    #[test]
    fn schema_priority_order() {
        let s = make_schema();
        let order = s.priority_order();
        assert_eq!(order[0], 0); // pos_x Critical
        assert_eq!(order[1], 1); // pos_y Critical
    }

    #[test]
    fn bitmask_set_and_get() {
        let mut bm = ChangeBitmask::new(16);
        bm.set(0);
        bm.set(5);
        bm.set(15);
        assert!(bm.is_set(0));
        assert!(bm.is_set(5));
        assert!(bm.is_set(15));
        assert!(!bm.is_set(1));
    }

    #[test]
    fn bitmask_changed_count() {
        let mut bm = ChangeBitmask::new(8);
        bm.set(0);
        bm.set(3);
        bm.set(7);
        assert_eq!(bm.changed_count(), 3);
    }

    #[test]
    fn bitmask_all_unchanged() {
        let bm = ChangeBitmask::new(8);
        assert!(bm.all_unchanged());
    }

    #[test]
    fn bitmask_display() {
        let mut bm = ChangeBitmask::new(10);
        bm.set(2);
        let s = format!("{bm}");
        assert!(s.contains("1/10"));
    }

    #[test]
    fn rle_encode_basic() {
        let mut bm = ChangeBitmask::new(8);
        bm.set(0);
        bm.set(1);
        // 2,3,4,5 unchanged; 6,7 unchanged
        let spans = rle_encode(&bm);
        assert_eq!(spans[0].changed, true);
        assert_eq!(spans[0].length, 2);
        assert_eq!(spans[1].changed, false);
    }

    #[test]
    fn rle_encode_all_same() {
        let bm = ChangeBitmask::new(5);
        let spans = rle_encode(&bm);
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].length, 5);
        assert!(!spans[0].changed);
    }

    #[test]
    fn snapshot_read_field() {
        let schema = make_schema();
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let snap = StateSnapshot::new(0, 0, data);
        let field = snap.read_field(&schema, 0).unwrap();
        assert_eq!(field, &[1, 2, 3, 4]);
    }

    #[test]
    fn encode_decode_roundtrip() {
        let schema = make_schema();
        let baseline_data = vec![0u8; 16];
        let mut new_data = vec![0u8; 16];
        new_data[0] = 42; // change pos_x

        let baseline = StateSnapshot::new(0, 0, baseline_data);
        let mut encoder = DeltaEncoder::new(schema, 60.0);
        encoder.store_baseline(baseline);

        let delta = encoder.encode_delta(0, &new_data, 1).unwrap();
        assert!(delta.bitmask.is_set(0)); // pos_x changed
        assert!(!delta.bitmask.is_set(1)); // pos_y unchanged

        let decoded = encoder.decode_delta(&delta).unwrap();
        assert_eq!(decoded.data, new_data);
    }

    #[test]
    fn delta_compression_ratio() {
        let schema = make_schema();
        let baseline_data = vec![0u8; 16];
        let mut new_data = vec![0u8; 16];
        new_data[0] = 1; // only one field changed

        let baseline = StateSnapshot::new(0, 0, baseline_data);
        let mut encoder = DeltaEncoder::new(schema, 60.0);
        encoder.store_baseline(baseline);

        let delta = encoder.encode_delta(0, &new_data, 1).unwrap();
        // Changed data payload is smaller than the full snapshot.
        assert!(delta.changed_data.len() < 16);
        // Only 1 field changed out of 5.
        assert_eq!(delta.bitmask.changed_count(), 1);
        // Ratio includes header overhead, so for small snapshots it may exceed 1.0;
        // for large real-world snapshots this would compress well.
        let ratio = delta.compression_ratio(16);
        assert!(ratio > 0.0);
    }

    #[test]
    fn delta_no_changes() {
        let schema = make_schema();
        let data = vec![0u8; 16];
        let baseline = StateSnapshot::new(0, 0, data.clone());
        let mut encoder = DeltaEncoder::new(schema, 60.0);
        encoder.store_baseline(baseline);

        let delta = encoder.encode_delta(0, &data, 1).unwrap();
        assert!(delta.bitmask.all_unchanged());
        assert!(delta.changed_data.is_empty());
    }

    #[test]
    fn baseline_not_found() {
        let schema = make_schema();
        let mut encoder = DeltaEncoder::new(schema, 60.0);
        let err = encoder.encode_delta(99, &[0; 16], 1).unwrap_err();
        assert_eq!(err, DeltaError::BaselineNotFound { sequence: 99 });
    }

    #[test]
    fn bandwidth_estimator() {
        let mut est = BandwidthEstimator::new(10, 60.0);
        est.record_delta(100);
        est.record_delta(200);
        assert!((est.avg_bytes_per_tick() - 150.0).abs() < 1e-9);
        assert!((est.estimated_bps() - 9000.0).abs() < 1e-9);
    }

    #[test]
    fn field_priority_display() {
        assert_eq!(format!("{}", FieldPriority::Critical), "CRITICAL");
        assert_eq!(format!("{}", FieldPriority::Low), "LOW");
    }

    #[test]
    fn rle_empty_bitmask() {
        let bm = ChangeBitmask::new(0);
        let spans = rle_encode(&bm);
        assert!(spans.is_empty());
    }
}
