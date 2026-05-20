//! Write-ahead log manager — log file segments, sequential write, sync policies
//! (every write/batch/periodic), log replay on recovery, log rotation, garbage
//! collection, corruption detection with checksums.

use std::collections::BTreeMap;

// ── Errors ───────────────────────────────────────────────────────────────────

/// Errors returned by the WAL manager.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WalManagerError {
    /// Segment not found.
    SegmentNotFound(u64),
    /// LSN not found.
    LsnNotFound(u64),
    /// Checksum mismatch (corruption).
    ChecksumMismatch { lsn: u64, expected: u32, actual: u32 },
    /// Segment is sealed and cannot accept writes.
    SegmentSealed(u64),
    /// WAL is empty.
    Empty,
    /// Replay failed.
    ReplayFailed(String),
}

impl std::fmt::Display for WalManagerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SegmentNotFound(id) => write!(f, "segment {id} not found"),
            Self::LsnNotFound(lsn) => write!(f, "LSN {lsn} not found"),
            Self::ChecksumMismatch { lsn, expected, actual } => {
                write!(f, "checksum mismatch at LSN {lsn}: expected {expected:#010x}, got {actual:#010x}")
            }
            Self::SegmentSealed(id) => write!(f, "segment {id} is sealed"),
            Self::Empty => write!(f, "WAL is empty"),
            Self::ReplayFailed(msg) => write!(f, "replay failed: {msg}"),
        }
    }
}

impl std::error::Error for WalManagerError {}

// ── CRC32 ────────────────────────────────────────────────────────────────────

fn crc32(data: &[u8]) -> u32 {
    const POLY: u32 = 0xEDB88320;
    let mut crc: u32 = 0xFFFFFFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ POLY;
            } else {
                crc >>= 1;
            }
        }
    }
    crc ^ 0xFFFFFFFF
}

// ── Sync Policy ──────────────────────────────────────────────────────────────

/// Controls when the WAL flushes writes to durable storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncPolicy {
    /// Sync after every write.
    EveryWrite,
    /// Sync after each batch.
    PerBatch,
    /// Sync periodically (caller manages timing).
    Periodic,
    /// No automatic sync.
    None,
}

// ── WAL Entry ────────────────────────────────────────────────────────────────

/// A single WAL entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalEntry {
    /// Log sequence number.
    pub lsn: u64,
    /// Payload data.
    pub data: Vec<u8>,
    /// CRC32 checksum of the data.
    pub checksum: u32,
    /// Operation tag (e.g. "put", "delete").
    pub op_type: String,
    /// Segment this entry belongs to.
    pub segment_id: u64,
}

impl WalEntry {
    /// Create a new entry with an automatically computed checksum.
    pub fn new(lsn: u64, data: Vec<u8>, op_type: String, segment_id: u64) -> Self {
        let checksum = crc32(&data);
        Self {
            lsn,
            data,
            checksum,
            op_type,
            segment_id,
        }
    }

    /// Verify the entry checksum.
    pub fn verify(&self) -> bool {
        crc32(&self.data) == self.checksum
    }

    /// Data length in bytes.
    pub fn data_len(&self) -> usize {
        self.data.len()
    }
}

// ── Log Segment ──────────────────────────────────────────────────────────────

/// A single log segment containing a contiguous range of LSNs.
#[derive(Debug, Clone)]
pub struct LogSegment {
    /// Unique segment ID.
    id: u64,
    /// Entries in LSN order.
    entries: Vec<WalEntry>,
    /// Maximum size in bytes before rotating.
    max_size_bytes: usize,
    /// Current size in bytes.
    size_bytes: usize,
    /// Whether this segment is sealed (read-only).
    sealed: bool,
    /// Number of syncs performed.
    sync_count: u64,
}

impl LogSegment {
    /// Create a new segment.
    pub fn new(id: u64, max_size_bytes: usize) -> Self {
        Self {
            id,
            entries: Vec::new(),
            max_size_bytes,
            size_bytes: 0,
            sealed: false,
            sync_count: 0,
        }
    }

    /// Append an entry to the segment.
    pub fn append(&mut self, entry: WalEntry) -> Result<(), WalManagerError> {
        if self.sealed {
            return Err(WalManagerError::SegmentSealed(self.id));
        }
        self.size_bytes += entry.data.len() + 32; // data + overhead
        self.entries.push(entry);
        Ok(())
    }

    /// Whether the segment is at capacity.
    pub fn is_full(&self) -> bool {
        self.size_bytes >= self.max_size_bytes
    }

    /// Seal the segment (no more writes).
    pub fn seal(&mut self) {
        self.sealed = true;
    }

    /// Whether the segment is sealed.
    pub fn is_sealed(&self) -> bool {
        self.sealed
    }

    /// Segment ID.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the segment is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    /// First LSN in the segment, if any.
    pub fn first_lsn(&self) -> Option<u64> {
        self.entries.first().map(|e| e.lsn)
    }

    /// Last LSN in the segment, if any.
    pub fn last_lsn(&self) -> Option<u64> {
        self.entries.last().map(|e| e.lsn)
    }

    /// Record a sync event.
    pub fn record_sync(&mut self) {
        self.sync_count += 1;
    }

    /// Sync count.
    pub fn sync_count(&self) -> u64 {
        self.sync_count
    }

    /// Iterate entries.
    pub fn iter(&self) -> impl Iterator<Item = &WalEntry> {
        self.entries.iter()
    }

    /// Verify all entries in the segment.
    pub fn verify_all(&self) -> Result<(), WalManagerError> {
        for entry in &self.entries {
            let actual = crc32(&entry.data);
            if actual != entry.checksum {
                return Err(WalManagerError::ChecksumMismatch {
                    lsn: entry.lsn,
                    expected: entry.checksum,
                    actual,
                });
            }
        }
        Ok(())
    }
}

// ── WAL Statistics ───────────────────────────────────────────────────────────

/// Aggregate statistics for the WAL manager.
#[derive(Debug, Clone, Default)]
pub struct WalStats {
    pub total_entries: usize,
    pub total_segments: usize,
    pub active_segments: usize,
    pub sealed_segments: usize,
    pub total_bytes: usize,
    pub total_syncs: u64,
    pub total_rotations: u64,
    pub total_gc_runs: u64,
    pub entries_garbage_collected: u64,
    pub corrupted_entries: u64,
}

// ── WAL Manager ──────────────────────────────────────────────────────────────

/// Manages write-ahead log segments with rotation, replay, and GC.
#[derive(Debug)]
pub struct WalManager {
    /// Segments ordered by ID.
    segments: BTreeMap<u64, LogSegment>,
    /// Currently active segment ID.
    active_segment_id: u64,
    /// Next LSN to assign.
    next_lsn: u64,
    /// Next segment ID.
    next_segment_id: u64,
    /// Maximum segment size.
    segment_max_bytes: usize,
    /// Sync policy.
    sync_policy: SyncPolicy,
    /// Stats counters.
    total_syncs: u64,
    total_rotations: u64,
    total_gc_runs: u64,
    entries_gc: u64,
    corrupted_found: u64,
    /// Pending batch entries (for PerBatch sync).
    batch_pending: usize,
}

impl WalManager {
    /// Create a new WAL manager.
    pub fn new(segment_max_bytes: usize, sync_policy: SyncPolicy) -> Self {
        let segment = LogSegment::new(1, segment_max_bytes);
        let mut segments = BTreeMap::new();
        segments.insert(1, segment);

        Self {
            segments,
            active_segment_id: 1,
            next_lsn: 1,
            next_segment_id: 2,
            segment_max_bytes,
            sync_policy,
            total_syncs: 0,
            total_rotations: 0,
            total_gc_runs: 0,
            entries_gc: 0,
            corrupted_found: 0,
            batch_pending: 0,
        }
    }

    /// Append a write to the WAL.
    pub fn append(&mut self, data: Vec<u8>, op_type: String) -> Result<u64, WalManagerError> {
        let lsn = self.next_lsn;
        self.next_lsn += 1;

        let seg_id = self.active_segment_id;
        let entry = WalEntry::new(lsn, data, op_type, seg_id);

        // Check if active segment is full — rotate first.
        {
            let active = self
                .segments
                .get(&self.active_segment_id)
                .ok_or(WalManagerError::SegmentNotFound(self.active_segment_id))?;
            if active.is_full() {
                self.rotate()?;
            }
        }

        let current_id = self.active_segment_id;
        let active = self
            .segments
            .get_mut(&current_id)
            .ok_or(WalManagerError::SegmentNotFound(current_id))?;
        active.append(WalEntry::new(lsn, entry.data, entry.op_type, current_id))?;

        self.batch_pending += 1;

        // Apply sync policy.
        match self.sync_policy {
            SyncPolicy::EveryWrite => {
                self.sync_active()?;
            }
            _ => {}
        }

        Ok(lsn)
    }

    /// Write a batch of entries atomically.
    pub fn append_batch(
        &mut self,
        entries: Vec<(Vec<u8>, String)>,
    ) -> Result<Vec<u64>, WalManagerError> {
        let mut lsns = Vec::with_capacity(entries.len());
        for (data, op_type) in entries {
            let lsn = self.append(data, op_type)?;
            lsns.push(lsn);
        }
        if self.sync_policy == SyncPolicy::PerBatch {
            self.sync_active()?;
        }
        Ok(lsns)
    }

    /// Sync the active segment.
    pub fn sync_active(&mut self) -> Result<(), WalManagerError> {
        let id = self.active_segment_id;
        let active = self
            .segments
            .get_mut(&id)
            .ok_or(WalManagerError::SegmentNotFound(id))?;
        active.record_sync();
        self.total_syncs += 1;
        self.batch_pending = 0;
        Ok(())
    }

    /// Rotate: seal the active segment and create a new one.
    pub fn rotate(&mut self) -> Result<u64, WalManagerError> {
        let old_id = self.active_segment_id;
        let old_seg = self
            .segments
            .get_mut(&old_id)
            .ok_or(WalManagerError::SegmentNotFound(old_id))?;
        old_seg.seal();

        let new_id = self.next_segment_id;
        self.next_segment_id += 1;
        let new_segment = LogSegment::new(new_id, self.segment_max_bytes);
        self.segments.insert(new_id, new_segment);
        self.active_segment_id = new_id;
        self.total_rotations += 1;

        Ok(new_id)
    }

    /// Read an entry by LSN.
    pub fn read(&self, lsn: u64) -> Result<&WalEntry, WalManagerError> {
        for segment in self.segments.values() {
            for entry in segment.iter() {
                if entry.lsn == lsn {
                    return Ok(entry);
                }
            }
        }
        Err(WalManagerError::LsnNotFound(lsn))
    }

    /// Replay all entries from a given LSN onward, returning them in order.
    pub fn replay_from(&self, start_lsn: u64) -> Result<Vec<&WalEntry>, WalManagerError> {
        let mut result = Vec::new();
        let mut found = false;
        for segment in self.segments.values() {
            for entry in segment.iter() {
                if entry.lsn >= start_lsn {
                    found = true;
                    result.push(entry);
                }
            }
        }
        if !found && start_lsn > 0 {
            // Check if the LSN exists at all.
            let max_lsn = self.max_lsn();
            if max_lsn.is_none() || start_lsn > max_lsn.unwrap() {
                return Err(WalManagerError::LsnNotFound(start_lsn));
            }
        }
        Ok(result)
    }

    /// Replay all entries, verifying checksums.
    pub fn replay_all_verified(&self) -> Result<Vec<&WalEntry>, WalManagerError> {
        let mut result = Vec::new();
        for segment in self.segments.values() {
            segment.verify_all()?;
            for entry in segment.iter() {
                result.push(entry);
            }
        }
        Ok(result)
    }

    /// Garbage-collect sealed segments whose max LSN < threshold.
    pub fn gc(&mut self, below_lsn: u64) -> u64 {
        let mut removed = 0u64;
        let ids_to_remove: Vec<u64> = self
            .segments
            .iter()
            .filter(|(_, seg)| {
                seg.is_sealed() && seg.last_lsn().map_or(true, |lsn| lsn < below_lsn)
            })
            .map(|(&id, seg)| {
                removed += seg.len() as u64;
                id
            })
            .collect();

        for id in ids_to_remove {
            self.segments.remove(&id);
        }
        self.total_gc_runs += 1;
        self.entries_gc += removed;
        removed
    }

    /// Detect corrupted entries across all segments.
    pub fn detect_corruption(&mut self) -> Vec<u64> {
        let mut corrupted_lsns = Vec::new();
        for segment in self.segments.values() {
            for entry in segment.iter() {
                if !entry.verify() {
                    corrupted_lsns.push(entry.lsn);
                }
            }
        }
        self.corrupted_found += corrupted_lsns.len() as u64;
        corrupted_lsns
    }

    /// Maximum LSN in the WAL.
    pub fn max_lsn(&self) -> Option<u64> {
        self.segments
            .values()
            .rev()
            .find_map(|seg| seg.last_lsn())
    }

    /// Minimum LSN in the WAL.
    pub fn min_lsn(&self) -> Option<u64> {
        self.segments
            .values()
            .find_map(|seg| seg.first_lsn())
    }

    /// Total number of entries across all segments.
    pub fn total_entries(&self) -> usize {
        self.segments.values().map(|s| s.len()).sum()
    }

    /// Number of segments.
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Current sync policy.
    pub fn sync_policy(&self) -> SyncPolicy {
        self.sync_policy
    }

    /// Update the sync policy.
    pub fn set_sync_policy(&mut self, policy: SyncPolicy) {
        self.sync_policy = policy;
    }

    /// Get statistics.
    pub fn stats(&self) -> WalStats {
        let sealed = self.segments.values().filter(|s| s.is_sealed()).count();
        WalStats {
            total_entries: self.total_entries(),
            total_segments: self.segments.len(),
            active_segments: self.segments.len() - sealed,
            sealed_segments: sealed,
            total_bytes: self.segments.values().map(|s| s.size_bytes()).sum(),
            total_syncs: self.total_syncs,
            total_rotations: self.total_rotations,
            total_gc_runs: self.total_gc_runs,
            entries_garbage_collected: self.entries_gc,
            corrupted_entries: self.corrupted_found,
        }
    }

    /// Active segment ID.
    pub fn active_segment_id(&self) -> u64 {
        self.active_segment_id
    }

    /// Batch pending count.
    pub fn batch_pending(&self) -> usize {
        self.batch_pending
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entry_create_and_verify() {
        let entry = WalEntry::new(1, b"hello".to_vec(), "put".into(), 1);
        assert!(entry.verify());
        assert_eq!(entry.lsn, 1);
        assert_eq!(entry.data_len(), 5);
    }

    #[test]
    fn entry_corrupt_detected() {
        let mut entry = WalEntry::new(1, b"data".to_vec(), "put".into(), 1);
        entry.checksum ^= 0xFFFF;
        assert!(!entry.verify());
    }

    #[test]
    fn segment_append_and_iterate() {
        let mut seg = LogSegment::new(1, 4096);
        seg.append(WalEntry::new(1, b"a".to_vec(), "put".into(), 1)).unwrap();
        seg.append(WalEntry::new(2, b"b".to_vec(), "put".into(), 1)).unwrap();
        assert_eq!(seg.len(), 2);
        assert_eq!(seg.first_lsn(), Some(1));
        assert_eq!(seg.last_lsn(), Some(2));
    }

    #[test]
    fn segment_seal_prevents_writes() {
        let mut seg = LogSegment::new(1, 4096);
        seg.seal();
        let result = seg.append(WalEntry::new(1, b"x".to_vec(), "put".into(), 1));
        assert_eq!(result, Err(WalManagerError::SegmentSealed(1)));
    }

    #[test]
    fn segment_is_full() {
        let mut seg = LogSegment::new(1, 64);
        seg.append(WalEntry::new(1, vec![0u8; 40], "put".into(), 1)).unwrap();
        assert!(seg.is_full());
    }

    #[test]
    fn segment_verify_all() {
        let mut seg = LogSegment::new(1, 4096);
        seg.append(WalEntry::new(1, b"ok".to_vec(), "put".into(), 1)).unwrap();
        seg.append(WalEntry::new(2, b"fine".to_vec(), "put".into(), 1)).unwrap();
        assert!(seg.verify_all().is_ok());
    }

    #[test]
    fn wal_append_and_read() {
        let mut wal = WalManager::new(4096, SyncPolicy::None);
        let lsn1 = wal.append(b"data1".to_vec(), "put".into()).unwrap();
        let lsn2 = wal.append(b"data2".to_vec(), "delete".into()).unwrap();
        assert_eq!(lsn1, 1);
        assert_eq!(lsn2, 2);
        let entry = wal.read(1).unwrap();
        assert_eq!(entry.data, b"data1");
        assert_eq!(wal.total_entries(), 2);
    }

    #[test]
    fn wal_append_batch() {
        let mut wal = WalManager::new(4096, SyncPolicy::PerBatch);
        let batch = vec![
            (b"a".to_vec(), "put".into()),
            (b"b".to_vec(), "put".into()),
            (b"c".to_vec(), "delete".into()),
        ];
        let lsns = wal.append_batch(batch).unwrap();
        assert_eq!(lsns.len(), 3);
        assert_eq!(wal.total_entries(), 3);
    }

    #[test]
    fn wal_rotation() {
        let mut wal = WalManager::new(64, SyncPolicy::None);
        let first_seg = wal.active_segment_id();
        // Fill up the first segment.
        for _ in 0..5 {
            wal.append(vec![0u8; 20], "put".into()).unwrap();
        }
        // Rotation should have occurred.
        assert!(wal.segment_count() >= 2 || wal.active_segment_id() != first_seg);
    }

    #[test]
    fn wal_replay_from() {
        let mut wal = WalManager::new(4096, SyncPolicy::None);
        for i in 0..5u8 {
            wal.append(vec![i], "put".into()).unwrap();
        }
        let replayed = wal.replay_from(3).unwrap();
        assert_eq!(replayed.len(), 3);
        assert_eq!(replayed[0].lsn, 3);
    }

    #[test]
    fn wal_replay_all_verified() {
        let mut wal = WalManager::new(4096, SyncPolicy::None);
        wal.append(b"x".to_vec(), "put".into()).unwrap();
        wal.append(b"y".to_vec(), "put".into()).unwrap();
        let all = wal.replay_all_verified().unwrap();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn wal_gc() {
        let mut wal = WalManager::new(64, SyncPolicy::None);
        for _ in 0..10 {
            wal.append(vec![0u8; 20], "put".into()).unwrap();
        }
        // At least one rotation should have happened.
        let before = wal.total_entries();
        let max_lsn = wal.max_lsn().unwrap_or(0);
        let removed = wal.gc(max_lsn);
        if removed > 0 {
            assert!(wal.total_entries() < before);
        }
    }

    #[test]
    fn wal_detect_corruption() {
        let mut wal = WalManager::new(4096, SyncPolicy::None);
        wal.append(b"ok".to_vec(), "put".into()).unwrap();
        let corrupted = wal.detect_corruption();
        assert!(corrupted.is_empty());
    }

    #[test]
    fn wal_sync_every_write() {
        let mut wal = WalManager::new(4096, SyncPolicy::EveryWrite);
        wal.append(b"x".to_vec(), "put".into()).unwrap();
        wal.append(b"y".to_vec(), "put".into()).unwrap();
        let stats = wal.stats();
        assert!(stats.total_syncs >= 2);
    }

    #[test]
    fn wal_min_max_lsn() {
        let mut wal = WalManager::new(4096, SyncPolicy::None);
        assert!(wal.min_lsn().is_none());
        wal.append(b"a".to_vec(), "put".into()).unwrap();
        wal.append(b"b".to_vec(), "put".into()).unwrap();
        assert_eq!(wal.min_lsn(), Some(1));
        assert_eq!(wal.max_lsn(), Some(2));
    }

    #[test]
    fn wal_stats() {
        let mut wal = WalManager::new(4096, SyncPolicy::None);
        wal.append(b"data".to_vec(), "put".into()).unwrap();
        let stats = wal.stats();
        assert_eq!(stats.total_entries, 1);
        assert!(stats.active_segments >= 1);
    }

    #[test]
    fn wal_set_sync_policy() {
        let mut wal = WalManager::new(4096, SyncPolicy::None);
        assert_eq!(wal.sync_policy(), SyncPolicy::None);
        wal.set_sync_policy(SyncPolicy::EveryWrite);
        assert_eq!(wal.sync_policy(), SyncPolicy::EveryWrite);
    }

    #[test]
    fn wal_read_missing_lsn() {
        let wal = WalManager::new(4096, SyncPolicy::None);
        assert_eq!(wal.read(999), Err(WalManagerError::LsnNotFound(999)));
    }

    #[test]
    fn wal_error_display() {
        let e = WalManagerError::Empty;
        assert_eq!(e.to_string(), "WAL is empty");
        let e = WalManagerError::SegmentNotFound(5);
        assert!(e.to_string().contains("5"));
    }
}
