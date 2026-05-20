//! Write-ahead log — append-only log entries with LSN (log sequence number),
//! log segments, truncation (discard before LSN), replay from LSN, CRC32
//! checksum per entry, and compaction.

use std::collections::BTreeMap;

// ── Errors ───────────────────────────────────────────────────────────────────

/// Errors returned by WAL operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WalError {
    /// LSN not found in the log.
    LsnNotFound(u64),
    /// Checksum mismatch (data corruption).
    ChecksumMismatch { expected: u32, actual: u32 },
    /// Segment is full.
    SegmentFull,
    /// Log is empty.
    Empty,
}

impl std::fmt::Display for WalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LsnNotFound(lsn) => write!(f, "LSN {lsn} not found"),
            Self::ChecksumMismatch { expected, actual } => {
                write!(f, "checksum mismatch: expected {expected:#010x}, got {actual:#010x}")
            }
            Self::SegmentFull => write!(f, "segment full"),
            Self::Empty => write!(f, "log is empty"),
        }
    }
}

// ── CRC32 ────────────────────────────────────────────────────────────────────

/// Compute CRC32 (Castagnoli / CRC-32C) using a simple table-based approach.
fn crc32(data: &[u8]) -> u32 {
    // CRC-32/ISO-HDLC polynomial (used by zlib, PNG, etc.)
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

// ── Log Entry ────────────────────────────────────────────────────────────────

/// A single WAL entry.
#[derive(Debug, Clone)]
pub struct LogEntry {
    /// Log sequence number (unique, monotonically increasing).
    pub lsn: u64,
    /// The payload data.
    pub data: Vec<u8>,
    /// CRC32 checksum of the data.
    pub checksum: u32,
    /// Optional operation type tag for filtering.
    pub op_type: Option<String>,
}

impl LogEntry {
    /// Verify the entry's checksum.
    pub fn verify(&self) -> bool {
        crc32(&self.data) == self.checksum
    }

    /// Data length in bytes.
    pub fn data_len(&self) -> usize {
        self.data.len()
    }
}

// ── Segment ──────────────────────────────────────────────────────────────────

/// A log segment containing a contiguous range of entries.
#[derive(Debug, Clone)]
pub struct Segment {
    id: u64,
    entries: Vec<LogEntry>,
    max_entries: usize,
    total_bytes: usize,
}

impl Segment {
    fn new(id: u64, max_entries: usize) -> Self {
        Self {
            id,
            entries: Vec::new(),
            max_entries,
            total_bytes: 0,
        }
    }

    fn is_full(&self) -> bool {
        self.entries.len() >= self.max_entries
    }

    fn first_lsn(&self) -> Option<u64> {
        self.entries.first().map(|e| e.lsn)
    }

    fn last_lsn(&self) -> Option<u64> {
        self.entries.last().map(|e| e.lsn)
    }

    fn append(&mut self, entry: LogEntry) -> Result<(), WalError> {
        if self.is_full() {
            return Err(WalError::SegmentFull);
        }
        self.total_bytes += entry.data.len();
        self.entries.push(entry);
        Ok(())
    }
}

// ── WAL Statistics ───────────────────────────────────────────────────────────

/// Statistics for the write-ahead log.
#[derive(Debug, Clone, Default)]
pub struct WalStats {
    pub total_entries: usize,
    pub total_bytes: usize,
    pub total_segments: usize,
    pub first_lsn: Option<u64>,
    pub last_lsn: Option<u64>,
    pub total_appends: u64,
    pub total_truncations: u64,
    pub total_compactions: u64,
}

// ── WriteAheadLog ────────────────────────────────────────────────────────────

/// Append-only write-ahead log with segmented storage, checksums, truncation,
/// replay, and compaction.
pub struct WriteAheadLog {
    segments: BTreeMap<u64, Segment>,
    next_lsn: u64,
    next_segment_id: u64,
    entries_per_segment: usize,
    total_appends: u64,
    total_truncations: u64,
    total_compactions: u64,
}

impl WriteAheadLog {
    /// Create a new WAL with the given entries-per-segment limit.
    pub fn new(entries_per_segment: usize) -> Self {
        assert!(entries_per_segment > 0, "entries per segment must be > 0");
        let mut segments = BTreeMap::new();
        segments.insert(0, Segment::new(0, entries_per_segment));
        Self {
            segments,
            next_lsn: 1,
            next_segment_id: 1,
            entries_per_segment,
            total_appends: 0,
            total_truncations: 0,
            total_compactions: 0,
        }
    }

    /// Append data to the log, returning the assigned LSN.
    pub fn append(&mut self, data: Vec<u8>) -> u64 {
        self.append_typed(data, None)
    }

    /// Append data with an operation type tag.
    pub fn append_typed(&mut self, data: Vec<u8>, op_type: Option<String>) -> u64 {
        let lsn = self.next_lsn;
        self.next_lsn += 1;
        let checksum = crc32(&data);

        let entry = LogEntry {
            lsn,
            data,
            checksum,
            op_type,
        };

        // Try to append to the active (last) segment.
        let active_id = *self.segments.keys().next_back().unwrap();
        let active = self.segments.get_mut(&active_id).unwrap();
        if active.is_full() {
            // Create a new segment.
            let new_id = self.next_segment_id;
            self.next_segment_id += 1;
            let mut new_seg = Segment::new(new_id, self.entries_per_segment);
            new_seg.append(entry).unwrap();
            self.segments.insert(new_id, new_seg);
        } else {
            active.append(entry).unwrap();
        }

        self.total_appends += 1;
        lsn
    }

    /// Read a specific entry by LSN.
    pub fn read(&self, lsn: u64) -> Result<&LogEntry, WalError> {
        for segment in self.segments.values() {
            if let Some(first) = segment.first_lsn() {
                if let Some(last) = segment.last_lsn() {
                    if lsn >= first && lsn <= last {
                        for entry in &segment.entries {
                            if entry.lsn == lsn {
                                return Ok(entry);
                            }
                        }
                    }
                }
            }
        }
        Err(WalError::LsnNotFound(lsn))
    }

    /// Verify a specific entry's checksum.
    pub fn verify(&self, lsn: u64) -> Result<bool, WalError> {
        let entry = self.read(lsn)?;
        Ok(entry.verify())
    }

    /// Replay all entries from the given LSN (inclusive) to the end.
    pub fn replay_from(&self, from_lsn: u64) -> Vec<&LogEntry> {
        let mut result = Vec::new();
        for segment in self.segments.values() {
            for entry in &segment.entries {
                if entry.lsn >= from_lsn {
                    result.push(entry);
                }
            }
        }
        result
    }

    /// Replay all entries.
    pub fn replay_all(&self) -> Vec<&LogEntry> {
        let mut result = Vec::new();
        for segment in self.segments.values() {
            for entry in &segment.entries {
                result.push(entry);
            }
        }
        result
    }

    /// Replay entries with a specific operation type.
    pub fn replay_by_type(&self, op_type: &str) -> Vec<&LogEntry> {
        let mut result = Vec::new();
        for segment in self.segments.values() {
            for entry in &segment.entries {
                if entry.op_type.as_deref() == Some(op_type) {
                    result.push(entry);
                }
            }
        }
        result
    }

    /// Truncate: discard all entries with LSN < the given LSN.
    /// Removes entire segments that fall before the truncation point.
    pub fn truncate_before(&mut self, lsn: u64) -> usize {
        let mut removed = 0;
        let segments_to_remove: Vec<u64> = self
            .segments
            .iter()
            .filter(|(_, seg)| seg.last_lsn().is_some_and(|last| last < lsn))
            .map(|(id, _)| *id)
            .collect();

        for seg_id in &segments_to_remove {
            if let Some(seg) = self.segments.remove(seg_id) {
                removed += seg.entries.len();
            }
        }

        // For the first remaining segment, trim entries below the LSN.
        if let Some(first_seg) = self.segments.values_mut().next() {
            let before = first_seg.entries.len();
            first_seg.entries.retain(|e| e.lsn >= lsn);
            let after = first_seg.entries.len();
            removed += before - after;
            // Recalculate bytes.
            first_seg.total_bytes = first_seg.entries.iter().map(|e| e.data.len()).sum();
        }

        // Ensure at least one segment exists.
        if self.segments.is_empty() {
            let new_id = self.next_segment_id;
            self.next_segment_id += 1;
            self.segments
                .insert(new_id, Segment::new(new_id, self.entries_per_segment));
        }

        if removed > 0 {
            self.total_truncations += 1;
        }
        removed
    }

    /// Compact: keep only entries matching a predicate.
    pub fn compact<P>(&mut self, predicate: P) -> usize
    where
        P: Fn(&LogEntry) -> bool,
    {
        let mut removed = 0;
        for segment in self.segments.values_mut() {
            let before = segment.entries.len();
            segment.entries.retain(|e| predicate(e));
            removed += before - segment.entries.len();
            segment.total_bytes = segment.entries.iter().map(|e| e.data.len()).sum();
        }

        // Remove empty segments (keep at least one).
        let empty_ids: Vec<u64> = self
            .segments
            .iter()
            .filter(|(_, seg)| seg.entries.is_empty())
            .map(|(id, _)| *id)
            .collect();
        for id in empty_ids {
            if self.segments.len() > 1 {
                self.segments.remove(&id);
            }
        }

        if removed > 0 {
            self.total_compactions += 1;
        }
        removed
    }

    /// Total number of entries across all segments.
    pub fn len(&self) -> usize {
        self.segments.values().map(|s| s.entries.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Total data bytes across all entries.
    pub fn total_bytes(&self) -> usize {
        self.segments.values().map(|s| s.total_bytes).sum()
    }

    /// Number of segments.
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// First LSN in the log.
    pub fn first_lsn(&self) -> Option<u64> {
        self.segments
            .values()
            .find_map(|s| s.first_lsn())
    }

    /// Last LSN in the log.
    pub fn last_lsn(&self) -> Option<u64> {
        self.segments
            .values()
            .rev()
            .find_map(|s| s.last_lsn())
    }

    /// Next LSN that will be assigned.
    pub fn next_lsn(&self) -> u64 {
        self.next_lsn
    }

    /// Statistics snapshot.
    pub fn stats(&self) -> WalStats {
        WalStats {
            total_entries: self.len(),
            total_bytes: self.total_bytes(),
            total_segments: self.segments.len(),
            first_lsn: self.first_lsn(),
            last_lsn: self.last_lsn(),
            total_appends: self.total_appends,
            total_truncations: self.total_truncations,
            total_compactions: self.total_compactions,
        }
    }

    /// Verify all entries in the log. Returns (ok_count, corrupt_count).
    pub fn verify_all(&self) -> (usize, usize) {
        let mut ok = 0;
        let mut corrupt = 0;
        for segment in self.segments.values() {
            for entry in &segment.entries {
                if entry.verify() {
                    ok += 1;
                } else {
                    corrupt += 1;
                }
            }
        }
        (ok, corrupt)
    }

    /// Clear the entire log.
    pub fn clear(&mut self) {
        self.segments.clear();
        let new_id = self.next_segment_id;
        self.next_segment_id += 1;
        self.segments
            .insert(new_id, Segment::new(new_id, self.entries_per_segment));
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_append_and_read() {
        let mut wal = WriteAheadLog::new(100);
        let lsn = wal.append(b"hello".to_vec());
        let entry = wal.read(lsn).unwrap();
        assert_eq!(entry.data, b"hello");
        assert_eq!(entry.lsn, lsn);
    }

    #[test]
    fn test_lsn_monotonic() {
        let mut wal = WriteAheadLog::new(100);
        let l1 = wal.append(b"a".to_vec());
        let l2 = wal.append(b"b".to_vec());
        let l3 = wal.append(b"c".to_vec());
        assert!(l1 < l2);
        assert!(l2 < l3);
    }

    #[test]
    fn test_checksum_verify() {
        let mut wal = WriteAheadLog::new(100);
        let lsn = wal.append(b"data with checksum".to_vec());
        assert!(wal.verify(lsn).unwrap());
    }

    #[test]
    fn test_crc32_known() {
        // "hello" CRC32 = 0x3610a686
        let c = crc32(b"hello");
        assert_eq!(c, 0x3610a686);
    }

    #[test]
    fn test_replay_from() {
        let mut wal = WriteAheadLog::new(100);
        let l1 = wal.append(b"one".to_vec());
        let l2 = wal.append(b"two".to_vec());
        let _l3 = wal.append(b"three".to_vec());
        let entries = wal.replay_from(l2);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].lsn, l2);
        let _ = l1;
    }

    #[test]
    fn test_replay_all() {
        let mut wal = WriteAheadLog::new(100);
        wal.append(b"a".to_vec());
        wal.append(b"b".to_vec());
        wal.append(b"c".to_vec());
        let entries = wal.replay_all();
        assert_eq!(entries.len(), 3);
    }

    #[test]
    fn test_truncate_before() {
        let mut wal = WriteAheadLog::new(100);
        let _l1 = wal.append(b"old1".to_vec());
        let _l2 = wal.append(b"old2".to_vec());
        let l3 = wal.append(b"keep1".to_vec());
        let _l4 = wal.append(b"keep2".to_vec());
        let removed = wal.truncate_before(l3);
        assert_eq!(removed, 2);
        assert_eq!(wal.len(), 2);
        assert_eq!(wal.first_lsn(), Some(l3));
    }

    #[test]
    fn test_segment_rotation() {
        let mut wal = WriteAheadLog::new(3); // 3 entries per segment
        for i in 0..7 {
            wal.append(format!("entry{i}").into_bytes());
        }
        assert!(wal.segment_count() >= 3);
        assert_eq!(wal.len(), 7);
    }

    #[test]
    fn test_truncate_removes_segments() {
        let mut wal = WriteAheadLog::new(2);
        wal.append(b"a".to_vec());
        wal.append(b"b".to_vec());
        let l3 = wal.append(b"c".to_vec());
        wal.append(b"d".to_vec());
        wal.truncate_before(l3);
        // First segment (with a, b) should be removed.
        assert_eq!(wal.len(), 2);
    }

    #[test]
    fn test_compact() {
        let mut wal = WriteAheadLog::new(100);
        wal.append_typed(b"insert".to_vec(), Some("INSERT".into()));
        wal.append_typed(b"delete".to_vec(), Some("DELETE".into()));
        wal.append_typed(b"insert2".to_vec(), Some("INSERT".into()));
        // Keep only INSERTs.
        let removed = wal.compact(|e| e.op_type.as_deref() == Some("INSERT"));
        assert_eq!(removed, 1);
        assert_eq!(wal.len(), 2);
    }

    #[test]
    fn test_replay_by_type() {
        let mut wal = WriteAheadLog::new(100);
        wal.append_typed(b"i1".to_vec(), Some("INSERT".into()));
        wal.append_typed(b"d1".to_vec(), Some("DELETE".into()));
        wal.append_typed(b"i2".to_vec(), Some("INSERT".into()));
        let inserts = wal.replay_by_type("INSERT");
        assert_eq!(inserts.len(), 2);
    }

    #[test]
    fn test_total_bytes() {
        let mut wal = WriteAheadLog::new(100);
        wal.append(b"12345".to_vec());
        wal.append(b"abcde".to_vec());
        assert_eq!(wal.total_bytes(), 10);
    }

    #[test]
    fn test_first_last_lsn() {
        let mut wal = WriteAheadLog::new(100);
        assert_eq!(wal.first_lsn(), None);
        let l1 = wal.append(b"first".to_vec());
        let l2 = wal.append(b"second".to_vec());
        assert_eq!(wal.first_lsn(), Some(l1));
        assert_eq!(wal.last_lsn(), Some(l2));
    }

    #[test]
    fn test_lsn_not_found() {
        let wal = WriteAheadLog::new(100);
        assert!(matches!(wal.read(999), Err(WalError::LsnNotFound(999))));
    }

    #[test]
    fn test_verify_all() {
        let mut wal = WriteAheadLog::new(100);
        wal.append(b"one".to_vec());
        wal.append(b"two".to_vec());
        let (ok, corrupt) = wal.verify_all();
        assert_eq!(ok, 2);
        assert_eq!(corrupt, 0);
    }

    #[test]
    fn test_clear() {
        let mut wal = WriteAheadLog::new(100);
        wal.append(b"data".to_vec());
        wal.append(b"more".to_vec());
        wal.clear();
        assert!(wal.is_empty());
        assert_eq!(wal.segment_count(), 1);
    }

    #[test]
    fn test_stats() {
        let mut wal = WriteAheadLog::new(5);
        for i in 0..10 {
            wal.append(format!("entry{i}").into_bytes());
        }
        let stats = wal.stats();
        assert_eq!(stats.total_entries, 10);
        assert_eq!(stats.total_appends, 10);
        assert!(stats.total_segments >= 2);
    }

    #[test]
    fn test_truncate_all_keeps_empty_segment() {
        let mut wal = WriteAheadLog::new(100);
        let l1 = wal.append(b"a".to_vec());
        wal.truncate_before(l1 + 1);
        assert!(wal.is_empty());
        assert_eq!(wal.segment_count(), 1);
    }

    #[test]
    fn test_compact_all() {
        let mut wal = WriteAheadLog::new(100);
        wal.append(b"a".to_vec());
        wal.append(b"b".to_vec());
        let removed = wal.compact(|_| false);
        assert_eq!(removed, 2);
        assert!(wal.is_empty());
    }
}
