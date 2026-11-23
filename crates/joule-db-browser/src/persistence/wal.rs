//! Write-Ahead Log (WAL) for crash recovery

use joule_db_core::storage::PageId;
use std::collections::VecDeque;

/// WAL entry for crash recovery
#[derive(Debug, Clone)]
pub struct WalEntry {
    /// Sequence number
    pub seq: u64,
    /// Page ID
    pub page_id: PageId,
    /// Page data
    pub data: Vec<u8>,
    /// Timestamp (milliseconds since epoch)
    pub timestamp: u64,
    /// Committed flag
    pub committed: bool,
}

impl WalEntry {
    /// Create a new WAL entry
    pub fn new(seq: u64, page_id: PageId, data: Vec<u8>) -> Self {
        Self {
            seq,
            page_id,
            data,
            timestamp: Self::current_timestamp(),
            committed: false,
        }
    }

    /// Get current timestamp
    fn current_timestamp() -> u64 {
        #[cfg(target_arch = "wasm32")]
        {
            js_sys::Date::now() as u64
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0)
        }
    }

    /// Serialize WAL entry to bytes
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(32 + self.data.len());

        // Header: seq (8) + page_id (8) + timestamp (8) + committed (1) + data_len (4)
        buf.extend_from_slice(&self.seq.to_le_bytes());
        buf.extend_from_slice(&self.page_id.to_le_bytes());
        buf.extend_from_slice(&self.timestamp.to_le_bytes());
        buf.push(if self.committed { 1 } else { 0 });
        buf.extend_from_slice(&(self.data.len() as u32).to_le_bytes());
        buf.extend_from_slice(&self.data);

        buf
    }

    /// Deserialize WAL entry from bytes
    pub fn deserialize(data: &[u8]) -> Option<(Self, usize)> {
        if data.len() < 29 {
            return None;
        }

        let seq = u64::from_le_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
        ]);
        let page_id = u64::from_le_bytes([
            data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15],
        ]);
        let timestamp = u64::from_le_bytes([
            data[16], data[17], data[18], data[19], data[20], data[21], data[22], data[23],
        ]);
        let committed = data[24] != 0;
        let data_len = u32::from_le_bytes([data[25], data[26], data[27], data[28]]) as usize;

        if data.len() < 29 + data_len {
            return None;
        }

        let entry_data = data[29..29 + data_len].to_vec();
        let entry = Self {
            seq,
            page_id,
            timestamp,
            committed,
            data: entry_data,
        };

        Some((entry, 29 + data_len))
    }
}

/// WAL manager for handling write-ahead logging
#[derive(Debug)]
pub struct WalManager {
    /// Pending entries not yet committed
    entries: VecDeque<WalEntry>,
    /// Next sequence number
    next_seq: u64,
    /// Maximum entries before forced flush
    max_entries: usize,
}

impl WalManager {
    /// Create a new WAL manager
    pub fn new() -> Self {
        Self {
            entries: VecDeque::new(),
            next_seq: 1,
            max_entries: 1000,
        }
    }

    /// Create with custom max entries
    pub fn with_max_entries(max_entries: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            next_seq: 1,
            max_entries,
        }
    }

    /// Append a new entry to the WAL
    pub fn append(&mut self, page_id: PageId, data: Vec<u8>) -> u64 {
        let seq = self.next_seq;
        self.next_seq += 1;

        let entry = WalEntry::new(seq, page_id, data);
        self.entries.push_back(entry);

        seq
    }

    /// Mark an entry as committed
    pub fn commit(&mut self, seq: u64) {
        for entry in &mut self.entries {
            if entry.seq == seq {
                entry.committed = true;
                break;
            }
        }
    }

    /// Get all pending (uncommitted) entries
    pub fn pending(&self) -> impl Iterator<Item = &WalEntry> {
        self.entries.iter().filter(|e| !e.committed)
    }

    /// Get all committed entries
    pub fn committed(&self) -> impl Iterator<Item = &WalEntry> {
        self.entries.iter().filter(|e| e.committed)
    }

    /// Remove committed entries up to a sequence number
    pub fn truncate(&mut self, up_to_seq: u64) {
        while let Some(entry) = self.entries.front() {
            if entry.seq <= up_to_seq && entry.committed {
                self.entries.pop_front();
            } else {
                break;
            }
        }
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Check if WAL needs flushing
    pub fn needs_flush(&self) -> bool {
        self.entries.len() >= self.max_entries
    }

    /// Get pending entry count
    pub fn pending_count(&self) -> usize {
        self.entries.iter().filter(|e| !e.committed).count()
    }

    /// Serialize all entries
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        for entry in &self.entries {
            buf.extend_from_slice(&entry.serialize());
        }
        buf
    }

    /// Deserialize entries from bytes
    pub fn deserialize(data: &[u8]) -> Self {
        let mut manager = Self::new();
        let mut offset = 0;

        while offset < data.len() {
            if let Some((entry, consumed)) = WalEntry::deserialize(&data[offset..]) {
                if entry.seq >= manager.next_seq {
                    manager.next_seq = entry.seq + 1;
                }
                manager.entries.push_back(entry);
                offset += consumed;
            } else {
                break;
            }
        }

        manager
    }
}

impl Default for WalManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wal_entry_roundtrip() {
        let entry = WalEntry::new(1, 42, b"test data".to_vec());
        let serialized = entry.serialize();
        let (deserialized, _) = WalEntry::deserialize(&serialized).unwrap();

        assert_eq!(deserialized.seq, entry.seq);
        assert_eq!(deserialized.page_id, entry.page_id);
        assert_eq!(deserialized.data, entry.data);
    }

    #[test]
    fn test_wal_manager() {
        let mut wal = WalManager::new();

        let seq1 = wal.append(1, b"page1".to_vec());
        let seq2 = wal.append(2, b"page2".to_vec());

        assert_eq!(wal.pending_count(), 2);

        wal.commit(seq1);
        assert_eq!(wal.pending_count(), 1);

        wal.truncate(seq1);
        assert_eq!(wal.pending_count(), 1);
    }
}
