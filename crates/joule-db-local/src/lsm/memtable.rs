//! In-memory sorted table (MemTable) for the LSM-Tree engine.
//!
//! Uses a BTreeMap for sorted key-value storage with tombstone support.

use std::collections::BTreeMap;
use std::ops::Bound;

/// Entry in a MemTable: Some(value) for data, None for tombstone (delete).
pub type MemEntry = Option<Vec<u8>>;

/// In-memory sorted table backed by a BTreeMap.
pub struct MemTable {
    data: BTreeMap<Vec<u8>, MemEntry>,
    size_bytes: usize,
}

impl MemTable {
    /// Create an empty MemTable.
    pub fn new() -> Self {
        Self {
            data: BTreeMap::new(),
            size_bytes: 0,
        }
    }

    /// Get a value by key. Returns:
    /// - Some(Some(value)) if key exists with data
    /// - Some(None) if key has a tombstone (was deleted)
    /// - None if key is not in this memtable
    pub fn get(&self, key: &[u8]) -> Option<&MemEntry> {
        self.data.get(key)
    }

    /// Insert a key-value pair.
    pub fn put(&mut self, key: Vec<u8>, value: Vec<u8>) {
        let entry_size = key.len() + value.len() + 16; // approximate overhead
        if let Some(old) = self.data.insert(key, Some(value)) {
            // Subtract old entry size
            let old_size = old.as_ref().map_or(8, |v| v.len() + 8);
            self.size_bytes = self.size_bytes.saturating_sub(old_size);
        }
        self.size_bytes += entry_size;
    }

    /// Mark a key as deleted (tombstone).
    pub fn delete(&mut self, key: Vec<u8>) {
        let entry_size = key.len() + 8;
        if let Some(old) = self.data.insert(key, None) {
            let old_size = old.as_ref().map_or(8, |v| v.len() + 8);
            self.size_bytes = self.size_bytes.saturating_sub(old_size);
        }
        self.size_bytes += entry_size;
    }

    /// Iterate over all entries in sorted order.
    pub fn iter(&self) -> impl Iterator<Item = (&Vec<u8>, &MemEntry)> {
        self.data.iter()
    }

    /// Range scan over entries.
    pub fn range(
        &self,
        start: Bound<&[u8]>,
        end: Bound<&[u8]>,
    ) -> impl Iterator<Item = (&Vec<u8>, &MemEntry)> {
        let start = match start {
            Bound::Included(k) => Bound::Included(k.to_vec()),
            Bound::Excluded(k) => Bound::Excluded(k.to_vec()),
            Bound::Unbounded => Bound::Unbounded,
        };
        let end = match end {
            Bound::Included(k) => Bound::Included(k.to_vec()),
            Bound::Excluded(k) => Bound::Excluded(k.to_vec()),
            Bound::Unbounded => Bound::Unbounded,
        };
        self.data.range((start, end))
    }

    /// Number of entries (including tombstones).
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the memtable is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Approximate memory usage in bytes.
    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    /// Drain all entries, returning ownership. Resets the memtable.
    pub fn drain(&mut self) -> BTreeMap<Vec<u8>, MemEntry> {
        self.size_bytes = 0;
        std::mem::take(&mut self.data)
    }

    /// Consume the memtable and return all entries.
    pub fn drain_owned(mut self) -> BTreeMap<Vec<u8>, MemEntry> {
        self.drain()
    }
}

impl Default for MemTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memtable_put_get() {
        let mut mt = MemTable::new();
        mt.put(b"key1".to_vec(), b"value1".to_vec());
        mt.put(b"key2".to_vec(), b"value2".to_vec());

        assert_eq!(mt.get(b"key1"), Some(&Some(b"value1".to_vec())));
        assert_eq!(mt.get(b"key2"), Some(&Some(b"value2".to_vec())));
        assert_eq!(mt.get(b"key3"), None);
    }

    #[test]
    fn test_memtable_delete_tombstone() {
        let mut mt = MemTable::new();
        mt.put(b"key1".to_vec(), b"value1".to_vec());
        mt.delete(b"key1".to_vec());

        // Should return tombstone (Some(None))
        assert_eq!(mt.get(b"key1"), Some(&None));
    }

    #[test]
    fn test_memtable_overwrite() {
        let mut mt = MemTable::new();
        mt.put(b"key1".to_vec(), b"old".to_vec());
        mt.put(b"key1".to_vec(), b"new".to_vec());

        assert_eq!(mt.get(b"key1"), Some(&Some(b"new".to_vec())));
        assert_eq!(mt.len(), 1);
    }

    #[test]
    fn test_memtable_iter_sorted() {
        let mut mt = MemTable::new();
        mt.put(b"c".to_vec(), b"3".to_vec());
        mt.put(b"a".to_vec(), b"1".to_vec());
        mt.put(b"b".to_vec(), b"2".to_vec());

        let keys: Vec<&[u8]> = mt.iter().map(|(k, _)| k.as_slice()).collect();
        assert_eq!(
            keys,
            vec![b"a".as_slice(), b"b".as_slice(), b"c".as_slice()]
        );
    }

    #[test]
    fn test_memtable_range() {
        let mut mt = MemTable::new();
        for i in 0u8..10 {
            mt.put(vec![i], vec![i * 10]);
        }

        let range: Vec<u8> = mt
            .range(Bound::Included(&[3]), Bound::Excluded(&[7]))
            .map(|(k, _)| k[0])
            .collect();
        assert_eq!(range, vec![3, 4, 5, 6]);
    }

    #[test]
    fn test_memtable_size_tracking() {
        let mut mt = MemTable::new();
        assert_eq!(mt.size_bytes(), 0);
        mt.put(b"key".to_vec(), b"value".to_vec());
        assert!(mt.size_bytes() > 0);
    }

    #[test]
    fn test_memtable_drain() {
        let mut mt = MemTable::new();
        mt.put(b"a".to_vec(), b"1".to_vec());
        mt.put(b"b".to_vec(), b"2".to_vec());

        let drained = mt.drain();
        assert_eq!(drained.len(), 2);
        assert!(mt.is_empty());
        assert_eq!(mt.size_bytes(), 0);
    }
}
