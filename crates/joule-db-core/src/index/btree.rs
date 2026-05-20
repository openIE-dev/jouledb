//! BTree-based index implementation
//!
//! Provides an ordered index using BTreeMap for efficient range queries.

use std::collections::BTreeMap;
use std::sync::RwLock;

use super::traits::{
    Bound, Index, IndexEntry, IndexIterator, OrderedIndex, ScanDirection, UniqueIndex,
};
use crate::error::IndexError;

/// BTree-based ordered index
///
/// Uses a BTreeMap for ordered key-value storage with efficient range queries.
pub struct BTreeIndex {
    data: RwLock<BTreeMap<Vec<u8>, Vec<u8>>>,
}

impl BTreeIndex {
    /// Create a new empty BTree index
    pub fn new() -> Self {
        Self {
            data: RwLock::new(BTreeMap::new()),
        }
    }
}

impl Default for BTreeIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl Index for BTreeIndex {
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, IndexError> {
        let data = self.data.read().map_err(|_| IndexError::Corrupted {
            reason: "lock poisoned".to_string(),
        })?;
        Ok(data.get(key).cloned())
    }

    fn insert(&mut self, key: &[u8], value: &[u8]) -> Result<(), IndexError> {
        let mut data = self.data.write().map_err(|_| IndexError::Corrupted {
            reason: "lock poisoned".to_string(),
        })?;
        data.insert(key.to_vec(), value.to_vec());
        Ok(())
    }

    fn delete(&mut self, key: &[u8]) -> Result<bool, IndexError> {
        let mut data = self.data.write().map_err(|_| IndexError::Corrupted {
            reason: "lock poisoned".to_string(),
        })?;
        Ok(data.remove(key).is_some())
    }

    fn range(
        &self,
        start: Bound<&[u8]>,
        end: Bound<&[u8]>,
        direction: ScanDirection,
    ) -> Result<Box<dyn IndexIterator + '_>, IndexError> {
        let data = self.data.read().map_err(|_| IndexError::Corrupted {
            reason: "lock poisoned".to_string(),
        })?;

        // Collect entries in the range
        let entries: Vec<IndexEntry> = data
            .iter()
            .filter(|(k, _)| {
                let key = k.as_slice();
                let start_ok = match &start {
                    Bound::Included(s) => key >= *s,
                    Bound::Excluded(s) => key > *s,
                    Bound::Unbounded => true,
                };
                let end_ok = match &end {
                    Bound::Included(e) => key <= *e,
                    Bound::Excluded(e) => key < *e,
                    Bound::Unbounded => true,
                };
                start_ok && end_ok
            })
            .map(|(k, v)| IndexEntry::new(k.clone(), v.clone()))
            .collect();

        let entries = match direction {
            ScanDirection::Forward => entries,
            ScanDirection::Backward => entries.into_iter().rev().collect(),
        };

        Ok(Box::new(VecIterator::new(entries)))
    }
}

impl OrderedIndex for BTreeIndex {
    fn min(&self) -> Result<Option<IndexEntry>, IndexError> {
        let data = self.data.read().map_err(|_| IndexError::Corrupted {
            reason: "lock poisoned".to_string(),
        })?;
        Ok(data
            .iter()
            .next()
            .map(|(k, v)| IndexEntry::new(k.clone(), v.clone())))
    }

    fn max(&self) -> Result<Option<IndexEntry>, IndexError> {
        let data = self.data.read().map_err(|_| IndexError::Corrupted {
            reason: "lock poisoned".to_string(),
        })?;
        Ok(data
            .iter()
            .next_back()
            .map(|(k, v)| IndexEntry::new(k.clone(), v.clone())))
    }

    fn at_rank(&self, rank: usize) -> Result<Option<IndexEntry>, IndexError> {
        let data = self.data.read().map_err(|_| IndexError::Corrupted {
            reason: "lock poisoned".to_string(),
        })?;
        Ok(data
            .iter()
            .nth(rank)
            .map(|(k, v)| IndexEntry::new(k.clone(), v.clone())))
    }

    fn rank_of(&self, key: &[u8]) -> Result<Option<usize>, IndexError> {
        let data = self.data.read().map_err(|_| IndexError::Corrupted {
            reason: "lock poisoned".to_string(),
        })?;

        if !data.contains_key(key) {
            return Ok(None);
        }

        let rank = data.keys().take_while(|k| k.as_slice() < key).count();
        Ok(Some(rank))
    }
}

/// Hash-based unique index
///
/// Uses a HashMap with uniqueness enforcement on insert.
pub struct HashUniqueIndex {
    data: RwLock<std::collections::HashMap<Vec<u8>, Vec<u8>>>,
}

impl HashUniqueIndex {
    /// Create a new empty hash unique index
    pub fn new() -> Self {
        Self {
            data: RwLock::new(std::collections::HashMap::new()),
        }
    }
}

impl Default for HashUniqueIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl Index for HashUniqueIndex {
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, IndexError> {
        let data = self.data.read().map_err(|_| IndexError::Corrupted {
            reason: "lock poisoned".to_string(),
        })?;
        Ok(data.get(key).cloned())
    }

    fn insert(&mut self, key: &[u8], value: &[u8]) -> Result<(), IndexError> {
        let mut data = self.data.write().map_err(|_| IndexError::Corrupted {
            reason: "lock poisoned".to_string(),
        })?;
        data.insert(key.to_vec(), value.to_vec());
        Ok(())
    }

    fn delete(&mut self, key: &[u8]) -> Result<bool, IndexError> {
        let mut data = self.data.write().map_err(|_| IndexError::Corrupted {
            reason: "lock poisoned".to_string(),
        })?;
        Ok(data.remove(key).is_some())
    }

    fn range(
        &self,
        _start: Bound<&[u8]>,
        _end: Bound<&[u8]>,
        _direction: ScanDirection,
    ) -> Result<Box<dyn IndexIterator + '_>, IndexError> {
        // Hash index doesn't support efficient range queries
        // Return all entries (inefficient but correct)
        let data = self.data.read().map_err(|_| IndexError::Corrupted {
            reason: "lock poisoned".to_string(),
        })?;

        let entries: Vec<IndexEntry> = data
            .iter()
            .map(|(k, v)| IndexEntry::new(k.clone(), v.clone()))
            .collect();

        Ok(Box::new(VecIterator::new(entries)))
    }
}

impl UniqueIndex for HashUniqueIndex {
    fn insert_unique(&mut self, key: &[u8], value: &[u8]) -> Result<(), IndexError> {
        let mut data = self.data.write().map_err(|_| IndexError::Corrupted {
            reason: "lock poisoned".to_string(),
        })?;

        if data.contains_key(key) {
            return Err(IndexError::DuplicateKey { key: key.to_vec() });
        }

        data.insert(key.to_vec(), value.to_vec());
        Ok(())
    }
}

/// BTree index with uniqueness enforcement
///
/// Combines ordered storage with unique key constraint.
pub struct BTreeUniqueIndex {
    inner: BTreeIndex,
}

impl BTreeUniqueIndex {
    /// Create a new empty unique BTree index
    pub fn new() -> Self {
        Self {
            inner: BTreeIndex::new(),
        }
    }
}

impl Default for BTreeUniqueIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl Index for BTreeUniqueIndex {
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, IndexError> {
        self.inner.get(key)
    }

    fn insert(&mut self, key: &[u8], value: &[u8]) -> Result<(), IndexError> {
        self.inner.insert(key, value)
    }

    fn delete(&mut self, key: &[u8]) -> Result<bool, IndexError> {
        self.inner.delete(key)
    }

    fn range(
        &self,
        start: Bound<&[u8]>,
        end: Bound<&[u8]>,
        direction: ScanDirection,
    ) -> Result<Box<dyn IndexIterator + '_>, IndexError> {
        self.inner.range(start, end, direction)
    }
}

impl OrderedIndex for BTreeUniqueIndex {
    fn min(&self) -> Result<Option<IndexEntry>, IndexError> {
        self.inner.min()
    }

    fn max(&self) -> Result<Option<IndexEntry>, IndexError> {
        self.inner.max()
    }

    fn at_rank(&self, rank: usize) -> Result<Option<IndexEntry>, IndexError> {
        self.inner.at_rank(rank)
    }

    fn rank_of(&self, key: &[u8]) -> Result<Option<usize>, IndexError> {
        self.inner.rank_of(key)
    }
}

impl UniqueIndex for BTreeUniqueIndex {
    fn insert_unique(&mut self, key: &[u8], value: &[u8]) -> Result<(), IndexError> {
        // Check if key exists
        if self.inner.get(key)?.is_some() {
            return Err(IndexError::DuplicateKey { key: key.to_vec() });
        }
        self.inner.insert(key, value)
    }
}

/// Simple vector-based iterator for index results
struct VecIterator {
    entries: Vec<IndexEntry>,
    position: usize,
}

impl VecIterator {
    fn new(entries: Vec<IndexEntry>) -> Self {
        Self {
            entries,
            position: 0,
        }
    }
}

impl Iterator for VecIterator {
    type Item = Result<IndexEntry, IndexError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.entries.len() {
            return None;
        }
        let entry = self.entries[self.position].clone();
        self.position += 1;
        Some(Ok(entry))
    }
}

impl IndexIterator for VecIterator {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_btree_index_basic() {
        let mut index = BTreeIndex::new();

        index.insert(b"key1", b"value1").unwrap();
        index.insert(b"key2", b"value2").unwrap();
        index.insert(b"key3", b"value3").unwrap();

        assert_eq!(index.get(b"key1").unwrap(), Some(b"value1".to_vec()));
        assert_eq!(index.get(b"key2").unwrap(), Some(b"value2".to_vec()));
        assert_eq!(index.get(b"nonexistent").unwrap(), None);
    }

    #[test]
    fn test_btree_index_delete() {
        let mut index = BTreeIndex::new();

        index.insert(b"key1", b"value1").unwrap();
        assert!(index.delete(b"key1").unwrap());
        assert!(!index.delete(b"key1").unwrap()); // Already deleted
        assert_eq!(index.get(b"key1").unwrap(), None);
    }

    #[test]
    fn test_btree_index_ordered() {
        let mut index = BTreeIndex::new();

        index.insert(b"c", b"3").unwrap();
        index.insert(b"a", b"1").unwrap();
        index.insert(b"b", b"2").unwrap();

        // Min should be 'a'
        let min = index.min().unwrap().unwrap();
        assert_eq!(min.key, b"a".to_vec());

        // Max should be 'c'
        let max = index.max().unwrap().unwrap();
        assert_eq!(max.key, b"c".to_vec());

        // Rank of 'b' should be 1 (second element)
        assert_eq!(index.rank_of(b"b").unwrap(), Some(1));

        // Element at rank 1 should be 'b'
        let at_1 = index.at_rank(1).unwrap().unwrap();
        assert_eq!(at_1.key, b"b".to_vec());
    }

    #[test]
    fn test_btree_index_range() {
        let mut index = BTreeIndex::new();

        for i in 0u8..10 {
            index.insert(&[i], &[i * 10]).unwrap();
        }

        // Range [3, 7]
        let mut iter = index
            .range(
                Bound::Included(&[3]),
                Bound::Included(&[7]),
                ScanDirection::Forward,
            )
            .unwrap();

        let entries = iter.collect_all().unwrap();
        assert_eq!(entries.len(), 5);
        assert_eq!(entries[0].key, vec![3]);
        assert_eq!(entries[4].key, vec![7]);
    }

    #[test]
    fn test_btree_index_range_backward() {
        let mut index = BTreeIndex::new();

        for i in 0u8..10 {
            index.insert(&[i], &[i * 10]).unwrap();
        }

        let mut iter = index
            .range(
                Bound::Included(&[3]),
                Bound::Included(&[7]),
                ScanDirection::Backward,
            )
            .unwrap();

        let entries = iter.collect_all().unwrap();
        assert_eq!(entries.len(), 5);
        assert_eq!(entries[0].key, vec![7]); // Backward starts from end
        assert_eq!(entries[4].key, vec![3]);
    }

    #[test]
    fn test_unique_index_enforces_uniqueness() {
        let mut index = HashUniqueIndex::new();

        index.insert_unique(b"key1", b"value1").unwrap();

        // Should fail on duplicate
        let result = index.insert_unique(b"key1", b"value2");
        assert!(matches!(result, Err(IndexError::DuplicateKey { .. })));

        // Original value should be unchanged
        assert_eq!(index.get(b"key1").unwrap(), Some(b"value1".to_vec()));
    }

    #[test]
    fn test_btree_unique_index() {
        let mut index = BTreeUniqueIndex::new();

        index.insert_unique(b"b", b"2").unwrap();
        index.insert_unique(b"a", b"1").unwrap();
        index.insert_unique(b"c", b"3").unwrap();

        // Should maintain order
        let min = index.min().unwrap().unwrap();
        assert_eq!(min.key, b"a".to_vec());

        // Should enforce uniqueness
        let result = index.insert_unique(b"a", b"new");
        assert!(matches!(result, Err(IndexError::DuplicateKey { .. })));
    }

    #[test]
    fn test_index_count() {
        let mut index = BTreeIndex::new();

        for i in 0u8..10 {
            index.insert(&[i], &[i]).unwrap();
        }

        assert_eq!(index.count().unwrap(), 10);

        // Count range
        assert_eq!(
            index
                .count_range(Bound::Included(&[3]), Bound::Excluded(&[7]),)
                .unwrap(),
            4
        ); // 3, 4, 5, 6
    }

    #[test]
    fn test_index_contains() {
        let mut index = BTreeIndex::new();

        index.insert(b"key1", b"value1").unwrap();

        assert!(index.contains(b"key1").unwrap());
        assert!(!index.contains(b"nonexistent").unwrap());
    }

    #[test]
    fn test_empty_index() {
        let index = BTreeIndex::new();

        assert_eq!(index.min().unwrap(), None);
        assert_eq!(index.max().unwrap(), None);
        assert_eq!(index.count().unwrap(), 0);
    }
}
