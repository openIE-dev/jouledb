//! Index traits

use crate::error::IndexError;

/// Scan direction for range queries
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanDirection {
    /// Ascending order (smallest to largest)
    Forward,
    /// Descending order (largest to smallest)
    Backward,
}

/// Bound for range queries
#[derive(Debug, Clone)]
pub enum Bound<T> {
    /// Include this value in the range
    Included(T),
    /// Exclude this value from the range
    Excluded(T),
    /// No bound (extends to infinity)
    Unbounded,
}

impl<T> Bound<T> {
    /// Map the inner value
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> Bound<U> {
        match self {
            Bound::Included(v) => Bound::Included(f(v)),
            Bound::Excluded(v) => Bound::Excluded(f(v)),
            Bound::Unbounded => Bound::Unbounded,
        }
    }

    /// Get reference to inner value if present
    pub fn as_ref(&self) -> Bound<&T> {
        match self {
            Bound::Included(v) => Bound::Included(v),
            Bound::Excluded(v) => Bound::Excluded(v),
            Bound::Unbounded => Bound::Unbounded,
        }
    }
}

/// An entry returned from index scans
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexEntry {
    /// The key
    pub key: Vec<u8>,
    /// The value
    pub value: Vec<u8>,
}

impl IndexEntry {
    /// Create a new index entry
    pub fn new(key: Vec<u8>, value: Vec<u8>) -> Self {
        Self { key, value }
    }
}

/// Iterator over index entries
pub trait IndexIterator: Iterator<Item = Result<IndexEntry, IndexError>> + Send {
    /// Collect all remaining entries into a vector
    fn collect_all(&mut self) -> Result<Vec<IndexEntry>, IndexError> {
        let mut results = Vec::new();
        while let Some(result) = self.next() {
            results.push(result?);
        }
        Ok(results)
    }
}

/// Core index trait
///
/// An index maps keys to values with efficient lookup and optional range queries.
pub trait Index: Send + Sync {
    /// Point lookup - get value for a key
    ///
    /// Returns None if the key doesn't exist.
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, IndexError>;

    /// Insert or update a key-value pair
    ///
    /// If the key exists, the value is updated.
    fn insert(&mut self, key: &[u8], value: &[u8]) -> Result<(), IndexError>;

    /// Delete a key
    ///
    /// Returns true if the key existed and was deleted.
    fn delete(&mut self, key: &[u8]) -> Result<bool, IndexError>;

    /// Check if a key exists
    fn contains(&self, key: &[u8]) -> Result<bool, IndexError> {
        Ok(self.get(key)?.is_some())
    }

    /// Range scan
    ///
    /// Returns an iterator over entries in the specified range.
    fn range(
        &self,
        start: Bound<&[u8]>,
        end: Bound<&[u8]>,
        direction: ScanDirection,
    ) -> Result<Box<dyn IndexIterator + '_>, IndexError>;

    /// Scan all entries
    fn scan(&self, direction: ScanDirection) -> Result<Box<dyn IndexIterator + '_>, IndexError> {
        self.range(Bound::Unbounded, Bound::Unbounded, direction)
    }

    /// Count entries in a range
    fn count_range(&self, start: Bound<&[u8]>, end: Bound<&[u8]>) -> Result<usize, IndexError> {
        let mut iter = self.range(start, end, ScanDirection::Forward)?;
        let mut count = 0;
        while iter.next().is_some() {
            count += 1;
        }
        Ok(count)
    }

    /// Count all entries
    fn count(&self) -> Result<usize, IndexError> {
        self.count_range(Bound::Unbounded, Bound::Unbounded)
    }
}

/// Ordered index trait
///
/// An index that maintains key ordering and supports min/max operations.
pub trait OrderedIndex: Index {
    /// Get the entry with the smallest key
    fn min(&self) -> Result<Option<IndexEntry>, IndexError>;

    /// Get the entry with the largest key
    fn max(&self) -> Result<Option<IndexEntry>, IndexError>;

    /// Get the entry at a specific rank (0-indexed position)
    fn at_rank(&self, rank: usize) -> Result<Option<IndexEntry>, IndexError>;

    /// Get the rank of a key (0-indexed position)
    fn rank_of(&self, key: &[u8]) -> Result<Option<usize>, IndexError>;
}

/// Unique index trait
///
/// An index that enforces uniqueness of keys.
pub trait UniqueIndex: Index {
    /// Insert a key-value pair, failing if the key already exists
    fn insert_unique(&mut self, key: &[u8], value: &[u8]) -> Result<(), IndexError>;
}

/// Similarity index trait
///
/// An index that supports approximate retrieval based on similarity.
/// This is used for holographic and vector search.
pub trait SimilarityIndex: Index {
    /// Search for entries similar to the query key
    ///
    /// Returns up to `limit` entries with their similarity scores.
    fn search(&self, query_key: &[u8], limit: usize) -> Result<Vec<(IndexEntry, f32)>, IndexError>;

    /// Get current signal-to-noise ratio (SNR) for the index
    fn snr(&self) -> f32;

    /// Get estimated capacity remaining
    fn estimated_capacity(&self) -> usize;
}
