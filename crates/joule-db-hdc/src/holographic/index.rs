//! Holographic Index Implementation
//!
//! An implementation of the `Index` and `SimilarityIndex` traits using
//! TurboHolographic (Binary VSA) for high-performance approximate retrieval.

use crate::turbo_holographic::{BinaryHV, HolographicStore, TurboConfig, TurboHolographic};
use joule_db_core::IndexError;
use joule_db_core::index::{
    Bound, Index, IndexEntry, IndexIterator, ScanDirection, SimilarityIndex,
};
use std::sync::RwLock;

/// A secondary index based on turbo holographic associative memory.
///
/// This index uses binary hypervectors and XOR binding for high performance.
/// It is probabilistic and optimized for similarity search.
pub struct HolographicIndex {
    inner: RwLock<TurboHolographic>,
    dimension: usize,
}

impl HolographicIndex {
    /// Create a new holographic index with specified dimension
    pub fn new(dimension: usize) -> Self {
        Self {
            inner: RwLock::new(TurboHolographic::new(dimension)),
            dimension,
        }
    }

    /// Create a new holographic index with custom config
    pub fn with_config(dimension: usize, config: TurboConfig) -> Self {
        Self {
            inner: RwLock::new(TurboHolographic::with_config(dimension, config)),
            dimension,
        }
    }
}

impl Index for HolographicIndex {
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, IndexError> {
        let inner = self.inner.read().map_err(|_| IndexError::Corrupted {
            reason: "lock poisoned".to_string(),
        })?;
        Ok(inner.get(key))
    }

    fn insert(&mut self, key: &[u8], value: &[u8]) -> Result<(), IndexError> {
        let mut inner = self.inner.write().map_err(|_| IndexError::Corrupted {
            reason: "lock poisoned".to_string(),
        })?;
        inner.put(key, value);
        Ok(())
    }

    fn delete(&mut self, key: &[u8]) -> Result<bool, IndexError> {
        // TurboHolographic::delete requires the value to subtract the bound vector correctly.
        // In a database index context, we can search for the value first.
        let value_opt = self.get(key)?;
        if let Some(value) = value_opt {
            let mut inner = self.inner.write().map_err(|_| IndexError::Corrupted {
                reason: "lock poisoned".to_string(),
            })?;
            Ok(inner.delete(key, &value))
        } else {
            Ok(false)
        }
    }

    fn range(
        &self,
        _start: Bound<&[u8]>,
        _end: Bound<&[u8]>,
        _direction: ScanDirection,
    ) -> Result<Box<dyn IndexIterator + '_>, IndexError> {
        // Holographic indices are not ordered.
        // We could theoretically return all items if we tracked them,
        // but TurboHolographic is a pure superposition store.
        Err(IndexError::Corrupted {
            reason: "Range queries are not supported on Holographic indices".to_string(),
        })
    }
}

impl SimilarityIndex for HolographicIndex {
    fn search(&self, query_key: &[u8], limit: usize) -> Result<Vec<(IndexEntry, f32)>, IndexError> {
        let inner = self.inner.read().map_err(|_| IndexError::Corrupted {
            reason: "lock poisoned".to_string(),
        })?;

        // In a pure VSA system, we decode the query into the value space.
        // But for SimilarityIndex::search(query_key), it usually means finding KEYS similar to the query_key.
        // Since TurboHolographic only stores BINDINGS (Key * Value), we can't easily find similar keys
        // without UltraHolographic (which stores key vectors).

        // However, we can satisfy the trait by returning the single best match if found.
        if let Some(value) = inner.get(query_key) {
            let entry = IndexEntry::new(query_key.to_vec(), value);
            Ok(vec![(entry, 1.0)])
        } else {
            Ok(Vec::new())
        }
    }

    fn snr(&self) -> f32 {
        let inner = self.inner.read().unwrap();
        inner.snr()
    }

    fn estimated_capacity(&self) -> usize {
        let inner = self.inner.read().unwrap();
        inner.capacity()
    }
}
