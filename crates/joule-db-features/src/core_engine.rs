//! Adapter from `joule_db_core::Engine` to the feature `StorageEngine` trait.
//!
//! Time series (and the other feature persistence types) already speak
//! `StorageEngine`. This wraps the on-disk B-tree engine so those writes
//! survive process restart.

use std::path::Path;
use std::sync::Arc;

use joule_db_core::{DiskBackend, Engine};

use crate::persistence::{PersistenceError, StorageEngine};

/// B-tree engine used as a feature-store backend.
pub struct CoreEngineStore {
    engine: Arc<Engine>,
}

impl CoreEngineStore {
    /// Open or create a durable engine at `path` (a single database file).
    pub fn open(path: impl AsRef<Path>) -> Result<Self, PersistenceError> {
        let backend = DiskBackend::open(path.as_ref())
            .map_err(|e| PersistenceError::Storage(e.to_string()))?;
        let engine = Engine::open_or_create(backend)
            .map_err(|e| PersistenceError::Storage(e.to_string()))?;
        Ok(Self {
            engine: Arc::new(engine),
        })
    }

    /// Flush dirty pages so a later `open` of the same path sees the writes.
    pub fn sync(&self) -> Result<(), PersistenceError> {
        self.engine
            .sync()
            .map_err(|e| PersistenceError::Storage(e.to_string()))
    }

    /// Shared handle for `TimeSeriesPersistence::new`.
    pub fn shared(self: &Arc<Self>) -> Arc<Self> {
        Arc::clone(self)
    }
}

impl StorageEngine for CoreEngineStore {
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, PersistenceError> {
        self.engine
            .get(key)
            .map_err(|e| PersistenceError::Storage(e.to_string()))
    }

    fn put(&self, key: &[u8], value: &[u8]) -> Result<(), PersistenceError> {
        self.engine
            .put(key, value)
            .map_err(|e| PersistenceError::Storage(e.to_string()))
    }

    fn delete(&self, key: &[u8]) -> Result<bool, PersistenceError> {
        self.engine
            .delete(key)
            .map_err(|e| PersistenceError::Storage(e.to_string()))
    }

    fn prefix_scan(&self, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>, PersistenceError> {
        let iter = self
            .engine
            .prefix_scan(prefix)
            .map_err(|e| PersistenceError::Storage(e.to_string()))?;
        let mut out = Vec::new();
        for item in iter {
            let entry = item.map_err(|e| PersistenceError::Storage(e.to_string()))?;
            out.push((entry.key, entry.value));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::{PersistedDataPoint, TimeSeriesPersistence};
    use std::collections::HashMap;

    #[test]
    fn timeseries_durable_write_readback() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("timeseries.wdb");

        {
            let store = Arc::new(CoreEngineStore::open(&path).expect("open engine"));
            let ts = TimeSeriesPersistence::new(store.clone());
            ts.register_metric("cpu").unwrap();
            ts.write(
                "cpu",
                PersistedDataPoint {
                    timestamp: 1_700_000_000_000,
                    value: 42.5,
                    tags: HashMap::from([("host".into(), "a".into())]),
                },
            )
            .unwrap();
            ts.write(
                "cpu",
                PersistedDataPoint {
                    timestamp: 1_700_000_001_000,
                    value: 43.0,
                    tags: HashMap::new(),
                },
            )
            .unwrap();
            store.sync().unwrap();
        }

        let store = Arc::new(CoreEngineStore::open(&path).expect("reopen engine"));
        let ts = TimeSeriesPersistence::new(store);
        let points = ts
            .query("cpu", 1_700_000_000_000, 1_700_000_001_000)
            .unwrap();
        assert_eq!(points.len(), 2, "durable read-back lost points: {points:?}");
        assert_eq!(points[0].value, 42.5);
        assert_eq!(points[0].tags.get("host").map(String::as_str), Some("a"));
        assert_eq!(points[1].value, 43.0);
        let metrics = ts.list_metrics().unwrap();
        assert!(metrics.iter().any(|m| m == "cpu"), "metrics={metrics:?}");
    }
}
