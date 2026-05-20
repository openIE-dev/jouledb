//! Signal Storage Connector
//!
//! Bridges SigQL to JouleDB persistent storage, allowing signals to be
//! stored and retrieved from the database engine.

#[cfg(feature = "storage")]
use std::sync::Arc;

#[cfg(feature = "storage")]
use joule_db_core::engine::Engine;

#[cfg(feature = "storage")]
use serde::{Deserialize, Serialize};

use super::IoError;
use super::IoResult;
use crate::types::DynSignal;

/// Prefix for signal data keys
const SIGNAL_DATA_PREFIX: &str = "__signal__::";
/// Suffix for signal metadata
const META_SUFFIX: &str = "::meta";
/// Suffix for signal samples
const DATA_SUFFIX: &str = "::data";

/// Serialized signal metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg(feature = "storage")]
pub struct SignalMetadata {
    /// Channel/signal name
    pub channel: String,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Start timestamp in nanoseconds
    pub start_ns: i64,
    /// Number of samples
    pub num_samples: u64,
}

/// Storage connector for SigQL ↔ JouleDB Engine integration
///
/// Allows storing and retrieving signal data from the persistent B-tree engine.
///
/// # Example
///
/// ```rust,ignore
/// use sigql::io::SignalStorageConnector;
/// use joule_db_core::engine::Engine;
/// use std::sync::Arc;
///
/// let engine = Arc::new(Engine::new(backend)?);
/// let connector = SignalStorageConnector::new(engine);
///
/// // Store a signal
/// connector.store_signal("eeg_channel1", &signal)?;
///
/// // Load it back
/// let loaded = connector.load_signal("eeg_channel1")?;
/// ```
#[cfg(feature = "storage")]
pub struct SignalStorageConnector {
    engine: Arc<Engine>,
}

#[cfg(feature = "storage")]
impl SignalStorageConnector {
    /// Create a new storage connector with the given engine
    pub fn new(engine: Arc<Engine>) -> Self {
        Self { engine }
    }

    /// Store a signal in the database under a named key
    ///
    /// Stores both metadata and sample data separately for efficient access.
    pub fn store_signal(&self, name: &str, signal: &DynSignal<f64>) -> IoResult<()> {
        let meta_key = format!("{}{}{}", SIGNAL_DATA_PREFIX, name, META_SUFFIX);
        let data_key = format!("{}{}{}", SIGNAL_DATA_PREFIX, name, DATA_SUFFIX);

        // Create metadata
        let metadata = SignalMetadata {
            channel: signal.channel.to_string(),
            sample_rate: signal.sample_rate,
            start_ns: signal.start_ns,
            num_samples: signal.samples.len() as u64,
        };

        // Serialize metadata
        let meta_bytes = bincode::serde::encode_to_vec(&metadata, bincode::config::standard())
            .map_err(|e| IoError::InvalidFormat(format!("Failed to serialize metadata: {}", e)))?;

        // Serialize samples (as f64 array)
        let data_bytes = bincode::serde::encode_to_vec(&signal.samples, bincode::config::standard())
            .map_err(|e| IoError::InvalidFormat(format!("Failed to serialize samples: {}", e)))?;

        // Store in engine
        self.engine
            .put(meta_key.as_bytes(), &meta_bytes)
            .map_err(|e| {
                IoError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    e.to_string(),
                ))
            })?;

        self.engine
            .put(data_key.as_bytes(), &data_bytes)
            .map_err(|e| {
                IoError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    e.to_string(),
                ))
            })?;

        Ok(())
    }

    /// Load a signal from the database by name
    pub fn load_signal(&self, name: &str) -> IoResult<DynSignal<f64>> {
        let meta_key = format!("{}{}{}", SIGNAL_DATA_PREFIX, name, META_SUFFIX);
        let data_key = format!("{}{}{}", SIGNAL_DATA_PREFIX, name, DATA_SUFFIX);

        // Load metadata
        let meta_bytes = self
            .engine
            .get(meta_key.as_bytes())
            .map_err(|e| {
                IoError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    e.to_string(),
                ))
            })?
            .ok_or_else(|| IoError::FileNotFound(format!("Signal '{}' not found", name)))?;

        let metadata: SignalMetadata = bincode::serde::decode_from_slice(&meta_bytes, bincode::config::standard())
            .map(|(v, _)| v)
            .map_err(|e| {
                IoError::InvalidFormat(format!("Failed to deserialize metadata: {}", e))
            })?;

        // Load samples
        let data_bytes = self
            .engine
            .get(data_key.as_bytes())
            .map_err(|e| {
                IoError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    e.to_string(),
                ))
            })?
            .ok_or_else(|| {
                IoError::FileNotFound(format!("Signal data for '{}' not found", name))
            })?;

        let samples: Vec<f64> = bincode::serde::decode_from_slice(&data_bytes, bincode::config::standard())
            .map(|(v, _)| v)
            .map_err(|e| IoError::InvalidFormat(format!("Failed to deserialize samples: {}", e)))?;

        Ok(DynSignal::new(
            metadata.channel,
            samples,
            metadata.sample_rate,
            metadata.start_ns,
        ))
    }

    /// List all stored signal names
    pub fn list_signals(&self) -> IoResult<Vec<String>> {
        let prefix = SIGNAL_DATA_PREFIX.as_bytes();
        let meta_suffix = META_SUFFIX;

        let mut signals = Vec::new();

        // Scan all keys with the signal prefix using range query
        let mut iter = self
            .engine
            .range(
                joule_db_core::index::Bound::Included(prefix),
                joule_db_core::index::Bound::Unbounded,
                joule_db_core::index::ScanDirection::Forward,
            )
            .map_err(|e| {
                IoError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    e.to_string(),
                ))
            })?;

        while let Some(result) = iter.next() {
            let entry = result.map_err(|e| {
                IoError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    e.to_string(),
                ))
            })?;

            let key_str = String::from_utf8_lossy(&entry.key);

            // Check if this is a metadata key
            if key_str.starts_with(SIGNAL_DATA_PREFIX) && key_str.ends_with(meta_suffix) {
                // Extract signal name
                let start = SIGNAL_DATA_PREFIX.len();
                let end = key_str.len() - meta_suffix.len();
                if end > start {
                    signals.push(key_str[start..end].to_string());
                }
            }

            // Stop if we've passed the prefix
            if !key_str.starts_with(SIGNAL_DATA_PREFIX) {
                break;
            }
        }

        Ok(signals)
    }

    /// Delete a signal from storage
    pub fn delete_signal(&self, name: &str) -> IoResult<bool> {
        let meta_key = format!("{}{}{}", SIGNAL_DATA_PREFIX, name, META_SUFFIX);
        let data_key = format!("{}{}{}", SIGNAL_DATA_PREFIX, name, DATA_SUFFIX);

        // Check if signal exists
        let exists = self
            .engine
            .get(meta_key.as_bytes())
            .map_err(|e| {
                IoError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    e.to_string(),
                ))
            })?
            .is_some();

        if !exists {
            return Ok(false);
        }

        // Delete both keys
        self.engine.delete(meta_key.as_bytes()).map_err(|e| {
            IoError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                e.to_string(),
            ))
        })?;

        self.engine.delete(data_key.as_bytes()).map_err(|e| {
            IoError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                e.to_string(),
            ))
        })?;

        Ok(true)
    }

    /// Get the underlying engine reference
    pub fn engine(&self) -> &Arc<Engine> {
        &self.engine
    }
}

#[cfg(all(test, feature = "storage"))]
mod tests {
    use super::*;
    use joule_db_core::storage::memory::MemoryBackend;

    #[test]
    fn test_store_load_roundtrip() {
        let backend = MemoryBackend::new();
        let engine = Arc::new(Engine::new(backend).unwrap());
        let connector = SignalStorageConnector::new(engine);

        let signal = DynSignal::new("test_channel", vec![1.0, 2.0, 3.0, 4.0, 5.0], 1000, 0);

        connector.store_signal("test_signal", &signal).unwrap();
        let loaded = connector.load_signal("test_signal").unwrap();

        assert_eq!(loaded.channel.as_str(), "test_channel");
        assert_eq!(loaded.sample_rate, 1000);
        assert_eq!(loaded.samples.len(), 5);
        assert!((loaded.samples[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_list_signals() {
        let backend = MemoryBackend::new();
        let engine = Arc::new(Engine::new(backend).unwrap());
        let connector = SignalStorageConnector::new(engine);

        let signal = DynSignal::new("ch", vec![1.0], 1000, 0);

        connector.store_signal("sig1", &signal).unwrap();
        connector.store_signal("sig2", &signal).unwrap();
        connector.store_signal("sig3", &signal).unwrap();

        let names = connector.list_signals().unwrap();
        assert_eq!(names.len(), 3);
        assert!(names.contains(&"sig1".to_string()));
        assert!(names.contains(&"sig2".to_string()));
        assert!(names.contains(&"sig3".to_string()));
    }

    #[test]
    fn test_delete_signal() {
        let backend = MemoryBackend::new();
        let engine = Arc::new(Engine::new(backend).unwrap());
        let connector = SignalStorageConnector::new(engine);

        let signal = DynSignal::new("ch", vec![1.0], 1000, 0);

        connector.store_signal("to_delete", &signal).unwrap();
        assert!(connector.load_signal("to_delete").is_ok());

        let deleted = connector.delete_signal("to_delete").unwrap();
        assert!(deleted);

        assert!(connector.load_signal("to_delete").is_err());

        // Deleting non-existent should return false
        let deleted_again = connector.delete_signal("to_delete").unwrap();
        assert!(!deleted_again);
    }
}
