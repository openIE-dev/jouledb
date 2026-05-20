//! Common traits and types for signal I/O
//!
//! This module defines the unified interface for reading and writing signals
//! across different file formats.

use std::collections::HashMap;
use std::path::Path;

use super::IoResult;
use crate::types::DynSignal;

/// Supported file formats
#[derive(Debug, Clone, PartialEq)]
pub enum Format {
    /// WAV audio format
    Wav,
    /// CSV/TSV tabular format
    Csv,
    /// European Data Format (EDF/EDF+/BDF)
    Edf,
    /// HDF5 hierarchical data format
    Hdf5 {
        /// Dataset path within the HDF5 file
        dataset: Option<String>,
    },
    /// Apache Parquet columnar format
    Parquet,
    /// Raw binary format
    Raw {
        /// Sample rate in Hz
        sample_rate: u32,
        /// Data type
        dtype: DataType,
    },
}

/// Data types for raw binary files
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    /// 32-bit float
    F32,
    /// 64-bit float
    F64,
    /// 16-bit signed integer
    I16,
    /// 32-bit signed integer
    I32,
}

impl DataType {
    /// Size in bytes
    pub fn size(&self) -> usize {
        match self {
            DataType::F32 => 4,
            DataType::F64 => 8,
            DataType::I16 => 2,
            DataType::I32 => 4,
        }
    }
}

/// Metadata associated with a signal
#[derive(Debug, Clone, Default)]
pub struct SignalMetadata {
    /// Signal/channel label
    pub label: String,
    /// Physical dimension/unit (e.g., "uV", "mV")
    pub physical_dimension: Option<String>,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Physical minimum value
    pub physical_min: Option<f64>,
    /// Physical maximum value
    pub physical_max: Option<f64>,
    /// Recording start time (nanoseconds since epoch)
    pub start_time: i64,
    /// Duration in nanoseconds
    pub duration_ns: Option<i64>,
    /// Additional key-value metadata
    pub annotations: HashMap<String, String>,
    /// Transducer type (for EDF)
    pub transducer: Option<String>,
    /// Prefiltering info (for EDF)
    pub prefiltering: Option<String>,
}

impl SignalMetadata {
    pub fn new(label: impl Into<String>, sample_rate: u32) -> Self {
        Self {
            label: label.into(),
            sample_rate,
            ..Default::default()
        }
    }

    pub fn with_unit(mut self, unit: impl Into<String>) -> Self {
        self.physical_dimension = Some(unit.into());
        self
    }

    pub fn with_range(mut self, min: f64, max: f64) -> Self {
        self.physical_min = Some(min);
        self.physical_max = Some(max);
        self
    }
}

/// Trait for reading signals from files
pub trait SignalReader {
    /// Read a single signal/channel from the file
    fn read(&self, path: &Path) -> IoResult<DynSignal<f64>>;

    /// Read all signals/channels from the file
    fn read_all(&self, path: &Path) -> IoResult<Vec<DynSignal<f64>>>;

    /// Read metadata without loading signal data
    fn read_metadata(&self, path: &Path) -> IoResult<Vec<SignalMetadata>>;

    /// Get supported file extensions
    fn extensions(&self) -> &[&str];
}

/// Trait for writing signals to files
pub trait SignalWriter {
    /// Write a single signal to the file
    fn write(&self, path: &Path, signal: &DynSignal<f64>) -> IoResult<()>;

    /// Write multiple signals to the file
    fn write_all(&self, path: &Path, signals: &[DynSignal<f64>]) -> IoResult<()>;

    /// Get supported file extensions
    fn extensions(&self) -> &[&str];
}

/// File info for quick inspection
#[derive(Debug, Clone)]
pub struct FileInfo {
    /// Detected format
    pub format: Format,
    /// Number of channels/signals
    pub num_channels: usize,
    /// Sample rate (if uniform across channels)
    pub sample_rate: Option<u32>,
    /// Total duration in seconds
    pub duration_seconds: Option<f64>,
    /// Total number of samples per channel
    pub num_samples: Option<usize>,
    /// Channel/signal labels
    pub channel_labels: Vec<String>,
}

/// Get file info without loading data
pub fn file_info(path: &Path) -> IoResult<FileInfo> {
    let format = super::detect_format(path)?;

    match format {
        Format::Wav => {
            let signal = super::wav::read_wav(path)?;
            Ok(FileInfo {
                format,
                num_channels: 1,
                sample_rate: Some(signal.sample_rate),
                duration_seconds: Some(signal.samples.len() as f64 / signal.sample_rate as f64),
                num_samples: Some(signal.samples.len()),
                channel_labels: vec![signal.channel.to_string()],
            })
        }
        Format::Edf => {
            let metadata = super::edf::read_edf_metadata(path)?;
            Ok(FileInfo {
                format,
                num_channels: metadata.len(),
                sample_rate: metadata.first().map(|m| m.sample_rate),
                duration_seconds: metadata
                    .first()
                    .and_then(|m| m.duration_ns.map(|d| d as f64 / 1e9)),
                num_samples: None, // Would need to calculate
                channel_labels: metadata.iter().map(|m| m.label.clone()).collect(),
            })
        }
        _ => {
            // For other formats, we'd need format-specific implementations
            Ok(FileInfo {
                format,
                num_channels: 0,
                sample_rate: None,
                duration_seconds: None,
                num_samples: None,
                channel_labels: vec![],
            })
        }
    }
}
