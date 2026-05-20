//! File I/O operations for signal data
//!
//! This module provides reading and writing signals to common file formats:
//!
//! ## Supported Formats
//!
//! | Format | Extension | Read | Write | Use Case |
//! |--------|-----------|------|-------|----------|
//! | WAV | `.wav` | ✓ | ✓ | Audio signals |
//! | CSV | `.csv` | ✓ | ✓ | Tabular data |
//! | EDF/EDF+ | `.edf` | ✓ | ✓ | Medical (EEG, ECG) |
//! | HDF5 | `.h5`, `.hdf5` | ✓ | ✓ | Scientific data |
//! | Parquet | `.parquet` | ✓ | ✓ | Columnar analytics |
//!
//! ## Usage
//!
//! ```rust,ignore
//! use sigql::io::{read_signal, write_signal, Format};
//!
//! // Auto-detect format from extension
//! let signal = read_signal("data.edf", None)?;
//!
//! // Explicit format
//! let signal = read_signal("data.bin", Some(Format::Raw {
//!     sample_rate: 1000,
//!     dtype: DataType::F64
//! }))?;
//!
//! // Write with auto-detection
//! write_signal("output.parquet", &signal, None)?;
//! ```

pub mod csv;
pub mod edf;
pub mod hdf5;
pub mod parquet;
mod traits;
pub mod wav;

/// MediaQL ingest pipeline — middle-out frequency-domain encoding for images, audio, video.
pub mod media_ingest;

/// Frequency-domain storage format (.freq) — sparse, progressive, compact.
pub mod freq_store;

/// Content-aware delta encoding + progressive quality tiers for CDN.
pub mod freq_delta;

/// Knowledge graph connector — bridges SigQL to graph backends (subsumes SigSPARQL).
pub mod graph_connector;

#[cfg(feature = "storage")]
pub mod sigql_connector;

pub use csv::{
    read_csv, read_csv_all_columns, read_csv_columns, read_csv_multi, write_csv, write_csv_multi,
};
pub use edf::{EdfHeader, EdfSignalInfo, read_edf, write_edf};
pub use hdf5::{Hdf5Options, read_hdf5, write_hdf5};
pub use parquet::{ParquetOptions, read_parquet, write_parquet};
pub use traits::{DataType, Format, SignalMetadata, SignalReader, SignalWriter};
pub use wav::{read_wav, write_wav};

pub use media_ingest::{MediaIngestConfig, MediaIngestPipeline};
pub use freq_store::{CompressionInfo, compression_info, read_freq, read_freq_progressive, write_freq};

#[cfg(feature = "storage")]
pub use sigql_connector::SignalStorageConnector;

use std::path::Path;
use thiserror::Error;

use crate::types::DynSignal;

/// Errors that can occur during file I/O operations
#[derive(Debug, Error)]
pub enum IoError {
    #[error("File not found: {0}")]
    FileNotFound(String),

    #[error("Invalid file format: {0}")]
    InvalidFormat(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("WAV error: {0}")]
    Wav(String),

    #[error("CSV error: {0}")]
    Csv(String),

    #[error("EDF error: {0}")]
    Edf(String),

    #[error("HDF5 error: {0}")]
    Hdf5(String),

    #[error("Parquet error: {0}")]
    Parquet(String),

    #[error("Column not found: {0}")]
    ColumnNotFound(String),

    #[error("Channel not found: {0}")]
    ChannelNotFound(String),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Unsupported sample format: {0}")]
    UnsupportedFormat(String),

    #[error("Unsupported file extension: {0}")]
    UnsupportedExtension(String),

    #[error("Missing required metadata: {0}")]
    MissingMetadata(String),
}

/// Result type for I/O operations
pub type IoResult<T> = Result<T, IoError>;

/// Read a signal from a file with auto-detection or explicit format
///
/// # Arguments
/// * `path` - Path to the file
/// * `options` - Optional read options (channel selection, etc.)
///
/// # Example
/// ```rust,ignore
/// let signal = read_signal("recording.edf", None)?;
/// let signals = read_signal_multi("recording.edf", None)?; // All channels
/// ```
pub fn read_signal(
    path: impl AsRef<Path>,
    options: Option<ReadOptions>,
) -> IoResult<DynSignal<f64>> {
    let path = path.as_ref();
    let format = detect_format(path)?;
    let options = options.unwrap_or_default();

    match format {
        Format::Wav => read_wav(path),
        Format::Csv => read_csv(path, options.channel.as_deref().unwrap_or("value")),
        Format::Edf => {
            let signals = read_edf(path)?;
            let channel_idx = options.channel_index.unwrap_or(0);
            signals
                .into_iter()
                .nth(channel_idx)
                .ok_or_else(|| IoError::ChannelNotFound(format!("Channel index {}", channel_idx)))
        }
        Format::Hdf5 { dataset } => {
            let dataset = dataset
                .or(options.channel.clone())
                .unwrap_or_else(|| "signal".to_string());
            read_hdf5(path, &dataset)
        }
        Format::Parquet => {
            let column = options.channel.as_deref().unwrap_or("value");
            read_parquet(path, column)
        }
        Format::Raw { sample_rate, dtype } => read_raw(path, sample_rate, dtype),
    }
}

/// Read all signals/channels from a multi-channel file
pub fn read_signal_multi(
    path: impl AsRef<Path>,
    _options: Option<ReadOptions>,
) -> IoResult<Vec<DynSignal<f64>>> {
    let path = path.as_ref();
    let format = detect_format(path)?;

    match format {
        Format::Wav => Ok(vec![read_wav(path)?]),
        Format::Csv => {
            // Read all numeric columns
            csv::read_csv_all_columns(path)
        }
        Format::Edf => read_edf(path),
        Format::Hdf5 { .. } => hdf5::read_hdf5_all(path),
        Format::Parquet => parquet::read_parquet_all(path),
        Format::Raw { .. } => Err(IoError::UnsupportedFormat(
            "Raw format requires explicit channel specification".into(),
        )),
    }
}

/// Write a signal to a file with auto-detection or explicit format
pub fn write_signal(
    path: impl AsRef<Path>,
    signal: &DynSignal<f64>,
    options: Option<WriteOptions>,
) -> IoResult<()> {
    let path = path.as_ref();
    let options = options.unwrap_or_default();
    let format = options
        .format
        .clone()
        .unwrap_or_else(|| detect_format(path).unwrap_or(Format::Csv));

    match format {
        Format::Wav => write_wav(path, signal),
        Format::Csv => write_csv(path, signal),
        Format::Edf => {
            let header = options.edf_header.unwrap_or_default();
            write_edf(path, &[signal.clone()], header)
        }
        Format::Hdf5 { dataset } => {
            let dataset = dataset.unwrap_or_else(|| "signal".to_string());
            let opts = options.hdf5.unwrap_or_default();
            write_hdf5(path, signal, &dataset, opts)
        }
        Format::Parquet => {
            let opts = options.parquet.unwrap_or_default();
            write_parquet(path, signal, opts)
        }
        Format::Raw { .. } => write_raw(path, signal),
    }
}

/// Write multiple signals to a file (for multi-channel formats)
pub fn write_signal_multi(
    path: impl AsRef<Path>,
    signals: &[DynSignal<f64>],
    options: Option<WriteOptions>,
) -> IoResult<()> {
    let path = path.as_ref();
    let options = options.unwrap_or_default();
    let format = options
        .format
        .clone()
        .unwrap_or_else(|| detect_format(path).unwrap_or(Format::Csv));

    match format {
        Format::Wav => {
            if signals.len() != 1 {
                return Err(IoError::UnsupportedFormat(
                    "WAV only supports single channel in this implementation".into(),
                ));
            }
            write_wav(path, &signals[0])
        }
        Format::Csv => csv::write_csv_multi(path, signals),
        Format::Edf => {
            let header = options.edf_header.unwrap_or_default();
            write_edf(path, signals, header)
        }
        Format::Hdf5 { .. } => {
            let opts = options.hdf5.unwrap_or_default();
            hdf5::write_hdf5_multi(path, signals, opts)
        }
        Format::Parquet => {
            let opts = options.parquet.unwrap_or_default();
            parquet::write_parquet_multi(path, signals, opts)
        }
        Format::Raw { .. } => Err(IoError::UnsupportedFormat(
            "Raw format doesn't support multi-channel".into(),
        )),
    }
}

/// Options for reading signals
#[derive(Debug, Clone, Default)]
pub struct ReadOptions {
    /// Channel/column name to read
    pub channel: Option<String>,
    /// Channel index (for formats like EDF)
    pub channel_index: Option<usize>,
    /// Time range to read (start_ns, end_ns)
    pub time_range: Option<(i64, i64)>,
    /// Resample to this rate while reading
    pub target_sample_rate: Option<u32>,
}

/// Options for writing signals
#[derive(Debug, Clone, Default)]
pub struct WriteOptions {
    /// Explicit format (otherwise auto-detected)
    pub format: Option<Format>,
    /// EDF-specific header info
    pub edf_header: Option<EdfHeader>,
    /// HDF5-specific options
    pub hdf5: Option<Hdf5Options>,
    /// Parquet-specific options
    pub parquet: Option<ParquetOptions>,
}

/// Detect file format from extension
fn detect_format(path: &Path) -> IoResult<Format> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
        .ok_or_else(|| IoError::UnsupportedExtension("No extension".into()))?;

    match ext.as_str() {
        "wav" | "wave" => Ok(Format::Wav),
        "csv" | "tsv" => Ok(Format::Csv),
        "edf" | "bdf" => Ok(Format::Edf),
        "h5" | "hdf5" | "hdf" => Ok(Format::Hdf5 { dataset: None }),
        "parquet" | "pq" => Ok(Format::Parquet),
        "bin" | "raw" | "dat" => Ok(Format::Raw {
            sample_rate: 1000, // Default, should be overridden
            dtype: DataType::F64,
        }),
        _ => Err(IoError::UnsupportedExtension(ext)),
    }
}

/// Read raw binary file
fn read_raw(path: &Path, sample_rate: u32, dtype: DataType) -> IoResult<DynSignal<f64>> {
    use std::io::Read;

    let mut file =
        std::fs::File::open(path).map_err(|_| IoError::FileNotFound(path.display().to_string()))?;

    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;

    let samples: Vec<f64> = match dtype {
        DataType::F32 => bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64)
            .collect(),
        DataType::F64 => bytes
            .chunks_exact(8)
            .map(|b| f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
            .collect(),
        DataType::I16 => bytes
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]) as f64 / 32768.0)
            .collect(),
        DataType::I32 => bytes
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64 / 2147483648.0)
            .collect(),
    };

    let channel = path.file_stem().and_then(|s| s.to_str()).unwrap_or("raw");

    Ok(DynSignal::new(channel, samples, sample_rate, 0))
}

/// Write raw binary file
fn write_raw(path: &Path, signal: &DynSignal<f64>) -> IoResult<()> {
    use std::io::Write;

    let mut file = std::fs::File::create(path)?;

    for sample in &signal.samples {
        file.write_all(&sample.to_le_bytes())?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_detect_format() {
        assert!(matches!(
            detect_format(Path::new("test.wav")),
            Ok(Format::Wav)
        ));
        assert!(matches!(
            detect_format(Path::new("test.csv")),
            Ok(Format::Csv)
        ));
        assert!(matches!(
            detect_format(Path::new("test.edf")),
            Ok(Format::Edf)
        ));
        assert!(matches!(
            detect_format(Path::new("test.h5")),
            Ok(Format::Hdf5 { .. })
        ));
        assert!(matches!(
            detect_format(Path::new("test.parquet")),
            Ok(Format::Parquet)
        ));
    }

    #[test]
    fn test_raw_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.bin");

        let signal = DynSignal::new("test", vec![1.0, 2.0, 3.0, 4.0, 5.0], 1000, 0);
        write_raw(&path, &signal).unwrap();

        let loaded = read_raw(&path, 1000, DataType::F64).unwrap();
        assert_eq!(loaded.samples.len(), 5);
        assert!((loaded.samples[0] - 1.0).abs() < 1e-10);
    }
}
