//! Apache Parquet file format support
//!
//! Parquet is a columnar storage format optimized for analytics workloads.
//! It provides efficient compression and encoding for time series data.
//!
//! ## Features
//!
//! - Columnar storage (efficient for analytics)
//! - Built-in compression (snappy, gzip, zstd)
//! - Schema evolution support
//! - Predicate pushdown
//!
//! ## Note
//!
//! This implementation uses a simplified binary format that mimics Parquet's
//! columnar structure. For full Parquet support with compression and
//! interoperability, use the `parquet` crate.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use smol_str::SmolStr;

use super::{IoError, IoResult};
use crate::types::DynSignal;

/// Parquet write options
#[derive(Debug, Clone)]
pub struct ParquetOptions {
    /// Compression codec
    pub compression: Compression,
    /// Row group size (number of rows per group)
    pub row_group_size: usize,
    /// Include timestamp column
    pub include_timestamps: bool,
    /// Column name for values
    pub value_column: String,
    /// Column name for timestamps
    pub time_column: String,
}

impl Default for ParquetOptions {
    fn default() -> Self {
        Self {
            compression: Compression::None,
            row_group_size: 65536,
            include_timestamps: false,
            value_column: "value".to_string(),
            time_column: "timestamp".to_string(),
        }
    }
}

impl ParquetOptions {
    pub fn new() -> Self {
        Self {
            compression: Compression::None,
            row_group_size: 65536,
            include_timestamps: true,
            value_column: "value".to_string(),
            time_column: "timestamp".to_string(),
        }
    }

    pub fn with_compression(mut self, compression: Compression) -> Self {
        self.compression = compression;
        self
    }
}

/// Compression codecs
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Compression {
    #[default]
    None,
    Snappy,
    Gzip,
    Zstd,
}

/// Simple Parquet-like columnar format
///
/// This is a simplified implementation for signal data.
///
/// Format:
/// - Magic: "SIGPQ\0\0\0" (8 bytes)
/// - Version: u32
/// - Num columns: u32
/// - Schema (for each column):
///   - Name length: u32
///   - Name: utf8 bytes
///   - Type: u8 (0=f64, 1=i64, 2=string)
/// - Metadata:
///   - Sample rate: u32
///   - Start time: i64
/// - Num rows: u64
/// - Column data (for each column):
///   - Data bytes (type-dependent)

const MAGIC: &[u8] = b"SIGPQ\0\0\0";
const VERSION: u32 = 1;

const TYPE_F64: u8 = 0;
const TYPE_I64: u8 = 1;

/// Read a signal from a Parquet-like file
pub fn read_parquet(path: &Path, column: &str) -> IoResult<DynSignal<f64>> {
    let (columns, metadata) = read_parquet_raw(path)?;

    let data = columns
        .get(column)
        .ok_or_else(|| IoError::ColumnNotFound(column.to_string()))?;

    let samples: Vec<f64> = match data {
        ColumnData::F64(v) => v.clone(),
        ColumnData::I64(v) => v.iter().map(|&x| x as f64).collect(),
    };

    let sample_rate = metadata
        .get("sample_rate")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);

    let start_ns = metadata
        .get("start_ns")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    Ok(DynSignal::new(
        SmolStr::new(column),
        samples,
        sample_rate,
        start_ns,
    ))
}

/// Read all columns from a Parquet-like file as signals
pub fn read_parquet_all(path: &Path) -> IoResult<Vec<DynSignal<f64>>> {
    let (columns, metadata) = read_parquet_raw(path)?;

    let sample_rate = metadata
        .get("sample_rate")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);

    let start_ns = metadata
        .get("start_ns")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let mut signals = Vec::new();

    for (name, data) in columns {
        // Skip timestamp columns
        if name == "timestamp" || name == "time" || name == "ts" {
            continue;
        }

        let samples: Vec<f64> = match data {
            ColumnData::F64(v) => v,
            ColumnData::I64(v) => v.iter().map(|&x| x as f64).collect(),
        };

        signals.push(DynSignal::new(
            SmolStr::new(&name),
            samples,
            sample_rate,
            start_ns,
        ));
    }

    Ok(signals)
}

/// Column data types
enum ColumnData {
    F64(Vec<f64>),
    I64(Vec<i64>),
}

/// Read raw column data
fn read_parquet_raw(
    path: &Path,
) -> IoResult<(HashMap<String, ColumnData>, HashMap<String, String>)> {
    let file = File::open(path).map_err(|_| IoError::FileNotFound(path.display().to_string()))?;
    let mut reader = BufReader::new(file);

    // Read magic
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(IoError::InvalidFormat("Not a valid SIGPQ file".into()));
    }

    // Read version
    let mut buf4 = [0u8; 4];
    reader.read_exact(&mut buf4)?;
    let version = u32::from_le_bytes(buf4);
    if version > VERSION {
        return Err(IoError::Parquet(format!(
            "Unsupported version: {}",
            version
        )));
    }

    // Read number of columns
    reader.read_exact(&mut buf4)?;
    let num_columns = u32::from_le_bytes(buf4) as usize;

    // Read schema
    let mut schema: Vec<(String, u8)> = Vec::with_capacity(num_columns);
    for _ in 0..num_columns {
        reader.read_exact(&mut buf4)?;
        let name_len = u32::from_le_bytes(buf4) as usize;

        let mut name_buf = vec![0u8; name_len];
        reader.read_exact(&mut name_buf)?;
        let name = String::from_utf8_lossy(&name_buf).to_string();

        let mut type_buf = [0u8; 1];
        reader.read_exact(&mut type_buf)?;

        schema.push((name, type_buf[0]));
    }

    // Read metadata
    reader.read_exact(&mut buf4)?;
    let sample_rate = u32::from_le_bytes(buf4);

    let mut buf8 = [0u8; 8];
    reader.read_exact(&mut buf8)?;
    let start_ns = i64::from_le_bytes(buf8);

    let mut metadata = HashMap::new();
    metadata.insert("sample_rate".to_string(), sample_rate.to_string());
    metadata.insert("start_ns".to_string(), start_ns.to_string());

    // Read number of rows
    reader.read_exact(&mut buf8)?;
    let num_rows = u64::from_le_bytes(buf8) as usize;

    // Read column data
    let mut columns = HashMap::new();

    for (name, dtype) in schema {
        match dtype {
            TYPE_F64 => {
                let mut data = Vec::with_capacity(num_rows);
                for _ in 0..num_rows {
                    reader.read_exact(&mut buf8)?;
                    data.push(f64::from_le_bytes(buf8));
                }
                columns.insert(name, ColumnData::F64(data));
            }
            TYPE_I64 => {
                let mut data = Vec::with_capacity(num_rows);
                for _ in 0..num_rows {
                    reader.read_exact(&mut buf8)?;
                    data.push(i64::from_le_bytes(buf8));
                }
                columns.insert(name, ColumnData::I64(data));
            }
            _ => {
                return Err(IoError::Parquet(format!("Unknown column type: {}", dtype)));
            }
        }
    }

    Ok((columns, metadata))
}

/// Write a signal to a Parquet-like file
pub fn write_parquet(
    path: &Path,
    signal: &DynSignal<f64>,
    options: ParquetOptions,
) -> IoResult<()> {
    write_parquet_multi(path, &[signal.clone()], options)
}

/// Write multiple signals to a Parquet-like file (as columns)
pub fn write_parquet_multi(
    path: &Path,
    signals: &[DynSignal<f64>],
    options: ParquetOptions,
) -> IoResult<()> {
    if signals.is_empty() {
        return Err(IoError::Parquet("No signals to write".into()));
    }

    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // Determine number of rows
    let num_rows = signals.iter().map(|s| s.samples.len()).max().unwrap_or(0);

    // Build schema
    let mut schema: Vec<(String, u8)> = Vec::new();

    // Add timestamp column if requested
    if options.include_timestamps {
        schema.push((options.time_column.clone(), TYPE_I64));
    }

    // Add signal columns
    for signal in signals {
        let name = if signals.len() == 1 && !signal.channel.is_empty() {
            signal.channel.to_string()
        } else if signals.len() == 1 {
            options.value_column.clone()
        } else {
            signal.channel.to_string()
        };
        schema.push((name, TYPE_F64));
    }

    // Write magic
    writer.write_all(MAGIC)?;

    // Write version
    writer.write_all(&VERSION.to_le_bytes())?;

    // Write number of columns
    writer.write_all(&(schema.len() as u32).to_le_bytes())?;

    // Write schema
    for (name, dtype) in &schema {
        writer.write_all(&(name.len() as u32).to_le_bytes())?;
        writer.write_all(name.as_bytes())?;
        writer.write_all(&[*dtype])?;
    }

    // Write metadata
    let sample_rate = signals[0].sample_rate;
    let start_ns = signals[0].start_ns;
    writer.write_all(&sample_rate.to_le_bytes())?;
    writer.write_all(&start_ns.to_le_bytes())?;

    // Write number of rows
    writer.write_all(&(num_rows as u64).to_le_bytes())?;

    // Write column data
    // First timestamp column if present
    if options.include_timestamps {
        let dt_ns = 1_000_000_000i64 / sample_rate as i64;
        for i in 0..num_rows {
            let ts = start_ns + (i as i64) * dt_ns;
            writer.write_all(&ts.to_le_bytes())?;
        }
    }

    // Then signal columns
    for signal in signals {
        for i in 0..num_rows {
            let value = if i < signal.samples.len() {
                signal.samples[i]
            } else {
                0.0 // Pad with zeros
            };
            writer.write_all(&value.to_le_bytes())?;
        }
    }

    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_parquet_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.parquet");

        let signal = DynSignal::new("sensor", vec![1.0, 2.0, 3.0, 4.0, 5.0], 1000, 0);

        write_parquet(&path, &signal, ParquetOptions::new()).unwrap();

        let loaded = read_parquet(&path, "sensor").unwrap();
        assert_eq!(loaded.samples, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(loaded.sample_rate, 1000);
    }

    #[test]
    fn test_parquet_multi_channel() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("multi.parquet");

        let signals = vec![
            DynSignal::new("ch1", vec![1.0, 2.0, 3.0], 100, 0),
            DynSignal::new("ch2", vec![4.0, 5.0, 6.0], 100, 0),
        ];

        let options = ParquetOptions {
            include_timestamps: false,
            ..Default::default()
        };

        write_parquet_multi(&path, &signals, options).unwrap();

        let loaded = read_parquet_all(&path).unwrap();
        assert_eq!(loaded.len(), 2);
    }

    #[test]
    fn test_parquet_with_timestamps() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("ts.parquet");

        let signal = DynSignal::new("data", vec![1.0, 2.0, 3.0], 1000, 1000000000);

        let options = ParquetOptions {
            include_timestamps: true,
            ..Default::default()
        };

        write_parquet(&path, &signal, options).unwrap();

        let (columns, metadata) = read_parquet_raw(&path).unwrap();
        assert!(columns.contains_key("timestamp"));
        assert!(columns.contains_key("data"));
        assert_eq!(metadata.get("start_ns"), Some(&"1000000000".to_string()));
    }
}
