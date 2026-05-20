//! CSV file I/O for signal data
//!
//! Supports reading signal data from CSV files with automatic column detection.

use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

use csv::ReaderBuilder;
use smol_str::SmolStr;

use super::{IoError, IoResult};
use crate::types::{DynSignal, SignalMetadata};

/// Read a specific column from a CSV file into a DynSignal
///
/// The CSV file is expected to have a header row. The specified column
/// is extracted and parsed as f64 values.
///
/// # Arguments
///
/// * `path` - Path to the CSV file
/// * `column` - Name of the column to read
///
/// # Example
///
/// ```ignore
/// use std::path::Path;
/// use sigql::io::read_csv;
///
/// // CSV file with header: timestamp,value,quality
/// let signal = read_csv(Path::new("data.csv"), "value")?;
/// ```
pub fn read_csv(path: &Path, column: &str) -> IoResult<DynSignal<f64>> {
    let file = File::open(path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            IoError::FileNotFound(path.display().to_string())
        } else {
            IoError::Io(e)
        }
    })?;
    let reader = BufReader::new(file);

    let mut csv_reader = ReaderBuilder::new().has_headers(true).from_reader(reader);

    // Find column index
    let headers = csv_reader
        .headers()
        .map_err(|e| IoError::Csv(e.to_string()))?;

    let column_index = headers
        .iter()
        .position(|h| h == column)
        .ok_or_else(|| IoError::ColumnNotFound(column.to_string()))?;

    // Read values from the specified column
    let mut samples = Vec::new();
    for result in csv_reader.records() {
        let record = result.map_err(|e| IoError::Csv(e.to_string()))?;

        if let Some(value_str) = record.get(column_index) {
            let value: f64 = value_str
                .trim()
                .parse()
                .map_err(|e: std::num::ParseFloatError| {
                    IoError::ParseError(format!("Failed to parse '{}': {}", value_str, e))
                })?;
            samples.push(value);
        }
    }

    if samples.is_empty() {
        return Err(IoError::InvalidFormat("No data found in CSV".to_string()));
    }

    let channel_name = format!(
        "{}.{}",
        path.file_stem().and_then(|s| s.to_str()).unwrap_or("csv"),
        column
    );

    Ok(DynSignal {
        samples,
        sample_rate: 1, // Default to 1 Hz, caller should set appropriate rate
        channel: SmolStr::new(channel_name),
        start_ns: 0,
        metadata: SignalMetadata {
            source: Some(SmolStr::new(path.display().to_string())),
            calibrated: false,
            noise_floor: None,
            artifact_mask: None,
            units: None,
        },
    })
}

/// Write a DynSignal to a CSV file
///
/// Creates a CSV with columns: index, time_s, value
///
/// # Arguments
///
/// * `path` - Output path for the CSV file
/// * `signal` - The signal to write
///
/// # Example
///
/// ```ignore
/// use std::path::Path;
/// use sigql::io::write_csv;
///
/// write_csv(Path::new("output.csv"), &signal)?;
/// ```
pub fn write_csv(path: &Path, signal: &DynSignal<f64>) -> IoResult<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // Write header
    writeln!(writer, "index,time_s,value")?;

    // Write samples
    let dt = 1.0 / signal.sample_rate as f64;
    for (i, &value) in signal.samples.iter().enumerate() {
        let time = i as f64 * dt;
        writeln!(writer, "{},{:.9},{:.15}", i, time, value)?;
    }

    writer.flush()?;
    Ok(())
}

/// Read multiple columns from a CSV file
///
/// Returns a vector of signals, one for each specified column.
pub fn read_csv_columns(path: &Path, columns: &[&str]) -> IoResult<Vec<DynSignal<f64>>> {
    let file = File::open(path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            IoError::FileNotFound(path.display().to_string())
        } else {
            IoError::Io(e)
        }
    })?;
    let reader = BufReader::new(file);

    let mut csv_reader = ReaderBuilder::new().has_headers(true).from_reader(reader);

    // Find column indices
    let headers = csv_reader
        .headers()
        .map_err(|e| IoError::Csv(e.to_string()))?;

    let column_indices: Vec<usize> = columns
        .iter()
        .map(|col| {
            headers
                .iter()
                .position(|h| h == *col)
                .ok_or_else(|| IoError::ColumnNotFound(col.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Read values from all specified columns
    let mut samples_vecs: Vec<Vec<f64>> = vec![Vec::new(); columns.len()];

    for result in csv_reader.records() {
        let record = result.map_err(|e| IoError::Csv(e.to_string()))?;

        for (i, &col_idx) in column_indices.iter().enumerate() {
            if let Some(value_str) = record.get(col_idx) {
                let value: f64 =
                    value_str
                        .trim()
                        .parse()
                        .map_err(|e: std::num::ParseFloatError| {
                            IoError::ParseError(format!("Failed to parse '{}': {}", value_str, e))
                        })?;
                samples_vecs[i].push(value);
            }
        }
    }

    let file_stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("csv");

    let signals = columns
        .iter()
        .zip(samples_vecs.into_iter())
        .map(|(col, samples)| DynSignal {
            samples,
            sample_rate: 1,
            channel: SmolStr::new(format!("{}.{}", file_stem, col)),
            start_ns: 0,
            metadata: SignalMetadata {
                source: Some(SmolStr::new(path.display().to_string())),
                calibrated: false,
                noise_floor: None,
                artifact_mask: None,
                units: None,
            },
        })
        .collect();

    Ok(signals)
}

/// Write multiple signals to a CSV file with a shared time axis
pub fn write_csv_multi(path: &Path, signals: &[DynSignal<f64>]) -> IoResult<()> {
    if signals.is_empty() {
        return Err(IoError::InvalidFormat("No signals to write".to_string()));
    }

    // Check all signals have the same length
    let len = signals[0].samples.len();
    let sample_rate = signals[0].sample_rate;
    for sig in signals.iter().skip(1) {
        if sig.samples.len() != len {
            return Err(IoError::InvalidFormat(
                "All signals must have the same length".to_string(),
            ));
        }
    }

    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // Write header
    write!(writer, "index,time_s")?;
    for sig in signals {
        write!(writer, ",{}", sig.channel)?;
    }
    writeln!(writer)?;

    // Write samples
    let dt = 1.0 / sample_rate as f64;
    for i in 0..len {
        let time = i as f64 * dt;
        write!(writer, "{},{:.9}", i, time)?;
        for sig in signals {
            write!(writer, ",{:.15}", sig.samples[i])?;
        }
        writeln!(writer)?;
    }

    writer.flush()?;
    Ok(())
}

/// Read multiple columns from a CSV file (alias for read_csv_columns)
///
/// This is a convenience function that reads all specified columns.
pub fn read_csv_multi(path: &Path, columns: &[&str]) -> IoResult<Vec<DynSignal<f64>>> {
    read_csv_columns(path, columns)
}

/// Read all numeric columns from a CSV file
///
/// This function reads the CSV headers and attempts to read all columns
/// that contain valid numeric data.
pub fn read_csv_all_columns(path: &Path) -> IoResult<Vec<DynSignal<f64>>> {
    let file = File::open(path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            IoError::FileNotFound(path.display().to_string())
        } else {
            IoError::Io(e)
        }
    })?;
    let reader = BufReader::new(file);

    let mut csv_reader = ReaderBuilder::new().has_headers(true).from_reader(reader);

    // Get all headers
    let headers = csv_reader
        .headers()
        .map_err(|e| IoError::Csv(e.to_string()))?
        .clone();

    // Skip 'index' and 'time_s' columns if present (these are generated by write_csv)
    let skip_columns: std::collections::HashSet<&str> = ["index", "time_s", "time", "timestamp"]
        .iter()
        .copied()
        .collect();

    let numeric_columns: Vec<(usize, String)> = headers
        .iter()
        .enumerate()
        .filter(|(_, name)| !skip_columns.contains(name))
        .map(|(idx, name)| (idx, name.to_string()))
        .collect();

    if numeric_columns.is_empty() {
        return Err(IoError::InvalidFormat(
            "No numeric columns found".to_string(),
        ));
    }

    // Read all data into column vectors
    let mut samples_vecs: Vec<Vec<f64>> = vec![Vec::new(); numeric_columns.len()];
    let mut valid_columns: Vec<bool> = vec![true; numeric_columns.len()];

    for result in csv_reader.records() {
        let record = result.map_err(|e| IoError::Csv(e.to_string()))?;

        for (i, (col_idx, _)) in numeric_columns.iter().enumerate() {
            if !valid_columns[i] {
                continue;
            }

            if let Some(value_str) = record.get(*col_idx) {
                match value_str.trim().parse::<f64>() {
                    Ok(value) if value.is_finite() => {
                        samples_vecs[i].push(value);
                    }
                    _ => {
                        // Column contains non-numeric data, mark as invalid
                        valid_columns[i] = false;
                        samples_vecs[i].clear();
                    }
                }
            }
        }
    }

    let file_stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("csv");

    let signals: Vec<DynSignal<f64>> = numeric_columns
        .iter()
        .zip(samples_vecs.into_iter())
        .zip(valid_columns.iter())
        .filter(|((_, samples), valid)| **valid && !samples.is_empty())
        .map(|(((_, col_name), samples), _)| {
            DynSignal {
                samples,
                sample_rate: 1, // Default, caller should set appropriate rate
                channel: SmolStr::new(format!("{}.{}", file_stem, col_name)),
                start_ns: 0,
                metadata: SignalMetadata {
                    source: Some(SmolStr::new(path.display().to_string())),
                    calibrated: false,
                    noise_floor: None,
                    artifact_mask: None,
                    units: None,
                },
            }
        })
        .collect();

    if signals.is_empty() {
        return Err(IoError::InvalidFormat(
            "No valid numeric columns found".to_string(),
        ));
    }

    Ok(signals)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_csv_roundtrip() {
        // Create test signal
        let samples: Vec<f64> = (0..100).map(|i| (i as f64).sin()).collect();
        let signal = DynSignal {
            samples,
            sample_rate: 100,
            channel: SmolStr::new("test"),
            start_ns: 0,
            metadata: SignalMetadata::default(),
        };

        // Write to temp file
        let temp_file = NamedTempFile::with_suffix(".csv").unwrap();
        write_csv(temp_file.path(), &signal).unwrap();

        // Read back (the "value" column)
        let loaded = read_csv(temp_file.path(), "value").unwrap();

        assert_eq!(loaded.samples.len(), signal.samples.len());
        for (a, b) in signal.samples.iter().zip(loaded.samples.iter()) {
            assert!((a - b).abs() < 1e-10, "Mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_csv_column_not_found() {
        let temp_file = NamedTempFile::with_suffix(".csv").unwrap();
        std::fs::write(temp_file.path(), "a,b,c\n1,2,3\n").unwrap();

        let result = read_csv(temp_file.path(), "nonexistent");
        assert!(matches!(result, Err(IoError::ColumnNotFound(_))));
    }

    #[test]
    fn test_read_nonexistent_file() {
        let result = read_csv(Path::new("/nonexistent/file.csv"), "value");
        assert!(matches!(result, Err(IoError::FileNotFound(_))));
    }

    #[test]
    fn test_csv_multi_write_read() {
        let signal1 = DynSignal {
            samples: vec![1.0, 2.0, 3.0],
            sample_rate: 10,
            channel: SmolStr::new("sig1"),
            start_ns: 0,
            metadata: SignalMetadata::default(),
        };
        let signal2 = DynSignal {
            samples: vec![4.0, 5.0, 6.0],
            sample_rate: 10,
            channel: SmolStr::new("sig2"),
            start_ns: 0,
            metadata: SignalMetadata::default(),
        };

        let temp_file = NamedTempFile::with_suffix(".csv").unwrap();
        write_csv_multi(temp_file.path(), &[signal1, signal2]).unwrap();

        // Read back
        let signals = read_csv_columns(temp_file.path(), &["sig1", "sig2"]).unwrap();
        assert_eq!(signals.len(), 2);
        assert_eq!(signals[0].samples, vec![1.0, 2.0, 3.0]);
        assert_eq!(signals[1].samples, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_read_csv_all_columns() {
        let temp_file = NamedTempFile::with_suffix(".csv").unwrap();
        std::fs::write(temp_file.path(), "index,time_s,col1,col2,text\n0,0.0,1.0,4.0,hello\n1,0.1,2.0,5.0,world\n2,0.2,3.0,6.0,test\n").unwrap();

        let signals = read_csv_all_columns(temp_file.path()).unwrap();

        // Should read col1 and col2, skip index/time_s/text
        assert_eq!(signals.len(), 2);
        assert_eq!(signals[0].samples, vec![1.0, 2.0, 3.0]);
        assert_eq!(signals[1].samples, vec![4.0, 5.0, 6.0]);
    }
}
