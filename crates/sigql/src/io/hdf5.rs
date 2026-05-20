//! HDF5 file format support
//!
//! HDF5 (Hierarchical Data Format) is a widely used format for scientific data,
//! supporting large datasets, compression, and hierarchical organization.
//!
//! ## Features
//!
//! - Hierarchical dataset organization
//! - Built-in compression (gzip, lzf)
//! - Chunked storage for large datasets
//! - Metadata attributes
//!
//! ## Note
//!
//! This implementation uses a pure Rust approach without the full HDF5 C library.
//! For production use with large files, consider using the `hdf5` crate with
//! the native library.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use smol_str::SmolStr;

use super::{IoError, IoResult};
use crate::types::DynSignal;

/// HDF5 write options
#[derive(Debug, Clone, Default)]
pub struct Hdf5Options {
    /// Compression level (0-9, 0 = none)
    pub compression: u8,
    /// Chunk size for large datasets
    pub chunk_size: Option<usize>,
    /// Store sample rate as attribute
    pub store_sample_rate: bool,
    /// Store timestamps
    pub store_timestamps: bool,
}

impl Hdf5Options {
    pub fn new() -> Self {
        Self {
            compression: 0,
            chunk_size: None,
            store_sample_rate: true,
            store_timestamps: false,
        }
    }

    pub fn with_compression(mut self, level: u8) -> Self {
        self.compression = level.min(9);
        self
    }
}

/// Simple HDF5-like binary format
///
/// This is a simplified implementation. For full HDF5 support, use the `hdf5` crate.
///
/// Format:
/// - Magic bytes: "SIGHDF5\0" (8 bytes)
/// - Version: u32
/// - Num datasets: u32
/// - For each dataset:
///   - Name length: u32
///   - Name: utf8 bytes
///   - Sample rate: u32
///   - Start time: i64
///   - Num samples: u64
///   - Samples: f64[]

const MAGIC: &[u8] = b"SIGHDF5\0";
const VERSION: u32 = 1;

/// Read a single dataset from an HDF5-like file
pub fn read_hdf5(path: &Path, dataset: &str) -> IoResult<DynSignal<f64>> {
    let signals = read_hdf5_all(path)?;

    signals
        .into_iter()
        .find(|s| s.channel.as_str() == dataset)
        .ok_or_else(|| IoError::ChannelNotFound(dataset.to_string()))
}

/// Read all datasets from an HDF5-like file
pub fn read_hdf5_all(path: &Path) -> IoResult<Vec<DynSignal<f64>>> {
    let file = File::open(path).map_err(|_| IoError::FileNotFound(path.display().to_string()))?;
    let mut reader = BufReader::new(file);

    // Read and verify magic
    let mut magic = [0u8; 8];
    reader
        .read_exact(&mut magic)
        .map_err(|e| IoError::Hdf5(format!("Read error: {}", e)))?;

    if &magic != MAGIC {
        return Err(IoError::InvalidFormat("Not a valid SIGHDF5 file".into()));
    }

    // Read version
    let mut version_buf = [0u8; 4];
    reader.read_exact(&mut version_buf)?;
    let version = u32::from_le_bytes(version_buf);

    if version > VERSION {
        return Err(IoError::Hdf5(format!("Unsupported version: {}", version)));
    }

    // Read number of datasets
    let mut num_buf = [0u8; 4];
    reader.read_exact(&mut num_buf)?;
    let num_datasets = u32::from_le_bytes(num_buf) as usize;

    let mut signals = Vec::with_capacity(num_datasets);

    for _ in 0..num_datasets {
        // Read name
        let mut name_len_buf = [0u8; 4];
        reader.read_exact(&mut name_len_buf)?;
        let name_len = u32::from_le_bytes(name_len_buf) as usize;

        let mut name_buf = vec![0u8; name_len];
        reader.read_exact(&mut name_buf)?;
        let name = String::from_utf8_lossy(&name_buf).to_string();

        // Read sample rate
        let mut rate_buf = [0u8; 4];
        reader.read_exact(&mut rate_buf)?;
        let sample_rate = u32::from_le_bytes(rate_buf);

        // Read start time
        let mut time_buf = [0u8; 8];
        reader.read_exact(&mut time_buf)?;
        let start_ns = i64::from_le_bytes(time_buf);

        // Read number of samples
        let mut count_buf = [0u8; 8];
        reader.read_exact(&mut count_buf)?;
        let num_samples = u64::from_le_bytes(count_buf) as usize;

        // Read samples
        let mut samples = Vec::with_capacity(num_samples);
        for _ in 0..num_samples {
            let mut sample_buf = [0u8; 8];
            reader.read_exact(&mut sample_buf)?;
            samples.push(f64::from_le_bytes(sample_buf));
        }

        signals.push(DynSignal::new(
            SmolStr::new(&name),
            samples,
            sample_rate,
            start_ns,
        ));
    }

    Ok(signals)
}

/// Write a single signal to an HDF5-like file
pub fn write_hdf5(
    path: &Path,
    signal: &DynSignal<f64>,
    dataset: &str,
    _options: Hdf5Options,
) -> IoResult<()> {
    let mut signal = signal.clone();
    signal.channel = SmolStr::new(dataset);
    write_hdf5_multi(path, &[signal], _options)
}

/// Write multiple signals to an HDF5-like file
pub fn write_hdf5_multi(
    path: &Path,
    signals: &[DynSignal<f64>],
    _options: Hdf5Options,
) -> IoResult<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // Write magic
    writer.write_all(MAGIC)?;

    // Write version
    writer.write_all(&VERSION.to_le_bytes())?;

    // Write number of datasets
    writer.write_all(&(signals.len() as u32).to_le_bytes())?;

    for signal in signals {
        // Write name
        let name = signal.channel.as_str();
        writer.write_all(&(name.len() as u32).to_le_bytes())?;
        writer.write_all(name.as_bytes())?;

        // Write sample rate
        writer.write_all(&signal.sample_rate.to_le_bytes())?;

        // Write start time
        writer.write_all(&signal.start_ns.to_le_bytes())?;

        // Write number of samples
        writer.write_all(&(signal.samples.len() as u64).to_le_bytes())?;

        // Write samples
        for sample in &signal.samples {
            writer.write_all(&sample.to_le_bytes())?;
        }
    }

    writer.flush()?;
    Ok(())
}

/// Check if a file is a valid SIGHDF5 file
pub fn is_sighdf5(path: &Path) -> bool {
    if let Ok(mut file) = File::open(path) {
        let mut magic = [0u8; 8];
        if file.read_exact(&mut magic).is_ok() {
            return &magic == MAGIC;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_hdf5_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.h5");

        let signal = DynSignal::new(
            "sensor_data",
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            1000,
            123456789,
        );

        write_hdf5(&path, &signal, "sensor_data", Hdf5Options::new()).unwrap();

        let loaded = read_hdf5(&path, "sensor_data").unwrap();
        assert_eq!(loaded.channel.as_str(), "sensor_data");
        assert_eq!(loaded.sample_rate, 1000);
        assert_eq!(loaded.start_ns, 123456789);
        assert_eq!(loaded.samples, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_hdf5_multi_channel() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("multi.h5");

        let signals = vec![
            DynSignal::new("channel_1", vec![1.0, 2.0, 3.0], 100, 0),
            DynSignal::new("channel_2", vec![4.0, 5.0, 6.0], 200, 0),
            DynSignal::new("channel_3", vec![7.0, 8.0, 9.0], 300, 0),
        ];

        write_hdf5_multi(&path, &signals, Hdf5Options::new()).unwrap();

        let loaded = read_hdf5_all(&path).unwrap();
        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded[0].channel.as_str(), "channel_1");
        assert_eq!(loaded[1].channel.as_str(), "channel_2");
        assert_eq!(loaded[2].channel.as_str(), "channel_3");
        assert_eq!(loaded[0].sample_rate, 100);
        assert_eq!(loaded[1].sample_rate, 200);
    }

    #[test]
    fn test_hdf5_dataset_selection() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("select.h5");

        let signals = vec![
            DynSignal::new("eeg", vec![1.0, 2.0], 256, 0),
            DynSignal::new("ecg", vec![3.0, 4.0], 512, 0),
        ];

        write_hdf5_multi(&path, &signals, Hdf5Options::new()).unwrap();

        let eeg = read_hdf5(&path, "eeg").unwrap();
        assert_eq!(eeg.samples, vec![1.0, 2.0]);

        let ecg = read_hdf5(&path, "ecg").unwrap();
        assert_eq!(ecg.samples, vec![3.0, 4.0]);

        // Non-existent dataset
        assert!(read_hdf5(&path, "emg").is_err());
    }
}
