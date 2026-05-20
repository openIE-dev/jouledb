//! EDF/EDF+ file format support
//!
//! European Data Format is the standard format for medical time series data,
//! particularly EEG, ECG, and polysomnography recordings.
//!
//! ## Format Overview
//!
//! EDF files consist of:
//! - A fixed-size header with recording information
//! - Per-signal headers with calibration and labeling
//! - Data records containing interleaved samples
//!
//! ## Supported Variants
//!
//! - EDF: Original European Data Format
//! - EDF+: Extended format with annotations
//! - BDF: BioSemi Data Format (24-bit variant)

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use smol_str::SmolStr;

use super::traits::SignalMetadata;
use super::{IoError, IoResult};
use crate::types::DynSignal;

/// EDF file header information
#[derive(Debug, Clone, Default)]
pub struct EdfHeader {
    /// Patient identification
    pub patient_id: String,
    /// Recording identification
    pub recording_id: String,
    /// Start date (DD.MM.YY)
    pub start_date: String,
    /// Start time (HH.MM.SS)
    pub start_time: String,
    /// Number of data records
    pub num_records: i32,
    /// Duration of each record in seconds
    pub record_duration: f64,
    /// Version (0 for EDF, 1 for EDF+)
    pub version: u8,
}

impl EdfHeader {
    pub fn new() -> Self {
        let now = chrono::Local::now();
        Self {
            patient_id: "X X X X".to_string(),
            recording_id: "Startdate X X X X".to_string(),
            start_date: now.format("%d.%m.%y").to_string(),
            start_time: now.format("%H.%M.%S").to_string(),
            num_records: -1, // Will be calculated
            record_duration: 1.0,
            version: 0,
        }
    }

    pub fn with_patient(mut self, patient_id: impl Into<String>) -> Self {
        self.patient_id = patient_id.into();
        self
    }

    pub fn with_recording(mut self, recording_id: impl Into<String>) -> Self {
        self.recording_id = recording_id.into();
        self
    }
}

/// Information about a single EDF signal/channel
#[derive(Debug, Clone)]
pub struct EdfSignalInfo {
    /// Signal label (e.g., "EEG Fp1")
    pub label: String,
    /// Transducer type
    pub transducer: String,
    /// Physical dimension (e.g., "uV")
    pub physical_dimension: String,
    /// Physical minimum
    pub physical_min: f64,
    /// Physical maximum
    pub physical_max: f64,
    /// Digital minimum
    pub digital_min: i32,
    /// Digital maximum
    pub digital_max: i32,
    /// Prefiltering info
    pub prefiltering: String,
    /// Number of samples per record
    pub samples_per_record: usize,
}

impl Default for EdfSignalInfo {
    fn default() -> Self {
        Self {
            label: "Signal".to_string(),
            transducer: "Unknown".to_string(),
            physical_dimension: "uV".to_string(),
            physical_min: -3200.0,
            physical_max: 3200.0,
            digital_min: -32768,
            digital_max: 32767,
            prefiltering: "None".to_string(),
            samples_per_record: 256,
        }
    }
}

impl EdfSignalInfo {
    /// Create from a signal with auto-detected parameters
    pub fn from_signal(signal: &DynSignal<f64>, record_duration: f64) -> Self {
        let (min, max) = signal
            .samples
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &x| {
                (min.min(x), max.max(x))
            });

        // Add 10% margin
        let margin = (max - min) * 0.1;
        let physical_min = min - margin;
        let physical_max = max + margin;

        let samples_per_record = (signal.sample_rate as f64 * record_duration) as usize;

        Self {
            label: signal.channel.to_string(),
            transducer: "Unknown".to_string(),
            physical_dimension: "uV".to_string(),
            physical_min,
            physical_max,
            digital_min: -32768,
            digital_max: 32767,
            prefiltering: "None".to_string(),
            samples_per_record,
        }
    }

    /// Calculate gain for digital to physical conversion
    fn gain(&self) -> f64 {
        (self.physical_max - self.physical_min) / (self.digital_max - self.digital_min) as f64
    }

    /// Calculate offset for digital to physical conversion
    fn offset(&self) -> f64 {
        self.physical_min - self.gain() * self.digital_min as f64
    }
}

/// Read all signals from an EDF file
pub fn read_edf(path: &Path) -> IoResult<Vec<DynSignal<f64>>> {
    let file = File::open(path).map_err(|_| IoError::FileNotFound(path.display().to_string()))?;
    let mut reader = BufReader::new(file);

    // Read header
    let (header, signals_info) = read_edf_header(&mut reader)?;

    // Calculate total samples
    let num_signals = signals_info.len();
    if num_signals == 0 {
        return Ok(vec![]);
    }

    // Read data records
    let mut all_samples: Vec<Vec<f64>> = vec![Vec::new(); num_signals];

    let num_records = if header.num_records < 0 {
        // Calculate from file size
        let current_pos = reader.stream_position().unwrap_or(0);
        let file_size = reader.seek(SeekFrom::End(0)).unwrap_or(0);
        reader.seek(SeekFrom::Start(current_pos)).ok();

        let samples_per_record: usize = signals_info.iter().map(|s| s.samples_per_record).sum();
        let bytes_per_record = samples_per_record * 2; // 16-bit samples
        let data_size = file_size - current_pos;
        (data_size / bytes_per_record as u64) as i32
    } else {
        header.num_records
    };

    for _ in 0..num_records {
        for (i, info) in signals_info.iter().enumerate() {
            let mut buf = vec![0u8; info.samples_per_record * 2];
            reader
                .read_exact(&mut buf)
                .map_err(|e| IoError::Edf(format!("Read error: {}", e)))?;

            // Convert 16-bit integers to f64
            for chunk in buf.chunks_exact(2) {
                let digital = i16::from_le_bytes([chunk[0], chunk[1]]) as i32;
                let physical = info.gain() * digital as f64 + info.offset();
                all_samples[i].push(physical);
            }
        }
    }

    // Parse start time from header
    let start_timestamp = parse_edf_datetime(&header.start_date, &header.start_time);

    // Create signals
    let mut signals = Vec::new();
    for (i, info) in signals_info.iter().enumerate() {
        let sample_rate = (info.samples_per_record as f64 / header.record_duration) as u32;
        let signal = DynSignal::new(
            SmolStr::new(&info.label),
            std::mem::take(&mut all_samples[i]),
            sample_rate,
            start_timestamp,
        );
        signals.push(signal);
    }

    Ok(signals)
}

/// Parse EDF date (DD.MM.YY) and time (HH.MM.SS) strings into a Unix timestamp (nanoseconds)
fn parse_edf_datetime(date_str: &str, time_str: &str) -> i64 {
    // Parse date: DD.MM.YY
    let date_parts: Vec<&str> = date_str.split('.').collect();
    if date_parts.len() != 3 {
        return 0;
    }

    let day: u32 = date_parts[0].trim().parse().unwrap_or(1);
    let month: u32 = date_parts[1].trim().parse().unwrap_or(1);
    let year_2digit: u32 = date_parts[2].trim().parse().unwrap_or(0);

    // EDF uses 2-digit years: 85-99 -> 1985-1999, 00-84 -> 2000-2084
    let year = if year_2digit >= 85 {
        1900 + year_2digit
    } else {
        2000 + year_2digit
    };

    // Parse time: HH.MM.SS
    let time_parts: Vec<&str> = time_str.split('.').collect();
    let (hour, minute, second) = if time_parts.len() == 3 {
        (
            time_parts[0].trim().parse().unwrap_or(0),
            time_parts[1].trim().parse().unwrap_or(0),
            time_parts[2].trim().parse().unwrap_or(0),
        )
    } else {
        (0u32, 0u32, 0u32)
    };

    // Use chrono to create timestamp
    use chrono::{NaiveDate, NaiveDateTime, NaiveTime, TimeZone, Utc};

    if let Some(date) = NaiveDate::from_ymd_opt(year as i32, month, day) {
        if let Some(time) = NaiveTime::from_hms_opt(hour, minute, second) {
            let datetime = NaiveDateTime::new(date, time);
            // Convert to Unix timestamp in nanoseconds
            if let Some(utc_datetime) = Utc.from_local_datetime(&datetime).single() {
                return utc_datetime.timestamp_nanos_opt().unwrap_or(0);
            }
        }
    }

    0 // Return 0 if parsing fails
}

/// Read only the metadata from an EDF file
pub fn read_edf_metadata(path: &Path) -> IoResult<Vec<SignalMetadata>> {
    let file = File::open(path).map_err(|_| IoError::FileNotFound(path.display().to_string()))?;
    let mut reader = BufReader::new(file);

    let (header, signals_info) = read_edf_header(&mut reader)?;

    let metadata: Vec<SignalMetadata> = signals_info
        .iter()
        .map(|info| {
            let sample_rate = (info.samples_per_record as f64 / header.record_duration) as u32;
            SignalMetadata {
                label: info.label.clone(),
                physical_dimension: Some(info.physical_dimension.clone()),
                sample_rate,
                physical_min: Some(info.physical_min),
                physical_max: Some(info.physical_max),
                transducer: Some(info.transducer.clone()),
                prefiltering: Some(info.prefiltering.clone()),
                ..Default::default()
            }
        })
        .collect();

    Ok(metadata)
}

/// Read EDF header and signal info
fn read_edf_header(reader: &mut BufReader<File>) -> IoResult<(EdfHeader, Vec<EdfSignalInfo>)> {
    // Read fixed header (256 bytes)
    let mut header_buf = [0u8; 256];
    reader
        .read_exact(&mut header_buf)
        .map_err(|e| IoError::Edf(format!("Header read error: {}", e)))?;

    // Parse header fields
    let version = parse_ascii(&header_buf[0..8]);
    let patient_id = parse_ascii(&header_buf[8..88]);
    let recording_id = parse_ascii(&header_buf[88..168]);
    let start_date = parse_ascii(&header_buf[168..176]);
    let start_time = parse_ascii(&header_buf[176..184]);
    let _header_bytes: usize = parse_ascii(&header_buf[184..192])
        .trim()
        .parse()
        .unwrap_or(256);
    let _reserved = parse_ascii(&header_buf[192..236]);
    let num_records: i32 = parse_ascii(&header_buf[236..244])
        .trim()
        .parse()
        .unwrap_or(-1);
    let record_duration: f64 = parse_ascii(&header_buf[244..252])
        .trim()
        .parse()
        .unwrap_or(1.0);
    let num_signals: usize = parse_ascii(&header_buf[252..256])
        .trim()
        .parse()
        .unwrap_or(0);

    let header = EdfHeader {
        patient_id,
        recording_id,
        start_date,
        start_time,
        num_records,
        record_duration,
        version: if version.starts_with("0") { 0 } else { 1 },
    };

    if num_signals == 0 {
        return Ok((header, vec![]));
    }

    // Read signal headers (256 bytes per signal)
    let signal_header_size = num_signals * 256;
    let mut signal_buf = vec![0u8; signal_header_size];
    reader
        .read_exact(&mut signal_buf)
        .map_err(|e| IoError::Edf(format!("Signal header read error: {}", e)))?;

    // Parse signal headers (each field is stored contiguously for all signals)
    let mut signals_info = Vec::with_capacity(num_signals);

    for i in 0..num_signals {
        let label = parse_ascii(&signal_buf[i * 16..(i + 1) * 16]);
        let transducer =
            parse_ascii(&signal_buf[num_signals * 16 + i * 80..num_signals * 16 + (i + 1) * 80]);
        let physical_dimension =
            parse_ascii(&signal_buf[num_signals * 96 + i * 8..num_signals * 96 + (i + 1) * 8]);
        let physical_min: f64 =
            parse_ascii(&signal_buf[num_signals * 104 + i * 8..num_signals * 104 + (i + 1) * 8])
                .trim()
                .parse()
                .unwrap_or(-3200.0);
        let physical_max: f64 =
            parse_ascii(&signal_buf[num_signals * 112 + i * 8..num_signals * 112 + (i + 1) * 8])
                .trim()
                .parse()
                .unwrap_or(3200.0);
        let digital_min: i32 =
            parse_ascii(&signal_buf[num_signals * 120 + i * 8..num_signals * 120 + (i + 1) * 8])
                .trim()
                .parse()
                .unwrap_or(-32768);
        let digital_max: i32 =
            parse_ascii(&signal_buf[num_signals * 128 + i * 8..num_signals * 128 + (i + 1) * 8])
                .trim()
                .parse()
                .unwrap_or(32767);
        let prefiltering =
            parse_ascii(&signal_buf[num_signals * 136 + i * 80..num_signals * 136 + (i + 1) * 80]);
        let samples_per_record: usize =
            parse_ascii(&signal_buf[num_signals * 216 + i * 8..num_signals * 216 + (i + 1) * 8])
                .trim()
                .parse()
                .unwrap_or(256);

        signals_info.push(EdfSignalInfo {
            label,
            transducer,
            physical_dimension,
            physical_min,
            physical_max,
            digital_min,
            digital_max,
            prefiltering,
            samples_per_record,
        });
    }

    Ok((header, signals_info))
}

/// Write signals to an EDF file
pub fn write_edf(path: &Path, signals: &[DynSignal<f64>], header: EdfHeader) -> IoResult<()> {
    if signals.is_empty() {
        return Err(IoError::Edf("No signals to write".into()));
    }

    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // Prepare signal info
    let signals_info: Vec<EdfSignalInfo> = signals
        .iter()
        .map(|s| EdfSignalInfo::from_signal(s, header.record_duration))
        .collect();

    // Calculate number of records
    let max_samples = signals.iter().map(|s| s.samples.len()).max().unwrap_or(0);
    let samples_per_record = signals_info[0].samples_per_record;
    let num_records = (max_samples + samples_per_record - 1) / samples_per_record;

    // Write main header
    write_edf_header(&mut writer, &header, &signals_info, num_records)?;

    // Write data records
    for record_idx in 0..num_records {
        for (sig_idx, signal) in signals.iter().enumerate() {
            let info = &signals_info[sig_idx];
            let start = record_idx * info.samples_per_record;
            let _end = (start + info.samples_per_record).min(signal.samples.len());

            // Convert to 16-bit integers
            for i in start..info.samples_per_record * (record_idx + 1) {
                let physical = if i < signal.samples.len() {
                    signal.samples[i]
                } else {
                    0.0 // Pad with zeros
                };

                // Convert to digital value
                let digital = ((physical - info.offset()) / info.gain())
                    .round()
                    .clamp(info.digital_min as f64, info.digital_max as f64)
                    as i16;

                writer
                    .write_all(&digital.to_le_bytes())
                    .map_err(|e| IoError::Edf(format!("Write error: {}", e)))?;
            }
        }
    }

    writer.flush()?;
    Ok(())
}

/// Write EDF header
fn write_edf_header(
    writer: &mut BufWriter<File>,
    header: &EdfHeader,
    signals_info: &[EdfSignalInfo],
    num_records: usize,
) -> IoResult<()> {
    let num_signals = signals_info.len();
    let header_bytes = 256 + num_signals * 256;

    // Main header (256 bytes)
    write_ascii(writer, "0       ", 8)?; // Version
    write_ascii(writer, &header.patient_id, 80)?;
    write_ascii(writer, &header.recording_id, 80)?;
    write_ascii(writer, &header.start_date, 8)?;
    write_ascii(writer, &header.start_time, 8)?;
    write_ascii(writer, &header_bytes.to_string(), 8)?;
    write_ascii(writer, "", 44)?; // Reserved
    write_ascii(writer, &num_records.to_string(), 8)?;
    write_ascii(writer, &format!("{:.6}", header.record_duration), 8)?;
    write_ascii(writer, &num_signals.to_string(), 4)?;

    // Signal headers (each field for all signals, then next field)
    // Labels (16 bytes each)
    for info in signals_info {
        write_ascii(writer, &info.label, 16)?;
    }

    // Transducer (80 bytes each)
    for info in signals_info {
        write_ascii(writer, &info.transducer, 80)?;
    }

    // Physical dimension (8 bytes each)
    for info in signals_info {
        write_ascii(writer, &info.physical_dimension, 8)?;
    }

    // Physical min (8 bytes each)
    for info in signals_info {
        write_ascii(writer, &format!("{:.6}", info.physical_min), 8)?;
    }

    // Physical max (8 bytes each)
    for info in signals_info {
        write_ascii(writer, &format!("{:.6}", info.physical_max), 8)?;
    }

    // Digital min (8 bytes each)
    for info in signals_info {
        write_ascii(writer, &info.digital_min.to_string(), 8)?;
    }

    // Digital max (8 bytes each)
    for info in signals_info {
        write_ascii(writer, &info.digital_max.to_string(), 8)?;
    }

    // Prefiltering (80 bytes each)
    for info in signals_info {
        write_ascii(writer, &info.prefiltering, 80)?;
    }

    // Samples per record (8 bytes each)
    for info in signals_info {
        write_ascii(writer, &info.samples_per_record.to_string(), 8)?;
    }

    // Reserved (32 bytes each)
    for _ in signals_info {
        write_ascii(writer, "", 32)?;
    }

    Ok(())
}

/// Parse ASCII string from bytes
fn parse_ascii(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).trim().to_string()
}

/// Write ASCII string padded to length
fn write_ascii(writer: &mut BufWriter<File>, s: &str, len: usize) -> IoResult<()> {
    let s = if s.len() > len { &s[..len] } else { s };
    let mut buf = vec![b' '; len];
    buf[..s.len()].copy_from_slice(s.as_bytes());
    writer
        .write_all(&buf)
        .map_err(|e| IoError::Edf(format!("Write error: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;
    use tempfile::tempdir;

    #[test]
    fn test_edf_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.edf");

        // Create test signal (10 Hz sine wave)
        let sample_rate = 256;
        let duration = 2.0;
        let samples: Vec<f64> = (0..(sample_rate as f64 * duration) as usize)
            .map(|i| 100.0 * (2.0 * PI * 10.0 * i as f64 / sample_rate as f64).sin())
            .collect();

        let signal = DynSignal::new("EEG Fp1", samples, sample_rate, 0);

        // Write
        let header = EdfHeader::new().with_patient("Test Patient");
        write_edf(&path, &[signal.clone()], header).unwrap();

        // Read back
        let loaded = read_edf(&path).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].sample_rate, sample_rate);
        assert_eq!(loaded[0].samples.len(), signal.samples.len());

        // Check values (with some tolerance due to 16-bit quantization)
        for (a, b) in loaded[0].samples.iter().zip(signal.samples.iter()) {
            assert!((a - b).abs() < 0.5, "Sample mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_edf_multi_channel() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("multi.edf");

        let sample_rate = 256;
        let samples1: Vec<f64> = (0..512).map(|i| (i as f64 * 0.1).sin() * 100.0).collect();
        let samples2: Vec<f64> = (0..512).map(|i| (i as f64 * 0.2).cos() * 50.0).collect();

        let signals = vec![
            DynSignal::new("EEG Fp1", samples1, sample_rate, 0),
            DynSignal::new("EEG Fp2", samples2, sample_rate, 0),
        ];

        write_edf(&path, &signals, EdfHeader::new()).unwrap();

        let loaded = read_edf(&path).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].channel.as_str(), "EEG Fp1");
        assert_eq!(loaded[1].channel.as_str(), "EEG Fp2");
    }

    #[test]
    fn test_edf_metadata() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("meta.edf");

        let signal = DynSignal::new("ECG", vec![0.0; 256], 256, 0);
        write_edf(&path, &[signal], EdfHeader::new()).unwrap();

        let metadata = read_edf_metadata(&path).unwrap();
        assert_eq!(metadata.len(), 1);
        assert_eq!(metadata[0].label, "ECG");
        assert_eq!(metadata[0].sample_rate, 256);
    }
}
