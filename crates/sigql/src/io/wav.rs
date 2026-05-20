//! WAV file I/O using the hound crate
//!
//! Supports reading and writing WAV files with automatic conversion to f64 signals.

use std::path::Path;

use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use smol_str::SmolStr;

use super::{IoError, IoResult};
use crate::types::{DynSignal, SignalMetadata};

/// Read a WAV file into a DynSignal
///
/// Automatically converts to f64 samples regardless of the original format.
/// For stereo files, returns the first channel only.
///
/// # Example
///
/// ```ignore
/// use std::path::Path;
/// use sigql::io::read_wav;
///
/// let signal = read_wav(Path::new("audio.wav"))?;
/// println!("Sample rate: {} Hz", signal.sample_rate);
/// println!("Samples: {}", signal.samples.len());
/// ```
pub fn read_wav(path: &Path) -> IoResult<DynSignal<f64>> {
    let reader = WavReader::open(path).map_err(|e| {
        if e.to_string().contains("No such file") {
            IoError::FileNotFound(path.display().to_string())
        } else {
            IoError::Wav(e.to_string())
        }
    })?;

    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let channels = spec.channels as usize;
    let bits_per_sample = spec.bits_per_sample;
    let sample_format = spec.sample_format;

    let samples: Vec<f64> = match sample_format {
        SampleFormat::Int => {
            let max_value = (1i64 << (bits_per_sample - 1)) as f64;
            reader
                .into_samples::<i32>()
                .enumerate()
                .filter_map(|(i, s)| {
                    // Only take first channel for multi-channel files
                    if i % channels == 0 {
                        s.ok().map(|v| v as f64 / max_value)
                    } else {
                        s.ok(); // consume but discard
                        None
                    }
                })
                .collect()
        }
        SampleFormat::Float => {
            reader
                .into_samples::<f32>()
                .enumerate()
                .filter_map(|(i, s)| {
                    // Only take first channel for multi-channel files
                    if i % channels == 0 {
                        s.ok().map(|v| v as f64)
                    } else {
                        s.ok(); // consume but discard
                        None
                    }
                })
                .collect()
        }
    };

    let channel_name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("wav_signal");

    Ok(DynSignal {
        samples,
        sample_rate,
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

/// Write a DynSignal to a WAV file
///
/// Writes as 32-bit float format for maximum precision.
///
/// # Example
///
/// ```ignore
/// use std::path::Path;
/// use sigql::io::write_wav;
///
/// write_wav(Path::new("output.wav"), &signal)?;
/// ```
pub fn write_wav(path: &Path, signal: &DynSignal<f64>) -> IoResult<()> {
    let spec = WavSpec {
        channels: 1,
        sample_rate: signal.sample_rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };

    let mut writer = WavWriter::create(path, spec).map_err(|e| IoError::Wav(e.to_string()))?;

    for &sample in &signal.samples {
        writer
            .write_sample(sample as f32)
            .map_err(|e| IoError::Wav(e.to_string()))?;
    }

    writer.finalize().map_err(|e| IoError::Wav(e.to_string()))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;
    use tempfile::NamedTempFile;

    #[test]
    fn test_wav_roundtrip() {
        // Create a test signal
        let sample_rate = 44100;
        let freq = 440.0;
        let samples: Vec<f64> = (0..sample_rate)
            .map(|i| (2.0 * PI * freq * i as f64 / sample_rate as f64).sin())
            .collect();

        let signal = DynSignal {
            samples,
            sample_rate,
            channel: SmolStr::new("test"),
            start_ns: 0,
            metadata: SignalMetadata::default(),
        };

        // Write to temp file
        let temp_file = NamedTempFile::with_suffix(".wav").unwrap();
        write_wav(temp_file.path(), &signal).unwrap();

        // Read back
        let loaded = read_wav(temp_file.path()).unwrap();

        assert_eq!(loaded.sample_rate, sample_rate);
        assert_eq!(loaded.samples.len(), signal.samples.len());

        // Check values are approximately equal (float32 precision loss)
        for (a, b) in signal.samples.iter().zip(loaded.samples.iter()) {
            assert!((a - b).abs() < 1e-6, "Mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_read_nonexistent_file() {
        let result = read_wav(Path::new("/nonexistent/file.wav"));
        assert!(result.is_err());
        match result {
            Err(IoError::FileNotFound(_)) | Err(IoError::Wav(_)) => {}
            _ => panic!("Expected FileNotFound or Wav error"),
        }
    }
}
