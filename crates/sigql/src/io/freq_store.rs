//! Frequency-Domain Storage Format (.freq)
//!
//! On-disk format for storing frequency coefficients (DCT/FFT/STFT) with:
//! - **Sparse storage**: only significant coefficients stored (most are near-zero)
//! - **Progressive loading**: low-frequency coefficients first → fast thumbnails
//! - **Compact encoding**: magnitude + phase as f32 pairs
//!
//! File layout:
//! ```text
//! [Header: 32 bytes]
//!   magic: "FREQ" (4 bytes)
//!   version: u8
//!   transform: u8 (0=Dct2d, 1=Stft, 2=Fft2d, 3=Dwt2d)
//!   quality: u8
//!   _pad: u8
//!   shape_x: u32
//!   shape_y: u32
//!   entry_count: u64
//!   phash: u64
//!
//! [Entries: entry_count × 12 bytes each]
//!   position_x: u16
//!   position_y: u16
//!   magnitude: f32
//!   phase: f32
//! ```
//!
//! Entries are sorted by frequency magnitude (descending), so reading the first N
//! entries gives an increasingly accurate approximation.

use crate::io::{IoError, IoResult};
use crate::types::signal::{
    CoefficientEntry, FreqTransform, FrequencyCoefficients, IngestedMedia, MediaTypeTag,
};
use std::io::{Read, Write};
use std::path::Path;

const MAGIC: [u8; 4] = *b"FREQ";
const VERSION: u8 = 1;
const HEADER_SIZE: usize = 32;
const ENTRY_SIZE: usize = 12; // 2 + 2 + 4 + 4

/// Write an IngestedMedia to a .freq file.
pub fn write_freq(path: &Path, media: &IngestedMedia) -> IoResult<()> {
    let mut file = std::fs::File::create(path)?;

    // Sort entries by magnitude descending (progressive loading)
    let mut entries = media.coefficients.entries.clone();
    entries.sort_by(|a, b| {
        b.magnitude
            .partial_cmp(&a.magnitude)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Header
    file.write_all(&MAGIC)?;
    file.write_all(&[VERSION])?;
    file.write_all(&[transform_to_u8(media.coefficients.transform)])?;
    file.write_all(&[media.coefficients.quality])?;
    file.write_all(&[0u8])?; // padding
    file.write_all(&media.coefficients.shape.0.to_le_bytes())?;
    file.write_all(&media.coefficients.shape.1.to_le_bytes())?;
    file.write_all(&(entries.len() as u64).to_le_bytes())?;
    file.write_all(&media.phash.to_le_bytes())?;

    // Entries
    for entry in &entries {
        file.write_all(&(entry.position.0 as u16).to_le_bytes())?;
        file.write_all(&(entry.position.1 as u16).to_le_bytes())?;
        file.write_all(&entry.magnitude.to_le_bytes())?;
        file.write_all(&entry.phase.to_le_bytes())?;
    }

    Ok(())
}

/// Read an IngestedMedia from a .freq file.
pub fn read_freq(path: &Path) -> IoResult<IngestedMedia> {
    read_freq_progressive(path, None)
}

/// Read with progressive loading: only the first `max_entries` coefficients.
/// Coefficients are stored magnitude-descending, so fewer entries = lower quality preview.
pub fn read_freq_progressive(path: &Path, max_entries: Option<usize>) -> IoResult<IngestedMedia> {
    let mut file =
        std::fs::File::open(path).map_err(|_| IoError::FileNotFound(path.display().to_string()))?;

    // Read header
    let mut header = [0u8; HEADER_SIZE];
    file.read_exact(&mut header)?;

    if &header[0..4] != &MAGIC {
        return Err(IoError::InvalidFormat("Not a .freq file".into()));
    }

    let version = header[4];
    if version != VERSION {
        return Err(IoError::InvalidFormat(format!(
            "Unsupported version: {}",
            version
        )));
    }

    let transform = u8_to_transform(header[5])?;
    let quality = header[6];
    let shape_x = u32::from_le_bytes([header[8], header[9], header[10], header[11]]);
    let shape_y = u32::from_le_bytes([header[12], header[13], header[14], header[15]]);
    let entry_count = u64::from_le_bytes([
        header[16], header[17], header[18], header[19], header[20], header[21], header[22],
        header[23],
    ]) as usize;
    let phash = u64::from_le_bytes([
        header[24], header[25], header[26], header[27], header[28], header[29], header[30],
        header[31],
    ]);

    // Read entries (progressive: only up to max_entries)
    let read_count = max_entries.map(|m| m.min(entry_count)).unwrap_or(entry_count);
    let mut entries = Vec::with_capacity(read_count);

    let mut entry_buf = [0u8; ENTRY_SIZE];
    for _ in 0..read_count {
        file.read_exact(&mut entry_buf)?;
        let pos_x = u16::from_le_bytes([entry_buf[0], entry_buf[1]]) as u32;
        let pos_y = u16::from_le_bytes([entry_buf[2], entry_buf[3]]) as u32;
        let magnitude = f32::from_le_bytes([entry_buf[4], entry_buf[5], entry_buf[6], entry_buf[7]]);
        let phase = f32::from_le_bytes([entry_buf[8], entry_buf[9], entry_buf[10], entry_buf[11]]);

        entries.push(CoefficientEntry {
            position: (pos_x, pos_y),
            magnitude,
            phase,
        });
    }

    let media_type = match transform {
        FreqTransform::Dct2d | FreqTransform::Fft2d | FreqTransform::Dwt2d => MediaTypeTag::Image,
        FreqTransform::Stft => MediaTypeTag::Audio,
    };

    Ok(IngestedMedia {
        coefficients: FrequencyCoefficients {
            entries,
            shape: (shape_x, shape_y),
            transform,
            quality,
        },
        phash,
        width: shape_x,
        height: shape_y,
        media_type,
    })
}

/// Get the compression ratio: original size vs stored size.
pub fn compression_info(media: &IngestedMedia) -> CompressionInfo {
    let original_size = media.width as usize * media.height as usize * 4; // f32 per pixel
    let stored_size = HEADER_SIZE + media.coefficients.entries.len() * ENTRY_SIZE;
    let ratio = if stored_size > 0 {
        original_size as f64 / stored_size as f64
    } else {
        0.0
    };

    CompressionInfo {
        original_bytes: original_size,
        stored_bytes: stored_size,
        compression_ratio: ratio,
        coefficient_count: media.coefficients.entries.len(),
        sparsity: 1.0
            - (media.coefficients.entries.len() as f64
                / (media.width as f64 * media.height as f64)),
    }
}

/// Compression statistics.
#[derive(Debug, Clone)]
pub struct CompressionInfo {
    pub original_bytes: usize,
    pub stored_bytes: usize,
    pub compression_ratio: f64,
    pub coefficient_count: usize,
    /// Fraction of zero/dropped coefficients (0.0 = dense, 1.0 = empty)
    pub sparsity: f64,
}

fn transform_to_u8(t: FreqTransform) -> u8 {
    match t {
        FreqTransform::Dct2d => 0,
        FreqTransform::Stft => 1,
        FreqTransform::Fft2d => 2,
        FreqTransform::Dwt2d => 3,
    }
}

fn u8_to_transform(b: u8) -> IoResult<FreqTransform> {
    match b {
        0 => Ok(FreqTransform::Dct2d),
        1 => Ok(FreqTransform::Stft),
        2 => Ok(FreqTransform::Fft2d),
        3 => Ok(FreqTransform::Dwt2d),
        _ => Err(IoError::InvalidFormat(format!(
            "Unknown transform type: {}",
            b
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::media_ingest::{MediaIngestConfig, MediaIngestPipeline};
    use tempfile::tempdir;

    #[test]
    fn test_freq_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.freq");

        // Create a test image and ingest it
        let pipeline = MediaIngestPipeline::new(MediaIngestConfig {
            quality: 90,
            coefficient_threshold: 0.001,
            ..Default::default()
        });

        let width = 16u32;
        let height = 16u32;
        let pixels: Vec<f64> = (0..width * height)
            .map(|i| (i as f64) / (width * height) as f64)
            .collect();

        let ingested = pipeline.ingest_image(&pixels, width, height).unwrap();

        // Write
        write_freq(&path, &ingested).unwrap();

        // Read back
        let loaded = read_freq(&path).unwrap();

        assert_eq!(loaded.width, ingested.width);
        assert_eq!(loaded.height, ingested.height);
        assert_eq!(loaded.phash, ingested.phash);
        assert_eq!(loaded.coefficients.quality, ingested.coefficients.quality);
        assert_eq!(
            loaded.coefficients.entries.len(),
            ingested.coefficients.entries.len()
        );
    }

    #[test]
    fn test_progressive_loading() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("prog.freq");

        let pipeline = MediaIngestPipeline::new(MediaIngestConfig::default());

        let width = 32u32;
        let height = 32u32;
        let pixels: Vec<f64> = (0..width * height)
            .map(|i| (i as f64 / (width * height) as f64).sin())
            .collect();

        let ingested = pipeline.ingest_image(&pixels, width, height).unwrap();
        write_freq(&path, &ingested).unwrap();

        // Load only first 10 coefficients (thumbnail quality)
        let thumbnail = read_freq_progressive(&path, Some(10)).unwrap();
        assert!(thumbnail.coefficients.entries.len() <= 10);

        // Load all (full quality)
        let full = read_freq(&path).unwrap();
        assert!(full.coefficients.entries.len() >= thumbnail.coefficients.entries.len());
    }

    #[test]
    fn test_compression_ratio() {
        let pipeline = MediaIngestPipeline::new(MediaIngestConfig {
            quality: 50, // Lower quality = fewer coefficients = higher compression
            coefficient_threshold: 0.05,
            ..Default::default()
        });

        let width = 64u32;
        let height = 64u32;
        let pixels: Vec<f64> = (0..width * height)
            .map(|i| (i as f64 / 100.0).sin() * 0.5 + 0.5)
            .collect();

        let ingested = pipeline.ingest_image(&pixels, width, height).unwrap();
        let info = compression_info(&ingested);

        // Should achieve some compression (sparse frequency representation)
        assert!(
            info.compression_ratio > 1.0,
            "Expected compression, got ratio {}",
            info.compression_ratio
        );
        assert!(info.sparsity > 0.0, "Expected some sparsity");
    }
}
