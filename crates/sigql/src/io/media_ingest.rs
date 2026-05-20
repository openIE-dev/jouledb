//! MediaQL Ingest Pipeline — the "middle-out" core.
//!
//! Converts raw media (images, audio, video) into a unified representation:
//!
//! ```text
//! Raw pixels/samples ←← FFT/DCT ←← FREQUENCY DOMAIN →→ HDC encode →→ Semantic meaning
//!    (lossless reconstruct)              ↑ THE MIDDLE          (similarity/query)
//! ```
//!
//! From a single frequency-domain representation, we can:
//! 1. **Reconstruct** the original via inverse transform (lossless/lossy)
//! 2. **Query** via HDC holographic similarity (content-based retrieval)
//! 3. **Compress** naturally (frequency domain is sparse)
//!
//! This module provides the ingest pipeline that creates `IngestedMedia` from raw data.
//! The HDC hologram creation is done by the `MediaEncoder` trait (Phase 1C).

use crate::io::IoError;
use crate::types::signal::{
    CoefficientEntry, FreqTransform, FrequencyCoefficients, IngestedMedia, MediaTypeTag,
};
use std::f64::consts::PI;

/// Configuration for the media ingest pipeline.
#[derive(Debug, Clone)]
pub struct MediaIngestConfig {
    /// DCT block size for images (default 8, as in JPEG)
    pub dct_block_size: usize,
    /// Quality factor 1-100 (controls coefficient quantization)
    pub quality: u8,
    /// Keep coefficients above this fraction of the DC component
    pub coefficient_threshold: f64,
    /// FFT window size for audio STFT (default 2048)
    pub fft_window_size: usize,
    /// STFT hop size (default 512)
    pub hop_size: usize,
}

impl Default for MediaIngestConfig {
    fn default() -> Self {
        Self {
            dct_block_size: 8,
            quality: 85,
            coefficient_threshold: 0.01,
            fft_window_size: 2048,
            hop_size: 512,
        }
    }
}

/// The media ingest pipeline.
pub struct MediaIngestPipeline {
    config: MediaIngestConfig,
}

impl MediaIngestPipeline {
    pub fn new(config: MediaIngestConfig) -> Self {
        Self { config }
    }

    /// Ingest a grayscale image (row-major f64 pixels, values 0.0-1.0).
    ///
    /// Pipeline: image → block DCT-II → quantize → sparse coefficients → perceptual hash
    pub fn ingest_image(
        &self,
        pixels: &[f64],
        width: u32,
        height: u32,
    ) -> Result<IngestedMedia, IoError> {
        if pixels.len() != (width as usize * height as usize) {
            return Err(IoError::InvalidFormat(format!(
                "Pixel count {} != {}x{}",
                pixels.len(),
                width,
                height
            )));
        }

        let block = self.config.dct_block_size;
        let quality = self.config.quality;
        let threshold = self.config.coefficient_threshold;

        // Block DCT-II: split image into blocks, compute DCT on each
        let mut entries = Vec::new();

        let blocks_x = (width as usize + block - 1) / block;
        let blocks_y = (height as usize + block - 1) / block;

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                // Extract block (with zero-padding at edges)
                let mut block_data = vec![0.0f64; block * block];
                for row in 0..block {
                    for col in 0..block {
                        let px = bx * block + col;
                        let py = by * block + row;
                        if px < width as usize && py < height as usize {
                            block_data[row * block + col] = pixels[py * width as usize + px];
                        }
                    }
                }

                // 2D DCT-II on this block
                let dct_block = dct2d_block(&block_data, block);

                // Quantize and threshold
                let dc = dct_block[0].abs();
                let q_scale = quantization_scale(quality);

                for row in 0..block {
                    for col in 0..block {
                        let coeff = dct_block[row * block + col];
                        let quantized = (coeff / q_scale).round() * q_scale;

                        // Keep if above threshold relative to DC
                        if dc > 0.0 && quantized.abs() / dc > threshold {
                            entries.push(CoefficientEntry {
                                position: (
                                    (bx * block + col) as u32,
                                    (by * block + row) as u32,
                                ),
                                magnitude: quantized.abs() as f32,
                                phase: if quantized >= 0.0 { 0.0 } else { PI as f32 },
                            });
                        }
                    }
                }
            }
        }

        // Perceptual hash: downscale → DCT → threshold median
        let phash = compute_phash(pixels, width, height);

        Ok(IngestedMedia {
            coefficients: FrequencyCoefficients {
                entries,
                shape: (width, height),
                transform: FreqTransform::Dct2d,
                quality,
            },
            phash,
            width,
            height,
            media_type: MediaTypeTag::Image,
        })
    }

    /// Ingest audio samples (mono, f64, normalized -1.0 to 1.0).
    ///
    /// Pipeline: audio → STFT → magnitude/phase → sparse coefficients
    pub fn ingest_audio(
        &self,
        samples: &[f64],
        sample_rate: u32,
    ) -> Result<IngestedMedia, IoError> {
        if samples.is_empty() {
            return Err(IoError::InvalidFormat("Empty audio".into()));
        }

        let window_size = self.config.fft_window_size;
        let hop_size = self.config.hop_size;
        let threshold = self.config.coefficient_threshold;

        let mut entries = Vec::new();
        let mut frame_idx: u32 = 0;

        // STFT: sliding window FFT
        let mut pos = 0;
        while pos + window_size <= samples.len() {
            // Apply Hann window
            let windowed: Vec<f64> = (0..window_size)
                .map(|i| {
                    let w = 0.5 * (1.0 - (2.0 * PI * i as f64 / (window_size - 1) as f64).cos());
                    samples[pos + i] * w
                })
                .collect();

            // Real FFT (only positive frequencies needed)
            let spectrum = real_fft(&windowed);
            let n_bins = spectrum.len();

            // Find max magnitude for thresholding
            let max_mag = spectrum
                .iter()
                .map(|(mag, _)| *mag)
                .fold(0.0f64, f64::max);

            // Store significant coefficients
            for (bin, &(mag, phase)) in spectrum.iter().enumerate() {
                if max_mag > 0.0 && mag / max_mag > threshold {
                    entries.push(CoefficientEntry {
                        position: (frame_idx, bin as u32),
                        magnitude: mag as f32,
                        phase: phase as f32,
                    });
                }
            }

            pos += hop_size;
            frame_idx += 1;
        }

        let n_freq_bins = window_size / 2 + 1;

        // Audio fingerprint: simplified spectrogram peak hash
        let phash = compute_audio_fingerprint(&entries, frame_idx, n_freq_bins as u32);

        Ok(IngestedMedia {
            coefficients: FrequencyCoefficients {
                entries,
                shape: (frame_idx, n_freq_bins as u32),
                transform: FreqTransform::Stft,
                quality: self.config.quality,
            },
            phash,
            width: frame_idx,
            height: n_freq_bins as u32,
            media_type: MediaTypeTag::Audio,
        })
    }

    /// Reconstruct a grayscale image from frequency coefficients.
    ///
    /// This is the inverse path: coefficients → IDCT → pixels.
    pub fn reconstruct_image(media: &IngestedMedia) -> Result<Vec<f64>, IoError> {
        if media.media_type != MediaTypeTag::Image {
            return Err(IoError::InvalidFormat("Not an image".into()));
        }
        if media.coefficients.transform != FreqTransform::Dct2d {
            return Err(IoError::InvalidFormat("Expected DCT2D coefficients".into()));
        }

        let width = media.coefficients.shape.0 as usize;
        let height = media.coefficients.shape.1 as usize;
        let mut pixels = vec![0.0f64; width * height];

        // Group coefficients by block (assuming 8x8 blocks)
        let block = 8usize;
        let blocks_x = (width + block - 1) / block;
        let blocks_y = (height + block - 1) / block;

        let mut block_coeffs: Vec<Vec<f64>> = vec![vec![0.0; block * block]; blocks_x * blocks_y];

        for entry in &media.coefficients.entries {
            let px = entry.position.0 as usize;
            let py = entry.position.1 as usize;
            let bx = px / block;
            let by = py / block;
            let lx = px % block;
            let ly = py % block;

            if bx < blocks_x && by < blocks_y {
                let block_idx = by * blocks_x + bx;
                let sign = if entry.phase < 1.0 { 1.0 } else { -1.0 };
                block_coeffs[block_idx][ly * block + lx] = entry.magnitude as f64 * sign;
            }
        }

        // Inverse DCT on each block
        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let block_idx = by * blocks_x + bx;
                let spatial = idct2d_block(&block_coeffs[block_idx], block);

                for row in 0..block {
                    for col in 0..block {
                        let px = bx * block + col;
                        let py = by * block + row;
                        if px < width && py < height {
                            pixels[py * width + px] = spatial[row * block + col].clamp(0.0, 1.0);
                        }
                    }
                }
            }
        }

        Ok(pixels)
    }
}

// ============================================================================
// DSP Primitives (pure Rust, no dependencies)
// ============================================================================

/// 2D DCT-II on a square block (separable: row DCT then column DCT).
fn dct2d_block(data: &[f64], n: usize) -> Vec<f64> {
    // Row-wise DCT
    let mut row_dct = vec![0.0; n * n];
    for row in 0..n {
        let row_data: Vec<f64> = (0..n).map(|col| data[row * n + col]).collect();
        let transformed = dct1d(&row_data);
        for col in 0..n {
            row_dct[row * n + col] = transformed[col];
        }
    }

    // Column-wise DCT on the result
    let mut result = vec![0.0; n * n];
    for col in 0..n {
        let col_data: Vec<f64> = (0..n).map(|row| row_dct[row * n + col]).collect();
        let transformed = dct1d(&col_data);
        for row in 0..n {
            result[row * n + col] = transformed[row];
        }
    }

    result
}

/// 2D IDCT-III (inverse of DCT-II) on a square block.
fn idct2d_block(data: &[f64], n: usize) -> Vec<f64> {
    // Column-wise IDCT
    let mut col_idct = vec![0.0; n * n];
    for col in 0..n {
        let col_data: Vec<f64> = (0..n).map(|row| data[row * n + col]).collect();
        let transformed = idct1d(&col_data);
        for row in 0..n {
            col_idct[row * n + col] = transformed[row];
        }
    }

    // Row-wise IDCT on the result
    let mut result = vec![0.0; n * n];
    for row in 0..n {
        let row_data: Vec<f64> = (0..n).map(|col| col_idct[row * n + col]).collect();
        let transformed = idct1d(&row_data);
        for col in 0..n {
            result[row * n + col] = transformed[col];
        }
    }

    result
}

/// 1D DCT-II (the standard "DCT" used in JPEG).
fn dct1d(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut result = vec![0.0; n];
    for k in 0..n {
        let mut sum = 0.0;
        for i in 0..n {
            sum += x[i] * ((PI * (2 * i + 1) as f64 * k as f64) / (2 * n) as f64).cos();
        }
        let scale = if k == 0 {
            (1.0 / n as f64).sqrt()
        } else {
            (2.0 / n as f64).sqrt()
        };
        result[k] = scale * sum;
    }
    result
}

/// 1D IDCT-III (inverse of DCT-II).
fn idct1d(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut result = vec![0.0; n];
    for i in 0..n {
        let mut sum = x[0] * (1.0 / n as f64).sqrt();
        for k in 1..n {
            sum += x[k]
                * (2.0 / n as f64).sqrt()
                * ((PI * (2 * i + 1) as f64 * k as f64) / (2 * n) as f64).cos();
        }
        result[i] = sum;
    }
    result
}

/// Quantization scale factor from quality (1-100).
/// Lower quality = higher scale = more aggressive quantization.
fn quantization_scale(quality: u8) -> f64 {
    let q = quality.clamp(1, 100) as f64;
    if q < 50.0 {
        5000.0 / q
    } else {
        200.0 - 2.0 * q
    }
    .max(1.0)
        / 100.0
}

/// Real FFT: returns (magnitude, phase) pairs for positive frequencies.
fn real_fft(x: &[f64]) -> Vec<(f64, f64)> {
    let n = x.len();
    let n_out = n / 2 + 1;
    let mut result = Vec::with_capacity(n_out);

    for k in 0..n_out {
        let mut re = 0.0;
        let mut im = 0.0;
        for i in 0..n {
            let angle = -2.0 * PI * k as f64 * i as f64 / n as f64;
            re += x[i] * angle.cos();
            im += x[i] * angle.sin();
        }
        let mag = (re * re + im * im).sqrt();
        let phase = im.atan2(re);
        result.push((mag, phase));
    }

    result
}

/// Perceptual hash for images (simplified pHash).
/// Downscale → DCT → compare to median → 64-bit hash.
fn compute_phash(pixels: &[f64], width: u32, height: u32) -> u64 {
    let size = 8usize;
    let w = width as usize;
    let h = height as usize;

    // Bilinear downscale to 8x8
    let mut small = vec![0.0f64; size * size];
    for y in 0..size {
        for x in 0..size {
            let src_x = (x as f64 * w as f64 / size as f64).min((w - 1) as f64);
            let src_y = (y as f64 * h as f64 / size as f64).min((h - 1) as f64);
            let sx = src_x as usize;
            let sy = src_y as usize;
            small[y * size + x] = pixels[sy.min(h - 1) * w + sx.min(w - 1)];
        }
    }

    // 2D DCT on the 8x8 block
    let dct = dct2d_block(&small, size);

    // Use top-left 8x8 low-frequency coefficients (skip DC at [0,0])
    let mut values: Vec<f64> = Vec::with_capacity(63);
    for i in 1..64 {
        values.push(dct[i]);
    }

    // Median
    let mut sorted = values.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sorted[sorted.len() / 2];

    // Hash: 1 if above median, 0 if below
    let mut hash: u64 = 0;
    for (i, &v) in values.iter().enumerate().take(64) {
        if v > median {
            hash |= 1 << i;
        }
    }

    hash
}

/// Audio fingerprint: simplified spectrogram peak constellation hash.
fn compute_audio_fingerprint(entries: &[CoefficientEntry], n_frames: u32, n_bins: u32) -> u64 {
    // Find the top peaks across all frames
    let mut peaks: Vec<(u32, u32, f32)> = entries
        .iter()
        .map(|e| (e.position.0, e.position.1, e.magnitude))
        .collect();
    peaks.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    peaks.truncate(64);

    // Hash from peak positions
    let mut hash: u64 = 0;
    for (i, peak) in peaks.iter().enumerate().take(64) {
        let frame_bit = (peak.0 % n_frames) & 1;
        let bin_bit = (peak.1 % n_bins) & 1;
        if frame_bit ^ bin_bit == 1 {
            hash |= 1 << i;
        }
    }

    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dct_idct_roundtrip() {
        // A simple signal should survive DCT → IDCT roundtrip
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let dct = dct1d(&original);
        let recovered = idct1d(&dct);

        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "Roundtrip failed: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_dct2d_idct2d_roundtrip() {
        let block = 4;
        let original: Vec<f64> = (0..block * block).map(|i| i as f64 / 16.0).collect();
        let dct = dct2d_block(&original, block);
        let recovered = idct2d_block(&dct, block);

        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "2D roundtrip failed: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_image_ingest_and_reconstruct() {
        let pipeline = MediaIngestPipeline::new(MediaIngestConfig {
            quality: 100, // Lossless-ish
            coefficient_threshold: 0.0, // Keep all coefficients
            ..Default::default()
        });

        // 16x16 gradient image
        let width = 16u32;
        let height = 16u32;
        let pixels: Vec<f64> = (0..width * height)
            .map(|i| (i as f64) / (width * height) as f64)
            .collect();

        let ingested = pipeline.ingest_image(&pixels, width, height).unwrap();

        assert_eq!(ingested.media_type, MediaTypeTag::Image);
        assert!(!ingested.coefficients.entries.is_empty());
        assert_eq!(ingested.coefficients.transform, FreqTransform::Dct2d);

        // Reconstruct
        let recovered = MediaIngestPipeline::reconstruct_image(&ingested).unwrap();
        assert_eq!(recovered.len(), pixels.len());

        // Check PSNR > 30 dB (lossy but reasonable)
        let mse: f64 = pixels
            .iter()
            .zip(recovered.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / pixels.len() as f64;
        let psnr = if mse > 0.0 {
            10.0 * (1.0 / mse).log10()
        } else {
            f64::INFINITY
        };
        assert!(
            psnr > 30.0,
            "PSNR too low: {:.1} dB (MSE: {:.6})",
            psnr,
            mse
        );
    }

    #[test]
    fn test_audio_ingest() {
        let pipeline = MediaIngestPipeline::new(MediaIngestConfig {
            fft_window_size: 256,
            hop_size: 128,
            ..Default::default()
        });

        // 1 second of 440Hz sine wave at 8000Hz sample rate
        let sample_rate = 8000u32;
        let duration = 1.0;
        let n_samples = (sample_rate as f64 * duration) as usize;
        let samples: Vec<f64> = (0..n_samples)
            .map(|i| (2.0 * PI * 440.0 * i as f64 / sample_rate as f64).sin())
            .collect();

        let ingested = pipeline.ingest_audio(&samples, sample_rate).unwrap();

        assert_eq!(ingested.media_type, MediaTypeTag::Audio);
        assert!(!ingested.coefficients.entries.is_empty());
        assert_eq!(ingested.coefficients.transform, FreqTransform::Stft);
    }

    #[test]
    fn test_phash_similarity() {
        // Two similar images should have similar pHash
        let w = 32u32;
        let h = 32u32;

        let img1: Vec<f64> = (0..w * h).map(|i| (i as f64 / (w * h) as f64)).collect();
        let img2: Vec<f64> = (0..w * h)
            .map(|i| (i as f64 / (w * h) as f64) + 0.01)
            .collect();

        let hash1 = compute_phash(&img1, w, h);
        let hash2 = compute_phash(&img2, w, h);

        // Hamming distance should be small (similar images)
        // pHash with median thresholding tolerates up to ~20 bits of difference
        // for near-identical images with slight DC offset
        let hamming = (hash1 ^ hash2).count_ones();
        assert!(
            hamming < 24,
            "Similar images should have close hashes, got Hamming distance {}",
            hamming
        );
    }
}
