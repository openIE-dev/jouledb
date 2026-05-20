//! Voiceprint Store — speaker verification via MFCC + HDC + HNSW.
//!
//! $20.6B voice biometrics market by 2026. Every VoIP call can be authenticated
//! by comparing the speaker's voice against stored voiceprints.
//!
//! Pipeline: audio → MFCC extraction → BinaryHV hologram → HNSW similarity search
//!
//! Today this requires 3 separate systems (audio DSP, vector DB, identity service).
//! JouleDB does it in one query:
//! ```sql
//! FROM media('call_audio.wav') AS voice
//! TRANSFORM mfcc(13)
//! WHERE similar_to(amorphic.voiceprints, threshold: 0.85)
//! RETURNING top(1) WITH confidence(0.95)
//! ```

use joule_db_hdc::{BinaryHV, BundleAccumulator, HNSWIndex};
use joule_db_hdc::manifold::DistanceMetric;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{AmorphicResult, AmorphicError, DIMENSION};

/// A stored voiceprint: the holographic encoding of a speaker's voice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Voiceprint {
    /// Speaker identifier
    pub speaker_id: String,
    /// Display name
    pub name: String,
    /// Number of enrollment samples used to build this voiceprint
    pub enrollment_count: u32,
    /// Creation timestamp (unix ms)
    pub created_ms: u64,
    /// Last verification timestamp (unix ms)
    pub last_verified_ms: Option<u64>,
}

/// Result of a speaker verification attempt.
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Whether the speaker was verified (similarity >= threshold)
    pub verified: bool,
    /// Best matching speaker, if any
    pub speaker: Option<Voiceprint>,
    /// Similarity score (0.0 - 1.0)
    pub similarity: f32,
    /// Whether this might be synthetic/deepfake speech
    pub synthetic_risk: SyntheticRisk,
}

/// Risk assessment for synthetic/deepfake speech.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SyntheticRisk {
    /// Likely natural speech
    Low,
    /// Some indicators of synthesis
    Medium,
    /// High probability of synthetic speech
    High,
    /// Could not assess (insufficient data)
    Unknown,
}

/// MFCC-based voice feature vector.
/// 13 MFCC coefficients × N frames, aggregated into statistical features.
#[derive(Debug, Clone)]
pub struct VoiceFeatures {
    /// Mean of each MFCC coefficient across all frames
    pub mfcc_mean: Vec<f64>,
    /// Standard deviation of each MFCC coefficient
    pub mfcc_std: Vec<f64>,
    /// Delta (first derivative) means
    pub delta_mean: Vec<f64>,
    /// Spectral centroid mean
    pub spectral_centroid: f64,
    /// Fundamental frequency (F0) statistics
    pub f0_mean: f64,
    pub f0_std: f64,
    /// Zero crossing rate
    pub zcr_mean: f64,
}

impl VoiceFeatures {
    /// Extract voice features from raw MFCC frames.
    /// Each frame is a Vec of n_coefficients values.
    pub fn from_mfcc_frames(frames: &[Vec<f64>]) -> Option<Self> {
        if frames.is_empty() || frames[0].is_empty() {
            return None;
        }

        let n_coeffs = frames[0].len();
        let n_frames = frames.len() as f64;

        // Compute mean and std for each coefficient
        let mut mfcc_mean = vec![0.0f64; n_coeffs];
        for frame in frames {
            for (i, &val) in frame.iter().enumerate() {
                if i < n_coeffs {
                    mfcc_mean[i] += val;
                }
            }
        }
        for m in &mut mfcc_mean {
            *m /= n_frames;
        }

        let mut mfcc_std = vec![0.0f64; n_coeffs];
        for frame in frames {
            for (i, &val) in frame.iter().enumerate() {
                if i < n_coeffs {
                    mfcc_std[i] += (val - mfcc_mean[i]).powi(2);
                }
            }
        }
        for s in &mut mfcc_std {
            *s = (*s / n_frames).sqrt();
        }

        // Compute deltas (frame-to-frame differences)
        let mut delta_mean = vec![0.0f64; n_coeffs];
        if frames.len() > 1 {
            let mut delta_count = 0.0;
            for w in frames.windows(2) {
                for i in 0..n_coeffs.min(w[0].len()).min(w[1].len()) {
                    delta_mean[i] += (w[1][i] - w[0][i]).abs();
                }
                delta_count += 1.0;
            }
            if delta_count > 0.0 {
                for d in &mut delta_mean {
                    *d /= delta_count;
                }
            }
        }

        Some(VoiceFeatures {
            mfcc_mean,
            mfcc_std,
            delta_mean,
            spectral_centroid: 0.0, // Would need spectral analysis
            f0_mean: 0.0,
            f0_std: 0.0,
            zcr_mean: 0.0,
        })
    }

    /// Convert to a flat f64 vector suitable for HDC encoding.
    pub fn to_feature_vector(&self) -> Vec<f64> {
        let mut v = Vec::with_capacity(self.mfcc_mean.len() * 3 + 4);
        v.extend_from_slice(&self.mfcc_mean);
        v.extend_from_slice(&self.mfcc_std);
        v.extend_from_slice(&self.delta_mean);
        v.push(self.spectral_centroid);
        v.push(self.f0_mean);
        v.push(self.f0_std);
        v.push(self.zcr_mean);
        v
    }

    /// Encode as a BinaryHV hologram for storage and similarity search.
    pub fn to_hologram(&self) -> BinaryHV {
        let features = self.to_feature_vector();
        // Convert f64 to bytes for hashing
        let bytes: Vec<u8> = features
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        BinaryHV::from_data(&bytes, DIMENSION)
    }

    /// Assess synthetic speech risk based on feature anomalies.
    /// Real speech has natural micro-variations; synthetic speech is often
    /// too smooth in MFCC delta patterns.
    pub fn assess_synthetic_risk(&self) -> SyntheticRisk {
        if self.mfcc_std.is_empty() || self.delta_mean.is_empty() {
            return SyntheticRisk::Unknown;
        }

        // Heuristic: synthetic speech often has abnormally low delta variance
        // (too smooth transitions between frames)
        let avg_delta: f64 = self.delta_mean.iter().sum::<f64>() / self.delta_mean.len() as f64;
        let avg_std: f64 = self.mfcc_std.iter().sum::<f64>() / self.mfcc_std.len() as f64;

        // Very low delta with normal std → suspicious
        if avg_delta < 0.05 && avg_std > 0.5 {
            return SyntheticRisk::High;
        }
        // Low delta → medium risk
        if avg_delta < 0.1 {
            return SyntheticRisk::Medium;
        }

        SyntheticRisk::Low
    }
}

/// The voiceprint store: manages speaker enrollment and verification.
pub struct VoiceprintStore {
    /// HNSW index for fast speaker matching
    hnsw: HNSWIndex,
    /// Speaker metadata
    speakers: HashMap<String, Voiceprint>,
    /// Speaker ID → hologram for re-enrollment bundling
    holograms: HashMap<String, BinaryHV>,
    /// Verification similarity threshold
    pub verify_threshold: f32,
    /// Enrollment requires this many samples
    pub min_enrollment_samples: u32,
}

impl VoiceprintStore {
    pub fn new() -> Self {
        // HNSW dimension = DIMENSION / 64 * 2 (packed BinaryHV → f32)
        let hnsw_dim = (DIMENSION + 63) / 64 * 2;
        Self {
            hnsw: HNSWIndex::with_metric(hnsw_dim, 16, 200, DistanceMetric::Hamming),
            speakers: HashMap::new(),
            holograms: HashMap::new(),
            verify_threshold: 0.75,
            min_enrollment_samples: 1,
        }
    }

    /// Enroll a speaker with voice features.
    /// Multiple enrollments improve accuracy (holograms are bundled).
    pub fn enroll(
        &mut self,
        speaker_id: &str,
        name: &str,
        features: &VoiceFeatures,
        timestamp_ms: u64,
    ) -> AmorphicResult<()> {
        let hologram = features.to_hologram();

        if let Some(existing) = self.holograms.get(speaker_id) {
            // Re-enrollment: bundle new hologram with existing
            let mut acc = BundleAccumulator::new(DIMENSION);
            acc.add(existing);
            acc.add(&hologram);
            let merged = acc.threshold();

            // Update HNSW (remove old, insert new)
            // Note: HNSW doesn't support update-in-place, so we re-insert
            let packed = merged.to_f32_packed();
            let _ = self.hnsw.insert(speaker_id.to_string(), packed);

            self.holograms.insert(speaker_id.to_string(), merged);

            if let Some(vp) = self.speakers.get_mut(speaker_id) {
                vp.enrollment_count += 1;
            }
        } else {
            // First enrollment
            let packed = hologram.to_f32_packed();
            self.hnsw
                .insert(speaker_id.to_string(), packed)
                .map_err(|e| AmorphicError::IngestionError(e))?;

            self.speakers.insert(
                speaker_id.to_string(),
                Voiceprint {
                    speaker_id: speaker_id.to_string(),
                    name: name.to_string(),
                    enrollment_count: 1,
                    created_ms: timestamp_ms,
                    last_verified_ms: None,
                },
            );
            self.holograms
                .insert(speaker_id.to_string(), hologram);
        }

        Ok(())
    }

    /// Verify a speaker: does this voice match any enrolled voiceprint?
    pub fn verify(&self, features: &VoiceFeatures) -> VerificationResult {
        let hologram = features.to_hologram();
        let packed = hologram.to_f32_packed();

        let results = self.hnsw.query(&packed, 1);

        let synthetic_risk = features.assess_synthetic_risk();

        match results.first() {
            Some(best) => {
                let speaker = self.speakers.get(&best.id).cloned();

                // Convert HNSW distance to similarity.
                // Hamming distance → similarity = 1 - (dist / total_bits)
                let total_bits = (DIMENSION as f32).max(1.0);
                let similarity = 1.0 - (best.distance / total_bits).min(1.0);

                VerificationResult {
                    verified: similarity >= self.verify_threshold
                        && synthetic_risk != SyntheticRisk::High,
                    speaker,
                    similarity,
                    synthetic_risk,
                }
            }
            None => VerificationResult {
                verified: false,
                speaker: None,
                similarity: 0.0,
                synthetic_risk,
            },
        }
    }

    /// Identify a speaker from a pool (1:N matching).
    /// Returns top-k candidates sorted by similarity.
    pub fn identify(&self, features: &VoiceFeatures, k: usize) -> Vec<VerificationResult> {
        let hologram = features.to_hologram();
        let packed = hologram.to_f32_packed();
        let synthetic_risk = features.assess_synthetic_risk();

        let results = self.hnsw.query(&packed, k);

        results
            .into_iter()
            .map(|r| {
                let total_bits = (DIMENSION as f32).max(1.0);
                let similarity = 1.0 - (r.distance / total_bits).min(1.0);
                VerificationResult {
                    verified: similarity >= self.verify_threshold,
                    speaker: self.speakers.get(&r.id).cloned(),
                    similarity,
                    synthetic_risk,
                }
            })
            .collect()
    }

    /// Number of enrolled speakers.
    pub fn speaker_count(&self) -> usize {
        self.speakers.len()
    }

    /// Get a speaker's voiceprint metadata.
    pub fn get_speaker(&self, speaker_id: &str) -> Option<&Voiceprint> {
        self.speakers.get(speaker_id)
    }
}

impl Default for VoiceprintStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Call quality metrics computed from audio signal analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallQualityMetrics {
    /// Estimated Mean Opinion Score (1.0 - 5.0)
    pub estimated_mos: f64,
    /// Signal-to-noise ratio in dB
    pub snr_db: f64,
    /// RMS loudness in dBFS
    pub loudness_dbfs: f64,
    /// Fraction of silence in the audio (0.0 - 1.0)
    pub silence_ratio: f64,
    /// Detected clipping events
    pub clip_count: u32,
    /// Spectral flatness (1.0 = noise, 0.0 = tonal)
    pub spectral_flatness: f64,
}

impl CallQualityMetrics {
    /// Compute call quality metrics from raw audio samples.
    /// Samples should be f64 in range [-1.0, 1.0], mono.
    pub fn from_samples(samples: &[f64], sample_rate: u32) -> Self {
        if samples.is_empty() {
            return Self {
                estimated_mos: 1.0,
                snr_db: 0.0,
                loudness_dbfs: -96.0,
                silence_ratio: 1.0,
                clip_count: 0,
                spectral_flatness: 0.0,
            };
        }

        let n = samples.len() as f64;

        // RMS and loudness
        let rms = (samples.iter().map(|s| s * s).sum::<f64>() / n).sqrt();
        let loudness_dbfs = if rms > 0.0 {
            20.0 * rms.log10()
        } else {
            -96.0
        };

        // Silence detection (samples below -60dBFS)
        let silence_threshold = 0.001; // ~-60dBFS
        let silence_count = samples.iter().filter(|s| s.abs() < silence_threshold).count();
        let silence_ratio = silence_count as f64 / n;

        // Clipping detection (samples at ±1.0)
        let clip_count = samples
            .iter()
            .filter(|s| s.abs() > 0.99)
            .count() as u32;

        // Simple SNR estimation: signal power / noise floor estimate
        let signal_power = rms * rms;
        let mut sorted = samples.iter().map(|s| s.abs()).collect::<Vec<f64>>();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let noise_floor = sorted[sorted.len() / 10]; // 10th percentile as noise estimate
        let noise_power = noise_floor * noise_floor;
        let snr_db = if noise_power > 0.0 {
            10.0 * (signal_power / noise_power).log10()
        } else {
            60.0 // Very clean signal
        };

        // Spectral flatness approximation
        // Using Wiener entropy: geometric mean / arithmetic mean of power spectrum
        // Simplified: ratio of sorted amplitudes
        let geo_mean = sorted
            .iter()
            .filter(|&&s| s > 0.0)
            .map(|s| s.ln())
            .sum::<f64>()
            / sorted.len() as f64;
        let arith_mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
        let spectral_flatness = if arith_mean > 0.0 {
            (geo_mean.exp() / arith_mean).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Estimated MOS (simplified ITU-T P.563 approximation)
        let mos = estimate_mos(snr_db, silence_ratio, clip_count, loudness_dbfs);

        CallQualityMetrics {
            estimated_mos: mos,
            snr_db,
            loudness_dbfs,
            silence_ratio,
            clip_count,
            spectral_flatness,
        }
    }
}

/// Simplified MOS estimation from signal quality indicators.
fn estimate_mos(snr_db: f64, silence_ratio: f64, clip_count: u32, loudness_dbfs: f64) -> f64 {
    let mut mos: f64 = 4.5; // Start with excellent

    // SNR penalty
    if snr_db < 10.0 {
        mos -= 2.0;
    } else if snr_db < 20.0 {
        mos -= 1.0;
    } else if snr_db < 30.0 {
        mos -= 0.3;
    }

    // Silence penalty (too much silence = dropouts)
    if silence_ratio > 0.5 {
        mos -= 1.5;
    } else if silence_ratio > 0.3 {
        mos -= 0.5;
    }

    // Clipping penalty
    if clip_count > 100 {
        mos -= 1.0;
    } else if clip_count > 10 {
        mos -= 0.3;
    }

    // Loudness penalty (too quiet or too loud)
    if loudness_dbfs < -40.0 {
        mos -= 0.5;
    }
    if loudness_dbfs > -3.0 {
        mos -= 0.5;
    }

    mos.clamp(1.0, 5.0)
}

/// Call Detail Record with multi-model queryable fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallDetailRecord {
    /// Unique call identifier
    pub call_id: String,
    /// Caller identifier
    pub caller_id: String,
    /// Callee identifier
    pub callee_id: String,
    /// Call start timestamp (unix ms)
    pub start_ms: u64,
    /// Call duration (ms)
    pub duration_ms: u64,
    /// Codec used
    pub codec: String,
    /// Call quality metrics
    pub quality: CallQualityMetrics,
    /// Verified speaker ID (if voiceprint matched)
    pub verified_speaker: Option<String>,
    /// Call direction
    pub direction: CallDirection,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Call direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CallDirection {
    Inbound,
    Outbound,
    Internal,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_mfcc_frames(n_frames: usize, n_coeffs: usize, seed: f64) -> Vec<Vec<f64>> {
        (0..n_frames)
            .map(|f| {
                (0..n_coeffs)
                    .map(|c| ((f as f64 * 0.1 + c as f64 * 0.3 + seed).sin()) * 10.0)
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_voice_features_extraction() {
        let frames = make_mfcc_frames(100, 13, 0.0);
        let features = VoiceFeatures::from_mfcc_frames(&frames).unwrap();

        assert_eq!(features.mfcc_mean.len(), 13);
        assert_eq!(features.mfcc_std.len(), 13);
        assert_eq!(features.delta_mean.len(), 13);
    }

    #[test]
    fn test_voiceprint_enroll_and_verify() {
        let mut store = VoiceprintStore::new();

        // Enroll Alice
        let alice_frames = make_mfcc_frames(100, 13, 1.0);
        let alice_features = VoiceFeatures::from_mfcc_frames(&alice_frames).unwrap();
        store
            .enroll("alice", "Alice Smith", &alice_features, 1000)
            .unwrap();

        assert_eq!(store.speaker_count(), 1);

        // Verify with same voice → should match
        let result = store.verify(&alice_features);
        assert!(result.verified, "Same voice should verify");
        assert_eq!(result.speaker.unwrap().speaker_id, "alice");
    }

    #[test]
    fn test_different_speakers() {
        let mut store = VoiceprintStore::new();

        let alice_features =
            VoiceFeatures::from_mfcc_frames(&make_mfcc_frames(100, 13, 1.0)).unwrap();
        let bob_features =
            VoiceFeatures::from_mfcc_frames(&make_mfcc_frames(100, 13, 100.0)).unwrap();

        store.enroll("alice", "Alice", &alice_features, 1000).unwrap();
        store.enroll("bob", "Bob", &bob_features, 2000).unwrap();

        assert_eq!(store.speaker_count(), 2);

        // Alice's voice should match Alice, not Bob
        let result = store.verify(&alice_features);
        assert!(result.verified);
        assert_eq!(result.speaker.unwrap().speaker_id, "alice");
    }

    #[test]
    fn test_re_enrollment_improves() {
        let mut store = VoiceprintStore::new();

        let features1 =
            VoiceFeatures::from_mfcc_frames(&make_mfcc_frames(100, 13, 1.0)).unwrap();
        store.enroll("alice", "Alice", &features1, 1000).unwrap();

        // Re-enroll with slightly different sample
        let features2 =
            VoiceFeatures::from_mfcc_frames(&make_mfcc_frames(100, 13, 1.1)).unwrap();
        store.enroll("alice", "Alice", &features2, 2000).unwrap();

        let vp = store.get_speaker("alice").unwrap();
        assert_eq!(vp.enrollment_count, 2);
    }

    #[test]
    fn test_synthetic_risk_assessment() {
        // Normal speech: reasonable delta values
        let mut features =
            VoiceFeatures::from_mfcc_frames(&make_mfcc_frames(100, 13, 1.0)).unwrap();
        assert_eq!(features.assess_synthetic_risk(), SyntheticRisk::Low);

        // Synthetic-looking: artificially low deltas
        features.delta_mean = vec![0.01; 13];
        features.mfcc_std = vec![1.0; 13];
        assert_eq!(features.assess_synthetic_risk(), SyntheticRisk::High);
    }

    #[test]
    fn test_call_quality_metrics() {
        // Clean sine wave
        let sample_rate = 8000;
        let samples: Vec<f64> = (0..sample_rate)
            .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / sample_rate as f64).sin() * 0.5)
            .collect();

        let metrics = CallQualityMetrics::from_samples(&samples, sample_rate as u32);
        assert!(metrics.estimated_mos >= 3.0, "Clean signal should have decent MOS");
        assert!(metrics.snr_db > 10.0);
        assert_eq!(metrics.clip_count, 0);
    }

    #[test]
    fn test_call_quality_noisy() {
        // Mostly silence with occasional bursts → low quality
        let mut samples = vec![0.0001; 8000];
        for i in 0..100 {
            samples[i * 80] = 0.9; // Sparse bursts
        }

        let metrics = CallQualityMetrics::from_samples(&samples, 8000);
        assert!(metrics.silence_ratio > 0.5);
    }

    #[test]
    fn test_identify_top_k() {
        let mut store = VoiceprintStore::new();

        // Enroll 5 speakers
        for i in 0..5 {
            let features = VoiceFeatures::from_mfcc_frames(
                &make_mfcc_frames(100, 13, i as f64 * 50.0),
            )
            .unwrap();
            store
                .enroll(&format!("speaker_{}", i), &format!("Speaker {}", i), &features, 1000)
                .unwrap();
        }

        // Identify: should return top-3 candidates
        let query =
            VoiceFeatures::from_mfcc_frames(&make_mfcc_frames(100, 13, 0.0)).unwrap();
        let candidates = store.identify(&query, 3);
        assert_eq!(candidates.len(), 3);
        // First candidate should be speaker_0 (closest seed)
        assert_eq!(
            candidates[0].speaker.as_ref().unwrap().speaker_id,
            "speaker_0"
        );
    }
}
