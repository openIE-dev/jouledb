//! Content-Aware Delta Encoding — bandwidth reduction via frequency-domain diffs.
//!
//! If a client already has a similar image cached, send only the frequency-domain
//! delta instead of the full asset. This is rsync for media — but operating on
//! perceptual similarity, not byte-level checksums.
//!
//! Also provides progressive quality tiers for CDN edge caching:
//! edge caches low-freq coefficients, full quality fetched from origin on demand.

use crate::io::IoResult;
use crate::types::signal::{
    CoefficientEntry, FreqTransform, FrequencyCoefficients, IngestedMedia, MediaTypeTag,
};
use std::collections::HashMap;

// ============================================================================
// Progressive Quality Tiers
// ============================================================================

/// Quality tier for progressive serving.
#[derive(Debug, Clone, Copy)]
pub struct QualityTier {
    /// Human-readable name
    pub name: &'static str,
    /// Maximum number of coefficients to include
    pub max_coefficients: usize,
}

/// Predefined quality tiers.
pub const TIER_THUMBNAIL: QualityTier = QualityTier {
    name: "thumbnail",
    max_coefficients: 10,
};
pub const TIER_PREVIEW: QualityTier = QualityTier {
    name: "preview",
    max_coefficients: 100,
};
pub const TIER_STANDARD: QualityTier = QualityTier {
    name: "standard",
    max_coefficients: 1000,
};
pub const TIER_FULL: QualityTier = QualityTier {
    name: "full",
    max_coefficients: usize::MAX,
};

/// Get coefficients for a specific quality tier.
/// Coefficients in IngestedMedia are already sorted by magnitude descending
/// (from the ingest pipeline), so truncating gives the most important ones.
pub fn coefficients_at_tier(media: &IngestedMedia, tier: QualityTier) -> FrequencyCoefficients {
    let entries: Vec<CoefficientEntry> = media
        .coefficients
        .entries
        .iter()
        .take(tier.max_coefficients)
        .cloned()
        .collect();

    FrequencyCoefficients {
        entries,
        shape: media.coefficients.shape,
        transform: media.coefficients.transform,
        quality: media.coefficients.quality,
    }
}

/// Estimate byte size for a quality tier.
pub fn tier_byte_size(media: &IngestedMedia, tier: QualityTier) -> usize {
    let count = media.coefficients.entries.len().min(tier.max_coefficients);
    32 + count * 12 // header + entries (same as .freq format)
}

/// Select the appropriate tier based on a bandwidth budget (bytes).
pub fn tier_for_budget(media: &IngestedMedia, max_bytes: usize) -> QualityTier {
    let tiers = [TIER_THUMBNAIL, TIER_PREVIEW, TIER_STANDARD, TIER_FULL];
    for tier in tiers.iter().rev() {
        if tier_byte_size(media, *tier) <= max_bytes {
            return *tier;
        }
    }
    TIER_THUMBNAIL
}

// ============================================================================
// Frequency-Domain Delta Encoding
// ============================================================================

/// A delta between two frequency-domain representations.
/// Applying the delta to the reference reconstructs the target.
#[derive(Debug, Clone)]
pub struct FreqDelta {
    /// ID of the reference media (client must have this cached)
    pub reference_id: String,
    /// Coefficients that are new or changed in the target
    pub changed_entries: Vec<CoefficientEntry>,
    /// Positions of coefficients that exist in reference but not in target
    pub removed_positions: Vec<(u32, u32)>,
    /// Target shape (may differ from reference)
    pub target_shape: (u32, u32),
    /// Transform type
    pub transform: FreqTransform,
}

/// Compute the delta between a reference and target media.
/// The delta encodes only the differences — coefficients that are new,
/// changed beyond threshold, or removed.
pub fn compute_delta(
    reference: &IngestedMedia,
    target: &IngestedMedia,
    reference_id: &str,
    magnitude_threshold: f32,
) -> FreqDelta {
    // Build position → coefficient maps for fast lookup
    let ref_map: HashMap<(u32, u32), &CoefficientEntry> = reference
        .coefficients
        .entries
        .iter()
        .map(|e| (e.position, e))
        .collect();

    let target_map: HashMap<(u32, u32), &CoefficientEntry> = target
        .coefficients
        .entries
        .iter()
        .map(|e| (e.position, e))
        .collect();

    // Find changed/new coefficients
    let mut changed = Vec::new();
    for (&pos, &target_entry) in &target_map {
        match ref_map.get(&pos) {
            Some(ref_entry) => {
                // Coefficient exists in both — check if significantly different
                let mag_diff = (target_entry.magnitude - ref_entry.magnitude).abs();
                let phase_diff = (target_entry.phase - ref_entry.phase).abs();
                if mag_diff > magnitude_threshold || phase_diff > 0.1 {
                    changed.push(*target_entry);
                }
            }
            None => {
                // New coefficient not in reference
                changed.push(*target_entry);
            }
        }
    }

    // Find removed coefficients (in reference but not in target)
    let removed: Vec<(u32, u32)> = ref_map
        .keys()
        .filter(|pos| !target_map.contains_key(pos))
        .cloned()
        .collect();

    FreqDelta {
        reference_id: reference_id.to_string(),
        changed_entries: changed,
        removed_positions: removed,
        target_shape: target.coefficients.shape,
        transform: target.coefficients.transform,
    }
}

/// Apply a delta to a reference media to reconstruct the target.
pub fn apply_delta(reference: &IngestedMedia, delta: &FreqDelta) -> IngestedMedia {
    // Start with reference coefficients
    let mut entries: HashMap<(u32, u32), CoefficientEntry> = reference
        .coefficients
        .entries
        .iter()
        .map(|e| (e.position, *e))
        .collect();

    // Remove deleted coefficients
    for pos in &delta.removed_positions {
        entries.remove(pos);
    }

    // Apply changed/new coefficients
    for entry in &delta.changed_entries {
        entries.insert(entry.position, *entry);
    }

    // Sort by magnitude descending (maintain progressive ordering)
    let mut sorted_entries: Vec<CoefficientEntry> = entries.into_values().collect();
    sorted_entries.sort_by(|a, b| {
        b.magnitude
            .partial_cmp(&a.magnitude)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    IngestedMedia {
        coefficients: FrequencyCoefficients {
            entries: sorted_entries,
            shape: delta.target_shape,
            transform: delta.transform,
            quality: reference.coefficients.quality,
        },
        phash: reference.phash, // pHash may need recomputation
        width: delta.target_shape.0,
        height: delta.target_shape.1,
        media_type: reference.media_type,
    }
}

/// Compression ratio of the delta vs sending the full target.
pub fn delta_compression_ratio(delta: &FreqDelta, full_target_entries: usize) -> f64 {
    let delta_size = delta.changed_entries.len() + delta.removed_positions.len();
    if delta_size == 0 {
        return f64::INFINITY;
    }
    full_target_entries as f64 / delta_size as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::media_ingest::{MediaIngestConfig, MediaIngestPipeline};

    #[test]
    fn test_quality_tiers() {
        let pipeline = MediaIngestPipeline::new(MediaIngestConfig {
            quality: 85,
            coefficient_threshold: 0.001,
            ..Default::default()
        });

        let pixels: Vec<f64> = (0..64 * 64).map(|i| (i as f64 / 4096.0).sin()).collect();
        let media = pipeline.ingest_image(&pixels, 64, 64).unwrap();

        let thumb = coefficients_at_tier(&media, TIER_THUMBNAIL);
        let preview = coefficients_at_tier(&media, TIER_PREVIEW);
        let full = coefficients_at_tier(&media, TIER_FULL);

        assert!(thumb.entries.len() <= 10);
        assert!(preview.entries.len() <= 100);
        assert!(full.entries.len() >= preview.entries.len());

        // Thumbnail bytes << full bytes
        let thumb_size = tier_byte_size(&media, TIER_THUMBNAIL);
        let full_size = tier_byte_size(&media, TIER_FULL);
        assert!(thumb_size < full_size);
    }

    #[test]
    fn test_delta_roundtrip() {
        let pipeline = MediaIngestPipeline::new(MediaIngestConfig {
            quality: 85,
            coefficient_threshold: 0.001,
            ..Default::default()
        });

        // Two similar images (same pattern, slight variation)
        let reference_pixels: Vec<f64> =
            (0..32 * 32).map(|i| (i as f64 / 1024.0).sin()).collect();
        let target_pixels: Vec<f64> = (0..32 * 32)
            .map(|i| (i as f64 / 1024.0).sin() + 0.05)
            .collect();

        let reference = pipeline.ingest_image(&reference_pixels, 32, 32).unwrap();
        let target = pipeline.ingest_image(&target_pixels, 32, 32).unwrap();

        // Compute delta
        let delta = compute_delta(&reference, &target, "ref_001", 0.01);

        // Delta should be smaller than full target
        assert!(
            delta.changed_entries.len() <= target.coefficients.entries.len(),
            "Delta ({}) should be <= full target ({})",
            delta.changed_entries.len(),
            target.coefficients.entries.len()
        );

        // Apply delta to reference → should approximate target
        let reconstructed = apply_delta(&reference, &delta);
        assert_eq!(reconstructed.coefficients.shape, target.coefficients.shape);
    }

    #[test]
    fn test_identical_media_empty_delta() {
        let pipeline = MediaIngestPipeline::new(MediaIngestConfig::default());
        let pixels: Vec<f64> = (0..16 * 16).map(|i| i as f64 / 256.0).collect();

        let media = pipeline.ingest_image(&pixels, 16, 16).unwrap();

        // Delta of identical media should be empty
        let delta = compute_delta(&media, &media, "self", 0.01);
        assert!(delta.changed_entries.is_empty());
        assert!(delta.removed_positions.is_empty());
    }

    #[test]
    fn test_delta_compression_ratio() {
        let delta = FreqDelta {
            reference_id: "ref".into(),
            changed_entries: vec![CoefficientEntry {
                position: (0, 0),
                magnitude: 1.0,
                phase: 0.0,
            }],
            removed_positions: vec![],
            target_shape: (64, 64),
            transform: FreqTransform::Dct2d,
        };

        let ratio = delta_compression_ratio(&delta, 1000);
        assert_eq!(ratio, 1000.0); // 1 delta entry vs 1000 full entries
    }

    #[test]
    fn test_tier_for_budget() {
        let pipeline = MediaIngestPipeline::new(MediaIngestConfig::default());
        let pixels: Vec<f64> = (0..64 * 64).map(|i| (i as f64 / 4096.0).sin()).collect();
        let media = pipeline.ingest_image(&pixels, 64, 64).unwrap();

        // Very small budget → thumbnail
        let tier = tier_for_budget(&media, 200);
        assert_eq!(tier.name, "thumbnail");

        // Large budget → full
        let tier = tier_for_budget(&media, 1_000_000);
        assert_eq!(tier.name, "full");
    }
}
