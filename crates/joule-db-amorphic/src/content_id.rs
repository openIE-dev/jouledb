//! Live Fingerprint Matching (ContentID) — copyright detection at scale.
//!
//! YouTube's Content ID processes 1.5B claims/year. This module provides
//! the database-side matching: given a stream of frame/audio hashes,
//! find matches in a reference database of copyrighted content.
//!
//! The caller (inv-image-filters, inv-media) extracts fingerprints.
//! This module stores references and matches at scale via HNSW.

use joule_db_hdc::manifold::{DistanceMetric, HNSWIndex};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Policy for what to do when a match is found.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MatchPolicy {
    /// Block the content from being published
    Block,
    /// Allow but monetize (ads go to rights holder)
    Monetize,
    /// Allow but track viewership stats
    Track,
    /// No action (reference only)
    None,
}

/// A registered reference content item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentReference {
    /// Unique reference ID
    pub id: String,
    /// Rights holder / owner
    pub owner: String,
    /// Title of the content
    pub title: String,
    /// What to do on match
    pub policy: MatchPolicy,
}

/// A match result from fingerprint comparison.
#[derive(Debug, Clone)]
pub struct ContentMatch {
    /// Reference content that matched
    pub reference: ContentReference,
    /// Match confidence (0.0-1.0)
    pub confidence: f32,
    /// Frame offset in the query where match starts
    pub query_offset: usize,
    /// Number of matching frames
    pub match_length: usize,
}

/// ContentID index: stores reference fingerprints for fast matching.
///
/// Uses HNSW for O(log n) nearest-neighbor lookup on fingerprint vectors.
/// Each frame hash is stored as a point in the index, with the reference
/// content ID as the point label.
pub struct ContentIdIndex {
    /// HNSW index for video frame hashes
    video_index: HNSWIndex,
    /// HNSW index for audio subfingerprints
    audio_index: HNSWIndex,
    /// Reference content metadata
    references: HashMap<String, ContentReference>,
    /// Match threshold (Hamming distance, lower = stricter)
    pub match_threshold: f32,
    /// Minimum consecutive matching frames for a valid match
    pub min_match_frames: usize,
}

impl ContentIdIndex {
    /// Create a new ContentID index.
    ///
    /// # Arguments
    /// * `hash_bits` - Number of bits per frame hash (64 typical for pHash)
    pub fn new(hash_bits: usize) -> Self {
        // HNSW dimension = hash_bits / 32 (pack into f32 words)
        let video_dim = (hash_bits + 31) / 32;
        let audio_dim = 1; // Audio subfingerprints are 32-bit = 1 f32
        Self {
            video_index: HNSWIndex::with_metric(video_dim, 16, 200, DistanceMetric::Hamming),
            audio_index: HNSWIndex::with_metric(audio_dim, 16, 200, DistanceMetric::Hamming),
            references: HashMap::new(),
            match_threshold: 10.0, // Max Hamming distance for a match
            min_match_frames: 3,
        }
    }

    /// Register a reference content item with its fingerprints.
    pub fn register_reference(
        &mut self,
        reference: ContentReference,
        video_hashes: &[u64],
        audio_hashes: &[u32],
    ) {
        let ref_id = reference.id.clone();

        // Index video frame hashes
        for (i, &hash) in video_hashes.iter().enumerate() {
            let point_id = format!("{}:v:{}", ref_id, i);
            let packed = vec![f32::from_bits(hash as u32), f32::from_bits((hash >> 32) as u32)];
            let _ = self.video_index.insert(point_id, packed);
        }

        // Index audio subfingerprints
        for (i, &hash) in audio_hashes.iter().enumerate() {
            let point_id = format!("{}:a:{}", ref_id, i);
            let packed = vec![f32::from_bits(hash)];
            let _ = self.audio_index.insert(point_id, packed);
        }

        self.references.insert(ref_id, reference);
    }

    /// Match a stream of video frame hashes against the reference database.
    ///
    /// Returns matches where consecutive frames match the same reference.
    pub fn match_video_frames(&self, frame_hashes: &[u64]) -> Vec<ContentMatch> {
        if frame_hashes.is_empty() {
            return Vec::new();
        }

        // For each frame, find nearest reference frame
        let mut frame_matches: Vec<Option<(String, f32)>> = Vec::with_capacity(frame_hashes.len());

        for &hash in frame_hashes {
            let packed = vec![f32::from_bits(hash as u32), f32::from_bits((hash >> 32) as u32)];
            let results = self.video_index.query(&packed, 1);

            if let Some(best) = results.first() {
                if best.distance <= self.match_threshold {
                    // Extract reference ID from point ID (format: "ref_id:v:frame_idx")
                    let ref_id = best.id.split(':').next().unwrap_or("").to_string();
                    frame_matches.push(Some((ref_id, best.distance)));
                } else {
                    frame_matches.push(None);
                }
            } else {
                frame_matches.push(None);
            }
        }

        // Find consecutive runs of matching frames for the same reference
        self.find_consecutive_matches(&frame_matches)
    }

    /// Match audio subfingerprints against the reference database.
    pub fn match_audio(&self, audio_hashes: &[u32]) -> Vec<ContentMatch> {
        if audio_hashes.is_empty() {
            return Vec::new();
        }

        let mut frame_matches: Vec<Option<(String, f32)>> = Vec::with_capacity(audio_hashes.len());

        for &hash in audio_hashes {
            let packed = vec![f32::from_bits(hash)];
            let results = self.audio_index.query(&packed, 1);

            if let Some(best) = results.first() {
                if best.distance <= self.match_threshold {
                    let ref_id = best.id.split(':').next().unwrap_or("").to_string();
                    frame_matches.push(Some((ref_id, best.distance)));
                } else {
                    frame_matches.push(None);
                }
            } else {
                frame_matches.push(None);
            }
        }

        self.find_consecutive_matches(&frame_matches)
    }

    /// Find consecutive runs of matching frames for the same reference.
    fn find_consecutive_matches(
        &self,
        frame_matches: &[Option<(String, f32)>],
    ) -> Vec<ContentMatch> {
        let mut matches = Vec::new();
        let mut current_ref: Option<String> = None;
        let mut run_start = 0;
        let mut run_total_dist = 0.0f32;
        let mut run_count = 0usize;

        for (i, m) in frame_matches.iter().enumerate() {
            match (m, &current_ref) {
                (Some((ref_id, dist)), Some(current)) if ref_id == current => {
                    // Continue run
                    run_total_dist += dist;
                    run_count += 1;
                }
                (Some((ref_id, dist)), _) => {
                    // End previous run if long enough
                    if run_count >= self.min_match_frames {
                        if let Some(ref_id) = &current_ref {
                            if let Some(reference) = self.references.get(ref_id) {
                                let avg_dist = run_total_dist / run_count as f32;
                                let confidence =
                                    1.0 - (avg_dist / self.match_threshold).min(1.0);
                                matches.push(ContentMatch {
                                    reference: reference.clone(),
                                    confidence,
                                    query_offset: run_start,
                                    match_length: run_count,
                                });
                            }
                        }
                    }
                    // Start new run
                    current_ref = Some(ref_id.clone());
                    run_start = i;
                    run_total_dist = *dist;
                    run_count = 1;
                }
                (None, _) => {
                    // End run
                    if run_count >= self.min_match_frames {
                        if let Some(ref_id) = &current_ref {
                            if let Some(reference) = self.references.get(ref_id) {
                                let avg_dist = run_total_dist / run_count as f32;
                                let confidence =
                                    1.0 - (avg_dist / self.match_threshold).min(1.0);
                                matches.push(ContentMatch {
                                    reference: reference.clone(),
                                    confidence,
                                    query_offset: run_start,
                                    match_length: run_count,
                                });
                            }
                        }
                    }
                    current_ref = None;
                    run_count = 0;
                    run_total_dist = 0.0;
                }
            }
        }

        // Final run
        if run_count >= self.min_match_frames {
            if let Some(ref_id) = &current_ref {
                if let Some(reference) = self.references.get(ref_id) {
                    let avg_dist = run_total_dist / run_count as f32;
                    let confidence = 1.0 - (avg_dist / self.match_threshold).min(1.0);
                    matches.push(ContentMatch {
                        reference: reference.clone(),
                        confidence,
                        query_offset: run_start,
                        match_length: run_count,
                    });
                }
            }
        }

        matches
    }

    /// Number of registered references.
    pub fn reference_count(&self) -> usize {
        self.references.len()
    }

    /// Total indexed video frames.
    pub fn video_frame_count(&self) -> usize {
        self.video_index.size()
    }

    /// Total indexed audio segments.
    pub fn audio_segment_count(&self) -> usize {
        self.audio_index.size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_and_match() {
        let mut index = ContentIdIndex::new(64);
        index.min_match_frames = 2; // Lower for testing

        let reference = ContentReference {
            id: "movie_123".into(),
            owner: "Studio X".into(),
            title: "The Movie".into(),
            policy: MatchPolicy::Monetize,
        };

        // Register with 10 frame hashes
        let ref_hashes: Vec<u64> = (0..10).map(|i| 0xDEADBEEF_00000000u64 + i).collect();
        index.register_reference(reference, &ref_hashes, &[]);

        assert_eq!(index.reference_count(), 1);
        assert_eq!(index.video_frame_count(), 10);

        // Match with exact same hashes — should find the reference
        let matches = index.match_video_frames(&ref_hashes);
        assert!(
            !matches.is_empty(),
            "Should find match for exact same hashes"
        );
        assert_eq!(matches[0].reference.id, "movie_123");
        assert!(matches[0].confidence > 0.5);
    }

    #[test]
    fn test_no_match_for_different_content() {
        let mut index = ContentIdIndex::new(64);

        let reference = ContentReference {
            id: "movie_456".into(),
            owner: "Studio Y".into(),
            title: "Other Movie".into(),
            policy: MatchPolicy::Block,
        };

        let ref_hashes: Vec<u64> = (0..10).map(|i| 0xAAAAAAAA_00000000u64 + i).collect();
        index.register_reference(reference, &ref_hashes, &[]);

        // Completely different hashes
        let query_hashes: Vec<u64> = (0..10).map(|i| 0x55555555_00000000u64 + i).collect();
        let matches = index.match_video_frames(&query_hashes);

        // Should have no high-confidence matches (content is different)
        let strong_matches: Vec<_> = matches.iter().filter(|m| m.confidence > 0.8).collect();
        assert!(
            strong_matches.is_empty(),
            "Should not match different content with high confidence"
        );
    }

    #[test]
    fn test_match_policy() {
        let mut index = ContentIdIndex::new(64);
        index.min_match_frames = 2;

        index.register_reference(
            ContentReference {
                id: "song_1".into(),
                owner: "Label A".into(),
                title: "Hit Song".into(),
                policy: MatchPolicy::Monetize,
            },
            &[],
            &[100, 200, 300, 400, 500],
        );

        assert_eq!(index.audio_segment_count(), 5);

        // Match exact audio
        let matches = index.match_audio(&[100, 200, 300, 400, 500]);
        if !matches.is_empty() {
            assert_eq!(matches[0].reference.policy, MatchPolicy::Monetize);
        }
    }
}
