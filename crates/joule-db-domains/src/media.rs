//! HDC-powered Media and Content module
//!
//! Provides holographic encoding for:
//! - Content similarity and recommendation
//! - Audience segmentation
//! - Content moderation
//! - Engagement prediction

use joule_db_hdc::{BinaryHV, BundleAccumulator};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const DIMENSION: usize = 10000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ContentType {
    Video,
    Audio,
    Article,
    Image,
    Podcast,
    Livestream,
    ShortForm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Genre {
    News,
    Entertainment,
    Sports,
    Education,
    Music,
    Gaming,
    Lifestyle,
    Technology,
    Documentary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ContentRating {
    General,
    Teen,
    Mature,
    Restricted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModerationStatus {
    Approved,
    Pending,
    Flagged,
    Rejected,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Content {
    pub id: String,
    pub title: String,
    pub content_type: ContentType,
    pub genre: Genre,
    pub rating: ContentRating,
    pub duration_secs: u32,
    pub creator_id: String,
    pub tags: Vec<String>,
    pub transcript: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Creator {
    pub id: String,
    pub name: String,
    pub subscriber_count: u64,
    pub total_views: u64,
    pub content_types: Vec<ContentType>,
    pub genres: Vec<Genre>,
    pub avg_engagement_rate: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Viewer {
    pub id: String,
    pub watch_history: Vec<String>,
    pub preferred_genres: Vec<Genre>,
    pub avg_watch_time_secs: u32,
    pub subscription_tier: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Engagement {
    pub content_id: String,
    pub views: u64,
    pub likes: u64,
    pub shares: u64,
    pub comments: u64,
    pub avg_watch_percentage: f32,
    pub click_through_rate: f32,
}

joule_db_hdc::define_domain_module! {
    /// HDC encoder for media domain data
    pub struct MediaLink {
        seed: 0x0ED1_0001,
        dimension: 10000,
        fields: ["content", "creator", "viewer", "genre", "type", "tag", "engagement"],
        scalars: ["duration", "views", "engagement", "subscribers", "percentage"],
        enums: {
            content_type_vectors: ContentType => [ContentType::Video, ContentType::Audio, ContentType::Article, ContentType::Image, ContentType::Podcast, ContentType::Livestream, ContentType::ShortForm],
            genre_vectors: Genre => [Genre::News, Genre::Entertainment, Genre::Sports, Genre::Education, Genre::Music, Genre::Gaming, Genre::Lifestyle, Genre::Technology, Genre::Documentary],
            rating_vectors: ContentRating => [ContentRating::General, ContentRating::Teen, ContentRating::Mature, ContentRating::Restricted],
            status_vectors: ModerationStatus => [ModerationStatus::Approved, ModerationStatus::Pending, ModerationStatus::Flagged, ModerationStatus::Rejected]
        },
        dynamic: {
            tag_vectors: "tag"
        },
    }
}

impl MediaLink {
    pub fn encode_content(&mut self, content: &Content) -> BinaryHV {
        let type_hv =
            self.field_vectors["type"].bind(&self.content_type_vectors[&content.content_type]);
        let genre_hv = self.field_vectors["genre"].bind(&self.genre_vectors[&content.genre]);
        let rating_hv = self.rating_vectors[&content.rating].clone();
        let duration_hv = self.encode_scalar("duration", content.duration_secs.min(36000), 36000);
        let title_hv = BinaryHV::from_hash(content.title.as_bytes(), DIMENSION);
        let mut components = vec![type_hv, genre_hv, rating_hv, duration_hv, title_hv];
        for tag in &content.tags {
            let tag_vec = self.tag_vectors(tag);
            components.push(self.field_vectors["tag"].bind(&tag_vec));
        }
        if let Some(ref transcript) = content.transcript {
            components.push(BinaryHV::from_hash(transcript.as_bytes(), DIMENSION));
        }
        self.bundle(&components)
    }

    pub fn encode_creator(&self, creator: &Creator) -> BinaryHV {
        let subs_hv = self.encode_scalar(
            "subscribers",
            (creator.subscriber_count / 1000).min(10000) as u32,
            10000,
        );
        let views_hv = self.encode_scalar(
            "views",
            (creator.total_views / 10000).min(100000) as u32,
            100000,
        );
        let engagement_hv = self.encode_scalar(
            "engagement",
            (creator.avg_engagement_rate * 1000.0) as u32,
            1000,
        );
        let mut components = vec![subs_hv, views_hv, engagement_hv];
        for ct in &creator.content_types {
            components.push(self.field_vectors["type"].bind(&self.content_type_vectors[ct]));
        }
        for genre in &creator.genres {
            components.push(self.field_vectors["genre"].bind(&self.genre_vectors[genre]));
        }
        self.bundle(&components)
    }

    pub fn encode_viewer(&mut self, viewer: &Viewer) -> BinaryHV {
        let watch_time_hv =
            self.encode_scalar("duration", viewer.avg_watch_time_secs.min(7200), 7200);
        let mut components = vec![watch_time_hv];
        for genre in &viewer.preferred_genres {
            components.push(self.field_vectors["genre"].bind(&self.genre_vectors[genre]));
        }
        for content_id in viewer.watch_history.iter().take(10) {
            components.push(BinaryHV::from_hash(content_id.as_bytes(), DIMENSION));
        }
        self.bundle(&components)
    }

    pub fn encode_engagement(&self, engagement: &Engagement) -> BinaryHV {
        let views_hv =
            self.encode_scalar("views", (engagement.views / 100).min(100000) as u32, 100000);
        let likes_hv = self.encode_scalar(
            "engagement",
            (engagement.likes / 10).min(10000) as u32,
            10000,
        );
        let watch_pct_hv = self.encode_scalar(
            "percentage",
            (engagement.avg_watch_percentage * 100.0) as u32,
            100,
        );
        let ctr_hv = self.encode_scalar(
            "percentage",
            (engagement.click_through_rate * 1000.0) as u32,
            1000,
        );
        self.bundle(&[views_hv, likes_hv, watch_pct_hv, ctr_hv])
    }
}

pub struct ContentLibrary {
    encoder: MediaLink,
    content_vectors: HashMap<String, BinaryHV>,
    content: HashMap<String, Content>,
}

impl ContentLibrary {
    pub fn new() -> Self {
        Self {
            encoder: MediaLink::new(),
            content_vectors: HashMap::new(),
            content: HashMap::new(),
        }
    }

    pub fn add_content(&mut self, content: Content) {
        let hv = self.encoder.encode_content(&content);
        self.content_vectors.insert(content.id.clone(), hv);
        self.content.insert(content.id.clone(), content);
    }

    pub fn find_similar(&self, content_id: &str, limit: usize) -> Vec<(String, f32)> {
        let query = match self.content_vectors.get(content_id) {
            Some(hv) => hv,
            None => return Vec::new(),
        };
        let mut results: Vec<_> = self
            .content_vectors
            .iter()
            .filter(|(id, _)| *id != content_id)
            .map(|(id, hv)| (id.clone(), query.similarity(hv)))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        results
    }

    pub fn content_count(&self) -> usize {
        self.content.len()
    }
}

impl Default for ContentLibrary {
    fn default() -> Self {
        Self::new()
    }
}

pub struct ContentRecommender {
    encoder: MediaLink,
    viewer_profiles: HashMap<String, BinaryHV>,
    content_library: ContentLibrary,
}

impl ContentRecommender {
    pub fn new() -> Self {
        Self {
            encoder: MediaLink::new(),
            viewer_profiles: HashMap::new(),
            content_library: ContentLibrary::new(),
        }
    }

    pub fn update_viewer_profile(&mut self, viewer: &Viewer) {
        let hv = self.encoder.encode_viewer(viewer);
        self.viewer_profiles.insert(viewer.id.clone(), hv);
    }

    pub fn add_content(&mut self, content: Content) {
        self.content_library.add_content(content);
    }

    pub fn recommend(&self, viewer_id: &str, limit: usize) -> Vec<(String, f32)> {
        let profile = match self.viewer_profiles.get(viewer_id) {
            Some(hv) => hv,
            None => return Vec::new(),
        };
        let mut results: Vec<_> = self
            .content_library
            .content_vectors
            .iter()
            .map(|(id, hv)| (id.clone(), profile.similarity(hv)))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        results
    }
}

impl Default for ContentRecommender {
    fn default() -> Self {
        Self::new()
    }
}

pub struct ContentModerator {
    encoder: MediaLink,
    approved_patterns: BundleAccumulator,
    rejected_patterns: BundleAccumulator,
    threshold: f32,
}

#[derive(Debug, Clone)]
pub struct ModerationResult {
    pub content_id: String,
    pub status: ModerationStatus,
    pub confidence: f32,
    pub flags: Vec<String>,
}

impl ContentModerator {
    pub fn new(threshold: f32) -> Self {
        Self {
            encoder: MediaLink::new(),
            approved_patterns: BundleAccumulator::new(DIMENSION),
            rejected_patterns: BundleAccumulator::new(DIMENSION),
            threshold,
        }
    }

    pub fn learn_approved(&mut self, content: &Content) {
        self.approved_patterns
            .add(&self.encoder.encode_content(content));
    }
    pub fn learn_rejected(&mut self, content: &Content) {
        self.rejected_patterns
            .add(&self.encoder.encode_content(content));
    }

    pub fn moderate(&mut self, content: &Content) -> ModerationResult {
        let hv = self.encoder.encode_content(content);
        let approved_sim = hv.similarity(&self.approved_patterns.threshold());
        let rejected_sim = hv.similarity(&self.rejected_patterns.threshold());
        let score = rejected_sim - approved_sim;
        let status = if score > self.threshold {
            ModerationStatus::Flagged
        } else {
            ModerationStatus::Approved
        };
        ModerationResult {
            content_id: content.id.clone(),
            status,
            confidence: score.abs(),
            flags: vec![],
        }
    }
}

impl Default for ContentModerator {
    fn default() -> Self {
        Self::new(0.3)
    }
}

pub struct EngagementPredictor {
    encoder: MediaLink,
    high_engagement_patterns: BundleAccumulator,
    low_engagement_patterns: BundleAccumulator,
}

impl EngagementPredictor {
    pub fn new() -> Self {
        Self {
            encoder: MediaLink::new(),
            high_engagement_patterns: BundleAccumulator::new(DIMENSION),
            low_engagement_patterns: BundleAccumulator::new(DIMENSION),
        }
    }

    pub fn learn_high_engagement(&mut self, content: &Content) {
        self.high_engagement_patterns
            .add(&self.encoder.encode_content(content));
    }
    pub fn learn_low_engagement(&mut self, content: &Content) {
        self.low_engagement_patterns
            .add(&self.encoder.encode_content(content));
    }

    pub fn predict(&mut self, content: &Content) -> f32 {
        let hv = self.encoder.encode_content(content);
        let high_sim = hv.similarity(&self.high_engagement_patterns.threshold());
        let low_sim = hv.similarity(&self.low_engagement_patterns.threshold());
        (high_sim - low_sim + 1.0) / 2.0
    }
}

impl Default for EngagementPredictor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_encoding() {
        let mut encoder = MediaLink::new();
        let content = Content {
            id: "V1".to_string(),
            title: "Tutorial".to_string(),
            content_type: ContentType::Video,
            genre: Genre::Education,
            rating: ContentRating::General,
            duration_secs: 600,
            creator_id: "C1".to_string(),
            tags: vec!["coding".to_string()],
            transcript: None,
        };
        assert_eq!(encoder.encode_content(&content).dimension(), DIMENSION);
    }

    #[test]
    fn test_creator_encoding() {
        let encoder = MediaLink::new();
        let creator = Creator {
            id: "C1".to_string(),
            name: "TechChannel".to_string(),
            subscriber_count: 100000,
            total_views: 5000000,
            content_types: vec![ContentType::Video],
            genres: vec![Genre::Technology],
            avg_engagement_rate: 0.05,
        };
        assert_eq!(encoder.encode_creator(&creator).dimension(), DIMENSION);
    }

    #[test]
    fn test_content_library() {
        let mut library = ContentLibrary::new();
        library.add_content(Content {
            id: "V1".to_string(),
            title: "Test".to_string(),
            content_type: ContentType::Video,
            genre: Genre::Entertainment,
            rating: ContentRating::General,
            duration_secs: 300,
            creator_id: "C1".to_string(),
            tags: vec![],
            transcript: None,
        });
        assert_eq!(library.content_count(), 1);
    }

    #[test]
    fn test_content_moderation() {
        let mut moderator = ContentModerator::new(0.3);
        let content = Content {
            id: "V1".to_string(),
            title: "Safe".to_string(),
            content_type: ContentType::Video,
            genre: Genre::Education,
            rating: ContentRating::General,
            duration_secs: 300,
            creator_id: "C1".to_string(),
            tags: vec![],
            transcript: None,
        };
        moderator.learn_approved(&content);
        let result = moderator.moderate(&content);
        assert_eq!(result.status, ModerationStatus::Approved);
    }

    #[test]
    fn test_engagement_prediction() {
        let mut predictor = EngagementPredictor::new();
        let content = Content {
            id: "V1".to_string(),
            title: "Viral".to_string(),
            content_type: ContentType::ShortForm,
            genre: Genre::Entertainment,
            rating: ContentRating::General,
            duration_secs: 60,
            creator_id: "C1".to_string(),
            tags: vec!["trending".to_string()],
            transcript: None,
        };
        predictor.learn_high_engagement(&content);
        let score = predictor.predict(&content);
        assert!(score >= 0.0 && score <= 1.0);
    }
}

// ============================================================================
// MediaQL: Frequency-Domain Media Encoders
// ============================================================================

/// Encodes frequency-domain coefficients into a BinaryHV hologram.
///
/// This is the "right half" of middle-out: frequency coefficients → HDC hologram.
/// The hologram captures the spectral signature of the media for similarity search.
pub struct FrequencyDomainEncoder {
    dimension: usize,
}

impl FrequencyDomainEncoder {
    /// Create a new frequency-domain encoder.
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    /// Encode image DCT coefficients into a BinaryHV.
    ///
    /// Strategy: encode the spatial frequency distribution as a hologram.
    /// Low frequencies → shape/structure, high frequencies → texture/detail.
    pub fn encode_image_coefficients(
        &self,
        coefficients: &[(u32, u32, f32)], // (x, y, magnitude)
        width: u32,
        height: u32,
    ) -> BinaryHV {
        let mut acc = BundleAccumulator::new(self.dimension);

        // Create base vectors for frequency bands
        let dc_base = BinaryHV::from_hash(b"freq:dc", self.dimension);
        let low_base = BinaryHV::from_hash(b"freq:low", self.dimension);
        let mid_base = BinaryHV::from_hash(b"freq:mid", self.dimension);
        let high_base = BinaryHV::from_hash(b"freq:high", self.dimension);

        let max_freq = ((width * width + height * height) as f64).sqrt();

        for &(x, y, mag) in coefficients {
            if mag.abs() < 1e-6 {
                continue;
            }

            // Classify frequency band based on position
            let freq_dist = ((x * x + y * y) as f64).sqrt() / max_freq;
            let band_hv = if freq_dist < 0.05 {
                &dc_base
            } else if freq_dist < 0.25 {
                &low_base
            } else if freq_dist < 0.6 {
                &mid_base
            } else {
                &high_base
            };

            // Encode position as a deterministic vector
            let pos_seed = format!("pos:{}:{}", x, y);
            let pos_hv = BinaryHV::from_hash(pos_seed.as_bytes(), self.dimension);

            // Bind band × position, weighted by magnitude
            let bound = band_hv.bind(&pos_hv);

            // Add with implicit magnitude weighting (more adds = stronger signal)
            let repeats = (mag.abs().log2().max(0.0) as usize + 1).min(5);
            for _ in 0..repeats {
                acc.add(&bound);
            }
        }

        acc.threshold()
    }

    /// Encode audio STFT coefficients into a BinaryHV.
    ///
    /// Strategy: encode the spectral envelope (which frequencies are present)
    /// and temporal structure (how they change over time).
    pub fn encode_audio_coefficients(
        &self,
        coefficients: &[(u32, u32, f32)], // (frame, freq_bin, magnitude)
        n_freq_bins: u32,
    ) -> BinaryHV {
        let mut acc = BundleAccumulator::new(self.dimension);

        // Create base vectors for frequency ranges
        let bass_base = BinaryHV::from_hash(b"audio:bass", self.dimension);
        let mid_base = BinaryHV::from_hash(b"audio:mid", self.dimension);
        let treble_base = BinaryHV::from_hash(b"audio:treble", self.dimension);

        for &(frame, bin, mag) in coefficients {
            if mag.abs() < 1e-6 {
                continue;
            }

            let freq_ratio = bin as f64 / n_freq_bins as f64;
            let band_hv = if freq_ratio < 0.1 {
                &bass_base
            } else if freq_ratio < 0.4 {
                &mid_base
            } else {
                &treble_base
            };

            // Encode temporal position via permutation (preserves sequence)
            let frame_shift = (frame as usize) % self.dimension;
            let temporal_hv = band_hv.permute_words(frame_shift);

            // Encode frequency bin
            let bin_seed = format!("bin:{}", bin);
            let bin_hv = BinaryHV::from_hash(bin_seed.as_bytes(), self.dimension);

            let bound = temporal_hv.bind(&bin_hv);

            let repeats = (mag.abs().log2().max(0.0) as usize + 1).min(3);
            for _ in 0..repeats {
                acc.add(&bound);
            }
        }

        acc.threshold()
    }

    /// Encode a perceptual hash as a BinaryHV.
    ///
    /// This allows pHash-based dedup queries through the same HDC similarity engine.
    pub fn encode_phash(&self, phash: u64) -> BinaryHV {
        let bytes = phash.to_le_bytes();
        BinaryHV::from_hash(&bytes, self.dimension)
    }
}

#[cfg(test)]
mod freq_encoder_tests {
    use super::*;

    #[test]
    fn test_image_encoding_similar_inputs() {
        let encoder = FrequencyDomainEncoder::new(DIMENSION);

        // Two similar images (same structure, slightly different magnitudes)
        let coeffs1: Vec<(u32, u32, f32)> = vec![
            (0, 0, 100.0), // DC
            (1, 0, 10.0),
            (0, 1, 8.0),
            (2, 2, 3.0),
        ];
        let coeffs2: Vec<(u32, u32, f32)> = vec![
            (0, 0, 105.0), // Slightly different DC
            (1, 0, 11.0),
            (0, 1, 7.5),
            (2, 2, 3.5),
        ];

        let hv1 = encoder.encode_image_coefficients(&coeffs1, 64, 64);
        let hv2 = encoder.encode_image_coefficients(&coeffs2, 64, 64);

        // Similar spectral content → high hologram similarity
        let sim = hv1.similarity(&hv2);
        assert!(
            sim > 0.6,
            "Similar images should have high similarity, got {}",
            sim
        );
    }

    #[test]
    fn test_image_encoding_different_inputs() {
        let encoder = FrequencyDomainEncoder::new(DIMENSION);

        // Low frequency image (smooth)
        let smooth: Vec<(u32, u32, f32)> = vec![
            (0, 0, 200.0),
            (1, 0, 5.0),
            (0, 1, 5.0),
        ];
        // High frequency image (textured)
        let textured: Vec<(u32, u32, f32)> = vec![
            (0, 0, 50.0),
            (30, 30, 20.0),
            (25, 25, 15.0),
        ];

        let hv1 = encoder.encode_image_coefficients(&smooth, 64, 64);
        let hv2 = encoder.encode_image_coefficients(&textured, 64, 64);

        let sim = hv1.similarity(&hv2);
        assert!(
            sim < 0.7,
            "Different spectral profiles should have lower similarity, got {}",
            sim
        );
    }

    #[test]
    fn test_audio_encoding() {
        let encoder = FrequencyDomainEncoder::new(DIMENSION);

        // Bass-heavy audio
        let bass: Vec<(u32, u32, f32)> = vec![
            (0, 5, 50.0),
            (1, 3, 40.0),
            (2, 7, 30.0),
        ];

        let hv = encoder.encode_audio_coefficients(&bass, 1024);
        // Should produce a valid hologram
        assert_eq!(hv.dimension(), DIMENSION);
    }
}
