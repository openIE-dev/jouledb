//! HDC-powered Multi-modal Data Fusion module
//!
//! Provides holographic encoding for:
//! - Text, image, audio unified embeddings
//! - Cross-modal similarity search
//! - Multi-modal entity resolution
//! - Fusion of heterogeneous data sources

use joule_db_hdc::{BinaryHV, BundleAccumulator};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const DIMENSION: usize = 10000;

// ============================================================================
// Core Types
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Modality {
    Text,
    Image,
    Audio,
    Video,
    Tabular,
    Graph,
    TimeSeries,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextData {
    pub content: String,
    pub language: String,
    pub tokens: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageData {
    pub id: String,
    pub width: u32,
    pub height: u32,
    pub color_histogram: Vec<f32>,
    pub edge_histogram: Vec<f32>,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioData {
    pub id: String,
    pub duration_ms: u64,
    pub sample_rate: u32,
    pub mfcc_features: Vec<f32>,
    pub spectral_features: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalEntity {
    pub id: String,
    pub text: Option<TextData>,
    pub image: Option<ImageData>,
    pub audio: Option<AudioData>,
    pub metadata: HashMap<String, String>,
}

// ============================================================================
// Multimodal Encoder (macro-generated core)
// ============================================================================

joule_db_hdc::define_domain_module! {
    /// HDC encoder core for multimodal domain data
    pub struct MultimodalLinkCore {
        seed: 0x0001_0001,
        dimension: 10000,
        fields: ["content", "token", "color", "edge", "mfcc", "spectral", "tag", "position"],
        scalars: ["intensity", "frequency", "position", "duration"],
        enums: {
            modality_vectors: Modality => [Modality::Text, Modality::Image, Modality::Audio,
                                           Modality::Video, Modality::Tabular, Modality::Graph,
                                           Modality::TimeSeries]
        },
    }
}

pub struct MultimodalLink {
    core: MultimodalLinkCore,
    token_cache: HashMap<String, BinaryHV>,
    char_vectors: HashMap<char, BinaryHV>,
}

impl MultimodalLink {
    pub fn new() -> Self {
        // Character vectors for text encoding - deterministic via from_hash
        let chars: Vec<char> = "abcdefghijklmnopqrstuvwxyz0123456789 .,!?-"
            .chars()
            .collect();
        let char_vectors: HashMap<char, BinaryHV> = chars
            .iter()
            .map(|c| (*c, BinaryHV::from_hash(&[*c as u8], DIMENSION)))
            .collect();

        Self {
            core: MultimodalLinkCore::new(),
            token_cache: HashMap::new(),
            char_vectors,
        }
    }

    fn get_token_vector(&mut self, token: &str) -> BinaryHV {
        if !self.token_cache.contains_key(token) {
            // Encode token using character n-grams
            let lower = token.to_lowercase();
            let chars: Vec<char> = lower.chars().collect();
            let mut components = Vec::new();

            for (i, c) in chars.iter().enumerate() {
                if let Some(char_hv) = self.char_vectors.get(c) {
                    components.push(char_hv.permute(i));
                }
            }

            let token_hv = if components.is_empty() {
                BinaryHV::from_hash(token.as_bytes(), DIMENSION)
            } else {
                self.core.bundle(&components)
            };

            self.token_cache.insert(token.to_string(), token_hv);
        }
        self.token_cache.get(token).unwrap().clone()
    }

    pub fn encode_text(&mut self, text: &TextData) -> BinaryHV {
        let mut components = vec![self.core.modality_vectors[&Modality::Text].clone()];

        // Encode tokens with position
        for (i, token) in text.tokens.iter().take(100).enumerate() {
            let token_hv = self.get_token_vector(token);
            let pos_hv = self.core.encode_scalar("position", i as u32, 100);
            components.push(
                self.core.field_vectors["token"]
                    .bind(&token_hv)
                    .bind(&pos_hv),
            );
        }

        self.core.bundle(&components)
    }

    pub fn encode_image(&mut self, image: &ImageData) -> BinaryHV {
        let mut components = vec![self.core.modality_vectors[&Modality::Image].clone()];

        // Encode color histogram
        for (i, &intensity) in image.color_histogram.iter().take(16).enumerate() {
            let int_scaled = (intensity * 100.0) as u32;
            let color_hv = self.core.field_vectors["color"].bind(&self.core.encode_scalar(
                "intensity",
                int_scaled,
                100,
            ));
            components.push(color_hv.permute(i));
        }

        // Encode edge histogram
        for (i, &strength) in image.edge_histogram.iter().take(16).enumerate() {
            let str_scaled = (strength * 100.0) as u32;
            let edge_hv = self.core.field_vectors["edge"].bind(&self.core.encode_scalar(
                "intensity",
                str_scaled,
                100,
            ));
            components.push(edge_hv.permute(i + 16));
        }

        // Encode tags
        for tag in &image.tags {
            let tag_hv = self.get_token_vector(tag);
            components.push(self.core.field_vectors["tag"].bind(&tag_hv));
        }

        self.core.bundle(&components)
    }

    pub fn encode_audio(&mut self, audio: &AudioData) -> BinaryHV {
        let mut components = vec![self.core.modality_vectors[&Modality::Audio].clone()];

        // Encode MFCC features
        for (i, &coef) in audio.mfcc_features.iter().take(13).enumerate() {
            let coef_scaled = ((coef + 50.0) / 100.0 * 100.0).clamp(0.0, 100.0) as u32;
            let mfcc_hv = self.core.field_vectors["mfcc"].bind(&self.core.encode_scalar(
                "frequency",
                coef_scaled,
                100,
            ));
            components.push(mfcc_hv.permute(i));
        }

        // Encode spectral features
        for (i, &feat) in audio.spectral_features.iter().take(8).enumerate() {
            let feat_scaled = (feat * 100.0).clamp(0.0, 100.0) as u32;
            let spec_hv = self.core.field_vectors["spectral"].bind(&self.core.encode_scalar(
                "intensity",
                feat_scaled,
                100,
            ));
            components.push(spec_hv.permute(i + 13));
        }

        // Duration encoding
        let dur_hv = self
            .core
            .encode_scalar("duration", (audio.duration_ms / 1000) as u32, 600);
        components.push(dur_hv);

        self.core.bundle(&components)
    }

    pub fn encode_multimodal(&mut self, entity: &MultimodalEntity) -> BinaryHV {
        let mut components = Vec::new();

        if let Some(ref text) = entity.text {
            components.push(self.encode_text(text));
        }
        if let Some(ref image) = entity.image {
            components.push(self.encode_image(image));
        }
        if let Some(ref audio) = entity.audio {
            components.push(self.encode_audio(audio));
        }

        if components.is_empty() {
            BinaryHV::from_hash(entity.id.as_bytes(), DIMENSION)
        } else {
            self.core.bundle(&components)
        }
    }
}

impl Default for MultimodalLink {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Multimodal Database
// ============================================================================

pub struct MultimodalDb {
    encoder: MultimodalLink,
    text_hologram: BundleAccumulator,
    image_hologram: BundleAccumulator,
    audio_hologram: BundleAccumulator,
    unified_hologram: BundleAccumulator,
    entity_vectors: HashMap<String, BinaryHV>,
    modality_vectors: HashMap<String, HashMap<Modality, BinaryHV>>,
    entities: HashMap<String, MultimodalEntity>,
}

impl MultimodalDb {
    pub fn new() -> Self {
        Self {
            encoder: MultimodalLink::new(),
            text_hologram: BundleAccumulator::new(DIMENSION),
            image_hologram: BundleAccumulator::new(DIMENSION),
            audio_hologram: BundleAccumulator::new(DIMENSION),
            unified_hologram: BundleAccumulator::new(DIMENSION),
            entity_vectors: HashMap::new(),
            modality_vectors: HashMap::new(),
            entities: HashMap::new(),
        }
    }

    pub fn insert(&mut self, entity: MultimodalEntity) {
        let unified_hv = self.encoder.encode_multimodal(&entity);
        self.unified_hologram.add(&unified_hv);
        self.entity_vectors.insert(entity.id.clone(), unified_hv);

        let mut mod_vecs = HashMap::new();

        if let Some(ref text) = entity.text {
            let text_hv = self.encoder.encode_text(text);
            self.text_hologram.add(&text_hv);
            mod_vecs.insert(Modality::Text, text_hv);
        }
        if let Some(ref image) = entity.image {
            let image_hv = self.encoder.encode_image(image);
            self.image_hologram.add(&image_hv);
            mod_vecs.insert(Modality::Image, image_hv);
        }
        if let Some(ref audio) = entity.audio {
            let audio_hv = self.encoder.encode_audio(audio);
            self.audio_hologram.add(&audio_hv);
            mod_vecs.insert(Modality::Audio, audio_hv);
        }

        self.modality_vectors.insert(entity.id.clone(), mod_vecs);
        self.entities.insert(entity.id.clone(), entity);
    }

    pub fn search_unified(
        &self,
        entity_id: &str,
        min_sim: f32,
        limit: usize,
    ) -> Vec<(String, f32)> {
        let query_hv = match self.entity_vectors.get(entity_id) {
            Some(hv) => hv,
            None => return Vec::new(),
        };

        let mut results: Vec<(String, f32)> = self
            .entity_vectors
            .iter()
            .filter(|(id, _)| *id != entity_id)
            .map(|(id, hv)| (id.clone(), query_hv.similarity(hv)))
            .filter(|(_, sim)| *sim >= min_sim)
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        results
    }

    pub fn search_by_modality(
        &self,
        entity_id: &str,
        modality: Modality,
        min_sim: f32,
        limit: usize,
    ) -> Vec<(String, f32)> {
        let query_hv = match self
            .modality_vectors
            .get(entity_id)
            .and_then(|m| m.get(&modality))
        {
            Some(hv) => hv,
            None => return Vec::new(),
        };

        let mut results: Vec<(String, f32)> = self
            .modality_vectors
            .iter()
            .filter(|(id, _)| *id != entity_id)
            .filter_map(|(id, mods)| {
                mods.get(&modality)
                    .map(|hv| (id.clone(), query_hv.similarity(hv)))
            })
            .filter(|(_, sim)| *sim >= min_sim)
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        results
    }

    pub fn cross_modal_search(
        &mut self,
        text: &TextData,
        target_modality: Modality,
        min_sim: f32,
        limit: usize,
    ) -> Vec<(String, f32)> {
        let query_hv = self.encoder.encode_text(text);

        let mut results: Vec<(String, f32)> = self
            .modality_vectors
            .iter()
            .filter_map(|(id, mods)| {
                mods.get(&target_modality)
                    .map(|hv| (id.clone(), query_hv.similarity(hv)))
            })
            .filter(|(_, sim)| *sim >= min_sim)
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        results
    }

    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }
}

impl Default for MultimodalDb {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Entity Resolution (Cross-Modal Matching)
// ============================================================================

pub struct EntityResolver {
    encoder: MultimodalLink,
    entity_signatures: HashMap<String, BinaryHV>,
    match_threshold: f32,
}

#[derive(Debug, Clone)]
pub struct EntityMatch {
    pub entity_a: String,
    pub entity_b: String,
    pub confidence: f32,
    pub matching_modalities: Vec<Modality>,
}

impl EntityResolver {
    pub fn new(threshold: f32) -> Self {
        Self {
            encoder: MultimodalLink::new(),
            entity_signatures: HashMap::new(),
            match_threshold: threshold,
        }
    }

    pub fn register_entity(&mut self, entity: &MultimodalEntity) {
        let sig = self.encoder.encode_multimodal(entity);
        self.entity_signatures.insert(entity.id.clone(), sig);
    }

    pub fn find_matches(&self, entity: &MultimodalEntity) -> Vec<EntityMatch> {
        let query_hv = match self.entity_signatures.get(&entity.id) {
            Some(hv) => hv,
            None => return Vec::new(),
        };

        let mut matches = Vec::new();
        for (id, sig) in &self.entity_signatures {
            if id == &entity.id {
                continue;
            }

            let sim = query_hv.similarity(sig);
            if sim >= self.match_threshold {
                matches.push(EntityMatch {
                    entity_a: entity.id.clone(),
                    entity_b: id.clone(),
                    confidence: sim,
                    matching_modalities: vec![], // Could be expanded
                });
            }
        }

        matches.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        matches
    }

    pub fn entity_count(&self) -> usize {
        self.entity_signatures.len()
    }
}

impl Default for EntityResolver {
    fn default() -> Self {
        Self::new(0.7)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_text() -> TextData {
        TextData {
            content: "Hello world".to_string(),
            language: "en".to_string(),
            tokens: vec!["hello".to_string(), "world".to_string()],
        }
    }

    fn create_test_image() -> ImageData {
        ImageData {
            id: "img1".to_string(),
            width: 640,
            height: 480,
            color_histogram: vec![0.1, 0.2, 0.3, 0.4],
            edge_histogram: vec![0.5, 0.6, 0.7, 0.8],
            tags: vec!["nature".to_string(), "landscape".to_string()],
        }
    }

    #[test]
    fn test_text_encoding() {
        let mut encoder = MultimodalLink::new();
        let text = create_test_text();
        let hv = encoder.encode_text(&text);
        assert_eq!(hv.dimension(), DIMENSION);
    }

    #[test]
    fn test_image_encoding() {
        let mut encoder = MultimodalLink::new();
        let image = create_test_image();
        let hv = encoder.encode_image(&image);
        assert_eq!(hv.dimension(), DIMENSION);
    }

    #[test]
    fn test_multimodal_db() {
        let mut db = MultimodalDb::new();

        let entity = MultimodalEntity {
            id: "entity1".to_string(),
            text: Some(create_test_text()),
            image: Some(create_test_image()),
            audio: None,
            metadata: HashMap::new(),
        };

        db.insert(entity);
        assert_eq!(db.entity_count(), 1);
    }

    #[test]
    fn test_multimodal_similarity() {
        let mut db = MultimodalDb::new();

        // Add similar entities
        db.insert(MultimodalEntity {
            id: "e1".to_string(),
            text: Some(TextData {
                content: "cat".to_string(),
                language: "en".to_string(),
                tokens: vec!["cat".to_string()],
            }),
            image: None,
            audio: None,
            metadata: HashMap::new(),
        });

        db.insert(MultimodalEntity {
            id: "e2".to_string(),
            text: Some(TextData {
                content: "cat".to_string(),
                language: "en".to_string(),
                tokens: vec!["cat".to_string()],
            }),
            image: None,
            audio: None,
            metadata: HashMap::new(),
        });

        let similar = db.search_unified("e1", 0.5, 10);
        assert!(!similar.is_empty());
    }

    #[test]
    fn test_entity_resolver() {
        let mut resolver = EntityResolver::new(0.5);

        let entity = MultimodalEntity {
            id: "e1".to_string(),
            text: Some(create_test_text()),
            image: None,
            audio: None,
            metadata: HashMap::new(),
        };

        resolver.register_entity(&entity);
        assert_eq!(resolver.entity_count(), 1);
    }
}
