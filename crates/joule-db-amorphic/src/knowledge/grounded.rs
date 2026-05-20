//! Grounded Input: Encode from reality, not just text.
//!
//! Flaw 17 in transformers: representations are statistical patterns in text,
//! not connected to anything real. "Hot" is a token embedding, not a temperature.
//!
//! Grounded input provides Encode from reality:
//! - **Audio**: frequency spectrum → BinaryHV (via SigQL MFCC/STFT)
//! - **Image**: spatial frequency → BinaryHV (via MediaQL DCT)
//! - **Sensor**: numeric readings → BinaryHV (via threshold encoding)
//! - **Structured data**: JSON/CSV → BinaryHV (via field binding)
//!
//! Each modality produces a BinaryHV that lives in the same algebra as
//! text concepts. Cross-modal similarity works natively.

use crate::BinaryHV;
use super::concept::KNOWLEDGE_DIM;

/// A grounded input from any modality.
#[derive(Clone, Debug)]
pub struct GroundedInput {
    /// The holographic encoding.
    pub vector: BinaryHV,
    /// What modality this came from.
    pub modality: Modality,
    /// Human-readable description.
    pub description: String,
    /// Raw numeric features (before encoding).
    pub features: Vec<f32>,
}

/// Input modality.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Modality {
    /// Natural language text.
    Text,
    /// Audio signal (frequency features).
    Audio,
    /// Image (spatial frequency features).
    Image,
    /// Numeric sensor readings.
    Sensor,
    /// Structured data (JSON, CSV, key-value).
    Structured,
}

/// Trait for anything that can be encoded into the holographic algebra.
/// This is the generalized Encode primitive — not just for text.
pub trait Groundable {
    /// Encode into a BinaryHV that lives in the same space as all other concepts.
    fn ground(&self, dim: usize) -> GroundedInput;
}

/// Encode numeric features into BinaryHV using thermometer/threshold encoding.
///
/// Each feature value is compared against N thresholds. Each threshold
/// comparison produces 1 bit. This preserves magnitude relationships:
/// similar values → similar vectors.
pub fn encode_numeric(features: &[f32], dim: usize, seed: u64) -> BinaryHV {
    if features.is_empty() {
        return BinaryHV::zeros(dim);
    }

    let bits_per_feature = dim / features.len();
    if bits_per_feature == 0 {
        return BinaryHV::from_embedding(features, dim, seed);
    }

    let num_words = (dim + 63) / 64;
    let mut words = vec![0u64; num_words];

    for (f_idx, &value) in features.iter().enumerate() {
        let base_bit = f_idx * bits_per_feature;

        // Thermometer encoding: set bits up to the value level
        // Normalize value to [0, 1] range using sigmoid
        let normalized = 1.0 / (1.0 + (-value).exp());
        let threshold_bits = (normalized * bits_per_feature as f32) as usize;

        for b in 0..threshold_bits.min(bits_per_feature) {
            let bit_idx = base_bit + b;
            if bit_idx < dim {
                words[bit_idx / 64] |= 1u64 << (bit_idx % 64);
            }
        }
    }

    BinaryHV::from_words(words, dim)
}

/// Encode a key-value map into BinaryHV using field binding.
/// Each field: field_name_hv ⊗ field_value_hv. All fields bundled.
pub fn encode_structured(
    fields: &[(&str, &str)],
    dim: usize,
    seed: u64,
) -> BinaryHV {
    if fields.is_empty() {
        return BinaryHV::zeros(dim);
    }

    let mut acc = joule_db_hdc::BundleAccumulator::new(dim);

    for (i, (key, value)) in fields.iter().enumerate() {
        let key_hv = BinaryHV::from_data(key.as_bytes(), dim);
        let value_hv = BinaryHV::from_data(value.as_bytes(), dim);
        let field_hv = key_hv.bind(&value_hv);
        acc.add(&field_hv);
    }

    acc.threshold()
}

/// Encode audio features (e.g., MFCC coefficients) into BinaryHV.
/// MFCCs are typically 13-40 float values per frame.
pub fn encode_audio_frame(mfcc_coefficients: &[f32], dim: usize) -> BinaryHV {
    encode_numeric(mfcc_coefficients, dim, 0xA0_D10_F8A_0E000)
}

/// Encode image features (e.g., DCT coefficients) into BinaryHV.
pub fn encode_image_patch(dct_coefficients: &[f32], dim: usize) -> BinaryHV {
    encode_numeric(dct_coefficients, dim, 0x10_A6E_9A7_C0000)
}

/// Encode sensor readings into BinaryHV.
/// Preserves magnitude: similar readings → similar vectors.
pub fn encode_sensor(readings: &[f32], dim: usize) -> BinaryHV {
    encode_numeric(readings, dim, 0x5E_450_8DA_7A000)
}

/// Multi-modal fusion: combine encodings from different modalities
/// into a single representation. Uses binding to preserve each modality's
/// contribution while creating a unified vector.
pub fn fuse_modalities(inputs: &[GroundedInput]) -> Option<BinaryHV> {
    if inputs.is_empty() {
        return None;
    }

    let dim = inputs[0].vector.dimension();
    let mut acc = joule_db_hdc::BundleAccumulator::new(dim);

    for (i, input) in inputs.iter().enumerate() {
        // Permute by modality index to distinguish contributions
        let permuted = input.vector.permute(i * 100);
        acc.add(&permuted);
    }

    Some(acc.threshold())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numeric_encoding_similar_values_similar_vectors() {
        let a = encode_numeric(&[1.0, 2.0, 3.0], 10000, 42);
        let b = encode_numeric(&[1.1, 2.1, 3.1], 10000, 42);
        let c = encode_numeric(&[10.0, 20.0, 30.0], 10000, 42);

        let sim_ab = a.similarity(&b);
        let sim_ac = a.similarity(&c);

        assert!(
            sim_ab > sim_ac,
            "similar values should produce similar vectors: ab={sim_ab}, ac={sim_ac}"
        );
    }

    #[test]
    fn test_structured_encoding() {
        let fields = [("name", "dog"), ("type", "animal"), ("sound", "bark")];
        let hv = encode_structured(&fields, 10000, 42);
        assert_eq!(hv.dimension(), 10000);

        // Different structure should be different
        let fields2 = [("name", "car"), ("type", "vehicle"), ("fuel", "gas")];
        let hv2 = encode_structured(&fields2, 10000, 42);

        let sim = hv.similarity(&hv2);
        assert!(sim < 0.6, "different structures should differ: {sim}");
    }

    #[test]
    fn test_audio_encoding() {
        // Simulated MFCC: 13 coefficients
        let mfcc = [1.0, -0.5, 0.3, 0.1, -0.2, 0.4, -0.1, 0.2, -0.3, 0.1, 0.0, -0.1, 0.05];
        let hv = encode_audio_frame(&mfcc, 10000);
        assert_eq!(hv.dimension(), 10000);
    }

    #[test]
    fn test_sensor_encoding() {
        let readings = [22.5, 65.0, 1013.25]; // temp, humidity, pressure
        let hv = encode_sensor(&readings, 10000);
        assert_eq!(hv.dimension(), 10000);
    }

    #[test]
    fn test_cross_modal_similarity() {
        // A concept encoded as text and as sensor data should be in the same space
        let text_hv = BinaryHV::from_data(b"hot", 10000);
        let sensor_hv = encode_sensor(&[100.0, 0.0, 0.0], 10000); // high temperature

        // They won't be similar (different encodings) but they're in the same space
        // — they can be bound, compared, bundled
        let bound = text_hv.bind(&sensor_hv);
        assert_eq!(bound.dimension(), 10000);
    }

    #[test]
    fn test_multimodal_fusion() {
        let text_input = GroundedInput {
            vector: BinaryHV::from_data(b"dog barking", 10000),
            modality: Modality::Text,
            description: "dog barking".to_string(),
            features: vec![],
        };

        let audio_input = GroundedInput {
            vector: encode_audio_frame(&[1.0, -0.5, 0.3, 0.1, -0.2, 0.4, -0.1, 0.2, -0.3, 0.1, 0.0, -0.1, 0.05], 10000),
            modality: Modality::Audio,
            description: "bark sound".to_string(),
            features: vec![1.0, -0.5, 0.3],
        };

        let fused = fuse_modalities(&[text_input.clone(), audio_input.clone()]);
        assert!(fused.is_some());

        let fused_hv = fused.unwrap();
        // Fused should be somewhat similar to both inputs
        let sim_text = fused_hv.similarity(&text_input.vector);
        let sim_audio = fused_hv.similarity(&audio_input.vector);
        assert!(sim_text > 0.4, "fused should relate to text: {sim_text}");
        assert!(sim_audio > 0.4, "fused should relate to audio: {sim_audio}");
    }

    #[test]
    fn test_empty_inputs() {
        let empty_numeric = encode_numeric(&[], 10000, 42);
        assert_eq!(empty_numeric.dimension(), 10000);

        let empty_structured = encode_structured(&[], 10000, 42);
        assert_eq!(empty_structured.dimension(), 10000);

        let empty_fusion = fuse_modalities(&[]);
        assert!(empty_fusion.is_none());
    }
}
