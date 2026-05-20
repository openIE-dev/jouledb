//! Reference energy baselines for calibrating the A-G energy label rating system.
//!
//! Provides per-size-class baselines (joules per token at FP16 on reference
//! hardware) and a database of known model energy profiles derived from
//! industry benchmark estimates.  The baselines serve as the "1.0x" reference
//! point that [`crate::energy_label::LabelThresholds`] classifies against.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::energy_label::{EnergyLabel, LabelThresholds};

// ---------------------------------------------------------------------------
// ModelSizeClass
// ---------------------------------------------------------------------------

/// Size classification for models, used to select the appropriate baseline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelSizeClass {
    /// < 10B parameters.
    Small,
    /// 10B -- 40B parameters.
    Medium,
    /// 40B -- 100B parameters.
    Large,
    /// 100B -- 250B parameters.
    XLarge,
    /// > 250B parameters.
    Frontier,
}

impl fmt::Display for ModelSizeClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Small => write!(f, "Small (<10B)"),
            Self::Medium => write!(f, "Medium (10B-40B)"),
            Self::Large => write!(f, "Large (40B-100B)"),
            Self::XLarge => write!(f, "XLarge (100B-250B)"),
            Self::Frontier => write!(f, "Frontier (>250B)"),
        }
    }
}

// ---------------------------------------------------------------------------
// EnergyBaseline
// ---------------------------------------------------------------------------

/// Reference energy baseline for a model size class.
///
/// The `joules_per_output_token_fp16` value is the "1.0x" reference used by
/// [`compute_energy_rating`] to produce the efficiency ratio that feeds into
/// the A-G label system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyBaseline {
    /// Which size class this baseline applies to.
    pub size_class: ModelSizeClass,
    /// Reference hardware description (e.g. "NVIDIA A100 80GB").
    pub reference_hardware: String,
    /// Baseline joules per output token at FP16 precision.
    pub joules_per_output_token_fp16: f64,
    /// Baseline joules per input token at FP16 precision (typically ~0.25x output).
    pub joules_per_input_token_fp16: f64,
    /// Human-readable description of this baseline.
    pub description: String,
}

// ---------------------------------------------------------------------------
// KnownModelEnergy
// ---------------------------------------------------------------------------

/// Known energy measurements for a specific model from published benchmarks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnownModelEnergy {
    /// Model identifier (e.g. "llama-3.1-70b").
    pub model_id: String,
    /// Parameter count.
    pub parameters: u64,
    /// Quantization level (e.g. "FP16", "FP8", "INT4").
    pub quantization: String,
    /// Hardware used for measurement.
    pub hardware: String,
    /// Joules per output token.
    pub joules_per_output_token: f64,
    /// Joules per input token.
    pub joules_per_input_token: f64,
    /// Citation or measurement origin.
    pub source: String,
    /// `true` = measured, `false` = estimated.
    pub measured: bool,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Classify a model by parameter count into a size class.
///
/// | Class     | Parameters     |
/// |-----------|---------------|
/// | Small     | < 10B         |
/// | Medium    | 10B -- 40B    |
/// | Large     | 40B -- 100B   |
/// | XLarge    | 100B -- 250B  |
/// | Frontier  | > 250B        |
pub fn classify_model_size(parameters: u64) -> ModelSizeClass {
    const B: u64 = 1_000_000_000;
    if parameters < 10 * B {
        ModelSizeClass::Small
    } else if parameters <= 40 * B {
        ModelSizeClass::Medium
    } else if parameters <= 100 * B {
        ModelSizeClass::Large
    } else if parameters <= 250 * B {
        ModelSizeClass::XLarge
    } else {
        ModelSizeClass::Frontier
    }
}

/// Get the reference baseline for a model size class.
///
/// Baselines are approximate FP16 values on reference hardware derived from
/// industry benchmark estimates (Hugging Face AI Energy Score, TokenPowerBench,
/// ML.ENERGY).
pub fn baseline_for_class(class: ModelSizeClass) -> EnergyBaseline {
    match class {
        ModelSizeClass::Small => EnergyBaseline {
            size_class: ModelSizeClass::Small,
            reference_hardware: "NVIDIA A100 80GB".to_string(),
            joules_per_output_token_fp16: 0.015,
            joules_per_input_token_fp16: 0.004,
            description: "Small models (<10B) at FP16 on A100".to_string(),
        },
        ModelSizeClass::Medium => EnergyBaseline {
            size_class: ModelSizeClass::Medium,
            reference_hardware: "NVIDIA A100 80GB".to_string(),
            joules_per_output_token_fp16: 0.06,
            joules_per_input_token_fp16: 0.015,
            description: "Medium models (10B-40B) at FP16 on A100".to_string(),
        },
        ModelSizeClass::Large => EnergyBaseline {
            size_class: ModelSizeClass::Large,
            reference_hardware: "4x NVIDIA A100 80GB".to_string(),
            joules_per_output_token_fp16: 0.15,
            joules_per_input_token_fp16: 0.038,
            description: "Large models (40B-100B) at FP16 on 4xA100".to_string(),
        },
        ModelSizeClass::XLarge => EnergyBaseline {
            size_class: ModelSizeClass::XLarge,
            reference_hardware: "8x NVIDIA H100 80GB".to_string(),
            joules_per_output_token_fp16: 0.50,
            joules_per_input_token_fp16: 0.125,
            description: "XLarge models (100B-250B) at FP16 on 8xH100".to_string(),
        },
        ModelSizeClass::Frontier => EnergyBaseline {
            size_class: ModelSizeClass::Frontier,
            reference_hardware: "8x NVIDIA H100 80GB".to_string(),
            joules_per_output_token_fp16: 1.0,
            joules_per_input_token_fp16: 0.25,
            description: "Frontier models (>250B) at FP16 on 8xH100".to_string(),
        },
    }
}

/// Compute the energy rating (A-G) for a model given its measured J/token
/// and parameter count.
///
/// 1. Classifies the model into a [`ModelSizeClass`] based on `parameters`.
/// 2. Looks up the reference baseline for that class.
/// 3. Computes `ratio = measured / baseline`.
/// 4. Classifies via [`LabelThresholds::default().classify(ratio)`].
///
/// Returns `(label, ratio)`.
pub fn compute_energy_rating(
    measured_joules_per_output_token: f64,
    parameters: u64,
) -> (EnergyLabel, f64) {
    let class = classify_model_size(parameters);
    let baseline = baseline_for_class(class);
    let ratio = if baseline.joules_per_output_token_fp16 == 0.0 {
        f64::MAX
    } else {
        measured_joules_per_output_token / baseline.joules_per_output_token_fp16
    };
    let label = LabelThresholds::default().classify(ratio);
    (label, ratio)
}

/// Look up known energy data for well-known models.
///
/// Returns `None` if `model_id` is not in the built-in database.
pub fn known_model_energy(model_id: &str) -> Option<KnownModelEnergy> {
    KNOWN_MODELS
        .iter()
        .find(|m| m.0 == model_id)
        .map(|m| KnownModelEnergy {
            model_id: m.0.to_string(),
            parameters: m.1,
            quantization: m.2.to_string(),
            hardware: m.3.to_string(),
            joules_per_output_token: m.4,
            joules_per_input_token: m.5,
            source: m.6.to_string(),
            measured: m.7,
        })
}

/// Returns all known model energy profiles.
pub fn all_known_models() -> Vec<KnownModelEnergy> {
    KNOWN_MODELS
        .iter()
        .map(|m| KnownModelEnergy {
            model_id: m.0.to_string(),
            parameters: m.1,
            quantization: m.2.to_string(),
            hardware: m.3.to_string(),
            joules_per_output_token: m.4,
            joules_per_input_token: m.5,
            source: m.6.to_string(),
            measured: m.7,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Known model database
// ---------------------------------------------------------------------------

/// Compact tuple representation for the static model table:
/// (model_id, parameters, quantization, hardware, j_per_output_token,
///  j_per_input_token, source, measured)
type ModelEntry = (
    &'static str, // model_id
    u64,          // parameters
    &'static str, // quantization
    &'static str, // hardware
    f64,          // joules_per_output_token
    f64,          // joules_per_input_token
    &'static str, // source
    bool,         // measured
);

static KNOWN_MODELS: &[ModelEntry] = &[
    // Llama family
    (
        "llama-3.1-8b",
        8_000_000_000,
        "FP16",
        "NVIDIA A100 80GB",
        0.015,
        0.004,
        "industry-benchmark-estimate",
        false,
    ),
    (
        "llama-3.1-70b",
        70_000_000_000,
        "FP16",
        "4x NVIDIA A100 80GB",
        0.12,
        0.030,
        "industry-benchmark-estimate",
        false,
    ),
    (
        "llama-3.1-405b",
        405_000_000_000,
        "FP16",
        "8x NVIDIA H100 80GB",
        0.85,
        0.213,
        "industry-benchmark-estimate",
        false,
    ),
    (
        "llama-3.3-70b",
        70_000_000_000,
        "FP16",
        "4x NVIDIA A100 80GB",
        0.12,
        0.030,
        "industry-benchmark-estimate",
        false,
    ),
    // Mistral family
    (
        "mistral-7b",
        7_000_000_000,
        "FP16",
        "NVIDIA A100 80GB",
        0.013,
        0.003,
        "industry-benchmark-estimate",
        false,
    ),
    (
        "mixtral-8x7b",
        46_000_000_000,
        "FP16",
        "NVIDIA A100 80GB",
        0.05,
        0.013,
        "industry-benchmark-estimate",
        false,
    ),
    // Qwen family
    (
        "qwen-2.5-72b",
        72_000_000_000,
        "FP16",
        "4x NVIDIA A100 80GB",
        0.13,
        0.033,
        "industry-benchmark-estimate",
        false,
    ),
    // DeepSeek
    (
        "deepseek-v3",
        671_000_000_000,
        "FP16",
        "8x NVIDIA H100 80GB",
        0.90,
        0.225,
        "industry-benchmark-estimate",
        false,
    ),
    // Proprietary class (estimates)
    (
        "gpt-4-class",
        200_000_000_000,
        "FP16",
        "8x NVIDIA H100 80GB",
        0.50,
        0.125,
        "industry-benchmark-estimate",
        false,
    ),
    (
        "claude-class",
        200_000_000_000,
        "FP16",
        "8x NVIDIA H100 80GB",
        0.45,
        0.113,
        "industry-benchmark-estimate",
        false,
    ),
];

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- classify_model_size ------------------------------------------------

    #[test]
    fn classify_small_model() {
        assert_eq!(classify_model_size(7_000_000_000), ModelSizeClass::Small);
    }

    #[test]
    fn classify_medium_model() {
        assert_eq!(classify_model_size(13_000_000_000), ModelSizeClass::Medium);
    }

    #[test]
    fn classify_large_model() {
        assert_eq!(classify_model_size(70_000_000_000), ModelSizeClass::Large);
    }

    #[test]
    fn classify_xlarge_model() {
        assert_eq!(classify_model_size(175_000_000_000), ModelSizeClass::XLarge);
    }

    #[test]
    fn classify_frontier_model() {
        assert_eq!(
            classify_model_size(405_000_000_000),
            ModelSizeClass::Frontier
        );
    }

    // -- baseline_for_class -------------------------------------------------

    #[test]
    fn baseline_small() {
        let b = baseline_for_class(ModelSizeClass::Small);
        assert!((b.joules_per_output_token_fp16 - 0.015).abs() < 1e-6);
    }

    #[test]
    fn baseline_monotonic() {
        let classes = [
            ModelSizeClass::Small,
            ModelSizeClass::Medium,
            ModelSizeClass::Large,
            ModelSizeClass::XLarge,
            ModelSizeClass::Frontier,
        ];
        for pair in classes.windows(2) {
            let lower = baseline_for_class(pair[0]);
            let upper = baseline_for_class(pair[1]);
            assert!(
                lower.joules_per_output_token_fp16 < upper.joules_per_output_token_fp16,
                "{} baseline ({}) should be less than {} baseline ({})",
                pair[0],
                lower.joules_per_output_token_fp16,
                pair[1],
                upper.joules_per_output_token_fp16,
            );
        }
    }

    // -- compute_energy_rating ----------------------------------------------

    #[test]
    fn rating_efficient_model() {
        // FP8 70B at 0.06 J/tok vs Large baseline 0.15 => ratio 0.40 => A
        let (label, ratio) = compute_energy_rating(0.06, 70_000_000_000);
        assert!(
            label == EnergyLabel::A || label == EnergyLabel::B,
            "Expected A or B, got {label} (ratio {ratio:.3})"
        );
    }

    #[test]
    fn rating_average_model() {
        // FP16 70B at 0.15 J/tok vs Large baseline 0.15 => ratio 1.0 => D
        let (label, ratio) = compute_energy_rating(0.15, 70_000_000_000);
        assert_eq!(label, EnergyLabel::D);
        assert!((ratio - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rating_inefficient_model() {
        // Overloaded 70B at 0.25 J/tok vs Large baseline 0.15 => ratio 1.67 => G
        let (label, ratio) = compute_energy_rating(0.25, 70_000_000_000);
        assert!(
            label == EnergyLabel::F || label == EnergyLabel::G,
            "Expected F or G, got {label} (ratio {ratio:.3})"
        );
    }

    // -- known_model_energy -------------------------------------------------

    #[test]
    fn known_model_llama() {
        let model = known_model_energy("llama-3.1-70b").expect("llama-3.1-70b should be known");
        assert_eq!(model.parameters, 70_000_000_000);
        assert!((model.joules_per_output_token - 0.12).abs() < 1e-6);
        assert_eq!(model.quantization, "FP16");
        assert!(!model.measured);
    }

    #[test]
    fn known_model_unknown() {
        assert!(known_model_energy("totally-made-up-model").is_none());
    }

    // -- serialization ------------------------------------------------------

    #[test]
    fn energy_baseline_serialization() {
        let baseline = baseline_for_class(ModelSizeClass::Large);
        let json = serde_json::to_string(&baseline).expect("serialize");
        let deserialized: EnergyBaseline = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.size_class, ModelSizeClass::Large);
        assert!(
            (deserialized.joules_per_output_token_fp16 - baseline.joules_per_output_token_fp16)
                .abs()
                < 1e-10
        );
        assert!(
            (deserialized.joules_per_input_token_fp16 - baseline.joules_per_input_token_fp16).abs()
                < 1e-10
        );
    }
}
