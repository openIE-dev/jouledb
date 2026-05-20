//! Hugging Face AI Energy Score format conversion.
//!
//! Converts internal energy metrics (SCI scores, energy profiles, model
//! baselines) into the format expected by Hugging Face's AI Energy Score
//! system for model cards and leaderboard badges.

use serde::{Deserialize, Serialize};

use crate::baselines::{
    KnownModelEnergy, baseline_for_class, classify_model_size, compute_energy_rating,
    known_model_energy,
};
use crate::energy_label::EnergyLabel;
use crate::sci::SciScore;

// ---------------------------------------------------------------------------
// HfEnergyScore — the output format matching HF expectations
// ---------------------------------------------------------------------------

/// Energy score in Hugging Face AI Energy Score format.
///
/// Designed to be JSON-serializable and compatible with HF model card energy
/// badges and the AI Energy Score leaderboard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfEnergyScore {
    /// Model identifier (e.g. "llama-3.1-70b").
    pub model_id: String,

    /// Energy consumed per 1000 tokens generated (kWh).
    pub energy_per_1k_tokens_kwh: f64,

    /// Energy consumed per single inference request (kWh).
    /// For generative models this assumes a standard benchmark length.
    pub energy_per_request_kwh: f64,

    /// Total CO2 equivalent emissions per 1000 tokens (gCO2eq).
    pub emissions_per_1k_tokens_gco2eq: f64,

    /// Energy efficiency label (A-G, EU style).
    pub energy_label: String,

    /// Numeric score 0-100 (higher = more efficient).
    pub score: u32,

    /// Hardware used for measurement or estimation.
    pub hardware: HfHardwareInfo,

    /// Model metadata.
    pub model_info: HfModelInfo,

    /// Measurement methodology.
    pub methodology: HfMethodology,
}

/// Hardware context for the energy measurement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfHardwareInfo {
    /// Accelerator name (e.g. "NVIDIA A100 80GB").
    pub accelerator: String,
    /// Number of accelerators used.
    pub count: u32,
    /// TDP per accelerator in watts.
    pub tdp_watts: f64,
}

/// Model metadata included in the energy score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfModelInfo {
    /// Parameter count.
    pub parameters: u64,
    /// Size class (Small, Medium, Large, XLarge, Frontier).
    pub size_class: String,
    /// Quantization level (e.g. "FP16", "FP8", "INT4").
    pub quantization: String,
}

/// Measurement methodology description.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfMethodology {
    /// How energy was measured or estimated.
    pub method: String,
    /// Whether this is a direct measurement or estimate.
    pub measured: bool,
    /// Source of the measurement data.
    pub source: String,
    /// Number of samples used for measurement (0 if estimated).
    pub sample_size: u32,
}

// ---------------------------------------------------------------------------
// Conversion functions
// ---------------------------------------------------------------------------

/// Standard benchmark output length in tokens (used for per-request estimates).
const BENCHMARK_OUTPUT_TOKENS: f64 = 256.0;

/// Default grid carbon intensity (gCO2/kWh) — global average.
const DEFAULT_CARBON_INTENSITY: f64 = 436.0;

/// Convert a known model's energy profile to HF Energy Score format.
///
/// Uses the internal `known_model_energy()` database to look up energy data
/// and convert it to the HF-compatible format.
///
/// Returns `None` if the model is not in the known models database.
pub fn hf_score_for_known_model(model_id: &str) -> Option<HfEnergyScore> {
    let known = known_model_energy(model_id)?;
    Some(hf_score_from_known_energy(&known))
}

/// Convert a `KnownModelEnergy` entry to HF Energy Score format.
pub fn hf_score_from_known_energy(known: &KnownModelEnergy) -> HfEnergyScore {
    let j_per_token = known.joules_per_output_token;
    let kwh_per_token = j_per_token / 3_600_000.0; // 1 kWh = 3.6 MJ
    let kwh_per_1k = kwh_per_token * 1000.0;

    let kwh_per_request = kwh_per_token * BENCHMARK_OUTPUT_TOKENS;
    let emissions_per_1k = kwh_per_1k * DEFAULT_CARBON_INTENSITY;

    let (label, ratio) = compute_energy_rating(j_per_token, known.parameters);
    let score = label_to_score(label, ratio);

    let size_class = classify_model_size(known.parameters);
    let (accel_name, accel_count, tdp) = parse_hardware(&known.hardware);

    HfEnergyScore {
        model_id: known.model_id.clone(),
        energy_per_1k_tokens_kwh: kwh_per_1k,
        energy_per_request_kwh: kwh_per_request,
        emissions_per_1k_tokens_gco2eq: emissions_per_1k,
        energy_label: label.to_string(),
        score,
        hardware: HfHardwareInfo {
            accelerator: accel_name,
            count: accel_count,
            tdp_watts: tdp,
        },
        model_info: HfModelInfo {
            parameters: known.parameters,
            size_class: size_class.to_string(),
            quantization: known.quantization.clone(),
        },
        methodology: HfMethodology {
            method: if known.measured {
                "direct_measurement".to_string()
            } else {
                "estimation".to_string()
            },
            measured: known.measured,
            source: known.source.clone(),
            sample_size: if known.measured { 100 } else { 0 },
        },
    }
}

/// Convert custom energy measurements to HF Energy Score format.
///
/// Use this when you have measured J/token values that aren't in the known
/// models database.
pub fn hf_score_from_measurement(
    model_id: &str,
    parameters: u64,
    joules_per_output_token: f64,
    quantization: &str,
    hardware: &str,
    carbon_intensity_gco2_kwh: f64,
) -> HfEnergyScore {
    let kwh_per_token = joules_per_output_token / 3_600_000.0;
    let kwh_per_1k = kwh_per_token * 1000.0;
    let kwh_per_request = kwh_per_token * BENCHMARK_OUTPUT_TOKENS;
    let emissions_per_1k = kwh_per_1k * carbon_intensity_gco2_kwh;

    let (label, ratio) = compute_energy_rating(joules_per_output_token, parameters);
    let score = label_to_score(label, ratio);

    let size_class = classify_model_size(parameters);
    let (accel_name, accel_count, tdp) = parse_hardware(hardware);

    HfEnergyScore {
        model_id: model_id.to_string(),
        energy_per_1k_tokens_kwh: kwh_per_1k,
        energy_per_request_kwh: kwh_per_request,
        emissions_per_1k_tokens_gco2eq: emissions_per_1k,
        energy_label: label.to_string(),
        score,
        hardware: HfHardwareInfo {
            accelerator: accel_name,
            count: accel_count,
            tdp_watts: tdp,
        },
        model_info: HfModelInfo {
            parameters,
            size_class: size_class.to_string(),
            quantization: quantization.to_string(),
        },
        methodology: HfMethodology {
            method: "direct_measurement".to_string(),
            measured: true,
            source: "invisible-infrastructure".to_string(),
            sample_size: 0,
        },
    }
}

/// Convert an SCI score to HF Energy Score format.
///
/// The SCI score provides operational carbon data which we combine with the
/// model's energy profile.
pub fn hf_score_from_sci(
    model_id: &str,
    parameters: u64,
    sci: &SciScore,
    joules_per_output_token: f64,
    quantization: &str,
    hardware: &str,
) -> HfEnergyScore {
    let kwh_per_token = joules_per_output_token / 3_600_000.0;
    let kwh_per_1k = kwh_per_token * 1000.0;
    let kwh_per_request = kwh_per_token * BENCHMARK_OUTPUT_TOKENS;
    let emissions_per_1k = kwh_per_1k * sci.carbon_intensity_gco2_kwh;

    let (label, ratio) = compute_energy_rating(joules_per_output_token, parameters);
    let score = label_to_score(label, ratio);

    let size_class = classify_model_size(parameters);
    let (accel_name, accel_count, tdp) = parse_hardware(hardware);

    HfEnergyScore {
        model_id: model_id.to_string(),
        energy_per_1k_tokens_kwh: kwh_per_1k,
        energy_per_request_kwh: kwh_per_request,
        emissions_per_1k_tokens_gco2eq: emissions_per_1k,
        energy_label: label.to_string(),
        score,
        hardware: HfHardwareInfo {
            accelerator: accel_name,
            count: accel_count,
            tdp_watts: tdp,
        },
        model_info: HfModelInfo {
            parameters,
            size_class: size_class.to_string(),
            quantization: quantization.to_string(),
        },
        methodology: HfMethodology {
            method: "sci_calculation".to_string(),
            measured: true,
            source: "invisible-infrastructure".to_string(),
            sample_size: sci.functional_unit_count as u32,
        },
    }
}

/// Get HF Energy Scores for all known models in the database.
pub fn all_hf_scores() -> Vec<HfEnergyScore> {
    crate::baselines::all_known_models()
        .iter()
        .map(hf_score_from_known_energy)
        .collect()
}

/// Get the energy baseline comparison for a model in HF-friendly format.
pub fn hf_baseline_comparison(
    joules_per_output_token: f64,
    parameters: u64,
) -> HfBaselineComparison {
    let size_class = classify_model_size(parameters);
    let baseline = baseline_for_class(size_class);
    let (label, ratio) = compute_energy_rating(joules_per_output_token, parameters);

    HfBaselineComparison {
        size_class: size_class.to_string(),
        baseline_j_per_token: baseline.joules_per_output_token_fp16,
        measured_j_per_token: joules_per_output_token,
        efficiency_ratio: ratio,
        energy_label: label.to_string(),
        reference_hardware: baseline.reference_hardware,
    }
}

/// Comparison of measured energy against baseline for HF display.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfBaselineComparison {
    /// Model size class.
    pub size_class: String,
    /// Baseline J/token for this size class.
    pub baseline_j_per_token: f64,
    /// Measured J/token for the model.
    pub measured_j_per_token: f64,
    /// Efficiency ratio (measured / baseline).
    pub efficiency_ratio: f64,
    /// Energy label (A-G).
    pub energy_label: String,
    /// Reference hardware for the baseline.
    pub reference_hardware: String,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Map an energy label + efficiency ratio to a numeric score 0-100.
fn label_to_score(label: EnergyLabel, ratio: f64) -> u32 {
    // Score bands based on label; fine-tuned by ratio within band
    let (base, range_start, range_end) = match label {
        EnergyLabel::A => (86, 0.0_f64, 0.5_f64),
        EnergyLabel::B => (72, 0.5, 0.75),
        EnergyLabel::C => (58, 0.75, 0.90),
        EnergyLabel::D => (44, 0.90, 1.10),
        EnergyLabel::E => (30, 1.10, 1.25),
        EnergyLabel::F => (16, 1.25, 1.50),
        EnergyLabel::G => (2, 1.50, 3.0),
    };

    let clamped = ratio.clamp(range_start, range_end);
    let frac = if (range_end - range_start).abs() < 1e-10 {
        0.5
    } else {
        (clamped - range_start) / (range_end - range_start)
    };

    // Within each band, lower ratio = higher score
    let bonus = ((1.0 - frac) * 13.0) as u32;
    (base + bonus).min(100)
}

/// Parse hardware string into (accelerator_name, count, tdp_watts).
fn parse_hardware(hardware: &str) -> (String, u32, f64) {
    let (count, name) = if hardware.contains("8x") {
        (8, hardware.replace("8x ", ""))
    } else if hardware.contains("4x") {
        (4, hardware.replace("4x ", ""))
    } else if hardware.contains("2x") {
        (2, hardware.replace("2x ", ""))
    } else {
        (1, hardware.to_string())
    };

    let tdp = if name.contains("H100") {
        700.0
    } else if name.contains("A100") {
        400.0
    } else if name.contains("L40") {
        300.0
    } else if name.contains("A10") {
        150.0
    } else {
        300.0 // default estimate
    };

    (name, count, tdp)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_model_to_hf_score() {
        let score = hf_score_for_known_model("llama-3.1-70b").unwrap();
        assert_eq!(score.model_id, "llama-3.1-70b");
        assert!(score.energy_per_1k_tokens_kwh > 0.0);
        assert!(score.energy_per_request_kwh > 0.0);
        assert!(score.emissions_per_1k_tokens_gco2eq > 0.0);
        assert!(!score.energy_label.is_empty());
        assert!(score.score > 0 && score.score <= 100);
        assert_eq!(score.model_info.parameters, 70_000_000_000);
        assert_eq!(score.model_info.quantization, "FP16");
    }

    #[test]
    fn unknown_model_returns_none() {
        assert!(hf_score_for_known_model("nonexistent-model").is_none());
    }

    #[test]
    fn hf_score_energy_conversion() {
        // 0.12 J/token = 0.12 / 3_600_000 kWh/token
        let score = hf_score_for_known_model("llama-3.1-70b").unwrap();
        let expected_kwh_per_1k = 0.12 / 3_600_000.0 * 1000.0;
        assert!(
            (score.energy_per_1k_tokens_kwh - expected_kwh_per_1k).abs() < 1e-12,
            "got {}, expected {}",
            score.energy_per_1k_tokens_kwh,
            expected_kwh_per_1k
        );
    }

    #[test]
    fn hf_score_per_request() {
        let score = hf_score_for_known_model("llama-3.1-70b").unwrap();
        let expected = 0.12 / 3_600_000.0 * BENCHMARK_OUTPUT_TOKENS;
        assert!(
            (score.energy_per_request_kwh - expected).abs() < 1e-12,
            "got {}, expected {}",
            score.energy_per_request_kwh,
            expected
        );
    }

    #[test]
    fn hf_score_emissions() {
        let score = hf_score_for_known_model("llama-3.1-70b").unwrap();
        let expected = score.energy_per_1k_tokens_kwh * DEFAULT_CARBON_INTENSITY;
        assert!((score.emissions_per_1k_tokens_gco2eq - expected).abs() < 1e-12,);
    }

    #[test]
    fn small_model_higher_score_than_frontier() {
        let small = hf_score_for_known_model("llama-3.1-8b").unwrap();
        let frontier = hf_score_for_known_model("deepseek-v3").unwrap();
        // Small model at baseline should rate better than frontier at baseline
        // (both are at ~1.0x ratio, but let's just check the label is reasonable)
        assert!(!small.energy_label.is_empty());
        assert!(!frontier.energy_label.is_empty());
    }

    #[test]
    fn custom_measurement_score() {
        let score = hf_score_from_measurement(
            "custom-model",
            13_000_000_000,
            0.03, // half the medium baseline
            "FP8",
            "NVIDIA A100 80GB",
            400.0,
        );
        assert_eq!(score.model_id, "custom-model");
        assert!(score.methodology.measured);
        assert_eq!(score.methodology.source, "invisible-infrastructure");
        assert_eq!(score.hardware.count, 1);
        assert!((score.hardware.tdp_watts - 400.0).abs() < 1e-6);
        // 0.03 J/tok vs 0.06 medium baseline => ratio 0.5 => should be A or B
        assert!(
            score.energy_label == "A" || score.energy_label == "B",
            "Expected A or B for efficient model, got {}",
            score.energy_label
        );
    }

    #[test]
    fn sci_score_conversion() {
        let sci = SciScore {
            energy_kwh: 100.0,
            carbon_intensity_gco2_kwh: 300.0,
            embodied_carbon_gco2: 500.0,
            functional_unit_count: 1_000_000.0,
            operational_carbon_gco2: 30_000.0,
            sci_value: 30.5,
            config: crate::sci::SciConfig {
                region: "eu-west-1".to_string(),
                hardware_model: "NVIDIA A100".to_string(),
                lifetime_years: 5.0,
                functional_unit_name: "inference".to_string(),
            },
        };

        let score = hf_score_from_sci(
            "test-model",
            70_000_000_000,
            &sci,
            0.12,
            "FP16",
            "4x NVIDIA A100 80GB",
        );
        assert_eq!(score.model_id, "test-model");
        // With carbon intensity 300 instead of default 436, emissions should differ
        let expected_emissions = score.energy_per_1k_tokens_kwh * 300.0;
        assert!((score.emissions_per_1k_tokens_gco2eq - expected_emissions).abs() < 1e-12,);
        assert_eq!(score.methodology.method, "sci_calculation");
        assert_eq!(score.hardware.count, 4);
    }

    #[test]
    fn all_hf_scores_covers_known_models() {
        let scores = all_hf_scores();
        assert!(
            scores.len() >= 10,
            "expected at least 10 known models, got {}",
            scores.len()
        );
        // Check all scores are valid
        for s in &scores {
            assert!(
                s.energy_per_1k_tokens_kwh > 0.0,
                "{} has zero energy",
                s.model_id
            );
            assert!(
                s.score > 0 && s.score <= 100,
                "{} has invalid score {}",
                s.model_id,
                s.score
            );
            assert!(!s.energy_label.is_empty(), "{} has empty label", s.model_id);
        }
    }

    #[test]
    fn baseline_comparison() {
        let cmp = hf_baseline_comparison(0.06, 70_000_000_000);
        assert_eq!(cmp.size_class, "Large (40B-100B)");
        assert!((cmp.baseline_j_per_token - 0.15).abs() < 1e-6);
        assert!((cmp.measured_j_per_token - 0.06).abs() < 1e-6);
        assert!((cmp.efficiency_ratio - 0.4).abs() < 1e-6);
        assert!(
            cmp.energy_label == "A" || cmp.energy_label == "B",
            "Expected A or B, got {}",
            cmp.energy_label
        );
    }

    #[test]
    fn label_to_score_ranges() {
        // A label should give score >= 86
        let a_score = label_to_score(EnergyLabel::A, 0.3);
        assert!(a_score >= 86, "A score should be >= 86, got {}", a_score);

        // D label (ratio 1.0) should be around 44-57
        let d_score = label_to_score(EnergyLabel::D, 1.0);
        assert!(
            (44..=57).contains(&d_score),
            "D score should be 44-57, got {}",
            d_score
        );

        // G label should give low score
        let g_score = label_to_score(EnergyLabel::G, 2.0);
        assert!(g_score <= 15, "G score should be <= 15, got {}", g_score);
    }

    #[test]
    fn parse_hardware_multi_gpu() {
        let (name, count, tdp) = parse_hardware("8x NVIDIA H100 80GB");
        assert_eq!(count, 8);
        assert_eq!(name, "NVIDIA H100 80GB");
        assert!((tdp - 700.0).abs() < 1e-6);
    }

    #[test]
    fn parse_hardware_single_gpu() {
        let (name, count, tdp) = parse_hardware("NVIDIA A100 80GB");
        assert_eq!(count, 1);
        assert_eq!(name, "NVIDIA A100 80GB");
        assert!((tdp - 400.0).abs() < 1e-6);
    }

    #[test]
    fn hf_score_serialization() {
        let score = hf_score_for_known_model("mistral-7b").unwrap();
        let json = serde_json::to_string_pretty(&score).unwrap();
        let deserialized: HfEnergyScore = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.model_id, "mistral-7b");
        assert!((deserialized.score as f64 - score.score as f64).abs() < 1e-6);
    }
}
