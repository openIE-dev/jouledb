//! AI Energy Receipt — every AI operation is metered.

use serde::{Deserialize, Serialize};
use std::time::Duration;

use super::tier::InferenceTier;

/// Energy receipt for any AI operation within JouleDB.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiReceipt {
    /// Which tier actually served this request.
    pub tier: InferenceTier,
    /// Model/method identifier (e.g., "hdc-similarity", "onnx/all-MiniLM-L6", "claude-sonnet-4-6").
    pub model: String,
    /// Energy consumed in joules (measured or estimated).
    pub energy_joules: f64,
    /// How the energy value was obtained.
    pub provenance: EnergyProvenance,
    /// Wall-clock latency in microseconds.
    pub latency_us: u64,
    /// Tokens consumed (only meaningful for Tiers 3-4).
    pub tokens: Option<TokenCount>,
}

impl AiReceipt {
    /// Create a receipt for a holographic (Tier 1) operation.
    pub fn holographic(model: &str, energy_joules: f64, latency_us: u64) -> Self {
        Self {
            tier: InferenceTier::Holographic,
            model: model.to_string(),
            energy_joules,
            provenance: EnergyProvenance::Calculated,
            latency_us,
            tokens: None,
        }
    }

    /// Create a receipt for an embedded (Tier 2) operation.
    pub fn embedded(model: &str, energy_joules: f64, latency_us: u64) -> Self {
        Self {
            tier: InferenceTier::Embedded,
            model: model.to_string(),
            energy_joules,
            provenance: EnergyProvenance::Modeled,
            latency_us,
            tokens: None,
        }
    }

    /// Create a receipt for an API (Tier 4) operation.
    pub fn frontier(model: &str, energy_joules: f64, latency_us: u64, tokens: TokenCount) -> Self {
        Self {
            tier: InferenceTier::Frontier,
            model: model.to_string(),
            energy_joules,
            provenance: EnergyProvenance::ApiReported,
            latency_us,
            tokens: Some(tokens),
        }
    }
}

/// How the energy value was obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnergyProvenance {
    /// Hardware counter (RAPL, IOKit, hwmon)
    Measured,
    /// Deterministic from operation count (HDC hamming ops)
    Calculated,
    /// FLOPs × J/FLOP from benchmarks
    Modeled,
    /// Provider-reported energy
    ApiReported,
}

/// Token usage for LLM operations.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TokenCount {
    pub input: u32,
    pub output: u32,
}
