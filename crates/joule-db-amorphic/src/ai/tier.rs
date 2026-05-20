//! Inference Tiers — from sub-microsecond holograms to frontier LLMs.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// The four inference tiers, ordered by cost (cheapest first).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum InferenceTier {
    /// Pure HDC: similarity, resonator, attention-SDM. Always available.
    /// Sub-microsecond. ~0.2 µJ per op. Works in WASM/IoT/browser.
    Holographic = 1,
    /// On-device models (ONNX/tract). Feature-gated.
    /// 1-50ms. ~0.5-5 mJ. Runs on edge/NPU.
    Embedded = 2,
    /// Local LLM (joule-train-infer / Ollama). Feature-gated.
    /// 100ms-10s. ~0.1-5 J. Needs GPU.
    Local = 3,
    /// Cloud API (verity-llm, 18 providers). Feature-gated.
    /// 200ms-30s. API cost. Anywhere with internet.
    Frontier = 4,
}

/// Constraints that guide tier auto-selection.
#[derive(Debug, Clone)]
pub struct TierConstraints {
    /// Maximum latency tolerated. None = no constraint.
    pub max_latency: Option<Duration>,
    /// Maximum energy for this operation (joules). None = no constraint.
    pub max_energy_joules: Option<f64>,
    /// Minimum acceptable confidence (0.0-1.0). Higher = may need stronger tier.
    pub min_confidence: f32,
    /// Explicitly allow only these tiers. None = all available.
    pub allowed_tiers: Option<Vec<InferenceTier>>,
    /// Prefer cheapest tier that meets constraints (default: true).
    pub prefer_cheapest: bool,
}

impl Default for TierConstraints {
    fn default() -> Self {
        Self {
            max_latency: None,
            max_energy_joules: None,
            min_confidence: 0.0,
            allowed_tiers: None,
            prefer_cheapest: true,
        }
    }
}

impl TierConstraints {
    pub fn with_max_latency(mut self, d: Duration) -> Self {
        self.max_latency = Some(d);
        self
    }

    pub fn with_max_energy(mut self, joules: f64) -> Self {
        self.max_energy_joules = Some(joules);
        self
    }

    pub fn with_min_confidence(mut self, c: f32) -> Self {
        self.min_confidence = c;
        self
    }

    pub fn only_tiers(mut self, tiers: Vec<InferenceTier>) -> Self {
        self.allowed_tiers = Some(tiers);
        self
    }

    /// Convenience: force Tier 1 only (sub-microsecond, zero cost).
    pub fn holographic_only() -> Self {
        Self::default().only_tiers(vec![InferenceTier::Holographic])
    }

    /// Convenience: allow up to local LLM but not cloud API.
    pub fn no_cloud() -> Self {
        Self::default().only_tiers(vec![
            InferenceTier::Holographic,
            InferenceTier::Embedded,
            InferenceTier::Local,
        ])
    }
}
