//! Shared numeric constants used across the JoulesPerBit workspace.
//!
//! Centralizes magic numbers so they have one canonical definition.

// ── Confidence (basis points, 0–10_000) ──────────────────────────────────────

/// High confidence threshold: 70.00% in basis points.
pub const HIGH_CONFIDENCE: u16 = 7_000;

/// Moderate confidence threshold: 50.00% in basis points.
pub const MODERATE_CONFIDENCE: u16 = 5_000;

/// Low confidence threshold: 20.00% in basis points.
pub const LOW_CONFIDENCE: u16 = 2_000;

/// Maximum basis points (100.00%).
pub const MAX_BASIS_POINTS: u16 = 10_000;

// ── Energy grid intensity ────────────────────────────────────────────────────

/// Average US grid carbon intensity in grams CO₂ per kWh (EPA 2024).
pub const GRID_INTENSITY_G_PER_KWH: u16 = 453;

// ── Timeouts (milliseconds) ──────────────────────────────────────────────────

/// Default HTTP/LLM request timeout.
pub const DEFAULT_TIMEOUT_MS: u64 = 30_000;

/// Default energy collection interval.
pub const DEFAULT_COLLECTION_INTERVAL_MS: u64 = 2_000;

/// Default circuit-breaker cooldown.
pub const DEFAULT_CIRCUIT_COOLDOWN_SECS: u64 = 60;

/// Default circuit-breaker failure threshold.
pub const DEFAULT_CIRCUIT_FAILURE_THRESHOLD: u32 = 3;

// ── TDP defaults (watts) ─────────────────────────────────────────────────────

/// TDP for Apple Silicon base chips (M1/M2/M3/M4).
pub const TDP_APPLE_BASE_W: f64 = 20.0;

/// TDP for Apple Silicon Pro chips.
pub const TDP_APPLE_PRO_W: f64 = 30.0;

/// TDP for Apple Silicon Max/Ultra chips.
pub const TDP_APPLE_MAX_W: f64 = 75.0;

/// TDP for Intel Core i7/i5.
pub const TDP_INTEL_MID_W: f64 = 65.0;

/// TDP for Intel Core i3.
pub const TDP_INTEL_I3_W: f64 = 45.0;

/// TDP for Intel Xeon / i9.
pub const TDP_INTEL_HIGH_W: f64 = 125.0;

/// TDP for AMD EPYC.
pub const TDP_AMD_EPYC_W: f64 = 200.0;

/// TDP for AMD Ryzen 9.
pub const TDP_AMD_RYZEN9_W: f64 = 105.0;

/// TDP for AMD Ryzen 7/5.
pub const TDP_AMD_RYZEN_MID_W: f64 = 65.0;

/// TDP for AWS Graviton.
pub const TDP_GRAVITON_W: f64 = 100.0;

/// TDP for generic ARM boards.
pub const TDP_ARM_GENERIC_W: f64 = 5.0;

/// Default fallback TDP when CPU is unrecognized.
pub const TDP_DEFAULT_W: f64 = 30.0;

/// Default TDP for WASM target (no hardware introspection).
pub const TDP_WASM_W: f64 = 5.0;

// ── Energy thresholds ────────────────────────────────────────────────────────

/// Memory pressure threshold (fraction of total).
pub const MEMORY_PRESSURE_THRESHOLD: f64 = 0.7;

/// Power envelope threshold as fraction of TDP.
pub const POWER_ENVELOPE_THRESHOLD: f64 = 0.8;

/// Energy drift threshold in basis points (30%).
pub const ENERGY_DRIFT_THRESHOLD: u16 = 3_000;

/// Confidence decay threshold in basis points (20%).
pub const CONFIDENCE_DECAY_THRESHOLD: u16 = 2_000;

// ── Cache TTLs (seconds) ─────────────────────────────────────────────────────

/// Default live query cache TTL (1 hour).
pub const DEFAULT_CACHE_TTL_SECS: u64 = 3_600;

/// Default query-ID cache TTL (30 days).
pub const DEFAULT_QID_TTL_SECS: u64 = 2_592_000;

// ── Cascade / LLM defaults ───────────────────────────────────────────────────

/// Default per-token TDP for energy estimation (milliwatts).
pub const DEFAULT_TDP_MW: u32 = 15_000;

/// Complexity gate reactive threshold.
pub const GATE_REACTIVE_THRESHOLD: f64 = 0.70;

/// Complexity gate reject threshold.
pub const GATE_REJECT_THRESHOLD: f64 = 0.95;

// ── Domain energy constants (microjoules per operation) ──────────────────────

/// Minimum energy for any in-memory operation.
pub const MIN_OP_ENERGY_UJ: u64 = 1;

/// Estimated idle CPU power draw for in-memory ops (milliwatts).
pub const IDLE_CPU_POWER_MW: f64 = 2.0;

/// Default fixed energy per operation for simple CRUD stores.
pub const DEFAULT_CRUD_ENERGY_UJ: u64 = 12;

/// Default fixed energy per search/traversal operation.
pub const DEFAULT_SEARCH_ENERGY_UJ: u64 = 30;

/// Default fixed energy per checkout/complex operation.
pub const DEFAULT_COMPLEX_ENERGY_UJ: u64 = 95;

/// Default fixed energy per report/aggregation.
pub const DEFAULT_REPORT_ENERGY_UJ: u64 = 45;

// ── Agent / session limits ───────────────────────────────────────────────────

/// Default max agent steps per task.
pub const DEFAULT_MAX_AGENT_STEPS: u32 = 10;

/// Default max tokens per LLM step.
pub const DEFAULT_MAX_TOKENS_PER_STEP: u32 = 4_096;

/// Default agent step timeout.
pub const DEFAULT_STEP_TIMEOUT_MS: u64 = 30_000;

/// Default LLM temperature.
pub const DEFAULT_TEMPERATURE: f32 = 0.3;

/// Default sliding context window size (messages).
pub const DEFAULT_SLIDING_WINDOW: usize = 40;

// ── Network ──────────────────────────────────────────────────────────────────

/// Default bind address for local servers.
pub const DEFAULT_BIND_ADDRESS: &str = "127.0.0.1";

/// Default rate limit (requests per minute).
pub const DEFAULT_RATE_LIMIT_RPM: u32 = 60;

/// Default max request body size (1 MiB).
pub const DEFAULT_MAX_BODY_BYTES: usize = 1_048_576;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confidence_ordering() {
        assert!(LOW_CONFIDENCE < MODERATE_CONFIDENCE);
        assert!(MODERATE_CONFIDENCE < HIGH_CONFIDENCE);
        assert!(HIGH_CONFIDENCE < MAX_BASIS_POINTS);
    }

    #[test]
    fn tdp_values_positive() {
        assert!(TDP_APPLE_BASE_W > 0.0);
        assert!(TDP_INTEL_HIGH_W > TDP_INTEL_MID_W);
        assert!(TDP_AMD_EPYC_W > TDP_AMD_RYZEN9_W);
    }

    #[test]
    fn energy_constants_non_zero() {
        assert!(MIN_OP_ENERGY_UJ > 0);
        assert!(DEFAULT_CRUD_ENERGY_UJ > 0);
        assert!(DEFAULT_COMPLEX_ENERGY_UJ > DEFAULT_CRUD_ENERGY_UJ);
    }
}
