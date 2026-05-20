//! # JouleDB Energy
//!
//! Energy profiling and hardware-aware execution engine.
//! Makes JouleDB the world's first energy-aware database.
//!
//! ## Features
//!
//! - Real-time hardware energy measurement (power draw in watts)
//! - Per-query energy estimation (joules per operation)
//! - Hardware-aware execution hints (thermal/power/memory adaptive)
//! - Energy budgets with enforcement (Joule compiler integration point)
//! - Cross-platform: macOS IOKit (detailed), Linux sysfs, estimation fallback

pub mod advisor;
#[cfg(not(target_arch = "wasm32"))]
pub mod bridge;
pub mod budget;
pub mod controller;
pub mod monitor;
pub mod platform;
pub mod router;
pub mod tracker;

pub use advisor::{ExecutionHint, HardwareAdvisor};
#[cfg(not(target_arch = "wasm32"))]
pub use bridge::JouleDbAccountant;
pub use budget::{EnergyBudget, EnergyBudgetError};
pub use controller::{
    AdaptiveBaseline, AdaptiveController, AdaptiveParams, DynamicDispatchThresholds,
};
#[cfg(not(target_arch = "wasm32"))]
pub use monitor::EnergyMonitor;
pub use monitor::{EnergySnapshot, ThermalState};
pub use router::{ComputeRouter, GpuOperationKind, RoutingDecision};
pub use tracker::{
    AlgorithmType, DeviceTarget, EnergyObservation, OperationEnergyTracker, OperationType,
};

/// Auto-detected platform information for startup display and TDP calibration.
#[derive(Debug, Clone)]
pub struct PlatformInfo {
    /// CPU brand string (e.g., "Apple M3 Pro")
    pub cpu_brand: String,
    /// Estimated TDP in watts
    pub tdp_watts: f64,
    /// Whether the TDP was auto-detected or is a fallback
    pub tdp_source: &'static str,
    /// Whether a GPU is available
    pub gpu_available: bool,
    /// Whether an NPU (Neural Engine) is available
    pub npu_available: bool,
    /// Whether a TPU is available
    pub tpu_available: bool,
    /// Whether an LPU (Language Processing Unit) is available
    pub lpu_available: bool,
}

/// Detect the current platform's CPU, TDP, and accelerator availability.
///
/// Uses `sysinfo` for CPU identification, then matches known CPU families
/// to estimate TDP. Also queries the platform provider for GPU/NPU/TPU.
#[cfg(not(target_arch = "wasm32"))]
pub fn detect_platform() -> PlatformInfo {
    use sysinfo::System;

    let mut sys = System::new();
    sys.refresh_cpu_all();

    let cpu_brand = sys
        .cpus()
        .first()
        .map(|c| c.brand().trim().to_string())
        .unwrap_or_else(|| "Unknown CPU".to_string());

    let (tdp_watts, tdp_source) = estimate_tdp(&cpu_brand);

    // Create a temporary platform provider to check accelerator availability
    let provider = platform::create_provider(tdp_watts);
    let gpu_available = provider.gpu_available();
    let npu_available = provider.npu_available();
    let tpu_available = provider.tpu_available();
    let lpu_available = provider.lpu_available();

    PlatformInfo {
        cpu_brand,
        tdp_watts,
        tdp_source,
        gpu_available,
        npu_available,
        tpu_available,
        lpu_available,
    }
}

/// WASM stub: returns a fixed platform profile (no hardware introspection).
#[cfg(target_arch = "wasm32")]
pub fn detect_platform() -> PlatformInfo {
    PlatformInfo {
        cpu_brand: "WASM".to_string(),
        tdp_watts: jpb_core::constants::TDP_WASM_W,
        tdp_source: "wasm-estimate",
        gpu_available: false,
        npu_available: false,
        tpu_available: false,
        lpu_available: false,
    }
}

/// Estimate TDP from CPU brand string using known families.
/// Constants sourced from `jpb_core::constants::TDP_*`.
fn estimate_tdp(brand: &str) -> (f64, &'static str) {
    use jpb_core::constants;
    let upper = brand.to_uppercase();

    // Apple Silicon
    if upper.contains("APPLE") {
        if upper.contains("ULTRA") || upper.contains("MAX") {
            return (constants::TDP_APPLE_MAX_W, "auto-detected");
        }
        if upper.contains("PRO") {
            return (constants::TDP_APPLE_PRO_W, "auto-detected");
        }
        return (constants::TDP_APPLE_BASE_W, "auto-detected");
    }

    // Intel
    if upper.contains("INTEL") {
        if upper.contains("XEON") || upper.contains("I9") {
            return (constants::TDP_INTEL_HIGH_W, "auto-detected");
        }
        if upper.contains("I3") {
            return (constants::TDP_INTEL_I3_W, "auto-detected");
        }
        return (constants::TDP_INTEL_MID_W, "auto-detected");
    }

    // AMD
    if upper.contains("AMD") {
        if upper.contains("EPYC") {
            return (constants::TDP_AMD_EPYC_W, "auto-detected");
        }
        if upper.contains("RYZEN 9") {
            return (constants::TDP_AMD_RYZEN9_W, "auto-detected");
        }
        return (constants::TDP_AMD_RYZEN_MID_W, "auto-detected");
    }

    // ARM (non-Apple, e.g., Raspberry Pi, AWS Graviton)
    if upper.contains("ARM") || upper.contains("AARCH64") || upper.contains("GRAVITON") {
        if upper.contains("GRAVITON") {
            return (constants::TDP_GRAVITON_W, "auto-detected");
        }
        return (constants::TDP_ARM_GENERIC_W, "auto-detected");
    }

    // Fallback
    (constants::TDP_DEFAULT_W, "default")
}

/// Configuration for the energy subsystem
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EnergyConfig {
    /// Enable energy monitoring
    pub enabled: bool,
    /// Collection interval in milliseconds (default: 2000)
    pub collection_interval_ms: u64,
    /// Enable per-query energy tracking
    pub per_query_tracking: bool,
    /// Enable hardware advisor
    pub hardware_advisor_enabled: bool,
    /// Default TDP watts for estimation when battery unavailable (desktop/server)
    pub default_tdp_watts: f64,
    /// Thermal throttle threshold
    pub thermal_throttle_threshold: ThermalState,
    /// Memory pressure threshold (0.0 - 1.0)
    pub memory_pressure_threshold: f64,
    /// Power envelope threshold as fraction of TDP (0.0 - 1.0)
    pub power_envelope_threshold: f64,
    /// Per-query energy budget in joules (None = unlimited)
    pub default_budget_joules: Option<f64>,
}

impl Default for EnergyConfig {
    fn default() -> Self {
        use jpb_core::constants;
        Self {
            enabled: true,
            collection_interval_ms: constants::DEFAULT_COLLECTION_INTERVAL_MS,
            per_query_tracking: true,
            hardware_advisor_enabled: true,
            default_tdp_watts: constants::TDP_DEFAULT_W,
            thermal_throttle_threshold: ThermalState::Serious,
            memory_pressure_threshold: constants::MEMORY_PRESSURE_THRESHOLD,
            power_envelope_threshold: constants::POWER_ENVELOPE_THRESHOLD,
            default_budget_joules: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = EnergyConfig::default();
        assert!(config.enabled);
        assert_eq!(config.collection_interval_ms, 2000);
        assert_eq!(config.default_tdp_watts, 30.0);
        assert_eq!(config.thermal_throttle_threshold, ThermalState::Serious);
        assert!(config.default_budget_joules.is_none());
    }

    #[test]
    fn test_config_serde_roundtrip() {
        let config = EnergyConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let parsed: EnergyConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.default_tdp_watts, config.default_tdp_watts);
        assert_eq!(parsed.enabled, config.enabled);
    }

    #[test]
    fn test_detect_platform_returns_valid_info() {
        let info = detect_platform();
        assert!(!info.cpu_brand.is_empty());
        assert!(info.tdp_watts > 0.0);
    }

    #[test]
    fn test_estimate_tdp_apple_silicon() {
        assert_eq!(estimate_tdp("Apple M3 Pro").0, 30.0);
        assert_eq!(estimate_tdp("Apple M2 Max").0, 75.0);
        assert_eq!(estimate_tdp("Apple M4 Ultra").0, 75.0);
        assert_eq!(estimate_tdp("Apple M1").0, 20.0);
    }

    #[test]
    fn test_estimate_tdp_intel() {
        assert_eq!(estimate_tdp("Intel Xeon E5-2690").0, 125.0);
        assert_eq!(estimate_tdp("Intel Core i9-13900K").0, 125.0);
        assert_eq!(estimate_tdp("Intel Core i7-12700").0, 65.0);
        assert_eq!(estimate_tdp("Intel Core i5-12400").0, 65.0);
    }

    #[test]
    fn test_estimate_tdp_amd() {
        assert_eq!(estimate_tdp("AMD EPYC 7763").0, 200.0);
        assert_eq!(estimate_tdp("AMD Ryzen 9 7950X").0, 105.0);
        assert_eq!(estimate_tdp("AMD Ryzen 7 5800X").0, 65.0);
    }

    #[test]
    fn test_estimate_tdp_fallback() {
        let (tdp, source) = estimate_tdp("Unknown Processor XYZ");
        assert_eq!(tdp, 30.0);
        assert_eq!(source, "default");
    }
}
