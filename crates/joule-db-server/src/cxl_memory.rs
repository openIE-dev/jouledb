//! CXL Memory Disaggregation Stub
//!
//! Types and trait definitions for future CXL 3.1 memory pooling support.
//! No runtime implementation yet — requires CXL-capable hardware
//! (Intel Sapphire Rapids+, AMD Genoa+).

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ============================================================================
// Errors
// ============================================================================

#[derive(Debug, Error)]
pub enum CxlMemoryError {
    #[error("CXL memory not available on this hardware")]
    NotAvailable,

    #[error("out of CXL memory: requested {requested} bytes, available {available} bytes")]
    OutOfMemory { requested: u64, available: u64 },

    #[error("invalid allocation id: {0}")]
    InvalidAllocation(u64),

    #[error("CXL hardware error: {0}")]
    HardwareError(String),
}

// ============================================================================
// Memory tiers
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CxlMemoryTier {
    /// Local DRAM — lowest latency, highest bandwidth
    Hot,
    /// CXL-attached memory — moderate latency, high bandwidth
    Warm,
    /// NVMe / persistent memory — highest latency, lowest power
    Cold,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CxlTierConfig {
    pub tier: CxlMemoryTier,
    pub capacity_bytes: u64,
    pub bandwidth_gbps: f64,
    pub latency_ns: u64,
}

// ============================================================================
// Configuration
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CxlMemoryConfig {
    pub enabled: bool,
    pub tiers: Vec<CxlTierConfig>,
    pub auto_tiering: bool,
}

impl Default for CxlMemoryConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            tiers: vec![
                CxlTierConfig {
                    tier: CxlMemoryTier::Hot,
                    capacity_bytes: 0,
                    bandwidth_gbps: 0.0,
                    latency_ns: 0,
                },
                CxlTierConfig {
                    tier: CxlMemoryTier::Warm,
                    capacity_bytes: 0,
                    bandwidth_gbps: 0.0,
                    latency_ns: 0,
                },
                CxlTierConfig {
                    tier: CxlMemoryTier::Cold,
                    capacity_bytes: 0,
                    bandwidth_gbps: 0.0,
                    latency_ns: 0,
                },
            ],
            auto_tiering: false,
        }
    }
}

// ============================================================================
// Tier stats
// ============================================================================

#[derive(Debug, Clone, Serialize)]
pub struct CxlTierStats {
    pub tier: CxlMemoryTier,
    pub capacity_bytes: u64,
    pub used_bytes: u64,
    pub allocation_count: u64,
}

// ============================================================================
// Backend trait (future implementation)
// ============================================================================

/// Trait for CXL memory backends. Implementations will be provided
/// when CXL 3.1 hardware becomes available for testing.
pub trait CxlMemoryBackend: Send + Sync {
    fn allocate(&self, size: u64, tier: CxlMemoryTier) -> Result<u64, CxlMemoryError>;
    fn deallocate(&self, allocation_id: u64) -> Result<(), CxlMemoryError>;
    fn read(&self, allocation_id: u64, offset: u64, len: u64) -> Result<Vec<u8>, CxlMemoryError>;
    fn write(&self, allocation_id: u64, offset: u64, data: &[u8]) -> Result<(), CxlMemoryError>;
    fn tier_stats(&self) -> Vec<CxlTierStats>;
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_disabled() {
        let config = CxlMemoryConfig::default();
        assert!(!config.enabled);
        assert!(!config.auto_tiering);
        assert_eq!(config.tiers.len(), 3);
    }

    #[test]
    fn test_tier_config_creation() {
        let tier = CxlTierConfig {
            tier: CxlMemoryTier::Warm,
            capacity_bytes: 64 * 1024 * 1024 * 1024, // 64 GB
            bandwidth_gbps: 36.0,
            latency_ns: 170,
        };
        assert_eq!(tier.tier, CxlMemoryTier::Warm);
        assert_eq!(tier.latency_ns, 170);
    }

    #[test]
    fn test_error_display() {
        let e = CxlMemoryError::NotAvailable;
        assert!(e.to_string().contains("not available"));

        let e = CxlMemoryError::OutOfMemory {
            requested: 1024,
            available: 512,
        };
        assert!(e.to_string().contains("1024"));
        assert!(e.to_string().contains("512"));

        let e = CxlMemoryError::InvalidAllocation(42);
        assert!(e.to_string().contains("42"));

        let e = CxlMemoryError::HardwareError("link degraded".into());
        assert!(e.to_string().contains("link degraded"));
    }
}
