//! Edge database configuration
//!
//! Configuration options optimized for edge/embedded devices.

#[cfg(not(feature = "std"))]
use alloc::string::String;

use crate::accelerator::{AcceleratorType, detect_accelerators};

/// Edge database configuration
#[derive(Debug, Clone)]
pub struct EdgeConfig {
    /// Preferred accelerator type
    pub accelerator: AcceleratorType,
    /// Maximum memory usage in bytes (0 = auto)
    pub max_memory_bytes: usize,
    /// Page size for storage (should be aligned to device block size)
    pub page_size: usize,
    /// Number of pages to cache in memory
    pub cache_pages: usize,
    /// Enable write-ahead logging
    pub enable_wal: bool,
    /// WAL sync mode
    pub wal_sync_mode: WalSyncMode,
    /// Batch size for accelerated operations
    pub batch_size: usize,
    /// Power mode
    pub power_mode: PowerMode,
    /// Enable DMA for transfers (if supported)
    pub enable_dma: bool,
    /// Enable zero-copy operations (if supported)
    pub enable_zero_copy: bool,
}

/// WAL synchronization mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WalSyncMode {
    /// No explicit sync (rely on OS)
    None,
    /// Sync on commit
    OnCommit,
    /// Sync periodically (ms interval)
    Periodic(u32),
    /// Full sync on every write
    Full,
}

/// Power consumption mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerMode {
    /// Maximum performance, highest power
    Performance,
    /// Balanced performance/power
    Balanced,
    /// Low power, reduced performance
    LowPower,
    /// Ultra low power (minimal activity)
    UltraLowPower,
}

impl Default for EdgeConfig {
    fn default() -> Self {
        // Auto-detect hardware instead of assuming no accelerator.
        // Dynamic heterogeneous compute: always use the best available hardware.
        Self::auto_detect()
    }
}

impl EdgeConfig {
    /// Base config with no accelerator (used internally to avoid recursion in Default)
    fn base() -> Self {
        Self {
            accelerator: AcceleratorType::None,
            max_memory_bytes: 64 * 1024 * 1024, // 64MB default
            page_size: 4096,
            cache_pages: 1024,
            enable_wal: true,
            wal_sync_mode: WalSyncMode::OnCommit,
            batch_size: 256,
            power_mode: PowerMode::Balanced,
            enable_dma: true,
            enable_zero_copy: true,
        }
    }

    /// Create config with auto-detected accelerator
    pub fn auto_detect() -> Self {
        let accelerators = detect_accelerators();

        // Pick the best accelerator (first non-CPU)
        let accelerator = accelerators
            .iter()
            .find(|a| a.accelerator_type != AcceleratorType::None)
            .map(|a| a.accelerator_type)
            .unwrap_or(AcceleratorType::None);

        let mut config = Self::base();
        config.accelerator = accelerator;

        // Adjust settings based on accelerator
        if let Some(caps) = accelerators
            .iter()
            .find(|a| a.accelerator_type == accelerator)
        {
            config.batch_size = caps.max_batch_size.min(1024);
            config.enable_dma = caps.supports_dma;
            config.enable_zero_copy = caps.supports_zero_copy;
        }

        config
    }

    /// Create minimal config for constrained devices
    pub fn minimal() -> Self {
        Self {
            accelerator: AcceleratorType::None,
            max_memory_bytes: 8 * 1024 * 1024, // 8MB
            page_size: 4096,
            cache_pages: 128,
            enable_wal: true,
            wal_sync_mode: WalSyncMode::OnCommit,
            batch_size: 64,
            power_mode: PowerMode::LowPower,
            enable_dma: false,
            enable_zero_copy: false,
        }
    }

    /// Create config optimized for Jetson devices
    pub fn jetson() -> Self {
        Self {
            accelerator: AcceleratorType::Cuda,
            max_memory_bytes: 256 * 1024 * 1024, // 256MB
            page_size: 4096,
            cache_pages: 4096,
            enable_wal: true,
            wal_sync_mode: WalSyncMode::OnCommit,
            batch_size: 1024,
            power_mode: PowerMode::Performance,
            enable_dma: true,
            enable_zero_copy: true,
        }
    }

    /// Create config optimized for Coral devices
    pub fn coral() -> Self {
        Self {
            accelerator: AcceleratorType::EdgeTpu,
            max_memory_bytes: 128 * 1024 * 1024, // 128MB
            page_size: 4096,
            cache_pages: 2048,
            enable_wal: true,
            wal_sync_mode: WalSyncMode::OnCommit,
            batch_size: 512,
            power_mode: PowerMode::Balanced,
            enable_dma: true,
            enable_zero_copy: false,
        }
    }

    /// Create config optimized for Raspberry Pi 5
    pub fn raspberry_pi5() -> Self {
        Self {
            accelerator: AcceleratorType::VideoCore,
            max_memory_bytes: 512 * 1024 * 1024, // 512MB
            page_size: 4096,
            cache_pages: 8192,
            enable_wal: true,
            wal_sync_mode: WalSyncMode::OnCommit,
            batch_size: 512,
            power_mode: PowerMode::Balanced,
            enable_dma: true,
            enable_zero_copy: true,
        }
    }

    // =========================================================================
    // MCU / Bare-Metal / RTOS Profiles
    // =========================================================================

    /// Create config for Alif Ensemble E7 (dual Cortex-M55 + dual Ethos-U55)
    ///
    /// The E7 is the most powerful MCU-class AI chip with dual NPUs.
    /// - 4MB internal SRAM
    /// - Dual Ethos-U55 capable of ~1 TOPS combined
    /// - Excellent for vision + vibration analysis simultaneously
    pub fn alif_ensemble_e7() -> Self {
        Self {
            accelerator: AcceleratorType::EthosU55,
            max_memory_bytes: 2 * 1024 * 1024, // 2MB (leave room for ML models)
            page_size: 512,                    // Smaller pages for embedded flash
            cache_pages: 256,
            enable_wal: true,
            wal_sync_mode: WalSyncMode::Periodic(100), // Batch syncs for flash longevity
            batch_size: 64,
            power_mode: PowerMode::Balanced,
            enable_dma: true,
            enable_zero_copy: true, // Shared SRAM allows zero-copy
        }
    }

    /// Create config for Alif Balletto B1 (Cortex-M55 + Ethos-U55 + BLE)
    ///
    /// First BLE SoC with dedicated NPU. Optimized for wearables.
    /// - 1MB internal SRAM
    /// - Single Ethos-U55
    /// - Ultra-low power BLE
    pub fn alif_balletto_b1() -> Self {
        Self {
            accelerator: AcceleratorType::EthosU55,
            max_memory_bytes: 512 * 1024, // 512KB (very constrained)
            page_size: 256,               // Tiny pages for minimal memory
            cache_pages: 64,
            enable_wal: false, // Disable WAL to save memory/flash
            wal_sync_mode: WalSyncMode::None,
            batch_size: 16,
            power_mode: PowerMode::UltraLowPower,
            enable_dma: true,
            enable_zero_copy: true,
        }
    }

    /// Create config for STM32N6 (Cortex-M7 + Neural-ART NPU)
    ///
    /// First STM32 with hard NPU. Massive ecosystem support.
    /// - Supports 8-bit and 4-bit quantized weights
    /// - ~600 GOPS at INT8
    /// - Uses STM32 Cube.AI toolchain
    pub fn stm32n6() -> Self {
        Self {
            accelerator: AcceleratorType::NeuralArt,
            max_memory_bytes: 1 * 1024 * 1024, // 1MB
            page_size: 512,
            cache_pages: 128,
            enable_wal: true,
            wal_sync_mode: WalSyncMode::OnCommit,
            batch_size: 32,
            power_mode: PowerMode::Balanced,
            enable_dma: true,
            enable_zero_copy: false, // NPU has separate memory space
        }
    }

    /// Create config for Renesas RA8 (Cortex-M85 + Helium MVE)
    ///
    /// First commercial Cortex-M85. No discrete NPU - the CPU IS the accelerator.
    /// - ~4x performance of M7 for ML tasks
    /// - Uses standard CMSIS-NN
    /// - Excellent mid-tier option when NPU is overkill
    pub fn renesas_ra8() -> Self {
        Self {
            accelerator: AcceleratorType::Helium,
            max_memory_bytes: 1 * 1024 * 1024, // 1MB typical
            page_size: 512,
            cache_pages: 256,
            enable_wal: true,
            wal_sync_mode: WalSyncMode::OnCommit,
            batch_size: 128, // Helium can handle larger batches
            power_mode: PowerMode::Balanced,
            enable_dma: true,
            enable_zero_copy: true,
        }
    }

    /// Create config for ESP32-S3 (Xtensa LX7 + vector instructions)
    ///
    /// The "budget" choice for edge AI. Good Rust support (esp-rs).
    /// - Vector instructions for NN acceleration
    /// - WiFi + BLE built-in
    /// - 512KB SRAM typical
    pub fn esp32_s3() -> Self {
        Self {
            accelerator: AcceleratorType::Esp32S3,
            max_memory_bytes: 256 * 1024, // 256KB usable for DB
            page_size: 256,
            cache_pages: 64,
            enable_wal: true,
            wal_sync_mode: WalSyncMode::Periodic(500), // Batch writes for flash
            batch_size: 16,
            power_mode: PowerMode::LowPower,
            enable_dma: true,
            enable_zero_copy: false,
        }
    }

    /// Create config for Analog Devices MAX78002 (Cortex-M4 + CNN Accelerator)
    ///
    /// Architectural outlier: massive hardware CNN with small MCU attached.
    /// - Ultra-low power vision/audio
    /// - ~1.8 TOPS at 250mW
    /// - Harder to program but unbeatable battery life
    pub fn max78002() -> Self {
        Self {
            accelerator: AcceleratorType::MaximCnn,
            max_memory_bytes: 256 * 1024, // 256KB SRAM
            page_size: 256,
            cache_pages: 128,
            enable_wal: false, // CNN accelerator handles persistence differently
            wal_sync_mode: WalSyncMode::None,
            batch_size: 1, // CNN processes one frame at a time
            power_mode: PowerMode::UltraLowPower,
            enable_dma: true,
            enable_zero_copy: true, // CNN shares memory with M4
        }
    }

    /// Create config for NXP MCX N94x (dual Cortex-M33 + eIQ Neutron NPU)
    ///
    /// NXP's answer to edge AI. Uses NXP eIQ toolchain.
    /// - Good for anomaly detection and voice
    /// - ~250 GOPS
    pub fn nxp_mcx_n94x() -> Self {
        Self {
            accelerator: AcceleratorType::EiqNeutron,
            max_memory_bytes: 512 * 1024, // 512KB
            page_size: 512,
            cache_pages: 128,
            enable_wal: true,
            wal_sync_mode: WalSyncMode::OnCommit,
            batch_size: 32,
            power_mode: PowerMode::Balanced,
            enable_dma: true,
            enable_zero_copy: false,
        }
    }

    /// Create config for GreenWaves GAP9 (RISC-V + NE16 + 9-core cluster)
    ///
    /// Extremely specialized for hearables (TWS earbuds) and low-power vision.
    /// - Custom RISC-V with extensions
    /// - NE16 AI accelerator
    /// - Very difficult to program (custom toolchain)
    /// - Highest efficiency for audio/video processing
    pub fn greenwaves_gap9() -> Self {
        Self {
            accelerator: AcceleratorType::GapNe16,
            max_memory_bytes: 1 * 1024 * 1024, // 1MB
            page_size: 256,
            cache_pages: 256,
            enable_wal: false, // Often used without filesystem
            wal_sync_mode: WalSyncMode::None,
            batch_size: 64,
            power_mode: PowerMode::UltraLowPower,
            enable_dma: true,
            enable_zero_copy: true,
        }
    }

    /// Create config for Silicon Labs EFR32xG24 (Cortex-M33 + MVP)
    ///
    /// Designed for Zigbee/Matter/BLE endpoints.
    /// - Matrix Vector Processor (MVP) for simple models
    /// - Glass break detection, wake word, etc.
    pub fn silabs_efr32xg24() -> Self {
        Self {
            accelerator: AcceleratorType::SilabsMvp,
            max_memory_bytes: 128 * 1024, // 128KB
            page_size: 256,
            cache_pages: 32,
            enable_wal: false,
            wal_sync_mode: WalSyncMode::None,
            batch_size: 8,
            power_mode: PowerMode::UltraLowPower,
            enable_dma: false,
            enable_zero_copy: false,
        }
    }

    /// Create config for Infineon PSoC Edge E84 (Cortex-M55 + Ethos-U55)
    ///
    /// New competitor to Alif. Deep ModusToolbox integration.
    /// - Targets smart home and HMI
    pub fn infineon_psoc_edge() -> Self {
        Self {
            accelerator: AcceleratorType::EthosU55,
            max_memory_bytes: 1 * 1024 * 1024, // 1MB
            page_size: 512,
            cache_pages: 256,
            enable_wal: true,
            wal_sync_mode: WalSyncMode::OnCommit,
            batch_size: 64,
            power_mode: PowerMode::Balanced,
            enable_dma: true,
            enable_zero_copy: true,
        }
    }

    /// Create config for Ambiq Apollo510 (Cortex-M55 + Helium)
    ///
    /// Focused on smartwatches/wearables.
    /// - Sub-threshold power tech for extreme battery life
    /// - ~6 μA/MHz at 0.8V
    pub fn ambiq_apollo510() -> Self {
        Self {
            accelerator: AcceleratorType::Helium,
            max_memory_bytes: 512 * 1024, // 512KB
            page_size: 256,
            cache_pages: 64,
            enable_wal: false, // Save power
            wal_sync_mode: WalSyncMode::None,
            batch_size: 32,
            power_mode: PowerMode::UltraLowPower,
            enable_dma: true,
            enable_zero_copy: true,
        }
    }

    /// Create config for Syntiant NDP120 (Cortex-M0+ + Neural Decision Processor)
    ///
    /// Used for always-on voice and sensor fusion.
    /// - Neural Decision Processor (hardware DNN)
    /// - ~140 μW for always-on keyword spotting
    pub fn syntiant_ndp120() -> Self {
        Self {
            accelerator: AcceleratorType::SyntiantNdp,
            max_memory_bytes: 64 * 1024, // 64KB - very constrained
            page_size: 128,              // Minimal page size
            cache_pages: 16,
            enable_wal: false,
            wal_sync_mode: WalSyncMode::None,
            batch_size: 1, // Process one audio frame
            power_mode: PowerMode::UltraLowPower,
            enable_dma: false,
            enable_zero_copy: false,
        }
    }

    /// Create config for generic CMSIS-NN target (any Cortex-M)
    ///
    /// Safe default for any Arm Cortex-M with CMSIS-NN support.
    /// Adjust memory settings based on actual device.
    pub fn cmsis_nn_generic() -> Self {
        Self {
            accelerator: AcceleratorType::CmsisNn,
            max_memory_bytes: 256 * 1024, // Conservative default
            page_size: 256,
            cache_pages: 64,
            enable_wal: true,
            wal_sync_mode: WalSyncMode::OnCommit,
            batch_size: 16,
            power_mode: PowerMode::Balanced,
            enable_dma: false, // Not all M-class have DMA
            enable_zero_copy: false,
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), ConfigError> {
        // MCU class accelerators can use smaller page sizes (down to 64 bytes)
        let min_page_size = if self.accelerator.is_mcu_class() {
            64
        } else {
            512
        };

        if self.page_size < min_page_size {
            return Err(ConfigError::InvalidPageSize(self.page_size));
        }
        if !self.page_size.is_power_of_two() {
            return Err(ConfigError::InvalidPageSize(self.page_size));
        }
        if self.cache_pages == 0 {
            return Err(ConfigError::InvalidCacheSize(0));
        }
        if self.batch_size == 0 {
            return Err(ConfigError::InvalidBatchSize(0));
        }
        Ok(())
    }
}

/// Configuration error
#[derive(Debug, Clone)]
pub enum ConfigError {
    InvalidPageSize(usize),
    InvalidCacheSize(usize),
    InvalidBatchSize(usize),
    AcceleratorNotAvailable(AcceleratorType),
}

impl core::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidPageSize(s) => {
                write!(f, "Invalid page size: {} (must be power of 2 >= 512)", s)
            }
            Self::InvalidCacheSize(s) => write!(f, "Invalid cache size: {}", s),
            Self::InvalidBatchSize(s) => write!(f, "Invalid batch size: {}", s),
            Self::AcceleratorNotAvailable(t) => write!(f, "Accelerator not available: {:?}", t),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ConfigError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = EdgeConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_auto_detect() {
        let config = EdgeConfig::auto_detect();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_minimal_config() {
        let config = EdgeConfig::minimal();
        assert!(config.validate().is_ok());
        assert_eq!(config.max_memory_bytes, 8 * 1024 * 1024);
    }

    #[test]
    fn test_jetson_config() {
        let config = EdgeConfig::jetson();
        assert!(config.validate().is_ok());
        assert_eq!(config.accelerator, AcceleratorType::Cuda);
    }

    #[test]
    fn test_invalid_page_size() {
        let mut config = EdgeConfig::default();
        config.page_size = 100; // Not power of 2
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_cache_size() {
        let mut config = EdgeConfig::default();
        config.cache_pages = 0;
        assert!(config.validate().is_err());
    }

    // =========================================================================
    // MCU Profile Tests
    // =========================================================================

    #[test]
    fn test_alif_ensemble_e7() {
        let config = EdgeConfig::alif_ensemble_e7();
        assert!(config.validate().is_ok());
        assert_eq!(config.accelerator, AcceleratorType::EthosU55);
        assert!(config.accelerator.is_mcu_class());
        assert!(config.accelerator.uses_cmsis_nn());
    }

    #[test]
    fn test_alif_balletto_b1() {
        let config = EdgeConfig::alif_balletto_b1();
        assert!(config.validate().is_ok());
        assert_eq!(config.accelerator, AcceleratorType::EthosU55);
        assert!(!config.enable_wal); // Disabled for memory savings
    }

    #[test]
    fn test_stm32n6() {
        let config = EdgeConfig::stm32n6();
        assert!(config.validate().is_ok());
        assert_eq!(config.accelerator, AcceleratorType::NeuralArt);
        assert!(config.accelerator.is_mcu_class());
        assert!(config.accelerator.is_discrete_npu());
    }

    #[test]
    fn test_renesas_ra8() {
        let config = EdgeConfig::renesas_ra8();
        assert!(config.validate().is_ok());
        assert_eq!(config.accelerator, AcceleratorType::Helium);
        assert!(!config.accelerator.is_discrete_npu()); // Helium is CPU extension
    }

    #[test]
    fn test_esp32_s3() {
        let config = EdgeConfig::esp32_s3();
        assert!(config.validate().is_ok());
        assert_eq!(config.accelerator, AcceleratorType::Esp32S3);
        assert!(config.accelerator.is_mcu_class());
    }

    #[test]
    fn test_max78002() {
        let config = EdgeConfig::max78002();
        assert!(config.validate().is_ok());
        assert_eq!(config.accelerator, AcceleratorType::MaximCnn);
        assert_eq!(config.power_mode, PowerMode::UltraLowPower);
    }

    #[test]
    fn test_nxp_mcx_n94x() {
        let config = EdgeConfig::nxp_mcx_n94x();
        assert!(config.validate().is_ok());
        assert_eq!(config.accelerator, AcceleratorType::EiqNeutron);
    }

    #[test]
    fn test_greenwaves_gap9() {
        let config = EdgeConfig::greenwaves_gap9();
        assert!(config.validate().is_ok());
        assert_eq!(config.accelerator, AcceleratorType::GapNe16);
    }

    #[test]
    fn test_silabs_efr32xg24() {
        let config = EdgeConfig::silabs_efr32xg24();
        assert!(config.validate().is_ok());
        assert_eq!(config.accelerator, AcceleratorType::SilabsMvp);
        assert_eq!(config.power_mode, PowerMode::UltraLowPower);
    }

    #[test]
    fn test_infineon_psoc_edge() {
        let config = EdgeConfig::infineon_psoc_edge();
        assert!(config.validate().is_ok());
        assert_eq!(config.accelerator, AcceleratorType::EthosU55);
    }

    #[test]
    fn test_ambiq_apollo510() {
        let config = EdgeConfig::ambiq_apollo510();
        assert!(config.validate().is_ok());
        assert_eq!(config.accelerator, AcceleratorType::Helium);
        assert_eq!(config.power_mode, PowerMode::UltraLowPower);
    }

    #[test]
    fn test_syntiant_ndp120() {
        let config = EdgeConfig::syntiant_ndp120();
        assert!(config.validate().is_ok());
        assert_eq!(config.accelerator, AcceleratorType::SyntiantNdp);
        assert_eq!(config.max_memory_bytes, 64 * 1024); // Very constrained
    }

    #[test]
    fn test_cmsis_nn_generic() {
        let config = EdgeConfig::cmsis_nn_generic();
        assert!(config.validate().is_ok());
        assert_eq!(config.accelerator, AcceleratorType::CmsisNn);
        assert!(config.accelerator.uses_cmsis_nn());
    }

    #[test]
    fn test_accelerator_classification() {
        // MCU class
        assert!(AcceleratorType::EthosU55.is_mcu_class());
        assert!(AcceleratorType::Helium.is_mcu_class());
        assert!(AcceleratorType::Esp32S3.is_mcu_class());

        // MPU class
        assert!(AcceleratorType::Cuda.is_mpu_class());
        assert!(AcceleratorType::EdgeTpu.is_mpu_class());
        assert!(AcceleratorType::Hailo.is_mpu_class());

        // Neither (CPU fallback)
        assert!(!AcceleratorType::None.is_mcu_class());
        assert!(!AcceleratorType::None.is_mpu_class());
    }

    #[test]
    fn test_typical_tops() {
        // Verify some known TOPS values
        assert!(AcceleratorType::Hailo.typical_tops().unwrap() > 20.0);
        assert!(AcceleratorType::EthosU55.typical_tops().unwrap() < 1.0);
        assert!(AcceleratorType::EdgeTpu.typical_tops().unwrap() > 1.0);
    }
}
