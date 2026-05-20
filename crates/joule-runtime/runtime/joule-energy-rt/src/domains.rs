//! Multi-Domain Hardware Energy Telemetry
//!
//! Extends kinematic energy tracking to all measurable hardware domains:
//! - CPU (package, cores, uncore)
//! - GPU (discrete and integrated)
//! - NPU/Neural Engine
//! - TPU (Google Tensor Processing Unit)
//! - DRAM/Memory
//! - Storage (SSD/NVMe)
//! - Network (NIC)
//!
//! Each domain provides full derivative chains (energy → power → jerk → snap → crackle → pop)
//! enabling comprehensive system-wide energy analysis.
//!
//! # Measurement Sources by Platform
//!
//! | Domain | Linux | macOS | Windows |
//! |--------|-------|-------|---------|
//! | CPU | RAPL | IOReport | EMI |
//! | GPU | NVML/ROCm | IOReport | NVML |
//! | NPU | N/A | IOReport | N/A |
//! | DRAM | RAPL | IOReport | EMI |
//! | Storage | Estimated | IOReport | Estimated |
//! | Network | Estimated | Estimated | Estimated |
//!
//! # Example
//!
//! ```ignore
//! use joule_energy_rt::domains::{MultiDomainMonitor, HardwareDomain};
//!
//! let mut monitor = MultiDomainMonitor::new()?;
//! monitor.start()?;
//!
//! // ... computation ...
//!
//! let telemetry = monitor.stop()?;
//!
//! // Access per-domain metrics
//! if let Some(cpu) = telemetry.get_domain(HardwareDomain::Cpu) {
//!     println!("CPU Power: {:.2} W", cpu.energy.power_w);
//!     println!("CPU Power Jerk: {:.3} W/s²", cpu.energy.power_jerk_w_per_s2);
//! }
//!
//! if let Some(gpu) = telemetry.get_domain(HardwareDomain::Gpu) {
//!     println!("GPU Power: {:.2} W", gpu.energy.power_w);
//! }
//!
//! // System-wide aggregate
//! println!("Total System Power: {:.2} W", telemetry.total_power_w());
//! ```

use crate::error::{Error, Result};
use crate::kinematics::{
    DerivativeComputer, EfficiencyMetrics, EnergyDerivatives, KinematicConfig, Sample,
    ThermalDerivatives, ThermodynamicCoupling,
};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

// ============================================================================
// Hardware Domains
// ============================================================================

/// Hardware domain for energy measurement
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HardwareDomain {
    /// CPU package (all cores + uncore)
    Cpu,
    /// Individual CPU core
    CpuCore(u32),
    /// CPU uncore (cache, memory controller)
    CpuUncore,

    /// Discrete GPU
    Gpu,
    /// Specific GPU by index
    GpuDevice(u32),
    /// Integrated GPU (shares power with CPU)
    IntegratedGpu,

    /// Neural Processing Unit (Apple ANE, Intel NPU, etc.)
    Npu,

    /// Tensor Processing Unit (Google TPU)
    Tpu,

    /// DRAM / System Memory
    Dram,
    /// Specific DIMM/channel
    DramChannel(u32),

    /// Storage subsystem
    Storage,
    /// Specific storage device
    StorageDevice(u32),

    /// Network interface
    Network,
    /// Specific NIC
    NetworkDevice(u32),

    /// Platform/SoC total (Apple Silicon unified)
    Platform,

    /// Intel XPU (Arc, Ponte Vecchio)
    IntelXpu,
    /// Specific Intel XPU
    IntelXpuDevice(u32),
    /// Intel Gaudi (HLML)
    IntelGaudi,
    /// Specific Gaudi device
    IntelGaudiDevice(u32),
    /// Groq LPU
    GroqLpu,
    /// AWS Neuron (Trainium/Inferentia)
    AwsNeuron,
    /// Specific Neuron device
    AwsNeuronDevice(u32),
    /// Cerebras WSE
    CerebrasWse,
    /// SambaNova RDU
    SambaNovaRdu,

    /// Custom/other domain
    Custom(u32),
}

impl HardwareDomain {
    /// Get human-readable name for the domain
    pub fn name(&self) -> String {
        match self {
            Self::Cpu => "CPU".to_string(),
            Self::CpuCore(n) => format!("CPU Core {}", n),
            Self::CpuUncore => "CPU Uncore".to_string(),
            Self::Gpu => "GPU".to_string(),
            Self::GpuDevice(n) => format!("GPU {}", n),
            Self::IntegratedGpu => "Integrated GPU".to_string(),
            Self::Npu => "NPU".to_string(),
            Self::Tpu => "TPU".to_string(),
            Self::Dram => "DRAM".to_string(),
            Self::DramChannel(n) => format!("DRAM Channel {}", n),
            Self::Storage => "Storage".to_string(),
            Self::StorageDevice(n) => format!("Storage Device {}", n),
            Self::Network => "Network".to_string(),
            Self::NetworkDevice(n) => format!("NIC {}", n),
            Self::Platform => "Platform".to_string(),
            Self::IntelXpu => "Intel XPU".to_string(),
            Self::IntelXpuDevice(n) => format!("Intel XPU {}", n),
            Self::IntelGaudi => "Intel Gaudi".to_string(),
            Self::IntelGaudiDevice(n) => format!("Intel Gaudi {}", n),
            Self::GroqLpu => "Groq LPU".to_string(),
            Self::AwsNeuron => "AWS Neuron".to_string(),
            Self::AwsNeuronDevice(n) => format!("AWS Neuron {}", n),
            Self::CerebrasWse => "Cerebras WSE".to_string(),
            Self::SambaNovaRdu => "SambaNova RDU".to_string(),
            Self::Custom(n) => format!("Custom Domain {}", n),
        }
    }

    /// Get short identifier
    pub fn id(&self) -> String {
        match self {
            Self::Cpu => "cpu".to_string(),
            Self::CpuCore(n) => format!("cpu{}", n),
            Self::CpuUncore => "uncore".to_string(),
            Self::Gpu => "gpu".to_string(),
            Self::GpuDevice(n) => format!("gpu{}", n),
            Self::IntegratedGpu => "igpu".to_string(),
            Self::Npu => "npu".to_string(),
            Self::Tpu => "tpu".to_string(),
            Self::Dram => "dram".to_string(),
            Self::DramChannel(n) => format!("dram{}", n),
            Self::Storage => "storage".to_string(),
            Self::StorageDevice(n) => format!("storage{}", n),
            Self::Network => "network".to_string(),
            Self::NetworkDevice(n) => format!("nic{}", n),
            Self::Platform => "platform".to_string(),
            Self::IntelXpu => "intel_xpu".to_string(),
            Self::IntelXpuDevice(n) => format!("intel_xpu{}", n),
            Self::IntelGaudi => "intel_gaudi".to_string(),
            Self::IntelGaudiDevice(n) => format!("intel_gaudi{}", n),
            Self::GroqLpu => "groq_lpu".to_string(),
            Self::AwsNeuron => "aws_neuron".to_string(),
            Self::AwsNeuronDevice(n) => format!("aws_neuron{}", n),
            Self::CerebrasWse => "cerebras_wse".to_string(),
            Self::SambaNovaRdu => "sambanova_rdu".to_string(),
            Self::Custom(n) => format!("custom{}", n),
        }
    }

    /// Check if this is a compute domain (CPU, GPU, NPU, TPU, accelerators)
    pub fn is_compute(&self) -> bool {
        matches!(
            self,
            Self::Cpu
                | Self::CpuCore(_)
                | Self::CpuUncore
                | Self::Gpu
                | Self::GpuDevice(_)
                | Self::IntegratedGpu
                | Self::Npu
                | Self::Tpu
                | Self::IntelXpu
                | Self::IntelXpuDevice(_)
                | Self::IntelGaudi
                | Self::IntelGaudiDevice(_)
                | Self::GroqLpu
                | Self::AwsNeuron
                | Self::AwsNeuronDevice(_)
                | Self::CerebrasWse
                | Self::SambaNovaRdu
        )
    }

    /// Check if this is a memory domain
    pub fn is_memory(&self) -> bool {
        matches!(self, Self::Dram | Self::DramChannel(_))
    }

    /// Check if this is an I/O domain
    pub fn is_io(&self) -> bool {
        matches!(
            self,
            Self::Storage | Self::StorageDevice(_) | Self::Network | Self::NetworkDevice(_)
        )
    }
}

// ============================================================================
// Domain Sample
// ============================================================================

/// A timestamped sample for a specific hardware domain
#[derive(Debug, Clone)]
pub struct DomainSample {
    /// Timestamp (seconds from start)
    pub time_s: f64,
    /// Energy reading (Joules)
    pub energy_j: f64,
    /// Temperature (Celsius), if available
    pub temp_c: Option<f64>,
    /// Utilization (0.0-1.0), if available
    pub utilization: Option<f64>,
    /// Frequency (Hz), if available
    pub frequency_hz: Option<f64>,
    /// Bandwidth (bytes/s), if available
    pub bandwidth_bytes_per_s: Option<f64>,
}

impl DomainSample {
    /// Create a basic sample with just energy
    pub fn new(time_s: f64, energy_j: f64) -> Self {
        Self {
            time_s,
            energy_j,
            temp_c: None,
            utilization: None,
            frequency_hz: None,
            bandwidth_bytes_per_s: None,
        }
    }

    /// Add temperature
    pub fn with_temp(mut self, temp_c: f64) -> Self {
        self.temp_c = Some(temp_c);
        self
    }

    /// Add utilization
    pub fn with_utilization(mut self, util: f64) -> Self {
        self.utilization = Some(util.clamp(0.0, 1.0));
        self
    }

    /// Add frequency
    pub fn with_frequency(mut self, freq_hz: f64) -> Self {
        self.frequency_hz = Some(freq_hz);
        self
    }

    /// Add bandwidth
    pub fn with_bandwidth(mut self, bw: f64) -> Self {
        self.bandwidth_bytes_per_s = Some(bw);
        self
    }
}

// ============================================================================
// Domain Reader Trait
// ============================================================================

/// Trait for reading energy from a specific hardware domain
pub trait DomainReader: Send + Sync {
    /// Get the hardware domain this reader measures
    fn domain(&self) -> HardwareDomain;

    /// Check if this domain is available on the current system
    fn is_available(&self) -> bool;

    /// Read current energy in Joules
    fn read_energy(&self) -> Result<f64>;

    /// Read current temperature in Celsius (optional)
    fn read_temperature(&self) -> Result<Option<f64>> {
        Ok(None)
    }

    /// Read current utilization 0.0-1.0 (optional)
    fn read_utilization(&self) -> Result<Option<f64>> {
        Ok(None)
    }

    /// Read current frequency in Hz (optional)
    fn read_frequency(&self) -> Result<Option<f64>> {
        Ok(None)
    }

    /// Read current bandwidth in bytes/s (optional)
    fn read_bandwidth(&self) -> Result<Option<f64>> {
        Ok(None)
    }

    /// Take a complete sample
    fn sample(&self, time_s: f64, base_energy: f64) -> Result<DomainSample> {
        let energy = self.read_energy()? - base_energy;
        let mut sample = DomainSample::new(time_s, energy);

        if let Ok(Some(temp)) = self.read_temperature() {
            sample.temp_c = Some(temp);
        }
        if let Ok(Some(util)) = self.read_utilization() {
            sample.utilization = Some(util);
        }
        if let Ok(Some(freq)) = self.read_frequency() {
            sample.frequency_hz = Some(freq);
        }
        if let Ok(Some(bw)) = self.read_bandwidth() {
            sample.bandwidth_bytes_per_s = Some(bw);
        }

        Ok(sample)
    }
}

// ============================================================================
// Domain Telemetry
// ============================================================================

/// Complete kinematic telemetry for a single hardware domain
#[derive(Debug, Clone, Default)]
pub struct DomainTelemetry {
    /// The hardware domain
    pub domain: Option<HardwareDomain>,

    /// Duration of measurement
    pub duration_s: f64,

    /// Number of samples
    pub sample_count: usize,

    /// Energy derivatives (power kinematics)
    pub energy: EnergyDerivatives,

    /// Thermal derivatives
    pub thermal: ThermalDerivatives,

    /// Thermodynamic coupling
    pub thermodynamics: ThermodynamicCoupling,

    /// Utilization statistics
    pub utilization: UtilizationStats,

    /// Frequency statistics
    pub frequency: FrequencyStats,

    /// Bandwidth statistics
    pub bandwidth: BandwidthStats,
}

/// Utilization statistics for a domain
#[derive(Debug, Clone, Copy, Default)]
pub struct UtilizationStats {
    /// Minimum utilization (0.0-1.0)
    pub min: f64,
    /// Maximum utilization
    pub max: f64,
    /// Mean utilization
    pub mean: f64,
    /// Time-weighted average (accounts for varying sample intervals)
    pub time_weighted_mean: f64,
}

/// Frequency statistics for a domain
#[derive(Debug, Clone, Copy, Default)]
pub struct FrequencyStats {
    /// Minimum frequency (Hz)
    pub min_hz: f64,
    /// Maximum frequency (Hz)
    pub max_hz: f64,
    /// Mean frequency (Hz)
    pub mean_hz: f64,
    /// Energy-weighted average frequency
    pub energy_weighted_mean_hz: f64,
}

/// Bandwidth statistics for a domain
#[derive(Debug, Clone, Copy, Default)]
pub struct BandwidthStats {
    /// Peak bandwidth (bytes/s)
    pub peak_bytes_per_s: f64,
    /// Mean bandwidth (bytes/s)
    pub mean_bytes_per_s: f64,
    /// Total bytes transferred
    pub total_bytes: f64,
    /// Energy per byte (J/B)
    pub energy_per_byte_j: f64,
}

// ============================================================================
// Multi-Domain Telemetry
// ============================================================================

/// Complete telemetry across all hardware domains
#[derive(Debug, Clone, Default)]
pub struct MultiDomainTelemetry {
    /// Total duration
    pub duration_s: f64,

    /// Per-domain telemetry
    pub domains: HashMap<HardwareDomain, DomainTelemetry>,

    /// System-wide aggregate
    pub system: SystemAggregate,

    /// Cross-domain correlations
    pub correlations: DomainCorrelations,
}

/// System-wide aggregate metrics
#[derive(Debug, Clone, Default)]
pub struct SystemAggregate {
    /// Total system energy (J)
    pub total_energy_j: f64,

    /// Total system power (W) - sum of all domains
    pub total_power_w: f64,

    /// Peak system power (W)
    pub peak_power_w: f64,

    /// Minimum system power (W)
    pub min_power_w: f64,

    /// Power breakdown by domain type
    pub compute_power_w: f64,
    pub memory_power_w: f64,
    pub io_power_w: f64,

    /// Efficiency metrics
    pub efficiency: EfficiencyMetrics,
}

/// Cross-domain correlations and relationships
#[derive(Debug, Clone, Default)]
pub struct DomainCorrelations {
    /// CPU-GPU power correlation coefficient (-1 to 1)
    pub cpu_gpu_correlation: Option<f64>,

    /// CPU-DRAM power correlation
    pub cpu_dram_correlation: Option<f64>,

    /// GPU-DRAM power correlation
    pub gpu_dram_correlation: Option<f64>,

    /// Compute to memory power ratio
    pub compute_memory_ratio: f64,

    /// Power transfer delays (cross-domain thermal lag)
    pub thermal_lag_cpu_gpu_s: Option<f64>,
}

impl MultiDomainTelemetry {
    /// Get telemetry for a specific domain
    pub fn get_domain(&self, domain: HardwareDomain) -> Option<&DomainTelemetry> {
        self.domains.get(&domain)
    }

    /// Get total system power
    pub fn total_power_w(&self) -> f64 {
        self.system.total_power_w
    }

    /// Get total system energy
    pub fn total_energy_j(&self) -> f64 {
        self.system.total_energy_j
    }

    /// Get all compute domains
    pub fn compute_domains(&self) -> Vec<(&HardwareDomain, &DomainTelemetry)> {
        self.domains
            .iter()
            .filter(|(d, _)| d.is_compute())
            .collect()
    }

    /// Get power breakdown as percentages
    pub fn power_breakdown_percent(&self) -> HashMap<HardwareDomain, f64> {
        let total = self.system.total_power_w;
        if total <= 0.0 {
            return HashMap::new();
        }

        self.domains
            .iter()
            .map(|(domain, telemetry)| (*domain, (telemetry.energy.power_w / total) * 100.0))
            .collect()
    }
}

// ============================================================================
// Platform-Specific Domain Readers
// ============================================================================

/// CPU energy reader using RAPL (Linux) or IOReport (macOS)
pub struct CpuDomainReader {
    #[cfg(target_os = "linux")]
    rapl: Option<crate::rapl::RAPLReader>,
    #[cfg(target_os = "macos")]
    iokit: Option<Box<dyn crate::platform::EnergyReader>>,
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    _placeholder: (),
}

impl CpuDomainReader {
    /// Create a new CPU domain reader
    pub fn new() -> Result<Self> {
        #[cfg(target_os = "linux")]
        {
            let rapl = crate::rapl::RAPLReader::new().ok();
            Ok(Self { rapl })
        }

        #[cfg(target_os = "macos")]
        {
            let iokit = crate::platform::create_reader().ok();
            Ok(Self { iokit })
        }

        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            Ok(Self { _placeholder: () })
        }
    }
}

impl DomainReader for CpuDomainReader {
    fn domain(&self) -> HardwareDomain {
        HardwareDomain::Cpu
    }

    fn is_available(&self) -> bool {
        #[cfg(target_os = "linux")]
        {
            self.rapl.is_some()
        }
        #[cfg(target_os = "macos")]
        {
            self.iokit.is_some()
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            false
        }
    }

    fn read_energy(&self) -> Result<f64> {
        #[cfg(target_os = "linux")]
        {
            self.rapl
                .as_ref()
                .ok_or(Error::Unsupported("RAPL not available".to_string()))?
                .read_package_energy()
        }

        #[cfg(target_os = "macos")]
        {
            self.iokit
                .as_ref()
                .ok_or(Error::Unsupported("IOKit not available".to_string()))?
                .read_energy()
        }

        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            Err(Error::Unsupported("Platform not supported".to_string()))
        }
    }

    fn read_temperature(&self) -> Result<Option<f64>> {
        #[cfg(target_os = "linux")]
        {
            // Read from thermal zone
            if let Ok(content) = std::fs::read_to_string("/sys/class/thermal/thermal_zone0/temp") {
                if let Ok(millidegrees) = content.trim().parse::<f64>() {
                    return Ok(Some(millidegrees / 1000.0));
                }
            }
            Ok(None)
        }

        #[cfg(target_os = "macos")]
        {
            self.iokit
                .as_ref()
                .and_then(|r| r.read_temperature().ok())
                .map(Some)
                .ok_or(Error::Unsupported("Temperature not available".to_string()))
        }

        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            Ok(None)
        }
    }
}

/// DRAM energy reader
pub struct DramDomainReader {
    #[cfg(target_os = "linux")]
    rapl: Option<crate::rapl::RAPLReader>,
    #[cfg(not(target_os = "linux"))]
    _placeholder: (),
}

impl DramDomainReader {
    pub fn new() -> Result<Self> {
        #[cfg(target_os = "linux")]
        {
            let rapl = crate::rapl::RAPLReader::new().ok();
            Ok(Self { rapl })
        }

        #[cfg(not(target_os = "linux"))]
        {
            Ok(Self { _placeholder: () })
        }
    }
}

impl DomainReader for DramDomainReader {
    fn domain(&self) -> HardwareDomain {
        HardwareDomain::Dram
    }

    fn is_available(&self) -> bool {
        #[cfg(target_os = "linux")]
        {
            self.rapl
                .as_ref()
                .map(|r| r.read_dram_energy().ok().flatten().is_some())
                .unwrap_or(false)
        }
        #[cfg(not(target_os = "linux"))]
        {
            false
        }
    }

    fn read_energy(&self) -> Result<f64> {
        #[cfg(target_os = "linux")]
        {
            self.rapl
                .as_ref()
                .ok_or(Error::Unsupported("RAPL not available".to_string()))?
                .read_dram_energy()?
                .ok_or(Error::Unsupported("DRAM energy not available".to_string()))
        }

        #[cfg(not(target_os = "linux"))]
        {
            Err(Error::Unsupported("DRAM energy not supported".to_string()))
        }
    }
}

/// GPU energy reader (NVIDIA NVML, AMD ROCm, Apple integrated)
pub struct GpuDomainReader {
    device_id: u32,
    /// Estimated power in watts (for platforms without direct measurement)
    estimated_power_w: f64,
    /// Accumulated energy from estimation
    accumulated_energy_j: std::sync::Mutex<f64>,
    last_read: std::sync::Mutex<std::time::Instant>,
}

impl GpuDomainReader {
    pub fn new(device_id: u32) -> Result<Self> {
        Ok(Self {
            device_id,
            estimated_power_w: 0.0,
            accumulated_energy_j: std::sync::Mutex::new(0.0),
            last_read: std::sync::Mutex::new(std::time::Instant::now()),
        })
    }

    /// Try to read GPU power via NVML (NVIDIA)
    #[cfg(target_os = "linux")]
    fn try_nvml_power(&self) -> Option<f64> {
        // In a full implementation, this would use nvidia-smi or NVML bindings
        // For now, return None and fall back to estimation
        None
    }

    /// Try to read GPU power via ROCm SMI (AMD)
    #[cfg(target_os = "linux")]
    fn try_rocm_power(&self) -> Option<f64> {
        // In a full implementation, this would use rocm-smi
        None
    }
}

impl DomainReader for GpuDomainReader {
    fn domain(&self) -> HardwareDomain {
        if self.device_id == 0 {
            HardwareDomain::Gpu
        } else {
            HardwareDomain::GpuDevice(self.device_id)
        }
    }

    fn is_available(&self) -> bool {
        // Check for GPU presence
        #[cfg(target_os = "linux")]
        {
            // Check for NVIDIA
            std::path::Path::new("/dev/nvidia0").exists()
                || std::path::Path::new("/dev/dri/renderD128").exists()
        }
        #[cfg(target_os = "macos")]
        {
            // Apple Silicon always has integrated GPU
            true
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            false
        }
    }

    fn read_energy(&self) -> Result<f64> {
        // For now, use time-based estimation
        // In a full implementation, this would use NVML/ROCm/Metal APIs
        let mut last = self.last_read.lock().unwrap();
        let mut accumulated = self.accumulated_energy_j.lock().unwrap();

        let now = std::time::Instant::now();
        let dt = now.duration_since(*last).as_secs_f64();
        *last = now;

        // Estimate: assume some base power draw
        let power = self.estimated_power_w.max(10.0); // At least 10W when active
        *accumulated += power * dt;

        Ok(*accumulated)
    }
}

/// Estimated domain reader for I/O devices (storage, network)
pub struct EstimatedDomainReader {
    domain: HardwareDomain,
    /// Energy per byte (J/B) - for I/O estimation
    energy_per_byte_j: f64,
    /// Base idle power (W)
    idle_power_w: f64,
    /// Total bytes transferred
    total_bytes: std::sync::Mutex<u64>,
    /// Accumulated energy
    accumulated_energy_j: std::sync::Mutex<f64>,
    last_read: std::sync::Mutex<std::time::Instant>,
}

impl EstimatedDomainReader {
    /// Create for storage domain
    pub fn storage() -> Self {
        Self {
            domain: HardwareDomain::Storage,
            // Typical NVMe: ~5 nJ/byte for reads, ~10 nJ/byte for writes
            energy_per_byte_j: 7.5e-9,
            idle_power_w: 0.5, // ~0.5W idle
            total_bytes: std::sync::Mutex::new(0),
            accumulated_energy_j: std::sync::Mutex::new(0.0),
            last_read: std::sync::Mutex::new(std::time::Instant::now()),
        }
    }

    /// Create for network domain
    pub fn network() -> Self {
        Self {
            domain: HardwareDomain::Network,
            // Typical NIC: ~10-50 nJ/byte
            energy_per_byte_j: 25e-9,
            idle_power_w: 1.0, // ~1W idle for Ethernet
            total_bytes: std::sync::Mutex::new(0),
            accumulated_energy_j: std::sync::Mutex::new(0.0),
            last_read: std::sync::Mutex::new(std::time::Instant::now()),
        }
    }

    /// Record bytes transferred
    pub fn record_bytes(&self, bytes: u64) {
        let mut total = self.total_bytes.lock().unwrap();
        *total = total.saturating_add(bytes);
    }
}

impl DomainReader for EstimatedDomainReader {
    fn domain(&self) -> HardwareDomain {
        self.domain
    }

    fn is_available(&self) -> bool {
        true // Estimation is always "available"
    }

    fn read_energy(&self) -> Result<f64> {
        let mut last = self.last_read.lock().unwrap();
        let mut accumulated = self.accumulated_energy_j.lock().unwrap();
        let total_bytes = *self.total_bytes.lock().unwrap();

        let now = std::time::Instant::now();
        let dt = now.duration_since(*last).as_secs_f64();
        *last = now;

        // Idle power + I/O energy
        let idle_energy = self.idle_power_w * dt;
        let io_energy = total_bytes as f64 * self.energy_per_byte_j;

        *accumulated = idle_energy + io_energy;
        Ok(*accumulated)
    }
}

// ============================================================================
// NVIDIA NVML Domain Reader (via libloading)
// ============================================================================

/// NVIDIA GPU energy reader using NVML (NVIDIA Management Library).
///
/// Uses dynamic loading (libloading) to access NVML at runtime without requiring
/// the NVML SDK at compile time. On Linux, loads `libnvidia-ml.so.1`; on Windows,
/// loads `nvml.dll`.
///
/// NVML's `nvmlDeviceGetTotalEnergyConsumption` reports millijoules, which we
/// convert to joules. Temperature and power are also available.
#[cfg(feature = "accelerators")]
pub struct NvmlDomainReader {
    device_index: u32,
    /// Opaque NVML device handle (void* from C API)
    device_handle: *mut std::ffi::c_void,
    /// Loaded NVML shared library
    _lib: libloading::Library,
    /// nvmlDeviceGetTotalEnergyConsumption(device, *mut u64) -> u32
    fn_get_energy: unsafe extern "C" fn(*mut std::ffi::c_void, *mut u64) -> u32,
    /// nvmlDeviceGetPowerUsage(device, *mut u32) -> u32
    fn_get_power: unsafe extern "C" fn(*mut std::ffi::c_void, *mut u32) -> u32,
    /// nvmlDeviceGetTemperature(device, sensor_type, *mut u32) -> u32
    fn_get_temp: unsafe extern "C" fn(*mut std::ffi::c_void, u32, *mut u32) -> u32,
    /// nvmlShutdown() -> u32
    fn_shutdown: unsafe extern "C" fn() -> u32,
}

// SAFETY: The NVML library handles are thread-safe per NVIDIA documentation.
// All NVML functions are safe to call from multiple threads.
#[cfg(feature = "accelerators")]
unsafe impl Send for NvmlDomainReader {}
#[cfg(feature = "accelerators")]
unsafe impl Sync for NvmlDomainReader {}

#[cfg(feature = "accelerators")]
impl NvmlDomainReader {
    /// Create a new NVML reader for the given GPU device index.
    ///
    /// Attempts to load the NVML shared library, initialize it, and obtain a
    /// device handle. Returns an error if NVML is not installed or the device
    /// index is invalid.
    pub fn new(device_index: u32) -> Result<Self> {
        // Platform-specific library name
        #[cfg(target_os = "linux")]
        let lib_name = "libnvidia-ml.so.1";
        #[cfg(target_os = "windows")]
        let lib_name = "nvml.dll";
        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        let lib_name = "libnvidia-ml.so.1"; // Fallback, will fail to load

        // SAFETY: We're loading a well-known system library. The symbol signatures
        // match the NVML C API documentation.
        let lib = unsafe { libloading::Library::new(lib_name) }
            .map_err(|e| Error::Unsupported(format!("NVML not found: {}", e)))?;

        // Load nvmlInit_v2
        let fn_init: libloading::Symbol<unsafe extern "C" fn() -> u32> = unsafe {
            lib.get(b"nvmlInit_v2\0")
                .map_err(|e| Error::Unsupported(format!("nvmlInit_v2 not found: {}", e)))?
        };

        let ret = unsafe { fn_init() };
        if ret != 0 {
            return Err(Error::Unsupported(format!(
                "nvmlInit_v2 failed with code {}",
                ret
            )));
        }

        // Load nvmlDeviceGetHandleByIndex_v2
        let fn_get_handle: libloading::Symbol<
            unsafe extern "C" fn(u32, *mut *mut std::ffi::c_void) -> u32,
        > = unsafe {
            lib.get(b"nvmlDeviceGetHandleByIndex_v2\0")
                .map_err(|e| Error::Unsupported(format!("nvmlDeviceGetHandleByIndex: {}", e)))?
        };

        let mut device_handle: *mut std::ffi::c_void = std::ptr::null_mut();
        let ret = unsafe { fn_get_handle(device_index, &mut device_handle) };
        if ret != 0 {
            return Err(Error::Unsupported(format!(
                "nvmlDeviceGetHandleByIndex failed for device {}: code {}",
                device_index, ret
            )));
        }

        // Load energy/power/temperature functions.
        // Extract raw fn pointers before moving `lib` into the struct — the
        // Symbol borrows lib, so we must dereference first.
        let fn_get_energy = unsafe {
            let sym: libloading::Symbol<
                unsafe extern "C" fn(*mut std::ffi::c_void, *mut u64) -> u32,
            > = lib
                .get(b"nvmlDeviceGetTotalEnergyConsumption\0")
                .map_err(|e| Error::Unsupported(format!("energy function: {}", e)))?;
            *sym
        };

        let fn_get_power = unsafe {
            let sym: libloading::Symbol<
                unsafe extern "C" fn(*mut std::ffi::c_void, *mut u32) -> u32,
            > = lib
                .get(b"nvmlDeviceGetPowerUsage\0")
                .map_err(|e| Error::Unsupported(format!("power function: {}", e)))?;
            *sym
        };

        let fn_get_temp = unsafe {
            let sym: libloading::Symbol<
                unsafe extern "C" fn(*mut std::ffi::c_void, u32, *mut u32) -> u32,
            > = lib
                .get(b"nvmlDeviceGetTemperature\0")
                .map_err(|e| Error::Unsupported(format!("temperature function: {}", e)))?;
            *sym
        };

        let fn_shutdown = unsafe {
            let sym: libloading::Symbol<unsafe extern "C" fn() -> u32> = lib
                .get(b"nvmlShutdown\0")
                .map_err(|e| Error::Unsupported(format!("shutdown function: {}", e)))?;
            *sym
        };

        Ok(Self {
            device_index,
            device_handle,
            _lib: lib,
            fn_get_energy,
            fn_get_power,
            fn_get_temp,
            fn_shutdown,
        })
    }
}

#[cfg(feature = "accelerators")]
impl Drop for NvmlDomainReader {
    fn drop(&mut self) {
        // Best-effort shutdown of NVML
        unsafe {
            (self.fn_shutdown)();
        }
    }
}

#[cfg(feature = "accelerators")]
impl DomainReader for NvmlDomainReader {
    fn domain(&self) -> HardwareDomain {
        if self.device_index == 0 {
            HardwareDomain::Gpu
        } else {
            HardwareDomain::GpuDevice(self.device_index)
        }
    }

    fn is_available(&self) -> bool {
        // If we got this far (constructor succeeded), NVML is available
        true
    }

    fn read_energy(&self) -> Result<f64> {
        let mut millijoules: u64 = 0;
        let ret = unsafe { (self.fn_get_energy)(self.device_handle, &mut millijoules) };
        if ret != 0 {
            return Err(Error::Unsupported(format!(
                "nvmlDeviceGetTotalEnergyConsumption failed: code {}",
                ret
            )));
        }
        // Convert millijoules to joules
        Ok(millijoules as f64 / 1000.0)
    }

    fn read_temperature(&self) -> Result<Option<f64>> {
        let mut temp: u32 = 0;
        // NVML_TEMPERATURE_GPU = 0
        let ret = unsafe { (self.fn_get_temp)(self.device_handle, 0, &mut temp) };
        if ret != 0 {
            return Ok(None);
        }
        Ok(Some(temp as f64))
    }

    fn read_utilization(&self) -> Result<Option<f64>> {
        // Power usage as a proxy for utilization (normalized by typical TDP)
        let mut milliwatts: u32 = 0;
        let ret = unsafe { (self.fn_get_power)(self.device_handle, &mut milliwatts) };
        if ret != 0 {
            return Ok(None);
        }
        // Assume ~300W TDP for normalization; actual utilization would need
        // nvmlDeviceGetUtilizationRates which requires a different struct.
        let power_w = milliwatts as f64 / 1000.0;
        Ok(Some((power_w / 300.0).clamp(0.0, 1.0)))
    }
}

// ============================================================================
// AMD ROCm SMI Domain Reader (via libloading)
// ============================================================================

/// AMD GPU energy reader using ROCm SMI (System Management Interface).
///
/// Linux-only. Uses dynamic loading to access `librocm_smi64.so`.
/// `rsmi_dev_energy_count_get` returns energy in microjoules.
#[cfg(all(target_os = "linux", feature = "accelerators"))]
pub struct RocmDomainReader {
    device_index: u32,
    /// Loaded ROCm SMI shared library
    _lib: libloading::Library,
    /// rsmi_dev_energy_count_get(device_index, *mut u64, *mut f32, *mut u64) -> i32
    /// Returns: (energy_accumulator_uj, counter_resolution, timestamp)
    fn_get_energy: unsafe extern "C" fn(u32, *mut u64, *mut f32, *mut u64) -> i32,
    /// rsmi_dev_temp_metric_get(device_index, sensor_type, metric, *mut i64) -> i32
    fn_get_temp: unsafe extern "C" fn(u32, u32, u32, *mut i64) -> i32,
    /// rsmi_dev_power_ave_get(device_index, sensor_id, *mut u64) -> i32
    fn_get_power: unsafe extern "C" fn(u32, u32, *mut u64) -> i32,
    /// rsmi_shut_down() -> i32
    fn_shutdown: unsafe extern "C" fn() -> i32,
}

#[cfg(all(target_os = "linux", feature = "accelerators"))]
unsafe impl Send for RocmDomainReader {}
#[cfg(all(target_os = "linux", feature = "accelerators"))]
unsafe impl Sync for RocmDomainReader {}

#[cfg(all(target_os = "linux", feature = "accelerators"))]
impl RocmDomainReader {
    /// Create a new ROCm SMI reader for the given AMD GPU device index.
    pub fn new(device_index: u32) -> Result<Self> {
        let lib = unsafe { libloading::Library::new("librocm_smi64.so") }
            .map_err(|e| Error::Unsupported(format!("ROCm SMI not found: {}", e)))?;

        // rsmi_init(init_flags) -> i32
        let fn_init: libloading::Symbol<unsafe extern "C" fn(u64) -> i32> = unsafe {
            lib.get(b"rsmi_init\0")
                .map_err(|e| Error::Unsupported(format!("rsmi_init not found: {}", e)))?
        };

        let ret = unsafe { fn_init(0) };
        if ret != 0 {
            return Err(Error::Unsupported(format!(
                "rsmi_init failed with code {}",
                ret
            )));
        }

        // Extract raw fn pointers before moving `lib` into the struct
        let fn_get_energy = unsafe {
            let sym: libloading::Symbol<
                unsafe extern "C" fn(u32, *mut u64, *mut f32, *mut u64) -> i32,
            > = lib
                .get(b"rsmi_dev_energy_count_get\0")
                .map_err(|e| Error::Unsupported(format!("energy function: {}", e)))?;
            *sym
        };

        let fn_get_temp = unsafe {
            let sym: libloading::Symbol<
                unsafe extern "C" fn(u32, u32, u32, *mut i64) -> i32,
            > = lib
                .get(b"rsmi_dev_temp_metric_get\0")
                .map_err(|e| Error::Unsupported(format!("temperature function: {}", e)))?;
            *sym
        };

        let fn_get_power = unsafe {
            let sym: libloading::Symbol<unsafe extern "C" fn(u32, u32, *mut u64) -> i32> = lib
                .get(b"rsmi_dev_power_ave_get\0")
                .map_err(|e| Error::Unsupported(format!("power function: {}", e)))?;
            *sym
        };

        let fn_shutdown = unsafe {
            let sym: libloading::Symbol<unsafe extern "C" fn() -> i32> = lib
                .get(b"rsmi_shut_down\0")
                .map_err(|e| Error::Unsupported(format!("shutdown function: {}", e)))?;
            *sym
        };

        Ok(Self {
            device_index,
            _lib: lib,
            fn_get_energy,
            fn_get_temp,
            fn_get_power,
            fn_shutdown,
        })
    }
}

#[cfg(all(target_os = "linux", feature = "accelerators"))]
impl Drop for RocmDomainReader {
    fn drop(&mut self) {
        unsafe {
            (self.fn_shutdown)();
        }
    }
}

#[cfg(all(target_os = "linux", feature = "accelerators"))]
impl DomainReader for RocmDomainReader {
    fn domain(&self) -> HardwareDomain {
        if self.device_index == 0 {
            HardwareDomain::Gpu
        } else {
            HardwareDomain::GpuDevice(self.device_index)
        }
    }

    fn is_available(&self) -> bool {
        true
    }

    fn read_energy(&self) -> Result<f64> {
        let mut energy_uj: u64 = 0;
        let mut resolution: f32 = 0.0;
        let mut timestamp: u64 = 0;
        let ret = unsafe {
            (self.fn_get_energy)(
                self.device_index,
                &mut energy_uj,
                &mut resolution,
                &mut timestamp,
            )
        };
        if ret != 0 {
            return Err(Error::Unsupported(format!(
                "rsmi_dev_energy_count_get failed: code {}",
                ret
            )));
        }
        // Convert microjoules to joules
        Ok(energy_uj as f64 / 1_000_000.0)
    }

    fn read_temperature(&self) -> Result<Option<f64>> {
        let mut temp_millideg: i64 = 0;
        // RSMI_TEMP_TYPE_EDGE = 0, RSMI_TEMP_CURRENT = 0
        let ret =
            unsafe { (self.fn_get_temp)(self.device_index, 0, 0, &mut temp_millideg) };
        if ret != 0 {
            return Ok(None);
        }
        // ROCm SMI returns millidegrees Celsius
        Ok(Some(temp_millideg as f64 / 1000.0))
    }

    fn read_utilization(&self) -> Result<Option<f64>> {
        let mut microwatts: u64 = 0;
        let ret = unsafe { (self.fn_get_power)(self.device_index, 0, &mut microwatts) };
        if ret != 0 {
            return Ok(None);
        }
        // Power as proxy for utilization, normalized by ~300W TDP
        let power_w = microwatts as f64 / 1_000_000.0;
        Ok(Some((power_w / 300.0).clamp(0.0, 1.0)))
    }
}

// ============================================================================
// Intel Level Zero Sysman Domain Reader (via libloading)
// ============================================================================

/// Intel GPU/XPU energy reader using Level Zero Sysman API.
///
/// Available on Linux and Windows. Uses dynamic loading to access `libze_loader.so`
/// (Linux) or `ze_loader.dll` (Windows). `zesPowerGetEnergyCounter` returns
/// energy in microjoules.
#[cfg(feature = "accelerators")]
pub struct LevelZeroDomainReader {
    device_index: u32,
    /// Loaded Level Zero library
    _lib: libloading::Library,
    /// Opaque driver handle (retained for future Sysman queries)
    _driver_handle: *mut std::ffi::c_void,
    /// Opaque device handle (retained for future Sysman queries)
    _device_handle: *mut std::ffi::c_void,
    /// Opaque power handle (from zesPowerGet)
    power_handle: *mut std::ffi::c_void,
    /// zesPowerGetEnergyCounter(power_handle, *mut ZesPowerEnergyCounter) -> i32
    fn_get_energy: unsafe extern "C" fn(*mut std::ffi::c_void, *mut LevelZeroEnergyCounter) -> i32,
}

/// Level Zero energy counter structure (zes_power_energy_counter_t)
#[cfg(feature = "accelerators")]
#[repr(C)]
struct LevelZeroEnergyCounter {
    /// Total energy consumed in microjoules
    energy: u64,
    /// Monotonic timestamp in microseconds
    timestamp: u64,
}

#[cfg(feature = "accelerators")]
unsafe impl Send for LevelZeroDomainReader {}
#[cfg(feature = "accelerators")]
unsafe impl Sync for LevelZeroDomainReader {}

#[cfg(feature = "accelerators")]
impl LevelZeroDomainReader {
    /// Create a new Level Zero Sysman reader for the given device index.
    pub fn new(device_index: u32) -> Result<Self> {
        #[cfg(target_os = "linux")]
        let lib_name = "libze_loader.so";
        #[cfg(target_os = "windows")]
        let lib_name = "ze_loader.dll";
        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        let lib_name = "libze_loader.so";

        let lib = unsafe { libloading::Library::new(lib_name) }
            .map_err(|e| Error::Unsupported(format!("Level Zero not found: {}", e)))?;

        // zeInit(flags) -> ze_result_t
        let fn_init: libloading::Symbol<unsafe extern "C" fn(u32) -> i32> = unsafe {
            lib.get(b"zeInit\0")
                .map_err(|e| Error::Unsupported(format!("zeInit not found: {}", e)))?
        };

        // ZE_INIT_FLAG_GPU_ONLY = 1
        let ret = unsafe { fn_init(1) };
        if ret != 0 {
            return Err(Error::Unsupported(format!(
                "zeInit failed with code {}",
                ret
            )));
        }

        // zeDriverGet(*mut count, *mut drivers) -> ze_result_t
        let fn_driver_get: libloading::Symbol<
            unsafe extern "C" fn(*mut u32, *mut *mut std::ffi::c_void) -> i32,
        > = unsafe {
            lib.get(b"zeDriverGet\0")
                .map_err(|e| Error::Unsupported(format!("zeDriverGet: {}", e)))?
        };

        let mut driver_count: u32 = 0;
        let ret = unsafe { fn_driver_get(&mut driver_count, std::ptr::null_mut()) };
        if ret != 0 || driver_count == 0 {
            return Err(Error::Unsupported(
                "No Level Zero drivers found".to_string(),
            ));
        }

        let mut drivers = vec![std::ptr::null_mut(); driver_count as usize];
        let ret = unsafe { fn_driver_get(&mut driver_count, drivers.as_mut_ptr()) };
        if ret != 0 {
            return Err(Error::Unsupported(format!(
                "zeDriverGet failed: code {}",
                ret
            )));
        }

        let driver_handle = drivers[0];

        // zesDeviceGet(driver, *mut count, *mut devices) -> ze_result_t
        // Note: Sysman uses zes* prefix. Device enumeration can also be done via
        // zeDeviceGet, but for Sysman power API we need zes handles.
        let fn_device_get: libloading::Symbol<
            unsafe extern "C" fn(*mut std::ffi::c_void, *mut u32, *mut *mut std::ffi::c_void) -> i32,
        > = unsafe {
            lib.get(b"zeDeviceGet\0")
                .map_err(|e| Error::Unsupported(format!("zeDeviceGet: {}", e)))?
        };

        let mut device_count: u32 = 0;
        let ret =
            unsafe { fn_device_get(driver_handle, &mut device_count, std::ptr::null_mut()) };
        if ret != 0 || device_count == 0 {
            return Err(Error::Unsupported(
                "No Level Zero devices found".to_string(),
            ));
        }

        if device_index >= device_count {
            return Err(Error::Unsupported(format!(
                "Device index {} out of range ({})",
                device_index, device_count
            )));
        }

        let mut devices = vec![std::ptr::null_mut(); device_count as usize];
        let ret =
            unsafe { fn_device_get(driver_handle, &mut device_count, devices.as_mut_ptr()) };
        if ret != 0 {
            return Err(Error::Unsupported(format!(
                "zeDeviceGet failed: code {}",
                ret
            )));
        }

        let device_handle = devices[device_index as usize];

        // zesPowerGet(device, *mut count, *mut power_handles) -> ze_result_t
        let fn_power_get: libloading::Symbol<
            unsafe extern "C" fn(*mut std::ffi::c_void, *mut u32, *mut *mut std::ffi::c_void) -> i32,
        > = unsafe {
            lib.get(b"zesPowerGet\0")
                .map_err(|e| Error::Unsupported(format!("zesPowerGet: {}", e)))?
        };

        let mut power_count: u32 = 0;
        let ret =
            unsafe { fn_power_get(device_handle, &mut power_count, std::ptr::null_mut()) };
        if ret != 0 || power_count == 0 {
            return Err(Error::Unsupported(
                "No Level Zero power domains found".to_string(),
            ));
        }

        let mut power_handles = vec![std::ptr::null_mut(); power_count as usize];
        let ret = unsafe {
            fn_power_get(
                device_handle,
                &mut power_count,
                power_handles.as_mut_ptr(),
            )
        };
        if ret != 0 {
            return Err(Error::Unsupported(format!(
                "zesPowerGet failed: code {}",
                ret
            )));
        }

        let power_handle = power_handles[0]; // First (total) power domain

        // Extract raw fn pointer before moving `lib` into the struct
        let fn_get_energy = unsafe {
            let sym: libloading::Symbol<
                unsafe extern "C" fn(*mut std::ffi::c_void, *mut LevelZeroEnergyCounter) -> i32,
            > = lib
                .get(b"zesPowerGetEnergyCounter\0")
                .map_err(|e| Error::Unsupported(format!("zesPowerGetEnergyCounter: {}", e)))?;
            *sym
        };

        Ok(Self {
            device_index,
            _lib: lib,
            _driver_handle: driver_handle,
            _device_handle: device_handle,
            power_handle,
            fn_get_energy,
        })
    }
}

#[cfg(feature = "accelerators")]
impl DomainReader for LevelZeroDomainReader {
    fn domain(&self) -> HardwareDomain {
        if self.device_index == 0 {
            HardwareDomain::IntelXpu
        } else {
            HardwareDomain::IntelXpuDevice(self.device_index)
        }
    }

    fn is_available(&self) -> bool {
        true
    }

    fn read_energy(&self) -> Result<f64> {
        let mut counter = LevelZeroEnergyCounter {
            energy: 0,
            timestamp: 0,
        };
        let ret = unsafe { (self.fn_get_energy)(self.power_handle, &mut counter) };
        if ret != 0 {
            return Err(Error::Unsupported(format!(
                "zesPowerGetEnergyCounter failed: code {}",
                ret
            )));
        }
        // Convert microjoules to joules
        Ok(counter.energy as f64 / 1_000_000.0)
    }
}

// ============================================================================
// Intel Gaudi HLML Domain Reader (via libloading)
// ============================================================================

/// Intel Gaudi (Habana Labs) energy reader using HLML.
///
/// Linux-only. Uses dynamic loading to access `libhlml.so`.
/// `hlml_device_get_power_usage` returns milliwatts (similar to NVML).
#[cfg(all(target_os = "linux", feature = "accelerators"))]
pub struct HlmlDomainReader {
    device_index: u32,
    /// Opaque HLML device handle
    device_handle: *mut std::ffi::c_void,
    /// Loaded HLML library
    _lib: libloading::Library,
    /// hlml_device_get_power_usage(device, *mut u32) -> i32
    fn_get_power: unsafe extern "C" fn(*mut std::ffi::c_void, *mut u32) -> i32,
    /// hlml_device_get_temperature(device, sensor_type, *mut u32) -> i32
    fn_get_temp: unsafe extern "C" fn(*mut std::ffi::c_void, u32, *mut u32) -> i32,
    /// hlml_shutdown() -> i32
    fn_shutdown: unsafe extern "C" fn() -> i32,
    /// Accumulated energy from power integration (milliwatts -> joules)
    accumulated_energy_j: std::sync::Mutex<f64>,
    last_read: std::sync::Mutex<Instant>,
}

#[cfg(all(target_os = "linux", feature = "accelerators"))]
unsafe impl Send for HlmlDomainReader {}
#[cfg(all(target_os = "linux", feature = "accelerators"))]
unsafe impl Sync for HlmlDomainReader {}

#[cfg(all(target_os = "linux", feature = "accelerators"))]
impl HlmlDomainReader {
    /// Create a new HLML reader for the given Gaudi device index.
    pub fn new(device_index: u32) -> Result<Self> {
        let lib = unsafe { libloading::Library::new("libhlml.so") }
            .map_err(|e| Error::Unsupported(format!("HLML not found: {}", e)))?;

        // hlml_init() -> i32
        let fn_init: libloading::Symbol<unsafe extern "C" fn() -> i32> = unsafe {
            lib.get(b"hlml_init\0")
                .map_err(|e| Error::Unsupported(format!("hlml_init not found: {}", e)))?
        };

        let ret = unsafe { fn_init() };
        if ret != 0 {
            return Err(Error::Unsupported(format!(
                "hlml_init failed with code {}",
                ret
            )));
        }

        // hlml_device_get_handle_by_index(index, *mut handle) -> i32
        let fn_get_handle: libloading::Symbol<
            unsafe extern "C" fn(u32, *mut *mut std::ffi::c_void) -> i32,
        > = unsafe {
            lib.get(b"hlml_device_get_handle_by_index\0")
                .map_err(|e| Error::Unsupported(format!("hlml_device_get_handle: {}", e)))?
        };

        let mut device_handle: *mut std::ffi::c_void = std::ptr::null_mut();
        let ret = unsafe { fn_get_handle(device_index, &mut device_handle) };
        if ret != 0 {
            return Err(Error::Unsupported(format!(
                "hlml_device_get_handle_by_index failed for device {}: code {}",
                device_index, ret
            )));
        }

        // Extract raw fn pointers before moving `lib` into the struct
        let fn_get_power = unsafe {
            let sym: libloading::Symbol<
                unsafe extern "C" fn(*mut std::ffi::c_void, *mut u32) -> i32,
            > = lib
                .get(b"hlml_device_get_power_usage\0")
                .map_err(|e| Error::Unsupported(format!("power function: {}", e)))?;
            *sym
        };

        let fn_get_temp = unsafe {
            let sym: libloading::Symbol<
                unsafe extern "C" fn(*mut std::ffi::c_void, u32, *mut u32) -> i32,
            > = lib
                .get(b"hlml_device_get_temperature\0")
                .map_err(|e| Error::Unsupported(format!("temperature function: {}", e)))?;
            *sym
        };

        let fn_shutdown = unsafe {
            let sym: libloading::Symbol<unsafe extern "C" fn() -> i32> = lib
                .get(b"hlml_shutdown\0")
                .map_err(|e| Error::Unsupported(format!("shutdown function: {}", e)))?;
            *sym
        };

        Ok(Self {
            device_index,
            device_handle,
            _lib: lib,
            fn_get_power,
            fn_get_temp,
            fn_shutdown,
            accumulated_energy_j: std::sync::Mutex::new(0.0),
            last_read: std::sync::Mutex::new(Instant::now()),
        })
    }

    /// Read instantaneous power in watts
    fn read_power_w(&self) -> Result<f64> {
        let mut milliwatts: u32 = 0;
        let ret = unsafe { (self.fn_get_power)(self.device_handle, &mut milliwatts) };
        if ret != 0 {
            return Err(Error::Unsupported(format!(
                "hlml_device_get_power_usage failed: code {}",
                ret
            )));
        }
        Ok(milliwatts as f64 / 1000.0)
    }
}

#[cfg(all(target_os = "linux", feature = "accelerators"))]
impl Drop for HlmlDomainReader {
    fn drop(&mut self) {
        unsafe {
            (self.fn_shutdown)();
        }
    }
}

#[cfg(all(target_os = "linux", feature = "accelerators"))]
impl DomainReader for HlmlDomainReader {
    fn domain(&self) -> HardwareDomain {
        if self.device_index == 0 {
            HardwareDomain::IntelGaudi
        } else {
            HardwareDomain::IntelGaudiDevice(self.device_index)
        }
    }

    fn is_available(&self) -> bool {
        true
    }

    fn read_energy(&self) -> Result<f64> {
        // HLML provides power (milliwatts), not cumulative energy.
        // We integrate power over time to get energy.
        let power_w = self.read_power_w()?;

        let mut last = self.last_read.lock().unwrap();
        let mut accumulated = self.accumulated_energy_j.lock().unwrap();

        let now = Instant::now();
        let dt = now.duration_since(*last).as_secs_f64();
        *last = now;

        *accumulated += power_w * dt;
        Ok(*accumulated)
    }

    fn read_temperature(&self) -> Result<Option<f64>> {
        let mut temp: u32 = 0;
        // HLML_TEMPERATURE_ON_CHIP = 0
        let ret = unsafe { (self.fn_get_temp)(self.device_handle, 0, &mut temp) };
        if ret != 0 {
            return Ok(None);
        }
        Ok(Some(temp as f64))
    }
}

// ============================================================================
// Google TPU Domain Reader (Tier 3 estimation)
// ============================================================================

/// Google TPU energy reader using Tier 3 TDP-based estimation.
///
/// Always available (no libloading required). Detects TPU presence via
/// `/dev/accel0` or the `CLOUD_TPU_TASK_ID` environment variable. Determines
/// TPU generation from `TPU_ACCELERATOR_TYPE` and uses published TDP values.
///
/// | Generation | TDP (W) |
/// |------------|---------|
/// | v5e        | 100     |
/// | v5p        | 250     |
/// | v6e        | 170     |
/// | v7         | 200     |
/// | default    | 150     |
pub struct TpuDomainReader {
    /// TPU generation string (e.g. "v5e", "v5p", "v6e", "v7")
    generation: String,
    /// Thermal Design Power in watts for this generation
    tdp_watts: f64,
    /// Whether a TPU was detected
    available: bool,
    /// Accumulated energy from TDP estimation
    accumulated_energy_j: std::sync::Mutex<f64>,
    last_read: std::sync::Mutex<Instant>,
}

impl TpuDomainReader {
    /// Create a new TPU reader. Detects TPU presence and generation automatically.
    pub fn new() -> Result<Self> {
        let available = std::path::Path::new("/dev/accel0").exists()
            || std::env::var("CLOUD_TPU_TASK_ID").is_ok();

        let (generation, tdp_watts) = Self::detect_generation();

        Ok(Self {
            generation,
            tdp_watts,
            available,
            accumulated_energy_j: std::sync::Mutex::new(0.0),
            last_read: std::sync::Mutex::new(Instant::now()),
        })
    }

    /// Detect TPU generation from `TPU_ACCELERATOR_TYPE` environment variable.
    ///
    /// Parses strings like "v5e-256", "v5p-128", "v6e-4", "v7-256" to extract
    /// the generation prefix and map to the corresponding TDP.
    fn detect_generation() -> (String, f64) {
        if let Ok(accel_type) = std::env::var("TPU_ACCELERATOR_TYPE") {
            let lower = accel_type.to_lowercase();
            if lower.starts_with("v5e") {
                return ("v5e".to_string(), 100.0);
            } else if lower.starts_with("v5p") {
                return ("v5p".to_string(), 250.0);
            } else if lower.starts_with("v6e") {
                return ("v6e".to_string(), 170.0);
            } else if lower.starts_with("v7") {
                return ("v7".to_string(), 200.0);
            } else if lower.starts_with("v4") {
                return ("v4".to_string(), 175.0);
            }
        }
        // Default: unknown generation
        ("unknown".to_string(), 150.0)
    }

    /// Get the detected TPU generation string
    pub fn generation(&self) -> &str {
        &self.generation
    }

    /// Get the TDP estimate in watts
    pub fn tdp_watts(&self) -> f64 {
        self.tdp_watts
    }
}

impl DomainReader for TpuDomainReader {
    fn domain(&self) -> HardwareDomain {
        HardwareDomain::Tpu
    }

    fn is_available(&self) -> bool {
        self.available
    }

    fn read_energy(&self) -> Result<f64> {
        // Tier 3 estimation: energy = TDP * elapsed_time
        let mut last = self.last_read.lock().unwrap();
        let mut accumulated = self.accumulated_energy_j.lock().unwrap();

        let now = Instant::now();
        let dt = now.duration_since(*last).as_secs_f64();
        *last = now;

        *accumulated += self.tdp_watts * dt;
        Ok(*accumulated)
    }
}

// ============================================================================
// AWS Neuron Domain Reader (Tier 3 estimation)
// ============================================================================

/// AWS Trainium/Inferentia energy reader using Tier 3 TDP-based estimation.
///
/// Always available (no libloading required). Detects Neuron devices via
/// `/dev/neuron0` or the `NEURON_RT_VISIBLE_CORES` environment variable.
///
/// | Chip         | TDP (W) |
/// |--------------|---------|
/// | Trainium1    | 210     |
/// | Trainium2    | 400     |
/// | Inferentia2  | 115     |
/// | default      | 210     |
pub struct NeuronDomainReader {
    /// Chip variant (e.g. "trainium1", "trainium2", "inferentia2")
    chip_variant: String,
    /// Thermal Design Power in watts
    tdp_watts: f64,
    /// Whether a Neuron device was detected
    available: bool,
    /// Accumulated energy from TDP estimation
    accumulated_energy_j: std::sync::Mutex<f64>,
    last_read: std::sync::Mutex<Instant>,
}

impl NeuronDomainReader {
    /// Create a new Neuron reader. Detects device presence and chip variant
    /// automatically.
    pub fn new() -> Result<Self> {
        let available = std::path::Path::new("/dev/neuron0").exists()
            || std::env::var("NEURON_RT_VISIBLE_CORES").is_ok();

        let (chip_variant, tdp_watts) = Self::detect_variant();

        Ok(Self {
            chip_variant,
            tdp_watts,
            available,
            accumulated_energy_j: std::sync::Mutex::new(0.0),
            last_read: std::sync::Mutex::new(Instant::now()),
        })
    }

    /// Detect chip variant from environment or instance metadata.
    ///
    /// Checks `NEURON_DEVICE_TYPE` and falls back to instance type heuristics:
    /// - `trn1` instances -> Trainium1
    /// - `trn2` instances -> Trainium2
    /// - `inf2` instances -> Inferentia2
    fn detect_variant() -> (String, f64) {
        // Check explicit device type env var
        if let Ok(device_type) = std::env::var("NEURON_DEVICE_TYPE") {
            let lower = device_type.to_lowercase();
            if lower.contains("trainium2") || lower.contains("trn2") {
                return ("trainium2".to_string(), 400.0);
            } else if lower.contains("trainium") || lower.contains("trn1") {
                return ("trainium1".to_string(), 210.0);
            } else if lower.contains("inferentia2") || lower.contains("inf2") {
                return ("inferentia2".to_string(), 115.0);
            }
        }

        // Check instance type for AWS classification
        if let Ok(instance_type) = std::env::var("AWS_INSTANCE_TYPE") {
            let lower = instance_type.to_lowercase();
            if lower.starts_with("trn2") {
                return ("trainium2".to_string(), 400.0);
            } else if lower.starts_with("trn1") {
                return ("trainium1".to_string(), 210.0);
            } else if lower.starts_with("inf2") {
                return ("inferentia2".to_string(), 115.0);
            }
        }

        // Default to Trainium1
        ("trainium1".to_string(), 210.0)
    }

    /// Get the detected chip variant
    pub fn chip_variant(&self) -> &str {
        &self.chip_variant
    }

    /// Get the TDP estimate in watts
    pub fn tdp_watts(&self) -> f64 {
        self.tdp_watts
    }
}

impl DomainReader for NeuronDomainReader {
    fn domain(&self) -> HardwareDomain {
        HardwareDomain::AwsNeuron
    }

    fn is_available(&self) -> bool {
        self.available
    }

    fn read_energy(&self) -> Result<f64> {
        // Tier 3 estimation: energy = TDP * elapsed_time
        let mut last = self.last_read.lock().unwrap();
        let mut accumulated = self.accumulated_energy_j.lock().unwrap();

        let now = Instant::now();
        let dt = now.duration_since(*last).as_secs_f64();
        *last = now;

        *accumulated += self.tdp_watts * dt;
        Ok(*accumulated)
    }
}

// ============================================================================
// Groq LPU Domain Reader (Tier 3 estimation)
// ============================================================================

/// Groq LPU energy reader using Tier 3 TDP-based estimation.
///
/// Always available (no libloading required). Detects Groq hardware via the
/// `GROQ_DEVICE` environment variable.
///
/// | Chip  | TDP (W) |
/// |-------|---------|
/// | LPU-1 | 300     |
pub struct GroqDomainReader {
    /// Chip variant
    chip_variant: String,
    /// Thermal Design Power in watts
    tdp_watts: f64,
    /// Whether a Groq device was detected
    available: bool,
    /// Accumulated energy from TDP estimation
    accumulated_energy_j: std::sync::Mutex<f64>,
    last_read: std::sync::Mutex<Instant>,
}

impl GroqDomainReader {
    /// Create a new Groq reader. Detects device presence via `GROQ_DEVICE` env var.
    pub fn new() -> Result<Self> {
        let available = std::env::var("GROQ_DEVICE").is_ok();

        let (chip_variant, tdp_watts) = Self::detect_variant();

        Ok(Self {
            chip_variant,
            tdp_watts,
            available,
            accumulated_energy_j: std::sync::Mutex::new(0.0),
            last_read: std::sync::Mutex::new(Instant::now()),
        })
    }

    /// Detect Groq chip variant from environment.
    fn detect_variant() -> (String, f64) {
        if let Ok(device) = std::env::var("GROQ_DEVICE") {
            let lower = device.to_lowercase();
            if lower.contains("lpu-2") || lower.contains("lpu2") {
                // Future-proof: if Groq releases LPU-2, estimate higher TDP
                return ("lpu-2".to_string(), 350.0);
            }
        }
        // Default: LPU-1
        ("lpu-1".to_string(), 300.0)
    }

    /// Get the detected chip variant
    pub fn chip_variant(&self) -> &str {
        &self.chip_variant
    }

    /// Get the TDP estimate in watts
    pub fn tdp_watts(&self) -> f64 {
        self.tdp_watts
    }
}

impl DomainReader for GroqDomainReader {
    fn domain(&self) -> HardwareDomain {
        HardwareDomain::GroqLpu
    }

    fn is_available(&self) -> bool {
        self.available
    }

    fn read_energy(&self) -> Result<f64> {
        // Tier 3 estimation: energy = TDP * elapsed_time
        let mut last = self.last_read.lock().unwrap();
        let mut accumulated = self.accumulated_energy_j.lock().unwrap();

        let now = Instant::now();
        let dt = now.duration_since(*last).as_secs_f64();
        *last = now;

        *accumulated += self.tdp_watts * dt;
        Ok(*accumulated)
    }
}

// ============================================================================
// Multi-Domain Monitor
// ============================================================================

/// Monitor that tracks energy across multiple hardware domains
pub struct MultiDomainMonitor {
    config: KinematicConfig,
    readers: HashMap<HardwareDomain, Box<dyn DomainReader>>,
    samples: HashMap<HardwareDomain, VecDeque<DomainSample>>,
    base_energy: HashMap<HardwareDomain, f64>,
    start_time: Option<Instant>,
    running: Arc<AtomicBool>,
    total_ops: u64,
    total_bytes: u64,
}

impl MultiDomainMonitor {
    /// Create a new multi-domain monitor with auto-detected domains
    pub fn new() -> Result<Self> {
        Self::with_config(KinematicConfig::default())
    }

    /// Create with specific configuration
    pub fn with_config(config: KinematicConfig) -> Result<Self> {
        let mut readers: HashMap<HardwareDomain, Box<dyn DomainReader>> = HashMap::new();

        // Try to add CPU reader
        if let Ok(cpu_reader) = CpuDomainReader::new() {
            if cpu_reader.is_available() {
                readers.insert(HardwareDomain::Cpu, Box::new(cpu_reader));
            }
        }

        // Try to add DRAM reader
        if let Ok(dram_reader) = DramDomainReader::new() {
            if dram_reader.is_available() {
                readers.insert(HardwareDomain::Dram, Box::new(dram_reader));
            }
        }

        // Try to add GPU reader
        if let Ok(gpu_reader) = GpuDomainReader::new(0) {
            if gpu_reader.is_available() {
                readers.insert(HardwareDomain::Gpu, Box::new(gpu_reader));
            }
        }

        // Try accelerator-specific readers (require libloading feature)
        #[cfg(feature = "accelerators")]
        {
            // Try NVML (NVIDIA GPUs) — prefer NVML over generic GpuDomainReader
            if let Ok(nvml) = NvmlDomainReader::new(0) {
                if nvml.is_available() {
                    readers.insert(nvml.domain(), Box::new(nvml));
                }
            }

            // Try ROCm SMI (AMD GPUs)
            #[cfg(target_os = "linux")]
            {
                if let Ok(rocm) = RocmDomainReader::new(0) {
                    if rocm.is_available() {
                        readers.insert(rocm.domain(), Box::new(rocm));
                    }
                }
            }

            // Try Level Zero Sysman (Intel XPUs)
            if let Ok(l0) = LevelZeroDomainReader::new(0) {
                if l0.is_available() {
                    readers.insert(l0.domain(), Box::new(l0));
                }
            }

            // Try HLML (Intel Gaudi)
            #[cfg(target_os = "linux")]
            {
                if let Ok(hlml) = HlmlDomainReader::new(0) {
                    if hlml.is_available() {
                        readers.insert(hlml.domain(), Box::new(hlml));
                    }
                }
            }
        }

        // Cloud accelerators — always available (no libloading needed)
        // TPU detection via /dev/accel0 or CLOUD_TPU_TASK_ID
        if let Ok(tpu) = TpuDomainReader::new() {
            if tpu.is_available() {
                readers.insert(tpu.domain(), Box::new(tpu));
            }
        }

        // AWS Neuron detection via /dev/neuron0 or NEURON_RT_VISIBLE_CORES
        if let Ok(neuron) = NeuronDomainReader::new() {
            if neuron.is_available() {
                readers.insert(neuron.domain(), Box::new(neuron));
            }
        }

        // Groq LPU detection via GROQ_DEVICE
        if let Ok(groq) = GroqDomainReader::new() {
            if groq.is_available() {
                readers.insert(groq.domain(), Box::new(groq));
            }
        }

        // Add estimated I/O domains
        readers.insert(
            HardwareDomain::Storage,
            Box::new(EstimatedDomainReader::storage()),
        );
        readers.insert(
            HardwareDomain::Network,
            Box::new(EstimatedDomainReader::network()),
        );

        Ok(Self {
            config,
            readers,
            samples: HashMap::new(),
            base_energy: HashMap::new(),
            start_time: None,
            running: Arc::new(AtomicBool::new(false)),
            total_ops: 0,
            total_bytes: 0,
        })
    }

    /// Get list of available domains
    pub fn available_domains(&self) -> Vec<HardwareDomain> {
        self.readers.keys().copied().collect()
    }

    /// Add a custom domain reader
    pub fn add_reader(&mut self, reader: Box<dyn DomainReader>) {
        let domain = reader.domain();
        self.readers.insert(domain, reader);
    }

    /// Start monitoring
    pub fn start(&mut self) -> Result<()> {
        if self.running.load(Ordering::SeqCst) {
            return Err(Error::AlreadyRunning);
        }

        self.start_time = Some(Instant::now());
        self.samples.clear();
        self.base_energy.clear();

        // Initialize base energy for each domain
        for (domain, reader) in &self.readers {
            if let Ok(energy) = reader.read_energy() {
                self.base_energy.insert(*domain, energy);
                self.samples.insert(*domain, VecDeque::new());
            }
        }

        self.running.store(true, Ordering::SeqCst);
        Ok(())
    }

    /// Take a sample from all domains
    pub fn sample(&mut self) -> Result<()> {
        let start_time = self.start_time.ok_or(Error::NotStarted)?;
        let time_s = start_time.elapsed().as_secs_f64();

        for (domain, reader) in &self.readers {
            let base = self.base_energy.get(domain).copied().unwrap_or(0.0);
            if let Ok(sample) = reader.sample(time_s, base) {
                if let Some(buffer) = self.samples.get_mut(domain) {
                    buffer.push_back(sample);

                    // Maintain buffer size
                    while buffer.len() > self.config.buffer_size {
                        buffer.pop_front();
                    }
                }
            }
        }

        Ok(())
    }

    /// Record operations (for efficiency metrics)
    pub fn record_ops(&mut self, ops: u64) {
        self.total_ops = self.total_ops.saturating_add(ops);
    }

    /// Record bytes processed
    pub fn record_bytes(&mut self, bytes: u64) {
        self.total_bytes = self.total_bytes.saturating_add(bytes);
    }

    /// Stop monitoring and compute telemetry
    pub fn stop(&mut self) -> Result<MultiDomainTelemetry> {
        self.running.store(false, Ordering::SeqCst);

        let start_time = self.start_time.ok_or(Error::NotStarted)?;
        let duration_s = start_time.elapsed().as_secs_f64();

        // Take final sample
        let _ = self.sample();

        let computer =
            DerivativeComputer::new(self.config.derivative_order, self.config.smoothing_window);

        let mut domains = HashMap::new();
        let mut total_energy = 0.0;
        let mut total_power = 0.0;
        let mut compute_power = 0.0;
        let mut memory_power = 0.0;
        let mut io_power = 0.0;

        // Process each domain
        for (domain, sample_buffer) in &self.samples {
            let samples: Vec<Sample> = sample_buffer
                .iter()
                .map(|ds| Sample {
                    time_s: ds.time_s,
                    energy_j: ds.energy_j,
                    temp_c: ds.temp_c,
                    ops: None,
                    bytes: None,
                })
                .collect();

            if samples.len() < 2 {
                continue;
            }

            let (energy_chain, temp_chain) = computer.compute_derivatives(&samples);
            let energy = EnergyDerivatives::from_chain(&energy_chain);
            let thermal = ThermalDerivatives::from_chain(&temp_chain);
            let thermodynamics = ThermodynamicCoupling::compute(&energy, &thermal);

            // Compute utilization/frequency/bandwidth stats from domain samples
            let utilization = Self::compute_utilization_stats(sample_buffer);
            let frequency = Self::compute_frequency_stats(sample_buffer);
            let bandwidth = Self::compute_bandwidth_stats(sample_buffer, energy.energy_j);

            let domain_telemetry = DomainTelemetry {
                domain: Some(*domain),
                duration_s,
                sample_count: samples.len(),
                energy,
                thermal,
                thermodynamics,
                utilization,
                frequency,
                bandwidth,
            };

            // Aggregate power
            total_energy += energy.energy_j;
            total_power += energy.power_w;

            if domain.is_compute() {
                compute_power += energy.power_w;
            } else if domain.is_memory() {
                memory_power += energy.power_w;
            } else if domain.is_io() {
                io_power += energy.power_w;
            }

            domains.insert(*domain, domain_telemetry);
        }

        // Compute efficiency
        let efficiency = EfficiencyMetrics::compute(
            total_energy,
            duration_s,
            self.total_ops,
            self.total_bytes,
            total_power,
        );

        // Compute correlations
        let correlations = Self::compute_correlations(&domains);

        Ok(MultiDomainTelemetry {
            duration_s,
            domains,
            system: SystemAggregate {
                total_energy_j: total_energy,
                total_power_w: total_power,
                peak_power_w: total_power * 1.2, // Placeholder
                min_power_w: total_power * 0.8,  // Placeholder
                compute_power_w: compute_power,
                memory_power_w: memory_power,
                io_power_w: io_power,
                efficiency,
            },
            correlations,
        })
    }

    fn compute_utilization_stats(samples: &VecDeque<DomainSample>) -> UtilizationStats {
        let utils: Vec<f64> = samples.iter().filter_map(|s| s.utilization).collect();
        if utils.is_empty() {
            return UtilizationStats::default();
        }

        let min = utils.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = utils.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean = utils.iter().sum::<f64>() / utils.len() as f64;

        UtilizationStats {
            min,
            max,
            mean,
            time_weighted_mean: mean, // Simplified
        }
    }

    fn compute_frequency_stats(samples: &VecDeque<DomainSample>) -> FrequencyStats {
        let freqs: Vec<f64> = samples.iter().filter_map(|s| s.frequency_hz).collect();
        if freqs.is_empty() {
            return FrequencyStats::default();
        }

        let min = freqs.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = freqs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean = freqs.iter().sum::<f64>() / freqs.len() as f64;

        FrequencyStats {
            min_hz: min,
            max_hz: max,
            mean_hz: mean,
            energy_weighted_mean_hz: mean, // Simplified
        }
    }

    fn compute_bandwidth_stats(
        samples: &VecDeque<DomainSample>,
        total_energy: f64,
    ) -> BandwidthStats {
        let bws: Vec<f64> = samples
            .iter()
            .filter_map(|s| s.bandwidth_bytes_per_s)
            .collect();
        if bws.is_empty() {
            return BandwidthStats::default();
        }

        let peak = bws.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean = bws.iter().sum::<f64>() / bws.len() as f64;
        let total_bytes = mean * samples.len() as f64 * 0.01; // Rough estimate

        let energy_per_byte = if total_bytes > 0.0 {
            total_energy / total_bytes
        } else {
            0.0
        };

        BandwidthStats {
            peak_bytes_per_s: peak,
            mean_bytes_per_s: mean,
            total_bytes,
            energy_per_byte_j: energy_per_byte,
        }
    }

    fn compute_correlations(
        _domains: &HashMap<HardwareDomain, DomainTelemetry>,
    ) -> DomainCorrelations {
        // Placeholder - in a full implementation, compute actual correlations
        DomainCorrelations::default()
    }
}

// ============================================================================
// Display Implementations
// ============================================================================

impl std::fmt::Display for MultiDomainTelemetry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "╔══════════════════════════════════════════════════════════════════════════╗"
        )?;
        writeln!(
            f,
            "║                    MULTI-DOMAIN ENERGY TELEMETRY                         ║"
        )?;
        writeln!(
            f,
            "╠══════════════════════════════════════════════════════════════════════════╣"
        )?;
        writeln!(
            f,
            "║  Duration: {:.3} s | Domains: {}",
            self.duration_s,
            self.domains.len()
        )?;
        writeln!(
            f,
            "╠══════════════════════════════════════════════════════════════════════════╣"
        )?;
        writeln!(
            f,
            "║  SYSTEM AGGREGATE                                                        ║"
        )?;
        writeln!(
            f,
            "║    Total Energy:  {:>12.6} J",
            self.system.total_energy_j
        )?;
        writeln!(
            f,
            "║    Total Power:   {:>12.3} W",
            self.system.total_power_w
        )?;
        writeln!(
            f,
            "║    Compute Power: {:>12.3} W ({:.1}%)",
            self.system.compute_power_w,
            if self.system.total_power_w > 0.0 {
                self.system.compute_power_w / self.system.total_power_w * 100.0
            } else {
                0.0
            }
        )?;
        writeln!(
            f,
            "║    Memory Power:  {:>12.3} W ({:.1}%)",
            self.system.memory_power_w,
            if self.system.total_power_w > 0.0 {
                self.system.memory_power_w / self.system.total_power_w * 100.0
            } else {
                0.0
            }
        )?;
        writeln!(
            f,
            "║    I/O Power:     {:>12.3} W ({:.1}%)",
            self.system.io_power_w,
            if self.system.total_power_w > 0.0 {
                self.system.io_power_w / self.system.total_power_w * 100.0
            } else {
                0.0
            }
        )?;
        writeln!(
            f,
            "╠══════════════════════════════════════════════════════════════════════════╣"
        )?;
        writeln!(
            f,
            "║  PER-DOMAIN BREAKDOWN                                                    ║"
        )?;

        for (domain, telemetry) in &self.domains {
            writeln!(
                f,
                "║  ┌─ {:15} ─────────────────────────────────────────────────┐",
                domain.name()
            )?;
            writeln!(
                f,
                "║  │  Energy: {:>10.6} J  Power: {:>8.3} W  Jerk: {:>10.3} W/s²",
                telemetry.energy.energy_j,
                telemetry.energy.power_w,
                telemetry.energy.power_jerk_w_per_s2
            )?;
            if telemetry.thermal.temp_c > 0.0 {
                writeln!(
                    f,
                    "║  │  Temp: {:>6.1} °C  Heating: {:>8.3} °C/s",
                    telemetry.thermal.temp_c, telemetry.thermal.heating_rate_c_per_s
                )?;
            }
            writeln!(
                f,
                "║  └──────────────────────────────────────────────────────────────────┘"
            )?;
        }

        writeln!(
            f,
            "╚══════════════════════════════════════════════════════════════════════════╝"
        )
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_domain() {
        assert_eq!(HardwareDomain::Cpu.name(), "CPU");
        assert_eq!(HardwareDomain::CpuCore(3).name(), "CPU Core 3");
        assert_eq!(HardwareDomain::Gpu.id(), "gpu");
        assert!(HardwareDomain::Cpu.is_compute());
        assert!(HardwareDomain::Dram.is_memory());
        assert!(HardwareDomain::Storage.is_io());
    }

    #[test]
    fn test_domain_sample() {
        let sample = DomainSample::new(1.0, 0.5)
            .with_temp(65.0)
            .with_utilization(0.75)
            .with_frequency(3_500_000_000.0);

        assert_eq!(sample.time_s, 1.0);
        assert_eq!(sample.energy_j, 0.5);
        assert_eq!(sample.temp_c, Some(65.0));
        assert_eq!(sample.utilization, Some(0.75));
        assert_eq!(sample.frequency_hz, Some(3_500_000_000.0));
    }

    #[test]
    fn test_estimated_reader() {
        let storage = EstimatedDomainReader::storage();
        assert_eq!(storage.domain(), HardwareDomain::Storage);
        assert!(storage.is_available());

        let network = EstimatedDomainReader::network();
        assert_eq!(network.domain(), HardwareDomain::Network);
    }

    #[test]
    fn test_new_hardware_domain_variants() {
        // Names
        assert_eq!(HardwareDomain::IntelXpu.name(), "Intel XPU");
        assert_eq!(HardwareDomain::IntelXpuDevice(1).name(), "Intel XPU 1");
        assert_eq!(HardwareDomain::IntelGaudi.name(), "Intel Gaudi");
        assert_eq!(HardwareDomain::IntelGaudiDevice(2).name(), "Intel Gaudi 2");
        assert_eq!(HardwareDomain::GroqLpu.name(), "Groq LPU");
        assert_eq!(HardwareDomain::AwsNeuron.name(), "AWS Neuron");
        assert_eq!(HardwareDomain::AwsNeuronDevice(0).name(), "AWS Neuron 0");
        assert_eq!(HardwareDomain::CerebrasWse.name(), "Cerebras WSE");
        assert_eq!(HardwareDomain::SambaNovaRdu.name(), "SambaNova RDU");

        // IDs
        assert_eq!(HardwareDomain::IntelXpu.id(), "intel_xpu");
        assert_eq!(HardwareDomain::IntelGaudi.id(), "intel_gaudi");
        assert_eq!(HardwareDomain::GroqLpu.id(), "groq_lpu");
        assert_eq!(HardwareDomain::AwsNeuron.id(), "aws_neuron");
        assert_eq!(HardwareDomain::CerebrasWse.id(), "cerebras_wse");
        assert_eq!(HardwareDomain::SambaNovaRdu.id(), "sambanova_rdu");

        // All new accelerator domains are compute
        assert!(HardwareDomain::IntelXpu.is_compute());
        assert!(HardwareDomain::IntelXpuDevice(0).is_compute());
        assert!(HardwareDomain::IntelGaudi.is_compute());
        assert!(HardwareDomain::IntelGaudiDevice(0).is_compute());
        assert!(HardwareDomain::GroqLpu.is_compute());
        assert!(HardwareDomain::AwsNeuron.is_compute());
        assert!(HardwareDomain::AwsNeuronDevice(0).is_compute());
        assert!(HardwareDomain::CerebrasWse.is_compute());
        assert!(HardwareDomain::SambaNovaRdu.is_compute());

        // Not memory or I/O
        assert!(!HardwareDomain::IntelXpu.is_memory());
        assert!(!HardwareDomain::GroqLpu.is_io());
    }

    #[test]
    fn test_tpu_reader_not_available() {
        // On dev machines without TPU hardware, the reader should construct but
        // report not available (unless env vars are set).
        let tpu = TpuDomainReader::new().unwrap();
        assert_eq!(tpu.domain(), HardwareDomain::Tpu);
        // Default generation when env var not set
        assert_eq!(tpu.generation(), "unknown");
        assert!((tpu.tdp_watts() - 150.0).abs() < 0.01);
    }

    #[test]
    fn test_neuron_reader_not_available() {
        let neuron = NeuronDomainReader::new().unwrap();
        assert_eq!(neuron.domain(), HardwareDomain::AwsNeuron);
        // Default variant when env var not set
        assert_eq!(neuron.chip_variant(), "trainium1");
        assert!((neuron.tdp_watts() - 210.0).abs() < 0.01);
    }

    #[test]
    fn test_groq_reader_not_available() {
        let groq = GroqDomainReader::new().unwrap();
        assert_eq!(groq.domain(), HardwareDomain::GroqLpu);
        assert_eq!(groq.chip_variant(), "lpu-1");
        assert!((groq.tdp_watts() - 300.0).abs() < 0.01);
        // Should not be available without GROQ_DEVICE env var
        assert!(!groq.is_available());
    }

    #[test]
    fn test_tier3_energy_estimation() {
        // Verify that Tier 3 readers accumulate energy based on TDP * time
        let groq = GroqDomainReader::new().unwrap();
        // First read initializes the timer
        let e1 = groq.read_energy().unwrap();
        // Second read should accumulate some (tiny) amount of energy
        let e2 = groq.read_energy().unwrap();
        // Energy should be non-negative and non-decreasing
        assert!(e2 >= e1);
    }
}
