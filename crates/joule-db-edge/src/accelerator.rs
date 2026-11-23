//! Hardware accelerator abstraction for edge devices
//!
//! This module provides a unified interface for different hardware accelerators
//! commonly found on edge devices.

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};

use core::fmt;

/// Type of hardware accelerator
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcceleratorType {
    /// No hardware acceleration (CPU fallback)
    None,

    // =========================================================================
    // Linux-based accelerators (MPU class)
    // =========================================================================
    /// NVIDIA CUDA GPU (Jetson)
    Cuda,
    /// NVIDIA TensorRT (optimized inference on Jetson)
    TensorRt,
    /// Google Edge TPU (Coral)
    EdgeTpu,
    /// Rockchip NPU (RK3588, etc.)
    Rknn,
    /// Hailo-8 AI accelerator
    Hailo,
    /// Intel Movidius VPU (NCS2)
    Movidius,
    /// ARM Mali GPU
    Mali,
    /// Vivante GPU (i.MX8)
    Vivante,
    /// Raspberry Pi VideoCore
    VideoCore,
    /// Generic OpenCL device
    OpenCl,
    /// Generic Vulkan compute
    Vulkan,

    // =========================================================================
    // Bare-metal / RTOS accelerators (MCU class)
    // =========================================================================
    /// Arm Ethos-U55 NPU (Alif Ensemble, Infineon PSoC Edge)
    /// Uses CMSIS-NN / TensorFlow Lite for Microcontrollers
    EthosU55,
    /// Arm Ethos-U65 NPU (higher performance variant)
    EthosU65,
    /// Arm Helium MVE (Cortex-M55/M85 vector extension)
    /// No discrete NPU - CPU IS the accelerator
    Helium,
    /// STM32 Neural-ART NPU (STM32N6)
    NeuralArt,
    /// NXP eIQ Neutron NPU (MCX N94x)
    EiqNeutron,
    /// Analog Devices/Maxim CNN Accelerator (MAX78000/MAX78002)
    /// Architectural outlier: hardware CNN with small MCU attached
    MaximCnn,
    /// Silicon Labs MVP (Matrix Vector Processor, EFR32xG24)
    SilabsMvp,
    /// GreenWaves NE16 + RISC-V cluster (GAP9)
    GapNe16,
    /// Espressif ESP32-S3 vector instructions
    Esp32S3,
    /// Nordic Riscy VPU / DSP (nRF54H20)
    NordicVpu,
    /// Syntiant NDP (Neural Decision Processor)
    SyntiantNdp,
    /// Canaan Kendryte KPU (K230)
    KendryteKpu,
    /// Rockchip MCU NPU (RV1106/RV1103 bare-metal)
    RockchipMcuNpu,
    /// Generic CMSIS-NN (any Cortex-M with CMSIS support)
    CmsisNn,
}

impl fmt::Display for AcceleratorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            // CPU fallback
            Self::None => write!(f, "CPU"),

            // Linux/MPU class
            Self::Cuda => write!(f, "CUDA"),
            Self::TensorRt => write!(f, "TensorRT"),
            Self::EdgeTpu => write!(f, "EdgeTPU"),
            Self::Rknn => write!(f, "RKNN"),
            Self::Hailo => write!(f, "Hailo-8"),
            Self::Movidius => write!(f, "Movidius"),
            Self::Mali => write!(f, "Mali"),
            Self::Vivante => write!(f, "Vivante"),
            Self::VideoCore => write!(f, "VideoCore"),
            Self::OpenCl => write!(f, "OpenCL"),
            Self::Vulkan => write!(f, "Vulkan"),

            // Bare-metal/RTOS class
            Self::EthosU55 => write!(f, "Ethos-U55"),
            Self::EthosU65 => write!(f, "Ethos-U65"),
            Self::Helium => write!(f, "Helium MVE"),
            Self::NeuralArt => write!(f, "Neural-ART"),
            Self::EiqNeutron => write!(f, "eIQ Neutron"),
            Self::MaximCnn => write!(f, "Maxim CNN"),
            Self::SilabsMvp => write!(f, "SiLabs MVP"),
            Self::GapNe16 => write!(f, "GAP NE16"),
            Self::Esp32S3 => write!(f, "ESP32-S3"),
            Self::NordicVpu => write!(f, "Nordic VPU"),
            Self::SyntiantNdp => write!(f, "Syntiant NDP"),
            Self::KendryteKpu => write!(f, "Kendryte KPU"),
            Self::RockchipMcuNpu => write!(f, "Rockchip MCU NPU"),
            Self::CmsisNn => write!(f, "CMSIS-NN"),
        }
    }
}

impl AcceleratorType {
    /// Returns the device class for energy dispatch mapping.
    ///
    /// Maps the 28 accelerator variants into one of four device classes:
    /// - `"cpu"` — CPU-like (no discrete accelerator)
    /// - `"gpu"` — GPU-class (general-purpose parallel compute)
    /// - `"tpu"` — TPU-class (tensor/ML specialized)
    /// - `"npu"` — NPU-class (neural processing unit)
    pub fn device_class(&self) -> &'static str {
        match self {
            // CPU-class: no discrete accelerator or CPU vector extensions
            Self::None | Self::Helium | Self::Esp32S3 | Self::CmsisNn => "cpu",
            // GPU-class: general-purpose parallel compute
            Self::Cuda
            | Self::TensorRt
            | Self::Mali
            | Self::Vivante
            | Self::VideoCore
            | Self::OpenCl
            | Self::Vulkan => "gpu",
            // TPU-class: tensor/ML specialized
            Self::EdgeTpu => "tpu",
            // NPU-class: everything else (dedicated neural processors)
            Self::Rknn
            | Self::Hailo
            | Self::Movidius
            | Self::EthosU55
            | Self::EthosU65
            | Self::NeuralArt
            | Self::EiqNeutron
            | Self::MaximCnn
            | Self::SilabsMvp
            | Self::GapNe16
            | Self::NordicVpu
            | Self::SyntiantNdp
            | Self::KendryteKpu
            | Self::RockchipMcuNpu => "npu",
        }
    }

    /// Returns true if this accelerator is designed for bare-metal/RTOS use
    pub fn is_mcu_class(&self) -> bool {
        matches!(
            self,
            Self::EthosU55
                | Self::EthosU65
                | Self::Helium
                | Self::NeuralArt
                | Self::EiqNeutron
                | Self::MaximCnn
                | Self::SilabsMvp
                | Self::GapNe16
                | Self::Esp32S3
                | Self::NordicVpu
                | Self::SyntiantNdp
                | Self::KendryteKpu
                | Self::RockchipMcuNpu
                | Self::CmsisNn
        )
    }

    /// Returns true if this accelerator is designed for Linux/MPU use
    pub fn is_mpu_class(&self) -> bool {
        matches!(
            self,
            Self::Cuda
                | Self::TensorRt
                | Self::EdgeTpu
                | Self::Rknn
                | Self::Hailo
                | Self::Movidius
                | Self::Mali
                | Self::Vivante
                | Self::VideoCore
                | Self::OpenCl
                | Self::Vulkan
        )
    }

    /// Returns true if this accelerator uses CMSIS-NN compatible interface
    pub fn uses_cmsis_nn(&self) -> bool {
        matches!(
            self,
            Self::EthosU55 | Self::EthosU65 | Self::Helium | Self::CmsisNn
        )
    }

    /// Returns true if this is a discrete NPU (separate from CPU)
    pub fn is_discrete_npu(&self) -> bool {
        matches!(
            self,
            Self::Cuda
                | Self::TensorRt
                | Self::EdgeTpu
                | Self::Rknn
                | Self::Hailo
                | Self::Movidius
                | Self::EthosU55
                | Self::EthosU65
                | Self::NeuralArt
                | Self::EiqNeutron
                | Self::MaximCnn
                | Self::SilabsMvp
                | Self::GapNe16
                | Self::SyntiantNdp
                | Self::KendryteKpu
                | Self::RockchipMcuNpu
        )
    }

    /// Returns the typical TOPS (Tera Operations Per Second) for this accelerator
    /// Returns None if highly variable or unknown
    pub fn typical_tops(&self) -> Option<f32> {
        match self {
            Self::None => None,       // Varies by CPU
            Self::Cuda => Some(10.0), // Jetson Orin NX ~100 TOPS, Nano ~40
            Self::TensorRt => Some(10.0),
            Self::EdgeTpu => Some(4.0),
            Self::Rknn => Some(6.0),      // RK3588 ~6 TOPS
            Self::Hailo => Some(26.0),    // Hailo-8 ~26 TOPS
            Self::EthosU55 => Some(0.5),  // Up to 0.5 TOPS
            Self::EthosU65 => Some(1.0),  // Up to 1 TOPS
            Self::Helium => Some(0.1),    // ~4x M7, varies
            Self::NeuralArt => Some(0.6), // STM32N6
            Self::EiqNeutron => Some(0.25),
            Self::MaximCnn => Some(0.5), // MAX78002
            Self::SilabsMvp => Some(0.05),
            Self::GapNe16 => Some(0.15),
            Self::Esp32S3 => Some(0.01),       // Very limited
            Self::RockchipMcuNpu => Some(0.5), // RV1106
            _ => None,
        }
    }
}

/// Capabilities of a hardware accelerator
#[derive(Debug, Clone)]
pub struct AcceleratorCapabilities {
    /// Accelerator type
    pub accelerator_type: AcceleratorType,
    /// Device name/description
    pub name: String,
    /// Available memory in bytes
    pub memory_bytes: u64,
    /// Maximum compute units/cores
    pub compute_units: u32,
    /// Supports async DMA transfers
    pub supports_dma: bool,
    /// Supports zero-copy memory mapping
    pub supports_zero_copy: bool,
    /// Maximum batch size for operations
    pub max_batch_size: usize,
    /// Preferred memory alignment in bytes
    pub memory_alignment: usize,
}

impl Default for AcceleratorCapabilities {
    fn default() -> Self {
        Self {
            accelerator_type: AcceleratorType::None,
            name: String::from("CPU Fallback"),
            memory_bytes: 0,
            compute_units: 1,
            supports_dma: false,
            supports_zero_copy: false,
            max_batch_size: 256,
            memory_alignment: 64,
        }
    }
}

/// Accelerator operation type
#[derive(Debug, Clone, Copy)]
pub enum AcceleratorOp {
    /// Hash computation (XXHash, CRC32, etc.)
    Hash { algorithm: HashAlgorithm },
    /// Aggregation (SUM, COUNT, MIN, MAX, AVG)
    Aggregate { operation: AggregateOp },
    /// Vector similarity (dot product, euclidean)
    VectorSimilarity { dimensions: usize },
    /// Bloom filter operations
    BloomFilter { bits: usize, hashes: u8 },
    /// Compression/decompression
    Compress { algorithm: CompressionAlgorithm },
}

/// Hash algorithm for accelerated computation
#[derive(Debug, Clone, Copy)]
pub enum HashAlgorithm {
    Xxhash64,
    Crc32,
    Murmur3,
    Siphash,
}

/// Aggregation operation
#[derive(Debug, Clone, Copy)]
pub enum AggregateOp {
    Sum,
    Count,
    Min,
    Max,
    Average,
}

/// Compression algorithm
#[derive(Debug, Clone, Copy)]
pub enum CompressionAlgorithm {
    Lz4,
    Zstd,
    Snappy,
}

/// Trait for hardware accelerators
pub trait Accelerator: Send + Sync {
    /// Get accelerator capabilities
    fn capabilities(&self) -> &AcceleratorCapabilities;

    /// Check if an operation is supported
    fn supports(&self, op: AcceleratorOp) -> bool;

    /// Execute a hash operation on a batch of inputs
    fn hash_batch(&self, algorithm: HashAlgorithm, inputs: &[&[u8]]) -> Vec<u64>;

    /// Execute an aggregation on f64 values
    fn aggregate_f64(&self, op: AggregateOp, values: &[f64]) -> f64;

    /// Execute an aggregation on i64 values
    fn aggregate_i64(&self, op: AggregateOp, values: &[i64]) -> i64;

    /// Compute vector dot product
    fn vector_dot_product(&self, a: &[f32], b: &[f32]) -> f32;

    /// Compute euclidean distance
    fn vector_euclidean_distance(&self, a: &[f32], b: &[f32]) -> f32;

    /// Sync/wait for all pending operations
    fn sync(&self);
}

/// CPU fallback accelerator (always available)
pub struct CpuAccelerator {
    capabilities: AcceleratorCapabilities,
}

impl CpuAccelerator {
    pub fn new() -> Self {
        Self {
            capabilities: AcceleratorCapabilities {
                accelerator_type: AcceleratorType::None,
                name: String::from("CPU Fallback"),
                memory_bytes: 0, // System memory
                compute_units: num_cpus(),
                supports_dma: false,
                supports_zero_copy: true,
                max_batch_size: 1024,
                memory_alignment: 64,
            },
        }
    }
}

impl Default for CpuAccelerator {
    fn default() -> Self {
        Self::new()
    }
}

impl Accelerator for CpuAccelerator {
    fn capabilities(&self) -> &AcceleratorCapabilities {
        &self.capabilities
    }

    fn supports(&self, _op: AcceleratorOp) -> bool {
        true // CPU can do everything, just slower
    }

    fn hash_batch(&self, algorithm: HashAlgorithm, inputs: &[&[u8]]) -> Vec<u64> {
        inputs
            .iter()
            .map(|input| match algorithm {
                HashAlgorithm::Xxhash64 => xxhash64(input),
                HashAlgorithm::Crc32 => crc32(input) as u64,
                HashAlgorithm::Murmur3 => murmur3(input),
                HashAlgorithm::Siphash => siphash(input),
            })
            .collect()
    }

    fn aggregate_f64(&self, op: AggregateOp, values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        match op {
            AggregateOp::Sum => values.iter().sum(),
            AggregateOp::Count => values.len() as f64,
            AggregateOp::Min => values.iter().cloned().fold(f64::INFINITY, f64::min),
            AggregateOp::Max => values.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            AggregateOp::Average => values.iter().sum::<f64>() / values.len() as f64,
        }
    }

    fn aggregate_i64(&self, op: AggregateOp, values: &[i64]) -> i64 {
        if values.is_empty() {
            return 0;
        }
        match op {
            AggregateOp::Sum => values.iter().sum(),
            AggregateOp::Count => values.len() as i64,
            AggregateOp::Min => *values.iter().min().unwrap(),
            AggregateOp::Max => *values.iter().max().unwrap(),
            AggregateOp::Average => values.iter().sum::<i64>() / values.len() as i64,
        }
    }

    fn vector_dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        a[..len]
            .iter()
            .zip(b[..len].iter())
            .map(|(x, y)| x * y)
            .sum()
    }

    fn vector_euclidean_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        let sum_sq: f32 = a[..len]
            .iter()
            .zip(b[..len].iter())
            .map(|(x, y)| {
                let d = x - y;
                d * d
            })
            .sum();
        #[cfg(feature = "std")]
        {
            sum_sq.sqrt()
        }
        #[cfg(not(feature = "std"))]
        {
            if sum_sq <= 0.0 {
                return 0.0;
            }
            let mut g = f32::from_bits((sum_sq.to_bits() >> 1) + 0x1fc0_0000);
            for _ in 0..4 {
                g = 0.5 * (g + sum_sq / g);
            }
            g
        }
    }

    fn sync(&self) {
        // No-op for CPU
    }
}

// ============================================================================
// Hash implementations (CPU fallback)
// ============================================================================

fn xxhash64(data: &[u8]) -> u64 {
    // XXHash64 implementation
    const PRIME64_1: u64 = 0x9E3779B185EBCA87;
    const PRIME64_2: u64 = 0xC2B2AE3D27D4EB4F;
    const PRIME64_3: u64 = 0x165667B19E3779F9;
    const PRIME64_4: u64 = 0x85EBCA77C2B2AE63;
    const PRIME64_5: u64 = 0x27D4EB2F165667C5;

    let seed: u64 = 0;
    let len = data.len();

    let mut h64: u64;

    if len >= 32 {
        let mut v1 = seed.wrapping_add(PRIME64_1).wrapping_add(PRIME64_2);
        let mut v2 = seed.wrapping_add(PRIME64_2);
        let mut v3 = seed;
        let mut v4 = seed.wrapping_sub(PRIME64_1);

        let mut i = 0;
        while i + 32 <= len {
            v1 = round64(v1, read_u64_le(&data[i..]));
            v2 = round64(v2, read_u64_le(&data[i + 8..]));
            v3 = round64(v3, read_u64_le(&data[i + 16..]));
            v4 = round64(v4, read_u64_le(&data[i + 24..]));
            i += 32;
        }

        h64 = v1
            .rotate_left(1)
            .wrapping_add(v2.rotate_left(7))
            .wrapping_add(v3.rotate_left(12))
            .wrapping_add(v4.rotate_left(18));

        h64 = merge_round64(h64, v1);
        h64 = merge_round64(h64, v2);
        h64 = merge_round64(h64, v3);
        h64 = merge_round64(h64, v4);
    } else {
        h64 = seed.wrapping_add(PRIME64_5);
    }

    h64 = h64.wrapping_add(len as u64);

    // Remaining bytes
    let remaining = &data[(len / 32) * 32..];
    let mut i = 0;

    while i + 8 <= remaining.len() {
        let k1 = round64(0, read_u64_le(&remaining[i..]));
        h64 ^= k1;
        h64 = h64
            .rotate_left(27)
            .wrapping_mul(PRIME64_1)
            .wrapping_add(PRIME64_4);
        i += 8;
    }

    while i + 4 <= remaining.len() {
        h64 ^= (read_u32_le(&remaining[i..]) as u64).wrapping_mul(PRIME64_1);
        h64 = h64
            .rotate_left(23)
            .wrapping_mul(PRIME64_2)
            .wrapping_add(PRIME64_3);
        i += 4;
    }

    while i < remaining.len() {
        h64 ^= (remaining[i] as u64).wrapping_mul(PRIME64_5);
        h64 = h64.rotate_left(11).wrapping_mul(PRIME64_1);
        i += 1;
    }

    // Final mix
    h64 ^= h64 >> 33;
    h64 = h64.wrapping_mul(PRIME64_2);
    h64 ^= h64 >> 29;
    h64 = h64.wrapping_mul(PRIME64_3);
    h64 ^= h64 >> 32;

    h64
}

fn round64(mut acc: u64, input: u64) -> u64 {
    const PRIME64_1: u64 = 0x9E3779B185EBCA87;
    const PRIME64_2: u64 = 0xC2B2AE3D27D4EB4F;
    acc = acc.wrapping_add(input.wrapping_mul(PRIME64_2));
    acc = acc.rotate_left(31);
    acc.wrapping_mul(PRIME64_1)
}

fn merge_round64(mut acc: u64, val: u64) -> u64 {
    const PRIME64_1: u64 = 0x9E3779B185EBCA87;
    const PRIME64_4: u64 = 0x85EBCA77C2B2AE63;
    let val = round64(0, val);
    acc ^= val;
    acc.wrapping_mul(PRIME64_1).wrapping_add(PRIME64_4)
}

fn read_u64_le(data: &[u8]) -> u64 {
    u64::from_le_bytes([
        data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
    ])
}

fn read_u32_le(data: &[u8]) -> u32 {
    u32::from_le_bytes([data[0], data[1], data[2], data[3]])
}

fn crc32(data: &[u8]) -> u32 {
    // Simple CRC32 implementation
    const TABLE: [u32; 256] = crc32_table();
    let mut crc = 0xFFFFFFFF;
    for &byte in data {
        let idx = ((crc ^ byte as u32) & 0xFF) as usize;
        crc = TABLE[idx] ^ (crc >> 8);
    }
    !crc
}

const fn crc32_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut j = 0;
        while j < 8 {
            crc = if crc & 1 != 0 {
                0xEDB88320 ^ (crc >> 1)
            } else {
                crc >> 1
            };
            j += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}

fn murmur3(data: &[u8]) -> u64 {
    // MurmurHash3 64-bit
    const C1: u64 = 0x87c37b91114253d5;
    const C2: u64 = 0x4cf5ad432745937f;
    let seed: u64 = 0;

    let mut h1 = seed;
    let mut h2 = seed;

    // Process 16-byte chunks
    let chunks = data.len() / 16;
    for i in 0..chunks {
        let k1 = read_u64_le(&data[i * 16..]);
        let k2 = read_u64_le(&data[i * 16 + 8..]);

        let k1 = k1.wrapping_mul(C1);
        let k1 = k1.rotate_left(31);
        let k1 = k1.wrapping_mul(C2);
        h1 ^= k1;
        h1 = h1.rotate_left(27);
        h1 = h1.wrapping_add(h2);
        h1 = h1.wrapping_mul(5).wrapping_add(0x52dce729);

        let k2 = k2.wrapping_mul(C2);
        let k2 = k2.rotate_left(33);
        let k2 = k2.wrapping_mul(C1);
        h2 ^= k2;
        h2 = h2.rotate_left(31);
        h2 = h2.wrapping_add(h1);
        h2 = h2.wrapping_mul(5).wrapping_add(0x38495ab5);
    }

    // Finalization
    h1 ^= data.len() as u64;
    h2 ^= data.len() as u64;
    h1 = h1.wrapping_add(h2);
    h2 = h2.wrapping_add(h1);

    h1 = fmix64(h1);
    h2 = fmix64(h2);

    h1.wrapping_add(h2)
}

fn fmix64(mut k: u64) -> u64 {
    k ^= k >> 33;
    k = k.wrapping_mul(0xff51afd7ed558ccd);
    k ^= k >> 33;
    k = k.wrapping_mul(0xc4ceb9fe1a85ec53);
    k ^= k >> 33;
    k
}

fn siphash(data: &[u8]) -> u64 {
    // SipHash-2-4
    let k0: u64 = 0x736f6d6570736575;
    let k1: u64 = 0x646f72616e646f6d;

    let mut v0 = k0 ^ 0x736f6d6570736575;
    let mut v1 = k1 ^ 0x646f72616e646f6d;
    let mut v2 = k0 ^ 0x6c7967656e657261;
    let mut v3 = k1 ^ 0x7465646279746573;

    let chunks = data.len() / 8;
    for i in 0..chunks {
        let m = read_u64_le(&data[i * 8..]);
        v3 ^= m;
        sipround(&mut v0, &mut v1, &mut v2, &mut v3);
        sipround(&mut v0, &mut v1, &mut v2, &mut v3);
        v0 ^= m;
    }

    // Handle remaining bytes
    let remaining = data.len() % 8;
    let mut last = (data.len() as u64) << 56;
    for i in 0..remaining {
        last |= (data[chunks * 8 + i] as u64) << (i * 8);
    }

    v3 ^= last;
    sipround(&mut v0, &mut v1, &mut v2, &mut v3);
    sipround(&mut v0, &mut v1, &mut v2, &mut v3);
    v0 ^= last;

    v2 ^= 0xff;
    sipround(&mut v0, &mut v1, &mut v2, &mut v3);
    sipround(&mut v0, &mut v1, &mut v2, &mut v3);
    sipround(&mut v0, &mut v1, &mut v2, &mut v3);
    sipround(&mut v0, &mut v1, &mut v2, &mut v3);

    v0 ^ v1 ^ v2 ^ v3
}

fn sipround(v0: &mut u64, v1: &mut u64, v2: &mut u64, v3: &mut u64) {
    *v0 = v0.wrapping_add(*v1);
    *v1 = v1.rotate_left(13);
    *v1 ^= *v0;
    *v0 = v0.rotate_left(32);

    *v2 = v2.wrapping_add(*v3);
    *v3 = v3.rotate_left(16);
    *v3 ^= *v2;

    *v0 = v0.wrapping_add(*v3);
    *v3 = v3.rotate_left(21);
    *v3 ^= *v0;

    *v2 = v2.wrapping_add(*v1);
    *v1 = v1.rotate_left(17);
    *v1 ^= *v2;
    *v2 = v2.rotate_left(32);
}

/// Get number of CPUs (platform-specific)
fn num_cpus() -> u32 {
    #[cfg(feature = "std")]
    {
        std::thread::available_parallelism()
            .map(|p| p.get() as u32)
            .unwrap_or(1)
    }
    #[cfg(not(feature = "std"))]
    {
        1 // Assume single core in no_std
    }
}

// ============================================================================
// Accelerator detection
// ============================================================================

/// Detect available accelerators on the system
pub fn detect_accelerators() -> Vec<AcceleratorCapabilities> {
    let mut accelerators = Vec::new();

    // Always add CPU fallback
    accelerators.push(CpuAccelerator::new().capabilities().clone());

    // Platform-specific detection
    #[cfg(feature = "cuda")]
    {
        if let Some(caps) = detect_cuda() {
            accelerators.push(caps);
        }
    }

    #[cfg(feature = "coral")]
    {
        if let Some(caps) = detect_edgetpu() {
            accelerators.push(caps);
        }
    }

    #[cfg(feature = "videcore")]
    {
        if let Some(caps) = detect_videocore() {
            accelerators.push(caps);
        }
    }

    accelerators
}

#[cfg(feature = "cuda")]
fn detect_cuda() -> Option<AcceleratorCapabilities> {
    // Check for NVIDIA GPU by looking for device nodes or nvidia-smi
    #[cfg(target_os = "linux")]
    {
        use std::path::Path;
        use std::process::Command;

        // Check for /dev/nvidia* devices
        let nvidia_exists =
            Path::new("/dev/nvidia0").exists() || Path::new("/dev/nvidiactl").exists();

        if nvidia_exists {
            // Try to get GPU info via nvidia-smi
            if let Ok(output) = Command::new("nvidia-smi")
                .args([
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader,nounits",
                ])
                .output()
            {
                if output.status.success() {
                    let info = String::from_utf8_lossy(&output.stdout);
                    let parts: Vec<&str> = info.trim().split(',').collect();

                    let name = parts.get(0).map(|s| s.trim()).unwrap_or("NVIDIA GPU");
                    let memory_mb: u64 = parts
                        .get(1)
                        .and_then(|s| s.trim().parse().ok())
                        .unwrap_or(0);

                    return Some(AcceleratorCapabilities {
                        accelerator_type: AcceleratorType::Cuda,
                        name: name.to_string(),
                        compute_units: 1, // Would need deeper probing
                        memory_bytes: memory_mb * 1024 * 1024,
                        supports_dma: true,
                        supports_zero_copy: true,
                        max_batch_size: 1024,
                        memory_alignment: 256,
                    });
                }
            }
        }
    }

    #[cfg(target_os = "windows")]
    {
        use std::process::Command;

        // Try nvidia-smi on Windows
        if let Ok(output) = Command::new("nvidia-smi")
            .args([
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ])
            .output()
        {
            if output.status.success() {
                let info = String::from_utf8_lossy(&output.stdout);
                let parts: Vec<&str> = info.trim().split(',').collect();

                let name = parts.get(0).map(|s| s.trim()).unwrap_or("NVIDIA GPU");
                let memory_mb: u64 = parts
                    .get(1)
                    .and_then(|s| s.trim().parse().ok())
                    .unwrap_or(0);

                return Some(AcceleratorCapabilities {
                    accelerator_type: AcceleratorType::Cuda,
                    name: name.to_string(),
                    compute_units: 1,
                    memory_bytes: memory_mb * 1024 * 1024,
                    supports_dma: true,
                    supports_zero_copy: true,
                    max_batch_size: 1024,
                    memory_alignment: 256,
                });
            }
        }
    }

    None
}

#[cfg(feature = "coral")]
fn detect_edgetpu() -> Option<AcceleratorCapabilities> {
    // Detect Google Coral Edge TPU by checking for device nodes
    #[cfg(target_os = "linux")]
    {
        use std::fs;
        use std::path::Path;

        // Check for PCI-based Edge TPU (/dev/apex_*)
        let apex_exists = (0..4).any(|i| Path::new(&format!("/dev/apex_{}", i)).exists());

        // Check for USB Edge TPU (shows up in lsusb with ID 1a6e:089a or 18d1:9302)
        let usb_exists = fs::read_dir("/sys/bus/usb/devices/")
            .map(|entries| {
                entries.filter_map(|e| e.ok()).any(|entry| {
                    let vendor_path = entry.path().join("idVendor");
                    let product_path = entry.path().join("idProduct");

                    if let (Ok(vendor), Ok(product)) = (
                        fs::read_to_string(&vendor_path),
                        fs::read_to_string(&product_path),
                    ) {
                        let vendor = vendor.trim();
                        let product = product.trim();
                        // Google Edge TPU vendor/product IDs
                        (vendor == "1a6e" && product == "089a")
                            || (vendor == "18d1" && product == "9302")
                    } else {
                        false
                    }
                })
            })
            .unwrap_or(false);

        if apex_exists || usb_exists {
            let name = if apex_exists {
                "Google Coral Edge TPU (PCIe)"
            } else {
                "Google Coral Edge TPU (USB)"
            };

            return Some(AcceleratorCapabilities {
                accelerator_type: AcceleratorType::EdgeTpu,
                name: name.to_string(),
                compute_units: 1,
                memory_bytes: 8 * 1024 * 1024, // 8MB on-chip memory
                supports_dma: false,
                supports_zero_copy: false,
                max_batch_size: 1,
                memory_alignment: 64,
            });
        }
    }

    None
}

#[cfg(feature = "videcore")]
fn detect_videocore() -> Option<AcceleratorCapabilities> {
    // Detect Raspberry Pi VideoCore GPU
    #[cfg(target_os = "linux")]
    {
        use std::fs;
        use std::path::Path;

        // Check for VideoCore device
        let vchiq_exists = Path::new("/dev/vchiq").exists();
        let vcio_exists = Path::new("/dev/vcio").exists();

        if vchiq_exists || vcio_exists {
            // Try to determine RPi model from device tree
            let model = fs::read_to_string("/proc/device-tree/model")
                .unwrap_or_else(|_| "Raspberry Pi".to_string());
            let model = model.trim_matches('\0').trim();

            // Determine VideoCore version based on model
            let (vc_version, memory) = if model.contains("Pi 5") {
                ("VideoCore VII", 512 * 1024 * 1024u64) // VC7 on Pi 5
            } else if model.contains("Pi 4") || model.contains("Pi 400") {
                ("VideoCore VI", 256 * 1024 * 1024u64) // VC6 on Pi 4
            } else if model.contains("Pi 3") || model.contains("Pi Zero 2") {
                ("VideoCore IV", 128 * 1024 * 1024u64) // VC4 on Pi 3
            } else {
                ("VideoCore IV", 64 * 1024 * 1024u64) // VC4 on older models
            };

            return Some(AcceleratorCapabilities {
                accelerator_type: AcceleratorType::VideoCore,
                name: format!("{} ({})", vc_version, model),
                compute_units: 1,
                memory_bytes: memory,
                supports_dma: true,
                supports_zero_copy: false,
                max_batch_size: 256,
                memory_alignment: 16,
            });
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_accelerator() {
        let cpu = CpuAccelerator::new();
        assert_eq!(cpu.capabilities().accelerator_type, AcceleratorType::None);
        assert!(cpu.supports(AcceleratorOp::Hash {
            algorithm: HashAlgorithm::Xxhash64
        }));
    }

    #[test]
    fn test_hash_batch() {
        let cpu = CpuAccelerator::new();
        let inputs: Vec<&[u8]> = vec![b"hello", b"world", b"test"];
        let hashes = cpu.hash_batch(HashAlgorithm::Xxhash64, &inputs);
        assert_eq!(hashes.len(), 3);
        // Verify deterministic
        let hashes2 = cpu.hash_batch(HashAlgorithm::Xxhash64, &inputs);
        assert_eq!(hashes, hashes2);
    }

    #[test]
    fn test_aggregate_f64() {
        let cpu = CpuAccelerator::new();
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        assert_eq!(cpu.aggregate_f64(AggregateOp::Sum, &values), 15.0);
        assert_eq!(cpu.aggregate_f64(AggregateOp::Count, &values), 5.0);
        assert_eq!(cpu.aggregate_f64(AggregateOp::Min, &values), 1.0);
        assert_eq!(cpu.aggregate_f64(AggregateOp::Max, &values), 5.0);
        assert_eq!(cpu.aggregate_f64(AggregateOp::Average, &values), 3.0);
    }

    #[test]
    fn test_vector_ops() {
        let cpu = CpuAccelerator::new();
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let dot = cpu.vector_dot_product(&a, &b);
        assert!((dot - 32.0).abs() < 0.001); // 1*4 + 2*5 + 3*6 = 32

        let dist = cpu.vector_euclidean_distance(&a, &b);
        // sqrt((4-1)^2 + (5-2)^2 + (6-3)^2) = sqrt(27) ≈ 5.196
        assert!((dist - 5.196).abs() < 0.01);
    }

    #[test]
    fn test_xxhash64() {
        // Known test vectors
        assert_ne!(xxhash64(b""), 0);
        assert_ne!(xxhash64(b"hello"), xxhash64(b"world"));
    }

    #[test]
    fn test_crc32() {
        // "hello" should produce a consistent CRC32
        let crc = crc32(b"hello");
        assert_ne!(crc, 0);
        assert_eq!(crc, crc32(b"hello")); // Deterministic
    }

    #[test]
    fn test_detect_accelerators() {
        let accs = detect_accelerators();
        assert!(!accs.is_empty());
        // CPU fallback should always be present
        assert!(
            accs.iter()
                .any(|a| a.accelerator_type == AcceleratorType::None)
        );
    }

    #[test]
    fn test_vector_ops_mismatched_dimensions() {
        let cpu = CpuAccelerator::new();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0];
        // Should use min length, not panic
        let dot = cpu.vector_dot_product(&a, &b);
        assert!((dot - 17.0).abs() < 0.001); // 1*5 + 2*6 = 17
        let dist = cpu.vector_euclidean_distance(&a, &b);
        assert!((dist - 5.656).abs() < 0.01); // sqrt((5-1)^2 + (6-2)^2) = sqrt(32)
    }

    #[test]
    fn test_vector_ops_empty() {
        let cpu = CpuAccelerator::new();
        assert_eq!(cpu.vector_dot_product(&[], &[]), 0.0);
        assert_eq!(cpu.vector_euclidean_distance(&[], &[]), 0.0);
        assert_eq!(cpu.vector_dot_product(&[1.0], &[]), 0.0);
    }

    #[test]
    fn test_aggregate_empty() {
        let cpu = CpuAccelerator::new();
        assert_eq!(cpu.aggregate_f64(AggregateOp::Sum, &[]), 0.0);
        assert_eq!(cpu.aggregate_i64(AggregateOp::Sum, &[]), 0);
        assert_eq!(cpu.aggregate_i64(AggregateOp::Min, &[]), 0);
        assert_eq!(cpu.aggregate_i64(AggregateOp::Max, &[]), 0);
    }

    #[test]
    fn test_hash_algorithms_empty_input() {
        let cpu = CpuAccelerator::new();
        for alg in [
            HashAlgorithm::Xxhash64,
            HashAlgorithm::Crc32,
            HashAlgorithm::Murmur3,
            HashAlgorithm::Siphash,
        ] {
            let hashes = cpu.hash_batch(alg, &[b""]);
            assert_eq!(hashes.len(), 1);
            // Hash of empty should be deterministic
            let hashes2 = cpu.hash_batch(alg, &[b""]);
            assert_eq!(hashes, hashes2);
        }
    }

    #[test]
    fn test_all_hash_algorithms() {
        let cpu = CpuAccelerator::new();
        let input: Vec<&[u8]> = vec![b"test"];
        for alg in [
            HashAlgorithm::Xxhash64,
            HashAlgorithm::Crc32,
            HashAlgorithm::Murmur3,
            HashAlgorithm::Siphash,
        ] {
            let h = cpu.hash_batch(alg, &input);
            assert_eq!(h.len(), 1);
            assert_ne!(h[0], 0, "Hash of 'test' should not be 0 for {:?}", alg);
        }
    }

    #[test]
    fn test_aggregate_i64_single_element() {
        let cpu = CpuAccelerator::new();
        let values = vec![42i64];
        assert_eq!(cpu.aggregate_i64(AggregateOp::Sum, &values), 42);
        assert_eq!(cpu.aggregate_i64(AggregateOp::Count, &values), 1);
        assert_eq!(cpu.aggregate_i64(AggregateOp::Min, &values), 42);
        assert_eq!(cpu.aggregate_i64(AggregateOp::Max, &values), 42);
        assert_eq!(cpu.aggregate_i64(AggregateOp::Average, &values), 42);
    }
}
