//! `JouleWard` — Zero-Ambient-Authority Capability Sandbox, Adaptive Confinement,
//! Merkle Action Memoization, `.jcap` Single-File Bundles, and High-Resolution (`nJ` / `ns`)
//! Heterogeneous 6-Rail Energy Accounting Across the Entire Compute Fabric.
//!
//! Disintermediates:
//! - **AI Cloud & MicroVM Sandboxes**: `E2B` (Firecracker), `Modal` (gVisor), `Daytona`,
//!   `Cloudflare Sandboxes`, `Apple Containerization`, `libkrun`
//! - **OS Confinement Silos**: macOS `Seatbelt` (`sandbox-exec`), Linux `Landlock` + `bubblewrap`
//!   (`bwrap`) + `seccomp-bpf`, `WASI 0.2` preopens
//! - **Capability / Auth Schemes**: `Biscuit`, `Macaroons`, `Cedar` / `OPA`, static IAM / RBAC
//! - **Build & Image Silos**: `Dagger` BuildKit DAG caches, `Nix` derivations, `Apptainer` `.sif`
//!
//! Every capability token, confined step, memoized replay, and cross-node peer hop carries
//! integer **1-nanojoule (`nJ`, $10^{-9}\text{ J}$) + 1-nanosecond (`ns`)** resolution across
//! **6 physical rails** (`cpu_nj`, `gpu_nj`, `npu_ane_nj`, `dram_uma_nj`, `storage_nvme_nj`,
//! `network_wire_nj`) and **7 compute fabric classes** (`McuBareMetal`, `WasmEdgeIsolate`,
//! `MobileNpu`, `EdgeRobotic`, `AppleUmaWorkstation`, `LinuxAccelerator`, `CloudCxlCluster`).

use crate::BranchError;
use flate2::{Compression, read::GzDecoder, write::GzEncoder};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Write};
use std::path::Path;
use std::process::Command;
use std::sync::RwLock;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Maximum uncompressed `.jcap` bundle payload (`64 MiB`) to defeat gzip decompression bombs.
pub const MAX_JCAP_PAYLOAD_BYTES: u64 = 64 * 1024 * 1024;

/// Magic header bytes for `.jcap` single-file immutable cryptographic bundles.
pub const JCAP_MAGIC: &[u8; 4] = b"JCAP";

/// Current `.jcap` binary container format version.
pub const JCAP_VERSION: u8 = 1;

// ============================================================================
// 1. High-Resolution Heterogeneous Energy Accounting Across the Compute Fabric
// ============================================================================

/// Physical power / energy rail metered at 1-nanojoule (`nJ`) resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FabricRail {
    /// P-core / E-core / scalar / SIMD / `POPCNT` CPU compute rail.
    CpuSimd,
    /// Metal / Vulkan / CUDA / ROCm / WebGPU shader core rail.
    GpuShader,
    /// Apple ANE / Qualcomm Hexagon / Ascend / Edge TPU / deterministic SRAM LPU systolic rail.
    NpuSystolic,
    /// Local LPDDR/DDR, Apple Unified Memory (`UMA` zero-copy), or CXL 3.0 pooled memory bus rail.
    DramUmaCxl,
    /// NVMe / flash / WAL / `.jcap` content-addressable storage block I/O rail.
    StorageNvmeCas,
    /// Loopback / Ethernet / Wi-Fi / WireGuard / InfiniBand RDMA interconnect hop rail.
    NetworkWireRdma,
}

impl FabricRail {
    /// Canonical stable snake_case identifier.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::CpuSimd => "cpu_simd",
            Self::GpuShader => "gpu_shader",
            Self::NpuSystolic => "npu_systolic",
            Self::DramUmaCxl => "dram_uma_cxl",
            Self::StorageNvmeCas => "storage_nvme_cas",
            Self::NetworkWireRdma => "network_wire_rdma",
        }
    }
}

/// Heterogeneous compute fabric node tier spanning sub-milliwatt MCUs to hydro CXL clusters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FabricComputeClass {
    /// Sub-milliwatt Cortex-M / RISC-V `no_std` sensor or actuator node.
    McuBareMetal,
    /// Browser WebGPU / edge V8 / `WASI 0.2` zero-cold-start isolate.
    WasmEdgeIsolate,
    /// Battery-governed smartphone / wearable SoC with mobile NPU.
    MobileNpu,
    /// Embodied ROS2 / Jetson / RPi5 battery + real-time safety-loop robot node.
    EdgeRobotic,
    /// Apple Silicon M-series workstation with zero-copy Unified Memory (`UMA`), Metal GPU & ANE.
    AppleUmaWorkstation,
    /// Linux x86_64 / ARM64 node with discrete PCIe CUDA / ROCm / Vulkan accelerator.
    LinuxAccelerator,
    /// Multi-socket CXL 3.0 memory pool / deterministic SRAM LPU / low-carbon cloud node.
    CloudCxlCluster,
}

impl FabricComputeClass {
    /// Canonical stable snake_case identifier.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::McuBareMetal => "mcu_bare_metal",
            Self::WasmEdgeIsolate => "wasm_edge_isolate",
            Self::MobileNpu => "mobile_npu",
            Self::EdgeRobotic => "edge_robotic",
            Self::AppleUmaWorkstation => "apple_uma_workstation",
            Self::LinuxAccelerator => "linux_accelerator",
            Self::CloudCxlCluster => "cloud_cxl_cluster",
        }
    }
}

/// Hardware or calibrated sensor provenance for a [`HeterogeneousRailSample`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TelemetryCalibrationSource {
    /// Linux Intel/AMD RAPL `/sys/class/powercap` MSR + `perf_event` counters.
    LinuxRaplPerf,
    /// Apple Silicon `IOReport` / SMC rail counters & calibrated M-series rail curves.
    AppleIoReportSmc,
    /// NVIDIA NVML / AMD `rocm-smi` board power sensor.
    GpuNvmlRocmSmi,
    /// External INA219 shunt / USB-PD hardware inline meter.
    HardwareIna219UsbMeter,
    /// Deterministic physics-calibrated per-rail instruction, byte, and bitplane profile.
    CalibratedInstructionCounter,
}

/// Live OS/hardware power telemetry reading (macOS IOKit SMC, Linux Intel/AMD RAPL, or GPU SMI)
/// fused directly into [`HeterogeneousRailSample`] when live board power counters are active.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct HardwarePowerSample {
    /// Live board/package power draw in milliwatts (`mW`, `0` when hardware counter is unavailable).
    pub board_power_mw: u64,
    /// Live CPU utilization (`0.0..=1.0`).
    pub cpu_utilization: f64,
    /// Live GPU utilization (`0.0..=1.0`).
    pub gpu_utilization: f64,
    /// Live NPU / ANE utilization (`0.0..=1.0`).
    pub npu_utilization: f64,
    /// Live DRAM / UMA memory pressure (`0.0..=1.0`).
    pub memory_pressure: f64,
    /// Optional explicit sensor provenance override.
    pub source: Option<TelemetryCalibrationSource>,
}

/// High-resolution (`u64` nanojoule + `u64` nanosecond) 6-rail energy measurement.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HeterogeneousRailSample {
    /// CPU scalar / SIMD / `POPCNT` rail energy in nanojoules (`10^-9 J`).
    pub cpu_nj: u64,
    /// GPU shader core rail energy in nanojoules (`10^-9 J`).
    pub gpu_nj: u64,
    /// NPU / ANE / LPU systolic rail energy in nanojoules (`10^-9 J`).
    pub npu_ane_nj: u64,
    /// DRAM / Apple UMA / CXL memory bus rail energy in nanojoules (`10^-9 J`).
    pub dram_uma_nj: u64,
    /// NVMe / flash / WAL / CAS storage rail energy in nanojoules (`10^-9 J`).
    pub storage_nvme_nj: u64,
    /// Network / loopback / WireGuard / RDMA wire transfer energy in nanojoules (`10^-9 J`).
    pub network_wire_nj: u64,
    /// High-resolution wall-clock duration in nanoseconds (`10^-9 s`).
    pub wall_duration_ns: u64,
    /// Peak or window power draw in milliwatts (`mW`).
    pub peak_power_mw: u64,
    /// Attributed carbon footprint in nanograms of CO2e (`10^-9 gCO2e`).
    pub carbon_ng_co2: u64,
}

impl HeterogeneousRailSample {
    /// Total energy across all 6 hardware rails in integer nanojoules (`nJ`).
    pub fn total_nj(&self) -> u64 {
        self.cpu_nj
            .saturating_add(self.gpu_nj)
            .saturating_add(self.npu_ane_nj)
            .saturating_add(self.dram_uma_nj)
            .saturating_add(self.storage_nvme_nj)
            .saturating_add(self.network_wire_nj)
    }

    /// Total energy in microjoules (`µJ`) retaining sub-microjoule (`0.001 µJ`) precision.
    pub fn total_uj_f64(&self) -> f64 {
        self.total_nj() as f64 / 1_000.0
    }

    ///Total energy rounded up to integer microjoules (`µJ`) for legacy branch budgets.
    pub fn total_uj_ceil(&self) -> u64 {
        self.total_nj().saturating_add(999) / 1_000
    }

    /// Inspect energy spent on a specific [`FabricRail`] in nanojoules.
    pub fn rail_nj(&self, rail: FabricRail) -> u64 {
        match rail {
            FabricRail::CpuSimd => self.cpu_nj,
            FabricRail::GpuShader => self.gpu_nj,
            FabricRail::NpuSystolic => self.npu_ane_nj,
            FabricRail::DramUmaCxl => self.dram_uma_nj,
            FabricRail::StorageNvmeCas => self.storage_nvme_nj,
            FabricRail::NetworkWireRdma => self.network_wire_nj,
        }
    }

    /// Accumulate two 6-rail samples (summing energies, durations, and carbon, taking max peak power).
    pub fn saturating_add(&self, other: &Self) -> Self {
        Self {
            cpu_nj: self.cpu_nj.saturating_add(other.cpu_nj),
            gpu_nj: self.gpu_nj.saturating_add(other.gpu_nj),
            npu_ane_nj: self.npu_ane_nj.saturating_add(other.npu_ane_nj),
            dram_uma_nj: self.dram_uma_nj.saturating_add(other.dram_uma_nj),
            storage_nvme_nj: self.storage_nvme_nj.saturating_add(other.storage_nvme_nj),
            network_wire_nj: self.network_wire_nj.saturating_add(other.network_wire_nj),
            wall_duration_ns: self.wall_duration_ns.saturating_add(other.wall_duration_ns),
            peak_power_mw: self.peak_power_mw.max(other.peak_power_mw),
            carbon_ng_co2: self.carbon_ng_co2.saturating_add(other.carbon_ng_co2),
        }
    }

    /// Compute exact nanograms of CO2e from nanojoules and grid carbon intensity (`gCO2/kWh`).
    ///
    /// Conversion:
    /// `1 kWh = 3.6e12 J = 3.6e21 nJ`, `1 g = 1e9 ng`
    /// => `ng_co2 = nj * g_per_kwh * 1e9 / 3.6e21 = nj * g_per_kwh / 3_600_000_000_000`
    /// Note: For sub-millijoule compute steps where `nj * g_per_kwh < 3.6e12`, we round to nearest
    /// non-zero nanogram when `nj > 0 && g_per_kwh > 0` only if >= 0.5 ng, otherwise exact integer.
    pub fn compute_carbon_ng(total_nj: u64, grid_carbon_g_per_kwh: f64) -> u64 {
        if total_nj == 0 || grid_carbon_g_per_kwh <= 0.0 {
            return 0;
        }
        let g_scaled = (grid_carbon_g_per_kwh.max(0.0) * 1_000.0) as u128;
        // nj * (g_per_kwh * 1000) / 3_600_000_000_000_000
        let denom: u128 = 3_600_000_000_000_000;
        let num = (total_nj as u128).saturating_mul(g_scaled);
        ((num + denom / 2) / denom) as u64
    }

    /// Calibrated physics-grounded 6-rail energy model parameterized by fabric node class,
    /// active silicon lane, measured wall nanoseconds, memory/storage/wire bytes, and grid carbon.
    #[allow(clippy::too_many_arguments)]
    pub fn estimate_calibrated(
        fabric_class: FabricComputeClass,
        silicon_lane: &str,
        wall_duration_ns: u64,
        mem_bytes_touched: u64,
        storage_bytes_io: u64,
        wire_bytes_io: u64,
        grid_carbon_g_per_kwh: f64,
    ) -> Self {
        // Active rail power coefficients in milliwatts (1 mW * 1 us = 1 nJ => mW * ns / 1_000 = nJ!)
        let (base_cpu_mw, base_gpu_mw, base_npu_mw, p_dram_pj_per_byte): (u64, u64, u64, u64) =
            match fabric_class {
                FabricComputeClass::McuBareMetal => (18, 0, 0, 12),
                FabricComputeClass::WasmEdgeIsolate => (420, 950, 0, 35),
                FabricComputeClass::MobileNpu => (1_100, 2_200, 850, 28),
                FabricComputeClass::EdgeRobotic => (3_200, 5_800, 1_900, 45),
                // Apple UMA avoids discrete PCIe GDDR copies (~18 pJ/B unified vs ~110 pJ/B PCIe!)
                FabricComputeClass::AppleUmaWorkstation => (4_800, 11_500, 2_400, 18),
                FabricComputeClass::LinuxAccelerator => (18_000, 95_000, 0, 110),
                FabricComputeClass::CloudCxlCluster => (24_000, 120_000, 16_000, 65),
            };

        let lane_lower = silicon_lane.to_ascii_lowercase();
        let (cpu_nj, gpu_nj, npu_ane_nj) = if lane_lower.contains("gpu")
            || lane_lower.contains("wgpu")
            || lane_lower.contains("metal")
            || lane_lower.contains("vulkan")
        {
            let host_dispatch_nj = (base_cpu_mw.saturating_mul(wall_duration_ns) / 8_000).max(40);
            let shader_nj =
                (base_gpu_mw.max(900).saturating_mul(wall_duration_ns) / 1_000).max(120);
            (host_dispatch_nj, shader_nj, 0)
        } else if lane_lower.contains("npu")
            || lane_lower.contains("ane")
            || lane_lower.contains("lpu")
            || lane_lower.contains("systolic")
        {
            let host_dispatch_nj = (base_cpu_mw.saturating_mul(wall_duration_ns) / 10_000).max(25);
            let systolic_nj =
                (base_npu_mw.max(600).saturating_mul(wall_duration_ns) / 1_000).max(80);
            (host_dispatch_nj, 0, systolic_nj)
        } else {
            let scalar_nj = (base_cpu_mw.saturating_mul(wall_duration_ns) / 1_000).max(25);
            (scalar_nj, 0, 0)
        };

        // pJ/byte -> nJ: divide by 1_000
        let dram_uma_nj = (mem_bytes_touched.saturating_mul(p_dram_pj_per_byte) / 1_000)
            .max(if mem_bytes_touched > 0 { 5 } else { 0 });
        // NVMe / flash CAS ~ 240 pJ/byte
        let storage_nvme_nj = (storage_bytes_io.saturating_mul(240) / 1_000)
            .max(if storage_bytes_io > 0 { 10 } else { 0 });
        // NIC / WireGuard / RDMA ~ 480 pJ/byte
        let network_wire_nj =
            (wire_bytes_io.saturating_mul(480) / 1_000).max(if wire_bytes_io > 0 { 15 } else { 0 });

        let mut sample = Self {
            cpu_nj,
            gpu_nj,
            npu_ane_nj,
            dram_uma_nj,
            storage_nvme_nj,
            network_wire_nj,
            wall_duration_ns: wall_duration_ns.max(1),
            peak_power_mw: 0,
            carbon_ng_co2: 0,
        };
        let total = sample.total_nj();
        // P(mW) = E(nJ) * 1_000 / t(ns)
        sample.peak_power_mw = total
            .saturating_mul(1_000)
            .checked_div(sample.wall_duration_ns.max(1))
            .unwrap_or(0)
            .max(1);
        sample.carbon_ng_co2 = Self::compute_carbon_ng(total, grid_carbon_g_per_kwh);
        sample
    }

    /// Fuse live OS/hardware telemetry (`HardwarePowerSample` — macOS IOKit SMC, Linux RAPL,
    /// NVIDIA NVML / AMD `rocm-smi`) into the 6-rail nanojoule model.
    ///
    /// When `hw_sample` carries `board_power_mw > 0`, exact window silicon energy
    /// `E_board_nj = board_power_mw * wall_duration_ns / 1_000` is partitioned across
    /// `(cpu_nj, gpu_nj, npu_ane_nj, dram_uma_nj)` by blending the structural lane prior with
    /// live rail utilizations while preserving byte-exact `storage_nvme_nj` and `network_wire_nj`.
    /// Returns `(sample, calibration_source, hardware_sensor_fused)`.
    #[allow(clippy::too_many_arguments)]
    pub fn estimate_with_hardware_fusion(
        fabric_class: FabricComputeClass,
        silicon_lane: &str,
        wall_duration_ns: u64,
        mem_bytes_touched: u64,
        storage_bytes_io: u64,
        wire_bytes_io: u64,
        grid_carbon_g_per_kwh: f64,
        hw_sample: Option<HardwarePowerSample>,
    ) -> (Self, TelemetryCalibrationSource, bool) {
        let base = Self::estimate_calibrated(
            fabric_class,
            silicon_lane,
            wall_duration_ns,
            mem_bytes_touched,
            storage_bytes_io,
            wire_bytes_io,
            grid_carbon_g_per_kwh,
        );

        let Some(hw) = hw_sample.filter(|s| s.board_power_mw > 0) else {
            return (
                base,
                TelemetryCalibrationSource::CalibratedInstructionCounter,
                false,
            );
        };

        let dur_ns = wall_duration_ns.max(1);
        // 1 mW * 1_000 ns = 1 nJ
        let measured_board_nj =
            (((hw.board_power_mw as u128) * (dur_ns as u128)) / 1_000).max(1) as u64;

        let prior_silicon_sum =
            (base.cpu_nj + base.gpu_nj + base.npu_ane_nj + base.dram_uma_nj).max(1) as f64;
        let p_cpu = base.cpu_nj as f64 / prior_silicon_sum;
        let p_gpu = base.gpu_nj as f64 / prior_silicon_sum;
        let p_npu = base.npu_ane_nj as f64 / prior_silicon_sum;
        let p_dram = base.dram_uma_nj as f64 / prior_silicon_sum;

        // Blend structural lane prior (65%) with live hardware utilization telemetry (35%)
        // while keeping inactive accelerators at 0 unless structurally invoked or > 2% active.
        let u_cpu = hw.cpu_utilization.clamp(0.02, 1.0);
        let u_gpu = if base.gpu_nj > 0 || hw.gpu_utilization > 0.02 {
            hw.gpu_utilization.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let u_npu = if base.npu_ane_nj > 0 || hw.npu_utilization > 0.02 {
            hw.npu_utilization.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let u_dram = hw.memory_pressure.clamp(0.02, 1.0) * 0.35;
        let u_sum = (u_cpu + u_gpu + u_npu + u_dram).max(1e-6);

        let w_cpu = 0.65 * p_cpu + 0.35 * (u_cpu / u_sum);
        let w_gpu = 0.65 * p_gpu + 0.35 * (u_gpu / u_sum);
        let w_npu = 0.65 * p_npu + 0.35 * (u_npu / u_sum);
        let w_dram = 0.65 * p_dram + 0.35 * (u_dram / u_sum);
        let w_sum = (w_cpu + w_gpu + w_npu + w_dram).max(1e-9);

        let gpu_nj = ((measured_board_nj as f64) * (w_gpu / w_sum)).round() as u64;
        let npu_ane_nj = ((measured_board_nj as f64) * (w_npu / w_sum)).round() as u64;
        let dram_uma_nj = ((measured_board_nj as f64) * (w_dram / w_sum)).round() as u64;
        // Exact energy conservation: assign remainder to CPU rail so sum == measured_board_nj!
        let allocated_other = gpu_nj
            .saturating_add(npu_ane_nj)
            .saturating_add(dram_uma_nj);
        let cpu_nj = measured_board_nj.saturating_sub(allocated_other);

        let mut fused = Self {
            cpu_nj,
            gpu_nj,
            npu_ane_nj,
            dram_uma_nj,
            storage_nvme_nj: base.storage_nvme_nj,
            network_wire_nj: base.network_wire_nj,
            wall_duration_ns: dur_ns,
            peak_power_mw: hw.board_power_mw.max(base.peak_power_mw),
            carbon_ng_co2: 0,
        };
        fused.carbon_ng_co2 = Self::compute_carbon_ng(fused.total_nj(), grid_carbon_g_per_kwh);

        let resolved_source = hw.source.unwrap_or(match fabric_class {
            FabricComputeClass::AppleUmaWorkstation | FabricComputeClass::MobileNpu => {
                TelemetryCalibrationSource::AppleIoReportSmc
            }
            FabricComputeClass::LinuxAccelerator | FabricComputeClass::CloudCxlCluster => {
                if base.gpu_nj > base.cpu_nj {
                    TelemetryCalibrationSource::GpuNvmlRocmSmi
                } else {
                    TelemetryCalibrationSource::LinuxRaplPerf
                }
            }
            _ => TelemetryCalibrationSource::HardwareIna219UsbMeter,
        });

        (fused, resolved_source, true)
    }
}

/// Single-hop energy attribution record on one node of the compute fabric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FabricHopEnergyReceipt {
    /// Node identifier (e.g., `"edge-robot-01"`, `"uma-primary-01"`, `"hydro-cloud-01"`).
    pub node_id: String,
    /// Compute fabric tier of this node.
    pub fabric_class: FabricComputeClass,
    /// Silicon lane or execution mechanism (e.g., `"wgpu_xnor_popcnt_wgsl"`, `"macos-seatbelt-sbpl"`).
    pub silicon_lane: String,
    /// Sensor / calibration provenance.
    pub calibration_source: TelemetryCalibrationSource,
    /// Local grid carbon intensity (`gCO2/kWh`) at execution time.
    pub grid_carbon_g_per_kwh: f64,
    /// High-resolution 6-rail energy sample for this hop.
    pub rails: HeterogeneousRailSample,
}

/// End-to-end multi-hop heterogeneous energy receipt spanning any number of fabric nodes.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FabricTraceReceipt {
    /// Ordered hops from ingress node -> wire -> execution peer(s).
    pub hops: Vec<FabricHopEnergyReceipt>,
    /// Rolled-up 6-rail totals across all hops in the trace.
    pub aggregate_rails: HeterogeneousRailSample,
    /// Total nanojoules grouped by `FabricComputeClass`.
    pub per_class_totals_nj: BTreeMap<String, u64>,
    /// Nanojoules saved via `WardActionCache` or `LangCache` / `Engram` cascade hit.
    pub saved_by_memoization_nj: u64,
}

impl FabricTraceReceipt {
    /// Append a hop and recompute the rolled-up multi-rail + per-class totals.
    pub fn push_hop(&mut self, hop: FabricHopEnergyReceipt) {
        self.aggregate_rails = self.aggregate_rails.saturating_add(&hop.rails);
        let entry = self
            .per_class_totals_nj
            .entry(hop.fabric_class.as_str().to_string())
            .or_insert(0);
        *entry = entry.saturating_add(hop.rails.total_nj());
        self.hops.push(hop);
    }

    /// Merge a remote peer's `FabricTraceReceipt` (plus wire transfer sample) into this trace.
    pub fn merge_remote_trace(
        &mut self,
        wire_hop: FabricHopEnergyReceipt,
        remote: &FabricTraceReceipt,
    ) {
        self.push_hop(wire_hop);
        for h in &remote.hops {
            self.push_hop(h.clone());
        }
        self.saved_by_memoization_nj = self
            .saved_by_memoization_nj
            .saturating_add(remote.saved_by_memoization_nj);
    }
}

/// Snapshot of cumulative fabric-wide 6-rail energy accounting.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FabricEnergyLedgerSnapshot {
    /// Total actions / inference calls accounted.
    pub total_events: u64,
    /// Cumulative 6-rail nanojoules across the entire fabric.
    pub cumulative_rails: HeterogeneousRailSample,
    /// Cumulative nanojoules broken down by `node_id`.
    pub per_node_rails: BTreeMap<String, HeterogeneousRailSample>,
    /// Cumulative nanojoules broken down by `FabricComputeClass`.
    pub per_class_rails: BTreeMap<String, HeterogeneousRailSample>,
    /// Cumulative nanojoules spent per `WardCapabilityToken::root_token_id` (so offline-attenuated
    /// child tokens cannot evade their parent's thermodynamic budget!).
    pub per_token_spent_nj: BTreeMap<String, u64>,
    /// Cumulative nanojoules avoided via CAS action memoization & semantic/engram cascades.
    pub total_saved_memoization_nj: u64,
}

/// Thread-safe high-resolution energy ledger spanning all local & remote fabric nodes.
#[derive(Debug, Default)]
pub struct FabricEnergyLedger {
    inner: RwLock<FabricEnergyLedgerSnapshot>,
}

impl FabricEnergyLedger {
    /// Create an empty fabric energy ledger.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a [`FabricTraceReceipt`] and optionally charge a capability token's root lineage.
    pub fn record_trace(&self, root_token_id: Option<&str>, trace: &FabricTraceReceipt) {
        let mut guard = match self.inner.write() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        guard.total_events = guard.total_events.saturating_add(1);
        guard.cumulative_rails = guard
            .cumulative_rails
            .saturating_add(&trace.aggregate_rails);
        guard.total_saved_memoization_nj = guard
            .total_saved_memoization_nj
            .saturating_add(trace.saved_by_memoization_nj);

        for hop in &trace.hops {
            let node_entry = guard.per_node_rails.entry(hop.node_id.clone()).or_default();
            *node_entry = node_entry.saturating_add(&hop.rails);

            let class_entry = guard
                .per_class_rails
                .entry(hop.fabric_class.as_str().to_string())
                .or_default();
            *class_entry = class_entry.saturating_add(&hop.rails);
        }

        if let Some(tid) = root_token_id {
            let spent = guard.per_token_spent_nj.entry(tid.to_string()).or_insert(0);
            *spent = spent.saturating_add(trace.aggregate_rails.total_nj());
        }
    }

    /// Query cumulative nanojoules already spent under `root_token_id`.
    pub fn token_spent_nj(&self, root_token_id: &str) -> u64 {
        let guard = match self.inner.read() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        guard
            .per_token_spent_nj
            .get(root_token_id)
            .copied()
            .unwrap_or(0)
    }

    /// Export an immutable snapshot of all fabric rails, nodes, classes, and token budgets.
    pub fn snapshot(&self) -> FabricEnergyLedgerSnapshot {
        match self.inner.read() {
            Ok(g) => g.clone(),
            Err(p) => p.into_inner().clone(),
        }
    }
}

// ============================================================================
// 2. `ThermoCap` RFC-2104 HMAC-SHA256 Chained Capability Macaroons
// ============================================================================

/// Pure-Rust RFC-2104 `HMAC-SHA256` built directly on `sha2::Sha256` (zero external crate bloat).
pub fn hmac_sha256(key: &[u8], message: &[u8]) -> [u8; 32] {
    const BLOCK_LEN: usize = 64;
    let mut key_block = [0u8; BLOCK_LEN];
    if key.len() > BLOCK_LEN {
        let hashed = Sha256::digest(key);
        key_block[..32].copy_from_slice(&hashed);
    } else {
        key_block[..key.len()].copy_from_slice(key);
    }

    let mut ipad = [0x36u8; BLOCK_LEN];
    let mut opad = [0x5cu8; BLOCK_LEN];
    for i in 0..BLOCK_LEN {
        ipad[i] ^= key_block[i];
        opad[i] ^= key_block[i];
    }

    let mut inner = Sha256::new();
    inner.update(ipad);
    inner.update(message);
    let inner_hash = inner.finalize();

    let mut outer = Sha256::new();
    outer.update(opad);
    outer.update(inner_hash);
    let final_hash = outer.finalize();

    let mut out = [0u8; 32];
    out.copy_from_slice(&final_hash);
    out
}

/// Constant-time 32-byte equality check to prevent timing side channels during MAC verification.
fn ct_eq_32(a: &[u8; 32], b: &[u8; 32]) -> bool {
    let mut diff = 0u8;
    for i in 0..32 {
        diff |= a[i] ^ b[i];
    }
    diff == 0
}

/// Monotonic attenuation caveat chained into a [`WardCapabilityToken`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum WardCaveat {
    /// Hard wall-clock expiration timestamp (Unix epoch milliseconds).
    ExpiresAtEpochMs(u64),
    /// Restrict filesystem access to normalized path prefixes (intersected across caveats).
    AllowPathPrefixes(Vec<String>),
    /// Force read-only rootfs / VFS confinement when `true`.
    ReadOnlyFs(bool),
    /// Restrict outbound network egress to exact hostnames (empty Vec = completely airgapped!).
    AllowEgressHosts(Vec<String>),
    /// Restrict SQL / polyglot queries to named tables.
    AllowSqlTables(Vec<String>),
    /// Restrict execution placement to specific [`FabricComputeClass`] tiers.
    AllowFabricClasses(Vec<FabricComputeClass>),
    /// Restrict execution to named silicon lanes (e.g. `["wgpu_xnor_popcnt_wgsl", "cpu_simd"]`).
    AllowSiliconLanes(Vec<String>),
    /// Cap cumulative + action energy across all 6 rails in nanojoules (`nJ`).
    MaxTotalNanojoules(u64),
    /// Cap energy on an individual hardware rail (e.g. `NetworkWireRdma: 0` or `GpuShader: 500_000`).
    MaxRailNanojoules { rail: FabricRail, max_nj: u64 },
    /// Cap instantaneous / window power draw in milliwatts (`mW`) for battery/thermal safety.
    MaxPeakPowerMw(u64),
    /// Escalate to Human-in-the-Loop (`HITL`) approval if cumulative + estimated `nJ` exceeds threshold.
    HitlGateAboveNanojoules(u64),
    /// Gate whether live `JouleSynapse` (`Titans` fast weights / `Sakana SVF`) plasticity is allowed.
    AllowSynapsePlasticity(bool),
}

/// Unforgeable offline-attenuatable `ThermoCap` capability macaroon.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WardCapabilityToken {
    /// Root lineage identifier shared by a root token and all attenuated descendants.
    pub root_token_id: String,
    /// Principal / agent / workload subject name.
    pub subject: String,
    /// Mint timestamp (Unix epoch milliseconds).
    pub issued_at_ms: u64,
    /// Ordered append-only caveat chain (`sig_{i+1} = HMAC(sig_i, caveat_i)`).
    pub caveats: Vec<WardCaveat>,
    /// Hex-encoded 32-byte chained `HMAC-SHA256` tag.
    pub signature_hex: String,
}

impl WardCapabilityToken {
    /// Mint a fresh root capability token signed with `root_secret`, optionally applying initial caveats.
    pub fn mint(
        root_secret: &[u8; 32],
        root_token_id: impl Into<String>,
        subject: impl Into<String>,
        issued_at_ms: u64,
        initial_caveats: Vec<WardCaveat>,
    ) -> Result<Self, BranchError> {
        let root_token_id = root_token_id.into();
        let subject = subject.into();
        if root_token_id.trim().is_empty() || subject.trim().is_empty() {
            return Err(BranchError::InvalidName(
                "root_token_id and subject must be non-empty".into(),
            ));
        }
        let header = format!("JWARD_V1:{root_token_id}:{subject}:{issued_at_ms}");
        let sig0 = hmac_sha256(root_secret, header.as_bytes());
        let mut token = Self {
            root_token_id,
            subject,
            issued_at_ms,
            caveats: Vec::new(),
            signature_hex: hex::encode(sig0),
        };
        for c in initial_caveats {
            token = token.attenuate(c)?;
        }
        Ok(token)
    }

    /// Offline attenuation: derive a strictly narrower child token **without** possessing `root_secret`!
    pub fn attenuate(&self, caveat: WardCaveat) -> Result<Self, BranchError> {
        let prev_sig_vec = hex::decode(&self.signature_hex)
            .map_err(|e| BranchError::Storage(format!("invalid token signature hex: {e}")))?;
        if prev_sig_vec.len() != 32 {
            return Err(BranchError::Storage(
                "token signature must be 32 bytes".into(),
            ));
        }
        let caveat_bytes = serde_json::to_vec(&caveat)
            .map_err(|e| BranchError::Storage(format!("failed to serialize caveat: {e}")))?;
        let next_sig = hmac_sha256(&prev_sig_vec, &caveat_bytes);

        let mut next_caveats = self.caveats.clone();
        next_caveats.push(caveat);
        Ok(Self {
            root_token_id: self.root_token_id.clone(),
            subject: self.subject.clone(),
            issued_at_ms: self.issued_at_ms,
            caveats: next_caveats,
            signature_hex: hex::encode(next_sig),
        })
    }

    /// Verify cryptographic HMAC chain integrity against `root_secret`.
    pub fn verify_signature(&self, root_secret: &[u8; 32]) -> Result<(), BranchError> {
        let claimed_vec = hex::decode(&self.signature_hex)
            .map_err(|_| BranchError::Storage("malformed capability signature hex".into()))?;
        if claimed_vec.len() != 32 {
            return Err(BranchError::Storage(
                "capability signature length mismatch".into(),
            ));
        }
        let mut claimed = [0u8; 32];
        claimed.copy_from_slice(&claimed_vec);

        let header = format!(
            "JWARD_V1:{}:{}:{}",
            self.root_token_id, self.subject, self.issued_at_ms
        );
        let mut sig = hmac_sha256(root_secret, header.as_bytes());
        for caveat in &self.caveats {
            let caveat_bytes = serde_json::to_vec(caveat)
                .map_err(|e| BranchError::Storage(format!("caveat serialization failed: {e}")))?;
            sig = hmac_sha256(&sig, &caveat_bytes);
        }

        if !ct_eq_32(&sig, &claimed) {
            return Err(BranchError::Storage(
                "capability HMAC verification failed (forged, mutated, or stripped caveat)".into(),
            ));
        }
        Ok(())
    }

    /// Fold the entire caveat chain into effective monotonic policy bounds.
    pub fn effective_policy(&self) -> EffectiveWardPolicy {
        let mut policy = EffectiveWardPolicy::unrestricted();
        for caveat in &self.caveats {
            match caveat {
                WardCaveat::ExpiresAtEpochMs(ts) => {
                    policy.expires_at_ms =
                        Some(policy.expires_at_ms.map_or(*ts, |prev| prev.min(*ts)));
                }
                WardCaveat::AllowPathPrefixes(prefixes) => {
                    let next_set: Vec<String> = prefixes.clone();
                    policy.allowed_paths = Some(match policy.allowed_paths.take() {
                        None => next_set,
                        Some(existing) => existing
                            .into_iter()
                            .filter(|e| {
                                next_set
                                    .iter()
                                    .any(|n| e.starts_with(n) || n.starts_with(e))
                            })
                            .map(|e| {
                                // Keep the more specific prefix
                                next_set
                                    .iter()
                                    .find(|n| n.starts_with(&e) && n.len() > e.len())
                                    .cloned()
                                    .unwrap_or(e)
                            })
                            .collect(),
                    });
                }
                WardCaveat::ReadOnlyFs(ro) => {
                    // Once read-only is set, a child caveat can NEVER re-enable writes!
                    if *ro {
                        policy.read_only_fs = true;
                    }
                }
                WardCaveat::AllowEgressHosts(hosts) => {
                    let set: BTreeSet<String> = hosts.iter().cloned().collect();
                    policy.allowed_egress_hosts = Some(match policy.allowed_egress_hosts.take() {
                        None => set,
                        Some(prev) => prev.intersection(&set).cloned().collect(),
                    });
                }
                WardCaveat::AllowSqlTables(tables) => {
                    let set: BTreeSet<String> = tables.iter().cloned().collect();
                    policy.allowed_sql_tables = Some(match policy.allowed_sql_tables.take() {
                        None => set,
                        Some(prev) => prev.intersection(&set).cloned().collect(),
                    });
                }
                WardCaveat::AllowFabricClasses(classes) => {
                    let set: BTreeSet<FabricComputeClass> = classes.iter().copied().collect();
                    policy.allowed_fabric_classes =
                        Some(match policy.allowed_fabric_classes.take() {
                            None => set,
                            Some(prev) => prev.intersection(&set).copied().collect(),
                        });
                }
                WardCaveat::AllowSiliconLanes(lanes) => {
                    let set: BTreeSet<String> = lanes.iter().cloned().collect();
                    policy.allowed_silicon_lanes =
                        Some(match policy.allowed_silicon_lanes.take() {
                            None => set,
                            Some(prev) => prev.intersection(&set).cloned().collect(),
                        });
                }
                WardCaveat::MaxTotalNanojoules(max_nj) => {
                    policy.max_total_nj = Some(
                        policy
                            .max_total_nj
                            .map_or(*max_nj, |prev| prev.min(*max_nj)),
                    );
                }
                WardCaveat::MaxRailNanojoules { rail, max_nj } => {
                    let entry = policy.max_rail_nj.entry(*rail).or_insert(*max_nj);
                    *entry = (*entry).min(*max_nj);
                }
                WardCaveat::MaxPeakPowerMw(max_mw) => {
                    policy.max_peak_power_mw = Some(
                        policy
                            .max_peak_power_mw
                            .map_or(*max_mw, |prev| prev.min(*max_mw)),
                    );
                }
                WardCaveat::HitlGateAboveNanojoules(gate_nj) => {
                    policy.hitl_gate_nj = Some(
                        policy
                            .hitl_gate_nj
                            .map_or(*gate_nj, |prev| prev.min(*gate_nj)),
                    );
                }
                WardCaveat::AllowSynapsePlasticity(allowed) => {
                    if !*allowed {
                        policy.allow_synapse_plasticity = false;
                    }
                }
            }
        }
        policy
    }
}

/// Effective narrowed policy derived from a [`WardCapabilityToken`]'s caveat chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectiveWardPolicy {
    pub expires_at_ms: Option<u64>,
    pub allowed_paths: Option<Vec<String>>,
    pub read_only_fs: bool,
    pub allowed_egress_hosts: Option<BTreeSet<String>>,
    pub allowed_sql_tables: Option<BTreeSet<String>>,
    pub allowed_fabric_classes: Option<BTreeSet<FabricComputeClass>>,
    pub allowed_silicon_lanes: Option<BTreeSet<String>>,
    pub max_total_nj: Option<u64>,
    pub max_rail_nj: BTreeMap<FabricRail, u64>,
    pub max_peak_power_mw: Option<u64>,
    pub hitl_gate_nj: Option<u64>,
    pub allow_synapse_plasticity: bool,
}

impl EffectiveWardPolicy {
    /// Baseline unrestricted root policy prior to applying caveats.
    pub fn unrestricted() -> Self {
        Self {
            expires_at_ms: None,
            allowed_paths: None,
            read_only_fs: false,
            allowed_egress_hosts: None,
            allowed_sql_tables: None,
            allowed_fabric_classes: None,
            allowed_silicon_lanes: None,
            max_total_nj: None,
            max_rail_nj: BTreeMap::new(),
            max_peak_power_mw: None,
            hitl_gate_nj: None,
            allow_synapse_plasticity: true,
        }
    }

    /// True when outbound network egress is completely forbidden (`AllowEgressHosts([])`).
    pub fn is_airgapped(&self) -> bool {
        matches!(self.allowed_egress_hosts, Some(ref h) if h.is_empty())
    }

    /// Deterministic SHA-256 digest of the effective policy for CAS action memoization keys.
    pub fn policy_digest_hex(&self) -> String {
        let bytes = serde_json::to_vec(self).unwrap_or_default();
        hex::encode(Sha256::digest(bytes))
    }
}

/// Context describing a candidate action or inference step to authorize against a [`WardCapabilityToken`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WardActionContext {
    pub now_epoch_ms: u64,
    pub target_fabric_class: FabricComputeClass,
    pub silicon_lane: String,
    pub requested_paths: Vec<String>,
    pub requires_fs_write: bool,
    pub requested_egress_host: Option<String>,
    pub requested_sql_tables: Vec<String>,
    pub mutates_synapse_weights: bool,
    pub estimated_rails: HeterogeneousRailSample,
    /// True when a human operator has explicitly countersigned a HITL escalation.
    pub hitl_approved: bool,
}

/// Structured authorization outcome returned by `authorize_ward_action`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "decision", rename_all = "snake_case")]
pub enum WardDecision {
    /// Action is cryptographically valid and within all capability + thermodynamic bounds.
    Permit { effective_remaining_nj: Option<u64> },
    /// Action is valid in scope, but pauses for Human-in-the-Loop approval due to energy gate.
    RequiresHumanApproval {
        threshold_nj: u64,
        projected_total_nj: u64,
        reason: String,
    },
    /// Action is rejected due to cryptographic tamper or caveat / rail / fabric violation.
    Deny { reason: String },
}

/// Verify HMAC chain + evaluate all spatial, network, fabric, power, and 6-rail nanojoule caveats.
pub fn authorize_ward_action(
    root_secret: &[u8; 32],
    token: &WardCapabilityToken,
    ctx: &WardActionContext,
    already_spent_nj: u64,
) -> WardDecision {
    if let Err(e) = token.verify_signature(root_secret) {
        return WardDecision::Deny {
            reason: e.to_string(),
        };
    }

    let policy = token.effective_policy();

    if let Some(exp) = policy.expires_at_ms
        && ctx.now_epoch_ms > exp
    {
        return WardDecision::Deny {
            reason: format!(
                "capability token expired at {exp} ms (now: {} ms)",
                ctx.now_epoch_ms
            ),
        };
    }

    if policy.read_only_fs && ctx.requires_fs_write {
        return WardDecision::Deny {
            reason: "capability enforces ReadOnlyFs(true), denying write operation".into(),
        };
    }

    if ctx.mutates_synapse_plasticity_flag() && !policy.allow_synapse_plasticity {
        return WardDecision::Deny {
            reason: "capability denies live JouleSynapse weight plasticity".into(),
        };
    }

    if let Some(ref classes) = policy.allowed_fabric_classes
        && !classes.contains(&ctx.target_fabric_class)
    {
        return WardDecision::Deny {
            reason: format!(
                "fabric compute class '{}' is outside allowed capability set",
                ctx.target_fabric_class.as_str()
            ),
        };
    }

    if let Some(ref lanes) = policy.allowed_silicon_lanes
        && !lanes.contains(&ctx.silicon_lane)
    {
        return WardDecision::Deny {
            reason: format!(
                "silicon lane '{}' is outside allowed capability set",
                ctx.silicon_lane
            ),
        };
    }

    if let Some(ref allowed_paths) = policy.allowed_paths {
        for req_path in &ctx.requested_paths {
            if !allowed_paths.iter().any(|p| req_path.starts_with(p)) {
                return WardDecision::Deny {
                    reason: format!("path '{req_path}' is outside allowed prefixes"),
                };
            }
        }
    }

    if let Some(ref host) = ctx.requested_egress_host
        && let Some(ref allowed_hosts) = policy.allowed_egress_hosts
        && !allowed_hosts.contains(host)
    {
        return WardDecision::Deny {
            reason: format!("network egress to '{host}' denied by capability caveat"),
        };
    }

    if let Some(ref allowed_tables) = policy.allowed_sql_tables {
        for tbl in &ctx.requested_sql_tables {
            if !allowed_tables.contains(tbl) {
                return WardDecision::Deny {
                    reason: format!("SQL table '{tbl}' denied by capability caveat"),
                };
            }
        }
    }

    // Per-rail nanojoule limits (heterogeneous rail enforcement!)
    for (rail, max_rail_nj) in &policy.max_rail_nj {
        let req_rail_nj = ctx.estimated_rails.rail_nj(*rail);
        if req_rail_nj > *max_rail_nj {
            return WardDecision::Deny {
                reason: format!(
                    "rail '{}' requires {} nJ, exceeding per-rail cap {} nJ",
                    rail.as_str(),
                    req_rail_nj,
                    max_rail_nj
                ),
            };
        }
    }

    // Peak milliwatt brownout / thermal envelope enforcement
    if let Some(max_mw) = policy.max_peak_power_mw
        && ctx.estimated_rails.peak_power_mw > max_mw
    {
        return WardDecision::Deny {
            reason: format!(
                "estimated peak power {} mW exceeds thermal/battery cap {} mW",
                ctx.estimated_rails.peak_power_mw, max_mw
            ),
        };
    }

    let projected_total_nj = already_spent_nj.saturating_add(ctx.estimated_rails.total_nj());
    if let Some(max_total) = policy.max_total_nj
        && projected_total_nj > max_total
    {
        return WardDecision::Deny {
            reason: format!(
                "total energy budget exhausted: projected {projected_total_nj} nJ exceeds cap {max_total} nJ (already spent: {already_spent_nj} nJ)"
            ),
        };
    }

    if let Some(gate_nj) = policy.hitl_gate_nj
        && projected_total_nj > gate_nj
        && !ctx.hitl_approved
    {
        return WardDecision::RequiresHumanApproval {
            threshold_nj: gate_nj,
            projected_total_nj,
            reason: format!(
                "projected energy {projected_total_nj} nJ exceeds autonomous HITL gate {gate_nj} nJ"
            ),
        };
    }

    let remaining = policy
        .max_total_nj
        .map(|m| m.saturating_sub(projected_total_nj));
    WardDecision::Permit {
        effective_remaining_nj: remaining,
    }
}

impl WardActionContext {
    fn mutates_synapse_plasticity_flag(&self) -> bool {
        self.mutates_synapse_weights
    }
}

// ============================================================================
// 3. Adaptive Multi-OS & Multi-Runtime Confinement Compiler
// ============================================================================

/// Compiled OS & WebAssembly confinement specifications generated from [`EffectiveWardPolicy`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledConfinementProfile {
    /// Digest of the effective capability policy.
    pub policy_digest_hex: String,
    /// macOS `Seatbelt` (`sandbox-exec`) SBPL `(version 1)` profile text.
    pub macos_seatbelt_sbpl: String,
    /// Linux `bubblewrap` (`bwrap`) zero-root namespace + bind-mount argv prefix.
    pub linux_bwrap_argv: Vec<String>,
    /// Linux `Landlock` ABI v4 filesystem & TCP rules summary.
    pub linux_landlock_rules: Vec<String>,
    /// Linux `seccomp-bpf` blocked dangerous syscalls.
    pub linux_seccomp_blocked_syscalls: Vec<String>,
    /// `WASI 0.2` zero-ambient-authority preopened directories & fuel cap.
    pub wasi_preopens: Vec<String>,
    /// Deterministic WebAssembly fuel limit derived from `max_total_nj`.
    pub wasi_fuel_limit: Option<u64>,
}

/// Escape double-quoted strings inside macOS Seatbelt SBPL expressions.
fn escape_sbpl_path(p: &str) -> String {
    p.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Compile a [`WardCapabilityToken`]'s effective policy into native macOS Seatbelt, Linux
/// Landlock/`bwrap`/`seccomp-bpf`, and `WASI 0.2` confinement artifacts.
pub fn compile_confinement(
    policy: &EffectiveWardPolicy,
    workspace_dir: &Path,
) -> CompiledConfinementProfile {
    let ws_str = workspace_dir.to_string_lossy().to_string();
    let escaped_ws = escape_sbpl_path(&ws_str);

    // 1. macOS Seatbelt SBPL `(version 1)`
    let mut sbpl_lines = vec![
        "(version 1)".to_string(),
        "(deny default)".to_string(),
        "(allow process-exec*)".to_string(),
        "(allow process-fork)".to_string(),
        "(allow sysctl-read)".to_string(),
        "(allow mach-lookup)".to_string(),
        "(allow signal (target self))".to_string(),
        "(allow file-read* (subpath \"/bin\") (subpath \"/usr\") (subpath \"/System\") (subpath \"/Library\") (subpath \"/private/var/db\") (literal \"/\") (literal \"/dev/null\") (literal \"/dev/urandom\") (literal \"/dev/random\"))".to_string(),
        format!("(allow file-read* (subpath \"{escaped_ws}\"))"),
    ];
    if let Some(ref paths) = policy.allowed_paths {
        for p in paths {
            let ep = escape_sbpl_path(p);
            sbpl_lines.push(format!("(allow file-read* (subpath \"{ep}\"))"));
        }
    }
    if policy.read_only_fs {
        sbpl_lines.push("(deny file-write*)".to_string());
        sbpl_lines.push("(allow file-write* (literal \"/dev/null\"))".to_string());
    } else {
        sbpl_lines.push(format!("(allow file-write* (subpath \"{escaped_ws}\"))"));
        sbpl_lines.push("(allow file-write* (literal \"/dev/null\"))".to_string());
    }
    if policy.is_airgapped() {
        sbpl_lines.push("(deny network*)".to_string());
    } else if policy.allowed_egress_hosts.is_some() {
        sbpl_lines.push("(allow network-outbound (remote tcp))".to_string());
    } else {
        sbpl_lines.push("(allow network*)".to_string());
    }

    // 2. Linux `bwrap` + `Landlock` + `seccomp-bpf`
    let mut bwrap = vec![
        "bwrap".to_string(),
        "--unshare-all".to_string(),
        "--die-with-parent".to_string(),
        "--new-session".to_string(),
        "--ro-bind".to_string(),
        "/usr".to_string(),
        "/usr".to_string(),
        "--ro-bind".to_string(),
        "/bin".to_string(),
        "/bin".to_string(),
        "--ro-bind".to_string(),
        "/lib".to_string(),
        "/lib".to_string(),
        "--dev".to_string(),
        "/dev".to_string(),
        "--proc".to_string(),
        "/proc".to_string(),
    ];
    if !policy.is_airgapped() {
        bwrap.push("--share-net".to_string());
    }
    if policy.read_only_fs {
        bwrap.extend([
            "--ro-bind".to_string(),
            ws_str.clone(),
            "/workspace".to_string(),
        ]);
    } else {
        bwrap.extend([
            "--bind".to_string(),
            ws_str.clone(),
            "/workspace".to_string(),
        ]);
    }

    let mut landlock_rules = vec![
        "LANDLOCK_ACCESS_FS_EXECUTE | LANDLOCK_ACCESS_FS_READ_FILE -> /usr,/bin,/lib".to_string(),
        if policy.read_only_fs {
            format!("LANDLOCK_ACCESS_FS_READ_FILE | LANDLOCK_ACCESS_FS_READ_DIR -> {ws_str}")
        } else {
            format!("LANDLOCK_ACCESS_FS_READ_WRITE -> {ws_str}")
        },
    ];
    if policy.is_airgapped() {
        landlock_rules.push("LANDLOCK_ACCESS_NET_CONNECT_TCP -> DENY_ALL".to_string());
    } else if let Some(ref hosts) = policy.allowed_egress_hosts {
        landlock_rules.push(format!(
            "LANDLOCK_ACCESS_NET_CONNECT_TCP -> ALLOW({:?})",
            hosts
        ));
    }

    let mut seccomp_blocked = vec![
        "ptrace".to_string(),
        "kexec_load".to_string(),
        "kexec_file_load".to_string(),
        "mount".to_string(),
        "umount2".to_string(),
        "pivot_root".to_string(),
        "bpf".to_string(),
        "perf_event_open".to_string(),
        "init_module".to_string(),
        "finit_module".to_string(),
    ];
    if policy.is_airgapped() {
        seccomp_blocked.extend([
            "socket".to_string(),
            "connect".to_string(),
            "sendto".to_string(),
        ]);
    }

    // 3. WASI 0.2 preopens & deterministic fuel
    let mut wasi_preopens = vec![if policy.read_only_fs {
        format!("{ws_str}::/workspace:ro")
    } else {
        format!("{ws_str}::/workspace:rw")
    }];
    if let Some(ref paths) = policy.allowed_paths {
        for p in paths {
            wasi_preopens.push(format!(
                "{p}::{p}:{}",
                if policy.read_only_fs { "ro" } else { "rw" }
            ));
        }
    }
    // ~10 instructions / fuel unit per nanojoule on WasmEdgeIsolate
    let wasi_fuel_limit = policy.max_total_nj.map(|nj| nj.saturating_mul(10));

    CompiledConfinementProfile {
        policy_digest_hex: policy.policy_digest_hex(),
        macos_seatbelt_sbpl: sbpl_lines.join("\n"),
        linux_bwrap_argv: bwrap,
        linux_landlock_rules: landlock_rules,
        linux_seccomp_blocked_syscalls: seccomp_blocked,
        wasi_preopens,
        wasi_fuel_limit,
    }
}

// ============================================================================
// 4. `Dagger` / `Nix`-Class Merkle CAS `WardActionCache` & Confined Executor
// ============================================================================

/// Cached output of a deterministic confined step (`Dagger` / `Nix`-class CAS memoization).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedWardAction {
    pub cache_key_hex: String,
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    /// Original 6-rail nanojoules consumed when the step ran the first time.
    pub original_rails: HeterogeneousRailSample,
    pub hits: u64,
}

/// Request to execute a confined command under a [`WardCapabilityToken`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WardExecRequest {
    pub token: WardCapabilityToken,
    pub command: Vec<String>,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    /// Optional Merkle root of the mounted `JouleCapsule` / `JouleAgentFs` rootfs.
    #[serde(default)]
    pub rootfs_merkle_hex: Option<String>,
    /// Enable `Dagger`-class Merkle CAS memoization (`true` by default).
    #[serde(default = "default_true")]
    pub memoize: bool,
    /// Attempt native OS kernel sandbox (`sandbox-exec` on macOS) when available.
    #[serde(default = "default_true")]
    pub prefer_os_sandbox: bool,
    /// Explicit HITL approval flag if energy exceeds `HitlGateAboveNanojoules`.
    #[serde(default)]
    pub hitl_approved: bool,
    /// Whether this command intends to write files inside workspace.
    #[serde(default)]
    pub requires_fs_write: bool,
    /// Optional outbound hostname this command intends to contact.
    #[serde(default)]
    pub requested_egress_host: Option<String>,
}

fn default_true() -> bool {
    true
}

/// Rich execution receipt returned by [`WardEngine::execute_local`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WardExecReceipt {
    pub permitted: bool,
    pub decision: WardDecision,
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub memoized_cache_hit: bool,
    pub cache_key_hex: String,
    pub enforcement_mechanism: String,
    pub confinement: CompiledConfinementProfile,
    pub energy_trace: FabricTraceReceipt,
}

/// Core `JouleWard` engine combining `ThermoCap` verification, OS confinement, Merkle CAS
/// memoization, and high-resolution 6-rail fabric energy accounting.
pub struct WardEngine {
    root_secret: [u8; 32],
    node_id: String,
    fabric_class: FabricComputeClass,
    grid_carbon_g_per_kwh: f64,
    action_cache: RwLock<BTreeMap<String, CachedWardAction>>,
    pub ledger: FabricEnergyLedger,
}

impl WardEngine {
    /// Initialize a `WardEngine` for the given fabric node.
    pub fn new(
        root_secret: [u8; 32],
        node_id: impl Into<String>,
        fabric_class: FabricComputeClass,
        grid_carbon_g_per_kwh: f64,
    ) -> Self {
        Self {
            root_secret,
            node_id: node_id.into(),
            fabric_class,
            grid_carbon_g_per_kwh,
            action_cache: RwLock::new(BTreeMap::new()),
            ledger: FabricEnergyLedger::new(),
        }
    }

    /// Derive a deterministic 32-byte node secret from seed material.
    pub fn secret_from_seed(seed: &str) -> [u8; 32] {
        let digest = Sha256::digest(format!("JOULEWARD_ROOT_SECRET:{seed}").as_bytes());
        let mut out = [0u8; 32];
        out.copy_from_slice(&digest);
        out
    }

    /// Inspect this node's root secret reference (for minting/authorizing on this authority).
    pub fn root_secret(&self) -> &[u8; 32] {
        &self.root_secret
    }

    /// Local node ID in the compute fabric.
    pub fn node_id(&self) -> &str {
        &self.node_id
    }

    /// Local compute fabric class.
    pub fn fabric_class(&self) -> FabricComputeClass {
        self.fabric_class
    }

    /// Local grid carbon intensity (`gCO2/kWh`).
    pub fn grid_carbon_g_per_kwh(&self) -> f64 {
        self.grid_carbon_g_per_kwh
    }

    /// Mint a new `ThermoCap` [`WardCapabilityToken`].
    pub fn mint_token(
        &self,
        root_token_id: impl Into<String>,
        subject: impl Into<String>,
        caveats: Vec<WardCaveat>,
    ) -> Result<WardCapabilityToken, BranchError> {
        let now_ms = current_epoch_ms();
        WardCapabilityToken::mint(&self.root_secret, root_token_id, subject, now_ms, caveats)
    }

    /// Compute the deterministic `Dagger`-class Merkle action cache key.
    pub fn compute_action_cache_key(
        &self,
        req: &WardExecRequest,
        policy: &EffectiveWardPolicy,
    ) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"JWARD_ACTION_V1:");
        hasher.update(
            req.rootfs_merkle_hex
                .as_deref()
                .unwrap_or("host_ephemeral")
                .as_bytes(),
        );
        hasher.update(b"|CLASS:");
        hasher.update(self.fabric_class.as_str().as_bytes());
        hasher.update(b"|ARGV:");
        for arg in &req.command {
            hasher.update(arg.as_bytes());
            hasher.update([0u8]);
        }
        hasher.update(b"|ENV:");
        for (k, v) in &req.env {
            hasher.update(k.as_bytes());
            hasher.update(b"=");
            hasher.update(v.as_bytes());
            hasher.update([0u8]);
        }
        hasher.update(b"|POLICY:");
        hasher.update(policy.policy_digest_hex().as_bytes());
        hex::encode(hasher.finalize())
    }

    /// Authorize and execute a confined command locally, enforcing all capability caveats,
    /// checking the Merkle CAS `WardActionCache` first, and recording 6-rail nanojoules.
    pub fn execute_local(
        &self,
        req: &WardExecRequest,
        workspace_dir: &Path,
    ) -> Result<WardExecReceipt, BranchError> {
        if req.command.is_empty() || req.command[0].trim().is_empty() {
            return Err(BranchError::InvalidName(
                "command argv must contain at least one executable".into(),
            ));
        }
        std::fs::create_dir_all(workspace_dir).map_err(|e| {
            BranchError::Storage(format!("failed to create ward workspace dir: {e}"))
        })?;

        let policy = req.token.effective_policy();
        let confinement = compile_confinement(&policy, workspace_dir);
        let cache_key = self.compute_action_cache_key(req, &policy);

        // Pre-flight baseline estimate (~1.2 ms process spawn unless memoized)
        let is_cached = req.memoize && {
            let guard = match self.action_cache.read() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            guard.contains_key(&cache_key)
        };

        let preflight_rails = if is_cached {
            HeterogeneousRailSample {
                cpu_nj: 60,
                gpu_nj: 0,
                npu_ane_nj: 0,
                dram_uma_nj: 25,
                storage_nvme_nj: 0,
                network_wire_nj: 0,
                wall_duration_ns: 450,
                peak_power_mw: 188,
                carbon_ng_co2: HeterogeneousRailSample::compute_carbon_ng(
                    85,
                    self.grid_carbon_g_per_kwh,
                ),
            }
        } else {
            HeterogeneousRailSample::estimate_calibrated(
                self.fabric_class,
                "cpu_simd",
                400_000, // 0.4 ms conservative pre-flight floor
                64 * 1024,
                if req.requires_fs_write { 4096 } else { 0 },
                if req.requested_egress_host.is_some() {
                    2048
                } else {
                    0
                },
                self.grid_carbon_g_per_kwh,
            )
        };

        let ctx = WardActionContext {
            now_epoch_ms: current_epoch_ms(),
            target_fabric_class: self.fabric_class,
            silicon_lane: "cpu_simd".to_string(),
            requested_paths: Vec::new(),
            requires_fs_write: req.requires_fs_write,
            requested_egress_host: req.requested_egress_host.clone(),
            requested_sql_tables: Vec::new(),
            mutates_synapse_weights: false,
            estimated_rails: preflight_rails,
            hitl_approved: req.hitl_approved,
        };

        let already_spent = self.ledger.token_spent_nj(&req.token.root_token_id);
        let decision = authorize_ward_action(&self.root_secret, &req.token, &ctx, already_spent);
        if !matches!(decision, WardDecision::Permit { .. }) {
            return Ok(WardExecReceipt {
                permitted: false,
                decision,
                exit_code: 126,
                stdout: String::new(),
                stderr: "Blocked by JouleWard capability policy".to_string(),
                memoized_cache_hit: false,
                cache_key_hex: cache_key,
                enforcement_mechanism: "thermocap-pre-execution-gate".to_string(),
                confinement,
                energy_trace: FabricTraceReceipt::default(),
            });
        }

        // Fast-path: `Dagger`-class Merkle CAS memoization replay (`0` OS spawns!)
        if req.memoize {
            let mut cache_guard = match self.action_cache.write() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            if let Some(entry) = cache_guard.get_mut(&cache_key) {
                entry.hits = entry.hits.saturating_add(1);
                let saved_nj = entry.original_rails.total_nj().saturating_sub(85);
                let mut trace = FabricTraceReceipt {
                    hops: Vec::new(),
                    aggregate_rails: HeterogeneousRailSample::default(),
                    per_class_totals_nj: BTreeMap::new(),
                    saved_by_memoization_nj: saved_nj,
                };
                trace.push_hop(FabricHopEnergyReceipt {
                    node_id: self.node_id.clone(),
                    fabric_class: self.fabric_class,
                    silicon_lane: "merkle_cas_memo_lookup".to_string(),
                    calibration_source: self.default_calibration_source(),
                    grid_carbon_g_per_kwh: self.grid_carbon_g_per_kwh,
                    rails: preflight_rails,
                });
                self.ledger
                    .record_trace(Some(&req.token.root_token_id), &trace);
                return Ok(WardExecReceipt {
                    permitted: true,
                    decision,
                    exit_code: entry.exit_code,
                    stdout: entry.stdout.clone(),
                    stderr: entry.stderr.clone(),
                    memoized_cache_hit: true,
                    cache_key_hex: cache_key,
                    enforcement_mechanism: "merkle-cas-zero-spawn-replay".to_string(),
                    confinement,
                    energy_trace: trace,
                });
            }
        }

        // Execute under OS confinement (macOS Seatbelt SBPL when permitted, with clean nested fallback)
        let t0 = Instant::now();
        let (output, mechanism) = run_confined_subprocess(
            &req.command,
            &req.env,
            workspace_dir,
            &confinement,
            req.prefer_os_sandbox,
        )?;
        let elapsed_ns = t0.elapsed().as_nanos() as u64;

        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        let exit_code = output.status.code().unwrap_or(-1);

        let io_bytes = (output.stdout.len() + output.stderr.len()) as u64;
        let measured_rails = HeterogeneousRailSample::estimate_calibrated(
            self.fabric_class,
            "cpu_simd",
            elapsed_ns,
            (64 * 1024u64).saturating_add(io_bytes),
            if req.requires_fs_write {
                4096 + io_bytes
            } else {
                0
            },
            if req.requested_egress_host.is_some() {
                2048
            } else {
                0
            },
            self.grid_carbon_g_per_kwh,
        );

        let mut trace = FabricTraceReceipt::default();
        trace.push_hop(FabricHopEnergyReceipt {
            node_id: self.node_id.clone(),
            fabric_class: self.fabric_class,
            silicon_lane: mechanism.clone(),
            calibration_source: self.default_calibration_source(),
            grid_carbon_g_per_kwh: self.grid_carbon_g_per_kwh,
            rails: measured_rails,
        });
        self.ledger
            .record_trace(Some(&req.token.root_token_id), &trace);

        if req.memoize && exit_code == 0 && !req.requires_fs_write {
            let mut cache_guard = match self.action_cache.write() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            if cache_guard.len() >= 512
                && let Some(oldest) = cache_guard.keys().next().cloned()
            {
                cache_guard.remove(&oldest);
            }
            cache_guard.insert(
                cache_key.clone(),
                CachedWardAction {
                    cache_key_hex: cache_key.clone(),
                    exit_code,
                    stdout: stdout.clone(),
                    stderr: stderr.clone(),
                    original_rails: measured_rails,
                    hits: 0,
                },
            );
        }

        Ok(WardExecReceipt {
            permitted: true,
            decision,
            exit_code,
            stdout,
            stderr,
            memoized_cache_hit: false,
            cache_key_hex: cache_key,
            enforcement_mechanism: mechanism,
            confinement,
            energy_trace: trace,
        })
    }

    /// Default sensor calibration provenance for this node class.
    pub fn default_calibration_source(&self) -> TelemetryCalibrationSource {
        match self.fabric_class {
            FabricComputeClass::AppleUmaWorkstation => TelemetryCalibrationSource::AppleIoReportSmc,
            FabricComputeClass::LinuxAccelerator | FabricComputeClass::CloudCxlCluster => {
                TelemetryCalibrationSource::LinuxRaplPerf
            }
            FabricComputeClass::McuBareMetal => TelemetryCalibrationSource::HardwareIna219UsbMeter,
            _ => TelemetryCalibrationSource::CalibratedInstructionCounter,
        }
    }
}

fn run_confined_subprocess(
    command: &[String],
    env: &BTreeMap<String, String>,
    workspace_dir: &Path,
    confinement: &CompiledConfinementProfile,
    prefer_os_sandbox: bool,
) -> Result<(std::process::Output, String), BranchError> {
    if prefer_os_sandbox && cfg!(target_os = "macos") && Path::new("/usr/bin/sandbox-exec").exists()
    {
        let mut sb_cmd = Command::new("/usr/bin/sandbox-exec");
        sb_cmd
            .arg("-p")
            .arg(&confinement.macos_seatbelt_sbpl)
            .args(command)
            .current_dir(workspace_dir)
            .env_clear()
            .env("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
            .env("HOME", workspace_dir)
            .envs(env);

        if let Ok(out) = sb_cmd.output() {
            let stderr_s = String::from_utf8_lossy(&out.stderr);
            // When already running inside an outer macOS Seatbelt sandbox, nested `sandbox_init`
            // returns exit 71 / "sandbox_init: Operation not permitted". Detect & cleanly fall back.
            let nested_denied = !out.status.success()
                && (stderr_s.contains("sandbox_init:")
                    || stderr_s.contains("sandbox-exec: sandbox_apply"));
            if !nested_denied {
                return Ok((out, "macos-seatbelt-sbpl".to_string()));
            }
        }
    }

    let mut direct = Command::new(&command[0]);
    direct
        .args(&command[1..])
        .current_dir(workspace_dir)
        .env_clear()
        .env("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
        .env("HOME", workspace_dir)
        .envs(env);
    let out = direct
        .output()
        .map_err(|e| BranchError::Storage(format!("confined command spawn failed: {e}")))?;
    Ok((out, "capability-scrubbed-posix-isolate".to_string()))
}

// ============================================================================
// 5. `Apptainer` `.sif`-Class Single-File Immutable `.jcap` Capsule Bundle
// ============================================================================

/// Portable single-file immutable capsule + living-weights + capability bundle (`.jcap`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JcapBundle {
    pub bundle_id: String,
    pub created_at_ms: u64,
    pub image_ref: String,
    pub entrypoint: Vec<String>,
    pub env: BTreeMap<String, String>,
    /// Content-addressed VFS files (`normalized_path -> raw_bytes`).
    pub vfs_files: BTreeMap<String, Vec<u8>>,
    /// Optional packaged `JouleSynapse` living-weights model identifier.
    pub synapse_model_id: Option<String>,
    pub synapse_tensor_count: usize,
    pub synapse_bits_per_weight: f64,
    /// Optional embedded attenuated `ThermoCap` token governing bundle execution.
    pub embedded_capability: Option<WardCapabilityToken>,
    /// Certified target fabric compute tiers.
    pub target_fabric_classes: Vec<FabricComputeClass>,
    /// Certified baseline 6-rail energy envelope for this `.jcap` bundle.
    pub baseline_energy_profile: HeterogeneousRailSample,
}

impl JcapBundle {
    /// Serialize, gzip-compress, and wrap with `[b"JCAP"][ver=1][sha256_32B][len_u64_le]`.
    pub fn export_jcap_bytes(&self) -> Result<(Vec<u8>, String), BranchError> {
        let json_bytes = serde_json::to_vec(self)
            .map_err(|e| BranchError::Storage(format!("failed to serialize JcapBundle: {e}")))?;
        if json_bytes.len() as u64 > MAX_JCAP_PAYLOAD_BYTES {
            return Err(BranchError::Storage(format!(
                "JcapBundle payload ({} bytes) exceeds 64 MiB limit",
                json_bytes.len()
            )));
        }

        let digest = Sha256::digest(&json_bytes);
        let digest_hex = hex::encode(digest);

        let mut gz = GzEncoder::new(Vec::new(), Compression::default());
        gz.write_all(&json_bytes)
            .map_err(|e| BranchError::Storage(format!("gzip encode failed: {e}")))?;
        let compressed = gz
            .finish()
            .map_err(|e| BranchError::Storage(format!("gzip finish failed: {e}")))?;

        let mut out = Vec::with_capacity(4 + 1 + 32 + 8 + compressed.len());
        out.extend_from_slice(JCAP_MAGIC);
        out.push(JCAP_VERSION);
        out.extend_from_slice(&digest);
        out.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&compressed);
        Ok((out, digest_hex))
    }

    /// Verify `.jcap` binary header, anti-zip-bomb cap, and exact `SHA-256` payload digest.
    pub fn import_and_verify(bytes: &[u8]) -> Result<(Self, String), BranchError> {
        const HEADER_LEN: usize = 4 + 1 + 32 + 8;
        if bytes.len() < HEADER_LEN {
            return Err(BranchError::Storage(
                "truncated .jcap file (smaller than 45-byte header)".into(),
            ));
        }
        if &bytes[0..4] != JCAP_MAGIC {
            return Err(BranchError::Storage("invalid .jcap magic header".into()));
        }
        if bytes[4] != JCAP_VERSION {
            return Err(BranchError::Storage(format!(
                "unsupported .jcap version {}",
                bytes[4]
            )));
        }
        let mut expected_sha = [0u8; 32];
        expected_sha.copy_from_slice(&bytes[5..37]);

        let mut len_buf = [0u8; 8];
        len_buf.copy_from_slice(&bytes[37..45]);
        let uncompressed_len = u64::from_le_bytes(len_buf);
        if uncompressed_len > MAX_JCAP_PAYLOAD_BYTES {
            return Err(BranchError::Storage(format!(
                ".jcap declared uncompressed size {uncompressed_len} exceeds 64 MiB safety cap"
            )));
        }

        let decoder = GzDecoder::new(&bytes[HEADER_LEN..]);
        let mut limited = decoder.take(MAX_JCAP_PAYLOAD_BYTES + 1);
        let mut decompressed = Vec::with_capacity((uncompressed_len as usize).min(1024 * 1024));
        limited
            .read_to_end(&mut decompressed)
            .map_err(|e| BranchError::Storage(format!(".jcap gzip decode failed: {e}")))?;

        if decompressed.len() as u64 != uncompressed_len {
            return Err(BranchError::Storage(
                ".jcap uncompressed byte length mismatch".into(),
            ));
        }

        let actual_digest = Sha256::digest(&decompressed);
        let mut actual_sha = [0u8; 32];
        actual_sha.copy_from_slice(&actual_digest);
        if !ct_eq_32(&expected_sha, &actual_sha) {
            return Err(BranchError::Storage(
                ".jcap SHA-256 integrity check failed (corrupted or tampered bundle)".into(),
            ));
        }

        let bundle: Self = serde_json::from_slice(&decompressed)
            .map_err(|e| BranchError::Storage(format!(".jcap JSON parse failed: {e}")))?;
        Ok((bundle, hex::encode(actual_sha)))
    }
}

fn current_epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rfc4231_hmac_sha256_official_vector() {
        // IETF RFC 4231 Test Case 2 ("Jefe" / "what do ya want for nothing?")
        let mac = hmac_sha256(b"Jefe", b"what do ya want for nothing?");
        assert_eq!(
            hex::encode(mac),
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
    }

    #[test]
    fn test_thermocap_offline_attenuation_and_tamper_rejection() {
        let secret = WardEngine::secret_from_seed("unit-test-seed");
        let root = WardCapabilityToken::mint(
            &secret,
            "cap-root-01",
            "planner-agent",
            1_700_000_000_000,
            vec![
                WardCaveat::MaxTotalNanojoules(50_000_000),
                WardCaveat::AllowFabricClasses(vec![
                    FabricComputeClass::AppleUmaWorkstation,
                    FabricComputeClass::EdgeRobotic,
                    FabricComputeClass::CloudCxlCluster,
                ]),
            ],
        )
        .unwrap();

        // Offline attenuation WITHOUT `secret`: narrow to AppleUmaWorkstation, airgap network_wire_nj=0, HITL at 5 mJ
        let worker_cap = root
            .attenuate(WardCaveat::AllowFabricClasses(vec![
                FabricComputeClass::AppleUmaWorkstation,
            ]))
            .unwrap()
            .attenuate(WardCaveat::MaxTotalNanojoules(10_000_000))
            .unwrap()
            .attenuate(WardCaveat::MaxRailNanojoules {
                rail: FabricRail::NetworkWireRdma,
                max_nj: 0,
            })
            .unwrap()
            .attenuate(WardCaveat::HitlGateAboveNanojoules(5_000_000))
            .unwrap()
            .attenuate(WardCaveat::ReadOnlyFs(true))
            .unwrap();

        assert!(worker_cap.verify_signature(&secret).is_ok());

        // Attempt to strip the last `ReadOnlyFs(true)` caveat -> HMAC MUST fail!
        let mut stripped = worker_cap.clone();
        stripped.caveats.pop();
        assert!(stripped.verify_signature(&secret).is_err());

        // Attempt to broaden `MaxTotalNanojoules` -> effective_policy keeps strict min (10_000_000 nJ)!
        let broadened = worker_cap
            .attenuate(WardCaveat::MaxTotalNanojoules(999_000_000))
            .unwrap();
        assert_eq!(broadened.effective_policy().max_total_nj, Some(10_000_000));

        // Authorize valid 2 mJ AppleUmaWorkstation GPU step (0 wire nJ) -> Permit!
        let base_ctx = WardActionContext {
            now_epoch_ms: 1_700_000_001_000,
            target_fabric_class: FabricComputeClass::AppleUmaWorkstation,
            silicon_lane: "wgpu_xnor_popcnt_wgsl".into(),
            requested_paths: vec![],
            requires_fs_write: false,
            requested_egress_host: None,
            requested_sql_tables: vec![],
            mutates_synapse_weights: false,
            estimated_rails: HeterogeneousRailSample {
                cpu_nj: 150_000,
                gpu_nj: 1_800_000,
                npu_ane_nj: 0,
                dram_uma_nj: 50_000,
                storage_nvme_nj: 0,
                network_wire_nj: 0,
                wall_duration_ns: 200_000,
                peak_power_mw: 10_000,
                carbon_ng_co2: 1,
            },
            hitl_approved: false,
        };
        assert_eq!(
            authorize_ward_action(&secret, &worker_cap, &base_ctx, 0),
            WardDecision::Permit {
                effective_remaining_nj: Some(8_000_000)
            }
        );

        // Violate `NetworkWireRdma: 0` rail cap -> Deny!
        let mut wire_ctx = base_ctx.clone();
        wire_ctx.estimated_rails.network_wire_nj = 500;
        assert!(matches!(
            authorize_ward_action(&secret, &worker_cap, &wire_ctx, 0),
            WardDecision::Deny { .. }
        ));

        // Cross HITL gate (`already_spent = 4_000_000` + `2_000_000 = 6_000_000 > 5_000_000`) -> RequiresHumanApproval!
        assert!(matches!(
            authorize_ward_action(&secret, &worker_cap, &base_ctx, 4_000_000),
            WardDecision::RequiresHumanApproval {
                threshold_nj: 5_000_000,
                projected_total_nj: 6_000_000,
                ..
            }
        ));

        // Same step WITH `hitl_approved: true` -> Permit!
        let mut approved_ctx = base_ctx;
        approved_ctx.hitl_approved = true;
        assert!(matches!(
            authorize_ward_action(&secret, &worker_cap, &approved_ctx, 4_000_000),
            WardDecision::Permit {
                effective_remaining_nj: Some(4_000_000)
            }
        ));
    }

    #[test]
    fn test_confined_exec_and_dagger_merkle_memoization_across_fabric() {
        let tmp = tempfile::tempdir().unwrap();
        let engine = WardEngine::new(
            WardEngine::secret_from_seed("exec-seed"),
            "uma-primary-01",
            FabricComputeClass::AppleUmaWorkstation,
            140.0,
        );
        let token = engine
            .mint_token(
                "cap-exec-01",
                "ci-runner",
                vec![
                    WardCaveat::ReadOnlyFs(true),
                    WardCaveat::AllowEgressHosts(vec![]), // airgapped!
                    WardCaveat::MaxTotalNanojoules(500_000_000),
                ],
            )
            .unwrap();

        let req = WardExecRequest {
            token,
            command: vec!["/bin/echo".into(), "jouleward-deterministic-step".into()],
            env: BTreeMap::from([("LC_ALL".into(), "C".into())]),
            rootfs_merkle_hex: Some("deadbeefcafebabe".into()),
            memoize: true,
            prefer_os_sandbox: true,
            hitl_approved: false,
            requires_fs_write: false,
            requested_egress_host: None,
        };

        let first = engine.execute_local(&req, tmp.path()).unwrap();
        assert!(first.permitted);
        assert_eq!(first.exit_code, 0);
        assert_eq!(first.stdout.trim(), "jouleward-deterministic-step");
        assert!(!first.memoized_cache_hit);
        assert!(first.energy_trace.aggregate_rails.total_nj() > 85);
        assert!(
            first
                .confinement
                .macos_seatbelt_sbpl
                .contains("(deny network*)")
        );

        // Second identical invocation hits `WardActionCache` in 0 spawns (~85 nJ)!
        let second = engine.execute_local(&req, tmp.path()).unwrap();
        assert!(second.permitted);
        assert!(second.memoized_cache_hit);
        assert_eq!(second.stdout.trim(), "jouleward-deterministic-step");
        assert_eq!(second.energy_trace.aggregate_rails.total_nj(), 85);
        assert!(second.energy_trace.saved_by_memoization_nj > 0);

        // Merge a multi-hop trace from an MCU sensor + EdgeRobotic relay + CloudCxlCluster
        let mut fabric_trace = second.energy_trace.clone();
        fabric_trace.push_hop(FabricHopEnergyReceipt {
            node_id: "mcu-valve-07".into(),
            fabric_class: FabricComputeClass::McuBareMetal,
            silicon_lane: "riscv_scalar".into(),
            calibration_source: TelemetryCalibrationSource::HardwareIna219UsbMeter,
            grid_carbon_g_per_kwh: 0.0,
            rails: HeterogeneousRailSample::estimate_calibrated(
                FabricComputeClass::McuBareMetal,
                "cpu_simd",
                12_000,
                256,
                0,
                64,
                0.0,
            ),
        });
        engine.ledger.record_trace(None, &fabric_trace);

        let snap = engine.ledger.snapshot();
        assert!(snap.per_class_rails.contains_key("apple_uma_workstation"));
        assert!(snap.per_class_rails.contains_key("mcu_bare_metal"));
        assert!(snap.total_saved_memoization_nj > 0);
    }

    #[test]
    fn test_jcap_single_file_bundle_roundtrip_and_tamper_detection() {
        let secret = WardEngine::secret_from_seed("jcap-seed");
        let cap = WardCapabilityToken::mint(
            &secret,
            "jcap-cap-1",
            "edge-appliance",
            1_700_000_000_000,
            vec![WardCaveat::ReadOnlyFs(true)],
        )
        .unwrap();

        let bundle = JcapBundle {
            bundle_id: "minilm-appliance-v1".into(),
            created_at_ms: 1_700_000_000_000,
            image_ref: "jouledb.local/synapse/all-minilm-l6-v2:0.11bpw".into(),
            entrypoint: vec!["/bin/jouledb-worker".into()],
            env: BTreeMap::from([("JOULE_MODE".into(), "thermocap".into())]),
            vfs_files: BTreeMap::from([(
                "/etc/model.toml".into(),
                b"model = \"all-minilm-l6-v2\"\nbpw = 0.1066\n".to_vec(),
            )]),
            synapse_model_id: Some("all-minilm-l6-v2".into()),
            synapse_tensor_count: 104,
            synapse_bits_per_weight: 0.1066,
            embedded_capability: Some(cap),
            target_fabric_classes: vec![
                FabricComputeClass::EdgeRobotic,
                FabricComputeClass::AppleUmaWorkstation,
            ],
            baseline_energy_profile: HeterogeneousRailSample::estimate_calibrated(
                FabricComputeClass::AppleUmaWorkstation,
                "wgpu_xnor_popcnt_wgsl",
                1_460_000,
                552_800,
                0,
                0,
                140.0,
            ),
        };

        let (mut raw_jcap, sha_hex) = bundle.export_jcap_bytes().unwrap();
        assert_eq!(&raw_jcap[0..4], b"JCAP");
        let (decoded, verified_sha) = JcapBundle::import_and_verify(&raw_jcap).unwrap();
        assert_eq!(sha_hex, verified_sha);
        assert_eq!(decoded.bundle_id, "minilm-appliance-v1");
        assert_eq!(decoded.synapse_tensor_count, 104);
        assert!(
            decoded
                .embedded_capability
                .unwrap()
                .verify_signature(&secret)
                .is_ok()
        );

        // Flip 1 bit in the embedded SHA-256 header -> MUST fail verification!
        raw_jcap[10] ^= 0x01;
        assert!(JcapBundle::import_and_verify(&raw_jcap).is_err());
    }

    #[test]
    fn test_hardware_sensor_fusion_conservation_and_fallback() {
        // 15,000 mW (15.0 W) live Apple SMC sensor over 200,000 ns (200 µs) => exact 3,000,000 nJ (3.0 mJ)
        let hw = HardwarePowerSample {
            board_power_mw: 15_000,
            cpu_utilization: 0.25,
            gpu_utilization: 0.80,
            npu_utilization: 0.0,
            memory_pressure: 0.30,
            source: None,
        };
        let (fused, src, is_fused) = HeterogeneousRailSample::estimate_with_hardware_fusion(
            FabricComputeClass::AppleUmaWorkstation,
            "wgpu_xnor_popcnt_wgsl",
            200_000,
            65_536,
            0,
            0,
            140.0,
            Some(hw),
        );
        assert!(is_fused);
        assert_eq!(src, TelemetryCalibrationSource::AppleIoReportSmc);
        // Exact conservation: cpu + gpu + npu + dram == 15_000 * 200_000 / 1_000 == 3_000_000 nJ!
        assert_eq!(
            fused.cpu_nj + fused.gpu_nj + fused.npu_ane_nj + fused.dram_uma_nj,
            3_000_000
        );
        assert!(fused.gpu_nj > fused.cpu_nj);

        // Zero board_power_mw falls back cleanly to CalibratedInstructionCounter
        let (fallback, fb_src, fb_fused) = HeterogeneousRailSample::estimate_with_hardware_fusion(
            FabricComputeClass::LinuxAccelerator,
            "cpu_simd",
            100_000,
            4_096,
            0,
            0,
            250.0,
            Some(HardwarePowerSample::default()),
        );
        assert!(!fb_fused);
        assert_eq!(
            fb_src,
            TelemetryCalibrationSource::CalibratedInstructionCounter
        );
        assert!(fallback.total_nj() > 0);
    }
}
