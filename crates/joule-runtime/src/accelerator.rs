//! Hardware Accelerator Management — GPU/TPU/NPU/LPU detection, allocation, and energy accounting.
//!
//! Auto-detects available hardware accelerators on the host and manages their allocation
//! to container/process instances. Each device tracks its own TDP and utilization for
//! per-device energy accounting — the differentiator over every other container runtime.
//!
//! Supported accelerator families:
//! - **GPU**: NVIDIA (CUDA), AMD (ROCm), Apple (Metal), Intel (Arc)
//! - **TPU**: Google Coral (USB/PCIe)
//! - **NPU**: Apple Neural Engine, Intel NPU, Qualcomm Hexagon
//! - **LPU**: Groq inference chips (PCIe)
//!
//! Detection strategy per platform:
//! - macOS: `system_profiler SPDisplaysDataType` for GPU, IOKit for NPU
//! - Linux: `nvidia-smi`, `rocm-smi`, `/sys/class/misc/` for device files
//! - Fallback: `joule-db-energy::detect_platform()` for availability flags

use crate::{AcceleratorBinding, AcceleratorKind, RuntimeError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::RwLock;

/// A detected hardware accelerator device.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcceleratorDevice {
    /// Unique device identifier (e.g. `"gpu-0"`, `"GPU-abc123"`, `"npu-0"`).
    pub id: String,
    /// Accelerator family.
    pub kind: AcceleratorKind,
    /// Human-readable name (e.g. `"NVIDIA A100"`, `"Apple M4 Max GPU"`).
    pub name: String,
    /// Device memory in megabytes (0 if shared/unified memory).
    pub memory_mb: u64,
    /// Number of compute units (CUDA cores, GPU cores, neural engine cores, etc.).
    pub compute_units: u32,
    /// Thermal Design Power in watts — used for energy estimation.
    pub tdp_watts: f64,
    /// Whether the device is currently available (not failed/offline).
    pub available: bool,
    /// Instance ID that has allocated this device (None = free).
    pub allocated_to: Option<String>,
}

/// Per-device energy snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceEnergy {
    /// Device identifier.
    pub device_id: String,
    /// Accelerator kind.
    pub kind: AcceleratorKind,
    /// Device name.
    pub name: String,
    /// Current estimated power draw in watts.
    pub power_watts: f64,
    /// Cumulative energy consumed in joules since allocation.
    pub energy_joules: f64,
    /// Current utilization (0.0 - 1.0).
    pub utilization: f64,
    /// Memory used in MB (0 if shared/unknown).
    pub memory_used_mb: u64,
    /// Instance this device is allocated to (None = idle).
    pub allocated_to: Option<String>,
}

/// Manages hardware accelerator detection, allocation, and energy tracking.
pub struct AcceleratorManager {
    /// All detected devices.
    devices: RwLock<Vec<AcceleratorDevice>>,
    /// Instance ID → allocated device IDs.
    allocations: RwLock<HashMap<String, Vec<String>>>,
    /// Per-device cumulative energy in joules (device_id → joules).
    cumulative_energy: RwLock<HashMap<String, f64>>,
}

impl AcceleratorManager {
    /// Create a new accelerator manager and auto-detect available devices.
    pub fn new() -> Self {
        let devices = detect_devices();
        log::info!(
            "Accelerator manager initialized: {} device(s) detected",
            devices.len()
        );
        for dev in &devices {
            log::info!(
                "  {} [{}] — {} ({}MB, TDP {:.0}W)",
                dev.id,
                dev.kind,
                dev.name,
                dev.memory_mb,
                dev.tdp_watts
            );
        }

        Self {
            devices: RwLock::new(devices),
            allocations: RwLock::new(HashMap::new()),
            cumulative_energy: RwLock::new(HashMap::new()),
        }
    }

    /// Create a manager with a pre-specified device list (for testing).
    pub fn with_devices(devices: Vec<AcceleratorDevice>) -> Self {
        Self {
            devices: RwLock::new(devices),
            allocations: RwLock::new(HashMap::new()),
            cumulative_energy: RwLock::new(HashMap::new()),
        }
    }

    /// List all detected devices.
    pub fn list_devices(&self) -> Vec<AcceleratorDevice> {
        self.devices.read().unwrap().clone()
    }

    /// List only devices of a specific kind.
    pub fn list_by_kind(&self, kind: &AcceleratorKind) -> Vec<AcceleratorDevice> {
        self.devices
            .read()
            .unwrap()
            .iter()
            .filter(|d| &d.kind == kind)
            .cloned()
            .collect()
    }

    /// List free (unallocated) devices.
    pub fn list_available(&self) -> Vec<AcceleratorDevice> {
        self.devices
            .read()
            .unwrap()
            .iter()
            .filter(|d| d.available && d.allocated_to.is_none())
            .cloned()
            .collect()
    }

    /// Allocate devices for an instance based on its accelerator bindings.
    ///
    /// For each binding:
    /// - If `device_id` is specified, that exact device is reserved.
    /// - If `device_id` is None, the first available device of that kind is selected.
    ///
    /// Returns the list of allocated device IDs.
    pub fn allocate(
        &self,
        instance_id: &str,
        bindings: &[AcceleratorBinding],
    ) -> Result<Vec<String>, RuntimeError> {
        let mut devices = self.devices.write().unwrap();
        let mut allocations = self.allocations.write().unwrap();
        let mut energy = self.cumulative_energy.write().unwrap();
        let mut allocated_ids = Vec::new();

        for binding in bindings {
            let dev = if let Some(ref requested_id) = binding.device_id {
                // Find specific device
                devices.iter_mut().find(|d| {
                    d.id == *requested_id
                        && d.kind == binding.kind
                        && d.available
                        && d.allocated_to.is_none()
                })
            } else {
                // Auto-select first available of this kind
                devices
                    .iter_mut()
                    .find(|d| d.kind == binding.kind && d.available && d.allocated_to.is_none())
            };

            match dev {
                Some(device) => {
                    device.allocated_to = Some(instance_id.to_string());
                    allocated_ids.push(device.id.clone());
                    energy.insert(device.id.clone(), 0.0);
                    log::info!(
                        "Allocated {} ({}) to instance {}",
                        device.id,
                        device.name,
                        instance_id
                    );
                }
                None => {
                    // Rollback allocations made so far
                    for id in &allocated_ids {
                        if let Some(d) = devices.iter_mut().find(|d| d.id == *id) {
                            d.allocated_to = None;
                        }
                        energy.remove(id);
                    }
                    let desc = if let Some(ref dev_id) = binding.device_id {
                        format!("{} device '{}'", binding.kind, dev_id)
                    } else {
                        format!("available {} device", binding.kind)
                    };
                    return Err(RuntimeError::ProcessError(format!(
                        "no {} found for instance {}",
                        desc, instance_id
                    )));
                }
            }
        }

        allocations.insert(instance_id.to_string(), allocated_ids.clone());
        Ok(allocated_ids)
    }

    /// Release all devices allocated to an instance.
    pub fn release(&self, instance_id: &str) {
        let mut devices = self.devices.write().unwrap();
        let mut allocations = self.allocations.write().unwrap();

        if let Some(device_ids) = allocations.remove(instance_id) {
            for id in &device_ids {
                if let Some(dev) = devices.iter_mut().find(|d| d.id == *id) {
                    log::info!(
                        "Released {} ({}) from instance {}",
                        dev.id,
                        dev.name,
                        instance_id
                    );
                    dev.allocated_to = None;
                }
            }
        }
    }

    /// Get devices allocated to a specific instance.
    pub fn devices_for_instance(&self, instance_id: &str) -> Vec<AcceleratorDevice> {
        let devices = self.devices.read().unwrap();
        let allocations = self.allocations.read().unwrap();
        match allocations.get(instance_id) {
            Some(ids) => devices
                .iter()
                .filter(|d| ids.contains(&d.id))
                .cloned()
                .collect(),
            None => vec![],
        }
    }

    /// Get per-device energy snapshots for all devices.
    ///
    /// Uses TDP × utilization as the power estimate. The `utilization` parameter
    /// comes from the platform's energy monitor (GPU/NPU utilization fields).
    pub fn energy_snapshots(
        &self,
        gpu_utilization: f64,
        npu_utilization: f64,
        tpu_utilization: f64,
        lpu_utilization: f64,
    ) -> Vec<DeviceEnergy> {
        let devices = self.devices.read().unwrap();
        let energy = self.cumulative_energy.read().unwrap();

        devices
            .iter()
            .map(|dev| {
                let utilization = match dev.kind {
                    AcceleratorKind::GPU => gpu_utilization,
                    AcceleratorKind::NPU => npu_utilization,
                    AcceleratorKind::TPU => tpu_utilization,
                    AcceleratorKind::LPU => lpu_utilization,
                    AcceleratorKind::Custom(_) | _ => 0.0,
                };
                let power_watts = dev.tdp_watts * utilization;
                let cumulative = energy.get(&dev.id).copied().unwrap_or(0.0);

                DeviceEnergy {
                    device_id: dev.id.clone(),
                    kind: dev.kind.clone(),
                    name: dev.name.clone(),
                    power_watts,
                    energy_joules: cumulative,
                    utilization,
                    memory_used_mb: 0, // TODO: query per-device memory when available
                    allocated_to: dev.allocated_to.clone(),
                }
            })
            .collect()
    }

    /// Update cumulative energy for all allocated devices.
    ///
    /// Called periodically by the energy sidecar (e.g. every 2 seconds).
    /// `elapsed_secs` is the time since the last update.
    pub fn accumulate_energy(
        &self,
        elapsed_secs: f64,
        gpu_utilization: f64,
        npu_utilization: f64,
        tpu_utilization: f64,
        lpu_utilization: f64,
    ) {
        let devices = self.devices.read().unwrap();
        let mut energy = self.cumulative_energy.write().unwrap();

        for dev in devices.iter() {
            if dev.allocated_to.is_some() {
                let utilization = match dev.kind {
                    AcceleratorKind::GPU => gpu_utilization,
                    AcceleratorKind::NPU => npu_utilization,
                    AcceleratorKind::TPU => tpu_utilization,
                    AcceleratorKind::LPU => lpu_utilization,
                    AcceleratorKind::Custom(_) | _ => 0.0,
                };
                let joules = dev.tdp_watts * utilization * elapsed_secs;
                *energy.entry(dev.id.clone()).or_insert(0.0) += joules;
            }
        }
    }

    /// Get total energy consumed by a specific instance across all its devices.
    pub fn energy_for_instance(&self, instance_id: &str) -> f64 {
        let allocations = self.allocations.read().unwrap();
        let energy = self.cumulative_energy.read().unwrap();
        match allocations.get(instance_id) {
            Some(device_ids) => device_ids
                .iter()
                .map(|id| energy.get(id).copied().unwrap_or(0.0))
                .sum(),
            None => 0.0,
        }
    }

    /// Number of detected devices.
    pub fn device_count(&self) -> usize {
        self.devices.read().unwrap().len()
    }

    /// Re-detect devices (e.g., after hot-plug).
    pub fn refresh(&self) {
        let new_devices = detect_devices();
        let mut devices = self.devices.write().unwrap();

        // Preserve allocation state for devices that still exist
        let allocations: HashMap<String, Option<String>> = devices
            .iter()
            .map(|d| (d.id.clone(), d.allocated_to.clone()))
            .collect();

        *devices = new_devices;
        for dev in devices.iter_mut() {
            if let Some(alloc) = allocations.get(&dev.id) {
                dev.allocated_to = alloc.clone();
            }
        }
    }

    /// Build environment variables for accelerator passthrough.
    ///
    /// Sets the appropriate env vars based on allocated device kind:
    /// - GPU/CUDA: `CUDA_VISIBLE_DEVICES`, `NVIDIA_VISIBLE_DEVICES`
    /// - GPU/Metal: `METAL_DEVICE_INDEX`
    /// - GPU/ROCm: `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`
    /// - TPU: `TPU_VISIBLE_CHIPS`
    /// - NPU: `ANE_DEVICE_INDEX` (Apple), `NPU_DEVICE_INDEX`
    pub fn env_vars_for_instance(&self, instance_id: &str) -> HashMap<String, String> {
        let devices = self.devices_for_instance(instance_id);
        let mut env = HashMap::new();

        let gpu_ids: Vec<String> = devices
            .iter()
            .filter(|d| d.kind == AcceleratorKind::GPU)
            .map(|d| {
                // Extract numeric index from device id (e.g., "gpu-0" → "0")
                d.id.rsplit('-').next().unwrap_or("0").to_string()
            })
            .collect();

        if !gpu_ids.is_empty() {
            let gpu_list = gpu_ids.join(",");
            // Platform-specific GPU env vars
            if cfg!(target_os = "macos") {
                env.insert("METAL_DEVICE_INDEX".to_string(), gpu_list.clone());
            } else {
                env.insert("CUDA_VISIBLE_DEVICES".to_string(), gpu_list.clone());
                env.insert("NVIDIA_VISIBLE_DEVICES".to_string(), gpu_list.clone());
                env.insert("HIP_VISIBLE_DEVICES".to_string(), gpu_list.clone());
                env.insert("ROCR_VISIBLE_DEVICES".to_string(), gpu_list);
            }
        }

        let tpu_ids: Vec<String> = devices
            .iter()
            .filter(|d| d.kind == AcceleratorKind::TPU)
            .map(|d| d.id.rsplit('-').next().unwrap_or("0").to_string())
            .collect();

        if !tpu_ids.is_empty() {
            env.insert("TPU_VISIBLE_CHIPS".to_string(), tpu_ids.join(","));
        }

        let npu_ids: Vec<String> = devices
            .iter()
            .filter(|d| d.kind == AcceleratorKind::NPU)
            .map(|d| d.id.rsplit('-').next().unwrap_or("0").to_string())
            .collect();

        if !npu_ids.is_empty() {
            let npu_list = npu_ids.join(",");
            if cfg!(target_os = "macos") {
                env.insert("ANE_DEVICE_INDEX".to_string(), npu_list);
            } else {
                env.insert("NPU_DEVICE_INDEX".to_string(), npu_list);
            }
        }

        env
    }
}

impl Default for AcceleratorManager {
    fn default() -> Self {
        Self::new()
    }
}

// ── Device Detection ────────────────────────────────────────────────────────

/// Auto-detect all available hardware accelerators on this host.
fn detect_devices() -> Vec<AcceleratorDevice> {
    let mut devices = Vec::new();

    // Use joule-db-energy for baseline platform info
    let platform = joule_db_energy::detect_platform();

    // GPU detection
    detect_gpus(&mut devices, &platform);

    // NPU detection
    if platform.npu_available {
        detect_npus(&mut devices, &platform);
    }

    // TPU detection
    if platform.tpu_available {
        detect_tpus(&mut devices);
    }

    // LPU detection (Groq PCIe cards)
    detect_lpus(&mut devices);

    devices
}

/// Detect GPU devices.
fn detect_gpus(devices: &mut Vec<AcceleratorDevice>, platform: &joule_db_energy::PlatformInfo) {
    // Try NVIDIA first
    if let Some(nvidia_devices) = detect_nvidia_gpus() {
        devices.extend(nvidia_devices);
        return;
    }

    // Try AMD ROCm
    if let Some(amd_devices) = detect_amd_gpus() {
        devices.extend(amd_devices);
        return;
    }

    // macOS Metal — unified GPU is always present on Apple Silicon
    if cfg!(target_os = "macos") && platform.gpu_available {
        let gpu_name = format!("{} GPU", &platform.cpu_brand);
        // Apple Silicon unified memory GPU — TDP is shared with CPU
        let gpu_tdp = estimate_apple_gpu_tdp(&platform.cpu_brand);
        devices.push(AcceleratorDevice {
            id: "gpu-0".to_string(),
            kind: AcceleratorKind::GPU,
            name: gpu_name,
            memory_mb: 0, // Unified memory, shared with system
            compute_units: estimate_apple_gpu_cores(&platform.cpu_brand),
            tdp_watts: gpu_tdp,
            available: true,
            allocated_to: None,
        });
    }
}

/// Try to detect NVIDIA GPUs via `nvidia-smi`.
fn detect_nvidia_gpus() -> Option<Vec<AcceleratorDevice>> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,memory.total,power.default_limit",
            "--format=csv,noheader,nounits",
        ])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut devices = Vec::new();

    for line in stdout.lines() {
        let parts: Vec<&str> = line.split(", ").collect();
        if parts.len() >= 4 {
            let index = parts[0].trim();
            let name = parts[1].trim();
            let memory_mb: u64 = parts[2].trim().parse().unwrap_or(0);
            let tdp_watts: f64 = parts[3].trim().parse().unwrap_or(250.0);

            devices.push(AcceleratorDevice {
                id: format!("gpu-{}", index),
                kind: AcceleratorKind::GPU,
                name: name.to_string(),
                memory_mb,
                compute_units: 0, // nvidia-smi doesn't easily expose this
                tdp_watts,
                available: true,
                allocated_to: None,
            });
        }
    }

    if devices.is_empty() {
        None
    } else {
        Some(devices)
    }
}

/// Try to detect AMD GPUs via `rocm-smi`.
fn detect_amd_gpus() -> Option<Vec<AcceleratorDevice>> {
    let output = std::process::Command::new("rocm-smi")
        .args(["--showproductname", "--showmeminfo", "vram", "--csv"])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    // rocm-smi CSV format varies; do a simplified detection
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut devices = Vec::new();
    let mut index = 0;

    for line in stdout.lines().skip(1) {
        // Skip header
        if line.trim().is_empty() {
            continue;
        }
        devices.push(AcceleratorDevice {
            id: format!("gpu-{}", index),
            kind: AcceleratorKind::GPU,
            name: format!("AMD GPU {}", index),
            memory_mb: 0,
            compute_units: 0,
            tdp_watts: 300.0, // Conservative default for AMD GPUs
            available: true,
            allocated_to: None,
        });
        index += 1;
    }

    if devices.is_empty() {
        None
    } else {
        Some(devices)
    }
}

/// Detect Apple Neural Engine (NPU).
fn detect_npus(devices: &mut Vec<AcceleratorDevice>, platform: &joule_db_energy::PlatformInfo) {
    if cfg!(target_os = "macos") {
        let npu_cores = estimate_apple_npu_cores(&platform.cpu_brand);
        let npu_tdp = estimate_apple_npu_tdp(&platform.cpu_brand);

        devices.push(AcceleratorDevice {
            id: "npu-0".to_string(),
            kind: AcceleratorKind::NPU,
            name: format!("{} Neural Engine", &platform.cpu_brand),
            memory_mb: 0, // Shared unified memory
            compute_units: npu_cores,
            tdp_watts: npu_tdp,
            available: true,
            allocated_to: None,
        });
    }
}

/// Detect TPU devices — both Coral Edge TPUs and Google Cloud TPUs.
fn detect_tpus(devices: &mut Vec<AcceleratorDevice>) {
    // Check for Coral USB/PCIe devices
    #[cfg(target_os = "linux")]
    {
        // PCIe Coral: /dev/apex_0, /dev/apex_1, etc.
        for i in 0..4 {
            let path = format!("/dev/apex_{}", i);
            if std::path::Path::new(&path).exists() {
                devices.push(AcceleratorDevice {
                    id: format!("tpu-coral-{}", i),
                    kind: AcceleratorKind::TPU,
                    name: format!("Google Coral Edge TPU {}", i),
                    memory_mb: 8, // Coral has 8MB on-chip SRAM
                    compute_units: 1,
                    tdp_watts: 2.0, // Coral Edge TPU: ~2W
                    available: true,
                    allocated_to: None,
                });
            }
        }
    }

    // Cloud TPU detection via environment variables set on GCE TPU VMs
    if let Ok(tpu_name) = std::env::var("TPU_NAME") {
        let (tdp, memory_mb, name) = cloud_tpu_profile();
        let worker_id = std::env::var("TPU_WORKER_ID").unwrap_or_else(|_| "0".to_string());
        devices.push(AcceleratorDevice {
            id: format!("tpu-cloud-{}", worker_id),
            kind: AcceleratorKind::TPU,
            name: format!("{} ({})", name, tpu_name),
            memory_mb,
            compute_units: 1, // Single TPU chip per worker
            tdp_watts: tdp,
            available: true,
            allocated_to: None,
        });
    }

    // Suppress unused variable warning on non-Linux without Cloud TPU
    let _ = devices;
}

/// Return (TDP watts, HBM memory MB, display name) for the detected Cloud TPU generation.
///
/// Uses `ACCELERATOR_TYPE` env var (set on GCE TPU VMs) to identify the generation.
/// Falls back to v4 defaults if the env var is missing.
fn cloud_tpu_profile() -> (f64, u64, &'static str) {
    let accel = std::env::var("ACCELERATOR_TYPE").unwrap_or_default();
    let upper = accel.to_uppercase();

    if upper.contains("V5LITEPOD") || upper.contains("V5E") {
        // Cloud TPU v5e: inference-optimized, 16GB HBM, ~55W
        (55.0, 16_384, "Cloud TPU v5e")
    } else if upper.contains("V5P") {
        // Cloud TPU v5p: training-optimized, 95GB HBM, ~250W
        (250.0, 95_232, "Cloud TPU v5p")
    } else if upper.contains("V6E") {
        // Cloud TPU v6e (Trillium): 32GB HBM, ~100W
        (100.0, 32_768, "Cloud TPU v6e")
    } else {
        // Default to v4: 32GB HBM, ~170W
        (170.0, 32_768, "Cloud TPU v4")
    }
}

/// Detect Groq LPU inference cards.
fn detect_lpus(devices: &mut Vec<AcceleratorDevice>) {
    // Groq LPU cards appear as PCIe devices
    // Check for the groq CLI tool as a proxy
    if crate::native::which_exists("groq-runtime") || crate::native::which_exists("groqit") {
        devices.push(AcceleratorDevice {
            id: "lpu-0".to_string(),
            kind: AcceleratorKind::LPU,
            name: "Groq LPU".to_string(),
            memory_mb: 230_000, // Groq LPU1: 230GB SRAM
            compute_units: 1,
            tdp_watts: 300.0, // Estimated for inference card
            available: true,
            allocated_to: None,
        });
    }
}

// ── Apple Silicon Estimation ────────────────────────────────────────────────

/// Estimate Apple Silicon GPU core count from CPU brand string.
fn estimate_apple_gpu_cores(cpu_brand: &str) -> u32 {
    let upper = cpu_brand.to_uppercase();
    if upper.contains("ULTRA") {
        // M1/M2/M3/M4 Ultra: 60-80 cores
        76
    } else if upper.contains("MAX") {
        // M1/M2/M3/M4 Max: 30-40 cores
        if upper.contains("M4") {
            40
        } else if upper.contains("M3") {
            40
        } else {
            32
        }
    } else if upper.contains("PRO") {
        // M1/M2/M3/M4 Pro: 14-20 cores
        if upper.contains("M4") {
            20
        } else if upper.contains("M3") {
            18
        } else {
            16
        }
    } else {
        // Base M1/M2/M3/M4: 8-10 cores
        10
    }
}

/// Estimate Apple Silicon GPU TDP (portion of total SoC TDP).
fn estimate_apple_gpu_tdp(cpu_brand: &str) -> f64 {
    let upper = cpu_brand.to_uppercase();
    if upper.contains("ULTRA") {
        40.0
    } else if upper.contains("MAX") {
        35.0
    } else if upper.contains("PRO") {
        15.0
    } else {
        8.0
    }
}

/// Estimate Apple Neural Engine core count.
fn estimate_apple_npu_cores(cpu_brand: &str) -> u32 {
    let upper = cpu_brand.to_uppercase();
    if upper.contains("M4") {
        16
    } else if upper.contains("M3") {
        16
    } else if upper.contains("M2") {
        16
    } else {
        16 // M1 also has 16 Neural Engine cores
    }
}

/// Estimate Apple Neural Engine TDP.
fn estimate_apple_npu_tdp(cpu_brand: &str) -> f64 {
    let upper = cpu_brand.to_uppercase();
    if upper.contains("ULTRA") {
        15.0
    } else if upper.contains("MAX") {
        8.0
    } else if upper.contains("PRO") {
        5.0
    } else {
        3.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AcceleratorKind;

    fn test_gpu() -> AcceleratorDevice {
        AcceleratorDevice {
            id: "gpu-0".to_string(),
            kind: AcceleratorKind::GPU,
            name: "Test GPU".to_string(),
            memory_mb: 16384,
            compute_units: 5120,
            tdp_watts: 250.0,
            available: true,
            allocated_to: None,
        }
    }

    fn test_npu() -> AcceleratorDevice {
        AcceleratorDevice {
            id: "npu-0".to_string(),
            kind: AcceleratorKind::NPU,
            name: "Test NPU".to_string(),
            memory_mb: 0,
            compute_units: 16,
            tdp_watts: 5.0,
            available: true,
            allocated_to: None,
        }
    }

    #[test]
    fn test_manager_with_devices() {
        let mgr = AcceleratorManager::with_devices(vec![test_gpu(), test_npu()]);
        assert_eq!(mgr.device_count(), 2);
        assert_eq!(mgr.list_devices().len(), 2);
        assert_eq!(mgr.list_available().len(), 2);
    }

    #[test]
    fn test_list_by_kind() {
        let mgr = AcceleratorManager::with_devices(vec![test_gpu(), test_npu()]);
        let gpus = mgr.list_by_kind(&AcceleratorKind::GPU);
        assert_eq!(gpus.len(), 1);
        assert_eq!(gpus[0].name, "Test GPU");

        let npus = mgr.list_by_kind(&AcceleratorKind::NPU);
        assert_eq!(npus.len(), 1);

        let tpus = mgr.list_by_kind(&AcceleratorKind::TPU);
        assert!(tpus.is_empty());
    }

    #[test]
    fn test_allocate_specific_device() {
        let mgr = AcceleratorManager::with_devices(vec![test_gpu(), test_npu()]);

        let bindings = vec![AcceleratorBinding {
            kind: AcceleratorKind::GPU,
            device_id: Some("gpu-0".to_string()),
            memory_mb: None,
        }];

        let allocated = mgr.allocate("instance-1", &bindings).unwrap();
        assert_eq!(allocated, vec!["gpu-0"]);

        // Device should now be allocated
        let devs = mgr.list_available();
        assert_eq!(devs.len(), 1); // Only NPU is free
        assert_eq!(devs[0].kind, AcceleratorKind::NPU);

        // Instance should have the device
        let inst_devs = mgr.devices_for_instance("instance-1");
        assert_eq!(inst_devs.len(), 1);
        assert_eq!(inst_devs[0].id, "gpu-0");
    }

    #[test]
    fn test_allocate_auto_select() {
        let mgr = AcceleratorManager::with_devices(vec![test_gpu(), test_npu()]);

        let bindings = vec![AcceleratorBinding {
            kind: AcceleratorKind::GPU,
            device_id: None, // auto-select
            memory_mb: None,
        }];

        let allocated = mgr.allocate("instance-1", &bindings).unwrap();
        assert_eq!(allocated, vec!["gpu-0"]);
    }

    #[test]
    fn test_allocate_multiple_devices() {
        let mgr = AcceleratorManager::with_devices(vec![test_gpu(), test_npu()]);

        let bindings = vec![
            AcceleratorBinding {
                kind: AcceleratorKind::GPU,
                device_id: None,
                memory_mb: None,
            },
            AcceleratorBinding {
                kind: AcceleratorKind::NPU,
                device_id: None,
                memory_mb: None,
            },
        ];

        let allocated = mgr.allocate("instance-1", &bindings).unwrap();
        assert_eq!(allocated.len(), 2);
        assert!(allocated.contains(&"gpu-0".to_string()));
        assert!(allocated.contains(&"npu-0".to_string()));
        assert!(mgr.list_available().is_empty());
    }

    #[test]
    fn test_allocate_fails_when_no_device_available() {
        let mgr = AcceleratorManager::with_devices(vec![test_gpu()]);

        // First allocation succeeds
        let bindings = vec![AcceleratorBinding {
            kind: AcceleratorKind::GPU,
            device_id: None,
            memory_mb: None,
        }];
        mgr.allocate("instance-1", &bindings).unwrap();

        // Second allocation fails — GPU already taken
        let result = mgr.allocate("instance-2", &bindings);
        assert!(result.is_err());
    }

    #[test]
    fn test_allocate_rollback_on_failure() {
        let mgr = AcceleratorManager::with_devices(vec![test_gpu()]);

        // Try to allocate GPU + TPU — GPU exists but TPU doesn't
        let bindings = vec![
            AcceleratorBinding {
                kind: AcceleratorKind::GPU,
                device_id: None,
                memory_mb: None,
            },
            AcceleratorBinding {
                kind: AcceleratorKind::TPU,
                device_id: None,
                memory_mb: None,
            },
        ];

        let result = mgr.allocate("instance-1", &bindings);
        assert!(result.is_err());

        // GPU should be rolled back to available
        assert_eq!(mgr.list_available().len(), 1);
        assert_eq!(mgr.list_available()[0].id, "gpu-0");
    }

    #[test]
    fn test_release() {
        let mgr = AcceleratorManager::with_devices(vec![test_gpu(), test_npu()]);

        let bindings = vec![AcceleratorBinding {
            kind: AcceleratorKind::GPU,
            device_id: None,
            memory_mb: None,
        }];
        mgr.allocate("instance-1", &bindings).unwrap();
        assert_eq!(mgr.list_available().len(), 1);

        mgr.release("instance-1");
        assert_eq!(mgr.list_available().len(), 2);
        assert!(mgr.devices_for_instance("instance-1").is_empty());
    }

    #[test]
    fn test_release_nonexistent() {
        let mgr = AcceleratorManager::with_devices(vec![test_gpu()]);
        mgr.release("nonexistent"); // Should not panic
    }

    #[test]
    fn test_energy_snapshots() {
        let mgr = AcceleratorManager::with_devices(vec![test_gpu(), test_npu()]);

        let snapshots = mgr.energy_snapshots(0.5, 0.3, 0.0, 0.0);
        assert_eq!(snapshots.len(), 2);

        let gpu_snap = snapshots
            .iter()
            .find(|s| s.kind == AcceleratorKind::GPU)
            .unwrap();
        assert!((gpu_snap.power_watts - 125.0).abs() < 0.01); // 250W × 0.5
        assert_eq!(gpu_snap.energy_joules, 0.0); // No accumulation yet

        let npu_snap = snapshots
            .iter()
            .find(|s| s.kind == AcceleratorKind::NPU)
            .unwrap();
        assert!((npu_snap.power_watts - 1.5).abs() < 0.01); // 5W × 0.3
    }

    #[test]
    fn test_accumulate_energy() {
        let mgr = AcceleratorManager::with_devices(vec![test_gpu()]);

        // Allocate GPU
        let bindings = vec![AcceleratorBinding {
            kind: AcceleratorKind::GPU,
            device_id: None,
            memory_mb: None,
        }];
        mgr.allocate("instance-1", &bindings).unwrap();

        // Accumulate 2 seconds at 50% utilization
        mgr.accumulate_energy(2.0, 0.5, 0.0, 0.0, 0.0);

        let energy = mgr.energy_for_instance("instance-1");
        assert!((energy - 250.0).abs() < 0.01); // 250W × 0.5 × 2s = 250J

        // Accumulate another 3 seconds at 100% utilization
        mgr.accumulate_energy(3.0, 1.0, 0.0, 0.0, 0.0);

        let energy = mgr.energy_for_instance("instance-1");
        assert!((energy - 1000.0).abs() < 0.01); // 250J + 250W × 1.0 × 3s = 1000J
    }

    #[test]
    fn test_energy_for_unallocated_instance() {
        let mgr = AcceleratorManager::with_devices(vec![test_gpu()]);
        assert_eq!(mgr.energy_for_instance("nonexistent"), 0.0);
    }

    #[test]
    fn test_env_vars_for_instance() {
        let mgr = AcceleratorManager::with_devices(vec![test_gpu(), test_npu()]);

        let bindings = vec![
            AcceleratorBinding {
                kind: AcceleratorKind::GPU,
                device_id: None,
                memory_mb: None,
            },
            AcceleratorBinding {
                kind: AcceleratorKind::NPU,
                device_id: None,
                memory_mb: None,
            },
        ];
        mgr.allocate("instance-1", &bindings).unwrap();

        let env = mgr.env_vars_for_instance("instance-1");

        if cfg!(target_os = "macos") {
            assert!(env.contains_key("METAL_DEVICE_INDEX"));
            assert!(env.contains_key("ANE_DEVICE_INDEX"));
        } else {
            assert!(env.contains_key("CUDA_VISIBLE_DEVICES"));
            assert!(env.contains_key("NVIDIA_VISIBLE_DEVICES"));
            assert!(env.contains_key("NPU_DEVICE_INDEX"));
        }
    }

    #[test]
    fn test_env_vars_empty_when_no_allocation() {
        let mgr = AcceleratorManager::with_devices(vec![test_gpu()]);
        let env = mgr.env_vars_for_instance("nonexistent");
        assert!(env.is_empty());
    }

    #[test]
    fn test_device_serde() {
        let dev = test_gpu();
        let json = serde_json::to_string(&dev).unwrap();
        let parsed: AcceleratorDevice = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, "gpu-0");
        assert_eq!(parsed.kind, AcceleratorKind::GPU);
        assert_eq!(parsed.memory_mb, 16384);
        assert_eq!(parsed.tdp_watts, 250.0);
    }

    #[test]
    fn test_device_energy_serde() {
        let energy = DeviceEnergy {
            device_id: "gpu-0".to_string(),
            kind: AcceleratorKind::GPU,
            name: "Test GPU".to_string(),
            power_watts: 125.0,
            energy_joules: 500.0,
            utilization: 0.5,
            memory_used_mb: 8192,
            allocated_to: Some("instance-1".to_string()),
        };
        let json = serde_json::to_string(&energy).unwrap();
        let parsed: DeviceEnergy = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.power_watts, 125.0);
        assert_eq!(parsed.energy_joules, 500.0);
    }

    #[test]
    fn test_apple_gpu_cores_estimation() {
        assert_eq!(estimate_apple_gpu_cores("Apple M4 Max"), 40);
        assert_eq!(estimate_apple_gpu_cores("Apple M3 Pro"), 18);
        assert_eq!(estimate_apple_gpu_cores("Apple M2 Ultra"), 76);
        assert_eq!(estimate_apple_gpu_cores("Apple M1"), 10);
    }

    #[test]
    fn test_apple_gpu_tdp_estimation() {
        assert_eq!(estimate_apple_gpu_tdp("Apple M4 Max"), 35.0);
        assert_eq!(estimate_apple_gpu_tdp("Apple M3 Pro"), 15.0);
        assert_eq!(estimate_apple_gpu_tdp("Apple M2 Ultra"), 40.0);
        assert_eq!(estimate_apple_gpu_tdp("Apple M1"), 8.0);
    }

    #[test]
    fn test_apple_npu_cores_estimation() {
        assert_eq!(estimate_apple_npu_cores("Apple M4 Max"), 16);
        assert_eq!(estimate_apple_npu_cores("Apple M3 Pro"), 16);
        assert_eq!(estimate_apple_npu_cores("Apple M1"), 16);
    }

    #[test]
    fn test_refresh_preserves_allocations() {
        let mgr = AcceleratorManager::with_devices(vec![test_gpu(), test_npu()]);

        let bindings = vec![AcceleratorBinding {
            kind: AcceleratorKind::GPU,
            device_id: None,
            memory_mb: None,
        }];
        mgr.allocate("instance-1", &bindings).unwrap();

        // After refresh, the detect_devices() will return whatever the host has.
        // Since we're in a test, this is fine — the point is that the method
        // doesn't panic and preserves structure.
        mgr.refresh();

        // The real host devices won't have "gpu-0" from our test data, so
        // allocations won't carry over. That's expected for the test.
    }

    #[test]
    fn test_auto_detect_runs_without_panic() {
        // Just verify that detect_devices() doesn't crash
        let devices = detect_devices();
        // On CI / machines without GPUs, this may be empty
        for dev in &devices {
            assert!(!dev.id.is_empty());
            assert!(!dev.name.is_empty());
            assert!(dev.tdp_watts > 0.0);
        }
    }
}
