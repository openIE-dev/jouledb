//! Unified Resource Limits — cross-backend resource constraint API.
//!
//! Provides a single `ResourceLimits` struct that works across all four
//! runtime backends (Native, Container, VM, WASM). Each backend translates
//! the unified limits into its own enforcement mechanism:
//!
//! - **Native**: ulimit / cgroups v2 (Linux), launchctl (macOS)
//! - **Container**: OCI runtime spec (--cpus, --memory, --pids-limit)
//! - **VM**: hypervisor config (vCPUs, memory, disk)
//! - **WASM**: fuel limits, memory caps, timeout
//!
//! Also includes energy limits (microjoules budget) — unique to JoulesPerBit.

use crate::{RuntimeConfig, RuntimeError, RuntimeMode};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Unified resource limits for any sandbox backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// CPU limits.
    #[serde(default)]
    pub cpu: CpuLimits,
    /// Memory limits.
    #[serde(default)]
    pub memory: MemoryLimits,
    /// Disk I/O limits.
    #[serde(default)]
    pub disk: DiskLimits,
    /// Network bandwidth limits.
    #[serde(default)]
    pub network: NetworkLimits,
    /// Process/thread limits.
    #[serde(default)]
    pub process: ProcessLimits,
    /// Energy limits (unique to JoulesPerBit).
    #[serde(default)]
    pub energy: EnergyLimits,
    /// Wall-clock timeout for the entire workload.
    #[serde(default)]
    pub timeout: Option<TimeoutConfig>,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            cpu: CpuLimits::default(),
            memory: MemoryLimits::default(),
            disk: DiskLimits::default(),
            network: NetworkLimits::default(),
            process: ProcessLimits::default(),
            energy: EnergyLimits::default(),
            timeout: None,
        }
    }
}

/// CPU resource limits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuLimits {
    /// Maximum number of CPU cores (fractional allowed, e.g. 1.5).
    pub max_cores: Option<f64>,
    /// CPU shares (relative weight, default 1024).
    pub shares: Option<u32>,
    /// CPU period in microseconds (cgroups, default 100000).
    pub period_us: Option<u64>,
    /// CPU quota in microseconds (cgroups, per period).
    pub quota_us: Option<u64>,
    /// Pin to specific CPU cores (e.g., [0, 1, 2]).
    pub affinity: Option<Vec<u32>>,
}

impl Default for CpuLimits {
    fn default() -> Self {
        Self {
            max_cores: None,
            shares: None,
            period_us: None,
            quota_us: None,
            affinity: None,
        }
    }
}

/// Memory resource limits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLimits {
    /// Hard memory limit in bytes (OOM kill above this).
    pub hard_limit_bytes: Option<u64>,
    /// Soft memory limit in bytes (reclaim above this).
    pub soft_limit_bytes: Option<u64>,
    /// Swap limit in bytes (0 = no swap).
    pub swap_limit_bytes: Option<u64>,
    /// Kernel memory limit in bytes.
    pub kernel_limit_bytes: Option<u64>,
    /// Whether to disable OOM killer (container keeps running at limit).
    pub disable_oom_kill: bool,
}

impl Default for MemoryLimits {
    fn default() -> Self {
        Self {
            hard_limit_bytes: None,
            soft_limit_bytes: None,
            swap_limit_bytes: Some(0), // No swap by default
            kernel_limit_bytes: None,
            disable_oom_kill: false,
        }
    }
}

/// Disk I/O limits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskLimits {
    /// Maximum disk space in bytes.
    pub max_bytes: Option<u64>,
    /// Read bandwidth limit in bytes/sec.
    pub read_bps: Option<u64>,
    /// Write bandwidth limit in bytes/sec.
    pub write_bps: Option<u64>,
    /// Read IOPS limit.
    pub read_iops: Option<u64>,
    /// Write IOPS limit.
    pub write_iops: Option<u64>,
    /// Whether storage is tmpfs-only (no persistent writes).
    pub tmpfs_only: bool,
}

impl Default for DiskLimits {
    fn default() -> Self {
        Self {
            max_bytes: None,
            read_bps: None,
            write_bps: None,
            read_iops: None,
            write_iops: None,
            tmpfs_only: false,
        }
    }
}

/// Network bandwidth limits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkLimits {
    /// Egress bandwidth limit in bytes/sec.
    pub egress_bps: Option<u64>,
    /// Ingress bandwidth limit in bytes/sec.
    pub ingress_bps: Option<u64>,
    /// Maximum concurrent connections.
    pub max_connections: Option<u32>,
}

impl Default for NetworkLimits {
    fn default() -> Self {
        Self {
            egress_bps: None,
            ingress_bps: None,
            max_connections: None,
        }
    }
}

/// Process/thread limits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessLimits {
    /// Maximum number of processes (--pids-limit in containers).
    pub max_pids: Option<u32>,
    /// Maximum number of threads per process.
    pub max_threads: Option<u32>,
    /// Maximum open file descriptors.
    pub max_open_files: Option<u64>,
}

impl Default for ProcessLimits {
    fn default() -> Self {
        Self {
            max_pids: None,
            max_threads: None,
            max_open_files: None,
        }
    }
}

/// Energy limits — unique to JoulesPerBit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyLimits {
    /// Maximum energy budget in microjoules (hard kill if exceeded).
    pub max_energy_uj: Option<u64>,
    /// Warning threshold (0.0-1.0).
    pub warning_threshold: f64,
    /// Sampling interval for energy counters in milliseconds.
    pub sample_interval_ms: u64,
    /// Whether to hard-kill on budget exceeded.
    pub hard_kill: bool,
}

impl Default for EnergyLimits {
    fn default() -> Self {
        Self {
            max_energy_uj: None,
            warning_threshold: 0.8,
            sample_interval_ms: 100,
            hard_kill: true,
        }
    }
}

/// Timeout configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutConfig {
    /// Maximum wall-clock time for the workload.
    #[serde(with = "duration_secs")]
    pub max_duration: Duration,
    /// Grace period before SIGKILL after SIGTERM.
    #[serde(with = "duration_secs")]
    pub grace_period: Duration,
}

impl Default for TimeoutConfig {
    fn default() -> Self {
        Self {
            max_duration: Duration::from_secs(3600), // 1 hour
            grace_period: Duration::from_secs(10),
        }
    }
}

impl ResourceLimits {
    /// Validate limits for consistency.
    pub fn validate(&self) -> Result<(), RuntimeError> {
        if let (Some(hard), Some(soft)) = (self.memory.hard_limit_bytes, self.memory.soft_limit_bytes) {
            if soft > hard {
                return Err(RuntimeError::ConfigError(
                    "memory soft limit cannot exceed hard limit".into(),
                ));
            }
        }
        if let Some(cores) = self.cpu.max_cores {
            if cores <= 0.0 {
                return Err(RuntimeError::ConfigError(
                    "CPU core limit must be positive".into(),
                ));
            }
        }
        if let Some(ref timeout) = self.timeout {
            if timeout.max_duration.is_zero() {
                return Err(RuntimeError::ConfigError(
                    "timeout duration must be non-zero".into(),
                ));
            }
        }
        Ok(())
    }

    /// Convert to container runtime arguments (docker/podman).
    pub fn to_container_args(&self) -> Vec<String> {
        let mut args = Vec::new();

        // CPU
        if let Some(cores) = self.cpu.max_cores {
            args.push(format!("--cpus={:.2}", cores));
        }
        if let Some(shares) = self.cpu.shares {
            args.push(format!("--cpu-shares={}", shares));
        }
        if let Some(ref affinity) = self.cpu.affinity {
            let cpuset: String = affinity
                .iter()
                .map(|c| c.to_string())
                .collect::<Vec<_>>()
                .join(",");
            args.push(format!("--cpuset-cpus={}", cpuset));
        }

        // Memory
        if let Some(hard) = self.memory.hard_limit_bytes {
            args.push(format!("--memory={}", hard));
        }
        if let Some(soft) = self.memory.soft_limit_bytes {
            args.push(format!("--memory-reservation={}", soft));
        }
        if let Some(swap) = self.memory.swap_limit_bytes {
            args.push(format!("--memory-swap={}", swap));
        }
        if self.memory.disable_oom_kill {
            args.push("--oom-kill-disable".to_string());
        }

        // Process limits
        if let Some(pids) = self.process.max_pids {
            args.push(format!("--pids-limit={}", pids));
        }
        if let Some(fds) = self.process.max_open_files {
            args.push(format!("--ulimit=nofile={}:{}", fds, fds));
        }

        // Disk I/O
        if let Some(rbps) = self.disk.read_bps {
            args.push(format!("--device-read-bps=/dev/sda:{}", rbps));
        }
        if let Some(wbps) = self.disk.write_bps {
            args.push(format!("--device-write-bps=/dev/sda:{}", wbps));
        }

        // tmpfs
        if self.disk.tmpfs_only {
            args.push("--read-only".to_string());
            args.push("--tmpfs=/tmp:rw,noexec,nosuid,size=256m".to_string());
        }

        args
    }

    /// Convert to RuntimeConfig overrides for the WASM backend.
    pub fn apply_to_runtime_config(&self, config: &mut RuntimeConfig) {
        if let Some(hard) = self.memory.hard_limit_bytes {
            config.wasm_max_memory_bytes = hard as usize;
        }

        // Map energy budget to WASM fuel (rough: 1 µJ ≈ 100 fuel units)
        if let Some(energy_uj) = self.energy.max_energy_uj {
            config.wasm_fuel = energy_uj.saturating_mul(100);
        }

        if let Some(ref timeout) = self.timeout {
            config.wasm_timeout_secs = timeout.max_duration.as_secs();
        }
    }

    /// Convert to VM configuration overrides.
    pub fn apply_to_vm_config(&self, config: &mut RuntimeConfig) {
        if let Some(cores) = self.cpu.max_cores {
            config.vm_cpu_cores = cores.ceil() as u32;
        }
        if let Some(hard) = self.memory.hard_limit_bytes {
            config.vm_memory_mb = (hard / (1024 * 1024)) as u32;
        }
    }

    /// Get the appropriate limits for a given runtime mode.
    pub fn for_mode(&self, mode: RuntimeMode) -> ResolvedLimits {
        match mode {
            RuntimeMode::Native => ResolvedLimits {
                container_args: vec![],
                env_vars: self.to_env_vars(),
                description: "Native: limits enforced via ulimit/cgroups".into(),
            },
            RuntimeMode::VM => ResolvedLimits {
                container_args: vec![],
                env_vars: self.to_env_vars(),
                description: "VM: limits enforced via hypervisor config".into(),
            },
            RuntimeMode::WASM => ResolvedLimits {
                container_args: vec![],
                env_vars: self.to_env_vars(),
                description: "WASM: limits enforced via fuel/memory caps".into(),
            },
        }
    }

    /// Convert energy limits to environment variables for the sandbox.
    fn to_env_vars(&self) -> std::collections::HashMap<String, String> {
        let mut vars = std::collections::HashMap::new();

        if let Some(uj) = self.energy.max_energy_uj {
            vars.insert("JOULE_MAX_ENERGY_UJ".to_string(), uj.to_string());
        }
        vars.insert(
            "JOULE_ENERGY_WARN_THRESHOLD".to_string(),
            self.energy.warning_threshold.to_string(),
        );
        vars.insert(
            "JOULE_ENERGY_SAMPLE_MS".to_string(),
            self.energy.sample_interval_ms.to_string(),
        );

        if let Some(ref timeout) = self.timeout {
            vars.insert(
                "JOULE_TIMEOUT_SECS".to_string(),
                timeout.max_duration.as_secs().to_string(),
            );
        }

        vars
    }
}

/// Resolved limits for a specific runtime mode.
#[derive(Debug, Clone)]
pub struct ResolvedLimits {
    /// Container runtime arguments (empty for non-container modes).
    pub container_args: Vec<String>,
    /// Environment variables to inject.
    pub env_vars: std::collections::HashMap<String, String>,
    /// Human-readable description of enforcement.
    pub description: String,
}

/// Serde helper for Duration as seconds.
mod duration_secs {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub fn serialize<S: Serializer>(duration: &Duration, ser: S) -> Result<S::Ok, S::Error> {
        duration.as_secs().serialize(ser)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(de: D) -> Result<Duration, D::Error> {
        let secs = u64::deserialize(de)?;
        Ok(Duration::from_secs(secs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_limits_default() {
        let limits = ResourceLimits::default();
        assert!(limits.cpu.max_cores.is_none());
        assert!(limits.memory.hard_limit_bytes.is_none());
        assert_eq!(limits.memory.swap_limit_bytes, Some(0));
        assert!(limits.energy.max_energy_uj.is_none());
        assert!(limits.timeout.is_none());
    }

    #[test]
    fn test_resource_limits_validate() {
        let mut limits = ResourceLimits::default();
        assert!(limits.validate().is_ok());

        // Soft > hard should fail
        limits.memory.hard_limit_bytes = Some(1_000_000);
        limits.memory.soft_limit_bytes = Some(2_000_000);
        assert!(limits.validate().is_err());

        // Fix it
        limits.memory.soft_limit_bytes = Some(500_000);
        assert!(limits.validate().is_ok());
    }

    #[test]
    fn test_resource_limits_validate_cpu() {
        let mut limits = ResourceLimits::default();
        limits.cpu.max_cores = Some(0.0);
        assert!(limits.validate().is_err());

        limits.cpu.max_cores = Some(-1.0);
        assert!(limits.validate().is_err());

        limits.cpu.max_cores = Some(2.5);
        assert!(limits.validate().is_ok());
    }

    #[test]
    fn test_container_args_cpu() {
        let mut limits = ResourceLimits::default();
        limits.cpu.max_cores = Some(2.0);
        limits.cpu.shares = Some(512);
        limits.cpu.affinity = Some(vec![0, 1]);

        let args = limits.to_container_args();
        assert!(args.contains(&"--cpus=2.00".to_string()));
        assert!(args.contains(&"--cpu-shares=512".to_string()));
        assert!(args.contains(&"--cpuset-cpus=0,1".to_string()));
    }

    #[test]
    fn test_container_args_memory() {
        let mut limits = ResourceLimits::default();
        limits.memory.hard_limit_bytes = Some(4_294_967_296); // 4 GB
        limits.memory.soft_limit_bytes = Some(2_147_483_648); // 2 GB
        limits.memory.disable_oom_kill = true;

        let args = limits.to_container_args();
        assert!(args.contains(&"--memory=4294967296".to_string()));
        assert!(args.contains(&"--memory-reservation=2147483648".to_string()));
        assert!(args.contains(&"--oom-kill-disable".to_string()));
    }

    #[test]
    fn test_container_args_process() {
        let mut limits = ResourceLimits::default();
        limits.process.max_pids = Some(100);
        limits.process.max_open_files = Some(1024);

        let args = limits.to_container_args();
        assert!(args.contains(&"--pids-limit=100".to_string()));
        assert!(args.contains(&"--ulimit=nofile=1024:1024".to_string()));
    }

    #[test]
    fn test_container_args_tmpfs() {
        let mut limits = ResourceLimits::default();
        limits.disk.tmpfs_only = true;

        let args = limits.to_container_args();
        assert!(args.contains(&"--read-only".to_string()));
        assert!(args.iter().any(|a| a.starts_with("--tmpfs=")));
    }

    #[test]
    fn test_apply_to_wasm_config() {
        let mut limits = ResourceLimits::default();
        limits.memory.hard_limit_bytes = Some(128 * 1024 * 1024); // 128 MB
        limits.energy.max_energy_uj = Some(10_000_000); // 10 J
        limits.timeout = Some(TimeoutConfig {
            max_duration: Duration::from_secs(60),
            grace_period: Duration::from_secs(5),
        });

        let mut config = RuntimeConfig::wasm();
        limits.apply_to_runtime_config(&mut config);

        assert_eq!(config.wasm_max_memory_bytes, 128 * 1024 * 1024);
        assert_eq!(config.wasm_fuel, 1_000_000_000); // 10M * 100
        assert_eq!(config.wasm_timeout_secs, 60);
    }

    #[test]
    fn test_apply_to_vm_config() {
        let mut limits = ResourceLimits::default();
        limits.cpu.max_cores = Some(8.0);
        limits.memory.hard_limit_bytes = Some(16 * 1024 * 1024 * 1024); // 16 GB

        let mut config = RuntimeConfig::vm();
        limits.apply_to_vm_config(&mut config);

        assert_eq!(config.vm_cpu_cores, 8);
        assert_eq!(config.vm_memory_mb, 16384);
    }

    #[test]
    fn test_energy_limits_env_vars() {
        let mut limits = ResourceLimits::default();
        limits.energy.max_energy_uj = Some(50_000_000);
        limits.energy.warning_threshold = 0.9;
        limits.energy.sample_interval_ms = 50;

        let vars = limits.to_env_vars();
        assert_eq!(vars.get("JOULE_MAX_ENERGY_UJ").unwrap(), "50000000");
        assert_eq!(vars.get("JOULE_ENERGY_WARN_THRESHOLD").unwrap(), "0.9");
        assert_eq!(vars.get("JOULE_ENERGY_SAMPLE_MS").unwrap(), "50");
    }

    #[test]
    fn test_for_mode() {
        let mut limits = ResourceLimits::default();
        limits.energy.max_energy_uj = Some(1_000_000);

        let resolved = limits.for_mode(RuntimeMode::WASM);
        assert!(resolved.env_vars.contains_key("JOULE_MAX_ENERGY_UJ"));
    }

    #[test]
    fn test_resource_limits_serde() {
        let mut limits = ResourceLimits::default();
        limits.cpu.max_cores = Some(4.0);
        limits.memory.hard_limit_bytes = Some(8_000_000_000);
        limits.energy.max_energy_uj = Some(100_000_000);
        limits.timeout = Some(TimeoutConfig::default());

        let json = serde_json::to_string(&limits).unwrap();
        let parsed: ResourceLimits = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.cpu.max_cores, Some(4.0));
        assert_eq!(parsed.memory.hard_limit_bytes, Some(8_000_000_000));
        assert_eq!(parsed.energy.max_energy_uj, Some(100_000_000));
    }

    #[test]
    fn test_timeout_config_serde() {
        let tc = TimeoutConfig {
            max_duration: Duration::from_secs(300),
            grace_period: Duration::from_secs(15),
        };
        let json = serde_json::to_string(&tc).unwrap();
        let parsed: TimeoutConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.max_duration, Duration::from_secs(300));
        assert_eq!(parsed.grace_period, Duration::from_secs(15));
    }
}
