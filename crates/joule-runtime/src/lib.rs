//! Joule Runtime — Energy-Aware Container Daemon
//!
//! Runs databases, processes, and OCI containers with hardware energy
//! telemetry — power_watts, energy_joules, thermal_state, GPU/NPU utilization.
//!
//! Four backend types:
//! - **Native**: Bare-metal process execution (zero overhead)
//! - **Container**: OCI container lifecycle via docker/podman/nerdctl
//! - **VM**: Full hardware isolation via Apple Virtualization.framework / KVM
//! - **WASM**: Sandboxed execution via Wasmtime
//!
//! VM and WASM backends are feature-gated behind `vm-isolation` and `wasm-isolation`.
//! The energy sidecar is feature-gated behind `energy-sidecar` (default on).

pub mod accelerator;
pub mod agent_lifecycle;
pub mod agent_sandbox;
pub mod attestation;
pub mod backend;
pub mod competence;
pub mod contract;
pub mod catalog;
pub mod cluster;
pub mod container;
pub mod dispatch;
pub mod daemon;
pub mod daemon_client;
pub mod dashboard;
pub mod db_tuner;
pub mod energy_enforcer;
pub mod energy_trace;
pub mod governor;
pub mod graceful_shutdown;
pub mod health_monitor;
pub mod image;
pub mod llm;
pub mod manager;
pub mod native;
pub mod network_isolation;
pub mod networking;
pub mod platform_service;
pub mod promotion;
pub mod registry;
pub mod resource_limits;
pub mod sandbox;
pub mod substrate;

#[cfg(feature = "energy-sidecar")]
pub mod energy_wrapper;

#[cfg(feature = "vm-isolation")]
pub mod vm_backend;

#[cfg(feature = "wasm-isolation")]
pub mod wasm_backend;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;

// Re-exports
pub use backend::RuntimeBackend;
pub use manager::RuntimeManager;
pub use registry::InstanceRegistry;

/// Runtime isolation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RuntimeMode {
    /// Bare-metal process execution — current behavior, zero overhead.
    Native,
    /// Full hardware isolation via hypervisor (Virtualization.framework / KVM).
    VM,
    /// Sandboxed execution via Wasmtime WASI runtime.
    WASM,
}

impl Default for RuntimeMode {
    fn default() -> Self {
        Self::Native
    }
}

impl fmt::Display for RuntimeMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Native => write!(f, "native"),
            Self::VM => write!(f, "vm"),
            Self::WASM => write!(f, "wasm"),
        }
    }
}

impl FromStr for RuntimeMode {
    type Err = RuntimeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "native" => Ok(Self::Native),
            "vm" => Ok(Self::VM),
            "wasm" => Ok(Self::WASM),
            _ => Err(RuntimeError::ConfigError(format!(
                "unknown runtime mode '{}', expected: native, vm, wasm",
                s
            ))),
        }
    }
}

/// Which database engine to run.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DatabaseEngine {
    /// JouleDB — the energy-aware database (native experience).
    #[serde(alias = "joule-db", alias = "joule")]
    JouleDB,
    /// PostgreSQL.
    #[serde(alias = "pg", alias = "postgresql")]
    Postgres,
    /// MySQL / MariaDB.
    #[serde(alias = "mariadb")]
    MySQL,
    /// Redis.
    Redis,
    /// MongoDB.
    #[serde(alias = "mongo")]
    MongoDB,
    /// SQLite (embedded, no server process).
    #[serde(alias = "sqlite3")]
    SQLite,
    /// Custom database — user supplies the binary name.
    Custom(String),
}

impl Default for DatabaseEngine {
    fn default() -> Self {
        Self::JouleDB
    }
}

impl fmt::Display for DatabaseEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::JouleDB => write!(f, "jouledb"),
            Self::Postgres => write!(f, "postgres"),
            Self::MySQL => write!(f, "mysql"),
            Self::Redis => write!(f, "redis"),
            Self::MongoDB => write!(f, "mongodb"),
            Self::SQLite => write!(f, "sqlite"),
            Self::Custom(name) => write!(f, "{}", name),
        }
    }
}

impl FromStr for DatabaseEngine {
    type Err = RuntimeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "jouledb" | "joule-db" | "joule" => Ok(Self::JouleDB),
            "postgres" | "postgresql" | "pg" | "pgsql" => Ok(Self::Postgres),
            "mysql" | "mariadb" => Ok(Self::MySQL),
            "redis" => Ok(Self::Redis),
            "mongodb" | "mongo" | "mongod" => Ok(Self::MongoDB),
            "sqlite" | "sqlite3" => Ok(Self::SQLite),
            other => Ok(Self::Custom(other.to_string())),
        }
    }
}

impl DatabaseEngine {
    /// Returns the default binary name for this engine.
    pub fn binary_name(&self) -> &str {
        match self {
            Self::JouleDB => "joule-db-server",
            Self::Postgres => "postgres",
            Self::MySQL => "mysqld",
            Self::Redis => "redis-server",
            Self::MongoDB => "mongod",
            Self::SQLite => "sqlite3",
            Self::Custom(name) => name,
        }
    }

    /// Returns the standard default port.
    pub fn default_port(&self) -> u16 {
        match self {
            Self::JouleDB => 8080,
            Self::Postgres => 5432,
            Self::MySQL => 3306,
            Self::Redis => 6379,
            Self::MongoDB => 27017,
            Self::SQLite => 0,
            Self::Custom(_) => 0,
        }
    }

    /// Human-readable display name.
    pub fn display_name(&self) -> &str {
        match self {
            Self::JouleDB => "JouleDB",
            Self::Postgres => "PostgreSQL",
            Self::MySQL => "MySQL",
            Self::Redis => "Redis",
            Self::MongoDB => "MongoDB",
            Self::SQLite => "SQLite",
            Self::Custom(name) => name,
        }
    }

    /// Whether this engine has a server process to launch.
    pub fn has_server_process(&self) -> bool {
        !matches!(self, Self::SQLite)
    }
}

/// What kind of workload to run — databases, arbitrary processes, or OCI containers.
///
/// This is the generalized successor to `DatabaseEngine`. For backward compatibility,
/// `DatabaseEngine` is preserved as the inner type for database workloads.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum WorkloadKind {
    /// A known database engine (backward-compatible with `DatabaseEngine`).
    Database {
        #[serde(default)]
        engine: DatabaseEngine,
    },
    /// An arbitrary process — user supplies the binary path and arguments.
    Process {
        binary: String,
        #[serde(default)]
        args: Vec<String>,
    },
    /// An OCI container image reference (e.g. `"nginx:latest"`, `"ghcr.io/user/app:v1"`).
    Container { image: String },
}

impl Default for WorkloadKind {
    fn default() -> Self {
        Self::Database {
            engine: DatabaseEngine::default(),
        }
    }
}

impl fmt::Display for WorkloadKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Database { engine } => write!(f, "database:{}", engine),
            Self::Process { binary, .. } => write!(f, "process:{}", binary),
            Self::Container { image } => write!(f, "container:{}", image),
        }
    }
}

impl WorkloadKind {
    /// Create a database workload from a `DatabaseEngine`.
    pub fn database(engine: DatabaseEngine) -> Self {
        Self::Database { engine }
    }

    /// Create a process workload from a binary path and arguments.
    pub fn process(binary: impl Into<String>, args: Vec<String>) -> Self {
        Self::Process {
            binary: binary.into(),
            args,
        }
    }

    /// Create a container workload from an image reference.
    pub fn container(image: impl Into<String>) -> Self {
        Self::Container {
            image: image.into(),
        }
    }

    /// Returns the `DatabaseEngine` if this is a database workload.
    pub fn as_database(&self) -> Option<&DatabaseEngine> {
        match self {
            Self::Database { engine } => Some(engine),
            _ => None,
        }
    }

    /// Whether this workload has a long-running server process.
    pub fn has_server_process(&self) -> bool {
        match self {
            Self::Database { engine } => engine.has_server_process(),
            Self::Process { .. } => true,
            Self::Container { .. } => true,
        }
    }

    /// Human-readable display name.
    pub fn display_name(&self) -> String {
        match self {
            Self::Database { engine } => engine.display_name().to_string(),
            Self::Process { binary, .. } => {
                // Extract filename from path
                std::path::Path::new(binary)
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| binary.clone())
            }
            Self::Container { image } => image.clone(),
        }
    }
}

/// Hardware accelerator type for GPU/TPU/NPU/LPU passthrough.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AcceleratorKind {
    /// GPU — CUDA (NVIDIA), ROCm (AMD), Metal (Apple).
    GPU,
    /// TPU — Google Tensor Processing Unit (Coral USB/PCIe).
    TPU,
    /// NPU — Neural Processing Unit (Apple Neural Engine, Intel NPU, Qualcomm Hexagon).
    NPU,
    /// LPU — Language Processing Unit (Groq inference chips).
    LPU,
    /// FPGA — Field-Programmable Gate Array (Xilinx, Intel Altera).
    FPGA,
    /// DPU — Data Processing Unit (NVIDIA BlueField, AMD Pensando).
    DPU,
    /// VPU — Vision Processing Unit (Intel Movidius).
    VPU,
    /// DSP — Digital Signal Processor (Qualcomm Hexagon DSP, TI C66x).
    DSP,
    /// RDU — Reconfigurable Dataflow Unit (SambaNova).
    RDU,
    /// WSE — Wafer-Scale Engine (Cerebras).
    WSE,
    /// Neuromorphic — Intel Loihi 2, BrainChip Akida, SpiNNaker2.
    Neuromorphic,
    /// Photonic — Lightmatter, Luminous Computing.
    Photonic,
    /// Custom accelerator type.
    Custom(String),
}

impl fmt::Display for AcceleratorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GPU => write!(f, "gpu"),
            Self::TPU => write!(f, "tpu"),
            Self::NPU => write!(f, "npu"),
            Self::LPU => write!(f, "lpu"),
            Self::FPGA => write!(f, "fpga"),
            Self::DPU => write!(f, "dpu"),
            Self::VPU => write!(f, "vpu"),
            Self::DSP => write!(f, "dsp"),
            Self::RDU => write!(f, "rdu"),
            Self::WSE => write!(f, "wse"),
            Self::Neuromorphic => write!(f, "neuromorphic"),
            Self::Photonic => write!(f, "photonic"),
            Self::Custom(name) => write!(f, "{}", name),
        }
    }
}

impl FromStr for AcceleratorKind {
    type Err = RuntimeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "gpu" | "cuda" | "metal" | "rocm" | "vulkan" | "directml" => Ok(Self::GPU),
            "tpu" | "coral" | "trillium" | "ironwood" => Ok(Self::TPU),
            "npu" | "neural-engine" | "ane" | "ethos" | "ethos-u85" | "hexagon-npu" | "exynos-npu" => Ok(Self::NPU),
            "lpu" | "groq" | "groq3" => Ok(Self::LPU),
            "fpga" | "xilinx" | "altera" | "versal" | "agilex" | "lattice" => Ok(Self::FPGA),
            "dpu" | "ipu" | "bluefield" | "pensando" => Ok(Self::DPU),
            "vpu" | "movidius" | "myriad" | "ambarella" => Ok(Self::VPU),
            "dsp" | "hexagon" | "hexagon-dsp" | "c66x" | "tensilica" => Ok(Self::DSP),
            "rdu" | "sambanova" | "sn50" | "sn40l" => Ok(Self::RDU),
            "wse" | "cerebras" | "wse-3" => Ok(Self::WSE),
            "neuromorphic" | "loihi" | "loihi2" | "akida" | "spinnaker" | "spinnaker2" => Ok(Self::Neuromorphic),
            "photonic" | "lightmatter" | "luminous" => Ok(Self::Photonic),
            other => Ok(Self::Custom(other.to_string())),
        }
    }
}

/// Binding of a hardware accelerator to an instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcceleratorBinding {
    /// Type of accelerator.
    pub kind: AcceleratorKind,
    /// Specific device ID (e.g. `"0"`, `"GPU-abc123"`). None = auto-select.
    pub device_id: Option<String>,
    /// Memory limit in megabytes. None = use all available.
    pub memory_mb: Option<u64>,
}

/// Volume mount from host to instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeMount {
    /// Path on the host filesystem.
    pub host_path: String,
    /// Path inside the container/instance.
    pub container_path: String,
    /// Whether the mount is read-only.
    #[serde(default)]
    pub read_only: bool,
}

/// Runtime error types.
#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("unsupported runtime mode: {0} (compile with the appropriate feature flag)")]
    UnsupportedMode(RuntimeMode),

    #[error("VM error: {0}")]
    VMError(String),

    #[error("WASM error: {0}")]
    WasmError(String),

    #[error("instance not found: {0}")]
    InstanceNotFound(String),

    #[error("instance already exists: {0}")]
    InstanceAlreadyExists(String),

    #[error("configuration error: {0}")]
    ConfigError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("process error: {0}")]
    ProcessError(String),

    #[error("internal error: {0}")]
    Internal(String),
}

/// Unique instance identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct InstanceId(pub String);

impl InstanceId {
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4().to_string())
    }

    pub fn from_string(s: String) -> Self {
        Self(s)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for InstanceId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for InstanceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// State of a running instance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InstanceState {
    Starting,
    Running,
    Stopping,
    Stopped,
    Failed(String),
}

impl fmt::Display for InstanceState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Starting => write!(f, "starting"),
            Self::Running => write!(f, "running"),
            Self::Stopping => write!(f, "stopping"),
            Self::Stopped => write!(f, "stopped"),
            Self::Failed(reason) => write!(f, "failed: {}", reason),
        }
    }
}

/// Metadata about a running instance (database, process, or container).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceInfo {
    pub id: InstanceId,
    pub name: String,
    /// Database engine (kept for backward compatibility; canonical source is `workload`).
    #[serde(default)]
    pub engine: DatabaseEngine,
    /// The generalized workload type (database, process, or container).
    #[serde(default)]
    pub workload: WorkloadKind,
    pub mode: RuntimeMode,
    pub state: InstanceState,
    pub created_at: DateTime<Utc>,
    pub pid: Option<u32>,
    pub ports: Vec<PortMapping>,
    pub data_dir: String,
    pub node_id: Option<String>,
    /// Port for the Joule energy sidecar HTTP server (external engines only).
    #[serde(default)]
    pub energy_port: Option<u16>,
    /// Hardware accelerator bindings (GPU/TPU/NPU/LPU passthrough).
    #[serde(default)]
    pub accelerators: Vec<AcceleratorBinding>,
    /// Volume mounts from host to instance.
    #[serde(default)]
    pub volumes: Vec<VolumeMount>,
    /// Environment variables passed to the instance.
    #[serde(default)]
    pub env_vars: HashMap<String, String>,
    /// Labels for metadata and filtering (e.g. `{"team": "ml", "env": "staging"}`).
    #[serde(default)]
    pub labels: HashMap<String, String>,
}

/// Port mapping from host to instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortMapping {
    pub protocol: String,
    pub host_port: u16,
    pub instance_port: u16,
}

/// Configuration for the runtime manager.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    pub mode: RuntimeMode,
    /// VM memory allocation in MB (default: 4096).
    pub vm_memory_mb: u32,
    /// VM CPU core count (default: 4).
    pub vm_cpu_cores: u32,
    /// VM kernel path (optional, for custom kernels).
    pub vm_kernel_path: Option<String>,
    /// VM disk image path (optional).
    pub vm_disk_image: Option<String>,
    /// WASM max memory in bytes (default: 256MB).
    pub wasm_max_memory_bytes: usize,
    /// WASM fuel limit (default: 1 billion).
    pub wasm_fuel: u64,
    /// WASM execution timeout in seconds (default: 30).
    pub wasm_timeout_secs: u64,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            mode: RuntimeMode::Native,
            vm_memory_mb: 4096,
            vm_cpu_cores: 4,
            vm_kernel_path: None,
            vm_disk_image: None,
            wasm_max_memory_bytes: 256 * 1024 * 1024, // 256 MB
            wasm_fuel: 1_000_000_000,
            wasm_timeout_secs: 30,
        }
    }
}

impl RuntimeConfig {
    pub fn native() -> Self {
        Self::default()
    }

    pub fn vm() -> Self {
        Self {
            mode: RuntimeMode::VM,
            ..Self::default()
        }
    }

    pub fn wasm() -> Self {
        Self {
            mode: RuntimeMode::WASM,
            ..Self::default()
        }
    }

    pub fn validate(&self) -> Result<(), RuntimeError> {
        match self.mode {
            RuntimeMode::VM => {
                if self.vm_memory_mb < 512 {
                    return Err(RuntimeError::ConfigError(
                        "VM memory must be at least 512 MB".into(),
                    ));
                }
                if self.vm_cpu_cores < 1 {
                    return Err(RuntimeError::ConfigError(
                        "VM must have at least 1 CPU core".into(),
                    ));
                }
            }
            RuntimeMode::WASM => {
                if self.wasm_max_memory_bytes < 1024 * 1024 {
                    return Err(RuntimeError::ConfigError(
                        "WASM max memory must be at least 1 MB".into(),
                    ));
                }
                if self.wasm_timeout_secs == 0 {
                    return Err(RuntimeError::ConfigError(
                        "WASM timeout must be greater than 0".into(),
                    ));
                }
            }
            RuntimeMode::Native => {}
        }
        Ok(())
    }
}

/// Server configuration overrides passed when starting an instance.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ServerOverrides {
    // JouleDB-specific port fields (kept for backward compat)
    pub http_port: Option<u16>,
    pub tcp_port: Option<u16>,
    pub pgwire_port: Option<u16>,
    pub raft_port: Option<u16>,
    pub data_dir: Option<String>,
    pub auth_enabled: Option<bool>,
    pub extra_args: Vec<String>,
    /// Generic engine port override (used for non-JouleDB engines).
    #[serde(default)]
    pub engine_port: Option<u16>,
    /// Bind address override.
    #[serde(default)]
    pub bind_address: Option<String>,
}

#[cfg(test)]
mod proptest_verify;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_mode_display() {
        assert_eq!(RuntimeMode::Native.to_string(), "native");
        assert_eq!(RuntimeMode::VM.to_string(), "vm");
        assert_eq!(RuntimeMode::WASM.to_string(), "wasm");
    }

    #[test]
    fn test_runtime_mode_parse() {
        assert_eq!(
            "native".parse::<RuntimeMode>().unwrap(),
            RuntimeMode::Native
        );
        assert_eq!("vm".parse::<RuntimeMode>().unwrap(), RuntimeMode::VM);
        assert_eq!("wasm".parse::<RuntimeMode>().unwrap(), RuntimeMode::WASM);
        assert_eq!(
            "NATIVE".parse::<RuntimeMode>().unwrap(),
            RuntimeMode::Native
        );
        assert_eq!("VM".parse::<RuntimeMode>().unwrap(), RuntimeMode::VM);
        assert_eq!("Wasm".parse::<RuntimeMode>().unwrap(), RuntimeMode::WASM);
        assert!("invalid".parse::<RuntimeMode>().is_err());
    }

    #[test]
    fn test_runtime_mode_roundtrip() {
        for mode in [RuntimeMode::Native, RuntimeMode::VM, RuntimeMode::WASM] {
            let s = mode.to_string();
            let parsed: RuntimeMode = s.parse().unwrap();
            assert_eq!(parsed, mode);
        }
    }

    #[test]
    fn test_runtime_mode_serde() {
        let mode = RuntimeMode::VM;
        let json = serde_json::to_string(&mode).unwrap();
        assert_eq!(json, "\"vm\"");
        let parsed: RuntimeMode = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, mode);
    }

    #[test]
    fn test_instance_id_unique() {
        let id1 = InstanceId::new();
        let id2 = InstanceId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_instance_state_display() {
        assert_eq!(InstanceState::Running.to_string(), "running");
        assert_eq!(InstanceState::Stopped.to_string(), "stopped");
        assert_eq!(
            InstanceState::Failed("oom".into()).to_string(),
            "failed: oom"
        );
    }

    #[test]
    fn test_runtime_config_defaults() {
        let cfg = RuntimeConfig::default();
        assert_eq!(cfg.mode, RuntimeMode::Native);
        assert_eq!(cfg.vm_memory_mb, 4096);
        assert_eq!(cfg.vm_cpu_cores, 4);
        assert_eq!(cfg.wasm_max_memory_bytes, 256 * 1024 * 1024);
        assert_eq!(cfg.wasm_fuel, 1_000_000_000);
    }

    #[test]
    fn test_runtime_config_validate() {
        // Native always valid
        let cfg = RuntimeConfig::native();
        assert!(cfg.validate().is_ok());

        // VM with too little memory
        let mut cfg = RuntimeConfig::vm();
        cfg.vm_memory_mb = 100;
        assert!(cfg.validate().is_err());

        // WASM with zero timeout
        let mut cfg = RuntimeConfig::wasm();
        cfg.wasm_timeout_secs = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_instance_info_serde() {
        let info = InstanceInfo {
            id: InstanceId::from_string("test-123".into()),
            name: "my-db".into(),
            engine: DatabaseEngine::JouleDB,
            workload: WorkloadKind::database(DatabaseEngine::JouleDB),
            mode: RuntimeMode::Native,
            state: InstanceState::Running,
            created_at: Utc::now(),
            pid: Some(12345),
            ports: vec![PortMapping {
                protocol: "http".into(),
                host_port: 8080,
                instance_port: 8080,
            }],
            data_dir: "/tmp/jouledb".into(),
            node_id: None,
            energy_port: None,
            accelerators: vec![],
            volumes: vec![],
            env_vars: HashMap::new(),
            labels: HashMap::new(),
        };
        let json = serde_json::to_string(&info).unwrap();
        let parsed: InstanceInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, info.id);
        assert_eq!(parsed.name, info.name);
        assert_eq!(parsed.mode, info.mode);
        assert_eq!(parsed.engine, DatabaseEngine::JouleDB);
        assert_eq!(
            parsed.workload,
            WorkloadKind::database(DatabaseEngine::JouleDB)
        );
        assert!(parsed.accelerators.is_empty());
        assert!(parsed.volumes.is_empty());
        assert!(parsed.env_vars.is_empty());
        assert!(parsed.labels.is_empty());
    }

    #[test]
    fn test_instance_info_backward_compat() {
        // Old JSON without engine/energy_port/workload/accelerators/volumes/env_vars/labels
        // should default correctly
        let json = r#"{
            "id": "old-123",
            "name": "legacy",
            "mode": "native",
            "state": "running",
            "created_at": "2026-01-01T00:00:00Z",
            "pid": null,
            "ports": [],
            "data_dir": "/tmp/test",
            "node_id": null
        }"#;
        let parsed: InstanceInfo = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.engine, DatabaseEngine::JouleDB);
        assert!(parsed.energy_port.is_none());
        assert_eq!(parsed.workload, WorkloadKind::default());
        assert!(parsed.accelerators.is_empty());
        assert!(parsed.volumes.is_empty());
        assert!(parsed.env_vars.is_empty());
        assert!(parsed.labels.is_empty());
    }

    #[test]
    fn test_server_overrides_default() {
        let overrides = ServerOverrides::default();
        assert!(overrides.http_port.is_none());
        assert!(overrides.tcp_port.is_none());
        assert!(overrides.extra_args.is_empty());
        assert!(overrides.engine_port.is_none());
        assert!(overrides.bind_address.is_none());
    }

    // --- DatabaseEngine tests ---

    #[test]
    fn test_database_engine_display() {
        assert_eq!(DatabaseEngine::JouleDB.to_string(), "jouledb");
        assert_eq!(DatabaseEngine::Postgres.to_string(), "postgres");
        assert_eq!(DatabaseEngine::MySQL.to_string(), "mysql");
        assert_eq!(DatabaseEngine::Redis.to_string(), "redis");
        assert_eq!(DatabaseEngine::MongoDB.to_string(), "mongodb");
        assert_eq!(DatabaseEngine::SQLite.to_string(), "sqlite");
        assert_eq!(
            DatabaseEngine::Custom("clickhouse".into()).to_string(),
            "clickhouse"
        );
    }

    #[test]
    fn test_database_engine_parse() {
        assert_eq!(
            "jouledb".parse::<DatabaseEngine>().unwrap(),
            DatabaseEngine::JouleDB
        );
        assert_eq!(
            "joule-db".parse::<DatabaseEngine>().unwrap(),
            DatabaseEngine::JouleDB
        );
        assert_eq!(
            "joule".parse::<DatabaseEngine>().unwrap(),
            DatabaseEngine::JouleDB
        );
        assert_eq!(
            "postgres".parse::<DatabaseEngine>().unwrap(),
            DatabaseEngine::Postgres
        );
        assert_eq!(
            "pg".parse::<DatabaseEngine>().unwrap(),
            DatabaseEngine::Postgres
        );
        assert_eq!(
            "pgsql".parse::<DatabaseEngine>().unwrap(),
            DatabaseEngine::Postgres
        );
        assert_eq!(
            "postgresql".parse::<DatabaseEngine>().unwrap(),
            DatabaseEngine::Postgres
        );
        assert_eq!(
            "mysql".parse::<DatabaseEngine>().unwrap(),
            DatabaseEngine::MySQL
        );
        assert_eq!(
            "mariadb".parse::<DatabaseEngine>().unwrap(),
            DatabaseEngine::MySQL
        );
        assert_eq!(
            "redis".parse::<DatabaseEngine>().unwrap(),
            DatabaseEngine::Redis
        );
        assert_eq!(
            "mongodb".parse::<DatabaseEngine>().unwrap(),
            DatabaseEngine::MongoDB
        );
        assert_eq!(
            "mongo".parse::<DatabaseEngine>().unwrap(),
            DatabaseEngine::MongoDB
        );
        assert_eq!(
            "sqlite".parse::<DatabaseEngine>().unwrap(),
            DatabaseEngine::SQLite
        );
        // Unknown names become Custom
        assert_eq!(
            "clickhouse".parse::<DatabaseEngine>().unwrap(),
            DatabaseEngine::Custom("clickhouse".into())
        );
    }

    #[test]
    fn test_database_engine_serde() {
        let engine = DatabaseEngine::Postgres;
        let json = serde_json::to_string(&engine).unwrap();
        assert_eq!(json, "\"postgres\"");
        let parsed: DatabaseEngine = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, engine);

        // Custom roundtrip
        let custom = DatabaseEngine::Custom("cockroach".into());
        let json = serde_json::to_string(&custom).unwrap();
        let parsed: DatabaseEngine = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, custom);
    }

    #[test]
    fn test_database_engine_properties() {
        assert_eq!(DatabaseEngine::Postgres.binary_name(), "postgres");
        assert_eq!(DatabaseEngine::Postgres.default_port(), 5432);
        assert_eq!(DatabaseEngine::Postgres.display_name(), "PostgreSQL");
        assert!(DatabaseEngine::Postgres.has_server_process());

        assert_eq!(DatabaseEngine::Redis.binary_name(), "redis-server");
        assert_eq!(DatabaseEngine::Redis.default_port(), 6379);
        assert!(DatabaseEngine::Redis.has_server_process());

        assert_eq!(DatabaseEngine::SQLite.default_port(), 0);
        assert!(!DatabaseEngine::SQLite.has_server_process());

        assert_eq!(DatabaseEngine::JouleDB.default_port(), 8080);
        assert!(DatabaseEngine::JouleDB.has_server_process());
    }

    #[test]
    fn test_database_engine_default() {
        assert_eq!(DatabaseEngine::default(), DatabaseEngine::JouleDB);
    }

    // --- WorkloadKind tests ---

    #[test]
    fn test_workload_kind_default() {
        let wk = WorkloadKind::default();
        assert_eq!(
            wk,
            WorkloadKind::Database {
                engine: DatabaseEngine::JouleDB
            }
        );
    }

    #[test]
    fn test_workload_kind_constructors() {
        let db = WorkloadKind::database(DatabaseEngine::Postgres);
        assert_eq!(db.as_database(), Some(&DatabaseEngine::Postgres));
        assert!(db.has_server_process());
        assert_eq!(db.display_name(), "PostgreSQL");

        let proc = WorkloadKind::process("/usr/bin/nginx", vec!["-g".into(), "daemon off;".into()]);
        assert!(proc.as_database().is_none());
        assert!(proc.has_server_process());
        assert_eq!(proc.display_name(), "nginx");

        let container = WorkloadKind::container("nginx:latest");
        assert!(container.as_database().is_none());
        assert!(container.has_server_process());
        assert_eq!(container.display_name(), "nginx:latest");
    }

    #[test]
    fn test_workload_kind_display() {
        assert_eq!(
            WorkloadKind::database(DatabaseEngine::Redis).to_string(),
            "database:redis"
        );
        assert_eq!(
            WorkloadKind::process("myapp", vec![]).to_string(),
            "process:myapp"
        );
        assert_eq!(
            WorkloadKind::container("ghcr.io/user/app:v1").to_string(),
            "container:ghcr.io/user/app:v1"
        );
    }

    #[test]
    fn test_workload_kind_serde() {
        let variants = vec![
            WorkloadKind::database(DatabaseEngine::Postgres),
            WorkloadKind::process("nginx", vec!["-g".into()]),
            WorkloadKind::container("redis:7"),
        ];
        for wk in variants {
            let json = serde_json::to_string(&wk).unwrap();
            let parsed: WorkloadKind = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, wk);
        }
    }

    #[test]
    fn test_workload_kind_sqlite_no_server() {
        let wk = WorkloadKind::database(DatabaseEngine::SQLite);
        assert!(!wk.has_server_process());
    }

    // --- AcceleratorKind tests ---

    #[test]
    fn test_accelerator_kind_display() {
        assert_eq!(AcceleratorKind::GPU.to_string(), "gpu");
        assert_eq!(AcceleratorKind::TPU.to_string(), "tpu");
        assert_eq!(AcceleratorKind::NPU.to_string(), "npu");
        assert_eq!(AcceleratorKind::LPU.to_string(), "lpu");
        assert_eq!(AcceleratorKind::Custom("fpga".into()).to_string(), "fpga");
    }

    #[test]
    fn test_accelerator_kind_parse() {
        assert_eq!(
            "gpu".parse::<AcceleratorKind>().unwrap(),
            AcceleratorKind::GPU
        );
        assert_eq!(
            "cuda".parse::<AcceleratorKind>().unwrap(),
            AcceleratorKind::GPU
        );
        assert_eq!(
            "metal".parse::<AcceleratorKind>().unwrap(),
            AcceleratorKind::GPU
        );
        assert_eq!(
            "tpu".parse::<AcceleratorKind>().unwrap(),
            AcceleratorKind::TPU
        );
        assert_eq!(
            "coral".parse::<AcceleratorKind>().unwrap(),
            AcceleratorKind::TPU
        );
        assert_eq!(
            "npu".parse::<AcceleratorKind>().unwrap(),
            AcceleratorKind::NPU
        );
        assert_eq!(
            "neural-engine".parse::<AcceleratorKind>().unwrap(),
            AcceleratorKind::NPU
        );
        assert_eq!(
            "ane".parse::<AcceleratorKind>().unwrap(),
            AcceleratorKind::NPU
        );
        assert_eq!(
            "lpu".parse::<AcceleratorKind>().unwrap(),
            AcceleratorKind::LPU
        );
        assert_eq!(
            "groq".parse::<AcceleratorKind>().unwrap(),
            AcceleratorKind::LPU
        );
        assert_eq!(
            "fpga".parse::<AcceleratorKind>().unwrap(),
            AcceleratorKind::FPGA
        );
        assert_eq!("xilinx".parse::<AcceleratorKind>().unwrap(), AcceleratorKind::FPGA);
        assert_eq!("versal".parse::<AcceleratorKind>().unwrap(), AcceleratorKind::FPGA);
        assert_eq!("agilex".parse::<AcceleratorKind>().unwrap(), AcceleratorKind::FPGA);
        assert_eq!("dpu".parse::<AcceleratorKind>().unwrap(), AcceleratorKind::DPU);
        assert_eq!("ipu".parse::<AcceleratorKind>().unwrap(), AcceleratorKind::DPU);
        assert_eq!("bluefield".parse::<AcceleratorKind>().unwrap(), AcceleratorKind::DPU);
        assert_eq!("vpu".parse::<AcceleratorKind>().unwrap(), AcceleratorKind::VPU);
        assert_eq!("movidius".parse::<AcceleratorKind>().unwrap(), AcceleratorKind::VPU);
        assert_eq!("dsp".parse::<AcceleratorKind>().unwrap(), AcceleratorKind::DSP);
        assert_eq!("hexagon-dsp".parse::<AcceleratorKind>().unwrap(), AcceleratorKind::DSP);
        assert_eq!("tensilica".parse::<AcceleratorKind>().unwrap(), AcceleratorKind::DSP);
        assert_eq!("rdu".parse::<AcceleratorKind>().unwrap(), AcceleratorKind::RDU);
        assert_eq!("sambanova".parse::<AcceleratorKind>().unwrap(), AcceleratorKind::RDU);
        assert_eq!("sn50".parse::<AcceleratorKind>().unwrap(), AcceleratorKind::RDU);
        assert_eq!("wse".parse::<AcceleratorKind>().unwrap(), AcceleratorKind::WSE);
        assert_eq!("cerebras".parse::<AcceleratorKind>().unwrap(), AcceleratorKind::WSE);
        assert_eq!("neuromorphic".parse::<AcceleratorKind>().unwrap(), AcceleratorKind::Neuromorphic);
        assert_eq!("loihi".parse::<AcceleratorKind>().unwrap(), AcceleratorKind::Neuromorphic);
        assert_eq!("loihi2".parse::<AcceleratorKind>().unwrap(), AcceleratorKind::Neuromorphic);
        assert_eq!("akida".parse::<AcceleratorKind>().unwrap(), AcceleratorKind::Neuromorphic);
        assert_eq!("photonic".parse::<AcceleratorKind>().unwrap(), AcceleratorKind::Photonic);
        assert_eq!("lightmatter".parse::<AcceleratorKind>().unwrap(), AcceleratorKind::Photonic);
        assert_eq!("luminous".parse::<AcceleratorKind>().unwrap(), AcceleratorKind::Photonic);
        // Unknown falls through to Custom
        assert_eq!("quantum".parse::<AcceleratorKind>().unwrap(), AcceleratorKind::Custom("quantum".into()));
    }

    #[test]
    fn test_accelerator_kind_serde() {
        let kinds = vec![
            AcceleratorKind::GPU,
            AcceleratorKind::TPU,
            AcceleratorKind::NPU,
            AcceleratorKind::LPU,
            AcceleratorKind::FPGA,
            AcceleratorKind::DPU,
            AcceleratorKind::VPU,
            AcceleratorKind::DSP,
            AcceleratorKind::RDU,
            AcceleratorKind::WSE,
            AcceleratorKind::Neuromorphic,
            AcceleratorKind::Photonic,
            AcceleratorKind::Custom("quantum".into()),
        ];
        for kind in kinds {
            let json = serde_json::to_string(&kind).unwrap();
            let parsed: AcceleratorKind = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, kind);
        }
    }

    // --- AcceleratorBinding tests ---

    #[test]
    fn test_accelerator_binding_serde() {
        let binding = AcceleratorBinding {
            kind: AcceleratorKind::GPU,
            device_id: Some("0".into()),
            memory_mb: Some(16384),
        };
        let json = serde_json::to_string(&binding).unwrap();
        let parsed: AcceleratorBinding = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.kind, AcceleratorKind::GPU);
        assert_eq!(parsed.device_id.as_deref(), Some("0"));
        assert_eq!(parsed.memory_mb, Some(16384));
    }

    // --- VolumeMount tests ---

    #[test]
    fn test_volume_mount_serde() {
        let mount = VolumeMount {
            host_path: "/data/models".into(),
            container_path: "/models".into(),
            read_only: true,
        };
        let json = serde_json::to_string(&mount).unwrap();
        let parsed: VolumeMount = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.host_path, "/data/models");
        assert_eq!(parsed.container_path, "/models");
        assert!(parsed.read_only);
    }

    #[test]
    fn test_volume_mount_read_only_default() {
        // read_only should default to false
        let json = r#"{"host_path": "/tmp", "container_path": "/data"}"#;
        let parsed: VolumeMount = serde_json::from_str(json).unwrap();
        assert!(!parsed.read_only);
    }

    // --- InstanceInfo with new fields ---

    #[test]
    fn test_instance_info_with_accelerators() {
        let info = InstanceInfo {
            id: InstanceId::from_string("gpu-test".into()),
            name: "llm-runner".into(),
            engine: DatabaseEngine::default(),
            workload: WorkloadKind::container("llama3:8b"),
            mode: RuntimeMode::Native,
            state: InstanceState::Running,
            created_at: Utc::now(),
            pid: Some(42),
            ports: vec![],
            data_dir: "/tmp/llm".into(),
            node_id: None,
            energy_port: None,
            accelerators: vec![AcceleratorBinding {
                kind: AcceleratorKind::GPU,
                device_id: Some("0".into()),
                memory_mb: Some(16384),
            }],
            volumes: vec![VolumeMount {
                host_path: "/models".into(),
                container_path: "/data/models".into(),
                read_only: true,
            }],
            env_vars: HashMap::from([("CUDA_VISIBLE_DEVICES".into(), "0".into())]),
            labels: HashMap::from([("team".into(), "ml".into())]),
        };

        let json = serde_json::to_string(&info).unwrap();
        let parsed: InstanceInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.workload, WorkloadKind::container("llama3:8b"));
        assert_eq!(parsed.accelerators.len(), 1);
        assert_eq!(parsed.accelerators[0].kind, AcceleratorKind::GPU);
        assert_eq!(parsed.volumes.len(), 1);
        assert_eq!(parsed.env_vars.get("CUDA_VISIBLE_DEVICES").unwrap(), "0");
        assert_eq!(parsed.labels.get("team").unwrap(), "ml");
    }
}
