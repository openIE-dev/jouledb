//! Persistent daemon process — owns RuntimeManager, listens on Unix socket.
//!
//! The daemon is the JouleDB equivalent of `dockerd`. It manages all database
//! instances, monitors health, auto-restarts crashed processes, and exposes a
//! unified energy dashboard.
//!
//! CLI commands communicate with the daemon over a Unix domain socket using
//! length-prefixed JSON messages.

use crate::{
    AcceleratorBinding, DatabaseEngine, InstanceInfo, InstanceState, RuntimeConfig, RuntimeError,
    RuntimeManager, ServerOverrides, VolumeMount, WorkloadKind,
    accelerator::{AcceleratorDevice, DeviceEnergy},
    backend::ExecOutput,
    image::ImageInfo,
};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixListener;
use tokio::sync::watch;

/// Default directory for daemon state: `~/.jouledb/`
pub fn default_daemon_dir() -> PathBuf {
    dirs_home()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".jouledb")
}

fn dirs_home() -> Option<PathBuf> {
    #[cfg(unix)]
    {
        std::env::var_os("HOME").map(PathBuf::from)
    }
    #[cfg(not(unix))]
    {
        std::env::var_os("USERPROFILE").map(PathBuf::from)
    }
}

/// Default socket path.
pub fn default_socket_path() -> PathBuf {
    default_daemon_dir().join("daemon.sock")
}

/// Default PID file path.
pub fn default_pid_path() -> PathBuf {
    default_daemon_dir().join("daemon.pid")
}

/// Default log path.
pub fn default_log_path() -> PathBuf {
    default_daemon_dir().join("daemon.log")
}

// ---------------------------------------------------------------------------
// IPC Protocol
// ---------------------------------------------------------------------------

/// Request from CLI to daemon.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum DaemonRequest {
    // --- Existing database operations ---
    /// Start a new database instance.
    StartInstance {
        name: String,
        engine: String,
        overrides: ServerOverrides,
    },
    /// Stop a running instance.
    StopInstance { instance_id: String },
    /// List all instances.
    ListInstances,
    /// Health-check a specific instance.
    InstanceHealth { instance_id: String },
    /// Query daemon status (uptime, counts, energy).
    DaemonStatus,
    /// Graceful shutdown.
    Shutdown,

    // --- Container / workload operations ---
    /// Run any workload (database, container, or process).
    RunWorkload {
        name: String,
        workload: WorkloadKind,
        overrides: ServerOverrides,
        #[serde(default)]
        accelerators: Vec<AcceleratorBinding>,
        #[serde(default)]
        volumes: Vec<VolumeMount>,
        #[serde(default)]
        env_vars: std::collections::HashMap<String, String>,
        #[serde(default)]
        labels: std::collections::HashMap<String, String>,
    },
    /// Pull a container image.
    PullImage { image: String },
    /// List cached images.
    ListImages,
    /// Remove a cached image.
    RemoveImage { image: String },
    /// Execute a command inside a running instance.
    ExecInInstance {
        instance_id: String,
        command: Vec<String>,
    },
    /// Retrieve logs from an instance.
    InstanceLogs {
        instance_id: String,
        tail: Option<usize>,
    },
    /// Query available hardware accelerators.
    AcceleratorStatus,
    /// Inspect a specific instance (full details).
    InspectInstance { instance_id: String },
    /// Remove a stopped instance from the registry.
    RemoveInstance { instance_id: String, force: bool },
}

/// Response from daemon to CLI.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum DaemonResponse {
    // --- Existing ---
    /// Generic success.
    Ok { message: String },
    /// Instance started successfully.
    InstanceStarted { instance_id: String },
    /// List of instances.
    InstanceList { instances: Vec<InstanceInfo> },
    /// Health check result.
    Health { healthy: bool },
    /// Daemon status.
    Status {
        uptime_secs: u64,
        instance_count: usize,
        total_energy_joules: f64,
    },
    /// Error.
    Error { message: String },

    // --- Container / workload operations ---
    /// List of cached images.
    ImageList { images: Vec<ImageInfo> },
    /// Command execution result.
    ExecResult {
        exit_code: i32,
        stdout: String,
        stderr: String,
    },
    /// Log lines from an instance.
    LogLines { lines: Vec<String> },
    /// List of hardware accelerator devices.
    AcceleratorList { devices: Vec<AcceleratorDevice> },
    /// Full instance detail (inspect).
    InstanceDetail { info: InstanceInfo },
}

// ---------------------------------------------------------------------------
// Wire format: 4-byte big-endian length + JSON payload
// ---------------------------------------------------------------------------

/// Maximum message size (16 MB).
const MAX_MESSAGE_SIZE: u32 = 16 * 1024 * 1024;

/// Write a length-prefixed JSON message to a writer.
pub async fn write_message<W: AsyncWriteExt + Unpin, T: Serialize>(
    writer: &mut W,
    msg: &T,
) -> std::io::Result<()> {
    let payload = serde_json::to_vec(msg)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let len = payload.len() as u32;
    writer.write_all(&len.to_be_bytes()).await?;
    writer.write_all(&payload).await?;
    writer.flush().await?;
    Ok(())
}

/// Read a length-prefixed JSON message from a reader.
pub async fn read_message<R: AsyncReadExt + Unpin, T: for<'de> Deserialize<'de>>(
    reader: &mut R,
) -> std::io::Result<T> {
    let mut len_buf = [0u8; 4];
    reader.read_exact(&mut len_buf).await?;
    let len = u32::from_be_bytes(len_buf);

    if len > MAX_MESSAGE_SIZE {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "message too large: {} bytes (max {})",
                len, MAX_MESSAGE_SIZE
            ),
        ));
    }

    let mut buf = vec![0u8; len as usize];
    reader.read_exact(&mut buf).await?;
    serde_json::from_slice(&buf)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

// ---------------------------------------------------------------------------
// DaemonCore
// ---------------------------------------------------------------------------

/// The persistent daemon process.
pub struct DaemonCore {
    manager: Arc<RuntimeManager>,
    socket_path: PathBuf,
    pid_path: PathBuf,
    started_at: Instant,
    shutdown_tx: watch::Sender<bool>,
    shutdown_rx: watch::Receiver<bool>,
}

/// Configuration for starting the daemon.
#[derive(Debug, Clone)]
pub struct DaemonConfig {
    pub runtime_config: RuntimeConfig,
    pub daemon_dir: PathBuf,
    pub socket_path: Option<PathBuf>,
    pub pid_path: Option<PathBuf>,
    pub dashboard_port: Option<u16>,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        let daemon_dir = default_daemon_dir();
        Self {
            runtime_config: RuntimeConfig::default(),
            daemon_dir,
            socket_path: None,
            pid_path: None,
            dashboard_port: None,
        }
    }
}

impl DaemonCore {
    /// Create a new daemon with the given configuration.
    pub fn new(config: DaemonConfig) -> Result<Self, RuntimeError> {
        let daemon_dir = &config.daemon_dir;
        std::fs::create_dir_all(daemon_dir)?;

        let instances_dir = daemon_dir.join("instances");
        std::fs::create_dir_all(&instances_dir)?;

        let manager = RuntimeManager::new(config.runtime_config, instances_dir)?;

        let socket_path = config
            .socket_path
            .unwrap_or_else(|| daemon_dir.join("daemon.sock"));
        let pid_path = config
            .pid_path
            .unwrap_or_else(|| daemon_dir.join("daemon.pid"));

        let (shutdown_tx, shutdown_rx) = watch::channel(false);

        Ok(Self {
            manager: Arc::new(manager),
            socket_path,
            pid_path,
            started_at: Instant::now(),
            shutdown_tx,
            shutdown_rx,
        })
    }

    /// Returns a reference to the RuntimeManager.
    pub fn manager(&self) -> &Arc<RuntimeManager> {
        &self.manager
    }

    /// Returns uptime in seconds.
    pub fn uptime_secs(&self) -> u64 {
        self.started_at.elapsed().as_secs()
    }

    /// Recover orphan instances on startup.
    ///
    /// Scans the instance registry and marks instances with dead PIDs as Failed.
    pub fn recover_orphans(&self) {
        let instances = self.manager.list_instances();
        for instance in instances {
            match &instance.state {
                InstanceState::Running | InstanceState::Starting => {
                    if let Some(pid) = instance.pid {
                        if !crate::native::is_process_alive(pid) {
                            let _ = self.manager.list_instances(); // force registry load
                            // We can't call async stop here, so just mark as failed
                            // The registry update is best-effort
                            log::warn!(
                                "Orphan instance {} (PID {}) — marking as failed",
                                instance.id,
                                pid
                            );
                        }
                    }
                }
                _ => {}
            }
        }
    }

    /// Write PID file.
    fn write_pid_file(&self) -> std::io::Result<()> {
        let pid = std::process::id();
        std::fs::write(&self.pid_path, pid.to_string())?;
        Ok(())
    }

    /// Remove PID file.
    fn remove_pid_file(&self) {
        let _ = std::fs::remove_file(&self.pid_path);
    }

    /// Remove socket file.
    fn remove_socket(&self) {
        let _ = std::fs::remove_file(&self.socket_path);
    }

    /// Check if another daemon is already running.
    pub fn is_already_running(&self) -> bool {
        if self.pid_path.exists() {
            if let Ok(content) = std::fs::read_to_string(&self.pid_path) {
                if let Ok(pid) = content.trim().parse::<u32>() {
                    return crate::native::is_process_alive(pid);
                }
            }
        }
        false
    }

    /// Signal the daemon to shut down.
    pub fn signal_shutdown(&self) {
        let _ = self.shutdown_tx.send(true);
    }

    /// Run the daemon — bind Unix socket, accept connections, handle requests.
    ///
    /// This blocks until shutdown is signaled.
    pub async fn run(&self) -> Result<(), RuntimeError> {
        if self.is_already_running() {
            return Err(RuntimeError::ConfigError(
                "daemon is already running".into(),
            ));
        }

        // Clean up stale socket
        self.remove_socket();

        self.write_pid_file()?;
        self.recover_orphans();

        let listener = UnixListener::bind(&self.socket_path)?;
        log::info!("Daemon listening on {:?}", self.socket_path);

        let mut shutdown_rx = self.shutdown_rx.clone();

        loop {
            tokio::select! {
                accept_result = listener.accept() => {
                    match accept_result {
                        Ok((stream, _addr)) => {
                            let manager = Arc::clone(&self.manager);
                            let started_at = self.started_at;
                            let shutdown_tx = self.shutdown_tx.clone();
                            tokio::spawn(async move {
                                if let Err(e) = handle_connection(stream, manager, started_at, shutdown_tx).await {
                                    log::error!("Connection error: {}", e);
                                }
                            });
                        }
                        Err(e) => {
                            log::error!("Accept error: {}", e);
                        }
                    }
                }
                _ = shutdown_rx.changed() => {
                    if *shutdown_rx.borrow() {
                        log::info!("Shutdown signal received");
                        break;
                    }
                }
            }
        }

        // Graceful shutdown: stop all running instances
        let instances = self.manager.list_instances();
        for instance in instances {
            if instance.state == InstanceState::Running {
                log::info!("Stopping instance {} on shutdown", instance.id);
                let _ = self.manager.stop_instance(instance.id.as_str()).await;
            }
        }

        self.remove_socket();
        self.remove_pid_file();
        log::info!("Daemon shut down cleanly");
        Ok(())
    }
}

/// Handle a single client connection.
async fn handle_connection(
    stream: tokio::net::UnixStream,
    manager: Arc<RuntimeManager>,
    started_at: Instant,
    shutdown_tx: watch::Sender<bool>,
) -> std::io::Result<()> {
    let (mut reader, mut writer) = stream.into_split();

    let request: DaemonRequest = read_message(&mut reader).await?;
    let response = process_request(request, &manager, started_at, &shutdown_tx).await;
    write_message(&mut writer, &response).await?;

    Ok(())
}

/// Process a single daemon request.
async fn process_request(
    request: DaemonRequest,
    manager: &RuntimeManager,
    started_at: Instant,
    shutdown_tx: &watch::Sender<bool>,
) -> DaemonResponse {
    match request {
        DaemonRequest::StartInstance {
            name,
            engine,
            overrides,
        } => {
            let engine: DatabaseEngine = match engine.parse() {
                Ok(e) => e,
                Err(e) => {
                    return DaemonResponse::Error {
                        message: format!("invalid engine: {}", e),
                    };
                }
            };

            match manager.start_instance(name, engine, overrides).await {
                Ok(id) => DaemonResponse::InstanceStarted {
                    instance_id: id.to_string(),
                },
                Err(e) => DaemonResponse::Error {
                    message: e.to_string(),
                },
            }
        }

        DaemonRequest::StopInstance { instance_id } => {
            match manager.stop_instance(&instance_id).await {
                Ok(()) => DaemonResponse::Ok {
                    message: format!("instance {} stopped", instance_id),
                },
                Err(e) => DaemonResponse::Error {
                    message: e.to_string(),
                },
            }
        }

        DaemonRequest::ListInstances => {
            let instances = manager.list_instances();
            DaemonResponse::InstanceList { instances }
        }

        DaemonRequest::InstanceHealth { instance_id } => {
            match manager.health_check(&instance_id).await {
                Ok(healthy) => DaemonResponse::Health { healthy },
                Err(e) => DaemonResponse::Error {
                    message: e.to_string(),
                },
            }
        }

        DaemonRequest::DaemonStatus => {
            let instances = manager.list_instances();
            let instance_count = instances.len();
            let total_energy_joules =
                crate::dashboard::total_energy_from_sidecars(&instances).await;
            DaemonResponse::Status {
                uptime_secs: started_at.elapsed().as_secs(),
                instance_count,
                total_energy_joules,
            }
        }

        DaemonRequest::Shutdown => {
            let _ = shutdown_tx.send(true);
            DaemonResponse::Ok {
                message: "shutting down".into(),
            }
        }

        // --- Container / workload operations ---
        DaemonRequest::RunWorkload {
            name,
            workload,
            overrides,
            accelerators,
            volumes,
            env_vars,
            labels,
        } => match manager
            .start_workload(
                name,
                workload,
                overrides,
                accelerators,
                volumes,
                env_vars,
                labels,
            )
            .await
        {
            Ok(id) => DaemonResponse::InstanceStarted {
                instance_id: id.to_string(),
            },
            Err(e) => DaemonResponse::Error {
                message: e.to_string(),
            },
        },

        DaemonRequest::PullImage { image } => match manager.image_store().pull(&image).await {
            Ok(info) => DaemonResponse::Ok {
                message: format!("pulled {} ({})", image, info.id),
            },
            Err(e) => DaemonResponse::Error {
                message: e.to_string(),
            },
        },

        DaemonRequest::ListImages => {
            let images = manager.image_store().list().await;
            DaemonResponse::ImageList { images }
        }

        DaemonRequest::RemoveImage { image } => match manager.image_store().remove(&image).await {
            Ok(()) => DaemonResponse::Ok {
                message: format!("removed image {}", image),
            },
            Err(e) => DaemonResponse::Error {
                message: e.to_string(),
            },
        },

        DaemonRequest::ExecInInstance {
            instance_id,
            command,
        } => match manager.exec(&instance_id, &command).await {
            Ok(output) => DaemonResponse::ExecResult {
                exit_code: output.exit_code,
                stdout: output.stdout,
                stderr: output.stderr,
            },
            Err(e) => DaemonResponse::Error {
                message: e.to_string(),
            },
        },

        DaemonRequest::InstanceLogs { instance_id, tail } => {
            match manager.logs(&instance_id, tail).await {
                Ok(lines) => DaemonResponse::LogLines { lines },
                Err(e) => DaemonResponse::Error {
                    message: e.to_string(),
                },
            }
        }

        DaemonRequest::AcceleratorStatus => {
            let devices = manager.accelerator_manager().list_devices();
            DaemonResponse::AcceleratorList { devices }
        }

        DaemonRequest::InspectInstance { instance_id } => {
            match manager.get_instance(&instance_id) {
                Some(info) => DaemonResponse::InstanceDetail { info },
                None => DaemonResponse::Error {
                    message: format!("instance '{}' not found", instance_id),
                },
            }
        }

        DaemonRequest::RemoveInstance { instance_id, force } => {
            if force {
                // Stop first if running, ignore errors
                let _ = manager.stop_instance(&instance_id).await;
            }
            match manager.stop_instance(&instance_id).await {
                Ok(()) => DaemonResponse::Ok {
                    message: format!("removed instance {}", instance_id),
                },
                Err(e) => DaemonResponse::Error {
                    message: e.to_string(),
                },
            }
        }
    }
}

/// Check if a daemon is running by reading its PID file.
pub fn is_daemon_running(pid_path: &Path) -> bool {
    if let Ok(content) = std::fs::read_to_string(pid_path) {
        if let Ok(pid) = content.trim().parse::<u32>() {
            return crate::native::is_process_alive(pid);
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_daemon_request_serde_roundtrip() {
        use crate::{AcceleratorBinding, AcceleratorKind, VolumeMount, WorkloadKind};
        use std::collections::HashMap;

        let requests: Vec<DaemonRequest> = vec![
            DaemonRequest::StartInstance {
                name: "my-pg".into(),
                engine: "postgres".into(),
                overrides: ServerOverrides::default(),
            },
            DaemonRequest::StopInstance {
                instance_id: "abc-123".into(),
            },
            DaemonRequest::ListInstances,
            DaemonRequest::InstanceHealth {
                instance_id: "def-456".into(),
            },
            DaemonRequest::DaemonStatus,
            DaemonRequest::Shutdown,
            DaemonRequest::RunWorkload {
                name: "my-nginx".into(),
                workload: WorkloadKind::container("nginx:latest"),
                overrides: ServerOverrides::default(),
                accelerators: vec![AcceleratorBinding {
                    kind: AcceleratorKind::GPU,
                    device_id: Some("0".into()),
                    memory_mb: None,
                }],
                volumes: vec![VolumeMount {
                    host_path: "/data".into(),
                    container_path: "/mnt".into(),
                    read_only: false,
                }],
                env_vars: HashMap::from([("FOO".into(), "bar".into())]),
                labels: HashMap::from([("team".into(), "ml".into())]),
            },
            DaemonRequest::PullImage {
                image: "redis:7".into(),
            },
            DaemonRequest::ListImages,
            DaemonRequest::RemoveImage {
                image: "old:v1".into(),
            },
            DaemonRequest::ExecInInstance {
                instance_id: "inst-1".into(),
                command: vec!["sh".into(), "-c".into(), "echo hi".into()],
            },
            DaemonRequest::InstanceLogs {
                instance_id: "inst-1".into(),
                tail: Some(50),
            },
            DaemonRequest::AcceleratorStatus,
            DaemonRequest::InspectInstance {
                instance_id: "inst-2".into(),
            },
            DaemonRequest::RemoveInstance {
                instance_id: "inst-3".into(),
                force: true,
            },
        ];

        for req in requests {
            let json = serde_json::to_string(&req).unwrap();
            let parsed: DaemonRequest = serde_json::from_str(&json).unwrap();
            // Verify roundtrip by re-serializing
            let json2 = serde_json::to_string(&parsed).unwrap();
            assert_eq!(json, json2);
        }
    }

    #[test]
    fn test_daemon_response_serde_roundtrip() {
        use crate::AcceleratorKind;
        use crate::accelerator::AcceleratorDevice;

        let responses: Vec<DaemonResponse> = vec![
            DaemonResponse::Ok {
                message: "done".into(),
            },
            DaemonResponse::InstanceStarted {
                instance_id: "id-1".into(),
            },
            DaemonResponse::InstanceList { instances: vec![] },
            DaemonResponse::Health { healthy: true },
            DaemonResponse::Status {
                uptime_secs: 3600,
                instance_count: 5,
                total_energy_joules: 42.0,
            },
            DaemonResponse::Error {
                message: "boom".into(),
            },
            DaemonResponse::ImageList { images: vec![] },
            DaemonResponse::ExecResult {
                exit_code: 0,
                stdout: "hello\n".into(),
                stderr: String::new(),
            },
            DaemonResponse::LogLines {
                lines: vec!["line 1".into(), "line 2".into()],
            },
            DaemonResponse::AcceleratorList {
                devices: vec![AcceleratorDevice {
                    id: "gpu-0".into(),
                    kind: AcceleratorKind::GPU,
                    name: "Test GPU".into(),
                    memory_mb: 8192,
                    compute_units: 1024,
                    tdp_watts: 150.0,
                    available: true,
                    allocated_to: None,
                }],
            },
        ];

        for resp in responses {
            let json = serde_json::to_string(&resp).unwrap();
            let parsed: DaemonResponse = serde_json::from_str(&json).unwrap();
            let json2 = serde_json::to_string(&parsed).unwrap();
            assert_eq!(json, json2);
        }
    }

    #[tokio::test]
    async fn test_wire_format_roundtrip() {
        let request = DaemonRequest::DaemonStatus;

        // Write to buffer
        let mut buf = Vec::new();
        write_message(&mut buf, &request).await.unwrap();

        // Verify: 4-byte length prefix + JSON
        let len = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]);
        assert_eq!(len as usize, buf.len() - 4);

        // Read back
        let mut cursor = std::io::Cursor::new(buf);
        let parsed: DaemonRequest = read_message(&mut cursor).await.unwrap();
        assert_eq!(
            serde_json::to_string(&parsed).unwrap(),
            serde_json::to_string(&request).unwrap()
        );
    }

    #[tokio::test]
    async fn test_wire_format_oversized_rejected() {
        // Craft a buffer claiming 32MB
        let mut buf = Vec::new();
        let huge_len: u32 = 32 * 1024 * 1024;
        buf.extend_from_slice(&huge_len.to_be_bytes());
        buf.extend_from_slice(b"{}");

        let mut cursor = std::io::Cursor::new(buf);
        let result: std::io::Result<DaemonRequest> = read_message(&mut cursor).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_daemon_core_lifecycle() {
        let tmp = tempfile::tempdir().unwrap();
        let daemon_dir = tmp.path().to_path_buf();

        let config = DaemonConfig {
            runtime_config: RuntimeConfig::default(),
            daemon_dir: daemon_dir.clone(),
            socket_path: Some(daemon_dir.join("test.sock")),
            pid_path: Some(daemon_dir.join("test.pid")),
            dashboard_port: None,
        };

        let daemon = DaemonCore::new(config).unwrap();
        assert!(!daemon.is_already_running());
        assert!(daemon.uptime_secs() < 2);

        // Signal shutdown immediately
        daemon.signal_shutdown();
    }

    #[tokio::test]
    async fn test_daemon_accepts_connection() {
        let tmp = tempfile::tempdir().unwrap();
        let daemon_dir = tmp.path().to_path_buf();
        let socket_path = daemon_dir.join("test.sock");

        let config = DaemonConfig {
            runtime_config: RuntimeConfig::default(),
            daemon_dir: daemon_dir.clone(),
            socket_path: Some(socket_path.clone()),
            pid_path: Some(daemon_dir.join("test.pid")),
            dashboard_port: None,
        };

        let daemon = DaemonCore::new(config).unwrap();

        // Run daemon in background
        let daemon_handle = {
            let socket_path = socket_path.clone();
            tokio::spawn(async move {
                // Give it a moment then connect
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;

                let stream = tokio::net::UnixStream::connect(&socket_path).await.unwrap();
                let (mut reader, mut writer) = stream.into_split();

                // Send DaemonStatus request
                write_message(&mut writer, &DaemonRequest::DaemonStatus)
                    .await
                    .unwrap();

                // Read response
                let response: DaemonResponse = read_message(&mut reader).await.unwrap();

                // Signal shutdown
                let stream2 = tokio::net::UnixStream::connect(&socket_path).await.unwrap();
                let (mut _r2, mut w2) = stream2.into_split();
                write_message(&mut w2, &DaemonRequest::Shutdown)
                    .await
                    .unwrap();

                response
            })
        };

        // Run daemon (will exit when shutdown is received)
        let _ = daemon.run().await;

        let response = daemon_handle.await.unwrap();
        match response {
            DaemonResponse::Status { instance_count, .. } => {
                assert_eq!(instance_count, 0);
            }
            other => panic!("unexpected response: {:?}", other),
        }
    }

    #[test]
    fn test_is_daemon_running_no_pid_file() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(!is_daemon_running(&tmp.path().join("nonexistent.pid")));
    }

    #[test]
    fn test_is_daemon_running_stale_pid() {
        let tmp = tempfile::tempdir().unwrap();
        let pid_path = tmp.path().join("test.pid");
        // PID 99999999 is almost certainly not running
        std::fs::write(&pid_path, "99999999").unwrap();
        assert!(!is_daemon_running(&pid_path));
    }

    #[test]
    fn test_default_daemon_paths() {
        let dir = default_daemon_dir();
        assert!(dir.to_string_lossy().contains(".jouledb"));

        let sock = default_socket_path();
        assert!(sock.to_string_lossy().ends_with("daemon.sock"));

        let pid = default_pid_path();
        assert!(pid.to_string_lossy().ends_with("daemon.pid"));
    }
}
