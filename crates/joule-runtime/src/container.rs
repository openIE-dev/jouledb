//! Container Backend — OCI container lifecycle management.
//!
//! Implements `RuntimeBackend` for container workloads by delegating to
//! an available container runtime (`docker`, `podman`, `nerdctl`).
//!
//! Container instances get the same energy telemetry as database instances:
//! the energy sidecar monitors per-container CPU/GPU/NPU power draw.

use crate::{
    InstanceInfo, InstanceState, RuntimeConfig, RuntimeError, ServerOverrides,
    backend::{ExecOutput, RuntimeBackend},
    image::ImageStore,
    networking::NetworkManager,
};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::RwLock;

/// Tracks a running container.
struct ContainerProcess {
    /// Container ID assigned by the container runtime.
    container_id: String,
    /// Image reference used to start the container.
    image: String,
    /// Host PID of the container process (if known).
    pid: Option<u32>,
}

/// Backend for running OCI container workloads.
pub struct ContainerBackend {
    /// Running containers keyed by instance ID.
    containers: RwLock<HashMap<String, ContainerProcess>>,
    /// Image cache and registry manager.
    image_store: ImageStore,
    /// Network and port manager.
    network_manager: NetworkManager,
}

impl ContainerBackend {
    /// Create a new container backend.
    pub fn new(image_cache_dir: PathBuf) -> Self {
        Self {
            containers: RwLock::new(HashMap::new()),
            image_store: ImageStore::new(image_cache_dir),
            network_manager: NetworkManager::new(),
        }
    }

    /// Get a reference to the image store.
    pub fn image_store(&self) -> &ImageStore {
        &self.image_store
    }

    /// Get a reference to the network manager.
    pub fn network_manager(&self) -> &NetworkManager {
        &self.network_manager
    }

    /// Find the container tool to use.
    fn find_tool() -> Result<String, RuntimeError> {
        for tool in ["docker", "podman", "nerdctl"] {
            if crate::native::which_exists(tool) {
                return Ok(tool.to_string());
            }
        }
        Err(RuntimeError::ProcessError(
            "no container runtime found. Install docker, podman, or nerdctl.".into(),
        ))
    }

    /// Build `docker run` arguments from instance info.
    fn build_run_args(instance: &InstanceInfo, overrides: &ServerOverrides) -> Vec<String> {
        let mut args = vec![
            "run".to_string(),
            "-d".to_string(), // detached
            "--name".to_string(),
            instance.name.clone(),
        ];

        // Port mappings
        for port in &instance.ports {
            args.push("-p".to_string());
            args.push(format!("{}:{}", port.host_port, port.instance_port));
        }

        // Volume mounts
        for vol in &instance.volumes {
            args.push("-v".to_string());
            let mount = if vol.read_only {
                format!("{}:{}:ro", vol.host_path, vol.container_path)
            } else {
                format!("{}:{}", vol.host_path, vol.container_path)
            };
            args.push(mount);
        }

        // Environment variables
        for (key, value) in &instance.env_vars {
            args.push("-e".to_string());
            args.push(format!("{}={}", key, value));
        }

        // Labels
        for (key, value) in &instance.labels {
            args.push("--label".to_string());
            args.push(format!("{}={}", key, value));
        }

        // GPU passthrough
        for accel in &instance.accelerators {
            match accel.kind {
                crate::AcceleratorKind::GPU => {
                    if let Some(ref device_id) = accel.device_id {
                        // NVIDIA-specific: --gpus device=0
                        args.push("--gpus".to_string());
                        args.push(format!("device={}", device_id));
                    } else {
                        args.push("--gpus".to_string());
                        args.push("all".to_string());
                    }
                }
                _ => {
                    // Other accelerators: pass as device mapping
                    if let Some(ref device_id) = accel.device_id {
                        args.push("--device".to_string());
                        args.push(device_id.clone());
                    }
                }
            }
        }

        // Bind address override
        if let Some(ref bind) = overrides.bind_address {
            args.push("-e".to_string());
            args.push(format!("BIND_ADDRESS={}", bind));
        }

        // Extra args passthrough
        for arg in &overrides.extra_args {
            args.push(arg.clone());
        }

        // Image reference (must be last)
        if let crate::WorkloadKind::Container { ref image } = instance.workload {
            args.push(image.clone());
        }

        args
    }
}

impl RuntimeBackend for ContainerBackend {
    async fn start(
        &self,
        _config: &RuntimeConfig,
        instance: &InstanceInfo,
        overrides: &ServerOverrides,
    ) -> Result<(), RuntimeError> {
        let image = match &instance.workload {
            crate::WorkloadKind::Container { image } => image.clone(),
            other => {
                return Err(RuntimeError::ProcessError(format!(
                    "container backend cannot run workload type: {}",
                    other
                )));
            }
        };

        let tool = Self::find_tool()?;

        // Ensure image is available locally
        if !self.image_store.has(&image).await {
            log::info!("Image not cached, pulling: {}", image);
            self.image_store.pull(&image).await?;
        }

        let args = Self::build_run_args(instance, overrides);

        log::info!(
            "Starting container '{}' with {} {:?}",
            instance.name,
            tool,
            args
        );

        let output = tokio::process::Command::new(&tool)
            .args(&args)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .output()
            .await
            .map_err(|e| RuntimeError::ProcessError(format!("failed to start container: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(RuntimeError::ProcessError(format!(
                "container start failed: {}",
                stderr.trim()
            )));
        }

        let container_id = String::from_utf8_lossy(&output.stdout).trim().to_string();
        log::info!(
            "Container '{}' started: {}",
            instance.name,
            &container_id[..12.min(container_id.len())]
        );

        let mut containers = self.containers.write().unwrap();
        containers.insert(
            instance.id.0.clone(),
            ContainerProcess {
                container_id,
                image,
                pid: None,
            },
        );

        Ok(())
    }

    async fn stop(&self, instance_id: &str) -> Result<(), RuntimeError> {
        let proc_info = {
            let containers = self.containers.read().unwrap();
            containers
                .get(instance_id)
                .map(|c| c.container_id.clone())
                .ok_or_else(|| RuntimeError::InstanceNotFound(instance_id.into()))?
        };

        let tool = Self::find_tool()?;

        log::info!("Stopping container {}", instance_id);

        // Graceful stop (10 second timeout)
        let output = tokio::process::Command::new(&tool)
            .args(["stop", "-t", "10", &proc_info])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped())
            .output()
            .await
            .map_err(|e| RuntimeError::ProcessError(format!("failed to stop container: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            log::warn!("Container stop warning: {}", stderr.trim());
        }

        // Remove the container
        let _ = tokio::process::Command::new(&tool)
            .args(["rm", "-f", &proc_info])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .output()
            .await;

        // Clean up tracking
        self.containers.write().unwrap().remove(instance_id);
        self.network_manager.disconnect_all(instance_id).await;

        Ok(())
    }

    async fn status(&self, instance_id: &str) -> Result<InstanceState, RuntimeError> {
        let container_id = {
            let containers = self.containers.read().unwrap();
            match containers.get(instance_id) {
                Some(c) => c.container_id.clone(),
                None => return Ok(InstanceState::Stopped),
            }
        };

        let tool = match Self::find_tool() {
            Ok(t) => t,
            Err(_) => return Ok(InstanceState::Stopped),
        };

        let output = tokio::process::Command::new(&tool)
            .args(["inspect", "--format", "{{.State.Status}}", &container_id])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .output()
            .await
            .map_err(|e| {
                RuntimeError::ProcessError(format!("failed to inspect container: {}", e))
            })?;

        let status = String::from_utf8_lossy(&output.stdout).trim().to_string();
        match status.as_str() {
            "running" => Ok(InstanceState::Running),
            "created" | "restarting" => Ok(InstanceState::Starting),
            "paused" | "exited" | "dead" => Ok(InstanceState::Stopped),
            _ => Ok(InstanceState::Stopped),
        }
    }

    async fn health_check(&self, instance_id: &str) -> Result<bool, RuntimeError> {
        match self.status(instance_id).await? {
            InstanceState::Running => Ok(true),
            _ => Ok(false),
        }
    }

    async fn exec(
        &self,
        instance_id: &str,
        command: &[String],
    ) -> Result<ExecOutput, RuntimeError> {
        let container_id = {
            let containers = self.containers.read().unwrap();
            containers
                .get(instance_id)
                .map(|c| c.container_id.clone())
                .ok_or_else(|| RuntimeError::InstanceNotFound(instance_id.into()))?
        };

        let tool = Self::find_tool()?;

        let mut cmd = tokio::process::Command::new(&tool);
        cmd.arg("exec").arg(&container_id);
        for arg in command {
            cmd.arg(arg);
        }

        let output = cmd
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .output()
            .await
            .map_err(|e| RuntimeError::ProcessError(format!("exec failed: {}", e)))?;

        Ok(ExecOutput {
            exit_code: output.status.code().unwrap_or(-1),
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
        })
    }

    async fn logs(
        &self,
        instance_id: &str,
        tail: Option<usize>,
    ) -> Result<Vec<String>, RuntimeError> {
        let container_id = {
            let containers = self.containers.read().unwrap();
            containers
                .get(instance_id)
                .map(|c| c.container_id.clone())
                .ok_or_else(|| RuntimeError::InstanceNotFound(instance_id.into()))?
        };

        let tool = Self::find_tool()?;

        let mut cmd = tokio::process::Command::new(&tool);
        cmd.arg("logs");
        if let Some(n) = tail {
            cmd.arg("--tail").arg(n.to_string());
        }
        cmd.arg(&container_id);

        let output = cmd
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .output()
            .await
            .map_err(|e| RuntimeError::ProcessError(format!("logs failed: {}", e)))?;

        // Docker mixes stdout and stderr in logs
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let mut lines: Vec<String> = stdout.lines().map(String::from).collect();
        lines.extend(stderr.lines().map(String::from));
        Ok(lines)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AcceleratorBinding, AcceleratorKind, DatabaseEngine, InstanceId, PortMapping, RuntimeMode,
        VolumeMount, WorkloadKind,
    };
    use chrono::Utc;

    fn test_container_instance() -> InstanceInfo {
        InstanceInfo {
            id: InstanceId::from_string("container-001".into()),
            name: "test-nginx".into(),
            engine: DatabaseEngine::default(),
            workload: WorkloadKind::container("nginx:latest"),
            mode: RuntimeMode::Native,
            state: InstanceState::Starting,
            created_at: Utc::now(),
            pid: None,
            ports: vec![PortMapping {
                protocol: "http".into(),
                host_port: 8080,
                instance_port: 80,
            }],
            data_dir: "/tmp/container-test".into(),
            node_id: None,
            energy_port: None,
            accelerators: vec![],
            volumes: vec![],
            env_vars: HashMap::new(),
            labels: HashMap::new(),
        }
    }

    #[test]
    fn test_build_run_args_basic() {
        let instance = test_container_instance();
        let overrides = ServerOverrides::default();
        let args = ContainerBackend::build_run_args(&instance, &overrides);

        assert!(args.contains(&"run".to_string()));
        assert!(args.contains(&"-d".to_string()));
        assert!(args.contains(&"--name".to_string()));
        assert!(args.contains(&"test-nginx".to_string()));
        assert!(args.contains(&"-p".to_string()));
        assert!(args.contains(&"8080:80".to_string()));
        // Last arg should be the image
        assert_eq!(args.last().unwrap(), "nginx:latest");
    }

    #[test]
    fn test_build_run_args_with_volumes() {
        let mut instance = test_container_instance();
        instance.volumes = vec![
            VolumeMount {
                host_path: "/data".into(),
                container_path: "/app/data".into(),
                read_only: false,
            },
            VolumeMount {
                host_path: "/config".into(),
                container_path: "/etc/app".into(),
                read_only: true,
            },
        ];
        let args = ContainerBackend::build_run_args(&instance, &ServerOverrides::default());
        assert!(args.contains(&"/data:/app/data".to_string()));
        assert!(args.contains(&"/config:/etc/app:ro".to_string()));
    }

    #[test]
    fn test_build_run_args_with_env() {
        let mut instance = test_container_instance();
        instance.env_vars =
            HashMap::from([("DATABASE_URL".into(), "postgres://localhost/db".into())]);
        let args = ContainerBackend::build_run_args(&instance, &ServerOverrides::default());
        assert!(args.contains(&"-e".to_string()));
        assert!(args.contains(&"DATABASE_URL=postgres://localhost/db".to_string()));
    }

    #[test]
    fn test_build_run_args_with_labels() {
        let mut instance = test_container_instance();
        instance.labels = HashMap::from([("env".into(), "staging".into())]);
        let args = ContainerBackend::build_run_args(&instance, &ServerOverrides::default());
        assert!(args.contains(&"--label".to_string()));
        assert!(args.contains(&"env=staging".to_string()));
    }

    #[test]
    fn test_build_run_args_with_gpu() {
        let mut instance = test_container_instance();
        instance.accelerators = vec![AcceleratorBinding {
            kind: AcceleratorKind::GPU,
            device_id: Some("0".into()),
            memory_mb: None,
        }];
        let args = ContainerBackend::build_run_args(&instance, &ServerOverrides::default());
        assert!(args.contains(&"--gpus".to_string()));
        assert!(args.contains(&"device=0".to_string()));
    }

    #[test]
    fn test_build_run_args_with_gpu_all() {
        let mut instance = test_container_instance();
        instance.accelerators = vec![AcceleratorBinding {
            kind: AcceleratorKind::GPU,
            device_id: None,
            memory_mb: None,
        }];
        let args = ContainerBackend::build_run_args(&instance, &ServerOverrides::default());
        assert!(args.contains(&"--gpus".to_string()));
        assert!(args.contains(&"all".to_string()));
    }

    #[test]
    fn test_build_run_args_with_extra_args() {
        let instance = test_container_instance();
        let overrides = ServerOverrides {
            extra_args: vec!["--memory".into(), "512m".into()],
            ..Default::default()
        };
        let args = ContainerBackend::build_run_args(&instance, &overrides);
        assert!(args.contains(&"--memory".to_string()));
        assert!(args.contains(&"512m".to_string()));
    }

    #[test]
    fn test_container_backend_new() {
        let dir = tempfile::tempdir().unwrap();
        let backend = ContainerBackend::new(dir.path().join("images"));
        assert!(backend.containers.read().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_status_unknown_container() {
        let dir = tempfile::tempdir().unwrap();
        let backend = ContainerBackend::new(dir.path().join("images"));
        let state = backend.status("nonexistent").await.unwrap();
        assert_eq!(state, InstanceState::Stopped);
    }

    #[tokio::test]
    async fn test_health_check_unknown_container() {
        let dir = tempfile::tempdir().unwrap();
        let backend = ContainerBackend::new(dir.path().join("images"));
        let healthy = backend.health_check("nonexistent").await.unwrap();
        assert!(!healthy);
    }
}
