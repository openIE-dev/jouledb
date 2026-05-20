//! Client for communicating with the JouleDB daemon over Unix socket.
//!
//! Used by CLI commands to delegate instance management to the persistent daemon
//! process instead of creating ephemeral RuntimeManagers.

use crate::daemon::{
    DaemonRequest, DaemonResponse, default_pid_path, default_socket_path, is_daemon_running,
    read_message, write_message,
};
use crate::{
    AcceleratorBinding, InstanceInfo, RuntimeError, ServerOverrides, VolumeMount, WorkloadKind,
    accelerator::AcceleratorDevice, backend::ExecOutput, image::ImageInfo,
};
use std::collections::HashMap;
use std::path::PathBuf;

/// Client that talks to the daemon over a Unix domain socket.
pub struct DaemonClient {
    socket_path: PathBuf,
    pid_path: PathBuf,
}

/// Status information from the daemon.
#[derive(Debug, Clone)]
pub struct DaemonStatus {
    pub uptime_secs: u64,
    pub instance_count: usize,
    pub total_energy_joules: f64,
}

impl Default for DaemonClient {
    fn default() -> Self {
        Self {
            socket_path: default_socket_path(),
            pid_path: default_pid_path(),
        }
    }
}

impl DaemonClient {
    /// Create a client pointing at a specific socket path.
    pub fn new(socket_path: PathBuf, pid_path: PathBuf) -> Self {
        Self {
            socket_path,
            pid_path,
        }
    }

    /// Check if the daemon is currently running.
    pub fn is_daemon_running(&self) -> bool {
        is_daemon_running(&self.pid_path)
    }

    /// Send a request and receive a response.
    async fn send(&self, request: DaemonRequest) -> Result<DaemonResponse, RuntimeError> {
        let stream = tokio::net::UnixStream::connect(&self.socket_path)
            .await
            .map_err(|e| {
                RuntimeError::ProcessError(format!(
                    "cannot connect to daemon at {:?}: {}",
                    self.socket_path, e
                ))
            })?;

        let (mut reader, mut writer) = stream.into_split();

        write_message(&mut writer, &request).await.map_err(|e| {
            RuntimeError::ProcessError(format!("failed to send request to daemon: {}", e))
        })?;

        let response: DaemonResponse = read_message(&mut reader).await.map_err(|e| {
            RuntimeError::ProcessError(format!("failed to read daemon response: {}", e))
        })?;

        Ok(response)
    }

    /// Start a new database instance via the daemon.
    pub async fn start_instance(
        &self,
        name: String,
        engine: String,
        overrides: ServerOverrides,
    ) -> Result<String, RuntimeError> {
        let response = self
            .send(DaemonRequest::StartInstance {
                name,
                engine,
                overrides,
            })
            .await?;

        match response {
            DaemonResponse::InstanceStarted { instance_id } => Ok(instance_id),
            DaemonResponse::Error { message } => Err(RuntimeError::ProcessError(message)),
            other => Err(RuntimeError::ProcessError(format!(
                "unexpected response: {:?}",
                other
            ))),
        }
    }

    /// Stop an instance via the daemon.
    pub async fn stop_instance(&self, instance_id: &str) -> Result<(), RuntimeError> {
        let response = self
            .send(DaemonRequest::StopInstance {
                instance_id: instance_id.to_string(),
            })
            .await?;

        match response {
            DaemonResponse::Ok { .. } => Ok(()),
            DaemonResponse::Error { message } => Err(RuntimeError::ProcessError(message)),
            other => Err(RuntimeError::ProcessError(format!(
                "unexpected response: {:?}",
                other
            ))),
        }
    }

    /// List all instances managed by the daemon.
    pub async fn list_instances(&self) -> Result<Vec<InstanceInfo>, RuntimeError> {
        let response = self.send(DaemonRequest::ListInstances).await?;

        match response {
            DaemonResponse::InstanceList { instances } => Ok(instances),
            DaemonResponse::Error { message } => Err(RuntimeError::ProcessError(message)),
            other => Err(RuntimeError::ProcessError(format!(
                "unexpected response: {:?}",
                other
            ))),
        }
    }

    /// Check health of a specific instance.
    pub async fn health(&self, instance_id: &str) -> Result<bool, RuntimeError> {
        let response = self
            .send(DaemonRequest::InstanceHealth {
                instance_id: instance_id.to_string(),
            })
            .await?;

        match response {
            DaemonResponse::Health { healthy } => Ok(healthy),
            DaemonResponse::Error { message } => Err(RuntimeError::ProcessError(message)),
            other => Err(RuntimeError::ProcessError(format!(
                "unexpected response: {:?}",
                other
            ))),
        }
    }

    /// Get daemon status.
    pub async fn status(&self) -> Result<DaemonStatus, RuntimeError> {
        let response = self.send(DaemonRequest::DaemonStatus).await?;

        match response {
            DaemonResponse::Status {
                uptime_secs,
                instance_count,
                total_energy_joules,
            } => Ok(DaemonStatus {
                uptime_secs,
                instance_count,
                total_energy_joules,
            }),
            DaemonResponse::Error { message } => Err(RuntimeError::ProcessError(message)),
            other => Err(RuntimeError::ProcessError(format!(
                "unexpected response: {:?}",
                other
            ))),
        }
    }

    /// Request graceful daemon shutdown.
    pub async fn shutdown(&self) -> Result<(), RuntimeError> {
        let response = self.send(DaemonRequest::Shutdown).await?;

        match response {
            DaemonResponse::Ok { .. } => Ok(()),
            DaemonResponse::Error { message } => Err(RuntimeError::ProcessError(message)),
            other => Err(RuntimeError::ProcessError(format!(
                "unexpected response: {:?}",
                other
            ))),
        }
    }

    // --- Container / workload operations ---

    /// Run any workload via the daemon.
    pub async fn run_workload(
        &self,
        name: String,
        workload: WorkloadKind,
        overrides: ServerOverrides,
        accelerators: Vec<AcceleratorBinding>,
        volumes: Vec<VolumeMount>,
        env_vars: HashMap<String, String>,
        labels: HashMap<String, String>,
    ) -> Result<String, RuntimeError> {
        let response = self
            .send(DaemonRequest::RunWorkload {
                name,
                workload,
                overrides,
                accelerators,
                volumes,
                env_vars,
                labels,
            })
            .await?;

        match response {
            DaemonResponse::InstanceStarted { instance_id } => Ok(instance_id),
            DaemonResponse::Error { message } => Err(RuntimeError::ProcessError(message)),
            other => Err(RuntimeError::ProcessError(format!(
                "unexpected response: {:?}",
                other
            ))),
        }
    }

    /// Pull a container image via the daemon.
    pub async fn pull_image(&self, image: &str) -> Result<String, RuntimeError> {
        let response = self
            .send(DaemonRequest::PullImage {
                image: image.to_string(),
            })
            .await?;

        match response {
            DaemonResponse::Ok { message } => Ok(message),
            DaemonResponse::Error { message } => Err(RuntimeError::ProcessError(message)),
            other => Err(RuntimeError::ProcessError(format!(
                "unexpected response: {:?}",
                other
            ))),
        }
    }

    /// List cached images via the daemon.
    pub async fn list_images(&self) -> Result<Vec<ImageInfo>, RuntimeError> {
        let response = self.send(DaemonRequest::ListImages).await?;

        match response {
            DaemonResponse::ImageList { images } => Ok(images),
            DaemonResponse::Error { message } => Err(RuntimeError::ProcessError(message)),
            other => Err(RuntimeError::ProcessError(format!(
                "unexpected response: {:?}",
                other
            ))),
        }
    }

    /// Remove a cached image via the daemon.
    pub async fn remove_image(&self, image: &str) -> Result<(), RuntimeError> {
        let response = self
            .send(DaemonRequest::RemoveImage {
                image: image.to_string(),
            })
            .await?;

        match response {
            DaemonResponse::Ok { .. } => Ok(()),
            DaemonResponse::Error { message } => Err(RuntimeError::ProcessError(message)),
            other => Err(RuntimeError::ProcessError(format!(
                "unexpected response: {:?}",
                other
            ))),
        }
    }

    /// Execute a command inside a running instance.
    pub async fn exec_in_instance(
        &self,
        instance_id: &str,
        command: Vec<String>,
    ) -> Result<ExecOutput, RuntimeError> {
        let response = self
            .send(DaemonRequest::ExecInInstance {
                instance_id: instance_id.to_string(),
                command,
            })
            .await?;

        match response {
            DaemonResponse::ExecResult {
                exit_code,
                stdout,
                stderr,
            } => Ok(ExecOutput {
                exit_code,
                stdout,
                stderr,
            }),
            DaemonResponse::Error { message } => Err(RuntimeError::ProcessError(message)),
            other => Err(RuntimeError::ProcessError(format!(
                "unexpected response: {:?}",
                other
            ))),
        }
    }

    /// Retrieve logs from an instance.
    pub async fn instance_logs(
        &self,
        instance_id: &str,
        tail: Option<usize>,
    ) -> Result<Vec<String>, RuntimeError> {
        let response = self
            .send(DaemonRequest::InstanceLogs {
                instance_id: instance_id.to_string(),
                tail,
            })
            .await?;

        match response {
            DaemonResponse::LogLines { lines } => Ok(lines),
            DaemonResponse::Error { message } => Err(RuntimeError::ProcessError(message)),
            other => Err(RuntimeError::ProcessError(format!(
                "unexpected response: {:?}",
                other
            ))),
        }
    }

    /// Query available hardware accelerators.
    pub async fn accelerator_status(&self) -> Result<Vec<AcceleratorDevice>, RuntimeError> {
        let response = self.send(DaemonRequest::AcceleratorStatus).await?;

        match response {
            DaemonResponse::AcceleratorList { devices } => Ok(devices),
            DaemonResponse::Error { message } => Err(RuntimeError::ProcessError(message)),
            other => Err(RuntimeError::ProcessError(format!(
                "unexpected response: {:?}",
                other
            ))),
        }
    }

    /// Inspect a specific instance (full details).
    pub async fn inspect_instance(&self, instance_id: &str) -> Result<InstanceInfo, RuntimeError> {
        let response = self
            .send(DaemonRequest::InspectInstance {
                instance_id: instance_id.to_string(),
            })
            .await?;

        match response {
            DaemonResponse::InstanceDetail { info } => Ok(info),
            DaemonResponse::Error { message } => Err(RuntimeError::ProcessError(message)),
            other => Err(RuntimeError::ProcessError(format!(
                "unexpected response: {:?}",
                other
            ))),
        }
    }

    /// Remove a stopped instance.
    pub async fn remove_instance(
        &self,
        instance_id: &str,
        force: bool,
    ) -> Result<(), RuntimeError> {
        let response = self
            .send(DaemonRequest::RemoveInstance {
                instance_id: instance_id.to_string(),
                force,
            })
            .await?;

        match response {
            DaemonResponse::Ok { .. } => Ok(()),
            DaemonResponse::Error { message } => Err(RuntimeError::ProcessError(message)),
            other => Err(RuntimeError::ProcessError(format!(
                "unexpected response: {:?}",
                other
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_default() {
        let client = DaemonClient::default();
        assert!(client.socket_path.to_string_lossy().contains("daemon.sock"));
        assert!(client.pid_path.to_string_lossy().contains("daemon.pid"));
    }

    #[test]
    fn test_client_not_running() {
        let tmp = tempfile::tempdir().unwrap();
        let client = DaemonClient::new(
            tmp.path().join("nonexistent.sock"),
            tmp.path().join("nonexistent.pid"),
        );
        assert!(!client.is_daemon_running());
    }

    #[tokio::test]
    async fn test_client_connect_error() {
        let tmp = tempfile::tempdir().unwrap();
        let client = DaemonClient::new(
            tmp.path().join("nonexistent.sock"),
            tmp.path().join("nonexistent.pid"),
        );

        // Should fail to connect
        let result = client.list_instances().await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("cannot connect to daemon"));
    }

    #[tokio::test]
    async fn test_client_full_roundtrip() {
        use crate::RuntimeConfig;
        use crate::daemon::{DaemonConfig, DaemonCore};

        let tmp = tempfile::tempdir().unwrap();
        let daemon_dir = tmp.path().to_path_buf();
        let socket_path = daemon_dir.join("test.sock");
        let pid_path = daemon_dir.join("test.pid");

        let config = DaemonConfig {
            runtime_config: RuntimeConfig::default(),
            daemon_dir: daemon_dir.clone(),
            socket_path: Some(socket_path.clone()),
            pid_path: Some(pid_path.clone()),
            dashboard_port: None,
        };

        let daemon = DaemonCore::new(config).unwrap();

        // Client + daemon in parallel
        let client_handle = {
            let socket_path = socket_path.clone();
            let pid_path = pid_path.clone();
            tokio::spawn(async move {
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;

                let client = DaemonClient::new(socket_path, pid_path);

                // List instances (should be empty)
                let instances = client.list_instances().await.unwrap();
                assert!(instances.is_empty());

                // Get status
                let status = client.status().await.unwrap();
                assert_eq!(status.instance_count, 0);

                // Shutdown
                client.shutdown().await.unwrap();
            })
        };

        let _ = daemon.run().await;
        client_handle.await.unwrap();
    }
}
