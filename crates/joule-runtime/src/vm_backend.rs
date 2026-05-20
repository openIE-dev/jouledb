//! VM isolation backend using InvisibleVM.
//!
//! Provides full hardware isolation via Apple Virtualization.framework (macOS)
//! or KVM (Linux). JouleDB runs inside a lightweight VM with shared filesystem
//! for data directory access.
//!
//! This backend interacts with InvisibleVM via its CLI binary (`invisible-vm`)
//! rather than compile-time library linkage, keeping the two projects decoupled.

use crate::{
    InstanceInfo, InstanceState, RuntimeConfig, RuntimeError, ServerOverrides,
    backend::RuntimeBackend,
};
use std::collections::HashMap;
use std::sync::RwLock;

/// VM backend — runs JouleDB inside a hardware-isolated virtual machine.
///
/// Uses the `invisible-vm` CLI to create, manage, and destroy VMs.
/// Each JouleDB instance gets its own VM with a VirtioFS mount
/// for the data directory.
pub struct VmBackend {
    /// Tracks VM instance IDs and their child process PIDs.
    vms: RwLock<HashMap<String, VmProcess>>,
}

struct VmProcess {
    pid: Option<u32>,
    vm_name: String,
}

impl VmBackend {
    pub fn new() -> Self {
        Self {
            vms: RwLock::new(HashMap::new()),
        }
    }

    /// Find the invisible-vm CLI binary (cross-platform).
    fn find_vm_binary() -> Result<String, RuntimeError> {
        let candidates = if cfg!(windows) {
            vec![
                "invisible-vm.exe",
                "../invisible/invisible-vm/target/release/invisible-vm.exe",
                "../invisible/invisible-vm/target/debug/invisible-vm.exe",
            ]
        } else {
            vec![
                "invisible-vm",
                "../invisible/invisible-vm/target/release/invisible-vm",
                "../invisible/invisible-vm/target/debug/invisible-vm",
            ]
        };

        for candidate in &candidates {
            let path = std::path::Path::new(candidate);
            if path.exists() {
                return Ok(candidate.to_string());
            }
        }

        // Try which (Unix) or where (Windows)
        let (cmd, arg) = if cfg!(windows) {
            ("where", "invisible-vm.exe")
        } else {
            ("which", "invisible-vm")
        };

        if let Ok(output) = std::process::Command::new(cmd).arg(arg).output() {
            if output.status.success() {
                let path = String::from_utf8_lossy(&output.stdout)
                    .lines()
                    .next()
                    .unwrap_or("")
                    .trim()
                    .to_string();
                if !path.is_empty() {
                    return Ok(path);
                }
            }
        }

        Err(RuntimeError::VMError(
            "invisible-vm binary not found. Install InvisibleVM or add it to PATH.".into(),
        ))
    }

    /// Build CLI arguments for creating a VM.
    fn build_create_args(config: &RuntimeConfig, instance: &InstanceInfo) -> Vec<String> {
        let mut args = vec![
            "create".into(),
            "--name".into(),
            instance.name.clone(),
            "--memory".into(),
            config.vm_memory_mb.to_string(),
            "--cpus".into(),
            config.vm_cpu_cores.to_string(),
        ];

        // Mount data directory via shared filesystem
        args.push("--share".into());
        args.push(format!("{}:/data", instance.data_dir));

        // Kernel path
        if let Some(kernel) = &config.vm_kernel_path {
            args.push("--kernel".into());
            args.push(kernel.clone());
        }

        // Disk image
        if let Some(disk) = &config.vm_disk_image {
            args.push("--disk".into());
            args.push(disk.clone());
        }

        // Tell the VM which database binary to run inside
        args.push("--exec".into());
        args.push(instance.engine.binary_name().into());

        args
    }
}

impl Default for VmBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl RuntimeBackend for VmBackend {
    async fn start(
        &self,
        config: &RuntimeConfig,
        instance: &InstanceInfo,
        _overrides: &ServerOverrides,
    ) -> Result<(), RuntimeError> {
        let binary = Self::find_vm_binary()?;
        let args = Self::build_create_args(config, instance);

        log::info!(
            "Starting VM instance '{}' — {}MB RAM, {} cores",
            instance.name,
            config.vm_memory_mb,
            config.vm_cpu_cores,
        );

        // Ensure data directory exists
        std::fs::create_dir_all(&instance.data_dir)?;

        // Spawn the VM via invisible-vm CLI
        let child = std::process::Command::new(&binary)
            .args(&args)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| RuntimeError::VMError(format!("failed to spawn {}: {}", binary, e)))?;

        let pid = child.id();
        let mut vms = self.vms.write().unwrap();
        vms.insert(
            instance.id.0.clone(),
            VmProcess {
                pid: Some(pid),
                vm_name: instance.name.clone(),
            },
        );

        log::info!("VM instance '{}' created (PID {})", instance.name, pid);
        Ok(())
    }

    async fn stop(&self, instance_id: &str) -> Result<(), RuntimeError> {
        let vm = {
            let mut vms = self.vms.write().unwrap();
            vms.remove(instance_id)
                .ok_or_else(|| RuntimeError::InstanceNotFound(instance_id.into()))?
        };

        log::info!("Stopping VM instance {} ({})", instance_id, vm.vm_name);

        // Try to stop via invisible-vm CLI
        if let Ok(binary) = Self::find_vm_binary() {
            let _ = std::process::Command::new(&binary)
                .args(["stop", "--name", &vm.vm_name])
                .output();
        }

        // Also kill the process if it's still running
        if let Some(pid) = vm.pid {
            #[cfg(unix)]
            unsafe {
                libc::kill(pid as i32, libc::SIGTERM);
            }

            #[cfg(not(unix))]
            {
                let _ = std::process::Command::new("taskkill")
                    .args(["/PID", &pid.to_string(), "/F"])
                    .output();
            }
        }

        Ok(())
    }

    async fn status(&self, instance_id: &str) -> Result<InstanceState, RuntimeError> {
        let vms = self.vms.read().unwrap();
        match vms.get(instance_id) {
            Some(vm) => {
                if let Some(pid) = vm.pid {
                    if is_process_running(pid) {
                        return Ok(InstanceState::Running);
                    }
                }
                Ok(InstanceState::Stopped)
            }
            None => Ok(InstanceState::Stopped),
        }
    }

    async fn health_check(&self, instance_id: &str) -> Result<bool, RuntimeError> {
        match self.status(instance_id).await? {
            InstanceState::Running => Ok(true),
            _ => Ok(false),
        }
    }
}

/// Check if a process is running by PID (cross-platform).
fn is_process_running(pid: u32) -> bool {
    #[cfg(unix)]
    {
        unsafe { libc::kill(pid as i32, 0) == 0 }
    }
    #[cfg(not(unix))]
    {
        std::process::Command::new("tasklist")
            .args(["/FI", &format!("PID eq {}", pid)])
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).contains(&pid.to_string()))
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{InstanceId, PortMapping, RuntimeMode};
    use chrono::Utc;

    fn test_instance() -> InstanceInfo {
        InstanceInfo {
            id: InstanceId::from_string("test-vm-001".into()),
            name: "test-vm-db".into(),
            engine: Default::default(),
            mode: RuntimeMode::VM,
            state: InstanceState::Starting,
            created_at: Utc::now(),
            pid: None,
            ports: vec![],
            data_dir: "/tmp/jouledb-test-vm".into(),
            node_id: None,
            energy_port: None,
        }
    }

    #[test]
    fn test_build_create_args() {
        let config = RuntimeConfig::vm();
        let instance = test_instance();
        let args = VmBackend::build_create_args(&config, &instance);

        assert!(args.contains(&"create".to_string()));
        assert!(args.contains(&"--name".to_string()));
        assert!(args.contains(&"test-vm-db".to_string()));
        assert!(args.contains(&"--memory".to_string()));
        assert!(args.contains(&"4096".to_string()));
        assert!(args.contains(&"--cpus".to_string()));
        assert!(args.contains(&"4".to_string()));
        assert!(args.contains(&"--share".to_string()));
    }

    #[test]
    fn test_build_create_args_custom() {
        let config = RuntimeConfig {
            mode: RuntimeMode::VM,
            vm_memory_mb: 8192,
            vm_cpu_cores: 8,
            vm_kernel_path: Some("/path/to/kernel".into()),
            vm_disk_image: Some("/path/to/disk.img".into()),
            ..Default::default()
        };
        let instance = test_instance();
        let args = VmBackend::build_create_args(&config, &instance);

        assert!(args.contains(&"8192".to_string()));
        assert!(args.contains(&"8".to_string()));
        assert!(args.contains(&"--kernel".to_string()));
        assert!(args.contains(&"--disk".to_string()));
    }

    #[tokio::test]
    async fn test_vm_backend_status_missing() {
        let backend = VmBackend::new();
        let state = backend.status("nonexistent").await.unwrap();
        assert_eq!(state, InstanceState::Stopped);
    }

    #[tokio::test]
    async fn test_vm_backend_health_missing() {
        let backend = VmBackend::new();
        let healthy = backend.health_check("nonexistent").await.unwrap();
        assert!(!healthy);
    }

    #[tokio::test]
    async fn test_vm_backend_stop_not_found() {
        let backend = VmBackend::new();
        assert!(backend.stop("nonexistent").await.is_err());
    }
}
