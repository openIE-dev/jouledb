//! WASM isolation backend using InvisibleVM's container runtime.
//!
//! Provides sandboxed execution via Wasmtime with resource limits (memory caps,
//! fuel/timeout). Best suited for embedded/edge scenarios where the query engine
//! runs in a WASM sandbox without full network server capabilities.
//!
//! This backend interacts with InvisibleVM's container-runtime via its CLI
//! rather than compile-time library linkage, keeping the two projects decoupled.

use crate::{
    InstanceInfo, InstanceState, RuntimeConfig, RuntimeError, ServerOverrides,
    backend::RuntimeBackend,
};
use std::collections::HashMap;
use std::sync::RwLock;

/// WASM backend — runs JouleDB query engine in a Wasmtime sandbox.
///
/// Uses the `invisible-vm` CLI's container subcommands to run WASM modules
/// with configurable resource limits.
pub struct WasmBackend {
    /// Tracks container instances and their process PIDs.
    containers: RwLock<HashMap<String, WasmProcess>>,
}

struct WasmProcess {
    pid: Option<u32>,
    container_name: String,
}

impl WasmBackend {
    pub fn new() -> Result<Self, RuntimeError> {
        Ok(Self {
            containers: RwLock::new(HashMap::new()),
        })
    }

    /// Find the invisible-vm CLI binary for WASM execution (cross-platform).
    fn find_wasm_binary() -> Result<String, RuntimeError> {
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

        Err(RuntimeError::WasmError(
            "invisible-vm binary not found. Install InvisibleVM or add it to PATH.".into(),
        ))
    }

    /// Build CLI arguments for running a WASM container.
    fn build_run_args(config: &RuntimeConfig, instance: &InstanceInfo) -> Vec<String> {
        let mut args = vec![
            "container".into(),
            "run".into(),
            "--name".into(),
            instance.name.clone(),
        ];

        // Resource limits
        args.push("--max-memory".into());
        args.push(format!("{}M", config.wasm_max_memory_bytes / (1024 * 1024)));

        args.push("--fuel".into());
        args.push(config.wasm_fuel.to_string());

        args.push("--timeout".into());
        args.push(format!("{}s", config.wasm_timeout_secs));

        // Mount data directory
        args.push("--volume".into());
        args.push(format!("{}:/data", instance.data_dir));

        // WASM module path (JouleDB browser build)
        args.push("joule-db-browser/pkg/joule_db_browser_bg.wasm".into());

        args
    }
}

impl Default for WasmBackend {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| Self {
            containers: RwLock::new(HashMap::new()),
        })
    }
}

impl RuntimeBackend for WasmBackend {
    async fn start(
        &self,
        config: &RuntimeConfig,
        instance: &InstanceInfo,
        _overrides: &ServerOverrides,
    ) -> Result<(), RuntimeError> {
        // Only JouleDB and SQLite support WASM mode
        let spec = crate::catalog::get_spec(&instance.engine);
        if !spec.supports_wasm {
            return Err(RuntimeError::WasmError(format!(
                "{} does not support WASM isolation mode. Use native or vm mode instead.",
                spec.display_name
            )));
        }

        let binary = Self::find_wasm_binary()?;
        let args = Self::build_run_args(config, instance);

        log::info!(
            "Starting WASM instance '{}' — {}MB max memory, {} fuel, {}s timeout",
            instance.name,
            config.wasm_max_memory_bytes / (1024 * 1024),
            config.wasm_fuel,
            config.wasm_timeout_secs
        );

        // Ensure data directory exists
        std::fs::create_dir_all(&instance.data_dir)?;

        // Check WASM module exists
        let wasm_path = std::path::Path::new("joule-db-browser/pkg/joule_db_browser_bg.wasm");
        if !wasm_path.exists() {
            return Err(RuntimeError::WasmError(format!(
                "JouleDB WASM module not found at {}. Build with `wasm-pack build joule-db-browser`.",
                wasm_path.display()
            )));
        }

        let child = std::process::Command::new(&binary)
            .args(&args)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| RuntimeError::WasmError(format!("failed to spawn {}: {}", binary, e)))?;

        let pid = child.id();
        let mut containers = self.containers.write().unwrap();
        containers.insert(
            instance.id.0.clone(),
            WasmProcess {
                pid: Some(pid),
                container_name: instance.name.clone(),
            },
        );

        log::info!("WASM instance '{}' started (PID {})", instance.name, pid);
        Ok(())
    }

    async fn stop(&self, instance_id: &str) -> Result<(), RuntimeError> {
        let container = {
            let mut containers = self.containers.write().unwrap();
            containers
                .remove(instance_id)
                .ok_or_else(|| RuntimeError::InstanceNotFound(instance_id.into()))?
        };

        log::info!(
            "Stopping WASM instance {} ({})",
            instance_id,
            container.container_name
        );

        if let Some(pid) = container.pid {
            #[cfg(unix)]
            unsafe {
                libc::kill(pid as i32, libc::SIGTERM);
            }

            #[cfg(not(unix))]
            {
                let _ = std::process::Command::new("taskkill")
                    .args(&["/PID", &pid.to_string(), "/F"])
                    .output();
            }
        }

        Ok(())
    }

    async fn status(&self, instance_id: &str) -> Result<InstanceState, RuntimeError> {
        let containers = self.containers.read().unwrap();
        match containers.get(instance_id) {
            Some(c) => {
                if let Some(pid) = c.pid {
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
            id: InstanceId::from_string("test-wasm-001".into()),
            name: "test-wasm-db".into(),
            engine: Default::default(),
            mode: RuntimeMode::WASM,
            state: InstanceState::Starting,
            created_at: Utc::now(),
            pid: None,
            ports: vec![],
            data_dir: "/tmp/jouledb-test-wasm".into(),
            node_id: None,
            energy_port: None,
        }
    }

    #[test]
    fn test_build_run_args() {
        let config = RuntimeConfig::wasm();
        let instance = test_instance();
        let args = WasmBackend::build_run_args(&config, &instance);

        assert!(args.contains(&"container".to_string()));
        assert!(args.contains(&"run".to_string()));
        assert!(args.contains(&"--name".to_string()));
        assert!(args.contains(&"--max-memory".to_string()));
        assert!(args.contains(&"--fuel".to_string()));
        assert!(args.contains(&"--timeout".to_string()));
        assert!(args.contains(&"--volume".to_string()));
    }

    #[test]
    fn test_build_run_args_custom() {
        let config = RuntimeConfig {
            mode: RuntimeMode::WASM,
            wasm_max_memory_bytes: 512 * 1024 * 1024,
            wasm_fuel: 2_000_000_000,
            wasm_timeout_secs: 60,
            ..Default::default()
        };
        let instance = test_instance();
        let args = WasmBackend::build_run_args(&config, &instance);

        assert!(args.contains(&"512M".to_string()));
        assert!(args.contains(&"2000000000".to_string()));
        assert!(args.contains(&"60s".to_string()));
    }

    #[tokio::test]
    async fn test_wasm_backend_status_missing() {
        let backend = WasmBackend::new().unwrap();
        let state = backend.status("nonexistent").await.unwrap();
        assert_eq!(state, InstanceState::Stopped);
    }

    #[tokio::test]
    async fn test_wasm_backend_health_missing() {
        let backend = WasmBackend::new().unwrap();
        let healthy = backend.health_check("nonexistent").await.unwrap();
        assert!(!healthy);
    }

    #[tokio::test]
    async fn test_wasm_backend_stop_not_found() {
        let backend = WasmBackend::new().unwrap();
        assert!(backend.stop("nonexistent").await.is_err());
    }
}
