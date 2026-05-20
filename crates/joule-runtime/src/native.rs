use crate::{
    DatabaseEngine, InstanceInfo, InstanceState, RuntimeConfig, RuntimeError, ServerOverrides,
    backend::RuntimeBackend,
    catalog::{self, DatabaseSpec},
};
use std::collections::HashMap;
use std::sync::RwLock;

/// Native backend — runs any database as a bare-metal process.
///
/// This is the zero-overhead path: the database binary is spawned directly
/// as a child process with no isolation layer. Works for JouleDB, Postgres,
/// MySQL, Redis, MongoDB, or any custom engine.
pub struct NativeBackend {
    /// Tracks running process info keyed by instance ID.
    processes: RwLock<HashMap<String, ProcessInfo>>,
}

/// Tracks a running database process.
struct ProcessInfo {
    pid: u32,
    engine: DatabaseEngine,
    health_cmd: Option<Vec<String>>,
    port: u16,
}

impl NativeBackend {
    pub fn new() -> Self {
        Self {
            processes: RwLock::new(HashMap::new()),
        }
    }

    /// Find a database binary using the catalog spec + PATH search.
    fn find_binary(spec: &DatabaseSpec) -> Result<String, RuntimeError> {
        // Try spec candidates first (with platform .exe suffix)
        for candidate in spec.binary_candidates {
            let name = if cfg!(windows) && !candidate.ends_with(".exe") {
                format!("{}.exe", candidate)
            } else {
                candidate.to_string()
            };
            if which_exists(&name) {
                return Ok(name);
            }
        }

        // Fall back to bare binary name on PATH
        let bare = if cfg!(windows) {
            format!("{}.exe", spec.display_name.to_lowercase())
        } else {
            spec.binary_candidates
                .first()
                .map(|s| s.to_string())
                .unwrap_or_default()
        };
        if !bare.is_empty() && which_exists(&bare) {
            return Ok(bare);
        }

        Err(RuntimeError::ProcessError(format!(
            "{} binary not found. Install {} or add it to PATH.",
            spec.display_name, spec.display_name
        )))
    }

    /// Build command-line arguments using the engine's catalog spec.
    fn build_args(
        spec: &DatabaseSpec,
        instance: &InstanceInfo,
        overrides: &ServerOverrides,
    ) -> Vec<String> {
        let mut args = Vec::new();

        // Data directory
        if let Some(flag) = spec.data_dir_flag {
            args.push(flag.into());
            args.push(instance.data_dir.clone());
        }

        // Port
        if let Some(flag) = spec.port_flag {
            let port = overrides
                .engine_port
                .or(overrides.http_port) // backward compat for JouleDB
                .unwrap_or(spec.default_port);
            if port > 0 {
                args.push(flag.into());
                args.push(port.to_string());
            }
        }

        // Default args from catalog
        for arg in spec.default_args {
            args.push((*arg).into());
        }

        // Extra args passthrough
        for arg in &overrides.extra_args {
            args.push(arg.clone());
        }

        args
    }

    /// Build the health check command with port substitution.
    fn build_health_cmd(spec: &DatabaseSpec, port: u16) -> Option<Vec<String>> {
        spec.health_check_cmd.map(|cmd| {
            cmd.iter()
                .map(|part| part.replace("{port}", &port.to_string()))
                .collect()
        })
    }
}

impl Default for NativeBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl RuntimeBackend for NativeBackend {
    async fn start(
        &self,
        _config: &RuntimeConfig,
        instance: &InstanceInfo,
        overrides: &ServerOverrides,
    ) -> Result<(), RuntimeError> {
        let spec = catalog::get_spec(&instance.engine);
        let binary = Self::find_binary(&spec)?;
        let args = Self::build_args(&spec, instance, overrides);
        let port = overrides
            .engine_port
            .or(overrides.http_port)
            .unwrap_or(spec.default_port);

        log::info!(
            "Starting native {} instance '{}' with binary: {} {:?}",
            spec.display_name,
            instance.name,
            binary,
            args
        );

        // Ensure data directory exists
        std::fs::create_dir_all(&instance.data_dir)?;

        // Spawn the server process as a daemon
        let mut cmd = std::process::Command::new(&binary);
        cmd.args(&args)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());

        // Inject instance env vars (includes accelerator passthrough)
        for (key, value) in &instance.env_vars {
            cmd.env(key, value);
        }

        let child = cmd.spawn().map_err(|e| {
            RuntimeError::ProcessError(format!("failed to spawn {}: {}", binary, e))
        })?;

        let pid = child.id();
        log::info!(
            "{} instance '{}' started with PID {}",
            spec.display_name,
            instance.name,
            pid
        );

        let health_cmd = Self::build_health_cmd(&spec, port);
        let mut processes = self.processes.write().unwrap();
        processes.insert(
            instance.id.0.clone(),
            ProcessInfo {
                pid,
                engine: instance.engine.clone(),
                health_cmd,
                port,
            },
        );

        Ok(())
    }

    async fn stop(&self, instance_id: &str) -> Result<(), RuntimeError> {
        let proc_info = {
            let mut processes = self.processes.write().unwrap();
            processes
                .remove(instance_id)
                .ok_or_else(|| RuntimeError::InstanceNotFound(instance_id.into()))?
        };

        log::info!("Stopping instance {} (PID {})", instance_id, proc_info.pid);

        // Send SIGTERM for graceful shutdown
        #[cfg(unix)]
        {
            unsafe {
                libc::kill(proc_info.pid as i32, libc::SIGTERM);
            }

            // Wait up to 10 seconds for graceful shutdown
            for _ in 0..100 {
                if !is_process_running(proc_info.pid) {
                    log::info!("Instance {} stopped gracefully", instance_id);
                    return Ok(());
                }
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            }

            // Force kill if still running
            log::warn!(
                "Instance {} did not stop gracefully, sending SIGKILL",
                instance_id
            );
            unsafe {
                libc::kill(proc_info.pid as i32, libc::SIGKILL);
            }
        }

        #[cfg(not(unix))]
        {
            let _ = std::process::Command::new("taskkill")
                .args(&["/PID", &proc_info.pid.to_string(), "/F"])
                .output();
        }

        Ok(())
    }

    async fn status(&self, instance_id: &str) -> Result<InstanceState, RuntimeError> {
        let processes = self.processes.read().unwrap();
        match processes.get(instance_id) {
            Some(info) => {
                if is_process_running(info.pid) {
                    Ok(InstanceState::Running)
                } else {
                    Ok(InstanceState::Stopped)
                }
            }
            None => Ok(InstanceState::Stopped),
        }
    }

    async fn health_check(&self, instance_id: &str) -> Result<bool, RuntimeError> {
        let processes = self.processes.read().unwrap();
        match processes.get(instance_id) {
            Some(info) => {
                // First check: is the process alive?
                if !is_process_running(info.pid) {
                    return Ok(false);
                }
                // Second check: run engine-specific health command if available
                if let Some(cmd) = &info.health_cmd {
                    if cmd.is_empty() {
                        return Ok(true);
                    }
                    let binary = if cfg!(windows) && !cmd[0].ends_with(".exe") {
                        format!("{}.exe", cmd[0])
                    } else {
                        cmd[0].clone()
                    };
                    match std::process::Command::new(&binary)
                        .args(&cmd[1..])
                        .stdout(std::process::Stdio::null())
                        .stderr(std::process::Stdio::null())
                        .status()
                    {
                        Ok(status) => Ok(status.success()),
                        Err(_) => Ok(true), // health binary not found → fall back to PID check
                    }
                } else {
                    Ok(true)
                }
            }
            None => Ok(false),
        }
    }
}

/// Check if a process is running by PID (public for energy sidecar health checks).
pub fn is_process_alive(pid: u32) -> bool {
    is_process_running(pid)
}

/// Check if a process is running by PID.
fn is_process_running(pid: u32) -> bool {
    #[cfg(unix)]
    {
        unsafe { libc::kill(pid as i32, 0) == 0 }
    }
    #[cfg(not(unix))]
    {
        std::process::Command::new("tasklist")
            .args(&["/FI", &format!("PID eq {}", pid)])
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).contains(&pid.to_string()))
            .unwrap_or(false)
    }
}

/// Check if a binary exists on PATH (cross-platform).
pub(crate) fn which_exists(name: &str) -> bool {
    // Check if it's a path
    if name.contains('/') || name.contains('\\') {
        return std::path::Path::new(name).exists();
    }

    // Check PATH using std::env::split_paths (handles `:` on Unix, `;` on Windows)
    if let Ok(path) = std::env::var("PATH") {
        for dir in std::env::split_paths(&path) {
            let full = dir.join(name);
            if full.exists() {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DatabaseEngine, InstanceId, InstanceState, PortMapping, RuntimeMode, WorkloadKind,
    };
    use chrono::Utc;
    use std::collections::HashMap;

    fn test_instance() -> InstanceInfo {
        InstanceInfo {
            id: InstanceId::from_string("test-native-001".into()),
            name: "test-db".into(),
            engine: DatabaseEngine::JouleDB,
            workload: WorkloadKind::database(DatabaseEngine::JouleDB),
            mode: RuntimeMode::Native,
            state: InstanceState::Starting,
            created_at: Utc::now(),
            pid: None,
            ports: vec![],
            data_dir: "/tmp/jouledb-test-native".into(),
            node_id: None,
            energy_port: None,
            accelerators: vec![],
            volumes: vec![],
            env_vars: HashMap::new(),
            labels: HashMap::new(),
        }
    }

    fn test_instance_pg() -> InstanceInfo {
        InstanceInfo {
            id: InstanceId::from_string("test-pg-001".into()),
            name: "test-pg".into(),
            engine: DatabaseEngine::Postgres,
            workload: WorkloadKind::database(DatabaseEngine::Postgres),
            mode: RuntimeMode::Native,
            state: InstanceState::Starting,
            created_at: Utc::now(),
            pid: None,
            ports: vec![],
            data_dir: "/tmp/pg-test".into(),
            node_id: None,
            energy_port: Some(15432),
            accelerators: vec![],
            volumes: vec![],
            env_vars: HashMap::new(),
            labels: HashMap::new(),
        }
    }

    fn test_instance_redis() -> InstanceInfo {
        InstanceInfo {
            id: InstanceId::from_string("test-redis-001".into()),
            name: "test-redis".into(),
            engine: DatabaseEngine::Redis,
            workload: WorkloadKind::database(DatabaseEngine::Redis),
            mode: RuntimeMode::Native,
            state: InstanceState::Starting,
            created_at: Utc::now(),
            pid: None,
            ports: vec![],
            data_dir: "/tmp/redis-test".into(),
            node_id: None,
            energy_port: Some(16379),
            accelerators: vec![],
            volumes: vec![],
            env_vars: HashMap::new(),
            labels: HashMap::new(),
        }
    }

    #[test]
    fn test_build_args_jouledb() {
        let spec = catalog::get_spec(&DatabaseEngine::JouleDB);
        let instance = test_instance();
        let overrides = ServerOverrides::default();
        let args = NativeBackend::build_args(&spec, &instance, &overrides);
        assert_eq!(args[0], "--data");
        assert_eq!(args[1], "/tmp/jouledb-test-native");
        // Default port 8080 should be included
        assert!(args.contains(&"--port".to_string()));
        assert!(args.contains(&"8080".to_string()));
    }

    #[test]
    fn test_build_args_jouledb_with_port() {
        let spec = catalog::get_spec(&DatabaseEngine::JouleDB);
        let instance = test_instance();
        let overrides = ServerOverrides {
            http_port: Some(9090),
            ..Default::default()
        };
        let args = NativeBackend::build_args(&spec, &instance, &overrides);
        assert!(args.contains(&"--port".to_string()));
        assert!(args.contains(&"9090".to_string()));
    }

    #[test]
    fn test_build_args_postgres() {
        let spec = catalog::get_spec(&DatabaseEngine::Postgres);
        let instance = test_instance_pg();
        let overrides = ServerOverrides {
            engine_port: Some(5433),
            ..Default::default()
        };
        let args = NativeBackend::build_args(&spec, &instance, &overrides);
        assert!(args.contains(&"-D".to_string()));
        assert!(args.contains(&"/tmp/pg-test".to_string()));
        assert!(args.contains(&"-p".to_string()));
        assert!(args.contains(&"5433".to_string()));
    }

    #[test]
    fn test_build_args_redis() {
        let spec = catalog::get_spec(&DatabaseEngine::Redis);
        let instance = test_instance_redis();
        let overrides = ServerOverrides::default();
        let args = NativeBackend::build_args(&spec, &instance, &overrides);
        assert!(args.contains(&"--dir".to_string()));
        assert!(args.contains(&"/tmp/redis-test".to_string()));
        assert!(args.contains(&"--port".to_string()));
        assert!(args.contains(&"6379".to_string()));
        // Default args from catalog
        assert!(args.contains(&"--save".to_string()));
    }

    #[test]
    fn test_build_args_with_extra() {
        let spec = catalog::get_spec(&DatabaseEngine::JouleDB);
        let instance = test_instance();
        let overrides = ServerOverrides {
            extra_args: vec!["--query_timeout".into(), "5000".into()],
            ..Default::default()
        };
        let args = NativeBackend::build_args(&spec, &instance, &overrides);
        assert!(args.contains(&"--query_timeout".to_string()));
        assert!(args.contains(&"5000".to_string()));
    }

    #[test]
    fn test_build_health_cmd_postgres() {
        let spec = catalog::get_spec(&DatabaseEngine::Postgres);
        let cmd = NativeBackend::build_health_cmd(&spec, 5433).unwrap();
        assert_eq!(cmd[0], "pg_isready");
        assert!(cmd.contains(&"5433".to_string()));
        assert!(!cmd.iter().any(|s| s.contains("{port}")));
    }

    #[test]
    fn test_build_health_cmd_jouledb_none() {
        let spec = catalog::get_spec(&DatabaseEngine::JouleDB);
        assert!(NativeBackend::build_health_cmd(&spec, 8080).is_none());
    }

    #[test]
    fn test_build_health_cmd_redis() {
        let spec = catalog::get_spec(&DatabaseEngine::Redis);
        let cmd = NativeBackend::build_health_cmd(&spec, 6380).unwrap();
        assert_eq!(cmd[0], "redis-cli");
        assert!(cmd.contains(&"6380".to_string()));
    }

    #[tokio::test]
    async fn test_status_unknown_instance() {
        let backend = NativeBackend::new();
        let state = backend.status("nonexistent").await.unwrap();
        assert_eq!(state, InstanceState::Stopped);
    }

    #[tokio::test]
    async fn test_health_check_unknown_instance() {
        let backend = NativeBackend::new();
        let healthy = backend.health_check("nonexistent").await.unwrap();
        assert!(!healthy);
    }
}
