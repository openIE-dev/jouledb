//! `jouledb run` — launch any workload with Joule energy telemetry.
//!
//! Auto-detects workload type from the argument:
//!
//! ```bash
//! jouledb run postgres                           # database (known engine)
//! jouledb run redis --port 6380                  # database with port override
//! jouledb run nginx:latest -p 8080:80            # OCI container
//! jouledb run ghcr.io/user/llm:v1 --gpu 0       # container with GPU passthrough
//! jouledb run ./my-binary -- --flag              # arbitrary process
//! jouledb run mysql -- --skip-grant-tables       # extra args after --
//! ```

use crate::{Config, Result, output::Output};
use clap::Args;
use colored::Colorize;
use std::collections::HashMap;

/// Run any workload with Joule energy telemetry.
///
/// Accepts database engine names (postgres, redis, mysql, etc.),
/// OCI container image references (nginx:latest, ghcr.io/user/app:v1),
/// or executable paths (./my-binary, /usr/bin/app).
#[derive(Args)]
pub struct RunCommand {
    /// Workload to run: engine name, container image, or binary path.
    pub workload: String,

    /// Publish a port (host:container). Repeatable.
    #[arg(short = 'p', long = "publish")]
    pub publish: Vec<String>,

    /// Port for the database/process to listen on (default: engine's standard port).
    #[arg(long)]
    pub port: Option<u16>,

    /// Data directory (default: ./{name}-data-{id}).
    #[arg(short, long)]
    pub data: Option<String>,

    /// Port for the energy sidecar HTTP server (default: engine_port + 10000).
    #[arg(long)]
    pub energy_port: Option<u16>,

    /// Runtime isolation mode: native, vm, wasm (default: native).
    #[arg(short, long, default_value = "native")]
    pub mode: String,

    /// Instance name (default: auto-generated).
    #[arg(long)]
    pub name: Option<String>,

    /// Run in foreground (don't daemonize).
    #[arg(long)]
    pub foreground: bool,

    /// GPU device IDs to passthrough. Repeatable (e.g. --gpu 0 --gpu 1).
    #[arg(long)]
    pub gpu: Vec<String>,

    /// TPU device IDs to passthrough.
    #[arg(long)]
    pub tpu: Vec<String>,

    /// NPU passthrough (auto-detect Neural Engine).
    #[arg(long)]
    pub npu: bool,

    /// GPU memory limit in MB.
    #[arg(long)]
    pub gpu_memory: Option<u64>,

    /// Volume mount (host:container[:ro]). Repeatable.
    #[arg(short = 'v', long = "volume")]
    pub volumes: Vec<String>,

    /// Environment variable (KEY=VALUE). Repeatable.
    #[arg(short = 'e', long = "env")]
    pub envs: Vec<String>,

    /// Label (key=value). Repeatable.
    #[arg(short = 'l', long = "label")]
    pub label: Vec<String>,

    /// Energy budget (e.g. "100J/hour"). Instance is throttled when exceeded.
    #[arg(long)]
    pub energy_budget: Option<String>,

    /// Extra arguments passed through to the workload.
    #[arg(last = true)]
    pub extra_args: Vec<String>,
}

/// Auto-detect workload type from the argument string.
fn detect_workload_kind(input: &str) -> joule_runtime::WorkloadKind {
    use joule_runtime::{DatabaseEngine, WorkloadKind};

    // Known database engine names
    let known_engines = [
        "jouledb",
        "joule-db",
        "joule",
        "postgres",
        "postgresql",
        "pg",
        "pgsql",
        "mysql",
        "mariadb",
        "redis",
        "mongodb",
        "mongo",
        "mongod",
        "sqlite",
        "sqlite3",
    ];

    let lower = input.to_lowercase();

    if known_engines.contains(&lower.as_str()) {
        let engine: DatabaseEngine = input.parse().unwrap_or_default();
        return WorkloadKind::database(engine);
    }

    // LLM model reference: known families like llama3:8b, mistral:7b, deepseek-r1:671b
    if joule_runtime::llm::is_llm_model_ref(input) {
        return WorkloadKind::container(input);
    }

    // Path to executable: starts with ./ or / or \ (check before container detection)
    if input.starts_with("./") || input.starts_with('/') || input.starts_with('\\') {
        return WorkloadKind::process(input, vec![]);
    }

    // Container image: contains ':' (tag) or '/' (registry/repo)
    if input.contains(':') || input.contains('/') {
        return WorkloadKind::container(input);
    }

    // Fall through: treat as custom database engine
    let engine: DatabaseEngine = input.parse().unwrap_or_default();
    WorkloadKind::database(engine)
}

/// Parse volume mount string: "host:container[:ro]"
fn parse_volume(s: &str) -> std::result::Result<joule_runtime::VolumeMount, String> {
    let parts: Vec<&str> = s.splitn(3, ':').collect();
    match parts.len() {
        2 => Ok(joule_runtime::VolumeMount {
            host_path: parts[0].into(),
            container_path: parts[1].into(),
            read_only: false,
        }),
        3 => Ok(joule_runtime::VolumeMount {
            host_path: parts[0].into(),
            container_path: parts[1].into(),
            read_only: parts[2] == "ro",
        }),
        _ => Err(format!(
            "invalid volume mount '{}', expected host:container[:ro]",
            s
        )),
    }
}

/// Parse port publish string: "host:container" or just "port" (both sides same).
fn parse_port_publish(s: &str) -> std::result::Result<(u16, u16), String> {
    let parts: Vec<&str> = s.splitn(2, ':').collect();
    match parts.len() {
        1 => {
            let port: u16 = parts[0]
                .parse()
                .map_err(|_| format!("invalid port '{}'", s))?;
            Ok((port, port))
        }
        2 => {
            let host: u16 = parts[0]
                .parse()
                .map_err(|_| format!("invalid host port '{}'", parts[0]))?;
            let container: u16 = parts[1]
                .parse()
                .map_err(|_| format!("invalid container port '{}'", parts[1]))?;
            Ok((host, container))
        }
        _ => Err(format!(
            "invalid port mapping '{}', expected host:container",
            s
        )),
    }
}

/// Parse env var: "KEY=VALUE"
fn parse_env(s: &str) -> std::result::Result<(String, String), String> {
    match s.split_once('=') {
        Some((key, value)) => Ok((key.into(), value.into())),
        None => Err(format!("invalid env var '{}', expected KEY=VALUE", s)),
    }
}

/// Parse label: "key=value"
fn parse_label(s: &str) -> std::result::Result<(String, String), String> {
    match s.split_once('=') {
        Some((key, value)) => Ok((key.into(), value.into())),
        None => Err(format!("invalid label '{}', expected key=value", s)),
    }
}

/// Build accelerator bindings from CLI flags.
fn build_accelerator_bindings(cmd: &RunCommand) -> Vec<joule_runtime::AcceleratorBinding> {
    use joule_runtime::{AcceleratorBinding, AcceleratorKind};

    let mut bindings = Vec::new();

    for device_id in &cmd.gpu {
        bindings.push(AcceleratorBinding {
            kind: AcceleratorKind::GPU,
            device_id: Some(device_id.clone()),
            memory_mb: cmd.gpu_memory,
        });
    }

    // If --gpu with no device IDs is not possible (clap Vec), auto-detect is via empty Vec
    // GPU with no specific ID: user can do --gpu auto or similar in the future

    for device_id in &cmd.tpu {
        bindings.push(AcceleratorBinding {
            kind: AcceleratorKind::TPU,
            device_id: Some(device_id.clone()),
            memory_mb: None,
        });
    }

    if cmd.npu {
        bindings.push(AcceleratorBinding {
            kind: AcceleratorKind::NPU,
            device_id: None,
            memory_mb: None,
        });
    }

    bindings
}

pub async fn execute(cmd: RunCommand, _config: &Config, output: &Output) -> Result<()> {
    use joule_runtime::daemon_client::DaemonClient;
    use joule_runtime::{RuntimeConfig, RuntimeMode, ServerOverrides, WorkloadKind, catalog};

    let workload = detect_workload_kind(&cmd.workload);

    // Parse mode
    let mode: RuntimeMode = cmd
        .mode
        .parse()
        .map_err(|e: joule_runtime::RuntimeError| crate::error::CliError::Runtime(e.to_string()))?;

    // Parse port publishes
    let port_publishes: Vec<(u16, u16)> = cmd
        .publish
        .iter()
        .map(|s| parse_port_publish(s))
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| crate::error::CliError::Runtime(e))?;

    // Parse volumes
    let volumes: Vec<joule_runtime::VolumeMount> = cmd
        .volumes
        .iter()
        .map(|s| parse_volume(s))
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| crate::error::CliError::Runtime(e))?;

    // Parse env vars
    let env_vars: HashMap<String, String> = cmd
        .envs
        .iter()
        .map(|s| parse_env(s))
        .collect::<std::result::Result<HashMap<_, _>, _>>()
        .map_err(|e| crate::error::CliError::Runtime(e))?;

    // Parse labels
    let labels: HashMap<String, String> = cmd
        .label
        .iter()
        .map(|s| parse_label(s))
        .collect::<std::result::Result<HashMap<_, _>, _>>()
        .map_err(|e| crate::error::CliError::Runtime(e))?;

    // Build accelerator bindings
    let accelerators = build_accelerator_bindings(&cmd);

    match &workload {
        WorkloadKind::Database { engine } => {
            execute_database(
                engine,
                &cmd,
                mode,
                &volumes,
                &env_vars,
                &labels,
                &accelerators,
                output,
            )
            .await
        }
        WorkloadKind::Container { image } => {
            execute_container(
                image,
                &cmd,
                mode,
                &port_publishes,
                &volumes,
                &env_vars,
                &labels,
                &accelerators,
                output,
            )
            .await
        }
        WorkloadKind::Process { binary, .. } => {
            execute_process(
                binary,
                &cmd,
                mode,
                &volumes,
                &env_vars,
                &labels,
                &accelerators,
                output,
            )
            .await
        }
    }
}

async fn execute_database(
    engine: &joule_runtime::DatabaseEngine,
    cmd: &RunCommand,
    mode: joule_runtime::RuntimeMode,
    volumes: &[joule_runtime::VolumeMount],
    env_vars: &HashMap<String, String>,
    labels: &HashMap<String, String>,
    accelerators: &[joule_runtime::AcceleratorBinding],
    output: &Output,
) -> Result<()> {
    use joule_runtime::daemon_client::DaemonClient;
    use joule_runtime::{DatabaseEngine, RuntimeConfig, ServerOverrides, WorkloadKind, catalog};

    let spec = catalog::get_spec(engine);

    let engine_port = cmd.port.unwrap_or(spec.default_port);
    let energy_port = if *engine != DatabaseEngine::JouleDB {
        cmd.energy_port.unwrap_or(engine_port + 10000)
    } else {
        0
    };

    let data_dir = cmd.data.clone().unwrap_or_else(|| {
        let short_id = &uuid::Uuid::new_v4().to_string()[..8];
        format!("./{}-data-{}", engine, short_id)
    });

    let overrides = ServerOverrides {
        engine_port: Some(engine_port),
        http_port: if *engine == DatabaseEngine::JouleDB {
            Some(engine_port)
        } else {
            None
        },
        data_dir: Some(data_dir.clone()),
        extra_args: cmd.extra_args.clone(),
        ..Default::default()
    };

    print_database_banner(
        engine,
        &spec,
        engine_port,
        energy_port,
        &data_dir,
        mode,
        accelerators,
    );

    let instance_name = cmd
        .name
        .clone()
        .unwrap_or_else(|| format!("{}-{}", engine, &uuid::Uuid::new_v4().to_string()[..8]));

    // Daemon-first routing
    let daemon_client = DaemonClient::default();
    if daemon_client.is_daemon_running() {
        output.verbose("Daemon detected — delegating to daemon...");

        let instance_id = daemon_client
            .start_instance(instance_name, engine.to_string(), overrides)
            .await
            .map_err(|e| crate::error::CliError::Runtime(e.to_string()))?;

        output.success(&format!(
            "{} started via daemon (instance: {})",
            spec.display_name,
            &instance_id[..8.min(instance_id.len())]
        ));

        if cmd.foreground {
            wait_foreground(&daemon_client, &instance_id, output).await?;
        }

        return Ok(());
    }

    // Direct RuntimeManager
    output.verbose("No daemon running — starting directly.");
    output.verbose("Tip: run 'jouledb daemon start' for persistent instance management.");

    let config = RuntimeConfig {
        mode,
        ..Default::default()
    };
    let runtime_data = std::path::PathBuf::from(&data_dir);
    std::fs::create_dir_all(&runtime_data).map_err(|e| {
        crate::error::CliError::Runtime(format!("failed to create data directory: {}", e))
    })?;

    let manager = joule_runtime::RuntimeManager::new(config, runtime_data)
        .map_err(|e| crate::error::CliError::Runtime(e.to_string()))?;

    let workload = joule_runtime::WorkloadKind::database(engine.clone());
    let instance_id = manager
        .start_workload(
            instance_name,
            workload,
            overrides,
            accelerators.to_vec(),
            volumes.to_vec(),
            env_vars.clone(),
            labels.clone(),
        )
        .await
        .map_err(|e| crate::error::CliError::Runtime(e.to_string()))?;

    output.success(&format!(
        "{} started (instance: {})",
        spec.display_name, instance_id
    ));

    if cmd.foreground {
        output.info("Press Ctrl+C to stop...");
        tokio::signal::ctrl_c().await.ok();
        output.info("Shutting down...");
        manager
            .stop_instance(instance_id.as_str())
            .await
            .map_err(|e| crate::error::CliError::Runtime(e.to_string()))?;
        output.success("Stopped.");
    }

    Ok(())
}

async fn execute_container(
    image: &str,
    cmd: &RunCommand,
    mode: joule_runtime::RuntimeMode,
    port_publishes: &[(u16, u16)],
    volumes: &[joule_runtime::VolumeMount],
    env_vars: &HashMap<String, String>,
    labels: &HashMap<String, String>,
    accelerators: &[joule_runtime::AcceleratorBinding],
    output: &Output,
) -> Result<()> {
    use joule_runtime::daemon_client::DaemonClient;
    use joule_runtime::{RuntimeConfig, ServerOverrides, WorkloadKind};

    print_container_banner(image, port_publishes, mode, accelerators);

    let instance_name = cmd.name.clone().unwrap_or_else(|| {
        let base = image.split('/').last().unwrap_or(image);
        let base = base.split(':').next().unwrap_or(base);
        format!("{}-{}", base, &uuid::Uuid::new_v4().to_string()[..8])
    });

    let data_dir = cmd.data.clone().unwrap_or_else(|| {
        let short_id = &uuid::Uuid::new_v4().to_string()[..8];
        format!("./{}-data-{}", instance_name, short_id)
    });

    let overrides = ServerOverrides {
        engine_port: cmd.port,
        data_dir: Some(data_dir.clone()),
        extra_args: cmd.extra_args.clone(),
        ..Default::default()
    };

    let workload = WorkloadKind::container(image);

    // Daemon-first routing
    let daemon_client = DaemonClient::default();
    if daemon_client.is_daemon_running() {
        output.verbose("Daemon detected — delegating to daemon...");

        let instance_id = daemon_client
            .run_workload(
                instance_name,
                workload,
                overrides,
                accelerators.to_vec(),
                volumes.to_vec(),
                env_vars.clone(),
                labels.clone(),
            )
            .await
            .map_err(|e| crate::error::CliError::Runtime(e.to_string()))?;

        output.success(&format!(
            "Container {} started via daemon (instance: {})",
            image,
            &instance_id[..8.min(instance_id.len())]
        ));

        if cmd.foreground {
            wait_foreground(&daemon_client, &instance_id, output).await?;
        }

        return Ok(());
    }

    // Direct RuntimeManager fallback
    output.verbose("No daemon running — starting directly.");

    let config = RuntimeConfig {
        mode,
        ..Default::default()
    };
    let runtime_data = std::path::PathBuf::from(&data_dir);
    std::fs::create_dir_all(&runtime_data).map_err(|e| {
        crate::error::CliError::Runtime(format!("failed to create data directory: {}", e))
    })?;

    let manager = joule_runtime::RuntimeManager::new(config, runtime_data)
        .map_err(|e| crate::error::CliError::Runtime(e.to_string()))?;

    let instance_id = manager
        .start_workload(
            instance_name,
            workload,
            overrides,
            accelerators.to_vec(),
            volumes.to_vec(),
            env_vars.clone(),
            labels.clone(),
        )
        .await
        .map_err(|e| crate::error::CliError::Runtime(e.to_string()))?;

    output.success(&format!(
        "Container {} started (instance: {})",
        image, instance_id
    ));

    if cmd.foreground {
        output.info("Press Ctrl+C to stop...");
        tokio::signal::ctrl_c().await.ok();
        output.info("Shutting down...");
        manager
            .stop_instance(instance_id.as_str())
            .await
            .map_err(|e| crate::error::CliError::Runtime(e.to_string()))?;
        output.success("Stopped.");
    }

    Ok(())
}

async fn execute_process(
    binary: &str,
    cmd: &RunCommand,
    mode: joule_runtime::RuntimeMode,
    volumes: &[joule_runtime::VolumeMount],
    env_vars: &HashMap<String, String>,
    labels: &HashMap<String, String>,
    accelerators: &[joule_runtime::AcceleratorBinding],
    output: &Output,
) -> Result<()> {
    use joule_runtime::daemon_client::DaemonClient;
    use joule_runtime::{RuntimeConfig, ServerOverrides, WorkloadKind};

    print_process_banner(binary, mode, accelerators);

    let bin_name = std::path::Path::new(binary)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| binary.to_string());

    let instance_name = cmd
        .name
        .clone()
        .unwrap_or_else(|| format!("{}-{}", bin_name, &uuid::Uuid::new_v4().to_string()[..8]));

    let data_dir = cmd.data.clone().unwrap_or_else(|| {
        let short_id = &uuid::Uuid::new_v4().to_string()[..8];
        format!("./{}-data-{}", bin_name, short_id)
    });

    let overrides = ServerOverrides {
        engine_port: cmd.port,
        data_dir: Some(data_dir.clone()),
        extra_args: cmd.extra_args.clone(),
        ..Default::default()
    };

    let workload = WorkloadKind::process(binary, cmd.extra_args.clone());

    // Daemon-first routing
    let daemon_client = DaemonClient::default();
    if daemon_client.is_daemon_running() {
        output.verbose("Daemon detected — delegating to daemon...");

        let instance_id = daemon_client
            .run_workload(
                instance_name,
                workload,
                overrides,
                accelerators.to_vec(),
                volumes.to_vec(),
                env_vars.clone(),
                labels.clone(),
            )
            .await
            .map_err(|e| crate::error::CliError::Runtime(e.to_string()))?;

        output.success(&format!(
            "Process {} started via daemon (instance: {})",
            binary,
            &instance_id[..8.min(instance_id.len())]
        ));

        if cmd.foreground {
            wait_foreground(&daemon_client, &instance_id, output).await?;
        }

        return Ok(());
    }

    // Direct RuntimeManager fallback
    output.verbose("No daemon running — starting directly.");

    let config = RuntimeConfig {
        mode,
        ..Default::default()
    };
    let runtime_data = std::path::PathBuf::from(&data_dir);
    std::fs::create_dir_all(&runtime_data).map_err(|e| {
        crate::error::CliError::Runtime(format!("failed to create data directory: {}", e))
    })?;

    let manager = joule_runtime::RuntimeManager::new(config, runtime_data)
        .map_err(|e| crate::error::CliError::Runtime(e.to_string()))?;

    let instance_id = manager
        .start_workload(
            instance_name,
            workload,
            overrides,
            accelerators.to_vec(),
            volumes.to_vec(),
            env_vars.clone(),
            labels.clone(),
        )
        .await
        .map_err(|e| crate::error::CliError::Runtime(e.to_string()))?;

    output.success(&format!(
        "Process {} started (instance: {})",
        binary, instance_id
    ));

    if cmd.foreground {
        output.info("Press Ctrl+C to stop...");
        tokio::signal::ctrl_c().await.ok();
        output.info("Shutting down...");
        manager
            .stop_instance(instance_id.as_str())
            .await
            .map_err(|e| crate::error::CliError::Runtime(e.to_string()))?;
        output.success("Stopped.");
    }

    Ok(())
}

/// Wait in foreground mode — Ctrl+C stops the instance via daemon.
async fn wait_foreground(
    daemon_client: &joule_runtime::daemon_client::DaemonClient,
    instance_id: &str,
    output: &Output,
) -> Result<()> {
    output.info("Press Ctrl+C to stop...");
    tokio::signal::ctrl_c().await.ok();
    output.info("Shutting down...");
    daemon_client
        .stop_instance(instance_id)
        .await
        .map_err(|e| crate::error::CliError::Runtime(e.to_string()))?;
    output.success("Stopped.");
    Ok(())
}

fn print_database_banner(
    engine: &joule_runtime::DatabaseEngine,
    spec: &joule_runtime::catalog::DatabaseSpec,
    engine_port: u16,
    energy_port: u16,
    data_dir: &str,
    mode: joule_runtime::RuntimeMode,
    accelerators: &[joule_runtime::AcceleratorBinding],
) {
    let platform = joule_db_energy::detect_platform();

    eprintln!();
    eprintln!(
        "  {} {}  {}",
        spec.display_name.bold(),
        engine_port.to_string().cyan(),
        mode.to_string().dimmed()
    );

    if *engine != joule_runtime::DatabaseEngine::JouleDB && energy_port > 0 {
        eprintln!(
            "  {}   http://127.0.0.1:{}/energy",
            "Energy".bold(),
            energy_port.to_string().cyan()
        );
    }

    eprintln!("  {}     {}", "Data".bold(), data_dir.dimmed());

    if !accelerators.is_empty() {
        let accel_str: Vec<String> = accelerators
            .iter()
            .map(|a| match &a.device_id {
                Some(id) => format!("{}:{}", a.kind, id),
                None => a.kind.to_string(),
            })
            .collect();
        eprintln!("  {}    {}", "Accel".bold(), accel_str.join(", ").green());
    }

    eprintln!();
    eprintln!(
        "  {}  {}  {}",
        platform.cpu_brand.bold(),
        format!("{}W", platform.tdp_watts as u32).yellow(),
        {
            let mut caps = Vec::new();
            if platform.gpu_available {
                caps.push("GPU");
            }
            if platform.npu_available {
                caps.push("Neural Engine");
            }
            if platform.tpu_available {
                caps.push("TPU");
            }
            if caps.is_empty() {
                "CPU only".to_string()
            } else {
                caps.join(" + ")
            }
        }
        .dimmed()
    );
    eprintln!();
}

fn print_container_banner(
    image: &str,
    port_publishes: &[(u16, u16)],
    mode: joule_runtime::RuntimeMode,
    accelerators: &[joule_runtime::AcceleratorBinding],
) {
    let platform = joule_db_energy::detect_platform();

    eprintln!();
    eprintln!(
        "  {} {}  {}",
        "Container".bold(),
        image.cyan(),
        mode.to_string().dimmed()
    );

    if !port_publishes.is_empty() {
        let ports_str: Vec<String> = port_publishes
            .iter()
            .map(|(h, c)| {
                if h == c {
                    h.to_string()
                } else {
                    format!("{}:{}", h, c)
                }
            })
            .collect();
        eprintln!("  {}    {}", "Ports".bold(), ports_str.join(", ").cyan());
    }

    if !accelerators.is_empty() {
        let accel_str: Vec<String> = accelerators
            .iter()
            .map(|a| match &a.device_id {
                Some(id) => format!("{}:{}", a.kind, id),
                None => a.kind.to_string(),
            })
            .collect();
        eprintln!("  {}    {}", "Accel".bold(), accel_str.join(", ").green());
    }

    eprintln!();
    eprintln!(
        "  {}  {}  {}",
        platform.cpu_brand.bold(),
        format!("{}W", platform.tdp_watts as u32).yellow(),
        {
            let mut caps = Vec::new();
            if platform.gpu_available {
                caps.push("GPU");
            }
            if platform.npu_available {
                caps.push("Neural Engine");
            }
            if platform.tpu_available {
                caps.push("TPU");
            }
            if caps.is_empty() {
                "CPU only".to_string()
            } else {
                caps.join(" + ")
            }
        }
        .dimmed()
    );
    eprintln!();
}

fn print_process_banner(
    binary: &str,
    mode: joule_runtime::RuntimeMode,
    accelerators: &[joule_runtime::AcceleratorBinding],
) {
    let platform = joule_db_energy::detect_platform();

    eprintln!();
    eprintln!(
        "  {} {}  {}",
        "Process".bold(),
        binary.cyan(),
        mode.to_string().dimmed()
    );

    if !accelerators.is_empty() {
        let accel_str: Vec<String> = accelerators
            .iter()
            .map(|a| match &a.device_id {
                Some(id) => format!("{}:{}", a.kind, id),
                None => a.kind.to_string(),
            })
            .collect();
        eprintln!("  {}    {}", "Accel".bold(), accel_str.join(", ").green());
    }

    eprintln!();
    eprintln!(
        "  {}  {}  {}",
        platform.cpu_brand.bold(),
        format!("{}W", platform.tdp_watts as u32).yellow(),
        {
            let mut caps = Vec::new();
            if platform.gpu_available {
                caps.push("GPU");
            }
            if platform.npu_available {
                caps.push("Neural Engine");
            }
            if platform.tpu_available {
                caps.push("TPU");
            }
            if caps.is_empty() {
                "CPU only".to_string()
            } else {
                caps.join(" + ")
            }
        }
        .dimmed()
    );
    eprintln!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_database_engines() {
        for name in [
            "postgres",
            "Postgres",
            "POSTGRES",
            "pg",
            "pgsql",
            "postgresql",
        ] {
            let wk = detect_workload_kind(name);
            assert!(wk.as_database().is_some(), "failed for '{}'", name);
        }
        for name in ["redis", "mysql", "mongodb", "sqlite", "jouledb", "joule-db"] {
            let wk = detect_workload_kind(name);
            assert!(wk.as_database().is_some(), "failed for '{}'", name);
        }
    }

    #[test]
    fn test_detect_container_images() {
        let cases = [
            "nginx:latest",
            "redis:7",
            "ghcr.io/user/app:v1",
            "docker.io/library/postgres:16",
            "my-registry.com/image",
        ];
        for input in cases {
            let wk = detect_workload_kind(input);
            match wk {
                joule_runtime::WorkloadKind::Container { image } => {
                    assert_eq!(image, input);
                }
                other => panic!("expected Container for '{}', got {:?}", input, other),
            }
        }
    }

    #[test]
    fn test_detect_process() {
        let cases = ["./my-binary", "/usr/bin/app", "./build/server"];
        for input in cases {
            let wk = detect_workload_kind(input);
            match wk {
                joule_runtime::WorkloadKind::Process { binary, .. } => {
                    assert_eq!(binary, input);
                }
                other => panic!("expected Process for '{}', got {:?}", input, other),
            }
        }
    }

    #[test]
    fn test_detect_unknown_falls_to_database() {
        // Unknown bare names treated as custom database engine
        let wk = detect_workload_kind("clickhouse");
        assert!(wk.as_database().is_some());
    }

    #[test]
    fn test_parse_volume() {
        let v = parse_volume("/host/path:/container/path").unwrap();
        assert_eq!(v.host_path, "/host/path");
        assert_eq!(v.container_path, "/container/path");
        assert!(!v.read_only);

        let v = parse_volume("/data:/mnt:ro").unwrap();
        assert!(v.read_only);

        assert!(parse_volume("invalid").is_err());
    }

    #[test]
    fn test_parse_port_publish() {
        assert_eq!(parse_port_publish("8080:80").unwrap(), (8080, 80));
        assert_eq!(parse_port_publish("3000").unwrap(), (3000, 3000));
        assert!(parse_port_publish("invalid").is_err());
    }

    #[test]
    fn test_parse_env() {
        let (k, v) = parse_env("FOO=bar").unwrap();
        assert_eq!(k, "FOO");
        assert_eq!(v, "bar");

        let (k, v) = parse_env("PATH=/usr/bin:/usr/local/bin").unwrap();
        assert_eq!(k, "PATH");
        assert_eq!(v, "/usr/bin:/usr/local/bin");

        assert!(parse_env("NOEQUALS").is_err());
    }

    #[test]
    fn test_parse_label() {
        let (k, v) = parse_label("team=ml").unwrap();
        assert_eq!(k, "team");
        assert_eq!(v, "ml");

        assert!(parse_label("noeq").is_err());
    }

    #[test]
    fn test_detect_llm_model_refs() {
        // LLM model refs should be detected as containers
        let cases = [
            "llama3:8b",
            "mistral:7b",
            "deepseek-r1:671b",
            "phi-3",
            "qwen2.5:72b",
            "gemma2:9b",
        ];
        for input in cases {
            let wk = detect_workload_kind(input);
            match wk {
                joule_runtime::WorkloadKind::Container { image } => {
                    assert_eq!(image, input, "LLM model ref should be container");
                }
                other => panic!(
                    "expected Container for LLM ref '{}', got {:?}",
                    input, other
                ),
            }
        }
    }
}
