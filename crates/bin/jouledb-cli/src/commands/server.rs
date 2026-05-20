//! Server management commands

use crate::{Config, Result, output::Output};
use clap::Subcommand;

#[derive(Subcommand)]
pub enum ServerCommands {
    /// Start the JouleDB server
    Start {
        /// Bind address
        #[arg(short, long, default_value = "127.0.0.1")]
        host: String,

        /// Port number
        #[arg(short, long, default_value = "9000")]
        port: u16,

        /// Data directory
        #[arg(short, long, default_value = "./jouledb-data")]
        data_dir: String,

        /// Run in foreground (don't daemonize)
        #[arg(short, long)]
        foreground: bool,

        /// Configuration file
        #[arg(long)]
        config_file: Option<String>,

        /// Runtime isolation mode: native (bare metal), vm (hardware isolation), wasm (sandboxed)
        #[arg(long, default_value = "native")]
        mode: String,
    },

    /// Stop the running server
    Stop {
        /// Force stop (SIGKILL instead of SIGTERM)
        #[arg(short, long)]
        force: bool,
    },

    /// Restart the server
    Restart {
        /// Force restart
        #[arg(short, long)]
        force: bool,
    },

    /// Show server status
    Status,

    /// Show server logs
    Logs {
        /// Number of lines to show
        #[arg(short, long, default_value = "50")]
        lines: usize,

        /// Follow log output
        #[arg(short, long)]
        follow: bool,

        /// Log level filter
        #[arg(long)]
        level: Option<String>,
    },

    /// Show server information
    Info,

    /// List running JouleDB instances
    Ps,

    /// Stop a running JouleDB instance
    Down {
        /// Instance ID or name to stop
        instance: Option<String>,

        /// Stop all running instances
        #[arg(long)]
        all: bool,
    },
}

pub async fn execute(cmd: ServerCommands, config: &Config, output: &Output) -> Result<()> {
    match cmd {
        ServerCommands::Start {
            host,
            port,
            data_dir,
            foreground,
            config_file,
            mode,
        } => {
            start_server(
                &host,
                port,
                &data_dir,
                foreground,
                config_file.as_deref(),
                &mode,
                output,
            )
            .await
        }
        ServerCommands::Stop { force } => stop_server(force, output).await,
        ServerCommands::Restart { force } => restart_server(force, output).await,
        ServerCommands::Status => show_status(config, output).await,
        ServerCommands::Logs {
            lines,
            follow,
            level,
        } => show_logs(lines, follow, level.as_deref(), output).await,
        ServerCommands::Info => show_info(config, output).await,
        ServerCommands::Ps => show_instances(output).await,
        ServerCommands::Down { instance, all } => stop_instances(instance, all, output).await,
    }
}

async fn start_server(
    host: &str,
    port: u16,
    data_dir: &str,
    foreground: bool,
    config_file: Option<&str>,
    mode: &str,
    output: &Output,
) -> Result<()> {
    use crate::CliError;
    use std::path::Path;
    use std::process::Command;

    output.info(&format!(
        "Starting JouleDB server on {}:{} (mode: {})",
        host, port, mode
    ));
    output.verbose(&format!("Data directory: {}", data_dir));

    // Ensure data directory exists
    std::fs::create_dir_all(data_dir)?;

    // Build server arguments
    let mut args = vec![
        "--addr".to_string(),
        format!("{}:{}", host, port),
        "--data".to_string(),
        data_dir.to_string(),
        "--mode".to_string(),
        mode.to_string(),
    ];

    if let Some(cfg) = config_file {
        args.push("--config".to_string());
        args.push(cfg.to_string());
    }

    // Try to find the server binary
    let server_binary = find_server_binary();

    if server_binary.is_none() {
        output.error("Server binary 'joule-db-server' not found in PATH");
        output.info("To start the server manually, run:");
        output.raw(&format!("  joule-db-server {}", args.join(" ")));
        output.info("Install the server with: cargo install joule-db-server");
        return Err(CliError::Server("Server binary not found".to_string()));
    }

    let binary_path = server_binary.unwrap();
    output.verbose(&format!("Using server binary: {}", binary_path));

    if foreground {
        output.info("Running in foreground mode (Ctrl+C to stop)");

        // Spawn server process in foreground
        let status = Command::new(&binary_path)
            .args(&args)
            .status()
            .map_err(|e| CliError::Server(format!("Failed to start server: {}", e)))?;

        if !status.success() {
            return Err(CliError::Server(format!(
                "Server exited with status: {}",
                status
            )));
        }
    } else {
        output.info("Starting server as daemon...");

        // Check if server is already running
        let pid_file = Path::new(data_dir).join("joule-db.pid");
        if pid_file.exists() {
            let pid_str = std::fs::read_to_string(&pid_file)?;
            if let Ok(pid) = pid_str.trim().parse::<u32>() {
                if is_process_running(pid) {
                    output.warning(&format!("Server already running with PID {}", pid));
                    return Ok(());
                } else {
                    output.verbose("Stale PID file found, removing...");
                    let _ = std::fs::remove_file(&pid_file);
                }
            }
        }

        // Spawn detached server process
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;

            let child = Command::new(&binary_path)
                .args(&args)
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .process_group(0) // Create new process group
                .spawn()
                .map_err(|e| CliError::Server(format!("Failed to spawn server: {}", e)))?;

            let pid = child.id();

            // Write PID file
            std::fs::write(&pid_file, pid.to_string())
                .map_err(|e| CliError::Server(format!("Failed to write PID file: {}", e)))?;

            output.success(&format!("Server started with PID {}", pid));
            output.info(&format!("PID file: {}", pid_file.display()));
            output.info("Use 'jouledb server stop' to stop the server");
        }

        #[cfg(not(unix))]
        {
            // On Windows, we don't detach properly, just spawn in background
            let child = Command::new(&binary_path)
                .args(&args)
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
                .map_err(|e| CliError::Server(format!("Failed to spawn server: {}", e)))?;

            let pid = child.id();

            // Write PID file
            std::fs::write(&pid_file, pid.to_string())
                .map_err(|e| CliError::Server(format!("Failed to write PID file: {}", e)))?;

            output.success(&format!("Server started with PID {}", pid));
            output.info(&format!("PID file: {}", pid_file.display()));
            output.warning("Windows daemon mode is limited. Consider using a service manager.");
        }
    }

    Ok(())
}

/// Find the server binary in PATH
fn find_server_binary() -> Option<String> {
    use std::process::Command;

    // Try 'which' on Unix-like systems
    #[cfg(unix)]
    {
        if let Ok(output) = Command::new("which").arg("joule-db-server").output() {
            if output.status.success() {
                if let Ok(path) = String::from_utf8(output.stdout) {
                    let path = path.trim();
                    if !path.is_empty() {
                        return Some(path.to_string());
                    }
                }
            }
        }
    }

    // Try 'where' on Windows
    #[cfg(windows)]
    {
        if let Ok(output) = Command::new("where").arg("joule-db-server.exe").output() {
            if output.status.success() {
                if let Ok(path) = String::from_utf8(output.stdout) {
                    let path = path.lines().next().unwrap_or("").trim();
                    if !path.is_empty() {
                        return Some(path.to_string());
                    }
                }
            }
        }
    }

    None
}

/// Check if a process is running
fn is_process_running(pid: u32) -> bool {
    #[cfg(unix)]
    {
        use std::process::Command;

        // Send signal 0 to check if process exists (doesn't actually send a signal)
        Command::new("kill")
            .arg("-0")
            .arg(pid.to_string())
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }

    #[cfg(not(unix))]
    {
        // On Windows, use tasklist
        use std::process::Command;

        Command::new("tasklist")
            .arg("/FI")
            .arg(format!("PID eq {}", pid))
            .output()
            .map(|output| String::from_utf8_lossy(&output.stdout).contains(&pid.to_string()))
            .unwrap_or(false)
    }
}

async fn stop_server(force: bool, output: &Output) -> Result<()> {
    use crate::CliError;
    use std::path::Path;
    use std::process::Command;

    if force {
        output.warning("Force stopping server...");
    } else {
        output.info("Stopping server gracefully...");
    }

    // Look for PID file in default data directory
    let data_dir = "./jouledb-data";
    let pid_file = Path::new(data_dir).join("joule-db.pid");

    if !pid_file.exists() {
        output.warning("No PID file found. Server may not be running.");
        output.info(&format!("Expected PID file at: {}", pid_file.display()));
        return Ok(());
    }

    // Read PID from file
    let pid_str = std::fs::read_to_string(&pid_file)
        .map_err(|e| CliError::Server(format!("Failed to read PID file: {}", e)))?;

    let pid = pid_str
        .trim()
        .parse::<u32>()
        .map_err(|e| CliError::Server(format!("Invalid PID in file: {}", e)))?;

    output.verbose(&format!("Found server PID: {}", pid));

    // Check if process is running
    if !is_process_running(pid) {
        output.warning(&format!("Process {} is not running", pid));
        output.info("Cleaning up stale PID file...");
        let _ = std::fs::remove_file(&pid_file);
        return Ok(());
    }

    // Send signal to stop the process
    #[cfg(unix)]
    {
        let signal = if force { "KILL" } else { "TERM" };
        let signal_arg = if force { "-9" } else { "-15" };

        let result = Command::new("kill")
            .arg(signal_arg)
            .arg(pid.to_string())
            .output()
            .map_err(|e| CliError::Server(format!("Failed to send signal: {}", e)))?;

        if !result.status.success() {
            let stderr = String::from_utf8_lossy(&result.stderr);
            return Err(CliError::Server(format!(
                "Failed to stop server: {}",
                stderr
            )));
        }

        output.success(&format!("Sent SIG{} to process {}", signal, pid));

        // Wait a bit for graceful shutdown
        if !force {
            output.info("Waiting for server to stop...");
            for i in 0..10 {
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                if !is_process_running(pid) {
                    break;
                }
                if i == 9 {
                    output.warning("Server did not stop gracefully. Use --force to kill.");
                    return Ok(());
                }
            }
        }
    }

    #[cfg(not(unix))]
    {
        // On Windows, use taskkill
        let force_flag = if force { "/F" } else { "" };
        let mut cmd = Command::new("taskkill");
        cmd.arg("/PID").arg(pid.to_string());
        if force {
            cmd.arg("/F");
        }

        let result = cmd
            .output()
            .map_err(|e| CliError::Server(format!("Failed to stop server: {}", e)))?;

        if !result.status.success() {
            let stderr = String::from_utf8_lossy(&result.stderr);
            return Err(CliError::Server(format!(
                "Failed to stop server: {}",
                stderr
            )));
        }

        output.success(&format!("Stopped process {}", pid));
    }

    // Clean up PID file
    if let Err(e) = std::fs::remove_file(&pid_file) {
        output.warning(&format!("Failed to remove PID file: {}", e));
    } else {
        output.verbose("Removed PID file");
    }

    output.success("Server stopped successfully");

    Ok(())
}

async fn restart_server(force: bool, output: &Output) -> Result<()> {
    output.info("Restarting server...");

    // Stop the server (ignore errors if not running)
    let _ = stop_server(force, output).await;

    // Wait a bit to ensure clean shutdown
    output.verbose("Waiting for clean shutdown...");
    tokio::time::sleep(std::time::Duration::from_secs(2)).await;

    // Start with default parameters
    output.info("Starting server with default parameters...");
    start_server(
        "127.0.0.1",
        9000,
        "./jouledb-data",
        false,    // daemon mode
        None,     // no config file
        "native", // default mode
        output,
    )
    .await?;

    output.success("Server restarted successfully");

    Ok(())
}

async fn show_status(config: &Config, output: &Output) -> Result<()> {
    output.section("Server Status");

    let url = format!(
        "http://{}:{}/health",
        config.connection.host, config.connection.port
    );

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()?;

    match client.get(&url).send().await {
        Ok(response) if response.status().is_success() => {
            output.key_value(vec![
                ("Status", "Running".to_string()),
                (
                    "Address",
                    format!("{}:{}", config.connection.host, config.connection.port),
                ),
                ("Health", "OK".to_string()),
            ]);

            // Try to get more details
            if let Ok(body) = response.text().await {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&body) {
                    if let Some(version) = json.get("version") {
                        output.verbose(&format!("Version: {}", version));
                    }
                }
            }
        }
        Ok(response) => {
            output.key_value(vec![
                ("Status", "Unhealthy".to_string()),
                (
                    "Address",
                    format!("{}:{}", config.connection.host, config.connection.port),
                ),
                ("HTTP Status", response.status().to_string()),
            ]);
        }
        Err(_) => {
            output.key_value(vec![
                ("Status", "Not Running".to_string()),
                (
                    "Address",
                    format!("{}:{}", config.connection.host, config.connection.port),
                ),
            ]);
        }
    }

    Ok(())
}

async fn show_logs(lines: usize, follow: bool, level: Option<&str>, output: &Output) -> Result<()> {
    output.info(&format!("Showing last {} log lines", lines));

    if let Some(lvl) = level {
        output.verbose(&format!("Filtering by level: {}", lvl));
    }

    if follow {
        output.info("Following logs (Ctrl+C to stop)...");
    }

    // Log file location (platform-aware)
    let log_path = if cfg!(windows) {
        // Windows: %LOCALAPPDATA%\JouleDB\server.log or fallback
        std::env::var("LOCALAPPDATA")
            .map(|d| {
                std::path::PathBuf::from(d)
                    .join("JouleDB")
                    .join("server.log")
            })
            .unwrap_or_else(|_| std::path::PathBuf::from("./jouledb-data/server.log"))
    } else if cfg!(target_os = "macos") {
        std::path::PathBuf::from(
            std::env::var("HOME")
                .map(|h| format!("{}/Library/Logs/JouleDB/server.log", h))
                .unwrap_or_else(|_| "/var/log/joule-db/server.log".to_string()),
        )
    } else {
        std::path::PathBuf::from("/var/log/joule-db/server.log")
    };

    if log_path.exists() {
        let content = std::fs::read_to_string(&log_path)?;
        let log_lines: Vec<_> = content.lines().rev().take(lines).collect();
        for line in log_lines.into_iter().rev() {
            output.raw(line);
        }
    } else {
        output.warning(&format!("Log file not found at {}", log_path.display()));
        if cfg!(unix) {
            output.info("Try: journalctl -u joule-db -n 50");
        }
    }

    Ok(())
}

async fn show_info(config: &Config, output: &Output) -> Result<()> {
    output.section("Server Information");

    let url = format!(
        "http://{}:{}/info",
        config.connection.host, config.connection.port
    );

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()?;

    match client.get(&url).send().await {
        Ok(response) if response.status().is_success() => {
            if let Ok(body) = response.text().await {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&body) {
                    output.data(&json)?;
                } else {
                    output.raw(&body);
                }
            }
        }
        _ => {
            output.warning("Could not connect to server");
            output.key_value(vec![
                ("Configured Host", config.connection.host.clone()),
                ("Configured Port", config.connection.port.to_string()),
            ]);
        }
    }

    Ok(())
}

async fn show_instances(output: &Output) -> Result<()> {
    output.section("JouleDB Instances");

    // Try daemon first
    let daemon_client = joule_runtime::daemon_client::DaemonClient::default();
    if daemon_client.is_daemon_running() {
        match daemon_client.list_instances().await {
            Ok(instances) => {
                if instances.is_empty() {
                    output.info("No running instances.");
                    return Ok(());
                }
                for inst in &instances {
                    output.key_value(vec![
                        (
                            "ID",
                            inst.id.as_str()[..8.min(inst.id.as_str().len())].to_string(),
                        ),
                        ("Name", inst.name.clone()),
                        ("Engine", inst.engine.to_string()),
                        ("Mode", inst.mode.to_string()),
                        ("State", inst.state.to_string()),
                        ("PID", inst.pid.map_or("-".to_string(), |p| p.to_string())),
                        ("Data", inst.data_dir.clone()),
                    ]);
                }
                return Ok(());
            }
            Err(e) => {
                output.verbose(&format!("Daemon query failed, falling back to file: {}", e));
            }
        }
    }

    // Fallback: read instances.json directly
    let data_dir = "./jouledb-data";
    let instances_path = std::path::Path::new(data_dir).join("instances.json");

    if !instances_path.exists() {
        output.info("No instances registered.");
        output.verbose(&format!(
            "Expected registry at: {}",
            instances_path.display()
        ));
        return Ok(());
    }

    let content = std::fs::read_to_string(&instances_path)
        .map_err(|e| crate::CliError::Server(format!("Failed to read instances: {}", e)))?;

    let instances: std::collections::HashMap<String, serde_json::Value> =
        serde_json::from_str(&content).unwrap_or_default();

    if instances.is_empty() {
        output.info("No running instances.");
        return Ok(());
    }

    for (id, info) in &instances {
        let name = info
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let mode = info.get("mode").and_then(|v| v.as_str()).unwrap_or("?");
        let state = info
            .get("state")
            .and_then(|v| {
                if v.is_string() {
                    Some(v.as_str().unwrap_or("?").to_string())
                } else {
                    Some(format!("{}", v))
                }
            })
            .unwrap_or_else(|| "?".to_string());
        let pid = info
            .get("pid")
            .and_then(|v| v.as_u64())
            .map(|p| p.to_string())
            .unwrap_or_else(|| "-".to_string());
        let data = info.get("data_dir").and_then(|v| v.as_str()).unwrap_or("?");

        output.key_value(vec![
            ("ID", id[..8.min(id.len())].to_string()),
            ("Name", name.to_string()),
            ("Mode", mode.to_string()),
            ("State", state),
            ("PID", pid),
            ("Data", data.to_string()),
        ]);
    }

    Ok(())
}

async fn stop_instances(instance: Option<String>, all: bool, output: &Output) -> Result<()> {
    if !all && instance.is_none() {
        output.error("Specify an instance ID/name or use --all to stop all instances.");
        return Ok(());
    }

    // Try daemon first
    let daemon_client = joule_runtime::daemon_client::DaemonClient::default();
    if daemon_client.is_daemon_running() {
        if all {
            match daemon_client.list_instances().await {
                Ok(instances) => {
                    for inst in &instances {
                        output.info(&format!("Stopping '{}'...", inst.name));
                        let _ = daemon_client.stop_instance(inst.id.as_str()).await;
                        output.success(&format!("Stopped '{}'", inst.name));
                    }
                    return Ok(());
                }
                Err(e) => {
                    output.verbose(&format!("Daemon query failed: {}", e));
                }
            }
        } else if let Some(target) = &instance {
            // Try stopping by ID directly
            match daemon_client.stop_instance(target).await {
                Ok(()) => {
                    output.success(&format!("Stopped '{}'", target));
                    return Ok(());
                }
                Err(e) => {
                    output.verbose(&format!("Daemon stop failed (trying file fallback): {}", e));
                }
            }
        }
    }

    // Fallback: direct file-based approach
    let data_dir = "./jouledb-data";
    let instances_path = std::path::Path::new(data_dir).join("instances.json");

    if !instances_path.exists() {
        output.info("No instances registered.");
        return Ok(());
    }

    let content = std::fs::read_to_string(&instances_path)
        .map_err(|e| crate::CliError::Server(format!("Failed to read instances: {}", e)))?;

    let mut instances: std::collections::HashMap<String, serde_json::Value> =
        serde_json::from_str(&content).unwrap_or_default();

    if instances.is_empty() {
        output.info("No running instances.");
        return Ok(());
    }

    let to_stop: Vec<String> = if all {
        instances.keys().cloned().collect()
    } else {
        let target = instance.as_deref().unwrap_or("");
        // Match by ID prefix or name
        instances
            .iter()
            .filter(|(id, info)| {
                id.starts_with(target)
                    || info
                        .get("name")
                        .and_then(|v| v.as_str())
                        .map(|n| n == target)
                        .unwrap_or(false)
            })
            .map(|(id, _)| id.clone())
            .collect()
    };

    if to_stop.is_empty() {
        output.warning(&format!(
            "No instance matching '{}'",
            instance.as_deref().unwrap_or("")
        ));
        return Ok(());
    }

    for id in &to_stop {
        if let Some(info) = instances.get(id) {
            let name = info
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            if let Some(pid) = info.get("pid").and_then(|v| v.as_u64()) {
                output.info(&format!("Stopping '{}' (PID {})...", name, pid));
                #[cfg(unix)]
                {
                    let _ = std::process::Command::new("kill")
                        .arg("-15")
                        .arg(pid.to_string())
                        .output();
                }
                #[cfg(not(unix))]
                {
                    let _ = std::process::Command::new("taskkill")
                        .args(["/PID", &pid.to_string()])
                        .output();
                }
            }
            output.success(&format!("Stopped '{}'", name));
        }
        instances.remove(id);
    }

    // Write back
    let json = serde_json::to_string_pretty(&instances).unwrap_or_default();
    let _ = std::fs::write(&instances_path, json);

    Ok(())
}
