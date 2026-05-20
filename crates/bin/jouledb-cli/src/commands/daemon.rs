//! `jouledb daemon` — manage the persistent JouleDB daemon.
//!
//! ```bash
//! jouledb daemon start           # Start daemon in background
//! jouledb daemon start -f        # Start daemon in foreground
//! jouledb daemon stop            # Graceful shutdown
//! jouledb daemon status          # Show uptime, instance count, energy
//! jouledb daemon logs            # Tail daemon log
//! jouledb daemon install         # Install as OS service (auto-start on login)
//! jouledb daemon uninstall       # Remove OS service
//! ```

use crate::{Config, Result, output::Output};
use clap::Subcommand;
use colored::Colorize;

#[derive(Subcommand)]
pub enum DaemonCommands {
    /// Start the daemon process
    Start {
        /// Run in foreground (don't daemonize)
        #[arg(short, long)]
        foreground: bool,

        /// Dashboard HTTP port (default: 7000)
        #[arg(long)]
        dashboard_port: Option<u16>,
    },

    /// Stop the running daemon
    Stop,

    /// Show daemon status
    Status,

    /// Show daemon logs
    Logs {
        /// Number of lines to show
        #[arg(short, long, default_value = "50")]
        lines: usize,
    },

    /// Install daemon as OS service (auto-start on login)
    Install,

    /// Remove daemon OS service
    Uninstall,
}

pub async fn execute(cmd: DaemonCommands, _config: &Config, output: &Output) -> Result<()> {
    match cmd {
        DaemonCommands::Start {
            foreground,
            dashboard_port,
        } => start_daemon(foreground, dashboard_port, output).await,
        DaemonCommands::Stop => stop_daemon(output).await,
        DaemonCommands::Status => show_status(output).await,
        DaemonCommands::Logs { lines } => show_logs(lines, output).await,
        DaemonCommands::Install => install_service(output),
        DaemonCommands::Uninstall => uninstall_service(output),
    }
}

async fn start_daemon(
    foreground: bool,
    dashboard_port: Option<u16>,
    output: &Output,
) -> Result<()> {
    use joule_runtime::daemon::is_daemon_running;
    use joule_runtime::daemon::{DaemonConfig, DaemonCore, default_daemon_dir, default_pid_path};

    let pid_path = default_pid_path();

    // Check if already running
    if is_daemon_running(&pid_path) {
        output.warning("Daemon is already running.");
        output.info("Use 'jouledb daemon status' to see details.");
        return Ok(());
    }

    let daemon_dir = default_daemon_dir();
    output.info(&format!("Starting JouleDB daemon..."));
    output.verbose(&format!("Daemon directory: {}", daemon_dir.display()));

    let config = DaemonConfig {
        runtime_config: joule_runtime::RuntimeConfig::default(),
        daemon_dir: daemon_dir.clone(),
        socket_path: None,
        pid_path: None,
        dashboard_port,
    };

    if foreground {
        // Run in foreground — useful for debugging and for launchd/systemd
        output.info("Running in foreground (Ctrl+C to stop)...");

        let daemon =
            DaemonCore::new(config).map_err(|e| crate::error::CliError::Runtime(e.to_string()))?;

        // Start dashboard
        #[cfg(feature = "energy-sidecar")]
        {
            use joule_runtime::dashboard::{
                DEFAULT_DASHBOARD_PORT, DashboardState, start_dashboard,
            };
            let dash_port = dashboard_port.unwrap_or(DEFAULT_DASHBOARD_PORT);
            let dash_state = std::sync::Arc::new(DashboardState {
                manager: std::sync::Arc::clone(daemon.manager()),
            });

            match start_dashboard(dash_state, dash_port).await {
                Ok(_handle) => {
                    eprintln!(
                        "  {} http://127.0.0.1:{}",
                        "Dashboard".bold(),
                        dash_port.to_string().cyan()
                    );
                }
                Err(e) => {
                    output.warning(&format!("Dashboard failed to start: {}", e));
                }
            }
        }

        // Start health monitor
        let health_shutdown_rx = {
            // We'll use the daemon's internal shutdown for this
            // The health monitor runs inside the daemon loop
            // For simplicity, we create a separate one
            let (_tx, rx) = tokio::sync::watch::channel(false);
            let monitor = joule_runtime::health_monitor::HealthMonitor::new(
                std::sync::Arc::clone(daemon.manager()),
                joule_runtime::health_monitor::HealthMonitorConfig::default(),
            );
            tokio::spawn(async move {
                monitor.run(rx).await;
            });
            _tx
        };

        print_daemon_banner(dashboard_port);

        // Run daemon (blocks until shutdown)
        daemon
            .run()
            .await
            .map_err(|e| crate::error::CliError::Runtime(e.to_string()))?;

        // Signal health monitor to stop
        let _ = health_shutdown_rx.send(true);

        output.success("Daemon stopped.");
    } else {
        // Daemonize: spawn a new process with --foreground
        output.info("Starting daemon as background process...");

        let exe = std::env::current_exe().map_err(|e| {
            crate::error::CliError::Runtime(format!("cannot find current executable: {}", e))
        })?;

        let mut args = vec![
            "daemon".to_string(),
            "start".to_string(),
            "--foreground".to_string(),
        ];
        if let Some(port) = dashboard_port {
            args.push("--dashboard-port".to_string());
            args.push(port.to_string());
        }

        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            let child = std::process::Command::new(&exe)
                .args(&args)
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .process_group(0)
                .spawn()
                .map_err(|e| {
                    crate::error::CliError::Runtime(format!("failed to spawn daemon: {}", e))
                })?;

            output.success(&format!("Daemon started (PID {})", child.id()));
        }

        #[cfg(not(unix))]
        {
            let child = std::process::Command::new(&exe)
                .args(&args)
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
                .map_err(|e| {
                    crate::error::CliError::Runtime(format!("failed to spawn daemon: {}", e))
                })?;

            output.success(&format!("Daemon started (PID {})", child.id()));
        }

        output.info("Use 'jouledb daemon status' to check on the daemon.");
    }

    Ok(())
}

async fn stop_daemon(output: &Output) -> Result<()> {
    use joule_runtime::daemon_client::DaemonClient;

    let client = DaemonClient::default();

    if !client.is_daemon_running() {
        output.info("Daemon is not running.");
        return Ok(());
    }

    output.info("Stopping daemon...");

    match client.shutdown().await {
        Ok(()) => {
            output.success("Daemon shutdown initiated.");
            // Wait briefly for clean exit
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            if !client.is_daemon_running() {
                output.success("Daemon stopped.");
            } else {
                output.info("Daemon is still shutting down (stopping instances)...");
            }
        }
        Err(e) => {
            output.warning(&format!("Failed to send shutdown: {}", e));
            output.info("The daemon may have already stopped.");
        }
    }

    Ok(())
}

async fn show_status(output: &Output) -> Result<()> {
    use joule_runtime::daemon_client::DaemonClient;

    let client = DaemonClient::default();

    if !client.is_daemon_running() {
        output.info("Daemon is not running.");
        output.info("Start it with: jouledb daemon start");
        return Ok(());
    }

    match client.status().await {
        Ok(status) => {
            let uptime = format_uptime(status.uptime_secs);
            output.section("JouleDB Daemon");
            output.key_value(vec![
                ("Status", "Running".to_string()),
                ("Uptime", uptime),
                ("Instances", status.instance_count.to_string()),
                (
                    "Total Energy",
                    format!("{:.2} J", status.total_energy_joules),
                ),
            ]);

            // Also list instances
            if let Ok(instances) = client.list_instances().await {
                if !instances.is_empty() {
                    output.section("Instances");
                    for inst in &instances {
                        output.key_value(vec![
                            ("ID", inst.id.as_str()[..8].to_string()),
                            ("Name", inst.name.clone()),
                            ("Engine", inst.engine.to_string()),
                            ("State", inst.state.to_string()),
                            ("PID", inst.pid.map_or("-".to_string(), |p| p.to_string())),
                        ]);
                    }
                }
            }
        }
        Err(e) => {
            output.warning(&format!("Failed to query daemon: {}", e));
        }
    }

    Ok(())
}

async fn show_logs(lines: usize, output: &Output) -> Result<()> {
    let log_path = joule_runtime::daemon::default_log_path();

    if !log_path.exists() {
        output.info("No daemon log file found.");
        output.verbose(&format!("Expected at: {}", log_path.display()));
        return Ok(());
    }

    let content = std::fs::read_to_string(&log_path)
        .map_err(|e| crate::error::CliError::Runtime(format!("failed to read log file: {}", e)))?;

    let log_lines: Vec<_> = content.lines().rev().take(lines).collect();
    for line in log_lines.into_iter().rev() {
        output.raw(line);
    }

    Ok(())
}

fn install_service(output: &Output) -> Result<()> {
    match joule_runtime::platform_service::install_service() {
        Ok(msg) => {
            output.success(&msg);
            output.info("The daemon will start automatically on login.");
        }
        Err(e) => {
            output.error(&format!("Failed to install service: {}", e));
        }
    }
    Ok(())
}

fn uninstall_service(output: &Output) -> Result<()> {
    match joule_runtime::platform_service::uninstall_service() {
        Ok(msg) => {
            output.success(&msg);
        }
        Err(e) => {
            output.error(&format!("Failed to uninstall service: {}", e));
        }
    }
    Ok(())
}

fn format_uptime(secs: u64) -> String {
    if secs < 60 {
        format!("{}s", secs)
    } else if secs < 3600 {
        format!("{}m {}s", secs / 60, secs % 60)
    } else if secs < 86400 {
        format!("{}h {}m", secs / 3600, (secs % 3600) / 60)
    } else {
        format!("{}d {}h", secs / 86400, (secs % 86400) / 3600)
    }
}

fn print_daemon_banner(dashboard_port: Option<u16>) {
    let dash_port = dashboard_port.unwrap_or(joule_runtime::dashboard::DEFAULT_DASHBOARD_PORT);

    eprintln!();
    eprintln!("  {} {}", "JouleDB Daemon".bold(), "running".green());
    eprintln!(
        "  {}   http://127.0.0.1:{}",
        "Dashboard".bold(),
        dash_port.to_string().cyan()
    );
    eprintln!(
        "  {}      {}",
        "Socket".bold(),
        joule_runtime::daemon::default_socket_path()
            .display()
            .to_string()
            .dimmed()
    );
    eprintln!();
}
