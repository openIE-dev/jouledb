//! `jouledb compute` — energy-metered polyglot compute via the daemon.
//!
//! ```bash
//! jouledb compute start python         # Start a Python kernel
//! jouledb compute exec <id> "print(1)" # Execute code in a kernel
//! jouledb compute ls                   # List active kernels
//! jouledb compute stop <id>            # Stop a kernel
//! jouledb compute stop-all             # Stop all kernels
//! jouledb compute run notebook.jnb     # Execute a notebook file
//! ```

use crate::{Config, Result, error::CliError, output::Output};
use clap::Subcommand;
use colored::Colorize;
use joule_runtime::compute::KernelKind;
use joule_runtime::daemon_client::DaemonClient;

#[derive(Subcommand)]
pub enum ComputeCommands {
    /// Start a new compute kernel
    Start {
        /// Kernel type: python, julia, shell, claude
        kind: String,
    },

    /// Execute code in a running kernel
    Exec {
        /// Kernel ID (from `compute start` or `compute ls`)
        kernel_id: String,

        /// Code to execute (or "-" to read from stdin)
        code: String,
    },

    /// List active compute kernels
    Ls,

    /// Stop a specific compute kernel
    Stop {
        /// Kernel ID to stop
        kernel_id: String,
    },

    /// Stop all compute kernels
    StopAll,

    /// Execute a .jnb notebook file
    Run {
        /// Path to notebook file
        path: String,
    },
}

pub async fn execute(cmd: ComputeCommands, _config: &Config, output: &Output) -> Result<()> {
    match cmd {
        ComputeCommands::Start { kind } => start_kernel(&kind, output).await,
        ComputeCommands::Exec { kernel_id, code } => exec_cell(&kernel_id, &code, output).await,
        ComputeCommands::Ls => list_kernels(output).await,
        ComputeCommands::Stop { kernel_id } => stop_kernel(&kernel_id, output).await,
        ComputeCommands::StopAll => stop_all(output).await,
        ComputeCommands::Run { path } => run_notebook(&path, output).await,
    }
}

fn parse_kernel_kind(s: &str) -> Result<KernelKind> {
    match s.to_lowercase().as_str() {
        "python" | "py" => Ok(KernelKind::Python),
        "julia" | "jl" => Ok(KernelKind::Julia),
        "shell" | "sh" | "bash" => Ok(KernelKind::Shell),
        "claude" => Ok(KernelKind::Claude),
        _ => Err(CliError::InvalidInput(format!(
            "unknown kernel type '{}' — use python, julia, shell, or claude",
            s
        ))),
    }
}

fn client() -> DaemonClient {
    DaemonClient::default()
}

fn require_daemon(client: &DaemonClient) -> Result<()> {
    if !client.is_daemon_running() {
        return Err(CliError::Runtime(
            "daemon is not running — start it with 'jouledb daemon start'".into(),
        ));
    }
    Ok(())
}

fn format_energy(joules: f64) -> String {
    if joules >= 1.0 {
        format!("{:.3} J", joules)
    } else if joules >= 0.001 {
        format!("{:.3} mJ", joules * 1_000.0)
    } else {
        format!("{:.1} µJ", joules * 1_000_000.0)
    }
}

async fn start_kernel(kind_str: &str, output: &Output) -> Result<()> {
    let kind = parse_kernel_kind(kind_str)?;
    let c = client();
    require_daemon(&c)?;

    let kernel_id = c
        .start_kernel(kind)
        .await
        .map_err(|e| CliError::Runtime(e.to_string()))?;

    output.success(&format!(
        "Started {} kernel: {}",
        kind,
        kernel_id.bright_cyan()
    ));
    Ok(())
}

async fn exec_cell(kernel_id: &str, code: &str, output: &Output) -> Result<()> {
    let code = if code == "-" {
        use std::io::Read;
        let mut buf = String::new();
        std::io::stdin()
            .read_to_string(&mut buf)
            .map_err(|e| CliError::Io(e))?;
        buf
    } else {
        code.to_string()
    };

    let c = client();
    require_daemon(&c)?;

    let result = c
        .execute_cell(kernel_id, &code)
        .await
        .map_err(|e| CliError::Runtime(e.to_string()))?;

    // Print output
    if !result.stdout.is_empty() {
        print!("{}", result.stdout);
    }
    if !result.stderr.is_empty() {
        eprint!("{}", result.stderr.yellow());
    }

    // Energy receipt
    let provenance_tag = format!("{:?}", result.provenance).to_lowercase();
    let mut receipt = format!(
        "⚡ {} ({}) · {:.3}s · {:.2} W [{}]",
        format_energy(result.energy_joules),
        provenance_tag,
        result.duration_secs,
        result.power_watts,
        if result.success {
            "ok".green().to_string()
        } else {
            "error".red().to_string()
        }
    );

    // Claude remote energy decomposition
    if let Some(remote_j) = result.remote_energy_joules {
        let tokens = result.tokens_estimated.unwrap_or(0);
        receipt.push_str(&format!(
            " + {} remote (~{} tokens)",
            format_energy(remote_j),
            tokens
        ));
    }

    output.info(&receipt);
    Ok(())
}

async fn list_kernels(output: &Output) -> Result<()> {
    let c = client();
    require_daemon(&c)?;

    let kernels = c
        .list_kernels()
        .await
        .map_err(|e| CliError::Runtime(e.to_string()))?;

    if kernels.is_empty() {
        output.info("No active kernels.");
        return Ok(());
    }

    output.info(&format!("{} active kernel(s):", kernels.len()));
    for k in &kernels {
        let alive_tag = if k.alive {
            "alive".green()
        } else {
            "dead".red()
        };
        println!(
            "  {} {} [{}]  pid={:?}  cells={}  energy={}",
            k.id.bright_cyan(),
            format!("{:?}", k.kind).to_lowercase(),
            alive_tag,
            k.pid,
            k.cells_executed,
            format_energy(k.cumulative_energy_joules),
        );
    }
    Ok(())
}

async fn stop_kernel(kernel_id: &str, output: &Output) -> Result<()> {
    let c = client();
    require_daemon(&c)?;

    c.stop_kernel(kernel_id)
        .await
        .map_err(|e| CliError::Runtime(e.to_string()))?;

    output.success(&format!("Stopped kernel {}", kernel_id.bright_cyan()));
    Ok(())
}

async fn stop_all(output: &Output) -> Result<()> {
    let c = client();
    require_daemon(&c)?;

    c.stop_all_kernels()
        .await
        .map_err(|e| CliError::Runtime(e.to_string()))?;

    output.success("Stopped all compute kernels.");
    Ok(())
}

async fn run_notebook(path: &str, output: &Output) -> Result<()> {
    use joule_runtime::compute::Notebook;

    let c = client();
    require_daemon(&c)?;

    let notebook = Notebook::load(std::path::Path::new(path))
        .map_err(|e| CliError::Io(e))?;

    let kernel_id = c
        .start_kernel(notebook.kernel)
        .await
        .map_err(|e| CliError::Runtime(e.to_string()))?;

    output.info(&format!(
        "Running {} ({} cells) on {} kernel {}",
        path,
        notebook.cells.len(),
        notebook.kernel,
        kernel_id.bright_cyan(),
    ));

    let mut total_energy = 0.0;
    let mut total_duration = 0.0;
    let mut failures = 0;

    for (i, cell) in notebook.cells.iter().enumerate() {
        let result = c
            .execute_cell(&kernel_id, &cell.source)
            .await
            .map_err(|e| CliError::Runtime(e.to_string()))?;

        if !result.stdout.is_empty() {
            print!("{}", result.stdout);
        }
        if !result.stderr.is_empty() {
            eprint!("{}", result.stderr.yellow());
        }

        let status = if result.success {
            "ok".green()
        } else {
            failures += 1;
            "FAIL".red()
        };

        output.verbose(&format!(
            "  [{}] cell {} · {} · {}",
            status,
            i + 1,
            format_energy(result.energy_joules),
            format!("{:.3}s", result.duration_secs),
        ));

        total_energy += result.energy_joules;
        if let Some(remote) = result.remote_energy_joules {
            total_energy += remote;
        }
        total_duration += result.duration_secs;
    }

    // Stop the kernel after notebook execution
    let _ = c.stop_kernel(&kernel_id).await;

    let summary = format!(
        "Notebook complete: {} cells · {} · {:.3}s · {} failures",
        notebook.cells.len(),
        format_energy(total_energy),
        total_duration,
        failures,
    );

    if failures > 0 {
        output.warning(&summary);
    } else {
        output.success(&summary);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_kernel_kind() {
        assert_eq!(parse_kernel_kind("python").unwrap(), KernelKind::Python);
        assert_eq!(parse_kernel_kind("py").unwrap(), KernelKind::Python);
        assert_eq!(parse_kernel_kind("julia").unwrap(), KernelKind::Julia);
        assert_eq!(parse_kernel_kind("jl").unwrap(), KernelKind::Julia);
        assert_eq!(parse_kernel_kind("shell").unwrap(), KernelKind::Shell);
        assert_eq!(parse_kernel_kind("sh").unwrap(), KernelKind::Shell);
        assert_eq!(parse_kernel_kind("bash").unwrap(), KernelKind::Shell);
        assert_eq!(parse_kernel_kind("claude").unwrap(), KernelKind::Claude);
        assert_eq!(parse_kernel_kind("PYTHON").unwrap(), KernelKind::Python);
        assert!(parse_kernel_kind("ruby").is_err());
    }

    #[test]
    fn test_format_energy() {
        assert!(format_energy(1.5).contains("J"));
        assert!(format_energy(0.005).contains("mJ"));
        assert!(format_energy(0.0001).contains("µJ"));
    }

    #[test]
    fn test_require_daemon_not_running() {
        let c = DaemonClient::new(
            std::path::PathBuf::from("/tmp/nonexistent.sock"),
            std::path::PathBuf::from("/tmp/nonexistent.pid"),
        );
        assert!(require_daemon(&c).is_err());
    }
}
