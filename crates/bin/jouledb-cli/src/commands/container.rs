//! Container management CLI commands.
//!
//! ```bash
//! jouledb ps                          # list all running instances
//! jouledb logs <id> -f --tail 100     # follow container logs
//! jouledb exec <id> -- sh             # exec into running container
//! jouledb stop <id>                   # stop instance
//! jouledb rm <id>                     # remove stopped instance
//! jouledb images                      # list cached images
//! jouledb pull <image>                # pull image from registry
//! jouledb inspect <id>                # show instance details + energy
//! ```

use crate::{Config, Result, output::Output};
use clap::Args;
use colored::Colorize;

/// List running instances.
#[derive(Args)]
pub struct PsCommand {
    /// Show all instances (including stopped).
    #[arg(short, long)]
    pub all: bool,

    /// Filter by label (key=value).
    #[arg(short = 'l', long)]
    pub label: Vec<String>,
}

/// View instance logs.
#[derive(Args)]
pub struct LogsCommand {
    /// Instance ID or name.
    pub instance: String,

    /// Number of lines to show from the end.
    #[arg(long, default_value = "100")]
    pub tail: usize,
}

/// Execute a command in a running instance.
#[derive(Args)]
pub struct ExecCommand {
    /// Instance ID or name.
    pub instance: String,

    /// Command to execute.
    #[arg(last = true)]
    pub command: Vec<String>,
}

/// Stop a running instance.
#[derive(Args)]
pub struct StopCommand {
    /// Instance ID(s) or name(s).
    pub instances: Vec<String>,
}

/// Remove a stopped instance.
#[derive(Args)]
pub struct RmCommand {
    /// Instance ID(s) or name(s).
    pub instances: Vec<String>,

    /// Force removal (stop if running, then remove).
    #[arg(short, long)]
    pub force: bool,
}

/// Pull an image from a registry.
#[derive(Args)]
pub struct PullCommand {
    /// Image reference (e.g. nginx:latest, ghcr.io/user/app:v1).
    pub image: String,
}

/// Inspect an instance (show details + energy).
#[derive(Args)]
pub struct InspectCommand {
    /// Instance ID or name.
    pub instance: String,
}

/// List cached images.
#[derive(Args)]
pub struct ImagesCommand;

/// List known LLM model profiles with energy estimates.
#[derive(Args)]
pub struct ModelsCommand;

pub async fn execute_ps(cmd: PsCommand, _config: &Config, output: &Output) -> Result<()> {
    use joule_runtime::daemon_client::DaemonClient;

    let daemon_client = DaemonClient::default();
    if !daemon_client.is_daemon_running() {
        output.warning("Daemon is not running. Start it with: jouledb daemon start");
        return Ok(());
    }

    let instances = daemon_client
        .list_instances()
        .await
        .map_err(|e| crate::error::CliError::Runtime(e.to_string()))?;

    if instances.is_empty() {
        output.info("No instances running.");
        return Ok(());
    }

    // Filter by label if specified
    let label_filters: Vec<(&str, &str)> =
        cmd.label.iter().filter_map(|l| l.split_once('=')).collect();

    let filtered: Vec<_> = instances
        .iter()
        .filter(|inst| {
            // State filter
            if !cmd.all {
                if !matches!(
                    inst.state,
                    joule_runtime::InstanceState::Running | joule_runtime::InstanceState::Starting
                ) {
                    return false;
                }
            }
            // Label filter
            for (key, value) in &label_filters {
                match inst.labels.get(*key) {
                    Some(v) if v == value => {}
                    _ => return false,
                }
            }
            true
        })
        .collect();

    if filtered.is_empty() {
        output.info("No matching instances.");
        return Ok(());
    }

    // Print table header
    eprintln!(
        "{:<12} {:<20} {:<18} {:<10} {:<12} {}",
        "ID".bold(),
        "NAME".bold(),
        "WORKLOAD".bold(),
        "STATE".bold(),
        "PORTS".bold(),
        "CREATED".bold(),
    );

    for inst in filtered {
        let short_id = &inst.id.0[..12.min(inst.id.0.len())];
        let ports_str = if inst.ports.is_empty() {
            "-".to_string()
        } else {
            inst.ports
                .iter()
                .map(|p| {
                    if p.host_port == p.instance_port {
                        p.host_port.to_string()
                    } else {
                        format!("{}:{}", p.host_port, p.instance_port)
                    }
                })
                .collect::<Vec<_>>()
                .join(",")
        };

        let state_str = match &inst.state {
            joule_runtime::InstanceState::Running => "running".green().to_string(),
            joule_runtime::InstanceState::Starting => "starting".yellow().to_string(),
            joule_runtime::InstanceState::Stopping => "stopping".yellow().to_string(),
            joule_runtime::InstanceState::Stopped => "stopped".dimmed().to_string(),
            joule_runtime::InstanceState::Failed(r) => format!("failed: {}", &r[..20.min(r.len())])
                .red()
                .to_string(),
        };

        let created = inst.created_at.format("%Y-%m-%d %H:%M").to_string();

        eprintln!(
            "{:<12} {:<20} {:<18} {:<10} {:<12} {}",
            short_id,
            &inst.name[..20.min(inst.name.len())],
            inst.workload.display_name()[..18.min(inst.workload.display_name().len())].to_string(),
            state_str,
            ports_str,
            created.dimmed(),
        );
    }

    Ok(())
}

pub async fn execute_logs(cmd: LogsCommand, _config: &Config, output: &Output) -> Result<()> {
    use joule_runtime::daemon_client::DaemonClient;

    let daemon_client = DaemonClient::default();
    if daemon_client.is_daemon_running() {
        match daemon_client
            .instance_logs(&cmd.instance, Some(cmd.tail))
            .await
        {
            Ok(lines) => {
                for line in &lines {
                    eprintln!("{}", line);
                }
                if lines.is_empty() {
                    output.info("No logs available.");
                }
            }
            Err(e) => output.error(&format!("Failed to get logs: {}", e)),
        }
    } else {
        let manager = create_temp_manager()?;
        let instance_id = resolve_instance(&manager, &cmd.instance)?;
        match manager.logs(&instance_id, Some(cmd.tail)).await {
            Ok(lines) => {
                for line in &lines {
                    eprintln!("{}", line);
                }
                if lines.is_empty() {
                    output.info("No logs available.");
                }
            }
            Err(e) => output.error(&format!("Failed to get logs: {}", e)),
        }
    }

    Ok(())
}

pub async fn execute_exec(cmd: ExecCommand, _config: &Config, output: &Output) -> Result<()> {
    use joule_runtime::daemon_client::DaemonClient;

    if cmd.command.is_empty() {
        return Err(crate::error::CliError::Runtime(
            "no command specified".into(),
        ));
    }

    let daemon_client = DaemonClient::default();
    if daemon_client.is_daemon_running() {
        match daemon_client
            .exec_in_instance(&cmd.instance, cmd.command.clone())
            .await
        {
            Ok(exec_output) => {
                if !exec_output.stdout.is_empty() {
                    print!("{}", exec_output.stdout);
                }
                if !exec_output.stderr.is_empty() {
                    eprint!("{}", exec_output.stderr);
                }
                if exec_output.exit_code != 0 {
                    std::process::exit(exec_output.exit_code);
                }
            }
            Err(e) => output.error(&format!("Exec failed: {}", e)),
        }
    } else {
        let manager = create_temp_manager()?;
        let instance_id = resolve_instance(&manager, &cmd.instance)?;
        match manager.exec(&instance_id, &cmd.command).await {
            Ok(exec_output) => {
                if !exec_output.stdout.is_empty() {
                    print!("{}", exec_output.stdout);
                }
                if !exec_output.stderr.is_empty() {
                    eprint!("{}", exec_output.stderr);
                }
                if exec_output.exit_code != 0 {
                    std::process::exit(exec_output.exit_code);
                }
            }
            Err(e) => output.error(&format!("Exec failed: {}", e)),
        }
    }

    Ok(())
}

pub async fn execute_stop(cmd: StopCommand, _config: &Config, output: &Output) -> Result<()> {
    use joule_runtime::daemon_client::DaemonClient;

    if cmd.instances.is_empty() {
        return Err(crate::error::CliError::Runtime(
            "no instance specified".into(),
        ));
    }

    let daemon_client = DaemonClient::default();
    if daemon_client.is_daemon_running() {
        for instance in &cmd.instances {
            match daemon_client.stop_instance(instance).await {
                Ok(()) => output.success(&format!("Stopped {}", instance)),
                Err(e) => output.error(&format!("Failed to stop {}: {}", instance, e)),
            }
        }
    } else {
        // Direct manager fallback
        let manager = create_temp_manager()?;
        for instance in &cmd.instances {
            let id = resolve_instance(&manager, instance)?;
            match manager.stop_instance(&id).await {
                Ok(()) => output.success(&format!("Stopped {}", instance)),
                Err(e) => output.error(&format!("Failed to stop {}: {}", instance, e)),
            }
        }
    }

    Ok(())
}

pub async fn execute_rm(cmd: RmCommand, _config: &Config, output: &Output) -> Result<()> {
    use joule_runtime::daemon_client::DaemonClient;

    if cmd.instances.is_empty() {
        return Err(crate::error::CliError::Runtime(
            "no instance specified".into(),
        ));
    }

    let daemon_client = DaemonClient::default();
    if daemon_client.is_daemon_running() {
        for instance in &cmd.instances {
            match daemon_client.remove_instance(instance, cmd.force).await {
                Ok(()) => output.success(&format!("Removed {}", instance)),
                Err(e) => {
                    if cmd.force {
                        output.warning(&format!("Could not remove {}: {}", instance, e));
                    } else {
                        output.error(&format!("Failed to remove {}: {}", instance, e));
                    }
                }
            }
        }
    } else {
        output.warning("Daemon is not running. Nothing to remove.");
    }

    Ok(())
}

pub async fn execute_pull(cmd: PullCommand, _config: &Config, output: &Output) -> Result<()> {
    use joule_runtime::daemon_client::DaemonClient;

    output.info(&format!("Pulling image {}...", cmd.image.cyan()));

    let daemon_client = DaemonClient::default();
    if daemon_client.is_daemon_running() {
        match daemon_client.pull_image(&cmd.image).await {
            Ok(msg) => output.success(&msg),
            Err(e) => output.error(&format!("Failed to pull {}: {}", cmd.image, e)),
        }
    } else {
        let manager = create_temp_manager()?;
        match manager.image_store().pull(&cmd.image).await {
            Ok(info) => output.success(&format!("Pulled {} ({})", cmd.image, info.id)),
            Err(e) => output.error(&format!("Failed to pull {}: {}", cmd.image, e)),
        }
    }

    Ok(())
}

pub async fn execute_images(_config: &Config, output: &Output) -> Result<()> {
    use joule_runtime::daemon_client::DaemonClient;

    let daemon_client = DaemonClient::default();
    let images = if daemon_client.is_daemon_running() {
        daemon_client
            .list_images()
            .await
            .map_err(|e| crate::error::CliError::Runtime(e.to_string()))?
    } else {
        let manager = create_temp_manager()?;
        manager.image_store().list().await
    };

    if images.is_empty() {
        output.info("No cached images.");
        return Ok(());
    }

    eprintln!(
        "{:<40} {:<20} {:<12}",
        "IMAGE".bold(),
        "ID".bold(),
        "SIZE".bold(),
    );

    for img in images {
        eprintln!(
            "{:<40} {:<20} {:<12}",
            img.reference,
            &img.id[..12.min(img.id.len())],
            format_size(img.size_bytes),
        );
    }

    Ok(())
}

pub async fn execute_inspect(cmd: InspectCommand, _config: &Config, output: &Output) -> Result<()> {
    use joule_runtime::daemon_client::DaemonClient;

    let daemon_client = DaemonClient::default();
    if daemon_client.is_daemon_running() {
        match daemon_client.inspect_instance(&cmd.instance).await {
            Ok(info) => {
                let json = serde_json::to_string_pretty(&info)
                    .map_err(|e| crate::error::CliError::Runtime(e.to_string()))?;
                eprintln!("{}", json);
            }
            Err(e) => output.error(&format!("Inspect failed: {}", e)),
        }
    } else {
        // Fallback: try local manager
        let manager = create_temp_manager()?;
        let instance_id = resolve_instance(&manager, &cmd.instance)?;
        match manager.get_instance(&instance_id) {
            Some(info) => {
                let json = serde_json::to_string_pretty(&info)
                    .map_err(|e| crate::error::CliError::Runtime(e.to_string()))?;
                eprintln!("{}", json);
            }
            None => output.error(&format!("Instance '{}' not found", cmd.instance)),
        }
    }

    Ok(())
}

pub async fn execute_models(_config: &Config, output: &Output) -> Result<()> {
    use joule_runtime::AcceleratorKind;

    let tmp_dir = dirs::data_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("/tmp"))
        .join("jouledb")
        .join("models");
    let llm_runtime = joule_runtime::llm::LlmRuntime::new(tmp_dir);

    let mut profiles: Vec<_> = llm_runtime.list_profiles().into_iter().collect();
    profiles.sort_by(|a, b| a.model_id.cmp(&b.model_id));

    if profiles.is_empty() {
        output.info("No model profiles registered.");
        return Ok(());
    }

    eprintln!(
        "{:<22} {:<10} {:>8} {:>9} {:>12} {:>12} {:>12}",
        "MODEL".bold(),
        "BACKEND".bold(),
        "PARAMS".bold(),
        "CTX LEN".bold(),
        "J/tok GPU".bold(),
        "J/tok NPU".bold(),
        "VRAM MB".bold(),
    );

    for p in profiles {
        let gpu_j = llm_runtime
            .estimate_joules_per_token(&p.model_id, &AcceleratorKind::GPU)
            .map(|j| format!("{:.4}", j))
            .unwrap_or_else(|| "-".into());
        let npu_j = llm_runtime
            .estimate_joules_per_token(&p.model_id, &AcceleratorKind::NPU)
            .map(|j| format!("{:.4}", j))
            .unwrap_or_else(|| "-".into());

        eprintln!(
            "{:<22} {:<10} {:>7.1}B {:>9} {:>12} {:>12} {:>12}",
            p.model_id,
            p.preferred_backend.to_string(),
            p.params_b,
            p.context_length,
            gpu_j,
            npu_j,
            if p.min_vram_mb > 0 {
                p.min_vram_mb.to_string()
            } else {
                "unified".into()
            },
        );
    }

    Ok(())
}

/// Create a temporary RuntimeManager for direct operations.
fn create_temp_manager() -> Result<joule_runtime::RuntimeManager> {
    let data_dir = dirs::data_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("/tmp"))
        .join("jouledb");
    let config = joule_runtime::RuntimeConfig::default();
    joule_runtime::RuntimeManager::new(config, data_dir)
        .map_err(|e| crate::error::CliError::Runtime(e.to_string()).into())
}

/// Resolve instance by ID prefix or name.
fn resolve_instance(manager: &joule_runtime::RuntimeManager, input: &str) -> Result<String> {
    let instances = manager.list_instances();
    for inst in &instances {
        if inst.id.0.starts_with(input) || inst.name == input {
            return Ok(inst.id.0.clone());
        }
    }
    Err(crate::error::CliError::Runtime(format!("instance '{}' not found", input)).into())
}

/// Human-readable size formatting.
fn format_size(bytes: u64) -> String {
    if bytes == 0 {
        return "0 B".into();
    }
    let units = ["B", "KB", "MB", "GB", "TB"];
    let i = (bytes as f64).log(1024.0) as usize;
    let i = i.min(units.len() - 1);
    let size = bytes as f64 / 1024_f64.powi(i as i32);
    format!("{:.1} {}", size, units[i])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(0), "0 B");
        assert_eq!(format_size(1024), "1.0 KB");
        assert_eq!(format_size(1048576), "1.0 MB");
        assert_eq!(format_size(1073741824), "1.0 GB");
    }
}
