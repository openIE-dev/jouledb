//! Configuration commands

use crate::{Config, Result, error::CliError, output::Output};
use clap::Subcommand;

#[derive(Subcommand)]
pub enum ConfigCommands {
    /// Initialize configuration
    Init {
        /// Force overwrite existing config
        #[arg(short, long)]
        force: bool,
    },

    /// Get a configuration value
    Get {
        /// Configuration key (e.g., connection.host)
        key: String,
    },

    /// Set a configuration value
    Set {
        /// Configuration key
        key: String,

        /// Value to set
        value: String,
    },

    /// List all configuration values
    List,

    /// Show configuration file path
    Path,

    /// Edit configuration in default editor
    Edit,

    /// Reset configuration to defaults
    Reset {
        /// Skip confirmation
        #[arg(short, long)]
        force: bool,
    },
}

pub async fn execute(cmd: ConfigCommands, config: &Config, output: &Output) -> Result<()> {
    match cmd {
        ConfigCommands::Init { force } => init_config(force, output).await,
        ConfigCommands::Get { key } => get_config(&key, config, output).await,
        ConfigCommands::Set { key, value } => set_config(&key, &value, config, output).await,
        ConfigCommands::List => list_config(config, output).await,
        ConfigCommands::Path => show_path(output).await,
        ConfigCommands::Edit => edit_config(output).await,
        ConfigCommands::Reset { force } => reset_config(force, output).await,
    }
}

async fn init_config(force: bool, output: &Output) -> Result<()> {
    let config_path = Config::default_config_path()?;

    if config_path.exists() && !force {
        output.warning(&format!(
            "Configuration already exists at: {}",
            config_path.display()
        ));
        output.info("Use --force to overwrite");
        return Ok(());
    }

    // Create parent directories
    if let Some(parent) = config_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Create default config
    let default_config = Config::default();
    default_config.save(None)?;

    output.success(&format!(
        "Configuration initialized at: {}",
        config_path.display()
    ));

    output.raw("");
    output.info("Default settings:");
    output.key_value(vec![
        ("Host", default_config.connection.host),
        ("Port", default_config.connection.port.to_string()),
        ("Cloud API", default_config.cloud.api_url),
    ]);

    output.raw("");
    output.info("Edit with: jouledb config edit");

    Ok(())
}

async fn get_config(key: &str, config: &Config, output: &Output) -> Result<()> {
    let value = match key {
        "connection.host" => Some(config.connection.host.clone()),
        "connection.port" => Some(config.connection.port.to_string()),
        "connection.database" => config.connection.database.clone(),
        "connection.username" => config.connection.username.clone(),
        "connection.tls" => Some(config.connection.tls.to_string()),
        "connection.timeout" => Some(config.connection.timeout.to_string()),
        "cloud.api_url" => Some(config.cloud.api_url.clone()),
        "cloud.project_id" => config.cloud.project_id.clone(),
        "cloud.cluster_id" => config.cloud.cluster_id.clone(),
        "output.format" => Some(config.output.format.clone()),
        "output.colors" => Some(config.output.colors.to_string()),
        "shell.prompt" => Some(config.shell.prompt.clone()),
        "shell.autocomplete" => Some(config.shell.autocomplete.to_string()),
        "shell.history_size" => Some(config.shell.history_size.to_string()),
        _ => None,
    };

    match value {
        Some(v) => output.raw(&v),
        None => {
            return Err(CliError::InvalidInput(format!(
                "Unknown config key: {}",
                key
            )));
        }
    }

    Ok(())
}

async fn set_config(key: &str, value: &str, config: &Config, output: &Output) -> Result<()> {
    let mut new_config = config.clone();

    match key {
        "connection.host" => new_config.connection.host = value.to_string(),
        "connection.port" => {
            new_config.connection.port = value
                .parse()
                .map_err(|_| CliError::InvalidInput("Invalid port number".into()))?;
        }
        "connection.database" => new_config.connection.database = Some(value.to_string()),
        "connection.username" => new_config.connection.username = Some(value.to_string()),
        "connection.tls" => {
            new_config.connection.tls = value
                .parse()
                .map_err(|_| CliError::InvalidInput("Invalid boolean".into()))?;
        }
        "connection.timeout" => {
            new_config.connection.timeout = value
                .parse()
                .map_err(|_| CliError::InvalidInput("Invalid timeout".into()))?;
        }
        "cloud.api_url" => new_config.cloud.api_url = value.to_string(),
        "cloud.project_id" => new_config.cloud.project_id = Some(value.to_string()),
        "cloud.cluster_id" => new_config.cloud.cluster_id = Some(value.to_string()),
        "output.format" => new_config.output.format = value.to_string(),
        "output.colors" => {
            new_config.output.colors = value
                .parse()
                .map_err(|_| CliError::InvalidInput("Invalid boolean".into()))?;
        }
        "shell.prompt" => new_config.shell.prompt = value.to_string(),
        "shell.autocomplete" => {
            new_config.shell.autocomplete = value
                .parse()
                .map_err(|_| CliError::InvalidInput("Invalid boolean".into()))?;
        }
        "shell.history_size" => {
            new_config.shell.history_size = value
                .parse()
                .map_err(|_| CliError::InvalidInput("Invalid number".into()))?;
        }
        _ => {
            return Err(CliError::InvalidInput(format!(
                "Unknown config key: {}",
                key
            )));
        }
    }

    new_config.save(None)?;
    output.success(&format!("Set {} = {}", key, value));

    Ok(())
}

async fn list_config(config: &Config, output: &Output) -> Result<()> {
    output.section("Connection");
    output.key_value(vec![
        ("host", config.connection.host.clone()),
        ("port", config.connection.port.to_string()),
        (
            "database",
            config
                .connection
                .database
                .clone()
                .unwrap_or_else(|| "(not set)".into()),
        ),
        (
            "username",
            config
                .connection
                .username
                .clone()
                .unwrap_or_else(|| "(not set)".into()),
        ),
        ("tls", config.connection.tls.to_string()),
        ("timeout", format!("{}s", config.connection.timeout)),
    ]);

    output.section("Cloud");
    output.key_value(vec![
        ("api_url", config.cloud.api_url.clone()),
        (
            "project_id",
            config
                .cloud
                .project_id
                .clone()
                .unwrap_or_else(|| "(not set)".into()),
        ),
        (
            "cluster_id",
            config
                .cloud
                .cluster_id
                .clone()
                .unwrap_or_else(|| "(not set)".into()),
        ),
    ]);

    output.section("Output");
    output.key_value(vec![
        ("format", config.output.format.clone()),
        ("colors", config.output.colors.to_string()),
        ("table_style", config.output.table_style.clone()),
    ]);

    output.section("Shell");
    output.key_value(vec![
        ("prompt", config.shell.prompt.clone()),
        ("autocomplete", config.shell.autocomplete.to_string()),
        (
            "syntax_highlighting",
            config.shell.syntax_highlighting.to_string(),
        ),
        ("history_size", config.shell.history_size.to_string()),
    ]);

    Ok(())
}

async fn show_path(output: &Output) -> Result<()> {
    let config_path = Config::default_config_path()?;
    let creds_path = Config::credentials_path()?;

    output.key_value(vec![
        ("Config", config_path.display().to_string()),
        ("Credentials", creds_path.display().to_string()),
    ]);

    Ok(())
}

async fn edit_config(output: &Output) -> Result<()> {
    let config_path = Config::default_config_path()?;

    // Ensure config exists
    if !config_path.exists() {
        output.info("Creating default configuration...");
        let default_config = Config::default();
        default_config.save(None)?;
    }

    // Try to open in editor
    let editor = std::env::var("EDITOR").unwrap_or_else(|_| "nano".to_string());

    output.info(&format!(
        "Opening {} with {}...",
        config_path.display(),
        editor
    ));

    let status = std::process::Command::new(&editor)
        .arg(&config_path)
        .status();

    match status {
        Ok(s) if s.success() => {
            output.success("Configuration saved");
        }
        Ok(_) => {
            output.warning("Editor exited with non-zero status");
        }
        Err(_) => {
            output.error(&format!("Could not open editor: {}", editor));
            output.info(&format!("Edit manually: {}", config_path.display()));
        }
    }

    Ok(())
}

async fn reset_config(force: bool, output: &Output) -> Result<()> {
    if !force {
        output.warning("This will reset all configuration to defaults!");
        output.info("Use --force to confirm");
        return Ok(());
    }

    let config_path = Config::default_config_path()?;

    if config_path.exists() {
        std::fs::remove_file(&config_path)?;
    }

    let default_config = Config::default();
    default_config.save(None)?;

    output.success("Configuration reset to defaults");

    Ok(())
}
