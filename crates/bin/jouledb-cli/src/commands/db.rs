//! Database management commands

use crate::{Config, Result, error::CliError, output::Output};
use clap::Subcommand;

#[derive(Subcommand)]
pub enum DbCommands {
    /// Create a new database
    Create {
        /// Database name
        name: String,

        /// Owner username
        #[arg(short, long)]
        owner: Option<String>,

        /// Character encoding
        #[arg(long, default_value = "UTF8")]
        encoding: String,
    },

    /// Drop a database
    Drop {
        /// Database name
        name: String,

        /// Skip confirmation
        #[arg(short, long)]
        force: bool,
    },

    /// List all databases
    List,

    /// Show database information
    Info {
        /// Database name
        name: String,
    },

    /// Connect to a database (set as default)
    Use {
        /// Database name
        name: String,
    },

    /// Show current database
    Current,

    /// Rename a database
    Rename {
        /// Current name
        old_name: String,

        /// New name
        new_name: String,
    },
}

pub async fn execute(cmd: DbCommands, config: &Config, output: &Output) -> Result<()> {
    match cmd {
        DbCommands::Create {
            name,
            owner,
            encoding,
        } => create_database(&name, owner.as_deref(), &encoding, config, output).await,
        DbCommands::Drop { name, force } => drop_database(&name, force, config, output).await,
        DbCommands::List => list_databases(config, output).await,
        DbCommands::Info { name } => show_database_info(&name, config, output).await,
        DbCommands::Use { name } => use_database(&name, config, output).await,
        DbCommands::Current => show_current_database(config, output).await,
        DbCommands::Rename { old_name, new_name } => {
            rename_database(&old_name, &new_name, config, output).await
        }
    }
}

async fn create_database(
    name: &str,
    owner: Option<&str>,
    encoding: &str,
    config: &Config,
    output: &Output,
) -> Result<()> {
    output.info(&format!("Creating database '{}'...", name));

    let url = format!(
        "http://{}:{}/databases",
        config.connection.host, config.connection.port
    );

    let body = serde_json::json!({
        "name": name,
        "owner": owner,
        "encoding": encoding,
    });

    let client = reqwest::Client::new();
    let response = client.post(&url).json(&body).send().await?;

    if response.status().is_success() {
        output.success(&format!("Database '{}' created successfully", name));
    } else {
        let status = response.status();
        let message = response.text().await.unwrap_or_default();
        return Err(CliError::CloudApi {
            status: status.as_u16(),
            message,
        });
    }

    Ok(())
}

async fn drop_database(name: &str, force: bool, config: &Config, output: &Output) -> Result<()> {
    if !force {
        output.warning(&format!(
            "This will permanently delete database '{}' and all its data!",
            name
        ));

        // In a real implementation, prompt for confirmation
        output.info("Use --force to skip this confirmation");
        return Ok(());
    }

    output.info(&format!("Dropping database '{}'...", name));

    let url = format!(
        "http://{}:{}/databases/{}",
        config.connection.host, config.connection.port, name
    );

    let client = reqwest::Client::new();
    let response = client.delete(&url).send().await?;

    if response.status().is_success() {
        output.success(&format!("Database '{}' dropped successfully", name));
    } else {
        let status = response.status();
        let message = response.text().await.unwrap_or_default();
        return Err(CliError::CloudApi {
            status: status.as_u16(),
            message,
        });
    }

    Ok(())
}

async fn list_databases(config: &Config, output: &Output) -> Result<()> {
    output.verbose("Fetching database list...");

    let url = format!(
        "http://{}:{}/databases",
        config.connection.host, config.connection.port
    );

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()?;

    match client.get(&url).send().await {
        Ok(response) if response.status().is_success() => {
            if let Ok(databases) = response.json::<Vec<serde_json::Value>>().await {
                if databases.is_empty() {
                    output.info("No databases found");
                } else {
                    let mut rows = Vec::new();
                    for db in databases {
                        rows.push(vec![
                            db.get("name")
                                .and_then(|v| v.as_str())
                                .unwrap_or("?")
                                .to_string(),
                            db.get("owner")
                                .and_then(|v| v.as_str())
                                .unwrap_or("-")
                                .to_string(),
                            db.get("size")
                                .and_then(|v| v.as_str())
                                .unwrap_or("-")
                                .to_string(),
                            db.get("tables")
                                .and_then(|v| v.as_i64())
                                .map(|v| v.to_string())
                                .unwrap_or("-".to_string()),
                        ]);
                    }
                    output.table(vec!["Name", "Owner", "Size", "Tables"], rows);
                }
            }
        }
        _ => {
            // Show default/mock response when server is not available
            output.warning("Could not connect to server. Showing example output:");
            output.table(
                vec!["Name", "Owner", "Size", "Tables"],
                vec![
                    vec![
                        "default".into(),
                        "admin".into(),
                        "128 MB".into(),
                        "12".into(),
                    ],
                    vec![
                        "analytics".into(),
                        "analytics_user".into(),
                        "2.4 GB".into(),
                        "45".into(),
                    ],
                ],
            );
        }
    }

    Ok(())
}

async fn show_database_info(name: &str, config: &Config, output: &Output) -> Result<()> {
    output.section(&format!("Database: {}", name));

    let url = format!(
        "http://{}:{}/databases/{}",
        config.connection.host, config.connection.port, name
    );

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()?;

    match client.get(&url).send().await {
        Ok(response) if response.status().is_success() => {
            if let Ok(db) = response.json::<serde_json::Value>().await {
                output.data(&db)?;
            }
        }
        _ => {
            output.key_value(vec![
                ("Name", name.to_string()),
                ("Status", "Unknown (server not available)".to_string()),
            ]);
        }
    }

    Ok(())
}

async fn use_database(name: &str, config: &Config, output: &Output) -> Result<()> {
    // Update the config to set the default database
    let mut new_config = config.clone();
    new_config.connection.database = Some(name.to_string());
    new_config.save(None)?;

    output.success(&format!("Now using database '{}'", name));
    Ok(())
}

async fn show_current_database(config: &Config, output: &Output) -> Result<()> {
    match &config.connection.database {
        Some(db) => {
            output.info(&format!("Current database: {}", db));
        }
        None => {
            output.info("No database selected. Use 'jouledb db use <name>' to select one.");
        }
    }
    Ok(())
}

async fn rename_database(
    old_name: &str,
    new_name: &str,
    config: &Config,
    output: &Output,
) -> Result<()> {
    output.info(&format!(
        "Renaming database '{}' to '{}'...",
        old_name, new_name
    ));

    let url = format!(
        "http://{}:{}/databases/{}",
        config.connection.host, config.connection.port, old_name
    );

    let body = serde_json::json!({
        "name": new_name,
    });

    let client = reqwest::Client::new();
    let response = client.patch(&url).json(&body).send().await?;

    if response.status().is_success() {
        output.success(&format!("Database renamed to '{}'", new_name));
    } else {
        let status = response.status();
        let message = response.text().await.unwrap_or_default();
        return Err(CliError::CloudApi {
            status: status.as_u16(),
            message,
        });
    }

    Ok(())
}
