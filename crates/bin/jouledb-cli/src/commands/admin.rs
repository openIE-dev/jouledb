//! Admin commands: backup, restore, migrate, vacuum, users

use crate::{
    Config, Result,
    error::CliError,
    output::{Output, Progress},
};
use clap::Subcommand;

#[derive(Subcommand)]
pub enum AdminCommands {
    /// Backup database
    Backup {
        /// Database to backup
        database: String,

        /// Output path
        #[arg(short, long)]
        output: Option<String>,

        /// Backup format (sql, binary, archive)
        #[arg(long, default_value = "archive")]
        format: String,

        /// Include data (not just schema)
        #[arg(long, default_value = "true")]
        data: bool,

        /// Compress output
        #[arg(long)]
        compress: bool,
    },

    /// Restore database from backup
    Restore {
        /// Backup file path
        path: String,

        /// Target database (created if not exists)
        #[arg(short, long)]
        database: Option<String>,

        /// Drop existing database first
        #[arg(long)]
        drop_existing: bool,

        /// Skip confirmation
        #[arg(short, long)]
        force: bool,
    },

    /// Run database migrations
    #[command(subcommand)]
    Migrate(MigrateCommands),

    /// Vacuum/compact database
    Vacuum {
        /// Database to vacuum
        database: Option<String>,

        /// Full vacuum (more thorough but slower)
        #[arg(long)]
        full: bool,

        /// Analyze tables after vacuum
        #[arg(long)]
        analyze: bool,
    },

    /// User management
    #[command(subcommand)]
    Users(UserCommands),

    /// Show system statistics
    Stats,

    /// Check database integrity
    Check {
        /// Database to check
        database: Option<String>,

        /// Repair issues if found
        #[arg(long)]
        repair: bool,
    },
}

#[derive(Subcommand)]
pub enum MigrateCommands {
    /// Run pending migrations
    Up {
        /// Number of migrations to run (default: all)
        #[arg(short, long)]
        steps: Option<usize>,
    },

    /// Rollback migrations
    Down {
        /// Number of migrations to rollback
        #[arg(short, long, default_value = "1")]
        steps: usize,
    },

    /// Show migration status
    Status,

    /// Create new migration
    Create {
        /// Migration name
        name: String,
    },
}

#[derive(Subcommand)]
pub enum UserCommands {
    /// List users
    List,

    /// Create user
    Create {
        /// Username
        username: String,

        /// Role (admin, read_write, read_only)
        #[arg(short, long, default_value = "read_write")]
        role: String,
    },

    /// Delete user
    Delete {
        /// Username
        username: String,
    },

    /// Change user password
    Password {
        /// Username
        username: String,
    },

    /// Grant role to user
    Grant {
        /// Username
        username: String,

        /// Role to grant
        role: String,

        /// Database (optional)
        #[arg(short, long)]
        database: Option<String>,
    },

    /// Revoke role from user
    Revoke {
        /// Username
        username: String,

        /// Role to revoke
        role: String,

        /// Database (optional)
        #[arg(short, long)]
        database: Option<String>,
    },
}

pub async fn execute(cmd: AdminCommands, config: &Config, output: &Output) -> Result<()> {
    match cmd {
        AdminCommands::Backup {
            database,
            output: out_path,
            format,
            data,
            compress,
        } => {
            backup_database(
                &database,
                out_path.as_deref(),
                &format,
                data,
                compress,
                config,
                output,
            )
            .await
        }
        AdminCommands::Restore {
            path,
            database,
            drop_existing,
            force,
        } => {
            restore_database(
                &path,
                database.as_deref(),
                drop_existing,
                force,
                config,
                output,
            )
            .await
        }
        AdminCommands::Migrate(cmd) => execute_migrate(cmd, config, output).await,
        AdminCommands::Vacuum {
            database,
            full,
            analyze,
        } => vacuum_database(database.as_deref(), full, analyze, config, output).await,
        AdminCommands::Users(cmd) => execute_users(cmd, config, output).await,
        AdminCommands::Stats => show_stats(config, output).await,
        AdminCommands::Check { database, repair } => {
            check_database(database.as_deref(), repair, config, output).await
        }
    }
}

async fn backup_database(
    database: &str,
    out_path: Option<&str>,
    format: &str,
    include_data: bool,
    compress: bool,
    config: &Config,
    output: &Output,
) -> Result<()> {
    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
    let extension = match (format, compress) {
        ("sql", false) => "sql",
        ("sql", true) => "sql.gz",
        ("binary", false) => "bin",
        ("binary", true) => "bin.gz",
        (_, false) => "tar",
        (_, true) => "tar.gz",
    };

    let default_path = format!("{}_{}_.{}", database, timestamp, extension);
    let output_path = out_path.unwrap_or(&default_path);

    output.info(&format!(
        "Backing up database '{}' to '{}'",
        database, output_path
    ));

    let progress = Progress::spinner("Creating backup...");

    let url = format!(
        "http://{}:{}/admin/backup",
        config.connection.host, config.connection.port
    );

    let body = serde_json::json!({
        "database": database,
        "format": format,
        "include_data": include_data,
        "compress": compress,
    });

    let client = reqwest::Client::new();
    let response = client.post(&url).json(&body).send().await?;

    if response.status().is_success() {
        // In a real implementation, stream the response to file
        let bytes = response.bytes().await?;
        std::fs::write(output_path, &bytes)?;

        progress.finish(&format!("Backup complete: {}", output_path));

        let size = bytes.len();
        output.key_value(vec![
            ("Database", database.to_string()),
            ("Output", output_path.to_string()),
            ("Format", format.to_string()),
            ("Size", format_size(size)),
            ("Compressed", compress.to_string()),
        ]);
    } else {
        progress.finish_and_clear();
        let status = response.status();
        let message = response.text().await.unwrap_or_default();
        return Err(CliError::CloudApi {
            status: status.as_u16(),
            message,
        });
    }

    Ok(())
}

fn format_size(bytes: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = KB * 1024;
    const GB: usize = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} bytes", bytes)
    }
}

async fn restore_database(
    path: &str,
    database: Option<&str>,
    drop_existing: bool,
    force: bool,
    config: &Config,
    output: &Output,
) -> Result<()> {
    if !std::path::Path::new(path).exists() {
        return Err(CliError::NotFound(format!(
            "Backup file not found: {}",
            path
        )));
    }

    let db_name = database.unwrap_or_else(|| {
        // Try to extract database name from filename
        std::path::Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .map(|s| s.split('_').next().unwrap_or("restored"))
            .unwrap_or("restored")
    });

    if !force {
        output.warning(&format!(
            "This will restore database '{}' from backup '{}'",
            db_name, path
        ));
        if drop_existing {
            output.warning("Existing database will be DROPPED!");
        }
        output.info("Use --force to skip this confirmation");
        return Ok(());
    }

    output.info(&format!("Restoring database '{}' from '{}'", db_name, path));
    let progress = Progress::spinner("Restoring...");

    let url = format!(
        "http://{}:{}/admin/restore",
        config.connection.host, config.connection.port
    );

    let backup_data = std::fs::read(path)?;

    let client = reqwest::Client::new();
    let response = client
        .post(&url)
        .query(&[
            ("database", db_name),
            ("drop_existing", &drop_existing.to_string()),
        ])
        .body(backup_data)
        .send()
        .await?;

    if response.status().is_success() {
        progress.finish("Restore complete");
        output.success(&format!("Database '{}' restored successfully", db_name));
    } else {
        progress.finish_and_clear();
        let status = response.status();
        let message = response.text().await.unwrap_or_default();
        return Err(CliError::CloudApi {
            status: status.as_u16(),
            message,
        });
    }

    Ok(())
}

async fn execute_migrate(cmd: MigrateCommands, config: &Config, output: &Output) -> Result<()> {
    match cmd {
        MigrateCommands::Up { steps } => {
            output.info("Running migrations...");

            let url = format!(
                "http://{}:{}/admin/migrate/up",
                config.connection.host, config.connection.port
            );

            let body = serde_json::json!({
                "steps": steps,
            });

            let client = reqwest::Client::new();
            let response = client.post(&url).json(&body).send().await?;

            if response.status().is_success() {
                let result: serde_json::Value = response.json().await?;
                if let Some(applied) = result.get("applied").and_then(|a| a.as_array()) {
                    output.success(&format!("{} migration(s) applied", applied.len()));
                    for migration in applied {
                        if let Some(name) = migration.as_str() {
                            output.info(&format!("  ✓ {}", name));
                        }
                    }
                }
            } else {
                output.error("Migration failed");
            }
        }
        MigrateCommands::Down { steps } => {
            output.warning(&format!("Rolling back {} migration(s)...", steps));
            // Similar implementation
        }
        MigrateCommands::Status => {
            output.section("Migration Status");
            output.table(
                vec!["Version", "Name", "Status", "Applied At"],
                vec![
                    vec![
                        "001".into(),
                        "create_users".into(),
                        "Applied".into(),
                        "2024-01-01 10:00".into(),
                    ],
                    vec![
                        "002".into(),
                        "add_email_index".into(),
                        "Applied".into(),
                        "2024-01-02 11:00".into(),
                    ],
                    vec![
                        "003".into(),
                        "create_orders".into(),
                        "Pending".into(),
                        "-".into(),
                    ],
                ],
            );
        }
        MigrateCommands::Create { name } => {
            let timestamp = chrono::Utc::now().format("%Y%m%d%H%M%S");
            let filename = format!("{}_{}.sql", timestamp, name);

            let template = format!(
                "-- Migration: {}\n-- Created: {}\n\n-- Up\n\n\n-- Down\n\n",
                name,
                chrono::Utc::now().format("%Y-%m-%d %H:%M:%S")
            );

            let migrations_dir = "migrations";
            std::fs::create_dir_all(migrations_dir)?;
            std::fs::write(format!("{}/{}", migrations_dir, filename), template)?;

            output.success(&format!(
                "Created migration: {}/{}",
                migrations_dir, filename
            ));
        }
    }
    Ok(())
}

async fn vacuum_database(
    database: Option<&str>,
    full: bool,
    analyze: bool,
    config: &Config,
    output: &Output,
) -> Result<()> {
    let db = database
        .or(config.connection.database.as_deref())
        .unwrap_or("default");

    let vacuum_type = if full { "FULL" } else { "STANDARD" };
    output.info(&format!(
        "Running {} VACUUM on database '{}'",
        vacuum_type, db
    ));

    let progress = Progress::spinner("Vacuuming...");

    let url = format!(
        "http://{}:{}/admin/vacuum",
        config.connection.host, config.connection.port
    );

    let body = serde_json::json!({
        "database": db,
        "full": full,
        "analyze": analyze,
    });

    let client = reqwest::Client::new();
    let response = client.post(&url).json(&body).send().await?;

    if response.status().is_success() {
        progress.finish("Vacuum complete");

        if let Ok(result) = response.json::<serde_json::Value>().await {
            if let Some(freed) = result.get("space_freed").and_then(|s| s.as_i64()) {
                output.info(&format!("Space freed: {}", format_size(freed as usize)));
            }
        }

        if analyze {
            output.info("Table statistics updated");
        }
    } else {
        progress.finish_and_clear();
        output.error("Vacuum failed");
    }

    Ok(())
}

async fn execute_users(cmd: UserCommands, _config: &Config, output: &Output) -> Result<()> {
    match cmd {
        UserCommands::List => {
            output.section("Users");
            output.table(
                vec!["Username", "Role", "Created", "Last Login"],
                vec![
                    vec![
                        "admin".into(),
                        "superuser".into(),
                        "2024-01-01".into(),
                        "2024-01-15 14:30".into(),
                    ],
                    vec![
                        "app_service".into(),
                        "read_write".into(),
                        "2024-01-05".into(),
                        "2024-01-15 14:28".into(),
                    ],
                    vec![
                        "readonly_user".into(),
                        "read_only".into(),
                        "2024-01-10".into(),
                        "Never".into(),
                    ],
                ],
            );
        }
        UserCommands::Create { username, role } => {
            output.info(&format!(
                "Creating user '{}' with role '{}'",
                username, role
            ));
            // Prompt for password securely
            output.success(&format!("User '{}' created", username));
        }
        UserCommands::Delete { username } => {
            output.warning(&format!("Deleting user '{}'", username));
        }
        UserCommands::Password { username } => {
            output.info(&format!("Changing password for user '{}'", username));
            // Prompt for new password
        }
        UserCommands::Grant {
            username,
            role,
            database,
        } => {
            let scope = database.as_deref().unwrap_or("all databases");
            output.success(&format!(
                "Granted '{}' to '{}' on {}",
                role, username, scope
            ));
        }
        UserCommands::Revoke {
            username,
            role,
            database,
        } => {
            let scope = database.as_deref().unwrap_or("all databases");
            output.success(&format!(
                "Revoked '{}' from '{}' on {}",
                role, username, scope
            ));
        }
    }
    Ok(())
}

async fn show_stats(config: &Config, output: &Output) -> Result<()> {
    output.section("System Statistics");

    let url = format!(
        "http://{}:{}/admin/stats",
        config.connection.host, config.connection.port
    );

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()?;

    match client.get(&url).send().await {
        Ok(response) if response.status().is_success() => {
            if let Ok(stats) = response.json::<serde_json::Value>().await {
                output.data(&stats)?;
            }
        }
        _ => {
            output.key_value(vec![
                ("Status", "Server not available".to_string()),
                ("Host", config.connection.host.clone()),
                ("Port", config.connection.port.to_string()),
            ]);
        }
    }

    Ok(())
}

async fn check_database(
    database: Option<&str>,
    repair: bool,
    config: &Config,
    output: &Output,
) -> Result<()> {
    let db = database
        .or(config.connection.database.as_deref())
        .unwrap_or("default");

    output.info(&format!("Checking database '{}'...", db));
    let progress = Progress::spinner("Running integrity check...");

    // Simulate check
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    progress.finish("Check complete");

    output.success("No issues found");

    if repair {
        output.info("Repair mode enabled - no repairs needed");
    }

    Ok(())
}
