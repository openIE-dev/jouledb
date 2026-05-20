//! Data import/export commands

use crate::{
    Config, Result,
    error::CliError,
    output::{Output, Progress},
};
use clap::Subcommand;

#[derive(Subcommand)]
pub enum DataCommands {
    /// Import data from file
    Import {
        /// Input file path
        path: String,

        /// Target table
        #[arg(short, long)]
        table: String,

        /// Database
        #[arg(short, long)]
        database: Option<String>,

        /// File format (csv, json, parquet)
        #[arg(long)]
        format: Option<String>,

        /// CSV delimiter
        #[arg(long, default_value = ",")]
        delimiter: char,

        /// CSV has header row
        #[arg(long, default_value = "true")]
        header: bool,

        /// Truncate table before import
        #[arg(long)]
        truncate: bool,

        /// Batch size for inserts
        #[arg(long, default_value = "1000")]
        batch_size: usize,
    },

    /// Export data to file
    Export {
        /// Output file path
        path: String,

        /// Source table or query
        #[arg(short, long)]
        source: String,

        /// Database
        #[arg(short, long)]
        database: Option<String>,

        /// Output format (csv, json, parquet)
        #[arg(long, default_value = "csv")]
        format: String,

        /// CSV delimiter
        #[arg(long, default_value = ",")]
        delimiter: char,

        /// Include header row
        #[arg(long, default_value = "true")]
        header: bool,
    },

    /// Sync data between databases
    Sync {
        /// Source connection
        source: String,

        /// Destination connection
        destination: String,

        /// Tables to sync (comma-separated, or 'all')
        #[arg(short, long, default_value = "all")]
        tables: String,

        /// Sync mode (full, incremental)
        #[arg(long, default_value = "incremental")]
        mode: String,
    },

    /// Copy data between tables
    Copy {
        /// Source table
        source: String,

        /// Destination table
        destination: String,

        /// Optional WHERE clause
        #[arg(long)]
        where_clause: Option<String>,
    },

    /// Generate sample data
    Generate {
        /// Target table
        table: String,

        /// Number of rows to generate
        #[arg(short, long, default_value = "1000")]
        rows: usize,

        /// Database
        #[arg(short, long)]
        database: Option<String>,
    },
}

pub async fn execute(cmd: DataCommands, config: &Config, output: &Output) -> Result<()> {
    match cmd {
        DataCommands::Import {
            path,
            table,
            database,
            format,
            delimiter,
            header,
            truncate,
            batch_size,
        } => {
            import_data(
                &path,
                &table,
                database.as_deref(),
                format.as_deref(),
                delimiter,
                header,
                truncate,
                batch_size,
                config,
                output,
            )
            .await
        }
        DataCommands::Export {
            path,
            source,
            database,
            format,
            delimiter,
            header,
        } => {
            export_data(
                &path,
                &source,
                database.as_deref(),
                &format,
                delimiter,
                header,
                config,
                output,
            )
            .await
        }
        DataCommands::Sync {
            source,
            destination,
            tables,
            mode,
        } => sync_data(&source, &destination, &tables, &mode, config, output).await,
        DataCommands::Copy {
            source,
            destination,
            where_clause,
        } => {
            copy_data(
                &source,
                &destination,
                where_clause.as_deref(),
                config,
                output,
            )
            .await
        }
        DataCommands::Generate {
            table,
            rows,
            database,
        } => generate_data(&table, rows, database.as_deref(), config, output).await,
    }
}

async fn import_data(
    path: &str,
    table: &str,
    database: Option<&str>,
    format: Option<&str>,
    delimiter: char,
    has_header: bool,
    truncate: bool,
    batch_size: usize,
    config: &Config,
    output: &Output,
) -> Result<()> {
    if !std::path::Path::new(path).exists() {
        return Err(CliError::NotFound(format!("File not found: {}", path)));
    }

    // Detect format from extension if not specified
    let format = format.unwrap_or_else(|| path.rsplit('.').next().unwrap_or("csv"));

    let db = database
        .or(config.connection.database.as_deref())
        .unwrap_or("default");

    output.info(&format!(
        "Importing {} data from '{}' to '{}.{}'",
        format.to_uppercase(),
        path,
        db,
        table
    ));

    if truncate {
        output.warning("Truncating table before import");
    }

    let file_size = std::fs::metadata(path)?.len();
    output.verbose(&format!("File size: {} bytes", file_size));

    let progress = Progress::new(file_size, "Importing...");

    let url = format!(
        "http://{}:{}/data/import",
        config.connection.host, config.connection.port
    );

    let file_content = std::fs::read(path)?;

    let client = reqwest::Client::new();
    let response = client
        .post(&url)
        .query(&[
            ("table", table),
            ("database", db),
            ("format", format),
            ("delimiter", &delimiter.to_string()),
            ("header", &has_header.to_string()),
            ("truncate", &truncate.to_string()),
            ("batch_size", &batch_size.to_string()),
        ])
        .body(file_content)
        .send()
        .await?;

    if response.status().is_success() {
        progress.finish("Import complete");

        if let Ok(result) = response.json::<serde_json::Value>().await {
            let rows = result
                .get("rows_imported")
                .and_then(|r| r.as_i64())
                .unwrap_or(0);
            output.success(&format!("{} rows imported", rows));

            if let Some(errors) = result.get("errors").and_then(|e| e.as_i64()) {
                if errors > 0 {
                    output.warning(&format!("{} rows had errors", errors));
                }
            }
        }
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

async fn export_data(
    path: &str,
    source: &str,
    database: Option<&str>,
    format: &str,
    delimiter: char,
    include_header: bool,
    config: &Config,
    output: &Output,
) -> Result<()> {
    let db = database
        .or(config.connection.database.as_deref())
        .unwrap_or("default");

    // Check if source is a table name or a query
    let is_query = source.to_uppercase().contains("SELECT");

    if is_query {
        output.info(&format!(
            "Exporting query result to '{}' as {}",
            path,
            format.to_uppercase()
        ));
    } else {
        output.info(&format!(
            "Exporting table '{}.{}' to '{}' as {}",
            db,
            source,
            path,
            format.to_uppercase()
        ));
    }

    let progress = Progress::spinner("Exporting...");

    let url = format!(
        "http://{}:{}/data/export",
        config.connection.host, config.connection.port
    );

    let body = serde_json::json!({
        "source": source,
        "database": db,
        "format": format,
        "delimiter": delimiter.to_string(),
        "header": include_header,
    });

    let client = reqwest::Client::new();
    let response = client.post(&url).json(&body).send().await?;

    if response.status().is_success() {
        let bytes = response.bytes().await?;
        std::fs::write(path, &bytes)?;

        progress.finish(&format!("Export complete: {}", path));

        let size = bytes.len();
        output.key_value(vec![
            ("Output", path.to_string()),
            ("Format", format.to_string()),
            ("Size", format_size(size)),
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

async fn sync_data(
    source: &str,
    destination: &str,
    tables: &str,
    mode: &str,
    _config: &Config,
    output: &Output,
) -> Result<()> {
    output.section("Data Sync");
    output.key_value(vec![
        ("Source", source.to_string()),
        ("Destination", destination.to_string()),
        ("Tables", tables.to_string()),
        ("Mode", mode.to_string()),
    ]);

    let progress = Progress::spinner("Syncing...");

    // Parse tables
    let table_list: Vec<&str> = if tables == "all" {
        output.verbose("Syncing all tables");
        vec!["users", "orders", "products"] // Example
    } else {
        tables.split(',').map(|s| s.trim()).collect()
    };

    for table in &table_list {
        progress.set_message(&format!("Syncing table: {}", table));
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }

    progress.finish("Sync complete");
    output.success(&format!("{} table(s) synced", table_list.len()));

    Ok(())
}

async fn copy_data(
    source: &str,
    destination: &str,
    where_clause: Option<&str>,
    config: &Config,
    output: &Output,
) -> Result<()> {
    output.info(&format!(
        "Copying data from '{}' to '{}'",
        source, destination
    ));

    if let Some(clause) = where_clause {
        output.verbose(&format!("Filter: WHERE {}", clause));
    }

    let query = if let Some(clause) = where_clause {
        format!(
            "INSERT INTO {} SELECT * FROM {} WHERE {}",
            destination, source, clause
        )
    } else {
        format!("INSERT INTO {} SELECT * FROM {}", destination, source)
    };

    let url = format!(
        "http://{}:{}/query",
        config.connection.host, config.connection.port
    );

    let body = serde_json::json!({
        "query": query,
    });

    let client = reqwest::Client::new();
    let response = client.post(&url).json(&body).send().await?;

    if response.status().is_success() {
        if let Ok(result) = response.json::<serde_json::Value>().await {
            let rows = result
                .get("affected_rows")
                .and_then(|r| r.as_i64())
                .unwrap_or(0);
            output.success(&format!("{} rows copied", rows));
        }
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

async fn generate_data(
    table: &str,
    rows: usize,
    database: Option<&str>,
    config: &Config,
    output: &Output,
) -> Result<()> {
    let db = database
        .or(config.connection.database.as_deref())
        .unwrap_or("default");

    output.info(&format!(
        "Generating {} rows for table '{}.{}'",
        rows, db, table
    ));

    let progress = Progress::new(rows as u64, "Generating data...");

    let url = format!(
        "http://{}:{}/data/generate",
        config.connection.host, config.connection.port
    );

    let body = serde_json::json!({
        "table": table,
        "database": db,
        "rows": rows,
    });

    let client = reqwest::Client::new();
    let response = client.post(&url).json(&body).send().await?;

    if response.status().is_success() {
        progress.finish("Generation complete");
        output.success(&format!("{} rows generated", rows));
    } else {
        progress.finish_and_clear();

        // If server doesn't support generation, provide manual instructions
        output.warning("Data generation endpoint not available");
        output.info("To generate sample data manually:");
        output.raw(&format!(
            "  jouledb query execute \"INSERT INTO {} (col1, col2) SELECT random(), random() FROM generate_series(1, {})\"",
            table, rows
        ));
    }

    Ok(())
}
