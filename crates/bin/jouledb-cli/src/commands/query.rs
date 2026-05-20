//! Query execution commands

use crate::{Config, Result, error::CliError, output::Output};
use clap::Subcommand;

#[derive(Subcommand)]
pub enum QueryCommands {
    /// Execute a SQL query
    Execute {
        /// SQL query to execute
        query: String,

        /// Database to use
        #[arg(short, long)]
        database: Option<String>,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Execute SQL from a file
    File {
        /// Path to SQL file
        path: String,

        /// Database to use
        #[arg(short, long)]
        database: Option<String>,
    },

    /// Explain query execution plan
    Explain {
        /// SQL query
        query: String,

        /// Show detailed analysis
        #[arg(long)]
        analyze: bool,
    },

    /// Open interactive SQL shell
    Shell {
        /// Database to connect to
        #[arg(short, long)]
        database: Option<String>,
    },
}

pub async fn execute(cmd: QueryCommands, config: &Config, output: &Output) -> Result<()> {
    match cmd {
        QueryCommands::Execute {
            query,
            database,
            json,
        } => execute_query(&query, database.as_deref(), json, config, output).await,
        QueryCommands::File { path, database } => {
            execute_file(&path, database.as_deref(), config, output).await
        }
        QueryCommands::Explain { query, analyze } => {
            explain_query(&query, analyze, config, output).await
        }
        QueryCommands::Shell { database } => {
            execute_shell(config, output, database.as_deref()).await
        }
    }
}

async fn execute_query(
    query: &str,
    database: Option<&str>,
    json_output: bool,
    config: &Config,
    output: &Output,
) -> Result<()> {
    let db = database
        .or(config.connection.database.as_deref())
        .unwrap_or("default");

    output.verbose(&format!("Executing query on database '{}'", db));
    output.verbose(&format!("Query: {}", query));

    let url = format!(
        "http://{}:{}/query",
        config.connection.host, config.connection.port
    );

    let body = serde_json::json!({
        "query": query,
        "database": db,
    });

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(config.connection.timeout))
        .build()?;

    let response = client.post(&url).json(&body).send().await?;

    if response.status().is_success() {
        let result: serde_json::Value = response.json().await?;

        if json_output {
            output.data(&result)?;
        } else {
            // Try to format as table
            if let Some(rows) = result.get("rows").and_then(|r| r.as_array()) {
                if let Some(columns) = result.get("columns").and_then(|c| c.as_array()) {
                    let headers: Vec<&str> = columns.iter().filter_map(|c| c.as_str()).collect();

                    let table_rows: Vec<Vec<String>> = rows
                        .iter()
                        .map(|row| {
                            if let Some(arr) = row.as_array() {
                                arr.iter().map(|v| format_value(v)).collect()
                            } else {
                                vec![]
                            }
                        })
                        .collect();

                    output.table(headers, table_rows);

                    // Show row count
                    if let Some(count) = result.get("row_count").and_then(|c| c.as_i64()) {
                        output.info(&format!("{} row(s) returned", count));
                    }
                } else {
                    output.data(&result)?;
                }
            } else {
                // Non-SELECT query (INSERT, UPDATE, DELETE)
                if let Some(affected) = result.get("affected_rows").and_then(|a| a.as_i64()) {
                    output.success(&format!("{} row(s) affected", affected));
                } else {
                    output.success("Query executed successfully");
                }
            }
        }

        // Show execution time if available
        if let Some(time_ms) = result.get("execution_time_ms").and_then(|t| t.as_f64()) {
            output.verbose(&format!("Execution time: {:.2}ms", time_ms));
        }
    } else {
        let status = response.status();
        let error_body = response.text().await.unwrap_or_default();
        return Err(CliError::Query(format!("{}: {}", status, error_body)));
    }

    Ok(())
}

fn format_value(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => "NULL".to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::String(s) => s.clone(),
        _ => value.to_string(),
    }
}

async fn execute_file(
    path: &str,
    database: Option<&str>,
    config: &Config,
    output: &Output,
) -> Result<()> {
    output.info(&format!("Executing SQL file: {}", path));

    let content = std::fs::read_to_string(path)?;

    // Split by semicolons (simple approach - doesn't handle semicolons in strings)
    let queries: Vec<&str> = content
        .split(';')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty() && !s.starts_with("--"))
        .collect();

    output.info(&format!("Found {} statement(s)", queries.len()));

    for (i, query) in queries.iter().enumerate() {
        output.verbose(&format!(
            "Executing statement {}/{}...",
            i + 1,
            queries.len()
        ));
        execute_query(query, database, false, config, output).await?;
    }

    output.success("File executed successfully");
    Ok(())
}

async fn explain_query(query: &str, analyze: bool, config: &Config, output: &Output) -> Result<()> {
    let explain_query = if analyze {
        format!("EXPLAIN ANALYZE {}", query)
    } else {
        format!("EXPLAIN {}", query)
    };

    output.section("Query Execution Plan");
    execute_query(&explain_query, None, false, config, output).await
}

/// Execute interactive SQL shell
pub async fn execute_shell(config: &Config, output: &Output, database: Option<&str>) -> Result<()> {
    use rustyline::DefaultEditor;
    use rustyline::error::ReadlineError;

    let db = database
        .or(config.connection.database.as_deref())
        .unwrap_or("default");

    output.info(&format!(
        "JouleDB Shell - Connected to {}:{}/{}",
        config.connection.host, config.connection.port, db
    ));
    output.info("Type 'help' for commands, 'exit' to quit.");
    output.raw("");

    let mut rl = DefaultEditor::new().map_err(|e| CliError::Other(e.to_string()))?;

    // Load history
    let history_path = dirs::data_dir().map(|p| p.join("jouledb").join("history.txt"));

    if let Some(ref path) = history_path {
        let _ = rl.load_history(path);
    }

    let prompt = format!("{}> ", db);

    loop {
        let readline = rl.readline(&prompt);

        match readline {
            Ok(line) => {
                let line = line.trim();

                if line.is_empty() {
                    continue;
                }

                let _ = rl.add_history_entry(line);

                // Handle special commands
                match line.to_lowercase().as_str() {
                    "exit" | "quit" | "\\q" => {
                        output.info("Goodbye!");
                        break;
                    }
                    "help" | "\\h" | "\\?" => {
                        print_shell_help();
                        continue;
                    }
                    "\\l" => {
                        // List databases
                        if let Err(e) = crate::commands::db::execute(
                            crate::commands::db::DbCommands::List,
                            config,
                            output,
                        )
                        .await
                        {
                            output.error(&e.to_string());
                        }
                        continue;
                    }
                    "\\dt" => {
                        // List tables
                        let query = "SELECT table_name, table_type FROM information_schema.tables WHERE table_schema = 'public'";
                        if let Err(e) = execute_query(query, Some(db), false, config, output).await
                        {
                            output.error(&e.to_string());
                        }
                        continue;
                    }
                    "\\d" => {
                        output.info("Usage: \\d <table_name> - describe table");
                        continue;
                    }
                    _ if line.starts_with("\\d ") => {
                        let table = line.strip_prefix("\\d ").unwrap().trim();
                        let query = format!(
                            "SELECT column_name, data_type, is_nullable FROM information_schema.columns WHERE table_name = '{}'",
                            table
                        );
                        if let Err(e) = execute_query(&query, Some(db), false, config, output).await
                        {
                            output.error(&e.to_string());
                        }
                        continue;
                    }
                    _ if line.starts_with("\\c ") => {
                        let new_db = line.strip_prefix("\\c ").unwrap().trim();
                        output.success(&format!("Connected to database '{}'", new_db));
                        // In a real implementation, this would change the connection
                        continue;
                    }
                    _ => {}
                }

                // Execute SQL
                if let Err(e) = execute_query(line, Some(db), false, config, output).await {
                    output.error(&e.to_string());
                }
            }
            Err(ReadlineError::Interrupted) => {
                output.info("Use 'exit' or Ctrl+D to quit");
            }
            Err(ReadlineError::Eof) => {
                output.info("Goodbye!");
                break;
            }
            Err(err) => {
                output.error(&format!("Error: {:?}", err));
                break;
            }
        }
    }

    // Save history
    if let Some(ref path) = history_path {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = rl.save_history(path);
    }

    Ok(())
}

fn print_shell_help() {
    println!(
        "
JouleDB Shell Commands:

  SQL Commands:
    SELECT ...        Execute a SELECT query
    INSERT ...        Insert data
    UPDATE ...        Update data
    DELETE ...        Delete data

  Meta Commands:
    \\l               List databases
    \\dt              List tables in current database
    \\d <table>       Describe table structure
    \\c <database>    Connect to different database

  General:
    help, \\h, \\?    Show this help
    exit, quit, \\q   Exit the shell

  Tips:
    - Queries don't need semicolons (but they're allowed)
    - Use UP/DOWN arrows for history
    - Tab for auto-completion (coming soon)
"
    );
}
