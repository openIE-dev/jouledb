//! jouledb-cli — the public JouleDB command-line interface.
//!
//! Derived from `crates/bin/joule-cli`. Retains only the database +
//! universal-energy-layer surface: `run`, `db {server, database, query,
//! admin, data, cloud, config, ledger, daemon, shell}`, and `status`.
//! The dev-platform Lux subcommands (init / build / dev / search / repl
//! / deploy / export / ide / energy-on-Lux-source) are dropped because
//! they drag the verity / flowG / Lux crates out of the public closure.

use clap::{Parser, Subcommand};
use colored::Colorize;

mod cloud;
mod commands;
mod config;
mod error;
mod output;

pub use config::Config;
pub use error::{CliError, Result};

/// jouledb — energy-aware database CLI.
#[derive(Parser)]
#[command(name = "jouledb-cli")]
#[command(author = "OpenIE")]
#[command(version)]
#[command(about = "The world's first energy-aware database — joules per query, native engine + universal energy layer", long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    /// Configuration file path
    #[arg(short, long, global = true, env = "JOULEDB_CONFIG")]
    config: Option<String>,

    /// Output format (text, json, table)
    #[arg(short, long, global = true, default_value = "text")]
    format: output::OutputFormat,

    /// Enable verbose output
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Suppress non-essential output
    #[arg(short, long, global = true)]
    quiet: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run any database / container / process with JouleDB energy
    /// telemetry. `jouledb-cli run postgres`, `… redis --port 6380`,
    /// `… ./my-binary`, etc.
    Run(commands::run::RunCommand),

    /// JouleDB operations (server, query, admin, data, cloud, ledger,
    /// daemon, shell).
    #[command(subcommand)]
    Db(DbCommands),

    /// Show connection / daemon status.
    Status,
}

/// JouleDB subcommands.
#[derive(Subcommand)]
enum DbCommands {
    /// Server management
    #[command(subcommand)]
    Server(commands::server::ServerCommands),
    /// Database operations
    #[command(subcommand)]
    Database(commands::db::DbCommands),
    /// Query operations
    #[command(subcommand)]
    Query(commands::query::QueryCommands),
    /// Admin operations (backup, restore, migrations)
    #[command(subcommand)]
    Admin(commands::admin::AdminCommands),
    /// Data import/export
    #[command(subcommand)]
    Data(commands::data::DataCommands),
    /// Cloud management
    #[command(subcommand)]
    Cloud(commands::cloud::CloudCommands),
    /// Configuration
    #[command(subcommand)]
    Config(commands::config::ConfigCommands),
    /// Energy ledger: verify receipts, view history, export
    #[command(subcommand)]
    Ledger(commands::ledger::LedgerCommands),
    /// Manage the persistent JouleDB daemon
    #[command(subcommand)]
    Daemon(commands::daemon::DaemonCommands),
    /// Open interactive SQL shell
    Shell {
        /// Database to connect to
        #[arg(short, long)]
        database: Option<String>,
    },
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    let config = match Config::load(cli.config.as_deref()) {
        Ok(c) => c,
        Err(e) => {
            if !cli.quiet {
                eprintln!("{}: {}", "Error loading config".red(), e);
            }
            Config::default()
        }
    };

    let output = output::Output::new(cli.format, cli.verbose, cli.quiet);

    let result = match cli.command {
        Commands::Run(cmd) => commands::run::execute(cmd, &config, &output).await,

        Commands::Db(cmd) => match cmd {
            DbCommands::Server(c) => commands::server::execute(c, &config, &output).await,
            DbCommands::Database(c) => commands::db::execute(c, &config, &output).await,
            DbCommands::Query(c) => commands::query::execute(c, &config, &output).await,
            DbCommands::Admin(c) => commands::admin::execute(c, &config, &output).await,
            DbCommands::Data(c) => commands::data::execute(c, &config, &output).await,
            DbCommands::Cloud(c) => commands::cloud::execute(c, &config, &output).await,
            DbCommands::Config(c) => commands::config::execute(c, &config, &output).await,
            DbCommands::Ledger(c) => commands::ledger::execute(c, &config, &output).await,
            DbCommands::Daemon(c) => commands::daemon::execute(c, &config, &output).await,
            DbCommands::Shell { database } => {
                commands::query::execute_shell(&config, &output, database.as_deref()).await
            }
        },

        Commands::Status => commands::status::execute(&config, &output).await,
    };

    if let Err(e) = result {
        output.error(&e.to_string());
        std::process::exit(1);
    }
}
