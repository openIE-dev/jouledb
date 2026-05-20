//! Web module commands: list, run, info, domains

use crate::{Config, Result, error::CliError, output::Output};
use clap::Subcommand;

#[derive(Subcommand)]
pub enum WebCommands {
    /// List all registered modules (optionally filter by domain)
    List {
        /// Filter by domain (e.g. "financial", "bioinformatics")
        #[arg(short, long)]
        domain: Option<String>,
    },

    /// Invoke a module with JSON arguments
    Run {
        /// Module name (e.g. "black_scholes", "stats_engine")
        module: String,

        /// JSON arguments string (e.g. '{"data": [1,2,3]}')
        #[arg(default_value = "{}")]
        args: String,
    },

    /// Show details for a specific module
    Info {
        /// Module name
        module: String,
    },

    /// List available domain categories
    Domains,
}

pub async fn execute(cmd: WebCommands, _config: &Config, output: &Output) -> Result<()> {
    match cmd {
        WebCommands::List { domain } => list(output, domain.as_deref()),
        WebCommands::Run { module, args } => run(output, &module, &args),
        WebCommands::Info { module } => info(output, &module),
        WebCommands::Domains => domains(output),
    }
}

fn list(output: &Output, domain: Option<&str>) -> Result<()> {
    let registry = joule_web::dispatch::ModuleRegistry::new();

    let entries = match domain {
        Some(d) => registry.list_domain(d),
        None => registry.list(),
    };

    if entries.is_empty() {
        match domain {
            Some(d) => output.info(&format!("No modules found in domain '{}'.", d)),
            None => output.info("No modules registered."),
        }
        return Ok(());
    }

    let rows: Vec<Vec<String>> = entries
        .iter()
        .map(|e| {
            vec![
                e.name.to_string(),
                e.domain.to_string(),
                e.doc.to_string(),
            ]
        })
        .collect();

    output.table(vec!["Module", "Domain", "Description"], rows);

    output.info(&format!("{} module(s) total", entries.len()));

    Ok(())
}

fn run(output: &Output, module: &str, args_str: &str) -> Result<()> {
    let registry = joule_web::dispatch::ModuleRegistry::new();

    let args: serde_json::Value = serde_json::from_str(args_str)
        .map_err(|e| CliError::InvalidInput(format!("invalid JSON args: {}", e)))?;

    let result = registry
        .invoke(module, &args)
        .map_err(|e| CliError::Other(e))?;

    // Print the result JSON
    let pretty = serde_json::to_string_pretty(&result.result)
        .map_err(|e| CliError::Other(format!("JSON serialization error: {}", e)))?;
    output.raw(&pretty);

    // Print energy receipt
    output.raw("");
    output.section("Energy Receipt");
    output.key_value(vec![
        ("Module", result.module),
        ("Duration", format!("{:.6} s", result.duration_secs)),
        ("Energy", format!("{:.6} J", result.energy_joules)),
    ]);

    Ok(())
}

fn info(output: &Output, module: &str) -> Result<()> {
    let registry = joule_web::dispatch::ModuleRegistry::new();

    let entry = registry
        .get(module)
        .ok_or_else(|| CliError::NotFound(format!("module '{}' not found", module)))?;

    output.key_value(vec![
        ("Name", entry.name.to_string()),
        ("Domain", entry.domain.to_string()),
        ("Description", entry.doc.to_string()),
    ]);

    Ok(())
}

fn domains(output: &Output) -> Result<()> {
    let registry = joule_web::dispatch::ModuleRegistry::new();
    let domains = registry.domains();

    if domains.is_empty() {
        output.info("No domains registered.");
        return Ok(());
    }

    let rows: Vec<Vec<String>> = domains
        .iter()
        .map(|d| {
            let count = registry.list_domain(d).len();
            vec![d.to_string(), count.to_string()]
        })
        .collect();

    output.table(vec!["Domain", "Modules"], rows);

    Ok(())
}
