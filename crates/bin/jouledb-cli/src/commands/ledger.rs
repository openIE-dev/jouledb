//! Ledger commands: verify, history, export, batches, stats

use crate::{Config, Result, error::CliError, output::Output};
use clap::Subcommand;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Subcommand)]
pub enum LedgerCommands {
    /// Verify a receipt's Merkle inclusion proof
    Verify {
        /// Receipt ID to verify
        receipt_id: String,
    },

    /// List recent energy receipts
    History {
        /// Maximum number of receipts to show
        #[arg(short, long, default_value = "20")]
        limit: usize,
    },

    /// Export receipts as JSON or CSV
    Export {
        /// Output format (json, csv)
        #[arg(short, long, default_value = "json")]
        format: String,

        /// Output file (stdout if omitted)
        #[arg(short, long)]
        output: Option<String>,

        /// Maximum number of receipts to export
        #[arg(short, long, default_value = "1000")]
        limit: usize,
    },

    /// List committed batches
    Batches {
        /// Maximum number of batches to show
        #[arg(short, long, default_value = "20")]
        limit: usize,
    },

    /// Show aggregate ledger statistics
    Stats,
}

pub async fn execute(cmd: LedgerCommands, config: &Config, output: &Output) -> Result<()> {
    match cmd {
        LedgerCommands::Verify { receipt_id } => verify(config, output, &receipt_id).await,
        LedgerCommands::History { limit } => history(config, output, limit).await,
        LedgerCommands::Export {
            format,
            output: out_path,
            limit,
        } => export(config, output, &format, out_path.as_deref(), limit).await,
        LedgerCommands::Batches { limit } => batches(config, output, limit).await,
        LedgerCommands::Stats => stats(config, output).await,
    }
}

fn server_url(config: &Config) -> String {
    format!(
        "http://{}:{}",
        config.connection.host, config.connection.port
    )
}

fn http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .expect("failed to build HTTP client")
}

// ---- Verify ----

#[derive(Debug, Deserialize)]
struct VerifyResponse {
    verified: bool,
    receipt: ReceiptData,
    batch: BatchSummary,
}

#[derive(Debug, Deserialize)]
struct ReceiptData {
    receipt_id: String,
    qid: String,
    tenant_id: String,
    energy_joules_total: f64,
    kwh: f64,
    kg_co2e: f64,
    device_target: String,
    algorithm_type: String,
}

async fn verify(config: &Config, output: &Output, receipt_id: &str) -> Result<()> {
    let url = format!(
        "{}/api/v1/ledger/receipts/{}/verify",
        server_url(config),
        receipt_id
    );
    let resp = http_client()
        .get(&url)
        .send()
        .await
        .map_err(|e| CliError::Other(format!("Failed to connect: {}", e)))?;

    if resp.status() == reqwest::StatusCode::NOT_FOUND {
        output.error(&format!("Receipt '{}' not found", receipt_id));
        return Ok(());
    }

    if !resp.status().is_success() {
        output.error(&format!("Server error: {}", resp.status()));
        return Ok(());
    }

    let data: VerifyResponse = resp
        .json()
        .await
        .map_err(|e| CliError::Other(format!("Invalid response: {}", e)))?;

    if data.verified {
        output.success(&format!(
            "Receipt '{}' is VERIFIED in batch '{}'",
            receipt_id, data.batch.batch_id
        ));
    } else {
        output.error(&format!("Receipt '{}' FAILED verification", receipt_id));
    }

    output.key_value(vec![
        ("Query ID", data.receipt.qid),
        ("Tenant", data.receipt.tenant_id),
        (
            "Energy",
            format!("{:.6} J", data.receipt.energy_joules_total),
        ),
        ("kWh", format!("{:.9}", data.receipt.kwh)),
        ("kgCO2e", format!("{:.9}", data.receipt.kg_co2e)),
        ("Device", data.receipt.device_target),
        ("Algorithm", data.receipt.algorithm_type),
        ("Batch", data.batch.batch_id),
        ("Merkle Root", data.batch.merkle_root),
    ]);

    Ok(())
}

// ---- History ----

#[derive(Debug, Serialize, Deserialize)]
struct ReceiptSummary {
    receipt_id: String,
    #[allow(dead_code)]
    qid: String,
    tenant_id: String,
    energy_joules_total: f64,
    kwh: f64,
    #[allow(dead_code)]
    kg_co2e: f64,
    device_target: String,
    algorithm_type: String,
    timestamp_start: String,
    #[allow(dead_code)]
    timestamp_end: String,
    batch_id: String,
}

async fn history(config: &Config, output: &Output, limit: usize) -> Result<()> {
    let url = format!(
        "{}/api/v1/ledger/receipts?limit={}",
        server_url(config),
        limit
    );
    let resp = http_client()
        .get(&url)
        .send()
        .await
        .map_err(|e| CliError::Other(format!("Failed to connect: {}", e)))?;

    if !resp.status().is_success() {
        output.error(&format!("Server error: {}", resp.status()));
        return Ok(());
    }

    let receipts: Vec<ReceiptSummary> = resp
        .json()
        .await
        .map_err(|e| CliError::Other(format!("Invalid response: {}", e)))?;

    if receipts.is_empty() {
        output.info("No receipts found.");
        return Ok(());
    }

    let rows: Vec<Vec<String>> = receipts
        .iter()
        .map(|r| {
            vec![
                r.receipt_id.chars().take(12).collect::<String>(),
                r.tenant_id.clone(),
                format!("{:.4} J", r.energy_joules_total),
                format!("{:.6e}", r.kwh),
                r.device_target.clone(),
                r.algorithm_type.clone(),
                r.timestamp_start.clone(),
                r.batch_id.chars().take(12).collect::<String>(),
            ]
        })
        .collect();

    output.table(
        vec![
            "Receipt ID",
            "Tenant",
            "Energy",
            "kWh",
            "Device",
            "Algorithm",
            "Timestamp",
            "Batch",
        ],
        rows,
    );

    Ok(())
}

// ---- Export ----

async fn export(
    config: &Config,
    output: &Output,
    format: &str,
    out_path: Option<&str>,
    limit: usize,
) -> Result<()> {
    let url = format!(
        "{}/api/v1/ledger/receipts?limit={}",
        server_url(config),
        limit
    );
    let resp = http_client()
        .get(&url)
        .send()
        .await
        .map_err(|e| CliError::Other(format!("Failed to connect: {}", e)))?;

    if !resp.status().is_success() {
        output.error(&format!("Server error: {}", resp.status()));
        return Ok(());
    }

    let receipts: Vec<ReceiptSummary> = resp
        .json()
        .await
        .map_err(|e| CliError::Other(format!("Invalid response: {}", e)))?;

    let content = match format {
        "csv" => {
            let mut lines = vec![
                "receipt_id,tenant_id,energy_joules,kwh,kg_co2e,device,algorithm,timestamp_start,batch_id"
                    .to_string(),
            ];
            for r in &receipts {
                lines.push(format!(
                    "{},{},{},{},{},{},{},{},{}",
                    r.receipt_id,
                    r.tenant_id,
                    r.energy_joules_total,
                    r.kwh,
                    r.kg_co2e,
                    r.device_target,
                    r.algorithm_type,
                    r.timestamp_start,
                    r.batch_id,
                ));
            }
            lines.join("\n")
        }
        _ => serde_json::to_string_pretty(&receipts)
            .map_err(|e| CliError::Other(format!("JSON error: {}", e)))?,
    };

    match out_path {
        Some(path) => {
            std::fs::write(path, &content)
                .map_err(|e| CliError::Other(format!("Write error: {}", e)))?;
            output.success(&format!("Exported {} receipts to {}", receipts.len(), path));
        }
        None => {
            println!("{}", content);
        }
    }

    Ok(())
}

// ---- Batches ----

#[derive(Debug, Deserialize)]
struct BatchSummary {
    batch_id: String,
    merkle_root: String,
    receipt_count: usize,
    #[allow(dead_code)]
    time_start: String,
    time_end: String,
    aggregate_kwh: f64,
    aggregate_kg_co2e: f64,
    issuer: String,
}

async fn batches(config: &Config, output: &Output, limit: usize) -> Result<()> {
    let url = format!("{}/api/v1/ledger/batches", server_url(config));
    let resp = http_client()
        .get(&url)
        .send()
        .await
        .map_err(|e| CliError::Other(format!("Failed to connect: {}", e)))?;

    if !resp.status().is_success() {
        output.error(&format!("Server error: {}", resp.status()));
        return Ok(());
    }

    let mut batch_list: Vec<BatchSummary> = resp
        .json()
        .await
        .map_err(|e| CliError::Other(format!("Invalid response: {}", e)))?;

    batch_list.truncate(limit);

    if batch_list.is_empty() {
        output.info("No batches found.");
        return Ok(());
    }

    let rows: Vec<Vec<String>> = batch_list
        .iter()
        .map(|b| {
            vec![
                b.batch_id.chars().take(12).collect::<String>(),
                b.receipt_count.to_string(),
                format!("{:.6e}", b.aggregate_kwh),
                format!("{:.6e}", b.aggregate_kg_co2e),
                b.merkle_root.chars().take(16).collect::<String>(),
                b.time_end.clone(),
                b.issuer.clone(),
            ]
        })
        .collect();

    output.table(
        vec![
            "Batch ID",
            "Receipts",
            "kWh",
            "kgCO2e",
            "Merkle Root",
            "End Time",
            "Issuer",
        ],
        rows,
    );

    Ok(())
}

// ---- Stats ----

#[derive(Debug, Deserialize)]
struct LedgerStats {
    total_receipts: usize,
    total_batches: usize,
    total_energy_joules: f64,
    total_kwh: f64,
    total_kg_co2e: f64,
    by_device: HashMap<String, f64>,
    by_algorithm: HashMap<String, f64>,
    oldest_receipt: Option<String>,
    newest_receipt: Option<String>,
}

async fn stats(config: &Config, output: &Output) -> Result<()> {
    let url = format!("{}/api/v1/ledger/stats", server_url(config));
    let resp = http_client()
        .get(&url)
        .send()
        .await
        .map_err(|e| CliError::Other(format!("Failed to connect: {}", e)))?;

    if !resp.status().is_success() {
        output.error(&format!("Server error: {}", resp.status()));
        return Ok(());
    }

    let s: LedgerStats = resp
        .json()
        .await
        .map_err(|e| CliError::Other(format!("Invalid response: {}", e)))?;

    output.section("Ledger Statistics");

    output.key_value(vec![
        ("Total Receipts", s.total_receipts.to_string()),
        ("Total Batches", s.total_batches.to_string()),
        ("Total Energy", format!("{:.6} J", s.total_energy_joules)),
        ("Total kWh", format!("{:.9}", s.total_kwh)),
        ("Total kgCO2e", format!("{:.9}", s.total_kg_co2e)),
        (
            "Oldest Receipt",
            s.oldest_receipt.unwrap_or_else(|| "-".to_string()),
        ),
        (
            "Newest Receipt",
            s.newest_receipt.unwrap_or_else(|| "-".to_string()),
        ),
    ]);

    if !s.by_device.is_empty() {
        output.raw("");
        output.info("Energy by Device:");
        let mut device_rows: Vec<Vec<String>> = s
            .by_device
            .iter()
            .map(|(dev, j)| vec![dev.clone(), format!("{:.6} J", j)])
            .collect();
        device_rows.sort_by(|a, b| a[0].cmp(&b[0]));
        output.table(vec!["Device", "Energy (J)"], device_rows);
    }

    if !s.by_algorithm.is_empty() {
        output.raw("");
        output.info("Energy by Algorithm:");
        let mut algo_rows: Vec<Vec<String>> = s
            .by_algorithm
            .iter()
            .map(|(algo, j)| vec![algo.clone(), format!("{:.6} J", j)])
            .collect();
        algo_rows.sort_by(|a, b| a[0].cmp(&b[0]));
        output.table(vec!["Algorithm", "Energy (J)"], algo_rows);
    }

    Ok(())
}
