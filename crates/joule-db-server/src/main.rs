//! JouleDB — The Energy-Aware Database
//!
//! Single binary: `jouledb`, `jouledb query "SELECT ..."`, `jouledb shell`

use std::time::Duration;

use clap::{Parser, Subcommand};
use joule_db_server::{ProductionServer, Server, ServerConfig};

/// JouleDB — The Energy-Aware Database
#[derive(Parser)]
#[command(
    name = "jouledb",
    version,
    about = "JouleDB — The Energy-Aware Database"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Listen port (binds HTTP, TCP+1, PgWire+2)
    #[arg(short, long, default_value = "8080")]
    port: u16,

    /// Data directory
    #[arg(short, long, default_value = "./joule-db-data")]
    data: String,

    /// Query timeout in milliseconds (0 = no timeout)
    #[arg(long, default_value = "30000")]
    query_timeout: u64,

    /// Slow query threshold in milliseconds (queries slower than this are logged)
    #[arg(long, default_value = "1000")]
    slow_query_threshold: u64,

    /// Runtime isolation mode: native (bare metal), vm (hardware isolation), wasm (sandboxed)
    #[arg(long, default_value = "native", env = "JOULEDB_RUNTIME_MODE")]
    mode: String,

    /// JWP (Joule Wire Protocol) bind address. Enables JWP transport.
    #[arg(long, env = "JOULEDB_JWP_ADDR")]
    jwp_addr: Option<String>,

    /// Disable JWP transport (enabled by default on port+3)
    #[arg(long)]
    no_jwp: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Execute a SQL query against a running server
    Query {
        /// SQL to execute
        sql: String,

        /// Server URL
        #[arg(short, long, default_value = "http://127.0.0.1:8080")]
        url: String,
    },
    /// Open an interactive SQL shell
    Shell {
        /// Server URL
        #[arg(short, long, default_value = "http://127.0.0.1:8080")]
        url: String,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Query { sql, url }) => run_query(&url, &sql).await,
        Some(Commands::Shell { url }) => run_shell(&url).await,
        None => {
            run_server(
                cli.port,
                cli.data,
                cli.query_timeout,
                cli.slow_query_threshold,
                cli.mode,
                cli.jwp_addr,
                cli.no_jwp,
            )
            .await
        }
    }
}

async fn run_server(
    port: u16,
    data: String,
    query_timeout: u64,
    slow_query_threshold: u64,
    mode: String,
    jwp_addr: Option<String>,
    no_jwp: bool,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing_subscriber::fmt::init();

    let mut config = ServerConfig::default();
    config.runtime_mode = mode;
    let bind_host = std::env::var("JOULEDB_BIND_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    config.http_addr = format!("{}:{}", bind_host, port);
    config.tcp_addr = format!("{}:{}", bind_host, port + 1);
    config.pgwire_addr = format!("{}:{}", bind_host, port + 2);
    config.db_path = data;

    // Query timeout and slow query logging (env vars override CLI flags)
    config.query_timeout_ms = std::env::var("JOULEDB_QUERY_TIMEOUT_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(query_timeout);
    config.slow_query_threshold_ms = std::env::var("JOULEDB_SLOW_QUERY_THRESHOLD_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(slow_query_threshold);

    // Everything else is on by default — TCP, PgWire, energy monitoring.
    // Auth, replication, TLS are env-var only for operators who need them.
    if let Ok(secret) = std::env::var("JOULEDB_JWT_SECRET") {
        config.auth_enabled = true;
        config.auth_jwt_secret = Some(secret);
    }
    if let Ok(role) = std::env::var("JOULEDB_REPLICATION_ROLE") {
        config.enable_replication = true;
        config.replication_role = Some(role);
    }
    if let Ok(addr) = std::env::var("JOULEDB_REPLICATION_LEADER_ADDR") {
        config.enable_replication = true;
        config.replication_leader_addr = Some(addr);
    }
    #[cfg(feature = "tls")]
    {
        if let Ok(cert) = std::env::var("JOULEDB_TLS_CERT") {
            config.tls_cert_path = Some(cert);
        }
        if let Ok(key) = std::env::var("JOULEDB_TLS_KEY") {
            config.tls_key_path = Some(key);
        }
    }

    // Rate limiting
    if let Ok(val) = std::env::var("JOULEDB_RATE_LIMIT_RPM") {
        if let Ok(rpm) = val.parse::<u64>() {
            config.rate_limit_requests_per_minute = rpm;
        }
    }
    if std::env::var("JOULEDB_DISABLE_RATE_LIMIT").is_ok() {
        config.rate_limiting_enabled = false;
    }

    // Raft consensus (multi-node clustering)
    if let Ok(node_id) = std::env::var("JOULEDB_RAFT_NODE_ID") {
        config.enable_raft = true;
        config.raft_node_id = Some(node_id);
    }
    if let Ok(addr) = std::env::var("JOULEDB_RAFT_ADDR") {
        config.raft_addr = addr;
    }
    if let Ok(peers) = std::env::var("JOULEDB_RAFT_PEERS") {
        config.raft_peers = peers.split(',').map(|s| s.trim().to_string()).collect();
    }
    if let Ok(secret) = std::env::var("JOULEDB_RAFT_MASTER_SECRET") {
        config.raft_master_secret = Some(secret);
    }

    // JWP (Joule Wire Protocol) transport — enabled by default
    if !no_jwp {
        config.enable_jwp = true;
        config.jwp_addr = jwp_addr.unwrap_or_else(|| format!("{}:{}", bind_host, port + 3));
    }

    // Energy receipt ledger (blockchain-anchored attestation)
    if std::env::var("JOULEDB_LEDGER_ENABLED").is_ok() {
        config.enable_ledger = true;
    }
    if let Ok(dir) = std::env::var("JOULEDB_LEDGER_DIR") {
        config.ledger_dir = Some(dir);
    }
    if let Ok(region) = std::env::var("JOULEDB_GRID_REGION") {
        config.ledger_grid_region = Some(region);
    }
    if let Ok(factor) = std::env::var("JOULEDB_GRID_FACTOR") {
        if let Ok(f) = factor.parse::<f64>() {
            config.ledger_grid_factor = Some(f);
        }
    }

    let server =
        Server::new(config.clone()).map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
            Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("{}", e),
            ))
        })?;

    // Startup banner
    let pi = server.platform_info();
    let mut devices = Vec::new();
    if pi.gpu_available {
        devices.push("GPU");
    }
    if pi.npu_available {
        devices.push("Neural Engine");
    }
    if pi.tpu_available {
        devices.push("TPU");
    }
    let devices_str = if devices.is_empty() {
        "CPU only".to_string()
    } else {
        devices.join(" + ")
    };

    println!();
    println!("  JouleDB v{}", env!("CARGO_PKG_VERSION"));
    println!("  Mode     {}", config.runtime_mode);
    println!();
    println!("  HTTP     http://{}", config.http_addr);
    println!("  TCP      {}", config.tcp_addr);
    println!(
        "  PgWire   {}  (psql -h 127.0.0.1 -p {})",
        config.pgwire_addr,
        port + 2
    );
    println!("  Data     {}", config.db_path);
    if config.enable_jwp {
        println!("  JWP      {}", config.jwp_addr);
    }
    if config.enable_raft {
        println!(
            "  Raft     {}  (node: {})",
            config.raft_addr,
            config.raft_node_id.as_deref().unwrap_or("auto")
        );
    }
    if config.enable_ledger {
        println!(
            "  Ledger   {}",
            config.ledger_dir.as_deref().unwrap_or("(in-memory)")
        );
    }
    println!();
    println!(
        "  {}  {}W  {}",
        truncate(&pi.cpu_brand, 30),
        pi.tdp_watts,
        devices_str
    );
    let timeout_str = if config.query_timeout_ms > 0 {
        format!("{}ms", config.query_timeout_ms)
    } else {
        "disabled".to_string()
    };
    println!(
        "  Timeout    {}  Slow query  ≥{}ms",
        timeout_str, config.slow_query_threshold_ms
    );
    println!();

    let prod_server = ProductionServer::new(Duration::from_secs(30));

    prod_server
        .run_with_shutdown(|shutdown| async move {
            let _server_handle = {
                let shutdown_notify = shutdown.shutdown_notify();
                tokio::spawn(async move {
                    shutdown_notify.notified().await;
                    tracing::info!("Shutdown signal received");
                })
            };

            let server_task = tokio::spawn(async move { server.run().await });

            tokio::select! {
                result = server_task => {
                    let inner = result.map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                        Box::new(std::io::Error::new(std::io::ErrorKind::Other, format!("{}", e)))
                    })?;
                    inner.map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                        Box::new(std::io::Error::new(std::io::ErrorKind::Other, format!("{}", e)))
                    })?;
                }
                _ = shutdown.wait_for_shutdown() => {
                    tracing::info!("Graceful shutdown initiated");
                }
            }

            Ok(())
        })
        .await?;

    Ok(())
}

async fn run_query(url: &str, sql: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/api/v1/query", url))
        .json(&serde_json::json!({ "sql": sql }))
        .send()
        .await?;

    let status = resp.status();
    let body: serde_json::Value = resp.json().await?;

    if !status.is_success() {
        eprintln!(
            "Error ({}): {}",
            status,
            serde_json::to_string_pretty(&body)?
        );
        std::process::exit(1);
    }

    println!("{}", serde_json::to_string_pretty(&body)?);
    Ok(())
}

async fn run_shell(url: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("JouleDB Shell — {}", url);
    println!("Type SQL, or \\q to exit.\n");

    let client = reqwest::Client::new();
    let stdin = std::io::stdin();

    loop {
        eprint!("jouledb> ");
        let mut line = String::new();
        if stdin.read_line(&mut line)? == 0 {
            break;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.eq_ignore_ascii_case("quit")
            || trimmed.eq_ignore_ascii_case("exit")
            || trimmed == "\\q"
        {
            break;
        }

        match client
            .post(format!("{}/api/v1/query", url))
            .json(&serde_json::json!({ "sql": trimmed }))
            .send()
            .await
        {
            Ok(resp) => {
                let body: serde_json::Value = resp.json().await.unwrap_or_default();
                println!(
                    "{}",
                    serde_json::to_string_pretty(&body).unwrap_or_default()
                );
            }
            Err(e) => eprintln!("Connection error: {}", e),
        }
    }

    Ok(())
}

fn truncate(s: &str, max: usize) -> &str {
    if s.len() <= max { s } else { &s[..max] }
}
