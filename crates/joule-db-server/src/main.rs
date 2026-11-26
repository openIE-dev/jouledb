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
    /// Inspect live Heterogeneous Energy Fabric topology and probe multi-silicon hardware lanes
    Fabric {
        /// Server URL
        #[arg(short, long, default_value = "http://127.0.0.1:8080")]
        url: String,
    },
    /// Zero-bloat Post-Docker `JouleCapsule` OCI & CoW container commands (no `dockerd`)
    Capsule {
        /// Server URL
        #[arg(short, long, default_value = "http://127.0.0.1:8080")]
        url: String,

        #[command(subcommand)]
        action: CapsuleAction,
    },
    /// Post-Frozen-Weights `JouleSynapse` & Database-Native `JouleHarness` commands
    Synapse {
        /// Server URL
        #[arg(short, long, default_value = "http://127.0.0.1:8080")]
        url: String,

        #[command(subcommand)]
        action: SynapseAction,
    },
    /// `JouleWard` zero-ambient-authority capability sandbox, `.jcap` bundle & 6-rail nanojoule ledger
    Ward {
        /// Server URL
        #[arg(short, long, default_value = "http://127.0.0.1:8080")]
        url: String,

        #[command(subcommand)]
        action: WardAction,
    },
    /// Drop-in `ollama run` / OpenAI `/v1/chat/completions` disintermediation with 6-rail nanojoule receipts
    Run {
        /// Living model ID (`jouledb-cascade-v1` or ingested model such as `all-minilm-l6-v2`)
        model: String,
        /// Prompt text
        prompt: String,
        /// Server URL
        #[arg(short, long, default_value = "http://127.0.0.1:8080")]
        url: String,
        /// Route across the Heterogeneous Compute Fabric when optimal
        #[arg(long)]
        fabric: bool,
    },
    /// Zero-DDL Amorphous Semantic Continuum, Programmable Morph Rules & Multi-GB Streaming CLI
    Amorphic {
        #[command(subcommand)]
        action: AmorphicAction,
    },
    /// Inspect HTTP/wire compatibility adapters and run in-process entity-merge smoke check
    Bridge,
    /// Run technical security-control self-tests (unaudited) or emit CycloneDX v1.6 SBOM
    Trust {
        /// Output CycloneDX v1.6 SBOM + model manifest JSON instead of running control self-tests
        #[arg(long)]
        sbom: bool,
    },
    /// Multi-format import & multiset BLAKE3 reconciliation tool (SQLite, SQL dump, JSONL, _bulk, CSV)
    Migrate {
        /// Source ecosystem (`postgres`, `sqlite`, `mysql`, `mongo`, `elastic`, `csv`, `jsonl`)
        #[arg(long, default_value = "postgres")]
        from: String,
        /// Optional path to a `.db`/`.sqlite`, `.sql`, `.jsonl`, `_bulk`, or `.csv` file (omit to run in-process parser smoke check)
        #[arg(short, long)]
        input: Option<String>,
    },
    /// Execute the 6 fast in-process subsystem smoke checks (not an external benchmark suite)
    #[command(alias = "powerhouse")]
    SelfCheck,
    /// Run the Model Context Protocol (MCP + SEP-1865 MCP Apps) server over stdio for Claude Desktop, Claude Code & Antigravity
    #[command(alias = "mcp-stdio")]
    Mcp {
        /// Persistent database directory for MCP tools & JouleDB Studio UI
        #[arg(short, long, default_value = "./joule-db-mcp")]
        data: String,
    },
}

#[derive(Subcommand)]
enum AmorphicAction {
    /// Stream an existing heterogeneous directory (or synthesize + stream `--generate-bytes`) in O(E) bounded RAM
    StreamCorpus {
        /// Directory path containing `.jsonl`, `.csv`, `.xml`, `.kv`, `.log`, `.nt` files
        #[arg(short, long)]
        dir: String,
        /// Optional target raw byte count to synthesize across 6 silos before streaming
        #[arg(long, default_value_t = 0)]
        generate_bytes: u64,
        /// Number of distinct entities when synthesizing (`--generate-bytes > 0`)
        #[arg(long, default_value_t = 1024)]
        entities: usize,
        /// Remove generated files after streaming receipt is computed
        #[arg(long)]
        cleanup: bool,
    },
    /// Run the programmable MorphRule DSL + VSA algebraic unbinding + Custom Lens certification suite
    MorphSuite,
}

#[derive(Subcommand)]
enum WardAction {
    /// Inspect high-resolution (`nJ` / `ns`) 6-rail energy accounting across the compute fabric
    Ledger,
    /// Mint a signed `ThermoCap` capability token with optional nanojoule & read-only caveats
    Mint {
        /// Root token lineage ID
        #[arg(long, default_value = "cli-cap-01")]
        id: String,
        /// Subject / agent name
        #[arg(long, default_value = "cli-operator")]
        subject: String,
        /// Optional total nanojoule ceiling
        #[arg(long)]
        max_nj: Option<u64>,
        /// Enforce read-only filesystem
        #[arg(long)]
        read_only: bool,
        /// Airgap network (`AllowEgressHosts([])`)
        #[arg(long)]
        airgap: bool,
    },
    /// Execute a command under `JouleWard` capability confinement + Merkle CAS memoization
    Exec {
        /// Route across the Heterogeneous Compute Fabric
        #[arg(long)]
        fabric: bool,
        /// Command and arguments
        #[arg(trailing_var_arg = true, required = true)]
        command: Vec<String>,
    },
    /// Export a single-file SHA-256 verified `.jcap` bundle (`Apptainer` `.sif` disintermediation)
    JcapExport {
        /// Bundle ID
        #[arg(long)]
        id: String,
        /// Output `.jcap` file path
        #[arg(short, long)]
        out: String,
        /// Optional living model ID to embed metadata for
        #[arg(long)]
        model: Option<String>,
    },
}

#[derive(Subcommand)]
enum SynapseAction {
    /// Show live `LittleBit-2`, `Titans`, `Engram`, `Mooncake` KV, and `wgpu` kernel status
    Status,
    /// Ingest a HuggingFace `.safetensors` checkpoint into sub-1-bit `LittleBit-2` + SQL `synapse_tensors`
    Ingest {
        /// Model identifier (e.g. `minilm-l6-v2`)
        #[arg(short, long)]
        model: String,
        /// Path to `.safetensors` file
        #[arg(short, long)]
        file: String,
        /// Latent binary rank (default: 8)
        #[arg(long, default_value_t = 8)]
        rank: usize,
        /// Sparse outlier cap per tensor (default: 8)
        #[arg(long, default_value_t = 8)]
        outliers: usize,
    },
    /// Query SQL `synapse_tensors` rows for an ingested model
    Layers {
        /// Model identifier
        model: String,
    },
    /// Compile a `MIT SEAL` runtime self-edit into `Engram` + live `Titans` fast weights
    SelfEdit {
        /// Trigger phrase
        #[arg(long)]
        phrase: String,
        /// Fact / correction payload
        #[arg(long)]
        fact: String,
    },
}

#[derive(Subcommand)]
enum CapsuleAction {
    /// List content-addressed OCI images and CAS deduplication stats
    Images,
    /// List live CoW `JouleCapsule` instances
    Ps,
    /// Pull an OCI Distribution v2 image natively over HTTPS without `dockerd`
    Pull {
        /// Image reference (`alpine:3.20`, `ghcr.io/org/app:v1`, etc.)
        image_ref: String,
    },
    /// Build an OCI image from a local `Joulefile` or `Dockerfile`
    Build {
        /// Image tag (`my-service:v1`)
        #[arg(short, long)]
        tag: String,
        /// Path to `Joulefile` or `Dockerfile`
        #[arg(short, long, default_value = "Dockerfile")]
        file: String,
    },
    /// Spawn a running `JouleCapsule` backed by an O(1) CoW `JouleAgentFs` branch
    Run {
        /// Capsule ID / name
        #[arg(short, long)]
        name: String,
        /// Image reference
        image_ref: String,
        /// Optional microjoule ceiling
        #[arg(long)]
        budget_uj: Option<u64>,
    },
    /// Fork a running capsule's rootfs + database branch in O(1) CoW
    Fork {
        /// Source capsule ID
        source_id: String,
        /// New forked capsule ID
        new_id: String,
        /// Optional microjoule ceiling
        #[arg(long)]
        budget_uj: Option<u64>,
    },
    /// Execute a command inside a capsule's isolated CoW rootfs
    Exec {
        /// Target capsule ID
        capsule_id: String,
        /// Command and arguments
        #[arg(trailing_var_arg = true, required = true)]
        command: Vec<String>,
    },
    /// Diff rootfs paths between two CoW capsules (`docker diff` superset)
    Diff {
        /// Base capsule ID
        base_id: String,
        /// Target capsule ID
        target_id: String,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Query { sql, url }) => run_query(&url, &sql).await,
        Some(Commands::Shell { url }) => run_shell(&url).await,
        Some(Commands::Fabric { url }) => run_fabric_probe(&url).await,
        Some(Commands::Capsule { url, action }) => run_capsule_cli(&url, action).await,
        Some(Commands::Synapse { url, action }) => run_synapse_cli(&url, action).await,
        Some(Commands::Ward { url, action }) => run_ward_cli(&url, action).await,
        Some(Commands::Run {
            model,
            prompt,
            url,
            fabric,
        }) => run_cascade_cli(&url, &model, &prompt, fabric).await,
        Some(Commands::Amorphic { action }) => run_amorphic_cli(action),
        Some(Commands::Bridge) => {
            println!(
                "{}",
                serde_json::to_string_pretty(
                    &joule_db_server::universal_bridge::run_universal_bridge_certification()
                )?
            );
            Ok(())
        }
        Some(Commands::Trust { sbom }) => {
            let val = if sbom {
                joule_db_server::trust_compliance::generate_cyclonedx_sbom_and_aibom()
            } else {
                joule_db_server::trust_compliance::execute_full_trust_and_compliance_suite()?
            };
            println!("{}", serde_json::to_string_pretty(&val)?);
            Ok(())
        }
        Some(Commands::Migrate { from, input }) => {
            if let Some(path_str) = input {
                let fmt = joule_db_server::live_migration::MigrationSourceFormat::parse_str(&from);
                let records = joule_db_server::live_migration::load_migration_source_from_path(
                    fmt,
                    std::path::Path::new(&path_str),
                )
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                    Box::new(std::io::Error::other(e))
                })?;
                let cert = joule_db_server::live_migration::execute_zero_downtime_migration(
                    "cli-live-migration",
                    fmt,
                    records,
                    &[],
                );
                println!("{}", serde_json::to_string_pretty(&cert)?);
            } else {
                println!(
                    "{}",
                    serde_json::to_string_pretty(
                        &joule_db_server::live_migration::run_all_ecosystems_migration_certification()
                    )?
                );
            }
            Ok(())
        }
        Some(Commands::SelfCheck) => {
            println!(
                "{}",
                serde_json::to_string_pretty(
                    &joule_db_server::distributed_jepsen_consensus::run_v1_global_powerhouse_suite()
                )?
            );
            Ok(())
        }
        Some(Commands::Mcp { data }) => {
            std::fs::create_dir_all(&data)?;
            let amorphic_path = format!("{}/amorphic", data);
            let amorphic_store = joule_db_amorphic::DurableAmorphicStore::open(&amorphic_path)
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                    Box::new(std::io::Error::other(format!("Failed to open amorphic store: {e}")))
                })?;
            let amorphic = std::sync::Arc::new(
                joule_db_server::amorphic_adapter::AmorphicTableStorage::new(amorphic_store),
            );
            let executor =
                joule_db_server::query::SimpleQueryExecutor::with_amorphic(amorphic);
            let handler = std::sync::Arc::new(
                joule_db_server::mcp_bridge::DatabaseToolHandler::new(std::sync::Arc::new(
                    executor,
                )),
            );
            joule_db_server::mcp_transport::run_stdio_transport(Some(handler)).await;
            Ok(())
        }
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
    // Loopback binds use trust auth unless JOULEDB_JWT_SECRET is explicitly configured,
    // so psql, psycopg, SQLAlchemy, JDBC, and node-postgres connect out of the box.
    let loopback = matches!(bind_host.as_str(), "127.0.0.1" | "localhost" | "::1");
    config.auth_enabled = !loopback;
    if let Ok(secret) = std::env::var("JOULEDB_JWT_SECRET") {
        config.auth_enabled = true;
        config.auth_jwt_secret = Some(secret);
    }
    if config.auth_enabled && config.auth_jwt_secret.is_none() {
        eprintln!(
            "jouledb: binding {bind_host} without JOULEDB_JWT_SECRET: clients will be refused. \
             Set JOULEDB_JWT_SECRET (also the pgwire password)."
        );
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
    if std::env::var_os("JOULEDB_NO_FLIGHT_SQL").is_none() {
        config.flight_sql_addr = Some(
            std::env::var("JOULEDB_FLIGHT_SQL_ADDR")
                .unwrap_or_else(|_| format!("{}:{}", bind_host, port + 4)),
        );
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

async fn run_fabric_probe(url: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let client = reqwest::Client::new();
    let topo: serde_json::Value = client
        .get(format!(
            "{}/api/v1/fabric/topology",
            url.trim_end_matches('/')
        ))
        .send()
        .await?
        .json()
        .await?;
    let probe: serde_json::Value = client
        .post(format!(
            "{}/api/v1/fabric/execute",
            url.trim_end_matches('/')
        ))
        .json(&serde_json::json!({
            "workload": joule_db_server::fabric_routes::default_gpu_probe_workload(),
            "batch_count": 64,
            "vector_dim": 512
        }))
        .send()
        .await?
        .json()
        .await?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "topology": topo,
            "gpu_or_simd_probe": probe,
        }))?
    );
    Ok(())
}

async fn run_capsule_cli(
    url: &str,
    action: CapsuleAction,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let base = url.trim_end_matches('/');
    let client = reqwest::Client::new();
    let out: serde_json::Value = match action {
        CapsuleAction::Images => {
            client
                .get(format!("{base}/api/v1/capsules/images"))
                .send()
                .await?
                .json()
                .await?
        }
        CapsuleAction::Ps => {
            client
                .get(format!("{base}/api/v1/capsules"))
                .send()
                .await?
                .json()
                .await?
        }
        CapsuleAction::Pull { image_ref } => {
            client
                .post(format!("{base}/api/v1/capsules/images/pull"))
                .json(&serde_json::json!({ "image_ref": image_ref }))
                .send()
                .await?
                .json()
                .await?
        }
        CapsuleAction::Build { tag, file } => {
            let joulefile = std::fs::read_to_string(&file)?;
            let mut context_files = std::collections::BTreeMap::new();
            if let Some(parent) = std::path::Path::new(&file).parent() {
                let dir = if parent.as_os_str().is_empty() {
                    std::path::Path::new(".")
                } else {
                    parent
                };
                if let Ok(entries) = std::fs::read_dir(dir) {
                    for entry in entries.flatten() {
                        if let Ok(ft) = entry.file_type() {
                            if ft.is_file() {
                                if let (Some(name), Ok(content)) = (
                                    entry.file_name().to_str(),
                                    std::fs::read_to_string(entry.path()),
                                ) {
                                    context_files.insert(name.to_string(), content);
                                }
                            }
                        }
                    }
                }
            }
            client
                .post(format!("{base}/api/v1/capsules/images/build"))
                .json(&serde_json::json!({
                    "tag": tag,
                    "joulefile": joulefile,
                    "context_files": context_files,
                    "base64_encoded": false
                }))
                .send()
                .await?
                .json()
                .await?
        }
        CapsuleAction::Run {
            name,
            image_ref,
            budget_uj,
        } => {
            client
                .post(format!("{base}/api/v1/capsules/spawn"))
                .json(&serde_json::json!({
                    "capsule_id": name,
                    "image_ref": image_ref,
                    "energy_budget_uj": budget_uj
                }))
                .send()
                .await?
                .json()
                .await?
        }
        CapsuleAction::Fork {
            source_id,
            new_id,
            budget_uj,
        } => {
            client
                .post(format!("{base}/api/v1/capsules/{source_id}/fork"))
                .json(&serde_json::json!({
                    "new_capsule_id": new_id,
                    "energy_budget_uj": budget_uj
                }))
                .send()
                .await?
                .json()
                .await?
        }
        CapsuleAction::Exec {
            capsule_id,
            command,
        } => {
            client
                .post(format!("{base}/api/v1/capsules/{capsule_id}/exec"))
                .json(&serde_json::json!({ "command": command }))
                .send()
                .await?
                .json()
                .await?
        }
        CapsuleAction::Diff { base_id, target_id } => {
            client
                .get(format!("{base}/api/v1/capsules/{base_id}/diff/{target_id}"))
                .send()
                .await?
                .json()
                .await?
        }
    };
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}

async fn run_synapse_cli(
    url: &str,
    action: SynapseAction,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let base = url.trim_end_matches('/');
    let client = reqwest::Client::new();
    let out: serde_json::Value = match action {
        SynapseAction::Status => {
            client
                .get(format!("{base}/api/v1/synapse/status"))
                .send()
                .await?
                .json()
                .await?
        }
        SynapseAction::Ingest {
            model,
            file,
            rank,
            outliers,
        } => {
            let abs_path =
                std::fs::canonicalize(&file).unwrap_or_else(|_| std::path::PathBuf::from(&file));
            client
                .post(format!("{base}/api/v1/synapse/ingest/safetensors"))
                .json(&serde_json::json!({
                    "model_id": model,
                    "file_path": abs_path.to_string_lossy(),
                    "target_rank": rank,
                    "max_outliers": outliers,
                }))
                .send()
                .await?
                .json()
                .await?
        }
        SynapseAction::Layers { model } => {
            let esc = model.replace('\'', "''");
            client
                .post(format!("{base}/api/v1/query"))
                .json(&serde_json::json!({
                    "sql": format!(
                        "SELECT tensor_name, layer_idx, source_dtype, rows, cols, bits_per_weight, compression_vs_fp16, reconstruction_cosine FROM synapse_tensors WHERE model_id = '{esc}' ORDER BY layer_idx ASC, tensor_name ASC"
                    )
                }))
                .send()
                .await?
                .json()
                .await?
        }
        SynapseAction::SelfEdit { phrase, fact } => {
            let key_vec: Vec<f32> = (0..16)
                .map(|i| if i % 2 == 0 { 0.5 } else { -0.5 })
                .collect();
            let target_vec: Vec<f32> = (0..16)
                .map(|i| ((i as f32 + phrase.len() as f32) * 0.25).sin())
                .collect();
            client
                .post(format!("{base}/api/v1/harness/self-edit"))
                .json(&serde_json::json!({
                    "trigger_phrase": phrase,
                    "correction_fact": fact,
                    "key_vec": key_vec,
                    "target_vec": target_vec,
                }))
                .send()
                .await?
                .json()
                .await?
        }
    };
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}

async fn run_ward_cli(
    url: &str,
    action: WardAction,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use base64::Engine;
    let base = url.trim_end_matches('/');
    let client = reqwest::Client::new();
    match action {
        WardAction::Ledger => {
            let out: serde_json::Value = client
                .get(format!("{base}/api/v1/ward/ledger"))
                .send()
                .await?
                .json()
                .await?;
            println!("{}", serde_json::to_string_pretty(&out)?);
        }
        WardAction::Mint {
            id,
            subject,
            max_nj,
            read_only,
            airgap,
        } => {
            let mut caveats = Vec::new();
            if let Some(nj) = max_nj {
                caveats.push(serde_json::json!({ "kind": "max_total_nanojoules", "value": nj }));
            }
            if read_only {
                caveats.push(serde_json::json!({ "kind": "read_only_fs", "value": true }));
            }
            if airgap {
                caveats.push(serde_json::json!({ "kind": "allow_egress_hosts", "value": [] }));
            }
            let out: serde_json::Value = client
                .post(format!("{base}/api/v1/ward/tokens/mint"))
                .json(&serde_json::json!({
                    "root_token_id": id,
                    "subject": subject,
                    "caveats": caveats,
                }))
                .send()
                .await?
                .json()
                .await?;
            println!("{}", serde_json::to_string_pretty(&out)?);
        }
        WardAction::Exec { fabric, command } => {
            let minted: serde_json::Value = client
                .post(format!("{base}/api/v1/ward/tokens/mint"))
                .json(&serde_json::json!({
                    "root_token_id": "cli-ephemeral-exec",
                    "subject": "jouledb-ward-cli",
                    "caveats": [
                        { "kind": "read_only_fs", "value": true },
                        { "kind": "allow_egress_hosts", "value": [] }
                    ]
                }))
                .send()
                .await?
                .json()
                .await?;
            let token = minted
                .get("token")
                .cloned()
                .ok_or("failed to mint ephemeral capability token")?;
            let out: serde_json::Value = client
                .post(format!("{base}/api/v1/ward/exec"))
                .json(&serde_json::json!({
                    "token": token,
                    "command": command,
                    "memoize": true,
                    "route_across_fabric": fabric,
                }))
                .send()
                .await?
                .json()
                .await?;
            println!("{}", serde_json::to_string_pretty(&out)?);
        }
        WardAction::JcapExport { id, out, model } => {
            let resp: serde_json::Value = client
                .post(format!("{base}/api/v1/ward/jcap/export"))
                .json(&serde_json::json!({
                    "bundle_id": id,
                    "synapse_model_id": model,
                }))
                .send()
                .await?
                .json()
                .await?;
            if let Some(b64) = resp.get("jcap_base64").and_then(|v| v.as_str()) {
                let bytes = base64::engine::general_purpose::STANDARD.decode(b64)?;
                std::fs::write(&out, &bytes)?;
                println!(
                    "Exported verified .jcap bundle '{}' ({} bytes, sha256={})",
                    out,
                    bytes.len(),
                    resp.get("payload_sha256")
                        .and_then(|v| v.as_str())
                        .unwrap_or("?")
                );
            } else {
                println!("{}", serde_json::to_string_pretty(&resp)?);
            }
        }
    }
    Ok(())
}

async fn run_cascade_cli(
    url: &str,
    model: &str,
    prompt: &str,
    fabric: bool,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let base = url.trim_end_matches('/');
    let client = reqwest::Client::new();
    let out: serde_json::Value = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": model,
            "messages": [{ "role": "user", "content": prompt }],
            "x_route_across_fabric": fabric,
        }))
        .send()
        .await?
        .json()
        .await?;
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}

fn run_amorphic_cli(
    action: AmorphicAction,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    match action {
        AmorphicAction::StreamCorpus {
            dir,
            generate_bytes,
            entities,
            cleanup,
        } => {
            let p = std::path::Path::new(&dir);
            let rec = if generate_bytes > 0 {
                let r = joule_db_amorphic::streaming_continuum::generate_and_stream_heterogeneous_corpus(
                    p,
                    generate_bytes,
                    entities,
                    &[],
                )?;
                if cleanup {
                    let _ = std::fs::remove_dir_all(p);
                }
                r
            } else {
                let mut eng =
                    joule_db_amorphic::streaming_continuum::StreamingAmorphicContinuum::new();
                eng.ingest_directory(p)?
            };
            println!("{}", serde_json::to_string_pretty(&rec)?);
        }
        AmorphicAction::MorphSuite => {
            let morph_rec = joule_db_amorphic::morph_rules::run_programmable_morph_suite()?;
            println!("{}", serde_json::to_string_pretty(&morph_rec)?);
        }
    }
    Ok(())
}

fn truncate(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}
