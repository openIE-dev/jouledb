//! Bridge between the JouleDB server and the energy receipt ledger.
//!
//! Initializes the ledger subsystem (collector, committer, backend) from
//! `ServerConfig` and wires it into the energy-aware executor.

use std::path::PathBuf;
use std::sync::Arc;

use joule_db_ledger::{
    BatchCommitter, CarbonConfig, CollectorMetrics, FileLedgerBackend, LedgerConfig,
    MemoryLedgerBackend, ReceiptCollector, ReceiptStore,
};
use tokio::sync::RwLock;

use crate::ServerConfig;
use crate::energy::LedgerMetrics;

/// Ledger subsystem handles, returned by `init_ledger()`.
pub struct LedgerHandles {
    /// The collector to pass to `EnergyAwareExecutor`.
    pub collector: Arc<ReceiptCollector>,
    /// The receipt store for verification endpoint lookups.
    pub store: Arc<RwLock<ReceiptStore>>,
    /// Collector metrics for Prometheus integration.
    pub collector_metrics: CollectorMetrics,
    /// Background committer task handle.
    pub committer_handle: tokio::task::JoinHandle<()>,
}

/// Spawn a background task that periodically updates `LedgerMetrics` from
/// the collector metrics and receipt store (every 10 seconds).
pub fn spawn_ledger_metrics_refresh(
    ledger_metrics: Arc<LedgerMetrics>,
    collector_metrics: CollectorMetrics,
    store: Arc<RwLock<ReceiptStore>>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(10));
        loop {
            interval.tick().await;
            let store_guard = store.read().await;
            ledger_metrics.update(&collector_metrics, &store_guard);
        }
    })
}

/// Initialize the energy receipt ledger from server configuration.
///
/// Returns `None` if `config.enable_ledger` is false.
pub fn init_ledger(config: &ServerConfig) -> Option<LedgerHandles> {
    if !config.enable_ledger {
        return None;
    }

    let carbon = CarbonConfig {
        grid_factor_kg_co2e_per_kwh: config.ledger_grid_factor.unwrap_or(0.4),
        grid_region: config
            .ledger_grid_region
            .clone()
            .unwrap_or_else(|| "UNKNOWN".to_string()),
        grid_factor_source: "configured".to_string(),
    };

    let ledger_config = LedgerConfig {
        batch_max_receipts: config.ledger_batch_max_receipts,
        batch_max_interval_secs: config.ledger_batch_interval_secs,
        carbon,
        issuer: config
            .raft_node_id
            .clone()
            .unwrap_or_else(|| "standalone".to_string()),
        enable_tenant_isolation: false,
        ..Default::default()
    };

    let channel_size = ledger_config.batch_max_receipts * 2;
    let (tx, rx) = tokio::sync::mpsc::channel(channel_size);

    let (committer_handle, store) = if let Some(ref dir) = config.ledger_dir {
        let path = PathBuf::from(dir).join("commitments.jsonl");
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        match FileLedgerBackend::new(path) {
            Ok(backend) => {
                let backend = Arc::new(backend);
                BatchCommitter::spawn(ledger_config.clone(), backend, rx)
            }
            Err(e) => {
                tracing::error!(
                    "Failed to init file ledger backend: {}, falling back to memory",
                    e
                );
                let backend = Arc::new(MemoryLedgerBackend::new());
                BatchCommitter::spawn(ledger_config.clone(), backend, rx)
            }
        }
    } else {
        let backend = Arc::new(MemoryLedgerBackend::new());
        BatchCommitter::spawn(ledger_config.clone(), backend, rx)
    };

    let collector = Arc::new(ReceiptCollector::new(ledger_config, tx));
    let collector_metrics = collector.metrics();

    tracing::info!(
        dir = config.ledger_dir.as_deref().unwrap_or("(in-memory)"),
        batch_max = config.ledger_batch_max_receipts,
        interval_secs = config.ledger_batch_interval_secs,
        "Energy receipt ledger initialized"
    );

    Some(LedgerHandles {
        collector,
        store,
        collector_metrics,
        committer_handle,
    })
}
