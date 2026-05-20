use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::mpsc;

use crate::carbon::{self, CarbonConfig};
use crate::error::LedgerError;
use crate::receipt::{ExecutionStage, LedgerEnergyReceipt};

/// Metrics exposed by the ReceiptCollector for operational monitoring.
#[derive(Debug, Clone)]
pub struct CollectorMetrics {
    dropped: Arc<AtomicU64>,
    sent: Arc<AtomicU64>,
}

impl CollectorMetrics {
    /// Number of receipts dropped due to backpressure (channel full).
    pub fn receipts_dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    /// Number of receipts successfully sent to the committer.
    pub fn receipts_sent(&self) -> u64 {
        self.sent.load(Ordering::Relaxed)
    }
}

/// Configuration for the energy receipt ledger.
#[derive(Debug, Clone)]
pub struct LedgerConfig {
    /// Maximum receipts per batch before auto-commit.
    pub batch_max_receipts: usize,
    /// Maximum time window (seconds) before auto-commit.
    pub batch_max_interval_secs: u64,
    /// Carbon conversion configuration.
    pub carbon: CarbonConfig,
    /// Issuer node identifier.
    pub issuer: String,
    /// When true, receipts are batched per-tenant instead of in a single stream.
    pub enable_tenant_isolation: bool,
    /// Maximum retry attempts before permanently dropping a failed batch.
    pub retry_max_attempts: u32,
    /// Maximum number of batches held in the retry queue.
    pub retry_max_queue: usize,
}

impl Default for LedgerConfig {
    fn default() -> Self {
        Self {
            batch_max_receipts: 1000,
            batch_max_interval_secs: 60,
            carbon: CarbonConfig::default(),
            issuer: "unknown".to_string(),
            enable_tenant_isolation: false,
            retry_max_attempts: 5,
            retry_max_queue: 64,
        }
    }
}

/// Collects energy observations and forwards them to the batch committer.
///
/// The collector is the integration point between the query executor and
/// the ledger subsystem. It receives per-query energy data, builds receipts,
/// and sends them to the committer via an mpsc channel.
pub struct ReceiptCollector {
    config: Arc<LedgerConfig>,
    tx: mpsc::Sender<LedgerEnergyReceipt>,
    dropped_count: Arc<AtomicU64>,
    sent_count: Arc<AtomicU64>,
}

impl ReceiptCollector {
    /// Create a new collector.
    ///
    /// The `tx` sender connects to the `BatchCommitter`'s receiver.
    pub fn new(config: LedgerConfig, tx: mpsc::Sender<LedgerEnergyReceipt>) -> Self {
        Self {
            config: Arc::new(config),
            tx,
            dropped_count: Arc::new(AtomicU64::new(0)),
            sent_count: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Get collector metrics for Prometheus integration.
    pub fn metrics(&self) -> CollectorMetrics {
        CollectorMetrics {
            dropped: self.dropped_count.clone(),
            sent: self.sent_count.clone(),
        }
    }

    /// Record a query's energy data as a receipt and send it to the committer.
    ///
    /// This is the primary API called from `EnergyAwareExecutor` after each query.
    /// Uses `try_send()` to avoid blocking the synchronous query executor.
    pub fn record(
        &self,
        qid: &str,
        tenant_id: &str,
        workload_tag: Option<&str>,
        energy_joules: f64,
        device_target: &str,
        algorithm_type: &str,
        timestamp_start: chrono::DateTime<chrono::Utc>,
        timestamp_end: chrono::DateTime<chrono::Utc>,
    ) -> Result<LedgerEnergyReceipt, LedgerError> {
        self.record_with_stages(
            qid,
            tenant_id,
            workload_tag,
            energy_joules,
            device_target,
            algorithm_type,
            timestamp_start,
            timestamp_end,
            None,
        )
    }

    /// Record a query's energy data with optional per-stage breakdown.
    pub fn record_with_stages(
        &self,
        qid: &str,
        tenant_id: &str,
        workload_tag: Option<&str>,
        energy_joules: f64,
        device_target: &str,
        algorithm_type: &str,
        timestamp_start: chrono::DateTime<chrono::Utc>,
        timestamp_end: chrono::DateTime<chrono::Utc>,
        stage_energy: Option<std::collections::HashMap<ExecutionStage, f64>>,
    ) -> Result<LedgerEnergyReceipt, LedgerError> {
        let kwh = carbon::joules_to_kwh(energy_joules);
        let kg_co2e = carbon::joules_to_kg_co2e(energy_joules, &self.config.carbon);
        let receipt_id = LedgerEnergyReceipt::compute_id(qid, tenant_id, &timestamp_start);

        let mut receipt = LedgerEnergyReceipt {
            receipt_id,
            qid: qid.to_string(),
            tenant_id: tenant_id.to_string(),
            workload_tag: workload_tag.map(String::from),
            branch_id: None,
            energy_joules_total: energy_joules,
            energy_joules_by_stage: Default::default(),
            kwh,
            kg_co2e,
            grid_region: self.config.carbon.grid_region.clone(),
            grid_factor_source: self.config.carbon.grid_factor_source.clone(),
            timestamp_start,
            timestamp_end,
            device_target: device_target.to_string(),
            algorithm_type: algorithm_type.to_string(),
        };

        if let Some(stages) = stage_energy {
            receipt.set_stage_energy(stages);
        }

        match self.tx.try_send(receipt.clone()) {
            Ok(()) => {
                self.sent_count.fetch_add(1, Ordering::Relaxed);
            }
            Err(mpsc::error::TrySendError::Full(_)) => {
                self.dropped_count.fetch_add(1, Ordering::Relaxed);
                tracing::warn!(
                    receipt_id = %receipt.receipt_id,
                    "Ledger receipt dropped: channel full (backpressure)"
                );
                // Return the receipt anyway — caller may still want it
                return Ok(receipt);
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                return Err(LedgerError::ChannelClosed);
            }
        }

        Ok(receipt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};

    #[tokio::test]
    async fn record_creates_receipt() {
        let (tx, mut rx) = mpsc::channel(100);
        let collector = ReceiptCollector::new(LedgerConfig::default(), tx);

        let ts = Utc.with_ymd_and_hms(2026, 2, 24, 12, 0, 0).unwrap();
        let receipt = collector
            .record(
                "q1",
                "tenant1",
                Some("oltp"),
                0.005,
                "cpu",
                "btree",
                ts,
                ts + chrono::Duration::milliseconds(50),
            )
            .unwrap();

        assert_eq!(receipt.qid, "q1");
        assert_eq!(receipt.tenant_id, "tenant1");
        assert!((receipt.energy_joules_total - 0.005).abs() < 1e-15);
        assert!(receipt.kwh > 0.0);
        assert!(receipt.kg_co2e > 0.0);

        // Should be in channel
        let received = rx.recv().await.unwrap();
        assert_eq!(received.receipt_id, receipt.receipt_id);
    }

    #[tokio::test]
    async fn record_returns_err_on_closed_channel() {
        let (tx, rx) = mpsc::channel(1);
        drop(rx); // close receiver

        let collector = ReceiptCollector::new(LedgerConfig::default(), tx);
        let ts = Utc.with_ymd_and_hms(2026, 2, 24, 12, 0, 0).unwrap();
        let result = collector.record(
            "q1",
            "tenant1",
            None,
            0.005,
            "cpu",
            "btree",
            ts,
            ts + chrono::Duration::milliseconds(50),
        );
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn backpressure_returns_ok_and_increments_counter() {
        // Channel size 1, fill it, then try to send another
        let (tx, _rx) = mpsc::channel(1);
        let collector = ReceiptCollector::new(LedgerConfig::default(), tx);
        let ts = Utc.with_ymd_and_hms(2026, 2, 24, 12, 0, 0).unwrap();

        // First send fills the channel
        let r1 = collector.record("q1", "t1", None, 0.001, "cpu", "btree", ts, ts);
        assert!(r1.is_ok());
        assert_eq!(collector.metrics().receipts_sent(), 1);

        // Second send hits backpressure — returns Ok, not Err
        let r2 = collector.record(
            "q2",
            "t1",
            None,
            0.002,
            "cpu",
            "btree",
            ts + chrono::Duration::milliseconds(1),
            ts + chrono::Duration::milliseconds(2),
        );
        assert!(r2.is_ok());
        assert_eq!(collector.metrics().receipts_dropped(), 1);
        assert_eq!(collector.metrics().receipts_sent(), 1); // only 1 actually sent
    }

    #[tokio::test]
    async fn sent_counter_increments() {
        let (tx, _rx) = mpsc::channel(100);
        let collector = ReceiptCollector::new(LedgerConfig::default(), tx);
        let ts = Utc.with_ymd_and_hms(2026, 2, 24, 12, 0, 0).unwrap();

        for i in 0..5 {
            collector
                .record(
                    &format!("q{}", i),
                    "t1",
                    None,
                    0.001,
                    "cpu",
                    "btree",
                    ts + chrono::Duration::milliseconds(i as i64),
                    ts + chrono::Duration::milliseconds(i as i64 + 1),
                )
                .unwrap();
        }
        assert_eq!(collector.metrics().receipts_sent(), 5);
        assert_eq!(collector.metrics().receipts_dropped(), 0);
    }

    #[tokio::test]
    async fn record_carbon_uses_config() {
        let config = LedgerConfig {
            carbon: CarbonConfig {
                grid_factor_kg_co2e_per_kwh: 0.05,
                grid_region: "FR".to_string(),
                grid_factor_source: "rte-france".to_string(),
            },
            ..Default::default()
        };
        let (tx, _rx) = mpsc::channel(100);
        let collector = ReceiptCollector::new(config, tx);

        let ts = Utc.with_ymd_and_hms(2026, 2, 24, 12, 0, 0).unwrap();
        let receipt = collector
            .record("q1", "t1", None, 3_600_000.0, "cpu", "scan", ts, ts)
            .unwrap();

        assert_eq!(receipt.grid_region, "FR");
        // 1 kWh * 0.05 = 0.05 kgCO2e
        assert!((receipt.kg_co2e - 0.05).abs() < 1e-10);
    }
}
