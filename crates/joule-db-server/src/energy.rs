//! Energy metrics and monitoring integration for JouleDB Server.
//!
//! Bridges `joule-db-energy` into the server's metrics infrastructure.
//! All energy metrics register into the existing `MetricsRegistry` and
//! appear on the Prometheus `/metrics` endpoint automatically.

use crate::metrics::{Counter, Gauge, Histogram, Labels, MetricsRegistry};
use joule_db_energy::{EnergyObservation, EnergySnapshot, ThermalState};
use std::sync::Arc;

/// Energy-specific Prometheus metrics.
pub struct EnergyMetrics {
    /// Total energy consumed by database operations (joules)
    pub energy_joules_total: Arc<Counter>,
    /// Energy per query (histogram in joules)
    pub energy_joules_per_query: Arc<Histogram>,
    /// Current system power draw (watts)
    pub power_draw_watts: Arc<Gauge>,
    /// Current thermal state (0=Nominal, 1=Fair, 2=Serious, 3=Critical)
    pub thermal_state: Arc<Gauge>,
    /// CPU utilization (0-100)
    pub cpu_utilization: Arc<Gauge>,
    /// GPU utilization (0-100)
    pub gpu_utilization: Arc<Gauge>,
    /// Memory pressure (0-100)
    pub memory_pressure: Arc<Gauge>,
    /// Number of queries that exceeded their energy budget
    pub energy_budget_exceeded_total: Arc<Counter>,
    /// GPU availability (0 or 1)
    pub gpu_available: Arc<Gauge>,
    /// NPU utilization (0-100)
    pub npu_utilization: Arc<Gauge>,
    /// NPU availability (0 or 1)
    pub npu_available: Arc<Gauge>,
    /// TPU utilization (0-100)
    pub tpu_utilization: Arc<Gauge>,
    /// TPU availability (0 or 1)
    pub tpu_available: Arc<Gauge>,
    /// Battery charge percentage (0-100, -1 if no battery)
    pub battery_percent: Arc<Gauge>,
    /// Cumulative energy since server start (joules)
    pub cumulative_joules: Arc<Gauge>,
    /// Total queries tracked for energy
    pub queries_tracked: Arc<Counter>,
}

/// Energy histogram bucket boundaries (in joules).
/// Range: sub-millijoule (point reads) to multi-joule (large scans).
const ENERGY_BUCKETS: [f64; 10] = [
    0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0,
];

impl EnergyMetrics {
    /// Register all energy metrics into the existing metrics registry.
    pub fn new(registry: &MetricsRegistry) -> Self {
        Self {
            energy_joules_total: registry.register_counter(
                "energy_joules_total",
                "Total energy consumed by database operations in joules",
                Labels::new(),
            ),
            energy_joules_per_query: registry.register_histogram_with_buckets(
                "energy_joules_per_query",
                "Energy consumed per query in joules",
                Labels::new(),
                &ENERGY_BUCKETS,
            ),
            power_draw_watts: registry.register_gauge(
                "power_draw_watts",
                "Current system power draw in watts",
                Labels::new(),
            ),
            thermal_state: registry.register_gauge(
                "thermal_state",
                "Current thermal state (0=Nominal, 1=Fair, 2=Serious, 3=Critical)",
                Labels::new(),
            ),
            cpu_utilization: registry.register_gauge(
                "cpu_utilization_percent",
                "CPU utilization percentage (0-100)",
                Labels::new(),
            ),
            gpu_utilization: registry.register_gauge(
                "gpu_utilization_percent",
                "GPU utilization percentage (0-100)",
                Labels::new(),
            ),
            memory_pressure: registry.register_gauge(
                "memory_pressure_percent",
                "Memory pressure percentage (0-100)",
                Labels::new(),
            ),
            energy_budget_exceeded_total: registry.register_counter(
                "energy_budget_exceeded_total",
                "Number of queries that exceeded their energy budget",
                Labels::new(),
            ),
            gpu_available: registry.register_gauge(
                "gpu_available",
                "Whether a GPU is available for compute (0 or 1)",
                Labels::new(),
            ),
            npu_utilization: registry.register_gauge(
                "npu_utilization_percent",
                "NPU utilization percentage (0-100)",
                Labels::new(),
            ),
            npu_available: registry.register_gauge(
                "npu_available",
                "Whether an NPU is available for compute (0 or 1)",
                Labels::new(),
            ),
            tpu_utilization: registry.register_gauge(
                "tpu_utilization_percent",
                "TPU utilization percentage (0-100)",
                Labels::new(),
            ),
            tpu_available: registry.register_gauge(
                "tpu_available",
                "Whether a TPU is available for compute (0 or 1)",
                Labels::new(),
            ),
            battery_percent: registry.register_gauge(
                "battery_percent",
                "Battery charge percentage (0-100, -1 if no battery)",
                Labels::new(),
            ),
            cumulative_joules: registry.register_gauge(
                "energy_cumulative_joules",
                "Cumulative energy consumed since server start in joules",
                Labels::new(),
            ),
            queries_tracked: registry.register_counter(
                "energy_queries_tracked_total",
                "Total number of queries tracked for energy consumption",
                Labels::new(),
            ),
        }
    }

    /// Update gauge metrics from a hardware energy snapshot.
    pub fn update_from_snapshot(&self, snapshot: &EnergySnapshot) {
        // Gauge::set takes i64, so we scale floats to integer representations
        self.power_draw_watts
            .set((snapshot.power_watts * 1000.0) as i64); // milliwatts precision
        self.thermal_state.set(match snapshot.thermal_state {
            ThermalState::Nominal => 0,
            ThermalState::Fair => 1,
            ThermalState::Serious => 2,
            ThermalState::Critical => 3,
        });
        self.cpu_utilization
            .set((snapshot.cpu_utilization * 100.0) as i64);
        self.gpu_utilization
            .set((snapshot.gpu_utilization * 100.0) as i64);
        self.memory_pressure
            .set((snapshot.memory_pressure * 100.0) as i64);
        self.gpu_available
            .set(if snapshot.gpu_available { 1 } else { 0 });
        self.npu_utilization
            .set((snapshot.npu_utilization * 100.0) as i64);
        self.npu_available
            .set(if snapshot.npu_available { 1 } else { 0 });
        self.tpu_utilization
            .set((snapshot.tpu_utilization * 100.0) as i64);
        self.tpu_available
            .set(if snapshot.tpu_available { 1 } else { 0 });
        self.battery_percent
            .set(snapshot.battery_percent.map(|p| p as i64).unwrap_or(-1));
        self.cumulative_joules
            .set((snapshot.cumulative_joules * 1000.0) as i64); // millijoules precision
    }

    /// Record an energy observation from the OperationEnergyTracker.
    pub fn record_observation(&self, obs: &EnergyObservation) {
        self.energy_joules_per_query.observe(obs.estimated_joules);
        self.queries_tracked.inc();

        // Accumulate into total (approximate: convert joules to micro-joules as u64)
        let micro_joules = (obs.estimated_joules * 1_000_000.0) as u64;
        self.energy_joules_total.inc_by(micro_joules);
    }
}

// ============================================================================
// Ledger-specific Prometheus metrics
// ============================================================================

/// Ledger metrics registered into the existing Prometheus endpoint.
pub struct LedgerMetrics {
    /// Receipts successfully sent to committer
    pub receipts_sent: Arc<Gauge>,
    /// Receipts dropped due to backpressure
    pub receipts_dropped: Arc<Gauge>,
    /// Batches committed to backend
    pub batches_committed: Arc<Gauge>,
    /// Total energy tracked through ledger (millijoules for integer precision)
    pub total_energy_millijoules: Arc<Gauge>,
}

impl LedgerMetrics {
    /// Register ledger metrics into the existing metrics registry.
    pub fn new(registry: &MetricsRegistry) -> Self {
        Self {
            receipts_sent: registry.register_gauge(
                "ledger_receipts_sent",
                "Energy receipts sent to ledger committer",
                Labels::new(),
            ),
            receipts_dropped: registry.register_gauge(
                "ledger_receipts_dropped",
                "Energy receipts dropped due to backpressure",
                Labels::new(),
            ),
            batches_committed: registry.register_gauge(
                "ledger_batches_committed",
                "Total Merkle batches committed to ledger backend",
                Labels::new(),
            ),
            total_energy_millijoules: registry.register_gauge(
                "ledger_total_energy_millijoules",
                "Total energy tracked through ledger (millijoules)",
                Labels::new(),
            ),
        }
    }

    /// Update from collector metrics + receipt store snapshot.
    pub fn update(
        &self,
        collector_metrics: &joule_db_ledger::CollectorMetrics,
        store: &joule_db_ledger::ReceiptStore,
    ) {
        self.receipts_sent
            .set(collector_metrics.receipts_sent() as i64);
        self.receipts_dropped
            .set(collector_metrics.receipts_dropped() as i64);
        self.batches_committed.set(store.batches.len() as i64);

        let total_energy: f64 = store
            .receipts
            .values()
            .map(|(r, _, _)| r.energy_joules_total)
            .sum();
        self.total_energy_millijoules
            .set((total_energy * 1000.0) as i64);
    }
}

/// JSON response for the /energy endpoint.
#[derive(Debug, serde::Serialize)]
pub struct EnergyStatusResponse {
    pub power_watts: f64,
    pub thermal_state: String,
    pub cpu_utilization: f64,
    pub gpu_available: bool,
    pub gpu_utilization: f64,
    pub npu_available: bool,
    pub npu_utilization: f64,
    pub tpu_available: bool,
    pub tpu_utilization: f64,
    pub memory_pressure: f64,
    pub battery_percent: Option<f64>,
    pub battery_charging: bool,
    pub advisor_hints: Vec<String>,
    pub cumulative_energy_joules: f64,
    pub queries_tracked: u64,
}

impl EnergyStatusResponse {
    pub fn from_snapshot(
        snapshot: &EnergySnapshot,
        hints: &[joule_db_energy::ExecutionHint],
        queries_tracked: u64,
    ) -> Self {
        Self {
            power_watts: snapshot.power_watts,
            thermal_state: snapshot.thermal_state.to_string(),
            cpu_utilization: snapshot.cpu_utilization,
            gpu_available: snapshot.gpu_available,
            gpu_utilization: snapshot.gpu_utilization,
            npu_available: snapshot.npu_available,
            npu_utilization: snapshot.npu_utilization,
            tpu_available: snapshot.tpu_available,
            tpu_utilization: snapshot.tpu_utilization,
            memory_pressure: snapshot.memory_pressure,
            battery_percent: snapshot.battery_percent,
            battery_charging: snapshot.battery_charging,
            advisor_hints: hints.iter().map(|h| h.to_string()).collect(),
            cumulative_energy_joules: snapshot.cumulative_joules,
            queries_tracked,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_energy_metrics_creation() {
        let registry = MetricsRegistry::new();
        let metrics = EnergyMetrics::new(&registry);
        assert_eq!(metrics.energy_joules_total.get(), 0);
        assert_eq!(metrics.queries_tracked.get(), 0);
    }

    #[test]
    fn test_update_from_snapshot() {
        let registry = MetricsRegistry::new();
        let metrics = EnergyMetrics::new(&registry);

        let snapshot = EnergySnapshot {
            power_watts: 18.5,
            cpu_utilization: 0.35,
            thermal_state: ThermalState::Nominal,
            memory_pressure: 0.2,
            gpu_available: true,
            gpu_utilization: 0.12,
            battery_percent: Some(78.0),
            ..EnergySnapshot::default()
        };

        metrics.update_from_snapshot(&snapshot);

        assert_eq!(metrics.power_draw_watts.get(), 18500); // milliwatts
        assert_eq!(metrics.thermal_state.get(), 0);
        assert_eq!(metrics.cpu_utilization.get(), 35);
        assert_eq!(metrics.gpu_utilization.get(), 12);
        assert_eq!(metrics.gpu_available.get(), 1);
        assert_eq!(metrics.battery_percent.get(), 78);
    }

    #[test]
    fn test_record_observation() {
        let registry = MetricsRegistry::new();
        let metrics = EnergyMetrics::new(&registry);

        let obs = EnergyObservation {
            operation: joule_db_energy::OperationType::Search,
            device: joule_db_energy::DeviceTarget::Cpu,
            algorithm: joule_db_energy::AlgorithmType::Hdc,
            duration_secs: 0.01,
            estimated_joules: 0.005,
            power_watts_at_start: 20.0,
        };

        metrics.record_observation(&obs);
        assert_eq!(metrics.queries_tracked.get(), 1);
        assert!(metrics.energy_joules_total.get() > 0);
    }

    #[test]
    fn test_energy_status_response() {
        let snapshot = EnergySnapshot {
            power_watts: 18.5,
            thermal_state: ThermalState::Nominal,
            cpu_utilization: 0.35,
            gpu_available: true,
            gpu_utilization: 0.12,
            memory_pressure: 0.2,
            battery_percent: Some(78.0),
            battery_charging: false,
            cumulative_joules: 100.0,
            ..EnergySnapshot::default()
        };

        let hints = vec![joule_db_energy::ExecutionHint::Normal];
        let response = EnergyStatusResponse::from_snapshot(&snapshot, &hints, 500);

        assert_eq!(response.power_watts, 18.5);
        assert_eq!(response.thermal_state, "Nominal");
        assert_eq!(response.advisor_hints, vec!["normal"]);
        assert_eq!(response.queries_tracked, 500);
    }

    // --- LedgerMetrics tests ---

    #[test]
    fn test_ledger_metrics_zero_initial() {
        let registry = MetricsRegistry::new();
        let lm = LedgerMetrics::new(&registry);
        assert_eq!(lm.receipts_sent.get(), 0);
        assert_eq!(lm.receipts_dropped.get(), 0);
        assert_eq!(lm.batches_committed.get(), 0);
        assert_eq!(lm.total_energy_millijoules.get(), 0);
    }

    #[test]
    fn test_ledger_metrics_update() {
        use joule_db_ledger::{CollectorMetrics, LedgerConfig, ReceiptCollector, ReceiptStore};
        use std::collections::HashMap;

        let registry = MetricsRegistry::new();
        let lm = LedgerMetrics::new(&registry);

        // Build collector metrics from a real collector
        let (tx, _rx) = tokio::sync::mpsc::channel(100);
        let collector = ReceiptCollector::new(LedgerConfig::default(), tx);
        let cm = collector.metrics();

        // Build a store with a receipt
        let mut store = ReceiptStore::new();
        let receipt = joule_db_ledger::LedgerEnergyReceipt {
            receipt_id: "r1".to_string(),
            qid: "q1".to_string(),
            tenant_id: "t1".to_string(),
            workload_tag: None,
            energy_joules_total: 2.5,
            energy_joules_by_stage: HashMap::new(),
            kwh: 0.0,
            kg_co2e: 0.0,
            grid_region: "US".to_string(),
            grid_factor_source: "test".to_string(),
            timestamp_start: chrono::Utc::now(),
            timestamp_end: chrono::Utc::now(),
            device_target: "cpu".to_string(),
            algorithm_type: "btree".to_string(),
            branch_id: None,
        };
        store
            .receipts
            .insert("r1".to_string(), (receipt, "b1".to_string(), 0));

        lm.update(&cm, &store);

        assert_eq!(lm.receipts_sent.get(), 0); // nothing sent yet
        assert_eq!(lm.batches_committed.get(), 0); // no batches in store
        assert_eq!(lm.total_energy_millijoules.get(), 2500); // 2.5 J = 2500 mJ
    }

    #[test]
    fn test_ledger_metrics_prometheus_format() {
        use crate::metrics::PrometheusExporter;
        let registry = MetricsRegistry::new();
        let _lm = LedgerMetrics::new(&registry);

        let exporter = PrometheusExporter::new(Arc::new(registry));
        let output = exporter.export();
        assert!(output.contains("ledger_receipts_sent"));
        assert!(output.contains("ledger_receipts_dropped"));
        assert!(output.contains("ledger_batches_committed"));
        assert!(output.contains("ledger_total_energy_millijoules"));
    }
}
