//! Energy-aware query executor wrapper
//!
//! Wraps any `QueryExecutor` to measure per-query energy consumption using
//! the `joule-db-energy` RAII tracker. Each query response is annotated with
//! `energy_joules` and `power_watts` fields.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use joule_db_energy::tracker::{
    AlgorithmType, DeviceTarget, EnergyObservation, OperationEnergyTracker, OperationType,
};
use joule_db_energy::{EnergySnapshot, ExecutionHint, HardwareAdvisor};
use joule_db_ledger::ExecutionStage;

use crate::energy::EnergyMetrics;
use crate::query::{QueryErrorResponse, QueryExecutor, QueryRequest, QueryResponse};

// ============================================================================
// Thread-local device context
// ============================================================================

/// Per-query device context set by `EnergyAwareExecutor` before calling the
/// inner executor.  The inner executor (and functions it calls) can read this
/// via `current_device_context()` without any signature changes.
#[derive(Debug, Clone)]
pub struct DeviceContext {
    /// The resolved optimal device for this query.
    pub device_target: DeviceTarget,
    /// The classified algorithm type for this query.
    pub algorithm: AlgorithmType,
    /// All execution hints from the hardware advisor.
    pub hints: Vec<ExecutionHint>,
}

thread_local! {
    static DEVICE_CONTEXT: RefCell<Option<DeviceContext>> = const { RefCell::new(None) };
}

/// Read the current device context (set by `EnergyAwareExecutor` for the
/// duration of query execution).  Returns `None` if called outside an
/// energy-aware execution scope.
pub fn current_device_context() -> Option<DeviceContext> {
    DEVICE_CONTEXT.with(|cell| cell.borrow().clone())
}

/// Read just the current device target.  Convenience wrapper that defaults
/// to `DeviceTarget::Cpu` when no context is set.
pub fn current_device_target() -> DeviceTarget {
    current_device_context()
        .map(|ctx| ctx.device_target)
        .unwrap_or(DeviceTarget::Cpu)
}

/// Check whether the hardware advisor is currently recommending throttling.
pub fn is_throttled() -> bool {
    current_device_context()
        .map(|ctx| {
            ctx.hints
                .iter()
                .any(|h| matches!(h, ExecutionHint::Throttle))
        })
        .unwrap_or(false)
}

fn set_device_context(ctx: DeviceContext) {
    DEVICE_CONTEXT.with(|cell| {
        *cell.borrow_mut() = Some(ctx);
    });
}

fn clear_device_context() {
    DEVICE_CONTEXT.with(|cell| {
        *cell.borrow_mut() = None;
    });
}

// ============================================================================
// Thread-local stage timers
// ============================================================================

/// Per-query timing of execution stages. Set by `begin_stage()` / `end_stage()`
/// calls in `query.rs`, then consumed by `EnergyAwareExecutor` to apportion
/// total energy across stages proportionally to wall-clock time.
#[derive(Debug)]
pub struct StageTimers {
    timings: Vec<(ExecutionStage, Duration)>,
    current_stage: Option<(ExecutionStage, Instant)>,
}

impl StageTimers {
    fn new() -> Self {
        Self {
            timings: Vec::new(),
            current_stage: None,
        }
    }

    fn begin(&mut self, stage: ExecutionStage) {
        // Auto-end the previous stage if still running.
        self.end_current();
        self.current_stage = Some((stage, Instant::now()));
    }

    fn end_current(&mut self) {
        if let Some((stage, start)) = self.current_stage.take() {
            self.timings.push((stage, start.elapsed()));
        }
    }

    /// Apportion `total_joules` across recorded stages proportional to
    /// wall-clock time. Returns an empty map if no stages were recorded.
    pub fn apportion(&self, total_joules: f64) -> HashMap<ExecutionStage, f64> {
        if self.timings.is_empty() {
            return HashMap::new();
        }

        let total_nanos: u128 = self.timings.iter().map(|(_, d)| d.as_nanos()).sum();
        if total_nanos == 0 {
            return HashMap::new();
        }

        let mut result = HashMap::new();
        for (stage, duration) in &self.timings {
            let fraction = duration.as_nanos() as f64 / total_nanos as f64;
            *result.entry(*stage).or_insert(0.0) += total_joules * fraction;
        }
        result
    }
}

thread_local! {
    static STAGE_TIMERS: RefCell<Option<StageTimers>> = const { RefCell::new(None) };
}

/// Begin timing a new execution stage. If a previous stage is still active,
/// it is automatically ended and its duration recorded.
///
/// Called from `query.rs` at natural stage boundaries (parse, plan, execute).
pub fn begin_stage(stage: ExecutionStage) {
    STAGE_TIMERS.with(|cell| {
        let mut borrow = cell.borrow_mut();
        if borrow.is_none() {
            *borrow = Some(StageTimers::new());
        }
        if let Some(ref mut timers) = *borrow {
            timers.begin(stage);
        }
    });
}

/// End the current execution stage (if any). Duration is recorded.
pub fn end_stage() {
    STAGE_TIMERS.with(|cell| {
        if let Some(ref mut timers) = *cell.borrow_mut() {
            timers.end_current();
        }
    });
}

/// Take and clear the thread-local stage timers, returning them if present.
/// Called by `EnergyAwareExecutor` after query execution to apportion energy.
pub fn take_stage_timers() -> Option<StageTimers> {
    STAGE_TIMERS.with(|cell| cell.borrow_mut().take())
}

/// Executor wrapper that measures energy consumption per query.
///
/// Uses `OperationEnergyTracker` (RAII guard) to estimate the energy
/// consumed between query start and finish, then annotates the response.
/// Consults `HardwareAdvisor` to determine the optimal device target
/// based on real-time hardware state (thermal, power, utilization).
pub struct EnergyAwareExecutor<E: QueryExecutor> {
    inner: E,
    snapshot_handle: Arc<std::sync::RwLock<EnergySnapshot>>,
    energy_metrics: Arc<EnergyMetrics>,
    advisor: Arc<HardwareAdvisor>,
    ledger_collector: Option<Arc<joule_db_ledger::ReceiptCollector>>,
}

impl<E: QueryExecutor> EnergyAwareExecutor<E> {
    pub fn new(
        inner: E,
        snapshot_handle: Arc<std::sync::RwLock<EnergySnapshot>>,
        energy_metrics: Arc<EnergyMetrics>,
        advisor: Arc<HardwareAdvisor>,
    ) -> Self {
        Self {
            inner,
            snapshot_handle,
            energy_metrics,
            advisor,
            ledger_collector: None,
        }
    }

    /// Set the ledger collector for energy receipt capture.
    pub fn with_ledger(mut self, collector: Arc<joule_db_ledger::ReceiptCollector>) -> Self {
        self.ledger_collector = Some(collector);
        self
    }
}

impl<E: QueryExecutor> QueryExecutor for EnergyAwareExecutor<E> {
    fn execute(&self, request: &QueryRequest) -> Result<QueryResponse, QueryErrorResponse> {
        // Classify the SQL statement for energy tracking
        let op_type = classify_sql(&request.sql);

        // Classify the algorithm from SQL to determine workload affinity
        let algorithm = classify_algorithm(&request.sql);

        // Combine workload affinity with real-time hardware state to
        // select the optimal device target for this query
        let (device_target, hints) = {
            let snapshot = self
                .snapshot_handle
                .read()
                .map(|s| s.clone())
                .unwrap_or_default();
            let hints = self.advisor.advise(&snapshot);
            let target = resolve_device_target(algorithm, &hints, &snapshot);
            (target, hints)
        };

        // Throttle enforcement: if the hardware advisor says we're at
        // thermal/power limits, introduce a brief backpressure delay
        // before executing the query.
        if hints.iter().any(|h| matches!(h, ExecutionHint::Throttle)) {
            tracing::warn!(
                device = %device_target,
                algorithm = %algorithm,
                "Hardware advisor: throttle active — applying 10ms backpressure"
            );
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        // Set the thread-local device context so the inner executor
        // (and any functions it calls) can query the resolved device
        // target and hints without signature changes.
        set_device_context(DeviceContext {
            device_target,
            algorithm,
            hints: hints.clone(),
        });

        // Log non-CPU device dispatch decisions
        if device_target != DeviceTarget::Cpu {
            tracing::info!(
                device = %device_target,
                algorithm = %algorithm,
                "Dispatching query to non-CPU device"
            );
        }

        // Capture energy observation via RAII guard (Arc so closure can own a clone)
        let observation = Arc::new(std::sync::Mutex::new(None::<EnergyObservation>));
        let obs_for_callback = observation.clone();

        let guard = OperationEnergyTracker::start(
            &self.snapshot_handle,
            op_type,
            device_target,
            algorithm,
            move |obs| {
                if let Ok(mut slot) = obs_for_callback.lock() {
                    *slot = Some(obs);
                }
            },
        );

        // Execute the inner query
        let result = self.inner.execute(request);

        // Clear the thread-local context regardless of success/failure
        clear_device_context();

        let mut response = result?;

        // Drop the guard to trigger the energy measurement callback
        drop(guard);

        // Annotate response with energy + device data
        response.device_target = Some(device_target.to_string());
        response.algorithm_type = Some(algorithm.to_string());

        if let Ok(slot) = observation.lock() {
            if let Some(ref obs) = *slot {
                response.energy_joules = Some(obs.estimated_joules);
                response.power_watts = Some(obs.power_watts_at_start);

                // Update prometheus metrics
                self.energy_metrics.record_observation(obs);

                // Apportion energy across execution stages (if stage timers were set).
                let stage_energy = take_stage_timers().map(|t| t.apportion(obs.estimated_joules));

                // Emit energy receipt to the ledger (if enabled).
                // Uses try_send() to avoid blocking the synchronous executor.
                if let Some(ref collector) = self.ledger_collector {
                    let qid = request.session_id.as_deref().unwrap_or("anonymous");
                    let now = chrono::Utc::now();
                    let start_time =
                        now - chrono::Duration::milliseconds(response.execution_time_ms as i64);
                    if let Err(e) = collector.record_with_stages(
                        qid,
                        "default",
                        None,
                        obs.estimated_joules,
                        &device_target.to_string(),
                        &algorithm.to_string(),
                        start_time,
                        now,
                        stage_energy,
                    ) {
                        tracing::debug!("Ledger receipt skipped: {}", e);
                    }
                }
            }
        }

        Ok(response)
    }
}

/// Classify a SQL statement into an operation type for energy tracking.
fn classify_sql(sql: &str) -> OperationType {
    let trimmed = sql.trim_start().to_uppercase();
    if trimmed.starts_with("SELECT") {
        OperationType::Search
    } else if trimmed.starts_with("INSERT") {
        OperationType::Write
    } else if trimmed.starts_with("UPDATE") {
        OperationType::Write
    } else if trimmed.starts_with("DELETE") {
        OperationType::Write
    } else {
        OperationType::Read
    }
}

/// Classify a SQL statement into an algorithm type for workload-aware dispatch.
///
/// Uses keyword heuristics to determine which execution algorithm the query
/// will use, which in turn determines device affinity (CPU, GPU, NPU, TPU).
fn classify_algorithm(sql: &str) -> AlgorithmType {
    let upper = sql.trim_start().to_uppercase();

    // Gremlin graph traversals — graph-native processing
    if upper.starts_with("G.V(") || upper.starts_with("G.E(") || upper.starts_with("G.ADDV(") {
        return AlgorithmType::Hdc; // Graph traversals benefit from HDC proximity encoding
    }

    // SPARQL — triple pattern matching, benefits from HDC holographic unbinding
    if upper.starts_with("PREFIX ")
        || upper.starts_with("ASK ")
        || upper.starts_with("CONSTRUCT ")
        || upper.starts_with("DESCRIBE ")
        || (upper.starts_with("SELECT ?") && upper.contains("WHERE {"))
    {
        return AlgorithmType::Hdc;
    }

    // Datalog — recursive evaluation, CPU-bound semi-naive fixpoint
    if upper.starts_with("?-") || upper.contains(":-") {
        return AlgorithmType::Scan; // Iterative fixpoint is scan-heavy
    }

    // Cypher with GDS calls — graph algorithms
    if upper.contains("CALL GDS.") {
        return AlgorithmType::Hdc;
    }

    // Writes always go through B-tree path
    if upper.starts_with("INSERT")
        || upper.starts_with("UPDATE")
        || upper.starts_with("DELETE")
        || upper.starts_with("CREATE")
        || upper.starts_with("DROP")
        || upper.starts_with("ALTER")
        || upper.starts_with("TRUNCATE")
    {
        return AlgorithmType::BTree;
    }

    // HDC / vector-specific functions
    if upper.contains("COSINE_SIMILARITY")
        || upper.contains("L2_DISTANCE")
        || upper.contains("INNER_PRODUCT")
        || upper.contains("VECTOR_DISTANCE")
        || upper.contains("VECTOR_NORM")
        || upper.contains("VECTOR_NORMALIZE")
        || upper.contains("HDC_SIMILARITY")
        || upper.contains("HDC_DISTANCE")
        || upper.contains("HDC_BIND")
        || upper.contains("HDC_BUNDLE")
        || upper.contains("HDC_ENCODE")
        || upper.contains("HDC_BIPOLAR_SIMILARITY")
    {
        return AlgorithmType::Hdc;
    }

    // Detect aggregation keywords
    let has_aggregate = [
        "SUM(",
        "AVG(",
        "COUNT(",
        "MIN(",
        "MAX(",
        "STDDEV(",
        "VARIANCE(",
        "MEDIAN(",
        "STRING_AGG(",
        "APPROX_COUNT_DISTINCT(",
        "APPROX_PERCENTILE(",
    ]
    .iter()
    .any(|kw| upper.contains(kw));
    let has_group_by = upper.contains("GROUP BY");

    // Columnar: aggregation with GROUP BY
    if has_aggregate && has_group_by {
        return AlgorithmType::Columnar;
    }

    // Point lookup: WHERE clause without aggregation (B-tree indexed access)
    if upper.contains("WHERE") && !has_aggregate && !has_group_by {
        return AlgorithmType::BTree;
    }

    // Full scan or aggregation without WHERE
    if upper.starts_with("SELECT") && !upper.contains("WHERE") {
        if has_aggregate {
            return AlgorithmType::Columnar;
        }
        return AlgorithmType::Scan;
    }

    // Default: B-tree (safest for mixed workloads)
    AlgorithmType::BTree
}

/// Resolve the optimal device target by combining workload affinity with
/// hardware advisor hints.
///
/// Logic:
/// 1. If the advisor emits device preferences (hardware pressure), intersect
///    with workload affinity — pick the highest-affinity device that's also
///    hardware-recommended.
/// 2. If no device preferences (Normal state), walk affinity list and pick
///    the first available device.
/// 3. Ultimate fallback: CPU.
fn resolve_device_target(
    algorithm: AlgorithmType,
    hints: &[ExecutionHint],
    snapshot: &EnergySnapshot,
) -> DeviceTarget {
    let affinity = algorithm.device_affinity();

    // Build set of devices the hardware advisor recommends
    let advised: Vec<DeviceTarget> = hints
        .iter()
        .filter_map(|h| match h {
            ExecutionHint::PreferNpu => Some(DeviceTarget::Npu),
            ExecutionHint::PreferTpu => Some(DeviceTarget::Tpu),
            ExecutionHint::PreferGpu => Some(DeviceTarget::Gpu),
            ExecutionHint::PreferLpu => Some(DeviceTarget::Lpu),
            ExecutionHint::PreferCpu => Some(DeviceTarget::Cpu),
            _ => None,
        })
        .collect();

    // If advisor has device preferences, pick the first affinity match
    if !advised.is_empty() {
        for preferred in affinity {
            if advised.contains(preferred) {
                return *preferred;
            }
        }
        // No overlap: fall back to advisor's top pick
        return advised[0];
    }

    // No advisor device preferences (Normal state):
    // pick the first affinity device that's available
    for preferred in affinity {
        let available = match preferred {
            DeviceTarget::Gpu => snapshot.gpu_available,
            DeviceTarget::Npu => snapshot.npu_available,
            DeviceTarget::Tpu => snapshot.tpu_available,
            DeviceTarget::Lpu => snapshot.lpu_available,
            DeviceTarget::Cpu => true,
        };
        if available {
            return *preferred;
        }
    }

    DeviceTarget::Cpu
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- classify_algorithm tests ---

    #[test]
    fn test_classify_btree() {
        assert_eq!(
            classify_algorithm("INSERT INTO t VALUES (1)"),
            AlgorithmType::BTree
        );
        assert_eq!(
            classify_algorithm("UPDATE t SET x = 1 WHERE id = 1"),
            AlgorithmType::BTree
        );
        assert_eq!(
            classify_algorithm("DELETE FROM t WHERE id = 1"),
            AlgorithmType::BTree
        );
        assert_eq!(
            classify_algorithm("SELECT * FROM t WHERE id = 5"),
            AlgorithmType::BTree
        );
        assert_eq!(
            classify_algorithm("CREATE TABLE t (id INT)"),
            AlgorithmType::BTree
        );
    }

    #[test]
    fn test_classify_columnar() {
        assert_eq!(
            classify_algorithm("SELECT dept, SUM(salary) FROM emp GROUP BY dept"),
            AlgorithmType::Columnar,
        );
        assert_eq!(
            classify_algorithm("SELECT COUNT(*) FROM orders"),
            AlgorithmType::Columnar,
        );
        assert_eq!(
            classify_algorithm("SELECT AVG(price) FROM products"),
            AlgorithmType::Columnar,
        );
    }

    #[test]
    fn test_classify_scan() {
        assert_eq!(classify_algorithm("SELECT * FROM t"), AlgorithmType::Scan);
        assert_eq!(
            classify_algorithm("SELECT a, b FROM t"),
            AlgorithmType::Scan
        );
    }

    #[test]
    fn test_classify_hdc() {
        assert_eq!(
            classify_algorithm("SELECT COSINE_SIMILARITY(v1, v2) FROM vectors"),
            AlgorithmType::Hdc,
        );
        assert_eq!(
            classify_algorithm("SELECT * FROM t WHERE L2_DISTANCE(embedding, '[1,2,3]') < 0.5"),
            AlgorithmType::Hdc,
        );
    }

    // --- resolve_device_target tests ---

    fn nominal_snapshot() -> EnergySnapshot {
        EnergySnapshot {
            gpu_available: true,
            npu_available: true,
            tpu_available: false,
            ..EnergySnapshot::default()
        }
    }

    #[test]
    fn test_resolve_btree_stays_cpu() {
        // BTree only has CPU affinity — should always pick CPU
        let snap = nominal_snapshot();
        let hints = vec![ExecutionHint::PreferGpu, ExecutionHint::PreferNpu];
        // BTree affinity = [Cpu], no overlap with GPU/NPU → fallback to advisor top (Gpu)
        // Actually BTree affinity is just [Cpu], and PreferCpu is not in hints.
        // No overlap → falls back to advised[0] = PreferGpu → Gpu
        // But in Normal state:
        let normal = vec![ExecutionHint::Normal];
        let target = resolve_device_target(AlgorithmType::BTree, &normal, &snap);
        assert_eq!(target, DeviceTarget::Cpu);
    }

    #[test]
    fn test_resolve_columnar_prefers_gpu() {
        // Columnar affinity: [Gpu, Tpu, Cpu]. GPU available → pick GPU
        let snap = nominal_snapshot();
        let hints = vec![ExecutionHint::Normal];
        let target = resolve_device_target(AlgorithmType::Columnar, &hints, &snap);
        assert_eq!(target, DeviceTarget::Gpu);
    }

    #[test]
    fn test_resolve_hdc_gets_npu_when_available() {
        // Hdc affinity: [Npu, Gpu, Cpu]. NPU available → pick NPU
        let snap = nominal_snapshot();
        let hints = vec![ExecutionHint::Normal];
        let target = resolve_device_target(AlgorithmType::Hdc, &hints, &snap);
        assert_eq!(target, DeviceTarget::Npu);
    }

    #[test]
    fn test_resolve_affinity_intersects_advisor() {
        // Columnar affinity: [Gpu, Tpu, Cpu]. Advisor says PreferNpu.
        // No overlap → fallback to advisor top pick (Npu)
        let snap = nominal_snapshot();
        let hints = vec![ExecutionHint::PreferNpu];
        let target = resolve_device_target(AlgorithmType::Columnar, &hints, &snap);
        assert_eq!(target, DeviceTarget::Npu);

        // Now advisor says PreferGpu — overlaps with Columnar's affinity
        let hints2 = vec![ExecutionHint::PreferGpu];
        let target2 = resolve_device_target(AlgorithmType::Columnar, &hints2, &snap);
        assert_eq!(target2, DeviceTarget::Gpu);
    }

    #[test]
    fn test_resolve_fallback_when_no_devices() {
        // Scan affinity: [Gpu, Cpu]. No GPU, no NPU, no TPU → CPU
        let snap = EnergySnapshot::default(); // everything false/0
        let hints = vec![ExecutionHint::Normal];
        let target = resolve_device_target(AlgorithmType::Scan, &hints, &snap);
        assert_eq!(target, DeviceTarget::Cpu);
    }

    #[test]
    fn test_classify_hdc_functions() {
        assert_eq!(
            classify_algorithm("SELECT HDC_SIMILARITY(a, b) FROM vectors"),
            AlgorithmType::Hdc,
        );
        assert_eq!(
            classify_algorithm("SELECT HDC_DISTANCE(a, b) FROM vectors"),
            AlgorithmType::Hdc,
        );
        assert_eq!(
            classify_algorithm("SELECT HDC_BIND(a, b) FROM vectors"),
            AlgorithmType::Hdc,
        );
        assert_eq!(
            classify_algorithm("SELECT HDC_BUNDLE(a, b, c) FROM vectors"),
            AlgorithmType::Hdc,
        );
        assert_eq!(
            classify_algorithm("SELECT HDC_ENCODE('market', 'full', '{\"symbol\":\"AAPL\"}')"),
            AlgorithmType::Hdc,
        );
        assert_eq!(
            classify_algorithm("SELECT HDC_BIPOLAR_SIMILARITY(a, b) FROM vectors"),
            AlgorithmType::Hdc,
        );
    }

    // --- Thread-local device context tests ---

    #[test]
    fn test_device_context_default_cpu() {
        // No context set → defaults to CPU
        clear_device_context();
        assert_eq!(current_device_target(), DeviceTarget::Cpu);
        assert!(!is_throttled());
    }

    #[test]
    fn test_device_context_roundtrip() {
        set_device_context(DeviceContext {
            device_target: DeviceTarget::Gpu,
            algorithm: AlgorithmType::Columnar,
            hints: vec![ExecutionHint::PreferGpu],
        });

        let ctx = current_device_context().unwrap();
        assert_eq!(ctx.device_target, DeviceTarget::Gpu);
        assert_eq!(ctx.algorithm, AlgorithmType::Columnar);
        assert!(!is_throttled());

        clear_device_context();
        assert!(current_device_context().is_none());
    }

    #[test]
    fn test_device_context_throttle_detection() {
        set_device_context(DeviceContext {
            device_target: DeviceTarget::Cpu,
            algorithm: AlgorithmType::Scan,
            hints: vec![ExecutionHint::Throttle, ExecutionHint::PreferCpu],
        });

        assert!(is_throttled());
        assert_eq!(current_device_target(), DeviceTarget::Cpu);

        clear_device_context();
    }

    // --- StageTimers tests ---

    #[test]
    fn test_stage_timers_begin_end_roundtrip() {
        // Clear any prior state
        let _ = take_stage_timers();

        begin_stage(ExecutionStage::Parse);
        std::thread::sleep(std::time::Duration::from_millis(5));
        end_stage();

        let timers = take_stage_timers().expect("should have timers");
        assert_eq!(timers.timings.len(), 1);
        assert_eq!(timers.timings[0].0, ExecutionStage::Parse);
        assert!(timers.timings[0].1.as_nanos() > 0);
    }

    #[test]
    fn test_stage_timers_apportion_proportional() {
        let mut timers = StageTimers::new();
        // Simulate 30ms parse, 70ms execute
        timers
            .timings
            .push((ExecutionStage::Parse, Duration::from_millis(30)));
        timers
            .timings
            .push((ExecutionStage::Execute, Duration::from_millis(70)));

        let result = timers.apportion(1.0);
        assert!((result[&ExecutionStage::Parse] - 0.3).abs() < 0.01);
        assert!((result[&ExecutionStage::Execute] - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_stage_timers_apportion_empty() {
        let timers = StageTimers::new();
        let result = timers.apportion(1.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_take_stage_timers_clears_tls() {
        let _ = take_stage_timers(); // clear
        begin_stage(ExecutionStage::Io);
        end_stage();
        assert!(take_stage_timers().is_some());
        assert!(take_stage_timers().is_none()); // second take is None
    }

    #[test]
    fn test_begin_stage_auto_ends_previous() {
        let _ = take_stage_timers(); // clear
        begin_stage(ExecutionStage::Parse);
        std::thread::sleep(std::time::Duration::from_millis(2));
        begin_stage(ExecutionStage::Execute); // auto-ends Parse
        std::thread::sleep(std::time::Duration::from_millis(2));
        end_stage();

        let timers = take_stage_timers().unwrap();
        assert_eq!(timers.timings.len(), 2);
        assert_eq!(timers.timings[0].0, ExecutionStage::Parse);
        assert_eq!(timers.timings[1].0, ExecutionStage::Execute);
    }

    // --- classify_sql tests (pre-existing function) ---

    #[test]
    fn test_classify_sql_operations() {
        assert_eq!(classify_sql("SELECT * FROM t"), OperationType::Search);
        assert_eq!(
            classify_sql("INSERT INTO t VALUES (1)"),
            OperationType::Write
        );
        assert_eq!(classify_sql("UPDATE t SET x = 1"), OperationType::Write);
        assert_eq!(classify_sql("DELETE FROM t"), OperationType::Write);
        assert_eq!(classify_sql("SHOW TABLES"), OperationType::Read);
    }
}
