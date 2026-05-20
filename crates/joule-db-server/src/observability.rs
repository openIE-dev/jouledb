//! Full Observability Stack for JouleDB
//!
//! This module provides comprehensive observability features for production deployment:
//!
//! - **OpenTelemetry Integration**: Standard observability protocol for traces, metrics, and logs
//! - **Prometheus Metrics**: Enhanced metrics with cardinality management
//! - **Structured Logging**: JSON-formatted logs with trace correlation
//! - **Health Checks**: Deep dependency probing with SLA tracking
//! - **Alerting**: Alert rules and notification integration
//! - **Dashboards**: Pre-configured Grafana dashboard definitions
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    Application Layer                             │
//! │  (Instrumented with spans, metrics, and structured logs)        │
//! └──────────────────────┬──────────────────────────────────────────┘
//!                        │
//! ┌──────────────────────▼──────────────────────────────────────────┐
//! │                 Observability Pipeline                           │
//! │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────────┐ │
//! │  │ Traces  │  │ Metrics │  │  Logs   │  │   Health Checks     │ │
//! │  └────┬────┘  └────┬────┘  └────┬────┘  └─────────┬───────────┘ │
//! │       │            │            │                 │             │
//! │       ▼            ▼            ▼                 ▼             │
//! │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────────┐ │
//! │  │ OTel    │  │ Prom    │  │ Loki/   │  │ Kubernetes Probes   │ │
//! │  │ Export  │  │ Export  │  │ Elastic │  │ & External Monitors │ │
//! │  └─────────┘  └─────────┘  └─────────┘  └─────────────────────┘ │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

// ============================================================================
// Configuration
// ============================================================================

/// Observability configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservabilityConfig {
    /// Service name for identification
    pub service_name: String,
    /// Service version
    pub service_version: String,
    /// Environment (production, staging, development)
    pub environment: String,
    /// Instance ID
    pub instance_id: String,

    /// Enable tracing
    pub tracing_enabled: bool,
    /// Trace sampling rate (0.0 to 1.0)
    pub trace_sample_rate: f64,
    /// OTLP trace endpoint
    pub otlp_trace_endpoint: Option<String>,

    /// Enable metrics
    pub metrics_enabled: bool,
    /// Prometheus endpoint path
    pub prometheus_path: String,
    /// Metrics push gateway URL
    pub pushgateway_url: Option<String>,

    /// Enable structured logging
    pub logging_enabled: bool,
    /// Log level
    pub log_level: String,
    /// Log format (json, text)
    pub log_format: String,

    /// Health check interval
    pub health_check_interval: Duration,
    /// Health check timeout
    pub health_check_timeout: Duration,
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            service_name: "joule-db".to_string(),
            service_version: env!("CARGO_PKG_VERSION").to_string(),
            environment: "production".to_string(),
            instance_id: generate_instance_id(),

            tracing_enabled: true,
            trace_sample_rate: 0.1, // 10% sampling by default
            otlp_trace_endpoint: None,

            metrics_enabled: true,
            prometheus_path: "/metrics".to_string(),
            pushgateway_url: None,

            logging_enabled: true,
            log_level: "info".to_string(),
            log_format: "json".to_string(),

            health_check_interval: Duration::from_secs(30),
            health_check_timeout: Duration::from_secs(5),
        }
    }
}

// ============================================================================
// OpenTelemetry Integration
// ============================================================================

/// Resource attributes for OpenTelemetry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtelResource {
    pub service_name: String,
    pub service_version: String,
    pub service_instance_id: String,
    pub deployment_environment: String,
    pub host_name: String,
    pub process_pid: u32,
    pub telemetry_sdk_name: String,
    pub telemetry_sdk_version: String,
    pub telemetry_sdk_language: String,
}

impl OtelResource {
    /// Create from config
    pub fn from_config(config: &ObservabilityConfig) -> Self {
        Self {
            service_name: config.service_name.clone(),
            service_version: config.service_version.clone(),
            service_instance_id: config.instance_id.clone(),
            deployment_environment: config.environment.clone(),
            host_name: hostname().unwrap_or_else(|| "unknown".to_string()),
            process_pid: std::process::id(),
            telemetry_sdk_name: "joule-db".to_string(),
            telemetry_sdk_version: env!("CARGO_PKG_VERSION").to_string(),
            telemetry_sdk_language: "rust".to_string(),
        }
    }

    /// Convert to OTLP attributes format
    pub fn to_attributes(&self) -> Vec<OtelAttribute> {
        vec![
            OtelAttribute::new("service.name", &self.service_name),
            OtelAttribute::new("service.version", &self.service_version),
            OtelAttribute::new("service.instance.id", &self.service_instance_id),
            OtelAttribute::new("deployment.environment", &self.deployment_environment),
            OtelAttribute::new("host.name", &self.host_name),
            OtelAttribute::new_int("process.pid", self.process_pid as i64),
            OtelAttribute::new("telemetry.sdk.name", &self.telemetry_sdk_name),
            OtelAttribute::new("telemetry.sdk.version", &self.telemetry_sdk_version),
            OtelAttribute::new("telemetry.sdk.language", &self.telemetry_sdk_language),
        ]
    }
}

/// OpenTelemetry attribute
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtelAttribute {
    pub key: String,
    pub value: OtelAttributeValue,
}

impl OtelAttribute {
    /// Create string attribute
    pub fn new(key: &str, value: &str) -> Self {
        Self {
            key: key.to_string(),
            value: OtelAttributeValue::String(value.to_string()),
        }
    }

    /// Create integer attribute
    pub fn new_int(key: &str, value: i64) -> Self {
        Self {
            key: key.to_string(),
            value: OtelAttributeValue::Int(value),
        }
    }

    /// Create float attribute
    pub fn new_float(key: &str, value: f64) -> Self {
        Self {
            key: key.to_string(),
            value: OtelAttributeValue::Double(value),
        }
    }

    /// Create boolean attribute
    pub fn new_bool(key: &str, value: bool) -> Self {
        Self {
            key: key.to_string(),
            value: OtelAttributeValue::Bool(value),
        }
    }
}

/// OpenTelemetry attribute value
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OtelAttributeValue {
    String(String),
    Int(i64),
    Double(f64),
    Bool(bool),
    StringArray(Vec<String>),
    IntArray(Vec<i64>),
}

/// OpenTelemetry span representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtelSpan {
    /// Trace ID (16 bytes hex-encoded)
    pub trace_id: String,
    /// Span ID (8 bytes hex-encoded)
    pub span_id: String,
    /// Parent span ID
    pub parent_span_id: Option<String>,
    /// Span name
    pub name: String,
    /// Span kind (SERVER, CLIENT, PRODUCER, CONSUMER, INTERNAL)
    pub kind: SpanKind,
    /// Start time (nanoseconds since epoch)
    pub start_time_unix_nano: u64,
    /// End time (nanoseconds since epoch)
    pub end_time_unix_nano: u64,
    /// Attributes
    pub attributes: Vec<OtelAttribute>,
    /// Events
    pub events: Vec<SpanEvent>,
    /// Status
    pub status: SpanStatus,
    /// Links to other spans
    pub links: Vec<SpanLink>,
}

/// Span kind
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpanKind {
    /// Internal operation
    Internal,
    /// Server-side of a synchronous RPC
    Server,
    /// Client-side of a synchronous RPC
    Client,
    /// Producer of an async message
    Producer,
    /// Consumer of an async message
    Consumer,
}

impl Default for SpanKind {
    fn default() -> Self {
        Self::Internal
    }
}

/// Span event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanEvent {
    pub name: String,
    pub time_unix_nano: u64,
    pub attributes: Vec<OtelAttribute>,
}

/// Span status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanStatus {
    pub code: StatusCode,
    pub message: Option<String>,
}

impl Default for SpanStatus {
    fn default() -> Self {
        Self {
            code: StatusCode::Unset,
            message: None,
        }
    }
}

/// Status code
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StatusCode {
    Unset,
    Ok,
    Error,
}

/// Span link
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanLink {
    pub trace_id: String,
    pub span_id: String,
    pub attributes: Vec<OtelAttribute>,
}

// ============================================================================
// Enhanced Prometheus Metrics
// ============================================================================

/// Database-specific metrics with semantic naming
pub struct DatabaseMetricsEnhanced {
    /// Query metrics
    pub queries: QueryMetrics,
    /// Connection metrics
    pub connections: ConnectionMetrics,
    /// Storage metrics
    pub storage: StorageMetrics,
    /// Replication metrics
    pub replication: ReplicationMetricsEnhanced,
    /// Transaction metrics
    pub transactions: TransactionMetrics,
    /// Cache metrics
    pub cache: CacheMetrics,
}

impl DatabaseMetricsEnhanced {
    /// Create new database metrics
    pub fn new() -> Self {
        Self {
            queries: QueryMetrics::new(),
            connections: ConnectionMetrics::new(),
            storage: StorageMetrics::new(),
            replication: ReplicationMetricsEnhanced::new(),
            transactions: TransactionMetrics::new(),
            cache: CacheMetrics::new(),
        }
    }

    /// Export all metrics in Prometheus format
    pub fn export_prometheus(&self) -> String {
        let mut output = String::new();

        output.push_str(&self.queries.export());
        output.push_str(&self.connections.export());
        output.push_str(&self.storage.export());
        output.push_str(&self.replication.export());
        output.push_str(&self.transactions.export());
        output.push_str(&self.cache.export());

        output
    }
}

impl Default for DatabaseMetricsEnhanced {
    fn default() -> Self {
        Self::new()
    }
}

/// Query-related metrics
pub struct QueryMetrics {
    /// Total queries executed
    total: AtomicU64,
    /// Queries by type (select, insert, update, delete)
    by_type: Arc<RwLock<HashMap<String, AtomicU64>>>,
    /// Query latency histogram
    latency_buckets: Arc<RwLock<Vec<(f64, AtomicU64)>>>,
    /// Slow queries count
    slow_queries: AtomicU64,
    /// Query errors
    errors: AtomicU64,
}

impl QueryMetrics {
    fn new() -> Self {
        let buckets = vec![
            0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
        ];

        Self {
            total: AtomicU64::new(0),
            by_type: Arc::new(RwLock::new(HashMap::new())),
            latency_buckets: Arc::new(RwLock::new(
                buckets
                    .into_iter()
                    .map(|b| (b, AtomicU64::new(0)))
                    .collect(),
            )),
            slow_queries: AtomicU64::new(0),
            errors: AtomicU64::new(0),
        }
    }

    /// Record a query execution
    pub fn record(&self, query_type: &str, duration: Duration, success: bool) {
        self.total.fetch_add(1, Ordering::Relaxed);

        // Update by type
        {
            let mut by_type = crate::lock_util::write_lock(&self.by_type);
            by_type
                .entry(query_type.to_string())
                .or_insert_with(|| AtomicU64::new(0))
                .fetch_add(1, Ordering::Relaxed);
        }

        // Update latency histogram
        let secs = duration.as_secs_f64();
        {
            let buckets = crate::lock_util::read_lock(&self.latency_buckets);
            for (threshold, count) in buckets.iter() {
                if secs <= *threshold {
                    count.fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        // Check for slow query (> 1s)
        if secs > 1.0 {
            self.slow_queries.fetch_add(1, Ordering::Relaxed);
        }

        if !success {
            self.errors.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn export(&self) -> String {
        let mut output = String::new();

        output.push_str("# HELP joule_db_queries_total Total number of queries executed\n");
        output.push_str("# TYPE joule_db_queries_total counter\n");
        output.push_str(&format!(
            "joule_db_queries_total {}\n",
            self.total.load(Ordering::Relaxed)
        ));

        output.push_str("# HELP joule_db_queries_by_type_total Queries by type\n");
        output.push_str("# TYPE joule_db_queries_by_type_total counter\n");
        {
            let by_type = crate::lock_util::read_lock(&self.by_type);
            for (qtype, count) in by_type.iter() {
                output.push_str(&format!(
                    "joule_db_queries_by_type_total{{type=\"{}\"}} {}\n",
                    qtype,
                    count.load(Ordering::Relaxed)
                ));
            }
        }

        output.push_str("# HELP joule_db_query_duration_seconds Query latency histogram\n");
        output.push_str("# TYPE joule_db_query_duration_seconds histogram\n");
        {
            let buckets = crate::lock_util::read_lock(&self.latency_buckets);
            for (threshold, count) in buckets.iter() {
                output.push_str(&format!(
                    "joule_db_query_duration_seconds_bucket{{le=\"{}\"}} {}\n",
                    threshold,
                    count.load(Ordering::Relaxed)
                ));
            }
        }

        output.push_str("# HELP joule_db_slow_queries_total Slow queries count\n");
        output.push_str("# TYPE joule_db_slow_queries_total counter\n");
        output.push_str(&format!(
            "joule_db_slow_queries_total {}\n",
            self.slow_queries.load(Ordering::Relaxed)
        ));

        output.push_str("# HELP joule_db_query_errors_total Query errors\n");
        output.push_str("# TYPE joule_db_query_errors_total counter\n");
        output.push_str(&format!(
            "joule_db_query_errors_total {}\n",
            self.errors.load(Ordering::Relaxed)
        ));

        output
    }
}

/// Connection-related metrics
pub struct ConnectionMetrics {
    /// Current active connections
    active: AtomicU64,
    /// Total connections established
    total: AtomicU64,
    /// Connection errors
    errors: AtomicU64,
    /// Max connections
    max: AtomicU64,
    /// Connection wait time histogram
    wait_time_sum: AtomicU64,
}

impl ConnectionMetrics {
    fn new() -> Self {
        Self {
            active: AtomicU64::new(0),
            total: AtomicU64::new(0),
            errors: AtomicU64::new(0),
            max: AtomicU64::new(100),
            wait_time_sum: AtomicU64::new(0),
        }
    }

    /// Record connection established
    pub fn connection_opened(&self) {
        self.active.fetch_add(1, Ordering::Relaxed);
        self.total.fetch_add(1, Ordering::Relaxed);
    }

    /// Record connection closed
    pub fn connection_closed(&self) {
        self.active.fetch_sub(1, Ordering::Relaxed);
    }

    /// Record connection error
    pub fn connection_error(&self) {
        self.errors.fetch_add(1, Ordering::Relaxed);
    }

    /// Set max connections
    pub fn set_max(&self, max: u64) {
        self.max.store(max, Ordering::Relaxed);
    }

    fn export(&self) -> String {
        let mut output = String::new();

        output.push_str("# HELP joule_db_connections_active Current active connections\n");
        output.push_str("# TYPE joule_db_connections_active gauge\n");
        output.push_str(&format!(
            "joule_db_connections_active {}\n",
            self.active.load(Ordering::Relaxed)
        ));

        output.push_str("# HELP joule_db_connections_total Total connections established\n");
        output.push_str("# TYPE joule_db_connections_total counter\n");
        output.push_str(&format!(
            "joule_db_connections_total {}\n",
            self.total.load(Ordering::Relaxed)
        ));

        output.push_str("# HELP joule_db_connections_max Maximum connections\n");
        output.push_str("# TYPE joule_db_connections_max gauge\n");
        output.push_str(&format!(
            "joule_db_connections_max {}\n",
            self.max.load(Ordering::Relaxed)
        ));

        output.push_str("# HELP joule_db_connection_errors_total Connection errors\n");
        output.push_str("# TYPE joule_db_connection_errors_total counter\n");
        output.push_str(&format!(
            "joule_db_connection_errors_total {}\n",
            self.errors.load(Ordering::Relaxed)
        ));

        output
    }
}

/// Storage-related metrics
pub struct StorageMetrics {
    /// Total bytes stored
    bytes_total: AtomicU64,
    /// Data bytes
    bytes_data: AtomicU64,
    /// Index bytes
    bytes_index: AtomicU64,
    /// WAL bytes
    bytes_wal: AtomicU64,
    /// Read IOPS
    read_iops: AtomicU64,
    /// Write IOPS
    write_iops: AtomicU64,
    /// Page reads
    page_reads: AtomicU64,
    /// Page writes
    page_writes: AtomicU64,
    /// Compaction count
    compactions: AtomicU64,
}

impl StorageMetrics {
    fn new() -> Self {
        Self {
            bytes_total: AtomicU64::new(0),
            bytes_data: AtomicU64::new(0),
            bytes_index: AtomicU64::new(0),
            bytes_wal: AtomicU64::new(0),
            read_iops: AtomicU64::new(0),
            write_iops: AtomicU64::new(0),
            page_reads: AtomicU64::new(0),
            page_writes: AtomicU64::new(0),
            compactions: AtomicU64::new(0),
        }
    }

    /// Update storage sizes
    pub fn update_sizes(&self, data: u64, index: u64, wal: u64) {
        self.bytes_data.store(data, Ordering::Relaxed);
        self.bytes_index.store(index, Ordering::Relaxed);
        self.bytes_wal.store(wal, Ordering::Relaxed);
        self.bytes_total
            .store(data + index + wal, Ordering::Relaxed);
    }

    /// Record page read
    pub fn page_read(&self) {
        self.page_reads.fetch_add(1, Ordering::Relaxed);
        self.read_iops.fetch_add(1, Ordering::Relaxed);
    }

    /// Record page write
    pub fn page_write(&self) {
        self.page_writes.fetch_add(1, Ordering::Relaxed);
        self.write_iops.fetch_add(1, Ordering::Relaxed);
    }

    /// Record compaction
    pub fn compaction(&self) {
        self.compactions.fetch_add(1, Ordering::Relaxed);
    }

    fn export(&self) -> String {
        let mut output = String::new();

        output.push_str("# HELP joule_db_storage_bytes_total Total storage bytes\n");
        output.push_str("# TYPE joule_db_storage_bytes_total gauge\n");
        output.push_str(&format!(
            "joule_db_storage_bytes_total {}\n",
            self.bytes_total.load(Ordering::Relaxed)
        ));

        output.push_str("# HELP joule_db_storage_data_bytes Data storage bytes\n");
        output.push_str("# TYPE joule_db_storage_data_bytes gauge\n");
        output.push_str(&format!(
            "joule_db_storage_data_bytes {}\n",
            self.bytes_data.load(Ordering::Relaxed)
        ));

        output.push_str("# HELP joule_db_storage_index_bytes Index storage bytes\n");
        output.push_str("# TYPE joule_db_storage_index_bytes gauge\n");
        output.push_str(&format!(
            "joule_db_storage_index_bytes {}\n",
            self.bytes_index.load(Ordering::Relaxed)
        ));

        output.push_str("# HELP joule_db_storage_wal_bytes WAL storage bytes\n");
        output.push_str("# TYPE joule_db_storage_wal_bytes gauge\n");
        output.push_str(&format!(
            "joule_db_storage_wal_bytes {}\n",
            self.bytes_wal.load(Ordering::Relaxed)
        ));

        output.push_str("# HELP joule_db_page_reads_total Page reads\n");
        output.push_str("# TYPE joule_db_page_reads_total counter\n");
        output.push_str(&format!(
            "joule_db_page_reads_total {}\n",
            self.page_reads.load(Ordering::Relaxed)
        ));

        output.push_str("# HELP joule_db_page_writes_total Page writes\n");
        output.push_str("# TYPE joule_db_page_writes_total counter\n");
        output.push_str(&format!(
            "joule_db_page_writes_total {}\n",
            self.page_writes.load(Ordering::Relaxed)
        ));

        output.push_str("# HELP joule_db_compactions_total Compactions\n");
        output.push_str("# TYPE joule_db_compactions_total counter\n");
        output.push_str(&format!(
            "joule_db_compactions_total {}\n",
            self.compactions.load(Ordering::Relaxed)
        ));

        output
    }
}

/// Replication-related metrics
pub struct ReplicationMetricsEnhanced {
    /// Replication lag in milliseconds
    lag_ms: AtomicU64,
    /// Bytes replicated
    bytes_replicated: AtomicU64,
    /// Replication errors
    errors: AtomicU64,
    /// Is leader
    is_leader: AtomicU64,
    /// Follower count
    follower_count: AtomicU64,
}

impl ReplicationMetricsEnhanced {
    fn new() -> Self {
        Self {
            lag_ms: AtomicU64::new(0),
            bytes_replicated: AtomicU64::new(0),
            errors: AtomicU64::new(0),
            is_leader: AtomicU64::new(0),
            follower_count: AtomicU64::new(0),
        }
    }

    /// Update replication lag
    pub fn set_lag(&self, ms: u64) {
        self.lag_ms.store(ms, Ordering::Relaxed);
    }

    /// Record bytes replicated
    pub fn bytes_replicated(&self, bytes: u64) {
        self.bytes_replicated.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record replication error
    pub fn error(&self) {
        self.errors.fetch_add(1, Ordering::Relaxed);
    }

    /// Set leader status
    pub fn set_leader(&self, is_leader: bool) {
        self.is_leader
            .store(if is_leader { 1 } else { 0 }, Ordering::Relaxed);
    }

    /// Set follower count
    pub fn set_follower_count(&self, count: u64) {
        self.follower_count.store(count, Ordering::Relaxed);
    }

    fn export(&self) -> String {
        let mut output = String::new();

        output.push_str("# HELP joule_db_replication_lag_ms Replication lag in milliseconds\n");
        output.push_str("# TYPE joule_db_replication_lag_ms gauge\n");
        output.push_str(&format!(
            "joule_db_replication_lag_ms {}\n",
            self.lag_ms.load(Ordering::Relaxed)
        ));

        output.push_str("# HELP joule_db_replication_bytes_total Bytes replicated\n");
        output.push_str("# TYPE joule_db_replication_bytes_total counter\n");
        output.push_str(&format!(
            "joule_db_replication_bytes_total {}\n",
            self.bytes_replicated.load(Ordering::Relaxed)
        ));

        output.push_str("# HELP joule_db_is_leader Whether this node is the leader\n");
        output.push_str("# TYPE joule_db_is_leader gauge\n");
        output.push_str(&format!(
            "joule_db_is_leader {}\n",
            self.is_leader.load(Ordering::Relaxed)
        ));

        output.push_str("# HELP joule_db_follower_count Number of followers\n");
        output.push_str("# TYPE joule_db_follower_count gauge\n");
        output.push_str(&format!(
            "joule_db_follower_count {}\n",
            self.follower_count.load(Ordering::Relaxed)
        ));

        output
    }
}

/// Transaction-related metrics
pub struct TransactionMetrics {
    /// Active transactions
    active: AtomicU64,
    /// Total commits
    commits: AtomicU64,
    /// Total rollbacks
    rollbacks: AtomicU64,
    /// Deadlocks
    deadlocks: AtomicU64,
    /// Lock waits
    lock_waits: AtomicU64,
}

impl TransactionMetrics {
    fn new() -> Self {
        Self {
            active: AtomicU64::new(0),
            commits: AtomicU64::new(0),
            rollbacks: AtomicU64::new(0),
            deadlocks: AtomicU64::new(0),
            lock_waits: AtomicU64::new(0),
        }
    }

    /// Transaction started
    pub fn begin(&self) {
        self.active.fetch_add(1, Ordering::Relaxed);
    }

    /// Transaction committed
    pub fn commit(&self) {
        self.active.fetch_sub(1, Ordering::Relaxed);
        self.commits.fetch_add(1, Ordering::Relaxed);
    }

    /// Transaction rolled back
    pub fn rollback(&self) {
        self.active.fetch_sub(1, Ordering::Relaxed);
        self.rollbacks.fetch_add(1, Ordering::Relaxed);
    }

    /// Deadlock detected
    pub fn deadlock(&self) {
        self.deadlocks.fetch_add(1, Ordering::Relaxed);
    }

    /// Lock wait recorded
    pub fn lock_wait(&self) {
        self.lock_waits.fetch_add(1, Ordering::Relaxed);
    }

    fn export(&self) -> String {
        let mut output = String::new();

        output.push_str("# HELP joule_db_transactions_active Active transactions\n");
        output.push_str("# TYPE joule_db_transactions_active gauge\n");
        output.push_str(&format!(
            "joule_db_transactions_active {}\n",
            self.active.load(Ordering::Relaxed)
        ));

        output.push_str("# HELP joule_db_transactions_commits_total Transaction commits\n");
        output.push_str("# TYPE joule_db_transactions_commits_total counter\n");
        output.push_str(&format!(
            "joule_db_transactions_commits_total {}\n",
            self.commits.load(Ordering::Relaxed)
        ));

        output.push_str("# HELP joule_db_transactions_rollbacks_total Transaction rollbacks\n");
        output.push_str("# TYPE joule_db_transactions_rollbacks_total counter\n");
        output.push_str(&format!(
            "joule_db_transactions_rollbacks_total {}\n",
            self.rollbacks.load(Ordering::Relaxed)
        ));

        output.push_str("# HELP joule_db_deadlocks_total Deadlocks detected\n");
        output.push_str("# TYPE joule_db_deadlocks_total counter\n");
        output.push_str(&format!(
            "joule_db_deadlocks_total {}\n",
            self.deadlocks.load(Ordering::Relaxed)
        ));

        output
    }
}

/// Cache-related metrics
pub struct CacheMetrics {
    /// Cache hits
    hits: AtomicU64,
    /// Cache misses
    misses: AtomicU64,
    /// Cache evictions
    evictions: AtomicU64,
    /// Cache size in bytes
    size_bytes: AtomicU64,
    /// Cache entries
    entries: AtomicU64,
}

impl CacheMetrics {
    fn new() -> Self {
        Self {
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            size_bytes: AtomicU64::new(0),
            entries: AtomicU64::new(0),
        }
    }

    /// Cache hit
    pub fn hit(&self) {
        self.hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Cache miss
    pub fn miss(&self) {
        self.misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Cache eviction
    pub fn eviction(&self) {
        self.evictions.fetch_add(1, Ordering::Relaxed);
    }

    /// Update cache size
    pub fn set_size(&self, bytes: u64, entries: u64) {
        self.size_bytes.store(bytes, Ordering::Relaxed);
        self.entries.store(entries, Ordering::Relaxed);
    }

    /// Get cache hit ratio
    pub fn hit_ratio(&self) -> f64 {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }

    fn export(&self) -> String {
        let mut output = String::new();

        output.push_str("# HELP joule_db_cache_hits_total Cache hits\n");
        output.push_str("# TYPE joule_db_cache_hits_total counter\n");
        output.push_str(&format!(
            "joule_db_cache_hits_total {}\n",
            self.hits.load(Ordering::Relaxed)
        ));

        output.push_str("# HELP joule_db_cache_misses_total Cache misses\n");
        output.push_str("# TYPE joule_db_cache_misses_total counter\n");
        output.push_str(&format!(
            "joule_db_cache_misses_total {}\n",
            self.misses.load(Ordering::Relaxed)
        ));

        output.push_str("# HELP joule_db_cache_hit_ratio Cache hit ratio\n");
        output.push_str("# TYPE joule_db_cache_hit_ratio gauge\n");
        output.push_str(&format!(
            "joule_db_cache_hit_ratio {:.4}\n",
            self.hit_ratio()
        ));

        output.push_str("# HELP joule_db_cache_size_bytes Cache size in bytes\n");
        output.push_str("# TYPE joule_db_cache_size_bytes gauge\n");
        output.push_str(&format!(
            "joule_db_cache_size_bytes {}\n",
            self.size_bytes.load(Ordering::Relaxed)
        ));

        output.push_str("# HELP joule_db_cache_entries Cache entries count\n");
        output.push_str("# TYPE joule_db_cache_entries gauge\n");
        output.push_str(&format!(
            "joule_db_cache_entries {}\n",
            self.entries.load(Ordering::Relaxed)
        ));

        output
    }
}

// ============================================================================
// Deep Health Checks
// ============================================================================

/// Health check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckResult {
    /// Check name
    pub name: String,
    /// Status
    pub status: HealthStatus,
    /// Duration of check
    pub duration_ms: u64,
    /// Additional details
    pub details: Option<String>,
    /// Timestamp
    pub timestamp: u64,
}

/// Health status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// Healthy
    Healthy,
    /// Degraded but functional
    Degraded,
    /// Unhealthy
    Unhealthy,
}

/// Health check definition
pub trait HealthCheck: Send + Sync {
    /// Check name
    fn name(&self) -> &str;

    /// Perform the health check
    fn check(&self) -> HealthCheckResult;

    /// Is this check critical for readiness?
    fn is_critical(&self) -> bool {
        true
    }
}

/// Storage health check
pub struct StorageHealthCheck {
    name: String,
    path: std::path::PathBuf,
}

impl StorageHealthCheck {
    /// Create storage health check
    pub fn new(path: impl Into<std::path::PathBuf>) -> Self {
        Self {
            name: "storage".to_string(),
            path: path.into(),
        }
    }
}

impl HealthCheck for StorageHealthCheck {
    fn name(&self) -> &str {
        &self.name
    }

    fn check(&self) -> HealthCheckResult {
        let start = Instant::now();
        let timestamp = current_timestamp();

        // Check if path exists and is writable
        let status = if self.path.exists() {
            // Try to create a temp file
            let test_path = self.path.join(".health_check");
            match std::fs::write(&test_path, "health") {
                Ok(_) => {
                    let _ = std::fs::remove_file(&test_path);
                    HealthStatus::Healthy
                }
                Err(_) => HealthStatus::Unhealthy,
            }
        } else {
            HealthStatus::Unhealthy
        };

        // Check disk space
        let details = match get_disk_space(&self.path) {
            Some((free, total)) => {
                let usage = (total - free) as f64 / total as f64;
                if usage > 0.95 {
                    Some(format!("Disk space critical: {:.1}% used", usage * 100.0))
                } else if usage > 0.85 {
                    Some(format!("Disk space warning: {:.1}% used", usage * 100.0))
                } else {
                    Some(format!("Disk space OK: {:.1}% used", usage * 100.0))
                }
            }
            None => None,
        };

        HealthCheckResult {
            name: self.name.clone(),
            status,
            duration_ms: start.elapsed().as_millis() as u64,
            details,
            timestamp,
        }
    }
}

/// Memory health check
pub struct MemoryHealthCheck {
    name: String,
    max_usage_ratio: f64,
}

impl MemoryHealthCheck {
    /// Create memory health check
    pub fn new(max_usage_ratio: f64) -> Self {
        Self {
            name: "memory".to_string(),
            max_usage_ratio,
        }
    }
}

impl HealthCheck for MemoryHealthCheck {
    fn name(&self) -> &str {
        &self.name
    }

    fn check(&self) -> HealthCheckResult {
        let start = Instant::now();
        let timestamp = current_timestamp();

        // Get current memory usage (simplified - in production would use /proc/meminfo or system calls)
        let (status, details) = match get_memory_usage() {
            Some((used, total)) => {
                let ratio = used as f64 / total as f64;
                if ratio > self.max_usage_ratio {
                    (
                        HealthStatus::Unhealthy,
                        Some(format!(
                            "Memory usage critical: {:.1}% ({} MB / {} MB)",
                            ratio * 100.0,
                            used / 1024 / 1024,
                            total / 1024 / 1024
                        )),
                    )
                } else if ratio > self.max_usage_ratio * 0.9 {
                    (
                        HealthStatus::Degraded,
                        Some(format!("Memory usage high: {:.1}%", ratio * 100.0)),
                    )
                } else {
                    (
                        HealthStatus::Healthy,
                        Some(format!("Memory usage OK: {:.1}%", ratio * 100.0)),
                    )
                }
            }
            None => (
                HealthStatus::Healthy,
                Some("Memory info unavailable".to_string()),
            ),
        };

        HealthCheckResult {
            name: self.name.clone(),
            status,
            duration_ms: start.elapsed().as_millis() as u64,
            details,
            timestamp,
        }
    }
}

/// Replication health check
pub struct ReplicationHealthCheck {
    name: String,
    max_lag_ms: u64,
    lag_getter: Arc<dyn Fn() -> u64 + Send + Sync>,
}

impl ReplicationHealthCheck {
    /// Create replication health check
    pub fn new(max_lag_ms: u64, lag_getter: impl Fn() -> u64 + Send + Sync + 'static) -> Self {
        Self {
            name: "replication".to_string(),
            max_lag_ms,
            lag_getter: Arc::new(lag_getter),
        }
    }
}

impl HealthCheck for ReplicationHealthCheck {
    fn name(&self) -> &str {
        &self.name
    }

    fn check(&self) -> HealthCheckResult {
        let start = Instant::now();
        let timestamp = current_timestamp();

        let lag = (self.lag_getter)();

        let (status, details) = if lag > self.max_lag_ms * 2 {
            (
                HealthStatus::Unhealthy,
                Some(format!("Replication lag critical: {} ms", lag)),
            )
        } else if lag > self.max_lag_ms {
            (
                HealthStatus::Degraded,
                Some(format!("Replication lag high: {} ms", lag)),
            )
        } else {
            (
                HealthStatus::Healthy,
                Some(format!("Replication lag OK: {} ms", lag)),
            )
        };

        HealthCheckResult {
            name: self.name.clone(),
            status,
            duration_ms: start.elapsed().as_millis() as u64,
            details,
            timestamp,
        }
    }

    fn is_critical(&self) -> bool {
        false // Replication lag doesn't prevent serving reads
    }
}

/// Health check manager
pub struct HealthCheckManager {
    checks: Arc<RwLock<Vec<Box<dyn HealthCheck>>>>,
    last_results: Arc<RwLock<Vec<HealthCheckResult>>>,
    check_interval: Duration,
}

impl HealthCheckManager {
    /// Create new health check manager
    pub fn new(check_interval: Duration) -> Self {
        Self {
            checks: Arc::new(RwLock::new(Vec::new())),
            last_results: Arc::new(RwLock::new(Vec::new())),
            check_interval,
        }
    }

    /// Register a health check
    pub fn register(&self, check: Box<dyn HealthCheck>) {
        crate::lock_util::write_lock(&self.checks).push(check);
    }

    /// Run all health checks
    pub fn run_checks(&self) -> Vec<HealthCheckResult> {
        let checks = crate::lock_util::read_lock(&self.checks);
        let mut results = Vec::with_capacity(checks.len());

        for check in checks.iter() {
            results.push(check.check());
        }

        *crate::lock_util::write_lock(&self.last_results) = results.clone();
        results
    }

    /// Get overall health status
    pub fn overall_status(&self) -> HealthStatus {
        let results = crate::lock_util::read_lock(&self.last_results);

        let has_unhealthy = results.iter().any(|r| r.status == HealthStatus::Unhealthy);
        let has_degraded = results.iter().any(|r| r.status == HealthStatus::Degraded);

        if has_unhealthy {
            HealthStatus::Unhealthy
        } else if has_degraded {
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        }
    }

    /// Check if ready to serve traffic (liveness)
    pub fn is_live(&self) -> bool {
        // Always live if process is running
        true
    }

    /// Check if ready to receive traffic (readiness)
    pub fn is_ready(&self) -> bool {
        let results = crate::lock_util::read_lock(&self.last_results);
        let checks = crate::lock_util::read_lock(&self.checks);

        // Check critical checks
        for (check, result) in checks.iter().zip(results.iter()) {
            if check.is_critical() && result.status == HealthStatus::Unhealthy {
                return false;
            }
        }

        true
    }

    /// Get last results
    pub fn last_results(&self) -> Vec<HealthCheckResult> {
        crate::lock_util::read_lock(&self.last_results).clone()
    }

    /// Export health status as JSON
    pub fn export_json(&self) -> String {
        let results = crate::lock_util::read_lock(&self.last_results).clone();
        let overall = self.overall_status();

        serde_json::json!({
            "status": format!("{:?}", overall),
            "checks": results,
            "live": self.is_live(),
            "ready": self.is_ready(),
            "timestamp": current_timestamp()
        })
        .to_string()
    }
}

// ============================================================================
// Alert Rules
// ============================================================================

/// Alert severity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

/// Alert state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertState {
    Pending,
    Firing,
    Resolved,
}

/// Alert rule definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    /// Rule name
    pub name: String,
    /// Description
    pub description: String,
    /// Severity
    pub severity: AlertSeverity,
    /// PromQL expression (for reference)
    pub expr: String,
    /// Duration before alerting
    pub for_duration: Duration,
    /// Labels
    pub labels: HashMap<String, String>,
    /// Annotations
    pub annotations: HashMap<String, String>,
}

/// Alert instance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    /// Rule that triggered the alert
    pub rule: AlertRule,
    /// Current state
    pub state: AlertState,
    /// When alert started pending
    pub started_at: u64,
    /// When alert started firing (if firing)
    pub firing_at: Option<u64>,
    /// When alert was resolved (if resolved)
    pub resolved_at: Option<u64>,
    /// Current value
    pub value: f64,
}

/// Predefined alert rules for JouleDB
pub fn default_alert_rules() -> Vec<AlertRule> {
    vec![
        AlertRule {
            name: "HighQueryLatency".to_string(),
            description: "Query latency is above 1 second".to_string(),
            severity: AlertSeverity::Warning,
            expr: "histogram_quantile(0.99, joule_db_query_duration_seconds_bucket) > 1"
                .to_string(),
            for_duration: Duration::from_secs(300),
            labels: HashMap::from([("component".to_string(), "query".to_string())]),
            annotations: HashMap::from([(
                "summary".to_string(),
                "High query latency detected".to_string(),
            )]),
        },
        AlertRule {
            name: "HighReplicationLag".to_string(),
            description: "Replication lag is above 5 seconds".to_string(),
            severity: AlertSeverity::Warning,
            expr: "joule_db_replication_lag_ms > 5000".to_string(),
            for_duration: Duration::from_secs(60),
            labels: HashMap::from([("component".to_string(), "replication".to_string())]),
            annotations: HashMap::from([(
                "summary".to_string(),
                "Replication lag is high".to_string(),
            )]),
        },
        AlertRule {
            name: "LowCacheHitRatio".to_string(),
            description: "Cache hit ratio is below 80%".to_string(),
            severity: AlertSeverity::Warning,
            expr: "joule_db_cache_hit_ratio < 0.8".to_string(),
            for_duration: Duration::from_secs(600),
            labels: HashMap::from([("component".to_string(), "cache".to_string())]),
            annotations: HashMap::from([(
                "summary".to_string(),
                "Cache efficiency is degraded".to_string(),
            )]),
        },
        AlertRule {
            name: "HighConnectionCount".to_string(),
            description: "Connection count is above 90% of max".to_string(),
            severity: AlertSeverity::Critical,
            expr: "joule_db_connections_active / joule_db_connections_max > 0.9".to_string(),
            for_duration: Duration::from_secs(60),
            labels: HashMap::from([("component".to_string(), "connections".to_string())]),
            annotations: HashMap::from([(
                "summary".to_string(),
                "Connection pool nearly exhausted".to_string(),
            )]),
        },
        AlertRule {
            name: "NoLeader".to_string(),
            description: "No leader node in the cluster".to_string(),
            severity: AlertSeverity::Critical,
            expr: "sum(joule_db_is_leader) == 0".to_string(),
            for_duration: Duration::from_secs(30),
            labels: HashMap::from([("component".to_string(), "cluster".to_string())]),
            annotations: HashMap::from([(
                "summary".to_string(),
                "Cluster has no leader".to_string(),
            )]),
        },
        AlertRule {
            name: "HighDiskUsage".to_string(),
            description: "Disk usage is above 85%".to_string(),
            severity: AlertSeverity::Warning,
            expr: "(joule_db_storage_bytes_total / node_filesystem_size_bytes) > 0.85".to_string(),
            for_duration: Duration::from_secs(300),
            labels: HashMap::from([("component".to_string(), "storage".to_string())]),
            annotations: HashMap::from([(
                "summary".to_string(),
                "Disk space running low".to_string(),
            )]),
        },
        // Energy alert rules
        AlertRule {
            name: "HighPowerDraw".to_string(),
            description: "Power draw exceeds 80% of TDP for sustained period".to_string(),
            severity: AlertSeverity::Warning,
            expr: "power_draw_watts / 1000 > (joule_db_config_tdp_watts * 0.8)".to_string(),
            for_duration: Duration::from_secs(120),
            labels: HashMap::from([("component".to_string(), "energy".to_string())]),
            annotations: HashMap::from([(
                "summary".to_string(),
                "System power draw is approaching TDP limit".to_string(),
            )]),
        },
        AlertRule {
            name: "ThermalThrottling".to_string(),
            description: "Thermal state is Serious or Critical".to_string(),
            severity: AlertSeverity::Critical,
            expr: "thermal_state >= 2".to_string(),
            for_duration: Duration::from_secs(30),
            labels: HashMap::from([("component".to_string(), "energy".to_string())]),
            annotations: HashMap::from([(
                "summary".to_string(),
                "Hardware thermal throttling detected — queries may be degraded".to_string(),
            )]),
        },
    ]
}

// ============================================================================
// Grafana Dashboard
// ============================================================================

/// Grafana dashboard panel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardPanel {
    pub title: String,
    pub panel_type: String,
    pub targets: Vec<String>, // PromQL queries
    pub description: Option<String>,
}

/// Grafana dashboard definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardDefinition {
    pub title: String,
    pub description: String,
    pub refresh: String,
    pub panels: Vec<DashboardPanel>,
}

/// Generate default JouleDB dashboard
pub fn default_dashboard() -> DashboardDefinition {
    DashboardDefinition {
        title: "JouleDB Overview".to_string(),
        description: "Comprehensive monitoring dashboard for JouleDB".to_string(),
        refresh: "30s".to_string(),
        panels: vec![
            // Row 1: Overview
            DashboardPanel {
                title: "Queries per Second".to_string(),
                panel_type: "graph".to_string(),
                targets: vec!["rate(joule_db_queries_total[5m])".to_string()],
                description: Some("Total query throughput".to_string()),
            },
            DashboardPanel {
                title: "Query Latency (p99)".to_string(),
                panel_type: "graph".to_string(),
                targets: vec![
                    "histogram_quantile(0.99, rate(joule_db_query_duration_seconds_bucket[5m]))"
                        .to_string(),
                ],
                description: Some("99th percentile query latency".to_string()),
            },
            DashboardPanel {
                title: "Active Connections".to_string(),
                panel_type: "gauge".to_string(),
                targets: vec!["joule_db_connections_active".to_string()],
                description: Some("Current active connections".to_string()),
            },
            // Row 2: Storage
            DashboardPanel {
                title: "Storage Size".to_string(),
                panel_type: "graph".to_string(),
                targets: vec![
                    "joule_db_storage_data_bytes".to_string(),
                    "joule_db_storage_index_bytes".to_string(),
                    "joule_db_storage_wal_bytes".to_string(),
                ],
                description: Some("Storage breakdown by type".to_string()),
            },
            DashboardPanel {
                title: "IOPS".to_string(),
                panel_type: "graph".to_string(),
                targets: vec![
                    "rate(joule_db_page_reads_total[5m])".to_string(),
                    "rate(joule_db_page_writes_total[5m])".to_string(),
                ],
                description: Some("Read and write IOPS".to_string()),
            },
            // Row 3: Cache & Transactions
            DashboardPanel {
                title: "Cache Hit Ratio".to_string(),
                panel_type: "gauge".to_string(),
                targets: vec!["joule_db_cache_hit_ratio".to_string()],
                description: Some("Buffer cache hit ratio".to_string()),
            },
            DashboardPanel {
                title: "Transactions".to_string(),
                panel_type: "graph".to_string(),
                targets: vec![
                    "rate(joule_db_transactions_commits_total[5m])".to_string(),
                    "rate(joule_db_transactions_rollbacks_total[5m])".to_string(),
                ],
                description: Some("Transaction commits and rollbacks".to_string()),
            },
            // Row 4: Replication
            DashboardPanel {
                title: "Replication Lag".to_string(),
                panel_type: "graph".to_string(),
                targets: vec!["joule_db_replication_lag_ms".to_string()],
                description: Some("Replication lag in milliseconds".to_string()),
            },
            DashboardPanel {
                title: "Cluster Status".to_string(),
                panel_type: "stat".to_string(),
                targets: vec![
                    "joule_db_is_leader".to_string(),
                    "joule_db_follower_count".to_string(),
                ],
                description: Some("Leader status and follower count".to_string()),
            },
            // Row 5: Energy (visible when energy feature is enabled)
            DashboardPanel {
                title: "Power Draw (Watts)".to_string(),
                panel_type: "graph".to_string(),
                targets: vec![
                    "power_draw_watts / 1000".to_string(), // stored as milliwatts
                ],
                description: Some("System power consumption in watts".to_string()),
            },
            DashboardPanel {
                title: "Energy per Query (Joules)".to_string(),
                panel_type: "graph".to_string(),
                targets: vec![
                    "histogram_quantile(0.50, rate(energy_joules_per_query_bucket[5m]))"
                        .to_string(),
                    "histogram_quantile(0.99, rate(energy_joules_per_query_bucket[5m]))"
                        .to_string(),
                ],
                description: Some("Energy consumed per query (p50 and p99)".to_string()),
            },
            DashboardPanel {
                title: "Thermal State".to_string(),
                panel_type: "stat".to_string(),
                targets: vec![
                    "thermal_state".to_string(), // 0=Nominal, 1=Fair, 2=Serious, 3=Critical
                ],
                description: Some(
                    "Hardware thermal state (0=Nominal, 1=Fair, 2=Serious, 3=Critical)".to_string(),
                ),
            },
            DashboardPanel {
                title: "Cumulative Energy (Joules)".to_string(),
                panel_type: "graph".to_string(),
                targets: vec![
                    "energy_cumulative_joules / 1000".to_string(), // stored as millijoules
                ],
                description: Some("Total energy consumed since server start".to_string()),
            },
            DashboardPanel {
                title: "GPU Utilization".to_string(),
                panel_type: "gauge".to_string(),
                targets: vec!["gpu_utilization_percent".to_string()],
                description: Some("GPU utilization percentage".to_string()),
            },
        ],
    }
}

/// Export dashboard as JSON (Grafana import format)
pub fn export_dashboard_json(dashboard: &DashboardDefinition) -> String {
    // Simplified export - in production would use full Grafana JSON model
    serde_json::to_string_pretty(dashboard).unwrap_or_default()
}

// ============================================================================
// Structured Logging with Trace Correlation
// ============================================================================

/// Structured log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredLog {
    /// Timestamp (RFC3339)
    pub timestamp: String,
    /// Log level
    pub level: String,
    /// Message
    pub message: String,
    /// Service name
    pub service: String,
    /// Instance ID
    pub instance: String,
    /// Trace ID (for correlation)
    pub trace_id: Option<String>,
    /// Span ID
    pub span_id: Option<String>,
    /// Additional fields
    pub fields: HashMap<String, serde_json::Value>,
}

impl StructuredLog {
    /// Create a new structured log
    pub fn new(level: &str, message: &str, config: &ObservabilityConfig) -> Self {
        Self {
            timestamp: chrono_timestamp(),
            level: level.to_uppercase(),
            message: message.to_string(),
            service: config.service_name.clone(),
            instance: config.instance_id.clone(),
            trace_id: None,
            span_id: None,
            fields: HashMap::new(),
        }
    }

    /// Add trace context
    pub fn with_trace(mut self, trace_id: &str, span_id: &str) -> Self {
        self.trace_id = Some(trace_id.to_string());
        self.span_id = Some(span_id.to_string());
        self
    }

    /// Add a field
    pub fn field<V: Serialize>(mut self, key: &str, value: V) -> Self {
        if let Ok(v) = serde_json::to_value(value) {
            self.fields.insert(key.to_string(), v);
        }
        self
    }

    /// Output as JSON
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Generate instance ID
fn generate_instance_id() -> String {
    use std::time::SystemTime;
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("jouledb-{:x}", now)
}

/// Get hostname
fn hostname() -> Option<String> {
    std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("HOST"))
        .ok()
}

/// Get current timestamp
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Get RFC3339 timestamp
fn chrono_timestamp() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Nanos, true)
}

/// Get disk space using platform-specific statvfs
fn get_disk_space(path: &std::path::Path) -> Option<(u64, u64)> {
    #[cfg(unix)]
    {
        let c_path = std::ffi::CString::new(path.to_str()?).ok()?;
        let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
        let ret = unsafe { libc::statvfs(c_path.as_ptr(), &mut stat) };
        if ret == 0 {
            let avail = stat.f_bavail as u64 * stat.f_frsize as u64;
            let total = stat.f_blocks as u64 * stat.f_frsize as u64;
            return Some((avail, total));
        }
    }
    let _ = path;
    None
}

/// Get memory usage (RSS) using platform-specific APIs
fn get_memory_usage() -> Option<(u64, u64)> {
    #[cfg(target_os = "macos")]
    {
        use std::mem::size_of;
        let mut info: libc::mach_task_basic_info = unsafe { std::mem::zeroed() };
        let mut count =
            (size_of::<libc::mach_task_basic_info>() / size_of::<libc::natural_t>()) as u32;
        #[allow(deprecated)]
        let ret = unsafe {
            libc::task_info(
                libc::mach_task_self(),
                libc::MACH_TASK_BASIC_INFO,
                &mut info as *mut _ as *mut i32,
                &mut count,
            )
        };
        if ret == 0 {
            // Return (used_rss, virtual_size) — virtual_size as rough "total"
            return Some((info.resident_size as u64, info.virtual_size as u64));
        }
    }

    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            let mut rss_kb = None;
            let mut vm_kb = None;
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    rss_kb = line
                        .split_whitespace()
                        .nth(1)
                        .and_then(|s| s.parse::<u64>().ok());
                } else if line.starts_with("VmSize:") {
                    vm_kb = line
                        .split_whitespace()
                        .nth(1)
                        .and_then(|s| s.parse::<u64>().ok());
                }
            }
            if let (Some(rss), Some(vm)) = (rss_kb, vm_kb) {
                return Some((rss * 1024, vm * 1024));
            }
        }
    }

    None
}

// ============================================================================
// OTLP HTTP/JSON Span Exporter
// ============================================================================

/// Lightweight OTLP HTTP/JSON exporter for sending trace spans to an
/// OpenTelemetry-compatible collector (e.g. Jaeger, Tempo, Datadog).
/// Uses the existing `reqwest` client — no additional dependencies required.
pub struct OtlpExporter {
    endpoint: String,
    resource: OtelResource,
    client: reqwest::Client,
}

impl OtlpExporter {
    /// Create a new OTLP exporter targeting the given endpoint.
    /// The endpoint should be the OTLP HTTP/JSON traces endpoint,
    /// e.g. `http://localhost:4318/v1/traces`.
    pub fn new(endpoint: &str, config: &ObservabilityConfig) -> Self {
        Self {
            endpoint: endpoint.to_string(),
            resource: OtelResource::from_config(config),
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(5))
                .build()
                .unwrap_or_default(),
        }
    }

    /// Export a batch of spans to the OTLP collector.
    /// Returns `Ok(())` on success or a descriptive error string.
    pub async fn export(&self, spans: &[OtelSpan]) -> Result<(), String> {
        if spans.is_empty() {
            return Ok(());
        }

        // Build OTLP JSON payload
        let resource_spans = serde_json::json!({
            "resourceSpans": [{
                "resource": {
                    "attributes": self.resource.to_attributes()
                        .iter()
                        .map(|a| serde_json::json!({
                            "key": a.key,
                            "value": match &a.value {
                                OtelAttributeValue::String(s) => serde_json::json!({"stringValue": s}),
                                OtelAttributeValue::Int(i) => serde_json::json!({"intValue": i.to_string()}),
                                OtelAttributeValue::Double(d) => serde_json::json!({"doubleValue": d}),
                                OtelAttributeValue::Bool(b) => serde_json::json!({"boolValue": b}),
                                OtelAttributeValue::StringArray(a) => serde_json::json!({"arrayValue": {"values": a.iter().map(|s| serde_json::json!({"stringValue": s})).collect::<Vec<_>>()}}),
                                OtelAttributeValue::IntArray(a) => serde_json::json!({"arrayValue": {"values": a.iter().map(|i| serde_json::json!({"intValue": i.to_string()})).collect::<Vec<_>>()}}),
                            }
                        }))
                        .collect::<Vec<_>>()
                },
                "scopeSpans": [{
                    "scope": {
                        "name": "joule-db",
                        "version": env!("CARGO_PKG_VERSION")
                    },
                    "spans": spans.iter().map(|s| serde_json::json!({
                        "traceId": s.trace_id,
                        "spanId": s.span_id,
                        "parentSpanId": s.parent_span_id.as_deref().unwrap_or(""),
                        "name": s.name,
                        "kind": match s.kind {
                            SpanKind::Server => 2,
                            SpanKind::Client => 3,
                            SpanKind::Producer => 4,
                            SpanKind::Consumer => 5,
                            SpanKind::Internal => 1,
                        },
                        "startTimeUnixNano": s.start_time_unix_nano.to_string(),
                        "endTimeUnixNano": s.end_time_unix_nano.to_string(),
                        "status": {
                            "code": match s.status.code {
                                StatusCode::Ok => 1,
                                StatusCode::Error => 2,
                                StatusCode::Unset => 0,
                            }
                        }
                    })).collect::<Vec<_>>()
                }]
            }]
        });

        let resp = self
            .client
            .post(&self.endpoint)
            .header("Content-Type", "application/json")
            .json(&resource_spans)
            .send()
            .await
            .map_err(|e| format!("OTLP export failed: {}", e))?;

        if resp.status().is_success() {
            Ok(())
        } else {
            Err(format!("OTLP collector returned {}", resp.status()))
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_observability_config_default() {
        let config = ObservabilityConfig::default();
        assert_eq!(config.service_name, "joule-db");
        assert!(config.tracing_enabled);
        assert!(config.metrics_enabled);
        assert_eq!(config.prometheus_path, "/metrics");
    }

    #[test]
    fn test_otel_resource() {
        let config = ObservabilityConfig::default();
        let resource = OtelResource::from_config(&config);

        assert_eq!(resource.service_name, "joule-db");
        assert_eq!(resource.telemetry_sdk_language, "rust");

        let attrs = resource.to_attributes();
        assert!(!attrs.is_empty());
    }

    #[test]
    fn test_otel_attribute() {
        let attr = OtelAttribute::new("key", "value");
        assert_eq!(attr.key, "key");

        let int_attr = OtelAttribute::new_int("count", 42);
        match int_attr.value {
            OtelAttributeValue::Int(v) => assert_eq!(v, 42),
            _ => panic!("Expected Int"),
        }
    }

    #[test]
    fn test_database_metrics() {
        let metrics = DatabaseMetricsEnhanced::new();

        // Record some queries
        metrics
            .queries
            .record("select", Duration::from_millis(50), true);
        metrics
            .queries
            .record("insert", Duration::from_millis(100), true);
        metrics
            .queries
            .record("select", Duration::from_secs(2), true); // slow

        let output = metrics.export_prometheus();
        assert!(output.contains("joule_db_queries_total"));
        assert!(output.contains("joule_db_slow_queries_total"));
    }

    #[test]
    fn test_connection_metrics() {
        let metrics = DatabaseMetricsEnhanced::new();

        metrics.connections.connection_opened();
        metrics.connections.connection_opened();
        metrics.connections.connection_closed();

        let output = metrics.connections.export();
        assert!(output.contains("joule_db_connections_active 1"));
        assert!(output.contains("joule_db_connections_total 2"));
    }

    #[test]
    fn test_cache_metrics() {
        let metrics = CacheMetrics::new();

        metrics.hit();
        metrics.hit();
        metrics.hit();
        metrics.miss();

        let ratio = metrics.hit_ratio();
        assert!((ratio - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_health_check_manager() {
        let manager = HealthCheckManager::new(Duration::from_secs(30));

        // Run checks (none registered)
        let results = manager.run_checks();
        assert!(results.is_empty());

        assert!(manager.is_live());
        assert!(manager.is_ready());
    }

    #[test]
    fn test_memory_health_check() {
        let check = MemoryHealthCheck::new(0.9);
        let result = check.check();

        assert_eq!(result.name, "memory");
        assert!(result.details.is_some());
    }

    #[test]
    fn test_alert_rules() {
        let rules = default_alert_rules();
        assert!(!rules.is_empty());

        // Check critical rules exist
        let has_no_leader = rules.iter().any(|r| r.name == "NoLeader");
        assert!(has_no_leader);

        // Check energy alert rules
        let has_high_power = rules.iter().any(|r| r.name == "HighPowerDraw");
        let has_thermal = rules.iter().any(|r| r.name == "ThermalThrottling");
        assert!(has_high_power);
        assert!(has_thermal);

        // ThermalThrottling should be Critical severity
        let thermal_rule = rules
            .iter()
            .find(|r| r.name == "ThermalThrottling")
            .unwrap();
        assert_eq!(thermal_rule.severity, AlertSeverity::Critical);
        assert_eq!(thermal_rule.for_duration, Duration::from_secs(30));
    }

    #[test]
    fn test_dashboard() {
        let dashboard = default_dashboard();

        assert_eq!(dashboard.title, "JouleDB Overview");
        assert!(!dashboard.panels.is_empty());

        let json = export_dashboard_json(&dashboard);
        assert!(json.contains("JouleDB Overview"));

        // Check energy panels exist
        let has_power_panel = dashboard
            .panels
            .iter()
            .any(|p| p.title == "Power Draw (Watts)");
        let has_energy_panel = dashboard
            .panels
            .iter()
            .any(|p| p.title == "Energy per Query (Joules)");
        let has_thermal_panel = dashboard.panels.iter().any(|p| p.title == "Thermal State");
        assert!(has_power_panel);
        assert!(has_energy_panel);
        assert!(has_thermal_panel);
    }

    #[test]
    fn test_structured_log() {
        let config = ObservabilityConfig::default();
        let log = StructuredLog::new("INFO", "Test message", &config)
            .with_trace("abc123", "def456")
            .field("key", "value")
            .field("count", 42);

        assert_eq!(log.level, "INFO");
        assert_eq!(log.message, "Test message");
        assert_eq!(log.trace_id, Some("abc123".to_string()));
        assert!(log.fields.contains_key("key"));

        let json = log.to_json();
        assert!(json.contains("Test message"));
    }

    #[test]
    fn test_span_status() {
        let status = SpanStatus::default();
        assert_eq!(status.code, StatusCode::Unset);
        assert!(status.message.is_none());
    }

    #[test]
    fn test_span_kind() {
        let kind = SpanKind::default();
        assert_eq!(kind, SpanKind::Internal);
    }

    #[test]
    fn test_health_status_serialization() {
        let status = HealthStatus::Healthy;
        let json = serde_json::to_string(&status).unwrap();
        assert!(json.contains("Healthy"));
    }

    #[test]
    fn test_alert_severity() {
        assert!(AlertSeverity::Critical as u8 > AlertSeverity::Warning as u8);
        assert!(AlertSeverity::Warning as u8 > AlertSeverity::Info as u8);
    }

    #[test]
    fn test_transaction_metrics() {
        let metrics = TransactionMetrics::new();

        metrics.begin();
        metrics.begin();
        metrics.commit();
        metrics.rollback();

        let output = metrics.export();
        assert!(output.contains("joule_db_transactions_commits_total 1"));
        assert!(output.contains("joule_db_transactions_rollbacks_total 1"));
    }

    #[test]
    fn test_storage_metrics() {
        let metrics = StorageMetrics::new();

        metrics.update_sizes(1000, 500, 200);
        metrics.page_read();
        metrics.page_write();
        metrics.compaction();

        let output = metrics.export();
        assert!(output.contains("joule_db_storage_bytes_total 1700"));
        assert!(output.contains("joule_db_page_reads_total 1"));
    }

    #[test]
    fn test_chrono_timestamp_is_valid_rfc3339() {
        let ts = chrono_timestamp();
        // Should be a valid RFC3339 timestamp parseable by chrono
        let parsed = chrono::DateTime::parse_from_rfc3339(&ts);
        assert!(parsed.is_ok(), "Timestamp '{}' should be valid RFC3339", ts);
        // Should contain the current year
        assert!(
            ts.starts_with("202"),
            "Timestamp should be in the 2020s: {}",
            ts
        );
    }

    #[test]
    fn test_get_disk_space_returns_something() {
        let result = get_disk_space(std::path::Path::new("/"));
        // On Unix, this should succeed
        #[cfg(unix)]
        assert!(result.is_some(), "Should get disk space on Unix");
        let _ = result; // suppress unused warning on other platforms
    }

    #[test]
    fn test_get_memory_usage_returns_something() {
        let result = get_memory_usage();
        // On macOS and Linux, this should succeed
        #[cfg(any(target_os = "macos", target_os = "linux"))]
        assert!(result.is_some(), "Should get memory usage on macOS/Linux");
        let _ = result;
    }

    #[test]
    fn test_otlp_exporter_creation() {
        let config = ObservabilityConfig::default();
        let exporter = OtlpExporter::new("http://localhost:4318/v1/traces", &config);
        assert_eq!(exporter.endpoint, "http://localhost:4318/v1/traces");
        assert_eq!(exporter.resource.service_name, "joule-db");
    }
}
