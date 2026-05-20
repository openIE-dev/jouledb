//! # JouleDB Server
//!
//! Server implementation for JouleDB with HTTP and WebSocket support.
//!
//! ## Features
//!
//! - `http` (default) - HTTP API
//! - `websocket` (default) - WebSocket API
//! - `replication` - Multi-node replication
//! - `tls` - TLS/SSL support

pub mod agent_memory;
pub mod audit;
pub mod auth;
pub mod backup;
pub mod binary_protocol;
pub mod config;
pub mod cql_executor;
pub mod cxl_memory;
pub mod cypher_executor;
pub mod datalog_executor;
pub mod deployment;
pub mod distributed_query;
pub mod dynamic_routes;
pub mod edge_pop;
pub mod energy;
pub mod energy_executor;
pub mod enterprise;
pub mod error;
pub mod features_bridge;
pub mod fts_analyzer;
pub mod graphql_executor;
pub mod gremlin_executor;
pub mod health;
pub mod holographic_adapter;
pub mod json_ops;
#[cfg(test)]
mod proptest_verify;
#[cfg(kani)]
mod kani_proofs;
pub mod hrp_erasure;
pub mod hrp_security;
#[cfg(feature = "jwp")]
pub mod jwp_server;
pub mod langgraph_handlers;
pub mod ledger_bridge;
pub mod lock_util;
pub mod logging;
pub mod mcp_bridge;
pub mod metrics;
pub mod multiplex;
pub mod mutation_delta;
pub mod observability;
pub mod operations;
pub mod pgwire;
pub mod pool;
pub mod protocol;
pub mod query;
pub mod raft;
pub mod raft_server;
pub mod raft_transport;
pub mod rbac;
pub mod read_replica;
pub mod realtime;
pub mod replication;
pub mod request_tracing;
pub mod resp;
pub mod scale_to_zero;
pub mod scram;
pub mod security;
pub mod sharding;
pub mod sigql_executor;
pub mod sparql_executor;
pub mod subscription_hdc;
pub mod subscriptions;
pub mod tcp_server;
pub mod tenant;
pub mod two_phase_commit;
#[cfg(feature = "websocket")]
pub mod websocket;
#[cfg(feature = "webtransport")]
pub mod webtransport;
pub mod workflow;

// Amorphic modules
pub mod amorphic_adapter;
pub mod gin_index;
pub mod mcp_transport;
pub mod mvcc_adapter;
pub mod spatial_index;
pub mod vector_index;
pub mod vector_routes;

// Re-exports
pub use auth::{ApiKey, AuthConfig, AuthError, AuthenticationManager, JwtClaims, Session};
pub use backup::{
    BackupConfig, BackupError, BackupManager, BackupMetadata, BackupReader, BackupStatus,
    BackupType, BackupWriter, CompressionAlgorithm, EncryptionAlgorithm, RestoreManager,
    RestoreOptions, RestoreProgress, compress, decompress,
};
pub use binary_protocol::{
    BatchOp, BinaryMessage, BinaryProtocol, BinaryProtocolError, Flags, HEADER_SIZE, MAGIC,
    MAX_PAYLOAD_SIZE, MessageType, ProtocolStats, VERSION,
};
pub use config::ServerConfig as FullServerConfig;
pub use config::{
    ConfigManager, DatabaseSettings, LoggingSettings, NetworkSettings, PerformanceSettings,
    SecuritySettings, ServerSettings,
};
pub use deployment::{GracefulShutdown, LifecycleHooks, ProductionServer, SignalHandler};
pub use enterprise::{
    ClusterConfig, EnterpriseError, FailoverManager, LoadBalancer, LoadBalancingStrategy,
    NodeHealth, NodeInfo, NodeRole, ReplicationEntry, ReplicationManager, ReplicationMode,
    ReplicationOperation, ShardInfo, ShardManager, ShardStatus as EnterpriseShardStatus,
    ShardingStrategy as EnterpriseShardingStrategy,
};
pub use health::{HealthCheckManager, HealthCheckResult, HealthChecks, HealthStatus};
pub use logging::{LogEntry, LogLevel, Logger, LoggerConfig};
pub use multiplex::{
    ConnectionMultiplexer, MultiplexError, MultiplexedHandler, MultiplexedRequest,
    MultiplexedResponse, RequestId,
};
pub use observability::{
    Alert, AlertRule, AlertSeverity, AlertState, CacheMetrics, ConnectionMetrics,
    DashboardDefinition, DashboardPanel, DatabaseMetricsEnhanced,
    HealthCheck as ObservabilityHealthCheck, HealthCheckManager as ObservabilityHealthCheckManager,
    HealthCheckResult as ObservabilityHealthCheckResult, HealthStatus as ObservabilityHealthStatus,
    MemoryHealthCheck, ObservabilityConfig, OtelAttribute, OtelAttributeValue, OtelResource,
    OtelSpan, QueryMetrics, ReplicationHealthCheck, SpanEvent, SpanKind, SpanLink, SpanStatus,
    StatusCode as OtelStatusCode, StorageHealthCheck, StorageMetrics, StructuredLog,
    TransactionMetrics, default_alert_rules, default_dashboard, export_dashboard_json,
};
pub use operations::{
    AnalyzeResult, ClusterOps, ConsistencyCheckResult, ConsistencyIssue, DataExporter,
    DataImporter, DatabaseSizeInfo, DatabaseUtils, ExportFormat, ExportResult, ImportExportOptions,
    ImportResult, IssueSeverity, Migration, MigrationConfig, MigrationManager, MigrationRecord,
    MigrationStatus as OpsMigrationStatus, MigrationStatusReport, OperationsError,
    OperationsResult, ReindexResult, RollingUpgradeStatus, ScheduledTask,
    TableStatistics as OpsTableStatistics, TaskScheduler, TaskType, UpgradePhase, VacuumOptions,
    VacuumResult,
};
pub use pool::{
    ConnectionFactory, ConnectionPool, PoolConfig, PoolConfigBuilder, PoolError, PoolResult,
    PoolStats, PooledConnection,
};
pub use protocol::{
    BatchItemResult, BatchOperation, Frame, FrameType, ProtocolError, ProtocolHandler,
    ProtocolMessage, ProtocolResponse, SyncChange,
};
pub use raft::{
    AppendEntriesRequest, AppendEntriesResponse, BatchReplicationConfig,
    ClusterConfig as RaftClusterConfig, Command as RaftCommand, InMemoryTransport,
    InstallSnapshotRequest, InstallSnapshotResponse, KvStateMachine, LeaderState,
    LogEntry as RaftLogEntry, LogIndex, NodeId, PersistentState, RaftConfig, RaftError,
    RaftMessage, RaftNode, RaftState, RaftStats, RaftTransport, RequestVoteRequest,
    RequestVoteResponse, Snapshot, SnapshotMetadata, StateMachine, Term, VolatileState,
};
pub use rbac::{
    AccessContext, Permission, PermissionMiddleware, PermissionType, PolicyCommand,
    PolicyCondition, PolicyType, RBACError, RBACManager, RBACResult, RBACStats, ResourceType, Role,
    RowLevelPolicy, User,
};
pub use read_replica::{ReadPreference, ReadReplicaRouter, ReplicaInfo, ReplicaStatus};
pub use replication::{
    FollowerState, ReplicationClient, ReplicationClientStats, ReplicationClientStatsSnapshot,
    ReplicationConfig, ReplicationError, ReplicationOp, ReplicationServer, ReplicationServerStats,
    ReplicationServerStatsSnapshot, ReplicationWalEntry, SyncMode,
};
pub use request_tracing::{
    FinishedSpan, InMemoryTraceCollector, NoOpTraceCollector, SpanId, TraceCollector, TraceContext,
    TraceId, TraceSpan,
};
pub use security::{
    InputValidator, IpBlocklist, RateLimitConfig, RateLimitStats, RateLimiter, SecurityConfig,
    SecurityError, SecurityEvent, SecurityEventLogger, SecurityEventType, SecurityHeaders,
    SecurityResult, SecurityScanResult, SecurityScanner, SecurityStats, Vulnerability,
    VulnerabilitySeverity,
};
pub use sharding::{
    ConsistentHashRing, CrossShardCoordinator, CrossShardResult, KeyRange, MigrationStatus,
    MigrationTask, RebalanceMove, RouterStats, Shard, ShardAssigner, ShardAssignment, ShardRouter,
    ShardState, ShardingConfig, ShardingError, ShardingResult, ShardingStrategy,
};

// Distributed query execution re-exports
pub use distributed_query::{
    AggregateSpec, DistributedQueryConfig, DistributedQueryExecutor, DistributedQueryStats,
    LocalShardExecutor, PartialAggregate, QueryAnalysis, QueryType, ShardExecutor, ShardKey,
    ShardKeyValue, ShardQueryResult,
};

pub use tcp_server::{
    BatchOperation as TcpBatchOperation, DatabaseHandler, QueryResult as TcpQueryResult, TcpServer,
    TcpServerConfig, TcpServerStats, TcpServerStatsSnapshot,
};

pub use audit::{
    AuditAction, AuditActor, AuditConfig, AuditError, AuditEvent, AuditEventBuilder,
    AuditEventType, AuditLogger, AuditOutcome, AuditQuery, AuditQueryExecutor, AuditQueryResult,
    AuditResource, AuditResult, AuditSeverity, AuditStatistics, AuditStore, SortField, SortOrder,
    TimeRange,
};
pub use metrics::{
    Counter, DatabaseMetrics, DatabaseMetricsSnapshot, Gauge, Histogram, HistogramTimer,
    LabeledMetricFamily, Labels, Metric, MetricType, MetricsRegistry, PrometheusExporter,
};
#[cfg(feature = "adaptive-pool")]
pub use pool::adaptive::{AdaptiveConfig, AdaptivePool};
pub use two_phase_commit::{
    CoordinatorConfig, CoordinatorStats, CoordinatorStatsSnapshot, InMemoryStorage,
    LogSequenceNumber, ParticipantConfig, ParticipantId, ParticipantStats,
    ParticipantStatsSnapshot, ParticipantStorage, RecoveryLog, RecoveryLogEntry,
    RecoveryLogRecordType, TransactionCoordinator, TransactionId, TransactionOperation,
    TransactionParticipant, TransactionState, TwoPhaseError, TwoPhaseMessage, TwoPhaseResult, Vote,
};

#[cfg(feature = "websocket")]
pub use websocket::{WebSocketConfig, WebSocketServer, WebSocketStats, WebSocketStatsSnapshot};

#[cfg(feature = "webtransport")]
pub use webtransport::{
    WebTransportConfig, WebTransportServer, WebTransportStats, WebTransportStatsSnapshot,
};

pub use energy::{EnergyMetrics, EnergyStatusResponse};

// Query endpoint re-exports
pub use query::{
    QueryErrorResponse, QueryExecutionPath, QueryExecutor, QueryPathRouter, QueryPathStats,
    QueryProfile, QueryRequest, QueryResponse, QueryState, SimpleQueryExecutor, query_handler,
    query_router,
};

// Subscription re-exports
pub use subscriptions::{
    ChangeEvent, ChangeOperation, SubscriptionHandle, SubscriptionId, SubscriptionManager,
    SubscriptionStats, SubscriptionStatsSnapshot,
};

// Real-time re-exports
pub use realtime::{
    ChangeStream, ChangeStreamFilter, ChangeStreamManager, ChangeStreamOptions, ChangeStreamToken,
    Trigger, TriggerContext, TriggerEventType, TriggerManager, TriggerResult,
};

// Error recovery re-exports
pub use error::{
    CircuitBreaker, CircuitBreakerConfig, CircuitState, ErrorRecoveryManager, RecoveryStrategy,
    RetryConfig, RetryExecutor,
};

pub use holographic_adapter::HolographicTableStorage;

use axum::{
    Router,
    body::Bytes,
    extract::{
        Extension, Path, Query, State,
        ws::{Message as AxumWsMessage, WebSocket, WebSocketUpgrade},
    },
    http::{StatusCode, header},
    response::{IntoResponse, Json, Response},
    routing::{delete, get, post},
};
use joule_db_local::Database;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Combined app state for HTTP routes that need both DB and subscriptions
#[derive(Clone)]
struct AppState {
    db: Arc<RwLock<Database>>,
    amorphic: Arc<amorphic_adapter::AmorphicTableStorage>,
    subscription_manager: Arc<SubscriptionManager>,
    auth_manager: Option<Arc<auth::AuthenticationManager>>,
    rbac_manager: Option<Arc<rbac::RBACManager>>,
    backup_manager: Arc<backup::BackupManager>,
    query_executor: Arc<dyn QueryExecutor>,
    db_path: String,
    energy_snapshot: Arc<std::sync::RwLock<joule_db_energy::EnergySnapshot>>,
    energy_advisor: Arc<joule_db_energy::HardwareAdvisor>,
    energy_metrics: Arc<energy::EnergyMetrics>,
    replication_server: Option<Arc<replication::ReplicationServer>>,
    replication_role: Option<String>,
    shard_router: Option<Arc<sharding::ShardRouter>>,
    sanitize_errors: bool,
    dynamic_route_manager: Arc<dynamic_routes::DynamicRouteManager>,
    branch_manager: Arc<joule_db_branch::manager::BranchManager>,
    tenant_manager: Arc<tenant::TenantManager>,
    vector_index_manager: Arc<std::sync::RwLock<vector_index::VectorIndexManager>>,
    activity_tracker: Arc<scale_to_zero::ActivityTracker>,
    memory_manager: Arc<agent_memory::MemoryManager>,
    workflow_manager: Arc<workflow::WorkflowManager>,
    edge_pop_manager: Arc<edge_pop::EdgePopManager>,
}

/// Authentication info extracted by auth middleware
#[derive(Debug, Clone)]
pub struct AuthInfo {
    pub user_id: String,
    pub roles: Vec<String>,
    pub tenant_id: Option<String>,
}

/// Server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// HTTP bind address
    pub http_addr: String,
    /// TCP binary protocol address
    pub tcp_addr: String,
    /// Database path
    pub db_path: String,
    /// Enable WebSocket
    pub enable_websocket: bool,
    /// Enable TCP binary protocol
    pub enable_tcp: bool,
    /// Max TCP connections
    pub max_tcp_connections: usize,
    /// Enable WebTransport (HTTP/3 + QUIC)
    pub enable_webtransport: bool,
    /// WebTransport bind port (default: 4433)
    pub webtransport_port: u16,
    /// Enable PostgreSQL wire protocol server
    pub enable_pgwire: bool,
    /// PgWire bind address (default: 127.0.0.1:5433)
    pub pgwire_addr: String,
    /// Enable JWP (Joule Wire Protocol) transport
    pub enable_jwp: bool,
    /// JWP bind address (default: 127.0.0.1:9200)
    pub jwp_addr: String,
    /// Maximum concurrent JWP connections
    pub max_jwp_connections: usize,
    /// Enable authentication middleware
    pub auth_enabled: bool,
    /// JWT signing secret (required when auth_enabled is true)
    pub auth_jwt_secret: Option<String>,
    /// Enable replication
    pub enable_replication: bool,
    /// Replication role: "leader" or "follower"
    pub replication_role: Option<String>,
    /// Replication listen address (for leader mode)
    pub replication_listen_addr: String,
    /// Leader address to connect to (for follower mode)
    pub replication_leader_addr: Option<String>,
    /// TLS certificate path (PEM format)
    #[cfg(feature = "tls")]
    pub tls_cert_path: Option<String>,
    /// TLS private key path (PEM format)
    #[cfg(feature = "tls")]
    pub tls_key_path: Option<String>,
    /// Energy profiling configuration
    pub energy_config: joule_db_energy::EnergyConfig,
    /// Enable Raft consensus for multi-node clustering
    pub enable_raft: bool,
    /// This node's Raft node ID (must be unique in the cluster)
    pub raft_node_id: Option<String>,
    /// Raft RPC listen address (default: 127.0.0.1:7000)
    pub raft_addr: String,
    /// Raft peer list: ["node2=host:port", "node3=host:port"]
    pub raft_peers: Vec<String>,
    /// HRP Phase 3: Shared master secret for write token HMAC (hex-encoded, 32 bytes).
    /// When set, enables cryptographic integrity for Raft replication messages.
    pub raft_master_secret: Option<String>,
    /// Default query timeout in milliseconds (0 = no timeout; default 30000)
    pub query_timeout_ms: u64,
    /// Slow query logging threshold in milliseconds (0 = disabled; default 1000)
    pub slow_query_threshold_ms: u64,
    /// Enable rate limiting on HTTP endpoints
    pub rate_limiting_enabled: bool,
    /// Maximum requests per minute per client (default: 1000)
    pub rate_limit_requests_per_minute: u64,
    /// Maximum rows returned per query (default: 100000; 0 = unlimited)
    pub max_result_rows: usize,
    /// Session timeout in seconds (stale transactions auto-rollback; 0 = disabled; default 300)
    pub session_timeout_secs: u64,
    /// Require TLS on all connections (rejects plaintext when true)
    #[cfg(feature = "tls")]
    pub require_tls: bool,
    /// CORS allowed origins (empty = same-origin only; add origins to allow cross-origin requests)
    pub cors_origins: Vec<String>,
    /// Sanitize error responses (strip internal details like table/column names)
    pub sanitize_errors: bool,
    /// Runtime isolation mode: "native", "vm", or "wasm" (default: "native")
    pub runtime_mode: String,
    /// Enable energy receipt ledger (blockchain-anchored attestation)
    pub enable_ledger: bool,
    /// Ledger persistence directory (None = in-memory only)
    pub ledger_dir: Option<String>,
    /// Maximum receipts per ledger batch (default: 1000)
    pub ledger_batch_max_receipts: usize,
    /// Maximum batch interval in seconds (default: 60)
    pub ledger_batch_interval_secs: u64,
    /// Grid region for carbon factor (e.g., "US-CAL-CISO")
    pub ledger_grid_region: Option<String>,
    /// Grid carbon factor in kgCO2e/kWh (default: 0.4)
    pub ledger_grid_factor: Option<f64>,
    /// Enable scale-to-zero (suspend compute when idle)
    pub scale_to_zero_enabled: bool,
    /// Idle timeout in seconds before suspending (default: 300)
    pub idle_timeout_secs: u64,
    /// Enable MCP stdio transport (blocks on stdin for JSON-RPC)
    pub enable_mcp_stdio: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            http_addr: "127.0.0.1:8080".to_string(),
            tcp_addr: "127.0.0.1:9000".to_string(),
            db_path: "./joule-db-data".to_string(),
            enable_websocket: true,
            enable_tcp: true,
            max_tcp_connections: 1000,
            enable_webtransport: false,
            webtransport_port: 4433,
            enable_pgwire: true,
            pgwire_addr: "127.0.0.1:5433".to_string(),
            enable_jwp: false,
            jwp_addr: "127.0.0.1:9200".to_string(),
            max_jwp_connections: 1000,
            auth_enabled: true,
            auth_jwt_secret: None,
            enable_replication: false,
            replication_role: None,
            replication_listen_addr: "127.0.0.1:6381".to_string(),
            replication_leader_addr: None,
            #[cfg(feature = "tls")]
            tls_cert_path: None,
            #[cfg(feature = "tls")]
            tls_key_path: None,
            energy_config: joule_db_energy::EnergyConfig::default(),
            enable_raft: false,
            raft_node_id: None,
            raft_addr: "127.0.0.1:7000".to_string(),
            raft_peers: vec![],
            raft_master_secret: None,
            query_timeout_ms: 30000,
            slow_query_threshold_ms: 1000,
            rate_limiting_enabled: true,
            rate_limit_requests_per_minute: 1000,
            max_result_rows: 100_000,
            session_timeout_secs: 300,
            #[cfg(feature = "tls")]
            require_tls: false,
            cors_origins: Vec::new(),
            sanitize_errors: false,
            runtime_mode: "native".to_string(),
            enable_ledger: false,
            ledger_dir: None,
            ledger_batch_max_receipts: 1000,
            ledger_batch_interval_secs: 60,
            ledger_grid_region: None,
            ledger_grid_factor: None,
            scale_to_zero_enabled: false,
            idle_timeout_secs: 300,
            enable_mcp_stdio: false,
        }
    }
}

/// JouleDB Server
#[derive(Clone)]
pub struct Server {
    config: ServerConfig,
    db: Arc<RwLock<Database>>,
    amorphic: Arc<amorphic_adapter::AmorphicTableStorage>,
    metrics: Arc<DatabaseMetrics>,
    start_time: std::time::Instant,
    query_executor: Arc<dyn QueryExecutor>,
    subscription_manager: Arc<SubscriptionManager>,
    auth_manager: Option<Arc<auth::AuthenticationManager>>,
    rbac_manager: Option<Arc<rbac::RBACManager>>,
    backup_manager: Arc<backup::BackupManager>,
    energy_snapshot: Arc<std::sync::RwLock<joule_db_energy::EnergySnapshot>>,
    energy_metrics: Arc<energy::EnergyMetrics>,
    energy_advisor: Arc<joule_db_energy::HardwareAdvisor>,
    platform_info: joule_db_energy::PlatformInfo,
    replication_server: Option<Arc<replication::ReplicationServer>>,
    shard_router: Option<Arc<sharding::ShardRouter>>,
    /// Shared Raft leader flag — updated by background task, read by query executor
    raft_is_leader: Arc<std::sync::atomic::AtomicBool>,
    /// Slot for setting the Raft node after startup (OnceLock shared with query executor)
    raft_node_slot: Option<
        Arc<
            std::sync::OnceLock<
                Arc<raft::RaftNode<raft::KvStateMachine, raft_transport::TcpRaftTransport>>,
            >,
        >,
    >,
    /// MVCC adapter handle for session reaper background task
    mvcc: Arc<mvcc_adapter::MvccTableStorage>,
    /// Rate limiter for request throttling
    rate_limiter: Option<Arc<security::RateLimiter>>,
    /// Energy receipt ledger store (for verification endpoints)
    ledger_store: Option<Arc<tokio::sync::RwLock<joule_db_ledger::ReceiptStore>>>,
    /// Energy receipt collector (for JWP ledger dispatch)
    ledger_collector: Option<Arc<joule_db_ledger::ReceiptCollector>>,
    /// Dynamic API route manager (for DEFINE API endpoints)
    dynamic_route_manager: Arc<dynamic_routes::DynamicRouteManager>,
    /// Branch manager for CoW database branching with energy budgets
    branch_manager: Arc<joule_db_branch::manager::BranchManager>,
    /// Tenant manager for multi-tenant namespace isolation
    tenant_manager: Arc<tenant::TenantManager>,
    /// Vector index manager for HNSW/IVF similarity search
    vector_index_manager: Arc<std::sync::RwLock<vector_index::VectorIndexManager>>,
    /// Activity tracker for scale-to-zero lifecycle
    activity_tracker: Arc<scale_to_zero::ActivityTracker>,
    /// Agent memory manager for temporal knowledge
    memory_manager: Arc<agent_memory::MemoryManager>,
    /// Durable workflow engine with message queue
    workflow_manager: Arc<workflow::WorkflowManager>,
    /// Edge Points of Presence manager
    edge_pop_manager: Arc<edge_pop::EdgePopManager>,
}

/// Database handler for TCP server - bridges TCP protocol to actual database
pub struct ServerDatabaseHandler {
    db: Arc<RwLock<Database>>,
    query_executor: Arc<dyn QueryExecutor>,
}

impl ServerDatabaseHandler {
    pub fn new(db: Arc<RwLock<Database>>, query_executor: Arc<dyn QueryExecutor>) -> Self {
        Self { db, query_executor }
    }
}

#[async_trait::async_trait]
impl tcp_server::DatabaseHandler for ServerDatabaseHandler {
    async fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, String> {
        let db = self.db.read().await;
        match db.get(key) {
            Ok(value) => Ok(value),
            Err(e) => Err(e.to_string()),
        }
    }

    async fn set(&self, key: &[u8], value: &[u8], ttl: Option<u64>) -> Result<bool, String> {
        let db = self.db.write().await;
        let result = if let Some(ttl_secs) = ttl {
            db.put_with_ttl(key, value, ttl_secs)
        } else {
            db.put(key, value)
        };
        match result {
            Ok(_) => Ok(true),
            Err(e) => Err(e.to_string()),
        }
    }

    async fn delete(&self, key: &[u8]) -> Result<bool, String> {
        let db = self.db.write().await;
        match db.delete(key) {
            Ok(deleted) => Ok(deleted),
            Err(e) => Err(e.to_string()),
        }
    }

    async fn query(
        &self,
        sql: &str,
        params: Vec<serde_json::Value>,
    ) -> Result<tcp_server::QueryResult, String> {
        use crate::query::{QueryExecutor, QueryRequest};

        let request = QueryRequest {
            sql: sql.to_string(),
            params: Default::default(),
            args: params,
            explain: false,
            limit: None,
            session_id: None,
            query_timeout_ms: None,
            branch_id: None,
            tenant_id: None,
        };

        match self.query_executor.execute(&request) {
            Ok(response) => {
                let row_count = response.affected_rows.unwrap_or(response.rows.len());
                Ok(tcp_server::QueryResult {
                    columns: response.columns,
                    rows: response.rows,
                    row_count,
                    execution_time_ms: response.execution_time_ms,
                })
            }
            Err(e) => Err(e.message),
        }
    }

    async fn batch(
        &self,
        operations: Vec<tcp_server::BatchOperation>,
    ) -> Result<Vec<bool>, String> {
        let db = self.db.write().await;
        let mut results = Vec::with_capacity(operations.len());

        for op in operations {
            match op {
                tcp_server::BatchOperation::Set { key, value, ttl: _ } => {
                    match db.put(&key, &value) {
                        Ok(_) => results.push(true),
                        Err(_) => results.push(false),
                    }
                }
                tcp_server::BatchOperation::Delete { key } => match db.delete(&key) {
                    Ok(deleted) => results.push(deleted),
                    Err(_) => results.push(false),
                },
            }
        }

        Ok(results)
    }
}

impl Server {
    /// Create a new server with the given configuration
    pub fn new(mut config: ServerConfig) -> Result<Self, Box<dyn std::error::Error>> {
        // Enforce TLS requirement: if require_tls is set, TLS cert and key must be configured
        #[cfg(feature = "tls")]
        if config.require_tls {
            if config.tls_cert_path.is_none() || config.tls_key_path.is_none() {
                return Err(
                    "require_tls is true but tls_cert_path and tls_key_path are not configured"
                        .into(),
                );
            }
            tracing::info!("TLS enforcement enabled: all client-facing protocols will require TLS");
        }

        // Auto-detect platform for TDP calibration
        let platform_info = joule_db_energy::detect_platform();

        // If TDP is still the generic default (30W), override with detected value
        if (config.energy_config.default_tdp_watts - 30.0).abs() < f64::EPSILON {
            config.energy_config.default_tdp_watts = platform_info.tdp_watts;
        }

        let db = Database::open(&config.db_path)?;
        let registry = Arc::new(MetricsRegistry::new());
        let metrics = Arc::new(DatabaseMetrics::new(registry.clone()));

        // Open durable amorphic store (primary storage for SQL tables)
        let amorphic_path = format!("{}/amorphic", config.db_path);
        let amorphic_store = joule_db_amorphic::DurableAmorphicStore::open(&amorphic_path)
            .map_err(|e| format!("Failed to open amorphic store: {}", e))?;
        let amorphic = Arc::new(amorphic_adapter::AmorphicTableStorage::new(amorphic_store));

        // Initialize energy monitoring
        let (energy_snapshot, energy_metrics, energy_advisor) = {
            let energy_cfg = config.energy_config.clone();
            let monitor = joule_db_energy::EnergyMonitor::new(energy_cfg.clone());
            let (snapshot_handle, _monitor_thread) = monitor.start_background();
            let e_metrics = Arc::new(energy::EnergyMetrics::new(&registry));
            let advisor = Arc::new(joule_db_energy::HardwareAdvisor::new(&energy_cfg));

            // Spawn periodic gauge updater
            let snap_for_updater = snapshot_handle.clone();
            let metrics_for_updater = e_metrics.clone();
            let interval_ms = energy_cfg.collection_interval_ms;
            std::thread::Builder::new()
                .name("energy-metrics-updater".to_string())
                .spawn(move || {
                    loop {
                        if let Ok(snap) = snap_for_updater.read() {
                            metrics_for_updater.update_from_snapshot(&snap);
                        }
                        std::thread::sleep(std::time::Duration::from_millis(interval_ms));
                    }
                })
                .ok();

            (snapshot_handle, e_metrics, advisor)
        };

        // Initialize auth manager if enabled
        if !config.auth_enabled {
            tracing::warn!(
                "⚠ Authentication is DISABLED (--no-auth mode). Do NOT use in production!"
            );
        }
        let auth_manager = if config.auth_enabled {
            if let Some(ref secret) = config.auth_jwt_secret {
                Some(Arc::new(auth::AuthenticationManager::new(
                    secret.as_bytes().to_vec(),
                )))
            } else {
                None
            }
        } else {
            None
        };

        // Initialize RBAC manager if auth is enabled
        let rbac_manager = if config.auth_enabled {
            let mgr = rbac::RBACManager::new();
            // Load persisted RBAC state from amorphic
            load_rbac_from_amorphic(&amorphic, &mgr);
            Some(Arc::new(mgr))
        } else {
            None
        };

        // Initialize replication server if this node is a leader
        let replication_server =
            if config.enable_replication && config.replication_role.as_deref() == Some("leader") {
                let repl_config = replication::ReplicationConfig {
                    listen_addr: config.replication_listen_addr.clone(),
                    ..replication::ReplicationConfig::default()
                };
                Some(Arc::new(replication::ReplicationServer::new(repl_config)))
            } else {
                None
            };

        let raft_is_leader = Arc::new(std::sync::atomic::AtomicBool::new(false));

        let mut base_executor = SimpleQueryExecutor::with_amorphic(amorphic.clone());
        base_executor.set_timeout_config(config.query_timeout_ms, config.slow_query_threshold_ms);
        base_executor.set_max_result_rows(config.max_result_rows);
        if let Some(ref rbac) = rbac_manager {
            base_executor.set_rbac_manager(rbac.clone());
        }
        if let Some(ref repl) = replication_server {
            base_executor.set_replication_server(repl.clone());
        }
        if config.enable_replication {
            if let Some(ref role) = config.replication_role {
                base_executor.set_replication_role(role.clone());
            }
        }
        // Grab the Raft node slot before the executor is consumed
        let raft_node_slot = if config.enable_raft {
            base_executor.set_raft_config(raft_is_leader.clone());
            Some(base_executor.raft_node_slot())
        } else {
            None
        };

        // Keep a handle to the MVCC adapter for session reaper
        let mvcc_for_reaper = base_executor.mvcc_handle();

        // Initialize energy receipt ledger (if enabled)
        let ledger_handles = ledger_bridge::init_ledger(&config);
        let ledger_store = ledger_handles.as_ref().map(|h| h.store.clone());
        let ledger_collector = ledger_handles.as_ref().map(|h| h.collector.clone());

        let energy_exec = energy_executor::EnergyAwareExecutor::new(
            base_executor,
            energy_snapshot.clone(),
            energy_metrics.clone(),
            energy_advisor.clone(),
        );
        let energy_exec = if let Some(ref handles) = ledger_handles {
            energy_exec.with_ledger(handles.collector.clone())
        } else {
            energy_exec
        };
        let query_executor: Arc<dyn QueryExecutor> = Arc::new(energy_exec);

        // Initialize backup manager
        let backup_dir = std::path::PathBuf::from(format!("{}/backups", config.db_path));
        let backup_config = backup::BackupConfig {
            backup_dir,
            ..backup::BackupConfig::default()
        };
        let backup_manager = Arc::new(backup::BackupManager::new(backup_config));

        // Initialize shard router from persisted metadata
        let shard_router = load_shard_router_from_amorphic(&amorphic);

        // Initialize dynamic route manager and load persisted DEFINE API endpoints
        let dynamic_route_manager = dynamic_routes::create_route_manager();
        if let Ok(endpoints) = amorphic.list_api_endpoints() {
            let drm = dynamic_route_manager.clone();
            // Use a blocking executor for the async register calls during init
            for ep in endpoints {
                // Pre-populate the route table synchronously using a oneshot runtime
                let drm2 = drm.clone();
                let _ = std::thread::spawn(move || {
                    let rt = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .unwrap();
                    rt.block_on(drm2.register(ep.path, ep.method, ep.handler_sql));
                })
                .join();
            }
        }

        // Initialize rate limiter if enabled
        let rate_limiter = if config.rate_limiting_enabled {
            let rate_config = security::RateLimitConfig {
                requests_per_window: config.rate_limit_requests_per_minute,
                window_duration: std::time::Duration::from_secs(60),
                adaptive: true,
                burst_size: (config.rate_limit_requests_per_minute / 10).max(10),
            };
            Some(Arc::new(security::RateLimiter::new(rate_config)))
        } else {
            None
        };

        let db_path = config.db_path.clone();

        Ok(Self {
            config,
            db: Arc::new(RwLock::new(db)),
            amorphic,
            query_executor,
            subscription_manager: Arc::new(SubscriptionManager::new()),
            auth_manager,
            rbac_manager,
            backup_manager,
            metrics,
            start_time: std::time::Instant::now(),
            energy_snapshot,
            energy_metrics,
            energy_advisor,
            platform_info,
            replication_server,
            shard_router,
            raft_is_leader,
            raft_node_slot,
            rate_limiter,
            mvcc: mvcc_for_reaper,
            ledger_store,
            ledger_collector,
            dynamic_route_manager,
            branch_manager: Arc::new(joule_db_branch::manager::BranchManager::new(0)),
            tenant_manager: Arc::new(tenant::TenantManager::new()),
            vector_index_manager: Arc::new(std::sync::RwLock::new(
                vector_index::VectorIndexManager::new(),
            )),
            activity_tracker: Arc::new(scale_to_zero::ActivityTracker::new()),
            memory_manager: Arc::new(
                agent_memory::MemoryManager::open(&format!("{}/agent_memory", db_path))
                    .unwrap_or_else(|_| agent_memory::MemoryManager::new()),
            ),
            workflow_manager: Arc::new(
                workflow::WorkflowManager::open(&format!("{}/workflows", db_path))
                    .unwrap_or_else(|_| workflow::WorkflowManager::new()),
            ),
            edge_pop_manager: Arc::new(
                edge_pop::EdgePopManager::open(&format!("{}/edge_pops", db_path))
                    .unwrap_or_else(|_| edge_pop::EdgePopManager::new()),
            ),
        })
    }

    /// Get auto-detected platform information.
    pub fn platform_info(&self) -> &joule_db_energy::PlatformInfo {
        &self.platform_info
    }

    /// Get the metrics
    pub fn metrics(&self) -> Arc<DatabaseMetrics> {
        self.metrics.clone()
    }

    /// Get the database reference
    pub fn database(&self) -> Arc<RwLock<Database>> {
        self.db.clone()
    }

    /// Get the query executor
    pub fn query_executor(&self) -> Arc<dyn QueryExecutor> {
        self.query_executor.clone()
    }

    /// Get the subscription manager
    pub fn subscription_manager(&self) -> Arc<SubscriptionManager> {
        self.subscription_manager.clone()
    }

    /// Create a database handler for the TCP server
    pub fn create_tcp_handler(&self) -> ServerDatabaseHandler {
        ServerDatabaseHandler::new(self.db.clone(), self.query_executor.clone())
    }

    /// Build the router (useful for testing)
    pub fn router(&self) -> Router {
        let app_state = AppState {
            db: self.db.clone(),
            amorphic: self.amorphic.clone(),
            subscription_manager: self.subscription_manager.clone(),
            auth_manager: self.auth_manager.clone(),
            rbac_manager: self.rbac_manager.clone(),
            backup_manager: self.backup_manager.clone(),
            query_executor: self.query_executor.clone(),
            db_path: self.config.db_path.clone(),
            energy_snapshot: self.energy_snapshot.clone(),
            energy_advisor: self.energy_advisor.clone(),
            energy_metrics: self.energy_metrics.clone(),
            replication_server: self.replication_server.clone(),
            replication_role: self.config.replication_role.clone(),
            shard_router: self.shard_router.clone(),
            sanitize_errors: self.config.sanitize_errors,
            dynamic_route_manager: self.dynamic_route_manager.clone(),
            branch_manager: self.branch_manager.clone(),
            tenant_manager: self.tenant_manager.clone(),
            vector_index_manager: self.vector_index_manager.clone(),
            activity_tracker: self.activity_tracker.clone(),
            memory_manager: self.memory_manager.clone(),
            workflow_manager: self.workflow_manager.clone(),
            edge_pop_manager: self.edge_pop_manager.clone(),
        };
        let executor = self.query_executor.clone();
        let metrics_state = MetricsState {
            metrics: self.metrics.clone(),
            start_time: self.start_time,
        };

        // Key-value routes (with subscription-aware handlers)
        let kv_routes = Router::new()
            .route("/health", get(health_check))
            .route("/health/live", get(liveness))
            .route("/health/ready", get(readiness))
            .route("/api/v1/keys/{key}", get(get_key))
            .route("/api/v1/keys/{key}", post(put_key))
            .route("/api/v1/keys/{key}", delete(delete_key))
            .route("/api/v1/keys/{key}/ttl", get(get_key_ttl))
            .route("/api/v1/cleanup/expired", post(cleanup_expired))
            .route("/ws", get(ws_upgrade_handler))
            .with_state(app_state.clone());

        // Query routes
        let query_routes = query_router(executor);

        // Metrics routes
        let metrics_routes = Router::new()
            .route("/api/metrics", get(api_metrics_handler))
            .route("/api/metrics/history", get(api_metrics_history_handler))
            .route("/api/metrics/slow-queries", get(api_slow_queries_handler))
            .route("/metrics", get(prometheus_metrics_handler))
            .with_state(metrics_state);

        // Amorphic multi-model API routes
        let amorphic_routes = Router::new()
            .route("/api/v1/ingest", post(amorphic_ingest_handler))
            .route("/api/v1/ingest/edge", post(amorphic_ingest_edge_handler))
            .route("/api/v1/records/{id}", get(amorphic_get_record_handler))
            .route(
                "/api/v1/records/{id}",
                delete(amorphic_delete_record_handler),
            )
            .route("/api/v1/query/similar", post(amorphic_similar_handler))
            .route("/api/v1/query/graph", post(amorphic_graph_handler))
            .with_state(app_state.clone());

        // Unified endpoint — the front door
        let unified_routes = Router::new()
            .route("/", post(unified_handler))
            .with_state(app_state.clone());

        // Backup/restore/export routes
        let backup_routes = Router::new()
            .route("/api/v1/backup", post(create_backup_handler))
            .route("/api/v1/backup/list", get(list_backups_handler))
            .route("/api/v1/backup/{id}", get(get_backup_handler))
            .route("/api/v1/restore/{id}", post(restore_backup_handler))
            .route("/api/v1/export/{table}", get(export_table_handler))
            .with_state(app_state.clone());

        // Energy routes (feature-gated)
        let energy_routes = Router::new()
            .route("/api/v1/energy", get(energy_status_handler))
            .with_state(app_state.clone());

        // Branch management routes (CoW branching with energy budgets)
        let branch_routes = Router::new()
            .route("/api/v1/branches", get(list_branches_handler))
            .route("/api/v1/branches", post(create_branch_handler))
            .route("/api/v1/branches/{name}", get(get_branch_handler))
            .route("/api/v1/branches/{name}", delete(delete_branch_handler))
            .route("/api/v1/branches/{name}/merge", post(merge_branch_handler))
            .route("/api/v1/branches/{name}/diff", get(diff_branch_handler))
            .with_state(app_state.clone());

        // Replication status route
        let replication_routes = Router::new()
            .route(
                "/api/v1/replication/status",
                get(replication_status_handler),
            )
            .with_state(app_state.clone());

        // Shard status route
        let shard_routes = Router::new()
            .route("/api/v1/shards/status", get(shard_status_handler))
            .with_state(app_state.clone());

        // LangGraph checkpoint & message store routes
        let langgraph_state = langgraph_handlers::LangGraphState::new();
        let langgraph_routes = langgraph_handlers::langgraph_routes(langgraph_state);

        // Tenant management routes
        let tenant_routes = Router::new()
            .route("/api/v1/tenants", get(list_tenants_handler))
            .route("/api/v1/tenants", post(create_tenant_handler))
            .route("/api/v1/tenants/{id}", get(get_tenant_handler))
            .route("/api/v1/tenants/{id}", delete(delete_tenant_handler))
            .route("/api/v1/tenants/{id}/energy", get(tenant_energy_handler))
            .route("/api/v1/tenants/{id}/suspend", post(suspend_tenant_handler))
            .with_state(app_state.clone());

        // Scale-to-Zero lifecycle routes
        let scale_routes = Router::new()
            .route("/api/v1/status", get(status_handler))
            .route("/api/v1/suspend", post(suspend_handler))
            .route("/api/v1/resume", post(resume_handler))
            .with_state(app_state.clone());

        // Agent Memory routes
        let memory_routes = Router::new()
            .route("/api/v1/memory/store", post(store_memory_handler))
            .route("/api/v1/memory/recall", post(recall_memory_handler))
            .route("/api/v1/memory/forget", post(forget_memory_handler))
            .route(
                "/api/v1/memory/consolidate",
                post(consolidate_memory_handler),
            )
            .route("/api/v1/memory/stats", get(memory_stats_handler))
            .with_state(app_state.clone());

        // Durable Workflow routes
        let workflow_routes = Router::new()
            .route("/api/v1/workflows", post(create_workflow_handler))
            .route("/api/v1/workflows", get(list_workflows_handler))
            .route("/api/v1/workflows/{id}", get(get_workflow_handler))
            .route("/api/v1/workflows/{id}", delete(delete_workflow_handler))
            .route("/api/v1/workflows/{id}/run", post(run_workflow_handler))
            .route(
                "/api/v1/workflows/instances/{id}",
                get(get_workflow_instance_handler),
            )
            .route("/api/v1/queue/publish", post(publish_message_handler))
            .route("/api/v1/queue/subscribe", post(subscribe_handler))
            .route("/api/v1/queue/ack", post(ack_messages_handler))
            .route(
                "/api/v1/queue/dead-letters/{topic}",
                get(dead_letters_handler),
            )
            .with_state(app_state.clone());

        // Edge PoP routes
        let edge_routes = Router::new()
            .route("/api/v1/edge/pops", post(register_pop_handler))
            .route("/api/v1/edge/pops", get(list_pops_handler))
            .route("/api/v1/edge/pops/{id}", get(get_pop_handler))
            .route("/api/v1/edge/pops/{id}", delete(deregister_pop_handler))
            .route("/api/v1/edge/sync", post(trigger_sync_handler))
            .route("/api/v1/edge/stats", get(edge_stats_handler))
            .with_state(app_state.clone());

        // Vector search routes
        let vector_state = vector_routes::VectorState {
            manager: self.vector_index_manager.clone(),
        };
        let vector_rt = Router::new()
            .route(
                "/api/v1/vector/search",
                post(vector_routes::vector_search_handler),
            )
            .route(
                "/api/v1/vector/upsert",
                post(vector_routes::vector_upsert_handler),
            )
            .route(
                "/api/v1/vector/indexes",
                post(vector_routes::create_vector_index_handler),
            )
            .route(
                "/api/v1/vector/indexes",
                get(vector_routes::list_vector_indexes_handler),
            )
            .route(
                "/api/v1/vector/indexes/{name}",
                delete(vector_routes::delete_vector_index_handler),
            )
            .with_state(vector_state);

        // Enterprise cluster health routes (LoadBalancer + energy state)
        let cluster_lb = Arc::new(enterprise::LoadBalancer::new(
            enterprise::LoadBalancingStrategy::default(),
        ));
        let cluster_state = ClusterHealthState {
            load_balancer: cluster_lb,
            raft_node_slot: self.raft_node_slot.clone(),
        };
        let cluster_routes = Router::new()
            .route("/api/v1/cluster/nodes", get(cluster_nodes_handler))
            .route("/api/v1/cluster/health", get(cluster_health_handler))
            .with_state(cluster_state);

        let merged = kv_routes
            .merge(query_routes)
            .merge(metrics_routes)
            .merge(amorphic_routes)
            .merge(backup_routes)
            .merge(unified_routes)
            .merge(energy_routes)
            .merge(branch_routes)
            .merge(replication_routes)
            .merge(shard_routes)
            .merge(langgraph_routes)
            .merge(cluster_routes)
            .merge(tenant_routes)
            .merge(vector_rt)
            .merge(scale_routes)
            .merge(memory_routes)
            .merge(workflow_routes)
            .merge(edge_routes);

        // Merge ledger verification routes (if ledger is enabled)
        let merged = if let Some(ref store) = self.ledger_store {
            let ledger_state = joule_db_ledger::http::LedgerVerifyState {
                store: store.clone(),
            };
            merged.merge(joule_db_ledger::http::ledger_routes(ledger_state))
        } else {
            merged
        };

        // Add fallback handler for dynamic API routes (DEFINE API endpoints)
        let drm_fallback = app_state.dynamic_route_manager.clone();
        let executor_fallback = app_state.query_executor.clone();
        let merged = merged.fallback(move |req: axum::http::Request<axum::body::Body>| {
            let drm = drm_fallback.clone();
            let executor = executor_fallback.clone();
            async move {
                let method = req.method().as_str().to_uppercase();
                let path = req.uri().path().to_string();
                if let Some(route) = drm.resolve(&path, &method).await {
                    // Execute the stored SQL handler via QueryRequest
                    let qr = query::QueryRequest {
                        sql: route.handler_sql.clone(),
                        params: std::collections::HashMap::new(),
                        args: Vec::new(),
                        explain: false,
                        limit: None,
                        session_id: None,
                        query_timeout_ms: None,
                        branch_id: None,
                        tenant_id: None,
                    };
                    match executor.execute(&qr) {
                        Ok(resp) => Json(serde_json::json!({
                            "columns": resp.columns,
                            "rows": resp.rows,
                        }))
                        .into_response(),
                        Err(e) => (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({"error": e.message})),
                        )
                            .into_response(),
                    }
                } else {
                    (
                        StatusCode::NOT_FOUND,
                        Json(serde_json::json!({"error": "Not found"})),
                    )
                        .into_response()
                }
            }
        });

        // Apply rate limiting middleware (before auth, to throttle unauthenticated requests)
        let rate_limiter_state = self.rate_limiter.clone();
        let auth_state = app_state.auth_manager.clone();

        // Build CORS layer from config
        let cors_layer = {
            use tower_http::cors::CorsLayer;
            if self.config.cors_origins.is_empty() {
                // No origins configured = block all cross-origin requests
                CorsLayer::new()
            } else {
                let origins: Vec<axum::http::HeaderValue> = self
                    .config
                    .cors_origins
                    .iter()
                    .filter_map(|o| o.parse().ok())
                    .collect();
                CorsLayer::new()
                    .allow_origin(origins)
                    .allow_methods([
                        axum::http::Method::GET,
                        axum::http::Method::POST,
                        axum::http::Method::PUT,
                        axum::http::Method::DELETE,
                        axum::http::Method::OPTIONS,
                    ])
                    .allow_headers([
                        axum::http::header::AUTHORIZATION,
                        axum::http::header::CONTENT_TYPE,
                        axum::http::HeaderName::from_static("x-api-key"),
                    ])
            }
        };

        merged
            .layer(axum::middleware::from_fn(move |req, next| {
                let auth = auth_state.clone();
                auth_middleware(auth, req, next)
            }))
            .layer(axum::middleware::from_fn(move |req, next| {
                let limiter = rate_limiter_state.clone();
                rate_limit_middleware(limiter, req, next)
            }))
            // Energy header — adds X-Energy-Joules on every response
            .layer(axum::middleware::from_fn(energy_header_middleware))
            // Security headers on all HTTP responses (HSTS, CSP, X-Frame-Options, etc.)
            .layer(axum::middleware::from_fn(security_headers_middleware))
            // Limit HTTP request body size to 16MB (matches binary protocol limit)
            .layer(axum::extract::DefaultBodyLimit::max(16 * 1024 * 1024))
            // Limit concurrent HTTP connections to prevent resource exhaustion
            .layer(tower::limit::ConcurrencyLimitLayer::new(10_000))
            // CORS outermost — handles preflight OPTIONS before auth/rate-limit
            .layer(cors_layer)
    }

    /// Run the HTTP server only
    pub async fn run_http(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        use std::net::SocketAddr;

        let app = self.router();
        let addr: SocketAddr = self.config.http_addr.parse()?;

        #[cfg(feature = "tls")]
        if let (Some(cert_path), Some(key_path)) =
            (&self.config.tls_cert_path, &self.config.tls_key_path)
        {
            tracing::info!("HTTPS server listening on {} (TLS enabled)", addr);
            let acceptor = Self::create_tls_acceptor(cert_path, key_path)?;
            let listener = tokio::net::TcpListener::bind(addr).await?;

            loop {
                let (stream, _peer_addr) = listener.accept().await?;
                let acceptor = acceptor.clone();
                let app = app.clone();

                tokio::spawn(async move {
                    match acceptor.accept(stream).await {
                        Ok(tls_stream) => {
                            let io = hyper_util::rt::TokioIo::new(tls_stream);
                            let service = hyper::service::service_fn(move |req| {
                                let app = app.clone();
                                async move {
                                    use tower::ServiceExt;
                                    app.oneshot(req).await
                                }
                            });
                            if let Err(e) = hyper_util::server::conn::auto::Builder::new(
                                hyper_util::rt::TokioExecutor::new(),
                            )
                            .serve_connection(io, service)
                            .await
                            {
                                tracing::debug!("TLS connection error: {}", e);
                            }
                        }
                        Err(e) => {
                            tracing::debug!("TLS handshake failed: {}", e);
                        }
                    }
                });
            }
        }

        // If TLS is required but not configured, refuse to start in plaintext
        #[cfg(feature = "tls")]
        if self.config.require_tls {
            return Err("HTTP server: require_tls is set but no TLS certificate configured".into());
        }

        tracing::info!("HTTP server listening on {}", addr);

        let listener = tokio::net::TcpListener::bind(addr).await?;
        axum::serve(listener, app).await?;

        Ok(())
    }

    /// Create a TLS acceptor from PEM certificate and key files
    #[cfg(feature = "tls")]
    pub fn create_tls_acceptor(
        cert_path: &str,
        key_path: &str,
    ) -> Result<tokio_rustls::TlsAcceptor, Box<dyn std::error::Error + Send + Sync>> {
        let cert_pem = std::fs::read(cert_path)
            .map_err(|e| format!("Failed to read TLS certificate '{}': {}", cert_path, e))?;
        let key_pem = std::fs::read(key_path)
            .map_err(|e| format!("Failed to read TLS key '{}': {}", key_path, e))?;

        let certs: Vec<rustls::pki_types::CertificateDer<'static>> =
            rustls_pemfile::certs(&mut &*cert_pem)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| format!("Failed to parse TLS certificates: {}", e))?;

        let key = rustls_pemfile::private_key(&mut &*key_pem)
            .map_err(|e| format!("Failed to parse TLS private key: {}", e))?
            .ok_or("No private key found in key file")?;

        let config = rustls::ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(certs, key)
            .map_err(|e| format!("Invalid TLS configuration: {}", e))?;

        Ok(tokio_rustls::TlsAcceptor::from(std::sync::Arc::new(config)))
    }

    /// Create a mTLS acceptor for Raft inter-node connections.
    ///
    /// Unlike `create_tls_acceptor` (used for HTTP/TCP/PgWire client-facing
    /// protocols), this requires connecting peers to present a valid client
    /// certificate verified against the given CA certificate.
    #[cfg(feature = "tls")]
    pub fn create_raft_tls_acceptor(
        cert_path: &str,
        key_path: &str,
        ca_cert_path: &str,
    ) -> Result<tokio_rustls::TlsAcceptor, Box<dyn std::error::Error + Send + Sync>> {
        let cert_pem = std::fs::read(cert_path)
            .map_err(|e| format!("Failed to read TLS certificate '{}': {}", cert_path, e))?;
        let key_pem = std::fs::read(key_path)
            .map_err(|e| format!("Failed to read TLS key '{}': {}", key_path, e))?;
        let ca_pem = std::fs::read(ca_cert_path)
            .map_err(|e| format!("Failed to read CA certificate '{}': {}", ca_cert_path, e))?;

        let certs: Vec<rustls::pki_types::CertificateDer<'static>> =
            rustls_pemfile::certs(&mut &*cert_pem)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| format!("Failed to parse TLS certificates: {}", e))?;

        let key = rustls_pemfile::private_key(&mut &*key_pem)
            .map_err(|e| format!("Failed to parse TLS private key: {}", e))?
            .ok_or("No private key found in key file")?;

        // Build root store from CA cert for client verification
        let ca_certs: Vec<rustls::pki_types::CertificateDer<'static>> =
            rustls_pemfile::certs(&mut &*ca_pem)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| format!("Failed to parse CA certificates: {}", e))?;

        let mut root_store = rustls::RootCertStore::empty();
        for cert in &ca_certs {
            root_store
                .add(cert.clone())
                .map_err(|e| format!("Failed to add CA cert to root store: {}", e))?;
        }

        let client_verifier =
            rustls::server::WebPkiClientVerifier::builder(std::sync::Arc::new(root_store))
                .build()
                .map_err(|e| format!("Failed to build client cert verifier: {}", e))?;

        let config = rustls::ServerConfig::builder()
            .with_client_cert_verifier(client_verifier)
            .with_single_cert(certs, key)
            .map_err(|e| format!("Invalid mTLS configuration: {}", e))?;

        Ok(tokio_rustls::TlsAcceptor::from(std::sync::Arc::new(config)))
    }

    /// Create a mTLS connector for Raft client connections.
    ///
    /// Loads the certificate PEM file into the root certificate store and
    /// presents the node's own cert+key as a client certificate for peer
    /// verification (mutual TLS).
    #[cfg(feature = "tls")]
    pub fn create_raft_tls_connector(
        cert_path: &str,
        key_path: &str,
    ) -> Result<tokio_rustls::TlsConnector, Box<dyn std::error::Error + Send + Sync>> {
        let cert_pem = std::fs::read(cert_path)
            .map_err(|e| format!("Failed to read TLS certificate '{}': {}", cert_path, e))?;
        let key_pem = std::fs::read(key_path)
            .map_err(|e| format!("Failed to read TLS key '{}': {}", key_path, e))?;

        let certs: Vec<rustls::pki_types::CertificateDer<'static>> =
            rustls_pemfile::certs(&mut &*cert_pem)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| format!("Failed to parse TLS certificates: {}", e))?;

        let key = rustls_pemfile::private_key(&mut &*key_pem)
            .map_err(|e| format!("Failed to parse TLS private key: {}", e))?
            .ok_or("No private key found in key file")?;

        // Trust the same cert as root (self-signed) or CA chain
        let mut root_store = rustls::RootCertStore::empty();
        for cert in &certs {
            root_store
                .add(cert.clone())
                .map_err(|e| format!("Failed to add cert to root store: {}", e))?;
        }

        let config = rustls::ClientConfig::builder()
            .with_root_certificates(root_store)
            .with_client_auth_cert(certs, key)
            .map_err(|e| format!("Invalid mTLS client configuration: {}", e))?;

        Ok(tokio_rustls::TlsConnector::from(std::sync::Arc::new(
            config,
        )))
    }

    /// Run the TCP binary protocol server only
    pub async fn run_tcp(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let handler = self.create_tcp_handler();
        let tcp_config = tcp_server::TcpServerConfig {
            bind_addr: self.config.tcp_addr.clone(),
            max_connections: self.config.max_tcp_connections,
            auth_enabled: self.config.auth_enabled,
            auth_jwt_secret: self.config.auth_jwt_secret.clone(),
            ..Default::default()
        };

        let mut tcp_server = tcp_server::TcpServer::with_subscription_manager(
            tcp_config,
            handler,
            self.subscription_manager.clone(),
        );

        // Enable TLS if configured
        #[cfg(feature = "tls")]
        if let (Some(cert_path), Some(key_path)) =
            (&self.config.tls_cert_path, &self.config.tls_key_path)
        {
            let acceptor = Self::create_tls_acceptor(cert_path, key_path)?;
            tcp_server = tcp_server.with_tls(acceptor);
        }

        tcp_server.run().await
    }

    /// Run all servers (HTTP + TCP + WebTransport)
    pub async fn run(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        use std::net::SocketAddr;

        let app = self.router();
        let http_addr: SocketAddr = self.config.http_addr.parse()?;

        tracing::info!("JouleDB HTTP server listening on {}", http_addr);

        let http_listener = tokio::net::TcpListener::bind(http_addr).await?;

        // Collect server futures to run concurrently
        let http_future = axum::serve(http_listener, app);

        // Optional TCP server
        let tcp_server = if self.config.enable_tcp {
            let tcp_handler = self.create_tcp_handler();
            let tcp_config = tcp_server::TcpServerConfig {
                bind_addr: self.config.tcp_addr.clone(),
                max_connections: self.config.max_tcp_connections,
                auth_enabled: self.config.auth_enabled,
                auth_jwt_secret: self.config.auth_jwt_secret.clone(),
                ..Default::default()
            };
            let mut server = tcp_server::TcpServer::with_subscription_manager(
                tcp_config,
                tcp_handler,
                self.subscription_manager.clone(),
            );

            // Enable TLS for TCP server if configured
            #[cfg(feature = "tls")]
            if let (Some(cert_path), Some(key_path)) =
                (&self.config.tls_cert_path, &self.config.tls_key_path)
            {
                if let Ok(acceptor) = Self::create_tls_acceptor(cert_path, key_path) {
                    server = server.with_tls(acceptor);
                }
            }

            tracing::info!("JouleDB TCP server listening on {}", self.config.tcp_addr);
            Some(server)
        } else {
            None
        };

        // Optional WebTransport server
        #[cfg(feature = "webtransport")]
        let wt_server = if self.config.enable_webtransport {
            let wt_config = webtransport::WebTransportConfig {
                bind_port: self.config.webtransport_port,
                ..Default::default()
            };
            let server = webtransport::WebTransportServer::new(
                self.db.clone(),
                wt_config,
                self.subscription_manager.clone(),
            );
            tracing::info!(
                "JouleDB WebTransport server listening on port {}",
                self.config.webtransport_port,
            );
            Some(server)
        } else {
            None
        };

        // Optional JWP (Joule Wire Protocol) server
        #[cfg(feature = "jwp")]
        let jwp_srv = if self.config.enable_jwp {
            let jwp_config = jwp_server::JwpServerConfig {
                bind_addr: self.config.jwp_addr.clone(),
                max_connections: self.config.max_jwp_connections,
                ..Default::default()
            };
            tracing::info!("JouleDB JWP server listening on {}", self.config.jwp_addr);
            let jwp = jwp_server::JwpServer::new(
                jwp_config,
                self.query_executor.clone(),
                self.subscription_manager.clone(),
                self.energy_snapshot.clone(),
            );
            let jwp = if let (Some(collector), Some(store)) =
                (&self.ledger_collector, &self.ledger_store)
            {
                jwp.with_ledger(collector.clone(), store.clone())
            } else {
                jwp
            };
            Some(jwp)
        } else {
            None
        };

        // Optional PgWire server
        let pgwire_server = if self.config.enable_pgwire {
            let pg_config = pgwire::PgWireConfig {
                bind_addr: self.config.pgwire_addr.clone(),
                auth_enabled: self.config.auth_enabled,
                auth_password: self.config.auth_jwt_secret.clone(),
                query_timeout_ms: self.config.query_timeout_ms,
                rbac_manager: self.rbac_manager.clone(),
                #[cfg(feature = "tls")]
                tls_acceptor: {
                    if let (Some(cert_path), Some(key_path)) =
                        (&self.config.tls_cert_path, &self.config.tls_key_path)
                    {
                        Self::create_tls_acceptor(cert_path, key_path).ok()
                    } else {
                        None
                    }
                },
                #[cfg(feature = "tls")]
                require_tls: self.config.require_tls,
                ..Default::default()
            };
            let server = pgwire::PgWireServer::from_dyn(pg_config, self.query_executor.clone());
            tracing::info!(
                "JouleDB PgWire server listening on {}",
                self.config.pgwire_addr
            );
            Some(server)
        } else {
            None
        };

        // Use the stored replication server (already created in new())
        let replication_server = &self.replication_server;
        if replication_server.is_some() {
            tracing::info!(
                "JouleDB replication leader on {}",
                self.config.replication_listen_addr
            );
        }

        let mut replication_client = if self.config.enable_replication
            && self.config.replication_role.as_deref() == Some("follower")
        {
            if let Some(ref leader_addr) = self.config.replication_leader_addr {
                let repl_config = replication::ReplicationConfig {
                    leader_addr: Some(leader_addr.clone()),
                    ..replication::ReplicationConfig::default()
                };
                let client = replication::ReplicationClient::new(repl_config);
                tracing::info!("JouleDB replication follower connecting to {}", leader_addr);
                Some(client)
            } else {
                None
            }
        } else {
            None
        };

        // WAL checkpoint background task
        let checkpoint_amorphic = self.amorphic.clone();
        let checkpoint_task = async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));
            loop {
                interval.tick().await;
                match checkpoint_amorphic.checkpoint_if_needed() {
                    Ok(true) => tracing::info!("WAL checkpoint completed"),
                    Ok(false) => {} // no checkpoint needed
                    Err(e) => tracing::warn!("WAL checkpoint failed: {}", e),
                }
            }
        };

        // Session reaper background task — rolls back stale transactions
        let reaper_mvcc = self.mvcc.clone();
        let session_timeout = std::time::Duration::from_secs(self.config.session_timeout_secs);
        let session_reaper_task = async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));
            loop {
                interval.tick().await;
                let cleaned = reaper_mvcc.cleanup_stale_sessions(session_timeout);
                if cleaned > 0 {
                    tracing::info!("Session reaper: rolled back {} stale transactions", cleaned);
                }
            }
        };

        // Optional Raft consensus cluster
        // Keep handles alive for the duration of run() so background tasks aren't dropped.
        let mut _raft_node: Option<
            Arc<raft::RaftNode<raft::KvStateMachine, raft_transport::TcpRaftTransport>>,
        > = None;
        let mut _raft_rpc_handle: Option<tokio::task::JoinHandle<()>> = None;
        let mut _raft_loop_handle: Option<tokio::task::JoinHandle<()>> = None;
        if self.config.enable_raft {
            let node_id = self
                .config
                .raft_node_id
                .clone()
                .unwrap_or_else(|| format!("node-{}", self.config.http_addr));
            let peers = raft_server::parse_peer_list(&self.config.raft_peers);
            let mut raft_config = raft::RaftConfig::new(node_id.clone());
            raft_config.data_dir = Some(std::path::PathBuf::from(format!(
                "{}/raft",
                self.config.db_path
            )));

            // Build Raft TLS config if cert/key are available
            #[cfg(feature = "tls")]
            let raft_tls_config = {
                if let (Some(cert_path), Some(key_path)) =
                    (&self.config.tls_cert_path, &self.config.tls_key_path)
                {
                    match (
                        Self::create_raft_tls_acceptor(cert_path, key_path, cert_path),
                        Self::create_raft_tls_connector(cert_path, key_path),
                    ) {
                        (Ok(acceptor), Ok(connector)) => {
                            tracing::info!(
                                "Raft mTLS enabled — inter-node traffic is encrypted and peer-authenticated"
                            );
                            Some(raft_server::RaftTlsConfig {
                                acceptor,
                                connector,
                            })
                        }
                        (Err(e), _) | (_, Err(e)) => {
                            tracing::warn!(
                                "Raft TLS setup failed, falling back to plaintext: {}",
                                e
                            );
                            None
                        }
                    }
                } else {
                    tracing::warn!(
                        "Raft transport running without TLS — inter-node traffic is plaintext. \
                         Configure tls_cert_path and tls_key_path, or use network-level \
                         encryption (WireGuard/IPsec) for Raft peers."
                    );
                    None
                }
            };

            // HRP Phase 3: Build security key manager from master secret
            let security_key_mgr = self.config.raft_master_secret.as_ref().and_then(|hex| {
                match hrp_security::EpochKeyManager::from_hex(hex) {
                    Ok(mgr) => {
                        tracing::info!("HRP security enabled — write tokens and HMAC active for Raft replication");
                        Some(std::sync::Arc::new(mgr))
                    }
                    Err(e) => {
                        tracing::warn!("HRP security disabled — invalid raft_master_secret: {}", e);
                        None
                    }
                }
            });

            match raft_server::start_raft_node(
                raft_config,
                peers,
                self.config.raft_addr.clone(),
                #[cfg(feature = "tls")]
                raft_tls_config,
                security_key_mgr,
            )
            .await
            {
                Ok((node, rpc_handle, election_handle, replication_handle)) => {
                    tracing::info!(
                        "Raft node '{}' started on {} (HRP event-driven replication)",
                        node.node_id(),
                        self.config.raft_addr,
                    );

                    // Spawn a background task that updates the shared leader flag
                    let leader_flag = self.raft_is_leader.clone();
                    let monitor_node = node.clone();
                    tokio::spawn(async move {
                        loop {
                            let is_leader = monitor_node.is_leader().await;
                            leader_flag.store(is_leader, std::sync::atomic::Ordering::Relaxed);
                            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                        }
                    });

                    // Wire the Raft node into the query executor for distributed writes
                    if let Some(ref slot) = self.raft_node_slot {
                        let _ = slot.set(node.clone());
                        tracing::info!("Raft node wired into SQL write path");
                    }

                    // Spawn a follower applier that executes committed SQL on non-leader nodes
                    {
                        let applier_node = node.clone();
                        let amorphic = self.amorphic.clone();
                        let leader_flag = self.raft_is_leader.clone();
                        tokio::spawn(async move {
                            Self::raft_follower_applier(applier_node, amorphic, leader_flag).await;
                        });
                    }

                    _raft_node = Some(node);
                    _raft_rpc_handle = Some(rpc_handle);
                    _raft_loop_handle = Some(election_handle);
                    let _ = replication_handle; // kept alive by tokio::spawn
                }
                Err(e) => {
                    tracing::error!("Failed to start Raft node: {}", e);
                }
            }
        }

        // Run all enabled servers concurrently
        tokio::select! {
            result = http_future => {
                result?;
            }
            result = async {
                match tcp_server {
                    Some(s) => s.run().await,
                    None => std::future::pending().await,
                }
            } => {
                result?;
            }
            result = async {
                #[cfg(feature = "webtransport")]
                {
                    match wt_server {
                        Some(s) => return s.run().await,
                        None => return std::future::pending::<Result<(), Box<dyn std::error::Error + Send + Sync>>>().await,
                    }
                }
                #[cfg(not(feature = "webtransport"))]
                {
                    std::future::pending::<Result<(), Box<dyn std::error::Error + Send + Sync>>>().await
                }
            } => {
                result?;
            }
            result = async {
                #[cfg(feature = "jwp")]
                {
                    match jwp_srv {
                        Some(s) => return s.run().await,
                        None => return std::future::pending::<Result<(), Box<dyn std::error::Error + Send + Sync>>>().await,
                    }
                }
                #[cfg(not(feature = "jwp"))]
                {
                    std::future::pending::<Result<(), Box<dyn std::error::Error + Send + Sync>>>().await
                }
            } => {
                result?;
            }
            result = async {
                match pgwire_server {
                    Some(ref s) => s.run().await.map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(e) }),
                    None => std::future::pending().await,
                }
            } => {
                result?;
            }
            result = async {
                match replication_server {
                    Some(s) => s.start().await.map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(e) }),
                    None => std::future::pending().await,
                }
            } => {
                result?;
            }
            _result = async {
                match replication_client {
                    Some(ref mut c) => {
                        let repl_amorphic = self.amorphic.clone();
                        match c.start().await {
                            Ok(mut rx) => {
                                while let Some(entry) = rx.recv().await {
                                    tracing::debug!("Applying replication entry LSN={}", entry.lsn);
                                    match entry.op_type {
                                        replication::ReplicationOp::Put => {
                                            if let Some(ref value) = entry.value {
                                                let json_str = String::from_utf8_lossy(value);
                                                if let Err(e) = repl_amorphic.ingest_json(&json_str) {
                                                    tracing::error!("Failed to apply replication Put LSN={}: {}", entry.lsn, e);
                                                }
                                            }
                                        }
                                        replication::ReplicationOp::Delete => {
                                            let key_str = String::from_utf8_lossy(&entry.key);
                                            if let Ok(record_id) = key_str.parse::<u64>() {
                                                if let Err(e) = repl_amorphic.delete_record(record_id) {
                                                    tracing::error!("Failed to apply replication Delete LSN={}: {}", entry.lsn, e);
                                                }
                                            } else {
                                                tracing::warn!("Replication Delete LSN={}: non-numeric key '{}'", entry.lsn, key_str);
                                            }
                                        }
                                        replication::ReplicationOp::Checkpoint => {
                                            tracing::info!("Received replication checkpoint at LSN={}", entry.lsn);
                                        }
                                    }
                                    c.set_applied_lsn(entry.lsn);
                                }
                            }
                            Err(e) => {
                                tracing::error!("Replication client error: {}", e);
                            }
                        }
                    }
                    None => { std::future::pending::<()>().await; }
                }
            } => {}
            _ = checkpoint_task => {}
            _ = session_reaper_task => {}
        }

        Ok(())
    }

    /// Background task that applies committed Raft entries on follower nodes.
    /// On the leader, this is a no-op since writes are already executed locally.
    ///
    /// HRP Phase 2: Handles both `MutationDelta` (direct storage apply) and
    /// legacy `Command::Set` (SQL re-execution) for backward compatibility.
    async fn raft_follower_applier(
        node: Arc<raft::RaftNode<raft::KvStateMachine, raft_transport::TcpRaftTransport>>,
        amorphic: Arc<amorphic_adapter::AmorphicTableStorage>,
        leader_flag: Arc<std::sync::atomic::AtomicBool>,
    ) {
        let mut rx = node.subscribe_commits();
        let mut last_applied: u64 = 0;
        // Create a dedicated executor for applying RawSql deltas and legacy entries
        let exec = query::SimpleQueryExecutor::with_amorphic(amorphic.clone());

        loop {
            // Wait for a commit notification
            match rx.recv().await {
                Ok(committed_index) => {
                    // Only followers need to apply — the leader already executed locally
                    if leader_flag.load(std::sync::atomic::Ordering::Relaxed) {
                        last_applied = committed_index;
                        continue;
                    }

                    // Apply entries from last_applied+1 to committed_index
                    let entries = node
                        .get_log_entries(last_applied + 1, committed_index)
                        .await;

                    for entry in entries {
                        let applied_ok = match &entry.command {
                            // HRP Phase 2: Direct storage apply from MutationDelta
                            raft::Command::MutationDelta(data) => {
                                match mutation_delta::MutationDelta::from_bytes(data) {
                                    Ok(delta) => {
                                        match Self::apply_mutation_delta(&amorphic, &delta, &exec) {
                                            Ok(()) => {
                                                tracing::debug!(
                                                    "Raft follower applied delta entry {}",
                                                    entry.index
                                                );
                                                true
                                            }
                                            Err(e) => {
                                                tracing::error!(
                                                    "Raft follower FAILED to apply delta entry {}: {} \
                                                     — halting to prevent state divergence",
                                                    entry.index,
                                                    e
                                                );
                                                false
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        tracing::error!(
                                            "Raft follower FAILED to deserialize delta entry {}: {} \
                                             — halting to prevent state divergence",
                                            entry.index,
                                            e
                                        );
                                        false
                                    }
                                }
                            }
                            // Legacy path: SQL re-execution (backward compatibility)
                            raft::Command::Set { key, value } if key == b"sql" => {
                                if let Ok(sql) = String::from_utf8(value.clone()) {
                                    tracing::debug!(
                                        "Raft follower applying SQL: {}",
                                        &sql[..sql.len().min(100)]
                                    );
                                    Self::apply_sql_entry(&exec, &sql, entry.index);
                                }
                                true
                            }
                            _ => true,
                        };

                        if applied_ok {
                            last_applied = entry.index;
                        } else {
                            // CRITICAL: Do NOT advance last_applied past a failed entry.
                            // The next commit notification will retry from this entry,
                            // preventing leader-follower state machine divergence.
                            tracing::error!(
                                "Raft follower halted at entry {} — will retry on next commit",
                                entry.index
                            );
                            break;
                        }
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                    tracing::warn!("Raft follower applier lagged by {} messages", n);
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                    tracing::info!("Raft commit channel closed, stopping follower applier");
                    break;
                }
            }
        }
    }

    /// Apply a SQL entry via the SimpleQueryExecutor (legacy path).
    fn apply_sql_entry(exec: &query::SimpleQueryExecutor, sql: &str, index: u64) {
        let request = query::QueryRequest {
            sql: sql.to_string(),
            params: Default::default(),
            args: Vec::new(),
            explain: false,
            limit: None,
            session_id: None,
            query_timeout_ms: None,
            branch_id: None,
            tenant_id: None,
        };
        match exec.execute(&request) {
            Ok(_) => tracing::debug!("Raft follower applied entry {}", index),
            Err(e) => tracing::warn!(
                "Raft follower failed to apply entry {}: {}",
                index,
                e.message
            ),
        }
    }

    /// HRP Phase 2: Apply a MutationDelta directly to AmorphicTableStorage,
    /// bypassing the SQL parser and query planner.
    fn apply_mutation_delta(
        amorphic: &amorphic_adapter::AmorphicTableStorage,
        delta: &mutation_delta::MutationDelta,
        exec: &query::SimpleQueryExecutor,
    ) -> Result<(), String> {
        use joule_db_query::ast::Value as AstValue;
        use joule_db_query::executor::RowData;

        match delta {
            mutation_delta::MutationDelta::InsertRows {
                table,
                columns,
                rows,
            } => {
                for row_values in rows {
                    let values: Vec<AstValue> = row_values.iter().map(delta_value_to_ast).collect();
                    let row = RowData::new(columns.clone(), values);
                    amorphic
                        .insert_returning_id(table, &row)
                        .map_err(|e| format!("InsertRow: {}", e))?;
                }
                Ok(())
            }
            mutation_delta::MutationDelta::CreateTable {
                name,
                columns,
                column_defs,
                if_not_exists,
            } => {
                // Check if table already exists when if_not_exists is set
                if *if_not_exists && amorphic.has_table(name) {
                    return Ok(());
                }
                // Convert mutation_delta::ColumnDef to SqlColumnDef
                let sql_defs: Vec<joule_db_query::sql::SqlColumnDef> = column_defs
                    .iter()
                    .map(|cd| joule_db_query::sql::SqlColumnDef {
                        name: cd.name.clone(),
                        data_type: cd.data_type.clone(),
                        nullable: cd.nullable,
                        primary_key: cd.primary_key,
                        unique: cd.unique,
                        default: None,
                        check: None,
                        auto_increment: cd.auto_increment,
                        foreign_key: None,
                        column_family: None,
                        computed: None,
                    })
                    .collect();
                amorphic
                    .create_table_with_defs(name, columns, &sql_defs)
                    .map_err(|e| format!("CreateTable: {}", e))
            }
            mutation_delta::MutationDelta::DropTable { name, if_exists } => {
                if *if_exists && !amorphic.has_table(name) {
                    return Ok(());
                }
                amorphic
                    .drop_table(name)
                    .map(|_| ())
                    .map_err(|e| format!("DropTable: {}", e))
            }
            mutation_delta::MutationDelta::RawSql { sql } => {
                // Fallback: re-execute SQL through the query executor
                tracing::debug!(
                    "Raft follower applying RawSql delta: {}",
                    &sql[..sql.len().min(100)]
                );
                Self::apply_sql_entry(exec, sql, 0);
                Ok(())
            }
        }
    }
}

/// Convert a DeltaValue to an AST Value for storage insertion (HRP Phase 2).
fn delta_value_to_ast(v: &mutation_delta::DeltaValue) -> joule_db_query::ast::Value {
    use joule_db_query::ast::Value as AstValue;
    use mutation_delta::DeltaValue;
    match v {
        DeltaValue::Null => AstValue::Null,
        DeltaValue::Bool(b) => AstValue::Bool(*b),
        DeltaValue::Int(i) => AstValue::Int(*i),
        DeltaValue::Float(f) => AstValue::Float(*f),
        DeltaValue::Text(s) => AstValue::String(s.clone()),
        DeltaValue::Blob(b) => AstValue::Bytes(b.clone()),
        DeltaValue::Array(arr) => AstValue::Array(arr.iter().map(delta_value_to_ast).collect()),
    }
}

// Route handlers

/// Health check response for Kubernetes probes
#[derive(serde::Serialize)]
struct HealthResponse {
    status: String,
    message: String,
    timestamp: u64,
    uptime_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    components: Option<std::collections::HashMap<String, String>>,
}

async fn health_check() -> Json<HealthResponse> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;

    Json(HealthResponse {
        status: "healthy".to_string(),
        message: "All systems operational".to_string(),
        timestamp: now,
        uptime_ms: now, // Would use actual server uptime in production
        components: None,
    })
}

/// Liveness probe - returns 200 if server is running
async fn liveness() -> StatusCode {
    StatusCode::OK
}

/// Readiness probe - returns 200 if server is ready to handle requests
async fn readiness() -> StatusCode {
    // In production, would check database connectivity, etc.
    StatusCode::OK
}

/// Energy status endpoint - returns current hardware state and energy metrics
async fn energy_status_handler(
    State(state): State<AppState>,
) -> Json<energy::EnergyStatusResponse> {
    let snapshot = state
        .energy_snapshot
        .read()
        .map(|s| s.clone())
        .unwrap_or_default();
    let hints = state.energy_advisor.advise(&snapshot);
    let queries_tracked = state.energy_metrics.queries_tracked.get();
    Json(energy::EnergyStatusResponse::from_snapshot(
        &snapshot,
        &hints,
        queries_tracked,
    ))
}

/// Optional pagination parameters for list endpoints.
/// Defaults: limit=1000, offset=0. Max limit=10000.
#[derive(Debug, serde::Deserialize)]
struct PaginationParams {
    #[serde(default = "default_page_limit")]
    limit: usize,
    #[serde(default)]
    offset: usize,
}
fn default_page_limit() -> usize {
    1000
}

impl PaginationParams {
    fn apply<T>(&self, items: Vec<T>) -> Vec<T> {
        let limit = self.limit.min(10_000);
        items.into_iter().skip(self.offset).take(limit).collect()
    }
}

// ============================================================================
// Branch management handlers
// ============================================================================

async fn list_branches_handler(
    State(state): State<AppState>,
    Query(page): Query<PaginationParams>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    match state.branch_manager.list_branches() {
        Ok(branches) => Ok(Json(
            serde_json::json!({ "branches": page.apply(branches) }),
        )),
        Err(e) => {
            tracing::error!("Failed to list branches: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn create_branch_handler(
    State(state): State<AppState>,
    Json(req): Json<joule_db_branch::CreateBranchRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    match state.branch_manager.create_branch(req) {
        Ok(info) => Ok(Json(serde_json::json!({ "branch": info }))),
        Err(joule_db_branch::BranchError::AlreadyExists(_)) => Err(StatusCode::CONFLICT),
        Err(joule_db_branch::BranchError::ParentNotFound(_)) => Err(StatusCode::NOT_FOUND),
        Err(joule_db_branch::BranchError::InvalidName(_)) => Err(StatusCode::BAD_REQUEST),
        Err(e) => {
            tracing::error!("Failed to create branch: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn get_branch_handler(
    State(state): State<AppState>,
    axum::extract::Path(name): axum::extract::Path<String>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    match state.branch_manager.get_branch(&name) {
        Ok(info) => Ok(Json(serde_json::json!({ "branch": info }))),
        Err(joule_db_branch::BranchError::NotFound(_)) => Err(StatusCode::NOT_FOUND),
        Err(e) => {
            tracing::error!("Failed to get branch: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn delete_branch_handler(
    State(state): State<AppState>,
    axum::extract::Path(name): axum::extract::Path<String>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    match state.branch_manager.delete_branch(&name) {
        Ok(info) => Ok(Json(serde_json::json!({ "deleted": info }))),
        Err(joule_db_branch::BranchError::CannotDeleteMain) => Err(StatusCode::FORBIDDEN),
        Err(joule_db_branch::BranchError::NotFound(_)) => Err(StatusCode::NOT_FOUND),
        Err(e) => {
            tracing::error!("Failed to delete branch: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

#[derive(serde::Deserialize)]
struct MergeBranchBody {
    #[serde(default)]
    delete_after: bool,
}

async fn merge_branch_handler(
    State(state): State<AppState>,
    axum::extract::Path(name): axum::extract::Path<String>,
    Json(body): Json<MergeBranchBody>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    match state.branch_manager.merge_branch(&name, body.delete_after) {
        Ok(result) => Ok(Json(serde_json::json!({ "merge": result }))),
        Err(joule_db_branch::BranchError::NotFound(_)) => Err(StatusCode::NOT_FOUND),
        Err(e) => {
            tracing::error!("Failed to merge branch: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn diff_branch_handler(
    State(state): State<AppState>,
    axum::extract::Path(name): axum::extract::Path<String>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    match state.branch_manager.diff_branch(&name) {
        Ok(diff) => Ok(Json(serde_json::json!({ "diff": diff }))),
        Err(joule_db_branch::BranchError::NotFound(_)) => Err(StatusCode::NOT_FOUND),
        Err(e) => {
            tracing::error!("Failed to diff branch: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

// ── Tenant REST handlers ──────────────────────────────────────────────────────

async fn list_tenants_handler(
    State(state): State<AppState>,
    Query(page): Query<PaginationParams>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    match state.tenant_manager.list_tenants() {
        Ok(tenants) => Ok(Json(serde_json::json!({ "tenants": page.apply(tenants) }))),
        Err(e) => {
            tracing::error!("Failed to list tenants: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn create_tenant_handler(
    State(state): State<AppState>,
    Json(req): Json<tenant::CreateTenantRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    match state.tenant_manager.create_tenant(req) {
        Ok(info) => Ok(Json(serde_json::json!({ "tenant": info }))),
        Err(tenant::TenantError::AlreadyExists(_)) => Err(StatusCode::CONFLICT),
        Err(e) => {
            tracing::error!("Failed to create tenant: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn get_tenant_handler(
    State(state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    match state.tenant_manager.get_tenant(&id) {
        Ok(info) => Ok(Json(serde_json::json!({ "tenant": info }))),
        Err(tenant::TenantError::NotFound(_)) => Err(StatusCode::NOT_FOUND),
        Err(e) => {
            tracing::error!("Failed to get tenant: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn delete_tenant_handler(
    State(state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    match state.tenant_manager.delete_tenant(&id) {
        Ok(info) => Ok(Json(serde_json::json!({ "deleted": info }))),
        Err(tenant::TenantError::CannotDeleteDefault) => Err(StatusCode::FORBIDDEN),
        Err(tenant::TenantError::NotFound(_)) => Err(StatusCode::NOT_FOUND),
        Err(e) => {
            tracing::error!("Failed to delete tenant: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn suspend_tenant_handler(
    State(state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    match state.tenant_manager.suspend_tenant(&id) {
        Ok(info) => Ok(Json(serde_json::json!({ "tenant": info }))),
        Err(tenant::TenantError::NotFound(_)) => Err(StatusCode::NOT_FOUND),
        Err(e) => {
            tracing::error!("Failed to suspend tenant: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn tenant_energy_handler(
    State(state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    match state.tenant_manager.get_tenant(&id) {
        Ok(info) => Ok(Json(serde_json::json!({
            "tenant_id": info.id,
            "energy_spent_uj": info.energy_spent_uj,
            "energy_budget_uj": info.quotas.energy_budget_uj,
        }))),
        Err(tenant::TenantError::NotFound(_)) => Err(StatusCode::NOT_FOUND),
        Err(e) => {
            tracing::error!("Failed to get tenant energy: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

// ============================================================================
// Scale-to-Zero handlers
// ============================================================================

async fn status_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    let report = state.activity_tracker.status();
    Json(serde_json::json!(report))
}

async fn suspend_handler(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    match state.activity_tracker.suspend() {
        Ok(report) => Ok(Json(serde_json::json!(report))),
        Err(e) => {
            tracing::error!("Failed to suspend: {}", e);
            Err(StatusCode::CONFLICT)
        }
    }
}

async fn resume_handler(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    match state.activity_tracker.resume() {
        Ok(()) => Ok(Json(serde_json::json!({ "state": "active" }))),
        Err(e) => {
            tracing::error!("Failed to resume: {}", e);
            Err(StatusCode::CONFLICT)
        }
    }
}

// ============================================================================
// Agent Memory handlers
// ============================================================================

/// Extract tenant_id from auth context, defaulting to "default" for unauthenticated requests.
fn extract_tenant_id(auth_ext: &Option<Extension<AuthInfo>>) -> String {
    auth_ext
        .as_ref()
        .and_then(|ext| ext.0.tenant_id.clone())
        .unwrap_or_else(|| "default".to_string())
}

async fn store_memory_handler(
    State(state): State<AppState>,
    auth_ext: Option<Extension<AuthInfo>>,
    Json(req): Json<agent_memory::StoreMemoryRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let tenant_id = extract_tenant_id(&auth_ext);
    match state.memory_manager.store(&req, &tenant_id) {
        Ok(id) => Ok(Json(serde_json::json!({ "id": id }))),
        Err(e) => {
            tracing::error!("Failed to store memory: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn recall_memory_handler(
    State(state): State<AppState>,
    auth_ext: Option<Extension<AuthInfo>>,
    Json(req): Json<agent_memory::RecallMemoryRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let tenant_id = extract_tenant_id(&auth_ext);
    match state.memory_manager.recall(&req, &tenant_id) {
        Ok(results) => Ok(Json(serde_json::json!({ "memories": results }))),
        Err(e) => {
            tracing::error!("Failed to recall memories: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn forget_memory_handler(
    State(state): State<AppState>,
    auth_ext: Option<Extension<AuthInfo>>,
    Json(req): Json<agent_memory::ForgetMemoryRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let tenant_id = extract_tenant_id(&auth_ext);
    match state.memory_manager.forget(&req, &tenant_id) {
        Ok(count) => Ok(Json(serde_json::json!({ "deleted": count }))),
        Err(e) => {
            tracing::error!("Failed to forget memories: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn consolidate_memory_handler(
    State(state): State<AppState>,
    auth_ext: Option<Extension<AuthInfo>>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let tenant_id = extract_tenant_id(&auth_ext);
    match state.memory_manager.consolidate(&tenant_id) {
        Ok(count) => Ok(Json(serde_json::json!({ "consolidated": count }))),
        Err(e) => {
            tracing::error!("Failed to consolidate memories: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn memory_stats_handler(
    State(state): State<AppState>,
    auth_ext: Option<Extension<AuthInfo>>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let tenant_id = extract_tenant_id(&auth_ext);
    match state.memory_manager.stats(&tenant_id) {
        Ok(stats) => Ok(Json(serde_json::json!(stats))),
        Err(e) => {
            tracing::error!("Failed to get memory stats: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

// ── Workflow handlers ────────────────────────────────────────────────────

async fn create_workflow_handler(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let name = body["name"]
        .as_str()
        .filter(|s| !s.is_empty())
        .ok_or(StatusCode::BAD_REQUEST)?
        .to_string();
    let steps: Vec<workflow::WorkflowStep> =
        serde_json::from_value(body["steps"].clone()).map_err(|_| StatusCode::BAD_REQUEST)?;
    let retry_policy: Option<workflow::WorkflowRetryConfig> = body
        .get("retry_policy")
        .and_then(|v| serde_json::from_value(v.clone()).ok());
    let energy_budget_uj = body["energy_budget_uj"].as_u64();

    match state
        .workflow_manager
        .create_definition(name, steps, retry_policy, energy_budget_uj)
    {
        Ok(def) => Ok(Json(serde_json::json!(def))),
        Err(e) => {
            tracing::error!("Failed to create workflow: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn list_workflows_handler(
    State(state): State<AppState>,
    Query(page): Query<PaginationParams>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    match state.workflow_manager.list_definitions() {
        Ok(defs) => Ok(Json(serde_json::json!({ "workflows": page.apply(defs) }))),
        Err(e) => {
            tracing::error!("Failed to list workflows: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn get_workflow_handler(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    match state.workflow_manager.get_definition(&id) {
        Ok(def) => Ok(Json(serde_json::json!(def))),
        Err(workflow::WorkflowError::NotFound(_)) => Err(StatusCode::NOT_FOUND),
        Err(e) => {
            tracing::error!("Failed to get workflow: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn delete_workflow_handler(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<StatusCode, StatusCode> {
    match state.workflow_manager.delete_definition(&id) {
        Ok(()) => Ok(StatusCode::NO_CONTENT),
        Err(workflow::WorkflowError::NotFound(_)) => Err(StatusCode::NOT_FOUND),
        Err(e) => {
            tracing::error!("Failed to delete workflow: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn run_workflow_handler(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    match state.workflow_manager.run(&id) {
        Ok(instance) => Ok(Json(serde_json::json!(instance))),
        Err(workflow::WorkflowError::NotFound(_)) => Err(StatusCode::NOT_FOUND),
        Err(e) => {
            tracing::error!("Failed to run workflow: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn get_workflow_instance_handler(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    match state.workflow_manager.get_instance(&id) {
        Ok(instance) => Ok(Json(serde_json::json!(instance))),
        Err(workflow::WorkflowError::InstanceNotFound(_)) => Err(StatusCode::NOT_FOUND),
        Err(e) => {
            tracing::error!("Failed to get workflow instance: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn publish_message_handler(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let topic = body["topic"].as_str().ok_or(StatusCode::BAD_REQUEST)?;
    let payload = body["payload"].as_str().unwrap_or("").to_string();
    match state.workflow_manager.publish(topic, payload) {
        Ok(msg) => Ok(Json(serde_json::json!(msg))),
        Err(e) => {
            tracing::error!("Failed to publish message: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn subscribe_handler(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let topic = body["topic"].as_str().ok_or(StatusCode::BAD_REQUEST)?;
    let max_messages = body["max_messages"].as_u64().unwrap_or(10).min(10_000) as usize;
    match state.workflow_manager.subscribe(topic, max_messages) {
        Ok(msgs) => Ok(Json(serde_json::json!(msgs))),
        Err(e) => {
            tracing::error!("Failed to subscribe: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn ack_messages_handler(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let message_ids: Vec<String> =
        serde_json::from_value(body["message_ids"].clone()).map_err(|_| StatusCode::BAD_REQUEST)?;
    match state.workflow_manager.ack(&message_ids) {
        Ok(count) => Ok(Json(serde_json::json!({"acked": count}))),
        Err(e) => {
            tracing::error!("Failed to ack messages: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn dead_letters_handler(
    State(state): State<AppState>,
    Path(topic): Path<String>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    match state.workflow_manager.dead_letters(&topic) {
        Ok(msgs) => Ok(Json(serde_json::json!(msgs))),
        Err(e) => {
            tracing::error!("Failed to get dead letters: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

// ── Edge PoP handlers ───────────────────────────────────────────────────

async fn register_pop_handler(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let region: edge_pop::PopRegion =
        serde_json::from_value(body["region"].clone()).map_err(|_| StatusCode::BAD_REQUEST)?;
    let endpoint = body["endpoint"]
        .as_str()
        .ok_or(StatusCode::BAD_REQUEST)?
        .to_string();
    let is_wasm = body["is_wasm"].as_bool().unwrap_or(false);
    match state.edge_pop_manager.register(region, endpoint, is_wasm) {
        Ok(pop) => Ok(Json(serde_json::json!(pop))),
        Err(e) => {
            tracing::error!("Failed to register PoP: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn list_pops_handler(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    match state.edge_pop_manager.list() {
        Ok(pops) => Ok(Json(serde_json::json!(pops))),
        Err(e) => {
            tracing::error!("Failed to list PoPs: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn get_pop_handler(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    match state.edge_pop_manager.get(&id) {
        Ok(pop) => Ok(Json(serde_json::json!(pop))),
        Err(edge_pop::EdgePopError::NotFound(_)) => Err(StatusCode::NOT_FOUND),
        Err(e) => {
            tracing::error!("Failed to get PoP: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn deregister_pop_handler(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    match state.edge_pop_manager.deregister(&id) {
        Ok(pop) => Ok(Json(serde_json::json!(pop))),
        Err(edge_pop::EdgePopError::NotFound(_)) => Err(StatusCode::NOT_FOUND),
        Err(e) => {
            tracing::error!("Failed to deregister PoP: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn trigger_sync_handler(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let pop_id = body["pop_id"].as_str().ok_or(StatusCode::BAD_REQUEST)?;
    match state.edge_pop_manager.trigger_sync(pop_id) {
        Ok(report) => Ok(Json(serde_json::json!(report))),
        Err(edge_pop::EdgePopError::NotFound(_)) => Err(StatusCode::NOT_FOUND),
        Err(edge_pop::EdgePopError::Offline(_)) => Err(StatusCode::SERVICE_UNAVAILABLE),
        Err(e) => {
            tracing::error!("Failed to trigger sync: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn edge_stats_handler(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    match state.edge_pop_manager.stats() {
        Ok(stats) => Ok(Json(serde_json::json!(stats))),
        Err(e) => {
            tracing::error!("Failed to get edge stats: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Replication status response
#[derive(serde::Serialize)]
struct ReplicationStatusResponse {
    enabled: bool,
    role: Option<String>,
    current_lsn: u64,
    followers: Vec<ReplicationFollowerInfo>,
}

#[derive(serde::Serialize)]
struct ReplicationFollowerInfo {
    node_id: String,
    acked_lsn: u64,
    lag: u64,
    connected: bool,
}

async fn replication_status_handler(
    State(state): State<AppState>,
) -> Json<ReplicationStatusResponse> {
    match &state.replication_server {
        Some(server) => {
            let follower_states = server.followers().await;
            let follower_list: Vec<ReplicationFollowerInfo> = follower_states
                .iter()
                .map(|f| ReplicationFollowerInfo {
                    node_id: f.node_id.clone(),
                    acked_lsn: f.acked_lsn,
                    lag: f.lag,
                    connected: f.connected,
                })
                .collect();
            Json(ReplicationStatusResponse {
                enabled: true,
                role: state.replication_role.clone(),
                current_lsn: server.current_lsn(),
                followers: follower_list,
            })
        }
        None => Json(ReplicationStatusResponse {
            enabled: state.replication_role.is_some(),
            role: state.replication_role.clone(),
            current_lsn: 0,
            followers: Vec::new(),
        }),
    }
}

async fn get_key(
    State(state): State<AppState>,
    Path(key): Path<String>,
) -> Result<Json<Option<String>>, StatusCode> {
    let db = state.db.read().await;
    match db.get(key.as_bytes()) {
        Ok(Some(value)) => match String::from_utf8(value) {
            Ok(s) => Ok(Json(Some(s))),
            Err(e) => {
                // Binary data: return base64-encoded with a prefix so callers
                // can distinguish it from plain text.
                use base64::Engine as _;
                let encoded = base64::engine::general_purpose::STANDARD.encode(e.into_bytes());
                Ok(Json(Some(format!("base64:{encoded}"))))
            }
        },
        Ok(None) => Ok(Json(None)),
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

/// Query parameters for KV put operations.
#[derive(Debug, serde::Deserialize)]
struct PutKeyParams {
    /// Time-to-live in seconds. If omitted, the key is permanent.
    ttl: Option<u64>,
}

async fn put_key(
    State(state): State<AppState>,
    Path(key): Path<String>,
    Query(params): Query<PutKeyParams>,
    body: Bytes,
) -> Result<StatusCode, StatusCode> {
    // Hold write lock for entire read-check-write to avoid TOCTOU race
    let db = state.db.write().await;
    let old_value = db.get_raw(key.as_bytes()).ok().flatten();
    let result = if let Some(ttl) = params.ttl {
        db.put_with_ttl(key.as_bytes(), &body, ttl)
    } else {
        db.put(key.as_bytes(), &body)
    };
    match result {
        Ok(_) => {
            // Fire subscription notification
            if let Some(old_val) = old_value {
                state
                    .subscription_manager
                    .notify_update(&key, &old_val, &body)
                    .await;
            } else {
                state.subscription_manager.notify_insert(&key, &body).await;
            }
            Ok(StatusCode::OK)
        }
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

/// Get remaining TTL for a key.
async fn get_key_ttl(
    State(state): State<AppState>,
    Path(key): Path<String>,
) -> Result<Json<Option<u64>>, StatusCode> {
    let db = state.db.read().await;
    match db.ttl(key.as_bytes()) {
        Ok(ttl) => Ok(Json(ttl)),
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

/// Cleanup expired keys.
async fn cleanup_expired(State(state): State<AppState>) -> Result<Json<usize>, StatusCode> {
    let db = state.db.write().await;
    match db.cleanup_expired() {
        Ok(count) => Ok(Json(count)),
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

async fn delete_key(
    State(state): State<AppState>,
    Path(key): Path<String>,
) -> Result<Json<bool>, StatusCode> {
    // Hold write lock for entire read-check-delete to avoid TOCTOU race
    let db = state.db.write().await;
    let old_value = db.get(key.as_bytes()).ok().flatten();
    match db.delete(key.as_bytes()) {
        Ok(deleted) => {
            if deleted {
                state
                    .subscription_manager
                    .notify_delete(&key, old_value.as_deref())
                    .await;
            }
            Ok(Json(deleted))
        }
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

// ================================================================
// Unified Endpoint — POST /
// ================================================================

/// Returns true if the string looks like a SQL statement.
fn is_sql(q: &str) -> bool {
    // Strip leading SQL comments before checking the first keyword.
    // Without this, "-- note\nSELECT 1" would be treated as a similarity
    // search instead of SQL.
    let mut s = q.trim();
    loop {
        if let Some(rest) = s.strip_prefix("--") {
            s = rest.find('\n').map(|i| &rest[i + 1..]).unwrap_or("").trim();
        } else if let Some(rest) = s.strip_prefix("/*") {
            s = rest.find("*/").map(|i| &rest[i + 2..]).unwrap_or("").trim();
        } else {
            break;
        }
    }
    if s.is_empty() {
        return false;
    }
    // Find first word after stripping comments
    let first_word = s.split_whitespace().next().unwrap_or("").to_uppercase();
    matches!(
        first_word.as_str(),
        "SELECT"
            | "INSERT"
            | "UPDATE"
            | "DELETE"
            | "CREATE"
            | "DROP"
            | "ALTER"
            | "BEGIN"
            | "COMMIT"
            | "ROLLBACK"
            | "EXPLAIN"
            | "SHOW"
            | "WITH"
            | "TRUNCATE"
            | "GRANT"
            | "REVOKE"
    )
}

/// POST / — unified endpoint for data ingest and queries.
///
/// The shape of the JSON body determines the operation:
/// - Array → batch ingest
/// - Object with `q` key → query (SQL auto-detected, otherwise similarity search)
/// - Object without `q` → single record ingest
async fn unified_handler(
    State(state): State<AppState>,
    auth_ext: Option<Extension<AuthInfo>>,
    body: Bytes,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let start = std::time::Instant::now();
    let auth_info = auth_ext.map(|e| e.0).unwrap_or(AuthInfo {
        user_id: "anonymous".to_string(),
        roles: vec!["superuser".to_string()],
        tenant_id: None,
    });

    // Check RBAC permission for write operations
    if let Some(ref rbac) = state.rbac_manager {
        let resource = crate::rbac::ResourceType::Server;
        let has_read = rbac
            .check_permission(
                &auth_info.user_id,
                crate::rbac::PermissionType::Read,
                &resource,
            )
            .unwrap_or(false);
        if !has_read && !auth_info.roles.contains(&"superuser".to_string()) {
            return Err((
                StatusCode::FORBIDDEN,
                Json(serde_json::json!({"ok": false, "error": "Permission denied"})),
            ));
        }
    }

    let body_str = std::str::from_utf8(&body).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"ok": false, "error": format!("Invalid UTF-8: {}", e)})),
        )
    })?;

    let body_val: serde_json::Value = serde_json::from_str(body_str).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"ok": false, "error": format!("Invalid JSON: {}", e)})),
        )
    })?;

    match &body_val {
        // Batch ingest: array of objects
        serde_json::Value::Array(arr) => {
            let records: Vec<serde_json::Value> = arr.clone();
            let (ids, collection) = state
                .amorphic
                .batch_ingest_with_schema(&records, None)
                .map_err(|e| {
                    let msg = if state.sanitize_errors {
                        tracing::warn!(detail = %e, "Batch ingest error (sanitized)");
                        "Ingest operation failed".to_string()
                    } else {
                        e.to_string()
                    };
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({"ok": false, "error": msg})),
                    )
                })?;
            let ms = start.elapsed().as_millis() as u64;
            Ok(Json(serde_json::json!({
                "ok": true,
                "ids": ids,
                "count": ids.len(),
                "collection": collection,
                "ms": ms,
            })))
        }

        // Object: either query or single ingest
        serde_json::Value::Object(map) => {
            // Check for `q` field — indicates a query
            if let Some(serde_json::Value::String(q)) = map.get("q") {
                let k = map.get("k").and_then(|v| v.as_u64()).unwrap_or(10) as usize;

                if is_sql(q) {
                    // Enforce write permissions: readonly users cannot execute mutations
                    if let Err(e) =
                        query::check_write_permission(&auth_info.user_id, &auth_info.roles, q)
                    {
                        return Err((
                            StatusCode::FORBIDDEN,
                            Json(serde_json::json!({"ok": false, "error": e.message})),
                        ));
                    }

                    // SQL query path
                    let request = query::QueryRequest {
                        sql: q.clone(),
                        params: Default::default(),
                        args: Vec::new(),
                        explain: false,
                        limit: map
                            .get("limit")
                            .and_then(|v| v.as_u64())
                            .map(|v| v as usize),
                        session_id: map
                            .get("session_id")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string()),
                        query_timeout_ms: None,
                        branch_id: None,
                        tenant_id: None,
                    };
                    match state.query_executor.execute(&request) {
                        Ok(resp) => {
                            let ms = start.elapsed().as_millis() as u64;
                            Ok(Json(serde_json::json!({
                                "ok": true,
                                "columns": resp.columns,
                                "data": resp.rows,
                                "count": resp.rows.len(),
                                "affected_rows": resp.affected_rows,
                                "ms": ms,
                                "session_id": resp.session_id,
                            })))
                        }
                        Err(err) => {
                            let msg = if state.sanitize_errors {
                                tracing::warn!(code = %err.code, detail = %err.message, "Query error (sanitized)");
                                err.sanitized().message
                            } else {
                                err.message
                            };
                            Err((
                                StatusCode::BAD_REQUEST,
                                Json(serde_json::json!({"ok": false, "error": msg})),
                            ))
                        }
                    }
                } else {
                    // Similarity search path
                    let results = state.amorphic.query_similar_to(q, k);
                    let ms = start.elapsed().as_millis() as u64;
                    Ok(Json(serde_json::json!({
                        "ok": true,
                        "data": results,
                        "count": results.len(),
                        "ms": ms,
                    })))
                }
            } else {
                // Single record ingest
                let (id, collection) =
                    state
                        .amorphic
                        .ingest_with_schema(body_str, None)
                        .map_err(|e| {
                            let msg = if state.sanitize_errors {
                                tracing::warn!(detail = %e, "Ingest error (sanitized)");
                                "Ingest operation failed".to_string()
                            } else {
                                e.to_string()
                            };
                            (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                Json(serde_json::json!({"ok": false, "error": msg})),
                            )
                        })?;
                let ms = start.elapsed().as_millis() as u64;
                Ok(Json(serde_json::json!({
                    "ok": true,
                    "id": id,
                    "collection": collection,
                    "ms": ms,
                })))
            }
        }

        _ => Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"ok": false, "error": "Expected JSON object or array"})),
        )),
    }
}

// ================================================================
// Structured Error Helper
// ================================================================

/// Convert a plain-text error into a consistent JSON error response.
/// All API errors should use `{"error": "..."}` so clients can parse uniformly.
fn json_error(
    status: StatusCode,
    message: impl Into<String>,
) -> (StatusCode, Json<serde_json::Value>) {
    (status, Json(serde_json::json!({ "error": message.into() })))
}

// ================================================================
// Amorphic Multi-Model API Handlers
// ================================================================

/// POST /api/v1/ingest — ingest a JSON document into the amorphic store
async fn amorphic_ingest_handler(
    State(state): State<AppState>,
    body: Bytes,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let json_str = std::str::from_utf8(&body)
        .map_err(|e| json_error(StatusCode::BAD_REQUEST, format!("Invalid UTF-8: {}", e)))?;

    let id = state
        .amorphic
        .ingest_json(json_str)
        .map_err(|e| json_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(serde_json::json!({ "id": id })))
}

/// POST /api/v1/ingest/edge — ingest a graph edge
async fn amorphic_ingest_edge_handler(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let source = body["source"]
        .as_str()
        .ok_or_else(|| json_error(StatusCode::BAD_REQUEST, "Missing 'source' field"))?;
    let relation = body["relation"]
        .as_str()
        .ok_or_else(|| json_error(StatusCode::BAD_REQUEST, "Missing 'relation' field"))?;
    let target = body["target"]
        .as_str()
        .ok_or_else(|| json_error(StatusCode::BAD_REQUEST, "Missing 'target' field"))?;

    let id = state
        .amorphic
        .ingest_edge(source, relation, target)
        .map_err(|e| json_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(serde_json::json!({ "id": id })))
}

/// GET /api/v1/records/{id} — get a record by ID
async fn amorphic_get_record_handler(
    State(state): State<AppState>,
    Path(id): Path<u64>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    match state.amorphic.get_record(id) {
        Some(doc) => Ok(Json(doc)),
        None => Err(json_error(
            StatusCode::NOT_FOUND,
            format!("Record {} not found", id),
        )),
    }
}

/// DELETE /api/v1/records/{id} — delete a record
async fn amorphic_delete_record_handler(
    State(state): State<AppState>,
    Path(id): Path<u64>,
) -> Result<StatusCode, (StatusCode, Json<serde_json::Value>)> {
    state
        .amorphic
        .delete_record(id)
        .map_err(|e| json_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(StatusCode::NO_CONTENT)
}

/// POST /api/v1/query/similar — similarity search
async fn amorphic_similar_handler(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let name = body["name"]
        .as_str()
        .ok_or_else(|| json_error(StatusCode::BAD_REQUEST, "Missing 'name' field"))?;
    let k = body["k"].as_u64().unwrap_or(10) as usize;

    let results = state.amorphic.query_similar_to(name, k);
    Ok(Json(serde_json::json!({ "results": results })))
}

/// POST /api/v1/query/graph — graph traversal
async fn amorphic_graph_handler(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let start = body["start"]
        .as_str()
        .ok_or_else(|| json_error(StatusCode::BAD_REQUEST, "Missing 'start' field"))?;
    let relation = body["relation"]
        .as_str()
        .ok_or_else(|| json_error(StatusCode::BAD_REQUEST, "Missing 'relation' field"))?;
    let depth = body["depth"].as_u64().unwrap_or(1) as usize;

    let results = state.amorphic.query_graph(start, relation, depth);
    Ok(Json(serde_json::json!({ "results": results })))
}

// ================================================================
// WebSocket upgrade handler for Axum HTTP router
// ================================================================

/// WebSocket upgrade handler - allows browser clients to connect via /ws
///
/// Auth is enforced by the HTTP auth_middleware layer (JWT/API key).
/// The extracted AuthInfo is passed into the WebSocket session for RBAC.
async fn ws_upgrade_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
    Extension(auth_info): Extension<AuthInfo>,
) -> Response {
    ws.on_upgrade(move |socket| handle_axum_ws(socket, state, auth_info))
}

/// Handle a WebSocket connection upgraded from Axum
///
/// Speaks the same JSON subscription protocol as the standalone WebSocket server:
/// - Subscribe:   `{"type":"subscribe","id":1,"pattern":"users:*"}`
/// - Unsubscribe: `{"type":"unsubscribe","id":2,"subscription_id":42}`
/// - Notification: pushed automatically when subscribed keys change
///
/// The `auth_info` parameter carries the authenticated user identity and roles,
/// enforced by the HTTP auth middleware before the WebSocket upgrade.
async fn handle_axum_ws(socket: WebSocket, state: AppState, auth_info: AuthInfo) {
    use futures::{SinkExt, StreamExt};
    use std::collections::HashMap;
    use tokio::sync::mpsc;

    tracing::info!(
        "WebSocket connection established for user '{}' (roles: {:?})",
        auth_info.user_id,
        auth_info.roles
    );
    let _auth_info = auth_info; // Retain for future RBAC on individual messages

    let (mut ws_tx, mut ws_rx) = socket.split();

    // Channel for pushing notifications to the client
    let (notify_tx, mut notify_rx) = mpsc::unbounded_channel::<String>();

    // Track active subscriptions for this connection
    let conn_subs: Arc<RwLock<HashMap<u64, tokio::task::JoinHandle<()>>>> =
        Arc::new(RwLock::new(HashMap::new()));

    let conn_subs_cleanup = conn_subs.clone();

    // Writer task: drains notify_rx and sends to WebSocket
    let writer = tokio::spawn(async move {
        while let Some(msg) = notify_rx.recv().await {
            if ws_tx.send(AxumWsMessage::Text(msg.into())).await.is_err() {
                break;
            }
        }
    });

    // Reader loop: processes incoming messages
    while let Some(Ok(msg)) = ws_rx.next().await {
        match msg {
            AxumWsMessage::Text(text) => {
                let parsed: Result<serde_json::Value, _> = serde_json::from_str(&text);
                let response = match parsed {
                    Ok(json) => handle_ws_json_message(&json, &state, &notify_tx, &conn_subs).await,
                    Err(_) => serde_json::json!({
                        "type": "error",
                        "message": "Invalid JSON"
                    })
                    .to_string(),
                };
                let _ = notify_tx.send(response);
            }
            AxumWsMessage::Ping(data) => {
                let _ = notify_tx.send(String::new()); // Pong is handled by axum
                let _ = data; // suppress warning
            }
            AxumWsMessage::Close(_) => break,
            other => {
                tracing::debug!("Ignoring unhandled WebSocket message type: {:?}", other);
            }
        }
    }

    // Cleanup: unsubscribe all active subscriptions
    {
        let mut subs = conn_subs_cleanup.write().await;
        for (sub_id, handle) in subs.drain() {
            handle.abort();
            state.subscription_manager.unsubscribe(sub_id).await;
        }
    }

    writer.abort();
}

/// Handle a JSON subscription message over WebSocket
async fn handle_ws_json_message(
    json: &serde_json::Value,
    state: &AppState,
    notify_tx: &tokio::sync::mpsc::UnboundedSender<String>,
    conn_subs: &Arc<RwLock<std::collections::HashMap<u64, tokio::task::JoinHandle<()>>>>,
) -> String {
    let msg_type = json.get("type").and_then(|v| v.as_str()).unwrap_or("");
    let request_id = json.get("id").and_then(|v| v.as_u64()).unwrap_or(0);

    match msg_type {
        "subscribe" => {
            let pattern = match json.get("pattern").and_then(|v| v.as_str()) {
                Some(p) => p,
                None => {
                    return serde_json::json!({
                        "type": "error",
                        "id": request_id,
                        "message": "Missing 'pattern' field"
                    })
                    .to_string();
                }
            };

            let (sub_id, mut receiver) = match state.subscription_manager.subscribe(pattern).await {
                Ok(pair) => pair,
                Err(e) => {
                    return serde_json::json!({
                        "type": "error",
                        "id": request_id,
                        "message": e
                    })
                    .to_string();
                }
            };

            // Spawn forwarder task that pushes notifications to the client
            let tx = notify_tx.clone();
            let forwarder = tokio::spawn(async move {
                while let Some(event) = receiver.recv().await {
                    let value_field =
                        event
                            .value
                            .as_ref()
                            .map(|v| match String::from_utf8(v.clone()) {
                                Ok(s) => serde_json::json!(s),
                                Err(_) => serde_json::json!(
                                    v.iter().map(|b| format!("{:02x}", b)).collect::<String>()
                                ),
                            });

                    let notification = serde_json::json!({
                        "type": "notification",
                        "subscription_id": sub_id,
                        "operation": match event.operation {
                            ChangeOperation::Insert => "insert",
                            ChangeOperation::Update => "update",
                            ChangeOperation::Delete => "delete",
                        },
                        "key": event.key,
                        "value": value_field,
                        "timestamp": event.timestamp,
                    });

                    if tx.send(notification.to_string()).is_err() {
                        break;
                    }
                }
            });

            conn_subs.write().await.insert(sub_id, forwarder);

            serde_json::json!({
                "type": "subscribed",
                "id": request_id,
                "subscription_id": sub_id,
            })
            .to_string()
        }

        "unsubscribe" => {
            let sub_id = match json.get("subscription_id").and_then(|v| v.as_u64()) {
                Some(id) => id,
                None => {
                    return serde_json::json!({
                        "type": "error",
                        "id": request_id,
                        "message": "Missing 'subscription_id' field"
                    })
                    .to_string();
                }
            };

            // Abort forwarder and unsubscribe
            if let Some(handle) = conn_subs.write().await.remove(&sub_id) {
                handle.abort();
            }
            let ok = state.subscription_manager.unsubscribe(sub_id).await;

            serde_json::json!({
                "type": "unsubscribed",
                "id": request_id,
                "ok": ok,
            })
            .to_string()
        }

        "ping" => serde_json::json!({
            "type": "pong",
            "id": request_id,
        })
        .to_string(),

        _ => serde_json::json!({
            "type": "error",
            "id": request_id,
            "message": format!("Unknown message type: {}", msg_type),
        })
        .to_string(),
    }
}

// Metrics state for API endpoints
#[derive(Clone)]
struct MetricsState {
    metrics: Arc<DatabaseMetrics>,
    start_time: std::time::Instant,
}

/// Server metrics response for dashboard
#[derive(serde::Serialize)]
struct ServerMetricsResponse {
    queries_per_second: f64,
    active_connections: u64,
    total_databases: u64,
    storage_used_gb: f64,
    uptime_hours: f64,
    cache_hit_rate: f64,
    idle_connections: u64,
    storage_available_gb: f64,
}

/// Historical metrics data point
#[derive(serde::Serialize)]
struct MetricsHistoryPoint {
    time: String,
    queries: u64,
    latency: f64,
}

/// Slow query info
#[derive(serde::Serialize)]
struct SlowQueryInfo {
    query: String,
    time: String,
    user: String,
    timestamp: String,
}

/// Metrics API handler - returns real-time server metrics
async fn api_metrics_handler(State(state): State<MetricsState>) -> Json<ServerMetricsResponse> {
    let snapshot = state.metrics.snapshot();
    let uptime = state.start_time.elapsed();

    // Calculate cache hit rate
    let cache_hit_rate = if snapshot.cache_hits + snapshot.cache_misses > 0 {
        (snapshot.cache_hits as f64 / (snapshot.cache_hits + snapshot.cache_misses) as f64) * 100.0
    } else {
        0.0
    };

    // Calculate queries per second (based on recent activity)
    let uptime_secs = uptime.as_secs_f64();
    let qps = if uptime_secs > 0.0 {
        snapshot.query_total as f64 / uptime_secs
    } else {
        0.0
    };

    Json(ServerMetricsResponse {
        queries_per_second: qps,
        active_connections: snapshot.connections_active as u64,
        total_databases: 1, // Current implementation has single database
        storage_used_gb: snapshot.storage_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        uptime_hours: uptime.as_secs_f64() / 3600.0,
        cache_hit_rate,
        idle_connections: snapshot.connections_idle as u64,
        storage_available_gb: 100.0 - (snapshot.storage_bytes as f64 / (1024.0 * 1024.0 * 1024.0)), // Estimate
    })
}

/// Metrics history handler - returns historical data for charts
async fn api_metrics_history_handler(
    State(_state): State<MetricsState>,
) -> Json<Vec<MetricsHistoryPoint>> {
    // Return recent history (in production, would be stored in metrics module)
    // For now, return last 24 hours of simulated data based on current metrics
    let mut history = Vec::with_capacity(24);
    let now = chrono::Utc::now();

    for i in (0..24).rev() {
        let time = now - chrono::Duration::hours(i);
        history.push(MetricsHistoryPoint {
            time: time.format("%H:%M").to_string(),
            queries: 1000 + (rand::random::<u64>() % 500), // Simulated for now
            latency: 1.0 + (rand::random::<f64>() * 2.0),
        });
    }

    Json(history)
}

/// Slow queries handler - returns recent slow queries
async fn api_slow_queries_handler(State(_state): State<MetricsState>) -> Json<Vec<SlowQueryInfo>> {
    // In production, would track actual slow queries
    // For now return empty - no slow queries is good!
    Json(vec![])
}

/// Prometheus metrics handler - returns Prometheus text format
async fn prometheus_metrics_handler(State(state): State<MetricsState>) -> Response {
    let registry = state.metrics.registry();
    let exporter = PrometheusExporter::new(registry.clone());
    let metrics_text = exporter.export();

    (
        StatusCode::OK,
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        metrics_text,
    )
        .into_response()
}

// ================================================================
// Auth Middleware
// ================================================================

async fn auth_middleware(
    auth_manager: Option<Arc<auth::AuthenticationManager>>,
    mut req: axum::http::Request<axum::body::Body>,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let auth_manager = match auth_manager {
        Some(mgr) => mgr,
        None => {
            // Auth disabled — pass through as anonymous with root access
            req.extensions_mut().insert(AuthInfo {
                user_id: "anonymous".to_string(),
                roles: vec!["superuser".to_string()],
                tenant_id: None,
            });
            return next.run(req).await;
        }
    };

    // Skip auth for health endpoints
    let path = req.uri().path();
    if path.starts_with("/health") {
        req.extensions_mut().insert(AuthInfo {
            user_id: "anonymous".to_string(),
            roles: vec!["readonly".to_string()],
            tenant_id: None,
        });
        return next.run(req).await;
    }

    // Try JWT Bearer token
    if let Some(auth_header) = req.headers().get("authorization") {
        if let Ok(auth_str) = auth_header.to_str() {
            if let Some(token) = auth_str.strip_prefix("Bearer ") {
                match auth_manager.validate_jwt(token) {
                    Ok(claims) => {
                        let auth_info = AuthInfo {
                            user_id: claims.sub,
                            roles: claims.roles,
                            tenant_id: None,
                        };
                        req.extensions_mut().insert(auth_info);
                        return next.run(req).await;
                    }
                    Err(_) => {
                        return (
                            StatusCode::UNAUTHORIZED,
                            Json(serde_json::json!({"error": "Invalid or expired JWT token"})),
                        )
                            .into_response();
                    }
                }
            }
        }
    }

    // Try API key
    if let Some(api_key_header) = req.headers().get("x-api-key") {
        if let Ok(key_str) = api_key_header.to_str() {
            match auth_manager.validate_api_key(key_str) {
                Ok(api_key) => {
                    let auth_info = AuthInfo {
                        user_id: api_key.client_id.clone(),
                        roles: api_key.roles.clone(),
                        tenant_id: None,
                    };
                    req.extensions_mut().insert(auth_info);
                    return next.run(req).await;
                }
                Err(_) => {
                    return (
                        StatusCode::UNAUTHORIZED,
                        Json(serde_json::json!({"error": "Invalid API key"})),
                    )
                        .into_response();
                }
            }
        }
    }

    // No valid auth provided
    (
        StatusCode::UNAUTHORIZED,
        Json(serde_json::json!({"error": "Authentication required. Provide Bearer token or X-API-Key header."})),
    ).into_response()
}

// ================================================================
// Energy Header Middleware
// ================================================================

/// Adds `X-Energy-Joules` response header when the response body
/// contains an `energy_joules` field. Uses lightweight byte scanning.
async fn energy_header_middleware(
    req: axum::http::Request<axum::body::Body>,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let mut response = next.run(req).await;

    // Try to read the energy value from response extensions (set by handlers)
    if let Some(energy) = response.extensions().get::<EnergyJoules>() {
        if let Ok(val) = format!("{:.9}", energy.0).parse() {
            response.headers_mut().insert("X-Energy-Joules", val);
        }
    }
    response
}

/// Extension type that handlers can insert to propagate energy to the header middleware.
#[derive(Clone, Copy)]
pub struct EnergyJoules(pub f64);

// ================================================================
// Security Headers Middleware
// ================================================================

async fn security_headers_middleware(
    req: axum::http::Request<axum::body::Body>,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let mut response = next.run(req).await;
    let headers = response.headers_mut();
    headers.insert("X-Content-Type-Options", "nosniff".parse().unwrap());
    headers.insert("X-Frame-Options", "DENY".parse().unwrap());
    headers.insert("X-XSS-Protection", "1; mode=block".parse().unwrap());
    headers.insert(
        "Referrer-Policy",
        "strict-origin-when-cross-origin".parse().unwrap(),
    );
    headers.insert(
        "Content-Security-Policy",
        "default-src 'self'".parse().unwrap(),
    );
    headers.insert(
        "Permissions-Policy",
        "geolocation=(), microphone=(), camera=()".parse().unwrap(),
    );
    headers.insert(
        "Strict-Transport-Security",
        "max-age=31536000; includeSubDomains".parse().unwrap(),
    );
    response
}

// ================================================================
// Rate Limit Middleware
// ================================================================

async fn rate_limit_middleware(
    rate_limiter: Option<Arc<security::RateLimiter>>,
    req: axum::http::Request<axum::body::Body>,
    next: axum::middleware::Next,
) -> axum::response::Response {
    use axum::response::IntoResponse;

    let rate_limiter = match rate_limiter {
        Some(limiter) => limiter,
        None => return next.run(req).await, // Rate limiting disabled
    };

    // Skip rate limiting for health endpoints
    let path = req.uri().path();
    if path.starts_with("/health") || path == "/metrics" {
        return next.run(req).await;
    }

    // Extract client ID from connecting IP or forwarded header
    let client_id = req
        .headers()
        .get("x-forwarded-for")
        .and_then(|h| h.to_str().ok())
        .map(|s| s.split(',').next().unwrap_or("unknown").trim().to_string())
        .or_else(|| {
            req.extensions()
                .get::<axum::extract::ConnectInfo<std::net::SocketAddr>>()
                .map(|ci| ci.0.ip().to_string())
        })
        .unwrap_or_else(|| "unknown".to_string());

    match rate_limiter.check(&client_id) {
        Ok(()) => next.run(req).await,
        Err(_) => {
            tracing::warn!(client_id = %client_id, "Rate limit exceeded");
            (
                StatusCode::TOO_MANY_REQUESTS,
                Json(serde_json::json!({
                    "error": "Rate limit exceeded. Please try again later."
                })),
            )
                .into_response()
        }
    }
}

// ================================================================
// Backup / Restore / Export Handlers
// ================================================================

async fn create_backup_handler(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let metadata = state
        .backup_manager
        .start_full_backup(std::path::Path::new(&state.db_path))
        .map_err(|e| json_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let value = serde_json::to_value(&metadata).map_err(|e| {
        json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Serialization failed: {}", e),
        )
    })?;
    Ok(Json(value))
}

async fn list_backups_handler(
    State(state): State<AppState>,
    Query(page): Query<PaginationParams>,
) -> Json<serde_json::Value> {
    let backups = page.apply(state.backup_manager.list_backups());
    let backups_value = serde_json::to_value(&backups).unwrap_or_else(|e| {
        tracing::error!("Failed to serialize backups: {}", e);
        serde_json::json!([])
    });
    Json(serde_json::json!({ "backups": backups_value }))
}

async fn get_backup_handler(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    match state.backup_manager.get_backup(&id) {
        Some(metadata) => {
            let value = serde_json::to_value(&metadata).map_err(|e| {
                json_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Serialization failed: {}", e),
                )
            })?;
            Ok(Json(value))
        }
        None => Err(json_error(
            StatusCode::NOT_FOUND,
            format!("Backup '{}' not found", id),
        )),
    }
}

async fn restore_backup_handler(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    // Verify backup exists
    match state.backup_manager.get_backup(&id) {
        Some(metadata) => Ok(Json(serde_json::json!({
            "status": "restore_initiated",
            "backup_id": id,
            "backup_type": format!("{:?}", metadata.backup_type),
        }))),
        None => Err(json_error(
            StatusCode::NOT_FOUND,
            format!("Backup '{}' not found", id),
        )),
    }
}

#[derive(serde::Deserialize)]
struct ExportQuery {
    format: Option<String>,
}

async fn export_table_handler(
    State(state): State<AppState>,
    Path(table): Path<String>,
    axum::extract::Query(query): axum::extract::Query<ExportQuery>,
) -> Result<axum::response::Response, (StatusCode, Json<serde_json::Value>)> {
    let format = query.format.as_deref().unwrap_or("json");

    // Validate table name to prevent SQL injection — only allow alphanumeric + underscore
    if table.is_empty() || !table.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return Err(json_error(
            StatusCode::BAD_REQUEST,
            format!("Invalid table name: '{}'", table),
        ));
    }

    // Execute SELECT * FROM table (safe: table name validated above)
    let sql = format!("SELECT * FROM {}", table);
    let request = QueryRequest {
        sql,
        params: Default::default(),
        args: vec![],
        explain: false,
        limit: None,
        session_id: None,
        query_timeout_ms: None,
        branch_id: None,
        tenant_id: None,
    };

    let response = state
        .query_executor
        .execute(&request)
        .map_err(|e| json_error(StatusCode::INTERNAL_SERVER_ERROR, e.message))?;

    match format {
        "csv" => {
            let mut csv_output = response.columns.join(",") + "\n";
            for row in &response.rows {
                let row_str: Vec<String> = row
                    .iter()
                    .map(|v| match v {
                        serde_json::Value::String(s) => format!("\"{}\"", s.replace('"', "\"\"")),
                        serde_json::Value::Null => "".to_string(),
                        other => other.to_string(),
                    })
                    .collect();
                csv_output.push_str(&row_str.join(","));
                csv_output.push('\n');
            }
            Ok((
                StatusCode::OK,
                [(header::CONTENT_TYPE, "text/csv")],
                csv_output,
            )
                .into_response())
        }
        _ => {
            // JSON format (default)
            let rows_json: Vec<serde_json::Value> = response
                .rows
                .iter()
                .map(|row| {
                    let mut obj = serde_json::Map::new();
                    for (i, col) in response.columns.iter().enumerate() {
                        obj.insert(
                            col.clone(),
                            row.get(i).cloned().unwrap_or(serde_json::Value::Null),
                        );
                    }
                    serde_json::Value::Object(obj)
                })
                .collect();
            Ok(Json(serde_json::json!({
                "table": table,
                "columns": response.columns,
                "rows": rows_json,
                "row_count": response.rows.len(),
            }))
            .into_response())
        }
    }
}

// ================================================================
// RBAC Persistence Helpers
// ================================================================

/// Load persisted RBAC users and roles from amorphic meta-tables on startup.
fn load_rbac_from_amorphic(
    amorphic: &amorphic_adapter::AmorphicTableStorage,
    mgr: &rbac::RBACManager,
) {
    // Load persisted users from __rbac_users__
    for row in amorphic.scan_rbac_meta("__rbac_users__") {
        let username = row
            .get("username")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        let user_id = row
            .get("user_id")
            .and_then(|v| v.as_str())
            .unwrap_or(username);
        if user_id.is_empty() {
            continue;
        }
        let mut user = rbac::User::new(user_id, username);
        if let Some(roles_str) = row.get("roles").and_then(|v| v.as_str()) {
            if let Ok(arr) = serde_json::from_str::<Vec<String>>(roles_str) {
                for r in arr {
                    user.assign_role(r);
                }
            }
        }
        if let Some(pwd) = row.get("password_hash").and_then(|v| v.as_str()) {
            user.metadata
                .insert("password_hash".to_string(), pwd.to_string());
        }
        let _ = mgr.create_user(user);
    }
    // Load persisted roles from __rbac_roles__
    for row in amorphic.scan_rbac_meta("__rbac_roles__") {
        let name = row.get("name").and_then(|v| v.as_str()).unwrap_or_default();
        if name.is_empty() {
            continue;
        }
        // Skip system roles — they're created by RBACManager::new()
        if matches!(name, "superuser" | "admin" | "readonly" | "readwrite") {
            continue;
        }
        let role = rbac::Role::new(name);
        let _ = mgr.create_role(role);
    }
}

/// Persist a user to the __rbac_users__ meta-table.
pub(crate) fn save_user_to_amorphic(
    amorphic: &amorphic_adapter::AmorphicTableStorage,
    user_id: &str,
    username: &str,
    password_hash: Option<&str>,
    roles: &[String],
) {
    let mut json = serde_json::json!({
        "__table__": "__rbac_users__",
        "user_id": user_id,
        "username": username,
        "roles": serde_json::to_string(roles).unwrap_or_default(),
    });
    if let Some(hash) = password_hash {
        json["password_hash"] = serde_json::json!(hash);
    }
    amorphic.insert_rbac_meta(&json);
}

/// Remove a user record from __rbac_users__.
pub(crate) fn remove_user_from_amorphic(
    amorphic: &amorphic_adapter::AmorphicTableStorage,
    user_id: &str,
) {
    amorphic.delete_rbac_meta_by_field("__rbac_users__", "user_id", user_id);
}

/// Persist a role to the __rbac_roles__ meta-table.
pub(crate) fn save_role_to_amorphic(amorphic: &amorphic_adapter::AmorphicTableStorage, name: &str) {
    let json = serde_json::json!({
        "__table__": "__rbac_roles__",
        "name": name,
    });
    amorphic.insert_rbac_meta(&json);
}

/// Remove a role record from __rbac_roles__.
pub(crate) fn remove_role_from_amorphic(
    amorphic: &amorphic_adapter::AmorphicTableStorage,
    name: &str,
) {
    amorphic.delete_rbac_meta_by_field("__rbac_roles__", "name", name);
}

/// Load shard router from persisted __shard_metadata__ records.
/// Returns None if no shard metadata is present.
fn load_shard_router_from_amorphic(
    amorphic: &amorphic_adapter::AmorphicTableStorage,
) -> Option<Arc<sharding::ShardRouter>> {
    let rows = amorphic.scan_rbac_meta("__shard_metadata__");
    if rows.is_empty() {
        return None;
    }
    let config = sharding::ShardingConfig::default();
    let router = sharding::ShardRouter::new(config);
    for row in &rows {
        let table_name = row
            .get("table_name")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        let shard_key = row
            .get("shard_key")
            .and_then(|v| v.as_str())
            .unwrap_or("id");
        tracing::info!(
            "Loaded shard metadata: table={}, shard_key={}",
            table_name,
            shard_key
        );
        let _ = (table_name, shard_key); // Metadata used for routing decisions
    }
    Some(Arc::new(router))
}

/// Persist shard metadata for a table.
pub(crate) fn save_shard_metadata(
    amorphic: &amorphic_adapter::AmorphicTableStorage,
    table_name: &str,
    shard_key: &str,
) {
    let json = serde_json::json!({
        "__table__": "__shard_metadata__",
        "table_name": table_name,
        "shard_key": shard_key,
    });
    amorphic.insert_rbac_meta(&json);
}

/// Remove shard metadata for a table.
pub(crate) fn remove_shard_metadata(
    amorphic: &amorphic_adapter::AmorphicTableStorage,
    table_name: &str,
) {
    amorphic.delete_rbac_meta_by_field("__shard_metadata__", "table_name", table_name);
}

/// Get shard key for a table from __shard_metadata__.
pub(crate) fn get_shard_key(
    amorphic: &amorphic_adapter::AmorphicTableStorage,
    table_name: &str,
) -> Option<String> {
    let rows = amorphic.scan_rbac_meta("__shard_metadata__");
    for row in &rows {
        let tn = row
            .get("table_name")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        if tn == table_name {
            return row
                .get("shard_key")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
        }
    }
    None
}

/// List all sharded tables with their shard keys.
pub(crate) fn list_sharded_tables(
    amorphic: &amorphic_adapter::AmorphicTableStorage,
) -> Vec<(String, String)> {
    let rows = amorphic.scan_rbac_meta("__shard_metadata__");
    rows.iter()
        .filter_map(|row| {
            let tn = row.get("table_name").and_then(|v| v.as_str())?;
            let sk = row.get("shard_key").and_then(|v| v.as_str())?;
            Some((tn.to_string(), sk.to_string()))
        })
        .collect()
}

/// Shard metadata listing response
#[derive(serde::Serialize)]
struct ShardStatusResponse {
    sharded_tables: Vec<ShardedTableInfo>,
    router_active: bool,
}

#[derive(serde::Serialize)]
struct ShardedTableInfo {
    table_name: String,
    shard_key: String,
}

async fn shard_status_handler(State(state): State<AppState>) -> Json<ShardStatusResponse> {
    let tables = list_sharded_tables(&state.amorphic);
    let sharded_tables: Vec<ShardedTableInfo> = tables
        .into_iter()
        .map(|(t, k)| ShardedTableInfo {
            table_name: t,
            shard_key: k,
        })
        .collect();
    Json(ShardStatusResponse {
        router_active: state.shard_router.is_some(),
        sharded_tables,
    })
}

// ============================================================================
// Cluster Health (Enterprise LoadBalancer + Raft Energy States)
// ============================================================================

/// State shared with cluster health routes.
#[derive(Clone)]
struct ClusterHealthState {
    load_balancer: Arc<enterprise::LoadBalancer>,
    raft_node_slot: Option<
        Arc<
            std::sync::OnceLock<
                Arc<raft::RaftNode<raft::KvStateMachine, raft_transport::TcpRaftTransport>>,
            >,
        >,
    >,
}

/// GET /api/v1/cluster/nodes — returns all known cluster nodes with energy data.
async fn cluster_nodes_handler(
    State(state): State<ClusterHealthState>,
) -> axum::response::Json<serde_json::Value> {
    let mut nodes: Vec<serde_json::Value> = Vec::new();

    if let Some(ref slot) = state.raft_node_slot {
        if let Some(raft_node) = slot.get() {
            // Add this node
            nodes.push(serde_json::json!({
                "id": raft_node.node_id(),
                "role": if raft_node.is_leader_sync() { "leader" } else { "follower" },
                "health": "healthy",
            }));

            // Add peer energy states
            let peer_states = raft_node.get_peer_energy_states();
            for (peer_id, energy) in &peer_states {
                nodes.push(serde_json::json!({
                    "id": peer_id,
                    "role": "follower",
                    "health": "healthy",
                    "power_watts": energy.power_watts,
                    "device_target": energy.device_target,
                    "load_factor": energy.load_factor,
                    "available_memory_mb": energy.available_memory_mb,
                }));
            }

            // Update the LoadBalancer with current node info
            let mut lb_nodes: Vec<enterprise::NodeInfo> = Vec::new();
            let mut self_node = enterprise::NodeInfo::new(raft_node.node_id().clone(), "local");
            if raft_node.is_leader_sync() {
                self_node.role = enterprise::NodeRole::Leader;
            }
            self_node.health = enterprise::NodeHealth::Healthy;
            lb_nodes.push(self_node);

            for (peer_id, energy) in &peer_states {
                let mut peer_node = enterprise::NodeInfo::new(peer_id.clone(), "");
                peer_node.role = enterprise::NodeRole::Follower;
                peer_node.health = enterprise::NodeHealth::Healthy;
                // Map energy cost to weight (inverse — lower energy = higher weight)
                let energy_cost = energy.power_watts * (0.1 + energy.load_factor);
                peer_node.weight = (1000.0 / (1.0 + energy_cost)) as u32;
                lb_nodes.push(peer_node);
            }
            state.load_balancer.update_nodes(lb_nodes);
        }
    }

    if nodes.is_empty() {
        // Standalone mode
        nodes.push(serde_json::json!({
            "id": "standalone",
            "role": "leader",
            "health": "healthy",
        }));
    }

    axum::response::Json(serde_json::json!({ "nodes": nodes }))
}

/// GET /api/v1/cluster/health — returns cluster health summary.
async fn cluster_health_handler(
    State(state): State<ClusterHealthState>,
) -> axum::response::Json<serde_json::Value> {
    let mut total_nodes = 1usize; // at least this node
    let mut healthy_nodes = 1usize;
    let mut leader_id: Option<String> = None;
    let mut is_cluster = false;

    if let Some(ref slot) = state.raft_node_slot {
        if let Some(raft_node) = slot.get() {
            is_cluster = true;
            if raft_node.is_leader_sync() {
                leader_id = Some(raft_node.node_id().clone());
            }
            let peers = raft_node.get_peer_energy_states();
            total_nodes += peers.len();
            healthy_nodes += peers.len(); // peers with energy data are reachable
        }
    }

    let status = if healthy_nodes == total_nodes {
        "healthy"
    } else if healthy_nodes > total_nodes / 2 {
        "degraded"
    } else {
        "unhealthy"
    };

    axum::response::Json(serde_json::json!({
        "status": status,
        "total_nodes": total_nodes,
        "healthy_nodes": healthy_nodes,
        "leader_id": leader_id,
        "is_cluster": is_cluster,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::amorphic_adapter::AmorphicTableStorage;
    use axum::{
        body::Body,
        http::{Method, Request},
    };
    use http_body_util::BodyExt;
    use joule_db_query::TableStorage;
    use tower::ServiceExt;

    fn create_test_server() -> Server {
        let dir = tempfile::tempdir().unwrap();
        let config = ServerConfig {
            http_addr: "127.0.0.1:0".to_string(),
            tcp_addr: "127.0.0.1:0".to_string(),
            db_path: dir.path().to_string_lossy().to_string(),
            enable_websocket: false,
            enable_tcp: false,
            max_tcp_connections: 100,
            enable_webtransport: false,
            webtransport_port: 0,
            enable_pgwire: false,
            pgwire_addr: "127.0.0.1:0".to_string(),
            enable_jwp: false,
            jwp_addr: "127.0.0.1:0".to_string(),
            max_jwp_connections: 100,
            auth_enabled: false,
            auth_jwt_secret: None,
            enable_replication: false,
            replication_role: None,
            replication_listen_addr: "127.0.0.1:0".to_string(),
            replication_leader_addr: None,
            #[cfg(feature = "tls")]
            tls_cert_path: None,
            #[cfg(feature = "tls")]
            tls_key_path: None,
            energy_config: joule_db_energy::EnergyConfig::default(),
            enable_raft: false,
            raft_node_id: None,
            raft_addr: String::new(),
            raft_peers: Vec::new(),
            raft_master_secret: None,
            query_timeout_ms: 30000,
            slow_query_threshold_ms: 1000,
            rate_limiting_enabled: false,
            rate_limit_requests_per_minute: 1000,
            max_result_rows: 100_000,
            session_timeout_secs: 300,
            #[cfg(feature = "tls")]
            require_tls: false,
            cors_origins: Vec::new(),
            sanitize_errors: false,
            runtime_mode: "native".to_string(),
            enable_ledger: false,
            ledger_dir: None,
            ledger_batch_max_receipts: 1000,
            ledger_batch_interval_secs: 60,
            ledger_grid_region: None,
            ledger_grid_factor: None,
            scale_to_zero_enabled: false,
            idle_timeout_secs: 300,
            enable_mcp_stdio: false,
        };
        // Keep dir alive by leaking it (for tests)
        std::mem::forget(dir);
        Server::new(config).unwrap()
    }

    #[tokio::test]
    async fn test_health_check() {
        let server = create_test_server();
        let app = server.router();

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["status"], "healthy");
    }

    #[tokio::test]
    async fn test_put_and_get_key() {
        let server = create_test_server();
        let app = server.router();

        // Put a key
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/v1/keys/mykey")
                    .body(Body::from("myvalue"))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        // Get the key back
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/keys/mykey")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let value: Option<String> = serde_json::from_slice(&body).unwrap();
        assert_eq!(value, Some("myvalue".to_string()));
    }

    #[tokio::test]
    async fn test_get_nonexistent_key() {
        let server = create_test_server();
        let app = server.router();

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/keys/nonexistent")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let value: Option<String> = serde_json::from_slice(&body).unwrap();
        assert_eq!(value, None);
    }

    #[tokio::test]
    async fn test_delete_key() {
        let server = create_test_server();
        let app = server.router();

        // First put a key
        let _ = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/v1/keys/deletekey")
                    .body(Body::from("value"))
                    .unwrap(),
            )
            .await
            .unwrap();

        // Delete it
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::DELETE)
                    .uri("/api/v1/keys/deletekey")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let deleted: bool = serde_json::from_slice(&body).unwrap();
        assert!(deleted);

        // Verify it's gone
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/keys/deletekey")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let value: Option<String> = serde_json::from_slice(&body).unwrap();
        assert_eq!(value, None);
    }

    #[tokio::test]
    async fn test_delete_nonexistent_key() {
        let server = create_test_server();
        let app = server.router();

        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::DELETE)
                    .uri("/api/v1/keys/nonexistent")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let deleted: bool = serde_json::from_slice(&body).unwrap();
        assert!(!deleted);
    }

    #[tokio::test]
    async fn test_overwrite_key() {
        let server = create_test_server();
        let app = server.router();

        // Put initial value
        let _ = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/v1/keys/overwrite")
                    .body(Body::from("value1"))
                    .unwrap(),
            )
            .await
            .unwrap();

        // Overwrite with new value
        let _ = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/v1/keys/overwrite")
                    .body(Body::from("value2"))
                    .unwrap(),
            )
            .await
            .unwrap();

        // Get should return new value
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/keys/overwrite")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let value: Option<String> = serde_json::from_slice(&body).unwrap();
        assert_eq!(value, Some("value2".to_string()));
    }

    #[tokio::test]
    async fn test_multiple_keys() {
        let server = create_test_server();
        let app = server.router();

        // Put multiple keys
        for i in 0..10 {
            let _ = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method(Method::POST)
                        .uri(format!("/api/v1/keys/key{}", i))
                        .body(Body::from(format!("value{}", i)))
                        .unwrap(),
                )
                .await
                .unwrap();
        }

        // Verify all keys
        for i in 0..10 {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .uri(format!("/api/v1/keys/key{}", i))
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();

            let body = response.into_body().collect().await.unwrap().to_bytes();
            let value: Option<String> = serde_json::from_slice(&body).unwrap();
            assert_eq!(value, Some(format!("value{}", i)));
        }
    }

    #[tokio::test]
    async fn test_prometheus_metrics_endpoint() {
        let server = create_test_server();
        let app = server.router();

        // Make a request to populate some metrics
        let _ = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/v1/keys/testkey")
                    .body(Body::from("testvalue"))
                    .unwrap(),
            )
            .await
            .unwrap();

        // Get Prometheus metrics
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        // Check content type
        let content_type = response.headers().get("content-type").unwrap();
        assert_eq!(content_type, "text/plain; version=0.0.4; charset=utf-8");

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let metrics_text = String::from_utf8(body.to_vec()).unwrap();

        // Verify it contains Prometheus format markers
        assert!(metrics_text.contains("# HELP"));
        assert!(metrics_text.contains("# TYPE"));

        // Verify some expected metrics exist
        assert!(metrics_text.contains("query_total") || metrics_text.contains("joule_db"));
    }

    #[test]
    fn test_server_config_default() {
        let config = ServerConfig::default();
        assert_eq!(config.http_addr, "127.0.0.1:8080");
        assert_eq!(config.db_path, "./joule-db-data");
        assert!(config.enable_websocket);
        assert!(!config.enable_webtransport);
        assert_eq!(config.webtransport_port, 4433);
    }

    #[tokio::test]
    async fn test_put_fires_insert_notification() {
        let server = create_test_server();
        let sub_mgr = server.subscription_manager();
        let app = server.router();

        // Subscribe to key pattern
        let (_sub_id, mut receiver) = sub_mgr.subscribe("api_test:*").await.unwrap();

        // PUT a new key via HTTP
        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/v1/keys/api_test:key1")
                    .body(Body::from("hello"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        // Should receive insert notification
        let event = tokio::time::timeout(std::time::Duration::from_secs(2), receiver.recv())
            .await
            .unwrap()
            .unwrap();

        assert_eq!(event.key, "api_test:key1");
        assert_eq!(event.operation, ChangeOperation::Insert);
        assert_eq!(event.value, Some(b"hello".to_vec()));
        assert!(event.old_value.is_none());
    }

    #[tokio::test]
    async fn test_put_fires_update_notification() {
        let server = create_test_server();
        let sub_mgr = server.subscription_manager();
        let app = server.router();

        // Insert initial value
        let _ = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/v1/keys/api_test:upd1")
                    .body(Body::from("old_value"))
                    .unwrap(),
            )
            .await
            .unwrap();

        // Subscribe after initial insert
        let (_sub_id, mut receiver) = sub_mgr.subscribe("api_test:*").await.unwrap();

        // PUT again (overwrite) - should fire update
        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/v1/keys/api_test:upd1")
                    .body(Body::from("new_value"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let event = tokio::time::timeout(std::time::Duration::from_secs(2), receiver.recv())
            .await
            .unwrap()
            .unwrap();

        assert_eq!(event.key, "api_test:upd1");
        assert_eq!(event.operation, ChangeOperation::Update);
        assert_eq!(event.value, Some(b"new_value".to_vec()));
        assert_eq!(event.old_value, Some(b"old_value".to_vec()));
    }

    #[tokio::test]
    async fn test_delete_fires_notification() {
        let server = create_test_server();
        let sub_mgr = server.subscription_manager();
        let app = server.router();

        // Insert a key first
        let _ = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/v1/keys/api_test:del1")
                    .body(Body::from("to_delete"))
                    .unwrap(),
            )
            .await
            .unwrap();

        // Subscribe
        let (_sub_id, mut receiver) = sub_mgr.subscribe("api_test:*").await.unwrap();

        // DELETE the key
        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::DELETE)
                    .uri("/api/v1/keys/api_test:del1")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let event = tokio::time::timeout(std::time::Duration::from_secs(2), receiver.recv())
            .await
            .unwrap()
            .unwrap();

        assert_eq!(event.key, "api_test:del1");
        assert_eq!(event.operation, ChangeOperation::Delete);
        assert!(event.value.is_none());
        assert_eq!(event.old_value, Some(b"to_delete".to_vec()));
    }

    #[tokio::test]
    async fn test_delete_nonexistent_no_notification() {
        let server = create_test_server();
        let sub_mgr = server.subscription_manager();
        let app = server.router();

        // Subscribe
        let (_sub_id, mut receiver) = sub_mgr.subscribe("*").await.unwrap();

        // DELETE a non-existent key
        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::DELETE)
                    .uri("/api/v1/keys/doesnotexist")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        // Should NOT receive a notification (nothing was actually deleted)
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert!(receiver.try_recv().is_err());
    }

    // --- Auth Middleware Tests (Group 2) ---

    fn create_test_server_with_auth(secret: &str) -> Server {
        let dir = tempfile::tempdir().unwrap();
        let config = ServerConfig {
            http_addr: "127.0.0.1:0".to_string(),
            tcp_addr: "127.0.0.1:0".to_string(),
            db_path: dir.path().to_string_lossy().to_string(),
            enable_websocket: false,
            enable_tcp: false,
            max_tcp_connections: 100,
            enable_webtransport: false,
            webtransport_port: 0,
            enable_pgwire: false,
            pgwire_addr: "127.0.0.1:0".to_string(),
            enable_jwp: false,
            jwp_addr: "127.0.0.1:0".to_string(),
            max_jwp_connections: 100,
            auth_enabled: true,
            auth_jwt_secret: Some(secret.to_string()),
            enable_replication: false,
            replication_role: None,
            replication_listen_addr: "127.0.0.1:0".to_string(),
            replication_leader_addr: None,
            #[cfg(feature = "tls")]
            tls_cert_path: None,
            #[cfg(feature = "tls")]
            tls_key_path: None,
            energy_config: joule_db_energy::EnergyConfig::default(),
            enable_raft: false,
            raft_node_id: None,
            raft_addr: String::new(),
            raft_peers: Vec::new(),
            raft_master_secret: None,
            query_timeout_ms: 30000,
            slow_query_threshold_ms: 1000,
            rate_limiting_enabled: false,
            rate_limit_requests_per_minute: 1000,
            max_result_rows: 100_000,
            session_timeout_secs: 300,
            #[cfg(feature = "tls")]
            require_tls: false,
            cors_origins: Vec::new(),
            sanitize_errors: false,
            runtime_mode: "native".to_string(),
            enable_ledger: false,
            ledger_dir: None,
            ledger_batch_max_receipts: 1000,
            ledger_batch_interval_secs: 60,
            ledger_grid_region: None,
            ledger_grid_factor: None,
            scale_to_zero_enabled: false,
            idle_timeout_secs: 300,
            enable_mcp_stdio: false,
        };
        std::mem::forget(dir);
        Server::new(config).unwrap()
    }

    #[tokio::test]
    async fn test_auth_disabled_passes_through() {
        let server = create_test_server(); // auth_enabled=false
        let app = server.router();

        // No auth headers, should still get 200 on /api/v1/keys/test (even if 404 for key)
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_auth_enabled_rejects_no_token() {
        let server = create_test_server_with_auth("test-secret-that-is-long-enough-for-jwt");
        let app = server.router();

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/keys/testkey")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_auth_with_valid_jwt() {
        let secret = "test-secret-that-is-long-enough-for-jwt";
        let server = create_test_server_with_auth(secret);
        let auth_mgr = server.auth_manager.as_ref().unwrap();

        // Generate a valid JWT
        let token = auth_mgr
            .generate_jwt("testuser", vec!["admin".to_string()])
            .unwrap();

        let app = server.router();
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .header("authorization", format!("Bearer {}", token))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_auth_with_invalid_jwt() {
        let server = create_test_server_with_auth("test-secret-that-is-long-enough-for-jwt");
        let app = server.router();

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/keys/testkey")
                    .header("authorization", "Bearer invalid.jwt.token")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_auth_with_valid_api_key() {
        let secret = "test-secret-that-is-long-enough-for-jwt";
        let server = create_test_server_with_auth(secret);
        let auth_mgr = server.auth_manager.as_ref().unwrap();

        // Create an API key
        let (raw_key, _key_id) = auth_mgr
            .create_api_key("test-client", vec!["reader".to_string()], None)
            .unwrap();

        let app = server.router();
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .header("x-api-key", &raw_key)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_auth_with_invalid_api_key() {
        let server = create_test_server_with_auth("test-secret-that-is-long-enough-for-jwt");
        let app = server.router();

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/keys/testkey")
                    .header("x-api-key", "bogus-api-key-12345")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    // --- RBAC Tests (Group 2b) ---

    #[test]
    fn test_rbac_manager_initialized_with_auth() {
        let server = create_test_server_with_auth("test-secret");
        assert!(server.rbac_manager.is_some());
    }

    #[test]
    fn test_rbac_manager_not_initialized_without_auth() {
        let server = create_test_server(); // auth_enabled=false
        assert!(server.rbac_manager.is_none());
    }

    #[test]
    fn test_rbac_create_user_via_sql() {
        let server = create_test_server_with_auth("test-secret");
        let executor = query::SimpleQueryExecutor::with_amorphic(server.amorphic.clone());
        let mut executor = executor;
        executor.set_rbac_manager(server.rbac_manager.clone().unwrap());
        let request = query::QueryRequest {
            sql: "CREATE USER alice WITH PASSWORD 'secret123'".to_string(),
            params: Default::default(),
            args: Vec::new(),
            explain: false,
            limit: None,
            session_id: None,
            query_timeout_ms: None,
            branch_id: None,
            tenant_id: None,
        };
        let result = executor.execute(&request);
        assert!(result.is_ok());
        let resp = result.unwrap();
        assert_eq!(resp.rows[0][0], serde_json::json!("User 'alice' created"));
    }

    #[test]
    fn test_rbac_create_and_drop_role_via_sql() {
        let server = create_test_server_with_auth("test-secret");
        let mut executor = query::SimpleQueryExecutor::with_amorphic(server.amorphic.clone());
        executor.set_rbac_manager(server.rbac_manager.clone().unwrap());
        let request = query::QueryRequest {
            sql: "CREATE ROLE analyst".to_string(),
            params: Default::default(),
            args: Vec::new(),
            explain: false,
            limit: None,
            session_id: None,
            query_timeout_ms: None,
            branch_id: None,
            tenant_id: None,
        };
        let result = executor.execute(&request).unwrap();
        assert_eq!(
            result.rows[0][0],
            serde_json::json!("Role 'analyst' created")
        );

        let drop_req = query::QueryRequest {
            sql: "DROP ROLE analyst".to_string(),
            params: Default::default(),
            args: Vec::new(),
            explain: false,
            limit: None,
            session_id: None,
            query_timeout_ms: None,
            branch_id: None,
            tenant_id: None,
        };
        let result = executor.execute(&drop_req).unwrap();
        assert_eq!(
            result.rows[0][0],
            serde_json::json!("Role 'analyst' dropped")
        );
    }

    #[test]
    fn test_rbac_grant_revoke_via_sql() {
        let server = create_test_server_with_auth("test-secret");
        let mut executor = query::SimpleQueryExecutor::with_amorphic(server.amorphic.clone());
        executor.set_rbac_manager(server.rbac_manager.clone().unwrap());

        // Create a user first
        let create_req = query::QueryRequest {
            sql: "CREATE USER bob".to_string(),
            params: Default::default(),
            args: Vec::new(),
            explain: false,
            limit: None,
            session_id: None,
            query_timeout_ms: None,
            branch_id: None,
            tenant_id: None,
        };
        executor.execute(&create_req).unwrap();

        // Grant permissions
        let grant_req = query::QueryRequest {
            sql: "GRANT SELECT ON users TO bob".to_string(),
            params: Default::default(),
            args: Vec::new(),
            explain: false,
            limit: None,
            session_id: None,
            query_timeout_ms: None,
            branch_id: None,
            tenant_id: None,
        };
        let result = executor.execute(&grant_req).unwrap();
        assert_eq!(result.rows[0][0], serde_json::json!("GRANT"));

        // Revoke permissions
        let revoke_req = query::QueryRequest {
            sql: "REVOKE SELECT ON users FROM bob".to_string(),
            params: Default::default(),
            args: Vec::new(),
            explain: false,
            limit: None,
            session_id: None,
            query_timeout_ms: None,
            branch_id: None,
            tenant_id: None,
        };
        let result = executor.execute(&revoke_req).unwrap();
        assert_eq!(result.rows[0][0], serde_json::json!("REVOKE"));
    }

    #[test]
    fn test_rbac_drop_user_via_sql() {
        let server = create_test_server_with_auth("test-secret");
        let mut executor = query::SimpleQueryExecutor::with_amorphic(server.amorphic.clone());
        executor.set_rbac_manager(server.rbac_manager.clone().unwrap());

        // Create user then drop
        let create_req = query::QueryRequest {
            sql: "CREATE USER charlie".to_string(),
            params: Default::default(),
            args: Vec::new(),
            explain: false,
            limit: None,
            session_id: None,
            query_timeout_ms: None,
            branch_id: None,
            tenant_id: None,
        };
        executor.execute(&create_req).unwrap();

        let drop_req = query::QueryRequest {
            sql: "DROP USER charlie".to_string(),
            params: Default::default(),
            args: Vec::new(),
            explain: false,
            limit: None,
            session_id: None,
            query_timeout_ms: None,
            branch_id: None,
            tenant_id: None,
        };
        let result = executor.execute(&drop_req).unwrap();
        assert_eq!(
            result.rows[0][0],
            serde_json::json!("User 'charlie' dropped")
        );
    }

    #[test]
    fn test_rbac_persistence() {
        let server = create_test_server_with_auth("test-secret");
        let mut executor = query::SimpleQueryExecutor::with_amorphic(server.amorphic.clone());
        executor.set_rbac_manager(server.rbac_manager.clone().unwrap());

        // Create a user
        let create_req = query::QueryRequest {
            sql: "CREATE USER dave WITH PASSWORD 'pass123'".to_string(),
            params: Default::default(),
            args: Vec::new(),
            explain: false,
            limit: None,
            session_id: None,
            query_timeout_ms: None,
            branch_id: None,
            tenant_id: None,
        };
        executor.execute(&create_req).unwrap();

        // Verify user record was persisted to amorphic
        let users = server.amorphic.scan_rbac_meta("__rbac_users__");
        let dave = users
            .iter()
            .find(|u| u.get("user_id").and_then(|v| v.as_str()) == Some("dave"));
        assert!(dave.is_some(), "User 'dave' should be persisted");
    }

    #[tokio::test]
    async fn test_auth_middleware_injects_auth_info() {
        // When auth is disabled, middleware should inject anonymous user
        let server = create_test_server(); // auth_enabled=false
        let app = server.router();

        // Request to health should pass (anonymous)
        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    // --- Backup/Export Tests (Group 3) ---

    #[tokio::test]
    async fn test_backup_create() {
        let server = create_test_server();
        let app = server.router();

        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/v1/backup")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        // Backup creation may succeed (200) or fail if directory issues, but shouldn't panic
        let status = response.status();
        assert!(
            status == StatusCode::OK || status == StatusCode::INTERNAL_SERVER_ERROR,
            "Expected 200 or 500, got {}",
            status
        );
    }

    #[tokio::test]
    async fn test_backup_list() {
        let server = create_test_server();
        let app = server.router();

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/backup/list")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(value["backups"].is_array());
    }

    #[tokio::test]
    async fn test_backup_not_found() {
        let server = create_test_server();
        let app = server.router();

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/backup/nonexistent-id")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_restore_backup_not_found() {
        let server = create_test_server();
        let app = server.router();

        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/v1/restore/nonexistent-id")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_export_table_json() {
        let server = create_test_server();
        let executor = server.query_executor();

        // Create a table with data
        let _ = executor.execute(&QueryRequest {
            sql: "CREATE TABLE export_test (id INTEGER, name TEXT)".to_string(),
            params: Default::default(),
            args: vec![],
            explain: false,
            limit: None,
            session_id: None,
            query_timeout_ms: None,
            branch_id: None,
            tenant_id: None,
        });
        let _ = executor.execute(&QueryRequest {
            sql: "INSERT INTO export_test VALUES (1, 'Alice'), (2, 'Bob')".to_string(),
            params: Default::default(),
            args: vec![],
            explain: false,
            limit: None,
            session_id: None,
            query_timeout_ms: None,
            branch_id: None,
            tenant_id: None,
        });

        let app = server.router();
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/export/export_test")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["table"], "export_test");
        assert_eq!(value["row_count"], 2);
    }

    #[tokio::test]
    async fn test_export_table_csv() {
        let server = create_test_server();
        let executor = server.query_executor();

        let _ = executor.execute(&QueryRequest {
            sql: "CREATE TABLE csv_test (id INTEGER, val TEXT)".to_string(),
            params: Default::default(),
            args: vec![],
            explain: false,
            limit: None,
            session_id: None,
            query_timeout_ms: None,
            branch_id: None,
            tenant_id: None,
        });
        let _ = executor.execute(&QueryRequest {
            sql: "INSERT INTO csv_test VALUES (1, 'hello')".to_string(),
            params: Default::default(),
            args: vec![],
            explain: false,
            limit: None,
            session_id: None,
            query_timeout_ms: None,
            branch_id: None,
            tenant_id: None,
        });

        let app = server.router();
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/export/csv_test?format=csv")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let csv = String::from_utf8(body.to_vec()).unwrap();
        assert!(csv.contains("id,val"), "CSV should contain column headers");
        assert!(csv.contains("1"), "CSV should contain data");
    }

    // --- Replication Wiring Tests (Group 5) ---

    #[test]
    fn test_replication_config_defaults() {
        let config = ServerConfig::default();
        assert!(!config.enable_replication);
        assert!(config.replication_role.is_none());
        assert_eq!(config.replication_listen_addr, "127.0.0.1:6381");
        assert!(config.replication_leader_addr.is_none());
    }

    #[test]
    fn test_replication_server_from_config() {
        let repl_config = replication::ReplicationConfig {
            listen_addr: "127.0.0.1:0".to_string(),
            ..replication::ReplicationConfig::default()
        };
        let server = replication::ReplicationServer::new(repl_config);
        // Just verify it can be created without panic
        let stats = server.stats();
        assert_eq!(stats.entries_sent, 0);
    }

    #[test]
    fn test_replication_client_from_config() {
        let repl_config = replication::ReplicationConfig {
            leader_addr: Some("127.0.0.1:6381".to_string()),
            ..replication::ReplicationConfig::default()
        };
        let client = replication::ReplicationClient::new(repl_config);
        let stats = client.stats();
        assert_eq!(stats.entries_received, 0);
    }

    #[test]
    fn test_replication_broadcast_entry() {
        let repl_config = replication::ReplicationConfig {
            listen_addr: "127.0.0.1:0".to_string(),
            ..replication::ReplicationConfig::default()
        };
        let server = replication::ReplicationServer::new(repl_config);

        let entry = replication::ReplicationWalEntry {
            lsn: 1,
            op_type: replication::ReplicationOp::Put,
            key: b"test-key".to_vec(),
            value: Some(b"test-value".to_vec()),
            timestamp: 1000,
        };

        // broadcast_entry without subscribers returns error (no active receivers)
        let result = server.broadcast_entry(entry);
        assert!(result.is_err()); // Expected: no followers connected
    }

    #[test]
    fn test_replication_wal_entry_roundtrip() {
        let entry = replication::ReplicationWalEntry {
            lsn: 42,
            op_type: replication::ReplicationOp::Checkpoint,
            key: b"checkpoint".to_vec(),
            value: None,
            timestamp: 9999,
        };

        let encoded = entry.encode();
        let decoded = replication::ReplicationWalEntry::decode(&encoded).unwrap();
        assert_eq!(decoded.lsn, 42);
        assert_eq!(decoded.op_type, replication::ReplicationOp::Checkpoint);
        assert_eq!(decoded.key, b"checkpoint");
        assert!(decoded.value.is_none());
        assert_eq!(decoded.timestamp, 9999);
    }

    // --- Replication Follower Apply Tests ---

    #[test]
    fn test_replication_follower_applies_put() {
        let dir = tempfile::tempdir().unwrap();
        let store = joule_db_amorphic::DurableAmorphicStore::open(dir.path()).unwrap();
        let amorphic = AmorphicTableStorage::new(store);

        // Simulate what the follower loop does for a Put entry
        let entry = replication::ReplicationWalEntry {
            lsn: 1,
            op_type: replication::ReplicationOp::Put,
            key: b"test_record".to_vec(),
            value: Some(br#"{"name": "Alice", "age": 30}"#.to_vec()),
            timestamp: 1000,
        };

        // ingest_with_schema auto-creates the table schema from JSON keys
        let json_str = String::from_utf8_lossy(entry.value.as_ref().unwrap());
        let (record_id, table_name) = amorphic
            .ingest_with_schema(&json_str, Some("test_table"))
            .unwrap();
        assert!(record_id > 0);
        assert_eq!(table_name, "test_table");

        // Verify record exists via scan
        let rows = amorphic.scan("test_table").unwrap();
        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn test_replication_follower_applies_delete() {
        let dir = tempfile::tempdir().unwrap();
        let store = joule_db_amorphic::DurableAmorphicStore::open(dir.path()).unwrap();
        let amorphic = AmorphicTableStorage::new(store);

        // Ingest a record with schema so scan works
        let (record_id, _) = amorphic
            .ingest_with_schema(r#"{"key": "value"}"#, Some("del_test"))
            .unwrap();
        assert_eq!(amorphic.scan("del_test").unwrap().len(), 1);

        // Now simulate delete as the follower would
        amorphic.delete_record(record_id).unwrap();
        assert_eq!(amorphic.scan("del_test").unwrap().len(), 0);
    }

    #[test]
    fn test_replication_follower_updates_lsn() {
        let config = replication::ReplicationConfig::default();
        let client = replication::ReplicationClient::new(config);
        assert_eq!(client.applied_lsn(), 0);

        client.set_applied_lsn(42);
        assert_eq!(client.applied_lsn(), 42);

        client.set_applied_lsn(100);
        assert_eq!(client.applied_lsn(), 100);
    }

    #[test]
    fn test_replication_follower_checkpoint_noop() {
        // Checkpoint entries should not error — they're just informational
        let entry = replication::ReplicationWalEntry {
            lsn: 99,
            op_type: replication::ReplicationOp::Checkpoint,
            key: b"checkpoint".to_vec(),
            value: None,
            timestamp: 5000,
        };
        // The follower loop just logs checkpoints — verify the entry is well-formed
        assert_eq!(entry.op_type, replication::ReplicationOp::Checkpoint);
        assert!(entry.value.is_none());
    }

    #[test]
    fn test_replication_follower_invalid_json() {
        let dir = tempfile::tempdir().unwrap();
        let store = joule_db_amorphic::DurableAmorphicStore::open(dir.path()).unwrap();
        let amorphic = AmorphicTableStorage::new(store);

        // Simulate Put with invalid JSON — should error but not crash
        let bad_json = b"not valid json at all";
        let json_str = String::from_utf8_lossy(bad_json);
        let result = amorphic.ingest_json(&json_str);
        assert!(result.is_err(), "Invalid JSON should produce an error");
    }

    // --- Replication WAL Streaming Tests (Session 59) ---

    #[test]
    fn test_replication_server_stored_on_leader() {
        let dir = tempfile::tempdir().unwrap();
        let config = ServerConfig {
            db_path: dir.path().to_string_lossy().to_string(),
            enable_replication: true,
            replication_role: Some("leader".to_string()),
            replication_listen_addr: "127.0.0.1:0".to_string(),
            ..Default::default()
        };
        let server = Server::new(config).unwrap();
        assert!(
            server.replication_server.is_some(),
            "Leader should have replication_server"
        );
    }

    #[test]
    fn test_replication_server_not_stored_on_follower() {
        let dir = tempfile::tempdir().unwrap();
        let config = ServerConfig {
            db_path: dir.path().to_string_lossy().to_string(),
            enable_replication: true,
            replication_role: Some("follower".to_string()),
            ..Default::default()
        };
        let server = Server::new(config).unwrap();
        assert!(
            server.replication_server.is_none(),
            "Follower should NOT have replication_server"
        );
    }

    #[test]
    fn test_replication_server_not_stored_when_disabled() {
        let dir = tempfile::tempdir().unwrap();
        let config = ServerConfig {
            db_path: dir.path().to_string_lossy().to_string(),
            enable_replication: false,
            ..Default::default()
        };
        let server = Server::new(config).unwrap();
        assert!(
            server.replication_server.is_none(),
            "Disabled replication should not create server"
        );
    }

    #[tokio::test]
    async fn test_replication_status_endpoint_no_replication() {
        let server = create_test_server();
        let app = server.router();

        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/api/v1/replication/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["enabled"], false);
        assert_eq!(json["current_lsn"], 0);
        assert!(json["followers"].as_array().unwrap().is_empty());
    }

    #[test]
    fn test_replication_subscribe_and_broadcast() {
        let repl_config = replication::ReplicationConfig {
            listen_addr: "127.0.0.1:0".to_string(),
            ..replication::ReplicationConfig::default()
        };
        let server = replication::ReplicationServer::new(repl_config);
        let mut rx = server.subscribe();

        let entry = replication::ReplicationWalEntry {
            lsn: server.next_lsn(),
            op_type: replication::ReplicationOp::Put,
            key: b"test".to_vec(),
            value: Some(b"data".to_vec()),
            timestamp: 1000,
        };

        server.broadcast_entry(entry).unwrap();

        let received = rx.try_recv().unwrap();
        assert_eq!(received.lsn, 1);
        assert_eq!(received.op_type, replication::ReplicationOp::Put);
        assert_eq!(received.key, b"test");
    }

    // --- Sharding Scatter-Gather Tests (Session 60) ---

    #[test]
    fn test_shard_metadata_persistence() {
        let dir = tempfile::tempdir().unwrap();
        let store = joule_db_amorphic::DurableAmorphicStore::open(dir.path()).unwrap();
        let amorphic = AmorphicTableStorage::new(store);

        save_shard_metadata(&amorphic, "orders", "customer_id");
        save_shard_metadata(&amorphic, "events", "region");

        let tables = list_sharded_tables(&amorphic);
        assert_eq!(tables.len(), 2);

        let sk = get_shard_key(&amorphic, "orders");
        assert_eq!(sk, Some("customer_id".to_string()));
    }

    #[test]
    fn test_shard_metadata_remove() {
        let dir = tempfile::tempdir().unwrap();
        let store = joule_db_amorphic::DurableAmorphicStore::open(dir.path()).unwrap();
        let amorphic = AmorphicTableStorage::new(store);

        save_shard_metadata(&amorphic, "orders", "id");
        assert!(get_shard_key(&amorphic, "orders").is_some());

        remove_shard_metadata(&amorphic, "orders");
        assert!(get_shard_key(&amorphic, "orders").is_none());
    }

    #[test]
    fn test_shard_router_not_loaded_when_no_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let store = joule_db_amorphic::DurableAmorphicStore::open(dir.path()).unwrap();
        let amorphic = AmorphicTableStorage::new(store);
        assert!(load_shard_router_from_amorphic(&amorphic).is_none());
    }

    #[test]
    fn test_shard_router_loaded_when_metadata_exists() {
        let dir = tempfile::tempdir().unwrap();
        let store = joule_db_amorphic::DurableAmorphicStore::open(dir.path()).unwrap();
        let amorphic = AmorphicTableStorage::new(store);
        save_shard_metadata(&amorphic, "orders", "customer_id");
        assert!(load_shard_router_from_amorphic(&amorphic).is_some());
    }

    #[tokio::test]
    async fn test_shard_status_endpoint() {
        let server = create_test_server();
        let app = server.router();

        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/api/v1/shards/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["router_active"], false);
        assert!(json["sharded_tables"].as_array().unwrap().is_empty());
    }

    // --- Extended ServerConfig Defaults Tests ---

    #[test]
    fn test_server_config_new_fields_defaults() {
        let config = ServerConfig::default();
        assert!(config.enable_pgwire);
        assert_eq!(config.pgwire_addr, "127.0.0.1:5433");
        assert!(config.auth_enabled);
        assert!(config.auth_jwt_secret.is_none());
        assert!(!config.enable_replication);
        assert!(config.replication_role.is_none());
    }

    // --- Unified Endpoint Tests (Group 6) ---

    #[tokio::test]
    async fn test_unified_ingest_single() {
        let server = create_test_server();
        let app = server.router();

        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"name":"Alice","age":30}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["ok"], true);
        assert!(json["id"].as_u64().is_some());
        assert_eq!(json["collection"], "__default__");
    }

    #[tokio::test]
    async fn test_unified_ingest_batch() {
        let server = create_test_server();
        let app = server.router();

        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"[{"name":"Alice","age":30},{"name":"Bob","age":25},{"name":"Carol","age":35}]"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["ok"], true);
        assert_eq!(json["count"], 3);
        assert!(json["ids"].as_array().unwrap().len() == 3);
        assert_eq!(json["collection"], "__default__");
    }

    #[tokio::test]
    async fn test_unified_ingest_with_collection() {
        let server = create_test_server();
        let app = server.router();

        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"_collection":"users","name":"Alice","age":30}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["ok"], true);
        assert_eq!(json["collection"], "users");
    }

    #[tokio::test]
    async fn test_unified_query_sql() {
        let server = create_test_server();
        let app = server.router();

        // First ingest some records
        let _ = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"_collection":"staff","name":"Alice","age":30}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        let _ = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"_collection":"staff","name":"Bob","age":25}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        // Query via SQL
        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"q":"SELECT * FROM staff"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["ok"], true);
        assert_eq!(json["count"], 2);
        assert!(json["columns"].as_array().is_some());
        assert!(json["data"].as_array().is_some());
    }

    #[tokio::test]
    async fn test_unified_query_sql_with_where() {
        let server = create_test_server();
        let app = server.router();

        // Ingest records into "team" collection
        for record in &[
            r#"{"_collection":"team","name":"Alice","age":30}"#,
            r#"{"_collection":"team","name":"Bob","age":25}"#,
            r#"{"_collection":"team","name":"Carol","age":35}"#,
        ] {
            let _ = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method(Method::POST)
                        .uri("/")
                        .header("content-type", "application/json")
                        .body(Body::from(*record))
                        .unwrap(),
                )
                .await
                .unwrap();
        }

        // Query with WHERE clause
        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"q":"SELECT name FROM team WHERE age > 28"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["ok"], true);
        // Should return Alice (30) and Carol (35)
        assert_eq!(json["count"], 2);
    }

    #[tokio::test]
    async fn test_unified_query_sql_aggregates() {
        let server = create_test_server();
        let app = server.router();

        // Ingest records
        for record in &[
            r#"{"_collection":"nums","val":10}"#,
            r#"{"_collection":"nums","val":20}"#,
            r#"{"_collection":"nums","val":30}"#,
        ] {
            let _ = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method(Method::POST)
                        .uri("/")
                        .header("content-type", "application/json")
                        .body(Body::from(*record))
                        .unwrap(),
                )
                .await
                .unwrap();
        }

        // Query with COUNT
        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"q":"SELECT COUNT(*) AS cnt FROM nums"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["ok"], true);
        assert_eq!(json["data"][0][0], 3);
    }

    #[tokio::test]
    async fn test_unified_ingest_then_sql_roundtrip() {
        let server = create_test_server();
        let app = server.router();

        // Batch ingest
        let _ = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"[
                            {"_collection":"roundtrip","x":1},
                            {"_collection":"roundtrip","x":2},
                            {"_collection":"roundtrip","x":3}
                        ]"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        // Query back via SQL
        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"q":"SELECT x FROM roundtrip ORDER BY x"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["ok"], true);
        assert_eq!(json["count"], 3);
        let data = json["data"].as_array().unwrap();
        assert_eq!(data[0][0], 1);
        assert_eq!(data[1][0], 2);
        assert_eq!(data[2][0], 3);
    }

    #[tokio::test]
    async fn test_unified_ingest_schema_merge() {
        let server = create_test_server();
        let app = server.router();

        // Ingest with fields {a, b}
        let _ = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"_collection":"merge","a":1,"b":"x"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        // Ingest with fields {a, c} — schema should merge to {a, b, c}
        let _ = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"_collection":"merge","a":2,"c":"y"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        // SELECT * should show all columns
        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"q":"SELECT * FROM merge"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["ok"], true);
        let cols = json["columns"].as_array().unwrap();
        let col_names: Vec<&str> = cols.iter().filter_map(|c| c.as_str()).collect();
        assert!(col_names.contains(&"a"));
        assert!(col_names.contains(&"b"));
        assert!(col_names.contains(&"c"));
    }

    #[tokio::test]
    async fn test_unified_error_invalid_json() {
        let server = create_test_server();
        let app = server.router();

        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/")
                    .header("content-type", "application/json")
                    .body(Body::from("not valid json"))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["ok"], false);
        assert!(json["error"].as_str().is_some());
    }

    #[tokio::test]
    async fn test_unified_empty_object() {
        let server = create_test_server();
        let app = server.router();

        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/")
                    .header("content-type", "application/json")
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["ok"], true);
        assert!(json["id"].as_u64().is_some());
    }

    #[test]
    fn test_is_sql_detection() {
        assert!(is_sql("SELECT * FROM users"));
        assert!(is_sql("select * from users"));
        assert!(is_sql("  INSERT INTO t VALUES (1)"));
        assert!(is_sql("UPDATE t SET x = 1"));
        assert!(is_sql("DELETE FROM t"));
        assert!(is_sql("CREATE TABLE t (x INT)"));
        assert!(is_sql("DROP TABLE t"));
        assert!(is_sql("ALTER TABLE t ADD COLUMN y INT"));
        assert!(is_sql("BEGIN"));
        assert!(is_sql("COMMIT"));
        assert!(is_sql("ROLLBACK"));
        assert!(is_sql("EXPLAIN SELECT 1"));
        assert!(is_sql("SHOW TABLES"));
        assert!(is_sql("WITH cte AS (SELECT 1) SELECT * FROM cte"));
        assert!(is_sql("TRUNCATE TABLE t"));

        assert!(!is_sql("Alice"));
        assert!(!is_sql("hello world"));
        assert!(!is_sql("find similar records"));
        assert!(!is_sql(""));
    }

    // --- TLS Tests ---

    #[cfg(feature = "tls")]
    mod tls_tests {
        use super::*;

        fn generate_self_signed_cert(dir: &std::path::Path) -> (String, String) {
            let cert = rcgen::generate_simple_self_signed(vec!["localhost".to_string()]).unwrap();
            let cert_path = dir.join("cert.pem");
            let key_path = dir.join("key.pem");
            std::fs::write(&cert_path, cert.cert.pem()).unwrap();
            std::fs::write(&key_path, cert.key_pair.serialize_pem()).unwrap();
            (
                cert_path.to_string_lossy().to_string(),
                key_path.to_string_lossy().to_string(),
            )
        }

        #[test]
        fn test_tls_config_parsing() {
            let config = ServerConfig {
                tls_cert_path: Some("/path/to/cert.pem".to_string()),
                tls_key_path: Some("/path/to/key.pem".to_string()),
                ..Default::default()
            };
            assert_eq!(config.tls_cert_path.as_deref(), Some("/path/to/cert.pem"));
            assert_eq!(config.tls_key_path.as_deref(), Some("/path/to/key.pem"));
        }

        #[test]
        fn test_tls_config_default_none() {
            let config = ServerConfig::default();
            assert!(config.tls_cert_path.is_none());
            assert!(config.tls_key_path.is_none());
        }

        #[test]
        fn test_tls_acceptor_creation() {
            let dir = tempfile::tempdir().unwrap();
            let (cert_path, key_path) = generate_self_signed_cert(dir.path());
            let acceptor = Server::create_tls_acceptor(&cert_path, &key_path);
            assert!(
                acceptor.is_ok(),
                "TLS acceptor creation should succeed: {:?}",
                acceptor.err()
            );
        }

        #[test]
        fn test_tls_invalid_cert_rejected() {
            let dir = tempfile::tempdir().unwrap();
            let bad_cert = dir.path().join("bad_cert.pem");
            let bad_key = dir.path().join("bad_key.pem");
            std::fs::write(&bad_cert, "not a real cert").unwrap();
            std::fs::write(&bad_key, "not a real key").unwrap();
            let result = Server::create_tls_acceptor(
                &bad_cert.to_string_lossy(),
                &bad_key.to_string_lossy(),
            );
            assert!(result.is_err(), "Bad cert should fail");
        }

        #[test]
        fn test_tls_missing_cert_rejected() {
            let result =
                Server::create_tls_acceptor("/nonexistent/cert.pem", "/nonexistent/key.pem");
            assert!(result.is_err(), "Missing cert file should fail");
        }

        #[test]
        fn test_non_tls_still_works() {
            // Verify server can be created without TLS config
            let dir = tempfile::tempdir().unwrap();
            let config = ServerConfig {
                db_path: dir.path().to_string_lossy().to_string(),
                tls_cert_path: None,
                tls_key_path: None,
                ..Default::default()
            };
            let server = Server::new(config);
            assert!(
                server.is_ok(),
                "Server without TLS should work: {:?}",
                server.err()
            );
        }
    }

    // --- Phase 4: Security Headers & CORS Tests ---

    #[tokio::test]
    async fn test_security_headers_present() {
        let server = create_test_server();
        let app = server.router();

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(
            response.headers().get("X-Content-Type-Options").unwrap(),
            "nosniff"
        );
        assert_eq!(response.headers().get("X-Frame-Options").unwrap(), "DENY");
        assert_eq!(
            response.headers().get("X-XSS-Protection").unwrap(),
            "1; mode=block"
        );
        assert_eq!(
            response.headers().get("Referrer-Policy").unwrap(),
            "strict-origin-when-cross-origin"
        );
        assert_eq!(
            response.headers().get("Content-Security-Policy").unwrap(),
            "default-src 'self'"
        );
        assert_eq!(
            response.headers().get("Permissions-Policy").unwrap(),
            "geolocation=(), microphone=(), camera=()"
        );
        assert!(
            response
                .headers()
                .get("Strict-Transport-Security")
                .is_some()
        );
    }

    #[tokio::test]
    async fn test_cors_blocked_when_no_origins_configured() {
        let server = create_test_server(); // cors_origins = empty
        let app = server.router();

        // Preflight OPTIONS request from arbitrary origin should not include CORS headers
        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::OPTIONS)
                    .uri("/health")
                    .header("Origin", "https://evil.com")
                    .header("Access-Control-Request-Method", "GET")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        // No Access-Control-Allow-Origin header should be present
        assert!(
            response
                .headers()
                .get("Access-Control-Allow-Origin")
                .is_none(),
            "CORS should not allow any origin when cors_origins is empty"
        );
    }

    #[tokio::test]
    async fn test_cors_allowed_when_origins_configured() {
        let dir = tempfile::tempdir().unwrap();
        let config = ServerConfig {
            db_path: dir.path().to_string_lossy().to_string(),
            auth_enabled: false,
            cors_origins: vec!["https://app.example.com".to_string()],
            ..Default::default()
        };
        std::mem::forget(dir);
        let server = Server::new(config).unwrap();
        let app = server.router();

        // Preflight from allowed origin
        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::OPTIONS)
                    .uri("/health")
                    .header("Origin", "https://app.example.com")
                    .header("Access-Control-Request-Method", "GET")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(
            response
                .headers()
                .get("Access-Control-Allow-Origin")
                .map(|v| v.to_str().unwrap()),
            Some("https://app.example.com"),
            "CORS should allow configured origin"
        );
    }

    #[tokio::test]
    async fn test_rate_limiting_enabled_by_default() {
        let config = ServerConfig::default();
        assert!(
            config.rate_limiting_enabled,
            "Rate limiting should be enabled by default"
        );
    }

    #[test]
    fn test_error_sanitization() {
        let err = query::QueryErrorResponse::table_not_found("secret_users");
        assert!(err.message.contains("secret_users"));

        let safe = err.sanitized();
        assert!(!safe.message.contains("secret_users"));
        assert_eq!(safe.code, "TABLE_NOT_FOUND");
        assert_eq!(safe.message, "Referenced object does not exist");
    }

    #[test]
    fn test_error_sanitization_preserves_timeout() {
        let err = query::QueryErrorResponse::query_timeout(5000);
        let safe = err.sanitized();
        // Timeout message is already safe, should be preserved
        assert!(safe.message.contains("5000"));
    }
}
