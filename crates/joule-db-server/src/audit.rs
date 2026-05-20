//! Comprehensive Audit Logging System for JouleDB Server
//!
//! Provides enterprise-grade audit logging capabilities:
//! - Audit event types (authentication, query, data modification, admin operations)
//! - Tamper-resistant audit log storage with cryptographic checksums
//! - Structured audit entries with timestamps, user context, and resource details
//! - Log rotation and retention policies
//! - Compliance-ready format (SOC 2, HIPAA compatible)
//! - Flexible query interface for audit log analysis

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

// ============================================================================
// Error Types
// ============================================================================

/// Audit system errors
#[derive(Debug, Clone, PartialEq)]
pub enum AuditError {
    /// Storage operation failed
    StorageError(String),
    /// Tamper detected in audit log
    TamperDetected {
        entry_id: String,
        expected_hash: String,
        actual_hash: String,
    },
    /// Invalid query parameters
    InvalidQuery(String),
    /// Rotation failed
    RotationFailed(String),
    /// Serialization error
    SerializationError(String),
    /// Entry not found
    EntryNotFound(String),
    /// Configuration error
    ConfigError(String),
}

impl fmt::Display for AuditError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StorageError(msg) => write!(f, "Audit storage error: {}", msg),
            Self::TamperDetected {
                entry_id,
                expected_hash,
                actual_hash,
            } => {
                write!(
                    f,
                    "Tamper detected in entry {}: expected {}, got {}",
                    entry_id, expected_hash, actual_hash
                )
            }
            Self::InvalidQuery(msg) => write!(f, "Invalid audit query: {}", msg),
            Self::RotationFailed(msg) => write!(f, "Log rotation failed: {}", msg),
            Self::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
            Self::EntryNotFound(id) => write!(f, "Audit entry not found: {}", id),
            Self::ConfigError(msg) => write!(f, "Configuration error: {}", msg),
        }
    }
}

impl std::error::Error for AuditError {}

pub type AuditResult<T> = Result<T, AuditError>;

// ============================================================================
// Audit Event Types
// ============================================================================

/// Categories of audit events for compliance tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum AuditEventType {
    /// Authentication events (login, logout, token refresh)
    Authentication,
    /// Authorization events (permission checks, access denials)
    Authorization,
    /// Query operations (reads, searches)
    Query,
    /// Data modification (create, update, delete)
    DataModification,
    /// Administrative operations (config changes, user management)
    AdminAction,
    /// Schema changes (table creation, index modifications)
    SchemaChange,
    /// Security events (failed logins, suspicious activity)
    SecurityEvent,
    /// System events (startup, shutdown, errors)
    SystemEvent,
    /// Data export operations
    DataExport,
    /// Data import operations
    DataImport,
    /// Backup operations
    BackupOperation,
    /// Replication events
    ReplicationEvent,
}

impl fmt::Display for AuditEventType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Authentication => write!(f, "AUTHENTICATION"),
            Self::Authorization => write!(f, "AUTHORIZATION"),
            Self::Query => write!(f, "QUERY"),
            Self::DataModification => write!(f, "DATA_MODIFICATION"),
            Self::AdminAction => write!(f, "ADMIN_ACTION"),
            Self::SchemaChange => write!(f, "SCHEMA_CHANGE"),
            Self::SecurityEvent => write!(f, "SECURITY_EVENT"),
            Self::SystemEvent => write!(f, "SYSTEM_EVENT"),
            Self::DataExport => write!(f, "DATA_EXPORT"),
            Self::DataImport => write!(f, "DATA_IMPORT"),
            Self::BackupOperation => write!(f, "BACKUP_OPERATION"),
            Self::ReplicationEvent => write!(f, "REPLICATION_EVENT"),
        }
    }
}

/// Specific audit actions within each event type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum AuditAction {
    // Authentication actions
    LoginSuccess,
    LoginFailure,
    Logout,
    TokenIssued,
    TokenRefreshed,
    TokenRevoked,
    PasswordChanged,
    MfaEnabled,
    MfaDisabled,
    ApiKeyCreated,
    ApiKeyRevoked,

    // Authorization actions
    PermissionGranted,
    PermissionDenied,
    RoleAssigned,
    RoleRevoked,
    AccessDenied,

    // Query actions
    QueryExecuted,
    SearchExecuted,
    BulkQueryExecuted,

    // Data modification actions
    RecordCreated,
    RecordUpdated,
    RecordDeleted,
    BulkInsert,
    BulkUpdate,
    BulkDelete,

    // Admin actions
    ConfigChanged,
    UserCreated,
    UserDeleted,
    UserModified,
    ServiceStarted,
    ServiceStopped,
    MaintenanceStarted,
    MaintenanceCompleted,

    // Schema actions
    TableCreated,
    TableDropped,
    IndexCreated,
    IndexDropped,
    SchemaAltered,

    // Security actions
    SuspiciousActivity,
    RateLimitExceeded,
    IpBlocked,
    IpUnblocked,
    IntrusionDetected,

    // Backup/Restore actions
    BackupStarted,
    BackupCompleted,
    BackupFailed,
    RestoreStarted,
    RestoreCompleted,
    RestoreFailed,

    // Replication actions
    ReplicationStarted,
    ReplicationSynced,
    ReplicationFailed,
    NodeJoined,
    NodeLeft,

    // Custom action
    Custom(String),
}

impl fmt::Display for AuditAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Custom(action) => write!(f, "CUSTOM:{}", action),
            _ => write!(f, "{:?}", self),
        }
    }
}

/// Outcome of an audited operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AuditOutcome {
    Success,
    Failure,
    Partial,
    Denied,
    Error,
}

impl fmt::Display for AuditOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Success => write!(f, "success"),
            Self::Failure => write!(f, "failure"),
            Self::Partial => write!(f, "partial"),
            Self::Denied => write!(f, "denied"),
            Self::Error => write!(f, "error"),
        }
    }
}

/// Severity level for audit events
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AuditSeverity {
    Debug,
    Info,
    Notice,
    Warning,
    Error,
    Critical,
    Alert,
    Emergency,
}

impl fmt::Display for AuditSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Debug => write!(f, "debug"),
            Self::Info => write!(f, "info"),
            Self::Notice => write!(f, "notice"),
            Self::Warning => write!(f, "warning"),
            Self::Error => write!(f, "error"),
            Self::Critical => write!(f, "critical"),
            Self::Alert => write!(f, "alert"),
            Self::Emergency => write!(f, "emergency"),
        }
    }
}

// ============================================================================
// Audit Event Structure
// ============================================================================

/// User/actor context for audit events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditActor {
    /// Unique user/service identifier
    pub id: String,
    /// Actor type (user, service, system)
    pub actor_type: String,
    /// Display name
    pub name: Option<String>,
    /// Email address (for users)
    pub email: Option<String>,
    /// Roles at time of action
    pub roles: Vec<String>,
    /// Session ID if applicable
    pub session_id: Option<String>,
    /// Source IP address
    pub ip_address: Option<String>,
    /// User agent string
    pub user_agent: Option<String>,
    /// Geographic location (if available)
    pub geo_location: Option<String>,
}

impl Default for AuditActor {
    fn default() -> Self {
        Self {
            id: "system".to_string(),
            actor_type: "system".to_string(),
            name: Some("System".to_string()),
            email: None,
            roles: vec![],
            session_id: None,
            ip_address: None,
            user_agent: None,
            geo_location: None,
        }
    }
}

/// Resource being accessed/modified
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditResource {
    /// Resource type (table, document, config, etc.)
    pub resource_type: String,
    /// Resource identifier
    pub resource_id: String,
    /// Parent resource (e.g., database for table)
    pub parent_id: Option<String>,
    /// Resource path/location
    pub path: Option<String>,
    /// Additional resource attributes
    pub attributes: HashMap<String, String>,
}

impl AuditResource {
    pub fn new(resource_type: impl Into<String>, resource_id: impl Into<String>) -> Self {
        Self {
            resource_type: resource_type.into(),
            resource_id: resource_id.into(),
            parent_id: None,
            path: None,
            attributes: HashMap::new(),
        }
    }

    pub fn with_path(mut self, path: impl Into<String>) -> Self {
        self.path = Some(path.into());
        self
    }

    pub fn with_attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.insert(key.into(), value.into());
        self
    }
}

/// Complete audit event entry - compliance-ready structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    /// Unique event identifier (UUID v4)
    pub id: String,
    /// Event timestamp (ISO 8601 format for compliance)
    pub timestamp: String,
    /// Unix timestamp in milliseconds (for efficient querying)
    pub timestamp_ms: u64,
    /// Event type category
    pub event_type: AuditEventType,
    /// Specific action performed
    pub action: AuditAction,
    /// Outcome of the action
    pub outcome: AuditOutcome,
    /// Severity level
    pub severity: AuditSeverity,
    /// Actor who performed the action
    pub actor: AuditActor,
    /// Resource(s) involved
    pub resources: Vec<AuditResource>,
    /// Human-readable description
    pub description: String,
    /// Detailed information about the event
    pub details: HashMap<String, serde_json::Value>,
    /// Previous value (for modifications)
    pub old_value: Option<serde_json::Value>,
    /// New value (for modifications)
    pub new_value: Option<serde_json::Value>,
    /// Error message if outcome is failure/error
    pub error_message: Option<String>,
    /// Error code if applicable
    pub error_code: Option<String>,
    /// Duration of operation in milliseconds
    pub duration_ms: Option<u64>,
    /// Request ID for correlation
    pub request_id: Option<String>,
    /// Transaction ID for grouping related events
    pub transaction_id: Option<String>,
    /// Parent event ID for hierarchical events
    pub parent_event_id: Option<String>,
    /// Server/node identifier
    pub server_id: String,
    /// Service name
    pub service_name: String,
    /// Service version
    pub service_version: String,
    /// Environment (production, staging, development)
    pub environment: String,
    /// Compliance tags (HIPAA, SOC2, etc.)
    pub compliance_tags: Vec<String>,
    /// Hash of previous entry (for tamper detection)
    pub prev_hash: String,
    /// Hash of this entry
    pub entry_hash: String,
    /// Sequence number within the log
    pub sequence: u64,
}

impl AuditEvent {
    /// Calculate the hash of this entry (excluding entry_hash field)
    pub fn calculate_hash(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.id.as_bytes());
        hasher.update(self.timestamp.as_bytes());
        hasher.update(format!("{}", self.timestamp_ms).as_bytes());
        hasher.update(format!("{:?}", self.event_type).as_bytes());
        hasher.update(format!("{:?}", self.action).as_bytes());
        hasher.update(format!("{:?}", self.outcome).as_bytes());
        hasher.update(self.actor.id.as_bytes());
        hasher.update(self.description.as_bytes());
        hasher.update(self.prev_hash.as_bytes());
        hasher.update(format!("{}", self.sequence).as_bytes());

        // Include details hash
        if let Ok(details_json) = serde_json::to_string(&self.details) {
            hasher.update(details_json.as_bytes());
        }

        let result = hasher.finalize();
        hex::encode(result)
    }

    /// Verify the integrity of this entry
    pub fn verify_integrity(&self) -> bool {
        self.entry_hash == self.calculate_hash()
    }
}

/// Builder for constructing audit events
pub struct AuditEventBuilder {
    event_type: AuditEventType,
    action: AuditAction,
    outcome: AuditOutcome,
    severity: AuditSeverity,
    actor: AuditActor,
    resources: Vec<AuditResource>,
    description: String,
    details: HashMap<String, serde_json::Value>,
    old_value: Option<serde_json::Value>,
    new_value: Option<serde_json::Value>,
    error_message: Option<String>,
    error_code: Option<String>,
    duration_ms: Option<u64>,
    request_id: Option<String>,
    transaction_id: Option<String>,
    parent_event_id: Option<String>,
    compliance_tags: Vec<String>,
}

impl AuditEventBuilder {
    pub fn new(event_type: AuditEventType, action: AuditAction) -> Self {
        Self {
            event_type,
            action,
            outcome: AuditOutcome::Success,
            severity: AuditSeverity::Info,
            actor: AuditActor::default(),
            resources: vec![],
            description: String::new(),
            details: HashMap::new(),
            old_value: None,
            new_value: None,
            error_message: None,
            error_code: None,
            duration_ms: None,
            request_id: None,
            transaction_id: None,
            parent_event_id: None,
            compliance_tags: vec![],
        }
    }

    pub fn outcome(mut self, outcome: AuditOutcome) -> Self {
        self.outcome = outcome;
        self
    }

    pub fn severity(mut self, severity: AuditSeverity) -> Self {
        self.severity = severity;
        self
    }

    pub fn actor(mut self, actor: AuditActor) -> Self {
        self.actor = actor;
        self
    }

    pub fn resource(mut self, resource: AuditResource) -> Self {
        self.resources.push(resource);
        self
    }

    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    pub fn detail(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.details.insert(key.into(), value);
        self
    }

    pub fn old_value(mut self, value: serde_json::Value) -> Self {
        self.old_value = Some(value);
        self
    }

    pub fn new_value(mut self, value: serde_json::Value) -> Self {
        self.new_value = Some(value);
        self
    }

    pub fn error(mut self, message: impl Into<String>, code: Option<String>) -> Self {
        self.error_message = Some(message.into());
        self.error_code = code;
        self.outcome = AuditOutcome::Error;
        self
    }

    pub fn duration_ms(mut self, duration: u64) -> Self {
        self.duration_ms = Some(duration);
        self
    }

    pub fn request_id(mut self, id: impl Into<String>) -> Self {
        self.request_id = Some(id.into());
        self
    }

    pub fn transaction_id(mut self, id: impl Into<String>) -> Self {
        self.transaction_id = Some(id.into());
        self
    }

    pub fn parent_event_id(mut self, id: impl Into<String>) -> Self {
        self.parent_event_id = Some(id.into());
        self
    }

    pub fn compliance_tag(mut self, tag: impl Into<String>) -> Self {
        self.compliance_tags.push(tag.into());
        self
    }

    pub fn hipaa(self) -> Self {
        self.compliance_tag("HIPAA")
    }

    pub fn soc2(self) -> Self {
        self.compliance_tag("SOC2")
    }

    pub fn gdpr(self) -> Self {
        self.compliance_tag("GDPR")
    }

    pub fn pci_dss(self) -> Self {
        self.compliance_tag("PCI-DSS")
    }
}

// ============================================================================
// Audit Configuration
// ============================================================================

/// Configuration for audit logging system
#[derive(Debug, Clone)]
pub struct AuditConfig {
    /// Enable audit logging
    pub enabled: bool,
    /// Directory for audit log files
    pub log_directory: PathBuf,
    /// Maximum size of a single log file (bytes)
    pub max_file_size: u64,
    /// Maximum number of log files to retain
    pub max_files: usize,
    /// Retention period in days (0 = unlimited)
    pub retention_days: u32,
    /// Server identifier
    pub server_id: String,
    /// Service name
    pub service_name: String,
    /// Service version
    pub service_version: String,
    /// Environment name
    pub environment: String,
    /// Event types to include (empty = all)
    pub include_event_types: Vec<AuditEventType>,
    /// Event types to exclude
    pub exclude_event_types: Vec<AuditEventType>,
    /// Minimum severity to log
    pub min_severity: AuditSeverity,
    /// Enable tamper detection
    pub tamper_detection: bool,
    /// Sync writes to disk immediately
    pub sync_writes: bool,
    /// Buffer size for in-memory events
    pub buffer_size: usize,
    /// Flush interval in milliseconds
    pub flush_interval_ms: u64,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            log_directory: PathBuf::from("./audit_logs"),
            max_file_size: 100 * 1024 * 1024, // 100MB
            max_files: 100,
            retention_days: 365, // 1 year for compliance
            server_id: format!(
                "server-{}",
                Uuid::new_v4()
                    .to_string()
                    .split('-')
                    .next()
                    .unwrap_or("unknown")
            ),
            service_name: "joule_db".to_string(),
            service_version: env!("CARGO_PKG_VERSION").to_string(),
            environment: "production".to_string(),
            include_event_types: vec![],
            exclude_event_types: vec![],
            min_severity: AuditSeverity::Info,
            tamper_detection: true,
            sync_writes: true,
            buffer_size: 1000,
            flush_interval_ms: 1000,
        }
    }
}

// ============================================================================
// Audit Store
// ============================================================================

/// Persistent storage for audit logs with tamper detection
pub struct AuditStore {
    config: AuditConfig,
    current_file: Option<File>,
    current_file_size: u64,
    current_file_path: PathBuf,
    sequence: u64,
    prev_hash: String,
    buffer: VecDeque<AuditEvent>,
}

impl AuditStore {
    /// Create a new audit store
    pub fn new(config: AuditConfig) -> AuditResult<Self> {
        // Create log directory if it doesn't exist
        if !config.log_directory.exists() {
            fs::create_dir_all(&config.log_directory).map_err(|e| {
                AuditError::StorageError(format!("Failed to create log directory: {}", e))
            })?;
        }

        // Determine current file path and sequence
        let (current_file_path, sequence, prev_hash) = Self::recover_state(&config)?;

        let mut store = Self {
            config,
            current_file: None,
            current_file_size: 0,
            current_file_path,
            sequence,
            prev_hash,
            buffer: VecDeque::new(),
        };

        store.open_current_file()?;
        Ok(store)
    }

    /// Recover state from existing log files
    fn recover_state(config: &AuditConfig) -> AuditResult<(PathBuf, u64, String)> {
        let mut latest_sequence = 0u64;
        let mut latest_hash = "genesis".to_string();
        let mut latest_file: Option<PathBuf> = None;

        if let Ok(entries) = fs::read_dir(&config.log_directory) {
            for entry in entries.filter_map(|e| e.ok()) {
                let path = entry.path();
                if path.extension().map(|e| e == "audit").unwrap_or(false) {
                    // Read last line to get sequence and hash
                    if let Ok(file) = File::open(&path) {
                        let reader = BufReader::new(file);
                        if let Some(last_line) = reader.lines().filter_map(|l| l.ok()).last() {
                            if let Ok(event) = serde_json::from_str::<AuditEvent>(&last_line) {
                                if event.sequence > latest_sequence {
                                    latest_sequence = event.sequence;
                                    latest_hash = event.entry_hash.clone();
                                    latest_file = Some(path);
                                }
                            }
                        }
                    }
                }
            }
        }

        let current_file_path = latest_file.unwrap_or_else(|| {
            config.log_directory.join(format!(
                "audit_{}.audit",
                chrono::Utc::now().format("%Y%m%d_%H%M%S")
            ))
        });

        Ok((current_file_path, latest_sequence, latest_hash))
    }

    /// Open or create the current log file
    fn open_current_file(&mut self) -> AuditResult<()> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.current_file_path)
            .map_err(|e| AuditError::StorageError(format!("Failed to open log file: {}", e)))?;

        self.current_file_size = file.metadata().map(|m| m.len()).unwrap_or(0);

        self.current_file = Some(file);
        Ok(())
    }

    /// Write an audit event to storage
    pub fn write(&mut self, event: AuditEvent) -> AuditResult<()> {
        // Check if rotation is needed
        if self.current_file_size >= self.config.max_file_size {
            self.rotate()?;
        }

        let json = serde_json::to_string(&event)
            .map_err(|e| AuditError::SerializationError(e.to_string()))?;

        let line = format!("{}\n", json);
        let bytes = line.as_bytes();

        if let Some(ref mut file) = self.current_file {
            file.write_all(bytes)
                .map_err(|e| AuditError::StorageError(format!("Failed to write: {}", e)))?;

            if self.config.sync_writes {
                file.sync_all()
                    .map_err(|e| AuditError::StorageError(format!("Failed to sync: {}", e)))?;
            }

            self.current_file_size += bytes.len() as u64;
        }

        self.sequence = event.sequence;
        self.prev_hash = event.entry_hash;

        Ok(())
    }

    /// Rotate log files
    pub fn rotate(&mut self) -> AuditResult<()> {
        // Close current file
        self.current_file = None;

        // Create new file path
        self.current_file_path = self.config.log_directory.join(format!(
            "audit_{}.audit",
            chrono::Utc::now().format("%Y%m%d_%H%M%S")
        ));

        self.current_file_size = 0;
        self.open_current_file()?;

        // Clean up old files
        self.cleanup_old_files()?;

        Ok(())
    }

    /// Clean up old log files based on retention policy
    fn cleanup_old_files(&self) -> AuditResult<()> {
        let mut files: Vec<(PathBuf, SystemTime)> = vec![];

        if let Ok(entries) = fs::read_dir(&self.config.log_directory) {
            for entry in entries.filter_map(|e| e.ok()) {
                let path = entry.path();
                if path.extension().map(|e| e == "audit").unwrap_or(false) {
                    if let Ok(metadata) = path.metadata() {
                        if let Ok(modified) = metadata.modified() {
                            files.push((path, modified));
                        }
                    }
                }
            }
        }

        // Sort by modification time (oldest first)
        files.sort_by(|a, b| a.1.cmp(&b.1));

        // Remove files exceeding max_files
        while files.len() > self.config.max_files {
            if let Some((path, _)) = files.first() {
                fs::remove_file(path).map_err(|e| {
                    AuditError::RotationFailed(format!("Failed to remove old file: {}", e))
                })?;
                files.remove(0);
            }
        }

        // Remove files older than retention period
        if self.config.retention_days > 0 {
            let retention_threshold = SystemTime::now()
                .checked_sub(std::time::Duration::from_secs(
                    self.config.retention_days as u64 * 86400,
                ))
                .unwrap_or(UNIX_EPOCH);

            for (path, modified) in files.iter() {
                if *modified < retention_threshold {
                    let _ = fs::remove_file(path);
                }
            }
        }

        Ok(())
    }

    /// Verify integrity of all audit logs
    pub fn verify_integrity(&self) -> AuditResult<Vec<AuditEvent>> {
        let mut corrupted = vec![];
        let mut prev_hash = "genesis".to_string();

        let mut files: Vec<PathBuf> = vec![];
        if let Ok(entries) = fs::read_dir(&self.config.log_directory) {
            for entry in entries.filter_map(|e| e.ok()) {
                let path = entry.path();
                if path.extension().map(|e| e == "audit").unwrap_or(false) {
                    files.push(path);
                }
            }
        }
        files.sort();

        for file_path in files {
            if let Ok(file) = File::open(&file_path) {
                let reader = BufReader::new(file);
                for line in reader.lines().filter_map(|l| l.ok()) {
                    if let Ok(event) = serde_json::from_str::<AuditEvent>(&line) {
                        // Verify chain
                        if event.prev_hash != prev_hash {
                            corrupted.push(event.clone());
                        }
                        // Verify self-hash
                        if !event.verify_integrity() {
                            corrupted.push(event.clone());
                        }
                        prev_hash = event.entry_hash.clone();
                    }
                }
            }
        }

        Ok(corrupted)
    }

    /// Get the next sequence number
    pub fn next_sequence(&self) -> u64 {
        self.sequence + 1
    }

    /// Get the previous hash
    pub fn prev_hash(&self) -> &str {
        &self.prev_hash
    }
}

// ============================================================================
// Audit Logger
// ============================================================================

/// Main audit logger for recording events
pub struct AuditLogger {
    config: AuditConfig,
    store: Arc<RwLock<AuditStore>>,
}

impl AuditLogger {
    /// Create a new audit logger
    pub fn new(config: AuditConfig) -> AuditResult<Self> {
        let store = AuditStore::new(config.clone())?;
        Ok(Self {
            config,
            store: Arc::new(RwLock::new(store)),
        })
    }

    /// Check if an event should be logged based on configuration
    fn should_log(&self, event_type: &AuditEventType, severity: &AuditSeverity) -> bool {
        if !self.config.enabled {
            return false;
        }

        if *severity < self.config.min_severity {
            return false;
        }

        if !self.config.exclude_event_types.is_empty()
            && self.config.exclude_event_types.contains(event_type)
        {
            return false;
        }

        if !self.config.include_event_types.is_empty()
            && !self.config.include_event_types.contains(event_type)
        {
            return false;
        }

        true
    }

    /// Log an audit event from a builder
    pub fn log(&self, builder: AuditEventBuilder) -> AuditResult<String> {
        if !self.should_log(&builder.event_type, &builder.severity) {
            return Ok(String::new());
        }

        let mut store = self
            .store
            .write()
            .map_err(|e| AuditError::StorageError(format!("Lock poisoned: {}", e)))?;

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();

        let timestamp_ms = now.as_millis() as u64;
        let timestamp = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true);

        let sequence = store.next_sequence();
        let prev_hash = store.prev_hash().to_string();

        let mut event = AuditEvent {
            id: Uuid::new_v4().to_string(),
            timestamp,
            timestamp_ms,
            event_type: builder.event_type,
            action: builder.action,
            outcome: builder.outcome,
            severity: builder.severity,
            actor: builder.actor,
            resources: builder.resources,
            description: builder.description,
            details: builder.details,
            old_value: builder.old_value,
            new_value: builder.new_value,
            error_message: builder.error_message,
            error_code: builder.error_code,
            duration_ms: builder.duration_ms,
            request_id: builder.request_id,
            transaction_id: builder.transaction_id,
            parent_event_id: builder.parent_event_id,
            server_id: self.config.server_id.clone(),
            service_name: self.config.service_name.clone(),
            service_version: self.config.service_version.clone(),
            environment: self.config.environment.clone(),
            compliance_tags: builder.compliance_tags,
            prev_hash,
            entry_hash: String::new(),
            sequence,
        };

        // Calculate hash after all fields are set
        if self.config.tamper_detection {
            event.entry_hash = event.calculate_hash();
        }

        let event_id = event.id.clone();
        store.write(event)?;

        Ok(event_id)
    }

    /// Log a simple authentication event
    pub fn log_auth(
        &self,
        action: AuditAction,
        actor: AuditActor,
        outcome: AuditOutcome,
        description: &str,
    ) -> AuditResult<String> {
        self.log(
            AuditEventBuilder::new(AuditEventType::Authentication, action)
                .actor(actor)
                .outcome(outcome)
                .description(description)
                .soc2(),
        )
    }

    /// Log a query event
    pub fn log_query(
        &self,
        actor: AuditActor,
        resource: AuditResource,
        query: &str,
        duration_ms: u64,
    ) -> AuditResult<String> {
        self.log(
            AuditEventBuilder::new(AuditEventType::Query, AuditAction::QueryExecuted)
                .actor(actor)
                .resource(resource)
                .description(format!("Query executed: {}", truncate_string(query, 100)))
                .detail("query", serde_json::Value::String(query.to_string()))
                .duration_ms(duration_ms),
        )
    }

    /// Log a data modification event
    pub fn log_data_change(
        &self,
        actor: AuditActor,
        resource: AuditResource,
        action: AuditAction,
        old_value: Option<serde_json::Value>,
        new_value: Option<serde_json::Value>,
    ) -> AuditResult<String> {
        let description = format!("Data modification: {:?}", action);
        let mut builder = AuditEventBuilder::new(AuditEventType::DataModification, action)
            .actor(actor)
            .resource(resource)
            .description(description)
            .hipaa()
            .soc2();

        if let Some(old) = old_value {
            builder = builder.old_value(old);
        }
        if let Some(new) = new_value {
            builder = builder.new_value(new);
        }

        self.log(builder)
    }

    /// Log an admin action
    pub fn log_admin(
        &self,
        actor: AuditActor,
        action: AuditAction,
        description: &str,
        details: HashMap<String, serde_json::Value>,
    ) -> AuditResult<String> {
        let mut builder = AuditEventBuilder::new(AuditEventType::AdminAction, action)
            .actor(actor)
            .description(description)
            .severity(AuditSeverity::Notice)
            .soc2();

        for (key, value) in details {
            builder = builder.detail(key, value);
        }

        self.log(builder)
    }

    /// Log a security event
    pub fn log_security(
        &self,
        actor: AuditActor,
        action: AuditAction,
        severity: AuditSeverity,
        description: &str,
    ) -> AuditResult<String> {
        self.log(
            AuditEventBuilder::new(AuditEventType::SecurityEvent, action)
                .actor(actor)
                .severity(severity)
                .description(description)
                .soc2()
                .hipaa(),
        )
    }

    /// Verify audit log integrity
    pub fn verify_integrity(&self) -> AuditResult<Vec<AuditEvent>> {
        let store = self
            .store
            .read()
            .map_err(|e| AuditError::StorageError(format!("Lock poisoned: {}", e)))?;
        store.verify_integrity()
    }

    /// Force log rotation
    pub fn rotate(&self) -> AuditResult<()> {
        let mut store = self
            .store
            .write()
            .map_err(|e| AuditError::StorageError(format!("Lock poisoned: {}", e)))?;
        store.rotate()
    }
}

// ============================================================================
// Audit Query Interface
// ============================================================================

/// Time range for queries
#[derive(Debug, Clone)]
pub struct TimeRange {
    pub start: Option<u64>,
    pub end: Option<u64>,
}

impl TimeRange {
    pub fn new() -> Self {
        Self {
            start: None,
            end: None,
        }
    }

    pub fn from(start: u64) -> Self {
        Self {
            start: Some(start),
            end: None,
        }
    }

    pub fn until(end: u64) -> Self {
        Self {
            start: None,
            end: Some(end),
        }
    }

    pub fn between(start: u64, end: u64) -> Self {
        Self {
            start: Some(start),
            end: Some(end),
        }
    }

    pub fn last_hours(hours: u64) -> Self {
        let end = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let start = end.saturating_sub(hours * 3600 * 1000);
        Self::between(start, end)
    }

    pub fn last_days(days: u64) -> Self {
        Self::last_hours(days * 24)
    }
}

impl Default for TimeRange {
    fn default() -> Self {
        Self::new()
    }
}

/// Query parameters for searching audit logs
#[derive(Debug, Clone, Default)]
pub struct AuditQuery {
    /// Filter by event types
    pub event_types: Vec<AuditEventType>,
    /// Filter by actions
    pub actions: Vec<AuditAction>,
    /// Filter by outcomes
    pub outcomes: Vec<AuditOutcome>,
    /// Filter by actor ID
    pub actor_id: Option<String>,
    /// Filter by resource type
    pub resource_type: Option<String>,
    /// Filter by resource ID
    pub resource_id: Option<String>,
    /// Filter by time range
    pub time_range: TimeRange,
    /// Filter by minimum severity
    pub min_severity: Option<AuditSeverity>,
    /// Filter by compliance tags
    pub compliance_tags: Vec<String>,
    /// Filter by request ID
    pub request_id: Option<String>,
    /// Filter by transaction ID
    pub transaction_id: Option<String>,
    /// Text search in description
    pub description_contains: Option<String>,
    /// Maximum number of results
    pub limit: Option<usize>,
    /// Offset for pagination
    pub offset: usize,
    /// Sort by field
    pub sort_by: SortField,
    /// Sort order
    pub sort_order: SortOrder,
}

/// Fields available for sorting
#[derive(Debug, Clone, Copy, Default)]
pub enum SortField {
    #[default]
    Timestamp,
    Severity,
    EventType,
    ActorId,
}

/// Sort order
#[derive(Debug, Clone, Copy, Default)]
pub enum SortOrder {
    Ascending,
    #[default]
    Descending,
}

impl AuditQuery {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn event_type(mut self, event_type: AuditEventType) -> Self {
        self.event_types.push(event_type);
        self
    }

    pub fn action(mut self, action: AuditAction) -> Self {
        self.actions.push(action);
        self
    }

    pub fn outcome(mut self, outcome: AuditOutcome) -> Self {
        self.outcomes.push(outcome);
        self
    }

    pub fn actor(mut self, actor_id: impl Into<String>) -> Self {
        self.actor_id = Some(actor_id.into());
        self
    }

    pub fn resource(
        mut self,
        resource_type: impl Into<String>,
        resource_id: impl Into<String>,
    ) -> Self {
        self.resource_type = Some(resource_type.into());
        self.resource_id = Some(resource_id.into());
        self
    }

    pub fn time_range(mut self, range: TimeRange) -> Self {
        self.time_range = range;
        self
    }

    pub fn min_severity(mut self, severity: AuditSeverity) -> Self {
        self.min_severity = Some(severity);
        self
    }

    pub fn compliance_tag(mut self, tag: impl Into<String>) -> Self {
        self.compliance_tags.push(tag.into());
        self
    }

    pub fn request_id(mut self, id: impl Into<String>) -> Self {
        self.request_id = Some(id.into());
        self
    }

    pub fn transaction_id(mut self, id: impl Into<String>) -> Self {
        self.transaction_id = Some(id.into());
        self
    }

    pub fn description_contains(mut self, text: impl Into<String>) -> Self {
        self.description_contains = Some(text.into());
        self
    }

    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    pub fn offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }

    pub fn sort(mut self, field: SortField, order: SortOrder) -> Self {
        self.sort_by = field;
        self.sort_order = order;
        self
    }
}

/// Query results with pagination info
#[derive(Debug, Clone)]
pub struct AuditQueryResult {
    pub events: Vec<AuditEvent>,
    pub total_count: usize,
    pub offset: usize,
    pub limit: Option<usize>,
    pub has_more: bool,
}

/// Audit query executor
pub struct AuditQueryExecutor {
    log_directory: PathBuf,
}

impl AuditQueryExecutor {
    pub fn new(log_directory: PathBuf) -> Self {
        Self { log_directory }
    }

    /// Execute an audit query
    pub fn execute(&self, query: &AuditQuery) -> AuditResult<AuditQueryResult> {
        let mut all_events = Vec::new();

        // Read all audit files
        let mut files: Vec<PathBuf> = vec![];
        if let Ok(entries) = fs::read_dir(&self.log_directory) {
            for entry in entries.filter_map(|e| e.ok()) {
                let path = entry.path();
                if path.extension().map(|e| e == "audit").unwrap_or(false) {
                    files.push(path);
                }
            }
        }
        files.sort();

        for file_path in files {
            if let Ok(file) = File::open(&file_path) {
                let reader = BufReader::new(file);
                for line in reader.lines().filter_map(|l| l.ok()) {
                    if let Ok(event) = serde_json::from_str::<AuditEvent>(&line) {
                        if self.matches_query(&event, query) {
                            all_events.push(event);
                        }
                    }
                }
            }
        }

        // Sort results
        match (query.sort_by, query.sort_order) {
            (SortField::Timestamp, SortOrder::Ascending) => {
                all_events.sort_by(|a, b| a.timestamp_ms.cmp(&b.timestamp_ms));
            }
            (SortField::Timestamp, SortOrder::Descending) => {
                all_events.sort_by(|a, b| b.timestamp_ms.cmp(&a.timestamp_ms));
            }
            (SortField::Severity, SortOrder::Ascending) => {
                all_events.sort_by(|a, b| a.severity.cmp(&b.severity));
            }
            (SortField::Severity, SortOrder::Descending) => {
                all_events.sort_by(|a, b| b.severity.cmp(&a.severity));
            }
            (SortField::EventType, SortOrder::Ascending) => {
                all_events.sort_by(|a, b| {
                    format!("{:?}", a.event_type).cmp(&format!("{:?}", b.event_type))
                });
            }
            (SortField::EventType, SortOrder::Descending) => {
                all_events.sort_by(|a, b| {
                    format!("{:?}", b.event_type).cmp(&format!("{:?}", a.event_type))
                });
            }
            (SortField::ActorId, SortOrder::Ascending) => {
                all_events.sort_by(|a, b| a.actor.id.cmp(&b.actor.id));
            }
            (SortField::ActorId, SortOrder::Descending) => {
                all_events.sort_by(|a, b| b.actor.id.cmp(&a.actor.id));
            }
        }

        let total_count = all_events.len();

        // Apply pagination
        let events: Vec<AuditEvent> = all_events
            .into_iter()
            .skip(query.offset)
            .take(query.limit.unwrap_or(usize::MAX))
            .collect();

        let returned_count = events.len();
        let has_more = query.offset + returned_count < total_count;

        Ok(AuditQueryResult {
            events,
            total_count,
            offset: query.offset,
            limit: query.limit,
            has_more,
        })
    }

    /// Check if an event matches the query criteria
    fn matches_query(&self, event: &AuditEvent, query: &AuditQuery) -> bool {
        // Event type filter
        if !query.event_types.is_empty() && !query.event_types.contains(&event.event_type) {
            return false;
        }

        // Action filter
        if !query.actions.is_empty() && !query.actions.contains(&event.action) {
            return false;
        }

        // Outcome filter
        if !query.outcomes.is_empty() && !query.outcomes.contains(&event.outcome) {
            return false;
        }

        // Actor filter
        if let Some(ref actor_id) = query.actor_id {
            if event.actor.id != *actor_id {
                return false;
            }
        }

        // Resource filter
        if let Some(ref resource_type) = query.resource_type {
            if !event
                .resources
                .iter()
                .any(|r| r.resource_type == *resource_type)
            {
                return false;
            }
        }
        if let Some(ref resource_id) = query.resource_id {
            if !event
                .resources
                .iter()
                .any(|r| r.resource_id == *resource_id)
            {
                return false;
            }
        }

        // Time range filter
        if let Some(start) = query.time_range.start {
            if event.timestamp_ms < start {
                return false;
            }
        }
        if let Some(end) = query.time_range.end {
            if event.timestamp_ms > end {
                return false;
            }
        }

        // Severity filter
        if let Some(min_severity) = query.min_severity {
            if event.severity < min_severity {
                return false;
            }
        }

        // Compliance tags filter
        if !query.compliance_tags.is_empty() {
            if !query
                .compliance_tags
                .iter()
                .any(|tag| event.compliance_tags.contains(tag))
            {
                return false;
            }
        }

        // Request ID filter
        if let Some(ref request_id) = query.request_id {
            if event.request_id.as_ref() != Some(request_id) {
                return false;
            }
        }

        // Transaction ID filter
        if let Some(ref transaction_id) = query.transaction_id {
            if event.transaction_id.as_ref() != Some(transaction_id) {
                return false;
            }
        }

        // Description contains filter
        if let Some(ref text) = query.description_contains {
            if !event
                .description
                .to_lowercase()
                .contains(&text.to_lowercase())
            {
                return false;
            }
        }

        true
    }

    /// Get statistics about audit logs
    pub fn get_statistics(&self) -> AuditResult<AuditStatistics> {
        let mut stats = AuditStatistics::default();

        if let Ok(entries) = fs::read_dir(&self.log_directory) {
            for entry in entries.filter_map(|e| e.ok()) {
                let path = entry.path();
                if path.extension().map(|e| e == "audit").unwrap_or(false) {
                    stats.file_count += 1;
                    if let Ok(metadata) = path.metadata() {
                        stats.total_size_bytes += metadata.len();
                    }

                    if let Ok(file) = File::open(&path) {
                        let reader = BufReader::new(file);
                        for line in reader.lines().filter_map(|l| l.ok()) {
                            if let Ok(event) = serde_json::from_str::<AuditEvent>(&line) {
                                stats.total_events += 1;
                                *stats.events_by_type.entry(event.event_type).or_insert(0) += 1;
                                *stats.events_by_outcome.entry(event.outcome).or_insert(0) += 1;

                                if stats.oldest_event.is_none()
                                    || event.timestamp_ms < stats.oldest_event.unwrap_or(u64::MAX)
                                {
                                    stats.oldest_event = Some(event.timestamp_ms);
                                }
                                if stats.newest_event.is_none()
                                    || event.timestamp_ms > stats.newest_event.unwrap_or(0)
                                {
                                    stats.newest_event = Some(event.timestamp_ms);
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(stats)
    }
}

/// Audit log statistics
#[derive(Debug, Clone, Default)]
pub struct AuditStatistics {
    pub total_events: usize,
    pub file_count: usize,
    pub total_size_bytes: u64,
    pub events_by_type: HashMap<AuditEventType, usize>,
    pub events_by_outcome: HashMap<AuditOutcome, usize>,
    pub oldest_event: Option<u64>,
    pub newest_event: Option<u64>,
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Truncate a string to a maximum length
fn truncate_string(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_config() -> (AuditConfig, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let config = AuditConfig {
            log_directory: temp_dir.path().to_path_buf(),
            max_file_size: 1024 * 1024, // 1MB for tests
            max_files: 5,
            retention_days: 30,
            ..Default::default()
        };
        (config, temp_dir)
    }

    fn create_test_actor() -> AuditActor {
        AuditActor {
            id: "user-123".to_string(),
            actor_type: "user".to_string(),
            name: Some("Test User".to_string()),
            email: Some("test@example.com".to_string()),
            roles: vec!["admin".to_string()],
            session_id: Some("session-456".to_string()),
            ip_address: Some("192.168.1.1".to_string()),
            user_agent: Some("Mozilla/5.0".to_string()),
            geo_location: None,
        }
    }

    #[test]
    fn test_audit_event_hash_calculation() {
        let event = AuditEvent {
            id: "event-1".to_string(),
            timestamp: "2024-01-01T00:00:00.000Z".to_string(),
            timestamp_ms: 1704067200000,
            event_type: AuditEventType::Authentication,
            action: AuditAction::LoginSuccess,
            outcome: AuditOutcome::Success,
            severity: AuditSeverity::Info,
            actor: AuditActor::default(),
            resources: vec![],
            description: "User logged in".to_string(),
            details: HashMap::new(),
            old_value: None,
            new_value: None,
            error_message: None,
            error_code: None,
            duration_ms: None,
            request_id: None,
            transaction_id: None,
            parent_event_id: None,
            server_id: "server-1".to_string(),
            service_name: "joule_db".to_string(),
            service_version: "1.0.0".to_string(),
            environment: "test".to_string(),
            compliance_tags: vec!["SOC2".to_string()],
            prev_hash: "genesis".to_string(),
            entry_hash: String::new(),
            sequence: 1,
        };

        let hash = event.calculate_hash();
        assert!(!hash.is_empty());
        assert_eq!(hash.len(), 64); // SHA-256 hex length
    }

    #[test]
    fn test_audit_event_integrity_verification() {
        let mut event = AuditEvent {
            id: "event-1".to_string(),
            timestamp: "2024-01-01T00:00:00.000Z".to_string(),
            timestamp_ms: 1704067200000,
            event_type: AuditEventType::Authentication,
            action: AuditAction::LoginSuccess,
            outcome: AuditOutcome::Success,
            severity: AuditSeverity::Info,
            actor: AuditActor::default(),
            resources: vec![],
            description: "User logged in".to_string(),
            details: HashMap::new(),
            old_value: None,
            new_value: None,
            error_message: None,
            error_code: None,
            duration_ms: None,
            request_id: None,
            transaction_id: None,
            parent_event_id: None,
            server_id: "server-1".to_string(),
            service_name: "joule_db".to_string(),
            service_version: "1.0.0".to_string(),
            environment: "test".to_string(),
            compliance_tags: vec![],
            prev_hash: "genesis".to_string(),
            entry_hash: String::new(),
            sequence: 1,
        };

        event.entry_hash = event.calculate_hash();
        assert!(event.verify_integrity());

        // Tamper with the event
        event.description = "Tampered description".to_string();
        assert!(!event.verify_integrity());
    }

    #[test]
    fn test_audit_logger_basic_logging() {
        let (config, _temp_dir) = create_test_config();
        let logger = AuditLogger::new(config).unwrap();

        let event_id = logger
            .log(
                AuditEventBuilder::new(AuditEventType::Authentication, AuditAction::LoginSuccess)
                    .actor(create_test_actor())
                    .description("User logged in successfully")
                    .outcome(AuditOutcome::Success),
            )
            .unwrap();

        assert!(!event_id.is_empty());
    }

    #[test]
    fn test_audit_logger_auth_event() {
        let (config, _temp_dir) = create_test_config();
        let logger = AuditLogger::new(config).unwrap();

        let event_id = logger
            .log_auth(
                AuditAction::LoginSuccess,
                create_test_actor(),
                AuditOutcome::Success,
                "User authenticated via password",
            )
            .unwrap();

        assert!(!event_id.is_empty());
    }

    #[test]
    fn test_audit_logger_query_event() {
        let (config, _temp_dir) = create_test_config();
        let logger = AuditLogger::new(config).unwrap();

        let resource =
            AuditResource::new("table", "users").with_path("/databases/main/tables/users");

        let event_id = logger
            .log_query(
                create_test_actor(),
                resource,
                "SELECT * FROM users WHERE id = 123",
                150,
            )
            .unwrap();

        assert!(!event_id.is_empty());
    }

    #[test]
    fn test_audit_logger_data_change_event() {
        let (config, _temp_dir) = create_test_config();
        let logger = AuditLogger::new(config).unwrap();

        let resource =
            AuditResource::new("document", "doc-123").with_attribute("collection", "users");

        let old_value = serde_json::json!({"name": "John", "age": 30});
        let new_value = serde_json::json!({"name": "John", "age": 31});

        let event_id = logger
            .log_data_change(
                create_test_actor(),
                resource,
                AuditAction::RecordUpdated,
                Some(old_value),
                Some(new_value),
            )
            .unwrap();

        assert!(!event_id.is_empty());
    }

    #[test]
    fn test_audit_query_executor() {
        let (config, _temp_dir) = create_test_config();
        let log_dir = config.log_directory.clone();
        let logger = AuditLogger::new(config).unwrap();

        // Log several events
        for i in 0..5 {
            let mut actor = create_test_actor();
            actor.id = format!("user-{}", i);

            logger
                .log(
                    AuditEventBuilder::new(
                        AuditEventType::Authentication,
                        AuditAction::LoginSuccess,
                    )
                    .actor(actor)
                    .description(format!("Login event {}", i)),
                )
                .unwrap();
        }

        // Query events
        let executor = AuditQueryExecutor::new(log_dir);
        let query = AuditQuery::new().event_type(AuditEventType::Authentication);

        let result = executor.execute(&query).unwrap();
        assert_eq!(result.total_count, 5);
        assert_eq!(result.events.len(), 5);
    }

    #[test]
    fn test_audit_query_with_filters() {
        let (config, _temp_dir) = create_test_config();
        let log_dir = config.log_directory.clone();
        let logger = AuditLogger::new(config).unwrap();

        // Log authentication events
        let actor = create_test_actor();
        logger
            .log_auth(
                AuditAction::LoginSuccess,
                actor.clone(),
                AuditOutcome::Success,
                "Successful login",
            )
            .unwrap();

        logger
            .log_auth(
                AuditAction::LoginFailure,
                actor.clone(),
                AuditOutcome::Failure,
                "Failed login",
            )
            .unwrap();

        // Log query event
        let resource = AuditResource::new("table", "users");
        logger
            .log_query(actor, resource, "SELECT * FROM users", 100)
            .unwrap();

        // Query only failures
        let executor = AuditQueryExecutor::new(log_dir);
        let query = AuditQuery::new().outcome(AuditOutcome::Failure);

        let result = executor.execute(&query).unwrap();
        assert_eq!(result.total_count, 1);
        assert_eq!(result.events[0].action, AuditAction::LoginFailure);
    }

    #[test]
    fn test_audit_event_builder() {
        let builder =
            AuditEventBuilder::new(AuditEventType::DataModification, AuditAction::RecordCreated)
                .actor(create_test_actor())
                .resource(AuditResource::new("document", "doc-1"))
                .description("Created new document")
                .detail("size_bytes", serde_json::json!(1024))
                .new_value(serde_json::json!({"key": "value"}))
                .duration_ms(50)
                .request_id("req-123")
                .transaction_id("txn-456")
                .hipaa()
                .soc2()
                .gdpr();

        assert_eq!(builder.event_type, AuditEventType::DataModification);
        assert_eq!(builder.action, AuditAction::RecordCreated);
        assert!(builder.compliance_tags.contains(&"HIPAA".to_string()));
        assert!(builder.compliance_tags.contains(&"SOC2".to_string()));
        assert!(builder.compliance_tags.contains(&"GDPR".to_string()));
    }

    #[test]
    fn test_audit_statistics() {
        let (config, _temp_dir) = create_test_config();
        let log_dir = config.log_directory.clone();
        let logger = AuditLogger::new(config).unwrap();

        // Log various events
        let actor = create_test_actor();
        for _ in 0..3 {
            logger
                .log_auth(
                    AuditAction::LoginSuccess,
                    actor.clone(),
                    AuditOutcome::Success,
                    "Login",
                )
                .unwrap();
        }

        let resource = AuditResource::new("table", "users");
        for _ in 0..2 {
            logger
                .log_query(actor.clone(), resource.clone(), "SELECT 1", 10)
                .unwrap();
        }

        let executor = AuditQueryExecutor::new(log_dir);
        let stats = executor.get_statistics().unwrap();

        assert_eq!(stats.total_events, 5);
        assert_eq!(
            *stats
                .events_by_type
                .get(&AuditEventType::Authentication)
                .unwrap_or(&0),
            3
        );
        assert_eq!(
            *stats
                .events_by_type
                .get(&AuditEventType::Query)
                .unwrap_or(&0),
            2
        );
    }

    #[test]
    fn test_audit_store_integrity_verification() {
        let (config, _temp_dir) = create_test_config();
        let logger = AuditLogger::new(config).unwrap();

        // Log events
        let actor = create_test_actor();
        for i in 0..5 {
            logger
                .log(
                    AuditEventBuilder::new(
                        AuditEventType::SystemEvent,
                        AuditAction::ServiceStarted,
                    )
                    .actor(actor.clone())
                    .description(format!("Event {}", i)),
                )
                .unwrap();
        }

        // Verify integrity
        let corrupted = logger.verify_integrity().unwrap();
        assert!(corrupted.is_empty(), "Expected no corrupted entries");
    }

    #[test]
    fn test_time_range_queries() {
        let range = TimeRange::last_hours(24);
        assert!(range.start.is_some());
        assert!(range.end.is_some());

        let range = TimeRange::last_days(7);
        assert!(range.start.is_some());

        let range = TimeRange::between(1000, 2000);
        assert_eq!(range.start, Some(1000));
        assert_eq!(range.end, Some(2000));
    }

    #[test]
    fn test_audit_severity_ordering() {
        assert!(AuditSeverity::Debug < AuditSeverity::Info);
        assert!(AuditSeverity::Info < AuditSeverity::Warning);
        assert!(AuditSeverity::Warning < AuditSeverity::Error);
        assert!(AuditSeverity::Error < AuditSeverity::Critical);
        assert!(AuditSeverity::Critical < AuditSeverity::Emergency);
    }

    #[test]
    fn test_audit_resource_builder() {
        let resource = AuditResource::new("table", "users")
            .with_path("/db/main/users")
            .with_attribute("schema", "public")
            .with_attribute("database", "main");

        assert_eq!(resource.resource_type, "table");
        assert_eq!(resource.resource_id, "users");
        assert_eq!(resource.path, Some("/db/main/users".to_string()));
        assert_eq!(
            resource.attributes.get("schema"),
            Some(&"public".to_string())
        );
    }

    #[test]
    fn test_disabled_audit_logger() {
        let (mut config, _temp_dir) = create_test_config();
        config.enabled = false;
        let logger = AuditLogger::new(config).unwrap();

        let event_id = logger
            .log(
                AuditEventBuilder::new(AuditEventType::Authentication, AuditAction::LoginSuccess)
                    .actor(create_test_actor())
                    .description("This should not be logged"),
            )
            .unwrap();

        // When disabled, returns empty string
        assert!(event_id.is_empty());
    }

    #[test]
    fn test_min_severity_filter() {
        let (mut config, _temp_dir) = create_test_config();
        config.min_severity = AuditSeverity::Warning;
        let log_dir = config.log_directory.clone();
        let logger = AuditLogger::new(config).unwrap();

        // This should not be logged (Info < Warning)
        let event_id1 = logger
            .log(
                AuditEventBuilder::new(AuditEventType::Query, AuditAction::QueryExecuted)
                    .severity(AuditSeverity::Info)
                    .description("Info event"),
            )
            .unwrap();

        // This should be logged (Warning >= Warning)
        let event_id2 = logger
            .log(
                AuditEventBuilder::new(
                    AuditEventType::SecurityEvent,
                    AuditAction::SuspiciousActivity,
                )
                .severity(AuditSeverity::Warning)
                .description("Warning event"),
            )
            .unwrap();

        assert!(event_id1.is_empty());
        assert!(!event_id2.is_empty());

        let executor = AuditQueryExecutor::new(log_dir);
        let result = executor.execute(&AuditQuery::new()).unwrap();
        assert_eq!(result.total_count, 1);
    }
}
