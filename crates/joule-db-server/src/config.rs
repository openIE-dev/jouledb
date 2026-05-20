//! Configuration Management for JouleDB Server
//!
//! Provides configuration loading, validation, and management

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Server Configuration
///
/// Complete configuration for JouleDB Server
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Server settings
    pub server: ServerSettings,
    /// Database settings
    pub database: DatabaseSettings,
    /// Network settings
    pub network: NetworkSettings,
    /// Security settings
    pub security: SecuritySettings,
    /// Performance settings
    pub performance: PerformanceSettings,
    /// Logging settings
    pub logging: LoggingSettings,
    /// Sharding settings
    #[serde(default)]
    pub sharding: ShardingSettings,
}

/// Server settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerSettings {
    /// Number of CPU cores to use
    pub num_cores: Option<usize>,
    /// Bind address
    pub bind_address: String,
    /// Port
    pub port: u16,
    /// Worker threads
    pub worker_threads: Option<usize>,
    /// Graceful shutdown timeout (seconds)
    pub shutdown_timeout: u64,
}

impl Default for ServerSettings {
    fn default() -> Self {
        Self {
            num_cores: None, // Auto-detect
            bind_address: "0.0.0.0".to_string(),
            port: 8080,
            worker_threads: None,
            shutdown_timeout: 30,
        }
    }
}

/// Database settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseSettings {
    /// Data directory
    pub data_dir: String,
    /// WAL directory
    pub wal_dir: String,
    /// Page size (bytes)
    pub page_size: usize,
    /// Cache size (bytes)
    pub cache_size: u64,
    /// Max database size (bytes)
    pub max_db_size: Option<u64>,
    /// Compression enabled
    pub compression: bool,
    /// Encryption enabled
    pub encryption: bool,
}

impl Default for DatabaseSettings {
    fn default() -> Self {
        Self {
            data_dir: "./data".to_string(),
            wal_dir: "./wal".to_string(),
            page_size: 4096,
            cache_size: 100 * 1024 * 1024, // 100MB
            max_db_size: None,
            compression: true,
            encryption: false,
        }
    }
}

/// Network settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkSettings {
    /// QUIC enabled
    pub quic_enabled: bool,
    /// HTTP/2 enabled
    pub http2_enabled: bool,
    /// Max connections
    pub max_connections: u32,
    /// Connection timeout (seconds)
    pub connection_timeout: u64,
    /// Keep-alive timeout (seconds)
    pub keep_alive_timeout: u64,
    /// TLS certificate path
    pub tls_cert_path: Option<String>,
    /// TLS key path
    pub tls_key_path: Option<String>,
}

impl Default for NetworkSettings {
    fn default() -> Self {
        Self {
            quic_enabled: false,
            http2_enabled: true,
            max_connections: 1000,
            connection_timeout: 60,
            keep_alive_timeout: 300,
            tls_cert_path: None,
            tls_key_path: None,
        }
    }
}

/// Security settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecuritySettings {
    /// Authentication enabled
    pub auth_enabled: bool,
    /// JWT secret
    pub jwt_secret: Option<String>,
    /// Rate limiting enabled
    pub rate_limiting_enabled: bool,
    /// Max requests per second
    pub max_requests_per_sec: u32,
    /// CORS allowed origins
    pub cors_origins: Vec<String>,
    /// API key required
    pub api_key_required: bool,
}

impl Default for SecuritySettings {
    fn default() -> Self {
        Self {
            auth_enabled: false,
            jwt_secret: None,
            rate_limiting_enabled: true,
            max_requests_per_sec: 100,
            cors_origins: Vec::new(),
            api_key_required: false,
        }
    }
}

/// Performance settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSettings {
    /// io_uring enabled
    pub io_uring_enabled: bool,
    /// Group commit enabled
    pub group_commit_enabled: bool,
    /// Group commit threshold (bytes)
    pub group_commit_threshold: usize,
    /// GPU compute enabled
    pub gpu_compute_enabled: bool,
    /// Batch size
    pub batch_size: usize,
}

impl Default for PerformanceSettings {
    fn default() -> Self {
        Self {
            io_uring_enabled: false,
            group_commit_enabled: true,
            group_commit_threshold: 4096,
            gpu_compute_enabled: false,
            batch_size: 1000,
        }
    }
}

/// Logging settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingSettings {
    /// Log level (trace, debug, info, warn, error)
    pub level: String,
    /// Log file path
    pub file_path: Option<String>,
    /// Console logging enabled
    pub console_enabled: bool,
    /// JSON format
    pub json_format: bool,
    /// Max log file size (bytes)
    pub max_file_size: u64,
    /// Max log files
    pub max_files: u32,
}

impl Default for LoggingSettings {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            file_path: None,
            console_enabled: true,
            json_format: false,
            max_file_size: 10 * 1024 * 1024, // 10MB
            max_files: 10,
        }
    }
}

/// Sharding settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardingSettings {
    /// Enable distributed query routing via consistent hash ring
    pub enabled: bool,
    /// Initial number of shards
    pub initial_shards: usize,
    /// Replication factor
    pub replication_factor: usize,
    /// Virtual nodes per shard (for consistent hashing)
    pub virtual_nodes: usize,
    /// Default shard key column name
    pub default_shard_key: String,
    /// Query timeout (seconds) for distributed queries
    pub query_timeout_secs: u64,
    /// Max parallel shard queries
    pub max_parallel_shards: usize,
    /// Enable consistent reads (quorum-based)
    pub consistent_reads: bool,
    /// Enable result cache for distributed queries
    pub enable_result_cache: bool,
    /// Max cache entries
    pub max_cache_entries: usize,
    /// Cache TTL (seconds)
    pub cache_ttl_secs: u64,
    /// Auto-rebalance when shard sizes diverge
    pub auto_rebalance: bool,
}

impl Default for ShardingSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            initial_shards: 16,
            replication_factor: 3,
            virtual_nodes: 150,
            default_shard_key: "id".to_string(),
            query_timeout_secs: 30,
            max_parallel_shards: 16,
            consistent_reads: false,
            enable_result_cache: true,
            max_cache_entries: 10_000,
            cache_ttl_secs: 60,
            auto_rebalance: true,
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            server: ServerSettings::default(),
            database: DatabaseSettings::default(),
            network: NetworkSettings::default(),
            security: SecuritySettings::default(),
            performance: PerformanceSettings::default(),
            logging: LoggingSettings::default(),
            sharding: ShardingSettings::default(),
        }
    }
}

/// Configuration Manager
pub struct ConfigManager {
    config: Arc<RwLock<ServerConfig>>,
    overrides: Arc<RwLock<HashMap<String, String>>>,
}

impl ConfigManager {
    /// Create new config manager
    pub fn new() -> Self {
        Self {
            config: Arc::new(RwLock::new(ServerConfig::default())),
            overrides: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Load configuration from file
    pub fn load_from_file(&self, path: &str) -> Result<(), String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read config file: {}", e))?;

        let config: ServerConfig =
            serde_json::from_str(&content).map_err(|e| format!("Failed to parse config: {}", e))?;

        *self
            .config
            .write()
            .map_err(|_| "Config lock poisoned".to_string())? = config;
        Ok(())
    }

    /// Load configuration from environment variables
    pub fn load_from_env(&self) -> Result<(), String> {
        let mut config = self
            .config
            .write()
            .map_err(|_| "Config lock poisoned".to_string())?;

        if let Ok(num_cores) = std::env::var("JOULE_DB_NUM_CORES") {
            if let Ok(cores) = num_cores.parse::<usize>() {
                config.server.num_cores = Some(cores);
            }
        }

        if let Ok(port) = std::env::var("JOULE_DB_PORT") {
            if let Ok(port_num) = port.parse::<u16>() {
                config.server.port = port_num;
            }
        }

        if let Ok(bind) = std::env::var("JOULE_DB_BIND_ADDRESS") {
            config.server.bind_address = bind;
        }

        if let Ok(data_dir) = std::env::var("JOULE_DB_DATA_DIR") {
            config.database.data_dir = data_dir;
        }

        if let Ok(log_level) = std::env::var("JOULE_DB_LOG_LEVEL") {
            config.logging.level = log_level;
        }

        if let Ok(auth_enabled) = std::env::var("JOULE_DB_AUTH_ENABLED") {
            config.security.auth_enabled = auth_enabled.parse().unwrap_or(false);
        }

        if let Ok(jwt_secret) = std::env::var("JOULE_DB_JWT_SECRET") {
            config.security.jwt_secret = Some(jwt_secret);
        }

        if let Ok(sharding) = std::env::var("JOULE_DB_SHARDING_ENABLED") {
            config.sharding.enabled = sharding.parse().unwrap_or(false);
        }

        if let Ok(shards) = std::env::var("JOULE_DB_SHARDING_INITIAL_SHARDS") {
            if let Ok(n) = shards.parse::<usize>() {
                config.sharding.initial_shards = n;
            }
        }

        if let Ok(rf) = std::env::var("JOULE_DB_SHARDING_REPLICATION_FACTOR") {
            if let Ok(n) = rf.parse::<usize>() {
                config.sharding.replication_factor = n;
            }
        }

        if let Ok(key) = std::env::var("JOULE_DB_SHARDING_DEFAULT_KEY") {
            config.sharding.default_shard_key = key;
        }

        Ok(())
    }

    /// Get configuration (recovers from lock poisoning)
    pub fn get(&self) -> ServerConfig {
        crate::lock_util::read_lock(&self.config).clone()
    }

    /// Get configuration safely
    pub fn try_get(&self) -> Result<ServerConfig, String> {
        Ok(crate::lock_util::read_lock(&self.config).clone())
    }

    /// Get configuration reference
    pub fn get_arc(&self) -> Arc<RwLock<ServerConfig>> {
        self.config.clone()
    }

    /// Set override
    pub fn set_override(&self, key: String, value: String) -> Result<(), String> {
        self.overrides
            .write()
            .map_err(|_| "Override lock poisoned".to_string())?
            .insert(key, value);
        Ok(())
    }

    /// Get override
    pub fn get_override(&self, key: &str) -> Option<String> {
        self.overrides.read().ok()?.get(key).cloned()
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let config = self
            .config
            .read()
            .map_err(|_| vec!["Config lock poisoned".to_string()])?;
        let mut errors = Vec::new();

        if config.server.port == 0 {
            errors.push("Port cannot be 0".to_string());
        }

        if config.database.page_size < 512 || config.database.page_size > 65536 {
            errors.push("Page size must be between 512 and 65536".to_string());
        }

        if config.database.cache_size == 0 {
            errors.push("Cache size cannot be 0".to_string());
        }

        if config.network.max_connections == 0 {
            errors.push("Max connections cannot be 0".to_string());
        }

        if config.security.auth_enabled && config.security.jwt_secret.is_none() {
            errors.push("JWT secret required when auth is enabled".to_string());
        }

        if config.network.quic_enabled && config.network.tls_cert_path.is_none() {
            errors.push("TLS certificate required for QUIC".to_string());
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Save configuration to file
    pub fn save_to_file(&self, path: &str) -> Result<(), String> {
        use std::io::Write;

        let config = self
            .config
            .read()
            .map_err(|_| "Config lock poisoned".to_string())?;
        let json = serde_json::to_string_pretty(&*config)
            .map_err(|e| format!("Failed to serialize config: {}", e))?;

        let mut file = std::fs::File::create(path)
            .map_err(|e| format!("Failed to create config file: {}", e))?;

        file.write_all(json.as_bytes())
            .map_err(|e| format!("Failed to write config file: {}", e))?;

        Ok(())
    }
}

impl Default for ConfigManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ServerConfig::default();
        assert_eq!(config.server.port, 8080);
        assert_eq!(config.database.page_size, 4096);
        assert!(!config.security.auth_enabled);
    }

    #[test]
    fn test_config_manager_validate() {
        let manager = ConfigManager::new();
        assert!(manager.validate().is_ok());
    }

    #[test]
    fn test_sharding_defaults() {
        let config = ServerConfig::default();
        assert!(!config.sharding.enabled);
        assert_eq!(config.sharding.initial_shards, 16);
        assert_eq!(config.sharding.replication_factor, 3);
        assert_eq!(config.sharding.virtual_nodes, 150);
        assert_eq!(config.sharding.default_shard_key, "id");
    }

    #[test]
    fn test_sharding_serde_roundtrip() {
        let config = ServerConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: ServerConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.sharding.enabled, config.sharding.enabled);
        assert_eq!(
            deserialized.sharding.initial_shards,
            config.sharding.initial_shards
        );
    }

    #[test]
    fn test_sharding_omitted_in_json_uses_defaults() {
        // If sharding section is missing from JSON, defaults should be used
        let json = r#"{
            "server": {"bind_address": "0.0.0.0", "port": 8080, "shutdown_timeout": 30},
            "database": {"data_dir": "./data", "wal_dir": "./wal", "page_size": 4096, "cache_size": 104857600, "compression": true, "encryption": false},
            "network": {"quic_enabled": false, "http2_enabled": true, "max_connections": 1000, "connection_timeout": 60, "keep_alive_timeout": 300},
            "security": {"auth_enabled": false, "rate_limiting_enabled": true, "max_requests_per_sec": 100, "cors_origins": [], "api_key_required": false},
            "performance": {"io_uring_enabled": false, "group_commit_enabled": true, "group_commit_threshold": 4096, "gpu_compute_enabled": false, "batch_size": 1000},
            "logging": {"level": "info", "console_enabled": true, "json_format": false, "max_file_size": 10485760, "max_files": 10}
        }"#;
        let config: ServerConfig = serde_json::from_str(json).unwrap();
        assert!(!config.sharding.enabled);
        assert_eq!(config.sharding.initial_shards, 16);
    }

    #[test]
    fn test_config_manager_override() {
        let manager = ConfigManager::new();
        manager.set_override("test_key".to_string(), "test_value".to_string());
        assert_eq!(
            manager.get_override("test_key"),
            Some("test_value".to_string())
        );
    }
}
