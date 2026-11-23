//! Persistence configuration

/// Configuration options for persistence backends
#[derive(Debug, Clone)]
pub struct PersistenceConfig {
    /// Database name
    pub db_name: String,
    /// Enable compression
    pub compression_enabled: bool,
    /// Encryption key (if encryption is enabled)
    pub encryption_key: Option<Vec<u8>>,
    /// Page cache size (number of pages)
    pub cache_size: usize,
    /// Enable WAL (Write-Ahead Logging)
    pub wal_enabled: bool,
    /// Sync mode (sync after every write)
    pub sync_on_write: bool,
}

impl Default for PersistenceConfig {
    fn default() -> Self {
        Self {
            db_name: "joule_db".to_string(),
            compression_enabled: false,
            encryption_key: None,
            cache_size: 1024,
            wal_enabled: true,
            sync_on_write: false,
        }
    }
}

impl PersistenceConfig {
    /// Create a new config with the given database name
    pub fn new(db_name: &str) -> Self {
        Self {
            db_name: db_name.to_string(),
            ..Default::default()
        }
    }

    /// Enable compression
    pub fn with_compression(mut self) -> Self {
        self.compression_enabled = true;
        self
    }

    /// Set encryption key
    pub fn with_encryption(mut self, key: Vec<u8>) -> Self {
        self.encryption_key = Some(key);
        self
    }

    /// Set cache size
    pub fn with_cache_size(mut self, size: usize) -> Self {
        self.cache_size = size;
        self
    }

    /// Disable WAL
    pub fn without_wal(mut self) -> Self {
        self.wal_enabled = false;
        self
    }

    /// Enable sync on write
    pub fn with_sync_on_write(mut self) -> Self {
        self.sync_on_write = true;
        self
    }
}

/// Backend type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    /// In-memory storage (no persistence)
    Memory,
    /// IndexedDB storage (browser)
    IndexedDB,
    /// OPFS storage (browser, high performance)
    OPFS,
    /// OPFS Virtual Block Device (browser, maximum performance)
    OPFSVBD,
}

impl std::fmt::Display for BackendType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Memory => write!(f, "Memory"),
            Self::IndexedDB => write!(f, "IndexedDB"),
            Self::OPFS => write!(f, "OPFS"),
            Self::OPFSVBD => write!(f, "OPFS-VBD"),
        }
    }
}

/// Backend selection result
#[derive(Debug, Clone)]
pub struct BackendSelection {
    /// Selected backend type
    pub backend_type: BackendType,
    /// Reason for selection
    pub reason: String,
}

impl BackendSelection {
    /// Create a new backend selection
    pub fn new(backend_type: BackendType, reason: &str) -> Self {
        Self {
            backend_type,
            reason: reason.to_string(),
        }
    }
}
