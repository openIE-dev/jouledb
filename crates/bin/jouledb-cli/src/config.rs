//! CLI Configuration Management

use crate::error::{CliError, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// CLI Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Default connection settings
    #[serde(default)]
    pub connection: ConnectionConfig,

    /// Cloud settings
    #[serde(default)]
    pub cloud: CloudConfig,

    /// Output preferences
    #[serde(default)]
    pub output: OutputConfig,

    /// Shell settings
    #[serde(default)]
    pub shell: ShellConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            connection: ConnectionConfig::default(),
            cloud: CloudConfig::default(),
            output: OutputConfig::default(),
            shell: ShellConfig::default(),
        }
    }
}

/// Connection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionConfig {
    /// Server host
    #[serde(default = "default_host")]
    pub host: String,

    /// Server port
    #[serde(default = "default_port")]
    pub port: u16,

    /// Default database
    pub database: Option<String>,

    /// Username
    pub username: Option<String>,

    /// Use TLS
    #[serde(default)]
    pub tls: bool,

    /// Connection timeout in seconds
    #[serde(default = "default_timeout")]
    pub timeout: u64,
}

fn default_host() -> String {
    "localhost".to_string()
}

fn default_port() -> u16 {
    9000
}

fn default_timeout() -> u64 {
    30
}

impl Default for ConnectionConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            database: None,
            username: None,
            tls: false,
            timeout: default_timeout(),
        }
    }
}

/// Cloud configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CloudConfig {
    /// Cloud API base URL
    #[serde(default = "default_cloud_url")]
    pub api_url: String,

    /// Current project ID
    pub project_id: Option<String>,

    /// Current cluster ID
    pub cluster_id: Option<String>,
}

fn default_cloud_url() -> String {
    "https://api.jouledb.cloud".to_string()
}

/// Output configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Default output format
    #[serde(default = "default_format")]
    pub format: String,

    /// Use colors
    #[serde(default = "default_true")]
    pub colors: bool,

    /// Table style
    #[serde(default = "default_table_style")]
    pub table_style: String,
}

fn default_format() -> String {
    "text".to_string()
}

fn default_true() -> bool {
    true
}

fn default_table_style() -> String {
    "rounded".to_string()
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            format: default_format(),
            colors: true,
            table_style: default_table_style(),
        }
    }
}

/// Shell configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShellConfig {
    /// History file path
    pub history_file: Option<String>,

    /// Max history size
    #[serde(default = "default_history_size")]
    pub history_size: usize,

    /// Enable auto-complete
    #[serde(default = "default_true")]
    pub autocomplete: bool,

    /// Enable syntax highlighting
    #[serde(default = "default_true")]
    pub syntax_highlighting: bool,

    /// Prompt format
    #[serde(default = "default_prompt")]
    pub prompt: String,
}

fn default_history_size() -> usize {
    1000
}

fn default_prompt() -> String {
    "jouledb> ".to_string()
}

impl Default for ShellConfig {
    fn default() -> Self {
        Self {
            history_file: None,
            history_size: default_history_size(),
            autocomplete: true,
            syntax_highlighting: true,
            prompt: default_prompt(),
        }
    }
}

impl Config {
    /// Load configuration from file
    pub fn load(path: Option<&str>) -> Result<Self> {
        let config_path = if let Some(p) = path {
            PathBuf::from(p)
        } else {
            Self::default_config_path()?
        };

        if config_path.exists() {
            let content = std::fs::read_to_string(&config_path)?;
            let config: Config = toml::from_str(&content)?;
            Ok(config)
        } else {
            Ok(Config::default())
        }
    }

    /// Save configuration to file
    pub fn save(&self, path: Option<&str>) -> Result<()> {
        let config_path = if let Some(p) = path {
            PathBuf::from(p)
        } else {
            Self::default_config_path()?
        };

        // Create parent directories
        if let Some(parent) = config_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let content = toml::to_string_pretty(self).map_err(|e| CliError::Config(e.to_string()))?;
        std::fs::write(&config_path, content)?;
        Ok(())
    }

    /// Get default config path
    pub fn default_config_path() -> Result<PathBuf> {
        let config_dir = dirs::config_dir()
            .ok_or_else(|| CliError::Config("Could not determine config directory".into()))?;
        Ok(config_dir.join("jouledb").join("config.toml"))
    }

    /// Get credentials path
    pub fn credentials_path() -> Result<PathBuf> {
        let config_dir = dirs::config_dir()
            .ok_or_else(|| CliError::Config("Could not determine config directory".into()))?;
        Ok(config_dir.join("jouledb").join("credentials.json"))
    }

    /// Get connection URL
    pub fn connection_url(&self) -> String {
        let scheme = if self.connection.tls {
            "jouledbs"
        } else {
            "jouledb"
        };
        let db = self.connection.database.as_deref().unwrap_or("default");
        format!(
            "{}://{}:{}/{}",
            scheme, self.connection.host, self.connection.port, db
        )
    }
}

/// Stored credentials
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Credentials {
    /// Access token
    pub access_token: Option<String>,

    /// Refresh token
    pub refresh_token: Option<String>,

    /// Token expiry
    pub expires_at: Option<i64>,

    /// User email
    pub email: Option<String>,
}

impl Credentials {
    /// Load credentials
    pub fn load() -> Result<Self> {
        let path = Config::credentials_path()?;
        if path.exists() {
            let content = std::fs::read_to_string(&path)?;
            let creds: Credentials = serde_json::from_str(&content)?;
            Ok(creds)
        } else {
            Ok(Credentials::default())
        }
    }

    /// Save credentials
    pub fn save(&self) -> Result<()> {
        let path = Config::credentials_path()?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, content)?;
        Ok(())
    }

    /// Clear credentials
    pub fn clear() -> Result<()> {
        let path = Config::credentials_path()?;
        if path.exists() {
            std::fs::remove_file(&path)?;
        }
        Ok(())
    }

    /// Check if authenticated
    pub fn is_authenticated(&self) -> bool {
        if let Some(token) = &self.access_token {
            if !token.is_empty() {
                // Check expiry
                if let Some(expires) = self.expires_at {
                    let now = chrono::Utc::now().timestamp();
                    return expires > now;
                }
                return true;
            }
        }
        false
    }
}
