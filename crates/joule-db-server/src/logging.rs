//! Logging System for JouleDB Server
//!
//! Provides structured logging, log rotation, and log levels

use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

/// Log level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum LogLevel {
    Trace = 0,
    Debug = 1,
    Info = 2,
    Warn = 3,
    Error = 4,
}

impl From<&str> for LogLevel {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "trace" => LogLevel::Trace,
            "debug" => LogLevel::Debug,
            "info" => LogLevel::Info,
            "warn" => LogLevel::Warn,
            "error" => LogLevel::Error,
            _ => LogLevel::Info,
        }
    }
}

impl std::fmt::Display for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LogLevel::Trace => write!(f, "TRACE"),
            LogLevel::Debug => write!(f, "DEBUG"),
            LogLevel::Info => write!(f, "INFO"),
            LogLevel::Warn => write!(f, "WARN"),
            LogLevel::Error => write!(f, "ERROR"),
        }
    }
}

/// Log entry
#[derive(Debug, Clone, Serialize)]
pub struct LogEntry {
    pub timestamp: u64,
    pub level: String,
    pub target: String,
    pub message: String,
    pub fields: serde_json::Value,
}

/// Logger configuration
#[derive(Debug, Clone)]
pub struct LoggerConfig {
    pub level: LogLevel,
    pub file_path: Option<PathBuf>,
    pub console_enabled: bool,
    pub json_format: bool,
    pub max_file_size: u64,
    pub max_files: u32,
}

impl Default for LoggerConfig {
    fn default() -> Self {
        Self {
            level: LogLevel::Info,
            file_path: None,
            console_enabled: true,
            json_format: false,
            max_file_size: 10 * 1024 * 1024, // 10MB
            max_files: 10,
        }
    }
}

/// Logger
pub struct Logger {
    config: LoggerConfig,
    file: Option<Arc<RwLock<File>>>,
    current_file_size: Arc<RwLock<u64>>,
}

impl Logger {
    /// Create new logger
    pub fn new(config: LoggerConfig) -> Result<Self, String> {
        let file = if let Some(ref path) = config.file_path {
            let file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .map_err(|e| format!("Failed to open log file: {}", e))?;

            Some(Arc::new(RwLock::new(file)))
        } else {
            None
        };

        let current_size = if let Some(ref path) = config.file_path {
            if path.exists() {
                std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
            } else {
                0
            }
        } else {
            0
        };

        Ok(Self {
            config,
            file,
            current_file_size: Arc::new(RwLock::new(current_size)),
        })
    }

    /// Log message
    pub fn log(&self, level: LogLevel, target: &str, message: &str, fields: serde_json::Value) {
        if level < self.config.level {
            return;
        }

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let entry = LogEntry {
            timestamp,
            level: level.to_string(),
            target: target.to_string(),
            message: message.to_string(),
            fields,
        };

        // Console output
        if self.config.console_enabled {
            if self.config.json_format {
                if let Ok(json) = serde_json::to_string(&entry) {
                    eprintln!("{}", json);
                }
            } else {
                eprintln!("[{}] {} [{}] {}", timestamp, level, target, message);
            }
        }

        // File output
        if let Some(ref file) = self.file {
            let log_line = if self.config.json_format {
                format!("{}\n", serde_json::to_string(&entry).unwrap_or_default())
            } else {
                format!("[{}] {} [{}] {}\n", timestamp, level, target, message)
            };

            if let Ok(mut file) = file.write() {
                if file.write_all(log_line.as_bytes()).is_ok() {
                    // Check if rotation needed
                    let mut size = crate::lock_util::write_lock(&self.current_file_size);
                    *size += log_line.len() as u64;

                    if *size >= self.config.max_file_size {
                        drop(file);
                        drop(size);
                        let _ = self.rotate_log();
                    }
                }
            }
        }
    }

    /// Rotate log file
    fn rotate_log(&self) -> Result<(), String> {
        if let Some(ref path) = self.config.file_path {
            // Rename existing files
            for i in (1..self.config.max_files).rev() {
                let old_path = path.with_extension(format!("log.{}", i));
                let new_path = path.with_extension(format!("log.{}", i + 1));

                if old_path.exists() {
                    std::fs::rename(&old_path, &new_path)
                        .map_err(|e| format!("Failed to rotate log file: {}", e))?;
                }
            }

            // Rename current file
            let rotated_path = path.with_extension("log.1");
            if path.exists() {
                std::fs::rename(path, &rotated_path)
                    .map_err(|e| format!("Failed to rename log file: {}", e))?;
            }

            // Create new file
            let new_file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .map_err(|e| format!("Failed to create new log file: {}", e))?;

            *crate::lock_util::write_lock(&self.current_file_size) = 0;

            if let Some(ref file) = self.file {
                *crate::lock_util::write_lock(&file) = new_file;
            }
        }

        Ok(())
    }

    /// Trace log
    pub fn trace(&self, target: &str, message: &str) {
        self.log(LogLevel::Trace, target, message, serde_json::json!({}));
    }

    /// Debug log
    pub fn debug(&self, target: &str, message: &str) {
        self.log(LogLevel::Debug, target, message, serde_json::json!({}));
    }

    /// Info log
    pub fn info(&self, target: &str, message: &str) {
        self.log(LogLevel::Info, target, message, serde_json::json!({}));
    }

    /// Warn log
    pub fn warn(&self, target: &str, message: &str) {
        self.log(LogLevel::Warn, target, message, serde_json::json!({}));
    }

    /// Error log
    pub fn error(&self, target: &str, message: &str) {
        self.log(LogLevel::Error, target, message, serde_json::json!({}));
    }

    /// Log with fields
    pub fn log_with_fields(
        &self,
        level: LogLevel,
        target: &str,
        message: &str,
        fields: serde_json::Value,
    ) {
        self.log(level, target, message, fields);
    }
}

/// Global logger singleton
static GLOBAL_LOGGER: std::sync::OnceLock<Arc<Logger>> = std::sync::OnceLock::new();

/// Initialize global logger
pub fn init_global_logger(config: LoggerConfig) -> Result<(), String> {
    let logger = Logger::new(config)?;
    GLOBAL_LOGGER
        .set(Arc::new(logger))
        .map_err(|_| "Logger already initialized".to_string())
}

/// Get global logger
pub fn get_logger() -> Option<&'static Arc<Logger>> {
    GLOBAL_LOGGER.get()
}

/// Log macros helpers
pub fn trace(target: &str, message: &str) {
    if let Some(logger) = get_logger() {
        logger.trace(target, message);
    }
}

pub fn debug(target: &str, message: &str) {
    if let Some(logger) = get_logger() {
        logger.debug(target, message);
    }
}

pub fn info(target: &str, message: &str) {
    if let Some(logger) = get_logger() {
        logger.info(target, message);
    }
}

pub fn warn(target: &str, message: &str) {
    if let Some(logger) = get_logger() {
        logger.warn(target, message);
    }
}

pub fn error(target: &str, message: &str) {
    if let Some(logger) = get_logger() {
        logger.error(target, message);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_level_from_str() {
        assert_eq!(LogLevel::from("trace"), LogLevel::Trace);
        assert_eq!(LogLevel::from("DEBUG"), LogLevel::Debug);
        assert_eq!(LogLevel::from("Info"), LogLevel::Info);
        assert_eq!(LogLevel::from("WARN"), LogLevel::Warn);
        assert_eq!(LogLevel::from("error"), LogLevel::Error);
        assert_eq!(LogLevel::from("invalid"), LogLevel::Info);
    }

    #[test]
    fn test_log_level_display() {
        assert_eq!(format!("{}", LogLevel::Trace), "TRACE");
        assert_eq!(format!("{}", LogLevel::Error), "ERROR");
    }

    #[test]
    fn test_logger_creation() {
        let config = LoggerConfig::default();
        let logger = Logger::new(config).unwrap();
        // Just verify creation works
        logger.info("test", "Test message");
    }

    #[test]
    fn test_log_level_ordering() {
        assert!(LogLevel::Trace < LogLevel::Debug);
        assert!(LogLevel::Debug < LogLevel::Info);
        assert!(LogLevel::Info < LogLevel::Warn);
        assert!(LogLevel::Warn < LogLevel::Error);
    }
}
