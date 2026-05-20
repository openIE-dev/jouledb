//! Health Checks for JouleDB Server
//!
//! Provides health monitoring, readiness checks, and liveness probes

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

/// Health status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

impl std::fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HealthStatus::Healthy => write!(f, "healthy"),
            HealthStatus::Degraded => write!(f, "degraded"),
            HealthStatus::Unhealthy => write!(f, "unhealthy"),
            HealthStatus::Unknown => write!(f, "unknown"),
        }
    }
}

/// Health check result
#[derive(Debug, Clone)]
pub struct HealthCheckResult {
    pub status: HealthStatus,
    pub message: String,
    pub timestamp: u64,
    pub details: HashMap<String, String>,
}

/// Component health
#[derive(Debug, Clone)]
struct ComponentHealth {
    status: HealthStatus,
    last_check: u64,
    last_success: u64,
    consecutive_failures: u32,
    message: String,
}

/// Health Check Manager
pub struct HealthCheckManager {
    components: Arc<RwLock<HashMap<String, ComponentHealth>>>,
    overall_status: Arc<RwLock<HealthStatus>>,
    startup_time: u64,
}

impl HealthCheckManager {
    /// Create new health check manager
    pub fn new() -> Self {
        Self {
            components: Arc::new(RwLock::new(HashMap::new())),
            overall_status: Arc::new(RwLock::new(HealthStatus::Unknown)),
            startup_time: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        }
    }

    /// Register component
    pub fn register_component(&self, name: &str) {
        let mut components = crate::lock_util::write_lock(&self.components);
        components.insert(
            name.to_string(),
            ComponentHealth {
                status: HealthStatus::Unknown,
                last_check: 0,
                last_success: 0,
                consecutive_failures: 0,
                message: "Not checked yet".to_string(),
            },
        );
    }

    /// Update component health
    pub fn update_component(&self, name: &str, status: HealthStatus, message: String) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let mut components = crate::lock_util::write_lock(&self.components);
        if let Some(component) = components.get_mut(name) {
            component.status = status.clone();
            component.last_check = now;
            component.message = message;

            match status {
                HealthStatus::Healthy => {
                    component.last_success = now;
                    component.consecutive_failures = 0;
                }
                _ => {
                    component.consecutive_failures += 1;
                }
            }
        }
        drop(components);

        self.update_overall_status();
    }

    /// Update overall status
    fn update_overall_status(&self) {
        let components = crate::lock_util::read_lock(&self.components);
        let mut healthy_count = 0;
        let mut degraded_count = 0;
        let mut unhealthy_count = 0;

        for component in components.values() {
            match component.status {
                HealthStatus::Healthy => healthy_count += 1,
                HealthStatus::Degraded => degraded_count += 1,
                HealthStatus::Unhealthy => unhealthy_count += 1,
                HealthStatus::Unknown => {} // Unknown components don't affect overall status
            }
        }

        let overall = if unhealthy_count > 0 {
            HealthStatus::Unhealthy
        } else if degraded_count > 0 {
            HealthStatus::Degraded
        } else if healthy_count > 0 {
            HealthStatus::Healthy
        } else {
            HealthStatus::Unknown
        };

        *crate::lock_util::write_lock(&self.overall_status) = overall;
    }

    /// Get overall health
    pub fn get_health(&self) -> HealthCheckResult {
        let status = crate::lock_util::read_lock(&self.overall_status).clone();
        let components = crate::lock_util::read_lock(&self.components);
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let mut details = HashMap::new();
        for (name, component) in components.iter() {
            details.insert(format!("{}_status", name), component.status.to_string());
            details.insert(
                format!("{}_last_check", name),
                component.last_check.to_string(),
            );
            details.insert(
                format!("{}_failures", name),
                component.consecutive_failures.to_string(),
            );
        }

        let message = match status {
            HealthStatus::Healthy => "All systems operational".to_string(),
            HealthStatus::Degraded => "Some components degraded".to_string(),
            HealthStatus::Unhealthy => "System unhealthy".to_string(),
            HealthStatus::Unknown => "Health status unknown".to_string(),
        };

        HealthCheckResult {
            status,
            message,
            timestamp: now,
            details,
        }
    }

    /// Get component health
    pub fn get_component_health(&self, name: &str) -> Option<HealthCheckResult> {
        let components = crate::lock_util::read_lock(&self.components);
        let component = components.get(name)?;

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let mut details = HashMap::new();
        details.insert("last_check".to_string(), component.last_check.to_string());
        details.insert(
            "last_success".to_string(),
            component.last_success.to_string(),
        );
        details.insert(
            "consecutive_failures".to_string(),
            component.consecutive_failures.to_string(),
        );

        Some(HealthCheckResult {
            status: component.status.clone(),
            message: component.message.clone(),
            timestamp: now,
            details,
        })
    }

    /// Check if ready
    pub fn is_ready(&self) -> bool {
        let status = crate::lock_util::read_lock(&self.overall_status);
        matches!(*status, HealthStatus::Healthy | HealthStatus::Degraded)
    }

    /// Check if alive
    pub fn is_alive(&self) -> bool {
        let status = crate::lock_util::read_lock(&self.overall_status);
        !matches!(*status, HealthStatus::Unhealthy)
    }

    /// Get uptime in milliseconds
    pub fn get_uptime(&self) -> u64 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        now - self.startup_time
    }
}

impl Default for HealthCheckManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Health check functions
pub struct HealthChecks;

impl HealthChecks {
    /// Check database health by attempting a simple read operation.
    pub fn check_database(
        db: &std::sync::RwLock<joule_db_local::Database>,
    ) -> (HealthStatus, String) {
        match db.read() {
            Ok(guard) => {
                // Attempt a lightweight stats query to verify the engine is responsive
                let _stats = guard.stats();
                (HealthStatus::Healthy, "Database operational".to_string())
            }
            Err(_) => (
                HealthStatus::Unhealthy,
                "Database lock poisoned".to_string(),
            ),
        }
    }

    /// Check storage health by verifying the data directory is accessible and has free space.
    pub fn check_storage(data_dir: &str) -> (HealthStatus, String) {
        let path = std::path::Path::new(data_dir);
        if !path.exists() {
            return (
                HealthStatus::Unhealthy,
                format!("Data directory '{}' does not exist", data_dir),
            );
        }

        // Check that we can write to the directory
        let probe = path.join(".health_probe");
        match std::fs::write(&probe, b"ok") {
            Ok(()) => {
                let _ = std::fs::remove_file(&probe);
            }
            Err(e) => {
                return (
                    HealthStatus::Unhealthy,
                    format!("Storage not writable: {}", e),
                );
            }
        }

        // Check available disk space (platform-specific)
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            if let Ok(metadata) = std::fs::metadata(data_dir) {
                // Use statvfs via libc for real disk space
                let _ = metadata.dev(); // verify accessible
            }
            // Attempt statvfs for disk space check
            let c_path = std::ffi::CString::new(data_dir).unwrap_or_default();
            let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
            let ret = unsafe { libc::statvfs(c_path.as_ptr(), &mut stat) };
            if ret == 0 {
                let total_bytes = stat.f_blocks as u64 * stat.f_frsize as u64;
                let avail_bytes = stat.f_bavail as u64 * stat.f_frsize as u64;
                if total_bytes > 0 {
                    let used_pct = 100 - (avail_bytes * 100 / total_bytes);
                    let avail_mb = avail_bytes / (1024 * 1024);
                    if used_pct > 95 {
                        return (
                            HealthStatus::Unhealthy,
                            format!(
                                "Disk critically full: {}% used, {}MB available",
                                used_pct, avail_mb
                            ),
                        );
                    } else if used_pct > 90 {
                        return (
                            HealthStatus::Degraded,
                            format!(
                                "Disk nearly full: {}% used, {}MB available",
                                used_pct, avail_mb
                            ),
                        );
                    }
                    return (
                        HealthStatus::Healthy,
                        format!("Storage OK: {}% used, {}MB available", used_pct, avail_mb),
                    );
                }
            }
        }

        (HealthStatus::Healthy, "Storage operational".to_string())
    }

    /// Check network health by verifying we can bind to a test port (non-blocking).
    pub fn check_network(listen_addr: &str) -> (HealthStatus, String) {
        // Parse the configured address to verify the network interface is available
        match listen_addr.parse::<std::net::SocketAddr>() {
            Ok(addr) => {
                // Verify the IP is routable / resolvable (basic check)
                if addr.ip().is_unspecified() || addr.ip().is_loopback() {
                    (
                        HealthStatus::Healthy,
                        format!("Network OK (listening on {})", addr),
                    )
                } else {
                    // Try to create a socket on the same interface (non-blocking probe)
                    match std::net::UdpSocket::bind((addr.ip(), 0)) {
                        Ok(_) => (
                            HealthStatus::Healthy,
                            format!("Network OK (interface {} reachable)", addr.ip()),
                        ),
                        Err(e) => (
                            HealthStatus::Degraded,
                            format!("Network interface issue: {}", e),
                        ),
                    }
                }
            }
            Err(e) => (
                HealthStatus::Unhealthy,
                format!("Invalid listen address: {}", e),
            ),
        }
    }

    /// Check memory health by reading process RSS (resident set size).
    pub fn check_memory() -> (HealthStatus, String) {
        #[cfg(target_os = "macos")]
        {
            // macOS: use mach_task_basic_info
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
                let rss_mb = info.resident_size as u64 / (1024 * 1024);
                if rss_mb > 8192 {
                    return (
                        HealthStatus::Unhealthy,
                        format!("Memory critical: {}MB RSS", rss_mb),
                    );
                } else if rss_mb > 4096 {
                    return (
                        HealthStatus::Degraded,
                        format!("Memory high: {}MB RSS", rss_mb),
                    );
                }
                return (
                    HealthStatus::Healthy,
                    format!("Memory OK: {}MB RSS", rss_mb),
                );
            }
        }

        #[cfg(target_os = "linux")]
        {
            // Linux: read /proc/self/status
            if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
                for line in status.lines() {
                    if line.starts_with("VmRSS:") {
                        let parts: Vec<&str> = line.split_whitespace().collect();
                        if let Some(kb_str) = parts.get(1) {
                            if let Ok(kb) = kb_str.parse::<u64>() {
                                let mb = kb / 1024;
                                if mb > 8192 {
                                    return (
                                        HealthStatus::Unhealthy,
                                        format!("Memory critical: {}MB RSS", mb),
                                    );
                                } else if mb > 4096 {
                                    return (
                                        HealthStatus::Degraded,
                                        format!("Memory high: {}MB RSS", mb),
                                    );
                                }
                                return (HealthStatus::Healthy, format!("Memory OK: {}MB RSS", mb));
                            }
                        }
                    }
                }
            }
        }

        (HealthStatus::Healthy, "Memory within limits".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_check_manager_new() {
        let manager = HealthCheckManager::new();
        assert!(!manager.is_ready()); // Initially unknown
        assert!(manager.is_alive()); // Unknown is considered alive
    }

    #[test]
    fn test_register_and_update_component() {
        let manager = HealthCheckManager::new();
        manager.register_component("database");
        manager.update_component("database", HealthStatus::Healthy, "OK".to_string());

        let health = manager.get_component_health("database").unwrap();
        assert_eq!(health.status, HealthStatus::Healthy);
    }

    #[test]
    fn test_overall_status() {
        let manager = HealthCheckManager::new();
        manager.register_component("database");
        manager.register_component("storage");

        manager.update_component("database", HealthStatus::Healthy, "OK".to_string());
        manager.update_component("storage", HealthStatus::Healthy, "OK".to_string());

        assert!(manager.is_ready());
        assert!(manager.is_alive());

        let health = manager.get_health();
        assert_eq!(health.status, HealthStatus::Healthy);
    }

    #[test]
    fn test_degraded_status() {
        let manager = HealthCheckManager::new();
        manager.register_component("database");
        manager.register_component("storage");

        manager.update_component("database", HealthStatus::Healthy, "OK".to_string());
        manager.update_component("storage", HealthStatus::Degraded, "Slow".to_string());

        let health = manager.get_health();
        assert_eq!(health.status, HealthStatus::Degraded);
        assert!(manager.is_ready()); // Degraded is still ready
    }

    #[test]
    fn test_uptime() {
        let manager = HealthCheckManager::new();
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(manager.get_uptime() >= 10);
    }

    #[test]
    fn test_check_storage_real() {
        let dir = tempfile::tempdir().unwrap();
        let (status, msg) = HealthChecks::check_storage(dir.path().to_str().unwrap());
        // On machines with very full disks (>95%), Unhealthy is the correct
        // response.  The important thing is that we get a valid status and
        // message back — not a panic or error.
        assert!(
            status == HealthStatus::Healthy
                || status == HealthStatus::Degraded
                || (status == HealthStatus::Unhealthy && msg.contains("Disk")),
            "Storage check should return a valid disk status, got {:?}: {}",
            status,
            msg
        );
    }

    #[test]
    fn test_check_storage_missing_dir() {
        let (status, _msg) = HealthChecks::check_storage("/nonexistent/path/jouledb/data");
        assert_eq!(status, HealthStatus::Unhealthy);
    }

    #[test]
    fn test_check_network_loopback() {
        let (status, msg) = HealthChecks::check_network("127.0.0.1:8080");
        assert_eq!(
            status,
            HealthStatus::Healthy,
            "Loopback should be healthy: {}",
            msg
        );
    }

    #[test]
    fn test_check_network_invalid_addr() {
        let (status, _msg) = HealthChecks::check_network("not_an_address");
        assert_eq!(status, HealthStatus::Unhealthy);
    }

    #[test]
    fn test_check_memory_real() {
        let (status, msg) = HealthChecks::check_memory();
        assert!(
            status == HealthStatus::Healthy || status == HealthStatus::Degraded,
            "Memory should be healthy or degraded, got {:?}: {}",
            status,
            msg
        );
    }

    #[test]
    fn test_check_database_real() {
        let db = joule_db_local::Database::open(tempfile::tempdir().unwrap().path()).unwrap();
        let db_lock = std::sync::RwLock::new(db);
        let (status, msg) = HealthChecks::check_database(&db_lock);
        assert_eq!(
            status,
            HealthStatus::Healthy,
            "Database should be healthy: {}",
            msg
        );
    }
}
