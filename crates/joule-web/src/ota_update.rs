//! Over-the-air update management: firmware version model, update packages
//! (version, size, checksum), rollout management (percentage-based),
//! device eligibility, update status tracking, and rollback capability.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ── Types ──

/// Firmware version (semver-style).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Version {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl Version {
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self { major, minor, patch }
    }

    pub fn is_newer_than(&self, other: &Version) -> bool {
        (self.major, self.minor, self.patch) > (other.major, other.minor, other.patch)
    }
}

impl PartialOrd for Version {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Version {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.major, self.minor, self.patch).cmp(&(other.major, other.minor, other.patch))
    }
}

impl std::fmt::Display for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// An update package containing firmware binary metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdatePackage {
    pub id: String,
    pub version: Version,
    pub size_bytes: u64,
    pub checksum: String,
    pub release_notes: String,
    pub min_version: Option<Version>,
    pub created_at: DateTime<Utc>,
    pub is_critical: bool,
}

impl UpdatePackage {
    pub fn new(version: Version, size_bytes: u64, checksum: &str) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            version,
            size_bytes,
            checksum: checksum.to_string(),
            release_notes: String::new(),
            min_version: None,
            created_at: Utc::now(),
            is_critical: false,
        }
    }

    pub fn with_release_notes(mut self, notes: &str) -> Self {
        self.release_notes = notes.to_string();
        self
    }

    pub fn with_min_version(mut self, min: Version) -> Self {
        self.min_version = Some(min);
        self
    }

    pub fn with_critical(mut self, critical: bool) -> Self {
        self.is_critical = critical;
        self
    }
}

/// Status of an update for a specific device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UpdateStatus {
    Pending,
    Downloading,
    Downloaded,
    Installing,
    Installed,
    Failed,
    RolledBack,
    Skipped,
}

impl UpdateStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Downloading => "downloading",
            Self::Downloaded => "downloaded",
            Self::Installing => "installing",
            Self::Installed => "installed",
            Self::Failed => "failed",
            Self::RolledBack => "rolled_back",
            Self::Skipped => "skipped",
        }
    }

    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Installed | Self::Failed | Self::RolledBack | Self::Skipped)
    }
}

/// Per-device update tracking.
#[derive(Debug, Clone)]
pub struct DeviceUpdate {
    pub device_id: String,
    pub package_id: String,
    pub status: UpdateStatus,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub error_message: Option<String>,
    pub previous_version: Version,
    pub attempts: u32,
}

/// Rollout configuration.
#[derive(Debug, Clone)]
pub struct RolloutConfig {
    pub id: String,
    pub package_id: String,
    /// Percentage of eligible devices (0..=100).
    pub percentage: u32,
    pub paused: bool,
    pub created_at: DateTime<Utc>,
    /// Max concurrent updates.
    pub max_concurrent: usize,
    /// Max allowed failures before auto-pause.
    pub max_failures: u32,
}

impl RolloutConfig {
    pub fn new(package_id: &str, percentage: u32) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            package_id: package_id.to_string(),
            percentage: percentage.min(100),
            paused: false,
            created_at: Utc::now(),
            max_concurrent: 10,
            max_failures: 5,
        }
    }

    pub fn with_max_concurrent(mut self, max: usize) -> Self {
        self.max_concurrent = max;
        self
    }

    pub fn with_max_failures(mut self, max: u32) -> Self {
        self.max_failures = max;
        self
    }
}

/// A registered device in the OTA system.
#[derive(Debug, Clone)]
pub struct OtaDevice {
    pub id: String,
    pub current_version: Version,
    pub device_type: String,
    pub tags: Vec<String>,
    pub update_history: Vec<String>, // package ids
}

impl OtaDevice {
    pub fn new(id: &str, current_version: Version, device_type: &str) -> Self {
        Self {
            id: id.to_string(),
            current_version,
            device_type: device_type.to_string(),
            tags: Vec::new(),
            update_history: Vec::new(),
        }
    }

    pub fn with_tag(mut self, tag: &str) -> Self {
        self.tags.push(tag.to_string());
        self
    }
}

// ── OTA Manager ──

/// Over-the-air update manager.
pub struct OtaManager {
    packages: HashMap<String, UpdatePackage>,
    devices: HashMap<String, OtaDevice>,
    device_updates: Vec<DeviceUpdate>,
    rollouts: HashMap<String, RolloutConfig>,
    max_retries: u32,
}

impl OtaManager {
    pub fn new() -> Self {
        Self {
            packages: HashMap::new(),
            devices: HashMap::new(),
            device_updates: Vec::new(),
            rollouts: HashMap::new(),
            max_retries: 3,
        }
    }

    pub fn set_max_retries(&mut self, max: u32) {
        self.max_retries = max;
    }

    /// Register an update package.
    pub fn add_package(&mut self, pkg: UpdatePackage) -> String {
        let id = pkg.id.clone();
        self.packages.insert(id.clone(), pkg);
        id
    }

    /// Register a device.
    pub fn register_device(&mut self, device: OtaDevice) {
        self.devices.insert(device.id.clone(), device);
    }

    /// Check if a device is eligible for a specific package.
    pub fn is_eligible(&self, device_id: &str, package_id: &str) -> bool {
        let device = match self.devices.get(device_id) {
            Some(d) => d,
            None => return false,
        };
        let pkg = match self.packages.get(package_id) {
            Some(p) => p,
            None => return false,
        };

        // Must be newer than current version.
        if !pkg.version.is_newer_than(&device.current_version) {
            return false;
        }

        // Must meet minimum version requirement.
        if let Some(min_ver) = &pkg.min_version {
            if device.current_version < *min_ver {
                return false;
            }
        }

        // Not already installed or in-progress.
        let active = self.device_updates.iter().any(|du| {
            du.device_id == device_id && du.package_id == package_id && !du.status.is_terminal()
        });
        if active {
            return false;
        }

        // Not already successfully installed.
        let installed = self.device_updates.iter().any(|du| {
            du.device_id == device_id && du.package_id == package_id && du.status == UpdateStatus::Installed
        });
        !installed
    }

    /// Get all eligible devices for a package.
    pub fn eligible_devices(&self, package_id: &str) -> Vec<String> {
        let mut result: Vec<String> = self.devices.keys()
            .filter(|id| self.is_eligible(id, package_id))
            .cloned()
            .collect();
        result.sort();
        result
    }

    /// Create a rollout for a package.
    pub fn create_rollout(&mut self, config: RolloutConfig) -> String {
        let id = config.id.clone();
        self.rollouts.insert(id.clone(), config);
        id
    }

    /// Compute which devices should receive an update in this rollout wave.
    pub fn rollout_wave(&self, rollout_id: &str) -> Vec<String> {
        let rollout = match self.rollouts.get(rollout_id) {
            Some(r) => r,
            None => return Vec::new(),
        };

        if rollout.paused {
            return Vec::new();
        }

        // Check failure count.
        let failure_count = self.device_updates.iter()
            .filter(|du| du.package_id == rollout.package_id && du.status == UpdateStatus::Failed)
            .count() as u32;
        if failure_count >= rollout.max_failures {
            return Vec::new();
        }

        let eligible = self.eligible_devices(&rollout.package_id);
        let count = (eligible.len() as f64 * rollout.percentage as f64 / 100.0).ceil() as usize;
        let count = count.min(rollout.max_concurrent);

        // Exclude devices with active updates.
        let active_count = self.device_updates.iter()
            .filter(|du| du.package_id == rollout.package_id && !du.status.is_terminal())
            .count();
        let remaining = rollout.max_concurrent.saturating_sub(active_count);
        let count = count.min(remaining);

        eligible.into_iter().take(count).collect()
    }

    /// Start an update for a device.
    pub fn start_update(&mut self, device_id: &str, package_id: &str) -> Result<(), String> {
        if !self.is_eligible(device_id, package_id) {
            return Err("device not eligible for this update".to_string());
        }

        let device_version = self.devices.get(device_id).unwrap().current_version.clone();

        let du = DeviceUpdate {
            device_id: device_id.to_string(),
            package_id: package_id.to_string(),
            status: UpdateStatus::Pending,
            started_at: Some(Utc::now()),
            completed_at: None,
            error_message: None,
            previous_version: device_version,
            attempts: 1,
        };

        self.device_updates.push(du);
        Ok(())
    }

    /// Update the status of a device's update.
    pub fn update_status(&mut self, device_id: &str, package_id: &str, status: UpdateStatus) -> bool {
        // Find the most recent (last) active update for this device+package.
        let found = self.device_updates.iter_mut().rev().find(|du| {
            du.device_id == device_id && du.package_id == package_id && !du.status.is_terminal()
        });

        if let Some(du) = found {
            du.status = status;
            if status.is_terminal() {
                du.completed_at = Some(Utc::now());
            }
            if status == UpdateStatus::Installed {
                // Update device version.
                if let Some(pkg) = self.packages.get(package_id) {
                    let new_ver = pkg.version.clone();
                    let pkg_id = package_id.to_string();
                    if let Some(device) = self.devices.get_mut(device_id) {
                        device.current_version = new_ver;
                        device.update_history.push(pkg_id);
                    }
                }
            }
            true
        } else {
            false
        }
    }

    /// Mark an update as failed with an error message.
    pub fn fail_update(&mut self, device_id: &str, package_id: &str, error: &str) -> bool {
        let found = self.device_updates.iter_mut().rev().find(|du| {
            du.device_id == device_id && du.package_id == package_id && !du.status.is_terminal()
        });

        if let Some(du) = found {
            du.status = UpdateStatus::Failed;
            du.error_message = Some(error.to_string());
            du.completed_at = Some(Utc::now());
            true
        } else {
            false
        }
    }

    /// Rollback a device to its previous version.
    pub fn rollback(&mut self, device_id: &str, package_id: &str) -> Result<(), String> {
        let du = self.device_updates.iter_mut().rev().find(|du| {
            du.device_id == device_id && du.package_id == package_id
                && (du.status == UpdateStatus::Installed || du.status == UpdateStatus::Failed)
        });

        match du {
            Some(du) => {
                let prev = du.previous_version.clone();
                du.status = UpdateStatus::RolledBack;
                du.completed_at = Some(Utc::now());
                if let Some(device) = self.devices.get_mut(device_id) {
                    device.current_version = prev;
                }
                Ok(())
            }
            None => Err("no installed or failed update found for rollback".to_string()),
        }
    }

    /// Pause a rollout.
    pub fn pause_rollout(&mut self, rollout_id: &str) -> bool {
        if let Some(r) = self.rollouts.get_mut(rollout_id) {
            r.paused = true;
            true
        } else {
            false
        }
    }

    /// Resume a rollout.
    pub fn resume_rollout(&mut self, rollout_id: &str) -> bool {
        if let Some(r) = self.rollouts.get_mut(rollout_id) {
            r.paused = false;
            true
        } else {
            false
        }
    }

    /// Retry a failed update.
    pub fn retry_update(&mut self, device_id: &str, package_id: &str) -> Result<(), String> {
        let max_attempts = self.device_updates.iter()
            .filter(|du| du.device_id == device_id && du.package_id == package_id)
            .map(|du| du.attempts)
            .max()
            .unwrap_or(0);

        if max_attempts >= self.max_retries {
            return Err(format!("max retries ({}) exceeded", self.max_retries));
        }

        // Reset the last failed update.
        let found = self.device_updates.iter_mut().rev().find(|du| {
            du.device_id == device_id && du.package_id == package_id && du.status == UpdateStatus::Failed
        });

        if let Some(du) = found {
            du.status = UpdateStatus::Pending;
            du.error_message = None;
            du.completed_at = None;
            du.attempts += 1;
            Ok(())
        } else {
            Err("no failed update found to retry".to_string())
        }
    }

    /// Get the update status for a device and package.
    pub fn get_device_update(&self, device_id: &str, package_id: &str) -> Option<&DeviceUpdate> {
        self.device_updates.iter().rev().find(|du| {
            du.device_id == device_id && du.package_id == package_id
        })
    }

    /// Summary counts of update statuses for a package.
    pub fn update_summary(&self, package_id: &str) -> HashMap<UpdateStatus, usize> {
        let mut counts = HashMap::new();
        for du in &self.device_updates {
            if du.package_id == package_id {
                *counts.entry(du.status).or_insert(0) += 1;
            }
        }
        counts
    }

    pub fn package_count(&self) -> usize {
        self.packages.len()
    }

    pub fn device_count(&self) -> usize {
        self.devices.len()
    }
}

impl Default for OtaManager {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (OtaManager, String, String) {
        let mut mgr = OtaManager::new();
        let pkg = UpdatePackage::new(Version::new(2, 0, 0), 1024, "abc123");
        let pkg_id = mgr.add_package(pkg);
        let dev = OtaDevice::new("d1", Version::new(1, 0, 0), "sensor");
        mgr.register_device(dev);
        (mgr, pkg_id, "d1".to_string())
    }

    #[test]
    fn version_ordering() {
        let v1 = Version::new(1, 0, 0);
        let v2 = Version::new(1, 1, 0);
        let v3 = Version::new(2, 0, 0);
        assert!(v2.is_newer_than(&v1));
        assert!(v3.is_newer_than(&v2));
        assert!(!v1.is_newer_than(&v2));
    }

    #[test]
    fn version_display() {
        assert_eq!(Version::new(1, 2, 3).to_string(), "1.2.3");
    }

    #[test]
    fn add_package() {
        let mut mgr = OtaManager::new();
        let pkg = UpdatePackage::new(Version::new(1, 0, 0), 512, "hash")
            .with_release_notes("Initial release")
            .with_critical(true);
        let id = mgr.add_package(pkg);
        assert_eq!(mgr.package_count(), 1);
        assert!(!id.is_empty());
    }

    #[test]
    fn eligibility_basic() {
        let (mgr, pkg_id, dev_id) = setup();
        assert!(mgr.is_eligible(&dev_id, &pkg_id));
    }

    #[test]
    fn eligibility_already_newer() {
        let mut mgr = OtaManager::new();
        let pkg = UpdatePackage::new(Version::new(1, 0, 0), 512, "hash");
        let pkg_id = mgr.add_package(pkg);
        let dev = OtaDevice::new("d1", Version::new(2, 0, 0), "sensor");
        mgr.register_device(dev);
        assert!(!mgr.is_eligible("d1", &pkg_id));
    }

    #[test]
    fn eligibility_min_version() {
        let mut mgr = OtaManager::new();
        let pkg = UpdatePackage::new(Version::new(3, 0, 0), 512, "hash")
            .with_min_version(Version::new(2, 0, 0));
        let pkg_id = mgr.add_package(pkg);
        // Device on 1.0.0 doesn't meet min_version of 2.0.0.
        let dev = OtaDevice::new("d1", Version::new(1, 0, 0), "sensor");
        mgr.register_device(dev);
        assert!(!mgr.is_eligible("d1", &pkg_id));
    }

    #[test]
    fn start_and_complete_update() {
        let (mut mgr, pkg_id, dev_id) = setup();
        mgr.start_update(&dev_id, &pkg_id).unwrap();
        mgr.update_status(&dev_id, &pkg_id, UpdateStatus::Downloading);
        mgr.update_status(&dev_id, &pkg_id, UpdateStatus::Installing);
        mgr.update_status(&dev_id, &pkg_id, UpdateStatus::Installed);
        let du = mgr.get_device_update(&dev_id, &pkg_id).unwrap();
        assert_eq!(du.status, UpdateStatus::Installed);
        // Device version should be updated.
        assert_eq!(mgr.devices.get("d1").unwrap().current_version, Version::new(2, 0, 0));
    }

    #[test]
    fn fail_and_rollback() {
        let (mut mgr, pkg_id, dev_id) = setup();
        mgr.start_update(&dev_id, &pkg_id).unwrap();
        mgr.update_status(&dev_id, &pkg_id, UpdateStatus::Installed);
        // Now rollback.
        mgr.rollback(&dev_id, &pkg_id).unwrap();
        let du = mgr.get_device_update(&dev_id, &pkg_id).unwrap();
        assert_eq!(du.status, UpdateStatus::RolledBack);
        assert_eq!(mgr.devices.get("d1").unwrap().current_version, Version::new(1, 0, 0));
    }

    #[test]
    fn retry_failed_update() {
        let (mut mgr, pkg_id, dev_id) = setup();
        mgr.start_update(&dev_id, &pkg_id).unwrap();
        mgr.fail_update(&dev_id, &pkg_id, "timeout");
        mgr.retry_update(&dev_id, &pkg_id).unwrap();
        let du = mgr.get_device_update(&dev_id, &pkg_id).unwrap();
        assert_eq!(du.status, UpdateStatus::Pending);
        assert_eq!(du.attempts, 2);
    }

    #[test]
    fn retry_max_exceeded() {
        let (mut mgr, pkg_id, dev_id) = setup();
        mgr.set_max_retries(2);
        mgr.start_update(&dev_id, &pkg_id).unwrap();
        mgr.fail_update(&dev_id, &pkg_id, "err1");
        mgr.retry_update(&dev_id, &pkg_id).unwrap();
        mgr.fail_update(&dev_id, &pkg_id, "err2");
        // Now at max retries.
        assert!(mgr.retry_update(&dev_id, &pkg_id).is_err());
    }

    #[test]
    fn rollout_creation() {
        let (mut mgr, pkg_id, _) = setup();
        let rc = RolloutConfig::new(&pkg_id, 50).with_max_concurrent(5);
        let rid = mgr.create_rollout(rc);
        assert!(!rid.is_empty());
    }

    #[test]
    fn rollout_wave_basic() {
        let (mut mgr, pkg_id, _) = setup();
        // Add more devices.
        for i in 2..=10 {
            let dev = OtaDevice::new(&format!("d{}", i), Version::new(1, 0, 0), "sensor");
            mgr.register_device(dev);
        }
        let rc = RolloutConfig::new(&pkg_id, 50);
        let rid = mgr.create_rollout(rc);
        let wave = mgr.rollout_wave(&rid);
        assert!(!wave.is_empty());
        assert!(wave.len() <= 10); // max_concurrent default
    }

    #[test]
    fn rollout_paused() {
        let (mut mgr, pkg_id, _) = setup();
        let rc = RolloutConfig::new(&pkg_id, 100);
        let rid = mgr.create_rollout(rc);
        mgr.pause_rollout(&rid);
        let wave = mgr.rollout_wave(&rid);
        assert!(wave.is_empty());
    }

    #[test]
    fn rollout_resume() {
        let (mut mgr, pkg_id, _) = setup();
        let rc = RolloutConfig::new(&pkg_id, 100);
        let rid = mgr.create_rollout(rc);
        mgr.pause_rollout(&rid);
        mgr.resume_rollout(&rid);
        let wave = mgr.rollout_wave(&rid);
        assert!(!wave.is_empty());
    }

    #[test]
    fn update_summary_counts() {
        let (mut mgr, pkg_id, dev_id) = setup();
        mgr.start_update(&dev_id, &pkg_id).unwrap();
        mgr.fail_update(&dev_id, &pkg_id, "err");
        let summary = mgr.update_summary(&pkg_id);
        assert_eq!(summary.get(&UpdateStatus::Failed), Some(&1));
    }

    #[test]
    fn eligible_devices_list() {
        let (mgr, pkg_id, _) = setup();
        let eligible = mgr.eligible_devices(&pkg_id);
        assert_eq!(eligible, vec!["d1".to_string()]);
    }

    #[test]
    fn not_eligible_after_install() {
        let (mut mgr, pkg_id, dev_id) = setup();
        mgr.start_update(&dev_id, &pkg_id).unwrap();
        mgr.update_status(&dev_id, &pkg_id, UpdateStatus::Installed);
        assert!(!mgr.is_eligible(&dev_id, &pkg_id));
    }

    #[test]
    fn update_status_terminal() {
        assert!(UpdateStatus::Installed.is_terminal());
        assert!(UpdateStatus::Failed.is_terminal());
        assert!(UpdateStatus::RolledBack.is_terminal());
        assert!(!UpdateStatus::Pending.is_terminal());
        assert!(!UpdateStatus::Downloading.is_terminal());
    }
}
