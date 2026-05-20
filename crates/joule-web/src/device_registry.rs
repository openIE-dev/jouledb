//! IoT device registry: registration with id/type/capabilities/firmware,
//! device state tracking (online/offline/error), heartbeat monitoring,
//! device grouping, firmware version management, and device search/filter.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ── Types ──

/// Operational state of a device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeviceState {
    Online,
    Offline,
    Error,
    Maintenance,
    Provisioning,
}

impl DeviceState {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Online => "online",
            Self::Offline => "offline",
            Self::Error => "error",
            Self::Maintenance => "maintenance",
            Self::Provisioning => "provisioning",
        }
    }
}

/// Kind/type of IoT device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeviceType {
    Sensor,
    Actuator,
    Gateway,
    Controller,
    Camera,
    Display,
    Custom,
}

impl DeviceType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Sensor => "sensor",
            Self::Actuator => "actuator",
            Self::Gateway => "gateway",
            Self::Controller => "controller",
            Self::Camera => "camera",
            Self::Display => "display",
            Self::Custom => "custom",
        }
    }
}

/// A device capability descriptor.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Capability {
    pub name: String,
    pub version: String,
}

impl Capability {
    pub fn new(name: &str, version: &str) -> Self {
        Self {
            name: name.to_string(),
            version: version.to_string(),
        }
    }
}

/// Firmware version using semantic versioning components.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FirmwareVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
    pub label: Option<String>,
}

impl FirmwareVersion {
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self { major, minor, patch, label: None }
    }

    pub fn with_label(mut self, label: &str) -> Self {
        self.label = Some(label.to_string());
        self
    }

    pub fn to_string_repr(&self) -> String {
        let base = format!("{}.{}.{}", self.major, self.minor, self.patch);
        match &self.label {
            Some(l) => format!("{}-{}", base, l),
            None => base,
        }
    }

    /// Returns true if `self` is newer than `other`.
    pub fn is_newer_than(&self, other: &FirmwareVersion) -> bool {
        (self.major, self.minor, self.patch) > (other.major, other.minor, other.patch)
    }
}

impl std::fmt::Display for FirmwareVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_string_repr())
    }
}

impl PartialOrd for FirmwareVersion {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FirmwareVersion {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.major, self.minor, self.patch).cmp(&(other.major, other.minor, other.patch))
    }
}

/// A registered IoT device.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Device {
    pub id: String,
    pub name: String,
    pub device_type: DeviceType,
    pub state: DeviceState,
    pub capabilities: Vec<Capability>,
    pub firmware: FirmwareVersion,
    pub group: Option<String>,
    pub tags: Vec<String>,
    pub registered_at: DateTime<Utc>,
    pub last_heartbeat: Option<DateTime<Utc>>,
    pub metadata: HashMap<String, String>,
    pub error_message: Option<String>,
}

impl Device {
    pub fn new(name: &str, device_type: DeviceType, firmware: FirmwareVersion) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name: name.to_string(),
            device_type,
            state: DeviceState::Provisioning,
            capabilities: Vec::new(),
            firmware,
            group: None,
            tags: Vec::new(),
            registered_at: Utc::now(),
            last_heartbeat: None,
            metadata: HashMap::new(),
            error_message: None,
        }
    }

    pub fn with_capability(mut self, cap: Capability) -> Self {
        self.capabilities.push(cap);
        self
    }

    pub fn with_group(mut self, group: &str) -> Self {
        self.group = Some(group.to_string());
        self
    }

    pub fn with_tag(mut self, tag: &str) -> Self {
        self.tags.push(tag.to_string());
        self
    }

    pub fn has_capability(&self, name: &str) -> bool {
        self.capabilities.iter().any(|c| c.name == name)
    }

    pub fn is_online(&self) -> bool {
        self.state == DeviceState::Online
    }
}

/// Search / filter criteria for devices.
#[derive(Debug, Clone, Default)]
pub struct DeviceFilter {
    pub device_type: Option<DeviceType>,
    pub state: Option<DeviceState>,
    pub group: Option<String>,
    pub tag: Option<String>,
    pub capability: Option<String>,
    pub name_contains: Option<String>,
    pub firmware_min: Option<FirmwareVersion>,
}

// ── Registry ──

/// IoT device registry with heartbeat monitoring and search.
pub struct DeviceRegistry {
    devices: HashMap<String, Device>,
    /// Maximum seconds between heartbeats before a device is considered offline.
    heartbeat_timeout_secs: i64,
    groups: HashMap<String, Vec<String>>,
}

impl DeviceRegistry {
    pub fn new(heartbeat_timeout_secs: i64) -> Self {
        Self {
            devices: HashMap::new(),
            heartbeat_timeout_secs,
            groups: HashMap::new(),
        }
    }

    /// Register a new device and return its id.
    pub fn register(&mut self, device: Device) -> String {
        let id = device.id.clone();
        if let Some(group) = device.group.clone() {
            self.groups.entry(group).or_default().push(id.clone());
        }
        self.devices.insert(id.clone(), device);
        id
    }

    /// Remove a device by id.
    pub fn unregister(&mut self, id: &str) -> Option<Device> {
        if let Some(device) = self.devices.remove(id) {
            if let Some(group) = &device.group {
                if let Some(members) = self.groups.get_mut(group) {
                    members.retain(|m| m != id);
                }
            }
            Some(device)
        } else {
            None
        }
    }

    /// Get a device by id.
    pub fn get(&self, id: &str) -> Option<&Device> {
        self.devices.get(id)
    }

    /// Get a mutable reference to a device.
    pub fn get_mut(&mut self, id: &str) -> Option<&mut Device> {
        self.devices.get_mut(id)
    }

    /// Record a heartbeat from a device.
    pub fn heartbeat(&mut self, id: &str, now: DateTime<Utc>) -> bool {
        if let Some(device) = self.devices.get_mut(id) {
            device.last_heartbeat = Some(now);
            if device.state == DeviceState::Offline || device.state == DeviceState::Provisioning {
                device.state = DeviceState::Online;
            }
            true
        } else {
            false
        }
    }

    /// Check all devices for heartbeat timeouts and mark stale ones offline.
    pub fn check_heartbeats(&mut self, now: DateTime<Utc>) -> Vec<String> {
        let timeout = self.heartbeat_timeout_secs;
        let mut timed_out = Vec::new();
        for (id, device) in &mut self.devices {
            if device.state == DeviceState::Online {
                let stale = match device.last_heartbeat {
                    Some(hb) => now.signed_duration_since(hb).num_seconds() > timeout,
                    None => true,
                };
                if stale {
                    device.state = DeviceState::Offline;
                    timed_out.push(id.clone());
                }
            }
        }
        timed_out.sort();
        timed_out
    }

    /// Set a device's state to error with a message.
    pub fn set_error(&mut self, id: &str, message: &str) -> bool {
        if let Some(device) = self.devices.get_mut(id) {
            device.state = DeviceState::Error;
            device.error_message = Some(message.to_string());
            true
        } else {
            false
        }
    }

    /// Set a device's state.
    pub fn set_state(&mut self, id: &str, state: DeviceState) -> bool {
        if let Some(device) = self.devices.get_mut(id) {
            device.state = state;
            if state != DeviceState::Error {
                device.error_message = None;
            }
            true
        } else {
            false
        }
    }

    /// Update device firmware version.
    pub fn update_firmware(&mut self, id: &str, version: FirmwareVersion) -> bool {
        if let Some(device) = self.devices.get_mut(id) {
            device.firmware = version;
            true
        } else {
            false
        }
    }

    /// Assign a device to a group.
    pub fn assign_group(&mut self, id: &str, group: &str) -> bool {
        if let Some(device) = self.devices.get_mut(id) {
            // Remove from old group.
            if let Some(old_group) = device.group.take() {
                if let Some(members) = self.groups.get_mut(&old_group) {
                    members.retain(|m| m != id);
                }
            }
            device.group = Some(group.to_string());
            self.groups.entry(group.to_string()).or_default().push(id.to_string());
            true
        } else {
            false
        }
    }

    /// Get all device ids in a group.
    pub fn group_members(&self, group: &str) -> Vec<String> {
        let mut members = self.groups.get(group).cloned().unwrap_or_default();
        members.sort();
        members
    }

    /// Search/filter devices.
    pub fn search(&self, filter: &DeviceFilter) -> Vec<&Device> {
        self.devices.values().filter(|d| {
            if let Some(dt) = filter.device_type {
                if d.device_type != dt { return false; }
            }
            if let Some(s) = filter.state {
                if d.state != s { return false; }
            }
            if let Some(g) = &filter.group {
                if d.group.as_deref() != Some(g.as_str()) { return false; }
            }
            if let Some(tag) = &filter.tag {
                if !d.tags.contains(tag) { return false; }
            }
            if let Some(cap) = &filter.capability {
                if !d.has_capability(cap) { return false; }
            }
            if let Some(name) = &filter.name_contains {
                if !d.name.to_lowercase().contains(&name.to_lowercase()) { return false; }
            }
            if let Some(min_fw) = &filter.firmware_min {
                if d.firmware < *min_fw { return false; }
            }
            true
        }).collect()
    }

    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    /// Get firmware version distribution across all devices.
    pub fn firmware_distribution(&self) -> HashMap<String, usize> {
        let mut dist = HashMap::new();
        for device in self.devices.values() {
            *dist.entry(device.firmware.to_string_repr()).or_insert(0) += 1;
        }
        dist
    }

    /// Count devices by state.
    pub fn state_counts(&self) -> HashMap<DeviceState, usize> {
        let mut counts = HashMap::new();
        for device in self.devices.values() {
            *counts.entry(device.state).or_insert(0) += 1;
        }
        counts
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    fn make_device(name: &str) -> Device {
        Device::new(name, DeviceType::Sensor, FirmwareVersion::new(1, 0, 0))
    }

    #[test]
    fn register_and_get() {
        let mut reg = DeviceRegistry::new(60);
        let d = make_device("temp-1");
        let id = reg.register(d);
        assert!(reg.get(&id).is_some());
        assert_eq!(reg.device_count(), 1);
    }

    #[test]
    fn unregister() {
        let mut reg = DeviceRegistry::new(60);
        let d = make_device("temp-1");
        let id = reg.register(d);
        let removed = reg.unregister(&id);
        assert!(removed.is_some());
        assert_eq!(reg.device_count(), 0);
    }

    #[test]
    fn heartbeat_brings_online() {
        let mut reg = DeviceRegistry::new(60);
        let d = make_device("s1");
        let id = reg.register(d);
        assert_eq!(reg.get(&id).unwrap().state, DeviceState::Provisioning);
        reg.heartbeat(&id, Utc::now());
        assert_eq!(reg.get(&id).unwrap().state, DeviceState::Online);
    }

    #[test]
    fn heartbeat_timeout() {
        let mut reg = DeviceRegistry::new(30);
        let d = make_device("s1");
        let id = reg.register(d);
        let now = Utc::now();
        reg.heartbeat(&id, now);
        let later = now + Duration::seconds(60);
        let timed_out = reg.check_heartbeats(later);
        assert!(timed_out.contains(&id));
        assert_eq!(reg.get(&id).unwrap().state, DeviceState::Offline);
    }

    #[test]
    fn set_error_and_clear() {
        let mut reg = DeviceRegistry::new(60);
        let d = make_device("s1");
        let id = reg.register(d);
        reg.set_error(&id, "sensor stuck");
        assert_eq!(reg.get(&id).unwrap().state, DeviceState::Error);
        assert_eq!(reg.get(&id).unwrap().error_message.as_deref(), Some("sensor stuck"));
        reg.set_state(&id, DeviceState::Online);
        assert!(reg.get(&id).unwrap().error_message.is_none());
    }

    #[test]
    fn device_grouping() {
        let mut reg = DeviceRegistry::new(60);
        let d1 = make_device("s1").with_group("floor-1");
        let d2 = make_device("s2").with_group("floor-1");
        let id1 = reg.register(d1);
        let id2 = reg.register(d2);
        let members = reg.group_members("floor-1");
        assert!(members.contains(&id1));
        assert!(members.contains(&id2));
    }

    #[test]
    fn reassign_group() {
        let mut reg = DeviceRegistry::new(60);
        let d = make_device("s1").with_group("floor-1");
        let id = reg.register(d);
        reg.assign_group(&id, "floor-2");
        assert!(reg.group_members("floor-1").is_empty());
        assert!(reg.group_members("floor-2").contains(&id));
    }

    #[test]
    fn firmware_version_ordering() {
        let v1 = FirmwareVersion::new(1, 0, 0);
        let v2 = FirmwareVersion::new(1, 1, 0);
        let v3 = FirmwareVersion::new(2, 0, 0);
        assert!(v2.is_newer_than(&v1));
        assert!(v3.is_newer_than(&v2));
        assert!(!v1.is_newer_than(&v2));
    }

    #[test]
    fn firmware_version_display() {
        let v = FirmwareVersion::new(1, 2, 3).with_label("beta");
        assert_eq!(v.to_string(), "1.2.3-beta");
    }

    #[test]
    fn update_firmware() {
        let mut reg = DeviceRegistry::new(60);
        let d = make_device("s1");
        let id = reg.register(d);
        reg.update_firmware(&id, FirmwareVersion::new(2, 0, 0));
        assert_eq!(reg.get(&id).unwrap().firmware, FirmwareVersion::new(2, 0, 0));
    }

    #[test]
    fn search_by_type() {
        let mut reg = DeviceRegistry::new(60);
        reg.register(Device::new("s1", DeviceType::Sensor, FirmwareVersion::new(1, 0, 0)));
        reg.register(Device::new("g1", DeviceType::Gateway, FirmwareVersion::new(1, 0, 0)));
        let filter = DeviceFilter { device_type: Some(DeviceType::Sensor), ..Default::default() };
        let results = reg.search(&filter);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].device_type, DeviceType::Sensor);
    }

    #[test]
    fn search_by_name() {
        let mut reg = DeviceRegistry::new(60);
        reg.register(make_device("Temperature Sensor"));
        reg.register(make_device("Humidity Sensor"));
        let filter = DeviceFilter { name_contains: Some("temp".to_string()), ..Default::default() };
        let results = reg.search(&filter);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn search_by_capability() {
        let mut reg = DeviceRegistry::new(60);
        let d = make_device("s1").with_capability(Capability::new("bluetooth", "5.0"));
        reg.register(d);
        reg.register(make_device("s2"));
        let filter = DeviceFilter { capability: Some("bluetooth".to_string()), ..Default::default() };
        let results = reg.search(&filter);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn search_by_tag() {
        let mut reg = DeviceRegistry::new(60);
        let d = make_device("s1").with_tag("outdoor");
        reg.register(d);
        reg.register(make_device("s2"));
        let filter = DeviceFilter { tag: Some("outdoor".to_string()), ..Default::default() };
        let results = reg.search(&filter);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn firmware_distribution() {
        let mut reg = DeviceRegistry::new(60);
        reg.register(Device::new("a", DeviceType::Sensor, FirmwareVersion::new(1, 0, 0)));
        reg.register(Device::new("b", DeviceType::Sensor, FirmwareVersion::new(1, 0, 0)));
        reg.register(Device::new("c", DeviceType::Sensor, FirmwareVersion::new(2, 0, 0)));
        let dist = reg.firmware_distribution();
        assert_eq!(dist.get("1.0.0"), Some(&2));
        assert_eq!(dist.get("2.0.0"), Some(&1));
    }

    #[test]
    fn state_counts() {
        let mut reg = DeviceRegistry::new(60);
        let d1 = make_device("s1");
        let d2 = make_device("s2");
        let id1 = reg.register(d1);
        let id2 = reg.register(d2);
        reg.heartbeat(&id1, Utc::now());
        // id1 is online, id2 is provisioning.
        let counts = reg.state_counts();
        assert_eq!(counts.get(&DeviceState::Online), Some(&1));
        assert_eq!(counts.get(&DeviceState::Provisioning), Some(&1));
        let _ = id2;
    }

    #[test]
    fn device_has_capability() {
        let d = make_device("s1")
            .with_capability(Capability::new("wifi", "6"))
            .with_capability(Capability::new("ble", "5.2"));
        assert!(d.has_capability("wifi"));
        assert!(!d.has_capability("zigbee"));
    }

    #[test]
    fn heartbeat_for_unknown_device() {
        let mut reg = DeviceRegistry::new(60);
        assert!(!reg.heartbeat("unknown", Utc::now()));
    }

    #[test]
    fn search_by_firmware_min() {
        let mut reg = DeviceRegistry::new(60);
        reg.register(Device::new("a", DeviceType::Sensor, FirmwareVersion::new(1, 0, 0)));
        reg.register(Device::new("b", DeviceType::Sensor, FirmwareVersion::new(2, 0, 0)));
        let filter = DeviceFilter { firmware_min: Some(FirmwareVersion::new(2, 0, 0)), ..Default::default() };
        let results = reg.search(&filter);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].firmware, FirmwareVersion::new(2, 0, 0));
    }
}
