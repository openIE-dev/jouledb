use crate::{InstanceId, InstanceInfo, InstanceState, RuntimeError};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::RwLock;

/// Tracks all JouleDB instances with JSON persistence.
pub struct InstanceRegistry {
    instances: RwLock<HashMap<String, InstanceInfo>>,
    persistence_path: Option<PathBuf>,
}

impl InstanceRegistry {
    /// Create an in-memory-only registry (no persistence).
    pub fn new() -> Self {
        Self {
            instances: RwLock::new(HashMap::new()),
            persistence_path: None,
        }
    }

    /// Create a registry that persists state to `{data_dir}/instances.json`.
    pub fn with_persistence(data_dir: &Path) -> Result<Self, RuntimeError> {
        let path = data_dir.join("instances.json");
        let instances = if path.exists() {
            let data = std::fs::read_to_string(&path)?;
            serde_json::from_str(&data).unwrap_or_default()
        } else {
            HashMap::new()
        };

        Ok(Self {
            instances: RwLock::new(instances),
            persistence_path: Some(path),
        })
    }

    /// Register a new instance.
    pub fn register(&self, info: InstanceInfo) -> Result<(), RuntimeError> {
        let id = info.id.0.clone();
        let mut instances = self.instances.write().unwrap();
        if instances.contains_key(&id) {
            return Err(RuntimeError::InstanceAlreadyExists(id));
        }
        instances.insert(id, info);
        drop(instances);
        self.persist();
        Ok(())
    }

    /// Remove an instance from the registry.
    pub fn deregister(&self, instance_id: &str) -> Result<InstanceInfo, RuntimeError> {
        let mut instances = self.instances.write().unwrap();
        let info = instances
            .remove(instance_id)
            .ok_or_else(|| RuntimeError::InstanceNotFound(instance_id.into()))?;
        drop(instances);
        self.persist();
        Ok(info)
    }

    /// Get instance info by ID.
    pub fn get(&self, instance_id: &str) -> Option<InstanceInfo> {
        let instances = self.instances.read().unwrap();
        instances.get(instance_id).cloned()
    }

    /// List all registered instances.
    pub fn list(&self) -> Vec<InstanceInfo> {
        let instances = self.instances.read().unwrap();
        instances.values().cloned().collect()
    }

    /// Update the state of an instance.
    pub fn update_state(
        &self,
        instance_id: &str,
        state: InstanceState,
    ) -> Result<(), RuntimeError> {
        let mut instances = self.instances.write().unwrap();
        let info = instances
            .get_mut(instance_id)
            .ok_or_else(|| RuntimeError::InstanceNotFound(instance_id.into()))?;
        info.state = state;
        drop(instances);
        self.persist();
        Ok(())
    }

    /// Update the PID of an instance.
    pub fn update_pid(&self, instance_id: &str, pid: Option<u32>) -> Result<(), RuntimeError> {
        let mut instances = self.instances.write().unwrap();
        let info = instances
            .get_mut(instance_id)
            .ok_or_else(|| RuntimeError::InstanceNotFound(instance_id.into()))?;
        info.pid = pid;
        drop(instances);
        self.persist();
        Ok(())
    }

    /// Number of registered instances.
    pub fn count(&self) -> usize {
        let instances = self.instances.read().unwrap();
        instances.len()
    }

    /// Persist the registry to disk (atomic write via temp + rename).
    fn persist(&self) {
        if let Some(path) = &self.persistence_path {
            let instances = self.instances.read().unwrap();
            let json = serde_json::to_string_pretty(&*instances).unwrap_or_default();
            drop(instances);

            let tmp_path = path.with_extension("json.tmp");
            if let Err(e) = std::fs::write(&tmp_path, &json) {
                log::error!("Failed to write instance registry: {}", e);
                return;
            }
            if let Err(e) = std::fs::rename(&tmp_path, path) {
                log::error!("Failed to rename instance registry: {}", e);
            }
        }
    }
}

impl Default for InstanceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{InstanceId, InstanceState, PortMapping, RuntimeMode, WorkloadKind};
    use chrono::Utc;
    use std::collections::HashMap;

    fn test_instance(name: &str) -> InstanceInfo {
        InstanceInfo {
            id: InstanceId::new(),
            name: name.into(),
            engine: Default::default(),
            workload: WorkloadKind::default(),
            mode: RuntimeMode::Native,
            state: InstanceState::Running,
            created_at: Utc::now(),
            pid: Some(1234),
            ports: vec![PortMapping {
                protocol: "http".into(),
                host_port: 8080,
                instance_port: 8080,
            }],
            data_dir: "/tmp/test".into(),
            node_id: None,
            energy_port: None,
            accelerators: vec![],
            volumes: vec![],
            env_vars: HashMap::new(),
            labels: HashMap::new(),
        }
    }

    #[test]
    fn test_register_and_get() {
        let registry = InstanceRegistry::new();
        let info = test_instance("db1");
        let id = info.id.0.clone();

        registry.register(info).unwrap();
        let got = registry.get(&id).unwrap();
        assert_eq!(got.name, "db1");
    }

    #[test]
    fn test_register_duplicate() {
        let registry = InstanceRegistry::new();
        let info = test_instance("db1");
        let id = info.id.0.clone();

        registry.register(info).unwrap();

        let info2 = InstanceInfo {
            id: InstanceId::from_string(id),
            name: "db1-dup".into(),
            engine: Default::default(),
            workload: WorkloadKind::default(),
            mode: RuntimeMode::Native,
            state: InstanceState::Running,
            created_at: Utc::now(),
            pid: None,
            ports: vec![],
            data_dir: "/tmp/test2".into(),
            node_id: None,
            energy_port: None,
            accelerators: vec![],
            volumes: vec![],
            env_vars: HashMap::new(),
            labels: HashMap::new(),
        };
        assert!(registry.register(info2).is_err());
    }

    #[test]
    fn test_deregister() {
        let registry = InstanceRegistry::new();
        let info = test_instance("db1");
        let id = info.id.0.clone();

        registry.register(info).unwrap();
        assert_eq!(registry.count(), 1);

        let removed = registry.deregister(&id).unwrap();
        assert_eq!(removed.name, "db1");
        assert_eq!(registry.count(), 0);
    }

    #[test]
    fn test_deregister_not_found() {
        let registry = InstanceRegistry::new();
        assert!(registry.deregister("nonexistent").is_err());
    }

    #[test]
    fn test_list() {
        let registry = InstanceRegistry::new();
        registry.register(test_instance("db1")).unwrap();
        registry.register(test_instance("db2")).unwrap();

        let list = registry.list();
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn test_update_state() {
        let registry = InstanceRegistry::new();
        let info = test_instance("db1");
        let id = info.id.0.clone();

        registry.register(info).unwrap();
        registry.update_state(&id, InstanceState::Stopped).unwrap();

        let got = registry.get(&id).unwrap();
        assert_eq!(got.state, InstanceState::Stopped);
    }

    #[test]
    fn test_persistence_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let registry = InstanceRegistry::with_persistence(dir.path()).unwrap();

        let info = test_instance("db1");
        let id = info.id.0.clone();
        registry.register(info).unwrap();

        // Load from disk
        let registry2 = InstanceRegistry::with_persistence(dir.path()).unwrap();
        let got = registry2.get(&id).unwrap();
        assert_eq!(got.name, "db1");
        assert_eq!(got.state, InstanceState::Running);
    }

    #[test]
    fn test_empty_persistence() {
        let dir = tempfile::tempdir().unwrap();
        let registry = InstanceRegistry::with_persistence(dir.path()).unwrap();
        assert_eq!(registry.count(), 0);
    }
}
