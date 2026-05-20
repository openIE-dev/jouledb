//! Container Networking — port allocation, network management, DNS.
//!
//! Manages port assignments for container instances and provides
//! network isolation modes (bridge, host, none).

use crate::RuntimeError;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use tokio::sync::RwLock;

/// Network isolation mode for containers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NetworkMode {
    /// Bridge mode — isolated networking with port mapping (default).
    Bridge,
    /// Host mode — container shares host network namespace.
    Host,
    /// No networking.
    None,
}

impl Default for NetworkMode {
    fn default() -> Self {
        Self::Bridge
    }
}

/// A container network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Network {
    /// Network name.
    pub name: String,
    /// Isolation mode.
    pub mode: NetworkMode,
    /// Optional subnet (e.g. `"172.18.0.0/16"`).
    pub subnet: Option<String>,
    /// Instance IDs connected to this network.
    pub members: Vec<String>,
}

/// Manages container networking and port allocation.
pub struct NetworkManager {
    /// Named networks.
    networks: RwLock<HashMap<String, Network>>,
    /// Port allocator for host port mapping.
    port_allocator: RwLock<PortAllocator>,
}

/// Tracks allocated host ports to avoid conflicts.
struct PortAllocator {
    /// Port range for auto-assignment.
    range_start: u16,
    range_end: u16,
    /// Currently allocated ports.
    allocated: HashSet<u16>,
}

impl PortAllocator {
    fn new(range_start: u16, range_end: u16) -> Self {
        Self {
            range_start,
            range_end,
            allocated: HashSet::new(),
        }
    }

    /// Allocate a specific port. Returns error if already taken.
    fn allocate(&mut self, port: u16) -> Result<u16, RuntimeError> {
        if self.allocated.contains(&port) {
            return Err(RuntimeError::ConfigError(format!(
                "port {} is already allocated",
                port
            )));
        }
        self.allocated.insert(port);
        Ok(port)
    }

    /// Auto-allocate the next available port from the ephemeral range.
    fn allocate_next(&mut self) -> Result<u16, RuntimeError> {
        for port in self.range_start..=self.range_end {
            if !self.allocated.contains(&port) {
                self.allocated.insert(port);
                return Ok(port);
            }
        }
        Err(RuntimeError::ConfigError(
            "no available ports in allocation range".into(),
        ))
    }

    /// Release a previously allocated port.
    fn release(&mut self, port: u16) {
        self.allocated.remove(&port);
    }

    /// Check if a port is available.
    fn is_available(&self, port: u16) -> bool {
        !self.allocated.contains(&port)
    }

    /// Number of allocated ports.
    fn count(&self) -> usize {
        self.allocated.len()
    }
}

impl NetworkManager {
    /// Create a new network manager with default port range (49152-65535).
    pub fn new() -> Self {
        Self {
            networks: RwLock::new(HashMap::new()),
            port_allocator: RwLock::new(PortAllocator::new(49152, 65535)),
        }
    }

    /// Create a new network manager with a custom port range.
    pub fn with_port_range(range_start: u16, range_end: u16) -> Self {
        Self {
            networks: RwLock::new(HashMap::new()),
            port_allocator: RwLock::new(PortAllocator::new(range_start, range_end)),
        }
    }

    /// Create a named network.
    pub async fn create_network(
        &self,
        name: String,
        mode: NetworkMode,
        subnet: Option<String>,
    ) -> Result<(), RuntimeError> {
        let mut networks = self.networks.write().await;
        if networks.contains_key(&name) {
            return Err(RuntimeError::ConfigError(format!(
                "network '{}' already exists",
                name
            )));
        }
        networks.insert(
            name.clone(),
            Network {
                name,
                mode,
                subnet,
                members: vec![],
            },
        );
        Ok(())
    }

    /// Remove a network (must have no members).
    pub async fn remove_network(&self, name: &str) -> Result<(), RuntimeError> {
        let mut networks = self.networks.write().await;
        let network = networks
            .get(name)
            .ok_or_else(|| RuntimeError::ConfigError(format!("network '{}' not found", name)))?;

        if !network.members.is_empty() {
            return Err(RuntimeError::ConfigError(format!(
                "network '{}' still has {} members",
                name,
                network.members.len()
            )));
        }
        networks.remove(name);
        Ok(())
    }

    /// Connect an instance to a network.
    pub async fn connect(&self, network_name: &str, instance_id: &str) -> Result<(), RuntimeError> {
        let mut networks = self.networks.write().await;
        let network = networks.get_mut(network_name).ok_or_else(|| {
            RuntimeError::ConfigError(format!("network '{}' not found", network_name))
        })?;

        if !network.members.contains(&instance_id.to_string()) {
            network.members.push(instance_id.to_string());
        }
        Ok(())
    }

    /// Disconnect an instance from a network.
    pub async fn disconnect(
        &self,
        network_name: &str,
        instance_id: &str,
    ) -> Result<(), RuntimeError> {
        let mut networks = self.networks.write().await;
        let network = networks.get_mut(network_name).ok_or_else(|| {
            RuntimeError::ConfigError(format!("network '{}' not found", network_name))
        })?;
        network.members.retain(|m| m != instance_id);
        Ok(())
    }

    /// List all networks.
    pub async fn list_networks(&self) -> Vec<Network> {
        self.networks.read().await.values().cloned().collect()
    }

    /// Get a specific network.
    pub async fn get_network(&self, name: &str) -> Option<Network> {
        self.networks.read().await.get(name).cloned()
    }

    /// Allocate a specific host port.
    pub async fn allocate_port(&self, port: u16) -> Result<u16, RuntimeError> {
        self.port_allocator.write().await.allocate(port)
    }

    /// Auto-allocate the next available host port.
    pub async fn allocate_next_port(&self) -> Result<u16, RuntimeError> {
        self.port_allocator.write().await.allocate_next()
    }

    /// Release a previously allocated port.
    pub async fn release_port(&self, port: u16) {
        self.port_allocator.write().await.release(port);
    }

    /// Check if a port is available.
    pub async fn is_port_available(&self, port: u16) -> bool {
        self.port_allocator.read().await.is_available(port)
    }

    /// Number of allocated ports.
    pub async fn allocated_port_count(&self) -> usize {
        self.port_allocator.read().await.count()
    }

    /// Disconnect an instance from all networks (called on instance stop).
    pub async fn disconnect_all(&self, instance_id: &str) {
        let mut networks = self.networks.write().await;
        for network in networks.values_mut() {
            network.members.retain(|m| m != instance_id);
        }
    }
}

impl Default for NetworkManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_mode_default() {
        assert_eq!(NetworkMode::default(), NetworkMode::Bridge);
    }

    #[test]
    fn test_port_allocator_basic() {
        let mut alloc = PortAllocator::new(49152, 49162);
        assert_eq!(alloc.count(), 0);
        assert!(alloc.is_available(49152));

        let port = alloc.allocate(49152).unwrap();
        assert_eq!(port, 49152);
        assert!(!alloc.is_available(49152));
        assert_eq!(alloc.count(), 1);

        // Double-allocate should fail
        assert!(alloc.allocate(49152).is_err());

        alloc.release(49152);
        assert!(alloc.is_available(49152));
        assert_eq!(alloc.count(), 0);
    }

    #[test]
    fn test_port_allocator_next() {
        let mut alloc = PortAllocator::new(50000, 50002);
        assert_eq!(alloc.allocate_next().unwrap(), 50000);
        assert_eq!(alloc.allocate_next().unwrap(), 50001);
        assert_eq!(alloc.allocate_next().unwrap(), 50002);
        // Range exhausted
        assert!(alloc.allocate_next().is_err());

        alloc.release(50001);
        assert_eq!(alloc.allocate_next().unwrap(), 50001);
    }

    #[tokio::test]
    async fn test_network_manager_create_remove() {
        let nm = NetworkManager::new();
        nm.create_network("test-net".into(), NetworkMode::Bridge, None)
            .await
            .unwrap();

        let nets = nm.list_networks().await;
        assert_eq!(nets.len(), 1);
        assert_eq!(nets[0].name, "test-net");

        // Duplicate should fail
        assert!(
            nm.create_network("test-net".into(), NetworkMode::Bridge, None)
                .await
                .is_err()
        );

        nm.remove_network("test-net").await.unwrap();
        assert!(nm.list_networks().await.is_empty());
    }

    #[tokio::test]
    async fn test_network_manager_connect_disconnect() {
        let nm = NetworkManager::new();
        nm.create_network("app-net".into(), NetworkMode::Bridge, None)
            .await
            .unwrap();

        nm.connect("app-net", "inst-1").await.unwrap();
        nm.connect("app-net", "inst-2").await.unwrap();

        let net = nm.get_network("app-net").await.unwrap();
        assert_eq!(net.members.len(), 2);

        // Can't remove network with members
        assert!(nm.remove_network("app-net").await.is_err());

        nm.disconnect("app-net", "inst-1").await.unwrap();
        nm.disconnect("app-net", "inst-2").await.unwrap();
        nm.remove_network("app-net").await.unwrap();
    }

    #[tokio::test]
    async fn test_network_manager_disconnect_all() {
        let nm = NetworkManager::new();
        nm.create_network("net-a".into(), NetworkMode::Bridge, None)
            .await
            .unwrap();
        nm.create_network("net-b".into(), NetworkMode::Bridge, None)
            .await
            .unwrap();

        nm.connect("net-a", "inst-x").await.unwrap();
        nm.connect("net-b", "inst-x").await.unwrap();

        nm.disconnect_all("inst-x").await;

        assert!(nm.get_network("net-a").await.unwrap().members.is_empty());
        assert!(nm.get_network("net-b").await.unwrap().members.is_empty());
    }

    #[tokio::test]
    async fn test_network_manager_port_allocation() {
        let nm = NetworkManager::with_port_range(60000, 60005);

        let p1 = nm.allocate_next_port().await.unwrap();
        assert_eq!(p1, 60000);

        let p2 = nm.allocate_port(60003).await.unwrap();
        assert_eq!(p2, 60003);

        assert!(!nm.is_port_available(60000).await);
        assert!(!nm.is_port_available(60003).await);
        assert!(nm.is_port_available(60001).await);

        nm.release_port(60000).await;
        assert!(nm.is_port_available(60000).await);

        assert_eq!(nm.allocated_port_count().await, 1); // only 60003 left
    }

    #[test]
    fn test_network_serde() {
        let net = Network {
            name: "mynet".into(),
            mode: NetworkMode::Bridge,
            subnet: Some("172.20.0.0/16".into()),
            members: vec!["a".into(), "b".into()],
        };
        let json = serde_json::to_string(&net).unwrap();
        let parsed: Network = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.name, "mynet");
        assert_eq!(parsed.mode, NetworkMode::Bridge);
        assert_eq!(parsed.subnet, Some("172.20.0.0/16".into()));
        assert_eq!(parsed.members.len(), 2);
    }
}
