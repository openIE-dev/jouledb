use crate::{
    DatabaseEngine, InstanceId, InstanceInfo, RuntimeConfig, RuntimeError, RuntimeMode,
    ServerOverrides, manager::RuntimeManager,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Configuration for a single node in a cluster.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConfig {
    pub node_id: String,
    pub addr: String,
    pub mode: RuntimeMode,
    pub data_dir: String,
    pub raft_port: u16,
    pub http_port: u16,
    pub tcp_port: u16,
    pub pgwire_port: u16,
}

/// Cluster deployment configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterConfig {
    pub name: String,
    pub nodes: Vec<NodeConfig>,
    pub raft_master_secret: Option<String>,
}

impl ClusterConfig {
    pub fn new(name: String) -> Self {
        Self {
            name,
            nodes: Vec::new(),
            raft_master_secret: None,
        }
    }

    pub fn add_node(&mut self, node: NodeConfig) {
        self.nodes.push(node);
    }

    pub fn with_secret(mut self, secret: String) -> Self {
        self.raft_master_secret = Some(secret);
        self
    }

    /// Build Raft peer list string from all nodes.
    pub fn raft_peers(&self) -> Vec<String> {
        self.nodes
            .iter()
            .map(|n| format!("{}={}:{}", n.node_id, n.addr, n.raft_port))
            .collect()
    }

    /// Validate cluster configuration.
    pub fn validate(&self) -> Result<(), RuntimeError> {
        if self.nodes.is_empty() {
            return Err(RuntimeError::ConfigError(
                "cluster must have at least one node".into(),
            ));
        }

        // Check for duplicate node IDs
        let mut seen = std::collections::HashSet::new();
        for node in &self.nodes {
            if !seen.insert(&node.node_id) {
                return Err(RuntimeError::ConfigError(format!(
                    "duplicate node_id: {}",
                    node.node_id
                )));
            }
        }

        // Check for duplicate addresses
        let mut addrs = std::collections::HashSet::new();
        for node in &self.nodes {
            let addr = format!("{}:{}", node.addr, node.raft_port);
            if !addrs.insert(addr.clone()) {
                return Err(RuntimeError::ConfigError(format!(
                    "duplicate raft address: {}",
                    addr
                )));
            }
        }

        Ok(())
    }
}

/// Manages multi-node JouleDB clusters with mixed isolation modes.
///
/// Raft-based clustering is currently only supported for JouleDB.
/// Other engines will need engine-specific replication (e.g., Postgres
/// streaming replication, Redis Sentinel) in future releases.
pub struct ClusterManager {
    config: ClusterConfig,
    managers: Vec<(String, RuntimeManager)>,
}

impl ClusterManager {
    /// Create a cluster manager for local deployment (all nodes on this machine).
    pub fn new_local(config: ClusterConfig) -> Result<Self, RuntimeError> {
        config.validate()?;

        let mut managers = Vec::new();
        for node in &config.nodes {
            let runtime_config = RuntimeConfig {
                mode: node.mode,
                ..Default::default()
            };
            let data_dir = PathBuf::from(&node.data_dir);
            let manager = RuntimeManager::new(runtime_config, data_dir)?;
            managers.push((node.node_id.clone(), manager));
        }

        Ok(Self { config, managers })
    }

    /// Start all nodes in the cluster.
    pub async fn start_all(&self) -> Result<Vec<InstanceId>, RuntimeError> {
        let peers = self.config.raft_peers();
        let mut instance_ids = Vec::new();

        for (i, (node_id, manager)) in self.managers.iter().enumerate() {
            let node_config = &self.config.nodes[i];

            let mut overrides = ServerOverrides {
                http_port: Some(node_config.http_port),
                tcp_port: Some(node_config.tcp_port),
                pgwire_port: Some(node_config.pgwire_port),
                raft_port: Some(node_config.raft_port),
                data_dir: Some(node_config.data_dir.clone()),
                ..Default::default()
            };

            // Add Raft configuration as extra args
            overrides.extra_args.push("--raft-node-id".into());
            overrides.extra_args.push(node_id.clone());
            overrides.extra_args.push("--raft-peers".into());
            overrides.extra_args.push(peers.join(","));

            if let Some(secret) = &self.config.raft_master_secret {
                overrides.extra_args.push("--raft-master-secret".into());
                overrides.extra_args.push(secret.clone());
            }

            let name = format!("{}-{}", self.config.name, node_id);
            let id = manager
                .start_instance(name, DatabaseEngine::JouleDB, overrides)
                .await?;
            instance_ids.push(id);
        }

        Ok(instance_ids)
    }

    /// Stop all nodes in the cluster.
    pub async fn stop_all(&self) -> Result<(), RuntimeError> {
        for (_node_id, manager) in &self.managers {
            for instance in manager.list_instances() {
                manager.stop_instance(instance.id.as_str()).await?;
            }
        }
        Ok(())
    }

    /// Get cluster status.
    pub fn status(&self) -> Vec<(String, Vec<InstanceInfo>)> {
        self.managers
            .iter()
            .map(|(node_id, manager)| (node_id.clone(), manager.list_instances()))
            .collect()
    }

    /// Get the cluster configuration.
    pub fn config(&self) -> &ClusterConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cluster_config_raft_peers() {
        let mut config = ClusterConfig::new("test-cluster".into());
        config.add_node(NodeConfig {
            node_id: "node1".into(),
            addr: "198.51.100.1".into(),
            mode: RuntimeMode::Native,
            data_dir: "/data/node1".into(),
            raft_port: 7000,
            http_port: 8080,
            tcp_port: 9000,
            pgwire_port: 5433,
        });
        config.add_node(NodeConfig {
            node_id: "node2".into(),
            addr: "198.51.100.2".into(),
            mode: RuntimeMode::VM,
            data_dir: "/data/node2".into(),
            raft_port: 7000,
            http_port: 8080,
            tcp_port: 9000,
            pgwire_port: 5433,
        });

        let peers = config.raft_peers();
        assert_eq!(peers.len(), 2);
        assert!(peers[0].contains("node1=198.51.100.1:7000"));
        assert!(peers[1].contains("node2=198.51.100.2:7000"));
    }

    #[test]
    fn test_cluster_config_validate_empty() {
        let config = ClusterConfig::new("empty".into());
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_cluster_config_validate_duplicate_ids() {
        let mut config = ClusterConfig::new("dup".into());
        config.add_node(NodeConfig {
            node_id: "node1".into(),
            addr: "198.51.100.1".into(),
            mode: RuntimeMode::Native,
            data_dir: "/data/n1".into(),
            raft_port: 7000,
            http_port: 8080,
            tcp_port: 9000,
            pgwire_port: 5433,
        });
        config.add_node(NodeConfig {
            node_id: "node1".into(), // duplicate
            addr: "198.51.100.2".into(),
            mode: RuntimeMode::Native,
            data_dir: "/data/n2".into(),
            raft_port: 7001,
            http_port: 8081,
            tcp_port: 9001,
            pgwire_port: 5434,
        });
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_cluster_config_validate_ok() {
        let mut config = ClusterConfig::new("good".into());
        config.add_node(NodeConfig {
            node_id: "node1".into(),
            addr: "198.51.100.1".into(),
            mode: RuntimeMode::Native,
            data_dir: "/data/n1".into(),
            raft_port: 7000,
            http_port: 8080,
            tcp_port: 9000,
            pgwire_port: 5433,
        });
        config.add_node(NodeConfig {
            node_id: "node2".into(),
            addr: "198.51.100.2".into(),
            mode: RuntimeMode::VM,
            data_dir: "/data/n2".into(),
            raft_port: 7000,
            http_port: 8080,
            tcp_port: 9000,
            pgwire_port: 5433,
        });
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_cluster_config_serde() {
        let mut config = ClusterConfig::new("serde-test".into());
        config.add_node(NodeConfig {
            node_id: "node1".into(),
            addr: "127.0.0.1".into(),
            mode: RuntimeMode::Native,
            data_dir: "/tmp/n1".into(),
            raft_port: 7000,
            http_port: 8080,
            tcp_port: 9000,
            pgwire_port: 5433,
        });
        let json = serde_json::to_string(&config).unwrap();
        let parsed: ClusterConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.name, "serde-test");
        assert_eq!(parsed.nodes.len(), 1);
    }
}
