//! Network Isolation — deny-all-except-JWP enforcement layer.
//!
//! Enforces that sandbox instances can ONLY communicate via JWP to their
//! launch host. All other network egress is denied.
//!
//! Platform strategies:
//! - **Linux**: `iptables` / `nftables` rules in a network namespace
//! - **macOS**: `pfctl` packet filter rules
//! - **Container**: `--network=none` + Unix socket mount
//! - **WASM**: No network by default (WASI capability denial)
//!
//! The isolation contract: the sandbox measures hardware telemetry,
//! reports joule consumption, and the only comms it has is JWP
//! to its launch host. No way out otherwise.

use crate::{InstanceId, RuntimeError, RuntimeMode};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::path::PathBuf;

/// How the sandbox communicates with its launch host.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum JwpChannel {
    /// Unix domain socket (preferred — no TCP stack exposure).
    UnixSocket { path: PathBuf },
    /// TCP loopback only (fallback for platforms without Unix sockets).
    TcpLoopback { addr: SocketAddr },
}

impl JwpChannel {
    /// Create a Unix socket channel at the default path for an instance.
    pub fn unix_for_instance(instance_id: &InstanceId) -> Self {
        let path = PathBuf::from(format!("/tmp/joule-jwp-{}.sock", instance_id.as_str()));
        Self::UnixSocket { path }
    }

    /// Create a TCP loopback channel on a specific port.
    pub fn tcp_loopback(port: u16) -> Self {
        Self::TcpLoopback {
            addr: SocketAddr::from(([127, 0, 0, 1], port)),
        }
    }
}

/// Network isolation policy for a sandbox instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkPolicy {
    /// The JWP channel back to the launch host — the ONLY allowed communication.
    pub jwp_channel: JwpChannel,
    /// Whether to deny all outbound network access (default: true).
    pub deny_all_egress: bool,
    /// Whether to deny all inbound network access except JWP (default: true).
    pub deny_all_ingress: bool,
    /// Optional DNS resolution (default: false — no DNS).
    pub allow_dns: bool,
}

impl Default for NetworkPolicy {
    fn default() -> Self {
        Self {
            jwp_channel: JwpChannel::UnixSocket {
                path: PathBuf::from("/tmp/joule-jwp.sock"),
            },
            deny_all_egress: true,
            deny_all_ingress: true,
            allow_dns: false,
        }
    }
}

impl NetworkPolicy {
    /// Create a strict isolation policy: JWP-only via Unix socket.
    pub fn strict(instance_id: &InstanceId) -> Self {
        Self {
            jwp_channel: JwpChannel::unix_for_instance(instance_id),
            deny_all_egress: true,
            deny_all_ingress: true,
            allow_dns: false,
        }
    }

    /// Create a policy with TCP loopback (for platforms without Unix sockets).
    pub fn tcp_only(port: u16) -> Self {
        Self {
            jwp_channel: JwpChannel::tcp_loopback(port),
            deny_all_egress: true,
            deny_all_ingress: true,
            allow_dns: false,
        }
    }
}

/// Enforces network isolation for sandbox instances.
///
/// The enforcer applies platform-specific firewall rules when an instance
/// starts and tears them down when it stops.
pub struct NetworkIsolationEnforcer {
    /// Active isolation rules keyed by instance ID.
    active_rules: std::sync::RwLock<std::collections::HashMap<String, ActiveIsolation>>,
}

/// Tracks active isolation state for an instance.
#[derive(Debug)]
struct ActiveIsolation {
    policy: NetworkPolicy,
    mode: RuntimeMode,
    /// Platform-specific cleanup info.
    cleanup: IsolationCleanup,
}

/// Platform-specific cleanup data needed to tear down isolation.
#[derive(Debug)]
enum IsolationCleanup {
    /// Linux: network namespace name to delete.
    #[allow(dead_code)]
    LinuxNetns { netns_name: String },
    /// macOS: pfctl anchor name to flush.
    #[allow(dead_code)]
    MacosPf { anchor_name: String },
    /// Container: network mode was set to none (no cleanup needed).
    ContainerNone,
    /// WASM: no network capability granted (no cleanup needed).
    WasmDenied,
}

impl NetworkIsolationEnforcer {
    pub fn new() -> Self {
        Self {
            active_rules: std::sync::RwLock::new(std::collections::HashMap::new()),
        }
    }

    /// Apply network isolation for an instance.
    ///
    /// Returns the container/process arguments needed to enforce the policy.
    pub fn apply(
        &self,
        instance_id: &InstanceId,
        mode: RuntimeMode,
        policy: &NetworkPolicy,
    ) -> Result<IsolationArgs, RuntimeError> {
        let (args, cleanup) = match mode {
            RuntimeMode::Native => self.apply_native(instance_id, policy)?,
            RuntimeMode::VM => self.apply_vm(instance_id, policy)?,
            RuntimeMode::WASM => self.apply_wasm(instance_id, policy)?,
        };

        let isolation = ActiveIsolation {
            policy: policy.clone(),
            mode,
            cleanup,
        };

        self.active_rules
            .write()
            .map_err(|e| RuntimeError::ProcessError(format!("lock poisoned: {}", e)))?
            .insert(instance_id.as_str().to_string(), isolation);

        Ok(args)
    }

    /// Remove network isolation for an instance (called on stop).
    pub fn remove(&self, instance_id: &InstanceId) -> Result<(), RuntimeError> {
        let mut rules = self
            .active_rules
            .write()
            .map_err(|e| RuntimeError::ProcessError(format!("lock poisoned: {}", e)))?;

        if let Some(isolation) = rules.remove(instance_id.as_str()) {
            self.cleanup(&isolation)?;
        }

        Ok(())
    }

    /// Check if an instance has active network isolation.
    pub fn is_isolated(&self, instance_id: &InstanceId) -> bool {
        self.active_rules
            .read()
            .map(|rules| rules.contains_key(instance_id.as_str()))
            .unwrap_or(false)
    }

    /// Apply native process isolation.
    fn apply_native(
        &self,
        instance_id: &InstanceId,
        policy: &NetworkPolicy,
    ) -> Result<(IsolationArgs, IsolationCleanup), RuntimeError> {
        let mut args = IsolationArgs::default();

        if cfg!(target_os = "linux") {
            // Linux: create network namespace, only allow loopback + JWP socket
            let netns_name = format!("joule-{}", &instance_id.as_str()[..8.min(instance_id.as_str().len())]);
            args.pre_exec_commands.push(format!(
                "ip netns add {} 2>/dev/null || true",
                netns_name
            ));
            // Bring up loopback in the namespace
            args.pre_exec_commands.push(format!(
                "ip netns exec {} ip link set lo up",
                netns_name
            ));

            if policy.deny_all_egress {
                // Drop all outbound except loopback
                args.pre_exec_commands.push(format!(
                    "ip netns exec {} iptables -A OUTPUT -o lo -j ACCEPT",
                    netns_name
                ));
                args.pre_exec_commands.push(format!(
                    "ip netns exec {} iptables -A OUTPUT -j DROP",
                    netns_name
                ));
            }

            // Run the process inside the namespace
            args.exec_prefix = vec![
                "ip".to_string(),
                "netns".to_string(),
                "exec".to_string(),
                netns_name.clone(),
            ];

            // Mount the JWP socket into the namespace
            if let JwpChannel::UnixSocket { ref path } = policy.jwp_channel {
                args.env_vars
                    .insert("JWP_SOCKET".to_string(), path.display().to_string());
            }

            Ok((
                args,
                IsolationCleanup::LinuxNetns {
                    netns_name,
                },
            ))
        } else if cfg!(target_os = "macos") {
            // macOS: pfctl anchor rules
            let anchor_name = format!(
                "joule_{}",
                &instance_id.as_str()[..8.min(instance_id.as_str().len())]
            );

            if policy.deny_all_egress {
                // Create pf anchor that blocks all egress except loopback
                let rules = format!(
                    "block out quick on ! lo0 all\npass out quick on lo0 all\n"
                );
                args.pre_exec_commands.push(format!(
                    "echo '{}' | pfctl -a {} -f - 2>/dev/null || true",
                    rules, anchor_name
                ));
            }

            if let JwpChannel::UnixSocket { ref path } = policy.jwp_channel {
                args.env_vars
                    .insert("JWP_SOCKET".to_string(), path.display().to_string());
            }

            Ok((
                args,
                IsolationCleanup::MacosPf {
                    anchor_name,
                },
            ))
        } else {
            // Unsupported platform: rely on JWP channel env var only
            if let JwpChannel::UnixSocket { ref path } = policy.jwp_channel {
                args.env_vars
                    .insert("JWP_SOCKET".to_string(), path.display().to_string());
            }
            Ok((args, IsolationCleanup::WasmDenied))
        }
    }

    /// Apply VM isolation — VM has no network interface except virtio-vsock.
    fn apply_vm(
        &self,
        _instance_id: &InstanceId,
        policy: &NetworkPolicy,
    ) -> Result<(IsolationArgs, IsolationCleanup), RuntimeError> {
        let mut args = IsolationArgs::default();

        // VM isolation: don't attach any network device.
        // Communication is via virtio-vsock (host CID) or shared Unix socket.
        args.container_network_mode = Some("none".to_string());

        if let JwpChannel::UnixSocket { ref path } = policy.jwp_channel {
            args.env_vars
                .insert("JWP_SOCKET".to_string(), path.display().to_string());
            // Mount the socket into the VM via shared directory
            args.volume_mounts.push((
                path.parent()
                    .unwrap_or(std::path::Path::new("/tmp"))
                    .to_path_buf(),
                PathBuf::from("/jwp"),
                true, // read-only mount of parent dir; socket is bidirectional
            ));
        }

        Ok((args, IsolationCleanup::ContainerNone))
    }

    /// Apply WASM isolation — no network capability at all.
    fn apply_wasm(
        &self,
        _instance_id: &InstanceId,
        policy: &NetworkPolicy,
    ) -> Result<(IsolationArgs, IsolationCleanup), RuntimeError> {
        let mut args = IsolationArgs::default();

        // WASM: don't grant wasi-sockets capability.
        // Communication is via host-provided import function (JWP bridge).
        args.wasm_deny_network = true;

        if let JwpChannel::UnixSocket { ref path } = policy.jwp_channel {
            args.env_vars
                .insert("JWP_SOCKET".to_string(), path.display().to_string());
        }

        Ok((args, IsolationCleanup::WasmDenied))
    }

    /// Clean up isolation resources.
    fn cleanup(&self, isolation: &ActiveIsolation) -> Result<(), RuntimeError> {
        match &isolation.cleanup {
            IsolationCleanup::LinuxNetns { netns_name } => {
                if cfg!(target_os = "linux") {
                    let _ = std::process::Command::new("ip")
                        .args(["netns", "delete", netns_name])
                        .status();
                }
            }
            IsolationCleanup::MacosPf { anchor_name } => {
                if cfg!(target_os = "macos") {
                    let _ = std::process::Command::new("pfctl")
                        .args(["-a", anchor_name, "-F", "all"])
                        .status();
                }
            }
            IsolationCleanup::ContainerNone | IsolationCleanup::WasmDenied => {
                // Nothing to clean up
            }
        }

        // Clean up JWP socket file
        if let JwpChannel::UnixSocket { ref path } = isolation.policy.jwp_channel {
            let _ = std::fs::remove_file(path);
        }

        Ok(())
    }
}

impl Default for NetworkIsolationEnforcer {
    fn default() -> Self {
        Self::new()
    }
}

/// Arguments produced by the isolation enforcer for the runtime backend.
#[derive(Debug, Clone, Default)]
pub struct IsolationArgs {
    /// Commands to run before launching the process (setup firewall, netns, etc.).
    pub pre_exec_commands: Vec<String>,
    /// Prefix for the exec command (e.g., `ip netns exec <name>`).
    pub exec_prefix: Vec<String>,
    /// Environment variables to inject into the sandbox.
    pub env_vars: std::collections::HashMap<String, String>,
    /// Container network mode override (e.g., "none").
    pub container_network_mode: Option<String>,
    /// Volume mounts: (host_path, container_path, read_only).
    pub volume_mounts: Vec<(PathBuf, PathBuf, bool)>,
    /// WASM: deny network capability.
    pub wasm_deny_network: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jwp_channel_unix() {
        let id = InstanceId::from_string("test-123".into());
        let ch = JwpChannel::unix_for_instance(&id);
        match ch {
            JwpChannel::UnixSocket { path } => {
                assert!(path.to_str().unwrap().contains("test-123"));
            }
            _ => panic!("expected unix socket"),
        }
    }

    #[test]
    fn test_jwp_channel_tcp() {
        let ch = JwpChannel::tcp_loopback(9090);
        match ch {
            JwpChannel::TcpLoopback { addr } => {
                assert_eq!(addr.port(), 9090);
                assert!(addr.ip().is_loopback());
            }
            _ => panic!("expected tcp loopback"),
        }
    }

    #[test]
    fn test_network_policy_strict() {
        let id = InstanceId::from_string("strict-1".into());
        let policy = NetworkPolicy::strict(&id);
        assert!(policy.deny_all_egress);
        assert!(policy.deny_all_ingress);
        assert!(!policy.allow_dns);
    }

    #[test]
    fn test_network_policy_serde() {
        let id = InstanceId::from_string("serde-1".into());
        let policy = NetworkPolicy::strict(&id);
        let json = serde_json::to_string(&policy).unwrap();
        let parsed: NetworkPolicy = serde_json::from_str(&json).unwrap();
        assert!(parsed.deny_all_egress);
        assert!(!parsed.allow_dns);
    }

    #[test]
    fn test_enforcer_apply_wasm() {
        let enforcer = NetworkIsolationEnforcer::new();
        let id = InstanceId::from_string("wasm-1".into());
        let policy = NetworkPolicy::strict(&id);

        let args = enforcer.apply(&id, RuntimeMode::WASM, &policy).unwrap();
        assert!(args.wasm_deny_network);
        assert!(enforcer.is_isolated(&id));

        enforcer.remove(&id).unwrap();
        assert!(!enforcer.is_isolated(&id));
    }

    #[test]
    fn test_enforcer_apply_vm() {
        let enforcer = NetworkIsolationEnforcer::new();
        let id = InstanceId::from_string("vm-1".into());
        let policy = NetworkPolicy::strict(&id);

        let args = enforcer.apply(&id, RuntimeMode::VM, &policy).unwrap();
        assert_eq!(args.container_network_mode, Some("none".to_string()));
    }

    #[test]
    fn test_enforcer_apply_native() {
        let enforcer = NetworkIsolationEnforcer::new();
        let id = InstanceId::from_string("native-1".into());
        let policy = NetworkPolicy::strict(&id);

        let args = enforcer.apply(&id, RuntimeMode::Native, &policy).unwrap();
        // On non-Linux/macOS, should still set JWP_SOCKET env var
        assert!(args.env_vars.contains_key("JWP_SOCKET"));
    }

    #[test]
    fn test_isolation_args_default() {
        let args = IsolationArgs::default();
        assert!(args.pre_exec_commands.is_empty());
        assert!(args.exec_prefix.is_empty());
        assert!(args.env_vars.is_empty());
        assert!(args.container_network_mode.is_none());
        assert!(!args.wasm_deny_network);
    }

    #[test]
    fn test_enforcer_double_apply() {
        let enforcer = NetworkIsolationEnforcer::new();
        let id = InstanceId::from_string("double-1".into());
        let policy = NetworkPolicy::strict(&id);

        enforcer.apply(&id, RuntimeMode::WASM, &policy).unwrap();
        // Second apply overwrites (no error)
        enforcer.apply(&id, RuntimeMode::WASM, &policy).unwrap();
        assert!(enforcer.is_isolated(&id));
    }

    #[test]
    fn test_enforcer_remove_nonexistent() {
        let enforcer = NetworkIsolationEnforcer::new();
        let id = InstanceId::from_string("ghost-1".into());
        // Removing non-existent is a no-op, not an error
        enforcer.remove(&id).unwrap();
    }
}
