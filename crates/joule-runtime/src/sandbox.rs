//! Sandbox — unified orchestration of all five isolation gaps.
//!
//! Composes network isolation, energy enforcement, cryptographic attestation,
//! resource limits, and graceful shutdown into a single `Sandbox` that wraps
//! any workload instance.
//!
//! The sandbox contract:
//! - Measures hardware telemetry (RAPL, GPU) and reports joule consumption
//! - Communicates ONLY via JWP to its launch host
//! - Signs every energy receipt with HMAC-SHA256
//! - Hard-kills the workload if energy budget is exceeded
//! - Drains in-flight JWP frames on shutdown before terminating
//!
//! Usage:
//! ```ignore
//! let config = SandboxConfig::strict(instance_id, 50_000_000 /* 50 J */);
//! let sandbox = Sandbox::new(config)?;
//! sandbox.start(meter)?;
//! // ... workload runs, JWP frames flow through sandbox.pending() ...
//! let result = sandbox.stop(Some(pid));
//! ```

use crate::attestation::{AttestationKey, ReceiptSigner, ReceiptVerifier, SignedEnergyReceipt};
use crate::energy_enforcer::{EnergyEnforcer, EnergyEnforcerConfig, EnergyEnforcerState, EnforcerResult};
use crate::graceful_shutdown::{PendingOperations, ShutdownConfig, ShutdownCoordinator, ShutdownResult};
use crate::network_isolation::{IsolationArgs, NetworkIsolationEnforcer, NetworkPolicy};
use crate::resource_limits::ResourceLimits;
use crate::{InstanceId, RuntimeError, RuntimeMode};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Full sandbox configuration composing all five isolation gaps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxConfig {
    /// Instance this sandbox wraps.
    pub instance_id: String,
    /// Runtime mode for the workload.
    pub mode: RuntimeMode,
    /// Network isolation policy (JWP-only).
    pub network: NetworkPolicy,
    /// Resource limits (CPU, memory, disk, energy, timeout).
    pub resource_limits: ResourceLimits,
    /// Graceful shutdown configuration.
    pub shutdown: ShutdownConfig,
}

impl SandboxConfig {
    /// Create a strict sandbox: JWP-only, energy-budgeted, hard-kill on exceed.
    pub fn strict(instance_id: &InstanceId, max_energy_uj: u64) -> Self {
        let mut resource_limits = ResourceLimits::default();
        resource_limits.energy.max_energy_uj = Some(max_energy_uj);
        resource_limits.energy.hard_kill = true;
        resource_limits.disk.tmpfs_only = true;
        resource_limits.process.max_pids = Some(256);

        Self {
            instance_id: instance_id.as_str().to_string(),
            mode: RuntimeMode::Native,
            network: NetworkPolicy::strict(instance_id),
            resource_limits,
            shutdown: ShutdownConfig::default(),
        }
    }

    /// Create a sandbox for WASM workloads.
    pub fn wasm(instance_id: &InstanceId, max_energy_uj: u64) -> Self {
        let mut config = Self::strict(instance_id, max_energy_uj);
        config.mode = RuntimeMode::WASM;
        config
    }

    /// Create a sandbox for VM workloads.
    pub fn vm(instance_id: &InstanceId, max_energy_uj: u64) -> Self {
        let mut config = Self::strict(instance_id, max_energy_uj);
        config.mode = RuntimeMode::VM;
        config
    }
}

/// A running sandbox instance with all isolation enforced.
pub struct Sandbox {
    config: SandboxConfig,
    /// Tracks in-flight JWP frames for graceful shutdown.
    pending: Arc<PendingOperations>,
    /// Network isolation enforcer (shared, manages firewall rules).
    network_enforcer: Arc<NetworkIsolationEnforcer>,
    /// Energy budget enforcer (hard kill on exceed).
    energy_enforcer: Option<EnergyEnforcer>,
    /// Attestation key for signing energy receipts.
    attestation_key: AttestationKey,
    /// Receipt signer (inside the sandbox).
    signer: ReceiptSigner,
    /// Network isolation arguments produced during setup.
    isolation_args: Option<IsolationArgs>,
}

impl Sandbox {
    /// Create a new sandbox with the given configuration.
    pub fn new(config: SandboxConfig) -> Result<Self, RuntimeError> {
        let instance_id = InstanceId::from_string(config.instance_id.clone());

        // Generate attestation key
        let attestation_key = AttestationKey::generate(&config.instance_id);
        let signer = ReceiptSigner::new(attestation_key.clone());

        // Set up network isolation
        let network_enforcer = Arc::new(NetworkIsolationEnforcer::new());
        let isolation_args =
            network_enforcer.apply(&instance_id, config.mode, &config.network)?;

        // Set up energy enforcer
        let energy_enforcer = config
            .resource_limits
            .energy
            .max_energy_uj
            .map(|max_uj| {
                EnergyEnforcer::new(EnergyEnforcerConfig {
                    max_energy_uj: max_uj,
                    sample_interval: std::time::Duration::from_millis(
                        config.resource_limits.energy.sample_interval_ms,
                    ),
                    warning_threshold: config.resource_limits.energy.warning_threshold,
                    hard_kill: config.resource_limits.energy.hard_kill,
                    target_pid: None, // Set later via set_target_pid
                })
            });

        let pending = Arc::new(PendingOperations::new());

        Ok(Self {
            config,
            pending,
            network_enforcer,
            energy_enforcer,
            attestation_key,
            signer,
            isolation_args: Some(isolation_args),
        })
    }

    /// Set the target PID for energy enforcement (call after process starts).
    pub fn set_target_pid(&mut self, pid: u32) {
        if let Some(ref mut enforcer) = self.energy_enforcer {
            let state = enforcer.state();
            // Re-create with target PID
            let config = EnergyEnforcerConfig {
                max_energy_uj: state.budget_uj(),
                sample_interval: std::time::Duration::from_millis(
                    self.config.resource_limits.energy.sample_interval_ms,
                ),
                warning_threshold: self.config.resource_limits.energy.warning_threshold,
                hard_kill: self.config.resource_limits.energy.hard_kill,
                target_pid: Some(pid),
            };
            *enforcer = EnergyEnforcer::new(config);
        }
    }

    /// Start energy enforcement. Call after the workload process is running.
    pub fn start_energy_enforcement(
        &mut self,
        meter: Box<dyn inv_energy::meter::EnergyMeter + Send>,
    ) -> Result<(), RuntimeError> {
        if let Some(ref mut enforcer) = self.energy_enforcer {
            enforcer.start(meter).map_err(|e| {
                RuntimeError::ProcessError(format!("energy enforcer start failed: {}", e))
            })?;
        }
        Ok(())
    }

    /// Get the pending operations tracker (share with JWP frame handlers).
    pub fn pending(&self) -> Arc<PendingOperations> {
        Arc::clone(&self.pending)
    }

    /// Get the energy enforcer state for monitoring.
    pub fn energy_state(&self) -> Option<Arc<EnergyEnforcerState>> {
        self.energy_enforcer.as_ref().map(|e| e.state())
    }

    /// Take the isolation args (consumed when passing to the backend).
    pub fn take_isolation_args(&mut self) -> Option<IsolationArgs> {
        self.isolation_args.take()
    }

    /// Get the container runtime arguments from resource limits.
    pub fn container_args(&self) -> Vec<String> {
        self.config.resource_limits.to_container_args()
    }

    /// Get environment variables to inject into the sandbox process.
    pub fn env_vars(&self) -> std::collections::HashMap<String, String> {
        let mut vars = std::collections::HashMap::new();

        // Resource limit env vars
        let resolved = self.config.resource_limits.for_mode(self.config.mode);
        vars.extend(resolved.env_vars);

        // Isolation env vars
        if let Some(ref args) = self.isolation_args {
            vars.extend(args.env_vars.clone());
        }

        // Attestation: inject key transport bytes as hex
        vars.insert(
            "JOULE_ATTESTATION_KEY".to_string(),
            hex_encode(&self.attestation_key.to_bytes()),
        );

        vars
    }

    /// Sign an energy measurement and produce an attested receipt.
    pub fn sign_receipt(
        &mut self,
        energy_uj: u64,
        carbon_ugco2eq: u64,
    ) -> SignedEnergyReceipt {
        self.signer.sign(energy_uj, carbon_ugco2eq)
    }

    /// Create a verifier for receipts from this sandbox (used by launch host).
    pub fn create_verifier(&self) -> ReceiptVerifier {
        ReceiptVerifier::new(self.attestation_key.clone())
    }

    /// Get the attestation key (for transport to the sandbox via JWP).
    pub fn attestation_key(&self) -> &AttestationKey {
        &self.attestation_key
    }

    /// Stop the sandbox gracefully.
    ///
    /// 1. Drains in-flight JWP frames
    /// 2. Signs a final energy receipt
    /// 3. Cleans up network isolation
    /// 4. Terminates the process (SIGTERM → grace → SIGKILL)
    pub fn stop(mut self, target_pid: Option<u32>) -> SandboxStopResult {
        // Stop energy enforcer and get result
        let enforcer_result = self.energy_enforcer.take().and_then(|e| e.stop());

        // Build final energy receipt
        let final_energy_uj = enforcer_result
            .as_ref()
            .map(|r| r.consumed_uj)
            .unwrap_or(0);
        let final_receipt = self.sign_receipt(
            final_energy_uj,
            (final_energy_uj as f64 * 0.000_233) as u64,
        );

        // Set up graceful shutdown with cleanup callbacks
        let instance_id = InstanceId::from_string(self.config.instance_id.clone());
        let network_enforcer = Arc::clone(&self.network_enforcer);

        let mut coordinator =
            ShutdownCoordinator::new(self.config.shutdown.clone(), Arc::clone(&self.pending));

        // Cleanup callback: remove network isolation
        let cleanup_id = instance_id.clone();
        coordinator.on_finalize(move || {
            let _ = network_enforcer.remove(&cleanup_id);
        });

        // Execute graceful shutdown
        let shutdown_result = coordinator.shutdown(target_pid);

        SandboxStopResult {
            shutdown: shutdown_result,
            enforcer: enforcer_result,
            final_receipt,
        }
    }
}

/// Combined result of stopping a sandbox.
#[derive(Debug)]
pub struct SandboxStopResult {
    /// Graceful shutdown result (drain timing, operations stats).
    pub shutdown: ShutdownResult,
    /// Energy enforcer result (total consumption, exceeded flag).
    pub enforcer: Option<EnforcerResult>,
    /// Final signed energy receipt.
    pub final_receipt: SignedEnergyReceipt,
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sandbox_config_strict() {
        let id = InstanceId::from_string("test-1".into());
        let config = SandboxConfig::strict(&id, 50_000_000);
        assert_eq!(config.instance_id, "test-1");
        assert_eq!(config.mode, RuntimeMode::Native);
        assert!(config.network.deny_all_egress);
        assert_eq!(
            config.resource_limits.energy.max_energy_uj,
            Some(50_000_000)
        );
        assert!(config.resource_limits.energy.hard_kill);
        assert!(config.resource_limits.disk.tmpfs_only);
    }

    #[test]
    fn test_sandbox_config_wasm() {
        let id = InstanceId::from_string("wasm-1".into());
        let config = SandboxConfig::wasm(&id, 10_000_000);
        assert_eq!(config.mode, RuntimeMode::WASM);
    }

    #[test]
    fn test_sandbox_config_vm() {
        let id = InstanceId::from_string("vm-1".into());
        let config = SandboxConfig::vm(&id, 100_000_000);
        assert_eq!(config.mode, RuntimeMode::VM);
    }

    #[test]
    fn test_sandbox_config_serde() {
        let id = InstanceId::from_string("serde-1".into());
        let config = SandboxConfig::strict(&id, 50_000_000);
        let json = serde_json::to_string(&config).unwrap();
        let parsed: SandboxConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.instance_id, "serde-1");
        assert_eq!(
            parsed.resource_limits.energy.max_energy_uj,
            Some(50_000_000)
        );
    }

    #[test]
    fn test_sandbox_creation() {
        let id = InstanceId::from_string("create-1".into());
        let config = SandboxConfig::strict(&id, 50_000_000);
        let sandbox = Sandbox::new(config).unwrap();

        assert_eq!(sandbox.pending().in_flight(), 0);
        assert!(sandbox.energy_state().is_some());
        assert!(sandbox.isolation_args.is_some());
    }

    #[test]
    fn test_sandbox_env_vars() {
        let id = InstanceId::from_string("env-1".into());
        let config = SandboxConfig::strict(&id, 50_000_000);
        let sandbox = Sandbox::new(config).unwrap();

        let vars = sandbox.env_vars();
        assert!(vars.contains_key("JOULE_MAX_ENERGY_UJ"));
        assert!(vars.contains_key("JOULE_ATTESTATION_KEY"));
        assert_eq!(vars.get("JOULE_MAX_ENERGY_UJ").unwrap(), "50000000");
    }

    #[test]
    fn test_sandbox_sign_and_verify() {
        let id = InstanceId::from_string("sign-1".into());
        let config = SandboxConfig::strict(&id, 50_000_000);
        let mut sandbox = Sandbox::new(config).unwrap();

        let receipt = sandbox.sign_receipt(42_000_000, 9_800);
        let mut verifier = sandbox.create_verifier();
        verifier.verify(&receipt).unwrap();
    }

    #[test]
    fn test_sandbox_container_args() {
        let id = InstanceId::from_string("args-1".into());
        let mut config = SandboxConfig::strict(&id, 50_000_000);
        config.resource_limits.cpu.max_cores = Some(2.0);
        config.resource_limits.memory.hard_limit_bytes = Some(4_294_967_296);

        let sandbox = Sandbox::new(config).unwrap();
        let args = sandbox.container_args();
        assert!(args.iter().any(|a| a.starts_with("--cpus=")));
        assert!(args.iter().any(|a| a.starts_with("--memory=")));
        assert!(args.contains(&"--read-only".to_string())); // tmpfs_only
    }

    #[test]
    fn test_sandbox_pending_operations() {
        let id = InstanceId::from_string("pending-1".into());
        let config = SandboxConfig::strict(&id, 50_000_000);
        let sandbox = Sandbox::new(config).unwrap();

        let pending = sandbox.pending();
        assert!(pending.begin());
        assert_eq!(pending.in_flight(), 1);
        pending.complete();
        assert_eq!(pending.in_flight(), 0);
    }

    #[test]
    fn test_sandbox_stop_no_pid() {
        let id = InstanceId::from_string("stop-1".into());
        let config = SandboxConfig::strict(&id, 50_000_000);
        let sandbox = Sandbox::new(config).unwrap();

        let result = sandbox.stop(None);
        assert!(result.shutdown.drained_cleanly);
        assert_eq!(result.final_receipt.instance_id, "stop-1");
    }

    #[test]
    fn test_sandbox_stop_with_energy() {
        let id = InstanceId::from_string("energy-stop-1".into());
        let config = SandboxConfig::strict(&id, 1_000_000_000); // 1000 J
        let mut sandbox = Sandbox::new(config).unwrap();

        // Start energy enforcement
        let meter = inv_energy::meter::EstimationMeter::new(15.0);
        sandbox.start_energy_enforcement(Box::new(meter)).unwrap();

        // Let it sample a couple of times
        std::thread::sleep(std::time::Duration::from_millis(250));

        let result = sandbox.stop(None);
        assert!(result.shutdown.drained_cleanly);
        assert!(result.enforcer.is_some());
        let enforcer = result.enforcer.unwrap();
        assert!(!enforcer.exceeded);
        assert!(enforcer.sample_count > 0);
    }

    #[test]
    fn test_sandbox_take_isolation_args() {
        let id = InstanceId::from_string("take-1".into());
        let config = SandboxConfig::strict(&id, 50_000_000);
        let mut sandbox = Sandbox::new(config).unwrap();

        // First take should succeed
        let args = sandbox.take_isolation_args();
        assert!(args.is_some());

        // Second take should be None
        assert!(sandbox.take_isolation_args().is_none());
    }
}
