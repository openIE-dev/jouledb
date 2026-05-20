use crate::{
    AcceleratorBinding, DatabaseEngine, InstanceId, InstanceInfo, InstanceState, PortMapping,
    RuntimeConfig, RuntimeError, RuntimeMode, ServerOverrides, VolumeMount, WorkloadKind,
    accelerator::AcceleratorManager,
    agent_lifecycle::{AgentFindings, AgentPhase, AgentPool},
    agent_sandbox::{AgentSandbox, AgentSandboxConfig, AgentSandboxResult, RecallRecommendation},
    backend::{ExecOutput, RuntimeBackend},
    catalog,
    competence::DelegationLevel,
    contract::{ContractError, ContractProposal, ContractResponse, ExtensionRequest, ExtensionResponse},
    dispatch::{DispatchMesh, DispatchNode, StatusReport, TaskAction, TaskIntent, TaskOutcome},
    energy_trace::EnergySample,
    promotion::{PromotionPipeline, PromotionSummary, PromotionTarget},
    registry::InstanceRegistry,
};
use chrono::Utc;
use std::collections::HashMap;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;

/// Top-level orchestrator that manages instances across isolation modes and workload types.
pub struct RuntimeManager {
    config: RuntimeConfig,
    registry: InstanceRegistry,
    /// Backend for database and process workloads (native, VM, or WASM).
    process_backend: Box<dyn RuntimeBackendDyn>,
    /// Backend for OCI container workloads.
    container_backend: crate::container::ContainerBackend,
    /// Hardware accelerator detection, allocation, and energy tracking.
    accelerator_manager: AcceleratorManager,
    /// LLM runtime for inference telemetry and model management.
    llm_runtime: crate::llm::LlmRuntime,
    /// Pool of available agents for contract-based work.
    agent_pool: AgentPool,
    /// Active agent sandboxes (keyed by agent ID).
    active_agents: HashMap<String, AgentSandbox>,
    /// Federated dispatch mesh — nodes self-select for work by competence.
    dispatch_mesh: DispatchMesh,
    /// Promotion pipeline — converges LLM inference to deterministic execution.
    promotion_pipeline: PromotionPipeline,
}

/// Object-safe wrapper around RuntimeBackend for dynamic dispatch.
///
/// All borrowed parameters are tied to a single lifetime `'a` so that
/// the returned future can capture them safely.
pub trait RuntimeBackendDyn: Send + Sync {
    fn start_dyn<'a>(
        &'a self,
        config: &'a RuntimeConfig,
        instance: &'a InstanceInfo,
        overrides: &'a ServerOverrides,
    ) -> Pin<Box<dyn Future<Output = Result<(), RuntimeError>> + Send + 'a>>;

    fn stop_dyn<'a>(
        &'a self,
        instance_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<(), RuntimeError>> + Send + 'a>>;

    fn status_dyn<'a>(
        &'a self,
        instance_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<InstanceState, RuntimeError>> + Send + 'a>>;

    fn health_check_dyn<'a>(
        &'a self,
        instance_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<bool, RuntimeError>> + Send + 'a>>;

    fn exec_dyn<'a>(
        &'a self,
        instance_id: &'a str,
        command: &'a [String],
    ) -> Pin<Box<dyn Future<Output = Result<ExecOutput, RuntimeError>> + Send + 'a>>;

    fn logs_dyn<'a>(
        &'a self,
        instance_id: &'a str,
        tail: Option<usize>,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<String>, RuntimeError>> + Send + 'a>>;
}

/// Blanket implementation: any RuntimeBackend is also a RuntimeBackendDyn.
impl<T: RuntimeBackend> RuntimeBackendDyn for T {
    fn start_dyn<'a>(
        &'a self,
        config: &'a RuntimeConfig,
        instance: &'a InstanceInfo,
        overrides: &'a ServerOverrides,
    ) -> Pin<Box<dyn Future<Output = Result<(), RuntimeError>> + Send + 'a>> {
        Box::pin(self.start(config, instance, overrides))
    }

    fn stop_dyn<'a>(
        &'a self,
        instance_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<(), RuntimeError>> + Send + 'a>> {
        Box::pin(self.stop(instance_id))
    }

    fn status_dyn<'a>(
        &'a self,
        instance_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<InstanceState, RuntimeError>> + Send + 'a>> {
        Box::pin(self.status(instance_id))
    }

    fn health_check_dyn<'a>(
        &'a self,
        instance_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<bool, RuntimeError>> + Send + 'a>> {
        Box::pin(self.health_check(instance_id))
    }

    fn exec_dyn<'a>(
        &'a self,
        instance_id: &'a str,
        command: &'a [String],
    ) -> Pin<Box<dyn Future<Output = Result<ExecOutput, RuntimeError>> + Send + 'a>> {
        Box::pin(self.exec(instance_id, command))
    }

    fn logs_dyn<'a>(
        &'a self,
        instance_id: &'a str,
        tail: Option<usize>,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<String>, RuntimeError>> + Send + 'a>> {
        Box::pin(self.logs(instance_id, tail))
    }
}

impl RuntimeManager {
    /// Create a new RuntimeManager, selecting the process backend based on config.mode.
    pub fn new(config: RuntimeConfig, data_dir: PathBuf) -> Result<Self, RuntimeError> {
        config.validate()?;

        let registry = InstanceRegistry::with_persistence(&data_dir)?;
        let process_backend: Box<dyn RuntimeBackendDyn> = match config.mode {
            RuntimeMode::Native => Box::new(crate::native::NativeBackend::new()),
            RuntimeMode::VM => {
                #[cfg(feature = "vm-isolation")]
                {
                    Box::new(crate::vm_backend::VmBackend::new())
                }
                #[cfg(not(feature = "vm-isolation"))]
                {
                    return Err(RuntimeError::UnsupportedMode(RuntimeMode::VM));
                }
            }
            RuntimeMode::WASM => {
                #[cfg(feature = "wasm-isolation")]
                {
                    Box::new(crate::wasm_backend::WasmBackend::new()?)
                }
                #[cfg(not(feature = "wasm-isolation"))]
                {
                    return Err(RuntimeError::UnsupportedMode(RuntimeMode::WASM));
                }
            }
        };

        let image_dir = data_dir.parent().unwrap_or(&data_dir).join("images");
        let container_backend = crate::container::ContainerBackend::new(image_dir);
        let accelerator_manager = AcceleratorManager::new();
        let model_cache = data_dir.parent().unwrap_or(&data_dir).join("models");
        let llm_runtime = crate::llm::LlmRuntime::new(model_cache);

        Ok(Self {
            config,
            registry,
            process_backend,
            container_backend,
            accelerator_manager,
            llm_runtime,
            agent_pool: AgentPool::new(),
            active_agents: HashMap::new(),
            dispatch_mesh: DispatchMesh::new(),
            promotion_pipeline: PromotionPipeline::new(),
        })
    }

    /// Dispatch to the appropriate backend based on workload type.
    fn backend_for(&self, workload: &WorkloadKind) -> &dyn RuntimeBackendDyn {
        match workload {
            WorkloadKind::Container { .. } => &self.container_backend,
            _ => &*self.process_backend,
        }
    }

    /// Start a new database instance (backward-compatible entry point).
    pub async fn start_instance(
        &self,
        name: String,
        engine: DatabaseEngine,
        overrides: ServerOverrides,
    ) -> Result<InstanceId, RuntimeError> {
        let workload = WorkloadKind::database(engine.clone());
        self.start_workload(
            name,
            workload,
            overrides,
            vec![],
            vec![],
            HashMap::new(),
            HashMap::new(),
        )
        .await
    }

    /// Start any workload type — the generalized entry point.
    pub async fn start_workload(
        &self,
        name: String,
        workload: WorkloadKind,
        overrides: ServerOverrides,
        accelerators: Vec<AcceleratorBinding>,
        volumes: Vec<VolumeMount>,
        env_vars: HashMap<String, String>,
        labels: HashMap<String, String>,
    ) -> Result<InstanceId, RuntimeError> {
        let id = InstanceId::new();

        // Derive engine + spec from workload
        let engine = match &workload {
            WorkloadKind::Database { engine } => engine.clone(),
            _ => DatabaseEngine::default(), // placeholder for non-database workloads
        };
        let spec = catalog::get_workload_spec(&workload);

        let data_dir = overrides
            .data_dir
            .clone()
            .unwrap_or_else(|| format!("./{}-data-{}", workload.display_name(), &id.0[..8]));

        let ports = match &workload {
            WorkloadKind::Database { engine } => build_port_mappings_for_engine(engine, &overrides),
            _ => {
                // For processes/containers, use engine_port if specified
                if let Some(port) = overrides.engine_port {
                    vec![PortMapping {
                        protocol: spec.protocol_name.into(),
                        host_port: port,
                        instance_port: port,
                    }]
                } else {
                    vec![]
                }
            }
        };

        // Energy sidecar port: auto-assign for non-JouleDB workloads
        let energy_port = match &workload {
            WorkloadKind::Database { engine } if *engine == DatabaseEngine::JouleDB => None,
            _ => {
                let base = overrides.engine_port.unwrap_or(spec.default_port);
                if base > 0 { Some(base + 10000) } else { None }
            }
        };

        let mut info = InstanceInfo {
            id: id.clone(),
            name,
            workload: workload.clone(),
            engine,
            mode: self.config.mode,
            state: InstanceState::Starting,
            created_at: Utc::now(),
            pid: None,
            ports,
            data_dir,
            node_id: None,
            energy_port,
            accelerators,
            volumes,
            env_vars,
            labels,
        };

        // Allocate hardware accelerators if requested
        if !info.accelerators.is_empty() {
            match self
                .accelerator_manager
                .allocate(info.id.as_str(), &info.accelerators)
            {
                Ok(_device_ids) => {
                    // Inject accelerator env vars into the instance
                    let accel_env = self
                        .accelerator_manager
                        .env_vars_for_instance(info.id.as_str());
                    for (key, value) in accel_env {
                        info.env_vars.entry(key).or_insert(value);
                    }
                }
                Err(e) => {
                    return Err(e);
                }
            }
        }

        self.registry.register(info.clone())?;

        let backend = self.backend_for(&workload);
        match backend.start_dyn(&self.config, &info, &overrides).await {
            Ok(()) => {
                self.registry
                    .update_state(id.as_str(), InstanceState::Running)?;
                Ok(id)
            }
            Err(e) => {
                self.registry
                    .update_state(id.as_str(), InstanceState::Failed(e.to_string()))?;
                Err(e)
            }
        }
    }

    /// Stop a running instance.
    pub async fn stop_instance(&self, instance_id: &str) -> Result<(), RuntimeError> {
        let workload = self
            .registry
            .get(instance_id)
            .map(|i| i.workload.clone())
            .unwrap_or_default();

        self.registry
            .update_state(instance_id, InstanceState::Stopping)?;

        let backend = self.backend_for(&workload);
        match backend.stop_dyn(instance_id).await {
            Ok(()) => {
                self.accelerator_manager.release(instance_id);
                self.registry.deregister(instance_id)?;
                Ok(())
            }
            Err(e) => {
                self.accelerator_manager.release(instance_id);
                self.registry
                    .update_state(instance_id, InstanceState::Failed(e.to_string()))?;
                Err(e)
            }
        }
    }

    /// List all registered instances.
    pub fn list_instances(&self) -> Vec<InstanceInfo> {
        self.registry.list()
    }

    /// Get info about a specific instance.
    pub fn get_instance(&self, instance_id: &str) -> Option<InstanceInfo> {
        self.registry.get(instance_id)
    }

    /// Check if a specific instance is healthy.
    pub async fn health_check(&self, instance_id: &str) -> Result<bool, RuntimeError> {
        let workload = self
            .registry
            .get(instance_id)
            .map(|i| i.workload.clone())
            .unwrap_or_default();
        self.backend_for(&workload)
            .health_check_dyn(instance_id)
            .await
    }

    /// Execute a command inside a running instance.
    pub async fn exec(
        &self,
        instance_id: &str,
        command: &[String],
    ) -> Result<ExecOutput, RuntimeError> {
        let workload = self
            .registry
            .get(instance_id)
            .map(|i| i.workload.clone())
            .unwrap_or_default();
        self.backend_for(&workload)
            .exec_dyn(instance_id, command)
            .await
    }

    /// Retrieve logs from an instance.
    pub async fn logs(
        &self,
        instance_id: &str,
        tail: Option<usize>,
    ) -> Result<Vec<String>, RuntimeError> {
        let workload = self
            .registry
            .get(instance_id)
            .map(|i| i.workload.clone())
            .unwrap_or_default();
        self.backend_for(&workload)
            .logs_dyn(instance_id, tail)
            .await
    }

    /// Get the current runtime mode.
    pub fn mode(&self) -> RuntimeMode {
        self.config.mode
    }

    /// Get the current config.
    pub fn config(&self) -> &RuntimeConfig {
        &self.config
    }

    /// Get a reference to the container backend's image store.
    pub fn image_store(&self) -> &crate::image::ImageStore {
        self.container_backend.image_store()
    }

    /// Get a reference to the container backend's network manager.
    pub fn network_manager(&self) -> &crate::networking::NetworkManager {
        self.container_backend.network_manager()
    }

    /// Get a reference to the accelerator manager.
    pub fn accelerator_manager(&self) -> &AcceleratorManager {
        &self.accelerator_manager
    }

    /// Get a reference to the LLM runtime.
    pub fn llm_runtime(&self) -> &crate::llm::LlmRuntime {
        &self.llm_runtime
    }

    // --- Agent orchestration ---

    /// Register an agent in the pool, making it available for enlistment.
    pub fn register_agent(&mut self, agent_id: String) {
        self.agent_pool.register(agent_id);
    }

    /// Get a reference to the agent pool.
    pub fn agent_pool(&self) -> &AgentPool {
        &self.agent_pool
    }

    /// Propose a contract to an agent, creating its sandbox.
    ///
    /// Selects the first available (Pooled) agent, creates an `AgentSandbox`,
    /// and delivers the proposal. Returns the agent ID on success.
    pub fn enlist_agent(
        &mut self,
        proposal: ContractProposal,
        sandbox_config: AgentSandboxConfig,
    ) -> Result<String, RuntimeError> {
        // Find an available agent from the pool
        let available = self.agent_pool.agents_in_phase(AgentPhase::Pooled);
        let agent_id = available
            .into_iter()
            .next()
            .ok_or_else(|| RuntimeError::Internal("no agents available in pool".into()))?;

        // Create the sandbox
        let mut sandbox = AgentSandbox::new(agent_id.clone(), sandbox_config)?;

        // Deliver the proposal
        sandbox
            .propose_contract(proposal)
            .map_err(|e| RuntimeError::Internal(format!("contract proposal failed: {e}")))?;

        // Track the active sandbox
        self.active_agents.insert(agent_id.clone(), sandbox);

        Ok(agent_id)
    }

    /// Process an agent's response to a contract proposal.
    ///
    /// On Accept: the agent moves to Engaged phase.
    /// On Reject: the sandbox is removed and the agent returns to pool.
    /// On CounterPropose: the agent stays in Negotiating (host can revise and re-propose).
    pub fn process_agent_response(
        &mut self,
        agent_id: &str,
        response: ContractResponse,
    ) -> Result<AgentPhase, RuntimeError> {
        let sandbox = self
            .active_agents
            .get_mut(agent_id)
            .ok_or_else(|| RuntimeError::Internal(format!("no active sandbox for agent {agent_id}")))?;

        let phase = sandbox
            .process_response(response)
            .map_err(|e| RuntimeError::Internal(format!("contract response failed: {e}")))?;

        // If rejected, clean up
        if phase == AgentPhase::Pooled {
            self.active_agents.remove(agent_id);
        }

        Ok(phase)
    }

    /// Feed an energy sample to an active agent's trace analyzer.
    ///
    /// Returns a `RecallRecommendation` if the agent shows anomalous behavior
    /// (Warburg effect detected).
    pub fn feed_agent_energy(
        &mut self,
        agent_id: &str,
        sample: EnergySample,
    ) -> Result<Option<RecallRecommendation>, RuntimeError> {
        let sandbox = self
            .active_agents
            .get_mut(agent_id)
            .ok_or_else(|| RuntimeError::Internal(format!("no active sandbox for agent {agent_id}")))?;

        Ok(sandbox.feed_energy_sample(sample))
    }

    /// Process an extension request from an engaged agent.
    pub fn process_agent_extension(
        &mut self,
        agent_id: &str,
        request: &ExtensionRequest,
    ) -> Result<ExtensionResponse, RuntimeError> {
        let sandbox = self
            .active_agents
            .get_mut(agent_id)
            .ok_or_else(|| RuntimeError::Internal(format!("no active sandbox for agent {agent_id}")))?;

        sandbox
            .process_extension_request(request)
            .map_err(|e| RuntimeError::Internal(format!("extension request failed: {e}")))
    }

    /// Agent voluntarily returns with findings, closing its sandbox.
    pub fn agent_voluntary_return(
        &mut self,
        agent_id: &str,
        findings: AgentFindings,
        target_pid: Option<u32>,
    ) -> Result<AgentSandboxResult, RuntimeError> {
        let sandbox = self
            .active_agents
            .remove(agent_id)
            .ok_or_else(|| RuntimeError::Internal(format!("no active sandbox for agent {agent_id}")))?;

        sandbox
            .voluntary_return(findings, target_pid)
            .map_err(|e| RuntimeError::Internal(format!("voluntary return failed: {e}")))
    }

    /// Force-recall an agent, closing its sandbox.
    pub fn recall_agent(
        &mut self,
        agent_id: &str,
        reason: &str,
        target_pid: Option<u32>,
    ) -> Result<AgentSandboxResult, RuntimeError> {
        let sandbox = self
            .active_agents
            .remove(agent_id)
            .ok_or_else(|| RuntimeError::Internal(format!("no active sandbox for agent {agent_id}")))?;

        Ok(sandbox.recall(reason, target_pid))
    }

    /// List all agents currently engaged (working under contract).
    pub fn engaged_agents(&self) -> Vec<&str> {
        self.active_agents
            .iter()
            .filter(|(_, s)| s.agent_phase() == AgentPhase::Engaged)
            .map(|(id, _)| id.as_str())
            .collect()
    }

    /// Get the phase of a specific agent.
    pub fn agent_phase(&self, agent_id: &str) -> Option<AgentPhase> {
        // Check active sandboxes first
        if let Some(sandbox) = self.active_agents.get(agent_id) {
            return Some(sandbox.agent_phase());
        }
        // Check the pool
        self.agent_pool
            .get(agent_id)
            .and_then(|a| a.lock().ok().map(|a| a.phase()))
    }

    /// Get the energy consumed by an active agent.
    pub fn agent_energy_consumed(&self, agent_id: &str) -> Option<u64> {
        self.active_agents
            .get(agent_id)
            .map(|s| s.energy_consumed_uj())
    }

    /// Number of active agent sandboxes.
    pub fn active_agent_count(&self) -> usize {
        self.active_agents.len()
    }

    // --- Federated dispatch (§4) ---

    /// Register a dispatch node in the mesh.
    pub fn register_dispatch_node(&mut self, node: DispatchNode) {
        self.dispatch_mesh.register_node(node);
    }

    /// Get a reference to the dispatch mesh.
    pub fn dispatch_mesh(&self) -> &DispatchMesh {
        &self.dispatch_mesh
    }

    /// Get a mutable reference to the dispatch mesh.
    pub fn dispatch_mesh_mut(&mut self) -> &mut DispatchMesh {
        &mut self.dispatch_mesh
    }

    /// Dispatch a task via federated self-selection.
    ///
    /// Returns the node ID that self-selected for the task, or None if
    /// no node is competent (task should escalate to LLM fallback).
    pub fn dispatch_task(&mut self, intent: &TaskIntent) -> Option<String> {
        // A5: try deterministic resolution first
        if self.promotion_pipeline.try_resolve(&intent.purpose).is_some() {
            // Task resolved deterministically — no node dispatch needed.
            // The caller should use the promoted resolution directly.
            return None;
        }

        self.dispatch_mesh.dispatch(intent)
    }

    /// Submit a status report from a node to the observer.
    ///
    /// Updates the node's competence ledger and records the report.
    pub fn submit_status_report(&mut self, report: StatusReport) {
        // If the task was acted on successfully, record as promotion candidate
        if report.action == TaskAction::Acted && report.outcome == TaskOutcome::Success {
            // The report itself is a signal that this domain pattern can
            // potentially be promoted — but we need the LLM trigger/resolution
            // pair for that. The promotion pipeline is fed separately via
            // record_llm_result().
        }

        self.dispatch_mesh.report(report);
    }

    // --- Promotion pipeline (§5) ---

    /// Try to resolve a request deterministically (skip LLM).
    ///
    /// Returns the deterministic resolution if a promoted path exists.
    pub fn try_deterministic_resolve(&mut self, trigger: &str) -> Option<&str> {
        self.promotion_pipeline.try_resolve(trigger)
    }

    /// Record an LLM invocation result as a promotion candidate (A5 obligation).
    ///
    /// Every validated LLM result MUST be recorded here. The pipeline
    /// auto-promotes after sufficient validated occurrences.
    pub fn record_llm_result(
        &mut self,
        trigger: &str,
        resolution: &str,
        domain: &str,
        target: PromotionTarget,
        validated: bool,
        llm_energy_uj: u64,
    ) -> Option<String> {
        self.promotion_pipeline
            .record_llm_result(trigger, resolution, domain, target, validated, llm_energy_uj)
    }

    /// Get the promotion pipeline summary.
    pub fn promotion_summary(&self) -> PromotionSummary {
        self.promotion_pipeline.summary()
    }

    /// Get the promotion rate P(t).
    pub fn promotion_rate(&self) -> f64 {
        self.promotion_pipeline.promotion_rate()
    }

    /// Get the dispatch mesh promotion rate (fraction of tasks handled
    /// without LLM escalation).
    pub fn dispatch_promotion_rate(&self) -> f64 {
        self.dispatch_mesh.promotion_rate()
    }
}

/// Build port mappings based on the database engine.
///
/// JouleDB gets its full multi-protocol mapping (http, tcp, pgwire, raft).
/// Other engines get a single mapping using their native protocol.
fn build_port_mappings_for_engine(
    engine: &DatabaseEngine,
    overrides: &ServerOverrides,
) -> Vec<PortMapping> {
    match engine {
        DatabaseEngine::JouleDB => {
            let mut ports = Vec::new();
            if let Some(port) = overrides.http_port {
                ports.push(PortMapping {
                    protocol: "http".into(),
                    host_port: port,
                    instance_port: port,
                });
            }
            if let Some(port) = overrides.tcp_port {
                ports.push(PortMapping {
                    protocol: "tcp".into(),
                    host_port: port,
                    instance_port: port,
                });
            }
            if let Some(port) = overrides.pgwire_port {
                ports.push(PortMapping {
                    protocol: "pgwire".into(),
                    host_port: port,
                    instance_port: port,
                });
            }
            if let Some(port) = overrides.raft_port {
                ports.push(PortMapping {
                    protocol: "raft".into(),
                    host_port: port,
                    instance_port: port,
                });
            }
            ports
        }
        other => {
            let spec = catalog::get_spec(other);
            let port = overrides.engine_port.unwrap_or(spec.default_port);
            if port > 0 {
                vec![PortMapping {
                    protocol: spec.protocol_name.into(),
                    host_port: port,
                    instance_port: port,
                }]
            } else {
                vec![]
            }
        }
    }
}
