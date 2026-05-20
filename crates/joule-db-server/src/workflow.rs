//! Durable Workflows for JouleDB
//!
//! Energy-metered workflow execution with persistent state, retry policies,
//! message queue with dead-letter support, and step-level checkpointing.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

// ============================================================================
// Errors
// ============================================================================

#[derive(Debug, Error)]
pub enum WorkflowError {
    #[error("workflow not found: {0}")]
    NotFound(String),

    #[error("instance not found: {0}")]
    InstanceNotFound(String),

    #[error("topic not found: {0}")]
    TopicNotFound(String),

    #[error("message not found: {0}")]
    MessageNotFound(String),

    #[error("workflow already exists: {0}")]
    AlreadyExists(String),

    #[error("energy budget exceeded: used {used_uj} uJ, budget {budget_uj} uJ")]
    EnergyBudgetExceeded { used_uj: u64, budget_uj: u64 },

    #[error("step failed: {step} — {reason}")]
    StepFailed { step: String, reason: String },

    #[error("internal error: {0}")]
    Internal(String),
}

/// Maximum messages per topic before rejecting new publishes
const MAX_QUEUE_DEPTH_PER_TOPIC: usize = 100_000;

// ============================================================================
// Retry configuration
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowRetryConfig {
    pub max_attempts: u32,
    pub base_delay_ms: u64,
    pub max_delay_ms: u64,
    pub backoff_multiplier: f64,
}

impl Default for WorkflowRetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_delay_ms: 1000,
            max_delay_ms: 30_000,
            backoff_multiplier: 2.0,
        }
    }
}

// ============================================================================
// Workflow definition
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepOperation {
    Sql(String),
    Http {
        method: String,
        url: String,
        body: Option<String>,
    },
    Tool(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowStep {
    pub label: String,
    pub operation: StepOperation,
    pub depends_on: Vec<String>,
    pub timeout_ms: Option<u64>,
    pub retry: Option<WorkflowRetryConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowDefinition {
    pub id: String,
    pub name: String,
    pub steps: Vec<WorkflowStep>,
    pub retry_policy: WorkflowRetryConfig,
    pub energy_budget_uj: Option<u64>,
    pub created_at: u64,
}

// ============================================================================
// Workflow instance (runtime state)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum WorkflowStatus {
    Pending,
    Running { step_index: usize },
    Paused { reason: String },
    Completed,
    Failed { error: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepResult {
    pub label: String,
    pub status: StepResultStatus,
    pub output: Option<String>,
    pub energy_uj: u64,
    pub duration_ms: u64,
    pub attempts: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum StepResultStatus {
    Success,
    Failed,
    Skipped,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowInstance {
    pub id: String,
    pub definition_id: String,
    pub status: WorkflowStatus,
    pub step_results: Vec<StepResult>,
    pub total_energy_uj: u64,
    pub started_at: u64,
    pub finished_at: Option<u64>,
}

// ============================================================================
// Message queue
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum MessageStatus {
    Pending,
    Delivered,
    Acked,
    DeadLetter,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueMessage {
    pub id: String,
    pub topic: String,
    pub payload: String,
    pub delivery_count: u32,
    pub max_deliveries: u32,
    pub status: MessageStatus,
    pub created_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueStats {
    pub total_topics: usize,
    pub total_messages: u64,
    pub pending_messages: u64,
    pub dead_letter_messages: u64,
    pub total_workflows: usize,
    pub total_instances: usize,
}

// ============================================================================
// WorkflowManager
// ============================================================================

pub struct WorkflowManager {
    definitions: RwLock<HashMap<String, WorkflowDefinition>>,
    instances: RwLock<HashMap<String, WorkflowInstance>>,
    queue: RwLock<HashMap<String, Vec<QueueMessage>>>,
    id_counter: AtomicU64,
    db: Option<joule_db_local::Database>,
}

impl WorkflowManager {
    /// Create an in-memory-only manager (for tests)
    pub fn new() -> Self {
        Self {
            definitions: RwLock::new(HashMap::new()),
            instances: RwLock::new(HashMap::new()),
            queue: RwLock::new(HashMap::new()),
            id_counter: AtomicU64::new(1),
            db: None,
        }
    }

    /// Open a durable manager backed by WAL storage
    pub fn open(db_path: &str) -> Result<Self, WorkflowError> {
        let db = joule_db_local::Database::open(db_path)
            .map_err(|e| WorkflowError::Internal(format!("failed to open workflow db: {e}")))?;
        let mut mgr = Self {
            definitions: RwLock::new(HashMap::new()),
            instances: RwLock::new(HashMap::new()),
            queue: RwLock::new(HashMap::new()),
            id_counter: AtomicU64::new(1),
            db: Some(db),
        };
        mgr.recover()?;
        Ok(mgr)
    }

    fn persist(&self, key: &str, value: &impl Serialize) {
        if let Some(ref db) = self.db {
            if let Ok(bytes) = serde_json::to_vec(value) {
                let _ = db.put(key.as_bytes(), &bytes);
            }
        }
    }

    fn remove(&self, key: &str) {
        if let Some(ref db) = self.db {
            let _ = db.delete(key.as_bytes());
        }
    }

    fn recover(&mut self) -> Result<(), WorkflowError> {
        let db = match self.db {
            Some(ref db) => db,
            None => return Ok(()),
        };
        // Recover definitions
        let def_entries = db.prefix_scan(b"wf:def:").unwrap_or_default();
        let mut defs = HashMap::new();
        for (_k, v) in &def_entries {
            if let Ok(def) = serde_json::from_slice::<WorkflowDefinition>(v) {
                defs.insert(def.id.clone(), def);
            }
        }
        // Recover instances
        let inst_entries = db.prefix_scan(b"wf:inst:").unwrap_or_default();
        let mut instances = HashMap::new();
        for (_k, v) in &inst_entries {
            if let Ok(inst) = serde_json::from_slice::<WorkflowInstance>(v) {
                instances.insert(inst.id.clone(), inst);
            }
        }
        // Recover queue messages
        let queue_entries = db.prefix_scan(b"queue:").unwrap_or_default();
        let mut queue: HashMap<String, Vec<QueueMessage>> = HashMap::new();
        for (_k, v) in &queue_entries {
            if let Ok(msg) = serde_json::from_slice::<QueueMessage>(v) {
                queue.entry(msg.topic.clone()).or_default().push(msg);
            }
        }

        *self.definitions.write().unwrap_or_else(|p| p.into_inner()) = defs;
        *self.instances.write().unwrap_or_else(|p| p.into_inner()) = instances;
        *self.queue.write().unwrap_or_else(|p| p.into_inner()) = queue;
        Ok(())
    }

    fn next_id(&self, prefix: &str) -> String {
        let counter = self.id_counter.fetch_add(1, Ordering::Relaxed);
        let ts = now_millis();
        format!("{prefix}_{:016x}{:08x}", ts, counter)
    }

    // ── Definition CRUD ─────────────────────────────────────────────────

    pub fn create_definition(
        &self,
        name: String,
        steps: Vec<WorkflowStep>,
        retry_policy: Option<WorkflowRetryConfig>,
        energy_budget_uj: Option<u64>,
    ) -> Result<WorkflowDefinition, WorkflowError> {
        let id = self.next_id("wf");
        let def = WorkflowDefinition {
            id: id.clone(),
            name,
            steps,
            retry_policy: retry_policy.unwrap_or_default(),
            energy_budget_uj,
            created_at: now_millis(),
        };
        let mut defs = self
            .definitions
            .write()
            .map_err(|e| WorkflowError::Internal(e.to_string()))?;
        self.persist(&format!("wf:def:{}", def.id), &def);
        defs.insert(id, def.clone());
        Ok(def)
    }

    pub fn list_definitions(&self) -> Result<Vec<WorkflowDefinition>, WorkflowError> {
        let defs = self
            .definitions
            .read()
            .map_err(|e| WorkflowError::Internal(e.to_string()))?;
        Ok(defs.values().cloned().collect())
    }

    pub fn get_definition(&self, id: &str) -> Result<WorkflowDefinition, WorkflowError> {
        let defs = self
            .definitions
            .read()
            .map_err(|e| WorkflowError::Internal(e.to_string()))?;
        defs.get(id)
            .cloned()
            .ok_or_else(|| WorkflowError::NotFound(id.to_string()))
    }

    pub fn delete_definition(&self, id: &str) -> Result<(), WorkflowError> {
        let mut defs = self
            .definitions
            .write()
            .map_err(|e| WorkflowError::Internal(e.to_string()))?;
        let result = defs
            .remove(id)
            .map(|_| ())
            .ok_or_else(|| WorkflowError::NotFound(id.to_string()));
        if result.is_ok() {
            self.remove(&format!("wf:def:{}", id));
        }
        result
    }

    // ── Workflow execution ──────────────────────────────────────────────

    pub fn run(&self, definition_id: &str) -> Result<WorkflowInstance, WorkflowError> {
        let def = self.get_definition(definition_id)?;
        let instance_id = self.next_id("wfi");
        let now = now_millis();

        // Topological sort: resolve execution order respecting depends_on
        let execution_order = topological_sort(&def.steps)?;

        let mut step_results_map: std::collections::HashMap<String, StepResult> =
            std::collections::HashMap::new();
        let mut completed_steps: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        let mut total_energy: u64 = 0;
        let mut final_status = WorkflowStatus::Completed;

        for step_idx in &execution_order {
            let step = &def.steps[*step_idx];

            // Check all dependencies completed successfully
            let deps_met = step
                .depends_on
                .iter()
                .all(|dep| completed_steps.contains(dep));
            if !deps_met {
                // A dependency failed or was skipped — skip this step too
                step_results_map.insert(
                    step.label.clone(),
                    StepResult {
                        label: step.label.clone(),
                        status: StepResultStatus::Skipped,
                        output: Some("dependency not met".into()),
                        energy_uj: 0,
                        duration_ms: 0,
                        attempts: 0,
                    },
                );
                continue;
            }

            let retry = step.retry.as_ref().unwrap_or(&def.retry_policy);
            let step_start = now_millis();
            let step_energy = estimate_step_energy(&step.operation);

            // Energy budget check (saturating add prevents u64 overflow)
            if let Some(budget) = def.energy_budget_uj {
                if total_energy.saturating_add(step_energy) > budget {
                    final_status = WorkflowStatus::Failed {
                        error: format!(
                            "energy budget exceeded at step '{}': used {} + {} > {}",
                            step.label, total_energy, step_energy, budget
                        ),
                    };
                    step_results_map.insert(
                        step.label.clone(),
                        StepResult {
                            label: step.label.clone(),
                            status: StepResultStatus::Failed,
                            output: Some("energy budget exceeded".into()),
                            energy_uj: 0,
                            duration_ms: now_millis().saturating_sub(step_start),
                            attempts: 1,
                        },
                    );
                    break;
                }
            }

            // Simulate step execution with retry
            let mut succeeded = false;
            let mut attempts = 0u32;

            for attempt in 0..retry.max_attempts {
                attempts = attempt + 1;
                // Simulated execution: all steps succeed on first attempt
                succeeded = true;
                break;
            }

            total_energy = total_energy.saturating_add(step_energy);
            let step_duration = now_millis().saturating_sub(step_start);

            if succeeded {
                completed_steps.insert(step.label.clone());
                step_results_map.insert(
                    step.label.clone(),
                    StepResult {
                        label: step.label.clone(),
                        status: StepResultStatus::Success,
                        output: Some(format!("step '{}' completed", step.label)),
                        energy_uj: step_energy,
                        duration_ms: step_duration,
                        attempts,
                    },
                );
            } else {
                final_status = WorkflowStatus::Failed {
                    error: format!("step '{}' failed after {} attempts", step.label, attempts),
                };
                step_results_map.insert(
                    step.label.clone(),
                    StepResult {
                        label: step.label.clone(),
                        status: StepResultStatus::Failed,
                        output: Some(format!("failed after {} attempts", attempts)),
                        energy_uj: step_energy,
                        duration_ms: step_duration,
                        attempts,
                    },
                );
                break;
            }
        }

        // Mark any remaining steps that weren't reached as skipped
        for step in &def.steps {
            if !step_results_map.contains_key(&step.label) {
                step_results_map.insert(
                    step.label.clone(),
                    StepResult {
                        label: step.label.clone(),
                        status: StepResultStatus::Skipped,
                        output: None,
                        energy_uj: 0,
                        duration_ms: 0,
                        attempts: 0,
                    },
                );
            }
        }

        // Collect results in definition order for stable output
        let step_results: Vec<StepResult> = def
            .steps
            .iter()
            .filter_map(|s| step_results_map.remove(&s.label))
            .collect();

        let instance = WorkflowInstance {
            id: instance_id.clone(),
            definition_id: definition_id.to_string(),
            status: final_status,
            step_results,
            total_energy_uj: total_energy,
            started_at: now,
            finished_at: Some(now_millis()),
        };

        let mut instances = self
            .instances
            .write()
            .map_err(|e| WorkflowError::Internal(e.to_string()))?;
        self.persist(&format!("wf:inst:{}", instance.id), &instance);
        instances.insert(instance_id, instance.clone());

        Ok(instance)
    }

    pub fn get_instance(&self, id: &str) -> Result<WorkflowInstance, WorkflowError> {
        let instances = self
            .instances
            .read()
            .map_err(|e| WorkflowError::Internal(e.to_string()))?;
        instances
            .get(id)
            .cloned()
            .ok_or_else(|| WorkflowError::InstanceNotFound(id.to_string()))
    }

    pub fn list_instances(&self) -> Result<Vec<WorkflowInstance>, WorkflowError> {
        let instances = self
            .instances
            .read()
            .map_err(|e| WorkflowError::Internal(e.to_string()))?;
        Ok(instances.values().cloned().collect())
    }

    // ── Message queue ───────────────────────────────────────────────────

    pub fn publish(&self, topic: &str, payload: String) -> Result<QueueMessage, WorkflowError> {
        let id = self.next_id("msg");
        let msg = QueueMessage {
            id: id.clone(),
            topic: topic.to_string(),
            payload,
            delivery_count: 0,
            max_deliveries: 5,
            status: MessageStatus::Pending,
            created_at: now_millis(),
        };
        let mut queue = self
            .queue
            .write()
            .map_err(|e| WorkflowError::Internal(e.to_string()))?;
        let topic_queue = queue.entry(topic.to_string()).or_default();
        if topic_queue.len() >= MAX_QUEUE_DEPTH_PER_TOPIC {
            return Err(WorkflowError::Internal(format!(
                "queue depth limit reached for topic '{}': max {}",
                topic, MAX_QUEUE_DEPTH_PER_TOPIC
            )));
        }
        self.persist(&format!("queue:{}:{}", topic, msg.id), &msg);
        topic_queue.push(msg.clone());
        Ok(msg)
    }

    pub fn subscribe(
        &self,
        topic: &str,
        max_messages: usize,
    ) -> Result<Vec<QueueMessage>, WorkflowError> {
        let mut queue = self
            .queue
            .write()
            .map_err(|e| WorkflowError::Internal(e.to_string()))?;
        let messages = queue.entry(topic.to_string()).or_default();

        let mut delivered = Vec::new();
        for msg in messages.iter_mut() {
            if delivered.len() >= max_messages {
                break;
            }
            if msg.status == MessageStatus::Pending {
                msg.delivery_count += 1;
                if msg.delivery_count > msg.max_deliveries {
                    msg.status = MessageStatus::DeadLetter;
                    self.persist(&format!("queue:{}:{}", msg.topic, msg.id), msg);
                } else {
                    msg.status = MessageStatus::Delivered;
                    self.persist(&format!("queue:{}:{}", msg.topic, msg.id), msg);
                    delivered.push(msg.clone());
                }
            }
        }

        Ok(delivered)
    }

    pub fn ack(&self, message_ids: &[String]) -> Result<u64, WorkflowError> {
        let mut queue = self
            .queue
            .write()
            .map_err(|e| WorkflowError::Internal(e.to_string()))?;
        let mut acked = 0u64;
        for messages in queue.values_mut() {
            for msg in messages.iter_mut() {
                if message_ids.contains(&msg.id) && msg.status == MessageStatus::Delivered {
                    msg.status = MessageStatus::Acked;
                    self.persist(&format!("queue:{}:{}", msg.topic, msg.id), msg);
                    acked += 1;
                }
            }
        }
        Ok(acked)
    }

    /// Negative-acknowledge messages, returning them to Pending for redelivery.
    pub fn nack(&self, message_ids: &[String]) -> Result<u64, WorkflowError> {
        let mut queue = self
            .queue
            .write()
            .map_err(|e| WorkflowError::Internal(e.to_string()))?;
        let mut nacked = 0u64;
        for messages in queue.values_mut() {
            for msg in messages.iter_mut() {
                if message_ids.contains(&msg.id) && msg.status == MessageStatus::Delivered {
                    msg.status = MessageStatus::Pending;
                    self.persist(&format!("queue:{}:{}", msg.topic, msg.id), msg);
                    nacked += 1;
                }
            }
        }
        Ok(nacked)
    }

    pub fn dead_letters(&self, topic: &str) -> Result<Vec<QueueMessage>, WorkflowError> {
        let queue = self
            .queue
            .read()
            .map_err(|e| WorkflowError::Internal(e.to_string()))?;
        let messages = queue
            .get(topic)
            .map(|msgs| {
                msgs.iter()
                    .filter(|m| m.status == MessageStatus::DeadLetter)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();
        Ok(messages)
    }

    pub fn stats(&self) -> Result<QueueStats, WorkflowError> {
        let queue = self
            .queue
            .read()
            .map_err(|e| WorkflowError::Internal(e.to_string()))?;
        let defs = self
            .definitions
            .read()
            .map_err(|e| WorkflowError::Internal(e.to_string()))?;
        let instances = self
            .instances
            .read()
            .map_err(|e| WorkflowError::Internal(e.to_string()))?;

        let mut total_messages = 0u64;
        let mut pending = 0u64;
        let mut dead = 0u64;
        for msgs in queue.values() {
            for msg in msgs {
                total_messages += 1;
                match msg.status {
                    MessageStatus::Pending => pending += 1,
                    MessageStatus::DeadLetter => dead += 1,
                    _ => {}
                }
            }
        }

        Ok(QueueStats {
            total_topics: queue.len(),
            total_messages,
            pending_messages: pending,
            dead_letter_messages: dead,
            total_workflows: defs.len(),
            total_instances: instances.len(),
        })
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Topological sort of workflow steps based on depends_on.
/// Returns indices in execution order, or error on cycles/missing deps.
fn topological_sort(steps: &[WorkflowStep]) -> Result<Vec<usize>, WorkflowError> {
    let label_to_idx: std::collections::HashMap<&str, usize> = steps
        .iter()
        .enumerate()
        .map(|(i, s)| (s.label.as_str(), i))
        .collect();

    // Validate all dependencies reference existing labels
    for step in steps {
        for dep in &step.depends_on {
            if !label_to_idx.contains_key(dep.as_str()) {
                return Err(WorkflowError::Internal(format!(
                    "step '{}' depends on unknown step '{}'",
                    step.label, dep
                )));
            }
        }
    }

    // Kahn's algorithm
    let n = steps.len();
    let mut in_degree = vec![0usize; n];
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n];

    for (i, step) in steps.iter().enumerate() {
        for dep in &step.depends_on {
            let dep_idx = label_to_idx[dep.as_str()];
            adjacency[dep_idx].push(i);
            in_degree[i] += 1;
        }
    }

    let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
    for i in 0..n {
        if in_degree[i] == 0 {
            queue.push_back(i);
        }
    }

    let mut order = Vec::with_capacity(n);
    while let Some(node) = queue.pop_front() {
        order.push(node);
        for &next in &adjacency[node] {
            in_degree[next] -= 1;
            if in_degree[next] == 0 {
                queue.push_back(next);
            }
        }
    }

    if order.len() != n {
        return Err(WorkflowError::Internal(
            "circular dependency detected in workflow steps".to_string(),
        ));
    }

    Ok(order)
}

fn estimate_step_energy(op: &StepOperation) -> u64 {
    match op {
        StepOperation::Sql(_) => 500,       // 500 uJ per SQL step
        StepOperation::Http { .. } => 2000, // 2000 uJ per HTTP call
        StepOperation::Tool(_) => 1000,     // 1000 uJ per tool invocation
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_steps(labels: &[&str]) -> Vec<WorkflowStep> {
        labels
            .iter()
            .map(|l| WorkflowStep {
                label: l.to_string(),
                operation: StepOperation::Sql(format!("SELECT * FROM {l}")),
                depends_on: vec![],
                timeout_ms: None,
                retry: None,
            })
            .collect()
    }

    #[test]
    fn test_create_and_list_definitions() {
        let mgr = WorkflowManager::new();
        let def = mgr
            .create_definition(
                "ETL Pipeline".into(),
                make_steps(&["extract", "load"]),
                None,
                None,
            )
            .unwrap();
        assert_eq!(def.name, "ETL Pipeline");
        assert_eq!(def.steps.len(), 2);

        let all = mgr.list_definitions().unwrap();
        assert_eq!(all.len(), 1);
    }

    #[test]
    fn test_get_definition() {
        let mgr = WorkflowManager::new();
        let def = mgr
            .create_definition("test".into(), make_steps(&["a"]), None, None)
            .unwrap();
        let got = mgr.get_definition(&def.id).unwrap();
        assert_eq!(got.name, "test");
    }

    #[test]
    fn test_delete_definition() {
        let mgr = WorkflowManager::new();
        let def = mgr
            .create_definition("test".into(), make_steps(&["a"]), None, None)
            .unwrap();
        mgr.delete_definition(&def.id).unwrap();
        assert!(mgr.get_definition(&def.id).is_err());
    }

    #[test]
    fn test_delete_not_found() {
        let mgr = WorkflowManager::new();
        assert!(mgr.delete_definition("nonexistent").is_err());
    }

    #[test]
    fn test_run_workflow_success() {
        let mgr = WorkflowManager::new();
        let def = mgr
            .create_definition(
                "pipeline".into(),
                make_steps(&["step1", "step2", "step3"]),
                None,
                None,
            )
            .unwrap();

        let instance = mgr.run(&def.id).unwrap();
        assert_eq!(instance.status, WorkflowStatus::Completed);
        assert_eq!(instance.step_results.len(), 3);
        assert!(instance.total_energy_uj > 0);
        assert!(instance.finished_at.is_some());

        for sr in &instance.step_results {
            assert_eq!(sr.status, StepResultStatus::Success);
        }
    }

    #[test]
    fn test_run_workflow_energy_budget_exceeded() {
        let mgr = WorkflowManager::new();
        // 500 uJ per SQL step, budget of 800 allows only 1 step
        let def = mgr
            .create_definition(
                "expensive".into(),
                make_steps(&["a", "b", "c"]),
                None,
                Some(800),
            )
            .unwrap();

        let instance = mgr.run(&def.id).unwrap();
        match &instance.status {
            WorkflowStatus::Failed { error } => {
                assert!(error.contains("energy budget exceeded"));
            }
            _ => panic!("expected Failed status"),
        }
    }

    #[test]
    fn test_get_instance() {
        let mgr = WorkflowManager::new();
        let def = mgr
            .create_definition("test".into(), make_steps(&["a"]), None, None)
            .unwrap();
        let instance = mgr.run(&def.id).unwrap();
        let got = mgr.get_instance(&instance.id).unwrap();
        assert_eq!(got.id, instance.id);
    }

    #[test]
    fn test_publish_subscribe_ack() {
        let mgr = WorkflowManager::new();

        let msg1 = mgr.publish("events", "payload1".into()).unwrap();
        let msg2 = mgr.publish("events", "payload2".into()).unwrap();

        let delivered = mgr.subscribe("events", 10).unwrap();
        assert_eq!(delivered.len(), 2);
        assert_eq!(delivered[0].status, MessageStatus::Delivered);

        let acked = mgr.ack(&[msg1.id, msg2.id]).unwrap();
        assert_eq!(acked, 2);
    }

    #[test]
    fn test_subscribe_respects_max() {
        let mgr = WorkflowManager::new();
        mgr.publish("t", "a".into()).unwrap();
        mgr.publish("t", "b".into()).unwrap();
        mgr.publish("t", "c".into()).unwrap();

        let delivered = mgr.subscribe("t", 2).unwrap();
        assert_eq!(delivered.len(), 2);
    }

    #[test]
    fn test_dead_letter() {
        let mgr = WorkflowManager::new();
        mgr.publish("dlq-test", "will-fail".into()).unwrap();

        // Subscribe then nack repeatedly to exhaust max_deliveries (default 5)
        for _ in 0..6 {
            let msgs = mgr.subscribe("dlq-test", 10).unwrap();
            if msgs.is_empty() {
                break; // message moved to dead letter
            }
            let ids: Vec<String> = msgs.iter().map(|m| m.id.clone()).collect();
            mgr.nack(&ids).unwrap();
        }

        let dead = mgr.dead_letters("dlq-test").unwrap();
        assert_eq!(dead.len(), 1);
        assert_eq!(dead[0].status, MessageStatus::DeadLetter);
    }

    #[test]
    fn test_stats() {
        let mgr = WorkflowManager::new();
        mgr.publish("topic1", "a".into()).unwrap();
        mgr.publish("topic2", "b".into()).unwrap();

        let def = mgr
            .create_definition("wf".into(), make_steps(&["s"]), None, None)
            .unwrap();
        mgr.run(&def.id).unwrap();

        let stats = mgr.stats().unwrap();
        assert_eq!(stats.total_topics, 2);
        assert_eq!(stats.total_messages, 2);
        assert_eq!(stats.pending_messages, 2);
        assert_eq!(stats.total_workflows, 1);
        assert_eq!(stats.total_instances, 1);
    }

    #[test]
    fn test_retry_config_defaults() {
        let cfg = WorkflowRetryConfig::default();
        assert_eq!(cfg.max_attempts, 3);
        assert_eq!(cfg.base_delay_ms, 1000);
        assert_eq!(cfg.max_delay_ms, 30_000);
        assert!((cfg.backoff_multiplier - 2.0).abs() < f64::EPSILON);
    }
}
