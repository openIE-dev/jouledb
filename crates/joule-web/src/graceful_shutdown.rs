//! Graceful shutdown orchestration.
//!
//! Provides a cooperative shutdown framework: signal propagation, drain phase with
//! timeout, task tracking, ordered component teardown, and health reporting during
//! drain. Pure Rust — no OS signals or async runtime dependencies in core logic.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

// ── Shutdown phase ──────────────────────────────────────────────

/// Current phase of the shutdown lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShutdownPhase {
    /// Normal operation — accepting new work.
    Running,
    /// Drain phase — rejecting new work, waiting for in-flight tasks.
    Draining,
    /// All tasks drained or timeout expired — tearing down components.
    TearingDown,
    /// Shutdown complete.
    Terminated,
}

impl fmt::Display for ShutdownPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShutdownPhase::Running => write!(f, "running"),
            ShutdownPhase::Draining => write!(f, "draining"),
            ShutdownPhase::TearingDown => write!(f, "tearing_down"),
            ShutdownPhase::Terminated => write!(f, "terminated"),
        }
    }
}

// ── Shutdown reason ─────────────────────────────────────────────

/// Why the shutdown was initiated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShutdownReason {
    /// Received a termination signal (e.g. SIGTERM).
    Signal(String),
    /// Administrative command.
    Admin(String),
    /// Internal error forced shutdown.
    Error(String),
    /// Programmatic shutdown request.
    Requested,
}

impl fmt::Display for ShutdownReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShutdownReason::Signal(s) => write!(f, "signal:{s}"),
            ShutdownReason::Admin(s) => write!(f, "admin:{s}"),
            ShutdownReason::Error(s) => write!(f, "error:{s}"),
            ShutdownReason::Requested => write!(f, "requested"),
        }
    }
}

// ── Task tracking ───────────────────────────────────────────────

/// Identifies an in-flight task.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TaskId(pub String);

/// Metadata for a tracked task.
#[derive(Debug, Clone)]
pub struct TrackedTask {
    pub id: TaskId,
    pub component: String,
    pub start_ms: u64,
    pub description: String,
}

/// Tracks in-flight tasks for drain coordination.
#[derive(Debug, Clone)]
pub struct TaskTracker {
    tasks: HashMap<String, TrackedTask>,
    next_id: u64,
}

impl TaskTracker {
    pub fn new() -> Self {
        Self { tasks: HashMap::new(), next_id: 0 }
    }

    /// Register a new task, returning its ID.
    pub fn register(&mut self, component: &str, description: &str, start_ms: u64) -> TaskId {
        let id_str = format!("task-{}", self.next_id);
        self.next_id += 1;
        let tid = TaskId(id_str.clone());
        self.tasks.insert(id_str, TrackedTask {
            id: tid.clone(),
            component: component.to_string(),
            start_ms,
            description: description.to_string(),
        });
        tid
    }

    /// Mark a task as complete.
    pub fn complete(&mut self, id: &TaskId) -> bool {
        self.tasks.remove(&id.0).is_some()
    }

    /// Number of in-flight tasks.
    pub fn count(&self) -> usize {
        self.tasks.len()
    }

    /// All in-flight tasks.
    pub fn in_flight(&self) -> Vec<&TrackedTask> {
        let mut v: Vec<&TrackedTask> = self.tasks.values().collect();
        v.sort_by_key(|t| &t.id.0);
        v
    }

    /// In-flight tasks for a specific component.
    pub fn in_flight_for(&self, component: &str) -> Vec<&TrackedTask> {
        let mut v: Vec<&TrackedTask> = self.tasks.values()
            .filter(|t| t.component == component)
            .collect();
        v.sort_by_key(|t| &t.id.0);
        v
    }

    /// True when all tasks have been drained.
    pub fn is_drained(&self) -> bool {
        self.tasks.is_empty()
    }
}

impl Default for TaskTracker {
    fn default() -> Self { Self::new() }
}

// ── Component teardown ──────────────────────────────────────────

/// A component that participates in ordered teardown.
#[derive(Debug, Clone)]
pub struct ShutdownComponent {
    pub name: String,
    /// Lower priority tears down first. Components with the same priority
    /// tear down in registration order.
    pub priority: u32,
    pub status: ComponentStatus,
}

/// Status of a component during shutdown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComponentStatus {
    Running,
    Draining,
    ShutDown,
    Failed,
}

impl fmt::Display for ComponentStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ComponentStatus::Running => write!(f, "running"),
            ComponentStatus::Draining => write!(f, "draining"),
            ComponentStatus::ShutDown => write!(f, "shut_down"),
            ComponentStatus::Failed => write!(f, "failed"),
        }
    }
}

// ── Shutdown hook ───────────────────────────────────────────────

/// Outcome of a shutdown hook execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HookResult {
    Ok,
    Failed(String),
    Skipped(String),
}

/// A named hook to run during shutdown.
#[derive(Clone)]
pub struct ShutdownHook {
    pub name: String,
    pub priority: u32,
    action: Arc<dyn Fn() -> HookResult + Send + Sync>,
}

impl fmt::Debug for ShutdownHook {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ShutdownHook")
            .field("name", &self.name)
            .field("priority", &self.priority)
            .finish()
    }
}

impl ShutdownHook {
    pub fn new(name: &str, priority: u32, action: impl Fn() -> HookResult + Send + Sync + 'static) -> Self {
        Self { name: name.to_string(), priority, action: Arc::new(action) }
    }

    pub fn run(&self) -> HookResult {
        (self.action)()
    }
}

// ── Health during drain ─────────────────────────────────────────

/// Health status reported during drain phase.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DrainHealth {
    pub phase: ShutdownPhase,
    pub in_flight_tasks: usize,
    pub components_remaining: usize,
    pub accepting_traffic: bool,
    pub ready: bool,
}

// ── Shutdown coordinator ────────────────────────────────────────

/// Central orchestrator for graceful shutdown.
#[derive(Debug)]
pub struct ShutdownCoordinator {
    inner: Arc<Mutex<CoordinatorInner>>,
}

#[derive(Debug)]
struct CoordinatorInner {
    phase: ShutdownPhase,
    reason: Option<ShutdownReason>,
    drain_timeout_ms: u64,
    drain_started_ms: Option<u64>,
    tracker: TaskTracker,
    components: Vec<ShutdownComponent>,
    hooks: Vec<ShutdownHookEntry>,
    hook_results: Vec<(String, HookResult)>,
}

#[derive(Debug)]
struct ShutdownHookEntry {
    name: String,
    priority: u32,
    hook: ShutdownHook,
}

impl ShutdownCoordinator {
    /// Create a new coordinator with the given drain timeout in milliseconds.
    pub fn new(drain_timeout_ms: u64) -> Self {
        Self {
            inner: Arc::new(Mutex::new(CoordinatorInner {
                phase: ShutdownPhase::Running,
                reason: None,
                drain_timeout_ms,
                drain_started_ms: None,
                tracker: TaskTracker::new(),
                components: Vec::new(),
                hooks: Vec::new(),
                hook_results: Vec::new(),
            })),
        }
    }

    /// Register a component for ordered teardown.
    pub fn register_component(&self, name: &str, priority: u32) {
        let mut inner = self.inner.lock().unwrap();
        inner.components.push(ShutdownComponent {
            name: name.to_string(),
            priority,
            status: ComponentStatus::Running,
        });
    }

    /// Register a shutdown hook.
    pub fn register_hook(&self, hook: ShutdownHook) {
        let mut inner = self.inner.lock().unwrap();
        let entry = ShutdownHookEntry {
            name: hook.name.clone(),
            priority: hook.priority,
            hook,
        };
        inner.hooks.push(entry);
    }

    /// Register an in-flight task. Returns None if not in Running or Draining phase.
    pub fn register_task(&self, component: &str, description: &str, start_ms: u64) -> Option<TaskId> {
        let mut inner = self.inner.lock().unwrap();
        match inner.phase {
            ShutdownPhase::Running => Some(inner.tracker.register(component, description, start_ms)),
            _ => None,
        }
    }

    /// Mark a task as complete.
    pub fn complete_task(&self, id: &TaskId) -> bool {
        let mut inner = self.inner.lock().unwrap();
        inner.tracker.complete(id)
    }

    /// Initiate shutdown — transitions to Draining phase.
    pub fn initiate(&self, reason: ShutdownReason, now_ms: u64) -> bool {
        let mut inner = self.inner.lock().unwrap();
        if inner.phase != ShutdownPhase::Running {
            return false;
        }
        inner.phase = ShutdownPhase::Draining;
        inner.reason = Some(reason);
        inner.drain_started_ms = Some(now_ms);
        for comp in &mut inner.components {
            comp.status = ComponentStatus::Draining;
        }
        true
    }

    /// Check whether the drain timeout has expired.
    pub fn is_drain_timeout(&self, now_ms: u64) -> bool {
        let inner = self.inner.lock().unwrap();
        if let Some(started) = inner.drain_started_ms {
            now_ms.saturating_sub(started) >= inner.drain_timeout_ms
        } else {
            false
        }
    }

    /// Try to advance from Draining to TearingDown.
    /// Returns true if the transition happened (all drained or timeout).
    pub fn try_advance_to_teardown(&self, now_ms: u64) -> bool {
        let mut inner = self.inner.lock().unwrap();
        if inner.phase != ShutdownPhase::Draining {
            return false;
        }
        let drained = inner.tracker.is_drained();
        let timed_out = inner.drain_started_ms
            .map(|s| now_ms.saturating_sub(s) >= inner.drain_timeout_ms)
            .unwrap_or(false);
        if drained || timed_out {
            inner.phase = ShutdownPhase::TearingDown;
            true
        } else {
            false
        }
    }

    /// Execute teardown: run hooks in priority order, mark components shut down.
    /// Returns the list of hook results.
    pub fn execute_teardown(&self) -> Vec<(String, HookResult)> {
        let mut inner = self.inner.lock().unwrap();
        if inner.phase != ShutdownPhase::TearingDown {
            return Vec::new();
        }

        // Sort hooks by priority (lower first).
        inner.hooks.sort_by_key(|h| h.priority);
        let mut results = Vec::new();
        for entry in &inner.hooks {
            let result = entry.hook.run();
            results.push((entry.name.clone(), result));
        }

        // Sort components by priority (lower first) and mark shut down.
        inner.components.sort_by_key(|c| c.priority);
        for comp in &mut inner.components {
            comp.status = ComponentStatus::ShutDown;
        }

        inner.hook_results = results.clone();
        inner.phase = ShutdownPhase::Terminated;
        results
    }

    /// Current phase.
    pub fn phase(&self) -> ShutdownPhase {
        self.inner.lock().unwrap().phase
    }

    /// Shutdown reason, if initiated.
    pub fn reason(&self) -> Option<ShutdownReason> {
        self.inner.lock().unwrap().reason.clone()
    }

    /// In-flight task count.
    pub fn in_flight_count(&self) -> usize {
        self.inner.lock().unwrap().tracker.count()
    }

    /// Health report for load balancer probes during drain.
    pub fn drain_health(&self) -> DrainHealth {
        let inner = self.inner.lock().unwrap();
        let accepting = inner.phase == ShutdownPhase::Running;
        let ready = inner.phase == ShutdownPhase::Running;
        DrainHealth {
            phase: inner.phase,
            in_flight_tasks: inner.tracker.count(),
            components_remaining: inner.components.iter()
                .filter(|c| c.status != ComponentStatus::ShutDown)
                .count(),
            accepting_traffic: accepting,
            ready,
        }
    }

    /// Component statuses.
    pub fn component_statuses(&self) -> Vec<(String, ComponentStatus)> {
        let inner = self.inner.lock().unwrap();
        inner.components.iter()
            .map(|c| (c.name.clone(), c.status))
            .collect()
    }

    /// Mark a specific component as failed during teardown.
    pub fn mark_component_failed(&self, name: &str) -> bool {
        let mut inner = self.inner.lock().unwrap();
        for comp in &mut inner.components {
            if comp.name == name {
                comp.status = ComponentStatus::Failed;
                return true;
            }
        }
        false
    }

    /// Get hook execution results (available after teardown).
    pub fn hook_results(&self) -> Vec<(String, HookResult)> {
        self.inner.lock().unwrap().hook_results.clone()
    }
}

impl Clone for ShutdownCoordinator {
    fn clone(&self) -> Self {
        Self { inner: Arc::clone(&self.inner) }
    }
}

// ── Shutdown signal ─────────────────────────────────────────────

/// A lightweight signal that can be checked by multiple listeners.
#[derive(Debug, Clone)]
pub struct ShutdownSignal {
    triggered: Arc<Mutex<bool>>,
    reason: Arc<Mutex<Option<ShutdownReason>>>,
}

impl ShutdownSignal {
    pub fn new() -> Self {
        Self {
            triggered: Arc::new(Mutex::new(false)),
            reason: Arc::new(Mutex::new(None)),
        }
    }

    /// Trigger the signal.
    pub fn trigger(&self, reason: ShutdownReason) {
        *self.triggered.lock().unwrap() = true;
        *self.reason.lock().unwrap() = Some(reason);
    }

    /// Check if triggered.
    pub fn is_triggered(&self) -> bool {
        *self.triggered.lock().unwrap()
    }

    /// Get the reason if triggered.
    pub fn reason(&self) -> Option<ShutdownReason> {
        self.reason.lock().unwrap().clone()
    }
}

impl Default for ShutdownSignal {
    fn default() -> Self { Self::new() }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_display() {
        assert_eq!(ShutdownPhase::Running.to_string(), "running");
        assert_eq!(ShutdownPhase::Draining.to_string(), "draining");
        assert_eq!(ShutdownPhase::TearingDown.to_string(), "tearing_down");
        assert_eq!(ShutdownPhase::Terminated.to_string(), "terminated");
    }

    #[test]
    fn test_reason_display() {
        assert_eq!(ShutdownReason::Signal("SIGTERM".into()).to_string(), "signal:SIGTERM");
        assert_eq!(ShutdownReason::Admin("deploy".into()).to_string(), "admin:deploy");
        assert_eq!(ShutdownReason::Error("oom".into()).to_string(), "error:oom");
        assert_eq!(ShutdownReason::Requested.to_string(), "requested");
    }

    #[test]
    fn test_task_tracker_register_complete() {
        let mut tracker = TaskTracker::new();
        assert!(tracker.is_drained());
        assert_eq!(tracker.count(), 0);

        let t1 = tracker.register("http", "GET /api", 100);
        let t2 = tracker.register("http", "POST /data", 200);
        let t3 = tracker.register("grpc", "ListItems", 300);

        assert_eq!(tracker.count(), 3);
        assert!(!tracker.is_drained());

        assert_eq!(tracker.in_flight_for("http").len(), 2);
        assert_eq!(tracker.in_flight_for("grpc").len(), 1);
        assert_eq!(tracker.in_flight_for("ws").len(), 0);

        assert!(tracker.complete(&t1));
        assert_eq!(tracker.count(), 2);

        // Double complete returns false
        assert!(!tracker.complete(&t1));

        assert!(tracker.complete(&t2));
        assert!(tracker.complete(&t3));
        assert!(tracker.is_drained());
    }

    #[test]
    fn test_task_tracker_in_flight() {
        let mut tracker = TaskTracker::new();
        tracker.register("a", "task-a", 10);
        tracker.register("b", "task-b", 20);

        let tasks = tracker.in_flight();
        assert_eq!(tasks.len(), 2);
        // Sorted by id
        assert_eq!(tasks[0].component, "a");
        assert_eq!(tasks[1].component, "b");
    }

    #[test]
    fn test_coordinator_lifecycle() {
        let coord = ShutdownCoordinator::new(5000);
        assert_eq!(coord.phase(), ShutdownPhase::Running);

        coord.register_component("http", 10);
        coord.register_component("db", 20);

        let t1 = coord.register_task("http", "req1", 100).unwrap();
        let t2 = coord.register_task("http", "req2", 200).unwrap();
        assert_eq!(coord.in_flight_count(), 2);

        // Initiate shutdown
        assert!(coord.initiate(ShutdownReason::Signal("SIGTERM".into()), 1000));
        assert_eq!(coord.phase(), ShutdownPhase::Draining);
        assert_eq!(coord.reason(), Some(ShutdownReason::Signal("SIGTERM".into())));

        // Can't initiate again
        assert!(!coord.initiate(ShutdownReason::Requested, 1001));

        // Can't register new tasks during drain
        assert!(coord.register_task("http", "req3", 300).is_none());

        // Complete tasks
        assert!(coord.complete_task(&t1));
        assert!(coord.complete_task(&t2));

        // Advance to teardown
        assert!(coord.try_advance_to_teardown(1100));
        assert_eq!(coord.phase(), ShutdownPhase::TearingDown);
    }

    #[test]
    fn test_drain_timeout() {
        let coord = ShutdownCoordinator::new(3000);
        let _t = coord.register_task("http", "long-req", 100).unwrap();

        coord.initiate(ShutdownReason::Requested, 1000);

        // Not timed out yet
        assert!(!coord.is_drain_timeout(2000));
        assert!(!coord.try_advance_to_teardown(2000));

        // Timed out
        assert!(coord.is_drain_timeout(4000));
        assert!(coord.try_advance_to_teardown(4000));
        assert_eq!(coord.phase(), ShutdownPhase::TearingDown);
    }

    #[test]
    fn test_hooks_and_teardown() {
        let coord = ShutdownCoordinator::new(1000);
        coord.register_component("http", 10);
        coord.register_component("db", 20);

        let log: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));

        let log1 = Arc::clone(&log);
        coord.register_hook(ShutdownHook::new("flush-cache", 5, move || {
            log1.lock().unwrap().push("flush-cache".into());
            HookResult::Ok
        }));

        let log2 = Arc::clone(&log);
        coord.register_hook(ShutdownHook::new("close-db", 10, move || {
            log2.lock().unwrap().push("close-db".into());
            HookResult::Ok
        }));

        coord.register_hook(ShutdownHook::new("fail-hook", 15, || {
            HookResult::Failed("disk error".into())
        }));

        coord.initiate(ShutdownReason::Admin("deploy".into()), 100);
        coord.try_advance_to_teardown(200);

        let results = coord.execute_teardown();
        assert_eq!(coord.phase(), ShutdownPhase::Terminated);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0], ("flush-cache".into(), HookResult::Ok));
        assert_eq!(results[1], ("close-db".into(), HookResult::Ok));
        assert_eq!(results[2], ("fail-hook".into(), HookResult::Failed("disk error".into())));

        // Hooks run in priority order
        let order = log.lock().unwrap().clone();
        assert_eq!(order, vec!["flush-cache", "close-db"]);

        // Components are shut down
        let statuses = coord.component_statuses();
        for (_, status) in &statuses {
            assert_eq!(*status, ComponentStatus::ShutDown);
        }
    }

    #[test]
    fn test_drain_health() {
        let coord = ShutdownCoordinator::new(5000);
        coord.register_component("http", 10);

        let health = coord.drain_health();
        assert_eq!(health.phase, ShutdownPhase::Running);
        assert!(health.accepting_traffic);
        assert!(health.ready);
        assert_eq!(health.components_remaining, 1);

        coord.register_task("http", "req", 100);
        coord.initiate(ShutdownReason::Requested, 200);

        let health = coord.drain_health();
        assert_eq!(health.phase, ShutdownPhase::Draining);
        assert!(!health.accepting_traffic);
        assert!(!health.ready);
        assert_eq!(health.in_flight_tasks, 1);
    }

    #[test]
    fn test_mark_component_failed() {
        let coord = ShutdownCoordinator::new(1000);
        coord.register_component("db", 10);
        coord.register_component("cache", 20);

        assert!(coord.mark_component_failed("db"));
        assert!(!coord.mark_component_failed("nonexistent"));

        let statuses = coord.component_statuses();
        assert_eq!(statuses[0], ("db".into(), ComponentStatus::Failed));
        assert_eq!(statuses[1], ("cache".into(), ComponentStatus::Running));
    }

    #[test]
    fn test_shutdown_signal() {
        let signal = ShutdownSignal::new();
        assert!(!signal.is_triggered());
        assert!(signal.reason().is_none());

        signal.trigger(ShutdownReason::Signal("SIGINT".into()));
        assert!(signal.is_triggered());
        assert_eq!(signal.reason(), Some(ShutdownReason::Signal("SIGINT".into())));
    }

    #[test]
    fn test_signal_clone_shares_state() {
        let s1 = ShutdownSignal::new();
        let s2 = s1.clone();

        s1.trigger(ShutdownReason::Requested);
        assert!(s2.is_triggered());
    }

    #[test]
    fn test_coordinator_clone_shares_state() {
        let c1 = ShutdownCoordinator::new(5000);
        let c2 = c1.clone();

        c1.register_component("http", 10);
        c1.initiate(ShutdownReason::Requested, 100);

        assert_eq!(c2.phase(), ShutdownPhase::Draining);
    }

    #[test]
    fn test_execute_teardown_not_in_teardown_phase() {
        let coord = ShutdownCoordinator::new(1000);
        let results = coord.execute_teardown();
        assert!(results.is_empty());
    }

    #[test]
    fn test_hook_skip_result() {
        let hook = ShutdownHook::new("optional", 1, || {
            HookResult::Skipped("not needed".into())
        });
        assert_eq!(hook.run(), HookResult::Skipped("not needed".into()));
    }

    #[test]
    fn test_full_lifecycle_with_all_tasks_drained() {
        let coord = ShutdownCoordinator::new(10_000);
        coord.register_component("web", 10);
        coord.register_component("queue", 20);
        coord.register_component("db", 30);

        coord.register_hook(ShutdownHook::new("flush", 1, || HookResult::Ok));

        let t1 = coord.register_task("web", "req-1", 0).unwrap();
        let t2 = coord.register_task("web", "req-2", 10).unwrap();
        let t3 = coord.register_task("queue", "msg-1", 20).unwrap();

        // Initiate
        coord.initiate(ShutdownReason::Signal("SIGTERM".into()), 100);
        assert_eq!(coord.phase(), ShutdownPhase::Draining);

        // Can't advance yet
        assert!(!coord.try_advance_to_teardown(200));

        // Complete all tasks
        coord.complete_task(&t1);
        coord.complete_task(&t2);
        coord.complete_task(&t3);

        // Now we can advance
        assert!(coord.try_advance_to_teardown(300));
        assert_eq!(coord.phase(), ShutdownPhase::TearingDown);

        let results = coord.execute_teardown();
        assert_eq!(results.len(), 1);
        assert_eq!(coord.phase(), ShutdownPhase::Terminated);
    }

    #[test]
    fn test_component_status_display() {
        assert_eq!(ComponentStatus::Running.to_string(), "running");
        assert_eq!(ComponentStatus::Draining.to_string(), "draining");
        assert_eq!(ComponentStatus::ShutDown.to_string(), "shut_down");
        assert_eq!(ComponentStatus::Failed.to_string(), "failed");
    }

    #[test]
    fn test_hook_results_after_teardown() {
        let coord = ShutdownCoordinator::new(1000);
        coord.register_hook(ShutdownHook::new("h1", 1, || HookResult::Ok));

        // Before teardown — no results
        assert!(coord.hook_results().is_empty());

        coord.initiate(ShutdownReason::Requested, 0);
        coord.try_advance_to_teardown(0);
        coord.execute_teardown();

        let results = coord.hook_results();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "h1");
    }

    #[test]
    fn test_drain_health_after_teardown() {
        let coord = ShutdownCoordinator::new(100);
        coord.register_component("svc", 1);
        coord.initiate(ShutdownReason::Requested, 0);
        coord.try_advance_to_teardown(0);
        coord.execute_teardown();

        let health = coord.drain_health();
        assert_eq!(health.phase, ShutdownPhase::Terminated);
        assert!(!health.accepting_traffic);
        assert_eq!(health.components_remaining, 0);
    }
}
