//! DAG-based task scheduler — dependency graph with topological sort and critical path.
//!
//! Replaces task runners and DAG schedulers with a pure-Rust implementation.
//! Models tasks as nodes, dependencies as directed edges, detects cycles,
//! computes topological execution order, identifies parallel execution groups
//! of independent tasks, and performs critical path analysis.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

// ── Task Status ───────────────────────────────────────────────

/// Status of a task in the DAG.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskStatus {
    /// Not yet started.
    Pending,
    /// Ready to run (all dependencies satisfied).
    Ready,
    /// Currently executing.
    Running,
    /// Completed successfully.
    Completed,
    /// Failed with error.
    Failed,
    /// Skipped (e.g., upstream failure).
    Skipped,
}

// ── Task Node ─────────────────────────────────────────────────

/// A task node in the DAG.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskNode {
    pub id: String,
    pub label: String,
    /// Estimated duration in ms (for critical path analysis).
    pub estimated_duration_ms: u64,
    /// Actual duration in ms (filled after completion).
    pub actual_duration_ms: Option<u64>,
    /// Current status.
    pub status: TaskStatus,
    /// Priority (higher = runs first within a group).
    pub priority: u32,
}

// ── DAG Error ─────────────────────────────────────────────────

/// Errors from DAG operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DagError {
    /// Cycle detected in the graph.
    CycleDetected(Vec<String>),
    /// Task not found.
    TaskNotFound(String),
    /// Dependency already exists.
    DuplicateDependency { from: String, to: String },
    /// Task already exists.
    DuplicateTask(String),
    /// Would create a self-loop.
    SelfLoop(String),
}

impl std::fmt::Display for DagError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CycleDetected(path) => write!(f, "cycle detected: {}", path.join(" -> ")),
            Self::TaskNotFound(id) => write!(f, "task not found: {}", id),
            Self::DuplicateDependency { from, to } => {
                write!(f, "duplicate dependency: {} -> {}", from, to)
            }
            Self::DuplicateTask(id) => write!(f, "duplicate task: {}", id),
            Self::SelfLoop(id) => write!(f, "self-loop on task: {}", id),
        }
    }
}

// ── Execution Group ───────────────────────────────────────────

/// A group of tasks that can execute in parallel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionGroup {
    /// Level in the topological ordering (0 = no dependencies).
    pub level: usize,
    /// Task IDs that can run concurrently at this level.
    pub task_ids: Vec<String>,
}

// ── Critical Path ─────────────────────────────────────────────

/// The critical path through the DAG.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticalPath {
    /// Task IDs on the critical path, in order.
    pub path: Vec<String>,
    /// Total estimated duration of the critical path.
    pub total_duration_ms: u64,
}

// ── Task DAG ──────────────────────────────────────────────────

/// Directed acyclic graph for task scheduling.
#[derive(Debug)]
pub struct TaskDag {
    /// Task nodes by ID.
    nodes: HashMap<String, TaskNode>,
    /// Adjacency list: task_id -> set of tasks it depends on (predecessors).
    dependencies: HashMap<String, HashSet<String>>,
    /// Reverse adjacency: task_id -> set of tasks that depend on it (successors).
    dependents: HashMap<String, HashSet<String>>,
    /// Insertion order for deterministic iteration.
    insertion_order: Vec<String>,
}

impl TaskDag {
    /// Create a new empty DAG.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            dependencies: HashMap::new(),
            dependents: HashMap::new(),
            insertion_order: Vec::new(),
        }
    }

    /// Add a task to the DAG.
    pub fn add_task(
        &mut self,
        id: &str,
        label: &str,
        estimated_duration_ms: u64,
    ) -> Result<(), DagError> {
        if self.nodes.contains_key(id) {
            return Err(DagError::DuplicateTask(id.to_string()));
        }
        let node = TaskNode {
            id: id.to_string(),
            label: label.to_string(),
            estimated_duration_ms,
            actual_duration_ms: None,
            status: TaskStatus::Pending,
            priority: 0,
        };
        self.nodes.insert(id.to_string(), node);
        self.dependencies.entry(id.to_string()).or_default();
        self.dependents.entry(id.to_string()).or_default();
        self.insertion_order.push(id.to_string());
        Ok(())
    }

    /// Add a task with priority.
    pub fn add_task_with_priority(
        &mut self,
        id: &str,
        label: &str,
        estimated_duration_ms: u64,
        priority: u32,
    ) -> Result<(), DagError> {
        self.add_task(id, label, estimated_duration_ms)?;
        if let Some(node) = self.nodes.get_mut(id) {
            node.priority = priority;
        }
        Ok(())
    }

    /// Add a dependency: `task_id` depends on `depends_on_id` (depends_on must complete first).
    pub fn add_dependency(
        &mut self,
        task_id: &str,
        depends_on_id: &str,
    ) -> Result<(), DagError> {
        if task_id == depends_on_id {
            return Err(DagError::SelfLoop(task_id.to_string()));
        }
        if !self.nodes.contains_key(task_id) {
            return Err(DagError::TaskNotFound(task_id.to_string()));
        }
        if !self.nodes.contains_key(depends_on_id) {
            return Err(DagError::TaskNotFound(depends_on_id.to_string()));
        }

        let deps = self.dependencies.entry(task_id.to_string()).or_default();
        if deps.contains(depends_on_id) {
            return Err(DagError::DuplicateDependency {
                from: task_id.to_string(),
                to: depends_on_id.to_string(),
            });
        }

        deps.insert(depends_on_id.to_string());
        self.dependents
            .entry(depends_on_id.to_string())
            .or_default()
            .insert(task_id.to_string());

        // Check for cycles.
        if self.has_cycle() {
            // Roll back.
            self.dependencies
                .get_mut(task_id)
                .unwrap()
                .remove(depends_on_id);
            self.dependents
                .get_mut(depends_on_id)
                .unwrap()
                .remove(task_id);
            return Err(DagError::CycleDetected(vec![
                task_id.to_string(),
                depends_on_id.to_string(),
            ]));
        }

        Ok(())
    }

    /// Get a task by ID.
    pub fn get_task(&self, id: &str) -> Option<&TaskNode> {
        self.nodes.get(id)
    }

    /// Get a mutable task reference.
    pub fn get_task_mut(&mut self, id: &str) -> Option<&mut TaskNode> {
        self.nodes.get_mut(id)
    }

    /// Number of tasks.
    pub fn task_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of dependency edges.
    pub fn edge_count(&self) -> usize {
        self.dependencies.values().map(|s| s.len()).sum()
    }

    /// Get the dependencies of a task (what it depends on).
    pub fn dependencies_of(&self, id: &str) -> Vec<String> {
        self.dependencies
            .get(id)
            .map(|s| {
                let mut v: Vec<String> = s.iter().cloned().collect();
                v.sort();
                v
            })
            .unwrap_or_default()
    }

    /// Get the dependents of a task (what depends on it).
    pub fn dependents_of(&self, id: &str) -> Vec<String> {
        self.dependents
            .get(id)
            .map(|s| {
                let mut v: Vec<String> = s.iter().cloned().collect();
                v.sort();
                v
            })
            .unwrap_or_default()
    }

    /// Detect if the DAG contains a cycle (should always be false if add_dependency is used).
    pub fn has_cycle(&self) -> bool {
        self.topological_sort().is_none()
    }

    /// Compute topological sort (Kahn's algorithm). Returns None if there's a cycle.
    pub fn topological_sort(&self) -> Option<Vec<String>> {
        let mut in_degree: HashMap<String, usize> = HashMap::new();
        for id in self.nodes.keys() {
            in_degree.insert(
                id.clone(),
                self.dependencies.get(id).map_or(0, |s| s.len()),
            );
        }

        let mut queue: VecDeque<String> = VecDeque::new();
        // Use insertion order for determinism among tasks with same in-degree.
        for id in &self.insertion_order {
            if in_degree.get(id).copied().unwrap_or(0) == 0 {
                queue.push_back(id.clone());
            }
        }

        let mut sorted = Vec::new();

        while let Some(id) = queue.pop_front() {
            sorted.push(id.clone());
            if let Some(succs) = self.dependents.get(&id) {
                let mut ordered_succs: Vec<String> = succs.iter().cloned().collect();
                ordered_succs.sort();
                for succ in ordered_succs {
                    if let Some(deg) = in_degree.get_mut(&succ) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push_back(succ);
                        }
                    }
                }
            }
        }

        if sorted.len() == self.nodes.len() {
            Some(sorted)
        } else {
            None // Cycle detected.
        }
    }

    /// Compute parallel execution groups (tasks at same level can run concurrently).
    pub fn execution_groups(&self) -> Option<Vec<ExecutionGroup>> {
        let topo = self.topological_sort()?;

        // Compute the level of each task (longest path from a source).
        let mut levels: HashMap<String, usize> = HashMap::new();
        for id in &topo {
            let max_dep_level = self
                .dependencies
                .get(id)
                .map(|deps| {
                    deps.iter()
                        .filter_map(|d| levels.get(d))
                        .max()
                        .copied()
                        .map(|l| l + 1)
                        .unwrap_or(0)
                })
                .unwrap_or(0);
            levels.insert(id.clone(), max_dep_level);
        }

        // Group by level.
        let max_level = levels.values().max().copied().unwrap_or(0);
        let mut groups = Vec::new();

        for level in 0..=max_level {
            let mut task_ids: Vec<String> = levels
                .iter()
                .filter(|(_k, v)| **v == level)
                .map(|(k, _)| k.clone())
                .collect();
            task_ids.sort_by(|a, b| {
                let pa = self.nodes.get(a).map_or(0, |n| n.priority);
                let pb = self.nodes.get(b).map_or(0, |n| n.priority);
                pb.cmp(&pa).then_with(|| a.cmp(b))
            });
            groups.push(ExecutionGroup { level, task_ids });
        }

        Some(groups)
    }

    /// Compute the critical path (longest weighted path through the DAG).
    pub fn critical_path(&self) -> Option<CriticalPath> {
        let topo = self.topological_sort()?;

        // Longest-path DP.
        let mut dist: HashMap<String, u64> = HashMap::new();
        let mut predecessor: HashMap<String, Option<String>> = HashMap::new();

        for id in &topo {
            let node_dur = self.nodes.get(id).map_or(0, |n| n.estimated_duration_ms);
            let best_prev = self
                .dependencies
                .get(id)
                .map(|deps| {
                    deps.iter()
                        .filter_map(|d| dist.get(d).map(|v| (d.clone(), *v)))
                        .max_by_key(|(_, v)| *v)
                })
                .unwrap_or(None);

            let (prev_id, prev_dist) = match best_prev {
                Some((pid, pd)) => (Some(pid), pd),
                None => (None, 0),
            };

            dist.insert(id.clone(), prev_dist + node_dur);
            predecessor.insert(id.clone(), prev_id);
        }

        // Find the task with the maximum distance.
        let end_task = dist.iter().max_by_key(|(_, v)| **v)?.0.clone();
        let total_duration = dist[&end_task];

        // Trace back.
        let mut path = vec![end_task.clone()];
        let mut current = end_task;
        while let Some(Some(prev)) = predecessor.get(&current) {
            path.push(prev.clone());
            current = prev.clone();
        }
        path.reverse();

        Some(CriticalPath {
            path,
            total_duration_ms: total_duration,
        })
    }

    /// Update a task's status.
    pub fn set_status(&mut self, id: &str, status: TaskStatus) -> bool {
        if let Some(node) = self.nodes.get_mut(id) {
            node.status = status;
            true
        } else {
            false
        }
    }

    /// Record actual duration for a completed task.
    pub fn record_duration(&mut self, id: &str, duration_ms: u64) -> bool {
        if let Some(node) = self.nodes.get_mut(id) {
            node.actual_duration_ms = Some(duration_ms);
            true
        } else {
            false
        }
    }

    /// Get all tasks that are ready to run (pending + all deps completed).
    pub fn ready_tasks(&self) -> Vec<String> {
        let mut ready = Vec::new();
        for (id, node) in &self.nodes {
            if node.status != TaskStatus::Pending {
                continue;
            }
            let all_deps_done = self
                .dependencies
                .get(id)
                .map(|deps| {
                    deps.iter().all(|d| {
                        self.nodes
                            .get(d)
                            .map_or(false, |n| n.status == TaskStatus::Completed)
                    })
                })
                .unwrap_or(true);
            if all_deps_done {
                ready.push(id.clone());
            }
        }
        ready.sort();
        ready
    }

    /// Get tasks by status.
    pub fn tasks_with_status(&self, status: TaskStatus) -> Vec<String> {
        let mut result: Vec<String> = self
            .nodes
            .iter()
            .filter(|(_, n)| n.status == status)
            .map(|(id, _)| id.clone())
            .collect();
        result.sort();
        result
    }

    /// Get all leaf tasks (no dependents).
    pub fn leaf_tasks(&self) -> Vec<String> {
        let mut leaves: Vec<String> = self
            .nodes
            .keys()
            .filter(|id| {
                self.dependents
                    .get(*id)
                    .map_or(true, |s| s.is_empty())
            })
            .cloned()
            .collect();
        leaves.sort();
        leaves
    }

    /// Get all root tasks (no dependencies).
    pub fn root_tasks(&self) -> Vec<String> {
        let mut roots: Vec<String> = self
            .nodes
            .keys()
            .filter(|id| {
                self.dependencies
                    .get(*id)
                    .map_or(true, |s| s.is_empty())
            })
            .cloned()
            .collect();
        roots.sort();
        roots
    }
}

impl Default for TaskDag {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn diamond_dag() -> TaskDag {
        // A -> B -> D
        // A -> C -> D
        let mut dag = TaskDag::new();
        dag.add_task("A", "Start", 100).unwrap();
        dag.add_task("B", "Left", 200).unwrap();
        dag.add_task("C", "Right", 300).unwrap();
        dag.add_task("D", "End", 100).unwrap();
        dag.add_dependency("B", "A").unwrap();
        dag.add_dependency("C", "A").unwrap();
        dag.add_dependency("D", "B").unwrap();
        dag.add_dependency("D", "C").unwrap();
        dag
    }

    #[test]
    fn test_add_task() {
        let mut dag = TaskDag::new();
        dag.add_task("t1", "Task 1", 100).unwrap();
        assert_eq!(dag.task_count(), 1);
        assert_eq!(dag.get_task("t1").unwrap().label, "Task 1");
    }

    #[test]
    fn test_duplicate_task() {
        let mut dag = TaskDag::new();
        dag.add_task("t1", "Task 1", 100).unwrap();
        let err = dag.add_task("t1", "Duplicate", 200).unwrap_err();
        assert!(matches!(err, DagError::DuplicateTask(_)));
    }

    #[test]
    fn test_add_dependency() {
        let mut dag = TaskDag::new();
        dag.add_task("a", "A", 100).unwrap();
        dag.add_task("b", "B", 100).unwrap();
        dag.add_dependency("b", "a").unwrap();
        assert_eq!(dag.dependencies_of("b"), vec!["a"]);
        assert_eq!(dag.dependents_of("a"), vec!["b"]);
    }

    #[test]
    fn test_self_loop_rejected() {
        let mut dag = TaskDag::new();
        dag.add_task("a", "A", 100).unwrap();
        let err = dag.add_dependency("a", "a").unwrap_err();
        assert!(matches!(err, DagError::SelfLoop(_)));
    }

    #[test]
    fn test_cycle_detected() {
        let mut dag = TaskDag::new();
        dag.add_task("a", "A", 100).unwrap();
        dag.add_task("b", "B", 100).unwrap();
        dag.add_dependency("b", "a").unwrap();
        let err = dag.add_dependency("a", "b").unwrap_err();
        assert!(matches!(err, DagError::CycleDetected(_)));
    }

    #[test]
    fn test_topological_sort() {
        let dag = diamond_dag();
        let sorted = dag.topological_sort().unwrap();
        // A must come before B and C; D must come after B and C.
        let pos_a = sorted.iter().position(|x| x == "A").unwrap();
        let pos_b = sorted.iter().position(|x| x == "B").unwrap();
        let pos_c = sorted.iter().position(|x| x == "C").unwrap();
        let pos_d = sorted.iter().position(|x| x == "D").unwrap();
        assert!(pos_a < pos_b);
        assert!(pos_a < pos_c);
        assert!(pos_b < pos_d);
        assert!(pos_c < pos_d);
    }

    #[test]
    fn test_execution_groups() {
        let dag = diamond_dag();
        let groups = dag.execution_groups().unwrap();
        // Level 0: A, Level 1: B,C (parallel), Level 2: D.
        assert_eq!(groups.len(), 3);
        assert_eq!(groups[0].task_ids, vec!["A"]);
        assert!(groups[1].task_ids.contains(&"B".to_string()));
        assert!(groups[1].task_ids.contains(&"C".to_string()));
        assert_eq!(groups[2].task_ids, vec!["D"]);
    }

    #[test]
    fn test_critical_path() {
        let dag = diamond_dag();
        // A(100) -> C(300) -> D(100) = 500 (longer than A->B->D = 400)
        let cp = dag.critical_path().unwrap();
        assert_eq!(cp.path, vec!["A", "C", "D"]);
        assert_eq!(cp.total_duration_ms, 500);
    }

    #[test]
    fn test_ready_tasks() {
        let mut dag = diamond_dag();
        // Initially, only A is ready (no dependencies).
        let ready = dag.ready_tasks();
        assert_eq!(ready, vec!["A"]);
        // Complete A -> B and C become ready.
        dag.set_status("A", TaskStatus::Completed);
        let ready = dag.ready_tasks();
        assert!(ready.contains(&"B".to_string()));
        assert!(ready.contains(&"C".to_string()));
    }

    #[test]
    fn test_task_status_tracking() {
        let mut dag = TaskDag::new();
        dag.add_task("t", "Task", 100).unwrap();
        assert_eq!(dag.get_task("t").unwrap().status, TaskStatus::Pending);
        dag.set_status("t", TaskStatus::Running);
        assert_eq!(dag.get_task("t").unwrap().status, TaskStatus::Running);
        dag.set_status("t", TaskStatus::Completed);
        assert_eq!(dag.get_task("t").unwrap().status, TaskStatus::Completed);
    }

    #[test]
    fn test_record_duration() {
        let mut dag = TaskDag::new();
        dag.add_task("t", "Task", 100).unwrap();
        dag.record_duration("t", 150);
        assert_eq!(dag.get_task("t").unwrap().actual_duration_ms, Some(150));
    }

    #[test]
    fn test_root_and_leaf_tasks() {
        let dag = diamond_dag();
        assert_eq!(dag.root_tasks(), vec!["A"]);
        assert_eq!(dag.leaf_tasks(), vec!["D"]);
    }

    #[test]
    fn test_tasks_with_status() {
        let mut dag = diamond_dag();
        dag.set_status("A", TaskStatus::Completed);
        dag.set_status("B", TaskStatus::Running);
        let completed = dag.tasks_with_status(TaskStatus::Completed);
        assert_eq!(completed, vec!["A"]);
        let pending = dag.tasks_with_status(TaskStatus::Pending);
        assert_eq!(pending.len(), 2); // C and D.
    }

    #[test]
    fn test_edge_count() {
        let dag = diamond_dag();
        assert_eq!(dag.edge_count(), 4);
    }

    #[test]
    fn test_duplicate_dependency() {
        let mut dag = TaskDag::new();
        dag.add_task("a", "A", 100).unwrap();
        dag.add_task("b", "B", 100).unwrap();
        dag.add_dependency("b", "a").unwrap();
        let err = dag.add_dependency("b", "a").unwrap_err();
        assert!(matches!(err, DagError::DuplicateDependency { .. }));
    }

    #[test]
    fn test_unknown_task_dependency() {
        let mut dag = TaskDag::new();
        dag.add_task("a", "A", 100).unwrap();
        let err = dag.add_dependency("a", "nonexistent").unwrap_err();
        assert!(matches!(err, DagError::TaskNotFound(_)));
    }

    #[test]
    fn test_linear_chain() {
        let mut dag = TaskDag::new();
        dag.add_task("1", "First", 10).unwrap();
        dag.add_task("2", "Second", 20).unwrap();
        dag.add_task("3", "Third", 30).unwrap();
        dag.add_dependency("2", "1").unwrap();
        dag.add_dependency("3", "2").unwrap();
        let groups = dag.execution_groups().unwrap();
        assert_eq!(groups.len(), 3);
        assert_eq!(groups[0].task_ids.len(), 1);
    }

    #[test]
    fn test_default_constructor() {
        let dag = TaskDag::default();
        assert_eq!(dag.task_count(), 0);
    }

    #[test]
    fn test_error_display() {
        let err = DagError::CycleDetected(vec!["a".into(), "b".into()]);
        let msg = format!("{}", err);
        assert!(msg.contains("cycle"));
    }

    #[test]
    fn test_priority_in_execution_groups() {
        let mut dag = TaskDag::new();
        dag.add_task_with_priority("low", "Low", 100, 1).unwrap();
        dag.add_task_with_priority("high", "High", 100, 10).unwrap();
        let groups = dag.execution_groups().unwrap();
        // Both at level 0; high priority should come first.
        assert_eq!(groups[0].task_ids[0], "high");
    }
}
