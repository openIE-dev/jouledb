//! Mission Planning — Task decomposition, waypoint sequencing, contingency
//! handling, and a mission state machine for autonomous vehicle operations.
//!
//! The mission planner manages a high-level mission definition: a sequence of
//! tasks with preconditions, waypoint assignments, priority ordering, and
//! fallback contingency plans. It drives the overall autonomy lifecycle from
//! mission load through completion or abort.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Mission planning errors.
#[derive(Debug, Clone, PartialEq)]
pub enum MissionError {
    /// Mission has no tasks.
    EmptyMission,
    /// Referenced task ID does not exist.
    UnknownTask(String),
    /// Precondition not met for a task.
    PreconditionFailed(String),
    /// Mission is in an invalid state for the requested operation.
    InvalidState(String),
    /// Contingency plan not available.
    NoContingency(String),
    /// Cycle detected in task dependencies.
    CyclicDependency(String),
}

impl fmt::Display for MissionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyMission => write!(f, "mission has no tasks"),
            Self::UnknownTask(m) => write!(f, "unknown task: {m}"),
            Self::PreconditionFailed(m) => write!(f, "precondition failed: {m}"),
            Self::InvalidState(m) => write!(f, "invalid state: {m}"),
            Self::NoContingency(m) => write!(f, "no contingency: {m}"),
            Self::CyclicDependency(m) => write!(f, "cyclic dependency: {m}"),
        }
    }
}

impl std::error::Error for MissionError {}

// ── Mission Waypoint ────────────────────────────────────────────

/// A waypoint in the mission plan.
#[derive(Debug, Clone, PartialEq)]
pub struct MissionWaypoint {
    pub id: String,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub heading: Option<f64>,
    pub speed_target: Option<f64>,
    pub loiter_time: f64,
}

impl MissionWaypoint {
    pub fn new(id: &str, x: f64, y: f64, z: f64) -> Self {
        Self {
            id: id.to_string(),
            x,
            y,
            z,
            heading: None,
            speed_target: None,
            loiter_time: 0.0,
        }
    }

    pub fn with_heading(mut self, h: f64) -> Self {
        self.heading = Some(h);
        self
    }

    pub fn with_speed(mut self, s: f64) -> Self {
        self.speed_target = Some(s);
        self
    }

    pub fn with_loiter(mut self, t: f64) -> Self {
        self.loiter_time = t.max(0.0);
        self
    }

    pub fn distance_to(&self, other: &MissionWaypoint) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

impl fmt::Display for MissionWaypoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "WP[{}]({:.1},{:.1},{:.1})", self.id, self.x, self.y, self.z)
    }
}

// ── Task ────────────────────────────────────────────────────────

/// Task status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskStatus {
    Pending,
    Active,
    Complete,
    Failed,
    Skipped,
}

impl fmt::Display for TaskStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pending => write!(f, "Pending"),
            Self::Active => write!(f, "Active"),
            Self::Complete => write!(f, "Complete"),
            Self::Failed => write!(f, "Failed"),
            Self::Skipped => write!(f, "Skipped"),
        }
    }
}

/// Task priority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

impl fmt::Display for TaskPriority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Low => write!(f, "Low"),
            Self::Normal => write!(f, "Normal"),
            Self::High => write!(f, "High"),
            Self::Critical => write!(f, "Critical"),
        }
    }
}

/// A mission task: a unit of work with waypoints, dependencies, and contingencies.
#[derive(Debug, Clone)]
pub struct MissionTask {
    pub id: String,
    pub name: String,
    pub priority: TaskPriority,
    pub status: TaskStatus,
    pub waypoints: Vec<MissionWaypoint>,
    pub depends_on: Vec<String>,
    pub contingency_task: Option<String>,
    pub timeout_secs: f64,
    pub retries: usize,
    pub max_retries: usize,
}

impl MissionTask {
    pub fn new(id: &str, name: &str) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            priority: TaskPriority::Normal,
            status: TaskStatus::Pending,
            waypoints: Vec::new(),
            depends_on: Vec::new(),
            contingency_task: None,
            timeout_secs: 300.0,
            retries: 0,
            max_retries: 2,
        }
    }

    pub fn with_priority(mut self, p: TaskPriority) -> Self {
        self.priority = p;
        self
    }

    pub fn with_waypoint(mut self, wp: MissionWaypoint) -> Self {
        self.waypoints.push(wp);
        self
    }

    pub fn with_dependency(mut self, dep: &str) -> Self {
        self.depends_on.push(dep.to_string());
        self
    }

    pub fn with_contingency(mut self, task_id: &str) -> Self {
        self.contingency_task = Some(task_id.to_string());
        self
    }

    pub fn with_timeout(mut self, t: f64) -> Self {
        self.timeout_secs = t.max(1.0);
        self
    }

    pub fn with_max_retries(mut self, r: usize) -> Self {
        self.max_retries = r;
        self
    }

    /// Check if all dependencies are satisfied (completed).
    pub fn dependencies_met(&self, completed: &[String]) -> bool {
        self.depends_on.iter().all(|d| completed.contains(d))
    }

    /// Estimated total distance across all waypoints.
    pub fn total_distance(&self) -> f64 {
        self.waypoints
            .windows(2)
            .map(|w| w[0].distance_to(&w[1]))
            .sum()
    }
}

impl fmt::Display for MissionTask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Task[{}]({}, {}, wps={})",
            self.id,
            self.name,
            self.status,
            self.waypoints.len()
        )
    }
}

// ── Mission State Machine ───────────────────────────────────────

/// Overall mission state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissionState {
    /// Mission loaded but not started.
    Ready,
    /// Mission is executing tasks.
    Running,
    /// Mission is paused.
    Paused,
    /// Executing a contingency plan.
    Contingency,
    /// Mission completed successfully.
    Complete,
    /// Mission aborted.
    Aborted,
}

impl fmt::Display for MissionState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ready => write!(f, "Ready"),
            Self::Running => write!(f, "Running"),
            Self::Paused => write!(f, "Paused"),
            Self::Contingency => write!(f, "Contingency"),
            Self::Complete => write!(f, "Complete"),
            Self::Aborted => write!(f, "Aborted"),
        }
    }
}

// ── Mission Planner ─────────────────────────────────────────────

/// Mission planner: manages task sequencing, execution, and contingencies.
#[derive(Debug, Clone)]
pub struct MissionPlanner {
    state: MissionState,
    tasks: Vec<MissionTask>,
    task_index: HashMap<String, usize>,
    current_task_idx: Option<usize>,
    completed_tasks: Vec<String>,
    failed_tasks: Vec<String>,
    elapsed_secs: f64,
}

impl MissionPlanner {
    pub fn new() -> Self {
        Self {
            state: MissionState::Ready,
            tasks: Vec::new(),
            task_index: HashMap::new(),
            current_task_idx: None,
            completed_tasks: Vec::new(),
            failed_tasks: Vec::new(),
            elapsed_secs: 0.0,
        }
    }

    /// Add a task to the mission.
    pub fn add_task(&mut self, task: MissionTask) {
        let idx = self.tasks.len();
        self.task_index.insert(task.id.clone(), idx);
        self.tasks.push(task);
    }

    /// Validate the mission: check for unknown dependencies and cycles.
    pub fn validate(&self) -> Result<(), MissionError> {
        if self.tasks.is_empty() {
            return Err(MissionError::EmptyMission);
        }
        for task in &self.tasks {
            for dep in &task.depends_on {
                if !self.task_index.contains_key(dep) {
                    return Err(MissionError::UnknownTask(format!(
                        "task {} depends on unknown task {}",
                        task.id, dep
                    )));
                }
            }
            if let Some(ref ct) = task.contingency_task {
                if !self.task_index.contains_key(ct) {
                    return Err(MissionError::UnknownTask(format!(
                        "contingency {} for task {} not found",
                        ct, task.id
                    )));
                }
            }
        }
        self.check_cycles()?;
        Ok(())
    }

    /// Start the mission.
    pub fn start(&mut self) -> Result<(), MissionError> {
        if self.state != MissionState::Ready {
            return Err(MissionError::InvalidState(format!(
                "cannot start from state {}",
                self.state
            )));
        }
        self.validate()?;
        self.state = MissionState::Running;
        self.advance()?;
        Ok(())
    }

    /// Pause the mission.
    pub fn pause(&mut self) -> Result<(), MissionError> {
        if self.state != MissionState::Running && self.state != MissionState::Contingency {
            return Err(MissionError::InvalidState("not running".into()));
        }
        self.state = MissionState::Paused;
        Ok(())
    }

    /// Resume a paused mission.
    pub fn resume(&mut self) -> Result<(), MissionError> {
        if self.state != MissionState::Paused {
            return Err(MissionError::InvalidState("not paused".into()));
        }
        self.state = MissionState::Running;
        Ok(())
    }

    /// Abort the mission.
    pub fn abort(&mut self) {
        self.state = MissionState::Aborted;
        if let Some(idx) = self.current_task_idx {
            self.tasks[idx].status = TaskStatus::Failed;
        }
    }

    /// Mark the current task as complete and advance to the next.
    pub fn complete_current_task(&mut self) -> Result<(), MissionError> {
        if self.state != MissionState::Running && self.state != MissionState::Contingency {
            return Err(MissionError::InvalidState("not running".into()));
        }
        if let Some(idx) = self.current_task_idx {
            self.tasks[idx].status = TaskStatus::Complete;
            self.completed_tasks.push(self.tasks[idx].id.clone());
        }
        self.advance()
    }

    /// Mark the current task as failed and handle contingency.
    pub fn fail_current_task(&mut self) -> Result<(), MissionError> {
        if self.state != MissionState::Running && self.state != MissionState::Contingency {
            return Err(MissionError::InvalidState("not running".into()));
        }
        if let Some(idx) = self.current_task_idx {
            let task = &mut self.tasks[idx];
            task.retries += 1;
            if task.retries <= task.max_retries {
                // Retry the same task.
                task.status = TaskStatus::Active;
                return Ok(());
            }
            task.status = TaskStatus::Failed;
            self.failed_tasks.push(task.id.clone());

            // Try contingency.
            if let Some(ref ct) = task.contingency_task.clone() {
                if let Some(&ci) = self.task_index.get(ct) {
                    self.state = MissionState::Contingency;
                    self.tasks[ci].status = TaskStatus::Active;
                    self.current_task_idx = Some(ci);
                    return Ok(());
                }
            }
            // Skip to next if no contingency.
            self.advance()?;
        }
        Ok(())
    }

    /// Advance to the next pending task whose dependencies are met.
    fn advance(&mut self) -> Result<(), MissionError> {
        // Topological ordering: pick the highest-priority ready task.
        let mut best: Option<usize> = None;
        for (i, task) in self.tasks.iter().enumerate() {
            if task.status != TaskStatus::Pending {
                continue;
            }
            if !task.dependencies_met(&self.completed_tasks) {
                continue;
            }
            if best.map_or(true, |b| task.priority > self.tasks[b].priority) {
                best = Some(i);
            }
        }

        match best {
            Some(i) => {
                self.tasks[i].status = TaskStatus::Active;
                self.current_task_idx = Some(i);
                if self.state == MissionState::Contingency {
                    self.state = MissionState::Running;
                }
                Ok(())
            }
            None => {
                self.state = MissionState::Complete;
                self.current_task_idx = None;
                Ok(())
            }
        }
    }

    fn check_cycles(&self) -> Result<(), MissionError> {
        // Simple DFS cycle detection on the dependency graph.
        let n = self.tasks.len();
        let mut visited = vec![0u8; n]; // 0=unvisited, 1=in-progress, 2=done

        for start in 0..n {
            if visited[start] == 0 {
                let mut stack = vec![(start, false)];
                while let Some((node, returning)) = stack.pop() {
                    if returning {
                        visited[node] = 2;
                        continue;
                    }
                    if visited[node] == 1 {
                        return Err(MissionError::CyclicDependency(
                            self.tasks[node].id.clone(),
                        ));
                    }
                    if visited[node] == 2 {
                        continue;
                    }
                    visited[node] = 1;
                    stack.push((node, true));
                    for dep in &self.tasks[node].depends_on {
                        if let Some(&di) = self.task_index.get(dep) {
                            if visited[di] == 1 {
                                return Err(MissionError::CyclicDependency(
                                    format!("{} -> {}", self.tasks[node].id, dep),
                                ));
                            }
                            if visited[di] == 0 {
                                stack.push((di, false));
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Get the current mission state.
    pub fn state(&self) -> MissionState {
        self.state
    }

    /// Get the current task, if any.
    pub fn current_task(&self) -> Option<&MissionTask> {
        self.current_task_idx.map(|i| &self.tasks[i])
    }

    /// Get all tasks.
    pub fn tasks(&self) -> &[MissionTask] {
        &self.tasks
    }

    /// Compute a topological ordering of tasks (for display/sequencing).
    pub fn topological_order(&self) -> Result<Vec<String>, MissionError> {
        let n = self.tasks.len();
        let mut in_degree = vec![0usize; n];
        for task in &self.tasks {
            for dep in &task.depends_on {
                if let Some(&i) = self.task_index.get(dep) {
                    let _ = i; // dep has edge to this task
                }
            }
        }
        // Build adjacency: dep → task (dep must come before task).
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (ti, task) in self.tasks.iter().enumerate() {
            for dep in &task.depends_on {
                if let Some(&di) = self.task_index.get(dep) {
                    adj[di].push(ti);
                    in_degree[ti] += 1;
                }
            }
        }

        let mut queue: Vec<usize> = (0..n).filter(|i| in_degree[*i] == 0).collect();
        // Sort by priority descending within the same level.
        queue.sort_by(|a, b| self.tasks[*b].priority.cmp(&self.tasks[*a].priority));

        let mut order = Vec::new();
        while let Some(node) = queue.pop() {
            order.push(self.tasks[node].id.clone());
            for &next in &adj[node] {
                in_degree[next] -= 1;
                if in_degree[next] == 0 {
                    queue.push(next);
                    queue.sort_by(|a, b| self.tasks[*b].priority.cmp(&self.tasks[*a].priority));
                }
            }
        }
        if order.len() != n {
            return Err(MissionError::CyclicDependency("topological sort incomplete".into()));
        }
        Ok(order)
    }

    /// Total estimated distance across all tasks.
    pub fn total_distance(&self) -> f64 {
        self.tasks.iter().map(|t| t.total_distance()).sum()
    }

    /// Mission completion ratio.
    pub fn progress(&self) -> f64 {
        if self.tasks.is_empty() {
            return 0.0;
        }
        self.completed_tasks.len() as f64 / self.tasks.len() as f64
    }
}

impl fmt::Display for MissionPlanner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Mission(state={}, tasks={}, done={}, progress={:.0}%)",
            self.state,
            self.tasks.len(),
            self.completed_tasks.len(),
            self.progress() * 100.0
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn wp(id: &str, x: f64, y: f64) -> MissionWaypoint {
        MissionWaypoint::new(id, x, y, 0.0)
    }

    fn simple_mission() -> MissionPlanner {
        let mut mp = MissionPlanner::new();
        mp.add_task(
            MissionTask::new("t1", "Go to A")
                .with_waypoint(wp("w1", 0.0, 0.0))
                .with_waypoint(wp("w2", 10.0, 0.0)),
        );
        mp.add_task(
            MissionTask::new("t2", "Go to B")
                .with_dependency("t1")
                .with_waypoint(wp("w3", 10.0, 0.0))
                .with_waypoint(wp("w4", 10.0, 10.0)),
        );
        mp
    }

    #[test]
    fn test_waypoint_display() {
        let w = wp("alpha", 1.5, 2.5);
        assert!(format!("{w}").contains("alpha"));
    }

    #[test]
    fn test_waypoint_distance() {
        let a = wp("a", 0.0, 0.0);
        let b = wp("b", 3.0, 4.0);
        assert!((a.distance_to(&b) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_task_display() {
        let t = MissionTask::new("t1", "Navigate");
        assert!(format!("{t}").contains("Navigate"));
    }

    #[test]
    fn test_task_dependencies_met() {
        let t = MissionTask::new("t1", "A").with_dependency("t0");
        assert!(!t.dependencies_met(&[]));
        assert!(t.dependencies_met(&["t0".to_string()]));
    }

    #[test]
    fn test_task_total_distance() {
        let t = MissionTask::new("t1", "A")
            .with_waypoint(wp("a", 0.0, 0.0))
            .with_waypoint(wp("b", 3.0, 4.0));
        assert!((t.total_distance() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_mission_validate() {
        let mp = MissionPlanner::new();
        assert_eq!(mp.validate(), Err(MissionError::EmptyMission));
    }

    #[test]
    fn test_validate_unknown_dep() {
        let mut mp = MissionPlanner::new();
        mp.add_task(MissionTask::new("t1", "A").with_dependency("unknown"));
        assert!(mp.validate().is_err());
    }

    #[test]
    fn test_start_mission() {
        let mut mp = simple_mission();
        mp.start().unwrap();
        assert_eq!(mp.state(), MissionState::Running);
        assert!(mp.current_task().is_some());
        assert_eq!(mp.current_task().unwrap().id, "t1");
    }

    #[test]
    fn test_complete_task() {
        let mut mp = simple_mission();
        mp.start().unwrap();
        mp.complete_current_task().unwrap();
        assert_eq!(mp.current_task().unwrap().id, "t2");
    }

    #[test]
    fn test_mission_complete() {
        let mut mp = simple_mission();
        mp.start().unwrap();
        mp.complete_current_task().unwrap();
        mp.complete_current_task().unwrap();
        assert_eq!(mp.state(), MissionState::Complete);
    }

    #[test]
    fn test_pause_resume() {
        let mut mp = simple_mission();
        mp.start().unwrap();
        mp.pause().unwrap();
        assert_eq!(mp.state(), MissionState::Paused);
        mp.resume().unwrap();
        assert_eq!(mp.state(), MissionState::Running);
    }

    #[test]
    fn test_abort() {
        let mut mp = simple_mission();
        mp.start().unwrap();
        mp.abort();
        assert_eq!(mp.state(), MissionState::Aborted);
    }

    #[test]
    fn test_fail_with_retry() {
        let mut mp = MissionPlanner::new();
        mp.add_task(MissionTask::new("t1", "A").with_max_retries(2));
        mp.start().unwrap();
        mp.fail_current_task().unwrap(); // retry 1
        assert_eq!(mp.current_task().unwrap().status, TaskStatus::Active);
        mp.fail_current_task().unwrap(); // retry 2
        assert_eq!(mp.current_task().unwrap().status, TaskStatus::Active);
    }

    #[test]
    fn test_fail_with_contingency() {
        let mut mp = MissionPlanner::new();
        mp.add_task(
            MissionTask::new("t1", "Main")
                .with_max_retries(0)
                .with_contingency("t_fallback"),
        );
        mp.add_task(MissionTask::new("t_fallback", "Fallback"));
        mp.start().unwrap();
        mp.fail_current_task().unwrap();
        assert_eq!(mp.state(), MissionState::Contingency);
        assert_eq!(mp.current_task().unwrap().id, "t_fallback");
    }

    #[test]
    fn test_topological_order() {
        let mp = simple_mission();
        let order = mp.topological_order().unwrap();
        let t1_pos = order.iter().position(|s| s == "t1").unwrap();
        let t2_pos = order.iter().position(|s| s == "t2").unwrap();
        assert!(t1_pos < t2_pos);
    }

    #[test]
    fn test_progress() {
        let mut mp = simple_mission();
        assert!((mp.progress() - 0.0).abs() < 1e-9);
        mp.start().unwrap();
        mp.complete_current_task().unwrap();
        assert!((mp.progress() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_total_distance() {
        let mp = simple_mission();
        let d = mp.total_distance();
        assert!(d > 0.0);
    }

    #[test]
    fn test_mission_display() {
        let mp = simple_mission();
        let s = format!("{mp}");
        assert!(s.contains("Ready"));
        assert!(s.contains("tasks=2"));
    }

    #[test]
    fn test_priority_ordering() {
        let mut mp = MissionPlanner::new();
        mp.add_task(MissionTask::new("low", "Low").with_priority(TaskPriority::Low));
        mp.add_task(MissionTask::new("high", "High").with_priority(TaskPriority::High));
        mp.start().unwrap();
        assert_eq!(mp.current_task().unwrap().id, "high");
    }

    #[test]
    fn test_cannot_start_twice() {
        let mut mp = simple_mission();
        mp.start().unwrap();
        assert!(mp.start().is_err());
    }

    #[test]
    fn test_waypoint_with_loiter() {
        let w = wp("w1", 0.0, 0.0).with_loiter(30.0).with_heading(1.5);
        assert_eq!(w.loiter_time, 30.0);
        assert_eq!(w.heading, Some(1.5));
    }
}
