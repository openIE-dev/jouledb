//! Gantt chart with critical path, milestones, and resource management.
//!
//! Replaces dhtmlxGantt / frappe-gantt. Provides task dependencies,
//! critical path calculation, slack time, and timeline scaling.
//! Pure Rust — no browser dependency.

use std::collections::{HashMap, HashSet, VecDeque};

// ── Data types ───────────────────────────────────────────────────

/// A date represented as days since epoch (for simplicity and testability).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GanttDate(pub i64);

impl GanttDate {
    pub fn new(days: i64) -> Self {
        Self(days)
    }

    pub fn days_between(self, other: GanttDate) -> i64 {
        (other.0 - self.0).abs()
    }
}

impl std::ops::Add<i64> for GanttDate {
    type Output = GanttDate;
    fn add(self, days: i64) -> GanttDate {
        GanttDate(self.0 + days)
    }
}

impl std::ops::Sub for GanttDate {
    type Output = i64;
    fn sub(self, other: GanttDate) -> i64 {
        self.0 - other.0
    }
}

/// A task in the Gantt chart.
#[derive(Debug, Clone)]
pub struct GanttTask {
    pub id: String,
    pub name: String,
    pub start: GanttDate,
    pub end: GanttDate,
    pub progress: u8,
    pub dependencies: Vec<String>,
    pub resource: Option<String>,
    pub milestone: bool,
}

impl GanttTask {
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        start: GanttDate,
        end: GanttDate,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            start,
            end,
            progress: 0,
            dependencies: Vec::new(),
            resource: None,
            milestone: false,
        }
    }

    pub fn milestone(id: impl Into<String>, name: impl Into<String>, date: GanttDate) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            start: date,
            end: date,
            progress: 0,
            dependencies: Vec::new(),
            resource: None,
            milestone: true,
        }
    }

    pub fn with_progress(mut self, progress: u8) -> Self {
        self.progress = progress.min(100);
        self
    }

    pub fn with_dependency(mut self, dep: impl Into<String>) -> Self {
        self.dependencies.push(dep.into());
        self
    }

    pub fn with_resource(mut self, resource: impl Into<String>) -> Self {
        self.resource = Some(resource.into());
        self
    }

    pub fn duration_days(&self) -> i64 {
        self.end - self.start
    }
}

/// Timeline scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeScale {
    Days,
    Weeks,
    Months,
}

/// Validation errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GanttError {
    DuplicateTaskId(String),
    CircularDependency(Vec<String>),
    EndBeforeStart(String),
    MissingDependency { task: String, dependency: String },
}

impl std::fmt::Display for GanttError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateTaskId(id) => write!(f, "duplicate task id: {id}"),
            Self::CircularDependency(cycle) => write!(f, "circular dependency: {}", cycle.join(" -> ")),
            Self::EndBeforeStart(id) => write!(f, "task {id}: end before start"),
            Self::MissingDependency { task, dependency } => {
                write!(f, "task {task}: missing dependency {dependency}")
            }
        }
    }
}

impl std::error::Error for GanttError {}

// ── GanttChart ───────────────────────────────────────────────────

/// The Gantt chart model.
#[derive(Debug, Clone)]
pub struct GanttChart {
    tasks: Vec<GanttTask>,
    task_map: HashMap<String, usize>,
    scale: TimeScale,
}

impl GanttChart {
    pub fn new() -> Self {
        Self {
            tasks: Vec::new(),
            task_map: HashMap::new(),
            scale: TimeScale::Days,
        }
    }

    pub fn with_scale(mut self, scale: TimeScale) -> Self {
        self.scale = scale;
        self
    }

    pub fn add_task(&mut self, task: GanttTask) -> Result<(), GanttError> {
        if self.task_map.contains_key(&task.id) {
            return Err(GanttError::DuplicateTaskId(task.id.clone()));
        }
        let idx = self.tasks.len();
        self.task_map.insert(task.id.clone(), idx);
        self.tasks.push(task);
        Ok(())
    }

    pub fn tasks(&self) -> &[GanttTask] {
        &self.tasks
    }

    pub fn find_task(&self, id: &str) -> Option<&GanttTask> {
        self.task_map.get(id).map(|i| &self.tasks[*i])
    }

    pub fn scale(&self) -> TimeScale {
        self.scale
    }

    /// Validate the chart: check dates, dependencies, and cycles.
    pub fn validate(&self) -> Vec<GanttError> {
        let mut errors = Vec::new();

        for task in &self.tasks {
            if task.end < task.start {
                errors.push(GanttError::EndBeforeStart(task.id.clone()));
            }
            for dep in &task.dependencies {
                if !self.task_map.contains_key(dep) {
                    errors.push(GanttError::MissingDependency {
                        task: task.id.clone(),
                        dependency: dep.clone(),
                    });
                }
            }
        }

        // Check for circular dependencies via DFS.
        if let Some(cycle) = self.detect_cycle() {
            errors.push(GanttError::CircularDependency(cycle));
        }

        errors
    }

    fn detect_cycle(&self) -> Option<Vec<String>> {
        let mut visited: HashSet<&str> = HashSet::new();
        let mut in_stack: HashSet<&str> = HashSet::new();
        let mut stack: Vec<&str> = Vec::new();

        for task in &self.tasks {
            if !visited.contains(task.id.as_str()) {
                if let Some(cycle) = self.dfs_cycle(
                    task.id.as_str(),
                    &mut visited,
                    &mut in_stack,
                    &mut stack,
                ) {
                    return Some(cycle);
                }
            }
        }
        None
    }

    fn dfs_cycle<'a>(
        &'a self,
        node: &'a str,
        visited: &mut HashSet<&'a str>,
        in_stack: &mut HashSet<&'a str>,
        stack: &mut Vec<&'a str>,
    ) -> Option<Vec<String>> {
        visited.insert(node);
        in_stack.insert(node);
        stack.push(node);

        if let Some(&idx) = self.task_map.get(node) {
            for dep in &self.tasks[idx].dependencies {
                if !visited.contains(dep.as_str()) {
                    if let Some(cycle) = self.dfs_cycle(dep.as_str(), visited, in_stack, stack) {
                        return Some(cycle);
                    }
                } else if in_stack.contains(dep.as_str()) {
                    // Found cycle — extract it.
                    let cycle_start = stack.iter().position(|s| *s == dep.as_str()).unwrap();
                    let mut cycle: Vec<String> =
                        stack[cycle_start..].iter().map(|s| s.to_string()).collect();
                    cycle.push(dep.clone());
                    return Some(cycle);
                }
            }
        }

        stack.pop();
        in_stack.remove(node);
        None
    }

    /// Compute critical path (longest path through dependency graph).
    /// Returns (path as task ids, total duration in days).
    pub fn critical_path(&self) -> (Vec<String>, i64) {
        // Topological sort.
        let order = match self.topological_sort() {
            Some(o) => o,
            None => return (Vec::new(), 0),
        };

        // Compute earliest finish for each task.
        let mut earliest_finish: HashMap<&str, i64> = HashMap::new();
        let mut predecessor: HashMap<&str, Option<&str>> = HashMap::new();

        for &task_id in &order {
            let task = &self.tasks[self.task_map[task_id]];
            let mut earliest_start = task.start.0;

            let mut best_pred: Option<&str> = None;
            for dep in &task.dependencies {
                if let Some(&ef) = earliest_finish.get(dep.as_str()) {
                    if ef >= earliest_start {
                        earliest_start = ef;
                        best_pred = Some(dep.as_str());
                    }
                }
            }

            let ef = earliest_start + task.duration_days();
            earliest_finish.insert(task_id, ef);
            predecessor.insert(task_id, best_pred);
        }

        // Find the task with the latest earliest finish.
        let last_task = earliest_finish
            .iter()
            .max_by_key(|(_, ef)| **ef)
            .map(|(id, ef)| (*id, *ef));

        match last_task {
            Some((last_id, total_duration)) => {
                let mut path = Vec::new();
                let mut current: Option<&str> = Some(last_id);
                while let Some(id) = current {
                    path.push(id.to_string());
                    current = predecessor.get(id).copied().flatten();
                }
                path.reverse();
                (path, total_duration)
            }
            None => (Vec::new(), 0),
        }
    }

    fn topological_sort(&self) -> Option<Vec<&str>> {
        let mut in_degree: HashMap<&str, usize> = HashMap::new();
        let mut dependents: HashMap<&str, Vec<&str>> = HashMap::new();

        for task in &self.tasks {
            in_degree.entry(task.id.as_str()).or_insert(0);
            dependents.entry(task.id.as_str()).or_default();
            for dep in &task.dependencies {
                // task depends on dep, so dep -> task
                dependents
                    .entry(dep.as_str())
                    .or_default()
                    .push(task.id.as_str());
                *in_degree.entry(task.id.as_str()).or_insert(0) += 1;
            }
        }

        let mut queue: VecDeque<&str> = VecDeque::new();
        for (&id, &deg) in &in_degree {
            if deg == 0 {
                queue.push_back(id);
            }
        }

        let mut order = Vec::new();
        while let Some(id) = queue.pop_front() {
            order.push(id);
            if let Some(deps) = dependents.get(id) {
                for dep in deps {
                    if let Some(deg) = in_degree.get_mut(dep) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push_back(dep);
                        }
                    }
                }
            }
        }

        if order.len() == self.tasks.len() {
            Some(order)
        } else {
            None // Cycle exists.
        }
    }

    /// Compute slack time for each task (latest start - earliest start).
    pub fn slack_times(&self) -> HashMap<String, i64> {
        let order = match self.topological_sort() {
            Some(o) => o,
            None => return HashMap::new(),
        };

        // Forward pass: earliest start / earliest finish.
        let mut es: HashMap<&str, i64> = HashMap::new();
        let mut ef: HashMap<&str, i64> = HashMap::new();

        for &task_id in &order {
            let task = &self.tasks[self.task_map[task_id]];
            let mut earliest = task.start.0;
            for dep in &task.dependencies {
                if let Some(&dep_ef) = ef.get(dep.as_str()) {
                    earliest = earliest.max(dep_ef);
                }
            }
            es.insert(task_id, earliest);
            ef.insert(task_id, earliest + task.duration_days());
        }

        // Backward pass: latest finish / latest start.
        let project_end = ef.values().copied().max().unwrap_or(0);
        let mut lf: HashMap<&str, i64> = HashMap::new();
        let mut ls: HashMap<&str, i64> = HashMap::new();

        // Initialize all to project end.
        for task in &self.tasks {
            lf.insert(task.id.as_str(), project_end);
        }

        // Build reverse dependency map.
        let mut reverse_deps: HashMap<&str, Vec<&str>> = HashMap::new();
        for task in &self.tasks {
            for dep in &task.dependencies {
                reverse_deps
                    .entry(dep.as_str())
                    .or_default()
                    .push(task.id.as_str());
            }
        }

        for &task_id in order.iter().rev() {
            let task = &self.tasks[self.task_map[task_id]];
            // LF = min(LS of all successors), or project_end if no successors.
            if let Some(succs) = reverse_deps.get(task_id) {
                let min_ls = succs
                    .iter()
                    .filter_map(|s| ls.get(s).copied())
                    .min()
                    .unwrap_or(project_end);
                lf.insert(task_id, min_ls);
            }
            ls.insert(task_id, lf[task_id] - task.duration_days());
        }

        // Slack = LS - ES.
        let mut slacks = HashMap::new();
        for task in &self.tasks {
            let slack = ls.get(task.id.as_str()).copied().unwrap_or(0)
                - es.get(task.id.as_str()).copied().unwrap_or(0);
            slacks.insert(task.id.clone(), slack.max(0));
        }
        slacks
    }

    /// Get all tasks assigned to a resource.
    pub fn tasks_for_resource(&self, resource: &str) -> Vec<&GanttTask> {
        self.tasks
            .iter()
            .filter(|t| t.resource.as_deref() == Some(resource))
            .collect()
    }

    /// Get all unique resources.
    pub fn resources(&self) -> Vec<String> {
        let mut res: HashSet<String> = HashSet::new();
        for task in &self.tasks {
            if let Some(r) = &task.resource {
                res.insert(r.clone());
            }
        }
        let mut result: Vec<String> = res.into_iter().collect();
        result.sort();
        result
    }

    /// Project start date.
    pub fn project_start(&self) -> Option<GanttDate> {
        self.tasks.iter().map(|t| t.start).min()
    }

    /// Project end date.
    pub fn project_end(&self) -> Option<GanttDate> {
        self.tasks.iter().map(|t| t.end).max()
    }
}

impl Default for GanttChart {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_chart() -> GanttChart {
        let mut chart = GanttChart::new();
        chart
            .add_task(
                GanttTask::new("t1", "Design", GanttDate(0), GanttDate(5))
                    .with_resource("Alice"),
            )
            .unwrap();
        chart
            .add_task(
                GanttTask::new("t2", "Implement", GanttDate(5), GanttDate(15))
                    .with_dependency("t1")
                    .with_resource("Bob"),
            )
            .unwrap();
        chart
            .add_task(
                GanttTask::new("t3", "Test", GanttDate(5), GanttDate(10))
                    .with_dependency("t1")
                    .with_resource("Alice"),
            )
            .unwrap();
        chart
            .add_task(
                GanttTask::new("t4", "Deploy", GanttDate(15), GanttDate(17))
                    .with_dependency("t2")
                    .with_dependency("t3")
                    .with_resource("Bob"),
            )
            .unwrap();
        chart
            .add_task(GanttTask::milestone("m1", "Release", GanttDate(17)).with_dependency("t4"))
            .unwrap();
        chart
    }

    #[test]
    fn test_add_tasks() {
        let chart = sample_chart();
        assert_eq!(chart.tasks().len(), 5);
    }

    #[test]
    fn test_duplicate_id() {
        let mut chart = GanttChart::new();
        chart
            .add_task(GanttTask::new("t1", "A", GanttDate(0), GanttDate(1)))
            .unwrap();
        let result = chart.add_task(GanttTask::new("t1", "B", GanttDate(0), GanttDate(1)));
        assert!(matches!(result, Err(GanttError::DuplicateTaskId(_))));
    }

    #[test]
    fn test_validate_valid() {
        let chart = sample_chart();
        let errors = chart.validate();
        assert!(errors.is_empty(), "expected no errors, got: {errors:?}");
    }

    #[test]
    fn test_validate_end_before_start() {
        let mut chart = GanttChart::new();
        chart
            .add_task(GanttTask::new("t1", "Bad", GanttDate(10), GanttDate(5)))
            .unwrap();
        let errors = chart.validate();
        assert!(errors.iter().any(|e| matches!(e, GanttError::EndBeforeStart(_))));
    }

    #[test]
    fn test_validate_circular_dependency() {
        let mut chart = GanttChart::new();
        chart
            .add_task(
                GanttTask::new("a", "A", GanttDate(0), GanttDate(1)).with_dependency("b"),
            )
            .unwrap();
        chart
            .add_task(
                GanttTask::new("b", "B", GanttDate(0), GanttDate(1)).with_dependency("a"),
            )
            .unwrap();
        let errors = chart.validate();
        assert!(errors
            .iter()
            .any(|e| matches!(e, GanttError::CircularDependency(_))));
    }

    #[test]
    fn test_critical_path() {
        let chart = sample_chart();
        let (path, duration) = chart.critical_path();
        assert!(!path.is_empty());
        // Critical path should go through the longest chain.
        assert!(path.contains(&"t1".to_string()));
        assert!(path.contains(&"t2".to_string()));
        assert!(duration > 0);
    }

    #[test]
    fn test_slack_times() {
        let chart = sample_chart();
        let slacks = chart.slack_times();
        // Tasks on the critical path should have 0 slack.
        // t3 (5 days) is parallel to t2 (10 days), so t3 has slack.
        assert!(slacks["t3"] > 0);
    }

    #[test]
    fn test_milestone() {
        let chart = sample_chart();
        let m1 = chart.find_task("m1").unwrap();
        assert!(m1.milestone);
        assert_eq!(m1.duration_days(), 0);
    }

    #[test]
    fn test_resources() {
        let chart = sample_chart();
        let resources = chart.resources();
        assert_eq!(resources.len(), 2);
        assert!(resources.contains(&"Alice".to_string()));
        assert!(resources.contains(&"Bob".to_string()));
    }

    #[test]
    fn test_tasks_for_resource() {
        let chart = sample_chart();
        let alice_tasks = chart.tasks_for_resource("Alice");
        assert_eq!(alice_tasks.len(), 2);
    }

    #[test]
    fn test_project_dates() {
        let chart = sample_chart();
        assert_eq!(chart.project_start(), Some(GanttDate(0)));
        assert_eq!(chart.project_end(), Some(GanttDate(17)));
    }

    #[test]
    fn test_task_duration() {
        let t = GanttTask::new("t", "Task", GanttDate(3), GanttDate(10));
        assert_eq!(t.duration_days(), 7);
    }

    #[test]
    fn test_time_scale() {
        let chart = GanttChart::new().with_scale(TimeScale::Weeks);
        assert_eq!(chart.scale(), TimeScale::Weeks);
    }
}
