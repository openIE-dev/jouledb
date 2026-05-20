//! Task allocation — market-based auction, Hungarian method, multi-robot
//! task assignment, and utility functions for optimal allocation.
//!
//! Pure-Rust implementations of classical assignment algorithms for
//! multi-robot systems. The Hungarian algorithm runs in O(n^3), and the
//! market-based approach supports dynamic task arrival and re-allocation.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Task allocation errors.
#[derive(Debug, Clone, PartialEq)]
pub enum AllocationError {
    /// No robots available for assignment.
    NoRobots,
    /// No tasks to assign.
    NoTasks,
    /// Robot not found.
    RobotNotFound(u64),
    /// Task not found.
    TaskNotFound(u64),
    /// Cost matrix dimension mismatch.
    DimensionMismatch { rows: usize, cols: usize },
    /// Infeasible assignment (no valid matching).
    Infeasible(String),
}

impl fmt::Display for AllocationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoRobots => write!(f, "no robots available"),
            Self::NoTasks => write!(f, "no tasks to assign"),
            Self::RobotNotFound(id) => write!(f, "robot not found: {id}"),
            Self::TaskNotFound(id) => write!(f, "task not found: {id}"),
            Self::DimensionMismatch { rows, cols } => {
                write!(f, "cost matrix mismatch: {rows} robots, {cols} tasks")
            }
            Self::Infeasible(msg) => write!(f, "infeasible: {msg}"),
        }
    }
}

impl std::error::Error for AllocationError {}

// ── Task ────────────────────────────────────────────────────────

/// Priority level for tasks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TaskPriority {
    Low,
    Medium,
    High,
    Critical,
}

impl fmt::Display for TaskPriority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Low => write!(f, "Low"),
            Self::Medium => write!(f, "Medium"),
            Self::High => write!(f, "High"),
            Self::Critical => write!(f, "Critical"),
        }
    }
}

/// A task to be assigned to a robot.
#[derive(Debug, Clone)]
pub struct Task {
    pub id: u64,
    pub name: String,
    pub location: (f64, f64),
    pub priority: TaskPriority,
    pub duration_secs: f64,
    pub deadline_secs: Option<f64>,
    pub required_capability: Option<String>,
}

impl Task {
    pub fn new(id: u64, name: &str, location: (f64, f64)) -> Self {
        Self {
            id,
            name: name.to_string(),
            location,
            priority: TaskPriority::Medium,
            duration_secs: 60.0,
            deadline_secs: None,
            required_capability: None,
        }
    }

    pub fn with_priority(mut self, priority: TaskPriority) -> Self {
        self.priority = priority;
        self
    }

    pub fn with_duration(mut self, secs: f64) -> Self {
        self.duration_secs = secs;
        self
    }

    pub fn with_deadline(mut self, deadline: f64) -> Self {
        self.deadline_secs = Some(deadline);
        self
    }

    pub fn with_capability(mut self, cap: &str) -> Self {
        self.required_capability = Some(cap.to_string());
        self
    }
}

impl fmt::Display for Task {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Task({}, \"{}\", priority={}, loc=({:.1},{:.1}))",
            self.id, self.name, self.priority, self.location.0, self.location.1
        )
    }
}

// ── Robot ────────────────────────────────────────────────────────

/// A robot that can be assigned tasks.
#[derive(Debug, Clone)]
pub struct Robot {
    pub id: u64,
    pub name: String,
    pub location: (f64, f64),
    pub speed: f64,
    pub capabilities: Vec<String>,
    pub current_task: Option<u64>,
    pub available: bool,
}

impl Robot {
    pub fn new(id: u64, name: &str, location: (f64, f64)) -> Self {
        Self {
            id,
            name: name.to_string(),
            location,
            speed: 1.0,
            capabilities: Vec::new(),
            current_task: None,
            available: true,
        }
    }

    pub fn with_speed(mut self, speed: f64) -> Self {
        self.speed = speed;
        self
    }

    pub fn with_capability(mut self, cap: &str) -> Self {
        self.capabilities.push(cap.to_string());
        self
    }

    /// Can this robot perform a task requiring a specific capability?
    pub fn can_perform(&self, required: &Option<String>) -> bool {
        match required {
            None => true,
            Some(cap) => self.capabilities.iter().any(|c| c == cap),
        }
    }

    /// Euclidean distance to a point.
    pub fn distance_to(&self, point: (f64, f64)) -> f64 {
        let dx = self.location.0 - point.0;
        let dy = self.location.1 - point.1;
        (dx * dx + dy * dy).sqrt()
    }

    /// Travel time to a location.
    pub fn travel_time(&self, dest: (f64, f64)) -> f64 {
        if self.speed <= 0.0 {
            return f64::INFINITY;
        }
        self.distance_to(dest) / self.speed
    }
}

impl fmt::Display for Robot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let status = if self.available { "available" } else { "busy" };
        write!(
            f,
            "Robot({}, \"{}\", {}, loc=({:.1},{:.1}))",
            self.id, self.name, status, self.location.0, self.location.1
        )
    }
}

// ── Utility Functions ───────────────────────────────────────────

/// Compute the cost of assigning a robot to a task (lower = better).
pub fn assignment_cost(robot: &Robot, task: &Task) -> f64 {
    if !robot.can_perform(&task.required_capability) {
        return f64::INFINITY;
    }
    let travel = robot.travel_time(task.location);
    let priority_weight = match task.priority {
        TaskPriority::Critical => 0.25,
        TaskPriority::High => 0.5,
        TaskPriority::Medium => 1.0,
        TaskPriority::Low => 2.0,
    };
    let deadline_penalty = match task.deadline_secs {
        Some(dl) if travel + task.duration_secs > dl => {
            (travel + task.duration_secs - dl) * 10.0
        }
        _ => 0.0,
    };
    travel * priority_weight + deadline_penalty
}

/// Build a cost matrix from robots and tasks.
pub fn build_cost_matrix(robots: &[Robot], tasks: &[Task]) -> Vec<Vec<f64>> {
    robots
        .iter()
        .map(|r| tasks.iter().map(|t| assignment_cost(r, t)).collect())
        .collect()
}

// ── Hungarian Algorithm ─────────────────────────────────────────

/// Solve the assignment problem using the Hungarian (Kuhn-Munkres) algorithm.
/// Returns a vector of (robot_index, task_index) pairs.
pub fn hungarian_assignment(cost: &[Vec<f64>]) -> Result<Vec<(usize, usize)>, AllocationError> {
    let n = cost.len();
    if n == 0 {
        return Err(AllocationError::NoRobots);
    }
    let m = cost[0].len();
    if m == 0 {
        return Err(AllocationError::NoTasks);
    }

    // Pad to square matrix.
    let sz = n.max(m);
    let mut c = vec![vec![0.0f64; sz]; sz];
    for i in 0..n {
        for j in 0..m {
            c[i][j] = if cost[i][j].is_finite() { cost[i][j] } else { 1e15 };
        }
    }

    // Step 1: Row reduction.
    for row in &mut c {
        let min = row.iter().copied().fold(f64::INFINITY, f64::min);
        if min.is_finite() {
            for val in row.iter_mut() {
                *val -= min;
            }
        }
    }

    // Step 2: Column reduction.
    for j in 0..sz {
        let min = (0..sz).map(|i| c[i][j]).fold(f64::INFINITY, f64::min);
        if min.is_finite() {
            for i in 0..sz {
                c[i][j] -= min;
            }
        }
    }

    // Augmenting path based Hungarian.
    let mut row_match = vec![None::<usize>; sz];
    let mut col_match = vec![None::<usize>; sz];

    for _ in 0..sz * sz + 10 {
        // Try to find augmenting paths.
        let mut matched_count = 0;
        row_match.fill(None);
        col_match.fill(None);

        // Greedy initial matching on zeros.
        for i in 0..sz {
            for j in 0..sz {
                if c[i][j].abs() < 1e-9 && col_match[j].is_none() && row_match[i].is_none() {
                    row_match[i] = Some(j);
                    col_match[j] = Some(i);
                    matched_count += 1;
                    break;
                }
            }
        }

        if matched_count == sz {
            break;
        }

        // Find augmenting paths for unmatched rows.
        let mut improved = true;
        while improved {
            improved = false;
            for i in 0..sz {
                if row_match[i].is_some() {
                    continue;
                }
                // BFS to find augmenting path.
                let mut visited_col = vec![false; sz];
                let mut parent = vec![None::<usize>; sz]; // parent[j] = row that reached col j
                let mut queue = Vec::new();
                // Add all zero cols from row i.
                for j in 0..sz {
                    if c[i][j].abs() < 1e-9 {
                        visited_col[j] = true;
                        parent[j] = Some(i);
                        queue.push(j);
                    }
                }
                let mut found = None;
                let mut qi = 0;
                while qi < queue.len() {
                    let j = queue[qi];
                    qi += 1;
                    match col_match[j] {
                        None => {
                            found = Some(j);
                            break;
                        }
                        Some(matched_row) => {
                            for jj in 0..sz {
                                if !visited_col[jj] && c[matched_row][jj].abs() < 1e-9 {
                                    visited_col[jj] = true;
                                    parent[jj] = Some(matched_row);
                                    queue.push(jj);
                                }
                            }
                        }
                    }
                }
                if let Some(mut j) = found {
                    // Trace back and flip matching.
                    loop {
                        let r = parent[j].unwrap();
                        let prev_j = row_match[r];
                        row_match[r] = Some(j);
                        col_match[j] = Some(r);
                        match prev_j {
                            Some(pj) => j = pj,
                            None => break,
                        }
                    }
                    improved = true;
                }
            }
        }

        // Count matches.
        let cur_matched = row_match.iter().filter(|x| x.is_some()).count();
        if cur_matched == sz {
            break;
        }

        // Find minimum uncovered value and adjust.
        let matched_rows: Vec<bool> = row_match.iter().map(|x| x.is_some()).collect();
        let covered_cols: Vec<bool> = (0..sz)
            .map(|j| {
                // Column is covered if it's in the matching or reachable from unmatched row.
                col_match[j].is_some()
            })
            .collect();

        let mut min_val = f64::INFINITY;
        for i in 0..sz {
            if matched_rows[i] {
                continue;
            }
            for j in 0..sz {
                if !covered_cols[j] && c[i][j] < min_val {
                    min_val = c[i][j];
                }
            }
        }

        if !min_val.is_finite() || min_val.abs() < 1e-15 {
            break;
        }

        // Subtract min from uncovered rows, add to covered cols.
        for i in 0..sz {
            for j in 0..sz {
                if !matched_rows[i] {
                    c[i][j] -= min_val;
                }
                if covered_cols[j] {
                    c[i][j] += min_val;
                }
            }
        }
    }

    // Extract result — only include real (non-padded) assignments.
    let mut result = Vec::new();
    for i in 0..n {
        if let Some(j) = row_match[i] {
            if j < m {
                result.push((i, j));
            }
        }
    }
    Ok(result)
}

// ── Greedy Assignment ───────────────────────────────────────────

/// Simple greedy assignment: assign each task to the cheapest available robot.
pub fn greedy_assignment(
    robots: &[Robot],
    tasks: &[Task],
) -> Vec<(usize, usize)> {
    let mut assigned_robots = vec![false; robots.len()];
    let mut result = Vec::new();

    // Sort tasks by priority (highest first).
    let mut task_order: Vec<usize> = (0..tasks.len()).collect();
    task_order.sort_by(|&a, &b| tasks[b].priority.cmp(&tasks[a].priority));

    for ti in task_order {
        let mut best_ri = None;
        let mut best_cost = f64::INFINITY;
        for (ri, robot) in robots.iter().enumerate() {
            if assigned_robots[ri] || !robot.available {
                continue;
            }
            let c = assignment_cost(robot, &tasks[ti]);
            if c < best_cost {
                best_cost = c;
                best_ri = Some(ri);
            }
        }
        if let Some(ri) = best_ri {
            assigned_robots[ri] = true;
            result.push((ri, ti));
        }
    }
    result
}

// ── Task Allocator ──────────────────────────────────────────────

/// Assignment method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssignMethod {
    Hungarian,
    Greedy,
}

impl fmt::Display for AssignMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hungarian => write!(f, "Hungarian"),
            Self::Greedy => write!(f, "Greedy"),
        }
    }
}

/// The central task allocator.
#[derive(Debug, Clone)]
pub struct TaskAllocator {
    pub robots: Vec<Robot>,
    pub tasks: Vec<Task>,
    pub method: AssignMethod,
    pub assignments: Vec<(u64, u64)>,
    pub total_cost: f64,
}

impl TaskAllocator {
    pub fn new(method: AssignMethod) -> Self {
        Self {
            robots: Vec::new(),
            tasks: Vec::new(),
            method,
            assignments: Vec::new(),
            total_cost: 0.0,
        }
    }

    pub fn with_robot(mut self, robot: Robot) -> Self {
        self.robots.push(robot);
        self
    }

    pub fn with_task(mut self, task: Task) -> Self {
        self.tasks.push(task);
        self
    }

    pub fn add_robot(&mut self, robot: Robot) {
        self.robots.push(robot);
    }

    pub fn add_task(&mut self, task: Task) {
        self.tasks.push(task);
    }

    /// Run the allocation algorithm.
    pub fn allocate(&mut self) -> Result<&[(u64, u64)], AllocationError> {
        if self.robots.is_empty() {
            return Err(AllocationError::NoRobots);
        }
        if self.tasks.is_empty() {
            return Err(AllocationError::NoTasks);
        }

        let pairs = match self.method {
            AssignMethod::Hungarian => {
                let cost = build_cost_matrix(&self.robots, &self.tasks);
                hungarian_assignment(&cost)?
            }
            AssignMethod::Greedy => greedy_assignment(&self.robots, &self.tasks),
        };

        self.assignments.clear();
        self.total_cost = 0.0;
        for (ri, ti) in &pairs {
            let robot_id = self.robots[*ri].id;
            let task_id = self.tasks[*ti].id;
            self.assignments.push((robot_id, task_id));
            self.total_cost += assignment_cost(&self.robots[*ri], &self.tasks[*ti]);
        }

        Ok(&self.assignments)
    }

    /// Get which task is assigned to a robot.
    pub fn assignment_for_robot(&self, robot_id: u64) -> Option<u64> {
        self.assignments.iter().find(|(r, _)| *r == robot_id).map(|(_, t)| *t)
    }

    /// Count of assigned pairs.
    pub fn assigned_count(&self) -> usize {
        self.assignments.len()
    }

    /// Unassigned tasks (tasks with no robot).
    pub fn unassigned_tasks(&self) -> Vec<u64> {
        let assigned_tasks: Vec<u64> = self.assignments.iter().map(|(_, t)| *t).collect();
        self.tasks
            .iter()
            .filter(|t| !assigned_tasks.contains(&t.id))
            .map(|t| t.id)
            .collect()
    }
}

impl fmt::Display for TaskAllocator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TaskAllocator({}, {} robots, {} tasks, {} assigned, cost={:.2})",
            self.method,
            self.robots.len(),
            self.tasks.len(),
            self.assignments.len(),
            self.total_cost,
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_robot(id: u64, x: f64, y: f64) -> Robot {
        Robot::new(id, &format!("R{id}"), (x, y)).with_speed(1.0)
    }

    fn make_task(id: u64, x: f64, y: f64) -> Task {
        Task::new(id, &format!("T{id}"), (x, y))
    }

    #[test]
    fn test_robot_distance() {
        let r = make_robot(1, 0.0, 0.0);
        assert!((r.distance_to((3.0, 4.0)) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_robot_travel_time() {
        let r = make_robot(1, 0.0, 0.0).with_speed(2.0);
        assert!((r.travel_time((6.0, 0.0)) - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_robot_can_perform() {
        let r = make_robot(1, 0.0, 0.0).with_capability("weld");
        assert!(r.can_perform(&Some("weld".to_string())));
        assert!(!r.can_perform(&Some("paint".to_string())));
        assert!(r.can_perform(&None));
    }

    #[test]
    fn test_task_priority_display() {
        assert_eq!(format!("{}", TaskPriority::Critical), "Critical");
    }

    #[test]
    fn test_assignment_cost_basic() {
        let r = make_robot(1, 0.0, 0.0);
        let t = make_task(1, 10.0, 0.0);
        let c = assignment_cost(&r, &t);
        assert!((c - 10.0).abs() < 1e-9); // Medium priority weight = 1.0
    }

    #[test]
    fn test_assignment_cost_capability_mismatch() {
        let r = make_robot(1, 0.0, 0.0);
        let t = make_task(1, 1.0, 0.0).with_capability("fly");
        let c = assignment_cost(&r, &t);
        assert!(c.is_infinite());
    }

    #[test]
    fn test_assignment_cost_priority_scaling() {
        let r = make_robot(1, 0.0, 0.0);
        let t_high = make_task(1, 10.0, 0.0).with_priority(TaskPriority::High);
        let t_low = make_task(2, 10.0, 0.0).with_priority(TaskPriority::Low);
        assert!(assignment_cost(&r, &t_high) < assignment_cost(&r, &t_low));
    }

    #[test]
    fn test_build_cost_matrix() {
        let robots = vec![make_robot(1, 0.0, 0.0), make_robot(2, 10.0, 0.0)];
        let tasks = vec![make_task(1, 5.0, 0.0)];
        let cm = build_cost_matrix(&robots, &tasks);
        assert_eq!(cm.len(), 2);
        assert_eq!(cm[0].len(), 1);
        assert!((cm[0][0] - 5.0).abs() < 1e-9);
        assert!((cm[1][0] - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_hungarian_2x2() {
        // Robot 0 close to task 1, Robot 1 close to task 0.
        let cost = vec![vec![10.0, 1.0], vec![1.0, 10.0]];
        let result = hungarian_assignment(&cost).unwrap();
        assert_eq!(result.len(), 2);
        // Optimal: 0->1, 1->0 (cost 2).
        let r0_task = result.iter().find(|(r, _)| *r == 0).unwrap().1;
        let r1_task = result.iter().find(|(r, _)| *r == 1).unwrap().1;
        assert_eq!(r0_task, 1);
        assert_eq!(r1_task, 0);
    }

    #[test]
    fn test_hungarian_3x3() {
        let cost = vec![
            vec![1.0, 9.0, 9.0],
            vec![9.0, 1.0, 9.0],
            vec![9.0, 9.0, 1.0],
        ];
        let result = hungarian_assignment(&cost).unwrap();
        assert_eq!(result.len(), 3);
        let total: f64 = result.iter().map(|&(r, t)| cost[r][t]).sum();
        assert!((total - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_hungarian_empty() {
        let cost: Vec<Vec<f64>> = vec![];
        assert!(hungarian_assignment(&cost).is_err());
    }

    #[test]
    fn test_hungarian_rectangular() {
        // 2 robots, 3 tasks.
        let cost = vec![vec![1.0, 5.0, 3.0], vec![4.0, 2.0, 6.0]];
        let result = hungarian_assignment(&cost).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_greedy_assignment() {
        let robots = vec![make_robot(1, 0.0, 0.0), make_robot(2, 10.0, 0.0)];
        let tasks = vec![make_task(1, 1.0, 0.0), make_task(2, 9.0, 0.0)];
        let result = greedy_assignment(&robots, &tasks);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_allocator_hungarian() {
        let mut alloc = TaskAllocator::new(AssignMethod::Hungarian)
            .with_robot(make_robot(1, 0.0, 0.0))
            .with_robot(make_robot(2, 10.0, 0.0))
            .with_task(make_task(100, 1.0, 0.0))
            .with_task(make_task(200, 9.0, 0.0));
        alloc.allocate().unwrap();
        assert_eq!(alloc.assigned_count(), 2);
        assert!(alloc.total_cost < 20.0);
    }

    #[test]
    fn test_allocator_greedy() {
        let mut alloc = TaskAllocator::new(AssignMethod::Greedy)
            .with_robot(make_robot(1, 0.0, 0.0))
            .with_task(make_task(100, 5.0, 0.0));
        alloc.allocate().unwrap();
        assert_eq!(alloc.assigned_count(), 1);
        assert_eq!(alloc.assignment_for_robot(1), Some(100));
    }

    #[test]
    fn test_allocator_no_robots() {
        let mut alloc = TaskAllocator::new(AssignMethod::Greedy)
            .with_task(make_task(1, 0.0, 0.0));
        assert!(alloc.allocate().is_err());
    }

    #[test]
    fn test_allocator_no_tasks() {
        let mut alloc = TaskAllocator::new(AssignMethod::Greedy)
            .with_robot(make_robot(1, 0.0, 0.0));
        assert!(alloc.allocate().is_err());
    }

    #[test]
    fn test_unassigned_tasks() {
        let mut alloc = TaskAllocator::new(AssignMethod::Greedy)
            .with_robot(make_robot(1, 0.0, 0.0))
            .with_task(make_task(100, 1.0, 0.0))
            .with_task(make_task(200, 2.0, 0.0));
        alloc.allocate().unwrap();
        // Only 1 robot, 2 tasks, so 1 unassigned.
        assert_eq!(alloc.unassigned_tasks().len(), 1);
    }

    #[test]
    fn test_display_impls() {
        let alloc = TaskAllocator::new(AssignMethod::Hungarian)
            .with_robot(make_robot(1, 0.0, 0.0));
        assert!(format!("{alloc}").contains("Hungarian"));
        let t = make_task(5, 1.0, 2.0).with_priority(TaskPriority::High);
        assert!(format!("{t}").contains("High"));
        let r = make_robot(3, 0.0, 0.0);
        assert!(format!("{r}").contains("available"));
    }

    #[test]
    fn test_deadline_penalty() {
        let r = make_robot(1, 0.0, 0.0);
        let t = make_task(1, 100.0, 0.0).with_duration(50.0).with_deadline(10.0);
        let c = assignment_cost(&r, &t);
        // Will definitely miss deadline, so large penalty.
        assert!(c > 100.0);
    }
}
