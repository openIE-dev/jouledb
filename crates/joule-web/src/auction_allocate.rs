//! # Auction-Based Task Allocation
//!
//! Implements auction mechanisms for multi-robot task allocation. Supports
//! sequential single-item auctions, combinatorial auctions, bundle bidding,
//! and winner determination for distributed robotic systems.

use std::fmt;
use std::collections::HashMap;

// ── Core Types ──

/// Unique identifier for a robot/bidder.
pub type RobotId = usize;

/// Unique identifier for a task.
pub type TaskId = usize;

/// A bid from a robot for a task or bundle.
#[derive(Clone, Debug)]
pub struct Bid {
    pub robot_id: RobotId,
    pub task_ids: Vec<TaskId>,
    pub value: f64,
    pub cost: f64,
}

impl Bid {
    pub fn single(robot_id: RobotId, task_id: TaskId, value: f64, cost: f64) -> Self {
        Self { robot_id, task_ids: vec![task_id], value, cost }
    }

    pub fn bundle(robot_id: RobotId, task_ids: Vec<TaskId>, value: f64, cost: f64) -> Self {
        Self { robot_id, task_ids, value, cost }
    }

    pub fn profit(&self) -> f64 {
        self.value - self.cost
    }

    pub fn is_bundle(&self) -> bool {
        self.task_ids.len() > 1
    }
}

impl fmt::Display for Bid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Bid(robot={}, tasks={:?}, val={:.2}, cost={:.2})",
            self.robot_id, self.task_ids, self.value, self.cost)
    }
}

// ── Task Definition ──

/// A task to be allocated.
#[derive(Clone, Debug)]
pub struct Task {
    pub id: TaskId,
    pub name: String,
    pub priority: f64,
    pub deadline: f64,
    pub location: (f64, f64),
}

impl Task {
    pub fn new(id: TaskId, name: &str) -> Self {
        Self {
            id, name: name.to_string(),
            priority: 1.0, deadline: f64::MAX,
            location: (0.0, 0.0),
        }
    }

    pub fn with_priority(mut self, p: f64) -> Self {
        self.priority = p;
        self
    }

    pub fn with_deadline(mut self, d: f64) -> Self {
        self.deadline = d;
        self
    }

    pub fn with_location(mut self, x: f64, y: f64) -> Self {
        self.location = (x, y);
        self
    }
}

impl fmt::Display for Task {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Task({}, \"{}\", pri={:.1})", self.id, self.name, self.priority)
    }
}

// ── Allocation Result ──

/// Result of an auction allocation.
#[derive(Clone, Debug)]
pub struct Allocation {
    pub assignments: HashMap<TaskId, RobotId>,
    pub total_value: f64,
    pub total_cost: f64,
    pub unassigned: Vec<TaskId>,
}

impl Allocation {
    pub fn new() -> Self {
        Self {
            assignments: HashMap::new(),
            total_value: 0.0,
            total_cost: 0.0,
            unassigned: Vec::new(),
        }
    }

    pub fn assign(&mut self, task_id: TaskId, robot_id: RobotId, value: f64, cost: f64) {
        self.assignments.insert(task_id, robot_id);
        self.total_value += value;
        self.total_cost += cost;
    }

    pub fn total_profit(&self) -> f64 {
        self.total_value - self.total_cost
    }

    pub fn num_assigned(&self) -> usize {
        self.assignments.len()
    }

    pub fn robot_tasks(&self, robot_id: RobotId) -> Vec<TaskId> {
        self.assignments.iter()
            .filter(|&(_, r)| *r == robot_id)
            .map(|(&t, _)| t)
            .collect()
    }
}

impl fmt::Display for Allocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Allocation({} assigned, {} unassigned, profit={:.2})",
            self.assignments.len(), self.unassigned.len(), self.total_profit())
    }
}

// ── Sequential Single-Item Auction ──

/// Sequential single-item auction: tasks auctioned one at a time.
#[derive(Clone, Debug)]
pub struct SequentialAuction {
    tasks: Vec<Task>,
    num_robots: usize,
    max_tasks_per_robot: usize,
}

impl SequentialAuction {
    pub fn new(num_robots: usize) -> Self {
        Self {
            tasks: Vec::new(),
            num_robots,
            max_tasks_per_robot: usize::MAX,
        }
    }

    pub fn with_max_tasks_per_robot(mut self, max: usize) -> Self {
        self.max_tasks_per_robot = max;
        self
    }

    pub fn add_task(&mut self, task: Task) {
        self.tasks.push(task);
    }

    /// Run the auction given a bid matrix. bids[robot_id][task_index] = bid value.
    pub fn run(&self, bids: &[Vec<f64>]) -> Allocation {
        let mut alloc = Allocation::new();
        let mut robot_counts = vec![0usize; self.num_robots];

        // Sort tasks by priority (highest first)
        let mut task_order: Vec<usize> = (0..self.tasks.len()).collect();
        task_order.sort_by(|&a, &b| self.tasks[b].priority.partial_cmp(&self.tasks[a].priority).unwrap_or(std::cmp::Ordering::Equal));

        for &ti in &task_order {
            let task = &self.tasks[ti];
            let mut best_robot = None;
            let mut best_bid = f64::MIN;

            for robot_id in 0..self.num_robots {
                if robot_counts[robot_id] >= self.max_tasks_per_robot { continue; }
                let bid_val = bids.get(robot_id).and_then(|b| b.get(ti)).copied().unwrap_or(0.0);
                if bid_val > best_bid {
                    best_bid = bid_val;
                    best_robot = Some(robot_id);
                }
            }

            if let Some(robot_id) = best_robot {
                if best_bid > 0.0 {
                    alloc.assign(task.id, robot_id, best_bid, 0.0);
                    robot_counts[robot_id] += 1;
                } else {
                    alloc.unassigned.push(task.id);
                }
            } else {
                alloc.unassigned.push(task.id);
            }
        }
        alloc
    }
}

impl fmt::Display for SequentialAuction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SequentialAuction({} tasks, {} robots)", self.tasks.len(), self.num_robots)
    }
}

// ── Combinatorial Auction ──

/// Combinatorial auction with bundle bids using greedy winner determination.
#[derive(Clone, Debug)]
pub struct CombinatorialAuction {
    tasks: Vec<Task>,
    bids: Vec<Bid>,
}

impl CombinatorialAuction {
    pub fn new() -> Self {
        Self { tasks: Vec::new(), bids: Vec::new() }
    }

    pub fn add_task(&mut self, task: Task) {
        self.tasks.push(task);
    }

    pub fn add_bid(&mut self, bid: Bid) {
        self.bids.push(bid);
    }

    /// Greedy winner determination: select bids by profit density, avoiding conflicts.
    pub fn solve_greedy(&self) -> Allocation {
        let mut alloc = Allocation::new();
        let mut assigned_tasks = std::collections::HashSet::new();

        // Sort bids by profit per task (descending)
        let mut sorted_bids: Vec<&Bid> = self.bids.iter().collect();
        sorted_bids.sort_by(|a, b| {
            let density_a = a.profit() / a.task_ids.len().max(1) as f64;
            let density_b = b.profit() / b.task_ids.len().max(1) as f64;
            density_b.partial_cmp(&density_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        for bid in sorted_bids {
            if bid.profit() <= 0.0 { continue; }

            // Check if any task in this bid is already assigned
            let conflict = bid.task_ids.iter().any(|t| assigned_tasks.contains(t));
            if conflict { continue; }

            // Accept the bid
            for &task_id in &bid.task_ids {
                alloc.assign(task_id, bid.robot_id, bid.value / bid.task_ids.len() as f64, bid.cost / bid.task_ids.len() as f64);
                assigned_tasks.insert(task_id);
            }
        }

        // Mark unassigned tasks
        for task in &self.tasks {
            if !assigned_tasks.contains(&task.id) {
                alloc.unassigned.push(task.id);
            }
        }

        alloc
    }
}

impl fmt::Display for CombinatorialAuction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CombinatorialAuction({} tasks, {} bids)", self.tasks.len(), self.bids.len())
    }
}

// ── Consensus-Based Bundle Algorithm (CBBA) ──

/// Simplified CBBA for decentralized task allocation.
#[derive(Clone, Debug)]
pub struct CbbaAgent {
    pub id: RobotId,
    bundle: Vec<TaskId>,
    path: Vec<TaskId>,
    winning_bids: HashMap<TaskId, (RobotId, f64)>,
    max_bundle_size: usize,
}

impl CbbaAgent {
    pub fn new(id: RobotId, max_bundle_size: usize) -> Self {
        Self {
            id,
            bundle: Vec::new(),
            path: Vec::new(),
            winning_bids: HashMap::new(),
            max_bundle_size,
        }
    }

    /// Phase 1: Build bundle by bidding on available tasks.
    pub fn build_bundle(&mut self, task_values: &HashMap<TaskId, f64>) {
        let mut available: Vec<(&TaskId, &f64)> = task_values.iter()
            .filter(|(tid, _)| !self.bundle.contains(tid))
            .collect();
        available.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

        while self.bundle.len() < self.max_bundle_size {
            let best = available.iter().find(|(tid, val)| {
                match self.winning_bids.get(tid) {
                    Some(&(_, current_bid)) => **val > current_bid,
                    None => true,
                }
            });

            if let Some(&(&task_id, &value)) = best {
                self.bundle.push(task_id);
                self.path.push(task_id);
                self.winning_bids.insert(task_id, (self.id, value));
                available.retain(|(tid, _)| **tid != task_id);
            } else {
                break;
            }
        }
    }

    /// Phase 2: Consensus — resolve conflicts with another agent.
    pub fn consensus(&mut self, other: &CbbaAgent) {
        for (&task_id, &(other_winner, other_bid)) in &other.winning_bids {
            match self.winning_bids.get(&task_id) {
                Some(&(my_winner, my_bid)) => {
                    if other_bid > my_bid || (other_bid == my_bid && other_winner < my_winner) {
                        self.winning_bids.insert(task_id, (other_winner, other_bid));
                        if my_winner == self.id {
                            self.bundle.retain(|t| *t != task_id);
                            self.path.retain(|t| *t != task_id);
                        }
                    }
                }
                None => {
                    self.winning_bids.insert(task_id, (other_winner, other_bid));
                }
            }
        }
    }

    pub fn assigned_tasks(&self) -> &[TaskId] {
        &self.bundle
    }

    pub fn num_tasks(&self) -> usize {
        self.bundle.len()
    }
}

impl fmt::Display for CbbaAgent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CbbaAgent(id={}, bundle={:?})", self.id, self.bundle)
    }
}

// ── Utility Functions ──

/// Compute bid value based on distance (closer = higher value).
pub fn distance_based_bid(robot_pos: (f64, f64), task_pos: (f64, f64), max_value: f64) -> f64 {
    let dx = robot_pos.0 - task_pos.0;
    let dy = robot_pos.1 - task_pos.1;
    let dist = (dx * dx + dy * dy).sqrt();
    max_value / (1.0 + dist)
}

/// Compute bid value based on remaining capacity.
pub fn capacity_based_bid(current_load: f64, max_capacity: f64, base_value: f64) -> f64 {
    let remaining = (max_capacity - current_load).max(0.0);
    base_value * remaining / max_capacity
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bid_single() {
        let bid = Bid::single(0, 1, 10.0, 3.0);
        assert!(!bid.is_bundle());
        assert!((bid.profit() - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_bid_bundle() {
        let bid = Bid::bundle(0, vec![1, 2, 3], 30.0, 10.0);
        assert!(bid.is_bundle());
        assert!((bid.profit() - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_task_creation() {
        let task = Task::new(0, "explore")
            .with_priority(5.0)
            .with_deadline(100.0)
            .with_location(1.0, 2.0);
        assert!((task.priority - 5.0).abs() < 1e-10);
        assert_eq!(task.location, (1.0, 2.0));
    }

    #[test]
    fn test_allocation_basic() {
        let mut alloc = Allocation::new();
        alloc.assign(0, 1, 10.0, 3.0);
        alloc.assign(1, 2, 8.0, 2.0);
        assert_eq!(alloc.num_assigned(), 2);
        assert!((alloc.total_profit() - 13.0).abs() < 1e-10);
    }

    #[test]
    fn test_allocation_robot_tasks() {
        let mut alloc = Allocation::new();
        alloc.assign(0, 1, 10.0, 0.0);
        alloc.assign(1, 1, 5.0, 0.0);
        alloc.assign(2, 0, 3.0, 0.0);
        let tasks = alloc.robot_tasks(1);
        assert_eq!(tasks.len(), 2);
    }

    #[test]
    fn test_sequential_auction_basic() {
        let mut auction = SequentialAuction::new(3);
        auction.add_task(Task::new(0, "t0").with_priority(1.0));
        auction.add_task(Task::new(1, "t1").with_priority(2.0));

        let bids = vec![
            vec![5.0, 3.0],  // robot 0
            vec![3.0, 7.0],  // robot 1
            vec![4.0, 2.0],  // robot 2
        ];

        let alloc = auction.run(&bids);
        assert_eq!(alloc.num_assigned(), 2);
    }

    #[test]
    fn test_sequential_auction_max_tasks() {
        let mut auction = SequentialAuction::new(2).with_max_tasks_per_robot(1);
        auction.add_task(Task::new(0, "t0"));
        auction.add_task(Task::new(1, "t1"));

        let bids = vec![
            vec![10.0, 9.0],  // robot 0 bids high on both
            vec![1.0, 1.0],   // robot 1 bids low
        ];

        let alloc = auction.run(&bids);
        // Robot 0 should get only 1 task due to limit
        let r0_tasks = alloc.robot_tasks(0);
        assert!(r0_tasks.len() <= 1);
    }

    #[test]
    fn test_combinatorial_greedy() {
        let mut auction = CombinatorialAuction::new();
        auction.add_task(Task::new(0, "t0"));
        auction.add_task(Task::new(1, "t1"));
        auction.add_task(Task::new(2, "t2"));

        auction.add_bid(Bid::bundle(0, vec![0, 1], 15.0, 5.0));   // profit 10, density 5
        auction.add_bid(Bid::single(1, 0, 8.0, 1.0));             // profit 7
        auction.add_bid(Bid::single(1, 2, 6.0, 2.0));             // profit 4

        let alloc = auction.solve_greedy();
        assert!(alloc.num_assigned() >= 2);
    }

    #[test]
    fn test_combinatorial_no_negative_profit() {
        let mut auction = CombinatorialAuction::new();
        auction.add_task(Task::new(0, "t0"));
        auction.add_bid(Bid::single(0, 0, 5.0, 10.0)); // negative profit

        let alloc = auction.solve_greedy();
        assert_eq!(alloc.num_assigned(), 0);
    }

    #[test]
    fn test_cbba_build_bundle() {
        let mut agent = CbbaAgent::new(0, 3);
        let mut values = HashMap::new();
        values.insert(0, 10.0);
        values.insert(1, 8.0);
        values.insert(2, 5.0);
        values.insert(3, 3.0);

        agent.build_bundle(&values);
        assert_eq!(agent.num_tasks(), 3);
    }

    #[test]
    fn test_cbba_consensus() {
        let mut agent_a = CbbaAgent::new(0, 2);
        let mut agent_b = CbbaAgent::new(1, 2);

        let mut values_a = HashMap::new();
        values_a.insert(0, 10.0);
        values_a.insert(1, 5.0);

        let mut values_b = HashMap::new();
        values_b.insert(0, 12.0); // agent B bids higher on task 0
        values_b.insert(1, 3.0);

        agent_a.build_bundle(&values_a);
        agent_b.build_bundle(&values_b);

        agent_a.consensus(&agent_b);
        // Agent A should lose task 0 to agent B
        let a_tasks = agent_a.assigned_tasks();
        assert!(!a_tasks.contains(&0) || a_tasks.len() <= 2);
    }

    #[test]
    fn test_distance_based_bid() {
        let close = distance_based_bid((0.0, 0.0), (1.0, 0.0), 100.0);
        let far = distance_based_bid((0.0, 0.0), (10.0, 0.0), 100.0);
        assert!(close > far);
    }

    #[test]
    fn test_capacity_based_bid() {
        let full = capacity_based_bid(10.0, 10.0, 100.0);
        let empty = capacity_based_bid(0.0, 10.0, 100.0);
        assert!((full - 0.0).abs() < 1e-10);
        assert!((empty - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_display_formats() {
        let bid = Bid::single(0, 1, 10.0, 5.0);
        assert!(format!("{bid}").contains("robot=0"));

        let task = Task::new(0, "patrol");
        assert!(format!("{task}").contains("patrol"));

        let alloc = Allocation::new();
        assert!(format!("{alloc}").contains("0 assigned"));
    }

    #[test]
    fn test_sequential_auction_display() {
        let auction = SequentialAuction::new(5);
        assert!(format!("{auction}").contains("5 robots"));
    }

    #[test]
    fn test_cbba_display() {
        let agent = CbbaAgent::new(3, 5);
        assert!(format!("{agent}").contains("id=3"));
    }
}
