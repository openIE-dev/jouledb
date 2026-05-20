//! Resource management — allocation with deadlock detection (wait-for graph,
//! cycle detection), Banker's algorithm for deadlock avoidance, resource
//! release, priority inversion detection, resource usage statistics.

use std::collections::{HashMap, HashSet, VecDeque};

// ── Resource ────────────────────────────────────────────────────────────────

/// A resource type with a finite number of instances.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resource {
    pub name: String,
    pub total_instances: u32,
    pub available: u32,
}

// ── Process Info ────────────────────────────────────────────────────────────

/// Information about a process in the resource manager.
#[derive(Debug, Clone)]
pub struct ProcessInfo {
    pub pid: u32,
    pub name: String,
    pub priority: u32,
    /// Resources currently held: resource_name -> count held.
    pub held: HashMap<String, u32>,
    /// Resources currently requested (waiting): resource_name -> count needed.
    pub waiting: HashMap<String, u32>,
    /// Maximum claim (for Banker's algorithm): resource_name -> max needed.
    pub max_claim: HashMap<String, u32>,
}

// ── Error ───────────────────────────────────────────────────────────────────

/// Resource manager errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResourceError {
    ResourceNotFound(String),
    ProcessNotFound(u32),
    AlreadyExists(String),
    InsufficientResources { resource: String, requested: u32, available: u32 },
    DeadlockDetected(Vec<u32>),
    UnsafeState(String),
    InvalidRelease { resource: String, process: u32 },
}

impl std::fmt::Display for ResourceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResourceError::ResourceNotFound(r) => write!(f, "resource not found: {r}"),
            ResourceError::ProcessNotFound(p) => write!(f, "process not found: {p}"),
            ResourceError::AlreadyExists(n) => write!(f, "already exists: {n}"),
            ResourceError::InsufficientResources {
                resource,
                requested,
                available,
            } => write!(
                f,
                "insufficient: {resource} requested={requested} available={available}"
            ),
            ResourceError::DeadlockDetected(pids) => write!(f, "deadlock: {pids:?}"),
            ResourceError::UnsafeState(msg) => write!(f, "unsafe state: {msg}"),
            ResourceError::InvalidRelease { resource, process } => {
                write!(f, "invalid release: {resource} by process {process}")
            }
        }
    }
}

// ── Usage Statistics ────────────────────────────────────────────────────────

/// Resource usage statistics.
#[derive(Debug, Clone)]
pub struct ResourceStats {
    pub total_resources: usize,
    pub total_processes: usize,
    pub total_allocations: u64,
    pub total_releases: u64,
    pub total_denials: u64,
    pub deadlocks_detected: u64,
    pub utilization: HashMap<String, f64>,
}

// ── ResourceManager ─────────────────────────────────────────────────────────

/// Manages resource allocation, deadlock detection, and deadlock avoidance.
#[derive(Debug)]
pub struct ResourceManager {
    resources: HashMap<String, Resource>,
    processes: HashMap<u32, ProcessInfo>,
    alloc_count: u64,
    release_count: u64,
    denial_count: u64,
    deadlock_count: u64,
}

impl ResourceManager {
    pub fn new() -> Self {
        Self {
            resources: HashMap::new(),
            processes: HashMap::new(),
            alloc_count: 0,
            release_count: 0,
            denial_count: 0,
            deadlock_count: 0,
        }
    }

    /// Register a resource type.
    pub fn add_resource(&mut self, name: &str, instances: u32) -> Result<(), ResourceError> {
        if self.resources.contains_key(name) {
            return Err(ResourceError::AlreadyExists(name.into()));
        }
        self.resources.insert(
            name.to_string(),
            Resource {
                name: name.to_string(),
                total_instances: instances,
                available: instances,
            },
        );
        Ok(())
    }

    /// Register a process.
    pub fn add_process(&mut self, pid: u32, name: &str, priority: u32) -> Result<(), ResourceError> {
        if self.processes.contains_key(&pid) {
            return Err(ResourceError::AlreadyExists(format!("pid {pid}")));
        }
        self.processes.insert(
            pid,
            ProcessInfo {
                pid,
                name: name.to_string(),
                priority,
                held: HashMap::new(),
                waiting: HashMap::new(),
                max_claim: HashMap::new(),
            },
        );
        Ok(())
    }

    /// Set maximum claim for a process (needed for Banker's algorithm).
    pub fn set_max_claim(
        &mut self,
        pid: u32,
        resource: &str,
        max: u32,
    ) -> Result<(), ResourceError> {
        if !self.resources.contains_key(resource) {
            return Err(ResourceError::ResourceNotFound(resource.into()));
        }
        let proc = self
            .processes
            .get_mut(&pid)
            .ok_or(ResourceError::ProcessNotFound(pid))?;
        proc.max_claim.insert(resource.to_string(), max);
        Ok(())
    }

    /// Request (allocate) resources. Grants immediately if available.
    pub fn request(
        &mut self,
        pid: u32,
        resource: &str,
        count: u32,
    ) -> Result<(), ResourceError> {
        if !self.processes.contains_key(&pid) {
            return Err(ResourceError::ProcessNotFound(pid));
        }
        let res = self
            .resources
            .get(resource)
            .ok_or_else(|| ResourceError::ResourceNotFound(resource.into()))?;

        if res.available < count {
            self.denial_count += 1;
            // Record waiting
            if let Some(proc) = self.processes.get_mut(&pid) {
                proc.waiting.insert(resource.to_string(), count);
            }
            return Err(ResourceError::InsufficientResources {
                resource: resource.into(),
                requested: count,
                available: res.available,
            });
        }

        // Grant the request
        if let Some(res) = self.resources.get_mut(resource) {
            res.available -= count;
        }
        if let Some(proc) = self.processes.get_mut(&pid) {
            *proc.held.entry(resource.to_string()).or_insert(0) += count;
            proc.waiting.remove(resource);
        }
        self.alloc_count += 1;
        Ok(())
    }

    /// Release resources held by a process.
    pub fn release(
        &mut self,
        pid: u32,
        resource: &str,
        count: u32,
    ) -> Result<(), ResourceError> {
        let proc = self
            .processes
            .get(&pid)
            .ok_or(ResourceError::ProcessNotFound(pid))?;
        let held = *proc.held.get(resource).unwrap_or(&0);
        if held < count {
            return Err(ResourceError::InvalidRelease {
                resource: resource.into(),
                process: pid,
            });
        }

        if let Some(proc) = self.processes.get_mut(&pid) {
            let entry = proc.held.get_mut(resource).unwrap();
            *entry -= count;
            if *entry == 0 {
                proc.held.remove(resource);
            }
        }
        if let Some(res) = self.resources.get_mut(resource) {
            res.available += count;
        }
        self.release_count += 1;
        Ok(())
    }

    /// Release all resources held by a process.
    pub fn release_all(&mut self, pid: u32) -> Result<(), ResourceError> {
        let proc = self
            .processes
            .get(&pid)
            .ok_or(ResourceError::ProcessNotFound(pid))?;
        let held: Vec<(String, u32)> = proc.held.iter().map(|(k, v)| (k.clone(), *v)).collect();

        for (res_name, count) in held {
            if let Some(res) = self.resources.get_mut(&res_name) {
                res.available += count;
            }
            self.release_count += 1;
        }
        if let Some(proc) = self.processes.get_mut(&pid) {
            proc.held.clear();
            proc.waiting.clear();
        }
        Ok(())
    }

    // ── Deadlock Detection (Wait-For Graph) ──

    /// Build the wait-for graph.
    /// An edge from P_i to P_j means P_i is waiting for a resource held by P_j.
    pub fn wait_for_graph(&self) -> HashMap<u32, Vec<u32>> {
        let mut graph: HashMap<u32, Vec<u32>> = HashMap::new();

        for (pid, proc) in &self.processes {
            for (res_name, _) in &proc.waiting {
                // Find which processes hold this resource
                for (other_pid, other_proc) in &self.processes {
                    if other_pid != pid && other_proc.held.contains_key(res_name) {
                        graph.entry(*pid).or_default().push(*other_pid);
                    }
                }
            }
        }
        graph
    }

    /// Detect deadlock by finding cycles in the wait-for graph.
    /// Returns the set of PIDs involved in any cycle, or empty if no deadlock.
    pub fn detect_deadlock(&mut self) -> Vec<u32> {
        let graph = self.wait_for_graph();
        let cycle = find_cycle_in_graph(&graph);
        if !cycle.is_empty() {
            self.deadlock_count += 1;
        }
        cycle
    }

    // ── Banker's Algorithm (Deadlock Avoidance) ──

    /// Check if granting a request would leave the system in a safe state.
    /// Uses the Banker's algorithm.
    pub fn is_safe_state(&self) -> bool {
        let resource_names: Vec<String> = {
            let mut names: Vec<String> = self.resources.keys().cloned().collect();
            names.sort();
            names
        };

        // Available vector
        let mut available: Vec<u32> = resource_names
            .iter()
            .map(|r| self.resources[r].available)
            .collect();

        // Need matrix (max_claim - held) for each process
        let pids: Vec<u32> = {
            let mut p: Vec<u32> = self.processes.keys().copied().collect();
            p.sort();
            p
        };

        let mut finished = vec![false; pids.len()];

        // Allocation matrix
        let allocation: Vec<Vec<u32>> = pids
            .iter()
            .map(|pid| {
                let proc = &self.processes[pid];
                resource_names
                    .iter()
                    .map(|r| *proc.held.get(r).unwrap_or(&0))
                    .collect()
            })
            .collect();

        let need: Vec<Vec<u32>> = pids
            .iter()
            .map(|pid| {
                let proc = &self.processes[pid];
                resource_names
                    .iter()
                    .map(|r| {
                        let max = *proc.max_claim.get(r).unwrap_or(&0);
                        let held = *proc.held.get(r).unwrap_or(&0);
                        max.saturating_sub(held)
                    })
                    .collect()
            })
            .collect();

        // Find safe sequence
        let mut count = 0;
        loop {
            let mut found = false;
            for i in 0..pids.len() {
                if finished[i] {
                    continue;
                }
                // Check if need[i] <= available
                let can_finish = need[i]
                    .iter()
                    .zip(available.iter())
                    .all(|(n, a)| *n <= *a);

                if can_finish {
                    // Process can finish — release its resources
                    for j in 0..resource_names.len() {
                        available[j] += allocation[i][j];
                    }
                    finished[i] = true;
                    found = true;
                    count += 1;
                }
            }
            if !found {
                break;
            }
        }

        count == pids.len()
    }

    /// Request with safety check (Banker's algorithm).
    /// Only grants if the resulting state is safe.
    pub fn safe_request(
        &mut self,
        pid: u32,
        resource: &str,
        count: u32,
    ) -> Result<(), ResourceError> {
        if !self.processes.contains_key(&pid) {
            return Err(ResourceError::ProcessNotFound(pid));
        }
        let avail = self
            .resources
            .get(resource)
            .ok_or_else(|| ResourceError::ResourceNotFound(resource.into()))?
            .available;

        if avail < count {
            self.denial_count += 1;
            return Err(ResourceError::InsufficientResources {
                resource: resource.into(),
                requested: count,
                available: avail,
            });
        }

        // Tentatively allocate
        if let Some(res) = self.resources.get_mut(resource) {
            res.available -= count;
        }
        if let Some(proc) = self.processes.get_mut(&pid) {
            *proc.held.entry(resource.to_string()).or_insert(0) += count;
        }

        // Check safety
        if self.is_safe_state() {
            self.alloc_count += 1;
            Ok(())
        } else {
            // Roll back
            if let Some(res) = self.resources.get_mut(resource) {
                res.available += count;
            }
            if let Some(proc) = self.processes.get_mut(&pid) {
                let entry = proc.held.get_mut(resource).unwrap();
                *entry -= count;
                if *entry == 0 {
                    proc.held.remove(resource);
                }
            }
            self.denial_count += 1;
            Err(ResourceError::UnsafeState(format!(
                "granting {count} of {resource} to pid {pid} would be unsafe"
            )))
        }
    }

    // ── Priority Inversion Detection ──

    /// Detect priority inversion: a high-priority process waiting for a
    /// resource held by a low-priority process.
    pub fn detect_priority_inversion(&self) -> Vec<PriorityInversion> {
        let mut inversions = Vec::new();

        for (pid, proc) in &self.processes {
            for (res_name, _) in &proc.waiting {
                for (other_pid, other_proc) in &self.processes {
                    if other_pid != pid
                        && other_proc.held.contains_key(res_name)
                        && proc.priority > other_proc.priority
                    {
                        inversions.push(PriorityInversion {
                            high_priority_pid: *pid,
                            high_priority: proc.priority,
                            low_priority_pid: *other_pid,
                            low_priority: other_proc.priority,
                            resource: res_name.clone(),
                        });
                    }
                }
            }
        }
        inversions.sort_by_key(|i| (i.high_priority_pid, i.low_priority_pid));
        inversions
    }

    /// Get resource usage statistics.
    pub fn stats(&self) -> ResourceStats {
        let mut utilization = HashMap::new();
        for (name, res) in &self.resources {
            let used = res.total_instances - res.available;
            let util = if res.total_instances > 0 {
                used as f64 / res.total_instances as f64
            } else {
                0.0
            };
            utilization.insert(name.clone(), util);
        }

        ResourceStats {
            total_resources: self.resources.len(),
            total_processes: self.processes.len(),
            total_allocations: self.alloc_count,
            total_releases: self.release_count,
            total_denials: self.denial_count,
            deadlocks_detected: self.deadlock_count,
            utilization,
        }
    }

    /// Get a process by PID.
    pub fn get_process(&self, pid: u32) -> Option<&ProcessInfo> {
        self.processes.get(&pid)
    }

    /// Get a resource by name.
    pub fn get_resource(&self, name: &str) -> Option<&Resource> {
        self.resources.get(name)
    }

    /// Remove a process and release all its resources.
    pub fn remove_process(&mut self, pid: u32) -> Result<(), ResourceError> {
        self.release_all(pid)?;
        self.processes
            .remove(&pid)
            .ok_or(ResourceError::ProcessNotFound(pid))?;
        Ok(())
    }
}

// ── Priority Inversion ─────────────────────────────────────────────────────

/// Description of a priority inversion scenario.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PriorityInversion {
    pub high_priority_pid: u32,
    pub high_priority: u32,
    pub low_priority_pid: u32,
    pub low_priority: u32,
    pub resource: String,
}

// ── Cycle Detection ─────────────────────────────────────────────────────────

/// Find a cycle in a directed graph. Returns PIDs in the cycle, or empty vec.
fn find_cycle_in_graph(graph: &HashMap<u32, Vec<u32>>) -> Vec<u32> {
    let mut visited: HashSet<u32> = HashSet::new();
    let mut in_stack: HashSet<u32> = HashSet::new();

    let all_nodes: HashSet<u32> = graph
        .keys()
        .chain(graph.values().flat_map(|v| v.iter()))
        .copied()
        .collect();

    for &node in &all_nodes {
        if !visited.contains(&node) {
            let mut path = Vec::new();
            if dfs_cycle(graph, node, &mut visited, &mut in_stack, &mut path) {
                return path;
            }
        }
    }
    Vec::new()
}

fn dfs_cycle(
    graph: &HashMap<u32, Vec<u32>>,
    node: u32,
    visited: &mut HashSet<u32>,
    in_stack: &mut HashSet<u32>,
    path: &mut Vec<u32>,
) -> bool {
    visited.insert(node);
    in_stack.insert(node);
    path.push(node);

    if let Some(neighbors) = graph.get(&node) {
        for &next in neighbors {
            if !visited.contains(&next) {
                if dfs_cycle(graph, next, visited, in_stack, path) {
                    return true;
                }
            } else if in_stack.contains(&next) {
                // Found a cycle — trim path to just the cycle
                if let Some(pos) = path.iter().position(|p| *p == next) {
                    *path = path[pos..].to_vec();
                }
                return true;
            }
        }
    }

    path.pop();
    in_stack.remove(&node);
    false
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_basic() -> ResourceManager {
        let mut rm = ResourceManager::new();
        rm.add_resource("CPU", 4).unwrap();
        rm.add_resource("Disk", 2).unwrap();
        rm.add_process(1, "proc_a", 10).unwrap();
        rm.add_process(2, "proc_b", 5).unwrap();
        rm
    }

    #[test]
    fn test_add_resource() {
        let mut rm = ResourceManager::new();
        rm.add_resource("CPU", 4).unwrap();
        let res = rm.get_resource("CPU").unwrap();
        assert_eq!(res.total_instances, 4);
        assert_eq!(res.available, 4);
    }

    #[test]
    fn test_add_duplicate_resource() {
        let mut rm = ResourceManager::new();
        rm.add_resource("CPU", 4).unwrap();
        let result = rm.add_resource("CPU", 2);
        assert!(matches!(result, Err(ResourceError::AlreadyExists(_))));
    }

    #[test]
    fn test_request_and_release() {
        let mut rm = setup_basic();
        rm.request(1, "CPU", 2).unwrap();
        assert_eq!(rm.get_resource("CPU").unwrap().available, 2);
        rm.release(1, "CPU", 1).unwrap();
        assert_eq!(rm.get_resource("CPU").unwrap().available, 3);
    }

    #[test]
    fn test_insufficient_resources() {
        let mut rm = setup_basic();
        let result = rm.request(1, "CPU", 10);
        assert!(matches!(
            result,
            Err(ResourceError::InsufficientResources { .. })
        ));
    }

    #[test]
    fn test_release_all() {
        let mut rm = setup_basic();
        rm.request(1, "CPU", 2).unwrap();
        rm.request(1, "Disk", 1).unwrap();
        rm.release_all(1).unwrap();
        assert_eq!(rm.get_resource("CPU").unwrap().available, 4);
        assert_eq!(rm.get_resource("Disk").unwrap().available, 2);
    }

    #[test]
    fn test_invalid_release() {
        let mut rm = setup_basic();
        let result = rm.release(1, "CPU", 1);
        assert!(matches!(result, Err(ResourceError::InvalidRelease { .. })));
    }

    #[test]
    fn test_deadlock_detection_no_deadlock() {
        let mut rm = setup_basic();
        rm.request(1, "CPU", 1).unwrap();
        let cycle = rm.detect_deadlock();
        assert!(cycle.is_empty());
    }

    #[test]
    fn test_deadlock_detection_with_cycle() {
        let mut rm = ResourceManager::new();
        rm.add_resource("A", 1).unwrap();
        rm.add_resource("B", 1).unwrap();
        rm.add_process(1, "p1", 5).unwrap();
        rm.add_process(2, "p2", 5).unwrap();

        // p1 holds A, wants B
        rm.request(1, "A", 1).unwrap();
        // p2 holds B, wants A
        rm.request(2, "B", 1).unwrap();

        // Now both are waiting
        let _ = rm.request(1, "B", 1); // fails, p1 waits
        let _ = rm.request(2, "A", 1); // fails, p2 waits

        let cycle = rm.detect_deadlock();
        assert!(!cycle.is_empty());
    }

    #[test]
    fn test_wait_for_graph() {
        let mut rm = ResourceManager::new();
        rm.add_resource("R", 1).unwrap();
        rm.add_process(1, "p1", 5).unwrap();
        rm.add_process(2, "p2", 5).unwrap();

        rm.request(1, "R", 1).unwrap(); // p1 holds R
        let _ = rm.request(2, "R", 1); // p2 waits for R

        let graph = rm.wait_for_graph();
        assert!(graph.contains_key(&2));
        assert!(graph[&2].contains(&1));
    }

    #[test]
    fn test_bankers_safe_state() {
        let mut rm = ResourceManager::new();
        rm.add_resource("R", 10).unwrap();
        rm.add_process(1, "p1", 5).unwrap();
        rm.add_process(2, "p2", 5).unwrap();
        rm.set_max_claim(1, "R", 5).unwrap();
        rm.set_max_claim(2, "R", 5).unwrap();

        rm.request(1, "R", 2).unwrap();
        rm.request(2, "R", 2).unwrap();
        // Available = 6, need1 = 3, need2 = 3 — safe
        assert!(rm.is_safe_state());
    }

    #[test]
    fn test_safe_request_grants() {
        let mut rm = ResourceManager::new();
        rm.add_resource("R", 10).unwrap();
        rm.add_process(1, "p1", 5).unwrap();
        rm.set_max_claim(1, "R", 5).unwrap();
        rm.safe_request(1, "R", 3).unwrap();
        assert_eq!(rm.get_resource("R").unwrap().available, 7);
    }

    #[test]
    fn test_safe_request_denies_unsafe() {
        let mut rm = ResourceManager::new();
        rm.add_resource("R", 4).unwrap();
        rm.add_process(1, "p1", 5).unwrap();
        rm.add_process(2, "p2", 5).unwrap();
        rm.set_max_claim(1, "R", 3).unwrap();
        rm.set_max_claim(2, "R", 3).unwrap();

        rm.safe_request(1, "R", 2).unwrap();
        rm.safe_request(2, "R", 1).unwrap();
        // Available = 1. need1 = 1, need2 = 2.
        // If we give p2 one more, available = 0, need1 = 1 > 0, need2 = 1 > 0 -> unsafe
        let result = rm.safe_request(2, "R", 1);
        assert!(matches!(result, Err(ResourceError::UnsafeState(_))));
    }

    #[test]
    fn test_priority_inversion_detection() {
        let mut rm = ResourceManager::new();
        rm.add_resource("Lock", 1).unwrap();
        rm.add_process(1, "high_prio", 100).unwrap();
        rm.add_process(2, "low_prio", 1).unwrap();

        rm.request(2, "Lock", 1).unwrap();
        let _ = rm.request(1, "Lock", 1); // high prio waits

        let inversions = rm.detect_priority_inversion();
        assert_eq!(inversions.len(), 1);
        assert_eq!(inversions[0].high_priority_pid, 1);
        assert_eq!(inversions[0].low_priority_pid, 2);
    }

    #[test]
    fn test_no_priority_inversion() {
        let mut rm = ResourceManager::new();
        rm.add_resource("Lock", 1).unwrap();
        rm.add_process(1, "low", 1).unwrap();
        rm.add_process(2, "high", 100).unwrap();

        rm.request(2, "Lock", 1).unwrap();
        let _ = rm.request(1, "Lock", 1); // low prio waits for high prio — not inversion

        let inversions = rm.detect_priority_inversion();
        assert!(inversions.is_empty());
    }

    #[test]
    fn test_stats() {
        let mut rm = setup_basic();
        rm.request(1, "CPU", 2).unwrap();
        rm.release(1, "CPU", 1).unwrap();
        let stats = rm.stats();
        assert_eq!(stats.total_allocations, 1);
        assert_eq!(stats.total_releases, 1);
        assert_eq!(stats.total_resources, 2);
        assert_eq!(stats.total_processes, 2);
    }

    #[test]
    fn test_remove_process() {
        let mut rm = setup_basic();
        rm.request(1, "CPU", 2).unwrap();
        rm.remove_process(1).unwrap();
        assert_eq!(rm.get_resource("CPU").unwrap().available, 4);
        assert!(rm.get_process(1).is_none());
    }

    #[test]
    fn test_resource_utilization() {
        let mut rm = ResourceManager::new();
        rm.add_resource("Mem", 100).unwrap();
        rm.add_process(1, "user", 5).unwrap();
        rm.request(1, "Mem", 75).unwrap();
        let stats = rm.stats();
        let util = stats.utilization.get("Mem").unwrap();
        assert!((util - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_multiple_resource_types() {
        let mut rm = ResourceManager::new();
        rm.add_resource("CPU", 4).unwrap();
        rm.add_resource("GPU", 2).unwrap();
        rm.add_resource("RAM", 16).unwrap();
        rm.add_process(1, "app", 5).unwrap();

        rm.request(1, "CPU", 2).unwrap();
        rm.request(1, "GPU", 1).unwrap();
        rm.request(1, "RAM", 8).unwrap();

        let proc = rm.get_process(1).unwrap();
        assert_eq!(*proc.held.get("CPU").unwrap(), 2);
        assert_eq!(*proc.held.get("GPU").unwrap(), 1);
        assert_eq!(*proc.held.get("RAM").unwrap(), 8);
    }

    #[test]
    fn test_process_not_found() {
        let mut rm = ResourceManager::new();
        rm.add_resource("R", 1).unwrap();
        let result = rm.request(999, "R", 1);
        assert!(matches!(result, Err(ResourceError::ProcessNotFound(999))));
    }

    #[test]
    fn test_denial_count() {
        let mut rm = ResourceManager::new();
        rm.add_resource("R", 1).unwrap();
        rm.add_process(1, "p1", 5).unwrap();
        rm.request(1, "R", 1).unwrap();
        rm.add_process(2, "p2", 5).unwrap();
        let _ = rm.request(2, "R", 1); // denied
        let stats = rm.stats();
        assert_eq!(stats.total_denials, 1);
    }
}
