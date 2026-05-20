//! Process tree simulation — spawn/kill, parent-child hierarchy, PID allocation,
//! process states, signal delivery, wait/reap, process groups, tree visualization.

use std::collections::{HashMap, HashSet, VecDeque};

// ── Process State ───────────────────────────────────────────────────────────

/// State a process can be in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessState {
    Running,
    Sleeping,
    Stopped,
    Zombie,
}

impl std::fmt::Display for ProcessState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProcessState::Running => write!(f, "R"),
            ProcessState::Sleeping => write!(f, "S"),
            ProcessState::Stopped => write!(f, "T"),
            ProcessState::Zombie => write!(f, "Z"),
        }
    }
}

// ── Signal ──────────────────────────────────────────────────────────────────

/// Signals that can be delivered to processes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Signal {
    SigHup,
    SigInt,
    SigKill,
    SigTerm,
    SigStop,
    SigCont,
    SigChld,
    SigUsr1,
    SigUsr2,
}

// ── Process ─────────────────────────────────────────────────────────────────

/// A simulated process.
#[derive(Debug, Clone)]
pub struct Process {
    pub pid: u32,
    pub ppid: u32,
    pub pgid: u32,
    pub name: String,
    pub state: ProcessState,
    pub exit_code: Option<i32>,
    pub children: Vec<u32>,
    pub pending_signals: VecDeque<Signal>,
    pub created_at: u64,
    pub cpu_time: u64,
}

// ── Error ───────────────────────────────────────────────────────────────────

/// Process tree errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProcessError {
    NotFound(u32),
    AlreadyDead(u32),
    NoChildren(u32),
    InvalidOperation(String),
}

impl std::fmt::Display for ProcessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProcessError::NotFound(pid) => write!(f, "process {pid} not found"),
            ProcessError::AlreadyDead(pid) => write!(f, "process {pid} already dead"),
            ProcessError::NoChildren(pid) => write!(f, "process {pid} has no children"),
            ProcessError::InvalidOperation(msg) => write!(f, "invalid operation: {msg}"),
        }
    }
}

// ── ProcessTree ─────────────────────────────────────────────────────────────

/// Manages a tree of simulated processes.
#[derive(Debug)]
pub struct ProcessTree {
    processes: HashMap<u32, Process>,
    next_pid: u32,
    clock: u64,
}

impl ProcessTree {
    /// Create a new process tree with an init process (PID 1).
    pub fn new() -> Self {
        let mut processes = HashMap::new();
        let init = Process {
            pid: 1,
            ppid: 0,
            pgid: 1,
            name: "init".to_string(),
            state: ProcessState::Running,
            exit_code: None,
            children: Vec::new(),
            pending_signals: VecDeque::new(),
            created_at: 0,
            cpu_time: 0,
        };
        processes.insert(1, init);
        Self {
            processes,
            next_pid: 2,
            clock: 1,
        }
    }

    fn tick(&mut self) -> u64 {
        let t = self.clock;
        self.clock += 1;
        t
    }

    fn alloc_pid(&mut self) -> u32 {
        let pid = self.next_pid;
        self.next_pid += 1;
        pid
    }

    /// Spawn a new process as a child of `parent_pid`.
    pub fn spawn(&mut self, parent_pid: u32, name: &str) -> Result<u32, ProcessError> {
        // Check parent exists and is alive
        let parent_pgid = {
            let parent = self
                .processes
                .get(&parent_pid)
                .ok_or(ProcessError::NotFound(parent_pid))?;
            if parent.state == ProcessState::Zombie {
                return Err(ProcessError::AlreadyDead(parent_pid));
            }
            parent.pgid
        };

        let pid = self.alloc_pid();
        let now = self.tick();

        let child = Process {
            pid,
            ppid: parent_pid,
            pgid: parent_pgid,
            name: name.to_string(),
            state: ProcessState::Running,
            exit_code: None,
            children: Vec::new(),
            pending_signals: VecDeque::new(),
            created_at: now,
            cpu_time: 0,
        };
        self.processes.insert(pid, child);

        // Add to parent's children
        if let Some(parent) = self.processes.get_mut(&parent_pid) {
            parent.children.push(pid);
        }
        Ok(pid)
    }

    /// Get a reference to a process by PID.
    pub fn get(&self, pid: u32) -> Option<&Process> {
        self.processes.get(&pid)
    }

    /// Kill a process (send SIGKILL — immediately transitions to zombie).
    pub fn kill(&mut self, pid: u32) -> Result<(), ProcessError> {
        self.send_signal(pid, Signal::SigKill)
    }

    /// Send a signal to a process.
    pub fn send_signal(&mut self, pid: u32, signal: Signal) -> Result<(), ProcessError> {
        // Check existence and get state
        let state = {
            let proc = self
                .processes
                .get(&pid)
                .ok_or(ProcessError::NotFound(pid))?;
            proc.state
        };

        if state == ProcessState::Zombie {
            return Err(ProcessError::AlreadyDead(pid));
        }

        match signal {
            Signal::SigKill => {
                // Immediate death — mark as zombie
                self.make_zombie(pid, 137);
            }
            Signal::SigTerm | Signal::SigInt | Signal::SigHup => {
                // Mark as zombie with signal-based exit code
                let code = match signal {
                    Signal::SigTerm => 143,
                    Signal::SigInt => 130,
                    Signal::SigHup => 129,
                    _ => 128,
                };
                self.make_zombie(pid, code);
            }
            Signal::SigStop => {
                if let Some(proc) = self.processes.get_mut(&pid) {
                    proc.state = ProcessState::Stopped;
                }
            }
            Signal::SigCont => {
                if let Some(proc) = self.processes.get_mut(&pid) {
                    if proc.state == ProcessState::Stopped {
                        proc.state = ProcessState::Running;
                    }
                }
            }
            _ => {
                if let Some(proc) = self.processes.get_mut(&pid) {
                    proc.pending_signals.push_back(signal);
                }
            }
        }
        Ok(())
    }

    /// Transition a process to Zombie state, reparenting its children to init.
    fn make_zombie(&mut self, pid: u32, exit_code: i32) {
        // Collect children to reparent
        let children: Vec<u32> = self
            .processes
            .get(&pid)
            .map(|p| p.children.clone())
            .unwrap_or_default();

        // Reparent children to init (pid 1)
        for child_pid in &children {
            if let Some(child) = self.processes.get_mut(child_pid) {
                child.ppid = 1;
            }
        }
        // Add children to init's child list
        if pid != 1 {
            if let Some(init) = self.processes.get_mut(&1) {
                for child_pid in &children {
                    if !init.children.contains(child_pid) {
                        init.children.push(*child_pid);
                    }
                }
            }
        }

        // Mark as zombie
        if let Some(proc) = self.processes.get_mut(&pid) {
            proc.state = ProcessState::Zombie;
            proc.exit_code = Some(exit_code);
            proc.children.clear();
        }

        // Send SIGCHLD to parent
        let ppid = self.processes.get(&pid).map(|p| p.ppid).unwrap_or(0);
        if ppid > 0 {
            if let Some(parent) = self.processes.get_mut(&ppid) {
                parent.pending_signals.push_back(Signal::SigChld);
            }
        }
    }

    /// Wait for any zombie child of `parent_pid` and reap it.
    /// Returns (child_pid, exit_code) or error if no zombie children.
    pub fn wait(&mut self, parent_pid: u32) -> Result<(u32, i32), ProcessError> {
        let parent = self
            .processes
            .get(&parent_pid)
            .ok_or(ProcessError::NotFound(parent_pid))?;

        if parent.children.is_empty() {
            return Err(ProcessError::NoChildren(parent_pid));
        }

        // Find a zombie child
        let zombie_pid = parent.children.iter().find(|cpid| {
            self.processes
                .get(cpid)
                .is_some_and(|c| c.state == ProcessState::Zombie)
        }).copied();

        if let Some(zpid) = zombie_pid {
            let exit_code = self.processes[&zpid].exit_code.unwrap_or(0);
            // Remove from parent's children
            if let Some(parent) = self.processes.get_mut(&parent_pid) {
                parent.children.retain(|c| *c != zpid);
            }
            // Remove zombie process
            self.processes.remove(&zpid);
            Ok((zpid, exit_code))
        } else {
            Err(ProcessError::InvalidOperation(format!(
                "no zombie children for pid {parent_pid}"
            )))
        }
    }

    /// Set a process's state to Sleeping.
    pub fn sleep_process(&mut self, pid: u32) -> Result<(), ProcessError> {
        let proc = self
            .processes
            .get_mut(&pid)
            .ok_or(ProcessError::NotFound(pid))?;
        if proc.state == ProcessState::Zombie {
            return Err(ProcessError::AlreadyDead(pid));
        }
        proc.state = ProcessState::Sleeping;
        Ok(())
    }

    /// Wake a sleeping process.
    pub fn wake_process(&mut self, pid: u32) -> Result<(), ProcessError> {
        let proc = self
            .processes
            .get_mut(&pid)
            .ok_or(ProcessError::NotFound(pid))?;
        if proc.state == ProcessState::Sleeping {
            proc.state = ProcessState::Running;
        }
        Ok(())
    }

    /// Set process group for a process.
    pub fn set_pgid(&mut self, pid: u32, pgid: u32) -> Result<(), ProcessError> {
        let proc = self
            .processes
            .get_mut(&pid)
            .ok_or(ProcessError::NotFound(pid))?;
        proc.pgid = pgid;
        Ok(())
    }

    /// Create a new process group (using the process's own PID as PGID).
    pub fn create_process_group(&mut self, pid: u32) -> Result<u32, ProcessError> {
        let proc = self
            .processes
            .get_mut(&pid)
            .ok_or(ProcessError::NotFound(pid))?;
        proc.pgid = pid;
        Ok(pid)
    }

    /// Get all PIDs in a process group.
    pub fn process_group_members(&self, pgid: u32) -> Vec<u32> {
        let mut members: Vec<u32> = self
            .processes
            .values()
            .filter(|p| p.pgid == pgid)
            .map(|p| p.pid)
            .collect();
        members.sort();
        members
    }

    /// Send a signal to all processes in a group.
    pub fn signal_group(&mut self, pgid: u32, signal: Signal) -> Result<usize, ProcessError> {
        let pids: Vec<u32> = self
            .processes
            .values()
            .filter(|p| p.pgid == pgid && p.state != ProcessState::Zombie)
            .map(|p| p.pid)
            .collect();

        let count = pids.len();
        for pid in pids {
            let _ = self.send_signal(pid, signal);
        }
        Ok(count)
    }

    /// Add CPU time to a process.
    pub fn add_cpu_time(&mut self, pid: u32, ticks: u64) -> Result<(), ProcessError> {
        let proc = self
            .processes
            .get_mut(&pid)
            .ok_or(ProcessError::NotFound(pid))?;
        proc.cpu_time += ticks;
        Ok(())
    }

    /// Drain pending signals from a process.
    pub fn drain_signals(&mut self, pid: u32) -> Result<Vec<Signal>, ProcessError> {
        let proc = self
            .processes
            .get_mut(&pid)
            .ok_or(ProcessError::NotFound(pid))?;
        let signals: Vec<Signal> = proc.pending_signals.drain(..).collect();
        Ok(signals)
    }

    /// List all PIDs.
    pub fn all_pids(&self) -> Vec<u32> {
        let mut pids: Vec<u32> = self.processes.keys().copied().collect();
        pids.sort();
        pids
    }

    /// Count of live (non-zombie) processes.
    pub fn live_count(&self) -> usize {
        self.processes
            .values()
            .filter(|p| p.state != ProcessState::Zombie)
            .count()
    }

    /// Total process count (including zombies).
    pub fn total_count(&self) -> usize {
        self.processes.len()
    }

    /// Get children of a process (PIDs).
    pub fn children_of(&self, pid: u32) -> Result<Vec<u32>, ProcessError> {
        let proc = self
            .processes
            .get(&pid)
            .ok_or(ProcessError::NotFound(pid))?;
        Ok(proc.children.clone())
    }

    /// Get all descendants of a process (BFS).
    pub fn descendants(&self, pid: u32) -> Result<Vec<u32>, ProcessError> {
        if !self.processes.contains_key(&pid) {
            return Err(ProcessError::NotFound(pid));
        }
        let mut result = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back(pid);
        while let Some(current) = queue.pop_front() {
            if let Some(proc) = self.processes.get(&current) {
                for child in &proc.children {
                    result.push(*child);
                    queue.push_back(*child);
                }
            }
        }
        result.sort();
        Ok(result)
    }

    /// Render the process tree as a string (indented hierarchy).
    pub fn render_tree(&self, root_pid: u32) -> String {
        let mut output = String::new();
        self.render_tree_inner(root_pid, &mut output, "", true);
        output
    }

    fn render_tree_inner(&self, pid: u32, output: &mut String, prefix: &str, is_last: bool) {
        let Some(proc) = self.processes.get(&pid) else {
            return;
        };
        let connector = if pid == 1 { "" } else if is_last { "`-- " } else { "|-- " };
        output.push_str(&format!(
            "{prefix}{connector}[{}] {} ({})\n",
            proc.pid, proc.name, proc.state
        ));

        let child_prefix = if pid == 1 {
            String::new()
        } else if is_last {
            format!("{prefix}    ")
        } else {
            format!("{prefix}|   ")
        };

        let children = &proc.children;
        for (i, child_pid) in children.iter().enumerate() {
            let last = i == children.len() - 1;
            self.render_tree_inner(*child_pid, output, &child_prefix, last);
        }
    }

    /// Kill a process and all its descendants.
    pub fn kill_tree(&mut self, pid: u32) -> Result<usize, ProcessError> {
        let desc = self.descendants(pid)?;
        let mut count = 0;
        // Kill descendants in reverse order (leaves first)
        for dpid in desc.iter().rev() {
            if self.processes.get(dpid).is_some_and(|p| p.state != ProcessState::Zombie) {
                let _ = self.kill(*dpid);
                count += 1;
            }
        }
        // Kill the root
        if self.processes.get(&pid).is_some_and(|p| p.state != ProcessState::Zombie) {
            self.kill(pid)?;
            count += 1;
        }
        Ok(count)
    }

    /// Check if a PID exists.
    pub fn exists(&self, pid: u32) -> bool {
        self.processes.contains_key(&pid)
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_process() {
        let tree = ProcessTree::new();
        let init = tree.get(1).unwrap();
        assert_eq!(init.pid, 1);
        assert_eq!(init.name, "init");
        assert_eq!(init.state, ProcessState::Running);
    }

    #[test]
    fn test_spawn_child() {
        let mut tree = ProcessTree::new();
        let pid = tree.spawn(1, "bash").unwrap();
        assert!(pid > 1);
        let proc = tree.get(pid).unwrap();
        assert_eq!(proc.ppid, 1);
        assert_eq!(proc.name, "bash");
        assert_eq!(proc.state, ProcessState::Running);
    }

    #[test]
    fn test_spawn_from_dead_parent() {
        let mut tree = ProcessTree::new();
        let pid = tree.spawn(1, "child").unwrap();
        tree.kill(pid).unwrap();
        let result = tree.spawn(pid, "grandchild");
        assert!(result.is_err());
    }

    #[test]
    fn test_kill_makes_zombie() {
        let mut tree = ProcessTree::new();
        let pid = tree.spawn(1, "victim").unwrap();
        tree.kill(pid).unwrap();
        let proc = tree.get(pid).unwrap();
        assert_eq!(proc.state, ProcessState::Zombie);
        assert_eq!(proc.exit_code, Some(137));
    }

    #[test]
    fn test_wait_reaps_zombie() {
        let mut tree = ProcessTree::new();
        let pid = tree.spawn(1, "child").unwrap();
        tree.kill(pid).unwrap();
        let (reaped, code) = tree.wait(1).unwrap();
        assert_eq!(reaped, pid);
        assert_eq!(code, 137);
        assert!(!tree.exists(pid));
    }

    #[test]
    fn test_reparent_to_init() {
        let mut tree = ProcessTree::new();
        let parent = tree.spawn(1, "parent").unwrap();
        let child = tree.spawn(parent, "child").unwrap();
        tree.kill(parent).unwrap();
        let proc = tree.get(child).unwrap();
        assert_eq!(proc.ppid, 1);
    }

    #[test]
    fn test_stop_and_cont() {
        let mut tree = ProcessTree::new();
        let pid = tree.spawn(1, "proc").unwrap();
        tree.send_signal(pid, Signal::SigStop).unwrap();
        assert_eq!(tree.get(pid).unwrap().state, ProcessState::Stopped);
        tree.send_signal(pid, Signal::SigCont).unwrap();
        assert_eq!(tree.get(pid).unwrap().state, ProcessState::Running);
    }

    #[test]
    fn test_sleep_and_wake() {
        let mut tree = ProcessTree::new();
        let pid = tree.spawn(1, "sleeper").unwrap();
        tree.sleep_process(pid).unwrap();
        assert_eq!(tree.get(pid).unwrap().state, ProcessState::Sleeping);
        tree.wake_process(pid).unwrap();
        assert_eq!(tree.get(pid).unwrap().state, ProcessState::Running);
    }

    #[test]
    fn test_process_group() {
        let mut tree = ProcessTree::new();
        let p1 = tree.spawn(1, "a").unwrap();
        let p2 = tree.spawn(1, "b").unwrap();
        tree.create_process_group(p1).unwrap();
        tree.set_pgid(p2, p1).unwrap();
        let members = tree.process_group_members(p1);
        assert!(members.contains(&p1));
        assert!(members.contains(&p2));
    }

    #[test]
    fn test_signal_group() {
        let mut tree = ProcessTree::new();
        let p1 = tree.spawn(1, "a").unwrap();
        let p2 = tree.spawn(1, "b").unwrap();
        tree.set_pgid(p1, 100).unwrap();
        tree.set_pgid(p2, 100).unwrap();
        let count = tree.signal_group(100, Signal::SigKill).unwrap();
        assert_eq!(count, 2);
        assert_eq!(tree.get(p1).unwrap().state, ProcessState::Zombie);
        assert_eq!(tree.get(p2).unwrap().state, ProcessState::Zombie);
    }

    #[test]
    fn test_pending_signals() {
        let mut tree = ProcessTree::new();
        let pid = tree.spawn(1, "handler").unwrap();
        tree.send_signal(pid, Signal::SigUsr1).unwrap();
        tree.send_signal(pid, Signal::SigUsr2).unwrap();
        let signals = tree.drain_signals(pid).unwrap();
        assert_eq!(signals, vec![Signal::SigUsr1, Signal::SigUsr2]);
    }

    #[test]
    fn test_descendants() {
        let mut tree = ProcessTree::new();
        let a = tree.spawn(1, "a").unwrap();
        let b = tree.spawn(a, "b").unwrap();
        let c = tree.spawn(b, "c").unwrap();
        let desc = tree.descendants(a).unwrap();
        assert!(desc.contains(&b));
        assert!(desc.contains(&c));
    }

    #[test]
    fn test_kill_tree() {
        let mut tree = ProcessTree::new();
        let a = tree.spawn(1, "a").unwrap();
        let _b = tree.spawn(a, "b").unwrap();
        let _c = tree.spawn(a, "c").unwrap();
        let killed = tree.kill_tree(a).unwrap();
        assert_eq!(killed, 3);
    }

    #[test]
    fn test_render_tree() {
        let mut tree = ProcessTree::new();
        let sh = tree.spawn(1, "bash").unwrap();
        let _vim = tree.spawn(sh, "vim").unwrap();
        let output = tree.render_tree(1);
        assert!(output.contains("init"));
        assert!(output.contains("bash"));
        assert!(output.contains("vim"));
    }

    #[test]
    fn test_live_vs_total_count() {
        let mut tree = ProcessTree::new();
        let p = tree.spawn(1, "child").unwrap();
        assert_eq!(tree.live_count(), 2);
        assert_eq!(tree.total_count(), 2);
        tree.kill(p).unwrap();
        assert_eq!(tree.live_count(), 1);
        assert_eq!(tree.total_count(), 2); // zombie still counted
    }

    #[test]
    fn test_cpu_time() {
        let mut tree = ProcessTree::new();
        let pid = tree.spawn(1, "worker").unwrap();
        tree.add_cpu_time(pid, 100).unwrap();
        tree.add_cpu_time(pid, 50).unwrap();
        assert_eq!(tree.get(pid).unwrap().cpu_time, 150);
    }

    #[test]
    fn test_sigchld_on_child_death() {
        let mut tree = ProcessTree::new();
        let child = tree.spawn(1, "child").unwrap();
        tree.kill(child).unwrap();
        let sigs = tree.drain_signals(1).unwrap();
        assert!(sigs.contains(&Signal::SigChld));
    }

    #[test]
    fn test_pid_allocation_monotonic() {
        let mut tree = ProcessTree::new();
        let p1 = tree.spawn(1, "a").unwrap();
        let p2 = tree.spawn(1, "b").unwrap();
        let p3 = tree.spawn(1, "c").unwrap();
        assert!(p1 < p2);
        assert!(p2 < p3);
    }
}
