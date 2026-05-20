//! Round-robin / fair scheduler — time-slice scheduling with priority and aging.
//!
//! Replaces ad-hoc JS schedulers with a pure-Rust round-robin scheduler.
//! Supports time-slice scheduling, priority levels with aging to prevent
//! starvation, preemption simulation, scheduler statistics, and
//! ready/blocked/completed queue management.

use std::collections::{HashMap, VecDeque};

// ── Errors ──────────────────────────────────────────────────────

/// Scheduler domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchedulerError {
    /// Process not found.
    ProcessNotFound(u64),
    /// Process already exists.
    DuplicateProcess(u64),
    /// Process is not in expected state.
    InvalidState { id: u64, expected: &'static str, actual: &'static str },
    /// Scheduler is stopped.
    SchedulerStopped,
}

impl std::fmt::Display for SchedulerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ProcessNotFound(id) => write!(f, "process not found: {id}"),
            Self::DuplicateProcess(id) => write!(f, "process already exists: {id}"),
            Self::InvalidState { id, expected, actual } => {
                write!(f, "process {id}: expected {expected}, got {actual}")
            }
            Self::SchedulerStopped => write!(f, "scheduler is stopped"),
        }
    }
}

impl std::error::Error for SchedulerError {}

// ── Priority ────────────────────────────────────────────────────

/// Priority level (0 = highest).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Priority(pub u32);

impl Priority {
    pub const REALTIME: Priority = Priority(0);
    pub const HIGH: Priority = Priority(1);
    pub const NORMAL: Priority = Priority(2);
    pub const LOW: Priority = Priority(3);
    pub const IDLE: Priority = Priority(4);
}

impl Default for Priority {
    fn default() -> Self {
        Self::NORMAL
    }
}

// ── Process State ───────────────────────────────────────────────

/// State of a scheduled process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessState {
    Ready,
    Running,
    Blocked,
    Completed,
    Preempted,
}

impl ProcessState {
    fn name(&self) -> &'static str {
        match self {
            Self::Ready => "Ready",
            Self::Running => "Running",
            Self::Blocked => "Blocked",
            Self::Completed => "Completed",
            Self::Preempted => "Preempted",
        }
    }
}

// ── Process ─────────────────────────────────────────────────────

/// A process managed by the scheduler.
#[derive(Debug, Clone)]
pub struct Process {
    pub id: u64,
    pub name: String,
    pub state: ProcessState,
    pub base_priority: Priority,
    /// Effective priority after aging.
    pub effective_priority: Priority,
    /// Total work units to complete.
    pub total_work: u64,
    /// Work units completed.
    pub work_done: u64,
    /// CPU time consumed (simulated ticks).
    pub cpu_time: u64,
    /// How many times this process was scheduled.
    pub schedule_count: u64,
    /// How many ticks this process has waited without running.
    pub wait_ticks: u64,
    /// How many times this process was preempted.
    pub preempt_count: u64,
    /// Time slice allocated per scheduling round.
    pub time_slice: u64,
}

impl Process {
    pub fn new(id: u64, name: impl Into<String>, total_work: u64) -> Self {
        Self {
            id,
            name: name.into(),
            state: ProcessState::Ready,
            base_priority: Priority::default(),
            effective_priority: Priority::default(),
            total_work,
            work_done: 0,
            cpu_time: 0,
            schedule_count: 0,
            wait_ticks: 0,
            preempt_count: 0,
            time_slice: 10,
        }
    }

    /// Set the base priority.
    pub fn with_priority(mut self, p: Priority) -> Self {
        self.base_priority = p;
        self.effective_priority = p;
        self
    }

    /// Set the time slice.
    pub fn with_time_slice(mut self, ts: u64) -> Self {
        self.time_slice = ts;
        self
    }

    /// Whether the process has completed all its work.
    pub fn is_finished(&self) -> bool {
        self.work_done >= self.total_work
    }

    /// Progress as a fraction (0.0..=1.0).
    pub fn progress(&self) -> f64 {
        if self.total_work == 0 {
            return 1.0;
        }
        self.work_done as f64 / self.total_work as f64
    }
}

// ── Scheduler Statistics ────────────────────────────────────────

/// Aggregate scheduler statistics.
#[derive(Debug, Clone, Default)]
pub struct SchedulerStats {
    pub total_processes: usize,
    pub ready_count: usize,
    pub running_count: usize,
    pub blocked_count: usize,
    pub completed_count: usize,
    pub total_context_switches: u64,
    pub total_ticks_elapsed: u64,
    pub total_preemptions: u64,
}

// ── Scheduler Event ─────────────────────────────────────────────

/// Events emitted by the scheduler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchedulerEvent {
    Scheduled { id: u64, tick: u64 },
    Preempted { id: u64, tick: u64 },
    Completed { id: u64, tick: u64 },
    Blocked { id: u64, tick: u64 },
    Unblocked { id: u64, tick: u64 },
    Aged { id: u64, old_priority: u32, new_priority: u32 },
}

// ── Round Robin Scheduler ───────────────────────────────────────

/// Fair round-robin scheduler with priority and aging.
pub struct RoundRobinScheduler {
    processes: HashMap<u64, Process>,
    ready_queue: VecDeque<u64>,
    blocked_set: Vec<u64>,
    current_running: Option<u64>,
    events: Vec<SchedulerEvent>,
    current_tick: u64,
    context_switches: u64,
    /// Aging threshold: after this many wait ticks, boost priority by 1 level.
    aging_threshold: u64,
    stopped: bool,
}

impl RoundRobinScheduler {
    pub fn new() -> Self {
        Self {
            processes: HashMap::new(),
            ready_queue: VecDeque::new(),
            blocked_set: Vec::new(),
            current_running: None,
            events: Vec::new(),
            current_tick: 0,
            context_switches: 0,
            aging_threshold: 20,
            stopped: false,
        }
    }

    /// Set the aging threshold (ticks of waiting before priority boost).
    pub fn with_aging_threshold(mut self, threshold: u64) -> Self {
        self.aging_threshold = threshold;
        self
    }

    /// Add a process to the scheduler.
    pub fn add_process(&mut self, process: Process) -> Result<(), SchedulerError> {
        if self.stopped {
            return Err(SchedulerError::SchedulerStopped);
        }
        if self.processes.contains_key(&process.id) {
            return Err(SchedulerError::DuplicateProcess(process.id));
        }
        let id = process.id;
        self.processes.insert(id, process);
        self.ready_queue.push_back(id);
        Ok(())
    }

    /// Run one scheduling cycle: pick the next process, execute its time slice.
    pub fn schedule_one(&mut self) -> Result<Option<u64>, SchedulerError> {
        if self.stopped {
            return Err(SchedulerError::SchedulerStopped);
        }

        // Apply aging to waiting processes
        self.apply_aging();

        // If something is currently running, preempt it
        if let Some(running_id) = self.current_running.take() {
            if let Some(proc) = self.processes.get_mut(&running_id) {
                if proc.state == ProcessState::Running && !proc.is_finished() {
                    proc.state = ProcessState::Preempted;
                    proc.preempt_count += 1;
                    self.events.push(SchedulerEvent::Preempted {
                        id: running_id,
                        tick: self.current_tick,
                    });
                    // Back to ready queue (priority insertion)
                    proc.state = ProcessState::Ready;
                    self.insert_by_priority(running_id);
                }
            }
        }

        // Pick next from ready queue
        let next_id = match self.ready_queue.pop_front() {
            Some(id) => id,
            None => return Ok(None),
        };

        let time_slice = {
            let proc = self.processes.get_mut(&next_id).unwrap();
            proc.state = ProcessState::Running;
            proc.schedule_count += 1;
            proc.wait_ticks = 0;
            self.context_switches += 1;
            self.events.push(SchedulerEvent::Scheduled {
                id: next_id,
                tick: self.current_tick,
            });
            proc.time_slice
        };

        // Execute the time slice
        let remaining_work = {
            let proc = self.processes.get(&next_id).unwrap();
            proc.total_work.saturating_sub(proc.work_done)
        };
        let work_done = time_slice.min(remaining_work);

        {
            let proc = self.processes.get_mut(&next_id).unwrap();
            proc.work_done += work_done;
            proc.cpu_time += work_done;
        }

        self.current_tick += work_done;
        self.current_running = Some(next_id);

        // Increment wait ticks for other ready processes
        let ready_ids: Vec<u64> = self.ready_queue.iter().copied().collect();
        for rid in ready_ids {
            if let Some(p) = self.processes.get_mut(&rid) {
                p.wait_ticks += work_done;
            }
        }

        // Check if process completed
        if self.processes.get(&next_id).unwrap().is_finished() {
            let proc = self.processes.get_mut(&next_id).unwrap();
            proc.state = ProcessState::Completed;
            self.current_running = None;
            self.events.push(SchedulerEvent::Completed {
                id: next_id,
                tick: self.current_tick,
            });
        }

        Ok(Some(next_id))
    }

    /// Insert a process ID into the ready queue ordered by effective priority.
    fn insert_by_priority(&mut self, id: u64) {
        let eff = self.processes.get(&id).map(|p| p.effective_priority.0).unwrap_or(u32::MAX);
        let pos = self.ready_queue.iter().position(|rid| {
            self.processes.get(rid).map(|p| p.effective_priority.0).unwrap_or(u32::MAX) > eff
        });
        match pos {
            Some(i) => self.ready_queue.insert(i, id),
            None => self.ready_queue.push_back(id),
        }
    }

    /// Apply aging: boost priority of processes that have waited too long.
    fn apply_aging(&mut self) {
        let threshold = self.aging_threshold;
        let ready_ids: Vec<u64> = self.ready_queue.iter().copied().collect();
        for id in ready_ids {
            if let Some(proc) = self.processes.get_mut(&id) {
                if proc.wait_ticks >= threshold && proc.effective_priority.0 > 0 {
                    let old = proc.effective_priority.0;
                    proc.effective_priority = Priority(old - 1);
                    proc.wait_ticks = 0;
                    self.events.push(SchedulerEvent::Aged {
                        id,
                        old_priority: old,
                        new_priority: old - 1,
                    });
                }
            }
        }
    }

    /// Block a process (e.g., waiting for I/O).
    pub fn block_process(&mut self, id: u64) -> Result<(), SchedulerError> {
        let proc = self
            .processes
            .get_mut(&id)
            .ok_or(SchedulerError::ProcessNotFound(id))?;
        let state_name = proc.state.name();
        if proc.state != ProcessState::Running && proc.state != ProcessState::Ready {
            return Err(SchedulerError::InvalidState {
                id,
                expected: "Running or Ready",
                actual: state_name,
            });
        }
        proc.state = ProcessState::Blocked;
        self.ready_queue.retain(|pid| *pid != id);
        if self.current_running == Some(id) {
            self.current_running = None;
        }
        self.blocked_set.push(id);
        self.events.push(SchedulerEvent::Blocked {
            id,
            tick: self.current_tick,
        });
        Ok(())
    }

    /// Unblock a process (I/O completed).
    pub fn unblock_process(&mut self, id: u64) -> Result<(), SchedulerError> {
        let proc = self
            .processes
            .get_mut(&id)
            .ok_or(SchedulerError::ProcessNotFound(id))?;
        if proc.state != ProcessState::Blocked {
            let state_name = proc.state.name();
            return Err(SchedulerError::InvalidState {
                id,
                expected: "Blocked",
                actual: state_name,
            });
        }
        proc.state = ProcessState::Ready;
        // Reset effective priority on unblock
        proc.effective_priority = proc.base_priority;
        self.blocked_set.retain(|pid| *pid != id);
        self.insert_by_priority(id);
        self.events.push(SchedulerEvent::Unblocked {
            id,
            tick: self.current_tick,
        });
        Ok(())
    }

    /// Run the scheduler until all processes complete or no more work.
    /// Returns IDs of completed processes in completion order.
    pub fn run_to_completion(&mut self) -> Result<Vec<u64>, SchedulerError> {
        let mut completed = Vec::new();
        loop {
            match self.schedule_one()? {
                Some(id) => {
                    if self.processes.get(&id).map(|p| p.is_finished()).unwrap_or(false) {
                        completed.push(id);
                    }
                }
                None => break,
            }
        }
        Ok(completed)
    }

    /// Get a reference to a process.
    pub fn get_process(&self, id: u64) -> Option<&Process> {
        self.processes.get(&id)
    }

    /// Stop the scheduler.
    pub fn stop(&mut self) {
        self.stopped = true;
    }

    /// Collect statistics.
    pub fn stats(&self) -> SchedulerStats {
        SchedulerStats {
            total_processes: self.processes.len(),
            ready_count: self.ready_queue.len(),
            running_count: if self.current_running.is_some() {
                1
            } else {
                0
            },
            blocked_count: self.blocked_set.len(),
            completed_count: self
                .processes
                .values()
                .filter(|p| p.state == ProcessState::Completed)
                .count(),
            total_context_switches: self.context_switches,
            total_ticks_elapsed: self.current_tick,
            total_preemptions: self
                .processes
                .values()
                .map(|p| p.preempt_count)
                .sum(),
        }
    }

    /// Drain events.
    pub fn drain_events(&mut self) -> Vec<SchedulerEvent> {
        std::mem::take(&mut self.events)
    }

    /// Current tick.
    pub fn current_tick(&self) -> u64 {
        self.current_tick
    }
}

impl Default for RoundRobinScheduler {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_process() {
        let mut sched = RoundRobinScheduler::new();
        sched
            .add_process(Process::new(1, "p1", 100))
            .unwrap();
        assert_eq!(sched.get_process(1).unwrap().state, ProcessState::Ready);
    }

    #[test]
    fn test_duplicate_process() {
        let mut sched = RoundRobinScheduler::new();
        sched.add_process(Process::new(1, "p1", 100)).unwrap();
        let err = sched.add_process(Process::new(1, "dup", 50)).unwrap_err();
        assert_eq!(err, SchedulerError::DuplicateProcess(1));
    }

    #[test]
    fn test_schedule_single_process() {
        let mut sched = RoundRobinScheduler::new();
        sched
            .add_process(Process::new(1, "p1", 20).with_time_slice(10))
            .unwrap();
        let scheduled = sched.schedule_one().unwrap();
        assert_eq!(scheduled, Some(1));
        assert_eq!(sched.get_process(1).unwrap().work_done, 10);
    }

    #[test]
    fn test_run_to_completion() {
        let mut sched = RoundRobinScheduler::new();
        sched
            .add_process(Process::new(1, "p1", 20).with_time_slice(10))
            .unwrap();
        sched
            .add_process(Process::new(2, "p2", 10).with_time_slice(10))
            .unwrap();
        let completed = sched.run_to_completion().unwrap();
        assert!(completed.contains(&1));
        assert!(completed.contains(&2));
    }

    #[test]
    fn test_round_robin_fairness() {
        let mut sched = RoundRobinScheduler::new();
        sched
            .add_process(Process::new(1, "p1", 30).with_time_slice(10))
            .unwrap();
        sched
            .add_process(Process::new(2, "p2", 30).with_time_slice(10))
            .unwrap();

        // First cycle
        sched.schedule_one().unwrap(); // p1 runs
        assert_eq!(sched.get_process(1).unwrap().work_done, 10);

        sched.schedule_one().unwrap(); // p2 runs
        assert_eq!(sched.get_process(2).unwrap().work_done, 10);

        // Second cycle
        sched.schedule_one().unwrap(); // p1 runs
        assert_eq!(sched.get_process(1).unwrap().work_done, 20);
    }

    #[test]
    fn test_preemption() {
        let mut sched = RoundRobinScheduler::new();
        sched
            .add_process(Process::new(1, "p1", 30).with_time_slice(10))
            .unwrap();
        sched
            .add_process(Process::new(2, "p2", 30).with_time_slice(10))
            .unwrap();

        sched.schedule_one().unwrap(); // p1 runs
        sched.schedule_one().unwrap(); // p1 preempted, p2 runs
        assert!(sched.get_process(1).unwrap().preempt_count >= 1);
    }

    #[test]
    fn test_block_and_unblock() {
        let mut sched = RoundRobinScheduler::new();
        sched
            .add_process(Process::new(1, "p1", 30).with_time_slice(10))
            .unwrap();
        sched
            .add_process(Process::new(2, "p2", 30).with_time_slice(10))
            .unwrap();

        sched.schedule_one().unwrap(); // p1 runs
        sched.block_process(1).unwrap();

        // Only p2 should run now
        sched.schedule_one().unwrap();
        assert_eq!(sched.get_process(2).unwrap().work_done, 10);

        // Unblock p1
        sched.unblock_process(1).unwrap();
        assert_eq!(sched.get_process(1).unwrap().state, ProcessState::Ready);
    }

    #[test]
    fn test_priority_ordering() {
        let mut sched = RoundRobinScheduler::new();
        sched
            .add_process(
                Process::new(1, "low", 20)
                    .with_priority(Priority::LOW)
                    .with_time_slice(10),
            )
            .unwrap();
        sched
            .add_process(
                Process::new(2, "high", 20)
                    .with_priority(Priority::HIGH)
                    .with_time_slice(10),
            )
            .unwrap();

        // High-priority process should be scheduled first (it was added second
        // but after the first scheduling cycle, priority ordering kicks in)
        sched.schedule_one().unwrap(); // low runs (it was added first)
        sched.schedule_one().unwrap(); // high runs
        // After preemption of low, high should be ahead
        let scheduled = sched.schedule_one().unwrap();
        assert_eq!(scheduled, Some(2)); // high runs before low
    }

    #[test]
    fn test_aging_prevents_starvation() {
        let mut sched = RoundRobinScheduler::new().with_aging_threshold(15);
        sched
            .add_process(
                Process::new(1, "high", 50)
                    .with_priority(Priority::HIGH)
                    .with_time_slice(10),
            )
            .unwrap();
        sched
            .add_process(
                Process::new(2, "low", 50)
                    .with_priority(Priority::LOW)
                    .with_time_slice(10),
            )
            .unwrap();

        // Run several cycles; the low-priority process should eventually get aged
        for _ in 0..10 {
            sched.schedule_one().unwrap();
        }
        // Low-priority process should have been scheduled at least once
        assert!(sched.get_process(2).unwrap().schedule_count > 0);
    }

    #[test]
    fn test_process_completion() {
        let mut sched = RoundRobinScheduler::new();
        sched
            .add_process(Process::new(1, "p1", 10).with_time_slice(10))
            .unwrap();
        sched.schedule_one().unwrap();
        assert_eq!(
            sched.get_process(1).unwrap().state,
            ProcessState::Completed
        );
    }

    #[test]
    fn test_process_progress() {
        let proc = Process::new(1, "p1", 100);
        assert_eq!(proc.progress(), 0.0);
        let mut proc2 = proc;
        proc2.work_done = 50;
        assert!((proc2.progress() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_zero_work_process() {
        let proc = Process::new(1, "empty", 0);
        assert!(proc.is_finished());
        assert_eq!(proc.progress(), 1.0);
    }

    #[test]
    fn test_stats() {
        let mut sched = RoundRobinScheduler::new();
        sched
            .add_process(Process::new(1, "p1", 10).with_time_slice(10))
            .unwrap();
        sched
            .add_process(Process::new(2, "p2", 20).with_time_slice(10))
            .unwrap();
        sched.run_to_completion().unwrap();
        let stats = sched.stats();
        assert_eq!(stats.total_processes, 2);
        assert_eq!(stats.completed_count, 2);
        assert!(stats.total_context_switches > 0);
    }

    #[test]
    fn test_scheduler_stop() {
        let mut sched = RoundRobinScheduler::new();
        sched.stop();
        let err = sched
            .add_process(Process::new(1, "p1", 10))
            .unwrap_err();
        assert_eq!(err, SchedulerError::SchedulerStopped);
    }

    #[test]
    fn test_schedule_empty() {
        let mut sched = RoundRobinScheduler::new();
        let result = sched.schedule_one().unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_events_emitted() {
        let mut sched = RoundRobinScheduler::new();
        sched
            .add_process(Process::new(1, "p1", 10).with_time_slice(10))
            .unwrap();
        sched.schedule_one().unwrap();
        let events = sched.drain_events();
        assert!(events.iter().any(|e| matches!(e, SchedulerEvent::Scheduled { .. })));
        assert!(events.iter().any(|e| matches!(e, SchedulerEvent::Completed { .. })));
    }

    #[test]
    fn test_block_nonexistent() {
        let mut sched = RoundRobinScheduler::new();
        let err = sched.block_process(99).unwrap_err();
        assert_eq!(err, SchedulerError::ProcessNotFound(99));
    }

    #[test]
    fn test_cpu_time_tracking() {
        let mut sched = RoundRobinScheduler::new();
        sched
            .add_process(Process::new(1, "p1", 30).with_time_slice(10))
            .unwrap();
        sched.schedule_one().unwrap();
        assert_eq!(sched.get_process(1).unwrap().cpu_time, 10);
    }
}
